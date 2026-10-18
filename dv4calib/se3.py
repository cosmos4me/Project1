"""SE(3) maps of the camera pose x (6-d exponential coordinates of camera <- base, translation part first) in the float32
convention of the released EasyHeC code.

hat, _so3_exp_map, se3_exp_map, _se3_V_matrix, _get_se3_V_input: PyTorch3D (BSD, Copyright (c) Meta Platforms, Inc. and
affiliates), as copied by EasyHeC (easyhec/utils/pytorch3d_se3.py).  to_array and se3_log_map (backend 'opencv', the
released code path): EasyHeC (MIT, Copyright (c) 2023 SU Lab, UCSD).  Licence texts: third_party_licenses/."""
from typing import Tuple

import cv2
import numpy as np
import torch
from trimesh.caching import TrackedArray

SCALE = np.array([1000.0] * 3 + [10 * 180 / np.pi] * 3)          # m -> mm, rad -> 0.1 deg (error units)


def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


def _so3_exp_map(
    log_rot: torch.Tensor, eps: float = 0.0001
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        fac1[:, None, None] * skews
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R, rot_angles, skews, skews_square


def se3_exp_map(log_transform: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of SE(3) matrices `log_transform`
    to a batch of 4x4 SE(3) matrices using the exponential map.
    See e.g. [1], Sec 9.4.2. for more detailed description.

    A SE(3) matrix has the following form:
        ```
        [ R 0 ]
        [ T 1 ] ,
        ```
    where `R` is a 3x3 rotation matrix and `T` is a 3-D translation vector.
    SE(3) matrices are commonly used to represent rigid motions or camera extrinsics.

    In the SE(3) logarithmic representation SE(3) matrices are
    represented as 6-dimensional vectors `[log_translation | log_rotation]`,
    i.e. a concatenation of two 3D vectors `log_translation` and `log_rotation`.

    The conversion from the 6D representation to a 4x4 SE(3) matrix `transform`
    is done as follows:
        ```
        transform = exp( [ hat(log_rotation) 0 ]
                         [   log_translation 1 ] ) ,
        ```
    where `exp` is the matrix exponential and `hat` is the Hat operator [2].

    Note that for any `log_transform` with `0 <= ||log_rotation|| < 2pi`
    (i.e. the rotation angle is between 0 and 2pi), the following identity holds:
    ```
    se3_log_map(se3_exponential_map(log_transform)) == log_transform
    ```

    The conversion has a singularity around `||log(transform)|| = 0`
    which is handled by clamping controlled with the `eps` argument.

    Args:
        log_transform: Batch of vectors of shape `(minibatch, 6)`.
        eps: A threshold for clipping the squared norm of the rotation logarithm
            to avoid unstable gradients in the singular case.

    Returns:
        Batch of transformation matrices of shape `(minibatch, 4, 4)`.

    Raises:
        ValueError if `log_transform` is of incorrect shape.

    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    [2] https://en.wikipedia.org/wiki/Hat_operator
    """

    if log_transform.ndim != 2 or log_transform.shape[1] != 6:
        raise ValueError("Expected input to be of shape (N, 6).")

    N, _ = log_transform.shape

    log_translation = log_transform[..., :3]
    log_rotation = log_transform[..., 3:]

    # rotation is an exponential map of log_rotation
    (
        R,
        rotation_angles,
        log_rotation_hat,
        log_rotation_hat_square,
    ) = _so3_exp_map(log_rotation, eps=eps)

    # translation is V @ T
    V = _se3_V_matrix(
        log_rotation,
        log_rotation_hat,
        log_rotation_hat_square,
        rotation_angles,
        eps=eps,
    )
    T = torch.bmm(V, log_translation[:, :, None])[:, :, 0]

    transform = torch.zeros(
        N, 4, 4, dtype=log_transform.dtype, device=log_transform.device
    )

    transform[:, :3, :3] = R
    transform[:, :3, 3] = T
    transform[:, 3, 3] = 1.0

    return transform.permute(0, 2, 1)


def _se3_V_matrix(
    log_rotation: torch.Tensor,
    log_rotation_hat: torch.Tensor,
    log_rotation_hat_square: torch.Tensor,
    rotation_angles: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    A helper function that computes the "V" matrix from [1], Sec 9.4.2.
    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    """

    V = (
        torch.eye(3, dtype=log_rotation.dtype, device=log_rotation.device)[None]
        + log_rotation_hat
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        * ((1 - torch.cos(rotation_angles)) / (rotation_angles**2))[:, None, None]
        + (
            log_rotation_hat_square
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            * ((rotation_angles - torch.sin(rotation_angles)) / (rotation_angles**3))[
                :, None, None
            ]
        )
    )

    return V


def _get_se3_V_input(log_rotation: torch.Tensor, eps: float = 1e-4):
    """
    A helper function that computes the input variables to the `_se3_V_matrix`
    function.
    """
    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    nrms = (log_rotation**2).sum(-1)
    rotation_angles = torch.clamp(nrms, eps).sqrt()
    log_rotation_hat = hat(log_rotation)
    log_rotation_hat_square = torch.bmm(log_rotation_hat, log_rotation_hat)
    return log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles


def to_array(x, dtype=float):
    if isinstance(x, np.ndarray):
        return x.astype(dtype)
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype)
    elif isinstance(x, list):
        return [to_array(a) for a in x]
    elif isinstance(x, dict):
        return {k: to_array(v) for k, v in x.items()}
    elif isinstance(x, TrackedArray):
        return np.array(x)
    else:
        return x


def se3_log_map(transform: torch.Tensor, eps: float = 1e-4, backend="opencv", test_acc=True):
    """EasyHeC's se3_log_map with backend 'opencv' (the only one the released code path uses)."""
    if backend != "opencv":
        raise NotImplementedError(backend)
    log_rotation = []
    for tsfm in transform:
        cv2_rot = -cv2.Rodrigues(to_array(tsfm[:3, :3]))[0]
        log_rotation.append(torch.from_numpy(cv2_rot.reshape(-1)).to(transform.device).float())
    log_rotation = torch.stack(log_rotation, dim=0)
    T = transform[:, 3, :3]
    V = _se3_V_matrix(*_get_se3_V_input(log_rotation), eps=eps)
    log_translation = torch.linalg.solve(V, T[:, :, None])[:, :, 0]
    dof6 = torch.cat((log_translation, log_rotation), dim=1)
    if test_acc:
        err = (se3_exp_map(dof6) - transform).abs().max()
        if err > 0.1:
            raise RuntimeError()
    return dof6


def dof_of(T: np.ndarray) -> np.ndarray:
    return se3_log_map(torch.as_tensor(np.asarray(T), dtype=torch.float32)[None].permute(0, 2, 1),
                       backend="opencv")[0].double().numpy()          # released code path is float32


def pose_of(dof: np.ndarray) -> np.ndarray:
    return se3_exp_map(torch.as_tensor(np.asarray(dof), dtype=torch.float32)[None]).permute(0, 2, 1)[0].double().numpy()


def pose64(x):
    return se3_exp_map(x.reshape(1, 6).to(torch.float64)).permute(0, 2, 1)[0]
