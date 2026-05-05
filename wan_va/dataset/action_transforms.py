from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def to_numpy_array(array_like) -> np.ndarray:
    if torch.is_tensor(array_like):
        array_like = array_like.detach().cpu().numpy()
    return np.asarray(array_like, dtype=np.float32)


def get_relative_pose(pose) -> np.ndarray:
    pose_np = to_numpy_array(pose)
    if pose_np.ndim != 2 or pose_np.shape[1] < 7:
        raise ValueError(
            f"Expected pose with shape [T, >=7], got {tuple(pose_np.shape)}"
        )
    if pose_np.shape[0] == 0:
        return pose_np.copy()

    rot = R.from_quat(pose_np[:, 3:7])
    first_rot = R.from_quat(np.repeat(pose_np[:1, 3:7], pose_np.shape[0], axis=0))
    relative_trans = pose_np[:, :3] - pose_np[0:1, :3]
    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat().astype(np.float32, copy=False)
    return np.concatenate([relative_trans, relative_quat], axis=1).astype(
        np.float32, copy=False
    )


def get_relative_xyz_action(action, xyz_slice: slice = slice(0, 3)) -> np.ndarray:
    action_np = to_numpy_array(action).copy()
    if action_np.ndim != 2:
        raise ValueError(
            f"Expected action with shape [T, D], got {tuple(action_np.shape)}"
        )
    if action_np.shape[0] == 0:
        return action_np

    action_np[:, xyz_slice] -= action_np[0:1, xyz_slice]
    return action_np


def map_binary_gripper_01_to_pm1(action, gripper_idx: int = 6) -> np.ndarray:
    action_np = to_numpy_array(action).copy()
    if action_np.ndim != 2:
        raise ValueError(
            f"Expected action with shape [T, D], got {tuple(action_np.shape)}"
        )
    if action_np.shape[0] == 0:
        return action_np

    gripper = action_np[:, gripper_idx]
    gripper = np.where(gripper >= 0.5, 1.0, -1.0).astype(np.float32, copy=False)
    action_np[:, gripper_idx] = gripper
    return action_np
