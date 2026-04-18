import numpy as np
from scipy.spatial.transform import Rotation as R

from wan_va.dataset.action_transforms import (
    get_relative_pose,
    get_relative_xyz_action,
)


def test_get_relative_xyz_action_anchors_first_xyz():
    action = np.array(
        [
            [0.6, -0.1, 0.4, 0.0],
            [0.7, 0.0, 0.2, 1.0],
            [0.5, -0.3, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    relative = get_relative_xyz_action(action)

    np.testing.assert_allclose(
        relative[0, :3], np.zeros(3, dtype=np.float32), atol=1e-6
    )
    np.testing.assert_allclose(
        relative[1, :3], np.array([0.1, 0.1, -0.2]), atol=1e-6
    )
    np.testing.assert_allclose(relative[:, 3], action[:, 3], atol=1e-6)


def test_get_relative_pose_anchors_translation_and_rotation():
    pose = np.array(
        [
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            [2.0, 4.0, 6.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    relative = get_relative_pose(pose)

    np.testing.assert_allclose(
        relative[0, :3], np.zeros(3, dtype=np.float32), atol=1e-6
    )
    np.testing.assert_allclose(
        relative[1, :3], np.array([1.0, 2.0, 3.0]), atol=1e-6
    )
    assert_rot_equal(relative[0, 3:], np.array([0.0, 0.0, 0.0, 1.0]))
    assert_rot_equal(relative[1, 3:], np.array([0.0, 0.0, 1.0, 0.0]))


def assert_rot_equal(actual_quat, expected_quat):
    actual_rot = R.from_quat(np.asarray(actual_quat))
    expected_rot = R.from_quat(np.asarray(expected_quat))
    delta = expected_rot.inv() * actual_rot
    np.testing.assert_allclose(delta.as_rotvec(), np.zeros(3), atol=1e-6)
