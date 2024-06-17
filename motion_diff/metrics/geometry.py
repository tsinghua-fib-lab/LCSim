import math

import torch

EPS = 1e-10


def corners_from_bboxes(bboxes: torch.Tensor) -> torch.Tensor:
    assert bboxes.shape[-1] == 5  # [x, y, length, width, yaw]
    assert bboxes.ndim >= 2
    x, y, length, width, yaw = torch.unbind(bboxes, dim=-1)
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    x_corners = torch.stack(
        [
            x + length / 2 * cos + width / 2 * sin,
            x - length / 2 * cos + width / 2 * sin,
            x - length / 2 * cos - width / 2 * sin,
            x + length / 2 * cos - width / 2 * sin,
        ],
        dim=-1,
    )
    y_corners = torch.stack(
        [
            y + length / 2 * sin - width / 2 * cos,
            y - length / 2 * sin - width / 2 * cos,
            y - length / 2 * sin + width / 2 * cos,
            y + length / 2 * sin + width / 2 * cos,
        ],
        dim=-1,
    )
    # [..., 4, 2]
    return torch.stack([x_corners, y_corners], dim=-1)


def has_overlap(bboxes_a: torch.Tensor, bboxes_b: torch.Tensor) -> torch.Tensor:
    """Checks if 5 dof bboxes (with any prefix shape) overlap with each other.

    It does a 1:1 comparison of equivalent batch indices.

    The algorithm first computes bboxes_a's projection on bboxes_b's axes and
    check if there is an overlap between the projection. It then computes
    bboxes_b's projection on bboxes_a's axes and check overlap. Two bboxes are
    overlapped if and only if there is overlap in both steps.

    Args:
      bboxes_a: Bounding boxes of the above format of shape (..., 5). The last
        dimension represents [x, y, length, width, yaw].
      bboxes_b: Bounding boxes of the above format of shape (..., 5).

    Returns:
      Boolean array which specifies whether `bboxes_a` and `bboxes_b` overlap each
        other of shape (...).
    """
    assert bboxes_a.shape[-1] == 5
    assert bboxes_b.shape == bboxes_a.shape

    def _overlap_a_over_b(first, second):
        c, s = torch.cos(first[..., 4]), torch.sin(first[..., 4])
        # [x, y, length, width, yaw]
        normals_t = torch.stack(
            [torch.stack([c, -s], axis=-1), torch.stack([s, c], axis=-1)],
            axis=-2,
        )
        corners_a = corners_from_bboxes(first)
        corners_b = corners_from_bboxes(second)
        proj_a = torch.matmul(corners_a, normals_t)
        min_a = torch.min(proj_a, axis=-2).values
        max_a = torch.max(proj_a, axis=-2).values
        proj_b = torch.matmul(corners_b, normals_t)
        min_b = torch.min(proj_b, axis=-2).values
        max_b = torch.max(proj_b, axis=-2).values
        distance = torch.minimum(max_a, max_b) - torch.maximum(min_a, min_b)
        return torch.all(distance > 0, axis=-1)

    return torch.logical_and(
        _overlap_a_over_b(bboxes_a, bboxes_b),
        _overlap_a_over_b(bboxes_b, bboxes_a),
    )


def compute_pairwise_overlaps(traj: torch.Tensor) -> torch.Tensor:
    """Computes an overlap mask among all agent pairs for all steps.

    5 dof trajectories have [x, y, length, width, yaw] for last dimension.

    Args:
      traj: Bounding boxes of the above format of shape (num_objects, 5).

    Returns:
      Boolean array of shape (num_objects, ) which denotes whether
        any of the objects in the trajectory are in overlap.
    """
    assert traj.shape[-1] == 5
    assert traj.ndim == 2

    # (num_objects, num_objects, 5)
    traj_a = traj.unsqueeze(0).expand(traj.shape[0], -1, -1)
    traj_b = traj.unsqueeze(1).expand(-1, traj.shape[0], -1)
    self_mask = torch.eye(traj.shape[0], dtype=torch.bool, device=traj.device)
    # (num_objects, num_objects)
    return torch.where(self_mask, False, has_overlap(traj_a, traj_b)).sum(dim=-1) > 0


def wrap_yaws(yaws: torch.Tensor) -> torch.Tensor:
    """Wraps yaw angles between pi and -pi radians."""
    return (yaws + torch.pi) % (2 * torch.pi) - torch.pi


def wrap_angle(
    angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi
) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)
