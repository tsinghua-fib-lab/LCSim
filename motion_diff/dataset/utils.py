import math

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from torch_cluster import radius


def wrap_angle(
    angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi
) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)


class TargetBuilder(BaseTransform):
    def __init__(
        self,
        num_historical_steps: int = 11,
        num_future_steps: int = 80,
    ) -> None:
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps

    def __call__(self, data: HeteroData) -> HeteroData:
        assert (
            self.num_historical_steps + self.num_future_steps
            == data["agent"]["xyz"].shape[1]
        )
        num_nodes = data["agent"]["xyz"].shape[0]
        # transform xyz and heading to local coordinate centered at the last historical step
        origin = data["agent"]["xyz"][:, self.num_historical_steps - 1]
        theta = data["agent"]["heading"][:, self.num_historical_steps - 1].squeeze(-1)
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(num_nodes, 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        data["agent"]["target"] = origin.new_zeros(num_nodes, self.num_future_steps, 4)
        data["agent"]["target"][..., :2] = torch.bmm(
            data["agent"]["xyz"][:, self.num_historical_steps :, :2]
            - origin[:, :2].unsqueeze(1),
            rot_mat,
        )
        if data["agent"]["xyz"].size(2) == 3:
            data["agent"]["target"][..., 2] = data["agent"]["xyz"][
                :, self.num_historical_steps :, 2
            ] - origin[:, 2].unsqueeze(-1)
        data["agent"]["target"][..., 3] = wrap_angle(
            data["agent"]["heading"][:, self.num_historical_steps :].squeeze(-1)
            - theta[:, None],
        )
        return data


def generate_batch_polylines_from_map(
    polylines,
    ids,
    point_sampled_interval=1,
    num_points_each_polyline=20,
):
    """generate batch of polylines from map

    Args:
        polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, type]

    Returns:
        ret_polylines: (num_polylines, num_points_each_polyline, 6)
        ret_polylines_mask: (num_polylines, num_points_each_polyline)
        ret_center_points: (num_polylines, 7)
    """
    point_dim = polylines.shape[-1]

    sampled_points = polylines[::point_sampled_interval]
    # break points by ids
    break_idxs = np.where(ids[:-1] != ids[1:])[0] + 1
    polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
    ret_polylines = []
    ret_polylines_mask = []
    ret_center_points = []

    def append_single_polyline(new_polyline):
        cur_polyline = np.zeros(
            (num_points_each_polyline, point_dim - 1), dtype=np.float32
        )
        cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.bool8)
        cur_valid_mask[: len(new_polyline)] = 1
        ret_polylines_mask.append(cur_valid_mask)
        # center
        center_point = np.zeros((point_dim), dtype=np.float32)
        center_point[:6] = new_polyline[:, :6].mean(axis=0)
        center_point[6] = new_polyline[0, 6]
        ret_center_points.append(center_point)
        # transform points to center local coordinate
        origin = center_point[:2]
        module = np.clip(
            np.sqrt(center_point[3] ** 2 + center_point[4] ** 2), 1e-6, 1e6
        )
        cos, sin = center_point[3] / module, center_point[4] / module
        rot_mat = np.array([[cos, -sin], [sin, cos]])
        new_polyline[:, :2] -= origin
        new_polyline[:, :2] = new_polyline[:, :2].dot(rot_mat)
        new_polyline[:, 2] = new_polyline[:, 2] - center_point[2]
        new_polyline[:, 3:5] = new_polyline[:, 3:5].dot(rot_mat)
        cur_polyline[: len(new_polyline)] = new_polyline[:, :6]
        ret_polylines.append(cur_polyline)

    for k in range(len(polyline_list)):
        if polyline_list[k].__len__() <= 0:
            continue
        for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
            append_single_polyline(
                polyline_list[k][idx : idx + num_points_each_polyline]
            )

    ret_polylines = np.stack(ret_polylines, axis=0)
    ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)
    ret_center_points = np.stack(ret_center_points, axis=0)
    return ret_polylines, ret_polylines_mask, ret_center_points
