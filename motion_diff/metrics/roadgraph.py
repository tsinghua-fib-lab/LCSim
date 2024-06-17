import torch
from torch_geometric.data import Batch, HeteroData

from .geometry import wrap_angle

LANE_HEADING_DIFF_K = 64

polyline_type = {
    # for lane
    "TYPE_UNDEFINED": -1,
    "TYPE_FREEWAY": 1,
    "TYPE_SURFACE_STREET": 2,
    "TYPE_BIKE_LANE": 3,
    # for roadline
    "TYPE_UNKNOWN": -1,
    "TYPE_BROKEN_SINGLE_WHITE": 6,
    "TYPE_SOLID_SINGLE_WHITE": 7,
    "TYPE_SOLID_DOUBLE_WHITE": 8,
    "TYPE_BROKEN_SINGLE_YELLOW": 9,
    "TYPE_BROKEN_DOUBLE_YELLOW": 10,
    "TYPE_SOLID_SINGLE_YELLOW": 11,
    "TYPE_SOLID_DOUBLE_YELLOW": 12,
    "TYPE_PASSING_DOUBLE_YELLOW": 13,
    # for roadedge
    "TYPE_ROAD_EDGE_BOUNDARY": 15,
    "TYPE_ROAD_EDGE_MEDIAN": 16,
    # for stopsign
    "TYPE_STOP_SIGN": 17,
    # for crosswalk
    "TYPE_CROSSWALK": 18,
    # for speed bump
    "TYPE_SPEED_BUMP": 19,
}


def compute_offroad_rate(
    data: HeteroData, traj: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """compute the offroad rate between agents of predicted trajectory

    Args:
        data (HeteroData): batch data
        traj (torch.Tensor): predicted trajectory of shape (num_agents, num_steps, 5).
            The last dimension represents [x, y, length, width, yaw].
        mask (torch.Tensor): mask of shape (num_agents, ) indicating valid agents.

    Returns:
        torch.Tensor: offroad rate of shape (1, )
    """
    assert traj.shape[-1] == 5
    assert traj.ndim == 3
    assert traj.shape[0] == data["agent"]["xyz"].shape[0]

    # only compute vehicles
    vehicle_mask = data["agent"]["type"] == 1
    mask = mask & vehicle_mask
    roadgraph_points = torch.cat(
        [
            data["roadgraph_points"]["xyz"],
            data["roadgraph_points"]["dir"],
            data["roadgraph_points"]["ids"].unsqueeze(-1),
            data["roadgraph_points"]["type"].unsqueeze(-1),
        ],
        dim=-1,
    ).contiguous()
    batch_a = (
        data["agent"]["batch"] if isinstance(data, Batch) else torch.zeros_like(mask)
    )
    batch_r = (
        data["roadgraph_points"]["batch"]
        if isinstance(data, Batch)
        else torch.zeros_like(data["roadgraph_points"]["type"])
    )
    batch_size = batch_a.max().item() + 1
    offroad_cnt = 0
    for b in range(batch_size):
        mask_a = batch_a == b
        if not torch.any(mask_a & mask):
            continue
        traj_b = traj[mask_a]
        mask_b = mask[mask_a]
        roadgraph_points_b = roadgraph_points[batch_r == b]
        offroad = compute_signed_distance_to_nearest_road_edge_point(
            traj_b[mask_b][..., :2].reshape(-1, 2), roadgraph_points_b
        )
        offroad_cnt += torch.sum(offroad > 0)
    offroad_rate = offroad_cnt / torch.sum(mask) / traj.shape[1]
    return offroad_rate


def compute_lane_heading_diff(
    data: HeteroData, traj: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """compute the lane heading difference between agents of predicted trajectory

    Args:
        data (HeteroData): batch data
        traj (torch.Tensor): predicted trajectory of shape (num_agents, num_steps, 5).
            The last dimension represents [x, y, length, width, yaw].
        mask (torch.Tensor): mask of shape (num_agents, ) indicating valid agents.

    Returns:
        torch.Tensor: lane heading difference of shape (1, )
    """
    assert traj.shape[-1] == 5
    assert traj.ndim == 3
    assert traj.shape[0] == data["agent"]["xyz"].shape[0]

    # only compute vehicles
    vehicle_mask = data["agent"]["type"] == 1
    mask = mask & vehicle_mask
    roadgraph_points = torch.cat(
        [
            data["roadgraph_points"]["xyz"],
            data["roadgraph_points"]["dir"],
            data["roadgraph_points"]["type"].unsqueeze(-1),
        ],
        dim=-1,
    ).contiguous()
    batch_a = (
        data["agent"]["batch"] if isinstance(data, Batch) else torch.zeros_like(mask)
    )
    batch_r = (
        data["roadgraph_points"]["batch"]
        if isinstance(data, Batch)
        else torch.zeros_like(data["roadgraph_points"]["type"])
    )
    batch_size = batch_a.max().item() + 1

    lane_heading_diff = 0
    for b in range(batch_size):
        mask_a = batch_a == b
        if not torch.any(mask_a & mask):
            continue
        traj_b = traj[mask_a]
        mask_b = mask[mask_a]
        traj_b = traj_b[mask_b]
        mask_r = batch_r == b
        mask_center_line = torch.logical_or(
            roadgraph_points[..., -1] == polyline_type["TYPE_FREEWAY"],
            roadgraph_points[..., -1] == polyline_type["TYPE_SURFACE_STREET"],
        )
        center_points = roadgraph_points[mask_r & mask_center_line]
        if LANE_HEADING_DIFF_K > center_points.shape[0]:
            continue
        traj_xy = traj_b[..., :2].reshape(-1, 2)
        traj_heading = traj_b[..., -1].reshape(-1)
        center_points_xy = center_points[..., :2]
        center_points_heading = torch.atan2(center_points[:, 4], center_points[:, 3])
        distances = torch.sum((traj_xy[:, None] - center_points_xy) ** 2, dim=-1)
        _, topk_indices = torch.topk(-distances, LANE_HEADING_DIFF_K, dim=-1)
        diff = torch.abs(
            wrap_angle(traj_heading[..., None] - center_points_heading[topk_indices])
        ).min(dim=-1)[0]
        lane_heading_diff += torch.sum(diff)
    lane_heading_diff = lane_heading_diff / torch.sum(mask) / traj.shape[1]
    return lane_heading_diff


def compute_signed_distance_to_nearest_road_edge_point(
    query_points: torch.Tensor,
    roadgraph_points: torch.Tensor,
) -> torch.Tensor:
    """Computes the signed distance from a set of queries to roadgraph points.

    Args:
      query_points: A set of query points for the metric of shape
        (num_query_points, 2).
      roadgraph_points: A set of roadgraph points of shape (num_points, 7). The
        last dimension represents [x, y, z, dir_x, dir_y, dir_z, id, type].
      z_stretch: Tolerance in the z dimension which determines how close to
        associate points in the roadgraph. This is used to fix problems with
        overpasses.

    Returns:
      Signed distances of the query points with the closest road edge points of
        shape (num_query_points). If the value is negative, it means that the
        actor is on the correct side of the road, if it is positive, it is
        considered `offroad`.
    """
    # Shape: (num_points).
    is_road_edge = torch.logical_or(
        roadgraph_points[..., -1] == polyline_type["TYPE_ROAD_EDGE_BOUNDARY"],
        roadgraph_points[..., -1] == polyline_type["TYPE_ROAD_EDGE_MEDIAN"],
    )
    roadgraph_points = roadgraph_points[is_road_edge]
    # Shape: (num_points, 2).
    sampled_points = roadgraph_points[..., :2]
    # Shape: (num_query_points, num_points, 2).
    differences = sampled_points - query_points[:, None]
    # Stretch difference in altitude to avoid over/underpasses.
    square_distances = torch.sum(differences**2, axis=-1)
    # Shape: (num_query_points).
    nearest_indices = torch.argmin(square_distances, axis=-1)
    prior_indices = torch.maximum(
        torch.zeros_like(nearest_indices), nearest_indices - 1
    )
    nearest_xys = sampled_points[nearest_indices, :2]
    # Direction of the road edge at the nearest points. Should be normed and
    # tangent to the road edge.
    # Shape: (num_points, 2).
    nearest_vector_xys = roadgraph_points[..., 3:6][nearest_indices, :2]
    # Direction of the road edge at the points that precede the nearest points.
    # Shape: (num_points, 2).
    prior_vector_xys = roadgraph_points[..., 3:6][prior_indices, :2]
    # Shape: (num_query_points, 2).
    points_to_edge = query_points[..., :2] - nearest_xys
    # Get the signed distance to the half-plane boundary with a cross product.
    # pad the result to 3D
    points_to_edge = torch.nn.functional.pad(points_to_edge, (0, 1))
    nearest_vector_xys = torch.nn.functional.pad(nearest_vector_xys, (0, 1))
    prior_vector_xys = torch.nn.functional.pad(prior_vector_xys, (0, 1))
    # Shape: (num_query_points). use the z component of the cross product
    cross_product = torch.cross(points_to_edge, nearest_vector_xys)[..., 2]
    cross_product_prior = torch.cross(points_to_edge, prior_vector_xys)[..., 2]
    # If the prior point is contiguous, consider both half-plane distances.
    # Shape: (num_points).
    prior_point_in_same_curve = (
        roadgraph_points[..., -2][nearest_indices]
        == roadgraph_points[..., -2][prior_indices]
    )
    offroad_sign = torch.sign(
        torch.where(
            torch.logical_and(
                prior_point_in_same_curve, cross_product_prior < cross_product
            ),
            cross_product_prior,
            cross_product,
        )
    )
    # Shape: (num_query_points).
    return torch.linalg.norm(nearest_xys - query_points[:, :2], axis=-1) * offroad_sign
