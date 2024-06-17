import math
from typing import Tuple

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.int64]


def generate_batch_polylines_from_map(
    polylines,
    ids,
    point_sampled_interval=1,
    num_points_each_polyline=20,
):
    """generate batch of polylines from map

    Args:
        polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, type]
        ids (num_points): polyline point ids

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


def get_polyline_dir(polyline):
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    diff = polyline - polyline_pre
    if len(diff) > 1:
        diff[0] = diff[1]
    polyline_dir = diff / np.clip(
        np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1e9
    )
    return polyline_dir


# reference: argoverse2-api(https://github.com/argoverse/av2-api)
def interp_polyline_by_fixed_waypt_interval(
    polyline: NDArrayFloat, waypt_interval: float
) -> Tuple[NDArrayFloat, int]:
    """Resample waypoints of a polyline so that waypoints appear roughly at fixed intervals from the start.

    Args:
        polyline: array pf shape (N,2) or (N,3) representing a polyline.
        waypt_interval: space interval between waypoints, in meters.

    Returns:
        interp_polyline: array of shape (N,2) or (N,3) representing a resampled/interpolated polyline.
        num_waypts: number of computed waypoints.

    Raises:
        RuntimeError: If `polyline` doesn't have shape (N,2) or (N,3).
    """
    if polyline.shape[1] not in [2, 3]:
        raise RuntimeError("Polyline must have shape (N,2) or (N,3)")

    if polyline.shape[0] < 2:
        return polyline, 1

    # get the total length in meters of the line segment
    len_m = get_polyline_length(polyline)

    # count number of waypoints to get the desired length
    # add one for the extra endpoint
    num_waypts = math.floor(len_m / waypt_interval) + 1
    interp_polyline = interp_arc(t=num_waypts, points=polyline)
    return interp_polyline, num_waypts


# reference: argoverse2-api(https://github.com/argoverse/av2-api)s
def get_polyline_length(polyline: NDArrayFloat) -> float:
    """Calculate the length of a polyline.

    Args:
        polyline: Numpy array of shape (N,3)

    Returns:
        The length of the polyline as a scalar.

    Raises:
        RuntimeError: If `polyline` doesn't have shape (N,2) or (N,3).
    """
    if polyline.shape[1] not in [2, 3]:
        raise RuntimeError("Polyline must have shape (N,2) or (N,3)")
    offsets = np.diff(polyline, axis=0)  # type: ignore
    return float(np.linalg.norm(offsets, axis=1).sum())  # type: ignore


# reference: argoverse2-api(https://github.com/argoverse/av2-api)
def interp_arc(t: int, points: NDArrayFloat) -> NDArrayFloat:
    """Linearly interpolate equally-spaced points along a polyline, either in 2d or 3d.

    Args:
        t: number of points that will be uniformly interpolated and returned
        points: Numpy array of shape (N,2) or (N,3), representing 2d or 3d-coordinates of the arc.

    Returns:
        Numpy array of shape (N,2)

    Raises:
        ValueError: If `points` is not in R^2 or R^3.
    """
    if points.ndim != 2:
        raise ValueError("Input array must be (N,2) or (N,3) in shape.")

    # the number of points on the curve itself
    n, _ = points.shape

    # equally spaced in arclength -- the number of points that will be uniformly interpolated
    eq_spaced_points = np.linspace(0, 1, t)

    # Compute the chordal arclength of each segment.
    # Compute differences between each x coord, to get the dx's
    # Do the same to get dy's. Then the hypotenuse length is computed as a norm.
    chordlen: NDArrayFloat = np.linalg.norm(np.diff(points, axis=0), axis=1)  # type: ignore
    # Normalize the arclengths to a unit total
    chordlen = chordlen / np.sum(chordlen)
    # cumulative arclength

    cumarc: NDArrayFloat = np.zeros(len(chordlen) + 1)
    cumarc[1:] = np.cumsum(chordlen)

    # which interval did each point fall in, in terms of eq_spaced_points? (bin index)
    tbins: NDArrayInt = np.digitize(eq_spaced_points, bins=cumarc).astype(int)  # type: ignore

    # #catch any problems at the ends
    tbins[np.where((tbins <= 0) | (eq_spaced_points <= 0))] = 1  # type: ignore
    tbins[np.where((tbins >= n) | (eq_spaced_points >= 1))] = n - 1

    s = np.divide((eq_spaced_points - cumarc[tbins - 1]), chordlen[tbins - 1])
    anchors = points[tbins - 1, :]
    # broadcast to scale each row of `points` by a different row of s
    offsets = (points[tbins, :] - points[tbins - 1, :]) * s.reshape(-1, 1)
    points_interp: NDArrayFloat = anchors + offsets

    return points_interp


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def check_collision(self_bbox: NDArrayFloat, other_bboxes: NDArrayFloat) -> bool:
    """check collision between self bbox and other bboxes

    Args:
        self_bbox (NDArrayFloat): (1, 5) [x, y, l, w, heading]
        other_bboxes (NDArrayFloat): (num, 5) [x, y, l, w, heading]

    Returns:
        bool: whether collision happens
    """

    def _overlap_a_over_b(first, second):
        c, s = np.cos(first[..., 4]), np.sin(first[..., 4])
        # same as rotation matrix.
        normals_t = np.stack(
            [np.stack([c, -s], axis=-1), np.stack([s, c], axis=-1)], axis=-2
        )

        # 1) Computes corners for bboxes.
        corners_a = corners_from_bboxes(first)
        corners_b = corners_from_bboxes(second)
        # 2) Project corners of first bbox to second bbox's axes.
        # Forces float32 computation for better accuracy.
        # Otherwise running on TPU will default to bfloat and does not produce
        # accurate results.
        # (..., 4, 2).
        proj_a = np.matmul(corners_a, normals_t)
        # (..., 2).
        min_a = np.min(proj_a, axis=-2)
        max_a = np.max(proj_a, axis=-2)
        proj_b = np.matmul(corners_b, normals_t)
        min_b = np.min(proj_b, axis=-2)
        max_b = np.max(proj_b, axis=-2)
        # 3) Check if the projection along axis overlaps.
        distance = np.minimum(max_a, max_b) - np.maximum(min_a, min_b)
        return np.all(distance > 0, axis=-1)

    return np.logical_and(
        _overlap_a_over_b(self_bbox, other_bboxes),
        _overlap_a_over_b(other_bboxes, self_bbox),
    )


def corners_from_bboxes(bboxes: NDArrayFloat) -> NDArrayFloat:
    assert bboxes.shape[-1] == 5  # [x, y, length, width, yaw]
    assert bboxes.ndim >= 2
    x, y, length, width, yaw = (
        bboxes[..., 0],
        bboxes[..., 1],
        bboxes[..., 2],
        bboxes[..., 3],
        bboxes[..., 4],
    )
    cos, sin = np.cos(yaw), np.sin(yaw)
    x_corners = np.stack(
        [
            x + length / 2 * cos + width / 2 * sin,
            x - length / 2 * cos + width / 2 * sin,
            x - length / 2 * cos - width / 2 * sin,
            x + length / 2 * cos - width / 2 * sin,
        ],
        axis=-1,
    )
    y_corners = np.stack(
        [
            y + length / 2 * sin - width / 2 * cos,
            y - length / 2 * sin - width / 2 * cos,
            y - length / 2 * sin + width / 2 * cos,
            y + length / 2 * sin + width / 2 * cos,
        ],
        axis=-1,
    )
    # [..., 4, 2]
    return np.stack([x_corners, y_corners], axis=-1)


def frenet_projection(
    point: NDArrayFloat, reference_line: NDArrayFloat
) -> Tuple[float, float, int]:
    """frenet projection

    Args:
        point (NDArrayFloat): (2,) [x, y]
        reference_line (NDArrayFloat): (num, 2) [x, y]

    Returns:
        Tuple[float, float]: [d, s, closest_idx]
    """
    assert reference_line.ndim == 2
    assert point.shape == (2,)
    closest_idx = np.argmin(np.linalg.norm(reference_line - point, axis=-1))
    closest_point = reference_line[closest_idx]
    # distance to reference line
    d = np.linalg.norm(point - closest_point)
    # distance along reference line
    s = np.sum(
        np.linalg.norm(reference_line[1:] - reference_line[:-1], axis=-1)[:closest_idx]
    )
    return d, s, closest_idx


def get_length(points: NDArrayFloat) -> float:
    """get length of polyline

    Args:
        points (NDArrayFloat): (num_points, 2)

    Returns:
        float: length
    """
    return np.sum(np.linalg.norm(points[1:] - points[:-1], axis=-1))
