from typing import List, Optional

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

PX_PER_METER = 5.0

ROAD_GRAPH_COLORS = {
    # Consistent with MapElementIds
    1: np.array([230, 230, 230]) / 255.0,  # 'LaneCenter-Freeway',
    2: np.array([230, 230, 230]) / 255.0,  # 'LaneCenter-SurfaceStreet',
    3: np.array([230, 230, 230]) / 255.0,  # 'LaneCenter-BikeLane',
    6: np.array([169, 169, 169]) / 255.0,  # 'RoadLine-BrokenSingleWhite',
    7: np.array([169, 169, 169]) / 255.0,  # 'RoadLine-SolidSingleWhite',
    8: np.array([169, 169, 169]) / 255.0,  # 'RoadLine-SolidDoubleWhite',
    9: np.array([255, 165, 0]) / 255.0,  # 'RoadLine-BrokenSingleYellow',
    10: np.array([255, 165, 0]) / 255.0,  # 'RoadLine-BrokenDoubleYellow'
    11: np.array([255, 165, 0]) / 255.0,  # 'RoadLine-SolidSingleYellow',
    12: np.array([255, 165, 0]) / 255.0,  # 'RoadLine-SolidDoubleYellow',
    13: np.array([120, 120, 120]) / 255.0,  # 'RoadLine-PassingDoubleYellow',
    15: np.array([80, 80, 80]) / 255.0,  # 'RoadEdgeBoundary',
    16: np.array([80, 80, 80]) / 255.0,  # 'RoadEdgeMedian',
    17: np.array([255, 0, 0]) / 255.0,  # 'StopSign',  # One point
    18: np.array([200, 200, 200]) / 255.0,  # 'Crosswalk',  # Polygon
    19: np.array([200, 200, 200]) / 255.0,  # 'SpeedBump',  # Polygon
    20: np.array([255, 0, 0]) / 255.0,  # 'Exit',  # One point
}

BROKEN_LINE_TYPE = {6, 9, 10}

AGENT_COLORS = {
    "context": np.array([0.4, 0.4, 0.4]),  # Context agents, grey.
    "center": np.array([0, 0.6, 0.8]),  # Modeled agents, dark blue.
    "trajectory": np.array([0, 0.6, 0.8]),
}


def get_color_from_type(type_id: int) -> np.ndarray:
    """Returns color for a given type."""
    if type_id in ROAD_GRAPH_COLORS:
        return ROAD_GRAPH_COLORS[type_id]
    return np.array([0, 0, 0])


def init_fig_ax(config: dict) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Initializes a figure with given size in pixel."""
    fig, ax = plt.subplots()
    # Sets output image to pixel resolution.
    dpi = 100
    fig_size = config["fig_size"]
    fig.set_size_inches(fig_size[0], fig_size[1])
    fig.set_dpi(dpi)
    fig.set_facecolor("white")
    return fig, ax


def img_from_fig(fig: matplotlib.figure.Figure) -> np.ndarray:
    """Returns a [H, W, 3] uint8 np image from fig.canvas.tostring_rgb()."""
    # Just enough margin in the figure to display xticks and yticks.
    fig.subplots_adjust(
        left=0.08, bottom=0.06, right=0.98, top=0.96, wspace=0.0, hspace=0.0
    )
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


def center_at_xy(ax: matplotlib.axes.Axes, xy: np.ndarray, config: dict) -> None:
    minx, maxx, miny, maxy = config["range"]
    ax.axis(
        (
            xy[0] + minx,
            xy[0] + maxx,
            xy[1] + miny,
            xy[1] + maxy,
        )
    )
    axises = config["axises"]
    if not axises:
        ax.set_axis_off()
    ax.set_aspect("equal")


def plot_numpy_bounding_boxes(
    ax: matplotlib.axes.Axes,
    bboxes: np.ndarray,
    color: np.ndarray,
    alpha: Optional[float] = 1.0,
) -> None:
    """Plots multiple bounding boxes.

    Args:
      ax: Fig handles.
      bboxes: Shape (num_bbox, 5), with last dimension as (x, y, length, width,
        yaw).
      color: Shape (3,), represents RGB color for drawing.
      alpha: Alpha value for drawing, i.e. 0 means fully transparent.
    """
    if bboxes.ndim != 2 or bboxes.shape[1] != 5 or color.shape != (3,):
        raise ValueError(
            (
                "Expect bboxes rank 2, last dimension of bbox 5, color of size 3,"
                " got{}, {}, {} respectively"
            ).format(bboxes.ndim, bboxes.shape[1], color.shape)
        )

    c = np.cos(bboxes[:, 4])
    s = np.sin(bboxes[:, 4])
    pt = np.array((bboxes[:, 0], bboxes[:, 1]))  # (2, N)
    length, width = bboxes[:, 2], bboxes[:, 3]
    u = np.array((c, s))
    ut = np.array((s, -c))

    # Compute box corner coordinates.
    tl = pt + length / 2 * u - width / 2 * ut
    tr = pt + length / 2 * u + width / 2 * ut
    br = pt - length / 2 * u + width / 2 * ut
    bl = pt - length / 2 * u - width / 2 * ut

    # Compute heading arrow using center left/right/front.
    cl = pt - width / 2 * ut
    cr = pt + width / 2 * ut
    cf = pt + length / 2 * u

    # Draw bboxes.
    ax.plot(
        [tl[0, :], tr[0, :], br[0, :], bl[0, :], tl[0, :]],
        [tl[1, :], tr[1, :], br[1, :], bl[1, :], tl[1, :]],
        color=color,
        zorder=4,
        alpha=alpha,
    )

    # Draw heading arrow.
    ax.plot(
        [cl[0, :], cr[0, :], cf[0, :], cl[0, :]],
        [cl[1, :], cr[1, :], cf[1, :], cl[1, :]],
        color=color,
        zorder=4,
        alpha=alpha,
    )


def plot_numpy_trajectories(
    ax: matplotlib.axes.Axes,
    trajectories: List[np.ndarray],
    color: np.ndarray,
    alpha: Optional[float] = 1.0,
) -> None:
    """Plots multiple trajectories.

    Args:
      ax: Fig handles.
      trajectories: List of shape (num_steps, 2), represents points to draw.
      color: Shape (3,), represents RGB color for drawing.
      alpha: Alpha value for drawing, i.e. 0 means fully transparent.
    """
    colors = np.linspace(0, 1, trajectories[0].shape[0])
    for traj in trajectories:
        if traj.ndim != 2 or traj.shape[1] != 2 or traj.shape[0] != colors.shape[0]:
            continue
        ax.scatter(
            traj[:, 0],
            traj[:, 1],
            c=colors,
            cmap="viridis",
            s=15,
            alpha=alpha,
            edgecolor="none",
        )


def plot_dashed_line(
    ax: matplotlib.axes.Axes,
    polyline: np.ndarray,
    dashes: tuple[int, int],
    color: np.ndarray,
    alpha: Optional[float] = 1.0,
    range_xy: Optional[list[int]] = None,
) -> None:
    """Plots a dashed line along the given polyline.

    Args:
      ax: Fig handles.
      polyline: Shape (N, 2), represents points to draw.
      dashes: Tuple of two integers, represents the length of dashes and gaps.
      color: Shape (3,), represents RGB color for drawing.
      alpha: Alpha value for drawing, i.e. 0 means fully transparent.
    """
    if polyline.ndim != 2 or polyline.shape[1] != 2 or color.shape != (3,):
        raise ValueError(
            (
                "Expect polyline rank 2, last dimension of polyline 2, color of size 3,"
                " got{}, {}, {} respectively"
            ).format(polyline.ndim, polyline.shape[1], color.shape)
        )
    dash_length, gap_length = dashes
    p1 = polyline[0]
    for i in range(1, len(polyline)):
        p2 = polyline[i]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        d = np.sqrt(dx**2 + dy**2)
        if d == 0:
            continue
        n = int(d / (dash_length + gap_length))
        if n == 0:
            continue
        dx /= d
        dy /= d
        for j in range(n):
            x1 = p1[0] + j * (dash_length + gap_length) * dx
            y1 = p1[1] + j * (dash_length + gap_length) * dy
            x2 = x1 + dash_length * dx
            y2 = y1 + dash_length * dy
            if out_of_range([x1, y1], range_xy) and out_of_range([x2, y2], range_xy):
                continue
            ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha)
        p1 = p2


def out_of_range(xy: np.ndarray, range_xy: list[int]) -> bool:
    """Returns True if xy is out of range."""
    if range_xy is None:
        return False
    return (
        xy[0] < range_xy[0]
        or xy[0] > range_xy[1]
        or xy[1] < range_xy[2]
        or xy[1] > range_xy[3]
    )
