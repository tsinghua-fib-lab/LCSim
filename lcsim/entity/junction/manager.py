from typing import Dict, List, Optional, Tuple

from ..gen.map import map_pb2
from ..utils import plot
from .junction import Junction


class JunctionManager:
    """
    Manager for all junctions in a map.
    """

    junctions: Dict[int, Junction]  # junction_id -> Junction

    def __init__(self, pbs: List[map_pb2.Junction], lane_manager, engine):
        # pbs: map_pb2.Map.junctions
        self.junctions = {}
        for pb in pbs:
            self.junctions[pb.id] = Junction(pb, lane_manager, engine)
        self.engine = engine

    def init_polyline_emb(self, motion_diffuser):
        for junction in self.junctions.values():
            junction.init_polyline_emb(motion_diffuser)

    def init_roadgraph_data(self):
        for junction in self.junctions.values():
            junction.init_roadgraph_data()

    def get_junction_by_id(self, junction_id: int) -> Optional[Junction]:
        return self.junctions.get(junction_id)

    def get_unique_junction(self) -> Junction:
        assert len(self.junctions) == 1
        return list(self.junctions.values())[0]

    def render(self, config: dict):
        """
        Render img centered at specific junction.
        """
        fig, ax = plot.init_fig_ax(config)
        assert config["center_type"] == "junction"
        _id = config["center_id"]
        center_junction = self.get_junction_by_id(_id)
        assert center_junction is not None
        agent_ids = []
        # center junction
        agent_ids.extend(center_junction.get_agent_ids())
        center_junction.plot_polylines(ax)
        # surrounding roads
        minx, maxx, miny, maxy = config["range"]
        center_xy = center_junction.center
        range_xy = [
            center_xy[0] + minx,
            center_xy[0] + maxx,
            center_xy[1] + miny,
            center_xy[1] + maxy,
        ]
        for _id in center_junction.get_surrounding_roads():
            road = self.engine.road_manager.get_road_by_id(_id)
            assert road is not None
            agent_ids.extend(road.get_agent_ids())
            road.plot_polylines(ax, range_xy)
        # agents
        agent_ids = list(set(agent_ids))
        if len(agent_ids) > 0:
            agent_bbox = self.engine.agent_manager.get_agent_bbox(agent_ids)
            plot.plot_numpy_bounding_boxes(
                ax,
                agent_bbox,
                plot.AGENT_COLORS["context"],
            )
            if config["plot_ref_traj"]:
                ref_trajs = self.engine.agent_manager.get_agent_ref_trajs(agent_ids, 80)
                plot.plot_numpy_trajectories(
                    ax,
                    ref_trajs,
                    plot.AGENT_COLORS["trajectory"],
                    alpha=0.5,
                )
            if config["plot_id"]:
                for i, _id in enumerate(agent_ids):
                    xy = agent_bbox[i, :2]
                    ax.text(
                        xy[0] - 2,
                        xy[1] + 2,
                        str(_id),
                        fontsize=8,
                        color="black",
                    )
        plot.center_at_xy(ax, center_xy, config)
        return plot.img_from_fig(fig)
