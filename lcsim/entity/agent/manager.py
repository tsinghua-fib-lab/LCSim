from heapq import merge
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..gen.agent import agent_pb2
from ..utils import plot, polygon
from .agent import Agent
from .policy import Action


class AgentManager:
    """
    Manager for all agents in a simulation.
    """

    agents: Dict[int, Agent]  # agent_id -> Agent
    ads_id: int  # id of autonomous driving system in scenario
    agents_waiting_list: List[
        Agent
    ]  # list of agents waiting for departure, sorted by departure time
    active_agents: List[Agent]  # list of active agents

    def __init__(self, pb: agent_pb2.Agents, engine):
        # pbs: agent_pb2.Agents.agents
        self.agents = {}
        for agent_pb in pb.agents:
            self.agents[agent_pb.id] = Agent(agent_pb, engine)
        self.ads_id = pb.agents[pb.ads_index].id
        # sim engine
        self.engine = engine
        # sort agents by departure time
        self.agents_waiting_list = sorted(
            self.agents.values(), key=lambda x: x.get_departure_time()
        )
        self.active_agents = []

    def get_agent_by_id(self, agent_id: int) -> Optional[Agent]:
        """
        get agent by id, return None if not found
        """
        return self.agents.get(agent_id)

    def get_ads(self) -> Optional[Agent]:
        """
        get autonomous driving system agent
        """
        return self.get_agent_by_id(self.ads_id)

    def check_departure_and_arrival(
        self, current_time: float, allow_new_agents: bool = True
    ):
        """
        Check agents departure and arrival, move agents between waiting list and active list.
        """
        # check active agents arrival
        arrival_agents_index = []
        new_waiting_list = []
        for i, a in enumerate(self.active_agents):
            if not a.runtime.status == Agent.Status.FINISHED:
                continue
            a.runtime.status = (
                Agent.Status.WAITING if a.next_trip(current_time) else Agent.Status.IDLE
            )
            arrival_agents_index.append(i)
            if a.runtime.status == Agent.Status.WAITING:
                new_waiting_list.append(a)
        # replace finished agents with new departure agents
        moved = 0
        if allow_new_agents:
            while (
                self.agents_waiting_list
                and self.agents_waiting_list[0].get_departure_time() <= current_time
            ):
                a = self.agents_waiting_list.pop(0)
                if self.engine.no_static and not a.dynamic_flag and a.id != self.ads_id:
                    continue
                a.runtime.status = Agent.Status.ACTIVE
                a.add_to_lane()
                if moved < len(arrival_agents_index):
                    self.active_agents[arrival_agents_index[moved]] = a
                else:
                    self.active_agents.append(a)
                moved += 1
        # remove finished agents
        rest_arrival = arrival_agents_index[moved:]
        for i in sorted(rest_arrival, reverse=True):
            self.active_agents.pop(i)
        # sort new waiting list and merge with waiting list
        self.agents_waiting_list = list(
            merge(
                self.agents_waiting_list,
                sorted(new_waiting_list, key=lambda x: x.get_departure_time()),
                key=lambda x: x.get_departure_time(),
            )
        )

    def prepare_obs(self):
        """
        Prepare agents' observation for update stage.
        """
        for a in self.active_agents:
            a.prepare_obs()

    def update(self, dt: float, actions: Dict[int, Action] = {}):
        """
        Update stage of simulation:
        - get action for each agent
        - update agent status
        """
        for a in self.active_agents:
            if not a.id in actions:
                a.update_runtime_by_action(a.get_action(), dt)
            else:
                a.update_runtime_by_action(actions[a.id], dt)

    def reset(self):
        """
        Reset all agents.
        """
        for a in self.agents.values():
            a.reset()
        self.agents_waiting_list = sorted(
            self.agents.values(), key=lambda x: x.get_departure_time()
        )
        self.active_agents = []

    def render(self, config: dict) -> Optional[np.ndarray]:
        """
        Render img centered at specific agent.
        """
        assert config["center_type"] == "agent"
        _id = config["center_id"]
        _id = self.ads_id if _id == "ads" else int(_id)
        center_agent = self.get_agent_by_id(_id)
        assert center_agent is not None
        if not center_agent.runtime.status == Agent.Status.ACTIVE:
            return None
        fig, ax = plot.init_fig_ax(config)
        # get surrounding roads and junctions for plot
        minx, maxx, miny, maxy = config["range"]
        center_xy = center_agent.runtime.xy
        range_xy = [
            center_xy[0] + minx,
            center_xy[0] + maxx,
            center_xy[1] + miny,
            center_xy[1] + maxy,
        ]
        cur_lane = center_agent.runtime.lane
        cur_s = center_agent.runtime.s
        heading = center_agent.runtime.heading
        # center agent
        center_bbox = np.array(
            [
                [
                    center_xy[0],
                    center_xy[1],
                    center_agent.length,
                    center_agent.width,
                    heading,
                ]
            ]
        )
        plot.plot_numpy_bounding_boxes(ax, center_bbox, plot.AGENT_COLORS["center"])
        # agent ids
        agent_ids = []
        if cur_lane is None:
            # render whole scenario
            center_junction = self.engine.junction_manager.get_unique_junction()
            agent_ids.extend(center_junction.get_agent_ids())
            center_junction.plot_polylines(ax)
        elif cur_lane.in_junction():
            agent_ids = []
            center_junction = cur_lane.parent_junction
            agent_ids.extend(center_junction.get_agent_ids())
            center_junction.plot_polylines(ax)
            # surrounding roads
            for _id in center_junction.get_surrounding_roads():
                road = self.engine.road_manager.get_road_by_id(_id)
                assert road is not None
                agent_ids.extend(road.get_agent_ids())
                road.plot_polylines(ax, range_xy)
        else:
            # cur road
            center_road = cur_lane.parent_road
            agent_ids.extend(center_road.get_agent_ids())
            center_road.plot_polylines(ax, range_xy)
            # connected roads
            for rn in center_road.pb.road_connections:
                _id = rn.road_id
                road = self.engine.road_manager.get_road_by_id(_id)
                assert road is not None
                agent_ids.extend(road.get_agent_ids())
                road.plot_polylines(ax, range_xy)
            # check if plot junction and surrounding roads
            junc_ids = set()
            road_ids = set()
            road_begin = center_xy - cur_s * np.array(
                [np.cos(heading), np.sin(heading)]
            )
            road_end = center_xy + (cur_lane.length - cur_s) * np.array(
                [np.cos(heading), np.sin(heading)]
            )
            if (
                not plot.out_of_range(road_begin, range_xy)
                and center_road.get_pre_junction_id() is not None
            ):
                junc_ids.add(center_road.get_pre_junction_id())
            if (
                not plot.out_of_range(road_end, range_xy)
                and center_road.get_succ_junction_id() is not None
            ):
                junc_ids.add(center_road.get_succ_junction_id())
            for junc_id in junc_ids:
                junction = self.engine.junction_manager.get_junction_by_id(junc_id)
                assert junction is not None
                agent_ids.extend(junction.get_agent_ids())
                junction.plot_polylines(ax)
                road_ids.update(junction.get_surrounding_roads())
            road_ids = list(road_ids - {center_road.id})
            for _id in road_ids:
                road = self.engine.road_manager.get_road_by_id(_id)
                assert road is not None
                agent_ids.extend(road.get_agent_ids())
                road.plot_polylines(ax, range_xy)
        # other agents
        agent_ids = set(agent_ids) - {center_agent.id}
        if len(agent_ids) > 0:
            agent_bbox = self.get_agent_bbox(agent_ids)
            plot.plot_numpy_bounding_boxes(
                ax,
                agent_bbox,
                plot.AGENT_COLORS["context"],
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
            agent_ids.add(center_agent.id)
            if config["plot_ref_traj"]:
                ref_trajs = self.get_agent_ref_trajs(agent_ids, 60)
                plot.plot_numpy_trajectories(
                    ax,
                    ref_trajs,
                    plot.AGENT_COLORS["trajectory"],
                    alpha=0.8,
                )

        plot.center_at_xy(ax, center_xy, config)
        return plot.img_from_fig(fig)

    def get_agent_bbox(self, ids: List[int]) -> np.ndarray:
        """
        Get bounding box for agent ids. (num_bbox, 5)
        """
        bboxes = []
        for _id in ids:
            a = self.get_agent_by_id(_id)
            if a is None:
                continue
            xy = a.runtime.xy
            heading = a.runtime.heading
            bboxes.append([xy[0], xy[1], a.length, a.width, heading])
        return np.array(bboxes)

    def get_agent_ref_trajs(self, ids: List[int], time_step: int) -> List[np.ndarray]:
        """
        Get reference trajectory for agent ids. (num_agents, num_steps, 2)
        """
        trajs = []
        for _id in ids:
            a = self.get_agent_by_id(_id)
            if (
                a is None
                or a.ref_trajectory is None
                or a.pb.attribute.type != agent_pb2.AGENT_TYPE_VEHICLE
            ):
                continue
            index = a.ref_trajectory.index
            trajs.append(a.ref_trajectory.xy[index : index + time_step])
        return trajs

    def check_collision(self, agent_id: int) -> bool:
        """
        Check collision for agent.
        """
        agent = self.get_agent_by_id(agent_id)
        if agent is None:
            return False
        if agent.runtime.status != Agent.Status.ACTIVE:
            return False
        check_ids = []
        if agent.runtime.lane is not None:
            if agent.runtime.lane.in_junction():
                check_ids.extend(agent.runtime.lane.parent_junction.get_agent_ids())
            elif agent.runtime.lane.in_road():
                check_ids.extend(agent.runtime.lane.parent_road.get_agent_ids())
        else:
            check_ids.extend([a.id for a in self.active_agents])
        check_ids = list(set(check_ids) - {agent_id})
        if len(check_ids) == 0:
            return False
        check_bbox = self.get_agent_bbox(check_ids)
        self_bbox = self.get_agent_bbox([agent_id])
        return np.sum(polygon.check_collision(self_bbox, check_bbox)) > 0

    def check_collision_all(self) -> np.ndarray:
        """
        Check collision for all agents. (num_agents, )
        """
        traj = self.get_agent_bbox([a.id for a in self.active_agents])
        if len(traj) == 0:
            return np.array([])
        # expand to (num_agents, num_agents, 5)
        traj_a = traj[:, None, :]
        traj_b = traj[None, :, :]
        # self mask (num_agents, num_agents)
        self_mask = np.eye(traj_a.shape[0], dtype=bool)
        return np.where(self_mask, False, polygon.check_collision(traj_a, traj_b)).sum(
            axis=-1
        )

    def get_collision_rate(self) -> float:
        """
        Get collision rate for all agents.
        """
        if len(self.active_agents) == 0:
            return 0.0
        return self.check_collision_all().sum() / len(self.active_agents)

    def get_offroad_rate(self) -> float:
        """
        Get offroad rate for all agents.
        """
        if len(self.active_agents) == 0:
            return 0.0
        poses = self.get_agent_bbox([a.id for a in self.active_agents])[..., :2]
        junc = self.engine.junction_manager.get_unique_junction()
        return junc.check_offroads(poses).sum() / len(poses)

    def get_agent_behavior(self) -> List[Dict[str, float]]:
        """
        Get agent behaviors. (velocity, acceleration, relative distance)
        """
        res = []
        for a in self.active_agents:
            res.append(a.get_behavior())
        return res
