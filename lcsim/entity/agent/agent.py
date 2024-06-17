import math
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..gen.agent import agent_pb2
from ..lane import lane
from ..utils import polygon
from .policy import (
    Action,
    BasePolicy,
    BicycleExpert,
    LaneIDM,
    Observation,
    StateExpert,
    Trajectory,
    TrajIDM,
)
from .route import Route
from .schedule import Schedule


class Agent:
    """
    Agent
    """

    class Status(Enum):
        WAITING = 0
        ACTIVE = 1
        FINISHED = 2
        IDLE = 3

    class Runtime:
        status: "Agent.Status"
        lane: "lane.Lane" = None  # current lane
        s: float = 0.0  # current s on lane
        v: float = 0.0  # current speed
        xy: np.ndarray = None  # current position (x,y)
        heading: float = 0.0  # current heading [-pi, pi]

        def __str__(self) -> str:
            return f"status: {self.status}, lane: {self.lane.id}, s: {self.s}, v: {self.v}, xy: {self.xy}, heading: {self.heading}"

        def is_finished(self) -> bool:
            return (
                self.status == Agent.Status.IDLE or self.status == Agent.Status.FINISHED
            )

    class HistoryState:
        xyz: np.ndarray  # (num_steps, 3)
        vel: np.ndarray  # (num_steps, 2)
        heading: np.ndarray  # (num_steps, 1)
        shape: np.ndarray  # (num_steps, 3)
        type: np.ndarray  # (1,)
        valid: np.ndarray  # (num_steps, 1)

        def __init__(self) -> None:
            self.xyz = np.zeros((11, 3))
            self.vel = np.zeros((11, 2))
            self.heading = np.zeros((11, 1))
            self.shape = np.zeros((11, 3))
            self.type = np.ones(1, dtype=np.int32)
            self.valid = np.zeros((11, 1), dtype=np.bool8)

        def update(
            self,
            xyz: np.ndarray,
            vel: np.ndarray,
            heading: np.ndarray,
            shape: np.ndarray,
        ):
            # move forward and add new state at the end
            self.xyz[:-1] = self.xyz[1:]
            self.vel[:-1] = self.vel[1:]
            self.heading[:-1] = self.heading[1:]
            self.shape[:-1] = self.shape[1:]
            self.valid[:-1] = self.valid[1:]
            self.xyz[-1] = xyz
            self.vel[-1] = vel
            self.heading[-1] = heading
            self.shape[-1] = shape
            self.valid[-1] = True

    MIN_VIEW_DISTANCE = 50.0  # minimum view distance
    VIEW_DISTANCE_FACTOR = 12.0  # look ahead next 12s
    CLOSE_TO_END_THRESHOLD = 5.0  # close to end threshold
    NUM_HISTORICAL_STEPS = 11  # number of historical steps
    ROUTE_LENGTH = 10  # length of route
    DYNAMIC_THRESHOLD = 1.0  # dynamic threshold
    # static features
    pb: agent_pb2.Agent
    # sim features
    schedule: Schedule  # schedule of the agent
    route: Route  # route of current trip
    obs: Observation  # observation of the agent
    runtime: Runtime  # current runtime
    history_states: HistoryState  # history states
    dynamic_flag: bool = True  # dynamic flag
    # policy
    policy: BasePolicy
    # generated features
    ref_trajectory: Optional[Trajectory] = None  # reference trajectory (num_steps,)

    def __init__(self, pb: agent_pb2.Agent, engine):
        # pb: agent_pb2.Agent
        self.pb = pb
        self.schedule = Schedule(pb.schedules)
        self.route = Route(
            self.schedule.get_trip().routes,
            pb.home,
            self.schedule.get_trip().end,
            engine,
        )
        # sim engine
        self.engine = engine
        # home
        home = self.pb.home
        if home.lane_position.ByteSize() > 0:
            l: lane.Lane = self.engine.lane_manager.get_lane_by_id(
                home.lane_position.lane_id
            )
            s = home.lane_position.s
            # init runtime
            self.runtime = self.Runtime()
            self.runtime.status = self.Status.WAITING
            self.runtime.lane = l
            self.runtime.s = s
            self.runtime.v = 0.0
            self.runtime.xy = l.get_position_by_s(s)
            self.runtime.heading = l.get_direction_by_s(s)
        elif home.xy_position.ByteSize() > 0:
            self.engine.junction_manager.get_unique_junction().add_agent_id(self.id)
            # init runtime
            xy = np.array([home.xy_position.x, home.xy_position.y])
            self.runtime = self.Runtime()
            self.runtime.status = self.Status.WAITING
            self.runtime.lane = None
            self.runtime.s = 0.0
            trip = self.schedule.get_trip()
            assert len(trip.agent_states) > 0
            init_state = trip.agent_states[0]
            self.runtime.v = np.sqrt(
                init_state.velocity_x**2 + init_state.velocity_y**2
            )
            self.runtime.xy = xy
            self.runtime.heading = init_state.heading
            xy = np.array([[s.position.x, s.position.y] for s in trip.agent_states])
            vel = np.array([[s.velocity_x, s.velocity_y] for s in trip.agent_states])
            heading = np.array([s.heading for s in trip.agent_states])
            self.set_ref_trajectory(xy, vel, heading)
            self.dynamic_flag = (
                np.linalg.norm(np.abs(np.max(xy, axis=0) - np.min(xy, axis=0)))
                > self.DYNAMIC_THRESHOLD
            ) and pb.attribute.type == agent_pb2.AGENT_TYPE_VEHICLE
        else:
            raise ValueError("home position not set")
        self.history_states = self.HistoryState()
        xy, v, heading = self.runtime.xy, self.runtime.v, self.runtime.heading
        xyz = np.array([xy[0], xy[1], 0.0])
        vel = np.array([v * np.cos(heading), v * np.sin(heading)])
        heading = np.array([heading])
        shape = np.array([self.length, self.width, 0.0])
        self.history_states.update(xyz, vel, heading, shape)
        # init policy
        self.policy = (
            LaneIDM(self)
            if self.ref_trajectory is None
            else TrajIDM(self) if engine.reactive else StateExpert(self)
        )

    @property
    def id(self) -> int:
        return self.pb.id

    @property
    def length(self) -> float:
        return self.pb.attribute.length

    @property
    def width(self) -> float:
        return self.pb.attribute.width

    @property
    def heading(self) -> float:
        return self.runtime.heading

    @property
    def v(self) -> float:
        return self.runtime.v

    def prepare_obs(self):
        """
        Prepare agent's observation for policy.
        """
        obs = Observation()
        if isinstance(self.policy, LaneIDM):
            # get observation for lane idm policy
            obs.cur_lane = self.runtime.lane
            obs.s = self.runtime.s
            # get ahead lanes
            obs.ahead_lanes = []
            view_dis = max(
                self.MIN_VIEW_DISTANCE, self.runtime.v * self.VIEW_DISTANCE_FACTOR
            )
            look_dis = obs.cur_lane.length - obs.s
            cur_lane = obs.cur_lane
            junc_index = 0
            while look_dis < view_dis:
                if cur_lane.in_junction():
                    cur_lane = cur_lane.unique_successor()
                    junc_index += 1
                else:
                    cur_lane = self.route.get_junc_lane_by_pre_lane(
                        cur_lane, junc_index
                    )
                if cur_lane is None:
                    break
                obs.ahead_lanes.append((look_dis, cur_lane))
                look_dis += cur_lane.length
            # get ahead agents
            obs.ahead_vehicle = None
            ahead = obs.cur_lane.get_next_agent_by_s(obs.s)
            if ahead is not None:
                ahead_a: Agent = self.engine.agent_manager.get_agent_by_id(ahead[1])
                obs.ahead_vehicle = (
                    ahead[0] - (self.length + ahead_a.length) / 2.0,
                    ahead_a.runtime.v,
                )
            else:
                # check agents in ahead lanes
                for al in obs.ahead_lanes:
                    ahead = al[1].get_first_agent()
                    if ahead is not None:
                        ahead_a: Agent = self.engine.agent_manager.get_agent_by_id(
                            ahead[1]
                        )
                        obs.ahead_vehicle = (
                            al[0] + ahead[0] - (self.length + ahead_a.length) / 2.0,
                            ahead_a.runtime.v,
                        )
                        break
        elif isinstance(self.policy, BicycleExpert) or isinstance(
            self.policy, StateExpert
        ):
            obs.ref_trajectory = self.ref_trajectory
            obs.cur_runtime = (self.runtime.xy, self.runtime.v, self.runtime.heading)
        elif isinstance(self.policy, TrajIDM):
            obs.ref_trajectory = self.ref_trajectory
            obs.cur_runtime = (self.runtime.xy, self.runtime.v, self.runtime.heading)
            # get closest ahead agent
            obs.ahead_vehicle = None
            ahead_dis, ahead_v = math.inf, 0.0
            agent_list: List[Agent] = self.engine.agent_manager.active_agents
            pos, heading = self.runtime.xy, self.runtime.heading
            rot_mat = np.array(
                [
                    [np.cos(heading), -np.sin(heading)],
                    [np.sin(heading), np.cos(heading)],
                ]
            )
            for a in agent_list:
                if a.id == self.id:
                    continue
                # project agent position to current coordinate system
                xy = a.runtime.xy - pos
                xy = np.dot(xy, rot_mat)
                # check if in width range
                length_a, width_a = a.length, a.width
                rel_heading = polygon.wrap_angle(a.runtime.heading - heading)
                cos, sin = np.cos(rel_heading), np.sin(rel_heading)
                y_corners = [
                    xy[1] + length_a / 2.0 * sin - width_a / 2.0 * cos,
                    xy[1] + length_a / 2.0 * sin + width_a / 2.0 * cos,
                    xy[1] - length_a / 2.0 * sin + width_a / 2.0 * cos,
                    xy[1] - length_a / 2.0 * sin - width_a / 2.0 * cos,
                ]
                if (
                    max(y_corners) < -self.width / 2.0
                    or min(y_corners) > self.width / 2.0
                ):
                    continue
                # check ahead
                x_corners = [
                    xy[0] + length_a / 2.0 * cos + width_a / 2.0 * sin,
                    xy[0] + length_a / 2.0 * cos - width_a / 2.0 * sin,
                    xy[0] - length_a / 2.0 * cos - width_a / 2.0 * sin,
                    xy[0] - length_a / 2.0 * cos + width_a / 2.0 * sin,
                ]
                if min(x_corners) < self.length / 2.0:
                    continue
                # check distance
                dis = min(x_corners) - self.length / 2.0
                if dis < ahead_dis:
                    ahead_dis = dis
                    ahead_v = a.runtime.v * np.cos(rel_heading)
            if ahead_dis != math.inf:
                obs.ahead_vehicle = (ahead_dis, ahead_v)
        else:
            raise ValueError(f"unknown policy: {self.policy}")
        self.obs = obs

    def get_action(self) -> Action:
        """
        Get action for agent update.
        """
        return self.policy.get_action(self.obs)

    def update_runtime_by_action(self, action: Action, dt: float):
        """
        Update agent runtime by action.
        """
        if action.type == Action.ActionType.LANE:
            v, ds = self._compute_v_and_dis(self.v, action.data.acc, dt)
            self.runtime.v = v
            s = self.runtime.s + ds
            cur_lane = self.runtime.lane
            if s > self.runtime.lane.length:
                # move to next lane
                while s > cur_lane.length:
                    s -= cur_lane.length
                    next_lane = self.route.next(cur_lane, self.runtime.s, self.v)
                    if next_lane is None:
                        break
                    cur_lane = next_lane
            self.runtime.lane = cur_lane
            self.runtime.s = s
            if isinstance(self.policy, LaneIDM) and self.policy.lc_runtime.is_lc:
                # lane changing, check if completed
                target_lane = self.policy.lc_runtime.target_lane
                if (
                    self.policy.lc_runtime.completed_length + ds
                    >= self.policy.lc_runtime.total_length
                ):
                    self.runtime.s = target_lane.project_from_lane(cur_lane, s)
                    self.runtime.lane = target_lane
                    self.policy.clear_lc()
                else:
                    self.policy.lc_runtime.completed_length += ds
                    self.policy.lc_runtime.target_lane_s = (
                        target_lane.project_from_lane(cur_lane, s)
                    )
                    target_lane.add_agent(self.id, self.policy.lc_runtime.target_lane_s)
            xy = cur_lane.get_position_by_s(s)
            heading = cur_lane.get_direction_by_s(s)
            if isinstance(self.policy, LaneIDM) and self.policy.lc_runtime.is_lc:
                # lane changing, update position and heading by completed ratio
                target_lane = self.policy.lc_runtime.target_lane
                xy_lc = self.policy.lc_runtime.target_lane.get_position_by_s(
                    self.policy.lc_runtime.target_lane_s
                )
                ratio = (
                    self.policy.lc_runtime.completed_length
                    / self.policy.lc_runtime.total_length
                )
                xy = xy * (1 - ratio) + xy_lc * ratio
                heading_bias = np.arctan2(
                    (cur_lane.width + target_lane.width) / 2.0,
                    self.policy.lc_runtime.total_length,
                ) * (1 - abs(ratio * 2 - 1))
                if target_lane == cur_lane.left_neighbor():
                    heading += heading_bias
                elif target_lane == cur_lane.right_neighbor():
                    heading -= heading_bias
                else:
                    raise ValueError(
                        f"lane changing error: {cur_lane.id} -> {target_lane.id}"
                    )
                heading = np.arctan2(np.sin(heading), np.cos(heading))
            self.runtime.xy = xy
            self.runtime.heading = heading
            if self.check_close_to_end():
                self.runtime.status = self.Status.FINISHED
                return
            self.add_to_lane()
        elif action.type == Action.ActionType.STATE:
            new_state = action.data
            xy = np.array([new_state.x, new_state.y])
            v = np.sqrt(new_state.vx**2 + new_state.vy**2)
            heading = new_state.heading
            self.runtime.xy = xy
            self.runtime.v = v
            self.runtime.heading = heading
            self.ref_trajectory.next(xy)
            if self.ref_trajectory.reach_end():
                # reach end of trajectory
                self.runtime.status = self.Status.FINISHED
        elif action.type == Action.ActionType.BICYCLE:
            acc, steer = action.data.acc, action.data.steer
            xy, v, heading = self.runtime.xy, self.runtime.v, self.runtime.heading
            delta = v * dt + 0.5 * acc * dt**2
            delta_heading = steer * (v * dt + 0.5 * acc * dt**2)
            new_heading = polygon.wrap_angle(heading + delta_heading)
            new_xy = xy + np.array(
                [delta * np.cos(new_heading), delta * np.sin(new_heading)]
            )
            new_v = v + acc * dt
            self.runtime.xy = new_xy
            self.runtime.v = new_v
            self.runtime.heading = new_heading
            self.ref_trajectory.next(xy)
            if self.ref_trajectory.reach_end():
                # reach end of trajectory
                self.runtime.status = self.Status.FINISHED
        elif action.type == Action.ActionType.DELTA:
            dx, dy, dheading = action.data.dx, action.data.dy, action.data.dheading
            xy, v, heading = self.runtime.xy, self.runtime.v, self.runtime.heading
            new_xy = xy + np.array([dx, dy])
            new_heading = polygon.wrap_angle(heading + dheading)
            self.runtime.xy = new_xy
            self.runtime.heading = new_heading
            self.ref_trajectory.next(xy)
            if self.ref_trajectory.reach_end():
                # reach end of trajectory
                self.runtime.status = self.Status.FINISHED
        else:
            raise ValueError(f"unknown action type: {action.type}")
        xyz = np.array([self.runtime.xy[0], self.runtime.xy[1], 0.0])
        heading = self.runtime.heading
        vel = np.array(
            [self.runtime.v * np.cos(heading), self.runtime.v * np.sin(heading)]
        )
        heading = np.array([heading])
        shape = np.array([self.length, self.width, 0.0])
        self.history_states.update(xyz, vel, heading, shape)

    def reset(self):
        """
        Reset agent status.
        """
        self.__init__(self.pb, self.engine)

    def check_close_to_end(self) -> bool:
        if (
            self.runtime.lane.parent_road == self.route.end_lane.parent_road
            and self.route.end_s - self.runtime.s < self.CLOSE_TO_END_THRESHOLD
        ):
            return True
        return False

    def get_departure_time(self) -> float:
        """
        Get departure time of an agent, return inf if no more trips.
        """
        return self.schedule.get_departure_time()

    def next_trip(self, time: float) -> bool:
        """
        Move to next trip in schedule, return False if no more trips.
        """
        return self.schedule.next_trip(time)

    def add_to_lane(self):
        """
        Add agent to current lane.
        """
        if self.runtime.lane is None:
            return
        self.runtime.lane.add_agent(self.id, self.runtime.s)

    def set_ref_trajectory(self, xy: np.ndarray, vel: np.ndarray, heading: np.ndarray):
        """
        Set reference trajectory for agent.
        """
        self.ref_trajectory = Trajectory(xy, vel, heading)

    def get_route(self) -> np.ndarray:
        """
        Get future route of the agent.
        """
        assert self.ref_trajectory is not None
        route = np.zeros((self.ROUTE_LENGTH, 2))
        # motion_vector = self.ref_trajectory.get_motion_vector()
        motion_vector = self.ref_trajectory.get_route(
            self.runtime.xy, self.runtime.heading
        )
        length = min(motion_vector.shape[0], self.ROUTE_LENGTH)
        route[:length] = motion_vector[:length]
        return route

    def is_offroad(self) -> bool:
        """
        Check if agent is off road.
        """
        if self.runtime.lane is None:
            junc = self.engine.junction_manager.get_unique_junction()
            return junc.check_offroad(self.runtime.xy)
        l = self.runtime.lane
        if l.in_junction():
            return l.parent_junction.check_offroad(self.runtime.xy)
        elif l.in_road():
            return l.parent_road.check_offroad(self.runtime.xy)
        return False

    def _compute_v_and_dis(self, v: float, a: float, dt: float) -> Tuple[float, float]:
        dv = a * dt
        if v + dv < 0:
            # brake to stop
            return 0, -v * v / (2 * a)
        return v + dv, (v + 0.5 * dv) * dt

    def get_behavior(self) -> Dict[str, Union[str, float]]:
        """
        Get agent behavior.
        """
        cur_v = self.runtime.v
        last_v = np.linalg.norm(self.history_states.vel[-2])
        acc = (cur_v - last_v) / 0.1
        rel_dis = -1 if self.obs.ahead_vehicle is None else self.obs.ahead_vehicle[0]
        return {
            "vel": self.runtime.v,
            "acc": acc,
            "rel_dis": rel_dis,
        }
