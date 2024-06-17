import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...gen.agent import agent_pb2
from ...lane import lane
from ...utils import polygon
from ..route import Route
from .type import Action, BasePolicy, Observation


class LaneIDM(BasePolicy):
    """
    IDM policy for agent move on lane.
    """

    class LCRuntime:
        """
        Lane change runtime.
        """

        is_lc: bool = False  # lane change or not
        total_length: float = 0.0  # total length of lane change
        completed_length: float = 0.0  # completed length of lane change
        target_lane: lane.Lane = None  # target lane for lane change
        target_lane_s: float = 0.0  # s on target lane

    LC_LENGTH_FACTOR = 3  # lane change length factor(s)
    # parameters for IDM
    usual_braking_a: float = -1.5  # usual braking deceleration [m/s^2]
    max_braking_a: float = -3.0  # max braking deceleration [m/s^2]
    max_a = 1.5  # max acceleration [m/s^2]
    max_v: float = 15.0  # max velocity [m/s]
    min_gap: float = 2.0  # min gap [m]
    headway: float = 2.0  # time headway [s]
    # parameters for lane change
    force_lc: bool = False  # force lane change
    last_lc_time: float = 0.0  # last lane change time
    lc_runtime: LCRuntime  # lane change runtime

    def __init__(self, agent):
        super().__init__(agent)
        pb: agent_pb2.Agent = agent.pb
        attr = pb.attribute
        self.usual_braking_a = attr.usual_braking_acceleration
        self.max_braking_a = attr.max_braking_acceleration
        self.max_a = attr.max_acceleration
        self.max_v = attr.max_speed
        self.min_gap = pb.vehicle_attribute.min_gap
        self.lc_runtime = LaneIDM.LCRuntime()

    def get_action(self, obs: Observation) -> Action:
        """
        Get action for agent update.
        """
        data = Action.LaneAction()
        distance, ahead_v = (
            obs.ahead_vehicle if obs.ahead_vehicle is not None else (math.inf, 0.0)
        )
        self._policy_car_follow(
            data,
            self.agent.v,
            min(self.max_v, obs.cur_lane.max_speed),
            ahead_v,
            distance,
        )
        self._policy_lane(data, obs.cur_lane, obs.ahead_lanes)
        if self.lc_runtime.is_lc:
            # lane changing, look into target lane for acc
            target_lane = self.lc_runtime.target_lane
            neighbor_s = self.lc_runtime.target_lane_s
            ahead = target_lane.get_next_agent_by_s(neighbor_s)
            if ahead is not None:
                # car following
                ahead_a = self.agent.engine.agent_manager.get_agent_by_id(ahead[1])
                assert ahead_a is not None
                ahead_v = ahead_a.runtime.v
                distance = ahead[0] - (self.agent.length + ahead_a.length) / 2.0
                lc_action = Action.LaneAction()
                self._policy_car_follow(
                    lc_action, self.agent.v, target_lane.max_speed, ahead_v, distance
                )
                data.acc = min(data.acc, lc_action.acc)
        if (not self.lc_runtime.is_lc) and (obs.cur_lane.in_road()):
            self._policy_lane_change(data, obs.cur_lane, obs.s, self.agent.v)
        return Action(Action.ActionType.LANE, data)

    def clear_lc(self):
        self.lc_runtime = LaneIDM.LCRuntime()

    def _policy_car_follow(
        self,
        action: Action.LaneAction,
        self_v: float,
        target_v: float,
        ahead_v: float,
        distance: float,
    ):
        """
        Set acceleration for car following with IDM.
        """
        acc = 0
        if distance < 0:
            acc = -math.inf
        else:
            s_star = self.min_gap + max(
                self_v * self.headway
                + self_v
                * (self_v - ahead_v)
                / (2 * math.sqrt(-self.usual_braking_a * self.max_a)),
                0,
            )
            acc = self.max_a * (1 - (self_v / target_v) ** 4 - (s_star / distance) ** 2)
        acc = max(self.max_braking_a, min(self.max_a, acc))
        action.acc = acc

    def _policy_lane(
        self,
        action: Action.LaneAction,
        cur_lane: lane.Lane,
        ahead_lanes: List[Tuple[float, lane.Lane]],
    ):
        """
        Set acceleration for lane speed control.
        """
        if len(ahead_lanes) == 0:
            return
        pass

    def _policy_lane_change(
        self, action: Action.LaneAction, cur_lane: lane.Lane, cur_s: float, cur_v: float
    ):
        reverse_s = cur_lane.length - cur_s
        lc_length = max(
            self.LC_LENGTH_FACTOR * cur_v, self.agent.length
        )  # at least one vehicle length
        lc: Route.LC = self.agent.route.get_lc(cur_lane, cur_s, cur_v)
        if (
            not lc.in_candidate
            and reverse_s - lc.force_lc_length <= lc_length * lc.count
        ):
            # force lane change
            target_lane = cur_lane.neighbor_lane(lc.side)
            assert target_lane is not None
            self.last_lc_time = self.agent.engine.current_time
            action.start_lane_change(target_lane, lc_length)
            # acc adjustment for lane change
            neighbor_s = target_lane.project_from_lane(cur_lane, cur_s)
            ahead = target_lane.get_next_agent_by_s(neighbor_s)
            if ahead is not None:
                # car following
                ahead_a = self.agent.engine.agent_manager.get_agent_by_id(ahead[1])
                assert ahead_a is not None
                ahead_v = ahead_a.runtime.v
                distance = ahead[0] - (self.agent.length + ahead_a.length) / 2.0
                lc_action = Action.LaneAction()
                self._policy_car_follow(
                    lc_action, cur_v, target_lane.max_speed, ahead_v, distance
                )
                action.acc = min(action.acc, lc_action.acc)
            # update lc runtime
            self.lc_runtime.is_lc = True
            self.lc_runtime.total_length = lc_length
            self.lc_runtime.completed_length = 0
            self.lc_runtime.target_lane = target_lane
            self.lc_runtime.target_lane_s = neighbor_s


class TrajIDM(BasePolicy):
    """
    IDM following given trajectory.
    """

    MAX_ACC = 3.0  # max acceleration [m/s^2]
    MAX_STEER = 0.3  # max steering angle [rad]
    SPEED_LIMIT = 0.6  # speed limit [m/s]
    # parameters for IDM
    usual_braking_a: float = -2  # usual braking deceleration [m/s^2]
    max_braking_a: float = -2  # max braking deceleration [m/s^2]
    max_a = 2  # max acceleration [m/s^2]
    max_v: float = 40.0  # max velocity [m/s]
    min_gap: float = 10.0  # min gap [m]
    headway: float = 2.0  # time headway [s]

    def __init__(self, agent):
        super().__init__(agent)
        # pb: agent_pb2.Agent = agent.pb
        # attr = pb.attribute
        # self.usual_braking_a = attr.usual_braking_acceleration
        # self.max_braking_a = attr.max_braking_acceleration
        # self.max_a = attr.max_acceleration
        # self.max_v = attr.max_speed
        # self.min_gap = pb.vehicle_attribute.min_gap

    def get_action(self, obs: Observation, dt: float = 0.1) -> Action:
        """
        Get action for agent update.
        """
        assert obs.ref_trajectory is not None
        # get action from traj
        xy, vel, heading = obs.ref_trajectory.next_waypoint()
        v = np.linalg.norm(vel)
        # get current state
        cur_xy, cur_vel, cur_heading = obs.cur_runtime
        # calculate acc
        acc = (v - cur_vel) / dt
        acc = np.clip(acc, -self.MAX_ACC, self.MAX_ACC)
        # calculate steer
        delta_heading = polygon.wrap_angle(heading - cur_heading)
        steer = (
            0
            if cur_vel < self.SPEED_LIMIT or v < self.SPEED_LIMIT
            else delta_heading / (cur_vel * dt + 0.5 * acc * dt**2)
        )
        steer = np.clip(steer, -self.MAX_STEER, self.MAX_STEER)
        if obs.ahead_vehicle is not None:
            # idm car following
            idm_acc = self._get_acc(
                cur_vel, self.max_v, obs.ahead_vehicle[1], obs.ahead_vehicle[0]
            )
            acc = max(min(acc, idm_acc), -self.MAX_ACC)
        data = Action.BicycleAction()
        data.acc = acc
        data.steer = steer
        return Action(Action.ActionType.BICYCLE, data)

    def _get_acc(self, self_v: float, target_v: float, ahead_v: float, distance: float):
        """
        Get acceleration for car following with IDM.
        """
        acc = 0
        if distance - self.min_gap < 0:
            acc = -math.inf
        else:
            s_star = self.min_gap + max(
                self_v * self.headway
                + self_v
                * (self_v - ahead_v)
                / (2 * math.sqrt(-self.usual_braking_a * self.max_a)),
                0,
            )
            acc = self.max_a * (1 - (self_v / target_v) ** 4 - (s_star / distance) ** 2)
        acc = max(self.max_braking_a, min(self.max_a, acc))
        return acc
