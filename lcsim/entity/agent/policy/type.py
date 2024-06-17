from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...lane import lane
from ...utils import polygon


class Trajectory:
    xy: np.ndarray  # (num_steps, 2)
    vel: np.ndarray  # (num_steps, 2)
    heading: np.ndarray  # (num_steps,)

    # frenet coordinate of current position on trajectory
    index: int
    d: float
    s: float

    def __init__(self, xy: np.ndarray, vel: np.ndarray, heading: np.ndarray):
        self.xy = xy
        self.vel = vel
        self.heading = heading
        self.index, self.d, self.s = 0, 0.0, 0.0

    def __len__(self) -> int:
        return len(self.xy)

    def cur_waypoint(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.xy[self.index], self.vel[self.index], self.heading[self.index]

    def next_waypoint(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self) == 1:
            return self.cur_waypoint()
        assert self.index + 1 < len(self.xy)
        return (
            self.xy[self.index + 1],
            self.vel[self.index + 1],
            self.heading[self.index + 1],
        )

    def next(self, pos: np.ndarray):
        """
        Compute Frenet coordinate of pos on trajectory, set index to next waypoint.
        """
        assert self.index + 1 <= len(self.xy)
        d, s, index = polygon.frenet_projection(pos, self.xy)
        self.d, self.s, self.index = d, s, index

    def reach_end(self) -> bool:
        return self.index >= len(self.xy) - 1

    def get_route(self, cur_xy: np.array, cur_heading: np.array) -> np.ndarray:
        route = self.xy[self.index + 1 :]
        # project to coordinate system of current position
        rot_matrix = np.array(
            [
                [np.cos(cur_heading), -np.sin(cur_heading)],
                [np.sin(cur_heading), np.cos(cur_heading)],
            ]
        )
        return np.dot(route - cur_xy, rot_matrix)

    def get_motion_vector(self) -> np.ndarray:
        _, _, heading = self.cur_waypoint()
        motion_vector = self.xy[self.index + 1 :] - self.xy[self.index : -1]
        # rotate motion vector to heading direction
        rot_matrix = np.array(
            [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
        )
        return np.dot(motion_vector, rot_matrix)


class Action:
    """
    Action for agent update.
    """

    class ActionType(Enum):
        UNSPECIFIED = 0
        BICYCLE = 1
        DELTA = 2
        STATE = 3
        LANE = 4

    class BicycleAction:
        """
        update by bicycle model with acc and steer.
        """

        MAX_ACC = 6.0
        MAX_STEER = 0.3

        acc: float = 0.0
        steer: float = 0.0

        def __str__(self) -> str:
            return f"acc: {self.acc}, steer: {self.steer}"

        def init_with_normalized(self, acc: float, steer: float) -> None:
            self.acc = acc * self.MAX_ACC
            self.steer = steer * self.MAX_STEER

    class DeltaAction:
        """
        update by delta between now and next step.
        """

        dx: float = 0.0
        dy: float = 0.0
        dheading: float = 0.0

        def __str__(self) -> str:
            return f"dx: {self.dx}, dy: {self.dy}, dheading: {self.dheading}"

    class StateAction:
        """
        directly update state.
        """

        x: float = 0.0
        y: float = 0.0
        vx: float = 0.0
        vy: float = 0.0
        heading: float = 0.0

        def __str__(self) -> str:
            return f"x: {self.x}, y: {self.y}, vx: {self.vx}, vy: {self.vy}, heading: {self.heading}"

    class LaneAction:
        """
        update by move on lane.
        """

        acc: float = 0.0
        start_lc: bool = False  # either start lane change or not
        lc_target: Optional[lane.Lane] = None  # target lane for lane change
        lc_length: float = 0.0  # length of lane change

        def __str__(self) -> str:
            return f"acc: {self.acc}, start_lc: {self.start_lc}, lc_target: {self.lc_target}, lc_length: {self.lc_length}"

        def start_lane_change(self, target: lane.Lane, length: float) -> None:
            self.start_lc = True
            self.lc_target = target
            self.lc_length = length

    ACTION_MAP = {
        ActionType.BICYCLE: BicycleAction,
        ActionType.DELTA: DeltaAction,
        ActionType.STATE: StateAction,
        ActionType.LANE: LaneAction,
    }

    type: ActionType
    data: Union[BicycleAction, DeltaAction, StateAction, LaneAction]

    def __init__(
        self,
        type: ActionType,
        data: Union[BicycleAction, DeltaAction, StateAction, LaneAction],
    ):
        assert isinstance(data, self.ACTION_MAP[type])
        self.type = type
        self.data = data


class Observation:
    """
    Observation for agent update.
    """

    # ptr of surrounding sim elements
    cur_lane: lane.Lane  # current lane
    s: float  # s on current lane
    ahead_lanes: List[Tuple[float, lane.Lane]]  # [distance, lane] ahead of current lane
    ahead_vehicle: Optional[Tuple[float, float]] = (
        None  # [distance, vel] of vehicle ahead, None if no vehicle
    )

    # embedding generated by scene encoder: (emb_dim,)
    scene_emb: Optional[np.ndarray]

    # reference trajectory for expert policy
    ref_trajectory: Optional[Trajectory]
    cur_runtime: Optional[Tuple[np.ndarray, float, float]]  # [xy, vel, heading]


class BasePolicy:
    """
    Base policy for agent move.
    """

    agent = None  # agent of this policy

    def __init__(self, agent):
        self.agent = agent

    def get_action(self, obs: Observation) -> Action:
        """
        Get action for agent update.
        """
        raise NotImplementedError
