import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...utils import polygon
from .type import Action, BasePolicy, Observation


class StateExpert(BasePolicy):
    """
    Get next state from expert trajectory.
    """

    def __init__(self, agent):
        super().__init__(agent)

    def get_action(self, obs: Observation) -> Action:
        """
        Get action for agent update.
        """
        assert obs.ref_trajectory is not None
        # get next state
        xy, vel, heading = obs.ref_trajectory.next_waypoint()
        data = Action.StateAction()
        data.x, data.y, data.vx, data.vy, data.heading = (
            xy[0],
            xy[1],
            vel[0],
            vel[1],
            heading,
        )
        return Action(type=Action.ActionType.STATE, data=data)


class BicycleExpert(BasePolicy):
    """
    Infer bicycle action from expert trajectory.
    """

    MAX_ACC = 6.0  # max acceleration [m/s^2]
    MAX_STEER = 0.3  # max steering angle [rad]
    SPEED_LIMIT = 0.6  # speed limit [m/s]

    def __init__(self, agent):
        super().__init__(agent)

    def get_action(
        self, obs: Observation, dt: float = 0.1, estimate_heading_from_vel: bool = True
    ) -> Action:
        """
        Get action for agent update.
        """
        assert obs.ref_trajectory is not None
        # get next state
        xy, vel, heading = obs.ref_trajectory.next_waypoint()
        v = np.linalg.norm(vel)
        # get current state
        cur_xy, cur_vel, cur_heading = obs.cur_runtime
        # calculate acc
        acc = (v - cur_vel) / dt
        acc = np.clip(acc, -self.MAX_ACC, self.MAX_ACC)
        # calculate steer
        heading = np.arctan2(vel[1], vel[0]) if estimate_heading_from_vel else heading
        delta_heading = polygon.wrap_angle(heading - cur_heading)
        steer = (
            0
            if cur_vel < self.SPEED_LIMIT or v < self.SPEED_LIMIT
            else delta_heading / (cur_vel * dt + 0.5 * acc * dt**2)
        )
        steer = np.clip(steer, -self.MAX_STEER, self.MAX_STEER)
        data = Action.BicycleAction()
        data.acc, data.steer = acc, steer
        return Action(type=Action.ActionType.BICYCLE, data=data)
