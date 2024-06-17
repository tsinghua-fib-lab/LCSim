import copy
import os
from typing import List

import gymnasium as gym
from gymnasium import spaces

from ..engine import Engine
from ..entity.agent.policy import Action


class BicycleSingleEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"]}
    agent_ids: List[int]
    engine_list: List[Engine]
    cur_index: int

    def __init__(self, config: dict, render_mode=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
        self.observation_space = spaces.Dict(
            {
                "scene_emb": spaces.Box(
                    low=-100.0, high=100.0, shape=(128,), dtype=float
                ),
                "route": spaces.Box(low=-100.0, high=100.0, shape=(10, 2), dtype=float),
            }
        )
        self.release_idle = config["control"]["release_idle"]
        scenario_path_list = sorted(os.listdir(config["input"]["dataset_path"]))
        num_scenarios = config["input"]["num_scenarios"]
        assert num_scenarios <= len(scenario_path_list)
        try:
            worker_index, num_workers = config.worker_index, config.num_workers
        except AttributeError:
            worker_index, num_workers = 1, 1
        # worker_index, num_workers = 1, 8
        scenario_path_list = scenario_path_list[:num_scenarios]
        # split the scenarios into different workers
        data_dirs = [
            scenario_path_list[i]
            for i in range(worker_index - 1, len(scenario_path_list), num_workers)
        ]
        # initialize the engine
        self.engine_list, self.agent_ids = [], []
        for data_dir in data_dirs:
            config = copy.deepcopy(config)
            config["input"]["data_dir"] = os.path.join(
                config["input"]["dataset_path"], data_dir
            )
            engine = Engine(config)
            # release deep model after initialization
            if self.release_idle:
                engine.release_deep_models()
            self.engine_list.append(engine)
            self.agent_ids.append(engine.agent_manager.ads_id)
        self.cur_index = 0
        self.engine = self.engine_list[self.cur_index]
        if self.release_idle:
            self.engine.load_deep_models()
        self.agent_id = self.agent_ids[self.cur_index]

    def reset(self, seed=None, options=None):
        self.engine.reset()
        obs, _, _, _, info = self.engine.get_step_return(self.agent_id)
        if self.release_idle:
            self.engine.release_deep_models()
        # move to the next scenario
        self.cur_index = (self.cur_index + 1) % len(self.engine_list)
        self.engine = self.engine_list[self.cur_index]
        if self.release_idle:
            self.engine.load_deep_models()
        self.agent_id = self.agent_ids[self.cur_index]
        return obs, info

    def step(self, action):
        data = Action.BicycleAction()
        data.init_with_normalized(action[0], action[1])
        a = Action(
            type=Action.ActionType.BICYCLE,
            data=data,
        )
        action_dict = {self.engine.agent_manager.get_ads().id: a}
        self.engine.step(action_dict)
        return self.engine.get_step_return(self.agent_id)

    def render(self):
        if self.render_mode is None:
            return
        return self.engine.render()
