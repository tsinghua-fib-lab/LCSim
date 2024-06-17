import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch

from motion_diff.model import MotionDiff
from motion_diff.modules.guidance import RealGuidance

from ..entity import Action
from ..entity.agent.manager import AgentManager
from ..entity.gen.agent import agent_pb2
from ..entity.gen.map import map_pb2
from ..entity.junction.manager import JunctionManager
from ..entity.lane.manager import LaneManager
from ..entity.road.manager import RoadManager
from ..entity.utils import polygon


class Engine:
    """
    Simulation engine.
    """

    # config
    config: dict
    # managers
    road_manager: RoadManager
    lane_manager: LaneManager
    junction_manager: JunctionManager
    agent_manager: AgentManager
    # deep models
    motion_diffuser: MotionDiff
    guidance: RealGuidance = None
    # clock
    start_time: float
    total_steps: int
    step_interval: float
    current_time: float
    current_step: int
    # control
    enable_poly_emb: bool
    enable_plan: bool
    plan_interval: int
    plan_init_step: int
    device: str
    allow_new_obj: bool
    reactive: bool
    no_static: bool

    def __init__(self, config: dict):
        """init simulation engine from config file.

        Args:
            config (dict): config file.
        """
        self.config = config

        # init clock
        step_config = config["control"]["step"]
        self.start_time = self.current_time = step_config["start"]
        self.total_steps = step_config["total"]
        self.step_interval = step_config["interval"]
        self.current_step = 0

        # control
        control_config = config["control"]
        self.enable_poly_emb = control_config["enable_poly_emb"]
        self.enable_plan = control_config["enable_plan"]
        if self.enable_plan:
            assert self.enable_poly_emb
        self.plan_interval = control_config["plan_interval"]
        self.plan_init_step = control_config["plan_init_step"]
        self.device = control_config["device"]
        self.allow_new_obj = control_config["allow_new_obj"]
        self.reactive = control_config["reactive"]
        self.no_static = control_config["no_static"]

        # reward
        self.reward_config = config["reward"]

        # init entity managers
        input_config = config["input"]
        data_dir = input_config["data_dir"]
        path_map, path_agents = os.path.join(data_dir, "map.pb"), os.path.join(
            data_dir, "agents.pb"
        )
        map_pb = map_pb2.Map()
        map_pb.ParseFromString(open(path_map, "rb").read())
        self.lane_manager = LaneManager(map_pb.lanes)
        self.road_manager = RoadManager(map_pb.roads, self.lane_manager)
        self.junction_manager = JunctionManager(
            map_pb.junctions, self.lane_manager, self
        )
        agents_pb = agent_pb2.Agents()
        agents_pb.ParseFromString(open(path_agents, "rb").read())
        self.agent_manager = AgentManager(agents_pb, self)

        # init deep models
        try:
            model_path = input_config["model_path"]
            model_config_path = input_config["model_config_path"]
            model_config = yaml.safe_load(open(model_config_path, "r"))
            self.motion_diffuser = MotionDiff.load_from_checkpoint(
                checkpoint_path=model_path,
                map_location=self.device,
                cfg=model_config,
            )
            self.motion_diffuser.eval()
            for para in self.motion_diffuser.parameters():
                para.requires_grad = False
            if control_config["guidance"]:
                self.guidance = RealGuidance()
        except Exception as e:
            print(f"Failed to load deep models: {e}")
            self.motion_diffuser = None

        if self.enable_poly_emb:
            # init polyline embeddings for lanes, roads and junctions
            self.lane_manager.init_polyline_emb(self.motion_diffuser)
            self.road_manager.init_polyline_emb(self.motion_diffuser)
            self.junction_manager.init_polyline_emb(self.motion_diffuser)
            self.junction_manager.init_roadgraph_data()

        # prepare
        self.agent_manager.check_departure_and_arrival(self.current_time)
        self.lane_manager.prepare()
        self.agent_manager.prepare_obs()

    def reset(self):
        """
        Reset the simulation engine to the initial state.
        """
        self.current_time = self.start_time
        self.current_step = 0
        # reset managers
        self.agent_manager.reset()
        self.lane_manager.reset()
        # prepare
        self.agent_manager.check_departure_and_arrival(self.current_time)
        self.agent_manager.prepare_obs()

    def step(self, actions: Dict[int, Action] = {}):
        """Step the simulation engine.

        Args:
            actions (Dict[int, Action]):  outer actions, agents provided won't be update by engine itself.

        Returns:
            Dict[int, Observation]: observations for agents.
        """
        self.update(actions)
        self.prepare()
        # update clock
        self.current_time += self.step_interval
        self.current_step += 1

    def get_step_return(self, agent_id: int):
        """
        Get the return of the current step, [obs, reward, terminated, truncated, info].
        """
        agent = self.agent_manager.get_agent_by_id(agent_id)
        assert agent is not None
        if not agent.runtime.is_finished():
            data_list, ids_list = [], []
            for junc in self.junction_manager.junctions.values():
                hetero = junc.get_hetero_data()
                if hetero is None:
                    continue
                data_list.append(hetero[0])
                ids_list.extend(hetero[1])
            if len(data_list) == 0:
                return
            batch = Batch.from_data_list(data_list).to(self.device)
            scene_enc = self.motion_diffuser.agent_encoder(
                batch, batch["roadgraph"]["poly_emb"]
            )["agent"]
            ads_index = ids_list.index(self.agent_manager.ads_id)
            scene_emb = scene_enc[ads_index, -1].detach().cpu().numpy()
            route = self.agent_manager.get_ads().get_route()
            obs = {
                "scene_emb": scene_emb,
                "route": route,
            }
            torch.cuda.empty_cache()
        else:
            obs = {
                "scene_emb": np.zeros(128),
                "route": np.zeros((10, 2)),
            }
        rewards, terminated = self._compute_rewards(
            self.reward_config.keys(), self.agent_manager.ads_id
        )
        reward = sum([rewards[key] * coef for key, coef in self.reward_config.items()])
        truncated = (
            self.current_step >= self.total_steps
            or self.agent_manager.get_ads().runtime.is_finished()
        )
        return obs, reward, terminated, truncated, {"rewards": rewards}

    def prepare(self):
        """
        Prepare stage: check departure and arrival, prepare observation for agents.
        """
        # prepare
        self.agent_manager.check_departure_and_arrival(
            self.current_time, self.allow_new_obj
        )
        self.lane_manager.prepare()
        self.agent_manager.prepare_obs()
        # plan trajectory
        if (
            self.enable_plan
            and self.current_step >= self.plan_init_step
            and (self.current_step - self.plan_init_step) % self.plan_interval == 0
        ):
            self._plan_trajectory()

    def update(self, actions: Dict[int, Action] = {}):
        """
        Update stage: get action for each agent, update agent status.
        """
        self.agent_manager.update(self.step_interval, actions)

    def render(self) -> np.ndarray:
        """
        Render the current simulation state, return the rendered image.
        """
        config = self.config["render"]
        if config["center_type"] == "agent":
            return self.agent_manager.render(config)
        elif config["center_type"] == "junction":
            return self.junction_manager.render(config)
        else:
            raise ValueError(f"Unknown center type {config['center_type']}")

    def release_deep_models(self):
        """
        Release deep models for saving memory.
        """
        self.motion_diffuser = None
        torch.cuda.empty_cache()

    def load_deep_models(self):
        """
        Load deep models.
        """
        model_path = self.config["input"]["model_path"]
        model_config_path = self.config["input"]["model_config_path"]
        model_config = yaml.safe_load(open(model_config_path, "r"))
        self.motion_diffuser = MotionDiff.load_from_checkpoint(
            checkpoint_path=model_path,
            map_location=self.device,
            cfg=model_config,
        )
        self.motion_diffuser.eval()
        for para in self.motion_diffuser.parameters():
            para.requires_grad = False

    def _plan_trajectory(self):
        """
        Generate planned trajectory for agents.
        """
        data_list, ids_list = [], []
        for junc in self.junction_manager.junctions.values():
            hetero = junc.get_hetero_data()
            if hetero is None:
                continue
            data_list.append(hetero[0])
            ids_list.extend(hetero[1])
        if len(data_list) == 0:
            return
        batch = Batch.from_data_list(data_list).to(self.device)
        scene_enc = self.motion_diffuser.agent_encoder(
            batch, batch["roadgraph"]["poly_emb"]
        )
        sample = self.motion_diffuser.sampling(
            data=batch,
            scene_enc=scene_enc,
            show_progress=False,
            # eps_scaler=0.999,
            guide_fn=self.guidance,
        )
        trajs, headings, vels = self.motion_diffuser._reconstruct_traj(
            data=batch, sample=sample, with_heading_and_vel=True
        )  # (num_agents, num_steps, 2)
        assert trajs.shape[0] == len(ids_list)
        trajs = trajs.detach().cpu().numpy()
        headings = headings.detach().cpu().numpy()
        vels = vels.detach().cpu().numpy()
        for i, _id in enumerate(ids_list):
            a = self.agent_manager.get_agent_by_id(_id)
            assert a is not None
            if a.dynamic_flag:
                a.set_ref_trajectory(trajs[i], vels[i], headings[i])
        # free memory cache
        torch.cuda.empty_cache()

    def _compute_rewards(
        self, keys: List[str], agent_id: int
    ) -> Tuple[Dict[str, float], bool]:
        """
        Compute reward for target agent. Return reward and whether the episode is terminated.
        """
        rewards = {key: 0.0 for key in keys}
        terminated = False
        agent = self.agent_manager.get_agent_by_id(agent_id)
        assert agent is not None
        # dense reward for moving forward & smooth penalty
        his_states = agent.history_states
        if his_states.valid.sum() <= 1:
            # no history states
            rewards["forward"] = 0.0
            rewards["smooth"] = 0.0
            rewards["route_progress"] = 0.0
        else:
            last_pos = his_states.xyz[-2, :2]
            cur_pos = his_states.xyz[-1, :2]
            last_heading = his_states.heading[-2]
            cur_heading = his_states.heading[-1]
            cur_v = np.linalg.norm(his_states.vel[-1])
            # project the vector of last_pos -> cur_pos to the heading direction
            last_d, last_s, _ = polygon.frenet_projection(
                last_pos, agent.ref_trajectory.xy
            )
            cur_d, cur_s, _ = polygon.frenet_projection(
                cur_pos, agent.ref_trajectory.xy
            )
            rewards["forward"] = (cur_s - last_s) - (cur_d - last_d)
            rewards["route_progress"] = cur_s / polygon.get_length(
                agent.ref_trajectory.xy
            )
            # smooth penalty
            steer = polygon.wrap_angle(cur_heading - last_heading) / np.linalg.norm(
                cur_pos - last_pos
            )
            steer = np.clip(
                steer, -Action.BicycleAction.MAX_STEER, Action.BicycleAction.MAX_STEER
            )
            rewards["smooth"] = min(0, float(1 / cur_v - abs(steer)))
        # collision penalty
        collision = self.agent_manager.check_collision(agent_id)
        rewards["collision"] = -1.0 if collision else 0.0
        # offroad penalty
        offroad = agent.is_offroad()
        rewards["offroad"] = -1.0 if offroad else 0.0
        # destination reward
        if agent.runtime.is_finished():
            dest = agent.ref_trajectory.xy[-1]
            cur_xy = agent.runtime.xy
            dis = np.linalg.norm(dest - cur_xy)
            rewards["destination"] = 10 if dis < 2.5 else -5
        # check termination
        terminated = collision or offroad
        return rewards, terminated
