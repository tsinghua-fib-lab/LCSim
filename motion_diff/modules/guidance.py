import math
from typing import Mapping

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from ..metrics import geometry, roadgraph


def wrap_angle(
    angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi
) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)


class Repeller(nn.Module):
    def __init__(
        self,
        epsilon=1e-6,
        alpha=0.05,
        num_step=10,
        radius=6,
        beta=0.05,
    ):
        super(Repeller, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_step = num_step
        self.radius = radius  # desired minimum distance between agents
        self.beta = beta  # clipping parameter

    def loss(
        self,
        x: torch.Tensor,
        diffuser,
        data: HeteroData,
    ):
        # reconstruct the trajectory
        traj = diffuser._reconstruct_traj(data, x)  # (agent_num, time_step, 2)
        shape = data["agent"]["shape"][:, 0, :2]
        # compute pairwise distance between agents at each time step
        dist = torch.norm(traj.unsqueeze(1) - traj.unsqueeze(0), dim=-1, p=2)
        # compute loss
        a = torch.nn.functional.relu(
            (1 - dist / self.radius)
            * (1 - torch.eye(dist.shape[0], device=dist.device)).unsqueeze(-1)
        )
        return torch.sum(a) / (torch.sum(a > 0) + self.epsilon)


class NoOffRoad(nn.Module):
    def __init__(self, epsilon=1e-6, alpha=0.05, num_step=10, radius=1.0, beta=0.05):
        super(NoOffRoad, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_step = num_step
        self.radius = radius  # desired minimum distance between agents
        self.beta = beta  # clipping parameter

    def loss(
        self,
        x: torch.Tensor,
        diffuser,
        data: HeteroData,
    ):
        # reconstruct the trajectory
        traj = diffuser._reconstruct_traj(data, x)  # (agent_num, time_step, 2)
        traj = traj.reshape(-1, 2)
        roadgraph_points = torch.cat(
            [
                data["roadgraph_points"]["xyz"],
                data["roadgraph_points"]["dir"],
                data["roadgraph_points"]["ids"].unsqueeze(-1),
                data["roadgraph_points"]["type"].unsqueeze(-1),
            ],
            dim=-1,
        ).contiguous()
        # compute signed distance to nearest road edge point
        signed_dis = roadgraph.compute_signed_distance_to_nearest_road_edge_point(
            query_points=traj, roadgraph_points=roadgraph_points
        )
        # compute loss
        a = torch.nn.functional.relu(self.radius + signed_dis)
        return torch.sum(a) / (torch.sum(a > 0) + self.epsilon)


class OnLane(nn.Module):
    def __init__(self, epsilon=1e-6, alpha=0.05, num_step=10, radius=1.0, beta=0.05):
        super(OnLane, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_step = num_step
        self.radius = radius  # desired minimum distance between agents
        self.beta = beta  # clipping parameter

    def loss(
        self,
        x: torch.Tensor,
        diffuser,
        data: HeteroData,
    ):
        xy = diffuser._reconstruct_traj(data, x)  # (agent_num, time_step, 2)
        dir_xy = torch.cat(
            [xy[:, 1:, :] - xy[:, :-1, :], xy[:, -1:, :] - xy[:, -2:-1, :]], dim=1
        )
        query_points_xy = xy.reshape(-1, 2)
        query_points_dir_xy = dir_xy.reshape(-1, 2)
        types = data["roadgraph_points"]["type"]
        centerline_index = (types == 1) | (types == 2)
        xyz = data["roadgraph_points"]["xyz"]
        dir = data["roadgraph_points"]["dir"]
        centerline_xy = xyz[centerline_index][:, :2]
        centerline_dir = dir[centerline_index][:, :2]
        query_points_angle = torch.atan2(
            query_points_dir_xy[:, 1], query_points_dir_xy[:, 0]
        )
        centerline_angle = torch.atan2(centerline_dir[:, 1], centerline_dir[:, 0])
        angle_diff = torch.abs(
            wrap_angle(query_points_angle[:, None] - centerline_angle)
        )
        distance = torch.norm(
            query_points_xy[:, None, :] - centerline_xy[None, :, :], dim=-1
        )  # (agent_num * time_step, centerline_num)
        # find nearest centerline point within 0.2 radian
        mask = torch.logical_and(
            angle_diff < 0.2,
            distance < 5.0,
        )
        mask_distance = torch.where(mask, distance, float("inf"))
        min_dis = torch.min(mask_distance, dim=1)[0]  # (agent_num * time_step)
        # compute loss
        return torch.mean(min_dis)


class RealGuidance(nn.Module):
    def __init__(self, alpha=0.5, num_step=10, beta=0.1, guidance_steps=18):
        super(RealGuidance, self).__init__()
        self.alpha = alpha
        self.num_step = num_step
        self.beta = beta
        self.guidance_steps = guidance_steps
        self.guidance_modules = nn.ModuleDict(
            {
                "repeller": Repeller(),
                "no_offroad": NoOffRoad(),
                # "onlane": OnLane(),
                # "velocity": Velocity(),
            }
        )
        self.guidance_factors = {
            "repeller": 10.0,
            "no_offroad": 5.0,
            "onlane": 5.0,
            "velocity": 1.0,
        }

    @torch.enable_grad()
    def forward(
        self,
        x: torch.Tensor,
        diffuser,
        data: HeteroData,
    ):
        """
        x: (agent_num, time_step, 2) output of the model
        his_states: historical states of the agents
        diffuser: the diffuser module
        """
        x = x.clone().detach().requires_grad_(True)
        x_0 = x.clone().detach()
        for _ in range(self.num_step):
            loss = 0
            for name, module in self.guidance_modules.items():
                loss += module.loss(x, diffuser, data) * self.guidance_factors[name]
            grad = torch.autograd.grad(loss, x)[0]
            x = x - self.alpha * grad
            delta = torch.clip(x - x_0, -self.beta, self.beta)
            x = x_0 + delta
        return x
