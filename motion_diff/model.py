import inspect
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch_geometric.data import Batch, HeteroData

from .metrics import (
    compute_lane_heading_diff,
    compute_mmd_vel,
    compute_offroad_rate,
    compute_overlap_rate,
)
from .modules import AgentEncoder, DiffDecoder, MapEncoder
from .modules.layers.pca import PCA

P_MEAN = -1.2
P_STD = 1.2
VEL_MEAN = 0.2417  # 0.3741 argoverse, 0.2417 waymo
DYNAMIC_THRESHOLD = 1.0


class MotionDiff(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super(MotionDiff, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg["model"]
        self.target = self.cfg["target"]
        self.sigma_data = self.cfg["sigma_data"]
        self.input_dim = self.cfg["input_dim"]
        self.output_dim = self.cfg["output_dim"]
        self.output_head = self.cfg["diff_decoder"]["output_head"]
        self.pca_dim = self.cfg["diff_decoder"]["pca_dim"]
        self.num_historical_steps = self.cfg["num_historical_steps"]
        self.num_future_steps = self.cfg["num_future_steps"]
        self.lr = self.cfg["lr"]
        self.lr_scheduler = self.cfg["lr_scheduler"]
        self.weight_decay = self.cfg["weight_decay"]
        self.T_max = self.cfg["T_max"]
        pca_path = self.cfg["pca_path"]
        if self.target == "pca":
            assert pca_path is not None
        if pca_path is not None:
            self.pca = PCA(pickle.load(open(pca_path, "rb")))

        self.agent_encoder = AgentEncoder(cfg)
        self.map_encoder = MapEncoder(cfg)
        self.diff_decoder = DiffDecoder(cfg)

    def forward(
        self, data: HeteroData, noised_gt: torch.Tensor, noise_label: torch.Tensor
    ) -> torch.Tensor:
        map_enc = self.map_encoder(data)
        enc = self.agent_encoder(data, map_enc)
        out = self.diff_decoder(data, enc, noised_gt, noise_label)
        return out

    def training_step(self, batch, batch_idx):
        pre_mask = (
            batch["agent"]["valid"][:, self.num_historical_steps - 1 :]
            .squeeze(-1)
            .all(dim=-1)
        )
        gt = self._get_training_targets(batch)

        # add noise to gt
        batch_size = 1
        size = gt.shape[0]
        if isinstance(batch, Batch):
            batch_size = batch.num_graphs
            size = batch["agent"]["ptr"][1:] - batch["agent"]["ptr"][:-1]
        if self.target == "pca":
            rnd_normal = torch.randn((batch_size, 1)).to(gt.device)
            pre_mask = pre_mask.unsqueeze(-1)
        else:
            rnd_normal = torch.randn((batch_size, 1, 1)).to(gt.device)
            pre_mask = pre_mask.unsqueeze(-1).unsqueeze(-1)
        sigma = (rnd_normal * P_STD + P_MEAN).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        sigma = sigma.repeat_interleave(size, dim=0)
        n = torch.randn_like(gt) * sigma

        # precondition parameters
        cin = 1 / (sigma**2 + self.sigma_data**2).sqrt()
        cskip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        cout = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        cnoise = sigma.log() / 4

        # forward
        out = self.forward(batch, (gt + n) * cin, cnoise.flatten())
        d_gt = out * cout + gt * cskip

        # loss
        loss = F.mse_loss(d_gt, gt, reduction="none") * pre_mask
        assert size.sum() == loss.shape[0]
        s = size.reshape(-1, 1) if self.target == "pca" else size.reshape(-1, 1, 1)
        sum_weight = (weight / s).repeat_interleave(size, dim=0)
        loss = (loss * sum_weight)[pre_mask.flatten()].sum(dim=0).mean() / batch_size

        self.log(
            "train_loss",
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        scene_enc = self.agent_encoder(batch, self.map_encoder(batch))
        sample = self.sampling(data=batch, scene_enc=scene_enc, show_progress=False)
        pre_mask = (
            batch["agent"]["valid"][:, self.num_historical_steps - 1 :]
            .squeeze(-1)
            .all(dim=-1)
        )
        xy = batch["agent"]["xyz"][:, self.num_historical_steps :, :2]
        dynamic_mask = (
            torch.abs(torch.max(xy, dim=1)[0] - torch.min(xy, dim=1)[0]).norm(
                dim=-1, p=2
            )
            > DYNAMIC_THRESHOLD
        )
        mask = pre_mask & dynamic_mask
        gen_xy, gen_heading, _ = self._reconstruct_traj(
            batch, sample, with_heading_and_vel=True
        )

        # complete traj with [x, y, length, width, yaw] for metric calculation
        lw = (
            batch["agent"]["shape"][:, self.num_future_steps, :2]
            .unsqueeze(1)
            .repeat(1, self.num_future_steps, 1)
        )
        traj = torch.cat([gen_xy, lw, gen_heading.unsqueeze(-1)], dim=-1)

        # compute metrics
        overlap_rate = compute_overlap_rate(batch, traj, mask)
        offroad_rate = compute_offroad_rate(batch, traj, mask)
        lane_heading_diff = compute_lane_heading_diff(batch, traj, mask)
        # mmd_vel = compute_mmd_vel(batch, traj, mask)

        batch_size = batch.num_graphs if isinstance(batch, Batch) else 1
        self.log(
            "val_overlap_rate",
            value=overlap_rate,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val_offroad_rate",
            value=offroad_rate,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val_lane_heading_diff",
            value=lane_heading_diff,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        # self.log(
        #     "val_mmd_vel",
        #     value=mmd_vel,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        #     batch_size=batch_size,
        # )

    def validation(self, batch, batch_idx=0, use_real_data=False, guide_fn=None):
        xy = batch["agent"]["xyz"][:, self.num_historical_steps :, :2]
        dynamic_mask = (
            torch.abs(torch.max(xy, dim=1)[0] - torch.min(xy, dim=1)[0]).norm(
                dim=-1, p=2
            )
            > DYNAMIC_THRESHOLD
        )
        lw = (
            batch["agent"]["shape"][:, self.num_future_steps, :2]
            .unsqueeze(1)
            .repeat(1, self.num_future_steps, 1)
        )
        pre_mask = (
            batch["agent"]["valid"][:, self.num_historical_steps - 1 :]
            .squeeze(-1)
            .all(dim=-1)
        )
        type_mask = batch["agent"]["type"] == 1

        if use_real_data:
            gen_xy = xy
            gen_heading = batch["agent"]["heading"][:, self.num_historical_steps :, :]
            traj = torch.cat([gen_xy, lw, gen_heading], dim=-1)
            mask = pre_mask
        else:
            scene_enc = self.agent_encoder(batch, self.map_encoder(batch))
            sample = self.sampling(
                data=batch, scene_enc=scene_enc, show_progress=False, guide_fn=guide_fn
            )
            gen_xy, gen_heading, _ = self._reconstruct_traj(
                batch, sample, with_heading_and_vel=True
            )
            traj = torch.cat([gen_xy, lw, gen_heading.unsqueeze(-1)], dim=-1)
            mask = pre_mask & dynamic_mask & type_mask
            # mask = pre_mask & type_mask
        # compute metrics
        overlap_rate = compute_overlap_rate(batch, traj, mask)
        offroad_rate = compute_offroad_rate(batch, traj, mask)
        # lane_heading_diff = compute_lane_heading_diff(batch, traj, mask)
        return {
            "overlap_rate": overlap_rate.item(),
            "offroad_rate": offroad_rate.item(),
            # "lane_heading_diff": lane_heading_diff.item(),
        }

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.LSTMCell,
            nn.GRU,
            nn.GRUCell,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        params = {
            "optimizer": optimizer,
            "T_max": self.T_max,
            "total_steps": self.T_max,
            "max_lr": self.lr,
        }
        s = eval(self.lr_scheduler)
        scheduler = s(
            **{k: v for k, v in params.items() if k in inspect.signature(s).parameters}
        )
        return [optimizer], [scheduler]

    def sampling(
        self,
        data: HeteroData,
        num_steps=18,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
        eps_scaler=1.0,
        return_his: bool = False,
        guide_fn: Optional[nn.Module] = None,
        show_progress: bool = True,
        scene_enc: Optional[Dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | List[torch.Tensor]:
        if return_his:
            res = []
        device = next(self.parameters()).device
        # latent
        output_dim = self.output_dim + self.output_head
        if self.target == "pca":
            latent = torch.randn(
                (data["agent"]["num_nodes"], self.pca_dim),
                device=device,
            )
        else:
            latent = torch.randn(
                (data["agent"]["num_nodes"], self.num_future_steps, output_dim),
                device=device,
            )
        # scene encoding
        scene_enc = (
            self.agent_encoder(data, self.map_encoder(data))
            if scene_enc is None
            else scene_enc
        )
        # denoising time steps
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])],
        ).to(device)
        # sampling loop
        x_next = latent.to(torch.float64) * t_steps[0]
        if show_progress:
            bar = tqdm.tqdm(
                total=num_steps, unit="step", desc="Sampling", dynamic_ncols=True
            )
        for i, (t_cur, t_next) in list(
            enumerate(zip(t_steps[:-1], t_steps[1:]))
        ):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = torch.as_tensor(t_cur + gamma * t_cur).to(device)
            x_hat = x_cur + (
                t_hat**2 - t_cur**2
            ).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            denoised = self._edm_precondition(data, x_hat, t_hat, scene_enc).to(
                torch.float64
            )
            d_cur = (x_hat - denoised) / t_hat / eps_scaler
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self._edm_precondition(data, x_next, t_next, scene_enc).to(
                    torch.float64
                )
                d_prime = (x_next - denoised) / t_next / eps_scaler
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            # Apply guide function.
            if guide_fn is not None and i >= num_steps - guide_fn.guidance_steps:
                x_next = guide_fn(x_next, self, data)

            if return_his:
                res.append(x_next)
            if show_progress:
                bar.update(1)
        if show_progress:
            bar.close()
        if return_his:
            return res
        return x_next

    def _edm_precondition(
        self,
        data: HeteroData,
        x: torch.Tensor,
        sigma: torch.Tensor,
        scene_enc: Dict[str, torch.Tensor],
    ):
        sigma = (
            (
                sigma.to(torch.float32)
                .reshape(-1, 1)
                .repeat_interleave(x.shape[0], dim=0)
            )
            if self.target == "pca"
            else (
                sigma.to(torch.float32)
                .reshape(-1, 1, 1)
                .repeat_interleave(x.shape[0], dim=0)
            )
        )
        x = x.to(torch.float32)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4
        F_x = self.diff_decoder(data, scene_enc, x * c_in, c_noise.flatten())
        return c_skip * x + c_out * F_x

    def _get_training_targets(self, data: HeteroData) -> torch.Tensor:
        if self.target == "post_vel":
            traj = data["agent"]["target"][..., : self.output_dim]
            # diff
            gt = torch.cat((traj[:, :1], traj[:, 1:] - traj[:, :-1]), dim=1)
        elif self.target == "pca":
            traj = data["agent"]["target"][..., : self.input_dim]
            traj = traj.reshape(-1, self.num_future_steps * self.input_dim)
            gt = self.pca(traj)
        elif self.target == "module_vel":
            traj = data["agent"]["target"][..., : self.input_dim]
            vel = torch.cat(
                (traj[:, :1], traj[:, 1:] - traj[:, :-1]),
                dim=1,
            )
            module = torch.norm(vel, p=2, dim=-1, keepdim=True) - VEL_MEAN
            angle = torch.atan2(vel[..., 1], vel[..., 0]).unsqueeze(-1)
            gt = torch.cat((module, angle), dim=-1)
        else:
            raise ValueError(f"Unknown target type: {self.target}")
        return gt

    def _reconstruct_traj(
        self,
        data: HeteroData,
        sample: torch.Tensor,
        with_heading_and_vel: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_step = self.num_historical_steps - 1
        origin = data["agent"]["xyz"][:, time_step, :2]
        theta = data["agent"]["heading"][:, time_step].squeeze(-1)
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros((len(theta), 2, 2))
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        rot_mat_inv = rot_mat.transpose(1, 2)
        if self.target == "post_vel":
            a_center_traj = torch.cumsum(sample, dim=1)
        elif self.target == "module_vel":
            module = torch.clip(sample[..., 0] + VEL_MEAN, min=0)
            angle = sample[..., 1]
            motion_vector = torch.stack(
                (module * torch.cos(angle), module * torch.sin(angle)), dim=-1
            )
            a_center_traj = torch.cumsum(motion_vector, dim=1)
        elif self.target == "pca":
            a_center_traj = self.pca.forward(
                sample.reshape(sample.shape[0], -1), inverse=True
            ).reshape(-1, self.num_future_steps, 2)
        else:
            raise NotImplementedError
        # post pca
        if not self.target == "pca":
            a_center_traj = self.pca.forward(
                self.pca.forward(x=a_center_traj.reshape(a_center_traj.shape[0], -1)),
                inverse=True,
            ).reshape(a_center_traj.shape)
        traj = torch.bmm(a_center_traj, rot_mat_inv) + origin.unsqueeze(1)
        if not with_heading_and_vel:
            return traj
        traj_xy_with_origin = torch.cat((origin.unsqueeze(1), traj), dim=1)
        motion_vector = traj_xy_with_origin[:, 1:] - traj_xy_with_origin[:, :-1]
        heading = torch.atan2(motion_vector[..., 1], motion_vector[..., 0])
        vel = torch.zeros_like(motion_vector)
        vel[:, :-1, :] = (motion_vector[:, :-1] + motion_vector[:, 1:]) / 2 / 0.1
        vel[:, -1, :] = motion_vector[:, -1] / 0.1
        return traj, heading, vel
