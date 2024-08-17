from typing import List, Mapping, Optional

import torch
import torch.nn as nn
from torch_cluster import radius, radius_graph
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import dense_to_sparse

from .layers.attention import AttentionLayer
from .layers.fourier_embedding import FourierEmbedding
from .layers.mlp import MLPLayer
from .utils import (
    angle_between_2d_vectors,
    bipartite_dense_to_sparse,
    weight_init,
    wrap_angle,
)


class DiffDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg["model"]["diff_decoder"]
        self.target = cfg["model"]["target"]
        self.input_dim = cfg["model"]["input_dim"]
        self.hidden_dim = cfg["model"]["hidden_dim"]
        self.num_historical_steps = cfg["model"]["num_historical_steps"]
        self.num_future_steps = cfg["model"]["num_future_steps"]
        self.output_head = self.cfg["output_head"]
        self.num_t2m_steps = self.cfg["num_t2m_steps"]
        self.pl2m_radius = self.cfg["pl2m_radius"]
        self.a2m_radius = self.cfg["a2m_radius"]
        self.num_freq_bands = self.cfg["num_freq_bands"]
        self.num_layers = self.cfg["num_layers"]
        self.num_recurrent_steps = (
            1 if self.target == "pca" else self.cfg["num_recurrent_steps"]
        )
        self.num_heads = self.cfg["num_heads"]
        self.head_dim = self.cfg["head_dim"]
        self.dropout = self.cfg["dropout"]
        self.pca_dim = self.cfg["pca_dim"]

        input_dim_r_t = 4
        input_dim_r_pl2m = 3
        input_dim_r_a2m = 3
        input_dim_r_m2m = 3

        in_dim = output_dim = (
            self.pca_dim
            if self.pca_dim is not None and self.target == "pca"
            else self.num_future_steps * (self.input_dim + self.output_head)
        )
        self.mlp_in = MLPLayer(
            input_dim=in_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )
        self.mlp_out = MLPLayer(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=output_dim // self.num_recurrent_steps,
        )

        self.noise_emb = FourierEmbedding(
            input_dim=1,
            hidden_dim=self.hidden_dim,
            num_freq_bands=self.num_freq_bands,
            noise=True,
        )
        self.r_t2m_emb = FourierEmbedding(
            input_dim=input_dim_r_t,
            hidden_dim=self.hidden_dim,
            num_freq_bands=self.num_freq_bands,
        )
        self.r_pl2m_emb = FourierEmbedding(
            input_dim=input_dim_r_pl2m,
            hidden_dim=self.hidden_dim,
            num_freq_bands=self.num_freq_bands,
        )
        self.r_a2m_emb = FourierEmbedding(
            input_dim=input_dim_r_a2m,
            hidden_dim=self.hidden_dim,
            num_freq_bands=self.num_freq_bands,
        )
        self.r_m2m_emb = FourierEmbedding(
            input_dim=input_dim_r_m2m,
            hidden_dim=self.hidden_dim,
            num_freq_bands=self.num_freq_bands,
        )

        self.t2m_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    dropout=self.dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.pl2m_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    dropout=self.dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.a2m_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    dropout=self.dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.m2m_attn_layer = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    dropout=self.dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.apply(weight_init)

    def forward(
        self,
        data: HeteroData,
        scene_enc: Mapping[str, torch.Tensor],
        noised_gt: torch.Tensor,
        noise_labels: torch.Tensor,
    ):
        noise_emb = self.noise_emb(
            continuous_inputs=noise_labels.unsqueeze(-1), categorical_embs=None
        )  # (num_agents, hidden_dim)
        if self.pca_dim is not None and self.target == "pca":
            m = self.mlp_in(noised_gt)
        else:
            m = self.mlp_in(
                noised_gt.view(
                    -1, self.num_future_steps * (self.input_dim + self.output_head)
                )
            )  # num_nodes, hidden_dim
        m = m + noise_emb

        pos_m = data["agent"]["xyz"][
            :, self.num_historical_steps - 1, : self.input_dim
        ]  # (num_agents, 2)
        head_m = data["agent"]["heading"][:, self.num_historical_steps - 1].squeeze(
            -1
        )  # (num_agents,)
        head_vector_m = torch.stack([torch.cos(head_m), torch.sin(head_m)], dim=-1)

        x_t = scene_enc["agent"].reshape(
            -1, self.hidden_dim
        )  # (num_agents * num_historical_steps, hidden_dim)
        x_pl = scene_enc["roadgraph"]  # (num_nodes, hidden_dim)
        x_a = scene_enc["agent"][
            :, self.num_historical_steps - 1, :
        ]  # (num_agents, hidden_dim)

        mask_src = (
            data["agent"]["valid"][:, : self.num_historical_steps]
            .squeeze(-1)
            .contiguous()
        )
        mask_src[:, : -self.num_t2m_steps] = False
        mask_dst = (
            data["agent"]["valid"][:, self.num_historical_steps :]
            .squeeze(-1)
            .contiguous()
        )
        mask_dst = mask_dst.all(dim=-1, keepdim=True)

        # relation to self historical states
        pos_t = data["agent"]["xyz"][
            :, : self.num_historical_steps, : self.input_dim
        ].reshape(-1, self.input_dim)
        head_t = data["agent"]["heading"][:, : self.num_historical_steps].reshape(-1)
        edge_index_t2m = bipartite_dense_to_sparse(
            mask_src.unsqueeze(2) & mask_dst[:, -1:].unsqueeze(1)
        )
        rel_pos_t2m = pos_t[edge_index_t2m[0]] - pos_m[edge_index_t2m[1]]
        rel_head_t2m = wrap_angle(head_t[edge_index_t2m[0]] - head_m[edge_index_t2m[1]])
        r_t2m = torch.stack(
            [
                torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_m[edge_index_t2m[1]],
                    nbr_vector=rel_pos_t2m[:, :2],
                ),
                rel_head_t2m,
                (edge_index_t2m[0] % self.num_historical_steps)
                - (self.num_historical_steps - 1),
            ],
            dim=-1,
        )
        r_t2m = self.r_t2m_emb(continuous_inputs=r_t2m, categorical_embs=None)

        # relation to roadgraph polyline
        pl_center_pts = data["roadgraph"]["poly_center_points"]
        pos_pl = pl_center_pts[:, : self.input_dim].contiguous()
        dir_xy_pl = pl_center_pts[:, 3:5]
        head_pl = torch.atan2(dir_xy_pl[:, 1], dir_xy_pl[:, 0]).contiguous()
        edge_index_pl2m = radius(
            x=pos_m[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2m_radius,
            batch_x=data["agent"]["batch"] if isinstance(data, Batch) else None,
            batch_y=data["roadgraph"]["batch"] if isinstance(data, Batch) else None,
            max_num_neighbors=300,
        )
        edge_index_pl2m = edge_index_pl2m[:, mask_dst[edge_index_pl2m[1], 0]]
        rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]]
        rel_head_pl2m = wrap_angle(
            head_pl[edge_index_pl2m[0]] - head_m[edge_index_pl2m[1]]
        )
        r_pl2m = torch.stack(
            [
                torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_m[edge_index_pl2m[1]],
                    nbr_vector=rel_pos_pl2m[:, :2],
                ),
                rel_head_pl2m,
            ],
            dim=-1,
        )
        r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2m, categorical_embs=None)

        # relation to other agents
        edge_index_a2m = radius_graph(
            x=pos_m[:, :2],
            r=self.a2m_radius,
            batch=data["agent"]["batch"] if isinstance(data, Batch) else None,
            loop=False,
            max_num_neighbors=300,
        )
        edge_index_a2m = edge_index_a2m[
            :, mask_src[:, -1][edge_index_a2m[0]] & mask_dst[edge_index_a2m[1], 0]
        ]
        rel_pos_a2m = pos_m[edge_index_a2m[0]] - pos_m[edge_index_a2m[1]]
        rel_head_a2m = wrap_angle(head_m[edge_index_a2m[0]] - head_m[edge_index_a2m[1]])
        r_a2m = torch.stack(
            [
                torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_m[edge_index_a2m[1]],
                    nbr_vector=rel_pos_a2m[:, :2],
                ),
                rel_head_a2m,
            ],
            dim=-1,
        )
        r_a2m = self.r_a2m_emb(continuous_inputs=r_a2m, categorical_embs=None)

        # relation to other prediction modules
        edge_index_m2m = dense_to_sparse(mask_dst.unsqueeze(2) & mask_dst.unsqueeze(1))[
            0
        ]
        rel_pos_m2m = pos_m[edge_index_m2m[0]] - pos_m[edge_index_m2m[1]]
        rel_head_m2m = wrap_angle(head_m[edge_index_m2m[0]] - head_m[edge_index_m2m[1]])
        r_m2m = torch.stack(
            [
                torch.norm(rel_pos_m2m[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_m[edge_index_m2m[1]],
                    nbr_vector=rel_pos_m2m[:, :2],
                ),
                rel_head_m2m,
            ],
            dim=-1,
        )
        r_m2m = self.r_m2m_emb(continuous_inputs=r_m2m, categorical_embs=None)

        # attention layers
        out: List[Optional[torch.Tensor]] = [
            None for _ in range(self.num_recurrent_steps)
        ]
        for t in range(self.num_recurrent_steps):
            for i in range(self.num_layers):
                m = m.reshape(-1, self.hidden_dim)
                m = self.t2m_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
                m = self.pl2m_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                m = self.a2m_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
                m = self.m2m_attn_layer[i](m, r_m2m, edge_index_m2m)
            out[t] = self.mlp_out(m)
        # output
        out = torch.cat(out, dim=-1)
        if self.target != "pca":
            out = out.view(-1, self.num_future_steps, self.input_dim + self.output_head)
        return out
