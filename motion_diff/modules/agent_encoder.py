from typing import Mapping

import torch
import torch.nn as nn
from torch_cluster import radius, radius_graph
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse, subgraph

from .layers.attention import AttentionLayer
from .layers.fourier_embedding import FourierEmbedding
from .utils import angle_between_2d_vectors, weight_init, wrap_angle


class AgentEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg["model"]["agent_encoder"]
        self.input_dim = cfg["model"]["input_dim"]
        self.hidden_dim = cfg["model"]["hidden_dim"]
        self.num_historical_steps = cfg["model"]["num_historical_steps"]
        self.time_span = self.cfg["time_span"]
        self.a2a_radius = self.cfg["a2a_radius"]
        self.pl2a_radius = self.cfg["pl2a_radius"]
        self.num_freq_bands = self.cfg["num_freq_bands"]
        self.num_layers = self.cfg["num_layers"]
        self.num_heads = self.cfg["num_heads"]
        self.head_dim = self.cfg["head_dim"]
        self.dropout = self.cfg["dropout"]

        input_dim_x_a = 4 + self.input_dim
        input_dim_r_t = 4
        input_dim_r_pl2a = 3
        input_dim_r_a2a = 3

        self.type_a_emb = nn.Embedding(5, self.hidden_dim)
        self.x_a_emb = FourierEmbedding(
            input_dim=input_dim_x_a,
            hidden_dim=self.hidden_dim,
            num_freq_bands=self.num_freq_bands,
        )
        self.r_t_emb = FourierEmbedding(
            input_dim=input_dim_r_t,
            hidden_dim=self.hidden_dim,
            num_freq_bands=self.num_freq_bands,
        )
        self.r_pl2a_emb = FourierEmbedding(
            input_dim=input_dim_r_pl2a,
            hidden_dim=self.hidden_dim,
            num_freq_bands=self.num_freq_bands,
        )
        self.r_a2a_emb = FourierEmbedding(
            input_dim=input_dim_r_a2a,
            hidden_dim=self.hidden_dim,
            num_freq_bands=self.num_freq_bands,
        )
        self.t_attn_layer = nn.ModuleList(
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
        self.pl2a_attn_layer = nn.ModuleList(
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
        self.a2a_attn_layer = nn.ModuleList(
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
        self, data: HeteroData, map_enc: torch.Tensor
    ) -> Mapping[str, torch.Tensor]:
        mask = (
            data["agent"]["valid"][:, : self.num_historical_steps]
            .squeeze(-1)
            .contiguous()
        )  # (num_agents, num_historical_steps)
        pos_a = data["agent"]["xyz"][
            :, : self.num_historical_steps, : self.input_dim
        ]  # (num_agents, num_historical_steps, input_dim)
        motion_vector_a = torch.cat(
            [
                pos_a.new_zeros(data["agent"]["num_nodes"], 1, self.input_dim),
                pos_a[:, 1:] - pos_a[:, :-1],
            ],
            dim=1,
        )  # (num_agents, num_historical_steps, input_dim)
        head_a = (
            data["agent"]["heading"][:, : self.num_historical_steps]
            .squeeze(-1)
            .contiguous()
        )  # (num_agents, num_historical_steps)
        head_vector_a = torch.stack(
            [head_a.cos(), head_a.sin()], dim=-1
        )  # (num_agents, num_historical_steps, 2)

        pl_center_pts = data["roadgraph"][
            "poly_center_points"
        ]  # (num_polylines, 7) [x, y, z, dir_x, dir_y, dir_z, type]
        pos_pl = pl_center_pts[
            :, : self.input_dim
        ].contiguous()  # (num_polylines, input_dim)
        dir_xy_pl = pl_center_pts[:, 3:5]
        head_pl = torch.atan2(
            dir_xy_pl[:, 1], dir_xy_pl[:, 0]
        ).contiguous()  # (num_polylines,)

        vel = data["agent"]["vel"][
            :, : self.num_historical_steps
        ].contiguous()  # (num_agents, num_historical_steps, 2)
        shape = data["agent"]["shape"][:, : self.num_historical_steps]
        length, width, height = (
            shape[..., 0],
            shape[..., 1],
            shape[..., 2],
        )  # (num_agents, num_historical_steps)
        shape_a = [length, width] if self.input_dim == 2 else [length, width, height]

        # feature embedding for agent
        categorical_embs = [
            self.type_a_emb(data["agent"]["type"].long()).repeat_interleave(
                self.num_historical_steps, dim=0
            )
        ]  # (num_agents * num_historical_steps, hidden_dim)
        x_a = torch.stack(
            [
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]
                ),
                torch.norm(vel[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_a, nbr_vector=vel[:, :, :2]
                ),
            ]
            + shape_a,
            dim=-1,
        )
        x_a = self.x_a_emb(
            continuous_inputs=x_a.view(-1, x_a.size(-1)),
            categorical_embs=categorical_embs,
        )
        x_a = x_a.view(-1, self.num_historical_steps, self.hidden_dim)
        x_a_his, x_a_cur = x_a[:, :-1], x_a[:, -1]

        # relation for agents to historical steps
        pos_t = pos_a.reshape(-1, self.input_dim)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)
        mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)  # (A, T, T)
        mask_t[..., :-1] = False  # target only the last historical step
        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[
            :, edge_index_t[1] - edge_index_t[0] <= self.time_span
        ]
        rel_post_t = pos_t[edge_index_t[1]] - pos_t[edge_index_t[0]]
        rel_head_t = wrap_angle(head_t[edge_index_t[1]] - head_t[edge_index_t[0]])
        r_t = torch.stack(
            [
                torch.norm(rel_post_t[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_t[edge_index_t[1]],
                    nbr_vector=rel_post_t[:, :2],
                ),
                rel_head_t,
                edge_index_t[0] - edge_index_t[1],
            ],
            dim=-1,
        )
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)

        # relation for polylines to agents
        pos_s = pos_a[:, -1]
        head_s = head_a[:, -1]
        head_vector_s = head_vector_a[:, -1]
        mask_s = mask[:, -1]
        batch_s = data["agent"]["batch"]
        batch_pl = data["roadgraph"]["batch"]
        edge_index_pl2a = radius(
            x=pos_s[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2a_radius,
            batch_x=batch_s,
            batch_y=batch_pl,
            max_num_neighbors=300,
        )
        edge_index_pl2a = edge_index_pl2a[:, mask_s[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_head_pl2a = wrap_angle(
            head_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]]
        )
        r_pl2a = torch.stack(
            [
                torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_pl2a[1]],
                    nbr_vector=rel_pos_pl2a[:, :2],
                ),
                rel_head_pl2a,
            ],
            dim=-1,
        )
        r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)

        # relation for agents to agents
        edge_index_a2a = radius_graph(
            x=pos_s,
            r=self.a2a_radius,
            batch=batch_s,
            loop=False,
            max_num_neighbors=300,
        )
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [
                torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_a2a[1]],
                    nbr_vector=rel_pos_a2a[:, :2],
                ),
                rel_head_a2a,
            ],
            dim=-1,
        )
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

        # attention layers
        for i in range(self.num_layers):
            x_a = torch.cat([x_a_his, x_a_cur.unsqueeze(1)], dim=1).view(
                -1, self.hidden_dim
            )
            x_a = self.t_attn_layer[i](x_a, r_t, edge_index_t).reshape(
                -1, self.num_historical_steps, self.hidden_dim
            )
            x_a_his, x_a_cur = x_a[:, :-1], x_a[:, -1]
            x_a_cur = self.pl2a_attn_layer[i](
                (map_enc, x_a_cur), r_pl2a, edge_index_pl2a
            )
            x_a_cur = self.a2a_attn_layer[i](x_a_cur, r_a2a, edge_index_a2a)

        return {"agent": x_a, "roadgraph": map_enc}
