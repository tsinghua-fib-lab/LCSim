import torch
import torch.nn as nn

from .layers.fourier_embedding import FourierEmbedding
from .layers.polyline_encoder import PointNetPolylineEncoder
from .utils import weight_init


class MapEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg["model"]["map_encoder"]
        self.input_dim = cfg["model"]["input_dim"]
        self.polyline_encoder = PointNetPolylineEncoder(
            in_channels=self.input_dim * 2,
            hidden_dim=self.cfg["polyline"]["hidden_dim"],
            num_layers=self.cfg["polyline"]["num_layers"],
            num_pre_layers=self.cfg["polyline"]["num_pre_layers"],
            out_channels=cfg["model"]["hidden_dim"],
        )
        self.pl_type_embedding = nn.Embedding(
            self.cfg["num_pl_types"], cfg["model"]["hidden_dim"]
        )
        self.pl_emb = FourierEmbedding(
            input_dim=0,
            hidden_dim=cfg["model"]["hidden_dim"],
            num_freq_bands=self.cfg["num_freq_bands"],
        )
        self.apply(weight_init)

    def forward(self, data):
        poly_points = data["roadgraph"][
            "poly_points"
        ]  # (num_polylines, num_points_each_polylines, 6)
        if self.input_dim == 2:
            poly_points = torch.cat(
                [poly_points[..., :2], poly_points[..., 3:5]], dim=-1
            )  # (num_polylines, num_points_each_polylines, 4)
        poly_points_mask = data["roadgraph"][
            "poly_points_mask"
        ]  # (num_polylines, num_points_each_polylines)
        polylines_type = torch.clamp_min(
            data["roadgraph"]["poly_center_points"][..., -1].long(), 0
        )  # (num_polylines,)

        poly_category_embs = [
            self.pl_type_embedding(polylines_type)
        ]  # (num_polylines, hidden_dim)
        poly_feature = self.pl_emb(
            continuous_inputs=None, categorical_embs=poly_category_embs
        )  # (num_polylines, hidden_dim)
        poly_points_feature = self.polyline_encoder(
            poly_points, poly_points_mask
        )  # (num_polylines, hidden_dim)

        return poly_feature + poly_points_feature
