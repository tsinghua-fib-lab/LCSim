import torch
import torch.nn as nn

from ..utils import build_mlps, weight_init


class PointNetPolylineEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim: int,
        num_layers=3,
        num_pre_layers=1,
        out_channels=None,
    ):
        super().__init__()
        self.pre_mlps = build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False,
        )
        self.mlps = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False,
        )

        if out_channels is not None:
            self.out_mlps = build_mlps(
                c_in=hidden_dim,
                mlp_channels=[hidden_dim, out_channels],
                ret_before_act=True,
                without_norm=True,
            )
        else:
            self.out_mlps = None
        self.apply(weight_init)

    def forward(self, polylines, polylines_mask):
        """
        Args:
            polylines (num_polylines, num_points_each_polylines, C):
            polylines_mask (num_polylines, num_points_each_polylines):

        Returns:
        """
        num_polylines, num_points_each_polylines, C = polylines.shape

        # pre-mlp
        polylines_feature_valid = self.pre_mlps(polylines[polylines_mask])  # (N, C)
        polylines_feature = polylines.new_zeros(
            num_polylines,
            num_points_each_polylines,
            polylines_feature_valid.shape[-1],
        )
        polylines_feature[polylines_mask] = polylines_feature_valid

        # get global feature
        pooled_feature = polylines_feature.max(dim=1)[0]
        polylines_feature = torch.cat(
            (
                polylines_feature,
                pooled_feature[:, None, :].repeat(1, num_points_each_polylines, 1),
            ),
            dim=-1,
        )  # (num_polylines, num_points_each_polylines, 2 * C)

        # mlp
        polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        feature_buffers = polylines_feature.new_zeros(
            num_polylines,
            num_points_each_polylines,
            polylines_feature_valid.shape[-1],
        )
        feature_buffers[polylines_mask] = polylines_feature_valid

        # max-pooling
        feature_buffers = feature_buffers.max(dim=1)[0]  # (num_polylines, C)

        # out-mlp
        if self.out_mlps is not None:
            valid_mask = polylines_mask.sum(dim=-1) > 0
            feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])  # (N, C)
            feature_buffers = feature_buffers.new_zeros(
                num_polylines, feature_buffers_valid.shape[-1]
            )
            feature_buffers[valid_mask] = feature_buffers_valid
        return feature_buffers
