from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch_geometric.data import Dataset, HeteroData

from .utils import TargetBuilder, generate_batch_polylines_from_map, wrap_angle


class WaymoMotionDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = TargetBuilder(),
        data_len: Optional[int] = None,
    ):
        super(WaymoMotionDataset, self).__init__(root, transform)
        if not split in ["train", "val", "test"]:
            raise ValueError(
                f"split should be one of 'train', 'val', 'test', but got {split}"
            )
        h5path = Path(root) / f"{split}.h5"
        self.h5file = h5py.File(h5path, "r")
        self.data_len = len(self.h5file.keys())
        if data_len is not None and data_len < self.data_len:
            self.data_len = data_len

    def len(self) -> int:
        return self.data_len

    def get(self, idx: int) -> HeteroData:
        key = str(idx)
        return self.preprocess(key)

    def preprocess(self, key) -> HeteroData:
        raw_data = self.h5file[key]
        # roadgraph
        roadgraph_xyz = raw_data["roadgraph_xyz"][:]
        roadgraph_dir = raw_data["roadgraph_dir"][:]
        roadgraph_type = raw_data["roadgraph_type"][:]
        ids = raw_data["roadgraph_id"][:]
        num_roadgraph_points = roadgraph_xyz.shape[0]
        poly_points, poly_points_mask, poly_center_points = (
            generate_batch_polylines_from_map(
                np.concatenate(
                    [roadgraph_xyz, roadgraph_dir, roadgraph_type[..., None]],
                    axis=1,
                ),
                ids,
                point_sampled_interval=1,
                num_points_each_polyline=20,
            )
        )
        num_polylines = poly_points.shape[0]
        # agent
        agent_xyz = raw_data["agent_xyz"][:]
        agent_vel = raw_data["agent_vel"][:]
        agent_heading = wrap_angle(raw_data["agent_heading"][:])
        agent_shape = raw_data["agent_shape"][:]
        agent_type = raw_data["agent_type"][:]
        agent_valid = raw_data["agent_valid"][:]
        num_agents = agent_xyz.shape[0]
        # global attributes
        scenario_id = raw_data.attrs["id"]
        sdc_index = raw_data.attrs["sdc_index"]
        tracks_to_predict = raw_data.attrs["tracks_to_predict"]
        # hetero data
        data = HeteroData(
            roadgraph={
                "poly_points": torch.from_numpy(poly_points).to(torch.float32),
                "poly_points_mask": torch.from_numpy(poly_points_mask).to(torch.bool),
                "poly_center_points": torch.from_numpy(poly_center_points).to(
                    torch.float32
                ),
                "num_nodes": num_polylines,
            },
            roadgraph_points={
                "xyz": torch.from_numpy(roadgraph_xyz).to(torch.float32),
                "dir": torch.from_numpy(roadgraph_dir).to(torch.float32),
                "type": torch.from_numpy(roadgraph_type).to(torch.int32),
                "ids": torch.from_numpy(ids).to(torch.int32),
                "num_nodes": num_roadgraph_points,
            },
            agent={
                "xyz": torch.from_numpy(agent_xyz).to(torch.float32),
                "vel": torch.from_numpy(agent_vel).to(torch.float32),
                "heading": torch.from_numpy(agent_heading).to(torch.float32),
                "shape": torch.from_numpy(agent_shape).to(torch.float32),
                "type": torch.from_numpy(agent_type).to(torch.int32),
                "valid": torch.from_numpy(agent_valid).to(torch.bool),
                "num_nodes": num_agents,
            },
            global_attrs={
                "scenario_id": scenario_id,
                "sdc_index": sdc_index,
                "tracks_to_predict": tracks_to_predict,
            },
        )
        return data
