import os
from argparse import ArgumentParser
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import h5py
import tensorflow as tf
from tqdm import tqdm
from utils import decode_agent_feature, decode_roadgraph_feature
from waymo_open_dataset.protos import scenario_pb2


def main():
    parser = ArgumentParser(allow_abbrev=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--dataset", default="training")
    parser.add_argument("--out-dir")
    args = parser.parse_args()

    dataset_size = {
        "training": 486995,  # 487002
        "validation": 44097,
        "training_20s": 70541,
        "validation_interactive": 43479,
        "testing": 44920,
        "testing_interactive": 44154,
    }

    data_path = Path(args.data_dir) / args.dataset
    dataset = tf.data.TFRecordDataset(
        [str(p) for p in data_path.glob("*")], compression_type=""
    )
    output_path = Path(args.out_dir) / (args.dataset + ".h5")

    with h5py.File(output_path, "w") as hf:
        for i, data in tqdm(enumerate(dataset), total=dataset_size[args.dataset]):
            episode = hf.create_group(str(i))
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(data.numpy())
            # roadgraph
            roadgraph = decode_roadgraph_feature(scenario)
            for key, value in roadgraph.items():
                episode.create_dataset(key, data=value)
            # agent
            agent = decode_agent_feature(scenario)
            for key, value in agent.items():
                episode.create_dataset(key, data=value)
            # traffic light (not used)
            # global feature
            episode.attrs["id"] = scenario.scenario_id
            episode.attrs["sdc_index"] = scenario.sdc_track_index
            episode.attrs["tracks_to_predict"] = [
                cur_pred.track_index for cur_pred in scenario.tracks_to_predict
            ]
            episode.attrs["predict_difficulty"] = [
                cur_pred.difficulty for cur_pred in scenario.tracks_to_predict
            ]


if __name__ == "__main__":
    main()
