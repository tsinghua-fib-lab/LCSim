# LCSim

**LCSim: A Large-Scale Controllable Traffic Simulator**

[
[**Webpage**](https://tsinghua-fib-lab.github.io/LCSim/) |
[**Code**](https://github.com/tsinghua-fib-lab/LCSim) |
[**Paper**](https://arxiv.org/abs/2406.19781)
]

## Getting Started

**step 1**: clone the repository

```bash
git clone https://github.com/tsinghua-fib-lab/LCSim.git
```

**step 2**: create a virtual environment with conda or virtualenv

```bash
# with conda
conda create -n lcsim python=3.10
conda activate lcsim
# with virtualenv
virtualenv -p python3.10 lcsim
source lcsim/bin/activate
```

**step 3**: install the dependencies

```bash
pip install -r requirements.txt
```

## Training the diffusion model

Codes for training the diffusion model are in the `motion_diff` directory.

**step 1**: dataset preparation

- Download the [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/), we use version 1.2. We use the 9s scenario for training the diffusion model.
- Run the following command to preprocess the dataset

```bash
python3 motion_diff/dataset/process/process.py --data-dir /path/to/waymo_open_motion_dataset_v1.2 --output-dir /path/to/output_dir --dataset training/validation
```

This will generate the preprocessed dataset in h5 format (training.h5/validation.h5) in the output directory.

**step 2**: training the diffusion model

- Modify the configuration file `motion_diff/configs/config.yml` to specify the dataset path and other hyperparameters.

- Run the following command to train the diffusion model

```bash
python3 experiments/diff/train_md.py --config motion_diff/configs/config.yml --save /path/to/log_dir
```

The trained model will be saved in the log directory and you can check the training process in tensorboard by running `tensorboard --logdir /path/to/log_dir`. We trained our model for 200 epochs on the whole training set of WOMD, wihch takes about 20 days on 4 NVIDIA 4090 GPUs, the hyperparameters are the same as the ones in the config file.

## Running the simulation

**step 1**: scenario data preparation

We provide scenario construction tools from multiple sources, including the Waymo Open Motion Dataset (WOMD), the Argoverse dataset, and the [MOSS](https://moss.fiblab.net/) scenrios. You can convert them into the unified format for simulation by scripts in the `lcsim/scripts/scenario_converts` directory. Example scenarios are provided in the `examples/data` directory.

**step 2**: running the simulation

Whole prosess of running the simulation is in the `exapmles/simulation.ipynb` notebook. You can run the notebook to see the simulation results.

## Reinforcement learning

For multi-style reinforcement learning, you need to train the diffusion model first, and then specify the path to the trained model in the configuration file `experiments/rl/configs/waymo_ppo.yml`. Then you can run the following command to train the reinforcement learning model.

```bash
python3 experiments/rl/train_waymo.py 
```

By modifying the environment configuration in the configuration file, you can enable diffusive simulation and guidance for constructing driving scenarios with different styles.

## Citation

If you find this work useful, please consider citing our paper:

```
@misc{zhang2024lcsimlargescalecontrollabletraffic,
    title={LCSim: A Large-Scale Controllable Traffic Simulator}, 
    author={Yuheng Zhang and Tianjian Ouyang and Fudan Yu and Cong Ma and Lei Qiao and Wei Wu and Jian Yuan and Yong Li},
    year={2024},
    eprint={2406.19781},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    url={https://arxiv.org/abs/2406.19781}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
