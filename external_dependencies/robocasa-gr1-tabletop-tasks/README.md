# RoboCasa GR-1 Tabletop Tasks

<p align="center">
  <img width="95.0%" src="images/fig_task_v6.png">
</p>

This repository contains the official release of simulation environments for the GR-1 Tabletop Tasks developed for NVIDIA's general-purpose humanoid foundation models: "*GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*." Selected task setup images are shown above.

[[Website]](https://developer.nvidia.com/isaac/gr00t) [[News]](https://research.nvidia.com/publication/2025-03_nvidia-isaac-gr00t-n1-open-foundation-model-humanoid-robots) [[Paper]](https://arxiv.org/abs/2503.14734)

For business inquiries, please submit this form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

Built upon the [RoboCasa](https://github.com/robocasa/robocasa) simulation framework, this repository is a fork that extends its modular infrastructure to support the GR00T-N1.5-3B model. We augment RoboCasa's core functionality with additional environments, assets, and tooling tailored for the GR-1 Tabletop tasks, enabling evaluation of generalist robotic policies in diverse household tasks.

-------

## Getting Started
RoboCasaGR00TEnvs works across all major computing platforms. The easiest way to set up is through the [Anaconda](https://www.anaconda.com/) package management system. Follow the instructions below to install all three required repositories, their dependencies, and download the assets needed for the simulation task:

```bash
# 1. Set up conda environment
conda create -c conda-forge -n robocasa python=3.10
conda activate robocasa

# 2. Clone and install Isaac-GR00T
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4 
cd ..

# 3. Clone and install robosuite
git clone https://github.com/ARISE-Initiative/robosuite.git
pip install -e robosuite

# 4. Clone and install robocasa-gr1-tabletop-tasks
git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git
pip install -e robocasa-gr1-tabletop-tasks

# 5. Download assets
cd robocasa-gr1-tabletop-tasks
python robocasa/scripts/download_tabletop_assets.py -y
```

## Demo All Tasks
Run the following command to demo any available task. It sends random actions to the robot and renders an egocentric video, offering a quick, visual understanding of each task:

```bash
cd robocasa-gr1-tabletop-tasks
python3 robocasa/scripts/demo_task.py <TASK_NAME>
```

A list of available task names can be found later in this README.


## Playback of Demonstration Trajectories
The [GR00T Teleop Simulation Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim) contains an existing collection of teleoperated demonstrations spanning all 24 tabletop tasks, with 1000 human-collected demos per task.

Run the following command to play back a demo:

```bash
cd robocasa-gr1-tabletop-tasks
python3 robocasa/scripts/playback_dataset.py --dataset <HDF5_FILE> --n 1
```

This script replays the demonstration trajectories in simulation, rendering the robot’s actions and camera observations in sync. It’s a useful tool for visually inspecting the dataset and verifying its structure before using it for training or benchmarking.

## Simulation-based Evaluation for GR00T-N1.5-3B

The main purpose of this repository is to evaluate the model in simulation to better understand its behavior in closed-loop settings. This is especially useful for assessing quantitative performance on long-horizon or multi-step tasks.

Please refer to https://github.com/NVIDIA/Isaac-GR00T to install Isaac-GR00T.

Inside the Isaac-GR00T repository, run the inference server:

```bash
cd Isaac-GR00T
python3 scripts/inference_service.py --server \
    --model_path <MODEL_PATH> \
    --data_config fourier_gr1_arms_waist
```

Inside the Isaac-GR00T repository, run the simulation evaluation script to evaluate a single task with 10 episodes.

```bash
cd Isaac-GR00T
python3 scripts/simulation_service.py --client \
    --env_name <TASK_NAME> \
    --video_dir ./videos \
    --max_episode_steps 720 \
    --n_envs 5 \
    --n_episodes 10
```

This script will run the model in a simulated environment for a given number of episodes, collect success metrics, and save rollout videos for inspection. It's a complementary method to the offline evaluation that gives insight into how the policy performs when interacting with the environment. Here is a full list of 24 tabletop task names:

```bash
gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
```

## Post-training with Isaac-GR00T

We are using the latest `GR00T-N1.5-3B` model from https://github.com/NVIDIA/Isaac-GR00T.git

This example finetunes a series of tasks from the [Humanoid robot tabletop manipulation: 240k trajectories dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim). This runs on a H100 Node with 8 GPUs. Generally the zero-shot performance is already decent with average success rate of 42%, with this postraining it can reach higher success rate of 47%.

```bash
#!/bin/bash
ALL_DATASET_PATHS=(
  "YOUR_DATASET_PATH_1"
  "YOUR_DATASET_PATH_2"
)

python scripts/gr00t_finetune.py \
  --dataset-path "${ALL_DATASET_PATHS[@]}" \
  --num-gpus 8 --batch-size 60 --learning_rate 3e-5 \
  --output-dir OUTPUT_DIR \
  --data-config fourier_gr1_arms_waist --embodiment_tag gr1 \
  --tune-visual \ # tune visual encoder, can be omitted if you don't want to tune the visual encoder
  --max-steps 30000 --save-steps 5000
```

## Citation
```bibtex
@misc{nvidia2025gr00tn1openfoundation,
      title={GR00T N1: An Open Foundation Model for Generalist Humanoid Robots}, 
      author={NVIDIA and Johan Bjorck and Fernando Castañeda and Nikita Cherniadev and Xingye Da and Runyu Ding and Linxi "Jim" Fan and Yu Fang and Dieter Fox and Fengyuan Hu and Spencer Huang and Joel Jang and Zhenyu Jiang and Jan Kautz and Kaushil Kundalia and Lawrence Lao and Zhiqi Li and Zongyu Lin and Kevin Lin and Guilin Liu and Edith Llontop and Loic Magne and Ajay Mandlekar and Avnish Narayan and Soroush Nasiriany and Scott Reed and You Liang Tan and Guanzhi Wang and Zu Wang and Jing Wang and Qi Wang and Jiannan Xiang and Yuqi Xie and Yinzhen Xu and Zhenjia Xu and Seonghyeon Ye and Zhiding Yu and Ao Zhang and Hao Zhang and Yizhou Zhao and Ruijie Zheng and Yuke Zhu},
      year={2025},
      eprint={2503.14734},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2503.14734}, 
}
```
