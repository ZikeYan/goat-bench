#!/bin/bash
#SBATCH --job-name=goat
#SBATCH --output=slurm_logs/goat-ver-%j.out
#SBATCH --error=slurm_logs/goat-ver-%j.err
#SBATCH --gpus 4
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 4
#SBATCH --constraint=a40
#SBATCH --exclude=gundam,xaea-12,baymax
#SBATCH --partition=short
#SBATCH --signal=USR1@100
#SBATCH --requeue

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

source /srv/flash1/rramrakhya3/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate goat

export PYTHONPATH=/srv/flash1/rramrakhya3/fall_2023/habitat-sim/src_python/

TENSORBOARD_DIR="tb/goat/ver/resnetclip_rgb_multimodal/seed_2_v0.1.4/"
CHECKPOINT_DIR="data/new_checkpoints/goat/ver/resnetclip_rgb_multimodal/seed_2_v0.1.4/"
DATA_PATH="data/datasets/goat/v0.1.4/"

srun python -um goat.run \
  --run-type train \
  --exp-config config/experiments/ver_goat.yaml \
  habitat_baselines.trainer_name="ver" \
  habitat_baselines.num_environments=32 \
  habitat_baselines.rl.policy.name=PointNavResnetCLIPPolicy \
  habitat_baselines.rl.ddppo.train_encoder=False \
  habitat_baselines.rl.ddppo.backbone=resnet50_clip_avgpool \
  habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
  habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
  habitat.dataset.data_path=${DATA_PATH}/train/train.json.gz \
  +habitat/task/lab_sensors@habitat.task.lab_sensors.goat_goal_sensor=goat_goal_sensor \
  ~habitat.task.lab_sensors.objectgoal_sensor \
  habitat.task.lab_sensors.goat_goal_sensor.object_cache=data/clip_embeddings/ovon_stretch_final_cache.pkl \
  habitat.task.lab_sensors.goat_goal_sensor.image_cache=data/datasets/iin/hm3d/v2/train_goal_embeddings/ \
  habitat.task.lab_sensors.goat_goal_sensor.image_cache_encoder="CLIP_goat" \
  habitat.task.lab_sensors.goat_goal_sensor.language_cache="data/datasets/languagenav/hm3d/v5_final/train_embeddings/clip_train_embeddings.pkl" \
  habitat.task.measurements.success.success_distance=0.25 \
  habitat.dataset.type="Goat-v1" \
  habitat.task.measurements.distance_to_goal.type=GoatDistanceToGoal \
  habitat.task.measurements.success.type=GoatSuccess \
  habitat.task.measurements.spl.type=GoatSPL \
  habitat.task.measurements.soft_spl.type=GoatSoftSPL \
  +habitat/task/measurements@habitat.task.measurements.goat_distance_to_goal_reward=goat_distance_to_goal_reward \
  ~habitat.task.measurements.distance_to_goal_reward \
  habitat.simulator.type="GOATSim-v0" \
  habitat_baselines.rl.ddppo.distrib_backend="GLOO"
