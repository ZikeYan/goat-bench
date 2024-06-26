#!/bin/bash
#SBATCH --job-name=goat
#SBATCH --output=slurm_logs/eval/goat-%j.out
#SBATCH --error=slurm_logs/eval/goat-%j.err
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --constraint="a40|rtx_6000|2080_ti"
#SBATCH --partition=short
#SBATCH --exclude=xaea-12
#SBATCH --signal=USR1@100

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

source /srv/flash1/rramrakhya3/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate goat

export PYTHONPATH=/srv/flash1/rramrakhya3/fall_2023/habitat-sim/src_python/

DATA_PATH="data/datasets/languagenav/hm3d/v5_final/"
# eval_ckpt_path_dir="data/new_checkpoints/languagenav/ver/resnetclip_rgb_bert_text/seed_3/"
# tensorboard_dir="tb/languagenav/ver/resnetclip_rgb_bert_text/seed_3/val_unseen_easy/"
# split="val_unseen_easy"

echo "Evaluating ckpt: ${eval_ckpt_path_dir}"
echo "Data path: ${DATA_PATH}/${split}/${split}.json.gz"

srun python -um goat.run \
  --run-type eval \
  --exp-config config/experiments/ver_language_nav.yaml \
  habitat_baselines.num_environments=20 \
  habitat_baselines.rl.policy.name=PointNavResnetCLIPPolicy \
  habitat_baselines.tensorboard_dir=$tensorboard_dir \
  habitat_baselines.eval_ckpt_path_dir=$eval_ckpt_path_dir \
  habitat_baselines.checkpoint_folder=$eval_ckpt_path_dir \
  habitat.dataset.data_path="${DATA_PATH}/${split}/${split}.json.gz" \
  +habitat/task/lab_sensors@habitat.task.lab_sensors.language_goal_sensor=language_goal_sensor \
  ~habitat.task.lab_sensors.objectgoal_sensor \
  habitat.task.lab_sensors.language_goal_sensor.cache=data/datasets/languagenav/hm3d/v5_final/embeddings/${split}_clip_embedding.pkl \
  habitat.task.lab_sensors.language_goal_sensor.embedding_dim=1024 \
  habitat.task.measurements.success.success_distance=0.25 \
  habitat.dataset.type="LanguageNav-v1" \
  habitat.simulator.type="GOATSim-v0" \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.eval.use_ckpt_config=False \
  habitat_baselines.eval.split=$split

touch $checkpoint_counter
