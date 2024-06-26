#!/bin/bash

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

export PYTHONPATH=habitat-sim/src_python

DATA_PATH="data/dataset/goat-bench/data/datasets/goat_bench/hm3d/v1"
# eval_ckpt_path_dir="data/new_checkpoints/goat/ver/resnetclip_rgb_multimodal/seed_1/"
# tensorboard_dir="tb/goat/ver/resnetclip_rgb_multimodal/seed_1/val_seen/"
split="val_seen"

export OVON_IL_DONT_CHEAT=1

python -um goat_bench.run \
  --run-type eval \
  --exp-config config/experiments/ver_goat.yaml \
  habitat_baselines.num_environments=1 \
  habitat_baselines.trainer_name="goat_ppo" \
  habitat_baselines.video_dir="${tensorboard_dir}/videos" \
  habitat_baselines.rl.policy.name=PointNavResnetCLIPPolicy \
  habitat_baselines.rl.ddppo.train_encoder=False \
  habitat_baselines.rl.ddppo.backbone=resnet50_clip_avgpool \
  habitat_baselines.tensorboard_dir=$tensorboard_dir \
  habitat_baselines.eval_ckpt_path_dir=$eval_ckpt_path_dir \
  habitat.dataset.data_path="${DATA_PATH}/${split}/${split}.json.gz" \
  +habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map \
  +habitat/task/lab_sensors@habitat.task.lab_sensors.goat_goal_sensor=goat_goal_sensor \
  ~habitat.task.lab_sensors.objectgoal_sensor \
  habitat.task.measurements.top_down_map.type=GoatTopDownMap \
  habitat.task.measurements.top_down_map.draw_shortest_path=False \
  habitat.task.measurements.top_down_map.max_episode_steps=10000 \
  habitat.task.lab_sensors.goat_goal_sensor.object_cache=data/clip_embeddings/ovon_stretch_final_cache.pkl \
  habitat.task.lab_sensors.goat_goal_sensor.image_cache=data/datasets/iin/hm3d/v2/${split}_embeddings/ \
  habitat.task.lab_sensors.goat_goal_sensor.image_cache_encoder="CLIP_goat" \
  habitat.task.lab_sensors.goat_goal_sensor.language_cache="data/datasets/languagenav/hm3d/v5_final/embeddings/${split}_clip_embedding.pkl" \
  habitat.task.measurements.success.success_distance=0.25 \
  habitat.dataset.type="Goat-v1" \
  habitat.task.measurements.distance_to_goal.type=GoatDistanceToGoal \
  habitat.task.measurements.success.type=GoatSuccess \
  habitat.task.measurements.spl.type=GoatSPL \
  habitat.task.measurements.soft_spl.type=GoatSoftSPL \
  +habitat/task/measurements@habitat.task.measurements.goat_distance_to_goal_reward=goat_distance_to_goal_reward \
  ~habitat.task.measurements.distance_to_goal_reward \
  habitat.simulator.type="GOATSim-v0" \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.eval.use_ckpt_config=False \
  habitat_baselines.eval.split=$split \

touch $checkpoint_counter