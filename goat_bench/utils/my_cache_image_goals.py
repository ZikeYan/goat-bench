import argparse
import os
import torch
from habitat.config import read_write
from habitat_baselines.config.default import get_config

from goat_bench.utils.utils import write_json
import habitat
from habitat.tasks.nav.instance_image_nav_task import InstanceImageParameters
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration



class ImageCaptioner:
    def __init__(self, use_blip2=True):
        """
        Initialize the image captioner with the specified model.
        """
        self.use_blip2 = use_blip2
        if not use_blip2:
            model_name="Salesforce/blip-image-captioning-base"
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        else:
            model_name="Salesforce/blip2-opt-2.7b"
            self.processor = Blip2Processor.from_pretrained(model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
            self.prompt = "The main object in scene is"
    def generate_caption(self, image):
        """
        Generate a caption for the given image.

        Args:
        image_path (str): The path to the image file.

        Returns:
        str: The generated caption.
        """
        # Load and preprocess the image
        # image = Image.open(image_path).convert("RGB")
        if self.use_blip2:
            inputs = self.processor(image, self.prompt, return_tensors="pt").to("cuda")
            out = self.model.generate(**inputs, max_length=50)
            caption = self.processor.decode(out[0], skip_special_tokens=True).strip()
        else:
            inputs = self.processor(images=image, return_tensors="pt")
            # Generate caption
            out = self.model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

class CacheGoals:
    def __init__(
        self,
        config_path: str,
        output_path: str = "",
    ) -> None:
        self.device = torch.device("cuda")

        self.config_path = config_path
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.captioner = ImageCaptioner()
        # self.env = Env(get_config(self.config_path))
        
    def config_env(self, split):
        config = get_config(self.config_path)
        with read_write(config):
            config.habitat.dataset.data_path = config.habitat.dataset.data_path.replace("val_seen", split)
            config.habitat.task.lab_sensors.goat_goal_sensor.language_parse_cache = config.habitat.task.lab_sensors.goat_goal_sensor.language_parse_cache.replace("val_seen", split)
            config.habitat.dataset.split = split
        env = habitat.Env(config=config)
        return env

    def run(self, split):
        
        env = self.config_env(split)
        env.reset()
        episodes = env._dataset.episodes
        data_goal = {}
        for ep in episodes:
            # ep.episode_id
            scene_name = ep.scene_id.split('/')[-1].split('.')[0]
            if ep.scene_id != env.current_episode.scene_id:
                env.current_episode = ep
                env.reset()
                assert env.current_episode.scene_id == ep.scene_id
            for idx, task in enumerate(ep.tasks):
                # imagegoal
                if task[1]!='image':
                    continue
                object_id = task[2]
                pos_id = task[3]
                goal_param = ep.goals[idx][0]['image_goals'][pos_id]
                img_param = InstanceImageParameters(**goal_param)
                img = env.task.sensor_suite.sensors[
                    "goat_subtask_goal"
                ]._get_instance_image_goal(img_param)
                image = Image.fromarray(img)
                caption = self.captioner.generate_caption(image)
                metadata = dict(
                    hfov=img_param.hfov,
                    object_id=object_id,
                    position=img_param.position,
                    rotation=img_param.rotation,
                    image_caption=caption
                )
                out_fname = f"{scene_name}/{ep.episode_id}_{object_id}_{pos_id}"
                print(out_fname, caption)
                data_goal[out_fname] = metadata
                image_path = self.output_path + f'raw_image/{split}/' + out_fname + ".png"
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                image.save(image_path)
        write_json(data_goal, os.path.join(self.output_path, f"{split}_image_goal.json"))
    
    def run_cached(self, split):
        data_goal = {}
        
        # Define the directory containing the cached images
        cache_dir = os.path.join(self.output_path, f'raw_image/{split}/')
        
        for root, _, files in os.walk(cache_dir):
            for file in files:
                if file.endswith(".png"):
                    # Extract metadata from the file name
                    out_fname = os.path.join(root, file)
                    scene_name, episode_id, object_id, pos_id = file.replace('.png', '').split('_')
                    
                    # Load the cached image
                    image = Image.open(out_fname)
                    
                    # Generate caption from the cached image
                    caption = self.captioner.generate_caption(image)
                    
                    metadata = dict(
                        object_id=object_id,
                        image_caption=caption
                    )
                    
                    print(out_fname, caption)
                    data_goal[f"{scene_name}/{file.replace('.png', '')}"] = metadata
        
        # Write the metadata to a JSON file
        write_json(data_goal, os.path.join(self.output_path, f"image_goal_vqa_blip2/{split}_image_goal_blip2.json"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="goat-bench/config/tasks/instance_imagenav_stretch_hm3d.yaml",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
    )
    # parser.add_argument(
    #     "--split",
    #     type=str,
    #     default="",
    # )
    args = parser.parse_args()

    cache = CacheGoals(
        config_path=args.config,
        output_path=args.output_path,
    )
    for split in ["val_seen", "val_unseen", "val_seen_synonyms"]:
        cache.run_cached(split)
        # cache.run(split)
    # cache.run(args.split)
