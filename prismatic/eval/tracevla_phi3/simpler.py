import json
import os
from typing import Any, Dict, List, Optional, Union
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np
import torch
from PIL import Image
from transforms3d.euler import euler2axangle
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoModelForCausalLM
from trace_processor import TraceProcessor

def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.
    Using tf resize corresponding to RLDS resize.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = Image.fromarray(img.numpy())
    return img

class Phi3TraceVLAInference:
    """
    For future developers:
    - Refer to BaseVectorPolicy for the API contract
    """

    def __init__(
        self,
        model_path,
        cotracker_model_path,
        dataset_stats_path,
        action_scale: float = 1.0,
        n_action_bins: int = 256,
        sample: bool = False,
        temperature: float = 0.0,
        image_aug: bool = False,
        model_dtype: str = None,  # in ["bfloat16", "float16", "float32", None]; None will read from the model config
        device: int = 0,
    ) -> None:
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            num_crops=1, 
        )

        self.vla = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            _attn_implementation='flash_attention_2',
            use_cache=True
        ).to(device=device)
        
        with open(dataset_stats_path, "r") as f:
            self.norm_stats = json.load(f)
            
        self.action_scale = action_scale
        self.sample = sample
        self.temperature = temperature
        self.image_aug = image_aug

        self.sticky_action_is_ons = [False]
        self.gripper_action_repeats = [0]
        self.sticky_gripper_actions = [0.0]
        self.previous_gripper_actions = [None]

    def start(self, num_envs: int):
        self.num_envs = num_envs

        # policy setups
        self.policy_setups = [None] * self.num_envs
        self.sticky_gripper_num_repeats = [0] * self.num_envs
        self.unnorm_keys = [None] * self.num_envs

        # sticky actions
        self.sticky_action_is_ons = [False] * self.num_envs
        self.gripper_action_repeats = [0] * self.num_envs
        self.sticky_gripper_actions = [0.0] * self.num_envs
        self.previous_gripper_actions = [None] * self.num_envs

        # task descriptions
        self.task_descriptions = [None] * self.num_envs

    def start_episode(
        self,
        obs_dicts: List[Dict],
        info_dicts: List[Dict],
        output_dirs: Optional[List[str]] = None,
        ids: Optional[Union[List[int], np.ndarray]] = None,
    ):
        self.frame_count = 0
        ids = self.unwrap_ids(ids)
        policy_setups = get_batch_from_info(info_dicts, "policy_setup")
        task_descriptions = get_batch_from_info(info_dicts, "task_description")

        self.trace_processors = []
        for i in range(len(policy_setups)):
            self.trace_processor = TraceProcessor(cotracker_model=self.cotracker_model_path, 
                                                  window_size=15 if policy_setups[i] == 'google_robot' else 10,
                                                  device=self.vla.device,
                                                 )
            self.trace_processor.reset()
            self.trace_processors.append(self.trace_processor)
        
        assert (
            len(obs_dicts) == len(info_dicts) == len(output_dirs) == len(ids)
        ), f"Mismatch in lengths: {len(obs_dicts)=}, {len(info_dicts)=}, {len(output_dirs)=}, {len(ids)=}"

        for i, (policy_setup, task_description) in enumerate(zip(policy_setups, task_descriptions)):
            idx = ids[i]
            self.policy_setups[idx] = policy_setup
            if policy_setup == "widowx_bridge":
                self.sticky_gripper_num_repeats[idx] = 1
                self.unnorm_keys[idx] = "bridge_orig"
            elif policy_setup == "google_robot":
                self.sticky_gripper_num_repeats[idx] = 15
                self.unnorm_keys[idx] = "fractal20220817_data"

            self.sticky_action_is_ons[idx] = False
            self.gripper_action_repeats[idx] = 0
            self.sticky_gripper_actions[idx] = 0.0
            self.previous_gripper_actions[idx] = None
            self.task_descriptions[idx] = task_description

    def reset_states_at(self, idx, task_description):
        assert idx < self.num_envs, f"{idx=}, {self.num_envs=}"
        self.sticky_action_is_ons[idx] = False
        self.gripper_action_repeats[idx] = 0
        self.sticky_gripper_actions[idx] = 0.0
        self.previous_gripper_actions[idx] = None
        self.task_descriptions[idx] = task_description
        self.trace_processors[idx].reset()

    @torch.no_grad()
    def get_action(
        self,
        obs_dicts: List[Dict],
        info_dicts: List[Dict],
        ids: Optional[Union[List[int], np.ndarray]] = None,
    ) -> List[Dict[str, Any]]:
        ids = self.unwrap_ids(ids)
        assert len(ids) == self.num_envs, f"Expected {self.num_envs} ids, but got {len(ids)}"

        images = get_batch_from_info(info_dicts, "image")
        task_descriptions = get_batch_from_info(info_dicts, "task_description")

        # Compare and reset task descriptions if needed
        for i, task_description in enumerate(task_descriptions):
            if task_description != self.task_descriptions[i]:
                self.reset_states_at(i, task_description)

        assert len(images) == self.num_envs, f"{len(images)=}, {self.num_envs=}"
        
        self.frame_count += 1
        raw_actions = []
        for i in range(self.num_envs):
            image = images[i]
            image = Image.fromarray(image)
            image = resize_image(image, (256,256))
            image_overlaid, has_trace = self.trace_processors[i].process_image(image)
            image = resize_image(image, (336,336))
            image_overlaid = resize_image(image_overlaid, (336,336))
            
            task_description = self.task_descriptions[i]
            if not has_trace:
                prompt_message = {
                'role': 'user',
                'content': f'<|image_1|><|image_2|>\nWhat action should the robot take to {task_description}?',
                }
            else:
                prompt_message = {
                    'role': 'user',
                    'content': f'You are given two images: one with the original robot observation <|image_1|>, and another one marked with historial traces of the robot end effector and moving objects <|image_2|>.\nWhat action should the robot take to {task_description}?',
                }
            prompt = self.processor.tokenizer.apply_chat_template(
                [prompt_message], tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(prompt, [image, image_overlaid]).to(device=self.vla.device, dtype=torch.bfloat16)
            
            if self.policy_setups[i] == "widowx_bridge":
                action_norm_stats = self.norm_stats['bridge_orig']['action'] 
            elif self.policy_setups[i] == "google_robot":
                action_norm_stats = self.norm_stats['fractal20220817_data']['action'] 
            else:
                assert False
            self.vla.prepare_action_inference(action_norm_stats, self.processor.tokenizer.vocab_size)
            
            with torch.inference_mode():
                raw_action = self.vla.predict_action(**inputs)
                raw_actions.append(raw_action)
        
        
        actions_list = []
        for i in range(self.num_envs):
            actions = raw_actions[i]
            raw_action = {
                "world_vector": np.array(actions[:3]),
                "rotation_delta": np.array(actions[3:6]),
                "open_gripper": np.array(actions[6:7]),
            }
            action = {}
            action["world_vector"] = raw_action["world_vector"] * self.action_scale
            action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
            roll, pitch, yaw = action_rotation_delta
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action_rotation_axangle = action_rotation_ax * action_rotation_angle
            action["rot_axangle"] = action_rotation_axangle * self.action_scale

            if self.policy_setups[i] == "widowx_bridge":
                action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            elif self.policy_setups[i] == "google_robot":
                current_gripper_action = raw_action["open_gripper"]
                if self.previous_gripper_actions[i] is None:
                    relative_gripper_action = np.array([0])
                else:
                    relative_gripper_action = (
                        self.previous_gripper_actions[i] - current_gripper_action
                    )
                self.previous_gripper_actions[i] = current_gripper_action

                if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_ons[i]):
                    self.sticky_action_is_ons[i] = True
                    self.sticky_gripper_actions[i] = relative_gripper_action

                if self.sticky_action_is_ons[i]:
                    self.gripper_action_repeats[i] += 1
                    relative_gripper_action = self.sticky_gripper_actions[i]

                if self.gripper_action_repeats[i] == self.sticky_gripper_num_repeats[i]:
                    self.sticky_action_is_ons[i] = False
                    self.gripper_action_repeats[i] = 0
                    self.sticky_gripper_actions[i] = 0.0

                action["gripper"] = relative_gripper_action

            action["terminate_episode"] = np.array([0.0])

            actions_list.append(action)

        return actions_list
