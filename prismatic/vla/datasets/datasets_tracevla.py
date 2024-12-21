"""
datasets_trace.py

Implement data processing for TraceVLA
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import AutoProcessor

from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class TraceVLARLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    processor: AutoProcessor
    image_transform: ImageTransform
    history_window_size: int = 1
    random_drop_rate: float = 0.2

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        processor = self.processor
        tokenizer = self.processor.tokenizer
        
        traced = rlds_batch['observation']['traced'][0]
        if not traced or np.random.uniform() < self.random_drop_rate:
            images = [rlds_batch["observation"]["image_primary"][0], rlds_batch["observation"]["image_primary"][0]]
            images = [Image.fromarray(image) for image in images]
            prompt_message = {
            'role': 'user',
            'content': f'<|image_1|><|image_2|>\nWhat action should the robot take to {lang}?',
            }
        else:
            images = [rlds_batch["observation"]["image_primary"][0], rlds_batch["observation"]["image_secondary"][0]]
            images = [Image.fromarray(image) for image in images]
            ### For now, assume that number of images here is 1
            prompt_message = {
                'role': 'user',
                'content': f'You are given two images: one with the original robot observation <|image_1|>, and another one marked with historial traces of the robot end effector and moving objects <|image_2|>.\nWhat action should the robot take to {lang}?',
            }
        prompt = tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )
        processor_result = processor(prompt, images, return_tensors='pt')
        prompt_input_ids = processor_result['input_ids'].squeeze(0)


        action_tokenized = self.action_tokenizer(action)
        action_tokenized += '<|endoftext|>'
        ### the first id is '', remove it.
        action_ids = torch.tensor(list(tokenizer(action_tokenized, add_special_tokens=False).input_ids))[1:]
        input_ids = torch.cat([prompt_input_ids, action_ids], dim=0)
        ignore_index = -100
        labels = torch.cat(
            [
                torch.tensor([ignore_index] * prompt_input_ids.shape[0]),
                action_ids,
            ],
            dim=0,
        )
        image_sizes=torch.tensor([[336,336], [336,336]])
        return dict(pixel_values=processor_result['pixel_values'], input_ids=input_ids, labels=labels, image_sizes=image_sizes) 
    
class TraceVLARLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: TraceVLARLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        history_window_size: int = 1,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform


        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]
            

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary", "secondary"),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=history_window_size,                    # If we wanted to feed / predict more than one step
                future_action_window_size=0,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")