#!/usr/bin/env python3

import argparse
import numpy as np
import cv2
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs
import threading
from typing import Tuple
import time
from pathlib import Path
import io
import torch
from collections import deque
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import json
import time
from PIL import Image, ImageDraw
import torch
from collections import deque
import matplotlib.pyplot as plt
import os

def show_video(client, full_image=True):
    """
    This shows the video from the camera for a given duration.
    Full image is the image before resized to default 256x256.
    """
    res = client.get_observation()
    if res is None:
        print("No observation available... waiting")
        return None, None, None
    state = res['state']
    if full_image:
        org_img = res["full_image"][0]
        img = cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)
    else:
        org_img = res["image"]
        img = (org_img.reshape(3, 256, 256).transpose(1, 2, 0) * 255).astype(np.uint8)
    return org_img, img, state



def main():
    parser = argparse.ArgumentParser(description='Evaluation for WidowX Robot')
    parser.add_argument('--model_dir', type=str, default="../../phi3v_finetune_entire")
    parser.add_argument('--ip', type=str, default='10.137.68.110')
    parser.add_argument('--dataset_name', type=str, default='ours20_dataset')
    parser.add_argument('--port', type=int, default=14984)
    parser.add_argument('--e', type=int, default=1)    
    args = parser.parse_args()

    client = WidowXClient(host=args.ip, port=args.port)
    client.init(WidowXConfigs.DefaultEnvParams, image_size=256)


    # Load Processor & Phi3
    processor = AutoProcessor.from_pretrained(args.model_dir,
    trust_remote_code=True, num_crops=1)
    vla = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            _attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=True
    )

    INSTRUCTION = ### Put language instruction here
    phi3_prompt_message = {
            'role': 'user',
            'content': f'<|image_1|>\nWhat action should the robot take to {INSTRUCTION}?',
        }

    image_list = []
    step, running = 0, True
    dataset_stats_dir = args.model_dir + "/dataset_statistics.json"
    with open(dataset_stats_dir, 'r') as file: # Directory to dataset_statistics.json
        action_norm_stats = json.load(file)['our_dataset']['action']

    while running:  
        is_open = True
        with np.printoptions(precision=3, suppress=True):
            image = None
            while image is None: 
                image, _, _= show_video(client)
            img = Image.fromarray(np.uint8(image)).resize((336,336))

            phi3_prompt = processor.tokenizer.apply_chat_template(
                [phi3_prompt_message], tokenize=False, add_generation_prompt=True
            )

            image_list.append(img)
            inputs = processor(phi3_prompt, [img]).to("cuda", dtype=torch.bfloat16)
            vla.prepare_action_inference(action_norm_stats, processor.tokenizer.vocab_size)

            # Predict Action (7-DoF; un-normalize for BridgeData V2)
            start_time = time.time()
            actions = vla.predict_action(**inputs)
            end_time = time.time()
            print(actions)
            print(f'Get environment observation of step:{step}, time taken:{end_time-start_time}s')
            step += 1
            if is_open and actions[-1] < 0.05:
                is_open = False
                grip = True
            elif not is_open and actions[-1] > 0.95:
                is_open = True
                grip = True
            else:
                grip = False

            client.step_action(actions)
            if grip:
                time.sleep(0.1)

    os.makedirs('phi3_images', exist_ok=True)
    for i, img in enumerate(image_list):
        img.save(f'phi3_images/step_{i}.png')
    print(f"Saved {len(image_list)} images in phi3_images")

    client.stop()  # Properly stop the client
if __name__ == "__main__":
    main()