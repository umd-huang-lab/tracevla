import argparse
import os
import json
import time
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs

def show_video(client, full_image: bool = True):
    """
    Shows the video from the camera for a given duration.
    
    Args:
        client (WidowXClient): The robot client for controlling WidowX.
        full_image (bool): If True, returns the original un-resized image from the environment.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, dict]:
            - The original image (org_img).
            - The BGR or resized image (img).
            - The state dictionary from the environment observation.
            If no observation is available, returns (None, None, None).
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
        org_img = res["image"]  # shape should be (3, 256, 256)
        img = (org_img.reshape(3, 256, 256).transpose(1, 2, 0) * 255).astype(np.uint8)

    return org_img, img, state


def main():
    parser = argparse.ArgumentParser(description='Evaluation for WidowX Robot')
    parser.add_argument('--model_dir', type=str, default="../../phi3v_finetune_entire", 
                        help='Path to the directory containing model and tokenizer.')
    parser.add_argument('--ip', type=str, default='10.137.68.110', 
                        help='IP address of the WidowX robot.')
    parser.add_argument('--dataset_name', type=str, default='ours20_dataset', 
                        help='Name of the dataset used for normalization statistics.')
    parser.add_argument('--port', type=int, default=14984, 
                        help='Port to connect to the WidowX robot.')
    parser.add_argument('--e', type=int, default=1, 
                        help='Number of evaluation episodes (unused in this script).')
    args = parser.parse_args()

    # Initialize the WidowX client
    client = WidowXClient(host=args.ip, port=args.port)
    client.init(WidowXConfigs.DefaultEnvParams, image_size=256)

    # Load Processor & Model
    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        num_crops=1
    )
    vla = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        _attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=True
    )

    # Placeholder for your instruction
    INSTRUCTION = "### Put language instruction here"

    # Create the prompt message
    phi3_prompt_message = {
        'role': 'user',
        'content': f'<|image_1|>\nWhat action should the robot take to {INSTRUCTION}?'
    }

    # Load dataset statistics for action normalization
    dataset_stats_path = os.path.join(args.model_dir, "dataset_statistics.json")
    with open(dataset_stats_path, 'r') as file:
        action_norm_stats = json.load(file)[args.dataset_name]['action']

    image_list = []
    step = 0
    running = True
    is_open = True  # Tracks whether the gripper is currently open

    while running:
        with np.printoptions(precision=3, suppress=True):
            # Keep trying to retrieve a valid frame
            org_img, _, _ = None, None, None
            while org_img is None:
                org_img, _, _ = show_video(client)
            # Resize image for the model input
            img_pil = Image.fromarray(np.uint8(org_img)).resize((336, 336))

            # Prepare the prompt for the model
            phi3_prompt = processor.tokenizer.apply_chat_template(
                [phi3_prompt_message],
                tokenize=False,
                add_generation_prompt=True
            )

            # Keep track of images for optional debugging/visualization
            image_list.append(img_pil)

            # Preprocess inputs for the model
            inputs = processor(phi3_prompt, [img_pil]).to("cuda", dtype=torch.bfloat16)
            vla.prepare_action_inference(action_norm_stats, processor.tokenizer.vocab_size)

            # Predict action (7-DoF)
            start_time = time.time()
            actions = vla.predict_action(**inputs)
            end_time = time.time()

            print(actions)
            print(f"Step {step}: inference took {end_time - start_time:.3f} seconds")

            # Increment step
            step += 1

            # Simple example for deciding whether to open or close the gripper
            if is_open and actions[-1] < 0.05:
                is_open = False
                grip = True
            elif not is_open and actions[-1] > 0.95:
                is_open = True
                grip = True
            else:
                grip = False

            # Send the action to the robot
            client.step_action(actions)

            # Small pause if the gripper state changed
            if grip:
                time.sleep(0.1)

    # Save all recorded images
    os.makedirs('phi3_images', exist_ok=True)
    for i, im in enumerate(image_list):
        im.save(os.path.join('phi3_images', f'step_{i}.png'))
    print(f"Saved {len(image_list)} images to 'phi3_images/'")

    # Properly stop the client
    client.stop()


if __name__ == "__main__":
    main()
