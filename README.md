# TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for Generalist Robotic Policies

[![arXiv](https://img.shields.io/badge/arXiv-2412.10345-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2412.10345)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/openvla/openvla-7b)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)
 
[**Getting Started**](#getting-started) | [**Installation**](#installation) | [**Project Website**](https://tracevla.github.io/)


<hr style="border: 2px solid gray;"></hr>

## Latest Updates
- [2025-01-06] Initial release

<hr style="border: 2px solid gray;"></hr>

This branch provides the code to train/finetune a pretrained Phi3V-based VLA model with visual trace prompting technique. It is built on top of [OpenVLA](https://tracevla.github.io/)


## Installation

Use the setup commands below to get started:

```bash
# Create and activate conda environment
conda create -n tracevla_phi3 python=3.10 -y
conda activate tracevla_phi3

# Clone and install the tracevla repo
git clone https://github.com/openvla/tracevla.git
cd tracevla
git checkout phi3
pip install -e .
# Install Flash Attention 2 for training 
pip install "flash-attn==2.5.5" --no-build-isolation
```
If you run into any problems during the installation process, please file a GitHub Issue.

## Zero-shot model inference of pretrained checkpoint

To load a pretrained ``openvla_phi3v`` or ``tracevla_phi3v`` model for zero-shot instruction following:

```
# Load Processor & VLA
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
    num_crops=1, 
)

vla = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    _attn_implementation='flash_attention_2',
    use_cache=True
).to(device=device)

# Load Processor & VLA
with open(dataset_stats_path, "r") as f:
    self.norm_stats = json.load(f)
    vla.prepare_action_inference(action_norm_stats, self.processor.tokenizer.vocab_size)

# Grab image input & format prompt
image: Image.Image = get_from_camera(...)
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

# Predict the action (7-DoF; un-normalize for BridgeData V2)
prompt_message = {
    'role': 'user',
    'content': f'<|image_1|>\nWhat action should the robot take to {task_description}?',
}
prompt = processor.tokenizer.apply_chat_template(
    [prompt_message], tokenize=False, add_generation_prompt=True
)
inputs = self.processor(prompt, [image]).to("cuda:0", dtype=torch.bfloat16)
with torch.inference_mode():
    action = vla.predict_action(**inputs)

# Execute the action
robot.act(action, ...)
```
Additionally, we have also provided the implementation of both real Widowx arm and SimplerEnv policy wrappers under ``eval``.

## TraceVLA data downloading
[Upcoming] We will be releasing our visual trace annotated data on Bridge and Fractal dataset soon. Stay tuned for the update.

## Model finetuning

We provide a training script to finetune your model with TraceVLA format, please checkout ``vla-scripts/train.sh``.

## Citation

If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/abs/2412.10345):

```bibtex
@misc{zheng2024tracevlavisualtraceprompting,
      title={TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for Generalist Robotic Policies}, 
      author={Ruijie Zheng and Yongyuan Liang and Shuaiyi Huang and Jianfeng Gao and Hal Daumé III and Andrey Kolobov and Furong Huang and Jianwei Yang},
      year={2024},
      eprint={2412.10345},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2412.10345}, 
}
```