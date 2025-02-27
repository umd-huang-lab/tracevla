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

This branch provides the official implementation of TraceVLA to finetune a pretrained OpenVLA model with visual trace prompting technique. It is built on top of the original [OpenVLA](https://openvla.github.io/) codebase. For Phi3V based OpenVLA/TraceVLA model, please checkout ``phi3`` branch.


## Installation

Use the setup commands below to get started:

```bash
# Create and activate conda environment
conda create -n tracevla python=3.10 -y
conda activate tracevla

# Install depdencies of Co-Tracker:
git clone https://github.com/facebookresearch/co-tracker.git
cd co-tracker
pip install -e .

# Download Co-Tracker checkpoint
mkdir -p checkpoints
cd checkpoints
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
cd ..

# Clone and install the tracevla repo
git clone https://github.com/openvla/tracevla.git
cd tracevla
pip install -e .
# Install Flash Attention 2 for training 
pip install "flash-attn==2.5.5" --no-build-isolation
```
If you run into any problems during the installation process, please file a GitHub Issue.

## Zero-shot model inference of pretrained checkpoint

We have also provided the implementation of SimplerEnv policy wrapper of ``tracevla`` under ``prismatic/eval``. 
In particular, to load pretrained ``tracevla`` model for zero-shot instruction following:

```
model_path = "furonghuang-lab/tracevla_7b" 
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
).to(device='cuda')

# Load Visual Trace Processor
# cotracker_model_path corresponds to the path to your downloaded scaled_offline.pth checkpoint
from prismatic.eval.trace_processor import TraceProcessor
trace_processor = TraceProcessor(cotracker_model_path)

# Grab image input & format prompt
# In case where the visual trace returned by Co-Tracker is not valid, we use the default openvla prompt.
openvla_prompt_template = "In: What action should the robot take to {task_description}?\nOut:"
tracevla_prompt_template = "In: You are given two images: one with the original robot observation, and another one marked with historical traces of the robot end effector and moving objects, separated by a special separator token. What action should the robot take to {task_description}?\nOut:"

image: Image.Image = get_from_camera(...)
image_overlaid, has_trace = trace_processors.process_image(image)

if not has_trace:
    prompt = openvla_prompt_template.format(task_description=task_description)
    inputs = processor(prompt, [image, image]).to(device='cuda', dtype=torch.bfloat16)
else:
    prompt = tracevla_prompt_template.format(task_description=task_description)
    inputs = processor(prompt, [image, image_overlaid]).to(device='cuda', dtype=torch.bfloat16)

### Predict the action
with torch.inference_mode():
    action = vla.predict_action(**inputs)

# Execute the action
robot.act(action, ...)
```

## TraceVLA data downloading
[Coming soon] We will soon be releasing our visual trace annotated data on Bridge and Fractal dataset. Stay tuned for the update.

## Model finetuning

Under ``scripts/train.sh``, we provide a training script to reproduce the training run with TraceVLA format:
```
torchrun --nnodes 8 --nproc-per-node 8 \
         --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK \
              vla-scripts/train.py \
              --pretrained_checkpoint $PRETRAINED_CKPT_DIR \
              --vla.type 64gpu-tracevla \
              --data_root_dir $DATA_DIR \
              --run_root_dir $OUTPUT_DIR \
              --run_id $RUN_NAME \
              --wandb_project $WANDB_PROJECT \
              --wandb_entity $WANDB_ENTITY \
              --hf_token $HF_TOKEN

```

## Evaluation Results on SimplerEnv Fractal + SimplerEnv:

### Fractal:
| Policy/Settings | Pick up Coke | Move near | Open/Close Drawer | Put in Drawer | Average Success Rate |
|:------:|:------------:|:---------:|:------------:|:-----------:|:-------:|
| (Visual Matching) OpenVLA-7B | 23.7% | **65.0%** | 57.4% | 0.% | 36.5% |
| (Visual Matching) TraceVLA-7B | **45.0%** | 63.8% | **63.1%** | **11.1.%** | 45.8% |
| (Variant Aggregation) OpenVLA-7B | 61.3% | 55.8% | 24.9% | 1.0% | 35.8% |
| (Variant Aggregation) TraceVLA-7B | **64.3%** | **60.6%** | **61.6%** | **12.5.%** | **49.8%** |

### Bridge:
| Policy/Settings | Put Spoon | Put Carrot | Stack Block | Put Eggplant | Average Success Rate |
|:------:|:------------:|:---------:|:------------:|:-----------:|:-------:|
| OpenVLA-7B | 8.3% | **8.3%** | **4.2%** | **45.8%** | **16.7%** |
| TraceVLA-7B | 12.5% | **16.6%** | **16.6%** | **65.0%** | **27.7%** |


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