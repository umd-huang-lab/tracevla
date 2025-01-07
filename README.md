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

This branch provides the code to train/finetune a pretrained Phi3V-based VLA model with or without visual trace prompting technique. It is built on top of the original [OpenVLA](https://openvla.github.io/) codebase.


## Installation

Use the setup commands below to get started:

```bash
# Create and activate conda environment
conda create -n tracevla_phi3 python=3.10 -y
conda activate tracevla_phi3

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
git checkout phi3
pip install -e .
# Install Flash Attention 2 for training 
pip install "flash-attn==2.5.5" --no-build-isolation
```
If you run into any problems during the installation process, please file a GitHub Issue.

## Zero-shot model inference of pretrained checkpoint

We have also provided the implementation of SimplerEnv policy wrapper of both ``openvla_phi3`` and ``tracevla_phi3`` under ``prismatic/eval``. 
In particular, to load pretrained ``openvla_phi3v`` model for zero-shot instruction following:

```
model_path = "furonghuang-lab/openvla_phi3v" 
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

# Load dataset statistics 
# For dataset inside OXE split, you can find the ``dataset_statistics.json`` file under the model repo.
# For your custom dataset split, it will be automatically generated under the model checkpoint path after training on your custom data split.
with open(dataset_stats_path, "r") as f:
    self.norm_stats = json.load(f)
    vla.prepare_action_inference(action_norm_stats, processor.tokenizer.vocab_size)

# Grab image input & format prompt
image: Image.Image = get_from_camera(...)
image = resize_image(image, (336,336))
prompt_message = {
    'role': 'user',
    'content': f'<|image_1|>\nWhat action should the robot take to {task_description}?',
}

### Process the prompt & image
prompt = processor.tokenizer.apply_chat_template(
    [prompt_message], tokenize=False, add_generation_prompt=True
)
inputs = self.processor(prompt, [image]).to("cuda:0", dtype=torch.bfloat16)

### Predict the action
with torch.inference_mode():
    action = vla.predict_action(**inputs)

# Execute the action
robot.act(action, ...)
```
For ``tracevla_phi3v`` model, to instantiate the model & processor, set ``model_path="furonghuang-lab/tracevla_phi3v"``. Additionally, you also need to instantiate the visual trace processor:
```
from prismatic.eval import TraceProcessor
trace_processor = TraceProcessor(cotracker_model_path)
```
Where ``cotracker_model_path`` is the path of the downloaded cotracker checkpoint ``scaled_offline.pth``.

For processing the prompt & image, please use the following prompt template instead:
```
### Get visual trace overlaid image observation
image = resize_image(image, (256,256)) ### 256x256 is the resolution of Co-Tracker Input Resolution
image_overlaid, has_trace = self.trace_processors[i].process_image(image) 
image_overlaid = resize_image(image_overlaid, (336,336)) ### 336x336 is the resolution of Phi3V image encoder.

### Prepare TraceVLA prompt format
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
prompt = processor.tokenizer.apply_chat_template(
    [prompt_message], tokenize=False, add_generation_prompt=True
)

inputs = processor(prompt, [image, image_overlaid]).to("cuda:0", dtype=torch.bfloat16)
```

## TraceVLA data downloading
[Coming soon] We will soon be releasing our visual trace annotated data on Bridge and Fractal dataset. Stay tuned for the update.

## Model finetuning

Under ``vla-scripts/train.sh``, we provide a training script to finetune your model with TraceVLA format:
```
torchrun --nnodes 4 --nproc-per-node 8 \
            --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK \
            train_phi3.py \
            --tracevla \
            --model_name_or_path $MODEL_PATH \
            --data_mix bridge_orig \
            --output_dir $OUTPUT_DIR \
            --batch_size 1920 \
            --per_device_batch_size 60 \
            --data_root_dir $DATA_DIR \
            --shuffle_buffer_size 100000 \
            --learning_rate 1e-5 \
            --run_name $RUN_NAME \
            --num_train_epochs 20 \
            --bf16  \
            --use_flash_attention
```
In case if you only want to finetune the model with the original OpenVLA format, simply remove the ``--tracevla`` flag.

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