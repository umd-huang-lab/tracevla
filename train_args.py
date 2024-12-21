import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='microsoft/Phi-3-vision-128k-instruct',
        help='Model name or path to load from',
    )
    parser.add_argument(
        '--state_dict_path',
        type=str,
        default=None,
        help='path pf state_dict to load from',
    )

    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--bf16', action='store_true', help='Use BF16')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA')
    parser.add_argument('--use_qlora', action='store_true', help='Use QLora')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--per_device_batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_crops', type=int, default=1, help='Number of maximum image crops')
    parser.add_argument(
        '--num_train_epochs', type=int, default=1, help='Number of training epochs'
    )
    parser.add_argument('--learning_rate', type=float, default=4.0e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false', help='Disable tqdm')
    parser.add_argument('--lora_rank', type=int, default=64, help='LoRA rank')
    parser.add_argument(
        '--lora_alpha_ratio', type=float, default=2, help='LoRA alpha to rank ratio'
    )
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout')
    parser.add_argument('--freeze_vision_model', action='store_true', help='Freeze vision model')

    # Whether or not to use TraceVLA finetuning format
    parser.add_argument('--tracevla', action='store_true', help='Whether or not to use tracevla')

    # OpenVLA arguments
    parser.add_argument('--data_mix', type=str, default='bridge', help='Dataset Mixture')
    parser.add_argument('--data_root_dir', type=str, default='/home/aiscuser/robot_data', help='Root directory for data')
    parser.add_argument('--shuffle_buffer_size', type=int, default=100000, help='Shuffle buffer size')
    parser.add_argument('--run_name', type=str, default='default', help='Name of the run')
    parser.add_argument(
        '--action_loss_calculation',
        type=str,
        default='action_token_only',
        help='Cross-entropy loss over entire vocab ("full") or 256 action tokens only ("action_token_only")'
    )
    parser.add_argument('--image_aug', action='store_true', help='Use image augmentation')
    parser.add_argument('--save_steps', type=float, default=1000, help='Frequency to save model')

    return parser


def parse_args():
    parser = get_args_parser()
    args = parser.parse_args()

    # Additional validation & logic
    assert args.num_crops <= 16, 'num_crops must be <= 16'
    if args.use_qlora:
        args.use_lora = True

    return args
