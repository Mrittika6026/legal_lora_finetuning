import argparse
import json
from pathlib import Path

from .config import load_config, AppConfig
from .inference import generate_response
from .training import run_training


def _load_messages(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_train(args):
    cfg = load_config(args.config)
    run_training(cfg, resume_from_checkpoint=args.resume_from_checkpoint)


def _run_infer(args):
    cfg = load_config(args.config)
    if args.prompt_file:
        messages = _load_messages(args.prompt_file)
    else:
        messages = [
            {"role": "system", "content": cfg.inference.system_prompt},
            {"role": "user", "content": args.prompt},
        ]
    response = generate_response(cfg, messages)
    print(response)


def build_parser():
    parser = argparse.ArgumentParser(description="LoRA fine-tune CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Run training")
    train_p.add_argument(
        "--config",
        type=str,
        default="lora_finetune/configs/qwen3-4b.yaml",
        help="Path to YAML config."
    )
    train_p.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        dest="resume_from_checkpoint",
        help="Checkpoint path to resume from."
    )
    train_p.set_defaults(func=_run_train)

    infer_p = subparsers.add_parser("infer", help="Run inference with trained adapter")
    infer_p.add_argument(
        "--config",
        type=str,
        default="lora_finetune/configs/qwen3-4b.yaml",
        help="Path to YAML config."
    )
    infer_p.add_argument(
        "--prompt",
        type=str,
        default="Explain how tax exemption approval continues.",
        help="User prompt (ignored if --prompt-file is set)."
    )
    infer_p.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to JSON file containing messages list."
    )
    infer_p.set_defaults(func=_run_infer)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

