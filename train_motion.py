# coding=utf-8
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")


def parse_value(value: str):
    lowered = value.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def parse_overrides(items: list[str]) -> dict:
    overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"extra override must be key=value, got: {item}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = parse_value(value.strip())
    return overrides


def parse_device(value: str):
    if value.lower() == "cpu":
        return "cpu"
    if "," in value:
        return [int(x) for x in value.split(",") if x.strip()]
    return [int(value)]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a motion YOLO model.")
    parser.add_argument("--model", "--cfg", dest="model", required=True, help="Model YAML or checkpoint path.")
    parser.add_argument("--data", required=True, help="Dataset YAML.")
    parser.add_argument(
        "--motion",
        nargs="*",
        default=[],
        help="Motion feature names, e.g. gray_diff_short gray_diff_long. Leave empty for ordinary YOLO training.",
    )
    parser.add_argument("--weights", "--pretrained", dest="pretrained", default=True, help="Pretrained weights, or False.")
    parser.add_argument("--task", default="motion", help="Ultralytics task, default: motion.")
    parser.add_argument("--name", default=None, help="Run name. Defaults to <model-stem>-<imgsz>.")
    parser.add_argument("--project", default=None, help="Project directory.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="Device, e.g. 0, 0,1, or cpu.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--single-cls", action="store_true", help="Train as single-class dataset.")
    parser.add_argument("--resume", action="store_true", help="Resume training.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting an existing run directory.")
    parser.add_argument("overrides", nargs="*", help="Additional train overrides as key=value.")
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.name or f"{Path(args.model).stem}-{args.imgsz}"
    if args.single_cls and args.name is None:
        name += "-single"

    from ultralytics import YOLO

    model = YOLO(args.model, task=args.task)
    train_kwargs = {
        "name": name,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "pretrained": parse_value(str(args.pretrained)),
        "device": parse_device(args.device),
        "motion": args.motion,
        "workers": args.workers,
        "single_cls": args.single_cls,
        "resume": args.resume,
        "exist_ok": args.exist_ok,
        **parse_overrides(args.overrides),
    }
    if args.project:
        train_kwargs["project"] = args.project

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
