import argparse
from pathlib import Path

from tools.motion import MOTION_TYPE_SPECS


def add_source_args(group):
    group.add_argument("--video", type=Path, default=None, help="Path to a single video file")
    group.add_argument("--video-dir", type=Path, default=None, help="Directory containing multiple videos")
    group.add_argument("--workers", type=int, default=1, help="Parallel workers for multiple videos")
    group.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")


def add_type_args(group, motion_type):
    spec = MOTION_TYPE_SPECS[motion_type]
    for arg in spec["args"]:
        group.add_argument(*arg["flags"], dest=arg["dest"], **arg["kwargs"])


def add_type_feature_arg(group, motion_type):
    spec = MOTION_TYPE_SPECS[motion_type]
    group.add_argument(
        f"--{motion_type.replace('_', '-')}-features",
        dest=f"{motion_type}_features",
        nargs="+",
        choices=spec["feature_names"],
        default=list(spec["feature_names"]),
        help=f"Selected {motion_type} features (default: all)",
    )


def add_types_args(parser, motion_types):
    seen_dests = set()
    for motion_type in motion_types:
        spec = MOTION_TYPE_SPECS[motion_type]
        group = parser.add_argument_group(f"{motion_type} args", spec["description"])
        add_type_feature_arg(group, motion_type)
        for arg in spec["args"]:
            if arg["dest"] in seen_dests:
                continue
            group.add_argument(*arg["flags"], dest=arg["dest"], **arg["kwargs"])
            seen_dests.add(arg["dest"])


def motion_parser(motion_type=None):
    # 指定 motion_type 使用默认参数
    if motion_type is not None and motion_type not in MOTION_TYPE_SPECS:
        supported = ", ".join(MOTION_TYPE_SPECS)
        raise ValueError(f"Unsupported motion type '{motion_type}'. Supported types: {supported}")

    if motion_type is not None:
        parser = argparse.ArgumentParser(description=MOTION_TYPE_SPECS[motion_type]["description"])
        source_group = parser.add_argument_group("source args", "Input source")
        add_source_args(source_group)
        type_group = parser.add_argument_group(f"{motion_type} args", MOTION_TYPE_SPECS[motion_type]["description"])
        add_type_feature_arg(type_group, motion_type)
        add_type_args(type_group, motion_type)
        return parser

    # 命令行模式
    parser = argparse.ArgumentParser(description="Extract motion feature maps from videos")
    source_group = parser.add_argument_group("source args", "Input source")
    add_source_args(source_group)
    motion_group = parser.add_argument_group("motion args", "Motion type selection")
    motion_group.add_argument(
        "--motion",
        dest="motion_types",
        nargs="+",
        choices=tuple(MOTION_TYPE_SPECS),
        required=True,
        help="One or more motion feature types to extract",
    )
    add_types_args(parser, MOTION_TYPE_SPECS)
    return parser


def get_motion_kwargs_from_args(args, motion_types):
    motion_types = [motion_types] if isinstance(motion_types, str) else list(motion_types)
    kwargs = {}
    feature_names = []
    seen_features = set()
    seen_dests = set()
    for motion_type in motion_types:
        spec = MOTION_TYPE_SPECS[motion_type]
        selected_features = list(getattr(args, f"{motion_type}_features", None) or spec["feature_names"])
        for feature_name in selected_features:
            if feature_name not in seen_features:
                feature_names.append(feature_name)
                seen_features.add(feature_name)
        for arg_spec in spec["args"]:
            dest = arg_spec["dest"]
            if dest in seen_dests:
                continue
            kwargs[dest] = getattr(args, dest)
            seen_dests.add(dest)
    return kwargs, feature_names


def format_motion_args(args):
    motion_types = list(getattr(args, "motion_types", []) or [])
    lines = [
        "[source args]",
        f"video={getattr(args, 'video', None)}",
        f"video_dir={getattr(args, 'video_dir', None)}",
        f"workers={getattr(args, 'workers', 1)}",
        f"no_progress={getattr(args, 'no_progress', False)}",
    ]
    if motion_types:
        lines.append("")
        lines.append("[motion args]")
        lines.append(f"motion_types={motion_types}")

    for motion_type in motion_types:
        spec = MOTION_TYPE_SPECS[motion_type]
        lines.append("")
        lines.append(f"[{motion_type} args]")
        lines.append(f"feature_names={list(getattr(args, f'{motion_type}_features', spec['feature_names']))}")
        for arg_spec in spec["args"]:
            dest = arg_spec["dest"]
            lines.append(f"{dest}={getattr(args, dest)}")
    return "\n".join(lines)
