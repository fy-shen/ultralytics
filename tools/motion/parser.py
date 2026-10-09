import argparse
from pathlib import Path

from tools.motion import MOTION_TYPE_SPECS


MOTION_TYPE_ALIASES = {"gd": "gray_diff"}


class MotionHelpFormatter(argparse.RawTextHelpFormatter):
    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)
        return ", ".join(action.option_strings)


def add_source_args(group):
    group.add_argument("--video", type=Path, default=None, help="Path to a single video file")
    group.add_argument("--video-dir", type=Path, default=None, help="Directory containing multiple videos")
    group.add_argument("--workers", type=int, default=1, help="Parallel workers for multiple videos")
    group.add_argument(
        "--save-rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save full RGB frame sequence to the images directory during motion extraction",
    )
    group.add_argument("--overwrite-rgb", action="store_true", help="Overwrite existing RGB frame images")
    group.add_argument("--frames-only", action="store_true", help="Only extract RGB frames, without motion features")
    group.add_argument("--frame-stride", type=int, default=1, help="Save every Nth RGB frame in --frames-only mode")
    group.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")


def add_type_args(group, motion_type):
    spec = MOTION_TYPE_SPECS[motion_type]
    for arg in spec["args"]:
        group.add_argument(*arg["flags"], dest=arg["dest"], **arg["kwargs"])


def add_type_feature_arg(group, motion_type):
    spec = MOTION_TYPE_SPECS[motion_type]
    values = list(spec.get("feature_aliases", {})) or list(spec["feature_names"])
    values_text = ", ".join(values)
    option_name = "--gd-features" if motion_type == "gray_diff" else f"--{motion_type.replace('_', '-')}-features"
    group.add_argument(
        option_name,
        dest=f"{motion_type}_features",
        nargs="+",
        metavar="FEATURE",
        default=argparse.SUPPRESS,
        help=f"Selected features ({values_text}). Default: all.",
    )


def add_types_args(parser, motion_types):
    seen_dests = set()
    for motion_type in motion_types:
        spec = MOTION_TYPE_SPECS[motion_type]
        display_type = "gd" if motion_type == "gray_diff" else motion_type
        group = parser.add_argument_group(f"{display_type} args", spec["description"])
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
        parser = argparse.ArgumentParser(
            description=MOTION_TYPE_SPECS[motion_type]["description"],
            formatter_class=MotionHelpFormatter,
        )
        source_group = parser.add_argument_group("source args", "Input source")
        add_source_args(source_group)
        display_type = "gd" if motion_type == "gray_diff" else motion_type
        type_group = parser.add_argument_group(f"{display_type} args", MOTION_TYPE_SPECS[motion_type]["description"])
        add_type_feature_arg(type_group, motion_type)
        add_type_args(type_group, motion_type)
        return parser

    # 命令行模式
    parser = argparse.ArgumentParser(
        description="Extract motion feature maps from videos",
        formatter_class=MotionHelpFormatter,
        usage=(
            "%(prog)s [-h] [--video] [--video-dir] [--workers] [--save-rgb] "
            "[--frames-only] [--frame-stride] [--no-progress] [--motion] "
            "[feature/algorithm options]"
        ),
    )
    source_group = parser.add_argument_group("source args", "Input source")
    add_source_args(source_group)
    motion_group = parser.add_argument_group("motion args", "Motion type selection")
    motion_group.add_argument(
        "--motion",
        dest="motion_types",
        nargs="+",
        metavar="TYPE",
        default=None,
        help="Motion type(s): gd, gde, fgmask, flow.",
    )
    add_types_args(parser, MOTION_TYPE_SPECS)
    return parser


def get_motion_kwargs_from_args(args, motion_types):
    motion_types = [motion_types] if isinstance(motion_types, str) else list(motion_types)
    supported_cli_motion_types = set(MOTION_TYPE_ALIASES) | {x for x in MOTION_TYPE_SPECS if x != "gray_diff"}
    for motion_type in motion_types:
        if motion_type not in supported_cli_motion_types:
            supported_text = ", ".join(["gd"] + [x for x in MOTION_TYPE_SPECS if x != "gray_diff"])
            raise ValueError(f"Unsupported motion type '{motion_type}'. Supported values: {supported_text}")
    motion_types = [MOTION_TYPE_ALIASES.get(motion_type, motion_type) for motion_type in motion_types]
    kwargs = {}
    feature_names = []
    seen_features = set()
    seen_dests = set()
    for motion_type in motion_types:
        spec = MOTION_TYPE_SPECS[motion_type]
        raw_features = list(getattr(args, f"{motion_type}_features", None) or spec["feature_names"])
        aliases = dict(spec.get("feature_aliases", {}))
        supported = set(spec["feature_names"]) | set(aliases)
        selected_features = []
        for raw in raw_features:
            if raw not in supported:
                supported_text = ", ".join(sorted(supported))
                raise ValueError(f"Unsupported {motion_type} feature '{raw}'. Supported values: {supported_text}")
            selected_features.append(aliases.get(raw, raw))
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
    motion_types = [MOTION_TYPE_ALIASES.get(x, x) for x in list(getattr(args, "motion_types", []) or [])]
    lines = [
        "[source args]",
        f"video={getattr(args, 'video', None)}",
        f"video_dir={getattr(args, 'video_dir', None)}",
        f"workers={getattr(args, 'workers', 1)}",
        f"save_rgb={getattr(args, 'save_rgb', False)}",
        f"overwrite_rgb={getattr(args, 'overwrite_rgb', False)}",
        f"frames_only={getattr(args, 'frames_only', False)}",
        f"frame_stride={getattr(args, 'frame_stride', 1)}",
        f"no_progress={getattr(args, 'no_progress', False)}",
    ]
    if motion_types:
        lines.append("")
        lines.append("[motion args]")
        lines.append(f"motion_types={motion_types}")

    for motion_type in motion_types:
        spec = MOTION_TYPE_SPECS[motion_type]
        raw_features = list(getattr(args, f"{motion_type}_features", None) or spec["feature_names"])
        aliases = dict(spec.get("feature_aliases", {}))
        display_features = [aliases.get(name, name) for name in raw_features]
        lines.append("")
        lines.append(f"[{motion_type} args]")
        lines.append(f"feature_names={display_features}")
        for arg_spec in spec["args"]:
            dest = arg_spec["dest"]
            lines.append(f"{dest}={getattr(args, dest)}")
    return "\n".join(lines)
