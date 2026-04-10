from pathlib import Path
from tools.motion import VIDEO_SUFFIXES


def split_path(path):
    p = Path(path)
    return p.parent, p.stem, p.suffix


def get_arg(kwargs, *keys, type_func, default):
    for key in keys:
        value = kwargs.get(key)
        if value is not None:
            return type_func(value)
    return default


def iter_video_paths(video=None, video_dir=None):
    if video:
        path = Path(video)
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {video}")
        return [path]

    if not video_dir:
        raise ValueError("You must specify either --video or --video-dir")

    path = Path(video_dir)
    if not path.is_dir():
        raise NotADirectoryError(f"Video directory not found: {video_dir}")

    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)





