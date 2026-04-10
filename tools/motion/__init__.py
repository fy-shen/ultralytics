VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg")

from tools.motion.extractor import FEATURE_EXTRACTOR_REGISTRY, build_motion_type_specs

MOTION_TYPE_SPECS = build_motion_type_specs()

__all__ = (
    "VIDEO_SUFFIXES",
    "FEATURE_EXTRACTOR_REGISTRY",
    "MOTION_TYPE_SPECS",
)
