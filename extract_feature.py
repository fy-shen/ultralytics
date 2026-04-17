from tools.motion.parser import motion_parser, format_motion_args
from tools.motion.extractor import run_motion_extraction


if __name__ == "__main__":
    args = motion_parser().parse_args()
    print(format_motion_args(args))
    run_motion_extraction(args, args.motion_types)
