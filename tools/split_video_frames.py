import os
import cv2
import argparse


def split_video(in_video_path, out_image_path):
    """ split video"""
    videos = os.listdir(in_video_path)
    for video_name in videos:
        print(video_name)
        video_path = os.path.join(in_video_path, video_name)
        cap = cv2.VideoCapture(video_path)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            else:
                frame_name_path = os.path.join(out_image_path, f"{video_name.split('.')[0]}_{idx:06}.jpg")
                cv2.imwrite(frame_name_path, frame)
            idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_path', default="/home/sfy/SFY/disk1/data/drone_bird/", type=str,
                        help='data_root_path: The path of the dataset.')
    args = parser.parse_args()

    video_path = os.path.join(args.data_root_path, "videos")
    image_path = os.path.join(args.data_root_path, "images")
    os.makedirs(image_path, exist_ok=True)
    split_video(in_video_path=video_path, out_image_path=image_path)


if __name__ == "__main__":
    main()

