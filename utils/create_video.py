import glob
import os
import argparse
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

parser = argparse.ArgumentParser(description="Convert sequence of images to video.")
parser.add_argument("--image_path", type=str, default="data/MOT16/train/MOT16-13/img1", help="Path to the images.")
parser.add_argument("--output_path", type=str, default="data/MOT16/videos/MOT16-13-raw.mp4", help="Path to store the video.")
args = parser.parse_args()

image_path = args.image_path
output_path = args.output_path

images = sorted(glob.glob(os.path.join(image_path, "*.jpg")))
clip = ImageSequenceClip(images, fps=30)
clip.write_videofile(output_path, codec="libx264")