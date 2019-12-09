import argparse
import numpy as np
import cv2
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("img1", help="Path of img1 file")
    parser.add_argument("img2", help="Path of img2 file")
    parser.add_argument("directory", help="Image directory")
    args = parser.parse_args()
    return args

def calculate_per_pixel_change(img1, img2):
    epsilon = 1e-5
    img1_normalized = (img1 - img1.mean(axis=0)) / (img1.std(axis=0) + epsilon)
    img2_normalized = (img2 - img2.mean(axis=0)) / (img2.std(axis=0) + epsilon)
    img_diff = img1_normalized - img2_normalized
    manhattan_norm = np.sum(abs(norm))
    zero_norm = np.linalg.norm(img_diff.ravel(), 0)
    return manhattan_norm, zero_norm


def main():
    args = get_args()
    img1 = cv2.imread(os.path.join(args.directory, args.img1), cv2.IMREAD_COLOR)
    img2 = cv2.imread(os.path.join(args.directory, args.img2), cv2.IMREAD_COLOR)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) / 255.0
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) / 255.0
    manhattan_norm, zero_norm = calculate_per_pixel_change(gray1, gray2)
    print("Manhattan norm: {} per pixel: {}".format(manhattan_norm, manhattan_norm / cv.GetSize(gray1)))
    print("Zero norm: {} per pixel: {}".format(zero_norm, zero_norm / cv.GetSize(gray1)))


if __name__ == '__main__':
    main()