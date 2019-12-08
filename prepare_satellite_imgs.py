import cv2
import argparse
import numpy as np
import os
from image_matching import run_sift


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("img1", help="Path of img1 file")
    parser.add_argument("img2", help="Path of img2 file")
    parser.add_argument("pre_date", help="Date of pre-event")
    parser.add_argument("post_date", help="Date of post-event")
    parser.add_argument("directory", help="Image directory")
    parser.add_argument("-dimension", help="Tiled image dimension", default=512, required=False)
    parser.add_argument("-s", help="Splice images", default=False, required=False)
    args = parser.parse_args()
    return args


def get_splice_factor(img, dimension):
    s = len(img)
    while s > dimension:
        s = s / 2
    return int(s)


def splice(directory, fname, dimension, img):
    s = get_splice_factor(img, dimension)
    for i in range(int(len(img)/s)):
        for j in range(int(len(img[0])/s)):
            # print(i, j, s)
            # print(s*i, s * (i+1), s*j, s * (j+1))
            cv2.imwrite(os.path.join(directory, 'tiled/') + fname + '_' + str(s*i + j) + '.tif', img[s*i:s * (i+1), s*j:s * (j+1)])


def main():
    args = get_args()
    if args.s:
        img1 = cv2.imread(os.path.join(args.directory, 'pre_event', args.pre_date, args.img1) + '_jpeg_compressed.tif', cv2.IMREAD_COLOR)
        img2 = cv2.imread(os.path.join(args.directory, 'post_event', args.post_date, args.img2) + '_jpeg_compressed.tif', cv2.IMREAD_COLOR)
        splice(os.path.join(args.directory, 'pre_event', args.pre_date), args.img1, args.dimension, img1)
        splice(os.path.join(args.directory, 'post_event', args.post_date), args.img2, args.dimension, img2)

    pre_events = [f for f in os.listdir(os.path.join(args.directory, 'pre_event', args.pre_date, 'tiled')) if not f.startswith('._')]
    post_events = [f for f in os.listdir(os.path.join(args.directory, 'post_event', args.post_date, 'tiled')) if not f.startswith('._')]
    events = list(zip(pre_events, post_events))
    
    for i, j in events:
        pre_dir = os.path.join(args.directory, 'pre_event', args.pre_date)
        post_dir = os.path.join(args.directory, 'post_event', args.post_date)
        img1 = cv2.imread(os.path.join(pre_dir, 'tiled', i), cv2.IMREAD_COLOR)
        img2 = cv2.imread(os.path.join(post_dir, 'tiled', j), cv2.IMREAD_COLOR)
        try:
            run_sift(img1, img2, os.path.join(pre_dir, 'matches'), os.path.join(post_dir, 'matches'), i.split('.')[0], j.split('.')[0])
        except:
            continue

if __name__ == '__main__':
    main()