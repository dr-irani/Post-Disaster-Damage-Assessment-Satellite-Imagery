"""
This requires an older version of OpenCV to run SURF feature detection.
Version used: 3.4.2.
"""

import argparse
import numpy as np
import cv2
import sys
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-directory", default="/Volumes/ExtremeSSD/cs461_final_project/data/disaster_images/manually_selected", help="Image directory")
    parser.add_argument("-dimension", help="Tiled image dimension", default=512, required=False)
    parser.add_argument("-factor", help="Contrast factor", default=100, required=False)
    parser.add_argument("-c", help="Add contrast", default=False, required=False)
    parser.add_argument("-s", help="Splice images", default=False, required=False)
    args = parser.parse_args()
    return args


def run_sift(color1,color2,dir1,dir2,fname1,fname2):
    img1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SURF_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    output1 = cv2.drawKeypoints(color1,kp1,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    output2 = cv2.drawKeypoints(color2,kp2,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite(os.path.join(dir1, fname1) + '_features.tif', output1)
    # cv2.imwrite(os.path.join(dir2, fname2) + '_features.tif', output2)

    bf = cv2.BFMatcher(normType=cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    sortedgood = sorted(good, key = lambda x:x.distance)
    draw = sortedgood[:20]
    output3 = cv2.drawMatches(color1,kp1,color2,kp2,draw,None,flags=2)
    cv2.imwrite(os.path.join(dir2, fname2) + '_matches.tif', output3)

    if len(good) >= 4:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    else:
        print('NOT ENOUGH MATCHES FOR HOMOGRAPHY')
        return

    H = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]
    # dst = cv2.warpPerspective(color2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    # dst[0:img1.shape[0], 0:img1.shape[1]] = color1
    # cv2.imwrite(os.path.join(dir2, fname2) + '_stitched.tif', dst)
    dst = cv2.warpPerspective(color2, H, img2.shape)
    cv2.imwrite(os.path.join(dir2, fname2) + '_homography.tif', dst)
    return H


def main(argv):
    # img1path = argv[0]
    # img2path = argv[1]
    # color1 = cv2.imread(img1path, cv2.IMREAD_COLOR)
    # color2 = cv2.imread(img2path, cv2.IMREAD_COLOR)

    # run_sift(color1,color2)

    img_names = []

    pre_events = [f for f in os.listdir(os.path.join(args.directory, 'pre_event/charlotte', 'tiled')) if not f in img_names]
    post_events = [f for f in os.listdir(os.path.join(args.directory, 'post_event', args.post_date, 'tiled')) if not f in img_names]
    events = list(zip(pre_events, post_events))
    for i, j in events:
        pre_dir = os.path.join(args.directory, 'pre_event', args.pre_date)
        post_dir = os.path.join(args.directory, 'post_event', args.post_date)
        img1 = cv2.imread(os.path.join(pre_dir, 'tiled', i), cv2.IMREAD_COLOR)
        img2 = cv2.imread(os.path.join(post_dir, 'tiled', j), cv2.IMREAD_COLOR)
        try:
            T_matrix = run_sift(img1, img2, os.path.join(pre_dir, 'matches'), os.path.join(post_dir, 'matches'), i.split('.')[0], j.split('.')[0])
            print(i)
            print(H)
        except:
            continue



if __name__ == '__main__':
    main(sys.argv[1:])