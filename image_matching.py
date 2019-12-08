import numpy as np
import cv2
import sys
import os
# import pdb; pdb.set_trace()


def run_sift(color1,color2,output_dir,fname1,fname2):
    img1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    output1 = cv2.drawKeypoints(color1,kp1,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    output2 = cv2.drawKeypoints(color2,kp2,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(os.path(output_dir, fname1) + '_features.tif', output1)
    cv2.imwrite(os.path(output_dir, fname2) + '_features.tif', output2)

    bf = cv2.BFMatcher(normType=cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append(m)

    sortedgood = sorted(good, key = lambda x:x.distance)
    draw = sortedgood[:20]
    output3 = cv2.drawMatches(color1,kp1,color2,kp2,draw,None,flags=2)
    cv2.imwrite(os.path(output_dir, fname1) + '_matches.tif', output3)

    if len(good) >= 4:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    else:
        print('NOT ENOUGH MATCHES FOR HOMOGRAPHY')
        return

    H = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)[0]
    dst = cv2.warpPerspective(color2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    dst[0:img1.shape[0], 0:img1.shape[1]] = color1
    cv2.imwrite(os.path(output_dir, fname1) + '_stitched.tif', dst)


def main(argv):
    img1path = argv[0]
    img2path = argv[1]
    color1 = cv2.imread(img1path, cv2.IMREAD_COLOR)
    color2 = cv2.imread(img2path, cv2.IMREAD_COLOR)

    run_sift(color1,color2)


if __name__ == '__main__':
    main(sys.argv[1:])