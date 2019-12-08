import numpy as np
import cv2
import sys
# import pdb; pdb.set_trace()


def run_sift(color1,color2):
    img1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    output1 = cv2.drawKeypoints(color1,kp1,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    output2 = cv2.drawKeypoints(color2,kp2,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('siftoutput/output1.jpg',output1)
    cv2.imwrite('siftoutput/output2.jpg',output2)

    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    good = matches[:20]

    output3 = cv2.drawMatches(color1,kp1,color2,kp2,good,None,flags=2)
    cv2.imwrite('siftoutput/output3.jpg',output3)


def run_stitcher(color1, color2):
    stitcher = cv2.createStitcher()
    images = [color1, color2]
    (status, stitched) = stitcher.stitch(images)
    cv2.imwrite('siftoutput/stitched.jpg',stitched)


def main(argv):
    img1path = argv[0]
    img2path = argv[1]
    color1 = cv2.imread(img1path, cv2.IMREAD_COLOR)
    color2 = cv2.imread(img2path, cv2.IMREAD_COLOR)

    run_sift(color1,color2)
    run_stitcher(color1,color2)



if __name__ == '__main__':
    main(sys.argv[1:])