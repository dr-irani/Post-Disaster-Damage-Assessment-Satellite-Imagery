import cv2
import numpy as np
import sys, os


class pipeline: 
    def __init__(self):
        self.usable = {}

    def add_to_dict(self,d, k, v):
        if k in d:
            cur_usable = d[k]
            d[k] = cur_usable
        else:
            d[k]=[v]
        return d

    """
    binarizes the image, computes the percentage of pixels labeled to be road
    then if the percentage is over the threashold, save the mask name 
    (probably process later into actual sample name) into the usuable folder
    sample output with input of python3 density.py road is 
    {'road': ['road/dense.png']}
    """
    def density_filter(self,classname, threshold, img_name): 
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        """gray_image[gray_image >= 128] = 1
        gray_image[gray_image < 128] = 0"""
        gray_image=np.where(gray_image>=128,1,0)
        #print(np.max(gray_image))

        total_size = len(gray_image) * len(gray_image[0])
        #print(total_size)
        total_road = np.sum(gray_image)
        #print(total_road)
        total_percent = total_road / total_size 
        #print(total_percent)
        if total_percent > threshold: 
            self.usable = self.add_to_dict(self.usable, classname, img_name)
        return 

def main(argv):
    dir_name = argv[0]
    directory = os.fsencode(dir_name)
    p = pipeline()

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            p.density_filter("road", 0.01, dir_name+'/'+filename)
    print(p.usable)

if __name__ == '__main__':
    main(sys.argv[1:])