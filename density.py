import cv2
import numpy as np
import sys, os
import random
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader


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
    def density_filter(self, classname, threshold, img_name): 
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
            return True
        return False

    """
    sequence of transformations to apply to each image in a random order
    certain transformations have probability parameters.
    we can make this much more complicated if we want to
    """
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-25,25)),
        iaa.Fliplr(p=0.5),
        iaa.Flipud(p=0.5)],
        random_order=True)

    """
    takes name of binary image and applies a given sequence of transformations.
    'transformation' parameter is an imgaug Sequential object.
    dir_name is the parent directory (e.g. train or test)
    returns the transformed version of the image and its mask as arrays
    """
    def augment(self, transformation, dir_name, img_name):
        img = cv2.imread(dir_name+'/image/'+img_name, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(dir_name+'/mask/'+img_name[:-4]+'_mask.png', cv2.IMREAD_GRAYSCALE)
        mask = SegmentationMapsOnImage(mask, shape=img.shape)

        image_aug, mask_aug = transformation(image=img, segmentation_maps=mask)
        return image_aug, mask_aug.get_arr()


def main(argv):
    dir_name = argv[0]
    directory = os.fsencode(dir_name)
    p = pipeline()

    for file in os.listdir(os.fsencode(dir_name + '/image')):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            if p.density_filter(dir_name + '/image', 0.01, dir_name+'/image/'+filename):
                if random.randint(0,101) <= 10:
                    image_aug, mask_aug = p.augment(p.seq, dir_name, filename)
                    cv2.imwrite(dir_name+'/image_aug/' + filename[:-4] + '_aug.png', image_aug)
                    cv2.imwrite(dir_name+'/mask_aug/' + filename[:-4] + '_mask_aug.png', mask_aug)
    print(p.usable)

if __name__ == '__main__':
    main(sys.argv[1:])