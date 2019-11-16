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

        """
        sequence of transformations to apply to each image in a random order
        certain transformations have probability parameters.
        TODO: play around with iaa.Clouds augmentation
        """
        self.seq = iaa.Sequential([
        	iaa.SomeOf(4, [
        		iaa.Grayscale(alpha=(0.0, 1.0)),
        		iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
        		iaa.Fliplr(p=0.5),
        		iaa.Flipud(p=0.5),
        		iaa.Affine(rotate=(-10,10), mode='symmetric'),
        		iaa.SaltAndPepper(0.1, per_channel=True),
        		iaa.OneOf([
        			iaa.GaussianBlur(sigma=(0.0,2.0)),
        			iaa.Sharpen(alpha=(0.0,0.75))
        			]),
        		iaa.Noop(),
        		iaa.Noop(),
        		]),
        	],
            random_order=True)

    def add_to_dict(self,d, k, v):
        if k in d:
            cur_usable = d[k]
            cur_usable.append(v)
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
        if total_percent > threshold: 
            self.usable = self.add_to_dict(self.usable, classname, img_name[:-4] + '_aug.png')
            return True
        return False

    """
    takes name of binary image and applies a given sequence of transformations.
    'transformation' parameter is an imgaug Sequential object.
    dir_name is the parent directory (e.g. train or test)
    returns the transformed version of the image and its mask as arrays
    """
    def augment(self, transformation, dir_name, img_name):
        img = cv2.imread(dir_name+'/'+img_name[:-9]+'_sat.jpg', cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(dir_name+'/'+img_name, cv2.IMREAD_GRAYSCALE)
        print(type(img))
        print(type(mask))

        mask = SegmentationMapsOnImage(mask, shape=(len(mask), len(mask[0])))

        image_aug, mask_aug = transformation(image=img, segmentation_maps=mask)
        return image_aug, mask_aug.get_arr()

    def splice(self, dir_name, filename, image_aug, mask_aug, dim):
        dir_name+'/'+filename[:-9]+'_sat.jpg'
        s = len(image_aug)
        while s > dim:
            s = s / 2
        s = int(s)
        for i in range(int(len(image_aug)/s)):
            for j in range(int(len(image_aug[0])/s)):
                cv2.imwrite(dir_name+ '/' + filename[:-9] + '_sat_aug_' +str(i) +'.jpg', image_aug[s*i:s * (i+1), s*j:s * (j+1)])
                cv2.imwrite(dir_name+ '/' + filename[:-4] + '_aug_' +str(i) +'.png', mask_aug[s*i:s * (i+1), s*j:s * (j+1)])

def main(argv):
    dir_name = argv[0]
    directory = os.fsencode(dir_name)
    p = pipeline()

    for file in os.listdir(os.fsencode(dir_name)):
        filename = os.fsdecode(file)
        if filename.endswith(".png") and 'mask' in filename:
            kept = False 
            if 'roads' in dir_name: 
                kept = p.density_filter('roads', 0.01, dir_name+'/'+filename)
            else:
                kept = p.density_filter('buildings', 0.01, dir_name+'/'+filename)
            if kept:
                if random.randint(0,101) <= 10:
                    image_aug, mask_aug = p.augment(p.seq, dir_name, filename)
                    p.splice(dir_name, filename, image_aug, mask_aug, 1000)
                    #cv2.imwrite('./image_aug/' + filename[:-4] + '_aug.png', image_aug)
                    #cv2.imwrite('./mask_aug/' + filename[:-4] + '_mask_aug.png', mask_aug)
                    #cv2.imwrite(dir_name+ '/' + filename[:-4] + '_aug.jpg', image_aug)
                    #cv2.imwrite(dir_name+ '/' + filename[:-4] + '_aug.png', mask_aug)

    #print(p.usable)

if __name__ == '__main__':
    main(sys.argv[1:])