import cv2
import numpy as np
import sys, os
import random
from imgaug import augmenters as iaa
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader


class pipeline: 
    def __init__(self):
        self.usable = {}

        """
        sequence of geometrical transformations to apply to each image in a random order.
        certain transformations have probability parameters.
        """
        self.seq = iaa.Sequential([
        	iaa.SomeOf((0,3), [
        		iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
        		iaa.Fliplr(p=0.5),
        		iaa.Flipud(p=0.5),
        		iaa.Affine(rotate=(-10,10), mode='symmetric'),
        		]),
        	],
            random_order=True)

        """
        three options for adding noise. should only be applied to images, not masks.
        """
        self.noise = iaa.OneOf([
                iaa.SaltAndPepper(0.1, per_channel=True),
                iaa.GaussianBlur(sigma=(0.0,2.0)),
                iaa.Sharpen(alpha=(0.0,0.75))
            ])

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
    'seq' and 'noise' parameters are imgaug Sequential objects.
    dir_name is the parent directory (e.g. train or test)
    returns the transformed version of the image and its mask as arrays
    """
    def augment(self, seq, noise, dir_name, img_name):
        img = cv2.imread(dir_name+'/'+img_name[:-9]+'_sat.jpg', cv2.IMREAD_COLOR)
        mask = cv2.imread(dir_name+'/'+img_name, cv2.IMREAD_COLOR)

        if random.randint(0,2) == 1:
            seq_det = seq.to_deterministic()
            image_aug = seq_det.augment_image(image=img)
            mask_aug = seq_det.augment_image(image=mask)
        else:
            image_aug = noise.augment_image(image=img)
            mask_aug = mask

        return image_aug, mask_aug

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
                    image_aug, mask_aug = p.augment(p.seq, p.noise, dir_name, filename)
                    # p.splice(dir_name, filename, image_aug, mask_aug, 1000)
                    print('writing augmented')
                    cv2.imwrite(dir_name + '/output/' + filename[:-9] + '_sat_aug.jpg', image_aug)
                    cv2.imwrite(dir_name + '/output/' + filename[:-4] + '_aug.png', mask_aug)
                    # cv2.imwrite(dir_name+ '/' + filename[:-4] + '_aug.jpg', image_aug)
                    # cv2.imwrite(dir_name+ '/' + filename[:-4] + '_aug.png', mask_aug)
                else:
                    print('writing non-augmented')
                    image = cv2.imread(dir_name+'/'+filename[:-9]+'_sat.jpg', cv2.IMREAD_COLOR)
                    cv2.imwrite(dir_name + '/output/' + filename[:-9] + '_sat.jpg', image)
                    mask = cv2.imread(dir_name+'/'+filename, cv2.IMREAD_COLOR)
                    cv2.imwrite(dir_name + '/output/' + filename[:-4] + '.png', mask)

    #print(p.usable)

if __name__ == '__main__':
    main(sys.argv[1:])