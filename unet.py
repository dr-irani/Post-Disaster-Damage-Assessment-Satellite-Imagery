import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys,os,glob, time, math, tables, random
import PIL
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorboardX import SummaryWriter
import scipy.ndimage 
from sklearn.metrics import confusion_matrix
'''def load_dataset(data_path):
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )


    return train_loader'''

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )  

class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out


class SegmentationDataset():
    def __init__(self, datafolder, datatype='train', transform = torchvision.transforms.ToTensor()):
        self.datafolder = datafolder
        self.image_files_list = None
        if (datatype=='train'): 
            self.image_files_list = [s for s in os.listdir(datafolder+'/data') if 
                                '_%s.jpg' % datatype in s] 
            # Same for the labels files
            self.label_files_list =  [s for s in os.listdir(datafolder+'/mask') if 
                                '_%s.jpg' % datatype in s] 
        self.transform = transform

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.datafolder,
                                self.image_files_list[idx])
        image = Image.open(img_name)
        image = self.transform(image)
        # Same for the labels files
        #label = .... # Load in etc
        label = self.transform(label)
        return image, label

def main(argv):
    trainset = SegmentationDataset(datafolder = argv[0])
    model = ResUNet(n_class=2)
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=60)
    #train_loader = load_dataset(argv[0])
    #for batch_idx, (data, target) in enumerate(load_dataset(argv[0])):


if __name__ == '__main__':
    main(sys.argv[1:])