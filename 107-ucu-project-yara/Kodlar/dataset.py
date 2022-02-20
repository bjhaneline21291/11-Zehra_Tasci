from torch.utils.data import Dataset
from PIL import Image
import cv2
from torchvision import transforms
import random
import numpy as np
from utils import *
from torchvision.transforms import functional as F
from albumentations.torch.functional import img_to_tensor
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)
from albumentations.augmentations.transforms import RandomCrop

class SquareCenterCrop(object):
    """Crops the given PIL Image as a square at the center to save maximum.
    """

    def __init__(self):
        pass

    def __call__(self, img):
        """Returns cropped image

        img : PIL Image to be cropped.
        """
#         try:
#             h, w = img.size
#         except:
        img = Image.fromarray(img)
        h, w = img.size
        size = min(h, w)
        size = (int(size), int(size))
        return F.center_crop(img, size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """Returns unnormalized image.
        
        tensor : Tensor image of size (C, H, W) to be normalized.            
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)  # it is an inverse to normalization [ t.sub_(m).div_(s) ]
        return tensor

    
class ImageTransforms():

    FINAL_SIZE = (256, 256)
    MEAN = [0.881, 0.865, 0.875]
    STD = [0.157, 0.168, 0.154]
    
    def __init__(self):
        self.resize_to_final = transforms.Compose([
            SquareCenterCrop(),
            transforms.Resize(self.FINAL_SIZE)
        ])
        #self.necessary = lambda img: self.toTensor(self.resize_to_final(img))
        self.normalize = transforms.Normalize(self.MEAN, self.STD)
        self.unnormalize = UnNormalize(self.MEAN, self.STD)
        
    def augment(self):#p=.5):
        return Compose([
            RandomRotate90(),
            Flip(),
            Transpose(),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=.1),
                IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2),
            RandomCrop(256, 256, p=1),
            OneOf([
                MedianBlur(blur_limit=3, p=.1),
                Blur(blur_limit=3, p=.1),
            ], p=0.2),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomContrast(),
                RandomBrightness(),
            ], p=0.3),
            HueSaturationValue(p=0.7),
        ], p=1)


class TissuesDataset(Dataset):
    
    def __init__(self, img_files, mode='train', mask_files=None, augment=True, name=None):
        """Init
        img_files: list of str -- paths to images
        mask_files: list of str -- paths to masks
        mode: str in ['train', 'val', 'test']
        """
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.augment = augment
        if name is None:
            name = self.mode
        self.name = name
        self.img_files = img_files
        self.img_ids = [file[-8:-5] for file in self.img_files]
        self.num_samples = len(self.img_files)
        print('Dataset {} loaded'.format("'"+self.name+"'" if self.name is not None else ''))
        print('\t{} images'.format(self.num_samples))
        
        if self.mode == 'train' or self.mode == 'val':
            assert mask_files is not None
            self.mask_files = mask_files
            print('\t{} masks'.format(len(self.mask_files)))
        else:
            print('\tmasks are not provided')
            
        self.image_loader = imread #default_image_loader
        self.mask_loader = lambda img: imread(img, is_mask=True) #default_image_loader
        self.transforms = ImageTransforms()

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, i):  
        img = self.image_loader(self.img_files[i])
        
        if self.mode in ['train', 'val']:
            mask = self.mask_loader(self.mask_files[i])
            data = {"image": img, "mask": mask}
            if self.augment:
                augmented = self.transforms.augment()(**data)
            else:
                augmented = RandomCrop(256, 256, p=1)(**data)
            img, mask = augmented["image"], augmented["mask"]
            #mask = np.asarray(self.transforms.resize_to_final(mask))
            #print(np.unique(mask))
            
            #img = np.asarray(self.transforms.resize_to_final(img))
            img = img_to_tensor(img)
            img = self.transforms.normalize(img)
            
            return img, torch.from_numpy(np.expand_dims(mask, 0)).float()
        else:
            #img = np.asarray(self.transforms.resize_to_final(img))
            img = img_to_tensor(img)
            img = self.transforms.normalize(img)
            return img