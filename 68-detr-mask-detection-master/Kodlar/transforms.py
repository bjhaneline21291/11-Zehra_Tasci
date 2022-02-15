"""
Custom transformations for image localization problems
Typically shadows Torchvision implementations
"""
#%% Setup
from random import random

import torch
import torchvision.transforms.functional as F

#%% Compose
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes, classes):
        for transform in self.transforms:
            image, boxes, classes = transform(image, boxes, classes)

        return image, boxes, classes

#%% To Tensor
class ToTensor:
    def __call__(self, image, boxes, classes):
        # Convert the image to a tensor
        image = F.to_tensor(image)

        return image, boxes, classes

#%% Normalize Boxes
class NormalizeBoxes:
    """
    Normalizes the boxes in relation to image dimensions
    Boxes are assumed to be of the form x min, y min, x max, y max
    """
    def __call__(self, image, boxes, classes):
        boxes = boxes / torch.Tensor([[image.width, image.height, image.height, image.width]])

        return image, boxes, classes

#%% Resize
class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, boxes, classes):
        image = F.resize(img=image, size=self.size)

        return image, boxes, classes

#%% Affine
class RandomAffine:
    def __init__(self, dx=0.15, dy=0.15):
        self.dx = dx
        self.dy = dy

    def __call__(self, image, boxes, classes):
        # Get the random movements
        dx = torch.rand((1)) * self.dx
        dy = torch.rand((1)) * self.dy

        # Update the image
        image = F.affine(img=image, translate=(dx, dy), scale=1, angle=0, shear=0)

        # Update the boxes
        boxes = [box + torch.Tensor([dx, dy, 0, 0]) for box in boxes]

        return image, boxes, classes

#%% Mirror
class Mirror:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, image, boxes, classes):
        if random() < self.p:
            # Flip the image
            image = F.hflip(img=image)

            # Flip the boxes
            for box in boxes:
                box[0] = 1 - box[0]

        return image, boxes, classes

#%% ColourJitter
class ColourJitter:
    def __init__(self, d_hue=0.15, d_saturation=0.7, d_brightness=0.7, d_contrast=0.7, p=0.5):
        self.d_hue = d_hue
        self.d_saturation = d_saturation
        self.d_brightness = d_brightness
        self.d_contrast = d_contrast
        self.p = p

    def __call__(self, image, boxes, classes):
        if random() < self.p:
            image = F.adjust_hue(img=image, hue_factor=torch.rand([1]) * self.d_hue)
            image = F.adjust_saturation(img=image, saturation_factor=torch.rand([1]) * self.d_saturation + 1)
            image = F.adjust_brightness(img=image, brightness_factor=torch.rand([1]) * self.d_brightness + 1)
            image = F.adjust_contrast(img=image, contrast_factor=torch.rand([1]) * self.d_contrast + 1)

        return image, boxes, classes

#%% Noise
class RandomNoise:
    def __init__(self, frequency=0.1, p=0.5):
        self.frequency = frequency
        self.p = p

    def __call__(self, image, boxes, classes):
        if random() < self.p:
            # Create random noise
            noise = torch.rand_like(image)

            # Mask the noise
            noise[torch.rand_like(noise) > (self.frequency)] = 0

            # Update the image
            image = image + noise

        return image, boxes, classes
