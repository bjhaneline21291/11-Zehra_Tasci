import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

def imread(imgfile, is_mask=False):
    """Returns image [h, w, 3] or mask [h, w] (with values in [0,1]) 
    """
    img = cv2.cvtColor(cv2.imread(imgfile), cv2.COLOR_BGR2RGB)
    if is_mask:
        if img.ndim == 3:
            img = img.mean(2)
        img[img > 1] = 1
    return img
    
def imshow(img, title=None):
    if title is None:
        title = img.shape
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    
def imsave(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

def maskshow(mask, title=None):
    """Converts mask shape from [h, w] to [h, w, 3], mulptiplies by 255 and imshow
    """
    if mask.ndim == 2:
        mask = np.stack([mask, mask, mask], 2)
    elif mask.shape[2] == 1:
        mask = mask[:,:,0]
        mask = np.stack([mask, mask, mask], 2)
    imshow((mask*255).astype(np.uint8), title=None)
    
    
############################################################################
# For croping to the cover box
############################################################################
def trim_padding(img, th=0):
    """
    Removes zero-borders (returns mask)
    img: 2D (h, w) or 3D (h, w, c) np.array
    """
    #print(img.shape)
    if img.ndim == 2:
        mask = img > th
        return img[np.ix_(mask.any(1), mask.any(0))]
    else:
        mask = img.mean(axis=2) > th
        return mask  # then use it like: img[np.ix_(mask.any(1), mask.any(0))]

def get_cover(img, cover, borders=None):
    """
    Cuts image according to the cover (bbox)
    borders: tuple of int (h, w) representing zero-padding or int => (h,h)
    """
    img *= cover
    mask = trim_padding(cover)
    img = img[np.ix_(mask.any(1), mask.any(0))]
    if borders is not None:
        if isinstance(borders, int):
            borders = (borders, borders)
        assert isinstance(borders, tuple) and len(borders) == 2
        h, w = borders
        img = cv2.copyMakeBorder(img, h, h, w, w, cv2.BORDER_CONSTANT)
    return img 

############################################################################
# For generating patches
############################################################################
def get_patches(img, patch_size):
    (h, w) = patch_size
    (H, W) = img.shape[:2]
    
    xs = np.arange(0, W, w)
    if W - xs[-1] > w // 2:
        xs = np.hstack((xs, [W]))
    else:
        xs[-1] = W
    ys = np.arange(0, H, h)
    if H - ys[-1] > h // 2:
        ys = np.hstack((ys, [H]))
    else:
        ys[-1] = H
        
    cross_points = np.array (np.meshgrid(xs, ys))
    patches = []
    for (y1, y2) in zip(ys[:-1], ys[1:]):
        for (x1, x2) in zip(xs[:-1], xs[1:]):
            patches.append(img[y1:y2, x1:x2, :])
    return cross_points.transpose(1,2,0), patches

def draw_grid(img, cross_points, grid_type='points'):
    img = img.copy()
    points = cross_points.reshape(-1, 2)
    if grid_type=='points':
        for (x,y) in points:
            cv2.circle(img, (x,y), 3, (255,0,0), -1)
    else:
        for x in set(points[:,0]):
            cv2.line(img, (x, 0), (x, img.shape[0]), (255,0,0), 3)
        for y in set(points[:,1]):
            cv2.line(img, (0, y), (img.shape[1], y), (255,0,0), 3)
    plt.title(img.shape)
    plt.imshow(img)
    plt.axis('off')


############################################################################
# For exploring class balance
############################################################################
def plot_ratio(ratio_array, title='Ratios', legend=['Tissues', 'Backgrounds']):
    ratio_array = np.array(ratio_array)
    mean_ratio = ratio_array.mean()
    num_observations = len(ratio_array)
    plt.title(title)
    # print(ratio_array)
    plt.bar(range(num_observations), ratio_array, color='c')
    plt.bar(range(num_observations), 1 - ratio_array, bottom=ratio_array, color='orange')
    plt.hlines(mean_ratio, 0, num_observations)
    plt.legend(['Mean {}'.format(legend[0])] + legend)
    

############################################################################
# Augmentations
############################################################################
import imgaug as ia
from imgaug import augmenters as iaa

def get_augs():
    iaa_Blur = iaa.OneOf([
                    iaa.GaussianBlur((0, 1.5)), # blur images with a sigma between 0 and 1.5
                    iaa.AverageBlur(k=(2, 4)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 3 and 5
    ])
    iaa_Sharpen = iaa.OneOf([
                    iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 0.5), strength=(0, 0.5)), # emboss images
    ])
    iaa_Noise = iaa.OneOf([
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03*255), per_channel=0.1), # add gaussian noise to images
                    iaa.Dropout((0, 0.02), per_channel=0.1), # randomly remove up to 10% of the pixels
                    #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
    ])
    iaa_Affine = iaa.OneOf([
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                        rotate=(-45, 45), # rotate by -45 to +45 degrees
                        shear=(-16, 16), # shear by -16 to +16 degrees
                        order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                        cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                        mode='edge' # use zeros for padding
                    ),
                    iaa.PiecewiseAffine(scale=(0.01, 0.05)), # move parts of the image around # is it fast?
                    iaa.PerspectiveTransform(scale=(0.01, 0.1)) # is it fast?
                    #iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25), # move pixels locally around (with random strengths) # we need borders
    ])
    iaa_HSV = iaa.OneOf([
                    iaa.Grayscale(alpha=(0.0, 0.5)),  
                    iaa.Add((-10, 10), per_channel=0.01), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    iaa.Sequential([
                       iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                       iaa.WithChannels(0, iaa.Add((-100, 100))),
                       iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
                    ]),
                    iaa.ContrastNormalization((0.7, 1.6), per_channel=0), # improve or worsen the contrast
                    iaa.Multiply((0.7, 1.2), per_channel=0.01),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-2, 2),
                        first=iaa.Multiply((0.85, 1.2), per_channel=False),
                        second=iaa.ContrastNormalization((0.5, 1.7))),
    ])
    iaa_Simple = iaa.OneOf([
                iaa.Fliplr(0.5), # horizontally flip 50% of all images
                iaa.Flipud(0.5), # vertically flip 20% of all images
                iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode='edge',
                    pad_cval=(0, 255)
                ),
    ])
        
    augs = {
        'simple': iaa_Simple,
        'affine': iaa_Affine,
        'hsv': iaa_HSV,
        'sharpen': iaa_Sharpen,
        'blur': iaa_Blur,
        'noise': iaa_Noise,
    }
    return augs

def channel_shuffle(img):
    ch_arr = [0, 1, 2]
    np.random.shuffle(ch_arr)
    img = img[..., ch_arr]
    return img

###############################################################################################################################
import torch
from torch import nn
from torch.nn import functional as F
import utils
import numpy as np


class LossBinary:
    """
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss
        
#############################################################################################################################
# For testing
#############################################################################################################################
from torchvision.transforms import ToTensor, Normalize, Compose
def mask_overlay(image, mask, color=(0, 255, 0)):
    """
    Helper function to visualize mask on the top of the car
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img

def load_image(path, pad=True):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)
    
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if not pad:
        return img
    
    height, width, _ = img.shape
    
    if height % 32 == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = 32 - height % 32
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad
        
    if width % 32 == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = 32 - width % 32
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad
    
    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)

img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.881, 0.865, 0.875],
              std=[0.485, 0.456, 0.406])
])

def crop_image(img, pads):
    """
    img: numpy array of the shape (height, width)
    pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    
    @return padded image
    """
    (x_min_pad, y_min_pad, x_max_pad, y_max_pad) = pads
    height, width = img.shape[:2] 
    
    return img[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]
    