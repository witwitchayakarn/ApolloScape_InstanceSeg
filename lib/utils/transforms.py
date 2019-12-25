# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision.transforms import functional as F

from albumentations import (RandomBrightnessContrast, HueSaturationValue, RandomGamma,
                            CLAHE, Blur, GaussNoise,
                            ChannelShuffle, RGBShift, ChannelDropout,
                            RandomFog, RandomRain, RandomSnow, RandomShadow, RandomSunFlare)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class RandomTransformPixels(object):
    def __init__(self):
        self.random_brightness_contrast = RandomBrightnessContrast()
        self.hue_saturation_value = HueSaturationValue()
        self.random_gamma = RandomGamma()
        self.clahe = CLAHE()

        self.blur = Blur()
        self.gauss_noise = GaussNoise()

        self.channel_shuffle = ChannelShuffle()
        self.rgb_shift = RGBShift()
        self.channel_dropout = ChannelDropout()

        self.random_fog = RandomFog(fog_coef_upper=0.4)
        self.random_rain = RandomRain()
        self.random_snow = RandomSnow()
        self.random_shadow = RandomShadow()
        self.random_sunflare = RandomSunFlare(angle_upper=0.2)

    def __call__(self, image, target):
        image = np.array(image)

        image = self.random_brightness_contrast(image=image)['image']
        image = self.hue_saturation_value(image=image)['image']
        image = self.random_gamma(image=image)['image']
        image = self.clahe(image=image)['image']

        if random.random() < 0.5:
            image = self.blur(image=image)['image']
        else:
            image = self.gauss_noise(image=image)['image']

        r = random.choice(list(range(4)))
        if r == 1:
            image = self.channel_shuffle(image=image)['image']
        elif r == 2:
            image = self.rgb_shift(image=image)['image']
        elif r == 3:
            image = self.channel_dropout(image=image)['image']
        else:
            pass

        r = random.choice(list(range(6)))
        if r == 1:
            image = self.random_fog(image=image)['image']
        elif r == 2:
            image = self.random_rain(image=image)['image']
        elif r == 3:
            image = self.random_snow(image=image)['image']
        elif r == 4:
            image = self.random_shadow(image=image)['image']
        elif r == 5:
            image = self.random_sunflare(image=image)['image']
        else:
            pass

        return Image.fromarray(image), target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
