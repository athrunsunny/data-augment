# -*- coding:utf-8 -*-
"""
Time:     2021.10.31
Author:   Athrunsunny
Version:  V 0.1
File:     enhance.py
"""
import cv2
import torch
import numpy as np
import warnings
import numbers
import math
import random
import json
import os
import base64
import io
import PIL.Image
from tqdm import tqdm
from glob import glob
from torch import Tensor
from PIL import Image, ImageDraw
from torchvision import transforms

from torchtoolbox.transform import Cutout
from typing import Tuple, List, Optional
from torchvision.transforms import functional as F

ROOT_DIR = os.getcwd()
VERSION = '4.5.7'  # 根据labelme的版本来修改
functionList = ['resize', 'resize_', 'random_flip_horizon', 'random_flip_vertical', 'center_crop', 'random_equalize',
                'random_autocontrast', 'random_adjustSharpness', 'random_solarize', 'random_posterize',
                'random_grayscale', 'gaussian_blur', 'random_invert', 'random_cutout', 'random_erasing',
                'random_bright', 'random_contrast', 'random_saturation', 'add_gasuss_noise', 'add_salt_noise',
                'add_pepper_noise', 'mixup', 'random_perspective', 'random_rotate']


class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        return F.hflip(img)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        return F.vflip(img)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomErasing(torch.nn.Module):
    def __init__(self, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__()
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError("Argument value should be either a number or str or a sequence")
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")

        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(
            img: Tensor, scale: Tuple[float, float], ratio: Tuple[float, float], value: Optional[List[float]] = None
    ) -> Tuple[int, int, int, int, Tensor]:
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def forward(self, img):
        # cast self.value to script acceptable type
        if isinstance(self.value, (int, float)):
            value = [self.value, ]
        elif isinstance(self.value, str):
            value = None
        elif isinstance(self.value, tuple):
            value = list(self.value)
        else:
            value = self.value

        if value is not None and not (len(value) in (1, img.shape[-3])):
            raise ValueError(
                "If value is a sequence, it should have either a single value or "
                "{} (number of input channels)".format(img.shape[-3])
            )

        x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
        return F.erase(img, x, y, h, w, v, self.inplace)


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def random_perspective(im, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        new = np.zeros((n, 4))
        # warp boxes
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets


def cutout(im, labels):
    h, w = im.shape[:2]
    scales = [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction [0.5] * 1 +
    for s in scales:
        mask_h = random.randint(1, int(h * s))  # create random masks
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

    return im, labels


class DataAugmentation(object):
    to_tensor = transforms.ToTensor()
    to_image = transforms.ToPILImage()

    def __init__(self):
        super(DataAugmentation, self).__init__()
        self.transforms = transforms

    def resize(self, img, boxes, size):
        """
        将图像长和宽缩放到指定值size，并且相应调整boxes
        :param img: Image
        :param boxes: bbox坐标
        :param size:缩放大小
        :return:
        """
        w, h = img.size
        sw = size / w
        sh = size / h
        label, boxes = boxes[:, :1], boxes[:, 1:5]
        boxes = boxes * torch.Tensor([sw, sh, sw, sh])
        boxes = torch.cat((label, boxes), dim=1)
        return img.resize((size, size), Image.BILINEAR), boxes

    def resize_(self, img, boxes, size):
        """
        将图像短边缩放到指定值size,保持原有比例不变，并且相应调整boxes
        :param img: Image
        :param boxes: bbox坐标
        :param size:缩放大小
        :return:
        """
        w, h = img.size
        # min_size = min(w, h)
        # sw = sh = size / min_size
        sw = size[0] / w
        sh = size[1] / h
        ow = int(sw * w + 0.5)
        oh = int(sh * h + 0.5)
        label, boxes = boxes[:, :1], boxes[:, 1:5]
        boxes = boxes * torch.Tensor([sw, sh, sw, sh])
        boxes = torch.cat((label, boxes), dim=1)
        return img.resize((ow, oh), Image.BILINEAR), boxes

    def random_flip_horizon(self, img, boxes):
        """
        Horizontally flip the given image randomly with a given probability.
        :param img: Image
        :param boxes: bbox坐标
        :return:
        """
        p = torch.rand(1)
        if p > 0.5:
            transform = RandomHorizontalFlip()
            img = transform(img)
            w = img.width
            label, boxes = boxes[:, :1], boxes[:, 1:5]
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            boxes = torch.cat((label, boxes), dim=1)
        return img, boxes

    def random_flip_vertical(self, img, boxes):
        """
        Vertically flip the given image randomly with a given probability.
        :param img: Image
        :param boxes: bbox坐标
        :return:
        """
        p = torch.rand(1)
        if p > 0.5:
            transform = RandomVerticalFlip()
            img = transform(img)
            h = img.height
            label, boxes = boxes[:, :1], boxes[:, 1:5]
            ymin = h - boxes[:, 3]
            ymax = h - boxes[:, 1]
            boxes[:, 1] = ymin
            boxes[:, 3] = ymax
            boxes = torch.cat((label, boxes), dim=1)
        return img, boxes

    def center_crop(self, img, boxes, size=(600, 600)):
        """
        中心裁剪
        :param img: Image
        :param boxes: bbox坐标
        :param size: 裁剪大小（w,h）
        :return:
        """
        w, h = img.size
        ow, oh = size
        max_size = torch.as_tensor([ow - 1, oh - 1], dtype=torch.float32)
        i = int(round((h - oh) / 2.))
        j = int(round((w - ow) / 2.))
        img = img.crop((j, i, j + ow, i + oh))
        label, boxes = boxes[:, :1], boxes[:, 1:5]
        boxes = boxes - torch.Tensor([j, i, j, i])
        boxes = torch.min(boxes.reshape(-1, 2, 2), max_size)
        boxes = boxes.clamp(min=0).reshape(-1, 4)
        boxes = torch.cat((label, boxes), dim=1)
        return img, boxes

    def random_equalize(self, img, boxes, p=0.5):
        """
        Equalize the histogram of the given image randomly with a given probability.
        :param img: Image
        :param boxes: bbox坐标
        :param p:probability of the image being equalized
        :return:
        """
        transform = self.transforms.RandomEqualize(p=p)
        img = transform(img)
        return img, boxes

    def random_autocontrast(self, img, boxes, p=0.5):
        """
        Autocontrast the pixels of the given image randomly with a given probability.
        :param img: Image
        :param boxes: bbox坐标
        :param p:probability of the image being autocontrasted
        :return:
        """
        transform = self.transforms.RandomAutocontrast(p=p)
        img = transform(img)
        return img, boxes

    def random_adjustSharpness(self, img, boxes, sharpness_factor=1, p=0.5):
        """
        Adjust the sharpness of the image randomly with a given probability.
        :param img: Image
        :param boxes: bbox坐标
        :param sharpness_factor:How much to adjust the sharpness
        :param p:probability of the image being color inverted
        :return:
        """
        transform = self.transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor, p=p)
        img = transform(img)
        return img, boxes

    def random_solarize(self, img, boxes, threshold=1, p=0.5):
        """
        Solarize the image randomly with a given probability by inverting all pixel values above a threshold.
        :param img: Image
        :param boxes: bbox坐标
        :param threshold:all pixels equal or above this value are inverted
        :param p:probability of the image being color inverted
        :return:
        """
        transform = self.transforms.RandomSolarize(threshold=threshold, p=p)
        img = transform(img)
        return img, boxes

    def random_posterize(self, img, boxes, bits=0, p=0.5):
        """
        Posterize the image randomly with a given probability by reducing the number of bits for each color channel.
        :param img: Image
        :param boxes: bbox坐标
        :param bits:number of bits to keep for each channel (0-8)
        :param p:probability of the image being color inverted
        :return:
        """
        transform = self.transforms.RandomPosterize(bits=bits, p=p)
        img = transform(img)
        return img, boxes

    def random_grayscale(self, img, boxes, p=0.5):
        """
        Randomly convert image to grayscale with a probability of p (default 0.1).
        :param img: Image
        :param boxes: bbox坐标
        :param p:Grayscale version of the input image with probability p and unchanged with probability (1-p).
        :return:
        """
        transform = self.transforms.RandomGrayscale(p=p)
        img = transform(img)
        return img, boxes

    def gaussian_blur(self, img, boxes, kernel_size=5, sigma=(0.1, 2.0)):
        """
        Blurs image with randomly chosen Gaussian blur.
        :param img: Image
        :param boxes: bbox坐标
        :param kernel_size:Size of the Gaussian kernel
        :param sigma:Standard deviation to be used for creating kernel to perform blurring.
        :return:
        """
        transform = self.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        img = transform(img)
        return img, boxes

    def random_invert(self, img, boxes, p=0.5):
        """
        Inverts the colors of the given image randomly with a given probability.
        :param img: Image
        :param boxes: bbox坐标
        :param p:probability of the image being color inverted
        :return:
        """
        transform = self.transforms.RandomInvert(p=p)
        img = transform(img)
        return img, boxes

    def random_cutout_(self, img, boxes, p=0.5, scale=(0.02, 0.4), ratio=(0.4, 1 / 0.4), value=(0, 255),
                       pixel_level=False, inplace=False):
        """
        Random erase the given CV Image
        :param img: Image
        :param boxes: bbox坐标
        :param p:probability that the random erasing operation will be performed
        :param scale:range of proportion of erased area against input image
        :param ratio:range of aspect ratio of erased area
        :param value:erasing value
        :param pixel_level:filling one number or not. Default value is False
        :param inplace:boolean to make this transform inplace. Default set to False
        :return:
        """
        transform = Cutout(p=p, scale=scale, ratio=ratio, value=value, pixel_level=pixel_level, inplace=inplace)
        img = transform(img)
        return img, boxes

    def random_cutout(self, img, boxes):
        img = np.array(img)
        img, boxes = cutout(img, boxes)
        img = Image.fromarray(img)
        return img, boxes

    def random_rotate(self, img, boxes, degrees=5, expand=False, center=None, fill=0, resample=None):
        degree = torch.randint(0, degrees + 1, (1,))
        degree = degree.item()
        transform = self.transforms.RandomRotation(degrees=degree, expand=expand, center=center, fill=fill,
                                                   resample=resample)
        img = transform(img)
        return img, boxes

    def random_perspective(self, img, boxes, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                           border=(0, 0)):
        img = np.array(img)
        img, boxes = random_perspective(img, boxes.numpy(), degrees=degrees, translate=translate, scale=scale,
                                        shear=shear, perspective=perspective, border=border)
        img = Image.fromarray(img)
        return img, torch.from_numpy(boxes)

    def random_erasing(self, img, boxes, count=3, scale=0.01, ratio=0.4, value=0, inplace=False):
        """
        Randomly selects a rectangle region in an torch Tensor image and erases its pixels.
        :param img: Image
        :param boxes: bbox坐标
        :param scale:range of proportion of erased area against input image
        :param ratio:range of aspect ratio of erased area
        :param value:erasing value
        :param inplace:boolean to make this transform inplace. Default set to False
        :return:
        """
        scale = (scale, scale)
        ratio = (ratio, 1. / ratio)
        if count != 0:
            for num in range(count):
                transform = RandomErasing(scale=scale, ratio=ratio, value=value, inplace=inplace)
                img = transform(self.to_tensor(img))
                img = self.to_image(img)
            return img, boxes
        transform = RandomErasing(scale=scale, ratio=ratio, value=value, inplace=inplace)
        img = transform(self.to_tensor(img))
        return self.to_image(img), boxes

    def random_bright(self, img, boxes, u=32):
        """
        随机亮度
        :param img: Image
        :param boxes: bbox坐标
        :param u:
        :return:
        """
        img = self.to_tensor(img)
        alpha = np.random.uniform(-u, u) / 255
        img += alpha
        img = img.clamp(min=0.0, max=1.0)
        return self.to_image(img), boxes

    def random_contrast(self, img, boxes, lower=0.5, upper=1.5):
        """
        随机对比度
        :param img: Image
        :param boxes: bbox坐标
        :param lower:
        :param upper:
        :return:
        """
        img = self.to_tensor(img)
        alpha = np.random.uniform(lower, upper)
        img *= alpha
        img = img.clamp(min=0, max=1.0)
        return self.to_image(img), boxes

    def random_saturation(self, img, boxes, lower=0.5, upper=1.5):
        """
        随机饱和度
        :param img: Image
        :param boxes: bbox坐标
        :param lower:
        :param upper:
        :return:
        """
        img = self.to_tensor(img)
        alpha = np.random.uniform(lower, upper)
        img[1] = img[1] * alpha
        img[1] = img[1].clamp(min=0, max=1.0)
        return self.to_image(img), boxes

    def add_gasuss_noise(self, img, boxes, mean=0, std=0.1):
        """
        随机高斯噪声
        :param img: Image
        :param boxes: bbox坐标
        :param mean:
        :param std:
        :return:
        """
        img = self.to_tensor(img)
        noise = torch.normal(mean, std, img.shape)
        img += noise
        img = img.clamp(min=0, max=1.0)
        return self.to_image(img), boxes

    def add_salt_noise(self, img, boxes):
        """
        随机盐噪声
        :param img: Image
        :param boxes: bbox坐标
        :return:
        """
        img = self.to_tensor(img)
        noise = torch.rand(img.shape)
        alpha = np.random.random()
        img[noise[:, :, :] > alpha] = 1.0
        return self.to_image(img), boxes

    def add_pepper_noise(self, img, boxes):
        """
        随机椒噪声
        :param img: Image
        :param boxes: bbox坐标
        :return:
        """
        img = self.to_tensor(img)
        noise = torch.rand(img.shape)
        alpha = np.random.random()
        img[noise[:, :, :] > alpha] = 0
        return self.to_image(img), boxes

    def mixup(self, img1, img2, box1, box2, alpha=32.):
        """
        mixup
        :param img1: Image
        :param img2: Image
        :param box1: bbox1坐标
        :param box2: bbox2坐标
        :param alpha:
        :return:
        """
        p = torch.rand(1)
        if p > 0.5:
            max_w = max(img1.size[0], img2.size[0])
            max_h = max(img1.size[1], img2.size[1])
            img1, box1 = self.resize_(img1, box1, (max_w, max_h))
            img2, box2 = self.resize_(img2, box2, (max_w, max_h))

            img1 = self.to_tensor(img1)
            img2 = self.to_tensor(img2)
            weight = np.random.beta(alpha, alpha)
            miximg = weight * img1 + (1 - weight) * img2
            return self.to_image(miximg), torch.cat([box1, box2])
        return img1, box1

    def draw_img(self, img, boxes):
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle(list(box[1:]), outline='yellow', width=2)
        img.show()


def load_json_points(file, cls):
    assert isinstance(file, str)
    with open(file, 'r', encoding="utf-8") as f:
        doc = json.load(f)
    # point = [item['points'][0] + item['points'][1] for item in doc['shapes']]
    point = [[cls.index(item['label'])] + item['points'][0] + item['points'][1] for item in doc['shapes']]
    return torch.tensor(point)


def img_arr_to_b64(img_arr):
    img_pil = PIL.Image.fromarray(img_arr)
    f = io.BytesIO()
    img_pil.save(f, format="PNG")
    img_bin = f.getvalue()
    if hasattr(base64, "encodebytes"):
        img_b64 = base64.encodebytes(img_bin)
    else:
        img_b64 = base64.encodestring(img_bin)
    return img_b64


def create_json(img, imagePath, filename, cls, points):
    data = dict()
    data['version'] = VERSION
    data['flags'] = dict()
    info = list()
    for point in points:
        shape_info = dict()
        shape_info['label'] = cls[int(point[0].item())]
        if point is None:
            shape_info['points'] = [[], []]
        else:
            shape_info['points'] = [[point[1].item(), point[2].item()],
                                    [point[3].item(), point[4].item()]]
        shape_info['group_id'] = None
        shape_info['shape_type'] = 'rectangle'
        shape_info['flags'] = dict()
        info.append(shape_info)
    data['shapes'] = info
    data['imagePath'] = imagePath
    height, width = img.shape[:2]
    data['imageData'] = img_arr_to_b64(img).decode('utf-8')
    data['imageHeight'] = height
    data['imageWidth'] = width
    jsondata = json.dumps(data, indent=4, separators=(',', ': '))
    f = open(filename, 'w', encoding="utf-8")
    f.write(jsondata)
    f.close()


def get_all_class(files):
    classes = list()
    for filename in files:
        json_file = json.load(open(filename, "r", encoding="utf-8"))
        for item in json_file["shapes"]:
            label_class = item['label']
            if label_class not in classes:
                classes.append(label_class)
    return classes


def create_datasets(method, extimes=1, path=ROOT_DIR):
    """
    扩充数据集
    # 使用时将py放置在labelme生成的数据集下运行，增强后的图像会保存在‘create’文件夹中
    :param method:数据增强方法
    :param extimes: 数据增强之后需要生成的图像数量
    :param path: 保存路径，默认当前文件夹
    :return:
    """
    if 'mixup' in method:
        method.remove('mixup')
        method.insert(len(method)-1, 'mixup')
    classname = 'DataAugmentation()'
    files = glob(path + "\\*.json")
    cls = get_all_class(files)
    files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]
    externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
    imgfiles = list()
    for extern in externs:
        imgfiles.extend(glob(path + "\\*." + extern))

    for imgfile in tqdm(imgfiles):
        filename = '.'.join(imgfile.split('.')[:-1])
        imgfilename = filename.replace("\\", "/").split("/")[-1]
        if imgfilename in files:
            jsonpath = filename + '.json'

            for t in range(extimes):
                image = Image.open(imgfile)
                points = load_json_points(jsonpath, cls)

                for index, funcname in enumerate(method):
                    func = classname + '.' + funcname
                    if funcname == 'mixup':
                        imagepath = np.random.choice(imgfiles, size=1, replace=False)[0]
                        filename1 = '.'.join(imagepath.split('.')[:-1])
                        jsonpath1 = filename1 + '.json'
                        points1 = load_json_points(jsonpath1, cls)
                        image1 = Image.open(imagepath)
                        image, points = eval(func)(image, image1, points, points1)
                    else:
                        image, points = eval(func)(image, points)

                # for viz
                # print(points)
                # d = classname + '.' + 'draw_img'
                # eval(d)(image, points)

                new_name = str(t) + '_' + imgfilename
                new_image = os.path.join(ROOT_DIR, 'create', new_name + '.jpg')
                if not os.path.exists(os.path.join(ROOT_DIR, 'create')):
                    os.makedirs(os.path.join(ROOT_DIR, 'create'))
                image.save(new_image)

                new_json = os.path.join(ROOT_DIR, 'create', new_name + '.json')
                create_json(np.array(image), new_name, new_json, cls, points)


if __name__ == '__main__':
    """
    将该python文件放在数据集路径下运行，生成的图像保存在当前路径下的create目录中
    meth填想要的图像增强方法，所有方法已列在functionList中
    extimes表示扩充倍数
    """
    meth = ['random_rotate', 'random_flip_horizon', 'gaussian_blur', 'mixup', 'random_perspective', 'random_cutout']
    create_datasets(method=meth, extimes=3)
