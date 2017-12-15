#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: fcn_hd.py, read images from harddisk during training
# Author: Chengyao Shen scyscyao@gmail.com>

import os
import glob
import cv2
import numpy as np
import random

from ...utils.fs import download, get_dataset_path
from ..base import RNGDataFlow
try:
    from scipy.io import loadmat
    __all__ = ['FCN_HD']
except:
    logger.warn_dependency('FCN_HD', 'scipy.io')
    __all__ = []

IMG_W = 640
IMG_H = 480

class FCN_HD(RNGDataFlow):

    def __init__(self, name, data_dir=None, shuffle=True):
        if data_dir is None:
            data_dir = get_dataset_path('fcn_data_hd')
        else:
            data_dir = get_dataset_path(data_dir)
        self.data_root = data_dir
        assert os.path.isdir(self.data_root)

        self.shuffle = shuffle
        assert name in ['train', 'test', 'val']
        self._load(name)

    def _load(self, name):
        image_glob = os.path.join(self.data_root, name, 'images', '*.*g')
        self.imglist = glob.glob(image_glob)
        self.gt_dir = os.path.join(self.data_root, name, 'labels')

    def size(self):
    	return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
    		self.rng.shuffle(idxs)

        for k in idxs:
            fname = self.imglist[k]
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = cv2.resize(im,(IMG_W, IMG_H))
            # pts1 = np.float32([[0, 0], [IMG_W, 0], [0, IMG_H], [IMG_W, IMG_H]])
            # pts2 = np.float32([[0, -IMG_H], [IMG_W, 0], [0, 2*IMG_H], [IMG_W, IMG_H]])
            # M = cv2.getPerspectiveTransform(pts1, pts2)
            # dst = cv2.warpPerspective(im, M,(IMG_W, IMG_H))
            # cv2.imshow('im',im)
            # cv2.imshow('warped',dst)
            # cv2.waitKey(0)
            imgid = os.path.basename(fname)
            gt_file = os.path.join(self.gt_dir, imgid.split('.')[0]+'.png')
            gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(gt,(IMG_W, IMG_H),interpolation=cv2.INTER_NEAREST)

            yield [im, label]


if __name__ == '__main__':
	d = FCN_HD('val')
	for img, label in d.get_data():
		cv2.imshow('img', img)
		cv2.imshow('label', label*255)
		cv2.waitKey(1000)
