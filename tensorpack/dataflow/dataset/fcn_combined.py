#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: fcn.py
# Author: Yue Wang <valianter.wang@gmail.com>

import os
import glob
import cv2
import numpy as np
import random
from PIL import Image


from ...utils.fs import download, get_dataset_path
from ..base import RNGDataFlow
try:
    from scipy.io import loadmat
    __all__ = ['FCN_COMBINED']
except:
    logger.warn_dependency('FCN_COMBINED', 'scipy.io')
    __all__ = []

IMG_W = 640
IMG_H = 480

class FCN_COMBINED(RNGDataFlow):

    def __init__(self, name, data_dir=None, shuffle=True):
        if data_dir is None:
            data_dir = get_dataset_path('fcn_line_road')
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
        self.gt_dir0 = os.path.join(self.data_root, name, 'labels_line')
        self.gt_dir1 = os.path.join(self.data_root, name, 'labels_road')

        # self.data = np.zeros((len(image_files), IMG_H, IMG_W, 3), dtype='uint8')
        # self.label = np.zeros((len(image_files), IMG_H, IMG_W), dtype='uint8')
        #
        # for idx, f in enumerate(image_files):
        #     im = cv2.imread(f, cv2.IMREAD_COLOR)
        #     assert im is not None
        #     im = cv2.resize(im,(IMG_W, IMG_H))
        #     imgid = os.path.basename(f)
        #     gt_file = os.path.join(gt_dir, imgid)
        #     gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        #     gt = cv2.resize(gt,(IMG_W, IMG_H))
        #     # if gt.shape[0] > gt.shape[1]:
        #     # 	gt = gt.transpose()
        #     # assert gt.shape == (IMG_H, IMG_W)
        #     self.data[idx] = im
        #     self.label[idx] = gt

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
            gt_file0 = os.path.join(self.gt_dir0, imgid.split('.')[0]+'.png')
            gt0 = cv2.imread(gt_file0, cv2.IMREAD_GRAYSCALE)
            gt_file1 = os.path.join(self.gt_dir1, imgid.split('.')[0]+'.png')
            gt1 = cv2.imread(gt_file1, cv2.IMREAD_GRAYSCALE)
            # label_image = Image.open(gt_file)
            # label_array = np.array(label_image)

            label = cv2.resize(gt0,(IMG_W, IMG_H),interpolation=cv2.INTER_NEAREST)
            label1 = cv2.resize(gt1,(IMG_W, IMG_H),interpolation=cv2.INTER_NEAREST)
            mask = np.logical_and(label1>128, label==0)
            label[mask]=5
            yield [im, label]


if __name__ == '__main__':
	d = FCN_COMBINED('val')
	for img, label in d.get_data():
		cv2.imshow('img', img)
		cv2.imshow('label', label*255)
		cv2.waitKey(1000)
