#!/usr/bin/python
# encoding: utf-8
# ---------------------
# using Linemod dataset for training and testing
# @Author: Fan, Mo
# ---------------------

import os
import random
import numpy as np
from PIL import Image, ImageChops, ImageMath

import config as cfg
from utils.utils import (
    get_all_files,
    read_data_cfg,
)

class Linemod(object):
    def __init__(self, phase, arg=None):
        # Set parameters for training and testing
        self.data_options = read_data_cfg(arg)
        self.trainlist    = self.data_options['train']
        self.testlist     = self.data_options['valid']
        self.meshname     = self.data_options['mesh']
        self.backupdir    = self.data_options['backup']
        self.diam         = float(self.data_options['diam'])
        self.dataset_name = self.data_options['name']
        self.vx_threshold = self.diam * 0.1

        self.phase        = phase
        self.datasets_dir = os.path.join('LINEMOD', self.dataset_name)
        self.batch_size   = cfg.BATCH_SIZE
        self.image_size   = cfg.IMAGE_SIZE
        self.image_width  = 640   # axis x
        self.image_height = 480   # axis y
        self.bg_files     = None

        self.cell_size       = cfg.CELL_SIZE
        self.boxes_per_cell  = cfg.BOXES_PER_CELL
        self.num_classes     = cfg.NUM_CLASSES
        self.shuffle         = cfg.SHUFFLE
        self.train_imgname   = None
        self.test_imgname    = None
        self.imgname         = None
        self.train_gt_labels = None
        self.test_gt_labels  = None
        self.gt_labels       = None
        self.batch           = 0
        print("\n---------------Loading dataset---------------")
        self.prepare(self.phase)  # get the image files name and label files name
        # print(len(self.bg_files))
        print("----------Loading dataset complete-----------\n")

    def prepare(self, phase):
        """
        self.imgname:   A list of all training image files path
        self.gt_labels: A list of all ground true labels(which elements are lists like [[1,xx,xx,...],[1,xx,xx,...]] with integer and float numbers)
        these two matched respectively
        self.bg_files: A list of VOC images path
        """
        if phase == 'train':
            with open(self.trainlist, 'r') as f:
                self.imgname = [x.strip() for x in f.readlines()]  # a list of trianing files
            if self.shuffle:
                random.shuffle(self.imgname)
            self.bg_files = get_all_files('VOCdevkit/VOC2012/JPEGImages')
        elif phase == 'test':
            with open(self.testlist, 'r') as f:
                self.imgname = [x.strip() for x in f.readlines()]
        else:
            raise Exception('\n   Wrong phase...\n   Try again...')

    def next_batches(self):
        images   = np.zeros((self.batch_size, 416, 416, 3), np.float32)
        labels   = np.zeros((self.batch_size, 13, 13, 19), np.float32)
        # some parameters for data augmentation
        jitter     = 0.2
        hue        = 0.1
        saturation = 1.5
        exposure   = 1.5

        bgpath = self.bg_files[random.randint(0, len(self.bg_files) - 1)]

        for idx in range(self.batch_size):
            images[idx] = self.load_data_detection(self.imgname[idx + self.batch * self.batch_size], (416, 416),
                                                   jitter, hue, saturation, exposure, bgpath)
            labels[idx] = self.get_label(self.imgname[idx + self.batch * self.batch_size])

        images   = np.array(images) # nB X 416 X 416 X 3
        labels   = np.array(labels)   # nB X 13 X 13 X 20

        self.batch += 1
        return images, labels

    def get_label(self, label):
        # label: [21, ]
        output = np.zeros([13, 13, 20], np.float32)
        nW, nH = 13, 13
        x0  = label[1] * nW
        y0  = label[2] * nH
        gx0 = int(x0)
        gy0 = int(y0)
        x1  = label[3] * nW
        y1  = label[4] * nH
        x2  = label[5] * nW
        y2  = label[6] * nH
        x3  = label[7] * nW
        y3  = label[8] * nH
        x4  = label[9] * nW
        y4  = label[10] * nH
        x5  = label[11] * nW
        y5  = label[12] * nH
        x6  = label[13] * nW
        y6  = label[14] * nH
        x7  = label[15] * nW
        y7  = label[16] * nH
        x8  = label[17] * nW
        y8  = label[18] * nH
        cls = label[0]

        output[gx0][gy0][0]  = 1
        output[gx0][gy0][1]  = x0 - gx0
        output[gx0][gy0][2]  = y0 - gy0
        output[gx0][gy0][3]  = x1 - gx0
        output[gx0][gy0][4]  = y1 - gy0
        output[gx0][gy0][5]  = x2 - gx0
        output[gx0][gy0][6]  = y2 - gy0
        output[gx0][gy0][7]  = x3 - gx0
        output[gx0][gy0][8]  = y3 - gy0
        output[gx0][gy0][9]  = x4 - gx0
        output[gx0][gy0][10] = y4 - gy0
        output[gx0][gy0][11] = x5 - gx0
        output[gx0][gy0][12] = y5 - gy0
        output[gx0][gy0][13] = x6 - gx0
        output[gx0][gy0][14] = y6 - gy0
        output[gx0][gy0][15] = x7 - gx0
        output[gx0][gy0][16] = y7 - gy0
        output[gx0][gy0][17] = x8 - gx0
        output[gx0][gy0][18] = y8 - gy0
        output[gx0][gy0][19] = cls

        return output

    # ======================= data augmentation =========================
    def distort_image(self, im, hue, sat, val):
        im = im.convert('HSV')
        cs = list(im.split())
        cs[1] = cs[1].point(lambda i: i * sat)
        cs[2] = cs[2].point(lambda i: i * val)

        def change_hue(x):
            x += hue*255
            if x > 255:
                x -= 255
            if x < 0:
                x += 255
            return x
        cs[0] = cs[0].point(change_hue)
        im = Image.merge(im.mode, tuple(cs))

        im = im.convert('RGB')
        return im

    def rand_scale(self, s):
        scale = random.uniform(1, s)
        if(random.randint(1,10000)%2):
            return scale
        return 1./scale

    def random_distort_image(self, im, hue, saturation, exposure):
        dhue = random.uniform(-hue, hue)
        dsat = self.rand_scale(saturation)
        dexp = self.rand_scale(exposure)
        res  = self.distort_image(im, dhue, dsat, dexp)
        return res

    def data_augmentation(self, img, shape, jitter, hue, saturation, exposure):

        ow, oh = img.size

        dw =int(ow*jitter)
        dh =int(oh*jitter)

        pleft  = random.randint(-dw, dw)
        pright = random.randint(-dw, dw)
        ptop   = random.randint(-dh, dh)
        pbot   = random.randint(-dh, dh)

        swidth =  ow - pleft - pright
        sheight = oh - ptop - pbot

        sx = float(swidth)  / ow
        sy = float(sheight) / oh

        flip = random.randint(1,10000)%2
        cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

        dx = (float(pleft)/ow)/sx
        dy = (float(ptop) /oh)/sy

        sized = cropped.resize(shape)

        img = self.random_distort_image(sized, hue, saturation, exposure)

        return img, flip, dx, dy, sx, sy

    def change_background(self, img, mask, bg):
        # oh = img.height
        # ow = img.width
        ow, oh = img.size
        bg = bg.resize((ow, oh)).convert('RGB')

        imcs = list(img.split())
        bgcs = list(bg.split())
        maskcs = list(mask.split())
        fics = list(Image.new(img.mode, img.size).split())

        for c in range(len(imcs)):
            negmask = maskcs[c].point(lambda i: 1 - i / 255)
            posmask = maskcs[c].point(lambda i: i / 255)
            fics[c] = ImageMath.eval("a * c + b * d", a=imcs[c], b=bgcs[c], c=posmask, d=negmask).convert('L')
        out = Image.merge(img.mode, tuple(fics))

        return out

    def load_data_detection(self, imgpath, shape, jitter, hue, saturation, exposure, bgpath):
        maskpath = imgpath.replace('JPEGImages', 'mask').replace('/00', '/').replace('.jpg', '.png')

        ## data augmentation
        img = Image.open(imgpath).convert('RGB')
        mask = Image.open(maskpath).convert('RGB')
        bg = Image.open(bgpath).convert('RGB')

        img = self.change_background(img, mask, bg)
        img, flip, dx, dy, sx, sy = self.data_augmentation(img, shape, jitter, hue, saturation, exposure)
        ow, oh = img.size
        return img

if __name__ == "__main__":
    data = Linemod('train', 'cfg'+os.sep+'ape.data')
    data.next_batches()