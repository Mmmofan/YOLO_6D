import os
import yolo.config as cfg
import numpy as np
import cv2

class Linemod(object):
    def __init__(self, name):
        self.name = name
        self.batch_size = cfg.BATCH_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.train_list = None
        self.test_list = None
        self.data_path = 'linemod/'
        self.config_path = os.path.join(self.data_path, 'cfg/')
        self.label_path = os.path.join(self.config_path, self.name)
        self.image_path = os.path.join(self.data_path, self.name)
        self.train_file = None
        self.test_file = None
        self.label_id = None
        self.count = 0
        self.epoch = 0
        self.epoch_test = 0
        self.prepare()
        self.total_train_num = len(self.train_list)
        self.total_test_num = len(self.test_list)

    def prepare(self):
        self.train_file = os.path.join(self.label_path, 'train.txt')
        self.test_file = os.path.join(self.label_path, 'test.txt')
        with open(self.train_file, 'r') as train:
            self.train_list = [x.split() for x in train.readlines()]
        self.label_id = int(self.train_list[0][1])
        self.train_list = [self.train_list[x][0] for x in range(len(self.train_list))]

        with open(self.test_file, 'r') as test:
            self.test_list = [x.split() for x in test.readlines()]
        self.test_list = [self.test_list[x][0] for x in range(len(self.test_list))]

    def load_labels(self, phase):
        if phase == 'train':
            return self.train_list
        elif phase == 'test':
            return self.test_list

    def next_batches(self, label):
        images = np.zeros([self.batch_size, 416, 416, 3], dtype=np.float32)
        labels = np.zeros([self.batch_size, 15], dtype=np.float32)
        batch_list = label[self.epoch * self.batch_size : (self.epoch+1) * self.batch_size]
        num = 0
        while num < self.batch_size:
            #file_path = os.path.join(self.image_path, batch_list[num])
            images[num, :, :, :] = self.image_read(batch_list[num])
            label[num] = self.label_read()
            num += 1
            self.count += 1
            self.epoch += 1
        return images, labels

    def next_batches_test(self, label):
        images = np.zeros([self.batch_size, 416, 416, 3], dtype=np.float32)
        labels = np.zeros([self.batch_size, 15], dtype=np.float32)
        batch_list = label[self.epoch_test * self.batch_size : (self.epoch_test+1) * self.batch_size]
        num = 0
        while num < self.batch_size:
            #file_path = os.path.join(self.image_path, batch_list[num])
            images[num, :, :, :] = self.image_read(batch_list[num])
            label[num] = self.label_read()
            num += 1
            self.epoch_test += 1
        return images, labels

    def image_read(self, imagename):
        image = cv2.imread(imagename)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image / 255.0
        return image

    def label_read(self):
        label = np.zeros([1, 15], dtype=np.float32)
        label[0, self.label_id] = 1
        return label
