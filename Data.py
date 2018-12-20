"""
Get the training & testing datasets and set the labels
"""
import os
import random
from multiprocessing import Process, Queue

import tensorflow as tf

import config as cfg


class Data(object):
    def __init__(self):
        self.max_num = 1000
        self.tfrecords_file_num = 1
        self.makecsv()
        with open('imagenet_trian.csv', 'r') as trainfile:
            trainrecords = trainfile.readlines()
        self.total_file_num = len(trainrecords)

        # for 4 cpus, each process 1 lsit
        self.each_process_file_num = int(self.total_file_num / 4.0)
        list1 = trainrecords[:self.each_process_file_num]
        lsit2 = trainrecords[self.each_process_file_num:2*self.each_process_file_num]
        list3 = trainrecords[2*self.each_process_file_num:3*self.each_process_file_num]
        list4 = trainrecords[3*self.each_process_file_num:]

        # create 4 queues, 4 sub processes
        q1 = Queue()
        q2 = Queue()
        q3 = Queue()
        q4 = Queue()
        p1 = Process(target=gen_tfrecord, args=(list1, q1, ))
        p2 = Process(target=gen_tfrecord, args=(lsit2, q2, ))
        p3 = Process(target=gen_tfrecord, args=(list3, q3, ))
        p4 = Process(target=gen_tfrecord, args=(list4, q4, ))
        p_list = [p1, p2, p3, p4]
        _ = map(Process.start, p_list)
        
        #父进程循环查询队列的消息，并且每10秒更新一次
        progress_str = 'PID:%i Processing:%i/%i | PID:%i Processing:%i/%i | PID:%i Processing:%i/%i | PID:%i Processing:%i/%i \r'
        while(True):
            try:
                msg1 = q1.get()
                msg2 = q2.get()
                msg3 = q3.get()
                msg4 = q4.get()
                print(progress_str % (msg1[0],msg1[1],len(list1),msg2[0],msg2[1],len(list2),msg3[0],msg3[1],len(list3),msg4[0],msg4[1],len(list4)))
                time.sleep(10)
            except:
                break

        #构建Dataset
        with tf.device('/cpu:0'):
            train_files_names = os.listdir('data/datasets/imagenet/tfdata/')
            train_files = ['data/datasets/imagenet/tfdata/'+item for item in train_files_names]
            #train_files = ['testrecords.tfrecord']
            dataset_train = tf.data.TFRecordDataset(train_files)
            dataset_train = dataset_train.map(_parse_function, num_parallel_calls=4)
            dataset_train = dataset_train.repeat(10)
            dataset_train = dataset_train.batch(cfg.BATCH_SIZE)
            dataset_train = dataset_train.prefetch(cfg.BATCH_SIZE)
            iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
            self.next_images, self.next_labels = iterator.get_next()
            train_init_op = iterator.make_initializer(dataset_train)
        

    def makecsv(self):
        """
        Make 3 csv files (train, valid, test) for later use
        """
        classes = os.listdir('data/datasets/imagenet/')  # imagenet stored in data/datasets/imagenet/
        label_dict = {} # key is name, value is 0 - 999
        for i in range(len(classes)):
            label_dict[classes[i]] = i

        image_labels_list = [] # each element is folder name + class name
        for i in range(len(classes)):
            path = 'data/datasets/imagenet/' + classes[i] + '/'
            images_files = os.listdir(path)
            label = str(label_dict[classes[i]])
            for image_file in images_files:
                image_labels_list.append(path + image_file + ',' + label + '\n')

        random.shuffle(image_labels_list)
        num = len(image_labels_list)
        with open('imagenet_train.csv', 'w') as file:
            file.writelines(image_labels_list[:int(num * 0.8)])
        with open('imagenet_test.csv', 'w') as file:
            file.writelines(image_labels_list[int(num * 0.9):])
        with open('imagenet_valid.csv', 'w') as file:
            file.writelines(image_labels_list[int(num * 0.8):int(num * 0.9)])


    def make_example(self, image, label):
        return tf.train.Example(features=tf.train.Features(feature={
            'image' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            'label' : tf.trian.Feature(int64_lsit=tf.train.Int64List(Value=[label]))
        }))


    def gen_tfrecord(self, trainrecords, queue):
        tfrecords_file_num = 1
        file_num = 0
        total_num = len(trainrecords)
        writer = tf.python_io.TFRecordWriter("data/datasets/imagenet/tfdata/"+str(os.getpid())+"_"+str(tfrecords_file_num)+".tfrecord")
        pid = os.getpid()
 
       for record in trainrecords:
            file_num += 1
            fields = record.strip('\n').split(',')
            with open(fields[0], 'rb') as jpgfile:
                img = jpgfile.read()
            label = np.array(int(fields[1]))
            ex = make_example(img, label)
            writer.write(ex.SerializeToString())
            #每写入100条记录，向父进程发送消息，报告进度
            if file_num%100==0:
                queue.put((pid, file_num))
            if file_num%max_num==0 and file_num<total_num:
                writer.close()
                tfrecords_file_num += 1
                writer = tf.python_io.TFRecordWriter("data/datasets/imagenet/tfdata/"+str(os.getpid())+"_"+str(tfrecords_file_num)+".tfrecord")
        writer.close()
