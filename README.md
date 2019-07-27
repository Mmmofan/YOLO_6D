# Yolo 6d (singleshotposeestimate)

### UPDATE 2019.07.27
I've changed the loss layer and structures of codes, I'm not sure if it can be trained and valid correctly.
Hope someone test it and fix some bugs if it has.
issues or suggestions are welcomed and please feel free to  send me an email: [cokespace2@gmail.com](cokespace2@gmail.com), we can discuss a bit deeper.

---
## Introduction
Repoducibility of the following paper:
[Bugra Tekin, Sudipta N. Sinha and Pascal Fua, "Real-Time Seamless Single Shot 6D Object Pose Prediction", CVPR 2018.](https://arxiv.org/pdf/1711.08848.pdf)

You can check the ORIGINAL PYTORCH VERSION [here](https://github.com/microsoft/singleshotpose).

I'm rewriting it in TensorFlow but TF has many restricts (like tensor can not be assigned and so on), so it might take days until you can download this code and run it. Previous code didn't get the accuracy as the PyTorch version that's why I decided to rewrite it :)

If you're also interested in this paper and want to implement it on TensorFlow, this repo could be a help.

![SingleShotPose](https://btekin.github.io/single_shot_pose.png)

## Environment and dependencies
This code is test on Ubuntu 16.04 and Windows 10 with CUDA 9.1 and cudNN v7.1.4. 

Requirements:
 - Python==3.6.5
 - TensorFlow==1.8.0
 - TensorFlow-gpu==1.8.0
 - Numpy==1.15.4
 - opcnv-python==3.4.3
 - PIL==6.1.0

## Datasets and Weights
This code use LINEMOD dataset. In training you also will be using VOC dataset, when you clone this repo and go inside the main directory, you can download datasets by typing:
```
wget -O LINEMOD.tar --no-check-certificate "https://onedrive.live.com/download?cid=05750EBEE1537631&resid=5750EBEE1537631%21135&authkey=AJRHFmZbcjXxTmI"
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
tar xf LINEMOD.tar
tar xf VOCtrainval_11-May-2012.tar
```
Because of the PyTorch *.weights* files cannot be used in TensorFlow, you have to train it from scratch. 
Author use imagenet pretrained weights, we can choose coco to pretrain the yolo_v2 architecture.

## Pre-train
These weights had been trained on COCO dataset for classification. In paper, author use ImageNet pre-trained weights, I didn't test the difference. Hope someone can try it and give us some advices.

Because of in early iterations pose estimations are inaccuracy, the confidence value would be vary widely, so author suggested pre-train the model by changing the *CONF_OBJ_SCALE = 0* and *CONF_NOOBJ_SCALE = 0*, to achieve this, run:

```
python train.py --datacfg cfg/ape.data --pre True --gpu 2
```
the first argv *datacfg* represents the catogories of dataset, second argv *pre* represents if it is pre training.

## Train the model
When finish pre-train and set the right *.ckpt* files(make sure the name, path correct), run below to train the model
```
python train.py --datacfg cfg/ape.data --gpu 2
```


Hope you get what you want

Enjoy

--Fan
