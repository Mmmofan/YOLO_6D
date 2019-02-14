# Yolo 6d (singleshotpose)
Repoducibility of the following paper:

[Bugra Tekin, Sudipta N. Sinha and Pascal Fua, "Real-Time Seamless Single Shot 6D Object Pose Prediction", CVPR 2018.](https://arxiv.org/pdf/1711.08848.pdf)

I'm still training it and didn't get the accuracy as the Pytoch version which provided by author. However, in this repo I will share my understands and experiences in reproducing it.

If you're also interested in this paper and want to implement it on TensorFlow, this repo could be a help.

I will leave the **problems and newest prograss** in the **final part**. If anyone meet the same issues or can solve the problems please feel free to leave me a message or send me an email: [cokespace2@gmail.com](cokespace2@gmail.com), we can discuss these problems.

### Introduction
This is a repo to implement this paper in TensorFlow: [Real-Time.Seamless.Single.Shot.6D.Object.Pose.Prediction](https://arxiv.org/abs/1711.08848). 
The author is [Bugra Tekin](http://bugratekin.info), Sudipta N, etc.

Below is deprecated from [THIS REPO](https://github.com/Microsoft/singleshotpose).

We propose a single-shot approach for simultaneously detecting an object in an RGB image and predicting its 6D pose without requiring multiple stages or having to examine multiple hypotheses. The key component of our method is a new CNN architecture inspired by the YOLO network design that directly predicts the 2D image locations of the projected vertices of the object's 3D bounding box. The object's 6D pose is then estimated using a PnP algorithm.

As I understand, this code use YOLO_v2 architecture and add pose estimation to it. What different from yolo_v2 are ways of confidence value computation and the loss function.

![SingleShotPose](https://btekin.github.io/single_shot_pose.png)

### Environment and dependencies
This code is test on Linux (ubuntu 16.04) with CUDA 9.0 and cudNN v7. This implementation is write on TensorFlow 1.8.0 (former or later version can still run this) using Python 3.5.3. To run this you need to install following dependencies with PIP or CONDA: 
TensorFlow, Numpy, OpenCV.

### Datasets and Weights
This code use same datasets as original one - LINEMOD. In training you alse will need VOC dataset for some reason, when you clone this repo and inside the main code directory, you can just download the datasets using:
```
wget -O LINEMOD.tar --no-check-certificate "https://onedrive.live.com/download?cid=05750EBEE1537631&resid=5750EBEE1537631%21135&authkey=AJRHFmZbcjXxTmI"
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
tar xf LINEMOD.tar
tar xf VOCtrainval_11-May-2012.tar
```
Because of the PyTorch *.weights* files cannot be used in TensorFlow, you can down load the *.ckpt* files [here](). (These are weight files pre-trained on LINEMOD-Single, see next part)

Alternatively, you can directly go to the links above and manually download and extract the files at the corresponding directories. The whole download process might take a long while (~60 minutes).

### Pre-train
For better detection, I pre-train the model **twice**, I'll explain this.

First is train on YOLO_v2 with LINEMOD-Single([Download here](https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/object-detection-and-3d-pose-estimation/)), this Datasets is to train the classification task use yolo-v2 architecture based on COCO weights file. Author said in his paper that their initial weights are pre-trained on Imagenet which is very time-consume so I just download the *yolo_v2_coco.ckpt* files to continue pre-training.
You can check this pre-train code in [this repo](https://github.com/Mmmofan/Yolo_6d-pre-train-with-yolov2).

Second, because of in early iterations pose estimations are inaccuracy, the confidence value would be vary widely, so author suggested pre-train on this 6d-net by changing the *CONF_OBJ_SCALE = 0* and *CONF_NOOBJ_SCALE = 0* in [yolo/config.py](https://github.com/Mmmofan/YOLO_6D/blob/master/yolo/config.py).
Run this to pre-train:
```
python train.py --datacfg cfg/ape.data --pre True
```
so as other catogories.

### Train the model
When finish pre-train and set the right *.ckpt* files(make sure the name, path right), run this code to train the model
```
python train.py --datacfg cfg/ape.data
```

### Problems and prograss
I finish the pre-train step (it gets good accuracy on classification), but in training, I didn't get good result, this model's prediction even not a cubic of 2d projection of objects.
I think maybe the loss function are blamed on the inaccuray, if you find any incorrect of code, please send me an email [cokespace2@gmail.com](cokespace2@gmail.com).
The *valid.py* file is not finished yet, it now just to check whether one picture is predicted correctly.
Also, I just test the model using single object, which means the multi-object pose estimation have not done yet.