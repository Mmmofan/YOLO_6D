# Yolo 6d (singleshotpose)
Repoducibility of the following paper:

[Bugra Tekin, Sudipta N. Sinha and Pascal Fua, "Real-Time Seamless Single Shot 6D Object Pose Prediction", CVPR 2018.](https://arxiv.org/pdf/1711.08848.pdf)

I'm still training it and didn't get the accuracy as the Pytoch version which provided by author. 
If you're also interested in this paper and want to implement it on TensorFlow, this repo could be a help.
I will leave the **problems and newest prograss** in the **final part**. If anyone meet the same issues or can solve the problems please feel free to leave me a message or send me an email: [cokespace2@gmail.com](cokespace2@gmail.com), we can discuss a bit deeper.

---

### Introduction
This is a repo to implement this paper in TensorFlow: [Real-Time.Seamless.Single.Shot.6D.Object.Pose.Prediction](https://arxiv.org/abs/1711.08848). 
The author is [Bugra Tekin](http://bugratekin.info), Sudipta N, etc.

"*We propose a single-shot approach for simultaneously detecting an object in an RGB image and predicting its 6D pose without requiring multiple stages or having to examine multiple hypotheses. The key component of our method is a new CNN architecture inspired by the YOLO network design that directly predicts the 2D image locations of the projected vertices of the object's 3D bounding box. The object's 6D pose is then estimated using a PnP algorithm.*"
![SingleShotPose](https://btekin.github.io/single_shot_pose.png)
As I understand, this code use YOLO_v2 architecture and add pose estimation to it. What different from yolo_v2 are ways of confidence value computation and the loss function.

---

### Environment and dependencies
This code is test on Linux (ubuntu 16.04) with CUDA 9.0 and cudNN v7. This implementation is write on TensorFlow 1.8.0 (former or later version can still run this) using Python 3.5.3. To run this you need to install following dependencies with PIP or CONDA: 
Python 3.5.4, TensorFlow 1.8.0, Numpy 1.15.4, OpenCV 3.4.0.

---

### Datasets and Weights
This code use LINEMOD dataset. In training you also will be using VOC dataset for some reason, when you clone this repo and go inside the main directory, you can download datasets by typing:
```
wget -O LINEMOD.tar --no-check-certificate "https://onedrive.live.com/download?cid=05750EBEE1537631&resid=5750EBEE1537631%21135&authkey=AJRHFmZbcjXxTmI"
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
tar xf LINEMOD.tar
tar xf VOCtrainval_11-May-2012.tar
```
Because of the PyTorch *.weights* files cannot be used in TensorFlow,  you can down load the COCO pre-trained ***.ckpt*** files [here](https://drive.google.com/drive/folders/17yD4zWQCi3-D24CuxjzO6BGS6HZzKPHx). 

---

### Pre-train
These weights had been trained on COCO dataset for classification. In paper, author use ImageNet pre-trained weights, I didn't test the difference. Hope someone can try it and give us some advices.

Because of in early iterations pose estimations are inaccuracy, the confidence value would be vary widely, so author suggested pre-train the model by changing the *CONF_OBJ_SCALE = 0* and *CONF_NOOBJ_SCALE = 0*, to achieve this, run:

```
python train.py --datacfg cfg/ape.data --pre True --gpu 2
```
the first argv *datacfg* represents the catogories of dataset, second argv *pre* represents if it is pre training.

---

### Train the model
When finish pre-train and set the right *.ckpt* files(make sure the name, path right), run this code to train the model
```
python train.py --datacfg cfg/ape.data --gpu 2
```

---

### Test
After training, run
```
python valid.py --datacfg cfg/ape.data
```
to see the result and accuracy.

---

### Problems and prograss
I finish the pre-train step (it gets good accuracy on classification), but in training, I didn't get good result, the **coordinates loss** is hard to optimize and I have no answer for it.
I think maybe the loss function are blamed on the inaccuray, if you find any incorrect of code, please send me an email [cokespace2@gmail.com](cokespace2@gmail.com), I will be appreciate that.
The *valid.py* file is not finished yet, it now just to check whether one picture is predicted correctly.
Also, I just test the model using single object, the occlusion part will be later uploaded to this repo

Thanks for reading
Fan