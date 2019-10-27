# FCHarDNet
### Fully Convolutional HarDNet for Segmentation in Pytorch
### [Harmonic DenseNet: A low memory traffic network (ICCV 2019)](https://arxiv.org/abs/1909.00948)
### Refer to [Pytorch-HarDNet](https://github.com/PingoLH/Pytorch-HarDNet) for more information

#### This repo was forked from [meetshah1995/pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg)

<p align="center">
  <img src="pic/fchardnet70_arch.png" width="512" title="FC-HarDNet-70 Architecture">
</p>

### DataLoaders implemented

* [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
* [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html)
* [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
* [MIT Scene Parsing Benchmark](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)
* [Cityscapes](https://www.cityscapes-dataset.com/)
* [NYUDv2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
* [Sun-RGBD](http://rgbd.cs.princeton.edu/)


### Requirements

* pytorch >=0.4.0
* torchvision ==0.2.0
* scipy
* tqdm
* tensorboardX

### Usage

**Setup config file**

Please see the usage section in [meetshah1995/pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg)

**To train the model :**

```
python train.py [-h] [--config [CONFIG]]

--config                Configuration file to use (default: hardnet.yml)
```

**To validate the model :**

```
usage: validate.py [-h] [--config [CONFIG]] [--model_path [MODEL_PATH]] [--save_image]
                       [--eval_flip] [--measure_time]

  --config              Config file to be used
  --model_path          Path to the saved model
  --eval_flip           Enable evaluation with flipped image | False by default
  --measure_time        Enable evaluation with time (fps) measurement | True by default
  --save_image          Enable writing result images to out_rgb (pred label blended images) and out_predID

```

