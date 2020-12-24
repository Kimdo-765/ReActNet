# ReActNet

This is the pytorch implementation of our paper ["ReActNet: Towards Precise Binary NeuralNetwork with Generalized Activation Functions"](https://arxiv.org/abs/2003.03488), published in ECCV 2020. 

<div align=center>
<img width=60% src="https://github.com/liuzechun0216/images/blob/master/reactnet_github.jpg"/>
</div>

In this paper, we propose to generalize the traditional Sign and PReLU functions to RSign  and  RPReLU, which enable explicit learning of the distribution reshape and shift at near-zero extra cost. By adding simple learnable bias, ReActNet achieves 69.4% top-1 accuracy on Imagenet dataset with both weights and activations being binary, a near ResNet-level accuracy.

## Citation

If you find our code useful for your research, please consider citing:

    @inproceedings{liu2020reactnet,
      title={ReActNet: Towards Precise Binary Neural Network with Generalized Activation Functions},
      author={Liu, Zechun and Shen, Zhiqiang and Savvides, Marios and Cheng, Kwang-Ting},
      booktitle={European Conference on Computer Vision (ECCV)},
      year={2020}
    }

## Run

### 1. Requirements:
* python3.8 >, pytorch 1.4.0 >, torchvision 0.5.0 >
    
### 2. Data:
* Trained Model is Here
* https://drive.google.com/drive/folders/1hJqI-DHfcaL6B12hsf6-GOcKZYZI2ADT?usp=sharing

### 3. Steps to run:
(1) Step1:  binarizing activations
* Change directory to `./mobilenet/1_step1/` or `./mobilenetv2/1_step1/`
* run `bash run.sh`

(2) Step2:  binarizing weights + activations
* Change directory to `./mobilenet/2_step2/` or `./mobilenetv2/2_step2/`
* run `bash run.sh`
