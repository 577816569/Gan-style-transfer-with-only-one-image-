# Some explorations on GAN with style transfer

This work is only a small attempt to explore the field of GAN(Generative Adversarial Networks). The result of this work is really bad due to my lack of understanding in the field of deep learning and the terrible training environment(I trained the network on Colab). So if anyone see this repo(probably no one) and find a lot of mistakes please ignore them. 




## About this work
There have been already a lot of work explored the possibility to use GAN achieve style transfer. For example:

Cycle GAN : [[Paper]](https://arxiv.org/abs/1703.10593)

Cartoon GAN : [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf)

Gated GAN : [[Paper]](https://arxiv.org/abs/1904.02296)

These GAN could generate very good results. However, to train a GAN, we need thousands of paired/unpaired image datas, and if we want to achieve transfer only one image's style like the original paper wrote by Gatys [[Paper]](https://ieeexplore.ieee.org/document/7780634), that could be a problem. Also, Gatys' method of style transfer is online method, so the efficiency is not as high as GAN base method.

In this work, I use some simple tricks to achieve transfer only one image's style to another with GAN.

## Method

To solve the problem of lack of datasets, I cut the orignial style image into 256 square images randomly and the size of each image is 16 * 16. By conbining these small images, we can get a 256 * 256 style image. We can use this method to generate as many style images as we want. 

<img src="https://github.com/577816569/Some-explorations-on-gan-with-styletransfer/blob/master/style/mosaic.jpg" width = "300" height = "200" alt="Original style image" align=center /><img src="https://github.com/577816569/Some-explorations-on-gan-with-styletransfer/blob/master/style/0.jpg" width = "200" height = "200" alt="Transfered style image" align=center />

&emsp;&emsp;&emsp;&emsp;Original style image &emsp;&emsp;&emsp;                     Transfered style image
 
Also, the code of these work is base on the this repository: [https://github.com/eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)



##  Prerequisites
* Python 3
* Ptyorch 0.41+
* NVIDIA GPU + CUDA CuDNN


## Usage

* First, transfer the style image
` python transfer.py `
* Train the model
` python train.py`
Noted that train 3 epochs could get a good result.
## Result

Here are some result, and as you can see the result is not very good, because I haven't done any modification to some parameters in the code, also I am a beginner of deep learning field, so there might be a lot of mistakes.

<img src="https://github.com/577816569/Some-explorations-on-gan-with-styletransfer/blob/master/result/1.png" width = "150" height = "150" alt="Transfered style image" align=center /><img src="https://github.com/577816569/Some-explorations-on-gan-with-styletransfer/blob/master/result/3.png" width = "150" height = "150" alt="Transfered style image" align=center /><img src="https://github.com/577816569/Some-explorations-on-gan-with-styletransfer/blob/master/result/4.png" width = "150" height = "150" alt="Transfered style image" align=center /><img src="https://github.com/577816569/Some-explorations-on-gan-with-styletransfer/blob/master/result/5.png" width = "150" height = "150" alt="Transfered style image" align=center />


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTM0Mjk2NDQyMywxMjIyODc5OTM3LC0xMz
E4ODY5OTA3LDgxMDcwMzI3MSwtMzg0NDA3MDY0LC0xNjg5ODE1
NDgyLDEyMzkxNjQ3OTUsMTk4MTc1ODI2MywyMDg2OTY3NTMzLD
E4MjMyODIzNjgsLTE2MzA2MzUxOTEsLTQxNDQzOTIyMiw4OTU2
OTk0NjIsLTExMTkwNDE3MjEsLTE3MTkzMzQ2NDksLTE0MDIyNz
gzNTUsMzExNjcxMCwtMTQ4NTgzNjcwNiwtMTQ1NDUyMjc1LDE2
NDcyMjA2Nl19
-->