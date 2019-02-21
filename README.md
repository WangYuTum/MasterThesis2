# MasterThesis2
One-shot video objects tracking and segmentation: tracking via object parts and segmentation via mask propagation on parts. Combining the ideas from
FAVOS \[[1](https://github.com/JingchunCheng/FAVOS)\] and RGMP \[[2](https://github.com/seoungwugoh/RGMP)\], we propose to use
FPN \[[2](https://arxiv.org/abs/1612.03144)\] as backbone to track object parts and propagate their masks in a unified network similar to Mask R-CNN \[[3](https://arxiv.org/abs/1703.06870)\].
The proposed model does not need fine-tuning on the 0th frame where the ground truth segmentation mask is provided. Instead, our model learns a similarity representation via
cross-correlation between templar feature patch and search sub-window. Inspired by Fully-Convolutional Siamese Network for Object Tracking \[[4](https://arxiv.org/abs/1606.09549)\], our model also tracks object pixels
 via mask propagation. We believe that tracking object pixels is feasible even if the object motion is non-rigid as long as the model is trained using a large
 amount of data. Flownet \[[5](https://arxiv.org/abs/1504.06852)\] is a good example of tracking scene pixels from one frame to another regardless of object categories.


## Tensorflow
* TF version = 1.12.0
* CUDA 9.0, cuDNN >= 7.2
* (Optional) NCCL 2.2 for multi-gpu if use batch stats synchronization


## GPU
* **Trained** 4xP6000 (24GB), batch=512, adam(lr=0.256, ep=0.01), BN_mom=0.95
    * Accuracy: 22.29% (20ep)
* **Trained** 4xTitanXP (12GB), batch=256, SGD_momentum(base_lr=0.256, scaled_lr=0.128, momentum=0.9), BN_mom=0.95
    * Accuracy: 49.92 (10ep); training suspended after 80k iters for unknown reason
* **Training** 4xTitanXP (12GB), batch=256, SGD_momentum(base_lr=0.256, scaled_lr=0.128, momentum=0.9), BN_mom=0.95
    * Accuracy: 47.87% (10ep), 52.45% (20ep), 53.78% (30ep), 69.086 (40ep), 68.87% (50ep), 69.034(60ep)
* **TODO** Use Batch stats synchronization via Nvidia NCCL


## Notes on Multi-GPU (TF_version = 1.12.0)
* NCCL 2.2 is a must if use multi-gpu batch stats synchronization
* All model variables (convolution kernels, biases, dense weights, batch_norm gammas and betas) are created on CPU using tf.get_variable explicitly. All model parameters
are shared across GPUs (but not graphs/operations which are specified via tf.name_scope)
* Moving statistics are only created and updated locally on the 1st tower (tower_0)
* Specify tf.name_scope when building models on different towers (since different towers need to execute different graphs/operations)


## Dataset
* ImageNet2012 Train Set



## References
1. [Fast and Accurate Online Video Object Segmentation via Tracking Parts](https://github.com/JingchunCheng/FAVOS)
2. [Fast Video Object Segmentation by Reference-Guided Mask Propagation](https://github.com/seoungwugoh/RGMP)
3. [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
4. [Fully-Convolutional Siamese Networks for Object Tracking](https://arxiv.org/abs/1606.09549)
5. [FlowNet: Learning Optical Flow with Convolutional Networks](https://arxiv.org/abs/1504.06852)
6. [Tensorflow Model Zoo official: resnet](https://github.com/tensorflow/models/tree/r1.8.0/official/resnet)
