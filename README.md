# MasterThesis2
One-shot video objects tracking and segmentation: tracking via object parts and segmentation via mask propagation on parts. Combining the ideas from
FAVOS \[[1](https://github.com/JingchunCheng/FAVOS)\] and RGMP \[[2](https://github.com/seoungwugoh/RGMP)\], we propose to use
FPN \[[2](https://arxiv.org/abs/1612.03144)\] as backbone to track object parts and propagate their masks in a unified network similar to Mask R-CNN \[[3](https://arxiv.org/abs/1703.06870)\].
The proposed model does not need fine-tuning on the 0th frame where the ground truth segmentation mask is provided. Instead, our model learns a similarity representation via
cross-correlation between templar feature patch and search sub-window. Inspired by Fully-Convolutional Siamese Network for Object Tracking \[[4](https://arxiv.org/abs/1606.09549)\], our model also tracks object pixels
 via mask propagation. We believe that tracking object pixels is feasible even if the object motion is non-rigid as long as the model is trained using a large
 amount of data. Flownet \[[5](https://arxiv.org/abs/1504.06852)\] is a good example of tracking scene pixels from one frame to another regardless of object categories.
 
## Important Notes
* From now on, all newly created branches from the master branch implement multi-GPU model

## Backbone
* ResNet-50-v2 pre-trained on ImageNet2012 (from Tensorflow official model zoo \[[6](https://github.com/tensorflow/models/tree/r1.8.0/official/resnet)\])
* Extend the above ResNet to FPN \[[2](https://arxiv.org/abs/1612.03144)\], additional layers are initialised using tf.glorot_uniform_initializer().
* Extend the above FPN by adding additional cross-correlation layers (Multi-GPU).
* (TODO) Train extended FPN on bounding box tracking using CC operation (Multi-GPU).
* (TODO) Train extended FPN on bounding box regression using RPN (Multi-GPU).
* (TODO) Train the above network on bbox tracking & mask propagation simultaneously (Multi-GPU).
    * Inference on DAVIS-2016 by tracking and propagating object parts' masks


## Tensorflow
* TF version = 1.12.0
* CUDA 9.0, cuDNN >= 7.2
* (Optional) NCCL 2.2 for multi-gpu if use batch stats sync

## Notes on Multi-GPU (TF_version = 1.12.0)
* NCCL 2.2 is a must if use batch stats sync
* All model variables (convolution kernels, biases, dense weights, batch_norm gammas and betas) are created on CPU using tf.get_variable explicitly. All model parameters
are shared across GPUs (but not graphs/operations which are specified via tf.name_scope)
* Moving statistics are only created and updated locally on the 1st tower (tower_0)
* Specify tf.name_scope when building models on different towers (since different towers need to execute different graphs/operations)


## TODO lists
* **(Done)** Load pre-trained model on ImageNet2012
    * **(Done)** Verify validation on single GPU implementation
        * Official accuracy: 76.47% (fp32)
        * Test accuracy: 76.47% (fp32)
    * **(Done)** Verify validation on multiple GPUs implementation
* **(DONE)** Train ResNet-50-v2 on ImageNet2012 from scratch on single GPU
    * Verify validation on single GPU: 70.71% (70ep)
* **(DONE)** Train ResNet-50-v2 on ImageNet2012 from scratch on multiple GPUs
    * Verify validation on multiple GPUs: 75.96% (100ep, SGD)
* **(TODO)** Train FPN_CC_track on ImageNet15-VID from pre-trained ImageNet12 on multi-GPU
    * Verify accuray on validation/test sets
* **(TODO)** Train FPN_CC_track on ImageNet15-VID and Youtube-BB from pre-trained ImageNet12 on multi-GPU
    * Verify accuray on validation/test sets



## References
1. [Fast and Accurate Online Video Object Segmentation via Tracking Parts](https://github.com/JingchunCheng/FAVOS)
2. [Fast Video Object Segmentation by Reference-Guided Mask Propagation](https://github.com/seoungwugoh/RGMP)
3. [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
4. [Fully-Convolutional Siamese Networks for Object Tracking](https://arxiv.org/abs/1606.09549)
5. [FlowNet: Learning Optical Flow with Convolutional Networks](https://arxiv.org/abs/1504.06852)
6. [Tensorflow Model Zoo official: resnet](https://github.com/tensorflow/models/tree/r1.8.0/official/resnet)
