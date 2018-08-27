This directory contains data files for the R-CNN and similar networks (Fast R-CNN, Faster R-CNN, Mask R-CNN, R-FCN).

**object detection/segmentation**

* R-CNN
    * GitHub repository: https://github.com/rbgirshick/rcnn
    * in ONNX model zoo: https://github.com/onnx/models/tree/master/bvlc_reference_rcnn_ilsvrc13
    * Paper: https://arxiv.org/abs/1311.2524
* Fast R-CNN
    * GitHub repository (original): https://github.com/rbgirshick/fast-rcnn
    * Paper: https://arxiv.org/abs/1504.08083
* Faster R-CNN
    * GitHub repository (original): https://github.com/ShaoqingRen/faster_rcnn
    * GitHub repository (original): https://github.com/rbgirshick/py-faster-rcnn
    * GitHub repository (ResNet): https://github.com/tensorflow/models/tree/master/research/object_detection/
    * GitHub repository (InceptionV2): https://github.com/tensorflow/models/tree/master/research/object_detection/
    * Paper: https://arxiv.org/abs/1506.01497
* Mask R-CNN
    * Paper: https://arxiv.org/abs/1703.06870
* R-FCN
    * GitHub repository: https://github.com/YuwenXiong/py-R-FCN


|    Model | Scale |   Size WxH|   Mean subtraction | Channels order |
|---------------|-------|-----------|--------------------|-------|
| Faster-RCNN | `1.0` | `800x600` | `102.9801 115.9465 122.7717` | BGR |
| R-FCN | `1.0` | `800x600` | `102.9801 115.9465 122.7717` | BGR |
| Faster-RCNN, ResNet backbone | `1.0` | `300x300` | `103.939 116.779 123.68` | RGB |
| Faster-RCNN, InceptionV2 backbone | `0.00784 (2/255)` | `300x300` | `127.5 127.5 127.5` | RGB |
