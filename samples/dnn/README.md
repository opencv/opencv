# OpenCV deep learning module samples

## Model Zoo

### Object detection

|    Model | Scale |   Size WxH|   Mean subtraction | Channels order |
|---------------|-------|-----------|--------------------|-------|
| [MobileNet-SSD, Caffe](https://github.com/chuanqi305/MobileNet-SSD/) | `0.00784 (2/255)` | `300x300` | `127.5 127.5 127.5` | BGR |
| [OpenCV face detector](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector) | `1.0` | `300x300` | `104 177 123` | BGR |
| [SSDs from TensorFlow](https://github.com/tensorflow/models/tree/master/research/object_detection/) | `0.00784 (2/255)` | `300x300` | `127.5 127.5 127.5` | RGB |
| [YOLO](https://pjreddie.com/darknet/yolo/) | `0.00392 (1/255)` | `416x416` | `0 0 0` | RGB |
| [VGG16-SSD](https://github.com/weiliu89/caffe/tree/ssd) | `1.0` | `300x300` | `104 117 123` | BGR |
| [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn) | `1.0` | `800x600` | `102.9801, 115.9465, 122.7717` | BGR |
| [R-FCN](https://github.com/YuwenXiong/py-R-FCN) | `1.0` | `800x600` | `102.9801 115.9465 122.7717` | BGR |

### Classification
|    Model | Scale |   Size WxH|   Mean subtraction | Channels order |
|---------------|-------|-----------|--------------------|-------|
| GoogLeNet | `1.0` | `224x224` | `104 117 123` | BGR |
| [SqueezeNet](https://github.com/DeepScale/SqueezeNet) | `1.0` | `227x227` | `0 0 0` | BGR |

### Semantic segmentation
|    Model | Scale |   Size WxH|   Mean subtraction | Channels order |
|---------------|-------|-----------|--------------------|-------|
| [ENet](https://github.com/e-lab/ENet-training) | `0.00392 (1/255)` | `1024x512` | `0 0 0` | RGB |
| FCN8s | `1.0` | `500x500` | `0 0 0` | BGR |

## References
* [Models downloading script](https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/download_models.py)
* [Configuration files adopted for OpenCV](https://github.com/opencv/opencv_extra/tree/master/testdata/dnn)
* [How to import models from TensorFlow Object Detection API](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API)
* [Names of classes from different datasets](https://github.com/opencv/opencv/tree/master/samples/data/dnn)
