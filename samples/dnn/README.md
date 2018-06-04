# OpenCV deep learning module samples

## Model Zoo

### Object detection

|    Model | Scale |   Size WxH|   Mean subtraction | Channels order |
|---------------|-------|-----------|--------------------|-------|
| [MobileNet-SSD, Caffe](https://github.com/chuanqi305/MobileNet-SSD/) | `0.00784 (2/255)` | `300x300` | `127.5 127.5 127.5` | BGR |
| [OpenCV face detector](https://github.com/opencv/opencv/tree/3.4/samples/dnn/face_detector) | `1.0` | `300x300` | `104 177 123` | BGR |
| [SSDs from TensorFlow](https://github.com/tensorflow/models/tree/master/research/object_detection/) | `0.00784 (2/255)` | `300x300` | `127.5 127.5 127.5` | RGB |
| [YOLO](https://pjreddie.com/darknet/yolo/) | `0.00392 (1/255)` | `416x416` | `0 0 0` | RGB |
| [VGG16-SSD](https://github.com/weiliu89/caffe/tree/ssd) | `1.0` | `300x300` | `104 117 123` | BGR |
| [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn) | `1.0` | `800x600` | `102.9801 115.9465 122.7717` | BGR |
| [R-FCN](https://github.com/YuwenXiong/py-R-FCN) | `1.0` | `800x600` | `102.9801 115.9465 122.7717` | BGR |
| [Faster-RCNN, ResNet backbone](https://github.com/tensorflow/models/tree/master/research/object_detection/) | `1.0` | `300x300` | `103.939 116.779 123.68` | RGB |
| [Faster-RCNN, InceptionV2 backbone](https://github.com/tensorflow/models/tree/master/research/object_detection/) | `0.00784 (2/255)` | `300x300` | `127.5 127.5 127.5` | RGB |

#### Face detection
[An origin model](https://github.com/opencv/opencv/tree/3.4/samples/dnn/face_detector)
with single precision floating point weights has been quantized using [TensorFlow framework](https://www.tensorflow.org/).
To achieve the best accuracy run the model on BGR images resized to `300x300` applying mean subtraction
of values `(104, 177, 123)` for each blue, green and red channels correspondingly.

The following are accuracy metrics obtained using [COCO object detection evaluation
tool](http://cocodataset.org/#detections-eval) on [FDDB dataset](http://vis-www.cs.umass.edu/fddb/)
(see [script](https://github.com/opencv/opencv/blob/3.4/modules/dnn/misc/face_detector_accuracy.py))
applying resize to `300x300` and keeping an origin images' sizes.
```
AP - Average Precision                            | FP32/FP16 | UINT8          | FP32/FP16 | UINT8          |
AR - Average Recall                               | 300x300   | 300x300        | any size  | any size       |
--------------------------------------------------|-----------|----------------|-----------|----------------|
AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] | 0.408     | 0.408          | 0.378     | 0.328 (-0.050) |
AP @[ IoU=0.50      | area=   all | maxDets=100 ] | 0.849     | 0.849          | 0.797     | 0.790 (-0.007) |
AP @[ IoU=0.75      | area=   all | maxDets=100 ] | 0.251     | 0.251          | 0.208     | 0.140 (-0.068) |
AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ] | 0.050     | 0.051 (+0.001) | 0.107     | 0.070 (-0.037) |
AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] | 0.381     | 0.379 (-0.002) | 0.380     | 0.368 (-0.012) |
AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ] | 0.455     | 0.455          | 0.412     | 0.337 (-0.075) |
AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] | 0.299     | 0.299          | 0.279     | 0.246 (-0.033) |
AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] | 0.482     | 0.482          | 0.476     | 0.436 (-0.040) |
AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] | 0.496     | 0.496          | 0.491     | 0.451 (-0.040) |
AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ] | 0.189     | 0.193 (+0.004) | 0.284     | 0.232 (-0.052) |
AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] | 0.481     | 0.480 (-0.001) | 0.470     | 0.458 (-0.012) |
AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ] | 0.528     | 0.528          | 0.520     | 0.462 (-0.058) |
```

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
* [Names of classes from different datasets](https://github.com/opencv/opencv/tree/3.4/samples/data/dnn)
