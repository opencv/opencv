# OpenCV deep learning module samples

## Model Zoo

Check [a wiki](https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV) for a list of tested models.

If OpenCV is built with [Intel's Inference Engine support](https://github.com/opencv/opencv/wiki/Intel%27s-Deep-Learning-Inference-Engine-backend) you can use [Intel's pre-trained](https://github.com/opencv/open_model_zoo) models.

There are different preprocessing parameters such mean subtraction or scale factors for different models.
You may check the most popular models and their parameters at [models.yml](https://github.com/opencv/opencv/blob/master/samples/dnn/models.yml) configuration file. It might be also used for aliasing samples parameters. In example,

```bash
python object_detection.py opencv_fd --model /path/to/caffemodel --config /path/to/prototxt
```

Check `-h` option to know which values are used by default:

```bash
python object_detection.py opencv_fd -h
```

#### Face detection
[An origin model](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)
with single precision floating point weights has been quantized using [TensorFlow framework](https://www.tensorflow.org/).
To achieve the best accuracy run the model on BGR images resized to `300x300` applying mean subtraction
of values `(104, 177, 123)` for each blue, green and red channels correspondingly.

The following are accuracy metrics obtained using [COCO object detection evaluation
tool](http://cocodataset.org/#detections-eval) on [FDDB dataset](http://vis-www.cs.umass.edu/fddb/)
(see [script](https://github.com/opencv/opencv/blob/master/modules/dnn/misc/face_detector_accuracy.py))
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

## References
* [Models downloading script](https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/download_models.py)
* [Configuration files adopted for OpenCV](https://github.com/opencv/opencv_extra/tree/master/testdata/dnn)
* [How to import models from TensorFlow Object Detection API](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API)
* [Names of classes from different datasets](https://github.com/opencv/opencv/tree/master/samples/data/dnn)
