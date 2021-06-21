# Run PaddlePaddle model by OpenCV

This tutorial shows how to run PaddlePaddle model by opencv.

## Run PaddlePaddle ResNet50 model by OpenCV
### Environment Setup

```shell
pip install paddlepaddle-gpu
pip install paddlehub
pip install paddle2onnx
```

### Run PaddlePaddle model demo

Run the example code as below,

```shell
python paddle_resnet50.py
```

there are 3 part of this execution

- 1. Export PaddlePaddle ResNet50 model to onnx format;
- 2. Use `cv2.dnn.readNetFromONNX` load model file;
- 3. Preprocess image file and do inference.


## Run PaddleSeg Portrait Segmentation by OpenCV
### Environment Setup

```shell
pip install paddlepaddle-gpu
pip install paddlehub
```

### Get ONNX Model

Thanks for [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg),this ONNX model convert from HRNet w18 small v1.For more details,please refer to [HumanSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/contrib/HumanSeg/README.md).
```
wget 
```

### Run PaddleSeg Portrait Segmentation demo

Run the example code as below,

```shell
python paddle_humanseg.py
```

there are 3 part of this execution

- 1. Use `cv2.dnn.readNetFromONNX` load model file;
- 2. Preprocess image file and do inference.
- 3. Postprocess image file and visualization.

### Portrait segmentation visualization
<img src="./data/human_image.jpg" width="50%" height="50%"><img src="./data/result_test_human.png" width="50%" height="50%">