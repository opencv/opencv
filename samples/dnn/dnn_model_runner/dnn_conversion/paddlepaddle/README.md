# Run PaddlePaddle model by OpenCV

This tutorial shows how to run PaddlePaddle model by opencv.

## Environment Setup

```shell
pip install paddlepaddle-gpu
pip install paddlehub
pip install paddle2onnx
```

## Run PaddlePaddle model demo

Run the example code as below,

```shell
python paddle_resnet50.py
```

there are 3 part of this execution

- 1. Export PaddlePaddle ResNet50 model to onnx format;
- 2. Use `cv2.dnn.readNetFromONNX` load model file;
- 3. Preprocess image file and do inference.
