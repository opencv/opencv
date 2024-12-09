# Run PaddlePaddle model using OpenCV

These two demonstrations show how to inference PaddlePaddle model using OpenCV.

## Environment Setup

```shell
pip install paddlepaddle-gpu
pip install paddlehub
pip install paddle2onnx
```

## 1. Run PaddlePaddle ResNet50 using OpenCV

### Run PaddlePaddle model demo

Run the code sample as follows:

```shell
python paddle_resnet50.py
```

There are three parts to the process:

1. Export PaddlePaddle ResNet50 model to onnx format.
2. Use `cv2.dnn.readNetFromONNX` to load the model file.
3. Preprocess image file and do the inference.

## 2. Run PaddleSeg Portrait Segmentation using OpenCV

### Convert to ONNX Model

#### 1. Get Paddle Inference model

For more details, please refer to [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/contrib/HumanSeg/README.md).

```shell
wget https://x2paddle.bj.bcebos.com/inference/models/humanseg_hrnet18_small_v1.zip
unzip humanseg_hrnet18_small_v1.zip
```

Notes:

* The exported model must have a fixed input shape, as dynamic is not supported at this moment.

#### 2. Convert to ONNX model using paddle2onnx

To convert the model, use the following command:

```
paddle2onnx --model_dir humanseg_hrnet18_small_v1 \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --opset_version 11 \
            --save_file humanseg_hrnet18_tiny.onnx
```

The converted model can be found in the current directory by the name `humanseg_hrnet18_tiny.onnx` .

### Run PaddleSeg Portrait Segmentation demo

Run the code sample as follows:

```shell
python paddle_humanseg.py
```

There are three parts to the process:

1. Use `cv2.dnn.readNetFromONNX` to load the model file.
2. Preprocess image file and do inference.
3. Postprocess image file and visualize.

The resulting file can be found at `data/result_test_human.jpg` .

### Portrait segmentation visualization

<img src="../../../../data/messi5.jpg" width="50%" height="50%"><img src="./data/result_test_human.jpg" width="50%" height="50%">
