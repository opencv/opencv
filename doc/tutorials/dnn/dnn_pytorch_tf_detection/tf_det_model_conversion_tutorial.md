# Conversion of TensorFlow Detection Models and Launch with OpenCV Python

## Goals
In this tutorial you will learn how to:
* obtain frozen graphs of TensorFlow (TF) detection models
* run converted TensorFlow model with OpenCV Python API

We will explore the above-listed points by the example of SSD MobileNetV1.

## Introduction
Let's briefly view the key concepts involved in the pipeline of TensorFlow models transition with OpenCV API. The initial step in conversion of TensorFlow models into [cv.dnn_Net](https://docs.opencv.org/4.3.0/db/d30/classcv_1_1dnn_1_1Net.html#a82eb4d60b3c396cb85c79d267516cf15)
is obtaining the frozen TF model graph. Frozen graph defines the combination of the model graph structure with kept values of the required variables, for example, weights. The frozen graph is saved in [protobuf](https://en.wikipedia.org/wiki/Protocol_Buffers) (```.pb```) files.
There are special functions for reading ``.pb`` graphs in OpenCV: [``cv.dnn.readNetFromTensorflow``](https://docs.opencv.org/4.3.0/d6/d0f/group__dnn.html#gad820b280978d06773234ba6841e77e8d) and [cv.dnn.readNet](https://docs.opencv.org/4.4.0/d6/d0f/group__dnn.html#ga3b34fe7a29494a6a4295c169a7d32422).

## Practise
In this part we are going to cover the following points:
1. create a TF classification model conversion pipeline and provide the inference
2. provide the inference, process prediction results with OpenCV [``samples/dnn/object_detection.py``](https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.py)

### Model Preparation
The code in this subchapter is located in the ``samples/dnn/dnn_model_runner`` module and can be executed with the below line:

```
python -m dnn_model_runner.dnn_conversion.tf.detection.py_to_py_ssd_mobilenet
```

The following code contains the steps of the TF SSD MobileNetV1 model retrieval:

```python
    tf_model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
    graph_extraction_dir = "./"
    frozen_graph_path = extract_tf_frozen_graph(tf_model_name, graph_extraction_dir)
    print("Frozen graph path for {}: {}".format(tf_model_name, frozen_graph_path))
```

In ``extract_tf_frozen_graph`` function we extract the provided in model archive ``frozen_inference_graph.pb`` for its further processing:

```python
# define model archive name
tf_model_tar = model_name + '.tar.gz'
# define link to retrieve model archive
model_link = DETECTION_MODELS_URL + tf_model_tar

tf_frozen_graph_name = 'frozen_inference_graph'

try:
    urllib.request.urlretrieve(model_link, tf_model_tar)
except Exception:
    print("TF {} was not retrieved: {}".format(model_name, model_link))
    return

print("TF {} was retrieved.".format(model_name))

tf_model_tar = tarfile.open(tf_model_tar)
frozen_graph_path = ""

for model_tar_elem in tf_model_tar.getmembers():
    if tf_frozen_graph_name in os.path.basename(model_tar_elem.name):
        tf_model_tar.extract(model_tar_elem, extracted_model_path)
        frozen_graph_path = os.path.join(extracted_model_path, model_tar_elem.name)
        break
tf_model_tar.close()
```

After the successful execution of the above code we will get the following output:

```
TF ssd_mobilenet_v1_coco_2017_11_17 was retrieved.
Frozen graph path for ssd_mobilenet_v1_coco_2017_11_17: ./ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb
```

For the inference we will use the below image from the [MS COCO](https://cocodataset.org/#home) validation dataset:

![MS COCO bus](images/mscoco_000000001584.jpg)

To initiate the test process we need to provide an appropriate model configuration. We will use [``ssd_mobilenet_v1_coco.config``](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config) from [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection#tensorflow-object-detection-api).
TensorFlow Object Detection API framework contains helpful mechanisms for object detection model manipulations.

We will use this configuration to provide a text graph representation. To generate ``.pbtxt`` we will use the corresponding [``samples/dnn/tf_text_graph_ssd.py``](https://github.com/opencv/opencv/blob/master/samples/dnn/tf_text_graph_ssd.py) script:

``
python tf_text_graph_ssd.py --input ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb --config ssd_mobilenet_v1_coco_2017_11_17/ssd_mobilenet_v1_coco.config --output ssd_mobilenet_v1_coco_2017_11_17.pbtxt
``

After successful execution ``ssd_mobilenet_v1_coco_2017_11_17.pbtxt`` will be created.

Before we run ``object_detection.py``, let's have a look at the default values for the SSD MobileNetV1 test process configuration. They are located in [``models.yml``](https://github.com/opencv/opencv/blob/master/samples/dnn/models.yml):

```yml
ssd_tf:
  model: "ssd_mobilenet_v1_coco_2017_11_17.pb"
  config: "ssd_mobilenet_v1_coco_2017_11_17.pbtxt"
  mean: [0, 0, 0]
  scale: 1.0
  width: 300
  height: 300
  rgb: true
  classes: "object_detection_classes_coco.txt"
  sample: "object_detection"
```

To fetch these values we need to provide frozen graph ``ssd_mobilenet_v1_coco_2017_11_17.pb`` model and text graph ``ssd_mobilenet_v1_coco_2017_11_17.pbtxt``:

```
python object_detection.py ssd_tf --input ../data/mscoco_000000001584.jpg
```

This line is equivalent to:

```
python object_detection.py --model ssd_mobilenet_v1_coco_2017_11_17.pb --config  ssd_mobilenet_v1_coco_2017_11_17.pbtxt  --input ../data/mscoco_000000001584.jpg --width 300 --height 300
```

The result is:

![OpenCV bus result](images/opencv_bus_res.png)
