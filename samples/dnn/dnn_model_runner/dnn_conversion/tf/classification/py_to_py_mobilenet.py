import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from ...common.utils import set_tf_env


def get_tf_model_proto(tf_model):
    # define the directory for .pb model
    pb_model_path = "models"

    # define the name of .pb model
    pb_model_name = "mobilenet.pb"

    # create directory for further converted model
    os.makedirs(pb_model_path, exist_ok=True)

    # get model TF graph
    tf_model_graph = tf.function(lambda x: tf_model(x))

    # get concrete function
    tf_model_graph = tf_model_graph.get_concrete_function(
        tf.TensorSpec(tf_model.inputs[0].shape, tf_model.inputs[0].dtype))

    # obtain frozen concrete function
    frozen_tf_func = convert_variables_to_constants_v2(tf_model_graph)
    # get frozen graph
    frozen_tf_func.graph.as_graph_def()

    # save full tf model
    tf.io.write_graph(graph_or_graph_def=frozen_tf_func.graph,
                      logdir=pb_model_path,
                      name=pb_model_name,
                      as_text=False)

    return os.path.join(pb_model_path, pb_model_name)


def get_preprocessed_img(img_path):
    # read the image
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    input_img = input_img.astype(np.float32)

    # define preprocess parameters
    mean = np.array([1.0, 1.0, 1.0]) * 127.5
    scale = 1 / 127.5

    # prepare input blob to fit the model input:
    # 1. subtract mean
    # 2. scale to set pixel values from 0 to 1
    input_blob = cv2.dnn.blobFromImage(
        image=input_img,
        scalefactor=scale,
        size=(224, 224),  # img target size
        mean=mean,
        swapRB=True,  # BGR -> RGB
        crop=True  # center crop
    )
    print("Input blob shape: {}\n".format(input_blob.shape))

    return input_blob


def get_imagenet_labels(labels_path):
    with open(labels_path) as f:
        imagenet_labels = [line.strip() for line in f.readlines()]
    return imagenet_labels


def get_opencv_dnn_prediction(opencv_net, preproc_img, imagenet_labels):
    # set OpenCV DNN input
    opencv_net.setInput(preproc_img)

    # OpenCV DNN inference
    out = opencv_net.forward()
    print("OpenCV DNN prediction: \n")
    print("* shape: ", out.shape)

    # get the predicted class ID
    imagenet_class_id = np.argmax(out)

    # get confidence
    confidence = out[0][imagenet_class_id]
    print("* class ID: {}, label: {}".format(imagenet_class_id, imagenet_labels[imagenet_class_id]))
    print("* confidence: {:.4f}\n".format(confidence))


def get_tf_dnn_prediction(original_net, preproc_img, imagenet_labels):
    # inference
    preproc_img = preproc_img.transpose(0, 2, 3, 1)
    print("TF input blob shape: {}\n".format(preproc_img.shape))

    out = original_net(preproc_img)

    print("\nTensorFlow model prediction: \n")
    print("* shape: ", out.shape)

    # get the predicted class ID
    imagenet_class_id = np.argmax(out)
    print("* class ID: {}, label: {}".format(imagenet_class_id, imagenet_labels[imagenet_class_id]))

    # get confidence
    confidence = out[0][imagenet_class_id]
    print("* confidence: {:.4f}".format(confidence))


def main():
    # configure TF launching
    set_tf_env()

    # initialize TF MobileNet model
    original_tf_model = MobileNet(
        include_top=True,
        weights="imagenet"
    )

    # get TF frozen graph path
    full_pb_path = get_tf_model_proto(original_tf_model)

    # read frozen graph with OpenCV API
    opencv_net = cv2.dnn.readNetFromTensorflow(full_pb_path)
    print("OpenCV model was successfully read. Model layers: \n", opencv_net.getLayerNames())

    # get preprocessed image
    input_img = get_preprocessed_img("../data/squirrel_cls.jpg")

    # get ImageNet labels
    imagenet_labels = get_imagenet_labels("../data/dnn/classification_classes_ILSVRC2012.txt")

    # obtain OpenCV DNN predictions
    get_opencv_dnn_prediction(opencv_net, input_img, imagenet_labels)

    # obtain TF model predictions
    get_tf_dnn_prediction(original_tf_model, input_img, imagenet_labels)


if __name__ == "__main__":
    main()
