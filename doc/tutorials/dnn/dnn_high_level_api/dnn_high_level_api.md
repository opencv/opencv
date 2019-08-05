# High Level API: Usage and tips {#tutorial_dnn_high_level_api}

### Introduction
In this tutorial we will go through how you can use the new OpenCV Dnn API. The newly added DNN module allows us to run inference on deep neural networks exported from other frameworks, such as Caffe, Tensorflow, ONNX, and Torch.

### Usage
Here is a small example on how to run inference on the different API module. For a more detailed example, check the [samples](https://github.com/opencv/opencv/tree/master/samples/dnn)

#### Classification Module
@snippet dnn/classification.cpp Read and initialize network
@snippet dnn/classification.cpp Set Input Parameters
@snippet dnn/classification.cpp Network Forward pass

This will return the predicted class and the confidence of the model.

#### Detection Module
@snippet dnn/detection.cpp Read and initialize network
@snippet dnn/detection.cpp Set Input Parameters
@snippet dnn/detection.cpp Network Forward pass

This will store the predicted classes in ``classIds``, the bounding boxes in ``boxes`` and the confidence of the model in ``confidences``


#### Segmentation Module
@snippet dnn/segmentation.cpp Read and initialize network
@snippet dnn/segmentation.cpp Set Input Parameters
@snippet dnn/segmentation.cpp Network Forward pass

This will store the segmentation map in ``mask``

#### Keypoints Module
@snippet dnn/keypoints.cpp Read and initialize network
@snippet dnn/keypoints.cpp Set Input Parameters
@snippet dnn/keypoints.cpp Network Forward pass

This will return the predicted keypoints.


### Some explanations
If you are new to Machine Learning you might be wondering why do we need those variables, scale, Size, mean, etc. Usually statistical models,
(e.g Neural Networks) are trained on normalized data to help convergence. When working with images, it is common to do this normalization in a per-channel
fashion, so you take every image and subtract the per-channel mean and divide it by the per-channel standard deviation (std). Notice that the second operation (i.e dividing by the std)
is done to limit the range of our data, and since the values in an image are already limited (0-255), you will sometimes see this step being skipped and the image just being divided by 1.
Since the OpenCV Dnn Module is based on Caffe, and Caffe does skip this step (i.e divide the image by 1) this operation is not integrated into the API directly. So if we really want to
normalize the image we have to set the **scale** parameter to the inverse of the standard deviation. Thus, the **scale** parameter represents we will scale our image by,
and the **mean** represents the per-channel mean we want to subtract from it. It is also common to resize the images to a fixed size for convenience before feeding
them to the network, this size is specified by the **size** parameter in our code. Last but not least, the **swapRB** variable indicates to swap the order of a BGR image to RGB
or vice versa.
OpenCV reads images in BGR format, while other frameworks, such as PIL read them in RGB. Anyhow, setting this flag to **true** will swap the image channels and the **mean** vector accordingly.
