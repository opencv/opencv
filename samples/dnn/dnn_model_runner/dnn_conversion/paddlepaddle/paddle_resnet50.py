import paddle
import paddlehub as hub
import paddlehub.vision.transforms as T
import cv2 as cv
import numpy as np


def preprocess(image_path):
    ''' preprocess input image file to np.ndarray

    Args:
        image_path(str): Path of input image file

    Returns:
        ProcessedImage(numpy.ndarray): A numpy.ndarray
                variable which shape is (1, 3, 224, 224)
    '''
    transforms = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])],
            to_rgb=True)
    return np.expand_dims(transforms(image_path), axis=0)


def export_onnx_mobilenetv2(save_path):
    ''' export PaddlePaddle model to ONNX format

    Args:
        save_path(str): Path to save exported ONNX model

    Returns:
        None
    '''
    model = hub.Module(name="resnet50_vd_imagenet_ssld")
    input_spec = paddle.static.InputSpec(
            [1, 3, 224, 224], "float32", "image")
    paddle.onnx.export(model, save_path,
                       input_spec=[input_spec],
                       opset_version=10)


if __name__ == '__main__':
    save_path = './resnet50'
    image_file = './data/cat.jpg'
    labels = open('./data/labels.txt').read().strip().split('\n')
    model = export_onnx_mobilenetv2(save_path)

    # load mobilenetv2 use cv.dnn
    net = cv.dnn.readNetFromONNX(save_path + '.onnx')
    # read and preprocess image file
    im = preprocess(image_file)
    # inference
    net.setInput(im)
    result = net.forward(['save_infer_model/scale_0.tmp_0'])
    # post process
    class_id = np.argmax(result[0])
    label = labels[class_id]
    print("Image: {}".format(image_file))
    print("Predict Category: {}".format(label))
