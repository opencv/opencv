import os
import cv2
import numpy as np
import argparse

import torch
import torch.onnx
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights


def get_pytorch_onnx_model(pytorch_model):
    # define the directory for further converted model save
    onnx_model_path = "models"
    # define the name of further converted model
    onnx_model_name = "fcn_resnet101.onnx"

    # create directory for further converted model
    os.makedirs(onnx_model_path, exist_ok=True)

    # get full path to the converted model
    full_model_path = os.path.join(onnx_model_path, onnx_model_name)

    input_tensor = torch.randn(1, 3, 500, 500)

    # model export into ONNX format
    torch.onnx.export(
        pytorch_model,
        input_tensor,
        full_model_path,
        verbose=False,
        input_names=['input'],
        output_names=['output'],
        opset_version=11
    )

    return full_model_path

def get_preprocessed_img(img_path):

    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    input_img = input_img.astype(np.float32)

    # Coco mean and std parameters
    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    scale = 1 / 255.0
    std = [0.229, 0.224, 0.225]

    # prepare input blob to fit the model input:
    # 1. subtract mean
    # 2. scale to set pixel values from 0 to 1
    input_blob = cv2.dnn.blobFromImage(
        image=input_img,
        scalefactor=scale,
        size=(500, 500),  # img target size
        mean=mean,
        swapRB=True,  # BGR -> RGB
        crop=False  # center crop
    )

    input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    return input_blob

def visualization(origin_image, predict_img, save_path):

    # Color mapping for different classes, the color order are (B, G, R)
    voc_color_map_for_opencv = {
        0: (0, 0, 0),        # Background
        1: (0, 0, 128),      # Aeroplane
        2: (0, 128, 0),      # Bicycle
        3: (0, 128, 128),    # Bird
        4: (128, 0, 0),      # Boat 
        5: (128, 0, 128),    # Bottle
        6: (128, 128, 0),    # Bus
        7: (128, 128, 128),  # Car
        8: (0, 0, 64),       # Cat
        9: (0, 0, 192),      # Chair
        10: (0, 128, 64),    # Cow
        11: (0, 128, 192),   # Diningtable 
        12: (128, 0, 64),    # Dog 
        13: (128, 0, 192),   # Horse 
        14: (128, 128, 64),  # Motorbike 
        15: (128, 128, 192), # Person 
        16: (0, 64, 0),      # Pottedplant 
        17: (0, 64, 128),    # Sheep 
        18: (0, 192, 0),     # Sofa 
        19: (0, 192, 128),   # Train 
        20: (128, 64, 0)     # Tvmonitor 
    }

    segmentation_image = np.zeros((500, 500, 3), dtype=np.uint8)

    # Map class IDs to colors in the segmentation image
    for index in voc_color_map_for_opencv:
        segmentation_image[predict_img == index] = voc_color_map_for_opencv[index]
    
    # Blend the segmentation result with the original image
    result = cv2.addWeighted(origin_image, 0.6, segmentation_image, 0.4, 30)

    # Save the result image
    cv2.imwrite(save_path, result)
    print(f'Success predict the image at {save_path}')

def get_opencv_dnn_prediction(opencv_net, original_image, preproc_img):
    # set OpenCV DNN input
    opencv_net.setInput(preproc_img)

    # OpenCV DNN inference
    predict = opencv_net.forward()
    out = np.argmax(predict[0], axis=0)

    # Visualize the segmentation result and save it
    visualization(original_image, out, "./opencv_dnn_prediction.jpg")

def main():
    # initialize fcn_resnet50 with default weight
    weights = FCN_ResNet101_Weights.DEFAULT
    pytorch_model = fcn_resnet101(weights=weights)
    pytorch_model.eval()
    
    # get the path to the converted into ONNX PyTorch model
    full_model_path = get_pytorch_onnx_model(pytorch_model)

    # read converted .onnx model with OpenCV API
    opencv_net = cv2.dnn.readNetFromONNX(full_model_path)
    
    # Input image should be square
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input', help='Input picture which you want to predict')
    args, _ = parser.parse_known_args()

    original_image = cv2.imread(args.input, cv2.IMREAD_COLOR) 
    original_image = cv2.resize(original_image, (500, 500))
    # get preprocessed image
    input_img = get_preprocessed_img(args.input)
    
    # obtain OpenCV DNN predictions
    get_opencv_dnn_prediction(opencv_net, original_image, input_img)

if __name__ == "__main__":
    main()