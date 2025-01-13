#!/usr/bin/env python
'''
This sample inpaints the masked area in the given image.

Copyright (C) 2025, Bigvision LLC.

How to use:
    Sample command to run:
        `python inpainting.py`
    The system will ask you to draw the mask to be inpainted

    You can download lama inpainting model using
        `python download_models.py lama`

    References:
      Github: https://github.com/advimman/lama
      ONNX model: https://huggingface.co/Carve/LaMa-ONNX/blob/main/lama_fp32.onnx

      ONNX model was further quantized using block quantization from [opencv_zoo](https://github.com/opencv/opencv_zoo)

    Set environment variable OPENCV_DOWNLOAD_CACHE_DIR to point to the directory where models are downloaded. Also, point OPENCV_SAMPLES_DATA_PATH to opencv/samples/data.
'''
import argparse
import os.path
import numpy as np
import cv2 as cv
from common import *

def help():
    print(
        '''
        Use this script for image inpainting using OpenCV.

        Firstly, download required models i.e. lama using `download_models.py` (if not already done). Set environment variable OPENCV_DOWNLOAD_CACHE_DIR to specify where models should be downloaded. Also, point OPENCV_SAMPLES_DATA_PATH to opencv/samples/data.

        To run:
        Example: python inpainting.py lama

        Inpainting model path can also be specified using --model argument.
        '''
    )

def keyboard_shorcuts():
    print('''
    Keyboard Shorcuts:
        Press 'i' to increase brush size.
        Press 'd' to decrease brush size.
        Press ' ' (space bar) after selecting area to be inpainted.
        Press ESC to terminate the program.
    '''
    )

def get_args_parser():
    backends = ("default", "openvino", "opencv", "vkcom", "cuda")
    targets = ("cpu", "opencl", "opencl_fp16", "ncs2_vpu", "hddl_vpu", "vulkan", "cuda", "cuda_fp16")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                        help='An optional path to file with preprocessing parameters.')
    parser.add_argument('--input', '-i', default="baboon.jpg", help='Path to image file.', required=False)
    parser.add_argument('--backend', default="default", type=str, choices=backends,
            help="Choose one of computation backends: "
            "default: automatically (by default), "
            "openvino: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
            "opencv: OpenCV implementation, "
            "vkcom: VKCOM, "
            "cuda: CUDA, "
            "webnn: WebNN")
    parser.add_argument('--target', default="cpu", type=str, choices=targets,
            help="Choose one of target computation devices: "
            "cpu: CPU target (by default), "
            "opencl: OpenCL, "
            "opencl_fp16: OpenCL fp16 (half-float precision), "
            "ncs2_vpu: NCS2 VPU, "
            "hddl_vpu: HDDL VPU, "
            "vulkan: Vulkan, "
            "cuda: CUDA, "
            "cuda_fp16: CUDA fp16 (half-float preprocess)")
    args, _ = parser.parse_known_args()
    add_preproc_args(args.zoo, parser, 'inpainting', prefix="", alias="lama")
    parser = argparse.ArgumentParser(parents=[parser],
                                        description='Image inpainting using OpenCV.',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser.parse_args()


drawing = False
mask_gray = None
brush_size = 15

def draw_mask(event, x, y, flags, param):
    global drawing, mask_gray, brush_size
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            cv.circle(mask_gray, (x, y), brush_size, (255), thickness=-1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False

def main():
    global mask_gray, brush_size

    print("Model loading...")

    if hasattr(args, 'help'):
        help()
        exit(1)

    args.model = findModel(args.model, args.sha1)

    engine = cv.dnn.ENGINE_AUTO

    if args.backend != "default" or args.target != "cpu":
        engine = cv.dnn.ENGINE_CLASSIC

    net = cv.dnn.readNetFromONNX(args.model, engine)
    net.setPreferableBackend(get_backend_id(args.backend))
    net.setPreferableTarget(get_target_id(args.target))

    input_image = cv.imread(findFile(args.input))
    mask_gray = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)


    stdSize = 0.6
    stdWeight = 2
    stdImgSize = 512
    imgWidth = min(input_image.shape[:2])
    fontSize = min(1.5, (stdSize*imgWidth)/stdImgSize)
    fontThickness = max(1,(stdWeight*imgWidth)//stdImgSize)
    aspect_ratio = input_image.shape[0]/input_image.shape[1]


    keyboard_shorcuts()
    cv.namedWindow("Draw Mask")
    cv.setMouseCallback("Draw Mask", draw_mask)

    label = "Draw the mask on the image. Press space bar when done."
    labelSize, _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, fontSize, fontThickness)
    while True:
        display_image = input_image.copy()
        overlay = input_image.copy()

        alpha = 0.5
        cv.rectangle(overlay, (0, 0), (labelSize[0]+10, labelSize[1]+int(30*fontSize)), (255, 255, 255), cv.FILLED)
        cv.addWeighted(overlay, alpha, display_image, 1 - alpha, 0, display_image)

        cv.putText(display_image, label, (10, int(25*fontSize)), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)
        cv.putText(display_image, "Press 'i' to increase and 'd' to decrease brush size.", (10, int(50*fontSize)), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)
        display_image[mask_gray > 0] = [255, 255, 255]
        cv.imshow("Draw Mask", display_image)

        key = cv.waitKey(1) & 0xFF
        if key == ord('i'):  # Increase brush size
            brush_size += 1
            print(f"Brush size increased to {brush_size}")
        elif key == ord('d'):  # Decrease brush size
            brush_size = max(1, brush_size - 1)
            print(f"Brush size decreased to {brush_size}")
        elif key == ord(' '): # Press space bar to finish drawing
            break
        elif key == 27:
            exit()

    cv.destroyAllWindows()

    print("Processing image...")
    image_blob = cv.dnn.blobFromImage(input_image, args.scale, (args.width, args.height), args.mean, args.rgb, False)
    mask_blob = cv.dnn.blobFromImage(mask_gray, scalefactor=1.0, size=(args.width, args.height), mean=(0,), swapRB=False, crop=False)
    mask_blob = (mask_blob > 0).astype(np.float32)

    net.setInput(image_blob, "image")
    net.setInput(mask_blob, "mask")

    output = net.forward()

    # Postprocessing
    output_image = output[0]
    output_image = np.transpose(output_image, (1, 2, 0))
    output_image = (output_image).astype(np.uint8)
    width = output_image.shape[1]
    height = int(width*aspect_ratio)
    output_image = cv.resize(output_image, (width, height))
    print("Process completed!!!")

    cv.imshow("Inpainted Output", output_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return

if __name__ == '__main__':
    args = get_args_parser()
    main()