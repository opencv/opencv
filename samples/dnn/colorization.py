# Script is based on https://github.com/richzhang/colorization/blob/master/colorization/colorize.py
# To download the onnx model, see: https://storage.googleapis.com/ailia-models/colorization/colorizer.onnx
# python colorization.py --onnx_model_path colorizer.onnx --input ansel_adams3.jpg
import numpy as np
import argparse
import cv2 as cv
import numpy as np

def parse_args():
    backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
                cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_BACKEND_VKCOM, cv.dnn.DNN_BACKEND_CUDA)
    targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD,
               cv.dnn.DNN_TARGET_HDDL, cv.dnn.DNN_TARGET_VULKAN, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16)

    parser = argparse.ArgumentParser(description='iColor: deep interactive colorization')
    parser.add_argument('--input', default='baboon.jpg',help='Path to image.')
    parser.add_argument('--onnx_model_path', help='Path to onnx model', required=True)
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help="Choose one of computation backends: "
                             "%d: automatically (by default), "
                             "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                             "%d: OpenCV implementation, "
                             "%d: VKCOM, "
                             "%d: CUDA" % backends)
    parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                        help='Choose one of target computation devices: '
                             '%d: CPU target (by default), '
                             '%d: OpenCL, '
                             '%d: OpenCL fp16 (half-float precision), '
                             '%d: NCS2 VPU, '
                             '%d: HDDL VPU, '
                             '%d: Vulkan, '
                             '%d: CUDA, '
                             '%d: CUDA fp16 (half-float preprocess)'% targets)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    img_gray=cv.imread(cv.samples.findFile(args.input),cv.IMREAD_GRAYSCALE)

    img_gray_rs = cv.resize(img_gray, (256, 256), interpolation=cv.INTER_CUBIC)
    img_gray_rs = img_gray_rs.astype(np.float32)  # Convert to float to avoid data overflow
    img_gray_rs *= (100.0 / 255.0)      # Scale L channel to 0-100 range

    onnx_model_path = args.onnx_model_path  # Update this path to your ONNX model's path
    engine = cv.dnn.ENGINE_AUTO
    session = cv.dnn.readNetFromONNX(onnx_model_path, engine)
    session.setPreferableBackend(args.backend)
    session.setPreferableTarget(args.target)

    # Process each image in the batch (assuming batch processing is needed)
    blob = cv.dnn.blobFromImage(img_gray_rs, swapRB=False)  # Adjust swapRB according to your model's training
    session.setInput(blob)
    result_numpy = np.array(session.forward()[0])

    if result_numpy.shape[0] == 2:
        # Transpose result_numpy to shape (H, W, 2)
        ab = result_numpy.transpose((1, 2, 0))
    else:
        # If it's already (H, W, 2), assign it directly
        ab = result_numpy


    # Resize ab to match img_gray's dimensions if they are not the same
    h, w = img_gray.shape
    if ab.shape[:2] != (h, w):
        ab_resized = cv.resize(ab, (w, h), interpolation=cv.INTER_LINEAR)
    else:
        ab_resized = ab

    # Expand dimensions of L to match ab's dimensions
    img_l_expanded = np.expand_dims(img_gray, axis=-1)

    # Concatenate L with AB to get the LAB image
    lab_image = np.concatenate((img_l_expanded, ab_resized), axis=-1)

    # Convert the Lab image to a 32-bit float format
    lab_image = lab_image.astype(np.float32)

    # Normalize L channel to the range [0, 100] and AB channels to the range [-127, 127]
    lab_image[:, :, 0] *= (100.0 / 255.0)  # Rescale L channel
    #lab_image[:, :, 1:] -= 128              # Shift AB channels

    # Convert the LAB image to BGR
    image_bgr_out = cv.cvtColor(lab_image, cv.COLOR_Lab2BGR)
    cv.imshow("input image",img_gray)
    cv.imshow("output image",image_bgr_out)
    cv.waitKey(0)