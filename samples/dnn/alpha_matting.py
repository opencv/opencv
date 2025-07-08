"""
MODNet Alpha Matting with OpenCV DNN

This sample demonstrates human portrait alpha matting using MODNet model.
MODNet is a trimap-free portrait matting method that can produce high-quality
alpha mattes for portrait images in real-time.

Usage: 
    python alpha_matting.py --input=image.jpg
    python alpha_matting.py --input=video.mp4
    python alpha_matting.py (uses webcam)
"""

import cv2 as cv
import numpy as np
import argparse
import os
from common import *

def get_args_parser(func_args):
    backends = ("default", "openvino", "opencv", "vkcom", "cuda")
    targets = ("cpu", "opencl", "opencl_fp16", "ncs2_vpu", "hddl_vpu", "vulkan", "cuda", "cuda_fp16")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                        help='An optional path to file with preprocessing parameters.')
    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.', default=0, required=False)
    parser.add_argument('--backend', default="default", type=str, choices=backends,
                    help="Choose one of computation backends: "
                         "default: automatically (by default), "
                         "openvino: Intel's Deep Learning Inference Engine, "
                         "opencv: OpenCV implementation, "
                         "vkcom: VKCOM, "
                         "cuda: CUDA")
    parser.add_argument('--target', default="cpu", type=str, choices=targets,
                    help="Choose one of target computation devices: "
                         "cpu: CPU target (by default), "
                         "opencl: OpenCL, "
                         "opencl_fp16: OpenCL fp16 (half-float precision), "
                         "ncs2_vpu: NCS2 VPU, "
                         "hddl_vpu: HDDL VPU, "
                         "vulkan: Vulkan, "
                         "cuda: CUDA, "
                         "cuda_fp16: CUDA fp16 (half-float precision)")

    args, _ = parser.parse_known_args()
    add_preproc_args(args.zoo, parser, 'alpha_matting', 'modnet')
    parser = argparse.ArgumentParser(parents=[parser],
                                     description='''
        To run:
            python alpha_matting.py --input=path/to/your/input/image/or/video
            python alpha_matting.py (uses webcam)

        Model path can also be specified using --model argument
        ''', formatter_class=argparse.RawTextHelpFormatter)
    return parser.parse_args(func_args)

def postprocess_output(image, alpha_output):
    """Process model output to create alpha mask and composite."""
    h, w = image.shape[:2]
    
    alpha = alpha_output[0, 0] if alpha_output.ndim == 4 else alpha_output[0]
    alpha = cv.resize(alpha, (w, h))
    alpha = np.clip(alpha, 0, 1)
    
    alpha_mask = (alpha * 255).astype(np.uint8)
    
    alpha_3ch = cv.merge([alpha, alpha, alpha])
    composite = (image.astype(np.float32) * alpha_3ch).astype(np.uint8)
    
    return alpha_mask, composite

def loadModel(args, engine):
    net = cv.dnn.readNetFromONNX(args.model, engine)
    net.setPreferableBackend(get_backend_id(args.backend))
    net.setPreferableTarget(get_target_id(args.target))
    return net

def apply_modnet(model, image):
    out = model.forward()
    alpha_mask, composite = postprocess_output(image, out)
    t, _ = model.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv.putText(alpha_mask, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv.putText(composite, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv.imshow("Alpha Mask", alpha_mask)
    cv.imshow("Composite", composite)

def main(func_args=None):
    args = get_args_parser(func_args)
    engine = cv.dnn.ENGINE_AUTO
    if args.backend != "default" or args.target != "cpu":
        engine = cv.dnn.ENGINE_CLASSIC

    cap = cv.VideoCapture(cv.samples.findFile(args.input) if args.input else 0)
    if not cap.isOpened():
        print("Failed to open the input video")
        exit(-1)
    cv.namedWindow('Input', cv.WINDOW_AUTOSIZE)
    cv.namedWindow('Alpha Mask', cv.WINDOW_AUTOSIZE)
    cv.namedWindow('Composite', cv.WINDOW_AUTOSIZE)
    cv.moveWindow('Alpha Mask', 200, 50)
    cv.moveWindow('Composite', 400, 50)

    net = None
    if os.getenv('OPENCV_SAMPLES_DATA_PATH') is not None or hasattr(args, 'model'):
        try:
            args.model = findModel(args.model, args.sha1)
            net = loadModel(args, engine)
        except Exception as e:
            print("[WARN] Model file not provided or not found. Pass model using --model=/path/to/modnet.onnx")
            exit(-1)
    else:
        print("[WARN] Model file not provided. Pass model using --model=/path/to/modnet.onnx")
        exit(-1)

    while True:
        hasFrame, image = cap.read()
        if not hasFrame:
            print("Press any key to exit")
            cv.waitKey(0)
            break
        
        inp = cv.dnn.blobFromImage(image, args.scale, (args.width, args.height), args.mean, swapRB=args.rgb)
        net.setInput(inp)
        apply_modnet(net, image)

        cv.imshow("Input", image)
        key = cv.waitKey(1)
        if key == 27 or key == ord('q'):
            break
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()