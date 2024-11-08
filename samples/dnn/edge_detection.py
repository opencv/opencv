'''
This sample demonstrates edge detection with dexined and canny edge detection techniques.
For switching between deep learning based model(dexined) and canny edge detector, press space bar in case of video. In case of image, pass the argument --method for switching between dexined and canny.
'''

import cv2 as cv
import argparse
import numpy as np
from common import *

def get_args_parser(func_args):
    backends = ("default", "openvino", "opencv", "vkcom", "cuda")
    targets = ("cpu", "opencl", "opencl_fp16", "ncs2_vpu", "hddl_vpu", "vulkan", "cuda", "cuda_fp16")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                        help='An optional path to file with preprocessing parameters.')
    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.', default=0, required=False)
    parser.add_argument('--method', help='choose method: dexined or canny', default='canny', required=False)
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
    add_preproc_args(args.zoo, parser, 'edge_detection', 'dexined')
    parser = argparse.ArgumentParser(parents=[parser],
                                     description='''
        To run:
            Canny:
                python edge_detection.py --input=path/to/your/input/image/or/video (don't give --input flag if want to use device camera)
            Dexined:
                python edge_detection.py dexined --input=path/to/your/input/image/or/video

        "In case of video input, for switching between deep learning based model (Dexined) and Canny edge detector, press space bar. Pass as argument in case of image input."

        Model path can also be specified using --model argument
        ''', formatter_class=argparse.RawTextHelpFormatter)
    return parser.parse_args(func_args)

threshold1 = 0
threshold2 = 50
blur_amount = 5
gray = None

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def post_processing(output, shape):
    h, w = shape
    preds = []
    for p in output:
        img = sigmoid(p)
        img = np.squeeze(img)
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        img = cv.resize(img, (w, h))
        preds.append(img)
    fuse = preds[-1]
    ave = np.array(preds, dtype=np.float32)
    ave = np.uint8(np.mean(ave, axis=0))
    return fuse, ave

def apply_canny(image):
    global threshold1, threshold2, blur_amount
    kernel_size = 2 * blur_amount + 1
    blurred = cv.GaussianBlur(image, (kernel_size, kernel_size), 0)
    result = cv.Canny(blurred, threshold1, threshold2)
    cv.imshow('Output', result)

def setupCannyWindow(image):
    global gray
    cv.destroyWindow('Output')
    cv.namedWindow('Output', cv.WINDOW_AUTOSIZE)
    cv.moveWindow('Output', 200, 50)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    cv.createTrackbar('thrs1', 'Output', threshold1, 255, lambda value: [globals().__setitem__('threshold1', value), apply_canny(gray)])
    cv.createTrackbar('thrs2', 'Output', threshold2, 255, lambda value: [globals().__setitem__('threshold2', value), apply_canny(gray)])
    cv.createTrackbar('blur', 'Output', blur_amount, 20, lambda value: [globals().__setitem__('blur_amount', value), apply_canny(gray)])

def loadModel(args, engine):
    net = cv.dnn.readNetFromONNX(args.model, engine)
    net.setPreferableBackend(get_backend_id(args.backend))
    net.setPreferableTarget(get_target_id(args.target))
    return net

def apply_dexined(model, image):
    out = model.forward()
    result,_ = post_processing(out, image.shape[:2])
    t, _ = model.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    cv.putText(result, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv.imshow("Output", result)

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
    cv.namedWindow('Output', cv.WINDOW_AUTOSIZE)
    cv.moveWindow('Output', 200, 50)

    method = args.method
    if os.getenv('OPENCV_SAMPLES_DATA_PATH') is not None or hasattr(args, 'model'):
        try:
            args.model = findModel(args.model, args.sha1)
            method = 'dexined'
        except:
            print("[WARN] Model file not provided, using canny instead. Pass model using --model=/path/to/dexined.onnx to use dexined model.")
            method = 'canny'
            args.model = None
    else:
        print("[WARN] Model file not provided, using canny instead. Pass model using --model=/path/to/dexined.onnx to use dexined model.")
        method = 'canny'

    if method == 'canny':
        dummy = np.zeros((512, 512, 3), dtype="uint8")
        setupCannyWindow(dummy)
    net = None
    if method == "dexined":
        net = loadModel(args, engine)
    while cv.waitKey(1) < 0:
        hasFrame, image = cap.read()
        if not hasFrame:
            print("Press any key to exit")
            cv.waitKey(0)
            break
        if method == "canny":
            global gray
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            apply_canny(gray)
        elif method == "dexined":
            inp = cv.dnn.blobFromImage(image, args.scale, (args.width, args.height), args.mean, swapRB=args.rgb, crop=False)

            net.setInput(inp)
            apply_dexined(net, image)

        cv.imshow("Input", image)
        key = cv.waitKey(30)
        if key == ord(' ') and method == 'canny':
            if hasattr(args, 'model') and args.model is not None:
                print("model: ", args.model)
                method = "dexined"
                if net is None:
                    net = loadModel(args, engine)
                cv.destroyWindow('Output')
                cv.namedWindow('Output', cv.WINDOW_AUTOSIZE)
                cv.moveWindow('Output', 200, 50)
            else:
                print("[ERROR] Provide model file using --model to use dexined. Download model using python download_models.py dexined from dnn samples directory")
        elif key == ord(' ') and method=='dexined':
            method = "canny"
            setupCannyWindow(image)
        elif key == 27 or key == ord('q'):
            break
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()