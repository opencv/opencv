import cv2 as cv
import numpy as np
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dnn.common import *

def get_args_parser(func_args):
    backends = ("default", "openvino", "opencv", "vkcom", "cuda")
    targets = ("cpu", "opencl", "opencl_fp16", "ncs2_vpu", "hddl_vpu", "vulkan", "cuda", "cuda_fp16")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dnn', 'models.yml'),
                        help='An optional path to file with preprocessing parameters.')
    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.', default=0, required=False)
    parser.add_argument('--method', help='choose method: dexined or canny', default='canny', required=False)
    parser.add_argument('--type', type=int, default=0,
                        help='chartType: 0-Standard, 1-DigitalSG, 2-Vinyl, default:0')
    parser.add_argument('--num_charts', type=int, default=1,
                        help='Maximum number of charts in the image')
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
    add_preproc_args(args.zoo, parser, 'mcc', 'mcc')
    parser = argparse.ArgumentParser(parents=[parser],
                                     description='''
        To run:
            Default:
                python macbeth_chart_detection.py --input=path/to/your/input/image/or/video (don't give --input flag if want to use device camera)
            DNN model:
                python macbeth_chart_detection.py mcc --input=path/to/your/input/image/or/video

        Model path can also be specified using --model argument. And config path can be specified using --config.
        ''', formatter_class=argparse.RawTextHelpFormatter)
    return parser.parse_args(func_args)

def process_frame(frame, detector, chart_type, num_charts):
    image_copy = frame.copy()
    if not detector.process(frame, chart_type, num_charts, True):
        print("ChartColor not detected.")
    else:
        checkers = detector.getListColorChecker()
        detector.draw(checkers, frame)

        src = checkers[0].getChartsRGB(False)
        tgt = np.empty((src.shape), dtype=np.int32)
        detector.getRefColors(cv.mcc.MCC24, tgt)

        cv.imshow("image result | Press ESC to quit", frame)
        cv.imshow("original", image_copy)

        return src, tgt
    return None, None

def main(func_args=None):
    args = get_args_parser(func_args)

    if not (0 <= args.type <= 2):
        raise ValueError("chartType must be 0, 1, or 2")

    if os.getenv('OPENCV_SAMPLES_DATA_PATH') is not None:
        try:
            args.model = findModel(args.model, args.sha1)
            args.config = findModel(args.config, args.config_sha1)
        except:
            print("[WARN] Model file not provided, using default detector. Pass model using --model and config using --config to use dnn based detector.\n\n")
            args.model = None
            args.config = None
    else:
        args.model = None
        args.config = None
        print("[WARN] Model file not provided, using default detector. Pass model using --model and config using --config to use dnn based detector. Or, set OPENCV_SAMPLES_DATA_PATH environment variable.\n\n")

    if args.model and args.config:
        # Load the DNN from TensorFlow model
        engine = cv.dnn.ENGINE_AUTO
        if args.backend != "default" or args.target != "cpu":
            engine = cv.dnn.ENGINE_CLASSIC
        net = cv.dnn.readNetFromTensorflow(args.model, args.config, engine)
        net.setPreferableBackend(get_backend_id(args.backend))
        net.setPreferableTarget(get_target_id(args.target))

        detector = cv.mcc_CCheckerDetector.create(net)
        print("Detecting checkers using neural network.")
    else:
        detector = cv.mcc_CCheckerDetector.create()
        print("Detecting checkers using default method (no DNN).")

    is_video = True

    if args.input:
        image = cv.imread(findFile(args.input))
        if image is not None:
            is_video = False
        else:
            cap = cv.VideoCapture(findFile(args.input))
    else:
        cap = cv.VideoCapture(0)

    if is_video:
        print("To print the actual colors and reference colors for current frame press SPACEBAR. To resume press SPACEBAR again")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            src, tgt = process_frame(frame, detector, args.type, args.num_charts)

            key = cv.waitKey(10) & 0xFF
            if key == ord(' '):
                print("Actual colors: ", src)
                print("Reference colors: ", tgt)
                print("Press spacebar to resume.")

                pause_key = cv.waitKey(0)
                if pause_key == 27:
                    exit(0)
                print("Resumed! Processing continues...")
            elif key == 27:
                exit(0)
        print("Actual colors: ", src)
        print("Reference colors: ", tgt)
    else:
        src, tgt = process_frame(image, detector, args.type, args.num_charts)
        print("Actual colors: ", src)
        print("Reference colors: ", tgt)
        cv.waitKey(0)

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
