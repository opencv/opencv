import cv2 as cv
import numpy as np
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dnn.common import *

def get_args_parser(func_args):
    backends = ("default", "openvino", "opencv", "vkcom", "cuda", "webnn")
    targets = ("cpu", "opencl", "opencl_fp16", "vpu", "vulkan", "cuda", "cuda_fp16")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dnn', 'models.yml'),
                        help='An optional path to file with preprocessing parameters.')
    parser.add_argument('--input', default='mcc_ccm_test.jpg', help='Path to input image for computing CCM')
    parser.add_argument('--query', default='baboon.jpg', help='Path to query image to apply color correction')
    parser.add_argument('--ccm_file', help='Path to YAML file containing pre-computed CCM parameters')
    parser.add_argument('--chart_type', type=int, default=0,
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
                         "vpu: VPU, "
                         "vulkan: Vulkan, "
                         "cuda: CUDA, "
                         "cuda_fp16: CUDA fp16 (half-float preprocess)")

    args, _ = parser.parse_known_args()
    add_preproc_args(args.zoo, parser, 'mcc', 'mcc')
    parser = argparse.ArgumentParser(parents=[parser],
                                     description='''
        To run:
            Default (compute new CCM):
                python color_correction_model.py --input=path/to/your/input/image --query=path/to/query/image
            DNN model:
                python color_correction_model.py mcc --input=path/to/your/input/image --query=path/to/query/image
            Using pre-computed CCM:
                python color_correction_model.py --ccm_file=path/to/ccm_output.yaml --query=path/to/query/image

        Model path can also be specified using --model argument. And config path can be specified using --config.
        ''', formatter_class=argparse.RawTextHelpFormatter)
    return parser.parse_args(func_args)

def process_frame(frame, detector, num_charts):
    if not detector.process(frame, num_charts):
        return None

    checkers = detector.getListColorChecker()
    src = checkers[0].getChartsRGB(False)

    return src

def main(func_args=None):
    args = get_args_parser(func_args)

    if not (0 <= args.chart_type <= 2):
        raise ValueError("chartType must be 0, 1, or 2")

    # Validate arguments based on whether using pre-computed CCM
    if args.ccm_file:
        if not args.query:
            print("[ERROR] Query image path must be provided when using pre-computed CCM.")
            return -1
    else:
        if not args.input:
            print("[ERROR] Input image path must be provided when computing new CCM.")
            return -1

    # Read query image
    query_image = None
    if args.query:
        query_image = cv.imread(findFile(args.query))
        if query_image is None:
            print("[ERROR] Unable to read query image.")
            return -1

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

    # Create color correction model
    model = cv.ccm.ColorCorrectionModel()

    if args.ccm_file:
        # Load CCM from YAML file
        fs = cv.FileStorage(args.ccm_file, cv.FileStorage_READ)
        if not fs.isOpened():
            print(f"[ERROR] Unable to open CCM file: {args.ccm_file}")
            return -1
        model.read(fs.getNode("ColorCorrectionModel"))
        fs.release()
        print(f"Loaded CCM from file: {args.ccm_file}")
    else:
        # Read input image for computing new CCM
        image = cv.imread(findFile(args.input))
        if image is None:
            print("[ERROR] Unable to read input image.")
            return -1

        # Create color checker detector
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

        detector.setColorChartType(args.chart_type)

        # Process image to detect color checker
        src = process_frame(image, detector, args.num_charts)
        if src is None:
            print("No chart detected in the input image!")
            return -1

        print("Actual colors:", src)

        # Convert to double and normalize
        src = src.astype(np.float64) / 255.0

        # Create and configure color correction model
        model = cv.ccm.ColorCorrectionModel(src, cv.ccm.COLORCHECKER_MACBETH)
        model.setCcmType(cv.ccm.CCM_LINEAR)
        model.setDistance(cv.ccm.DISTANCE_CIE2000)
        model.setLinearization(cv.ccm.LINEARIZATION_GAMMA)
        model.setLinearizationGamma(2.2)

        # Compute color correction matrix
        ccm = model.compute()
        print("Computed CCM Matrix:\n", ccm)
        print("Loss:", model.getLoss())

        # Save model parameters to YAML file
        fs = cv.FileStorage("ccm_output.yaml", cv.FileStorage_WRITE)
        model.write(fs)
        fs.release()
        print("Model parameters saved to ccm_output.yaml")

        # Set query image for correction if not provided
        if query_image is None:
            print("[WARN] No query image provided, applying color correction on input image")
            query_image = image.copy()

    # Apply correction to query image
    model.setRGB(True)
    calibrated_image = np.empty_like(query_image)
    model.correctImage(query_image, calibrated_image)

    cv.imshow("Original Image", query_image)
    cv.imshow("Corrected Image", calibrated_image)
    cv.waitKey(0)

    return 0

if __name__ == "__main__":
    main()
