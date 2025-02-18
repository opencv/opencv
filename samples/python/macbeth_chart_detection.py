import cv2 as cv
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Basic chart detection using neural network')

    parser.add_argument('-t', type=int, default=0,
                        help='chartType: 0-Standard, 1-DigitalSG, 2-Vinyl, default:0')
    parser.add_argument('-m', type=str, default='',
                        help='File path of the .pb model file')
    parser.add_argument('-pb', type=str, default='',
                        help='File path of the .pbtxt file')
    parser.add_argument('-v', type=str, default='',
                        help='Input video file. If omitted, input comes from camera')
    parser.add_argument('-ci', type=int, default=0,
                        help='Camera id if input does not come from video (-v)')
    parser.add_argument('-nc', type=int, default=1,
                        help='Maximum number of charts in the image')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use this flag to run DNN inference on CUDA')

    args = parser.parse_args()

    if not (0 <= args.t <= 2):
        raise ValueError("chartType must be 0, 1, or 2")

    chartType = args.t

    model_path = args.m
    pbtxt_path = args.pb
    camera_id = args.ci
    max_charts = args.nc
    video_path = args.v
    use_gpu = args.use_gpu

    # Open input video stream (camera or file)
    if video_path:
        cap = cv.VideoCapture(video_path)
        wait_time = 10
    else:
        cap = cv.VideoCapture(camera_id)
        wait_time = 10

    if model_path and pbtxt_path:
        # Load the DNN from TensorFlow model
        net = cv.dnn.readNetFromTensorflow(model_path, pbtxt_path)

        if use_gpu:
            net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

        detector = cv.mcc_CCheckerDetector.create(net)
        print("Detecting checkers using neural network.")
    else:
        detector = cv.mcc_CCheckerDetector.create()
        print("Detecting checkers using default method (no DNN).")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No more frames or failed to grab frame.")
            break

        image_copy = frame.copy()

        if not detector.process(frame, chartType, max_charts, True):
            print("ChartColor not detected.")
        else:
            checkers = detector.getListColorChecker()
            for checker in checkers:
                cdraw = cv.mcc_CCheckerDraw.create(checker)
                cdraw.draw(frame)

        # Show results
        cv.imshow("image result | Press ESC to quit", frame)
        cv.imshow("original", image_copy)
        key = cv.waitKey(wait_time) & 0xFF
        if key == 27:
            break

    cv.waitKey(0)
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
