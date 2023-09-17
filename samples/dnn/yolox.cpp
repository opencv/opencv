/*
YOLOX is an anchor-free version of YOLO, with a simpler design but better performance! It aims to bridge the gap between research and industrial communities. YOLOX is a high-performing object detector, an improvement to the existing YOLO series. YOLO series are in constant exploration of techniques to improve the object detection techniques for optimal speed and accuracy trade-off for real-time applications.

Key features of the YOLOX object detector

    Anchor-free detectors significantly reduce the number of design parameters
    A decoupled head for classification, regression, and localization improves the convergence speed
    SimOTA advanced label assignment strategy reduces training time and avoids additional solver hyperparameters
    Strong data augmentations like MixUp and Mosiac to boost YOLOX performance

    model can be download https://github.com/opencv/opencv_zoo/blob/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx
    or https://github.com/opencv/opencv_zoo/blob/main/models/object_detection_yolox/object_detection_yolox_2022nov_int8.onnx
*/
#include <iostream>
#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>


#include <vector>
#include <string>
#include <utility>


using namespace std;
using namespace cv;
using namespace dnn;

vector< pair<dnn::Backend, dnn::Target> > backendTargetPairs = {
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_OPENCV, dnn::DNN_TARGET_CPU),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CUDA, dnn::DNN_TARGET_CUDA),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CUDA, dnn::DNN_TARGET_CUDA_FP16),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_TIMVX, dnn::DNN_TARGET_NPU),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CANN, dnn::DNN_TARGET_NPU) };

std::vector<std::string> labelYolox = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };


std::string keys =
"{ help  h          |                                               | Print help message. }"
"{ model m          | object_detection_yolox_2022nov.onnx           | Usage: Path to the model, defaults to object_detection_yolox_2022nov.onnx  }"
"{ input i          |                                               | Path to input image or video file. Skip this argument to capture frames from a camera.}"
"{ confidence c     | 0.5                                           | Class confidence }"
"{ nms              | 0.5                                           | Enter nms IOU threshold }"
"{ vis v            | 1                                             | Specify to open a window for result visualization. This flag is invalid when using camera. }"
"{ backend bt       | 0                                             | Choose one of computation backends: "
"0: (default) OpenCV implementation + CPU, "
"1: CUDA + GPU (CUDA), "
"2: CUDA + GPU (CUDA FP16), "
"3: TIM-VX + NPU, "
"4: CANN + NPU}";

pair<Mat, double> letterBox(Mat srcimg, Size targetSize = Size(640, 640));
Mat unLetterBox(Mat bbox, double letterboxScale);
Mat visualize(Mat dets, Mat srcimg, double letterbox_scale, double fps = -1);

pair<Mat, double> letterBox(Mat srcimg, Size targetSize)
{
    Mat paddedImg(targetSize.height, targetSize.width, CV_32FC3, Scalar::all(114.0));
    Mat resizeImg;

    double ratio = min(targetSize.height / double(srcimg.rows), targetSize.width / double(srcimg.cols));
    resize(srcimg, resizeImg, Size(int(srcimg.cols * ratio), int(srcimg.rows * ratio)), INTER_LINEAR);
    resizeImg.copyTo(paddedImg(Rect(0, 0, int(srcimg.cols * ratio), int(srcimg.rows * ratio))));
    return pair<Mat, double>(paddedImg, ratio);
}

Mat unLetterBox(Mat bbox, double letterboxScale)
{
    return bbox / letterboxScale;
}

Mat visualize(Mat dets, Mat srcimg, double letterboxScale, double fps, bool vis=true)
{
    Mat resImg = srcimg.clone();

    if (fps > 0)
        putText(resImg, format("FPS: %.2f", fps), Size(10, 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);

    if (!vis)
    {
        cout << "Class name : score in rect( [topleft] , [bottomright])\n";
    }
    for (int row = 0; row < dets.rows; row++)
    {
        Mat boxF = unLetterBox(dets(Rect(0, row, 4, 1)), letterboxScale);
        Mat box;
        boxF.convertTo(box, CV_32S);
        float score = dets.at<float>(row, 4);
        int clsId = int(dets.at<float>(row, 5));

        int x0 = box.at<int>(0, 0);
        int y0 = box.at<int>(0, 1);
        int x1 = box.at<int>(0, 2);
        int y1 = box.at<int>(0, 3);

        string text = format("%s : %f", labelYolox[clsId].c_str(), score * 100);
        int font = FONT_HERSHEY_SIMPLEX;
        int baseLine = 0;
        Size txtSize = getTextSize(text, font, 0.4, 1, &baseLine);
        rectangle(resImg, Point(x0, y0), Point(x1, y1), Scalar(0, 255, 0), 2);
        rectangle(resImg, Point(x0, y0 + 1), Point(x0 + txtSize.width + 1, y0 + int(1.5 * txtSize.height)), Scalar(255, 255, 255), -1);
        putText(resImg, text, Point(x0, y0 + txtSize.height), font, 0.4, Scalar(0, 0, 0), 1);
        if (!vis)
        {
            cout << text;
            cout << " in rect (" << Point(x0, y0) << ", " << Point(x1, y1) << ")\n";
        }
    }

    return resImg;
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);

    parser.about("Use this script to run Yolox deep learning networks in opencv_zoo using OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string model = parser.get<String>("model");
    float confThreshold = parser.get<float>("confidence");
    float nmsThreshold = parser.get<float>("nms");
    bool vis = parser.get<bool>("vis");
    int backendTargetid = parser.get<int>("backend");

    if (model.empty())
    {
        CV_Error(Error::StsError, "Model file " + model + " not found");
    }

    Ptr<ObjectDetectorYX> detector = ObjectDetectorYX::create(model, confThreshold, nmsThreshold, backendTargetPairs[backendTargetid].first, backendTargetPairs[backendTargetid].second);
    //! [Open a video file or an image file or a camera stream]
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(samples::findFile(parser.get<String>("input")));
    else
        cap.open(0);
    if (!cap.isOpened())
        CV_Error(Error::StsError, "Cannot opend video or file");
    Mat frame, inputBlob;
    double letterboxScale;

    static const std::string kWinName = model;
    while (waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            cout << "Frame is empty" << endl;
            waitKey();
            break;
        }
        pair<Mat, double> w = letterBox(frame, detector->getInputSize());
        inputBlob = get<0>(w);
        letterboxScale = get<1>(w);
        TickMeter tm;
        tm.start();
        Mat predictions;
        detector->detect(inputBlob, predictions);
        tm.stop();
        cout << "Inference time: " << tm.getTimeMilli() << " ms\n";
        Mat img = visualize(predictions, frame, letterboxScale, tm.getFPS(), vis);
        if (vis)
        {
            imshow(kWinName, img);
        }
    }
    return 0;
}
