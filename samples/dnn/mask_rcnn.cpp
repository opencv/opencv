// Deep learning based Object Detection and Instance Segmentation using Mask R-CNN 
// VIDEO 360 DEMO: https://www.youtube.com/watch?v=0tU8991QgE8
// Downloads:
// http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
// https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
// More Information : Alessandro de Oliveira Faria (A.K.A. CABELO)- cabelo@opensuse.org

#include <fstream>
#include <sstream>
#include <iostream>
#include <string.h>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "common.hpp"

std::string keys =
    "{ help  h     | | Print help message. \nUsage \n\t\t./mask_rcnn --image=logo.jpg \n\t\t ./mask_rcnn --media=teste.mp4}"
    "{ image m     |<none>| Path to input image file.  }"
    "{ video v     |<none>| Path to input video file.  }"
    "{ weights w | ./mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb | The pre-trained weights.  }"
    "{ textgraph g |./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt | iThe text graph file that has been tuned by the OpenCV’s DNN support group  }"
    "{ cthr         | .5 | Confidence threshold. }"
    "{ mthr        | .4 | Mask threshold. }";


using namespace cv;
using namespace dnn;
using namespace std;

float confThreshold;
float maskThreshold;

vector<string> classes;
vector<Scalar> colors;

void drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask);

void postprocess(Mat& frame, const vector<Mat>& outs);

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run object detection using Segmentation using Mask R-CNN in OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    confThreshold = parser.get<float>("cthr");
    maskThreshold = parser.get<float>("mthr");
    String textGraph = parser.get<string>("textgraph");
    String modelWeights = parser.get<string>("weights");

    string classesFile = "mscoco_labels.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    string colorsFile = "colors.txt";
    ifstream colorFptr(colorsFile.c_str());
    while (getline(colorFptr, line)) {
        char* pEnd;
        double r, g, b;
        r = strtod (line.c_str(), &pEnd);
        g = strtod (pEnd, NULL);
        b = strtod (pEnd, NULL);
        colors.push_back(Scalar(r, g, b, 255.0));
    }

    Net net = readNetFromTensorflow(modelWeights, textGraph);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    string str, outputFile;
    VideoCapture cap;
    VideoWriter video;
    Mat frame, blob;

    try {

        if (parser.has("image"))
        {
            str = parser.get<String>("image");
            cout << "Image file input : " << str << endl;
            ifstream ifile(str);
            if (!ifile) throw("error");
            cap.open(str);
        }
        else if (parser.has("video"))
        {
            str = parser.get<String>("video");
            ifstream ifile(str);
            if (!ifile) throw("error");
            cap.open(str);
        }
        else cap.open(parser.get<int>("device"));

    }
    catch(...) {
        cout << "Could not open the input image/video stream" << endl;
        return 0;
    }


    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    while (waitKey(1) < 0)
    {
        cap >> frame;

        if (frame.empty()) {
            cout << "Done processing !!!" << endl;
            waitKey(0);
            break;
        }
        
        blobFromImage(frame, blob, 1.0, Size(frame.cols, frame.rows), Scalar(), true, false);

        net.setInput(blob);

        std::vector<String> outNames(2);
        outNames[0] = "detection_out_final";
        outNames[1] = "detection_masks";
        vector<Mat> outs;
        net.forward(outs, outNames);

        postprocess(frame, outs);

        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("openCV Mask R-CNN : %0.0f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

        Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        
	imshow(kWinName, frame);

    }

    cap.release();
    if (!parser.has("image")) video.release();

    return 0;
}

void postprocess(Mat& frame, const vector<Mat>& outs)
{
    Mat outDetections = outs[0];
    Mat outMasks = outs[1];

    // Output size of masks is NxCxHxW where
    // N - number of detected boxes
    // C - number of classes (excluding background)
    // HxW - segmentation shape
    const int numDetections = outDetections.size[2];

    outDetections = outDetections.reshape(1, outDetections.total() / 7);
    for (int i = 0; i < numDetections; ++i)
    {
        float score = outDetections.at<float>(i, 2);
        if (score > confThreshold)
        {
            int classId = static_cast<int>(outDetections.at<float>(i, 1));
            int left = static_cast<int>(frame.cols * outDetections.at<float>(i, 3));
            int top = static_cast<int>(frame.rows * outDetections.at<float>(i, 4));
            int right = static_cast<int>(frame.cols * outDetections.at<float>(i, 5));
            int bottom = static_cast<int>(frame.rows * outDetections.at<float>(i, 6));

            left = max(0, min(left, frame.cols - 1));
            top = max(0, min(top, frame.rows - 1));
            right = max(0, min(right, frame.cols - 1));
            bottom = max(0, min(bottom, frame.rows - 1));
            Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

            Mat objectMask(outMasks.size[2], outMasks.size[3],CV_32F, outMasks.ptr<float>(i,classId));

            drawBox(frame, classId, score, box, objectMask);
        }
    }
}

void drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask)
{
    rectangle(frame, Point(box.x, box.y), Point(box.x+box.width, box.y+box.height), Scalar(255, 178, 50), 3);

    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    box.y = max(box.y, labelSize.height);
    rectangle(frame, Point(box.x, box.y - round(1.5*labelSize.height)), Point(box.x + round(1.5*labelSize.width), box.y + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);

    Scalar color = colors[classId%colors.size()];

    resize(objectMask, objectMask, Size(box.width, box.height));
    Mat mask = (objectMask > maskThreshold);
    Mat coloredRoi = (0.3 * color + 0.7 * frame(box));
    coloredRoi.convertTo(coloredRoi, CV_8UC3);

    vector<Mat> contours;
    Mat hierarchy;
    mask.convertTo(mask, CV_8U);
    findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    drawContours(coloredRoi, contours, -1, color, 5, LINE_8, hierarchy, 100);
    coloredRoi.copyTo(frame(box), mask);

}
