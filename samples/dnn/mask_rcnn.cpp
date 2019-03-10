// Deep learning based Object Detection and Instance Segmentation using Mask R-CNN
// Downloads:
// http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
// https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
// https://raw.githubusercontent.com/spmallick/learnopencv/master/Mask-RCNN/mscoco_labels.names
// It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html
//

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

string keys =
    "{ help        | | Print help message. }"
    "{ image       |<none>| Path to input image file.  }"
    "{ input i     | 0 | Path to input image file or video or camera id.  }"
    "{ classes     |mscoco_labels.names| Path to a text file with names of classes. }"
    "{ model  m    |mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb | The pre-trained weights.  }"
    "{ height h    |800| Preprocess input image by resizing to a specific height }"
    "{ width w     |800| Preprocess input image by resizing to a specific columns }"
    "{ config c    |mask_rcnn_inception_v2_coco_2018_01_28.pbtxt |The text graph file that has been tuned by the OpenCV’s DNN support group  }"
    "{ cthr        | .5 | Confidence threshold. }"
    "{ mthr        | .4 | Mask threshold. }";

void static drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask, float maskThreshold, vector<Scalar>& colors, vector<string>& classes);

void static postprocess(Mat& frame, const vector<Mat>& outs, float confThreshold, float maskThreshold, vector<Scalar>& colors, vector<string>& classes);

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("This sample demonstrates instance segmentation network called Mask-RCNN.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    float confThreshold;
    float maskThreshold;
    vector<string> classes;
    vector<Scalar> colors;
    int height;
    int width;

    confThreshold = parser.get<float>("cthr");
    maskThreshold = parser.get<float>("mthr");

    string input = parser.get<string>("input");
    string textGraph = parser.get<string>("config");
    string modelWeights = parser.get<string>("model");
    string classesFile = parser.get<string>("classes");

    height = parser.get<int>("height");
    width = parser.get<int>("width");


    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    for(int _c =0; _c<=10; _c++)
    {
        colors.push_back( Scalar(rand() % 255,rand() % 255,rand() % 255 ,255.0));
    }

    Net net = readNet(modelWeights, textGraph);

    string str;
    VideoCapture cap;
    Mat frame, blob;

    try {
        if (parser.has("image"))
        {
            str = parser.get<String>("image");
            frame = imread(str);
        }
        else if (input.find_first_not_of("0123456789") == std::string::npos)
            cap.open(atoi(input.c_str()));
        else
            cap.open(input);
    }
    catch(...) {
        cout << "Could not open the input image/video stream" << endl;
        return 0;
    }

    while (waitKey(1) < 0)
    {
        if (!parser.has("image"))
            cap >> frame;

        if (frame.empty()) {
            cout << "Done processing !!!" << endl;
            waitKey(0);
            break;
        }

        blobFromImage(frame, blob, 1.0, Size(width, height), Scalar(), true, false);

        net.setInput(blob);

        std::vector<String> outNames(2);
        outNames[0] = "detection_out_final";
        outNames[1] = "detection_masks";
        vector<Mat> outs;
        net.forward(outs, outNames);

        postprocess(frame, outs,maskThreshold,confThreshold,colors,classes);

        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("OpenCV Mask R-CNN : %0.0f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

        imshow("Mask R-CNN sample", frame);
        if (parser.has("image"))
            frame.release();
    }

    cap.release();

    return 0;
}

void postprocess(Mat& frame, const vector<Mat>& outs, float confThreshold, float maskThreshold, vector<Scalar>& colors, vector<string>& classes )
{
    Mat outDetections = outs[0];
    Mat outMasks = outs[1];

    // Output size of masks is NxCxHxW where
    // N - number of detected boxes
    // C - number of classes (excluding background)
    // HxW - segmentation shape
    const int numDetections = outDetections.size[2];

    outDetections = outDetections.reshape(1,(int)(outDetections.total() / 7));
    for ( int i = 0; i < numDetections; ++i)
    {
        float score = outDetections.at<float>(i, 2);
        if (score > confThreshold)
        {
            int classId = static_cast<int>(outDetections.at<float>(i, 1));
            int left = static_cast<int>(frame.cols * outDetections.at<float>(i, 3));
            int top = static_cast<int>(frame.rows * outDetections.at<float>(i, 4));
            int right = static_cast<int>(frame.cols * outDetections.at<float>(i, 5));
            int bottom = static_cast<int>(frame.rows * outDetections.at<float>(i, 6));

            Rect box(Point(left, top), Point(right, bottom));
            Rect imageBox(0, 0, frame.cols - 1, frame.rows - 1);
            box &= imageBox;

            Mat objectMask(outMasks.size[2], outMasks.size[3],CV_32FC1, outMasks.ptr<float>(i,classId));

            drawBox(frame, classId, score, box, objectMask, maskThreshold, colors, classes);
        }
    }
}

void drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask, float maskThreshold, vector<Scalar>& colors, vector<string>& classes)
{
    rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(255, 178, 50), 3);
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    box.y = max(box.y, labelSize.height);
    rectangle(frame, Point(box.x, box.y - static_cast<int>(round(1.5*labelSize.height))), Point(box.x + static_cast<int>(round(1.5*labelSize.width)), box.y + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);

    Scalar color = colors[classId%colors.size()];

    resize(objectMask, objectMask, Size(box.width, box.height));
    Mat mask = (objectMask > maskThreshold);
    Mat roi = frame(box);
    roi = roi * 0.7 + color * 0.3;
    roi.convertTo(roi, CV_8UC3);

    Mat hierarchy;
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    polylines(roi, contours, true, color, 2, LINE_8);
}
