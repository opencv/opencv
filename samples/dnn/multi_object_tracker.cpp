// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include <iostream>
#include <map>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#include "common.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;

const string about =
    "Follow several people at once with cv::MultiTracker.\n\n"
    "A detector finds people in each frame and the tracker decides which box belongs to which\n"
    "person, so every box keeps its number for as long as that person is on screen.\n\n"
    "Usage:\n"
    "  example_dnn_multi_object_tracker multi_object_tracker\n"
    "  example_dnn_multi_object_tracker multi_object_tracker --input=vtest.avi\n"
    "  example_dnn_multi_object_tracker multi_object_tracker --input=vtest.avi --reid\n\n"
    "--reid also loads a re-identification network, which helps ids survive when people cross.\n";

const string keys =
    "{ help  h        |                   | show this message }"
    "{ @alias         |                   | model alias from models.yml }"
    "{ zoo            | models.yml        | model description file }"
    "{ input i        | vtest.avi         | video file or camera index }"
    "{ reid           |                   | also use appearance to keep ids apart }"
    "{ conf           | 0.25              | detector confidence threshold }"
    "{ nms            | 0.45              | detector NMS threshold }"
    "{ device         | 0                 | camera index when --input is not a file }"
    "{ backend        | 0                 | 0 automatic, 1 Halide, 2 OpenVINO, 3 OpenCV, 4 VKCOM, 5 CUDA }"
    "{ target         | 0                 | 0 CPU, 1 OpenCL, 2 OpenCL FP16, 3 Myriad, 4 Vulkan, 6 CUDA, 7 CUDA FP16 }";

// One colour per id, so the same person keeps the same colour for as long as they are tracked.
static Scalar colorForId(int id)
{
    RNG rng((uint64)id * 0x9E3779B97F4A7C15ULL + 1);
    return Scalar(rng.uniform(64, 256), rng.uniform(64, 256), rng.uniform(64, 256));
}

// yolov8 emits one [1, 84, 8400] tensor: four box numbers then a score per class. The image is
// padded to a square first, so the boxes scale back by the padded length.
static void detect(const Mat& frame, Net& net, int inputSize, double inputScale, bool swapRB,
                   float confThreshold, float nmsThreshold,
                   vector<Rect2d>& boxes, vector<float>& scores, vector<int>& classIds)
{
    boxes.clear();
    scores.clear();
    classIds.clear();

    const int length = max(frame.rows, frame.cols);
    Mat square = Mat::zeros(Size(length, length), frame.type());
    frame.copyTo(square(Rect(0, 0, frame.cols, frame.rows)));
    const double backScale = (double)length / inputSize;

    Mat blob;
    blobFromImage(square, blob, inputScale, Size(inputSize, inputSize), Scalar(), swapRB, false, CV_32F);
    net.setInput(blob);

    vector<Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    Mat out = outs[0].reshape(0, outs[0].size[1]);   // [84, 8400]
    Mat cand;
    transpose(out, cand);                            // [8400, 84]

    vector<Rect2d> raw;
    vector<float> rawScores;
    vector<int> rawClasses;
    for (int i = 0; i < cand.rows; i++)
    {
        double maxScore = 0.0, minScore = 0.0;
        Point maxLoc, minLoc;
        minMaxLoc(cand.row(i).colRange(4, cand.cols), &minScore, &maxScore, &minLoc, &maxLoc);
        if (maxScore < confThreshold || maxLoc.x != 0)
            continue;                                // class 0 is "person" in COCO

        const double cx = cand.at<float>(i, 0);
        const double cy = cand.at<float>(i, 1);
        const double w = cand.at<float>(i, 2);
        const double h = cand.at<float>(i, 3);
        raw.push_back(Rect2d(cx - 0.5 * w, cy - 0.5 * h, w, h));
        rawScores.push_back((float)maxScore);
        rawClasses.push_back(maxLoc.x);
    }

    vector<int> keep;
    NMSBoxes(raw, rawScores, confThreshold, nmsThreshold, keep);
    for (size_t i = 0; i < keep.size(); i++)
    {
        const Rect2d& b = raw[keep[i]];
        boxes.push_back(Rect2d(b.x * backScale, b.y * backScale,
                               b.width * backScale, b.height * backScale));
        scores.push_back(rawScores[keep[i]]);
        classIds.push_back(rawClasses[keep[i]]);
    }
}

// One L2-normalised descriptor per detection. Model is used over blobFromImage because
// setInputScale takes a Scalar, so the per-channel std folds straight in.
static Mat describe(const Mat& frame, const Ptr<Model>& reid, const vector<Rect2d>& boxes)
{
    if (boxes.empty())
        return Mat();

    Mat features;
    for (size_t i = 0; i < boxes.size(); i++)
    {
        Rect roi = Rect((int)boxes[i].x, (int)boxes[i].y,
                        (int)boxes[i].width, (int)boxes[i].height) & Rect(0, 0, frame.cols, frame.rows);
        if (roi.width < 2 || roi.height < 2)
            roi = Rect(0, 0, min(2, frame.cols), min(2, frame.rows));

        vector<Mat> outs;
        reid->predict(frame(roi), outs);
        Mat f = outs[0].reshape(1, 1);
        f.convertTo(f, CV_32F);
        const double n = norm(f);
        if (n > 0)
            f /= n;
        features.push_back(f);
    }
    return features;
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    if (!parser.has("@alias") || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    const string modelName = parser.get<String>("@alias");
    const string zooFile = findFile(parser.get<String>("zoo"));
    const string allKeys = keys + genPreprocArguments(modelName, zooFile)
                                + genPreprocArguments(modelName, zooFile, "yolo_");
    parser = CommandLineParser(argc, argv, allKeys);
    parser.about(about);

    const bool useReid = parser.has("reid");
    const int backend = parser.get<int>("backend");
    const int target = parser.get<int>("target");

    // Detector
    const string yoloPath = findModel(parser.get<String>("yolo_model"), parser.get<String>("yolo_sha1"));
    Net yolo = readNetFromONNX(yoloPath);
    yolo.setPreferableBackend((dnn::Backend)backend);
    yolo.setPreferableTarget((dnn::Target)target);
    const int yoloSize = parser.get<int>("yolo_width");
    const double yoloScale = parser.get<double>("yolo_scale");
    const bool yoloSwapRB = parser.get<bool>("yolo_rgb");

    // Appearance, only when asked for
    Ptr<Model> reid;
    if (useReid)
    {
        const string reidPath = findModel(parser.get<String>("model"), parser.get<String>("sha1"));
        reid = makePtr<Model>(readNetFromONNX(reidPath));
        const Scalar mean = parser.get<Scalar>("mean");
        const Scalar std = parser.get<Scalar>("std");
        const double scale = parser.get<double>("scale");
        // (pixel - mean) * scale, per channel, with the std folded into the scale. swapRB swaps
        // channel 0 and 2 of both, so each is given in the model's own RGB order.
        reid->setInputMean(mean);
        reid->setInputScale(Scalar(scale / std[0], scale / std[1], scale / std[2]));
        reid->setInputSize(parser.get<int>("width"), parser.get<int>("height"));
        reid->setInputSwapRB(parser.get<bool>("rgb"));
        reid->setPreferableBackend((dnn::Backend)backend);
        reid->setPreferableTarget((dnn::Target)target);
    }

    MultiTracker::Params params;
    if (useReid)
        params.embeddingWeight = 0.4f;   // appearance helps, but motion still leads
    Ptr<MultiTracker> tracker = MultiTracker::create(params);

    const string inputName = parser.get<String>("input");
    VideoCapture cap;
    if (inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1))
        cap.open(inputName.empty() ? parser.get<int>("device") : inputName[0] - '0');
    else
        cap.open(findFile(inputName));
    if (!cap.isOpened())
    {
        cerr << "Could not open the input " << inputName << endl;
        return 1;
    }

    const string window = "MULTI OBJECT TRACKING";
    namedWindow(window, WINDOW_NORMAL);
    FontFace fontFace("sans");
    const int stdSize = 15, stdWeight = 400, stdImgSize = 512;

    Mat frame;
    for (;;)
    {
        cap >> frame;
        if (frame.empty())
            break;

        const int fontSize = max(10, stdSize * frame.cols / stdImgSize);
        const int fontWeight = max(200, stdWeight * frame.cols / stdImgSize);

        vector<Rect2d> detBoxes;
        vector<float> detScores;
        vector<int> detClasses;
        detect(frame, yolo, yoloSize, yoloScale, yoloSwapRB,
               parser.get<float>("conf"), parser.get<float>("nms"),
               detBoxes, detScores, detClasses);

        vector<int> ids;
        vector<Rect2d> boxes;
        vector<int> classes;
        if (useReid)
            tracker->update(detBoxes, detScores, detClasses, describe(frame, reid, detBoxes),
                            ids, boxes, classes);
        else
            tracker->update(detBoxes, detScores, detClasses, ids, boxes, classes);

        for (size_t i = 0; i < ids.size(); i++)
        {
            const Scalar color = colorForId(ids[i]);
            const Rect box((int)boxes[i].x, (int)boxes[i].y, (int)boxes[i].width, (int)boxes[i].height);
            rectangle(frame, box, color, 2);

            const string label = format("%d", ids[i]);
            Rect tr = getTextSize(Size(), label, Point(), fontFace, fontSize, fontWeight);
            tr.height += fontSize / 2;
            tr.width += fontSize / 2;
            tr.x = box.x;
            tr.y = max(0, box.y - tr.height);
            rectangle(frame, tr, color, FILLED);
            putText(frame, label, Point(tr.x + fontSize / 4, tr.y + fontSize),
                    Scalar::all(0), fontFace, fontSize, fontWeight);
        }

        const string status = format("tracking %d  (detections %d)%s",
                                     (int)ids.size(), (int)detBoxes.size(),
                                     useReid ? "  [re-id on]" : "");
        putText(frame, status, Point(10, fontSize + 5), Scalar(0, 255, 0),
                fontFace, fontSize, fontWeight);

        imshow(window, frame);
        if (waitKey(30) >= 0)
            break;
    }
    return 0;
}
