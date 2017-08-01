#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

const size_t inWidth = 300;
const size_t inHeight = 300;
const float WHRatio = inWidth / (float)inHeight;
const float inScaleFactor = 0.007843f;
const float meanVal = 127.5;
const char* classNames[] = {"background",
                            "aeroplane", "bicycle", "bird", "boat",
                            "bottle", "bus", "car", "cat", "chair",
                            "cow", "diningtable", "dog", "horse",
                            "motorbike", "person", "pottedplant",
                            "sheep", "sofa", "train", "tvmonitor"};

const char* about = "This sample uses Single-Shot Detector "
                    "(https://arxiv.org/abs/1512.02325)"
                    "to detect objects on image.\n"
                    ".caffemodel model's file is avaliable here: "
                    "https://github.com/chuanqi305/MobileNet-SSD/blob/master/MobileNetSSD_train.caffemodel\n";

const char* params
    = "{ help           | false | print usage         }"
      "{ proto          | MobileNetSSD_300x300.prototxt | model configuration }"
      "{ model          |       | model weights       }"
      "{ video          |       | video for detection }"
      "{ out            |       | path to output video file}"
      "{ min_confidence | 0.2   | min confidence      }";

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, params);

    if (parser.get<bool>("help"))
    {
        cout << about << endl;
        parser.printMessage();
        return 0;
    }

    String modelConfiguration = parser.get<string>("proto");
    String modelBinary = parser.get<string>("model");

    //! [Initialize network]
    dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);
    //! [Initialize network]

    VideoCapture cap(parser.get<String>("video"));
    if(!cap.isOpened()) // check if we succeeded
    {
        cap = VideoCapture(0);
        if(!cap.isOpened())
        {
            cout << "Couldn't find camera" << endl;
            return -1;
        }
    }

    Size inVideoSize = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),    //Acquire input size
                            (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));

    Size cropSize;
    if (inVideoSize.width / (float)inVideoSize.height > WHRatio)
    {
        cropSize = Size(static_cast<int>(inVideoSize.height * WHRatio),
                        inVideoSize.height);
    }
    else
    {
        cropSize = Size(inVideoSize.width,
                        static_cast<int>(inVideoSize.width / WHRatio));
    }

    Rect crop(Point((inVideoSize.width - cropSize.width) / 2,
                    (inVideoSize.height - cropSize.height) / 2),
              cropSize);

    VideoWriter outputVideo;
    outputVideo.open(parser.get<String>("out") ,
                     static_cast<int>(cap.get(CV_CAP_PROP_FOURCC)),
                     cap.get(CV_CAP_PROP_FPS), cropSize, true);

    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera
        //! [Prepare blob]

        Mat inputBlob = blobFromImage(frame, inScaleFactor,
                                      Size(inWidth, inHeight), meanVal); //Convert Mat to batch of images
        //! [Prepare blob]

        //! [Set input blob]
        net.setInput(inputBlob, "data");                //set the network input
        //! [Set input blob]

        TickMeter tm;
        tm.start();
        //! [Make forward pass]
        Mat detection = net.forward("detection_out");                                  //compute output
        tm.stop();
        cout << "Inference time, ms: " << tm.getTimeMilli() << endl;
        //! [Make forward pass]

        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        frame = frame(crop);

        float confidenceThreshold = parser.get<float>("min_confidence");
        for(int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);

            if(confidence > confidenceThreshold)
            {
                size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                ostringstream ss;
                ss << confidence;
                String conf(ss.str());

                Rect object((int)xLeftBottom, (int)yLeftBottom,
                            (int)(xRightTop - xLeftBottom),
                            (int)(yRightTop - yLeftBottom));

                rectangle(frame, object, Scalar(0, 255, 0));
                String label = String(classNames[objectClass]) + ": " + conf;
                int baseLine = 0;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
                                      Size(labelSize.width, labelSize.height + baseLine)),
                          Scalar(255, 255, 255), CV_FILLED);
                putText(frame, label, Point(xLeftBottom, yLeftBottom),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
            }
        }

        if (outputVideo.isOpened())
            outputVideo << frame;

        imshow("detections", frame);
        if (waitKey(1) >= 0) break;
    }

    return 0;
} // main
