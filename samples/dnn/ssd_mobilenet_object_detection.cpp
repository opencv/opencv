#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::dnn;

const size_t inWidth = 300;
const size_t inHeight = 300;
const float inScaleFactor = 0.007843f;
const float meanVal = 127.5;
const char* classNames[] = {"background",
                            "aeroplane", "bicycle", "bird", "boat",
                            "bottle", "bus", "car", "cat", "chair",
                            "cow", "diningtable", "dog", "horse",
                            "motorbike", "person", "pottedplant",
                            "sheep", "sofa", "train", "tvmonitor"};

const String keys
    = "{ help           | false | print usage         }"
      "{ proto          | MobileNetSSD_deploy.prototxt   | model configuration }"
      "{ model          | MobileNetSSD_deploy.caffemodel | model weights }"
      "{ camera_device  | 0     | camera device number }"
      "{ camera_width   | 640   | camera device width  }"
      "{ camera_height  | 480   | camera device height }"
      "{ video          |       | video or image for detection}"
      "{ out            |       | path to output video file}"
      "{ min_confidence | 0.2   | min confidence      }"
      "{ opencl         | false | enable OpenCL }"
;

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("This sample uses MobileNet Single-Shot Detector "
                 "(https://arxiv.org/abs/1704.04861) "
                 "to detect objects on camera/video/image.\n"
                 ".caffemodel model's file is available here: "
                 "https://github.com/chuanqi305/MobileNet-SSD\n"
                 "Default network is 300x300 and 20-classes VOC.\n");

    if (parser.get<bool>("help"))
    {
        parser.printMessage();
        return 0;
    }

    String modelConfiguration = parser.get<String>("proto");
    String modelBinary = parser.get<String>("model");
    CV_Assert(!modelConfiguration.empty() && !modelBinary.empty());

    //! [Initialize network]
    dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);
    //! [Initialize network]

    if (parser.get<bool>("opencl"))
    {
        net.setPreferableTarget(DNN_TARGET_OPENCL);
    }

    if (net.empty())
    {
        cerr << "Can't load network by using the following files: " << endl;
        cerr << "prototxt:   " << modelConfiguration << endl;
        cerr << "caffemodel: " << modelBinary << endl;
        cerr << "Models can be downloaded here:" << endl;
        cerr << "https://github.com/chuanqi305/MobileNet-SSD" << endl;
        exit(-1);
    }

    VideoCapture cap;
    if (!parser.has("video"))
    {
        int cameraDevice = parser.get<int>("camera_device");
        cap = VideoCapture(cameraDevice);
        if(!cap.isOpened())
        {
            cout << "Couldn't find camera: " << cameraDevice << endl;
            return -1;
        }

        cap.set(CAP_PROP_FRAME_WIDTH, parser.get<int>("camera_width"));
        cap.set(CAP_PROP_FRAME_HEIGHT, parser.get<int>("camera_height"));
    }
    else
    {
        cap.open(parser.get<String>("video"));
        if(!cap.isOpened())
        {
            cout << "Couldn't open image or video: " << parser.get<String>("video") << endl;
            return -1;
        }
    }

    //Acquire input size
    Size inVideoSize((int) cap.get(CAP_PROP_FRAME_WIDTH),
                     (int) cap.get(CAP_PROP_FRAME_HEIGHT));

    double fps = cap.get(CAP_PROP_FPS);
    int fourcc = static_cast<int>(cap.get(CAP_PROP_FOURCC));
    VideoWriter outputVideo;
    outputVideo.open(parser.get<String>("out") ,
                     (fourcc != 0 ? fourcc : VideoWriter::fourcc('M','J','P','G')),
                     (fps != 0 ? fps : 10.0), inVideoSize, true);

    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera/video or read image

        if (frame.empty())
        {
            waitKey();
            break;
        }

        if (frame.channels() == 4)
            cvtColor(frame, frame, COLOR_BGRA2BGR);

        //! [Prepare blob]
        Mat inputBlob = blobFromImage(frame, inScaleFactor,
                                      Size(inWidth, inHeight),
                                      Scalar(meanVal, meanVal, meanVal),
                                      false, false); //Convert Mat to batch of images
        //! [Prepare blob]

        //! [Set input blob]
        net.setInput(inputBlob); //set the network input
        //! [Set input blob]

        //! [Make forward pass]
        Mat detection = net.forward(); //compute output
        //! [Make forward pass]

        vector<double> layersTimings;
        double freq = getTickFrequency() / 1000;
        double time = net.getPerfProfile(layersTimings) / freq;

        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        if (!outputVideo.isOpened())
        {
            putText(frame, format("FPS: %.2f ; time: %.2f ms", 1000.f/time, time),
                    Point(20,20), 0, 0.5, Scalar(0,0,255));
        }
        else
            cout << "Inference time, ms: " << time << endl;

        float confidenceThreshold = parser.get<float>("min_confidence");
        for(int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);

            if(confidence > confidenceThreshold)
            {
                size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

                int left = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int top = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int right = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int bottom = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));
                String label = format("%s: %.2f", classNames[objectClass], confidence);
                int baseLine = 0;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                top = max(top, labelSize.height);
                rectangle(frame, Point(left, top - labelSize.height),
                          Point(left + labelSize.width, top + baseLine),
                          Scalar(255, 255, 255), FILLED);
                putText(frame, label, Point(left, top),
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
