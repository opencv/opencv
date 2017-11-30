/*
Sample of using OpenCV dnn module in real time with device capture (based in yolo_object_detection.cpp)
Author: Alessandro de Oliveira Faria cabelo@opensuse.org
http://assuntonerd.com.br
 VIDEO DEMO:
 https://www.youtube.com/watch?v=NHtRlndE2cg

 COMPILE:
 g++ `pkg-config --cflags opencv` `pkg-config --libs opencv` yolo_object_detection.cpp -o yolo_object_detection

 RUN in webcam:
 yolo_object_detection -source=0  -cfg=[PATH-TO-DARKNET]/cfg/yolo.cfg -model=[PATH-TO-DARKNET]/yolo.weights   -labels=[PATH-TO-DARKNET]/data/coco.names

 RUN with image:
 yolo_object_detection -source=../data/objects_dnn_example.png  -cfg=[PATH-TO-DARKNET]/cfg/yolo.cfg -model=[PATH-TO-DARKNET]/yolo.weights   -labels=[PATH-TO-DARKNET]/data/coco.names

 RUN in video:
 yolo_object_detection -source=[PATH-TO-VIDEO] -cfg=[PATH-TO-DARKNET]/cfg/yolo.cfg -model=[PATH-TO-DARKNET]/yolo.weights   -labels=[PATH-TO-DARKNET]/data/coco.names

*/
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <list>

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "opencv2/opencv_modules.hpp"

using namespace cv;
using namespace cv::dnn;
using namespace std;

const int network_width = 416;
const int network_height = 416;

const char* about = "This sample uses You only look once (YOLO)-Detector "
                    "(https://arxiv.org/abs/1612.08242)"
                    "to detect objects on capture device, video or image file\n";

const char* params = "{ help           | false | print usage          }"
                     "{ source         |       | device, video or img }"
                     "{ cfg            |       | model configuration  }"
                     "{ model          |       | model weights        }"
                     "{ labels         |       | label of the object  }"
                     "{ min_confidence | 0.24  | min confidence       }";



static void assetRoi(Rect &_roi, Mat &_frame  )
{
    if(_roi.x <= 0) _roi.x = 1;
    if(_roi.y <= 0) _roi.y = 1;
    if((_roi.width+_roi.x) >= (_frame.cols-1)) _roi.width = _roi.width-((_roi.width+_roi.x)-_frame.cols);
    if((_roi.height+_roi.y) >= (_frame.rows-1)) _roi.height = _roi.height-((_roi.height+_roi.y)-_frame.rows);
}

static void addLabels(cv::String filename,std::list<std::string> &_mylist)
{
    std::string str;
    std::ifstream file(filename.c_str());
    while (std::getline(file, str))
    {
        _mylist.push_back(str);
    }
}

static std::string returnLabel(size_t index, std::list<std::string> &_mylist)
{
    list<std::string>::iterator it = _mylist.begin();
    std::advance(it, (int)index);
    return (*it);

}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, params);
    if (parser.get<bool>("help"))
    {
        std::cout << about << std::endl;
        parser.printMessage();
        return 0;
    }

    String modelConfiguration = parser.get<string>("cfg");
    String modelBinary = parser.get<string>("model");
    String labels = parser.get<string>("labels");
    String src = parser.get<string>("source");
    std::list<std::string> mylist;
    addLabels(labels,mylist);

    //! [Initialize network]
    dnn::Net net = readNetFromDarknet(modelConfiguration, modelBinary);
    //! [Initialize network]

    if (net.empty())
    {
        cerr << "Can't load network by using the following files: " << endl;
        cerr << "cfg-file:     " << modelConfiguration << endl;
        cerr << "weights-file: " << modelBinary << endl;
        cerr << "Models can be downloaded here:" << endl;
        cerr << "https://pjreddie.com/darknet/yolo/" << endl;
        exit(-1);
    }

    bool grabFrame = true;
    VideoCapture cap;
    Mat frame;

    if( src.empty() || (isdigit(src[0]) && src.size() == 1) )
    {
        int camera = (src.empty() ? 0 : atoi(src.c_str()));
        if(!cap.open(camera))
        {
            cout << "Capture from camera #" <<  camera << " didn't work" << endl;
            return 0;
        }
    }
    else if( src.size() )
    {
        frame = imread( src, 1 );
        if( frame.empty() )
        {
            if(!cap.open( src ))
            {
                cout << "Could not read " << src << endl;
                return 0;
            }
        }
        else
        {
            grabFrame = false;
        }
    }

    while (true)
    {
        if(grabFrame)
        cap >> frame;

        //! [Resizing without keeping aspect ratio]
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(network_width, network_height));
        //! [Resizing without keeping aspect ratio]

        //! [Prepare blob]
        Mat inputBlob = blobFromImage(resized, 1 / 255.F); //Convert Mat to batch of images
        //! [Prepare blob]

        //! [Set input blob]
        net.setInput(inputBlob, "data");                //set the network input
        //! [Set input blob]

        //! [Make forward pass]
        cv::Mat detectionMat = net.forward("detection_out");	//compute output
        //! [Make forward pass]

        float confidenceThreshold = parser.get<float>("min_confidence");

        for (int i = 0; i < detectionMat.rows; i++)
        {

            const int probability_index = 5;
            const int probability_size = detectionMat.cols - probability_index;
            float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);

            size_t objectClass = std::max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
            float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

            if (confidence > confidenceThreshold)
            {
                    string sLabel = returnLabel(objectClass,mylist );
                    float x = detectionMat.at<float>(i, 0);
                    float y = detectionMat.at<float>(i, 1);
                    float width = detectionMat.at<float>(i, 2);
                    float height = detectionMat.at<float>(i, 3);
                    float xLeftBottom = (x - width / 2) * frame.cols;
                    float yLeftBottom = (y - height / 2) * frame.rows;
                    float xRightTop = (x + width / 2) * frame.cols;
                    float yRightTop = (y + height / 2) * frame.rows;

                    std::cout << "Class: " << objectClass << " " <<sLabel<<std::endl;
                    std::cout << "Confidence: " << confidence << std::endl;
                    std::cout << " " << xLeftBottom << " " << yLeftBottom << " " << xRightTop << " " << yRightTop << std::endl;

                    Rect object((int)xLeftBottom, (int)yLeftBottom, (int)(xRightTop - xLeftBottom), (int)(yRightTop - yLeftBottom));
                    assetRoi(object,frame);

                    rectangle(frame, object, Scalar(0, 255, 0),2);
                    cv::Mat roi = frame(object);
                    cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(0, 0, 0));
                    double alpha = 0.3;
                    cv::addWeighted(color, alpha, roi, 1.0 - alpha , 0.0, roi);

                    putText(frame, sLabel, Point(object.x, object.y+12),FONT_HERSHEY_PLAIN, 1 ,Scalar::all(255));
            }
        }

        imshow("YOLO",frame);
        if(cv::waitKey((grabFrame?30:0)) >= 0)
        break;
    }
    cap.release();
    return 0;
}
