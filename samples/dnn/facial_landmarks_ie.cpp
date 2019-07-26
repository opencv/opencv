// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
//  This sample demonstrates the use of pretrained face and facial landmarks detection OpenVINO networks with opencv's dnn module.
//
//  The sample uses two pretrained OpenVINO networks so the OpenVINO package has to be preinstalled.
//  Please install topologies described below using downloader.py (.../openvino/deployment_tools/tools/model_downloader) to run this sample.
//  Face detection model - face-detection-adas-0001: https://github.com/opencv/open_model_zoo/tree/master/intel_models/face-detection-adas-0001
//  Facial landmarks detection model - facial-landmarks-35-adas-0002: https://github.com/opencv/open_model_zoo/tree/master/intel_models/facial-landmarks-35-adas-0002
//

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <iostream>

int main(int argc, char** argv)
{
    const std::string win1 = "FaceLandmarkDetector";
    const Scalar green(0, 255, 0);
    const Scalar yellow(0, 255, 255);
    const Scalar black(0,0,0);

    CommandLineParser parser(argc, argv,
     "{ help  h              | |     Print the help message. }"
     "{ facestruct           | |     Full path to a Face detection model structure file (for example, .xml file).}"
     "{ faceweights          | |     Full path to a Face detection model weights file (for example, .bin file).}"
     "{ landmstruct          | |     Full path to a facial Landmarks detection model structure file (for example, .xml file).}"
     "{ landmweights         | |     Full path to a facial Landmarks detection model weights file (for example, .bin file).}"
     "{ input                | |     Full path to an input image or a video file. Skip this argument to capture frames from a camera.}"
     "{ facetarget           | 0 |   Choose one of target computation devices for facedetector: "
                                         "0: CPU target (by default), "
                                         "1: OpenCL, "
                                         "2: OpenCL fp16 (half-float precision), "
                                         "3: VPU }"
     "{ landmtarget          | 0 |   Choose one of target computation devices for landmarkdetector: "
                                         "0: CPU target (by default), "
                                         "1: OpenCL, "
                                         "2: OpenCL fp16 (half-float precision), "
                                         "3: VPU }"
        );

    parser.about("Use this script to run the face and facial landmarks detection deep learning IE networks using dnn API.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    //Parsing input arguments
    const std::string facexmlPath = parser.get<std::string>("facestruct");
    const std::string facebinPath = parser.get<std::string>("faceweights");
    const Target faceTarget = parser.get<Target>("facetarget");

    const std::string landmxmlPath = parser.get<std::string>("landmstruct");
    const std::string landmbinPath = parser.get<std::string>("landmweights");
    const Target landmTarget = parser.get<Target>("landmtarget");

    //Models' definition & initialization
    Net faceNet = readNet(facexmlPath, facebinPath);
    faceNet.setPreferableTarget(faceTarget);
    const unsigned int faceObjectSize = 7;
    const float faceConfThreshold = 0.7f;
    const unsigned int facecols = 672;
    const unsigned int facerows = 384;

    Net landmNet = readNet(landmxmlPath, landmbinPath);
    landmNet.setPreferableTarget(landmTarget);
    const unsigned int landmcols = 60;
    const unsigned int landmrows = 60;
    const float bb_enlarge_coefficient = 1.0f;

    //Input
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else if (!cap.open(0))
            {
                std::cout<<"No input available"<<std::endl;
                return 1;
            }

    Mat img;
    while (waitKey(1) < 0)
    {
        cap >> img;
        if (img.empty())
        {
           waitKey();
           break;
        }

        Mat shwimg;         //Image to show
        img.copyTo(shwimg);

        //Preprocessing
        Mat inputImg;
        resize(img, inputImg, Size(facecols, facerows));

        //Infering Facedetector
        faceNet.setInput(blobFromImage(inputImg));
        Mat outFaces = faceNet.forward();

        float* faceData = (float*)(outFaces.data);
        for (size_t i = 0; i < outFaces.total(); i += faceObjectSize)
        {
            //Face rectangle prediction processing
            float confidence = faceData[i + 2];
            if (confidence > faceConfThreshold)
            {
                int left = int(faceData[i + 3] * img.cols);
                left = std::max(left, 0);
                int top = int(faceData[i + 4] * img.rows);
                top = std::max(top, 0);
                int right  = int(faceData[i + 5] * img.cols);
                right = std::min(right, img.cols-2);
                int bottom = int(faceData[i + 6] * img.rows);
                bottom = std::min(bottom, img.rows-2);
                int width  = right - left + 1;
                int height = bottom - top + 1;

                rectangle(shwimg, Rect(left, top, width, height), green, 1);

                //Postprocessing for landmarks
                int max_of_sizes = std::max(width, height);
                int add_width = int(max_of_sizes*bb_enlarge_coefficient) - width;
                int add_height = int(max_of_sizes*bb_enlarge_coefficient) - height;

                Mat cropped;
                cv::copyMakeBorder(img(Rect(left, top, width, height)), cropped, add_height/2, (add_height+1)/2,
                                   add_width/2, (add_width+1)/2, BORDER_CONSTANT | BORDER_ISOLATED , black);

                Mat resized;
                resize(cropped, resized, Size(landmcols, landmrows));

                //Infering Landmarkdetector
                landmNet.setInput(blobFromImage(resized));
                Mat outLandms = landmNet.forward();

                float* landmData = (float*)(outLandms.data);
                size_t j = 0;
                //Landmarks processing
                for (; j < 18*2; j+=2)
                {
                    Point p (int(landmData[ j ] * cropped.cols + left-add_width/2),
                             int(landmData[j+1] * cropped.rows + top-add_height/2));
                    line(shwimg, p, p, yellow, 3);
                }

                Point ptsmas[17];
                for(; j < outLandms.total(); j+=2)
                {
                    ptsmas[(j-18*2)/2] = Point(int(landmData[j] * cropped.cols + left-add_width/2),
                                                  int(landmData[j+1] * cropped.rows + top-add_height/2));
                }
                const Point* ppt[1] = {ptsmas};
                int npt[] = {17};
                polylines(shwimg, ppt, npt, 1, false, yellow, 1);
            }
            else
            {
                break;
            }
        }
        imshow(win1, shwimg);
    }
}
