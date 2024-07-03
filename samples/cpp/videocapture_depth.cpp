#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace cv;

static void colorizeDisparity(const Mat& gray, Mat& rgb, double maxDisp=-1.f)
{
    CV_Assert(!gray.empty());
    CV_Assert(gray.type() == CV_8UC1);

    if(maxDisp <= 0)
    {
        maxDisp = 0;
        minMaxLoc(gray, nullptr, &maxDisp);
    }

    rgb = Mat::zeros(gray.size(), CV_8UC3);
    if(maxDisp < 1)
        return;

    Mat tmp;
    convertScaleAbs(gray, tmp, 255.f / maxDisp);
    applyColorMap(tmp, rgb, COLORMAP_JET);
}

static float getMaxDisparity(VideoCapture& capture, int minDistance)
{
    float b = (float)capture.get(CAP_OPENNI_DEPTH_GENERATOR_BASELINE); // mm
    float F = (float)capture.get(CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH); // pixels
    return b * F / minDistance;
}

static void colorizeDepth(const Mat& depth, Mat& rgb)
{
    CV_Assert(!depth.empty());

    normalize(depth, rgb, 0, 255, NORM_MINMAX, CV_8UC1);
    applyColorMap(rgb, rgb, COLORMAP_JET);
}

const char* keys = "{type t | | Camera Type: OpenNI, RealSense, Orbbec}"
                   "{dist d | 400 | The minimum measurable distance in milimeter between the camera and the object}"
                   "{help h | | Help}";

const char* about =
            "\nThis example demostrates how to get data from 3D cameras via OpenCV.\n"
            "Currently OpenCV supports 3 types of 3D cameras:\n"
            "1. Depth sensors compatible with OpenNI (Kinect, XtionPRO). "
            "Users must install OpenNI library and PrimeSensorModule for OpenNI and configure OpenCV with WITH_OPENNI flag ON in CMake.\n"
            "2. Depth sensors compatible with Intel® RealSense SDK (RealSense). "
            "Users must install Intel RealSense SDK 2.0 and configure OpenCV with WITH_LIBREALSENSE flag ON in CMake.\n"
            "3. Orbbec UVC depth sensors. "
            "For the use of OpenNI based Orbbec cameras, please refer to the example openni_orbbec_astra.cpp\n";

int main(int argc, char *argv[])
{
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if(parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    if (parser.has("type"))
    {
        int backend;
        int flagDepth, flagBGR, flagIR;
        bool hasDisparity = false;

        String camType = parser.get<String>("type");
        if (camType == "OpenNI")
        {
            backend = CAP_OPENNI2;
            flagDepth = CAP_OPENNI_DEPTH_MAP;
            flagBGR = CAP_OPENNI_BGR_IMAGE;
            flagIR = CAP_OPENNI_IR_IMAGE;
            hasDisparity = true;
        }
        else if (camType == "RealSense")
        {
            backend = CAP_INTELPERC;
            flagDepth = CAP_INTELPERC_DEPTH_MAP;
            flagBGR = CAP_INTELPERC_IMAGE;
            flagIR = CAP_INTELPERC_IR_MAP;
        }
        else if (camType == "Orbbec")
        {
            backend = CAP_OBSENSOR;
            flagDepth = CAP_OBSENSOR_DEPTH_MAP;
            flagBGR = CAP_OBSENSOR_BGR_IMAGE;
            flagIR = CAP_OBSENSOR_IR_IMAGE;  // Note output IR stream is not implemented for now.
        }
        else
        {
            std::cerr << "Invalid input of camera type." << std::endl;
            return -1;
        }

        VideoCapture capture(backend);
        if(!capture.isOpened())
        {
            std::cerr << "Fail to open depth camera." << std::endl;
            return -1;
        }

        if (camType == "OpenNI")
        {
            capture.set(CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_VGA_30HZ);
            capture.set(CAP_OPENNI_DEPTH_GENERATOR_PRESENT, true);
            capture.set(CAP_OPENNI_IMAGE_GENERATOR_PRESENT, true);
            capture.set(CAP_OPENNI_IR_GENERATOR_PRESENT, true);
        }

        Mat depthMap;
        Mat bgrImage;
        Mat irImage;
        Mat disparityMap;

        for(;;)
        {
            if(capture.grab())
            {
                if(capture.retrieve(depthMap, flagDepth))
                {
                    Mat colorDepthMap;
                    colorizeDepth(depthMap, colorDepthMap);
                    imshow("Colorized Depth Map", colorDepthMap);
                }

                if(capture.retrieve(bgrImage, flagBGR))
                    imshow("RGB Image", bgrImage);

                if(capture.retrieve(irImage, flagIR))
                {
                    if (camType == "OpenNI")
                    {
                        Mat ir8;
                        irImage.convertTo(ir8, CV_8U, 256.0 / 3500, 0.0);
                        imshow("Infrared Image", ir8);
                    }
                    else
                        imshow("Infrared Image", irImage);
                }

                if (hasDisparity)
                {
                    int minDist = parser.get<int>("dist"); // mm
                    if(capture.retrieve(disparityMap, CAP_OPENNI_DISPARITY_MAP))
                    {
                        Mat colorDisparityMap;
                        colorizeDisparity(disparityMap, colorDisparityMap, getMaxDisparity(capture, minDist));
                        colorDisparityMap.setTo(Scalar(0, 0, 0), disparityMap == 0);
                        imshow("Colorized Disparity Map", colorDisparityMap);
                    }
                }
            }

            if(waitKey(30) >= 0)
                break;
        }
    }
    else
    {
        parser.printMessage();
        return 0;
    }

    return 0;
}