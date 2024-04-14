#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;


static void colorizeDisparity(const Mat& gray, Mat& rgb, double maxDisp=-1.f)
{
    CV_Assert(!gray.empty());
    CV_Assert(gray.type() == CV_8UC1);

    if(maxDisp <= 0)
    {
        maxDisp = 0;
        minMaxLoc(gray, 0, &maxDisp);
    }

    rgb.create(gray.size(), CV_8UC3);
    rgb = Scalar::all(0);
    if(maxDisp < 1)
        return;

    Mat tmp;
    convertScaleAbs(gray, tmp, 255.f / maxDisp);
    applyColorMap(tmp, rgb, COLORMAP_JET);
}

static float getMaxDisparity(VideoCapture& capture)
{
    const int minDistance = 400; // mm
    float b = (float)capture.get(CAP_OPENNI_DEPTH_GENERATOR_BASELINE); // mm
    float F = (float)capture.get(CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH); // pixels
    return b * F / minDistance;
}

static void openni_cam()
{
    VideoCapture capture(CAP_OPENNI2);
   if (!capture.isOpened())
   {
        cerr << "Failed to open a capture object." << endl;
        return;
   }

    capture.set(CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_VGA_30HZ);
    capture.set(CAP_OPENNI_DEPTH_GENERATOR_PRESENT, true);
    capture.set(CAP_OPENNI_IMAGE_GENERATOR_PRESENT, true);
    capture.set(CAP_OPENNI_IR_GENERATOR_PRESENT, true);

    Mat depthMap;
    Mat bgrImage;
    Mat irImage;
    Mat disparityMap;

    for(;;)
    {
        if (!capture.grab())
        {
            cerr << "Can not grab images." << endl;
            return;
        }
        else
        {
            if(capture.retrieve(depthMap, CAP_OPENNI_DEPTH_MAP))
            {
                const float scaleFactor = 0.05f;
                Mat show;
                depthMap.convertTo(show, CV_8UC1, scaleFactor);
                imshow("DEPTH", show);
            }

            if(capture.retrieve(bgrImage, CAP_OPENNI_BGR_IMAGE))
                imshow("RGB", bgrImage);

            if(capture.retrieve(irImage, CAP_OPENNI_IR_IMAGE))
            {
                const float scaleFactor = 256.0f / 3500;
                Mat ir8;
                irImage.convertTo(ir8, CV_8U, scaleFactor);
                imshow("IR", ir8);
            }

            if(capture.retrieve(disparityMap, CAP_OPENNI_DISPARITY_MAP))
            {
                Mat colorDisparityMap;
                colorizeDisparity(disparityMap, colorDisparityMap, getMaxDisparity(capture));
                Mat validColorDisparityMap;
                colorDisparityMap.copyTo(validColorDisparityMap, disparityMap != 0);
                imshow("Colorized Disparity Map", validColorDisparityMap);
            }
        }

        if(waitKey(30) >= 0)
            break;
    }
}

static void realsense_cam()
{
   VideoCapture capture(CAP_INTELPERC);
   if (!capture.isOpened())
   {
        cerr << "Failed to open RealSense camera." << endl;
        return;
   }

    Mat depthMap;
    Mat bgrImage;
    Mat irImage;

    for(;;)
    {
        if(capture.grab())
        {
            if(capture.retrieve(depthMap,CAP_INTELPERC_DEPTH_MAP))
            {
                Mat adjMap;
                normalize(depthMap, adjMap, 0, 255, NORM_MINMAX, CV_8UC1);
                applyColorMap(adjMap, adjMap, COLORMAP_JET);
                imshow("DEPTH", adjMap);
            }

            if(capture.retrieve(bgrImage,CAP_INTELPERC_IMAGE))
                imshow("RGB", bgrImage);

            if(capture.retrieve(irImage,CAP_INTELPERC_IR_MAP))
                imshow("IR", irImage);
        }

        if(waitKey(30) >= 0)
            break;
    }
}

static void orbbec_cam()
{
    VideoCapture capture(0, CAP_OBSENSOR);
    if(!capture.isOpened())
    {
        cerr << "Failed to open Orbbec camera."  << endl;
        return;
    }

    // get the intrinsic parameters of Orbbec camera
    double fx = capture.get(CAP_PROP_OBSENSOR_INTRINSIC_FX);
    double fy = capture.get(CAP_PROP_OBSENSOR_INTRINSIC_FY);
    double cx = capture.get(CAP_PROP_OBSENSOR_INTRINSIC_CX);
    double cy = capture.get(CAP_PROP_OBSENSOR_INTRINSIC_CY);
    cout << "Camera intrinsic params: fx=" << fx << ", fy=" << fy << ", cx=" << cx << ", cy=" << cy << endl;

    Mat depthMap;
    Mat bgrImage;

    for(;;)
    {
        // Get the depth map:
        // obsensorCapture >> depthMap;

        // Another way to get the depth map:
        if (capture.grab())
        {
            if (capture.retrieve(depthMap, CAP_OBSENSOR_DEPTH_MAP))
            {
                Mat adjMap;
                normalize(depthMap, adjMap, 0, 255, NORM_MINMAX, CV_8UC1);
                applyColorMap(adjMap, adjMap, COLORMAP_JET);
                imshow("DEPTH", adjMap);
            }

            if (capture.retrieve(bgrImage, CAP_OBSENSOR_BGR_IMAGE))
                imshow("RGB", bgrImage);
        }

        if(waitKey(30) >= 0)
            break;
    }
}

const char* keys = "{camType | | Camera Type: OpenNI, RealSense, Orbbec}";

const char* about =
            "\nThis example demostrates how to get data from 3D cameras via OpenCV.\n"
            "Currently OpenCV supports 3 types of 3D cameras:\n"
            "1. Depth sensors compatible with OpenNI (Kinect, XtionPRO). "
            "Users must install OpenNI library and PrimeSensorModule for OpenNI and configure OpenCV with WITH_OPENNI flag ON in CMake.\n"
            "2. Depth sensors compatible with Intel® RealSense SDK (RealSense). "
            "Users must install Intel RealSense SDK 2.0 and configure OpenCV with WITH_LIBREALSENSE flag ON in CMake.\n"
            "3. Orbbec depth sensors.\n";

int main(int argc, char *argv[])
{
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    if(argc < 2)
    {
        parser.printMessage();
        return 0;
    }

    String camType = parser.get<String>("camType");
    if (camType == "OpenNI")
    {
        openni_cam();
    }
    else if (camType == "RealSense")
    {
        realsense_cam();
    }
    else if (camType == "Orbbec")
    {
        orbbec_cam();
    }
    else
    {
        cerr << "Invalid input." << endl;
        return -1;
    }

    return 0;
}
