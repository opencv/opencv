#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <cstdlib> // 包含 system 函数

using namespace cv;
using namespace std;

static void help()
{
    cout << "\nThis program demonstrates usage of depth sensors (Kinect, XtionPRO,...).\n"
         << "The user gets some of the supported output images.\n"
         "\nAll supported output map types:\n"
         "1.) Data given from depth generator\n"
         "   CAP_OPENNI_DEPTH_MAP            - depth values in mm (CV_16UC1)\n"
         "   CAP_OPENNI_POINT_CLOUD_MAP      - XYZ in meters (CV_32FC3)\n"
         "   CAP_OPENNI_DISPARITY_MAP        - disparity in pixels (CV_8UC1)\n"
         "   CAP_OPENNI_DISPARITY_MAP_32F    - disparity in pixels (CV_32FC1)\n"
         "   CAP_OPENNI_VALID_DEPTH_MASK     - mask of valid pixels (not occluded, not shaded etc.) (CV_8UC1)\n"
         "2.) Data given from RGB image generator\n"
         "   CAP_OPENNI_BGR_IMAGE            - color image (CV_8UC3)\n"
         "   CAP_OPENNI_GRAY_IMAGE           - gray image (CV_8UC1)\n"
         "2.) Data given from IR image generator\n"
         "   CAP_OPENNI_IR_IMAGE             - gray image (CV_16UC1)\n"
         << endl;
}

static void colorizeDisparity(const Mat& gray, Mat& rgb, double maxDisp = -1.f)
{
    CV_Assert(!gray.empty());
    CV_Assert(gray.type() == CV_8UC1);

    if (maxDisp <= 0)
    {
        maxDisp = 0;
        minMaxLoc(gray, 0, &maxDisp);
    }

    rgb.create(gray.size(), CV_8UC3);
    rgb = Scalar::all(0);
    if (maxDisp < 1)
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

static void printCommandLineParams()
{
    cout << "-cd=       Colorized disparity? (0 or 1; 1 by default) Ignored if disparity map is not selected to show." << endl;
    cout << "-fmd=      Fixed max disparity? (0 or 1; 0 by default) Ignored if disparity map is not colorized (-cd 0)." << endl;
    cout << "-mode=     image mode: resolution and fps, supported three values:  0 - CAP_OPENNI_VGA_30HZ, 1 - CAP_OPENNI_SXGA_15HZ," << endl;
    cout << "          2 - CAP_OPENNI_SXGA_30HZ (0 by default). Ignored if rgb image or gray image are not selected to show." << endl;
    cout << "-m=        Mask to set which output images are need. It is a string of size 6. Each element of this is '0' or '1' and" << endl;
    cout << "          determine: is depth map, disparity map, valid pixels mask, rgb image, gray image need or not (correspondently), ir image" << endl ;
    cout << "          By default -m=010100 i.e. disparity map and rgb image will be shown." << endl ;
    cout << "-r=        Filename of .oni video file. The data will grabbed from it." << endl ;
}

static void parseCommandLine(int argc, char* argv[], bool& isColorizeDisp, bool& isFixedMaxDisp, int& imageMode, bool retrievedImageFlags[],
                             string& filename, bool& isFileReading)
{
    filename.clear();
    cv::CommandLineParser parser(argc, argv, "{h help||}{cd|1|}{fmd|0|}{mode|-1|}{m|010100|}{r||}");
    if (parser.has("h"))
    {
        help();
        printCommandLineParams();
        exit(0);
    }
    isColorizeDisp = (parser.get<int>("cd") != 0);
    isFixedMaxDisp = (parser.get<int>("fmd") != 0);
    imageMode = parser.get<int>("mode");
    int flags = parser.get<int>("m");
    isFileReading = parser.has("r");
    if (isFileReading)
        filename = parser.get<string>("r");
    if (!parser.check())
    {
        parser.printErrors();
        help();
        exit(-1);
    }
    if (flags % 1000000 == 0)
    {
        cout << "No one output image is selected." << endl;
        exit(0);
    }
    for (int i = 0; i < 6; i++)
    {
        retrievedImageFlags[5 - i] = (flags % 10 != 0);
        flags /= 10;
    }
}

/*
 * To work with Kinect or XtionPRO the user must install OpenNI library and PrimeSensorModule for OpenNI and
 * configure OpenCV with WITH_OPENNI flag is ON (using CMake).
 */
int main(int argc, char* argv[])
{
    bool isColorizeDisp, isFixedMaxDisp;
    int imageMode;
    bool retrievedImageFlags[6];
    string filename;
    bool isVideoReading;
    parseCommandLine(argc, argv, isColorizeDisp, isFixedMaxDisp, imageMode, retrievedImageFlags, filename, isVideoReading);

    cout << "Device opening ..." << endl;
    VideoCapture capture;
    if (isVideoReading)
        capture.open(filename);
    else
    {
        cerr << "No video file provided. Exiting..." << endl;
        return -1;
    }

    cout << "done." << endl;

    if (!capture.isOpened())
    {
        cout << "Can not open a capture object." << endl;
        return -1;
    }

    // 创建子目录
    system("mkdir -p videocapture_openni");

    int frame_count = 0;

    for (;;)
    {
        Mat depthMap;
        Mat validDepthMap;
        Mat disparityMap;
        Mat bgrImage;
        Mat grayImage;
        Mat irImage;

        if (!capture.grab())
        {
            cout << "Can not grab images." << endl;
            return -1;
        }
        else
        {
            if (retrievedImageFlags[0] && capture.retrieve(depthMap, CAP_OPENNI_DEPTH_MAP))
            {
                string depth_filename = "videocapture_openni/depth_" + to_string(frame_count) + ".png";
                imwrite(depth_filename, depthMap);
                cout << "Saved " << depth_filename << endl;
            }

            if (retrievedImageFlags[1] && capture.retrieve(disparityMap, CAP_OPENNI_DISPARITY_MAP))
            {
                string disparity_filename = "videocapture_openni/disparity_" + to_string(frame_count) + ".png";
                imwrite(disparity_filename, disparityMap);
                cout << "Saved " << disparity_filename << endl;
            }

            if (retrievedImageFlags[2] && capture.retrieve(validDepthMap, CAP_OPENNI_VALID_DEPTH_MASK))
            {
                string valid_depth_filename = "videocapture_openni/valid_depth_" + to_string(frame_count) + ".png";
                imwrite(valid_depth_filename, validDepthMap);
                cout << "Saved " << valid_depth_filename << endl;
            }

            if (retrievedImageFlags[3] && capture.retrieve(bgrImage, CAP_OPENNI_BGR_IMAGE))
            {
                string bgr_filename = "videocapture_openni/bgr_" + to_string(frame_count) + ".jpg";
                imwrite(bgr_filename, bgrImage);
                cout << "Saved " << bgr_filename << endl;
            }

            if (retrievedImageFlags[4] && capture.retrieve(grayImage, CAP_OPENNI_GRAY_IMAGE))
            {
                string gray_filename = "videocapture_openni/gray_" + to_string(frame_count) + ".png";
                imwrite(gray_filename, grayImage);
                cout << "Saved " << gray_filename << endl;
            }

            if (retrievedImageFlags[5] && capture.retrieve(irImage, CAP_OPENNI_IR_IMAGE))
            {
                Mat ir8;
                irImage.convertTo(ir8, CV_8U, 256.0 / 3500, 0.0);
                string ir_filename = "videocapture_openni/ir_" + to_string(frame_count) + ".png";
                imwrite(ir_filename, ir8);
                cout << "Saved " << ir_filename << endl;
            }
        }

        frame_count++;

        if (frame_count >= 10) // 处理10帧后退出
            break;
    }

    return 0;
}

