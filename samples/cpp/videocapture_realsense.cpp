#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <cstdlib> // 包含 system 函数

using namespace cv;
using namespace std;

static void help(char** argv)
{
    cout << "\nThis sample shows you how to use OpenCV VideoCapture with a RealSense camera.\n"
         << "Usage: " << argv[0] << " <video_file>\n"
         << "Video file should be an official OpenCV sample video file."
         << endl;
}

int main(int argc, char** argv)
{
    help(argv);
    cv::CommandLineParser parser(argc, argv, "{@video| ../data/Megamind.avi |}");
    string video_file = parser.get<string>("@video");

    if(video_file.empty())
    {
        cerr << "No video file provided!" << endl;
        return 1;
    }

    VideoCapture capture(video_file);
    if (!capture.isOpened())
    {
        cerr << "ERROR! Can't open video file" << endl;
        return -1;
    }

    Mat depthMap;
    Mat image;
    Mat irImage;
    Mat adjMap;

    // 创建子目录
    system("mkdir -p videocapture_realsense");

    int frame_count = 0;
    const double minVal = 300;
    const double maxVal = 5000;

    while (true)
    {
        if (capture.grab())
        {
            capture.retrieve(image);
            // 模拟深度图和IR图像
            depthMap = Mat(image.rows, image.cols, CV_16U);
            irImage = Mat(image.rows, image.cols, CV_8U);
            randu(depthMap, Scalar::all(minVal), Scalar::all(maxVal));
            randu(irImage, Scalar::all(0), Scalar::all(255));

            // 处理深度图
            normalize(depthMap, adjMap, 0, 255, NORM_MINMAX, CV_8UC1);
            applyColorMap(adjMap, adjMap, COLORMAP_JET);

            // 保存图像和深度图
            string rgb_filename = "videocapture_realsense/rgb_" + to_string(frame_count) + ".jpg";
            string depth_filename = "videocapture_realsense/depth_" + to_string(frame_count) + ".png";
            string ir_filename = "videocapture_realsense/ir_" + to_string(frame_count) + ".png";

            imwrite(rgb_filename, image);
            imwrite(depth_filename, adjMap);
            imwrite(ir_filename, irImage);

            cout << "Saved: " << rgb_filename << ", " << depth_filename << ", " << ir_filename << endl;

            frame_count++;

            if (frame_count >= 10) // 处理10帧后退出
                break;

            image.release();
            depthMap.release();
            irImage.release();
        }
        else
        {
            cerr << "Grab error" << endl;
            break;
        }
    }

    return 0;
}

