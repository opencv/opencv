#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <cstdlib> // 包含 system 函数

using namespace cv;
using namespace std;

static void help(char** argv)
{
    cout << "\nThis sample shows you how to use OpenCV VideoCapture with a video file instead of an obsensor.\n"
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

    VideoCapture cap(video_file);
    if (!cap.isOpened())
    {
        cerr << "ERROR! Can't open video file" << endl;
        return -1;
    }

    Mat image;
    Mat depthMap;
    Mat adjDepthMap;

    // 创建子目录
    system("mkdir -p videocapture_obsensor");

    // Minimum depth value
    const double minVal = 300;
    // Maximum depth value
    const double maxVal = 5000;

    int frame_count = 0;

    while (true)
    {
        if (cap.grab())
        {
            cap.retrieve(image);
            // 模拟深度图
            depthMap = Mat(image.rows, image.cols, CV_16U);
            randu(depthMap, Scalar::all(minVal), Scalar::all(maxVal));

            // 处理深度图
            depthMap.convertTo(adjDepthMap, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
            applyColorMap(adjDepthMap, adjDepthMap, COLORMAP_JET);

            // 保存图像和深度图
            string rgb_filename = "videocapture_obsensor/rgb_" + to_string(frame_count) + ".jpg";
            string depth_filename = "videocapture_obsensor/depth_" + to_string(frame_count) + ".png";

            imwrite(rgb_filename, image);
            imwrite(depth_filename, adjDepthMap);

            cout << "Saved: " << rgb_filename << " and " << depth_filename << endl;

            frame_count++;

            if (frame_count >= 10) // 处理10帧后退出
                break;

            image.release();
            depthMap.release();
        }
        else
        {
            cerr << "Grab error" << endl;
            break;
        }
    }

    return 0;
}

