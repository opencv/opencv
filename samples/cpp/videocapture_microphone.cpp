#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <string>
#include <cstdlib> // 包含 system 函数

using namespace cv;
using namespace std;

static void help(char** argv)
{
    cout << "\nThis sample shows you how to capture audio data using the VideoCapture interface.\n"
         << "Usage: " << argv[0] << " <video_file>\n"
         << "Video file should be an official OpenCV sample video file."
         << endl;
}

namespace custom {
    string depthToString(int depth)
    {
        switch (depth)
        {
            case CV_8U: return "8U";
            case CV_8S: return "8S";
            case CV_16U: return "16U";
            case CV_16S: return "16S";
            case CV_32S: return "32S";
            case CV_32F: return "32F";
            case CV_64F: return "64F";
            default: return "User";
        }
    }
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

    Mat frame;
    VideoCapture cap;

    cap.open(video_file); // 使用视频文件而不是麦克风
    if (!cap.isOpened())
    {
        cerr << "ERROR! Can't open video file" << endl;
        return -1;
    }

    // 仅显示视频信息
    cout << "CAP_PROP_FRAME_WIDTH: " << cap.get(CAP_PROP_FRAME_WIDTH) << endl;
    cout << "CAP_PROP_FRAME_HEIGHT: " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
    cout << "CAP_PROP_FPS: " << cap.get(CAP_PROP_FPS) << endl;

    const double cvTickFreq = getTickFrequency();
    int64 sysTimeCurr = getTickCount();
    int64 sysTimePrev = sysTimeCurr;
    while ((sysTimeCurr - sysTimePrev) / cvTickFreq < 10)
    {
        if (cap.grab())
        {
            cap.retrieve(frame);
            sysTimeCurr = getTickCount();
        }
        else
        {
            cerr << "Grab error" << endl;
            break;
        }
    }

    // 创建子目录
    system("mkdir -p videocapture_microphone");

    // 保存视频帧
    string output_filename = "videocapture_microphone/frame.jpg";
    imwrite(output_filename, frame);
    cout << "Saved: " << output_filename << endl;

    return 0;
}

