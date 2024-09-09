#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include <cstdlib> // 包含 system 函数

using namespace cv;
using namespace std;

//hide the local functions in an anon namespace
namespace {
    void help(char** av) {
        cout << "The program captures frames from a video file, image sequence (01.jpg, 02.jpg ... 10.jpg) or camera connected to your computer." << endl
             << "Usage:\n" << av[0] << " <video file, image sequence or device number>" << endl
             << "q,Q,esc -- quit" << endl
             << "space   -- save frame" << endl << endl
             << "\tTo capture from a camera pass the device number. To find the device number, try ls /dev/video*" << endl
             << "\texample: " << av[0] << " 0" << endl
             << "\tYou may also pass a video file instead of a device number" << endl
             << "\texample: " << av[0] << " video.avi" << endl
             << "\tYou can also pass the path to an image sequence and OpenCV will treat the sequence just like a video." << endl
             << "\texample: " << av[0] << " right%%02d.jpg" << endl;
    }

    int process(VideoCapture& capture) {
        int n = 0;
        char filename[200];
        Mat frame;

        // 创建子目录
        system("mkdir -p videocapture_starter");

        for (;;) {
            capture >> frame;
            if (frame.empty())
                break;

            // 模拟保存帧
            snprintf(filename, sizeof(filename), "videocapture_starter/frame%.3d.jpg", n++);
            imwrite(filename, frame);
            cout << "Saved " << filename << endl;

            // 注释掉显示图像的代码
            // imshow(window_name, frame);
            // char key = (char)waitKey(30); //delay N millis, usually long enough to display and capture input

            // switch (key) {
            // case 'q':
            // case 'Q':
            // case 27: //escape key
            //     return 0;
            // case ' ': //Save an image
            //     snprintf(filename,sizeof(filename),"filename%.3d.jpg",n++);
            //     imwrite(filename,frame);
            //     cout << "Saved " << filename << endl;
            //     break;
            // default:
            //     break;
            // }

            // 限制处理帧数
            if (n >= 10) // 处理10帧后退出
                break;
        }
        return 0;
    }
}

int main(int ac, char** av) {
    cv::CommandLineParser parser(ac, av, "{help h||}{@input||}");
    if (parser.has("help"))
    {
        help(av);
        return 0;
    }
    std::string arg = parser.get<std::string>("@input");
    if (arg.empty()) {
        help(av);
        return 1;
    }
    VideoCapture capture(arg); //try to open string, this will attempt to open it as a video file or image sequence
    if (!capture.isOpened()) //if this fails, try to open as a video camera, through the use of an integer param
        capture.open(atoi(arg.c_str()));
    if (!capture.isOpened()) {
        cerr << "Failed to open the video device, video file or image sequence!\n" << endl;
        help(av);
        return 1;
    }
    return process(capture);
}

