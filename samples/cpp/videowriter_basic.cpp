/**
  @file videowriter_basic.cpp
  @brief A very basic sample for using VideoWriter and VideoCapture
  @author PkLab.net
  @date Aug 24, 2016
*/

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <stdio.h>
#include <cstdlib> // 包含 system 函数

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <video_file>" << endl;
        return 1;
    }

    string video_file = argv[1];

    Mat src;
    // 使用视频文件作为视频源
    VideoCapture cap(video_file);
    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open video file\n";
        return -1;
    }

    // get one frame from video to know frame size and type
    cap >> src;
    // check if we succeeded
    if (src.empty()) {
        cerr << "ERROR! blank frame grabbed\n";
        return -1;
    }
    bool isColor = (src.type() == CV_8UC3);

    //--- INITIALIZE VIDEOWRITER
    VideoWriter writer;
    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');  // select desired codec (must be available at runtime)
    double fps = 25.0;                          // framerate of the created video stream
    string filename = "./videowriter_basic/output.avi";   // name of the output video file

    // 创建子目录
    system("mkdir -p videowriter_basic");

    writer.open(filename, codec, fps, src.size(), isColor);
    // check if we succeeded
    if (!writer.isOpened()) {
        cerr << "Could not open the output video file for write\n";
        return -1;
    }

    //--- GRAB AND WRITE LOOP
    cout << "Writing videofile: " << filename << endl;
    int frame_count = 0;
    while (frame_count < 100) // 限制处理帧数为100帧
    {
        // check if we succeeded
        if (!cap.read(src)) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        // encode the frame into the videofile stream
        writer.write(src);
        frame_count++;
    }

    cout << "Saved video file: " << filename << endl;
    // the videofile will be closed and released automatically in VideoWriter destructor
    return 0;
}

