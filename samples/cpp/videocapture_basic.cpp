/**
  @file videocapture_basic.cpp
  @brief A very basic sample for using VideoCapture and VideoWriter
  @author PkLab.net
  @date Aug 24, 2016
*/

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <cstdlib> // For system()

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <video_file_path>" << endl;
        return -1;
    }

    string videoFilePath = argv[1];
    Mat frame;
    VideoCapture cap(videoFilePath);

    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open video file: " << videoFilePath << endl;
        return -1;
    }

    int frameWidth = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    int fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G');

    system("mkdir -p videocapture_basic");
    string outputFilePath = "videocapture_basic/output.avi";
    VideoWriter writer(outputFilePath, fourcc, fps, Size(frameWidth, frameHeight));

    if (!writer.isOpened()) {
        cerr << "Could not open the output video file for write\n";
        return -1;
    }

    cout << "Start processing video: " << videoFilePath << endl;
    cout << "Press any key to terminate" << endl;

    while (cap.read(frame)) {
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        writer.write(frame);
        // imshow("Live", frame); // 注释掉
        // if (waitKey(5) >= 0) // 注释掉
        //    break; // 注释掉
    }

    cout << "Video saved to " << outputFilePath << endl;
    return 0;
}

