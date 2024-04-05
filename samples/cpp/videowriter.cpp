/**
  @file videowriter.cpp
  @brief A sample for VideoWriter and VideoCapture with options to specify video codec, fps and resolution
  @date April 05, 2024
*/


#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

inline map<string, int> fourccByCodec() {
    map<string, int> res;
    res["h264"] = VideoWriter::fourcc('H', '2', '6', '4');
    res["h265"] = VideoWriter::fourcc('H', 'E', 'V', 'C');
    res["mpeg2"] = VideoWriter::fourcc('M', 'P', 'E', 'G');
    res["mpeg4"] = VideoWriter::fourcc('M', 'P', '4', '2');
    res["mjpeg"] = VideoWriter::fourcc('M', 'J', 'P', 'G');
    res["vp8"] = VideoWriter::fourcc('V', 'P', '8', '0');
    return res;
}

inline map<string, Size> sizeByResolution() {
    map<string, Size> res;
    res["720p"] = Size(1280, 720);
    res["1080p"] = Size(1920, 1080);
    res["4k"] = Size(3840, 2160);
    return res;
}

int main(int argc, char** argv) {
    const String keys =
        "{help h usage ? |      | Print help message   }"
        "{fps           |30    | fix frame per second for encoding (supported: fps > 0)   }"
        "{codec         |mjpeg | codec name (supported: 'h264', 'h265', 'mpeg2', 'mpeg4', 'mjpeg', 'vp8')  }"
        "{resolution    |720p  | video resolution for encoding (supported: '720p', '1080p', '4k') }";

    CommandLineParser parser(argc, argv, keys);
    parser.about("Video Capture and Write with codec and resolution options");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    double fps = parser.get<double>("fps");
    string codecStr = parser.get<string>("codec");
    string resolutionStr = parser.get<string>("resolution");

    auto codecMap = fourccByCodec();
    auto resolutionMap = sizeByResolution();

    if (codecMap.find(codecStr) == codecMap.end() || resolutionMap.find(resolutionStr) == resolutionMap.end()) {
        cerr << "Invalid codec or resolution!" << endl;
        return -1;
    }

    int codec = codecMap[codecStr];
    Size resolution = resolutionMap[resolutionStr];

    // Video Capture
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    cout << "Selected FPS: " << fps << endl;
    cout << "Selected Codec: " << codecStr << endl;
    cout << "Selected Resolution: " << resolutionStr << " (" << resolution.width << "x" << resolution.height << ")" << endl;

    // Set up VideoWriter
    string filename = "./output_new.avi";
    VideoWriter writer(filename, codec, fps, resolution, true); // Assuming color video
    if (!writer.isOpened()) {
        cerr << "Could not open the output video file for write\n";
        return -1;
    }

    cout << "Writing video file: " << filename << endl
         << "Press any key to terminate" << endl;

    Mat frame, resizedFrame;
    for (;;) {
        // Capture frame
        if (!cap.read(frame) || frame.empty()) {
            break;
        }

        // Resize frame to desired resolution
        resize(frame, resizedFrame, resolution);

        // Write resized frame to video
        writer.write(resizedFrame);

        // Show live
        imshow("Live", resizedFrame);
        if (waitKey(5) >= 0) break;
    }

    // VideoWriter and VideoCapture are automatically closed by their destructors

    return 0;
}
