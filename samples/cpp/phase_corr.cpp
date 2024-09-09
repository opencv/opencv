#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <cstdlib>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    // Parse command line arguments
    CommandLineParser parser(argc, argv, "{@input||input video file}{@output||output directory}");
    parser.about("This sample demonstrates phase correlation for video stabilization.");
    parser.printMessage();

    // Get the input video file and output directory from arguments
    string inputVideo = parser.get<string>("@input");
    string outputDir = parser.get<string>("@output");

    if (inputVideo.empty() || outputDir.empty()) {
        cerr << "Error: Input video file and output directory must be provided." << endl;
        return -1;
    }

    // Create the output directory if it does not exist
    string mkdirCmd = "mkdir -p " + outputDir;
    system(mkdirCmd.c_str());

    VideoCapture video(inputVideo);
    if (!video.isOpened()) {
        cerr << "Error: Cannot open video file: " << inputVideo << endl;
        return -1;
    }

    Mat frame, curr, prev, curr64f, prev64f, hann;
    int frameCount = 0;

    while (true)
    {
        video >> frame;
        if (frame.empty())
            break;

        cvtColor(frame, curr, COLOR_RGB2GRAY);

        if (prev.empty())
        {
            prev = curr.clone();
            createHanningWindow(hann, curr.size(), CV_64F);
        }

        prev.convertTo(prev64f, CV_64F);
        curr.convertTo(curr64f, CV_64F);

        Point2d shift = phaseCorrelate(prev64f, curr64f, hann);
        double radius = std::sqrt(shift.x * shift.x + shift.y * shift.y);

        if (radius > 5)
        {
            // Draw a circle and line indicating the shift direction...
            Point center(curr.cols >> 1, curr.rows >> 1);
            circle(frame, center, (int)radius, Scalar(0, 255, 0), 3, LINE_AA);
            line(frame, center, Point(center.x + (int)shift.x, center.y + (int)shift.y), Scalar(0, 255, 0), 3, LINE_AA);
        }

        // Save the processed frame
        string frameFilename = outputDir + "/frame_" + to_string(frameCount) + ".png";
        imwrite(frameFilename, frame);
        cout << "Frame saved at: " << frameFilename << endl;

        prev = curr.clone();
        frameCount++;
    }

    return 0;
}

