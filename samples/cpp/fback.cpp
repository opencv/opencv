#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static void help(char** argv)
{
    cout <<
            "\nThis program demonstrates dense optical flow algorithm by Gunnar Farneback\n"
            "Mainly the function: calcOpticalFlowFarneback()\n"
            "Call:\n"
        <<  argv[0]
        <<  " [video_file]\n" << endl;
}

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, const Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), color);
            circle(cflowmap, Point(x, y), 2, color, -1);
        }
}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, "{help h||}{@input||}");
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }

    string input = parser.get<string>("@input");
    VideoCapture cap;

    if (input.empty())
    {
        cap.open(0);  // Try to open the default camera
    }
    else
    {
        cap.open(samples::findFileOrKeep(input));  // Try to open the provided video file
    }

    if (!cap.isOpened())
    {
        cerr << "ERROR: Could not open camera or video file." << endl;
        return -1;
    }

    Mat flow, cflow, frame;
    UMat gray, prevgray, uflow;

    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        if (!prevgray.empty())
        {
            calcOpticalFlowFarneback(prevgray, gray, uflow, 0.5, 3, 15, 3, 5, 1.2, 0);
            cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
            uflow.copyTo(flow);
            drawOptFlowMap(flow, cflow, 16, Scalar(0, 255, 0));

            // Save the result image
            imwrite("optical_flow_result.png", cflow);
            cout << "Result image saved as: optical_flow_result.png" << endl;
            break;  // Process only the first frame for demonstration
        }
        std::swap(prevgray, gray);
    }
    return 0;
}

