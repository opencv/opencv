#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video/background_segm.hpp"
#include <iostream>
#include <string>
#include <cstdlib>

using namespace std;
using namespace cv;

static void help(char** argv)
{
    printf("\n"
           "This program demonstrated a simple method of connected components clean up of background subtraction\n"
           "When the program starts, it begins learning the background.\n"
           "You can toggle background learning on and off by hitting the space bar.\n"
           "Call\n"
           "%s [video file, else it reads camera 0]\n\n", argv[0]);
}

static void refineSegments(const Mat& img, Mat& mask, Mat& dst)
{
    int niters = 3;

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    Mat temp;

    dilate(mask, temp, Mat(), Point(-1,-1), niters);
    erode(temp, temp, Mat(), Point(-1,-1), niters*2);
    dilate(temp, temp, Mat(), Point(-1,-1), niters);

    findContours(temp, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    dst = Mat::zeros(img.size(), CV_8UC3);

    if (contours.size() == 0)
        return;

    int idx = 0, largestComp = 0;
    double maxArea = 0;

    for (; idx >= 0; idx = hierarchy[idx][0])
    {
        const vector<Point>& c = contours[idx];
        double area = fabs(contourArea(Mat(c)));
        if (area > maxArea)
        {
            maxArea = area;
            largestComp = idx;
        }
    }
    Scalar color(0, 0, 255);
    drawContours(dst, contours, largestComp, color, FILLED, LINE_8, hierarchy);
}

int main(int argc, char** argv)
{
    VideoCapture cap;
    bool update_bg_model = true;

    CommandLineParser parser(argc, argv, "{help h||}{@input||}");
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
    string input = parser.get<string>("@input");
    if (input.empty())
        cap.open(0);
    else
        cap.open(samples::findFileOrKeep(input));

    if (!cap.isOpened())
    {
        cout << "Can not open camera or video file" << endl;
        return -1;
    }

    // 创建子目录 "segment_objects"
    string outputDir = "segment_objects";
    system(("mkdir -p " + outputDir).c_str());  // 创建目录

    Mat tmp_frame, bgmask, out_frame;

    cap >> tmp_frame;
    if (tmp_frame.empty())
    {
        cout << "Can not read data from the video source" << endl;
        return -1;
    }

    Ptr<BackgroundSubtractorMOG2> bgsubtractor = createBackgroundSubtractorMOG2();
    bgsubtractor->setVarThreshold(10);

    int frame_count = 0;
    for (;;)
    {
        cap >> tmp_frame;
        if (tmp_frame.empty())
            break;

        bgsubtractor->apply(tmp_frame, bgmask, update_bg_model ? -1 : 0);
        refineSegments(tmp_frame, bgmask, out_frame);

        // 保存每一帧的处理结果
        string filename = outputDir + "/segmented_frame_" + to_string(frame_count++) + ".png";
        imwrite(filename, out_frame);
        cout << "Saved: " << filename << endl;

        // char keycode = (char)waitKey(30);
        // if (keycode == 27)
        //     break;
        // if (keycode == ' ')
        // {
        //     update_bg_model = !update_bg_model;
        //     cout << "Learn background is in state = " << update_bg_model << endl;
        // }
    }

    return 0;
}

