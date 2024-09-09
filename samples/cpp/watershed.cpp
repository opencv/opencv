#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib> // 包含 system 函数

using namespace cv;
using namespace std;

static void help(char** argv)
{
    cout << "\nThis program demonstrates the famous watershed segmentation algorithm in OpenCV: watershed()\n"
            "Usage:\n" << argv[0] <<" [image_name -- default is fruits.jpg]\n" << endl;

    cout << "Hot keys: \n"
        "\tESC - quit the program\n"
        "\tr - restore the original image\n"
        "\tw or SPACE - run watershed segmentation algorithm\n"
        "\t\t(before running it, *roughly* mark the areas to segment on the image)\n"
        "\t  (before that, roughly outline several markers on the image)\n";
}

Mat markerMask, img;
Point prevPt(-1, -1);

static void onMouse(int event, int x, int y, int flags, void*)
{
    if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
        return;
    if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
        prevPt = Point(-1, -1);
    else if (event == EVENT_LBUTTONDOWN)
        prevPt = Point(x, y);
    else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
    {
        Point pt(x, y);
        if (prevPt.x < 0)
            prevPt = pt;
        line(markerMask, prevPt, pt, Scalar::all(255), 5, 8, 0);
        line(img, prevPt, pt, Scalar::all(255), 5, 8, 0);
        prevPt = pt;
    }
}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, "{help h | | }{ @input | fruits.jpg | }");
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
    string filename = samples::findFile(parser.get<string>("@input"));
    Mat img0 = imread(filename, IMREAD_COLOR), imgGray;

    if (img0.empty())
    {
        cout << "Couldn't open image ";
        help(argv);
        return 0;
    }
    help(argv);

    // 创建子目录
    system("mkdir -p watershed");

    img0.copyTo(img);
    cvtColor(img, markerMask, COLOR_BGR2GRAY);
    cvtColor(markerMask, imgGray, COLOR_GRAY2BGR);
    markerMask = Scalar::all(0);

    setMouseCallback("image", onMouse, 0);

    // 模拟鼠标绘制操作
    // 这里可以直接调用 onMouse 函数来模拟鼠标事件，简单示例：
    onMouse(EVENT_LBUTTONDOWN, img.cols / 2, img.rows / 2, EVENT_FLAG_LBUTTON, nullptr);
    onMouse(EVENT_MOUSEMOVE, img.cols / 2 + 50, img.rows / 2, EVENT_FLAG_LBUTTON, nullptr);
    onMouse(EVENT_LBUTTONUP, img.cols / 2 + 50, img.rows / 2, EVENT_FLAG_LBUTTON, nullptr);

    int i, j, compCount = 0;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours(markerMask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    if (contours.empty())
        return 0;
    Mat markers(markerMask.size(), CV_32S);
    markers = Scalar::all(0);
    int idx = 0;
    for (; idx >= 0; idx = hierarchy[idx][0], compCount++)
        drawContours(markers, contours, idx, Scalar::all(compCount + 1), -1, 8, hierarchy, INT_MAX);

    if (compCount == 0)
        return 0;

    vector<Vec3b> colorTab;
    for (i = 0; i < compCount; i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);

        colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    double t = (double)getTickCount();
    watershed(img0, markers);
    t = (double)getTickCount() - t;
    printf("execution time = %gms\n", t * 1000. / getTickFrequency());

    Mat wshed(markers.size(), CV_8UC3);

    // paint the watershed image
    for (i = 0; i < markers.rows; i++)
        for (j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i, j);
            if (index == -1)
                wshed.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
            else if (index <= 0 || index > compCount)
                wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
            else
                wshed.at<Vec3b>(i, j) = colorTab[index - 1];
        }

    wshed = wshed * 0.5 + imgGray * 0.5;

    string output_filename = "watershed/watershed_transform.png";
    imwrite(output_filename, wshed);
    cout << "Saved watershed image: " << output_filename << endl;

    return 0;
}

