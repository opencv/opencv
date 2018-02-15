/*
* create_mask.cpp
*
* Author:
* Siddharth Kherada <siddharthkherada27[at]gmail[dot]com>
*
* This tutorial demonstrates how to make mask image (black and white).
* The program takes as input a source image and outputs its corresponding
* mask image.
*/

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <iostream>
#include <stdlib.h>

using namespace std;
using namespace cv;

Mat img0, img1, res1, final;

Point point;
int drag = 0;

int numpts = 100;
Point* pts = new Point[100];

int var = 0;
int flag = 0;
int flag1 = 0;

int minx,miny,maxx,maxy,lenx,leny;

void mouseHandler(int, int, int, int, void*);

void mouseHandler(int event, int x, int y, int, void*)
{

    if (event == EVENT_LBUTTONDOWN && !drag)
    {
        if(flag1 == 0)
        {
            if(var==0)
                img1 = img0.clone();
            point = Point(x, y);
            circle(img1,point,2,Scalar(0, 0, 255),-1, 8, 0);
            pts[var] = point;
            var++;
            drag  = 1;
            if(var>1)
                line(img1,pts[var-2], point, Scalar(0, 0, 255), 2, 8, 0);

            imshow("Source", img1);
        }
    }

    if (event == EVENT_LBUTTONUP && drag)
    {
        imshow("Source", img1);

        drag = 0;
    }
    if (event == EVENT_RBUTTONDOWN)
    {
        flag1 = 1;
        img1 = img0.clone();
        for(int i = var; i < numpts ; i++)
            pts[i] = point;

        if(var!=0)
        {
            const Point* pts3[1] = {&pts[0]};
            polylines( img1, pts3, &numpts,1, 1, Scalar(0,0,0), 2, 8, 0);
        }

        for(int i=0;i<var;i++)
        {
            minx = min(minx,pts[i].x);
            maxx = max(maxx,pts[i].x);
            miny = min(miny,pts[i].y);
            maxy = max(maxy,pts[i].y);
        }
        lenx = maxx - minx;
        leny = maxy - miny;

        imshow("Source", img1);
    }

    if (event == EVENT_RBUTTONUP)
    {
        flag = var;

        final = Mat::zeros(img0.size(),CV_8UC3);
        res1 = Mat::zeros(img0.size(),CV_8UC1);
        const Point* pts4[1] = {&pts[0]};

        fillPoly(res1, pts4,&numpts, 1, Scalar(255, 255, 255), 8, 0);
        bitwise_and(img0, img0, final,res1);
        imshow("mask",res1);
        imwrite("mask.png",res1);

        imshow("Source", img1);

    }
    if (event == EVENT_MBUTTONDOWN)
    {
        for(int i = 0; i < numpts ; i++)
        {
            pts[i].x=0;
            pts[i].y=0;
        }
        var = 0;
        flag1 = 0;
        minx = INT_MAX; miny = INT_MAX; maxx = INT_MIN; maxy = INT_MIN;
        imshow("Source", img0);
        drag = 0;
    }
}

static void help()
{
    cout << "\nThis program demonstrates using mouse events"
        "\nCall:\n"
        "./create_mask <image_name>\n"
        "\n"
        "\tleft mouse button - set a point to create mask shape"
        "\n"
        "\tright mouse button - create mask from points\n"
        "\tmiddle mouse button - reset\n" << endl;
}

int main(int argc, char **argv)
{
    cv::CommandLineParser parser(argc, argv, "{@input | ../data/lena.jpg | input image}");
    help();
    string input_image = parser.get<string>("@input");
    if (input_image.empty())
    {
        parser.printMessage();
        parser.printErrors();
        return 0;
    }

    Mat src = imread(input_image);

    minx = INT_MAX; miny = INT_MAX; maxx = INT_MIN; maxy = INT_MIN;

    img0 = src;

    res1 = Mat::zeros(img0.size(),CV_8UC1);
    final = Mat::zeros(img0.size(),CV_8UC3);
    //////////// source image ///////////////////

    namedWindow("Source", 1);
    setMouseCallback("Source", mouseHandler, NULL);
    imshow("Source", img0);
    waitKey(0);

    return 0;
}
