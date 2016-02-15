/*
* npr_demo.cpp
*
* Author:
* Siddharth Kherada <siddharthkherada27[at]gmail[dot]com>
*
* This tutorial demonstrates how to use OpenCV Non-Photorealistic Rendering Module.
* 1) Edge Preserve Smoothing
*    -> Using Normalized convolution Filter
*    -> Using Recursive Filter
* 2) Detail Enhancement
* 3) Pencil sketch/Color Pencil Drawing
* 4) Stylization
*
*/

#include <signal.h>
#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <iostream>
#include <stdlib.h>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        cout << "usage: " << argv[0] << " <Input image> "  << endl;
        exit(0);
    }

    int num,type;

    Mat src = imread(argv[1], IMREAD_COLOR);

    if(src.empty())
    {
        cout <<  "Image not found" << endl;
        exit(0);
    }

    cout << endl;
    cout << " Edge Preserve Filter" << endl;
    cout << "----------------------" << endl;

    cout << "Options: " << endl;
    cout << endl;

    cout << "1) Edge Preserve Smoothing" << endl;
    cout << "   -> Using Normalized convolution Filter" << endl;
    cout << "   -> Using Recursive Filter" << endl;
    cout << "2) Detail Enhancement" << endl;
    cout << "3) Pencil sketch/Color Pencil Drawing" << endl;
    cout << "4) Stylization" << endl;
    cout << endl;

    cout << "Press number 1-4 to choose from above techniques: ";

    cin >> num;

    Mat img;

    if(num == 1)
    {
        cout << endl;
        cout << "Press 1 for Normalized Convolution Filter and 2 for Recursive Filter: ";

        cin >> type;

        edgePreservingFilter(src,img,type);
        imshow("Edge Preserve Smoothing",img);

    }
    else if(num == 2)
    {
        detailEnhance(src,img);
        imshow("Detail Enhanced",img);
    }
    else if(num == 3)
    {
        Mat img1;
        pencilSketch(src,img1, img, 10 , 0.1f, 0.03f);
        imshow("Pencil Sketch",img1);
        imshow("Color Pencil Sketch",img);
    }
    else if(num == 4)
    {
        stylization(src,img);
        imshow("Stylization",img);
    }
    waitKey(0);
}
