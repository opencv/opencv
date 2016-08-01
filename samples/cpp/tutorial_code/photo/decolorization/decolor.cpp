/*
* decolor.cpp
*
* Author:
* Siddharth Kherada <siddharthkherada27[at]gmail[dot]com>
*
* This tutorial demonstrates how to use OpenCV Decolorization Module.
*
* Input:
* Color Image
*
* Output:
* 1) Grayscale image
* 2) Color boost image
*
*/

#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    CV_Assert(argc == 2);
    Mat src;
    src = imread(argv[1], IMREAD_COLOR);

    Mat gray = Mat(src.size(),CV_8UC1);
    Mat color_boost = Mat(src.size(),CV_8UC3);

    decolor(src,gray,color_boost);
    imshow("grayscale",gray);
    imshow("color_boost",color_boost);
    waitKey(0);
}
