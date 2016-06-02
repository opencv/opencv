#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>

using namespace std;
using namespace cv;

static void help()
{
    cout << "\nThis program demonstrates template match with mask.\n"
            "Usage:\n"
            "./mask_tmpl -i=<image_name> -t=<template_name> -m=<mask_name>, Default is ../data/lena_tmpl.jpg\n"
            << endl;
}

int main( int argc, const char** argv )
{
    cv::CommandLineParser parser(argc, argv,
        "{help h||}"
        "{ i | ../data/lena_tmpl.jpg | }"
        "{ t | ../data/tmpl.png | }"
        "{ m | ../data/mask.png | }");
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    string filename = parser.get<string>("i");
    string tmplname = parser.get<string>("t");
    string maskname = parser.get<string>("m");
    Mat img = imread(filename);
    Mat tmpl = imread(tmplname);
    Mat mask = imread(maskname);
    Mat res;

    if(img.empty())
    {
        help();
        cout << "can not open " << filename << endl;
        return -1;
    }

    if(tmpl.empty())
    {
        help();
        cout << "can not open " << tmplname << endl;
        return -1;
    }

    if(mask.empty())
    {
        help();
        cout << "can not open " << maskname << endl;
        return -1;
    }

    //int method = CV_TM_SQDIFF;
    int method = CV_TM_CCORR_NORMED;
    matchTemplate(img, tmpl, res, method, mask);

    double minVal, maxVal;
    Point minLoc, maxLoc;
    Rect rect;
    minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc);

    if(method == CV_TM_SQDIFF || method == CV_TM_SQDIFF_NORMED)
        rect = Rect(minLoc, tmpl.size());
    else
        rect = Rect(maxLoc, tmpl.size());

    rectangle(img, rect, Scalar(0, 255, 0), 2);

    imshow("detected template", img);
    waitKey();

    return 0;
}
