#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "color_constancy.hpp"

using namespace std;
using namespace cv;

void cv::colorConstancy(InputArray _src, OutputArray _dst, float diff_order, int mink_norm, float sigma)
{
    Mat source = _src.getMat();
    _dst.create(source.size(), CV_8UC3);
    Mat dst = _dst.getMat();

    Mat img = Mat(source.size(),CV_32FC3);
    source.convertTo(img,CV_32FC3,1.0);

    float white_R,white_G,white_B;
    Constancy obj;

    Mat output = Mat(img.size(),CV_32FC3);
    obj.general_cc(img,diff_order,mink_norm,sigma,white_R,white_G,white_B,output);

    output.convertTo(dst,CV_8UC3,1);
}

