#include "precomp.hpp"
#include "_lsvm_resizeimg.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

IplImage* resize_opencv(IplImage* img, float scale)
{
    IplImage* imgTmp;

    int W, H, tW, tH;

    W = img->width;
    H = img->height;

    tW = (int)(((float)W) * scale + 0.5);
    tH = (int)(((float)H) * scale + 0.5);

    imgTmp = cvCreateImage(cvSize(tW , tH), img->depth, img->nChannels);
    cvResize(img, imgTmp, CV_INTER_AREA);

    return imgTmp;
}