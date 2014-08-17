#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

void cv::colorTransfer(InputArray _src, InputArray _dst, OutputArray _color_transfer, int flag)
{
    Mat source = _src.getMat();
    Mat target = _dst.getMat();
    _color_transfer.create(source.size(), CV_8UC3);
    Mat output = _color_transfer.getMat();

    Mat img1 = Mat(source.size(),CV_32FC3);
    source.convertTo(img1,CV_32FC3,1.0/255.0);

    Mat img2 = Mat(target.size(),CV_32FC3);
    target.convertTo(img2,CV_32FC3,1.0/255.0);

    if(flag == 0)
    {
        cvtColor(img1,img1,COLOR_BGR2Lab);
        cvtColor(img2,img2,COLOR_BGR2Lab);
    }

    int cols = source.cols;
    Mat source_t = img1.t();
    Mat target_t = img2.t();

    cv::Mat src_reshaped = source_t.reshape ( 3, 1 );
    cv::Mat trg_reshaped = target_t.reshape ( 3, 1 );

    Scalar mean_source,stddev_source;
    meanStdDev(src_reshaped,mean_source,stddev_source);

    Scalar mean_target,stddev_target;
    meanStdDev(trg_reshaped,mean_target,stddev_target);

    vector <Mat> planes;
    split(src_reshaped,planes);
    planes[0] = planes[0] - mean_source[0];
    planes[1] = planes[1] - mean_source[1];
    planes[2] = planes[2] - mean_source[2];

    planes[0] = planes[0]*(stddev_target[0]/stddev_source[0]);
    planes[1] = planes[1]*(stddev_target[1]/stddev_source[1]);
    planes[2] = planes[2]*(stddev_target[2]/stddev_source[2]);

    planes[0] = planes[0] + mean_target[0];
    planes[1] = planes[1] + mean_target[1];
    planes[2] = planes[2] + mean_target[2];

    merge(planes,src_reshaped);
    Mat result = src_reshaped.reshape(3,cols);
    result = result.t();
    if(flag == 0)
        cvtColor(result,result,COLOR_Lab2BGR);

    result.convertTo(output,CV_8UC3,255);
}
