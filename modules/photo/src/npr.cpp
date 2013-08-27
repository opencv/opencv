#include "precomp.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdlib.h>

#include "npr.hpp"

using namespace std;
using namespace cv;

void cv::edgepreservefilter(InputArray _src, OutputArray _dst, int flags, float sigma_s, float sigma_r)
{
	Mat I = _src.getMat();
	_dst.create(I.size(), CV_8UC3);
	Mat dst = _dst.getMat();

	int h = I.size().height;
	int w = I.size().width;

	Mat res = Mat(h,w,CV_32FC3);
	dst.convertTo(res,CV_32FC3,1.0/255.0);

	Domain_Filter obj;

	Mat img = Mat(I.size(),CV_32FC3);
	I.convertTo(img,CV_32FC3,1.0/255.0);

	obj.filter(img, res, sigma_s, sigma_r, flags);

	convertScaleAbs(res, dst, 255,0);
}
