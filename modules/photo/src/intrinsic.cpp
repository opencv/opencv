#include <iostream>
//#include <stdio.h>
//#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "intrinsic.hpp"

//using namespace std;
using namespace cv;

void cv::intrinsicDecompose(InputArray _src, OutputArray _ref, OutputArray _shade, int window, int no_of_iter, float rho)
{
     Mat I = _src.getMat();
     _ref.create(I.size(), CV_8UC3);
     Mat ref = _ref.getMat();

     _shade.create(I.size(), CV_8UC1);
     Mat shade = _shade.getMat();

     Mat img = Mat(I.size(),CV_32FC3);
     I.convertTo(img,CV_32FC3,1.0/255.0);

     Intrinsic obj;
     obj.decompose(img,ref,shade,window,no_of_iter,rho);
}
