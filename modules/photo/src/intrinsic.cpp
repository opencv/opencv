#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include <vector>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iomanip>

#include "intrinsic.hpp"

using namespace std;
using namespace cv;

void intrinsic(InputArray _src, OutputArray _dst, int window, int no_of_iter, float rho)
{
     Mat I = _src.getMat();
     _dst.create(I.size(), CV_8UC1);
     Mat dst = _dst.getMat();


     Mat img = Mat(I.size(),CV_32FC3);
     I.convertTo(img,CV_32FC3,1.0/255.0);

     Mat ref = Mat(img.size(),CV_32FC3);
     Mat shade = Mat(img.size(),CV_32FC3);

     Intrinsic obj;
     obj.decompose(img,ref,shade,window,no_of_iter,rho);
}

