#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "math.h"
#include <vector>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iomanip>

using namespace std;
using namespace cv;



int main(int argc, char **argv)
{
     Mat source = imread(argv[1]);

     Mat img = Mat(source.size(),CV_32FC3);
     source.convertTo(img,CV_32FC3,1.0/255.0);

     Mat ref = Mat(img.size(),CV_32FC3);
     Mat shade = Mat(img.size(),CV_32FC3);
     automatic(img,ref,shade,3,100,1.9);

     return 0;
}

