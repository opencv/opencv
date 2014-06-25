#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    Mat src,dst,output;

    src = imread(argv[1]);
    dst = imread(argv[2]);

    colorTransfer(src,dst,output);
    imshow("src",src);
    imshow("target",dst);
    imshow("Output",output);
    waitKey(0);
}

