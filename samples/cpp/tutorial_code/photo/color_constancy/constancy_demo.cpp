#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    Mat src,dst;

    src = imread(argv[1]);

    colorConstancy(src,dst);
    imshow("src",src);
    imshow("Grey-world",dst);
    waitKey(0);
}

