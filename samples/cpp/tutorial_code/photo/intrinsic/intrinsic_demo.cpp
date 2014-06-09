#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    CV_Assert(argc == 2);
    Mat I,ref,shade;

    I = imread(argv[1]);

    intrinsic_decompose(I,ref,shade);
    
    imshow("reflectance",ref);
    imshow("shading",shade);
    imwrite("ref.jpg",ref);
    imwrite("shade.jpg",shade);
    waitKey(0);
}

