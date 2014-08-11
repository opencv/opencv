#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    Mat src,dst1,dst2;

    src = imread(argv[1]);

    greyEdge(src,dst1);
    weightedGreyEdge(src,dst2);

    imshow("src",src);
    imshow("Grey-Edge",dst1);
    imshow("Weighted Grey-Edge",dst2);
    waitKey(0);
}

