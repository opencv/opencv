#include "opencv2/photo.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

int main(int, char *argv[])
{
    Mat I,ref,shade;

    I = imread(argv[1]);

    intrinsicDecompose(I,ref,shade);

    imshow("reflectance",ref);
    imshow("shading",shade);
    waitKey(0);
}
