#include "opencv2/photo.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

int main(int, char *argv[])
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
