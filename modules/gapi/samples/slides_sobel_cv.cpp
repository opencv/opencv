#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>

int main(int argc, char *argv[])
{
    (void) argc;
    (void) argv;

    using namespace cv;
    Mat in_mat = imread("lena.png");
    Mat gx, gy;

    Sobel(in_mat, gx, CV_32F, 1, 0);
    Sobel(in_mat, gy, CV_32F, 0, 1);

    Mat mag;
    sqrt(gx.mul(gx) + gy.mul(gy), mag);

    Mat out_mat;
    mag.convertTo(out_mat, CV_8U);

    imwrite("lena-out.png", out_mat);
    return 0;
}
