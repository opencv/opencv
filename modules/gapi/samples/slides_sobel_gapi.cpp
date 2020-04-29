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
    Mat out_mat;

    GMat in;
    GMat gx  = gapi::Sobel(in, CV_32F, 1, 0);
    GMat gy  = gapi::Sobel(in, CV_32F, 0, 1);
    GMat mag = gapi::sqrt(  gapi::mul(gx, gx)
                          + gapi::mul(gy, gy));
    GMat out = gapi::convertTo(mag, CV_8U);

    GComputation sobel(GIn(in), GOut(out));
    sobel.apply(in_mat, out_mat);

    imwrite("lena-out.png", out_mat);
    return 0;
}
