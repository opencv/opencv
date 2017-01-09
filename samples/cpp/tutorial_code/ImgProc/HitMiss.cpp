#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

int main(){
    Mat input_image = (Mat_<uchar>(8, 8) <<
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 255, 255, 255, 0, 0, 0, 255,
        0, 255, 255, 255, 0, 0, 0, 0,
        0, 255, 255, 255, 0, 255, 0, 0,
        0, 0, 255, 0, 0, 0, 0, 0,
        0, 0, 255, 0, 0, 255, 255, 0,
        0, 255, 0, 255, 0, 0, 255, 0,
        0, 255, 255, 255, 0, 0, 0, 0);

    Mat kernel = (Mat_<uchar>(3, 3) <<
        0, 1, 0,
        1, -1, 1,
        0, 1, 0);

    Mat output_image;
    morphologyEx(input_image, output_image, MORPH_HITMISS, kernel);

    namedWindow("Original", CV_WINDOW_NORMAL);
    imshow("Original", input_image);
    namedWindow("Hit or Miss", CV_WINDOW_NORMAL);
    imshow("Hit or Miss", output_image);
    waitKey(0);
    return 0;
}