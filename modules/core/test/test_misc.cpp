#include "test_precomp.hpp"

using namespace cv;
using namespace std;

TEST(Core_Drawing, _914)
{
    const int rows = 256;
    const int cols = 256;

    Mat img(rows, cols, CV_8UC1, Scalar(255));

    line(img, Point(0, 10), Point(255, 10), Scalar(0), 2, 4);
    line(img, Point(-5, 20), Point(260, 20), Scalar(0), 2, 4);
    line(img, Point(10, 0), Point(10, 255), Scalar(0), 2, 4);

    double x0 = 0.0/pow(2.0, -2.0);
    double x1 = 255.0/pow(2.0, -2.0);
    double y = 30.5/pow(2.0, -2.0);

    line(img, Point(int(x0), int(y)), Point(int(x1), int(y)), Scalar(0), 2, 4, 2);

    int pixelsDrawn = rows*cols - countNonZero(img);
    ASSERT_EQ( (3*rows + cols)*3 - 3*9, pixelsDrawn);
}