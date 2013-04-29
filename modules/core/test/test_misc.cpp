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


TEST(Core_OutputArraySreate, _1997)
{
    struct local {
        static void create(OutputArray arr, Size submatSize, int type)
        {
            int sizes[] = {submatSize.width, submatSize.height};
            arr.create(sizeof(sizes)/sizeof(sizes[0]), sizes, type);
        }
    };

    Mat mat(Size(512, 512), CV_8U);
    Size submatSize = Size(256, 256);

    ASSERT_NO_THROW(local::create( mat(Rect(Point(), submatSize)), submatSize, mat.type() ));
}

TEST(Core_SaturateCast, NegativeNotClipped)
{
    double d = -1.0;
    unsigned int val = cv::saturate_cast<unsigned int>(d);

    ASSERT_EQ(0xffffffff, val);
}
