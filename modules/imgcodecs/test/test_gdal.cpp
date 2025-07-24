// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"
#include "test_common.hpp"

namespace opencv_test { namespace {

#ifdef HAVE_GDAL

static void test_gdal_read(const string filename, bool required = true) {
    const string path = cvtest::findDataFile(filename);
    Mat img;
    ASSERT_NO_THROW(img = imread(path, cv::IMREAD_LOAD_GDAL | cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR));
    if(!required && img.empty())
    {
        throw SkipTestException("GDAL is built wihout required back-end support");
    }
    ASSERT_FALSE(img.empty());
    EXPECT_EQ(3, img.cols);
    EXPECT_EQ(5, img.rows);
    EXPECT_EQ(CV_MAKETYPE(CV_32F, 7), img.type());
    EXPECT_EQ(101.125, (img.at<Vec<float, 7>>(0, 0)[0]));
    EXPECT_EQ(203.500, (img.at<Vec<float, 7>>(2, 1)[3]));
    EXPECT_EQ(305.875, (img.at<Vec<float, 7>>(4, 2)[6]));
}

TEST(Imgcodecs_gdal, read_envi)
{
    test_gdal_read("../cv/gdal/envi_test.raw");
}

TEST(Imgcodecs_gdal, read_fits)
{
    // .fit test is optional because GDAL may be built wihtout CFITSIO library support
    test_gdal_read("../cv/gdal/fits_test.fit", false);
}

#endif // HAVE_GDAL

}} // namespace
