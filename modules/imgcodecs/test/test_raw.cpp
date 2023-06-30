// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"

namespace opencv_test { namespace {

#ifdef HAVE_RAW

TEST(Imgcodecs_Raw, decode_raw)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    string filename2 = root + "../cv/rawfiles/RAW_KODAK_DC50.KDC";
    cv::Mat img2 = cv::imreadraw(filename2);
    ASSERT_FALSE(img2.empty());
    EXPECT_EQ(512, img2.rows);
    EXPECT_EQ(768, img2.cols);
}


#endif // HAVE_RAW

}} // namespace
