// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include "opencv2/objdetect/barcode.hpp"

namespace opencv_test{namespace{

typedef ::perf::TestBaseWithParam< tuple<string, cv::Size> > Perf_Barcode_multi;
typedef ::perf::TestBaseWithParam< tuple<string, cv::Size> > Perf_Barcode_single;

PERF_TEST_P_(Perf_Barcode_multi, detect)
{
    const string root = "cv/barcode/multiple/";
    const string name_current_image = get<0>(GetParam());
    const cv::Size sz = get<1>(GetParam());
    const string image_path = findDataFile(root + name_current_image);

    Mat src = imread(image_path);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
    cv::resize(src, src, sz);

    vector< Point > corners;
    auto bardet = barcode::BarcodeDetector();
    bool res = false;
    TEST_CYCLE()
    {
        res = bardet.detectMulti(src, corners);
    }
    SANITY_CHECK_NOTHING();
    ASSERT_TRUE(res);
}

PERF_TEST_P_(Perf_Barcode_multi, detect_decode)
{
    const string root = "cv/barcode/multiple/";
    const string name_current_image = get<0>(GetParam());
    const cv::Size sz = get<1>(GetParam());
    const string image_path = findDataFile(root + name_current_image);

    Mat src = imread(image_path);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
    cv::resize(src, src, sz);

    vector<std::string> decoded_info;
    vector<std::string> decoded_type;
    vector< Point > corners;
    auto bardet = barcode::BarcodeDetector();
    bool res = false;
    TEST_CYCLE()
    {
        res = bardet.detectAndDecodeWithType(src, decoded_info, decoded_type, corners);
    }
    SANITY_CHECK_NOTHING();
    ASSERT_TRUE(res);
}

PERF_TEST_P_(Perf_Barcode_single, detect)
{
    const string root = "cv/barcode/single/";
    const string name_current_image = get<0>(GetParam());
    const cv::Size sz = get<1>(GetParam());
    const string image_path = findDataFile(root + name_current_image);

    Mat src = imread(image_path);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
    cv::resize(src, src, sz);

    vector< Point > corners;
    auto bardet = barcode::BarcodeDetector();
    bool res = false;
    TEST_CYCLE()
    {
        res = bardet.detectMulti(src, corners);
    }
    SANITY_CHECK_NOTHING();
    ASSERT_TRUE(res);
}

PERF_TEST_P_(Perf_Barcode_single, detect_decode)
{
    const string root = "cv/barcode/single/";
    const string name_current_image = get<0>(GetParam());
    const cv::Size sz = get<1>(GetParam());
    const string image_path = findDataFile(root + name_current_image);

    Mat src = imread(image_path);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
    cv::resize(src, src, sz);

    vector<std::string> decoded_info;
    vector<std::string> decoded_type;
    vector< Point > corners;
    auto bardet = barcode::BarcodeDetector();
    bool res = false;
    TEST_CYCLE()
    {
        res = bardet.detectAndDecodeWithType(src, decoded_info, decoded_type, corners);
    }
    SANITY_CHECK_NOTHING();
    ASSERT_TRUE(res);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Perf_Barcode_multi,
    testing::Combine(
        testing::Values("4_barcodes.jpg"),
        testing::Values(cv::Size(2041, 2722), cv::Size(1361, 1815), cv::Size(680, 907))));
INSTANTIATE_TEST_CASE_P(/*nothing*/, Perf_Barcode_single,
    testing::Combine(
        testing::Values("book.jpg", "bottle_1.jpg", "bottle_2.jpg"),
        testing::Values(cv::Size(480, 360), cv::Size(640, 480), cv::Size(800, 600))));

}} //namespace
