// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include "../test/test_qr_utils.hpp"

namespace opencv_test
{
namespace
{

typedef ::perf::TestBaseWithParam< std::string > Perf_Objdetect_QRCode;

PERF_TEST_P_(Perf_Objdetect_QRCode, detect)
{
    const std::string name_current_image = GetParam();
    const std::string root = "cv/qrcode/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    std::vector< Point > corners;
    QRCodeDetector qrcode;
    TEST_CYCLE() ASSERT_TRUE(qrcode.detect(src, corners));
    const int pixels_error = 3;
    check_qr(root, name_current_image, "test_images", corners, {}, pixels_error);
    SANITY_CHECK_NOTHING();
}

#ifdef HAVE_QUIRC
PERF_TEST_P_(Perf_Objdetect_QRCode, decode)
{
    const std::string name_current_image = GetParam();
    const std::string root = "cv/qrcode/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    std::vector< Point > corners;
    std::string decoded_info;
    QRCodeDetector qrcode;
    ASSERT_TRUE(qrcode.detect(src, corners));
    TEST_CYCLE()
    {
        decoded_info = qrcode.decode(src, corners, straight_barcode);
        ASSERT_FALSE(decoded_info.empty());
    }
    const int pixels_error = 3;
    check_qr(root, name_current_image, "test_images", corners, {decoded_info}, pixels_error);
    SANITY_CHECK_NOTHING();
}
#endif

typedef ::perf::TestBaseWithParam<std::tuple<std::string, std::string>> Perf_Objdetect_QRCode_Multi;

static std::set<std::pair<std::string, std::string>> disabled_samples = {{"5_qrcodes.png", "aruco_based"}};

PERF_TEST_P_(Perf_Objdetect_QRCode_Multi, detectMulti)
{
    const std::string name_current_image = get<0>(GetParam());
    const std::string method = get<1>(GetParam());
    const std::string root = "cv/qrcode/multiple/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
    std::vector<Point> corners;
    GraphicalCodeDetector qrcode = QRCodeDetector();
    if (method == "aruco_based") {
        qrcode = QRCodeDetectorAruco();
    }
    TEST_CYCLE() ASSERT_TRUE(qrcode.detectMulti(src, corners));
    const int pixels_error = 7;
    check_qr(root, name_current_image, "multiple_images", corners, {}, pixels_error, true);
    SANITY_CHECK_NOTHING();
}

#ifdef HAVE_QUIRC
PERF_TEST_P_(Perf_Objdetect_QRCode_Multi, decodeMulti)
{
    const std::string name_current_image = get<0>(GetParam());
    std::string method = get<1>(GetParam());
    const std::string root = "cv/qrcode/multiple/";
    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
    if (disabled_samples.find({name_current_image, method}) != disabled_samples.end()) {
        throw SkipTestException(name_current_image + " is disabled sample for method " + method);
    }
    GraphicalCodeDetector qrcode = QRCodeDetector();
    if (method == "aruco_based") {
        qrcode = QRCodeDetectorAruco();
    }
    std::vector<Point2f> corners;
    ASSERT_TRUE(qrcode.detectMulti(src, corners));
    std::vector<Mat> straight_barcode;
    std::vector< cv::String > decoded_info;
    TEST_CYCLE()
    {
        ASSERT_TRUE(qrcode.decodeMulti(src, corners, decoded_info, straight_barcode));
    }
    ASSERT_TRUE(decoded_info.size() > 0ull);
    for(size_t i = 0; i < decoded_info.size(); i++) {
        ASSERT_FALSE(decoded_info[i].empty());
    }
    ASSERT_EQ(decoded_info.size(), straight_barcode.size());
    vector<Point> corners_result(corners.size());
    for (size_t i = 0ull; i < corners_result.size(); i++) {
        corners_result[i] = corners[i];
    }

    const int pixels_error = 7;
    check_qr(root, name_current_image, "multiple_images", corners_result, decoded_info, pixels_error, true);
    SANITY_CHECK_NOTHING();
}
#endif

INSTANTIATE_TEST_CASE_P(/*nothing*/, Perf_Objdetect_QRCode,
    ::testing::Values(
        "version_1_down.jpg", "version_1_left.jpg", "version_1_right.jpg", "version_1_up.jpg", "version_1_top.jpg",
        "version_5_down.jpg", "version_5_left.jpg",/*version_5_right.jpg*/ "version_5_up.jpg", "version_5_top.jpg",
        "russian.jpg", "kanji.jpg", "link_github_ocv.jpg", "link_ocv.jpg", "link_wiki_cv.jpg"
    )
);
// version_5_right.jpg DISABLED after tile fix, PR #22025

INSTANTIATE_TEST_CASE_P(/*nothing*/, Perf_Objdetect_QRCode_Multi,
    testing::Combine(testing::Values("2_qrcodes.png", "3_close_qrcodes.png", "3_qrcodes.png", "4_qrcodes.png",
                                     "5_qrcodes.png", "6_qrcodes.png", "7_qrcodes.png", "8_close_qrcodes.png"),
                     testing::Values("contours_based", "aruco_based")));


typedef ::perf::TestBaseWithParam< tuple< std::string, Size > > Perf_Objdetect_Not_QRCode;

PERF_TEST_P_(Perf_Objdetect_Not_QRCode, detect)
{
    std::vector<Point> corners;
    std::string type_gen = get<0>(GetParam());
    Size resolution = get<1>(GetParam());
    Mat not_qr_code(resolution, CV_8UC1, Scalar(0));
    if (type_gen == "random")
    {
        RNG rng;
        rng.fill(not_qr_code, RNG::UNIFORM, Scalar(0), Scalar(1));
    }
    if (type_gen == "chessboard")
    {
        uint8_t next_pixel = 0;
        for (int r = 0; r < not_qr_code.rows * not_qr_code.cols; r++)
        {
            int i = r / not_qr_code.cols;
            int j = r % not_qr_code.cols;
            not_qr_code.ptr<uchar>(i)[j] = next_pixel;
            next_pixel = 255 - next_pixel;
        }
    }

    QRCodeDetector qrcode;
    TEST_CYCLE() ASSERT_FALSE(qrcode.detect(not_qr_code, corners));
    SANITY_CHECK_NOTHING();
}

#ifdef HAVE_QUIRC
PERF_TEST_P_(Perf_Objdetect_Not_QRCode, decode)
{
    Mat straight_barcode;
    std::vector< Point > corners;
    corners.push_back(Point( 0, 0)); corners.push_back(Point( 0,  5));
    corners.push_back(Point(10, 0)); corners.push_back(Point(15, 15));

    std::string type_gen = get<0>(GetParam());
    Size resolution = get<1>(GetParam());
    Mat not_qr_code(resolution, CV_8UC1, Scalar(0));
    if (type_gen == "random")
    {
        RNG rng;
        rng.fill(not_qr_code, RNG::UNIFORM, Scalar(0), Scalar(1));
    }
    if (type_gen == "chessboard")
    {
        uint8_t next_pixel = 0;
        for (int r = 0; r < not_qr_code.rows * not_qr_code.cols; r++)
        {
            int i = r / not_qr_code.cols;
            int j = r % not_qr_code.cols;
            not_qr_code.ptr<uchar>(i)[j] = next_pixel;
            next_pixel = 255 - next_pixel;
        }
    }

    QRCodeDetector qrcode;
    TEST_CYCLE() ASSERT_TRUE(qrcode.decode(not_qr_code, corners, straight_barcode).empty());
    SANITY_CHECK_NOTHING();
}
#endif

INSTANTIATE_TEST_CASE_P(/*nothing*/, Perf_Objdetect_Not_QRCode,
      ::testing::Combine(
            ::testing::Values("zero", "random", "chessboard"),
            ::testing::Values(Size(640, 480),   Size(1280, 720),
                              Size(1920, 1080), Size(3840, 2160))
      ));

}
} // namespace
