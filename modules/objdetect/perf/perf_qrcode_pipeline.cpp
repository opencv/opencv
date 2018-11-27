// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

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
    SANITY_CHECK(corners);
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

    std::vector<uint8_t> decoded_info_uint8_t(decoded_info.begin(), decoded_info.end());
    SANITY_CHECK(decoded_info_uint8_t);
    SANITY_CHECK(straight_barcode);

}
#endif

INSTANTIATE_TEST_CASE_P(/*nothing*/, Perf_Objdetect_QRCode,
    ::testing::Values(
        "version_1_down.jpg", "version_1_left.jpg", "version_1_right.jpg", "version_1_up.jpg", "version_1_top.jpg",
        "version_5_down.jpg", "version_5_left.jpg", "version_5_right.jpg", "version_5_up.jpg", "version_5_top.jpg",
        "russian.jpg", "kanji.jpg", "link_github_ocv.jpg", "link_ocv.jpg", "link_wiki_cv.jpg"
    )
);

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
