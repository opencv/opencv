// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include "opencv2/photo.hpp"

namespace opencv_test {
namespace {

using namespace cv;
using namespace std;

PERF_TEST(CV_ccm_perf_480_640, correctImage)
{
    string path = cvtest::findDataFile("cv/mcc/mcc_ccm_test.yml");
    FileStorage fs(path, FileStorage::READ);
    Mat chartsRGB;
    fs["chartsRGB"] >> chartsRGB;
    fs.release();

    cv::ccm::ColorCorrectionModel model(
        chartsRGB.col(1).clone().reshape(3, chartsRGB.rows/3) / 255.0,
        cv::ccm::COLORCHECKER_MACBETH
    );
    model.compute();
    Mat img(480, 640, CV_8UC3);
    randu(img, 0, 255);
    img.convertTo(img, CV_64F, 1.0/255.0);

    Mat correctedImage;
    TEST_CYCLE() { model.correctImage(img, correctedImage); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST(CV_ccm_perf_720_1280, correctImage)
{
    string path = cvtest::findDataFile("cv/mcc/mcc_ccm_test.yml");
    FileStorage fs(path, FileStorage::READ);
    Mat chartsRGB;
    fs["chartsRGB"] >> chartsRGB;
    fs.release();

    cv::ccm::ColorCorrectionModel model(
        chartsRGB.col(1).clone().reshape(3, chartsRGB.rows/3) / 255.0,
        cv::ccm::COLORCHECKER_MACBETH
    );
    model.compute();
    Mat img(720, 1280, CV_8UC3);
    randu(img, 0, 255);
    img.convertTo(img, CV_64F, 1.0/255.0);

    Mat correctedImage;
    TEST_CYCLE() { model.correctImage(img, correctedImage); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST(CV_ccm_perf_1080_1920, correctImage)
{
    string path = cvtest::findDataFile("cv/mcc/mcc_ccm_test.yml");
    FileStorage fs(path, FileStorage::READ);
    Mat chartsRGB;
    fs["chartsRGB"] >> chartsRGB;
    fs.release();

    cv::ccm::ColorCorrectionModel model(
        chartsRGB.col(1).clone().reshape(3, chartsRGB.rows/3) / 255.0,
        cv::ccm::COLORCHECKER_MACBETH
    );
    model.compute();
    Mat img(1080, 1920, CV_8UC3);
    randu(img, 0, 255);
    img.convertTo(img, CV_64F, 1.0/255.0);

    Mat correctedImage;
    TEST_CYCLE() { model.correctImage(img, correctedImage); }
    SANITY_CHECK_NOTHING();
}

PERF_TEST(CV_ccm_perf_2160_3840, correctImage)
{
    string path = cvtest::findDataFile("cv/mcc/mcc_ccm_test.yml");
    FileStorage fs(path, FileStorage::READ);
    Mat chartsRGB;
    fs["chartsRGB"] >> chartsRGB;
    fs.release();

    cv::ccm::ColorCorrectionModel model(
        chartsRGB.col(1).clone().reshape(3, chartsRGB.rows/3) / 255.0,
        cv::ccm::COLORCHECKER_MACBETH
    );
    model.compute();
    Mat img(2160, 3840, CV_8UC3);
    randu(img, 0, 255);
    img.convertTo(img, CV_64F, 1.0/255.0);

    Mat correctedImage;
    TEST_CYCLE() { model.correctImage(img, correctedImage); }
    SANITY_CHECK_NOTHING();
}

} // namespace
} // namespace opencv_test
