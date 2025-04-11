// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include "opencv2/photo.hpp"

namespace opencv_test
{
namespace
{

using namespace std;

PERF_TEST(CV_mcc_perf, correctImage) {
    // read gold chartsRGB
    string path = cvtest::findDataFile("cv/mcc/mcc_ccm_test.yml");
    FileStorage fs(path, FileStorage::READ);
    Mat chartsRGB;
    FileNode node = fs["chartsRGB"];
    node >> chartsRGB;
    ASSERT_FALSE(chartsRGB.empty());
    fs.release();

    // compute CCM
    cv::ccm::ColorCorrectionModel model(chartsRGB.col(1).clone().reshape(3, chartsRGB.rows/3) / 255., cv::ccm::COLORCHECKER_MACBETH);
    Mat colorCorrectionMat = model.compute();

    Mat img(1000, 4000, CV_8UC3);
    randu(img, 0, 255);
    img.convertTo(img, CV_64F, 1. / 255.);

    Mat correctedImage;
    TEST_CYCLE() {
        model.correctImage(img, correctedImage);
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
} // namespace opencv_test
