// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include "opencv2/ccm.hpp"

namespace opencv_test
{
namespace
{

using namespace std;
using namespace cv::ccm;

PERF_TEST(CV_mcc_perf, infer) {
    // read gold chartsRGB
    string path = cvtest::findDataFile("cv/mcc/mcc_ccm_test.yml");
    FileStorage fs(path, FileStorage::READ);
    Mat chartsRGB;
    FileNode node = fs["chartsRGB"];
    node >> chartsRGB;
    fs.release();

    // compute CCM
    ColorCorrectionModel model(chartsRGB.col(1).clone().reshape(3, chartsRGB.rows/3) / 255., COLORCHECKER_Macbeth);
    model.run();

    Mat img(1000, 4000, CV_8UC3);
    randu(img, 0, 255);
    img.convertTo(img, CV_64F, 1. / 255.);

    TEST_CYCLE() {
        model.infer(img);
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
} // namespace opencv_test
