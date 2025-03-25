// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#include <vector>

namespace opencv_test
{
namespace
{

using namespace std;
using namespace cv::mcc;

/****************************************************************************************\
 *                Test detection works properly on the simplest images
\****************************************************************************************/

void runCCheckerDetectorBasic(std::string image_name, ColorChart chartType)
{
    Ptr<CCheckerDetector> detector = CCheckerDetector::create();
    std::string path = cvtest::findDataFile("mcc/" + image_name);
    cv::Mat img = imread(path);
    ASSERT_FALSE(img.empty()) << "Test image can't be loaded: " << path;

    detector->setColorChartType(chartType);
    ASSERT_TRUE(detector->process(img));
}
TEST(CV_mccRunCCheckerDetectorBasic, accuracy_SG140)
{
    runCCheckerDetectorBasic("SG140.png", SG140);
}
TEST(CV_mccRunCCheckerDetectorBasic, accuracy_MCC24)
{
    runCCheckerDetectorBasic("MCC24.png", MCC24);
}

TEST(CV_mccRunCCheckerDetectorBasic, accuracy_VINYL18)
{
    runCCheckerDetectorBasic("VINYL18.png", VINYL18);
}

} // namespace
} // namespace opencv_test
