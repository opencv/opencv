// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

using namespace std;

namespace opencv_test{namespace{

TEST(ColorNames, test)
{
    const std::string fname = findDataFile("objdetect/color_names.yaml.gz");
    ASSERT_FALSE(fname.empty());
    Ptr<ColorNamesFeatures> cnames = ColorNamesFeatures::create(fname);
    ASSERT_TRUE(cnames);
    const Size SZ(100, 100);
    Mat img = cvtest::randomMat(theRNG(), SZ, CV_8UC3, 0, 255, false);
    img.at<ColorNamesFeatures::PixType>(99, 99) = {255, 255, 255};
    Mat features;
    cnames->compute(img, features);
    ASSERT_EQ(features.type(), CV_32FC(10));
    ASSERT_EQ(features.size(), SZ);
    const float last_item = features.at<ColorNamesFeatures::FeatureType>(99, 99)[0];
    ASSERT_NEAR(last_item, 0.0087778, 0.00001);
}

}} // opencv_test::<anonymous>::
