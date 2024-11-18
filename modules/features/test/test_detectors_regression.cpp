// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace opencv_test { namespace {
const string FEATURES2D_DIR = "features2d";
const string IMAGE_FILENAME = "tsukuba.png";
const string DETECTOR_DIR = FEATURES2D_DIR + "/feature_detectors";
}} // namespace

#include "test_detectors_regression.impl.hpp"

namespace opencv_test { namespace {

/****************************************************************************************\
*                                Tests registrations                                     *
\****************************************************************************************/

TEST( Features2d_Detector_SIFT, regression )
{
    CV_FeatureDetectorTest test( "detector-sift", SIFT::create() );
    test.safe_run();
}

TEST( Features2d_Detector_FAST, regression )
{
    CV_FeatureDetectorTest test( "detector-fast", FastFeatureDetector::create() );
    test.safe_run();
}

TEST( Features2d_Detector_GFTT, regression )
{
    CV_FeatureDetectorTest test( "detector-gftt", GFTTDetector::create() );
    test.safe_run();
}

TEST( Features2d_Detector_Harris, regression )
{
    Ptr<GFTTDetector> gftt = GFTTDetector::create();
    gftt->setHarrisDetector(true);
    CV_FeatureDetectorTest test( "detector-harris", gftt);
    test.safe_run();
}

TEST( Features2d_Detector_MSER, DISABLED_regression )
{
    CV_FeatureDetectorTest test( "detector-mser", MSER::create() );
    test.safe_run();
}

TEST( Features2d_Detector_ORB, regression )
{
    CV_FeatureDetectorTest test( "detector-orb", ORB::create() );
    test.safe_run();
}

}} // namespace
