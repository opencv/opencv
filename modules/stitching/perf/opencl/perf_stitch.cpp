// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2014, Itseez, Inc, all rights reserved.

#include "perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

using namespace cv;
using namespace perf;
using namespace cvtest::ocl;
using namespace std;
using namespace std::tr1;

#define SURF_MATCH_CONFIDENCE 0.65f
#define ORB_MATCH_CONFIDENCE  0.3f
#define WORK_MEGAPIX 0.6

typedef TestBaseWithParam<string> stitch;

#ifdef HAVE_OPENCV_NONFREE_TODO_FIND_WHY_SURF_IS_NOT_ABLE_TO_STITCH_PANOS
#define TEST_DETECTORS testing::Values("surf", "orb")
#else
#define TEST_DETECTORS testing::Values<string>("orb")
#endif

OCL_PERF_TEST_P(stitch, a123, TEST_DETECTORS)
{
    UMat pano;

    vector<Mat> _imgs;
    _imgs.push_back( imread( getDataPath("stitching/a1.png") ) );
    _imgs.push_back( imread( getDataPath("stitching/a2.png") ) );
    _imgs.push_back( imread( getDataPath("stitching/a3.png") ) );
    vector<UMat> imgs = ToUMat(_imgs);

    Ptr<detail::FeaturesFinder> featuresFinder = GetParam() == "orb"
            ? Ptr<detail::FeaturesFinder>(new detail::OrbFeaturesFinder())
            : Ptr<detail::FeaturesFinder>(new detail::SurfFeaturesFinder());

    Ptr<detail::FeaturesMatcher> featuresMatcher = GetParam() == "orb"
            ? makePtr<detail::BestOf2NearestMatcher>(false, ORB_MATCH_CONFIDENCE)
            : makePtr<detail::BestOf2NearestMatcher>(false, SURF_MATCH_CONFIDENCE);

    declare.iterations(20);

    while(next())
    {
        Stitcher stitcher = Stitcher::createDefault();
        stitcher.setFeaturesFinder(featuresFinder);
        stitcher.setFeaturesMatcher(featuresMatcher);
        stitcher.setWarper(makePtr<SphericalWarper>());
        stitcher.setRegistrationResol(WORK_MEGAPIX);

        startTimer();
        stitcher.stitch(imgs, pano);
        stopTimer();
    }

    EXPECT_NEAR(pano.size().width, 1182, 50);
    EXPECT_NEAR(pano.size().height, 682, 30);

    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST_P(stitch, b12, TEST_DETECTORS)
{
    UMat pano;

    vector<Mat> imgs;
    imgs.push_back( imread( getDataPath("stitching/b1.png") ) );
    imgs.push_back( imread( getDataPath("stitching/b2.png") ) );

    Ptr<detail::FeaturesFinder> featuresFinder = GetParam() == "orb"
            ? Ptr<detail::FeaturesFinder>(new detail::OrbFeaturesFinder())
            : Ptr<detail::FeaturesFinder>(new detail::SurfFeaturesFinder());

    Ptr<detail::FeaturesMatcher> featuresMatcher = GetParam() == "orb"
            ? makePtr<detail::BestOf2NearestMatcher>(false, ORB_MATCH_CONFIDENCE)
            : makePtr<detail::BestOf2NearestMatcher>(false, SURF_MATCH_CONFIDENCE);

    declare.iterations(20);

    while(next())
    {
        Stitcher stitcher = Stitcher::createDefault();
        stitcher.setFeaturesFinder(featuresFinder);
        stitcher.setFeaturesMatcher(featuresMatcher);
        stitcher.setWarper(makePtr<SphericalWarper>());
        stitcher.setRegistrationResol(WORK_MEGAPIX);

        startTimer();
        stitcher.stitch(imgs, pano);
        stopTimer();
    }

    EXPECT_NEAR(pano.size().width, 1124, 50);
    EXPECT_NEAR(pano.size().height, 644, 30);

    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST_P(stitch, boat, TEST_DETECTORS)
{
    UMat pano;

    vector<Mat> _imgs;
    _imgs.push_back( imread( getDataPath("stitching/boat1.jpg") ) );
    _imgs.push_back( imread( getDataPath("stitching/boat2.jpg") ) );
    _imgs.push_back( imread( getDataPath("stitching/boat3.jpg") ) );
    _imgs.push_back( imread( getDataPath("stitching/boat4.jpg") ) );
    _imgs.push_back( imread( getDataPath("stitching/boat5.jpg") ) );
    _imgs.push_back( imread( getDataPath("stitching/boat6.jpg") ) );
    vector<UMat> imgs = ToUMat(_imgs);

    Ptr<detail::FeaturesFinder> featuresFinder = GetParam() == "orb"
            ? Ptr<detail::FeaturesFinder>(new detail::OrbFeaturesFinder())
            : Ptr<detail::FeaturesFinder>(new detail::SurfFeaturesFinder());

    Ptr<detail::FeaturesMatcher> featuresMatcher = GetParam() == "orb"
            ? makePtr<detail::BestOf2NearestMatcher>(false, ORB_MATCH_CONFIDENCE)
            : makePtr<detail::BestOf2NearestMatcher>(false, SURF_MATCH_CONFIDENCE);

    declare.iterations(20);

    while(next())
    {
        Stitcher stitcher = Stitcher::createDefault();
        stitcher.setFeaturesFinder(featuresFinder);
        stitcher.setFeaturesMatcher(featuresMatcher);
        stitcher.setWarper(makePtr<SphericalWarper>());
        stitcher.setRegistrationResol(WORK_MEGAPIX);

        startTimer();
        stitcher.stitch(imgs, pano);
        stopTimer();
    }

    EXPECT_NEAR(pano.size().width, 10789, 200);
    EXPECT_NEAR(pano.size().height, 2663, 100);

    SANITY_CHECK_NOTHING();
}
