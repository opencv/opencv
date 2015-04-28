// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using namespace testing;
using std::tr1::make_tuple;
using std::tr1::get;

typedef std::tr1::tuple<Size, MatDepth, bool> MomentsParams_t;
typedef perf::TestBaseWithParam<MomentsParams_t> MomentsFixture_val;

PERF_TEST_P(MomentsFixture_val, Moments1,
    ::testing::Combine(
    testing::Values(TYPICAL_MAT_SIZES),
    testing::Values(CV_16U, CV_16S, CV_32F, CV_64F),
    testing::Bool()))
{
    const MomentsParams_t params = GetParam();
    const Size srcSize = get<0>(params);
    const MatDepth srcDepth = get<1>(params);
    const bool binaryImage = get<2>(params);

    cv::Moments m;
    Mat src(srcSize, srcDepth);
    declare.in(src, WARMUP_RNG);

    TEST_CYCLE() m = cv::moments(src, binaryImage);

    int len = (int)sizeof(cv::Moments) / sizeof(double);
    cv::Mat mat(1, len, CV_64F, (void*)&m);
    //adding 1 to moments to avoid accidental tests fail on values close to 0
    mat += 1;


    SANITY_CHECK_MOMENTS(m, 2e-4, ERROR_RELATIVE);
}
