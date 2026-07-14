// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple<Size, int, int> VarRefParams;
typedef TestBaseWithParam<VarRefParams> DenseOpticalFlow_VariationalRefinement;

PERF_TEST_P(DenseOpticalFlow_VariationalRefinement, perf, Combine(Values(szQVGA, szVGA), Values(5, 10), Values(5, 10)))
{
    VarRefParams params = GetParam();
    Size sz = get<0>(params);
    int sorIter = get<1>(params);
    int fixedPointIter = get<2>(params);

    Mat frame1(sz, CV_8U);
    Mat frame2(sz, CV_8U);
    Mat flow(sz, CV_32FC2);

    randu(frame1, 0, 255);
    randu(frame2, 0, 255);
    flow.setTo(0.0f);

    TEST_CYCLE_N(10)
    {
        Ptr<VariationalRefinement> var = VariationalRefinement::create();
        var->setAlpha(20.0f);
        var->setGamma(10.0f);
        var->setDelta(5.0f);
        var->setSorIterations(sorIter);
        var->setFixedPointIterations(fixedPointIter);
        var->calc(frame1, frame2, flow);
    }

    SANITY_CHECK_NOTHING();
}

}} // namespace
