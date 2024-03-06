// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef testing::TestWithParam<tuple<Backend, Target> > Test_int64_sum;
TEST_P(Test_int64_sum, basic)
{
    Backend backend = get<0>(GetParam());
    Target target = get<1>(GetParam());

    int64_t a_value = 1000000000000000ll;
    int64_t b_value = 1;
    int64_t result_value = 1000000000000001ll;
    EXPECT_NE(int64_t(float(a_value) + float(b_value)), result_value);

    Mat a(3, 5, CV_64SC1, cv::Scalar_<int64_t>(a_value));
    Mat b = Mat::ones(3, 5, CV_64S);

    Net net;
    LayerParams lp;
    lp.type = "NaryEltwise";
    lp.name = "testLayer";
    lp.set("operation", "sum");
    int id = net.addLayerToPrev(lp.name, lp.type, lp);
    net.connect(0, 1, id, 1);

    vector<String> inpNames(2);
    inpNames[0] = "a";
    inpNames[1] = "b";
    net.setInputsNames(inpNames);
    net.setInput(a, inpNames[0]);
    net.setInput(b, inpNames[1]);

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat re;
    re = net.forward();
    EXPECT_EQ(re.depth(), CV_64S);
    auto ptr_re = (int64_t *) re.data;
    for (int i = 0; i < re.total(); i++)
        ASSERT_EQ(result_value, ptr_re[i]);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Test_int64_sum,
    dnnBackendsAndTargets()
);

}} // namespace
