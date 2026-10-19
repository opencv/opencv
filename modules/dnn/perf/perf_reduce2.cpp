// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Reduce2 (a layer of the new graph engine), called directly.

#include "perf_precomp.hpp"

namespace opencv_test {

typedef tuple<std::vector<int>, std::vector<int>, std::string, perf::MatType> Reduce2Params;
typedef TestBaseWithParam<Reduce2Params> Layer_Reduce2;

PERF_TEST_P_(Layer_Reduce2, reduce)
{
    const std::vector<int> shape = get<0>(GetParam()), axes = get<1>(GetParam());
    const std::string op = get<2>(GetParam());
    const int type = get<3>(GetParam());

    LayerParams lp;
    lp.type = "Reduce2";
    lp.name = "testLayer";
    lp.set("reduce", op);
    lp.set("keepdims", true);
    lp.set("axes", DictValue::arrayInt(axes.data(), (int)axes.size()));
    Ptr<Layer> layer = LayerFactory::createLayerInstance("Reduce2", lp);

    Mat input(shape, type);
    randu(input, 0, 10);
    std::vector<Mat> inputs{input}, outputs(1), internals;
    layer->forward(inputs, outputs, internals);
    TEST_CYCLE()
    {
        layer->forward(inputs, outputs, internals);
    }
    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_Reduce2, testing::Values(
    make_tuple(std::vector<int>{1, 256, 80, 80}, std::vector<int>{1}, "mean", perf::MatType(CV_32F)),
    make_tuple(std::vector<int>{1, 256, 80, 80}, std::vector<int>{2, 3}, "mean", perf::MatType(CV_32F)),
    make_tuple(std::vector<int>{1, 64, 160, 160}, std::vector<int>{0, 1, 2, 3}, "sum", perf::MatType(CV_32F)),
    make_tuple(std::vector<int>{32, 1024, 256}, std::vector<int>{0}, "max", perf::MatType(CV_32F)),
    make_tuple(std::vector<int>{4096, 768}, std::vector<int>{1}, "l2", perf::MatType(CV_32F)),
    make_tuple(std::vector<int>{64, 1000}, std::vector<int>{1}, "log_sum_exp", perf::MatType(CV_32F)),
    make_tuple(std::vector<int>{1, 256, 80, 80}, std::vector<int>{1}, "sum", perf::MatType(CV_16F)),
    make_tuple(std::vector<int>{1, 3, 640, 640}, std::vector<int>{2, 3}, "max", perf::MatType(CV_8U)),
    make_tuple(std::vector<int>{1, 256, 80, 80}, std::vector<int>{1}, "sum", perf::MatType(CV_32S)),
    make_tuple(std::vector<int>{8, 196, 768}, std::vector<int>{2}, "mean", perf::MatType(CV_32F))
));

} // namespace opencv_test
