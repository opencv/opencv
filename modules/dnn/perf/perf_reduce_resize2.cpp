// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Reduce2 and Resize2 (the layers of the new graph engine), called directly.

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

typedef tuple<std::vector<int>, Size, std::string, std::string> Resize2Params;
typedef TestBaseWithParam<Resize2Params> Layer_Resize2;

PERF_TEST_P_(Layer_Resize2, resize)
{
    const std::vector<int> shape = get<0>(GetParam());
    const Size dsize = get<1>(GetParam());
    const std::string interp = get<2>(GetParam()), coord = get<3>(GetParam());

    LayerParams lp;
    lp.type = "Resize2";
    lp.name = "testLayer";
    lp.set("interpolation", interp);
    lp.set("coordinate_transformation_mode", coord);
    lp.set("nearest_mode", "floor");
    lp.set("height", dsize.height);
    lp.set("width", dsize.width);
    Ptr<Layer> layer = LayerFactory::createLayerInstance("Resize2", lp);

    Mat input(shape, CV_32F);
    randu(input, 0.f, 1.f);
    std::vector<Mat> inputs{input}, outputs(1), internals;
    layer->forward(inputs, outputs, internals);
    TEST_CYCLE()
    {
        layer->forward(inputs, outputs, internals);
    }
    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_Resize2, testing::Values(
    make_tuple(std::vector<int>{1, 64, 80, 80}, Size(160, 160), "bilinear", "half_pixel"),
    make_tuple(std::vector<int>{1, 3, 224, 224}, Size(448, 448), "bilinear", "half_pixel"),
    make_tuple(std::vector<int>{1, 32, 160, 160}, Size(80, 80), "bilinear", "half_pixel"),
    make_tuple(std::vector<int>{1, 128, 40, 40}, Size(80, 80), "bilinear", "pytorch_half_pixel"),
    make_tuple(std::vector<int>{1, 3, 128, 128}, Size(512, 512), "cubic", "half_pixel"),
    make_tuple(std::vector<int>{1, 64, 80, 80}, Size(160, 160), "bilinear", "align_corners"),
    make_tuple(std::vector<int>{1, 256, 20, 20}, Size(40, 40), "nearest", "asymmetric")
));

} // namespace opencv_test
