// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"

namespace opencv_test {

static std::vector<int> broadcastShape(const std::vector<std::vector<int> >& shapes)
{
    size_t dims = 0;
    for (const auto& s : shapes)
        dims = std::max(dims, s.size());
    std::vector<int> out(dims, 1);
    for (const auto& s : shapes)
        for (size_t i = 0; i < s.size(); i++)
            out[dims - s.size() + i] = std::max(out[dims - s.size() + i], s[i]);
    return out;
}

static Mat randomInput(const std::vector<int>& shape, int type)
{
    Mat m(shape, type);
    if (type == CV_Bool)
        randu(Mat(1, (int)m.total(), CV_8U, m.data), 0, 2);
    else
        randu(m, type == CV_32F ? -1. : 0., type == CV_32F ? 1. : 1000.);
    return m;
}

typedef tuple<std::string, std::vector<std::vector<int> >, perf::MatType> NaryParams;
typedef TestBaseWithParam<NaryParams> Layer_NaryEltwiseCore;

PERF_TEST_P_(Layer_NaryEltwiseCore, forward)
{
    const std::string op = get<0>(GetParam());
    const std::vector<std::vector<int> > shapes = get<1>(GetParam());
    const int type = get<2>(GetParam());

    LayerParams lp;
    lp.type = "NaryEltwise";
    lp.name = "testLayer";
    lp.set("operation", op);
    Ptr<Layer> layer = LayerFactory::createLayerInstance("NaryEltwise", lp);

    std::vector<Mat> inputs, internals;
    for (size_t i = 0; i < shapes.size(); i++)
        inputs.push_back(randomInput(shapes[i], op == "where" && i == 0 ? (int)CV_Bool : type));
    if (op == "pow")
        inputs[0] = cv::abs(inputs[0]) + 0.5;
    const bool boolOut = op == "greater" || op == "and";
    std::vector<Mat> outputs{Mat(broadcastShape(shapes), boolOut ? (int)CV_Bool : type)};
    layer->finalize(inputs, outputs);
    layer->forward(inputs, outputs, internals);

    TEST_CYCLE()
    {
        layer->forward(inputs, outputs, internals);
    }
    SANITY_CHECK_NOTHING();
}

typedef std::vector<std::vector<int> > Shapes;
INSTANTIATE_TEST_CASE_P(/**/, Layer_NaryEltwiseCore, testing::Values(
    make_tuple("add", Shapes{{1, 64, 80, 80}, {1, 64, 1, 1}}, perf::MatType(CV_32F)),
    make_tuple("mul", Shapes{{1, 64, 80, 80}, {1, 1, 80, 80}}, perf::MatType(CV_32F)),
    make_tuple("add", Shapes{{8, 196, 768}, {768}}, perf::MatType(CV_32F)),
    make_tuple("div", Shapes{{8, 196, 768}, {8, 196, 1}}, perf::MatType(CV_32F)),
    make_tuple("pow", Shapes{{1, 64, 80, 80}, {1, 64, 80, 80}}, perf::MatType(CV_32F)),
    make_tuple("sum", Shapes{{1, 64, 80, 80}, {1, 64, 80, 80}, {1, 64, 80, 80}}, perf::MatType(CV_32F)),
    make_tuple("mean", Shapes{{1, 64, 80, 80}, {1, 64, 80, 80}, {1, 64, 80, 80}}, perf::MatType(CV_32F)),
    make_tuple("max", Shapes{{1, 64, 80, 80}, {1, 64, 80, 80}}, perf::MatType(CV_32S)),
    make_tuple("bitwise_and", Shapes{{1, 64, 80, 80}, {1, 64, 80, 80}}, perf::MatType(CV_32S)),
    make_tuple("and", Shapes{{1, 64, 80, 80}, {1, 64, 80, 80}}, perf::MatType(CV_Bool)),
    make_tuple("greater", Shapes{{1, 64, 80, 80}, {1, 64, 1, 1}}, perf::MatType(CV_32F)),
    make_tuple("where", Shapes{{1, 64, 80, 80}, {1, 64, 80, 80}, {1, 64, 80, 80}}, perf::MatType(CV_32F))
));

typedef tuple<std::string, std::string> ActivationParams;
typedef TestBaseWithParam<ActivationParams> Layer_ActivationCore;

PERF_TEST_P_(Layer_ActivationCore, forward)
{
    const std::string type = get<0>(GetParam()), variant = get<1>(GetParam());
    LayerParams lp;
    lp.type = type;
    lp.name = "testLayer";
    if (type == "Power")
    {
        lp.set("power", 2.f);
        lp.set("scale", 0.5f);
        lp.set("shift", 1.f);
    }
    if (type == "Exp" && variant == "base2")
        lp.set("base", 2.f);
    Ptr<Layer> layer = LayerFactory::createLayerInstance(type, lp);

    Mat input({1, 64, 80, 80}, CV_32F);
    randu(input, type == "Sqrt" || type == "Reciprocal" ? 0.1 : -3., 3.);
    std::vector<Mat> inputs{input}, outputs{Mat(input.size, CV_32F)}, internals;
    layer->forward(inputs, outputs, internals);

    TEST_CYCLE()
    {
        layer->forward(inputs, outputs, internals);
    }
    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_ActivationCore, testing::Values(
    make_tuple("Sqrt", ""), make_tuple("AbsVal", ""), make_tuple("Reciprocal", ""),
    make_tuple("Power", ""), make_tuple("ReLU", ""), make_tuple("TanH", ""),
    make_tuple("Exp", ""), make_tuple("Exp", "base2")
));

} // namespace opencv_test
