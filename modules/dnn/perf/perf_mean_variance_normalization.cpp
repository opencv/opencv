// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test {

typedef tuple<std::vector<int>, std::vector<int> > MeanVarianceNormalizationParams;
typedef TestBaseWithParam<MeanVarianceNormalizationParams> Layer_MeanVarianceNormalization;

PERF_TEST_P_(Layer_MeanVarianceNormalization, forward)
{
    const std::vector<int> shape = get<0>(GetParam());
    const std::vector<int> axes = get<1>(GetParam());

    LayerParams params;
    params.type = "MeanVarianceNormalization";
    params.name = "testLayer";
    if (!axes.empty())
        params.set("axes", DictValue::arrayInt(axes.data(), (int)axes.size()));
    Ptr<Layer> layer = LayerFactory::createLayerInstance(params.type, params);

    Mat input(shape, CV_32F);
    randu(input, -1.f, 1.f);
    std::vector<Mat> inputs{input}, outputs(1), internals;
    layer->forward(inputs, outputs, internals);

    TEST_CYCLE()
    {
        layer->forward(inputs, outputs, internals);
    }
    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_MeanVarianceNormalization, testing::Values(
    make_tuple(std::vector<int>{1, 16, 8, 8}, std::vector<int>{2, 3}),
    make_tuple(std::vector<int>{1, 64, 80, 80}, std::vector<int>{2, 3}),
    make_tuple(std::vector<int>{2, 64, 180, 240}, std::vector<int>()),
    make_tuple(std::vector<int>{8, 196, 768}, std::vector<int>{2}),
    make_tuple(std::vector<int>{8, 64, 32, 32}, std::vector<int>{0, 2}),
    make_tuple(std::vector<int>{1, 64, 80, 80}, std::vector<int>{0, 1, 2, 3})
));

} // namespace opencv_test
