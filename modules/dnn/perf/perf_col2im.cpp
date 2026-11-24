// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Col2Im (a layer of the new graph engine), called directly.

#include "perf_precomp.hpp"

namespace opencv_test {

typedef tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>> Col2ImParams;
typedef TestBaseWithParam<Col2ImParams> Layer_Col2Im;

PERF_TEST_P_(Layer_Col2Im, col2im)
{
    const std::vector<int> outShape = get<0>(GetParam());
    const std::vector<int> blockShape = get<1>(GetParam());
    const std::vector<int> strides = get<2>(GetParam());
    const std::vector<int> pads = get<3>(GetParam());

    const int D = (int)blockShape.size();
    ASSERT_EQ((int)outShape.size(), D + 2);
    ASSERT_EQ((int)strides.size(), D);
    ASSERT_EQ((int)pads.size(), 2 * D);

    const int N = outShape[0];
    const int C = outShape[1];

    Mat imageShapeMat(std::vector<int>{D}, CV_32S);
    Mat blockShapeMat(std::vector<int>{D}, CV_32S);
    int blockProd = 1;
    int L = 1;
    for (int d = 0; d < D; ++d)
    {
        const int imgDim = outShape[d + 2];
        const int blkDim = blockShape[d];
        imageShapeMat.at<int32_t>(d) = imgDim;
        blockShapeMat.at<int32_t>(d) = blkDim;
        blockProd *= blkDim;
        const int numBlocks = (imgDim + pads[d] + pads[d + D] - blkDim) / strides[d] + 1;
        ASSERT_GT(numBlocks, 0);
        L *= numBlocks;
    }

    LayerParams lp;
    lp.type = "Col2Im";
    lp.name = "testLayer";
    lp.set("strides", DictValue::arrayInt(strides.data(), (int)strides.size()));
    lp.set("pads", DictValue::arrayInt(pads.data(), (int)pads.size()));
    Ptr<Layer> layer = LayerFactory::createLayerInstance("Col2Im", lp);
    ASSERT_TRUE(layer);

    Mat data(std::vector<int>{N, C * blockProd, L}, CV_32F);
    randu(data, -1.f, 1.f);

    std::vector<Mat> inputs{data, imageShapeMat, blockShapeMat}, outputs(1), internals;
    layer->forward(inputs, outputs, internals);
    TEST_CYCLE()
    {
        layer->forward(inputs, outputs, internals);
    }
    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, Layer_Col2Im, testing::Values(
    make_tuple(std::vector<int>{1, 32, 32, 32}, std::vector<int>{3, 3}, std::vector<int>{1, 1}, std::vector<int>{1, 1, 1, 1}),
    make_tuple(std::vector<int>{1, 64, 64, 64}, std::vector<int>{3, 3}, std::vector<int>{1, 1}, std::vector<int>{1, 1, 1, 1}),
    make_tuple(std::vector<int>{1, 128, 64, 64}, std::vector<int>{3, 3}, std::vector<int>{1, 1}, std::vector<int>{0, 0, 0, 0}),
    make_tuple(std::vector<int>{1, 64, 128, 128}, std::vector<int>{3, 3}, std::vector<int>{2, 2}, std::vector<int>{1, 1, 1, 1}),
    make_tuple(std::vector<int>{4, 64, 56, 56}, std::vector<int>{3, 3}, std::vector<int>{1, 1}, std::vector<int>{1, 1, 1, 1}),
    make_tuple(std::vector<int>{1, 32, 8, 16, 16}, std::vector<int>{3, 3, 3}, std::vector<int>{1, 1, 1}, std::vector<int>{1, 1, 1, 1, 1, 1})
));

} // namespace opencv_test
