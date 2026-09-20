// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
#include <opencv2/dnn/all_layers.hpp>

namespace opencv_test {

typedef TestBaseWithParam<tuple<int, int> > DeconvolutionCoordinates;

PERF_TEST_P(DeconvolutionCoordinates, forward,
            testing::Combine(testing::Values(2, 3), testing::Values(1, 2, 3)))
{
    const int dims = get<0>(GetParam());
    const int kernelSize = get<1>(GetParam());
    std::vector<int> inputShape = {1, 16};
    for (int d = 0; d < dims; ++d)
        inputShape.push_back(dims == 2 ? 128 : 16);
    std::vector<int> weightShape = {16, 16};
    weightShape.resize(dims + 2, kernelSize);
    std::vector<int> kernel(dims, kernelSize), stride(dims, kernelSize == 2 ? 2 : 1);
    LayerParams lp;
    lp.set("kernel_size", DictValue::arrayInt(kernel.data(), dims));
    lp.set("stride", DictValue::arrayInt(stride.data(), dims));
    lp.set("num_output", 16);
    lp.set("bias_term", true);
    Mat input(inputShape, CV_32F), weights(weightShape, CV_32F), bias(1, 16, CV_32F);
    randu(input, -1.0f, 1.0f);
    randu(weights, -1.0f, 1.0f);
    randu(bias, -1.0f, 1.0f);
    lp.blobs = {weights, bias};
    Ptr<Layer> layer = DeconvolutionLayer::create(lp);
    std::vector<MatShape> outputShapes, internalShapes;
    layer->getMemoryShapes({input.shape()}, 0, outputShapes, internalShapes);
    std::vector<Mat> inputs(1, input), outputs, internals;
    for (const MatShape& shape : outputShapes)
        outputs.push_back(Mat(shape, CV_32F));
    for (const MatShape& shape : internalShapes)
        internals.push_back(Mat(shape, CV_32F));
    layer->finalize(inputs, outputs);
    layer->forward(inputs, outputs, internals);

    TEST_CYCLE()
    {
        layer->forward(inputs, outputs, internals);
    }
    SANITY_CHECK_NOTHING();
}

} // namespace
