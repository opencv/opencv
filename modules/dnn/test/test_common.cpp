// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "test_common.impl.hpp"  // shared with perf tests
#include <opencv2/dnn/shape_utils.hpp>

namespace opencv_test {
void runLayer(cv::Ptr<cv::dnn::Layer> layer, std::vector<cv::Mat> &inpBlobs, std::vector<cv::Mat> &outBlobs)
{
    size_t ninputs = inpBlobs.size();
    std::vector<cv::Mat> inp(ninputs), outp, intp;
    std::vector<cv::dnn::MatShape> inputs, outputs, internals;
    std::vector<cv::dnn::MatType> inputs_types, outputs_types, internals_types;

    for (size_t i = 0; i < ninputs; i++)
    {
        inp[i] = inpBlobs[i].clone();
        inputs.push_back(cv::dnn::shape(inp[i]));
        inputs_types.push_back(cv::dnn::MatType(inp[i].type()));
    }

    layer->getMemoryShapes(inputs, 0, outputs, internals);
    layer->getTypes(inputs_types, outputs.size(), internals.size(), outputs_types, internals_types);
    for (size_t i = 0; i < outputs.size(); i++)
    {
        outp.push_back(cv::Mat(outputs[i], outputs_types[i]));
    }
    for (size_t i = 0; i < internals.size(); i++)
    {
        intp.push_back(cv::Mat(internals[i], internals_types[i]));
    }

    layer->finalize(inp, outp);
    layer->forward(inp, outp, intp);

    size_t noutputs = outp.size();
    outBlobs.resize(noutputs);
    for (size_t i = 0; i < noutputs; i++)
        outBlobs[i] = outp[i];
}

}
