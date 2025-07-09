// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn.hpp>
#include<iostream>

namespace cv { namespace dnn {

class IfLayerImpl CV_FINAL : public IfLayer
{
public:
    explicit IfLayerImpl(const LayerParams& params)
    {
        // Generic layer parameters (no sub-graphs here)
        setParamsFrom(params);
    }

    // Inject pre-parsed sub-graphs (then = 0, else = 1)
    void setSubgraphs(const std::vector<Ptr<Graph>>& graphs) CV_OVERRIDE
    {
        std::cout<<"In the setSubgraphs function"<<std::endl;
        thenelse = graphs;
    }

    // Provide access to the sub-graphs for the Net executor
    std::vector<Ptr<Graph>>* subgraphs() const CV_OVERRIDE { return &thenelse; }

    // For control-flow layers the shapes of the outputs are defined by the
    // executed branch and therefore cannot be known up-front.  Return the
    // requested number of outputs with unspecified (empty) shapes so that
    // the network can be successfully initialized.
    bool getMemoryShapes(const std::vector<MatShape>& /*inputs*/,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        std::cout<<"In the getMemoryShapes function"<<std::endl;
        outputs.assign(std::max(1, requiredOutputs), MatShape());
        internals.clear();
        return false; // shapes are dynamic and will be resolved at runtime
    }

    // Explicitly mark that the layer produces dynamic output shapes.
    bool dynamicOutputShapes() const CV_OVERRIDE { return true; }

private:
    mutable std::vector<Ptr<Graph>> thenelse;
};

Ptr<IfLayer> IfLayer::create(const LayerParams& params)
{
    std::cout<<"In the if create function"<<std::endl;
    return makePtr<IfLayerImpl>(params);
}

}}  // namespace cv::dnn
