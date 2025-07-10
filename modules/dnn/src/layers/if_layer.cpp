// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn.hpp>

namespace cv { namespace dnn {

class IfLayerImpl CV_FINAL : public IfLayer
{
public:
    explicit IfLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
    }

    std::vector<Ptr<Graph>>* subgraphs() const CV_OVERRIDE { return &thenelse; }

    bool getMemoryShapes(const std::vector<MatShape>& /*inputs*/,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        outputs.assign(std::max(1, requiredOutputs), MatShape());
        internals.clear();
        return false;
    }

    bool dynamicOutputShapes() const CV_OVERRIDE { return true; }

private:
    mutable std::vector<Ptr<Graph>> thenelse;
};

Ptr<IfLayer> IfLayer::create(const LayerParams& params)
{
    return makePtr<IfLayerImpl>(params);
}

}}  // namespace cv::dnn
