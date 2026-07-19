// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"

#ifdef HAVE_METAL

#include "batch_norm.hpp"
#include "scale.hpp"

namespace cv { namespace dnn { namespace metal {

Ptr<BackendNode> BatchNormOp::create(
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs,
        const BatchNormConfiguration& config)
{
    AffineConfiguration affineConfig;
    affineConfig.scale = config.scale;
    affineConfig.bias = config.bias;
    affineConfig.axis = config.axis;
    return AffineOp::create(inputs, outputs, affineConfig);
}

}}}  // namespace cv::dnn::metal

#endif  // HAVE_METAL
