// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_OP_METAL_HPP
#define OPENCV_DNN_OP_METAL_HPP

#include <opencv2/dnn/dnn.hpp>

#include <memory>

#ifdef HAVE_METAL
#include "metal/metal.hpp"
#endif  // HAVE_METAL

namespace cv { namespace dnn {
#ifdef HAVE_METAL
namespace metal {
class Context;
class Operation;
class Tensor;
}  // namespace metal
#endif

CV__DNN_INLINE_NS_BEGIN

#ifdef HAVE_METAL
class MetalBackendWrapper final : public BackendWrapper
{
public:
    explicit MetalBackendWrapper(Mat& host);
    MetalBackendWrapper(const Ptr<BackendWrapper>& base, Mat& host);

    void copyToHost() CV_OVERRIDE;
    void setHostDirty() CV_OVERRIDE;
    void copyToDevice();
    void setDeviceDirty();

    const std::shared_ptr<metal::Tensor>& tensor() const;

private:
    std::shared_ptr<metal::Tensor> tensor_;
};

class MetalBackendNode final : public BackendNode
{
public:
    explicit MetalBackendNode(std::unique_ptr<metal::Operation> operation);
    ~MetalBackendNode() CV_OVERRIDE;
    void forward();

private:
    std::unique_ptr<metal::Operation> operation_;
};

Ptr<BackendNode> makeMetalBackendNode(std::unique_ptr<metal::Operation> operation);

void forwardMetal(const Ptr<BackendNode>& node);
bool haveMetal();
#else
static inline bool haveMetal()
{
    return false;
}
#endif

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn

#endif  // OPENCV_DNN_OP_METAL_HPP
