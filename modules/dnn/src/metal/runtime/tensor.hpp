#ifndef OPENCV_DNN_METAL_RUNTIME_TENSOR_HPP
#define OPENCV_DNN_METAL_RUNTIME_TENSOR_HPP

#include <opencv2/core/mat.hpp>
#include <opencv2/dnn/shape_utils.hpp>

#include <cstddef>
#include <memory>

namespace cv { namespace dnn { namespace metal {

class Buffer;

// Maintains the host and Metal representations of one DNN tensor allocation.
// Reshaped tensors share the allocation and synchronization state while
// keeping their own shape.
class CV_EXPORTS Tensor
{
public:
    static std::shared_ptr<Tensor> create(Mat& host);
    static std::shared_ptr<Tensor> reshape(const std::shared_ptr<Tensor>& base,
                                           const MatShape& shape);

    void copyToDevice();
    void copyToHost();
    void setHostDirty();
    void setDeviceDirty();

    const std::shared_ptr<Buffer>& bufferForRead();
    const std::shared_ptr<Buffer>& bufferForWrite();

    const MatShape& shape() const;
    int type() const;
    size_t total() const;
    size_t byteSize() const;
    bool isHostDirty() const;
    bool isDeviceDirty() const;

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

private:
    struct Storage;

    Tensor(const std::shared_ptr<Storage>& storage, const MatShape& shape);

    std::shared_ptr<Storage> storage_;
    MatShape shape_;
};

}}}  // namespace cv::dnn::metal

#endif  // OPENCV_DNN_METAL_RUNTIME_TENSOR_HPP
