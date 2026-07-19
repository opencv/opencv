#include "../../precomp.hpp"
#include "tensor.hpp"

#include "buffer.hpp"
#include "context.hpp"

#include <limits>
#include <mutex>

namespace cv { namespace dnn { namespace metal {

struct Tensor::Storage
{
    std::shared_ptr<Buffer> buffer;
    Mat host;
    mutable std::mutex mutex;
    bool hostDirty = true;
    bool deviceDirty = false;
};

static size_t shapeTotal(const MatShape& shape)
{
    CV_CheckFalse(shape.empty(), "Metal tensor shape must not be empty");

    size_t result = 1;
    for (size_t i = 0; i < shape.size(); ++i)
    {
        CV_CheckGT(shape[i], 0, "Metal tensor dimensions must be positive");
        CV_CheckLE(result, std::numeric_limits<size_t>::max() / static_cast<size_t>(shape[i]),
                   "Metal tensor element count overflows size_t");
        result *= static_cast<size_t>(shape[i]);
    }
    return result;
}

Tensor::Tensor(const std::shared_ptr<Storage>& storage,
               const MatShape& shape)
    : storage_(storage), shape_(shape)
{
}

std::shared_ptr<Tensor> Tensor::create(Mat& host)
{
    CV_CheckFalse(host.empty(), "Cannot create a Metal tensor from an empty Mat");
    CV_CheckTrue(host.isContinuous(), "Metal tensors require continuous Mat storage");

    std::shared_ptr<Storage> storage = std::make_shared<Storage>();
    storage->host = host;
    storage->buffer = Buffer::create(host.total() * host.elemSize());

    return std::shared_ptr<Tensor>(new Tensor(storage, cv::dnn::shape(host)));
}

std::shared_ptr<Tensor> Tensor::reshape(const std::shared_ptr<Tensor>& base,
                                       const MatShape& shape)
{
    CV_CheckTrue(base != nullptr, "Base Metal tensor must not be null");
    const size_t elementSize = base->storage_->host.elemSize();
    const size_t capacity = base->storage_->buffer->size() / elementSize;
    CV_CheckLE(shapeTotal(shape), capacity,
               "Metal tensor view exceeds the underlying allocation");
    return std::shared_ptr<Tensor>(new Tensor(base->storage_, shape));
}

void Tensor::copyToDevice()
{
    std::lock_guard<std::mutex> lock(storage_->mutex);
    if (!storage_->hostDirty)
        return;

    Context::get().upload(storage_->buffer, storage_->host.data,
                          storage_->host.total() * storage_->host.elemSize());
    storage_->hostDirty = false;
    storage_->deviceDirty = false;
}

void Tensor::copyToHost()
{
    std::lock_guard<std::mutex> lock(storage_->mutex);
    if (!storage_->deviceDirty)
        return;
    Context::get().download(storage_->buffer, storage_->host.data,
                            storage_->host.total() * storage_->host.elemSize());
    storage_->hostDirty = false;
    storage_->deviceDirty = false;
}

void Tensor::setHostDirty()
{
    std::lock_guard<std::mutex> lock(storage_->mutex);
    storage_->hostDirty = true;
    storage_->deviceDirty = false;
}

void Tensor::setDeviceDirty()
{
    std::lock_guard<std::mutex> lock(storage_->mutex);
    storage_->hostDirty = false;
    storage_->deviceDirty = true;
}

const std::shared_ptr<Buffer>& Tensor::bufferForRead()
{
    copyToDevice();
    return storage_->buffer;
}

const std::shared_ptr<Buffer>& Tensor::bufferForWrite()
{
    std::lock_guard<std::mutex> lock(storage_->mutex);
    storage_->hostDirty = false;
    storage_->deviceDirty = true;
    return storage_->buffer;
}

const MatShape& Tensor::shape() const
{
    return shape_;
}

int Tensor::type() const
{
    return storage_->host.type();
}

size_t Tensor::total() const
{
    return shapeTotal(shape_);
}

size_t Tensor::byteSize() const
{
    return total() * storage_->host.elemSize();
}

bool Tensor::isHostDirty() const
{
    std::lock_guard<std::mutex> lock(storage_->mutex);
    return storage_->hostDirty;
}

bool Tensor::isDeviceDirty() const
{
    std::lock_guard<std::mutex> lock(storage_->mutex);
    return storage_->deviceDirty;
}

}}} // namespace cv::dnn::metal
