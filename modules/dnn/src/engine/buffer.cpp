// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

void fitMat(Mat& m, size_t size)
{
    if (!m.isContinuous() || m.total()*m.elemSize() < size) {
        m.release();
        CV_Assert(size <= (size_t)INT_MAX);
        m.create((int)size, 1, CV_8U);
    }
}

Device::~Device() {}

struct CPUDevice : Device
{
    virtual string name() const
    { return
#if (defined __SSE2__) || (defined _M_X86) || (defined _M_X64)
        "x86 CPU";
#elif (defined __ARM_NEON)
        "ARM CPU";
#elif (defined CV_SIMD)
        "Generic CPU w. SIMD";
#endif
    }
    virtual int kind() const { return DEV_CPU; }
    virtual bool zeroCopy() const { return true; }
    virtual bool supportType(int typ) const {
#if (defined __ARM_NEON) && (defined __ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        return true;
#else
        return CV_MAT_DEPTH(typ) != CV_16F;
#endif
    }
    virtual MemoryManager* defaultMemoryManager() { return 0; }
};

Device* getCPUDevice()
{
    static CPUDevice defaultCPU;
    return &defaultCPU;
}

struct CPUMemoryManager : MemoryManager
{
    void* allocate(Device*, size_t bufsize) { return malloc(bufsize); }
    void release(Device*, void* handle) { free(handle); }
    void* map(Device*, void* handle, size_t, int) { return handle; }
    void unmap(Device*, void*, void*, size_t, int) {}
    void copyFromDevice(Device*, void* handle, size_t offset, size_t size, void* dst)
    { memcpy(dst, (char*)handle + offset, size); }
    void copyToDevice(Device*, const void* src, void* handle, size_t offset, size_t size)
    { memcpy((char*)handle + offset, src, size); }
};

Buffer::Buffer()
{
    device = 0;
    mm = 0;
    shared = 0;
    handle = 0;
    size = 0;
}

Buffer::Shared::Shared()
{
    ptr = 0;
    refcount = 1;
    mapcount = 0;
}

Buffer::~Buffer()
{
    release();
}

void Buffer::release()
{
    if(shared && CV_XADD(&shared->refcount, -1) == 1) {
        if (mm)
            mm->release(device, handle);
        else
            free(handle);
        delete shared;
    }
    handle = 0;
    shared = 0;
}

Buffer::Buffer(const Buffer& buf)
{
    if (buf.shared)
        CV_XADD(&buf.shared->refcount, 1);
    mm = buf.mm;
    device = buf.device;
    handle = buf.handle;
    shared = buf.shared;
}

Buffer::Buffer(const void* data, size_t size_, bool copy)
{
    mm = 0;
    device = 0;
    shared = 0;
    if (!copy) {
        handle = (void*)data;
        size = size_;
    } else {
        handle = 0;
        size = 0;
        if (size > 0) {
            fit(size_);
            memcpy(handle, data, size);
        }
    }
}

Buffer& Buffer::operator = (const Buffer& buf)
{
    if (this != &buf &&
        (!shared || !buf.shared || shared->refcount != buf.shared->refcount))
    {
        if (buf.shared)
            CV_XADD(&buf.shared->refcount, 1);
        release();
        mm = buf.mm;
        device = buf.device;
        handle = buf.handle;
        size = buf.size;
        shared = buf.shared;
    }
    return *this;
}

Buffer Buffer::allocate(size_t size, MemoryManager* mm, Device* device)
{
    Buffer buf;
    buf.mm = mm;
    buf.device = device;
    buf.fit(size);
    return buf;
}

void Buffer::fit(size_t size_)
{
    if (size_ > size) {
        release();
        size = size_;
        shared = new Shared;
        if (mm)
            handle = mm->allocate(device, size);
        else
            handle = malloc(size);
    }
}

void Buffer::set(const void* data, size_t size_, bool copy)
{
    if (!copy) {
        release();
        CV_Assert(data != 0 || size == 0);
        mm = 0;
        device = 0;
        handle = (void*)data;
        size = size_;
    } else {
        fit(size_);
        if (size_ > 0) {
            if (mm)
                mm->copyToDevice(device, data, handle, 0, size_);
            else
                memcpy(handle, data, size_);
        }
    }
}

void* Buffer::map(int access)
{
    if (mm) {
        if (!shared) return 0;
        if (CV_XADD(&shared->mapcount, 1) == 0)
            shared->ptr = mm->map(device, handle, size, access);
        return shared->ptr;
    } else {
        return handle;
    }
}

void Buffer::unmap(int access)
{
    if (mm && shared && CV_XADD(&shared->mapcount, -1) == 1) {
        mm->unmap(device, handle, shared->ptr, size, access);
        shared->ptr = 0;
    }
}

TensorShape::TensorShape()
{
    C = 0;
    layout = DNN_LAYOUT_UNKNOWN;
    ndims = 1;
    shape[0] = 0;
}

size_t TensorShape::total() const
{
    size_t p = 1;
    for (size_t i = 0; i < ndims; i++) p *= shape[i];
    return p;
}

bool TensorShape::empty() const
{
    return total() == 0;
}

void TensorShape::updateC()
{
    C = layout != DNN_LAYOUT_NCHWс ? 0 : ndims == 1 ? 1 :
        ndims == 2 ? (int)shape[0] : (int)(shape[0]*shape[ndims-1]);
}

TensorShape TensorShape::fromArray(InputArray m, int ndims_, DataLayout layout_)
{
    TensorShape shape;
    shape.layout = layout_;

    if (!m.empty()) {
        int dims[CV_MAX_DIM];
        shape.ndims = m.sizend(dims);
        CV_Assert(ndims_ < 0 || shape.ndims == ndims_ ||
                  (shape.ndims == 2 && ndims_ <= 1));
        if (ndims_ == 0) {
            CV_Assert(dims[0]*dims[1] == 1);
            shape.ndims = 0;
        } else if (ndims_ == 1) {
            CV_Assert(dims[0] == 1 || dims[1] == 1);
            shape.ndims = 1;
            dims[0] *= dims[1];
        }
        for (int i = 0; i < shape.ndims; i++)
            shape.shape[i] = dims[i];
    }
    shape.updateC();
    return shape;
}

int TensorShape::toMatShape(int* mshape, int maxdims) const
{
    int mdims = std::max(ndims, 2);
    CV_Assert(maxdims >= mdims);
    int i = 0;
    for (; i < ndims; i++) {
        int64_t sz_i = shape[i];
        CV_Assert(INT_MIN <= sz_i && sz_i <= INT_MAX);
        mshape[i] = (int)sz_i;
    }
    for (; i < mdims; i++)
        mshape[i] = 1;
    return mdims;
}

Tensor::Tensor()
{
    typ = -1;
}

Tensor::Tensor(const Tensor& t)
{
    shape = t.shape;
    typ = t.typ;
    buf = t.buf;
}

Tensor& Tensor::operator = (const Tensor& t)
{
    if (this != &t) {
        shape = t.shape;
        typ = t.typ;
        buf = t.buf;
    }
    return *this;
}

Tensor::~Tensor() {}

Tensor::Tensor(const TensorShape& shape_, int typ_)
{
    shape = shape_;
    typ = typ_;
    size_t sz = shape.total()*CV_ELEM_SIZE(typ);
    buf = Buffer::allocate(sz);
}

Tensor::Tensor(const TensorShape& shape_, int typ_, void* data, bool copy)
{
    typ = -1;
    set(shape_, typ_, data, copy);
}

Tensor::Tensor(InputArray arr, int ndims, bool copy)
{
    set(arr, ndims, copy);
}

void Tensor::set(const TensorShape& shape_, int typ_, void* data, bool copy)
{
    shape = shape_;
    typ = typ_;
    size_t size = shape.total()*CV_ELEM_SIZE(typ);
    buf.set(data, size, copy);
}

void Tensor::set(InputArray arr, int ndims, bool copy)
{
    Mat m = arr.getMat();
    TensorShape shape_ = TensorShape::fromArray(m, ndims);
    int typ_ = m.type();
    if (!arr.empty()) {
        CV_Assert(copy || arr.isMat() || arr.isVector()); // if it's UMat, for example,
                                                        // the mapped pointer may become invalid after the call
    }
    set(shape_, typ_, m.data, copy);
}

void Tensor::fit(const TensorShape& shape_, int typ_)
{
    shape = shape_;
    typ = typ_;
    size_t size = shape.total()*CV_ELEM_SIZE(typ);
    buf.fit(size);
}

bool Tensor::empty() const
{
    return shape.empty();
}

size_t Tensor::total() const
{
    return shape.total();
}

void* Tensor::data() const
{
    void* dataptr = buf.mm && buf.shared ? buf.shared->ptr : buf.handle;
    CV_Assert(dataptr != 0 || shape.empty()); // make sure the tensor is "mapped" to memory
    return dataptr;
}

Mat Tensor::getMat() const
{
    int mshape[CV_MAX_DIM];
    int mdims = shape.toMatShape(mshape, CV_MAX_DIM);
    void* dataptr = data();
    return Mat(mdims, mshape, typ, dataptr);
}

void* Tensor::map(int access)
{
    return buf.map(access);
}

void Tensor::unmap(int access)
{
    return buf.unmap(access);
}

CV__DNN_INLINE_NS_END
}}
