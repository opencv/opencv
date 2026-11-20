// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"

namespace cv {

Tensor::Tensor()
{
}

Tensor::Tensor(const MatShape& _shape, int _type)
{
    allocate(_shape, _type, ComputeDevice::current());
}

Tensor::Tensor(const MatShape& _shape, int _type, const ComputeDevice& dev)
{
    allocate(_shape, _type, dev);
}

Tensor::Tensor(const MatShape& _shape, int _type, const Scalar& value)
{
    allocate(_shape, _type, ComputeDevice::current());
    setTo(value);
}

Tensor::Tensor(const MatShape& _shape, int _type, const Scalar& value, const ComputeDevice& dev)
{
    allocate(_shape, _type, dev);
    setTo(value);
}

Tensor::Tensor(const Mat& m)
    : m_(m)
{
}

Tensor::Tensor(const UMat& u)
    : u_(u)
{
}

/** @brief True when the data, or at least a shape, is in the Mat header. */
bool Tensor::usesMat() const
{
    return m_.data != 0 || m_.dims != 0;
}

/** @brief True when the data, or at least a shape, is in the UMat header. */
bool Tensor::usesUMat() const
{
    return u_.u != 0 || u_.dims != 0;
}

/** @brief Picks the header an output writes: true for the UMat, false for the Mat.

A tensor in use keeps its header. Device data is pinned to its allocator, so a resize keeps it on
its device. An empty tensor gets the current device. */
bool Tensor::prepareOutput()
{
    if (usesUMat())
    {
        if (!u_.allocator && u_.u)
            u_.allocator = const_cast<MatAllocator*>(u_.u->currAllocator);
        return true;
    }
    if (usesMat())
        return false;
    const ComputeDevice dev = ComputeDevice::current();
    if (dev.isCPU())
        return false;
    u_.allocator = dev.allocator();
    return true;
}

/** @brief Allocates an empty tensor on @p dev. */
void Tensor::allocate(const MatShape& _shape, int _type, const ComputeDevice& dev)
{
    if (dev.isCPU())
    {
        m_.create(_shape, _type);
        return;
    }
    u_.allocator = dev.allocator();
    u_.create(_shape, _type);
}

ComputeDevice Tensor::device() const
{
    return usesUMat() ? ComputeDevice::of(u_) : ComputeDevice::cpu();
}

Tensor Tensor::to(const ComputeDevice& dev) const
{
    if (empty() || dev == device())
        return *this;
    Tensor dst;
    dst.allocate(shape(), type(), dev);
    copyTo(dst);
    return dst;
}

Tensor Tensor::to(const String& spec) const
{
    return to(ComputeDevice::find(spec));
}

Mat Tensor::getMat(AccessFlag access) const
{
    return usesUMat() ? u_.getMat(access) : m_;
}

UMat Tensor::getUMat(AccessFlag access) const
{
    return usesUMat() ? u_ : m_.getUMat(access);
}

MatShape Tensor::shape() const
{
    return usesUMat() ? u_.shape() : m_.shape();
}

int Tensor::type() const
{
    return usesUMat() ? u_.type() : m_.type();
}

int Tensor::depth() const
{
    return CV_MAT_DEPTH(type());
}

int Tensor::channels() const
{
    return CV_MAT_CN(type());
}

int Tensor::dims() const
{
    return usesUMat() ? u_.dims : m_.dims;
}

size_t Tensor::total() const
{
    return usesUMat() ? u_.total() : m_.total();
}

size_t Tensor::elemSize() const
{
    return CV_ELEM_SIZE(type());
}

bool Tensor::empty() const
{
    return usesUMat() ? u_.empty() : m_.empty();
}

bool Tensor::isContinuous() const
{
    return usesUMat() ? u_.isContinuous() : m_.isContinuous();
}

Tensor Tensor::reshape(const MatShape& newShape) const
{
    Tensor t;
    if (usesUMat())
        t.u_ = u_.reshape(0, newShape);
    else
        t.m_ = m_.reshape(0, newShape);
    return t;
}

Tensor Tensor::operator()(const std::vector<Range>& ranges) const
{
    Tensor t;
    if (usesUMat())
        t.u_ = u_(ranges);
    else
        t.m_ = m_(ranges);
    return t;
}

Tensor Tensor::operator()(const Rect& roi) const
{
    Tensor t;
    if (usesUMat())
        t.u_ = u_(roi);
    else
        t.m_ = m_(roi);
    return t;
}

void Tensor::create(const MatShape& _shape, int _type)
{
    if (prepareOutput())
        u_.create(_shape, _type);
    else
        m_.create(_shape, _type);
}

void Tensor::release()
{
    m_ = Mat();
    u_ = UMat();
}

Tensor Tensor::clone() const
{
    Tensor dst;
    copyTo(dst);
    return dst;
}

void Tensor::copyTo(Tensor& dst) const
{
    if (!dst.usesMat() && !dst.usesUMat() && !empty())
        dst.allocate(shape(), type(), device());
    copyTo(_OutputArray(dst));
}

void Tensor::copyTo(OutputArray dst) const
{
    if (usesUMat())
        u_.copyTo(dst);
    else
        m_.copyTo(dst);
}

void Tensor::convertTo(Tensor& dst, int rtype, double alpha, double beta) const
{
    if (!dst.usesMat() && !dst.usesUMat() && !empty())
    {
        const int dtype = rtype < 0 ? type() : CV_MAKETYPE(CV_MAT_DEPTH(rtype), channels());
        dst.allocate(shape(), dtype, device());
    }
    convertTo(_OutputArray(dst), rtype, alpha, beta);
}

void Tensor::convertTo(OutputArray dst, int rtype, double alpha, double beta) const
{
    if (usesUMat())
        u_.convertTo(dst, rtype, alpha, beta);
    else
        m_.convertTo(dst, rtype, alpha, beta);
}

Tensor& Tensor::setTo(InputArray value, InputArray mask)
{
    if (usesUMat())
        u_.setTo(value, mask);
    else
        m_.setTo(value, mask);
    return *this;
}

Tensor Tensor::zeros(const MatShape& _shape, int _type, const ComputeDevice& dev)
{
    return Tensor(_shape, _type, Scalar::all(0), dev);
}

_InputArray::_InputArray(const Tensor& t)
{
    if (t.usesUMat())
        init(+UMAT + ACCESS_READ, &t.u_);
    else
        init(+MAT + ACCESS_READ, &t.m_);
}

_OutputArray::_OutputArray(Tensor& t)
{
    if (t.prepareOutput())
        init(+UMAT + ACCESS_WRITE, &t.u_);
    else
        init(+MAT + ACCESS_WRITE, &t.m_);
}

_OutputArray::_OutputArray(const Tensor& t)
{
    if (t.usesUMat())
        init(+FIXED_TYPE + FIXED_SIZE + UMAT + ACCESS_WRITE, &t.u_);
    else
        init(+FIXED_TYPE + FIXED_SIZE + MAT + ACCESS_WRITE, &t.m_);
}

_InputOutputArray::_InputOutputArray(Tensor& t)
{
    if (t.prepareOutput())
        init(+UMAT + ACCESS_RW, &t.u_);
    else
        init(+MAT + ACCESS_RW, &t.m_);
}

_InputOutputArray::_InputOutputArray(const Tensor& t)
{
    if (t.usesUMat())
        init(+FIXED_TYPE + FIXED_SIZE + UMAT + ACCESS_RW, &t.u_);
    else
        init(+FIXED_TYPE + FIXED_SIZE + MAT + ACCESS_RW, &t.m_);
}

} // namespace cv
