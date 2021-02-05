/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_CUDAINL_HPP
#define OPENCV_CORE_CUDAINL_HPP

#include "opencv2/core/cuda.hpp"

//! @cond IGNORED

namespace cv { namespace cuda {

//===================================================================================
// GpuMat
//===================================================================================

inline
GpuMat::GpuMat(Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
{}

inline
GpuMat::GpuMat(int rows_, int cols_, int type_, Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
{
    if (rows_ > 0 && cols_ > 0)
        create(rows_, cols_, type_);
}

inline
GpuMat::GpuMat(Size size_, int type_, Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
{
    if (size_.height > 0 && size_.width > 0)
        create(size_.height, size_.width, type_);
}

inline
GpuMat::GpuMat(int rows_, int cols_, int type_, Scalar s_, Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
{
    if (rows_ > 0 && cols_ > 0)
    {
        create(rows_, cols_, type_);
        setTo(s_);
    }
}

inline
GpuMat::GpuMat(Size size_, int type_, Scalar s_, Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
{
    if (size_.height > 0 && size_.width > 0)
    {
        create(size_.height, size_.width, type_);
        setTo(s_);
    }
}

inline
GpuMat::GpuMat(const GpuMat& m)
    : flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data), refcount(m.refcount), datastart(m.datastart), dataend(m.dataend), allocator(m.allocator)
{
    if (refcount)
        CV_XADD(refcount, 1);
}

inline
GpuMat::GpuMat(InputArray arr, Allocator* allocator_) :
    flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
{
    upload(arr);
}

inline
GpuMat::~GpuMat()
{
    release();
}

inline
GpuMat& GpuMat::operator =(const GpuMat& m)
{
    if (this != &m)
    {
        GpuMat temp(m);
        swap(temp);
    }

    return *this;
}

inline
void GpuMat::create(Size size_, int type_)
{
    create(size_.height, size_.width, type_);
}

inline
void GpuMat::swap(GpuMat& b)
{
    std::swap(flags, b.flags);
    std::swap(rows, b.rows);
    std::swap(cols, b.cols);
    std::swap(step, b.step);
    std::swap(data, b.data);
    std::swap(datastart, b.datastart);
    std::swap(dataend, b.dataend);
    std::swap(refcount, b.refcount);
    std::swap(allocator, b.allocator);
}

inline
GpuMat GpuMat::clone() const
{
    GpuMat m;
    copyTo(m);
    return m;
}

inline
void GpuMat::copyTo(OutputArray dst, InputArray mask) const
{
    copyTo(dst, mask, Stream::Null());
}

inline
GpuMat& GpuMat::setTo(Scalar s)
{
    return setTo(s, Stream::Null());
}

inline
GpuMat& GpuMat::setTo(Scalar s, InputArray mask)
{
    return setTo(s, mask, Stream::Null());
}

inline
void GpuMat::convertTo(OutputArray dst, int rtype) const
{
    convertTo(dst, rtype, Stream::Null());
}

inline
void GpuMat::convertTo(OutputArray dst, int rtype, double alpha, double beta) const
{
    convertTo(dst, rtype, alpha, beta, Stream::Null());
}

inline
void GpuMat::convertTo(OutputArray dst, int rtype, double alpha, Stream& stream) const
{
    convertTo(dst, rtype, alpha, 0.0, stream);
}

inline
void GpuMat::assignTo(GpuMat& m, int _type) const
{
    if (_type < 0)
        m = *this;
    else
        convertTo(m, _type);
}

inline
uchar* GpuMat::ptr(int y)
{
    CV_DbgAssert( (unsigned)y < (unsigned)rows );
    return data + step * y;
}

inline
const uchar* GpuMat::ptr(int y) const
{
    CV_DbgAssert( (unsigned)y < (unsigned)rows );
    return data + step * y;
}

template<typename _Tp> inline
_Tp* GpuMat::ptr(int y)
{
    return (_Tp*)ptr(y);
}

template<typename _Tp> inline
const _Tp* GpuMat::ptr(int y) const
{
    return (const _Tp*)ptr(y);
}

template <class T> inline
GpuMat::operator PtrStepSz<T>() const
{
    return PtrStepSz<T>(rows, cols, (T*)data, step);
}

template <class T> inline
GpuMat::operator PtrStep<T>() const
{
    return PtrStep<T>((T*)data, step);
}

inline
GpuMat GpuMat::row(int y) const
{
    return GpuMat(*this, Range(y, y+1), Range::all());
}

inline
GpuMat GpuMat::col(int x) const
{
    return GpuMat(*this, Range::all(), Range(x, x+1));
}

inline
GpuMat GpuMat::rowRange(int startrow, int endrow) const
{
    return GpuMat(*this, Range(startrow, endrow), Range::all());
}

inline
GpuMat GpuMat::rowRange(Range r) const
{
    return GpuMat(*this, r, Range::all());
}

inline
GpuMat GpuMat::colRange(int startcol, int endcol) const
{
    return GpuMat(*this, Range::all(), Range(startcol, endcol));
}

inline
GpuMat GpuMat::colRange(Range r) const
{
    return GpuMat(*this, Range::all(), r);
}

inline
GpuMat GpuMat::operator ()(Range rowRange_, Range colRange_) const
{
    return GpuMat(*this, rowRange_, colRange_);
}

inline
GpuMat GpuMat::operator ()(Rect roi) const
{
    return GpuMat(*this, roi);
}

inline
bool GpuMat::isContinuous() const
{
    return (flags & Mat::CONTINUOUS_FLAG) != 0;
}

inline
size_t GpuMat::elemSize() const
{
    return CV_ELEM_SIZE(flags);
}

inline
size_t GpuMat::elemSize1() const
{
    return CV_ELEM_SIZE1(flags);
}

inline
int GpuMat::type() const
{
    return CV_MAT_TYPE(flags);
}

inline
int GpuMat::depth() const
{
    return CV_MAT_DEPTH(flags);
}

inline
int GpuMat::channels() const
{
    return CV_MAT_CN(flags);
}

inline
size_t GpuMat::step1() const
{
    return step / elemSize1();
}

inline
Size GpuMat::size() const
{
    return Size(cols, rows);
}

inline
bool GpuMat::empty() const
{
    return data == 0;
}

inline
void* GpuMat::cudaPtr() const
{
    return data;
}

static inline
GpuMat createContinuous(int rows, int cols, int type)
{
    GpuMat m;
    createContinuous(rows, cols, type, m);
    return m;
}

static inline
void createContinuous(Size size, int type, OutputArray arr)
{
    createContinuous(size.height, size.width, type, arr);
}

static inline
GpuMat createContinuous(Size size, int type)
{
    GpuMat m;
    createContinuous(size, type, m);
    return m;
}

static inline
void ensureSizeIsEnough(Size size, int type, OutputArray arr)
{
    ensureSizeIsEnough(size.height, size.width, type, arr);
}

static inline
void swap(GpuMat& a, GpuMat& b)
{
    a.swap(b);
}

//===================================================================================
// GpuMatND
//===================================================================================

inline
GpuMatND::GpuMatND() :
    flags(0), dims(0), data(nullptr), offset(0)
{
}

inline
GpuMatND::GpuMatND(SizeArray _size, int _type) :
    flags(0), dims(0), data(nullptr), offset(0)
{
    create(std::move(_size), _type);
}

inline
void GpuMatND::swap(GpuMatND& m) noexcept
{
    std::swap(*this, m);
}

inline
bool GpuMatND::isContinuous() const
{
    return (flags & Mat::CONTINUOUS_FLAG) != 0;
}

inline
bool GpuMatND::isSubmatrix() const
{
    return (flags & Mat::SUBMATRIX_FLAG) != 0;
}

inline
size_t GpuMatND::elemSize() const
{
    return CV_ELEM_SIZE(flags);
}

inline
size_t GpuMatND::elemSize1() const
{
    return CV_ELEM_SIZE1(flags);
}

inline
bool GpuMatND::empty() const
{
    return data == nullptr;
}

inline
bool GpuMatND::external() const
{
    return !empty() && data_.use_count() == 0;
}

inline
uchar* GpuMatND::getDevicePtr() const
{
    return data + offset;
}

inline
size_t GpuMatND::total() const
{
    size_t p = 1;
    for(auto s : size)
        p *= s;
    return p;
}

inline
size_t GpuMatND::totalMemSize() const
{
    return size[0] * step[0];
}

inline
int GpuMatND::type() const
{
    return CV_MAT_TYPE(flags);
}

//===================================================================================
// HostMem
//===================================================================================

inline
HostMem::HostMem(AllocType alloc_type_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(alloc_type_)
{
}

inline
HostMem::HostMem(const HostMem& m)
    : flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data), refcount(m.refcount), datastart(m.datastart), dataend(m.dataend), alloc_type(m.alloc_type)
{
    if( refcount )
        CV_XADD(refcount, 1);
}

inline
HostMem::HostMem(int rows_, int cols_, int type_, AllocType alloc_type_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(alloc_type_)
{
    if (rows_ > 0 && cols_ > 0)
        create(rows_, cols_, type_);
}

inline
HostMem::HostMem(Size size_, int type_, AllocType alloc_type_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(alloc_type_)
{
    if (size_.height > 0 && size_.width > 0)
        create(size_.height, size_.width, type_);
}

inline
HostMem::HostMem(InputArray arr, AllocType alloc_type_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(alloc_type_)
{
    arr.getMat().copyTo(*this);
}

inline
HostMem::~HostMem()
{
    release();
}

inline
HostMem& HostMem::operator =(const HostMem& m)
{
    if (this != &m)
    {
        HostMem temp(m);
        swap(temp);
    }

    return *this;
}

inline
void HostMem::swap(HostMem& b)
{
    std::swap(flags, b.flags);
    std::swap(rows, b.rows);
    std::swap(cols, b.cols);
    std::swap(step, b.step);
    std::swap(data, b.data);
    std::swap(datastart, b.datastart);
    std::swap(dataend, b.dataend);
    std::swap(refcount, b.refcount);
    std::swap(alloc_type, b.alloc_type);
}

inline
HostMem HostMem::clone() const
{
    HostMem m(size(), type(), alloc_type);
    createMatHeader().copyTo(m);
    return m;
}

inline
void HostMem::create(Size size_, int type_)
{
    create(size_.height, size_.width, type_);
}

inline
Mat HostMem::createMatHeader() const
{
    return Mat(size(), type(), data, step);
}

inline
bool HostMem::isContinuous() const
{
    return (flags & Mat::CONTINUOUS_FLAG) != 0;
}

inline
size_t HostMem::elemSize() const
{
    return CV_ELEM_SIZE(flags);
}

inline
size_t HostMem::elemSize1() const
{
    return CV_ELEM_SIZE1(flags);
}

inline
int HostMem::type() const
{
    return CV_MAT_TYPE(flags);
}

inline
int HostMem::depth() const
{
    return CV_MAT_DEPTH(flags);
}

inline
int HostMem::channels() const
{
    return CV_MAT_CN(flags);
}

inline
size_t HostMem::step1() const
{
    return step / elemSize1();
}

inline
Size HostMem::size() const
{
    return Size(cols, rows);
}

inline
bool HostMem::empty() const
{
    return data == 0;
}

static inline
void swap(HostMem& a, HostMem& b)
{
    a.swap(b);
}

//===================================================================================
// Stream
//===================================================================================

inline
Stream::Stream(const Ptr<Impl>& impl)
    : impl_(impl)
{
}

//===================================================================================
// Event
//===================================================================================

inline
Event::Event(const Ptr<Impl>& impl)
    : impl_(impl)
{
}

//===================================================================================
// Initialization & Info
//===================================================================================

inline
bool TargetArchs::has(int major, int minor)
{
    return hasPtx(major, minor) || hasBin(major, minor);
}

inline
bool TargetArchs::hasEqualOrGreater(int major, int minor)
{
    return hasEqualOrGreaterPtx(major, minor) || hasEqualOrGreaterBin(major, minor);
}

inline
DeviceInfo::DeviceInfo()
{
    device_id_ = getDevice();
}

inline
DeviceInfo::DeviceInfo(int device_id)
{
    CV_Assert( device_id >= 0 && device_id < getCudaEnabledDeviceCount() );
    device_id_ = device_id;
}

inline
int DeviceInfo::deviceID() const
{
    return device_id_;
}

inline
size_t DeviceInfo::freeMemory() const
{
    size_t _totalMemory = 0, _freeMemory = 0;
    queryMemory(_totalMemory, _freeMemory);
    return _freeMemory;
}

inline
size_t DeviceInfo::totalMemory() const
{
    size_t _totalMemory = 0, _freeMemory = 0;
    queryMemory(_totalMemory, _freeMemory);
    return _totalMemory;
}

inline
bool DeviceInfo::supports(FeatureSet feature_set) const
{
    int version = majorVersion() * 10 + minorVersion();
    return version >= feature_set;
}


}} // namespace cv { namespace cuda {

//===================================================================================
// Mat
//===================================================================================

namespace cv {

inline
Mat::Mat(const cuda::GpuMat& m)
    : flags(0), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows)
{
    m.download(*this);
}

}

//! @endcond

#endif // OPENCV_CORE_CUDAINL_HPP
