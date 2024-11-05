// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <cstddef>
#include <vector>
#include "opencv2/core/base.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/cvdef.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/opengl.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/core/array_helpers.hpp"

namespace cv {

/*************************************************************************************************\
                                        Input/Output Array
\*************************************************************************************************/

Mat _InputArray::getMat_(int i) const
{
  CV_Assert(ops != nullptr);
  return ops->getMat_(*this, i);
}

UMat _InputArray::getUMat(int i) const
{
  CV_Assert(ops != nullptr);
  return ops->getUMat(*this, i);
}

void _InputArray::getMatVector(std::vector<Mat>& mv) const
{
    CV_Assert(ops != nullptr);
    mv = ops->getMatVector(*this);
}

void _InputArray::getUMatVector(std::vector<UMat>& umv) const
{
    CV_Assert(ops != nullptr);
    umv = ops->getUMatVector(*this);
}

cuda::GpuMat _InputArray::getGpuMat() const
{
    CV_Assert(ops != nullptr);
    return ops->getGpuMat(*this);
}

void _InputArray::getGpuMatVector(std::vector<cuda::GpuMat>& gpumv) const
{
    CV_Assert(ops != nullptr);
    gpumv = ops->getGpuMatVector(*this);
}

ogl::Buffer _InputArray::getOGlBuffer() const
{
    CV_Assert(ops != nullptr);
    return ops->getOGlBuffer(*this);
}

_InputArray::KindFlag _InputArray::kind() const
{
    KindFlag k = getFlags() & KIND_MASK;
#if CV_VERSION_MAJOR < 5
    CV_DbgAssert(k != EXPR);
    CV_DbgAssert(k != STD_ARRAY);
#endif
    return k;
}

int _InputArray::rows(int i) const
{
    return size(i).height;
}

int _InputArray::cols(int i) const
{
    return size(i).width;
}

Size _InputArray::size(int i) const
{
    CV_Assert(ops != nullptr);
    return ops->size(*this, i);
}

int _InputArray::sizend(int* arrsz, int i) const
{
    CV_Assert(ops != nullptr);
    return ops->sizend(*this, arrsz, i);
}

bool _InputArray::sameSize(const _InputArray& arr) const
{
    CV_Assert(ops != nullptr);
    CV_Assert(arr.ops != nullptr);

    return ops->sameSize(*this, arr);
}

int _InputArray::dims(int i) const
{
    CV_Assert(ops != nullptr);
    return ops->dims(*this, i);
}

size_t _InputArray::total(int i) const
{
    return size(i).area();
}

int _InputArray::type(int i) const
{
    CV_Assert(ops != nullptr);
    return ops->type(*this, i);
}

int _InputArray::depth(int i) const
{
    return CV_MAT_DEPTH(type(i));
}

int _InputArray::channels(int i) const
{
    return CV_MAT_CN(type(i));
}

bool _InputArray::empty() const
{
    CV_Assert(ops != nullptr);
    return ops->empty(*this);
}

bool _InputArray::isContinuous(int i) const
{
    CV_Assert(ops != nullptr);
    return ops->isContinuous(*this, i);
}

bool _InputArray::isSubmatrix(int i) const
{
    CV_Assert(ops != nullptr);
    return ops->isSubmatrix(*this, i);
}

size_t _InputArray::offset(int i) const
{
    CV_Assert(ops != nullptr);
    return ops->offset(*this, i);
}

size_t _InputArray::step(int i) const
{
    CV_Assert(ops != nullptr);
    return ops->step(*this, i);
}

void _InputArray::copyTo(const _OutputArray& arr) const
{
    CV_Assert(ops != nullptr);
    ops->copyTo(*this, arr);
}

void _InputArray::copyTo(const _OutputArray& arr, const _InputArray & mask) const
{
    CV_Assert(ops != nullptr);
    ops->copyTo(*this, arr, mask);
}

bool _InputArray::isMat() const
{
    CV_Assert(ops != nullptr);
    return ops->isMat();
}

bool _InputArray::isUMat() const
{
    CV_Assert(ops != nullptr);
    return ops->isUMat();
}

bool _InputArray::isMatVector() const
{
    CV_Assert(ops != nullptr);
    return ops->isMatVector();
}

bool _InputArray::isUMatVector() const
{
    CV_Assert(ops != nullptr);
    return ops->isUMatVector();
}

bool _InputArray::isMatx() const
{
    CV_Assert(ops != nullptr);
    return ops->isMatx();
}

bool _InputArray::isVector() const
{
    CV_Assert(ops != nullptr);
    return ops->isVector();
}

bool _InputArray::isGpuMat() const
{
    CV_Assert(ops != nullptr);
    return ops->isGpuMat();
}

bool _InputArray::isGpuMatVector() const
{
    CV_Assert(ops != nullptr);
    return ops->isGpuMatVector();
}

bool _OutputArray::fixedSize() const
{
    return (getFlags() & FIXED_SIZE) == FIXED_SIZE;
}

bool _OutputArray::fixedType() const
{
    return (getFlags() & FIXED_TYPE) == FIXED_TYPE;
}

void _OutputArray::create(Size _sz, int mtype, int i, bool allowTransposed, _OutputArray::DepthMask fixedDepthMask) const
{
    CV_Assert(ops != nullptr);
    ops->create(*this, _sz, mtype, i, allowTransposed, fixedDepthMask, fixedSize(), fixedType());
}

void _OutputArray::create(int _rows, int _cols, int mtype, int i, bool allowTransposed, _OutputArray::DepthMask fixedDepthMask) const
{
    CV_Assert(ops != nullptr);
    ops->create(*this, _rows, _cols, mtype, i, allowTransposed, fixedDepthMask, fixedSize(), fixedType());
}

void _OutputArray::create(int d, const int* sizes, int mtype, int i,
                          bool allowTransposed, _OutputArray::DepthMask fixedDepthMask) const
{
    CV_Assert(ops != nullptr);
    ops->create(*this, d, sizes, CV_MAT_TYPE(mtype), i,
                allowTransposed, fixedDepthMask, fixedSize(), fixedType());
}

void _OutputArray::createSameSize(const _InputArray& arr, int mtype) const
{
    int arrsz[CV_MAX_DIM], d = arr.sizend(arrsz);
    create(d, arrsz, mtype);
}

void _OutputArray::release() const
{
    CV_Assert(ops != nullptr);
    ops->release(*this, fixedSize());
}

void _OutputArray::clear() const
{
    CV_Assert(ops != nullptr);
    ops->clear(*this, fixedSize());
}

bool _OutputArray::needed() const
{
    return kind() != NONE;
}

Mat& _OutputArray::getMatRef(int i) const
{
    CV_Assert(ops != nullptr);
    return ops->getMatRef(*this, i);
}

UMat& _OutputArray::getUMatRef(int i) const
{
    CV_Assert(ops != nullptr);
    return ops->getUMatRef(*this, i);
}

cuda::GpuMat& _OutputArray::getGpuMatRef() const
{
    CV_Assert(ops != nullptr);
    return ops->getGpuMatRef(*this);
}
std::vector<cuda::GpuMat>& _OutputArray::getGpuMatVecRef() const
{
    CV_Assert(ops != nullptr);
    return ops->getGpuMatVecRef(*this);
}

ogl::Buffer& _OutputArray::getOGlBufferRef() const
{
    CV_Assert(ops != nullptr);
    return ops->getOGlBufferRef(*this);
}

cuda::HostMem& _OutputArray::getHostMemRef() const
{
    CV_Assert(ops != nullptr);
    return ops->getHostMemRef(*this);
}

void _OutputArray::setTo(const _InputArray& arr, const _InputArray & mask) const
{
    CV_Assert(ops != nullptr);
    return ops->setTo(*this, arr, mask);
}

void _OutputArray::assign(const UMat& u) const
{
    CV_Assert(ops != nullptr);
    return ops->assign(*this, u);
}


void _OutputArray::assign(const Mat& m) const
{
    CV_Assert(ops != nullptr);
    return ops->assign(*this, m);
}


void _OutputArray::move(UMat& u) const
{
    CV_Assert(ops != nullptr);
    return ops->move(*this, u, fixedSize());
}


void _OutputArray::move(Mat& m) const
{
    CV_Assert(ops != nullptr);
    return ops->move(*this, m, fixedSize());
}


void _OutputArray::assign(const std::vector<UMat>& v) const
{
    CV_Assert(ops != nullptr);
    return ops->assign(*this, v);
}


void _OutputArray::assign(const std::vector<Mat>& v) const
{
    CV_Assert(ops != nullptr);
    return ops->assign(*this, v);
}


static _InputOutputArray _none;
InputOutputArray noArray() { return _none; }

[[noreturn]] void _InputArrayOpsBase::noCudaError()
{
    CV_Error(
        Error::StsNotImplemented,
        "CUDA support is not enabled in this OpenCV build (missing HAVE_CUDA)");
}

[[noreturn]] void _InputArrayOpsBase::noOpenGLError()
{
    CV_Error(
        Error::StsNotImplemented,
        "OpenGL support is not enabled in this OpenCV build (missing HAVE_OPENGL)");
}

[[noreturn]] void _InputArrayOpsBase::unsupportedTypeError()
{
    CV_Error(Error::StsNotImplemented, "Unknown/unsupported array type");
}

[[noreturn]] void _InputArrayOpsBase::customError(const Error::Code error, const std::string_view message)
{
    CV_Error(error, message.data());
}

[[noreturn]] void _InputArrayOpsBase::createError(const std::size_t size)
{
    CV_Error_(cv::Error::StsBadArg, ("Vectors with element size %zu are not supported. Please, modify OutputArray::create()\n", size));
}
} // cv::
