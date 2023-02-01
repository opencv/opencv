/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "opencv2/opencv_modules.hpp"

#ifndef HAVE_OPENCV_CUDEV

#error "opencv_cudev is required"

#else

#include "opencv2/core/cuda.hpp"
#include "opencv2/cudev.hpp"
#include "opencv2/core/cuda/utility.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudev;

device::ThrustAllocator::~ThrustAllocator()
{
}
namespace
{
    class DefaultThrustAllocator: public cv::cuda::device::ThrustAllocator
    {
    public:
        __device__ __host__ uchar* allocate(size_t numBytes) CV_OVERRIDE
        {
#ifndef __CUDA_ARCH__
            uchar* ptr;
            CV_CUDEV_SAFE_CALL(cudaMalloc(&ptr, numBytes));
            return ptr;
#else
            return NULL;
#endif
        }
        __device__ __host__ void deallocate(uchar* ptr, size_t numBytes) CV_OVERRIDE
        {
            CV_UNUSED(numBytes);
#ifndef __CUDA_ARCH__
            CV_CUDEV_SAFE_CALL(cudaFree(ptr));
#endif
        }
    };
    DefaultThrustAllocator defaultThrustAllocator;
    cv::cuda::device::ThrustAllocator* g_thrustAllocator = &defaultThrustAllocator;
}


cv::cuda::device::ThrustAllocator& cv::cuda::device::ThrustAllocator::getAllocator()
{
    return *g_thrustAllocator;
}

void cv::cuda::device::ThrustAllocator::setAllocator(cv::cuda::device::ThrustAllocator* allocator)
{
    if(allocator == NULL)
        g_thrustAllocator = &defaultThrustAllocator;
    else
        g_thrustAllocator = allocator;
}

namespace
{
    class DefaultAllocator : public GpuMat::Allocator
    {
    public:
        bool allocate(GpuMat* mat, int rows, int cols, size_t elemSize) CV_OVERRIDE;
        void free(GpuMat* mat) CV_OVERRIDE;
    };

    bool DefaultAllocator::allocate(GpuMat* mat, int rows, int cols, size_t elemSize)
    {
        if (rows > 1 && cols > 1)
        {
            CV_CUDEV_SAFE_CALL( cudaMallocPitch(&mat->data, &mat->step, elemSize * cols, rows) );
        }
        else
        {
            // Single row or single column must be continuous
            CV_CUDEV_SAFE_CALL( cudaMalloc(&mat->data, elemSize * cols * rows) );
            mat->step = elemSize * cols;
        }

        mat->refcount = (int*) fastMalloc(sizeof(int));

        return true;
    }

    void DefaultAllocator::free(GpuMat* mat)
    {
        cudaFree(mat->datastart);
        fastFree(mat->refcount);
    }

    DefaultAllocator cudaDefaultAllocator;
    GpuMat::Allocator* g_defaultAllocator = &cudaDefaultAllocator;
}

GpuMat::Allocator* cv::cuda::GpuMat::defaultAllocator()
{
    return g_defaultAllocator;
}

void cv::cuda::GpuMat::setDefaultAllocator(Allocator* allocator)
{
    CV_Assert( allocator != 0 );
    g_defaultAllocator = allocator;
}

/////////////////////////////////////////////////////
/// create

void cv::cuda::GpuMat::create(int _rows, int _cols, int _type)
{
    CV_DbgAssert( _rows >= 0 && _cols >= 0 );

    _type &= Mat::TYPE_MASK;

    if (rows == _rows && cols == _cols && type() == _type && data)
        return;

    if (data)
        release();

    if (_rows > 0 && _cols > 0)
    {
        flags = Mat::MAGIC_VAL + _type;
        rows = _rows;
        cols = _cols;

        const size_t esz = elemSize();

        bool allocSuccess = allocator->allocate(this, rows, cols, esz);

        if (!allocSuccess)
        {
            // custom allocator fails, try default allocator
            allocator = defaultAllocator();
            allocSuccess = allocator->allocate(this, rows, cols, esz);
            CV_Assert( allocSuccess );
        }

        if (esz * cols == step)
            flags |= Mat::CONTINUOUS_FLAG;

        int64 _nettosize = static_cast<int64>(step) * rows;
        size_t nettosize = static_cast<size_t>(_nettosize);

        datastart = data;
        dataend = data + nettosize;

        if (refcount)
            *refcount = 1;
    }
}

/////////////////////////////////////////////////////
/// release

void cv::cuda::GpuMat::release()
{
    CV_DbgAssert( allocator != 0 );

    if (refcount && CV_XADD(refcount, -1) == 1)
        allocator->free(this);

    dataend = data = datastart = 0;
    step = rows = cols = 0;
    refcount = 0;
}

/////////////////////////////////////////////////////
/// upload

void cv::cuda::GpuMat::upload(InputArray arr)
{
    Mat mat = arr.getMat();

    CV_DbgAssert( !mat.empty() );

    create(mat.size(), mat.type());

    CV_CUDEV_SAFE_CALL( cudaMemcpy2D(data, step, mat.data, mat.step, cols * elemSize(), rows, cudaMemcpyHostToDevice) );
}

void cv::cuda::GpuMat::upload(InputArray arr, Stream& _stream)
{
    Mat mat = arr.getMat();

    CV_DbgAssert( !mat.empty() );

    create(mat.size(), mat.type());

    cudaStream_t stream = StreamAccessor::getStream(_stream);
    CV_CUDEV_SAFE_CALL( cudaMemcpy2DAsync(data, step, mat.data, mat.step, cols * elemSize(), rows, cudaMemcpyHostToDevice, stream) );
}

/////////////////////////////////////////////////////
/// download

void cv::cuda::GpuMat::download(OutputArray _dst) const
{
    CV_DbgAssert( !empty() );

    _dst.create(size(), type());
    Mat dst = _dst.getMat();

    CV_CUDEV_SAFE_CALL( cudaMemcpy2D(dst.data, dst.step, data, step, cols * elemSize(), rows, cudaMemcpyDeviceToHost) );
}

void cv::cuda::GpuMat::download(OutputArray _dst, Stream& _stream) const
{
    CV_DbgAssert( !empty() );

    _dst.create(size(), type());
    Mat dst = _dst.getMat();

    cudaStream_t stream = StreamAccessor::getStream(_stream);
    CV_CUDEV_SAFE_CALL( cudaMemcpy2DAsync(dst.data, dst.step, data, step, cols * elemSize(), rows, cudaMemcpyDeviceToHost, stream) );
}

/////////////////////////////////////////////////////
/// copyTo

void cv::cuda::GpuMat::copyTo(OutputArray _dst) const
{
    CV_DbgAssert( !empty() );

    _dst.create(size(), type());
    GpuMat dst = _dst.getGpuMat();

    CV_CUDEV_SAFE_CALL( cudaMemcpy2D(dst.data, dst.step, data, step, cols * elemSize(), rows, cudaMemcpyDeviceToDevice) );
}

void cv::cuda::GpuMat::copyTo(OutputArray _dst, Stream& _stream) const
{
    CV_DbgAssert( !empty() );

    _dst.create(size(), type());
    GpuMat dst = _dst.getGpuMat();

    cudaStream_t stream = StreamAccessor::getStream(_stream);
    CV_CUDEV_SAFE_CALL( cudaMemcpy2DAsync(dst.data, dst.step, data, step, cols * elemSize(), rows, cudaMemcpyDeviceToDevice, stream) );
}

namespace
{
    template <size_t size> struct CopyToPolicy : DefaultTransformPolicy
    {
    };
    template <> struct CopyToPolicy<4> : DefaultTransformPolicy
    {
        enum {
            shift = 2
        };
    };
    template <> struct CopyToPolicy<8> : DefaultTransformPolicy
    {
        enum {
            shift = 1
        };
    };

    template <typename T>
    void copyWithMask(const GpuMat& src, const GpuMat& dst, const GpuMat& mask, Stream& stream)
    {
        gridTransformUnary_< CopyToPolicy<sizeof(typename VecTraits<T>::elem_type)> >(globPtr<T>(src), globPtr<T>(dst), identity<T>(), globPtr<uchar>(mask), stream);
    }
}

void cv::cuda::GpuMat::copyTo(OutputArray _dst, InputArray _mask, Stream& stream) const
{
    CV_DbgAssert( !empty() );
    CV_DbgAssert( depth() <= CV_64F && channels() <= 4 );

    GpuMat mask = _mask.getGpuMat();
    CV_DbgAssert( size() == mask.size() && mask.depth() == CV_8U && (mask.channels() == 1 || mask.channels() == channels()) );

    uchar* data0 = _dst.getGpuMat().data;

    _dst.create(size(), type());
    GpuMat dst = _dst.getGpuMat();

    // do not leave dst uninitialized
    if (dst.data != data0)
        dst.setTo(Scalar::all(0), stream);

    typedef void (*func_t)(const GpuMat& src, const GpuMat& dst, const GpuMat& mask, Stream& stream);
    static const func_t funcs[9][4] =
    {
        {0,0,0,0},
        {copyWithMask<uchar>, copyWithMask<uchar2>, copyWithMask<uchar3>, copyWithMask<uchar4>},
        {copyWithMask<ushort>, copyWithMask<ushort2>, copyWithMask<ushort3>, copyWithMask<ushort4>},
        {0,0,0,0},
        {copyWithMask<int>, copyWithMask<int2>, copyWithMask<int3>, copyWithMask<int4>},
        {0,0,0,0},
        {0,0,0,0},
        {0,0,0,0},
        {copyWithMask<double>, copyWithMask<double2>, copyWithMask<double3>, copyWithMask<double4>}
    };

    if (mask.channels() == channels())
    {
        const func_t func = funcs[elemSize1()][0];
        CV_DbgAssert( func != 0 );
        func(reshape(1), dst.reshape(1), mask.reshape(1), stream);
    }
    else
    {
        const func_t func = funcs[elemSize1()][channels() - 1];
        CV_DbgAssert( func != 0 );
        func(*this, dst, mask, stream);
    }
}

/////////////////////////////////////////////////////
/// setTo

namespace
{
    template <typename T>
    void setToWithOutMask(const GpuMat& mat, Scalar _scalar, Stream& stream)
    {
        Scalar_<typename VecTraits<T>::elem_type> scalar = _scalar;
        gridTransformUnary(constantPtr(VecTraits<T>::make(scalar.val), mat.rows, mat.cols), globPtr<T>(mat), identity<T>(), stream);
    }

    template <typename T>
    void setToWithMask(const GpuMat& mat, const GpuMat& mask, Scalar _scalar, Stream& stream)
    {
        Scalar_<typename VecTraits<T>::elem_type> scalar = _scalar;
        gridTransformUnary(constantPtr(VecTraits<T>::make(scalar.val), mat.rows, mat.cols), globPtr<T>(mat), identity<T>(), globPtr<uchar>(mask), stream);
    }
}

GpuMat& cv::cuda::GpuMat::setTo(Scalar value, Stream& stream)
{
    CV_DbgAssert( !empty() );
    CV_DbgAssert( depth() <= CV_64F && channels() <= 4 );

    if (value[0] == 0.0 && value[1] == 0.0 && value[2] == 0.0 && value[3] == 0.0)
    {
        // Zero fill

        if (stream)
            CV_CUDEV_SAFE_CALL( cudaMemset2DAsync(data, step, 0, cols * elemSize(), rows, StreamAccessor::getStream(stream)) );
        else
            CV_CUDEV_SAFE_CALL( cudaMemset2D(data, step, 0, cols * elemSize(), rows) );

        return *this;
    }

    if (depth() == CV_8U)
    {
        const int cn = channels();

        if (cn == 1
                || (cn == 2 && value[0] == value[1])
                || (cn == 3 && value[0] == value[1] && value[0] == value[2])
                || (cn == 4 && value[0] == value[1] && value[0] == value[2] && value[0] == value[3]))
        {
            const int val = cv::saturate_cast<uchar>(value[0]);

            if (stream)
                CV_CUDEV_SAFE_CALL( cudaMemset2DAsync(data, step, val, cols * elemSize(), rows, StreamAccessor::getStream(stream)) );
            else
                CV_CUDEV_SAFE_CALL( cudaMemset2D(data, step, val, cols * elemSize(), rows) );

            return *this;
        }
    }

    typedef void (*func_t)(const GpuMat& mat, Scalar scalar, Stream& stream);
    static const func_t funcs[7][4] =
    {
        {setToWithOutMask<uchar>,setToWithOutMask<uchar2>,setToWithOutMask<uchar3>,setToWithOutMask<uchar4>},
        {setToWithOutMask<schar>,setToWithOutMask<char2>,setToWithOutMask<char3>,setToWithOutMask<char4>},
        {setToWithOutMask<ushort>,setToWithOutMask<ushort2>,setToWithOutMask<ushort3>,setToWithOutMask<ushort4>},
        {setToWithOutMask<short>,setToWithOutMask<short2>,setToWithOutMask<short3>,setToWithOutMask<short4>},
        {setToWithOutMask<int>,setToWithOutMask<int2>,setToWithOutMask<int3>,setToWithOutMask<int4>},
        {setToWithOutMask<float>,setToWithOutMask<float2>,setToWithOutMask<float3>,setToWithOutMask<float4>},
        {setToWithOutMask<double>,setToWithOutMask<double2>,setToWithOutMask<double3>,setToWithOutMask<double4>}
    };

    funcs[depth()][channels() - 1](*this, value, stream);

    return *this;
}

GpuMat& cv::cuda::GpuMat::setTo(Scalar value, InputArray _mask, Stream& stream)
{
    CV_DbgAssert( !empty() );
    CV_DbgAssert( depth() <= CV_64F && channels() <= 4 );

    GpuMat mask = _mask.getGpuMat();

    if (mask.empty())
    {
        return setTo(value, stream);
    }

    CV_DbgAssert( size() == mask.size() && mask.type() == CV_8UC1 );

    typedef void (*func_t)(const GpuMat& mat, const GpuMat& mask, Scalar scalar, Stream& stream);
    static const func_t funcs[7][4] =
    {
        {setToWithMask<uchar>,setToWithMask<uchar2>,setToWithMask<uchar3>,setToWithMask<uchar4>},
        {setToWithMask<schar>,setToWithMask<char2>,setToWithMask<char3>,setToWithMask<char4>},
        {setToWithMask<ushort>,setToWithMask<ushort2>,setToWithMask<ushort3>,setToWithMask<ushort4>},
        {setToWithMask<short>,setToWithMask<short2>,setToWithMask<short3>,setToWithMask<short4>},
        {setToWithMask<int>,setToWithMask<int2>,setToWithMask<int3>,setToWithMask<int4>},
        {setToWithMask<float>,setToWithMask<float2>,setToWithMask<float3>,setToWithMask<float4>},
        {setToWithMask<double>,setToWithMask<double2>,setToWithMask<double3>,setToWithMask<double4>}
    };

    funcs[depth()][channels() - 1](*this, mask, value, stream);

    return *this;
}

/////////////////////////////////////////////////////
/// convertTo

namespace
{
    template <typename T> struct ConvertToPolicy : DefaultTransformPolicy
    {
    };
    template <> struct ConvertToPolicy<double> : DefaultTransformPolicy
    {
        enum {
            shift = 1
        };
    };

    template <typename T, typename D>
    void convertToNoScale(const GpuMat& src, const GpuMat& dst, Stream& stream)
    {
        typedef typename VecTraits<T>::elem_type src_elem_type;
        typedef typename VecTraits<D>::elem_type dst_elem_type;
        typedef typename LargerType<src_elem_type, float>::type larger_elem_type;
        typedef typename LargerType<float, dst_elem_type>::type scalar_type;

        gridTransformUnary_< ConvertToPolicy<scalar_type> >(globPtr<T>(src), globPtr<D>(dst), saturate_cast_func<T, D>(), stream);
    }

    template <typename T, typename D, typename S> struct Convertor : unary_function<T, D>
    {
        S alpha;
        S beta;

        __device__ __forceinline__ D operator ()(typename TypeTraits<T>::parameter_type src) const
        {
            return cudev::saturate_cast<D>(alpha * src + beta);
        }
    };

    template <typename T, typename D>
    void convertToScale(const GpuMat& src, const GpuMat& dst, double alpha, double beta, Stream& stream)
    {
        typedef typename VecTraits<T>::elem_type src_elem_type;
        typedef typename VecTraits<D>::elem_type dst_elem_type;
        typedef typename LargerType<src_elem_type, float>::type larger_elem_type;
        typedef typename LargerType<float, dst_elem_type>::type scalar_type;

        Convertor<T, D, scalar_type> op;
        op.alpha = cv::saturate_cast<scalar_type>(alpha);
        op.beta = cv::saturate_cast<scalar_type>(beta);

        gridTransformUnary_< ConvertToPolicy<scalar_type> >(globPtr<T>(src), globPtr<D>(dst), op, stream);
    }

    template <typename T, typename D>
    void convertScaleHalf(const GpuMat& src, const GpuMat& dst, Stream& stream)
    {
        typedef typename VecTraits<T>::elem_type src_elem_type;
        typedef typename VecTraits<D>::elem_type dst_elem_type;
        typedef typename LargerType<src_elem_type, float>::type larger_elem_type;
        typedef typename LargerType<float, dst_elem_type>::type scalar_type;

        gridTransformUnary_< ConvertToPolicy<scalar_type> >(globPtr<T>(src), globPtr<D>(dst), saturate_cast_fp16_func<T,D>(), stream);
    }
}

void cv::cuda::GpuMat::convertTo(OutputArray _dst, int rtype, Stream& stream) const
{
    if (rtype < 0)
        rtype = type();
    else
        rtype = CV_MAKE_TYPE(CV_MAT_DEPTH(rtype), channels());

    const int sdepth = depth();
    const int ddepth = CV_MAT_DEPTH(rtype);
    if (sdepth == ddepth)
    {
        if (stream)
            copyTo(_dst, stream);
        else
            copyTo(_dst);

        return;
    }

    CV_DbgAssert( sdepth <= CV_64F && ddepth <= CV_64F );

    GpuMat src = *this;

    _dst.create(size(), rtype);
    GpuMat dst = _dst.getGpuMat();

    typedef void (*func_t)(const GpuMat& src, const GpuMat& dst, Stream& stream);
    static const func_t funcs[7][7] =
    {
        {0, convertToNoScale<uchar, schar>, convertToNoScale<uchar, ushort>, convertToNoScale<uchar, short>, convertToNoScale<uchar, int>, convertToNoScale<uchar, float>, convertToNoScale<uchar, double>},
        {convertToNoScale<schar, uchar>, 0, convertToNoScale<schar, ushort>, convertToNoScale<schar, short>, convertToNoScale<schar, int>, convertToNoScale<schar, float>, convertToNoScale<schar, double>},
        {convertToNoScale<ushort, uchar>, convertToNoScale<ushort, schar>, 0, convertToNoScale<ushort, short>, convertToNoScale<ushort, int>, convertToNoScale<ushort, float>, convertToNoScale<ushort, double>},
        {convertToNoScale<short, uchar>, convertToNoScale<short, schar>, convertToNoScale<short, ushort>, 0, convertToNoScale<short, int>, convertToNoScale<short, float>, convertToNoScale<short, double>},
        {convertToNoScale<int, uchar>, convertToNoScale<int, schar>, convertToNoScale<int, ushort>, convertToNoScale<int, short>, 0, convertToNoScale<int, float>, convertToNoScale<int, double>},
        {convertToNoScale<float, uchar>, convertToNoScale<float, schar>, convertToNoScale<float, ushort>, convertToNoScale<float, short>, convertToNoScale<float, int>, 0, convertToNoScale<float, double>},
        {convertToNoScale<double, uchar>, convertToNoScale<double, schar>, convertToNoScale<double, ushort>, convertToNoScale<double, short>, convertToNoScale<double, int>, convertToNoScale<double, float>, 0}
    };

    funcs[sdepth][ddepth](src.reshape(1), dst.reshape(1), stream);
}

void cv::cuda::GpuMat::convertTo(OutputArray _dst, int rtype, double alpha, double beta, Stream& stream) const
{
    if (rtype < 0)
        rtype = type();
    else
        rtype = CV_MAKETYPE(CV_MAT_DEPTH(rtype), channels());

    const int sdepth = depth();
    const int ddepth = CV_MAT_DEPTH(rtype);

    GpuMat src = *this;

    _dst.create(size(), rtype);
    GpuMat dst = _dst.getGpuMat();

    typedef void (*func_t)(const GpuMat& src, const GpuMat& dst, double alpha, double beta, Stream& stream);
    static const func_t funcs[7][7] =
    {
        {convertToScale<uchar, uchar>, convertToScale<uchar, schar>, convertToScale<uchar, ushort>, convertToScale<uchar, short>, convertToScale<uchar, int>, convertToScale<uchar, float>, convertToScale<uchar, double>},
        {convertToScale<schar, uchar>, convertToScale<schar, schar>, convertToScale<schar, ushort>, convertToScale<schar, short>, convertToScale<schar, int>, convertToScale<schar, float>, convertToScale<schar, double>},
        {convertToScale<ushort, uchar>, convertToScale<ushort, schar>, convertToScale<ushort, ushort>, convertToScale<ushort, short>, convertToScale<ushort, int>, convertToScale<ushort, float>, convertToScale<ushort, double>},
        {convertToScale<short, uchar>, convertToScale<short, schar>, convertToScale<short, ushort>, convertToScale<short, short>, convertToScale<short, int>, convertToScale<short, float>, convertToScale<short, double>},
        {convertToScale<int, uchar>, convertToScale<int, schar>, convertToScale<int, ushort>, convertToScale<int, short>, convertToScale<int, int>, convertToScale<int, float>, convertToScale<int, double>},
        {convertToScale<float, uchar>, convertToScale<float, schar>, convertToScale<float, ushort>, convertToScale<float, short>, convertToScale<float, int>, convertToScale<float, float>, convertToScale<float, double>},
        {convertToScale<double, uchar>, convertToScale<double, schar>, convertToScale<double, ushort>, convertToScale<double, short>, convertToScale<double, int>, convertToScale<double, float>, convertToScale<double, double>}
    };

    funcs[sdepth][ddepth](src.reshape(1), dst.reshape(1), alpha, beta, stream);
}

void cv::cuda::convertFp16(InputArray _src, OutputArray _dst, Stream& stream)
{
    GpuMat src = _src.getGpuMat();
    int ddepth = 0;

    switch(src.depth())
    {
    case CV_32F:
        ddepth = CV_16S;
        break;
    case CV_16S:
        ddepth = CV_32F;
        break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "Unsupported input depth");
        return;
    }
    int type = CV_MAKE_TYPE(CV_MAT_DEPTH(ddepth), src.channels());
    _dst.create(src.size(), type);
    GpuMat dst = _dst.getGpuMat();

    typedef void (*func_t)(const GpuMat& src, const GpuMat& dst, Stream& stream);
    static const func_t funcs[] =
    {
        0, 0, 0,
        convertScaleHalf<float, short>, 0, convertScaleHalf<short, float>,
        0, 0,
    };

    funcs[ddepth](src.reshape(1), dst.reshape(1), stream);
}

#endif
