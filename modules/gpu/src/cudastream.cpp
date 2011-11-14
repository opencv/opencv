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

#include "precomp.hpp"

using namespace cv;
using namespace cv::gpu;


#if !defined (HAVE_CUDA)

void cv::gpu::Stream::create() { throw_nogpu(); }
void cv::gpu::Stream::release() { throw_nogpu(); }
cv::gpu::Stream::Stream() : impl(0) { throw_nogpu(); }
cv::gpu::Stream::~Stream() { throw_nogpu(); }
cv::gpu::Stream::Stream(const Stream& /*stream*/) { throw_nogpu(); }
Stream& cv::gpu::Stream::operator=(const Stream& /*stream*/) { throw_nogpu(); return *this; }
bool cv::gpu::Stream::queryIfComplete() { throw_nogpu(); return true; }
void cv::gpu::Stream::waitForCompletion() { throw_nogpu(); }
void cv::gpu::Stream::enqueueDownload(const GpuMat& /*src*/, Mat& /*dst*/) { throw_nogpu(); }
void cv::gpu::Stream::enqueueDownload(const GpuMat& /*src*/, CudaMem& /*dst*/) { throw_nogpu(); }
void cv::gpu::Stream::enqueueUpload(const CudaMem& /*src*/, GpuMat& /*dst*/) { throw_nogpu(); }
void cv::gpu::Stream::enqueueUpload(const Mat& /*src*/, GpuMat& /*dst*/) { throw_nogpu(); }
void cv::gpu::Stream::enqueueCopy(const GpuMat& /*src*/, GpuMat& /*dst*/) { throw_nogpu(); }
void cv::gpu::Stream::enqueueMemSet(GpuMat& /*src*/, Scalar /*val*/) { throw_nogpu(); }
void cv::gpu::Stream::enqueueMemSet(GpuMat& /*src*/, Scalar /*val*/, const GpuMat& /*mask*/) { throw_nogpu(); }
void cv::gpu::Stream::enqueueConvert(const GpuMat& /*src*/, GpuMat& /*dst*/, int /*type*/, double /*a*/, double /*b*/) { throw_nogpu(); }
Stream& cv::gpu::Stream::Null() { throw_nogpu(); static Stream s; return s; }
cv::gpu::Stream::operator bool() const { throw_nogpu(); return false; }

#else /* !defined (HAVE_CUDA) */

#include "opencv2/gpu/stream_accessor.hpp"

namespace cv { namespace gpu { namespace device 
{
    void copy_to_with_mask(const DevMem2Db& src, DevMem2Db dst, int depth, const DevMem2Db& mask, int channels, const cudaStream_t & stream = 0);

    template <typename T>
    void set_to_gpu(const DevMem2Db& mat, const T* scalar, int channels, cudaStream_t stream);
    template <typename T>
    void set_to_gpu(const DevMem2Db& mat, const T* scalar, const DevMem2Db& mask, int channels, cudaStream_t stream);

    void convert_gpu(const DevMem2Db& src, int sdepth, const DevMem2Db& dst, int ddepth, double alpha, double beta, cudaStream_t stream = 0);
}}}

using namespace ::cv::gpu::device;

struct Stream::Impl
{
    cudaStream_t stream;
    int ref_counter;
};

namespace
{
    template<class S, class D> void devcopy(const S& src, D& dst, cudaStream_t s, cudaMemcpyKind k)
    {
        dst.create(src.size(), src.type());
        size_t bwidth = src.cols * src.elemSize();
        cudaSafeCall( cudaMemcpy2DAsync(dst.data, dst.step, src.data, src.step, bwidth, src.rows, k, s) );
    };

    template <typename T>
    void kernelSet(GpuMat& src, const Scalar& s, cudaStream_t stream)
    {
        Scalar_<T> sf = s;
        set_to_gpu(src, sf.val, src.channels(), stream);
    }

    template <typename T>
    void kernelSetMask(GpuMat& src, const Scalar& s, const GpuMat& mask, cudaStream_t stream)
    {
        Scalar_<T> sf = s;
        set_to_gpu(src, sf.val, mask, src.channels(), stream);
    }
}

CV_EXPORTS cudaStream_t cv::gpu::StreamAccessor::getStream(const Stream& stream) { return stream.impl ? stream.impl->stream : 0; };

void cv::gpu::Stream::create()
{
    if (impl)
        release();

    cudaStream_t stream;
    cudaSafeCall( cudaStreamCreate( &stream ) );

    impl = (Stream::Impl*)fastMalloc(sizeof(Stream::Impl));

    impl->stream = stream;
    impl->ref_counter = 1;
}

void cv::gpu::Stream::release()
{
    if( impl && CV_XADD(&impl->ref_counter, -1) == 1 )
    {
        cudaSafeCall( cudaStreamDestroy( impl->stream ) );
        cv::fastFree( impl );
    }
}

cv::gpu::Stream::Stream() : impl(0) { create(); }
cv::gpu::Stream::~Stream() { release(); }

cv::gpu::Stream::Stream(const Stream& stream) : impl(stream.impl)
{
    if( impl )
        CV_XADD(&impl->ref_counter, 1);
}
Stream& cv::gpu::Stream::operator=(const Stream& stream)
{
    if( this != &stream )
    {
        if( stream.impl )
            CV_XADD(&stream.impl->ref_counter, 1);

        release();
        impl = stream.impl;
    }
    return *this;
}

bool cv::gpu::Stream::queryIfComplete()
{
    cudaError_t err = cudaStreamQuery( impl->stream );

    if (err == cudaErrorNotReady || err == cudaSuccess)
        return err == cudaSuccess;

    cudaSafeCall(err);
    return false;
}

void cv::gpu::Stream::waitForCompletion() { cudaSafeCall( cudaStreamSynchronize( impl->stream ) ); }

void cv::gpu::Stream::enqueueDownload(const GpuMat& src, Mat& dst)
{
    // if not -> allocation will be done, but after that dst will not point to page locked memory
    CV_Assert(src.cols == dst.cols && src.rows == dst.rows && src.type() == dst.type() );
    devcopy(src, dst, impl->stream, cudaMemcpyDeviceToHost);
}
void cv::gpu::Stream::enqueueDownload(const GpuMat& src, CudaMem& dst) { devcopy(src, dst, impl->stream, cudaMemcpyDeviceToHost); }

void cv::gpu::Stream::enqueueUpload(const CudaMem& src, GpuMat& dst){ devcopy(src, dst, impl->stream,   cudaMemcpyHostToDevice); }
void cv::gpu::Stream::enqueueUpload(const Mat& src, GpuMat& dst)  { devcopy(src, dst, impl->stream,   cudaMemcpyHostToDevice); }
void cv::gpu::Stream::enqueueCopy(const GpuMat& src, GpuMat& dst) { devcopy(src, dst, impl->stream, cudaMemcpyDeviceToDevice); }

void cv::gpu::Stream::enqueueMemSet(GpuMat& src, Scalar s)
{
    CV_Assert((src.depth() != CV_64F) || 
        (TargetArchs::builtWith(NATIVE_DOUBLE) && DeviceInfo().supports(NATIVE_DOUBLE)));

    if (s[0] == 0.0 && s[1] == 0.0 && s[2] == 0.0 && s[3] == 0.0)
    {
        cudaSafeCall( cudaMemset2DAsync(src.data, src.step, 0, src.cols * src.elemSize(), src.rows, impl->stream) );
        return;
    }
    if (src.depth() == CV_8U)
    {
        int cn = src.channels();

        if (cn == 1 || (cn == 2 && s[0] == s[1]) || (cn == 3 && s[0] == s[1] && s[0] == s[2]) || (cn == 4 && s[0] == s[1] && s[0] == s[2] && s[0] == s[3]))
        {
            int val = saturate_cast<uchar>(s[0]);
            cudaSafeCall( cudaMemset2DAsync(src.data, src.step, val, src.cols * src.elemSize(), src.rows, impl->stream) );
            return;
        }
    }

    typedef void (*set_caller_t)(GpuMat& src, const Scalar& s, cudaStream_t stream);
    static const set_caller_t set_callers[] =
    {
        kernelSet<uchar>, kernelSet<schar>, kernelSet<ushort>, kernelSet<short>,
        kernelSet<int>, kernelSet<float>, kernelSet<double>
    };
    set_callers[src.depth()](src, s, impl->stream);
}

void cv::gpu::Stream::enqueueMemSet(GpuMat& src, Scalar val, const GpuMat& mask)
{
    CV_Assert((src.depth() != CV_64F) || 
        (TargetArchs::builtWith(NATIVE_DOUBLE) && DeviceInfo().supports(NATIVE_DOUBLE)));

    CV_Assert(mask.type() == CV_8UC1);

    typedef void (*set_caller_t)(GpuMat& src, const Scalar& s, const GpuMat& mask, cudaStream_t stream);
    static const set_caller_t set_callers[] =
    {
        kernelSetMask<uchar>, kernelSetMask<schar>, kernelSetMask<ushort>, kernelSetMask<short>,
        kernelSetMask<int>, kernelSetMask<float>, kernelSetMask<double>
    };
    set_callers[src.depth()](src, val, mask, impl->stream);
}

void cv::gpu::Stream::enqueueConvert(const GpuMat& src, GpuMat& dst, int rtype, double alpha, double beta)
{
    CV_Assert((src.depth() != CV_64F && CV_MAT_DEPTH(rtype) != CV_64F) || 
        (TargetArchs::builtWith(NATIVE_DOUBLE) && DeviceInfo().supports(NATIVE_DOUBLE)));

    bool noScale = fabs(alpha-1) < std::numeric_limits<double>::epsilon() && fabs(beta) < std::numeric_limits<double>::epsilon();

    if( rtype < 0 )
        rtype = src.type();
    else
        rtype = CV_MAKETYPE(CV_MAT_DEPTH(rtype), src.channels());

    int sdepth = src.depth(), ddepth = CV_MAT_DEPTH(rtype);
    if( sdepth == ddepth && noScale )
    {
        src.copyTo(dst);
        return;
    }

    GpuMat temp;
    const GpuMat* psrc = &src;
    if( sdepth != ddepth && psrc == &dst )
        psrc = &(temp = src);

    dst.create( src.size(), rtype );
    convert_gpu(psrc->reshape(1), sdepth, dst.reshape(1), ddepth, alpha, beta, impl->stream);
}

cv::gpu::Stream::operator bool() const
{
    return impl && impl->stream;
}

cv::gpu::Stream::Stream(Impl* impl_) : impl(impl_) {}

cv::gpu::Stream& cv::gpu::Stream::Null()
{
    static Stream s((Impl*)0);
    return s;
}

#endif /* !defined (HAVE_CUDA) */
