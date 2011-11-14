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
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or bpied warranties, including, but not limited to, the bpied
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

void cv::gpu::add(const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::add(const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::subtract(const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::subtract(const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::multiply(const GpuMat&, const GpuMat&, GpuMat&, double, int, Stream&) { throw_nogpu(); }
void cv::gpu::multiply(const GpuMat&, const Scalar&, GpuMat&, double, int, Stream&) { throw_nogpu(); }
void cv::gpu::divide(const GpuMat&, const GpuMat&, GpuMat&, double, int, Stream&) { throw_nogpu(); }
void cv::gpu::divide(const GpuMat&, const Scalar&, GpuMat&, double, int, Stream&) { throw_nogpu(); }
void cv::gpu::divide(double, const GpuMat&, GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::absdiff(const GpuMat&, const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::absdiff(const GpuMat&, const Scalar&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::compare(const GpuMat&, const GpuMat&, GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::bitwise_not(const GpuMat&, GpuMat&, const GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::bitwise_or(const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::bitwise_and(const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::bitwise_xor(const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::min(const GpuMat&, const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::min(const GpuMat&, double, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::max(const GpuMat&, const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::max(const GpuMat&, double, GpuMat&, Stream&) { throw_nogpu(); }
double cv::gpu::threshold(const GpuMat&, GpuMat&, double, double, int, Stream&) {throw_nogpu(); return 0.0;}
void cv::gpu::pow(const GpuMat&, double, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::addWeighted(const GpuMat&, double, const GpuMat&, double, double, GpuMat&, int, Stream&) { throw_nogpu(); }

#else

////////////////////////////////////////////////////////////////////////
// Basic arithmetical operations (add subtract multiply divide)

namespace
{
    typedef NppStatus (*npp_arithm_8u_t)(const Npp8u* pSrc1, int nSrc1Step, const Npp8u* pSrc2, int nSrc2Step, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    typedef NppStatus (*npp_arithm_32s_t)(const Npp32s* pSrc1, int nSrc1Step, const Npp32s* pSrc2, int nSrc2Step, Npp32s* pDst, int nDstStep, NppiSize oSizeROI);
    typedef NppStatus (*npp_arithm_32f_t)(const Npp32f* pSrc1, int nSrc1Step, const Npp32f* pSrc2, int nSrc2Step, Npp32f* pDst, int nDstStep, NppiSize oSizeROI);

    void nppArithmCaller(const GpuMat& src1, const GpuMat& src2, GpuMat& dst,
                         npp_arithm_8u_t npp_func_8uc1, npp_arithm_8u_t npp_func_8uc4,
                         npp_arithm_32s_t npp_func_32sc1, npp_arithm_32f_t npp_func_32fc1, cudaStream_t stream)
    {
        NppiSize sz;
        sz.width  = src1.cols * src1.channels();
        sz.height = src1.rows;

        NppStreamHandler h(stream);

        if (src1.depth() == CV_8U && (src1.cols * src1.channels()) % 4 == 0)
        {
            sz.width /= 4;

            nppSafeCall( npp_func_8uc4(src1.ptr<Npp8u>(), static_cast<int>(src1.step), src2.ptr<Npp8u>(), static_cast<int>(src2.step), 
                dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz, 0) );
        }
        else if (src1.depth() == CV_8U)
        {
            nppSafeCall( npp_func_8uc1(src1.ptr<Npp8u>(), static_cast<int>(src1.step), src2.ptr<Npp8u>(), static_cast<int>(src2.step), 
                dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz, 0) );
        }
        else if (src1.depth() == CV_32S)
        {
            nppSafeCall( npp_func_32sc1(src1.ptr<Npp32s>(), static_cast<int>(src1.step), src2.ptr<Npp32s>(), static_cast<int>(src2.step), 
                dst.ptr<Npp32s>(), static_cast<int>(dst.step), sz) );
        }
        else if (src1.depth() == CV_32F)
        {
            nppSafeCall( npp_func_32fc1(src1.ptr<Npp32f>(), static_cast<int>(src1.step), src2.ptr<Npp32f>(), static_cast<int>(src2.step), 
                dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz) );
        }

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
}

////////////////////////////////////////////////////////////////////////
// add

namespace cv { namespace gpu { namespace device 
{
    template <typename T, typename D> 
    void add_gpu(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    template <typename T, typename D> 
    void add_gpu(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
}}}

void cv::gpu::add(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, int dtype, Stream& s)
{
    using namespace ::cv::gpu::device;

    typedef void (*func_t)(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    static const func_t funcs[7][7] = 
    {
        {add_gpu<unsigned char, unsigned char>, 0/*add_gpu<unsigned char, signed char>*/, add_gpu<unsigned char, unsigned short>, add_gpu<unsigned char, short>, add_gpu<unsigned char, int>, add_gpu<unsigned char, float>, add_gpu<unsigned char, double>},
        {0/*add_gpu<signed char, unsigned char>*/, 0/*add_gpu<signed char, signed char>*/, 0/*add_gpu<signed char, unsigned short>*/, 0/*add_gpu<signed char, short>*/, 0/*add_gpu<signed char, int>*/, 0/*add_gpu<signed char, float>*/, 0/*add_gpu<signed char, double>*/},
        {0/*add_gpu<unsigned short, unsigned char>*/, 0/*add_gpu<unsigned short, signed char>*/, add_gpu<unsigned short, unsigned short>, 0/*add_gpu<unsigned short, short>*/, add_gpu<unsigned short, int>, add_gpu<unsigned short, float>, add_gpu<unsigned short, double>},
        {0/*add_gpu<short, unsigned char>*/, 0/*add_gpu<short, signed char>*/, 0/*add_gpu<short, unsigned short>*/, add_gpu<short, short>, add_gpu<short, int>, add_gpu<short, float>, add_gpu<short, double>},
        {0/*add_gpu<int, unsigned char>*/, 0/*add_gpu<int, signed char>*/, 0/*add_gpu<int, unsigned short>*/, 0/*add_gpu<int, short>*/, add_gpu<int, int>, add_gpu<int, float>, add_gpu<int, double>},
        {0/*add_gpu<float, unsigned char>*/, 0/*add_gpu<float, signed char>*/, 0/*add_gpu<float, unsigned short>*/, 0/*add_gpu<float, short>*/, 0/*add_gpu<float, int>*/, add_gpu<float, float>, add_gpu<float, double>},
        {0/*add_gpu<double, unsigned char>*/, 0/*add_gpu<double, signed char>*/, 0/*add_gpu<double, unsigned short>*/, 0/*add_gpu<double, short>*/, 0/*add_gpu<double, int>*/, 0/*add_gpu<double, float>*/, add_gpu<double, double>}
    };

    CV_Assert(src1.type() == src2.type() && src1.size() == src2.size());
    CV_Assert(mask.empty() || (src1.channels() == 1 && mask.size() == src1.size() && mask.type() == CV_8U));

    if (dtype < 0)
        dtype = src1.depth();

    dst.create(src1.size(), CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src1.channels()));

    cudaStream_t stream = StreamAccessor::getStream(s);

    if (mask.empty() && dst.type() == src1.type() && (src1.depth() == CV_8U || src1.depth() == CV_32S || src1.depth() == CV_32F))
    {
        nppArithmCaller(src1, src2, dst, nppiAdd_8u_C1RSfs, nppiAdd_8u_C4RSfs, nppiAdd_32s_C1R, nppiAdd_32f_C1R, stream);
        return;
    }

    const func_t func = funcs[src1.depth()][dst.depth()];
    CV_Assert(func != 0);

    func(src1.reshape(1), src2.reshape(1), dst.reshape(1), mask, stream);
}

void cv::gpu::add(const GpuMat& src, const Scalar& sc, GpuMat& dst, const GpuMat& mask, int dtype, Stream& s)
{
    using namespace ::cv::gpu::device;

    typedef void (*func_t)(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    static const func_t funcs[7][7] = 
    {
        {add_gpu<unsigned char, unsigned char>, 0/*add_gpu<unsigned char, signed char>*/, add_gpu<unsigned char, unsigned short>, add_gpu<unsigned char, short>, add_gpu<unsigned char, int>, add_gpu<unsigned char, float>, add_gpu<unsigned char, double>},
        {0/*add_gpu<signed char, unsigned char>*/, 0/*add_gpu<signed char, signed char>*/, 0/*add_gpu<signed char, unsigned short>*/, 0/*add_gpu<signed char, short>*/, 0/*add_gpu<signed char, int>*/, 0/*add_gpu<signed char, float>*/, 0/*add_gpu<signed char, double>*/},
        {0/*add_gpu<unsigned short, unsigned char>*/, 0/*add_gpu<unsigned short, signed char>*/, add_gpu<unsigned short, unsigned short>, 0/*add_gpu<unsigned short, short>*/, add_gpu<unsigned short, int>, add_gpu<unsigned short, float>, add_gpu<unsigned short, double>},
        {0/*add_gpu<short, unsigned char>*/, 0/*add_gpu<short, signed char>*/, 0/*add_gpu<short, unsigned short>*/, add_gpu<short, short>, add_gpu<short, int>, add_gpu<short, float>, add_gpu<short, double>},
        {0/*add_gpu<int, unsigned char>*/, 0/*add_gpu<int, signed char>*/, 0/*add_gpu<int, unsigned short>*/, 0/*add_gpu<int, short>*/, add_gpu<int, int>, add_gpu<int, float>, add_gpu<int, double>},
        {0/*add_gpu<float, unsigned char>*/, 0/*add_gpu<float, signed char>*/, 0/*add_gpu<float, unsigned short>*/, 0/*add_gpu<float, short>*/, 0/*add_gpu<float, int>*/, add_gpu<float, float>, add_gpu<float, double>},
        {0/*add_gpu<double, unsigned char>*/, 0/*add_gpu<double, signed char>*/, 0/*add_gpu<double, unsigned short>*/, 0/*add_gpu<double, short>*/, 0/*add_gpu<double, int>*/, 0/*add_gpu<double, float>*/, add_gpu<double, double>}
    };

    CV_Assert(src.channels() == 1 || src.type() == CV_32FC2);
    CV_Assert(mask.empty() || (src.channels() == 1 && mask.size() == src.size() && mask.type() == CV_8U));

    if (dtype < 0)
        dtype = src.depth();

    dst.create(src.size(), CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src.channels()));

    cudaStream_t stream = StreamAccessor::getStream(s);

    if (mask.empty() && dst.type() == src.type() && src.depth() == CV_32F)
    {
        NppiSize sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        NppStreamHandler h(stream);

        if (src.type() == CV_32FC1)
        {
            nppSafeCall( nppiAddC_32f_C1R(src.ptr<Npp32f>(), static_cast<int>(src.step), static_cast<Npp32f>(sc.val[0]), 
                dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz) );
        }
        else
        {
            Npp32fc nValue;
            nValue.re = static_cast<Npp32f>(sc.val[0]);
            nValue.im = static_cast<Npp32f>(sc.val[1]);
            nppSafeCall( nppiAddC_32fc_C1R(src.ptr<Npp32fc>(), static_cast<int>(src.step), nValue, 
                dst.ptr<Npp32fc>(), static_cast<int>(dst.step), sz) );
        }

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );

        return;
    }

    const func_t func = funcs[src.depth()][dst.depth()];
    CV_Assert(func != 0);

    func(src, sc.val[0], dst, mask, stream);
}

////////////////////////////////////////////////////////////////////////
// subtract

namespace cv { namespace gpu { namespace device 
{
    template <typename T, typename D> 
    void subtract_gpu(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    template <typename T, typename D> 
    void subtract_gpu(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);
}}}

void cv::gpu::subtract(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, int dtype, Stream& s)
{
    using namespace ::cv::gpu::device;

    typedef void (*func_t)(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    static const func_t funcs[7][7] = 
    {
        {subtract_gpu<unsigned char, unsigned char>, 0/*subtract_gpu<unsigned char, signed char>*/, subtract_gpu<unsigned char, unsigned short>, subtract_gpu<unsigned char, short>, subtract_gpu<unsigned char, int>, subtract_gpu<unsigned char, float>, subtract_gpu<unsigned char, double>},
        {0/*subtract_gpu<signed char, unsigned char>*/, 0/*subtract_gpu<signed char, signed char>*/, 0/*subtract_gpu<signed char, unsigned short>*/, 0/*subtract_gpu<signed char, short>*/, 0/*subtract_gpu<signed char, int>*/, 0/*subtract_gpu<signed char, float>*/, 0/*subtract_gpu<signed char, double>*/},
        {0/*subtract_gpu<unsigned short, unsigned char>*/, 0/*subtract_gpu<unsigned short, signed char>*/, subtract_gpu<unsigned short, unsigned short>, 0/*subtract_gpu<unsigned short, short>*/, subtract_gpu<unsigned short, int>, subtract_gpu<unsigned short, float>, subtract_gpu<unsigned short, double>},
        {0/*subtract_gpu<short, unsigned char>*/, 0/*subtract_gpu<short, signed char>*/, 0/*subtract_gpu<short, unsigned short>*/, subtract_gpu<short, short>, subtract_gpu<short, int>, subtract_gpu<short, float>, subtract_gpu<short, double>},
        {0/*subtract_gpu<int, unsigned char>*/, 0/*subtract_gpu<int, signed char>*/, 0/*subtract_gpu<int, unsigned short>*/, 0/*subtract_gpu<int, short>*/, subtract_gpu<int, int>, subtract_gpu<int, float>, subtract_gpu<int, double>},
        {0/*subtract_gpu<float, unsigned char>*/, 0/*subtract_gpu<float, signed char>*/, 0/*subtract_gpu<float, unsigned short>*/, 0/*subtract_gpu<float, short>*/, 0/*subtract_gpu<float, int>*/, subtract_gpu<float, float>, subtract_gpu<float, double>},
        {0/*subtract_gpu<double, unsigned char>*/, 0/*subtract_gpu<double, signed char>*/, 0/*subtract_gpu<double, unsigned short>*/, 0/*subtract_gpu<double, short>*/, 0/*subtract_gpu<double, int>*/, 0/*subtract_gpu<double, float>*/, subtract_gpu<double, double>}
    };

    CV_Assert(src1.type() == src2.type() && src1.size() == src2.size());
    CV_Assert(mask.empty() || (src1.channels() == 1 && mask.size() == src1.size() && mask.type() == CV_8U));

    if (dtype < 0)
        dtype = src1.depth();

    dst.create(src1.size(), CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src1.channels()));

    cudaStream_t stream = StreamAccessor::getStream(s);

    if (mask.empty() && dst.type() == src1.type() && (src1.depth() == CV_8U || src1.depth() == CV_32S || src1.depth() == CV_32F))
    {
        nppArithmCaller(src2, src1, dst, nppiSub_8u_C1RSfs, nppiSub_8u_C4RSfs, nppiSub_32s_C1R, nppiSub_32f_C1R, stream);
        return;
    }

    const func_t func = funcs[src1.depth()][dst.depth()];
    CV_Assert(func != 0);

    func(src1.reshape(1), src2.reshape(1), dst.reshape(1), mask, stream);
}

void cv::gpu::subtract(const GpuMat& src, const Scalar& sc, GpuMat& dst, const GpuMat& mask, int dtype, Stream& s)
{
    using namespace ::cv::gpu::device;

    typedef void (*func_t)(const DevMem2Db& src1, double val, const DevMem2Db& dst, const PtrStepb& mask, cudaStream_t stream);

    static const func_t funcs[7][7] = 
    {
        {subtract_gpu<unsigned char, unsigned char>, 0/*subtract_gpu<unsigned char, signed char>*/, subtract_gpu<unsigned char, unsigned short>, subtract_gpu<unsigned char, short>, subtract_gpu<unsigned char, int>, subtract_gpu<unsigned char, float>, subtract_gpu<unsigned char, double>},
        {0/*subtract_gpu<signed char, unsigned char>*/, 0/*subtract_gpu<signed char, signed char>*/, 0/*subtract_gpu<signed char, unsigned short>*/, 0/*subtract_gpu<signed char, short>*/, 0/*subtract_gpu<signed char, int>*/, 0/*subtract_gpu<signed char, float>*/, 0/*subtract_gpu<signed char, double>*/},
        {0/*subtract_gpu<unsigned short, unsigned char>*/, 0/*subtract_gpu<unsigned short, signed char>*/, subtract_gpu<unsigned short, unsigned short>, 0/*subtract_gpu<unsigned short, short>*/, subtract_gpu<unsigned short, int>, subtract_gpu<unsigned short, float>, subtract_gpu<unsigned short, double>},
        {0/*subtract_gpu<short, unsigned char>*/, 0/*subtract_gpu<short, signed char>*/, 0/*subtract_gpu<short, unsigned short>*/, subtract_gpu<short, short>, subtract_gpu<short, int>, subtract_gpu<short, float>, subtract_gpu<short, double>},
        {0/*subtract_gpu<int, unsigned char>*/, 0/*subtract_gpu<int, signed char>*/, 0/*subtract_gpu<int, unsigned short>*/, 0/*subtract_gpu<int, short>*/, subtract_gpu<int, int>, subtract_gpu<int, float>, subtract_gpu<int, double>},
        {0/*subtract_gpu<float, unsigned char>*/, 0/*subtract_gpu<float, signed char>*/, 0/*subtract_gpu<float, unsigned short>*/, 0/*subtract_gpu<float, short>*/, 0/*subtract_gpu<float, int>*/, subtract_gpu<float, float>, subtract_gpu<float, double>},
        {0/*subtract_gpu<double, unsigned char>*/, 0/*subtract_gpu<double, signed char>*/, 0/*subtract_gpu<double, unsigned short>*/, 0/*subtract_gpu<double, short>*/, 0/*subtract_gpu<double, int>*/, 0/*subtract_gpu<double, float>*/, subtract_gpu<double, double>}
    };

    CV_Assert(src.channels() == 1 || src.type() == CV_32FC2);
    CV_Assert(mask.empty() || (src.channels() == 1 && mask.size() == src.size() && mask.type() == CV_8U));

    if (dtype < 0)
        dtype = src.depth();

    dst.create(src.size(), CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src.channels()));

    cudaStream_t stream = StreamAccessor::getStream(s);

    if (mask.empty() && dst.type() == src.type() && src.depth() == CV_32F)
    {
        NppiSize sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        NppStreamHandler h(stream);

        if (src.type() == CV_32FC1)
        {
            nppSafeCall( nppiSubC_32f_C1R(src.ptr<Npp32f>(), static_cast<int>(src.step), static_cast<Npp32f>(sc.val[0]), 
                dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz) );
        }
        else
        {
            Npp32fc nValue;
            nValue.re = static_cast<Npp32f>(sc.val[0]);
            nValue.im = static_cast<Npp32f>(sc.val[1]);
            nppSafeCall( nppiSubC_32fc_C1R(src.ptr<Npp32fc>(), static_cast<int>(src.step), nValue, 
                dst.ptr<Npp32fc>(), static_cast<int>(dst.step), sz) );
        }

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );

        return;
    }

    const func_t func = funcs[src.depth()][dst.depth()];
    CV_Assert(func != 0);

    func(src, sc.val[0], dst, mask, stream);
}

////////////////////////////////////////////////////////////////////////
// multiply

namespace cv { namespace gpu { namespace device 
{
    void multiply_gpu(const DevMem2D_<uchar4>& src1, const DevMem2Df& src2, const DevMem2D_<uchar4>& dst, cudaStream_t stream);
    void multiply_gpu(const DevMem2D_<short4>& src1, const DevMem2Df& src2, const DevMem2D_<short4>& dst, cudaStream_t stream);

    template <typename T, typename D> 
    void multiply_gpu(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);

    template <typename T, typename D> 
    void multiply_gpu(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);
}}}

void cv::gpu::multiply(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, double scale, int dtype, Stream& s)
{
    using namespace ::cv::gpu::device;

    typedef void (*func_t)(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);

    static const func_t funcs[7][7] = 
    {
        {multiply_gpu<unsigned char, unsigned char>, 0/*multiply_gpu<unsigned char, signed char>*/, multiply_gpu<unsigned char, unsigned short>, multiply_gpu<unsigned char, short>, multiply_gpu<unsigned char, int>, multiply_gpu<unsigned char, float>, multiply_gpu<unsigned char, double>},
        {0/*multiply_gpu<signed char, unsigned char>*/, 0/*multiply_gpu<signed char, signed char>*/, 0/*multiply_gpu<signed char, unsigned short>*/, 0/*multiply_gpu<signed char, short>*/, 0/*multiply_gpu<signed char, int>*/, 0/*multiply_gpu<signed char, float>*/, 0/*multiply_gpu<signed char, double>*/},
        {0/*multiply_gpu<unsigned short, unsigned char>*/, 0/*multiply_gpu<unsigned short, signed char>*/, multiply_gpu<unsigned short, unsigned short>, 0/*multiply_gpu<unsigned short, short>*/, multiply_gpu<unsigned short, int>, multiply_gpu<unsigned short, float>, multiply_gpu<unsigned short, double>},
        {0/*multiply_gpu<short, unsigned char>*/, 0/*multiply_gpu<short, signed char>*/, 0/*multiply_gpu<short, unsigned short>*/, multiply_gpu<short, short>, multiply_gpu<short, int>, multiply_gpu<short, float>, multiply_gpu<short, double>},
        {0/*multiply_gpu<int, unsigned char>*/, 0/*multiply_gpu<int, signed char>*/, 0/*multiply_gpu<int, unsigned short>*/, 0/*multiply_gpu<int, short>*/, multiply_gpu<int, int>, multiply_gpu<int, float>, multiply_gpu<int, double>},
        {0/*multiply_gpu<float, unsigned char>*/, 0/*multiply_gpu<float, signed char>*/, 0/*multiply_gpu<float, unsigned short>*/, 0/*multiply_gpu<float, short>*/, 0/*multiply_gpu<float, int>*/, multiply_gpu<float, float>, multiply_gpu<float, double>},
        {0/*multiply_gpu<double, unsigned char>*/, 0/*multiply_gpu<double, signed char>*/, 0/*multiply_gpu<double, unsigned short>*/, 0/*multiply_gpu<double, short>*/, 0/*multiply_gpu<double, int>*/, 0/*multiply_gpu<double, float>*/, multiply_gpu<double, double>}
    };

    cudaStream_t stream = StreamAccessor::getStream(s);

    if (src1.type() == CV_8UC4 && src2.type() == CV_32FC1)
    {
        CV_Assert(src1.size() == src2.size());

        dst.create(src1.size(), src1.type());

        multiply_gpu(static_cast<DevMem2D_<uchar4> >(src1), static_cast<DevMem2Df>(src2), static_cast<DevMem2D_<uchar4> >(dst), stream);
    }
    else if (src1.type() == CV_16SC4 && src2.type() == CV_32FC1)
    {
        CV_Assert(src1.size() == src2.size());

        dst.create(src1.size(), src1.type());

        multiply_gpu(static_cast<DevMem2D_<short4> >(src1), static_cast<DevMem2Df>(src2), static_cast<DevMem2D_<short4> >(dst), stream);
    }
    else
    {
        CV_Assert(src1.type() == src2.type() && src1.size() == src2.size());

        if (dtype < 0)
            dtype = src1.depth();

        dst.create(src1.size(), CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src1.channels()));


        if (scale == 1 && dst.type() == src1.type() && (src1.depth() == CV_8U || src1.depth() == CV_32S || src1.depth() == CV_32F))
        {
            nppArithmCaller(src2, src1, dst, nppiMul_8u_C1RSfs, nppiMul_8u_C4RSfs, nppiMul_32s_C1R, nppiMul_32f_C1R, stream);
            return;
        }

        const func_t func = funcs[src1.depth()][dst.depth()];
        CV_Assert(func != 0);

        func(src1.reshape(1), src2.reshape(1), dst.reshape(1), scale, stream);
    }
}

void cv::gpu::multiply(const GpuMat& src, const Scalar& sc, GpuMat& dst, double scale, int dtype, Stream& s)
{
    using namespace ::cv::gpu::device;

    typedef void (*func_t)(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);

    static const func_t funcs[7][7] = 
    {
        {multiply_gpu<unsigned char, unsigned char>, 0/*multiply_gpu<unsigned char, signed char>*/, multiply_gpu<unsigned char, unsigned short>, multiply_gpu<unsigned char, short>, multiply_gpu<unsigned char, int>, multiply_gpu<unsigned char, float>, multiply_gpu<unsigned char, double>},
        {0/*multiply_gpu<signed char, unsigned char>*/, 0/*multiply_gpu<signed char, signed char>*/, 0/*multiply_gpu<signed char, unsigned short>*/, 0/*multiply_gpu<signed char, short>*/, 0/*multiply_gpu<signed char, int>*/, 0/*multiply_gpu<signed char, float>*/, 0/*multiply_gpu<signed char, double>*/},
        {0/*multiply_gpu<unsigned short, unsigned char>*/, 0/*multiply_gpu<unsigned short, signed char>*/, multiply_gpu<unsigned short, unsigned short>, 0/*multiply_gpu<unsigned short, short>*/, multiply_gpu<unsigned short, int>, multiply_gpu<unsigned short, float>, multiply_gpu<unsigned short, double>},
        {0/*multiply_gpu<short, unsigned char>*/, 0/*multiply_gpu<short, signed char>*/, 0/*multiply_gpu<short, unsigned short>*/, multiply_gpu<short, short>, multiply_gpu<short, int>, multiply_gpu<short, float>, multiply_gpu<short, double>},
        {0/*multiply_gpu<int, unsigned char>*/, 0/*multiply_gpu<int, signed char>*/, 0/*multiply_gpu<int, unsigned short>*/, 0/*multiply_gpu<int, short>*/, multiply_gpu<int, int>, multiply_gpu<int, float>, multiply_gpu<int, double>},
        {0/*multiply_gpu<float, unsigned char>*/, 0/*multiply_gpu<float, signed char>*/, 0/*multiply_gpu<float, unsigned short>*/, 0/*multiply_gpu<float, short>*/, 0/*multiply_gpu<float, int>*/, multiply_gpu<float, float>, multiply_gpu<float, double>},
        {0/*multiply_gpu<double, unsigned char>*/, 0/*multiply_gpu<double, signed char>*/, 0/*multiply_gpu<double, unsigned short>*/, 0/*multiply_gpu<double, short>*/, 0/*multiply_gpu<double, int>*/, 0/*multiply_gpu<double, float>*/, multiply_gpu<double, double>}
    };

    //CV_Assert(src.channels() == 1);

    if (dtype < 0)
        dtype = src.depth();

    dst.create(src.size(), CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src.channels()));

    cudaStream_t stream = StreamAccessor::getStream(s);

    if (dst.type() == src.type() && src.type() == CV_32FC1 && scale == 1)
    {
        NppiSize sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        NppStreamHandler h(stream);

        nppSafeCall( nppiMulC_32f_C1R(src.ptr<Npp32f>(), static_cast<int>(src.step), static_cast<Npp32f>(sc.val[0]), 
            dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );

        return;
    }

    const func_t func = funcs[src.depth()][dst.depth()];
    CV_Assert(func != 0);

    func(src.reshape(1), sc.val[0], dst.reshape(1), scale, stream);
}

////////////////////////////////////////////////////////////////////////
// divide

namespace cv { namespace gpu { namespace device 
{
    void divide_gpu(const DevMem2D_<uchar4>& src1, const DevMem2Df& src2, const DevMem2D_<uchar4>& dst, cudaStream_t stream);
    void divide_gpu(const DevMem2D_<short4>& src1, const DevMem2Df& src2, const DevMem2D_<short4>& dst, cudaStream_t stream);

    template <typename T, typename D> 
    void divide_gpu(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);

    template <typename T, typename D> 
    void divide_gpu(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);

    template <typename T, typename D> 
    void divide_gpu(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
}}}

void cv::gpu::divide(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, double scale, int dtype, Stream& s)
{
    using namespace ::cv::gpu::device;

    typedef void (*func_t)(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, double scale, cudaStream_t stream);

    static const func_t funcs[7][7] = 
    {
        {divide_gpu<unsigned char, unsigned char>, 0/*divide_gpu<unsigned char, signed char>*/, divide_gpu<unsigned char, unsigned short>, divide_gpu<unsigned char, short>, divide_gpu<unsigned char, int>, divide_gpu<unsigned char, float>, divide_gpu<unsigned char, double>},
        {0/*divide_gpu<signed char, unsigned char>*/, 0/*divide_gpu<signed char, signed char>*/, 0/*divide_gpu<signed char, unsigned short>*/, 0/*divide_gpu<signed char, short>*/, 0/*divide_gpu<signed char, int>*/, 0/*divide_gpu<signed char, float>*/, 0/*divide_gpu<signed char, double>*/},
        {0/*divide_gpu<unsigned short, unsigned char>*/, 0/*divide_gpu<unsigned short, signed char>*/, divide_gpu<unsigned short, unsigned short>, 0/*divide_gpu<unsigned short, short>*/, divide_gpu<unsigned short, int>, divide_gpu<unsigned short, float>, divide_gpu<unsigned short, double>},
        {0/*divide_gpu<short, unsigned char>*/, 0/*divide_gpu<short, signed char>*/, 0/*divide_gpu<short, unsigned short>*/, divide_gpu<short, short>, divide_gpu<short, int>, divide_gpu<short, float>, divide_gpu<short, double>},
        {0/*divide_gpu<int, unsigned char>*/, 0/*divide_gpu<int, signed char>*/, 0/*divide_gpu<int, unsigned short>*/, 0/*divide_gpu<int, short>*/, divide_gpu<int, int>, divide_gpu<int, float>, divide_gpu<int, double>},
        {0/*divide_gpu<float, unsigned char>*/, 0/*divide_gpu<float, signed char>*/, 0/*divide_gpu<float, unsigned short>*/, 0/*divide_gpu<float, short>*/, 0/*divide_gpu<float, int>*/, divide_gpu<float, float>, divide_gpu<float, double>},
        {0/*divide_gpu<double, unsigned char>*/, 0/*divide_gpu<double, signed char>*/, 0/*divide_gpu<double, unsigned short>*/, 0/*divide_gpu<double, short>*/, 0/*divide_gpu<double, int>*/, 0/*divide_gpu<double, float>*/, divide_gpu<double, double>}
    };

    cudaStream_t stream = StreamAccessor::getStream(s);

    if (src1.type() == CV_8UC4 && src2.type() == CV_32FC1)
    {
        CV_Assert(src1.size() == src2.size());

        dst.create(src1.size(), src1.type());

        multiply_gpu(static_cast<DevMem2D_<uchar4> >(src1), static_cast<DevMem2Df>(src2), static_cast<DevMem2D_<uchar4> >(dst), stream);
    }
    else if (src1.type() == CV_16SC4 && src2.type() == CV_32FC1)
    {
        CV_Assert(src1.size() == src2.size());

        dst.create(src1.size(), src1.type());

        multiply_gpu(static_cast<DevMem2D_<short4> >(src1), static_cast<DevMem2Df>(src2), static_cast<DevMem2D_<short4> >(dst), stream);
    }
    else
    {
        CV_Assert(src1.type() == src2.type() && src1.size() == src2.size());

        if (dtype < 0)
            dtype = src1.depth();

        dst.create(src1.size(), CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src1.channels()));


        if (scale == 1 && dst.type() == src1.type() && (src1.depth() == CV_8U || src1.depth() == CV_32S || src1.depth() == CV_32F))
        {
            nppArithmCaller(src2, src1, dst, nppiDiv_8u_C1RSfs, nppiDiv_8u_C4RSfs, nppiDiv_32s_C1R, nppiDiv_32f_C1R, stream);
            return;
        }

        const func_t func = funcs[src1.depth()][dst.depth()];
        CV_Assert(func != 0);

        func(src1.reshape(1), src2.reshape(1), dst.reshape(1), scale, stream);
    }
}

void cv::gpu::divide(const GpuMat& src, const Scalar& sc, GpuMat& dst, double scale, int dtype, Stream& s)
{
    using namespace ::cv::gpu::device;

    typedef void (*func_t)(const DevMem2Db& src1, double val, const DevMem2Db& dst, double scale, cudaStream_t stream);

    static const func_t funcs[7][7] = 
    {
        {divide_gpu<unsigned char, unsigned char>, 0/*divide_gpu<unsigned char, signed char>*/, divide_gpu<unsigned char, unsigned short>, divide_gpu<unsigned char, short>, divide_gpu<unsigned char, int>, divide_gpu<unsigned char, float>, divide_gpu<unsigned char, double>},
        {0/*divide_gpu<signed char, unsigned char>*/, 0/*divide_gpu<signed char, signed char>*/, 0/*divide_gpu<signed char, unsigned short>*/, 0/*divide_gpu<signed char, short>*/, 0/*divide_gpu<signed char, int>*/, 0/*divide_gpu<signed char, float>*/, 0/*divide_gpu<signed char, double>*/},
        {0/*divide_gpu<unsigned short, unsigned char>*/, 0/*divide_gpu<unsigned short, signed char>*/, divide_gpu<unsigned short, unsigned short>, 0/*divide_gpu<unsigned short, short>*/, divide_gpu<unsigned short, int>, divide_gpu<unsigned short, float>, divide_gpu<unsigned short, double>},
        {0/*divide_gpu<short, unsigned char>*/, 0/*divide_gpu<short, signed char>*/, 0/*divide_gpu<short, unsigned short>*/, divide_gpu<short, short>, divide_gpu<short, int>, divide_gpu<short, float>, divide_gpu<short, double>},
        {0/*divide_gpu<int, unsigned char>*/, 0/*divide_gpu<int, signed char>*/, 0/*divide_gpu<int, unsigned short>*/, 0/*divide_gpu<int, short>*/, divide_gpu<int, int>, divide_gpu<int, float>, divide_gpu<int, double>},
        {0/*divide_gpu<float, unsigned char>*/, 0/*divide_gpu<float, signed char>*/, 0/*divide_gpu<float, unsigned short>*/, 0/*divide_gpu<float, short>*/, 0/*divide_gpu<float, int>*/, divide_gpu<float, float>, divide_gpu<float, double>},
        {0/*divide_gpu<double, unsigned char>*/, 0/*divide_gpu<double, signed char>*/, 0/*divide_gpu<double, unsigned short>*/, 0/*divide_gpu<double, short>*/, 0/*divide_gpu<double, int>*/, 0/*divide_gpu<double, float>*/, divide_gpu<double, double>}
    };

    CV_Assert(src.channels() == 1);

    if (dtype < 0)
        dtype = src.depth();

    dst.create(src.size(), CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src.channels()));

    cudaStream_t stream = StreamAccessor::getStream(s);

    if (dst.type() == src.type() && src.type() == CV_32FC1 && scale == 1)
    {
        NppiSize sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        NppStreamHandler h(stream);

        nppSafeCall( nppiDivC_32f_C1R(src.ptr<Npp32f>(), static_cast<int>(src.step), static_cast<Npp32f>(sc.val[0]), 
            dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );

        return;
    }

    const func_t func = funcs[src.depth()][dst.depth()];
    CV_Assert(func != 0);

    func(src, sc.val[0], dst, scale, stream);
}

void cv::gpu::divide(double scale, const GpuMat& src, GpuMat& dst, int dtype, Stream& s)
{
    using namespace ::cv::gpu::device;

    typedef void (*func_t)(double scalar, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);

    static const func_t funcs[7][7] = 
    {
        {divide_gpu<unsigned char, unsigned char>, 0/*divide_gpu<unsigned char, signed char>*/, divide_gpu<unsigned char, unsigned short>, divide_gpu<unsigned char, short>, divide_gpu<unsigned char, int>, divide_gpu<unsigned char, float>, divide_gpu<unsigned char, double>},
        {0/*divide_gpu<signed char, unsigned char>*/, 0/*divide_gpu<signed char, signed char>*/, 0/*divide_gpu<signed char, unsigned short>*/, 0/*divide_gpu<signed char, short>*/, 0/*divide_gpu<signed char, int>*/, 0/*divide_gpu<signed char, float>*/, 0/*divide_gpu<signed char, double>*/},
        {0/*divide_gpu<unsigned short, unsigned char>*/, 0/*divide_gpu<unsigned short, signed char>*/, divide_gpu<unsigned short, unsigned short>, 0/*divide_gpu<unsigned short, short>*/, divide_gpu<unsigned short, int>, divide_gpu<unsigned short, float>, divide_gpu<unsigned short, double>},
        {0/*divide_gpu<short, unsigned char>*/, 0/*divide_gpu<short, signed char>*/, 0/*divide_gpu<short, unsigned short>*/, divide_gpu<short, short>, divide_gpu<short, int>, divide_gpu<short, float>, divide_gpu<short, double>},
        {0/*divide_gpu<int, unsigned char>*/, 0/*divide_gpu<int, signed char>*/, 0/*divide_gpu<int, unsigned short>*/, 0/*divide_gpu<int, short>*/, divide_gpu<int, int>, divide_gpu<int, float>, divide_gpu<int, double>},
        {0/*divide_gpu<float, unsigned char>*/, 0/*divide_gpu<float, signed char>*/, 0/*divide_gpu<float, unsigned short>*/, 0/*divide_gpu<float, short>*/, 0/*divide_gpu<float, int>*/, divide_gpu<float, float>, divide_gpu<float, double>},
        {0/*divide_gpu<double, unsigned char>*/, 0/*divide_gpu<double, signed char>*/, 0/*divide_gpu<double, unsigned short>*/, 0/*divide_gpu<double, short>*/, 0/*divide_gpu<double, int>*/, 0/*divide_gpu<double, float>*/, divide_gpu<double, double>}
    };

    CV_Assert(src.channels() == 1);

    if (dtype < 0)
        dtype = src.depth();

    dst.create(src.size(), CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src.channels()));

    cudaStream_t stream = StreamAccessor::getStream(s);

    const func_t func = funcs[src.depth()][dst.depth()];
    CV_Assert(func != 0);

    func(scale, src, dst, stream);
}

//////////////////////////////////////////////////////////////////////////////
// absdiff

namespace cv { namespace gpu { namespace device 
{
    template <typename T>
    void absdiff_gpu(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);

    template <typename T> 
    void absdiff_gpu(const DevMem2Db& src1, double val, const DevMem2Db& dst, cudaStream_t stream);
}}}

void cv::gpu::absdiff(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& s)
{
    using namespace ::cv::gpu::device;

    typedef void (*func_t)(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);

    static const func_t funcs[] = 
    {
        0/*absdiff_gpu<unsigned char>*/, absdiff_gpu<signed char>, absdiff_gpu<unsigned short>, absdiff_gpu<short>, 0/*absdiff_gpu<int>*/, 0/*absdiff_gpu<float>*/, absdiff_gpu<double>
    };

    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());

    dst.create( src1.size(), src1.type() );

    cudaStream_t stream = StreamAccessor::getStream(s);

    NppiSize sz;
    sz.width  = src1.cols * src1.channels();
    sz.height = src1.rows;

    if (src1.depth() == CV_8U && (src1.cols * src1.channels()) % 4 == 0)
    {
        NppStreamHandler h(stream);

        sz.width /= 4;

        nppSafeCall( nppiAbsDiff_8u_C4R(src1.ptr<Npp8u>(), static_cast<int>(src1.step), src2.ptr<Npp8u>(), static_cast<int>(src2.step), 
            dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
    else if (src1.depth() == CV_8U)
    {
        NppStreamHandler h(stream);

        nppSafeCall( nppiAbsDiff_8u_C1R(src1.ptr<Npp8u>(), static_cast<int>(src1.step), src2.ptr<Npp8u>(), static_cast<int>(src2.step), 
            dst.ptr<Npp8u>(), static_cast<int>(dst.step), sz) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
    else if (src1.depth() == CV_32S)
    {
        NppStreamHandler h(stream);

        nppSafeCall( nppiAbsDiff_32s_C1R(src1.ptr<Npp32s>(), static_cast<int>(src1.step), src2.ptr<Npp32s>(), static_cast<int>(src2.step), 
            dst.ptr<Npp32s>(), static_cast<int>(dst.step), sz) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
    else if (src1.depth() == CV_32F)
    {
        NppStreamHandler h(stream);

        nppSafeCall( nppiAbsDiff_32f_C1R(src1.ptr<Npp32f>(), static_cast<int>(src1.step), src2.ptr<Npp32f>(), static_cast<int>(src2.step), 
            dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
    else
    {
        const func_t func = funcs[src1.depth()];
        CV_Assert(func != 0);

        func(src1.reshape(1), src2.reshape(1), dst.reshape(1), stream);
    }
}

void cv::gpu::absdiff(const GpuMat& src1, const Scalar& src2, GpuMat& dst, Stream& s)
{
    using namespace ::cv::gpu::device;

    typedef void (*func_t)(const DevMem2Db& src1, double val, const DevMem2Db& dst, cudaStream_t stream);

    static const func_t funcs[] = 
    {
        absdiff_gpu<unsigned char>, absdiff_gpu<signed char>, absdiff_gpu<unsigned short>, absdiff_gpu<short>,absdiff_gpu<int>, 0/*absdiff_gpu<float>*/, absdiff_gpu<double>
    };

    CV_Assert(src1.channels() == 1);

    dst.create(src1.size(), src1.type());

    cudaStream_t stream = StreamAccessor::getStream(s);

    if (src1.type() == CV_32FC1)
    {
        NppiSize sz;
        sz.width  = src1.cols;
        sz.height = src1.rows;

        cudaStream_t stream = StreamAccessor::getStream(s);

        NppStreamHandler h(stream);

        nppSafeCall( nppiAbsDiffC_32f_C1R(src1.ptr<Npp32f>(), static_cast<int>(src1.step), 
            dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz, static_cast<Npp32f>(src2.val[0])) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );

        return;
    }

    const func_t func = funcs[src1.depth()];
    CV_Assert(func != 0);

    func(src1, src2.val[0], dst, stream);
}


//////////////////////////////////////////////////////////////////////////////
// Comparison of two matrixes

namespace cv { namespace gpu { namespace device 
{
    template <typename T> void compare_eq(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template <typename T> void compare_ne(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template <typename T> void compare_lt(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
    template <typename T> void compare_le(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);
}}}

void cv::gpu::compare(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, int cmpop, Stream& stream)
{
    using namespace ::cv::gpu::device;

    typedef void (*func_t)(const DevMem2Db& src1, const DevMem2Db& src2, const DevMem2Db& dst, cudaStream_t stream);

    static const func_t funcs[7][4] = 
    {
        {compare_eq<unsigned char>, compare_ne<unsigned char>, compare_lt<unsigned char>, compare_le<unsigned char>},
        {compare_eq<signed char>, compare_ne<signed char>, compare_lt<signed char>, compare_le<signed char>},
        {compare_eq<unsigned short>, compare_ne<unsigned short>, compare_lt<unsigned short>, compare_le<unsigned short>},
        {compare_eq<short>, compare_ne<short>, compare_lt<short>, compare_le<short>},
        {compare_eq<int>, compare_ne<int>, compare_lt<int>, compare_le<int>},
        {compare_eq<float>, compare_ne<float>, compare_lt<float>, compare_le<float>},
        {compare_eq<double>, compare_ne<double>, compare_lt<double>, compare_le<double>}
    };

    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());

    int code;
    const GpuMat* psrc1;
    const GpuMat* psrc2;

    switch (cmpop)
    {
    case CMP_EQ:
        code = 0;
        psrc1 = &src1;
        psrc2 = &src2;
        break;
    case CMP_GE:
        code = 3;
        psrc1 = &src2;
        psrc2 = &src1;
        break;
    case CMP_GT:
        code = 2;
        psrc1 = &src2;
        psrc2 = &src1;
        break;
    case CMP_LE:
        code = 3;
        psrc1 = &src1;
        psrc2 = &src2;
        break;
    case CMP_LT:
        code = 2;
        psrc1 = &src1;
        psrc2 = &src2;
        break;
    case CMP_NE:
        code = 1;
        psrc1 = &src1;
        psrc2 = &src2;
        break;
    default:
        CV_Error(CV_StsBadFlag, "Incorrect compare operation");
    };

    dst.create(src1.size(), CV_MAKE_TYPE(CV_8U, src1.channels()));

    funcs[src1.depth()][code](psrc1->reshape(1), psrc2->reshape(1), dst.reshape(1), StreamAccessor::getStream(stream));
}


//////////////////////////////////////////////////////////////////////////////
// Unary bitwise logical operations

namespace cv { namespace gpu { namespace device 
{
    void bitwiseNotCaller(int rows, int cols, size_t elem_size1, int cn, const PtrStepb src, PtrStepb dst, cudaStream_t stream);

    template <typename T>
    void bitwiseMaskNotCaller(int rows, int cols, int cn, const PtrStepb src, const PtrStepb mask, PtrStepb dst, cudaStream_t stream);
}}}

namespace
{
    void bitwiseNotCaller(const GpuMat& src, GpuMat& dst, cudaStream_t stream)
    {
        dst.create(src.size(), src.type());

        ::cv::gpu::device::bitwiseNotCaller(src.rows, src.cols, src.elemSize1(), dst.channels(), src, dst, stream);
    }


    void bitwiseNotCaller(const GpuMat& src, GpuMat& dst, const GpuMat& mask, cudaStream_t stream)
    {
        using namespace ::cv::gpu::device;

        typedef void (*Caller)(int, int, int, const PtrStepb, const PtrStepb, PtrStepb, cudaStream_t);

        static Caller callers[] = 
        {
            bitwiseMaskNotCaller<unsigned char>, bitwiseMaskNotCaller<unsigned char>, 
            bitwiseMaskNotCaller<unsigned short>, bitwiseMaskNotCaller<unsigned short>,
            bitwiseMaskNotCaller<unsigned int>, bitwiseMaskNotCaller<unsigned int>,
            bitwiseMaskNotCaller<unsigned int>
        };

        CV_Assert(mask.type() == CV_8U && mask.size() == src.size());
        dst.create(src.size(), src.type());

        Caller caller = callers[src.depth()];
        CV_Assert(caller);

        int cn = src.depth() != CV_64F ? src.channels() : src.channels() * (sizeof(double) / sizeof(unsigned int));
        caller(src.rows, src.cols, cn, src, mask, dst, stream);
    }

}


void cv::gpu::bitwise_not(const GpuMat& src, GpuMat& dst, const GpuMat& mask, Stream& stream)
{
    if (mask.empty())
        bitwiseNotCaller(src, dst, StreamAccessor::getStream(stream));
    else
        bitwiseNotCaller(src, dst, mask, StreamAccessor::getStream(stream));
}


//////////////////////////////////////////////////////////////////////////////
// Binary bitwise logical operations

namespace cv { namespace gpu { namespace device 
{
    void bitwiseOrCaller(int rows, int cols, size_t elem_size1, int cn, const PtrStepb src1, const PtrStepb src2, PtrStepb dst, cudaStream_t stream);

    template <typename T>
    void bitwiseMaskOrCaller(int rows, int cols, int cn, const PtrStepb src1, const PtrStepb src2, const PtrStepb mask, PtrStepb dst, cudaStream_t stream);

    void bitwiseAndCaller(int rows, int cols, size_t elem_size1, int cn, const PtrStepb src1, const PtrStepb src2, PtrStepb dst, cudaStream_t stream);

    template <typename T>
    void bitwiseMaskAndCaller(int rows, int cols, int cn, const PtrStepb src1, const PtrStepb src2, const PtrStepb mask, PtrStepb dst, cudaStream_t stream);

    void bitwiseXorCaller(int rows, int cols, size_t elem_size1, int cn, const PtrStepb src1, const PtrStepb src2, PtrStepb dst, cudaStream_t stream);

    template <typename T>
    void bitwiseMaskXorCaller(int rows, int cols, int cn, const PtrStepb src1, const PtrStepb src2, const PtrStepb mask, PtrStepb dst, cudaStream_t stream);
}}}

namespace
{
    void bitwiseOrCaller(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, cudaStream_t stream)
    {
        CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
        dst.create(src1.size(), src1.type());

        ::cv::gpu::device::bitwiseOrCaller(dst.rows, dst.cols, dst.elemSize1(), dst.channels(), src1, src2, dst, stream);
    }

    void bitwiseOrCaller(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, cudaStream_t stream)
    {
        using namespace ::cv::gpu::device;

        typedef void (*Caller)(int, int, int, const PtrStepb, const PtrStepb, const PtrStepb, PtrStepb, cudaStream_t);

        static Caller callers[] = 
        {
            bitwiseMaskOrCaller<unsigned char>, bitwiseMaskOrCaller<unsigned char>, 
            bitwiseMaskOrCaller<unsigned short>, bitwiseMaskOrCaller<unsigned short>,
            bitwiseMaskOrCaller<unsigned int>, bitwiseMaskOrCaller<unsigned int>,
            bitwiseMaskOrCaller<unsigned int>
        };

        CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
        dst.create(src1.size(), src1.type());

        Caller caller = callers[src1.depth()];
        CV_Assert(caller);

        int cn = dst.depth() != CV_64F ? dst.channels() : dst.channels() * (sizeof(double) / sizeof(unsigned int));
        caller(dst.rows, dst.cols, cn, src1, src2, mask, dst, stream);
    }


    void bitwiseAndCaller(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, cudaStream_t stream)
    {
        CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
        dst.create(src1.size(), src1.type());

        ::cv::gpu::device::bitwiseAndCaller(dst.rows, dst.cols, dst.elemSize1(), dst.channels(), src1, src2, dst, stream);
    }


    void bitwiseAndCaller(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, cudaStream_t stream)
    {
        using namespace ::cv::gpu::device;

        typedef void (*Caller)(int, int, int, const PtrStepb, const PtrStepb, const PtrStepb, PtrStepb, cudaStream_t);

        static Caller callers[] = 
        {
            bitwiseMaskAndCaller<unsigned char>, bitwiseMaskAndCaller<unsigned char>, 
            bitwiseMaskAndCaller<unsigned short>, bitwiseMaskAndCaller<unsigned short>,
            bitwiseMaskAndCaller<unsigned int>, bitwiseMaskAndCaller<unsigned int>,
            bitwiseMaskAndCaller<unsigned int>
        };

        CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
        dst.create(src1.size(), src1.type());

        Caller caller = callers[src1.depth()];
        CV_Assert(caller);

        int cn = dst.depth() != CV_64F ? dst.channels() : dst.channels() * (sizeof(double) / sizeof(unsigned int));
        caller(dst.rows, dst.cols, cn, src1, src2, mask, dst, stream);
    }


    void bitwiseXorCaller(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, cudaStream_t stream)
    {
        CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
        dst.create(src1.size(), src1.type());

        ::cv::gpu::device::bitwiseXorCaller(dst.rows, dst.cols, dst.elemSize1(), dst.channels(), src1, src2, dst, stream);
    }


    void bitwiseXorCaller(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, cudaStream_t stream)
    {
        using namespace ::cv::gpu::device;

        typedef void (*Caller)(int, int, int, const PtrStepb, const PtrStepb, const PtrStepb, PtrStepb, cudaStream_t);

        static Caller callers[] = 
        {
            bitwiseMaskXorCaller<unsigned char>, bitwiseMaskXorCaller<unsigned char>, 
            bitwiseMaskXorCaller<unsigned short>, bitwiseMaskXorCaller<unsigned short>,
            bitwiseMaskXorCaller<unsigned int>, bitwiseMaskXorCaller<unsigned int>,
            bitwiseMaskXorCaller<unsigned int>
        };

        CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
        dst.create(src1.size(), src1.type());

        Caller caller = callers[src1.depth()];
        CV_Assert(caller);

        int cn = dst.depth() != CV_64F ? dst.channels() : dst.channels() * (sizeof(double) / sizeof(unsigned int));
        caller(dst.rows, dst.cols, cn, src1, src2, mask, dst, stream);
    }
}


void cv::gpu::bitwise_or(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, Stream& stream)
{
    if (mask.empty())
        bitwiseOrCaller(src1, src2, dst, StreamAccessor::getStream(stream));
    else
        bitwiseOrCaller(src1, src2, dst, mask, StreamAccessor::getStream(stream));
}


void cv::gpu::bitwise_and(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, Stream& stream)
{
    if (mask.empty())
        bitwiseAndCaller(src1, src2, dst, StreamAccessor::getStream(stream));
    else
        bitwiseAndCaller(src1, src2, dst, mask, StreamAccessor::getStream(stream));
}


void cv::gpu::bitwise_xor(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask, Stream& stream)
{
    if (mask.empty())
        bitwiseXorCaller(src1, src2, dst, StreamAccessor::getStream(stream));
    else
        bitwiseXorCaller(src1, src2, dst, mask, StreamAccessor::getStream(stream));
}


//////////////////////////////////////////////////////////////////////////////
// Minimum and maximum operations

namespace cv { namespace gpu { namespace device 
{
    template <typename T>
    void min_gpu(const DevMem2D_<T>& src1, const DevMem2D_<T>& src2, const DevMem2D_<T>& dst, cudaStream_t stream);

    template <typename T>
    void max_gpu(const DevMem2D_<T>& src1, const DevMem2D_<T>& src2, const DevMem2D_<T>& dst, cudaStream_t stream);

    template <typename T>
    void min_gpu(const DevMem2D_<T>& src1, T src2, const DevMem2D_<T>& dst, cudaStream_t stream);

    template <typename T>
    void max_gpu(const DevMem2D_<T>& src1, T src2, const DevMem2D_<T>& dst, cudaStream_t stream);
}}}

namespace
{
    template <typename T>
    void min_caller(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, cudaStream_t stream)
    {
        CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
        dst.create(src1.size(), src1.type());
        ::cv::gpu::device::min_gpu<T>(src1.reshape(1), src2.reshape(1), dst.reshape(1), stream);
    }

    template <typename T>
    void min_caller(const GpuMat& src1, double src2, GpuMat& dst, cudaStream_t stream)
    {
        dst.create(src1.size(), src1.type());
        ::cv::gpu::device::min_gpu<T>(src1.reshape(1), saturate_cast<T>(src2), dst.reshape(1), stream);
    }
    
    template <typename T>
    void max_caller(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, cudaStream_t stream)
    {
        CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
        dst.create(src1.size(), src1.type());
        ::cv::gpu::device::max_gpu<T>(src1.reshape(1), src2.reshape(1), dst.reshape(1), stream);
    }

    template <typename T>
    void max_caller(const GpuMat& src1, double src2, GpuMat& dst, cudaStream_t stream)
    {
        dst.create(src1.size(), src1.type());
        ::cv::gpu::device::max_gpu<T>(src1.reshape(1), saturate_cast<T>(src2), dst.reshape(1), stream);
    }
}

void cv::gpu::min(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream) 
{ 
    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
    CV_Assert((src1.depth() != CV_64F) || 
        (TargetArchs::builtWith(NATIVE_DOUBLE) && DeviceInfo().supports(NATIVE_DOUBLE)));

    typedef void (*func_t)(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, cudaStream_t stream);
    static const func_t funcs[] = 
    {
        min_caller<unsigned char>, min_caller<signed char>, min_caller<unsigned short>, min_caller<short>, min_caller<int>, 
        min_caller<float>, min_caller<double>
    };
    funcs[src1.depth()](src1, src2, dst, StreamAccessor::getStream(stream));
}
void cv::gpu::min(const GpuMat& src1, double src2, GpuMat& dst, Stream& stream) 
{
    CV_Assert((src1.depth() != CV_64F) || 
        (TargetArchs::builtWith(NATIVE_DOUBLE) && DeviceInfo().supports(NATIVE_DOUBLE)));

    typedef void (*func_t)(const GpuMat& src1, double src2, GpuMat& dst, cudaStream_t stream);
    static const func_t funcs[] = 
    {
        min_caller<unsigned char>, min_caller<signed char>, min_caller<unsigned short>, min_caller<short>, min_caller<int>, 
        min_caller<float>, min_caller<double>
    };
    funcs[src1.depth()](src1, src2, dst, StreamAccessor::getStream(stream));
}

void cv::gpu::max(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream) 
{ 
    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
    CV_Assert((src1.depth() != CV_64F) || 
        (TargetArchs::builtWith(NATIVE_DOUBLE) && DeviceInfo().supports(NATIVE_DOUBLE)));

    typedef void (*func_t)(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, cudaStream_t stream);
    static const func_t funcs[] = 
    {
        max_caller<unsigned char>, max_caller<signed char>, max_caller<unsigned short>, max_caller<short>, max_caller<int>, 
        max_caller<float>, max_caller<double>
    };
    funcs[src1.depth()](src1, src2, dst, StreamAccessor::getStream(stream));
}

void cv::gpu::max(const GpuMat& src1, double src2, GpuMat& dst, Stream& stream) 
{
    CV_Assert((src1.depth() != CV_64F) || 
        (TargetArchs::builtWith(NATIVE_DOUBLE) && DeviceInfo().supports(NATIVE_DOUBLE)));

    typedef void (*func_t)(const GpuMat& src1, double src2, GpuMat& dst, cudaStream_t stream);
    static const func_t funcs[] = 
    {
        max_caller<unsigned char>, max_caller<signed char>, max_caller<unsigned short>, max_caller<short>, max_caller<int>, 
        max_caller<float>, max_caller<double>
    };
    funcs[src1.depth()](src1, src2, dst, StreamAccessor::getStream(stream));
}

////////////////////////////////////////////////////////////////////////
// threshold

namespace cv { namespace gpu { namespace device 
{
    template <typename T>
    void threshold_gpu(const DevMem2Db& src, const DevMem2Db& dst, T thresh, T maxVal, int type, cudaStream_t stream);
}}}

namespace
{
    template <typename T> void threshold_caller(const GpuMat& src, GpuMat& dst, double thresh, double maxVal, int type, cudaStream_t stream)
    {
        ::cv::gpu::device::threshold_gpu<T>(src, dst, saturate_cast<T>(thresh), saturate_cast<T>(maxVal), type, stream);
    }
}

double cv::gpu::threshold(const GpuMat& src, GpuMat& dst, double thresh, double maxVal, int type, Stream& s)
{
    CV_Assert(src.channels() == 1 && src.depth() <= CV_64F);
    CV_Assert(type <= THRESH_TOZERO_INV);

    dst.create(src.size(), src.type());

    cudaStream_t stream = StreamAccessor::getStream(s);

    if (src.type() == CV_32FC1 && type == THRESH_TRUNC)
    {
        NppStreamHandler h(stream);

        NppiSize sz;
        sz.width  = src.cols;
        sz.height = src.rows;

        nppSafeCall( nppiThreshold_32f_C1R(src.ptr<Npp32f>(), static_cast<int>(src.step),
            dst.ptr<Npp32f>(), static_cast<int>(dst.step), sz, static_cast<Npp32f>(thresh), NPP_CMP_GREATER) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }
    else
    {
        CV_Assert((src.depth() != CV_64F) || 
            (TargetArchs::builtWith(NATIVE_DOUBLE) && DeviceInfo().supports(NATIVE_DOUBLE)));

        typedef void (*caller_t)(const GpuMat& src, GpuMat& dst, double thresh, double maxVal, int type, 
            cudaStream_t stream);

        static const caller_t callers[] = 
        {
            threshold_caller<unsigned char>, threshold_caller<signed char>, 
            threshold_caller<unsigned short>, threshold_caller<short>, 
            threshold_caller<int>, threshold_caller<float>, threshold_caller<double>
        };

        if (src.depth() != CV_32F && src.depth() != CV_64F)
        {
            thresh = cvFloor(thresh);
            maxVal = cvRound(maxVal);
        }

        callers[src.depth()](src, dst, thresh, maxVal, type, stream);
    }

    return thresh;
}

////////////////////////////////////////////////////////////////////////
// pow

namespace cv { namespace gpu { namespace device 
{
    template<typename T>
    void pow_caller(const DevMem2Db& src, float power, DevMem2Db dst, cudaStream_t stream);
}}}

void cv::gpu::pow(const GpuMat& src, double power, GpuMat& dst, Stream& stream)
{
    using namespace ::cv::gpu::device;

    CV_Assert(src.depth() != CV_64F);
    dst.create(src.size(), src.type());

    typedef void (*caller_t)(const DevMem2Db& src, float power, DevMem2Db dst, cudaStream_t stream);

    static const caller_t callers[] = 
    {
        pow_caller<unsigned char>,  pow_caller<signed char>, 
        pow_caller<unsigned short>, pow_caller<short>, 
        pow_caller<int>, pow_caller<float>
    };

    callers[src.depth()](src.reshape(1), (float)power, dst.reshape(1), StreamAccessor::getStream(stream));    
}

////////////////////////////////////////////////////////////////////////
// addWeighted

namespace cv { namespace gpu { namespace device 
{
    template <typename T1, typename T2, typename D>
    void addWeighted_gpu(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);
}}}

void cv::gpu::addWeighted(const GpuMat& src1, double alpha, const GpuMat& src2, double beta, double gamma, GpuMat& dst, int dtype, Stream& stream)
{
    using namespace ::cv::gpu::device;

    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.type() == src2.type() || (dtype >= 0 && src1.channels() == src2.channels()));

    dtype = dtype >= 0 ? CV_MAKETYPE(dtype, src1.channels()) : src1.type();

    dst.create(src1.size(), dtype);

    const GpuMat* psrc1 = &src1;
    const GpuMat* psrc2 = &src2;

    if (src1.depth() > src2.depth())
    {
        std::swap(psrc1, psrc2);
        std::swap(alpha, beta);
    }

    typedef void (*caller_t)(const DevMem2Db& src1, double alpha, const DevMem2Db& src2, double beta, double gamma, const DevMem2Db& dst, cudaStream_t stream);

    static const caller_t callers[7][7][7] =
    {
        {
            {
                addWeighted_gpu<unsigned char, unsigned char, unsigned char >,
                addWeighted_gpu<unsigned char, unsigned char, signed char >,
                addWeighted_gpu<unsigned char, unsigned char, unsigned short>,
                addWeighted_gpu<unsigned char, unsigned char, short >,
                addWeighted_gpu<unsigned char, unsigned char, int   >,
                addWeighted_gpu<unsigned char, unsigned char, float >,
                addWeighted_gpu<unsigned char, unsigned char, double>
            },
            {
                addWeighted_gpu<unsigned char, signed char, unsigned char >,
                addWeighted_gpu<unsigned char, signed char, signed char >,
                addWeighted_gpu<unsigned char, signed char, unsigned short>,
                addWeighted_gpu<unsigned char, signed char, short >,
                addWeighted_gpu<unsigned char, signed char, int   >,
                addWeighted_gpu<unsigned char, signed char, float >,
                addWeighted_gpu<unsigned char, signed char, double>
            },
            {
                addWeighted_gpu<unsigned char, unsigned short, unsigned char >,
                addWeighted_gpu<unsigned char, unsigned short, signed char >,
                addWeighted_gpu<unsigned char, unsigned short, unsigned short>,
                addWeighted_gpu<unsigned char, unsigned short, short >,
                addWeighted_gpu<unsigned char, unsigned short, int   >,
                addWeighted_gpu<unsigned char, unsigned short, float >,
                addWeighted_gpu<unsigned char, unsigned short, double>
            },
            {
                addWeighted_gpu<unsigned char, short, unsigned char >,
                addWeighted_gpu<unsigned char, short, signed char >,
                addWeighted_gpu<unsigned char, short, unsigned short>,
                addWeighted_gpu<unsigned char, short, short >,
                addWeighted_gpu<unsigned char, short, int   >,
                addWeighted_gpu<unsigned char, short, float >,
                addWeighted_gpu<unsigned char, short, double>
            },
            {
                addWeighted_gpu<unsigned char, int, unsigned char >,
                addWeighted_gpu<unsigned char, int, signed char >,
                addWeighted_gpu<unsigned char, int, unsigned short>,
                addWeighted_gpu<unsigned char, int, short >,
                addWeighted_gpu<unsigned char, int, int   >,
                addWeighted_gpu<unsigned char, int, float >,
                addWeighted_gpu<unsigned char, int, double>
            },
            {
                addWeighted_gpu<unsigned char, float, unsigned char >,
                addWeighted_gpu<unsigned char, float, signed char >,
                addWeighted_gpu<unsigned char, float, unsigned short>,
                addWeighted_gpu<unsigned char, float, short >,
                addWeighted_gpu<unsigned char, float, int   >,
                addWeighted_gpu<unsigned char, float, float >,
                addWeighted_gpu<unsigned char, float, double>
            },
            {
                addWeighted_gpu<unsigned char, double, unsigned char >,
                addWeighted_gpu<unsigned char, double, signed char >,
                addWeighted_gpu<unsigned char, double, unsigned short>,
                addWeighted_gpu<unsigned char, double, short >,
                addWeighted_gpu<unsigned char, double, int   >,
                addWeighted_gpu<unsigned char, double, float >,
                addWeighted_gpu<unsigned char, double, double>
            }
        },
        {
            {
                0/*addWeighted_gpu<signed char, unsigned char, unsigned char >*/,
                0/*addWeighted_gpu<signed char, unsigned char, signed char >*/,
                0/*addWeighted_gpu<signed char, unsigned char, unsigned short>*/,
                0/*addWeighted_gpu<signed char, unsigned char, short >*/,
                0/*addWeighted_gpu<signed char, unsigned char, int   >*/,
                0/*addWeighted_gpu<signed char, unsigned char, float >*/,
                0/*addWeighted_gpu<signed char, unsigned char, double>*/
            },
            {
                addWeighted_gpu<signed char, signed char, unsigned char >,
                addWeighted_gpu<signed char, signed char, signed char >,
                addWeighted_gpu<signed char, signed char, unsigned short>,
                addWeighted_gpu<signed char, signed char, short >,
                addWeighted_gpu<signed char, signed char, int   >,
                addWeighted_gpu<signed char, signed char, float >,
                addWeighted_gpu<signed char, signed char, double>
            },
            {
                addWeighted_gpu<signed char, unsigned short, unsigned char >,
                addWeighted_gpu<signed char, unsigned short, signed char >,
                addWeighted_gpu<signed char, unsigned short, unsigned short>,
                addWeighted_gpu<signed char, unsigned short, short >,
                addWeighted_gpu<signed char, unsigned short, int   >,
                addWeighted_gpu<signed char, unsigned short, float >,
                addWeighted_gpu<signed char, unsigned short, double>
            },
            {
                addWeighted_gpu<signed char, short, unsigned char >,
                addWeighted_gpu<signed char, short, signed char >,
                addWeighted_gpu<signed char, short, unsigned short>,
                addWeighted_gpu<signed char, short, short >,
                addWeighted_gpu<signed char, short, int   >,
                addWeighted_gpu<signed char, short, float >,
                addWeighted_gpu<signed char, short, double>
            },
            {
                addWeighted_gpu<signed char, int, unsigned char >,
                addWeighted_gpu<signed char, int, signed char >,
                addWeighted_gpu<signed char, int, unsigned short>,
                addWeighted_gpu<signed char, int, short >,
                addWeighted_gpu<signed char, int, int   >,
                addWeighted_gpu<signed char, int, float >,
                addWeighted_gpu<signed char, int, double>
            },
            {
                addWeighted_gpu<signed char, float, unsigned char >,
                addWeighted_gpu<signed char, float, signed char >,
                addWeighted_gpu<signed char, float, unsigned short>,
                addWeighted_gpu<signed char, float, short >,
                addWeighted_gpu<signed char, float, int   >,
                addWeighted_gpu<signed char, float, float >,
                addWeighted_gpu<signed char, float, double>
            },
            {
                addWeighted_gpu<signed char, double, unsigned char >,
                addWeighted_gpu<signed char, double, signed char >,
                addWeighted_gpu<signed char, double, unsigned short>,
                addWeighted_gpu<signed char, double, short >,
                addWeighted_gpu<signed char, double, int   >,
                addWeighted_gpu<signed char, double, float >,
                addWeighted_gpu<signed char, double, double>
            }
        },
        {
            {
                0/*addWeighted_gpu<unsigned short, unsigned char, unsigned char >*/,
                0/*addWeighted_gpu<unsigned short, unsigned char, signed char >*/,
                0/*addWeighted_gpu<unsigned short, unsigned char, unsigned short>*/,
                0/*addWeighted_gpu<unsigned short, unsigned char, short >*/,
                0/*addWeighted_gpu<unsigned short, unsigned char, int   >*/,
                0/*addWeighted_gpu<unsigned short, unsigned char, float >*/,
                0/*addWeighted_gpu<unsigned short, unsigned char, double>*/
            },
            {
                0/*addWeighted_gpu<unsigned short, signed char, unsigned char >*/,
                0/*addWeighted_gpu<unsigned short, signed char, signed char >*/,
                0/*addWeighted_gpu<unsigned short, signed char, unsigned short>*/,
                0/*addWeighted_gpu<unsigned short, signed char, short >*/,
                0/*addWeighted_gpu<unsigned short, signed char, int   >*/,
                0/*addWeighted_gpu<unsigned short, signed char, float >*/,
                0/*addWeighted_gpu<unsigned short, signed char, double>*/
            },
            {
                addWeighted_gpu<unsigned short, unsigned short, unsigned char >,
                addWeighted_gpu<unsigned short, unsigned short, signed char >,
                addWeighted_gpu<unsigned short, unsigned short, unsigned short>,
                addWeighted_gpu<unsigned short, unsigned short, short >,
                addWeighted_gpu<unsigned short, unsigned short, int   >,
                addWeighted_gpu<unsigned short, unsigned short, float >,
                addWeighted_gpu<unsigned short, unsigned short, double>
            },
            {
                addWeighted_gpu<unsigned short, short, unsigned char >,
                addWeighted_gpu<unsigned short, short, signed char >,
                addWeighted_gpu<unsigned short, short, unsigned short>,
                addWeighted_gpu<unsigned short, short, short >,
                addWeighted_gpu<unsigned short, short, int   >,
                addWeighted_gpu<unsigned short, short, float >,
                addWeighted_gpu<unsigned short, short, double>
            },
            {
                addWeighted_gpu<unsigned short, int, unsigned char >,
                addWeighted_gpu<unsigned short, int, signed char >,
                addWeighted_gpu<unsigned short, int, unsigned short>,
                addWeighted_gpu<unsigned short, int, short >,
                addWeighted_gpu<unsigned short, int, int   >,
                addWeighted_gpu<unsigned short, int, float >,
                addWeighted_gpu<unsigned short, int, double>
            },
            {
                addWeighted_gpu<unsigned short, float, unsigned char >,
                addWeighted_gpu<unsigned short, float, signed char >,
                addWeighted_gpu<unsigned short, float, unsigned short>,
                addWeighted_gpu<unsigned short, float, short >,
                addWeighted_gpu<unsigned short, float, int   >,
                addWeighted_gpu<unsigned short, float, float >,
                addWeighted_gpu<unsigned short, float, double>
            },
            {
                addWeighted_gpu<unsigned short, double, unsigned char >,
                addWeighted_gpu<unsigned short, double, signed char >,
                addWeighted_gpu<unsigned short, double, unsigned short>,
                addWeighted_gpu<unsigned short, double, short >,
                addWeighted_gpu<unsigned short, double, int   >,
                addWeighted_gpu<unsigned short, double, float >,
                addWeighted_gpu<unsigned short, double, double>
            }
        },
        {
            {
                0/*addWeighted_gpu<short, unsigned char, unsigned char >*/,
                0/*addWeighted_gpu<short, unsigned char, signed char >*/,
                0/*addWeighted_gpu<short, unsigned char, unsigned short>*/,
                0/*addWeighted_gpu<short, unsigned char, short >*/,
                0/*addWeighted_gpu<short, unsigned char, int   >*/,
                0/*addWeighted_gpu<short, unsigned char, float >*/,
                0/*addWeighted_gpu<short, unsigned char, double>*/
            },
            {
                0/*addWeighted_gpu<short, signed char, unsigned char >*/,
                0/*addWeighted_gpu<short, signed char, signed char >*/,
                0/*addWeighted_gpu<short, signed char, unsigned short>*/,
                0/*addWeighted_gpu<short, signed char, short >*/,
                0/*addWeighted_gpu<short, signed char, int   >*/,
                0/*addWeighted_gpu<short, signed char, float >*/,
                0/*addWeighted_gpu<short, signed char, double>*/
            },
            {
                0/*addWeighted_gpu<short, unsigned short, unsigned char >*/,
                0/*addWeighted_gpu<short, unsigned short, signed char >*/,
                0/*addWeighted_gpu<short, unsigned short, unsigned short>*/,
                0/*addWeighted_gpu<short, unsigned short, short >*/,
                0/*addWeighted_gpu<short, unsigned short, int   >*/,
                0/*addWeighted_gpu<short, unsigned short, float >*/,
                0/*addWeighted_gpu<short, unsigned short, double>*/
            },
            {
                addWeighted_gpu<short, short, unsigned char >,
                addWeighted_gpu<short, short, signed char >,
                addWeighted_gpu<short, short, unsigned short>,
                addWeighted_gpu<short, short, short >,
                addWeighted_gpu<short, short, int   >,
                addWeighted_gpu<short, short, float >,
                addWeighted_gpu<short, short, double>
            },
            {
                addWeighted_gpu<short, int, unsigned char >,
                addWeighted_gpu<short, int, signed char >,
                addWeighted_gpu<short, int, unsigned short>,
                addWeighted_gpu<short, int, short >,
                addWeighted_gpu<short, int, int   >,
                addWeighted_gpu<short, int, float >,
                addWeighted_gpu<short, int, double>
            },
            {
                addWeighted_gpu<short, float, unsigned char >,
                addWeighted_gpu<short, float, signed char >,
                addWeighted_gpu<short, float, unsigned short>,
                addWeighted_gpu<short, float, short >,
                addWeighted_gpu<short, float, int   >,
                addWeighted_gpu<short, float, float >,
                addWeighted_gpu<short, float, double>
            },
            {
                addWeighted_gpu<short, double, unsigned char >,
                addWeighted_gpu<short, double, signed char >,
                addWeighted_gpu<short, double, unsigned short>,
                addWeighted_gpu<short, double, short >,
                addWeighted_gpu<short, double, int   >,
                addWeighted_gpu<short, double, float >,
                addWeighted_gpu<short, double, double>
            }
        },
        {
            {
                0/*addWeighted_gpu<int, unsigned char, unsigned char >*/,
                0/*addWeighted_gpu<int, unsigned char, signed char >*/,
                0/*addWeighted_gpu<int, unsigned char, unsigned short>*/,
                0/*addWeighted_gpu<int, unsigned char, short >*/,
                0/*addWeighted_gpu<int, unsigned char, int   >*/,
                0/*addWeighted_gpu<int, unsigned char, float >*/,
                0/*addWeighted_gpu<int, unsigned char, double>*/
            },
            {
                0/*addWeighted_gpu<int, signed char, unsigned char >*/,
                0/*addWeighted_gpu<int, signed char, signed char >*/,
                0/*addWeighted_gpu<int, signed char, unsigned short>*/,
                0/*addWeighted_gpu<int, signed char, short >*/,
                0/*addWeighted_gpu<int, signed char, int   >*/,
                0/*addWeighted_gpu<int, signed char, float >*/,
                0/*addWeighted_gpu<int, signed char, double>*/
            },
            {
                0/*addWeighted_gpu<int, unsigned short, unsigned char >*/,
                0/*addWeighted_gpu<int, unsigned short, signed char >*/,
                0/*addWeighted_gpu<int, unsigned short, unsigned short>*/,
                0/*addWeighted_gpu<int, unsigned short, short >*/,
                0/*addWeighted_gpu<int, unsigned short, int   >*/,
                0/*addWeighted_gpu<int, unsigned short, float >*/,
                0/*addWeighted_gpu<int, unsigned short, double>*/
            },
            {
                0/*addWeighted_gpu<int, short, unsigned char >*/,
                0/*addWeighted_gpu<int, short, signed char >*/,
                0/*addWeighted_gpu<int, short, unsigned short>*/,
                0/*addWeighted_gpu<int, short, short >*/,
                0/*addWeighted_gpu<int, short, int   >*/,
                0/*addWeighted_gpu<int, short, float >*/,
                0/*addWeighted_gpu<int, short, double>*/
            },
            {
                addWeighted_gpu<int, int, unsigned char >,
                addWeighted_gpu<int, int, signed char >,
                addWeighted_gpu<int, int, unsigned short>,
                addWeighted_gpu<int, int, short >,
                addWeighted_gpu<int, int, int   >,
                addWeighted_gpu<int, int, float >,
                addWeighted_gpu<int, int, double>
            },
            {
                addWeighted_gpu<int, float, unsigned char >,
                addWeighted_gpu<int, float, signed char >,
                addWeighted_gpu<int, float, unsigned short>,
                addWeighted_gpu<int, float, short >,
                addWeighted_gpu<int, float, int   >,
                addWeighted_gpu<int, float, float >,
                addWeighted_gpu<int, float, double>
            },
            {
                addWeighted_gpu<int, double, unsigned char >,
                addWeighted_gpu<int, double, signed char >,
                addWeighted_gpu<int, double, unsigned short>,
                addWeighted_gpu<int, double, short >,
                addWeighted_gpu<int, double, int   >,
                addWeighted_gpu<int, double, float >,
                addWeighted_gpu<int, double, double>
            }
        },
        {
            {
                0/*addWeighted_gpu<float, unsigned char, unsigned char >*/,
                0/*addWeighted_gpu<float, unsigned char, signed char >*/,
                0/*addWeighted_gpu<float, unsigned char, unsigned short>*/,
                0/*addWeighted_gpu<float, unsigned char, short >*/,
                0/*addWeighted_gpu<float, unsigned char, int   >*/,
                0/*addWeighted_gpu<float, unsigned char, float >*/,
                0/*addWeighted_gpu<float, unsigned char, double>*/
            },
            {
                0/*addWeighted_gpu<float, signed char, unsigned char >*/,
                0/*addWeighted_gpu<float, signed char, signed char >*/,
                0/*addWeighted_gpu<float, signed char, unsigned short>*/,
                0/*addWeighted_gpu<float, signed char, short >*/,
                0/*addWeighted_gpu<float, signed char, int   >*/,
                0/*addWeighted_gpu<float, signed char, float >*/,
                0/*addWeighted_gpu<float, signed char, double>*/
            },
            {
                0/*addWeighted_gpu<float, unsigned short, unsigned char >*/,
                0/*addWeighted_gpu<float, unsigned short, signed char >*/,
                0/*addWeighted_gpu<float, unsigned short, unsigned short>*/,
                0/*addWeighted_gpu<float, unsigned short, short >*/,
                0/*addWeighted_gpu<float, unsigned short, int   >*/,
                0/*addWeighted_gpu<float, unsigned short, float >*/,
                0/*addWeighted_gpu<float, unsigned short, double>*/
            },
            {
                0/*addWeighted_gpu<float, short, unsigned char >*/,
                0/*addWeighted_gpu<float, short, signed char >*/,
                0/*addWeighted_gpu<float, short, unsigned short>*/,
                0/*addWeighted_gpu<float, short, short >*/,
                0/*addWeighted_gpu<float, short, int   >*/,
                0/*addWeighted_gpu<float, short, float >*/,
                0/*addWeighted_gpu<float, short, double>*/
            },
            {
                0/*addWeighted_gpu<float, int, unsigned char >*/,
                0/*addWeighted_gpu<float, int, signed char >*/,
                0/*addWeighted_gpu<float, int, unsigned short>*/,
                0/*addWeighted_gpu<float, int, short >*/,
                0/*addWeighted_gpu<float, int, int   >*/,
                0/*addWeighted_gpu<float, int, float >*/,
                0/*addWeighted_gpu<float, int, double>*/
            },
            {
                addWeighted_gpu<float, float, unsigned char >,
                addWeighted_gpu<float, float, signed char >,
                addWeighted_gpu<float, float, unsigned short>,
                addWeighted_gpu<float, float, short >,
                addWeighted_gpu<float, float, int   >,
                addWeighted_gpu<float, float, float >,
                addWeighted_gpu<float, float, double>
            },
            {
                addWeighted_gpu<float, double, unsigned char >,
                addWeighted_gpu<float, double, signed char >,
                addWeighted_gpu<float, double, unsigned short>,
                addWeighted_gpu<float, double, short >,
                addWeighted_gpu<float, double, int   >,
                addWeighted_gpu<float, double, float >,
                addWeighted_gpu<float, double, double>
            }
        },
        {
            {
                0/*addWeighted_gpu<double, unsigned char, unsigned char >*/,
                0/*addWeighted_gpu<double, unsigned char, signed char >*/,
                0/*addWeighted_gpu<double, unsigned char, unsigned short>*/,
                0/*addWeighted_gpu<double, unsigned char, short >*/,
                0/*addWeighted_gpu<double, unsigned char, int   >*/,
                0/*addWeighted_gpu<double, unsigned char, float >*/,
                0/*addWeighted_gpu<double, unsigned char, double>*/
            },
            {
                0/*addWeighted_gpu<double, signed char, unsigned char >*/,
                0/*addWeighted_gpu<double, signed char, signed char >*/,
                0/*addWeighted_gpu<double, signed char, unsigned short>*/,
                0/*addWeighted_gpu<double, signed char, short >*/,
                0/*addWeighted_gpu<double, signed char, int   >*/,
                0/*addWeighted_gpu<double, signed char, float >*/,
                0/*addWeighted_gpu<double, signed char, double>*/
            },
            {
                0/*addWeighted_gpu<double, unsigned short, unsigned char >*/,
                0/*addWeighted_gpu<double, unsigned short, signed char >*/,
                0/*addWeighted_gpu<double, unsigned short, unsigned short>*/,
                0/*addWeighted_gpu<double, unsigned short, short >*/,
                0/*addWeighted_gpu<double, unsigned short, int   >*/,
                0/*addWeighted_gpu<double, unsigned short, float >*/,
                0/*addWeighted_gpu<double, unsigned short, double>*/
            },
            {
                0/*addWeighted_gpu<double, short, unsigned char >*/,
                0/*addWeighted_gpu<double, short, signed char >*/,
                0/*addWeighted_gpu<double, short, unsigned short>*/,
                0/*addWeighted_gpu<double, short, short >*/,
                0/*addWeighted_gpu<double, short, int   >*/,
                0/*addWeighted_gpu<double, short, float >*/,
                0/*addWeighted_gpu<double, short, double>*/
            },
            {
                0/*addWeighted_gpu<double, int, unsigned char >*/,
                0/*addWeighted_gpu<double, int, signed char >*/,
                0/*addWeighted_gpu<double, int, unsigned short>*/,
                0/*addWeighted_gpu<double, int, short >*/,
                0/*addWeighted_gpu<double, int, int   >*/,
                0/*addWeighted_gpu<double, int, float >*/,
                0/*addWeighted_gpu<double, int, double>*/
            },
            {
                0/*addWeighted_gpu<double, float, unsigned char >*/,
                0/*addWeighted_gpu<double, float, signed char >*/,
                0/*addWeighted_gpu<double, float, unsigned short>*/,
                0/*addWeighted_gpu<double, float, short >*/,
                0/*addWeighted_gpu<double, float, int   >*/,
                0/*addWeighted_gpu<double, float, float >*/,
                0/*addWeighted_gpu<double, float, double>*/
            },
            {
                addWeighted_gpu<double, double, unsigned char >,
                addWeighted_gpu<double, double, signed char >,
                addWeighted_gpu<double, double, unsigned short>,
                addWeighted_gpu<double, double, short >,
                addWeighted_gpu<double, double, int   >,
                addWeighted_gpu<double, double, float >,
                addWeighted_gpu<double, double, double>
            }
        }
    };

    callers[psrc1->depth()][psrc2->depth()][dst.depth()](psrc1->reshape(1), alpha, psrc2->reshape(1), beta, gamma, dst.reshape(1), StreamAccessor::getStream(stream));
}

#endif
