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

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

void cv::gpu::gemm(const GpuMat&, const GpuMat&, double, const GpuMat&, double, GpuMat&, int, Stream&) { throw_no_cuda(); }

void cv::gpu::integral(const GpuMat&, GpuMat&, Stream&) { throw_no_cuda(); }
void cv::gpu::integralBuffered(const GpuMat&, GpuMat&, GpuMat&, Stream&) { throw_no_cuda(); }

void cv::gpu::sqrIntegral(const GpuMat&, GpuMat&, Stream&) { throw_no_cuda(); }

void cv::gpu::mulSpectrums(const GpuMat&, const GpuMat&, GpuMat&, int, bool, Stream&) { throw_no_cuda(); }
void cv::gpu::mulAndScaleSpectrums(const GpuMat&, const GpuMat&, GpuMat&, int, float, bool, Stream&) { throw_no_cuda(); }

void cv::gpu::dft(const GpuMat&, GpuMat&, Size, int, Stream&) { throw_no_cuda(); }

void cv::gpu::ConvolveBuf::create(Size, Size) { throw_no_cuda(); }
void cv::gpu::convolve(const GpuMat&, const GpuMat&, GpuMat&, bool) { throw_no_cuda(); }
void cv::gpu::convolve(const GpuMat&, const GpuMat&, GpuMat&, bool, ConvolveBuf&, Stream&) { throw_no_cuda(); }

#else /* !defined (HAVE_CUDA) */

namespace
{
    #define error_entry(entry)  { entry, #entry }

    struct ErrorEntry
    {
        int code;
        const char* str;
    };

    struct ErrorEntryComparer
    {
        int code;
        ErrorEntryComparer(int code_) : code(code_) {}
        bool operator()(const ErrorEntry& e) const { return e.code == code; }
    };

    String getErrorString(int code, const ErrorEntry* errors, size_t n)
    {
        size_t idx = std::find_if(errors, errors + n, ErrorEntryComparer(code)) - errors;

        const char* msg = (idx != n) ? errors[idx].str : "Unknown error code";
        String str = cv::format("%s [Code = %d]", msg, code);

        return str;
    }
}

#ifdef HAVE_CUBLAS
    namespace
    {
        const ErrorEntry cublas_errors[] =
        {
            error_entry( CUBLAS_STATUS_SUCCESS ),
            error_entry( CUBLAS_STATUS_NOT_INITIALIZED ),
            error_entry( CUBLAS_STATUS_ALLOC_FAILED ),
            error_entry( CUBLAS_STATUS_INVALID_VALUE ),
            error_entry( CUBLAS_STATUS_ARCH_MISMATCH ),
            error_entry( CUBLAS_STATUS_MAPPING_ERROR ),
            error_entry( CUBLAS_STATUS_EXECUTION_FAILED ),
            error_entry( CUBLAS_STATUS_INTERNAL_ERROR )
        };

        const size_t cublas_error_num = sizeof(cublas_errors) / sizeof(cublas_errors[0]);

        static inline void ___cublasSafeCall(cublasStatus_t err, const char* file, const int line, const char* func)
        {
            if (CUBLAS_STATUS_SUCCESS != err)
            {
                String msg = getErrorString(err, cublas_errors, cublas_error_num);
                cv::error(cv::Error::GpuApiCallError, msg, func, file, line);
            }
        }
    }

    #if defined(__GNUC__)
        #define cublasSafeCall(expr)  ___cublasSafeCall(expr, __FILE__, __LINE__, __func__)
    #else /* defined(__CUDACC__) || defined(__MSVC__) */
        #define cublasSafeCall(expr)  ___cublasSafeCall(expr, __FILE__, __LINE__, "")
    #endif
#endif // HAVE_CUBLAS

#ifdef HAVE_CUFFT
    namespace
    {
        //////////////////////////////////////////////////////////////////////////
        // CUFFT errors

        const ErrorEntry cufft_errors[] =
        {
            error_entry( CUFFT_INVALID_PLAN ),
            error_entry( CUFFT_ALLOC_FAILED ),
            error_entry( CUFFT_INVALID_TYPE ),
            error_entry( CUFFT_INVALID_VALUE ),
            error_entry( CUFFT_INTERNAL_ERROR ),
            error_entry( CUFFT_EXEC_FAILED ),
            error_entry( CUFFT_SETUP_FAILED ),
            error_entry( CUFFT_INVALID_SIZE ),
            error_entry( CUFFT_UNALIGNED_DATA )
        };

        const int cufft_error_num = sizeof(cufft_errors) / sizeof(cufft_errors[0]);

        void ___cufftSafeCall(int err, const char* file, const int line, const char* func)
        {
            if (CUFFT_SUCCESS != err)
            {
                String msg = getErrorString(err, cufft_errors, cufft_error_num);
                cv::error(cv::Error::GpuApiCallError, msg, func, file, line);
            }
        }
    }

    #if defined(__GNUC__)
        #define cufftSafeCall(expr)  ___cufftSafeCall(expr, __FILE__, __LINE__, __func__)
    #else /* defined(__CUDACC__) || defined(__MSVC__) */
        #define cufftSafeCall(expr)  ___cufftSafeCall(expr, __FILE__, __LINE__, "")
    #endif

#endif

////////////////////////////////////////////////////////////////////////
// gemm

void cv::gpu::gemm(const GpuMat& src1, const GpuMat& src2, double alpha, const GpuMat& src3, double beta, GpuMat& dst, int flags, Stream& stream)
{
#ifndef HAVE_CUBLAS
    (void)src1;
    (void)src2;
    (void)alpha;
    (void)src3;
    (void)beta;
    (void)dst;
    (void)flags;
    (void)stream;
    CV_Error(cv::Error::StsNotImplemented, "The library was build without CUBLAS");
#else
    // CUBLAS works with column-major matrices

    CV_Assert(src1.type() == CV_32FC1 || src1.type() == CV_32FC2 || src1.type() == CV_64FC1 || src1.type() == CV_64FC2);
    CV_Assert(src2.type() == src1.type() && (src3.empty() || src3.type() == src1.type()));

    if (src1.depth() == CV_64F)
    {
        if (!deviceSupports(NATIVE_DOUBLE))
            CV_Error(cv::Error::StsUnsupportedFormat, "The device doesn't support double");
    }

    bool tr1 = (flags & GEMM_1_T) != 0;
    bool tr2 = (flags & GEMM_2_T) != 0;
    bool tr3 = (flags & GEMM_3_T) != 0;

    if (src1.type() == CV_64FC2)
    {
        if (tr1 || tr2 || tr3)
            CV_Error(cv::Error::StsNotImplemented, "transpose operation doesn't implemented for CV_64FC2 type");
    }

    Size src1Size = tr1 ? Size(src1.rows, src1.cols) : src1.size();
    Size src2Size = tr2 ? Size(src2.rows, src2.cols) : src2.size();
    Size src3Size = tr3 ? Size(src3.rows, src3.cols) : src3.size();
    Size dstSize(src2Size.width, src1Size.height);

    CV_Assert(src1Size.width == src2Size.height);
    CV_Assert(src3.empty() || src3Size == dstSize);

    dst.create(dstSize, src1.type());

    if (beta != 0)
    {
        if (src3.empty())
        {
            if (stream)
                stream.enqueueMemSet(dst, Scalar::all(0));
            else
                dst.setTo(Scalar::all(0));
        }
        else
        {
            if (tr3)
            {
                gpu::transpose(src3, dst, stream);
            }
            else
            {
                if (stream)
                    stream.enqueueCopy(src3, dst);
                else
                    src3.copyTo(dst);
            }
        }
    }

    cublasHandle_t handle;
    cublasSafeCall( cublasCreate_v2(&handle) );

    cublasSafeCall( cublasSetStream_v2(handle, StreamAccessor::getStream(stream)) );

    cublasSafeCall( cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST) );

    const float alphaf = static_cast<float>(alpha);
    const float betaf = static_cast<float>(beta);

    const cuComplex alphacf = make_cuComplex(alphaf, 0);
    const cuComplex betacf = make_cuComplex(betaf, 0);

    const cuDoubleComplex alphac = make_cuDoubleComplex(alpha, 0);
    const cuDoubleComplex betac = make_cuDoubleComplex(beta, 0);

    cublasOperation_t transa = tr2 ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = tr1 ? CUBLAS_OP_T : CUBLAS_OP_N;

    switch (src1.type())
    {
    case CV_32FC1:
        cublasSafeCall( cublasSgemm_v2(handle, transa, transb, tr2 ? src2.rows : src2.cols, tr1 ? src1.cols : src1.rows, tr2 ? src2.cols : src2.rows,
            &alphaf,
            src2.ptr<float>(), static_cast<int>(src2.step / sizeof(float)),
            src1.ptr<float>(), static_cast<int>(src1.step / sizeof(float)),
            &betaf,
            dst.ptr<float>(), static_cast<int>(dst.step / sizeof(float))) );
        break;

    case CV_64FC1:
        cublasSafeCall( cublasDgemm_v2(handle, transa, transb, tr2 ? src2.rows : src2.cols, tr1 ? src1.cols : src1.rows, tr2 ? src2.cols : src2.rows,
            &alpha,
            src2.ptr<double>(), static_cast<int>(src2.step / sizeof(double)),
            src1.ptr<double>(), static_cast<int>(src1.step / sizeof(double)),
            &beta,
            dst.ptr<double>(), static_cast<int>(dst.step / sizeof(double))) );
        break;

    case CV_32FC2:
        cublasSafeCall( cublasCgemm_v2(handle, transa, transb, tr2 ? src2.rows : src2.cols, tr1 ? src1.cols : src1.rows, tr2 ? src2.cols : src2.rows,
            &alphacf,
            src2.ptr<cuComplex>(), static_cast<int>(src2.step / sizeof(cuComplex)),
            src1.ptr<cuComplex>(), static_cast<int>(src1.step / sizeof(cuComplex)),
            &betacf,
            dst.ptr<cuComplex>(), static_cast<int>(dst.step / sizeof(cuComplex))) );
        break;

    case CV_64FC2:
        cublasSafeCall( cublasZgemm_v2(handle, transa, transb, tr2 ? src2.rows : src2.cols, tr1 ? src1.cols : src1.rows, tr2 ? src2.cols : src2.rows,
            &alphac,
            src2.ptr<cuDoubleComplex>(), static_cast<int>(src2.step / sizeof(cuDoubleComplex)),
            src1.ptr<cuDoubleComplex>(), static_cast<int>(src1.step / sizeof(cuDoubleComplex)),
            &betac,
            dst.ptr<cuDoubleComplex>(), static_cast<int>(dst.step / sizeof(cuDoubleComplex))) );
        break;
    }

    cublasSafeCall( cublasDestroy_v2(handle) );
#endif
}

////////////////////////////////////////////////////////////////////////
// integral

void cv::gpu::integral(const GpuMat& src, GpuMat& sum, Stream& s)
{
    GpuMat buffer;
    gpu::integralBuffered(src, sum, buffer, s);
}

namespace cv { namespace gpu { namespace cudev
{
    namespace imgproc
    {
        void shfl_integral_gpu(const PtrStepSzb& img, PtrStepSz<unsigned int> integral, cudaStream_t stream);
    }
}}}

void cv::gpu::integralBuffered(const GpuMat& src, GpuMat& sum, GpuMat& buffer, Stream& s)
{
    CV_Assert(src.type() == CV_8UC1);

    cudaStream_t stream = StreamAccessor::getStream(s);

    cv::Size whole;
    cv::Point offset;

    src.locateROI(whole, offset);

    if (deviceSupports(WARP_SHUFFLE_FUNCTIONS) && src.cols <= 2048
        && offset.x % 16 == 0 && ((src.cols + 63) / 64) * 64 <= (static_cast<int>(src.step) - offset.x))
    {
        ensureSizeIsEnough(((src.rows + 7) / 8) * 8, ((src.cols + 63) / 64) * 64, CV_32SC1, buffer);

        cv::gpu::cudev::imgproc::shfl_integral_gpu(src, buffer, stream);

        sum.create(src.rows + 1, src.cols + 1, CV_32SC1);
        if (s)
            s.enqueueMemSet(sum, Scalar::all(0));
        else
            sum.setTo(Scalar::all(0));

        GpuMat inner = sum(Rect(1, 1, src.cols, src.rows));
        GpuMat res = buffer(Rect(0, 0, src.cols, src.rows));

        if (s)
            s.enqueueCopy(res, inner);
        else
            res.copyTo(inner);
    }
    else
    {
#ifndef HAVE_OPENCV_GPULEGACY
    throw_no_cuda();
#else
        sum.create(src.rows + 1, src.cols + 1, CV_32SC1);

        NcvSize32u roiSize;
        roiSize.width = src.cols;
        roiSize.height = src.rows;

        cudaDeviceProp prop;
        cudaSafeCall( cudaGetDeviceProperties(&prop, cv::gpu::getDevice()) );

        Ncv32u bufSize;
        ncvSafeCall( nppiStIntegralGetSize_8u32u(roiSize, &bufSize, prop) );
        ensureSizeIsEnough(1, bufSize, CV_8UC1, buffer);

        NppStStreamHandler h(stream);

        ncvSafeCall( nppiStIntegral_8u32u_C1R(const_cast<Ncv8u*>(src.ptr<Ncv8u>()), static_cast<int>(src.step),
            sum.ptr<Ncv32u>(), static_cast<int>(sum.step), roiSize, buffer.ptr<Ncv8u>(), bufSize, prop) );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
#endif
    }
}

//////////////////////////////////////////////////////////////////////////////
// sqrIntegral

void cv::gpu::sqrIntegral(const GpuMat& src, GpuMat& sqsum, Stream& s)
{
#ifndef HAVE_OPENCV_GPULEGACY
    (void) src;
    (void) sqsum;
    (void) s;
    throw_no_cuda();
#else
    CV_Assert(src.type() == CV_8U);

    NcvSize32u roiSize;
    roiSize.width = src.cols;
    roiSize.height = src.rows;

    cudaDeviceProp prop;
    cudaSafeCall( cudaGetDeviceProperties(&prop, cv::gpu::getDevice()) );

    Ncv32u bufSize;
    ncvSafeCall(nppiStSqrIntegralGetSize_8u64u(roiSize, &bufSize, prop));
    GpuMat buf(1, bufSize, CV_8U);

    cudaStream_t stream = StreamAccessor::getStream(s);

    NppStStreamHandler h(stream);

    sqsum.create(src.rows + 1, src.cols + 1, CV_64F);
    ncvSafeCall(nppiStSqrIntegral_8u64u_C1R(const_cast<Ncv8u*>(src.ptr<Ncv8u>(0)), static_cast<int>(src.step),
            sqsum.ptr<Ncv64u>(0), static_cast<int>(sqsum.step), roiSize, buf.ptr<Ncv8u>(0), bufSize, prop));

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
#endif
}

//////////////////////////////////////////////////////////////////////////////
// mulSpectrums

#ifdef HAVE_CUFFT

namespace cv { namespace gpu { namespace cudev
{
    void mulSpectrums(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, PtrStepSz<cufftComplex> c, cudaStream_t stream);

    void mulSpectrums_CONJ(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, PtrStepSz<cufftComplex> c, cudaStream_t stream);
}}}

#endif

void cv::gpu::mulSpectrums(const GpuMat& a, const GpuMat& b, GpuMat& c, int flags, bool conjB, Stream& stream)
{
#ifndef HAVE_CUFFT
    (void) a;
    (void) b;
    (void) c;
    (void) flags;
    (void) conjB;
    (void) stream;
    throw_no_cuda();
#else
    (void) flags;

    typedef void (*Caller)(const PtrStep<cufftComplex>, const PtrStep<cufftComplex>, PtrStepSz<cufftComplex>, cudaStream_t stream);

    static Caller callers[] = { cudev::mulSpectrums, cudev::mulSpectrums_CONJ };

    CV_Assert(a.type() == b.type() && a.type() == CV_32FC2);
    CV_Assert(a.size() == b.size());

    c.create(a.size(), CV_32FC2);

    Caller caller = callers[(int)conjB];
    caller(a, b, c, StreamAccessor::getStream(stream));
#endif
}

//////////////////////////////////////////////////////////////////////////////
// mulAndScaleSpectrums

#ifdef HAVE_CUFFT

namespace cv { namespace gpu { namespace cudev
{
    void mulAndScaleSpectrums(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, float scale, PtrStepSz<cufftComplex> c, cudaStream_t stream);

    void mulAndScaleSpectrums_CONJ(const PtrStep<cufftComplex> a, const PtrStep<cufftComplex> b, float scale, PtrStepSz<cufftComplex> c, cudaStream_t stream);
}}}

#endif

void cv::gpu::mulAndScaleSpectrums(const GpuMat& a, const GpuMat& b, GpuMat& c, int flags, float scale, bool conjB, Stream& stream)
{
#ifndef HAVE_CUFFT
    (void) a;
    (void) b;
    (void) c;
    (void) flags;
    (void) scale;
    (void) conjB;
    (void) stream;
    throw_no_cuda();
#else
    (void)flags;

    typedef void (*Caller)(const PtrStep<cufftComplex>, const PtrStep<cufftComplex>, float scale, PtrStepSz<cufftComplex>, cudaStream_t stream);
    static Caller callers[] = { cudev::mulAndScaleSpectrums, cudev::mulAndScaleSpectrums_CONJ };

    CV_Assert(a.type() == b.type() && a.type() == CV_32FC2);
    CV_Assert(a.size() == b.size());

    c.create(a.size(), CV_32FC2);

    Caller caller = callers[(int)conjB];
    caller(a, b, scale, c, StreamAccessor::getStream(stream));
#endif
}

//////////////////////////////////////////////////////////////////////////////
// dft

void cv::gpu::dft(const GpuMat& src, GpuMat& dst, Size dft_size, int flags, Stream& stream)
{
#ifndef HAVE_CUFFT
    (void) src;
    (void) dst;
    (void) dft_size;
    (void) flags;
    (void) stream;
    throw_no_cuda();
#else

    CV_Assert(src.type() == CV_32F || src.type() == CV_32FC2);

    // We don't support unpacked output (in the case of real input)
    CV_Assert(!(flags & DFT_COMPLEX_OUTPUT));

    bool is_1d_input = (dft_size.height == 1) || (dft_size.width == 1);
    int is_row_dft = flags & DFT_ROWS;
    int is_scaled_dft = flags & DFT_SCALE;
    int is_inverse = flags & DFT_INVERSE;
    bool is_complex_input = src.channels() == 2;
    bool is_complex_output = !(flags & DFT_REAL_OUTPUT);

    // We don't support real-to-real transform
    CV_Assert(is_complex_input || is_complex_output);

    GpuMat src_data;

    // Make sure here we work with the continuous input,
    // as CUFFT can't handle gaps
    src_data = src;
    createContinuous(src.rows, src.cols, src.type(), src_data);
    if (src_data.data != src.data)
        src.copyTo(src_data);

    Size dft_size_opt = dft_size;
    if (is_1d_input && !is_row_dft)
    {
        // If the source matrix is single column handle it as single row
        dft_size_opt.width = std::max(dft_size.width, dft_size.height);
        dft_size_opt.height = std::min(dft_size.width, dft_size.height);
    }

    cufftType dft_type = CUFFT_R2C;
    if (is_complex_input)
        dft_type = is_complex_output ? CUFFT_C2C : CUFFT_C2R;

    CV_Assert(dft_size_opt.width > 1);

    cufftHandle plan;
    if (is_1d_input || is_row_dft)
        cufftPlan1d(&plan, dft_size_opt.width, dft_type, dft_size_opt.height);
    else
        cufftPlan2d(&plan, dft_size_opt.height, dft_size_opt.width, dft_type);

    cufftSafeCall( cufftSetStream(plan, StreamAccessor::getStream(stream)) );

    if (is_complex_input)
    {
        if (is_complex_output)
        {
            createContinuous(dft_size, CV_32FC2, dst);
            cufftSafeCall(cufftExecC2C(
                    plan, src_data.ptr<cufftComplex>(), dst.ptr<cufftComplex>(),
                    is_inverse ? CUFFT_INVERSE : CUFFT_FORWARD));
        }
        else
        {
            createContinuous(dft_size, CV_32F, dst);
            cufftSafeCall(cufftExecC2R(
                    plan, src_data.ptr<cufftComplex>(), dst.ptr<cufftReal>()));
        }
    }
    else
    {
        // We could swap dft_size for efficiency. Here we must reflect it
        if (dft_size == dft_size_opt)
            createContinuous(Size(dft_size.width / 2 + 1, dft_size.height), CV_32FC2, dst);
        else
            createContinuous(Size(dft_size.width, dft_size.height / 2 + 1), CV_32FC2, dst);

        cufftSafeCall(cufftExecR2C(
                plan, src_data.ptr<cufftReal>(), dst.ptr<cufftComplex>()));
    }

    cufftSafeCall(cufftDestroy(plan));

    if (is_scaled_dft)
        multiply(dst, Scalar::all(1. / dft_size.area()), dst, 1, -1, stream);

#endif
}

//////////////////////////////////////////////////////////////////////////////
// convolve

void cv::gpu::ConvolveBuf::create(Size image_size, Size templ_size)
{
    result_size = Size(image_size.width - templ_size.width + 1,
                       image_size.height - templ_size.height + 1);

    block_size = user_block_size;
    if (user_block_size.width == 0 || user_block_size.height == 0)
        block_size = estimateBlockSize(result_size, templ_size);

    dft_size.width = 1 << int(ceil(std::log(block_size.width + templ_size.width - 1.) / std::log(2.)));
    dft_size.height = 1 << int(ceil(std::log(block_size.height + templ_size.height - 1.) / std::log(2.)));

    // CUFFT has hard-coded kernels for power-of-2 sizes (up to 8192),
    // see CUDA Toolkit 4.1 CUFFT Library Programming Guide
    if (dft_size.width > 8192)
        dft_size.width = getOptimalDFTSize(block_size.width + templ_size.width - 1);
    if (dft_size.height > 8192)
        dft_size.height = getOptimalDFTSize(block_size.height + templ_size.height - 1);

    // To avoid wasting time doing small DFTs
    dft_size.width = std::max(dft_size.width, 512);
    dft_size.height = std::max(dft_size.height, 512);

    createContinuous(dft_size, CV_32F, image_block);
    createContinuous(dft_size, CV_32F, templ_block);
    createContinuous(dft_size, CV_32F, result_data);

    spect_len = dft_size.height * (dft_size.width / 2 + 1);
    createContinuous(1, spect_len, CV_32FC2, image_spect);
    createContinuous(1, spect_len, CV_32FC2, templ_spect);
    createContinuous(1, spect_len, CV_32FC2, result_spect);

    // Use maximum result matrix block size for the estimated DFT block size
    block_size.width = std::min(dft_size.width - templ_size.width + 1, result_size.width);
    block_size.height = std::min(dft_size.height - templ_size.height + 1, result_size.height);
}


Size cv::gpu::ConvolveBuf::estimateBlockSize(Size result_size, Size /*templ_size*/)
{
    int width = (result_size.width + 2) / 3;
    int height = (result_size.height + 2) / 3;
    width = std::min(width, result_size.width);
    height = std::min(height, result_size.height);
    return Size(width, height);
}


void cv::gpu::convolve(const GpuMat& image, const GpuMat& templ, GpuMat& result, bool ccorr)
{
    ConvolveBuf buf;
    gpu::convolve(image, templ, result, ccorr, buf);
}

void cv::gpu::convolve(const GpuMat& image, const GpuMat& templ, GpuMat& result, bool ccorr, ConvolveBuf& buf, Stream& stream)
{
#ifndef HAVE_CUFFT
    (void) image;
    (void) templ;
    (void) result;
    (void) ccorr;
    (void) buf;
    (void) stream;
    throw_no_cuda();
#else
    using namespace cv::gpu::cudev::imgproc;

    CV_Assert(image.type() == CV_32F);
    CV_Assert(templ.type() == CV_32F);

    buf.create(image.size(), templ.size());
    result.create(buf.result_size, CV_32F);

    Size& block_size = buf.block_size;
    Size& dft_size = buf.dft_size;

    GpuMat& image_block = buf.image_block;
    GpuMat& templ_block = buf.templ_block;
    GpuMat& result_data = buf.result_data;

    GpuMat& image_spect = buf.image_spect;
    GpuMat& templ_spect = buf.templ_spect;
    GpuMat& result_spect = buf.result_spect;

    cufftHandle planR2C, planC2R;
    cufftSafeCall(cufftPlan2d(&planC2R, dft_size.height, dft_size.width, CUFFT_C2R));
    cufftSafeCall(cufftPlan2d(&planR2C, dft_size.height, dft_size.width, CUFFT_R2C));

    cufftSafeCall( cufftSetStream(planR2C, StreamAccessor::getStream(stream)) );
    cufftSafeCall( cufftSetStream(planC2R, StreamAccessor::getStream(stream)) );

    GpuMat templ_roi(templ.size(), CV_32F, templ.data, templ.step);
    gpu::copyMakeBorder(templ_roi, templ_block, 0, templ_block.rows - templ_roi.rows, 0,
                        templ_block.cols - templ_roi.cols, 0, Scalar(), stream);

    cufftSafeCall(cufftExecR2C(planR2C, templ_block.ptr<cufftReal>(),
                               templ_spect.ptr<cufftComplex>()));

    // Process all blocks of the result matrix
    for (int y = 0; y < result.rows; y += block_size.height)
    {
        for (int x = 0; x < result.cols; x += block_size.width)
        {
            Size image_roi_size(std::min(x + dft_size.width, image.cols) - x,
                                std::min(y + dft_size.height, image.rows) - y);
            GpuMat image_roi(image_roi_size, CV_32F, (void*)(image.ptr<float>(y) + x),
                             image.step);
            gpu::copyMakeBorder(image_roi, image_block, 0, image_block.rows - image_roi.rows,
                                0, image_block.cols - image_roi.cols, 0, Scalar(), stream);

            cufftSafeCall(cufftExecR2C(planR2C, image_block.ptr<cufftReal>(),
                                       image_spect.ptr<cufftComplex>()));
            gpu::mulAndScaleSpectrums(image_spect, templ_spect, result_spect, 0,
                                      1.f / dft_size.area(), ccorr, stream);
            cufftSafeCall(cufftExecC2R(planC2R, result_spect.ptr<cufftComplex>(),
                                       result_data.ptr<cufftReal>()));

            Size result_roi_size(std::min(x + block_size.width, result.cols) - x,
                                 std::min(y + block_size.height, result.rows) - y);
            GpuMat result_roi(result_roi_size, result.type(),
                              (void*)(result.ptr<float>(y) + x), result.step);
            GpuMat result_block(result_roi_size, result_data.type(),
                                result_data.ptr(), result_data.step);

            if (stream)
                stream.enqueueCopy(result_block, result_roi);
            else
                result_block.copyTo(result_roi);
        }
    }

    cufftSafeCall(cufftDestroy(planR2C));
    cufftSafeCall(cufftDestroy(planC2R));
#endif
}

#endif /* !defined (HAVE_CUDA) */
