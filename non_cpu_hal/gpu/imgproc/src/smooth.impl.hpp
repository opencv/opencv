// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL, GPU: GaussianBlur, a row pass into a buffer, then a column pass. Compiled twice through
// precomp.hpp, which supplies the shared helpers: the kernels with CV_ACCEL_GPU_KERNELS, the
// operation that launches them without.

#include "gpu/imgproc/include/gpu_hal_imgproc.hpp"

#ifdef CV_ACCEL_GPU_KERNELS

namespace cv { namespace accel { namespace CV_GPU_NS { namespace kernels {

namespace {

// GaussianBlur's arithmetic per depth: taps K, row results W (the buffer), column sums A. For 8U
// and 16U it is the CPU's fixed point (smooth.simd.hpp): exact integer sums rounded once at the end
template<typename T> struct Blur { typedef float K; typedef float W; typedef float A; };
template<> struct Blur<unsigned char> { typedef unsigned short K; typedef unsigned short W; typedef unsigned A; };
template<> struct Blur<unsigned short> { typedef unsigned K; typedef unsigned W; typedef unsigned long long A; };

template<typename K> struct Taps { K v[GAUSSIAN_MAX_TAPS]; int n; };

//! a column sum back to T: the fixed-point ones carry 16 (8U) or 32 (16U) fraction bits
template<typename T> __device__ inline T blurResult(typename Blur<T>::A a) { return saturate<T>(a); }
template<> __device__ inline unsigned char blurResult<unsigned char>(unsigned a)
{
    const unsigned v = (a + (1u << 15)) >> 16;
    return (unsigned char)(v < 255u ? v : 255u);
}
template<> __device__ inline unsigned short blurResult<unsigned short>(unsigned long long a)
{
    const unsigned long long v = (a + (1ull << 31)) >> 32;
    return (unsigned short)(v < 65535ull ? v : 65535ull);
}

/* The taps @p k centred at position @p i of a line of @p len pixels, @p CN channels of S each,
   which are @p stride bytes apart (along a row or down a column); BORDER_CONSTANT adds zeros. */
template<typename A, int CN, typename K, typename S>
__device__ inline void convolve(const Taps<K>& k, int i, int len, int border, const S* line, size_t stride, A* acc)
{
    for (int c = 0; c < CN; c++)
        acc[c] = 0;
    for (int j = 0; j < k.n; j++)
    {
        const int p = borderIndex(i + j - k.n / 2, len, border);
        if (p < 0)
            continue;
        const S* q = (const S*)((const unsigned char*)line + (size_t)p * stride);
        for (int c = 0; c < CN; c++)
            acc[c] += (A)k.v[j] * q[c];
    }
}

template<typename T, int CN>
__global__ void blurRows(const unsigned char* src, size_t sstep, unsigned char* buf, size_t bstep,
                         int rows, int cols, Taps<typename Blur<T>::K> k, int border)
{
    typedef typename Blur<T>::W W;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= cols)
        return;
    for (unsigned y = blockIdx.y * blockDim.y + threadIdx.y; y < (unsigned)rows; y += gridDim.y * blockDim.y)
    {
        W acc[CN];
        convolve<W, CN>(k, x, cols, border, (const T*)(src + (size_t)y * sstep), CN * sizeof(T), acc);
        W* out = (W*)(buf + (size_t)y * bstep) + x * CN;
        for (int c = 0; c < CN; c++)
            out[c] = acc[c];
    }
}

template<typename T, int CN>
__global__ void blurCols(const unsigned char* buf, size_t bstep, unsigned char* dst, size_t dstep,
                         int rows, int cols, Taps<typename Blur<T>::K> k, int border)
{
    typedef typename Blur<T>::A A;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= cols)
        return;
    const typename Blur<T>::W* column = (const typename Blur<T>::W*)buf + x * CN;
    for (unsigned y = blockIdx.y * blockDim.y + threadIdx.y; y < (unsigned)rows; y += gridDim.y * blockDim.y)
    {
        A acc[CN];
        convolve<A, CN>(k, (int)y, rows, border, column, bstep, acc);
        T* out = (T*)(dst + (size_t)y * dstep) + x * CN;
        for (int c = 0; c < CN; c++)
            out[c] = blurResult<T>(acc[c]);
    }
}

template<typename T, int CN>
CV_GPU(Error_t) gaussianBlurN(const unsigned char* src, size_t sstep, unsigned char* buf, size_t bstep,
                              unsigned char* dst, size_t dstep, int rows, int cols,
                              const void* kx, int kxn, const void* ky, int kyn, int border, CV_GPU(Stream_t) s)
{
    typedef typename Blur<T>::K K;
    Taps<K> tx, ty;
    tx.n = kxn;
    ty.n = kyn;
    memcpy(tx.v, kx, kxn * sizeof(K));
    memcpy(ty.v, ky, kyn * sizeof(K));
    const dim3 g = grid(rows, cols, BLOCK_X, BLOCK_Y), b(BLOCK_X, BLOCK_Y);
    void* rowArgs[] = { &src, &sstep, &buf, &bstep, &rows, &cols, &tx, &border };
    const CV_GPU(Error_t) r = CV_GPU(LaunchKernel)((const void*)blurRows<T, CN>, g, b, rowArgs, 0, s);
    if (r != CV_GPU(Success))
        return r;
    void* colArgs[] = { &buf, &bstep, &dst, &dstep, &rows, &cols, &ty, &border };
    return CV_GPU(LaunchKernel)((const void*)blurCols<T, CN>, g, b, colArgs, 0, s);
}

} // namespace

CV_GPU(Error_t) gaussianBlur(const unsigned char* src, size_t sstep, unsigned char* buf, size_t bstep,
                             unsigned char* dst, size_t dstep, int rows, int cols, int depth, int cn,
                             const void* kx, int kxn, const void* ky, int kyn, int border, CV_GPU(Stream_t) s)
{
    if (rows <= 0 || cols <= 0)
        return CV_GPU(Success);
    if (kxn < 1 || kxn > GAUSSIAN_MAX_TAPS || kyn < 1 || kyn > GAUSSIAN_MAX_TAPS)
        return CV_GPU(ErrorInvalidValue);
    switch (depth)
    {
    case CV_8U: CV_ACCEL_GPU_BY_CN(gaussianBlurN, unsigned char, src, sstep, buf, bstep, dst, dstep, rows, cols, kx, kxn, ky, kyn, border, s)
    case CV_16U: CV_ACCEL_GPU_BY_CN(gaussianBlurN, unsigned short, src, sstep, buf, bstep, dst, dstep, rows, cols, kx, kxn, ky, kyn, border, s)
    case CV_16S: CV_ACCEL_GPU_BY_CN(gaussianBlurN, short, src, sstep, buf, bstep, dst, dstep, rows, cols, kx, kxn, ky, kyn, border, s)
    case CV_32F: CV_ACCEL_GPU_BY_CN(gaussianBlurN, float, src, sstep, buf, bstep, dst, dstep, rows, cols, kx, kxn, ky, kyn, border, s)
    default: return CV_GPU(ErrorInvalidValue);
    }
}

}}}} // namespace cv::accel::CV_GPU_NS::kernels

#else // the operation

namespace cv { namespace accel { namespace CV_GPU_NS {

namespace {

//! a centred kernel of odd length the GPU takes, of @p type
bool validTaps(const Mat& k, int type)
{
    const int n = (int)k.total();
    return k.type() == type && k.isContinuous() && n % 2 == 1 && n <= kernels::GAUSSIAN_MAX_TAPS;
}

// two passes through a buffer, so the output may alias the input in any way; BORDER_WRAP is
// left to the CPU, whose floating-point filters reject it
bool GpuImgprocOps::gaussianBlur(const UMat& src, UMat& dst, const Mat& kx, const Mat& ky, int borderType,
                                 Stream& stream)
{
    const int depth = src.depth();
    const int tapType = depth == CV_8U ? CV_16U : depth == CV_16U ? CV_32U : CV_32F;
    if (!usableImage(src) || !usableImage(dst) || borderType == BORDER_WRAP || borderType > BORDER_REFLECT_101 ||
        (depth != CV_8U && depth != CV_16U && depth != CV_16S && depth != CV_32F) ||
        !validTaps(kx, tapType) || !validTaps(ky, tapType))
        return false;
    GpuStream& s = gpuStreamOf(stream);
    UMat buf;
    buf.createOn(getDevice(src), src.size(), CV_MAKETYPE(tapType, src.channels()));
    DeviceGuard g(s.device_);
    const unsigned char* sp = (const unsigned char*)src.handle(ACCESS_READ) + src.offset;
    unsigned char* bp = (unsigned char*)buf.handle(ACCESS_WRITE);
    unsigned char* dp = (unsigned char*)dst.handle(ACCESS_WRITE) + dst.offset;
    if (!launched(kernels::gaussianBlur(sp, src.step[0], bp, buf.step[0], dp, dst.step[0], src.rows, src.cols,
                                        depth, src.channels(), kx.ptr(), (int)kx.total(), ky.ptr(), (int)ky.total(),
                                        borderType, s.s_), "GaussianBlur"))
        return false;
    stream.retain({ src, buf, dst });
    return true;
}

} // namespace

}}} // namespace cv::accel::CV_GPU_NS

#endif // CV_ACCEL_GPU_KERNELS
