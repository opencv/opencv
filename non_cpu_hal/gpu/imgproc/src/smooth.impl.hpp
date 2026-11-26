// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL, GPU: GaussianBlur, a row pass into a buffer, then a column pass. Compiled twice through
// ../../precomp.hpp: the kernels with CV_ACCEL_GPU_KERNELS, the operation that launches them without.

namespace cv { namespace accel { namespace CV_GPU_NS { namespace kernels {

//! the longest GaussianBlur kernel, in taps
const int GAUSSIAN_MAX_TAPS = 63;
/** @brief GaussianBlur, a row pass into @p buf launched as @p rowPass, then a column pass as @p colPass.
@p kx, @p ky and @p buf are CV_16U for CV_8U images, CV_32U for CV_16U (fixed point, as the CPU's), CV_32F otherwise. */
CV_GPU(Error_t) gaussianBlur(const unsigned char* src, size_t sstep, unsigned char* buf, size_t bstep,
                             unsigned char* dst, size_t dstep, int rows, int cols, int depth, int cn,
                             const void* kx, int kxn, const void* ky, int kyn, int border,
                             const Launch& rowPass, const Launch& colPass, CV_GPU(Stream_t) s);

}}}} // namespace cv::accel::CV_GPU_NS::kernels

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

/* The taps @p k along @p lines of M lines @p lineStep bytes apart: N outputs each from @p i0 of @p len pixels @p stride
   bytes apart. Input t feeds output i through tap t - i, so each is read once; BORDER_CONSTANT adds zeros. */
template<typename A, int M, int N, int CN, typename K, typename S>
__device__ inline void convolve(const Taps<K>& k, int i0, int len, int border, const S* base, size_t stride,
                                size_t lineStep, int lines, A (&acc)[M][N][CN])
{
#pragma unroll
    for (int m = 0; m < M; m++)
    {
#pragma unroll
        for (int i = 0; i < N; i++)
            for (int c = 0; c < CN; c++)
                acc[m][i][c] = 0;
    }
    for (int t = 0; t < k.n + N - 1; t++)
    {
        const int p = borderIndex(i0 + t - k.n / 2, len, border);
        if (p < 0)
            continue;
#pragma unroll
        for (int m = 0; m < M; m++)
        {
            if (m >= lines)
                break;
            const S* q = (const S*)((const unsigned char*)base + (size_t)p * stride + m * lineStep);
            S v[CN];
            for (int c = 0; c < CN; c++)
                v[c] = q[c];
#pragma unroll
            for (int i = 0; i < N; i++)
            {
                const int j = t - i;
                if (j >= 0 && j < k.n)
                    for (int c = 0; c < CN; c++)
                        acc[m][i][c] += (A)k.v[j] * v[c];
            }
        }
    }
}

// a thread filters PX pixels along each of PY rows
template<typename T, int CN, int PX, int PY>
__global__ void blurRows(const unsigned char* src, size_t sstep, unsigned char* buf, size_t bstep,
                         int rows, int cols, Taps<typename Blur<T>::K> k, int border)
{
    typedef typename Blur<T>::W W;
    const int x0 = (int)(blockIdx.x * blockDim.x + threadIdx.x) * PX;
    if (x0 >= cols)
        return;
    for (unsigned y0 = (blockIdx.y * blockDim.y + threadIdx.y) * PY; y0 < (unsigned)rows; y0 += gridDim.y * blockDim.y * PY)
    {
        const int lines = min(PY, rows - (int)y0);
        W acc[PY][PX][CN];
        convolve(k, x0, cols, border, (const T*)(src + (size_t)y0 * sstep), CN * sizeof(T), sstep, lines, acc);
#pragma unroll
        for (int m = 0; m < PY; m++)
        {
            if (m >= lines)
                break;
            W* out = (W*)(buf + (size_t)(y0 + m) * bstep) + x0 * CN;
#pragma unroll
            for (int i = 0; i < PX; i++)
                if (x0 + i < cols)
                    for (int c = 0; c < CN; c++)
                        out[i * CN + c] = acc[m][i][c];
        }
    }
}

// a thread filters PY pixels down each of PX columns
template<typename T, int CN, int PX, int PY>
__global__ void blurCols(const unsigned char* buf, size_t bstep, unsigned char* dst, size_t dstep,
                         int rows, int cols, Taps<typename Blur<T>::K> k, int border)
{
    typedef typename Blur<T>::W W;
    typedef typename Blur<T>::A A;
    const int x0 = (int)(blockIdx.x * blockDim.x + threadIdx.x) * PX;
    if (x0 >= cols)
        return;
    const int lines = min(PX, cols - x0);
    for (unsigned y0 = (blockIdx.y * blockDim.y + threadIdx.y) * PY; y0 < (unsigned)rows; y0 += gridDim.y * blockDim.y * PY)
    {
        A acc[PX][PY][CN];
        convolve(k, (int)y0, rows, border, (const W*)buf + x0 * CN, bstep, CN * sizeof(W), lines, acc);
#pragma unroll
        for (int i = 0; i < PY; i++)
        {
            if (y0 + i >= (unsigned)rows)
                break;
            T* out = (T*)(dst + (size_t)(y0 + i) * dstep) + x0 * CN;
#pragma unroll
            for (int m = 0; m < PX; m++)
                if (m < lines)
                    for (int c = 0; c < CN; c++)
                        out[m * CN + c] = blurResult<T>(acc[m][i][c]);
        }
    }
}

template<typename T, int CN> struct BlurRowsKernel
{ template<int PX, int PY> static const void* get() { return (const void*)blurRows<T, CN, PX, PY>; } };
template<typename T, int CN> struct BlurColsKernel
{ template<int PX, int PY> static const void* get() { return (const void*)blurCols<T, CN, PX, PY>; } };

template<typename T, int CN>
CV_GPU(Error_t) gaussianBlurN(const unsigned char* src, size_t sstep, unsigned char* buf, size_t bstep,
                              unsigned char* dst, size_t dstep, int rows, int cols,
                              const void* kx, int kxn, const void* ky, int kyn, int border,
                              const Launch& rowPass, const Launch& colPass, CV_GPU(Stream_t) s)
{
    typedef typename Blur<T>::K K;
    const void* rowKernel = patchKernel<BlurRowsKernel<T, CN> >(rowPass);
    const void* colKernel = patchKernel<BlurColsKernel<T, CN> >(colPass);
    if (!rowKernel || !colKernel)
        return CV_GPU(ErrorInvalidValue);
    Taps<K> tx, ty;
    tx.n = kxn;
    ty.n = kyn;
    memcpy(tx.v, kx, kxn * sizeof(K));
    memcpy(ty.v, ky, kyn * sizeof(K));
    void* rowArgs[] = { &src, &sstep, &buf, &bstep, &rows, &cols, &tx, &border };
    const CV_GPU(Error_t) r = CV_GPU(LaunchKernel)(rowKernel, grid(rows, cols, rowPass),
                                                   dim3(rowPass.blockX, rowPass.blockY), rowArgs, 0, s);
    if (r != CV_GPU(Success))
        return r;
    void* colArgs[] = { &buf, &bstep, &dst, &dstep, &rows, &cols, &ty, &border };
    return CV_GPU(LaunchKernel)(colKernel, grid(rows, cols, colPass), dim3(colPass.blockX, colPass.blockY),
                                colArgs, 0, s);
}

} // namespace

CV_GPU(Error_t) gaussianBlur(const unsigned char* src, size_t sstep, unsigned char* buf, size_t bstep,
                             unsigned char* dst, size_t dstep, int rows, int cols, int depth, int cn,
                             const void* kx, int kxn, const void* ky, int kyn, int border,
                             const Launch& rowPass, const Launch& colPass, CV_GPU(Stream_t) s)
{
    if (rows <= 0 || cols <= 0)
        return CV_GPU(Success);
    if (kxn < 1 || kxn > GAUSSIAN_MAX_TAPS || kyn < 1 || kyn > GAUSSIAN_MAX_TAPS)
        return CV_GPU(ErrorInvalidValue);
    switch (depth)
    {
    case CV_8U: CV_ACCEL_GPU_BY_CN(gaussianBlurN, unsigned char, src, sstep, buf, bstep, dst, dstep, rows, cols, kx, kxn, ky, kyn, border, rowPass, colPass, s)
    case CV_16U: CV_ACCEL_GPU_BY_CN(gaussianBlurN, unsigned short, src, sstep, buf, bstep, dst, dstep, rows, cols, kx, kxn, ky, kyn, border, rowPass, colPass, s)
    case CV_16S: CV_ACCEL_GPU_BY_CN(gaussianBlurN, short, src, sstep, buf, bstep, dst, dstep, rows, cols, kx, kxn, ky, kyn, border, rowPass, colPass, s)
    case CV_32F: CV_ACCEL_GPU_BY_CN(gaussianBlurN, float, src, sstep, buf, bstep, dst, dstep, rows, cols, kx, kxn, ky, kyn, border, rowPass, colPass, s)
    default: return CV_GPU(ErrorInvalidValue);
    }
}

}}}} // namespace cv::accel::CV_GPU_NS::kernels

#else // the operation

namespace cv { namespace accel { namespace CV_GPU_NS {

namespace {

//! a kernel GaussianBlur passes: CV_64F, centred, of an odd length the GPU takes
bool validKernel(const Mat& k)
{
    const int n = (int)k.total();
    return k.type() == CV_64F && k.isContinuous() && n % 2 == 1 && n <= kernels::GAUSSIAN_MAX_TAPS;
}

/* The CPU's fixed-point taps (getGaussianKernelFixedPoint_ED() in smooth.dispatch.cpp): rounded with
   the error carried from tap to tap, the centre taking the rest. Scaling by 2^bits is exact, so the
   double arithmetic gives the CPU's softdouble results. */
template<typename T>
void fixedPointTaps(const double* k, int n, int bits, T* taps)
{
    const int64 one = (int64)1 << bits;
    double err = 0;
    int64 sum = 0;
    for (int i = 0; i < n / 2; i++)
    {
        const double v = k[i] * (double)one + err;
        const int64 r = cvRound(v);
        err = v - (double)r;
        taps[i] = taps[n - 1 - i] = (T)r;
        sum += r;
    }
    taps[n / 2] = (T)(one - 2 * sum);
}

//! the taps the CPU uses for @p depth: CV_16U with 8 fraction bits for CV_8U, CV_32U with 16 for CV_16U, else CV_32F
Mat cpuTaps(const Mat& k, int depth)
{
    const int n = (int)k.total();
    Mat taps;
    if (depth == CV_8U)
        fixedPointTaps(k.ptr<double>(), n, 8, (ushort*)(taps = Mat(1, n, CV_16U)).ptr());
    else if (depth == CV_16U)
        fixedPointTaps(k.ptr<double>(), n, 16, (unsigned*)(taps = Mat(1, n, CV_32U)).ptr());
    else
        k.reshape(1, 1).convertTo(taps, CV_32F);
    return taps;
}

//! gaussianBlur, one of the ImgprocOps the module mixes together (precomp.hpp)
template<typename Base>
struct GpuSmoothOps : Base
{
    bool gaussianBlur(const UMat& src, UMat& dst, const Mat& kx64, const Mat& ky64, int borderType,
                      Stream& stream) CV_OVERRIDE;
};

// two passes through a buffer, so the output may alias the input in any way; BORDER_WRAP is left to
// the CPU, whose floating-point filters reject it, and so is a region without BORDER_ISOLATED, which
// reads its parent beyond its edges
template<typename Base>
bool GpuSmoothOps<Base>::gaussianBlur(const UMat& src, UMat& dst, const Mat& kx64, const Mat& ky64, int borderType,
                                      Stream& stream)
{
    if (src.isSubmatrix() && !(borderType & BORDER_ISOLATED))
        return false;
    borderType &= ~BORDER_ISOLATED;
    const int depth = src.depth();
    const int tapType = depth == CV_8U ? CV_16U : depth == CV_16U ? CV_32U : CV_32F;
    if (!usableImage(src) || !usableImage(dst) || borderType == BORDER_WRAP || borderType > BORDER_REFLECT_101 ||
        (depth != CV_8U && depth != CV_16U && depth != CV_16S && depth != CV_32F) ||
        !validKernel(kx64) || !validKernel(ky64))
        return false;
    const Mat kx = cpuTaps(kx64, depth), ky = cpuTaps(ky64, depth);
    GpuStream& s = gpuStreamOf(stream);
    UMat buf;
    buf.createOn(getDevice(src), src.size(), CV_MAKETYPE(tapType, src.channels()));
    DeviceGuard g(s.device_);
    const unsigned char* sp = (const unsigned char*)src.handle(ACCESS_READ) + src.offset;
    unsigned char* bp = (unsigned char*)buf.handle(ACCESS_WRITE);
    unsigned char* dp = (unsigned char*)dst.handle(ACCESS_WRITE) + dst.offset;
    if (!launched(kernels::gaussianBlur(sp, src.step[0], bp, buf.step[0], dp, dst.step[0], src.rows, src.cols,
                                        depth, src.channels(), kx.ptr(), (int)kx.total(), ky.ptr(), (int)ky.total(),
                                        borderType, tunedLaunch("blurRows", src.type()),
                                        tunedLaunch("blurCols", src.type()), s.s_), "GaussianBlur"))
        return false;
    stream.retain({ src, buf, dst });
    return true;
}

} // namespace

}}} // namespace cv::accel::CV_GPU_NS

#endif // CV_ACCEL_GPU_KERNELS
