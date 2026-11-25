// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: kernels of the GPU backends (core and imgproc operations), compiled once per
// runtime, for CUDA by src/cuda/accel_kernels.cu and for HIP by src/hip/accel_kernels.hip. flip
// and transpose move elements as opaque groups of bytes, so one kernel serves every type of the
// same element size.

#include "gpu_kernels.hpp"
#include "opencv2/core/hal/interface.h"
#include <string.h>

namespace cv { namespace accel { namespace CV_GPU_NS { namespace kernels {

namespace {

//! the largest power of two dividing @p n, up to 16
constexpr int elemAlign(int n)
{
    return n % 16 == 0 ? 16 : n % 8 == 0 ? 8 : n % 4 == 0 ? 4 : n % 2 == 0 ? 2 : 1;
}
// one element of N bytes, aligned as far as N allows, so a copy is one wide load and store
template<int N> struct alignas(elemAlign(N)) Elem { unsigned char b[N]; };
// the same bytes at any alignment, for rows whose start or step is not a multiple of that
template<int N> struct Bytes { unsigned char b[N]; };

const int BLOCK_X = 32, BLOCK_Y = 8;
const int TILE = 32;

// gridDim.y is at most 65535: taller images are covered by a loop over y
const unsigned MAX_GRID_Y = 65535;

inline bool aligned(const void* src, size_t sstep, const void* dst, size_t dstep, size_t a)
{
    return (((size_t)src | sstep | (size_t)dst | dstep) & (a - 1)) == 0;
}

template<typename T>
__global__ void flipKernel(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                           int rows, int cols, bool flipRows, bool flipCols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= cols)
        return;
    const int sx = flipCols ? cols - 1 - x : x;
    for (unsigned y = blockIdx.y * blockDim.y + threadIdx.y; y < (unsigned)rows; y += gridDim.y * blockDim.y)
    {
        const unsigned sy = flipRows ? (unsigned)rows - 1 - y : y;
        ((T*)(dst + (size_t)y * dstep))[x] = ((const T*)(src + (size_t)sy * sstep))[sx];
    }
}

// each mirrored pair is swapped once, by the thread of its first element
template<typename T>
__global__ void flipInplaceKernel(unsigned char* data, size_t step, int rows, int cols, bool flipRows, bool flipCols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= cols)
        return;
    const int mx = flipCols ? cols - 1 - x : x;
    for (unsigned y = blockIdx.y * blockDim.y + threadIdx.y; y < (unsigned)rows; y += gridDim.y * blockDim.y)
    {
        const unsigned my = flipRows ? (unsigned)rows - 1 - y : y;
        if (y > my || (y == my && x >= mx))
            continue;
        T* a = (T*)(data + (size_t)y * step) + x;
        T* b = (T*)(data + (size_t)my * step) + mx;
        const T t = *a;
        *a = *b;
        *b = t;
    }
}

// a TILE x TILE block goes through shared memory, so both the reads and the writes walk along rows
template<typename T>
__global__ void transposeKernel(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                                int rows, int cols)
{
    __shared__ T tile[TILE][TILE + 1];
    const int x0 = blockIdx.x * TILE;
    for (int y0 = blockIdx.y * TILE; y0 < rows; y0 += gridDim.y * TILE)
    {
        const int x = x0 + threadIdx.x;
        for (int j = threadIdx.y; j < TILE; j += BLOCK_Y)
            if (x < cols && y0 + j < rows)
                tile[j][threadIdx.x] = ((const T*)(src + (size_t)(y0 + j) * sstep))[x];
        __syncthreads();
        const int ox = y0 + threadIdx.x;
        for (int j = threadIdx.y; j < TILE; j += BLOCK_Y)
            if (ox < rows && x0 + j < cols)
                ((T*)(dst + (size_t)(x0 + j) * dstep))[ox] = tile[threadIdx.x][j];
        __syncthreads();
    }
}

// each element above the diagonal is swapped with its counterpart below it
template<typename T>
__global__ void transposeInplaceKernel(unsigned char* data, size_t step, int n)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n)
        return;
    for (unsigned y = blockIdx.y * blockDim.y + threadIdx.y; y < (unsigned)x; y += gridDim.y * blockDim.y)
    {
        T* a = (T*)(data + (size_t)y * step) + x;
        T* b = (T*)(data + (size_t)x * step) + y;
        const T t = *a;
        *a = *b;
        *b = t;
    }
}

static inline dim3 grid(int rows, int cols, int bx, int by)
{
    const unsigned gy = (unsigned)((rows + by - 1) / by);
    return dim3((unsigned)((cols + bx - 1) / bx), gy < MAX_GRID_Y ? gy : MAX_GRID_Y);
}

// LaunchKernel returns the status of this launch, where GetLastError() after <<<>>> would
// return the oldest error left on the thread by any earlier call

template<int N>
CV_GPU(Error_t) flipN(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                      int rows, int cols, int mode, CV_GPU(Stream_t) s)
{
    // OpenCV's codes: 0 around the x axis (rows), > 0 around the y axis (columns), < 0 both
    bool flipRows = mode <= 0, flipCols = mode != 0;
    void* args[] = { &src, &sstep, &dst, &dstep, &rows, &cols, &flipRows, &flipCols };
    const bool wide = aligned(src, sstep, dst, dstep, alignof(Elem<N>));
    const void* k = wide ? (const void*)flipKernel<Elem<N> > : (const void*)flipKernel<Bytes<N> >;
    return CV_GPU(LaunchKernel)(k, grid(rows, cols, BLOCK_X, BLOCK_Y), dim3(BLOCK_X, BLOCK_Y),
                                args, 0, s);
}

template<int N>
CV_GPU(Error_t) flipInplaceN(unsigned char* data, size_t step, int rows, int cols, int mode, CV_GPU(Stream_t) s)
{
    bool flipRows = mode <= 0, flipCols = mode != 0;
    void* args[] = { &data, &step, &rows, &cols, &flipRows, &flipCols };
    const bool wide = aligned(data, step, data, step, alignof(Elem<N>));
    const void* k = wide ? (const void*)flipInplaceKernel<Elem<N> > : (const void*)flipInplaceKernel<Bytes<N> >;
    return CV_GPU(LaunchKernel)(k, grid(rows, cols, BLOCK_X, BLOCK_Y), dim3(BLOCK_X, BLOCK_Y),
                                args, 0, s);
}

template<int N>
CV_GPU(Error_t) transposeN(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                           int rows, int cols, CV_GPU(Stream_t) s)
{
    void* args[] = { &src, &sstep, &dst, &dstep, &rows, &cols };
    const bool wide = aligned(src, sstep, dst, dstep, alignof(Elem<N>));
    const void* k = wide ? (const void*)transposeKernel<Elem<N> >
                         : (const void*)transposeKernel<Bytes<N> >;
    return CV_GPU(LaunchKernel)(k, grid(rows, cols, TILE, TILE), dim3(TILE, BLOCK_Y), args, 0, s);
}

template<int N>
CV_GPU(Error_t) transposeInplaceN(unsigned char* data, size_t step, int n, CV_GPU(Stream_t) s)
{
    void* args[] = { &data, &step, &n };
    const bool wide = aligned(data, step, data, step, alignof(Elem<N>));
    const void* k = wide ? (const void*)transposeInplaceKernel<Elem<N> >
                         : (const void*)transposeInplaceKernel<Bytes<N> >;
    return CV_GPU(LaunchKernel)(k, grid(n, n, BLOCK_X, BLOCK_Y), dim3(BLOCK_X, BLOCK_Y), args, 0, s);
}

} // namespace

bool supportedElementSize(int esz)
{
    switch (esz)
    {
    case 1: case 2: case 3: case 4: case 6: case 8: case 12: case 16: case 24: case 32:
        return true;
    default:
        return false;
    }
}

#define CV_ACCEL_GPU_BY_SIZE(fn, ...) \
    switch (esz) \
    { \
    case 1: return fn<1>(__VA_ARGS__); \
    case 2: return fn<2>(__VA_ARGS__); \
    case 3: return fn<3>(__VA_ARGS__); \
    case 4: return fn<4>(__VA_ARGS__); \
    case 6: return fn<6>(__VA_ARGS__); \
    case 8: return fn<8>(__VA_ARGS__); \
    case 12: return fn<12>(__VA_ARGS__); \
    case 16: return fn<16>(__VA_ARGS__); \
    case 24: return fn<24>(__VA_ARGS__); \
    case 32: return fn<32>(__VA_ARGS__); \
    default: return CV_GPU(ErrorInvalidValue); \
    }

CV_GPU(Error_t) flip(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                     int rows, int cols, int esz, int mode, CV_GPU(Stream_t) s)
{
    if (rows <= 0 || cols <= 0)
        return CV_GPU(Success);
    CV_ACCEL_GPU_BY_SIZE(flipN, src, sstep, dst, dstep, rows, cols, mode, s)
}

CV_GPU(Error_t) flipInplace(unsigned char* data, size_t step, int rows, int cols, int esz, int mode,
                            CV_GPU(Stream_t) s)
{
    if (rows <= 0 || cols <= 0)
        return CV_GPU(Success);
    CV_ACCEL_GPU_BY_SIZE(flipInplaceN, data, step, rows, cols, mode, s)
}

CV_GPU(Error_t) transpose(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                          int rows, int cols, int esz, CV_GPU(Stream_t) s)
{
    if (rows <= 0 || cols <= 0)
        return CV_GPU(Success);
    CV_ACCEL_GPU_BY_SIZE(transposeN, src, sstep, dst, dstep, rows, cols, s)
}

CV_GPU(Error_t) transposeInplace(unsigned char* data, size_t step, int n, int esz, CV_GPU(Stream_t) s)
{
    if (n <= 0)
        return CV_GPU(Success);
    CV_ACCEL_GPU_BY_SIZE(transposeInplaceN, data, step, n, s)
}

#undef CV_ACCEL_GPU_BY_SIZE

// imgproc: the kernels are built from the device functions below, which future ones reuse

namespace {

// cv::BorderTypes; this file does not include core's C++ headers
enum { BORDER_CONSTANT = 0, BORDER_REPLICATE = 1, BORDER_REFLECT = 2, BORDER_WRAP = 3, BORDER_REFLECT_101 = 4 };

//! cv::borderInterpolate in O(1): the index position @p p of a line of @p len reads, -1 for BORDER_CONSTANT
__device__ inline int borderIndex(int p, int len, int border)
{
    if ((unsigned)p < (unsigned)len)
        return p;
    if (border == BORDER_REPLICATE)
        return p < 0 ? 0 : len - 1;
    if (border == BORDER_WRAP)
    {
        const long long r = (long long)p % len;
        return (int)(r < 0 ? r + len : r);
    }
    if (border == BORDER_REFLECT || border == BORDER_REFLECT_101)
    {
        if (len == 1)
            return 0;
        const int delta = border == BORDER_REFLECT_101;
        const long long period = 2 * ((long long)len - delta);
        long long r = (long long)p % period;
        if (r < 0)
            r += period;
        return (int)(r < len ? r : period - 1 + delta - r);
    }
    return -1;
}

//! saturate_cast<T>(v): rounded half to even as cvRound, clamped to the range of T
template<typename T> __device__ inline T saturate(float v);
template<> __device__ inline float saturate<float>(float v) { return v; }
template<> __device__ inline unsigned char saturate<unsigned char>(float v)
{ return (unsigned char)fminf(fmaxf(rintf(v), 0.f), 255.f); }
template<> __device__ inline unsigned short saturate<unsigned short>(float v)
{ return (unsigned short)fminf(fmaxf(rintf(v), 0.f), 65535.f); }
template<> __device__ inline short saturate<short>(float v)
{ return (short)fminf(fmaxf(rintf(v), -32768.f), 32767.f); }

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

//! a * b + c rounded twice, never contracted into an fma: HIP's __fmul_rn and __fadd_rn are plain operators
__device__ inline float mulThenAdd(float a, float b, float c)
{
#ifdef __clang__
#pragma clang fp contract(off)
    return a * b + c;
#else
    return __fadd_rn(__fmul_rn(a, b), c);
#endif
}

struct WarpParams
{
    float M[6];         // output (x, y) -> input
    float bval[4];      // the BORDER_CONSTANT pixel
    int interpolation, border;
};

//! input pixel (@p x, @p y) as floats; outside the image the border's pixel
template<typename T, int CN>
__device__ inline void readPixel(const unsigned char* src, size_t step, int rows, int cols, int x, int y,
                                 const WarpParams& p, float* v)
{
    x = borderIndex(x, cols, p.border);
    y = borderIndex(y, rows, p.border);
    const T* q = (const T*)(src + (size_t)(y < 0 ? 0 : y) * step) + (x < 0 ? 0 : x) * CN;
    for (int c = 0; c < CN; c++)
        v[c] = x < 0 || y < 0 ? p.bval[c] : (float)q[c];
}

//! bilinear interpolation at (@p sx, @p sy), with the CPU's operations (warp_common.vector.hpp)
template<typename T, int CN>
__device__ inline void sampleLinear(const unsigned char* src, size_t step, int rows, int cols, float sx, float sy,
                                    const WarpParams& p, float* v)
{
    const int ix = __float2int_rd(sx), iy = __float2int_rd(sy);
    const float fx = sx - (float)ix, fy = sy - (float)iy;
    float p00[CN], p01[CN], p10[CN], p11[CN];
    readPixel<T, CN>(src, step, rows, cols, ix, iy, p, p00);
    readPixel<T, CN>(src, step, rows, cols, ix + 1, iy, p, p01);
    readPixel<T, CN>(src, step, rows, cols, ix, iy + 1, p, p10);
    readPixel<T, CN>(src, step, rows, cols, ix + 1, iy + 1, p, p11);
    for (int c = 0; c < CN; c++)
    {
        const float v0 = fmaf(fx, p01[c] - p00[c], p00[c]);
        const float v1 = fmaf(fx, p11[c] - p10[c], p10[c]);
        v[c] = fmaf(fy, v1 - v0, v0);
    }
}

template<typename T, int CN>
__global__ void warpAffineKernel(const unsigned char* src, size_t sstep, int srows, int scols,
                                 unsigned char* dst, size_t dstep, int drows, int dcols, WarpParams p)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= dcols)
        return;
    for (unsigned y = blockIdx.y * blockDim.y + threadIdx.y; y < (unsigned)drows; y += gridDim.y * blockDim.y)
    {
        // the CPU's fma(M0, x, y*M1 + M2) (warp_kernels.simd.hpp)
        const float sx = fmaf(p.M[0], (float)x, mulThenAdd(p.M[1], (float)y, p.M[2]));
        const float sy = fmaf(p.M[3], (float)x, mulThenAdd(p.M[4], (float)y, p.M[5]));
        float v[CN];
        if (p.interpolation == INTER_NEAREST)
            readPixel<T, CN>(src, sstep, srows, scols, __float2int_rn(sx), __float2int_rn(sy), p, v);
        else
            sampleLinear<T, CN>(src, sstep, srows, scols, sx, sy, p, v);
        T* out = (T*)(dst + (size_t)y * dstep) + x * CN;
        for (int c = 0; c < CN; c++)
            out[c] = saturate<T>(v[c]);
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

template<typename T, int CN>
CV_GPU(Error_t) warpAffineN(const unsigned char* src, size_t sstep, int srows, int scols,
                            unsigned char* dst, size_t dstep, int drows, int dcols, WarpParams p,
                            CV_GPU(Stream_t) s)
{
    void* args[] = { &src, &sstep, &srows, &scols, &dst, &dstep, &drows, &dcols, &p };
    return CV_GPU(LaunchKernel)((const void*)warpAffineKernel<T, CN>, grid(drows, dcols, BLOCK_X, BLOCK_Y),
                                dim3(BLOCK_X, BLOCK_Y), args, 0, s);
}

} // namespace

#define CV_ACCEL_GPU_BY_CN(fn, T, ...) \
    switch (cn) \
    { \
    case 1: return fn<T, 1>(__VA_ARGS__); \
    case 2: return fn<T, 2>(__VA_ARGS__); \
    case 3: return fn<T, 3>(__VA_ARGS__); \
    case 4: return fn<T, 4>(__VA_ARGS__); \
    default: return CV_GPU(ErrorInvalidValue); \
    }

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

CV_GPU(Error_t) warpAffine(const unsigned char* src, size_t sstep, int srows, int scols,
                           unsigned char* dst, size_t dstep, int drows, int dcols, int depth, int cn,
                           const float* M, int interpolation, int border, const float* bval,
                           CV_GPU(Stream_t) s)
{
    if (drows <= 0 || dcols <= 0)
        return CV_GPU(Success);
    WarpParams p;
    memcpy(p.M, M, sizeof(p.M));
    memcpy(p.bval, bval, sizeof(p.bval));
    p.interpolation = interpolation;
    p.border = border;
    switch (depth)
    {
    case CV_8U: CV_ACCEL_GPU_BY_CN(warpAffineN, unsigned char, src, sstep, srows, scols, dst, dstep, drows, dcols, p, s)
    case CV_16U: CV_ACCEL_GPU_BY_CN(warpAffineN, unsigned short, src, sstep, srows, scols, dst, dstep, drows, dcols, p, s)
    case CV_32F: CV_ACCEL_GPU_BY_CN(warpAffineN, float, src, sstep, srows, scols, dst, dstep, drows, dcols, p, s)
    default: return CV_GPU(ErrorInvalidValue);
    }
}

#undef CV_ACCEL_GPU_BY_CN

}}}} // namespace cv::accel::CV_GPU_NS::kernels
