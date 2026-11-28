// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL, GPU: warpAffine with nearest and linear interpolation, in the CPU's float arithmetic.
// Compiled twice through ../../precomp.hpp: the kernels with CV_ACCEL_GPU_KERNELS, the operation that
// launches them without.

namespace cv { namespace accel { namespace CV_GPU_NS { namespace kernels {

//! cv::InterpolationFlags, which core does not see
enum { INTER_NEAREST = 0, INTER_LINEAR = 1 };

/** @brief warpAffine with the inverse map @p M (2x3), INTER_NEAREST or INTER_LINEAR, @p bval the
BORDER_CONSTANT pixel already in the range of @p depth, launched as @p l. */
CV_GPU(Error_t) warpAffine(const unsigned char* src, size_t sstep, int srows, int scols,
                           unsigned char* dst, size_t dstep, int drows, int dcols, int depth, int cn,
                           const float* M, int interpolation, int border, const float* bval,
                           const Launch& l, CV_GPU(Stream_t) s);

}}}} // namespace cv::accel::CV_GPU_NS::kernels

#ifdef CV_ACCEL_GPU_KERNELS

namespace cv { namespace accel { namespace CV_GPU_NS { namespace kernels {

namespace {

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

// a thread computes PX x PY output pixels
template<typename T, int CN, int PX, int PY>
__global__ void warpAffineKernel(const unsigned char* src, size_t sstep, int srows, int scols,
                                 unsigned char* dst, size_t dstep, int drows, int dcols, WarpParams p)
{
    const int x0 = (int)(blockIdx.x * blockDim.x + threadIdx.x) * PX;
    if (x0 >= dcols)
        return;
    for (unsigned y0 = (blockIdx.y * blockDim.y + threadIdx.y) * PY; y0 < (unsigned)drows; y0 += gridDim.y * blockDim.y * PY)
    {
#pragma unroll
        for (int m = 0; m < PY; m++)
        {
            const unsigned y = y0 + m;
            if (y >= (unsigned)drows)
                break;
            T* out = (T*)(dst + (size_t)y * dstep) + x0 * CN;
#pragma unroll
            for (int i = 0; i < PX; i++)
            {
                const int x = x0 + i;
                if (x >= dcols)
                    break;
                // the CPU's fma(M0, x, y*M1 + M2) (warp_kernels.simd.hpp)
                const float sx = fmaf(p.M[0], (float)x, mulThenAdd(p.M[1], (float)y, p.M[2]));
                const float sy = fmaf(p.M[3], (float)x, mulThenAdd(p.M[4], (float)y, p.M[5]));
                float v[CN];
                if (p.interpolation == INTER_NEAREST)
                    readPixel<T, CN>(src, sstep, srows, scols, __float2int_rn(sx), __float2int_rn(sy), p, v);
                else
                    sampleLinear<T, CN>(src, sstep, srows, scols, sx, sy, p, v);
                for (int c = 0; c < CN; c++)
                    out[i * CN + c] = saturate<T>(v[c]);
            }
        }
    }
}

template<typename T, int CN> struct WarpAffineKernel
{ template<int PX, int PY> static const void* get() { return (const void*)warpAffineKernel<T, CN, PX, PY>; } };

template<typename T, int CN>
CV_GPU(Error_t) warpAffineN(const unsigned char* src, size_t sstep, int srows, int scols,
                            unsigned char* dst, size_t dstep, int drows, int dcols, WarpParams p,
                            const Launch& l, CV_GPU(Stream_t) s)
{
    const void* k = patchKernel<WarpAffineKernel<T, CN> >(l);
    if (!k)
        return CV_GPU(ErrorInvalidValue);
    void* args[] = { &src, &sstep, &srows, &scols, &dst, &dstep, &drows, &dcols, &p };
    return CV_GPU(LaunchKernel)(k, grid(drows, dcols, l), dim3(l.blockX, l.blockY), args, 0, s);
}

} // namespace

CV_GPU(Error_t) warpAffine(const unsigned char* src, size_t sstep, int srows, int scols,
                           unsigned char* dst, size_t dstep, int drows, int dcols, int depth, int cn,
                           const float* M, int interpolation, int border, const float* bval,
                           const Launch& l, CV_GPU(Stream_t) s)
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
    case CV_8U: CV_ACCEL_GPU_BY_CN(warpAffineN, unsigned char, src, sstep, srows, scols, dst, dstep, drows, dcols, p, l, s)
    case CV_16U: CV_ACCEL_GPU_BY_CN(warpAffineN, unsigned short, src, sstep, srows, scols, dst, dstep, drows, dcols, p, l, s)
    case CV_32F: CV_ACCEL_GPU_BY_CN(warpAffineN, float, src, sstep, srows, scols, dst, dstep, drows, dcols, p, l, s)
    default: return CV_GPU(ErrorInvalidValue);
    }
}

}}}} // namespace cv::accel::CV_GPU_NS::kernels

#else // the operation

namespace cv { namespace accel { namespace CV_GPU_NS {

namespace {

// cv::InterpolationFlags and cv::WarpPolynomial values that core does not see
enum { INTER_AREA = 3, INTER_MAX = 7, WARP_INVERSE_MAP = 16 };

//! warpAffine, one of the ImgprocOps the module mixes together (precomp.hpp)
template<typename Base>
struct GpuWarpOps : Base
{
    bool warpAffine(const UMat& src, UMat& dst, InputArray M0, int flags, int borderType,
                    const Scalar& borderValue, Stream& stream) CV_OVERRIDE;
};

// the types the CPU warps in floating point (warp_kernels.simd.hpp), which the GPU follows, with M
// inverted as cv::warpAffine inverts it; an output that is the input gets a copy of it first
template<typename Base>
bool GpuWarpOps<Base>::warpAffine(const UMat& src, UMat& dst, InputArray M0, int flags, int borderType,
                                  const Scalar& borderValue, Stream& stream)
{
    const Mat m0 = M0.getMat();
    int interpolation = flags & INTER_MAX;
    if (interpolation == INTER_AREA)
        interpolation = kernels::INTER_LINEAR;
    const int depth = src.depth(), cn = src.channels();
    if (m0.size() != Size(3, 2) || (m0.type() != CV_32F && m0.type() != CV_64F) ||   // the CPU reports it
        !usableImage(src) || !usableImage(dst) || cn == 2 ||
        (depth != CV_8U && depth != CV_16U && depth != CV_32F) ||
        (interpolation != kernels::INTER_NEAREST && interpolation != kernels::INTER_LINEAR) || borderType > BORDER_REFLECT_101)
        return false;
    double M[6];
    Mat matM(2, 3, CV_64F, M);
    m0.convertTo(matM, CV_64F);
    if (!(flags & WARP_INVERSE_MAP))
    {
        double D = M[0]*M[4] - M[1]*M[3];
        D = D != 0 ? 1./D : 0;
        double A11 = M[4]*D, A22 = M[0]*D;
        M[0] = A11; M[1] *= -D;
        M[3] *= -D; M[4] = A22;
        double b1 = -M[0]*M[2] - M[1]*M[5];
        double b2 = -M[3]*M[2] - M[4]*M[5];
        M[2] = b1; M[5] = b2;
    }
    float m[6], bval[4];
    for (int i = 0; i < 6; i++)
        m[i] = (float)M[i];
    for (int c = 0; c < 4; c++)
        bval[c] = depth == CV_8U ? saturate_cast<uchar>(borderValue[c])
                : depth == CV_16U ? saturate_cast<ushort>(borderValue[c]) : (float)borderValue[c];
    const UMat in = src.u == dst.u ? src.clone() : src;
    GpuStream& s = gpuStreamOf(stream);
    DeviceGuard g(s.device_);
    const unsigned char* sp = (const unsigned char*)in.handle(ACCESS_READ) + in.offset;
    unsigned char* dp = (unsigned char*)dst.handle(ACCESS_WRITE) + dst.offset;
    if (!launched(kernels::warpAffine(sp, in.step[0], in.rows, in.cols, dp, dst.step[0], dst.rows, dst.cols,
                                      depth, cn, m, interpolation, borderType, bval,
                                      tunedLaunch("warpAffine", src.type()), s.s_), "warpAffine"))
        return false;
    stream.retain({ in, dst });
    return true;
}

} // namespace

}}} // namespace cv::accel::CV_GPU_NS

#endif // CV_ACCEL_GPU_KERNELS
