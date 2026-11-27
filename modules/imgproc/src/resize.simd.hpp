/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
// Copyright (C) 2000-2008, 2017, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
//M*/

#include "precomp.hpp"
#include "resize.hpp"
#include "hal_replacement.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/utils/buffer_area.private.hpp"

namespace cv {

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void resize_cpu(int src_type,
            const uchar * src_data, size_t src_step, int src_width, int src_height,
            uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
            double inv_scale_x, double inv_scale_y, int interpolation);

void resizeNN_( const Mat& src, Mat& dst, double fx, double fy );

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

const int INTER_RESIZE_COEF_BITS=11;
const int INTER_RESIZE_COEF_SCALE=1 << INTER_RESIZE_COEF_BITS;

// Nearest-neighbor row kernels. x_ofs[] holds non-decreasing byte offsets into the source row.

// Pixels below the returned end have x_ofs[x] + pad inside the row.
static inline int resizeNN_vec_end(const int* x_ofs, int width, int row_bytes, int pad)
{
    int end = width;
    while (end > 0 && x_ofs[end - 1] + pad >= row_bytes)
        end--;
    return end;
}

#if CV_AVX_512VBMI
// When the 64 byte offsets span at most 256 bytes, one 256-byte source window permuted with
// vpermb (v512_lut with byte indices) replaces the per-index LUT. The window is slid back
// to end at the row end when needed, so row_bytes must be >= 256.
static inline bool resizeNN_lut_u8_window(const uchar* S, const int* x_ofs, int row_bytes, v_uint8x64& out)
{
    const v_int32x16 o0 = v512_load(x_ofs), o1 = v512_load(x_ofs + 16);
    const v_int32x16 o2 = v512_load(x_ofs + 32), o3 = v512_load(x_ofs + 48);
    const int min_ofs = v_reduce_min(v_min(v_min(o0, o1), v_min(o2, o3)));
    const int max_ofs = v_reduce_max(v_max(v_max(o0, o1), v_max(o2, o3)));
    if (max_ofs - min_ofs > 255)
        return false;
    const int base = std::min(min_ofs, row_bytes - 256);
    const v_int32x16 vb = v512_setall_s32(base);
    const v_uint8x64 idx = v_pack_u(v_pack(v_sub(o0, vb), v_sub(o1, vb)),
                                    v_pack(v_sub(o2, vb), v_sub(o3, vb)));
    out = v512_lut(S + base, idx);
    return true;
}
#endif

// Some backends build the 8-bit and 16-bit LUTs from 32-bit loads, so vend must keep
// x_ofs[x] + 3 inside the row.
static CV_ALWAYS_INLINE void resizeNN_row_u8(uchar* D, const uchar* S, const int* x_ofs, int width,
                                             int row_bytes, int vend)
{
    int x = 0;
#if CV_AVX_512VBMI
    const int n = VTraits<v_uint8x64>::vlanes();
    // Short rows are copied into a 256-byte buffer so the window always fits.
    CV_DECL_ALIGNED(64) uchar sbuf[256];
    if (row_bytes < 256 && width >= n)
    {
        memcpy(sbuf, S, row_bytes);
        S = sbuf;
        row_bytes = 256;
        vend = width;
    }
    for (; x <= width - n; x += n)
    {
        v_uint8x64 r;
        if (resizeNN_lut_u8_window(S, x_ofs + x, row_bytes, r))
            v_store(D + x, r);
        else if (x + n <= vend)
            v_store(D + x, v512_lut(S, x_ofs + x));
        else
            break;
    }
#elif (CV_SIMD || CV_SIMD_SCALABLE)
    CV_UNUSED(row_bytes);
    const int n = VTraits<v_uint8>::vlanes();
    for (; x <= vend - n; x += n)
        v_store(D + x, vx_lut(S, x_ofs + x));
#else
    CV_UNUSED(row_bytes); CV_UNUSED(vend);
#endif
    for (; x < width; x++)
        D[x] = S[x_ofs[x]];
}

static CV_ALWAYS_INLINE void resizeNN_row_u16(uchar* D, const uchar* S, const int* x_ofs, int width, int vend)
{
    ushort* D16 = (ushort*)D;
    int x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int n = VTraits<v_uint16>::vlanes();
    for (; x <= vend - n; x += n)
        v_store(D16 + x, v_reinterpret_as_u16(vx_lut_pairs(S, x_ofs + x)));
#else
    CV_UNUSED(vend);
#endif
    for (; x < width; x++)
        D16[x] = *(const ushort*)(S + x_ofs[x]);
}

static CV_ALWAYS_INLINE void resizeNN_row_px3(uchar* D, const uchar* S, const int* x_ofs, int width)
{
    int x = 0;
    for (; x <= width - 4; x += 4, D += 12)
    {
        const uchar* s0 = S + x_ofs[x];
        const uchar* s1 = S + x_ofs[x + 1];
        const uchar* s2 = S + x_ofs[x + 2];
        const uchar* s3 = S + x_ofs[x + 3];
        D[0] = s0[0]; D[1] = s0[1]; D[2] = s0[2];
        D[3] = s1[0]; D[4] = s1[1]; D[5] = s1[2];
        D[6] = s2[0]; D[7] = s2[1]; D[8] = s2[2];
        D[9] = s3[0]; D[10] = s3[1]; D[11] = s3[2];
    }
    for (; x < width; x++, D += 3)
    {
        const uchar* s0 = S + x_ofs[x];
        D[0] = s0[0]; D[1] = s0[1]; D[2] = s0[2];
    }
}

static CV_ALWAYS_INLINE void resizeNN_row_u32(uchar* D, const uchar* S, const int* x_ofs, int width)
{
    uint32_t* D32 = (uint32_t*)D;
    int x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int n = VTraits<v_uint32>::vlanes();
    int CV_DECL_ALIGNED(CV_SIMD_WIDTH) idx[VTraits<v_uint32>::max_nlanes];
    for (; x <= width - n; x += n)
    {
        v_store_aligned(idx, v_shr<2>(vx_load(x_ofs + x)));
        v_store(D32 + x, vx_lut((const uint32_t*)S, idx));
    }
#endif
    for (; x < width; x++)
        D32[x] = *(const uint32_t*)(S + x_ofs[x]);
}

static CV_ALWAYS_INLINE void resizeNN_row_u64(uchar* D, const uchar* S, const int* x_ofs, int width)
{
    uint64_t* D64 = (uint64_t*)D;
    int x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int n = VTraits<v_uint64>::vlanes();
    int CV_DECL_ALIGNED(CV_SIMD_WIDTH) idx[VTraits<v_uint64>::max_nlanes];
    for (; x <= width - n; x += n)
    {
        for (int i = 0; i < n; i++)
            idx[i] = x_ofs[x + i] >> 3;
        v_store(D64 + x, vx_lut((const uint64_t*)S, idx));
    }
#endif
    for (; x < width; x++)
        D64[x] = *(const uint64_t*)(S + x_ofs[x]);
}

// Pixels below vend are copied as 16 bytes: the extra 4 bytes are overwritten by the next
// pixel, so vend must exclude the last pixel and any x_ofs[x] + 16 > row_bytes.
static inline int resizeNN_vec_end_px12(const int* x_ofs, int width, int row_bytes)
{
#if CV_SIMD128 && !CV_SIMD_SCALABLE
    return std::max(std::min(resizeNN_vec_end(x_ofs, width, row_bytes, 15), width - 1), 0);
#else
    CV_UNUSED(x_ofs); CV_UNUSED(width); CV_UNUSED(row_bytes);
    return 0;
#endif
}

static inline int resizeNN_vec_end_for(const int* x_ofs, int width, int row_bytes, int pix_size)
{
    if (pix_size == 1 || pix_size == 2)
        return resizeNN_vec_end(x_ofs, width, row_bytes, 3);
    if (pix_size == 12)
        return resizeNN_vec_end_px12(x_ofs, width, row_bytes);
    return 0;
}

static CV_ALWAYS_INLINE void resizeNN_row_px12(uchar* D, const uchar* S, const int* x_ofs,
                                               int width, int vend)
{
    int x = 0;
#if CV_SIMD128 && !CV_SIMD_SCALABLE
    for (; x < vend; x++, D += 12)
        v_store(D, v_load(S + x_ofs[x]));
#else
    CV_UNUSED(vend);
#endif
    for (; x < width; x++, D += 12)
    {
        const int* _tS = (const int*)(S + x_ofs[x]);
        int* _tD = (int*)D;
        _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2];
    }
}

static inline void interpolateCubic( float x, float* coeffs )
{
    const float A = -0.75f;

    coeffs[0] = ((A*(x + 1) - 5*A)*(x + 1) + 8*A)*(x + 1) - 4*A;
    coeffs[1] = ((A + 2)*x - (A + 3))*x*x + 1;
    coeffs[2] = ((A + 2)*(1 - x) - (A + 3))*(1 - x)*(1 - x) + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

static inline void interpolateLanczos4( float x, float* coeffs )
{
    static const double s45 = 0.70710678118654752440084436210485;
    static const double cs[][2]=
    {{1, 0}, {-s45, -s45}, {0, 1}, {s45, -s45}, {-1, 0}, {s45, s45}, {0, -1}, {-s45, s45}};

    float sum = 0;
    double y0=-(x+3)*CV_PI*0.25, s0 = std::sin(y0), c0= std::cos(y0);
    for(int i = 0; i < 8; i++ )
    {
        float y0_ = (x+3-i);
        if (fabs(y0_) >= 1e-6f)
        {
            double y = -y0_*CV_PI*0.25;
            coeffs[i] = (float)((cs[i][0]*s0 + cs[i][1]*c0)/(y*y));
        }
        else
        {
            // special handling for 'x' values:
            // - ~0.0: 0 0 0 1 0 0 0 0
            // - ~1.0: 0 0 0 0 1 0 0 0
            coeffs[i] = 1e30f;
        }
        sum += coeffs[i];
    }

    sum = 1.f/sum;
    for(int i = 0; i < 8; i++ )
        coeffs[i] *= sum;
}

template<typename ST, typename DT> struct Cast
{
    typedef ST type1;
    typedef DT rtype;

    DT operator()(ST val) const { return saturate_cast<DT>(val); }
};

template<typename ST, typename DT, int bits> struct FixedPtCast
{
    typedef ST type1;
    typedef DT rtype;
    enum { SHIFT = bits, DELTA = 1 << (bits-1) };

    DT operator()(ST val) const { return saturate_cast<DT>((val + DELTA)>>SHIFT); }
};

/****************************************************************************************\
*                                         Resize                                         *
\****************************************************************************************/

namespace nn_bitexact_detail {
static void resizeNN_bitexact2xUp_(const Mat& src, Mat& dst);
}

class resizeNNInvoker_ : public ParallelLoopBody
{
public:
    resizeNNInvoker_(const Mat& _src, Mat &_dst, int *_x_ofs, double _ify) :
        ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs), ify(_ify)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size ssize = src.size(), dsize = dst.size();
        const int pix_size = (int)src.elemSize();
        const int width = dsize.width;
        const int row_bytes = ssize.width * pix_size;
        const int vend = resizeNN_vec_end_for(x_ofs, width, row_bytes, pix_size);

        const size_t dst_row_bytes = (size_t)width * pix_size;
        const uchar* prevD = 0;
        int prev_sy = -1;

        for( int y = range.start; y < range.end; y++ )
        {
            uchar* D = dst.ptr(y);
            const int sy = std::min(cvFloor(y*ify), ssize.height-1);
            if( sy == prev_sy )
            {
                memcpy(D, prevD, dst_row_bytes);
                continue;
            }
            prevD = D;
            prev_sy = sy;

            const uchar* S = src.ptr(sy);
            int x = 0;

            switch( pix_size )
            {
            case 1:
                resizeNN_row_u8(D, S, x_ofs, width, row_bytes, vend);
                break;
            case 2:
                resizeNN_row_u16(D, S, x_ofs, width, vend);
                break;
            case 3:
                resizeNN_row_px3(D, S, x_ofs, width);
                break;
            case 4:
                resizeNN_row_u32(D, S, x_ofs, width);
                break;
            case 6:
                for( ; x < width; x++, D += 6 )
                {
                    const ushort* _tS = (const ushort*)(S + x_ofs[x]);
                    ushort* _tD = (ushort*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2];
                }
                break;
            case 8:
                resizeNN_row_u64(D, S, x_ofs, width);
                break;
            case 12:
                resizeNN_row_px12(D, S, x_ofs, width, vend);
                break;
            case 16:
                for( ; x < width; x++ )
                {
                    const int* _tS = (const int*)(S + x_ofs[x]);
                    int* _tD = (int*)(D + x * 16);
                    _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2]; _tD[3] = _tS[3];
                }
                break;
            default:
                for( x = 0; x < width; x++, D += pix_size )
                {
                    const uchar* _tS = S + x_ofs[x];
                    for (int k = 0; k < pix_size; k++)
                        D[k] = _tS[k];
                }
            }
        }
    }

private:
    const Mat& src;
    Mat& dst;
    int* x_ofs;
    double ify;

    resizeNNInvoker_(const resizeNNInvoker_&);
    resizeNNInvoker_& operator=(const resizeNNInvoker_&);
};

void resizeNN_( const Mat& src, Mat& dst, double fx, double fy )
{
    Size ssize = src.size(), dsize = dst.size();
    const double ifx = 1./fx, ify = 1./fy;
    const int pix_size = (int)src.elemSize();

    AutoBuffer<int> _x_ofs(dsize.width);
    int* x_ofs = _x_ofs.data();

    for( int x = 0; x < dsize.width; x++ )
    {
        const int sx = cvFloor(x*ifx);
        x_ofs[x] = std::min(sx, ssize.width-1) * pix_size;
    }

    Range range(0, dsize.height);
    resizeNNInvoker_ invoker(src, dst, x_ofs, ify);
    parallel_for_(range, invoker, dst.total()/(double)(1<<16));
}

namespace nn_bitexact_detail {

static inline void nn_bitexact_fixed_coefs(int src_sz, int dst_sz, int& ifc, int& ifc0)
{
    ifc = ((src_sz << 16) + dst_sz / 2) / dst_sz;
    ifc0 = ifc / 2 - src_sz % 2;
}

// The fixed-point mapping of an exact 2x upscale is sx = x >> 1 along this axis.
static inline bool is_up2x(int ssz, int dsz, int ifc, int ifc0)
{
    return dsz == 2 * ssz && ssz > 0 && ifc == (1 << 15) && ifc0 == (1 << 14) - ssz % 2;
}

static inline void nn_bitexact_row_cn3_up2(uchar* D, const uchar* S, int sw)
{
    int sx = 0;
    for (; sx <= sw - 4; sx += 4, S += 12, D += 24)
    {
        D[0] = S[0]; D[1] = S[1]; D[2] = S[2]; D[3] = S[0]; D[4] = S[1]; D[5] = S[2];
        D[6] = S[3]; D[7] = S[4]; D[8] = S[5]; D[9] = S[3]; D[10] = S[4]; D[11] = S[5];
        D[12] = S[6]; D[13] = S[7]; D[14] = S[8]; D[15] = S[6]; D[16] = S[7]; D[17] = S[8];
        D[18] = S[9]; D[19] = S[10]; D[20] = S[11]; D[21] = S[9]; D[22] = S[10]; D[23] = S[11];
    }
    for (; sx < sw; ++sx, S += 3, D += 6)
    {
        D[0] = S[0]; D[1] = S[1]; D[2] = S[2];
        D[3] = S[0]; D[4] = S[1]; D[5] = S[2];
    }
}

class resizeNN_bitexact2xUpInvoker_ : public ParallelLoopBody
{
public:
    resizeNN_bitexact2xUpInvoker_(const Mat& _src, Mat& _dst, const int* _sx_ofs) :
        src(_src), dst(_dst), sx_ofs(_sx_ofs) {}

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        const int sh = src.rows, width = dst.cols;
        const int pix_size = (int)src.elemSize();

        for (int y = range.start; y < range.end; y++)
        {
            const int sy = std::min(y >> 1, sh - 1);
            uchar* D = dst.ptr(y);
            const uchar* S = src.ptr(sy);
            int x = 0;

            if (pix_size == 1)
            {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                const int vl8 = VTraits<v_uint8>::vlanes();
                for (; x <= width - 2 * vl8; x += 2 * vl8)
                {
                    v_uint8 s = vx_load(S + (x >> 1)), lo, hi;
                    v_zip(s, s, lo, hi);
                    v_store(D + x, lo);
                    v_store(D + x + vl8, hi);
                }
#endif
                for (; x < width; x++)
                    D[x] = S[sx_ofs[x]];
            }
            else if (pix_size == 2)
            {
                const ushort* S16 = (const ushort*)S;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                const int vl16 = VTraits<v_uint16>::vlanes();
                for (; x <= width - 2 * vl16; x += 2 * vl16)
                {
                    v_uint16 s = vx_load(S16 + (x >> 1)), lo, hi;
                    v_zip(s, s, lo, hi);
                    v_store((ushort*)D + x, lo);
                    v_store((ushort*)D + x + vl16, hi);
                }
#endif
                for (; x < width; x++)
                    ((ushort*)D)[x] = S16[sx_ofs[x]];
            }
            else if (pix_size == 3)
            {
                nn_bitexact_row_cn3_up2(D, S, src.cols);
            }
            else if (pix_size == 4)
            {
                const uint32_t* S32 = (const uint32_t*)S;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                const int vl32 = VTraits<v_uint32>::vlanes();
                for (; x <= width - 2 * vl32; x += 2 * vl32)
                {
                    v_uint32 s = vx_load(S32 + (x >> 1)), lo, hi;
                    v_zip(s, s, lo, hi);
                    v_store((uint32_t*)D + x, lo);
                    v_store((uint32_t*)D + x + vl32, hi);
                }
#endif
                for (; x < width; x++)
                    ((uint32_t*)D)[x] = S32[sx_ofs[x]];
            }
        }
    }

private:
    const Mat& src;
    Mat& dst;
    const int* sx_ofs;
};

static void resizeNN_bitexact2xUp_(const Mat& src, Mat& dst)
{
    const int dw = dst.cols;
    AutoBuffer<int> _sx_ofs(dw);
    int* sx_ofs = _sx_ofs.data();
    for (int x = 0; x < dw; x++)
        sx_ofs[x] = x >> 1;

    Range range(0, dst.rows);
    resizeNN_bitexact2xUpInvoker_ invoker(src, dst, sx_ofs);
    parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
}

}  // namespace nn_bitexact_detail

class resizeNN_bitexactInvoker : public ParallelLoopBody
{
public:
    resizeNN_bitexactInvoker(const Mat& _src, Mat& _dst, int* _x_ofs, int _ify, int _ify0)
        : src(_src), dst(_dst), x_ofs(_x_ofs), ify(_ify), ify0(_ify0) {}

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size ssize = src.size(), dsize = dst.size();
        int pix_size = (int)src.elemSize();
        const int row_bytes = ssize.width * pix_size;
        const int vend = resizeNN_vec_end_for(x_ofs, dsize.width, row_bytes, pix_size);
        for( int y = range.start; y < range.end; y++ )
        {
            uchar* D = dst.ptr(y);
            int _sy = (ify * y + ify0) >> 16;
            int sy = std::min(_sy, ssize.height-1);
            const uchar* S = src.ptr(sy);

            int x = 0;
            switch( pix_size )
            {
            case 1:
                resizeNN_row_u8(D, S, x_ofs, dsize.width, row_bytes, vend);
                break;
            case 2:
                resizeNN_row_u16(D, S, x_ofs, dsize.width, vend);
                break;
            case 3:
                resizeNN_row_px3(D, S, x_ofs, dsize.width);
                break;
            case 4:
                resizeNN_row_u32(D, S, x_ofs, dsize.width);
                break;
            case 6:
                for( ; x < dsize.width; x++, D += 6 )
                {
                    const ushort* _tS = (const ushort*)(S + x_ofs[x]);
                    ushort* _tD = (ushort*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2];
                }
                break;
            case 8:
                resizeNN_row_u64(D, S, x_ofs, dsize.width);
                break;
            case 12:
                resizeNN_row_px12(D, S, x_ofs, dsize.width, vend);
                break;
            default:
                for( x = 0; x < dsize.width; x++, D += pix_size )
                {
                    const uchar* _tS = S + x_ofs[x];
                    for (int k = 0; k < pix_size; k++)
                        D[k] = _tS[k];
                }
            }
        }
    }
private:
    const Mat& src;
    Mat& dst;
    int* x_ofs;
    const int ify;
    const int ify0;
};

static void resizeNN_bitexact( const Mat& src, Mat& dst, double /*fx*/, double /*fy*/ )
{
    Size ssize = src.size(), dsize = dst.size();
    const int pix_size = (int)src.elemSize();
    int ifx, ifx0, ify, ify0;
    nn_bitexact_detail::nn_bitexact_fixed_coefs(ssize.width, dsize.width, ifx, ifx0);
    nn_bitexact_detail::nn_bitexact_fixed_coefs(ssize.height, dsize.height, ify, ify0);

    if (pix_size <= 4 &&
        nn_bitexact_detail::is_up2x(ssize.width, dsize.width, ifx, ifx0) &&
        nn_bitexact_detail::is_up2x(ssize.height, dsize.height, ify, ify0))
    {
        nn_bitexact_detail::resizeNN_bitexact2xUp_(src, dst);
        return;
    }

    cv::utils::BufferArea area;
    int* x_ofs = 0;
    area.allocate(x_ofs, dsize.width, CV_SIMD_WIDTH);
    area.commit();

    for( int x = 0; x < dsize.width; x++ )
    {
        int sx = (ifx * x + ifx0) >> 16;
        x_ofs[x] = std::min(sx, ssize.width-1) * pix_size;
    }
    Range range(0, dsize.height);
    resizeNN_bitexactInvoker invoker(src, dst, x_ofs, ify, ify0);
    parallel_for_(range, invoker, dst.total()/(double)(1<<16));
}

namespace linear_up_2x_detail {

static inline bool is_2x_up(int sw, int dw, int sh, int dh, double inv_scale_x, double inv_scale_y)
{
    return std::abs(inv_scale_x - 2.0) < DBL_EPSILON && std::abs(inv_scale_y - 2.0) < DBL_EPSILON &&
           dw == sw * 2 && dh == sh * 2 && sw > 0 && sh > 0;
}

}  // namespace linear_up_2x_detail

struct VResizeNoVec
{
    template<typename WT, typename T, typename BT>
    int operator()(const WT**, T*, const BT*, int ) const
    {
        return 0;
    }
};

struct HResizeNoVec
{
    template<typename T, typename WT, typename AT> inline
    int operator()(const T**, WT**, int, const int*,
        const AT*, int, int, int, int, int) const
    {
        return 0;
    }
};

#if (CV_SIMD || CV_SIMD_SCALABLE)

#if CV_NEON

struct VResizeLinearVec_32s8u
{
    int operator()(const int** src, uchar* dst, const short* beta, int width) const
    {
        const int *S0 = src[0], *S1 = src[1];
        int x = 0;
        v_int16 b0 = vx_setall_s16(beta[0]), b1 = vx_setall_s16(beta[1]);
        const int vlane8 = VTraits<v_uint8>::vlanes();
        const int vlane32 = VTraits<v_int32>::vlanes();
        const int vlane16 = VTraits<v_int16>::vlanes();

        if( (((size_t)S0|(size_t)S1)&(vlane8 - 1)) == 0 )
            for( ; x <= width - vlane8; x += vlane8)
                v_store(dst + x, v_rshr_pack_u<2>(v_add(v_mul_hi(v_pack(v_shr<4>(vx_load_aligned(S0 + x)), v_shr<4>(vx_load_aligned(S0 + x + vlane32))), b0), v_mul_hi(v_pack(v_shr<4>(vx_load_aligned(S1 + x)), v_shr<4>(vx_load_aligned(S1 + x + vlane32))), b1)),
                                                  v_add(v_mul_hi(v_pack(v_shr<4>(vx_load_aligned(S0 + x + 2 * vlane32)), v_shr<4>(vx_load_aligned(S0 + x + 3 * vlane32))), b0), v_mul_hi(v_pack(v_shr<4>(vx_load_aligned(S1 + x + 2 * vlane32)), v_shr<4>(vx_load_aligned(S1 + x + 3 * vlane32))), b1))));
        else
            for( ; x <= width - vlane8; x += vlane8)
                v_store(dst + x, v_rshr_pack_u<2>(v_add(v_mul_hi(v_pack(v_shr<4>(vx_load(S0 + x)), v_shr<4>(vx_load(S0 + x + vlane32))), b0), v_mul_hi(v_pack(v_shr<4>(vx_load(S1 + x)), v_shr<4>(vx_load(S1 + x + vlane32))), b1)),
                                                  v_add(v_mul_hi(v_pack(v_shr<4>(vx_load(S0 + x + 2 * vlane32)), v_shr<4>(vx_load(S0 + x + 3 * vlane32))), b0), v_mul_hi(v_pack(v_shr<4>(vx_load(S1 + x + 2 * vlane32)), v_shr<4>(vx_load(S1 + x + 3 * vlane32))), b1))));

        for( ; x <= width - vlane16; x += vlane16)
            v_rshr_pack_u_store<2>(dst + x, v_add(v_mul_hi(v_pack(v_shr<4>(vx_load(S0 + x)), v_shr<4>(vx_load(S0 + x + vlane32))), b0), v_mul_hi(v_pack(v_shr<4>(vx_load(S1 + x)), v_shr<4>(vx_load(S1 + x + vlane32))), b1)));

        return x;
    }
};

#else // CV_NEON

struct VResizeLinearVec_32s8u
{
    void inline loadAlignedData(const int* s0, const int* s1, v_int32&m0, v_int32&m1, v_int32&m2, v_int32&m3, v_int32&t0, v_int32&t1, v_int32&t2, v_int32&t3) const
    {
        m0 = vx_load_aligned(s0);
        m1 = vx_load_aligned(s0 + vlane32);
        m2 = vx_load_aligned(s0 + vlane32x2);
        m3 = vx_load_aligned(s0 + vlane32x3);
        t0 = vx_load_aligned(s1);
        t1 = vx_load_aligned(s1 + vlane32);
        t2 = vx_load_aligned(s1 + vlane32x2);
        t3 = vx_load_aligned(s1 + vlane32x3);
    }
    void inline loadAlignedData(const int* s0, const int* s1, v_int32&m0, v_int32&m1, v_int32&t0, v_int32&t1) const
    {
        m0 = vx_load_aligned(s0);
        m1 = vx_load_aligned(s0 + vlane32);
        t0 = vx_load_aligned(s1);
        t1 = vx_load_aligned(s1 + vlane32);
    }
    void inline loadData(const int* s0, const int* s1, v_int32&m0, v_int32&m1, v_int32&m2, v_int32&m3, v_int32&t0, v_int32&t1, v_int32&t2, v_int32&t3) const
    {
        m0 = vx_load(s0);
        m1 = vx_load(s0 + vlane32);
        m2 = vx_load(s0 + vlane32x2);
        m3 = vx_load(s0 + vlane32x3);
        t0 = vx_load(s1);
        t1 = vx_load(s1 + vlane32);
        t2 = vx_load(s1 + vlane32x2);
        t3 = vx_load(s1 + vlane32x3);
    }
    void inline loadData(const int* s0, const int* s1, v_int32&m0, v_int32&m1,v_int32&t0, v_int32&t1) const
    {
        m0 = vx_load(s0);
        m1 = vx_load(s0 + vlane32);
        t0 = vx_load(s1);
        t1 = vx_load(s1 + vlane32);
    }
    void inline shiftRight4(v_int32&a0, v_int32&a1, v_int32&a2, v_int32&a3) const
    {
        a0 = v_shr<4>(a0);
        a1 = v_shr<4>(a1);
        a2 = v_shr<4>(a2);
        a3 = v_shr<4>(a3);
    }
    int operator()(const int** src, uchar* dst, const short* beta, int width) const
    {
        const int *S0 = src[0], *S1 = src[1];
        int x = 0;
        v_int16 b0 = vx_setall_s16(beta[0]), b1 = vx_setall_s16(beta[1]);
        v_int16 a0, a1, a2, a3;
        v_int32 m0, m1, m2, m3, t0, t1, t2, t3;

        if( (((size_t)S0|(size_t)S1)&(vlane8 - 1)) == 0 )
        {
            for( ; x <= width - vlane8; x += vlane8)
            {
                const int *s0 = S0 + x;
                const int *s1 = S1 + x;
                loadAlignedData(s0, s1, m0, m1, m2, m3, t0, t1, t2, t3);
                shiftRight4(m0, m1, m2, m3);
                shiftRight4(t0, t1, t2, t3);
                a0 = v_mul_hi(v_pack(m0, m1), b0);
                a1 = v_mul_hi(v_pack(t0, t1), b1);
                a2 = v_mul_hi(v_pack(m2, m3), b0);
                a3 = v_mul_hi(v_pack(t2, t3), b1);
                v_store(dst + x, v_rshr_pack_u<2>(v_add(a0, a1),
                                                  v_add(a2, a3)));
            }
        }
        else
        {
            for( ; x <= width - vlane8; x += vlane8)
            {
                const int *s0 = S0 + x;
                const int *s1 = S1 + x;
                loadData(s0, s1, m0, m1, m2, m3, t0, t1, t2, t3);
                shiftRight4(m0, m1, m2, m3);
                shiftRight4(t0, t1, t2, t3);
                a0 = v_mul_hi(v_pack(m0, m1), b0);
                a1 = v_mul_hi(v_pack(t0, t1), b1);
                a2 = v_mul_hi(v_pack(m2, m3), b0);
                a3 = v_mul_hi(v_pack(t2, t3), b1);
                v_store(dst + x, v_rshr_pack_u<2>(v_add(a0, a1),
                                                  v_add(a2, a3)));
            }
        }
        for( ; x <= width - vlane16; x += vlane16)
        {
            const int *s0 = S0 + x;
            const int *s1 = S1 + x;
            loadData(s0, s1, m0, m1, t0, t1);
            shiftRight4(m0, m1, t0, t1);
            a0 = v_mul_hi(v_pack(m0, m1), b0);
            a1 = v_mul_hi(v_pack(t0, t1), b1);
            v_rshr_pack_u_store<2>(dst + x, v_add(a0, a1));
        }

        return x;
    }

    private:
        const int vlane8 = VTraits<v_uint8>::vlanes();
        const int vlane32 = VTraits<v_int32>::vlanes();
        const int vlane16 = VTraits<v_int16>::vlanes();
        const int vlane32x2 = vlane32 * 2;
        const int vlane32x3 = vlane32 * 3;
};

#endif // CV_NEON

static inline void v_pack_t(v_int32 a, v_int32 b, v_uint16& dst) {
    dst = v_pack_u(a, b);
}
static inline void v_pack_t(v_int32 a, v_int32 b, v_int16& dst) {
    dst = v_pack(a, b);
}

template<typename DT, typename VT>
struct VResizeLinearVec_32f16
{
    void inline loadAlignedData(const float* s0, const float* s1, v_float32&m0, v_float32&m1, v_float32&t0, v_float32&t1) const
    {
        m0 = vx_load_aligned(s0);
        m1 = vx_load_aligned(s0 + vlane32);
        t0 = vx_load_aligned(s1);
        t1 = vx_load_aligned(s1 + vlane32);
    }
    void inline loadData(const float* s0, const float* s1, v_float32&m0, v_float32&m1, v_float32&t0, v_float32&t1) const
    {
        m0 = vx_load(s0);
        m1 = vx_load(s0 + vlane32);
        t0 = vx_load(s1);
        t1 = vx_load(s1 + vlane32);
    }
    int operator()(const float** src, DT* dst, const float* beta, int width) const
    {
        const float *S0 = src[0], *S1 = src[1];
        int x = 0;

        v_float32 b0 = vx_setall_f32(beta[0]), b1 = vx_setall_f32(beta[1]);
        v_float32 m0, m1, t0, t1;
        VT res;

        if( (((size_t)S0|(size_t)S1)&(vlane8 - 1)) == 0 )
            for( ; x <= width - vlane16; x += vlane16)
            {
                loadAlignedData(S0 + x, S1 + x, m0, m1, t0, t1);
                v_pack_t(v_round(v_muladd(m0, b0, v_mul(t0, b1))),
                                          v_round(v_muladd(m1, b0, v_mul(t1, b1))), res);
                v_store(dst + x, res);
            }
        else
            for (; x <= width - vlane16; x += vlane16)
            {
                loadData(S0 + x, S1 + x, m0, m1, t0, t1);
                v_pack_t(v_round(v_muladd(m0, b0, v_mul(t0, b1))),
                                          v_round(v_muladd(m1, b0, v_mul(t1, b1))), res);
                v_store(dst + x, res);
            }
        for( ; x <= width - vlane32; x += vlane32)
        {
            v_int32 n = v_round(v_muladd(vx_load(S0 + x), b0, v_mul(vx_load(S1 + x), b1)));
            v_pack_t(n, n, res);
            v_store_low(dst + x, res);
        }

        return x;
    }

    private:
        const int vlane8 = VTraits<v_uint8>::vlanes();
        const int vlane16 = VTraits<VT>::vlanes();
        const int vlane32 = VTraits<v_float32>::vlanes();
};

typedef VResizeLinearVec_32f16<ushort, v_uint16> VResizeLinearVec_32f16u;
typedef VResizeLinearVec_32f16<short, v_int16> VResizeLinearVec_32f16s;

struct VResizeLinearVec_32f
{
    void inline loadAlignedData(const float* s0, const float* s1, v_float32&m0, v_float32&t0) const
    {
        m0 = vx_load_aligned(s0);
        t0 = vx_load_aligned(s1);
    }
    void inline loadData(const float* s0, const float* s1, v_float32&m0, v_float32&t0) const
    {
        m0 = vx_load(s0);
        t0 = vx_load(s1);
    }
    int operator()(const float** src, float* dst, const float* beta, int width) const
    {
        const float *S0 = src[0], *S1 = src[1];
        int x = 0;

        v_float32 b0 = vx_setall_f32(beta[0]), b1 = vx_setall_f32(beta[1]);
        v_float32 m0, t0;

        if( (((size_t)S0|(size_t)S1)&(vlane8 - 1)) == 0 )
            for( ; x <= width - vlane32; x += vlane32)
            {
                loadAlignedData(S0 + x, S1 + x, m0, t0);
                v_store(dst + x, v_muladd(m0, b0, v_mul(t0, b1)));
            }
        else
            for( ; x <= width - vlane32; x += vlane32)
            {
                loadData(S0 + x, S1 + x, m0, t0);
                v_store(dst + x, v_muladd(m0, b0, v_mul(t0, b1)));
            }

        return x;
    }

    private:
        const int vlane8 = VTraits<v_uint8>::vlanes();
        const int vlane32 = VTraits<v_float32>::vlanes();
};


struct VResizeCubicVec_32s8u
{
    void inline loadAlignedData(const int* s0, const int* s1, const int* s2, const int* s3, v_int32&m0, v_int32&m1, v_int32&t0, v_int32&t1, v_int32&u0, v_int32&u1, v_int32&v0, v_int32&v1) const
    {
        m0 = vx_load_aligned(s0);
        m1 = vx_load_aligned(s0 + vlane32);
        t0 = vx_load_aligned(s1);
        t1 = vx_load_aligned(s1 + vlane32);
        u0 = vx_load_aligned(s2);
        u1 = vx_load_aligned(s2 + vlane32);
        v0 = vx_load_aligned(s3);
        v1 = vx_load_aligned(s3 + vlane32);
    }
    void inline loadData(const int* s0, const int* s1, const int* s2, const int* s3, v_int32&m0, v_int32&m1, v_int32&t0, v_int32&t1, v_int32&u0, v_int32&u1, v_int32&v0, v_int32&v1) const
    {
        m0 = vx_load(s0);
        m1 = vx_load(s0 + vlane32);
        t0 = vx_load(s1);
        t1 = vx_load(s1 + vlane32);
        u0 = vx_load(s2);
        u1 = vx_load(s2 + vlane32);
        v0 = vx_load(s3);
        v1 = vx_load(s3 + vlane32);
    }
    int operator()(const int** src, uchar* dst, const short* beta, int width) const
    {
        const int *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        int x = 0;
        float scale = 1.f/(INTER_RESIZE_COEF_SCALE*INTER_RESIZE_COEF_SCALE);

        v_float32 b0 = vx_setall_f32(beta[0] * scale), b1 = vx_setall_f32(beta[1] * scale),
                  b2 = vx_setall_f32(beta[2] * scale), b3 = vx_setall_f32(beta[3] * scale);
        v_int32 m0, m1, t0, t1, u0, u1, v0, v1;

        if( (((size_t)S0|(size_t)S1|(size_t)S2|(size_t)S3)&(vlane8 - 1)) == 0 )
            for( ; x <= width - vlane16; x += vlane16)
            {
                loadAlignedData(S0 + x, S1 + x, S2 + x, S3 + x, m0, m1, t0, t1, u0, u1, v0, v1);
                v_pack_u_store(dst + x, v_pack(v_round(v_muladd(v_cvt_f32(m0),  b0,
                                                       v_muladd(v_cvt_f32(t0),  b1,
                                                       v_muladd(v_cvt_f32(u0),  b2,
                                                       v_mul(v_cvt_f32(v0), b3))))),
                                               v_round(v_muladd(v_cvt_f32(m1),  b0,
                                                       v_muladd(v_cvt_f32(t1),  b1,
                                                       v_muladd(v_cvt_f32(u1),  b2,
                                                       v_mul(v_cvt_f32(v1), b3)))))));
            }
        else
            for( ; x <= width - vlane16; x += vlane16)
            {
                loadData(S0 + x, S1 + x, S2 + x, S3 + x, m0, m1, t0, t1, u0, u1, v0, v1);
                v_pack_u_store(dst + x, v_pack(v_round(v_muladd(v_cvt_f32(m0),  b0,
                                                       v_muladd(v_cvt_f32(t0),  b1,
                                                       v_muladd(v_cvt_f32(u0),  b2,
                                                       v_mul(v_cvt_f32(v0), b3))))),
                                               v_round(v_muladd(v_cvt_f32(m1),  b0,
                                                       v_muladd(v_cvt_f32(t1),  b1,
                                                       v_muladd(v_cvt_f32(u1),  b2,
                                                       v_mul(v_cvt_f32(v1), b3)))))));
            }
        return x;
    }

    private:
        const int vlane8 = VTraits<v_uint8>::vlanes();
        const int vlane16 = VTraits<v_int16>::vlanes();
        const int vlane32 = VTraits<v_float32>::vlanes();
};

template<typename DT, typename VT>
struct VResizeCubicVec_32f16
{
    int operator()(const float** src, DT* dst, const float* beta, int width) const
    {
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        int x = 0;
        v_float32 b0 = vx_setall_f32(beta[0]), b1 = vx_setall_f32(beta[1]),
                  b2 = vx_setall_f32(beta[2]), b3 = vx_setall_f32(beta[3]);
        VT res;

        for (; x <= width - vlane16; x += vlane16)
        {
            v_pack_t(v_round(v_muladd(vx_load(S0 + x),  b0,
                                      v_muladd(vx_load(S1 + x),  b1,
                                      v_muladd(vx_load(S2 + x),  b2,
                                      v_mul(vx_load(S3 + x), b3))))),
                     v_round(v_muladd(vx_load(S0 + x + vlane32),  b0,
                                      v_muladd(vx_load(S1 + x + vlane32),  b1,
                                      v_muladd(vx_load(S2 + x + vlane32),  b2,
                                      v_mul(vx_load(S3 + x + vlane32), b3))))), res);
            v_store(dst + x, res);
        }

        return x;
    }

    private:
        const int vlane16 = VTraits<VT>::vlanes();
        const int vlane32 = VTraits<v_float32>::vlanes();
};
typedef VResizeCubicVec_32f16<ushort, v_uint16> VResizeCubicVec_32f16u;
typedef VResizeCubicVec_32f16<short, v_int16> VResizeCubicVec_32f16s;

struct VResizeCubicVec_32f
{
    int operator()(const float** src, float* dst, const float* beta, int width) const
    {
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        int x = 0;
        const int vlane32 = VTraits<v_float32>::vlanes();
        v_float32 b0 = vx_setall_f32(beta[0]), b1 = vx_setall_f32(beta[1]),
                  b2 = vx_setall_f32(beta[2]), b3 = vx_setall_f32(beta[3]);

        for( ; x <= width - vlane32; x += vlane32)
            v_store(dst + x, v_muladd(vx_load(S0 + x),  b0,
                             v_muladd(vx_load(S1 + x),  b1,
                             v_muladd(vx_load(S2 + x),  b2,
                             v_mul(vx_load(S3 + x), b3)))));

        return x;
    }
};


// 16-bit Lanczos4 vertical pass; ushort/short differ only by the final pack,
// handled by the v_pack_t() overloads (same approach as linear/cubic 32f16).
template<typename DT, typename VT>
struct VResizeLanczos4Vec_32f16
{
    int operator()(const float** src, DT* dst, const float* beta, int width ) const
    {
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3],
                    *S4 = src[4], *S5 = src[5], *S6 = src[6], *S7 = src[7];
        int x = 0;
        const int vlane16 = VTraits<VT>::vlanes();
        const int vlane32 = VTraits<v_float32>::vlanes();
        v_float32 b0 = vx_setall_f32(beta[0]), b1 = vx_setall_f32(beta[1]),
                  b2 = vx_setall_f32(beta[2]), b3 = vx_setall_f32(beta[3]),
                  b4 = vx_setall_f32(beta[4]), b5 = vx_setall_f32(beta[5]),
                  b6 = vx_setall_f32(beta[6]), b7 = vx_setall_f32(beta[7]);

        VT res;
        for( ; x <= width - vlane16; x += vlane16)
        {
            v_pack_t(v_round(v_muladd(vx_load(S0 + x),  b0,
                     v_muladd(vx_load(S1 + x),  b1,
                     v_muladd(vx_load(S2 + x),  b2,
                     v_muladd(vx_load(S3 + x),  b3,
                     v_muladd(vx_load(S4 + x),  b4,
                     v_muladd(vx_load(S5 + x),  b5,
                     v_muladd(vx_load(S6 + x),  b6,
                     v_mul(vx_load(S7 + x), b7))))))))),
                     v_round(v_muladd(vx_load(S0 + x + vlane32),  b0,
                     v_muladd(vx_load(S1 + x + vlane32),  b1,
                     v_muladd(vx_load(S2 + x + vlane32),  b2,
                     v_muladd(vx_load(S3 + x + vlane32),  b3,
                     v_muladd(vx_load(S4 + x + vlane32),  b4,
                     v_muladd(vx_load(S5 + x + vlane32),  b5,
                     v_muladd(vx_load(S6 + x + vlane32),  b6,
                     v_mul(vx_load(S7 + x + vlane32), b7))))))))),
                     res);
            v_store(dst + x, res);
        }

        return x;
    }
};
typedef VResizeLanczos4Vec_32f16<ushort, v_uint16> VResizeLanczos4Vec_32f16u;
typedef VResizeLanczos4Vec_32f16<short, v_int16> VResizeLanczos4Vec_32f16s;

struct VResizeLanczos4Vec_32f
{
    int operator()(const float** src, float* dst, const float* beta, int width ) const
    {
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3],
                    *S4 = src[4], *S5 = src[5], *S6 = src[6], *S7 = src[7];
        int x = 0;
        const int vlane32 = VTraits<v_float32>::vlanes();

        v_float32 b0 = vx_setall_f32(beta[0]), b1 = vx_setall_f32(beta[1]),
                  b2 = vx_setall_f32(beta[2]), b3 = vx_setall_f32(beta[3]),
                  b4 = vx_setall_f32(beta[4]), b5 = vx_setall_f32(beta[5]),
                  b6 = vx_setall_f32(beta[6]), b7 = vx_setall_f32(beta[7]);

        for( ; x <= width - vlane32; x += vlane32)
            v_store(dst + x, v_muladd(vx_load(S0 + x),  b0,
                             v_muladd(vx_load(S1 + x),  b1,
                             v_muladd(vx_load(S2 + x),  b2,
                             v_muladd(vx_load(S3 + x),  b3,
                             v_muladd(vx_load(S4 + x),  b4,
                             v_muladd(vx_load(S5 + x),  b5,
                             v_muladd(vx_load(S6 + x),  b6,
                                      v_mul(vx_load(S7 + x), b7)))))))));

        return x;
    }
};

#else

typedef VResizeNoVec VResizeLinearVec_32s8u;
typedef VResizeNoVec VResizeLinearVec_32f16u;
typedef VResizeNoVec VResizeLinearVec_32f16s;
typedef VResizeNoVec VResizeLinearVec_32f;

typedef VResizeNoVec VResizeCubicVec_32s8u;
typedef VResizeNoVec VResizeCubicVec_32f16u;
typedef VResizeNoVec VResizeCubicVec_32f16s;
typedef VResizeNoVec VResizeCubicVec_32f;

typedef VResizeNoVec VResizeLanczos4Vec_32f16u;
typedef VResizeNoVec VResizeLanczos4Vec_32f16s;
typedef VResizeNoVec VResizeLanczos4Vec_32f;

#endif

#if CV_SIMD  // fixed-width 128-bit kernels; not valid for scalable backends (RVV)

static int hresize_linear_u8_cn1(const uchar** src, int** dst, int count,
                                 const int* xofs, const short* alpha, int xmax)
{
    const int step = 8;
    const int len0 = xmax & -step;
    int dx = 0, k = 0;

    for( ; k <= count - 2; k += 2 )
    {
        const uchar *S0 = src[k];
        int *D0 = dst[k];
        const uchar *S1 = src[k + 1];
        int *D1 = dst[k + 1];

        for( dx = 0; dx < len0; dx += step )
        {
            v_int16x8 al = v_load((const short*)(alpha + dx * 2));
            v_int16x8 ah = v_load((const short*)(alpha + dx * 2 + 8));
            v_uint16x8 sl, sh;
            v_expand(v_lut_pairs(S0, xofs + dx), sl, sh);
            v_store(D0 + dx, v_dotprod(v_reinterpret_as_s16(sl), al));
            v_store(D0 + dx + 4, v_dotprod(v_reinterpret_as_s16(sh), ah));
            v_expand(v_lut_pairs(S1, xofs + dx), sl, sh);
            v_store(D1 + dx, v_dotprod(v_reinterpret_as_s16(sl), al));
            v_store(D1 + dx + 4, v_dotprod(v_reinterpret_as_s16(sh), ah));
        }
    }
    for( ; k < count; k++ )
    {
        const uchar *S = src[k];
        int *D = dst[k];
        for( dx = 0; dx < len0; dx += step )
        {
            v_int16x8 al = v_load((const short*)(alpha + dx * 2));
            v_int16x8 ah = v_load((const short*)(alpha + dx * 2 + 8));
            v_uint16x8 sl, sh;
            v_expand(v_lut_pairs(S, xofs + dx), sl, sh);
            v_store(D + dx, v_dotprod(v_reinterpret_as_s16(sl), al));
            v_store(D + dx + 4, v_dotprod(v_reinterpret_as_s16(sh), ah));
        }
    }
    return dx;
}

static int hresize_linear_u8_cn2(const uchar** src, int** dst, int count,
                                 const int* xofs, const short* alpha, int xmax)
{
    const int step = 8;
    const int len0 = xmax & -step;
    int dx = 0, k = 0;

    for( ; k <= count - 2; k += 2 )
    {
        const uchar *S0 = src[k];
        int *D0 = dst[k];
        const uchar *S1 = src[k + 1];
        int *D1 = dst[k + 1];

        for( dx = 0; dx < len0; dx += step )
        {
            int ofs[4] = { xofs[dx], xofs[dx + 2], xofs[dx + 4], xofs[dx + 6] };
            v_int16x8 al = v_load((const short*)(alpha + dx * 2));
            v_int16x8 ah = v_load((const short*)(alpha + dx * 2 + 8));
            v_uint16x8 sl, sh;
            v_expand(v_interleave_pairs(v_lut_quads(S0, ofs)), sl, sh);
            v_store(D0 + dx, v_dotprod(v_reinterpret_as_s16(sl), al));
            v_store(D0 + dx + 4, v_dotprod(v_reinterpret_as_s16(sh), ah));
            v_expand(v_interleave_pairs(v_lut_quads(S1, ofs)), sl, sh);
            v_store(D1 + dx, v_dotprod(v_reinterpret_as_s16(sl), al));
            v_store(D1 + dx + 4, v_dotprod(v_reinterpret_as_s16(sh), ah));
        }
    }
    for( ; k < count; k++ )
    {
        const uchar *S = src[k];
        int *D = dst[k];
        for( dx = 0; dx < len0; dx += step )
        {
            int ofs[4] = { xofs[dx], xofs[dx + 2], xofs[dx + 4], xofs[dx + 6] };
            v_int16x8 al = v_load((const short*)(alpha + dx * 2));
            v_int16x8 ah = v_load((const short*)(alpha + dx * 2 + 8));
            v_uint16x8 sl, sh;
            v_expand(v_interleave_pairs(v_lut_quads(S, ofs)), sl, sh);
            v_store(D + dx, v_dotprod(v_reinterpret_as_s16(sl), al));
            v_store(D + dx + 4, v_dotprod(v_reinterpret_as_s16(sh), ah));
        }
    }
    return dx;
}

static int hresize_linear_u8_cn3(const uchar** src, int** dst, int count,
                                 const int* xofs, const short* alpha, int dmax, int cn, int /*xmax*/)
{
    const int smax = xofs[dmax - cn];
    int dx = 0, k = 0;

    for( ; k <= count - 2; k += 2 )
    {
        const uchar *S0 = src[k];
        int *D0 = dst[k];
        const uchar *S1 = src[k + 1];
        int *D1 = dst[k + 1];

        for( dx = 0; (xofs[dx] + cn) < smax; dx += cn )
        {
            v_int16x8 a = v_load((const short*)(alpha + dx * 2));
            v_store(D0 + dx, v_dotprod(v_reinterpret_as_s16(v_or(v_load_expand_q(S0 + xofs[dx]), v_shl<16>(v_load_expand_q(S0 + xofs[dx] + cn)))), a));
            v_store(D1 + dx, v_dotprod(v_reinterpret_as_s16(v_or(v_load_expand_q(S1 + xofs[dx]), v_shl<16>(v_load_expand_q(S1 + xofs[dx] + cn)))), a));
        }
    }
    for( ; k < count; k++ )
    {
        const uchar *S = src[k];
        int *D = dst[k];
        for( dx = 0; (xofs[dx] + cn) < smax; dx += cn )
        {
            v_int16x8 a = v_load((const short*)(alpha + dx * 2));
            v_store(D + dx, v_dotprod(v_reinterpret_as_s16(v_or(v_load_expand_q(S + xofs[dx]), v_shl<16>(v_load_expand_q(S + xofs[dx] + cn)))), a));
        }
    }
    CV_DbgAssert(dx < dmax);
    return dx;
}

static int hresize_linear_u8_cn4(const uchar** src, int** dst, int count,
                                 const int* xofs, const short* alpha, int xmax)
{
    const int step = 8;
    const int len0 = xmax & -step;
    int dx = 0, k = 0;

    for( ; k <= count - 2; k += 2 )
    {
        const uchar *S0 = src[k];
        int *D0 = dst[k];
        const uchar *S1 = src[k + 1];
        int *D1 = dst[k + 1];

        for( dx = 0; dx < len0; dx += step )
        {
            v_int16x8 a0 = v_load((const short*)(alpha + dx * 2));
            v_int16x8 a1 = v_load((const short*)(alpha + dx * 2 + 8));
            v_store(D0 + dx, v_dotprod(v_reinterpret_as_s16(v_interleave_quads(v_load_expand(S0 + xofs[dx]))), a0));
            v_store(D0 + dx + 4, v_dotprod(v_reinterpret_as_s16(v_interleave_quads(v_load_expand(S0 + xofs[dx + 4]))), a1));
            v_store(D1 + dx, v_dotprod(v_reinterpret_as_s16(v_interleave_quads(v_load_expand(S1 + xofs[dx]))), a0));
            v_store(D1 + dx + 4, v_dotprod(v_reinterpret_as_s16(v_interleave_quads(v_load_expand(S1 + xofs[dx + 4]))), a1));
        }
    }
    for( ; k < count; k++ )
    {
        const uchar *S = src[k];
        int *D = dst[k];
        for( dx = 0; dx < len0; dx += step )
        {
            v_int16x8 a0 = v_load((const short*)(alpha + dx * 2));
            v_int16x8 a1 = v_load((const short*)(alpha + dx * 2 + 8));
            v_store(D + dx, v_dotprod(v_reinterpret_as_s16(v_interleave_quads(v_load_expand(S + xofs[dx]))), a0));
            v_store(D + dx + 4, v_dotprod(v_reinterpret_as_s16(v_interleave_quads(v_load_expand(S + xofs[dx + 4]))), a1));
        }
    }
    return dx;
}

// 16-bit (ushort/short) gather-to-f32; identical apart from the source type.
template<typename T> static int hresize_linear_fp_16(const T** src, float** dst, int count,
                                                     const int* xofs, const float* alpha, int cn, int xmax)
{
    const int step = 4;
    const int len0 = xmax & -step;
    int dx = 0, k = 0;

    for( ; k <= count - 2; k += 2 )
    {
        const T *S0 = src[k], *S1 = src[k + 1];
        float *D0 = dst[k], *D1 = dst[k + 1];
        for( dx = 0; dx < len0; dx += step )
        {
            int sx0 = xofs[dx + 0];
            int sx1 = xofs[dx + 1];
            int sx2 = xofs[dx + 2];
            int sx3 = xofs[dx + 3];
            v_float32x4 a_even, a_odd;
            v_load_deinterleave(alpha + dx * 2, a_even, a_odd);
            v_float32x4 s0((float)S0[sx0], (float)S0[sx1], (float)S0[sx2], (float)S0[sx3]);
            v_float32x4 s1((float)S0[sx0 + cn], (float)S0[sx1 + cn], (float)S0[sx2 + cn], (float)S0[sx3 + cn]);
            v_float32x4 t0((float)S1[sx0], (float)S1[sx1], (float)S1[sx2], (float)S1[sx3]);
            v_float32x4 t1((float)S1[sx0 + cn], (float)S1[sx1 + cn], (float)S1[sx2 + cn], (float)S1[sx3 + cn]);
            v_store(D0 + dx, v_add(v_mul(s0, a_even), v_mul(s1, a_odd)));
            v_store(D1 + dx, v_add(v_mul(t0, a_even), v_mul(t1, a_odd)));
        }
    }
    for( ; k < count; k++ )
    {
        const T *S = src[k];
        float *D = dst[k];
        for( dx = 0; dx < len0; dx += step )
        {
            int sx0 = xofs[dx + 0];
            int sx1 = xofs[dx + 1];
            int sx2 = xofs[dx + 2];
            int sx3 = xofs[dx + 3];
            v_float32x4 a_even, a_odd;
            v_load_deinterleave(alpha + dx * 2, a_even, a_odd);
            v_float32x4 s0((float)S[sx0], (float)S[sx1], (float)S[sx2], (float)S[sx3]);
            v_float32x4 s1((float)S[sx0 + cn], (float)S[sx1 + cn], (float)S[sx2 + cn], (float)S[sx3 + cn]);
            v_store(D + dx, v_add(v_mul(s0, a_even), v_mul(s1, a_odd)));
        }
    }
    return dx;
}

static int hresize_linear_fp_float(const float** src, float** dst, int count,
                                   const int* xofs, const float* alpha, int cn, int xmax)
{
    const int step = 4;
    const int len0 = xmax & -step;
    int dx = 0, k = 0;

    for( ; k <= count - 2; k += 2 )
    {
        const float *S0 = src[k], *S1 = src[k + 1];
        float *D0 = dst[k], *D1 = dst[k + 1];
        for( dx = 0; dx < len0; dx += step )
        {
            int sx0 = xofs[dx + 0];
            int sx1 = xofs[dx + 1];
            int sx2 = xofs[dx + 2];
            int sx3 = xofs[dx + 3];
            v_float32x4 a_even, a_odd;
            v_load_deinterleave(alpha + dx * 2, a_even, a_odd);
            v_float32x4 s0(S0[sx0], S0[sx1], S0[sx2], S0[sx3]);
            v_float32x4 s1(S0[sx0 + cn], S0[sx1 + cn], S0[sx2 + cn], S0[sx3 + cn]);
            v_float32x4 t0(S1[sx0], S1[sx1], S1[sx2], S1[sx3]);
            v_float32x4 t1(S1[sx0 + cn], S1[sx1 + cn], S1[sx2 + cn], S1[sx3 + cn]);
            v_store(D0 + dx, v_add(v_mul(s0, a_even), v_mul(s1, a_odd)));
            v_store(D1 + dx, v_add(v_mul(t0, a_even), v_mul(t1, a_odd)));
        }
    }
    for( ; k < count; k++ )
    {
        const float *S = src[k];
        float *D = dst[k];
        for( dx = 0; dx < len0; dx += step )
        {
            int sx0 = xofs[dx + 0];
            int sx1 = xofs[dx + 1];
            int sx2 = xofs[dx + 2];
            int sx3 = xofs[dx + 3];
            v_float32x4 a_even, a_odd;
            v_load_deinterleave(alpha + dx * 2, a_even, a_odd);
            v_float32x4 s0(S[sx0], S[sx1], S[sx2], S[sx3]);
            v_float32x4 s1(S[sx0 + cn], S[sx1 + cn], S[sx2 + cn], S[sx3 + cn]);
            v_store(D + dx, v_add(v_mul(s0, a_even), v_mul(s1, a_odd)));
        }
    }
    return dx;
}

struct HResizeLinearVecU8
{
    int operator()(const uchar** src, int** dst, int count, const int* xofs,
        const short* alpha/*[xmax]*/, int /*smax*/, int dmax, int cn, int /*xmin*/, int xmax) const
    {
        if( cn == 1 )
            return hresize_linear_u8_cn1(src, dst, count, xofs, alpha, xmax);
        if( cn == 2 )
            return hresize_linear_u8_cn2(src, dst, count, xofs, alpha, xmax);
        if( cn == 3 )
            return hresize_linear_u8_cn3(src, dst, count, xofs, alpha, dmax, cn, xmax);
        if( cn == 4 )
            return hresize_linear_u8_cn4(src, dst, count, xofs, alpha, xmax);
        return 0;
    }
};

typedef HResizeLinearVecU8 HResizeLinearVec_8u32s;

#endif // CV_SIMD

#if CV_SIMD  // fixed-width 128-bit kernels; not valid for scalable backends (RVV)

struct HResizeLinearVec_32f
{
    int operator()(const float** src, float** dst, int count, const int* xofs,
        const float* alpha, int, int, int cn, int, int xmax) const
    {
        return hresize_linear_fp_float(src, dst, count, xofs, alpha, cn, xmax);
    }
};

struct HResizeLinearVec_16u32f
{
    int operator()(const ushort** src, float** dst, int count, const int* xofs,
        const float* alpha, int, int, int cn, int, int xmax) const
    {
        return hresize_linear_fp_16(src, dst, count, xofs, alpha, cn, xmax);
    }
};

struct HResizeLinearVec_16s32f
{
    int operator()(const short** src, float** dst, int count, const int* xofs,
        const float* alpha, int, int, int cn, int, int xmax) const
    {
        return hresize_linear_fp_16(src, dst, count, xofs, alpha, cn, xmax);
    }
};

#else

typedef HResizeNoVec HResizeLinearVec_16u32f;
typedef HResizeNoVec HResizeLinearVec_16s32f;
typedef HResizeNoVec HResizeLinearVec_32f;

#endif

#if !CV_SIMD

typedef HResizeNoVec HResizeLinearVec_8u32s;

#endif

typedef HResizeNoVec HResizeLinearVec_64f;


template<typename T, typename WT, typename AT, int ONE, class VecOp>
struct HResizeLinear
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const T** src, WT** dst, int count,
                    const int* xofs, const AT* alpha,
                    int swidth, int dwidth, int cn, int xmin, int xmax ) const
    {
        int dx, k;
        VecOp vecOp;

        int dx0 = vecOp(src, dst, count,
            xofs, alpha, swidth, dwidth, cn, xmin, xmax );

        for( k = 0; k <= count - 2; k+=2 )
        {
            const T *S0 = src[k], *S1 = src[k+1];
            WT *D0 = dst[k], *D1 = dst[k+1];
            for( dx = dx0; dx < xmax; dx++ )
            {
                int sx = xofs[dx];
                WT a0 = alpha[dx*2], a1 = alpha[dx*2+1];
                WT t0 = S0[sx]*a0 + S0[sx + cn]*a1;
                WT t1 = S1[sx]*a0 + S1[sx + cn]*a1;
                D0[dx] = t0; D1[dx] = t1;
            }

            for( ; dx < dwidth; dx++ )
            {
                int sx = xofs[dx];
                D0[dx] = WT(S0[sx]*ONE); D1[dx] = WT(S1[sx]*ONE);
            }
        }

        for( ; k < count; k++ )
        {
            const T *S = src[k];
            WT *D = dst[k];
            for( dx = dx0; dx < xmax; dx++ )
            {
                int sx = xofs[dx];
                D[dx] = S[sx]*alpha[dx*2] + S[sx+cn]*alpha[dx*2+1];
            }

            for( ; dx < dwidth; dx++ )
                D[dx] = WT(S[xofs[dx]]*ONE);
        }
    }
};


template<typename T, typename WT, typename AT, class CastOp, class VecOp>
struct VResizeLinear
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const WT** src, T* dst, const AT* beta, int width ) const
    {
        WT b0 = beta[0], b1 = beta[1];
        const WT *S0 = src[0], *S1 = src[1];
        CastOp castOp;
        VecOp vecOp;

        int x = vecOp(src, dst, beta, width);
        #if CV_ENABLE_UNROLLED
        for( ; x <= width - 4; x += 4 )
        {
            WT t0, t1;
            t0 = S0[x]*b0 + S1[x]*b1;
            t1 = S0[x+1]*b0 + S1[x+1]*b1;
            dst[x] = castOp(t0); dst[x+1] = castOp(t1);
            t0 = S0[x+2]*b0 + S1[x+2]*b1;
            t1 = S0[x+3]*b0 + S1[x+3]*b1;
            dst[x+2] = castOp(t0); dst[x+3] = castOp(t1);
        }
        #endif
        for( ; x < width; x++ )
            dst[x] = castOp(S0[x]*b0 + S1[x]*b1);
    }
};

template<>
struct VResizeLinear<uchar, int, short, FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS*2>, VResizeLinearVec_32s8u>
{
    typedef uchar value_type;
    typedef int buf_type;
    typedef short alpha_type;

    void operator()(const buf_type** src, value_type* dst, const alpha_type* beta, int width ) const
    {
        alpha_type b0 = beta[0], b1 = beta[1];
        const buf_type *S0 = src[0], *S1 = src[1];
        VResizeLinearVec_32s8u vecOp;

        int x = vecOp(src, dst, beta, width);
        #if CV_ENABLE_UNROLLED
        for( ; x <= width - 4; x += 4 )
        {
            dst[x+0] = uchar(( ((b0 * (S0[x+0] >> 4)) >> 16) + ((b1 * (S1[x+0] >> 4)) >> 16) + 2)>>2);
            dst[x+1] = uchar(( ((b0 * (S0[x+1] >> 4)) >> 16) + ((b1 * (S1[x+1] >> 4)) >> 16) + 2)>>2);
            dst[x+2] = uchar(( ((b0 * (S0[x+2] >> 4)) >> 16) + ((b1 * (S1[x+2] >> 4)) >> 16) + 2)>>2);
            dst[x+3] = uchar(( ((b0 * (S0[x+3] >> 4)) >> 16) + ((b1 * (S1[x+3] >> 4)) >> 16) + 2)>>2);
        }
        #endif
        for( ; x < width; x++ )
            dst[x] = uchar(( ((b0 * (S0[x] >> 4)) >> 16) + ((b1 * (S1[x] >> 4)) >> 16) + 2)>>2);
    }
};


template<typename T, typename WT, typename AT>
struct HResizeCubic
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const T** src, WT** dst, int count,
                    const int* xofs, const AT* alpha,
                    int swidth, int dwidth, int cn, int xmin, int xmax ) const
    {
        for( int k = 0; k < count; k++ )
        {
            const T *S = src[k];
            WT *D = dst[k];
            int dx = 0, limit = xmin;
            for(;;)
            {
                for( ; dx < limit; dx++, alpha += 4 )
                {
                    int j, sx = xofs[dx] - cn;
                    WT v = 0;
                    for( j = 0; j < 4; j++ )
                    {
                        int sxj = sx + j*cn;
                        if( (unsigned)sxj >= (unsigned)swidth )
                        {
                            while( sxj < 0 )
                                sxj += cn;
                            while( sxj >= swidth )
                                sxj -= cn;
                        }
                        v += S[sxj]*alpha[j];
                    }
                    D[dx] = v;
                }
                if( limit == dwidth )
                    break;
                for( ; dx < xmax; dx++, alpha += 4 )
                {
                    int sx = xofs[dx];
                    D[dx] = S[sx-cn]*alpha[0] + S[sx]*alpha[1] +
                        S[sx+cn]*alpha[2] + S[sx+cn*2]*alpha[3];
                }
                limit = dwidth;
            }
            alpha -= dwidth*4;
        }
    }
};


template<typename T, typename WT, typename AT, class CastOp, class VecOp>
struct VResizeCubic
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const WT** src, T* dst, const AT* beta, int width ) const
    {
        WT b0 = beta[0], b1 = beta[1], b2 = beta[2], b3 = beta[3];
        const WT *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        CastOp castOp;
        VecOp vecOp;

        int x = vecOp(src, dst, beta, width);
        for( ; x < width; x++ )
            dst[x] = castOp(S0[x]*b0 + S1[x]*b1 + S2[x]*b2 + S3[x]*b3);
    }
};


template<typename T, typename WT, typename AT>
struct HResizeLanczos4
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const T** src, WT** dst, int count,
                    const int* xofs, const AT* alpha,
                    int swidth, int dwidth, int cn, int xmin, int xmax ) const
    {
        for( int k = 0; k < count; k++ )
        {
            const T *S = src[k];
            WT *D = dst[k];
            int dx = 0, limit = xmin;
            for(;;)
            {
                for( ; dx < limit; dx++, alpha += 8 )
                {
                    int j, sx = xofs[dx] - cn*3;
                    WT v = 0;
                    for( j = 0; j < 8; j++ )
                    {
                        int sxj = sx + j*cn;
                        if( (unsigned)sxj >= (unsigned)swidth )
                        {
                            while( sxj < 0 )
                                sxj += cn;
                            while( sxj >= swidth )
                                sxj -= cn;
                        }
                        v += S[sxj]*alpha[j];
                    }
                    D[dx] = v;
                }
                if( limit == dwidth )
                    break;
                for( ; dx < xmax; dx++, alpha += 8 )
                {
                    int sx = xofs[dx];
                    D[dx] = S[sx-cn*3]*alpha[0] + S[sx-cn*2]*alpha[1] +
                        S[sx-cn]*alpha[2] + S[sx]*alpha[3] +
                        S[sx+cn]*alpha[4] + S[sx+cn*2]*alpha[5] +
                        S[sx+cn*3]*alpha[6] + S[sx+cn*4]*alpha[7];
                }
                limit = dwidth;
            }
            alpha -= dwidth*8;
        }
    }
};


template<typename T, typename WT, typename AT, class CastOp, class VecOp>
struct VResizeLanczos4
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const WT** src, T* dst, const AT* beta, int width ) const
    {
        CastOp castOp;
        VecOp vecOp;
        int x = vecOp(src, dst, beta, width);
        #if CV_ENABLE_UNROLLED
        for( ; x <= width - 4; x += 4 )
        {
            WT b = beta[0];
            const WT* S = src[0];
            WT s0 = S[x]*b, s1 = S[x+1]*b, s2 = S[x+2]*b, s3 = S[x+3]*b;

            for( int k = 1; k < 8; k++ )
            {
                b = beta[k]; S = src[k];
                s0 += S[x]*b; s1 += S[x+1]*b;
                s2 += S[x+2]*b; s3 += S[x+3]*b;
            }

            dst[x] = castOp(s0); dst[x+1] = castOp(s1);
            dst[x+2] = castOp(s2); dst[x+3] = castOp(s3);
        }
        #endif
        for( ; x < width; x++ )
        {
            dst[x] = castOp(src[0][x]*beta[0] + src[1][x]*beta[1] +
                src[2][x]*beta[2] + src[3][x]*beta[3] + src[4][x]*beta[4] +
                src[5][x]*beta[5] + src[6][x]*beta[6] + src[7][x]*beta[7]);
        }
    }
};


static inline int clip(int x, int a, int b)
{
    return x >= a ? (x < b ? x : b-1) : a;
}

static const int MAX_ESIZE=16;

template <typename HResize, typename VResize>
class resizeGeneric_Invoker :
    public ParallelLoopBody
{
public:
    typedef typename HResize::value_type T;
    typedef typename HResize::buf_type WT;
    typedef typename HResize::alpha_type AT;

    resizeGeneric_Invoker(const Mat& _src, Mat &_dst, const int *_xofs, const int *_yofs,
        const AT* _alpha, const AT* __beta, const Size& _ssize, const Size &_dsize,
        int _ksize, int _xmin, int _xmax) :
        ParallelLoopBody(), src(_src), dst(_dst), xofs(_xofs), yofs(_yofs),
        alpha(_alpha), _beta(__beta), ssize(_ssize), dsize(_dsize),
        ksize(_ksize), xmin(_xmin), xmax(_xmax)
    {
        CV_Assert(ksize <= MAX_ESIZE);
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        int dy, cn = src.channels();
        HResize hresize;
        VResize vresize;

        int bufstep = (int)alignSize(dsize.width, 16);
        AutoBuffer<WT> _buffer(bufstep*ksize);
        const T* srows[MAX_ESIZE]={0};
        WT* rows[MAX_ESIZE]={0};
        int prev_sy[MAX_ESIZE];

        for(int k = 0; k < ksize; k++ )
        {
            prev_sy[k] = -1;
            rows[k] = _buffer.data() + bufstep*k;
        }

        const AT* beta = _beta + ksize * range.start;

        for( dy = range.start; dy < range.end; dy++, beta += ksize )
        {
            int sy0 = yofs[dy], k0=ksize, k1=0, ksize2 = ksize/2;

            for(int k = 0; k < ksize; k++ )
            {
                int sy = clip(sy0 - ksize2 + 1 + k, 0, ssize.height);
                for( k1 = std::max(k1, k); k1 < ksize; k1++ )
                {
                    if( k1 < MAX_ESIZE && sy == prev_sy[k1] ) // if the sy-th row has been computed already, reuse it.
                    {
                        if( k1 > k )
                            memcpy( rows[k], rows[k1], bufstep*sizeof(rows[0][0]) );
                        break;
                    }
                }
                if( k1 == ksize )
                    k0 = std::min(k0, k); // remember the first row that needs to be computed
                srows[k] = src.template ptr<T>(sy);
                prev_sy[k] = sy;
            }

            if( k0 < ksize )
                hresize( (const T**)(srows + k0), (WT**)(rows + k0), ksize - k0, xofs, (const AT*)(alpha),
                        ssize.width, dsize.width, cn, xmin, xmax );
            vresize( (const WT**)rows, (T*)(dst.data + dst.step*dy), beta, dsize.width );
        }
    }

private:
    Mat src;
    Mat dst;
    const int* xofs, *yofs;
    const AT* alpha, *_beta;
    Size ssize, dsize;
    const int ksize, xmin, xmax;

    resizeGeneric_Invoker& operator = (const resizeGeneric_Invoker&);
};

template<class HResize, class VResize>
static void resizeGeneric_( const Mat& src, Mat& dst,
                            const int* xofs, const void* _alpha,
                            const int* yofs, const void* _beta,
                            int xmin, int xmax, int ksize )
{
    typedef typename HResize::alpha_type AT;

    const AT* beta = (const AT*)_beta;
    Size ssize = src.size(), dsize = dst.size();
    int cn = src.channels();
    ssize.width *= cn;
    dsize.width *= cn;
    xmin *= cn;
    xmax *= cn;
    // image resize is a separable operation. In case of not too strong

    Range range(0, dsize.height);
    resizeGeneric_Invoker<HResize, VResize> invoker(src, dst, xofs, yofs, (const AT*)_alpha, beta,
        ssize, dsize, ksize, xmin, xmax);
    parallel_for_(range, invoker, dst.total()/(double)(1<<16));
}

template <typename T, typename WT>
struct ResizeAreaFastNoVec
{
    ResizeAreaFastNoVec(int, int) { }
    ResizeAreaFastNoVec(int, int, int, int) { }
    int operator() (const T*, T*, int) const
    { return 0; }
};

namespace area_fast_detail {

template<typename T> struct area_fast_has_nxn { enum { value = 1 }; };
template<> struct area_fast_has_nxn<float> { enum { value = 0 }; };
template<> struct area_fast_has_nxn<double> { enum { value = 0 }; };

template<typename T> struct area_fast_wt { typedef int type; };
template<> struct area_fast_wt<float> { typedef float type; };
template<> struct area_fast_wt<double> { typedef double type; };

template<typename T> struct area_fast_max_scale { enum { value = 4 }; };
// area_fast_u8_nxn keeps the column sums of up to 257 rows of 8U in 16 bits.
template<> struct area_fast_max_scale<uchar> { enum { value = 257 }; };

template<typename T>
static inline bool is_fast_scale(int scale_x, int scale_y, int cn)
{
    if (scale_x != scale_y || scale_x < 2 || scale_x > area_fast_max_scale<T>::value)
        return false;
    if (cn != 1 && cn != 3 && cn != 4)
        return false;
    return scale_x == 2 || area_fast_has_nxn<T>::value != 0;
}

template<typename T, typename WT, bool is_integral = std::is_integral<WT>::value>
struct area_fast_round_helper;

template<typename T, typename WT>
struct area_fast_round_helper<T, WT, true>
{
    static inline T round(WT sum, int area)
    {
        if (area == 4)
            return saturate_cast<T>((sum + 2) >> 2);
        return saturate_cast<T>(sum * (1.f / area));
    }
};

template<typename T, typename WT>
struct area_fast_round_helper<T, WT, false>
{
    static inline T round(WT sum, int area)
    {
        return saturate_cast<T>(sum * (1.f / area));
    }
};

template<typename T, typename WT>
static inline T area_fast_round(WT sum, int area)
{
    return area_fast_round_helper<T, WT>::round(sum, area);
}

template<typename T>
static int area_fast_tail_nxn_cn1(const T* S, T* D, int w, int dx, int scale, int step)
{
    typedef typename area_fast_wt<T>::type WT;
    const int area = scale * scale;

    for (; dx < w; ++dx)
    {
        const int sx = dx * scale;
        WT sum = 0;
        for (int ry = 0; ry < scale; ++ry)
            for (int k = 0; k < scale; ++k)
                sum += ((const T*)((const uchar*)S + (size_t)ry * step))[sx + k];
        D[dx] = area_fast_round<T>(sum, area);
    }
    return dx;
}

template<typename T>
static int area_fast_tail_nxn_cn3(const T* S, T* D, int w, int dx, int scale, int step)
{
    typedef typename area_fast_wt<T>::type WT;
    const int area = scale * scale;

    for (; dx < w; dx += 3)
    {
        const int sx = dx * scale;
        for (int c = 0; c < 3; ++c)
        {
            WT sum = 0;
            for (int ry = 0; ry < scale; ++ry)
                for (int k = 0; k < scale; ++k)
                    sum += ((const T*)((const uchar*)S + (size_t)ry * step))[sx + k * 3 + c];
            D[dx + c] = area_fast_round<T>(sum, area);
        }
    }
    return dx;
}

template<typename T>
static int area_fast_tail_nxn_cn4(const T* S, T* D, int w, int dx, int scale, int step)
{
    typedef typename area_fast_wt<T>::type WT;
    const int area = scale * scale;

    for (; dx < w; dx += 4)
    {
        const int sx = dx * scale;
        for (int c = 0; c < 4; ++c)
        {
            WT sum = 0;
            for (int ry = 0; ry < scale; ++ry)
                for (int k = 0; k < scale; ++k)
                    sum += ((const T*)((const uchar*)S + (size_t)ry * step))[sx + k * 4 + c];
            D[dx + c] = area_fast_round<T>(sum, area);
        }
    }
    return dx;
}

static int area_fast_tail_2x2(const float* S, float* D, int w, int dx, int cn, int step)
{
    const float* nextS = (const float*)((const uchar*)S + step);
    const float q = 0.25f;
    if (cn == 1)
        for (; dx < w; ++dx)
        {
            const int index = dx * 2;
            D[dx] = q * (S[index] + S[index + 1] + nextS[index] + nextS[index + 1]);
        }
    else if (cn == 3)
        for (; dx < w; dx += 3)
        {
            const int index = dx * 2;
            D[dx] = q * (S[index] + S[index + 3] + nextS[index] + nextS[index + 3]);
            D[dx + 1] = q * (S[index + 1] + S[index + 4] + nextS[index + 1] + nextS[index + 4]);
            D[dx + 2] = q * (S[index + 2] + S[index + 5] + nextS[index + 2] + nextS[index + 5]);
        }
    else
    {
        CV_Assert(cn == 4);
        for (; dx < w; dx += 4)
        {
            const int index = dx * 2;
            D[dx] = q * (S[index] + S[index + 4] + nextS[index] + nextS[index + 4]);
            D[dx + 1] = q * (S[index + 1] + S[index + 5] + nextS[index + 1] + nextS[index + 5]);
            D[dx + 2] = q * (S[index + 2] + S[index + 6] + nextS[index + 2] + nextS[index + 6]);
            D[dx + 3] = q * (S[index + 3] + S[index + 7] + nextS[index + 3] + nextS[index + 7]);
        }
    }
    return dx;
}

static inline int area_fast_f32_cn3_2x2(const float* S0, const float* S1, float* D, int w, int dx)
{
    const float q = 0.25f;
    for (; dx <= w - 12; dx += 12, S0 += 24, S1 += 24, D += 12)
    {
        D[0] = q * (S0[0] + S0[3] + S1[0] + S1[3]);
        D[1] = q * (S0[1] + S0[4] + S1[1] + S1[4]);
        D[2] = q * (S0[2] + S0[5] + S1[2] + S1[5]);
        D[3] = q * (S0[6] + S0[9] + S1[6] + S1[9]);
        D[4] = q * (S0[7] + S0[10] + S1[7] + S1[10]);
        D[5] = q * (S0[8] + S0[11] + S1[8] + S1[11]);
        D[6] = q * (S0[12] + S0[15] + S1[12] + S1[15]);
        D[7] = q * (S0[13] + S0[16] + S1[13] + S1[16]);
        D[8] = q * (S0[14] + S0[17] + S1[14] + S1[17]);
        D[9] = q * (S0[18] + S0[21] + S1[18] + S1[21]);
        D[10] = q * (S0[19] + S0[22] + S1[19] + S1[22]);
        D[11] = q * (S0[20] + S0[23] + S1[20] + S1[23]);
    }
    for (; dx < w; dx += 3, S0 += 6, S1 += 6, D += 3)
    {
        D[0] = q * (S0[0] + S0[3] + S1[0] + S1[3]);
        D[1] = q * (S0[1] + S0[4] + S1[1] + S1[4]);
        D[2] = q * (S0[2] + S0[5] + S1[2] + S1[5]);
    }
    return dx;
}

template<typename T>
static int area_fast_tail_2x2(const T* S, T* D, int w, int dx, int cn, int step)
{
    typedef typename area_fast_wt<T>::type WT;
    const int area = 4;
    const T* nextS = (const T*)((const uchar*)S + step);
    if (cn == 1)
        for (; dx < w; ++dx)
        {
            const int index = dx * 2;
            WT sum = S[index] + S[index + 1] + nextS[index] + nextS[index + 1];
            D[dx] = area_fast_round<T>(sum, area);
        }
    else if (cn == 3)
        for (; dx < w; dx += 3)
        {
            const int index = dx * 2;
            D[dx] = area_fast_round<T>(S[index] + S[index + 3] + nextS[index] + nextS[index + 3], area);
            D[dx + 1] = area_fast_round<T>(S[index + 1] + S[index + 4] + nextS[index + 1] + nextS[index + 4], area);
            D[dx + 2] = area_fast_round<T>(S[index + 2] + S[index + 5] + nextS[index + 2] + nextS[index + 5], area);
        }
    else
    {
        CV_Assert(cn == 4);
        for (; dx < w; dx += 4)
        {
            const int index = dx * 2;
            D[dx] = area_fast_round<T>(S[index] + S[index + 4] + nextS[index] + nextS[index + 4], area);
            D[dx + 1] = area_fast_round<T>(S[index + 1] + S[index + 5] + nextS[index + 1] + nextS[index + 5], area);
            D[dx + 2] = area_fast_round<T>(S[index + 2] + S[index + 6] + nextS[index + 2] + nextS[index + 6], area);
            D[dx + 3] = area_fast_round<T>(S[index + 3] + S[index + 7] + nextS[index + 3] + nextS[index + 7], area);
        }
    }
    return dx;
}

template<typename T>
static int area_fast_tail_nxn(const T* S, T* D, int w, int dx, int scale, int cn, int step)
{
    if (cn == 1)
        return area_fast_tail_nxn_cn1(S, D, w, dx, scale, step);
    if (cn == 3)
        return area_fast_tail_nxn_cn3(S, D, w, dx, scale, step);
    return area_fast_tail_nxn_cn4(S, D, w, dx, scale, step);
}

#if (CV_SIMD || CV_SIMD_SCALABLE)
// Any integer scale: sum the scale source rows column-wise into vsum, then add scale
// neighbouring columns per channel. vsum must hold w*scale elements, scale <= 257.
static int area_fast_u8_nxn(const uchar* S, uchar* D, int w, int step, int scale, int cn,
                            ushort* vsum)
{
    const int sw = w * scale;
    const int n = VTraits<v_uint8>::vlanes(), h = VTraits<v_uint16>::vlanes();
    int i = 0;
    for (; i <= sw - n; i += n)
    {
        v_uint16 lo = vx_setzero_u16(), hi = vx_setzero_u16();
        const uchar* p = S + i;
        for (int r = 0; r < scale; ++r, p += step)
        {
            v_uint16 l, u;
            v_expand(vx_load(p), l, u);
            lo = v_add(lo, l);
            hi = v_add(hi, u);
        }
        v_store(vsum + i, lo);
        v_store(vsum + i + h, hi);
    }
    for (; i < sw; ++i)
    {
        int s = 0;
        const uchar* p = S + i;
        for (int r = 0; r < scale; ++r, p += step)
            s += *p;
        vsum[i] = (ushort)s;
    }

    const int area = scale * scale;
    const int span = scale * cn;
    for (int dx = 0; dx < w; dx += cn)
    {
        const ushort* q = vsum + (dx / cn) * span;
        for (int c = 0; c < cn; ++c)
        {
            int s = 0;
            for (int k = c; k < span; k += cn)
                s += q[k];
            D[dx + c] = area_fast_round<uchar>(s, area);
        }
    }
    return w;
}
#endif

#if CV_SIMD
static inline v_uint32 area_fast_byte_sum_u32(v_uint32 v)
{
    const v_uint32 mask = vx_setall_u32(0xff);
    return v_add(v_add(v_shr<24>(v), v_and(v_shr<16>(v), mask)),
                 v_add(v_and(v_shr<8>(v), mask), v_and(v, mask)));
}

static inline int area_fast_4x4_pix_block()
{
    return VTraits<v_uint32>::vlanes();
}

static inline v_uint32 area_fast_plane_row_quads(const v_uint8& plane)
{
    return area_fast_byte_sum_u32(v_reinterpret_as_u32(plane));
}

// Matches area_fast_round<uchar>(sum, 16), i.e. cvRound(sum / 16.f) with ties to even.
static inline v_uint32 area_fast_round_div16(const v_uint32& sum)
{
    const v_uint32 odd = v_and(v_shr<4>(sum), vx_setall_u32(1));
    return v_shr<4>(v_add(v_add(sum, vx_setall_u32(7)), odd));
}

static inline void area_fast_store_bgra(uchar* dst, v_uint32 sb, v_uint32 sg, v_uint32 sr, v_uint32 sa)
{
    v_uint32 px = v_add(v_add(area_fast_round_div16(sb),
                              v_shl<8>(area_fast_round_div16(sg))),
                        v_add(v_shl<16>(area_fast_round_div16(sr)),
                              v_shl<24>(area_fast_round_div16(sa))));
    v_store(reinterpret_cast<uint32_t*>(dst), px);
}

static inline void area_fast_store_rgb_rounded(uchar* dst, const v_uint32& vb, const v_uint32& vg, const v_uint32& vr, int n)
{
    CV_DECL_ALIGNED(64) uint32_t b[VTraits<v_uint32>::max_nlanes], g[VTraits<v_uint32>::max_nlanes], r[VTraits<v_uint32>::max_nlanes];
    v_store(b, vb);
    v_store(g, vg);
    v_store(r, vr);
    uchar* p = dst;
    for (int i = 0; i < n; ++i)
    {
        p[0] = (uchar)b[i];
        p[1] = (uchar)g[i];
        p[2] = (uchar)r[i];
        p += 3;
    }
}

static inline void area_fast_store_rgb(uchar* dst, v_uint32 sb, v_uint32 sg, v_uint32 sr)
{
    area_fast_store_rgb_rounded(dst, area_fast_round_div16(sb), area_fast_round_div16(sg), area_fast_round_div16(sr),
                                VTraits<v_uint32>::vlanes());
}

static int area_fast_u8_cn4_4x4(const uchar* S, uchar* D, int w, int step, int dx)
{
    const uchar* rows[4] = { S, S + step, S + 2 * step, S + 3 * step };
    const int pixBlock = area_fast_4x4_pix_block();
    const int chBlock = pixBlock * 4;

    for (; dx <= w - chBlock; dx += chBlock)
    {
        const int sx = dx * 4;
        v_uint32 sb = vx_setzero_u32(), sg = sb, sr = sb, sa = sb;

        for (int ri = 0; ri < 4; ++ri)
        {
            v_uint8 b, g, r, a;
            v_load_deinterleave(rows[ri] + sx, b, g, r, a);
            sb = v_add(sb, area_fast_plane_row_quads(b));
            sg = v_add(sg, area_fast_plane_row_quads(g));
            sr = v_add(sr, area_fast_plane_row_quads(r));
            sa = v_add(sa, area_fast_plane_row_quads(a));
        }

        area_fast_store_bgra(D + dx, sb, sg, sr, sa);
    }
    return dx;
}

static int area_fast_u8_cn3_4x4(const uchar* S, uchar* D, int w, int step, int dx)
{
    const uchar* rows[4] = { S, S + step, S + 2 * step, S + 3 * step };
    const int pixBlock = area_fast_4x4_pix_block();
    const int chBlock = pixBlock * 3;

    for (; dx <= w - chBlock; dx += chBlock)
    {
        const int sx = dx * 4;
        v_uint32 sb = vx_setzero_u32(), sg = sb, sr = sb;

        for (int ri = 0; ri < 4; ++ri)
        {
            v_uint8 b, g, r;
            v_load_deinterleave(rows[ri] + sx, b, g, r);
            sb = v_add(sb, area_fast_plane_row_quads(b));
            sg = v_add(sg, area_fast_plane_row_quads(g));
            sr = v_add(sr, area_fast_plane_row_quads(r));
        }

        area_fast_store_rgb(D + dx, sb, sg, sr);
    }
    return dx;
}

static int area_fast_u8_cn1_4x4(const uchar* S, uchar* D, int w, int step, int dx)
{
    const uchar* rows[4] = { S, S + step, S + 2 * step, S + 3 * step };
    const int block = VTraits<v_uint16>::vlanes();
    const v_uint16 mask8 = vx_setall_u16(0xff);
    const v_uint32 mask16 = vx_setall_u32(0xffff);

    for (; dx <= w - block; dx += block)
    {
        const int sx = dx * 4;
        // Byte-pair sums of 4 rows fit in u16 (<= 4 * 510).
        v_uint16 p0 = vx_setzero_u16(), p1 = p0;
        for (int ri = 0; ri < 4; ++ri)
        {
            v_uint16 r0 = v_reinterpret_as_u16(vx_load(rows[ri] + sx));
            v_uint16 r1 = v_reinterpret_as_u16(vx_load(rows[ri] + sx + VTraits<v_uint8>::vlanes()));
            p0 = v_add(p0, v_add(v_shr<8>(r0), v_and(r0, mask8)));
            p1 = v_add(p1, v_add(v_shr<8>(r1), v_and(r1, mask8)));
        }
        v_uint32 q0 = v_reinterpret_as_u32(p0), q1 = v_reinterpret_as_u32(p1);
        v_uint32 s0 = v_add(v_shr<16>(q0), v_and(q0, mask16));
        v_uint32 s1 = v_add(v_shr<16>(q1), v_and(q1, mask16));
        v_pack_store(D + dx, v_pack(area_fast_round_div16(s0), area_fast_round_div16(s1)));
    }
    return dx;
}

#endif // CV_SIMD

} // namespace area_fast_detail

#if CV_NEON

class ResizeAreaFastVec_SIMD_8u
{
public:
    ResizeAreaFastVec_SIMD_8u(int _scale_x, int _scale_y, int _cn, int _step) :
        scale_x(_scale_x), scale_y(_scale_y), cn(_cn), step(_step)
    {
    }

    int operator() (const uchar* S, uchar* D, int w) const
    {
        if (scale_x != scale_y || scale_x < 2)
            return 0;
        if (scale_x == 3 || scale_x > 4)
            return area_fast_nxn(S, D, w);

        int dx = 0;
        const uchar* S0 = S, * S1 = S0 + step;

        if (scale_x == 4 && scale_y == 4)
        {
            if (cn == 1)
                return area_fast_detail::area_fast_u8_cn1_4x4(S, D, w, step, dx);
            if (cn == 3)
                return area_fast_detail::area_fast_u8_cn3_4x4(S, D, w, step, dx);
            if (cn == 4)
                return area_fast_detail::area_fast_u8_cn4_4x4(S, D, w, step, dx);
        }

        if (scale_x != 2 || scale_y != 2)
            return 0;

        const uint16x8_t v_2 = vdupq_n_u16(2);

        if (cn == 1)
        {
            for ( ; dx <= w - 16; dx += 16, S0 += 32, S1 += 32, D += 16)
            {
                uint8x16x2_t v_row0 = vld2q_u8(S0), v_row1 = vld2q_u8(S1);

                uint16x8_t v_dst0 = vaddl_u8(vget_low_u8(v_row0.val[0]), vget_low_u8(v_row0.val[1]));
                v_dst0 = vaddq_u16(v_dst0, vaddl_u8(vget_low_u8(v_row1.val[0]), vget_low_u8(v_row1.val[1])));
                v_dst0 = vshrq_n_u16(vaddq_u16(v_dst0, v_2), 2);

                uint16x8_t v_dst1 = vaddl_u8(vget_high_u8(v_row0.val[0]), vget_high_u8(v_row0.val[1]));
                v_dst1 = vaddq_u16(v_dst1, vaddl_u8(vget_high_u8(v_row1.val[0]), vget_high_u8(v_row1.val[1])));
                v_dst1 = vshrq_n_u16(vaddq_u16(v_dst1, v_2), 2);

                vst1q_u8(D, vcombine_u8(vmovn_u16(v_dst0), vmovn_u16(v_dst1)));
            }
        }
        else if (cn == 4)
        {
            for ( ; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                uint8x16_t v_row0 = vld1q_u8(S0), v_row1 = vld1q_u8(S1);

                uint16x8_t v_row00 = vmovl_u8(vget_low_u8(v_row0));
                uint16x8_t v_row01 = vmovl_u8(vget_high_u8(v_row0));
                uint16x8_t v_row10 = vmovl_u8(vget_low_u8(v_row1));
                uint16x8_t v_row11 = vmovl_u8(vget_high_u8(v_row1));

                uint16x4_t v_p0 = vadd_u16(vadd_u16(vget_low_u16(v_row00), vget_high_u16(v_row00)),
                                           vadd_u16(vget_low_u16(v_row10), vget_high_u16(v_row10)));
                uint16x4_t v_p1 = vadd_u16(vadd_u16(vget_low_u16(v_row01), vget_high_u16(v_row01)),
                                           vadd_u16(vget_low_u16(v_row11), vget_high_u16(v_row11)));
                uint16x8_t v_dst = vshrq_n_u16(vaddq_u16(vcombine_u16(v_p0, v_p1), v_2), 2);

                vst1_u8(D, vmovn_u16(v_dst));
            }
        }

        return dx;
    }

private:
    int area_fast_nxn(const uchar* S, uchar* D, int w) const
    {
        if (scale_x > area_fast_detail::area_fast_max_scale<uchar>::value ||
            (cn != 1 && cn != 3 && cn != 4))
            return 0;
        vsum.resize((size_t)w * scale_x);
        return area_fast_detail::area_fast_u8_nxn(S, D, w, step, scale_x, cn, vsum.data());
    }

    int scale_x, scale_y;
    int cn, step;
    mutable std::vector<ushort> vsum;
};

class ResizeAreaFastVec_SIMD_16u
{
public:
    ResizeAreaFastVec_SIMD_16u(int _scale_x, int _scale_y, int _cn, int _step) :
        scale_x(_scale_x), scale_y(_scale_y), cn(_cn), step(_step)
    {
    }

    int operator() (const ushort * S, ushort * D, int w) const
    {
        if (scale_x != 2 || scale_y != 2)
            return 0;

        int dx = 0;
        const ushort * S0 = S, * S1 = (const ushort *)((const uchar *)(S0) + step);
        const uint32x4_t v_2 = vdupq_n_u32(2);

        if (cn == 1)
        {
            for ( ; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                uint16x8x2_t v_row0 = vld2q_u16(S0), v_row1 = vld2q_u16(S1);

                uint32x4_t v_sum0 = vaddl_u16(vget_low_u16(v_row0.val[0]), vget_low_u16(v_row0.val[1]));
                v_sum0 = vaddq_u32(v_sum0, vaddl_u16(vget_low_u16(v_row1.val[0]), vget_low_u16(v_row1.val[1])));
                v_sum0 = vshrq_n_u32(vaddq_u32(v_sum0, v_2), 2);

                uint32x4_t v_sum1 = vaddl_u16(vget_high_u16(v_row0.val[0]), vget_high_u16(v_row0.val[1]));
                v_sum1 = vaddq_u32(v_sum1, vaddl_u16(vget_high_u16(v_row1.val[0]), vget_high_u16(v_row1.val[1])));
                v_sum1 = vshrq_n_u32(vaddq_u32(v_sum1, v_2), 2);

                vst1q_u16(D, vcombine_u16(vmovn_u32(v_sum0), vmovn_u32(v_sum1)));
            }
        }
        else if (cn == 4)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                uint16x8_t v_row0 = vld1q_u16(S0), v_row1 = vld1q_u16(S1);
                uint32x4_t v_sum = vaddq_u32(vaddl_u16(vget_low_u16(v_row0), vget_high_u16(v_row0)),
                                             vaddl_u16(vget_low_u16(v_row1), vget_high_u16(v_row1)));
                v_sum = vshrq_n_u32(vaddq_u32(v_sum, v_2), 2);
                vst1_u16(D, vmovn_u32(v_sum));
            }
        }

        return dx;
    }

private:
    int scale_x, scale_y;
    int cn, step;
};

class ResizeAreaFastVec_SIMD_16s
{
public:
    ResizeAreaFastVec_SIMD_16s(int _scale_x, int _scale_y, int _cn, int _step) :
        scale_x(_scale_x), scale_y(_scale_y), cn(_cn), step(_step)
    {
    }

    int operator() (const short * S, short * D, int w) const
    {
        if (scale_x != 2 || scale_y != 2)
            return 0;

        int dx = 0;
        const short * S0 = S, * S1 = (const short *)((const uchar *)(S0) + step);
        const int32x4_t v_2 = vdupq_n_s32(2);

        if (cn == 1)
        {
            for ( ; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                int16x8x2_t v_row0 = vld2q_s16(S0), v_row1 = vld2q_s16(S1);

                int32x4_t v_sum0 = vaddl_s16(vget_low_s16(v_row0.val[0]), vget_low_s16(v_row0.val[1]));
                v_sum0 = vaddq_s32(v_sum0, vaddl_s16(vget_low_s16(v_row1.val[0]), vget_low_s16(v_row1.val[1])));
                v_sum0 = vshrq_n_s32(vaddq_s32(v_sum0, v_2), 2);

                int32x4_t v_sum1 = vaddl_s16(vget_high_s16(v_row0.val[0]), vget_high_s16(v_row0.val[1]));
                v_sum1 = vaddq_s32(v_sum1, vaddl_s16(vget_high_s16(v_row1.val[0]), vget_high_s16(v_row1.val[1])));
                v_sum1 = vshrq_n_s32(vaddq_s32(v_sum1, v_2), 2);

                vst1q_s16(D, vcombine_s16(vmovn_s32(v_sum0), vmovn_s32(v_sum1)));
            }
        }
        else if (cn == 4)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                int16x8_t v_row0 = vld1q_s16(S0), v_row1 = vld1q_s16(S1);
                int32x4_t v_sum = vaddq_s32(vaddl_s16(vget_low_s16(v_row0), vget_high_s16(v_row0)),
                                            vaddl_s16(vget_low_s16(v_row1), vget_high_s16(v_row1)));
                v_sum = vshrq_n_s32(vaddq_s32(v_sum, v_2), 2);
                vst1_s16(D, vmovn_s32(v_sum));
            }
        }

        return dx;
    }

private:
    int scale_x, scale_y;
    int cn, step;
};

struct ResizeAreaFastVec_SIMD_32f
{
    ResizeAreaFastVec_SIMD_32f(int _scale_x, int _scale_y, int _cn, int _step) :
        cn(_cn), step(_step)
    {
        fast_mode = _scale_x == 2 && _scale_y == 2 && (cn == 1 || cn == 3 || cn == 4);
    }

    int operator() (const float * S, float * D, int w) const
    {
        if (!fast_mode)
            return 0;

        const float * S0 = S, * S1 = (const float *)((const uchar *)(S0) + step);
        int dx = 0;

        float32x4_t v_025 = vdupq_n_f32(0.25f);

        if (cn == 1)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                float32x4x2_t v_row0 = vld2q_f32(S0), v_row1 = vld2q_f32(S1);

                float32x4_t v_dst0 = vaddq_f32(v_row0.val[0], v_row0.val[1]);
                float32x4_t v_dst1 = vaddq_f32(v_row1.val[0], v_row1.val[1]);

                vst1q_f32(D, vmulq_f32(vaddq_f32(v_dst0, v_dst1), v_025));
            }
        }
        else if (cn == 3)
        {
            dx = area_fast_detail::area_fast_f32_cn3_2x2(S0, S1, D, w, dx);
        }
        else if (cn == 4)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                float32x4_t v_dst0 = vaddq_f32(vld1q_f32(S0), vld1q_f32(S0 + 4));
                float32x4_t v_dst1 = vaddq_f32(vld1q_f32(S1), vld1q_f32(S1 + 4));

                vst1q_f32(D, vmulq_f32(vaddq_f32(v_dst0, v_dst1), v_025));
            }
        }

        return dx;
    }

private:
    int cn;
    bool fast_mode;
    int step;
};

#elif CV_SIMD

class ResizeAreaFastVec_SIMD_8u
{
public:
    ResizeAreaFastVec_SIMD_8u(int _scale_x, int _scale_y, int _cn, int _step) :
        scale_x(_scale_x), scale_y(_scale_y), cn(_cn), step(_step) {}

    int operator() (const uchar* S, uchar* D, int w) const
    {
        if (scale_x != scale_y || scale_x < 2)
            return 0;
        if (scale_x == 3 || scale_x > 4)
            return area_fast_nxn(S, D, w);

        int dx = 0;

        if (scale_x == 4 && scale_y == 4)
        {
            if (cn == 1)
                return area_fast_detail::area_fast_u8_cn1_4x4(S, D, w, step, dx);
            if (cn == 3)
                return area_fast_detail::area_fast_u8_cn3_4x4(S, D, w, step, dx);
            if (cn == 4)
                return area_fast_detail::area_fast_u8_cn4_4x4(S, D, w, step, dx);
        }

        if (scale_x != 2 || scale_y != 2)
            return 0;

        const uchar* S0 = S;
        const uchar* S1 = S0 + step;

        if (cn == 1)
        {
            v_uint16 masklow = vx_setall_u16(0x00ff);
            for ( ; dx <= w - VTraits<v_uint16>::vlanes(); dx += VTraits<v_uint16>::vlanes(), S0 += VTraits<v_uint8>::vlanes(), S1 += VTraits<v_uint8>::vlanes(), D += VTraits<v_uint16>::vlanes())
            {
                v_uint16 r0 = v_reinterpret_as_u16(vx_load(S0));
                v_uint16 r1 = v_reinterpret_as_u16(vx_load(S1));
                v_rshr_pack_store<2>(D, v_add(v_add(v_add(v_shr<8>(r0), v_and(r0, masklow)), v_shr<8>(r1)), v_and(r1, masklow)));
            }
        }
        else if (cn == 3)
        {
            for ( ; dx <= w - 3*VTraits<v_uint8>::vlanes(); dx += 3*VTraits<v_uint8>::vlanes(), S0 += 6*VTraits<v_uint8>::vlanes(), S1 += 6*VTraits<v_uint8>::vlanes(), D += 3*VTraits<v_uint8>::vlanes())
            {
                v_uint16 t0, t1, t2, t3, t4, t5;
                v_uint16 s0, s1, s2, s3, s4, s5;
                s0 = v_add(vx_load_expand(S0), vx_load_expand(S1));
                s1 = v_add(vx_load_expand(S0 + VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + VTraits<v_uint16>::vlanes()));
                s2 = v_add(vx_load_expand(S0 + 2 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 2 * VTraits<v_uint16>::vlanes()));
                s3 = v_add(vx_load_expand(S0 + 3 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 3 * VTraits<v_uint16>::vlanes()));
                s4 = v_add(vx_load_expand(S0 + 4 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 4 * VTraits<v_uint16>::vlanes()));
                s5 = v_add(vx_load_expand(S0 + 5 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 5 * VTraits<v_uint16>::vlanes()));
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_uint16 bl, gl, rl;
#if CV_SIMD_WIDTH == 16
                bl = v_add(t0, t3); gl = v_add(t1, t4); rl = v_add(t2, t5);
#elif CV_SIMD_WIDTH == 32
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                bl = v_add(s0, s3); gl = v_add(s1, s4); rl = v_add(s2, s5);
#elif CV_SIMD_WIDTH == 64
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                bl = v_add(t0, t3); gl = v_add(t1, t4); rl = v_add(t2, t5);
#endif
                s0 = v_add(vx_load_expand(S0 + 6 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 6 * VTraits<v_uint16>::vlanes()));
                s1 = v_add(vx_load_expand(S0 + 7 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 7 * VTraits<v_uint16>::vlanes()));
                s2 = v_add(vx_load_expand(S0 + 8 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 8 * VTraits<v_uint16>::vlanes()));
                s3 = v_add(vx_load_expand(S0 + 9 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 9 * VTraits<v_uint16>::vlanes()));
                s4 = v_add(vx_load_expand(S0 + 10 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 10 * VTraits<v_uint16>::vlanes()));
                s5 = v_add(vx_load_expand(S0 + 11 * VTraits<v_uint16>::vlanes()), vx_load_expand(S1 + 11 * VTraits<v_uint16>::vlanes()));
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_uint16 bh, gh, rh;
#if CV_SIMD_WIDTH == 16
                bh = v_add(t0, t3); gh = v_add(t1, t4); rh = v_add(t2, t5);
#elif CV_SIMD_WIDTH == 32
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                bh = v_add(s0, s3); gh = v_add(s1, s4); rh = v_add(s2, s5);
#elif CV_SIMD_WIDTH == 64
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                bh = v_add(t0, t3); gh = v_add(t1, t4); rh = v_add(t2, t5);
#endif
                v_store_interleave(D, v_rshr_pack<2>(bl, bh), v_rshr_pack<2>(gl, gh), v_rshr_pack<2>(rl, rh));
            }
        }
        else
        {
            CV_Assert(cn == 4);
            for ( ; dx <= w - VTraits<v_uint8>::vlanes(); dx += VTraits<v_uint8>::vlanes(), S0 += 2*VTraits<v_uint8>::vlanes(), S1 += 2*VTraits<v_uint8>::vlanes(), D += VTraits<v_uint8>::vlanes())
            {
                v_uint32 r00, r01, r10, r11;
                v_load_deinterleave((uint32_t*)S0, r00, r01);
                v_load_deinterleave((uint32_t*)S1, r10, r11);

                v_uint16 r00l, r01l, r10l, r11l, r00h, r01h, r10h, r11h;
                v_expand(v_reinterpret_as_u8(r00), r00l, r00h);
                v_expand(v_reinterpret_as_u8(r01), r01l, r01h);
                v_expand(v_reinterpret_as_u8(r10), r10l, r10h);
                v_expand(v_reinterpret_as_u8(r11), r11l, r11h);
                v_store(D, v_rshr_pack<2>(v_add(v_add(v_add(r00l, r01l), r10l), r11l),
                                          v_add(v_add(v_add(r00h, r01h), r10h), r11h)));
            }
        }

        return dx;
    }

private:
    int area_fast_nxn(const uchar* S, uchar* D, int w) const
    {
        if (scale_x > area_fast_detail::area_fast_max_scale<uchar>::value ||
            (cn != 1 && cn != 3 && cn != 4))
            return 0;
        vsum.resize((size_t)w * scale_x);
        return area_fast_detail::area_fast_u8_nxn(S, D, w, step, scale_x, cn, vsum.data());
    }

    int scale_x, scale_y;
    int cn;
    int step;
    mutable std::vector<ushort> vsum;
};

class ResizeAreaFastVec_SIMD_16u
{
public:
    ResizeAreaFastVec_SIMD_16u(int _scale_x, int _scale_y, int _cn, int _step) :
        scale_x(_scale_x), scale_y(_scale_y), cn(_cn), step(_step) {}

    int operator() (const ushort* S, ushort* D, int w) const
    {
        if (scale_x != 2 || scale_y != 2)
            return 0;

        int dx = 0;
        const ushort* S0 = (const ushort*)S;
        const ushort* S1 = (const ushort*)((const uchar*)(S) + step);

        if (cn == 1)
        {
            v_uint32 masklow = vx_setall_u32(0x0000ffff);
            for (; dx <= w - VTraits<v_uint32>::vlanes(); dx += VTraits<v_uint32>::vlanes(), S0 += VTraits<v_uint16>::vlanes(), S1 += VTraits<v_uint16>::vlanes(), D += VTraits<v_uint32>::vlanes())
            {
                v_uint32 r0 = v_reinterpret_as_u32(vx_load(S0));
                v_uint32 r1 = v_reinterpret_as_u32(vx_load(S1));
                v_rshr_pack_store<2>(D, v_add(v_add(v_add(v_shr<16>(r0), v_and(r0, masklow)), v_shr<16>(r1)), v_and(r1, masklow)));
            }
        }
        else if (cn == 3)
        {
#if CV_SIMD_WIDTH == 16
            for ( ; dx <= w - 4; dx += 3, S0 += 6, S1 += 6, D += 3)
#if CV_SSE4_1
            {
                v_uint32 r0, r1, r2, r3;
                v_expand(vx_load(S0), r0, r1);
                v_expand(vx_load(S1), r2, r3);
                r0 = v_add(r0, r2); r1 = v_add(r1, r3);
                v_rshr_pack_store<2>(D, v_add(r0, v_rotate_left<1>(r1, r0)));
            }
#else
                v_rshr_pack_store<2>(D, v_add(v_add(v_add(v_load_expand(S0), v_load_expand(S0 + 3)), v_load_expand(S1)), v_load_expand(S1 + 3)));
#endif
#elif CV_SIMD_WIDTH == 32 || CV_SIMD_WIDTH == 64
            for ( ; dx <= w - 3*VTraits<v_uint16>::vlanes(); dx += 3*VTraits<v_uint16>::vlanes(), S0 += 6*VTraits<v_uint16>::vlanes(), S1 += 6*VTraits<v_uint16>::vlanes(), D += 3*VTraits<v_uint16>::vlanes())
            {
                v_uint32 t0, t1, t2, t3, t4, t5;
                v_uint32 s0, s1, s2, s3, s4, s5;
                s0 = v_add(vx_load_expand(S0), vx_load_expand(S1));
                s1 = v_add(vx_load_expand(S0 + VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + VTraits<v_uint32>::vlanes()));
                s2 = v_add(vx_load_expand(S0 + 2 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 2 * VTraits<v_uint32>::vlanes()));
                s3 = v_add(vx_load_expand(S0 + 3 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 3 * VTraits<v_uint32>::vlanes()));
                s4 = v_add(vx_load_expand(S0 + 4 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 4 * VTraits<v_uint32>::vlanes()));
                s5 = v_add(vx_load_expand(S0 + 5 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 5 * VTraits<v_uint32>::vlanes()));
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_uint32 bl, gl, rl;
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
#if CV_SIMD_WIDTH == 32
                bl = v_add(t0, t3); gl = v_add(t1, t4); rl = v_add(t2, t5);
#else //CV_SIMD_WIDTH == 64
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                bl = v_add(s0, s3); gl = v_add(s1, s4); rl = v_add(s2, s5);
#endif
                s0 = v_add(vx_load_expand(S0 + 6 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 6 * VTraits<v_uint32>::vlanes()));
                s1 = v_add(vx_load_expand(S0 + 7 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 7 * VTraits<v_uint32>::vlanes()));
                s2 = v_add(vx_load_expand(S0 + 8 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 8 * VTraits<v_uint32>::vlanes()));
                s3 = v_add(vx_load_expand(S0 + 9 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 9 * VTraits<v_uint32>::vlanes()));
                s4 = v_add(vx_load_expand(S0 + 10 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 10 * VTraits<v_uint32>::vlanes()));
                s5 = v_add(vx_load_expand(S0 + 11 * VTraits<v_uint32>::vlanes()), vx_load_expand(S1 + 11 * VTraits<v_uint32>::vlanes()));
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_uint32 bh, gh, rh;
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
#if CV_SIMD_WIDTH == 32
                bh = v_add(t0, t3); gh = v_add(t1, t4); rh = v_add(t2, t5);
#else //CV_SIMD_WIDTH == 64
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                bh = v_add(s0, s3); gh = v_add(s1, s4); rh = v_add(s2, s5);
#endif
                v_store_interleave(D, v_rshr_pack<2>(bl, bh), v_rshr_pack<2>(gl, gh), v_rshr_pack<2>(rl, rh));
            }
#elif CV_SIMD_WIDTH >= 64
            v_uint32 masklow = vx_setall_u32(0x0000ffff);
            for ( ; dx <= w - 3*VTraits<v_uint16>::vlanes(); dx += 3*VTraits<v_uint16>::vlanes(), S0 += 6*VTraits<v_uint16>::vlanes(), S1 += 6*VTraits<v_uint16>::vlanes(), D += 3*VTraits<v_uint16>::vlanes())
            {
                v_uint16 b0, g0, r0, b1, g1, r1;
                v_load_deinterleave(S0, b0, g0, r0);
                v_load_deinterleave(S1, b1, g1, r1);
                v_uint32 bl = (v_reinterpret_as_u32(b0) >> 16) + (v_reinterpret_as_u32(b0) & masklow) + (v_reinterpret_as_u32(b1) >> 16) + (v_reinterpret_as_u32(b1) & masklow);
                v_uint32 gl = (v_reinterpret_as_u32(g0) >> 16) + (v_reinterpret_as_u32(g0) & masklow) + (v_reinterpret_as_u32(g1) >> 16) + (v_reinterpret_as_u32(g1) & masklow);
                v_uint32 rl = (v_reinterpret_as_u32(r0) >> 16) + (v_reinterpret_as_u32(r0) & masklow) + (v_reinterpret_as_u32(r1) >> 16) + (v_reinterpret_as_u32(r1) & masklow);
                v_load_deinterleave(S0 + 3*VTraits<v_uint16>::vlanes(), b0, g0, r0);
                v_load_deinterleave(S1 + 3*VTraits<v_uint16>::vlanes(), b1, g1, r1);
                v_uint32 bh = (v_reinterpret_as_u32(b0) >> 16) + (v_reinterpret_as_u32(b0) & masklow) + (v_reinterpret_as_u32(b1) >> 16) + (v_reinterpret_as_u32(b1) & masklow);
                v_uint32 gh = (v_reinterpret_as_u32(g0) >> 16) + (v_reinterpret_as_u32(g0) & masklow) + (v_reinterpret_as_u32(g1) >> 16) + (v_reinterpret_as_u32(g1) & masklow);
                v_uint32 rh = (v_reinterpret_as_u32(r0) >> 16) + (v_reinterpret_as_u32(r0) & masklow) + (v_reinterpret_as_u32(r1) >> 16) + (v_reinterpret_as_u32(r1) & masklow);
                v_store_interleave(D, v_rshr_pack<2>(bl, bh), v_rshr_pack<2>(gl, gh), v_rshr_pack<2>(rl, rh));
            }
#endif
        }
        else
        {
            CV_Assert(cn == 4);
#if CV_SIMD_WIDTH >= 64
            for ( ; dx <= w - VTraits<v_uint16>::vlanes(); dx += VTraits<v_uint16>::vlanes(), S0 += 2*VTraits<v_uint16>::vlanes(), S1 += 2*VTraits<v_uint16>::vlanes(), D += VTraits<v_uint16>::vlanes())
            {
                v_uint64 r00, r01, r10, r11;
                v_load_deinterleave((uint64_t*)S0, r00, r01);
                v_load_deinterleave((uint64_t*)S1, r10, r11);

                v_uint32 r00l, r01l, r10l, r11l, r00h, r01h, r10h, r11h;
                v_expand(v_reinterpret_as_u16(r00), r00l, r00h);
                v_expand(v_reinterpret_as_u16(r01), r01l, r01h);
                v_expand(v_reinterpret_as_u16(r10), r10l, r10h);
                v_expand(v_reinterpret_as_u16(r11), r11l, r11h);
                v_store(D, v_rshr_pack<2>(v_add(r00l, r01l, r10l, r11l), v_add(r00h, r01h, r10h, r11h)));
            }
#else
            for ( ; dx <= w - VTraits<v_uint32>::vlanes(); dx += VTraits<v_uint32>::vlanes(), S0 += VTraits<v_uint16>::vlanes(), S1 += VTraits<v_uint16>::vlanes(), D += VTraits<v_uint32>::vlanes())
            {
                v_uint32 r0, r1, r2, r3;
                v_expand(vx_load(S0), r0, r1);
                v_expand(vx_load(S1), r2, r3);
                r0 = v_add(r0, r2); r1 = v_add(r1, r3);
                v_uint32 v_d;
#if CV_SIMD_WIDTH == 16
                v_d = v_add(r0, r1);
#elif CV_SIMD_WIDTH == 32
                v_uint32 t0, t1;
                v_recombine(r0, r1, t0, t1);
                v_d = v_add(t0, t1);
#endif
                v_rshr_pack_store<2>(D, v_d);
            }
#endif
        }

        return dx;
    }

private:
    int scale_x, scale_y;
    int cn;
    int step;
};

class ResizeAreaFastVec_SIMD_16s
{
public:
    ResizeAreaFastVec_SIMD_16s(int _scale_x, int _scale_y, int _cn, int _step) :
        scale_x(_scale_x), scale_y(_scale_y), cn(_cn), step(_step) {}

    int operator() (const short* S, short* D, int w) const
    {
        if (scale_x != 2 || scale_y != 2)
            return 0;

        int dx = 0;
        const short* S0 = (const short*)S;
        const short* S1 = (const short*)((const uchar*)(S) + step);

        if (cn == 1)
        {
            v_int32 masklow = vx_setall_s32(0x0000ffff);
            for (; dx <= w - VTraits<v_int32>::vlanes(); dx += VTraits<v_int32>::vlanes(), S0 += VTraits<v_int16>::vlanes(), S1 += VTraits<v_int16>::vlanes(), D += VTraits<v_int32>::vlanes())
            {
                v_int32 r0 = v_reinterpret_as_s32(vx_load(S0));
                v_int32 r1 = v_reinterpret_as_s32(vx_load(S1));
                v_rshr_pack_store<2>(D, v_add(v_add(v_add(v_shr<16>(r0), v_shr<16>(v_shl<16>(v_and(r0, masklow)))), v_shr<16>(r1)), v_shr<16>(v_shl<16>(v_and(r1, masklow)))));
            }
        }
        else if (cn == 3)
        {
#if CV_SIMD_WIDTH == 16
            for ( ; dx <= w - 4; dx += 3, S0 += 6, S1 += 6, D += 3)
                v_rshr_pack_store<2>(D, v_add(v_add(v_add(v_load_expand(S0), v_load_expand(S0 + 3)), v_load_expand(S1)), v_load_expand(S1 + 3)));
#elif CV_SIMD_WIDTH == 32 || CV_SIMD_WIDTH == 64
            for ( ; dx <= w - 3*VTraits<v_int16>::vlanes(); dx += 3*VTraits<v_int16>::vlanes(), S0 += 6*VTraits<v_int16>::vlanes(), S1 += 6*VTraits<v_int16>::vlanes(), D += 3*VTraits<v_int16>::vlanes())
            {
                v_int32 t0, t1, t2, t3, t4, t5;
                v_int32 s0, s1, s2, s3, s4, s5;
                s0 = v_add(vx_load_expand(S0), vx_load_expand(S1));
                s1 = v_add(vx_load_expand(S0 + VTraits<v_int32>::vlanes()), vx_load_expand(S1 + VTraits<v_int32>::vlanes()));
                s2 = v_add(vx_load_expand(S0 + 2 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 2 * VTraits<v_int32>::vlanes()));
                s3 = v_add(vx_load_expand(S0 + 3 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 3 * VTraits<v_int32>::vlanes()));
                s4 = v_add(vx_load_expand(S0 + 4 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 4 * VTraits<v_int32>::vlanes()));
                s5 = v_add(vx_load_expand(S0 + 5 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 5 * VTraits<v_int32>::vlanes()));
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_int32 bl, gl, rl;
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
#if CV_SIMD_WIDTH == 32
                bl = v_add(t0, t3); gl = v_add(t1, t4); rl = v_add(t2, t5);
#else //CV_SIMD_WIDTH == 64
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                bl = v_add(s0, s3); gl = v_add(s1, s4); rl = v_add(s2, s5);
#endif
                s0 = v_add(vx_load_expand(S0 + 6 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 6 * VTraits<v_int32>::vlanes()));
                s1 = v_add(vx_load_expand(S0 + 7 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 7 * VTraits<v_int32>::vlanes()));
                s2 = v_add(vx_load_expand(S0 + 8 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 8 * VTraits<v_int32>::vlanes()));
                s3 = v_add(vx_load_expand(S0 + 9 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 9 * VTraits<v_int32>::vlanes()));
                s4 = v_add(vx_load_expand(S0 + 10 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 10 * VTraits<v_int32>::vlanes()));
                s5 = v_add(vx_load_expand(S0 + 11 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 11 * VTraits<v_int32>::vlanes()));
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                v_int32 bh, gh, rh;
                v_zip(s0, s3, t0, t1); v_zip(s1, s4, t2, t3); v_zip(s2, s5, t4, t5);
#if CV_SIMD_WIDTH == 32
                bh = v_add(t0, t3); gh = v_add(t1, t4); rh = v_add(t2, t5);
#else //CV_SIMD_WIDTH == 64
                v_zip(t0, t3, s0, s1); v_zip(t1, t4, s2, s3); v_zip(t2, t5, s4, s5);
                bh = v_add(s0, s3); gh = v_add(s1, s4); rh = v_add(s2, s5);
#endif
                v_store_interleave(D, v_rshr_pack<2>(bl, bh), v_rshr_pack<2>(gl, gh), v_rshr_pack<2>(rl, rh));
            }
#elif CV_SIMD_WIDTH >= 64
            for ( ; dx <= w - 3*VTraits<v_int16>::vlanes(); dx += 3*VTraits<v_int16>::vlanes(), S0 += 6*VTraits<v_int16>::vlanes(), S1 += 6*VTraits<v_int16>::vlanes(), D += 3*VTraits<v_int16>::vlanes())
            {
                v_int16 b0, g0, r0, b1, g1, r1;
                v_load_deinterleave(S0, b0, g0, r0);
                v_load_deinterleave(S1, b1, g1, r1);
                v_int32 bl = (v_reinterpret_as_s32(b0) >> 16) + ((v_reinterpret_as_s32(b0) << 16) >> 16) + (v_reinterpret_as_s32(b1) >> 16) + ((v_reinterpret_as_s32(b1) << 16) >> 16);
                v_int32 gl = (v_reinterpret_as_s32(g0) >> 16) + ((v_reinterpret_as_s32(g0) << 16) >> 16) + (v_reinterpret_as_s32(g1) >> 16) + ((v_reinterpret_as_s32(g1) << 16) >> 16);
                v_int32 rl = (v_reinterpret_as_s32(r0) >> 16) + ((v_reinterpret_as_s32(r0) << 16) >> 16) + (v_reinterpret_as_s32(r1) >> 16) + ((v_reinterpret_as_s32(r1) << 16) >> 16);
                v_load_deinterleave(S0 + 3*VTraits<v_int16>::vlanes(), b0, g0, r0);
                v_load_deinterleave(S1 + 3*VTraits<v_int16>::vlanes(), b1, g1, r1);
                v_int32 bh = (v_reinterpret_as_s32(b0) >> 16) + ((v_reinterpret_as_s32(b0) << 16) >> 16) + (v_reinterpret_as_s32(b1) >> 16) + ((v_reinterpret_as_s32(b1) << 16) >> 16);
                v_int32 gh = (v_reinterpret_as_s32(g0) >> 16) + ((v_reinterpret_as_s32(g0) << 16) >> 16) + (v_reinterpret_as_s32(g1) >> 16) + ((v_reinterpret_as_s32(g1) << 16) >> 16);
                v_int32 rh = (v_reinterpret_as_s32(r0) >> 16) + ((v_reinterpret_as_s32(r0) << 16) >> 16) + (v_reinterpret_as_s32(r1) >> 16) + ((v_reinterpret_as_s32(r1) << 16) >> 16);
                v_store_interleave(D, v_rshr_pack<2>(bl, bh), v_rshr_pack<2>(gl, gh), v_rshr_pack<2>(rl, rh));
            }
#endif
        }
        else
        {
            CV_Assert(cn == 4);
            for (; dx <= w - VTraits<v_int16>::vlanes(); dx += VTraits<v_int16>::vlanes(), S0 += 2 * VTraits<v_int16>::vlanes(), S1 += 2 * VTraits<v_int16>::vlanes(), D += VTraits<v_int16>::vlanes())
            {
#if CV_SIMD_WIDTH >= 64
                v_int64 r00, r01, r10, r11;
                v_load_deinterleave((int64_t*)S0, r00, r01);
                v_load_deinterleave((int64_t*)S1, r10, r11);

                v_int32 r00l, r01l, r10l, r11l, r00h, r01h, r10h, r11h;
                v_expand(v_reinterpret_as_s16(r00), r00l, r00h);
                v_expand(v_reinterpret_as_s16(r01), r01l, r01h);
                v_expand(v_reinterpret_as_s16(r10), r10l, r10h);
                v_expand(v_reinterpret_as_s16(r11), r11l, r11h);
                v_store(D, v_rshr_pack<2>(v_add(r00l, r01l, r10l, r11l), v_add(r00h, r01h, r10h, r11h)));
#else
                v_int32 r0, r1, r2, r3;
                r0 = v_add(vx_load_expand(S0), vx_load_expand(S1));
                r1 = v_add(vx_load_expand(S0 + VTraits<v_int32>::vlanes()), vx_load_expand(S1 + VTraits<v_int32>::vlanes()));
                r2 = v_add(vx_load_expand(S0 + 2 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 2 * VTraits<v_int32>::vlanes()));
                r3 = v_add(vx_load_expand(S0 + 3 * VTraits<v_int32>::vlanes()), vx_load_expand(S1 + 3 * VTraits<v_int32>::vlanes()));
                v_int32 dl, dh;
#if CV_SIMD_WIDTH == 16
                dl = v_add(r0, r1); dh = v_add(r2, r3);
#elif CV_SIMD_WIDTH == 32
                v_int32 t0, t1, t2, t3;
                v_recombine(r0, r1, t0, t1); v_recombine(r2, r3, t2, t3);
                dl = v_add(t0, t1); dh = v_add(t2, t3);
#endif
                v_store(D, v_rshr_pack<2>(dl, dh));
#endif
            }
        }

        return dx;
    }

private:
    int scale_x, scale_y;
    int cn;
    int step;
};

struct ResizeAreaFastVec_SIMD_32f
{
    ResizeAreaFastVec_SIMD_32f(int _scale_x, int _scale_y, int _cn, int _step) :
        cn(_cn), step(_step)
    {
        fast_mode = _scale_x == 2 && _scale_y == 2 && (cn == 1 || cn == 3 || cn == 4);
    }

    int operator() (const float * S, float * D, int w) const
    {
        if (!fast_mode)
            return 0;

        const float * S0 = S, * S1 = (const float *)((const uchar *)(S0) + step);
        int dx = 0;

        if (cn == 1)
        {
            v_float32 v_025 = vx_setall_f32(0.25f);
            for ( ; dx <= w - VTraits<v_float32>::vlanes(); dx += VTraits<v_float32>::vlanes(), S0 += 2*VTraits<v_float32>::vlanes(), S1 += 2*VTraits<v_float32>::vlanes(), D += VTraits<v_float32>::vlanes())
            {
                v_float32 v_row00, v_row01, v_row10, v_row11;
                v_load_deinterleave(S0, v_row00, v_row01);
                v_load_deinterleave(S1, v_row10, v_row11);
                v_store(D, v_mul(v_add(v_add(v_row00, v_row01), v_add(v_row10, v_row11)), v_025));
            }
        }
        else if (cn == 3)
        {
            dx = area_fast_detail::area_fast_f32_cn3_2x2(S0, S1, D, w, dx);
        }
        else if (cn == 4)
        {
#if CV_SIMD_WIDTH == 16
            v_float32 v_025 = vx_setall_f32(0.25f);
            for (; dx <= w - VTraits<v_float32>::vlanes(); dx += VTraits<v_float32>::vlanes(), S0 += 2*VTraits<v_float32>::vlanes(), S1 += 2*VTraits<v_float32>::vlanes(), D += VTraits<v_float32>::vlanes())
                v_store(D, v_mul(v_add(v_add(vx_load(S0), vx_load(S0 + VTraits<v_float32>::vlanes())), v_add(vx_load(S1), vx_load(S1 + VTraits<v_float32>::vlanes()))), v_025));
#elif CV_SIMD256
            v_float32x8 v_025 = v256_setall_f32(0.25f);
            for (; dx <= w - VTraits<v_float32x8>::vlanes(); dx += VTraits<v_float32x8>::vlanes(), S0 += 2*VTraits<v_float32x8>::vlanes(), S1 += 2*VTraits<v_float32x8>::vlanes(), D += VTraits<v_float32x8>::vlanes())
            {
                v_float32x8 dst0, dst1;
                v_recombine(v_add(v256_load(S0), v256_load(S1)), v_add(v256_load(S0 + VTraits<v_float32x8>::vlanes()), v256_load(S1 + VTraits<v_float32x8>::vlanes())), dst0, dst1);
                v_store(D, v_mul(v_add(dst0, dst1), v_025));
            }
#endif
        }

        return dx;
    }

private:
    int cn;
    bool fast_mode;
    int step;
};

#else

typedef ResizeAreaFastNoVec<uchar, uchar> ResizeAreaFastVec_SIMD_8u;
typedef ResizeAreaFastNoVec<ushort, ushort> ResizeAreaFastVec_SIMD_16u;
typedef ResizeAreaFastNoVec<short, short> ResizeAreaFastVec_SIMD_16s;
typedef ResizeAreaFastNoVec<float, float> ResizeAreaFastVec_SIMD_32f;

#endif

template<typename T, typename SIMDVecOp>
struct ResizeAreaFastVec
{
    ResizeAreaFastVec(int _scale_x, int _scale_y, int _cn, int _step) :
        scale_x(_scale_x), scale_y(_scale_y), cn(_cn), step(_step),
        vecOp(_scale_x, _scale_y, _cn, _step)
    {
        fast_mode = area_fast_detail::is_fast_scale<T>(scale_x, scale_y, cn);
    }

    int operator() (const T* S, T* D, int w) const
    {
        if (!fast_mode)
            return 0;

        int dx = vecOp(S, D, w);

        if (scale_x == 2 && scale_y == 2)
            dx = area_fast_detail::area_fast_tail_2x2(S, D, w, dx, cn, step);
        else
            dx = area_fast_detail::area_fast_tail_nxn(S, D, w, dx, scale_x, cn, step);

        return dx > w ? w : dx;
    }

private:
    int scale_x, scale_y;
    int cn;
    bool fast_mode;
    int step;
    SIMDVecOp vecOp;
};

template <typename T, typename WT, typename VecOp>
class resizeAreaFast_Invoker :
    public ParallelLoopBody
{
public:
    resizeAreaFast_Invoker(const Mat &_src, Mat &_dst,
        int _scale_x, int _scale_y, const int* _ofs, const int* _xofs) :
        ParallelLoopBody(), src(_src), dst(_dst), scale_x(_scale_x),
        scale_y(_scale_y), ofs(_ofs), xofs(_xofs)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size ssize = src.size(), dsize = dst.size();
        int cn = src.channels();
        int area = scale_x*scale_y;
        float scale = 1.f/(area);
        int dwidth1 = (ssize.width/scale_x)*cn;
        dsize.width *= cn;
        ssize.width *= cn;
        int dy, dx, k = 0;

        VecOp vop(scale_x, scale_y, src.channels(), (int)src.step/*, area_ofs*/);

        for( dy = range.start; dy < range.end; dy++ )
        {
            T* D = (T*)(dst.data + dst.step*dy);
            int sy0 = dy*scale_y;
            int w = sy0 + scale_y <= ssize.height ? dwidth1 : 0;

            if( sy0 >= ssize.height )
            {
                for( dx = 0; dx < dsize.width; dx++ )
                    D[dx] = 0;
                continue;
            }

            dx = vop(src.template ptr<T>(sy0), D, w);
            for( ; dx < w; dx++ )
            {
                const T* S = src.template ptr<T>(sy0) + xofs[dx];
                WT sum = 0;
                k = 0;
                #if CV_ENABLE_UNROLLED
                for( ; k <= area - 4; k += 4 )
                    sum += S[ofs[k]] + S[ofs[k+1]] + S[ofs[k+2]] + S[ofs[k+3]];
                #endif
                for( ; k < area; k++ )
                    sum += S[ofs[k]];

                D[dx] = saturate_cast<T>(sum * scale);
            }

            for( ; dx < dsize.width; dx++ )
            {
                WT sum = 0;
                int count = 0, sx0 = xofs[dx];
                if( sx0 >= ssize.width )
                    D[dx] = 0;

                for( int sy = 0; sy < scale_y; sy++ )
                {
                    if( sy0 + sy >= ssize.height )
                        break;
                    const T* S = src.template ptr<T>(sy0 + sy) + sx0;
                    for( int sx = 0; sx < scale_x*cn; sx += cn )
                    {
                        if( sx0 + sx >= ssize.width )
                            break;
                        sum += S[sx];
                        count++;
                    }
                }

                D[dx] = saturate_cast<T>((float)sum/count);
            }
        }
    }

private:
    Mat src;
    Mat dst;
    int scale_x, scale_y;
    const int *ofs, *xofs;
};

template<typename T, typename WT, typename VecOp>
static void resizeAreaFast_( const Mat& src, Mat& dst, const int* ofs, const int* xofs,
                             int scale_x, int scale_y )
{
    Range range(0, dst.rows);
    resizeAreaFast_Invoker<T, WT, VecOp> invoker(src, dst, scale_x,
        scale_y, ofs, xofs);
    parallel_for_(range, invoker, dst.total()/(double)(1<<16));
}

struct DecimateAlpha
{
    int si, di;
    float alpha;
};


namespace inter_area {
#if (CV_SIMD || CV_SIMD_SCALABLE)
inline void saturate_store(const float* src, uchar* dst) {
    const v_int32 tmp0 = v_round(vx_load(src + 0 * VTraits<v_float32>::vlanes()));
    const v_int32 tmp1 = v_round(vx_load(src + 1 * VTraits<v_float32>::vlanes()));
    const v_int32 tmp2 = v_round(vx_load(src + 2 * VTraits<v_float32>::vlanes()));
    const v_int32 tmp3 = v_round(vx_load(src + 3 * VTraits<v_float32>::vlanes()));
    v_store(dst, v_pack(v_pack_u(tmp0, tmp1), v_pack_u(tmp2, tmp3)));
}

inline void saturate_store(const float* src, ushort* dst) {
    const v_int32 tmp0 = v_round(vx_load(src + 0 * VTraits<v_float32>::vlanes()));
    const v_int32 tmp1 = v_round(vx_load(src + 1 * VTraits<v_float32>::vlanes()));
    v_store(dst, v_pack_u(tmp0, tmp1));
}

inline void saturate_store(const float* src, short* dst) {
    const v_int32 tmp0 = v_round(vx_load(src + 0 * VTraits<v_float32>::vlanes()));
    const v_int32 tmp1 = v_round(vx_load(src + 1 * VTraits<v_float32>::vlanes()));
    v_store(dst, v_pack(tmp0, tmp1));
}

static inline v_float32 vx_setall(float coeff) { return vx_setall_f32(coeff); }

template <typename T>
struct VArea {};

template <>
struct VArea<float> {
    typedef v_float32 vWT;
};
#endif

#if (CV_SIMD128_64F || CV_SIMD_SCALABLE_64F)
static inline v_float64 vx_setall(double coeff) { return vx_setall_f64(coeff); }

template <>
struct VArea<double> {
    typedef v_float64 vWT;
};

#else
inline void mul(const double* buf, int width, double beta, double* sum) {
    for (int dx = 0; dx < width; ++dx) {
        sum[dx] = beta * buf[dx];
    }
}

inline void muladd(const double* buf, int width, double beta, double* sum) {
    for (int dx = 0; dx < width; ++dx) {
        sum[dx] += beta * buf[dx];
    }
}
#endif

template <typename T, typename WT>
inline void saturate_store(const WT* sum, int width, T* D) {
    int dx = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int step = VTraits<typename VArea<WT>::vWT>::vlanes() * sizeof(WT) / sizeof(T);
    for (; dx + step < width; dx += step) {
        saturate_store(sum + dx, D + dx);
    }
#endif
    for (; dx < width; ++dx) {
        D[dx] = saturate_cast<T>(sum[dx]);
    }
}

// Optimization when T == WT.
template <typename WT>
inline void saturate_store(const WT* sum, int width, WT* D) {
    std::copy(sum, sum + width, D);
}

template <typename WT>
inline void mul(const WT* buf, int width, WT beta, WT* sum) {
    int dx = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int step = VTraits<typename VArea<WT>::vWT>::vlanes();
    for (; dx + step < width; dx += step) {
        vx_store(sum + dx, v_mul(vx_setall(beta), vx_load(buf + dx)));
    }
#endif
    for (; dx < width; ++dx) {
        sum[dx] = beta * buf[dx];
    }
}

template <typename WT>
inline void muladd(const WT* buf, int width, WT beta, WT* sum) {
    int dx = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int step = VTraits<typename VArea<WT>::vWT>::vlanes();
    for (; dx + step < width; dx += step) {
        vx_store(sum + dx, v_add(vx_load(sum + dx), v_mul(vx_setall(beta), vx_load(buf + dx))));
    }
#endif
    for (; dx < width; ++dx) {
        sum[dx] += beta * buf[dx];
    }
}

}  // namespace inter_area

template<typename T, typename WT> class ResizeArea_Invoker :
    public ParallelLoopBody
{
public:
    ResizeArea_Invoker( const Mat& _src, Mat& _dst,
                        const DecimateAlpha* _xtab, int _xtab_size,
                        const DecimateAlpha* _ytab, int _ytab_size,
                        const int* _tabofs )
    {
        src = &_src;
        dst = &_dst;
        xtab0 = _xtab;
        xtab_size0 = _xtab_size;
        ytab = _ytab;
        ytab_size = _ytab_size;
        tabofs = _tabofs;
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size dsize = dst->size();
        int cn = dst->channels();
        dsize.width *= cn;
        AutoBuffer<WT> _buffer(dsize.width*2);
        const DecimateAlpha* xtab = xtab0;
        int xtab_size = xtab_size0;
        WT *buf = _buffer.data(), *sum = buf + dsize.width;
        int j_start = tabofs[range.start], j_end = tabofs[range.end], j, k, dx, prev_dy = ytab[j_start].di;

        for( dx = 0; dx < dsize.width; dx++ )
            sum[dx] = (WT)0;

        for( j = j_start; j < j_end; j++ )
        {
            WT beta = ytab[j].alpha;
            int dy = ytab[j].di;
            int sy = ytab[j].si;

            {
                const T* S = src->template ptr<T>(sy);
                for( dx = 0; dx < dsize.width; dx++ )
                    buf[dx] = (WT)0;

                if( cn == 1 )
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        buf[dxn] += S[xtab[k].si]*alpha;
                    }
                else if( cn == 2 )
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        WT t0 = buf[dxn] + S[sxn]*alpha;
                        WT t1 = buf[dxn+1] + S[sxn+1]*alpha;
                        buf[dxn] = t0; buf[dxn+1] = t1;
                    }
                else if( cn == 3 )
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        WT t0 = buf[dxn] + S[sxn]*alpha;
                        WT t1 = buf[dxn+1] + S[sxn+1]*alpha;
                        WT t2 = buf[dxn+2] + S[sxn+2]*alpha;
                        buf[dxn] = t0; buf[dxn+1] = t1; buf[dxn+2] = t2;
                    }
                else if( cn == 4 )
                {
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        WT t0 = buf[dxn] + S[sxn]*alpha;
                        WT t1 = buf[dxn+1] + S[sxn+1]*alpha;
                        buf[dxn] = t0; buf[dxn+1] = t1;
                        t0 = buf[dxn+2] + S[sxn+2]*alpha;
                        t1 = buf[dxn+3] + S[sxn+3]*alpha;
                        buf[dxn+2] = t0; buf[dxn+3] = t1;
                    }
                }
                else
                {
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        for( int c = 0; c < cn; c++ )
                            buf[dxn + c] += S[sxn + c]*alpha;
                    }
                }
            }

            if( dy != prev_dy )
            {
                inter_area::saturate_store(sum, dsize.width, dst->template ptr<T>(prev_dy));
                inter_area::mul(buf, dsize.width, beta, sum);
                prev_dy = dy;
            }
            else
            {
                inter_area::muladd(buf, dsize.width, beta, sum);
            }
        }

        inter_area::saturate_store(sum, dsize.width, dst->template ptr<T>(prev_dy));
    }

private:
    const Mat* src;
    Mat* dst;
    const DecimateAlpha* xtab0;
    const DecimateAlpha* ytab;
    int xtab_size0, ytab_size;
    const int* tabofs;
};


template <typename T, typename WT>
static void resizeArea_( const Mat& src, Mat& dst,
                         const DecimateAlpha* xtab, int xtab_size,
                         const DecimateAlpha* ytab, int ytab_size,
                         const int* tabofs )
{
    parallel_for_(Range(0, dst.rows),
                 ResizeArea_Invoker<T, WT>(src, dst, xtab, xtab_size, ytab, ytab_size, tabofs),
                 dst.total()/((double)(1 << 16)));
}

// Horizontal INTER_AREA taps per destination element (channel included), stored tap-major
// so a vector of neighbouring elements loads tap t contiguously. Elements with fewer than
// ntaps taps are padded with a valid index and zero weight; the padding is masked out.
struct AreaXPlan
{
    int ntaps;
    int width;
    const int* idx;
    const float* w;
    const int* cnt;
};

static void buildAreaXPlan( const DecimateAlpha* xtab, int xtab_size, int width, int cn,
                            AutoBuffer<int>& _idx, AutoBuffer<float>& _w, AutoBuffer<int>& _cnt,
                            AreaXPlan& plan )
{
    _cnt.allocate(width);
    int* cnt = _cnt.data();
    std::fill(cnt, cnt + width, 0);
    for( int k = 0; k < xtab_size; k++ )
        cnt[xtab[k].di]++;
    int ntaps = 0;
    for( int e = 0; e < width; e += cn )
        ntaps = std::max(ntaps, cnt[e]);

    _idx.allocate((size_t)ntaps * width);
    _w.allocate((size_t)ntaps * width);
    int* idx = _idx.data();
    float* w = _w.data();
    std::fill(w, w + (size_t)ntaps * width, 0.f);

    for( int k = 0, t = 0; k < xtab_size; k++ )
    {
        const int e = xtab[k].di;
        t = (k > 0 && xtab[k - 1].di == e) ? t + 1 : 0;
        for( int c = 0; c < cn; c++ )
        {
            idx[(size_t)t * width + e + c] = xtab[k].si + c;
            w[(size_t)t * width + e + c] = xtab[k].alpha;
            if( t == 0 )
                for( int p = cnt[e]; p < ntaps; p++ )
                    idx[(size_t)p * width + e + c] = xtab[k].si + c;
        }
    }
    for( int e = 0; e < width; e += cn )
        for( int c = 1; c < cn; c++ )
            cnt[e + c] = cnt[e];

    plan.ntaps = ntaps;
    plan.width = width;
    plan.idx = idx;
    plan.w = w;
    plan.cnt = cnt;
}

// S must be readable one float past the row end when cn == 3.
static void area_hpass_32f(float* D, const float* S, const AreaXPlan& p, int cn)
{
    const int width = p.width;
    int e = 0;
#if CV_SIMD128
    // 3 or 4 channels of a pixel share their taps: one 4-float load per tap, no gather.
    // A 3-channel store spills one float into the next pixel, which overwrites it, so the
    // last pixel is left to the scalar loop.
    if( cn == 3 || cn == 4 )
    {
        const int end = cn == 4 ? width : width - 3;
        for( ; e < end; e += cn )
        {
            v_float32x4 acc = v_setzero_f32();
            for( int t = 0; t < p.cnt[e]; t++ )
            {
                const size_t o = (size_t)t * width + e;
                acc = v_muladd(v_load(S + p.idx[o]), v_setall_f32(p.w[o]), acc);
            }
            v_store(D + e, acc);
        }
    }
    else
#endif
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int L = VTraits<v_float32>::vlanes();
        for( ; e <= width - L; e += L )
        {
            const v_int32 cnt = vx_load(p.cnt + e);
            v_float32 acc = vx_setzero_f32();
            for( int t = 0; t < p.ntaps; t++ )
            {
                const size_t o = (size_t)t * width + e;
                const v_float32 s = vx_lut(S, p.idx + o);
                const v_float32 m = v_reinterpret_as_f32(v_lt(vx_setall_s32(t), cnt));
                acc = v_select(m, v_muladd(s, vx_load(p.w + o), acc), acc);
            }
            v_store(D + e, acc);
        }
#endif
    }
    for( ; e < width; e++ )
    {
        float acc = 0.f;
        for( int t = 0; t < p.cnt[e]; t++ )
            acc += S[p.idx[(size_t)t * width + e]] * p.w[(size_t)t * width + e];
        D[e] = acc;
    }
}

// vrow = beta*S (first row of a destination row) or vrow += beta*S.
static inline void area_vacc_row(const float* S, int n, float beta, float* vrow, bool first)
{
    if( first )
        inter_area::mul(S, n, beta, vrow);
    else
        inter_area::muladd(S, n, beta, vrow);
}

static inline void area_vacc_row(const uchar* S, int n, float beta, float* vrow, bool first)
{
    int x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int L = VTraits<v_float32>::vlanes();
    const v_float32 vb = vx_setall_f32(beta);
    for( ; x <= n - L; x += L )
    {
        const v_float32 s = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand_q(S + x)));
        v_store(vrow + x, first ? v_mul(s, vb) : v_muladd(s, vb, vx_load(vrow + x)));
    }
#endif
    for( ; x < n; x++ )
        vrow[x] = first ? S[x]*beta : vrow[x] + S[x]*beta;
}

static inline void area_vacc_row(const ushort* S, int n, float beta, float* vrow, bool first)
{
    int x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int L = VTraits<v_float32>::vlanes();
    const v_float32 vb = vx_setall_f32(beta);
    for( ; x <= n - L; x += L )
    {
        const v_float32 s = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand(S + x)));
        v_store(vrow + x, first ? v_mul(s, vb) : v_muladd(s, vb, vx_load(vrow + x)));
    }
#endif
    for( ; x < n; x++ )
        vrow[x] = first ? S[x]*beta : vrow[x] + S[x]*beta;
}

static inline void area_vacc_row(const short* S, int n, float beta, float* vrow, bool first)
{
    int x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int L = VTraits<v_float32>::vlanes();
    const v_float32 vb = vx_setall_f32(beta);
    for( ; x <= n - L; x += L )
    {
        const v_float32 s = v_cvt_f32(vx_load_expand(S + x));
        v_store(vrow + x, first ? v_mul(s, vb) : v_muladd(s, vb, vx_load(vrow + x)));
    }
#endif
    for( ; x < n; x++ )
        vrow[x] = first ? S[x]*beta : vrow[x] + S[x]*beta;
}

// The weighted source rows of each destination row are summed first, so the horizontal
// pass runs once per destination row instead of once per (row, weight) pair. The order of
// the float additions differs from ResizeArea_Invoker.
template<typename T>
class ResizeAreaVFirst_Invoker : public ParallelLoopBody
{
public:
    ResizeAreaVFirst_Invoker( const Mat& _src, Mat& _dst, const AreaXPlan& _xplan,
                              const DecimateAlpha* _ytab, const int* _tabofs ) :
        src(_src), dst(_dst), xplan(_xplan), ytab(_ytab), tabofs(_tabofs)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        const int cn = src.channels(), swidth = src.cols * cn, dwidth = dst.cols * cn;
        const bool is_float = std::is_same<T, float>::value;
        AutoBuffer<float> _vrow(swidth + 1 + (is_float ? 0 : dwidth));
        float* vrow = _vrow.data();
        float* drow = vrow + swidth + 1;
        vrow[swidth] = 0.f;

        for( int dy = range.start; dy < range.end; dy++ )
        {
            for( int j = tabofs[dy]; j < tabofs[dy + 1]; j++ )
                area_vacc_row(src.ptr<T>(ytab[j].si), swidth, ytab[j].alpha, vrow, j == tabofs[dy]);
            if( is_float )
                area_hpass_32f(dst.ptr<float>(dy), vrow, xplan, cn);
            else
            {
                area_hpass_32f(drow, vrow, xplan, cn);
                inter_area::saturate_store(drow, dwidth, dst.ptr<T>(dy));
            }
        }
    }

private:
    const Mat& src;
    Mat& dst;
    const AreaXPlan& xplan;
    const DecimateAlpha* ytab;
    const int* tabofs;
};

template <typename T>
static void resizeAreaVFirst_( const Mat& src, Mat& dst,
                               const DecimateAlpha* xtab, int xtab_size,
                               const DecimateAlpha* ytab, int /*ytab_size*/,
                               const int* tabofs )
{
    AutoBuffer<int> _xidx, _xcnt;
    AutoBuffer<float> _xw;
    AreaXPlan xplan;
    buildAreaXPlan(xtab, xtab_size, dst.cols*dst.channels(), dst.channels(), _xidx, _xw, _xcnt, xplan);
    parallel_for_(Range(0, dst.rows),
                  ResizeAreaVFirst_Invoker<T>(src, dst, xplan, ytab, tabofs),
                  dst.total()/((double)(1 << 16)));
}



typedef void (*ResizeFunc)( const Mat& src, Mat& dst,
                            const int* xofs, const void* alpha,
                            const int* yofs, const void* beta,
                            int xmin, int xmax, int ksize );

typedef void (*ResizeAreaFastFunc)( const Mat& src, Mat& dst,
                                    const int* ofs, const int *xofs,
                                    int scale_x, int scale_y );

typedef void (*ResizeAreaFunc)( const Mat& src, Mat& dst,
                                const DecimateAlpha* xtab, int xtab_size,
                                const DecimateAlpha* ytab, int ytab_size,
                                const int* yofs);


static int computeResizeAreaTab( int ssize, int dsize, int cn, double scale, DecimateAlpha* tab )
{
    int k = 0;
    for(int dx = 0; dx < dsize; dx++ )
    {
        double fsx1 = dx * scale;
        double fsx2 = fsx1 + scale;
        double cellWidth = std::min(scale, ssize - fsx1);

        int sx1 = cvCeil(fsx1), sx2 = cvFloor(fsx2);

        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);

        if( sx1 - fsx1 > 1e-3 )
        {
            CV_Assert( k < ssize*2 );
            tab[k].di = dx * cn;
            tab[k].si = (sx1 - 1) * cn;
            tab[k++].alpha = (float)((sx1 - fsx1) / cellWidth);
        }

        for(int sx = sx1; sx < sx2; sx++ )
        {
            CV_Assert( k < ssize*2 );
            tab[k].di = dx * cn;
            tab[k].si = sx * cn;
            tab[k++].alpha = float(1.0 / cellWidth);
        }

        if( fsx2 - sx2 > 1e-3 )
        {
            CV_Assert( k < ssize*2 );
            tab[k].di = dx * cn;
            tab[k].si = sx2 * cn;
            tab[k++].alpha = (float)(std::min(std::min(fsx2 - sx2, 1.), cellWidth) / cellWidth);
        }
    }
    return k;
}

namespace linear_up_2x_detail {

#if (CV_SIMD || CV_SIMD_SCALABLE)

// Bit-exact with the fixed-point separable INTER_LINEAR path for an exact 2x upscale.
// The horizontal pass yields 512*E (even column, E = p[k-1] + 3*p[k]) or 512*O (odd
// column, O = 3*p[k] + p[k+1]); VResizeLinear then truncates each weighted row before
// summing: out = ((X_far >> 2) + ((3*X_near) >> 2) + 2) >> 2. Border neighbours clamp.

static inline void linear2x_scalar_px(uchar* D0, uchar* D1, const uchar* U, const uchar* M,
                                      const uchar* B, int k, int sw, int cn)
{
    const int l = std::max(k - 1, 0) * cn, c = k * cn, r = std::min(k + 1, sw - 1) * cn;
    uchar* e0 = D0 + 2 * c;
    uchar* e1 = D1 + 2 * c;
    for( int i = 0; i < cn; i++ )
    {
        const int Eu = U[l + i] + 3 * U[c + i], Ou = 3 * U[c + i] + U[r + i];
        const int Em = M[l + i] + 3 * M[c + i], Om = 3 * M[c + i] + M[r + i];
        const int Eb = B[l + i] + 3 * B[c + i], Ob = 3 * B[c + i] + B[r + i];
        const int tE = (3 * Em) >> 2, tO = (3 * Om) >> 2;
        e0[i]      = (uchar)(((Eu >> 2) + tE + 2) >> 2);
        e0[i + cn] = (uchar)(((Ou >> 2) + tO + 2) >> 2);
        e1[i]      = (uchar)(((Eb >> 2) + tE + 2) >> 2);
        e1[i + cn] = (uchar)(((Ob >> 2) + tO + 2) >> 2);
    }
}

static inline v_uint16 linear2x_tap(const v_uint16& side, const v_uint16& mid)
{
    return v_add(v_add(side, mid), v_shl<1>(mid));
}

static inline v_uint16 linear2x_mix(const v_uint16& far_, const v_uint16& near3)
{
    return v_shr<2>(v_add(v_add(v_shr<2>(far_), near3), vx_setall_u16(2)));
}

static inline void linear2x_core_u16(const v_uint16& ul, const v_uint16& uc, const v_uint16& ur,
                                     const v_uint16& ml, const v_uint16& mc, const v_uint16& mr,
                                     const v_uint16& bl, const v_uint16& bc, const v_uint16& br,
                                     v_uint16& e0, v_uint16& o0, v_uint16& e1, v_uint16& o1)
{
    const v_uint16 Em = linear2x_tap(ml, mc), Om = linear2x_tap(mr, mc);
    const v_uint16 tE = v_shr<2>(v_add(Em, v_shl<1>(Em)));
    const v_uint16 tO = v_shr<2>(v_add(Om, v_shl<1>(Om)));
    e0 = linear2x_mix(linear2x_tap(ul, uc), tE);
    o0 = linear2x_mix(linear2x_tap(ur, uc), tO);
    e1 = linear2x_mix(linear2x_tap(bl, bc), tE);
    o1 = linear2x_mix(linear2x_tap(br, bc), tO);
}

static inline void linear2x_core_u8(const v_uint8& ul, const v_uint8& uc, const v_uint8& ur,
                                    const v_uint8& ml, const v_uint8& mc, const v_uint8& mr,
                                    const v_uint8& bl, const v_uint8& bc, const v_uint8& br,
                                    v_uint8& e0, v_uint8& o0, v_uint8& e1, v_uint8& o1)
{
    v_uint16 e0l, o0l, e1l, o1l, e0h, o0h, e1h, o1h;
    linear2x_core_u16(v_expand_low(ul), v_expand_low(uc), v_expand_low(ur),
                      v_expand_low(ml), v_expand_low(mc), v_expand_low(mr),
                      v_expand_low(bl), v_expand_low(bc), v_expand_low(br), e0l, o0l, e1l, o1l);
    linear2x_core_u16(v_expand_high(ul), v_expand_high(uc), v_expand_high(ur),
                      v_expand_high(ml), v_expand_high(mc), v_expand_high(mr),
                      v_expand_high(bl), v_expand_high(bc), v_expand_high(br), e0h, o0h, e1h, o1h);
    e0 = v_pack(e0l, e0h); o0 = v_pack(o0l, o0h);
    e1 = v_pack(e1l, e1h); o1 = v_pack(o1l, o1h);
}

// Interleave even/odd output pixels of cn bytes each and store 2*vlanes bytes.
static inline void linear2x_store_pair(uchar* D, const v_uint8& e, const v_uint8& o, int cn)
{
    const int n = VTraits<v_uint8>::vlanes();
    if( cn == 1 )
    {
        v_uint8 lo, hi;
        v_zip(e, o, lo, hi);
        v_store(D, lo); v_store(D + n, hi);
    }
    else if( cn == 2 )
    {
        v_uint16 lo, hi;
        v_zip(v_reinterpret_as_u16(e), v_reinterpret_as_u16(o), lo, hi);
        v_store(D, v_reinterpret_as_u8(lo)); v_store(D + n, v_reinterpret_as_u8(hi));
    }
    else
    {
        v_uint32 lo, hi;
        v_zip(v_reinterpret_as_u32(e), v_reinterpret_as_u32(o), lo, hi);
        v_store(D, v_reinterpret_as_u8(lo)); v_store(D + n, v_reinterpret_as_u8(hi));
    }
}

static inline void linear2x_block_packed(v_uint8& e0, v_uint8& o0, v_uint8& e1, v_uint8& o1,
                                         const uchar* U, const uchar* M, const uchar* B, int x, int cn)
{
    linear2x_core_u8(vx_load(U + x - cn), vx_load(U + x), vx_load(U + x + cn),
                     vx_load(M + x - cn), vx_load(M + x), vx_load(M + x + cn),
                     vx_load(B + x - cn), vx_load(B + x), vx_load(B + x + cn), e0, o0, e1, o1);
}

// Vectorizes pixels [1, sw-1); returns the first pixel left for the scalar tail. The last
// block is shifted back to end at pixel sw-2, overlapping the previous one.
static inline int linear2x_row_pair_packed(uchar* D0, uchar* D1, const uchar* U, const uchar* M,
                                           const uchar* B, int sw, int cn)
{
    const int n = VTraits<v_uint8>::vlanes(), last = sw * cn - cn - n;
    if( last < cn )
        return 1;
    for( int x = cn; ; x += n )
    {
        x = std::min(x, last);
        v_uint8 e0, o0, e1, o1;
        linear2x_block_packed(e0, o0, e1, o1, U, M, B, x, cn);
        linear2x_store_pair(D0 + 2 * x, e0, o0, cn);
        linear2x_store_pair(D1 + 2 * x, e1, o1, cn);
        if( x == last )
            break;
    }
    return sw - 1;
}

// Same arithmetic on packed bytes (neighbours at +-3); only the outputs are split into
// planes so even/odd pixels can be interleaved.
static inline int linear2x_row_pair_cn3(uchar* D0, uchar* D1, const uchar* U, const uchar* M,
                                        const uchar* B, int sw)
{
    const int n = VTraits<v_uint8>::vlanes(), last = sw - 1 - n;
    if( last < 1 )
        return 1;
    uchar CV_DECL_ALIGNED(CV_SIMD_WIDTH) buf[4][VTraits<v_uint8>::max_nlanes * 3];
    for( int k = 1; ; k += n )
    {
        k = std::min(k, last);
        for( int v = 0; v < 3; v++ )
        {
            v_uint8 e0, o0, e1, o1;
            linear2x_block_packed(e0, o0, e1, o1, U, M, B, k * 3 + v * n, 3);
            v_store(buf[0] + v * n, e0);
            v_store(buf[1] + v * n, o0);
            v_store(buf[2] + v * n, e1);
            v_store(buf[3] + v * n, o1);
        }
        for( int r = 0; r < 2; r++ )
        {
            uchar* D = (r == 0 ? D0 : D1) + 6 * k;
            v_uint8 e[3], o[3], lo[3], hi[3];
            v_load_deinterleave(buf[2 * r], e[0], e[1], e[2]);
            v_load_deinterleave(buf[2 * r + 1], o[0], o[1], o[2]);
            for( int c = 0; c < 3; c++ )
                v_zip(e[c], o[c], lo[c], hi[c]);
            v_store_interleave(D, lo[0], lo[1], lo[2]);
            v_store_interleave(D + 3 * n, hi[0], hi[1], hi[2]);
        }
        if( k == last )
            break;
    }
    return sw - 1;
}

class resizeLinear2xFused8u_Invoker : public ParallelLoopBody
{
public:
    resizeLinear2xFused8u_Invoker(const Mat& _src, Mat& _dst) : src(_src), dst(_dst) {}

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        const int cn = src.channels(), sw = src.cols, sh = src.rows;

        for( int sy = range.start; sy < range.end; ++sy )
        {
            uchar* D0 = dst.ptr(2 * sy);
            uchar* D1 = dst.ptr(2 * sy + 1);
            const uchar* U = src.ptr(std::max(sy - 1, 0));
            const uchar* M = src.ptr(sy);
            const uchar* B = src.ptr(std::min(sy + 1, sh - 1));

            linear2x_scalar_px(D0, D1, U, M, B, 0, sw, cn);
            int k = cn == 3 ? linear2x_row_pair_cn3(D0, D1, U, M, B, sw)
                            : linear2x_row_pair_packed(D0, D1, U, M, B, sw, cn);
            for( ; k < sw; k++ )
                linear2x_scalar_px(D0, D1, U, M, B, k, sw, cn);
        }
    }

private:
    const Mat& src;
    Mat& dst;
};

static void resizeLinear2xUp_8u(const Mat& src, Mat& dst)
{
    Range range(0, src.rows);
    resizeLinear2xFused8u_Invoker invoker(src, dst);
    parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
}

#endif // (CV_SIMD || CV_SIMD_SCALABLE)

}  // namespace linear_up_2x_detail

void resize_cpu(int src_type,
            const uchar * src_data, size_t src_step, int src_width, int src_height,
            uchar * dst_data, size_t dst_step, int dst_width, int dst_height,
            double inv_scale_x, double inv_scale_y, int interpolation)
{
    CV_INSTRUMENT_REGION();

    CV_Assert((dst_width > 0 && dst_height > 0) || (inv_scale_x > 0 && inv_scale_y > 0));
    if (inv_scale_x < DBL_EPSILON || inv_scale_y < DBL_EPSILON)
    {
        inv_scale_x = static_cast<double>(dst_width) / src_width;
        inv_scale_y = static_cast<double>(dst_height) / src_height;
    }

    int  depth = CV_MAT_DEPTH(src_type), cn = CV_MAT_CN(src_type);
    Size dsize = Size(saturate_cast<int>(src_width*inv_scale_x),
                        saturate_cast<int>(src_height*inv_scale_y));
    CV_Assert( !dsize.empty() );

    Mat src(Size(src_width, src_height), src_type, const_cast<uchar*>(src_data), src_step);
    Mat dst(dsize, src_type, dst_data, dst_step);

    if( interpolation == INTER_NEAREST )
    {
        resizeNN_( src, dst, inv_scale_x, inv_scale_y );
        return;
    }

    if( interpolation == INTER_NEAREST_EXACT )
    {
        resizeNN_bitexact( src, dst, inv_scale_x, inv_scale_y );
        return;
    }

#if (CV_SIMD || CV_SIMD_SCALABLE)
    if( interpolation == INTER_LINEAR && depth == CV_8U && cn <= 4 &&
        linear_up_2x_detail::is_2x_up(src_width, dsize.width, src_height, dsize.height,
                                      inv_scale_x, inv_scale_y) )
    {
        linear_up_2x_detail::resizeLinear2xUp_8u(src, dst);
        return;
    }
#endif

    static ResizeFunc linear_tab[] =
    {
        resizeGeneric_<
            HResizeLinear<uchar, int, short,
                INTER_RESIZE_COEF_SCALE,
                HResizeLinearVec_8u32s>,
            VResizeLinear<uchar, int, short,
                FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS*2>,
                VResizeLinearVec_32s8u> >,
        0,
        resizeGeneric_<
            HResizeLinear<ushort, float, float, 1,
                HResizeLinearVec_16u32f>,
            VResizeLinear<ushort, float, float, Cast<float, ushort>,
                VResizeLinearVec_32f16u> >,
        resizeGeneric_<
            HResizeLinear<short, float, float, 1,
                HResizeLinearVec_16s32f>,
            VResizeLinear<short, float, float, Cast<float, short>,
                VResizeLinearVec_32f16s> >,
        0,
        resizeGeneric_<
            HResizeLinear<float, float, float, 1,
                HResizeLinearVec_32f>,
            VResizeLinear<float, float, float, Cast<float, float>,
                VResizeLinearVec_32f> >,
        resizeGeneric_<
            HResizeLinear<double, double, float, 1,
                HResizeNoVec>,
            VResizeLinear<double, double, float, Cast<double, double>,
                VResizeNoVec> >,
        0
    };

    static ResizeFunc cubic_tab[] =
    {
        resizeGeneric_<
            HResizeCubic<uchar, int, short>,
            VResizeCubic<uchar, int, short,
                FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS*2>,
                VResizeCubicVec_32s8u> >,
        0,
        resizeGeneric_<
            HResizeCubic<ushort, float, float>,
            VResizeCubic<ushort, float, float, Cast<float, ushort>,
            VResizeCubicVec_32f16u> >,
        resizeGeneric_<
            HResizeCubic<short, float, float>,
            VResizeCubic<short, float, float, Cast<float, short>,
            VResizeCubicVec_32f16s> >,
        0,
        resizeGeneric_<
            HResizeCubic<float, float, float>,
            VResizeCubic<float, float, float, Cast<float, float>,
            VResizeCubicVec_32f> >,
        resizeGeneric_<
            HResizeCubic<double, double, float>,
            VResizeCubic<double, double, float, Cast<double, double>,
            VResizeNoVec> >,
        0
    };

    static ResizeFunc lanczos4_tab[] =
    {
        resizeGeneric_<HResizeLanczos4<uchar, int, short>,
            VResizeLanczos4<uchar, int, short,
            FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS*2>,
            VResizeNoVec> >,
        0,
        resizeGeneric_<HResizeLanczos4<ushort, float, float>,
            VResizeLanczos4<ushort, float, float, Cast<float, ushort>,
            VResizeLanczos4Vec_32f16u> >,
        resizeGeneric_<HResizeLanczos4<short, float, float>,
            VResizeLanczos4<short, float, float, Cast<float, short>,
            VResizeLanczos4Vec_32f16s> >,
        0,
        resizeGeneric_<HResizeLanczos4<float, float, float>,
            VResizeLanczos4<float, float, float, Cast<float, float>,
            VResizeLanczos4Vec_32f> >,
        resizeGeneric_<HResizeLanczos4<double, double, float>,
            VResizeLanczos4<double, double, float, Cast<double, double>,
            VResizeNoVec> >,
        0
    };

    static ResizeAreaFastFunc areafast_tab[] =
    {
        resizeAreaFast_<uchar, int, ResizeAreaFastVec<uchar, ResizeAreaFastVec_SIMD_8u> >,
        0,
        resizeAreaFast_<ushort, float, ResizeAreaFastVec<ushort, ResizeAreaFastVec_SIMD_16u> >,
        resizeAreaFast_<short, float, ResizeAreaFastVec<short, ResizeAreaFastVec_SIMD_16s> >,
        0,
        resizeAreaFast_<float, float, ResizeAreaFastVec<float, ResizeAreaFastVec_SIMD_32f> >,
        resizeAreaFast_<double, double, ResizeAreaFastNoVec<double, double> >,
        0
    };

    static ResizeAreaFunc area_tab[] =
    {
        resizeAreaVFirst_<uchar>, 0, resizeAreaVFirst_<ushort>,
        resizeAreaVFirst_<short>, 0, resizeAreaVFirst_<float>,
        resizeArea_<double, double>, 0
    };

    double scale_x = 1./inv_scale_x, scale_y = 1./inv_scale_y;

    int iscale_x = saturate_cast<int>(scale_x);
    int iscale_y = saturate_cast<int>(scale_y);

    bool is_area_fast = std::abs(scale_x - iscale_x) < DBL_EPSILON &&
            std::abs(scale_y - iscale_y) < DBL_EPSILON;

    int k, sx, sy, dx, dy;


    {
        // in case of scale_x && scale_y is equal to 2
        // INTER_AREA (fast) also is equal to INTER_LINEAR
        if( interpolation == INTER_LINEAR && is_area_fast && iscale_x == 2 && iscale_y == 2 )
            interpolation = INTER_AREA;

        // true "area" interpolation is only implemented for the case (scale_x >= 1 && scale_y >= 1).
        // In other cases it is emulated using some variant of bilinear interpolation
        if( interpolation == INTER_AREA && scale_x >= 1 && scale_y >= 1 )
        {
            if( is_area_fast )
            {
                int area = iscale_x*iscale_y;
                size_t srcstep = src_step / src.elemSize1();
                AutoBuffer<int> _ofs(area + dsize.width*cn);
                int* ofs = _ofs.data();
                int* xofs = ofs + area;
                ResizeAreaFastFunc func = areafast_tab[depth];
                CV_Assert( func != 0 );

                for( sy = 0, k = 0; sy < iscale_y; sy++ )
                    for( sx = 0; sx < iscale_x; sx++ )
                        ofs[k++] = (int)(sy*srcstep + sx*cn);

                for( dx = 0; dx < dsize.width; dx++ )
                {
                    int j = dx * cn;
                    sx = iscale_x * j;
                    for( k = 0; k < cn; k++ )
                        xofs[j + k] = sx + k;
                }

                func( src, dst, ofs, xofs, iscale_x, iscale_y );
                return;
            }

            ResizeAreaFunc func = area_tab[depth];
            CV_Assert( func != 0 );

            AutoBuffer<DecimateAlpha> _xytab((src_width + src_height)*2);
            DecimateAlpha* xtab = _xytab.data(), *ytab = xtab + src_width*2;
            int xtab_size = 0, ytab_size = 0;

            ytab_size = computeResizeAreaTab(src_height, dsize.height, 1, scale_y, ytab);

            AutoBuffer<int> _tabofs(dsize.height + 1);
            int* tabofs = _tabofs.data();
            for( k = 0, dy = 0; k < ytab_size; k++ )
            {
                if( k == 0 || ytab[k].di != ytab[k-1].di )
                {
                    CV_Assert( ytab[k].di == dy );
                    tabofs[dy++] = k;
                }
            }
            tabofs[dy] = ytab_size;

            xtab_size = computeResizeAreaTab(src_width, dsize.width, cn, scale_x, xtab);
            func( src, dst, xtab, xtab_size, ytab, ytab_size, tabofs );
            return;
        }
    }

    int xmin = 0, xmax = dsize.width, width = dsize.width*cn;
    bool area_mode = interpolation == INTER_AREA;
    bool fixpt = depth == CV_8U;
    float fx, fy;
    ResizeFunc func=0;
    int ksize=0, ksize2;
    if( interpolation == INTER_CUBIC )
        ksize = 4, func = cubic_tab[depth];
    else if( interpolation == INTER_LANCZOS4 )
        ksize = 8, func = lanczos4_tab[depth];
    else if( interpolation == INTER_LINEAR || interpolation == INTER_AREA )
        ksize = 2, func = linear_tab[depth];
    else
        CV_Error( cv::Error::StsBadArg, "Unknown interpolation method" );
    ksize2 = ksize/2;

    CV_Assert( func != 0 );

    AutoBuffer<uchar> _buffer((width + dsize.height)*(sizeof(int) + sizeof(float)*ksize));
    int* xofs = (int*)_buffer.data();
    int* yofs = xofs + width;
    float* alpha = (float*)(yofs + dsize.height);
    short* ialpha = (short*)alpha;
    float* beta = alpha + width*ksize;
    short* ibeta = ialpha + width*ksize;
    float cbuf[MAX_ESIZE] = {0};

    for( dx = 0; dx < dsize.width; dx++ )
    {
        if( !area_mode )
        {
            fx = (float)((dx+0.5)*scale_x - 0.5);
            sx = cvFloor(fx);
            fx -= sx;
        }
        else
        {
            sx = cvFloor(dx*scale_x);
            fx = (float)((dx+1) - (sx+1)*inv_scale_x);
            fx = fx <= 0 ? 0.f : fx - cvFloor(fx);
        }

        if( sx < ksize2-1 )
        {
            xmin = dx+1;
            if( sx < 0 && (interpolation != INTER_CUBIC && interpolation != INTER_LANCZOS4))
                fx = 0, sx = 0;
        }

        if( sx + ksize2 >= src_width )
        {
            xmax = std::min( xmax, dx );
            if( sx >= src_width-1 && (interpolation != INTER_CUBIC && interpolation != INTER_LANCZOS4))
                fx = 0, sx = src_width-1;
        }

        for( k = 0, sx *= cn; k < cn; k++ )
            xofs[dx*cn + k] = sx + k;

        if( interpolation == INTER_CUBIC )
            interpolateCubic( fx, cbuf );
        else if( interpolation == INTER_LANCZOS4 )
            interpolateLanczos4( fx, cbuf );
        else
        {
            cbuf[0] = 1.f - fx;
            cbuf[1] = fx;
        }
        if( fixpt )
        {
            for( k = 0; k < ksize; k++ )
                ialpha[dx*cn*ksize + k] = saturate_cast<short>(cbuf[k]*INTER_RESIZE_COEF_SCALE);
            for( ; k < cn*ksize; k++ )
                ialpha[dx*cn*ksize + k] = ialpha[dx*cn*ksize + k - ksize];
        }
        else
        {
            for( k = 0; k < ksize; k++ )
                alpha[dx*cn*ksize + k] = cbuf[k];
            for( ; k < cn*ksize; k++ )
                alpha[dx*cn*ksize + k] = alpha[dx*cn*ksize + k - ksize];
        }
    }

    for( dy = 0; dy < dsize.height; dy++ )
    {
        if( !area_mode )
        {
            fy = (float)((dy+0.5)*scale_y - 0.5);
            sy = cvFloor(fy);
            fy -= sy;
        }
        else
        {
            sy = cvFloor(dy*scale_y);
            fy = (float)((dy+1) - (sy+1)*inv_scale_y);
            fy = fy <= 0 ? 0.f : fy - cvFloor(fy);
        }

        yofs[dy] = sy;
        if( interpolation == INTER_CUBIC )
            interpolateCubic( fy, cbuf );
        else if( interpolation == INTER_LANCZOS4 )
            interpolateLanczos4( fy, cbuf );
        else
        {
            cbuf[0] = 1.f - fy;
            cbuf[1] = fy;
        }

        if( fixpt )
        {
            for( k = 0; k < ksize; k++ )
                ibeta[dy*ksize + k] = saturate_cast<short>(cbuf[k]*INTER_RESIZE_COEF_SCALE);
        }
        else
        {
            for( k = 0; k < ksize; k++ )
                beta[dy*ksize + k] = cbuf[k];
        }
    }

    func( src, dst, xofs, fixpt ? (void*)ialpha : (void*)alpha, yofs,
          fixpt ? (void*)ibeta : (void*)beta, xmin, xmax, ksize );
}


#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END

} // namespace cv
