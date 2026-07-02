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

namespace nn_rational_detail {

static inline bool is_rational_up(int src_w, int dst_w, int src_h, int dst_h, int num, int den)
{
    return src_w > 0 && dst_w > src_w && dst_w * den == src_w * num &&
           src_h > 0 && dst_h > src_h && dst_h * den == src_h * num;
}

// One element T represents a whole pixel: ushort for 2x u8 channels (cn2),
// uint32_t for 4x u8 channels (cn4). The replication pattern is identical.
template<typename T> static inline void nn_row_up_13_10(T* D, const T* S, int sw)
{
    for( int sx = 0; sx < sw; sx += 10 )
    {
        const T s0 = S[sx], s1 = S[sx + 1], s2 = S[sx + 2], s3 = S[sx + 3], s4 = S[sx + 4];
        const T s5 = S[sx + 5], s6 = S[sx + 6], s7 = S[sx + 7], s8 = S[sx + 8], s9 = S[sx + 9];
        D[0] = s0; D[1] = s0;
        D[2] = s1;
        D[3] = s2;
        D[4] = s3; D[5] = s3;
        D[6] = s4;
        D[7] = s5;
        D[8] = s6; D[9] = s6;
        D[10] = s7;
        D[11] = s8;
        D[12] = s9;
        D += 13;
    }
}

template<typename T> static inline void nn_row_up_12_5(T* D, const T* S, int sw)
{
    for( int sx = 0; sx < sw; sx += 5 )
    {
        const T v0 = S[sx], v1 = S[sx + 1], v2 = S[sx + 2], v3 = S[sx + 3], v4 = S[sx + 4];
        D[0] = v0; D[1] = v0; D[2] = v0;
        D[3] = v1; D[4] = v1;
        D[5] = v2; D[6] = v2; D[7] = v2;
        D[8] = v3; D[9] = v3;
        D[10] = v4; D[11] = v4;
        D += 12;
    }
}

template<typename T> static inline void nn_row_up_17_5(T* D, const T* S, int sw)
{
    for( int sx = 0; sx < sw; sx += 5 )
    {
        const T v0 = S[sx], v1 = S[sx + 1], v2 = S[sx + 2], v3 = S[sx + 3], v4 = S[sx + 4];
        D[0] = v0; D[1] = v0; D[2] = v0; D[3] = v0;
        D[4] = v1; D[5] = v1; D[6] = v1;
        D[7] = v2; D[8] = v2; D[9] = v2; D[10] = v2;
        D[11] = v3; D[12] = v3; D[13] = v3;
        D[14] = v4; D[15] = v4; D[16] = v4;
        D += 17;
    }
}

template<typename T, void (*RowFunc)(T*, const T*, int)>
class resizeNNRationalUpInvoker_ : public ParallelLoopBody
{
public:
    resizeNNRationalUpInvoker_(const Mat& _src, Mat &_dst) : src(_src), dst(_dst) {}

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        const int sw = src.cols, sh = src.rows, dh = dst.rows;
        for( int y = range.start; y < range.end; y++ )
        {
            T* D = (T*)dst.ptr(y);
            const int sy = (y * sh) / dh;
            const T* S = (const T*)src.ptr(sy);
            RowFunc(D, S, sw);
        }
    }

private:
    const Mat& src;
    Mat& dst;

    resizeNNRationalUpInvoker_(const resizeNNRationalUpInvoker_&);
    resizeNNRationalUpInvoker_& operator=(const resizeNNRationalUpInvoker_&);
};

template<typename T, void (*RowFunc)(T*, const T*, int)>
static void resizeNNRationalUp_(const Mat& src, Mat& dst)
{
    Range range(0, dst.rows);
    resizeNNRationalUpInvoker_<T, RowFunc> invoker(src, dst);
    parallel_for_(range, invoker, dst.total()/(double)(1<<16));
}

}  // namespace nn_rational_detail

namespace nn_bitexact_detail {
static void resizeNN_bitexact2xUp_(const Mat& src, Mat& dst);
}

class resizeNNUpPixInvoker_ : public ParallelLoopBody
{
public:
    resizeNNUpPixInvoker_(const Mat& _src, Mat &_dst, const int* _sx_ofs, double _ify, int _pix_size) :
        ParallelLoopBody(), src(_src), dst(_dst), sx_ofs(_sx_ofs), ify(_ify), pix_size(_pix_size)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size ssize = src.size(), dsize = dst.size();
        const int width = dsize.width;

        for( int y = range.start; y < range.end; y++ )
        {
            uchar* D = dst.ptr(y);
            const int sy = std::min(cvFloor(y*ify), ssize.height-1);
            const uchar* S = src.ptr(sy);
            int x = 0;

            if( pix_size == 2 )
            {
                const ushort* S16 = (const ushort*)S;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                for( ; x <= width - VTraits<v_uint16>::vlanes(); x += VTraits<v_uint16>::vlanes() )
                    v_store((ushort*)D + x, vx_lut_pairs(S16, sx_ofs + x));
#endif
                for( ; x < width; x++ )
                    ((ushort*)D)[x] = S16[sx_ofs[x]];
            }
            else if( pix_size == 4 )
            {
                if( src.depth() == CV_32F )
                {
                    const float* SF = (const float*)S;
                    float* DF = (float*)D;
                    for( ; x < width; x++ )
                        DF[x] = SF[sx_ofs[x]];
                }
                else
                {
                    const uint32_t* S32 = (const uint32_t*)S;
                    for( ; x < width; x++ )
                        ((uint32_t*)D)[x] = S32[sx_ofs[x]];
                }
            }
        }
    }

private:
    const Mat& src;
    Mat& dst;
    const int* sx_ofs;
    double ify;
    int pix_size;

    resizeNNUpPixInvoker_(const resizeNNUpPixInvoker_&);
    resizeNNUpPixInvoker_& operator=(const resizeNNUpPixInvoker_&);
};

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

        for( int y = range.start; y < range.end; y++ )
        {
            uchar* D = dst.ptr(y);
            const int sy = std::min(cvFloor(y*ify), ssize.height-1);
            const uchar* S = src.ptr(sy);
            int x = 0;

            switch( pix_size )
            {
            case 1:
#if (CV_SIMD || CV_SIMD_SCALABLE)
#if CV_AVX_512VBMI
                for( ; x <= width - VTraits<v_uint8>::vlanes(); x += VTraits<v_uint8>::vlanes() )
                    v_store(D + x, vx_lut_u8_byteofs(S, x_ofs + x));
#else
                for( ; x <= width - VTraits<v_uint8>::vlanes(); x += VTraits<v_uint8>::vlanes() )
                    v_store(D + x, vx_lut(S, x_ofs + x));
#endif
#endif
                for( ; x < width; x++ )
                    D[x] = S[x_ofs[x]];
                break;
            case 2:
#if (CV_SIMD || CV_SIMD_SCALABLE)
#if CV_SIMD256 || CV_SIMD512
                for( ; x <= width - VTraits<v_uint16>::vlanes(); x += VTraits<v_uint16>::vlanes() )
                    v_store((ushort*)D + x, vx_lut_u16_byteofs(S, x_ofs + x));
#else
                for( ; x <= width - VTraits<v_uint16>::vlanes(); x += VTraits<v_uint16>::vlanes() )
                    v_store((ushort*)D + x, v_reinterpret_as_u16(vx_lut_pairs(S, x_ofs + x)));
#endif
#endif
                for( ; x < width; x++ )
                    ((ushort*)D)[x] = *(const ushort*)(S + x_ofs[x]);
                break;
            case 3:
                for( ; x < width; x++, D += 3 )
                {
                    const uchar* _tS = S + x_ofs[x];
                    D[0] = _tS[0]; D[1] = _tS[1]; D[2] = _tS[2];
                }
                break;
            case 4:
#if (CV_SIMD || CV_SIMD_SCALABLE)
#if CV_SIMD256 || CV_SIMD512
                for( ; x <= width - VTraits<v_uint32>::vlanes(); x += VTraits<v_uint32>::vlanes() )
                    v_store((uint32_t*)D + x, vx_lut_u32_byteofs(S, x_ofs + x));
#else
                for( ; x <= width - VTraits<v_uint32>::vlanes(); x += VTraits<v_uint32>::vlanes() )
                {
                    int CV_DECL_ALIGNED(CV_SIMD_WIDTH) sx_idx[VTraits<v_uint32>::max_nlanes];
                    for( int i = 0; i < VTraits<v_uint32>::vlanes(); i++ )
                        sx_idx[i] = x_ofs[x + i] >> 2;
                    v_store((uint32_t*)D + x, vx_lut((const uint32_t*)S, sx_idx));
                }
#endif
#endif
                for( ; x < width; x++ )
                    ((uint32_t*)D)[x] = *(const uint32_t*)(S + x_ofs[x]);
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
#if (CV_SIMD || CV_SIMD_SCALABLE)
                for( ; x <= width - VTraits<v_uint64>::vlanes(); x += VTraits<v_uint64>::vlanes() )
                {
                    int CV_DECL_ALIGNED(64) sx_idx[VTraits<v_uint64>::max_nlanes];
                    for( int i = 0; i < VTraits<v_uint64>::vlanes(); i++ )
                        sx_idx[i] = x_ofs[x + i] >> 3;
                    v_store((uint64_t*)D + x, vx_lut((const uint64_t*)S, sx_idx));
                }
#endif
                for( ; x < width; x++ )
                    ((uint64_t*)D)[x] = *(const uint64_t*)(S + x_ofs[x]);
                break;
            case 12:
                for( ; x < width; x++, D += 12 )
                {
                    const int* _tS = (const int*)(S + x_ofs[x]);
                    int* _tD = (int*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2];
                }
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

    using namespace nn_rational_detail;

    if( pix_size == 2 )
    {
        if( is_rational_up(ssize.width, dsize.width, ssize.height, dsize.height, 12, 5) )
        {
            resizeNNRationalUp_<ushort, nn_row_up_12_5<ushort> >(src, dst);
            return;
        }
        if( is_rational_up(ssize.width, dsize.width, ssize.height, dsize.height, 17, 5) )
        {
            resizeNNRationalUp_<ushort, nn_row_up_17_5<ushort> >(src, dst);
            return;
        }
        if( is_rational_up(ssize.width, dsize.width, ssize.height, dsize.height, 13, 10) )
        {
            resizeNNRationalUp_<ushort, nn_row_up_13_10<ushort> >(src, dst);
            return;
        }
    }
    else if( pix_size == 4 )
    {
        if( is_rational_up(ssize.width, dsize.width, ssize.height, dsize.height, 12, 5) )
        {
            resizeNNRationalUp_<uint32_t, nn_row_up_12_5<uint32_t> >(src, dst);
            return;
        }
        if( is_rational_up(ssize.width, dsize.width, ssize.height, dsize.height, 17, 5) )
        {
            resizeNNRationalUp_<uint32_t, nn_row_up_17_5<uint32_t> >(src, dst);
            return;
        }
        if( is_rational_up(ssize.width, dsize.width, ssize.height, dsize.height, 13, 10) )
        {
            resizeNNRationalUp_<uint32_t, nn_row_up_13_10<uint32_t> >(src, dst);
            return;
        }
    }

    if( fx >= 1.0 && pix_size == 4 )
    {
        AutoBuffer<int> sx_ofs(dsize.width);
        for( int x = 0; x < dsize.width; x++ )
        {
            const int sx = cvFloor(x*ifx);
            sx_ofs[x] = std::min(sx, ssize.width-1);
        }

        Range range(0, dsize.height);
        resizeNNUpPixInvoker_ invoker(src, dst, sx_ofs.data(), ify, pix_size);
        parallel_for_(range, invoker, dst.total()/(double)(1<<16));
        return;
    }

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

enum NNBitexactMode { GENERIC = 0, UP_2X = 1, DOWN_2X = 2, DOWN_4X = 3 };

static inline NNBitexactMode detect_h_mode(int sw, int dw, int ifx, int ifx0)
{
    if (dw == 2 * sw && sw > 0 && ifx == (1 << 15) && ifx0 == (1 << 15))
        return UP_2X;
    if (dw * 2 == sw && sw % 2 == 0 && ifx == (2 << 16) && ifx0 == (1 << 16))
        return DOWN_2X;
    if (dw * 4 == sw && sw % 2 == 0 && ifx == (4 << 16) && ifx0 == (2 << 16))
        return DOWN_4X;
    return GENERIC;
}

static inline NNBitexactMode detect_v_mode(int sh, int dh, int ify, int ify0)
{
    if (dh == 2 * sh && ify == (1 << 15) && ify0 == (1 << 15))
        return UP_2X;
    if (dh * 2 == sh && sh % 2 == 0 && ify == (2 << 16) && ify0 == (1 << 16))
        return DOWN_2X;
    if (dh * 4 == sh && sh % 2 == 0 && ify == (4 << 16) && ify0 == (2 << 16))
        return DOWN_4X;
    return GENERIC;
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

static inline void nn_bitexact_row_cn3_down(uchar* D, const uchar* S, int dw, int sx_mul, int sx_add, int sw)
{
    const int smax = sw - 1;
    int x = 0;
    for (; x <= dw - 4; x += 4, D += 12)
    {
        const int o0 = std::min(sx_mul * x + sx_add, smax) * 3;
        const int o1 = std::min(sx_mul * (x + 1) + sx_add, smax) * 3;
        const int o2 = std::min(sx_mul * (x + 2) + sx_add, smax) * 3;
        const int o3 = std::min(sx_mul * (x + 3) + sx_add, smax) * 3;
        D[0] = S[o0]; D[1] = S[o0 + 1]; D[2] = S[o0 + 2];
        D[3] = S[o1]; D[4] = S[o1 + 1]; D[5] = S[o1 + 2];
        D[6] = S[o2]; D[7] = S[o2 + 1]; D[8] = S[o2 + 2];
        D[9] = S[o3]; D[10] = S[o3 + 1]; D[11] = S[o3 + 2];
    }
    for (; x < dw; ++x, D += 3)
    {
        const int off = std::min(sx_mul * x + sx_add, smax) * 3;
        D[0] = S[off]; D[1] = S[off + 1]; D[2] = S[off + 2];
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
                for (; x <= width - VTraits<v_uint8>::vlanes(); x += VTraits<v_uint8>::vlanes())
                    v_store(D + x, vx_lut_pairs(S, sx_ofs + x));
#endif
                for (; x < width; x++)
                    D[x] = S[sx_ofs[x]];
            }
            else if (pix_size == 2)
            {
                const ushort* S16 = (const ushort*)S;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                for (; x <= width - VTraits<v_uint16>::vlanes(); x += VTraits<v_uint16>::vlanes())
                    v_store((ushort*)D + x, vx_lut_pairs(S16, sx_ofs + x));
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
                for (; x <= width - VTraits<v_uint32>::vlanes(); x += VTraits<v_uint32>::vlanes())
                    v_store((uint32_t*)D + x, vx_lut_pairs(S32, sx_ofs + x));
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

class resizeNN_bitexactArithDownInvoker_ : public ParallelLoopBody
{
public:
    resizeNN_bitexactArithDownInvoker_(const Mat& _src, Mat& _dst, const int* _x_ofs,
                                       int _sx_mul, int _sx_add, int _sy_mul, int _sy_add) :
        src(_src), dst(_dst), x_ofs(_x_ofs),
        sx_mul(_sx_mul), sx_add(_sx_add), sy_mul(_sy_mul), sy_add(_sy_add) {}

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size ssize = src.size(), dsize = dst.size();
        const int pix_size = (int)src.elemSize();
        const int sh = ssize.height;

        for (int y = range.start; y < range.end; y++)
        {
            const int sy = std::min(sy_mul * y + sy_add, sh - 1);
            uchar* D = dst.ptr(y);
            const uchar* S = src.ptr(sy);
            int x = 0;

            switch (pix_size)
            {
            case 1:
#if (CV_SIMD || CV_SIMD_SCALABLE)
#if CV_AVX_512VBMI
                for (; x <= dsize.width - VTraits<v_uint8>::vlanes(); x += VTraits<v_uint8>::vlanes())
                    v_store(D + x, vx_lut_u8_byteofs(S, x_ofs + x));
#else
                for (; x <= dsize.width - VTraits<v_uint8>::vlanes(); x += VTraits<v_uint8>::vlanes())
                    v_store(D + x, vx_lut(S, x_ofs + x));
#endif
#endif
                for (; x < dsize.width; x++)
                    D[x] = S[x_ofs[x]];
                break;
            case 2:
#if (CV_SIMD || CV_SIMD_SCALABLE)
#if CV_SIMD256 || CV_SIMD512
                for (; x <= dsize.width - VTraits<v_uint16>::vlanes(); x += VTraits<v_uint16>::vlanes())
                    v_store((ushort*)D + x, vx_lut_u16_byteofs(S, x_ofs + x));
#else
                for (; x <= dsize.width - VTraits<v_uint16>::vlanes(); x += VTraits<v_uint16>::vlanes())
                    v_store((ushort*)D + x, v_reinterpret_as_u16(vx_lut_pairs(S, x_ofs + x)));
#endif
#endif
                for (; x < dsize.width; x++)
                    ((ushort*)D)[x] = *(const ushort*)(S + x_ofs[x]);
                break;
            case 3:
                nn_bitexact_row_cn3_down(D, S, dsize.width, sx_mul, sx_add, ssize.width);
                break;
            case 4:
#if (CV_SIMD || CV_SIMD_SCALABLE)
#if CV_SIMD256 || CV_SIMD512
                for (; x <= dsize.width - VTraits<v_uint32>::vlanes(); x += VTraits<v_uint32>::vlanes())
                    v_store((uint32_t*)D + x, vx_lut_u32_byteofs(S, x_ofs + x));
#else
                for (; x <= dsize.width - VTraits<v_uint32>::vlanes(); x += VTraits<v_uint32>::vlanes())
                {
                    int CV_DECL_ALIGNED(CV_SIMD_WIDTH) sx_idx[VTraits<v_uint32>::max_nlanes];
                    for (int i = 0; i < VTraits<v_uint32>::vlanes(); i++)
                        sx_idx[i] = x_ofs[x + i] >> 2;
                    v_store((uint32_t*)D + x, vx_lut((const uint32_t*)S, sx_idx));
                }
#endif
#endif
                for (; x < dsize.width; x++)
                    ((uint32_t*)D)[x] = *(const uint32_t*)(S + x_ofs[x]);
                break;
            default:
                for (; x < dsize.width; x++, D += pix_size)
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
    const int* x_ofs;
    int sx_mul, sx_add, sy_mul, sy_add;
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

static void resizeNN_bitexactArithDown_(const Mat& src, Mat& dst, int sx_mul, int sx_add, int sy_mul, int sy_add)
{
    const int dw = dst.cols, sw = src.cols, pix_size = (int)src.elemSize();
    const int smax = sw - 1;
    AutoBuffer<int> _x_ofs(dw);
    int* x_ofs = _x_ofs.data();
    for (int x = 0; x < dw; x++)
    {
        const int sx = std::min(sx_mul * x + sx_add, smax);
        x_ofs[x] = sx * pix_size;
    }

    Range range(0, dst.rows);
    resizeNN_bitexactArithDownInvoker_ invoker(src, dst, x_ofs, sx_mul, sx_add, sy_mul, sy_add);
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
#if (CV_SIMD || CV_SIMD_SCALABLE)
#if CV_AVX_512VBMI
                for( ; x <= dsize.width - VTraits<v_uint8>::vlanes(); x += VTraits<v_uint8>::vlanes() )
                    v_store(D + x, vx_lut_u8_byteofs(S, x_ofs + x));
#else
                for( ; x <= dsize.width - VTraits<v_uint8>::vlanes(); x += VTraits<v_uint8>::vlanes() )
                    v_store(D + x, vx_lut(S, x_ofs + x));
#endif
#endif
                for( ; x < dsize.width; x++ )
                    D[x] = S[x_ofs[x]];
                break;
            case 2:
#if (CV_SIMD || CV_SIMD_SCALABLE)
#if CV_SIMD256 || CV_SIMD512
                for( ; x <= dsize.width - VTraits<v_uint16>::vlanes(); x += VTraits<v_uint16>::vlanes() )
                    v_store((ushort*)D + x, vx_lut_u16_byteofs(S, x_ofs + x));
#else
                for( ; x <= dsize.width - VTraits<v_uint16>::vlanes(); x += VTraits<v_uint16>::vlanes() )
                    v_store((ushort*)D + x, v_reinterpret_as_u16(vx_lut_pairs(S, x_ofs + x)));
#endif
#endif
                for( ; x < dsize.width; x++ )
                    *((ushort*)D + x) = *(const ushort*)(S + x_ofs[x]);
                break;
            case 3:
                for( ; x < dsize.width; x++, D += 3 )
                {
                    const uchar* _tS = S + x_ofs[x];
                    D[0] = _tS[0]; D[1] = _tS[1]; D[2] = _tS[2];
                }
                break;
            case 4:
#if (CV_SIMD || CV_SIMD_SCALABLE)
#if CV_SIMD256 || CV_SIMD512
                for( ; x <= dsize.width - VTraits<v_uint32>::vlanes(); x += VTraits<v_uint32>::vlanes() )
                    v_store((uint32_t*)D + x, vx_lut_u32_byteofs(S, x_ofs + x));
#else
                for( ; x <= dsize.width - VTraits<v_uint32>::vlanes(); x += VTraits<v_uint32>::vlanes() )
                {
                    int CV_DECL_ALIGNED(CV_SIMD_WIDTH) sx_idx[VTraits<v_uint32>::max_nlanes];
                    for( int i = 0; i < VTraits<v_uint32>::vlanes(); i++ )
                        sx_idx[i] = x_ofs[x + i] >> 2;
                    v_store((uint32_t*)D + x, vx_lut((const uint32_t*)S, sx_idx));
                }
#endif
#endif
                for( ; x < dsize.width; x++ )
                    *((uint32_t*)D + x) = *(const uint32_t*)(S + x_ofs[x]);
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
#if (CV_SIMD || CV_SIMD_SCALABLE)
                for( ; x <= dsize.width - VTraits<v_uint64>::vlanes(); x += VTraits<v_uint64>::vlanes() )
                {
                    int CV_DECL_ALIGNED(CV_SIMD_WIDTH) sx_idx[VTraits<v_uint64>::max_nlanes];
                    for( int i = 0; i < VTraits<v_uint64>::vlanes(); i++ )
                        sx_idx[i] = x_ofs[x + i] >> 3;
                    v_store((uint64_t*)D + x, vx_lut((const uint64_t*)S, sx_idx));
                }
#endif
                for( ; x < dsize.width; x++ )
                    *((uint64_t*)D + x) = *(const uint64_t*)(S + x_ofs[x]);
                break;
            case 12:
                for( ; x < dsize.width; x++, D += 12 )
                {
                    const int* _tS = (const int*)(S + x_ofs[x]);
                    int* _tD = (int*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2];
                }
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

    const nn_bitexact_detail::NNBitexactMode hmode =
        nn_bitexact_detail::detect_h_mode(ssize.width, dsize.width, ifx, ifx0);
    const nn_bitexact_detail::NNBitexactMode vmode =
        nn_bitexact_detail::detect_v_mode(ssize.height, dsize.height, ify, ify0);

    if (hmode == nn_bitexact_detail::UP_2X && vmode == nn_bitexact_detail::UP_2X)
    {
        nn_bitexact_detail::resizeNN_bitexact2xUp_(src, dst);
        return;
    }
    if (hmode == nn_bitexact_detail::DOWN_2X && vmode == nn_bitexact_detail::DOWN_2X)
    {
        nn_bitexact_detail::resizeNN_bitexactArithDown_(src, dst, 2, 1, 2, 1);
        return;
    }
    if (hmode == nn_bitexact_detail::DOWN_4X && vmode == nn_bitexact_detail::DOWN_4X)
    {
        nn_bitexact_detail::resizeNN_bitexactArithDown_(src, dst, 4, 2, 4, 2);
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
    const int step = 4;
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
            v_int16x8 a = v_load((const short*)(alpha + dx * 2));
            v_store(D0 + dx, v_dotprod(v_reinterpret_as_s16(v_interleave_quads(v_load_expand(S0 + xofs[dx]))), a));
            v_store(D1 + dx, v_dotprod(v_reinterpret_as_s16(v_interleave_quads(v_load_expand(S1 + xofs[dx]))), a));
        }
    }
    for( ; k < count; k++ )
    {
        const uchar *S = src[k];
        int *D = dst[k];
        for( dx = 0; dx < len0; dx += step )
        {
            v_int16x8 a = v_load((const short*)(alpha + dx * 2));
            v_store(D + dx, v_dotprod(v_reinterpret_as_s16(v_interleave_quads(v_load_expand(S + xofs[dx]))), a));
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

static inline bool is_fast_scale(int scale_x, int scale_y, int cn)
{
    if (scale_x != scale_y || scale_x < 2 || (scale_x > 4 && scale_x != 10))
        return false;
    if (scale_x == 2)
        return cn == 1 || cn == 3 || cn == 4;
    return cn == 1 || cn == 3 || cn == 4;
}

template<typename T>
static inline T area_fast_round(int sum, int area)
{
    if (area == 4)
        return saturate_cast<T>((sum + 2) >> 2);
    if (area == 100)
        return saturate_cast<T>((sum + 50) / 100);
    return saturate_cast<T>(sum * (1.f / area));
}

template<typename T>
static int area_fast_tail_nxn_cn1(const T* S, T* D, int w, int dx, int scale, int step)
{
    const int area = scale * scale;
    const T* rows[10] = { S, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    rows[1] = (const T*)((const uchar*)S + step);
    for (int ri = 2; ri < scale; ++ri)
        rows[ri] = (const T*)((const uchar*)rows[ri - 1] + step);

    for (; dx < w; ++dx)
    {
        const int sx = dx * scale;
        int sum = 0;
        for (int ry = 0; ry < scale; ++ry)
            for (int k = 0; k < scale; ++k)
                sum += rows[ry][sx + k];
        D[dx] = area_fast_round<T>(sum, area);
    }
    return dx;
}

template<typename T>
static int area_fast_tail_nxn_cn3(const T* S, T* D, int w, int dx, int scale, int step)
{
    const int area = scale * scale;
    const T* rows[10] = { S, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    rows[1] = (const T*)((const uchar*)S + step);
    for (int ri = 2; ri < scale; ++ri)
        rows[ri] = (const T*)((const uchar*)rows[ri - 1] + step);

    for (; dx < w; dx += 3)
    {
        const int sx = dx * scale;
        for (int c = 0; c < 3; ++c)
        {
            int sum = 0;
            for (int ry = 0; ry < scale; ++ry)
                for (int k = 0; k < scale; ++k)
                    sum += rows[ry][sx + k * 3 + c];
            D[dx + c] = area_fast_round<T>(sum, area);
        }
    }
    return dx;
}

template<typename T>
static int area_fast_tail_nxn_cn4(const T* S, T* D, int w, int dx, int scale, int step)
{
    const int area = scale * scale;
    const T* rows[10] = { S, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    rows[1] = (const T*)((const uchar*)S + step);
    for (int ri = 2; ri < scale; ++ri)
        rows[ri] = (const T*)((const uchar*)rows[ri - 1] + step);

    for (; dx < w; dx += 4)
    {
        const int sx = dx * scale;
        for (int c = 0; c < 4; ++c)
        {
            int sum = 0;
            for (int ry = 0; ry < scale; ++ry)
                for (int k = 0; k < scale; ++k)
                    sum += rows[ry][sx + k * 4 + c];
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
    const T* nextS = (const T*)((const uchar*)S + step);
    if (cn == 1)
        for (; dx < w; ++dx)
        {
            const int index = dx * 2;
            D[dx] = (T)((S[index] + S[index + 1] + nextS[index] + nextS[index + 1] + 2) >> 2);
        }
    else if (cn == 3)
        for (; dx < w; dx += 3)
        {
            const int index = dx * 2;
            D[dx] = (T)((S[index] + S[index + 3] + nextS[index] + nextS[index + 3] + 2) >> 2);
            D[dx + 1] = (T)((S[index + 1] + S[index + 4] + nextS[index + 1] + nextS[index + 4] + 2) >> 2);
            D[dx + 2] = (T)((S[index + 2] + S[index + 5] + nextS[index + 2] + nextS[index + 5] + 2) >> 2);
        }
    else
    {
        CV_Assert(cn == 4);
        for (; dx < w; dx += 4)
        {
            const int index = dx * 2;
            D[dx] = (T)((S[index] + S[index + 4] + nextS[index] + nextS[index + 4] + 2) >> 2);
            D[dx + 1] = (T)((S[index + 1] + S[index + 5] + nextS[index + 1] + nextS[index + 5] + 2) >> 2);
            D[dx + 2] = (T)((S[index + 2] + S[index + 6] + nextS[index + 2] + nextS[index + 6] + 2) >> 2);
            D[dx + 3] = (T)((S[index + 3] + S[index + 7] + nextS[index + 3] + nextS[index + 7] + 2) >> 2);
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

static inline uint32_t area_fast_sum10(const uchar* q)
{
    return (uint32_t)q[0] + q[1] + q[2] + q[3] + q[4] +
           q[5] + q[6] + q[7] + q[8] + q[9];
}


static inline void area_fast_sum10_cn3_pixel(const uchar* q, uint32_t& sb, uint32_t& sg, uint32_t& sr)
{
    sb += q[0]+q[3]+q[6]+q[9]+q[12]+q[15]+q[18]+q[21]+q[24]+q[27];
    sg += q[1]+q[4]+q[7]+q[10]+q[13]+q[16]+q[19]+q[22]+q[25]+q[28];
    sr += q[2]+q[5]+q[8]+q[11]+q[14]+q[17]+q[20]+q[23]+q[26]+q[29];
}

static inline void area_fast_sum10_cn4_pixel(const uchar* q, uint32_t& sb, uint32_t& sg, uint32_t& sr, uint32_t& sa)
{
    sb += q[0]+q[4]+q[8]+q[12]+q[16]+q[20]+q[24]+q[28]+q[32]+q[36];
    sg += q[1]+q[5]+q[9]+q[13]+q[17]+q[21]+q[25]+q[29]+q[33]+q[37];
    sr += q[2]+q[6]+q[10]+q[14]+q[18]+q[22]+q[26]+q[30]+q[34]+q[38];
    sa += q[3]+q[7]+q[11]+q[15]+q[19]+q[23]+q[27]+q[31]+q[35]+q[39];
}

static inline int area_fast_10x10_pix_block()
{
    return 8;
}

#if CV_SIMD && !CV_NEON
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

static inline int area_fast_nxn_pix_block(int scale)
{
    const int blk = VTraits<v_uint8>::vlanes() / scale;
    return std::min(blk, VTraits<v_uint32>::vlanes());
}

static inline v_uint32 area_fast_hsum_u8_groups(const uchar* p, int scale, int n)
{
    CV_DECL_ALIGNED(64) uint32_t tmp[16];
    const int nmax = VTraits<v_uint32>::vlanes();
    int i = 0;

    if (scale == 10)
    {
        for (; i < n && i < nmax; ++i)
            tmp[i] = area_fast_sum10(p + (size_t)i * 10);
        for (; i < nmax; ++i)
            tmp[i] = 0;
        return vx_load(tmp);
    }

    for (i = 0; i < n && i < nmax; ++i)
    {
        int s = 0;
        const uchar* q = p + (size_t)i * scale;
        for (int k = 0; k < scale; ++k)
            s += q[k];
        tmp[i] = (uint32_t)s;
    }
    for (; i < nmax; ++i)
        tmp[i] = 0;
    return vx_load(tmp);
}

static inline v_uint32 area_fast_plane_row_quads(const v_uint8& plane)
{
    return area_fast_byte_sum_u32(v_reinterpret_as_u32(plane));
}

static inline void area_fast_store_bgra(uchar* dst, v_uint32 sb, v_uint32 sg, v_uint32 sr, v_uint32 sa)
{
    v_uint32 px = v_add(v_add(v_shr<4>(sb), v_shl<8>(v_shr<4>(sg))),
                        v_add(v_shl<16>(v_shr<4>(sr)), v_shl<24>(v_shr<4>(sa))));
    v_store(reinterpret_cast<uint32_t*>(dst), px);
}

static inline void area_fast_store_gray_scaled(uchar* dst, const v_uint32& sum, int n, int area)
{
    CV_DECL_ALIGNED(64) uint32_t tmp[16];
    if (area == 16)
        v_store(tmp, v_shr<4>(sum));
    else
    {
        v_store(tmp, sum);
        const int half = area / 2;
        for (int i = 0; i < n; ++i)
            dst[i] = (uchar)((tmp[i] + half) / area);
        return;
    }
    for (int i = 0; i < n; ++i)
        dst[i] = (uchar)tmp[i];
}

static inline void area_fast_store_gray(uchar* dst, const v_uint32& sum, int n)
{
    area_fast_store_gray_scaled(dst, sum, n, 16);
}

static inline void area_fast_store_rgb_scaled(uchar* dst, v_uint32 sb, v_uint32 sg, v_uint32 sr, int n, int area)
{
    const int half = area / 2;
    CV_DECL_ALIGNED(64) uint32_t b[16], g[16], r[16];
    if (area == 16)
    {
        v_store(b, v_shr<4>(sb));
        v_store(g, v_shr<4>(sg));
        v_store(r, v_shr<4>(sr));
    }
    else
    {
        v_store(b, sb);
        v_store(g, sg);
        v_store(r, sr);
        for (int i = 0; i < n; ++i)
        {
            b[i] = (b[i] + half) / area;
            g[i] = (g[i] + half) / area;
            r[i] = (r[i] + half) / area;
        }
    }
    uchar* p = dst;
    for (int i = 0; i < n; ++i)
    {
        p[0] = (uchar)b[i];
        p[1] = (uchar)g[i];
        p[2] = (uchar)r[i];
        p += 3;
    }
}

static inline void area_fast_store_bgra_scaled(uchar* dst, v_uint32 sb, v_uint32 sg, v_uint32 sr, v_uint32 sa, int n, int area)
{
    const int half = area / 2;
    CV_DECL_ALIGNED(64) uint32_t b[16], g[16], r[16], a[16];
    if (area == 16)
    {
        v_store(b, v_shr<4>(sb));
        v_store(g, v_shr<4>(sg));
        v_store(r, v_shr<4>(sr));
        v_store(a, v_shr<4>(sa));
    }
    else
    {
        v_store(b, sb);
        v_store(g, sg);
        v_store(r, sr);
        v_store(a, sa);
        for (int i = 0; i < n; ++i)
        {
            b[i] = (b[i] + half) / area;
            g[i] = (g[i] + half) / area;
            r[i] = (r[i] + half) / area;
            a[i] = (a[i] + half) / area;
        }
    }
    for (int i = 0; i < n; ++i)
    {
        uint32_t px = b[i] | (g[i] << 8) | (r[i] << 16) | (a[i] << 24);
        reinterpret_cast<uint32_t*>(dst)[i] = px;
    }
}

static inline void area_fast_store_rgb(uchar* dst, v_uint32 sb, v_uint32 sg, v_uint32 sr)
{
    const int n = VTraits<v_uint32>::vlanes();
    CV_DECL_ALIGNED(64) uint32_t b[16], g[16], r[16];
    v_store(b, v_shr<4>(sb));
    v_store(g, v_shr<4>(sg));
    v_store(r, v_shr<4>(sr));
    uchar* p = dst;
    for (int i = 0; i < n; ++i)
    {
        p[0] = (uchar)b[i];
        p[1] = (uchar)g[i];
        p[2] = (uchar)r[i];
        p += 3;
    }
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
    const int block = area_fast_4x4_pix_block();

    for (; dx <= w - block; dx += block)
    {
        const int sx = dx * 4;
        v_uint32 sum = vx_setzero_u32();
        for (int ri = 0; ri < 4; ++ri)
            sum = v_add(sum, area_fast_plane_row_quads(vx_load(rows[ri] + sx)));
        area_fast_store_gray(D + dx, sum, block);
    }
    return dx;
}

static int area_fast_u8_cn1_10x10(const uchar* S, uchar* D, int w, int step, int dx)
{
    const int scale = 10;
    const uchar* rows[10] = { S, S + step, S + 2 * step, S + 3 * step, S + 4 * step,
                              S + 5 * step, S + 6 * step, S + 7 * step, S + 8 * step, S + 9 * step };
    const int pixBlock = area_fast_10x10_pix_block();

    for (; dx <= w - pixBlock; dx += pixBlock)
    {
        const int sx = dx * scale;
        uint32_t sums[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

        for (int ri = 0; ri < scale; ++ri)
        {
            const uchar* row = rows[ri] + sx;
            sums[0] += area_fast_sum10(row + 0);
            sums[1] += area_fast_sum10(row + 10);
            sums[2] += area_fast_sum10(row + 20);
            sums[3] += area_fast_sum10(row + 30);
            sums[4] += area_fast_sum10(row + 40);
            sums[5] += area_fast_sum10(row + 50);
            sums[6] += area_fast_sum10(row + 60);
            sums[7] += area_fast_sum10(row + 70);
        }

        for (int i = 0; i < pixBlock; ++i)
            D[dx + i] = (uchar)((sums[i] + 50) / 100);
    }
    return dx;
}

static int area_fast_u8_cn3_10x10(const uchar* S, uchar* D, int w, int step, int dx)
{
    const int scale = 10;
    const int area = scale * scale;
    const uchar* rows[10] = { S, S + step, S + 2 * step, S + 3 * step, S + 4 * step,
                              S + 5 * step, S + 6 * step, S + 7 * step, S + 8 * step, S + 9 * step };
    const int pixBlock = area_fast_10x10_pix_block();
    const int chBlock = pixBlock * 3;
    const int half = area / 2;

    for (; dx <= w - chBlock; dx += chBlock)
    {
        const int sx = dx * scale;
        uint32_t sb[8] = {}, sg[8] = {}, sr[8] = {};

        for (int pi = 0; pi < pixBlock; ++pi)
        {
            for (int ri = 0; ri < scale; ++ri)
            {
                const uchar* q = rows[ri] + sx + pi * scale * 3;
                area_fast_sum10_cn3_pixel(q, sb[pi], sg[pi], sr[pi]);
            }
        }

        uchar* p = D + dx;
        for (int pi = 0; pi < pixBlock; ++pi)
        {
            p[0] = (uchar)((sb[pi] + half) / area);
            p[1] = (uchar)((sg[pi] + half) / area);
            p[2] = (uchar)((sr[pi] + half) / area);
            p += 3;
        }
    }
    return dx;
}

static int area_fast_u8_cn4_10x10(const uchar* S, uchar* D, int w, int step, int dx)
{
    const int scale = 10;
    const int area = scale * scale;
    const uchar* rows[10] = { S, S + step, S + 2 * step, S + 3 * step, S + 4 * step,
                              S + 5 * step, S + 6 * step, S + 7 * step, S + 8 * step, S + 9 * step };
    const int pixBlock = area_fast_10x10_pix_block();
    const int chBlock = pixBlock * 4;
    const int half = area / 2;

    for (; dx <= w - chBlock; dx += chBlock)
    {
        const int sx = dx * scale;
        uint32_t sb[8] = {}, sg[8] = {}, sr[8] = {}, sa[8] = {};

        for (int pi = 0; pi < pixBlock; ++pi)
        {
            for (int ri = 0; ri < scale; ++ri)
            {
                const uchar* q = rows[ri] + sx + pi * scale * 4;
                area_fast_sum10_cn4_pixel(q, sb[pi], sg[pi], sr[pi], sa[pi]);
            }
        }

        uint32_t* p = reinterpret_cast<uint32_t*>(D + dx);
        for (int pi = 0; pi < pixBlock; ++pi)
        {
            const uint32_t b = (sb[pi] + half) / area;
            const uint32_t g = (sg[pi] + half) / area;
            const uint32_t r = (sr[pi] + half) / area;
            const uint32_t a = (sa[pi] + half) / area;
            p[pi] = b | (g << 8) | (r << 16) | (a << 24);
        }
    }
    return dx;
}
#endif // CV_SIMD && !CV_NEON

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
        if (scale_x != scale_y || scale_x < 2 || (scale_x > 4 && scale_x != 10))
            return 0;

        int dx = 0;
        const uchar* S0 = S, * S1 = S0 + step;

        uint16x8_t v_2 = vdupq_n_u16(2);
        const uint32x4_t mask = vdupq_n_u32(0xff);

        auto neon_byte_sum_u32 = [&](uint32x4_t v) {
            return vaddq_u32(vaddq_u32(vshrq_n_u32(v, 24), vandq_u32(vshrq_n_u32(v, 16), mask)),
                             vaddq_u32(vandq_u32(vshrq_n_u32(v, 8), mask), vandq_u32(v, mask)));
        };
        auto neon_plane_row_quads = [&](uint8x16_t plane) {
            return neon_byte_sum_u32(vreinterpretq_u32_u8(plane));
        };
        auto neon_store_bgra4 = [&](uchar* dst, uint32x4_t sb, uint32x4_t sg, uint32x4_t sr, uint32x4_t sa) {
            uint32x4_t px = vaddq_u32(vaddq_u32(vshrq_n_u32(sb, 4), vshlq_n_u32(vshrq_n_u32(sg, 4), 8)),
                                      vaddq_u32(vshlq_n_u32(vshrq_n_u32(sr, 4), 16), vshlq_n_u32(vshrq_n_u32(sa, 4), 24)));
            vst1q_u32(reinterpret_cast<uint32_t*>(dst), px);
        };

        if (scale_x == 10 && scale_y == 10 && cn == 1)
        {
            const uchar* rows[10] = { S0, S0 + step, S0 + 2 * step, S0 + 3 * step, S0 + 4 * step,
                                      S0 + 5 * step, S0 + 6 * step, S0 + 7 * step, S0 + 8 * step, S0 + 9 * step };
            const int pixBlock = area_fast_detail::area_fast_10x10_pix_block();
            for (; dx <= w - pixBlock; dx += pixBlock)
            {
                const int sx = dx * 10;
                uint32_t sums[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
                for (int ri = 0; ri < 10; ++ri)
                {
                    const uchar* row = rows[ri] + sx;
                    sums[0] += area_fast_detail::area_fast_sum10(row + 0);
                    sums[1] += area_fast_detail::area_fast_sum10(row + 10);
                    sums[2] += area_fast_detail::area_fast_sum10(row + 20);
                    sums[3] += area_fast_detail::area_fast_sum10(row + 30);
                    sums[4] += area_fast_detail::area_fast_sum10(row + 40);
                    sums[5] += area_fast_detail::area_fast_sum10(row + 50);
                    sums[6] += area_fast_detail::area_fast_sum10(row + 60);
                    sums[7] += area_fast_detail::area_fast_sum10(row + 70);
                }
                for (int i = 0; i < pixBlock; ++i)
                    D[dx + i] = (uchar)((sums[i] + 50) / 100);
            }
            return dx;
        }

        if (scale_x == 4 && scale_y == 4 && cn == 1)
        {
            const uchar* rows[4] = { S0, S1, S0 + 2 * step, S0 + 3 * step };
            for (; dx <= w - 8; dx += 8)
            {
                const int sx = dx * 4;
                uint32x4_t sum0 = vdupq_n_u32(0), sum1 = vdupq_n_u32(0);
                for (int ri = 0; ri < 4; ++ri)
                {
                    const uchar* r = rows[ri];
                    uint32x4_t a0 = vld1q_u32((const uint32_t*)(r + sx));
                    uint32x4_t a1 = vld1q_u32((const uint32_t*)(r + sx + 16));
                    sum0 = vaddq_u32(sum0, vaddq_u32(vaddq_u32(vshrq_n_u32(a0, 24), vandq_u32(vshrq_n_u32(a0, 16), mask)),
                                                     vaddq_u32(vandq_u32(vshrq_n_u32(a0, 8), mask), vandq_u32(a0, mask))));
                    sum1 = vaddq_u32(sum1, vaddq_u32(vaddq_u32(vshrq_n_u32(a1, 24), vandq_u32(vshrq_n_u32(a1, 16), mask)),
                                                     vaddq_u32(vandq_u32(vshrq_n_u32(a1, 8), mask), vandq_u32(a1, mask))));
                }
                uint16x4_t d0 = vmovn_u32(vshrq_n_u32(sum0, 4));
                uint16x4_t d1 = vmovn_u32(vshrq_n_u32(sum1, 4));
                vst1_u8(D + dx, vqmovn_u16(vcombine_u16(d0, d1)));
            }
            return dx;
        }

        if (scale_x == 4 && scale_y == 4 && cn == 4)
        {
            const uchar* rows[4] = { S0, S1, S0 + 2 * step, S0 + 3 * step };
            for (; dx <= w - 32; dx += 32)
            {
                const int sx = dx * 4;
                uint32x4_t sb0 = vdupq_n_u32(0), sg0 = sb0, sr0 = sb0, sa0 = sb0;
                uint32x4_t sb1 = sb0, sg1 = sb0, sr1 = sb0, sa1 = sb0;
                for (int ri = 0; ri < 4; ++ri)
                {
                    uint8x16x4_t p0 = vld4q_u8(rows[ri] + sx);
                    uint8x16x4_t p1 = vld4q_u8(rows[ri] + sx + 64);
                    sb0 = vaddq_u32(sb0, neon_plane_row_quads(p0.val[0]));
                    sg0 = vaddq_u32(sg0, neon_plane_row_quads(p0.val[1]));
                    sr0 = vaddq_u32(sr0, neon_plane_row_quads(p0.val[2]));
                    sa0 = vaddq_u32(sa0, neon_plane_row_quads(p0.val[3]));
                    sb1 = vaddq_u32(sb1, neon_plane_row_quads(p1.val[0]));
                    sg1 = vaddq_u32(sg1, neon_plane_row_quads(p1.val[1]));
                    sr1 = vaddq_u32(sr1, neon_plane_row_quads(p1.val[2]));
                    sa1 = vaddq_u32(sa1, neon_plane_row_quads(p1.val[3]));
                }
                neon_store_bgra4(D + dx, sb0, sg0, sr0, sa0);
                neon_store_bgra4(D + dx + 16, sb1, sg1, sr1, sa1);
            }
            for (; dx <= w - 16; dx += 16)
            {
                const int sx = dx * 4;
                uint32x4_t sb = vdupq_n_u32(0), sg = sb, sr = sb, sa = sb;
                for (int ri = 0; ri < 4; ++ri)
                {
                    uint8x16x4_t p = vld4q_u8(rows[ri] + sx);
                    sb = vaddq_u32(sb, neon_plane_row_quads(p.val[0]));
                    sg = vaddq_u32(sg, neon_plane_row_quads(p.val[1]));
                    sr = vaddq_u32(sr, neon_plane_row_quads(p.val[2]));
                    sa = vaddq_u32(sa, neon_plane_row_quads(p.val[3]));
                }
                neon_store_bgra4(D + dx, sb, sg, sr, sa);
            }
            return dx;
        }

        if (scale_x == 4 && scale_y == 4 && cn == 3)
        {
            const uchar* rows[4] = { S0, S1, S0 + 2 * step, S0 + 3 * step };
            for (; dx <= w - 24; dx += 24)
            {
                const int sx = dx * 4;
                uint32x4_t sb0 = vdupq_n_u32(0), sg0 = sb0, sr0 = sb0;
                uint32x4_t sb1 = sb0, sg1 = sb0, sr1 = sb0;
                for (int ri = 0; ri < 4; ++ri)
                {
                    uint8x16x3_t p0 = vld3q_u8(rows[ri] + sx);
                    uint8x16x3_t p1 = vld3q_u8(rows[ri] + sx + 48);
                    sb0 = vaddq_u32(sb0, neon_plane_row_quads(p0.val[0]));
                    sg0 = vaddq_u32(sg0, neon_plane_row_quads(p0.val[1]));
                    sr0 = vaddq_u32(sr0, neon_plane_row_quads(p0.val[2]));
                    sb1 = vaddq_u32(sb1, neon_plane_row_quads(p1.val[0]));
                    sg1 = vaddq_u32(sg1, neon_plane_row_quads(p1.val[1]));
                    sr1 = vaddq_u32(sr1, neon_plane_row_quads(p1.val[2]));
                }
                uint32_t bv0[4], gv0[4], rv0[4], bv1[4], gv1[4], rv1[4];
                vst1q_u32(bv0, vshrq_n_u32(sb0, 4));
                vst1q_u32(gv0, vshrq_n_u32(sg0, 4));
                vst1q_u32(rv0, vshrq_n_u32(sr0, 4));
                vst1q_u32(bv1, vshrq_n_u32(sb1, 4));
                vst1q_u32(gv1, vshrq_n_u32(sg1, 4));
                vst1q_u32(rv1, vshrq_n_u32(sr1, 4));
                for (int i = 0; i < 4; ++i)
                {
                    D[dx + i * 3 + 0] = (uchar)bv0[i];
                    D[dx + i * 3 + 1] = (uchar)gv0[i];
                    D[dx + i * 3 + 2] = (uchar)rv0[i];
                    D[dx + 12 + i * 3 + 0] = (uchar)bv1[i];
                    D[dx + 12 + i * 3 + 1] = (uchar)gv1[i];
                    D[dx + 12 + i * 3 + 2] = (uchar)rv1[i];
                }
            }
            for (; dx <= w - 12; dx += 12)
            {
                const int sx = dx * 4;
                uint32x4_t sb = vdupq_n_u32(0), sg = sb, sr = sb;
                for (int ri = 0; ri < 4; ++ri)
                {
                    uint8x16x3_t p = vld3q_u8(rows[ri] + sx);
                    sb = vaddq_u32(sb, neon_plane_row_quads(p.val[0]));
                    sg = vaddq_u32(sg, neon_plane_row_quads(p.val[1]));
                    sr = vaddq_u32(sr, neon_plane_row_quads(p.val[2]));
                }
                uint32_t bv[4], gv[4], rv[4];
                vst1q_u32(bv, vshrq_n_u32(sb, 4));
                vst1q_u32(gv, vshrq_n_u32(sg, 4));
                vst1q_u32(rv, vshrq_n_u32(sr, 4));
                for (int i = 0; i < 4; ++i)
                {
                    D[dx + i * 3 + 0] = (uchar)bv[i];
                    D[dx + i * 3 + 1] = (uchar)gv[i];
                    D[dx + i * 3 + 2] = (uchar)rv[i];
                }
            }
            return dx;
        }

        if (scale_x != 2 || scale_y != 2)
            return 0;

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
        else if (cn == 3)
        {
            for (; dx <= w - 6; dx += 6, S0 += 12, S1 += 12, D += 6)
            {
                uint8x8x3_t row0 = vld3_u8(S0);
                uint8x8x3_t row1 = vld3_u8(S1);
                for (int c = 0; c < 3; ++c)
                {
                    uint16x8_t row01 = vaddl_u8(row0.val[c], row1.val[c]);
                    D[c] = (uchar)((vgetq_lane_u16(row01, 0) + vgetq_lane_u16(row01, 1) + 2) >> 2);
                    D[c + 3] = (uchar)((vgetq_lane_u16(row01, 2) + vgetq_lane_u16(row01, 3) + 2) >> 2);
                }
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
    int scale_x, scale_y;
    int cn, step;
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

        uint32x4_t v_2 = vdupq_n_u32(2);

        if (cn == 1)
        {
            for ( ; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                uint16x8x2_t v_row0 = vld2q_u16(S0), v_row1 = vld2q_u16(S1);

                uint32x4_t v_dst0 = vaddl_u16(vget_low_u16(v_row0.val[0]), vget_low_u16(v_row0.val[1]));
                v_dst0 = vaddq_u32(v_dst0, vaddl_u16(vget_low_u16(v_row1.val[0]), vget_low_u16(v_row1.val[1])));
                v_dst0 = vshrq_n_u32(vaddq_u32(v_dst0, v_2), 2);

                uint32x4_t v_dst1 = vaddl_u16(vget_high_u16(v_row0.val[0]), vget_high_u16(v_row0.val[1]));
                v_dst1 = vaddq_u32(v_dst1, vaddl_u16(vget_high_u16(v_row1.val[0]), vget_high_u16(v_row1.val[1])));
                v_dst1 = vshrq_n_u32(vaddq_u32(v_dst1, v_2), 2);

                vst1q_u16(D, vcombine_u16(vmovn_u32(v_dst0), vmovn_u32(v_dst1)));
            }
        }
        else if (cn == 4)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                uint16x8_t v_row0 = vld1q_u16(S0), v_row1 = vld1q_u16(S1);
                uint32x4_t v_dst = vaddq_u32(vaddl_u16(vget_low_u16(v_row0), vget_high_u16(v_row0)),
                                             vaddl_u16(vget_low_u16(v_row1), vget_high_u16(v_row1)));
                vst1_u16(D, vmovn_u32(vshrq_n_u32(vaddq_u32(v_dst, v_2), 2)));
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

        int32x4_t v_2 = vdupq_n_s32(2);

        if (cn == 1)
        {
            for ( ; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                int16x8x2_t v_row0 = vld2q_s16(S0), v_row1 = vld2q_s16(S1);

                int32x4_t v_dst0 = vaddl_s16(vget_low_s16(v_row0.val[0]), vget_low_s16(v_row0.val[1]));
                v_dst0 = vaddq_s32(v_dst0, vaddl_s16(vget_low_s16(v_row1.val[0]), vget_low_s16(v_row1.val[1])));
                v_dst0 = vshrq_n_s32(vaddq_s32(v_dst0, v_2), 2);

                int32x4_t v_dst1 = vaddl_s16(vget_high_s16(v_row0.val[0]), vget_high_s16(v_row0.val[1]));
                v_dst1 = vaddq_s32(v_dst1, vaddl_s16(vget_high_s16(v_row1.val[0]), vget_high_s16(v_row1.val[1])));
                v_dst1 = vshrq_n_s32(vaddq_s32(v_dst1, v_2), 2);

                vst1q_s16(D, vcombine_s16(vmovn_s32(v_dst0), vmovn_s32(v_dst1)));
            }
        }
        else if (cn == 4)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                int16x8_t v_row0 = vld1q_s16(S0), v_row1 = vld1q_s16(S1);
                int32x4_t v_dst = vaddq_s32(vaddl_s16(vget_low_s16(v_row0), vget_high_s16(v_row0)),
                                            vaddl_s16(vget_low_s16(v_row1), vget_high_s16(v_row1)));
                vst1_s16(D, vmovn_s32(vshrq_n_s32(vaddq_s32(v_dst, v_2), 2)));
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
        if (scale_x != scale_y || scale_x < 2 || (scale_x > 4 && scale_x != 10))
            return 0;

        int dx = 0;

        if (scale_x == 10 && scale_y == 10)
        {
            if (cn == 1)
                return area_fast_detail::area_fast_u8_cn1_10x10(S, D, w, step, dx);
            if (cn == 3)
                return area_fast_detail::area_fast_u8_cn3_10x10(S, D, w, step, dx);
            if (cn == 4)
                return area_fast_detail::area_fast_u8_cn4_10x10(S, D, w, step, dx);
        }

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
#if CV_SIMD_WIDTH > 64
            for ( ; dx <= w - 3*VTraits<v_uint8>::vlanes(); dx += 3*VTraits<v_uint8>::vlanes(), S0 += 6*VTraits<v_uint8>::vlanes(), S1 += 6*VTraits<v_uint8>::vlanes(), D += 3*VTraits<v_uint8>::vlanes())
            {
                v_uint16 b0, g0, r0, b1, g1, r1;
                v_load_deinterleave(S0, b0, g0, r0);
                v_load_deinterleave(S1, b1, g1, r1);
                v_uint32 masklow = vx_setall_u32(0x00ff);
                v_uint32 bl = v_add(v_add(v_shr<16>(v_reinterpret_as_u32(b0)), v_and(v_reinterpret_as_u32(b0), masklow)),
                                    v_add(v_shr<16>(v_reinterpret_as_u32(b1)), v_and(v_reinterpret_as_u32(b1), masklow)));
                v_uint32 gl = v_add(v_add(v_shr<16>(v_reinterpret_as_u32(g0)), v_and(v_reinterpret_as_u32(g0), masklow)),
                                    v_add(v_shr<16>(v_reinterpret_as_u32(g1)), v_and(v_reinterpret_as_u32(g1), masklow)));
                v_uint32 rl = v_add(v_add(v_shr<16>(v_reinterpret_as_u32(r0)), v_and(v_reinterpret_as_u32(r0), masklow)),
                                    v_add(v_shr<16>(v_reinterpret_as_u32(r1)), v_and(v_reinterpret_as_u32(r1), masklow)));
                v_load_deinterleave(S0 + 3*VTraits<v_uint16>::vlanes(), b0, g0, r0);
                v_load_deinterleave(S1 + 3*VTraits<v_uint16>::vlanes(), b1, g1, r1);
                v_uint32 bh = v_add(v_add(v_shr<16>(v_reinterpret_as_u32(b0)), v_and(v_reinterpret_as_u32(b0), masklow)),
                                    v_add(v_shr<16>(v_reinterpret_as_u32(b1)), v_and(v_reinterpret_as_u32(b1), masklow)));
                v_uint32 gh = v_add(v_add(v_shr<16>(v_reinterpret_as_u32(g0)), v_and(v_reinterpret_as_u32(g0), masklow)),
                                    v_add(v_shr<16>(v_reinterpret_as_u32(g1)), v_and(v_reinterpret_as_u32(g1), masklow)));
                v_uint32 rh = v_add(v_add(v_shr<16>(v_reinterpret_as_u32(r0)), v_and(v_reinterpret_as_u32(r0), masklow)),
                                    v_add(v_shr<16>(v_reinterpret_as_u32(r1)), v_and(v_reinterpret_as_u32(r1), masklow)));
                v_store_interleave(D, v_rshr_pack<2>(bl, bh), v_rshr_pack<2>(gl, gh), v_rshr_pack<2>(rl, rh));
            }
#else
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
#endif
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
                v_store(D, v_rshr_pack<2>(v_add(v_add(v_add(r00l, r01l), r10l), r11l), v_add(v_add(v_add(r00h, r01h), r10h), r11h)));
            }
        }

        return dx;
    }

private:
    int scale_x, scale_y;
    int cn;
    int step;
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
        fast_mode = area_fast_detail::is_fast_scale(scale_x, scale_y, cn);
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

namespace area_rational_detail {

static inline int igcd(int a, int b)
{
    a = std::abs(a);
    b = std::abs(b);
    while (b)
    {
        int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static inline bool is_rational_10_3(int src_w, int dst_w, int src_h, int dst_h,
                                    double scale_x, double scale_y)
{
    const int gw = igcd(src_w, dst_w);
    const int gh = igcd(src_h, dst_h);
    if (src_w / gw != 10 || dst_w / gw != 3 || src_h / gh != 10 || dst_h / gh != 3)
        return false;
    return std::abs(scale_x - 10.0 / 3.0) < 1e-6 && std::abs(scale_y - 10.0 / 3.0) < 1e-6;
}

static inline bool is_rational_5_3(int src_w, int dst_w, int src_h, int dst_h,
                                   double scale_x, double scale_y)
{
    const int gw = igcd(src_w, dst_w);
    const int gh = igcd(src_h, dst_h);
    if (src_w / gw != 5 || dst_w / gw != 3 || src_h / gh != 5 || dst_h / gh != 3)
        return false;
    return std::abs(scale_x - 5.0 / 3.0) < 1e-6 && std::abs(scale_y - 5.0 / 3.0) < 1e-6;
}

static inline int ihoriz_phase0_at_b(const uchar* p)
{
    return 3 * p[0] + 3 * p[3] + 3 * p[6] + p[9];
}

static inline int ihoriz_phase0_at_g(const uchar* p)
{
    return 3 * p[1] + 3 * p[4] + 3 * p[7] + p[10];
}

static inline int ihoriz_phase0_at_r(const uchar* p)
{
    return 3 * p[2] + 3 * p[5] + 3 * p[8] + p[11];
}

static inline int ihoriz_phase1_at_b(const uchar* p)
{
    return 2 * p[0] + 3 * p[3] + 3 * p[6] + 2 * p[9];
}

static inline int ihoriz_phase1_at_g(const uchar* p)
{
    return 2 * p[1] + 3 * p[4] + 3 * p[7] + 2 * p[10];
}

static inline int ihoriz_phase1_at_r(const uchar* p)
{
    return 2 * p[2] + 3 * p[5] + 3 * p[8] + 2 * p[11];
}

static inline int ihoriz_phase2_at_b(const uchar* p)
{
    return p[0] + 3 * p[3] + 3 * p[6] + 3 * p[9];
}

static inline int ihoriz_phase2_at_g(const uchar* p)
{
    return p[1] + 3 * p[4] + 3 * p[7] + 3 * p[10];
}

static inline int ihoriz_phase2_at_r(const uchar* p)
{
    return p[2] + 3 * p[5] + 3 * p[8] + 3 * p[11];
}

static inline int ihoriz_phase0_cn1(const uchar* p)
{
    return 3 * p[0] + 3 * p[1] + 3 * p[2] + p[3];
}

static inline int ihoriz_phase1_cn1(const uchar* p)
{
    return 2 * p[0] + 3 * p[1] + 3 * p[2] + 2 * p[3];
}

static inline int ihoriz_phase2_cn1(const uchar* p)
{
    return p[0] + 3 * p[1] + 3 * p[2] + 3 * p[3];
}

static inline int ihoriz_phase0_cn4(const uchar* p, int c)
{
    return 3 * p[c] + 3 * p[4 + c] + 3 * p[8 + c] + p[12 + c];
}

static inline int ihoriz_phase1_cn4(const uchar* p, int c)
{
    return 2 * p[c] + 3 * p[4 + c] + 3 * p[8 + c] + 2 * p[12 + c];
}

static inline int ihoriz_phase2_cn4(const uchar* p, int c)
{
    return p[c] + 3 * p[4 + c] + 3 * p[8 + c] + 3 * p[12 + c];
}

static inline void vert_weights_r10_3(int dy, int vw[4])
{
    switch (dy % 3)
    {
    case 0:
        vw[0] = 3; vw[1] = 3; vw[2] = 3; vw[3] = 1;
        break;
    case 1:
        vw[0] = 2; vw[1] = 3; vw[2] = 3; vw[3] = 2;
        break;
    default:
        vw[0] = 1; vw[1] = 3; vw[2] = 3; vw[3] = 3;
        break;
    }
}

static inline float vw_dot4(const int vw[4], int h0, int h1, int h2, int h3)
{
    return (vw[0] * h0 + vw[1] * h1 + vw[2] * h2 + vw[3] * h3) * 0.01f;
}

static inline void accum_dst_period_r10_3_u8c1(float* sum, int di,
                                                 const uchar* rows[4], const int vw[4])
{
    sum[di + 0] = vw_dot4(vw, ihoriz_phase0_cn1(rows[0]), ihoriz_phase0_cn1(rows[1]),
                          ihoriz_phase0_cn1(rows[2]), ihoriz_phase0_cn1(rows[3]));
    sum[di + 1] = vw_dot4(vw, ihoriz_phase1_cn1(rows[0] + 3), ihoriz_phase1_cn1(rows[1] + 3),
                          ihoriz_phase1_cn1(rows[2] + 3), ihoriz_phase1_cn1(rows[3] + 3));
    sum[di + 2] = vw_dot4(vw, ihoriz_phase2_cn1(rows[0] + 6), ihoriz_phase2_cn1(rows[1] + 6),
                          ihoriz_phase2_cn1(rows[2] + 6), ihoriz_phase2_cn1(rows[3] + 6));
}

static inline void accum_dst_period_r10_3_u8c4(float* sum, int di,
                                                 const uchar* rows[4], const int vw[4])
{
    for (int c = 0; c < 4; ++c)
    {
        sum[di + c] = vw_dot4(vw, ihoriz_phase0_cn4(rows[0], c), ihoriz_phase0_cn4(rows[1], c),
                              ihoriz_phase0_cn4(rows[2], c), ihoriz_phase0_cn4(rows[3], c));
        sum[di + 4 + c] = vw_dot4(vw, ihoriz_phase1_cn4(rows[0] + 12, c), ihoriz_phase1_cn4(rows[1] + 12, c),
                                  ihoriz_phase1_cn4(rows[2] + 12, c), ihoriz_phase1_cn4(rows[3] + 12, c));
        sum[di + 8 + c] = vw_dot4(vw, ihoriz_phase2_cn4(rows[0] + 24, c), ihoriz_phase2_cn4(rows[1] + 24, c),
                                  ihoriz_phase2_cn4(rows[2] + 24, c), ihoriz_phase2_cn4(rows[3] + 24, c));
    }
}

static inline void accum_dst_period_r10_3_u8c3(float* sum, int di,
                                                 const uchar* rows[4], const int vw[4])
{
    sum[di + 0] = vw_dot4(vw, ihoriz_phase0_at_b(rows[0]), ihoriz_phase0_at_b(rows[1]),
                          ihoriz_phase0_at_b(rows[2]), ihoriz_phase0_at_b(rows[3]));
    sum[di + 1] = vw_dot4(vw, ihoriz_phase0_at_g(rows[0]), ihoriz_phase0_at_g(rows[1]),
                          ihoriz_phase0_at_g(rows[2]), ihoriz_phase0_at_g(rows[3]));
    sum[di + 2] = vw_dot4(vw, ihoriz_phase0_at_r(rows[0]), ihoriz_phase0_at_r(rows[1]),
                          ihoriz_phase0_at_r(rows[2]), ihoriz_phase0_at_r(rows[3]));

    sum[di + 3] = vw_dot4(vw, ihoriz_phase1_at_b(rows[0] + 9), ihoriz_phase1_at_b(rows[1] + 9),
                          ihoriz_phase1_at_b(rows[2] + 9), ihoriz_phase1_at_b(rows[3] + 9));
    sum[di + 4] = vw_dot4(vw, ihoriz_phase1_at_g(rows[0] + 9), ihoriz_phase1_at_g(rows[1] + 9),
                          ihoriz_phase1_at_g(rows[2] + 9), ihoriz_phase1_at_g(rows[3] + 9));
    sum[di + 5] = vw_dot4(vw, ihoriz_phase1_at_r(rows[0] + 9), ihoriz_phase1_at_r(rows[1] + 9),
                          ihoriz_phase1_at_r(rows[2] + 9), ihoriz_phase1_at_r(rows[3] + 9));

    sum[di + 6] = vw_dot4(vw, ihoriz_phase2_at_b(rows[0] + 18), ihoriz_phase2_at_b(rows[1] + 18),
                          ihoriz_phase2_at_b(rows[2] + 18), ihoriz_phase2_at_b(rows[3] + 18));
    sum[di + 7] = vw_dot4(vw, ihoriz_phase2_at_g(rows[0] + 18), ihoriz_phase2_at_g(rows[1] + 18),
                          ihoriz_phase2_at_g(rows[2] + 18), ihoriz_phase2_at_g(rows[3] + 18));
    sum[di + 8] = vw_dot4(vw, ihoriz_phase2_at_r(rows[0] + 18), ihoriz_phase2_at_r(rows[1] + 18),
                          ihoriz_phase2_at_r(rows[2] + 18), ihoriz_phase2_at_r(rows[3] + 18));
}

template<typename AccumPeriodFn>
static void accum_dst_row_r10_3_u8(float* sum, int dy, const Mat& src, int dst_width,
                                   int periodBytes, int outStride, AccumPeriodFn accumPeriod)
{
    const int sy0 = (dy * 10) / 3;
    const uchar* rows[4] = {
        src.ptr<uchar>(sy0), src.ptr<uchar>(sy0 + 1), src.ptr<uchar>(sy0 + 2), src.ptr<uchar>(sy0 + 3)
    };
    int vw[4];
    vert_weights_r10_3(dy, vw);

    int k = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int kBlock = 4;
    for (; k + kBlock <= dst_width / 3; k += kBlock)
    {
        for (int t = 0; t < kBlock; ++t)
        {
            const int off = periodBytes * (k + t);
            const uchar* r[4] = { rows[0] + off, rows[1] + off, rows[2] + off, rows[3] + off };
            accumPeriod(sum, outStride * (k + t), r, vw);
        }
    }
#endif
    for (; k < dst_width / 3; ++k)
    {
        const int off = periodBytes * k;
        const uchar* r[4] = { rows[0] + off, rows[1] + off, rows[2] + off, rows[3] + off };
        accumPeriod(sum, outStride * k, r, vw);
    }
}

static void accum_dst_row_r10_3_u8c1(float* sum, int dy, const Mat& src, int dst_width)
{
    accum_dst_row_r10_3_u8(sum, dy, src, dst_width, 10, 3, accum_dst_period_r10_3_u8c1);

    const int sy0 = (dy * 10) / 3;
    const uchar* rows[4] = {
        src.ptr<uchar>(sy0), src.ptr<uchar>(sy0 + 1), src.ptr<uchar>(sy0 + 2), src.ptr<uchar>(sy0 + 3)
    };
    int vw[4];
    vert_weights_r10_3(dy, vw);

    int k = dst_width / 3;
    for (int dx = k * 3; dx < dst_width; ++dx)
    {
        const int si = (dx * 10) / 3;
        int h[4];
        for (int ri = 0; ri < 4; ++ri)
        {
            const uchar* p = rows[ri] + si;
            switch (dx % 3)
            {
            case 0: h[ri] = ihoriz_phase0_cn1(p); break;
            case 1: h[ri] = ihoriz_phase1_cn1(p); break;
            default: h[ri] = ihoriz_phase2_cn1(p); break;
            }
        }
        sum[dx] = vw_dot4(vw, h[0], h[1], h[2], h[3]);
    }
}

static void accum_dst_row_r10_3_u8c3(float* sum, int dy, const Mat& src, int dst_width)
{
    accum_dst_row_r10_3_u8(sum, dy, src, dst_width, 30, 9, accum_dst_period_r10_3_u8c3);

    const int sy0 = (dy * 10) / 3;
    const uchar* rows[4] = {
        src.ptr<uchar>(sy0), src.ptr<uchar>(sy0 + 1), src.ptr<uchar>(sy0 + 2), src.ptr<uchar>(sy0 + 3)
    };
    int vw[4];
    vert_weights_r10_3(dy, vw);

    int k = dst_width / 3;
    for (int dx = k * 3; dx < dst_width; ++dx)
    {
        const int di = dx * 3;
        const int si = ((dx * 10) / 3) * 3;
        int hb[4], hg[4], hr[4];
        for (int ri = 0; ri < 4; ++ri)
        {
            const uchar* p = rows[ri] + si;
            switch (dx % 3)
            {
            case 0:
                hb[ri] = ihoriz_phase0_at_b(p);
                hg[ri] = ihoriz_phase0_at_g(p);
                hr[ri] = ihoriz_phase0_at_r(p);
                break;
            case 1:
                hb[ri] = ihoriz_phase1_at_b(p);
                hg[ri] = ihoriz_phase1_at_g(p);
                hr[ri] = ihoriz_phase1_at_r(p);
                break;
            default:
                hb[ri] = ihoriz_phase2_at_b(p);
                hg[ri] = ihoriz_phase2_at_g(p);
                hr[ri] = ihoriz_phase2_at_r(p);
                break;
            }
        }
        sum[di + 0] = vw_dot4(vw, hb[0], hb[1], hb[2], hb[3]);
        sum[di + 1] = vw_dot4(vw, hg[0], hg[1], hg[2], hg[3]);
        sum[di + 2] = vw_dot4(vw, hr[0], hr[1], hr[2], hr[3]);
    }
}

static void accum_dst_row_r10_3_u8c4(float* sum, int dy, const Mat& src, int dst_width)
{
    accum_dst_row_r10_3_u8(sum, dy, src, dst_width, 40, 12, accum_dst_period_r10_3_u8c4);

    const int sy0 = (dy * 10) / 3;
    const uchar* rows[4] = {
        src.ptr<uchar>(sy0), src.ptr<uchar>(sy0 + 1), src.ptr<uchar>(sy0 + 2), src.ptr<uchar>(sy0 + 3)
    };
    int vw[4];
    vert_weights_r10_3(dy, vw);

    int k = dst_width / 3;
    for (int dx = k * 3; dx < dst_width; ++dx)
    {
        const int di = dx * 4;
        const int si = ((dx * 10) / 3) * 4;
        int h[4][4];
        for (int ri = 0; ri < 4; ++ri)
        {
            const uchar* p = rows[ri] + si;
            for (int c = 0; c < 4; ++c)
            {
                switch (dx % 3)
                {
                case 0: h[ri][c] = ihoriz_phase0_cn4(p, c); break;
                case 1: h[ri][c] = ihoriz_phase1_cn4(p, c); break;
                default: h[ri][c] = ihoriz_phase2_cn4(p, c); break;
                }
            }
        }
        for (int c = 0; c < 4; ++c)
            sum[di + c] = vw_dot4(vw, h[0][c], h[1][c], h[2][c], h[3][c]);
    }
}

class ResizeAreaRational10_3_Invoker : public ParallelLoopBody
{
public:
    ResizeAreaRational10_3_Invoker(const Mat& _src, Mat& _dst, int _cn)
        : src(&_src), dst(&_dst), cn(_cn)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        const int dst_width = dst->cols;
        const int width_cn = dst_width * cn;
        AutoBuffer<float> _sum(width_cn);
        float* sum = _sum.data();

        for (int dy = range.start; dy < range.end; ++dy)
        {
            if (cn == 1)
                accum_dst_row_r10_3_u8c1(sum, dy, *src, dst_width);
            else if (cn == 3)
                accum_dst_row_r10_3_u8c3(sum, dy, *src, dst_width);
            else
                accum_dst_row_r10_3_u8c4(sum, dy, *src, dst_width);
            inter_area::saturate_store(sum, width_cn, dst->ptr<uchar>(dy));
        }
    }

private:
    const Mat* src;
    Mat* dst;
    int cn;
};

static void resizeAreaRational10_3_u8_(const Mat& src, Mat& dst)
{
    const int cn = src.channels();
    parallel_for_(Range(0, dst.rows),
                  ResizeAreaRational10_3_Invoker(src, dst, cn),
                  dst.total() / ((double)(1 << 16)));
}

static inline float vw_dot2_r5_3(const int vw[2], int h0, int h1)
{
    return (vw[0] * h0 + vw[1] * h1) * 0.04f;
}

static inline float vw_dot3_r5_3(const int vw[3], int h0, int h1, int h2)
{
    return (vw[0] * h0 + vw[1] * h1 + vw[2] * h2) * 0.04f;
}

static inline void vert_weights_r5_3(int dy, int vw[3], int& nrows)
{
    switch (dy % 3)
    {
    case 0:
        vw[0] = 3; vw[1] = 2; nrows = 2;
        break;
    case 1:
        vw[0] = 1; vw[1] = 3; vw[2] = 1; nrows = 3;
        break;
    default:
        vw[0] = 2; vw[1] = 3; nrows = 2;
        break;
    }
}

static inline int ihoriz_phase0_cn1_r5_3(const uchar* p)
{
    return 3 * p[0] + 2 * p[1];
}

static inline int ihoriz_phase1_cn1_r5_3(const uchar* p)
{
    return p[0] + 3 * p[1] + p[2];
}

static inline int ihoriz_phase2_cn1_r5_3(const uchar* p)
{
    return 2 * p[0] + 3 * p[1];
}

static inline int ihoriz_phase0_at_b_r5_3(const uchar* p)
{
    return 3 * p[0] + 2 * p[3];
}

static inline int ihoriz_phase0_at_g_r5_3(const uchar* p)
{
    return 3 * p[1] + 2 * p[4];
}

static inline int ihoriz_phase0_at_r_r5_3(const uchar* p)
{
    return 3 * p[2] + 2 * p[5];
}

static inline int ihoriz_phase1_at_b_r5_3(const uchar* p)
{
    return p[0] + 3 * p[3] + p[6];
}

static inline int ihoriz_phase1_at_g_r5_3(const uchar* p)
{
    return p[1] + 3 * p[4] + p[7];
}

static inline int ihoriz_phase1_at_r_r5_3(const uchar* p)
{
    return p[2] + 3 * p[5] + p[8];
}

static inline int ihoriz_phase2_at_b_r5_3(const uchar* p)
{
    return 2 * p[0] + 3 * p[3];
}

static inline int ihoriz_phase2_at_g_r5_3(const uchar* p)
{
    return 2 * p[1] + 3 * p[4];
}

static inline int ihoriz_phase2_at_r_r5_3(const uchar* p)
{
    return 2 * p[2] + 3 * p[5];
}

static inline int ihoriz_phase0_cn4_r5_3(const uchar* p, int c)
{
    return 3 * p[c] + 2 * p[4 + c];
}

static inline int ihoriz_phase1_cn4_r5_3(const uchar* p, int c)
{
    return p[c] + 3 * p[4 + c] + p[8 + c];
}

static inline int ihoriz_phase2_cn4_r5_3(const uchar* p, int c)
{
    return 2 * p[c] + 3 * p[4 + c];
}

static inline void accum_dst_period_r5_3_u8c1_2row(float* sum, int di,
                                                    const uchar* rows[3], const int vw[2])
{
    sum[di + 0] = vw_dot2_r5_3(vw, ihoriz_phase0_cn1_r5_3(rows[0]), ihoriz_phase0_cn1_r5_3(rows[1]));
    sum[di + 1] = vw_dot2_r5_3(vw, ihoriz_phase1_cn1_r5_3(rows[0] + 1), ihoriz_phase1_cn1_r5_3(rows[1] + 1));
    sum[di + 2] = vw_dot2_r5_3(vw, ihoriz_phase2_cn1_r5_3(rows[0] + 3), ihoriz_phase2_cn1_r5_3(rows[1] + 3));
}

static inline void accum_dst_period_r5_3_u8c1_3row(float* sum, int di,
                                                    const uchar* rows[3], const int vw[3])
{
    sum[di + 0] = vw_dot3_r5_3(vw, ihoriz_phase0_cn1_r5_3(rows[0]), ihoriz_phase0_cn1_r5_3(rows[1]),
                               ihoriz_phase0_cn1_r5_3(rows[2]));
    sum[di + 1] = vw_dot3_r5_3(vw, ihoriz_phase1_cn1_r5_3(rows[0] + 1), ihoriz_phase1_cn1_r5_3(rows[1] + 1),
                               ihoriz_phase1_cn1_r5_3(rows[2] + 1));
    sum[di + 2] = vw_dot3_r5_3(vw, ihoriz_phase2_cn1_r5_3(rows[0] + 3), ihoriz_phase2_cn1_r5_3(rows[1] + 3),
                               ihoriz_phase2_cn1_r5_3(rows[2] + 3));
}

static inline void accum_dst_period_r5_3_u8c3_2row(float* sum, int di,
                                                    const uchar* rows[3], const int vw[2])
{
    sum[di + 0] = vw_dot2_r5_3(vw, ihoriz_phase0_at_b_r5_3(rows[0]), ihoriz_phase0_at_b_r5_3(rows[1]));
    sum[di + 1] = vw_dot2_r5_3(vw, ihoriz_phase0_at_g_r5_3(rows[0]), ihoriz_phase0_at_g_r5_3(rows[1]));
    sum[di + 2] = vw_dot2_r5_3(vw, ihoriz_phase0_at_r_r5_3(rows[0]), ihoriz_phase0_at_r_r5_3(rows[1]));

    sum[di + 3] = vw_dot2_r5_3(vw, ihoriz_phase1_at_b_r5_3(rows[0] + 3), ihoriz_phase1_at_b_r5_3(rows[1] + 3));
    sum[di + 4] = vw_dot2_r5_3(vw, ihoriz_phase1_at_g_r5_3(rows[0] + 3), ihoriz_phase1_at_g_r5_3(rows[1] + 3));
    sum[di + 5] = vw_dot2_r5_3(vw, ihoriz_phase1_at_r_r5_3(rows[0] + 3), ihoriz_phase1_at_r_r5_3(rows[1] + 3));

    sum[di + 6] = vw_dot2_r5_3(vw, ihoriz_phase2_at_b_r5_3(rows[0] + 9), ihoriz_phase2_at_b_r5_3(rows[1] + 9));
    sum[di + 7] = vw_dot2_r5_3(vw, ihoriz_phase2_at_g_r5_3(rows[0] + 9), ihoriz_phase2_at_g_r5_3(rows[1] + 9));
    sum[di + 8] = vw_dot2_r5_3(vw, ihoriz_phase2_at_r_r5_3(rows[0] + 9), ihoriz_phase2_at_r_r5_3(rows[1] + 9));
}

static inline void accum_dst_period_r5_3_u8c3_3row(float* sum, int di,
                                                    const uchar* rows[3], const int vw[3])
{
    sum[di + 0] = vw_dot3_r5_3(vw, ihoriz_phase0_at_b_r5_3(rows[0]), ihoriz_phase0_at_b_r5_3(rows[1]),
                               ihoriz_phase0_at_b_r5_3(rows[2]));
    sum[di + 1] = vw_dot3_r5_3(vw, ihoriz_phase0_at_g_r5_3(rows[0]), ihoriz_phase0_at_g_r5_3(rows[1]),
                               ihoriz_phase0_at_g_r5_3(rows[2]));
    sum[di + 2] = vw_dot3_r5_3(vw, ihoriz_phase0_at_r_r5_3(rows[0]), ihoriz_phase0_at_r_r5_3(rows[1]),
                               ihoriz_phase0_at_r_r5_3(rows[2]));

    sum[di + 3] = vw_dot3_r5_3(vw, ihoriz_phase1_at_b_r5_3(rows[0] + 3), ihoriz_phase1_at_b_r5_3(rows[1] + 3),
                               ihoriz_phase1_at_b_r5_3(rows[2] + 3));
    sum[di + 4] = vw_dot3_r5_3(vw, ihoriz_phase1_at_g_r5_3(rows[0] + 3), ihoriz_phase1_at_g_r5_3(rows[1] + 3),
                               ihoriz_phase1_at_g_r5_3(rows[2] + 3));
    sum[di + 5] = vw_dot3_r5_3(vw, ihoriz_phase1_at_r_r5_3(rows[0] + 3), ihoriz_phase1_at_r_r5_3(rows[1] + 3),
                               ihoriz_phase1_at_r_r5_3(rows[2] + 3));

    sum[di + 6] = vw_dot3_r5_3(vw, ihoriz_phase2_at_b_r5_3(rows[0] + 9), ihoriz_phase2_at_b_r5_3(rows[1] + 9),
                               ihoriz_phase2_at_b_r5_3(rows[2] + 9));
    sum[di + 7] = vw_dot3_r5_3(vw, ihoriz_phase2_at_g_r5_3(rows[0] + 9), ihoriz_phase2_at_g_r5_3(rows[1] + 9),
                               ihoriz_phase2_at_g_r5_3(rows[2] + 9));
    sum[di + 8] = vw_dot3_r5_3(vw, ihoriz_phase2_at_r_r5_3(rows[0] + 9), ihoriz_phase2_at_r_r5_3(rows[1] + 9),
                               ihoriz_phase2_at_r_r5_3(rows[2] + 9));
}

static inline void accum_dst_period_r5_3_u8c4_2row(float* sum, int di,
                                                    const uchar* rows[3], const int vw[2])
{
    for (int c = 0; c < 4; ++c)
    {
        sum[di + c] = vw_dot2_r5_3(vw, ihoriz_phase0_cn4_r5_3(rows[0], c), ihoriz_phase0_cn4_r5_3(rows[1], c));
        sum[di + 4 + c] = vw_dot2_r5_3(vw, ihoriz_phase1_cn4_r5_3(rows[0] + 4, c), ihoriz_phase1_cn4_r5_3(rows[1] + 4, c));
        sum[di + 8 + c] = vw_dot2_r5_3(vw, ihoriz_phase2_cn4_r5_3(rows[0] + 12, c), ihoriz_phase2_cn4_r5_3(rows[1] + 12, c));
    }
}

static inline void accum_dst_period_r5_3_u8c4_3row(float* sum, int di,
                                                    const uchar* rows[3], const int vw[3])
{
    for (int c = 0; c < 4; ++c)
    {
        sum[di + c] = vw_dot3_r5_3(vw, ihoriz_phase0_cn4_r5_3(rows[0], c), ihoriz_phase0_cn4_r5_3(rows[1], c),
                                   ihoriz_phase0_cn4_r5_3(rows[2], c));
        sum[di + 4 + c] = vw_dot3_r5_3(vw, ihoriz_phase1_cn4_r5_3(rows[0] + 4, c), ihoriz_phase1_cn4_r5_3(rows[1] + 4, c),
                                       ihoriz_phase1_cn4_r5_3(rows[2] + 4, c));
        sum[di + 8 + c] = vw_dot3_r5_3(vw, ihoriz_phase2_cn4_r5_3(rows[0] + 12, c), ihoriz_phase2_cn4_r5_3(rows[1] + 12, c),
                                       ihoriz_phase2_cn4_r5_3(rows[2] + 12, c));
    }
}

template<typename AccumPeriodFn2, typename AccumPeriodFn3>
static void accum_dst_row_r5_3_u8_period(float* sum, int dy, const Mat& src, int dst_width,
                                         int periodBytes, int outStride, int nrows, const int* vw,
                                         AccumPeriodFn2 accum2, AccumPeriodFn3 accum3)
{
    const int sy0 = (dy * 5) / 3;
    const uchar* rows[3] = {
        src.ptr<uchar>(sy0), src.ptr<uchar>(sy0 + 1), src.ptr<uchar>(sy0 + 2)
    };

    int k = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int kBlock = 4;
    for (; k + kBlock <= dst_width / 3; k += kBlock)
    {
        for (int t = 0; t < kBlock; ++t)
        {
            const int off = periodBytes * (k + t);
            const uchar* r[3] = { rows[0] + off, rows[1] + off, rows[2] + off };
            if (nrows == 2)
                accum2(sum, outStride * (k + t), r, (const int*)vw);
            else
                accum3(sum, outStride * (k + t), r, (const int*)vw);
        }
    }
#endif
    for (; k < dst_width / 3; ++k)
    {
        const int off = periodBytes * k;
        const uchar* r[3] = { rows[0] + off, rows[1] + off, rows[2] + off };
        if (nrows == 2)
            accum2(sum, outStride * k, r, (const int*)vw);
        else
            accum3(sum, outStride * k, r, (const int*)vw);
    }
}

static void accum_dst_row_r5_3_u8c1(float* sum, int dy, const Mat& src, int dst_width)
{
    int vw[3], nrows;
    vert_weights_r5_3(dy, vw, nrows);

    accum_dst_row_r5_3_u8_period(sum, dy, src, dst_width, 5, 3, nrows, vw,
                                 accum_dst_period_r5_3_u8c1_2row, accum_dst_period_r5_3_u8c1_3row);

    const int sy0 = (dy * 5) / 3;
    const uchar* rows[3] = {
        src.ptr<uchar>(sy0), src.ptr<uchar>(sy0 + 1), src.ptr<uchar>(sy0 + 2)
    };

    int k = dst_width / 3;
    for (int dx = k * 3; dx < dst_width; ++dx)
    {
        const int si = (dx * 5) / 3;
        int h[3];
        for (int ri = 0; ri < nrows; ++ri)
        {
            const uchar* p = rows[ri] + si;
            switch (dx % 3)
            {
            case 0: h[ri] = ihoriz_phase0_cn1_r5_3(p); break;
            case 1: h[ri] = ihoriz_phase1_cn1_r5_3(p); break;
            default: h[ri] = ihoriz_phase2_cn1_r5_3(p); break;
            }
        }
        if (nrows == 2)
            sum[dx] = vw_dot2_r5_3(vw, h[0], h[1]);
        else
            sum[dx] = vw_dot3_r5_3(vw, h[0], h[1], h[2]);
    }
}

static void accum_dst_row_r5_3_u8c3(float* sum, int dy, const Mat& src, int dst_width)
{
    int vw[3], nrows;
    vert_weights_r5_3(dy, vw, nrows);

    accum_dst_row_r5_3_u8_period(sum, dy, src, dst_width, 15, 9, nrows, vw,
                                 accum_dst_period_r5_3_u8c3_2row, accum_dst_period_r5_3_u8c3_3row);

    const int sy0 = (dy * 5) / 3;
    const uchar* rows[3] = {
        src.ptr<uchar>(sy0), src.ptr<uchar>(sy0 + 1), src.ptr<uchar>(sy0 + 2)
    };

    int k = dst_width / 3;
    for (int dx = k * 3; dx < dst_width; ++dx)
    {
        const int di = dx * 3;
        const int si = ((dx * 5) / 3) * 3;
        int hb[3], hg[3], hr[3];
        for (int ri = 0; ri < nrows; ++ri)
        {
            const uchar* p = rows[ri] + si;
            switch (dx % 3)
            {
            case 0:
                hb[ri] = ihoriz_phase0_at_b_r5_3(p);
                hg[ri] = ihoriz_phase0_at_g_r5_3(p);
                hr[ri] = ihoriz_phase0_at_r_r5_3(p);
                break;
            case 1:
                hb[ri] = ihoriz_phase1_at_b_r5_3(p);
                hg[ri] = ihoriz_phase1_at_g_r5_3(p);
                hr[ri] = ihoriz_phase1_at_r_r5_3(p);
                break;
            default:
                hb[ri] = ihoriz_phase2_at_b_r5_3(p);
                hg[ri] = ihoriz_phase2_at_g_r5_3(p);
                hr[ri] = ihoriz_phase2_at_r_r5_3(p);
                break;
            }
        }
        if (nrows == 2)
        {
            sum[di + 0] = vw_dot2_r5_3(vw, hb[0], hb[1]);
            sum[di + 1] = vw_dot2_r5_3(vw, hg[0], hg[1]);
            sum[di + 2] = vw_dot2_r5_3(vw, hr[0], hr[1]);
        }
        else
        {
            sum[di + 0] = vw_dot3_r5_3(vw, hb[0], hb[1], hb[2]);
            sum[di + 1] = vw_dot3_r5_3(vw, hg[0], hg[1], hg[2]);
            sum[di + 2] = vw_dot3_r5_3(vw, hr[0], hr[1], hr[2]);
        }
    }
}

static void accum_dst_row_r5_3_u8c4(float* sum, int dy, const Mat& src, int dst_width)
{
    int vw[3], nrows;
    vert_weights_r5_3(dy, vw, nrows);

    accum_dst_row_r5_3_u8_period(sum, dy, src, dst_width, 20, 12, nrows, vw,
                                 accum_dst_period_r5_3_u8c4_2row, accum_dst_period_r5_3_u8c4_3row);

    const int sy0 = (dy * 5) / 3;
    const uchar* rows[3] = {
        src.ptr<uchar>(sy0), src.ptr<uchar>(sy0 + 1), src.ptr<uchar>(sy0 + 2)
    };

    int k = dst_width / 3;
    for (int dx = k * 3; dx < dst_width; ++dx)
    {
        const int di = dx * 4;
        const int si = ((dx * 5) / 3) * 4;
        int h[3][4];
        for (int ri = 0; ri < nrows; ++ri)
        {
            const uchar* p = rows[ri] + si;
            for (int c = 0; c < 4; ++c)
            {
                switch (dx % 3)
                {
                case 0: h[ri][c] = ihoriz_phase0_cn4_r5_3(p, c); break;
                case 1: h[ri][c] = ihoriz_phase1_cn4_r5_3(p, c); break;
                default: h[ri][c] = ihoriz_phase2_cn4_r5_3(p, c); break;
                }
            }
        }
        for (int c = 0; c < 4; ++c)
        {
            if (nrows == 2)
                sum[di + c] = vw_dot2_r5_3(vw, h[0][c], h[1][c]);
            else
                sum[di + c] = vw_dot3_r5_3(vw, h[0][c], h[1][c], h[2][c]);
        }
    }
}

class ResizeAreaRational5_3_Invoker : public ParallelLoopBody
{
public:
    ResizeAreaRational5_3_Invoker(const Mat& _src, Mat& _dst, int _cn)
        : src(&_src), dst(&_dst), cn(_cn)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        const int dst_width = dst->cols;
        const int width_cn = dst_width * cn;
        AutoBuffer<float> _sum(width_cn);
        float* sum = _sum.data();

        for (int dy = range.start; dy < range.end; ++dy)
        {
            if (cn == 1)
                accum_dst_row_r5_3_u8c1(sum, dy, *src, dst_width);
            else if (cn == 3)
                accum_dst_row_r5_3_u8c3(sum, dy, *src, dst_width);
            else
                accum_dst_row_r5_3_u8c4(sum, dy, *src, dst_width);
            inter_area::saturate_store(sum, width_cn, dst->ptr<uchar>(dy));
        }
    }

private:
    const Mat* src;
    Mat* dst;
    int cn;
};

static void resizeAreaRational5_3_u8_(const Mat& src, Mat& dst)
{
    const int cn = src.channels();
    parallel_for_(Range(0, dst.rows),
                  ResizeAreaRational5_3_Invoker(src, dst, cn),
                  dst.total() / ((double)(1 << 16)));
}

// --- 32FC rational fast paths (per-pixel fused H+V, same geometry as u8) ---

static inline float fhoriz_phase0_cn1_f32(const float* p)
{
    return 3.f * p[0] + 3.f * p[1] + 3.f * p[2] + p[3];
}

static inline float fhoriz_phase1_cn1_f32(const float* p)
{
    return 2.f * p[0] + 3.f * p[1] + 3.f * p[2] + 2.f * p[3];
}

static inline float fhoriz_phase2_cn1_f32(const float* p)
{
    return p[0] + 3.f * p[1] + 3.f * p[2] + 3.f * p[3];
}

static inline float fhoriz_phase0_cn1_r5_3_f32(const float* p)
{
    return 3.f * p[0] + 2.f * p[1];
}

static inline float fhoriz_phase1_cn1_r5_3_f32(const float* p)
{
    return p[0] + 3.f * p[1] + p[2];
}

static inline float fhoriz_phase2_cn1_r5_3_f32(const float* p)
{
    return 2.f * p[0] + 3.f * p[1];
}

static inline float fhoriz_phase0_cn4_f32(const float* p, int c)
{
    return 3.f * p[c] + 3.f * p[4 + c] + 3.f * p[8 + c] + p[12 + c];
}

static inline float fhoriz_phase1_cn4_f32(const float* p, int c)
{
    return 2.f * p[c] + 3.f * p[4 + c] + 3.f * p[8 + c] + 2.f * p[12 + c];
}

static inline float fhoriz_phase2_cn4_f32(const float* p, int c)
{
    return p[c] + 3.f * p[4 + c] + 3.f * p[8 + c] + 3.f * p[12 + c];
}

static inline float fhoriz_phase0_cn4_r5_3_f32(const float* p, int c)
{
    return 3.f * p[c] + 2.f * p[4 + c];
}

static inline float fhoriz_phase1_cn4_r5_3_f32(const float* p, int c)
{
    return p[c] + 3.f * p[4 + c] + p[8 + c];
}

static inline float fhoriz_phase2_cn4_r5_3_f32(const float* p, int c)
{
    return 2.f * p[c] + 3.f * p[4 + c];
}

static inline float fhoriz_phase0_at_c_f32(const float* p, int c)
{
    return 3.f * p[c] + 3.f * p[3 + c] + 3.f * p[6 + c] + p[9 + c];
}

static inline float fhoriz_phase1_at_c_f32(const float* p, int c)
{
    return 2.f * p[c] + 3.f * p[3 + c] + 3.f * p[6 + c] + 2.f * p[9 + c];
}

static inline float fhoriz_phase2_at_c_f32(const float* p, int c)
{
    return p[c] + 3.f * p[3 + c] + 3.f * p[6 + c] + 3.f * p[9 + c];
}

static inline float fhoriz_phase0_at_c_r5_3_f32(const float* p, int c)
{
    return 3.f * p[c] + 2.f * p[3 + c];
}

static inline float fhoriz_phase1_at_c_r5_3_f32(const float* p, int c)
{
    return p[c] + 3.f * p[3 + c] + p[6 + c];
}

static inline float fhoriz_phase2_at_c_r5_3_f32(const float* p, int c)
{
    return 2.f * p[c] + 3.f * p[3 + c];
}

static inline float fvw_dot4_f32(const int vw[4], float h0, float h1, float h2, float h3)
{
    return (vw[0] * h0 + vw[1] * h1 + vw[2] * h2 + vw[3] * h3) * 0.01f;
}

static inline float fvw_dot2_r5_3_f32(const int vw[2], float h0, float h1)
{
    return (vw[0] * h0 + vw[1] * h1) * 0.04f;
}

static inline float fvw_dot3_r5_3_f32(const int vw[3], float h0, float h1, float h2)
{
    return (vw[0] * h0 + vw[1] * h1 + vw[2] * h2) * 0.04f;
}

static inline float eval_r10_3_f32_cn1(int dx, int dy, const Mat& src)
{
    const int sy0 = (dy * 10) / 3;
    int vw[4];
    vert_weights_r10_3(dy, vw);
    const int si = (dx * 10) / 3;
    float h[4];
    for (int ri = 0; ri < 4; ++ri)
    {
        const float* p = src.ptr<float>(sy0 + ri) + si;
        switch (dx % 3)
        {
        case 0: h[ri] = fhoriz_phase0_cn1_f32(p); break;
        case 1: h[ri] = fhoriz_phase1_cn1_f32(p); break;
        default: h[ri] = fhoriz_phase2_cn1_f32(p); break;
        }
    }
    return fvw_dot4_f32(vw, h[0], h[1], h[2], h[3]);
}

static inline void eval_r10_3_f32_cn3_px(int dx, int dy, const Mat& src, float* dst3)
{
    const int sy0 = (dy * 10) / 3;
    int vw[4];
    vert_weights_r10_3(dy, vw);
    const int si = ((dx * 10) / 3) * 3;
    float h[3][4];
    for (int ri = 0; ri < 4; ++ri)
    {
        const float* p = src.ptr<float>(sy0 + ri) + si;
        switch (dx % 3)
        {
        case 0:
            h[0][ri] = fhoriz_phase0_at_c_f32(p, 0);
            h[1][ri] = fhoriz_phase0_at_c_f32(p, 1);
            h[2][ri] = fhoriz_phase0_at_c_f32(p, 2);
            break;
        case 1:
            h[0][ri] = fhoriz_phase1_at_c_f32(p, 0);
            h[1][ri] = fhoriz_phase1_at_c_f32(p, 1);
            h[2][ri] = fhoriz_phase1_at_c_f32(p, 2);
            break;
        default:
            h[0][ri] = fhoriz_phase2_at_c_f32(p, 0);
            h[1][ri] = fhoriz_phase2_at_c_f32(p, 1);
            h[2][ri] = fhoriz_phase2_at_c_f32(p, 2);
            break;
        }
    }
    for (int c = 0; c < 3; ++c)
        dst3[c] = fvw_dot4_f32(vw, h[c][0], h[c][1], h[c][2], h[c][3]);
}

static inline void eval_r10_3_f32_cn4_px(int dx, int dy, const Mat& src, float* dst4)
{
    const int sy0 = (dy * 10) / 3;
    int vw[4];
    vert_weights_r10_3(dy, vw);
    const int si = ((dx * 10) / 3) * 4;
    float h[4][4];
    for (int ri = 0; ri < 4; ++ri)
    {
        const float* p = src.ptr<float>(sy0 + ri) + si;
        switch (dx % 3)
        {
        case 0:
            h[0][ri] = fhoriz_phase0_cn4_f32(p, 0);
            h[1][ri] = fhoriz_phase0_cn4_f32(p, 1);
            h[2][ri] = fhoriz_phase0_cn4_f32(p, 2);
            h[3][ri] = fhoriz_phase0_cn4_f32(p, 3);
            break;
        case 1:
            h[0][ri] = fhoriz_phase1_cn4_f32(p, 0);
            h[1][ri] = fhoriz_phase1_cn4_f32(p, 1);
            h[2][ri] = fhoriz_phase1_cn4_f32(p, 2);
            h[3][ri] = fhoriz_phase1_cn4_f32(p, 3);
            break;
        default:
            h[0][ri] = fhoriz_phase2_cn4_f32(p, 0);
            h[1][ri] = fhoriz_phase2_cn4_f32(p, 1);
            h[2][ri] = fhoriz_phase2_cn4_f32(p, 2);
            h[3][ri] = fhoriz_phase2_cn4_f32(p, 3);
            break;
        }
    }
    for (int c = 0; c < 4; ++c)
        dst4[c] = fvw_dot4_f32(vw, h[c][0], h[c][1], h[c][2], h[c][3]);
}

static inline float eval_r5_3_f32_cn1(int dx, int dy, const Mat& src)
{
    const int sy0 = (dy * 5) / 3;
    int vw[3], nrows;
    vert_weights_r5_3(dy, vw, nrows);
    const int si = (dx * 5) / 3;
    float h[3];
    for (int ri = 0; ri < nrows; ++ri)
    {
        const float* p = src.ptr<float>(sy0 + ri) + si;
        switch (dx % 3)
        {
        case 0: h[ri] = fhoriz_phase0_cn1_r5_3_f32(p); break;
        case 1: h[ri] = fhoriz_phase1_cn1_r5_3_f32(p); break;
        default: h[ri] = fhoriz_phase2_cn1_r5_3_f32(p); break;
        }
    }
    return (nrows == 2) ? fvw_dot2_r5_3_f32(vw, h[0], h[1]) : fvw_dot3_r5_3_f32(vw, h[0], h[1], h[2]);
}

static inline void eval_r5_3_f32_cn3_px(int dx, int dy, const Mat& src, float* dst3)
{
    const int sy0 = (dy * 5) / 3;
    int vw[3], nrows;
    vert_weights_r5_3(dy, vw, nrows);
    const int si = ((dx * 5) / 3) * 3;
    float h[3][3];
    for (int ri = 0; ri < nrows; ++ri)
    {
        const float* p = src.ptr<float>(sy0 + ri) + si;
        switch (dx % 3)
        {
        case 0:
            h[0][ri] = fhoriz_phase0_at_c_r5_3_f32(p, 0);
            h[1][ri] = fhoriz_phase0_at_c_r5_3_f32(p, 1);
            h[2][ri] = fhoriz_phase0_at_c_r5_3_f32(p, 2);
            break;
        case 1:
            h[0][ri] = fhoriz_phase1_at_c_r5_3_f32(p, 0);
            h[1][ri] = fhoriz_phase1_at_c_r5_3_f32(p, 1);
            h[2][ri] = fhoriz_phase1_at_c_r5_3_f32(p, 2);
            break;
        default:
            h[0][ri] = fhoriz_phase2_at_c_r5_3_f32(p, 0);
            h[1][ri] = fhoriz_phase2_at_c_r5_3_f32(p, 1);
            h[2][ri] = fhoriz_phase2_at_c_r5_3_f32(p, 2);
            break;
        }
    }
    for (int c = 0; c < 3; ++c)
        dst3[c] = (nrows == 2) ? fvw_dot2_r5_3_f32(vw, h[c][0], h[c][1]) :
                                   fvw_dot3_r5_3_f32(vw, h[c][0], h[c][1], h[c][2]);
}

static inline void eval_r5_3_f32_cn4_px(int dx, int dy, const Mat& src, float* dst4)
{
    const int sy0 = (dy * 5) / 3;
    int vw[3], nrows;
    vert_weights_r5_3(dy, vw, nrows);
    const int si = ((dx * 5) / 3) * 4;
    float h[4][3];
    for (int ri = 0; ri < nrows; ++ri)
    {
        const float* p = src.ptr<float>(sy0 + ri) + si;
        switch (dx % 3)
        {
        case 0:
            h[0][ri] = fhoriz_phase0_cn4_r5_3_f32(p, 0);
            h[1][ri] = fhoriz_phase0_cn4_r5_3_f32(p, 1);
            h[2][ri] = fhoriz_phase0_cn4_r5_3_f32(p, 2);
            h[3][ri] = fhoriz_phase0_cn4_r5_3_f32(p, 3);
            break;
        case 1:
            h[0][ri] = fhoriz_phase1_cn4_r5_3_f32(p, 0);
            h[1][ri] = fhoriz_phase1_cn4_r5_3_f32(p, 1);
            h[2][ri] = fhoriz_phase1_cn4_r5_3_f32(p, 2);
            h[3][ri] = fhoriz_phase1_cn4_r5_3_f32(p, 3);
            break;
        default:
            h[0][ri] = fhoriz_phase2_cn4_r5_3_f32(p, 0);
            h[1][ri] = fhoriz_phase2_cn4_r5_3_f32(p, 1);
            h[2][ri] = fhoriz_phase2_cn4_r5_3_f32(p, 2);
            h[3][ri] = fhoriz_phase2_cn4_r5_3_f32(p, 3);
            break;
        }
    }
    for (int c = 0; c < 4; ++c)
        dst4[c] = (nrows == 2) ? fvw_dot2_r5_3_f32(vw, h[c][0], h[c][1]) :
                                   fvw_dot3_r5_3_f32(vw, h[c][0], h[c][1], h[c][2]);
}

static inline void accum_dst_period_r10_3_f32c1(float* D, int di,
                                                  const float* rows[4], const int vw[4])
{
    D[di + 0] = fvw_dot4_f32(vw, fhoriz_phase0_cn1_f32(rows[0]), fhoriz_phase0_cn1_f32(rows[1]),
                             fhoriz_phase0_cn1_f32(rows[2]), fhoriz_phase0_cn1_f32(rows[3]));
    D[di + 1] = fvw_dot4_f32(vw, fhoriz_phase1_cn1_f32(rows[0] + 3), fhoriz_phase1_cn1_f32(rows[1] + 3),
                             fhoriz_phase1_cn1_f32(rows[2] + 3), fhoriz_phase1_cn1_f32(rows[3] + 3));
    D[di + 2] = fvw_dot4_f32(vw, fhoriz_phase2_cn1_f32(rows[0] + 6), fhoriz_phase2_cn1_f32(rows[1] + 6),
                             fhoriz_phase2_cn1_f32(rows[2] + 6), fhoriz_phase2_cn1_f32(rows[3] + 6));
}

static inline void accum_dst_period_r10_3_f32c3(float* D, int di,
                                                  const float* rows[4], const int vw[4])
{
    D[di + 0] = fvw_dot4_f32(vw, fhoriz_phase0_at_c_f32(rows[0], 0), fhoriz_phase0_at_c_f32(rows[1], 0),
                             fhoriz_phase0_at_c_f32(rows[2], 0), fhoriz_phase0_at_c_f32(rows[3], 0));
    D[di + 1] = fvw_dot4_f32(vw, fhoriz_phase0_at_c_f32(rows[0], 1), fhoriz_phase0_at_c_f32(rows[1], 1),
                             fhoriz_phase0_at_c_f32(rows[2], 1), fhoriz_phase0_at_c_f32(rows[3], 1));
    D[di + 2] = fvw_dot4_f32(vw, fhoriz_phase0_at_c_f32(rows[0], 2), fhoriz_phase0_at_c_f32(rows[1], 2),
                             fhoriz_phase0_at_c_f32(rows[2], 2), fhoriz_phase0_at_c_f32(rows[3], 2));

    D[di + 3] = fvw_dot4_f32(vw, fhoriz_phase1_at_c_f32(rows[0] + 9, 0), fhoriz_phase1_at_c_f32(rows[1] + 9, 0),
                             fhoriz_phase1_at_c_f32(rows[2] + 9, 0), fhoriz_phase1_at_c_f32(rows[3] + 9, 0));
    D[di + 4] = fvw_dot4_f32(vw, fhoriz_phase1_at_c_f32(rows[0] + 9, 1), fhoriz_phase1_at_c_f32(rows[1] + 9, 1),
                             fhoriz_phase1_at_c_f32(rows[2] + 9, 1), fhoriz_phase1_at_c_f32(rows[3] + 9, 1));
    D[di + 5] = fvw_dot4_f32(vw, fhoriz_phase1_at_c_f32(rows[0] + 9, 2), fhoriz_phase1_at_c_f32(rows[1] + 9, 2),
                             fhoriz_phase1_at_c_f32(rows[2] + 9, 2), fhoriz_phase1_at_c_f32(rows[3] + 9, 2));

    D[di + 6] = fvw_dot4_f32(vw, fhoriz_phase2_at_c_f32(rows[0] + 18, 0), fhoriz_phase2_at_c_f32(rows[1] + 18, 0),
                             fhoriz_phase2_at_c_f32(rows[2] + 18, 0), fhoriz_phase2_at_c_f32(rows[3] + 18, 0));
    D[di + 7] = fvw_dot4_f32(vw, fhoriz_phase2_at_c_f32(rows[0] + 18, 1), fhoriz_phase2_at_c_f32(rows[1] + 18, 1),
                             fhoriz_phase2_at_c_f32(rows[2] + 18, 1), fhoriz_phase2_at_c_f32(rows[3] + 18, 1));
    D[di + 8] = fvw_dot4_f32(vw, fhoriz_phase2_at_c_f32(rows[0] + 18, 2), fhoriz_phase2_at_c_f32(rows[1] + 18, 2),
                             fhoriz_phase2_at_c_f32(rows[2] + 18, 2), fhoriz_phase2_at_c_f32(rows[3] + 18, 2));
}

static inline void accum_dst_period_r10_3_f32c4(float* D, int di,
                                                  const float* rows[4], const int vw[4])
{
    for (int c = 0; c < 4; ++c)
    {
        D[di + c] = fvw_dot4_f32(vw, fhoriz_phase0_cn4_f32(rows[0], c), fhoriz_phase0_cn4_f32(rows[1], c),
                                 fhoriz_phase0_cn4_f32(rows[2], c), fhoriz_phase0_cn4_f32(rows[3], c));
        D[di + 4 + c] = fvw_dot4_f32(vw, fhoriz_phase1_cn4_f32(rows[0] + 12, c), fhoriz_phase1_cn4_f32(rows[1] + 12, c),
                                     fhoriz_phase1_cn4_f32(rows[2] + 12, c), fhoriz_phase1_cn4_f32(rows[3] + 12, c));
        D[di + 8 + c] = fvw_dot4_f32(vw, fhoriz_phase2_cn4_f32(rows[0] + 24, c), fhoriz_phase2_cn4_f32(rows[1] + 24, c),
                                     fhoriz_phase2_cn4_f32(rows[2] + 24, c), fhoriz_phase2_cn4_f32(rows[3] + 24, c));
    }
}

template<typename AccumPeriodFn>
static void accum_dst_row_r10_3_f32(float* D, int dy, const Mat& src, int dst_width,
                                    int periodElems, int outStride, AccumPeriodFn accumPeriod)
{
    const int sy0 = (dy * 10) / 3;
    const float* rows[4] = {
        src.ptr<float>(sy0), src.ptr<float>(sy0 + 1), src.ptr<float>(sy0 + 2), src.ptr<float>(sy0 + 3)
    };
    int vw[4];
    vert_weights_r10_3(dy, vw);

    int k = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int kBlock = 4;
    for (; k + kBlock <= dst_width / 3; k += kBlock)
    {
        for (int t = 0; t < kBlock; ++t)
        {
            const int off = periodElems * (k + t);
            const float* r[4] = { rows[0] + off, rows[1] + off, rows[2] + off, rows[3] + off };
            accumPeriod(D, outStride * (k + t), r, vw);
        }
    }
#endif
    for (; k < dst_width / 3; ++k)
    {
        const int off = periodElems * k;
        const float* r[4] = { rows[0] + off, rows[1] + off, rows[2] + off, rows[3] + off };
        accumPeriod(D, outStride * k, r, vw);
    }
}

static void accum_dst_row_r10_3_f32c1(float* D, int dy, const Mat& src, int dst_width)
{
    accum_dst_row_r10_3_f32(D, dy, src, dst_width, 10, 3, accum_dst_period_r10_3_f32c1);

    const int sy0 = (dy * 10) / 3;
    int vw[4];
    vert_weights_r10_3(dy, vw);
    int k = dst_width / 3;
    for (int dx = k * 3; dx < dst_width; ++dx)
    {
        const int si = (dx * 10) / 3;
        float h[4];
        for (int ri = 0; ri < 4; ++ri)
        {
            const float* p = src.ptr<float>(sy0 + ri) + si;
            switch (dx % 3)
            {
            case 0: h[ri] = fhoriz_phase0_cn1_f32(p); break;
            case 1: h[ri] = fhoriz_phase1_cn1_f32(p); break;
            default: h[ri] = fhoriz_phase2_cn1_f32(p); break;
            }
        }
        D[dx] = fvw_dot4_f32(vw, h[0], h[1], h[2], h[3]);
    }
}

static void accum_dst_row_r10_3_f32c3(float* D, int dy, const Mat& src, int dst_width)
{
    accum_dst_row_r10_3_f32(D, dy, src, dst_width, 30, 9, accum_dst_period_r10_3_f32c3);

    int k = dst_width / 3;
    for (int dx = k * 3; dx < dst_width; ++dx)
        eval_r10_3_f32_cn3_px(dx, dy, src, D + dx * 3);
}

static void accum_dst_row_r10_3_f32c4(float* D, int dy, const Mat& src, int dst_width)
{
    accum_dst_row_r10_3_f32(D, dy, src, dst_width, 40, 12, accum_dst_period_r10_3_f32c4);

    int k = dst_width / 3;
    for (int dx = k * 3; dx < dst_width; ++dx)
        eval_r10_3_f32_cn4_px(dx, dy, src, D + dx * 4);
}

static inline void accum_dst_period_r5_3_f32c1_2row(float* D, int di,
                                                    const float* rows[3], const int vw[2])
{
    D[di + 0] = fvw_dot2_r5_3_f32(vw, fhoriz_phase0_cn1_r5_3_f32(rows[0]), fhoriz_phase0_cn1_r5_3_f32(rows[1]));
    D[di + 1] = fvw_dot2_r5_3_f32(vw, fhoriz_phase1_cn1_r5_3_f32(rows[0] + 1), fhoriz_phase1_cn1_r5_3_f32(rows[1] + 1));
    D[di + 2] = fvw_dot2_r5_3_f32(vw, fhoriz_phase2_cn1_r5_3_f32(rows[0] + 3), fhoriz_phase2_cn1_r5_3_f32(rows[1] + 3));
}

static inline void accum_dst_period_r5_3_f32c1_3row(float* D, int di,
                                                    const float* rows[3], const int vw[3])
{
    D[di + 0] = fvw_dot3_r5_3_f32(vw, fhoriz_phase0_cn1_r5_3_f32(rows[0]), fhoriz_phase0_cn1_r5_3_f32(rows[1]),
                                  fhoriz_phase0_cn1_r5_3_f32(rows[2]));
    D[di + 1] = fvw_dot3_r5_3_f32(vw, fhoriz_phase1_cn1_r5_3_f32(rows[0] + 1), fhoriz_phase1_cn1_r5_3_f32(rows[1] + 1),
                                  fhoriz_phase1_cn1_r5_3_f32(rows[2] + 1));
    D[di + 2] = fvw_dot3_r5_3_f32(vw, fhoriz_phase2_cn1_r5_3_f32(rows[0] + 3), fhoriz_phase2_cn1_r5_3_f32(rows[1] + 3),
                                  fhoriz_phase2_cn1_r5_3_f32(rows[2] + 3));
}

static inline void accum_dst_period_r5_3_f32c3_2row(float* D, int di,
                                                    const float* rows[3], const int vw[2])
{
    D[di + 0] = fvw_dot2_r5_3_f32(vw, fhoriz_phase0_at_c_r5_3_f32(rows[0], 0), fhoriz_phase0_at_c_r5_3_f32(rows[1], 0));
    D[di + 1] = fvw_dot2_r5_3_f32(vw, fhoriz_phase0_at_c_r5_3_f32(rows[0], 1), fhoriz_phase0_at_c_r5_3_f32(rows[1], 1));
    D[di + 2] = fvw_dot2_r5_3_f32(vw, fhoriz_phase0_at_c_r5_3_f32(rows[0], 2), fhoriz_phase0_at_c_r5_3_f32(rows[1], 2));
    D[di + 3] = fvw_dot2_r5_3_f32(vw, fhoriz_phase1_at_c_r5_3_f32(rows[0] + 3, 0), fhoriz_phase1_at_c_r5_3_f32(rows[1] + 3, 0));
    D[di + 4] = fvw_dot2_r5_3_f32(vw, fhoriz_phase1_at_c_r5_3_f32(rows[0] + 3, 1), fhoriz_phase1_at_c_r5_3_f32(rows[1] + 3, 1));
    D[di + 5] = fvw_dot2_r5_3_f32(vw, fhoriz_phase1_at_c_r5_3_f32(rows[0] + 3, 2), fhoriz_phase1_at_c_r5_3_f32(rows[1] + 3, 2));
    D[di + 6] = fvw_dot2_r5_3_f32(vw, fhoriz_phase2_at_c_r5_3_f32(rows[0] + 9, 0), fhoriz_phase2_at_c_r5_3_f32(rows[1] + 9, 0));
    D[di + 7] = fvw_dot2_r5_3_f32(vw, fhoriz_phase2_at_c_r5_3_f32(rows[0] + 9, 1), fhoriz_phase2_at_c_r5_3_f32(rows[1] + 9, 1));
    D[di + 8] = fvw_dot2_r5_3_f32(vw, fhoriz_phase2_at_c_r5_3_f32(rows[0] + 9, 2), fhoriz_phase2_at_c_r5_3_f32(rows[1] + 9, 2));
}

static inline void accum_dst_period_r5_3_f32c3_3row(float* D, int di,
                                                    const float* rows[3], const int vw[3])
{
    D[di + 0] = fvw_dot3_r5_3_f32(vw, fhoriz_phase0_at_c_r5_3_f32(rows[0], 0), fhoriz_phase0_at_c_r5_3_f32(rows[1], 0),
                                  fhoriz_phase0_at_c_r5_3_f32(rows[2], 0));
    D[di + 1] = fvw_dot3_r5_3_f32(vw, fhoriz_phase0_at_c_r5_3_f32(rows[0], 1), fhoriz_phase0_at_c_r5_3_f32(rows[1], 1),
                                  fhoriz_phase0_at_c_r5_3_f32(rows[2], 1));
    D[di + 2] = fvw_dot3_r5_3_f32(vw, fhoriz_phase0_at_c_r5_3_f32(rows[0], 2), fhoriz_phase0_at_c_r5_3_f32(rows[1], 2),
                                  fhoriz_phase0_at_c_r5_3_f32(rows[2], 2));
    D[di + 3] = fvw_dot3_r5_3_f32(vw, fhoriz_phase1_at_c_r5_3_f32(rows[0] + 3, 0), fhoriz_phase1_at_c_r5_3_f32(rows[1] + 3, 0),
                                  fhoriz_phase1_at_c_r5_3_f32(rows[2] + 3, 0));
    D[di + 4] = fvw_dot3_r5_3_f32(vw, fhoriz_phase1_at_c_r5_3_f32(rows[0] + 3, 1), fhoriz_phase1_at_c_r5_3_f32(rows[1] + 3, 1),
                                  fhoriz_phase1_at_c_r5_3_f32(rows[2] + 3, 1));
    D[di + 5] = fvw_dot3_r5_3_f32(vw, fhoriz_phase1_at_c_r5_3_f32(rows[0] + 3, 2), fhoriz_phase1_at_c_r5_3_f32(rows[1] + 3, 2),
                                  fhoriz_phase1_at_c_r5_3_f32(rows[2] + 3, 2));
    D[di + 6] = fvw_dot3_r5_3_f32(vw, fhoriz_phase2_at_c_r5_3_f32(rows[0] + 9, 0), fhoriz_phase2_at_c_r5_3_f32(rows[1] + 9, 0),
                                  fhoriz_phase2_at_c_r5_3_f32(rows[2] + 9, 0));
    D[di + 7] = fvw_dot3_r5_3_f32(vw, fhoriz_phase2_at_c_r5_3_f32(rows[0] + 9, 1), fhoriz_phase2_at_c_r5_3_f32(rows[1] + 9, 1),
                                  fhoriz_phase2_at_c_r5_3_f32(rows[2] + 9, 1));
    D[di + 8] = fvw_dot3_r5_3_f32(vw, fhoriz_phase2_at_c_r5_3_f32(rows[0] + 9, 2), fhoriz_phase2_at_c_r5_3_f32(rows[1] + 9, 2),
                                  fhoriz_phase2_at_c_r5_3_f32(rows[2] + 9, 2));
}

static inline void accum_dst_period_r5_3_f32c4_2row(float* D, int di,
                                                    const float* rows[3], const int vw[2])
{
    for (int c = 0; c < 4; ++c)
    {
        D[di + c] = fvw_dot2_r5_3_f32(vw, fhoriz_phase0_cn4_r5_3_f32(rows[0], c), fhoriz_phase0_cn4_r5_3_f32(rows[1], c));
        D[di + 4 + c] = fvw_dot2_r5_3_f32(vw, fhoriz_phase1_cn4_r5_3_f32(rows[0] + 4, c), fhoriz_phase1_cn4_r5_3_f32(rows[1] + 4, c));
        D[di + 8 + c] = fvw_dot2_r5_3_f32(vw, fhoriz_phase2_cn4_r5_3_f32(rows[0] + 12, c), fhoriz_phase2_cn4_r5_3_f32(rows[1] + 12, c));
    }
}

static inline void accum_dst_period_r5_3_f32c4_3row(float* D, int di,
                                                    const float* rows[3], const int vw[3])
{
    for (int c = 0; c < 4; ++c)
    {
        D[di + c] = fvw_dot3_r5_3_f32(vw, fhoriz_phase0_cn4_r5_3_f32(rows[0], c), fhoriz_phase0_cn4_r5_3_f32(rows[1], c),
                                      fhoriz_phase0_cn4_r5_3_f32(rows[2], c));
        D[di + 4 + c] = fvw_dot3_r5_3_f32(vw, fhoriz_phase1_cn4_r5_3_f32(rows[0] + 4, c), fhoriz_phase1_cn4_r5_3_f32(rows[1] + 4, c),
                                          fhoriz_phase1_cn4_r5_3_f32(rows[2] + 4, c));
        D[di + 8 + c] = fvw_dot3_r5_3_f32(vw, fhoriz_phase2_cn4_r5_3_f32(rows[0] + 12, c), fhoriz_phase2_cn4_r5_3_f32(rows[1] + 12, c),
                                          fhoriz_phase2_cn4_r5_3_f32(rows[2] + 12, c));
    }
}

template<typename AccumPeriodFn2, typename AccumPeriodFn3>
static void accum_dst_row_r5_3_f32_period(float* D, int dy, const Mat& src, int dst_width,
                                          int periodElems, int outStride, int nrows, const int* vw,
                                          AccumPeriodFn2 accum2, AccumPeriodFn3 accum3)
{
    const int sy0 = (dy * 5) / 3;
    const float* rows[3] = {
        src.ptr<float>(sy0), src.ptr<float>(sy0 + 1), src.ptr<float>(sy0 + 2)
    };

    int k = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int kBlock = 4;
    for (; k + kBlock <= dst_width / 3; k += kBlock)
    {
        for (int t = 0; t < kBlock; ++t)
        {
            const int off = periodElems * (k + t);
            const float* r[3] = { rows[0] + off, rows[1] + off, rows[2] + off };
            if (nrows == 2)
                accum2(D, outStride * (k + t), r, vw);
            else
                accum3(D, outStride * (k + t), r, vw);
        }
    }
#endif
    for (; k < dst_width / 3; ++k)
    {
        const int off = periodElems * k;
        const float* r[3] = { rows[0] + off, rows[1] + off, rows[2] + off };
        if (nrows == 2)
            accum2(D, outStride * k, r, vw);
        else
            accum3(D, outStride * k, r, vw);
    }
}

static void accum_dst_row_r5_3_f32c1(float* D, int dy, const Mat& src, int dst_width)
{
    int vw[3], nrows;
    vert_weights_r5_3(dy, vw, nrows);
    accum_dst_row_r5_3_f32_period(D, dy, src, dst_width, 5, 3, nrows, vw,
                                  accum_dst_period_r5_3_f32c1_2row, accum_dst_period_r5_3_f32c1_3row);

    int k = dst_width / 3;
    for (int dx = k * 3; dx < dst_width; ++dx)
        D[dx] = eval_r5_3_f32_cn1(dx, dy, src);
}

static void accum_dst_row_r5_3_f32c3(float* D, int dy, const Mat& src, int dst_width)
{
    int vw[3], nrows;
    vert_weights_r5_3(dy, vw, nrows);
    accum_dst_row_r5_3_f32_period(D, dy, src, dst_width, 15, 9, nrows, vw,
                                  accum_dst_period_r5_3_f32c3_2row, accum_dst_period_r5_3_f32c3_3row);

    int k = dst_width / 3;
    for (int dx = k * 3; dx < dst_width; ++dx)
        eval_r5_3_f32_cn3_px(dx, dy, src, D + dx * 3);
}

static void accum_dst_row_r5_3_f32c4(float* D, int dy, const Mat& src, int dst_width)
{
    int vw[3], nrows;
    vert_weights_r5_3(dy, vw, nrows);
    accum_dst_row_r5_3_f32_period(D, dy, src, dst_width, 20, 12, nrows, vw,
                                  accum_dst_period_r5_3_f32c4_2row, accum_dst_period_r5_3_f32c4_3row);

    int k = dst_width / 3;
    for (int dx = k * 3; dx < dst_width; ++dx)
        eval_r5_3_f32_cn4_px(dx, dy, src, D + dx * 4);
}

class ResizeAreaRational10_3_F32_Invoker : public ParallelLoopBody
{
public:
    ResizeAreaRational10_3_F32_Invoker(const Mat& _src, Mat& _dst, int _cn) : src(_src), dst(_dst), cn(_cn) {}

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        const int dst_width = dst.cols;
        for (int dy = range.start; dy < range.end; ++dy)
        {
            float* D = dst.ptr<float>(dy);
            if (cn == 1)
                accum_dst_row_r10_3_f32c1(D, dy, src, dst_width);
            else if (cn == 3)
                accum_dst_row_r10_3_f32c3(D, dy, src, dst_width);
            else
                accum_dst_row_r10_3_f32c4(D, dy, src, dst_width);
        }
    }

private:
    const Mat& src;
    Mat& dst;
    int cn;
};

class ResizeAreaRational5_3_F32_Invoker : public ParallelLoopBody
{
public:
    ResizeAreaRational5_3_F32_Invoker(const Mat& _src, Mat& _dst, int _cn) : src(_src), dst(_dst), cn(_cn) {}

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        const int dst_width = dst.cols;
        for (int dy = range.start; dy < range.end; ++dy)
        {
            float* D = dst.ptr<float>(dy);
            if (cn == 1)
                accum_dst_row_r5_3_f32c1(D, dy, src, dst_width);
            else if (cn == 3)
                accum_dst_row_r5_3_f32c3(D, dy, src, dst_width);
            else
                accum_dst_row_r5_3_f32c4(D, dy, src, dst_width);
        }
    }

private:
    const Mat& src;
    Mat& dst;
    int cn;
};


static void resizeAreaRational10_3_f32_(const Mat& src, Mat& dst)
{
    const int cn = src.channels();
    parallel_for_(Range(0, dst.rows),
                  ResizeAreaRational10_3_F32_Invoker(src, dst, cn),
                  dst.total() / ((double)(1 << 16)));
}

static void resizeAreaRational5_3_f32_(const Mat& src, Mat& dst)
{
    const int cn = src.channels();
    parallel_for_(Range(0, dst.rows),
                  ResizeAreaRational5_3_F32_Invoker(src, dst, cn),
                  dst.total() / ((double)(1 << 16)));
}

}  // namespace area_rational_detail

struct AreaHCol
{
    int di;
    int si_left;
    float a_left;
    int sx1, sx2;
    float a_mid;
    int si_right;
    float a_right;
};

static int buildAreaHColPlan( int ssize, int dst_width, int cn, double scale, AreaHCol* plan )
{
    for( int dx = 0; dx < dst_width; dx++ )
    {
        double fsx1 = dx * scale;
        double fsx2 = fsx1 + scale;
        double cellWidth = std::min(scale, ssize - fsx1);

        int sx1 = cvCeil(fsx1), sx2 = cvFloor(fsx2);
        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);

        AreaHCol& p = plan[dx];
        p.di = dx * cn;
        p.si_left = -1;
        p.a_left = 0.f;
        p.sx1 = sx1;
        p.sx2 = sx2;
        p.a_mid = (sx2 > sx1) ? float(1.0 / cellWidth) : 0.f;
        p.si_right = -1;
        p.a_right = 0.f;

        if( sx1 - fsx1 > 1e-3 )
        {
            p.si_left = (sx1 - 1) * cn;
            p.a_left = (float)((sx1 - fsx1) / cellWidth);
        }

        if( fsx2 - sx2 > 1e-3 )
        {
            p.si_right = sx2 * cn;
            p.a_right = (float)(std::min(std::min(fsx2 - sx2, 1.), cellWidth) / cellWidth);
        }
    }
    return dst_width;
}

namespace resize_area_detail {

static const bool use_area_hcol_plan = false;

template<typename WT>
static void zero_row_buf(WT* buf, int width_cn)
{
    for( int dx = 0; dx < width_cn; dx++ )
        buf[dx] = (WT)0;
}

template<>
void zero_row_buf<float>(float* buf, int width_cn)
{
    int dx = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int step = VTraits<v_float32>::vlanes();
    v_float32 vz = vx_setzero_f32();
    for( ; dx + step <= width_cn; dx += step )
        vx_store(buf + dx, vz);
#endif
    for( ; dx < width_cn; dx++ )
        buf[dx] = 0.f;
}

template<typename T, typename WT>
static void fill_row_buf_xtab(WT* buf, int width_cn, const T* S, int cn,
                              const DecimateAlpha* xtab, int xtab_size)
{
    zero_row_buf(buf, width_cn);
    int k;

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
        for( k = 0; k < xtab_size; k++ )
        {
            int sxn = xtab[k].si;
            int dxn = xtab[k].di;
            WT alpha = xtab[k].alpha;
            for( int c = 0; c < cn; c++ )
                buf[dxn + c] += S[sxn + c]*alpha;
        }
}

static void fill_row_buf_hcol_u8c1(float* buf, int width_cn, const uchar* S,
                                   const AreaHCol* plan, int ncols)
{
    CV_UNUSED(width_cn);
    const float a_mid = plan[0].a_mid;

    for( int c = 0; c < ncols; c++ )
    {
        const AreaHCol& p = plan[c];
        float b = 0.f;

        if( p.a_left > 0.f )
            b = (float)S[p.si_left] * p.a_left;

        if( p.sx2 > p.sx1 )
        {
            if( p.sx2 == p.sx1 + 1 )
                b += (float)S[p.sx1] * a_mid;
            else
            {
                float mid = 0.f;
                for( int sx = p.sx1; sx < p.sx2; sx++ )
                    mid += (float)S[sx];
                b += mid * p.a_mid;
            }
        }

        if( p.a_right > 0.f )
            b += (float)S[p.si_right] * p.a_right;

        buf[p.di] = b;
    }
}

static void fill_row_buf_hcol_u8c3(float* buf, int width_cn, const uchar* S,
                                   const AreaHCol* plan, int ncols)
{
    zero_row_buf(buf, width_cn);

    for( int c = 0; c < ncols; c++ )
    {
        const AreaHCol& p = plan[c];
        float b0 = 0.f, b1 = 0.f, b2 = 0.f;

        if( p.a_left > 0.f )
        {
            int sxn = p.si_left;
            b0 = (float)S[sxn] * p.a_left;
            b1 = (float)S[sxn + 1] * p.a_left;
            b2 = (float)S[sxn + 2] * p.a_left;
        }

        for( int sx = p.sx1; sx < p.sx2; sx++ )
        {
            int sxn = sx * 3;
            b0 += S[sxn];
            b1 += S[sxn + 1];
            b2 += S[sxn + 2];
        }
        if( p.sx2 > p.sx1 )
        {
            b0 *= p.a_mid;
            b1 *= p.a_mid;
            b2 *= p.a_mid;
        }

        if( p.a_right > 0.f )
        {
            int sxn = p.si_right;
            b0 += (float)S[sxn] * p.a_right;
            b1 += (float)S[sxn + 1] * p.a_right;
            b2 += (float)S[sxn + 2] * p.a_right;
        }

        buf[p.di] += b0;
        buf[p.di + 1] += b1;
        buf[p.di + 2] += b2;
    }
}

static void fill_row_buf_hcol_u8c4(float* buf, int width_cn, const uchar* S,
                                   const AreaHCol* plan, int ncols)
{
    zero_row_buf(buf, width_cn);

    for( int c = 0; c < ncols; c++ )
    {
        const AreaHCol& p = plan[c];
        float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;

        if( p.a_left > 0.f )
        {
            int sxn = p.si_left;
            b0 = (float)S[sxn] * p.a_left;
            b1 = (float)S[sxn + 1] * p.a_left;
            b2 = (float)S[sxn + 2] * p.a_left;
            b3 = (float)S[sxn + 3] * p.a_left;
        }

        for( int sx = p.sx1; sx < p.sx2; sx++ )
        {
            int sxn = sx * 4;
            b0 += S[sxn];
            b1 += S[sxn + 1];
            b2 += S[sxn + 2];
            b3 += S[sxn + 3];
        }
        if( p.sx2 > p.sx1 )
        {
            b0 *= p.a_mid;
            b1 *= p.a_mid;
            b2 *= p.a_mid;
            b3 *= p.a_mid;
        }

        if( p.a_right > 0.f )
        {
            int sxn = p.si_right;
            b0 += (float)S[sxn] * p.a_right;
            b1 += (float)S[sxn + 1] * p.a_right;
            b2 += (float)S[sxn + 2] * p.a_right;
            b3 += (float)S[sxn + 3] * p.a_right;
        }

        buf[p.di] += b0;
        buf[p.di + 1] += b1;
        buf[p.di + 2] += b2;
        buf[p.di + 3] += b3;
    }
}

template<typename T, typename WT>
struct fill_row_buf
{
    static void call(WT* buf, int width_cn, const T* S, int cn,
                     const DecimateAlpha* xtab, int xtab_size,
                     const AreaHCol* hcol, int hcol_cols)
    {
        CV_UNUSED(hcol);
        CV_UNUSED(hcol_cols);
        fill_row_buf_xtab(buf, width_cn, S, cn, xtab, xtab_size);
    }
};

template<>
struct fill_row_buf<uchar, float>
{
    static void call(float* buf, int width_cn, const uchar* S, int cn,
                     const DecimateAlpha* xtab, int xtab_size,
                     const AreaHCol* hcol, int hcol_cols)
    {
        if( hcol && cn == 1 )
            fill_row_buf_hcol_u8c1(buf, width_cn, S, hcol, hcol_cols);
        else if( hcol && cn == 3 )
            fill_row_buf_hcol_u8c3(buf, width_cn, S, hcol, hcol_cols);
        else if( hcol && cn == 4 )
            fill_row_buf_hcol_u8c4(buf, width_cn, S, hcol, hcol_cols);
        else
            fill_row_buf_xtab(buf, width_cn, S, cn, xtab, xtab_size);
    }
};

}  // namespace resize_area_detail

template<typename T, typename WT> class ResizeArea_Invoker :
    public ParallelLoopBody
{
public:
    ResizeArea_Invoker( const Mat& _src, Mat& _dst,
                        const DecimateAlpha* _xtab, int _xtab_size,
                        const DecimateAlpha* _ytab, int _ytab_size,
                        const int* _tabofs,
                        const AreaHCol* _hcol, int _hcol_cols )
    {
        src = &_src;
        dst = &_dst;
        xtab0 = _xtab;
        xtab_size0 = _xtab_size;
        ytab = _ytab;
        ytab_size = _ytab_size;
        tabofs = _tabofs;
        hcol = _hcol;
        hcol_cols = _hcol_cols;
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
        int j_start = tabofs[range.start], j_end = tabofs[range.end], j, dx, prev_dy = ytab[j_start].di;

        for( dx = 0; dx < dsize.width; dx++ )
            sum[dx] = (WT)0;

        for( j = j_start; j < j_end; j++ )
        {
            WT beta = ytab[j].alpha;
            int dy = ytab[j].di;
            int sy = ytab[j].si;

            {
                const T* S = src->template ptr<T>(sy);
                resize_area_detail::fill_row_buf<T, WT>::call(
                    buf, dsize.width, S, cn, xtab, xtab_size, hcol, hcol_cols);
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
    const AreaHCol* hcol;
    int hcol_cols;
};


template <typename T, typename WT>
static void resizeArea_( const Mat& src, Mat& dst,
                         const DecimateAlpha* xtab, int xtab_size,
                         const DecimateAlpha* ytab, int ytab_size,
                         const int* tabofs,
                         const AreaHCol* hcol, int hcol_cols )
{
    parallel_for_(Range(0, dst.rows),
                 ResizeArea_Invoker<T, WT>(src, dst, xtab, xtab_size, ytab, ytab_size, tabofs, hcol, hcol_cols),
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
                                const int* tabofs,
                                const AreaHCol* hcol, int hcol_cols );


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

static void build_linear_2x_up_tables(int src_width, int src_height, const Size& dsize, int cn,
                                      int* xofs, int* yofs, short* ialpha, short* ibeta,
                                      int& xmin, int& xmax)
{
    const double scale_x = 0.5, scale_y = 0.5;
    const int ksize = 2, ksize2 = 1;
    int dx, dy, sx, sy, k;
    float fx, fy;
    float cbuf[MAX_ESIZE] = {0};

    xmin = 0;
    xmax = dsize.width;

    for( dx = 0; dx < dsize.width; dx++ )
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = cvFloor(fx);
        fx -= sx;

        if( sx < ksize2 - 1 )
        {
            xmin = dx + 1;
            if( sx < 0 )
                fx = 0, sx = 0;
        }

        if( sx + ksize2 >= src_width )
        {
            xmax = std::min(xmax, dx);
            if( sx >= src_width - 1 )
                fx = 0, sx = src_width - 1;
        }

        for( k = 0, sx *= cn; k < cn; k++ )
            xofs[dx * cn + k] = sx + k;

        cbuf[0] = 1.f - fx;
        cbuf[1] = fx;
        for( k = 0; k < ksize; k++ )
            ialpha[dx * cn * ksize + k] = saturate_cast<short>(cbuf[k] * INTER_RESIZE_COEF_SCALE);
        for( ; k < cn * ksize; k++ )
            ialpha[dx * cn * ksize + k] = ialpha[dx * cn * ksize + k - ksize];
    }

    for( dy = 0; dy < dsize.height; dy++ )
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = cvFloor(fy);
        fy -= sy;
        yofs[dy] = sy;

        cbuf[0] = 1.f - fy;
        cbuf[1] = fy;
        for( k = 0; k < ksize; k++ )
            ibeta[dy * ksize + k] = saturate_cast<short>(cbuf[k] * INTER_RESIZE_COEF_SCALE);
    }
    CV_UNUSED(src_height);
}

static void build_linear_2x_up_tables_f32(int src_width, int src_height, const Size& dsize, int cn,
                                            int* xofs, int* yofs, float* alpha, float* beta,
                                            int& xmin, int& xmax)
{
    const double scale_x = 0.5, scale_y = 0.5;
    const int ksize = 2, ksize2 = 1;
    int dx, dy, sx, sy, k;
    float fx, fy;
    float cbuf[MAX_ESIZE] = {0};

    xmin = 0;
    xmax = dsize.width;

    for( dx = 0; dx < dsize.width; dx++ )
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = cvFloor(fx);
        fx -= sx;

        if( sx < ksize2 - 1 )
        {
            xmin = dx + 1;
            if( sx < 0 )
                fx = 0, sx = 0;
        }

        if( sx + ksize2 >= src_width )
        {
            xmax = std::min(xmax, dx);
            if( sx >= src_width - 1 )
                fx = 0, sx = src_width - 1;
        }

        for( k = 0, sx *= cn; k < cn; k++ )
            xofs[dx * cn + k] = sx + k;

        cbuf[0] = 1.f - fx;
        cbuf[1] = fx;
        for( k = 0; k < ksize; k++ )
            alpha[dx * cn * ksize + k] = cbuf[k];
        for( ; k < cn * ksize; k++ )
            alpha[dx * cn * ksize + k] = alpha[dx * cn * ksize + k - ksize];
    }

    for( dy = 0; dy < dsize.height; dy++ )
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = cvFloor(fy);
        fy -= sy;
        yofs[dy] = sy;

        cbuf[0] = 1.f - fy;
        cbuf[1] = fy;
        for( k = 0; k < ksize; k++ )
            beta[dy * ksize + k] = cbuf[k];
    }
    CV_UNUSED(src_height);
}

#if (CV_SIMD || CV_SIMD_SCALABLE)

static inline void linear2x_row_pair_tail_cn1(uchar* dst0, uchar* dst1,
                                              const uchar* src0, const uchar* src1, int sx, int sw)
{
    for( ; sx < sw - 1; ++sx )
    {
        const uchar p00 = src0[sx], p01 = src0[sx + 1];
        const uchar p10 = src1[sx], p11 = src1[sx + 1];
        const int dx = sx << 1;
        dst0[dx] = p00;
        dst0[dx + 1] = (uchar)((p00 + p01 + 1) >> 1);
        dst1[dx] = (uchar)((p00 + p10 + 1) >> 1);
        dst1[dx + 1] = (uchar)((p00 + p01 + p10 + p11 + 2) >> 2);
    }

    const int edx = (sw - 1) << 1;
    const uchar p0 = src0[sw - 1], p1 = src1[sw - 1];
    dst0[edx] = dst0[edx + 1] = p0;
    dst1[edx] = dst1[edx + 1] = (uchar)((p0 + p1 + 1) >> 1);
}

static inline void linear2x_pack_interleaved_u8(uchar* dst, const v_uint16& even, const v_uint16& odd)
{
    v_uint16 zip0, zip1;
    v_zip(even, odd, zip0, zip1);
    v_pack_store(dst, zip0);
    v_pack_store(dst + VTraits<v_uint16>::vlanes(), zip1);
}

static inline v_uint16 linear2x_avg2_u16(const v_uint16& a, const v_uint16& b)
{
    return v_shr(v_add(v_add(a, b), vx_setall_u16(1)), 1);
}

static inline v_uint16 linear2x_avg4_u16(const v_uint16& a, const v_uint16& b,
                                           const v_uint16& c, const v_uint16& d)
{
    return v_shr(v_add(v_add(v_add(a, b), v_add(c, d)), vx_setall_u16(2)), 2);
}

static inline void linear2x_fused_pack2rows_u8(uchar* dst0, uchar* dst1,
                                               const v_uint16& a0, const v_uint16& a1,
                                               const v_uint16& b0, const v_uint16& b1,
                                               const v_uint16& c0, const v_uint16& c1,
                                               const v_uint16& d0, const v_uint16& d1)
{
    const int n = VTraits<v_uint8>::vlanes();
    linear2x_pack_interleaved_u8(dst0, a0, linear2x_avg2_u16(a0, b0));
    linear2x_pack_interleaved_u8(dst0 + n, a1, linear2x_avg2_u16(a1, b1));
    linear2x_pack_interleaved_u8(dst1, linear2x_avg2_u16(a0, c0), linear2x_avg4_u16(a0, b0, c0, d0));
    linear2x_pack_interleaved_u8(dst1 + n, linear2x_avg2_u16(a1, c1), linear2x_avg4_u16(a1, b1, c1, d1));
}

static inline void linear2x_fused_planes_vec(uchar* dst0, uchar* dst1,
                                             const v_uint8& s0a, const v_uint8& s0b,
                                             const v_uint8& s1a, const v_uint8& s1b)
{
    v_uint16 a0, a1, b0, b1, c0, c1, d0, d1;
    v_expand(s0a, a0, a1);
    v_expand(s0b, b0, b1);
    v_expand(s1a, c0, c1);
    v_expand(s1b, d0, d1);
    linear2x_fused_pack2rows_u8(dst0, dst1, a0, a1, b0, b1, c0, c1, d0, d1);
}

static inline void linear2x_fused_block_cn1(uchar* dst0, uchar* dst1,
                                            const uchar* src0, const uchar* src1)
{
    linear2x_fused_planes_vec(dst0, dst1, vx_load(src0), vx_load(src0 + 1),
                              vx_load(src1), vx_load(src1 + 1));
}

static inline void linear2x_fused_row_pair_cn1(uchar* dst0, uchar* dst1,
                                               const uchar* src0, const uchar* src1, int sw)
{
    const int n = VTraits<v_uint8>::vlanes();
    int sx = 0;

    for( ; sx <= sw - (n + 1); sx += n )
        linear2x_fused_block_cn1(dst0 + (sx << 1), dst1 + (sx << 1), src0 + sx, src1 + sx);

    linear2x_row_pair_tail_cn1(dst0, dst1, src0, src1, sx, sw);
}

static inline void linear2x_fused_row_pair_cn3(uchar* dst0, uchar* dst1,
                                               const uchar* src0, const uchar* src1, int sw)
{
    const int n = VTraits<v_uint8>::vlanes();
    int sx = 0;
    uchar CV_DECL_ALIGNED(CV_SIMD_WIDTH) dr0[VTraits<v_uint8>::max_nlanes * 2];
    uchar CV_DECL_ALIGNED(CV_SIMD_WIDTH) dr1[VTraits<v_uint8>::max_nlanes * 2];
    uchar CV_DECL_ALIGNED(CV_SIMD_WIDTH) dg0[VTraits<v_uint8>::max_nlanes * 2];
    uchar CV_DECL_ALIGNED(CV_SIMD_WIDTH) dg1[VTraits<v_uint8>::max_nlanes * 2];
    uchar CV_DECL_ALIGNED(CV_SIMD_WIDTH) db0[VTraits<v_uint8>::max_nlanes * 2];
    uchar CV_DECL_ALIGNED(CV_SIMD_WIDTH) db1[VTraits<v_uint8>::max_nlanes * 2];

    for( ; sx <= sw - (n + 1); sx += n )
    {
        v_uint8 r00, g00, b00, r01, g01, b01, r10, g10, b10, r11, g11, b11;
        v_load_deinterleave(src0 + sx * 3, r00, g00, b00);
        v_load_deinterleave(src0 + (sx + 1) * 3, r01, g01, b01);
        v_load_deinterleave(src1 + sx * 3, r10, g10, b10);
        v_load_deinterleave(src1 + (sx + 1) * 3, r11, g11, b11);

        linear2x_fused_planes_vec(dr0, dr1, r00, r01, r10, r11);
        linear2x_fused_planes_vec(dg0, dg1, g00, g01, g10, g11);
        linear2x_fused_planes_vec(db0, db1, b00, b01, b10, b11);

        v_store_interleave(dst0 + (sx << 1) * 3, vx_load(dr0), vx_load(dg0), vx_load(db0));
        v_store_interleave(dst0 + (sx << 1) * 3 + n * 3, vx_load(dr0 + n), vx_load(dg0 + n), vx_load(db0 + n));
        v_store_interleave(dst1 + (sx << 1) * 3, vx_load(dr1), vx_load(dg1), vx_load(db1));
        v_store_interleave(dst1 + (sx << 1) * 3 + n * 3, vx_load(dr1 + n), vx_load(dg1 + n), vx_load(db1 + n));
    }

    for( ; sx < sw - 1; ++sx )
    {
        const uchar* p00 = src0 + sx * 3;
        const uchar* p01 = src0 + (sx + 1) * 3;
        const uchar* p10 = src1 + sx * 3;
        const uchar* p11 = src1 + (sx + 1) * 3;
        uchar* d0 = dst0 + (sx << 1) * 3;
        uchar* d1 = dst1 + (sx << 1) * 3;
        for( int c = 0; c < 3; ++c )
        {
            const uchar a = p00[c], b = p01[c], c0 = p10[c], d = p11[c];
            d0[c] = a;
            d0[c + 3] = (uchar)((a + b + 1) >> 1);
            d1[c] = (uchar)((a + c0 + 1) >> 1);
            d1[c + 3] = (uchar)((a + b + c0 + d + 2) >> 2);
        }
    }

    const int edx = (sw - 1) << 1;
    const uchar* p0 = src0 + (sw - 1) * 3;
    const uchar* p1 = src1 + (sw - 1) * 3;
    uchar* d0 = dst0 + edx * 3;
    uchar* d1 = dst1 + edx * 3;
    for( int c = 0; c < 3; ++c )
    {
        d0[c] = d0[c + 3] = p0[c];
        d1[c] = d1[c + 3] = (uchar)((p0[c] + p1[c] + 1) >> 1);
    }
}

static inline void linear2x_fused_row_pair_cn4(uchar* dst0, uchar* dst1,
                                               const uchar* src0, const uchar* src1, int sw)
{
    const int n = VTraits<v_uint8>::vlanes();
    int sx = 0;
    uchar CV_DECL_ALIGNED(CV_SIMD_WIDTH) d00[VTraits<v_uint8>::max_nlanes * 2];
    uchar CV_DECL_ALIGNED(CV_SIMD_WIDTH) d01[VTraits<v_uint8>::max_nlanes * 2];
    uchar CV_DECL_ALIGNED(CV_SIMD_WIDTH) d02[VTraits<v_uint8>::max_nlanes * 2];
    uchar CV_DECL_ALIGNED(CV_SIMD_WIDTH) d03[VTraits<v_uint8>::max_nlanes * 2];
    uchar CV_DECL_ALIGNED(CV_SIMD_WIDTH) d10[VTraits<v_uint8>::max_nlanes * 2];
    uchar CV_DECL_ALIGNED(CV_SIMD_WIDTH) d11[VTraits<v_uint8>::max_nlanes * 2];
    uchar CV_DECL_ALIGNED(CV_SIMD_WIDTH) d12[VTraits<v_uint8>::max_nlanes * 2];
    uchar CV_DECL_ALIGNED(CV_SIMD_WIDTH) d13[VTraits<v_uint8>::max_nlanes * 2];

    for( ; sx <= sw - (n + 1); sx += n )
    {
        v_uint8 c00, c01, c02, c03, c10, c11, c12, c13;
        v_uint8 c20, c21, c22, c23, c30, c31, c32, c33;
        v_load_deinterleave(src0 + (sx << 2), c00, c01, c02, c03);
        v_load_deinterleave(src0 + ((sx + 1) << 2), c10, c11, c12, c13);
        v_load_deinterleave(src1 + (sx << 2), c20, c21, c22, c23);
        v_load_deinterleave(src1 + ((sx + 1) << 2), c30, c31, c32, c33);

        linear2x_fused_planes_vec(d00, d10, c00, c10, c20, c30);
        linear2x_fused_planes_vec(d01, d11, c01, c11, c21, c31);
        linear2x_fused_planes_vec(d02, d12, c02, c12, c22, c32);
        linear2x_fused_planes_vec(d03, d13, c03, c13, c23, c33);

        v_store_interleave(dst0 + (sx << 3), vx_load(d00), vx_load(d01), vx_load(d02), vx_load(d03));
        v_store_interleave(dst0 + (sx << 3) + (n << 2), vx_load(d00 + n), vx_load(d01 + n),
                           vx_load(d02 + n), vx_load(d03 + n));
        v_store_interleave(dst1 + (sx << 3), vx_load(d10), vx_load(d11), vx_load(d12), vx_load(d13));
        v_store_interleave(dst1 + (sx << 3) + (n << 2), vx_load(d10 + n), vx_load(d11 + n),
                           vx_load(d12 + n), vx_load(d13 + n));
    }

    for( ; sx < sw - 1; ++sx )
    {
        const uchar* p00 = src0 + (sx << 2);
        const uchar* p01 = src0 + ((sx + 1) << 2);
        const uchar* p10 = src1 + (sx << 2);
        const uchar* p11 = src1 + ((sx + 1) << 2);
        uchar* d0 = dst0 + (sx << 3);
        uchar* d1 = dst1 + (sx << 3);
        for( int c = 0; c < 4; ++c )
        {
            const uchar a = p00[c], b = p01[c], c0 = p10[c], d = p11[c];
            d0[(c << 1)] = a;
            d0[(c << 1) + 1] = (uchar)((a + b + 1) >> 1);
            d1[(c << 1)] = (uchar)((a + c0 + 1) >> 1);
            d1[(c << 1) + 1] = (uchar)((a + b + c0 + d + 2) >> 2);
        }
    }

    const int edx = (sw - 1) << 3;
    const uchar* p0 = src0 + ((sw - 1) << 2);
    const uchar* p1 = src1 + ((sw - 1) << 2);
    uchar* d0 = dst0 + edx;
    uchar* d1 = dst1 + edx;
    for( int c = 0; c < 4; ++c )
    {
        d0[(c << 1)] = d0[(c << 1) + 1] = p0[c];
        d1[(c << 1)] = d1[(c << 1) + 1] = (uchar)((p0[c] + p1[c] + 1) >> 1);
    }
}

class resizeLinear2xFused8u_Invoker : public ParallelLoopBody
{
public:
    resizeLinear2xFused8u_Invoker(const Mat& _src, Mat& _dst) : src(_src), dst(_dst) {}

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        const int cn = src.channels();
        const int sw = src.cols;
        const int sh = src.rows;

        for( int sy = range.start; sy < range.end; ++sy )
        {
            const int dy = sy << 1;
            uchar* dst0 = dst.ptr(dy);
            uchar* dst1 = dst.ptr(dy + 1);
            const uchar* src0 = src.ptr(sy);
            const uchar* src1 = src.ptr(std::min(sy + 1, sh - 1));

            if( cn == 1 )
                linear2x_fused_row_pair_cn1(dst0, dst1, src0, src1, sw);
            else if( cn == 3 )
                linear2x_fused_row_pair_cn3(dst0, dst1, src0, src1, sw);
            else if( cn == 4 )
                linear2x_fused_row_pair_cn4(dst0, dst1, src0, src1, sw);
        }
    }

private:
    const Mat& src;
    Mat& dst;
};

static inline void resizeLinear2xUp_8u_fused(const Mat& src, Mat& dst)
{
    Range range(0, src.rows);
    resizeLinear2xFused8u_Invoker invoker(src, dst);
    parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
}

#endif // (CV_SIMD || CV_SIMD_SCALABLE)

static void resizeLinear2xUp_8u_generic(const Mat& src, Mat& dst)
{
    const int cn = src.channels();
    const Size& dsize = dst.size();
    const int width = dsize.width * cn;
    const int ksize = 2;
    int xmin = 0, xmax = dsize.width;

    AutoBuffer<uchar> _buffer((width + dsize.height) * (sizeof(int) + sizeof(short) * ksize));
    int* xofs = (int*)_buffer.data();
    int* yofs = xofs + width;
    short* ialpha = (short*)(yofs + dsize.height);
    short* ibeta = ialpha + width * ksize;

    build_linear_2x_up_tables(src.cols, src.rows, dsize, cn, xofs, yofs, ialpha, ibeta, xmin, xmax);

    resizeGeneric_<
        HResizeLinear<uchar, int, short, INTER_RESIZE_COEF_SCALE, HResizeLinearVec_8u32s>,
        VResizeLinear<uchar, int, short,
            FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS * 2>,
            VResizeLinearVec_32s8u> >(src, dst, xofs, ialpha, yofs, ibeta, xmin, xmax, ksize);
}

static void resizeLinear2xUp_8u(const Mat& src, Mat& dst)
{
    // Generic fixpt separable path is the INTER_LINEAR reference. The fused single-pass
    // kernel remains available for follow-up once bit-exact equivalence is validated.
    resizeLinear2xUp_8u_generic(src, dst);
}

static inline void resizeLinear2xUp_32f(const Mat& src, Mat& dst)
{
    const int cn = src.channels();
    const Size& dsize = dst.size();
    const int width = dsize.width * cn;
    const int ksize = 2;
    int xmin = 0, xmax = dsize.width;

    AutoBuffer<uchar> _buffer((width + dsize.height) * (sizeof(int) + sizeof(float) * ksize));
    int* xofs = (int*)_buffer.data();
    int* yofs = xofs + width;
    float* alpha = (float*)(yofs + dsize.height);
    float* beta = alpha + width * ksize;

    build_linear_2x_up_tables_f32(src.cols, src.rows, dsize, cn, xofs, yofs, alpha, beta, xmin, xmax);

    resizeGeneric_<
        HResizeLinear<float, float, float, 1, HResizeLinearVec_32f>,
        VResizeLinear<float, float, float, Cast<float, float>, VResizeLinearVec_32f> >(src, dst, xofs, alpha, yofs, beta, xmin, xmax, ksize);
}

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

    CALL_HAL(resize, cv_hal_resize, src_type, src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, inv_scale_x, inv_scale_y, interpolation);

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

    if( interpolation == INTER_LINEAR &&
        linear_up_2x_detail::is_2x_up(src_width, dsize.width, src_height, dsize.height,
                                      inv_scale_x, inv_scale_y) )
    {
        if( depth == CV_8U && cn != 2 )
        {
            linear_up_2x_detail::resizeLinear2xUp_8u(src, dst);
            return;
        }
    }

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
        resizeArea_<uchar, float>, 0, resizeArea_<ushort, float>,
        resizeArea_<short, float>, 0, resizeArea_<float, float>,
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
            CV_Assert( func != 0 && cn <= 4 );

            AutoBuffer<DecimateAlpha> _xytab((src_width + src_height)*2);
            DecimateAlpha* xtab = _xytab.data(), *ytab = xtab + src_width*2;
            int xtab_size = 0, ytab_size = 0;

            if( depth == CV_8U && (cn == 1 || cn == 3 || cn == 4) &&
                area_rational_detail::is_rational_10_3(src_width, dsize.width, src_height, dsize.height, scale_x, scale_y) )
            {
                area_rational_detail::resizeAreaRational10_3_u8_(src, dst);
                return;
            }

            if( depth == CV_8U && (cn == 1 || cn == 3 || cn == 4) &&
                area_rational_detail::is_rational_5_3(src_width, dsize.width, src_height, dsize.height, scale_x, scale_y) )
            {
                area_rational_detail::resizeAreaRational5_3_u8_(src, dst);
                return;
            }

            if( depth == CV_32F && (cn == 1 || cn == 3 || cn == 4) &&
                area_rational_detail::is_rational_10_3(src_width, dsize.width, src_height, dsize.height, scale_x, scale_y) )
            {
                area_rational_detail::resizeAreaRational10_3_f32_(src, dst);
                return;
            }

            if( depth == CV_32F && (cn == 1 || cn == 3 || cn == 4) &&
                area_rational_detail::is_rational_5_3(src_width, dsize.width, src_height, dsize.height, scale_x, scale_y) )
            {
                area_rational_detail::resizeAreaRational5_3_f32_(src, dst);
                return;
            }

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

            if( depth == CV_8U && cn == 1 )
            {
                AutoBuffer<AreaHCol> _hcol(dsize.width);
                const int hcol_cols = buildAreaHColPlan(src_width, dsize.width, cn, scale_x, _hcol.data());
                func( src, dst, xtab, 0, ytab, ytab_size, tabofs, _hcol.data(), hcol_cols );
                return;
            }

            xtab_size = computeResizeAreaTab(src_width, dsize.width, cn, scale_x, xtab);

            const AreaHCol* hcol = 0;
            int hcol_cols = 0;
            AutoBuffer<AreaHCol> _hcol;
            if( resize_area_detail::use_area_hcol_plan && depth == CV_8U && (cn == 3 || cn == 4) )
            {
                _hcol.allocate(dsize.width);
                hcol_cols = buildAreaHColPlan(src_width, dsize.width, cn, scale_x, _hcol.data());
                hcol = _hcol.data();
            }

            func( src, dst, xtab, xtab_size, ytab, ytab_size, tabofs, hcol, hcol_cols );
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
