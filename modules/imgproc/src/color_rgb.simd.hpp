// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace hal {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
// forward declarations

void cvtBGRtoBGR(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, int dcn, bool swapBlue);
void cvtBGRtoBGR5x5(const uchar * src_data, size_t src_step,
                    uchar * dst_data, size_t dst_step,
                    int width, int height,
                    int scn, bool swapBlue, int greenBits);
void cvtBGR5x5toBGR(const uchar * src_data, size_t src_step,
                    uchar * dst_data, size_t dst_step,
                    int width, int height,
                    int dcn, bool swapBlue, int greenBits);
void cvtBGRtoGray(const uchar * src_data, size_t src_step,
                  uchar * dst_data, size_t dst_step,
                  int width, int height,
                  int depth, int scn, bool swapBlue);
void cvtGraytoBGR(const uchar * src_data, size_t src_step,
                  uchar * dst_data, size_t dst_step,
                  int width, int height,
                  int depth, int dcn);
void cvtBGR5x5toGray(const uchar * src_data, size_t src_step,
                     uchar * dst_data, size_t dst_step,
                     int width, int height,
                     int greenBits);
void cvtGraytoBGR5x5(const uchar * src_data, size_t src_step,
                     uchar * dst_data, size_t dst_step,
                     int width, int height,
                     int greenBits);
void cvtRGBAtoMultipliedRGBA(const uchar * src_data, size_t src_step,
                             uchar * dst_data, size_t dst_step,
                             int width, int height);
void cvtMultipliedRGBAtoRGBA(const uchar * src_data, size_t src_step,
                             uchar * dst_data, size_t dst_step,
                             int width, int height);


#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#if defined(CV_CPU_BASELINE_MODE)
// included in color.hpp
#else
#include "color.simd_helpers.hpp"
#endif

namespace {
////////////////// Various 3/4-channel to 3/4-channel RGB transformations /////////////////

template<typename _Tp> struct v_type;

template<>
struct v_type<uchar>{
    typedef v_uint8 t;
};

template<>
struct v_type<ushort>{
    typedef v_uint16 t;
};

template<>
struct v_type<float>{
    typedef v_float32 t;
};

template<typename _Tp> struct v_set;

template<>
struct v_set<uchar>
{
    static inline v_type<uchar>::t set(uchar x)
    {
        return vx_setall_u8(x);
    }
};

template<>
struct v_set<ushort>
{
    static inline v_type<ushort>::t set(ushort x)
    {
        return vx_setall_u16(x);
    }
};

template<>
struct v_set<float>
{
    static inline v_type<float>::t set(float x)
    {
        return vx_setall_f32(x);
    }
};

template<typename _Tp>
struct RGB2RGB
{
    typedef _Tp channel_type;
    typedef typename v_type<_Tp>::t vt;

    RGB2RGB(int _srccn, int _dstcn, int _blueIdx) :
        srccn(_srccn), dstcn(_dstcn), blueIdx(_blueIdx)
    {
        CV_Assert(srccn == 3 || srccn == 4);
        CV_Assert(dstcn == 3 || dstcn == 4);
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn, dcn = dstcn, bi = blueIdx;
        int i = 0;
        _Tp alphav = ColorChannel<_Tp>::max();

#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<vt>::vlanes();

        for(; i <= n-vsize;
            i += vsize, src += vsize*scn, dst += vsize*dcn)
        {
            vt a, b, c, d;
            if(scn == 4)
            {
                v_load_deinterleave(src, a, b, c, d);
            }
            else
            {
                v_load_deinterleave(src, a, b, c);
                d = v_set<_Tp>::set(alphav);
            }
            if(bi == 2) {
                #if CV_SIMD_SCALABLE
                auto t = a; a = c; c = t; // swap(a, c);
                #else
                swap(a, c);
                #endif
            }

            if(dcn == 4)
            {
                v_store_interleave(dst, a, b, c, d);
            }
            else
            {
                v_store_interleave(dst, a, b, c);
            }
        }
        vx_cleanup();
#endif
        for ( ; i < n; i++, src += scn, dst += dcn )
        {
            _Tp t0 = src[0], t1 = src[1], t2 = src[2];
            dst[bi  ] = t0;
            dst[1]    = t1;
            dst[bi^2] = t2;
            if(dcn == 4)
            {
                _Tp d = scn == 4 ? src[3] : alphav;
                dst[3] = d;
            }
        }
    }

    int srccn, dstcn, blueIdx;
};


/////////// Transforming 16-bit (565 or 555) RGB to/from 24/32-bit (888[8]) RGB //////////

struct RGB5x52RGB
{
    typedef uchar channel_type;

    RGB5x52RGB(int _dstcn, int _blueIdx, int _greenBits)
        : dstcn(_dstcn), blueIdx(_blueIdx), greenBits(_greenBits)
    { }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx, gb = greenBits;
        int i = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<v_uint8>::vlanes();
        v_uint8 vz = vx_setzero_u8(), vn0 = vx_setall_u8(255);
        for(; i <= n-vsize;
            i += vsize, src += vsize*sizeof(ushort), dst += vsize*dcn)
        {
            v_uint16 t0 = v_reinterpret_as_u16(vx_load(src));
            v_uint16 t1 = v_reinterpret_as_u16(vx_load(src +
                                                       sizeof(ushort)*VTraits<v_uint16>::vlanes()));

            //TODO: shorten registers use when v_interleave is available
            v_uint8 r, g, b, a;
            v_uint16 b0 = v_shr<8>(v_shl<11>(t0));
            v_uint16 b1 = v_shr<8>(v_shl<11>(t1));
            b = v_pack(b0, b1);

            v_uint16 g0, g1, r0, r1, a0, a1;

            if( gb == 6 )
            {
                g0 = v_shr<8>(v_shl<10>(v_shr<5>(t0)));
                g1 = v_shr<8>(v_shl<10>(v_shr<5>(t1)));

                r0 = v_shl<3>(v_shr<11>(t0));
                r1 = v_shl<3>(v_shr<11>(t1));

                a = vn0;
            }
            else
            {
                g0 = v_shr<8>(v_shl<11>(v_shr<5>(t0)));
                g1 = v_shr<8>(v_shl<11>(v_shr<5>(t1)));

                r0 = v_shr<8>(v_shl<11>(v_shr<10>(t0)));
                r1 = v_shr<8>(v_shl<11>(v_shr<10>(t1)));

                a0 = v_shr<15>(t0);
                a1 = v_shr<15>(t1);
                a = v_pack(a0, a1);
                a = v_ne(a, vz);
            }
            g = v_pack(g0, g1);
            r = v_pack(r0, r1);

            if(bidx == 2) {
                #if CV_SIMD_SCALABLE
                auto t = r; r = b; b = t; // swap(b, r);
                #else
                swap(b, r);
                #endif
            }
            if(dcn == 4)
            {
                v_store_interleave(dst, b, g, r, a);
            }
            else
            {
                v_store_interleave(dst, b, g, r);
            }
        }
        vx_cleanup();
#endif

        for( ; i < n; i++, src += sizeof(ushort), dst += dcn )
        {
            unsigned t = ((const ushort*)src)[0];
            uchar b, g, r, a;

            b = (uchar)(t << 3);

            if( gb == 6 )
            {
                g = (uchar)((t >> 3) & ~3);
                r = (uchar)((t >> 8) & ~7);
                a = 255;
            }
            else
            {
                g = (uchar)((t >> 2) & ~7);
                r = (uchar)((t >> 7) & ~7);
                a = (uchar)(((t & 0x8000) >> 15) * 255);
            }

            dst[bidx]     = b;
            dst[1]        = g;
            dst[bidx ^ 2] = r;
            if( dcn == 4 )
                dst[3] = a;
        }
    }

    int dstcn, blueIdx, greenBits;
};


struct RGB2RGB5x5
{
    typedef uchar channel_type;

    RGB2RGB5x5(int _srccn, int _blueIdx, int _greenBits)
        : srccn(_srccn), blueIdx(_blueIdx), greenBits(_greenBits)
    { }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int scn = srccn, bidx = blueIdx, gb = greenBits;
        int i = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<v_uint8>::vlanes();
        v_uint16 vn3 = vx_setall_u16((ushort)(~3));
        v_uint16 vn7 = vx_setall_u16((ushort)(~7));
        v_uint16 vz = vx_setzero_u16();
        v_uint8 v7 = vx_setall_u8((uchar)(~7));
        for(; i <= n-vsize;
            i += vsize, src += vsize*scn, dst += vsize*sizeof(ushort))
        {
            v_uint8 r, g, b, a;
            if(scn == 3)
            {
                v_load_deinterleave(src, b, g, r);
                a = vx_setzero_u8();
            }
            else
            {
                v_load_deinterleave(src, b, g, r, a);
            }
            if(bidx == 2){
                #if CV_SIMD_SCALABLE
                auto t = r; r = b; b = t; // swap(b, r);
                #else
                swap(b, r);
                #endif
            }

            r = v_and(r, v7);

            //TODO: shorten registers use when v_deinterleave is available
            v_uint16 r0, r1, g0, g1, b0, b1, a0, a1;
            v_expand(r, r0, r1);
            v_expand(g, g0, g1);
            v_expand(b, b0, b1);
            v_expand(a, a0, a1);

            v_uint16 d0, d1;

            b0 = v_shr<3>(b0);
            b1 = v_shr<3>(b1);
            a0 = v_shl<15>(v_ne(a0, vz));
            a1 = v_shl<15>(v_ne(a1, vz));

            if(gb == 6)
            {
                d0 = v_or(v_or(b0, v_shl<3>(v_and(g0, vn3))), v_shl<8>(r0));
                d1 = v_or(v_or(b1, v_shl<3>(v_and(g1, vn3))), v_shl<8>(r1));
            }
            else
            {
                d0 = v_or(v_or(v_or(b0, v_shl<2>(v_and(g0, vn7))), v_shl<7>(r0)), a0);
                d1 = v_or(v_or(v_or(b1, v_shl<2>(v_and(g1, vn7))), v_shl<7>(r1)), a1);
            }

            v_store((ushort*)dst, d0);
            v_store(((ushort*)dst) + vsize/2, d1);
        }
        vx_cleanup();
#endif
        for ( ; i < n; i++, src += scn, dst += sizeof(ushort) )
        {
            uchar r = src[bidx^2];
            uchar g = src[1];
            uchar b = src[bidx];
            uchar a = scn == 4 ? src[3] : 0;

            ushort d;
            if (gb == 6)
            {
                d = (ushort)((b >> 3)|((g & ~3) << 3)|((r & ~7) << 8));
            }
            else
            {
                d = (ushort)((b >> 3)|((g & ~7) << 2)|((r & ~7) << 7)|(a ? 0x8000 : 0));
            }
            ((ushort*)dst)[0] = d;
        }
    }

    int srccn, blueIdx, greenBits;
};


///////////////////////////////// Color to/from Grayscale ////////////////////////////////

template<typename _Tp>
struct Gray2RGB
{
    typedef _Tp channel_type;
    typedef typename v_type<_Tp>::t vt;

    Gray2RGB(int _dstcn) : dstcn(_dstcn) {}
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int dcn = dstcn;
        int i = 0;
        _Tp alpha = ColorChannel<_Tp>::max();

#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<vt>::vlanes();
        vt valpha = v_set<_Tp>::set(alpha);
        for(; i <= n-vsize;
            i += vsize, src += vsize, dst += vsize*dcn)
        {
            vt g = vx_load(src);

            if(dcn == 3)
            {
                v_store_interleave(dst, g, g, g);
            }
            else
            {
                v_store_interleave(dst, g, g, g, valpha);
            }
        }
        vx_cleanup();
#endif
        for ( ; i < n; i++, src++, dst += dcn )
        {
            dst[0] = dst[1] = dst[2] = src[0];
            if(dcn == 4)
                dst[3] = alpha;
        }
    }

    int dstcn;
};


struct Gray2RGB5x5
{
    typedef uchar channel_type;

    Gray2RGB5x5(int _greenBits) : greenBits(_greenBits)
    { }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int gb = greenBits;
        int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<v_uint16>::vlanes();
        v_uint16 v3 = vx_setall_u16((ushort)(~3));
        for(; i <= n-vsize;
            i += vsize, src += vsize, dst += vsize*sizeof(ushort))
        {
            v_uint8 t8 = vx_load_low(src);
            v_uint16 t = v_expand_low(t8);

            v_uint16 t3 = v_shr<3>(t);

            v_uint16 d = t3;
            if(gb == 6)
            {
                d = v_or(d, v_or(v_shl<3>(v_and(t, v3)), v_shl<11>(t3)));
            }
            else
            {
                d = v_or(d, v_or(v_shl<5>(t3), v_shl<10>(t3)));
            }

            v_store((ushort*)dst, d);
        }
        vx_cleanup();
#endif

        for( ; i < n; i++, src++, dst += sizeof(ushort))
        {
            int t = src[0];
            int t3 = t >> 3;
            ushort d;
            if( gb == 6 )
            {
                d = (ushort)(t3 |((t & ~3) << 3)|(t3 << 11));
            }
            else
            {
                d = (ushort)(t3 |(t3 << 5)|(t3 << 10));
            }
            ((ushort*)dst)[0] = d;
        }
    }
    int greenBits;
};


struct RGB5x52Gray
{
    typedef uchar channel_type;

    static const int BY = BY15;
    static const int GY = GY15;
    static const int RY = RY15;
    static const int shift = gray_shift;

    RGB5x52Gray(int _greenBits) : greenBits(_greenBits)
    {
        CV_Assert(BY + GY + RY == (1 << shift));
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int gb = greenBits;
        int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<v_uint16>::vlanes();

        v_int16 bg2y;
        v_int16 r12y;
        v_int16 dummy;
        v_zip(vx_setall_s16(BY), vx_setall_s16(GY), bg2y, dummy);
        v_zip(vx_setall_s16(RY), vx_setall_s16( 1), r12y, dummy);
        v_int16 delta = vx_setall_s16(1 << (shift-1));

        for(; i <= n-vsize;
            i += vsize, src += vsize*sizeof(ushort), dst += vsize)
        {
            v_uint16 t = vx_load((ushort*)src);

            v_uint16 r, g, b;
            b = v_shr<8>(v_shl<11>(t));

            if(gb == 5)
            {
                g = v_shr<8>(v_shl<11>(v_shr<5>(t)));
                r = v_shr<8>(v_shl<11>(v_shr<10>(t)));
            }
            else
            {
                g = v_shr<8>(v_shl<10>(v_shr<5>(t)));
                r = v_shl<3>(v_shr<11>(t));
            }

            v_uint8 d;
            v_uint16 dx;

            v_int16 sr = v_reinterpret_as_s16(r);
            v_int16 sg = v_reinterpret_as_s16(g);
            v_int16 sb = v_reinterpret_as_s16(b);

            v_int16 bg0, bg1;
            v_int16 rd0, rd1;
            v_zip(sb, sg, bg0, bg1);
            v_zip(sr, delta, rd0, rd1);

            v_uint32 d0, d1;
            d0 = v_reinterpret_as_u32(v_add(v_dotprod(bg0, bg2y), v_dotprod(rd0, r12y)));
            d1 = v_reinterpret_as_u32(v_add(v_dotprod(bg1, bg2y), v_dotprod(rd1, r12y)));

            d0 = v_shr<shift>(d0);
            d1 = v_shr<shift>(d1);

            dx = v_pack(d0, d1);
            // high part isn't used
            d = v_pack(dx, dx);

            v_store_low(dst, d);
        }
        vx_cleanup();
#endif
        for( ; i < n; i++, src += sizeof(ushort), dst++)
        {
            int t = ((ushort*)src)[0];
            uchar r, g, b;
            b = (t << 3) & 0xf8;
            if( gb == 6 )
            {
                g = (t >> 3) & 0xfc;
                r = (t >> 8) & 0xf8;
            }
            else
            {
                g = (t >> 2) & 0xf8;
                r = (t >> 7) & 0xf8;
            }
            dst[0] = (uchar)CV_DESCALE(b*BY + g*GY + r*RY, shift);
        }
    }
    int greenBits;
};


template<typename _Tp> struct RGB2Gray
{
    typedef _Tp channel_type;

    RGB2Gray(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        static const float coeffs0[] = { R2YF, G2YF, B2YF };
        memcpy( coeffs, _coeffs ? _coeffs : coeffs0, 3*sizeof(coeffs[0]) );
        if(blueIdx == 0)
            std::swap(coeffs[0], coeffs[2]);
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn;
        float cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];
        for(int i = 0; i < n; i++, src += scn)
            dst[i] = saturate_cast<_Tp>(src[0]*cb + src[1]*cg + src[2]*cr);
    }
    int srccn;
    float coeffs[3];
};


template <>
struct RGB2Gray<float>
{
    typedef float channel_type;

    RGB2Gray(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        static const float coeffs0[] = { R2YF, G2YF, B2YF };
        for(int i = 0; i < 3; i++)
        {
            coeffs[i] = _coeffs ? _coeffs[i] : coeffs0[i];
        }
        if(blueIdx == 0)
            std::swap(coeffs[0], coeffs[2]);
    }

    void operator()(const float * src, float * dst, int n) const
    {
        int scn = srccn, i = 0;
        float cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];

#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<v_float32>::vlanes();
        v_float32 rv = vx_setall_f32(cr), gv = vx_setall_f32(cg), bv = vx_setall_f32(cb);
        for(; i <= n-vsize;
            i += vsize, src += vsize*scn, dst += vsize)
        {
            v_float32 r, g, b, a;
            if(scn == 3)
            {
                v_load_deinterleave(src, b, g, r);
            }
            else
            {
                v_load_deinterleave(src, b, g, r, a);
            }

            v_float32 d = v_fma(r, rv, v_fma(g, gv, v_mul(b, bv)));

            v_store(dst, d);
        }
        vx_cleanup();
#endif

        for ( ; i < n; i++, src += scn, dst++)
            dst[0] = src[0]*cb + src[1]*cg + src[2]*cr;
    }

    int srccn;
    float coeffs[3];
};

template<>
struct RGB2Gray<uchar>
{
    typedef uchar channel_type;

    static const int BY = BY15;
    static const int GY = GY15;
    static const int RY = RY15;
    static const int shift = gray_shift;

    RGB2Gray(int _srccn, int blueIdx, const int* _coeffs) : srccn(_srccn)
    {
        const int coeffs0[] = { RY, GY, BY };
        for(int i = 0; i < 3; i++)
                coeffs[i] = (short)(_coeffs ? _coeffs[i] : coeffs0[i]);
        if(blueIdx == 0)
            std::swap(coeffs[0], coeffs[2]);

        CV_Assert(coeffs[0] + coeffs[1] + coeffs[2] == (1 << shift));
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int scn = srccn;
        short cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];
        int i = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<v_uint8>::vlanes();
        v_int16 bg2y;
        v_int16 r12y;
        v_int16 dummy;
        v_zip(vx_setall_s16(cb), vx_setall_s16(cg), bg2y, dummy);
        v_zip(vx_setall_s16(cr), vx_setall_s16( 1), r12y, dummy);
        v_int16 delta = vx_setall_s16(1 << (shift-1));

        for( ; i <= n-vsize;
             i += vsize, src += scn*vsize, dst += vsize)
        {
            v_uint8 r, g, b, a;
            if(scn == 3)
            {
                v_load_deinterleave(src, b, g, r);
            }
            else
            {
                v_load_deinterleave(src, b, g, r, a);
            }

            //TODO: shorten registers use when v_deinterleave is available

            v_uint16 r0, r1, g0, g1, b0, b1;
            v_expand(r, r0, r1);
            v_expand(g, g0, g1);
            v_expand(b, b0, b1);

            v_int16 bg00, bg01, bg10, bg11;
            v_int16 rd00, rd01, rd10, rd11;
            v_zip(v_reinterpret_as_s16(b0), v_reinterpret_as_s16(g0), bg00, bg01);
            v_zip(v_reinterpret_as_s16(b1), v_reinterpret_as_s16(g1), bg10, bg11);
            v_zip(v_reinterpret_as_s16(r0), delta, rd00, rd01);
            v_zip(v_reinterpret_as_s16(r1), delta, rd10, rd11);

            v_uint32 y00, y01, y10, y11;
            y00 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_dotprod(bg00, bg2y), v_dotprod(rd00, r12y))));
            y01 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_dotprod(bg01, bg2y), v_dotprod(rd01, r12y))));
            y10 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_dotprod(bg10, bg2y), v_dotprod(rd10, r12y))));
            y11 = v_shr<shift>(v_reinterpret_as_u32(v_add(v_dotprod(bg11, bg2y), v_dotprod(rd11, r12y))));

            v_uint16 y0, y1;
            y0 = v_pack(y00, y01);
            y1 = v_pack(y10, y11);

            v_uint8 y = v_pack(y0, y1);
            v_store(dst, y);
        }
        vx_cleanup();
#endif

        for( ; i < n; i++, src += scn, dst++)
        {
            int b = src[0], g = src[1], r = src[2];
            uchar y = (uchar)CV_DESCALE(b*cb + g*cg + r*cr, shift);
            dst[0] = y;
        }
    }

    int srccn;
    short coeffs[3];
};


template<>
struct RGB2Gray<ushort>
{
    typedef ushort channel_type;

    static const int BY = BY15;
    static const int GY = GY15;
    static const int RY = RY15;
    static const int shift = gray_shift;
    static const int fix_shift = (int)(sizeof(short)*8 - shift);

    RGB2Gray(int _srccn, int blueIdx, const int* _coeffs) : srccn(_srccn)
    {
        const int coeffs0[] = { RY, GY, BY };
        for(int i = 0; i < 3; i++)
                coeffs[i] = (short)(_coeffs ? _coeffs[i] : coeffs0[i]);
        if(blueIdx == 0)
            std::swap(coeffs[0], coeffs[2]);

        CV_Assert(coeffs[0] + coeffs[1] + coeffs[2] == (1 << shift));
    }

    void operator()(const ushort* src, ushort* dst, int n) const
    {
        int scn = srccn;
        short cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];
        int i = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<v_uint16>::vlanes();

        v_int16 b2y = vx_setall_s16(cb);
        v_int16 g2y = vx_setall_s16(cg);
        v_int16 r2y = vx_setall_s16(cr);
        v_int16 one = vx_setall_s16(1);
        v_int16 z = vx_setzero_s16();

        v_int16 bg2y, r12y;
        v_int16 dummy;
        v_zip(b2y, g2y, bg2y, dummy);
        v_zip(r2y, one, r12y, dummy);

        v_int16 delta = vx_setall_s16(1 << (shift-1));

        for( ; i <= n-vsize;
             i += vsize, src += scn*vsize, dst += vsize)
        {
            v_uint16 r, g, b, a;
            if(scn == 3)
            {
                v_load_deinterleave(src, b, g, r);
            }
            else
            {
                v_load_deinterleave(src, b, g, r, a);
            }

            v_int16 sb = v_reinterpret_as_s16(b);
            v_int16 sr = v_reinterpret_as_s16(r);
            v_int16 sg = v_reinterpret_as_s16(g);

            v_int16 bg0, bg1;
            v_int16 rd0, rd1;
            v_zip(sb, sg, bg0, bg1);
            v_zip(sr, delta, rd0, rd1);

            // fixing 16bit signed multiplication
            v_int16 mr, mg, mb;
            mr = v_and(v_lt(sr, z), r2y);
            mg = v_and(v_lt(sg, z), g2y);
            mb = v_and(v_lt(sb, z), b2y);
            v_int16 fixmul = v_shl<fix_shift>(v_add_wrap(mr, v_add_wrap(mg, mb)));

            v_int32 sy0 = v_shr<shift>(v_add(v_dotprod(bg0, bg2y), v_dotprod(rd0, r12y)));
            v_int32 sy1 = v_shr<shift>(v_add(v_dotprod(bg1, bg2y), v_dotprod(rd1, r12y)));

            v_int16 y = v_add_wrap(v_pack(sy0, sy1), fixmul);

            v_store((short*)dst, y);
        }
        vx_cleanup();
#endif
        for( ; i < n; i++, src += scn, dst++)
        {
            int b = src[0], g = src[1], r = src[2];
            ushort d = (ushort)CV_DESCALE((unsigned)(b*cb + g*cg + r*cr), shift);
            dst[0] = d;
        }
    }

    int srccn;
    short coeffs[3];
};


/////////////////////////// RGBA <-> mRGBA (alpha premultiplied) //////////////

template<typename _Tp>
struct RGBA2mRGBA
{
    typedef _Tp channel_type;

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        _Tp max_val  = ColorChannel<_Tp>::max();
        _Tp half_val = ColorChannel<_Tp>::half();
        for( int i = 0; i < n; i++ )
        {
            _Tp v0 = *src++;
            _Tp v1 = *src++;
            _Tp v2 = *src++;
            _Tp v3 = *src++;

            *dst++ = (v0 * v3 + half_val) / max_val;
            *dst++ = (v1 * v3 + half_val) / max_val;
            *dst++ = (v2 * v3 + half_val) / max_val;
            *dst++ = v3;
        }
    }
};


template<>
struct RGBA2mRGBA<uchar>
{
    typedef uchar channel_type;

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        const uchar max_val  = 255;
        const uchar half_val = 128;

        int i = 0;
#if CV_SIMD
        const int vsize = VTraits<v_uint8>::vlanes();
        v_uint8 amask = v_reinterpret_as_u8(vx_setall_u32(0xFF000000));
        v_uint16 vh = vx_setall_u16(half_val+1);

        // processing 4 registers per loop cycle is about 10% faster
        // than processing 1 register
        for( ; i <= n-vsize;
             i += vsize, src += 4*vsize, dst += 4*vsize)
        {
            v_uint8 v[4];
            for(int j = 0; j < 4; j++)
                v[j] = vx_load(src + j*vsize);

            // r0,g0,b0,a0,r1,g1,b1,a1 => 00,00,00,a0,00,00,00,a1 =>
            // => 00,00,a0,a0,00,00,a1,a1
            // => a0,a0,a0,a0,a1,a1,a1,a1

            v_uint16 a16[4];
            for(int j = 0; j < 4; j++)
                a16[j] = v_reinterpret_as_u16(v_and(v[j], amask));

            v_uint32 a32[4];
            for(int j = 0; j < 4; j++)
                a32[j] = v_reinterpret_as_u32(v_or(a16[j], (v_shr(a16[j], 8))));

            v_uint8 a[4];
            for(int j = 0; j < 4; j++)
                a[j] = v_reinterpret_as_u8(v_or(a32[j], (v_shr(a32[j], 16))));

            v_uint16 m[8];
            for(int j = 0; j < 4; j++)
                v_mul_expand(v[j], a[j], m[j], m[j+4]);

            for(int j = 0; j < 8; j++)
                m[j] = v_add(m[j], vh);

            // div 255: (v+1+(v>>8))>8
            // +1 is in vh, has no effect on (v>>8)
            for(int j = 0; j < 8; j++)
                m[j] = v_shr((v_add(m[j], (v_shr(m[j], 8)))), 8);

            v_uint8 d[4];
            for(int j = 0; j < 4; j++)
                d[j] = v_pack(m[j], m[j+4]);

            for(int j = 0; j < 4; j++)
                d[j] = v_select(amask, a[j], d[j]);

            for(int j = 0; j < 4; j++)
                v_store(dst + j*vsize, d[j]);
        }

        vx_cleanup();
#endif
        for(; i < n; i++, src += 4, dst += 4 )
        {
            uchar v0 = src[0];
            uchar v1 = src[1];
            uchar v2 = src[2];
            uchar v3 = src[3];

            dst[0] = (v0 * v3 + half_val) / max_val;
            dst[1] = (v1 * v3 + half_val) / max_val;
            dst[2] = (v2 * v3 + half_val) / max_val;
            dst[3] = v3;
        }
    }
};


template<typename _Tp>
struct mRGBA2RGBA
{
    typedef _Tp channel_type;

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        _Tp max_val = ColorChannel<_Tp>::max();
        for( int i = 0; i < n; i++ )
        {
            _Tp v0 = *src++;
            _Tp v1 = *src++;
            _Tp v2 = *src++;
            _Tp v3 = *src++;
            _Tp v3_half = v3 / 2;

            *dst++ = (v3==0)? 0 : saturate_cast<_Tp>((v0 * max_val + v3_half) / v3);
            *dst++ = (v3==0)? 0 : saturate_cast<_Tp>((v1 * max_val + v3_half) / v3);
            *dst++ = (v3==0)? 0 : saturate_cast<_Tp>((v2 * max_val + v3_half) / v3);
            *dst++ = v3;
        }
    }
};


template<>
struct mRGBA2RGBA<uchar>
{
    typedef uchar channel_type;

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        uchar max_val = ColorChannel<uchar>::max();
        int i = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int vsize = VTraits<v_uint8>::vlanes();
        v_uint8 amask = v_reinterpret_as_u8(vx_setall_u32(0xFF000000));
        v_uint8 vmax = vx_setall_u8(max_val);

        for( ; i <= n-vsize/4;
             i += vsize/4, src += vsize, dst += vsize)
        {
            v_uint8 s = vx_load(src + 0*vsize);

            // r0,g0,b0,a0,r1,g1,b1,a1 => 00,00,00,a0,00,00,00,a1 =>
            // => 00,00,a0,a0,00,00,a1,a1
            // => a0,a0,a0,a0,a1,a1,a1,a1
            v_uint8 a;
            v_uint16 a16;
            v_uint32 a32;
            a16 = v_reinterpret_as_u16(v_and(s, amask));
            a32 = v_reinterpret_as_u32(v_or(a16, v_shr<8>(a16)));
            a = v_reinterpret_as_u8(v_or(a32, v_shr<16>(a32)));

            // s *= max_val
            v_uint16 s0, s1;
            v_mul_expand(s, vmax, s0, s1);

            // s += a/2
            v_uint16 ae0, ae1;
            v_expand(a, ae0, ae1);
            s0 = v_add(s0, v_shr<1>(ae0)); s1 = v_add(s1, v_shr<1>(ae1));

            // s, a -> u32 -> float
            v_uint32 u00, u01, u10, u11;
            v_int32 s00, s01, s10, s11;
            v_expand(s0, u00, u01);
            v_expand(s1, u10, u11);
            s00 = v_reinterpret_as_s32(u00);
            s01 = v_reinterpret_as_s32(u01);
            s10 = v_reinterpret_as_s32(u10);
            s11 = v_reinterpret_as_s32(u11);

            v_uint32 ua00, ua01, ua10, ua11;
            v_int32 a00, a01, a10, a11;
            v_expand(ae0, ua00, ua01);
            v_expand(ae1, ua10, ua11);
            a00 = v_reinterpret_as_s32(ua00);
            a01 = v_reinterpret_as_s32(ua01);
            a10 = v_reinterpret_as_s32(ua10);
            a11 = v_reinterpret_as_s32(ua11);

            v_float32 fs00, fs01, fs10, fs11;
            fs00 = v_cvt_f32(s00);
            fs01 = v_cvt_f32(s01);
            fs10 = v_cvt_f32(s10);
            fs11 = v_cvt_f32(s11);

            v_float32 fa00, fa01, fa10, fa11;
            fa00 = v_cvt_f32(a00);
            fa01 = v_cvt_f32(a01);
            fa10 = v_cvt_f32(a10);
            fa11 = v_cvt_f32(a11);

            // float d = (float)s/(float)a
            v_float32 fd00, fd01, fd10, fd11;
            fd00 = v_div(fs00, fa00);
            fd01 = v_div(fs01, fa01);
            fd10 = v_div(fs10, fa10);
            fd11 = v_div(fs11, fa11);

            // d -> u32 -> u8
            v_uint32 ud00, ud01, ud10, ud11;
            ud00 = v_reinterpret_as_u32(v_trunc(fd00));
            ud01 = v_reinterpret_as_u32(v_trunc(fd01));
            ud10 = v_reinterpret_as_u32(v_trunc(fd10));
            ud11 = v_reinterpret_as_u32(v_trunc(fd11));
            v_uint16 ud0, ud1;
            ud0 = v_pack(ud00, ud01);
            ud1 = v_pack(ud10, ud11);
            v_uint8 d;
            d = v_pack(ud0, ud1);

            // if a == 0 then d = 0
            v_uint8 am;
            am = v_ne(a, vx_setzero_u8());
            d = v_and(d, am);

            // put alpha values
            d = v_select(amask, a, d);

            v_store(dst, d);
        }

        vx_cleanup();
#endif
        for(; i < n; i++, src += 4, dst += 4 )
        {
            uchar v0 = src[0];
            uchar v1 = src[1];
            uchar v2 = src[2];
            uchar v3 = src[3];

            uchar v3_half = v3 / 2;

            dst[0] = (v3==0)? 0 : saturate_cast<uchar>((v0 * max_val + v3_half) / v3);
            dst[1] = (v3==0)? 0 : saturate_cast<uchar>((v1 * max_val + v3_half) / v3);
            dst[2] = (v3==0)? 0 : saturate_cast<uchar>((v2 * max_val + v3_half) / v3);
            dst[3] = v3;
        }
    }
};
} // namespace anon

// 8u, 16u, 32f
void cvtBGRtoBGR(const uchar * src_data, size_t src_step,
                 uchar * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, int dcn, bool swapBlue)
{
    CV_INSTRUMENT_REGION();

    int blueIdx = swapBlue ? 2 : 0;
    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2RGB<uchar>(scn, dcn, blueIdx));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2RGB<ushort>(scn, dcn, blueIdx));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2RGB<float>(scn, dcn, blueIdx));
}

// only 8u
void cvtBGRtoBGR5x5(const uchar * src_data, size_t src_step,
                    uchar * dst_data, size_t dst_step,
                    int width, int height,
                    int scn, bool swapBlue, int greenBits)
{
    CV_INSTRUMENT_REGION();

    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2RGB5x5(scn, swapBlue ? 2 : 0, greenBits));
}

// only 8u
void cvtBGR5x5toBGR(const uchar * src_data, size_t src_step,
                    uchar * dst_data, size_t dst_step,
                    int width, int height,
                    int dcn, bool swapBlue, int greenBits)
{
    CV_INSTRUMENT_REGION();

    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB5x52RGB(dcn, swapBlue ? 2 : 0, greenBits));
}

// 8u, 16u, 32f
void cvtBGRtoGray(const uchar * src_data, size_t src_step,
                  uchar * dst_data, size_t dst_step,
                  int width, int height,
                  int depth, int scn, bool swapBlue)
{
    CV_INSTRUMENT_REGION();

    int blueIdx = swapBlue ? 2 : 0;
    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Gray<uchar>(scn, blueIdx, 0));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Gray<ushort>(scn, blueIdx, 0));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Gray<float>(scn, blueIdx, 0));
}

// 8u, 16u, 32f
void cvtGraytoBGR(const uchar * src_data, size_t src_step,
                  uchar * dst_data, size_t dst_step,
                  int width, int height,
                  int depth, int dcn)
{
    CV_INSTRUMENT_REGION();

    if( depth == CV_8U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Gray2RGB<uchar>(dcn));
    else if( depth == CV_16U )
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Gray2RGB<ushort>(dcn));
    else
        CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Gray2RGB<float>(dcn));
}

// only 8u
void cvtBGR5x5toGray(const uchar * src_data, size_t src_step,
                     uchar * dst_data, size_t dst_step,
                     int width, int height,
                     int greenBits)
{
    CV_INSTRUMENT_REGION();

    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB5x52Gray(greenBits));
}

// only 8u
void cvtGraytoBGR5x5(const uchar * src_data, size_t src_step,
                     uchar * dst_data, size_t dst_step,
                     int width, int height,
                     int greenBits)
{
    CV_INSTRUMENT_REGION();

    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, Gray2RGB5x5(greenBits));
}

void cvtRGBAtoMultipliedRGBA(const uchar * src_data, size_t src_step,
                             uchar * dst_data, size_t dst_step,
                             int width, int height)
{
    CV_INSTRUMENT_REGION();

    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGBA2mRGBA<uchar>());
}

void cvtMultipliedRGBAtoRGBA(const uchar * src_data, size_t src_step,
                             uchar * dst_data, size_t dst_step,
                             int width, int height)
{
    CV_INSTRUMENT_REGION();

    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, mRGBA2RGBA<uchar>());
}

#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace
