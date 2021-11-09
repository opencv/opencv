// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#if !defined(GAPI_STANDALONE)

#include "precomp.hpp"

#include <opencv2/gapi/own/assert.hpp>
#include <opencv2/core/traits.hpp>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/hal/intrin.hpp>

#if CV_SIMD
#include "gfluidcore_func.hpp"
#endif

#include <opencv2/gapi/core.hpp>

#include <opencv2/gapi/fluid/gfluidbuffer.hpp>
#include <opencv2/gapi/fluid/gfluidkernel.hpp>
#include <opencv2/gapi/fluid/core.hpp>

#if CV_SSE4_1
#include "gfluidcore_simd_sse41.hpp"
#endif

#include "gfluidbuffer_priv.hpp"
#include "gfluidbackend.hpp"
#include "gfluidutils.hpp"

#include <math.h>

#include <cassert>
#include <cstdlib>

namespace cv {
namespace gapi {
namespace fluid {

//---------------------
//
// Arithmetic functions
//
//---------------------

template<typename DST, typename SRC1, typename SRC2>
static inline DST absdiff(SRC1 x, SRC2 y)
{
    auto result = x > y? x - y: y - x;
    return saturate<DST>(result, roundf);
}

template<typename DST, typename SRC1, typename SRC2>
static inline DST addWeighted(SRC1 src1, SRC2 src2, float alpha, float beta, float gamma)
{
    float dst = src1*alpha + src2*beta + gamma;
    return saturate<DST>(dst, roundf);
}

template<typename DST, typename SRC1, typename SRC2>
static inline DST add(SRC1 x, SRC2 y)
{
    return saturate<DST>(x + y, roundf);
}

template<typename DST, typename SRC1, typename SRC2>
static inline DST sub(SRC1 x, SRC2 y)
{
    return saturate<DST>(x - y, roundf);
}

template<typename DST, typename SRC1, typename SRC2>
static inline DST subr(SRC1 x, SRC2 y)
{
    return saturate<DST>(y - x, roundf); // reverse: y - x
}

template<typename DST, typename SRC1, typename SRC2>
static inline DST mul(SRC1 x, SRC2 y, float scale=1)
{
    auto result = scale * x * y;
    return saturate<DST>(result, rintf);
}

template<typename DST, typename SRC1, typename SRC2>
static inline
typename std::enable_if<!std::is_same<DST, float>::value, DST>::type
div(SRC1 x, SRC2 y, float scale=1)
{
    // like OpenCV: returns 0, if DST type=uchar/short/ushort and divider(y)=0
    auto result = y? scale * x / y: 0;
    return saturate<DST>(result, rintf);
}

template<typename DST, typename SRC1, typename SRC2>
static inline
typename std::enable_if<std::is_same<DST, float>::value, DST>::type
div(SRC1 x, SRC2 y, float scale = 1)
{
    // like OpenCV: returns inf/nan, if DST type=float and divider(y)=0
    auto result = scale * x / y;
    return saturate<DST>(result, rintf);
}

template<typename DST, typename SRC1, typename SRC2>
static inline DST divr(SRC1 x, SRC2 y, float scale=1)
{
    auto result = x? scale * y / x: 0; // reverse: y / x
    return saturate<DST>(result, rintf);
}

//---------------------------
//
// Fluid kernels: addWeighted
//
//---------------------------
#if CV_SIMD
CV_ALWAYS_INLINE v_float32 v_load_f32(const ushort* in)
{
    return v_cvt_f32(v_reinterpret_as_s32(vx_load_expand(in)));
}

CV_ALWAYS_INLINE v_float32 v_load_f32(const short* in)
{
    return v_cvt_f32(vx_load_expand(in));
}

CV_ALWAYS_INLINE v_float32 v_load_f32(const uchar* in)
{
    return v_cvt_f32(v_reinterpret_as_s32(vx_load_expand_q(in)));
}
#endif

#if CV_SSE2
CV_ALWAYS_INLINE void addw_short_store(short* out, const v_int32& c1, const v_int32& c2)
{
    vx_store(out, v_pack(c1, c2));
}

CV_ALWAYS_INLINE void addw_short_store(ushort* out, const v_int32& c1, const v_int32& c2)
{
    vx_store(out, v_pack_u(c1, c2));
}

template<typename SRC, typename DST>
CV_ALWAYS_INLINE int addw_simd(const SRC in1[], const SRC in2[], DST out[],
                               const float _alpha, const float _beta,
                               const float _gamma, int length)
{
    static_assert(((std::is_same<SRC, ushort>::value) && (std::is_same<DST, ushort>::value)) ||
                  ((std::is_same<SRC, short>::value) && (std::is_same<DST, short>::value)),
                  "This templated overload is only for short and ushort type combinations.");

    constexpr int nlanes = (std::is_same<DST, ushort>::value) ? static_cast<int>(v_uint16::nlanes) :
                                                                static_cast<int>(v_int16::nlanes);

    if (length < nlanes)
        return 0;

    v_float32 alpha = vx_setall_f32(_alpha);
    v_float32 beta = vx_setall_f32(_beta);
    v_float32 gamma = vx_setall_f32(_gamma);

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            v_float32 a1 = v_load_f32(&in1[x]);
            v_float32 a2 = v_load_f32(&in1[x + nlanes / 2]);
            v_float32 b1 = v_load_f32(&in2[x]);
            v_float32 b2 = v_load_f32(&in2[x + nlanes / 2]);

            addw_short_store(&out[x], v_round(v_fma(a1, alpha, v_fma(b1, beta, gamma))),
                                      v_round(v_fma(a2, alpha, v_fma(b2, beta, gamma))));
        }

        if (x < length)
        {
            x = length - nlanes;
            continue;  // process one more time (unaligned tail)
        }
        break;
    }
    return x;
}

template<typename SRC>
CV_ALWAYS_INLINE int addw_simd(const SRC in1[], const SRC in2[], uchar out[],
                               const float _alpha, const float _beta,
                               const float _gamma, int length)
{
    constexpr int nlanes = v_uint8::nlanes;

    if (length < nlanes)
        return 0;

    v_float32 alpha = vx_setall_f32(_alpha);
    v_float32 beta = vx_setall_f32(_beta);
    v_float32 gamma = vx_setall_f32(_gamma);

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            v_float32 a1 = v_load_f32(&in1[x]);
            v_float32 a2 = v_load_f32(&in1[x + nlanes / 4]);
            v_float32 a3 = v_load_f32(&in1[x + nlanes / 2]);
            v_float32 a4 = v_load_f32(&in1[x + 3 * nlanes / 4]);
            v_float32 b1 = v_load_f32(&in2[x]);
            v_float32 b2 = v_load_f32(&in2[x + nlanes / 4]);
            v_float32 b3 = v_load_f32(&in2[x + nlanes / 2]);
            v_float32 b4 = v_load_f32(&in2[x + 3 * nlanes / 4]);

            v_int32 sum1 = v_round(v_fma(a1, alpha, v_fma(b1, beta, gamma))),
                    sum2 = v_round(v_fma(a2, alpha, v_fma(b2, beta, gamma))),
                    sum3 = v_round(v_fma(a3, alpha, v_fma(b3, beta, gamma))),
                    sum4 = v_round(v_fma(a4, alpha, v_fma(b4, beta, gamma)));

            vx_store(&out[x], v_pack_u(v_pack(sum1, sum2), v_pack(sum3, sum4)));
        }

        if (x < length)
        {
            x = length - nlanes;
            continue;  // process one more time (unaligned tail)
        }
        break;
    }
    return x;
}

template<typename SRC>
CV_ALWAYS_INLINE int addw_simd(const SRC*, const SRC*, float*,
                               const float, const float,
                               const float, int)
{
    //Cases when dst type is float are successfully vectorized with compiler.
    return 0;
}
#endif  // CV_SSE2

template<typename DST, typename SRC1, typename SRC2>
static void run_addweighted(Buffer &dst, const View &src1, const View &src2,
                            double alpha, double beta, double gamma)
{
    static_assert(std::is_same<SRC1, SRC2>::value, "wrong types");

    const auto *in1 = src1.InLine<SRC1>(0);
    const auto *in2 = src2.InLine<SRC2>(0);
          auto *out = dst.OutLine<DST>();

    int width  = dst.length();
    int chan   = dst.meta().chan;
    int length = width * chan;

    // NB: assume in/out types are not 64-bits
    auto _alpha = static_cast<float>( alpha );
    auto _beta  = static_cast<float>( beta  );
    auto _gamma = static_cast<float>( gamma );

    int x = 0;
#if CV_SSE2
    x = addw_simd(in1, in2, out, _alpha, _beta, _gamma, length);
#endif

    for (; x < length; ++x)
        out[x] = addWeighted<DST>(in1[x], in2[x], _alpha, _beta, _gamma);
}

GAPI_FLUID_KERNEL(GFluidAddW, cv::gapi::core::GAddW, false)
{
    static const int Window = 1;

    static void run(const View &src1, double alpha, const View &src2,
                                      double beta, double gamma, int /*dtype*/,
                        Buffer &dst)
    {
        //      DST     SRC1    SRC2    OP               __VA_ARGS__
        BINARY_(uchar , uchar , uchar , run_addweighted, dst, src1, src2, alpha, beta, gamma);
        BINARY_(uchar , ushort, ushort, run_addweighted, dst, src1, src2, alpha, beta, gamma);
        BINARY_(uchar ,  short,  short, run_addweighted, dst, src1, src2, alpha, beta, gamma);
        BINARY_( short,  short,  short, run_addweighted, dst, src1, src2, alpha, beta, gamma);
        BINARY_(ushort, ushort, ushort, run_addweighted, dst, src1, src2, alpha, beta, gamma);
        BINARY_( float, uchar , uchar , run_addweighted, dst, src1, src2, alpha, beta, gamma);
        BINARY_( float, ushort, ushort, run_addweighted, dst, src1, src2, alpha, beta, gamma);
        BINARY_( float,  short,  short, run_addweighted, dst, src1, src2, alpha, beta, gamma);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

//--------------------------
//
// Fluid kernels: +, -, *, /
//
//--------------------------

enum Arithm { ARITHM_ABSDIFF, ARITHM_ADD, ARITHM_SUBTRACT, ARITHM_MULTIPLY, ARITHM_DIVIDE };

#if CV_SIMD
CV_ALWAYS_INLINE void absdiff_store(short out[], const v_int16& a, const v_int16& b, int x)
{
    vx_store(&out[x], v_absdiffs(a, b));
}

CV_ALWAYS_INLINE void absdiff_store(ushort out[], const v_uint16& a, const v_uint16& b, int x)
{
    vx_store(&out[x], v_absdiff(a, b));
}

CV_ALWAYS_INLINE void absdiff_store(uchar out[], const v_uint8& a, const v_uint8& b, int x)
{
    vx_store(&out[x], v_absdiff(a, b));
}

CV_ALWAYS_INLINE void absdiff_store(float out[], const v_float32& a, const v_float32& b, int x)
{
    vx_store(&out[x], v_absdiff(a, b));
}

template<typename T, typename VT>
CV_ALWAYS_INLINE int absdiff_impl(const T in1[], const T in2[], T out[], int length)
{
    constexpr int nlanes = static_cast<int>(VT::nlanes);

    if (length < nlanes)
        return 0;

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            VT a = vx_load(&in1[x]);
            VT b = vx_load(&in2[x]);
            absdiff_store(out, a, b, x);
        }

        if (x < length && (in1 != out) && (in2 != out))
        {
            x = length - nlanes;
            continue;  // process one more time (unaligned tail)
        }
        break;
    }

    return x;
}

template<typename T>
CV_ALWAYS_INLINE int absdiff_simd(const T in1[], const T in2[], T out[], int length)
{
    if (std::is_same<T, uchar>::value)
    {
        return absdiff_impl<uchar, v_uint8>(reinterpret_cast<const uchar*>(in1),
                                            reinterpret_cast<const uchar*>(in2),
                                            reinterpret_cast<uchar*>(out), length);
    }
    else if (std::is_same<T, ushort>::value)
    {
        return absdiff_impl<ushort, v_uint16>(reinterpret_cast<const ushort*>(in1),
                                              reinterpret_cast<const ushort*>(in2),
                                              reinterpret_cast<ushort*>(out), length);
    }
    else if (std::is_same<T, short>::value)
    {
        return absdiff_impl<short, v_int16>(reinterpret_cast<const short*>(in1),
                                            reinterpret_cast<const short*>(in2),
                                            reinterpret_cast<short*>(out), length);
    }
    else if (std::is_same<T, float>::value)
    {
        return absdiff_impl<float, v_float32>(reinterpret_cast<const float*>(in1),
                                              reinterpret_cast<const float*>(in2),
                                              reinterpret_cast<float*>(out), length);
    }

    return 0;
}

template<typename T, typename VT>
CV_ALWAYS_INLINE int add_simd_sametype(const T in1[], const T in2[], T out[], int length)
{
    constexpr int nlanes = static_cast<int>(VT::nlanes);

    if (length < nlanes)
        return 0;

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            VT a = vx_load(&in1[x]);
            VT b = vx_load(&in2[x]);
            vx_store(&out[x], a + b);
        }

        if (x < length && (in1 != out) && (in2 != out))
        {
            x = length - nlanes;
            continue;  // process one more time (unaligned tail)
        }
        break;
    }

    return x;
}

template<typename SRC, typename DST>
CV_ALWAYS_INLINE int add_simd(const SRC in1[], const SRC in2[], DST out[], int length)
{
    if (std::is_same<DST, float>::value && !std::is_same<SRC, float>::value)
        return 0;

    if (std::is_same<DST, SRC>::value)
    {
        if (std::is_same<DST, uchar>::value)
        {
            return add_simd_sametype<uchar, v_uint8>(reinterpret_cast<const uchar*>(in1),
                                                     reinterpret_cast<const uchar*>(in2),
                                                     reinterpret_cast<uchar*>(out), length);
        }
        else if (std::is_same<DST, short>::value)
        {
            return add_simd_sametype<short, v_int16>(reinterpret_cast<const short*>(in1),
                                                     reinterpret_cast<const short*>(in2),
                                                     reinterpret_cast<short*>(out), length);
        }
        else if (std::is_same<DST, float>::value)
        {
            return add_simd_sametype<float, v_float32>(reinterpret_cast<const float*>(in1),
                                                       reinterpret_cast<const float*>(in2),
                                                       reinterpret_cast<float*>(out), length);
        }
    }
    else if (std::is_same<SRC, short>::value && std::is_same<DST, uchar>::value)
    {
        constexpr int nlanes = static_cast<int>(v_uint8::nlanes);

        if (length < nlanes)
            return 0;

        int x = 0;
        for (;;)
        {
            for (; x <= length - nlanes; x += nlanes)
            {
                v_int16 a1 = vx_load(reinterpret_cast<const short*>(&in1[x]));
                v_int16 a2 = vx_load(reinterpret_cast<const short*>(&in1[x + nlanes / 2]));
                v_int16 b1 = vx_load(reinterpret_cast<const short*>(&in2[x]));
                v_int16 b2 = vx_load(reinterpret_cast<const short*>(&in2[x + nlanes / 2]));

                vx_store(reinterpret_cast<uchar*>(&out[x]), v_pack_u(a1 + b1, a2 + b2));
            }

            if (x < length)
            {
                CV_DbgAssert((reinterpret_cast<const short*>(in1) != reinterpret_cast<const short*>(out)) &&
                             (reinterpret_cast<const short*>(in2) != reinterpret_cast<const short*>(out)));
                x = length - nlanes;
                continue;  // process one more time (unaligned tail)
            }
            break;
        }

        return x;
    }
    else if (std::is_same<SRC, float>::value && std::is_same<DST, uchar>::value)
    {
        constexpr int nlanes = static_cast<int>(v_uint8::nlanes);

        if (length < nlanes)
            return 0;

        int x = 0;
        for (;;)
        {
            for (; x <= length - nlanes; x += nlanes)
            {
                v_float32 a1 = vx_load(reinterpret_cast<const float*>(&in1[x]));
                v_float32 a2 = vx_load(reinterpret_cast<const float*>(&in1[x + nlanes / 4]));
                v_float32 a3 = vx_load(reinterpret_cast<const float*>(&in1[x + 2 * nlanes / 4]));
                v_float32 a4 = vx_load(reinterpret_cast<const float*>(&in1[x + 3 * nlanes / 4]));

                v_float32 b1 = vx_load(reinterpret_cast<const float*>(&in2[x]));
                v_float32 b2 = vx_load(reinterpret_cast<const float*>(&in2[x + nlanes / 4]));
                v_float32 b3 = vx_load(reinterpret_cast<const float*>(&in2[x + 2 * nlanes / 4]));
                v_float32 b4 = vx_load(reinterpret_cast<const float*>(&in2[x + 3 * nlanes / 4]));

                vx_store(reinterpret_cast<uchar*>(&out[x]), v_pack_u(v_pack(v_round(a1 + b1), v_round(a2 + b2)),
                                                                     v_pack(v_round(a3 + b3), v_round(a4 + b4))));
            }

            if (x < length)
            {
                CV_DbgAssert((reinterpret_cast<const float*>(in1) != reinterpret_cast<const float*>(out)) &&
                             (reinterpret_cast<const float*>(in2) != reinterpret_cast<const float*>(out)));
                x = length - nlanes;
                continue;  // process one more time (unaligned tail)
            }
            break;
        }

        return x;
    }

    return 0;
}

template<typename T, typename VT>
CV_ALWAYS_INLINE int sub_simd_sametype(const T in1[], const T in2[], T out[], int length)
{
    constexpr int nlanes = static_cast<int>(VT::nlanes);

    if (length < nlanes)
        return 0;

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            VT a = vx_load(&in1[x]);
            VT b = vx_load(&in2[x]);
            vx_store(&out[x], a - b);
        }

        if (x < length && (in1 != out) && (in2 != out))
        {
            x = length - nlanes;
            continue;  // process one more time (unaligned tail)
        }
        break;
    }

    return x;
}

template<typename SRC, typename DST>
CV_ALWAYS_INLINE int sub_simd(const SRC in1[], const SRC in2[], DST out[], int length)
{
    if (std::is_same<DST, float>::value && !std::is_same<SRC, float>::value)
        return 0;

    if (std::is_same<DST, SRC>::value)
    {
        if (std::is_same<DST, uchar>::value)
        {
            return sub_simd_sametype<uchar, v_uint8>(reinterpret_cast<const uchar*>(in1),
                                                     reinterpret_cast<const uchar*>(in2),
                                                     reinterpret_cast<uchar*>(out), length);
        }
        else if (std::is_same<DST, short>::value)
        {
            return sub_simd_sametype<short, v_int16>(reinterpret_cast<const short*>(in1),
                                                     reinterpret_cast<const short*>(in2),
                                                     reinterpret_cast<short*>(out), length);
        }
        else if (std::is_same<DST, float>::value)
        {
            return sub_simd_sametype<float, v_float32>(reinterpret_cast<const float*>(in1),
                                                       reinterpret_cast<const float*>(in2),
                                                       reinterpret_cast<float*>(out), length);
        }
    }
    else if (std::is_same<SRC, short>::value && std::is_same<DST, uchar>::value)
    {
        constexpr int nlanes = static_cast<int>(v_uint8::nlanes);

        if (length < nlanes)
            return 0;

        int x = 0;
        for (;;)
        {
            for (; x <= length - nlanes; x += nlanes)
            {
                v_int16 a1 = vx_load(reinterpret_cast<const short*>(&in1[x]));
                v_int16 a2 = vx_load(reinterpret_cast<const short*>(&in1[x + nlanes / 2]));
                v_int16 b1 = vx_load(reinterpret_cast<const short*>(&in2[x]));
                v_int16 b2 = vx_load(reinterpret_cast<const short*>(&in2[x + nlanes / 2]));

                vx_store(reinterpret_cast<uchar*>(&out[x]), v_pack_u(a1 - b1, a2 - b2));
            }

            if (x < length)
            {
                CV_DbgAssert((reinterpret_cast<const short*>(in1) != reinterpret_cast<const short*>(out)) &&
                             (reinterpret_cast<const short*>(in2) != reinterpret_cast<const short*>(out)));
                x = length - nlanes;
                continue;  // process one more time (unaligned tail)
            }
            break;
        }

        return x;
    }
    else if (std::is_same<SRC, float>::value && std::is_same<DST, uchar>::value)
    {
        constexpr int nlanes = static_cast<int>(v_uint8::nlanes);

        if (length < nlanes)
            return 0;

        int x = 0;
        for (;;)
        {
            for (; x <= length - nlanes; x += nlanes)
            {
                v_float32 a1 = vx_load(reinterpret_cast<const float*>(&in1[x]));
                v_float32 a2 = vx_load(reinterpret_cast<const float*>(&in1[x + nlanes / 4]));
                v_float32 a3 = vx_load(reinterpret_cast<const float*>(&in1[x + 2 * nlanes / 4]));
                v_float32 a4 = vx_load(reinterpret_cast<const float*>(&in1[x + 3 * nlanes / 4]));

                v_float32 b1 = vx_load(reinterpret_cast<const float*>(&in2[x]));
                v_float32 b2 = vx_load(reinterpret_cast<const float*>(&in2[x + nlanes / 4]));
                v_float32 b3 = vx_load(reinterpret_cast<const float*>(&in2[x + 2 * nlanes / 4]));
                v_float32 b4 = vx_load(reinterpret_cast<const float*>(&in2[x + 3 * nlanes / 4]));

                vx_store(reinterpret_cast<uchar*>(&out[x]), v_pack_u(v_pack(v_round(a1 - b1), v_round(a2 - b2)),
                                                                     v_pack(v_round(a3 - b3), v_round(a4 - b4))));
            }

            if (x < length)
            {
                CV_DbgAssert((reinterpret_cast<const float*>(in1) != reinterpret_cast<const float*>(out)) &&
                             (reinterpret_cast<const float*>(in2) != reinterpret_cast<const float*>(out)));
                x = length - nlanes;
                continue;  // process one more time (unaligned tail)
            }
            break;
        }

        return x;
    }

    return 0;
}
#endif // CV_SIMD

template<typename DST, typename SRC1, typename SRC2>
static void run_arithm(Buffer &dst, const View &src1, const View &src2, Arithm arithm,
                       double scale=1)
{
    static_assert(std::is_same<SRC1, SRC2>::value, "wrong types");

    const auto *in1 = src1.InLine<SRC1>(0);
    const auto *in2 = src2.InLine<SRC2>(0);
          auto *out = dst.OutLine<DST>();

    int width  = dst.length();
    int chan   = dst.meta().chan;
    int length = width * chan;

    // NB: assume in/out types are not 64-bits
    float _scale = static_cast<float>( scale );

    int x = 0;

    switch (arithm)
    {
        case ARITHM_ADD:
        {
#if CV_SIMD
            x = add_simd(in1, in2, out, length);
#endif
            for (; x < length; ++x)
                out[x] = add<DST>(in1[x], in2[x]);
            break;
        }
        case ARITHM_SUBTRACT:
        {
#if CV_SIMD
            x = sub_simd(in1, in2, out, length);
#endif
            for (; x < length; ++x)
                out[x] = sub<DST>(in1[x], in2[x]);
            break;
        }
        case ARITHM_MULTIPLY:
        {
#if CV_SIMD
            x = mul_simd(in1, in2, out, length, scale);
#endif
            for (; x < length; ++x)
                out[x] = mul<DST>(in1[x], in2[x], _scale);
            break;
        }
        case ARITHM_DIVIDE:
        {
#if CV_SIMD
            x = div_simd(in1, in2, out, length, scale);
#endif
            for (; x < length; ++x)
                out[x] = div<DST>(in1[x], in2[x], _scale);
            break;
        }
        default: CV_Error(cv::Error::StsBadArg, "unsupported arithmetic operation");
    }
}

GAPI_FLUID_KERNEL(GFluidAdd, cv::gapi::core::GAdd, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, int /*dtype*/, Buffer &dst)
    {
        //      DST     SRC1    SRC2    OP          __VA_ARGS__
        BINARY_(uchar , uchar , uchar , run_arithm, dst, src1, src2, ARITHM_ADD);
        BINARY_(uchar ,  short,  short, run_arithm, dst, src1, src2, ARITHM_ADD);
        BINARY_(uchar ,  float,  float, run_arithm, dst, src1, src2, ARITHM_ADD);
        BINARY_( short,  short,  short, run_arithm, dst, src1, src2, ARITHM_ADD);
        BINARY_( float, uchar , uchar , run_arithm, dst, src1, src2, ARITHM_ADD);
        BINARY_( float,  short,  short, run_arithm, dst, src1, src2, ARITHM_ADD);
        BINARY_( float,  float,  float, run_arithm, dst, src1, src2, ARITHM_ADD);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidSub, cv::gapi::core::GSub, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, int /*dtype*/, Buffer &dst)
    {
        //      DST     SRC1    SRC2    OP          __VA_ARGS__
        BINARY_(uchar , uchar , uchar , run_arithm, dst, src1, src2, ARITHM_SUBTRACT);
        BINARY_(uchar ,  short,  short, run_arithm, dst, src1, src2, ARITHM_SUBTRACT);
        BINARY_(uchar ,  float,  float, run_arithm, dst, src1, src2, ARITHM_SUBTRACT);
        BINARY_( short,  short,  short, run_arithm, dst, src1, src2, ARITHM_SUBTRACT);
        BINARY_( float, uchar , uchar , run_arithm, dst, src1, src2, ARITHM_SUBTRACT);
        BINARY_( float,  short,  short, run_arithm, dst, src1, src2, ARITHM_SUBTRACT);
        BINARY_( float,  float,  float, run_arithm, dst, src1, src2, ARITHM_SUBTRACT);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidMul, cv::gapi::core::GMul, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, double scale, int /*dtype*/, Buffer &dst)
    {
        //      DST     SRC1    SRC2    OP          __VA_ARGS__
        BINARY_(uchar,  uchar,  uchar,  run_arithm, dst, src1, src2, ARITHM_MULTIPLY, scale);
        BINARY_(uchar,  ushort, ushort, run_arithm, dst, src1, src2, ARITHM_MULTIPLY, scale);
        BINARY_(uchar,  short,  short,  run_arithm, dst, src1, src2, ARITHM_MULTIPLY, scale);
        BINARY_(uchar,  float,  float,  run_arithm, dst, src1, src2, ARITHM_MULTIPLY, scale);
        BINARY_(short,  short,  short,  run_arithm, dst, src1, src2, ARITHM_MULTIPLY, scale);
        BINARY_(short,  ushort, ushort, run_arithm, dst, src1, src2, ARITHM_MULTIPLY, scale);
        BINARY_(short,  uchar,  uchar,  run_arithm, dst, src1, src2, ARITHM_MULTIPLY, scale);
        BINARY_(short,  float,  float,  run_arithm, dst, src1, src2, ARITHM_MULTIPLY, scale);
        BINARY_(ushort, ushort, ushort, run_arithm, dst, src1, src2, ARITHM_MULTIPLY, scale);
        BINARY_(ushort, uchar,  uchar,  run_arithm, dst, src1, src2, ARITHM_MULTIPLY, scale);
        BINARY_(ushort, short,  short,  run_arithm, dst, src1, src2, ARITHM_MULTIPLY, scale);
        BINARY_(ushort, float,  float,  run_arithm, dst, src1, src2, ARITHM_MULTIPLY, scale);
        BINARY_(float,  uchar,  uchar,  run_arithm, dst, src1, src2, ARITHM_MULTIPLY, scale);
        BINARY_(float,  ushort, ushort, run_arithm, dst, src1, src2, ARITHM_MULTIPLY, scale);
        BINARY_(float,  short,  short,  run_arithm, dst, src1, src2, ARITHM_MULTIPLY, scale);
        BINARY_(float,  float,  float,  run_arithm, dst, src1, src2, ARITHM_MULTIPLY, scale);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidDiv, cv::gapi::core::GDiv, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, double scale, int /*dtype*/, Buffer &dst)
    {
        //      DST     SRC1    SRC2    OP          __VA_ARGS__
        BINARY_(uchar , uchar , uchar , run_arithm, dst, src1, src2, ARITHM_DIVIDE, scale);
        BINARY_(uchar,  ushort, ushort, run_arithm, dst, src1, src2, ARITHM_DIVIDE, scale);
        BINARY_(uchar ,  short,  short, run_arithm, dst, src1, src2, ARITHM_DIVIDE, scale);
        BINARY_(uchar ,  float,  float, run_arithm, dst, src1, src2, ARITHM_DIVIDE, scale);
        BINARY_( short,  short,  short, run_arithm, dst, src1, src2, ARITHM_DIVIDE, scale);
        BINARY_( short, ushort, ushort, run_arithm, dst, src1, src2, ARITHM_DIVIDE, scale);
        BINARY_( short,  uchar,  uchar, run_arithm, dst, src1, src2, ARITHM_DIVIDE, scale);
        BINARY_( short,  float,  float, run_arithm, dst, src1, src2, ARITHM_DIVIDE, scale);
        BINARY_(ushort, ushort, ushort, run_arithm, dst, src1, src2, ARITHM_DIVIDE, scale);
        BINARY_(ushort, uchar , uchar , run_arithm, dst, src1, src2, ARITHM_DIVIDE, scale);
        BINARY_(ushort,  short,  short, run_arithm, dst, src1, src2, ARITHM_DIVIDE, scale);
        BINARY_(ushort,  float,  float, run_arithm, dst, src1, src2, ARITHM_DIVIDE, scale);
        BINARY_( float, uchar , uchar , run_arithm, dst, src1, src2, ARITHM_DIVIDE, scale);
        BINARY_( float, ushort, ushort, run_arithm, dst, src1, src2, ARITHM_DIVIDE, scale);
        BINARY_( float,  short,  short, run_arithm, dst, src1, src2, ARITHM_DIVIDE, scale);
        BINARY_( float,  float,  float, run_arithm, dst, src1, src2, ARITHM_DIVIDE, scale);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

template<typename DST, typename SRC1, typename SRC2>
static void run_absdiff(Buffer &dst, const View &src1, const View &src2)
{
    static_assert(std::is_same<SRC1, SRC2>::value, "wrong types");
    static_assert(std::is_same<SRC1, DST>::value, "wrong types");

    const auto *in1 = src1.InLine<SRC1>(0);
    const auto *in2 = src2.InLine<SRC2>(0);
    auto *out = dst.OutLine<DST>();

    int width = dst.length();
    int chan = dst.meta().chan;
    int length = width * chan;

    int x = 0;

#if CV_SIMD
    x = absdiff_simd(in1, in2, out, length);
#endif
    for (; x < length; ++x)
        out[x] = absdiff<DST>(in1[x], in2[x]);
}

GAPI_FLUID_KERNEL(GFluidAbsDiff, cv::gapi::core::GAbsDiff, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, Buffer &dst)
    {
        //      DST     SRC1    SRC2    OP          __VA_ARGS__
        BINARY_(uchar , uchar , uchar , run_absdiff, dst, src1, src2);
        BINARY_(ushort, ushort, ushort, run_absdiff, dst, src1, src2);
        BINARY_( short,  short,  short, run_absdiff, dst, src1, src2);
        BINARY_( float,  float,  float, run_absdiff, dst, src1, src2);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

//--------------------------------------
//
// Fluid kernels: +, -, *, / with Scalar
//
//--------------------------------------

static inline v_uint16x8  v_add_16u(const v_uint16x8 &x, const v_uint16x8 &y) { return x + y; }
static inline v_uint16x8  v_sub_16u(const v_uint16x8 &x, const v_uint16x8 &y) { return x - y; }
static inline v_uint16x8 v_subr_16u(const v_uint16x8 &x, const v_uint16x8 &y) { return y - x; }

static inline v_float32x4  v_add_32f(const v_float32x4 &x, const v_float32x4 &y) { return x + y; }
static inline v_float32x4  v_sub_32f(const v_float32x4 &x, const v_float32x4 &y) { return x - y; }
static inline v_float32x4 v_subr_32f(const v_float32x4 &x, const v_float32x4 &y) { return y - x; }

static inline int  s_add_8u(uchar x, uchar y) { return x + y; }
static inline int  s_sub_8u(uchar x, uchar y) { return x - y; }
static inline int s_subr_8u(uchar x, uchar y) { return y - x; }

static inline float  s_add_32f(float x, float y) { return x + y; }
static inline float  s_sub_32f(float x, float y) { return x - y; }
static inline float s_subr_32f(float x, float y) { return y - x; }

// manual SIMD if important case 8UC3
static void run_arithm_s3(uchar out[], const uchar in[], int width, const uchar scalar[],
                          v_uint16x8 (*v_op)(const v_uint16x8&, const v_uint16x8&),
                          int (*s_op)(uchar, uchar))
{
    int w = 0;

#if CV_SIMD128
    for (; w <= width-16; w+=16)
    {
        v_uint8x16 x, y, z;
        v_load_deinterleave(&in[3*w], x, y, z);

        v_uint16x8 r0, r1;

        v_expand(x, r0, r1);
        r0 = v_op(r0, v_setall_u16(scalar[0])); // x + scalar[0]
        r1 = v_op(r1, v_setall_u16(scalar[0]));
        x = v_pack(r0, r1);

        v_expand(y, r0, r1);
        r0 = v_op(r0, v_setall_u16(scalar[1])); // y + scalar[1]
        r1 = v_op(r1, v_setall_u16(scalar[1]));
        y = v_pack(r0, r1);

        v_expand(z, r0, r1);
        r0 = v_op(r0, v_setall_u16(scalar[2])); // z + scalar[2]
        r1 = v_op(r1, v_setall_u16(scalar[2]));
        z = v_pack(r0, r1);

        v_store_interleave(&out[3*w], x, y, z);
    }
#endif
    cv::util::suppress_unused_warning(v_op);
    for (; w < width; w++)
    {
        out[3*w    ] = saturate<uchar>( s_op(in[3*w    ], scalar[0]) );
        out[3*w + 1] = saturate<uchar>( s_op(in[3*w + 1], scalar[1]) );
        out[3*w + 2] = saturate<uchar>( s_op(in[3*w + 2], scalar[2]) );
    }
}

// manually SIMD if rounding 32F into 8U, single channel
static void run_arithm_s1(uchar out[], const float in[], int width, const float scalar[],
                          v_float32x4 (*v_op)(const v_float32x4&, const v_float32x4&),
                          float (*s_op)(float, float))
{
    int w = 0;

#if CV_SIMD128
    for (; w <= width-16; w+=16)
    {
        v_float32x4 r0, r1, r2, r3;
        r0 = v_load(&in[w     ]);
        r1 = v_load(&in[w +  4]);
        r2 = v_load(&in[w +  8]);
        r3 = v_load(&in[w + 12]);

        r0 = v_op(r0, v_setall_f32(scalar[0])); // r + scalar[0]
        r1 = v_op(r1, v_setall_f32(scalar[0]));
        r2 = v_op(r2, v_setall_f32(scalar[0]));
        r3 = v_op(r3, v_setall_f32(scalar[0]));

        v_int32x4 i0, i1, i2, i3;
        i0 = v_round(r0);
        i1 = v_round(r1);
        i2 = v_round(r2);
        i3 = v_round(r3);

        v_uint16x8 us0, us1;
        us0 = v_pack_u(i0, i1);
        us1 = v_pack_u(i2, i3);

        v_uint8x16 uc;
        uc = v_pack(us0, us1);

        v_store(&out[w], uc);
    }
#endif
    cv::util::suppress_unused_warning(v_op);
    for (; w < width; w++)
    {
        out[w] = saturate<uchar>(s_op(in[w], scalar[0]), roundf);
    }
}

static void run_arithm_s_add3(uchar out[], const uchar in[], int width, const uchar scalar[])
{
    run_arithm_s3(out, in, width, scalar, v_add_16u, s_add_8u);
}

static void run_arithm_s_sub3(uchar out[], const uchar in[], int width, const uchar scalar[])
{
    run_arithm_s3(out, in, width, scalar, v_sub_16u, s_sub_8u);
}

static void run_arithm_s_subr3(uchar out[], const uchar in[], int width, const uchar scalar[])
{
    run_arithm_s3(out, in, width, scalar, v_subr_16u, s_subr_8u); // reverse: subr
}

static void run_arithm_s_add1(uchar out[], const float in[], int width, const float scalar[])
{
    run_arithm_s1(out, in, width, scalar, v_add_32f, s_add_32f);
}

static void run_arithm_s_sub1(uchar out[], const float in[], int width, const float scalar[])
{
    run_arithm_s1(out, in, width, scalar, v_sub_32f, s_sub_32f);
}

static void run_arithm_s_subr1(uchar out[], const float in[], int width, const float scalar[])
{
    run_arithm_s1(out, in, width, scalar, v_subr_32f, s_subr_32f); // reverse: subr
}

// manually unroll the inner cycle by channels
template<typename DST, typename SRC, typename SCALAR, typename FUNC>
static void run_arithm_s(DST out[], const SRC in[], int width, int chan,
                         const SCALAR scalar[4], FUNC func)
{
    if (chan == 4)
    {
        for (int w=0; w < width; w++)
        {
            out[4*w + 0] = func(in[4*w + 0], scalar[0]);
            out[4*w + 1] = func(in[4*w + 1], scalar[1]);
            out[4*w + 2] = func(in[4*w + 2], scalar[2]);
            out[4*w + 3] = func(in[4*w + 3], scalar[3]);
        }
    }
    else
    if (chan == 3)
    {
        for (int w=0; w < width; w++)
        {
            out[3*w + 0] = func(in[3*w + 0], scalar[0]);
            out[3*w + 1] = func(in[3*w + 1], scalar[1]);
            out[3*w + 2] = func(in[3*w + 2], scalar[2]);
        }
    }
    else
    if (chan == 2)
    {
        for (int w=0; w < width; w++)
        {
            out[2*w + 0] = func(in[2*w + 0], scalar[0]);
            out[2*w + 1] = func(in[2*w + 1], scalar[1]);
        }
    }
    else
    if (chan == 1)
    {
        for (int w=0; w < width; w++)
        {
            out[w] = func(in[w], scalar[0]);
        }
    }
    else
        CV_Error(cv::Error::StsBadArg, "unsupported number of channels");
}

#if CV_SIMD
CV_ALWAYS_INLINE void absdiffc_short_store_c1c2c4(short* out_ptr, const v_int32& c1, const v_int32& c2)
{
    vx_store(out_ptr, v_pack(c1, c2));
}

CV_ALWAYS_INLINE void absdiffc_short_store_c1c2c4(ushort* out_ptr, const v_int32& c1, const v_int32& c2)
{
    vx_store(out_ptr, v_pack_u(c1, c2));
}

template<typename T>
CV_ALWAYS_INLINE int absdiffc_simd_c1c2c4(const T in[], T out[],
                                          const v_float32& s, const int length)
{
    static_assert((std::is_same<T, ushort>::value) || (std::is_same<T, short>::value),
                  "This templated overload is only for short or ushort type combinations.");

    constexpr int nlanes = (std::is_same<T, ushort>::value) ? static_cast<int>(v_uint16::nlanes) :
                                                              static_cast<int>(v_int16::nlanes);
    if (length < nlanes)
        return 0;

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            v_float32 a1 = v_load_f32(in + x);
            v_float32 a2 = v_load_f32(in + x + nlanes / 2);

            absdiffc_short_store_c1c2c4(&out[x], v_round(v_absdiff(a1, s)),
                                                 v_round(v_absdiff(a2, s)));
        }

        if (x < length && (in != out))
        {
            x = length - nlanes;
            continue;  // process unaligned tail
        }
        break;
    }
    return x;
}

template<>
CV_ALWAYS_INLINE int absdiffc_simd_c1c2c4<uchar>(const uchar in[], uchar out[],
                                                 const v_float32& s, const int length)
{
    constexpr int nlanes = static_cast<int>(v_uint8::nlanes);

    if (length < nlanes)
        return 0;

    int x = 0;
    for (;;)
    {
        for (; x <= length - nlanes; x += nlanes)
        {
            v_float32 a1 = v_load_f32(in + x);
            v_float32 a2 = v_load_f32(in + x + nlanes / 4);
            v_float32 a3 = v_load_f32(in + x + nlanes / 2);
            v_float32 a4 = v_load_f32(in + x + 3 * nlanes / 4);

            vx_store(&out[x], v_pack_u(v_pack(v_round(v_absdiff(a1, s)),
                                              v_round(v_absdiff(a2, s))),
                                       v_pack(v_round(v_absdiff(a3, s)),
                                              v_round(v_absdiff(a4, s)))));
        }

        if (x < length && (in != out))
        {
            x = length - nlanes;
            continue;  // process unaligned tail
        }
        break;
    }
    return x;
}

CV_ALWAYS_INLINE void absdiffc_short_store_c3(short* out_ptr, const v_int32& c1,
                                              const v_int32& c2, const v_int32& c3,
                                              const v_int32& c4, const v_int32& c5,
                                              const v_int32& c6)
{
    constexpr int nlanes = static_cast<int>(v_int16::nlanes);
    vx_store(out_ptr, v_pack(c1, c2));
    vx_store(out_ptr + nlanes, v_pack(c3, c4));
    vx_store(out_ptr + 2*nlanes, v_pack(c5, c6));
}

CV_ALWAYS_INLINE void absdiffc_short_store_c3(ushort* out_ptr, const v_int32& c1,
                                              const v_int32& c2, const v_int32& c3,
                                              const v_int32& c4, const v_int32& c5,
                                              const v_int32& c6)
{
    constexpr int nlanes = static_cast<int>(v_uint16::nlanes);
    vx_store(out_ptr, v_pack_u(c1, c2));
    vx_store(out_ptr + nlanes, v_pack_u(c3, c4));
    vx_store(out_ptr + 2*nlanes, v_pack_u(c5, c6));
}

template<typename T>
CV_ALWAYS_INLINE int absdiffc_simd_c3_impl(const T in[], T out[],
                                           const v_float32& s1, const v_float32& s2,
                                           const v_float32& s3, const int length)
{
    static_assert((std::is_same<T, ushort>::value) || (std::is_same<T, short>::value),
                  "This templated overload is only for short or ushort type combinations.");

    constexpr int nlanes = (std::is_same<T, ushort>::value) ? static_cast<int>(v_uint16::nlanes):
                                                              static_cast<int>(v_int16::nlanes);

    if (length < 3 * nlanes)
        return 0;

    int x = 0;
    for (;;)
    {
        for (; x <= length - 3 * nlanes; x += 3 * nlanes)
        {
            v_float32 a1 = v_load_f32(in + x);
            v_float32 a2 = v_load_f32(in + x + nlanes / 2);
            v_float32 a3 = v_load_f32(in + x + nlanes);
            v_float32 a4 = v_load_f32(in + x + 3 * nlanes / 2);
            v_float32 a5 = v_load_f32(in + x + 2 * nlanes);
            v_float32 a6 = v_load_f32(in + x + 5 * nlanes / 2);

            absdiffc_short_store_c3(&out[x], v_round(v_absdiff(a1, s1)),
                                             v_round(v_absdiff(a2, s2)),
                                             v_round(v_absdiff(a3, s3)),
                                             v_round(v_absdiff(a4, s1)),
                                             v_round(v_absdiff(a5, s2)),
                                             v_round(v_absdiff(a6, s3)));
        }

        if (x < length && (in != out))
        {
            x = length - 3 * nlanes;
            continue;  // process unaligned tail
        }
        break;
    }
    return x;
}

template<>
CV_ALWAYS_INLINE int absdiffc_simd_c3_impl<uchar>(const uchar in[], uchar out[],
                                                  const v_float32& s1, const v_float32& s2,
                                                  const v_float32& s3, const int length)
{
    constexpr int nlanes = static_cast<int>(v_uint8::nlanes);

    if (length < 3 * nlanes)
        return 0;

    int x = 0;

    for (;;)
    {
        for (; x <= length - 3 * nlanes; x += 3 * nlanes)
        {
            vx_store(&out[x],
                     v_pack_u(v_pack(v_round(v_absdiff(v_load_f32(in + x), s1)),
                                     v_round(v_absdiff(v_load_f32(in + x + nlanes/4), s2))),
                              v_pack(v_round(v_absdiff(v_load_f32(in + x + nlanes/2), s3)),
                                     v_round(v_absdiff(v_load_f32(in + x + 3*nlanes/4), s1)))));

            vx_store(&out[x + nlanes],
                     v_pack_u(v_pack(v_round(v_absdiff(v_load_f32(in + x + nlanes), s2)),
                                     v_round(v_absdiff(v_load_f32(in + x + 5*nlanes/4), s3))),
                              v_pack(v_round(v_absdiff(v_load_f32(in + x + 3*nlanes/2), s1)),
                                     v_round(v_absdiff(v_load_f32(in + x + 7*nlanes/4), s2)))));

            vx_store(&out[x + 2 * nlanes],
                     v_pack_u(v_pack(v_round(v_absdiff(v_load_f32(in + x + 2*nlanes), s3)),
                                     v_round(v_absdiff(v_load_f32(in + x + 9*nlanes/4), s1))),
                              v_pack(v_round(v_absdiff(v_load_f32(in + x + 5*nlanes/2), s2)),
                                     v_round(v_absdiff(v_load_f32(in + x + 11*nlanes/4), s3)))));
        }

        if (x < length && (in != out))
        {
            x = length - 3 * nlanes;
            continue;  // process unaligned tail
        }
        break;
    }
    return x;
}

template<typename T>
CV_ALWAYS_INLINE int absdiffc_simd_channels(const T in[], const float scalar[], T out[],
                                            const int width, int chan)
{
    int length = width * chan;
    v_float32 s = vx_load(scalar);

    return absdiffc_simd_c1c2c4(in, out, s, length);
}

template<typename T>
CV_ALWAYS_INLINE int absdiffc_simd_c3(const T in[], const float scalar[], T out[], int width)
{
    constexpr int chan = 3;
    int length = width * chan;

    v_float32 s1 = vx_load(scalar);
#if CV_SIMD_WIDTH == 32
    v_float32 s2 = vx_load(scalar + 2);
    v_float32 s3 = vx_load(scalar + 1);
#else
    v_float32 s2 = vx_load(scalar + 1);
    v_float32 s3 = vx_load(scalar + 2);
#endif

    return absdiffc_simd_c3_impl(in, out, s1, s2, s3, length);
}

template<typename T>
CV_ALWAYS_INLINE int absdiffc_simd(const T in[], const float scalar[], T out[], int width, int chan)
{
    switch (chan)
    {
    case 1:
    case 2:
    case 4:
        return absdiffc_simd_channels(in, scalar, out, width, chan);
    case 3:
        return absdiffc_simd_c3(in, scalar, out, width);
    default:
        break;
    }

    return 0;
}
#endif  // CV_SIMD

template<typename DST, typename SRC>
static void run_absdiffc(Buffer &dst, const View &src, const float scalar[])
{
    const auto *in = src.InLine<SRC>(0);
    auto *out = dst.OutLine<DST>();

    int width = dst.length();
    int chan = dst.meta().chan;

    int w = 0;
#if CV_SIMD
    w = absdiffc_simd(in, scalar, out, width, chan);
#endif

    for (; w < width*chan; ++w)
        out[w] = absdiff<DST>(in[w], scalar[w%chan]);
}

template<typename DST, typename SRC>
static void run_arithm_s(Buffer &dst, const View &src, const float scalar[4], Arithm arithm,
                         float scale=1)
{
    const auto *in  = src.InLine<SRC>(0);
          auto *out = dst.OutLine<DST>();

    int width  = dst.length();
    int chan   = dst.meta().chan;

    // What if we cast the scalar into the SRC type?
    const SRC myscal[4] = { static_cast<SRC>(scalar[0]), static_cast<SRC>(scalar[1]),
                            static_cast<SRC>(scalar[2]), static_cast<SRC>(scalar[3]) };
    bool usemyscal = (myscal[0] == scalar[0]) && (myscal[1] == scalar[1]) &&
                     (myscal[2] == scalar[2]) && (myscal[3] == scalar[3]);

    switch (arithm)
    {
    case ARITHM_ADD:
        if (usemyscal)
        {
            if (std::is_same<DST,uchar>::value &&
                std::is_same<SRC,uchar>::value &&
                chan == 3)
                run_arithm_s_add3((uchar*)out, (const uchar*)in, width, (const uchar*)myscal);
            else if (std::is_same<DST,uchar>::value &&
                     std::is_same<SRC,float>::value &&
                     chan == 1)
                run_arithm_s_add1((uchar*)out, (const float*)in, width, (const float*)myscal);
            else
                run_arithm_s(out, in, width, chan, myscal, add<DST,SRC,SRC>);
        }
        else
            run_arithm_s(out, in, width, chan, scalar, add<DST,SRC,float>);
        break;
    case ARITHM_SUBTRACT:
        if (usemyscal)
        {
            if (std::is_same<DST,uchar>::value &&
                std::is_same<SRC,uchar>::value &&
                chan == 3)
                run_arithm_s_sub3((uchar*)out, (const uchar*)in, width, (const uchar*)myscal);
            else if (std::is_same<DST,uchar>::value &&
                     std::is_same<SRC,float>::value &&
                     chan == 1)
                run_arithm_s_sub1((uchar*)out, (const float*)in, width, (const float*)myscal);
            else
                run_arithm_s(out, in, width, chan, myscal, sub<DST,SRC,SRC>);
        }
        else
            run_arithm_s(out, in, width, chan, scalar, sub<DST,SRC,float>);
        break;
    // TODO: optimize miltiplication and division
    case ARITHM_MULTIPLY:
        for (int w=0; w < width; w++)
            for (int c=0; c < chan; c++)
                out[chan*w + c] = mul<DST>(in[chan*w + c], scalar[c], scale);
        break;
    case ARITHM_DIVIDE:
        for (int w=0; w < width; w++)
            for (int c=0; c < chan; c++)
                out[chan*w + c] = div<DST>(in[chan*w + c], scalar[c], scale);
        break;
    default: CV_Error(cv::Error::StsBadArg, "unsupported arithmetic operation");
    }
}

template<typename DST, typename SRC>
static void run_arithm_rs(Buffer &dst, const View &src, const float scalar[4], Arithm arithm,
                          float scale=1)
{
    const auto *in  = src.InLine<SRC>(0);
          auto *out = dst.OutLine<DST>();

    int width  = dst.length();
    int chan   = dst.meta().chan;

    // What if we cast the scalar into the SRC type?
    const SRC myscal[4] = { static_cast<SRC>(scalar[0]), static_cast<SRC>(scalar[1]),
                            static_cast<SRC>(scalar[2]), static_cast<SRC>(scalar[3]) };
    bool usemyscal = (myscal[0] == scalar[0]) && (myscal[1] == scalar[1]) &&
                     (myscal[2] == scalar[2]) && (myscal[3] == scalar[3]);

    switch (arithm)
    {
    case ARITHM_SUBTRACT:
        if (usemyscal)
        {
            if (std::is_same<DST,uchar>::value &&
                std::is_same<SRC,uchar>::value &&
                chan == 3)
                run_arithm_s_subr3((uchar*)out, (const uchar*)in, width, (const uchar*)myscal);
            else if (std::is_same<DST,uchar>::value &&
                     std::is_same<SRC,float>::value &&
                     chan == 1)
                run_arithm_s_subr1((uchar*)out, (const float*)in, width, (const float*)myscal);
            else
                run_arithm_s(out, in, width, chan, myscal, subr<DST,SRC,SRC>);
        }
        else
            run_arithm_s(out, in, width, chan, scalar, subr<DST,SRC,float>);
        break;
    // TODO: optimize division
    case ARITHM_DIVIDE:
        for (int w=0; w < width; w++)
            for (int c=0; c < chan; c++)
                out[chan*w + c] = div<DST>(scalar[c], in[chan*w + c], scale);
        break;
    default: CV_Error(cv::Error::StsBadArg, "unsupported arithmetic operation");
    }
}

GAPI_FLUID_KERNEL(GFluidAbsDiffC, cv::gapi::core::GAbsDiffC, true)
{
    static const int Window = 1;

    static void run(const View &src, const cv::Scalar& _scalar, Buffer &dst, Buffer& scratch)
    {
        if (dst.y() == 0)
        {
            const int chan = src.meta().chan;
            float* sc = scratch.OutLine<float>();

            for (int i = 0; i < scratch.length(); ++i)
                sc[i] = static_cast<float>(_scalar[i % chan]);
        }

        const float* scalar = scratch.OutLine<float>();

        //     DST     SRC     OP            __VA_ARGS__
        UNARY_(uchar, uchar, run_absdiffc, dst, src, scalar);
        UNARY_(ushort, ushort, run_absdiffc, dst, src, scalar);
        UNARY_(short, short, run_absdiffc, dst, src, scalar);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }

    static void initScratch(const GMatDesc&, const GScalarDesc&, Buffer& scratch)
    {
#if CV_SIMD
        constexpr int buflen = static_cast<int>(v_float32::nlanes) + 2; // buffer size
#else
        constexpr int buflen = 4;
#endif
        cv::Size bufsize(buflen, 1);
        GMatDesc bufdesc = { CV_32F, 1, bufsize };
        Buffer buffer(bufdesc);
        scratch = std::move(buffer);
    }

    static void resetScratch(Buffer& /* scratch */)
    {
    }
};

GAPI_FLUID_KERNEL(GFluidAddC, cv::gapi::core::GAddC, false)
{
    static const int Window = 1;

    static void run(const View &src, const cv::Scalar &_scalar, int /*dtype*/, Buffer &dst)
    {
        const float scalar[4] = {
            static_cast<float>(_scalar[0]),
            static_cast<float>(_scalar[1]),
            static_cast<float>(_scalar[2]),
            static_cast<float>(_scalar[3])
        };

        //     DST     SRC     OP            __VA_ARGS__
        UNARY_(uchar , uchar , run_arithm_s, dst, src, scalar, ARITHM_ADD);
        UNARY_(uchar ,  short, run_arithm_s, dst, src, scalar, ARITHM_ADD);
        UNARY_(uchar ,  float, run_arithm_s, dst, src, scalar, ARITHM_ADD);
        UNARY_( short,  short, run_arithm_s, dst, src, scalar, ARITHM_ADD);
        UNARY_( float, uchar , run_arithm_s, dst, src, scalar, ARITHM_ADD);
        UNARY_( float,  short, run_arithm_s, dst, src, scalar, ARITHM_ADD);
        UNARY_( float,  float, run_arithm_s, dst, src, scalar, ARITHM_ADD);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidSubC, cv::gapi::core::GSubC, false)
{
    static const int Window = 1;

    static void run(const View &src, const cv::Scalar &_scalar, int /*dtype*/, Buffer &dst)
    {
        const float scalar[4] = {
            static_cast<float>(_scalar[0]),
            static_cast<float>(_scalar[1]),
            static_cast<float>(_scalar[2]),
            static_cast<float>(_scalar[3])
        };

        //     DST     SRC     OP            __VA_ARGS__
        UNARY_(uchar , uchar , run_arithm_s, dst, src, scalar, ARITHM_SUBTRACT);
        UNARY_(uchar ,  short, run_arithm_s, dst, src, scalar, ARITHM_SUBTRACT);
        UNARY_(uchar ,  float, run_arithm_s, dst, src, scalar, ARITHM_SUBTRACT);
        UNARY_( short,  short, run_arithm_s, dst, src, scalar, ARITHM_SUBTRACT);
        UNARY_( float, uchar , run_arithm_s, dst, src, scalar, ARITHM_SUBTRACT);
        UNARY_( float,  short, run_arithm_s, dst, src, scalar, ARITHM_SUBTRACT);
        UNARY_( float,  float, run_arithm_s, dst, src, scalar, ARITHM_SUBTRACT);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidSubRC, cv::gapi::core::GSubRC, false)
{
    static const int Window = 1;

    static void run(const cv::Scalar &_scalar, const View &src, int /*dtype*/, Buffer &dst)
    {
        const float scalar[4] = {
            static_cast<float>(_scalar[0]),
            static_cast<float>(_scalar[1]),
            static_cast<float>(_scalar[2]),
            static_cast<float>(_scalar[3])
        };

        //     DST     SRC     OP             __VA_ARGS__
        UNARY_(uchar , uchar , run_arithm_rs, dst, src, scalar, ARITHM_SUBTRACT);
        UNARY_(uchar ,  short, run_arithm_rs, dst, src, scalar, ARITHM_SUBTRACT);
        UNARY_(uchar ,  float, run_arithm_rs, dst, src, scalar, ARITHM_SUBTRACT);
        UNARY_( short,  short, run_arithm_rs, dst, src, scalar, ARITHM_SUBTRACT);
        UNARY_( float, uchar , run_arithm_rs, dst, src, scalar, ARITHM_SUBTRACT);
        UNARY_( float,  short, run_arithm_rs, dst, src, scalar, ARITHM_SUBTRACT);
        UNARY_( float,  float, run_arithm_rs, dst, src, scalar, ARITHM_SUBTRACT);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidMulC, cv::gapi::core::GMulC, false)
{
    static const int Window = 1;

    static void run(const View &src, const cv::Scalar &_scalar, int /*dtype*/, Buffer &dst)
    {
        const float scalar[4] = {
            static_cast<float>(_scalar[0]),
            static_cast<float>(_scalar[1]),
            static_cast<float>(_scalar[2]),
            static_cast<float>(_scalar[3])
        };
        const float scale = 1.f;

        //     DST     SRC     OP            __VA_ARGS__
        UNARY_(uchar , uchar , run_arithm_s, dst, src, scalar, ARITHM_MULTIPLY, scale);
        UNARY_(uchar ,  short, run_arithm_s, dst, src, scalar, ARITHM_MULTIPLY, scale);
        UNARY_(uchar ,  float, run_arithm_s, dst, src, scalar, ARITHM_MULTIPLY, scale);
        UNARY_( short,  short, run_arithm_s, dst, src, scalar, ARITHM_MULTIPLY, scale);
        UNARY_( float, uchar , run_arithm_s, dst, src, scalar, ARITHM_MULTIPLY, scale);
        UNARY_( float,  short, run_arithm_s, dst, src, scalar, ARITHM_MULTIPLY, scale);
        UNARY_( float,  float, run_arithm_s, dst, src, scalar, ARITHM_MULTIPLY, scale);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidMulCOld, cv::gapi::core::GMulCOld, false)
{
    static const int Window = 1;

    static void run(const View &src, double _scalar, int /*dtype*/, Buffer &dst)
    {
        const float scalar[4] = {
            static_cast<float>(_scalar),
            static_cast<float>(_scalar),
            static_cast<float>(_scalar),
            static_cast<float>(_scalar)
        };
        const float scale = 1.f;

        //     DST     SRC     OP            __VA_ARGS__
        UNARY_(uchar , uchar , run_arithm_s, dst, src, scalar, ARITHM_MULTIPLY, scale);
        UNARY_(uchar ,  short, run_arithm_s, dst, src, scalar, ARITHM_MULTIPLY, scale);
        UNARY_(uchar ,  float, run_arithm_s, dst, src, scalar, ARITHM_MULTIPLY, scale);
        UNARY_( short,  short, run_arithm_s, dst, src, scalar, ARITHM_MULTIPLY, scale);
        UNARY_( float, uchar , run_arithm_s, dst, src, scalar, ARITHM_MULTIPLY, scale);
        UNARY_( float,  short, run_arithm_s, dst, src, scalar, ARITHM_MULTIPLY, scale);
        UNARY_( float,  float, run_arithm_s, dst, src, scalar, ARITHM_MULTIPLY, scale);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidDivC, cv::gapi::core::GDivC, false)
{
    static const int Window = 1;

    static void run(const View &src, const cv::Scalar &_scalar, double _scale, int /*dtype*/,
                    Buffer &dst)
    {
        const float scalar[4] = {
            static_cast<float>(_scalar[0]),
            static_cast<float>(_scalar[1]),
            static_cast<float>(_scalar[2]),
            static_cast<float>(_scalar[3])
        };
        const float scale = static_cast<float>(_scale);

        //     DST     SRC     OP            __VA_ARGS__
        UNARY_(uchar , uchar , run_arithm_s, dst, src, scalar, ARITHM_DIVIDE, scale);
        UNARY_(uchar ,  short, run_arithm_s, dst, src, scalar, ARITHM_DIVIDE, scale);
        UNARY_(uchar ,  float, run_arithm_s, dst, src, scalar, ARITHM_DIVIDE, scale);
        UNARY_( short,  short, run_arithm_s, dst, src, scalar, ARITHM_DIVIDE, scale);
        UNARY_( float, uchar , run_arithm_s, dst, src, scalar, ARITHM_DIVIDE, scale);
        UNARY_( float,  short, run_arithm_s, dst, src, scalar, ARITHM_DIVIDE, scale);
        UNARY_( float,  float, run_arithm_s, dst, src, scalar, ARITHM_DIVIDE, scale);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidDivRC, cv::gapi::core::GDivRC, false)
{
    static const int Window = 1;

    static void run(const cv::Scalar &_scalar, const View &src, double _scale, int /*dtype*/,
                    Buffer &dst)
    {
        const float scalar[4] = {
            static_cast<float>(_scalar[0]),
            static_cast<float>(_scalar[1]),
            static_cast<float>(_scalar[2]),
            static_cast<float>(_scalar[3])
        };
        const float scale = static_cast<float>(_scale);

        //     DST     SRC     OP             __VA_ARGS__
        UNARY_(uchar , uchar , run_arithm_rs, dst, src, scalar, ARITHM_DIVIDE, scale);
        UNARY_(uchar ,  short, run_arithm_rs, dst, src, scalar, ARITHM_DIVIDE, scale);
        UNARY_(uchar ,  float, run_arithm_rs, dst, src, scalar, ARITHM_DIVIDE, scale);
        UNARY_( short,  short, run_arithm_rs, dst, src, scalar, ARITHM_DIVIDE, scale);
        UNARY_( float, uchar , run_arithm_rs, dst, src, scalar, ARITHM_DIVIDE, scale);
        UNARY_( float,  short, run_arithm_rs, dst, src, scalar, ARITHM_DIVIDE, scale);
        UNARY_( float,  float, run_arithm_rs, dst, src, scalar, ARITHM_DIVIDE, scale);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

//-------------------
//
// Fluid kernels: mask
//
//-------------------

template<typename DST, typename SRC>
static void run_mask(Buffer &dst, const View &src, const View &mask)
{
    static_assert(std::is_same<DST, SRC>::value,
        "Input and output types must match");

    int length  = dst.length(); // dst, src and mask have the same size and are single-channel

    const auto *in      = src.InLine<SRC>(0);
    const auto *in_mask = mask.InLine<uchar>(0);
          auto *out     = dst.OutLine<DST>();

    for (int l=0; l < length; l++)
    {
        out[l] = in_mask[l] ? in[l] : 0;
    }
}

GAPI_FLUID_KERNEL(GFluidMask, cv::gapi::core::GMask, false)
{
    static const int Window = 1;

    static void run(const View &src, const View &mask, Buffer &dst)
    {
        if (src.meta().chan != 1 || dst.meta().chan != 1)
            CV_Error(cv::Error::StsBadArg, "input and output must be single-channel");
        if (mask.meta().chan != 1 || mask.meta().depth != CV_8U)
            CV_Error(cv::Error::StsBadArg, "unsupported mask type");

        //     DST     SRC     OP        __VA_ARGS__
        UNARY_(uchar , uchar , run_mask, dst, src, mask);
        UNARY_( short,  short, run_mask, dst, src, mask);
        UNARY_(ushort, ushort, run_mask, dst, src, mask);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

//----------------------------
//
// Fluid math kernels: bitwise
//
//----------------------------

enum Bitwise { BW_AND, BW_OR, BW_XOR, BW_NOT };

template<typename DST, typename SRC1, typename SRC2>
static void run_bitwise2(Buffer &dst, const View &src1, const View &src2, Bitwise bitwise_op)
{
    static_assert(std::is_same<DST, SRC1>::value, "wrong types");
    static_assert(std::is_same<DST, SRC2>::value, "wrong types");

    const auto *in1 = src1.InLine<SRC1>(0);
    const auto *in2 = src2.InLine<SRC2>(0);
          auto *out = dst.OutLine<DST>();

    int width  = dst.length();
    int chan   = dst.meta().chan;
    int length = width * chan;

    switch (bitwise_op)
    {
    case BW_AND:
        for (int l=0; l < length; l++)
            out[l] = in1[l] & in2[l];
        break;
    case BW_OR:
        for (int l=0; l < length; l++)
            out[l] = in1[l] | in2[l];
        break;
    case BW_XOR:
        for (int l=0; l < length; l++)
            out[l] = in1[l] ^ in2[l];
        break;
    default: CV_Error(cv::Error::StsBadArg, "unsupported bitwise operation");
    }
}

template<typename DST, typename SRC>
static void run_bitwise1(Buffer &dst, const View &src, Bitwise bitwise_op)
{
    static_assert(std::is_same<DST, SRC>::value, "wrong types");

    const auto *in  = src.InLine<SRC>(0);
          auto *out = dst.OutLine<DST>();

    int width  = dst.length();
    int chan   = dst.meta().chan;
    int length = width * chan;

    switch (bitwise_op)
    {
    case BW_NOT:
        for (int l=0; l < length; l++)
            out[l] = ~in[l];
        break;
    default: CV_Error(cv::Error::StsBadArg, "unsupported bitwise operation");
    }
}

GAPI_FLUID_KERNEL(GFluidAnd, cv::gapi::core::GAnd, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, Buffer &dst)
    {

        //      DST     SRC1    SRC2    OP            __VA_ARGS__
        BINARY_(uchar , uchar , uchar , run_bitwise2, dst, src1, src2, BW_AND);
        BINARY_(ushort, ushort, ushort, run_bitwise2, dst, src1, src2, BW_AND);
        BINARY_( short,  short,  short, run_bitwise2, dst, src1, src2, BW_AND);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidOr, cv::gapi::core::GOr, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, Buffer &dst)
    {

        //      DST     SRC1    SRC2    OP            __VA_ARGS__
        BINARY_(uchar , uchar , uchar , run_bitwise2, dst, src1, src2, BW_OR);
        BINARY_(ushort, ushort, ushort, run_bitwise2, dst, src1, src2, BW_OR);
        BINARY_( short,  short,  short, run_bitwise2, dst, src1, src2, BW_OR);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidXor, cv::gapi::core::GXor, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, Buffer &dst)
    {

        //      DST     SRC1    SRC2    OP            __VA_ARGS__
        BINARY_(uchar , uchar , uchar , run_bitwise2, dst, src1, src2, BW_XOR);
        BINARY_(ushort, ushort, ushort, run_bitwise2, dst, src1, src2, BW_XOR);
        BINARY_( short,  short,  short, run_bitwise2, dst, src1, src2, BW_XOR);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidNot, cv::gapi::core::GNot, false)
{
    static const int Window = 1;

    static void run(const View &src, Buffer &dst)
    {
        //     DST     SRC     OP            __VA_ARGS__
        UNARY_(uchar , uchar , run_bitwise1, dst, src, BW_NOT);
        UNARY_(ushort, ushort, run_bitwise1, dst, src, BW_NOT);
        UNARY_( short,  short, run_bitwise1, dst, src, BW_NOT);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

//--------------------------------------
//
// Fluid math kernels: bitwise with Scalar
//
//--------------------------------------

static std::array<int,4> convertScalarForBitwise(const cv::Scalar &_scalar)
{
    std::array<int,4> scalarI = {
        static_cast<int>(_scalar[0]),
        static_cast<int>(_scalar[1]),
        static_cast<int>(_scalar[2]),
        static_cast<int>(_scalar[3])
    };

    if (!((_scalar[0] == scalarI[0]) && (_scalar[1] == scalarI[1]) &&
          (_scalar[2] == scalarI[2]) && (_scalar[3] == scalarI[3])))
    {
        CV_Error(cv::Error::StsBadArg, "Bitwise operations make sense with integral types only");
    }
    return scalarI;
}

template<typename DST>
static inline DST bw_andS(DST x, int y)
{
    return x & saturate<DST>(y);
}

template<typename DST>
static inline DST bw_orS(DST x, int y)
{
    return x | saturate<DST>(y);
}

template<typename DST>
static inline DST bw_xorS(DST x, int y)
{
    return x ^ saturate<DST>(y);
}

// manually unroll the inner cycle by channels
// (reuse arithmetic function above of the same purpose)
template<typename DST, typename FUNC>
static inline void run_bitwise_s(DST out[], const DST in[], int width, int chan,
                                 const int scalar[4], FUNC func)
{
    run_arithm_s(out, in, width, chan, scalar, func);
}

template<typename DST, typename SRC>
static void run_bitwise_s(Buffer &dst, const View &src, const int scalar[4], Bitwise bitwise_op)
{
    static_assert(std::is_same<DST, SRC>::value, "wrong types");

    const auto *in  = src.InLine<SRC>(0);
          auto *out = dst.OutLine<DST>();

    int width  = dst.length();
    int chan   = dst.meta().chan;

    switch (bitwise_op)
    {
    case BW_AND:
        run_bitwise_s(out, in, width, chan, scalar, bw_andS<DST>);
        break;
    case BW_OR:
        run_bitwise_s(out, in, width, chan, scalar, bw_orS<DST>);
        break;
    case BW_XOR:
        run_bitwise_s(out, in, width, chan, scalar, bw_xorS<DST>);
        break;
    default: CV_Error(cv::Error::StsBadArg, "unsupported bitwise operation");
    }
}

GAPI_FLUID_KERNEL(GFluidAndS, cv::gapi::core::GAndS, false)
{
    static const int Window = 1;

    static void run(const View &src, const cv::Scalar &_scalar, Buffer &dst)
    {
        std::array<int,4> scalar = convertScalarForBitwise(_scalar);

        //     DST     SRC     OP            __VA_ARGS__
        UNARY_(uchar , uchar , run_bitwise_s, dst, src, scalar.data(), BW_AND);
        UNARY_(ushort, ushort, run_bitwise_s, dst, src, scalar.data(), BW_AND);
        UNARY_( short,  short, run_bitwise_s, dst, src, scalar.data(), BW_AND);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidOrS, cv::gapi::core::GOrS, false)
{
    static const int Window = 1;

    static void run(const View &src, const cv::Scalar &_scalar, Buffer &dst)
    {
        std::array<int,4> scalar = convertScalarForBitwise(_scalar);

        //     DST     SRC     OP            __VA_ARGS__
        UNARY_(uchar , uchar , run_bitwise_s, dst, src, scalar.data(), BW_OR);
        UNARY_(ushort, ushort, run_bitwise_s, dst, src, scalar.data(), BW_OR);
        UNARY_( short,  short, run_bitwise_s, dst, src, scalar.data(), BW_OR);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidXorS, cv::gapi::core::GXorS, false)
{
    static const int Window = 1;

    static void run(const View &src, const cv::Scalar &_scalar, Buffer &dst)
    {
        std::array<int,4> scalar = convertScalarForBitwise(_scalar);

        //     DST     SRC     OP            __VA_ARGS__
        UNARY_(uchar , uchar , run_bitwise_s, dst, src, scalar.data(), BW_XOR);
        UNARY_(ushort, ushort, run_bitwise_s, dst, src, scalar.data(), BW_XOR);
        UNARY_( short,  short, run_bitwise_s, dst, src, scalar.data(), BW_XOR);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

//-------------------
//
// Fluid kernels: LUT
//
//-------------------

GAPI_FLUID_KERNEL(GFluidLUT, cv::gapi::core::GLUT, false)
{
    static const int Window = 1;

    static void run(const View &src, const cv::Mat& lut, Buffer &dst)
    {
        GAPI_Assert(CV_8U == dst.meta().depth);
        GAPI_Assert(CV_8U == src.meta().depth);

        GAPI_DbgAssert(CV_8U == lut.type());
        GAPI_DbgAssert(256 == lut.cols * lut.rows);
        GAPI_DbgAssert(dst.length() == src.length());
        GAPI_DbgAssert(dst.meta().chan == src.meta().chan);

        const auto *in  = src.InLine<uchar>(0);
              auto *out = dst.OutLine<uchar>();

        int width  = dst.length();
        int chan   = dst.meta().chan;
        int length = width * chan;

        for (int l=0; l < length; l++)
            out[l] = lut.data[ in[l] ];
    }
};

//-------------------------
//
// Fluid kernels: convertTo
//
//-------------------------

template<typename DST, typename SRC>
static void run_convertto(Buffer &dst, const View &src, double _alpha, double _beta)
{
    const auto *in  = src.InLine<SRC>(0);
          auto *out = dst.OutLine<DST>();

    int width  = dst.length();
    int chan   = dst.meta().chan;
    int length = width * chan;

    // NB: don't do this if SRC or DST is 64-bit
    auto alpha = static_cast<float>( _alpha );
    auto beta  = static_cast<float>( _beta  );

    // compute faster if no alpha no beta
    if (alpha == 1 && beta == 0)
    {
        // manual SIMD if need rounding
        if (std::is_integral<DST>::value && std::is_floating_point<SRC>::value)
        {
            GAPI_Assert(( std::is_same<SRC,float>::value ));

            int l = 0; // cycle index

        #if CV_SIMD128
            if (std::is_same<DST,uchar>::value)
            {
                for (; l <= length-16; l+=16)
                {
                    v_int32x4 i0, i1, i2, i3;
                    i0 = v_round( v_load( (float*)& in[l     ] ) );
                    i1 = v_round( v_load( (float*)& in[l +  4] ) );
                    i2 = v_round( v_load( (float*)& in[l +  8] ) );
                    i3 = v_round( v_load( (float*)& in[l + 12] ) );

                    v_uint16x8 us0, us1;
                    us0 = v_pack_u(i0, i1);
                    us1 = v_pack_u(i2, i3);

                    v_uint8x16 uc;
                    uc = v_pack(us0, us1);
                    v_store((uchar*)& out[l], uc);
                }
            }
            if (std::is_same<DST,ushort>::value)
            {
                for (; l <= length-8; l+=8)
                {
                    v_int32x4 i0, i1;
                    i0 = v_round( v_load( (float*)& in[l     ] ) );
                    i1 = v_round( v_load( (float*)& in[l +  4] ) );

                    v_uint16x8 us;
                    us = v_pack_u(i0, i1);
                    v_store((ushort*)& out[l], us);
                }
            }
        #endif

            // tail of SIMD cycle
            for (; l < length; l++)
            {
                out[l] = saturate<DST>(in[l], rintf);
            }
        }
        else if (std::is_integral<DST>::value) // here SRC is integral
        {
            for (int l=0; l < length; l++)
            {
                out[l] = saturate<DST>(in[l]);
            }
        }
        else // DST is floating-point, SRC is any
        {
            for (int l=0; l < length; l++)
            {
                out[l] = static_cast<DST>(in[l]);
            }
        }
    }
    else // if alpha or beta is non-trivial
    {
        // TODO: optimize if alpha and beta and data are integral
        for (int l=0; l < length; l++)
        {
            out[l] = saturate<DST>(in[l]*alpha + beta, rintf);
        }
    }
}

GAPI_FLUID_KERNEL(GFluidConvertTo, cv::gapi::core::GConvertTo, false)
{
    static const int Window = 1;

    static void run(const View &src, int /*rtype*/, double alpha, double beta, Buffer &dst)
    {
        //     DST     SRC     OP             __VA_ARGS__
        UNARY_(uchar , uchar , run_convertto, dst, src, alpha, beta);
        UNARY_(uchar , ushort, run_convertto, dst, src, alpha, beta);
        UNARY_(uchar ,  short, run_convertto, dst, src, alpha, beta);
        UNARY_(uchar ,  float, run_convertto, dst, src, alpha, beta);
        UNARY_(ushort, uchar , run_convertto, dst, src, alpha, beta);
        UNARY_(ushort, ushort, run_convertto, dst, src, alpha, beta);
        UNARY_(ushort,  short, run_convertto, dst, src, alpha, beta);
        UNARY_(ushort,  float, run_convertto, dst, src, alpha, beta);
        UNARY_( short, uchar , run_convertto, dst, src, alpha, beta);
        UNARY_( short, ushort, run_convertto, dst, src, alpha, beta);
        UNARY_( short,  short, run_convertto, dst, src, alpha, beta);
        UNARY_( short,  float, run_convertto, dst, src, alpha, beta);
        UNARY_( float, uchar , run_convertto, dst, src, alpha, beta);
        UNARY_( float, ushort, run_convertto, dst, src, alpha, beta);
        UNARY_( float,  short, run_convertto, dst, src, alpha, beta);
        UNARY_( float,  float, run_convertto, dst, src, alpha, beta);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

//-----------------------------
//
// Fluid math kernels: min, max
//
//-----------------------------

enum Minmax { MM_MIN, MM_MAX };

template<typename DST, typename SRC1, typename SRC2>
static void run_minmax(Buffer &dst, const View &src1, const View &src2, Minmax minmax)
{
    static_assert(std::is_same<DST, SRC1>::value, "wrong types");
    static_assert(std::is_same<DST, SRC2>::value, "wrong types");

    const auto *in1 = src1.InLine<SRC1>(0);
    const auto *in2 = src2.InLine<SRC2>(0);
          auto *out = dst.OutLine<DST>();

    int width = dst.length();
    int chan  = dst.meta().chan;

    int length = width * chan;

    switch (minmax)
    {
    case MM_MIN:
        for (int l=0; l < length; l++)
            out[l] = in1[l] < in2[l]? in1[l]: in2[l];
        break;
    case MM_MAX:
        for (int l=0; l < length; l++)
            out[l] = in1[l] > in2[l]? in1[l]: in2[l];
        break;
    default: CV_Error(cv::Error::StsBadArg, "unsupported min/max operation");
    }
}

GAPI_FLUID_KERNEL(GFluidMin, cv::gapi::core::GMin, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, Buffer &dst)
    {
        //      DST     SRC1    SRC2    OP          __VA_ARGS__
        BINARY_(uchar , uchar , uchar , run_minmax, dst, src1, src2, MM_MIN);
        BINARY_(ushort, ushort, ushort, run_minmax, dst, src1, src2, MM_MIN);
        BINARY_( short,  short,  short, run_minmax, dst, src1, src2, MM_MIN);
        BINARY_( float,  float,  float, run_minmax, dst, src1, src2, MM_MIN);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidMax, cv::gapi::core::GMax, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, Buffer &dst)
    {
        //      DST     SRC1    SRC2    OP          __VA_ARGS__
        BINARY_(uchar , uchar , uchar , run_minmax, dst, src1, src2, MM_MAX);
        BINARY_(ushort, ushort, ushort, run_minmax, dst, src1, src2, MM_MAX);
        BINARY_( short,  short,  short, run_minmax, dst, src1, src2, MM_MAX);
        BINARY_( float,  float,  float, run_minmax, dst, src1, src2, MM_MAX);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

//-----------------------
//
// Fluid kernels: compare
//
//-----------------------

enum Compare { CMP_EQ, CMP_NE, CMP_GE, CMP_GT, CMP_LE, CMP_LT };

template<typename DST, typename SRC1, typename SRC2>
static void run_cmp(Buffer &dst, const View &src1, const View &src2, Compare compare)
{
    static_assert(std::is_same<SRC1, SRC2>::value, "wrong types");
    static_assert(std::is_same<DST, uchar>::value, "wrong types");

    const auto *in1 = src1.InLine<SRC1>(0);
    const auto *in2 = src2.InLine<SRC2>(0);
          auto *out = dst.OutLine<DST>();

    int width = dst.length();
    int chan  = dst.meta().chan;

    int length = width * chan;

    switch (compare)
    {
    case CMP_EQ:
        for (int l=0; l < length; l++)
            out[l] = in1[l] == in2[l]? 255: 0;
        break;
    case CMP_NE:
        for (int l=0; l < length; l++)
            out[l] = in1[l] != in2[l]? 255: 0;
        break;
    case CMP_GE:
        for (int l=0; l < length; l++)
            out[l] = in1[l] >= in2[l]? 255: 0;
        break;
    case CMP_LE:
        for (int l=0; l < length; l++)
            out[l] = in1[l] <= in2[l]? 255: 0;
        break;
    case CMP_GT:
        for (int l=0; l < length; l++)
            out[l] = in1[l] > in2[l]? 255: 0;
        break;
    case CMP_LT:
        for (int l=0; l < length; l++)
            out[l] = in1[l] < in2[l]? 255: 0;
        break;
    default:
        CV_Error(cv::Error::StsBadArg, "unsupported compare operation");
    }
}

GAPI_FLUID_KERNEL(GFluidCmpEQ, cv::gapi::core::GCmpEQ, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, Buffer &dst)
    {
        //      DST    SRC1    SRC2    OP       __VA_ARGS__
        BINARY_(uchar, uchar , uchar , run_cmp, dst, src1, src2, CMP_EQ);
        BINARY_(uchar,  short,  short, run_cmp, dst, src1, src2, CMP_EQ);
        BINARY_(uchar,  float,  float, run_cmp, dst, src1, src2, CMP_EQ);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidCmpNE, cv::gapi::core::GCmpNE, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, Buffer &dst)
    {
        //      DST    SRC1    SRC2    OP       __VA_ARGS__
        BINARY_(uchar, uchar , uchar , run_cmp, dst, src1, src2, CMP_NE);
        BINARY_(uchar,  short,  short, run_cmp, dst, src1, src2, CMP_NE);
        BINARY_(uchar,  float,  float, run_cmp, dst, src1, src2, CMP_NE);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidCmpGE, cv::gapi::core::GCmpGE, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, Buffer &dst)
    {
        //      DST    SRC1    SRC2    OP       __VA_ARGS__
        BINARY_(uchar, uchar , uchar , run_cmp, dst, src1, src2, CMP_GE);
        BINARY_(uchar,  short,  short, run_cmp, dst, src1, src2, CMP_GE);
        BINARY_(uchar,  float,  float, run_cmp, dst, src1, src2, CMP_GE);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidCmpGT, cv::gapi::core::GCmpGT, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, Buffer &dst)
    {
        //      DST    SRC1    SRC2    OP       __VA_ARGS__
        BINARY_(uchar, uchar , uchar , run_cmp, dst, src1, src2, CMP_GT);
        BINARY_(uchar,  short,  short, run_cmp, dst, src1, src2, CMP_GT);
        BINARY_(uchar,  float,  float, run_cmp, dst, src1, src2, CMP_GT);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidCmpLE, cv::gapi::core::GCmpLE, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, Buffer &dst)
    {
        //      DST    SRC1    SRC2    OP       __VA_ARGS__
        BINARY_(uchar, uchar , uchar , run_cmp, dst, src1, src2, CMP_LE);
        BINARY_(uchar,  short,  short, run_cmp, dst, src1, src2, CMP_LE);
        BINARY_(uchar,  float,  float, run_cmp, dst, src1, src2, CMP_LE);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidCmpLT, cv::gapi::core::GCmpLT, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, Buffer &dst)
    {
        //      DST    SRC1    SRC2    OP       __VA_ARGS__
        BINARY_(uchar, uchar , uchar , run_cmp, dst, src1, src2, CMP_LT);
        BINARY_(uchar,  short,  short, run_cmp, dst, src1, src2, CMP_LT);
        BINARY_(uchar,  float,  float, run_cmp, dst, src1, src2, CMP_LT);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

//---------------------
//
// Compare with GScalar
//
//---------------------

template<typename DST, typename SRC, typename SCALAR=double>
static void run_cmp(DST out[], const SRC in[], int length, Compare compare, SCALAR s)
{
    switch (compare)
    {
    case CMP_EQ:
        for (int l=0; l < length; l++)
            out[l] = in[l] == s? 255: 0;
        break;
    case CMP_NE:
        for (int l=0; l < length; l++)
            out[l] = in[l] != s? 255: 0;
        break;
    case CMP_GE:
        for (int l=0; l < length; l++)
            out[l] = in[l] >= s? 255: 0;
        break;
    case CMP_LE:
        for (int l=0; l < length; l++)
            out[l] = in[l] <= s? 255: 0;
        break;
    case CMP_GT:
        for (int l=0; l < length; l++)
            out[l] = in[l] > s? 255: 0;
        break;
    case CMP_LT:
        for (int l=0; l < length; l++)
            out[l] = in[l] < s? 255: 0;
        break;
    default:
        CV_Error(cv::Error::StsBadArg, "unsupported compare operation");
    }
}

template<typename DST, typename SRC>
static void run_cmp(Buffer &dst, const View &src, Compare compare, const cv::Scalar &scalar)
{
    static_assert(std::is_same<DST, uchar>::value, "wrong types");

    const auto *in  = src.InLine<SRC>(0);
          auto *out = dst.OutLine<DST>();

    int width = dst.length();
    int chan  = dst.meta().chan;

    int length = width * chan;

    // compute faster if scalar rounds to SRC
    double d =                   scalar[0]  ;
    SRC    s = static_cast<SRC>( scalar[0] );

    if (s == d)
        run_cmp(out, in, length, compare, s);
    else
        run_cmp(out, in, length, compare, d);
}

GAPI_FLUID_KERNEL(GFluidCmpEQScalar, cv::gapi::core::GCmpEQScalar, false)
{
    static const int Window = 1;

    static void run(const View &src, const cv::Scalar &scalar, Buffer &dst)
    {
        //     DST    SRC     OP       __VA_ARGS__
        UNARY_(uchar, uchar , run_cmp, dst, src, CMP_EQ, scalar);
        UNARY_(uchar,  short, run_cmp, dst, src, CMP_EQ, scalar);
        UNARY_(uchar,  float, run_cmp, dst, src, CMP_EQ, scalar);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidCmpNEScalar, cv::gapi::core::GCmpNEScalar, false)
{
    static const int Window = 1;

    static void run(const View &src, const cv::Scalar &scalar, Buffer &dst)
    {
        //     DST    SRC     OP       __VA_ARGS__
        UNARY_(uchar, uchar , run_cmp, dst, src, CMP_NE, scalar);
        UNARY_(uchar,  short, run_cmp, dst, src, CMP_NE, scalar);
        UNARY_(uchar,  float, run_cmp, dst, src, CMP_NE, scalar);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidCmpGEScalar, cv::gapi::core::GCmpGEScalar, false)
{
    static const int Window = 1;

    static void run(const View &src, const cv::Scalar &scalar, Buffer &dst)
    {
        //     DST    SRC     OP       __VA_ARGS__
        UNARY_(uchar, uchar , run_cmp, dst, src, CMP_GE, scalar);
        UNARY_(uchar,  short, run_cmp, dst, src, CMP_GE, scalar);
        UNARY_(uchar,  float, run_cmp, dst, src, CMP_GE, scalar);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidCmpGTScalar, cv::gapi::core::GCmpGTScalar, false)
{
    static const int Window = 1;

    static void run(const View &src, const cv::Scalar &scalar, Buffer &dst)
    {
        //     DST    SRC     OP       __VA_ARGS__
        UNARY_(uchar, uchar , run_cmp, dst, src, CMP_GT, scalar);
        UNARY_(uchar,  short, run_cmp, dst, src, CMP_GT, scalar);
        UNARY_(uchar,  float, run_cmp, dst, src, CMP_GT, scalar);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidCmpLEScalar, cv::gapi::core::GCmpLEScalar, false)
{
    static const int Window = 1;

    static void run(const View &src, const cv::Scalar &scalar, Buffer &dst)
    {
        //     DST    SRC     OP       __VA_ARGS__
        UNARY_(uchar, uchar , run_cmp, dst, src, CMP_LE, scalar);
        UNARY_(uchar,  short, run_cmp, dst, src, CMP_LE, scalar);
        UNARY_(uchar,  float, run_cmp, dst, src, CMP_LE, scalar);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

GAPI_FLUID_KERNEL(GFluidCmpLTScalar, cv::gapi::core::GCmpLTScalar, false)
{
    static const int Window = 1;

    static void run(const View &src, const cv::Scalar &scalar, Buffer &dst)
    {
        //     DST    SRC     OP       __VA_ARGS__
        UNARY_(uchar, uchar , run_cmp, dst, src, CMP_LT, scalar);
        UNARY_(uchar,  short, run_cmp, dst, src, CMP_LT, scalar);
        UNARY_(uchar,  float, run_cmp, dst, src, CMP_LT, scalar);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

//-------------------------
//
// Fluid kernels: threshold
//
//-------------------------

template<typename DST, typename SRC>
static void run_threshold(Buffer &dst, const View &src, const cv::Scalar &thresh,
                                                        const cv::Scalar &maxval,
                                                                     int  type)
{
    static_assert(std::is_same<DST, SRC>::value, "wrong types");

    const auto *in  = src.InLine<SRC>(0);
          auto *out = dst.OutLine<DST>();

    int width = dst.length();
    int chan  = dst.meta().chan;

    int length = width * chan;

    DST thresh_ = saturate<DST>(thresh[0], floord);
    DST threshd = saturate<DST>(thresh[0], roundd);
    DST maxvald = saturate<DST>(maxval[0], roundd);

    switch (type)
    {
    case cv::THRESH_BINARY:
        for (int l=0; l < length; l++)
            out[l] = in[l] > thresh_? maxvald: 0;
        break;
    case cv::THRESH_BINARY_INV:
        for (int l=0; l < length; l++)
            out[l] = in[l] > thresh_? 0: maxvald;
        break;
    case cv::THRESH_TRUNC:
        for (int l=0; l < length; l++)
            out[l] = in[l] > thresh_? threshd: in[l];
        break;
    case cv::THRESH_TOZERO:
        for (int l=0; l < length; l++)
            out[l] = in[l] > thresh_? in[l]: 0;
        break;
    case cv::THRESH_TOZERO_INV:
        for (int l=0; l < length; l++)
            out[l] = in[l] > thresh_? 0: in[l];
        break;
    default: CV_Error(cv::Error::StsBadArg, "unsupported threshold type");
    }
}

GAPI_FLUID_KERNEL(GFluidThreshold, cv::gapi::core::GThreshold, false)
{
    static const int Window = 1;

    static void run(const View &src, const cv::Scalar &thresh,
                                     const cv::Scalar &maxval,
                                                  int  type,
                        Buffer &dst)
    {
        //     DST     SRC     OP             __VA_ARGS__
        UNARY_(uchar , uchar , run_threshold, dst, src, thresh, maxval, type);
        UNARY_(ushort, ushort, run_threshold, dst, src, thresh, maxval, type);
        UNARY_( short,  short, run_threshold, dst, src, thresh, maxval, type);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

//------------------------
//
// Fluid kernels: in-range
//
//------------------------

static void run_inrange3(uchar out[], const uchar in[], int width,
                         const uchar lower[], const uchar upper[])
{
    int w = 0; // cycle index

#if CV_SIMD128
    for (; w <= width-16; w+=16)
    {
        v_uint8x16 i0, i1, i2;
        v_load_deinterleave(&in[3*w], i0, i1, i2);

        v_uint8x16 o;
        o = (i0 >= v_setall_u8(lower[0])) & (i0 <= v_setall_u8(upper[0])) &
            (i1 >= v_setall_u8(lower[1])) & (i1 <= v_setall_u8(upper[1])) &
            (i2 >= v_setall_u8(lower[2])) & (i2 <= v_setall_u8(upper[2]));

        v_store(&out[w], o);
    }
#endif

    for (; w < width; w++)
    {
        out[w] = in[3*w  ] >= lower[0] && in[3*w  ] <= upper[0] &&
                 in[3*w+1] >= lower[1] && in[3*w+1] <= upper[1] &&
                 in[3*w+2] >= lower[2] && in[3*w+2] <= upper[2] ? 255: 0;
    }
}

template<typename DST, typename SRC>
static void run_inrange(Buffer &dst, const View &src, const cv::Scalar &upperb,
                                                      const cv::Scalar &lowerb)
{
    static_assert(std::is_same<DST, uchar>::value, "wrong types");

    const auto *in  = src.InLine<SRC>(0);
          auto *out = dst.OutLine<DST>();

    int width = src.length();
    int chan  = src.meta().chan;
    GAPI_Assert(dst.meta().chan == 1);

    SRC lower[4], upper[4];
    for (int c=0; c < chan; c++)
    {
        if (std::is_integral<SRC>::value)
        {
            // for integral input, in[i] >= lower equals in[i] >= ceil(lower)
            // so we can optimize compare operations by rounding lower/upper
            lower[c] = saturate<SRC>(lowerb[c],  ceild);
            upper[c] = saturate<SRC>(upperb[c], floord);
        }
        else
        {
            // FIXME: now values used in comparison are floats (while they
            // have double precision initially). Comparison float/float
            // may differ from float/double (how it should work in this case)
            //
            // Example: threshold=1/3 (or 1/10)
            lower[c] = static_cast<SRC>(lowerb[c]);
            upper[c] = static_cast<SRC>(upperb[c]);
        }
    }

    // manually SIMD for important case if RGB/BGR
    if (std::is_same<SRC,uchar>::value && chan==3)
    {
        run_inrange3((uchar*)out, (const uchar*)in, width,
                     (const uchar*)lower, (const uchar*)upper);
        return;
    }

    // TODO: please manually SIMD if multiple channels:
    // modern compilers would perfectly vectorize this code if one channel,
    // but may need help with de-interleaving channels if RGB/BGR image etc
    switch (chan)
    {
    case 1:
        for (int w=0; w < width; w++)
            out[w] = in[w] >= lower[0] && in[w] <= upper[0]? 255: 0;
        break;
    case 2:
        for (int w=0; w < width; w++)
            out[w] = in[2*w  ] >= lower[0] && in[2*w  ] <= upper[0] &&
                     in[2*w+1] >= lower[1] && in[2*w+1] <= upper[1] ? 255: 0;
        break;
    case 3:
        for (int w=0; w < width; w++)
            out[w] = in[3*w  ] >= lower[0] && in[3*w  ] <= upper[0] &&
                     in[3*w+1] >= lower[1] && in[3*w+1] <= upper[1] &&
                     in[3*w+2] >= lower[2] && in[3*w+2] <= upper[2] ? 255: 0;
        break;
    case 4:
        for (int w=0; w < width; w++)
            out[w] = in[4*w  ] >= lower[0] && in[4*w  ] <= upper[0] &&
                     in[4*w+1] >= lower[1] && in[4*w+1] <= upper[1] &&
                     in[4*w+2] >= lower[2] && in[4*w+2] <= upper[2] &&
                     in[4*w+3] >= lower[3] && in[4*w+3] <= upper[3] ? 255: 0;
        break;
    default: CV_Error(cv::Error::StsBadArg, "unsupported number of channels");
    }
}

GAPI_FLUID_KERNEL(GFluidInRange, cv::gapi::core::GInRange, false)
{
    static const int Window = 1;

    static void run(const View &src, const cv::Scalar &lowerb, const cv::Scalar& upperb,
                        Buffer &dst)
    {
        //       DST     SRC    OP           __VA_ARGS__
        INRANGE_(uchar, uchar , run_inrange, dst, src, upperb, lowerb);
        INRANGE_(uchar, ushort, run_inrange, dst, src, upperb, lowerb);
        INRANGE_(uchar,  short, run_inrange, dst, src, upperb, lowerb);
        INRANGE_(uchar,  float, run_inrange, dst, src, upperb, lowerb);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

//----------------------
//
// Fluid kernels: select
//
//----------------------

// manually vectored function for important case if RGB/BGR image
static void run_select_row3(int width, uchar out[], uchar in1[], uchar in2[], uchar in3[])
{
    int w = 0; // cycle index

#if CV_SIMD128
    for (; w <= width-16; w+=16)
    {
        v_uint8x16 a1, b1, c1;
        v_uint8x16 a2, b2, c2;
        v_uint8x16 mask;
        v_uint8x16 a, b, c;

        v_load_deinterleave(&in1[3*w], a1, b1, c1);
        v_load_deinterleave(&in2[3*w], a2, b2, c2);

        mask = v_load(&in3[w]);
        mask = mask != v_setzero_u8();

        a = v_select(mask, a1, a2);
        b = v_select(mask, b1, b2);
        c = v_select(mask, c1, c2);

        v_store_interleave(&out[3*w], a, b, c);
    }
#endif

    for (; w < width; w++)
    {
        out[3*w    ] = in3[w]? in1[3*w    ]: in2[3*w    ];
        out[3*w + 1] = in3[w]? in1[3*w + 1]: in2[3*w + 1];
        out[3*w + 2] = in3[w]? in1[3*w + 2]: in2[3*w + 2];
    }
}

// parameter chan is compile-time known constant, normally chan=1..4
template<int chan, typename DST, typename SRC1, typename SRC2, typename SRC3>
static void run_select_row(int width, DST out[], SRC1 in1[], SRC2 in2[], SRC3 in3[])
{
    if (std::is_same<DST,uchar>::value && chan==3)
    {
        // manually vectored function for important case if RGB/BGR image
        run_select_row3(width, (uchar*)out, (uchar*)in1, (uchar*)in2, (uchar*)in3);
        return;
    }

    // because `chan` is template parameter, its value is known at compilation time,
    // so that modern compilers would efficiently vectorize this cycle if chan==1
    // (if chan>1, compilers may need help with de-interleaving of the channels)
    for (int w=0; w < width; w++)
    {
        for (int c=0; c < chan; c++)
        {
            out[w*chan + c] = in3[w]? in1[w*chan + c]: in2[w*chan + c];
        }
    }
}

template<typename DST, typename SRC1, typename SRC2, typename SRC3>
static void run_select(Buffer &dst, const View &src1, const View &src2, const View &src3)
{
    static_assert(std::is_same<DST ,  SRC1>::value, "wrong types");
    static_assert(std::is_same<DST ,  SRC2>::value, "wrong types");
    static_assert(std::is_same<uchar, SRC3>::value, "wrong types");

    auto *out = dst.OutLine<DST>();

    const auto *in1 = src1.InLine<SRC1>(0);
    const auto *in2 = src2.InLine<SRC2>(0);
    const auto *in3 = src3.InLine<SRC3>(0);

    int width = dst.length();
    int chan  = dst.meta().chan;

    switch (chan)
    {
    case 1: run_select_row<1>(width, out, in1, in2, in3); break;
    case 2: run_select_row<2>(width, out, in1, in2, in3); break;
    case 3: run_select_row<3>(width, out, in1, in2, in3); break;
    case 4: run_select_row<4>(width, out, in1, in2, in3); break;
    default: CV_Error(cv::Error::StsBadArg, "unsupported number of channels");
    }
}

GAPI_FLUID_KERNEL(GFluidSelect, cv::gapi::core::GSelect, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, const View &src3, Buffer &dst)
    {
        //      DST     SRC1    SRC2    SRC3   OP          __VA_ARGS__
        SELECT_(uchar , uchar , uchar , uchar, run_select, dst, src1, src2, src3);
        SELECT_(ushort, ushort, ushort, uchar, run_select, dst, src1, src2, src3);
        SELECT_( short,  short,  short, uchar, run_select, dst, src1, src2, src3);

        CV_Error(cv::Error::StsBadArg, "unsupported combination of types");
    }
};

//----------------------------------------------------
//
// Fluid kernels: split, merge, polat2cart, cart2polar
//
//----------------------------------------------------

GAPI_FLUID_KERNEL(GFluidSplit3, cv::gapi::core::GSplit3, false)
{
    static const int Window = 1;

    static void run(const View &src, Buffer &dst1, Buffer &dst2, Buffer &dst3)
    {
        const auto *in   =  src.InLine<uchar>(0);
              auto *out1 = dst1.OutLine<uchar>();
              auto *out2 = dst2.OutLine<uchar>();
              auto *out3 = dst3.OutLine<uchar>();

        GAPI_Assert(3 == src.meta().chan);
        int width = src.length();

        int w = 0; // cycle counter

    #if CV_SIMD128
        for (; w <= width-16; w+=16)
        {
            v_uint8x16 a, b, c;
            v_load_deinterleave(&in[3*w], a, b, c);
            v_store(&out1[w], a);
            v_store(&out2[w], b);
            v_store(&out3[w], c);
        }
    #endif

        for (; w < width; w++)
        {
            out1[w] = in[3*w    ];
            out2[w] = in[3*w + 1];
            out3[w] = in[3*w + 2];
        }
    }
};

GAPI_FLUID_KERNEL(GFluidSplit4, cv::gapi::core::GSplit4, false)
{
    static const int Window = 1;

    static void run(const View &src, Buffer &dst1, Buffer &dst2, Buffer &dst3, Buffer &dst4)
    {
        const auto *in   =  src.InLine<uchar>(0);
              auto *out1 = dst1.OutLine<uchar>();
              auto *out2 = dst2.OutLine<uchar>();
              auto *out3 = dst3.OutLine<uchar>();
              auto *out4 = dst4.OutLine<uchar>();

        GAPI_Assert(4 == src.meta().chan);
        int width = src.length();

        int w = 0; // cycle counter

    #if CV_SIMD128
        for (; w <= width-16; w+=16)
        {
            v_uint8x16 a, b, c, d;
            v_load_deinterleave(&in[4*w], a, b, c, d);
            v_store(&out1[w], a);
            v_store(&out2[w], b);
            v_store(&out3[w], c);
            v_store(&out4[w], d);
        }
    #endif

        for (; w < width; w++)
        {
            out1[w] = in[4*w    ];
            out2[w] = in[4*w + 1];
            out3[w] = in[4*w + 2];
            out4[w] = in[4*w + 3];
        }
    }
};

GAPI_FLUID_KERNEL(GFluidMerge3, cv::gapi::core::GMerge3, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, const View &src3, Buffer &dst)
    {
        const auto *in1 = src1.InLine<uchar>(0);
        const auto *in2 = src2.InLine<uchar>(0);
        const auto *in3 = src3.InLine<uchar>(0);
              auto *out = dst.OutLine<uchar>();

        GAPI_Assert(3 == dst.meta().chan);
        int width = dst.length();

        int w = 0; // cycle counter

    #if CV_SIMD128
        for (; w <= width-16; w+=16)
        {
            v_uint8x16 a, b, c;
            a = v_load(&in1[w]);
            b = v_load(&in2[w]);
            c = v_load(&in3[w]);
            v_store_interleave(&out[3*w], a, b, c);
        }
    #endif

        for (; w < width; w++)
        {
            out[3*w    ] = in1[w];
            out[3*w + 1] = in2[w];
            out[3*w + 2] = in3[w];
        }
    }
};

GAPI_FLUID_KERNEL(GFluidMerge4, cv::gapi::core::GMerge4, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, const View &src3, const View &src4,
                    Buffer &dst)
    {
        const auto *in1 = src1.InLine<uchar>(0);
        const auto *in2 = src2.InLine<uchar>(0);
        const auto *in3 = src3.InLine<uchar>(0);
        const auto *in4 = src4.InLine<uchar>(0);
              auto *out = dst.OutLine<uchar>();

        GAPI_Assert(4 == dst.meta().chan);
        int width = dst.length();

        int w = 0; // cycle counter

    #if CV_SIMD128
        for (; w <= width-16; w+=16)
        {
            v_uint8x16 a, b, c, d;
            a = v_load(&in1[w]);
            b = v_load(&in2[w]);
            c = v_load(&in3[w]);
            d = v_load(&in4[w]);
            v_store_interleave(&out[4*w], a, b, c, d);
        }
    #endif

        for (; w < width; w++)
        {
            out[4*w    ] = in1[w];
            out[4*w + 1] = in2[w];
            out[4*w + 2] = in3[w];
            out[4*w + 3] = in4[w];
        }
    }
};

GAPI_FLUID_KERNEL(GFluidPolarToCart, cv::gapi::core::GPolarToCart, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, bool angleInDegrees,
                    Buffer &dst1, Buffer &dst2)
    {
        GAPI_Assert(src1.meta().depth == CV_32F);
        GAPI_Assert(src2.meta().depth == CV_32F);
        GAPI_Assert(dst1.meta().depth == CV_32F);
        GAPI_Assert(dst2.meta().depth == CV_32F);

        const auto * in1 = src1.InLine<float>(0);
        const auto * in2 = src2.InLine<float>(0);
              auto *out1 = dst1.OutLine<float>();
              auto *out2 = dst2.OutLine<float>();

        int width = src1.length();
        int chan  = src2.meta().chan;
        int length = width * chan;

        // SIMD: compiler vectoring!
        for (int l=0; l < length; l++)
        {
            float angle = angleInDegrees?
                          in2[l] * static_cast<float>(CV_PI / 180):
                          in2[l];
            float magnitude = in1[l];
            float x = magnitude * cosf(angle);
            float y = magnitude * sinf(angle);
            out1[l] = x;
            out2[l] = y;
        }
    }
};

GAPI_FLUID_KERNEL(GFluidCartToPolar, cv::gapi::core::GCartToPolar, false)
{
    static const int Window = 1;

    static void run(const View &src1, const View &src2, bool angleInDegrees,
                    Buffer &dst1, Buffer &dst2)
    {
        GAPI_Assert(src1.meta().depth == CV_32F);
        GAPI_Assert(src2.meta().depth == CV_32F);
        GAPI_Assert(dst1.meta().depth == CV_32F);
        GAPI_Assert(dst2.meta().depth == CV_32F);

        const auto * in1 = src1.InLine<float>(0);
        const auto * in2 = src2.InLine<float>(0);
              auto *out1 = dst1.OutLine<float>();
              auto *out2 = dst2.OutLine<float>();

        int width = src1.length();
        int chan  = src2.meta().chan;
        int length = width * chan;

        // SIMD: compiler vectoring!
        for (int l=0; l < length; l++)
        {
            float x = in1[l];
            float y = in2[l];
            float magnitude = hypotf(y, x);
            float angle_rad = atan2f(y, x);
            float angle = angleInDegrees?
                          angle_rad * static_cast<float>(180 / CV_PI):
                          angle_rad;
            out1[l] = magnitude;
            out2[l] = angle;
        }
    }
};

GAPI_FLUID_KERNEL(GFluidPhase, cv::gapi::core::GPhase, false)
{
    static const int Window = 1;

    static void run(const View &src_x,
                    const View &src_y,
                    bool angleInDegrees,
                    Buffer &dst)
    {
        const auto w = dst.length() * dst.meta().chan;
        if (src_x.meta().depth == CV_32F && src_y.meta().depth == CV_32F)
        {
            hal::fastAtan32f(src_y.InLine<float>(0),
                             src_x.InLine<float>(0),
                             dst.OutLine<float>(),
                             w,
                             angleInDegrees);
        }
        else if (src_x.meta().depth == CV_64F && src_y.meta().depth == CV_64F)
        {
            hal::fastAtan64f(src_y.InLine<double>(0),
                             src_x.InLine<double>(0),
                             dst.OutLine<double>(),
                             w,
                             angleInDegrees);
        } else GAPI_Assert(false && !"Phase supports 32F/64F input only!");
    }
};

template<typename T, typename Mapper, int chanNum>
struct LinearScratchDesc {
    using alpha_t = typename Mapper::alpha_type;
    using index_t = typename Mapper::index_type;

    alpha_t* alpha;
    alpha_t* clone;
    index_t* mapsx;
    alpha_t* beta;
    index_t* mapsy;
    T*       tmp;

    LinearScratchDesc(int /*inW*/, int /*inH*/, int outW, int outH,  void* data) {
        alpha = reinterpret_cast<alpha_t*>(data);
        clone = reinterpret_cast<alpha_t*>(alpha + outW);
        mapsx = reinterpret_cast<index_t*>(clone + outW*4);
        beta  = reinterpret_cast<alpha_t*>(mapsx + outW);
        mapsy = reinterpret_cast<index_t*>(beta  + outH);
        tmp   = reinterpret_cast<T*>      (mapsy + outH*2);
    }

    static int bufSize(int inW, int /*inH*/, int outW, int outH, int lpi) {
        auto size = outW * sizeof(alpha_t)     +
                    outW * sizeof(alpha_t) * 4 +  // alpha clones
                    outW * sizeof(index_t)     +
                    outH * sizeof(alpha_t)     +
                    outH * sizeof(index_t) * 2 +
                     inW * sizeof(T) * lpi * chanNum;

        return static_cast<int>(size);
    }
};
static inline double invRatio(int inSz, int outSz) {
    return static_cast<double>(outSz) / inSz;
}

static inline double ratio(int inSz, int outSz) {
    return 1 / invRatio(inSz, outSz);
}

template<typename T, typename Mapper, int chanNum = 1>
static inline void initScratchLinear(const cv::GMatDesc& in,
                                     const         Size& outSz,
                                     cv::gapi::fluid::Buffer& scratch,
                                     int  lpi) {
    using alpha_type = typename Mapper::alpha_type;
    static const auto unity = Mapper::unity;

    auto inSz = in.size;
    auto sbufsize = LinearScratchDesc<T, Mapper, chanNum>::bufSize(inSz.width, inSz.height, outSz.width, outSz.height, lpi);

    Size scratch_size{sbufsize, 1};

    cv::GMatDesc desc;
    desc.chan = 1;
    desc.depth = CV_8UC1;
    desc.size = scratch_size;

    cv::gapi::fluid::Buffer buffer(desc);
    scratch = std::move(buffer);

    double hRatio = ratio(in.size.width, outSz.width);
    double vRatio = ratio(in.size.height, outSz.height);

    LinearScratchDesc<T, Mapper, chanNum> scr(inSz.width, inSz.height, outSz.width, outSz.height, scratch.OutLineB());

    auto *alpha = scr.alpha;
    auto *clone = scr.clone;
    auto *index = scr.mapsx;

    for (int x = 0; x < outSz.width; x++) {
        auto map = Mapper::map(hRatio, 0, in.size.width, x);
        auto alpha0 = map.alpha0;
        auto index0 = map.index0;

        // TRICK:
        // Algorithm takes pair of input pixels, sx0'th and sx1'th,
        // and compute result as alpha0*src[sx0] + alpha1*src[sx1].
        // By definition: sx1 == sx0 + 1 either sx1 == sx0, and
        // alpha0 + alpha1 == unity (scaled appropriately).
        // Here we modify formulas for alpha0 and sx1: by assuming
        // that sx1 == sx0 + 1 always, and patching alpha0 so that
        // result remains intact.
        // Note that we need in.size.width >= 2, for both sx0 and
        // sx0+1 were indexing pixels inside the input's width.
        if (map.index1 != map.index0 + 1) {
            GAPI_DbgAssert(map.index1 == map.index0);
            GAPI_DbgAssert(in.size.width >= 2);
            if (map.index0 < in.size.width-1) {
                // sx1=sx0+1 fits inside row,
                // make sure alpha0=unity and alpha1=0,
                // so that result equals src[sx0]*unity
                alpha0 = saturate_cast<alpha_type>(unity);
            } else {
                // shift sx0 to left by 1 pixel,
                // and make sure that alpha0=0 and alpha1==1,
                // so that result equals to src[sx0+1]*unity
                alpha0 = 0;
                index0--;
            }
        }

        alpha[x] = alpha0;
        index[x] = index0;

        for (int l = 0; l < 4; l++) {
            clone[4*x + l] = alpha0;
        }
    }

    auto *beta    = scr.beta;
    auto *index_y = scr.mapsy;

    for (int y = 0; y < outSz.height; y++) {
        auto mapY = Mapper::map(vRatio, 0, in.size.height, y);
        beta[y] = mapY.alpha0;
        index_y[y] = mapY.index0;
        index_y[outSz.height + y] = mapY.index1;
    }
}

template<typename F, typename I>
struct MapperUnit {
    F alpha0, alpha1;
    I index0, index1;
};

inline static uint8_t calc(short alpha0, uint8_t src0, short alpha1, uint8_t src1) {
    constexpr static const int half = 1 << 14;
    return (src0 * alpha0 + src1 * alpha1 + half) >> 15;
}
struct Mapper {
    constexpr static const int ONE = 1 << 15;
    typedef short alpha_type;
    typedef short index_type;
    constexpr static const int unity = ONE;

    typedef MapperUnit<short, short> Unit;

    static inline Unit map(double ratio, int start, int max, int outCoord) {
        float f = static_cast<float>((outCoord + 0.5) * ratio - 0.5);
        int s = cvFloor(f);
        f -= s;

        Unit u;

        u.index0 = static_cast<short>(std::max(s - start, 0));
        u.index1 = static_cast<short>(((f == 0.0) || s + 1 >= max) ? s - start : s - start + 1);

        u.alpha0 = saturate_cast<short>(ONE * (1.0f - f));
        u.alpha1 = saturate_cast<short>(ONE * f);

        return u;
    }
};

template<typename T, class Mapper, int numChan>
static void calcRowLinearC(const cv::gapi::fluid::View  & in,
                           cv::gapi::fluid::Buffer& out,
                           cv::gapi::fluid::Buffer& scratch) {
    using alpha_type = typename Mapper::alpha_type;

    auto  inSz =  in.meta().size;
    auto outSz = out.meta().size;

    auto inY  = in.y();
    int outY = out.y();
    int lpi = out.lpi();

    GAPI_DbgAssert(outY + lpi <= outSz.height);
    GAPI_DbgAssert(lpi <= 4);

    LinearScratchDesc<T, Mapper, numChan> scr(inSz.width, inSz.height, outSz.width, outSz.height, scratch.OutLineB());

    const auto *alpha = scr.alpha;
    const auto *mapsx = scr.mapsx;
    const auto *beta_0 = scr.beta;
    const auto *mapsy = scr.mapsy;

    const auto *beta = beta_0 + outY;
    const T *src0[4];
    const T *src1[4];
    T* dst[4];

    for (int l = 0; l < lpi; l++) {
        auto index0 = mapsy[outY + l] - inY;
        auto index1 = mapsy[outSz.height + outY + l] - inY;
        src0[l] = in.InLine<const T>(index0);
        src1[l] = in.InLine<const T>(index1);
        dst[l] = out.OutLine<T>(l);
    }

#if CV_SSE4_1
    const auto* clone = scr.clone;
    auto* tmp = scr.tmp;

    if (inSz.width >= 16 && outSz.width >= 16)
    {
        sse42::calcRowLinear_8UC_Impl_<numChan>(reinterpret_cast<uint8_t**>(dst),
                                                reinterpret_cast<const uint8_t**>(src0),
                                                reinterpret_cast<const uint8_t**>(src1),
                                                reinterpret_cast<const short*>(alpha),
                                                reinterpret_cast<const short*>(clone),
                                                reinterpret_cast<const short*>(mapsx),
                                                reinterpret_cast<const short*>(beta),
                                                reinterpret_cast<uint8_t*>(tmp),
                                                inSz, outSz, lpi);

        return;
    }
#endif // CV_SSE4_1
    int length = out.length();
    for (int l = 0; l < lpi; l++) {
        constexpr static const auto unity = Mapper::unity;

        auto beta0 =                                   beta[l];
        auto beta1 = saturate_cast<alpha_type>(unity - beta[l]);

        for (int x = 0; x < length; x++) {
            auto alpha0 =                                   alpha[x];
            auto alpha1 = saturate_cast<alpha_type>(unity - alpha[x]);
            auto sx0 = mapsx[x];
            auto sx1 = sx0 + 1;

            for (int c = 0; c < numChan; c++) {
                auto idx0 = numChan*sx0 + c;
                auto idx1 = numChan*sx1 + c;
                T tmp0 = calc(beta0, src0[l][idx0], beta1, src1[l][idx0]);
                T tmp1 = calc(beta0, src0[l][idx1], beta1, src1[l][idx1]);
                dst[l][numChan * x + c] = calc(alpha0, tmp0, alpha1, tmp1);
            }
        }
    }
}

GAPI_FLUID_KERNEL(GFluidResize, cv::gapi::core::GResize, true)
{
    static const int Window = 1;
    static const int LPI = 4;
    static const auto Kind = GFluidKernel::Kind::Resize;

    constexpr static const int INTER_RESIZE_COEF_BITS = 11;
    constexpr static const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
    constexpr static const short ONE = INTER_RESIZE_COEF_SCALE;

   static void initScratch(const cv::GMatDesc& in,
                           cv::Size outSz, double fx, double fy, int /*interp*/,
                           cv::gapi::fluid::Buffer &scratch)
    {
       int outSz_w;
       int outSz_h;
       if (outSz.width == 0 || outSz.height == 0)
       {
           outSz_w = static_cast<int>(round(in.size.width * fx));
           outSz_h = static_cast<int>(round(in.size.height * fy));
       }
       else
       {
           outSz_w = outSz.width;
           outSz_h = outSz.height;
       }
       cv::Size outSize(outSz_w, outSz_h);

       if (in.chan == 3)
       {
           initScratchLinear<uchar, Mapper, 3>(in, outSize, scratch, LPI);
       }
       else if (in.chan == 4)
       {
           initScratchLinear<uchar, Mapper, 4>(in, outSize, scratch, LPI);
       }
    }

    static void resetScratch(cv::gapi::fluid::Buffer& /*scratch*/)
    {}

    static void run(const cv::gapi::fluid::View& in, cv::Size /*sz*/, double /*fx*/, double /*fy*/, int interp,
                    cv::gapi::fluid::Buffer& out,
                    cv::gapi::fluid::Buffer& scratch) {
        const int channels = in.meta().chan;
        GAPI_Assert((channels == 3 || channels == 4) && (interp == cv::INTER_LINEAR));

        if (channels == 3)
        {
            calcRowLinearC<uint8_t, Mapper, 3>(in, out, scratch);
        }
        else if (channels == 4)
        {
            calcRowLinearC<uint8_t, Mapper, 4>(in, out, scratch);
        }
    }
};

GAPI_FLUID_KERNEL(GFluidSqrt, cv::gapi::core::GSqrt, false)
{
    static const int Window = 1;

    static void run(const View &in, Buffer &out)
    {
        const auto w = out.length() * out.meta().chan;
        if (in.meta().depth == CV_32F)
        {
            hal::sqrt32f(in.InLine<float>(0),
                         out.OutLine<float>(0),
                         w);
        }
        else if (in.meta().depth == CV_64F)
        {
            hal::sqrt64f(in.InLine<double>(0),
                         out.OutLine<double>(0),
                         w);
        } else GAPI_Assert(false && !"Sqrt supports 32F/64F input only!");
    }
};

} // namespace fliud
} // namespace gapi
} // namespace cv

cv::gapi::GKernelPackage cv::gapi::core::fluid::kernels()
{
    using namespace cv::gapi::fluid;

    return cv::gapi::kernels
     <       GFluidAdd
            ,GFluidSub
            ,GFluidMul
            ,GFluidDiv
            ,GFluidAbsDiff
            ,GFluidAnd
            ,GFluidOr
            ,GFluidXor
            ,GFluidAndS
            ,GFluidOrS
            ,GFluidXorS
            ,GFluidMin
            ,GFluidMax
            ,GFluidCmpGT
            ,GFluidCmpGE
            ,GFluidCmpLE
            ,GFluidCmpLT
            ,GFluidCmpEQ
            ,GFluidCmpNE
            ,GFluidAddW
            ,GFluidNot
            ,GFluidLUT
            ,GFluidConvertTo
            ,GFluidSplit3
            ,GFluidSplit4
            ,GFluidMerge3
            ,GFluidMerge4
            ,GFluidSelect
            ,GFluidPolarToCart
            ,GFluidCartToPolar
            ,GFluidPhase
            ,GFluidAddC
            ,GFluidSubC
            ,GFluidSubRC
            ,GFluidMulC
            ,GFluidMulCOld
            ,GFluidDivC
            ,GFluidDivRC
            ,GFluidMask
            ,GFluidAbsDiffC
            ,GFluidCmpGTScalar
            ,GFluidCmpGEScalar
            ,GFluidCmpLEScalar
            ,GFluidCmpLTScalar
            ,GFluidCmpEQScalar
            ,GFluidCmpNEScalar
            ,GFluidThreshold
            ,GFluidInRange
            ,GFluidResize
            ,GFluidSqrt
        #if 0
            ,GFluidMean        -- not fluid
            ,GFluidSum         -- not fluid
            ,GFluidNormL1      -- not fluid
            ,GFluidNormL2      -- not fluid
            ,GFluidNormInf     -- not fluid
            ,GFluidIntegral    -- not fluid
            ,GFluidThresholdOT -- not fluid
            ,GFluidResize      -- not fluid (?)
            ,GFluidRemap       -- not fluid
            ,GFluidFlip        -- not fluid
            ,GFluidCrop        -- not fluid
            ,GFluidConcatHor
            ,GFluidConcatVert  -- not fluid
        #endif
        >();
}

#endif // !defined(GAPI_STANDALONE)
