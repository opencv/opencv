// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_HAL_INTRIN_MSA_HPP
#define OPENCV_HAL_INTRIN_MSA_HPP

#include <algorithm>
#include "opencv2/core/utility.hpp"

namespace cv
{

//! @cond IGNORED
CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

#define CV_SIMD128 1

//MSA implements 128-bit wide vector registers shared with the 64-bit wide floating-point unit registers.
//MSA and FPU can not be both present, unless the FPU has 64-bit floating-point registers.
#define CV_SIMD128_64F 1

struct v_uint8x16
{
    typedef uchar lane_type;
    enum { nlanes = 16 };

    v_uint8x16() {}
    explicit v_uint8x16(v16u8 v) : val(v) {}
    v_uint8x16(uchar v0, uchar v1, uchar v2, uchar v3, uchar v4, uchar v5, uchar v6, uchar v7,
               uchar v8, uchar v9, uchar v10, uchar v11, uchar v12, uchar v13, uchar v14, uchar v15)
    {
        uchar v[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15};
        val = msa_ld1q_u8(v);
    }

    uchar get0() const
    {
        return msa_getq_lane_u8(val, 0);
    }

    v16u8 val;
};

struct v_int8x16
{
    typedef schar lane_type;
    enum { nlanes = 16 };

    v_int8x16() {}
    explicit v_int8x16(v16i8 v) : val(v) {}
    v_int8x16(schar v0, schar v1, schar v2, schar v3, schar v4, schar v5, schar v6, schar v7,
               schar v8, schar v9, schar v10, schar v11, schar v12, schar v13, schar v14, schar v15)
    {
        schar v[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15};
        val = msa_ld1q_s8(v);
    }

    schar get0() const
    {
        return msa_getq_lane_s8(val, 0);
    }

    v16i8 val;
};

struct v_uint16x8
{
    typedef ushort lane_type;
    enum { nlanes = 8 };

    v_uint16x8() {}
    explicit v_uint16x8(v8u16 v) : val(v) {}
    v_uint16x8(ushort v0, ushort v1, ushort v2, ushort v3, ushort v4, ushort v5, ushort v6, ushort v7)
    {
        ushort v[] = {v0, v1, v2, v3, v4, v5, v6, v7};
        val = msa_ld1q_u16(v);
    }

    ushort get0() const
    {
        return msa_getq_lane_u16(val, 0);
    }

    v8u16 val;
};

struct v_int16x8
{
    typedef short lane_type;
    enum { nlanes = 8 };

    v_int16x8() {}
    explicit v_int16x8(v8i16 v) : val(v) {}
    v_int16x8(short v0, short v1, short v2, short v3, short v4, short v5, short v6, short v7)
    {
        short v[] = {v0, v1, v2, v3, v4, v5, v6, v7};
        val = msa_ld1q_s16(v);
    }

    short get0() const
    {
        return msa_getq_lane_s16(val, 0);
    }

    v8i16 val;
};

struct v_uint32x4
{
    typedef unsigned int lane_type;
    enum { nlanes = 4 };

    v_uint32x4() {}
    explicit v_uint32x4(v4u32 v) : val(v) {}
    v_uint32x4(unsigned int v0, unsigned int v1, unsigned int v2, unsigned int v3)
    {
        unsigned int v[] = {v0, v1, v2, v3};
        val = msa_ld1q_u32(v);
    }

    unsigned int get0() const
    {
        return msa_getq_lane_u32(val, 0);
    }

    v4u32 val;
};

struct v_int32x4
{
    typedef int lane_type;
    enum { nlanes = 4 };

    v_int32x4() {}
    explicit v_int32x4(v4i32 v) : val(v) {}
    v_int32x4(int v0, int v1, int v2, int v3)
    {
        int v[] = {v0, v1, v2, v3};
        val = msa_ld1q_s32(v);
    }

    int get0() const
    {
        return msa_getq_lane_s32(val, 0);
    }

    v4i32 val;
};

struct v_float32x4
{
    typedef float lane_type;
    enum { nlanes = 4 };

    v_float32x4() {}
    explicit v_float32x4(v4f32 v) : val(v) {}
    v_float32x4(float v0, float v1, float v2, float v3)
    {
        float v[] = {v0, v1, v2, v3};
        val = msa_ld1q_f32(v);
    }

    float get0() const
    {
        return msa_getq_lane_f32(val, 0);
    }

    v4f32 val;
};

struct v_uint64x2
{
    typedef uint64 lane_type;
    enum { nlanes = 2 };

    v_uint64x2() {}
    explicit v_uint64x2(v2u64 v) : val(v) {}
    v_uint64x2(uint64 v0, uint64 v1)
    {
        uint64 v[] = {v0, v1};
        val = msa_ld1q_u64(v);
    }

    uint64 get0() const
    {
        return msa_getq_lane_u64(val, 0);
    }

    v2u64 val;
};

struct v_int64x2
{
    typedef int64 lane_type;
    enum { nlanes = 2 };

    v_int64x2() {}
    explicit v_int64x2(v2i64 v) : val(v) {}
    v_int64x2(int64 v0, int64 v1)
    {
        int64 v[] = {v0, v1};
        val = msa_ld1q_s64(v);
    }

    int64 get0() const
    {
        return msa_getq_lane_s64(val, 0);
    }

    v2i64 val;
};

struct v_float64x2
{
    typedef double lane_type;
    enum { nlanes = 2 };

    v_float64x2() {}
    explicit v_float64x2(v2f64 v) : val(v) {}
    v_float64x2(double v0, double v1)
    {
        double v[] = {v0, v1};
        val = msa_ld1q_f64(v);
    }

    double get0() const
    {
        return msa_getq_lane_f64(val, 0);
    }

    v2f64 val;
};

#define OPENCV_HAL_IMPL_MSA_INIT(_Tpv, _Tp, suffix) \
inline v_##_Tpv v_setzero_##suffix() { return v_##_Tpv(msa_dupq_n_##suffix((_Tp)0)); } \
inline v_##_Tpv v_setall_##suffix(_Tp v) { return v_##_Tpv(msa_dupq_n_##suffix(v)); } \
inline v_uint8x16 v_reinterpret_as_u8(const v_##_Tpv& v) { return v_uint8x16(MSA_TPV_REINTERPRET(v16u8, v.val)); } \
inline v_int8x16 v_reinterpret_as_s8(const v_##_Tpv& v) { return v_int8x16(MSA_TPV_REINTERPRET(v16i8, v.val)); } \
inline v_uint16x8 v_reinterpret_as_u16(const v_##_Tpv& v) { return v_uint16x8(MSA_TPV_REINTERPRET(v8u16, v.val)); } \
inline v_int16x8 v_reinterpret_as_s16(const v_##_Tpv& v) { return v_int16x8(MSA_TPV_REINTERPRET(v8i16, v.val)); } \
inline v_uint32x4 v_reinterpret_as_u32(const v_##_Tpv& v) { return v_uint32x4(MSA_TPV_REINTERPRET(v4u32, v.val)); } \
inline v_int32x4 v_reinterpret_as_s32(const v_##_Tpv& v) { return v_int32x4(MSA_TPV_REINTERPRET(v4i32, v.val)); } \
inline v_uint64x2 v_reinterpret_as_u64(const v_##_Tpv& v) { return v_uint64x2(MSA_TPV_REINTERPRET(v2u64, v.val)); } \
inline v_int64x2 v_reinterpret_as_s64(const v_##_Tpv& v) { return v_int64x2(MSA_TPV_REINTERPRET(v2i64, v.val)); } \
inline v_float32x4 v_reinterpret_as_f32(const v_##_Tpv& v) { return v_float32x4(MSA_TPV_REINTERPRET(v4f32, v.val)); } \
inline v_float64x2 v_reinterpret_as_f64(const v_##_Tpv& v) { return v_float64x2(MSA_TPV_REINTERPRET(v2f64, v.val)); }

OPENCV_HAL_IMPL_MSA_INIT(uint8x16, uchar, u8)
OPENCV_HAL_IMPL_MSA_INIT(int8x16, schar, s8)
OPENCV_HAL_IMPL_MSA_INIT(uint16x8, ushort, u16)
OPENCV_HAL_IMPL_MSA_INIT(int16x8, short, s16)
OPENCV_HAL_IMPL_MSA_INIT(uint32x4, unsigned int, u32)
OPENCV_HAL_IMPL_MSA_INIT(int32x4, int, s32)
OPENCV_HAL_IMPL_MSA_INIT(uint64x2, uint64, u64)
OPENCV_HAL_IMPL_MSA_INIT(int64x2, int64, s64)
OPENCV_HAL_IMPL_MSA_INIT(float32x4, float, f32)
OPENCV_HAL_IMPL_MSA_INIT(float64x2, double, f64)

#define OPENCV_HAL_IMPL_MSA_PACK(_Tpvec, _Tpwvec, pack, mov, rshr) \
inline _Tpvec v_##pack(const _Tpwvec& a, const _Tpwvec& b) \
{ \
    return _Tpvec(mov(a.val, b.val)); \
} \
template<int n> inline \
_Tpvec v_rshr_##pack(const _Tpwvec& a, const _Tpwvec& b) \
{ \
    return _Tpvec(rshr(a.val, b.val, n)); \
}

OPENCV_HAL_IMPL_MSA_PACK(v_uint8x16, v_uint16x8, pack, msa_qpack_u16, msa_qrpackr_u16)
OPENCV_HAL_IMPL_MSA_PACK(v_int8x16, v_int16x8, pack, msa_qpack_s16, msa_qrpackr_s16)
OPENCV_HAL_IMPL_MSA_PACK(v_uint16x8, v_uint32x4, pack, msa_qpack_u32, msa_qrpackr_u32)
OPENCV_HAL_IMPL_MSA_PACK(v_int16x8, v_int32x4, pack, msa_qpack_s32, msa_qrpackr_s32)
OPENCV_HAL_IMPL_MSA_PACK(v_uint32x4, v_uint64x2, pack, msa_pack_u64, msa_rpackr_u64)
OPENCV_HAL_IMPL_MSA_PACK(v_int32x4, v_int64x2, pack, msa_pack_s64, msa_rpackr_s64)
OPENCV_HAL_IMPL_MSA_PACK(v_uint8x16, v_int16x8, pack_u, msa_qpacku_s16, msa_qrpackru_s16)
OPENCV_HAL_IMPL_MSA_PACK(v_uint16x8, v_int32x4, pack_u, msa_qpacku_s32, msa_qrpackru_s32)

#define OPENCV_HAL_IMPL_MSA_PACK_STORE(_Tpvec, _Tp, hreg, suffix, _Tpwvec, pack, mov, rshr) \
inline void v_##pack##_store(_Tp* ptr, const _Tpwvec& a) \
{ \
    hreg a1 = mov(a.val); \
    msa_st1_##suffix(ptr, a1); \
} \
template<int n> inline \
void v_rshr_##pack##_store(_Tp* ptr, const _Tpwvec& a) \
{ \
    hreg a1 = rshr(a.val, n); \
    msa_st1_##suffix(ptr, a1); \
}

OPENCV_HAL_IMPL_MSA_PACK_STORE(v_uint8x16, uchar, v8u8, u8, v_uint16x8, pack, msa_qmovn_u16, msa_qrshrn_n_u16)
OPENCV_HAL_IMPL_MSA_PACK_STORE(v_int8x16, schar, v8i8, s8, v_int16x8, pack, msa_qmovn_s16, msa_qrshrn_n_s16)
OPENCV_HAL_IMPL_MSA_PACK_STORE(v_uint16x8, ushort, v4u16, u16, v_uint32x4, pack, msa_qmovn_u32, msa_qrshrn_n_u32)
OPENCV_HAL_IMPL_MSA_PACK_STORE(v_int16x8, short, v4i16, s16, v_int32x4, pack, msa_qmovn_s32, msa_qrshrn_n_s32)
OPENCV_HAL_IMPL_MSA_PACK_STORE(v_uint32x4, unsigned, v2u32, u32, v_uint64x2, pack, msa_movn_u64, msa_rshrn_n_u64)
OPENCV_HAL_IMPL_MSA_PACK_STORE(v_int32x4, int, v2i32, s32, v_int64x2, pack, msa_movn_s64, msa_rshrn_n_s64)
OPENCV_HAL_IMPL_MSA_PACK_STORE(v_uint8x16, uchar, v8u8, u8, v_int16x8, pack_u, msa_qmovun_s16, msa_qrshrun_n_s16)
OPENCV_HAL_IMPL_MSA_PACK_STORE(v_uint16x8, ushort, v4u16, u16, v_int32x4, pack_u, msa_qmovun_s32, msa_qrshrun_n_s32)

// pack boolean
inline v_uint8x16 v_pack_b(const v_uint16x8& a, const v_uint16x8& b)
{
    return v_uint8x16(msa_pack_u16(a.val, b.val));
}

inline v_uint8x16 v_pack_b(const v_uint32x4& a, const v_uint32x4& b,
                           const v_uint32x4& c, const v_uint32x4& d)
{
    return v_uint8x16(msa_pack_u16(msa_pack_u32(a.val, b.val), msa_pack_u32(c.val, d.val)));
}

inline v_uint8x16 v_pack_b(const v_uint64x2& a, const v_uint64x2& b, const v_uint64x2& c,
                           const v_uint64x2& d, const v_uint64x2& e, const v_uint64x2& f,
                           const v_uint64x2& g, const v_uint64x2& h)
{
    v8u16 abcd = msa_pack_u32(msa_pack_u64(a.val, b.val), msa_pack_u64(c.val, d.val));
    v8u16 efgh = msa_pack_u32(msa_pack_u64(e.val, f.val), msa_pack_u64(g.val, h.val));
    return v_uint8x16(msa_pack_u16(abcd, efgh));
}

inline v_float32x4 v_matmul(const v_float32x4& v, const v_float32x4& m0,
                            const v_float32x4& m1, const v_float32x4& m2,
                            const v_float32x4& m3)
{
    v4f32 v0 = v.val;
    v4f32 res = msa_mulq_lane_f32(m0.val, v0, 0);
    res = msa_mlaq_lane_f32(res, m1.val, v0, 1);
    res = msa_mlaq_lane_f32(res, m2.val, v0, 2);
    res = msa_mlaq_lane_f32(res, m3.val, v0, 3);
    return v_float32x4(res);
}

inline v_float32x4 v_matmuladd(const v_float32x4& v, const v_float32x4& m0,
                               const v_float32x4& m1, const v_float32x4& m2,
                               const v_float32x4& a)
{
    v4f32 v0 = v.val;
    v4f32 res = msa_mulq_lane_f32(m0.val, v0, 0);
    res = msa_mlaq_lane_f32(res, m1.val, v0, 1);
    res = msa_mlaq_lane_f32(res, m2.val, v0, 2);
    res = msa_addq_f32(res, a.val);
    return v_float32x4(res);
}

#define OPENCV_HAL_IMPL_MSA_BIN_OP(bin_op, _Tpvec, intrin) \
inline _Tpvec bin_op (const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(intrin(a.val, b.val)); \
}

OPENCV_HAL_IMPL_MSA_BIN_OP(v_add, v_uint8x16, msa_qaddq_u8)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_sub, v_uint8x16, msa_qsubq_u8)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_add, v_int8x16, msa_qaddq_s8)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_sub, v_int8x16, msa_qsubq_s8)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_add, v_uint16x8, msa_qaddq_u16)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_sub, v_uint16x8, msa_qsubq_u16)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_add, v_int16x8, msa_qaddq_s16)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_sub, v_int16x8, msa_qsubq_s16)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_add, v_int32x4, msa_addq_s32)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_sub, v_int32x4, msa_subq_s32)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_mul, v_int32x4, msa_mulq_s32)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_add, v_uint32x4, msa_addq_u32)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_sub, v_uint32x4, msa_subq_u32)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_mul, v_uint32x4, msa_mulq_u32)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_add, v_float32x4, msa_addq_f32)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_sub, v_float32x4, msa_subq_f32)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_mul, v_float32x4, msa_mulq_f32)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_add, v_int64x2, msa_addq_s64)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_sub, v_int64x2, msa_subq_s64)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_add, v_uint64x2, msa_addq_u64)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_sub, v_uint64x2, msa_subq_u64)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_div, v_float32x4, msa_divq_f32)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_add, v_float64x2, msa_addq_f64)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_sub, v_float64x2, msa_subq_f64)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_mul, v_float64x2, msa_mulq_f64)
OPENCV_HAL_IMPL_MSA_BIN_OP(v_div, v_float64x2, msa_divq_f64)

// saturating multiply 8-bit, 16-bit
#define OPENCV_HAL_IMPL_MSA_MUL_SAT(_Tpvec, _Tpwvec)         \
inline _Tpvec v_mul(const _Tpvec& a, const _Tpvec& b)  \
{                                                            \
    _Tpwvec c, d;                                            \
    v_mul_expand(a, b, c, d);                                \
    return v_pack(c, d);                                     \
}

OPENCV_HAL_IMPL_MSA_MUL_SAT(v_int8x16,  v_int16x8)
OPENCV_HAL_IMPL_MSA_MUL_SAT(v_uint8x16, v_uint16x8)
OPENCV_HAL_IMPL_MSA_MUL_SAT(v_int16x8,  v_int32x4)
OPENCV_HAL_IMPL_MSA_MUL_SAT(v_uint16x8, v_uint32x4)

//  Multiply and expand
inline void v_mul_expand(const v_int8x16& a, const v_int8x16& b,
                         v_int16x8& c, v_int16x8& d)
{
    v16i8 a_lo, a_hi, b_lo, b_hi;

    ILVRL_B2_SB(a.val, msa_dupq_n_s8(0), a_lo, a_hi);
    ILVRL_B2_SB(b.val, msa_dupq_n_s8(0), b_lo, b_hi);
    c.val = msa_mulq_s16(msa_paddlq_s8(a_lo), msa_paddlq_s8(b_lo));
    d.val = msa_mulq_s16(msa_paddlq_s8(a_hi), msa_paddlq_s8(b_hi));
}

inline void v_mul_expand(const v_uint8x16& a, const v_uint8x16& b,
                         v_uint16x8& c, v_uint16x8& d)
{
    v16u8 a_lo, a_hi, b_lo, b_hi;

    ILVRL_B2_UB(a.val, msa_dupq_n_u8(0), a_lo, a_hi);
    ILVRL_B2_UB(b.val, msa_dupq_n_u8(0), b_lo, b_hi);
    c.val = msa_mulq_u16(msa_paddlq_u8(a_lo), msa_paddlq_u8(b_lo));
    d.val = msa_mulq_u16(msa_paddlq_u8(a_hi), msa_paddlq_u8(b_hi));
}

inline void v_mul_expand(const v_int16x8& a, const v_int16x8& b,
                         v_int32x4& c, v_int32x4& d)
{
    v8i16 a_lo, a_hi, b_lo, b_hi;

    ILVRL_H2_SH(a.val, msa_dupq_n_s16(0), a_lo, a_hi);
    ILVRL_H2_SH(b.val, msa_dupq_n_s16(0), b_lo, b_hi);
    c.val = msa_mulq_s32(msa_paddlq_s16(a_lo), msa_paddlq_s16(b_lo));
    d.val = msa_mulq_s32(msa_paddlq_s16(a_hi), msa_paddlq_s16(b_hi));
}

inline void v_mul_expand(const v_uint16x8& a, const v_uint16x8& b,
                         v_uint32x4& c, v_uint32x4& d)
{
    v8u16 a_lo, a_hi, b_lo, b_hi;

    ILVRL_H2_UH(a.val, msa_dupq_n_u16(0), a_lo, a_hi);
    ILVRL_H2_UH(b.val, msa_dupq_n_u16(0), b_lo, b_hi);
    c.val = msa_mulq_u32(msa_paddlq_u16(a_lo), msa_paddlq_u16(b_lo));
    d.val = msa_mulq_u32(msa_paddlq_u16(a_hi), msa_paddlq_u16(b_hi));
}

inline void v_mul_expand(const v_uint32x4& a, const v_uint32x4& b,
                         v_uint64x2& c, v_uint64x2& d)
{
    v4u32 a_lo, a_hi, b_lo, b_hi;

    ILVRL_W2_UW(a.val, msa_dupq_n_u32(0), a_lo, a_hi);
    ILVRL_W2_UW(b.val, msa_dupq_n_u32(0), b_lo, b_hi);
    c.val = msa_mulq_u64(msa_paddlq_u32(a_lo), msa_paddlq_u32(b_lo));
    d.val = msa_mulq_u64(msa_paddlq_u32(a_hi), msa_paddlq_u32(b_hi));
}

inline v_int16x8 v_mul_hi(const v_int16x8& a, const v_int16x8& b)
{
    v8i16 a_lo, a_hi, b_lo, b_hi;

    ILVRL_H2_SH(a.val, msa_dupq_n_s16(0), a_lo, a_hi);
    ILVRL_H2_SH(b.val, msa_dupq_n_s16(0), b_lo, b_hi);

    return v_int16x8(msa_packr_s32(msa_mulq_s32(msa_paddlq_s16(a_lo), msa_paddlq_s16(b_lo)),
                                   msa_mulq_s32(msa_paddlq_s16(a_hi), msa_paddlq_s16(b_hi)), 16));
}

inline v_uint16x8 v_mul_hi(const v_uint16x8& a, const v_uint16x8& b)
{
    v8u16 a_lo, a_hi, b_lo, b_hi;

    ILVRL_H2_UH(a.val, msa_dupq_n_u16(0), a_lo, a_hi);
    ILVRL_H2_UH(b.val, msa_dupq_n_u16(0), b_lo, b_hi);

    return v_uint16x8(msa_packr_u32(msa_mulq_u32(msa_paddlq_u16(a_lo), msa_paddlq_u16(b_lo)),
                                    msa_mulq_u32(msa_paddlq_u16(a_hi), msa_paddlq_u16(b_hi)), 16));
}

//////// Dot Product ////////

// 16 >> 32
inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b)
{ return v_int32x4(msa_dotp_s_w(a.val, b.val)); }
inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{ return v_int32x4(msa_dpadd_s_w(c.val , a.val, b.val)); }

// 32 >> 64
inline v_int64x2 v_dotprod(const v_int32x4& a, const v_int32x4& b)
{ return v_int64x2(msa_dotp_s_d(a.val, b.val)); }
inline v_int64x2 v_dotprod(const v_int32x4& a, const v_int32x4& b, const v_int64x2& c)
{ return v_int64x2(msa_dpadd_s_d(c.val , a.val, b.val)); }

// 8 >> 32
inline v_uint32x4 v_dotprod_expand(const v_uint8x16& a, const v_uint8x16& b)
{
    v8u16 even_a = msa_shrq_n_u16(msa_shlq_n_u16(MSA_TPV_REINTERPRET(v8u16, a.val), 8), 8);
    v8u16 odd_a  = msa_shrq_n_u16(MSA_TPV_REINTERPRET(v8u16, a.val), 8);
    v8u16 even_b = msa_shrq_n_u16(msa_shlq_n_u16(MSA_TPV_REINTERPRET(v8u16, b.val), 8), 8);
    v8u16 odd_b  = msa_shrq_n_u16(MSA_TPV_REINTERPRET(v8u16, b.val), 8);
    v4u32 prod   = msa_dotp_u_w(even_a, even_b);
    return v_uint32x4(msa_dpadd_u_w(prod, odd_a, odd_b));
}
inline v_uint32x4 v_dotprod_expand(const v_uint8x16& a, const v_uint8x16& b, const v_uint32x4& c)
{
    v8u16 even_a = msa_shrq_n_u16(msa_shlq_n_u16(MSA_TPV_REINTERPRET(v8u16, a.val), 8), 8);
    v8u16 odd_a  = msa_shrq_n_u16(MSA_TPV_REINTERPRET(v8u16, a.val), 8);
    v8u16 even_b = msa_shrq_n_u16(msa_shlq_n_u16(MSA_TPV_REINTERPRET(v8u16, b.val), 8), 8);
    v8u16 odd_b  = msa_shrq_n_u16(MSA_TPV_REINTERPRET(v8u16, b.val), 8);
    v4u32 prod   = msa_dpadd_u_w(c.val, even_a, even_b);
    return v_uint32x4(msa_dpadd_u_w(prod, odd_a, odd_b));
}

inline v_int32x4 v_dotprod_expand(const v_int8x16& a, const v_int8x16& b)
{
    v8i16 prod = msa_dotp_s_h(a.val, b.val);
    return v_int32x4(msa_hadd_s32(prod, prod));
}
inline v_int32x4 v_dotprod_expand(const v_int8x16& a, const v_int8x16& b,
                                  const v_int32x4& c)
{ return v_dotprod_expand(a, b) + c; }

// 16 >> 64
inline v_uint64x2 v_dotprod_expand(const v_uint16x8& a, const v_uint16x8& b)
{
    v4u32 even_a = msa_shrq_n_u32(msa_shlq_n_u32(MSA_TPV_REINTERPRET(v4u32, a.val), 16), 16);
    v4u32 odd_a  = msa_shrq_n_u32(MSA_TPV_REINTERPRET(v4u32, a.val), 16);
    v4u32 even_b = msa_shrq_n_u32(msa_shlq_n_u32(MSA_TPV_REINTERPRET(v4u32, b.val), 16), 16);
    v4u32 odd_b  = msa_shrq_n_u32(MSA_TPV_REINTERPRET(v4u32, b.val), 16);
    v2u64 prod   = msa_dotp_u_d(even_a, even_b);
    return v_uint64x2(msa_dpadd_u_d(prod, odd_a, odd_b));
}
inline v_uint64x2 v_dotprod_expand(const v_uint16x8& a, const v_uint16x8& b,
                                   const v_uint64x2& c)
{
    v4u32 even_a = msa_shrq_n_u32(msa_shlq_n_u32(MSA_TPV_REINTERPRET(v4u32, a.val), 16), 16);
    v4u32 odd_a  = msa_shrq_n_u32(MSA_TPV_REINTERPRET(v4u32, a.val), 16);
    v4u32 even_b = msa_shrq_n_u32(msa_shlq_n_u32(MSA_TPV_REINTERPRET(v4u32, b.val), 16), 16);
    v4u32 odd_b  = msa_shrq_n_u32(MSA_TPV_REINTERPRET(v4u32, b.val), 16);
    v2u64 prod   = msa_dpadd_u_d(c.val, even_a, even_b);
    return v_uint64x2(msa_dpadd_u_d(prod, odd_a, odd_b));
}

inline v_int64x2 v_dotprod_expand(const v_int16x8& a, const v_int16x8& b)
{
    v4i32 prod = msa_dotp_s_w(a.val, b.val);
    return v_int64x2(msa_hadd_s64(prod, prod));
}
inline v_int64x2 v_dotprod_expand(const v_int16x8& a, const v_int16x8& b, const v_int64x2& c)
{ return v_add(v_dotprod_expand(a, b), c); }

// 32 >> 64f
inline v_float64x2 v_dotprod_expand(const v_int32x4& a, const v_int32x4& b)
{ return v_cvt_f64(v_dotprod(a, b)); }
inline v_float64x2 v_dotprod_expand(const v_int32x4& a, const v_int32x4& b, const v_float64x2& c)
{ return v_add(v_dotprod_expand(a, b), c); }


//////// Fast Dot Product ////////

// 16 >> 32
inline v_int32x4 v_dotprod_fast(const v_int16x8& a, const v_int16x8& b)
{ return v_dotprod(a, b); }
inline v_int32x4 v_dotprod_fast(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{ return v_dotprod(a, b, c); }

// 32 >> 64
inline v_int64x2 v_dotprod_fast(const v_int32x4& a, const v_int32x4& b)
{ return v_dotprod(a, b); }
inline v_int64x2 v_dotprod_fast(const v_int32x4& a, const v_int32x4& b, const v_int64x2& c)
{ return v_dotprod(a, b, c); }

// 8 >> 32
inline v_uint32x4 v_dotprod_expand_fast(const v_uint8x16& a, const v_uint8x16& b)
{ return v_dotprod_expand(a, b); }
inline v_uint32x4 v_dotprod_expand_fast(const v_uint8x16& a, const v_uint8x16& b, const v_uint32x4& c)
{ return v_dotprod_expand(a, b, c); }
inline v_int32x4 v_dotprod_expand_fast(const v_int8x16& a, const v_int8x16& b)
{ return v_dotprod_expand(a, b); }
inline v_int32x4 v_dotprod_expand_fast(const v_int8x16& a, const v_int8x16& b, const v_int32x4& c)
{ return v_dotprod_expand(a, b, c); }

// 16 >> 64
inline v_uint64x2 v_dotprod_expand_fast(const v_uint16x8& a, const v_uint16x8& b)
{ return v_dotprod_expand(a, b); }
inline v_uint64x2 v_dotprod_expand_fast(const v_uint16x8& a, const v_uint16x8& b, const v_uint64x2& c)
{ return v_dotprod_expand(a, b, c); }
inline v_int64x2 v_dotprod_expand_fast(const v_int16x8& a, const v_int16x8& b)
{ return v_dotprod_expand(a, b); }
inline v_int64x2 v_dotprod_expand_fast(const v_int16x8& a, const v_int16x8& b, const v_int64x2& c)
{ return v_dotprod_expand(a, b, c); }

// 32 >> 64f
inline v_float64x2 v_dotprod_expand_fast(const v_int32x4& a, const v_int32x4& b)
{ return v_dotprod_expand(a, b); }
inline v_float64x2 v_dotprod_expand_fast(const v_int32x4& a, const v_int32x4& b, const v_float64x2& c)
{ return v_dotprod_expand(a, b, c); }

#define OPENCV_HAL_IMPL_MSA_LOGIC_OP(_Tpvec, _Tpv, suffix) \
OPENCV_HAL_IMPL_MSA_BIN_OP(v_and, _Tpvec, msa_andq_##suffix)   \
OPENCV_HAL_IMPL_MSA_BIN_OP(v_or, _Tpvec, msa_orrq_##suffix)   \
OPENCV_HAL_IMPL_MSA_BIN_OP(v_xor, _Tpvec, msa_eorq_##suffix)   \
inline _Tpvec v_not (const _Tpvec& a) \
{ \
    return _Tpvec(MSA_TPV_REINTERPRET(_Tpv, msa_mvnq_u8(MSA_TPV_REINTERPRET(v16u8, a.val)))); \
}

OPENCV_HAL_IMPL_MSA_LOGIC_OP(v_uint8x16, v16u8, u8)
OPENCV_HAL_IMPL_MSA_LOGIC_OP(v_int8x16, v16i8, s8)
OPENCV_HAL_IMPL_MSA_LOGIC_OP(v_uint16x8, v8u16, u16)
OPENCV_HAL_IMPL_MSA_LOGIC_OP(v_int16x8, v8i16, s16)
OPENCV_HAL_IMPL_MSA_LOGIC_OP(v_uint32x4, v4u32, u32)
OPENCV_HAL_IMPL_MSA_LOGIC_OP(v_int32x4, v4i32, s32)
OPENCV_HAL_IMPL_MSA_LOGIC_OP(v_uint64x2, v2u64, u64)
OPENCV_HAL_IMPL_MSA_LOGIC_OP(v_int64x2, v2i64, s64)

#define OPENCV_HAL_IMPL_MSA_FLT_BIT_OP(bin_op, intrin) \
inline v_float32x4 bin_op (const v_float32x4& a, const v_float32x4& b) \
{ \
    return v_float32x4(MSA_TPV_REINTERPRET(v4f32, intrin(MSA_TPV_REINTERPRET(v4i32, a.val), MSA_TPV_REINTERPRET(v4i32, b.val)))); \
}

OPENCV_HAL_IMPL_MSA_FLT_BIT_OP(v_and, msa_andq_s32)
OPENCV_HAL_IMPL_MSA_FLT_BIT_OP(v_or, msa_orrq_s32)
OPENCV_HAL_IMPL_MSA_FLT_BIT_OP(v_xor, msa_eorq_s32)

inline v_float32x4 v_not (const v_float32x4& a)
{
    return v_float32x4(MSA_TPV_REINTERPRET(v4f32, msa_mvnq_s32(MSA_TPV_REINTERPRET(v4i32, a.val))));
}

/* v_abs */
#define OPENCV_HAL_IMPL_MSA_ABS(_Tpuvec, _Tpsvec, usuffix, ssuffix) \
inline _Tpuvec v_abs(const _Tpsvec& a) \
{ \
    return v_reinterpret_as_##usuffix(_Tpsvec(msa_absq_##ssuffix(a.val))); \
}

OPENCV_HAL_IMPL_MSA_ABS(v_uint8x16, v_int8x16, u8, s8)
OPENCV_HAL_IMPL_MSA_ABS(v_uint16x8, v_int16x8, u16, s16)
OPENCV_HAL_IMPL_MSA_ABS(v_uint32x4, v_int32x4, u32, s32)

/* v_abs(float), v_sqrt, v_invsqrt */
#define OPENCV_HAL_IMPL_MSA_BASIC_FUNC(_Tpvec, func, intrin) \
inline _Tpvec func(const _Tpvec& a) \
{ \
    return _Tpvec(intrin(a.val)); \
}

OPENCV_HAL_IMPL_MSA_BASIC_FUNC(v_float32x4, v_abs, msa_absq_f32)
OPENCV_HAL_IMPL_MSA_BASIC_FUNC(v_float64x2, v_abs, msa_absq_f64)
OPENCV_HAL_IMPL_MSA_BASIC_FUNC(v_float32x4, v_sqrt, msa_sqrtq_f32)
OPENCV_HAL_IMPL_MSA_BASIC_FUNC(v_float32x4, v_invsqrt, msa_rsqrtq_f32)
OPENCV_HAL_IMPL_MSA_BASIC_FUNC(v_float64x2, v_sqrt, msa_sqrtq_f64)
OPENCV_HAL_IMPL_MSA_BASIC_FUNC(v_float64x2, v_invsqrt, msa_rsqrtq_f64)

#define OPENCV_HAL_IMPL_MSA_DBL_BIT_OP(bin_op, intrin) \
inline v_float64x2 bin_op (const v_float64x2& a, const v_float64x2& b) \
{ \
    return v_float64x2(MSA_TPV_REINTERPRET(v2f64, intrin(MSA_TPV_REINTERPRET(v2i64, a.val), MSA_TPV_REINTERPRET(v2i64, b.val)))); \
}

OPENCV_HAL_IMPL_MSA_DBL_BIT_OP(v_and, msa_andq_s64)
OPENCV_HAL_IMPL_MSA_DBL_BIT_OP(v_or, msa_orrq_s64)
OPENCV_HAL_IMPL_MSA_DBL_BIT_OP(v_xor, msa_eorq_s64)

inline v_float64x2 v_not (const v_float64x2& a)
{
    return v_float64x2(MSA_TPV_REINTERPRET(v2f64, msa_mvnq_s32(MSA_TPV_REINTERPRET(v4i32, a.val))));
}

// TODO: exp, log, sin, cos

#define OPENCV_HAL_IMPL_MSA_BIN_FUNC(_Tpvec, func, intrin) \
inline _Tpvec func(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(intrin(a.val, b.val)); \
}

OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_uint8x16, v_min, msa_minq_u8)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_uint8x16, v_max, msa_maxq_u8)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_int8x16, v_min, msa_minq_s8)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_int8x16, v_max, msa_maxq_s8)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_uint16x8, v_min, msa_minq_u16)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_uint16x8, v_max, msa_maxq_u16)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_int16x8, v_min, msa_minq_s16)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_int16x8, v_max, msa_maxq_s16)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_uint32x4, v_min, msa_minq_u32)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_uint32x4, v_max, msa_maxq_u32)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_int32x4, v_min, msa_minq_s32)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_int32x4, v_max, msa_maxq_s32)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_float32x4, v_min, msa_minq_f32)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_float32x4, v_max, msa_maxq_f32)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_float64x2, v_min, msa_minq_f64)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_float64x2, v_max, msa_maxq_f64)

#define OPENCV_HAL_IMPL_MSA_INT_CMP_OP(_Tpvec, _Tpv, suffix, not_suffix) \
inline _Tpvec v_eq (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(MSA_TPV_REINTERPRET(_Tpv, msa_ceqq_##suffix(a.val, b.val))); } \
inline _Tpvec v_ne (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(MSA_TPV_REINTERPRET(_Tpv, msa_mvnq_##not_suffix(msa_ceqq_##suffix(a.val, b.val)))); } \
inline _Tpvec v_lt (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(MSA_TPV_REINTERPRET(_Tpv, msa_cltq_##suffix(a.val, b.val))); } \
inline _Tpvec v_gt (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(MSA_TPV_REINTERPRET(_Tpv, msa_cgtq_##suffix(a.val, b.val))); } \
inline _Tpvec v_le (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(MSA_TPV_REINTERPRET(_Tpv, msa_cleq_##suffix(a.val, b.val))); } \
inline _Tpvec v_ge (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(MSA_TPV_REINTERPRET(_Tpv, msa_cgeq_##suffix(a.val, b.val))); }

OPENCV_HAL_IMPL_MSA_INT_CMP_OP(v_uint8x16, v16u8, u8, u8)
OPENCV_HAL_IMPL_MSA_INT_CMP_OP(v_int8x16, v16i8, s8, u8)
OPENCV_HAL_IMPL_MSA_INT_CMP_OP(v_uint16x8, v8u16, u16, u16)
OPENCV_HAL_IMPL_MSA_INT_CMP_OP(v_int16x8, v8i16, s16, u16)
OPENCV_HAL_IMPL_MSA_INT_CMP_OP(v_uint32x4, v4u32, u32, u32)
OPENCV_HAL_IMPL_MSA_INT_CMP_OP(v_int32x4, v4i32, s32, u32)
OPENCV_HAL_IMPL_MSA_INT_CMP_OP(v_float32x4, v4f32, f32, u32)
OPENCV_HAL_IMPL_MSA_INT_CMP_OP(v_uint64x2, v2u64, u64, u64)
OPENCV_HAL_IMPL_MSA_INT_CMP_OP(v_int64x2, v2i64, s64, u64)
OPENCV_HAL_IMPL_MSA_INT_CMP_OP(v_float64x2, v2f64, f64, u64)

inline v_float32x4 v_not_nan(const v_float32x4& a)
{ return v_float32x4(MSA_TPV_REINTERPRET(v4f32, msa_ceqq_f32(a.val, a.val))); }
inline v_float64x2 v_not_nan(const v_float64x2& a)
{ return v_float64x2(MSA_TPV_REINTERPRET(v2f64, msa_ceqq_f64(a.val, a.val))); }

OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_uint8x16, v_add_wrap, msa_addq_u8)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_int8x16, v_add_wrap, msa_addq_s8)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_uint16x8, v_add_wrap, msa_addq_u16)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_int16x8, v_add_wrap, msa_addq_s16)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_uint8x16, v_sub_wrap, msa_subq_u8)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_int8x16, v_sub_wrap, msa_subq_s8)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_uint16x8, v_sub_wrap, msa_subq_u16)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_int16x8, v_sub_wrap, msa_subq_s16)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_uint8x16, v_mul_wrap, msa_mulq_u8)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_int8x16, v_mul_wrap, msa_mulq_s8)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_uint16x8, v_mul_wrap, msa_mulq_u16)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_int16x8, v_mul_wrap, msa_mulq_s16)

OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_uint8x16, v_absdiff, msa_abdq_u8)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_uint16x8, v_absdiff, msa_abdq_u16)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_uint32x4, v_absdiff, msa_abdq_u32)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_float32x4, v_absdiff, msa_abdq_f32)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_float64x2, v_absdiff, msa_abdq_f64)

/** Saturating absolute difference **/
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_int8x16, v_absdiffs, msa_qabdq_s8)
OPENCV_HAL_IMPL_MSA_BIN_FUNC(v_int16x8, v_absdiffs, msa_qabdq_s16)

#define OPENCV_HAL_IMPL_MSA_BIN_FUNC2(_Tpvec, _Tpvec2, _Tpv, func, intrin) \
inline _Tpvec2 func(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec2(MSA_TPV_REINTERPRET(_Tpv, intrin(a.val, b.val))); \
}

OPENCV_HAL_IMPL_MSA_BIN_FUNC2(v_int8x16, v_uint8x16, v16u8, v_absdiff, msa_abdq_s8)
OPENCV_HAL_IMPL_MSA_BIN_FUNC2(v_int16x8, v_uint16x8, v8u16, v_absdiff, msa_abdq_s16)
OPENCV_HAL_IMPL_MSA_BIN_FUNC2(v_int32x4, v_uint32x4, v4u32, v_absdiff, msa_abdq_s32)

/* v_magnitude, v_sqr_magnitude, v_fma, v_muladd */
inline v_float32x4 v_magnitude(const v_float32x4& a, const v_float32x4& b)
{
    v_float32x4 x(msa_mlaq_f32(msa_mulq_f32(a.val, a.val), b.val, b.val));
    return v_sqrt(x);
}

inline v_float32x4 v_sqr_magnitude(const v_float32x4& a, const v_float32x4& b)
{
    return v_float32x4(msa_mlaq_f32(msa_mulq_f32(a.val, a.val), b.val, b.val));
}

inline v_float32x4 v_fma(const v_float32x4& a, const v_float32x4& b, const v_float32x4& c)
{
    return v_float32x4(msa_mlaq_f32(c.val, a.val, b.val));
}

inline v_int32x4 v_fma(const v_int32x4& a, const v_int32x4& b, const v_int32x4& c)
{
    return v_int32x4(msa_mlaq_s32(c.val, a.val, b.val));
}

inline v_float32x4 v_muladd(const v_float32x4& a, const v_float32x4& b, const v_float32x4& c)
{
    return v_fma(a, b, c);
}

inline v_int32x4 v_muladd(const v_int32x4& a, const v_int32x4& b, const v_int32x4& c)
{
    return v_fma(a, b, c);
}

inline v_float64x2 v_magnitude(const v_float64x2& a, const v_float64x2& b)
{
    v_float64x2 x(msa_mlaq_f64(msa_mulq_f64(a.val, a.val), b.val, b.val));
    return v_sqrt(x);
}

inline v_float64x2 v_sqr_magnitude(const v_float64x2& a, const v_float64x2& b)
{
    return v_float64x2(msa_mlaq_f64(msa_mulq_f64(a.val, a.val), b.val, b.val));
}

inline v_float64x2 v_fma(const v_float64x2& a, const v_float64x2& b, const v_float64x2& c)
{
    return v_float64x2(msa_mlaq_f64(c.val, a.val, b.val));
}

inline v_float64x2 v_muladd(const v_float64x2& a, const v_float64x2& b, const v_float64x2& c)
{
    return v_fma(a, b, c);
}

// trade efficiency for convenience
#define OPENCV_HAL_IMPL_MSA_SHIFT_OP(_Tpvec, suffix, _Tps, ssuffix) \
inline _Tpvec v_shl (const _Tpvec& a, int n) \
{ return _Tpvec(msa_shlq_##suffix(a.val, msa_dupq_n_##ssuffix((_Tps)n))); } \
inline _Tpvec v_shr (const _Tpvec& a, int n) \
{ return _Tpvec(msa_shrq_##suffix(a.val, msa_dupq_n_##ssuffix((_Tps)n))); } \
template<int n> inline _Tpvec v_shl(const _Tpvec& a) \
{ return _Tpvec(msa_shlq_n_##suffix(a.val, n)); } \
template<int n> inline _Tpvec v_shr(const _Tpvec& a) \
{ return _Tpvec(msa_shrq_n_##suffix(a.val, n)); } \
template<int n> inline _Tpvec v_rshr(const _Tpvec& a) \
{ return _Tpvec(msa_rshrq_n_##suffix(a.val, n)); }

OPENCV_HAL_IMPL_MSA_SHIFT_OP(v_uint8x16, u8, schar, s8)
OPENCV_HAL_IMPL_MSA_SHIFT_OP(v_int8x16, s8, schar, s8)
OPENCV_HAL_IMPL_MSA_SHIFT_OP(v_uint16x8, u16, short, s16)
OPENCV_HAL_IMPL_MSA_SHIFT_OP(v_int16x8, s16, short, s16)
OPENCV_HAL_IMPL_MSA_SHIFT_OP(v_uint32x4, u32, int, s32)
OPENCV_HAL_IMPL_MSA_SHIFT_OP(v_int32x4, s32, int, s32)
OPENCV_HAL_IMPL_MSA_SHIFT_OP(v_uint64x2, u64, int64, s64)
OPENCV_HAL_IMPL_MSA_SHIFT_OP(v_int64x2, s64, int64, s64)

/* v_rotate_right, v_rotate_left */
#define OPENCV_HAL_IMPL_MSA_ROTATE_OP(_Tpvec, _Tpv, _Tpvs, suffix) \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a) \
{ \
    return _Tpvec(MSA_TPV_REINTERPRET(_Tpv, msa_extq_##suffix(MSA_TPV_REINTERPRET(_Tpvs, a.val), msa_dupq_n_##suffix(0), n))); \
} \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a) \
{ \
    return _Tpvec(MSA_TPV_REINTERPRET(_Tpv, msa_extq_##suffix(msa_dupq_n_##suffix(0), MSA_TPV_REINTERPRET(_Tpvs, a.val), _Tpvec::nlanes - n))); \
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a) \
{ \
    return a; \
} \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(MSA_TPV_REINTERPRET(_Tpv, msa_extq_##suffix(MSA_TPV_REINTERPRET(_Tpvs, a.val), MSA_TPV_REINTERPRET(_Tpvs, b.val), n))); \
} \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(MSA_TPV_REINTERPRET(_Tpv, msa_extq_##suffix(MSA_TPV_REINTERPRET(_Tpvs, b.val), MSA_TPV_REINTERPRET(_Tpvs, a.val), _Tpvec::nlanes - n))); \
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a, const _Tpvec& b) \
{ \
    CV_UNUSED(b); \
    return a; \
}

OPENCV_HAL_IMPL_MSA_ROTATE_OP(v_uint8x16, v16u8, v16i8, s8)
OPENCV_HAL_IMPL_MSA_ROTATE_OP(v_int8x16, v16i8, v16i8, s8)
OPENCV_HAL_IMPL_MSA_ROTATE_OP(v_uint16x8, v8u16, v8i16, s16)
OPENCV_HAL_IMPL_MSA_ROTATE_OP(v_int16x8, v8i16, v8i16, s16)
OPENCV_HAL_IMPL_MSA_ROTATE_OP(v_uint32x4, v4u32, v4i32, s32)
OPENCV_HAL_IMPL_MSA_ROTATE_OP(v_int32x4, v4i32, v4i32, s32)
OPENCV_HAL_IMPL_MSA_ROTATE_OP(v_float32x4, v4f32, v4i32, s32)
OPENCV_HAL_IMPL_MSA_ROTATE_OP(v_uint64x2, v2u64, v2i64, s64)
OPENCV_HAL_IMPL_MSA_ROTATE_OP(v_int64x2, v2i64, v2i64, s64)
OPENCV_HAL_IMPL_MSA_ROTATE_OP(v_float64x2, v2f64, v2i64, s64)

#define OPENCV_HAL_IMPL_MSA_LOADSTORE_OP(_Tpvec, _Tp, suffix) \
inline _Tpvec v_load(const _Tp* ptr) \
{ return _Tpvec(msa_ld1q_##suffix(ptr)); } \
inline _Tpvec v_load_aligned(const _Tp* ptr) \
{ return _Tpvec(msa_ld1q_##suffix(ptr)); } \
inline _Tpvec v_load_low(const _Tp* ptr) \
{ return _Tpvec(msa_combine_##suffix(msa_ld1_##suffix(ptr), msa_dup_n_##suffix((_Tp)0))); } \
inline _Tpvec v_load_halves(const _Tp* ptr0, const _Tp* ptr1) \
{ return _Tpvec(msa_combine_##suffix(msa_ld1_##suffix(ptr0), msa_ld1_##suffix(ptr1))); } \
inline void v_store(_Tp* ptr, const _Tpvec& a) \
{ msa_st1q_##suffix(ptr, a.val); } \
inline void v_store_aligned(_Tp* ptr, const _Tpvec& a) \
{ msa_st1q_##suffix(ptr, a.val); } \
inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a) \
{ msa_st1q_##suffix(ptr, a.val); } \
inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode /*mode*/) \
{ msa_st1q_##suffix(ptr, a.val); } \
inline void v_store_low(_Tp* ptr, const _Tpvec& a) \
{ \
    int n  = _Tpvec::nlanes; \
    for( int i = 0; i < (n/2); i++ ) \
        ptr[i] = a.val[i]; \
} \
inline void v_store_high(_Tp* ptr, const _Tpvec& a) \
{ \
    int n  = _Tpvec::nlanes; \
    for( int i = 0; i < (n/2); i++ ) \
        ptr[i] = a.val[i+(n/2)]; \
}

OPENCV_HAL_IMPL_MSA_LOADSTORE_OP(v_uint8x16, uchar, u8)
OPENCV_HAL_IMPL_MSA_LOADSTORE_OP(v_int8x16, schar, s8)
OPENCV_HAL_IMPL_MSA_LOADSTORE_OP(v_uint16x8, ushort, u16)
OPENCV_HAL_IMPL_MSA_LOADSTORE_OP(v_int16x8, short, s16)
OPENCV_HAL_IMPL_MSA_LOADSTORE_OP(v_uint32x4, unsigned, u32)
OPENCV_HAL_IMPL_MSA_LOADSTORE_OP(v_int32x4, int, s32)
OPENCV_HAL_IMPL_MSA_LOADSTORE_OP(v_uint64x2, uint64, u64)
OPENCV_HAL_IMPL_MSA_LOADSTORE_OP(v_int64x2, int64, s64)
OPENCV_HAL_IMPL_MSA_LOADSTORE_OP(v_float32x4, float, f32)
OPENCV_HAL_IMPL_MSA_LOADSTORE_OP(v_float64x2, double, f64)


/** Reverse **/
inline v_uint8x16 v_reverse(const v_uint8x16 &a)
{
    v_uint8x16 c = v_uint8x16((v16u8)__builtin_msa_vshf_b((v16i8)((v2i64){0x08090A0B0C0D0E0F, 0x0001020304050607}), msa_dupq_n_s8(0), (v16i8)a.val));
    return c;
}

inline v_int8x16 v_reverse(const v_int8x16 &a)
{ return v_reinterpret_as_s8(v_reverse(v_reinterpret_as_u8(a))); }

inline v_uint16x8 v_reverse(const v_uint16x8 &a)
{
    v_uint16x8 c = v_uint16x8((v8u16)__builtin_msa_vshf_h((v8i16)((v2i64){0x0004000500060007, 0x0000000100020003}), msa_dupq_n_s16(0), (v8i16)a.val));
    return c;
}

inline v_int16x8 v_reverse(const v_int16x8 &a)
{ return v_reinterpret_as_s16(v_reverse(v_reinterpret_as_u16(a))); }

inline v_uint32x4 v_reverse(const v_uint32x4 &a)
{
    v_uint32x4 c;
    c.val[0] = a.val[3];
    c.val[1] = a.val[2];
    c.val[2] = a.val[1];
    c.val[3] = a.val[0];
    return c;
}

inline v_int32x4 v_reverse(const v_int32x4 &a)
{ return v_reinterpret_as_s32(v_reverse(v_reinterpret_as_u32(a))); }

inline v_float32x4 v_reverse(const v_float32x4 &a)
{ return v_reinterpret_as_f32(v_reverse(v_reinterpret_as_u32(a))); }

inline v_uint64x2 v_reverse(const v_uint64x2 &a)
{
    v_uint64x2 c;
    c.val[0] = a.val[1];
    c.val[1] = a.val[0];
    return c;
}

inline v_int64x2 v_reverse(const v_int64x2 &a)
{ return v_reinterpret_as_s64(v_reverse(v_reinterpret_as_u64(a))); }

inline v_float64x2 v_reverse(const v_float64x2 &a)
{ return v_reinterpret_as_f64(v_reverse(v_reinterpret_as_u64(a))); }


#define OPENCV_HAL_IMPL_MSA_REDUCE_OP_8U(func, cfunc) \
inline unsigned short v_reduce_##func(const v_uint16x8& a) \
{ \
    v8u16 a_lo, a_hi; \
    ILVRL_H2_UH(a.val, msa_dupq_n_u16(0), a_lo, a_hi); \
    v4u32 b = msa_##func##q_u32(msa_paddlq_u16(a_lo), msa_paddlq_u16(a_hi)); \
    v4u32 b_lo, b_hi; \
    ILVRL_W2_UW(b, msa_dupq_n_u32(0), b_lo, b_hi); \
    v2u64 c = msa_##func##q_u64(msa_paddlq_u32(b_lo), msa_paddlq_u32(b_hi)); \
    return (unsigned short)cfunc(c[0], c[1]); \
}

OPENCV_HAL_IMPL_MSA_REDUCE_OP_8U(max, std::max)
OPENCV_HAL_IMPL_MSA_REDUCE_OP_8U(min, std::min)

#define OPENCV_HAL_IMPL_MSA_REDUCE_OP_8S(func, cfunc) \
inline short v_reduce_##func(const v_int16x8& a) \
{ \
    v8i16 a_lo, a_hi; \
    ILVRL_H2_SH(a.val, msa_dupq_n_s16(0), a_lo, a_hi); \
    v4i32 b = msa_##func##q_s32(msa_paddlq_s16(a_lo), msa_paddlq_s16(a_hi)); \
    v4i32 b_lo, b_hi; \
    ILVRL_W2_SW(b, msa_dupq_n_s32(0), b_lo, b_hi); \
    v2i64 c = msa_##func##q_s64(msa_paddlq_s32(b_lo), msa_paddlq_s32(b_hi)); \
    return (short)cfunc(c[0], c[1]); \
}

OPENCV_HAL_IMPL_MSA_REDUCE_OP_8S(max, std::max)
OPENCV_HAL_IMPL_MSA_REDUCE_OP_8S(min, std::min)

#define OPENCV_HAL_IMPL_MSA_REDUCE_OP_4(_Tpvec, scalartype, func, cfunc) \
inline scalartype v_reduce_##func(const _Tpvec& a) \
{ \
    return (scalartype)cfunc(cfunc(a.val[0], a.val[1]), cfunc(a.val[2], a.val[3])); \
}

OPENCV_HAL_IMPL_MSA_REDUCE_OP_4(v_uint32x4, unsigned, max, std::max)
OPENCV_HAL_IMPL_MSA_REDUCE_OP_4(v_uint32x4, unsigned, min, std::min)
OPENCV_HAL_IMPL_MSA_REDUCE_OP_4(v_int32x4, int, max, std::max)
OPENCV_HAL_IMPL_MSA_REDUCE_OP_4(v_int32x4, int, min, std::min)
OPENCV_HAL_IMPL_MSA_REDUCE_OP_4(v_float32x4, float, max, std::max)
OPENCV_HAL_IMPL_MSA_REDUCE_OP_4(v_float32x4, float, min, std::min)


#define OPENCV_HAL_IMPL_MSA_REDUCE_OP_16(_Tpvec, scalartype, _Tpvec2, func) \
inline scalartype v_reduce_##func(const _Tpvec& a) \
{ \
    _Tpvec2 a1, a2; \
    v_expand(a, a1, a2); \
    return (scalartype)v_reduce_##func(v_##func(a1, a2)); \
}

OPENCV_HAL_IMPL_MSA_REDUCE_OP_16(v_uint8x16, uchar, v_uint16x8, min)
OPENCV_HAL_IMPL_MSA_REDUCE_OP_16(v_uint8x16, uchar, v_uint16x8, max)
OPENCV_HAL_IMPL_MSA_REDUCE_OP_16(v_int8x16, char, v_int16x8, min)
OPENCV_HAL_IMPL_MSA_REDUCE_OP_16(v_int8x16, char, v_int16x8, max)



#define OPENCV_HAL_IMPL_MSA_REDUCE_SUM(_Tpvec, scalartype, suffix) \
inline scalartype v_reduce_sum(const _Tpvec& a) \
{ \
    return (scalartype)msa_sum_##suffix(a.val); \
}

OPENCV_HAL_IMPL_MSA_REDUCE_SUM(v_uint8x16, unsigned short, u8)
OPENCV_HAL_IMPL_MSA_REDUCE_SUM(v_int8x16, short, s8)
OPENCV_HAL_IMPL_MSA_REDUCE_SUM(v_uint16x8, unsigned, u16)
OPENCV_HAL_IMPL_MSA_REDUCE_SUM(v_int16x8, int, s16)
OPENCV_HAL_IMPL_MSA_REDUCE_SUM(v_uint32x4, uint64_t, u32)
OPENCV_HAL_IMPL_MSA_REDUCE_SUM(v_int32x4, int64_t, s32)
OPENCV_HAL_IMPL_MSA_REDUCE_SUM(v_float32x4, float, f32)

inline uint64 v_reduce_sum(const v_uint64x2& a)
{ return (uint64)(msa_getq_lane_u64(a.val, 0) + msa_getq_lane_u64(a.val, 1)); }
inline int64 v_reduce_sum(const v_int64x2& a)
{ return (int64)(msa_getq_lane_s64(a.val, 0) + msa_getq_lane_s64(a.val, 1)); }
inline double v_reduce_sum(const v_float64x2& a)
{
    return msa_getq_lane_f64(a.val, 0) + msa_getq_lane_f64(a.val, 1);
}

/* v_reduce_sum4, v_reduce_sad */
inline v_float32x4 v_reduce_sum4(const v_float32x4& a, const v_float32x4& b,
                                 const v_float32x4& c, const v_float32x4& d)
{
    v4f32 u0 = msa_addq_f32(MSA_TPV_REINTERPRET(v4f32, msa_ilvevq_s32(MSA_TPV_REINTERPRET(v4i32, b.val), MSA_TPV_REINTERPRET(v4i32, a.val))),
                            MSA_TPV_REINTERPRET(v4f32, msa_ilvodq_s32(MSA_TPV_REINTERPRET(v4i32, b.val), MSA_TPV_REINTERPRET(v4i32, a.val)))); // a0+a1 b0+b1 a2+a3 b2+b3
    v4f32 u1 = msa_addq_f32(MSA_TPV_REINTERPRET(v4f32, msa_ilvevq_s32(MSA_TPV_REINTERPRET(v4i32, d.val), MSA_TPV_REINTERPRET(v4i32, c.val))),
                            MSA_TPV_REINTERPRET(v4f32, msa_ilvodq_s32(MSA_TPV_REINTERPRET(v4i32, d.val), MSA_TPV_REINTERPRET(v4i32, c.val)))); // c0+c1 d0+d1 c2+c3 d2+d3

    return v_float32x4(msa_addq_f32(MSA_TPV_REINTERPRET(v4f32, msa_ilvrq_s64(MSA_TPV_REINTERPRET(v2i64, u1), MSA_TPV_REINTERPRET(v2i64, u0))),
                                    MSA_TPV_REINTERPRET(v4f32, msa_ilvlq_s64(MSA_TPV_REINTERPRET(v2i64, u1), MSA_TPV_REINTERPRET(v2i64, u0)))));
}

inline unsigned v_reduce_sad(const v_uint8x16& a, const v_uint8x16& b)
{
    v16u8 t0 = msa_abdq_u8(a.val, b.val);
    v8u16 t1 = msa_paddlq_u8(t0);
    v4u32 t2 = msa_paddlq_u16(t1);
    return msa_sum_u32(t2);
}
inline unsigned v_reduce_sad(const v_int8x16& a, const v_int8x16& b)
{
    v16u8 t0 = MSA_TPV_REINTERPRET(v16u8, msa_abdq_s8(a.val, b.val));
    v8u16 t1 = msa_paddlq_u8(t0);
    v4u32 t2 = msa_paddlq_u16(t1);
    return msa_sum_u32(t2);
}
inline unsigned v_reduce_sad(const v_uint16x8& a, const v_uint16x8& b)
{
    v8u16 t0 = msa_abdq_u16(a.val, b.val);
    v4u32 t1 = msa_paddlq_u16(t0);
    return msa_sum_u32(t1);
}
inline unsigned v_reduce_sad(const v_int16x8& a, const v_int16x8& b)
{
    v8u16 t0 = MSA_TPV_REINTERPRET(v8u16, msa_abdq_s16(a.val, b.val));
    v4u32 t1 = msa_paddlq_u16(t0);
    return msa_sum_u32(t1);
}
inline unsigned v_reduce_sad(const v_uint32x4& a, const v_uint32x4& b)
{
    v4u32 t0 = msa_abdq_u32(a.val, b.val);
    return msa_sum_u32(t0);
}
inline unsigned v_reduce_sad(const v_int32x4& a, const v_int32x4& b)
{
    v4u32 t0 = MSA_TPV_REINTERPRET(v4u32, msa_abdq_s32(a.val, b.val));
    return msa_sum_u32(t0);
}
inline float v_reduce_sad(const v_float32x4& a, const v_float32x4& b)
{
    v4f32 t0 = msa_abdq_f32(a.val, b.val);
    return msa_sum_f32(t0);
}

/* v_popcount */
#define OPENCV_HAL_IMPL_MSA_POPCOUNT_SIZE8(_Tpvec) \
inline v_uint8x16 v_popcount(const _Tpvec& a) \
{ \
    v16u8 t = MSA_TPV_REINTERPRET(v16u8, msa_cntq_s8(MSA_TPV_REINTERPRET(v16i8, a.val))); \
    return v_uint8x16(t); \
}
OPENCV_HAL_IMPL_MSA_POPCOUNT_SIZE8(v_uint8x16)
OPENCV_HAL_IMPL_MSA_POPCOUNT_SIZE8(v_int8x16)

#define OPENCV_HAL_IMPL_MSA_POPCOUNT_SIZE16(_Tpvec) \
inline v_uint16x8 v_popcount(const _Tpvec& a) \
{ \
    v8u16 t = MSA_TPV_REINTERPRET(v8u16, msa_cntq_s16(MSA_TPV_REINTERPRET(v8i16, a.val))); \
    return v_uint16x8(t); \
}
OPENCV_HAL_IMPL_MSA_POPCOUNT_SIZE16(v_uint16x8)
OPENCV_HAL_IMPL_MSA_POPCOUNT_SIZE16(v_int16x8)

#define OPENCV_HAL_IMPL_MSA_POPCOUNT_SIZE32(_Tpvec) \
inline v_uint32x4 v_popcount(const _Tpvec& a) \
{ \
    v4u32 t = MSA_TPV_REINTERPRET(v4u32, msa_cntq_s32(MSA_TPV_REINTERPRET(v4i32, a.val))); \
    return v_uint32x4(t); \
}
OPENCV_HAL_IMPL_MSA_POPCOUNT_SIZE32(v_uint32x4)
OPENCV_HAL_IMPL_MSA_POPCOUNT_SIZE32(v_int32x4)

#define OPENCV_HAL_IMPL_MSA_POPCOUNT_SIZE64(_Tpvec) \
inline v_uint64x2 v_popcount(const _Tpvec& a) \
{ \
    v2u64 t = MSA_TPV_REINTERPRET(v2u64, msa_cntq_s64(MSA_TPV_REINTERPRET(v2i64, a.val))); \
    return v_uint64x2(t); \
}
OPENCV_HAL_IMPL_MSA_POPCOUNT_SIZE64(v_uint64x2)
OPENCV_HAL_IMPL_MSA_POPCOUNT_SIZE64(v_int64x2)

inline int v_signmask(const v_uint8x16& a)
{
    v8i8 m0 = msa_create_s8(CV_BIG_UINT(0x0706050403020100));
    v16u8 v0 = msa_shlq_u8(msa_shrq_n_u8(a.val, 7), msa_combine_s8(m0, m0));
    v8u16 v1 = msa_paddlq_u8(v0);
    v4u32 v2 = msa_paddlq_u16(v1);
    v2u64 v3 = msa_paddlq_u32(v2);
    return (int)msa_getq_lane_u64(v3, 0) + ((int)msa_getq_lane_u64(v3, 1) << 8);
}
inline int v_signmask(const v_int8x16& a)
{ return v_signmask(v_reinterpret_as_u8(a)); }

inline int v_signmask(const v_uint16x8& a)
{
    v4i16 m0 = msa_create_s16(CV_BIG_UINT(0x0003000200010000));
    v8u16 v0 = msa_shlq_u16(msa_shrq_n_u16(a.val, 15), msa_combine_s16(m0, m0));
    v4u32 v1 = msa_paddlq_u16(v0);
    v2u64 v2 = msa_paddlq_u32(v1);
    return (int)msa_getq_lane_u64(v2, 0) + ((int)msa_getq_lane_u64(v2, 1) << 4);
}
inline int v_signmask(const v_int16x8& a)
{ return v_signmask(v_reinterpret_as_u16(a)); }

inline int v_signmask(const v_uint32x4& a)
{
    v2i32 m0 = msa_create_s32(CV_BIG_UINT(0x0000000100000000));
    v4u32 v0 = msa_shlq_u32(msa_shrq_n_u32(a.val, 31), msa_combine_s32(m0, m0));
    v2u64 v1 = msa_paddlq_u32(v0);
    return (int)msa_getq_lane_u64(v1, 0) + ((int)msa_getq_lane_u64(v1, 1) << 2);
}
inline int v_signmask(const v_int32x4& a)
{ return v_signmask(v_reinterpret_as_u32(a)); }
inline int v_signmask(const v_float32x4& a)
{ return v_signmask(v_reinterpret_as_u32(a)); }

inline int v_signmask(const v_uint64x2& a)
{
    v2u64 v0 = msa_shrq_n_u64(a.val, 63);
    return (int)msa_getq_lane_u64(v0, 0) + ((int)msa_getq_lane_u64(v0, 1) << 1);
}
inline int v_signmask(const v_int64x2& a)
{ return v_signmask(v_reinterpret_as_u64(a)); }
inline int v_signmask(const v_float64x2& a)
{ return v_signmask(v_reinterpret_as_u64(a)); }

inline int v_scan_forward(const v_int8x16& a) { return trailingZeros32(v_signmask(a)); }
inline int v_scan_forward(const v_uint8x16& a) { return trailingZeros32(v_signmask(a)); }
inline int v_scan_forward(const v_int16x8& a) { return trailingZeros32(v_signmask(a)); }
inline int v_scan_forward(const v_uint16x8& a) { return trailingZeros32(v_signmask(a)); }
inline int v_scan_forward(const v_int32x4& a) { return trailingZeros32(v_signmask(a)); }
inline int v_scan_forward(const v_uint32x4& a) { return trailingZeros32(v_signmask(a)); }
inline int v_scan_forward(const v_float32x4& a) { return trailingZeros32(v_signmask(a)); }
inline int v_scan_forward(const v_int64x2& a) { return trailingZeros32(v_signmask(a)); }
inline int v_scan_forward(const v_uint64x2& a) { return trailingZeros32(v_signmask(a)); }
inline int v_scan_forward(const v_float64x2& a) { return trailingZeros32(v_signmask(a)); }

#define OPENCV_HAL_IMPL_MSA_CHECK_ALLANY(_Tpvec, _Tpvec2, suffix, shift) \
inline bool v_check_all(const v_##_Tpvec& a) \
{ \
    _Tpvec2 v0 = msa_shrq_n_##suffix(msa_mvnq_##suffix(a.val), shift); \
    v2u64 v1 = MSA_TPV_REINTERPRET(v2u64, v0); \
    return (msa_getq_lane_u64(v1, 0) | msa_getq_lane_u64(v1, 1)) == 0; \
} \
inline bool v_check_any(const v_##_Tpvec& a) \
{ \
    _Tpvec2 v0 = msa_shrq_n_##suffix(a.val, shift); \
    v2u64 v1 = MSA_TPV_REINTERPRET(v2u64, v0); \
    return (msa_getq_lane_u64(v1, 0) | msa_getq_lane_u64(v1, 1)) != 0; \
}

OPENCV_HAL_IMPL_MSA_CHECK_ALLANY(uint8x16, v16u8, u8, 7)
OPENCV_HAL_IMPL_MSA_CHECK_ALLANY(uint16x8, v8u16, u16, 15)
OPENCV_HAL_IMPL_MSA_CHECK_ALLANY(uint32x4, v4u32, u32, 31)
OPENCV_HAL_IMPL_MSA_CHECK_ALLANY(uint64x2, v2u64, u64, 63)

inline bool v_check_all(const v_int8x16& a)
{ return v_check_all(v_reinterpret_as_u8(a)); }
inline bool v_check_all(const v_int16x8& a)
{ return v_check_all(v_reinterpret_as_u16(a)); }
inline bool v_check_all(const v_int32x4& a)
{ return v_check_all(v_reinterpret_as_u32(a)); }
inline bool v_check_all(const v_float32x4& a)
{ return v_check_all(v_reinterpret_as_u32(a)); }

inline bool v_check_any(const v_int8x16& a)
{ return v_check_any(v_reinterpret_as_u8(a)); }
inline bool v_check_any(const v_int16x8& a)
{ return v_check_any(v_reinterpret_as_u16(a)); }
inline bool v_check_any(const v_int32x4& a)
{ return v_check_any(v_reinterpret_as_u32(a)); }
inline bool v_check_any(const v_float32x4& a)
{ return v_check_any(v_reinterpret_as_u32(a)); }

inline bool v_check_all(const v_int64x2& a)
{ return v_check_all(v_reinterpret_as_u64(a)); }
inline bool v_check_all(const v_float64x2& a)
{ return v_check_all(v_reinterpret_as_u64(a)); }
inline bool v_check_any(const v_int64x2& a)
{ return v_check_any(v_reinterpret_as_u64(a)); }
inline bool v_check_any(const v_float64x2& a)
{ return v_check_any(v_reinterpret_as_u64(a)); }

/* v_select */
#define OPENCV_HAL_IMPL_MSA_SELECT(_Tpvec, _Tpv, _Tpvu) \
inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(MSA_TPV_REINTERPRET(_Tpv, msa_bslq_u8(MSA_TPV_REINTERPRET(_Tpvu, mask.val), \
                  MSA_TPV_REINTERPRET(_Tpvu, b.val), MSA_TPV_REINTERPRET(_Tpvu, a.val)))); \
}

OPENCV_HAL_IMPL_MSA_SELECT(v_uint8x16, v16u8, v16u8)
OPENCV_HAL_IMPL_MSA_SELECT(v_int8x16, v16i8, v16u8)
OPENCV_HAL_IMPL_MSA_SELECT(v_uint16x8, v8u16, v16u8)
OPENCV_HAL_IMPL_MSA_SELECT(v_int16x8, v8i16, v16u8)
OPENCV_HAL_IMPL_MSA_SELECT(v_uint32x4, v4u32, v16u8)
OPENCV_HAL_IMPL_MSA_SELECT(v_int32x4, v4i32, v16u8)
OPENCV_HAL_IMPL_MSA_SELECT(v_float32x4, v4f32, v16u8)
OPENCV_HAL_IMPL_MSA_SELECT(v_float64x2, v2f64, v16u8)

#define OPENCV_HAL_IMPL_MSA_EXPAND(_Tpvec, _Tpwvec, _Tp, suffix, ssuffix, _Tpv, _Tpvs) \
inline void v_expand(const _Tpvec& a, _Tpwvec& b0, _Tpwvec& b1) \
{ \
    _Tpv a_lo = MSA_TPV_REINTERPRET(_Tpv, msa_ilvrq_##ssuffix(MSA_TPV_REINTERPRET(_Tpvs, a.val), msa_dupq_n_##ssuffix(0))); \
    _Tpv a_hi = MSA_TPV_REINTERPRET(_Tpv, msa_ilvlq_##ssuffix(MSA_TPV_REINTERPRET(_Tpvs, a.val), msa_dupq_n_##ssuffix(0))); \
    b0.val = msa_paddlq_##suffix(a_lo); \
    b1.val = msa_paddlq_##suffix(a_hi); \
} \
inline _Tpwvec v_expand_low(const _Tpvec& a) \
{ \
    _Tpv a_lo = MSA_TPV_REINTERPRET(_Tpv, msa_ilvrq_##ssuffix(MSA_TPV_REINTERPRET(_Tpvs, a.val), msa_dupq_n_##ssuffix(0))); \
    return _Tpwvec(msa_paddlq_##suffix(a_lo)); \
} \
inline _Tpwvec v_expand_high(const _Tpvec& a) \
{ \
    _Tpv a_hi = MSA_TPV_REINTERPRET(_Tpv, msa_ilvlq_##ssuffix(MSA_TPV_REINTERPRET(_Tpvs, a.val), msa_dupq_n_##ssuffix(0))); \
    return _Tpwvec(msa_paddlq_##suffix(a_hi)); \
} \
inline _Tpwvec v_load_expand(const _Tp* ptr) \
{ \
    return _Tpwvec(msa_movl_##suffix(msa_ld1_##suffix(ptr))); \
}

OPENCV_HAL_IMPL_MSA_EXPAND(v_uint8x16, v_uint16x8, uchar, u8, s8, v16u8, v16i8)
OPENCV_HAL_IMPL_MSA_EXPAND(v_int8x16, v_int16x8, schar, s8, s8, v16i8, v16i8)
OPENCV_HAL_IMPL_MSA_EXPAND(v_uint16x8, v_uint32x4, ushort, u16, s16, v8u16, v8i16)
OPENCV_HAL_IMPL_MSA_EXPAND(v_int16x8, v_int32x4, short, s16, s16, v8i16, v8i16)
OPENCV_HAL_IMPL_MSA_EXPAND(v_uint32x4, v_uint64x2, uint, u32, s32, v4u32, v4i32)
OPENCV_HAL_IMPL_MSA_EXPAND(v_int32x4, v_int64x2, int, s32, s32, v4i32, v4i32)

inline v_uint32x4 v_load_expand_q(const uchar* ptr)
{
    return v_uint32x4((v4u32){ptr[0], ptr[1], ptr[2], ptr[3]});
}

inline v_int32x4 v_load_expand_q(const schar* ptr)
{
    return v_int32x4((v4i32){ptr[0], ptr[1], ptr[2], ptr[3]});
}

/* v_zip, v_combine_low, v_combine_high, v_recombine */
#define OPENCV_HAL_IMPL_MSA_UNPACKS(_Tpvec, _Tpv, _Tpvs, ssuffix) \
inline void v_zip(const _Tpvec& a0, const _Tpvec& a1, _Tpvec& b0, _Tpvec& b1) \
{ \
    b0.val = MSA_TPV_REINTERPRET(_Tpv, msa_ilvrq_##ssuffix(MSA_TPV_REINTERPRET(_Tpvs, a1.val), MSA_TPV_REINTERPRET(_Tpvs, a0.val))); \
    b1.val = MSA_TPV_REINTERPRET(_Tpv, msa_ilvlq_##ssuffix(MSA_TPV_REINTERPRET(_Tpvs, a1.val), MSA_TPV_REINTERPRET(_Tpvs, a0.val))); \
} \
inline _Tpvec v_combine_low(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(MSA_TPV_REINTERPRET(_Tpv, msa_ilvrq_s64(MSA_TPV_REINTERPRET(v2i64, b.val), MSA_TPV_REINTERPRET(v2i64, a.val)))); \
} \
inline _Tpvec v_combine_high(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(MSA_TPV_REINTERPRET(_Tpv, msa_ilvlq_s64(MSA_TPV_REINTERPRET(v2i64, b.val), MSA_TPV_REINTERPRET(v2i64, a.val)))); \
} \
inline void v_recombine(const _Tpvec& a, const _Tpvec& b, _Tpvec& c, _Tpvec& d) \
{ \
    c.val = MSA_TPV_REINTERPRET(_Tpv, msa_ilvrq_s64(MSA_TPV_REINTERPRET(v2i64, b.val), MSA_TPV_REINTERPRET(v2i64, a.val))); \
    d.val = MSA_TPV_REINTERPRET(_Tpv, msa_ilvlq_s64(MSA_TPV_REINTERPRET(v2i64, b.val), MSA_TPV_REINTERPRET(v2i64, a.val))); \
}

OPENCV_HAL_IMPL_MSA_UNPACKS(v_uint8x16, v16u8, v16i8, s8)
OPENCV_HAL_IMPL_MSA_UNPACKS(v_int8x16, v16i8, v16i8, s8)
OPENCV_HAL_IMPL_MSA_UNPACKS(v_uint16x8, v8u16, v8i16, s16)
OPENCV_HAL_IMPL_MSA_UNPACKS(v_int16x8, v8i16, v8i16, s16)
OPENCV_HAL_IMPL_MSA_UNPACKS(v_uint32x4, v4u32, v4i32, s32)
OPENCV_HAL_IMPL_MSA_UNPACKS(v_int32x4, v4i32, v4i32, s32)
OPENCV_HAL_IMPL_MSA_UNPACKS(v_float32x4, v4f32, v4i32, s32)
OPENCV_HAL_IMPL_MSA_UNPACKS(v_float64x2, v2f64, v2i64, s64)

/* v_extract */
#define OPENCV_HAL_IMPL_MSA_EXTRACT(_Tpvec, _Tpv, _Tpvs, suffix) \
template <int s> \
inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(MSA_TPV_REINTERPRET(_Tpv, msa_extq_##suffix(MSA_TPV_REINTERPRET(_Tpvs, a.val), MSA_TPV_REINTERPRET(_Tpvs, b.val), s))); \
}

OPENCV_HAL_IMPL_MSA_EXTRACT(v_uint8x16, v16u8, v16i8, s8)
OPENCV_HAL_IMPL_MSA_EXTRACT(v_int8x16, v16i8, v16i8, s8)
OPENCV_HAL_IMPL_MSA_EXTRACT(v_uint16x8, v8u16, v8i16, s16)
OPENCV_HAL_IMPL_MSA_EXTRACT(v_int16x8, v8i16, v8i16, s16)
OPENCV_HAL_IMPL_MSA_EXTRACT(v_uint32x4, v4u32, v4i32, s32)
OPENCV_HAL_IMPL_MSA_EXTRACT(v_int32x4, v4i32, v4i32, s32)
OPENCV_HAL_IMPL_MSA_EXTRACT(v_uint64x2, v2u64, v2i64, s64)
OPENCV_HAL_IMPL_MSA_EXTRACT(v_int64x2, v2i64, v2i64, s64)
OPENCV_HAL_IMPL_MSA_EXTRACT(v_float32x4, v4f32, v4i32, s32)
OPENCV_HAL_IMPL_MSA_EXTRACT(v_float64x2, v2f64, v2i64, s64)

/* v_round, v_floor, v_ceil, v_trunc */
inline v_int32x4 v_round(const v_float32x4& a)
{
    return v_int32x4(msa_cvttintq_s32_f32(a.val));
}

inline v_int32x4 v_floor(const v_float32x4& a)
{
    v4i32 a1 = msa_cvttintq_s32_f32(a.val);
    return v_int32x4(msa_addq_s32(a1, MSA_TPV_REINTERPRET(v4i32, msa_cgtq_f32(msa_cvtfintq_f32_s32(a1), a.val))));
}

inline v_int32x4 v_ceil(const v_float32x4& a)
{
    v4i32 a1 = msa_cvttintq_s32_f32(a.val);
    return v_int32x4(msa_subq_s32(a1, MSA_TPV_REINTERPRET(v4i32, msa_cgtq_f32(a.val, msa_cvtfintq_f32_s32(a1)))));
}

inline v_int32x4 v_trunc(const v_float32x4& a)
{
    return v_int32x4(msa_cvttruncq_s32_f32(a.val));
}

inline v_int32x4 v_round(const v_float64x2& a)
{
    return v_int32x4(msa_pack_s64(msa_cvttintq_s64_f64(a.val), msa_dupq_n_s64(0)));
}

inline v_int32x4 v_round(const v_float64x2& a, const v_float64x2& b)
{
    return v_int32x4(msa_pack_s64(msa_cvttintq_s64_f64(a.val), msa_cvttintq_s64_f64(b.val)));
}

inline v_int32x4 v_floor(const v_float64x2& a)
{
    v2f64 a1 = msa_cvtrintq_f64(a.val);
    return v_int32x4(msa_pack_s64(msa_addq_s64(msa_cvttruncq_s64_f64(a1), MSA_TPV_REINTERPRET(v2i64, msa_cgtq_f64(a1, a.val))), msa_dupq_n_s64(0)));
}

inline v_int32x4 v_ceil(const v_float64x2& a)
{
    v2f64 a1 = msa_cvtrintq_f64(a.val);
    return v_int32x4(msa_pack_s64(msa_subq_s64(msa_cvttruncq_s64_f64(a1), MSA_TPV_REINTERPRET(v2i64, msa_cgtq_f64(a.val, a1))), msa_dupq_n_s64(0)));
}

inline v_int32x4 v_trunc(const v_float64x2& a)
{
    return v_int32x4(msa_pack_s64(msa_cvttruncq_s64_f64(a.val), msa_dupq_n_s64(0)));
}

#define OPENCV_HAL_IMPL_MSA_TRANSPOSE4x4(_Tpvec, _Tpv, _Tpvs, ssuffix) \
inline void v_transpose4x4(const _Tpvec& a0, const _Tpvec& a1, \
                           const _Tpvec& a2, const _Tpvec& a3, \
                           _Tpvec& b0, _Tpvec& b1, \
                           _Tpvec& b2, _Tpvec& b3) \
{ \
    _Tpv t00 = MSA_TPV_REINTERPRET(_Tpv, msa_ilvrq_##ssuffix(MSA_TPV_REINTERPRET(_Tpvs, a1.val), MSA_TPV_REINTERPRET(_Tpvs, a0.val))); \
    _Tpv t01 = MSA_TPV_REINTERPRET(_Tpv, msa_ilvlq_##ssuffix(MSA_TPV_REINTERPRET(_Tpvs, a1.val), MSA_TPV_REINTERPRET(_Tpvs, a0.val))); \
    _Tpv t10 = MSA_TPV_REINTERPRET(_Tpv, msa_ilvrq_##ssuffix(MSA_TPV_REINTERPRET(_Tpvs, a3.val), MSA_TPV_REINTERPRET(_Tpvs, a2.val))); \
    _Tpv t11 = MSA_TPV_REINTERPRET(_Tpv, msa_ilvlq_##ssuffix(MSA_TPV_REINTERPRET(_Tpvs, a3.val), MSA_TPV_REINTERPRET(_Tpvs, a2.val))); \
    b0.val = MSA_TPV_REINTERPRET(_Tpv, msa_ilvrq_s64(MSA_TPV_REINTERPRET(v2i64, t10), MSA_TPV_REINTERPRET(v2i64, t00))); \
    b1.val = MSA_TPV_REINTERPRET(_Tpv, msa_ilvlq_s64(MSA_TPV_REINTERPRET(v2i64, t10), MSA_TPV_REINTERPRET(v2i64, t00))); \
    b2.val = MSA_TPV_REINTERPRET(_Tpv, msa_ilvrq_s64(MSA_TPV_REINTERPRET(v2i64, t11), MSA_TPV_REINTERPRET(v2i64, t01))); \
    b3.val = MSA_TPV_REINTERPRET(_Tpv, msa_ilvlq_s64(MSA_TPV_REINTERPRET(v2i64, t11), MSA_TPV_REINTERPRET(v2i64, t01))); \
}

OPENCV_HAL_IMPL_MSA_TRANSPOSE4x4(v_uint32x4, v4u32, v4i32, s32)
OPENCV_HAL_IMPL_MSA_TRANSPOSE4x4(v_int32x4, v4i32, v4i32, s32)
OPENCV_HAL_IMPL_MSA_TRANSPOSE4x4(v_float32x4, v4f32, v4i32, s32)

#define OPENCV_HAL_IMPL_MSA_INTERLEAVED(_Tpvec, _Tp, suffix) \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec& a, v_##_Tpvec& b) \
{ \
    msa_ld2q_##suffix(ptr, &a.val, &b.val); \
} \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec& a, v_##_Tpvec& b, v_##_Tpvec& c) \
{ \
    msa_ld3q_##suffix(ptr, &a.val, &b.val, &c.val); \
} \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec& a, v_##_Tpvec& b, \
                                v_##_Tpvec& c, v_##_Tpvec& d) \
{ \
    msa_ld4q_##suffix(ptr, &a.val, &b.val, &c.val, &d.val); \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec& a, const v_##_Tpvec& b, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ \
    msa_st2q_##suffix(ptr, a.val, b.val); \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec& a, const v_##_Tpvec& b, \
                                const v_##_Tpvec& c, hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ \
    msa_st3q_##suffix(ptr, a.val, b.val, c.val); \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec& a, const v_##_Tpvec& b, \
                                const v_##_Tpvec& c, const v_##_Tpvec& d, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED ) \
{ \
    msa_st4q_##suffix(ptr, a.val, b.val, c.val, d.val); \
}

OPENCV_HAL_IMPL_MSA_INTERLEAVED(uint8x16, uchar, u8)
OPENCV_HAL_IMPL_MSA_INTERLEAVED(int8x16, schar, s8)
OPENCV_HAL_IMPL_MSA_INTERLEAVED(uint16x8, ushort, u16)
OPENCV_HAL_IMPL_MSA_INTERLEAVED(int16x8, short, s16)
OPENCV_HAL_IMPL_MSA_INTERLEAVED(uint32x4, unsigned, u32)
OPENCV_HAL_IMPL_MSA_INTERLEAVED(int32x4, int, s32)
OPENCV_HAL_IMPL_MSA_INTERLEAVED(float32x4, float, f32)
OPENCV_HAL_IMPL_MSA_INTERLEAVED(uint64x2, uint64, u64)
OPENCV_HAL_IMPL_MSA_INTERLEAVED(int64x2, int64, s64)
OPENCV_HAL_IMPL_MSA_INTERLEAVED(float64x2, double, f64)

/* v_cvt_f32, v_cvt_f64, v_cvt_f64_high */
inline v_float32x4 v_cvt_f32(const v_int32x4& a)
{
    return v_float32x4(msa_cvtfintq_f32_s32(a.val));
}

inline v_float32x4 v_cvt_f32(const v_float64x2& a)
{
    return v_float32x4(msa_cvtfq_f32_f64(a.val, msa_dupq_n_f64(0.0f)));
}

inline v_float32x4 v_cvt_f32(const v_float64x2& a, const v_float64x2& b)
{
    return v_float32x4(msa_cvtfq_f32_f64(a.val, b.val));
}

inline v_float64x2 v_cvt_f64(const v_int32x4& a)
{
    return v_float64x2(msa_cvtflq_f64_f32(msa_cvtfintq_f32_s32(a.val)));
}

inline v_float64x2 v_cvt_f64_high(const v_int32x4& a)
{
    return v_float64x2(msa_cvtfhq_f64_f32(msa_cvtfintq_f32_s32(a.val)));
}

inline v_float64x2 v_cvt_f64(const v_float32x4& a)
{
    return v_float64x2(msa_cvtflq_f64_f32(a.val));
}

inline v_float64x2 v_cvt_f64_high(const v_float32x4& a)
{
    return v_float64x2(msa_cvtfhq_f64_f32(a.val));
}

inline v_float64x2 v_cvt_f64(const v_int64x2& a)
{
    return v_float64x2(msa_cvtfintq_f64_s64(a.val));
}

////////////// Lookup table access ////////////////////
inline v_int8x16 v_lut(const schar* tab, const int* idx)
{
    schar CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idx[ 0]],
        tab[idx[ 1]],
        tab[idx[ 2]],
        tab[idx[ 3]],
        tab[idx[ 4]],
        tab[idx[ 5]],
        tab[idx[ 6]],
        tab[idx[ 7]],
        tab[idx[ 8]],
        tab[idx[ 9]],
        tab[idx[10]],
        tab[idx[11]],
        tab[idx[12]],
        tab[idx[13]],
        tab[idx[14]],
        tab[idx[15]]
    };
    return v_int8x16(msa_ld1q_s8(elems));
}
inline v_int8x16 v_lut_pairs(const schar* tab, const int* idx)
{
    schar CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idx[0]],
        tab[idx[0] + 1],
        tab[idx[1]],
        tab[idx[1] + 1],
        tab[idx[2]],
        tab[idx[2] + 1],
        tab[idx[3]],
        tab[idx[3] + 1],
        tab[idx[4]],
        tab[idx[4] + 1],
        tab[idx[5]],
        tab[idx[5] + 1],
        tab[idx[6]],
        tab[idx[6] + 1],
        tab[idx[7]],
        tab[idx[7] + 1]
    };
    return v_int8x16(msa_ld1q_s8(elems));
}
inline v_int8x16 v_lut_quads(const schar* tab, const int* idx)
{
    schar CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idx[0]],
        tab[idx[0] + 1],
        tab[idx[0] + 2],
        tab[idx[0] + 3],
        tab[idx[1]],
        tab[idx[1] + 1],
        tab[idx[1] + 2],
        tab[idx[1] + 3],
        tab[idx[2]],
        tab[idx[2] + 1],
        tab[idx[2] + 2],
        tab[idx[2] + 3],
        tab[idx[3]],
        tab[idx[3] + 1],
        tab[idx[3] + 2],
        tab[idx[3] + 3]
    };
    return v_int8x16(msa_ld1q_s8(elems));
}
inline v_uint8x16 v_lut(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut((schar*)tab, idx)); }
inline v_uint8x16 v_lut_pairs(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_pairs((schar*)tab, idx)); }
inline v_uint8x16 v_lut_quads(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_quads((schar*)tab, idx)); }


inline v_int16x8 v_lut(const short* tab, const int* idx)
{
    short CV_DECL_ALIGNED(32) elems[8] =
    {
        tab[idx[0]],
        tab[idx[1]],
        tab[idx[2]],
        tab[idx[3]],
        tab[idx[4]],
        tab[idx[5]],
        tab[idx[6]],
        tab[idx[7]]
    };
    return v_int16x8(msa_ld1q_s16(elems));
}
inline v_int16x8 v_lut_pairs(const short* tab, const int* idx)
{
    short CV_DECL_ALIGNED(32) elems[8] =
    {
        tab[idx[0]],
        tab[idx[0] + 1],
        tab[idx[1]],
        tab[idx[1] + 1],
        tab[idx[2]],
        tab[idx[2] + 1],
        tab[idx[3]],
        tab[idx[3] + 1]
    };
    return v_int16x8(msa_ld1q_s16(elems));
}
inline v_int16x8 v_lut_quads(const short* tab, const int* idx)
{
    return v_int16x8(msa_combine_s16(msa_ld1_s16(tab + idx[0]), msa_ld1_s16(tab + idx[1])));
}
inline v_uint16x8 v_lut(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut((short*)tab, idx)); }
inline v_uint16x8 v_lut_pairs(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_pairs((short*)tab, idx)); }
inline v_uint16x8 v_lut_quads(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_quads((short*)tab, idx)); }

inline v_int32x4 v_lut(const int* tab, const int* idx)
{
    int CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idx[0]],
        tab[idx[1]],
        tab[idx[2]],
        tab[idx[3]]
    };
    return v_int32x4(msa_ld1q_s32(elems));
}
inline v_int32x4 v_lut_pairs(const int* tab, const int* idx)
{
    return v_int32x4(msa_combine_s32(msa_ld1_s32(tab + idx[0]), msa_ld1_s32(tab + idx[1])));
}
inline v_int32x4 v_lut_quads(const int* tab, const int* idx)
{
    return v_int32x4(msa_ld1q_s32(tab + idx[0]));
}
inline v_uint32x4 v_lut(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut((int*)tab, idx)); }
inline v_uint32x4 v_lut_pairs(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_pairs((int*)tab, idx)); }
inline v_uint32x4 v_lut_quads(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_quads((int*)tab, idx)); }

inline v_int64x2 v_lut(const int64_t* tab, const int* idx)
{
    return v_int64x2(msa_combine_s64(msa_create_s64(tab[idx[0]]), msa_create_s64(tab[idx[1]])));
}
inline v_int64x2 v_lut_pairs(const int64_t* tab, const int* idx)
{
    return v_int64x2(msa_ld1q_s64(tab + idx[0]));
}
inline v_uint64x2 v_lut(const uint64_t* tab, const int* idx) { return v_reinterpret_as_u64(v_lut((const int64_t *)tab, idx)); }
inline v_uint64x2 v_lut_pairs(const uint64_t* tab, const int* idx) { return v_reinterpret_as_u64(v_lut_pairs((const int64_t *)tab, idx)); }

inline v_float32x4 v_lut(const float* tab, const int* idx)
{
    float CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idx[0]],
        tab[idx[1]],
        tab[idx[2]],
        tab[idx[3]]
    };
    return v_float32x4(msa_ld1q_f32(elems));
}
inline v_float32x4 v_lut_pairs(const float* tab, const int* idx)
{
    uint64 CV_DECL_ALIGNED(32) elems[2] =
    {
        *(uint64*)(tab + idx[0]),
        *(uint64*)(tab + idx[1])
    };
    return v_float32x4(MSA_TPV_REINTERPRET(v4f32, msa_ld1q_u64(elems)));
}
inline v_float32x4 v_lut_quads(const float* tab, const int* idx)
{
    return v_float32x4(msa_ld1q_f32(tab + idx[0]));
}

inline v_int32x4 v_lut(const int* tab, const v_int32x4& idxvec)
{
    int CV_DECL_ALIGNED(32) idx[4];
    v_store_aligned(idx, idxvec);

    return v_int32x4(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]);
}

inline v_uint32x4 v_lut(const unsigned* tab, const v_int32x4& idxvec)
{
    unsigned CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[msa_getq_lane_s32(idxvec.val, 0)],
        tab[msa_getq_lane_s32(idxvec.val, 1)],
        tab[msa_getq_lane_s32(idxvec.val, 2)],
        tab[msa_getq_lane_s32(idxvec.val, 3)]
    };
    return v_uint32x4(msa_ld1q_u32(elems));
}

inline v_float32x4 v_lut(const float* tab, const v_int32x4& idxvec)
{
    int CV_DECL_ALIGNED(32) idx[4];
    v_store_aligned(idx, idxvec);

    return v_float32x4(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]);
}

inline void v_lut_deinterleave(const float* tab, const v_int32x4& idxvec, v_float32x4& x, v_float32x4& y)
{
    int CV_DECL_ALIGNED(32) idx[4];
    v_store_aligned(idx, idxvec);

    v4f32 xy02 = msa_combine_f32(msa_ld1_f32(tab + idx[0]), msa_ld1_f32(tab + idx[2]));
    v4f32 xy13 = msa_combine_f32(msa_ld1_f32(tab + idx[1]), msa_ld1_f32(tab + idx[3]));
    x = v_float32x4(MSA_TPV_REINTERPRET(v4f32, msa_ilvevq_s32(MSA_TPV_REINTERPRET(v4i32, xy13), MSA_TPV_REINTERPRET(v4i32, xy02))));
    y = v_float32x4(MSA_TPV_REINTERPRET(v4f32, msa_ilvodq_s32(MSA_TPV_REINTERPRET(v4i32, xy13), MSA_TPV_REINTERPRET(v4i32, xy02))));
}

inline v_int8x16 v_interleave_pairs(const v_int8x16& vec)
{
    v_int8x16 c = v_int8x16(__builtin_msa_vshf_b((v16i8)((v2i64){0x0705060403010200, 0x0F0D0E0C0B090A08}), msa_dupq_n_s8(0), vec.val));
    return c;
}
inline v_uint8x16 v_interleave_pairs(const v_uint8x16& vec)
{ return v_reinterpret_as_u8(v_interleave_pairs(v_reinterpret_as_s8(vec))); }
inline v_int8x16 v_interleave_quads(const v_int8x16& vec)
{
    v_int8x16 c = v_int8x16(__builtin_msa_vshf_b((v16i8)((v2i64){0x0703060205010400, 0x0F0B0E0A0D090C08}), msa_dupq_n_s8(0), vec.val));
    return c;
}
inline v_uint8x16 v_interleave_quads(const v_uint8x16& vec) { return v_reinterpret_as_u8(v_interleave_quads(v_reinterpret_as_s8(vec))); }

inline v_int16x8 v_interleave_pairs(const v_int16x8& vec)
{
    v_int16x8 c = v_int16x8(__builtin_msa_vshf_h((v8i16)((v2i64){0x0003000100020000, 0x0007000500060004}), msa_dupq_n_s16(0), vec.val));
    return c;
}

inline v_uint16x8 v_interleave_pairs(const v_uint16x8& vec) { return v_reinterpret_as_u16(v_interleave_pairs(v_reinterpret_as_s16(vec))); }

inline v_int16x8 v_interleave_quads(const v_int16x8& vec)
{
    v_int16x8 c = v_int16x8(__builtin_msa_vshf_h((v8i16)((v2i64){0x0005000100040000, 0x0007000300060002}), msa_dupq_n_s16(0), vec.val));
    return c;
}

inline v_uint16x8 v_interleave_quads(const v_uint16x8& vec) { return v_reinterpret_as_u16(v_interleave_quads(v_reinterpret_as_s16(vec))); }

inline v_int32x4 v_interleave_pairs(const v_int32x4& vec)
{
    v_int32x4 c;
    c.val[0] = vec.val[0];
    c.val[1] = vec.val[2];
    c.val[2] = vec.val[1];
    c.val[3] = vec.val[3];
    return c;
}

inline v_uint32x4 v_interleave_pairs(const v_uint32x4& vec) { return v_reinterpret_as_u32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }
inline v_float32x4 v_interleave_pairs(const v_float32x4& vec) { return v_reinterpret_as_f32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }

inline v_int8x16 v_pack_triplets(const v_int8x16& vec)
{
    v_int8x16 c = v_int8x16(__builtin_msa_vshf_b((v16i8)((v2i64){0x0908060504020100, 0x131211100E0D0C0A}), msa_dupq_n_s8(0), vec.val));
    return c;
}

inline v_uint8x16 v_pack_triplets(const v_uint8x16& vec) { return v_reinterpret_as_u8(v_pack_triplets(v_reinterpret_as_s8(vec))); }

inline v_int16x8 v_pack_triplets(const v_int16x8& vec)
{
    v_int16x8 c = v_int16x8(__builtin_msa_vshf_h((v8i16)((v2i64){0x0004000200010000, 0x0009000800060005}), msa_dupq_n_s16(0), vec.val));
    return c;
}

inline v_uint16x8 v_pack_triplets(const v_uint16x8& vec) { return v_reinterpret_as_u16(v_pack_triplets(v_reinterpret_as_s16(vec))); }
inline v_int32x4 v_pack_triplets(const v_int32x4& vec) { return vec; }
inline v_uint32x4 v_pack_triplets(const v_uint32x4& vec) { return vec; }
inline v_float32x4 v_pack_triplets(const v_float32x4& vec) { return vec; }

inline v_float64x2 v_lut(const double* tab, const int* idx)
{
    double CV_DECL_ALIGNED(32) elems[2] =
    {
        tab[idx[0]],
        tab[idx[1]]
    };
    return v_float64x2(msa_ld1q_f64(elems));
}

inline v_float64x2 v_lut_pairs(const double* tab, const int* idx)
{
    return v_float64x2(msa_ld1q_f64(tab + idx[0]));
}

inline v_float64x2 v_lut(const double* tab, const v_int32x4& idxvec)
{
    int CV_DECL_ALIGNED(32) idx[4];
    v_store_aligned(idx, idxvec);

    return v_float64x2(tab[idx[0]], tab[idx[1]]);
}

inline void v_lut_deinterleave(const double* tab, const v_int32x4& idxvec, v_float64x2& x, v_float64x2& y)
{
    int CV_DECL_ALIGNED(32) idx[4];
    v_store_aligned(idx, idxvec);

    v2f64 xy0 = msa_ld1q_f64(tab + idx[0]);
    v2f64 xy1 = msa_ld1q_f64(tab + idx[1]);
    x = v_float64x2(MSA_TPV_REINTERPRET(v2f64, msa_ilvevq_s64(MSA_TPV_REINTERPRET(v2i64, xy1), MSA_TPV_REINTERPRET(v2i64, xy0))));
    y = v_float64x2(MSA_TPV_REINTERPRET(v2f64, msa_ilvodq_s64(MSA_TPV_REINTERPRET(v2i64, xy1), MSA_TPV_REINTERPRET(v2i64, xy0))));
}

template<int i, typename _Tp>
inline typename _Tp::lane_type v_extract_n(const _Tp& a)
{
    return v_rotate_right<i>(a).get0();
}

template<int i>
inline v_uint32x4 v_broadcast_element(const v_uint32x4& a)
{
    return v_setall_u32(v_extract_n<i>(a));
}
template<int i>
inline v_int32x4 v_broadcast_element(const v_int32x4& a)
{
    return v_setall_s32(v_extract_n<i>(a));
}
template<int i>
inline v_float32x4 v_broadcast_element(const v_float32x4& a)
{
    return v_setall_f32(v_extract_n<i>(a));
}

////// FP16 support ///////
#if CV_FP16
inline v_float32x4 v_load_expand(const hfloat* ptr)
{
#ifndef msa_ld1_f16
    v4f16 v = (v4f16)msa_ld1_s16((const short*)ptr);
#else
    v4f16 v = msa_ld1_f16((const __fp16*)ptr);
#endif
    return v_float32x4(msa_cvt_f32_f16(v));
}

inline void v_pack_store(hfloat* ptr, const v_float32x4& v)
{
    v4f16 hv = msa_cvt_f16_f32(v.val);

#ifndef msa_st1_f16
    msa_st1_s16((short*)ptr, (int16x4_t)hv);
#else
    msa_st1_f16((__fp16*)ptr, hv);
#endif
}
#else
inline v_float32x4 v_load_expand(const hfloat* ptr)
{
    float buf[4];
    for( int i = 0; i < 4; i++ )
        buf[i] = (float)ptr[i];
    return v_load(buf);
}

inline void v_pack_store(hfloat* ptr, const v_float32x4& v)
{
    float buf[4];
    v_store(buf, v);
    for( int i = 0; i < 4; i++ )
        ptr[i] = (hfloat)buf[i];
}
#endif

inline void v_cleanup() {}

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

}

#endif
