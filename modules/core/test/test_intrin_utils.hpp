#ifndef _TEST_UTILS_HPP_
#define _TEST_UTILS_HPP_

#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/ts.hpp"
#include <ostream>
#include <algorithm>

template <typename R> struct Data;
template <int N> struct initializer;

template <> struct initializer<16>
{
    template <typename R> static R init(const Data<R> & d)
    {
        return R(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
    }
};

template <> struct initializer<8>
{
    template <typename R> static R init(const Data<R> & d)
    {
        return R(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
    }
};

template <> struct initializer<4>
{
    template <typename R> static R init(const Data<R> & d)
    {
        return R(d[0], d[1], d[2], d[3]);
    }
};

template <> struct initializer<2>
{
    template <typename R> static R init(const Data<R> & d)
    {
        return R(d[0], d[1]);
    }
};

//==================================================================================================

template <typename R> struct Data
{
    typedef typename R::lane_type LaneType;
    Data()
    {
        for (int i = 0; i < R::nlanes; ++i)
            d[i] = (LaneType)(i + 1);
    }
    Data(LaneType val)
    {
        fill(val);
    }
    Data(const R & r)
    {
        *this = r;
    }
    operator R ()
    {
        return initializer<R::nlanes>().init(*this);
    }
    Data<R> & operator=(const R & r)
    {
        v_store(d, r);
        return *this;
    }
    template <typename T> Data<R> & operator*=(T m)
    {
        for (int i = 0; i < R::nlanes; ++i)
            d[i] *= (LaneType)m;
        return *this;
    }
    template <typename T> Data<R> & operator+=(T m)
    {
        for (int i = 0; i < R::nlanes; ++i)
            d[i] += (LaneType)m;
        return *this;
    }
    void fill(LaneType val)
    {
        for (int i = 0; i < R::nlanes; ++i)
            d[i] = val;
    }
    void reverse()
    {
        for (int i = 0; i < R::nlanes / 2; ++i)
            std::swap(d[i], d[R::nlanes - i - 1]);
    }
    const LaneType & operator[](int i) const
    {
        CV_Assert(i >= 0 && i < R::nlanes);
        return d[i];
    }
    LaneType & operator[](int i)
    {
        CV_Assert(i >= 0 && i < R::nlanes);
        return d[i];
    }
    const LaneType * mid() const
    {
        return d + R::nlanes / 2;
    }
    LaneType * mid()
    {
        return d + R::nlanes / 2;
    }
    bool operator==(const Data<R> & other) const
    {
        for (int i = 0; i < R::nlanes; ++i)
            if (d[i] != other.d[i])
                return false;
        return true;
    }
    void clear()
    {
        fill(0);
    }
    bool isZero() const
    {
        return isValue(0);
    }
    bool isValue(uchar val) const
    {
        for (int i = 0; i < R::nlanes; ++i)
            if (d[i] != val)
                return false;
        return true;
    }

    LaneType d[R::nlanes];
};

template<typename R> struct AlignedData
{
    Data<R> CV_DECL_ALIGNED(16) a; // aligned
    char dummy;
    Data<R> u; // unaligned
};

template <typename R> std::ostream & operator<<(std::ostream & out, const Data<R> & d)
{
    out << "{ ";
    for (int i = 0; i < R::nlanes; ++i)
    {
        // out << std::hex << +V_TypeTraits<typename R::lane_type>::reinterpret_int(d.d[i]);
        out << +d.d[i];
        if (i + 1 < R::nlanes)
            out << ", ";
    }
    out << " }";
    return out;
}

//==================================================================================================

template <typename R> struct RegTrait;

template <> struct RegTrait<cv::v_uint8x16> {
    typedef cv::v_uint16x8 w_reg;
    typedef cv::v_uint32x4 q_reg;
    typedef cv::v_uint8x16 u_reg;
    static cv::v_uint8x16 zero() { return cv::v_setzero_u8(); }
    static cv::v_uint8x16 all(uchar val) { return cv::v_setall_u8(val); }
};
template <> struct RegTrait<cv::v_int8x16> {
    typedef cv::v_int16x8 w_reg;
    typedef cv::v_int32x4 q_reg;
    typedef cv::v_uint8x16 u_reg;
    static cv::v_int8x16 zero() { return cv::v_setzero_s8(); }
    static cv::v_int8x16 all(schar val) { return cv::v_setall_s8(val); }
};

template <> struct RegTrait<cv::v_uint16x8> {
    typedef cv::v_uint32x4 w_reg;
    typedef cv::v_int16x8 int_reg;
    typedef cv::v_uint16x8 u_reg;
    static cv::v_uint16x8 zero() { return cv::v_setzero_u16(); }
    static cv::v_uint16x8 all(ushort val) { return cv::v_setall_u16(val); }
};

template <> struct RegTrait<cv::v_int16x8> {
    typedef cv::v_int32x4 w_reg;
    typedef cv::v_uint16x8 u_reg;
    static cv::v_int16x8 zero() { return cv::v_setzero_s16(); }
    static cv::v_int16x8 all(short val) { return cv::v_setall_s16(val); }
};

template <> struct RegTrait<cv::v_uint32x4> {
    typedef cv::v_uint64x2 w_reg;
    typedef cv::v_int32x4 int_reg;
    typedef cv::v_uint32x4 u_reg;
    static cv::v_uint32x4 zero() { return cv::v_setzero_u32(); }
    static cv::v_uint32x4 all(unsigned val) { return cv::v_setall_u32(val); }
};

template <> struct RegTrait<cv::v_int32x4> {
    typedef cv::v_int64x2 w_reg;
    typedef cv::v_uint32x4 u_reg;
    static cv::v_int32x4 zero() { return cv::v_setzero_s32(); }
    static cv::v_int32x4 all(int val) { return cv::v_setall_s32(val); }
};

template <> struct RegTrait<cv::v_uint64x2> {
    static cv::v_uint64x2 zero() { return cv::v_setzero_u64(); }
    static cv::v_uint64x2 all(uint64 val) { return cv::v_setall_u64(val); }
};

template <> struct RegTrait<cv::v_int64x2> {
    static cv::v_int64x2 zero() { return cv::v_setzero_s64(); }
    static cv::v_int64x2 all(int64 val) { return cv::v_setall_s64(val); }
};

template <> struct RegTrait<cv::v_float32x4> {
    typedef cv::v_int32x4 int_reg;
    typedef cv::v_float32x4 u_reg;
    static cv::v_float32x4 zero() { return cv::v_setzero_f32(); }
    static cv::v_float32x4 all(float val) { return cv::v_setall_f32(val); }
};

#if CV_SIMD128_64F
template <> struct RegTrait<cv::v_float64x2> {
    typedef cv::v_int32x4 int_reg;
    typedef cv::v_float64x2 u_reg;
    static cv::v_float64x2 zero() { return cv::v_setzero_f64(); }
    static cv::v_float64x2 all(double val) { return cv::v_setall_f64(val); }
};

#endif

#endif
