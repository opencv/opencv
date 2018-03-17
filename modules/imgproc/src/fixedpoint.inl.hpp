// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.


#ifndef _CV_FIXEDPOINT_HPP_
#define _CV_FIXEDPOINT_HPP_

#include "opencv2/core/softfloat.hpp"

#ifndef CV_ALWAYS_INLINE
    #if defined(__GNUC__) && (__GNUC__ > 3 ||(__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
        #define CV_ALWAYS_INLINE inline __attribute__((always_inline))
    #elif defined(_MSC_VER)
        #define CV_ALWAYS_INLINE __forceinline
    #else
        #define CV_ALWAYS_INLINE inline
    #endif
#endif

namespace
{

class fixedpoint64
{
private:
    static const int fixedShift = 32;

    int64_t val;
    fixedpoint64(int64_t _val) : val(_val) {}
    static CV_ALWAYS_INLINE uint64_t fixedround(const uint64_t& _val) { return (_val + ((1LL << fixedShift) >> 1)); }
public:
    typedef fixedpoint64 WT;
    CV_ALWAYS_INLINE fixedpoint64() { val = 0; }
    CV_ALWAYS_INLINE fixedpoint64(const int8_t& _val) { val = ((int64_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE fixedpoint64(const int16_t& _val) { val = ((int64_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE fixedpoint64(const int32_t& _val) { val = ((int64_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE fixedpoint64(const cv::softdouble& _val) { val = cvRound64(_val * cv::softdouble((int64_t)(1LL << fixedShift))); }
    CV_ALWAYS_INLINE fixedpoint64& operator = (const int8_t& _val) { val = ((int64_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE fixedpoint64& operator = (const int16_t& _val) { val = ((int64_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE fixedpoint64& operator = (const int32_t& _val) { val = ((int64_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE fixedpoint64& operator = (const cv::softdouble& _val) { val = cvRound64(_val * cv::softdouble((int64_t)(1LL << fixedShift))); return *this; }
    CV_ALWAYS_INLINE fixedpoint64& operator = (const fixedpoint64& _val) { val = _val.val; return *this; }
    template <typename ET>
    CV_ALWAYS_INLINE fixedpoint64 operator * (const ET& val2) const { return val * val2; } // Wrong rounding is possible for floating point types
    CV_ALWAYS_INLINE fixedpoint64 operator * (const fixedpoint64& val2) const
    {
        //Assume -0x00000000C0000000 <= val2 <=0x0000000100000000 INT64_MIN <= val <= INT64_MAX, so shifted multiplication result is inside [INT64_MIN, INT64_MAX] range
        uint64_t uval = (uint64_t)((val ^ (val >> 63)) - (val >> 63));
        uint64_t umul = (uint64_t)((val2.val ^ (val2.val >> 63)) - (val2.val >> 63));
        int64_t ressign = (val >> 63) ^ (val2.val >> 63);

        uint64_t sh0   = fixedround((uval & 0xFFFFFFFF) * (umul & 0xFFFFFFFF));
        uint64_t sh1_0 = (uval >> 32)        * (umul & 0xFFFFFFFF);
        uint64_t sh1_1 = (uval & 0xFFFFFFFF) * (umul >> 32);
        uint64_t sh2   = (uval >> 32)        * (umul >> 32);
        uint64_t val0_l = (sh1_0 & 0xFFFFFFFF) + (sh1_1 & 0xFFFFFFFF) + (sh0 >> 32);
        uint64_t val0_h = (sh2   & 0xFFFFFFFF) + (sh1_0 >> 32) + (sh1_1 >> 32) + (val0_l >> 32);
        val0_l &= 0xFFFFFFFF;

        if (ressign)
        {
            val0_l = (~val0_l + 1) & 0xFFFFFFFF;
            val0_h = val0_l ? ~val0_h : (~val0_h + 1);
        }
        return (int64_t)(val0_h << 32 | val0_l);
    }
    CV_ALWAYS_INLINE fixedpoint64 operator + (const fixedpoint64& val2) const { return fixedpoint64(val + val2.val); }
    CV_ALWAYS_INLINE fixedpoint64 operator - (const fixedpoint64& val2) const { return fixedpoint64(val - val2.val); }
    //    CV_ALWAYS_INLINE fixedpoint64 operator + (const fixedpoint64& val2) const
    //    {
    //        int64_t nfrac = (int64_t)frac + val2.frac;
    //        int64_t nval = (int64_t)val + val2.val + nfrac >> 32;
    //        return nval > MAXINT32 ? beConv(MAXINT32, MAXINT32) : beConv((int32_t)(nval), 0);
    //    }
    CV_ALWAYS_INLINE fixedpoint64 operator >> (int n) const { return fixedpoint64(val >> n); }
    CV_ALWAYS_INLINE fixedpoint64 operator << (int n) const { return fixedpoint64(val << n); }
    template <typename ET>
    CV_ALWAYS_INLINE operator ET() const { return cv::saturate_cast<ET>((int64_t)fixedround((uint64_t)val) >> fixedShift); }
    CV_ALWAYS_INLINE operator double() const { return (double)val / (1LL << fixedShift); }
    CV_ALWAYS_INLINE operator float() const { return (float)val / (1LL << fixedShift); }
    CV_ALWAYS_INLINE bool isZero() { return val == 0; }
    static CV_ALWAYS_INLINE fixedpoint64 zero() { return fixedpoint64(); }
    static CV_ALWAYS_INLINE fixedpoint64 one() { return fixedpoint64((int64_t)(1LL << fixedShift)); }
    friend class fixedpoint32;
};

class ufixedpoint64
{
private:
    static const int fixedShift = 32;

    uint64_t val;
    ufixedpoint64(uint64_t _val) : val(_val) {}
    static CV_ALWAYS_INLINE uint64_t fixedround(const uint64_t& _val) { return (_val + ((1LL << fixedShift) >> 1)); }
public:
    typedef ufixedpoint64 WT;
    CV_ALWAYS_INLINE ufixedpoint64() { val = 0; }
    CV_ALWAYS_INLINE ufixedpoint64(const uint8_t& _val) { val = ((uint64_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE ufixedpoint64(const uint16_t& _val) { val = ((uint64_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE ufixedpoint64(const uint32_t& _val) { val = ((uint64_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE ufixedpoint64(const cv::softdouble& _val) { val = _val.getSign() ? 0 : (uint64_t)cvRound64(_val * cv::softdouble((int64_t)(1LL << fixedShift))); }
    CV_ALWAYS_INLINE ufixedpoint64& operator = (const uint8_t& _val) { val = ((uint64_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE ufixedpoint64& operator = (const uint16_t& _val) { val = ((uint64_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE ufixedpoint64& operator = (const uint32_t& _val) { val = ((uint64_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE ufixedpoint64& operator = (const cv::softdouble& _val) { val = _val.getSign() ? 0 : (uint64_t)cvRound64(_val * cv::softdouble((int64_t)(1LL << fixedShift))); return *this; }
    CV_ALWAYS_INLINE ufixedpoint64& operator = (const ufixedpoint64& _val) { val = _val.val; return *this; }
    template <typename ET>
    CV_ALWAYS_INLINE ufixedpoint64 operator * (const ET& val2) const { return val * val2; } // Wrong rounding is possible for floating point types
    CV_ALWAYS_INLINE ufixedpoint64 operator * (const ufixedpoint64& val2) const
    {
        //Assume val2 <=0x0000000100000000, so shifted multiplication result is less than val and therefore than UINT64_MAX
        uint64_t sh0 = fixedround((val & 0xFFFFFFFF) * (val2.val & 0xFFFFFFFF));
        uint64_t sh1_0 = (val >> 32)        * (val2.val & 0xFFFFFFFF);
        uint64_t sh1_1 = (val & 0xFFFFFFFF) * (val2.val >> 32);
        uint64_t sh2 = (val >> 32)        * (val2.val >> 32);
        uint64_t val0_l = (sh1_0 & 0xFFFFFFFF) + (sh1_1 & 0xFFFFFFFF) + (sh0 >> 32);
        uint64_t val0_h = (sh2 & 0xFFFFFFFF) + (sh1_0 >> 32) + (sh1_1 >> 32) + (val0_l >> 32);
        val0_l &= 0xFFFFFFFF;

        return val0_h << 32 | val0_l;
    }
    CV_ALWAYS_INLINE ufixedpoint64 operator + (const ufixedpoint64& val2) const { return ufixedpoint64(val + val2.val); }
    CV_ALWAYS_INLINE ufixedpoint64 operator - (const ufixedpoint64& val2) const { return ufixedpoint64(val - val2.val); }
    //    CV_ALWAYS_INLINE fixedpoint64 operator + (const fixedpoint64& val2) const
    //    {
    //        int64_t nfrac = (int64_t)frac + val2.frac;
    //        int64_t nval = (int64_t)val + val2.val + nfrac >> 32;
    //        return nval > MAXINT32 ? beConv(MAXINT32, MAXINT32) : beConv((int32_t)(nval), 0);
    //    }
    CV_ALWAYS_INLINE ufixedpoint64 operator >> (int n) const { return ufixedpoint64(val >> n); }
    CV_ALWAYS_INLINE ufixedpoint64 operator << (int n) const { return ufixedpoint64(val << n); }
    template <typename ET>
    CV_ALWAYS_INLINE operator ET() const { return cv::saturate_cast<ET>(fixedround(val) >> fixedShift); }
    CV_ALWAYS_INLINE operator double() const { return (double)val / (1LL << fixedShift); }
    CV_ALWAYS_INLINE operator float() const { return (float)val / (1LL << fixedShift); }
    CV_ALWAYS_INLINE bool isZero() { return val == 0; }
    static CV_ALWAYS_INLINE ufixedpoint64 zero() { return ufixedpoint64(); }
    static CV_ALWAYS_INLINE ufixedpoint64 one() { return ufixedpoint64((uint64_t)(1ULL << fixedShift)); }
    friend class ufixedpoint32;
};

class fixedpoint32
{
private:
    static const int fixedShift = 16;

    int32_t val;
    fixedpoint32(int32_t _val) : val(_val) {}
    static CV_ALWAYS_INLINE uint32_t fixedround(const uint32_t& _val) { return (_val + ((1 << fixedShift) >> 1)); }
public:
    typedef fixedpoint64 WT;
    CV_ALWAYS_INLINE fixedpoint32() { val = 0; }
    CV_ALWAYS_INLINE fixedpoint32(const int8_t& _val) { val = ((int32_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE fixedpoint32(const uint8_t& _val) { val = ((int32_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE fixedpoint32(const int16_t& _val) { val = ((int32_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE fixedpoint32(const cv::softdouble& _val) { val = (int32_t)cvRound(_val * cv::softdouble((1 << fixedShift))); }
    CV_ALWAYS_INLINE fixedpoint32& operator = (const int8_t& _val) { val = ((int32_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE fixedpoint32& operator = (const uint8_t& _val) { val = ((int32_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE fixedpoint32& operator = (const int16_t& _val) { val = ((int32_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE fixedpoint32& operator = (const cv::softdouble& _val) { val = (int32_t)cvRound(_val * cv::softdouble((1 << fixedShift))); return *this; }
    CV_ALWAYS_INLINE fixedpoint32& operator = (const fixedpoint32& _val) { val = _val.val; return *this; }
    template <typename ET>
    CV_ALWAYS_INLINE fixedpoint32 operator * (const ET& val2) const { return val * val2; } // Wrong rounding is possible for floating point types
    CV_ALWAYS_INLINE fixedpoint64 operator * (const fixedpoint32& val2) const { return (int64_t)val * (int64_t)(val2.val); }
    CV_ALWAYS_INLINE fixedpoint32 operator + (const fixedpoint32& val2) const { return fixedpoint32(val + val2.val); }
    CV_ALWAYS_INLINE fixedpoint32 operator - (const fixedpoint32& val2) const { return fixedpoint32(val - val2.val); }
    //    CV_ALWAYS_INLINE fixedpoint32 operator + (const fixedpoint32& val2) const
    //    {
    //        int32_t nfrac = (int32_t)frac + val2.frac;
    //        int32_t nval = (int32_t)val + val2.val + nfrac >> 32;
    //        return nval > MAXINT32 ? beConv(MAXINT32, MAXINT32) : beConv((int32_t)(nval), 0);
    //    }
    CV_ALWAYS_INLINE fixedpoint32 operator >> (int n) const { return fixedpoint32(val >> n); }
    CV_ALWAYS_INLINE fixedpoint32 operator << (int n) const { return fixedpoint32(val << n); }
    template <typename ET>
    CV_ALWAYS_INLINE operator ET() const { return cv::saturate_cast<ET>((int32_t)fixedround((uint32_t)val) >> fixedShift); }
    CV_ALWAYS_INLINE operator double() const { return (double)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE operator float() const { return (float)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE bool isZero() { return val == 0; }
    static CV_ALWAYS_INLINE fixedpoint32 zero() { return fixedpoint32(); }
    static CV_ALWAYS_INLINE fixedpoint32 one() { return fixedpoint32((1 << fixedShift)); }
    friend class fixedpoint16;
};

class ufixedpoint32
{
private:
    static const int fixedShift = 16;

    uint32_t val;
    ufixedpoint32(uint32_t _val) : val(_val) {}
    static CV_ALWAYS_INLINE uint32_t fixedround(const uint32_t& _val) { return (_val + ((1 << fixedShift) >> 1)); }
public:
    typedef ufixedpoint64 WT;
    CV_ALWAYS_INLINE ufixedpoint32() { val = 0; }
    CV_ALWAYS_INLINE ufixedpoint32(const uint8_t& _val) { val = ((uint32_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE ufixedpoint32(const uint16_t& _val) { val = ((uint32_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE ufixedpoint32(const cv::softdouble& _val) { val = _val.getSign() ? 0 : (uint32_t)cvRound(_val * cv::softdouble((1 << fixedShift))); }
    CV_ALWAYS_INLINE ufixedpoint32& operator = (const uint8_t& _val) { val = ((uint32_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE ufixedpoint32& operator = (const uint16_t& _val) { val = ((uint32_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE ufixedpoint32& operator = (const cv::softdouble& _val) { val = _val.getSign() ? 0 : (uint32_t)cvRound(_val * cv::softdouble((1 << fixedShift))); return *this; }
    CV_ALWAYS_INLINE ufixedpoint32& operator = (const ufixedpoint32& _val) { val = _val.val; return *this; }
    template <typename ET>
    CV_ALWAYS_INLINE ufixedpoint32 operator * (const ET& val2) const { return val * val2; } // Wrong rounding is possible for floating point types
    CV_ALWAYS_INLINE ufixedpoint64 operator * (const ufixedpoint32& val2) const { return (uint64_t)val * (uint64_t)(val2.val); }
    CV_ALWAYS_INLINE ufixedpoint32 operator + (const ufixedpoint32& val2) const { return ufixedpoint32(val + val2.val); }
    CV_ALWAYS_INLINE ufixedpoint32 operator - (const ufixedpoint32& val2) const { return ufixedpoint32(val - val2.val); }
    //    CV_ALWAYS_INLINE fixedpoint32 operator + (const fixedpoint32& val2) const
    //    {
    //        int32_t nfrac = (int32_t)frac + val2.frac;
    //        int32_t nval = (int32_t)val + val2.val + nfrac >> 32;
    //        return nval > MAXINT32 ? beConv(MAXINT32, MAXINT32) : beConv((int32_t)(nval), 0);
    //    }
    CV_ALWAYS_INLINE ufixedpoint32 operator >> (int n) const { return ufixedpoint32(val >> n); }
    CV_ALWAYS_INLINE ufixedpoint32 operator << (int n) const { return ufixedpoint32(val << n); }
    template <typename ET>
    CV_ALWAYS_INLINE operator ET() const { return cv::saturate_cast<ET>(fixedround(val) >> fixedShift); }
    CV_ALWAYS_INLINE operator double() const { return (double)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE operator float() const { return (float)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE bool isZero() { return val == 0; }
    static CV_ALWAYS_INLINE ufixedpoint32 zero() { return ufixedpoint32(); }
    static CV_ALWAYS_INLINE ufixedpoint32 one() { return ufixedpoint32((1U << fixedShift)); }
    friend class ufixedpoint16;
};

class fixedpoint16
{
private:
    static const int fixedShift = 8;

    int16_t val;
    fixedpoint16(int16_t _val) : val(_val) {}
    static CV_ALWAYS_INLINE uint16_t fixedround(const uint16_t& _val) { return (_val + ((1 << fixedShift) >> 1)); }
public:
    typedef fixedpoint32 WT;
    CV_ALWAYS_INLINE fixedpoint16() { val = 0; }
    CV_ALWAYS_INLINE fixedpoint16(const int8_t& _val) { val = ((int16_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE fixedpoint16(const uint8_t& _val) { val = ((int16_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE fixedpoint16(const cv::softdouble& _val) { val = (int16_t)cvRound(_val * cv::softdouble((1 << fixedShift))); }
    CV_ALWAYS_INLINE fixedpoint16& operator = (const int8_t& _val) { val = ((int16_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE fixedpoint16& operator = (const cv::softdouble& _val) { val = (int16_t)cvRound(_val * cv::softdouble((1 << fixedShift))); return *this; }
    CV_ALWAYS_INLINE fixedpoint16& operator = (const fixedpoint16& _val) { val = _val.val; return *this; }
    template <typename ET>
    CV_ALWAYS_INLINE fixedpoint16 operator * (const ET& val2) const { return (int16_t)(val * val2); } // Wrong rounding is possible for floating point types
    CV_ALWAYS_INLINE fixedpoint32 operator * (const fixedpoint16& val2) const { return (int32_t)val * (int32_t)(val2.val); }
    CV_ALWAYS_INLINE fixedpoint16 operator + (const fixedpoint16& val2) const { return fixedpoint16((int16_t)(val + val2.val)); }
    CV_ALWAYS_INLINE fixedpoint16 operator - (const fixedpoint16& val2) const { return fixedpoint16((int16_t)(val - val2.val)); }
    CV_ALWAYS_INLINE fixedpoint16 operator >> (int n) const { return fixedpoint16((int16_t)(val >> n)); }
    CV_ALWAYS_INLINE fixedpoint16 operator << (int n) const { return fixedpoint16((int16_t)(val << n)); }
    template <typename ET>
    CV_ALWAYS_INLINE operator ET() const { return cv::saturate_cast<ET>((int16_t)fixedround((uint16_t)val) >> fixedShift); }
    CV_ALWAYS_INLINE operator double() const { return (double)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE operator float() const { return (float)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE bool isZero() { return val == 0; }
    static CV_ALWAYS_INLINE fixedpoint16 zero() { return fixedpoint16(); }
    static CV_ALWAYS_INLINE fixedpoint16 one() { return fixedpoint16((int16_t)(1 << fixedShift)); }
};

class ufixedpoint16
{
private:
    static const int fixedShift = 8;

    uint16_t val;
    ufixedpoint16(uint16_t _val) : val(_val) {}
    static CV_ALWAYS_INLINE uint16_t fixedround(const uint16_t& _val) { return (_val + ((1 << fixedShift) >> 1)); }
public:
    typedef ufixedpoint32 WT;
    CV_ALWAYS_INLINE ufixedpoint16() { val = 0; }
    CV_ALWAYS_INLINE ufixedpoint16(const uint8_t& _val) { val = ((uint16_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE ufixedpoint16(const cv::softdouble& _val) { val = _val.getSign() ? 0 : (uint16_t)cvRound(_val * cv::softdouble((int32_t)(1 << fixedShift))); }
    CV_ALWAYS_INLINE ufixedpoint16& operator = (const uint8_t& _val) { val = ((uint16_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE ufixedpoint16& operator = (const cv::softdouble& _val) { val = _val.getSign() ? 0 : (uint16_t)cvRound(_val * cv::softdouble((int32_t)(1 << fixedShift))); return *this; }
    CV_ALWAYS_INLINE ufixedpoint16& operator = (const ufixedpoint16& _val) { val = _val.val; return *this; }
    template <typename ET>
    CV_ALWAYS_INLINE ufixedpoint16 operator * (const ET& val2) const { return (uint16_t)(val * val2); } // Wrong rounding is possible for floating point types
    CV_ALWAYS_INLINE ufixedpoint32 operator * (const ufixedpoint16& val2) const { return ((uint32_t)val * (uint32_t)(val2.val)); }
    CV_ALWAYS_INLINE ufixedpoint16 operator + (const ufixedpoint16& val2) const { return ufixedpoint16((uint16_t)(val + val2.val)); }
    CV_ALWAYS_INLINE ufixedpoint16 operator - (const ufixedpoint16& val2) const { return ufixedpoint16((uint16_t)(val - val2.val)); }
    CV_ALWAYS_INLINE ufixedpoint16 operator >> (int n) const { return ufixedpoint16((uint16_t)(val >> n)); }
    CV_ALWAYS_INLINE ufixedpoint16 operator << (int n) const { return ufixedpoint16((uint16_t)(val << n)); }
    template <typename ET>
    CV_ALWAYS_INLINE operator ET() const { return cv::saturate_cast<ET>(fixedround(val) >> fixedShift); }
    CV_ALWAYS_INLINE operator double() const { return (double)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE operator float() const { return (float)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE bool isZero() { return val == 0; }
    static CV_ALWAYS_INLINE ufixedpoint16 zero() { return ufixedpoint16(); }
    static CV_ALWAYS_INLINE ufixedpoint16 one() { return ufixedpoint16((uint16_t)(1 << fixedShift)); }
};

}

#endif