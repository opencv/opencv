// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.


#ifndef _CV_FIXEDPOINT_HPP_
#define _CV_FIXEDPOINT_HPP_

namespace {

class fixedpoint64
{
private:
    int64_t val;
    fixedpoint64(int64_t _val) : val(_val) {}
    static CV_ALWAYS_INLINE uint64_t fixedround(const uint64_t& _val) { return (_val + ((1LL << fixedShift) >> 1)); }
public:
    static const int fixedShift = 32;

    typedef fixedpoint64 WT;
    typedef int64_t raw_t;
    CV_ALWAYS_INLINE fixedpoint64() { val = 0; }
    CV_ALWAYS_INLINE fixedpoint64(const fixedpoint64& v) { val = v.val; }
    CV_ALWAYS_INLINE fixedpoint64(const int8_t& _val) { val = ((int64_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE fixedpoint64(const uint8_t& _val) { val = ((int64_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE fixedpoint64(const int16_t& _val) { val = ((int64_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE fixedpoint64(const uint16_t& _val) { val = ((int64_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE fixedpoint64(const int32_t& _val) { val = ((int64_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE fixedpoint64(const cv::softdouble& _val) { val = cvRound64(_val * cv::softdouble((int64_t)(1LL << fixedShift))); }
    CV_ALWAYS_INLINE fixedpoint64& operator = (const int8_t& _val) { val = ((int64_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE fixedpoint64& operator = (const uint8_t& _val) { val = ((int64_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE fixedpoint64& operator = (const int16_t& _val) { val = ((int64_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE fixedpoint64& operator = (const uint16_t& _val) { val = ((int64_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE fixedpoint64& operator = (const int32_t& _val) { val = ((int64_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE fixedpoint64& operator = (const cv::softdouble& _val) { val = cvRound64(_val * cv::softdouble((int64_t)(1LL << fixedShift))); return *this; }
    CV_ALWAYS_INLINE fixedpoint64& operator = (const fixedpoint64& _val) { val = _val.val; return *this; }
    CV_ALWAYS_INLINE fixedpoint64 operator * (const int8_t& val2) const { return operator *(fixedpoint64(val2)); }
    CV_ALWAYS_INLINE fixedpoint64 operator * (const uint8_t& val2) const { return operator *(fixedpoint64(val2)); }
    CV_ALWAYS_INLINE fixedpoint64 operator * (const int16_t& val2) const { return operator *(fixedpoint64(val2)); }
    CV_ALWAYS_INLINE fixedpoint64 operator * (const uint16_t& val2) const { return operator *(fixedpoint64(val2)); }
    CV_ALWAYS_INLINE fixedpoint64 operator * (const int32_t& val2) const { return operator *(fixedpoint64(val2)); }
    CV_ALWAYS_INLINE fixedpoint64 operator * (const fixedpoint64& val2) const
    {
        bool sign_val = val < 0;
        bool sign_mul = val2.val < 0;
        uint64_t uval = sign_val ? (uint64_t)(-val) : (uint64_t)val;
        uint64_t umul = sign_mul ? (uint64_t)(-val2.val) : (uint64_t)val2.val;
        bool ressign = sign_val ^ sign_mul;

        uint64_t sh0   = fixedround((uval & 0xFFFFFFFF) * (umul & 0xFFFFFFFF));
        uint64_t sh1_0 = (uval >> 32)        * (umul & 0xFFFFFFFF);
        uint64_t sh1_1 = (uval & 0xFFFFFFFF) * (umul >> 32);
        uint64_t sh2   = (uval >> 32)        * (umul >> 32);
        uint64_t val0_l = (sh1_0 & 0xFFFFFFFF) + (sh1_1 & 0xFFFFFFFF) + (sh0 >> 32);
        uint64_t val0_h = (sh2   & 0xFFFFFFFF) + (sh1_0 >> 32) + (sh1_1 >> 32) + (val0_l >> 32);
        val0_l &= 0xFFFFFFFF;

        if (sh2 > CV_BIG_INT(0x7FFFFFFF) || val0_h > CV_BIG_INT(0x7FFFFFFF))
            return (int64_t)(ressign ? CV_BIG_UINT(0x8000000000000000) : CV_BIG_INT(0x7FFFFFFFFFFFFFFF));

        if (ressign)
        {
            return -(int64_t)(val0_h << 32 | val0_l);
        }
        return (int64_t)(val0_h << 32 | val0_l);
    }
    CV_ALWAYS_INLINE fixedpoint64 operator + (const fixedpoint64& val2) const
    {
        int64_t res = val + val2.val;
        return (int64_t)(((val ^ res) & (val2.val ^ res)) < 0 ? ~(res & CV_BIG_UINT(0x8000000000000000)) : res);
    }
    CV_ALWAYS_INLINE fixedpoint64 operator - (const fixedpoint64& val2) const
    {
        int64_t res = val - val2.val;
        return (int64_t)(((val ^ val2.val) & (val ^ res)) < 0 ? ~(res & CV_BIG_UINT(0x8000000000000000)) : res);
    }
    CV_ALWAYS_INLINE fixedpoint64 operator >> (int n) const { return fixedpoint64(val >> n); }
    CV_ALWAYS_INLINE fixedpoint64 operator << (int n) const { return fixedpoint64(val << n); }
    CV_ALWAYS_INLINE bool operator == (const fixedpoint64& val2) const { return val == val2.val; }
    template <typename ET>
    CV_ALWAYS_INLINE ET saturate_cast() const { return cv::saturate_cast<ET>((int64_t)fixedround((uint64_t)val) >> fixedShift); }
    CV_ALWAYS_INLINE operator double() const { return (double)val / (1LL << fixedShift); }
    CV_ALWAYS_INLINE operator float() const { return (float)val / (1LL << fixedShift); }
    CV_ALWAYS_INLINE operator uint8_t() const { return saturate_cast<uint8_t>(); }
    CV_ALWAYS_INLINE operator int8_t() const { return saturate_cast<int8_t>(); }
    CV_ALWAYS_INLINE operator uint16_t() const { return saturate_cast<uint16_t>(); }
    CV_ALWAYS_INLINE operator int16_t() const { return saturate_cast<int16_t>(); }
    CV_ALWAYS_INLINE operator int32_t() const { return saturate_cast<int32_t>(); }
    CV_ALWAYS_INLINE bool isZero() { return val == 0; }
    static CV_ALWAYS_INLINE fixedpoint64 zero() { return fixedpoint64(); }
    static CV_ALWAYS_INLINE fixedpoint64 one() { return fixedpoint64((int64_t)(1LL << fixedShift)); }
    friend class fixedpoint32;
};

class ufixedpoint64
{
private:
    uint64_t val;
    ufixedpoint64(uint64_t _val) : val(_val) {}
    static CV_ALWAYS_INLINE uint64_t fixedround(const uint64_t& _val) { return (_val + ((1LL << fixedShift) >> 1)); }
public:
    static const int fixedShift = 32;

    typedef ufixedpoint64 WT;
    typedef uint64_t raw_t;
    CV_ALWAYS_INLINE ufixedpoint64() { val = 0; }
    CV_ALWAYS_INLINE ufixedpoint64(const ufixedpoint64& v) { val = v.val; }
    CV_ALWAYS_INLINE ufixedpoint64(const uint8_t& _val) { val = ((uint64_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE ufixedpoint64(const uint16_t& _val) { val = ((uint64_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE ufixedpoint64(const uint32_t& _val) { val = ((uint64_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE ufixedpoint64(const cv::softdouble& _val) { val = _val.getSign() ? 0 : (uint64_t)cvRound64(_val * cv::softdouble((int64_t)(1LL << fixedShift))); }
    CV_ALWAYS_INLINE ufixedpoint64& operator = (const uint8_t& _val) { val = ((uint64_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE ufixedpoint64& operator = (const uint16_t& _val) { val = ((uint64_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE ufixedpoint64& operator = (const uint32_t& _val) { val = ((uint64_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE ufixedpoint64& operator = (const cv::softdouble& _val) { val = _val.getSign() ? 0 : (uint64_t)cvRound64(_val * cv::softdouble((int64_t)(1LL << fixedShift))); return *this; }
    CV_ALWAYS_INLINE ufixedpoint64& operator = (const ufixedpoint64& _val) { val = _val.val; return *this; }
    CV_ALWAYS_INLINE ufixedpoint64 operator * (const uint8_t& val2) const { return operator *(ufixedpoint64(val2)); }
    CV_ALWAYS_INLINE ufixedpoint64 operator * (const uint16_t& val2) const { return operator *(ufixedpoint64(val2)); }
    CV_ALWAYS_INLINE ufixedpoint64 operator * (const uint32_t& val2) const { return operator *(ufixedpoint64(val2)); }
    CV_ALWAYS_INLINE ufixedpoint64 operator * (const ufixedpoint64& val2) const
    {
        uint64_t sh0 = fixedround((val & 0xFFFFFFFF) * (val2.val & 0xFFFFFFFF));
        uint64_t sh1_0 = (val >> 32)        * (val2.val & 0xFFFFFFFF);
        uint64_t sh1_1 = (val & 0xFFFFFFFF) * (val2.val >> 32);
        uint64_t sh2   = (val >> 32)        * (val2.val >> 32);
        uint64_t val0_l = (sh1_0 & 0xFFFFFFFF) + (sh1_1 & 0xFFFFFFFF) + (sh0 >> 32);
        uint64_t val0_h = (sh2 & 0xFFFFFFFF) + (sh1_0 >> 32) + (sh1_1 >> 32) + (val0_l >> 32);
        val0_l &= 0xFFFFFFFF;

        if (sh2 > CV_BIG_INT(0xFFFFFFFF) || val0_h > CV_BIG_INT(0xFFFFFFFF))
            return (uint64_t)CV_BIG_UINT(0xFFFFFFFFFFFFFFFF);

        return (val0_h << 32 | val0_l);
    }
    CV_ALWAYS_INLINE ufixedpoint64 operator + (const ufixedpoint64& val2) const
    {
        uint64_t res = val + val2.val;
        return (uint64_t)((val > res) ? CV_BIG_UINT(0xFFFFFFFFFFFFFFFF) : res);
    }
    CV_ALWAYS_INLINE ufixedpoint64 operator - (const ufixedpoint64& val2) const
    {
        return val > val2.val ? (val - val2.val) : 0;
    }
    CV_ALWAYS_INLINE ufixedpoint64 operator >> (int n) const { return ufixedpoint64(val >> n); }
    CV_ALWAYS_INLINE ufixedpoint64 operator << (int n) const { return ufixedpoint64(val << n); }
    CV_ALWAYS_INLINE bool operator == (const ufixedpoint64& val2) const { return val == val2.val; }
    template <typename ET>
    CV_ALWAYS_INLINE ET saturate_cast() const { return cv::saturate_cast<ET>(fixedround(val) >> fixedShift); }
    CV_ALWAYS_INLINE operator double() const { return (double)val / (1LL << fixedShift); }
    CV_ALWAYS_INLINE operator float() const { return (float)val / (1LL << fixedShift); }
    CV_ALWAYS_INLINE operator uint8_t() const { return saturate_cast<uint8_t>(); }
    CV_ALWAYS_INLINE operator int8_t() const { return saturate_cast<int8_t>(); }
    CV_ALWAYS_INLINE operator uint16_t() const { return saturate_cast<uint16_t>(); }
    CV_ALWAYS_INLINE operator int16_t() const { return saturate_cast<int16_t>(); }
    CV_ALWAYS_INLINE operator int32_t() const { return saturate_cast<int32_t>(); }
    CV_ALWAYS_INLINE bool isZero() { return val == 0; }
    static CV_ALWAYS_INLINE ufixedpoint64 zero() { return ufixedpoint64(); }
    static CV_ALWAYS_INLINE ufixedpoint64 one() { return ufixedpoint64((uint64_t)(1ULL << fixedShift)); }

    static CV_ALWAYS_INLINE ufixedpoint64 fromRaw(uint64_t v) { return ufixedpoint64(v); }
    CV_ALWAYS_INLINE uint64_t raw() { return val; }
    CV_ALWAYS_INLINE uint32_t cvFloor() const { return cv::saturate_cast<uint32_t>(val >> fixedShift); }
    friend class ufixedpoint32;
};

class fixedpoint32
{
private:
    int32_t val;
    fixedpoint32(int32_t _val) : val(_val) {}
    static CV_ALWAYS_INLINE uint32_t fixedround(const uint32_t& _val) { return (_val + ((1 << fixedShift) >> 1)); }
public:
    static const int fixedShift = 16;

    typedef fixedpoint64 WT;
    typedef int32_t raw_t;
    CV_ALWAYS_INLINE fixedpoint32() { val = 0; }
    CV_ALWAYS_INLINE fixedpoint32(const fixedpoint32& v) { val = v.val; }
    CV_ALWAYS_INLINE fixedpoint32(const int8_t& _val) { val = ((int32_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE fixedpoint32(const uint8_t& _val) { val = ((int32_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE fixedpoint32(const int16_t& _val) { val = ((int32_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE fixedpoint32(const cv::softdouble& _val) { val = (int32_t)cvRound(_val * cv::softdouble((1 << fixedShift))); }
    CV_ALWAYS_INLINE fixedpoint32& operator = (const int8_t& _val) { val = ((int32_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE fixedpoint32& operator = (const uint8_t& _val) { val = ((int32_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE fixedpoint32& operator = (const int16_t& _val) { val = ((int32_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE fixedpoint32& operator = (const cv::softdouble& _val) { val = (int32_t)cvRound(_val * cv::softdouble((1 << fixedShift))); return *this; }
    CV_ALWAYS_INLINE fixedpoint32& operator = (const fixedpoint32& _val) { val = _val.val; return *this; }
    CV_ALWAYS_INLINE fixedpoint32 operator * (const int8_t& val2) const { return cv::saturate_cast<int32_t>((int64_t)val * val2); }
    CV_ALWAYS_INLINE fixedpoint32 operator * (const uint8_t& val2) const { return cv::saturate_cast<int32_t>((int64_t)val * val2); }
    CV_ALWAYS_INLINE fixedpoint32 operator * (const int16_t& val2) const { return cv::saturate_cast<int32_t>((int64_t)val * val2); }
    CV_ALWAYS_INLINE fixedpoint64 operator * (const fixedpoint32& val2) const { return (int64_t)val * (int64_t)(val2.val); }
    CV_ALWAYS_INLINE fixedpoint32 operator + (const fixedpoint32& val2) const
    {
        int32_t res = val + val2.val;
        return (int64_t)((val ^ res) & (val2.val ^ res)) >> 31 ? ~(res & ~0x7FFFFFFF) : res;
    }
    CV_ALWAYS_INLINE fixedpoint32 operator - (const fixedpoint32& val2) const
    {
        int32_t res = val - val2.val;
        return (int64_t)((val ^ val2.val) & (val ^ res)) >> 31 ? ~(res & ~0x7FFFFFFF) : res;
    }
    CV_ALWAYS_INLINE fixedpoint32 operator >> (int n) const { return fixedpoint32(val >> n); }
    CV_ALWAYS_INLINE fixedpoint32 operator << (int n) const { return fixedpoint32(val << n); }
    CV_ALWAYS_INLINE bool operator == (const fixedpoint32& val2) const { return val == val2.val; }
    template <typename ET>
    CV_ALWAYS_INLINE ET saturate_cast() const { return cv::saturate_cast<ET>((int32_t)fixedround((uint32_t)val) >> fixedShift); }
    CV_ALWAYS_INLINE operator fixedpoint64() const { return (int64_t)val << (fixedpoint64::fixedShift - fixedShift); }
    CV_ALWAYS_INLINE operator double() const { return (double)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE operator float() const { return (float)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE operator uint8_t() const { return saturate_cast<uint8_t>(); }
    CV_ALWAYS_INLINE operator int8_t() const { return saturate_cast<int8_t>(); }
    CV_ALWAYS_INLINE operator uint16_t() const { return saturate_cast<uint16_t>(); }
    CV_ALWAYS_INLINE operator int16_t() const { return saturate_cast<int16_t>(); }
    CV_ALWAYS_INLINE operator int32_t() const { return saturate_cast<int32_t>(); }
    CV_ALWAYS_INLINE bool isZero() { return val == 0; }
    static CV_ALWAYS_INLINE fixedpoint32 zero() { return fixedpoint32(); }
    static CV_ALWAYS_INLINE fixedpoint32 one() { return fixedpoint32((1 << fixedShift)); }
    friend class fixedpoint16;
};

class ufixedpoint32
{
private:
    uint32_t val;
    ufixedpoint32(uint32_t _val) : val(_val) {}
    static CV_ALWAYS_INLINE uint32_t fixedround(const uint32_t& _val) { return (_val + ((1 << fixedShift) >> 1)); }
public:
    static const int fixedShift = 16;

    typedef ufixedpoint64 WT;
    typedef uint32_t raw_t;
    CV_ALWAYS_INLINE ufixedpoint32() { val = 0; }
    CV_ALWAYS_INLINE ufixedpoint32(const ufixedpoint32& v) { val = v.val; }
    CV_ALWAYS_INLINE ufixedpoint32(const uint8_t& _val) { val = ((uint32_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE ufixedpoint32(const uint16_t& _val) { val = ((uint32_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE ufixedpoint32(const cv::softdouble& _val) { val = _val.getSign() ? 0 : (uint32_t)cvRound(_val * cv::softdouble((1 << fixedShift))); }
    CV_ALWAYS_INLINE ufixedpoint32& operator = (const uint8_t& _val) { val = ((uint32_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE ufixedpoint32& operator = (const uint16_t& _val) { val = ((uint32_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE ufixedpoint32& operator = (const cv::softdouble& _val) { val = _val.getSign() ? 0 : (uint32_t)cvRound(_val * cv::softdouble((1 << fixedShift))); return *this; }
    CV_ALWAYS_INLINE ufixedpoint32& operator = (const ufixedpoint32& _val) { val = _val.val; return *this; }
    CV_ALWAYS_INLINE ufixedpoint32 operator * (const uint8_t& val2) const { return cv::saturate_cast<uint32_t>((uint64_t)val * val2); }
    CV_ALWAYS_INLINE ufixedpoint32 operator * (const uint16_t& val2) const { return cv::saturate_cast<uint32_t>((uint64_t)val * val2); }
    CV_ALWAYS_INLINE ufixedpoint64 operator * (const ufixedpoint32& val2) const { return (uint64_t)val * (uint64_t)(val2.val); }
    CV_ALWAYS_INLINE ufixedpoint32 operator + (const ufixedpoint32& val2) const
    {
        uint32_t res = val + val2.val;
        return (val > res) ? 0xFFFFFFFF : res;
    }
    CV_ALWAYS_INLINE ufixedpoint32 operator - (const ufixedpoint32& val2) const
    {
        return val > val2.val ? (val - val2.val) : 0;
    }
    CV_ALWAYS_INLINE ufixedpoint32 operator >> (int n) const { return ufixedpoint32(val >> n); }
    CV_ALWAYS_INLINE ufixedpoint32 operator << (int n) const { return ufixedpoint32(val << n); }
    CV_ALWAYS_INLINE bool operator == (const ufixedpoint32& val2) const { return val == val2.val; }
    template <typename ET>
    CV_ALWAYS_INLINE ET saturate_cast() const { return cv::saturate_cast<ET>(fixedround(val) >> fixedShift); }
    CV_ALWAYS_INLINE operator ufixedpoint64() const { return (uint64_t)val << (ufixedpoint64::fixedShift - fixedShift); }
    CV_ALWAYS_INLINE operator double() const { return (double)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE operator float() const { return (float)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE operator uint8_t() const { return saturate_cast<uint8_t>(); }
    CV_ALWAYS_INLINE operator int8_t() const { return saturate_cast<int8_t>(); }
    CV_ALWAYS_INLINE operator uint16_t() const { return saturate_cast<uint16_t>(); }
    CV_ALWAYS_INLINE operator int16_t() const { return saturate_cast<int16_t>(); }
    CV_ALWAYS_INLINE operator int32_t() const { return saturate_cast<int32_t>(); }
    CV_ALWAYS_INLINE bool isZero() { return val == 0; }
    static CV_ALWAYS_INLINE ufixedpoint32 zero() { return ufixedpoint32(); }
    static CV_ALWAYS_INLINE ufixedpoint32 one() { return ufixedpoint32((1U << fixedShift)); }

    static CV_ALWAYS_INLINE ufixedpoint32 fromRaw(uint32_t v) { return ufixedpoint32(v); }
    CV_ALWAYS_INLINE uint32_t raw() { return val; }
    friend class ufixedpoint16;
};

class fixedpoint16
{
private:
    int16_t val;
    fixedpoint16(int16_t _val) : val(_val) {}
    static CV_ALWAYS_INLINE uint16_t fixedround(const uint16_t& _val) { return (_val + ((1 << fixedShift) >> 1)); }
public:
    static const int fixedShift = 8;

    typedef fixedpoint32 WT;
    typedef int16_t raw_t;
    CV_ALWAYS_INLINE fixedpoint16() { val = 0; }
    CV_ALWAYS_INLINE fixedpoint16(const fixedpoint16& v) { val = v.val; }
    CV_ALWAYS_INLINE fixedpoint16(const int8_t& _val) { val = ((int16_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE fixedpoint16(const cv::softdouble& _val) { val = (int16_t)cvRound(_val * cv::softdouble((1 << fixedShift))); }
    CV_ALWAYS_INLINE fixedpoint16& operator = (const int8_t& _val) { val = ((int16_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE fixedpoint16& operator = (const cv::softdouble& _val) { val = (int16_t)cvRound(_val * cv::softdouble((1 << fixedShift))); return *this; }
    CV_ALWAYS_INLINE fixedpoint16& operator = (const fixedpoint16& _val) { val = _val.val; return *this; }
    CV_ALWAYS_INLINE fixedpoint16 operator * (const int8_t& val2) const { return cv::saturate_cast<int16_t>((int32_t)val * val2); }
    CV_ALWAYS_INLINE fixedpoint32 operator * (const fixedpoint16& val2) const { return (int32_t)val * (int32_t)(val2.val); }
    CV_ALWAYS_INLINE fixedpoint16 operator + (const fixedpoint16& val2) const
    {
        int16_t res = val + val2.val;
        return ((val ^ res) & (val2.val ^ res)) >> 15 ? (int16_t)(~(res & ~0x7FFF)) : res;
    }
    CV_ALWAYS_INLINE fixedpoint16 operator - (const fixedpoint16& val2) const
    {
        int16_t res = val - val2.val;
        return ((val ^ val2.val) & (val ^ res)) >> 15 ? (int16_t)(~(res & ~(int16_t)0x7FFF)) : res;
    }
    CV_ALWAYS_INLINE fixedpoint16 operator >> (int n) const { return fixedpoint16((int16_t)(val >> n)); }
    CV_ALWAYS_INLINE fixedpoint16 operator << (int n) const { return fixedpoint16((int16_t)(val << n)); }
    CV_ALWAYS_INLINE bool operator == (const fixedpoint16& val2) const { return val == val2.val; }
    template <typename ET>
    CV_ALWAYS_INLINE ET saturate_cast() const { return cv::saturate_cast<ET>((int16_t)fixedround((uint16_t)val) >> fixedShift); }
    CV_ALWAYS_INLINE operator fixedpoint32() const { return (int32_t)val << (fixedpoint32::fixedShift - fixedShift); }
    CV_ALWAYS_INLINE operator double() const { return (double)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE operator float() const { return (float)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE operator uint8_t() const { return saturate_cast<uint8_t>(); }
    CV_ALWAYS_INLINE operator int8_t() const { return saturate_cast<int8_t>(); }
    CV_ALWAYS_INLINE operator uint16_t() const { return saturate_cast<uint16_t>(); }
    CV_ALWAYS_INLINE operator int16_t() const { return saturate_cast<int16_t>(); }
    CV_ALWAYS_INLINE operator int32_t() const { return saturate_cast<int32_t>(); }
    CV_ALWAYS_INLINE bool isZero() { return val == 0; }
    static CV_ALWAYS_INLINE fixedpoint16 zero() { return fixedpoint16(); }
    static CV_ALWAYS_INLINE fixedpoint16 one() { return fixedpoint16((int16_t)(1 << fixedShift)); }
};

class ufixedpoint16
{
private:
    uint16_t val;
    ufixedpoint16(uint16_t _val) : val(_val) {}
    static CV_ALWAYS_INLINE uint16_t fixedround(const uint16_t& _val) { return (_val + ((1 << fixedShift) >> 1)); }
public:
    static const int fixedShift = 8;

    typedef ufixedpoint32 WT;
    typedef uint16_t raw_t;
    CV_ALWAYS_INLINE ufixedpoint16() { val = 0; }
    CV_ALWAYS_INLINE ufixedpoint16(const ufixedpoint16& v) { val = v.val; }
    CV_ALWAYS_INLINE ufixedpoint16(const uint8_t& _val) { val = ((uint16_t)_val) << fixedShift; }
    CV_ALWAYS_INLINE ufixedpoint16(const cv::softdouble& _val) { val = _val.getSign() ? 0 : (uint16_t)cvRound(_val * cv::softdouble((int32_t)(1 << fixedShift))); }
    CV_ALWAYS_INLINE ufixedpoint16& operator = (const uint8_t& _val) { val = ((uint16_t)_val) << fixedShift; return *this; }
    CV_ALWAYS_INLINE ufixedpoint16& operator = (const cv::softdouble& _val) { val = _val.getSign() ? 0 : (uint16_t)cvRound(_val * cv::softdouble((int32_t)(1 << fixedShift))); return *this; }
    CV_ALWAYS_INLINE ufixedpoint16& operator = (const ufixedpoint16& _val) { val = _val.val; return *this; }
    CV_ALWAYS_INLINE ufixedpoint16 operator * (const uint8_t& val2) const { return cv::saturate_cast<uint16_t>((uint32_t)val * val2); }
    CV_ALWAYS_INLINE ufixedpoint32 operator * (const ufixedpoint16& val2) const { return ((uint32_t)val * (uint32_t)(val2.val)); }
    CV_ALWAYS_INLINE ufixedpoint16 operator + (const ufixedpoint16& val2) const
    {
        uint16_t res = val + val2.val;
        return (val > res) ? (uint16_t)0xFFFF : res;
    }
    CV_ALWAYS_INLINE ufixedpoint16 operator - (const ufixedpoint16& val2) const
    {
        return val > val2.val ? (uint16_t)(val - val2.val) : (uint16_t)0;
    }
    CV_ALWAYS_INLINE ufixedpoint16 operator >> (int n) const { return ufixedpoint16((uint16_t)(val >> n)); }
    CV_ALWAYS_INLINE ufixedpoint16 operator << (int n) const { return ufixedpoint16((uint16_t)(val << n)); }
    CV_ALWAYS_INLINE bool operator == (const ufixedpoint16& val2) const { return val == val2.val; }
    template <typename ET>
    CV_ALWAYS_INLINE ET saturate_cast() const { return cv::saturate_cast<ET>(fixedround(val) >> fixedShift); }
    CV_ALWAYS_INLINE operator ufixedpoint32() const { return (uint32_t)val << (ufixedpoint32::fixedShift - fixedShift); }
    CV_ALWAYS_INLINE operator double() const { return (double)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE operator float() const { return (float)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE operator uint8_t() const { return saturate_cast<uint8_t>(); }
    CV_ALWAYS_INLINE operator int8_t() const { return saturate_cast<int8_t>(); }
    CV_ALWAYS_INLINE operator uint16_t() const { return saturate_cast<uint16_t>(); }
    CV_ALWAYS_INLINE operator int16_t() const { return saturate_cast<int16_t>(); }
    CV_ALWAYS_INLINE operator int32_t() const { return saturate_cast<int32_t>(); }
    CV_ALWAYS_INLINE bool isZero() { return val == 0; }
    static CV_ALWAYS_INLINE ufixedpoint16 zero() { return ufixedpoint16(); }
    static CV_ALWAYS_INLINE ufixedpoint16 one() { return ufixedpoint16((uint16_t)(1 << fixedShift)); }

    static CV_ALWAYS_INLINE ufixedpoint16 fromRaw(uint16_t v) { return ufixedpoint16(v); }
    CV_ALWAYS_INLINE uint16_t raw() const { return val; }
};

}

#endif
