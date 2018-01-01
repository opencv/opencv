#pragma once

namespace cv
{

    template <typename T, typename DT>
    struct Cvt_SIMD
    {
        int operator() (const T *, DT *, int) const;
    };

    template <typename T, typename DT, typename WT>
    struct cvtScale_SIMD
    {
        int operator () (const T * t , DT * dt, int sz, WT, WT) const {
            Cvt_SIMD<T,DT> cvt;
            return cvt.operator()(t, dt, sz);
        }
    };

    float convertFp16toFp32SW(const float16& fp16);
    float16 convertFp32toFp16SW(float fp32);

namespace opt_AVX2
{
void cvtScale_s16s32f32Line_AVX2(const short* src, int* dst, float scale, float shift, int width);
}
namespace opt_SSE4_1
{
    int cvtScale_SIMD_u8u16f32_SSE41(const uchar * src, ushort * dst, int width, float scale, float shift);
    int cvtScale_SIMD_s8u16f32_SSE41(const schar * src, ushort * dst, int width, float scale, float shift);
    int cvtScale_SIMD_u16u16f32_SSE41(const ushort * src, ushort * dst, int width, float scale, float shift);
    int cvtScale_SIMD_s16u16f32_SSE41(const short * src, ushort * dst, int width, float scale, float shift);
    int cvtScale_SIMD_s32u16f32_SSE41(const int * src, ushort * dst, int width, float scale, float shift);
    int cvtScale_SIMD_f32u16f32_SSE41(const float * src, ushort * dst, int width, float scale, float shift);
    int cvtScale_SIMD_f64u16f32_SSE41(const double * src, ushort * dst, int width, float scale, float shift);
    int Cvt_SIMD_f64u16_SSE41(const double * src, ushort * dst, int width);
}
}