
namespace
{
float convertFp16SW(short fp16);
short convertFp16SW(float fp32);

#if !CV_FP16_TYPE
// const numbers for floating points format
const unsigned int kShiftSignificand    = 13;
const unsigned int kMaskFp16Significand = 0x3ff;
const unsigned int kBiasFp16Exponent    = 15;
const unsigned int kBiasFp32Exponent    = 127;
#endif

#if CV_FP16_TYPE
inline float convertFp16SW(short fp16)
{
    // Fp16 -> Fp32
    Cv16suf a;
    a.i = fp16;
    return (float)a.h;
}
#else
inline float convertFp16SW(short fp16)
{
    // Fp16 -> Fp32
    Cv16suf b;
    b.i = fp16;
    int exponent    = b.fmt.exponent - kBiasFp16Exponent;
    int significand = b.fmt.significand;

    Cv32suf a;
    a.i = 0;
    a.fmt.sign = b.fmt.sign; // sign bit
    if( exponent == 16 )
    {
        // Inf or NaN
        a.i = a.i | 0x7F800000;
        if( significand != 0 )
        {
            // NaN
#if defined(__x86_64__) || defined(_M_X64)
            // 64bit
            a.i = a.i | 0x7FC00000;
#endif
            a.fmt.significand = a.fmt.significand | (significand << kShiftSignificand);
        }
        return a.f;
    }
    else if ( exponent == -(int)kBiasFp16Exponent )
    {
        // subnormal in Fp16
        if( significand == 0 )
        {
            // zero
            return a.f;
        }
        else
        {
            int shift = -1;
            while( ( significand & 0x400 ) == 0 )
            {
                significand = significand << 1;
                shift++;
            }
            significand = significand & kMaskFp16Significand;
            exponent -= shift;
        }
    }

    a.fmt.exponent = (exponent+kBiasFp32Exponent);
    a.fmt.significand = significand << kShiftSignificand;
    return a.f;
}
#endif

#if CV_FP16_TYPE
inline short convertFp16SW(float fp32)
{
    // Fp32 -> Fp16
    Cv16suf a;
    a.h = (__fp16)fp32;
    return a.i;
}
#else
inline short convertFp16SW(float fp32)
{
    // Fp32 -> Fp16
    Cv32suf a;
    a.f = fp32;
    int exponent    = a.fmt.exponent - kBiasFp32Exponent;
    int significand = a.fmt.significand;

    Cv16suf result;
    result.i = 0;
    unsigned int absolute = a.i & 0x7fffffff;
    if( 0x477ff000 <= absolute )
    {
        // Inf in Fp16
        result.i = result.i | 0x7C00;
        if( exponent == 128 && significand != 0 )
        {
            // NaN
            result.i = (short)( result.i | 0x200 | ( significand >> kShiftSignificand ) );
        }
    }
    else if ( absolute < 0x33000001 )
    {
        // too small for fp16
        result.i = 0;
    }
    else if ( absolute < 0x387fe000 )
    {
        // subnormal in Fp16
        int fp16Significand = significand | 0x800000;
        int bitShift = (-exponent) - 1;
        fp16Significand = fp16Significand >> bitShift;

        // special cases to round up
        bitShift = exponent + 24;
        int threshold = ( ( 0x400000 >> bitShift ) | ( ( ( significand & ( 0x800000 >> bitShift ) ) >> ( 126 - a.fmt.exponent ) ) ^ 1 ) );
        if( absolute == 0x33c00000 )
        {
            result.i = 2;
        }
        else
        {
            if( threshold <= ( significand & ( 0xffffff >> ( exponent + 25 ) ) ) )
            {
                fp16Significand++;
            }
            result.i = (short)fp16Significand;
        }
    }
    else
    {
        // usual situation
        // exponent
        result.fmt.exponent = ( exponent + kBiasFp16Exponent );

        // significand;
        short fp16Significand = (short)(significand >> kShiftSignificand);
        result.fmt.significand = fp16Significand;

        // special cases to round up
        short lsb10bitsFp32 = (significand & 0x1fff);
        short threshold = 0x1000 + ( ( fp16Significand & 0x1 ) ? 0 : 1 );
        if( threshold <= lsb10bitsFp32 )
        {
            result.i++;
        }
        else if ( fp16Significand == kMaskFp16Significand && exponent == -15)
        {
            result.i++;
        }
    }

    // sign bit
    result.fmt.sign = a.fmt.sign;
    return result.i;
}
#endif

}

namespace cv
{
namespace opt_FP16
{
void cvtScaleHalf_SIMD32f16f( const float* src, size_t sstep, short* dst, size_t dstep, cv::Size size );
void cvtScaleHalf_SIMD16f32f( const short* src, size_t sstep, float* dst, size_t dstep, cv::Size size );
}
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