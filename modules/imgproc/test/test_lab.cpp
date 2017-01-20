/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

#include "opencv2/core.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/sse_utils.hpp"

#include "opencv2/core/softfloat.hpp"

using namespace cv;
using namespace std;

// computes cubic spline coefficients for a function: (xi=i, yi=f[i]), i=0..n
template<typename _Tp> static void splineBuild(const _Tp* f, int n, _Tp* tab)
{
    _Tp cn = 0;
    int i;
    tab[0] = tab[1] = (_Tp)0;

    for(i = 1; i < n-1; i++)
    {
        _Tp t = 3*(f[i+1] - 2*f[i] + f[i-1]);
        _Tp l = 1/(4 - tab[(i-1)*4]);
        tab[i*4] = l; tab[i*4+1] = (t - tab[(i-1)*4+1])*l;
    }

    for(i = n-1; i >= 0; i--)
    {
        _Tp c = tab[i*4+1] - tab[i*4]*cn;
        _Tp b = f[i+1] - f[i] - (cn + c*2)*(_Tp)0.3333333333333333;
        _Tp d = (cn - c)*(_Tp)0.3333333333333333;
        tab[i*4] = f[i]; tab[i*4+1] = b;
        tab[i*4+2] = c; tab[i*4+3] = d;
        cn = c;
    }
}

static void splineBuild(const softfloat* f, int n, float* tab)
{
    const softfloat f2(2), f3(3), f4(4);
    softfloat cn(0);
    softfloat* sftab = reinterpret_cast<softfloat*>(tab);
    int i;
    tab[0] = tab[1] = 0.0f;

    for(i = 1; i < n-1; i++)
    {
        softfloat t = (f[i+1] - f[i]*f2 + f[i-1])*f3;
        softfloat l = softfloat::one()/(f4 - sftab[(i-1)*4]);
        sftab[i*4] = l; sftab[i*4+1] = (t - sftab[(i-1)*4+1])*l;
    }

    for(i = n-1; i >= 0; i--)
    {
        softfloat c = sftab[i*4+1] - sftab[i*4]*cn;
        softfloat b = f[i+1] - f[i] - (cn + c*f2)/f3;
        softfloat d = (cn - c)/f3;
        sftab[i*4] = f[i]; sftab[i*4+1] = b;
        sftab[i*4+2] = c; sftab[i*4+3] = d;
        cn = c;
    }
}

// interpolates value of a function at x, 0 <= x <= n using a cubic spline.
template<typename _Tp> static inline _Tp splineInterpolate(_Tp x, const _Tp* tab, int n)
{
    // don't touch this function without urgent need - some versions of gcc fail to inline it correctly
    int ix = std::min(std::max(int(x), 0), n-1);
    x -= ix;
    tab += ix*4;
    return ((tab[3]*x + tab[2])*x + tab[1])*x + tab[0];
}

#if CV_NEON
template<typename _Tp> static inline void splineInterpolate(float32x4_t& v_x, const _Tp* tab, int n)
{
    int32x4_t v_ix = vcvtq_s32_f32(vminq_f32(vmaxq_f32(v_x, vdupq_n_f32(0)), vdupq_n_f32(n - 1)));
    v_x = vsubq_f32(v_x, vcvtq_f32_s32(v_ix));
    v_ix = vshlq_n_s32(v_ix, 2);

    int CV_DECL_ALIGNED(16) ix[4];
    vst1q_s32(ix, v_ix);

    float32x4_t v_tab0 = vld1q_f32(tab + ix[0]);
    float32x4_t v_tab1 = vld1q_f32(tab + ix[1]);
    float32x4_t v_tab2 = vld1q_f32(tab + ix[2]);
    float32x4_t v_tab3 = vld1q_f32(tab + ix[3]);

    float32x4x2_t v01 = vtrnq_f32(v_tab0, v_tab1);
    float32x4x2_t v23 = vtrnq_f32(v_tab2, v_tab3);

    v_tab0 = vcombine_f32(vget_low_f32(v01.val[0]), vget_low_f32(v23.val[0]));
    v_tab1 = vcombine_f32(vget_low_f32(v01.val[1]), vget_low_f32(v23.val[1]));
    v_tab2 = vcombine_f32(vget_high_f32(v01.val[0]), vget_high_f32(v23.val[0]));
    v_tab3 = vcombine_f32(vget_high_f32(v01.val[1]), vget_high_f32(v23.val[1]));

    v_x = vmlaq_f32(v_tab0, vmlaq_f32(v_tab1, vmlaq_f32(v_tab2, v_tab3, v_x), v_x), v_x);
}
#elif CV_SSE2
template<typename _Tp> static inline void splineInterpolate(__m128& v_x, const _Tp* tab, int n)
{
    __m128i v_ix = _mm_cvttps_epi32(_mm_min_ps(_mm_max_ps(v_x, _mm_setzero_ps()), _mm_set1_ps(float(n - 1))));
    v_x = _mm_sub_ps(v_x, _mm_cvtepi32_ps(v_ix));
    v_ix = _mm_slli_epi32(v_ix, 2);

    int CV_DECL_ALIGNED(16) ix[4];
    _mm_store_si128((__m128i *)ix, v_ix);

    __m128 v_tab0 = _mm_loadu_ps(tab + ix[0]);
    __m128 v_tab1 = _mm_loadu_ps(tab + ix[1]);
    __m128 v_tab2 = _mm_loadu_ps(tab + ix[2]);
    __m128 v_tab3 = _mm_loadu_ps(tab + ix[3]);

    __m128 v_tmp0 = _mm_unpacklo_ps(v_tab0, v_tab1);
    __m128 v_tmp1 = _mm_unpacklo_ps(v_tab2, v_tab3);
    __m128 v_tmp2 = _mm_unpackhi_ps(v_tab0, v_tab1);
    __m128 v_tmp3 = _mm_unpackhi_ps(v_tab2, v_tab3);

    v_tab0 = _mm_shuffle_ps(v_tmp0, v_tmp1, 0x44);
    v_tab2 = _mm_shuffle_ps(v_tmp2, v_tmp3, 0x44);
    v_tab1 = _mm_shuffle_ps(v_tmp0, v_tmp1, 0xee);
    v_tab3 = _mm_shuffle_ps(v_tmp2, v_tmp3, 0xee);

    __m128 v_l = _mm_mul_ps(v_x, v_tab3);
    v_l = _mm_add_ps(v_l, v_tab2);
    v_l = _mm_mul_ps(v_l, v_x);
    v_l = _mm_add_ps(v_l, v_tab1);
    v_l = _mm_mul_ps(v_l, v_x);
    v_x = _mm_add_ps(v_l, v_tab0);
}
#endif

template<typename _Tp> struct ColorChannel
{
    typedef float worktype_f;
    static _Tp max() { return std::numeric_limits<_Tp>::max(); }
    static _Tp half() { return (_Tp)(max()/2 + 1); }
};

template<> struct ColorChannel<float>
{
    typedef float worktype_f;
    static float max() { return 1.f; }
    static float half() { return 0.5f; }
};

///////////

static const float sRGB2XYZ_D65[] =
{
    0.412453f, 0.357580f, 0.180423f,
    0.212671f, 0.715160f, 0.072169f,
    0.019334f, 0.119193f, 0.950227f
};

static const float XYZ2sRGB_D65[] =
{
    3.240479f, -1.53715f, -0.498535f,
   -0.969256f,  1.875991f, 0.041556f,
    0.055648f, -0.204043f, 1.057311f
};

enum
{
    yuv_shift = 14,
    xyz_shift = 12,
    R2Y = 4899, // == R2YF*16384
    G2Y = 9617, // == G2YF*16384
    B2Y = 1868, // == B2YF*16384
    BLOCK_SIZE = 256
};

#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

static const double _1_3 = 0.333333333333;
static const float _1_3f = static_cast<float>(_1_3);

///////////////////////////////////// RGB <-> L*a*b* /////////////////////////////////////

static const float D65[] = { 0.950456f, 1.f, 1.088754f };

enum { LAB_CBRT_TAB_SIZE = 1024, GAMMA_TAB_SIZE = 1024 };
static float LabCbrtTab[LAB_CBRT_TAB_SIZE*4];
static const float LabCbrtTabScale = softfloat(LAB_CBRT_TAB_SIZE*2)/softfloat(3);

static float sRGBGammaTab[GAMMA_TAB_SIZE*4], sRGBInvGammaTab[GAMMA_TAB_SIZE*4];
static const float GammaTabScale((int)GAMMA_TAB_SIZE);

static ushort sRGBGammaTab_b[256], linearGammaTab_b[256];
enum { inv_gamma_shift = 12, INV_GAMMA_TAB_SIZE = (1 << inv_gamma_shift) };
static ushort sRGBInvGammaTab_b[INV_GAMMA_TAB_SIZE], linearInvGammaTab_b[INV_GAMMA_TAB_SIZE];
#undef lab_shift
#define lab_shift xyz_shift
#define gamma_shift 3
#define lab_shift2 (lab_shift + gamma_shift)
#define LAB_CBRT_TAB_SIZE_B (256*3/2*(1<<gamma_shift))
static ushort LabCbrtTab_b[LAB_CBRT_TAB_SIZE_B];

static bool enableBitExactness = true;
static bool enableRGB2LabInterpolation = true;
static bool enablePackedLab = true;
enum
{
    lab_lut_shift = 5,
    LAB_LUT_DIM = (1 << lab_lut_shift)+1,
    lab_base_shift = 14,
    LAB_BASE = (1 << lab_base_shift),
    trilinear_shift = 8 - lab_lut_shift + 1,
    TRILINEAR_BASE = (1 << trilinear_shift)
};
static int16_t RGB2LabLUT_s16[LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3*8];
static int16_t trilinearLUT[TRILINEAR_BASE*TRILINEAR_BASE*TRILINEAR_BASE*8];
static ushort LabToYF_b[256*2];
static const int minABvalue = -8145;
static int abToXZ_b[LAB_BASE*9/4];

#define clip(value) \
    value < 0.0f ? 0.0f : value > 1.0f ? 1.0f : value;

//TODO: remove
/// to compare with SF-version

static float LabCbrtTab_gold[LAB_CBRT_TAB_SIZE*4];

static float sRGBGammaTab_gold[GAMMA_TAB_SIZE*4], sRGBInvGammaTab_gold[GAMMA_TAB_SIZE*4];

static ushort sRGBGammaTab_b_gold[256], linearGammaTab_b_gold[256];

static ushort sRGBInvGammaTab_b_gold[INV_GAMMA_TAB_SIZE], linearInvGammaTab_b_gold[INV_GAMMA_TAB_SIZE];
static ushort LabCbrtTab_b_gold[LAB_CBRT_TAB_SIZE_B];

template<typename T> inline void compareAndPrint(T a, T b, int i, string s)
{
    if(false && a != b) //disabled
    {
        cout << s << " at i = " << i << " a = " << a << " b = " << b << " (a-b) = " << (a - b) << endl;
    }
}

static inline float applyGamma_gold(float x)
{
    return x <= 0.04045f ? x*(1.f/12.92f) : (float)std::pow((double)(x + 0.055)*(1./1.055), 2.4);
}

static inline float applyInvGamma_gold(float x)
{
    return x <= 0.0031308 ? x*12.92f : (float)(1.055*std::pow((double)x, 1./2.4) - 0.055);
}

// TODO: up to this

//all constants should be presented through integers to keep bit-exactness
static const softdouble gammaThreshold    = softdouble(809)/softdouble(20000);    //  0.04045
static const softdouble gammaInvThreshold = softdouble(7827)/softdouble(2500000); //  0.0031308
static const softdouble gammaLowScale     = softdouble(323)/softdouble(25);       // 12.92
static const softdouble gammaPower        = softdouble(12)/softdouble(5);         //  2.4
static const softdouble gammaXshift       = softdouble(11)/softdouble(200);       // 0.055

static inline softfloat applyGamma(softfloat x)
{
    //return x <= 0.04045f ? x*(1.f/12.92f) : (float)std::pow((double)(x + 0.055)*(1./1.055), 2.4);
    softdouble xd = x;
    return (xd <= gammaThreshold ?
                xd/gammaLowScale :
                pow((xd + gammaXshift)/(softdouble::one()+gammaXshift), gammaPower));
}

static inline softfloat applyInvGamma(softfloat x)
{
    //return x <= 0.0031308 ? x*12.92f : (float)(1.055*std::pow((double)x, 1./2.4) - 0.055);
    softdouble xd = x;
    return (xd <= gammaInvThreshold ?
                xd*gammaLowScale :
                pow(xd, softdouble::one()/gammaPower)*(softdouble::one()+gammaXshift) - gammaXshift);
}

static void initLabTabs()
{
    static bool initialized = false;
    if(!initialized)
    {
        static const softfloat lthresh = softfloat(216) / softfloat(24389); // 0.008856f = (6/29)^3
        static const softfloat lscale  = softfloat(841) / softfloat(108); // 7.787f = (29/3)^3/(29*4)
        static const softfloat lbias = softfloat(16) / softfloat(116);
        static const softfloat f255(255);

        softfloat f[LAB_CBRT_TAB_SIZE+1], g[GAMMA_TAB_SIZE+1], ig[GAMMA_TAB_SIZE+1];
        softfloat scale = softfloat::one()/softfloat(LabCbrtTabScale);
        int i;
        for(i = 0; i <= LAB_CBRT_TAB_SIZE; i++)
        {
            softfloat x = scale*softfloat(i);
            f[i] = x < lthresh ? mulAdd(x, lscale, lbias) : cbrt(x);
        }
        splineBuild(f, LAB_CBRT_TAB_SIZE, LabCbrtTab);

        //TODO: remove
        float f_gold[LAB_CBRT_TAB_SIZE+1], g_gold[GAMMA_TAB_SIZE+1], ig_gold[GAMMA_TAB_SIZE+1], scale_gold = 1.f/LabCbrtTabScale;
        for(i = 0; i <= LAB_CBRT_TAB_SIZE; i++)
        {
            float x = i*scale_gold;
            f_gold[i] = x < 0.008856f ? x*7.787f + 0.13793103448275862f : cvCbrt(x);

            compareAndPrint(softfloat(x), (scale*softfloat(i)), i, "LabCbrtTab x");
            compareAndPrint(softfloat(f_gold[i]), f[i], i, "LabCbrtTab f");
        }
        splineBuild(f_gold, LAB_CBRT_TAB_SIZE, LabCbrtTab_gold);

        for(i = 0; i < LAB_CBRT_TAB_SIZE*4; i++)
        {
            compareAndPrint(LabCbrtTab_gold[i], LabCbrtTab[i], i, "LabCbrtTab tab");
        }
        //TODO: up to this

        scale = softfloat::one()/softfloat(GammaTabScale);
        for(i = 0; i <= GAMMA_TAB_SIZE; i++)
        {
            softfloat x = scale*softfloat(i);
            g[i] = applyGamma(x);
            ig[i] = applyInvGamma(x);
        }
        splineBuild(g, GAMMA_TAB_SIZE, sRGBGammaTab);
        splineBuild(ig, GAMMA_TAB_SIZE, sRGBInvGammaTab);

        //TODO: remove
        scale_gold = 1.f/GammaTabScale;
        for(i = 0; i <= GAMMA_TAB_SIZE; i++)
        {
            float x = i*scale_gold;
            g_gold[i] = applyGamma_gold(x);
            ig_gold[i] = applyInvGamma_gold(x);

            compareAndPrint(softfloat(g_gold[i]), g[i], i, "sRGBGammaTab g");
            compareAndPrint(softfloat(ig_gold[i]), ig[i], i, "sRGBInvGammaTab ig");
        }
        splineBuild(g_gold, GAMMA_TAB_SIZE, sRGBGammaTab_gold);
        splineBuild(ig_gold, GAMMA_TAB_SIZE, sRGBInvGammaTab_gold);

        for(i = 0; i < GAMMA_TAB_SIZE*4; i++)
        {
            compareAndPrint( sRGBGammaTab_gold[i], sRGBGammaTab[i], i, "sRGBGammaTab tab" );
            compareAndPrint( sRGBInvGammaTab_gold[i], sRGBInvGammaTab[i], i, "sRGBInvGammaTab tab" );
        }
        //TODO: up to this

        static const softfloat intScale(255*(1 << gamma_shift));
        for(i = 0; i < 256; i++)
        {
            softfloat x = softfloat(i)/f255;
            sRGBGammaTab_b[i] = (ushort)(cvRound(intScale*applyGamma(x)));
            linearGammaTab_b[i] = (ushort)(i*(1 << gamma_shift));
        }
        static const softfloat invScale = softfloat::one()/softfloat((int)INV_GAMMA_TAB_SIZE);
        for(i = 0; i < INV_GAMMA_TAB_SIZE; i++)
        {
            softfloat x = invScale*softfloat(i);
            sRGBInvGammaTab_b[i] = (ushort)(cvRound(f255*applyInvGamma(x)));
            linearInvGammaTab_b[i] = (ushort)(cvTrunc(f255*x));
        }

        //TODO: remove
        for(i = 0; i < 256; i++)
        {
            float x = i*(1.f/255.f);
            sRGBGammaTab_b_gold[i] = saturate_cast<ushort>(255.f*(1 << gamma_shift)*applyGamma_gold(x));
            linearGammaTab_b_gold[i] = (ushort)(i*(1 << gamma_shift));

            compareAndPrint( sRGBGammaTab_b_gold[i], sRGBGammaTab_b[i], i, "sRGBGammaTab_b tab");
            compareAndPrint( linearGammaTab_b_gold[i], linearGammaTab_b[i], i, "linearGammaTab_b tab");
        }
        float invScale_gold = 1.f/INV_GAMMA_TAB_SIZE;
        for(i = 0; i < INV_GAMMA_TAB_SIZE; i++)
        {
            float x = i*invScale_gold;
            sRGBInvGammaTab_b_gold[i] = saturate_cast<ushort>(255.f*applyInvGamma_gold(x));
            linearInvGammaTab_b_gold[i] = (ushort)(255.f*x);

            compareAndPrint( sRGBInvGammaTab_b_gold[i], sRGBInvGammaTab_b[i], i, "sRGBInvGammaTab_b tab" );
            compareAndPrint( linearInvGammaTab_b_gold[i], linearInvGammaTab_b[i], i, "linearInvGammaTab_b tab" );
        }
        //TODO: up to this

        static const softfloat cbTabScale(softfloat::one()/(f255*(1 << gamma_shift)));
        static const softfloat lshift2(1 << lab_shift2);
        for(i = 0; i < LAB_CBRT_TAB_SIZE_B; i++)
        {
            softfloat x = cbTabScale*softfloat(i);
            LabCbrtTab_b[i] = (ushort)(cvRound(lshift2 * (x < lthresh ? mulAdd(x, lscale, lbias) : cbrt(x))));
        }

        //TODO: remove
        for(i = 0; i < LAB_CBRT_TAB_SIZE_B; i++)
        {
            float x = i*(1.f/(255.f*(1 << gamma_shift)));
            LabCbrtTab_b_gold[i] = saturate_cast<ushort>((1 << lab_shift2)*(x < 0.008856f ? x*7.787f + 0.13793103448275862f : cvCbrt(x)));

            compareAndPrint(LabCbrtTab_b_gold[i], LabCbrtTab_b[i], i, "LabCbrtTab_b tab" );
        }
        //TODO: up to this

        //Lookup table for L to y and ify calculations
        static const int BASE = (1 << 14);
        for(i = 0; i < 256; i++)
        {
            int y, ify;
            //8 * 255.0 / 100.0 == 20.4
            if( i <= 20)
            {
                //yy = li / 903.3f;
                //y = L*100/903.3f; 903.3f = (29/3)^3, 255 = 17*3*5
                y = cvRound(softfloat(i*BASE*20*9)/softfloat(17*29*29*29));
                //fy = 7.787f * yy + 16.0f / 116.0f; 7.787f = (29/3)^3/(29*4)
                ify = cvRound(softfloat(BASE)*(softfloat(16)/softfloat(116) + softfloat(i*5)/softfloat(3*17*29)));
            }
            else
            {
                //fy = (li + 16.0f) / 116.0f;
                softfloat fy = (softfloat(i*100*BASE)/softfloat(255*116) +
                                softfloat(16*BASE)/softfloat(116));
                ify = cvRound(fy);
                //yy = fy * fy * fy;
                y = cvRound(fy*fy*fy/softfloat(BASE*BASE));
            }

            LabToYF_b[i*2  ] = (ushort)y;   // 2260 <= y <= BASE
            LabToYF_b[i*2+1] = (ushort)ify; // 0 <= ify <= BASE
        }

        //Lookup table for a,b to x,z conversion
        for(i = minABvalue; i < LAB_BASE*9/4+minABvalue; i++)
        {
            int v;
            //6.f/29.f*BASE = 3389.730
            if(i <= 3390)
            {
                //fxz[k] = (fxz[k] - 16.0f / 116.0f) / 7.787f;
                // 7.787f = (29/3)^3/(29*4)
                v = i*108/841 - BASE*16/116*108/841;
            }
            else
            {
                //fxz[k] = fxz[k] * fxz[k] * fxz[k];
                v = i*i/BASE*i/BASE;
            }
            abToXZ_b[i-minABvalue] = v; // -1335 <= v <= 88231
        }

        if(enableRGB2LabInterpolation)
        {
            const float* _whitept = D65;
            softfloat coeffs[9];

            //RGB2Lab coeffs
            softfloat scaleWhite[] = { softfloat::one()/softfloat(_whitept[0]),
                                       softfloat::one(),
                                       softfloat::one()/softfloat(_whitept[2]) };

            for(i = 0; i < 3; i++ )
            {
                int j = i * 3;
                coeffs[j + 2] = scaleWhite[i] * softfloat(sRGB2XYZ_D65[j    ]);
                coeffs[j + 1] = scaleWhite[i] * softfloat(sRGB2XYZ_D65[j + 1]);
                coeffs[j + 0] = scaleWhite[i] * softfloat(sRGB2XYZ_D65[j + 2]);
            }

            softfloat D0 = coeffs[0], D1 = coeffs[1], D2 = coeffs[2],
                          D3 = coeffs[3], D4 = coeffs[4], D5 = coeffs[5],
                          D6 = coeffs[6], D7 = coeffs[7], D8 = coeffs[8];

            //TODO: remove
            static const float _a = 16.0f / 116.0f;
            float coeffs_gold[9];

            //RGB2Lab coeffs
            float scaleWhite_gold[] = { 1.0f / _whitept[0], 1.0f, 1.0f / _whitept[2] };

            for(i = 0; i < 3; i++ )
            {
                int j = i * 3;
                coeffs_gold[j + 2] = sRGB2XYZ_D65[j]     * scaleWhite_gold[i];
                coeffs_gold[j + 1] = sRGB2XYZ_D65[j + 1] * scaleWhite_gold[i];
                coeffs_gold[j + 0] = sRGB2XYZ_D65[j + 2] * scaleWhite_gold[i];

            }

            for(i = 0; i < 9; i++)
            {
                compareAndPrint( softfloat(coeffs_gold[i]), coeffs[i], i, "interp coeffs");
            }
            //TODO: up to this

            //903.3f = (29/3)^3
            static const softfloat lld(LAB_LUT_DIM - 1), f116(116), f16(16), f500(500), f200(200);
            static const softfloat f100(100), f128(128), f256(256), lbase((int)LAB_BASE);
            static const softfloat f9033 = softfloat(29*29*29)/softfloat(27);
            AutoBuffer<int16_t> RGB2Labprev(LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3);
            for(int p = 0; p < LAB_LUT_DIM; p++)
            {
                for(int q = 0; q < LAB_LUT_DIM; q++)
                {
                    for(int r = 0; r < LAB_LUT_DIM; r++)
                    {
                        //RGB 2 Lab LUT building
                        softfloat R = softfloat(p)/lld;
                        softfloat G = softfloat(q)/lld;
                        softfloat B = softfloat(r)/lld;

                        R = applyGamma(R);
                        G = applyGamma(G);
                        B = applyGamma(B);

                        softfloat X = R*D0 + G*D1 + B*D2;
                        softfloat Y = R*D3 + G*D4 + B*D5;
                        softfloat Z = R*D6 + G*D7 + B*D8;

                        softfloat FX = X > lthresh ? cbrt(X) : mulAdd(X, lscale, lbias);
                        softfloat FY = Y > lthresh ? cbrt(Y) : mulAdd(Y, lscale, lbias);
                        softfloat FZ = Z > lthresh ? cbrt(Z) : mulAdd(Z, lscale, lbias);

                        softfloat L = Y > lthresh ? (f116*FY - f16) : (f9033*Y);
                        softfloat a = f500 * (FX - FY);
                        softfloat b = f200 * (FY - FZ);

                        int idx = p*3 + q*LAB_LUT_DIM*3 + r*LAB_LUT_DIM*LAB_LUT_DIM*3;
                        RGB2Labprev[idx]   = (int16_t)(cvRound(lbase*L/f100));
                        RGB2Labprev[idx+1] = (int16_t)(cvRound(lbase*(a + f128)/f256));
                        RGB2Labprev[idx+2] = (int16_t)(cvRound(lbase*(b + f128)/f256));

                        //TODO: remove
                        //RGB 2 Lab LUT building
                        float R_gold = 1.0f*p/(LAB_LUT_DIM-1);
                        float G_gold = 1.0f*q/(LAB_LUT_DIM-1);
                        float B_gold = 1.0f*r/(LAB_LUT_DIM-1);

                        R_gold = applyGamma_gold(R_gold);
                        G_gold = applyGamma_gold(G_gold);
                        B_gold = applyGamma_gold(B_gold);

                        float X_gold = R_gold*coeffs_gold[0] + G_gold*coeffs_gold[1] + B_gold*coeffs_gold[2];
                        float Y_gold = R_gold*coeffs_gold[3] + G_gold*coeffs_gold[4] + B_gold*coeffs_gold[5];
                        float Z_gold = R_gold*coeffs_gold[6] + G_gold*coeffs_gold[7] + B_gold*coeffs_gold[8];

                        float FX_gold = X_gold > 0.008856f ? std::pow(X_gold, _1_3f) : (7.787f * X_gold + _a);
                        float FY_gold = Y_gold > 0.008856f ? std::pow(Y_gold, _1_3f) : (7.787f * Y_gold + _a);
                        float FZ_gold = Z_gold > 0.008856f ? std::pow(Z_gold, _1_3f) : (7.787f * Z_gold + _a);

                        float L_gold = Y_gold > 0.008856f ? (116.f * FY_gold - 16.f) : (903.3f * Y_gold);
                        float a_gold = 500.f * (FX_gold - FY_gold);
                        float b_gold = 200.f * (FY_gold - FZ_gold);

                        compareAndPrint( RGB2Labprev[idx  ], (int16_t)cvRound(LAB_BASE*L_gold/100.0f), idx, "RGB2Labprev L" );
                        compareAndPrint( RGB2Labprev[idx+1], (int16_t)cvRound(LAB_BASE*(a_gold+128.0f)/256.0f), idx, "RGB2Labprev a" );
                        compareAndPrint( RGB2Labprev[idx+2], (int16_t)cvRound(LAB_BASE*(b_gold+128.0f)/256.0f), idx, "RGB2Labprev b" );

                        //TODO: up to this
                    }
                }
            }
            for(int p = 0; p < LAB_LUT_DIM; p++)
            {
                for(int q = 0; q < LAB_LUT_DIM; q++)
                {
                    for(int r = 0; r < LAB_LUT_DIM; r++)
                    {
                        #define FILL(_p, _q, _r) \
                        do {\
                            int idxold = 0;\
                            idxold += min(p+(_p), (int)(LAB_LUT_DIM-1))*3;\
                            idxold += min(q+(_q), (int)(LAB_LUT_DIM-1))*LAB_LUT_DIM*3;\
                            idxold += min(r+(_r), (int)(LAB_LUT_DIM-1))*LAB_LUT_DIM*LAB_LUT_DIM*3;\
                            int idxnew = p*3*8 + q*LAB_LUT_DIM*3*8 + r*LAB_LUT_DIM*LAB_LUT_DIM*3*8+4*(_p)+2*(_q)+(_r);\
                            RGB2LabLUT_s16[idxnew]    = RGB2Labprev[idxold];\
                            RGB2LabLUT_s16[idxnew+8]  = RGB2Labprev[idxold+1];\
                            RGB2LabLUT_s16[idxnew+16] = RGB2Labprev[idxold+2];\
                        } while(0)

                        FILL(0, 0, 0); FILL(0, 0, 1);
                        FILL(0, 1, 0); FILL(0, 1, 1);
                        FILL(1, 0, 0); FILL(1, 0, 1);
                        FILL(1, 1, 0); FILL(1, 1, 1);

                        #undef FILL
                    }
                }
            }

            for(int16_t p = 0; p < TRILINEAR_BASE; p++)
            {
                int16_t pp = TRILINEAR_BASE - p;
                for(int16_t q = 0; q < TRILINEAR_BASE; q++)
                {
                    int16_t qq = TRILINEAR_BASE - q;
                    for(int16_t r = 0; r < TRILINEAR_BASE; r++)
                    {
                        int16_t rr = TRILINEAR_BASE - r;
                        int16_t* w = &trilinearLUT[8*p + 8*TRILINEAR_BASE*q + 8*TRILINEAR_BASE*TRILINEAR_BASE*r];
                        w[0]  = pp * qq * rr; w[1]  = pp * qq * r ; w[2]  = pp * q  * rr; w[3]  = pp * q  * r ;
                        w[4]  = p  * qq * rr; w[5]  = p  * qq * r ; w[6]  = p  * q  * rr; w[7]  = p  * q  * r ;
                    }
                }
            }
        }

        initialized = true;
    }
}

// cx, cy, cz are in [0; LAB_BASE]
static inline void trilinearInterpolate(int cx, int cy, int cz, int16_t* LUT,
                                        int& a, int& b, int& c)
{
    //LUT idx of origin pt of cube
    int tx = cx >> (lab_base_shift - lab_lut_shift);
    int ty = cy >> (lab_base_shift - lab_lut_shift);
    int tz = cz >> (lab_base_shift - lab_lut_shift);

    int16_t* baseLUT = &LUT[3*8*tx + (3*8*LAB_LUT_DIM)*ty + (3*8*LAB_LUT_DIM*LAB_LUT_DIM)*tz];
    int aa[8], bb[8], cc[8];
    for(int i = 0; i < 8; i++)
    {
        aa[i] = baseLUT[i]; bb[i] = baseLUT[i+8]; cc[i] = baseLUT[i+16];
    }

    //x, y, z are [0; TRILINEAR_BASE)
    static const int bitMask = (1 << trilinear_shift) - 1;
    int x = (cx >> (lab_base_shift - 8 - 1)) & bitMask;
    int y = (cy >> (lab_base_shift - 8 - 1)) & bitMask;
    int z = (cz >> (lab_base_shift - 8 - 1)) & bitMask;

    int w[8];
    for(int i = 0; i < 8; i++)
    {
        w[i] = trilinearLUT[8*x + 8*TRILINEAR_BASE*y + 8*TRILINEAR_BASE*TRILINEAR_BASE*z + i];
    }

    a = aa[0]*w[0]+aa[1]*w[1]+aa[2]*w[2]+aa[3]*w[3]+aa[4]*w[4]+aa[5]*w[5]+aa[6]*w[6]+aa[7]*w[7];
    b = bb[0]*w[0]+bb[1]*w[1]+bb[2]*w[2]+bb[3]*w[3]+bb[4]*w[4]+bb[5]*w[5]+bb[6]*w[6]+bb[7]*w[7];
    c = cc[0]*w[0]+cc[1]*w[1]+cc[2]*w[2]+cc[3]*w[3]+cc[4]*w[4]+cc[5]*w[5]+cc[6]*w[6]+cc[7]*w[7];

    a = CV_DESCALE(a, trilinear_shift*3);
    b = CV_DESCALE(b, trilinear_shift*3);
    c = CV_DESCALE(c, trilinear_shift*3);
}


// 8 inValues are in [0; LAB_BASE]
static inline void trilinearPackedInterpolate(const v_uint16x8 inX, const v_uint16x8 inY, const v_uint16x8 inZ,
                                              const int16_t* LUT,
                                              v_uint16x8& outA, v_uint16x8& outB, v_uint16x8& outC)
{
    //LUT idx of origin pt of cube
    v_uint16x8 idxsX = inX >> (lab_base_shift - lab_lut_shift);
    v_uint16x8 idxsY = inY >> (lab_base_shift - lab_lut_shift);
    v_uint16x8 idxsZ = inZ >> (lab_base_shift - lab_lut_shift);

    //x, y, z are [0; TRILINEAR_BASE)
    const uint16_t bitMask = (1 << trilinear_shift) - 1;
    v_uint16x8 bitMaskReg = v_setall_u16(bitMask);
    v_uint16x8 fracX = (inX >> (lab_base_shift - 8 - 1)) & bitMaskReg;
    v_uint16x8 fracY = (inY >> (lab_base_shift - 8 - 1)) & bitMaskReg;
    v_uint16x8 fracZ = (inZ >> (lab_base_shift - 8 - 1)) & bitMaskReg;

    //load values to interpolate for pix0, pix1, .., pix7
    v_int16x8 a0, a1, a2, a3, a4, a5, a6, a7;
    v_int16x8 b0, b1, b2, b3, b4, b5, b6, b7;
    v_int16x8 c0, c1, c2, c3, c4, c5, c6, c7;

    v_uint32x4 addrDw0, addrDw1, addrDw10, addrDw11;
    v_mul_expand(v_setall_u16(3*8), idxsX, addrDw0, addrDw1);
    v_mul_expand(v_setall_u16(3*8*LAB_LUT_DIM), idxsY, addrDw10, addrDw11);
    addrDw0 += addrDw10; addrDw1 += addrDw11;
    v_mul_expand(v_setall_u16(3*8*LAB_LUT_DIM*LAB_LUT_DIM), idxsZ, addrDw10, addrDw11);
    addrDw0 += addrDw10; addrDw1 += addrDw11;

    uint32_t CV_DECL_ALIGNED(16) addrofs[8];
    v_store_aligned(addrofs, addrDw0);
    v_store_aligned(addrofs + 4, addrDw1);

    const int16_t* ptr;
#define LOAD_ABC(n) ptr = LUT + addrofs[n]; a##n = v_load(ptr); b##n = v_load(ptr + 8); c##n = v_load(ptr + 16)
    LOAD_ABC(0);
    LOAD_ABC(1);
    LOAD_ABC(2);
    LOAD_ABC(3);
    LOAD_ABC(4);
    LOAD_ABC(5);
    LOAD_ABC(6);
    LOAD_ABC(7);
#undef LOAD_ABC

    //interpolation weights for pix0, pix1, .., pix7
    v_int16x8 w0, w1, w2, w3, w4, w5, w6, w7;
    v_mul_expand(v_setall_u16(8), fracX, addrDw0, addrDw1);
    v_mul_expand(v_setall_u16(8*TRILINEAR_BASE), fracY, addrDw10, addrDw11);
    addrDw0 += addrDw10; addrDw1 += addrDw11;
    v_mul_expand(v_setall_u16(8*TRILINEAR_BASE*TRILINEAR_BASE), fracZ, addrDw10, addrDw11);
    addrDw0 += addrDw10; addrDw1 += addrDw11;

    v_store_aligned(addrofs, addrDw0);
    v_store_aligned(addrofs + 4, addrDw1);

#define LOAD_W(n) ptr = trilinearLUT + addrofs[n]; w##n = v_load(ptr)
    LOAD_W(0);
    LOAD_W(1);
    LOAD_W(2);
    LOAD_W(3);
    LOAD_W(4);
    LOAD_W(5);
    LOAD_W(6);
    LOAD_W(7);
#undef LOAD_W

    //outA = descale(v_reg<8>(sum(dot(ai, wi))))
    v_uint32x4 part0, part1;
#define DOT_SHIFT_PACK(l, ll) \
    part0 = v_uint32x4(v_reduce_sum(v_dotprod(l##0, w0)),\
                       v_reduce_sum(v_dotprod(l##1, w1)),\
                       v_reduce_sum(v_dotprod(l##2, w2)),\
                       v_reduce_sum(v_dotprod(l##3, w3)));\
    part1 = v_uint32x4(v_reduce_sum(v_dotprod(l##4, w4)),\
                       v_reduce_sum(v_dotprod(l##5, w5)),\
                       v_reduce_sum(v_dotprod(l##6, w6)),\
                       v_reduce_sum(v_dotprod(l##7, w7)));\
    (ll) = v_rshr_pack<trilinear_shift*3>(part0, part1)

    DOT_SHIFT_PACK(a, outA);
    DOT_SHIFT_PACK(b, outB);
    DOT_SHIFT_PACK(c, outC);

#undef DOT_SHIFT_PACK
}


struct RGB2Lab_f
{
    typedef float channel_type;

    RGB2Lab_f(int _srccn, int _blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb)
    : srccn(_srccn), srgb(_srgb), blueIdx(_blueIdx)
    {
        volatile int _3 = 3;
        initLabTabs();

        useInterpolation = (!_coeffs && !_whitept && srgb && enableRGB2LabInterpolation);

        useBitExactness = enableBitExactness;

        if (!_coeffs)
            _coeffs = sRGB2XYZ_D65;
        if (!_whitept)
            _whitept = D65;

        if(useBitExactness)
        {
            softfloat scale[] = { softfloat::one() / softfloat(_whitept[0]),
                                  softfloat::one(),
                                  softfloat::one() / softfloat(_whitept[2]) };

            for( int i = 0; i < _3; i++ )
            {
                int j = i * 3;
                softfloat c0 = scale[i] * softfloat(_coeffs[j    ]);
                softfloat c1 = scale[i] * softfloat(_coeffs[j + 1]);
                softfloat c2 = scale[i] * softfloat(_coeffs[j + 2]);
                coeffs[j + (blueIdx ^ 2)] = c0;
                coeffs[j + 1]             = c1;
                coeffs[j + blueIdx]       = c2;

                CV_Assert( c0 >= 0 && c1 >= 0 && c2 >= 0 &&
                           c0 + c1 + c2 < softfloat((int)LAB_CBRT_TAB_SIZE) );
            }
        }
        else
        {
            float scale[] = { 1.0f / _whitept[0], 1.0f, 1.0f / _whitept[2] };

            for( int i = 0; i < _3; i++ )
            {
                int j = i * 3;
                coeffs[j + (blueIdx ^ 2)] = _coeffs[j] * scale[i];
                coeffs[j + 1] = _coeffs[j + 1] * scale[i];
                coeffs[j + blueIdx] = _coeffs[j + 2] * scale[i];

                CV_Assert( coeffs[j] >= 0 && coeffs[j + 1] >= 0 && coeffs[j + 2] >= 0 &&
                           coeffs[j] + coeffs[j + 1] + coeffs[j + 2] < 1.5f*LabCbrtTabScale );
            }
        }
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int i, scn = srccn, bIdx = blueIdx;
        float gscale = GammaTabScale;
        const float* gammaTab = srgb ? ( useBitExactness ? sRGBGammaTab : sRGBGammaTab_gold ) : 0;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        i = 0;
        if(useInterpolation)
        {
            if(enablePackedLab)
            {
                static const int nPixels = 4*2;
                for(; i < n - 3*nPixels; i += 3*nPixels, src += scn*nPixels)
                {
                    v_float32x4 rvec0, gvec0, bvec0, rvec1, gvec1, bvec1;
                    v_float32x4 dummy0, dummy1;
                    if(scn == 3)
                    {
                        v_load_deinterleave(src, rvec0, gvec0, bvec0);
                        v_load_deinterleave(src + scn*4, rvec1, gvec1, bvec1);
                    }
                    else // scn == 4
                    {
                        v_load_deinterleave(src, rvec0, gvec0, bvec0, dummy0);
                        v_load_deinterleave(src + scn*4, rvec1, gvec1, bvec1, dummy1);
                    }

                    if(bIdx)
                    {
                        dummy0 = rvec0; rvec0 = bvec0; bvec0 = dummy0;
                        dummy1 = rvec1; rvec1 = bvec1; bvec1 = dummy1;
                    }

                    v_float32x4 zerof = v_setzero_f32(), onef = v_setall_f32(1.0f);
                    /* clip() */
                    #define clipv(r) (r) = v_min(v_max((r), zerof), onef)
                    clipv(rvec0); clipv(rvec1);
                    clipv(gvec0); clipv(gvec1);
                    clipv(bvec0); clipv(bvec1);
                    #undef clipv
                    /* int iR = R*LAB_BASE, iG = G*LAB_BASE, iB = B*LAB_BASE, iL, ia, ib; */
                    v_float32x4 basef = v_setall_f32(LAB_BASE);
                    rvec0 *= basef, gvec0 *= basef, bvec0 *= basef;
                    rvec1 *= basef, gvec1 *= basef, bvec1 *= basef;

                    v_int32x4 irvec0, igvec0, ibvec0, irvec1, igvec1, ibvec1;
                    irvec0 = v_round(rvec0); irvec1 = v_round(rvec1);
                    igvec0 = v_round(gvec0); igvec1 = v_round(gvec1);
                    ibvec0 = v_round(bvec0); ibvec1 = v_round(bvec1);

                    v_int16x8 irvec, igvec, ibvec;
                    irvec = v_pack(irvec0, irvec1);
                    igvec = v_pack(igvec0, igvec1);
                    ibvec = v_pack(ibvec0, ibvec1);

                    v_uint16x8 uirvec = v_reinterpret_as_u16(irvec);
                    v_uint16x8 uigvec = v_reinterpret_as_u16(igvec);
                    v_uint16x8 uibvec = v_reinterpret_as_u16(ibvec);

                    v_uint16x8 ui_lvec, ui_avec, ui_bvec;
                    trilinearPackedInterpolate(uirvec, uigvec, uibvec, RGB2LabLUT_s16, ui_lvec, ui_avec, ui_bvec);
                    v_int16x8 i_lvec = v_reinterpret_as_s16(ui_lvec);
                    v_int16x8 i_avec = v_reinterpret_as_s16(ui_avec);
                    v_int16x8 i_bvec = v_reinterpret_as_s16(ui_bvec);

                    /* float L = iL*1.0f/LAB_BASE, a = ia*1.0f/LAB_BASE, b = ib*1.0f/LAB_BASE; */
                    v_int32x4 i_lvec0, i_avec0, i_bvec0, i_lvec1, i_avec1, i_bvec1;
                    v_expand(i_lvec, i_lvec0, i_lvec1);
                    v_expand(i_avec, i_avec0, i_avec1);
                    v_expand(i_bvec, i_bvec0, i_bvec1);

                    v_float32x4 l_vec0, a_vec0, b_vec0, l_vec1, a_vec1, b_vec1;
                    l_vec0 = v_cvt_f32(i_lvec0); l_vec1 = v_cvt_f32(i_lvec1);
                    a_vec0 = v_cvt_f32(i_avec0); a_vec1 = v_cvt_f32(i_avec1);
                    b_vec0 = v_cvt_f32(i_bvec0); b_vec1 = v_cvt_f32(i_bvec1);

                    /* dst[i] = L*100.0f */
                    l_vec0 = l_vec0*v_setall_f32(100.0f/LAB_BASE);
                    l_vec1 = l_vec1*v_setall_f32(100.0f/LAB_BASE);
                    /*
                    dst[i + 1] = a*256.0f - 128.0f;
                    dst[i + 2] = b*256.0f - 128.0f;
                    */
                    a_vec0 = a_vec0*v_setall_f32(256.0f/LAB_BASE) - v_setall_f32(128.0f);
                    a_vec1 = a_vec1*v_setall_f32(256.0f/LAB_BASE) - v_setall_f32(128.0f);
                    b_vec0 = b_vec0*v_setall_f32(256.0f/LAB_BASE) - v_setall_f32(128.0f);
                    b_vec1 = b_vec1*v_setall_f32(256.0f/LAB_BASE) - v_setall_f32(128.0f);

                    v_store_interleave(dst + i, l_vec0, a_vec0, b_vec0);
                    v_store_interleave(dst + i + 3*4, l_vec1, a_vec1, b_vec1);
                }
            }

            for(; i < n; i += 3, src += scn)
            {
                float R = clip(src[bIdx]);
                float G = clip(src[1]);
                float B = clip(src[bIdx^2]);

                int iR = cvRound(R*LAB_BASE), iG = cvRound(G*LAB_BASE), iB = cvRound(B*LAB_BASE);
                int iL, ia, ib;
                trilinearInterpolate(iR, iG, iB, RGB2LabLUT_s16, iL, ia, ib);
                float L = iL*1.0f/LAB_BASE, a = ia*1.0f/LAB_BASE, b = ib*1.0f/LAB_BASE;

                dst[i] = L*100.0f;
                dst[i + 1] = a*256.0f - 128.0f;
                dst[i + 2] = b*256.0f - 128.0f;
            }
        }

        static const float _a = 16.0f / 116.0f;
        for (; i < n; i += 3, src += scn )
        {
            float R = clip(src[0]);
            float G = clip(src[1]);
            float B = clip(src[2]);

            if (gammaTab)
            {
                R = splineInterpolate(R * gscale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G * gscale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B * gscale, gammaTab, GAMMA_TAB_SIZE);
            }
            float X = R*C0 + G*C1 + B*C2;
            float Y = R*C3 + G*C4 + B*C5;
            float Z = R*C6 + G*C7 + B*C8;
            // 7.787f = (29/3)^3/(29*4), 0.008856f = (6/29)^3, 903.3 = (29/3)^3
            float FX = X > 0.008856f ? cubeRoot(X) : (7.787f * X + _a);
            float FY = Y > 0.008856f ? cubeRoot(Y) : (7.787f * Y + _a);
            float FZ = Z > 0.008856f ? cubeRoot(Z) : (7.787f * Z + _a);

            float L = Y > 0.008856f ? (116.f * FY - 16.f) : (903.3f * Y);
            float a = 500.f * (FX - FY);
            float b = 200.f * (FY - FZ);

            dst[i] = L;
            dst[i + 1] = a;
            dst[i + 2] = b;
        }
    }

    int srccn;
    float coeffs[9];
    bool srgb;
    bool useInterpolation;
    int blueIdx;
    //just for accuracy measurement purposes
    bool useBitExactness;
};


// Performs conversion in floats
struct Lab2RGBfloat
{
    typedef float channel_type;

    Lab2RGBfloat( int _dstcn, int _blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb )
    : dstcn(_dstcn), srgb(_srgb), blueIdx(_blueIdx)
    {
        initLabTabs();

        if(!_coeffs)
            _coeffs = XYZ2sRGB_D65;
        if(!_whitept)
            _whitept = D65;

        //just for accuracy measurement purposes
        useBitExactness = enableBitExactness;
        if(enableBitExactness)
        {
            for( int i = 0; i < 3; i++ )
            {
                coeffs[i+(blueIdx^2)*3] = (softfloat(_coeffs[i]  )*softfloat(_whitept[i]));
                coeffs[i+3]             = (softfloat(_coeffs[i+3])*softfloat(_whitept[i]));
                coeffs[i+blueIdx*3]     = (softfloat(_coeffs[i+6])*softfloat(_whitept[i]));
            }

            lThresh = softfloat(8); // 0.008856f * 903.3f  = (6/29)^3*(29/3)^3 = 8
            fThresh = softfloat(6)/softfloat(29); // 7.787f * 0.008856f + 16.0f / 116.0f = 6/29
        }
        else
        {
            for( int i = 0; i < 3; i++ )
            {
                coeffs[i+(blueIdx^2)*3] = _coeffs[i]*_whitept[i];
                coeffs[i+3] = _coeffs[i+3]*_whitept[i];
                coeffs[i+blueIdx*3] = _coeffs[i+6]*_whitept[i];
            }

            lThresh = 0.008856f * 903.3f;
            fThresh = 7.787f * 0.008856f + 16.0f / 116.0f;
        }
        #if CV_SSE2
        haveSIMD = checkHardwareSupport(CV_CPU_SSE2);
        #endif
    }

    #if CV_SSE2
    void process(__m128& v_li0, __m128& v_li1, __m128& v_ai0,
                 __m128& v_ai1, __m128& v_bi0, __m128& v_bi1) const
    {
        // 903.3 = (29/3)^3, 7.787 = (29/3)^3/(29*4)
        __m128 v_y00 = _mm_mul_ps(v_li0, _mm_set1_ps(1.0f/903.3f));
        __m128 v_y01 = _mm_mul_ps(v_li1, _mm_set1_ps(1.0f/903.3f));
        __m128 v_fy00 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(7.787f), v_y00), _mm_set1_ps(16.0f/116.0f));
        __m128 v_fy01 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(7.787f), v_y01), _mm_set1_ps(16.0f/116.0f));

        __m128 v_fy10 = _mm_mul_ps(_mm_add_ps(v_li0, _mm_set1_ps(16.0f)), _mm_set1_ps(1.0f/116.0f));
        __m128 v_fy11 = _mm_mul_ps(_mm_add_ps(v_li1, _mm_set1_ps(16.0f)), _mm_set1_ps(1.0f/116.0f));
        __m128 v_y10 = _mm_mul_ps(_mm_mul_ps(v_fy10, v_fy10), v_fy10);
        __m128 v_y11 = _mm_mul_ps(_mm_mul_ps(v_fy11, v_fy11), v_fy11);

        __m128 v_cmpli0 = _mm_cmple_ps(v_li0, _mm_set1_ps(lThresh));
        __m128 v_cmpli1 = _mm_cmple_ps(v_li1, _mm_set1_ps(lThresh));
        v_y00 = _mm_and_ps(v_cmpli0, v_y00);
        v_y01 = _mm_and_ps(v_cmpli1, v_y01);
        v_fy00 = _mm_and_ps(v_cmpli0, v_fy00);
        v_fy01 = _mm_and_ps(v_cmpli1, v_fy01);
        v_y10 = _mm_andnot_ps(v_cmpli0, v_y10);
        v_y11 = _mm_andnot_ps(v_cmpli1, v_y11);
        v_fy10 = _mm_andnot_ps(v_cmpli0, v_fy10);
        v_fy11 = _mm_andnot_ps(v_cmpli1, v_fy11);
        __m128 v_y0 = _mm_or_ps(v_y00, v_y10);
        __m128 v_y1 = _mm_or_ps(v_y01, v_y11);
        __m128 v_fy0 = _mm_or_ps(v_fy00, v_fy10);
        __m128 v_fy1 = _mm_or_ps(v_fy01, v_fy11);

        __m128 v_fxz00 = _mm_add_ps(v_fy0, _mm_mul_ps(v_ai0, _mm_set1_ps(0.002f)));
        __m128 v_fxz01 = _mm_add_ps(v_fy1, _mm_mul_ps(v_ai1, _mm_set1_ps(0.002f)));
        __m128 v_fxz10 = _mm_sub_ps(v_fy0, _mm_mul_ps(v_bi0, _mm_set1_ps(0.005f)));
        __m128 v_fxz11 = _mm_sub_ps(v_fy1, _mm_mul_ps(v_bi1, _mm_set1_ps(0.005f)));

        __m128 v_fxz000 = _mm_mul_ps(_mm_sub_ps(v_fxz00, _mm_set1_ps(16.0f/116.0f)), _mm_set1_ps(1.0f/7.787f));
        __m128 v_fxz001 = _mm_mul_ps(_mm_sub_ps(v_fxz01, _mm_set1_ps(16.0f/116.0f)), _mm_set1_ps(1.0f/7.787f));
        __m128 v_fxz010 = _mm_mul_ps(_mm_sub_ps(v_fxz10, _mm_set1_ps(16.0f/116.0f)), _mm_set1_ps(1.0f/7.787f));
        __m128 v_fxz011 = _mm_mul_ps(_mm_sub_ps(v_fxz11, _mm_set1_ps(16.0f/116.0f)), _mm_set1_ps(1.0f/7.787f));

        __m128 v_fxz100 = _mm_mul_ps(_mm_mul_ps(v_fxz00, v_fxz00), v_fxz00);
        __m128 v_fxz101 = _mm_mul_ps(_mm_mul_ps(v_fxz01, v_fxz01), v_fxz01);
        __m128 v_fxz110 = _mm_mul_ps(_mm_mul_ps(v_fxz10, v_fxz10), v_fxz10);
        __m128 v_fxz111 = _mm_mul_ps(_mm_mul_ps(v_fxz11, v_fxz11), v_fxz11);

        __m128 v_cmpfxz00 = _mm_cmple_ps(v_fxz00, _mm_set1_ps(fThresh));
        __m128 v_cmpfxz01 = _mm_cmple_ps(v_fxz01, _mm_set1_ps(fThresh));
        __m128 v_cmpfxz10 = _mm_cmple_ps(v_fxz10, _mm_set1_ps(fThresh));
        __m128 v_cmpfxz11 = _mm_cmple_ps(v_fxz11, _mm_set1_ps(fThresh));
        v_fxz000 = _mm_and_ps(v_cmpfxz00, v_fxz000);
        v_fxz001 = _mm_and_ps(v_cmpfxz01, v_fxz001);
        v_fxz010 = _mm_and_ps(v_cmpfxz10, v_fxz010);
        v_fxz011 = _mm_and_ps(v_cmpfxz11, v_fxz011);
        v_fxz100 = _mm_andnot_ps(v_cmpfxz00, v_fxz100);
        v_fxz101 = _mm_andnot_ps(v_cmpfxz01, v_fxz101);
        v_fxz110 = _mm_andnot_ps(v_cmpfxz10, v_fxz110);
        v_fxz111 = _mm_andnot_ps(v_cmpfxz11, v_fxz111);
        __m128 v_x0 = _mm_or_ps(v_fxz000, v_fxz100);
        __m128 v_x1 = _mm_or_ps(v_fxz001, v_fxz101);
        __m128 v_z0 = _mm_or_ps(v_fxz010, v_fxz110);
        __m128 v_z1 = _mm_or_ps(v_fxz011, v_fxz111);

        __m128 v_ro0 = _mm_mul_ps(_mm_set1_ps(coeffs[0]), v_x0);
        __m128 v_ro1 = _mm_mul_ps(_mm_set1_ps(coeffs[0]), v_x1);
        __m128 v_go0 = _mm_mul_ps(_mm_set1_ps(coeffs[3]), v_x0);
        __m128 v_go1 = _mm_mul_ps(_mm_set1_ps(coeffs[3]), v_x1);
        __m128 v_bo0 = _mm_mul_ps(_mm_set1_ps(coeffs[6]), v_x0);
        __m128 v_bo1 = _mm_mul_ps(_mm_set1_ps(coeffs[6]), v_x1);
        v_ro0 = _mm_add_ps(v_ro0, _mm_mul_ps(_mm_set1_ps(coeffs[1]), v_y0));
        v_ro1 = _mm_add_ps(v_ro1, _mm_mul_ps(_mm_set1_ps(coeffs[1]), v_y1));
        v_go0 = _mm_add_ps(v_go0, _mm_mul_ps(_mm_set1_ps(coeffs[4]), v_y0));
        v_go1 = _mm_add_ps(v_go1, _mm_mul_ps(_mm_set1_ps(coeffs[4]), v_y1));
        v_bo0 = _mm_add_ps(v_bo0, _mm_mul_ps(_mm_set1_ps(coeffs[7]), v_y0));
        v_bo1 = _mm_add_ps(v_bo1, _mm_mul_ps(_mm_set1_ps(coeffs[7]), v_y1));
        v_ro0 = _mm_add_ps(v_ro0, _mm_mul_ps(_mm_set1_ps(coeffs[2]), v_z0));
        v_ro1 = _mm_add_ps(v_ro1, _mm_mul_ps(_mm_set1_ps(coeffs[2]), v_z1));
        v_go0 = _mm_add_ps(v_go0, _mm_mul_ps(_mm_set1_ps(coeffs[5]), v_z0));
        v_go1 = _mm_add_ps(v_go1, _mm_mul_ps(_mm_set1_ps(coeffs[5]), v_z1));
        v_bo0 = _mm_add_ps(v_bo0, _mm_mul_ps(_mm_set1_ps(coeffs[8]), v_z0));
        v_bo1 = _mm_add_ps(v_bo1, _mm_mul_ps(_mm_set1_ps(coeffs[8]), v_z1));

        v_li0 = _mm_min_ps(_mm_max_ps(v_ro0, _mm_setzero_ps()), _mm_set1_ps(1.0f));
        v_li1 = _mm_min_ps(_mm_max_ps(v_ro1, _mm_setzero_ps()), _mm_set1_ps(1.0f));
        v_ai0 = _mm_min_ps(_mm_max_ps(v_go0, _mm_setzero_ps()), _mm_set1_ps(1.0f));
        v_ai1 = _mm_min_ps(_mm_max_ps(v_go1, _mm_setzero_ps()), _mm_set1_ps(1.0f));
        v_bi0 = _mm_min_ps(_mm_max_ps(v_bo0, _mm_setzero_ps()), _mm_set1_ps(1.0f));
        v_bi1 = _mm_min_ps(_mm_max_ps(v_bo1, _mm_setzero_ps()), _mm_set1_ps(1.0f));
    }
    #endif

    void operator()(const float* src, float* dst, int n) const
    {
        int i = 0, dcn = dstcn;
        const float* gammaTab = srgb ? ( useBitExactness ? sRGBInvGammaTab : sRGBInvGammaTab_gold ) : 0;
        float gscale = GammaTabScale;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
        C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
        C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        float alpha = ColorChannel<float>::max();
        n *= 3;

        #if CV_SSE2
        if (haveSIMD)
        {
            for (; i <= n - 24; i += 24, dst += dcn * 8)
            {
                __m128 v_li0 = _mm_loadu_ps(src + i +  0);
                __m128 v_li1 = _mm_loadu_ps(src + i +  4);
                __m128 v_ai0 = _mm_loadu_ps(src + i +  8);
                __m128 v_ai1 = _mm_loadu_ps(src + i + 12);
                __m128 v_bi0 = _mm_loadu_ps(src + i + 16);
                __m128 v_bi1 = _mm_loadu_ps(src + i + 20);

                _mm_deinterleave_ps(v_li0, v_li1, v_ai0, v_ai1, v_bi0, v_bi1);

                process(v_li0, v_li1, v_ai0, v_ai1, v_bi0, v_bi1);

                if (gammaTab)
                {
                    __m128 v_gscale = _mm_set1_ps(gscale);
                    v_li0 = _mm_mul_ps(v_li0, v_gscale);
                    v_li1 = _mm_mul_ps(v_li1, v_gscale);
                    v_ai0 = _mm_mul_ps(v_ai0, v_gscale);
                    v_ai1 = _mm_mul_ps(v_ai1, v_gscale);
                    v_bi0 = _mm_mul_ps(v_bi0, v_gscale);
                    v_bi1 = _mm_mul_ps(v_bi1, v_gscale);

                    splineInterpolate(v_li0, gammaTab, GAMMA_TAB_SIZE);
                    splineInterpolate(v_li1, gammaTab, GAMMA_TAB_SIZE);
                    splineInterpolate(v_ai0, gammaTab, GAMMA_TAB_SIZE);
                    splineInterpolate(v_ai1, gammaTab, GAMMA_TAB_SIZE);
                    splineInterpolate(v_bi0, gammaTab, GAMMA_TAB_SIZE);
                    splineInterpolate(v_bi1, gammaTab, GAMMA_TAB_SIZE);
                }

                if( dcn == 4 )
                {
                    __m128 v_a0 = _mm_set1_ps(alpha);
                    __m128 v_a1 = _mm_set1_ps(alpha);
                    _mm_interleave_ps(v_li0, v_li1, v_ai0, v_ai1, v_bi0, v_bi1, v_a0, v_a1);

                    _mm_storeu_ps(dst +  0, v_li0);
                    _mm_storeu_ps(dst +  4, v_li1);
                    _mm_storeu_ps(dst +  8, v_ai0);
                    _mm_storeu_ps(dst + 12, v_ai1);
                    _mm_storeu_ps(dst + 16, v_bi0);
                    _mm_storeu_ps(dst + 20, v_bi1);
                    _mm_storeu_ps(dst + 24, v_a0);
                    _mm_storeu_ps(dst + 28, v_a1);
                }
                else
                {
                    _mm_interleave_ps(v_li0, v_li1, v_ai0, v_ai1, v_bi0, v_bi1);

                    _mm_storeu_ps(dst +  0, v_li0);
                    _mm_storeu_ps(dst +  4, v_li1);
                    _mm_storeu_ps(dst +  8, v_ai0);
                    _mm_storeu_ps(dst + 12, v_ai1);
                    _mm_storeu_ps(dst + 16, v_bi0);
                    _mm_storeu_ps(dst + 20, v_bi1);
                }
            }
        }
        #endif
        for (; i < n; i += 3, dst += dcn)
        {
            float li = src[i];
            float ai = src[i + 1];
            float bi = src[i + 2];

            // 903.3 = (29/3)^3, 7.787 = (29/3)^3/(29*4)
            float y, fy;
            if (li <= lThresh)
            {
                y = li / 903.3f;
                fy = 7.787f * y + 16.0f / 116.0f;
            }
            else
            {
                fy = (li + 16.0f) / 116.0f;
                y = fy * fy * fy;
            }

            float fxz[] = { ai / 500.0f + fy, fy - bi / 200.0f };

            for (int j = 0; j < 2; j++)
                if (fxz[j] <= fThresh)
                    fxz[j] = (fxz[j] - 16.0f / 116.0f) / 7.787f;
                else
                    fxz[j] = fxz[j] * fxz[j] * fxz[j];

            float x = fxz[0], z = fxz[1];
            float ro = C0 * x + C1 * y + C2 * z;
            float go = C3 * x + C4 * y + C5 * z;
            float bo = C6 * x + C7 * y + C8 * z;
            ro = clip(ro);
            go = clip(go);
            bo = clip(bo);

            if (gammaTab)
            {
                ro = splineInterpolate(ro * gscale, gammaTab, GAMMA_TAB_SIZE);
                go = splineInterpolate(go * gscale, gammaTab, GAMMA_TAB_SIZE);
                bo = splineInterpolate(bo * gscale, gammaTab, GAMMA_TAB_SIZE);
            }

            dst[0] = ro, dst[1] = go, dst[2] = bo;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn;
    float coeffs[9];
    bool srgb;
    float lThresh;
    float fThresh;
    #if CV_SSE2
    bool haveSIMD;
    #endif
    int blueIdx;
    //just for accuracy measurement purposes
    bool useBitExactness;
};


struct RGB2Lab_b
{
    typedef uchar channel_type;

    RGB2Lab_b(int _srccn, int _blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb)
    : srccn(_srccn), srgb(_srgb), blueIdx(_blueIdx)
    {
        static volatile int _3 = 3;
        initLabTabs();

        if (!_coeffs)
            _coeffs = sRGB2XYZ_D65;
        if (!_whitept)
            _whitept = D65;

        //just for accuracy measurement purposes
        useBitExactness = enableBitExactness;
        if(enableBitExactness)
        {
            static const softfloat lshift(1 << lab_shift);
            for( int i = 0; i < _3; i++ )
            {
                coeffs[i*3+(blueIdx^2)] = cvRound((lshift*softfloat(_coeffs[i*3  ]))/softfloat(_whitept[i]));
                coeffs[i*3+1]           = cvRound((lshift*softfloat(_coeffs[i*3+1]))/softfloat(_whitept[i]));
                coeffs[i*3+blueIdx]     = cvRound((lshift*softfloat(_coeffs[i*3+2]))/softfloat(_whitept[i]));

                CV_Assert(coeffs[i*3] >= 0 && coeffs[i*3+1] >= 0 && coeffs[i*3+2] >= 0 &&
                          coeffs[i*3] + coeffs[i*3+1] + coeffs[i*3+2] < 2*(1 << lab_shift));
            }
        }
        else
        {
            for( int i = 0; i < _3; i++ )
            {
                coeffs[i*3+(blueIdx^2)] = cvRound((float)(1 << lab_shift)*_coeffs[i*3  ]/_whitept[i]);
                coeffs[i*3+1]           = cvRound((float)(1 << lab_shift)*_coeffs[i*3+1]/_whitept[i]);
                coeffs[i*3+blueIdx]     = cvRound((float)(1 << lab_shift)*_coeffs[i*3+2]/_whitept[i]);

                CV_Assert(coeffs[i] >= 0 && coeffs[i*3+1] >= 0 && coeffs[i*3+2] >= 0 &&
                          coeffs[i*3] + coeffs[i*3+1] + coeffs[i*3+2] < 2*(1 << lab_shift) );
            }
        }
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        const int Lscale = (116*255+50)/100;
        const int Lshift = -((16*255*(1 << lab_shift2) + 50)/100);
        const ushort* tab = srgb ? ( useBitExactness ? sRGBGammaTab_b : sRGBGammaTab_b_gold ) :
                                   ( useBitExactness ? linearGammaTab_b : linearGammaTab_b_gold );
        const ushort* cbrtTab = useBitExactness ? LabCbrtTab_b : LabCbrtTab_b_gold;
        int i, scn = srccn;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        i = 0;
        for(; i < n; i += 3, src += scn )
        {
            int R = tab[src[0]], G = tab[src[1]], B = tab[src[2]];
            int fX = cbrtTab[CV_DESCALE(R*C0 + G*C1 + B*C2, lab_shift)];
            int fY = cbrtTab[CV_DESCALE(R*C3 + G*C4 + B*C5, lab_shift)];
            int fZ = cbrtTab[CV_DESCALE(R*C6 + G*C7 + B*C8, lab_shift)];

            int L = CV_DESCALE( Lscale*fY + Lshift, lab_shift2 );
            int a = CV_DESCALE( 500*(fX - fY) + 128*(1 << lab_shift2), lab_shift2 );
            int b = CV_DESCALE( 200*(fY - fZ) + 128*(1 << lab_shift2), lab_shift2 );

            dst[i] = saturate_cast<uchar>(L);
            dst[i+1] = saturate_cast<uchar>(a);
            dst[i+2] = saturate_cast<uchar>(b);
        }
    }

    int srccn;
    int coeffs[9];
    bool srgb;
    int blueIdx;
    // just for accuracy measurement purposes
    bool useBitExactness;
};


// Performs conversion in integers
struct Lab2RGBinteger
{
    typedef uchar channel_type;

    static const int base_shift = 14;
    static const int BASE = (1 << base_shift);
    // lThresh == (6/29)^3 * (29/3)^3 * BASE/100
    static const int lThresh = 1311;
    // fThresh == ((29/3)^3/(29*4) * (6/29)^3 + 16/116)*BASE
    static const int fThresh = 3390;
    // base16_116 == BASE*16/116
    static const int base16_116 = 2260;
    static const int shift = lab_shift+(base_shift-inv_gamma_shift);

    Lab2RGBinteger( int _dstcn, int blueIdx, const float* _coeffs,
                    const float* _whitept, bool srgb )
    : dstcn(_dstcn)
    {
        if(!_coeffs)
            _coeffs = XYZ2sRGB_D65;
        if(!_whitept)
            _whitept = D65;

        if(enableBitExactness)
        {
            static const softfloat lshift(1 << lab_shift);
            for(int i = 0; i < 3; i++)
            {
                coeffs[i+(blueIdx)*3]   = cvRound(lshift*softfloat(_coeffs[i  ])*softfloat(_whitept[i]));
                coeffs[i+3]             = cvRound(lshift*softfloat(_coeffs[i+3])*softfloat(_whitept[i]));
                coeffs[i+(blueIdx^2)*3] = cvRound(lshift*softfloat(_coeffs[i+6])*softfloat(_whitept[i]));
            }
        }
        else
        {
            for(int i = 0; i < 3; i++)
            {
                coeffs[i+(blueIdx)*3]   = cvRound((float)(1 << lab_shift)*_coeffs[i  ]*_whitept[i]);
                coeffs[i+3]             = cvRound((float)(1 << lab_shift)*_coeffs[i+3]*_whitept[i]);
                coeffs[i+(blueIdx^2)*3] = cvRound((float)(1 << lab_shift)*_coeffs[i+6]*_whitept[i]);
            }
        }

        tab = srgb ? (enableBitExactness ? sRGBInvGammaTab_b   : sRGBInvGammaTab_b_gold) :
                     (enableBitExactness ? linearInvGammaTab_b : linearInvGammaTab_b_gold);
    }

    // L, a, b should be in their natural range
    inline void process(const uchar LL, const uchar aa, const uchar bb, int& ro, int& go, int& bo) const
    {
        int x, y, z;
        int ify;

        y   = LabToYF_b[LL*2  ];
        ify = LabToYF_b[LL*2+1];

        //float fxz[] = { ai / 500.0f + fy, fy - bi / 200.0f };
        int adiv, bdiv;
        adiv = aa*BASE/500 - 128*BASE/500, bdiv = bb*BASE/200 - 128*BASE/200;

        int ifxz[] = {ify + adiv, ify - bdiv};

        for(int k = 0; k < 2; k++)
        {
            int& v = ifxz[k];
            v = abToXZ_b[v-minABvalue];
        }
        x = ifxz[0]; /* y = y */; z = ifxz[1];

        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2];
        int C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5];
        int C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

        ro = CV_DESCALE(C0 * x + C1 * y + C2 * z, shift);
        go = CV_DESCALE(C3 * x + C4 * y + C5 * z, shift);
        bo = CV_DESCALE(C6 * x + C7 * y + C8 * z, shift);

        ro = max(0, min((int)INV_GAMMA_TAB_SIZE-1, ro));
        go = max(0, min((int)INV_GAMMA_TAB_SIZE-1, go));
        bo = max(0, min((int)INV_GAMMA_TAB_SIZE-1, bo));

        ro = tab[ro];
        go = tab[go];
        bo = tab[bo];
    }

    // L, a, b shoule be in their natural range
    inline void processLabToXYZ(const v_uint8x16& lv, const v_uint8x16& av, const v_uint8x16& bv,
                                v_int32x4& xiv00, v_int32x4& yiv00, v_int32x4& ziv00,
                                v_int32x4& xiv01, v_int32x4& yiv01, v_int32x4& ziv01,
                                v_int32x4& xiv10, v_int32x4& yiv10, v_int32x4& ziv10,
                                v_int32x4& xiv11, v_int32x4& yiv11, v_int32x4& ziv11) const
    {
        v_uint16x8 lv0, lv1;
        v_expand(lv, lv0, lv1);
        // Load Y and IFY values from lookup-table
        // y = LabToYF_b[L_value*2], ify = LabToYF_b[L_value*2 + 1]
        // LabToYF_b[i*2  ] = y;   // 2260 <= y <= BASE
        // LabToYF_b[i*2+1] = ify; // 0 <= ify <= BASE
        uint16_t CV_DECL_ALIGNED(16) v_lv0[8], v_lv1[8];
        v_store_aligned(v_lv0, (lv0 << 1)); v_store_aligned(v_lv1, (lv1 << 1));
        v_int16x8 ify0, ify1;

        yiv00 = v_int32x4(LabToYF_b[v_lv0[0]  ], LabToYF_b[v_lv0[1]  ], LabToYF_b[v_lv0[2]  ], LabToYF_b[v_lv0[3]  ]);
        yiv01 = v_int32x4(LabToYF_b[v_lv0[4]  ], LabToYF_b[v_lv0[5]  ], LabToYF_b[v_lv0[6]  ], LabToYF_b[v_lv0[7]  ]);
        yiv10 = v_int32x4(LabToYF_b[v_lv1[0]  ], LabToYF_b[v_lv1[1]  ], LabToYF_b[v_lv1[2]  ], LabToYF_b[v_lv1[3]  ]);
        yiv11 = v_int32x4(LabToYF_b[v_lv1[4]  ], LabToYF_b[v_lv1[5]  ], LabToYF_b[v_lv1[6]  ], LabToYF_b[v_lv1[7]  ]);

        ify0 = v_int16x8(LabToYF_b[v_lv0[0]+1], LabToYF_b[v_lv0[1]+1], LabToYF_b[v_lv0[2]+1], LabToYF_b[v_lv0[3]+1],
                         LabToYF_b[v_lv0[4]+1], LabToYF_b[v_lv0[5]+1], LabToYF_b[v_lv0[6]+1], LabToYF_b[v_lv0[7]+1]);
        ify1 = v_int16x8(LabToYF_b[v_lv1[0]+1], LabToYF_b[v_lv1[1]+1], LabToYF_b[v_lv1[2]+1], LabToYF_b[v_lv1[3]+1],
                         LabToYF_b[v_lv1[4]+1], LabToYF_b[v_lv1[5]+1], LabToYF_b[v_lv1[6]+1], LabToYF_b[v_lv1[7]+1]);

        v_int16x8 adiv0, adiv1, bdiv0, bdiv1;
        v_uint16x8 av0, av1, bv0, bv1;
        v_expand(av, av0, av1); v_expand(bv, bv0, bv1);
        //adiv = aa*BASE/500 - 128*BASE/500, bdiv = bb*BASE/200 - 128*BASE/200;
        //approximations with reasonable precision
        v_uint16x8 mulA = v_setall_u16(53687);
        v_uint32x4 ma00, ma01, ma10, ma11;
        v_uint32x4 addA = v_setall_u32(1 << 7);
        v_mul_expand((av0 + (av0 << 2)), mulA, ma00, ma01);
        v_mul_expand((av1 + (av1 << 2)), mulA, ma10, ma11);
        adiv0 = v_reinterpret_as_s16(v_pack(((ma00 + addA) >> 13), ((ma01 + addA) >> 13)));
        adiv1 = v_reinterpret_as_s16(v_pack(((ma10 + addA) >> 13), ((ma11 + addA) >> 13)));

        v_uint16x8 mulB = v_setall_u16(41943);
        v_uint32x4 mb00, mb01, mb10, mb11;
        v_uint32x4 addB = v_setall_u32(1 << 4);
        v_mul_expand(bv0, mulB, mb00, mb01);
        v_mul_expand(bv1, mulB, mb10, mb11);
        bdiv0 = v_reinterpret_as_s16(v_pack((mb00 + addB) >> 9, (mb01 + addB) >> 9));
        bdiv1 = v_reinterpret_as_s16(v_pack((mb10 + addB) >> 9, (mb11 + addB) >> 9));

        // 0 <= adiv <= 8356, 0 <= bdiv <= 20890
        /* x = ifxz[0]; y = y; z = ifxz[1]; */
        v_uint16x8 xiv0, xiv1, ziv0, ziv1;
        v_int16x8 vSubA = v_setall_s16(-128*BASE/500 - minABvalue), vSubB = v_setall_s16(128*BASE/200-1 - minABvalue);

        // int ifxz[] = {ify + adiv, ify - bdiv};
        // ifxz[k] = abToXZ_b[ifxz[k]-minABvalue];
        xiv0 = v_reinterpret_as_u16(v_add_wrap(v_add_wrap(ify0, adiv0), vSubA));
        xiv1 = v_reinterpret_as_u16(v_add_wrap(v_add_wrap(ify1, adiv1), vSubA));
        ziv0 = v_reinterpret_as_u16(v_add_wrap(v_sub_wrap(ify0, bdiv0), vSubB));
        ziv1 = v_reinterpret_as_u16(v_add_wrap(v_sub_wrap(ify1, bdiv1), vSubB));

        uint16_t CV_DECL_ALIGNED(16) v_x0[8], v_x1[8], v_z0[8], v_z1[8];
        v_store_aligned(v_x0, xiv0 ); v_store_aligned(v_x1, xiv1 );
        v_store_aligned(v_z0, ziv0 ); v_store_aligned(v_z1, ziv1 );

        xiv00 = v_int32x4(abToXZ_b[v_x0[0]], abToXZ_b[v_x0[1]], abToXZ_b[v_x0[2]], abToXZ_b[v_x0[3]]);
        xiv01 = v_int32x4(abToXZ_b[v_x0[4]], abToXZ_b[v_x0[5]], abToXZ_b[v_x0[6]], abToXZ_b[v_x0[7]]);
        xiv10 = v_int32x4(abToXZ_b[v_x1[0]], abToXZ_b[v_x1[1]], abToXZ_b[v_x1[2]], abToXZ_b[v_x1[3]]);
        xiv11 = v_int32x4(abToXZ_b[v_x1[4]], abToXZ_b[v_x1[5]], abToXZ_b[v_x1[6]], abToXZ_b[v_x1[7]]);
        ziv00 = v_int32x4(abToXZ_b[v_z0[0]], abToXZ_b[v_z0[1]], abToXZ_b[v_z0[2]], abToXZ_b[v_z0[3]]);
        ziv01 = v_int32x4(abToXZ_b[v_z0[4]], abToXZ_b[v_z0[5]], abToXZ_b[v_z0[6]], abToXZ_b[v_z0[7]]);
        ziv10 = v_int32x4(abToXZ_b[v_z1[0]], abToXZ_b[v_z1[1]], abToXZ_b[v_z1[2]], abToXZ_b[v_z1[3]]);
        ziv11 = v_int32x4(abToXZ_b[v_z1[4]], abToXZ_b[v_z1[5]], abToXZ_b[v_z1[6]], abToXZ_b[v_z1[7]]);
        // abToXZ_b[i-minABvalue] = v; // -1335 <= v <= 88231
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int dcn = dstcn;
        float alpha = ColorChannel<float>::max();

        int i = 0;
        if(enablePackedLab)
        {
            v_float32x4 vldiv  = v_setall_f32(256.f/100.0f);
            v_float32x4 vf255  = v_setall_f32(255.f);
            static const int nPixels = 16;
            for(; i <= n*3-3*nPixels; i += 3*nPixels, dst += dcn*nPixels)
            {
                /*
                int L = saturate_cast<int>(src[i]*BASE/100.0f);
                int a = saturate_cast<int>(src[i + 1]*BASE/256);
                int b = saturate_cast<int>(src[i + 2]*BASE/256);
                */
                v_float32x4 vl[4], va[4], vb[4];
                for(int k = 0; k < 4; k++)
                {
                    v_load_deinterleave(src + i + k*3*4, vl[k], va[k], vb[k]);
                    vl[k] *= vldiv;
                }

                v_int32x4 ivl[4], iva[4], ivb[4];
                for(int k = 0; k < 4; k++)
                {
                    ivl[k] = v_round(vl[k]), iva[k] = v_round(va[k]), ivb[k] = v_round(vb[k]);
                }
                v_int16x8 ivl16[2], iva16[2], ivb16[2];
                ivl16[0] = v_pack(ivl[0], ivl[1]); iva16[0] = v_pack(iva[0], iva[1]); ivb16[0] = v_pack(ivb[0], ivb[1]);
                ivl16[1] = v_pack(ivl[2], ivl[3]); iva16[1] = v_pack(iva[2], iva[3]); ivb16[1] = v_pack(ivb[2], ivb[3]);
                v_uint8x16 ivl8, iva8, ivb8;
                ivl8 = v_reinterpret_as_u8(v_pack(ivl16[0], ivl16[1]));
                iva8 = v_reinterpret_as_u8(v_pack(iva16[0], iva16[1]));
                ivb8 = v_reinterpret_as_u8(v_pack(ivb16[0], ivb16[1]));

                v_int32x4 ixv[4], iyv[4], izv[4];

                processLabToXYZ(ivl8, iva8, ivb8, ixv[0], iyv[0], izv[0],
                                                  ixv[1], iyv[1], izv[1],
                                                  ixv[2], iyv[2], izv[2],
                                                  ixv[3], iyv[3], izv[3]);
                /*
                        ro = CV_DESCALE(C0 * x + C1 * y + C2 * z, shift);
                        go = CV_DESCALE(C3 * x + C4 * y + C5 * z, shift);
                        bo = CV_DESCALE(C6 * x + C7 * y + C8 * z, shift);
                */
                v_int32x4 C0 = v_setall_s32(coeffs[0]), C1 = v_setall_s32(coeffs[1]), C2 = v_setall_s32(coeffs[2]);
                v_int32x4 C3 = v_setall_s32(coeffs[3]), C4 = v_setall_s32(coeffs[4]), C5 = v_setall_s32(coeffs[5]);
                v_int32x4 C6 = v_setall_s32(coeffs[6]), C7 = v_setall_s32(coeffs[7]), C8 = v_setall_s32(coeffs[8]);
                v_int32x4 descaleShift = v_setall_s32(1 << (shift-1)), tabsz = v_setall_s32((int)INV_GAMMA_TAB_SIZE-1);
                for(int k = 0; k < 4; k++)
                {
                    v_int32x4 i_r, i_g, i_b;
                    v_uint32x4 r_vecs, g_vecs, b_vecs;
                    i_r = (ixv[k]*C0 + iyv[k]*C1 + izv[k]*C2 + descaleShift) >> shift;
                    i_g = (ixv[k]*C3 + iyv[k]*C4 + izv[k]*C5 + descaleShift) >> shift;
                    i_b = (ixv[k]*C6 + iyv[k]*C7 + izv[k]*C8 + descaleShift) >> shift;

                    //limit indices in table and then substitute
                    //ro = tab[ro]; go = tab[go]; bo = tab[bo];
                    int32_t CV_DECL_ALIGNED(16) rshifts[4], gshifts[4], bshifts[4];
                    v_int32x4 rs = v_max(v_setzero_s32(), v_min(tabsz, i_r));
                    v_int32x4 gs = v_max(v_setzero_s32(), v_min(tabsz, i_g));
                    v_int32x4 bs = v_max(v_setzero_s32(), v_min(tabsz, i_b));

                    v_store_aligned(rshifts, rs);
                    v_store_aligned(gshifts, gs);
                    v_store_aligned(bshifts, bs);

                    r_vecs = v_uint32x4(tab[rshifts[0]], tab[rshifts[1]], tab[rshifts[2]], tab[rshifts[3]]);
                    g_vecs = v_uint32x4(tab[gshifts[0]], tab[gshifts[1]], tab[gshifts[2]], tab[gshifts[3]]);
                    b_vecs = v_uint32x4(tab[bshifts[0]], tab[bshifts[1]], tab[bshifts[2]], tab[bshifts[3]]);

                    v_float32x4 v_r, v_g, v_b;
                    v_r = v_cvt_f32(v_reinterpret_as_s32(r_vecs))/vf255;
                    v_g = v_cvt_f32(v_reinterpret_as_s32(g_vecs))/vf255;
                    v_b = v_cvt_f32(v_reinterpret_as_s32(b_vecs))/vf255;

                    if(dcn == 4)
                    {
                        v_store_interleave(dst + k*dcn*4, v_b, v_g, v_r, v_setall_f32(alpha));
                    }
                    else // dcn == 3
                    {
                        v_store_interleave(dst + k*dcn*4, v_b, v_g, v_r);
                    }
                }
            }
        }

        for(; i < n*3; i += 3, dst += dcn)
        {
            int ro, go, bo;
            process((uchar)(src[i + 0]*255.f/100.f), (uchar)src[i + 1], (uchar)src[i + 2], ro, go, bo);

            dst[0] = bo/255.f;
            dst[1] = go/255.f;
            dst[2] = ro/255.f;
            if(dcn == 4)
                dst[3] = alpha;
        }
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
        i = 0;

        if(enablePackedLab)
        {
            static const int nPixels = 8*2;
            for(; i <= n*3-3*nPixels; i += 3*nPixels, dst += dcn*nPixels)
            {
                /*
                    int L = src[i + 0];
                    int a = src[i + 1];
                    int b = src[i + 2];
                */
                v_uint8x16 u8l, u8a, u8b;
                v_load_deinterleave(src + i, u8l, u8a, u8b);

                v_int32x4 xiv[4], yiv[4], ziv[4];
                processLabToXYZ(u8l, u8a, u8b, xiv[0], yiv[0], ziv[0],
                                               xiv[1], yiv[1], ziv[1],
                                               xiv[2], yiv[2], ziv[2],
                                               xiv[3], yiv[3], ziv[3]);
                /*
                        ro = CV_DESCALE(C0 * x + C1 * y + C2 * z, shift);
                        go = CV_DESCALE(C3 * x + C4 * y + C5 * z, shift);
                        bo = CV_DESCALE(C6 * x + C7 * y + C8 * z, shift);
                */
                v_int32x4 C0 = v_setall_s32(coeffs[0]), C1 = v_setall_s32(coeffs[1]), C2 = v_setall_s32(coeffs[2]);
                v_int32x4 C3 = v_setall_s32(coeffs[3]), C4 = v_setall_s32(coeffs[4]), C5 = v_setall_s32(coeffs[5]);
                v_int32x4 C6 = v_setall_s32(coeffs[6]), C7 = v_setall_s32(coeffs[7]), C8 = v_setall_s32(coeffs[8]);
                v_int32x4 descaleShift = v_setall_s32(1 << (shift-1));
                v_int32x4 tabsz = v_setall_s32((int)INV_GAMMA_TAB_SIZE-1);
                v_uint32x4 r_vecs[4], g_vecs[4], b_vecs[4];
                for(int k = 0; k < 4; k++)
                {
                    v_int32x4 i_r, i_g, i_b;
                    i_r = (xiv[k]*C0 + yiv[k]*C1 + ziv[k]*C2 + descaleShift) >> shift;
                    i_g = (xiv[k]*C3 + yiv[k]*C4 + ziv[k]*C5 + descaleShift) >> shift;
                    i_b = (xiv[k]*C6 + yiv[k]*C7 + ziv[k]*C8 + descaleShift) >> shift;

                    //limit indices in table and then substitute
                    //ro = tab[ro]; go = tab[go]; bo = tab[bo];
                    int32_t CV_DECL_ALIGNED(16) rshifts[4], gshifts[4], bshifts[4];
                    v_int32x4 rs = v_max(v_setzero_s32(), v_min(tabsz, i_r));
                    v_int32x4 gs = v_max(v_setzero_s32(), v_min(tabsz, i_g));
                    v_int32x4 bs = v_max(v_setzero_s32(), v_min(tabsz, i_b));

                    v_store_aligned(rshifts, rs);
                    v_store_aligned(gshifts, gs);
                    v_store_aligned(bshifts, bs);

                    r_vecs[k] = v_uint32x4(tab[rshifts[0]], tab[rshifts[1]], tab[rshifts[2]], tab[rshifts[3]]);
                    g_vecs[k] = v_uint32x4(tab[gshifts[0]], tab[gshifts[1]], tab[gshifts[2]], tab[gshifts[3]]);
                    b_vecs[k] = v_uint32x4(tab[bshifts[0]], tab[bshifts[1]], tab[bshifts[2]], tab[bshifts[3]]);
                }

                v_uint16x8 u_rvec0 = v_pack(r_vecs[0], r_vecs[1]), u_rvec1 = v_pack(r_vecs[2], r_vecs[3]);
                v_uint16x8 u_gvec0 = v_pack(g_vecs[0], g_vecs[1]), u_gvec1 = v_pack(g_vecs[2], g_vecs[3]);
                v_uint16x8 u_bvec0 = v_pack(b_vecs[0], b_vecs[1]), u_bvec1 = v_pack(b_vecs[2], b_vecs[3]);

                v_uint8x16 u8_b, u8_g, u8_r;
                u8_b = v_pack(u_bvec0, u_bvec1);
                u8_g = v_pack(u_gvec0, u_gvec1);
                u8_r = v_pack(u_rvec0, u_rvec1);

                if(dcn == 4)
                {
                    v_store_interleave(dst, u8_b, u8_g, u8_r, v_setall_u8(alpha));
                }
                else
                {
                    v_store_interleave(dst, u8_b, u8_g, u8_r);
                }
            }
        }

        for (; i < n*3; i += 3, dst += dcn)
        {
            int ro, go, bo;
            process(src[i + 0], src[i + 1], src[i + 2], ro, go, bo);

            dst[0] = saturate_cast<uchar>(bo);
            dst[1] = saturate_cast<uchar>(go);
            dst[2] = saturate_cast<uchar>(ro);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn;
    int coeffs[9];
    ushort* tab;
};


struct Lab2RGB_b
{
    typedef uchar channel_type;

    Lab2RGB_b( int _dstcn, int _blueIdx, const float* _coeffs,
               const float* _whitept, bool _srgb )
    : fcvt(3, _blueIdx, _coeffs, _whitept, _srgb ), icvt(_dstcn, _blueIdx, _coeffs, _whitept, _srgb), dstcn(_dstcn)
    {
        useBitExactness = (!_coeffs && !_whitept && _srgb && enableBitExactness);

        #if CV_NEON
        v_scale_inv = vdupq_n_f32(100.f/255.f);
        v_scale = vdupq_n_f32(255.f);
        v_alpha = vdup_n_u8(ColorChannel<uchar>::max());
        v_128 = vdupq_n_f32(128.0f);
        #elif CV_SSE2
        v_scale = _mm_set1_ps(255.f);
        v_alpha = _mm_set1_ps(ColorChannel<uchar>::max());
        v_zero = _mm_setzero_si128();
        haveSIMD = checkHardwareSupport(CV_CPU_SSE2);
        #endif
    }

    #if CV_SSE2
    // 16s x 8
    void process(__m128i v_r, __m128i v_g, __m128i v_b,
                 const __m128& v_coeffs_, const __m128& v_res_,
                 float * buf) const
    {
        __m128 v_r0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_r, v_zero));
        __m128 v_g0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_g, v_zero));
        __m128 v_b0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_b, v_zero));

        __m128 v_r1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_r, v_zero));
        __m128 v_g1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_g, v_zero));
        __m128 v_b1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_b, v_zero));

        __m128 v_coeffs = v_coeffs_;
        __m128 v_res = v_res_;

        v_r0 = _mm_sub_ps(_mm_mul_ps(v_r0, v_coeffs), v_res);
        v_g1 = _mm_sub_ps(_mm_mul_ps(v_g1, v_coeffs), v_res);

        v_coeffs = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v_coeffs), 0x49));
        v_res = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v_res), 0x49));

        v_r1 = _mm_sub_ps(_mm_mul_ps(v_r1, v_coeffs), v_res);
        v_b0 = _mm_sub_ps(_mm_mul_ps(v_b0, v_coeffs), v_res);

        v_coeffs = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v_coeffs), 0x49));
        v_res = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(v_res), 0x49));

        v_g0 = _mm_sub_ps(_mm_mul_ps(v_g0, v_coeffs), v_res);
        v_b1 = _mm_sub_ps(_mm_mul_ps(v_b1, v_coeffs), v_res);

        _mm_store_ps(buf, v_r0);
        _mm_store_ps(buf + 4, v_r1);
        _mm_store_ps(buf + 8, v_g0);
        _mm_store_ps(buf + 12, v_g1);
        _mm_store_ps(buf + 16, v_b0);
        _mm_store_ps(buf + 20, v_b1);
    }
    #endif

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        if(useBitExactness)
        {
            icvt(src, dst, n);
            return;
        }

        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
        float CV_DECL_ALIGNED(16) buf[3*BLOCK_SIZE];
        #if CV_SSE2
        __m128 v_coeffs = _mm_set_ps(100.f/255.f, 1.f, 1.f, 100.f/255.f);
        __m128 v_res = _mm_set_ps(0.f, 128.f, 128.f, 0.f);
        #endif

        i = 0;
        for(; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);
            j = 0;

            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24)
            {
                uint8x8x3_t v_src = vld3_u8(src + j);
                uint16x8_t v_t0 = vmovl_u8(v_src.val[0]),
                           v_t1 = vmovl_u8(v_src.val[1]),
                           v_t2 = vmovl_u8(v_src.val[2]);

                float32x4x3_t v_dst;
                v_dst.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t0))), v_scale_inv);
                v_dst.val[1] = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t1))), v_128);
                v_dst.val[2] = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_t2))), v_128);
                vst3q_f32(buf + j, v_dst);

                v_dst.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t0))), v_scale_inv);
                v_dst.val[1] = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t1))), v_128);
                v_dst.val[2] = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_t2))), v_128);
                vst3q_f32(buf + j + 12, v_dst);
            }
            #elif CV_SSE2
            if (haveSIMD)
            {
                for ( ; j <= (dn - 8) * 3; j += 24)
                {
                    __m128i v_src0 = _mm_loadu_si128((__m128i const *)(src + j));
                    __m128i v_src1 = _mm_loadl_epi64((__m128i const *)(src + j + 16));

                    process(_mm_unpacklo_epi8(v_src0, v_zero),
                            _mm_unpackhi_epi8(v_src0, v_zero),
                            _mm_unpacklo_epi8(v_src1, v_zero),
                            v_coeffs, v_res,
                            buf + j);
                }
            }
            #endif

            for( ; j < dn*3; j += 3 )
            {
                buf[j] = src[j]*(100.f/255.f);
                buf[j+1] = (float)(src[j+1] - 128);
                buf[j+2] = (float)(src[j+2] - 128);
            }
            fcvt(buf, buf, dn);
            j = 0;

            #if CV_NEON
            for ( ; j <= (dn - 8) * 3; j += 24, dst += dcn * 8)
            {
                float32x4x3_t v_src0 = vld3q_f32(buf + j), v_src1 = vld3q_f32(buf + j + 12);
                uint8x8_t v_dst0 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[0], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[0], v_scale)))));
                uint8x8_t v_dst1 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[1], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[1], v_scale)))));
                uint8x8_t v_dst2 = vqmovn_u16(vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src0.val[2], v_scale))),
                                                           vqmovn_u32(cv_vrndq_u32_f32(vmulq_f32(v_src1.val[2], v_scale)))));

                if (dcn == 4)
                {
                    uint8x8x4_t v_dst;
                    v_dst.val[0] = v_dst0;
                    v_dst.val[1] = v_dst1;
                    v_dst.val[2] = v_dst2;
                    v_dst.val[3] = v_alpha;
                    vst4_u8(dst, v_dst);
                }
                else
                {
                    uint8x8x3_t v_dst;
                    v_dst.val[0] = v_dst0;
                    v_dst.val[1] = v_dst1;
                    v_dst.val[2] = v_dst2;
                    vst3_u8(dst, v_dst);
                }
            }
            #elif CV_SSE2
            if (dcn == 3 && haveSIMD)
            {
                for ( ; j <= (dn * 3 - 16); j += 16, dst += 16)
                {
                    __m128 v_src0 = _mm_mul_ps(_mm_load_ps(buf + j), v_scale);
                    __m128 v_src1 = _mm_mul_ps(_mm_load_ps(buf + j + 4), v_scale);
                    __m128 v_src2 = _mm_mul_ps(_mm_load_ps(buf + j + 8), v_scale);
                    __m128 v_src3 = _mm_mul_ps(_mm_load_ps(buf + j + 12), v_scale);

                    __m128i v_dst0 = _mm_packs_epi32(_mm_cvtps_epi32(v_src0),
                                                     _mm_cvtps_epi32(v_src1));
                    __m128i v_dst1 = _mm_packs_epi32(_mm_cvtps_epi32(v_src2),
                                                     _mm_cvtps_epi32(v_src3));

                    _mm_storeu_si128((__m128i *)dst, _mm_packus_epi16(v_dst0, v_dst1));
                }

                int jr = j % 3;
                if (jr)
                    dst -= jr, j -= jr;
            }
            else if (dcn == 4 && haveSIMD)
            {
                for ( ; j <= (dn * 3 - 12); j += 12, dst += 16)
                {
                    __m128 v_buf0 = _mm_mul_ps(_mm_load_ps(buf + j), v_scale);
                    __m128 v_buf1 = _mm_mul_ps(_mm_load_ps(buf + j + 4), v_scale);
                    __m128 v_buf2 = _mm_mul_ps(_mm_load_ps(buf + j + 8), v_scale);

                    __m128 v_ba0 = _mm_unpackhi_ps(v_buf0, v_alpha);
                    __m128 v_ba1 = _mm_unpacklo_ps(v_buf2, v_alpha);

                    __m128i v_src0 = _mm_cvtps_epi32(_mm_shuffle_ps(v_buf0, v_ba0, 0x44));
                    __m128i v_src1 = _mm_shuffle_epi32(_mm_cvtps_epi32(_mm_shuffle_ps(v_ba0, v_buf1, 0x4e)), 0x78);
                    __m128i v_src2 = _mm_cvtps_epi32(_mm_shuffle_ps(v_buf1, v_ba1, 0x4e));
                    __m128i v_src3 = _mm_shuffle_epi32(_mm_cvtps_epi32(_mm_shuffle_ps(v_ba1, v_buf2, 0xee)), 0x78);

                    __m128i v_dst0 = _mm_packs_epi32(v_src0, v_src1);
                    __m128i v_dst1 = _mm_packs_epi32(v_src2, v_src3);

                    _mm_storeu_si128((__m128i *)dst, _mm_packus_epi16(v_dst0, v_dst1));
                }

                int jr = j % 3;
                if (jr)
                    dst -= jr, j -= jr;
            }
            #endif

            for( ; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    Lab2RGBfloat   fcvt;
    Lab2RGBinteger icvt;
    #if CV_NEON
    float32x4_t v_scale, v_scale_inv, v_128;
    uint8x8_t v_alpha;
    #elif CV_SSE2
    __m128 v_scale;
    __m128 v_alpha;
    __m128i v_zero;
    bool haveSIMD;
    #endif
    bool useBitExactness;
    int dstcn;
};


struct Lab2RGB_f
{
    typedef float channel_type;

    Lab2RGB_f( int _dstcn, int _blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb )
    : fcvt(_dstcn, _blueIdx, _coeffs, _whitept, _srgb), dstcn(_dstcn)
    { }

    void operator()(const float* src, float* dst, int n) const
    {
        fcvt(src, dst, n);
    }

    Lab2RGBfloat fcvt;
    int dstcn;
};

#undef clip

/////////////

TEST(ImgProc_Color, LabCheckWorking)
{
    //<bits, divisor, multiplier, toAdd>
//    MulFracConst<14, 9033, 1000, 0> mf;


//    mf.calc()
//    return;

    /*
    for(int L = 0; L < 256; L++)
    {
        int y, ify;
        int ycheck, ifycheck;
        static const int BASE = (1 << 14);
        if( L <= 0.008856f * 903.3f * 255.0f / 100.0f ) //20.39904324
        {
            //yy = li / 903.3f;
            //y = L*100/903.3f;
            //y = L*BASE*1000/9033/255;
            MulFracConst<14, 9033*255/5, 200*BASE, 0> mf0;
            y = (L*mf0.vmul + mf0.vadd)>>mf0.r;
            ycheck = cvRound((double)L*1000.0/9033.0/255.0*BASE);

            cout << "mf0: vmul " << mf0.vmul << " vadd " << mf0.vadd << " r " << mf0.r << endl;

            //fy = 7.787f * yy + 16.0f / 116.0f;
            //ify = BASE*16/116 + L*7787*BASE/255/9033;


            MulFracConst<14, 255*9033, 7787*BASE, BASE*16/116 +1> mf1;
            ify = (L*mf1.vmul + mf1.vadd) >> mf1.r;
            ifycheck = cvRound((double)L*7787.0*BASE/255.0/9033.0 +
                               BASE*16.0/116.0);

            cout << "mf1: vmul " << mf1.vmul << " vadd " << mf1.vadd << " r " << mf1.r << endl;

            cout << L << ": " << y << " " << ycheck;
            cout << " " << ify << " " << ifycheck << endl;
        }
        else
        {
            //fy = (li + 16.0f) / 116.0f;
            MulFracConst<14, 1479, 5*BASE, 16/116*BASE +1 > mf0;
            //ify = L/255*100/116*BASE + 16/116*BASE;
            ify = (L*mf0.vmul + mf0.vadd) >> mf0.r;
            ifycheck = cvRound((double)L*BASE*100.0/255.0/116.0 +
                               BASE*16.0/116.0);

            cout << "mf0: vmul " << mf0.vmul << " vadd " << mf0.vadd << " r " << mf0.r << endl;

            //yy = fy * fy * fy;
            y = ify*ify/BASE*ify/BASE;
            ycheck = cvRound(ifycheck*ifycheck/BASE*ifycheck/BASE);

            cout << L << ": " << y << " " << ycheck;
            cout << " " << ify << " " << ifycheck << endl;
        }
    }

    return;*/


    //TODO: make good test
    //return;

    //cv::setUseOptimized(false);

    //settings
    #define INT_DATA 1
    #define TO_BGR 1
    const bool randomFill = true;

    int dstChannels = 3;
    int blueIdx = 0;
    bool srgb = true;

    enableBitExactness = true; enableRGB2LabInterpolation = true;
    Lab2RGB_f interToBgr  (dstChannels, blueIdx, 0, 0, srgb);
    RGB2Lab_f interToLab  (dstChannels, blueIdx, 0, 0, srgb);
    Lab2RGB_b interToBgr_b(dstChannels, blueIdx, 0, 0, srgb);
    RGB2Lab_b interToLab_b(dstChannels, blueIdx, 0, 0, srgb);

    enableBitExactness = false; enableRGB2LabInterpolation = false;
    Lab2RGB_f goldToBgr  (dstChannels, blueIdx, 0, 0, srgb);
    RGB2Lab_f goldToLab  (dstChannels, blueIdx, 0, 0, srgb);
    Lab2RGB_b goldToBgr_b(dstChannels, blueIdx, 0, 0, srgb);
    RGB2Lab_b goldToLab_b(dstChannels, blueIdx, 0, 0, srgb);

    char bgrChannels[3] = {'b', 'g', 'r'};
    char labChannels[3] = {'l', 'a', 'b'};
    char* channel = TO_BGR ? bgrChannels : labChannels;

    int nPerfIters = 100;

    string dir;
    dir = "/home/savuor/logs/ocv/lab_precision/";
    dir += string(TO_BGR ? "lab2bgr/" : "rgb2lab/");

    const size_t pSize = 256+1;
    Mat  mGold(pSize, pSize, CV_32FC3);
    Mat   mSrc(pSize, pSize, CV_32FC3);
    Mat mInter(pSize, pSize, CV_32FC3);
    Mat   mBackGold(pSize, pSize, CV_32FC3);
    Mat  mBackInter(pSize, pSize, CV_32FC3);

    if(INT_DATA)
    {
        mGold  = Mat(pSize, pSize, CV_8UC3);
        mSrc   = Mat(pSize, pSize, CV_8UC3);
        mInter = Mat(pSize, pSize, CV_8UC3);
        mBackGold  = Mat(pSize, pSize, CV_8UC3);
        mBackInter = Mat(pSize, pSize, CV_8UC3);
    }

    Scalar vmean, vdev;
    std::vector<Mat> chDiff, chInter;
    double vmin[3], vmax[3]; Point minPt[3], maxPt[3];
    double maxMaxError[4] = {-100, -100, -100, -100};
    double times[4] = {1e9, 1e9, 1e9, 1e9};
    int count = 0;

    int blue = 0, l = 0;
#if TO_BGR
    for(; l < 100+1; l++)
#else
    for(; blue < 256+1; blue++)
#endif
    {
        for(size_t p = 0; p < pSize; p++)
        {
            float* pRow   = mSrc.ptr<float>(p);
            uchar* pRow_b = mSrc.ptr<uchar>(p);
            for(size_t q = 0; q < pSize; q++)
            {
                if(INT_DATA)
                {
                    if(TO_BGR)
                    {
                        //Lab
                        pRow_b[3*q + 0] = l*255/100;
                        pRow_b[3*q + 1] = q;
                        pRow_b[3*q + 2] = p;
                    }
                    else
                    {
                        //BGR
                        pRow_b[3*q + blueIdx]     = blue;
                        pRow_b[3*q + 1]           = q;
                        pRow_b[3*q + (blueIdx^2)] = p;
                    }
                }
                else
                {
                    if(TO_BGR)
                    {
                        //Lab
                        pRow[3*q + 0] = 1.0f*l;
                        pRow[3*q + 1] = 256.0f*q/(pSize-1)-128.0f;
                        pRow[3*q + 2] = 256.0f*p/(pSize-1)-128.0f;
                    }
                    else
                    {
                        //BGR
                        pRow[3*q + blueIdx]       = 1.0f*blue/(pSize-1);
                        pRow[3*q + 1]             = 1.0f*q/(pSize-1);
                        pRow[3*q + (blueIdx ^ 2)] = 1.0f*p/(pSize-1);
                    }
                }

            }
        }

        for(size_t p = 0; p < pSize; p++)
        {
            float* pSrc   =   mSrc.ptr<float>(p);
            float* pGold  =  mGold.ptr<float>(p);
            float* pInter = mInter.ptr<float>(p);
            float* pBackGold  = mBackGold.ptr<float>(p);
            float* pBackInter = mBackInter.ptr<float>(p);

            uchar* pSrc_b   =   mSrc.ptr<uchar>(p);
            uchar* pGold_b  =  mGold.ptr<uchar>(p);
            uchar* pInter_b = mInter.ptr<uchar>(p);
            uchar* pBackGold_b  = mBackGold.ptr<uchar>(p);
            uchar* pBackInter_b = mBackInter.ptr<uchar>(p);
            if(INT_DATA)
            {
                if(TO_BGR)
                {
                    interToBgr_b(pSrc_b, pInter_b, pSize);
                    goldToBgr_b(pSrc_b, pGold_b, pSize);

                    interToLab_b(pInter_b, pBackInter_b, pSize);
                    goldToLab_b(pGold_b, pBackGold_b, pSize);
                }
                else
                {
                    interToLab_b(pSrc_b, pInter_b, pSize);
                    goldToLab_b(pSrc_b, pGold_b, pSize);

                    interToBgr_b(pInter_b, pBackInter_b, pSize);
                    goldToBgr_b(pGold_b, pBackGold_b, pSize);
                }
            }
            else
            {
                if(TO_BGR)
                {
                    interToBgr(pSrc, pInter, pSize);
                    goldToBgr(pSrc, pGold, pSize);

                    interToLab(pInter, pBackInter, pSize);
                    goldToLab(pGold, pBackGold, pSize);
                }
                else
                {
                    interToLab(pSrc, pInter, pSize);
                    goldToLab(pSrc, pGold, pSize);

                    interToBgr(pInter, pBackInter, pSize);
                    goldToBgr(pGold, pBackGold, pSize);
                }
            }
        }

        std::cout << (TO_BGR ? l : blue) << ":" << endl;

        Mat diff = abs(mGold-mInter);
        meanStdDev(diff, vmean, vdev);
        std::cout << "absdiff: mean " << vmean << " stddev " << vdev << std::endl;
        split(diff, chDiff);
        split(mInter, chInter);
        for(int c = 0; c < 3; c++)
        {
            minMaxLoc(chDiff[c], &vmin[c], &vmax[c],
                      &minPt[c], &maxPt[c]);
            std::cout << " ch "  << channel[c];
            std::cout << " max " << vmax[c] << " at " << maxPt[c];
            maxMaxError[0] = max(maxMaxError[0], vmax[c]);
        }
        std::cout << std::endl;

        Mat backGoldDiff = abs(mBackGold - mSrc);
        meanStdDev(backGoldDiff, vmean, vdev);
        std::cout << "backGoldDiff: mean " << vmean << " stddev " << vdev << std::endl;
        split(backGoldDiff, chDiff);
        for(int c = 0; c < 3; c++)
        {
            minMaxLoc(chDiff[c], &vmin[c], &vmax[c],
                      &minPt[c], &maxPt[c]);
            std::cout << " ch "  << channel[c];
            std::cout << " max " << vmax[c] << " at " << maxPt[c];
            maxMaxError[1] = max(maxMaxError[1], vmax[c]);
        }
        std::cout << std::endl;

        Mat backInterDiff = abs(mBackInter - mSrc);
        meanStdDev(backInterDiff, vmean, vdev);
        std::cout << "backInterDiff: mean " << vmean << " stddev " << vdev << std::endl;
        split(backInterDiff, chDiff);
        for(int c = 0; c < 3; c++)
        {
            minMaxLoc(chDiff[c], &vmin[c], &vmax[c],
                      &minPt[c], &maxPt[c]);
            std::cout << " ch "  << channel[c];
            std::cout << " max " << vmax[c] << " at " << maxPt[c];
            maxMaxError[2] = max(maxMaxError[2], vmax[c]);
        }
        std::cout << std::endl;

        Mat backInterGoldDiff = abs(mBackInter - mBackGold);
        meanStdDev(backInterGoldDiff, vmean, vdev);
        std::cout << "backInterGoldDiff: mean " << vmean << " stddev " << vdev << std::endl;
        split(backInterGoldDiff, chDiff);
        for(int c = 0; c < 3; c++)
        {
            minMaxLoc(chDiff[c], &vmin[c], &vmax[c],
                      &minPt[c], &maxPt[c]);
            std::cout << " ch "  << channel[c];
            std::cout << " max " << vmax[c] << " at " << maxPt[c];
            maxMaxError[3] = max(maxMaxError[3], vmax[c]);
        }
        std::cout << std::endl;

        Mat tmp = INT_DATA ? mGold : (TO_BGR ? mGold*256 : mGold+Scalar(0, 128, 128));
        imwrite(format((dir + "noInter%03d.png").c_str(),  (TO_BGR ? l : blue)), tmp);

        tmp = INT_DATA ? mInter : (TO_BGR ? mInter*256 : mInter+Scalar(0, 128, 128));
        imwrite(format((dir + "useInter%03d.png").c_str(), (TO_BGR ? l : blue)), tmp);

        tmp = INT_DATA ? (TO_BGR ? chInter[2] : chInter[1]) : (TO_BGR ? chInter[2]*256 : chInter[1]+Scalar::all(128));
        imwrite(format((dir + "red%03d.png").c_str(),      (TO_BGR ? l : blue)), tmp);

        tmp = INT_DATA ? (mGold-mInter) : (TO_BGR ? (mGold-mInter)*256+Scalar::all(128) : (mGold-mInter)+Scalar::all(128));
        imwrite(format((dir + "diff%03d.png").c_str(),     (TO_BGR ? l : blue)), tmp);

        tmp = INT_DATA ? abs(mGold-mInter) : (TO_BGR ? abs(mGold-mInter)*256 : abs(mGold-mInter));
        imwrite(format((dir + "absdiff%03d.png").c_str(),  (TO_BGR ? l : blue)), tmp);

        tmp = INT_DATA ? backGoldDiff : (TO_BGR ? backGoldDiff+Scalar::all(128) : backGoldDiff*256);
        imwrite(format((dir + "backgolddiff%03d.png").c_str(),  (TO_BGR ? l : blue)), tmp);

        tmp = INT_DATA ? backInterDiff : (TO_BGR ? backInterDiff+Scalar::all(128) : backInterDiff*256);
        imwrite(format((dir + "backinterdiff%03d.png").c_str(), (TO_BGR ? l : blue)), tmp);

        if(randomFill)
        {
            RNG rng;
            for(size_t p = 0; p < pSize; p++)
            {
                float* pRow   = mSrc.ptr<float>(p);
                uchar* pRow_b = mSrc.ptr<uchar>(p);
                for(size_t q = 0; q < pSize; q++)
                {
                    if(INT_DATA)
                    {
                        if(TO_BGR)
                        {
                            //Lab
                            pRow_b[3*q + 0] = rng(256)*100/255;
                            pRow_b[3*q + 1] = rng(256);
                            pRow_b[3*q + 2] = rng(256);
                        }
                        else
                        {
                            //BGR
                            pRow_b[3*q + 0] = rng(256);
                            pRow_b[3*q + 1] = rng(256);
                            pRow_b[3*q + 2] = rng(256);
                        }
                    }
                    else
                    {
                        if(TO_BGR)
                        {
                            //Lab
                            pRow[3*q + 0] = (float)rng*100.0f;
                            pRow[3*q + 1] = 256.0f*(float)rng-128.0f;
                            pRow[3*q + 2] = 256.0f*(float)rng-128.0f;
                        }
                        else
                        {
                            //BGR
                            pRow[3*q + 0] = (float)rng;
                            pRow[3*q + 1] = (float)rng;
                            pRow[3*q + 2] = (float)rng;
                        }
                    }
                }
            }
        }

        //perf test
        std::cout.flush();
        std::cout << "perf: ";
        TickMeter tm; double t;
        //Lab to BGR
        tm.start();
        for(int i = 0; i < nPerfIters; i++)
        {
            for(size_t p = 0; p < pSize; p++)
            {
                float* pSrc   =   mSrc.ptr<float>(p);
                float* pInter = mInter.ptr<float>(p);
                uchar* pSrc_b   =   mSrc.ptr<uchar>(p);
                uchar* pInter_b = mInter.ptr<uchar>(p);
                if(INT_DATA)
                {
                    interToBgr_b(pSrc_b, pInter_b, pSize);
                }
                else
                {
                    interToBgr(pSrc, pInter, pSize);
                }
            }
        }
        tm.stop();
        t = tm.getTimeSec(); times[0] = min(times[0], t);
        std::cout << "inter lab2bgr: " << t << " ";
        tm.reset(); tm.start();
        for(int i = 0; i < nPerfIters; i++)
        {
            for(size_t p = 0; p < pSize; p++)
            {
                float* pSrc   =   mSrc.ptr<float>(p);
                float* pGold = mGold.ptr<float>(p);
                uchar* pSrc_b  =  mSrc.ptr<uchar>(p);
                uchar* pGold_b = mGold.ptr<uchar>(p);
                if(INT_DATA)
                {
                    goldToBgr_b(pSrc_b, pGold_b, pSize);
                }
                else
                {
                    goldToBgr(pSrc, pGold, pSize);
                }
            }
        }
        tm.stop();
        t = tm.getTimeSec(); times[1] = min(times[1], t);
        std::cout << "gold lab2bgr: " << t << " ";
        //RGB to Lab
        tm.reset(); tm.start();
        for(int i = 0; i < nPerfIters; i++)
        {
            for(size_t p = 0; p < pSize; p++)
            {
                float* pSrc   =   mSrc.ptr<float>(p);
                float* pInter = mInter.ptr<float>(p);
                uchar* pSrc_b   =   mSrc.ptr<uchar>(p);
                uchar* pInter_b = mInter.ptr<uchar>(p);
                if(INT_DATA)
                {
                    interToLab_b(pSrc_b, pInter_b, pSize);
                }
                else
                {
                    interToLab(pSrc, pInter, pSize);
                }
            }
        }
        tm.stop();
        t = tm.getTimeSec(); times[2] = min(times[2], t);
        std::cout << "inter rgb2lab: " << t << " ";
        tm.reset(); tm.start();
        for(int i = 0; i < nPerfIters; i++)
        {
            for(size_t p = 0; p < pSize; p++)
            {
                float* pSrc   =   mSrc.ptr<float>(p);
                float* pGold = mGold.ptr<float>(p);
                uchar* pSrc_b  =  mSrc.ptr<uchar>(p);
                uchar* pGold_b = mGold.ptr<uchar>(p);
                if(INT_DATA)
                {
                    goldToLab_b(pSrc_b, pGold_b, pSize);
                }
                else
                {
                    goldToLab(pSrc, pGold, pSize);
                }
            }
        }
        tm.stop();
        t = tm.getTimeSec(); times[3] = min(times[3], t);
        std::cout << "gold rgb2lab: " << t << " ";
        std::cout << std::endl;
        std::cout.flush();
        count++;
    }

    //max-max channel errors
    std::cout << std::endl << (TO_BGR ? "Lab2RGB" : "RGB2Lab") << " ";
    std::cout << "lab_lut_shift " << (int)lab_lut_shift << " ";
    for(int i = 0; i < 4; i++)
    {
        std::cout << maxMaxError[i] << "\t";
    }
    std::cout << std::endl;

    //overall perf
    std::cout << "perf: ";
    std::cout << "inter lab2bgr: " << times[0] << " ";
    std::cout << "gold lab2bgr: "  << times[1] << " ";
    std::cout << "inter rgb2lab: " << times[2] << " ";
    std::cout << "gold rgb2lab: "  << times[3] << " ";
    std::cout << std::endl;
}
