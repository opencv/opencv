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

#include "precomp.hpp"
#include "opencv2/core/opencl/runtime/opencl_clfft.hpp"
#include "opencv2/core/opencl/runtime/opencl_core.hpp"
#include "opencl_kernels_core.hpp"
#include <map>

namespace cv
{

// On Win64 optimized versions of DFT and DCT fail the tests (fixed in VS2010)
#if defined _MSC_VER && !defined CV_ICC && defined _M_X64 && _MSC_VER < 1600
# pragma optimize("", off)
# pragma warning(disable: 4748)
#endif

#if IPP_VERSION_X100 >= 710
#define USE_IPP_DFT 1
#else
#undef USE_IPP_DFT
#endif

/****************************************************************************************\
                               Discrete Fourier Transform
\****************************************************************************************/

static unsigned char bitrevTab[] =
{
  0x00,0x80,0x40,0xc0,0x20,0xa0,0x60,0xe0,0x10,0x90,0x50,0xd0,0x30,0xb0,0x70,0xf0,
  0x08,0x88,0x48,0xc8,0x28,0xa8,0x68,0xe8,0x18,0x98,0x58,0xd8,0x38,0xb8,0x78,0xf8,
  0x04,0x84,0x44,0xc4,0x24,0xa4,0x64,0xe4,0x14,0x94,0x54,0xd4,0x34,0xb4,0x74,0xf4,
  0x0c,0x8c,0x4c,0xcc,0x2c,0xac,0x6c,0xec,0x1c,0x9c,0x5c,0xdc,0x3c,0xbc,0x7c,0xfc,
  0x02,0x82,0x42,0xc2,0x22,0xa2,0x62,0xe2,0x12,0x92,0x52,0xd2,0x32,0xb2,0x72,0xf2,
  0x0a,0x8a,0x4a,0xca,0x2a,0xaa,0x6a,0xea,0x1a,0x9a,0x5a,0xda,0x3a,0xba,0x7a,0xfa,
  0x06,0x86,0x46,0xc6,0x26,0xa6,0x66,0xe6,0x16,0x96,0x56,0xd6,0x36,0xb6,0x76,0xf6,
  0x0e,0x8e,0x4e,0xce,0x2e,0xae,0x6e,0xee,0x1e,0x9e,0x5e,0xde,0x3e,0xbe,0x7e,0xfe,
  0x01,0x81,0x41,0xc1,0x21,0xa1,0x61,0xe1,0x11,0x91,0x51,0xd1,0x31,0xb1,0x71,0xf1,
  0x09,0x89,0x49,0xc9,0x29,0xa9,0x69,0xe9,0x19,0x99,0x59,0xd9,0x39,0xb9,0x79,0xf9,
  0x05,0x85,0x45,0xc5,0x25,0xa5,0x65,0xe5,0x15,0x95,0x55,0xd5,0x35,0xb5,0x75,0xf5,
  0x0d,0x8d,0x4d,0xcd,0x2d,0xad,0x6d,0xed,0x1d,0x9d,0x5d,0xdd,0x3d,0xbd,0x7d,0xfd,
  0x03,0x83,0x43,0xc3,0x23,0xa3,0x63,0xe3,0x13,0x93,0x53,0xd3,0x33,0xb3,0x73,0xf3,
  0x0b,0x8b,0x4b,0xcb,0x2b,0xab,0x6b,0xeb,0x1b,0x9b,0x5b,0xdb,0x3b,0xbb,0x7b,0xfb,
  0x07,0x87,0x47,0xc7,0x27,0xa7,0x67,0xe7,0x17,0x97,0x57,0xd7,0x37,0xb7,0x77,0xf7,
  0x0f,0x8f,0x4f,0xcf,0x2f,0xaf,0x6f,0xef,0x1f,0x9f,0x5f,0xdf,0x3f,0xbf,0x7f,0xff
};

static const double DFTTab[][2] =
{
{ 1.00000000000000000, 0.00000000000000000 },
{-1.00000000000000000, 0.00000000000000000 },
{ 0.00000000000000000, 1.00000000000000000 },
{ 0.70710678118654757, 0.70710678118654746 },
{ 0.92387953251128674, 0.38268343236508978 },
{ 0.98078528040323043, 0.19509032201612825 },
{ 0.99518472667219693, 0.09801714032956060 },
{ 0.99879545620517241, 0.04906767432741802 },
{ 0.99969881869620425, 0.02454122852291229 },
{ 0.99992470183914450, 0.01227153828571993 },
{ 0.99998117528260111, 0.00613588464915448 },
{ 0.99999529380957619, 0.00306795676296598 },
{ 0.99999882345170188, 0.00153398018628477 },
{ 0.99999970586288223, 0.00076699031874270 },
{ 0.99999992646571789, 0.00038349518757140 },
{ 0.99999998161642933, 0.00019174759731070 },
{ 0.99999999540410733, 0.00009587379909598 },
{ 0.99999999885102686, 0.00004793689960307 },
{ 0.99999999971275666, 0.00002396844980842 },
{ 0.99999999992818922, 0.00001198422490507 },
{ 0.99999999998204725, 0.00000599211245264 },
{ 0.99999999999551181, 0.00000299605622633 },
{ 0.99999999999887801, 0.00000149802811317 },
{ 0.99999999999971945, 0.00000074901405658 },
{ 0.99999999999992983, 0.00000037450702829 },
{ 0.99999999999998246, 0.00000018725351415 },
{ 0.99999999999999567, 0.00000009362675707 },
{ 0.99999999999999889, 0.00000004681337854 },
{ 0.99999999999999978, 0.00000002340668927 },
{ 0.99999999999999989, 0.00000001170334463 },
{ 1.00000000000000000, 0.00000000585167232 },
{ 1.00000000000000000, 0.00000000292583616 }
};

namespace {
template <typename T>
struct Constants {
    static const T sin_120;
    static const T fft5_2;
    static const T fft5_3;
    static const T fft5_4;
    static const T fft5_5;
};

template <typename T>
const T Constants<T>::sin_120 = (T)0.86602540378443864676372317075294;

template <typename T>
const T Constants<T>::fft5_2 = (T)0.559016994374947424102293417182819;

template <typename T>
const T Constants<T>::fft5_3 = (T)-0.951056516295153572116439333379382;

template <typename T>
const T Constants<T>::fft5_4 = (T)-1.538841768587626701285145288018455;

template <typename T>
const T Constants<T>::fft5_5 = (T)0.363271264002680442947733378740309;

}  //namespace

#define BitRev(i,shift) \
   ((int)((((unsigned)bitrevTab[(i)&255] << 24)+ \
           ((unsigned)bitrevTab[((i)>> 8)&255] << 16)+ \
           ((unsigned)bitrevTab[((i)>>16)&255] <<  8)+ \
           ((unsigned)bitrevTab[((i)>>24)])) >> (shift)))

static int
DFTFactorize( int n, int* factors )
{
    int nf = 0, f, i, j;

    if( n <= 5 )
    {
        factors[0] = n;
        return 1;
    }

    f = (((n - 1)^n)+1) >> 1;
    if( f > 1 )
    {
        factors[nf++] = f;
        n = f == n ? 1 : n/f;
    }

    for( f = 3; n > 1; )
    {
        int d = n/f;
        if( d*f == n )
        {
            factors[nf++] = f;
            n = d;
        }
        else
        {
            f += 2;
            if( f*f > n )
                break;
        }
    }

    if( n > 1 )
        factors[nf++] = n;

    f = (factors[0] & 1) == 0;
    for( i = f; i < (nf+f)/2; i++ )
        CV_SWAP( factors[i], factors[nf-i-1+f], j );

    return nf;
}

static void
DFTInit( int n0, int nf, const int* factors, int* itab, int elem_size, void* _wave, int inv_itab )
{
    int digits[34], radix[34];
    int n = factors[0], m = 0;
    int* itab0 = itab;
    int i, j, k;
    Complex<double> w, w1;
    double t;

    if( n0 <= 5 )
    {
        itab[0] = 0;
        itab[n0-1] = n0-1;

        if( n0 != 4 )
        {
            for( i = 1; i < n0-1; i++ )
                itab[i] = i;
        }
        else
        {
            itab[1] = 2;
            itab[2] = 1;
        }
        if( n0 == 5 )
        {
            if( elem_size == sizeof(Complex<double>) )
                ((Complex<double>*)_wave)[0] = Complex<double>(1.,0.);
            else
                ((Complex<float>*)_wave)[0] = Complex<float>(1.f,0.f);
        }
        if( n0 != 4 )
            return;
        m = 2;
    }
    else
    {
        // radix[] is initialized from index 'nf' down to zero
        CV_Assert (nf < 34);
        radix[nf] = 1;
        digits[nf] = 0;
        for( i = 0; i < nf; i++ )
        {
            digits[i] = 0;
            radix[nf-i-1] = radix[nf-i]*factors[nf-i-1];
        }

        if( inv_itab && factors[0] != factors[nf-1] )
            itab = (int*)_wave;

        if( (n & 1) == 0 )
        {
            int a = radix[1], na2 = n*a>>1, na4 = na2 >> 1;
            for( m = 0; (unsigned)(1 << m) < (unsigned)n; m++ )
                ;
            if( n <= 2  )
            {
                itab[0] = 0;
                itab[1] = na2;
            }
            else if( n <= 256 )
            {
                int shift = 10 - m;
                for( i = 0; i <= n - 4; i += 4 )
                {
                    j = (bitrevTab[i>>2]>>shift)*a;
                    itab[i] = j;
                    itab[i+1] = j + na2;
                    itab[i+2] = j + na4;
                    itab[i+3] = j + na2 + na4;
                }
            }
            else
            {
                int shift = 34 - m;
                for( i = 0; i < n; i += 4 )
                {
                    int i4 = i >> 2;
                    j = BitRev(i4,shift)*a;
                    itab[i] = j;
                    itab[i+1] = j + na2;
                    itab[i+2] = j + na4;
                    itab[i+3] = j + na2 + na4;
                }
            }

            digits[1]++;

            if( nf >= 2 )
            {
                for( i = n, j = radix[2]; i < n0; )
                {
                    for( k = 0; k < n; k++ )
                        itab[i+k] = itab[k] + j;
                    if( (i += n) >= n0 )
                        break;
                    j += radix[2];
                    for( k = 1; ++digits[k] >= factors[k]; k++ )
                    {
                        digits[k] = 0;
                        j += radix[k+2] - radix[k];
                    }
                }
            }
        }
        else
        {
            for( i = 0, j = 0;; )
            {
                itab[i] = j;
                if( ++i >= n0 )
                    break;
                j += radix[1];
                for( k = 0; ++digits[k] >= factors[k]; k++ )
                {
                    digits[k] = 0;
                    j += radix[k+2] - radix[k];
                }
            }
        }

        if( itab != itab0 )
        {
            itab0[0] = 0;
            for( i = n0 & 1; i < n0; i += 2 )
            {
                int k0 = itab[i];
                int k1 = itab[i+1];
                itab0[k0] = i;
                itab0[k1] = i+1;
            }
        }
    }

    if( (n0 & (n0-1)) == 0 )
    {
        w.re = w1.re = DFTTab[m][0];
        w.im = w1.im = -DFTTab[m][1];
    }
    else
    {
        t = -CV_PI*2/n0;
        w.im = w1.im = sin(t);
        w.re = w1.re = std::sqrt(1. - w1.im*w1.im);
    }
    n = (n0+1)/2;

    if( elem_size == sizeof(Complex<double>) )
    {
        Complex<double>* wave = (Complex<double>*)_wave;

        wave[0].re = 1.;
        wave[0].im = 0.;

        if( (n0 & 1) == 0 )
        {
            wave[n].re = -1.;
            wave[n].im = 0;
        }

        for( i = 1; i < n; i++ )
        {
            wave[i] = w;
            wave[n0-i].re = w.re;
            wave[n0-i].im = -w.im;

            t = w.re*w1.re - w.im*w1.im;
            w.im = w.re*w1.im + w.im*w1.re;
            w.re = t;
        }
    }
    else
    {
        Complex<float>* wave = (Complex<float>*)_wave;
        CV_Assert( elem_size == sizeof(Complex<float>) );

        wave[0].re = 1.f;
        wave[0].im = 0.f;

        if( (n0 & 1) == 0 )
        {
            wave[n].re = -1.f;
            wave[n].im = 0.f;
        }

        for( i = 1; i < n; i++ )
        {
            wave[i].re = (float)w.re;
            wave[i].im = (float)w.im;
            wave[n0-i].re = (float)w.re;
            wave[n0-i].im = (float)-w.im;

            t = w.re*w1.re - w.im*w1.im;
            w.im = w.re*w1.im + w.im*w1.re;
            w.re = t;
        }
    }
}

// Reference radix-2 implementation.
template<typename T> struct DFT_R2
{
    void operator()(Complex<T>* dst, const int c_n, const int n, const int dw0, const Complex<T>* wave) const {
        const int nx = n/2;
        for(int i = 0 ; i < c_n; i += n)
        {
            Complex<T>* v = dst + i;
            T r0 = v[0].re + v[nx].re;
            T i0 = v[0].im + v[nx].im;
            T r1 = v[0].re - v[nx].re;
            T i1 = v[0].im - v[nx].im;
            v[0].re = r0; v[0].im = i0;
            v[nx].re = r1; v[nx].im = i1;

            for( int j = 1, dw = dw0; j < nx; j++, dw += dw0 )
            {
                v = dst + i + j;
                r1 = v[nx].re*wave[dw].re - v[nx].im*wave[dw].im;
                i1 = v[nx].im*wave[dw].re + v[nx].re*wave[dw].im;
                r0 = v[0].re; i0 = v[0].im;

                v[0].re = r0 + r1; v[0].im = i0 + i1;
                v[nx].re = r0 - r1; v[nx].im = i0 - i1;
            }
        }
    }
};

// Reference radix-3 implementation.
template<typename T> struct DFT_R3
{
    void operator()(Complex<T>* dst, const int c_n, const int n, const int dw0, const Complex<T>* wave) const {
        const int nx = n / 3;
        for(int i = 0; i < c_n; i += n )
        {
            {
                Complex<T>* v = dst + i;
                T r1 = v[nx].re + v[nx*2].re;
                T i1 = v[nx].im + v[nx*2].im;
                T r0 = v[0].re;
                T i0 = v[0].im;
                T r2 = Constants<T>::sin_120*(v[nx].im - v[nx*2].im);
                T i2 = Constants<T>::sin_120*(v[nx*2].re - v[nx].re);
                v[0].re = r0 + r1; v[0].im = i0 + i1;
                r0 -= (T)0.5*r1; i0 -= (T)0.5*i1;
                v[nx].re = r0 + r2; v[nx].im = i0 + i2;
                v[nx*2].re = r0 - r2; v[nx*2].im = i0 - i2;
            }

            for(int j = 1, dw = dw0; j < nx; j++, dw += dw0 )
            {
                Complex<T>* v = dst + i + j;
                T r0 = v[nx].re*wave[dw].re - v[nx].im*wave[dw].im;
                T i0 = v[nx].re*wave[dw].im + v[nx].im*wave[dw].re;
                T i2 = v[nx*2].re*wave[dw*2].re - v[nx*2].im*wave[dw*2].im;
                T r2 = v[nx*2].re*wave[dw*2].im + v[nx*2].im*wave[dw*2].re;
                T r1 = r0 + i2; T i1 = i0 + r2;

                r2 = Constants<T>::sin_120*(i0 - r2); i2 = Constants<T>::sin_120*(i2 - r0);
                r0 = v[0].re; i0 = v[0].im;
                v[0].re = r0 + r1; v[0].im = i0 + i1;
                r0 -= (T)0.5*r1; i0 -= (T)0.5*i1;
                v[nx].re = r0 + r2; v[nx].im = i0 + i2;
                v[nx*2].re = r0 - r2; v[nx*2].im = i0 - i2;
            }
        }
    }
};

// Reference radix-5 implementation.
template<typename T> struct DFT_R5
{
    void operator()(Complex<T>* dst, const int c_n, const int n, const int dw0, const Complex<T>* wave) const {
        const int nx = n / 5;
        for(int i = 0; i < c_n; i += n )
        {
            for(int j = 0, dw = 0; j < nx; j++, dw += dw0 )
            {
                Complex<T>* v0 = dst + i + j;
                Complex<T>* v1 = v0 + nx*2;
                Complex<T>* v2 = v1 + nx*2;

                T r0, i0, r1, i1, r2, i2, r3, i3, r4, i4, r5, i5;

                r3 = v0[nx].re*wave[dw].re - v0[nx].im*wave[dw].im;
                i3 = v0[nx].re*wave[dw].im + v0[nx].im*wave[dw].re;
                r2 = v2[0].re*wave[dw*4].re - v2[0].im*wave[dw*4].im;
                i2 = v2[0].re*wave[dw*4].im + v2[0].im*wave[dw*4].re;

                r1 = r3 + r2; i1 = i3 + i2;
                r3 -= r2; i3 -= i2;

                r4 = v1[nx].re*wave[dw*3].re - v1[nx].im*wave[dw*3].im;
                i4 = v1[nx].re*wave[dw*3].im + v1[nx].im*wave[dw*3].re;
                r0 = v1[0].re*wave[dw*2].re - v1[0].im*wave[dw*2].im;
                i0 = v1[0].re*wave[dw*2].im + v1[0].im*wave[dw*2].re;

                r2 = r4 + r0; i2 = i4 + i0;
                r4 -= r0; i4 -= i0;

                r0 = v0[0].re; i0 = v0[0].im;
                r5 = r1 + r2; i5 = i1 + i2;

                v0[0].re = r0 + r5; v0[0].im = i0 + i5;

                r0 -= (T)0.25*r5; i0 -= (T)0.25*i5;
                r1 = Constants<T>::fft5_2*(r1 - r2); i1 = Constants<T>::fft5_2*(i1 - i2);
                r2 = -Constants<T>::fft5_3*(i3 + i4); i2 = Constants<T>::fft5_3*(r3 + r4);

                i3 *= -Constants<T>::fft5_5; r3 *= Constants<T>::fft5_5;
                i4 *= -Constants<T>::fft5_4; r4 *= Constants<T>::fft5_4;

                r5 = r2 + i3; i5 = i2 + r3;
                r2 -= i4; i2 -= r4;

                r3 = r0 + r1; i3 = i0 + i1;
                r0 -= r1; i0 -= i1;

                v0[nx].re = r3 + r2; v0[nx].im = i3 + i2;
                v2[0].re = r3 - r2; v2[0].im = i3 - i2;

                v1[0].re = r0 + r5; v1[0].im = i0 + i5;
                v1[nx].re = r0 - r5; v1[nx].im = i0 - i5;
            }
        }
    }
};

template<typename T> struct DFT_VecR2
{
    void operator()(Complex<T>* dst, const int c_n, const int n, const int dw0, const Complex<T>* wave) const {
        DFT_R2<T>()(dst, c_n, n, dw0, wave);
    }
};

template<typename T> struct DFT_VecR3
{
    void operator()(Complex<T>* dst, const int c_n, const int n, const int dw0, const Complex<T>* wave) const {
        DFT_R3<T>()(dst, c_n, n, dw0, wave);
    }
};

template<typename T> struct DFT_VecR4
{
    int operator()(Complex<T>*, int, int, int&, const Complex<T>*) const { return 1; }
};

#if CV_SSE3

// multiplies *a and *b:
//  r_re + i*r_im = (a_re + i*a_im)*(b_re + i*b_im)
// r_re and r_im are placed respectively in bits 31:0 and 63:32 of the resulting
// vector register.
inline __m128 complexMul(const Complex<float>* const a, const Complex<float>* const b) {
    const __m128 z = _mm_setzero_ps();
    const __m128 neg_elem0 = _mm_set_ps(0.0f,0.0f,0.0f,-0.0f);
    // v_a[31:0] is a->re and v_a[63:32] is a->im.
    const __m128 v_a = _mm_loadl_pi(z, (const __m64*)a);
    const __m128 v_b = _mm_loadl_pi(z, (const __m64*)b);
    // x_1 = v[nx] * wave[dw].
    const __m128 v_a_riri = _mm_shuffle_ps(v_a, v_a, _MM_SHUFFLE(0, 1, 0, 1));
    const __m128 v_b_irri = _mm_shuffle_ps(v_b, v_b, _MM_SHUFFLE(1, 0, 0, 1));
    const __m128 mul = _mm_mul_ps(v_a_riri, v_b_irri);
    const __m128 xored = _mm_xor_ps(mul, neg_elem0);
    return _mm_hadd_ps(xored, z);
}

// optimized radix-2 transform
template<> struct DFT_VecR2<float> {
    void operator()(Complex<float>* dst, const int c_n, const int n, const int dw0, const Complex<float>* wave) const {
        const __m128 z = _mm_setzero_ps();
        const int nx = n/2;
        for(int i = 0 ; i < c_n; i += n)
        {
            {
                Complex<float>* v = dst + i;
                float r0 = v[0].re + v[nx].re;
                float i0 = v[0].im + v[nx].im;
                float r1 = v[0].re - v[nx].re;
                float i1 = v[0].im - v[nx].im;
                v[0].re = r0; v[0].im = i0;
                v[nx].re = r1; v[nx].im = i1;
            }

            for( int j = 1, dw = dw0; j < nx; j++, dw += dw0 )
            {
                Complex<float>* v = dst + i + j;
                const __m128 x_1 = complexMul(&v[nx], &wave[dw]);
                const __m128 v_0 = _mm_loadl_pi(z, (const __m64*)&v[0]);
                _mm_storel_pi((__m64*)&v[0], _mm_add_ps(v_0, x_1));
                _mm_storel_pi((__m64*)&v[nx], _mm_sub_ps(v_0, x_1));
            }
        }
    }
};

// Optimized radix-3 implementation.
template<> struct DFT_VecR3<float> {
    void operator()(Complex<float>* dst, const int c_n, const int n, const int dw0, const Complex<float>* wave) const {
        const int nx = n / 3;
        const __m128 z = _mm_setzero_ps();
        const __m128 neg_elem1 = _mm_set_ps(0.0f,0.0f,-0.0f,0.0f);
        const __m128 sin_120 = _mm_set1_ps(Constants<float>::sin_120);
        const __m128 one_half = _mm_set1_ps(0.5f);
        for(int i = 0; i < c_n; i += n )
        {
            {
                Complex<float>* v = dst + i;

                float r1 = v[nx].re + v[nx*2].re;
                float i1 = v[nx].im + v[nx*2].im;
                float r0 = v[0].re;
                float i0 = v[0].im;
                float r2 = Constants<float>::sin_120*(v[nx].im - v[nx*2].im);
                float i2 = Constants<float>::sin_120*(v[nx*2].re - v[nx].re);
                v[0].re = r0 + r1; v[0].im = i0 + i1;
                r0 -= (float)0.5*r1; i0 -= (float)0.5*i1;
                v[nx].re = r0 + r2; v[nx].im = i0 + i2;
                v[nx*2].re = r0 - r2; v[nx*2].im = i0 - i2;
            }

            for(int j = 1, dw = dw0; j < nx; j++, dw += dw0 )
            {
                Complex<float>* v = dst + i + j;
                const __m128 x_0 = complexMul(&v[nx], &wave[dw]);
                const __m128 x_2 = complexMul(&v[nx*2], &wave[dw*2]);
                const __m128 x_1 = _mm_add_ps(x_0, x_2);

                const __m128 v_0 = _mm_loadl_pi(z, (const __m64*)&v[0]);
                _mm_storel_pi((__m64*)&v[0], _mm_add_ps(v_0, x_1));

                const __m128 x_3 = _mm_mul_ps(sin_120, _mm_xor_ps(_mm_sub_ps(x_2, x_0), neg_elem1));
                const __m128 x_3s = _mm_shuffle_ps(x_3, x_3, _MM_SHUFFLE(0, 1, 0, 1));
                const __m128 x_4 = _mm_sub_ps(v_0, _mm_mul_ps(one_half, x_1));
                _mm_storel_pi((__m64*)&v[nx], _mm_add_ps(x_4, x_3s));
                _mm_storel_pi((__m64*)&v[nx*2], _mm_sub_ps(x_4, x_3s));
            }
        }
    }
};

// optimized radix-4 transform
template<> struct DFT_VecR4<float>
{
    int operator()(Complex<float>* dst, int N, int n0, int& _dw0, const Complex<float>* wave) const
    {
        int n = 1, i, j, nx, dw, dw0 = _dw0;
        __m128 z = _mm_setzero_ps(), x02=z, x13=z, w01=z, w23=z, y01, y23, t0, t1;
        Cv32suf t; t.i = 0x80000000;
        __m128 neg0_mask = _mm_load_ss(&t.f);
        __m128 neg3_mask = _mm_shuffle_ps(neg0_mask, neg0_mask, _MM_SHUFFLE(0,1,2,3));

        for( ; n*4 <= N; )
        {
            nx = n;
            n *= 4;
            dw0 /= 4;

            for( i = 0; i < n0; i += n )
            {
                Complexf *v0, *v1;

                v0 = dst + i;
                v1 = v0 + nx*2;

                x02 = _mm_loadl_pi(x02, (const __m64*)&v0[0]);
                x13 = _mm_loadl_pi(x13, (const __m64*)&v0[nx]);
                x02 = _mm_loadh_pi(x02, (const __m64*)&v1[0]);
                x13 = _mm_loadh_pi(x13, (const __m64*)&v1[nx]);

                y01 = _mm_add_ps(x02, x13);
                y23 = _mm_sub_ps(x02, x13);
                t1 = _mm_xor_ps(_mm_shuffle_ps(y01, y23, _MM_SHUFFLE(2,3,3,2)), neg3_mask);
                t0 = _mm_movelh_ps(y01, y23);
                y01 = _mm_add_ps(t0, t1);
                y23 = _mm_sub_ps(t0, t1);

                _mm_storel_pi((__m64*)&v0[0], y01);
                _mm_storeh_pi((__m64*)&v0[nx], y01);
                _mm_storel_pi((__m64*)&v1[0], y23);
                _mm_storeh_pi((__m64*)&v1[nx], y23);

                for( j = 1, dw = dw0; j < nx; j++, dw += dw0 )
                {
                    v0 = dst + i + j;
                    v1 = v0 + nx*2;

                    x13 = _mm_loadl_pi(x13, (const __m64*)&v0[nx]);
                    w23 = _mm_loadl_pi(w23, (const __m64*)&wave[dw*2]);
                    x13 = _mm_loadh_pi(x13, (const __m64*)&v1[nx]); // x1, x3 = r1 i1 r3 i3
                    w23 = _mm_loadh_pi(w23, (const __m64*)&wave[dw*3]); // w2, w3 = wr2 wi2 wr3 wi3

                    t0 = _mm_mul_ps(_mm_moveldup_ps(x13), w23);
                    t1 = _mm_mul_ps(_mm_movehdup_ps(x13), _mm_shuffle_ps(w23, w23, _MM_SHUFFLE(2,3,0,1)));
                    x13 = _mm_addsub_ps(t0, t1);
                    // re(x1*w2), im(x1*w2), re(x3*w3), im(x3*w3)
                    x02 = _mm_loadl_pi(x02, (const __m64*)&v1[0]); // x2 = r2 i2
                    w01 = _mm_loadl_pi(w01, (const __m64*)&wave[dw]); // w1 = wr1 wi1
                    x02 = _mm_shuffle_ps(x02, x02, _MM_SHUFFLE(0,0,1,1));
                    w01 = _mm_shuffle_ps(w01, w01, _MM_SHUFFLE(1,0,0,1));
                    x02 = _mm_mul_ps(x02, w01);
                    x02 = _mm_addsub_ps(x02, _mm_movelh_ps(x02, x02));
                    // re(x0) im(x0) re(x2*w1), im(x2*w1)
                    x02 = _mm_loadl_pi(x02, (const __m64*)&v0[0]);

                    y01 = _mm_add_ps(x02, x13);
                    y23 = _mm_sub_ps(x02, x13);
                    t1 = _mm_xor_ps(_mm_shuffle_ps(y01, y23, _MM_SHUFFLE(2,3,3,2)), neg3_mask);
                    t0 = _mm_movelh_ps(y01, y23);
                    y01 = _mm_add_ps(t0, t1);
                    y23 = _mm_sub_ps(t0, t1);

                    _mm_storel_pi((__m64*)&v0[0], y01);
                    _mm_storeh_pi((__m64*)&v0[nx], y01);
                    _mm_storel_pi((__m64*)&v1[0], y23);
                    _mm_storeh_pi((__m64*)&v1[nx], y23);
                }
            }
        }

        _dw0 = dw0;
        return n;
    }
};

#endif

#ifdef USE_IPP_DFT
static IppStatus ippsDFTFwd_CToC( const Complex<float>* src, Complex<float>* dst,
                             const void* spec, uchar* buf)
{
    return CV_INSTRUMENT_FUN_IPP(ippsDFTFwd_CToC_32fc, (const Ipp32fc*)src, (Ipp32fc*)dst,
                                 (const IppsDFTSpec_C_32fc*)spec, buf);
}

static IppStatus ippsDFTFwd_CToC( const Complex<double>* src, Complex<double>* dst,
                             const void* spec, uchar* buf)
{
    return CV_INSTRUMENT_FUN_IPP(ippsDFTFwd_CToC_64fc, (const Ipp64fc*)src, (Ipp64fc*)dst,
                                 (const IppsDFTSpec_C_64fc*)spec, buf);
}

static IppStatus ippsDFTInv_CToC( const Complex<float>* src, Complex<float>* dst,
                             const void* spec, uchar* buf)
{
    return CV_INSTRUMENT_FUN_IPP(ippsDFTInv_CToC_32fc, (const Ipp32fc*)src, (Ipp32fc*)dst,
                                 (const IppsDFTSpec_C_32fc*)spec, buf);
}

static IppStatus ippsDFTInv_CToC( const Complex<double>* src, Complex<double>* dst,
                                  const void* spec, uchar* buf)
{
    return CV_INSTRUMENT_FUN_IPP(ippsDFTInv_CToC_64fc, (const Ipp64fc*)src, (Ipp64fc*)dst,
                                 (const IppsDFTSpec_C_64fc*)spec, buf);
}

static IppStatus ippsDFTFwd_RToPack( const float* src, float* dst,
                                     const void* spec, uchar* buf)
{
    return CV_INSTRUMENT_FUN_IPP(ippsDFTFwd_RToPack_32f, src, dst, (const IppsDFTSpec_R_32f*)spec, buf);
}

static IppStatus ippsDFTFwd_RToPack( const double* src, double* dst,
                                     const void* spec, uchar* buf)
{
    return CV_INSTRUMENT_FUN_IPP(ippsDFTFwd_RToPack_64f, src, dst, (const IppsDFTSpec_R_64f*)spec, buf);
}

static IppStatus ippsDFTInv_PackToR( const float* src, float* dst,
                                     const void* spec, uchar* buf)
{
    return CV_INSTRUMENT_FUN_IPP(ippsDFTInv_PackToR_32f, src, dst, (const IppsDFTSpec_R_32f*)spec, buf);
}

static IppStatus ippsDFTInv_PackToR( const double* src, double* dst,
                                     const void* spec, uchar* buf)
{
    return CV_INSTRUMENT_FUN_IPP(ippsDFTInv_PackToR_64f, src, dst, (const IppsDFTSpec_R_64f*)spec, buf);
}
#endif

struct OcvDftOptions;

typedef void (*DFTFunc)(const OcvDftOptions & c, const void* src, void* dst);

struct OcvDftOptions {
    int nf;
    int *factors;
    double scale;

    int* itab;
    void* wave;
    int tab_size;
    int n;

    bool isInverse;
    bool noPermute;
    bool isComplex;

    bool haveSSE3;

    DFTFunc dft_func;
    bool useIpp;

#ifdef USE_IPP_DFT
    uchar* ipp_spec;
    uchar* ipp_work;
#endif

    OcvDftOptions()
    {
        nf = 0;
        factors = 0;
        scale = 0;
        itab = 0;
        wave = 0;
        tab_size = 0;
        n = 0;
        isInverse = false;
        noPermute = false;
        isComplex = false;
        useIpp = false;
#ifdef USE_IPP_DFT
        ipp_spec = 0;
        ipp_work = 0;
#endif
        dft_func = 0;
        haveSSE3 = checkHardwareSupport(CV_CPU_SSE3);
    }
};

// mixed-radix complex discrete Fourier transform: double-precision version
template<typename T> static void
DFT(const OcvDftOptions & c, const Complex<T>* src, Complex<T>* dst)
{
    const Complex<T>* wave = (Complex<T>*)c.wave;
    const int * itab = c.itab;

    int n = c.n;
    int f_idx, nx;
    int inv = c.isInverse;
    int dw0 = c.tab_size, dw;
    int i, j, k;
    Complex<T> t;
    T scale = (T)c.scale;

    if(typeid(T) == typeid(float))
    {
        CALL_HAL(dft, cv_hal_dft, reinterpret_cast<const uchar*>(src), reinterpret_cast<uchar*>(dst), CV_32F,
                 c.nf, c.factors, c.scale, c.itab, c.wave, c.tab_size, c.n, c.isInverse, c.noPermute);
    }
    if(typeid(T) == typeid(double))
    {
        CALL_HAL(dft, cv_hal_dft, reinterpret_cast<const uchar*>(src), reinterpret_cast<uchar*>(dst), CV_64F,
                 c.nf, c.factors, c.scale, c.itab, c.wave, c.tab_size, c.n, c.isInverse, c.noPermute);
    }

    if( c.useIpp )
    {
#ifdef USE_IPP_DFT
        if( !inv )
        {
            if (ippsDFTFwd_CToC( src, dst, c.ipp_spec, c.ipp_work ) >= 0)
            {
                CV_IMPL_ADD(CV_IMPL_IPP);
                return;
            }
        }
        else
        {
            if (ippsDFTInv_CToC( src, dst, c.ipp_spec, c.ipp_work ) >= 0)
            {
                CV_IMPL_ADD(CV_IMPL_IPP);
                return;
            }
        }
        setIppErrorStatus();
#endif
    }

    int tab_step = c.tab_size == n ? 1 : c.tab_size == n*2 ? 2 : c.tab_size/n;

    // 0. shuffle data
    if( dst != src )
    {
        CV_Assert( !c.noPermute );
        if( !inv )
        {
            for( i = 0; i <= n - 2; i += 2, itab += 2*tab_step )
            {
                int k0 = itab[0], k1 = itab[tab_step];
                CV_Assert( (unsigned)k0 < (unsigned)n && (unsigned)k1 < (unsigned)n );
                dst[i] = src[k0]; dst[i+1] = src[k1];
            }

            if( i < n )
                dst[n-1] = src[n-1];
        }
        else
        {
            for( i = 0; i <= n - 2; i += 2, itab += 2*tab_step )
            {
                int k0 = itab[0], k1 = itab[tab_step];
                CV_Assert( (unsigned)k0 < (unsigned)n && (unsigned)k1 < (unsigned)n );
                t.re = src[k0].re; t.im = -src[k0].im;
                dst[i] = t;
                t.re = src[k1].re; t.im = -src[k1].im;
                dst[i+1] = t;
            }

            if( i < n )
            {
                t.re = src[n-1].re; t.im = -src[n-1].im;
                dst[i] = t;
            }
        }
    }
    else
    {
        if( !c.noPermute )
        {
            CV_Assert( c.factors[0] == c.factors[c.nf-1] );
            if( c.nf == 1 )
            {
                if( (n & 3) == 0 )
                {
                    int n2 = n/2;
                    Complex<T>* dsth = dst + n2;

                    for( i = 0; i < n2; i += 2, itab += tab_step*2 )
                    {
                        j = itab[0];
                        CV_Assert( (unsigned)j < (unsigned)n2 );

                        CV_SWAP(dst[i+1], dsth[j], t);
                        if( j > i )
                        {
                            CV_SWAP(dst[i], dst[j], t);
                            CV_SWAP(dsth[i+1], dsth[j+1], t);
                        }
                    }
                }
                // else do nothing
            }
            else
            {
                for( i = 0; i < n; i++, itab += tab_step )
                {
                    j = itab[0];
                    CV_Assert( (unsigned)j < (unsigned)n );
                    if( j > i )
                        CV_SWAP(dst[i], dst[j], t);
                }
            }
        }

        if( inv )
        {
            for( i = 0; i <= n - 2; i += 2 )
            {
                T t0 = -dst[i].im;
                T t1 = -dst[i+1].im;
                dst[i].im = t0; dst[i+1].im = t1;
            }

            if( i < n )
                dst[n-1].im = -dst[n-1].im;
        }
    }

    n = 1;
    // 1. power-2 transforms
    if( (c.factors[0] & 1) == 0 )
    {
        if( c.factors[0] >= 4 && c.haveSSE3)
        {
            DFT_VecR4<T> vr4;
            n = vr4(dst, c.factors[0], c.n, dw0, wave);
        }

        // radix-4 transform
        for( ; n*4 <= c.factors[0]; )
        {
            nx = n;
            n *= 4;
            dw0 /= 4;

            for( i = 0; i < c.n; i += n )
            {
                Complex<T> *v0, *v1;
                T r0, i0, r1, i1, r2, i2, r3, i3, r4, i4;

                v0 = dst + i;
                v1 = v0 + nx*2;

                r0 = v1[0].re; i0 = v1[0].im;
                r4 = v1[nx].re; i4 = v1[nx].im;

                r1 = r0 + r4; i1 = i0 + i4;
                r3 = i0 - i4; i3 = r4 - r0;

                r2 = v0[0].re; i2 = v0[0].im;
                r4 = v0[nx].re; i4 = v0[nx].im;

                r0 = r2 + r4; i0 = i2 + i4;
                r2 -= r4; i2 -= i4;

                v0[0].re = r0 + r1; v0[0].im = i0 + i1;
                v1[0].re = r0 - r1; v1[0].im = i0 - i1;
                v0[nx].re = r2 + r3; v0[nx].im = i2 + i3;
                v1[nx].re = r2 - r3; v1[nx].im = i2 - i3;

                for( j = 1, dw = dw0; j < nx; j++, dw += dw0 )
                {
                    v0 = dst + i + j;
                    v1 = v0 + nx*2;

                    r2 = v0[nx].re*wave[dw*2].re - v0[nx].im*wave[dw*2].im;
                    i2 = v0[nx].re*wave[dw*2].im + v0[nx].im*wave[dw*2].re;
                    r0 = v1[0].re*wave[dw].im + v1[0].im*wave[dw].re;
                    i0 = v1[0].re*wave[dw].re - v1[0].im*wave[dw].im;
                    r3 = v1[nx].re*wave[dw*3].im + v1[nx].im*wave[dw*3].re;
                    i3 = v1[nx].re*wave[dw*3].re - v1[nx].im*wave[dw*3].im;

                    r1 = i0 + i3; i1 = r0 + r3;
                    r3 = r0 - r3; i3 = i3 - i0;
                    r4 = v0[0].re; i4 = v0[0].im;

                    r0 = r4 + r2; i0 = i4 + i2;
                    r2 = r4 - r2; i2 = i4 - i2;

                    v0[0].re = r0 + r1; v0[0].im = i0 + i1;
                    v1[0].re = r0 - r1; v1[0].im = i0 - i1;
                    v0[nx].re = r2 + r3; v0[nx].im = i2 + i3;
                    v1[nx].re = r2 - r3; v1[nx].im = i2 - i3;
                }
            }
        }

        for( ; n < c.factors[0]; )
        {
            // do the remaining radix-2 transform
            n *= 2;
            dw0 /= 2;

            if(c.haveSSE3)
            {
                DFT_VecR2<T> vr2;
                vr2(dst, c.n, n, dw0, wave);
            }
            else
            {
                DFT_R2<T> vr2;
                vr2(dst, c.n, n, dw0, wave);
            }
        }
    }

    // 2. all the other transforms
    for( f_idx = (c.factors[0]&1) ? 0 : 1; f_idx < c.nf; f_idx++ )
    {
        int factor = c.factors[f_idx];
        nx = n;
        n *= factor;
        dw0 /= factor;

        if( factor == 3 )
        {
            if(c.haveSSE3)
            {
                DFT_VecR3<T> vr3;
                vr3(dst, c.n, n, dw0, wave);
            }
            else
            {
                DFT_R3<T> vr3;
                vr3(dst, c.n, n, dw0, wave);
            }
        }
        else if( factor == 5 )
        {
            DFT_R5<T> vr5;
            vr5(dst, c.n, n, dw0, wave);
        }
        else
        {
            // radix-"factor" - an odd number
            int p, q, factor2 = (factor - 1)/2;
            int d, dd, dw_f = c.tab_size/factor;
            AutoBuffer<Complex<T> > buf(factor2 * 2);
            Complex<T>* a = buf.data();
            Complex<T>* b = a + factor2;

            for( i = 0; i < c.n; i += n )
            {
                for( j = 0, dw = 0; j < nx; j++, dw += dw0 )
                {
                    Complex<T>* v = dst + i + j;
                    Complex<T> v_0 = v[0];
                    Complex<T> vn_0 = v_0;

                    if( j == 0 )
                    {
                        for( p = 1, k = nx; p <= factor2; p++, k += nx )
                        {
                            T r0 = v[k].re + v[n-k].re;
                            T i0 = v[k].im - v[n-k].im;
                            T r1 = v[k].re - v[n-k].re;
                            T i1 = v[k].im + v[n-k].im;

                            vn_0.re += r0; vn_0.im += i1;
                            a[p-1].re = r0; a[p-1].im = i0;
                            b[p-1].re = r1; b[p-1].im = i1;
                        }
                    }
                    else
                    {
                        const Complex<T>* wave_ = wave + dw*factor;
                        d = dw;

                        for( p = 1, k = nx; p <= factor2; p++, k += nx, d += dw )
                        {
                            T r2 = v[k].re*wave[d].re - v[k].im*wave[d].im;
                            T i2 = v[k].re*wave[d].im + v[k].im*wave[d].re;

                            T r1 = v[n-k].re*wave_[-d].re - v[n-k].im*wave_[-d].im;
                            T i1 = v[n-k].re*wave_[-d].im + v[n-k].im*wave_[-d].re;

                            T r0 = r2 + r1;
                            T i0 = i2 - i1;
                            r1 = r2 - r1;
                            i1 = i2 + i1;

                            vn_0.re += r0; vn_0.im += i1;
                            a[p-1].re = r0; a[p-1].im = i0;
                            b[p-1].re = r1; b[p-1].im = i1;
                        }
                    }

                    v[0] = vn_0;

                    for( p = 1, k = nx; p <= factor2; p++, k += nx )
                    {
                        Complex<T> s0 = v_0, s1 = v_0;
                        d = dd = dw_f*p;

                        for( q = 0; q < factor2; q++ )
                        {
                            T r0 = wave[d].re * a[q].re;
                            T i0 = wave[d].im * a[q].im;
                            T r1 = wave[d].re * b[q].im;
                            T i1 = wave[d].im * b[q].re;

                            s1.re += r0 + i0; s0.re += r0 - i0;
                            s1.im += r1 - i1; s0.im += r1 + i1;

                            d += dd;
                            d -= -(d >= c.tab_size) & c.tab_size;
                        }

                        v[k] = s0;
                        v[n-k] = s1;
                    }
                }
            }
        }
    }

    if( scale != 1 )
    {
        T re_scale = scale, im_scale = scale;
        if( inv )
            im_scale = -im_scale;

        for( i = 0; i < c.n; i++ )
        {
            T t0 = dst[i].re*re_scale;
            T t1 = dst[i].im*im_scale;
            dst[i].re = t0;
            dst[i].im = t1;
        }
    }
    else if( inv )
    {
        for( i = 0; i <= c.n - 2; i += 2 )
        {
            T t0 = -dst[i].im;
            T t1 = -dst[i+1].im;
            dst[i].im = t0;
            dst[i+1].im = t1;
        }

        if( i < c.n )
            dst[c.n-1].im = -dst[c.n-1].im;
    }
}


/* FFT of real vector
   output vector format:
     re(0), re(1), im(1), ... , re(n/2-1), im((n+1)/2-1) [, re((n+1)/2)] OR ...
     re(0), 0, re(1), im(1), ..., re(n/2-1), im((n+1)/2-1) [, re((n+1)/2), 0] */
template<typename T> static void
RealDFT(const OcvDftOptions & c, const T* src, T* dst)
{
    int n = c.n;
    int complex_output = c.isComplex;
    T scale = (T)c.scale;
    int j;
    dst += complex_output;

    if( c.useIpp )
    {
#ifdef USE_IPP_DFT
        if (ippsDFTFwd_RToPack( src, dst, c.ipp_spec, c.ipp_work ) >=0)
        {
            if( complex_output )
            {
                dst[-1] = dst[0];
                dst[0] = 0;
                if( (n & 1) == 0 )
                    dst[n] = 0;
            }
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }
        setIppErrorStatus();
#endif
    }
    CV_Assert( c.tab_size == n );

    if( n == 1 )
    {
        dst[0] = src[0]*scale;
    }
    else if( n == 2 )
    {
        T t = (src[0] + src[1])*scale;
        dst[1] = (src[0] - src[1])*scale;
        dst[0] = t;
    }
    else if( n & 1 )
    {
        dst -= complex_output;
        Complex<T>* _dst = (Complex<T>*)dst;
        _dst[0].re = src[0]*scale;
        _dst[0].im = 0;
        for( j = 1; j < n; j += 2 )
        {
            T t0 = src[c.itab[j]]*scale;
            T t1 = src[c.itab[j+1]]*scale;
            _dst[j].re = t0;
            _dst[j].im = 0;
            _dst[j+1].re = t1;
            _dst[j+1].im = 0;
        }
        OcvDftOptions sub_c = c;
        sub_c.isComplex = false;
        sub_c.isInverse = false;
        sub_c.noPermute = true;
        sub_c.scale = 1.;
        DFT(sub_c, _dst, _dst);
        if( !complex_output )
            dst[1] = dst[0];
    }
    else
    {
        T t0, t;
        T h1_re, h1_im, h2_re, h2_im;
        T scale2 = scale*(T)0.5;
        int n2 = n >> 1;

        c.factors[0] >>= 1;

        OcvDftOptions sub_c = c;
        sub_c.factors += (c.factors[0] == 1);
        sub_c.nf -= (c.factors[0] == 1);
        sub_c.isComplex = false;
        sub_c.isInverse = false;
        sub_c.noPermute = false;
        sub_c.scale = 1.;
        sub_c.n = n2;

        DFT(sub_c, (Complex<T>*)src, (Complex<T>*)dst);

        c.factors[0] <<= 1;

        t = dst[0] - dst[1];
        dst[0] = (dst[0] + dst[1])*scale;
        dst[1] = t*scale;

        t0 = dst[n2];
        t = dst[n-1];
        dst[n-1] = dst[1];

        const Complex<T> *wave = (const Complex<T>*)c.wave;

        for( j = 2, wave++; j < n2; j += 2, wave++ )
        {
            /* calc odd */
            h2_re = scale2*(dst[j+1] + t);
            h2_im = scale2*(dst[n-j] - dst[j]);

            /* calc even */
            h1_re = scale2*(dst[j] + dst[n-j]);
            h1_im = scale2*(dst[j+1] - t);

            /* rotate */
            t = h2_re*wave->re - h2_im*wave->im;
            h2_im = h2_re*wave->im + h2_im*wave->re;
            h2_re = t;
            t = dst[n-j-1];

            dst[j-1] = h1_re + h2_re;
            dst[n-j-1] = h1_re - h2_re;
            dst[j] = h1_im + h2_im;
            dst[n-j] = h2_im - h1_im;
        }

        if( j <= n2 )
        {
            dst[n2-1] = t0*scale;
            dst[n2] = -t*scale;
        }
    }

    if( complex_output && ((n & 1) == 0 || n == 1))
    {
        dst[-1] = dst[0];
        dst[0] = 0;
        if( n > 1 )
            dst[n] = 0;
    }
}

/* Inverse FFT of complex conjugate-symmetric vector
   input vector format:
      re[0], re[1], im[1], ... , re[n/2-1], im[n/2-1], re[n/2] OR
      re(0), 0, re(1), im(1), ..., re(n/2-1), im((n+1)/2-1) [, re((n+1)/2), 0] */
template<typename T> static void
CCSIDFT(const OcvDftOptions & c, const T* src, T* dst)
{
    int n = c.n;
    int complex_input = c.isComplex;
    int j, k;
    T scale = (T)c.scale;
    T save_s1 = 0.;
    T t0, t1, t2, t3, t;

    CV_Assert( c.tab_size == n );

    if( complex_input )
    {
        CV_Assert( src != dst );
        save_s1 = src[1];
        ((T*)src)[1] = src[0];
        src++;
    }
    if( c.useIpp )
    {
#ifdef USE_IPP_DFT
        if (ippsDFTInv_PackToR( src, dst, c.ipp_spec, c.ipp_work ) >=0)
        {
            if( complex_input )
                ((T*)src)[0] = (T)save_s1;
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }

        setIppErrorStatus();
#endif
    }
    if( n == 1 )
    {
        dst[0] = (T)(src[0]*scale);
    }
    else if( n == 2 )
    {
        t = (src[0] + src[1])*scale;
        dst[1] = (src[0] - src[1])*scale;
        dst[0] = t;
    }
    else if( n & 1 )
    {
        Complex<T>* _src = (Complex<T>*)(src-1);
        Complex<T>* _dst = (Complex<T>*)dst;

        _dst[0].re = src[0];
        _dst[0].im = 0;

        int n2 = (n+1) >> 1;

        for( j = 1; j < n2; j++ )
        {
            int k0 = c.itab[j], k1 = c.itab[n-j];
            t0 = _src[j].re; t1 = _src[j].im;
            _dst[k0].re = t0; _dst[k0].im = -t1;
            _dst[k1].re = t0; _dst[k1].im = t1;
        }

        OcvDftOptions sub_c = c;
        sub_c.isComplex = false;
        sub_c.isInverse = false;
        sub_c.noPermute = true;
        sub_c.scale = 1.;
        sub_c.n = n;

        DFT(sub_c, _dst, _dst);
        dst[0] *= scale;
        for( j = 1; j < n; j += 2 )
        {
            t0 = dst[j*2]*scale;
            t1 = dst[j*2+2]*scale;
            dst[j] = t0;
            dst[j+1] = t1;
        }
    }
    else
    {
        int inplace = src == dst;
        const Complex<T>* w = (const Complex<T>*)c.wave;

        t = src[1];
        t0 = (src[0] + src[n-1]);
        t1 = (src[n-1] - src[0]);
        dst[0] = t0;
        dst[1] = t1;

        int n2 = (n+1) >> 1;

        for( j = 2, w++; j < n2; j += 2, w++ )
        {
            T h1_re, h1_im, h2_re, h2_im;

            h1_re = (t + src[n-j-1]);
            h1_im = (src[j] - src[n-j]);

            h2_re = (t - src[n-j-1]);
            h2_im = (src[j] + src[n-j]);

            t = h2_re*w->re + h2_im*w->im;
            h2_im = h2_im*w->re - h2_re*w->im;
            h2_re = t;

            t = src[j+1];
            t0 = h1_re - h2_im;
            t1 = -h1_im - h2_re;
            t2 = h1_re + h2_im;
            t3 = h1_im - h2_re;

            if( inplace )
            {
                dst[j] = t0;
                dst[j+1] = t1;
                dst[n-j] = t2;
                dst[n-j+1]= t3;
            }
            else
            {
                int j2 = j >> 1;
                k = c.itab[j2];
                dst[k] = t0;
                dst[k+1] = t1;
                k = c.itab[n2-j2];
                dst[k] = t2;
                dst[k+1]= t3;
            }
        }

        if( j <= n2 )
        {
            t0 = t*2;
            t1 = src[n2]*2;

            if( inplace )
            {
                dst[n2] = t0;
                dst[n2+1] = t1;
            }
            else
            {
                k = c.itab[n2];
                dst[k*2] = t0;
                dst[k*2+1] = t1;
            }
        }

        c.factors[0] >>= 1;

        OcvDftOptions sub_c = c;
        sub_c.factors += (c.factors[0] == 1);
        sub_c.nf -= (c.factors[0] == 1);
        sub_c.isComplex = false;
        sub_c.isInverse = false;
        sub_c.noPermute = !inplace;
        sub_c.scale = 1.;
        sub_c.n = n2;

        DFT(sub_c, (Complex<T>*)dst, (Complex<T>*)dst);

        c.factors[0] <<= 1;

        for( j = 0; j < n; j += 2 )
        {
            t0 = dst[j]*scale;
            t1 = dst[j+1]*(-scale);
            dst[j] = t0;
            dst[j+1] = t1;
        }
    }
    if( complex_input )
        ((T*)src)[0] = (T)save_s1;
}

static void
CopyColumn( const uchar* _src, size_t src_step,
            uchar* _dst, size_t dst_step,
            int len, size_t elem_size )
{
    int i, t0, t1;
    const int* src = (const int*)_src;
    int* dst = (int*)_dst;
    src_step /= sizeof(src[0]);
    dst_step /= sizeof(dst[0]);

    if( elem_size == sizeof(int) )
    {
        for( i = 0; i < len; i++, src += src_step, dst += dst_step )
            dst[0] = src[0];
    }
    else if( elem_size == sizeof(int)*2 )
    {
        for( i = 0; i < len; i++, src += src_step, dst += dst_step )
        {
            t0 = src[0]; t1 = src[1];
            dst[0] = t0; dst[1] = t1;
        }
    }
    else if( elem_size == sizeof(int)*4 )
    {
        for( i = 0; i < len; i++, src += src_step, dst += dst_step )
        {
            t0 = src[0]; t1 = src[1];
            dst[0] = t0; dst[1] = t1;
            t0 = src[2]; t1 = src[3];
            dst[2] = t0; dst[3] = t1;
        }
    }
}


static void
CopyFrom2Columns( const uchar* _src, size_t src_step,
                  uchar* _dst0, uchar* _dst1,
                  int len, size_t elem_size )
{
    int i, t0, t1;
    const int* src = (const int*)_src;
    int* dst0 = (int*)_dst0;
    int* dst1 = (int*)_dst1;
    src_step /= sizeof(src[0]);

    if( elem_size == sizeof(int) )
    {
        for( i = 0; i < len; i++, src += src_step )
        {
            t0 = src[0]; t1 = src[1];
            dst0[i] = t0; dst1[i] = t1;
        }
    }
    else if( elem_size == sizeof(int)*2 )
    {
        for( i = 0; i < len*2; i += 2, src += src_step )
        {
            t0 = src[0]; t1 = src[1];
            dst0[i] = t0; dst0[i+1] = t1;
            t0 = src[2]; t1 = src[3];
            dst1[i] = t0; dst1[i+1] = t1;
        }
    }
    else if( elem_size == sizeof(int)*4 )
    {
        for( i = 0; i < len*4; i += 4, src += src_step )
        {
            t0 = src[0]; t1 = src[1];
            dst0[i] = t0; dst0[i+1] = t1;
            t0 = src[2]; t1 = src[3];
            dst0[i+2] = t0; dst0[i+3] = t1;
            t0 = src[4]; t1 = src[5];
            dst1[i] = t0; dst1[i+1] = t1;
            t0 = src[6]; t1 = src[7];
            dst1[i+2] = t0; dst1[i+3] = t1;
        }
    }
}


static void
CopyTo2Columns( const uchar* _src0, const uchar* _src1,
                uchar* _dst, size_t dst_step,
                int len, size_t elem_size )
{
    int i, t0, t1;
    const int* src0 = (const int*)_src0;
    const int* src1 = (const int*)_src1;
    int* dst = (int*)_dst;
    dst_step /= sizeof(dst[0]);

    if( elem_size == sizeof(int) )
    {
        for( i = 0; i < len; i++, dst += dst_step )
        {
            t0 = src0[i]; t1 = src1[i];
            dst[0] = t0; dst[1] = t1;
        }
    }
    else if( elem_size == sizeof(int)*2 )
    {
        for( i = 0; i < len*2; i += 2, dst += dst_step )
        {
            t0 = src0[i]; t1 = src0[i+1];
            dst[0] = t0; dst[1] = t1;
            t0 = src1[i]; t1 = src1[i+1];
            dst[2] = t0; dst[3] = t1;
        }
    }
    else if( elem_size == sizeof(int)*4 )
    {
        for( i = 0; i < len*4; i += 4, dst += dst_step )
        {
            t0 = src0[i]; t1 = src0[i+1];
            dst[0] = t0; dst[1] = t1;
            t0 = src0[i+2]; t1 = src0[i+3];
            dst[2] = t0; dst[3] = t1;
            t0 = src1[i]; t1 = src1[i+1];
            dst[4] = t0; dst[5] = t1;
            t0 = src1[i+2]; t1 = src1[i+3];
            dst[6] = t0; dst[7] = t1;
        }
    }
}


static void
ExpandCCS( uchar* _ptr, int n, int elem_size )
{
    int i;
    if( elem_size == (int)sizeof(float) )
    {
        float* p = (float*)_ptr;
        for( i = 1; i < (n+1)/2; i++ )
        {
            p[(n-i)*2] = p[i*2-1];
            p[(n-i)*2+1] = -p[i*2];
        }
        if( (n & 1) == 0 )
        {
            p[n] = p[n-1];
            p[n+1] = 0.f;
            n--;
        }
        for( i = n-1; i > 0; i-- )
            p[i+1] = p[i];
        p[1] = 0.f;
    }
    else
    {
        double* p = (double*)_ptr;
        for( i = 1; i < (n+1)/2; i++ )
        {
            p[(n-i)*2] = p[i*2-1];
            p[(n-i)*2+1] = -p[i*2];
        }
        if( (n & 1) == 0 )
        {
            p[n] = p[n-1];
            p[n+1] = 0.f;
            n--;
        }
        for( i = n-1; i > 0; i-- )
            p[i+1] = p[i];
        p[1] = 0.f;
    }
}

static void DFT_32f(const OcvDftOptions & c, const Complexf* src, Complexf* dst)
{
    DFT(c, src, dst);
}

static void DFT_64f(const OcvDftOptions & c, const Complexd* src, Complexd* dst)
{
    DFT(c, src, dst);
}


static void RealDFT_32f(const OcvDftOptions & c, const float* src, float* dst)
{
    RealDFT(c, src, dst);
}

static void RealDFT_64f(const OcvDftOptions & c, const double* src, double* dst)
{
    RealDFT(c, src, dst);
}

static void CCSIDFT_32f(const OcvDftOptions & c, const float* src, float* dst)
{
    CCSIDFT(c, src, dst);
}

static void CCSIDFT_64f(const OcvDftOptions & c, const double* src, double* dst)
{
    CCSIDFT(c, src, dst);
}

}

#ifdef USE_IPP_DFT
typedef IppStatus (CV_STDCALL* IppDFTGetSizeFunc)(int, int, IppHintAlgorithm, int*, int*, int*);
typedef IppStatus (CV_STDCALL* IppDFTInitFunc)(int, int, IppHintAlgorithm, void*, uchar*);
#endif

namespace cv
{
#if defined USE_IPP_DFT

typedef IppStatus (CV_STDCALL* ippiDFT_C_Func)(const Ipp32fc*, int, Ipp32fc*, int, const IppiDFTSpec_C_32fc*, Ipp8u*);
typedef IppStatus (CV_STDCALL* ippiDFT_R_Func)(const Ipp32f* , int, Ipp32f* , int, const IppiDFTSpec_R_32f* , Ipp8u*);

template <typename Dft>
class Dft_C_IPPLoop_Invoker : public ParallelLoopBody
{
public:

    Dft_C_IPPLoop_Invoker(const uchar * _src, size_t _src_step, uchar * _dst, size_t _dst_step, int _width,
                          const Dft& _ippidft, int _norm_flag, bool *_ok) :
        ParallelLoopBody(),
        src(_src), src_step(_src_step), dst(_dst), dst_step(_dst_step), width(_width),
        ippidft(_ippidft), norm_flag(_norm_flag), ok(_ok)
    {
        *ok = true;
    }

    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        IppStatus status;
        Ipp8u* pBuffer = 0;
        Ipp8u* pMemInit= 0;
        int sizeBuffer=0;
        int sizeSpec=0;
        int sizeInit=0;

        IppiSize srcRoiSize = {width, 1};

        status = ippiDFTGetSize_C_32fc(srcRoiSize, norm_flag, ippAlgHintNone, &sizeSpec, &sizeInit, &sizeBuffer );
        if ( status < 0 )
        {
            *ok = false;
            return;
        }

        IppiDFTSpec_C_32fc* pDFTSpec = (IppiDFTSpec_C_32fc*)CV_IPP_MALLOC( sizeSpec );

        if ( sizeInit > 0 )
            pMemInit = (Ipp8u*)CV_IPP_MALLOC( sizeInit );

        if ( sizeBuffer > 0 )
            pBuffer = (Ipp8u*)CV_IPP_MALLOC( sizeBuffer );

        status = ippiDFTInit_C_32fc( srcRoiSize, norm_flag, ippAlgHintNone, pDFTSpec, pMemInit );

        if ( sizeInit > 0 )
            ippFree( pMemInit );

        if ( status < 0 )
        {
            ippFree( pDFTSpec );
            if ( sizeBuffer > 0 )
                ippFree( pBuffer );
            *ok = false;
            return;
        }

        for( int i = range.start; i < range.end; ++i)
            if(!ippidft((Ipp32fc*)(src + src_step * i), src_step, (Ipp32fc*)(dst + dst_step * i), dst_step,
                        pDFTSpec, (Ipp8u*)pBuffer))
            {
                *ok = false;
            }

        if ( sizeBuffer > 0 )
            ippFree( pBuffer );

        ippFree( pDFTSpec );
        CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
    }

private:
    const uchar * src;
    size_t src_step;
    uchar * dst;
    size_t dst_step;
    int width;
    const Dft& ippidft;
    int norm_flag;
    bool *ok;

    const Dft_C_IPPLoop_Invoker& operator= (const Dft_C_IPPLoop_Invoker&);
};

template <typename Dft>
class Dft_R_IPPLoop_Invoker : public ParallelLoopBody
{
public:

    Dft_R_IPPLoop_Invoker(const uchar * _src, size_t _src_step, uchar * _dst, size_t _dst_step, int _width,
                          const Dft& _ippidft, int _norm_flag, bool *_ok) :
        ParallelLoopBody(),
        src(_src), src_step(_src_step), dst(_dst), dst_step(_dst_step), width(_width),
        ippidft(_ippidft), norm_flag(_norm_flag), ok(_ok)
    {
        *ok = true;
    }

    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        IppStatus status;
        Ipp8u* pBuffer = 0;
        Ipp8u* pMemInit= 0;
        int sizeBuffer=0;
        int sizeSpec=0;
        int sizeInit=0;

        IppiSize srcRoiSize = {width, 1};

        status = ippiDFTGetSize_R_32f(srcRoiSize, norm_flag, ippAlgHintNone, &sizeSpec, &sizeInit, &sizeBuffer );
        if ( status < 0 )
        {
            *ok = false;
            return;
        }

        IppiDFTSpec_R_32f* pDFTSpec = (IppiDFTSpec_R_32f*)CV_IPP_MALLOC( sizeSpec );

        if ( sizeInit > 0 )
            pMemInit = (Ipp8u*)CV_IPP_MALLOC( sizeInit );

        if ( sizeBuffer > 0 )
            pBuffer = (Ipp8u*)CV_IPP_MALLOC( sizeBuffer );

        status = ippiDFTInit_R_32f( srcRoiSize, norm_flag, ippAlgHintNone, pDFTSpec, pMemInit );

        if ( sizeInit > 0 )
            ippFree( pMemInit );

        if ( status < 0 )
        {
            ippFree( pDFTSpec );
            if ( sizeBuffer > 0 )
                ippFree( pBuffer );
            *ok = false;
            return;
        }

        for( int i = range.start; i < range.end; ++i)
            if(!ippidft((float*)(src + src_step * i), src_step, (float*)(dst + dst_step * i), dst_step,
                        pDFTSpec, (Ipp8u*)pBuffer))
            {
                *ok = false;
            }

        if ( sizeBuffer > 0 )
            ippFree( pBuffer );

        ippFree( pDFTSpec );
        CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
    }

private:
    const uchar * src;
    size_t src_step;
    uchar * dst;
    size_t dst_step;
    int width;
    const Dft& ippidft;
    int norm_flag;
    bool *ok;

    const Dft_R_IPPLoop_Invoker& operator= (const Dft_R_IPPLoop_Invoker&);
};

template <typename Dft>
bool Dft_C_IPPLoop(const uchar * src, size_t src_step, uchar * dst, size_t dst_step, int width, int height, const Dft& ippidft, int norm_flag)
{
    bool ok;
    parallel_for_(Range(0, height), Dft_C_IPPLoop_Invoker<Dft>(src, src_step, dst, dst_step, width, ippidft, norm_flag, &ok), (width * height)/(double)(1<<16) );
    return ok;
}

template <typename Dft>
bool Dft_R_IPPLoop(const uchar * src, size_t src_step, uchar * dst, size_t dst_step, int width, int height, const Dft& ippidft, int norm_flag)
{
    bool ok;
    parallel_for_(Range(0, height), Dft_R_IPPLoop_Invoker<Dft>(src, src_step, dst, dst_step, width, ippidft, norm_flag, &ok), (width * height)/(double)(1<<16) );
    return ok;
}

struct IPPDFT_C_Functor
{
    IPPDFT_C_Functor(ippiDFT_C_Func _func) : ippiDFT_CToC_32fc_C1R(_func){}

    bool operator()(const Ipp32fc* src, size_t srcStep, Ipp32fc* dst, size_t dstStep, const IppiDFTSpec_C_32fc* pDFTSpec, Ipp8u* pBuffer) const
    {
        return ippiDFT_CToC_32fc_C1R ? CV_INSTRUMENT_FUN_IPP(ippiDFT_CToC_32fc_C1R, src, static_cast<int>(srcStep), dst, static_cast<int>(dstStep), pDFTSpec, pBuffer) >= 0 : false;
    }
private:
    ippiDFT_C_Func ippiDFT_CToC_32fc_C1R;
};

struct IPPDFT_R_Functor
{
    IPPDFT_R_Functor(ippiDFT_R_Func _func) : ippiDFT_PackToR_32f_C1R(_func){}

    bool operator()(const Ipp32f* src, size_t srcStep, Ipp32f* dst, size_t dstStep, const IppiDFTSpec_R_32f* pDFTSpec, Ipp8u* pBuffer) const
    {
        return ippiDFT_PackToR_32f_C1R ? CV_INSTRUMENT_FUN_IPP(ippiDFT_PackToR_32f_C1R, src, static_cast<int>(srcStep), dst, static_cast<int>(dstStep), pDFTSpec, pBuffer) >= 0 : false;
    }
private:
    ippiDFT_R_Func ippiDFT_PackToR_32f_C1R;
};

static bool ippi_DFT_C_32F(const uchar * src, size_t src_step, uchar * dst, size_t dst_step, int width, int height, bool inv, int norm_flag)
{
    CV_INSTRUMENT_REGION_IPP();

    IppStatus status;
    Ipp8u* pBuffer = 0;
    Ipp8u* pMemInit= 0;
    int sizeBuffer=0;
    int sizeSpec=0;
    int sizeInit=0;

    IppiSize srcRoiSize = {width, height};

    status = ippiDFTGetSize_C_32fc(srcRoiSize, norm_flag, ippAlgHintNone, &sizeSpec, &sizeInit, &sizeBuffer );
    if ( status < 0 )
        return false;

    IppiDFTSpec_C_32fc* pDFTSpec = (IppiDFTSpec_C_32fc*)CV_IPP_MALLOC( sizeSpec );

    if ( sizeInit > 0 )
        pMemInit = (Ipp8u*)CV_IPP_MALLOC( sizeInit );

    if ( sizeBuffer > 0 )
        pBuffer = (Ipp8u*)CV_IPP_MALLOC( sizeBuffer );

    status = ippiDFTInit_C_32fc( srcRoiSize, norm_flag, ippAlgHintNone, pDFTSpec, pMemInit );

    if ( sizeInit > 0 )
        ippFree( pMemInit );

    if ( status < 0 )
    {
        ippFree( pDFTSpec );
        if ( sizeBuffer > 0 )
            ippFree( pBuffer );
        return false;
    }

    if (!inv)
        status = CV_INSTRUMENT_FUN_IPP(ippiDFTFwd_CToC_32fc_C1R, (Ipp32fc*)src, static_cast<int>(src_step), (Ipp32fc*)dst, static_cast<int>(dst_step), pDFTSpec, pBuffer);
    else
        status = CV_INSTRUMENT_FUN_IPP(ippiDFTInv_CToC_32fc_C1R, (Ipp32fc*)src, static_cast<int>(src_step), (Ipp32fc*)dst, static_cast<int>(dst_step), pDFTSpec, pBuffer);

    if ( sizeBuffer > 0 )
        ippFree( pBuffer );

    ippFree( pDFTSpec );

    if(status >= 0)
    {
        CV_IMPL_ADD(CV_IMPL_IPP);
        return true;
    }
    return false;
}

static bool ippi_DFT_R_32F(const uchar * src, size_t src_step, uchar * dst, size_t dst_step, int width, int height, bool inv, int norm_flag)
{
    CV_INSTRUMENT_REGION_IPP();

    IppStatus status;
    Ipp8u* pBuffer = 0;
    Ipp8u* pMemInit= 0;
    int sizeBuffer=0;
    int sizeSpec=0;
    int sizeInit=0;

    IppiSize srcRoiSize = {width, height};

    status = ippiDFTGetSize_R_32f(srcRoiSize, norm_flag, ippAlgHintNone, &sizeSpec, &sizeInit, &sizeBuffer );
    if ( status < 0 )
        return false;

    IppiDFTSpec_R_32f* pDFTSpec = (IppiDFTSpec_R_32f*)CV_IPP_MALLOC( sizeSpec );

    if ( sizeInit > 0 )
        pMemInit = (Ipp8u*)CV_IPP_MALLOC( sizeInit );

    if ( sizeBuffer > 0 )
        pBuffer = (Ipp8u*)CV_IPP_MALLOC( sizeBuffer );

    status = ippiDFTInit_R_32f( srcRoiSize, norm_flag, ippAlgHintNone, pDFTSpec, pMemInit );

    if ( sizeInit > 0 )
        ippFree( pMemInit );

    if ( status < 0 )
    {
        ippFree( pDFTSpec );
        if ( sizeBuffer > 0 )
            ippFree( pBuffer );
        return false;
    }

    if (!inv)
        status = CV_INSTRUMENT_FUN_IPP(ippiDFTFwd_RToPack_32f_C1R, (float*)src, static_cast<int>(src_step), (float*)dst, static_cast<int>(dst_step), pDFTSpec, pBuffer);
    else
        status = CV_INSTRUMENT_FUN_IPP(ippiDFTInv_PackToR_32f_C1R, (float*)src, static_cast<int>(src_step), (float*)dst, static_cast<int>(dst_step), pDFTSpec, pBuffer);

    if ( sizeBuffer > 0 )
        ippFree( pBuffer );

    ippFree( pDFTSpec );

    if(status >= 0)
    {
        CV_IMPL_ADD(CV_IMPL_IPP);
        return true;
    }
    return false;
}

#endif
}

#ifdef HAVE_OPENCL

namespace cv
{

enum FftType
{
    R2R = 0, // real to CCS in case forward transform, CCS to real otherwise
    C2R = 1, // complex to real in case inverse transform
    R2C = 2, // real to complex in case forward transform
    C2C = 3  // complex to complex
};

struct OCL_FftPlan
{
private:
    UMat twiddles;
    String buildOptions;
    int thread_count;
    int dft_size;
    int dft_depth;
    bool status;

public:
    OCL_FftPlan(int _size, int _depth) : dft_size(_size), dft_depth(_depth), status(true)
    {
        CV_Assert( dft_depth == CV_32F || dft_depth == CV_64F );

        int min_radix;
        std::vector<int> radixes, blocks;
        ocl_getRadixes(dft_size, radixes, blocks, min_radix);
        thread_count = dft_size / min_radix;

        if (thread_count > (int) ocl::Device::getDefault().maxWorkGroupSize())
        {
            status = false;
            return;
        }

        // generate string with radix calls
        String radix_processing;
        int n = 1, twiddle_size = 0;
        for (size_t i=0; i<radixes.size(); i++)
        {
            int radix = radixes[i], block = blocks[i];
            if (block > 1)
                radix_processing += format("fft_radix%d_B%d(smem,twiddles+%d,ind,%d,%d);", radix, block, twiddle_size, n, dft_size/radix);
            else
                radix_processing += format("fft_radix%d(smem,twiddles+%d,ind,%d,%d);", radix, twiddle_size, n, dft_size/radix);
            twiddle_size += (radix-1)*n;
            n *= radix;
        }

        twiddles.create(1, twiddle_size, CV_MAKE_TYPE(dft_depth, 2));
        if (dft_depth == CV_32F)
            fillRadixTable<float>(twiddles, radixes);
        else
            fillRadixTable<double>(twiddles, radixes);

        buildOptions = format("-D LOCAL_SIZE=%d -D kercn=%d -D FT=%s -D CT=%s%s -D RADIX_PROCESS=%s",
                              dft_size, min_radix, ocl::typeToStr(dft_depth), ocl::typeToStr(CV_MAKE_TYPE(dft_depth, 2)),
                              dft_depth == CV_64F ? " -D DOUBLE_SUPPORT" : "", radix_processing.c_str());
    }

    bool enqueueTransform(InputArray _src, OutputArray _dst, int num_dfts, int flags, int fftType, bool rows = true) const
    {
        if (!status)
            return false;

        UMat src = _src.getUMat();
        UMat dst = _dst.getUMat();

        size_t globalsize[2];
        size_t localsize[2];
        String kernel_name;

        bool is1d = (flags & DFT_ROWS) != 0 || num_dfts == 1;
        bool inv = (flags & DFT_INVERSE) != 0;
        String options = buildOptions;

        if (rows)
        {
            globalsize[0] = thread_count; globalsize[1] = src.rows;
            localsize[0] = thread_count; localsize[1] = 1;
            kernel_name = !inv ? "fft_multi_radix_rows" : "ifft_multi_radix_rows";
            if ((is1d || inv) && (flags & DFT_SCALE))
                options += " -D DFT_SCALE";
        }
        else
        {
            globalsize[0] = num_dfts; globalsize[1] = thread_count;
            localsize[0] = 1; localsize[1] = thread_count;
            kernel_name = !inv ? "fft_multi_radix_cols" : "ifft_multi_radix_cols";
            if (flags & DFT_SCALE)
                options += " -D DFT_SCALE";
        }

        options += src.channels() == 1 ? " -D REAL_INPUT" : " -D COMPLEX_INPUT";
        options += dst.channels() == 1 ? " -D REAL_OUTPUT" : " -D COMPLEX_OUTPUT";
        options += is1d ? " -D IS_1D" : "";

        if (!inv)
        {
            if ((is1d && src.channels() == 1) || (rows && (fftType == R2R)))
                options += " -D NO_CONJUGATE";
        }
        else
        {
            if (rows && (fftType == C2R || fftType == R2R))
                options += " -D NO_CONJUGATE";
            if (dst.cols % 2 == 0)
                options += " -D EVEN";
        }

        ocl::Kernel k(kernel_name.c_str(), ocl::core::fft_oclsrc, options);
        if (k.empty())
            return false;

        k.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnly(dst), ocl::KernelArg::ReadOnlyNoSize(twiddles), thread_count, num_dfts);
        return k.run(2, globalsize, localsize, false);
    }

private:
    static void ocl_getRadixes(int cols, std::vector<int>& radixes, std::vector<int>& blocks, int& min_radix)
    {
        int factors[34];
        int nf = DFTFactorize(cols, factors);

        int n = 1;
        int factor_index = 0;
        min_radix = INT_MAX;

        // 2^n transforms
        if ((factors[factor_index] & 1) == 0)
        {
            for( ; n < factors[factor_index];)
            {
                int radix = 2, block = 1;
                if (8*n <= factors[0])
                    radix = 8;
                else if (4*n <= factors[0])
                {
                    radix = 4;
                    if (cols % 12 == 0)
                        block = 3;
                    else if (cols % 8 == 0)
                        block = 2;
                }
                else
                {
                    if (cols % 10 == 0)
                        block = 5;
                    else if (cols % 8 == 0)
                        block = 4;
                    else if (cols % 6 == 0)
                        block = 3;
                    else if (cols % 4 == 0)
                        block = 2;
                }

                radixes.push_back(radix);
                blocks.push_back(block);
                min_radix = min(min_radix, block*radix);
                n *= radix;
            }
            factor_index++;
        }

        // all the other transforms
        for( ; factor_index < nf; factor_index++)
        {
            int radix = factors[factor_index], block = 1;
            if (radix == 3)
            {
                if (cols % 12 == 0)
                    block = 4;
                else if (cols % 9 == 0)
                    block = 3;
                else if (cols % 6 == 0)
                    block = 2;
            }
            else if (radix == 5)
            {
                if (cols % 10 == 0)
                    block = 2;
            }
            radixes.push_back(radix);
            blocks.push_back(block);
            min_radix = min(min_radix, block*radix);
        }
    }

    template <typename T>
    static void fillRadixTable(UMat twiddles, const std::vector<int>& radixes)
    {
        Mat tw = twiddles.getMat(ACCESS_WRITE);
        T* ptr = tw.ptr<T>();
        int ptr_index = 0;

        int n = 1;
        for (size_t i=0; i<radixes.size(); i++)
        {
            int radix = radixes[i];
            n *= radix;

            for (int j=1; j<radix; j++)
            {
                double theta = -CV_2PI*j/n;

                for (int k=0; k<(n/radix); k++)
                {
                    ptr[ptr_index++] = (T) cos(k*theta);
                    ptr[ptr_index++] = (T) sin(k*theta);
                }
            }
        }
    }
};

class OCL_FftPlanCache
{
public:
    static OCL_FftPlanCache & getInstance()
    {
        CV_SINGLETON_LAZY_INIT_REF(OCL_FftPlanCache, new OCL_FftPlanCache())
    }

    Ptr<OCL_FftPlan> getFftPlan(int dft_size, int depth)
    {
        int key = (dft_size << 16) | (depth & 0xFFFF);
        std::map<int, Ptr<OCL_FftPlan> >::iterator f = planStorage.find(key);
        if (f != planStorage.end())
        {
            return f->second;
        }
        else
        {
            Ptr<OCL_FftPlan> newPlan = Ptr<OCL_FftPlan>(new OCL_FftPlan(dft_size, depth));
            planStorage[key] = newPlan;
            return newPlan;
        }
    }

    ~OCL_FftPlanCache()
    {
        planStorage.clear();
    }

protected:
    OCL_FftPlanCache() :
        planStorage()
    {
    }
    std::map<int, Ptr<OCL_FftPlan> > planStorage;
};

static bool ocl_dft_rows(InputArray _src, OutputArray _dst, int nonzero_rows, int flags, int fftType)
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type);
    Ptr<OCL_FftPlan> plan = OCL_FftPlanCache::getInstance().getFftPlan(_src.cols(), depth);
    return plan->enqueueTransform(_src, _dst, nonzero_rows, flags, fftType, true);
}

static bool ocl_dft_cols(InputArray _src, OutputArray _dst, int nonzero_cols, int flags, int fftType)
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type);
    Ptr<OCL_FftPlan> plan = OCL_FftPlanCache::getInstance().getFftPlan(_src.rows(), depth);
    return plan->enqueueTransform(_src, _dst, nonzero_cols, flags, fftType, false);
}

inline FftType determineFFTType(bool real_input, bool complex_input, bool real_output, bool complex_output, bool inv)
{
    // output format is not specified
    if (!real_output && !complex_output)
        complex_output = true;

    // input or output format is ambiguous
    if (real_input == complex_input || real_output == complex_output)
        CV_Error(Error::StsBadArg, "Invalid FFT input or output format");

    FftType result = real_input ? (real_output ? R2R : R2C) : (real_output ? C2R : C2C);

    // Forward Complex to CCS not supported
    if (result == C2R && !inv)
        result = C2C;

    // Inverse CCS to Complex not supported
    if (result == R2C && inv)
        result = R2R;

    return result;
}

static bool ocl_dft(InputArray _src, OutputArray _dst, int flags, int nonzero_rows)
{
    int type = _src.type(), cn = CV_MAT_CN(type), depth = CV_MAT_DEPTH(type);
    Size ssize = _src.size();
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;

    if (!(cn == 1 || cn == 2)
        || !(depth == CV_32F || (depth == CV_64F && doubleSupport))
        || ((flags & DFT_REAL_OUTPUT) && (flags & DFT_COMPLEX_OUTPUT)))
        return false;

    // if is not a multiplication of prime numbers { 2, 3, 5 }
    if (ssize.area() != getOptimalDFTSize(ssize.area()))
        return false;

    UMat src = _src.getUMat();
    bool inv = (flags & DFT_INVERSE) != 0 ? 1 : 0;

    if( nonzero_rows <= 0 || nonzero_rows > _src.rows() )
        nonzero_rows = _src.rows();
    bool is1d = (flags & DFT_ROWS) != 0 || nonzero_rows == 1;

    FftType fftType = determineFFTType(cn == 1, cn == 2,
        (flags & DFT_REAL_OUTPUT) != 0, (flags & DFT_COMPLEX_OUTPUT) != 0, inv);

    UMat output;
    if (fftType == C2C || fftType == R2C)
    {
        // complex output
        _dst.create(src.size(), CV_MAKETYPE(depth, 2));
        output = _dst.getUMat();
    }
    else
    {
        // real output
        if (is1d)
        {
            _dst.create(src.size(), CV_MAKETYPE(depth, 1));
            output = _dst.getUMat();
        }
        else
        {
            _dst.create(src.size(), CV_MAKETYPE(depth, 1));
            output.create(src.size(), CV_MAKETYPE(depth, 2));
        }
    }

    bool result = false;
    if (!inv)
    {
        int nonzero_cols = fftType == R2R ? output.cols/2 + 1 : output.cols;
        result = ocl_dft_rows(src, output, nonzero_rows, flags, fftType);
        if (!is1d)
            result = result && ocl_dft_cols(output, _dst, nonzero_cols, flags, fftType);
    }
    else
    {
        if (fftType == C2C)
        {
            // complex output
            result = ocl_dft_rows(src, output, nonzero_rows, flags, fftType);
            if (!is1d)
                result = result && ocl_dft_cols(output, output, output.cols, flags, fftType);
        }
        else
        {
            if (is1d)
            {
                result = ocl_dft_rows(src, output, nonzero_rows, flags, fftType);
            }
            else
            {
                int nonzero_cols = src.cols/2 + 1;
                result = ocl_dft_cols(src, output, nonzero_cols, flags, fftType);
                result = result && ocl_dft_rows(output, _dst, nonzero_rows, flags, fftType);
            }
        }
    }
    return result;
}

} // namespace cv;

#endif

#ifdef HAVE_CLAMDFFT

namespace cv {

#define CLAMDDFT_Assert(func) \
    { \
        clfftStatus s = (func); \
        CV_Assert(s == CLFFT_SUCCESS); \
    }

class PlanCache
{
    struct FftPlan
    {
        FftPlan(const Size & _dft_size, int _src_step, int _dst_step, bool _doubleFP, bool _inplace, int _flags, FftType _fftType) :
            dft_size(_dft_size), src_step(_src_step), dst_step(_dst_step),
            doubleFP(_doubleFP), inplace(_inplace), flags(_flags), fftType(_fftType),
            context((cl_context)ocl::Context::getDefault().ptr()), plHandle(0)
        {
            bool dft_inverse = (flags & DFT_INVERSE) != 0;
            bool dft_scale = (flags & DFT_SCALE) != 0;
            bool dft_rows = (flags & DFT_ROWS) != 0;

            clfftLayout inLayout = CLFFT_REAL, outLayout = CLFFT_REAL;
            clfftDim dim = dft_size.height == 1 || dft_rows ? CLFFT_1D : CLFFT_2D;

            size_t batchSize = dft_rows ? dft_size.height : 1;
            size_t clLengthsIn[3] = { (size_t)dft_size.width, dft_rows ? 1 : (size_t)dft_size.height, 1 };
            size_t clStridesIn[3] = { 1, 1, 1 };
            size_t clStridesOut[3]  = { 1, 1, 1 };
            int elemSize = doubleFP ? sizeof(double) : sizeof(float);

            switch (fftType)
            {
            case C2C:
                inLayout = CLFFT_COMPLEX_INTERLEAVED;
                outLayout = CLFFT_COMPLEX_INTERLEAVED;
                clStridesIn[1] = src_step / (elemSize << 1);
                clStridesOut[1] = dst_step / (elemSize << 1);
                break;
            case R2C:
                inLayout = CLFFT_REAL;
                outLayout = CLFFT_HERMITIAN_INTERLEAVED;
                clStridesIn[1] = src_step / elemSize;
                clStridesOut[1] = dst_step / (elemSize << 1);
                break;
            case C2R:
                inLayout = CLFFT_HERMITIAN_INTERLEAVED;
                outLayout = CLFFT_REAL;
                clStridesIn[1] = src_step / (elemSize << 1);
                clStridesOut[1] = dst_step / elemSize;
                break;
            case R2R:
            default:
                CV_Error(Error::StsNotImplemented, "AMD Fft does not support this type");
                break;
            }

            clStridesIn[2] = dft_rows ? clStridesIn[1] : dft_size.width * clStridesIn[1];
            clStridesOut[2] = dft_rows ? clStridesOut[1] : dft_size.width * clStridesOut[1];

            CLAMDDFT_Assert(clfftCreateDefaultPlan(&plHandle, (cl_context)ocl::Context::getDefault().ptr(), dim, clLengthsIn))

            // setting plan properties
            CLAMDDFT_Assert(clfftSetPlanPrecision(plHandle, doubleFP ? CLFFT_DOUBLE : CLFFT_SINGLE));
            CLAMDDFT_Assert(clfftSetResultLocation(plHandle, inplace ? CLFFT_INPLACE : CLFFT_OUTOFPLACE))
            CLAMDDFT_Assert(clfftSetLayout(plHandle, inLayout, outLayout))
            CLAMDDFT_Assert(clfftSetPlanBatchSize(plHandle, batchSize))
            CLAMDDFT_Assert(clfftSetPlanInStride(plHandle, dim, clStridesIn))
            CLAMDDFT_Assert(clfftSetPlanOutStride(plHandle, dim, clStridesOut))
            CLAMDDFT_Assert(clfftSetPlanDistance(plHandle, clStridesIn[dim], clStridesOut[dim]))

            float scale = dft_scale ? 1.0f / (dft_rows ? dft_size.width : dft_size.area()) : 1.0f;
            CLAMDDFT_Assert(clfftSetPlanScale(plHandle, dft_inverse ? CLFFT_BACKWARD : CLFFT_FORWARD, scale))

            // ready to bake
            cl_command_queue queue = (cl_command_queue)ocl::Queue::getDefault().ptr();
            CLAMDDFT_Assert(clfftBakePlan(plHandle, 1, &queue, NULL, NULL))
        }

        ~FftPlan()
        {
            // Do not tear down clFFT.
            // The user application may still use clFFT even after OpenCV is unloaded.
            /*clfftDestroyPlan(&plHandle);*/
        }

        friend class PlanCache;

    private:
        Size dft_size;
        int src_step, dst_step;
        bool doubleFP;
        bool inplace;
        int flags;
        FftType fftType;

        cl_context context;
        clfftPlanHandle plHandle;
    };

public:
    static PlanCache & getInstance()
    {
        CV_SINGLETON_LAZY_INIT_REF(PlanCache, new PlanCache())
    }

    clfftPlanHandle getPlanHandle(const Size & dft_size, int src_step, int dst_step, bool doubleFP,
                                  bool inplace, int flags, FftType fftType)
    {
        cl_context currentContext = (cl_context)ocl::Context::getDefault().ptr();

        for (size_t i = 0, size = planStorage.size(); i < size; ++i)
        {
            const FftPlan * const plan = planStorage[i];

            if (plan->dft_size == dft_size &&
                plan->flags == flags &&
                plan->src_step == src_step &&
                plan->dst_step == dst_step &&
                plan->doubleFP == doubleFP &&
                plan->fftType == fftType &&
                plan->inplace == inplace)
            {
                if (plan->context != currentContext)
                {
                    planStorage.erase(planStorage.begin() + i);
                    break;
                }

                return plan->plHandle;
            }
        }

        // no baked plan is found, so let's create a new one
        Ptr<FftPlan> newPlan = Ptr<FftPlan>(new FftPlan(dft_size, src_step, dst_step, doubleFP, inplace, flags, fftType));
        planStorage.push_back(newPlan);

        return newPlan->plHandle;
    }

    ~PlanCache()
    {
        planStorage.clear();
    }

protected:
    PlanCache() :
        planStorage()
    {
    }

    std::vector<Ptr<FftPlan> > planStorage;
};

extern "C" {

static void CL_CALLBACK oclCleanupCallback(cl_event e, cl_int, void *p)
{
    UMatData * u = (UMatData *)p;

    if( u && CV_XADD(&u->urefcount, -1) == 1 )
        u->currAllocator->deallocate(u);
    u = 0;

    clReleaseEvent(e), e = 0;
}

}

static bool ocl_dft_amdfft(InputArray _src, OutputArray _dst, int flags)
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    Size ssize = _src.size();

    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;
    if ( (!doubleSupport && depth == CV_64F) ||
         !(type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2) ||
         _src.offset() != 0)
        return false;

    // if is not a multiplication of prime numbers { 2, 3, 5 }
    if (ssize.area() != getOptimalDFTSize(ssize.area()))
        return false;

    int dst_complex_input = cn == 2 ? 1 : 0;
    bool dft_inverse = (flags & DFT_INVERSE) != 0 ? 1 : 0;
    int dft_complex_output = (flags & DFT_COMPLEX_OUTPUT) != 0;
    bool dft_real_output = (flags & DFT_REAL_OUTPUT) != 0;

    CV_Assert(dft_complex_output + dft_real_output < 2);
    FftType fftType = (FftType)(dst_complex_input << 0 | dft_complex_output << 1);

    switch (fftType)
    {
    case C2C:
        _dst.create(ssize.height, ssize.width, CV_MAKE_TYPE(depth, 2));
        break;
    case R2C: // TODO implement it if possible
    case C2R: // TODO implement it if possible
    case R2R: // AMD Fft does not support this type
    default:
        return false;
    }

    UMat src = _src.getUMat(), dst = _dst.getUMat();
    bool inplace = src.u == dst.u;

    clfftPlanHandle plHandle = PlanCache::getInstance().
            getPlanHandle(ssize, (int)src.step, (int)dst.step,
                          depth == CV_64F, inplace, flags, fftType);

    // get the bufferSize
    size_t bufferSize = 0;
    CLAMDDFT_Assert(clfftGetTmpBufSize(plHandle, &bufferSize))
    UMat tmpBuffer(1, (int)bufferSize, CV_8UC1);

    cl_mem srcarg = (cl_mem)src.handle(ACCESS_READ);
    cl_mem dstarg = (cl_mem)dst.handle(ACCESS_RW);

    cl_command_queue queue = (cl_command_queue)ocl::Queue::getDefault().ptr();
    cl_event e = 0;

    CLAMDDFT_Assert(clfftEnqueueTransform(plHandle, dft_inverse ? CLFFT_BACKWARD : CLFFT_FORWARD,
                                          1, &queue, 0, NULL, &e,
                                          &srcarg, &dstarg, (cl_mem)tmpBuffer.handle(ACCESS_RW)))

    tmpBuffer.addref();
    clSetEventCallback(e, CL_COMPLETE, oclCleanupCallback, tmpBuffer.u);
    return true;
}

#undef DFT_ASSERT

}

#endif // HAVE_CLAMDFFT

namespace cv
{

template <typename T>
static void complementComplex(T * ptr, size_t step, int n, int len, int dft_dims)
{
    T* p0 = (T*)ptr;
    size_t dstep = step/sizeof(p0[0]);
    for(int i = 0; i < len; i++ )
    {
        T* p = p0 + dstep*i;
        T* q = dft_dims == 1 || i == 0 || i*2 == len ? p : p0 + dstep*(len-i);

        for( int j = 1; j < (n+1)/2; j++ )
        {
            p[(n-j)*2] = q[j*2];
            p[(n-j)*2+1] = -q[j*2+1];
        }
    }
}

static void complementComplexOutput(int depth, uchar * ptr, size_t step, int count, int len, int dft_dims)
{
    if( depth == CV_32F )
        complementComplex((float*)ptr, step, count, len, dft_dims);
    else
        complementComplex((double*)ptr, step, count, len, dft_dims);
}

enum DftMode {
    InvalidDft = 0,
    FwdRealToCCS,
    FwdRealToComplex,
    FwdComplexToComplex,
    InvCCSToReal,
    InvComplexToReal,
    InvComplexToComplex,
};

enum DftDims {
    InvalidDim = 0,
    OneDim,
    OneDimColWise,
    TwoDims
};

inline const char * modeName(DftMode m)
{
    switch (m)
    {
    case InvalidDft: return "InvalidDft";
    case FwdRealToCCS: return "FwdRealToCCS";
    case FwdRealToComplex: return "FwdRealToComplex";
    case FwdComplexToComplex: return "FwdComplexToComplex";
    case InvCCSToReal: return "InvCCSToReal";
    case InvComplexToReal: return "InvComplexToReal";
    case InvComplexToComplex: return "InvComplexToComplex";
    }
    return 0;
}

inline const char * dimsName(DftDims d)
{
    switch (d)
    {
    case InvalidDim: return "InvalidDim";
    case OneDim: return "OneDim";
    case OneDimColWise: return "OneDimColWise";
    case TwoDims: return "TwoDims";
    };
    return 0;
}

template <typename T>
inline bool isInv(T mode)
{
    switch ((DftMode)mode)
    {
        case InvCCSToReal:
        case InvComplexToReal:
        case InvComplexToComplex: return true;
        default: return false;
    }
}

inline DftMode determineMode(bool inv, int cn1, int cn2)
{
    if (!inv)
    {
        if (cn1 == 1 && cn2 == 1)
            return FwdRealToCCS;
        else if (cn1 == 1 && cn2 == 2)
            return FwdRealToComplex;
        else if (cn1 == 2 && cn2 == 2)
            return FwdComplexToComplex;
    }
    else
    {
        if (cn1 == 1 && cn2 == 1)
            return InvCCSToReal;
        else if (cn1 == 2 && cn2 == 1)
            return InvComplexToReal;
        else if (cn1 == 2 && cn2 == 2)
            return InvComplexToComplex;
    }
    return InvalidDft;
}


inline DftDims determineDims(int rows, int cols, bool isRowWise, bool isContinuous)
{
    // printf("%d x %d (%d, %d)\n", rows, cols, isRowWise, isContinuous);
    if (isRowWise)
        return OneDim;
    if (cols == 1 && rows > 1) // one-column-shaped input
    {
        if (isContinuous)
            return OneDim;
        else
            return OneDimColWise;
    }
    if (rows == 1)
        return OneDim;
    if (cols > 1 && rows > 1)
        return TwoDims;
    return InvalidDim;
}

class OcvDftImpl CV_FINAL : public hal::DFT2D
{
protected:
    Ptr<hal::DFT1D> contextA;
    Ptr<hal::DFT1D> contextB;
    bool needBufferA;
    bool needBufferB;
    bool inv;
    int width;
    int height;
    DftMode mode;
    int elem_size;
    int complex_elem_size;
    int depth;
    bool real_transform;
    int nonzero_rows;
    bool isRowTransform;
    bool isScaled;
    std::vector<int> stages;
    bool useIpp;
    int src_channels;
    int dst_channels;

    AutoBuffer<uchar> tmp_bufA;
    AutoBuffer<uchar> tmp_bufB;
    AutoBuffer<uchar> buf0;
    AutoBuffer<uchar> buf1;

public:
    OcvDftImpl()
    {
        needBufferA = false;
        needBufferB = false;
        inv = false;
        width = 0;
        height = 0;
        mode = InvalidDft;
        elem_size = 0;
        complex_elem_size = 0;
        depth = 0;
        real_transform = false;
        nonzero_rows = 0;
        isRowTransform = false;
        isScaled = false;
        useIpp = false;
        src_channels = 0;
        dst_channels = 0;
    }

    void init(int _width, int _height, int _depth, int _src_channels, int _dst_channels, int flags, int _nonzero_rows)
    {
        bool isComplex = _src_channels != _dst_channels;
        nonzero_rows = _nonzero_rows;
        width = _width;
        height = _height;
        depth = _depth;
        src_channels = _src_channels;
        dst_channels = _dst_channels;
        bool isInverse = (flags & CV_HAL_DFT_INVERSE) != 0;
        bool isInplace = (flags & CV_HAL_DFT_IS_INPLACE) != 0;
        bool isContinuous = (flags & CV_HAL_DFT_IS_CONTINUOUS) != 0;
        mode = determineMode(isInverse, _src_channels, _dst_channels);
        inv = isInverse;
        isRowTransform = (flags & CV_HAL_DFT_ROWS) != 0;
        isScaled = (flags & CV_HAL_DFT_SCALE) != 0;
        needBufferA = false;
        needBufferB = false;
        real_transform = (mode != FwdComplexToComplex && mode != InvComplexToComplex);

        elem_size = (depth == CV_32F) ? sizeof(float) : sizeof(double);
        complex_elem_size = elem_size * 2;
        if( !real_transform )
            elem_size = complex_elem_size;

#if defined USE_IPP_DFT
        CV_IPP_CHECK()
        {
            if (nonzero_rows == 0 && depth == CV_32F && ((width * height)>(int)(1<<6)))
            {
                if (mode == FwdComplexToComplex || mode == InvComplexToComplex || mode == FwdRealToCCS || mode == InvCCSToReal)
                {
                    useIpp = true;
                    return;
                }
            }
        }
#endif

        DftDims dims = determineDims(height, width, isRowTransform, isContinuous);
        if (dims == TwoDims)
        {
            stages.resize(2);
            if (mode == InvCCSToReal || mode == InvComplexToReal)
            {
                stages[0] = 1;
                stages[1] = 0;
            }
            else
            {
                stages[0] = 0;
                stages[1] = 1;
            }
        }
        else
        {
            stages.resize(1);
            if (dims == OneDimColWise)
                stages[0] = 1;
            else
                stages[0] = 0;
        }

        for(uint stageIndex = 0; stageIndex < stages.size(); ++stageIndex)
        {
            if (stageIndex == 1)
            {
                isInplace = true;
                isComplex = false;
            }

            int stage = stages[stageIndex];
            bool isLastStage = (stageIndex + 1 == stages.size());

            int len, count;

            int f = 0;
            if (inv)
                f |= CV_HAL_DFT_INVERSE;
            if (isScaled)
                f |= CV_HAL_DFT_SCALE;
            if (isRowTransform)
                f |= CV_HAL_DFT_ROWS;
            if (isComplex)
                f |= CV_HAL_DFT_COMPLEX_OUTPUT;
            if (real_transform)
                f |= CV_HAL_DFT_REAL_OUTPUT;
            if (!isLastStage)
                f |= CV_HAL_DFT_TWO_STAGE;

            if( stage == 0 ) // row-wise transform
            {
                if (width == 1 && !isRowTransform )
                {
                    len = height;
                    count = width;
                }
                else
                {
                    len = width;
                    count = height;
                }
                needBufferA = isInplace;
                contextA = hal::DFT1D::create(len, count, depth, f, &needBufferA);
                if (needBufferA)
                    tmp_bufA.allocate(len * complex_elem_size);
            }
            else
            {
                len = height;
                count = width;
                f |= CV_HAL_DFT_STAGE_COLS;
                needBufferB = isInplace;
                contextB = hal::DFT1D::create(len, count, depth, f, &needBufferB);
                if (needBufferB)
                    tmp_bufB.allocate(len * complex_elem_size);

                buf0.allocate(len * complex_elem_size);
                buf1.allocate(len * complex_elem_size);
            }
        }
    }

    void apply(const uchar * src, size_t src_step, uchar * dst, size_t dst_step) CV_OVERRIDE
    {
#if defined USE_IPP_DFT
        if (useIpp)
        {
            int ipp_norm_flag = !isScaled ? 8 : inv ? 2 : 1;
            if (!isRowTransform)
            {
                if (mode == FwdComplexToComplex || mode == InvComplexToComplex)
                {
                    if (ippi_DFT_C_32F(src, src_step, dst, dst_step, width, height, inv, ipp_norm_flag))
                    {
                        CV_IMPL_ADD(CV_IMPL_IPP);
                        return;
                    }
                    setIppErrorStatus();
                }
                else if (mode == FwdRealToCCS || mode == InvCCSToReal)
                {
                    if (ippi_DFT_R_32F(src, src_step, dst, dst_step, width, height, inv, ipp_norm_flag))
                    {
                        CV_IMPL_ADD(CV_IMPL_IPP);
                        return;
                    }
                    setIppErrorStatus();
                }
            }
            else
            {
                if (mode == FwdComplexToComplex || mode == InvComplexToComplex)
                {
                    ippiDFT_C_Func ippiFunc = inv ? (ippiDFT_C_Func)ippiDFTInv_CToC_32fc_C1R : (ippiDFT_C_Func)ippiDFTFwd_CToC_32fc_C1R;
                    if (Dft_C_IPPLoop(src, src_step, dst, dst_step, width, height, IPPDFT_C_Functor(ippiFunc),ipp_norm_flag))
                    {
                        CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
                        return;
                    }
                    setIppErrorStatus();
                }
                else if (mode == FwdRealToCCS || mode == InvCCSToReal)
                {
                    ippiDFT_R_Func ippiFunc = inv ? (ippiDFT_R_Func)ippiDFTInv_PackToR_32f_C1R : (ippiDFT_R_Func)ippiDFTFwd_RToPack_32f_C1R;
                    if (Dft_R_IPPLoop(src, src_step, dst, dst_step, width, height, IPPDFT_R_Functor(ippiFunc),ipp_norm_flag))
                    {
                        CV_IMPL_ADD(CV_IMPL_IPP|CV_IMPL_MT);
                        return;
                    }
                    setIppErrorStatus();
                }
            }
            return;
        }
#endif

        for(uint stageIndex = 0; stageIndex < stages.size(); ++stageIndex)
        {
            int stage_src_channels = src_channels;
            int stage_dst_channels = dst_channels;

            if (stageIndex == 1)
            {
                src = dst;
                src_step = dst_step;
                stage_src_channels = stage_dst_channels;
            }

            int stage = stages[stageIndex];
            bool isLastStage = (stageIndex + 1 == stages.size());
            bool isComplex = stage_src_channels != stage_dst_channels;

            if( stage == 0 )
                rowDft(src, src_step, dst, dst_step, isComplex, isLastStage);
            else
                colDft(src, src_step, dst, dst_step, stage_src_channels, stage_dst_channels, isLastStage);
        }
    }

protected:

    void rowDft(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, bool isComplex, bool isLastStage)
    {
        int len, count;
        if (width == 1 && !isRowTransform )
        {
            len = height;
            count = width;
        }
        else
        {
            len = width;
            count = height;
        }
        int dptr_offset = 0;
        int dst_full_len = len*elem_size;

        if( needBufferA )
        {
            if (mode == FwdRealToCCS && (len & 1) && len > 1)
                dptr_offset = elem_size;
        }

        if( !inv && isComplex )
            dst_full_len += (len & 1) ? elem_size : complex_elem_size;

        int nz = nonzero_rows;
        if( nz <= 0 || nz > count )
            nz = count;

        int i;
        for( i = 0; i < nz; i++ )
        {
            const uchar* sptr = src_data + src_step * i;
            uchar* dptr0 = dst_data + dst_step * i;
            uchar* dptr = dptr0;

            if( needBufferA )
                dptr = tmp_bufA.data();

            contextA->apply(sptr, dptr);

            if( needBufferA )
                memcpy( dptr0, dptr + dptr_offset, dst_full_len );
        }

        for( ; i < count; i++ )
        {
            uchar* dptr0 = dst_data + dst_step * i;
            memset( dptr0, 0, dst_full_len );
        }
        if(isLastStage &&  mode == FwdRealToComplex)
            complementComplexOutput(depth, dst_data, dst_step, len, nz, 1);
    }

    void colDft(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int stage_src_channels, int stage_dst_channels, bool isLastStage)
    {
        int len = height;
        int count = width;
        int a = 0, b = count;
        uchar *dbuf0, *dbuf1;
        const uchar* sptr0 = src_data;
        uchar* dptr0 = dst_data;

        dbuf0 = buf0.data(), dbuf1 = buf1.data();

        if( needBufferB )
        {
            dbuf1 = tmp_bufB.data();
            dbuf0 = buf1.data();
        }

        if( real_transform )
        {
            int even;
            a = 1;
            even = (count & 1) == 0;
            b = (count+1)/2;
            if( !inv )
            {
                memset( buf0.data(), 0, len*complex_elem_size );
                CopyColumn( sptr0, src_step, buf0.data(), complex_elem_size, len, elem_size );
                sptr0 += stage_dst_channels*elem_size;
                if( even )
                {
                    memset( buf1.data(), 0, len*complex_elem_size );
                    CopyColumn( sptr0 + (count-2)*elem_size, src_step,
                                buf1.data(), complex_elem_size, len, elem_size );
                }
            }
            else if( stage_src_channels == 1 )
            {
                CopyColumn( sptr0, src_step, buf0.data(), elem_size, len, elem_size );
                ExpandCCS( buf0.data(), len, elem_size );
                if( even )
                {
                    CopyColumn( sptr0 + (count-1)*elem_size, src_step,
                                buf1.data(), elem_size, len, elem_size );
                    ExpandCCS( buf1.data(), len, elem_size );
                }
                sptr0 += elem_size;
            }
            else
            {
                CopyColumn( sptr0, src_step, buf0.data(), complex_elem_size, len, complex_elem_size );
                if( even )
                {
                    CopyColumn( sptr0 + b*complex_elem_size, src_step,
                                   buf1.data(), complex_elem_size, len, complex_elem_size );
                }
                sptr0 += complex_elem_size;
            }

            if( even )
                contextB->apply(buf1.data(), dbuf1);
            contextB->apply(buf0.data(), dbuf0);

            if( stage_dst_channels == 1 )
            {
                if( !inv )
                {
                    // copy the half of output vector to the first/last column.
                    // before doing that, defgragment the vector
                    memcpy( dbuf0 + elem_size, dbuf0, elem_size );
                    CopyColumn( dbuf0 + elem_size, elem_size, dptr0,
                                   dst_step, len, elem_size );
                    if( even )
                    {
                        memcpy( dbuf1 + elem_size, dbuf1, elem_size );
                        CopyColumn( dbuf1 + elem_size, elem_size,
                                       dptr0 + (count-1)*elem_size,
                                       dst_step, len, elem_size );
                    }
                    dptr0 += elem_size;
                }
                else
                {
                    // copy the real part of the complex vector to the first/last column
                    CopyColumn( dbuf0, complex_elem_size, dptr0, dst_step, len, elem_size );
                    if( even )
                        CopyColumn( dbuf1, complex_elem_size, dptr0 + (count-1)*elem_size,
                                       dst_step, len, elem_size );
                    dptr0 += elem_size;
                }
            }
            else
            {
                CV_Assert( !inv );
                CopyColumn( dbuf0, complex_elem_size, dptr0,
                               dst_step, len, complex_elem_size );
                if( even )
                    CopyColumn( dbuf1, complex_elem_size,
                                   dptr0 + b*complex_elem_size,
                                   dst_step, len, complex_elem_size );
                dptr0 += complex_elem_size;
            }
        }

        for(int i = a; i < b; i += 2 )
        {
            if( i+1 < b )
            {
                CopyFrom2Columns( sptr0, src_step, buf0.data(), buf1.data(), len, complex_elem_size );
                contextB->apply(buf1.data(), dbuf1);
            }
            else
                CopyColumn( sptr0, src_step, buf0.data(), complex_elem_size, len, complex_elem_size );

            contextB->apply(buf0.data(), dbuf0);

            if( i+1 < b )
                CopyTo2Columns( dbuf0, dbuf1, dptr0, dst_step, len, complex_elem_size );
            else
                CopyColumn( dbuf0, complex_elem_size, dptr0, dst_step, len, complex_elem_size );
            sptr0 += 2*complex_elem_size;
            dptr0 += 2*complex_elem_size;
        }
        if(isLastStage && mode == FwdRealToComplex)
            complementComplexOutput(depth, dst_data, dst_step, count, len, 2);
    }
};

class OcvDftBasicImpl CV_FINAL : public hal::DFT1D
{
public:
    OcvDftOptions opt;
    int _factors[34];
    AutoBuffer<uchar> wave_buf;
    AutoBuffer<int> itab_buf;
#ifdef USE_IPP_DFT
    AutoBuffer<uchar> ippbuf;
    AutoBuffer<uchar> ippworkbuf;
#endif

public:
    OcvDftBasicImpl()
    {
        opt.factors = _factors;
    }
    void init(int len, int count, int depth, int flags, bool *needBuffer)
    {
        int prev_len = opt.n;

        int stage = (flags & CV_HAL_DFT_STAGE_COLS) != 0 ? 1 : 0;
        int complex_elem_size = depth == CV_32F ? sizeof(Complex<float>) : sizeof(Complex<double>);
        opt.isInverse = (flags & CV_HAL_DFT_INVERSE) != 0;
        bool real_transform = (flags & CV_HAL_DFT_REAL_OUTPUT) != 0;
        opt.isComplex = (stage == 0) && (flags & CV_HAL_DFT_COMPLEX_OUTPUT) != 0;
        bool needAnotherStage = (flags & CV_HAL_DFT_TWO_STAGE) != 0;

        opt.scale = 1;
        opt.tab_size = len;
        opt.n = len;

        opt.useIpp = false;
    #ifdef USE_IPP_DFT
        opt.ipp_spec = 0;
        opt.ipp_work = 0;

        if( CV_IPP_CHECK_COND && (opt.n*count >= 64) ) // use IPP DFT if available
        {
            int ipp_norm_flag = (flags & CV_HAL_DFT_SCALE) == 0 ? 8 : opt.isInverse ? 2 : 1;
            int specsize=0, initsize=0, worksize=0;
            IppDFTGetSizeFunc getSizeFunc = 0;
            IppDFTInitFunc initFunc = 0;

            if( real_transform && stage == 0 )
            {
                if( depth == CV_32F )
                {
                    getSizeFunc = ippsDFTGetSize_R_32f;
                    initFunc = (IppDFTInitFunc)ippsDFTInit_R_32f;
                }
                else
                {
                    getSizeFunc = ippsDFTGetSize_R_64f;
                    initFunc = (IppDFTInitFunc)ippsDFTInit_R_64f;
                }
            }
            else
            {
                if( depth == CV_32F )
                {
                    getSizeFunc = ippsDFTGetSize_C_32fc;
                    initFunc = (IppDFTInitFunc)ippsDFTInit_C_32fc;
                }
                else
                {
                    getSizeFunc = ippsDFTGetSize_C_64fc;
                    initFunc = (IppDFTInitFunc)ippsDFTInit_C_64fc;
                }
            }
            if( getSizeFunc(opt.n, ipp_norm_flag, ippAlgHintNone, &specsize, &initsize, &worksize) >= 0 )
            {
                ippbuf.allocate(specsize + initsize + 64);
                opt.ipp_spec = alignPtr(&ippbuf[0], 32);
                ippworkbuf.allocate(worksize + 32);
                opt.ipp_work = alignPtr(&ippworkbuf[0], 32);
                uchar* initbuf = alignPtr((uchar*)opt.ipp_spec + specsize, 32);
                if( initFunc(opt.n, ipp_norm_flag, ippAlgHintNone, opt.ipp_spec, initbuf) >= 0 )
                    opt.useIpp = true;
            }
            else
                setIppErrorStatus();
        }
    #endif

        if (!opt.useIpp)
        {
            if (len != prev_len)
            {
                opt.nf = DFTFactorize( opt.n, opt.factors );
            }
            bool inplace_transform = opt.factors[0] == opt.factors[opt.nf-1];
            if (len != prev_len || (!inplace_transform && opt.isInverse && real_transform))
            {
                wave_buf.allocate(opt.n*complex_elem_size);
                opt.wave = wave_buf.data();
                itab_buf.allocate(opt.n);
                opt.itab = itab_buf.data();
                DFTInit( opt.n, opt.nf, opt.factors, opt.itab, complex_elem_size,
                         opt.wave, stage == 0 && opt.isInverse && real_transform );
            }
            // otherwise reuse the tables calculated on the previous stage
            if (needBuffer)
            {
                if( (stage == 0 && ((*needBuffer && !inplace_transform) || (real_transform && (len & 1)))) ||
                    (stage == 1 && !inplace_transform) )
                {
                    *needBuffer = true;
                }
            }
        }
        else
        {
            if (needBuffer)
            {
                *needBuffer = false;
            }
        }

        {
            static DFTFunc dft_tbl[6] =
            {
                (DFTFunc)DFT_32f,
                (DFTFunc)RealDFT_32f,
                (DFTFunc)CCSIDFT_32f,
                (DFTFunc)DFT_64f,
                (DFTFunc)RealDFT_64f,
                (DFTFunc)CCSIDFT_64f
            };
            int idx = 0;
            if (stage == 0)
            {
                if (real_transform)
                {
                    if (!opt.isInverse)
                        idx = 1;
                    else
                        idx = 2;
                }
            }
            if (depth == CV_64F)
                idx += 3;

            opt.dft_func = dft_tbl[idx];
        }

        if(!needAnotherStage && (flags & CV_HAL_DFT_SCALE) != 0)
        {
            int rowCount = count;
            if (stage == 0 && (flags & CV_HAL_DFT_ROWS) != 0)
                rowCount = 1;
            opt.scale = 1./(len * rowCount);
        }
    }

    void apply(const uchar *src, uchar *dst) CV_OVERRIDE
    {
        opt.dft_func(opt, src, dst);
    }

    void free() {}
};

struct ReplacementDFT1D : public hal::DFT1D
{
    cvhalDFT *context;
    bool isInitialized;

    ReplacementDFT1D() : context(0), isInitialized(false) {}
    bool init(int len, int count, int depth, int flags, bool *needBuffer)
    {
        int res = cv_hal_dftInit1D(&context, len, count, depth, flags, needBuffer);
        isInitialized = (res == CV_HAL_ERROR_OK);
        return isInitialized;
    }
    void apply(const uchar *src, uchar *dst) CV_OVERRIDE
    {
        if (isInitialized)
        {
            CALL_HAL(dft1D, cv_hal_dft1D, context, src, dst);
        }
    }
    ~ReplacementDFT1D()
    {
        if (isInitialized)
        {
            CALL_HAL(dftFree1D, cv_hal_dftFree1D, context);
        }
    }
};

struct ReplacementDFT2D : public hal::DFT2D
{
    cvhalDFT *context;
    bool isInitialized;

    ReplacementDFT2D() : context(0), isInitialized(false) {}
    bool init(int width, int height, int depth,
              int src_channels, int dst_channels,
              int flags, int nonzero_rows)
    {
        int res = cv_hal_dftInit2D(&context, width, height, depth, src_channels, dst_channels, flags, nonzero_rows);
        isInitialized = (res == CV_HAL_ERROR_OK);
        return isInitialized;
    }
    void apply(const uchar *src, size_t src_step, uchar *dst, size_t dst_step) CV_OVERRIDE
    {
        if (isInitialized)
        {
            CALL_HAL(dft2D, cv_hal_dft2D, context, src, src_step, dst, dst_step);
        }
    }
    ~ReplacementDFT2D()
    {
        if (isInitialized)
        {
            CALL_HAL(dftFree2D, cv_hal_dftFree1D, context);
        }
    }
};

namespace hal {

//================== 1D ======================

Ptr<DFT1D> DFT1D::create(int len, int count, int depth, int flags, bool *needBuffer)
{
    {
        ReplacementDFT1D *impl = new ReplacementDFT1D();
        if (impl->init(len, count, depth, flags, needBuffer))
        {
            return Ptr<DFT1D>(impl);
        }
        delete impl;
    }
    {
        OcvDftBasicImpl *impl = new OcvDftBasicImpl();
        impl->init(len, count, depth, flags, needBuffer);
        return Ptr<DFT1D>(impl);
    }
}

//================== 2D ======================

Ptr<DFT2D> DFT2D::create(int width, int height, int depth,
                         int src_channels, int dst_channels,
                         int flags, int nonzero_rows)
{
    {
        ReplacementDFT2D *impl = new ReplacementDFT2D();
        if (impl->init(width, height, depth, src_channels, dst_channels, flags, nonzero_rows))
        {
            return Ptr<DFT2D>(impl);
        }
        delete impl;
    }
    {
        if(width == 1 && nonzero_rows > 0 )
        {
            CV_Error( cv::Error::StsNotImplemented,
            "This mode (using nonzero_rows with a single-column matrix) breaks the function's logic, so it is prohibited.\n"
            "For fast convolution/correlation use 2-column matrix or single-row matrix instead" );
        }
        OcvDftImpl *impl = new OcvDftImpl();
        impl->init(width, height, depth, src_channels, dst_channels, flags, nonzero_rows);
        return Ptr<DFT2D>(impl);
    }
}

} // cv::hal::
} // cv::


void cv::dft( InputArray _src0, OutputArray _dst, int flags, int nonzero_rows )
{
    CV_INSTRUMENT_REGION();

#ifdef HAVE_CLAMDFFT
    CV_OCL_RUN(ocl::haveAmdFft() && ocl::Device::getDefault().type() != ocl::Device::TYPE_CPU &&
            _dst.isUMat() && _src0.dims() <= 2 && nonzero_rows == 0,
               ocl_dft_amdfft(_src0, _dst, flags))
#endif

#ifdef HAVE_OPENCL
    CV_OCL_RUN(_dst.isUMat() && _src0.dims() <= 2,
               ocl_dft(_src0, _dst, flags, nonzero_rows))
#endif

    Mat src0 = _src0.getMat(), src = src0;
    bool inv = (flags & DFT_INVERSE) != 0;
    int type = src.type();
    int depth = src.depth();

    CV_Assert( type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2 );

    // Fail if DFT_COMPLEX_INPUT is specified, but src is not 2 channels.
    CV_Assert( !((flags & DFT_COMPLEX_INPUT) && src.channels() != 2) );

    if( !inv && src.channels() == 1 && (flags & DFT_COMPLEX_OUTPUT) )
        _dst.create( src.size(), CV_MAKETYPE(depth, 2) );
    else if( inv && src.channels() == 2 && (flags & DFT_REAL_OUTPUT) )
        _dst.create( src.size(), depth );
    else
        _dst.create( src.size(), type );

    Mat dst = _dst.getMat();

    int f = 0;
    if (src.isContinuous() && dst.isContinuous())
        f |= CV_HAL_DFT_IS_CONTINUOUS;
    if (inv)
        f |= CV_HAL_DFT_INVERSE;
    if (flags & DFT_ROWS)
        f |= CV_HAL_DFT_ROWS;
    if (flags & DFT_SCALE)
        f |= CV_HAL_DFT_SCALE;
    if (src.data == dst.data)
        f |= CV_HAL_DFT_IS_INPLACE;
    Ptr<hal::DFT2D> c = hal::DFT2D::create(src.cols, src.rows, depth, src.channels(), dst.channels(), f, nonzero_rows);
    c->apply(src.data, src.step, dst.data, dst.step);
}


void cv::idft( InputArray src, OutputArray dst, int flags, int nonzero_rows )
{
    CV_INSTRUMENT_REGION();

    dft( src, dst, flags | DFT_INVERSE, nonzero_rows );
}

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_mulSpectrums( InputArray _srcA, InputArray _srcB,
                              OutputArray _dst, int flags, bool conjB )
{
    int atype = _srcA.type(), btype = _srcB.type(),
            rowsPerWI = ocl::Device::getDefault().isIntel() ? 4 : 1;
    Size asize = _srcA.size(), bsize = _srcB.size();
    CV_Assert(asize == bsize);

    if ( !(atype == CV_32FC2 && btype == CV_32FC2) || flags != 0 )
        return false;

    UMat A = _srcA.getUMat(), B = _srcB.getUMat();
    CV_Assert(A.size() == B.size());

    _dst.create(A.size(), atype);
    UMat dst = _dst.getUMat();

    ocl::Kernel k("mulAndScaleSpectrums",
                  ocl::core::mulspectrums_oclsrc,
                  format("%s", conjB ? "-D CONJ" : ""));
    if (k.empty())
        return false;

    k.args(ocl::KernelArg::ReadOnlyNoSize(A), ocl::KernelArg::ReadOnlyNoSize(B),
           ocl::KernelArg::WriteOnly(dst), rowsPerWI);

    size_t globalsize[2] = { (size_t)asize.width, ((size_t)asize.height + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

}

#endif

namespace {

#define VAL(buf, elem) (((T*)((char*)data ## buf + (step ## buf * (elem))))[0])
#define MUL_SPECTRUMS_COL(A, B, C) \
    VAL(C, 0) = VAL(A, 0) * VAL(B, 0); \
    for (size_t j = 1; j <= rows - 2; j += 2) \
    { \
        double a_re = VAL(A, j), a_im = VAL(A, j + 1); \
        double b_re = VAL(B, j), b_im = VAL(B, j + 1); \
        if (conjB) b_im = -b_im; \
        double c_re = a_re * b_re - a_im * b_im; \
        double c_im = a_re * b_im + a_im * b_re; \
        VAL(C, j) = (T)c_re; VAL(C, j + 1) = (T)c_im; \
    } \
    if ((rows & 1) == 0) \
        VAL(C, rows-1) = VAL(A, rows-1) * VAL(B, rows-1)

template <typename T, bool conjB> static inline
void mulSpectrums_processCol_noinplace(const T* dataA, const T* dataB, T* dataC, size_t stepA, size_t stepB, size_t stepC, size_t rows)
{
    MUL_SPECTRUMS_COL(A, B, C);
}

template <typename T, bool conjB> static inline
void mulSpectrums_processCol_inplaceA(const T* dataB, T* dataAC, size_t stepB, size_t stepAC, size_t rows)
{
    MUL_SPECTRUMS_COL(AC, B, AC);
}
template <typename T, bool conjB, bool inplaceA> static inline
void mulSpectrums_processCol(const T* dataA, const T* dataB, T* dataC, size_t stepA, size_t stepB, size_t stepC, size_t rows)
{
    if (inplaceA)
        mulSpectrums_processCol_inplaceA<T, conjB>(dataB, dataC, stepB, stepC, rows);
    else
        mulSpectrums_processCol_noinplace<T, conjB>(dataA, dataB, dataC, stepA, stepB, stepC, rows);
}
#undef MUL_SPECTRUMS_COL
#undef VAL

template <typename T, bool conjB, bool inplaceA> static inline
void mulSpectrums_processCols(const T* dataA, const T* dataB, T* dataC, size_t stepA, size_t stepB, size_t stepC, size_t rows, size_t cols)
{
    mulSpectrums_processCol<T, conjB, inplaceA>(dataA, dataB, dataC, stepA, stepB, stepC, rows);
    if ((cols & 1) == 0)
    {
        mulSpectrums_processCol<T, conjB, inplaceA>(dataA + cols - 1, dataB + cols - 1, dataC + cols - 1, stepA, stepB, stepC, rows);
    }
}

#define VAL(buf, elem) (data ## buf[(elem)])
#define MUL_SPECTRUMS_ROW(A, B, C) \
    for (size_t j = j0; j < j1; j += 2) \
    { \
        double a_re = VAL(A, j), a_im = VAL(A, j + 1); \
        double b_re = VAL(B, j), b_im = VAL(B, j + 1); \
        if (conjB) b_im = -b_im; \
        double c_re = a_re * b_re - a_im * b_im; \
        double c_im = a_re * b_im + a_im * b_re; \
        VAL(C, j) = (T)c_re; VAL(C, j + 1) = (T)c_im; \
    }
template <typename T, bool conjB> static inline
void mulSpectrums_processRow_noinplace(const T* dataA, const T* dataB, T* dataC, size_t j0, size_t j1)
{
    MUL_SPECTRUMS_ROW(A, B, C);
}
template <typename T, bool conjB> static inline
void mulSpectrums_processRow_inplaceA(const T* dataB, T* dataAC, size_t j0, size_t j1)
{
    MUL_SPECTRUMS_ROW(AC, B, AC);
}
template <typename T, bool conjB, bool inplaceA> static inline
void mulSpectrums_processRow(const T* dataA, const T* dataB, T* dataC, size_t j0, size_t j1)
{
    if (inplaceA)
        mulSpectrums_processRow_inplaceA<T, conjB>(dataB, dataC, j0, j1);
    else
        mulSpectrums_processRow_noinplace<T, conjB>(dataA, dataB, dataC, j0, j1);
}
#undef MUL_SPECTRUMS_ROW
#undef VAL

template <typename T, bool conjB, bool inplaceA> static inline
void mulSpectrums_processRows(const T* dataA, const T* dataB, T* dataC, size_t stepA, size_t stepB, size_t stepC, size_t rows, size_t cols, size_t j0, size_t j1, bool is_1d_CN1)
{
    while (rows-- > 0)
    {
        if (is_1d_CN1)
            dataC[0] = dataA[0]*dataB[0];
        mulSpectrums_processRow<T, conjB, inplaceA>(dataA, dataB, dataC, j0, j1);
        if (is_1d_CN1 && (cols & 1) == 0)
            dataC[j1] = dataA[j1]*dataB[j1];

        dataA = (const T*)(((char*)dataA) + stepA);
        dataB = (const T*)(((char*)dataB) + stepB);
        dataC =       (T*)(((char*)dataC) + stepC);
    }
}


template <typename T, bool conjB, bool inplaceA> static inline
void mulSpectrums_Impl_(const T* dataA, const T* dataB, T* dataC, size_t stepA, size_t stepB, size_t stepC, size_t rows, size_t cols, size_t j0, size_t j1, bool is_1d, bool isCN1)
{
    if (!is_1d && isCN1)
    {
        mulSpectrums_processCols<T, conjB, inplaceA>(dataA, dataB, dataC, stepA, stepB, stepC, rows, cols);
    }
    mulSpectrums_processRows<T, conjB, inplaceA>(dataA, dataB, dataC, stepA, stepB, stepC, rows, cols, j0, j1, is_1d && isCN1);
}
template <typename T, bool conjB> static inline
void mulSpectrums_Impl(const T* dataA, const T* dataB, T* dataC, size_t stepA, size_t stepB, size_t stepC, size_t rows, size_t cols, size_t j0, size_t j1, bool is_1d, bool isCN1)
{
    if (dataA == dataC)
        mulSpectrums_Impl_<T, conjB, true>(dataA, dataB, dataC, stepA, stepB, stepC, rows, cols, j0, j1, is_1d, isCN1);
    else
        mulSpectrums_Impl_<T, conjB, false>(dataA, dataB, dataC, stepA, stepB, stepC, rows, cols, j0, j1, is_1d, isCN1);
}

} // namespace

void cv::mulSpectrums( InputArray _srcA, InputArray _srcB,
                       OutputArray _dst, int flags, bool conjB )
{
    CV_INSTRUMENT_REGION();

    CV_OCL_RUN(_dst.isUMat() && _srcA.dims() <= 2 && _srcB.dims() <= 2,
            ocl_mulSpectrums(_srcA, _srcB, _dst, flags, conjB))

    Mat srcA = _srcA.getMat(), srcB = _srcB.getMat();
    int depth = srcA.depth(), cn = srcA.channels(), type = srcA.type();
    size_t rows = srcA.rows, cols = srcA.cols;

    CV_Assert( type == srcB.type() && srcA.size() == srcB.size() );
    CV_Assert( type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2 );

    _dst.create( srcA.rows, srcA.cols, type );
    Mat dst = _dst.getMat();

    // correct inplace support
    // Case 'dst.data == srcA.data' is handled by implementation,
    // because it is used frequently (filter2D, matchTemplate)
    if (dst.data == srcB.data)
        srcB = srcB.clone(); // workaround for B only

    bool is_1d = (flags & DFT_ROWS)
        || (rows == 1)
        || (cols == 1 && srcA.isContinuous() && srcB.isContinuous() && dst.isContinuous());

    if( is_1d && !(flags & DFT_ROWS) )
        cols = cols + rows - 1, rows = 1;

    bool isCN1 = cn == 1;
    size_t j0 = isCN1 ? 1 : 0;
    size_t j1 = cols*cn - (((cols & 1) == 0 && cn == 1) ? 1 : 0);

    if (depth == CV_32F)
    {
        const float* dataA = srcA.ptr<float>();
        const float* dataB = srcB.ptr<float>();
        float* dataC = dst.ptr<float>();
        if (!conjB)
            mulSpectrums_Impl<float, false>(dataA, dataB, dataC, srcA.step, srcB.step, dst.step, rows, cols, j0, j1, is_1d, isCN1);
        else
            mulSpectrums_Impl<float, true>(dataA, dataB, dataC, srcA.step, srcB.step, dst.step, rows, cols, j0, j1, is_1d, isCN1);
    }
    else
    {
        const double* dataA = srcA.ptr<double>();
        const double* dataB = srcB.ptr<double>();
        double* dataC = dst.ptr<double>();
        if (!conjB)
            mulSpectrums_Impl<double, false>(dataA, dataB, dataC, srcA.step, srcB.step, dst.step, rows, cols, j0, j1, is_1d, isCN1);
        else
            mulSpectrums_Impl<double, true>(dataA, dataB, dataC, srcA.step, srcB.step, dst.step, rows, cols, j0, j1, is_1d, isCN1);
    }
}


/****************************************************************************************\
                               Discrete Cosine Transform
\****************************************************************************************/

namespace cv
{

/* DCT is calculated using DFT, as described here:
   http://www.ece.utexas.edu/~bevans/courses/ee381k/lectures/09_DCT/lecture9/:
*/
template<typename T> static void
DCT( const OcvDftOptions & c, const T* src, size_t src_step, T* dft_src, T* dft_dst, T* dst, size_t dst_step,
     const Complex<T>* dct_wave )
{
    static const T sin_45 = (T)0.70710678118654752440084436210485;

    int n = c.n;
    int j, n2 = n >> 1;

    src_step /= sizeof(src[0]);
    dst_step /= sizeof(dst[0]);
    T* dst1 = dst + (n-1)*dst_step;

    if( n == 1 )
    {
        dst[0] = src[0];
        return;
    }

    for( j = 0; j < n2; j++, src += src_step*2 )
    {
        dft_src[j] = src[0];
        dft_src[n-j-1] = src[src_step];
    }

    RealDFT(c, dft_src, dft_dst);
    src = dft_dst;

    dst[0] = (T)(src[0]*dct_wave->re*sin_45);
    dst += dst_step;
    for( j = 1, dct_wave++; j < n2; j++, dct_wave++,
                                    dst += dst_step, dst1 -= dst_step )
    {
        T t0 = dct_wave->re*src[j*2-1] - dct_wave->im*src[j*2];
        T t1 = -dct_wave->im*src[j*2-1] - dct_wave->re*src[j*2];
        dst[0] = t0;
        dst1[0] = t1;
    }

    dst[0] = src[n-1]*dct_wave->re;
}


template<typename T> static void
IDCT( const OcvDftOptions & c, const T* src, size_t src_step, T* dft_src, T* dft_dst, T* dst, size_t dst_step,
      const Complex<T>* dct_wave)
{
    static const T sin_45 = (T)0.70710678118654752440084436210485;
    int n = c.n;
    int j, n2 = n >> 1;

    src_step /= sizeof(src[0]);
    dst_step /= sizeof(dst[0]);
    const T* src1 = src + (n-1)*src_step;

    if( n == 1 )
    {
        dst[0] = src[0];
        return;
    }

    dft_src[0] = (T)(src[0]*2*dct_wave->re*sin_45);
    src += src_step;
    for( j = 1, dct_wave++; j < n2; j++, dct_wave++,
                                    src += src_step, src1 -= src_step )
    {
        T t0 = dct_wave->re*src[0] - dct_wave->im*src1[0];
        T t1 = -dct_wave->im*src[0] - dct_wave->re*src1[0];
        dft_src[j*2-1] = t0;
        dft_src[j*2] = t1;
    }

    dft_src[n-1] = (T)(src[0]*2*dct_wave->re);
    CCSIDFT(c, dft_src, dft_dst);

    for( j = 0; j < n2; j++, dst += dst_step*2 )
    {
        dst[0] = dft_dst[j];
        dst[dst_step] = dft_dst[n-j-1];
    }
}


static void
DCTInit( int n, int elem_size, void* _wave, int inv )
{
    static const double DctScale[] =
    {
    0.707106781186547570, 0.500000000000000000, 0.353553390593273790,
    0.250000000000000000, 0.176776695296636890, 0.125000000000000000,
    0.088388347648318447, 0.062500000000000000, 0.044194173824159223,
    0.031250000000000000, 0.022097086912079612, 0.015625000000000000,
    0.011048543456039806, 0.007812500000000000, 0.005524271728019903,
    0.003906250000000000, 0.002762135864009952, 0.001953125000000000,
    0.001381067932004976, 0.000976562500000000, 0.000690533966002488,
    0.000488281250000000, 0.000345266983001244, 0.000244140625000000,
    0.000172633491500622, 0.000122070312500000, 0.000086316745750311,
    0.000061035156250000, 0.000043158372875155, 0.000030517578125000
    };

    int i;
    Complex<double> w, w1;
    double t, scale;

    if( n == 1 )
        return;

    CV_Assert( (n&1) == 0 );

    if( (n & (n - 1)) == 0 )
    {
        int m;
        for( m = 0; (unsigned)(1 << m) < (unsigned)n; m++ )
            ;
        scale = (!inv ? 2 : 1)*DctScale[m];
        w1.re = DFTTab[m+2][0];
        w1.im = -DFTTab[m+2][1];
    }
    else
    {
        t = 1./(2*n);
        scale = (!inv ? 2 : 1)*std::sqrt(t);
        w1.im = sin(-CV_PI*t);
        w1.re = std::sqrt(1. - w1.im*w1.im);
    }
    n >>= 1;

    if( elem_size == sizeof(Complex<double>) )
    {
        Complex<double>* wave = (Complex<double>*)_wave;

        w.re = scale;
        w.im = 0.;

        for( i = 0; i <= n; i++ )
        {
            wave[i] = w;
            t = w.re*w1.re - w.im*w1.im;
            w.im = w.re*w1.im + w.im*w1.re;
            w.re = t;
        }
    }
    else
    {
        Complex<float>* wave = (Complex<float>*)_wave;
        CV_Assert( elem_size == sizeof(Complex<float>) );

        w.re = (float)scale;
        w.im = 0.f;

        for( i = 0; i <= n; i++ )
        {
            wave[i].re = (float)w.re;
            wave[i].im = (float)w.im;
            t = w.re*w1.re - w.im*w1.im;
            w.im = w.re*w1.im + w.im*w1.re;
            w.re = t;
        }
    }
}


typedef void (*DCTFunc)(const OcvDftOptions & c, const void* src, size_t src_step, void* dft_src,
                        void* dft_dst, void* dst, size_t dst_step, const void* dct_wave);

static void DCT_32f(const OcvDftOptions & c, const float* src, size_t src_step, float* dft_src, float* dft_dst,
                    float* dst, size_t dst_step, const Complexf* dct_wave)
{
    DCT(c, src, src_step, dft_src, dft_dst, dst, dst_step, dct_wave);
}

static void IDCT_32f(const OcvDftOptions & c, const float* src, size_t src_step, float* dft_src, float* dft_dst,
                    float* dst, size_t dst_step, const Complexf* dct_wave)
{
    IDCT(c, src, src_step, dft_src, dft_dst, dst, dst_step, dct_wave);
}

static void DCT_64f(const OcvDftOptions & c, const double* src, size_t src_step, double* dft_src, double* dft_dst,
                    double* dst, size_t dst_step, const Complexd* dct_wave)
{
    DCT(c, src, src_step, dft_src, dft_dst, dst, dst_step, dct_wave);
}

static void IDCT_64f(const OcvDftOptions & c, const double* src, size_t src_step, double* dft_src, double* dft_dst,
                     double* dst, size_t dst_step, const Complexd* dct_wave)
{
    IDCT(c, src, src_step, dft_src, dft_dst, dst, dst_step, dct_wave);
}

}

#ifdef HAVE_IPP
namespace cv
{

#if IPP_VERSION_X100 >= 900
typedef IppStatus (CV_STDCALL * ippiDCTFunc)(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep, const void* pDCTSpec, Ipp8u* pBuffer);
typedef IppStatus (CV_STDCALL * ippiDCTInit)(void* pDCTSpec, IppiSize roiSize, Ipp8u* pMemInit );
typedef IppStatus (CV_STDCALL * ippiDCTGetSize)(IppiSize roiSize, int* pSizeSpec, int* pSizeInit, int* pSizeBuf);
#elif IPP_VERSION_X100 >= 700
typedef IppStatus (CV_STDCALL * ippiDCTFunc)(const Ipp32f*, int, Ipp32f*, int, const void*, Ipp8u*);
typedef IppStatus (CV_STDCALL * ippiDCTInitAlloc)(void**, IppiSize, IppHintAlgorithm);
typedef IppStatus (CV_STDCALL * ippiDCTFree)(void* pDCTSpec);
typedef IppStatus (CV_STDCALL * ippiDCTGetBufSize)(const void*, int*);
#endif

class DctIPPLoop_Invoker : public ParallelLoopBody
{
public:
    DctIPPLoop_Invoker(const uchar * _src, size_t _src_step, uchar * _dst, size_t _dst_step, int _width, bool _inv, bool *_ok) :
        ParallelLoopBody(), src(_src), src_step(_src_step), dst(_dst), dst_step(_dst_step), width(_width), inv(_inv), ok(_ok)
    {
        *ok = true;
    }

    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        if(*ok == false)
            return;

#if IPP_VERSION_X100 >= 900
        IppiSize srcRoiSize = {width, 1};

        int specSize    = 0;
        int initSize    = 0;
        int bufferSize  = 0;

        Ipp8u* pDCTSpec = NULL;
        Ipp8u* pBuffer  = NULL;
        Ipp8u* pInitBuf = NULL;

        #define IPP_RETURN              \
            if(pDCTSpec)                \
                ippFree(pDCTSpec);      \
            if(pBuffer)                 \
                ippFree(pBuffer);       \
            if(pInitBuf)                \
                ippFree(pInitBuf);      \
            return;

        ippiDCTFunc     ippiDCT_32f_C1R   = inv ? (ippiDCTFunc)ippiDCTInv_32f_C1R         : (ippiDCTFunc)ippiDCTFwd_32f_C1R;
        ippiDCTInit     ippDctInit     = inv ? (ippiDCTInit)ippiDCTInvInit_32f         : (ippiDCTInit)ippiDCTFwdInit_32f;
        ippiDCTGetSize  ippDctGetSize  = inv ? (ippiDCTGetSize)ippiDCTInvGetSize_32f   : (ippiDCTGetSize)ippiDCTFwdGetSize_32f;

        if(ippDctGetSize(srcRoiSize, &specSize, &initSize, &bufferSize) < 0)
        {
            *ok = false;
            return;
        }

        pDCTSpec = (Ipp8u*)CV_IPP_MALLOC(specSize);
        if(!pDCTSpec && specSize)
        {
            *ok = false;
            return;
        }

        pBuffer  = (Ipp8u*)CV_IPP_MALLOC(bufferSize);
        if(!pBuffer && bufferSize)
        {
            *ok = false;
            IPP_RETURN
        }
        pInitBuf = (Ipp8u*)CV_IPP_MALLOC(initSize);
        if(!pInitBuf && initSize)
        {
            *ok = false;
            IPP_RETURN
        }

        if(ippDctInit(pDCTSpec, srcRoiSize, pInitBuf) < 0)
        {
            *ok = false;
            IPP_RETURN
        }

        for(int i = range.start; i < range.end; ++i)
        {
            if(CV_INSTRUMENT_FUN_IPP(ippiDCT_32f_C1R, (float*)(src + src_step * i), static_cast<int>(src_step), (float*)(dst + dst_step * i), static_cast<int>(dst_step), pDCTSpec, pBuffer) < 0)
            {
                *ok = false;
                IPP_RETURN
            }
        }
        IPP_RETURN
#undef IPP_RETURN
#elif IPP_VERSION_X100 >= 700
        void* pDCTSpec;
        AutoBuffer<uchar> buf;
        uchar* pBuffer = 0;
        int bufSize=0;

        IppiSize srcRoiSize = {width, 1};

        CV_SUPPRESS_DEPRECATED_START

        ippiDCTFunc ippDctFun           = inv ? (ippiDCTFunc)ippiDCTInv_32f_C1R             : (ippiDCTFunc)ippiDCTFwd_32f_C1R;
        ippiDCTInitAlloc ippInitAlloc   = inv ? (ippiDCTInitAlloc)ippiDCTInvInitAlloc_32f   : (ippiDCTInitAlloc)ippiDCTFwdInitAlloc_32f;
        ippiDCTFree ippFree             = inv ? (ippiDCTFree)ippiDCTInvFree_32f             : (ippiDCTFree)ippiDCTFwdFree_32f;
        ippiDCTGetBufSize ippGetBufSize = inv ? (ippiDCTGetBufSize)ippiDCTInvGetBufSize_32f : (ippiDCTGetBufSize)ippiDCTFwdGetBufSize_32f;

        if (ippInitAlloc(&pDCTSpec, srcRoiSize, ippAlgHintNone)>=0 && ippGetBufSize(pDCTSpec, &bufSize)>=0)
        {
            buf.allocate( bufSize );
            pBuffer = (uchar*)buf;

            for( int i = range.start; i < range.end; ++i)
            {
                if(ippDctFun((float*)(src + src_step * i), static_cast<int>(src_step), (float*)(dst + dst_step * i), static_cast<int>(dst_step), pDCTSpec, (Ipp8u*)pBuffer) < 0)
                {
                    *ok = false;
                    break;
                }
            }
        }
        else
            *ok = false;

        if (pDCTSpec)
            ippFree(pDCTSpec);

        CV_SUPPRESS_DEPRECATED_END
#else
        CV_UNUSED(range);
        *ok = false;
#endif
    }

private:
    const uchar * src;
    size_t src_step;
    uchar * dst;
    size_t dst_step;
    int width;
    bool inv;
    bool *ok;
};

static bool DctIPPLoop(const uchar * src, size_t src_step, uchar * dst, size_t dst_step, int width, int height, bool inv)
{
    bool ok;
    parallel_for_(Range(0, height), DctIPPLoop_Invoker(src, src_step, dst, dst_step, width, inv, &ok), height/(double)(1<<4) );
    return ok;
}

static bool ippi_DCT_32f(const uchar * src, size_t src_step, uchar * dst, size_t dst_step, int width, int height, bool inv, bool row)
{
    CV_INSTRUMENT_REGION_IPP();

    if(row)
        return DctIPPLoop(src, src_step, dst, dst_step, width, height, inv);
    else
    {
#if IPP_VERSION_X100 >= 900
        IppiSize srcRoiSize = {width, height};

        int specSize    = 0;
        int initSize    = 0;
        int bufferSize  = 0;

        Ipp8u* pDCTSpec = NULL;
        Ipp8u* pBuffer  = NULL;
        Ipp8u* pInitBuf = NULL;

        #define IPP_RELEASE             \
            if(pDCTSpec)                \
                ippFree(pDCTSpec);      \
            if(pBuffer)                 \
                ippFree(pBuffer);       \
            if(pInitBuf)                \
                ippFree(pInitBuf);      \

        ippiDCTFunc     ippiDCT_32f_C1R      = inv ? (ippiDCTFunc)ippiDCTInv_32f_C1R         : (ippiDCTFunc)ippiDCTFwd_32f_C1R;
        ippiDCTInit     ippDctInit     = inv ? (ippiDCTInit)ippiDCTInvInit_32f         : (ippiDCTInit)ippiDCTFwdInit_32f;
        ippiDCTGetSize  ippDctGetSize  = inv ? (ippiDCTGetSize)ippiDCTInvGetSize_32f   : (ippiDCTGetSize)ippiDCTFwdGetSize_32f;

        if(ippDctGetSize(srcRoiSize, &specSize, &initSize, &bufferSize) < 0)
            return false;

        pDCTSpec = (Ipp8u*)CV_IPP_MALLOC(specSize);
        if(!pDCTSpec && specSize)
            return false;

        pBuffer  = (Ipp8u*)CV_IPP_MALLOC(bufferSize);
        if(!pBuffer && bufferSize)
        {
            IPP_RELEASE
            return false;
        }
        pInitBuf = (Ipp8u*)CV_IPP_MALLOC(initSize);
        if(!pInitBuf && initSize)
        {
            IPP_RELEASE
            return false;
        }

        if(ippDctInit(pDCTSpec, srcRoiSize, pInitBuf) < 0)
        {
            IPP_RELEASE
            return false;
        }

        if(CV_INSTRUMENT_FUN_IPP(ippiDCT_32f_C1R, (float*)src, static_cast<int>(src_step), (float*)dst, static_cast<int>(dst_step), pDCTSpec, pBuffer) < 0)
        {
            IPP_RELEASE
            return false;
        }

        IPP_RELEASE
        return true;
#undef IPP_RELEASE
#elif IPP_VERSION_X100 >= 700
        IppStatus status;
        void* pDCTSpec;
        AutoBuffer<uchar> buf;
        uchar* pBuffer = 0;
        int bufSize=0;

        IppiSize srcRoiSize = {width, height};

        CV_SUPPRESS_DEPRECATED_START

        ippiDCTFunc ippDctFun           = inv ? (ippiDCTFunc)ippiDCTInv_32f_C1R             : (ippiDCTFunc)ippiDCTFwd_32f_C1R;
        ippiDCTInitAlloc ippInitAlloc   = inv ? (ippiDCTInitAlloc)ippiDCTInvInitAlloc_32f   : (ippiDCTInitAlloc)ippiDCTFwdInitAlloc_32f;
        ippiDCTFree ippFree             = inv ? (ippiDCTFree)ippiDCTInvFree_32f             : (ippiDCTFree)ippiDCTFwdFree_32f;
        ippiDCTGetBufSize ippGetBufSize = inv ? (ippiDCTGetBufSize)ippiDCTInvGetBufSize_32f : (ippiDCTGetBufSize)ippiDCTFwdGetBufSize_32f;

        status = ippStsErr;

        if (ippInitAlloc(&pDCTSpec, srcRoiSize, ippAlgHintNone)>=0 && ippGetBufSize(pDCTSpec, &bufSize)>=0)
        {
            buf.allocate( bufSize );
            pBuffer = (uchar*)buf;

            status = ippDctFun((float*)src, static_cast<int>(src_step), (float*)dst, static_cast<int>(dst_step), pDCTSpec, (Ipp8u*)pBuffer);
        }

        if (pDCTSpec)
            ippFree(pDCTSpec);

        CV_SUPPRESS_DEPRECATED_END

        return status >= 0;
#else
        CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(inv); CV_UNUSED(row);
        return false;
#endif
    }
}
}
#endif

namespace cv {

class OcvDctImpl CV_FINAL : public hal::DCT2D
{
public:
    OcvDftOptions opt;

    int _factors[34];
    AutoBuffer<uint> wave_buf;
    AutoBuffer<int> itab_buf;

    DCTFunc dct_func;
    bool isRowTransform;
    bool isInverse;
    bool isContinuous;
    int start_stage;
    int end_stage;
    int width;
    int height;
    int depth;

    void init(int _width, int _height, int _depth, int flags)
    {
        width = _width;
        height = _height;
        depth = _depth;
        isInverse = (flags & CV_HAL_DFT_INVERSE) != 0;
        isRowTransform = (flags & CV_HAL_DFT_ROWS) != 0;
        isContinuous = (flags & CV_HAL_DFT_IS_CONTINUOUS) != 0;
        static DCTFunc dct_tbl[4] =
        {
            (DCTFunc)DCT_32f,
            (DCTFunc)IDCT_32f,
            (DCTFunc)DCT_64f,
            (DCTFunc)IDCT_64f
        };
        dct_func = dct_tbl[(int)isInverse + (depth == CV_64F)*2];
        opt.nf = 0;
        opt.isComplex = false;
        opt.isInverse = false;
        opt.noPermute = false;
        opt.scale = 1.;
        opt.factors = _factors;

        if (isRowTransform || height == 1 || (width == 1 && isContinuous))
        {
            start_stage = end_stage = 0;
        }
        else
        {
            start_stage = (width == 1);
            end_stage = 1;
        }
    }
    void apply(const uchar *src, size_t src_step, uchar *dst, size_t dst_step) CV_OVERRIDE
    {
        CV_IPP_RUN(IPP_VERSION_X100 >= 700 && depth == CV_32F, ippi_DCT_32f(src, src_step, dst, dst_step, width, height, isInverse, isRowTransform))

        AutoBuffer<uchar> dct_wave;
        AutoBuffer<uchar> src_buf, dst_buf;
        uchar *src_dft_buf = 0, *dst_dft_buf = 0;
        int prev_len = 0;
        int elem_size = (depth == CV_32F) ? sizeof(float) : sizeof(double);
        int complex_elem_size = elem_size*2;

        for(int stage = start_stage ; stage <= end_stage; stage++ )
        {
            const uchar* sptr = src;
            uchar* dptr = dst;
            size_t sstep0, sstep1, dstep0, dstep1;
            int len, count;

            if( stage == 0 )
            {
                len = width;
                count = height;
                if( len == 1 && !isRowTransform )
                {
                    len = height;
                    count = 1;
                }
                sstep0 = src_step;
                dstep0 = dst_step;
                sstep1 = dstep1 = elem_size;
            }
            else
            {
                len = height;
                count = width;
                sstep1 = src_step;
                dstep1 = dst_step;
                sstep0 = dstep0 = elem_size;
            }

            opt.n = len;
            opt.tab_size = len;

            if( len != prev_len )
            {
                if( len > 1 && (len & 1) )
                    CV_Error( cv::Error::StsNotImplemented, "Odd-size DCT\'s are not implemented" );

                opt.nf = DFTFactorize( len, opt.factors );
                bool inplace_transform = opt.factors[0] == opt.factors[opt.nf-1];

                wave_buf.allocate(len*complex_elem_size);
                opt.wave = wave_buf.data();
                itab_buf.allocate(len);
                opt.itab = itab_buf.data();
                DFTInit( len, opt.nf, opt.factors, opt.itab, complex_elem_size, opt.wave, isInverse );

                dct_wave.allocate((len/2 + 1)*complex_elem_size);
                src_buf.allocate(len*elem_size);
                src_dft_buf = src_buf.data();
                if(!inplace_transform)
                {
                    dst_buf.allocate(len*elem_size);
                    dst_dft_buf = dst_buf.data();
                }
                else
                {
                    dst_dft_buf = src_buf.data();
                }
                DCTInit( len, complex_elem_size, dct_wave.data(), isInverse);
                prev_len = len;
            }
            // otherwise reuse the tables calculated on the previous stage
            for(unsigned i = 0; i < static_cast<unsigned>(count); i++ )
            {
                dct_func( opt, sptr + i*sstep0, sstep1, src_dft_buf, dst_dft_buf,
                          dptr + i*dstep0, dstep1, dct_wave.data());
            }
            src = dst;
            src_step = dst_step;
        }
    }
};

struct ReplacementDCT2D : public hal::DCT2D
{
    cvhalDFT *context;
    bool isInitialized;

    ReplacementDCT2D() : context(0), isInitialized(false) {}
    bool init(int width, int height, int depth, int flags)
    {
        int res = cv_hal_dctInit2D(&context, width, height, depth, flags);
        isInitialized = (res == CV_HAL_ERROR_OK);
        return isInitialized;
    }
    void apply(const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step) CV_OVERRIDE
    {
        if (isInitialized)
        {
            CALL_HAL(dct2D, cv_hal_dct2D, context, src_data, src_step, dst_data, dst_step);
        }
    }
    ~ReplacementDCT2D()
    {
        if (isInitialized)
        {
            CALL_HAL(dctFree2D, cv_hal_dctFree2D, context);
        }
    }
};

namespace hal {

Ptr<DCT2D> DCT2D::create(int width, int height, int depth, int flags)
{
    {
        ReplacementDCT2D *impl = new ReplacementDCT2D();
        if (impl->init(width, height, depth, flags))
        {
            return Ptr<DCT2D>(impl);
        }
        delete impl;
    }
    {
        OcvDctImpl *impl = new OcvDctImpl();
        impl->init(width, height, depth, flags);
        return Ptr<DCT2D>(impl);
    }
}

} // cv::hal::
} // cv::

void cv::dct( InputArray _src0, OutputArray _dst, int flags )
{
    CV_INSTRUMENT_REGION();

    Mat src0 = _src0.getMat(), src = src0;
    int type = src.type(), depth = src.depth();

    CV_Assert( type == CV_32FC1 || type == CV_64FC1 );
    _dst.create( src.rows, src.cols, type );
    Mat dst = _dst.getMat();

    int f = 0;
    if ((flags & DFT_ROWS) != 0)
        f |= CV_HAL_DFT_ROWS;
    if ((flags & DCT_INVERSE) != 0)
        f |= CV_HAL_DFT_INVERSE;
    if (src.isContinuous() && dst.isContinuous())
        f |= CV_HAL_DFT_IS_CONTINUOUS;

    Ptr<hal::DCT2D> c = hal::DCT2D::create(src.cols, src.rows, depth, f);
    c->apply(src.data, src.step, dst.data, dst.step);
}


void cv::idct( InputArray src, OutputArray dst, int flags )
{
    CV_INSTRUMENT_REGION();

    dct( src, dst, flags | DCT_INVERSE );
}

namespace cv
{

static const int optimalDFTSizeTab[] = {
1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48,
50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128, 135, 144, 150, 160,
162, 180, 192, 200, 216, 225, 240, 243, 250, 256, 270, 288, 300, 320, 324, 360, 375,
384, 400, 405, 432, 450, 480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720,
729, 750, 768, 800, 810, 864, 900, 960, 972, 1000, 1024, 1080, 1125, 1152, 1200,
1215, 1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536, 1600, 1620, 1728, 1800, 1875,
1920, 1944, 2000, 2025, 2048, 2160, 2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592,
2700, 2880, 2916, 3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840,
3888, 4000, 4050, 4096, 4320, 4374, 4500, 4608, 4800, 4860, 5000, 5120, 5184, 5400,
5625, 5760, 5832, 6000, 6075, 6144, 6250, 6400, 6480, 6561, 6750, 6912, 7200, 7290,
7500, 7680, 7776, 8000, 8100, 8192, 8640, 8748, 9000, 9216, 9375, 9600, 9720, 10000,
10125, 10240, 10368, 10800, 10935, 11250, 11520, 11664, 12000, 12150, 12288, 12500,
12800, 12960, 13122, 13500, 13824, 14400, 14580, 15000, 15360, 15552, 15625, 16000,
16200, 16384, 16875, 17280, 17496, 18000, 18225, 18432, 18750, 19200, 19440, 19683,
20000, 20250, 20480, 20736, 21600, 21870, 22500, 23040, 23328, 24000, 24300, 24576,
25000, 25600, 25920, 26244, 27000, 27648, 28125, 28800, 29160, 30000, 30375, 30720,
31104, 31250, 32000, 32400, 32768, 32805, 33750, 34560, 34992, 36000, 36450, 36864,
37500, 38400, 38880, 39366, 40000, 40500, 40960, 41472, 43200, 43740, 45000, 46080,
46656, 46875, 48000, 48600, 49152, 50000, 50625, 51200, 51840, 52488, 54000, 54675,
55296, 56250, 57600, 58320, 59049, 60000, 60750, 61440, 62208, 62500, 64000, 64800,
65536, 65610, 67500, 69120, 69984, 72000, 72900, 73728, 75000, 76800, 77760, 78125,
78732, 80000, 81000, 81920, 82944, 84375, 86400, 87480, 90000, 91125, 92160, 93312,
93750, 96000, 97200, 98304, 98415, 100000, 101250, 102400, 103680, 104976, 108000,
109350, 110592, 112500, 115200, 116640, 118098, 120000, 121500, 122880, 124416, 125000,
128000, 129600, 131072, 131220, 135000, 138240, 139968, 140625, 144000, 145800, 147456,
150000, 151875, 153600, 155520, 156250, 157464, 160000, 162000, 163840, 164025, 165888,
168750, 172800, 174960, 177147, 180000, 182250, 184320, 186624, 187500, 192000, 194400,
196608, 196830, 200000, 202500, 204800, 207360, 209952, 216000, 218700, 221184, 225000,
230400, 233280, 234375, 236196, 240000, 243000, 245760, 248832, 250000, 253125, 256000,
259200, 262144, 262440, 270000, 273375, 276480, 279936, 281250, 288000, 291600, 294912,
295245, 300000, 303750, 307200, 311040, 312500, 314928, 320000, 324000, 327680, 328050,
331776, 337500, 345600, 349920, 354294, 360000, 364500, 368640, 373248, 375000, 384000,
388800, 390625, 393216, 393660, 400000, 405000, 409600, 414720, 419904, 421875, 432000,
437400, 442368, 450000, 455625, 460800, 466560, 468750, 472392, 480000, 486000, 491520,
492075, 497664, 500000, 506250, 512000, 518400, 524288, 524880, 531441, 540000, 546750,
552960, 559872, 562500, 576000, 583200, 589824, 590490, 600000, 607500, 614400, 622080,
625000, 629856, 640000, 648000, 655360, 656100, 663552, 675000, 691200, 699840, 703125,
708588, 720000, 729000, 737280, 746496, 750000, 759375, 768000, 777600, 781250, 786432,
787320, 800000, 810000, 819200, 820125, 829440, 839808, 843750, 864000, 874800, 884736,
885735, 900000, 911250, 921600, 933120, 937500, 944784, 960000, 972000, 983040, 984150,
995328, 1000000, 1012500, 1024000, 1036800, 1048576, 1049760, 1062882, 1080000, 1093500,
1105920, 1119744, 1125000, 1152000, 1166400, 1171875, 1179648, 1180980, 1200000,
1215000, 1228800, 1244160, 1250000, 1259712, 1265625, 1280000, 1296000, 1310720,
1312200, 1327104, 1350000, 1366875, 1382400, 1399680, 1406250, 1417176, 1440000,
1458000, 1474560, 1476225, 1492992, 1500000, 1518750, 1536000, 1555200, 1562500,
1572864, 1574640, 1594323, 1600000, 1620000, 1638400, 1640250, 1658880, 1679616,
1687500, 1728000, 1749600, 1769472, 1771470, 1800000, 1822500, 1843200, 1866240,
1875000, 1889568, 1920000, 1944000, 1953125, 1966080, 1968300, 1990656, 2000000,
2025000, 2048000, 2073600, 2097152, 2099520, 2109375, 2125764, 2160000, 2187000,
2211840, 2239488, 2250000, 2278125, 2304000, 2332800, 2343750, 2359296, 2361960,
2400000, 2430000, 2457600, 2460375, 2488320, 2500000, 2519424, 2531250, 2560000,
2592000, 2621440, 2624400, 2654208, 2657205, 2700000, 2733750, 2764800, 2799360,
2812500, 2834352, 2880000, 2916000, 2949120, 2952450, 2985984, 3000000, 3037500,
3072000, 3110400, 3125000, 3145728, 3149280, 3188646, 3200000, 3240000, 3276800,
3280500, 3317760, 3359232, 3375000, 3456000, 3499200, 3515625, 3538944, 3542940,
3600000, 3645000, 3686400, 3732480, 3750000, 3779136, 3796875, 3840000, 3888000,
3906250, 3932160, 3936600, 3981312, 4000000, 4050000, 4096000, 4100625, 4147200,
4194304, 4199040, 4218750, 4251528, 4320000, 4374000, 4423680, 4428675, 4478976,
4500000, 4556250, 4608000, 4665600, 4687500, 4718592, 4723920, 4782969, 4800000,
4860000, 4915200, 4920750, 4976640, 5000000, 5038848, 5062500, 5120000, 5184000,
5242880, 5248800, 5308416, 5314410, 5400000, 5467500, 5529600, 5598720, 5625000,
5668704, 5760000, 5832000, 5859375, 5898240, 5904900, 5971968, 6000000, 6075000,
6144000, 6220800, 6250000, 6291456, 6298560, 6328125, 6377292, 6400000, 6480000,
6553600, 6561000, 6635520, 6718464, 6750000, 6834375, 6912000, 6998400, 7031250,
7077888, 7085880, 7200000, 7290000, 7372800, 7381125, 7464960, 7500000, 7558272,
7593750, 7680000, 7776000, 7812500, 7864320, 7873200, 7962624, 7971615, 8000000,
8100000, 8192000, 8201250, 8294400, 8388608, 8398080, 8437500, 8503056, 8640000,
8748000, 8847360, 8857350, 8957952, 9000000, 9112500, 9216000, 9331200, 9375000,
9437184, 9447840, 9565938, 9600000, 9720000, 9765625, 9830400, 9841500, 9953280,
10000000, 10077696, 10125000, 10240000, 10368000, 10485760, 10497600, 10546875, 10616832,
10628820, 10800000, 10935000, 11059200, 11197440, 11250000, 11337408, 11390625, 11520000,
11664000, 11718750, 11796480, 11809800, 11943936, 12000000, 12150000, 12288000, 12301875,
12441600, 12500000, 12582912, 12597120, 12656250, 12754584, 12800000, 12960000, 13107200,
13122000, 13271040, 13286025, 13436928, 13500000, 13668750, 13824000, 13996800, 14062500,
14155776, 14171760, 14400000, 14580000, 14745600, 14762250, 14929920, 15000000, 15116544,
15187500, 15360000, 15552000, 15625000, 15728640, 15746400, 15925248, 15943230, 16000000,
16200000, 16384000, 16402500, 16588800, 16777216, 16796160, 16875000, 17006112, 17280000,
17496000, 17578125, 17694720, 17714700, 17915904, 18000000, 18225000, 18432000, 18662400,
18750000, 18874368, 18895680, 18984375, 19131876, 19200000, 19440000, 19531250, 19660800,
19683000, 19906560, 20000000, 20155392, 20250000, 20480000, 20503125, 20736000, 20971520,
20995200, 21093750, 21233664, 21257640, 21600000, 21870000, 22118400, 22143375, 22394880,
22500000, 22674816, 22781250, 23040000, 23328000, 23437500, 23592960, 23619600, 23887872,
23914845, 24000000, 24300000, 24576000, 24603750, 24883200, 25000000, 25165824, 25194240,
25312500, 25509168, 25600000, 25920000, 26214400, 26244000, 26542080, 26572050, 26873856,
27000000, 27337500, 27648000, 27993600, 28125000, 28311552, 28343520, 28800000, 29160000,
29296875, 29491200, 29524500, 29859840, 30000000, 30233088, 30375000, 30720000, 31104000,
31250000, 31457280, 31492800, 31640625, 31850496, 31886460, 32000000, 32400000, 32768000,
32805000, 33177600, 33554432, 33592320, 33750000, 34012224, 34171875, 34560000, 34992000,
35156250, 35389440, 35429400, 35831808, 36000000, 36450000, 36864000, 36905625, 37324800,
37500000, 37748736, 37791360, 37968750, 38263752, 38400000, 38880000, 39062500, 39321600,
39366000, 39813120, 39858075, 40000000, 40310784, 40500000, 40960000, 41006250, 41472000,
41943040, 41990400, 42187500, 42467328, 42515280, 43200000, 43740000, 44236800, 44286750,
44789760, 45000000, 45349632, 45562500, 46080000, 46656000, 46875000, 47185920, 47239200,
47775744, 47829690, 48000000, 48600000, 48828125, 49152000, 49207500, 49766400, 50000000,
50331648, 50388480, 50625000, 51018336, 51200000, 51840000, 52428800, 52488000, 52734375,
53084160, 53144100, 53747712, 54000000, 54675000, 55296000, 55987200, 56250000, 56623104,
56687040, 56953125, 57600000, 58320000, 58593750, 58982400, 59049000, 59719680, 60000000,
60466176, 60750000, 61440000, 61509375, 62208000, 62500000, 62914560, 62985600, 63281250,
63700992, 63772920, 64000000, 64800000, 65536000, 65610000, 66355200, 66430125, 67108864,
67184640, 67500000, 68024448, 68343750, 69120000, 69984000, 70312500, 70778880, 70858800,
71663616, 72000000, 72900000, 73728000, 73811250, 74649600, 75000000, 75497472, 75582720,
75937500, 76527504, 76800000, 77760000, 78125000, 78643200, 78732000, 79626240, 79716150,
80000000, 80621568, 81000000, 81920000, 82012500, 82944000, 83886080, 83980800, 84375000,
84934656, 85030560, 86400000, 87480000, 87890625, 88473600, 88573500, 89579520, 90000000,
90699264, 91125000, 92160000, 93312000, 93750000, 94371840, 94478400, 94921875, 95551488,
95659380, 96000000, 97200000, 97656250, 98304000, 98415000, 99532800, 100000000,
100663296, 100776960, 101250000, 102036672, 102400000, 102515625, 103680000, 104857600,
104976000, 105468750, 106168320, 106288200, 107495424, 108000000, 109350000, 110592000,
110716875, 111974400, 112500000, 113246208, 113374080, 113906250, 115200000, 116640000,
117187500, 117964800, 118098000, 119439360, 119574225, 120000000, 120932352, 121500000,
122880000, 123018750, 124416000, 125000000, 125829120, 125971200, 126562500, 127401984,
127545840, 128000000, 129600000, 131072000, 131220000, 132710400, 132860250, 134217728,
134369280, 135000000, 136048896, 136687500, 138240000, 139968000, 140625000, 141557760,
141717600, 143327232, 144000000, 145800000, 146484375, 147456000, 147622500, 149299200,
150000000, 150994944, 151165440, 151875000, 153055008, 153600000, 155520000, 156250000,
157286400, 157464000, 158203125, 159252480, 159432300, 160000000, 161243136, 162000000,
163840000, 164025000, 165888000, 167772160, 167961600, 168750000, 169869312, 170061120,
170859375, 172800000, 174960000, 175781250, 176947200, 177147000, 179159040, 180000000,
181398528, 182250000, 184320000, 184528125, 186624000, 187500000, 188743680, 188956800,
189843750, 191102976, 191318760, 192000000, 194400000, 195312500, 196608000, 196830000,
199065600, 199290375, 200000000, 201326592, 201553920, 202500000, 204073344, 204800000,
205031250, 207360000, 209715200, 209952000, 210937500, 212336640, 212576400, 214990848,
216000000, 218700000, 221184000, 221433750, 223948800, 225000000, 226492416, 226748160,
227812500, 230400000, 233280000, 234375000, 235929600, 236196000, 238878720, 239148450,
240000000, 241864704, 243000000, 244140625, 245760000, 246037500, 248832000, 250000000,
251658240, 251942400, 253125000, 254803968, 255091680, 256000000, 259200000, 262144000,
262440000, 263671875, 265420800, 265720500, 268435456, 268738560, 270000000, 272097792,
273375000, 276480000, 279936000, 281250000, 283115520, 283435200, 284765625, 286654464,
288000000, 291600000, 292968750, 294912000, 295245000, 298598400, 300000000, 301989888,
302330880, 303750000, 306110016, 307200000, 307546875, 311040000, 312500000, 314572800,
314928000, 316406250, 318504960, 318864600, 320000000, 322486272, 324000000, 327680000,
328050000, 331776000, 332150625, 335544320, 335923200, 337500000, 339738624, 340122240,
341718750, 345600000, 349920000, 351562500, 353894400, 354294000, 358318080, 360000000,
362797056, 364500000, 368640000, 369056250, 373248000, 375000000, 377487360, 377913600,
379687500, 382205952, 382637520, 384000000, 388800000, 390625000, 393216000, 393660000,
398131200, 398580750, 400000000, 402653184, 403107840, 405000000, 408146688, 409600000,
410062500, 414720000, 419430400, 419904000, 421875000, 424673280, 425152800, 429981696,
432000000, 437400000, 439453125, 442368000, 442867500, 447897600, 450000000, 452984832,
453496320, 455625000, 460800000, 466560000, 468750000, 471859200, 472392000, 474609375,
477757440, 478296900, 480000000, 483729408, 486000000, 488281250, 491520000, 492075000,
497664000, 500000000, 503316480, 503884800, 506250000, 509607936, 510183360, 512000000,
512578125, 518400000, 524288000, 524880000, 527343750, 530841600, 531441000, 536870912,
537477120, 540000000, 544195584, 546750000, 552960000, 553584375, 559872000, 562500000,
566231040, 566870400, 569531250, 573308928, 576000000, 583200000, 585937500, 589824000,
590490000, 597196800, 597871125, 600000000, 603979776, 604661760, 607500000, 612220032,
614400000, 615093750, 622080000, 625000000, 629145600, 629856000, 632812500, 637009920,
637729200, 640000000, 644972544, 648000000, 655360000, 656100000, 663552000, 664301250,
671088640, 671846400, 675000000, 679477248, 680244480, 683437500, 691200000, 699840000,
703125000, 707788800, 708588000, 716636160, 720000000, 725594112, 729000000, 732421875,
737280000, 738112500, 746496000, 750000000, 754974720, 755827200, 759375000, 764411904,
765275040, 768000000, 777600000, 781250000, 786432000, 787320000, 791015625, 796262400,
797161500, 800000000, 805306368, 806215680, 810000000, 816293376, 819200000, 820125000,
829440000, 838860800, 839808000, 843750000, 849346560, 850305600, 854296875, 859963392,
864000000, 874800000, 878906250, 884736000, 885735000, 895795200, 900000000, 905969664,
906992640, 911250000, 921600000, 922640625, 933120000, 937500000, 943718400, 944784000,
949218750, 955514880, 956593800, 960000000, 967458816, 972000000, 976562500, 983040000,
984150000, 995328000, 996451875, 1000000000, 1006632960, 1007769600, 1012500000,
1019215872, 1020366720, 1024000000, 1025156250, 1036800000, 1048576000, 1049760000,
1054687500, 1061683200, 1062882000, 1073741824, 1074954240, 1080000000, 1088391168,
1093500000, 1105920000, 1107168750, 1119744000, 1125000000, 1132462080, 1133740800,
1139062500, 1146617856, 1152000000, 1166400000, 1171875000, 1179648000, 1180980000,
1194393600, 1195742250, 1200000000, 1207959552, 1209323520, 1215000000, 1220703125,
1224440064, 1228800000, 1230187500, 1244160000, 1250000000, 1258291200, 1259712000,
1265625000, 1274019840, 1275458400, 1280000000, 1289945088, 1296000000, 1310720000,
1312200000, 1318359375, 1327104000, 1328602500, 1342177280, 1343692800, 1350000000,
1358954496, 1360488960, 1366875000, 1382400000, 1399680000, 1406250000, 1415577600,
1417176000, 1423828125, 1433272320, 1440000000, 1451188224, 1458000000, 1464843750,
1474560000, 1476225000, 1492992000, 1500000000, 1509949440, 1511654400, 1518750000,
1528823808, 1530550080, 1536000000, 1537734375, 1555200000, 1562500000, 1572864000,
1574640000, 1582031250, 1592524800, 1594323000, 1600000000, 1610612736, 1612431360,
1620000000, 1632586752, 1638400000, 1640250000, 1658880000, 1660753125, 1677721600,
1679616000, 1687500000, 1698693120, 1700611200, 1708593750, 1719926784, 1728000000,
1749600000, 1757812500, 1769472000, 1771470000, 1791590400, 1800000000, 1811939328,
1813985280, 1822500000, 1843200000, 1845281250, 1866240000, 1875000000, 1887436800,
1889568000, 1898437500, 1911029760, 1913187600, 1920000000, 1934917632, 1944000000,
1953125000, 1966080000, 1968300000, 1990656000, 1992903750, 2000000000, 2013265920,
2015539200, 2025000000, 2038431744, 2040733440, 2048000000, 2050312500, 2073600000,
2097152000, 2099520000, 2109375000, 2123366400, 2125764000
};

}

int cv::getOptimalDFTSize( int size0 )
{
    int a = 0, b = sizeof(optimalDFTSizeTab)/sizeof(optimalDFTSizeTab[0]) - 1;
    if( (unsigned)size0 >= (unsigned)optimalDFTSizeTab[b] )
        return -1;

    while( a < b )
    {
        int c = (a + b) >> 1;
        if( size0 <= optimalDFTSizeTab[c] )
            b = c;
        else
            a = c+1;
    }

    return optimalDFTSizeTab[b];
}


#ifndef OPENCV_EXCLUDE_C_API

CV_IMPL void
cvDFT( const CvArr* srcarr, CvArr* dstarr, int flags, int nonzero_rows )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst0 = cv::cvarrToMat(dstarr), dst = dst0;
    int _flags = ((flags & CV_DXT_INVERSE) ? cv::DFT_INVERSE : 0) |
        ((flags & CV_DXT_SCALE) ? cv::DFT_SCALE : 0) |
        ((flags & CV_DXT_ROWS) ? cv::DFT_ROWS : 0);

    CV_Assert( src.size == dst.size );

    if( src.type() != dst.type() )
    {
        if( dst.channels() == 2 )
            _flags |= cv::DFT_COMPLEX_OUTPUT;
        else
            _flags |= cv::DFT_REAL_OUTPUT;
    }

    cv::dft( src, dst, _flags, nonzero_rows );
    CV_Assert( dst.data == dst0.data ); // otherwise it means that the destination size or type was incorrect
}


CV_IMPL void
cvMulSpectrums( const CvArr* srcAarr, const CvArr* srcBarr,
                CvArr* dstarr, int flags )
{
    cv::Mat srcA = cv::cvarrToMat(srcAarr),
        srcB = cv::cvarrToMat(srcBarr),
        dst = cv::cvarrToMat(dstarr);
    CV_Assert( srcA.size == dst.size && srcA.type() == dst.type() );

    cv::mulSpectrums(srcA, srcB, dst,
        (flags & CV_DXT_ROWS) ? cv::DFT_ROWS : 0,
        (flags & CV_DXT_MUL_CONJ) != 0 );
}


CV_IMPL void
cvDCT( const CvArr* srcarr, CvArr* dstarr, int flags )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src.size == dst.size && src.type() == dst.type() );
    int _flags = ((flags & CV_DXT_INVERSE) ? cv::DCT_INVERSE : 0) |
            ((flags & CV_DXT_ROWS) ? cv::DCT_ROWS : 0);
    cv::dct( src, dst, _flags );
}


CV_IMPL int
cvGetOptimalDFTSize( int size0 )
{
    return cv::getOptimalDFTSize(size0);
}

#endif  // OPENCV_EXCLUDE_C_API
/* End of file. */
