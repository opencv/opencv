/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#if defined _M_IX86 && defined _MSC_VER && _MSC_VER < 1700
#pragma float_control(precise, on)
#endif

namespace cv {

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void JacobiSVD32f(float* At, size_t astep, float* W, float* Vt, size_t vstep, int m, int n, int n1);
void JacobiSVD64f(double* At, size_t astep, double* W, double* Vt, size_t vstep, int m, int n, int n1);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

template<typename _Tp> static inline _Tp hypot(_Tp a, _Tp b)
{
    a = std::abs(a);
    b = std::abs(b);
    if( a > b )
    {
        b /= a;
        return a*std::sqrt(1 + b*b);
    }
    if( b > 0 )
    {
        a /= b;
        return b*std::sqrt(1 + a*a);
    }
    return 0;
}

template<typename T> struct VBLAS
{
    int dot(const T*, const T*, int, T*) const { return 0; }
    int givens(T*, T*, int, T, T) const { return 0; }
    void dotD(const T*, const T*, int, double*) const {}
    void givensD(T*, T*, int, T, T, double*, double*) const {}
};

#if CV_SIMD // TODO: enable for CV_SIMD_SCALABLE, GCC 13 related
template<> inline int VBLAS<float>::dot(const float* a, const float* b, int n, float* result) const
{
    if( n < 2*VTraits<v_float32>::vlanes() )
        return 0;
    int k = 0;
    v_float32 s0 = vx_setzero_f32();
    for( ; k <= n - VTraits<v_float32>::vlanes(); k += VTraits<v_float32>::vlanes() )
    {
        v_float32 a0 = vx_load(a + k);
        v_float32 b0 = vx_load(b + k);

        s0 = v_add(s0, v_mul(a0, b0));
    }
    *result = v_reduce_sum(s0);
    vx_cleanup();
    return k;
}


template<> inline int VBLAS<float>::givens(float* a, float* b, int n, float c, float s) const
{
    if( n < VTraits<v_float32>::vlanes())
        return 0;
    int k = 0;
    v_float32 c4 = vx_setall_f32(c), s4 = vx_setall_f32(s);
    v_float32 ns4 = vx_setall_f32(-s);
    for( ; k <= n - VTraits<v_float32>::vlanes(); k += VTraits<v_float32>::vlanes() )
    {
        v_float32 a0 = vx_load(a + k);
        v_float32 b0 = vx_load(b + k);
        v_float32 t0 = v_fma(a0, c4, v_mul(b0, s4));
        v_float32 t1 = v_fma(a0, ns4, v_mul(b0, c4));
        v_store(a + k, t0);
        v_store(b + k, t1);
    }
    vx_cleanup();
    return k;
}


#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
template<> inline void VBLAS<float>::dotD(const float* a, const float* b, int n, double* result) const
{
    if( n <= 0 )
        return;
    const int vl = VTraits<v_float32>::vlanes();
    const int half = vl / 2;
    v_float64 s0 = vx_setzero_f64(), s1 = vx_setzero_f64();
    int k = 0;
    for( ; k <= n - vl; k += vl )
    {
        v_float32 a0 = vx_load(a + k);
        v_float32 b0 = vx_load(b + k);
        s0 = v_add(s0, v_mul(v_cvt_f64(a0), v_cvt_f64(b0)));
        s1 = v_add(s1, v_mul(v_cvt_f64_high(a0), v_cvt_f64_high(b0)));
    }
    for( ; k + half <= n; k += half )
    {
        v_float32 a0 = vx_load_low(a + k);
        v_float32 b0 = vx_load_low(b + k);
        s0 = v_add(s0, v_mul(v_cvt_f64(a0), v_cvt_f64(b0)));
        s1 = v_add(s1, v_mul(v_cvt_f64_high(a0), v_cvt_f64_high(b0)));
    }
    *result += v_reduce_sum(v_add(s0, s1));
    for( ; k < n; k++ )
        *result += (double)a[k]*(double)b[k];
    vx_cleanup();
}


template<> inline void VBLAS<float>::givensD(float* a, float* b, int n, float c, float s,
                                            double* na, double* nb) const
{
    if( n <= 0 )
        return;
    const int vl = VTraits<v_float32>::vlanes();
    const int half = vl / 2;
    v_float32 c4 = vx_setall_f32(c), s4 = vx_setall_f32(s);
    v_float32 ns4 = vx_setall_f32(-s);
    v_float64 a0d = vx_setzero_f64(), a1d = vx_setzero_f64();
    v_float64 b0d = vx_setzero_f64(), b1d = vx_setzero_f64();
    int k = 0;
    for( ; k <= n - vl; k += vl )
    {
        v_float32 a0 = vx_load(a + k);
        v_float32 b0 = vx_load(b + k);
        v_float32 t0 = v_add(v_mul(a0, c4), v_mul(b0, s4));
        v_float32 t1 = v_add(v_mul(a0, ns4), v_mul(b0, c4));
        v_store(a + k, t0);
        v_store(b + k, t1);
        v_float64 t0lo = v_cvt_f64(t0), t0hi = v_cvt_f64_high(t0);
        v_float64 t1lo = v_cvt_f64(t1), t1hi = v_cvt_f64_high(t1);
        a0d = v_add(a0d, v_mul(t0lo, t0lo));
        a1d = v_add(a1d, v_mul(t0hi, t0hi));
        b0d = v_add(b0d, v_mul(t1lo, t1lo));
        b1d = v_add(b1d, v_mul(t1hi, t1hi));
    }
    for( ; k + half <= n; k += half )
    {
        v_float32 a0 = vx_load_low(a + k);
        v_float32 b0 = vx_load_low(b + k);
        v_float32 t0 = v_add(v_mul(a0, c4), v_mul(b0, s4));
        v_float32 t1 = v_add(v_mul(a0, ns4), v_mul(b0, c4));
        v_store_low(a + k, t0);
        v_store_low(b + k, t1);
        v_float64 t0lo = v_cvt_f64(t0), t0hi = v_cvt_f64_high(t0);
        v_float64 t1lo = v_cvt_f64(t1), t1hi = v_cvt_f64_high(t1);
        a0d = v_add(a0d, v_mul(t0lo, t0lo));
        a1d = v_add(a1d, v_mul(t0hi, t0hi));
        b0d = v_add(b0d, v_mul(t1lo, t1lo));
        b1d = v_add(b1d, v_mul(t1hi, t1hi));
    }
    *na += v_reduce_sum(v_add(a0d, a1d));
    *nb += v_reduce_sum(v_add(b0d, b1d));
    for( ; k < n; k++ )
    {
        float t0 = c*a[k] + s*b[k];
        float t1 = -s*a[k] + c*b[k];
        a[k] = t0; b[k] = t1;
        *na += (double)t0*t0;
        *nb += (double)t1*t1;
    }
    vx_cleanup();
}


template<> inline void VBLAS<double>::dotD(const double* a, const double* b, int n, double* result) const
{
    if( n <= 0 )
        return;
    const int vl = VTraits<v_float64>::vlanes();
    const int half = vl / 2;
    v_float64 s0 = vx_setzero_f64();
    int k = 0;
    for( ; k <= n - vl; k += vl )
    {
        v_float64 a0 = vx_load(a + k);
        v_float64 b0 = vx_load(b + k);
        s0 = v_add(s0, v_mul(a0, b0));
    }
    for( ; k + half <= n; k += half )
    {
        v_float64 a0 = vx_load_low(a + k);
        v_float64 b0 = vx_load_low(b + k);
        s0 = v_add(s0, v_mul(a0, b0));
    }
    *result += v_reduce_sum(s0);
    for( ; k < n; k++ )
        *result += a[k]*b[k];
    vx_cleanup();
}


template<> inline void VBLAS<double>::givensD(double* a, double* b, int n, double c, double s,
                                             double* na, double* nb) const
{
    if( n <= 0 )
        return;
    const int vl = VTraits<v_float64>::vlanes();
    const int half = vl / 2;
    v_float64 c2 = vx_setall_f64(c), s2 = vx_setall_f64(s);
    v_float64 ns2 = vx_setall_f64(-s);
    v_float64 nacc = vx_setzero_f64(), nbcc = vx_setzero_f64();
    int k = 0;
    for( ; k <= n - vl; k += vl )
    {
        v_float64 a0 = vx_load(a + k);
        v_float64 b0 = vx_load(b + k);
        v_float64 t0 = v_add(v_mul(a0, c2), v_mul(b0, s2));
        v_float64 t1 = v_add(v_mul(a0, ns2), v_mul(b0, c2));
        v_store(a + k, t0);
        v_store(b + k, t1);
        nacc = v_add(nacc, v_mul(t0, t0));
        nbcc = v_add(nbcc, v_mul(t1, t1));
    }
    for( ; k + half <= n; k += half )
    {
        v_float64 a0 = vx_load_low(a + k);
        v_float64 b0 = vx_load_low(b + k);
        v_float64 t0 = v_add(v_mul(a0, c2), v_mul(b0, s2));
        v_float64 t1 = v_add(v_mul(a0, ns2), v_mul(b0, c2));
        v_store_low(a + k, t0);
        v_store_low(b + k, t1);
        nacc = v_add(nacc, v_mul(t0, t0));
        nbcc = v_add(nbcc, v_mul(t1, t1));
    }
    *na += v_reduce_sum(nacc);
    *nb += v_reduce_sum(nbcc);
    for( ; k < n; k++ )
    {
        double t0 = c*a[k] + s*b[k];
        double t1 = -s*a[k] + c*b[k];
        a[k] = t0; b[k] = t1;
        *na += t0*t0;
        *nb += t1*t1;
    }
    vx_cleanup();
}
#endif // CV_SIMD_64F


#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
template<> inline int VBLAS<double>::dot(const double* a, const double* b, int n, double* result) const
{
    if( n < 2*VTraits<v_float64>::vlanes() )
        return 0;
    int k = 0;
    v_float64 s0 = vx_setzero_f64();
    for( ; k <= n - VTraits<v_float64>::vlanes(); k += VTraits<v_float64>::vlanes() )
    {
        v_float64 a0 = vx_load(a + k);
        v_float64 b0 = vx_load(b + k);

        s0 = v_add(s0, v_mul(a0, b0));
    }
    *result = v_reduce_sum(s0);
    vx_cleanup();
    return k;
}


template<> inline int VBLAS<double>::givens(double* a, double* b, int n, double c, double s) const
{
    int k = 0;
    v_float64 c2 = vx_setall_f64(c), s2 = vx_setall_f64(s);
    v_float64 ns2 = vx_setall_f64(-s);
    for( ; k <= n - VTraits<v_float64>::vlanes(); k += VTraits<v_float64>::vlanes() )
    {
        v_float64 a0 = vx_load(a + k);
        v_float64 b0 = vx_load(b + k);
        v_float64 t0 = v_fma(a0, c2, v_mul(b0, s2));
        v_float64 t1 = v_fma(a0, ns2, v_mul(b0, c2));
        v_store(a + k, t0);
        v_store(b + k, t1);
    }
    vx_cleanup();
    return k;
}

#endif // CV_SIMD_64F
#endif // CV_SIMD

template<typename _Tp> void
JacobiSVDImpl_(_Tp* At, size_t astep, _Tp* _W, _Tp* Vt, size_t vstep,
               int m, int n, int n1, double minval, _Tp eps)
{
    VBLAS<_Tp> vblas;
    AutoBuffer<double> Wbuf(n);
    double* W = Wbuf.data();
    int i, j, k, iter, max_iter = std::max(m, 30);
    _Tp c, s;
    double sd;
    astep /= sizeof(At[0]);
    vstep /= sizeof(Vt[0]);

    for( i = 0; i < n; i++ )
    {
        _Tp* Ai = At + i*astep;
        sd = 0;
        vblas.dotD(Ai, Ai, m, &sd);
        W[i] = sd;

        if( Vt )
        {
            for( k = 0; k < n; k++ )
                Vt[i*vstep + k] = 0;
            Vt[i*vstep + i] = 1;
        }
    }

    for( iter = 0; iter < max_iter; iter++ )
    {
        bool changed = false;

        for( i = 0; i < n-1; i++ )
            for( j = i+1; j < n; j++ )
            {
                _Tp *Ai = At + i*astep, *Aj = At + j*astep;
                double a = W[i], p = 0, b = W[j];

                vblas.dotD(Ai, Aj, m, &p);

                if( std::abs(p) <= eps*std::sqrt((double)a*b) )
                    continue;

                p *= 2;
                double beta = a - b, gamma = hypot((double)p, beta);
                if( beta < 0 )
                {
                    double delta = (gamma - beta)*0.5;
                    s = (_Tp)std::sqrt(delta/gamma);
                    c = (_Tp)(p/(gamma*s*2));
                }
                else
                {
                    c = (_Tp)std::sqrt((gamma + beta)/(gamma*2));
                    s = (_Tp)(p/(gamma*c*2));
                }

                a = b = 0;
                vblas.givensD(Ai, Aj, m, c, s, &a, &b);
                W[i] = a; W[j] = b;

                changed = true;

                if( Vt )
                {
                    _Tp *Vi = Vt + i*vstep, *Vj = Vt + j*vstep;
                    k = vblas.givens(Vi, Vj, n, c, s);

                    for( ; k < n; k++ )
                    {
                        _Tp t0 = c*Vi[k] + s*Vj[k];
                        _Tp t1 = -s*Vi[k] + c*Vj[k];
                        Vi[k] = t0; Vj[k] = t1;
                    }
                }
            }
        if( !changed )
            break;
    }

    for( i = 0; i < n; i++ )
    {
        _Tp* Ai = At + i*astep;
        sd = 0;
        vblas.dotD(Ai, Ai, m, &sd);
        W[i] = std::sqrt(sd);
    }

    for( i = 0; i < n-1; i++ )
    {
        j = i;
        for( k = i+1; k < n; k++ )
        {
            if( W[j] < W[k] )
                j = k;
        }
        if( i != j )
        {
            std::swap(W[i], W[j]);
            if( Vt )
            {
                for( k = 0; k < m; k++ )
                    std::swap(At[i*astep + k], At[j*astep + k]);

                for( k = 0; k < n; k++ )
                    std::swap(Vt[i*vstep + k], Vt[j*vstep + k]);
            }
        }
    }

    for( i = 0; i < n; i++ )
        _W[i] = (_Tp)W[i];

    if( !Vt )
        return;

    RNG rng(0x12345678);
    for( i = 0; i < n1; i++ )
    {
        sd = i < n ? W[i] : 0;

        for( int ii = 0; ii < 100 && sd <= minval; ii++ )
        {
            // if we got a zero singular value, then in order to get the corresponding left singular vector
            // we generate a random vector, project it to the previously computed left singular vectors,
            // subtract the projection and normalize the difference.
            const _Tp val0 = (_Tp)(1./m);
            for( k = 0; k < m; k++ )
            {
                _Tp val = (rng.next() & 256) != 0 ? val0 : -val0;
                At[i*astep + k] = val;
            }
            for( iter = 0; iter < 2; iter++ )
            {
                for( j = 0; j < i; j++ )
                {
                    sd = 0;
                    for( k = 0; k < m; k++ )
                        sd += At[i*astep + k]*At[j*astep + k];
                    _Tp asum = 0;
                    for( k = 0; k < m; k++ )
                    {
                        _Tp t = (_Tp)(At[i*astep + k] - sd*At[j*astep + k]);
                        At[i*astep + k] = t;
                        asum += std::abs(t);
                    }
                    asum = asum > eps*100 ? 1/asum : 0;
                    for( k = 0; k < m; k++ )
                        At[i*astep + k] *= asum;
                }
            }
            sd = 0;
            for( k = 0; k < m; k++ )
            {
                _Tp t = At[i*astep + k];
                sd += (double)t*t;
            }
            sd = std::sqrt(sd);
        }

        s = (_Tp)(sd > minval ? 1/sd : 0.);
        for( k = 0; k < m; k++ )
            At[i*astep + k] *= s;
    }
}

void JacobiSVD32f(float* At, size_t astep, float* W, float* Vt, size_t vstep, int m, int n, int n1)
{
    JacobiSVDImpl_(At, astep, W, Vt, vstep, m, n, n1, FLT_MIN, FLT_EPSILON*2);
}

void JacobiSVD64f(double* At, size_t astep, double* W, double* Vt, size_t vstep, int m, int n, int n1)
{
    JacobiSVDImpl_(At, astep, W, Vt, vstep, m, n, n1, DBL_MIN, DBL_EPSILON*10);
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
