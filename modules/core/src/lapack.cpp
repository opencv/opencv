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
#include <limits>

#if defined _M_IX86 && defined _MSC_VER && _MSC_VER < 1700
#pragma float_control(precise, on)
#endif

namespace cv
{

int LU(float* A, size_t astep, int m, float* b, size_t bstep, int n)
{
    CV_INSTRUMENT_REGION()

    return hal::LU32f(A, astep, m, b, bstep, n);
}

int LU(double* A, size_t astep, int m, double* b, size_t bstep, int n)
{
    CV_INSTRUMENT_REGION()

    return hal::LU64f(A, astep, m, b, bstep, n);
}

bool Cholesky(float* A, size_t astep, int m, float* b, size_t bstep, int n)
{
    CV_INSTRUMENT_REGION()

    return hal::Cholesky32f(A, astep, m, b, bstep, n);
}

bool Cholesky(double* A, size_t astep, int m, double* b, size_t bstep, int n)
{
    CV_INSTRUMENT_REGION()

    return hal::Cholesky64f(A, astep, m, b, bstep, n);
}

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


template<typename _Tp> bool
JacobiImpl_( _Tp* A, size_t astep, _Tp* W, _Tp* V, size_t vstep, int n, uchar* buf )
{
    const _Tp eps = std::numeric_limits<_Tp>::epsilon();
    int i, j, k, m;

    astep /= sizeof(A[0]);
    if( V )
    {
        vstep /= sizeof(V[0]);
        for( i = 0; i < n; i++ )
        {
            for( j = 0; j < n; j++ )
                V[i*vstep + j] = (_Tp)0;
            V[i*vstep + i] = (_Tp)1;
        }
    }

    int iters, maxIters = n*n*30;

    int* indR = (int*)alignPtr(buf, sizeof(int));
    int* indC = indR + n;
    _Tp mv = (_Tp)0;

    for( k = 0; k < n; k++ )
    {
        W[k] = A[(astep + 1)*k];
        if( k < n - 1 )
        {
            for( m = k+1, mv = std::abs(A[astep*k + m]), i = k+2; i < n; i++ )
            {
                _Tp val = std::abs(A[astep*k+i]);
                if( mv < val )
                    mv = val, m = i;
            }
            indR[k] = m;
        }
        if( k > 0 )
        {
            for( m = 0, mv = std::abs(A[k]), i = 1; i < k; i++ )
            {
                _Tp val = std::abs(A[astep*i+k]);
                if( mv < val )
                    mv = val, m = i;
            }
            indC[k] = m;
        }
    }

    if( n > 1 ) for( iters = 0; iters < maxIters; iters++ )
    {
        // find index (k,l) of pivot p
        for( k = 0, mv = std::abs(A[indR[0]]), i = 1; i < n-1; i++ )
        {
            _Tp val = std::abs(A[astep*i + indR[i]]);
            if( mv < val )
                mv = val, k = i;
        }
        int l = indR[k];
        for( i = 1; i < n; i++ )
        {
            _Tp val = std::abs(A[astep*indC[i] + i]);
            if( mv < val )
                mv = val, k = indC[i], l = i;
        }

        _Tp p = A[astep*k + l];
        if( std::abs(p) <= eps )
            break;
        _Tp y = (_Tp)((W[l] - W[k])*0.5);
        _Tp t = std::abs(y) + hypot(p, y);
        _Tp s = hypot(p, t);
        _Tp c = t/s;
        s = p/s; t = (p/t)*p;
        if( y < 0 )
            s = -s, t = -t;
        A[astep*k + l] = 0;

        W[k] -= t;
        W[l] += t;

        _Tp a0, b0;

#undef rotate
#define rotate(v0, v1) a0 = v0, b0 = v1, v0 = a0*c - b0*s, v1 = a0*s + b0*c

        // rotate rows and columns k and l
        for( i = 0; i < k; i++ )
            rotate(A[astep*i+k], A[astep*i+l]);
        for( i = k+1; i < l; i++ )
            rotate(A[astep*k+i], A[astep*i+l]);
        for( i = l+1; i < n; i++ )
            rotate(A[astep*k+i], A[astep*l+i]);

        // rotate eigenvectors
        if( V )
            for( i = 0; i < n; i++ )
                rotate(V[vstep*k+i], V[vstep*l+i]);

#undef rotate

        for( j = 0; j < 2; j++ )
        {
            int idx = j == 0 ? k : l;
            if( idx < n - 1 )
            {
                for( m = idx+1, mv = std::abs(A[astep*idx + m]), i = idx+2; i < n; i++ )
                {
                    _Tp val = std::abs(A[astep*idx+i]);
                    if( mv < val )
                        mv = val, m = i;
                }
                indR[idx] = m;
            }
            if( idx > 0 )
            {
                for( m = 0, mv = std::abs(A[idx]), i = 1; i < idx; i++ )
                {
                    _Tp val = std::abs(A[astep*i+idx]);
                    if( mv < val )
                        mv = val, m = i;
                }
                indC[idx] = m;
            }
        }
    }

    // sort eigenvalues & eigenvectors
    for( k = 0; k < n-1; k++ )
    {
        m = k;
        for( i = k+1; i < n; i++ )
        {
            if( W[m] < W[i] )
                m = i;
        }
        if( k != m )
        {
            std::swap(W[m], W[k]);
            if( V )
                for( i = 0; i < n; i++ )
                    std::swap(V[vstep*m + i], V[vstep*k + i]);
        }
    }

    return true;
}

static bool Jacobi( float* S, size_t sstep, float* e, float* E, size_t estep, int n, uchar* buf )
{
    return JacobiImpl_(S, sstep, e, E, estep, n, buf);
}

static bool Jacobi( double* S, size_t sstep, double* e, double* E, size_t estep, int n, uchar* buf )
{
    return JacobiImpl_(S, sstep, e, E, estep, n, buf);
}


template<typename T> struct VBLAS
{
    int dot(const T*, const T*, int, T*) const { return 0; }
    int givens(T*, T*, int, T, T) const { return 0; }
    int givensx(T*, T*, int, T, T, T*, T*) const { return 0; }
};

#if CV_SIMD128
template<> inline int VBLAS<float>::dot(const float* a, const float* b, int n, float* result) const
{
    if( n < 8 )
        return 0;
    int k = 0;
    v_float32x4 s0 = v_setzero_f32();
    for( ; k <= n - v_float32x4::nlanes; k += v_float32x4::nlanes )
    {
        v_float32x4 a0 = v_load(a + k);
        v_float32x4 b0 = v_load(b + k);

        s0 += a0 * b0;
    }
    *result = v_reduce_sum(s0);
    return k;
}


template<> inline int VBLAS<float>::givens(float* a, float* b, int n, float c, float s) const
{
    if( n < 4 )
        return 0;
    int k = 0;
    v_float32x4 c4 = v_setall_f32(c), s4 = v_setall_f32(s);
    for( ; k <= n - v_float32x4::nlanes; k += v_float32x4::nlanes )
    {
        v_float32x4 a0 = v_load(a + k);
        v_float32x4 b0 = v_load(b + k);
        v_float32x4 t0 = (a0 * c4) + (b0 * s4);
        v_float32x4 t1 = (b0 * c4) - (a0 * s4);
        v_store(a + k, t0);
        v_store(b + k, t1);
    }
    return k;
}


template<> inline int VBLAS<float>::givensx(float* a, float* b, int n, float c, float s,
                                             float* anorm, float* bnorm) const
{
    if( n < 4 )
        return 0;
    int k = 0;
    v_float32x4 c4 = v_setall_f32(c), s4 = v_setall_f32(s);
    v_float32x4 sa = v_setzero_f32(), sb = v_setzero_f32();
    for( ; k <= n - v_float32x4::nlanes; k += v_float32x4::nlanes )
    {
        v_float32x4 a0 = v_load(a + k);
        v_float32x4 b0 = v_load(b + k);
        v_float32x4 t0 = (a0 * c4) + (b0 * s4);
        v_float32x4 t1 = (b0 * c4) - (a0 * s4);
        v_store(a + k, t0);
        v_store(b + k, t1);
        sa += t0 + t0;
        sb += t1 + t1;
    }
    *anorm = v_reduce_sum(sa);
    *bnorm = v_reduce_sum(sb);
    return k;
}

#if CV_SIMD128_64F
template<> inline int VBLAS<double>::dot(const double* a, const double* b, int n, double* result) const
{
    if( n < 4 )
        return 0;
    int k = 0;
    v_float64x2 s0 = v_setzero_f64();
    for( ; k <= n - v_float64x2::nlanes; k += v_float64x2::nlanes )
    {
        v_float64x2 a0 = v_load(a + k);
        v_float64x2 b0 = v_load(b + k);

        s0 += a0 * b0;
    }
    double sbuf[2];
    v_store(sbuf, s0);
    *result = sbuf[0] + sbuf[1];
    return k;
}


template<> inline int VBLAS<double>::givens(double* a, double* b, int n, double c, double s) const
{
    int k = 0;
    v_float64x2 c2 = v_setall_f64(c), s2 = v_setall_f64(s);
    for( ; k <= n - v_float64x2::nlanes; k += v_float64x2::nlanes )
    {
        v_float64x2 a0 = v_load(a + k);
        v_float64x2 b0 = v_load(b + k);
        v_float64x2 t0 = (a0 * c2) + (b0 * s2);
        v_float64x2 t1 = (b0 * c2) - (a0 * s2);
        v_store(a + k, t0);
        v_store(b + k, t1);
    }
    return k;
}


template<> inline int VBLAS<double>::givensx(double* a, double* b, int n, double c, double s,
                                              double* anorm, double* bnorm) const
{
    int k = 0;
    v_float64x2 c2 = v_setall_f64(c), s2 = v_setall_f64(s);
    v_float64x2 sa = v_setzero_f64(), sb = v_setzero_f64();
    for( ; k <= n - v_float64x2::nlanes; k += v_float64x2::nlanes )
    {
        v_float64x2 a0 = v_load(a + k);
        v_float64x2 b0 = v_load(b + k);
        v_float64x2 t0 = (a0 * c2) + (b0 * s2);
        v_float64x2 t1 = (b0 * c2) - (a0 * s2);
        v_store(a + k, t0);
        v_store(b + k, t1);
        sa += t0 * t0;
        sb += t1 * t1;
    }
    double abuf[2], bbuf[2];
    v_store(abuf, sa);
    v_store(bbuf, sb);
    *anorm = abuf[0] + abuf[1];
    *bnorm = bbuf[0] + bbuf[1];
    return k;
}
#endif //CV_SIMD128_64F
#endif //CV_SIMD128

template<typename _Tp> void
JacobiSVDImpl_(_Tp* At, size_t astep, _Tp* _W, _Tp* Vt, size_t vstep,
               int m, int n, int n1, double minval, _Tp eps)
{
    VBLAS<_Tp> vblas;
    AutoBuffer<double> Wbuf(n);
    double* W = Wbuf;
    int i, j, k, iter, max_iter = std::max(m, 30);
    _Tp c, s;
    double sd;
    astep /= sizeof(At[0]);
    vstep /= sizeof(Vt[0]);

    for( i = 0; i < n; i++ )
    {
        for( k = 0, sd = 0; k < m; k++ )
        {
            _Tp t = At[i*astep + k];
            sd += (double)t*t;
        }
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

                for( k = 0; k < m; k++ )
                    p += (double)Ai[k]*Aj[k];

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
                for( k = 0; k < m; k++ )
                {
                    _Tp t0 = c*Ai[k] + s*Aj[k];
                    _Tp t1 = -s*Ai[k] + c*Aj[k];
                    Ai[k] = t0; Aj[k] = t1;

                    a += (double)t0*t0; b += (double)t1*t1;
                }
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
        for( k = 0, sd = 0; k < m; k++ )
        {
            _Tp t = At[i*astep + k];
            sd += (double)t*t;
        }
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


static void JacobiSVD(float* At, size_t astep, float* W, float* Vt, size_t vstep, int m, int n, int n1=-1)
{
    hal::SVD32f(At, astep, W, NULL, astep, Vt, vstep, m, n, n1);
}

static void JacobiSVD(double* At, size_t astep, double* W, double* Vt, size_t vstep, int m, int n, int n1=-1)
{
    hal::SVD64f(At, astep, W, NULL, astep, Vt, vstep, m, n, n1);
}

template <typename fptype> static inline int
decodeSVDParameters(const fptype* U, const fptype* Vt, int m, int n, int n1)
{
    int halSVDFlag = 0;
    if(Vt == NULL)
        halSVDFlag = CV_HAL_SVD_NO_UV;
    else if(n1 <= 0 || n1 == n)
    {
        halSVDFlag = CV_HAL_SVD_SHORT_UV;
        if(U == NULL)
            halSVDFlag |= CV_HAL_SVD_MODIFY_A;
    }
    else if(n1 == m)
    {
        halSVDFlag = CV_HAL_SVD_FULL_UV;
        if(U == NULL)
            halSVDFlag |= CV_HAL_SVD_MODIFY_A;
    }
    return halSVDFlag;
}

void hal::SVD32f(float* At, size_t astep, float* W, float* U, size_t ustep, float* Vt, size_t vstep, int m, int n, int n1)
{
    CALL_HAL(SVD32f, cv_hal_SVD32f, At, astep, W, U, ustep, Vt, vstep, m, n, decodeSVDParameters(U, Vt, m, n, n1))
    JacobiSVDImpl_(At, astep, W, Vt, vstep, m, n, !Vt ? 0 : n1 < 0 ? n : n1, FLT_MIN, FLT_EPSILON*2);
}

void hal::SVD64f(double* At, size_t astep, double* W, double* U, size_t ustep, double* Vt, size_t vstep, int m, int n, int n1)
{
    CALL_HAL(SVD64f, cv_hal_SVD64f, At, astep, W, U, ustep, Vt, vstep, m, n, decodeSVDParameters(U, Vt, m, n, n1))
    JacobiSVDImpl_(At, astep, W, Vt, vstep, m, n, !Vt ? 0 : n1 < 0 ? n : n1, DBL_MIN, DBL_EPSILON*10);
}

/* y[0:m,0:n] += diag(a[0:1,0:m]) * x[0:m,0:n] */
template<typename T1, typename T2, typename T3> static void
MatrAXPY( int m, int n, const T1* x, int dx,
         const T2* a, int inca, T3* y, int dy )
{
    int i;
    for( i = 0; i < m; i++, x += dx, y += dy )
    {
        T2 s = a[i*inca];
        int j = 0;
         #if CV_ENABLE_UNROLLED
        for(; j <= n - 4; j += 4 )
        {
            T3 t0 = (T3)(y[j]   + s*x[j]);
            T3 t1 = (T3)(y[j+1] + s*x[j+1]);
            y[j]   = t0;
            y[j+1] = t1;
            t0 = (T3)(y[j+2] + s*x[j+2]);
            t1 = (T3)(y[j+3] + s*x[j+3]);
            y[j+2] = t0;
            y[j+3] = t1;
        }
        #endif
        for( ; j < n; j++ )
            y[j] = (T3)(y[j] + s*x[j]);
    }
}

template<typename T> static void
SVBkSbImpl_( int m, int n, const T* w, int incw,
       const T* u, int ldu, bool uT,
       const T* v, int ldv, bool vT,
       const T* b, int ldb, int nb,
       T* x, int ldx, double* buffer, T eps )
{
    double threshold = 0;
    int udelta0 = uT ? ldu : 1, udelta1 = uT ? 1 : ldu;
    int vdelta0 = vT ? ldv : 1, vdelta1 = vT ? 1 : ldv;
    int i, j, nm = std::min(m, n);

    if( !b )
        nb = m;

    for( i = 0; i < n; i++ )
        for( j = 0; j < nb; j++ )
            x[i*ldx + j] = 0;

    for( i = 0; i < nm; i++ )
        threshold += w[i*incw];
    threshold *= eps;

    // v * inv(w) * uT * b
    for( i = 0; i < nm; i++, u += udelta0, v += vdelta0 )
    {
        double wi = w[i*incw];
        if( (double)std::abs(wi) <= threshold )
            continue;
        wi = 1/wi;

        if( nb == 1 )
        {
            double s = 0;
            if( b )
                for( j = 0; j < m; j++ )
                    s += u[j*udelta1]*b[j*ldb];
            else
                s = u[0];
            s *= wi;

            for( j = 0; j < n; j++ )
                x[j*ldx] = (T)(x[j*ldx] + s*v[j*vdelta1]);
        }
        else
        {
            if( b )
            {
                for( j = 0; j < nb; j++ )
                    buffer[j] = 0;
                MatrAXPY( m, nb, b, ldb, u, udelta1, buffer, 0 );
                for( j = 0; j < nb; j++ )
                    buffer[j] *= wi;
            }
            else
            {
                for( j = 0; j < nb; j++ )
                    buffer[j] = u[j*udelta1]*wi;
            }
            MatrAXPY( n, nb, buffer, 0, v, vdelta1, x, ldx );
        }
    }
}

static void
SVBkSb( int m, int n, const float* w, size_t wstep,
        const float* u, size_t ustep, bool uT,
        const float* v, size_t vstep, bool vT,
        const float* b, size_t bstep, int nb,
        float* x, size_t xstep, uchar* buffer )
{
    SVBkSbImpl_(m, n, w, wstep ? (int)(wstep/sizeof(w[0])) : 1,
                u, (int)(ustep/sizeof(u[0])), uT,
                v, (int)(vstep/sizeof(v[0])), vT,
                b, (int)(bstep/sizeof(b[0])), nb,
                x, (int)(xstep/sizeof(x[0])),
                (double*)alignPtr(buffer, sizeof(double)), (float)(DBL_EPSILON*2) );
}

static void
SVBkSb( int m, int n, const double* w, size_t wstep,
       const double* u, size_t ustep, bool uT,
       const double* v, size_t vstep, bool vT,
       const double* b, size_t bstep, int nb,
       double* x, size_t xstep, uchar* buffer )
{
    SVBkSbImpl_(m, n, w, wstep ? (int)(wstep/sizeof(w[0])) : 1,
                u, (int)(ustep/sizeof(u[0])), uT,
                v, (int)(vstep/sizeof(v[0])), vT,
                b, (int)(bstep/sizeof(b[0])), nb,
                x, (int)(xstep/sizeof(x[0])),
                (double*)alignPtr(buffer, sizeof(double)), DBL_EPSILON*2 );
}

}

/****************************************************************************************\
*                                 Determinant of the matrix                              *
\****************************************************************************************/

#define det2(m)   ((double)m(0,0)*m(1,1) - (double)m(0,1)*m(1,0))
#define det3(m)   (m(0,0)*((double)m(1,1)*m(2,2) - (double)m(1,2)*m(2,1)) -  \
                   m(0,1)*((double)m(1,0)*m(2,2) - (double)m(1,2)*m(2,0)) +  \
                   m(0,2)*((double)m(1,0)*m(2,1) - (double)m(1,1)*m(2,0)))

double cv::determinant( InputArray _mat )
{
    CV_INSTRUMENT_REGION()

    Mat mat = _mat.getMat();
    double result = 0;
    int type = mat.type(), rows = mat.rows;
    size_t step = mat.step;
    const uchar* m = mat.ptr();

    CV_Assert( !mat.empty() );
    CV_Assert( mat.rows == mat.cols && (type == CV_32F || type == CV_64F));

    #define Mf(y, x) ((float*)(m + y*step))[x]
    #define Md(y, x) ((double*)(m + y*step))[x]

    if( type == CV_32F )
    {
        if( rows == 2 )
            result = det2(Mf);
        else if( rows == 3 )
            result = det3(Mf);
        else if( rows == 1 )
            result = Mf(0,0);
        else
        {
            size_t bufSize = rows*rows*sizeof(float);
            AutoBuffer<uchar> buffer(bufSize);
            Mat a(rows, rows, CV_32F, (uchar*)buffer);
            mat.copyTo(a);

            result = hal::LU32f(a.ptr<float>(), a.step, rows, 0, 0, 0);
            if( result )
            {
                for( int i = 0; i < rows; i++ )
                    result *= a.at<float>(i,i);
            }
        }
    }
    else
    {
        if( rows == 2 )
            result = det2(Md);
        else if( rows == 3 )
            result = det3(Md);
        else if( rows == 1 )
            result = Md(0,0);
        else
        {
            size_t bufSize = rows*rows*sizeof(double);
            AutoBuffer<uchar> buffer(bufSize);
            Mat a(rows, rows, CV_64F, (uchar*)buffer);
            mat.copyTo(a);

            result = hal::LU64f(a.ptr<double>(), a.step, rows, 0, 0, 0);
            if( result )
            {
                for( int i = 0; i < rows; i++ )
                    result *= a.at<double>(i,i);
            }
        }
    }

    #undef Mf
    #undef Md

    return result;
}

/****************************************************************************************\
*                          Inverse (or pseudo-inverse) of a matrix                       *
\****************************************************************************************/

#define Sf( y, x ) ((float*)(srcdata + y*srcstep))[x]
#define Sd( y, x ) ((double*)(srcdata + y*srcstep))[x]
#define Df( y, x ) ((float*)(dstdata + y*dststep))[x]
#define Dd( y, x ) ((double*)(dstdata + y*dststep))[x]

double cv::invert( InputArray _src, OutputArray _dst, int method )
{
    CV_INSTRUMENT_REGION()

    bool result = false;
    Mat src = _src.getMat();
    int type = src.type();

    CV_Assert(type == CV_32F || type == CV_64F);

    size_t esz = CV_ELEM_SIZE(type);
    int m = src.rows, n = src.cols;

    if( method == DECOMP_SVD )
    {
        int nm = std::min(m, n);

        AutoBuffer<uchar> _buf((m*nm + nm + nm*n)*esz + sizeof(double));
        uchar* buf = alignPtr((uchar*)_buf, (int)esz);
        Mat u(m, nm, type, buf);
        Mat w(nm, 1, type, u.ptr() + m*nm*esz);
        Mat vt(nm, n, type, w.ptr() + nm*esz);

        SVD::compute(src, w, u, vt);
        SVD::backSubst(w, u, vt, Mat(), _dst);
        return type == CV_32F ?
            (w.ptr<float>()[0] >= FLT_EPSILON ?
             w.ptr<float>()[n-1]/w.ptr<float>()[0] : 0) :
            (w.ptr<double>()[0] >= DBL_EPSILON ?
             w.ptr<double>()[n-1]/w.ptr<double>()[0] : 0);
    }

    CV_Assert( m == n );

    if( method == DECOMP_EIG )
    {
        AutoBuffer<uchar> _buf((n*n*2 + n)*esz + sizeof(double));
        uchar* buf = alignPtr((uchar*)_buf, (int)esz);
        Mat u(n, n, type, buf);
        Mat w(n, 1, type, u.ptr() + n*n*esz);
        Mat vt(n, n, type, w.ptr() + n*esz);

        eigen(src, w, vt);
        transpose(vt, u);
        SVD::backSubst(w, u, vt, Mat(), _dst);
        return type == CV_32F ?
        (w.ptr<float>()[0] >= FLT_EPSILON ?
         w.ptr<float>()[n-1]/w.ptr<float>()[0] : 0) :
        (w.ptr<double>()[0] >= DBL_EPSILON ?
         w.ptr<double>()[n-1]/w.ptr<double>()[0] : 0);
    }

    CV_Assert( method == DECOMP_LU || method == DECOMP_CHOLESKY );

    _dst.create( n, n, type );
    Mat dst = _dst.getMat();

    if( n <= 3 )
    {
        const uchar* srcdata = src.ptr();
        uchar* dstdata = dst.ptr();
        size_t srcstep = src.step;
        size_t dststep = dst.step;

        if( n == 2 )
        {
            if( type == CV_32FC1 )
            {
                double d = det2(Sf);
                if( d != 0. )
                {
                    result = true;
                    d = 1./d;

                    #if CV_SSE2
                    if(USE_SSE2)
                    {
                        __m128 zero = _mm_setzero_ps();
                        __m128 t0 = _mm_loadl_pi(zero, (const __m64*)srcdata); //t0 = sf(0,0) sf(0,1)
                        __m128 t1 = _mm_loadh_pi(zero, (const __m64*)(srcdata+srcstep)); //t1 = sf(1,0) sf(1,1)
                        __m128 s0 = _mm_or_ps(t0, t1);
                        __m128 det =_mm_set1_ps((float)d);
                        s0 =  _mm_mul_ps(s0, det);
                        static const uchar CV_DECL_ALIGNED(16) inv[16] = {0,0,0,0,0,0,0,0x80,0,0,0,0x80,0,0,0,0};
                        __m128 pattern = _mm_load_ps((const float*)inv);
                        s0 = _mm_xor_ps(s0, pattern);//==-1*s0
                        s0 = _mm_shuffle_ps(s0, s0, _MM_SHUFFLE(0,2,1,3));
                        _mm_storel_pi((__m64*)dstdata, s0);
                        _mm_storeh_pi((__m64*)((float*)(dstdata+dststep)), s0);
                    }
                    else
                    #endif
                    {
                        double t0, t1;
                        t0 = Sf(0,0)*d;
                        t1 = Sf(1,1)*d;
                        Df(1,1) = (float)t0;
                        Df(0,0) = (float)t1;
                        t0 = -Sf(0,1)*d;
                        t1 = -Sf(1,0)*d;
                        Df(0,1) = (float)t0;
                        Df(1,0) = (float)t1;
                    }

                }
            }
            else
            {
                double d = det2(Sd);
                if( d != 0. )
                {
                    result = true;
                    d = 1./d;
                    #if CV_SSE2
                    if(USE_SSE2)
                    {
                        __m128d s0 = _mm_loadu_pd((const double*)srcdata); //s0 = sf(0,0) sf(0,1)
                        __m128d s1 = _mm_loadu_pd ((const double*)(srcdata+srcstep));//s1 = sf(1,0) sf(1,1)
                        __m128d sm = _mm_unpacklo_pd(s0, _mm_load_sd((const double*)(srcdata+srcstep)+1)); //sm = sf(0,0) sf(1,1) - main diagonal
                        __m128d ss = _mm_shuffle_pd(s0, s1, _MM_SHUFFLE2(0,1)); //ss = sf(0,1) sf(1,0) - secondary diagonal
                        __m128d det = _mm_load1_pd((const double*)&d);
                        sm =  _mm_mul_pd(sm, det);

                        static const uchar CV_DECL_ALIGNED(16) inv[8] = {0,0,0,0,0,0,0,0x80};
                        __m128d pattern = _mm_load1_pd((double*)inv);
                        ss = _mm_mul_pd(ss, det);
                        ss = _mm_xor_pd(ss, pattern);//==-1*ss

                        s0 = _mm_shuffle_pd(sm, ss, _MM_SHUFFLE2(0,1));
                        s1 = _mm_shuffle_pd(ss, sm, _MM_SHUFFLE2(0,1));
                        _mm_storeu_pd((double*)dstdata, s0);
                        _mm_storeu_pd((double*)(dstdata+dststep), s1);
                    }
                    else
                    #endif
                    {
                        double t0, t1;
                        t0 = Sd(0,0)*d;
                        t1 = Sd(1,1)*d;
                        Dd(1,1) = t0;
                        Dd(0,0) = t1;
                        t0 = -Sd(0,1)*d;
                        t1 = -Sd(1,0)*d;
                        Dd(0,1) = t0;
                        Dd(1,0) = t1;
                    }
                }
            }
        }
        else if( n == 3 )
        {
            if( type == CV_32FC1 )
            {
                double d = det3(Sf);

                if( d != 0. )
                {
                    double t[12];

                    result = true;
                    d = 1./d;
                    t[0] = (((double)Sf(1,1) * Sf(2,2) - (double)Sf(1,2) * Sf(2,1)) * d);
                    t[1] = (((double)Sf(0,2) * Sf(2,1) - (double)Sf(0,1) * Sf(2,2)) * d);
                    t[2] = (((double)Sf(0,1) * Sf(1,2) - (double)Sf(0,2) * Sf(1,1)) * d);

                    t[3] = (((double)Sf(1,2) * Sf(2,0) - (double)Sf(1,0) * Sf(2,2)) * d);
                    t[4] = (((double)Sf(0,0) * Sf(2,2) - (double)Sf(0,2) * Sf(2,0)) * d);
                    t[5] = (((double)Sf(0,2) * Sf(1,0) - (double)Sf(0,0) * Sf(1,2)) * d);

                    t[6] = (((double)Sf(1,0) * Sf(2,1) - (double)Sf(1,1) * Sf(2,0)) * d);
                    t[7] = (((double)Sf(0,1) * Sf(2,0) - (double)Sf(0,0) * Sf(2,1)) * d);
                    t[8] = (((double)Sf(0,0) * Sf(1,1) - (double)Sf(0,1) * Sf(1,0)) * d);

                    Df(0,0) = (float)t[0]; Df(0,1) = (float)t[1]; Df(0,2) = (float)t[2];
                    Df(1,0) = (float)t[3]; Df(1,1) = (float)t[4]; Df(1,2) = (float)t[5];
                    Df(2,0) = (float)t[6]; Df(2,1) = (float)t[7]; Df(2,2) = (float)t[8];
                }
            }
            else
            {
                double d = det3(Sd);
                if( d != 0. )
                {
                    result = true;
                    d = 1./d;
                    double t[9];

                    t[0] = (Sd(1,1) * Sd(2,2) - Sd(1,2) * Sd(2,1)) * d;
                    t[1] = (Sd(0,2) * Sd(2,1) - Sd(0,1) * Sd(2,2)) * d;
                    t[2] = (Sd(0,1) * Sd(1,2) - Sd(0,2) * Sd(1,1)) * d;

                    t[3] = (Sd(1,2) * Sd(2,0) - Sd(1,0) * Sd(2,2)) * d;
                    t[4] = (Sd(0,0) * Sd(2,2) - Sd(0,2) * Sd(2,0)) * d;
                    t[5] = (Sd(0,2) * Sd(1,0) - Sd(0,0) * Sd(1,2)) * d;

                    t[6] = (Sd(1,0) * Sd(2,1) - Sd(1,1) * Sd(2,0)) * d;
                    t[7] = (Sd(0,1) * Sd(2,0) - Sd(0,0) * Sd(2,1)) * d;
                    t[8] = (Sd(0,0) * Sd(1,1) - Sd(0,1) * Sd(1,0)) * d;

                    Dd(0,0) = t[0]; Dd(0,1) = t[1]; Dd(0,2) = t[2];
                    Dd(1,0) = t[3]; Dd(1,1) = t[4]; Dd(1,2) = t[5];
                    Dd(2,0) = t[6]; Dd(2,1) = t[7]; Dd(2,2) = t[8];
                }
            }
        }
        else
        {
            assert( n == 1 );

            if( type == CV_32FC1 )
            {
                double d = Sf(0,0);
                if( d != 0. )
                {
                    result = true;
                    Df(0,0) = (float)(1./d);
                }
            }
            else
            {
                double d = Sd(0,0);
                if( d != 0. )
                {
                    result = true;
                    Dd(0,0) = 1./d;
                }
            }
        }
        if( !result )
            dst = Scalar(0);
        return result;
    }

   int elem_size = CV_ELEM_SIZE(type);
    AutoBuffer<uchar> buf(n*n*elem_size);
    Mat src1(n, n, type, (uchar*)buf);
    src.copyTo(src1);
    setIdentity(dst);

    if( method == DECOMP_LU && type == CV_32F )
        result = hal::LU32f(src1.ptr<float>(), src1.step, n, dst.ptr<float>(), dst.step, n) != 0;
    else if( method == DECOMP_LU && type == CV_64F )
        result = hal::LU64f(src1.ptr<double>(), src1.step, n, dst.ptr<double>(), dst.step, n) != 0;
    else if( method == DECOMP_CHOLESKY && type == CV_32F )
        result = hal::Cholesky32f(src1.ptr<float>(), src1.step, n, dst.ptr<float>(), dst.step, n);
    else
        result = hal::Cholesky64f(src1.ptr<double>(), src1.step, n, dst.ptr<double>(), dst.step, n);

    if( !result )
        dst = Scalar(0);

    return result;
}



/****************************************************************************************\
*                              Solving a linear system                                   *
\****************************************************************************************/

bool cv::solve( InputArray _src, InputArray _src2arg, OutputArray _dst, int method )
{
    CV_INSTRUMENT_REGION()

    bool result = true;
    Mat src = _src.getMat(), _src2 = _src2arg.getMat();
    int type = src.type();
    bool is_normal = (method & DECOMP_NORMAL) != 0;

    CV_Assert( type == _src2.type() && (type == CV_32F || type == CV_64F) );

    method &= ~DECOMP_NORMAL;
    CV_Assert( (method != DECOMP_LU && method != DECOMP_CHOLESKY) ||
        is_normal || src.rows == src.cols );

    // check case of a single equation and small matrix
    if( (method == DECOMP_LU || method == DECOMP_CHOLESKY) && !is_normal &&
        src.rows <= 3 && src.rows == src.cols && _src2.cols == 1 )
    {
        _dst.create( src.cols, _src2.cols, src.type() );
        Mat dst = _dst.getMat();

        #define bf(y) ((float*)(bdata + y*src2step))[0]
        #define bd(y) ((double*)(bdata + y*src2step))[0]

        const uchar* srcdata = src.ptr();
        const uchar* bdata = _src2.ptr();
        uchar* dstdata = dst.ptr();
        size_t srcstep = src.step;
        size_t src2step = _src2.step;
        size_t dststep = dst.step;

        if( src.rows == 2 )
        {
            if( type == CV_32FC1 )
            {
                double d = det2(Sf);
                if( d != 0. )
                {
                    double t;
                    d = 1./d;
                    t = (float)(((double)bf(0)*Sf(1,1) - (double)bf(1)*Sf(0,1))*d);
                    Df(1,0) = (float)(((double)bf(1)*Sf(0,0) - (double)bf(0)*Sf(1,0))*d);
                    Df(0,0) = (float)t;
                }
                else
                    result = false;
            }
            else
            {
                double d = det2(Sd);
                if( d != 0. )
                {
                    double t;
                    d = 1./d;
                    t = (bd(0)*Sd(1,1) - bd(1)*Sd(0,1))*d;
                    Dd(1,0) = (bd(1)*Sd(0,0) - bd(0)*Sd(1,0))*d;
                    Dd(0,0) = t;
                }
                else
                    result = false;
            }
        }
        else if( src.rows == 3 )
        {
            if( type == CV_32FC1 )
            {
                double d = det3(Sf);
                if( d != 0. )
                {
                    float t[3];
                    d = 1./d;

                    t[0] = (float)(d*
                           (bf(0)*((double)Sf(1,1)*Sf(2,2) - (double)Sf(1,2)*Sf(2,1)) -
                            Sf(0,1)*((double)bf(1)*Sf(2,2) - (double)Sf(1,2)*bf(2)) +
                            Sf(0,2)*((double)bf(1)*Sf(2,1) - (double)Sf(1,1)*bf(2))));

                    t[1] = (float)(d*
                           (Sf(0,0)*(double)(bf(1)*Sf(2,2) - (double)Sf(1,2)*bf(2)) -
                            bf(0)*((double)Sf(1,0)*Sf(2,2) - (double)Sf(1,2)*Sf(2,0)) +
                            Sf(0,2)*((double)Sf(1,0)*bf(2) - (double)bf(1)*Sf(2,0))));

                    t[2] = (float)(d*
                           (Sf(0,0)*((double)Sf(1,1)*bf(2) - (double)bf(1)*Sf(2,1)) -
                            Sf(0,1)*((double)Sf(1,0)*bf(2) - (double)bf(1)*Sf(2,0)) +
                            bf(0)*((double)Sf(1,0)*Sf(2,1) - (double)Sf(1,1)*Sf(2,0))));

                    Df(0,0) = t[0];
                    Df(1,0) = t[1];
                    Df(2,0) = t[2];
                }
                else
                    result = false;
            }
            else
            {
                double d = det3(Sd);
                if( d != 0. )
                {
                    double t[9];

                    d = 1./d;

                    t[0] = ((Sd(1,1) * Sd(2,2) - Sd(1,2) * Sd(2,1))*bd(0) +
                            (Sd(0,2) * Sd(2,1) - Sd(0,1) * Sd(2,2))*bd(1) +
                            (Sd(0,1) * Sd(1,2) - Sd(0,2) * Sd(1,1))*bd(2))*d;

                    t[1] = ((Sd(1,2) * Sd(2,0) - Sd(1,0) * Sd(2,2))*bd(0) +
                            (Sd(0,0) * Sd(2,2) - Sd(0,2) * Sd(2,0))*bd(1) +
                            (Sd(0,2) * Sd(1,0) - Sd(0,0) * Sd(1,2))*bd(2))*d;

                    t[2] = ((Sd(1,0) * Sd(2,1) - Sd(1,1) * Sd(2,0))*bd(0) +
                            (Sd(0,1) * Sd(2,0) - Sd(0,0) * Sd(2,1))*bd(1) +
                            (Sd(0,0) * Sd(1,1) - Sd(0,1) * Sd(1,0))*bd(2))*d;

                    Dd(0,0) = t[0];
                    Dd(1,0) = t[1];
                    Dd(2,0) = t[2];
                }
                else
                    result = false;
            }
        }
        else
        {
            assert( src.rows == 1 );

            if( type == CV_32FC1 )
            {
                double d = Sf(0,0);
                if( d != 0. )
                    Df(0,0) = (float)(bf(0)/d);
                else
                    result = false;
            }
            else
            {
                double d = Sd(0,0);
                if( d != 0. )
                    Dd(0,0) = (bd(0)/d);
                else
                    result = false;
            }
        }
        return result;
    }

    int m = src.rows, m_ = m, n = src.cols, nb = _src2.cols;
    size_t esz = CV_ELEM_SIZE(type), bufsize = 0;
    size_t vstep = alignSize(n*esz, 16);
    size_t astep = method == DECOMP_SVD && !is_normal ? alignSize(m*esz, 16) : vstep;
    AutoBuffer<uchar> buffer;

    Mat src2 = _src2;
    _dst.create( src.cols, src2.cols, src.type() );
    Mat dst = _dst.getMat();

    if( m < n )
        CV_Error(CV_StsBadArg, "The function can not solve under-determined linear systems" );

    if( m == n )
        is_normal = false;
    else if( is_normal )
    {
        m_ = n;
        if( method == DECOMP_SVD )
            method = DECOMP_EIG;
    }

    size_t asize = astep*(method == DECOMP_SVD || is_normal ? n : m);
    bufsize += asize + 32;

    if( is_normal )
        bufsize += n*nb*esz;
    if( method == DECOMP_SVD || method == DECOMP_EIG )
        bufsize += n*5*esz + n*vstep + nb*sizeof(double) + 32;

    buffer.allocate(bufsize);
    uchar* ptr = alignPtr((uchar*)buffer, 16);

    Mat a(m_, n, type, ptr, astep);

    if( is_normal )
        mulTransposed(src, a, true);
    else if( method != DECOMP_SVD )
        src.copyTo(a);
    else
    {
        a = Mat(n, m_, type, ptr, astep);
        transpose(src, a);
    }
    ptr += asize;

    if( !is_normal )
    {
        if( method == DECOMP_LU || method == DECOMP_CHOLESKY )
            src2.copyTo(dst);
    }
    else
    {
        // a'*b
        if( method == DECOMP_LU || method == DECOMP_CHOLESKY )
            gemm( src, src2, 1, Mat(), 0, dst, GEMM_1_T );
        else
        {
            Mat tmp(n, nb, type, ptr);
            ptr += n*nb*esz;
            gemm( src, src2, 1, Mat(), 0, tmp, GEMM_1_T );
            src2 = tmp;
        }
    }

    if( method == DECOMP_LU )
    {
        if( type == CV_32F )
            result = hal::LU32f(a.ptr<float>(), a.step, n, dst.ptr<float>(), dst.step, nb) != 0;
        else
            result = hal::LU64f(a.ptr<double>(), a.step, n, dst.ptr<double>(), dst.step, nb) != 0;
    }
    else if( method == DECOMP_CHOLESKY )
    {
        if( type == CV_32F )
            result = hal::Cholesky32f(a.ptr<float>(), a.step, n, dst.ptr<float>(), dst.step, nb);
        else
            result = hal::Cholesky64f(a.ptr<double>(), a.step, n, dst.ptr<double>(), dst.step, nb);
    }
    else if( method == DECOMP_QR )
    {
        Mat rhsMat;
        if( is_normal || m == n )
        {
            src2.copyTo(dst);
            rhsMat = dst;
        }
        else
        {
            rhsMat = Mat(m, nb, type);
            src2.copyTo(rhsMat);
        }

        if( type == CV_32F )
            result = hal::QR32f(a.ptr<float>(), a.step, a.rows, a.cols, rhsMat.cols, rhsMat.ptr<float>(), rhsMat.step, NULL) != 0;
        else
            result = hal::QR64f(a.ptr<double>(), a.step, a.rows, a.cols, rhsMat.cols, rhsMat.ptr<double>(), rhsMat.step, NULL) != 0;

        if (rhsMat.rows != dst.rows)
            rhsMat.rowRange(0, dst.rows).copyTo(dst);
    }
    else
    {
        ptr = alignPtr(ptr, 16);
        Mat v(n, n, type, ptr, vstep), w(n, 1, type, ptr + vstep*n), u;
        ptr += n*(vstep + esz);

        if( method == DECOMP_EIG )
        {
            if( type == CV_32F )
                Jacobi(a.ptr<float>(), a.step, w.ptr<float>(), v.ptr<float>(), v.step, n, ptr);
            else
                Jacobi(a.ptr<double>(), a.step, w.ptr<double>(), v.ptr<double>(), v.step, n, ptr);
            u = v;
        }
        else
        {
            if( type == CV_32F )
                JacobiSVD(a.ptr<float>(), a.step, w.ptr<float>(), v.ptr<float>(), v.step, m_, n);
            else
                JacobiSVD(a.ptr<double>(), a.step, w.ptr<double>(), v.ptr<double>(), v.step, m_, n);
            u = a;
        }

        if( type == CV_32F )
        {
            SVBkSb(m_, n, w.ptr<float>(), 0, u.ptr<float>(), u.step, true,
                   v.ptr<float>(), v.step, true, src2.ptr<float>(),
                   src2.step, nb, dst.ptr<float>(), dst.step, ptr);
        }
        else
        {
            SVBkSb(m_, n, w.ptr<double>(), 0, u.ptr<double>(), u.step, true,
                   v.ptr<double>(), v.step, true, src2.ptr<double>(),
                   src2.step, nb, dst.ptr<double>(), dst.step, ptr);
        }
        result = true;
    }

    if( !result )
        dst = Scalar(0);

    return result;
}


/////////////////// finding eigenvalues and eigenvectors of a symmetric matrix ///////////////

bool cv::eigen( InputArray _src, OutputArray _evals, OutputArray _evects )
{
    CV_INSTRUMENT_REGION()

    Mat src = _src.getMat();
    int type = src.type();
    int n = src.rows;

    CV_Assert( src.rows == src.cols );
    CV_Assert (type == CV_32F || type == CV_64F);

    Mat v;
    if( _evects.needed() )
    {
        _evects.create(n, n, type);
        v = _evects.getMat();
    }

    size_t elemSize = src.elemSize(), astep = alignSize(n*elemSize, 16);
    AutoBuffer<uchar> buf(n*astep + n*5*elemSize + 32);
    uchar* ptr = alignPtr((uchar*)buf, 16);
    Mat a(n, n, type, ptr, astep), w(n, 1, type, ptr + astep*n);
    ptr += astep*n + elemSize*n;
    src.copyTo(a);
    bool ok = type == CV_32F ?
        Jacobi(a.ptr<float>(), a.step, w.ptr<float>(), v.ptr<float>(), v.step, n, ptr) :
        Jacobi(a.ptr<double>(), a.step, w.ptr<double>(), v.ptr<double>(), v.step, n, ptr);

    w.copyTo(_evals);
    return ok;
}

namespace cv
{

static void _SVDcompute( InputArray _aarr, OutputArray _w,
                         OutputArray _u, OutputArray _vt, int flags )
{
    Mat src = _aarr.getMat();
    int m = src.rows, n = src.cols;
    int type = src.type();
    bool compute_uv = _u.needed() || _vt.needed();
    bool full_uv = (flags & SVD::FULL_UV) != 0;

    CV_Assert( type == CV_32F || type == CV_64F );

    if( flags & SVD::NO_UV )
    {
        _u.release();
        _vt.release();
        compute_uv = full_uv = false;
    }

    bool at = false;
    if( m < n )
    {
        std::swap(m, n);
        at = true;
    }

    int urows = full_uv ? m : n;
    size_t esz = src.elemSize(), astep = alignSize(m*esz, 16), vstep = alignSize(n*esz, 16);
    AutoBuffer<uchar> _buf(urows*astep + n*vstep + n*esz + 32);
    uchar* buf = alignPtr((uchar*)_buf, 16);
    Mat temp_a(n, m, type, buf, astep);
    Mat temp_w(n, 1, type, buf + urows*astep);
    Mat temp_u(urows, m, type, buf, astep), temp_v;

    if( compute_uv )
        temp_v = Mat(n, n, type, alignPtr(buf + urows*astep + n*esz, 16), vstep);

    if( urows > n )
        temp_u = Scalar::all(0);

    if( !at )
        transpose(src, temp_a);
    else
        src.copyTo(temp_a);

    if( type == CV_32F )
    {
        JacobiSVD(temp_a.ptr<float>(), temp_u.step, temp_w.ptr<float>(),
              temp_v.ptr<float>(), temp_v.step, m, n, compute_uv ? urows : 0);
    }
    else
    {
        JacobiSVD(temp_a.ptr<double>(), temp_u.step, temp_w.ptr<double>(),
              temp_v.ptr<double>(), temp_v.step, m, n, compute_uv ? urows : 0);
    }
    temp_w.copyTo(_w);
    if( compute_uv )
    {
        if( !at )
        {
            if( _u.needed() )
                transpose(temp_u, _u);
            if( _vt.needed() )
                temp_v.copyTo(_vt);
        }
        else
        {
            if( _u.needed() )
                transpose(temp_v, _u);
            if( _vt.needed() )
                temp_u.copyTo(_vt);
        }
    }
}


void SVD::compute( InputArray a, OutputArray w, OutputArray u, OutputArray vt, int flags )
{
    CV_INSTRUMENT_REGION()

    _SVDcompute(a, w, u, vt, flags);
}

void SVD::compute( InputArray a, OutputArray w, int flags )
{
    CV_INSTRUMENT_REGION()

    _SVDcompute(a, w, noArray(), noArray(), flags);
}

void SVD::backSubst( InputArray _w, InputArray _u, InputArray _vt,
                     InputArray _rhs, OutputArray _dst )
{
    Mat w = _w.getMat(), u = _u.getMat(), vt = _vt.getMat(), rhs = _rhs.getMat();
    int type = w.type(), esz = (int)w.elemSize();
    int m = u.rows, n = vt.cols, nb = rhs.data ? rhs.cols : m, nm = std::min(m, n);
    size_t wstep = w.rows == 1 ? (size_t)esz : w.cols == 1 ? (size_t)w.step : (size_t)w.step + esz;
    AutoBuffer<uchar> buffer(nb*sizeof(double) + 16);
    CV_Assert( w.type() == u.type() && u.type() == vt.type() && u.data && vt.data && w.data );
    CV_Assert( u.cols >= nm && vt.rows >= nm &&
              (w.size() == Size(nm, 1) || w.size() == Size(1, nm) || w.size() == Size(vt.rows, u.cols)) );
    CV_Assert( rhs.data == 0 || (rhs.type() == type && rhs.rows == m) );

    _dst.create( n, nb, type );
    Mat dst = _dst.getMat();
    if( type == CV_32F )
        SVBkSb(m, n, w.ptr<float>(), wstep, u.ptr<float>(), u.step, false,
               vt.ptr<float>(), vt.step, true, rhs.ptr<float>(), rhs.step, nb,
               dst.ptr<float>(), dst.step, buffer);
    else if( type == CV_64F )
        SVBkSb(m, n, w.ptr<double>(), wstep, u.ptr<double>(), u.step, false,
               vt.ptr<double>(), vt.step, true, rhs.ptr<double>(), rhs.step, nb,
               dst.ptr<double>(), dst.step, buffer);
    else
        CV_Error( CV_StsUnsupportedFormat, "" );
}


SVD& SVD::operator ()(InputArray a, int flags)
{
    _SVDcompute(a, w, u, vt, flags);
    return *this;
}


void SVD::backSubst( InputArray rhs, OutputArray dst ) const
{
    backSubst( w, u, vt, rhs, dst );
}

}


void cv::SVDecomp(InputArray src, OutputArray w, OutputArray u, OutputArray vt, int flags)
{
    CV_INSTRUMENT_REGION()

    SVD::compute(src, w, u, vt, flags);
}

void cv::SVBackSubst(InputArray w, InputArray u, InputArray vt, InputArray rhs, OutputArray dst)
{
    CV_INSTRUMENT_REGION()

    SVD::backSubst(w, u, vt, rhs, dst);
}


CV_IMPL double
cvDet( const CvArr* arr )
{
    if( CV_IS_MAT(arr) && ((CvMat*)arr)->rows <= 3 )
    {
        CvMat* mat = (CvMat*)arr;
        int type = CV_MAT_TYPE(mat->type);
        int rows = mat->rows;
        uchar* m = mat->data.ptr;
        int step = mat->step;
        CV_Assert( rows == mat->cols );

        #define Mf(y, x) ((float*)(m + y*step))[x]
        #define Md(y, x) ((double*)(m + y*step))[x]

        if( type == CV_32F )
        {
            if( rows == 2 )
                return det2(Mf);
            if( rows == 3 )
                return det3(Mf);
        }
        else if( type == CV_64F )
        {
            if( rows == 2 )
                return det2(Md);
            if( rows == 3 )
                return det3(Md);
        }
    }
    return cv::determinant(cv::cvarrToMat(arr));
}


CV_IMPL double
cvInvert( const CvArr* srcarr, CvArr* dstarr, int method )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.type() == dst.type() && src.rows == dst.cols && src.cols == dst.rows );
    return cv::invert( src, dst, method == CV_CHOLESKY ? cv::DECOMP_CHOLESKY :
                      method == CV_SVD ? cv::DECOMP_SVD :
                      method == CV_SVD_SYM ? cv::DECOMP_EIG : cv::DECOMP_LU );
}


CV_IMPL int
cvSolve( const CvArr* Aarr, const CvArr* barr, CvArr* xarr, int method )
{
    cv::Mat A = cv::cvarrToMat(Aarr), b = cv::cvarrToMat(barr), x = cv::cvarrToMat(xarr);

    CV_Assert( A.type() == x.type() && A.cols == x.rows && x.cols == b.cols );
    bool is_normal = (method & CV_NORMAL) != 0;
    method &= ~CV_NORMAL;
    return cv::solve( A, b, x, (method == CV_CHOLESKY ? cv::DECOMP_CHOLESKY :
                                method == CV_SVD ? cv::DECOMP_SVD :
                                method == CV_SVD_SYM ? cv::DECOMP_EIG :
        A.rows > A.cols ? cv::DECOMP_QR : cv::DECOMP_LU) + (is_normal ? cv::DECOMP_NORMAL : 0) );
}


CV_IMPL void
cvEigenVV( CvArr* srcarr, CvArr* evectsarr, CvArr* evalsarr, double,
           int, int )
{
    cv::Mat src = cv::cvarrToMat(srcarr), evals0 = cv::cvarrToMat(evalsarr), evals = evals0;
    if( evectsarr )
    {
        cv::Mat evects0 = cv::cvarrToMat(evectsarr), evects = evects0;
        eigen(src, evals, evects);
        if( evects0.data != evects.data )
        {
            const uchar* p = evects0.ptr();
            evects.convertTo(evects0, evects0.type());
            CV_Assert( p == evects0.ptr() );
        }
    }
    else
        eigen(src, evals);
    if( evals0.data != evals.data )
    {
        const uchar* p = evals0.ptr();
        if( evals0.size() == evals.size() )
            evals.convertTo(evals0, evals0.type());
        else if( evals0.type() == evals.type() )
            cv::transpose(evals, evals0);
        else
            cv::Mat(evals.t()).convertTo(evals0, evals0.type());
        CV_Assert( p == evals0.ptr() );
    }
}


CV_IMPL void
cvSVD( CvArr* aarr, CvArr* warr, CvArr* uarr, CvArr* varr, int flags )
{
    cv::Mat a = cv::cvarrToMat(aarr), w = cv::cvarrToMat(warr), u, v;
    int m = a.rows, n = a.cols, type = a.type(), mn = std::max(m, n), nm = std::min(m, n);

    CV_Assert( w.type() == type &&
        (w.size() == cv::Size(nm,1) || w.size() == cv::Size(1, nm) ||
        w.size() == cv::Size(nm, nm) || w.size() == cv::Size(n, m)) );

    cv::SVD svd;

    if( w.size() == cv::Size(nm, 1) )
        svd.w = cv::Mat(nm, 1, type, w.ptr() );
    else if( w.isContinuous() )
        svd.w = w;

    if( uarr )
    {
        u = cv::cvarrToMat(uarr);
        CV_Assert( u.type() == type );
        svd.u = u;
    }

    if( varr )
    {
        v = cv::cvarrToMat(varr);
        CV_Assert( v.type() == type );
        svd.vt = v;
    }

    svd(a, ((flags & CV_SVD_MODIFY_A) ? cv::SVD::MODIFY_A : 0) |
        ((!svd.u.data && !svd.vt.data) ? cv::SVD::NO_UV : 0) |
        ((m != n && (svd.u.size() == cv::Size(mn, mn) ||
        svd.vt.size() == cv::Size(mn, mn))) ? cv::SVD::FULL_UV : 0));

    if( !u.empty() )
    {
        if( flags & CV_SVD_U_T )
            cv::transpose( svd.u, u );
        else if( u.data != svd.u.data )
        {
            CV_Assert( u.size() == svd.u.size() );
            svd.u.copyTo(u);
        }
    }

    if( !v.empty() )
    {
        if( !(flags & CV_SVD_V_T) )
            cv::transpose( svd.vt, v );
        else if( v.data != svd.vt.data )
        {
            CV_Assert( v.size() == svd.vt.size() );
            svd.vt.copyTo(v);
        }
    }

    if( w.data != svd.w.data )
    {
        if( w.size() == svd.w.size() )
            svd.w.copyTo(w);
        else
        {
            w = cv::Scalar(0);
            cv::Mat wd = w.diag();
            svd.w.copyTo(wd);
        }
    }
}


CV_IMPL void
cvSVBkSb( const CvArr* warr, const CvArr* uarr,
          const CvArr* varr, const CvArr* rhsarr,
          CvArr* dstarr, int flags )
{
    cv::Mat w = cv::cvarrToMat(warr), u = cv::cvarrToMat(uarr),
        v = cv::cvarrToMat(varr), rhs,
        dst = cv::cvarrToMat(dstarr), dst0 = dst;
    if( flags & CV_SVD_U_T )
    {
        cv::Mat tmp;
        transpose(u, tmp);
        u = tmp;
    }
    if( !(flags & CV_SVD_V_T) )
    {
        cv::Mat tmp;
        transpose(v, tmp);
        v = tmp;
    }
    if( rhsarr )
        rhs = cv::cvarrToMat(rhsarr);

    cv::SVD::backSubst(w, u, v, rhs, dst);
    CV_Assert( dst.data == dst0.data );
}
