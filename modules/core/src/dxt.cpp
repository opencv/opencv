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
#include "dxt.hpp"
#include "dxt.simd.hpp"
#include "dxt.simd_declarations.hpp"
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

#if defined USE_IPP_DFT
#if IPP_VERSION_X100 >= 202220
#define IPP_DISABLE_DFT32F ((depth == CV_32F) && (ippCPUID_AVX512F&cv::ipp::getIppFeatures()))
#else
#define IPP_DISABLE_DFT32F false
#endif
#endif

/****************************************************************************************\
                               Discrete Fourier Transform
\****************************************************************************************/

#ifdef HAVE_OPENCL
// factorization used by the OpenCL radix selection
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

// Per-transform options of the 1D engine: the immutable plan (shared by all rows), the kernel
// table resolved for the current CPU, and the per-executor workspace (see dxt.hpp).
struct OcvDftOptions {
    double scale;
    int n;

    bool isInverse;
    bool isComplex;     // DFT_COMPLEX_OUTPUT (forward) / complex input (inverse) for real transforms

    DFTFunc dft_func;
    bool useIpp;

#ifdef USE_IPP_DFT
    uchar* ipp_spec;
    uchar* ipp_work;
#endif

    const DftPlan* plan;
    const DFTKernels* kernels;
    uchar* workspace;

    OcvDftOptions()
    {
        plan = 0;
        kernels = 0;
        workspace = 0;
        scale = 0;
        n = 0;
        isInverse = false;
        isComplex = false;
        useIpp = false;
#ifdef USE_IPP_DFT
        ipp_spec = 0;
        ipp_work = 0;
#endif
        dft_func = 0;
    }
};


/****************************************************************************************\
                     New 1D engine: kernel table, plan builder, driver
\****************************************************************************************/

static DFTKernels getDFTKernelsDispatch(int depth)
{
    CV_CPU_DISPATCH(getDFTKernels_, (depth), CV_CPU_DISPATCH_MODES_ALL);
}

// The kernel tables are resolved once for the best available ISA; all further calls go
// through the function pointers with no dispatch overhead.
static const DFTKernels& getDFTKernels(int depth)
{
    static const DFTKernels k32 = getDFTKernelsDispatch(CV_32F);
    static const DFTKernels k64 = getDFTKernelsDispatch(CV_64F);
    return depth == CV_32F ? k32 : k64;
}

// exp(-2*pi*i*m/M) in double precision, m in [0, M): m = B*a + b -> hi[a]*lo[b] with B a power of
// two ~ sqrt(M). Both small tables are built by the rotation recurrence, re-seeded with exact
// sin/cos every 16 entries, so the error stays within a few ulps while only ~(B + M/B)/16 + 4
// trigonometric calls are made (4 for small M). Cheap to build, which matters because cv::dft()
// builds a fresh plan on every call.
struct DftTwiddleGen
{
    int M, logB, maskB;
    AutoBuffer<double, 1024> buf;
    double *hi_re, *hi_im, *lo_re, *lo_im;

    // re[i] + i*im[i] = w^i, i in [0, count), w = (wr, wi) = exp(i*stepAngle); the rotation
    // recurrence is re-seeded with exact sin/cos every 16 entries
    static void fillTable(double* re, double* im, int count, double wr, double wi, double stepAngle)
    {
        double cr = 1, ci = 0;
        for (int i = 0; i < count; i++)
        {
            if ((i & 15) == 0 && i > 0)
            {
                double t = stepAngle*i;
                cr = cos(t); ci = sin(t);
            }
            re[i] = cr; im[i] = ci;
            double t = cr*wr - ci*wi;
            ci = cr*wi + ci*wr;
            cr = t;
        }
    }

    void init(int _M)
    {
        M = _M;
        logB = 0;
        while ((1 << (2*logB)) < M) logB++;   // B = 2^logB >= sqrt(M)
        int B = 1 << logB;
        maskB = B - 1;
        int nhi = (M + B - 1)/B;
        buf.allocate(2*(B + nhi));
        hi_re = buf.data(); hi_im = hi_re + nhi;
        lo_re = hi_im + nhi; lo_im = lo_re + B;
        double a = -2*CV_PI/M;
        // the two step angles are computed exactly (4 trigonometric calls in total for small M)
        fillTable(lo_re, lo_im, B, cos(a), sin(a), a);
        fillTable(hi_re, hi_im, nhi, cos(a*B), sin(a*B), a*B);
    }

    void get(int m, double& re, double& im) const
    {
        CV_DbgAssert(0 <= m && m < M);
        int a = m >> logB, b = m & maskB;
        re = hi_re[a]*lo_re[b] - hi_im[a]*lo_im[b];
        im = hi_re[a]*lo_im[b] + hi_im[a]*lo_re[b];
    }
};

template<typename T>
void DftPlan::fillTables(size_t ntw, bool need_rtw, bool need_dct)
{
    DftPlan& p = *this;
    // all angles come from one base table exp(-2*pi*i*m/M), M a multiple of every angle grid used:
    // C2C: M = nc; real kinds with even n: M = n (=2nc) or 4n for DCT (W_4n^k); odd real: M = n = nc.
    int M = nc*(need_dct ? 8 : need_rtw ? 2 : 1);
    DftTwiddleGen gen;
    gen.init(M);
    p.tw.allocate((ntw*sizeof(T) + sizeof(double) - 1)/sizeof(double) + 1);
    T* cur = (T*)p.tw.data();
    int maxL = 1;
    for (int s = 0; s < p.nstages; s++)
        if (p.stages[s].radix <= 8) maxL = std::max(maxL, p.stages[s].span);
    AutoBuffer<double, 2048> twd_buf(2*maxL);     // leg-1 twiddles of the current stage in double
    double* twd = twd_buf.data();
    for (int s = 0; s < p.nstages; s++)
    {
        DftStage& st = p.stages[s];
        int r = st.radix, L = st.span, ts = st.tw_stride;
        T* twr = cur; T* twi = cur + (r - 1)*ts;
        cur = twi + (r - 1)*ts;
        int step = M/(r*L);
        if (r <= 8)
        {
            // Leg 1 (W_{rL}^j) by K independent interleaved rotation recurrences (a single chain is
            // latency-bound), each re-seeded from the exact generator every 16 of its steps; legs
            // q >= 2 as t_q(j) = t_{q-1}(j)*t_1(j), independent per entry (error <= q ulps).
            constexpr int K = 8;
            double cr[K], ci[K], wr, wi;
            gen.get(K*step, wr, wi);
            for (int k = 0; k < K; k++)
                gen.get(std::min(k, L - 1)*step, cr[k], ci[k]);
            double* t1r = twd; double* t1i = twd + L;      // leg 1 kept in double for the products
            for (int j = 0; j < L; j += K)
            {
                if ((j & (16*K - 1)) == 0 && j > 0)
                    for (int k = 0; k < K; k++)
                        gen.get(std::min(j + k, L - 1)*step, cr[k], ci[k]);
                for (int k = 0; k < K && j + k < L; k++)
                {
                    t1r[j+k] = cr[k]; t1i[j+k] = ci[k];
                    double t = cr[k]*wr - ci[k]*wi;
                    ci[k] = cr[k]*wi + ci[k]*wr;
                    cr[k] = t;
                }
            }
            for (int j = 0; j < L; j++)     // separate pass: vectorizes, the recurrence loop does not
            {
                twr[j] = (T)t1r[j]; twi[j] = (T)t1i[j];
            }
            for (int q = 2; q < r; q++)
            {
                const T* pr = twr + (q-2)*ts; const T* pi = twi + (q-2)*ts;
                T* tr = twr + (q-1)*ts; T* ti = twi + (q-1)*ts;
                for (int j = 0; j < L; j++)
                {
                    double ar = pr[j], ai = pi[j], br = t1r[j], bi = t1i[j];
                    tr[j] = (T)(ar*br - ai*bi);
                    ti[j] = (T)(ar*bi + ai*br);
                }
            }
        }
        else
        {
            // large odd radix: exact-ish generator lookup per entry
            for (int q = 1; q < r; q++)
                for (int j = 0; j < L; j++)
                {
                    double re, im;
                    gen.get(q*j*step, re, im);
                    twr[(q-1)*ts + j] = (T)re;
                    twi[(q-1)*ts + j] = (T)im;
                }
        }
        st.tw_re = twr; st.tw_im = twi;
        st.cs = st.sn = 0;
        if (r >= 7)
        {
            int h = (r - 1)/2;
            T* cs = cur; T* sn = cur + h*h;
            cur = sn + h*h;
            for (int pp = 1; pp <= h; pp++)
                for (int k = 1; k <= h; k++)
                {
                    double t = 2*CV_PI*((double)pp*k/r);
                    cs[(pp-1)*h + (k-1)] = (T)cos(t);
                    sn[(pp-1)*h + (k-1)] = (T)sin(t);
                }
            st.cs = cs; st.sn = sn;
        }
    }
    p.rtw_re = p.rtw_im = p.dct_re = p.dct_im = 0;
    if (need_rtw)
    {
        // W_n^k, k in [0, nc]
        T* wr = cur; T* wi = cur + (nc + 1);
        cur = wi + (nc + 1);
        int step = M/n;
        for (int k = 0; k <= nc; k++)
        {
            double re, im;
            gen.get(k*step, re, im);
            wr[k] = (T)re; wi[k] = (T)im;
        }
        p.rtw_re = wr; p.rtw_im = wi;
    }
    if (need_dct)
    {
        // s*W_4n^k, k in [0, nc]; s = 2*sqrt(1/(2n)) forward, sqrt(1/(2n)) inverse (the original normalization)
        T* wr = cur; T* wi = cur + (nc + 1);
        cur = wi + (nc + 1);
        double scale = (p.kind == DFT_KIND_DCT ? 2 : 1)*std::sqrt(1./(2*n));
        int step = M/(4*n);
        for (int k = 0; k <= nc; k++)
        {
            double re, im;
            gen.get(k*step, re, im);
            wr[k] = (T)(re*scale); wi[k] = (T)(im*scale);
        }
        p.dct_re = wr; p.dct_im = wi;
    }
}

// See dxt.hpp for the contract (in particular where `vl` must come from).
void DftPlan::build(int _kind, int _n, int _depth, int _vl)
{
    DftPlan& p = *this;
    CV_Assert(_n >= 1 && _vl >= 1);
    CV_Assert(_depth == CV_32F || _depth == CV_64F);
    bool real_kind = _kind != DFT_KIND_C2C;
    bool dct_kind = _kind == DFT_KIND_DCT || _kind == DFT_KIND_IDCT;
    if (dct_kind)
        CV_Assert(_n % 2 == 0);   // cv::dct() supports even n only (n == 1 is handled by the caller)
    p.kind = _kind; p.n = _n; p.depth = _depth; p.vl = _vl;
    p.nc = real_kind && _n % 2 == 0 ? _n/2 : _n;
    CV_Assert(nc < (1 << 28));
    size_t esz = depth == CV_32F ? sizeof(float) : sizeof(double);
    p.real_input = kind == DFT_KIND_R2C && (n & 1) != 0;
    p.need_tmp = kind == DFT_KIND_C2R || kind == DFT_KIND_DCT || kind == DFT_KIND_IDCT;
    p.pingpong = false;

    // factorization: 2^k part first (fused first stage r0 + radix-4 stages), then 3s, 5s, other odd.
    // (A radix-8 middle stage was tried and measured slower on NEON: with per-leg twiddle tables it
    // does not save loads over two radix-4 stages and keeps 16 data vectors live.)
    int k2 = 0, rest = nc;
    while ((rest & 1) == 0) { rest >>= 1; k2++; }
    int r0 = k2 == 0 ? 1 : k2 == 1 ? 2 : (k2 & 1) ? 8 : 4;
    p.first_radix = r0;
    int radices[DFT_MAX_STAGES], nst = 0;
    for (int m = k2 - (r0 == 1 ? 0 : r0 == 2 ? 1 : r0 == 4 ? 2 : 3); m > 0; m -= 2) radices[nst++] = 4;
    rest = nc >> k2;
    while (rest % 3 == 0) { CV_Assert(nst < DFT_MAX_STAGES); radices[nst++] = 3; rest /= 3; }
    while (rest % 5 == 0) { CV_Assert(nst < DFT_MAX_STAGES); radices[nst++] = 5; rest /= 5; }
    for (int f = 7; rest > 1; f += 2)
    {
        if (f*f > rest)
        {
            CV_Assert(nst < DFT_MAX_STAGES);
            radices[nst++] = rest;
            break;
        }
        while (rest % f == 0) { CV_Assert(nst < DFT_MAX_STAGES); radices[nst++] = f; rest /= f; }
    }
    p.nstages = nst;

    // stages
    int L = r0, max_h = 0;
    size_t ntw = 0;
    for (int s = 0; s < nst; s++)
    {
        DftStage& st = p.stages[s];
        int r = radices[s];
        st.radix = r; st.span = L; st.ngroups = nc/(r*L); st.tw_stride = L;
        if (vl > 1 && L >= vl)
        {
            st.mode = DFT_STAGE_FULL;
            st.nsteps = st.ngroups*((L + vl - 1)/vl);
            if (L % vl != 0)
                p.pingpong = true;   // back-off clamp => overlapping stores => must be out-of-place
        }
        else
        {
            st.mode = DFT_STAGE_SCALAR;
            st.nsteps = st.ngroups*L;
        }
        ntw += 2*(size_t)(r - 1)*st.tw_stride;
        if (r >= 7)
        {
            int h = (r - 1)/2;
            ntw += 2*(size_t)h*h;
            max_h = std::max(max_h, h);
        }
        L *= r;
    }
    CV_Assert(L == nc);

    // digit-reversal: position p = sum_s d_s*L_s holds input element sum_s d_s*M_s, M_s = nc/(L_s*r_s);
    // the radix sequence is r0 (if > 1) followed by the stage radices.
    //  - r0 == 1 (odd nc): itab holds two element indices (re, im) per complex element for the
    //    scalar gather (im index unused when real_input);
    //  - r0 > 1: the r0 inputs of group g are the column x[base(g) + q*nc/r0]; itab[base(g)] = g*r0
    //    tells the transposed first stage where the outputs of column base(g) go.
    int seq[DFT_MAX_STAGES + 1], nseq = 0;
    if (r0 > 1) seq[nseq++] = r0;
    for (int s = 0; s < nst; s++) seq[nseq++] = radices[s];
    int digits[DFT_MAX_STAGES + 1], weights[DFT_MAX_STAGES + 1];
    {
        int Ls = 1;
        for (int s = 0; s < nseq; s++)
        {
            digits[s] = 0;
            weights[s] = nc/(Ls*seq[s]);
            Ls *= seq[s];
        }
    }
    if (r0 == 1)
    {
        p.itab.allocate(nc*2);
        int* it_all = p.itab.data();
        // the two lowest digits are enumerated by plain nested loops, the higher ones by a counter
        int n_in = std::min(nseq, 2);
        int r_0 = n_in > 0 ? seq[0] : 1, w_0 = n_in > 0 ? weights[0] : 0;
        int r_1 = n_in > 1 ? seq[1] : 1, w_1 = n_in > 1 ? weights[1] : 0;
        for (int pos = 0, rev_hi = 0; pos < nc; )
        {
            for (int d1 = 0; d1 < r_1; d1++)
            {
                int rev = rev_hi + d1*w_1;
                int* it = it_all + pos*2;
                if (p.real_input)
                {
                    for (int d0 = 0; d0 < r_0; d0++, rev += w_0)
                    {
                        it[d0*2] = rev; it[d0*2+1] = rev;
                    }
                }
                else
                {
                    for (int d0 = 0; d0 < r_0; d0++, rev += w_0)
                    {
                        it[d0*2] = rev*2; it[d0*2+1] = rev*2 + 1;
                    }
                }
                pos += r_0;
            }
            for (int s = n_in; s < nseq; s++)
            {
                rev_hi += weights[s];
                if (++digits[s] < seq[s])
                    break;
                digits[s] = 0;
                rev_hi -= seq[s]*weights[s];
            }
        }
    }
    else
    {
        // gtab[base(g)] = g*r0, written in increasing base order (sequential stores): base is
        // counted up with the digits s = nseq-1 (base weight 1) .. 1 (base weight weights[1]),
        // g follows incrementally with the digit weights gw[s] = L_s/r0. The two fastest digits
        // are enumerated by plain nested loops.
        int M = nc/r0;
        p.itab.allocate(M);
        int* gtab = p.itab.data();
        int gw[DFT_MAX_STAGES + 1];
        {
            int Ls = 1;
            for (int s = 0; s < nseq; s++)
            {
                gw[s] = Ls/r0;      // valid for s >= 1 (L_s is a multiple of r0)
                Ls *= seq[s];
            }
        }
        int sa = nseq - 1, sb = nseq - 2;                  // fastest and second fastest digits
        int r_a = sa >= 1 ? seq[sa] : 1, g_a = sa >= 1 ? gw[sa] : 0;
        int r_b = sb >= 1 ? seq[sb] : 1, g_b = sb >= 1 ? gw[sb] : 0;
        for (int base = 0, g_hi = 0; base < M; )
        {
            for (int db = 0; db < r_b; db++)
            {
                int g = (g_hi + db*g_b)*r0;
                for (int da = 0; da < r_a; da++, g += g_a*r0)
                    gtab[base++] = g;
            }
            for (int s = sb - 1; s >= 1; s--)
            {
                g_hi += gw[s];
                if (++digits[s] < seq[s])
                    break;
                digits[s] = 0;
                g_hi -= seq[s]*gw[s];
            }
        }
    }

    // twiddle tables
    bool need_rtw = real_kind && n % 2 == 0;
    if (need_rtw) ntw += 2*(size_t)(nc + 1);
    if (dct_kind) ntw += 2*(size_t)(nc + 1);
    if (depth == CV_32F)
        fillTables<float>(ntw, need_rtw, dct_kind);
    else
        fillTables<double>(ntw, need_rtw, dct_kind);

    // workspace layout (from the 64-byte aligned base): pair0 (re, im), optional pair1 (re, im) or
    // the interleaved temp, radix-odd scratch (a[h], b[h] complex per lane).
    size_t pair_bytes = 2*(size_t)nc*esz;
    bool two_pairs = p.pingpong || p.need_tmp;
    p.pair1_ofs = two_pairs ? pair_bytes : 0;
    p.scratch_ofs = pair_bytes*(two_pairs ? 2 : 1);
    size_t scratch_bytes = 4*(size_t)max_h*std::max(vl, 1)*esz;
    p.ws_bytes = p.scratch_ofs + scratch_bytes + 64;
}

// Runs the transform described by c.plan: A (preproc) -> B stages (ping-pong) -> C (postproc).
// No allocations. sstep/dstep are in elements (used by DCT/IDCT only; 1 otherwise).
static void runDft(const OcvDftOptions& c, const void* src, size_t sstep, void* dst, size_t dstep,
                   bool complex_io)
{
    const DftPlan& p = *c.plan;
    const DFTKernels& k = *c.kernels;
    CV_DbgAssert(p.vl == k.vlanes);
    size_t esz = p.depth == CV_32F ? sizeof(float) : sizeof(double);
    uchar* base = alignPtr(c.workspace, 64);
    uchar* re0 = base;
    uchar* im0 = re0 + p.nc*esz;
    uchar* re1 = base + p.pair1_ofs;
    uchar* im1 = re1 + p.nc*esz;
    uchar* scratch = base + p.scratch_ofs;

    const void* asrc = src;
    size_t astep = sstep;
    bool conj = false;
    if (p.kind == DFT_KIND_C2R)
    {
        k.preprocCCS(p, src, re1, complex_io);
        asrc = re1; astep = 1; conj = true;
    }
    else if (p.kind == DFT_KIND_IDCT)
    {
        k.preprocIDCT(p, src, sstep, re1);
        asrc = re1; astep = 1; conj = true;
    }
    else if (p.kind == DFT_KIND_DCT)
    {
        k.preprocDCT(p, src, sstep, re1);
        asrc = re1; astep = 1;
    }
    else if (p.kind == DFT_KIND_C2C)
        conj = c.isInverse;
    CV_DbgAssert(astep == 1 || p.first_radix == 1);

    DftPreprocFunc pre = p.first_radix == 1 ? k.preprocRadix0 : p.first_radix == 2 ? k.preprocRadix2 :
                         p.first_radix == 4 ? k.preprocRadix4 : k.preprocRadix8;
    pre(p, asrc, astep, re0, im0, conj);

    uchar *sre = re0, *sim = im0, *dre = re1, *dim = im1;
    for (int i = 0; i < p.nstages; i++)
    {
        const DftStage& st = p.stages[i];
        DftStageFunc fn = st.radix == 4 ? k.radix4 : st.radix == 3 ? k.radix3 :
                          st.radix == 5 ? k.radix5 : k.radixOdd;
        if (p.pingpong)
        {
            fn(st, sre, sim, dre, dim, scratch);
            std::swap(sre, dre); std::swap(sim, dim);
        }
        else
            fn(st, sre, sim, sre, sim, scratch);
    }

    double scale = c.scale;
    switch (p.kind)
    {
    case DFT_KIND_C2C:
        k.postprocDFT(p, sre, sim, dst, scale, c.isInverse ? -scale : scale);
        break;
    case DFT_KIND_R2C:
        k.postprocRealDFT(p, sre, sim, dst, scale, complex_io);
        break;
    case DFT_KIND_C2R:
        if (p.nc == p.n)
            k.postprocReal(p, sre, dst, scale);
        else
            k.postprocDFT(p, sre, sim, dst, scale, -scale);
        break;
    case DFT_KIND_DCT:
        k.postprocDCT(p, sre, sim, dst, dstep);
        break;
    default:
        CV_Assert(p.kind == DFT_KIND_IDCT);
        k.postprocIDCT(p, sre, sim, dst, dstep);
    }
}

// mixed-radix complex discrete Fourier transform (forward or inverse)
template<typename T> static void
DFT(const OcvDftOptions & c, const Complex<T>* src, Complex<T>* dst)
{
    if( c.useIpp )
    {
#ifdef USE_IPP_DFT
        if( !c.isInverse )
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

    runDft(c, src, 1, dst, 1, false);
}


/* FFT of real vector
   output vector format:
     re(0), re(1), im(1), ... , re(n/2-1), im((n+1)/2-1) [, re((n+1)/2)] OR ...
     re(0), 0, re(1), im(1), ..., re(n/2-1), im((n+1)/2-1) [, re((n+1)/2), 0] */
template<typename T> static void
RealDFT(const OcvDftOptions & c, const T* src, T* dst)
{
    int complex_output = c.isComplex;

    if( c.useIpp )
    {
#ifdef USE_IPP_DFT
        int n = c.n;
        T* ipp_dst = dst + complex_output;
        if (ippsDFTFwd_RToPack( src, ipp_dst, c.ipp_spec, c.ipp_work ) >=0)
        {
            if( complex_output )
            {
                ipp_dst[-1] = ipp_dst[0];
                ipp_dst[0] = 0;
                if( (n & 1) == 0 )
                    ipp_dst[n] = 0;
            }
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }
        setIppErrorStatus();
#endif
    }
    runDft(c, src, 1, dst, 1, complex_output != 0);
}

/* Inverse FFT of complex conjugate-symmetric vector
   input vector format:
      re[0], re[1], im[1], ... , re[n/2-1], im[n/2-1], re[n/2] OR
      re(0), 0, re(1), im(1), ..., re(n/2-1), im((n+1)/2-1) [, re((n+1)/2), 0] */
template<typename T> static void
CCSIDFT(const OcvDftOptions & c, const T* src, T* dst)
{
    int complex_input = c.isComplex;

    if( complex_input )
        CV_Assert( src != dst );
    if( c.useIpp )
    {
#ifdef USE_IPP_DFT
        // IPP expects the packed format; with complex input the packed spectrum is src+1 once
        // im(0) (always zero) is overwritten with re(0)
        const T* ipp_src = src;
        T save_s1 = 0;
        if( complex_input )
        {
            save_s1 = src[1];
            ((T*)src)[1] = src[0];
            ipp_src = src + 1;
        }
        IppStatus status = ippsDFTInv_PackToR( ipp_src, dst, c.ipp_spec, c.ipp_work );
        if( complex_input )
            ((T*)src)[1] = save_s1;
        if( status >= 0 )
        {
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }
        setIppErrorStatus();
#endif
    }
    runDft(c, src, 1, dst, 1, complex_input != 0);
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

template<typename T, void (*fn)(const OcvDftOptions&, const T*, T*)>
static void dftWrap(const OcvDftOptions & c, const void* src, void* dst)
{
    fn(c, (const T*)src, (T*)dst);
}

}

#ifdef USE_IPP_DFT
typedef IppStatus (CV_STDCALL* IppDFTGetSizeFunc)(int, int, IppHintAlgorithm, int*, int*, int*);
typedef IppStatus (CV_STDCALL* IppDFTInitFunc)(int, int, IppHintAlgorithm, void*, uchar*);

template<typename SpecType, IppStatus (CV_STDCALL *init_fn)(int, int, IppHintAlgorithm, SpecType*, Ipp8u*)>
static IppStatus CV_STDCALL ippDFTInitWrap(int n, int flags, IppHintAlgorithm hint, void* spec, Ipp8u* initbuf)
{
    return init_fn(n, flags, hint, (SpecType*)spec, initbuf);
}
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
        _dst.createSameSize(src, CV_MAKETYPE(depth, 2));
        output = _dst.getUMat();
    }
    else
    {
        // real output
        if (is1d)
        {
            _dst.createSameSize(src, CV_MAKETYPE(depth, 1));
            output = _dst.getUMat();
        }
        else
        {
            _dst.createSameSize(src, CV_MAKETYPE(depth, 1));
            output.create(src.size, CV_MAKETYPE(depth, 2));
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

class OcvDftBasicImpl CV_FINAL : public hal::DFT1D
{
public:
    OcvDftOptions opt;
    // the plan is built once in init(), the workspace is allocated once here and reused by
    // every apply() (zero allocations per transform)
    DftPlan plan;
    AutoBuffer<uchar> ws;
#ifdef USE_IPP_DFT
    AutoBuffer<uchar> ippbuf;
    AutoBuffer<uchar> ippworkbuf;
#endif

public:
    OcvDftBasicImpl()
    {
    }
    void init(int len, int count, int depth, int flags, bool *needBuffer)
    {
        int stage = (flags & CV_HAL_DFT_STAGE_COLS) != 0 ? 1 : 0;
        opt.isInverse = (flags & CV_HAL_DFT_INVERSE) != 0;
        bool real_transform = (flags & CV_HAL_DFT_REAL_OUTPUT) != 0;
        opt.isComplex = (stage == 0) && (flags & CV_HAL_DFT_COMPLEX_OUTPUT) != 0;
        bool needAnotherStage = (flags & CV_HAL_DFT_TWO_STAGE) != 0;

        opt.scale = 1;
        opt.n = len;

        opt.useIpp = false;
    #ifdef USE_IPP_DFT
        opt.ipp_spec = 0;
        opt.ipp_work = 0;

        if( CV_IPP_CHECK_COND && (opt.n*count >= 64) && !IPP_DISABLE_DFT32F) // use IPP DFT if available
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
                    initFunc = ippDFTInitWrap<IppsDFTSpec_R_32f, ippsDFTInit_R_32f>;
                }
                else
                {
                    getSizeFunc = ippsDFTGetSize_R_64f;
                    initFunc = ippDFTInitWrap<IppsDFTSpec_R_64f, ippsDFTInit_R_64f>;
                }
            }
            else
            {
                if( depth == CV_32F )
                {
                    getSizeFunc = ippsDFTGetSize_C_32fc;
                    initFunc = ippDFTInitWrap<IppsDFTSpec_C_32fc, ippsDFTInit_C_32fc>;
                }
                else
                {
                    getSizeFunc = ippsDFTGetSize_C_64fc;
                    initFunc = ippDFTInitWrap<IppsDFTSpec_C_64fc, ippsDFTInit_C_64fc>;
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

        {
            // The plan is built even when IPP is enabled: the IPP calls fall back to the generic
            // code on failure.
            const DFTKernels& kernels = getDFTKernels(depth);
            int kind = stage == 0 && real_transform ? (opt.isInverse ? DFT_KIND_C2R : DFT_KIND_R2C)
                                                    : DFT_KIND_C2C;
            plan.build(kind, len, depth, kernels.vlanes);
            ws.allocate(plan.ws_bytes);
            opt.plan = &plan;
            opt.kernels = &kernels;
            opt.workspace = ws.data();
            if (needBuffer)
                *needBuffer = false;   // the engine supports in-place for every kind
        }

        {
            static DFTFunc dft_tbl[6] =
            {
                dftWrap<Complexf, DFT<float>>,
                dftWrap<float, RealDFT<float>>,
                dftWrap<float, CCSIDFT<float>>,
                dftWrap<Complexd, DFT<double>>,
                dftWrap<double, RealDFT<double>>,
                dftWrap<double, CCSIDFT<double>>
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

    // size of a private workspace for applyWithWorkspace() (0 when the IPP path is used)
    size_t workspaceSize() const { return opt.plan ? plan.ws_bytes : 0; }

    // Same as apply(), but with a caller-provided workspace: several threads may run the same
    // transform (the plan is immutable) as long as each of them uses its own workspace.
    void applyWithWorkspace(const uchar *src, uchar *dst, uchar* workspace) const
    {
        OcvDftOptions opt_ = opt;
        opt_.workspace = workspace;
        opt_.dft_func(opt_, src, dst);
    }

    void free() {}
};

// 2D transforms (and the multi-row 1D batches) distribute the rows / the column pairs over
// threads. The per-thread state is the 1D workspace (+ the column buffers); the plans are shared.
// Below this many elements the transform runs on the calling thread.
static const size_t DFT_PARALLEL_MIN_ELEMS = 1 << 16;

class OcvDftImpl CV_FINAL : public hal::DFT2D
{
protected:
    Ptr<hal::DFT1D> contextA;
    Ptr<hal::DFT1D> contextB;
    const OcvDftBasicImpl* basicA;   // contextA/contextB when they are our own implementation
    const OcvDftBasicImpl* basicB;   // (parallel path); 0 for external HAL contexts
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
        basicA = basicB = 0;
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
                basicA = dynamic_cast<const OcvDftBasicImpl*>(contextA.get());
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
                basicB = dynamic_cast<const OcvDftBasicImpl*>(contextB.get());
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

        size_t ws_size = basicA ? basicA->workspaceSize() : 0;
        if( ws_size > 0 && nz > 1 && (size_t)nz*len >= DFT_PARALLEL_MIN_ELEMS )
        {
            // every worker gets its own workspace (and row buffer) once per range
            const OcvDftBasicImpl* ctx = basicA;
            auto processRows = [&, ctx](const Range& range)
            {
                AutoBuffer<uchar> ws(ws_size);
                AutoBuffer<uchar> rowbuf(needBufferA ? len * complex_elem_size : 0);
                for( int i = range.start; i < range.end; i++ )
                {
                    const uchar* sptr = src_data + src_step * i;
                    uchar* dptr0 = dst_data + dst_step * i;
                    uchar* dptr = needBufferA ? rowbuf.data() : dptr0;
                    ctx->applyWithWorkspace(sptr, dptr, ws.data());
                    if( needBufferA )
                        memcpy( dptr0, dptr + dptr_offset, dst_full_len );
                }
            };
            parallel_for_(Range(0, nz), processRows, (double)nz*len/DFT_PARALLEL_MIN_ELEMS);
        }
        else
        {
            for( int i = 0; i < nz; i++ )
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
        }

        for( int i = nz; i < count; i++ )
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

        size_t ws_size = basicB ? basicB->workspaceSize() : 0;
        int npairs = (b - a + 1)/2;
        if( ws_size > 0 && npairs > 1 && (size_t)(b - a)*len >= DFT_PARALLEL_MIN_ELEMS )
        {
            // every worker gets its own column buffers and workspace once per range
            const OcvDftBasicImpl* ctx = basicB;
            size_t colbuf_size = (size_t)len*complex_elem_size;
            auto processPairs = [&, ctx](const Range& range)
            {
                AutoBuffer<uchar> ws(ws_size);
                AutoBuffer<uchar> cbuf(colbuf_size*(needBufferB ? 4 : 2));
                uchar* cbuf0 = cbuf.data();
                uchar* cbuf1 = cbuf0 + colbuf_size;
                uchar* cdbuf0 = needBufferB ? cbuf1 + colbuf_size : cbuf0;
                uchar* cdbuf1 = needBufferB ? cdbuf0 + colbuf_size : cbuf1;
                for( int pi = range.start; pi < range.end; pi++ )
                {
                    int i = a + pi*2;
                    const uchar* sptr = sptr0 + (size_t)pi*2*complex_elem_size;
                    uchar* dptr = dptr0 + (size_t)pi*2*complex_elem_size;
                    if( i+1 < b )
                    {
                        CopyFrom2Columns( sptr, src_step, cbuf0, cbuf1, len, complex_elem_size );
                        ctx->applyWithWorkspace(cbuf1, cdbuf1, ws.data());
                    }
                    else
                        CopyColumn( sptr, src_step, cbuf0, complex_elem_size, len, complex_elem_size );

                    ctx->applyWithWorkspace(cbuf0, cdbuf0, ws.data());

                    if( i+1 < b )
                        CopyTo2Columns( cdbuf0, cdbuf1, dptr, dst_step, len, complex_elem_size );
                    else
                        CopyColumn( cdbuf0, complex_elem_size, dptr, dst_step, len, complex_elem_size );
                }
            };
            parallel_for_(Range(0, npairs), processPairs, (double)(b - a)*len/DFT_PARALLEL_MIN_ELEMS);
        }
        else
        {
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
        }
        if(isLastStage && mode == FwdRealToComplex)
            complementComplexOutput(depth, dst_data, dst_step, count, len, 2);
    }
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
        _dst.createSameSize( src, CV_MAKETYPE(depth, 2) );
    else if( inv && src.channels() == 2 && (flags & DFT_REAL_OUTPUT) )
        _dst.createSameSize( src, depth );
    else
        _dst.createSameSize( src, type );

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

    _dst.createSameSize(A, atype);
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

void cv::divSpectrums( InputArray _srcA, InputArray _srcB, OutputArray _dst, int flags, bool conjB)
{
    Mat srcA = _srcA.getMat(), srcB = _srcB.getMat();
    int depth = srcA.depth(), cn = srcA.channels(), type = srcA.type();
    int rows = srcA.rows, cols = srcA.cols;
    int j, k;

    CV_Assert( type == srcB.type() && srcA.size() == srcB.size() );
    CV_Assert( type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2 );

    _dst.create( srcA.rows, srcA.cols, type );
    Mat dst = _dst.getMat();

    CV_Assert(dst.data != srcA.data); // non-inplace check
    CV_Assert(dst.data != srcB.data); // non-inplace check

    bool is_1d = (flags & DFT_ROWS) || (rows == 1 || (cols == 1 &&
    srcA.isContinuous() && srcB.isContinuous() && dst.isContinuous()));

    if( is_1d && !(flags & DFT_ROWS) )
        cols = cols + rows - 1, rows = 1;

    int ncols = cols*cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);

    if( depth == CV_32F )
    {
        const float* dataA = srcA.ptr<float>();
        const float* dataB = srcB.ptr<float>();
        float* dataC = dst.ptr<float>();
        float eps = FLT_EPSILON; // prevent div0 problems

        size_t stepA = srcA.step/sizeof(dataA[0]);
        size_t stepB = srcB.step/sizeof(dataB[0]);
        size_t stepC = dst.step/sizeof(dataC[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( rows % 2 == 0 )
                    dataC[(rows-1)*stepC] = dataA[(rows-1)*stepA] / (dataB[(rows-1)*stepB] + eps);
                if( !conjB )
                    for( j = 1; j <= rows - 2; j += 2 )
                    {
                        double denom = (double)dataB[j*stepB]*dataB[j*stepB] +
                        (double)dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + (double)eps;

                        double re = (double)dataA[j*stepA]*dataB[j*stepB] +
                        (double)dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = (double)dataA[(j+1)*stepA]*dataB[j*stepB] -
                        (double)dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = (float)(re / denom);
                        dataC[(j+1)*stepC] = (float)(im / denom);
                    }
                    else
                        for( j = 1; j <= rows - 2; j += 2 )
                        {

                            double denom = (double)dataB[j*stepB]*dataB[j*stepB] +
                            (double)dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + (double)eps;

                            double re = (double)dataA[j*stepA]*dataB[j*stepB] -
                            (double)dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                            double im = (double)dataA[(j+1)*stepA]*dataB[j*stepB] +
                            (double)dataA[j*stepA]*dataB[(j+1)*stepB];

                            dataC[j*stepC] = (float)(re / denom);
                            dataC[(j+1)*stepC] = (float)(im / denom);
                        }
                    if( k == 1 )
                        dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
            }
        }

        for( ; rows--; dataA += stepA, dataB += stepB, dataC += stepC )
        {
            if( is_1d && cn == 1 )
            {
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( cols % 2 == 0 )
                    dataC[j1] = dataA[j1] / (dataB[j1] + eps);
            }

            if( !conjB )
                for( j = j0; j < j1; j += 2 )
                {
                    double denom = (double)dataB[j]*dataB[j] + (double)dataB[j+1]*dataB[j+1] + (double)eps;
                    double re = (double)dataA[j]*dataB[j] + (double)dataA[j+1]*dataB[j+1];
                    double im = (double)dataA[j+1]*dataB[j] - (double)dataA[j]*dataB[j+1];
                    dataC[j] = (float)(re / denom);
                    dataC[j+1] = (float)(im / denom);
                }
                else
                    for( j = j0; j < j1; j += 2 )
                    {
                        double denom = (double)dataB[j]*dataB[j] + (double)dataB[j+1]*dataB[j+1] + (double)eps;
                        double re = (double)dataA[j]*dataB[j] - (double)dataA[j+1]*dataB[j+1];
                        double im = (double)dataA[j+1]*dataB[j] + (double)dataA[j]*dataB[j+1];
                        dataC[j] = (float)(re / denom);
                        dataC[j+1] = (float)(im / denom);
                    }
        }
    }
    else
    {
        const double* dataA = srcA.ptr<double>();
        const double* dataB = srcB.ptr<double>();
        double* dataC = dst.ptr<double>();
        double eps = DBL_EPSILON; // prevent div0 problems

        size_t stepA = srcA.step/sizeof(dataA[0]);
        size_t stepB = srcB.step/sizeof(dataB[0]);
        size_t stepC = dst.step/sizeof(dataC[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( rows % 2 == 0 )
                    dataC[(rows-1)*stepC] = dataA[(rows-1)*stepA] / (dataB[(rows-1)*stepB] + eps);
                if( !conjB )
                    for( j = 1; j <= rows - 2; j += 2 )
                    {
                        double denom = dataB[j*stepB]*dataB[j*stepB] +
                        dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + eps;

                        double re = dataA[j*stepA]*dataB[j*stepB] +
                        dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = dataA[(j+1)*stepA]*dataB[j*stepB] -
                        dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = re / denom;
                        dataC[(j+1)*stepC] = im / denom;
                    }
                    else
                        for( j = 1; j <= rows - 2; j += 2 )
                        {
                            double denom = dataB[j*stepB]*dataB[j*stepB] +
                            dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + eps;

                            double re = dataA[j*stepA]*dataB[j*stepB] -
                            dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                            double im = dataA[(j+1)*stepA]*dataB[j*stepB] +
                            dataA[j*stepA]*dataB[(j+1)*stepB];

                            dataC[j*stepC] = re / denom;
                            dataC[(j+1)*stepC] = im / denom;
                        }
                    if( k == 1 )
                        dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
            }
        }

        for( ; rows--; dataA += stepA, dataB += stepB, dataC += stepC )
        {
            if( is_1d && cn == 1 )
            {
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( cols % 2 == 0 )
                    dataC[j1] = dataA[j1] / (dataB[j1] + eps);
            }

            if( !conjB )
                for( j = j0; j < j1; j += 2 )
                {
                    double denom = dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps;
                    double re = dataA[j]*dataB[j] + dataA[j+1]*dataB[j+1];
                    double im = dataA[j+1]*dataB[j] - dataA[j]*dataB[j+1];
                    dataC[j] = re / denom;
                    dataC[j+1] = im / denom;
                }
                else
                    for( j = j0; j < j1; j += 2 )
                    {
                        double denom = dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps;
                        double re = dataA[j]*dataB[j] - dataA[j+1]*dataB[j+1];
                        double im = dataA[j+1]*dataB[j] + dataA[j]*dataB[j+1];
                        dataC[j] = re / denom;
                        dataC[j+1] = im / denom;
                    }
        }
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
// DCT-II / DCT-III through the new engine (plan kinds DFT_KIND_DCT / DFT_KIND_IDCT).
// src_step / dst_step are in bytes.
template<typename T> static void
DCT( const OcvDftOptions & c, const T* src, size_t src_step, T* dst, size_t dst_step )
{
    if( c.n == 1 )
    {
        dst[0] = src[0];
        return;
    }
    runDft(c, src, src_step/sizeof(T), dst, dst_step/sizeof(T), false);
}

template<typename T> static void
IDCT( const OcvDftOptions & c, const T* src, size_t src_step, T* dst, size_t dst_step )
{
    if( c.n == 1 )
    {
        dst[0] = src[0];
        return;
    }
    runDft(c, src, src_step/sizeof(T), dst, dst_step/sizeof(T), false);
}

typedef void (*DCTFunc)(const OcvDftOptions & c, const void* src, size_t src_step,
                        void* dst, size_t dst_step);

template<typename T, void (*fn)(const OcvDftOptions&, const T*, size_t, T*, size_t)>
static void dctWrap(const OcvDftOptions & c, const void* src, size_t src_step, void* dst, size_t dst_step)
{
    fn(c, (const T*)src, src_step, (T*)dst, dst_step);
}

}

#ifdef HAVE_IPP
namespace cv
{

#if IPP_VERSION_X100 >= 900
typedef IppStatus (CV_STDCALL * ippiDCTFunc)(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep, const void* pDCTSpec, Ipp8u* pBuffer);
typedef IppStatus (CV_STDCALL * ippiDCTInit)(void* pDCTSpec, IppiSize roiSize, Ipp8u* pMemInit );
typedef IppStatus (CV_STDCALL * ippiDCTGetSize)(IppiSize roiSize, int* pSizeSpec, int* pSizeInit, int* pSizeBuf);

template<typename SpecType, IppStatus (CV_STDCALL *fn)(const Ipp32f*, int, Ipp32f*, int, const SpecType*, Ipp8u*)>
static IppStatus CV_STDCALL ippiDCTWrap(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep, const void* pDCTSpec, Ipp8u* pBuffer)
{
    return fn(pSrc, srcStep, pDst, dstStep, (const SpecType*)pDCTSpec, pBuffer);
}

template<typename SpecType, IppStatus (CV_STDCALL *fn)(SpecType*, IppiSize, Ipp8u*)>
static IppStatus CV_STDCALL ippiDCTInitWrap(void* pDCTSpec, IppiSize roiSize, Ipp8u* pMemInit)
{
    return fn((SpecType*)pDCTSpec, roiSize, pMemInit);
}

template<IppStatus (CV_STDCALL *fn)(IppiSize, int*, int*, int*)>
static IppStatus CV_STDCALL ippiDCTGetSizeWrap(IppiSize roiSize, int* pSizeSpec, int* pSizeInit, int* pSizeBuf)
{
    return fn(roiSize, pSizeSpec, pSizeInit, pSizeBuf);
}

#elif IPP_VERSION_X100 >= 700
typedef IppStatus (CV_STDCALL * ippiDCTFunc)(const Ipp32f*, int, Ipp32f*, int, const void*, Ipp8u*);
typedef IppStatus (CV_STDCALL * ippiDCTInitAlloc)(void**, IppiSize, IppHintAlgorithm);
typedef IppStatus (CV_STDCALL * ippiDCTFree)(void* pDCTSpec);
typedef IppStatus (CV_STDCALL * ippiDCTGetBufSize)(const void*, int*);

template<typename SpecType, IppStatus (CV_STDCALL *fn)(const Ipp32f*, int, Ipp32f*, int, const SpecType*, Ipp8u*)>
static IppStatus CV_STDCALL ippiDCTWrap(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep, const void* pDCTSpec, Ipp8u* pBuffer)
{
    return fn(pSrc, srcStep, pDst, dstStep, (const SpecType*)pDCTSpec, pBuffer);
}

template<typename SpecType, IppStatus (CV_STDCALL *fn)(SpecType**, IppiSize, IppHintAlgorithm)>
static IppStatus CV_STDCALL ippiDCTInitAllocWrap(void** pDCTSpec, IppiSize roiSize, IppHintAlgorithm hint)
{
    return fn((SpecType**)pDCTSpec, roiSize, hint);
}

template<typename SpecType, IppStatus (CV_STDCALL *fn)(SpecType*)>
static IppStatus CV_STDCALL ippiDCTFreeWrap(void* pDCTSpec)
{
    return fn((SpecType*)pDCTSpec);
}

template<typename SpecType, IppStatus (CV_STDCALL *fn)(const SpecType*, int*)>
static IppStatus CV_STDCALL ippiDCTGetBufSizeWrap(const void* pDCTSpec, int* pSize)
{
    return fn((const SpecType*)pDCTSpec, pSize);
}
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

        ippiDCTFunc     ippiDCT_32f_C1R   = inv ? ippiDCTWrap<IppiDCTInvSpec_32f, ippiDCTInv_32f_C1R> : ippiDCTWrap<IppiDCTFwdSpec_32f, ippiDCTFwd_32f_C1R>;
        ippiDCTInit     ippDctInit     = inv ? ippiDCTInitWrap<IppiDCTInvSpec_32f, ippiDCTInvInit_32f> : ippiDCTInitWrap<IppiDCTFwdSpec_32f, ippiDCTFwdInit_32f>;
        ippiDCTGetSize  ippDctGetSize  = inv ? ippiDCTGetSizeWrap<ippiDCTInvGetSize_32f> : ippiDCTGetSizeWrap<ippiDCTFwdGetSize_32f>;

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

        ippiDCTFunc ippDctFun           = inv ? ippiDCTWrap<IppiDCTInvSpec_32f, ippiDCTInv_32f_C1R>             : ippiDCTWrap<IppiDCTFwdSpec_32f, ippiDCTFwd_32f_C1R>;
        ippiDCTInitAlloc ippInitAlloc   = inv ? ippiDCTInitAllocWrap<IppiDCTInvSpec_32f, ippiDCTInvInitAlloc_32f>   : ippiDCTInitAllocWrap<IppiDCTFwdSpec_32f, ippiDCTFwdInitAlloc_32f>;
        ippiDCTFree ippFree             = inv ? ippiDCTFreeWrap<IppiDCTInvSpec_32f, ippiDCTInvFree_32f>             : ippiDCTFreeWrap<IppiDCTFwdSpec_32f, ippiDCTFwdFree_32f>;
        ippiDCTGetBufSize ippGetBufSize = inv ? ippiDCTGetBufSizeWrap<IppiDCTInvSpec_32f, ippiDCTInvGetBufSize_32f> : ippiDCTGetBufSizeWrap<IppiDCTFwdSpec_32f, ippiDCTFwdGetBufSize_32f>;

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

        ippiDCTFunc     ippiDCT_32f_C1R      = inv ? ippiDCTWrap<IppiDCTInvSpec_32f, ippiDCTInv_32f_C1R> : ippiDCTWrap<IppiDCTFwdSpec_32f, ippiDCTFwd_32f_C1R>;
        ippiDCTInit     ippDctInit     = inv ? ippiDCTInitWrap<IppiDCTInvSpec_32f, ippiDCTInvInit_32f> : ippiDCTInitWrap<IppiDCTFwdSpec_32f, ippiDCTFwdInit_32f>;
        ippiDCTGetSize  ippDctGetSize  = inv ? ippiDCTGetSizeWrap<ippiDCTInvGetSize_32f> : ippiDCTGetSizeWrap<ippiDCTFwdGetSize_32f>;

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

        ippiDCTFunc ippDctFun           = inv ? ippiDCTWrap<IppiDCTInvSpec_32f, ippiDCTInv_32f_C1R>             : ippiDCTWrap<IppiDCTFwdSpec_32f, ippiDCTFwd_32f_C1R>;
        ippiDCTInitAlloc ippInitAlloc   = inv ? ippiDCTInitAllocWrap<IppiDCTInvSpec_32f, ippiDCTInvInitAlloc_32f>   : ippiDCTInitAllocWrap<IppiDCTFwdSpec_32f, ippiDCTFwdInitAlloc_32f>;
        ippiDCTFree ippFree             = inv ? ippiDCTFreeWrap<IppiDCTInvSpec_32f, ippiDCTInvFree_32f>             : ippiDCTFreeWrap<IppiDCTFwdSpec_32f, ippiDCTFwdFree_32f>;
        ippiDCTGetBufSize ippGetBufSize = inv ? ippiDCTGetBufSizeWrap<IppiDCTInvSpec_32f, ippiDCTInvGetBufSize_32f> : ippiDCTGetBufSizeWrap<IppiDCTFwdSpec_32f, ippiDCTFwdGetBufSize_32f>;

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
    DCTFunc dct_func;
    // per-stage plan + workspace, (re)built once per stage length, never per row
    DftPlan plan;
    AutoBuffer<uchar> ws;
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
            dctWrap<float, DCT<float>>,
            dctWrap<float, IDCT<float>>,
            dctWrap<double, DCT<double>>,
            dctWrap<double, IDCT<double>>
        };
        dct_func = dct_tbl[(int)isInverse + (depth == CV_64F)*2];
        opt.isComplex = false;
        opt.isInverse = false;
        opt.scale = 1.;

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

        int prev_len = 0;
        int elem_size = (depth == CV_32F) ? sizeof(float) : sizeof(double);
        const DFTKernels& kernels = getDFTKernels(depth);

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

            if( len != prev_len )
            {
                if( len > 1 && (len & 1) )
                    CV_Error( cv::Error::StsNotImplemented, "Odd-size DCT\'s are not implemented" );
                if( len > 1 )
                {
                    plan.build(isInverse ? DFT_KIND_IDCT : DFT_KIND_DCT, len, depth, kernels.vlanes);
                    ws.allocate(plan.ws_bytes);
                }
                opt.plan = &plan;
                opt.kernels = &kernels;
                opt.workspace = ws.data();
                prev_len = len;
            }
            // otherwise reuse the plan built on the previous stage (same length, only the steps differ)
            if( len > 1 && count > 1 && (size_t)count*len >= DFT_PARALLEL_MIN_ELEMS )
            {
                // the plan is shared, every worker gets its own workspace once per range
                auto processLines = [&](const Range& range)
                {
                    OcvDftOptions opt_ = opt;
                    AutoBuffer<uchar> ws_(plan.ws_bytes);
                    opt_.workspace = ws_.data();
                    for( int i = range.start; i < range.end; i++ )
                        dct_func( opt_, sptr + i*sstep0, sstep1, dptr + i*dstep0, dstep1 );
                };
                parallel_for_(Range(0, count), processLines, (double)count*len/DFT_PARALLEL_MIN_ELEMS);
            }
            else
            {
                for(unsigned i = 0; i < static_cast<unsigned>(count); i++ )
                    dct_func( opt, sptr + i*sstep0, sstep1, dptr + i*dstep0, dstep1 );
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
/* End of file. */
