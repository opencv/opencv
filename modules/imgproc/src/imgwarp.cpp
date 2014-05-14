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

/* ////////////////////////////////////////////////////////////////////
//
//  Geometrical transforms on images and matrices: rotation, zoom etc.
//
// */

#include "precomp.hpp"
#include "opencl_kernels.hpp"

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
static IppStatus sts = ippInit();
#endif

namespace cv
{
#if IPP_VERSION_X100 >= 701
    typedef IppStatus (CV_STDCALL* ippiResizeFunc)(const void*, int, const void*, int, IppiPoint, IppiSize, IppiBorderType, void*, void*, Ipp8u*);
    typedef IppStatus (CV_STDCALL* ippiResizeGetBufferSize)(void*, IppiSize, Ipp32u, int*);
    typedef IppStatus (CV_STDCALL* ippiResizeGetSrcOffset)(void*, IppiPoint, IppiPoint*);
#endif

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7) && 0
    typedef IppStatus (CV_STDCALL* ippiSetFunc)(const void*, void *, int, IppiSize);
    typedef IppStatus (CV_STDCALL* ippiWarpPerspectiveFunc)(const void*, IppiSize, int, IppiRect, void *, int, IppiRect, double [3][3], int);
    typedef IppStatus (CV_STDCALL* ippiWarpAffineBackFunc)(const void*, IppiSize, int, IppiRect, void *, int, IppiRect, double [2][3], int);

    template <int channels, typename Type>
    bool IPPSetSimple(cv::Scalar value, void *dataPointer, int step, IppiSize &size, ippiSetFunc func)
    {
        Type values[channels];
        for( int i = 0; i < channels; i++ )
            values[i] = saturate_cast<Type>(value[i]);
        return func(values, dataPointer, step, size) >= 0;
    }

    static bool IPPSet(const cv::Scalar &value, void *dataPointer, int step, IppiSize &size, int channels, int depth)
    {
        if( channels == 1 )
        {
            switch( depth )
            {
            case CV_8U:
                return ippiSet_8u_C1R(saturate_cast<Ipp8u>(value[0]), (Ipp8u *)dataPointer, step, size) >= 0;
            case CV_16U:
                return ippiSet_16u_C1R(saturate_cast<Ipp16u>(value[0]), (Ipp16u *)dataPointer, step, size) >= 0;
            case CV_32F:
                return ippiSet_32f_C1R(saturate_cast<Ipp32f>(value[0]), (Ipp32f *)dataPointer, step, size) >= 0;
            }
        }
        else
        {
            if( channels == 3 )
            {
                switch( depth )
                {
                case CV_8U:
                    return IPPSetSimple<3, Ipp8u>(value, dataPointer, step, size, (ippiSetFunc)ippiSet_8u_C3R);
                case CV_16U:
                    return IPPSetSimple<3, Ipp16u>(value, dataPointer, step, size, (ippiSetFunc)ippiSet_16u_C3R);
                case CV_32F:
                    return IPPSetSimple<3, Ipp32f>(value, dataPointer, step, size, (ippiSetFunc)ippiSet_32f_C3R);
                }
            }
            else if( channels == 4 )
            {
                switch( depth )
                {
                case CV_8U:
                    return IPPSetSimple<4, Ipp8u>(value, dataPointer, step, size, (ippiSetFunc)ippiSet_8u_C4R);
                case CV_16U:
                    return IPPSetSimple<4, Ipp16u>(value, dataPointer, step, size, (ippiSetFunc)ippiSet_16u_C4R);
                case CV_32F:
                    return IPPSetSimple<4, Ipp32f>(value, dataPointer, step, size, (ippiSetFunc)ippiSet_32f_C4R);
                }
            }
        }
        return false;
    }
#endif

/************** interpolation formulas and tables ***************/

const int INTER_RESIZE_COEF_BITS=11;
const int INTER_RESIZE_COEF_SCALE=1 << INTER_RESIZE_COEF_BITS;

const int INTER_REMAP_COEF_BITS=15;
const int INTER_REMAP_COEF_SCALE=1 << INTER_REMAP_COEF_BITS;

static uchar NNDeltaTab_i[INTER_TAB_SIZE2][2];

static float BilinearTab_f[INTER_TAB_SIZE2][2][2];
static short BilinearTab_i[INTER_TAB_SIZE2][2][2];

#if CV_SSE2
static short BilinearTab_iC4_buf[INTER_TAB_SIZE2+2][2][8];
static short (*BilinearTab_iC4)[2][8] = (short (*)[2][8])alignPtr(BilinearTab_iC4_buf, 16);
#endif

static float BicubicTab_f[INTER_TAB_SIZE2][4][4];
static short BicubicTab_i[INTER_TAB_SIZE2][4][4];

static float Lanczos4Tab_f[INTER_TAB_SIZE2][8][8];
static short Lanczos4Tab_i[INTER_TAB_SIZE2][8][8];

static inline void interpolateLinear( float x, float* coeffs )
{
    coeffs[0] = 1.f - x;
    coeffs[1] = x;
}

static inline void interpolateCubic( float x, float* coeffs )
{
    const float A = -0.75f;

    coeffs[0] = ((A*(x + 1) - 5*A)*(x + 1) + 8*A)*(x + 1) - 4*A;
    coeffs[1] = ((A + 2)*x - (A + 3))*x*x + 1;
    coeffs[2] = ((A + 2)*(1 - x) - (A + 3))*(1 - x)*(1 - x) + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

static inline void interpolateLanczos4( float x, float* coeffs )
{
    static const double s45 = 0.70710678118654752440084436210485;
    static const double cs[][2]=
    {{1, 0}, {-s45, -s45}, {0, 1}, {s45, -s45}, {-1, 0}, {s45, s45}, {0, -1}, {-s45, s45}};

    if( x < FLT_EPSILON )
    {
        for( int i = 0; i < 8; i++ )
            coeffs[i] = 0;
        coeffs[3] = 1;
        return;
    }

    float sum = 0;
    double y0=-(x+3)*CV_PI*0.25, s0 = sin(y0), c0=cos(y0);
    for(int i = 0; i < 8; i++ )
    {
        double y = -(x+3-i)*CV_PI*0.25;
        coeffs[i] = (float)((cs[i][0]*s0 + cs[i][1]*c0)/(y*y));
        sum += coeffs[i];
    }

    sum = 1.f/sum;
    for(int i = 0; i < 8; i++ )
        coeffs[i] *= sum;
}

static void initInterTab1D(int method, float* tab, int tabsz)
{
    float scale = 1.f/tabsz;
    if( method == INTER_LINEAR )
    {
        for( int i = 0; i < tabsz; i++, tab += 2 )
            interpolateLinear( i*scale, tab );
    }
    else if( method == INTER_CUBIC )
    {
        for( int i = 0; i < tabsz; i++, tab += 4 )
            interpolateCubic( i*scale, tab );
    }
    else if( method == INTER_LANCZOS4 )
    {
        for( int i = 0; i < tabsz; i++, tab += 8 )
            interpolateLanczos4( i*scale, tab );
    }
    else
        CV_Error( CV_StsBadArg, "Unknown interpolation method" );
}


static const void* initInterTab2D( int method, bool fixpt )
{
    static bool inittab[INTER_MAX+1] = {false};
    float* tab = 0;
    short* itab = 0;
    int ksize = 0;
    if( method == INTER_LINEAR )
        tab = BilinearTab_f[0][0], itab = BilinearTab_i[0][0], ksize=2;
    else if( method == INTER_CUBIC )
        tab = BicubicTab_f[0][0], itab = BicubicTab_i[0][0], ksize=4;
    else if( method == INTER_LANCZOS4 )
        tab = Lanczos4Tab_f[0][0], itab = Lanczos4Tab_i[0][0], ksize=8;
    else
        CV_Error( CV_StsBadArg, "Unknown/unsupported interpolation type" );

    if( !inittab[method] )
    {
        AutoBuffer<float> _tab(8*INTER_TAB_SIZE);
        int i, j, k1, k2;
        initInterTab1D(method, _tab, INTER_TAB_SIZE);
        for( i = 0; i < INTER_TAB_SIZE; i++ )
            for( j = 0; j < INTER_TAB_SIZE; j++, tab += ksize*ksize, itab += ksize*ksize )
            {
                int isum = 0;
                NNDeltaTab_i[i*INTER_TAB_SIZE+j][0] = j < INTER_TAB_SIZE/2;
                NNDeltaTab_i[i*INTER_TAB_SIZE+j][1] = i < INTER_TAB_SIZE/2;

                for( k1 = 0; k1 < ksize; k1++ )
                {
                    float vy = _tab[i*ksize + k1];
                    for( k2 = 0; k2 < ksize; k2++ )
                    {
                        float v = vy*_tab[j*ksize + k2];
                        tab[k1*ksize + k2] = v;
                        isum += itab[k1*ksize + k2] = saturate_cast<short>(v*INTER_REMAP_COEF_SCALE);
                    }
                }

                if( isum != INTER_REMAP_COEF_SCALE )
                {
                    int diff = isum - INTER_REMAP_COEF_SCALE;
                    int ksize2 = ksize/2, Mk1=ksize2, Mk2=ksize2, mk1=ksize2, mk2=ksize2;
                    for( k1 = ksize2; k1 < ksize2+2; k1++ )
                        for( k2 = ksize2; k2 < ksize2+2; k2++ )
                        {
                            if( itab[k1*ksize+k2] < itab[mk1*ksize+mk2] )
                                mk1 = k1, mk2 = k2;
                            else if( itab[k1*ksize+k2] > itab[Mk1*ksize+Mk2] )
                                Mk1 = k1, Mk2 = k2;
                        }
                    if( diff < 0 )
                        itab[Mk1*ksize + Mk2] = (short)(itab[Mk1*ksize + Mk2] - diff);
                    else
                        itab[mk1*ksize + mk2] = (short)(itab[mk1*ksize + mk2] - diff);
                }
            }
        tab -= INTER_TAB_SIZE2*ksize*ksize;
        itab -= INTER_TAB_SIZE2*ksize*ksize;
#if CV_SSE2
        if( method == INTER_LINEAR )
        {
            for( i = 0; i < INTER_TAB_SIZE2; i++ )
                for( j = 0; j < 4; j++ )
                {
                    BilinearTab_iC4[i][0][j*2] = BilinearTab_i[i][0][0];
                    BilinearTab_iC4[i][0][j*2+1] = BilinearTab_i[i][0][1];
                    BilinearTab_iC4[i][1][j*2] = BilinearTab_i[i][1][0];
                    BilinearTab_iC4[i][1][j*2+1] = BilinearTab_i[i][1][1];
                }
        }
#endif
        inittab[method] = true;
    }
    return fixpt ? (const void*)itab : (const void*)tab;
}

#ifndef __MINGW32__
static bool initAllInterTab2D()
{
    return  initInterTab2D( INTER_LINEAR, false ) &&
            initInterTab2D( INTER_LINEAR, true ) &&
            initInterTab2D( INTER_CUBIC, false ) &&
            initInterTab2D( INTER_CUBIC, true ) &&
            initInterTab2D( INTER_LANCZOS4, false ) &&
            initInterTab2D( INTER_LANCZOS4, true );
}

static volatile bool doInitAllInterTab2D = initAllInterTab2D();
#endif

template<typename ST, typename DT> struct Cast
{
    typedef ST type1;
    typedef DT rtype;

    DT operator()(ST val) const { return saturate_cast<DT>(val); }
};

template<typename ST, typename DT, int bits> struct FixedPtCast
{
    typedef ST type1;
    typedef DT rtype;
    enum { SHIFT = bits, DELTA = 1 << (bits-1) };

    DT operator()(ST val) const { return saturate_cast<DT>((val + DELTA)>>SHIFT); }
};

/****************************************************************************************\
*                                         Resize                                         *
\****************************************************************************************/

class resizeNNInvoker :
    public ParallelLoopBody
{
public:
    resizeNNInvoker(const Mat& _src, Mat &_dst, int *_x_ofs, int _pix_size4, double _ify) :
        ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs), pix_size4(_pix_size4),
        ify(_ify)
    {
    }

    virtual void operator() (const Range& range) const
    {
        Size ssize = src.size(), dsize = dst.size();
        int y, x, pix_size = (int)src.elemSize();

        for( y = range.start; y < range.end; y++ )
        {
            uchar* D = dst.data + dst.step*y;
            int sy = std::min(cvFloor(y*ify), ssize.height-1);
            const uchar* S = src.data + src.step*sy;

            switch( pix_size )
            {
            case 1:
                for( x = 0; x <= dsize.width - 2; x += 2 )
                {
                    uchar t0 = S[x_ofs[x]];
                    uchar t1 = S[x_ofs[x+1]];
                    D[x] = t0;
                    D[x+1] = t1;
                }

                for( ; x < dsize.width; x++ )
                    D[x] = S[x_ofs[x]];
                break;
            case 2:
                for( x = 0; x < dsize.width; x++ )
                    *(ushort*)(D + x*2) = *(ushort*)(S + x_ofs[x]);
                break;
            case 3:
                for( x = 0; x < dsize.width; x++, D += 3 )
                {
                    const uchar* _tS = S + x_ofs[x];
                    D[0] = _tS[0]; D[1] = _tS[1]; D[2] = _tS[2];
                }
                break;
            case 4:
                for( x = 0; x < dsize.width; x++ )
                    *(int*)(D + x*4) = *(int*)(S + x_ofs[x]);
                break;
            case 6:
                for( x = 0; x < dsize.width; x++, D += 6 )
                {
                    const ushort* _tS = (const ushort*)(S + x_ofs[x]);
                    ushort* _tD = (ushort*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2];
                }
                break;
            case 8:
                for( x = 0; x < dsize.width; x++, D += 8 )
                {
                    const int* _tS = (const int*)(S + x_ofs[x]);
                    int* _tD = (int*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1];
                }
                break;
            case 12:
                for( x = 0; x < dsize.width; x++, D += 12 )
                {
                    const int* _tS = (const int*)(S + x_ofs[x]);
                    int* _tD = (int*)D;
                    _tD[0] = _tS[0]; _tD[1] = _tS[1]; _tD[2] = _tS[2];
                }
                break;
            default:
                for( x = 0; x < dsize.width; x++, D += pix_size )
                {
                    const int* _tS = (const int*)(S + x_ofs[x]);
                    int* _tD = (int*)D;
                    for( int k = 0; k < pix_size4; k++ )
                        _tD[k] = _tS[k];
                }
            }
        }
    }

private:
    const Mat src;
    Mat dst;
    int* x_ofs, pix_size4;
    double ify;

    resizeNNInvoker(const resizeNNInvoker&);
    resizeNNInvoker& operator=(const resizeNNInvoker&);
};

static void
resizeNN( const Mat& src, Mat& dst, double fx, double fy )
{
    Size ssize = src.size(), dsize = dst.size();
    AutoBuffer<int> _x_ofs(dsize.width);
    int* x_ofs = _x_ofs;
    int pix_size = (int)src.elemSize();
    int pix_size4 = (int)(pix_size / sizeof(int));
    double ifx = 1./fx, ify = 1./fy;
    int x;

    for( x = 0; x < dsize.width; x++ )
    {
        int sx = cvFloor(x*ifx);
        x_ofs[x] = std::min(sx, ssize.width-1)*pix_size;
    }

    Range range(0, dsize.height);
    resizeNNInvoker invoker(src, dst, x_ofs, pix_size4, ify);
    parallel_for_(range, invoker, dst.total()/(double)(1<<16));
}


struct VResizeNoVec
{
    int operator()(const uchar**, uchar*, const uchar*, int ) const { return 0; }
};

struct HResizeNoVec
{
    int operator()(const uchar**, uchar**, int, const int*,
        const uchar*, int, int, int, int, int) const { return 0; }
};

#if CV_SSE2

struct VResizeLinearVec_32s8u
{
    int operator()(const uchar** _src, uchar* dst, const uchar* _beta, int width ) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;

        const int** src = (const int**)_src;
        const short* beta = (const short*)_beta;
        const int *S0 = src[0], *S1 = src[1];
        int x = 0;
        __m128i b0 = _mm_set1_epi16(beta[0]), b1 = _mm_set1_epi16(beta[1]);
        __m128i delta = _mm_set1_epi16(2);

        if( (((size_t)S0|(size_t)S1)&15) == 0 )
            for( ; x <= width - 16; x += 16 )
            {
                __m128i x0, x1, x2, y0, y1, y2;
                x0 = _mm_load_si128((const __m128i*)(S0 + x));
                x1 = _mm_load_si128((const __m128i*)(S0 + x + 4));
                y0 = _mm_load_si128((const __m128i*)(S1 + x));
                y1 = _mm_load_si128((const __m128i*)(S1 + x + 4));
                x0 = _mm_packs_epi32(_mm_srai_epi32(x0, 4), _mm_srai_epi32(x1, 4));
                y0 = _mm_packs_epi32(_mm_srai_epi32(y0, 4), _mm_srai_epi32(y1, 4));

                x1 = _mm_load_si128((const __m128i*)(S0 + x + 8));
                x2 = _mm_load_si128((const __m128i*)(S0 + x + 12));
                y1 = _mm_load_si128((const __m128i*)(S1 + x + 8));
                y2 = _mm_load_si128((const __m128i*)(S1 + x + 12));
                x1 = _mm_packs_epi32(_mm_srai_epi32(x1, 4), _mm_srai_epi32(x2, 4));
                y1 = _mm_packs_epi32(_mm_srai_epi32(y1, 4), _mm_srai_epi32(y2, 4));

                x0 = _mm_adds_epi16(_mm_mulhi_epi16( x0, b0 ), _mm_mulhi_epi16( y0, b1 ));
                x1 = _mm_adds_epi16(_mm_mulhi_epi16( x1, b0 ), _mm_mulhi_epi16( y1, b1 ));

                x0 = _mm_srai_epi16(_mm_adds_epi16(x0, delta), 2);
                x1 = _mm_srai_epi16(_mm_adds_epi16(x1, delta), 2);
                _mm_storeu_si128( (__m128i*)(dst + x), _mm_packus_epi16(x0, x1));
            }
        else
            for( ; x <= width - 16; x += 16 )
            {
                __m128i x0, x1, x2, y0, y1, y2;
                x0 = _mm_loadu_si128((const __m128i*)(S0 + x));
                x1 = _mm_loadu_si128((const __m128i*)(S0 + x + 4));
                y0 = _mm_loadu_si128((const __m128i*)(S1 + x));
                y1 = _mm_loadu_si128((const __m128i*)(S1 + x + 4));
                x0 = _mm_packs_epi32(_mm_srai_epi32(x0, 4), _mm_srai_epi32(x1, 4));
                y0 = _mm_packs_epi32(_mm_srai_epi32(y0, 4), _mm_srai_epi32(y1, 4));

                x1 = _mm_loadu_si128((const __m128i*)(S0 + x + 8));
                x2 = _mm_loadu_si128((const __m128i*)(S0 + x + 12));
                y1 = _mm_loadu_si128((const __m128i*)(S1 + x + 8));
                y2 = _mm_loadu_si128((const __m128i*)(S1 + x + 12));
                x1 = _mm_packs_epi32(_mm_srai_epi32(x1, 4), _mm_srai_epi32(x2, 4));
                y1 = _mm_packs_epi32(_mm_srai_epi32(y1, 4), _mm_srai_epi32(y2, 4));

                x0 = _mm_adds_epi16(_mm_mulhi_epi16( x0, b0 ), _mm_mulhi_epi16( y0, b1 ));
                x1 = _mm_adds_epi16(_mm_mulhi_epi16( x1, b0 ), _mm_mulhi_epi16( y1, b1 ));

                x0 = _mm_srai_epi16(_mm_adds_epi16(x0, delta), 2);
                x1 = _mm_srai_epi16(_mm_adds_epi16(x1, delta), 2);
                _mm_storeu_si128( (__m128i*)(dst + x), _mm_packus_epi16(x0, x1));
            }

        for( ; x < width - 4; x += 4 )
        {
            __m128i x0, y0;
            x0 = _mm_srai_epi32(_mm_loadu_si128((const __m128i*)(S0 + x)), 4);
            y0 = _mm_srai_epi32(_mm_loadu_si128((const __m128i*)(S1 + x)), 4);
            x0 = _mm_packs_epi32(x0, x0);
            y0 = _mm_packs_epi32(y0, y0);
            x0 = _mm_adds_epi16(_mm_mulhi_epi16(x0, b0), _mm_mulhi_epi16(y0, b1));
            x0 = _mm_srai_epi16(_mm_adds_epi16(x0, delta), 2);
            x0 = _mm_packus_epi16(x0, x0);
            *(int*)(dst + x) = _mm_cvtsi128_si32(x0);
        }

        return x;
    }
};


template<int shiftval> struct VResizeLinearVec_32f16
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width ) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;

        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1];
        ushort* dst = (ushort*)_dst;
        int x = 0;

        __m128 b0 = _mm_set1_ps(beta[0]), b1 = _mm_set1_ps(beta[1]);
        __m128i preshift = _mm_set1_epi32(shiftval);
        __m128i postshift = _mm_set1_epi16((short)shiftval);

        if( (((size_t)S0|(size_t)S1)&15) == 0 )
            for( ; x <= width - 16; x += 16 )
            {
                __m128 x0, x1, y0, y1;
                __m128i t0, t1, t2;
                x0 = _mm_load_ps(S0 + x);
                x1 = _mm_load_ps(S0 + x + 4);
                y0 = _mm_load_ps(S1 + x);
                y1 = _mm_load_ps(S1 + x + 4);

                x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
                x1 = _mm_add_ps(_mm_mul_ps(x1, b0), _mm_mul_ps(y1, b1));
                t0 = _mm_add_epi32(_mm_cvtps_epi32(x0), preshift);
                t2 = _mm_add_epi32(_mm_cvtps_epi32(x1), preshift);
                t0 = _mm_add_epi16(_mm_packs_epi32(t0, t2), postshift);

                x0 = _mm_load_ps(S0 + x + 8);
                x1 = _mm_load_ps(S0 + x + 12);
                y0 = _mm_load_ps(S1 + x + 8);
                y1 = _mm_load_ps(S1 + x + 12);

                x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
                x1 = _mm_add_ps(_mm_mul_ps(x1, b0), _mm_mul_ps(y1, b1));
                t1 = _mm_add_epi32(_mm_cvtps_epi32(x0), preshift);
                t2 = _mm_add_epi32(_mm_cvtps_epi32(x1), preshift);
                t1 = _mm_add_epi16(_mm_packs_epi32(t1, t2), postshift);

                _mm_storeu_si128( (__m128i*)(dst + x), t0);
                _mm_storeu_si128( (__m128i*)(dst + x + 8), t1);
            }
        else
            for( ; x <= width - 16; x += 16 )
            {
                __m128 x0, x1, y0, y1;
                __m128i t0, t1, t2;
                x0 = _mm_loadu_ps(S0 + x);
                x1 = _mm_loadu_ps(S0 + x + 4);
                y0 = _mm_loadu_ps(S1 + x);
                y1 = _mm_loadu_ps(S1 + x + 4);

                x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
                x1 = _mm_add_ps(_mm_mul_ps(x1, b0), _mm_mul_ps(y1, b1));
                t0 = _mm_add_epi32(_mm_cvtps_epi32(x0), preshift);
                t2 = _mm_add_epi32(_mm_cvtps_epi32(x1), preshift);
                t0 = _mm_add_epi16(_mm_packs_epi32(t0, t2), postshift);

                x0 = _mm_loadu_ps(S0 + x + 8);
                x1 = _mm_loadu_ps(S0 + x + 12);
                y0 = _mm_loadu_ps(S1 + x + 8);
                y1 = _mm_loadu_ps(S1 + x + 12);

                x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
                x1 = _mm_add_ps(_mm_mul_ps(x1, b0), _mm_mul_ps(y1, b1));
                t1 = _mm_add_epi32(_mm_cvtps_epi32(x0), preshift);
                t2 = _mm_add_epi32(_mm_cvtps_epi32(x1), preshift);
                t1 = _mm_add_epi16(_mm_packs_epi32(t1, t2), postshift);

                _mm_storeu_si128( (__m128i*)(dst + x), t0);
                _mm_storeu_si128( (__m128i*)(dst + x + 8), t1);
            }

        for( ; x < width - 4; x += 4 )
        {
            __m128 x0, y0;
            __m128i t0;
            x0 = _mm_loadu_ps(S0 + x);
            y0 = _mm_loadu_ps(S1 + x);

            x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
            t0 = _mm_add_epi32(_mm_cvtps_epi32(x0), preshift);
            t0 = _mm_add_epi16(_mm_packs_epi32(t0, t0), postshift);
            _mm_storel_epi64( (__m128i*)(dst + x), t0);
        }

        return x;
    }
};

typedef VResizeLinearVec_32f16<SHRT_MIN> VResizeLinearVec_32f16u;
typedef VResizeLinearVec_32f16<0> VResizeLinearVec_32f16s;

struct VResizeLinearVec_32f
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width ) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE) )
            return 0;

        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1];
        float* dst = (float*)_dst;
        int x = 0;

        __m128 b0 = _mm_set1_ps(beta[0]), b1 = _mm_set1_ps(beta[1]);

        if( (((size_t)S0|(size_t)S1)&15) == 0 )
            for( ; x <= width - 8; x += 8 )
            {
                __m128 x0, x1, y0, y1;
                x0 = _mm_load_ps(S0 + x);
                x1 = _mm_load_ps(S0 + x + 4);
                y0 = _mm_load_ps(S1 + x);
                y1 = _mm_load_ps(S1 + x + 4);

                x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
                x1 = _mm_add_ps(_mm_mul_ps(x1, b0), _mm_mul_ps(y1, b1));

                _mm_storeu_ps( dst + x, x0);
                _mm_storeu_ps( dst + x + 4, x1);
            }
        else
            for( ; x <= width - 8; x += 8 )
            {
                __m128 x0, x1, y0, y1;
                x0 = _mm_loadu_ps(S0 + x);
                x1 = _mm_loadu_ps(S0 + x + 4);
                y0 = _mm_loadu_ps(S1 + x);
                y1 = _mm_loadu_ps(S1 + x + 4);

                x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
                x1 = _mm_add_ps(_mm_mul_ps(x1, b0), _mm_mul_ps(y1, b1));

                _mm_storeu_ps( dst + x, x0);
                _mm_storeu_ps( dst + x + 4, x1);
            }

        return x;
    }
};


struct VResizeCubicVec_32s8u
{
    int operator()(const uchar** _src, uchar* dst, const uchar* _beta, int width ) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;

        const int** src = (const int**)_src;
        const short* beta = (const short*)_beta;
        const int *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        int x = 0;
        float scale = 1.f/(INTER_RESIZE_COEF_SCALE*INTER_RESIZE_COEF_SCALE);
        __m128 b0 = _mm_set1_ps(beta[0]*scale), b1 = _mm_set1_ps(beta[1]*scale),
            b2 = _mm_set1_ps(beta[2]*scale), b3 = _mm_set1_ps(beta[3]*scale);

        if( (((size_t)S0|(size_t)S1|(size_t)S2|(size_t)S3)&15) == 0 )
            for( ; x <= width - 8; x += 8 )
            {
                __m128i x0, x1, y0, y1;
                __m128 s0, s1, f0, f1;
                x0 = _mm_load_si128((const __m128i*)(S0 + x));
                x1 = _mm_load_si128((const __m128i*)(S0 + x + 4));
                y0 = _mm_load_si128((const __m128i*)(S1 + x));
                y1 = _mm_load_si128((const __m128i*)(S1 + x + 4));

                s0 = _mm_mul_ps(_mm_cvtepi32_ps(x0), b0);
                s1 = _mm_mul_ps(_mm_cvtepi32_ps(x1), b0);
                f0 = _mm_mul_ps(_mm_cvtepi32_ps(y0), b1);
                f1 = _mm_mul_ps(_mm_cvtepi32_ps(y1), b1);
                s0 = _mm_add_ps(s0, f0);
                s1 = _mm_add_ps(s1, f1);

                x0 = _mm_load_si128((const __m128i*)(S2 + x));
                x1 = _mm_load_si128((const __m128i*)(S2 + x + 4));
                y0 = _mm_load_si128((const __m128i*)(S3 + x));
                y1 = _mm_load_si128((const __m128i*)(S3 + x + 4));

                f0 = _mm_mul_ps(_mm_cvtepi32_ps(x0), b2);
                f1 = _mm_mul_ps(_mm_cvtepi32_ps(x1), b2);
                s0 = _mm_add_ps(s0, f0);
                s1 = _mm_add_ps(s1, f1);
                f0 = _mm_mul_ps(_mm_cvtepi32_ps(y0), b3);
                f1 = _mm_mul_ps(_mm_cvtepi32_ps(y1), b3);
                s0 = _mm_add_ps(s0, f0);
                s1 = _mm_add_ps(s1, f1);

                x0 = _mm_cvtps_epi32(s0);
                x1 = _mm_cvtps_epi32(s1);

                x0 = _mm_packs_epi32(x0, x1);
                _mm_storel_epi64( (__m128i*)(dst + x), _mm_packus_epi16(x0, x0));
            }
        else
            for( ; x <= width - 8; x += 8 )
            {
                __m128i x0, x1, y0, y1;
                __m128 s0, s1, f0, f1;
                x0 = _mm_loadu_si128((const __m128i*)(S0 + x));
                x1 = _mm_loadu_si128((const __m128i*)(S0 + x + 4));
                y0 = _mm_loadu_si128((const __m128i*)(S1 + x));
                y1 = _mm_loadu_si128((const __m128i*)(S1 + x + 4));

                s0 = _mm_mul_ps(_mm_cvtepi32_ps(x0), b0);
                s1 = _mm_mul_ps(_mm_cvtepi32_ps(x1), b0);
                f0 = _mm_mul_ps(_mm_cvtepi32_ps(y0), b1);
                f1 = _mm_mul_ps(_mm_cvtepi32_ps(y1), b1);
                s0 = _mm_add_ps(s0, f0);
                s1 = _mm_add_ps(s1, f1);

                x0 = _mm_loadu_si128((const __m128i*)(S2 + x));
                x1 = _mm_loadu_si128((const __m128i*)(S2 + x + 4));
                y0 = _mm_loadu_si128((const __m128i*)(S3 + x));
                y1 = _mm_loadu_si128((const __m128i*)(S3 + x + 4));

                f0 = _mm_mul_ps(_mm_cvtepi32_ps(x0), b2);
                f1 = _mm_mul_ps(_mm_cvtepi32_ps(x1), b2);
                s0 = _mm_add_ps(s0, f0);
                s1 = _mm_add_ps(s1, f1);
                f0 = _mm_mul_ps(_mm_cvtepi32_ps(y0), b3);
                f1 = _mm_mul_ps(_mm_cvtepi32_ps(y1), b3);
                s0 = _mm_add_ps(s0, f0);
                s1 = _mm_add_ps(s1, f1);

                x0 = _mm_cvtps_epi32(s0);
                x1 = _mm_cvtps_epi32(s1);

                x0 = _mm_packs_epi32(x0, x1);
                _mm_storel_epi64( (__m128i*)(dst + x), _mm_packus_epi16(x0, x0));
            }

        return x;
    }
};


template<int shiftval> struct VResizeCubicVec_32f16
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width ) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;

        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        ushort* dst = (ushort*)_dst;
        int x = 0;
        __m128 b0 = _mm_set1_ps(beta[0]), b1 = _mm_set1_ps(beta[1]),
            b2 = _mm_set1_ps(beta[2]), b3 = _mm_set1_ps(beta[3]);
        __m128i preshift = _mm_set1_epi32(shiftval);
        __m128i postshift = _mm_set1_epi16((short)shiftval);

        for( ; x <= width - 8; x += 8 )
        {
            __m128 x0, x1, y0, y1, s0, s1;
            __m128i t0, t1;
            x0 = _mm_loadu_ps(S0 + x);
            x1 = _mm_loadu_ps(S0 + x + 4);
            y0 = _mm_loadu_ps(S1 + x);
            y1 = _mm_loadu_ps(S1 + x + 4);

            s0 = _mm_mul_ps(x0, b0);
            s1 = _mm_mul_ps(x1, b0);
            y0 = _mm_mul_ps(y0, b1);
            y1 = _mm_mul_ps(y1, b1);
            s0 = _mm_add_ps(s0, y0);
            s1 = _mm_add_ps(s1, y1);

            x0 = _mm_loadu_ps(S2 + x);
            x1 = _mm_loadu_ps(S2 + x + 4);
            y0 = _mm_loadu_ps(S3 + x);
            y1 = _mm_loadu_ps(S3 + x + 4);

            x0 = _mm_mul_ps(x0, b2);
            x1 = _mm_mul_ps(x1, b2);
            y0 = _mm_mul_ps(y0, b3);
            y1 = _mm_mul_ps(y1, b3);
            s0 = _mm_add_ps(s0, x0);
            s1 = _mm_add_ps(s1, x1);
            s0 = _mm_add_ps(s0, y0);
            s1 = _mm_add_ps(s1, y1);

            t0 = _mm_add_epi32(_mm_cvtps_epi32(s0), preshift);
            t1 = _mm_add_epi32(_mm_cvtps_epi32(s1), preshift);

            t0 = _mm_add_epi16(_mm_packs_epi32(t0, t1), postshift);
            _mm_storeu_si128( (__m128i*)(dst + x), t0);
        }

        return x;
    }
};

typedef VResizeCubicVec_32f16<SHRT_MIN> VResizeCubicVec_32f16u;
typedef VResizeCubicVec_32f16<0> VResizeCubicVec_32f16s;

struct VResizeCubicVec_32f
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width ) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE) )
            return 0;

        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        float* dst = (float*)_dst;
        int x = 0;
        __m128 b0 = _mm_set1_ps(beta[0]), b1 = _mm_set1_ps(beta[1]),
            b2 = _mm_set1_ps(beta[2]), b3 = _mm_set1_ps(beta[3]);

        for( ; x <= width - 8; x += 8 )
        {
            __m128 x0, x1, y0, y1, s0, s1;
            x0 = _mm_loadu_ps(S0 + x);
            x1 = _mm_loadu_ps(S0 + x + 4);
            y0 = _mm_loadu_ps(S1 + x);
            y1 = _mm_loadu_ps(S1 + x + 4);

            s0 = _mm_mul_ps(x0, b0);
            s1 = _mm_mul_ps(x1, b0);
            y0 = _mm_mul_ps(y0, b1);
            y1 = _mm_mul_ps(y1, b1);
            s0 = _mm_add_ps(s0, y0);
            s1 = _mm_add_ps(s1, y1);

            x0 = _mm_loadu_ps(S2 + x);
            x1 = _mm_loadu_ps(S2 + x + 4);
            y0 = _mm_loadu_ps(S3 + x);
            y1 = _mm_loadu_ps(S3 + x + 4);

            x0 = _mm_mul_ps(x0, b2);
            x1 = _mm_mul_ps(x1, b2);
            y0 = _mm_mul_ps(y0, b3);
            y1 = _mm_mul_ps(y1, b3);
            s0 = _mm_add_ps(s0, x0);
            s1 = _mm_add_ps(s1, x1);
            s0 = _mm_add_ps(s0, y0);
            s1 = _mm_add_ps(s1, y1);

            _mm_storeu_ps( dst + x, s0);
            _mm_storeu_ps( dst + x + 4, s1);
        }

        return x;
    }
};

#else

typedef VResizeNoVec VResizeLinearVec_32s8u;
typedef VResizeNoVec VResizeLinearVec_32f16u;
typedef VResizeNoVec VResizeLinearVec_32f16s;
typedef VResizeNoVec VResizeLinearVec_32f;

typedef VResizeNoVec VResizeCubicVec_32s8u;
typedef VResizeNoVec VResizeCubicVec_32f16u;
typedef VResizeNoVec VResizeCubicVec_32f16s;
typedef VResizeNoVec VResizeCubicVec_32f;

#endif

typedef HResizeNoVec HResizeLinearVec_8u32s;
typedef HResizeNoVec HResizeLinearVec_16u32f;
typedef HResizeNoVec HResizeLinearVec_16s32f;
typedef HResizeNoVec HResizeLinearVec_32f;
typedef HResizeNoVec HResizeLinearVec_64f;


template<typename T, typename WT, typename AT, int ONE, class VecOp>
struct HResizeLinear
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const T** src, WT** dst, int count,
                    const int* xofs, const AT* alpha,
                    int swidth, int dwidth, int cn, int xmin, int xmax ) const
    {
        int dx, k;
        VecOp vecOp;

        int dx0 = vecOp((const uchar**)src, (uchar**)dst, count,
            xofs, (const uchar*)alpha, swidth, dwidth, cn, xmin, xmax );

        for( k = 0; k <= count - 2; k++ )
        {
            const T *S0 = src[k], *S1 = src[k+1];
            WT *D0 = dst[k], *D1 = dst[k+1];
            for( dx = dx0; dx < xmax; dx++ )
            {
                int sx = xofs[dx];
                WT a0 = alpha[dx*2], a1 = alpha[dx*2+1];
                WT t0 = S0[sx]*a0 + S0[sx + cn]*a1;
                WT t1 = S1[sx]*a0 + S1[sx + cn]*a1;
                D0[dx] = t0; D1[dx] = t1;
            }

            for( ; dx < dwidth; dx++ )
            {
                int sx = xofs[dx];
                D0[dx] = WT(S0[sx]*ONE); D1[dx] = WT(S1[sx]*ONE);
            }
        }

        for( ; k < count; k++ )
        {
            const T *S = src[k];
            WT *D = dst[k];
            for( dx = 0; dx < xmax; dx++ )
            {
                int sx = xofs[dx];
                D[dx] = S[sx]*alpha[dx*2] + S[sx+cn]*alpha[dx*2+1];
            }

            for( ; dx < dwidth; dx++ )
                D[dx] = WT(S[xofs[dx]]*ONE);
        }
    }
};


template<typename T, typename WT, typename AT, class CastOp, class VecOp>
struct VResizeLinear
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const WT** src, T* dst, const AT* beta, int width ) const
    {
        WT b0 = beta[0], b1 = beta[1];
        const WT *S0 = src[0], *S1 = src[1];
        CastOp castOp;
        VecOp vecOp;

        int x = vecOp((const uchar**)src, (uchar*)dst, (const uchar*)beta, width);
        #if CV_ENABLE_UNROLLED
        for( ; x <= width - 4; x += 4 )
        {
            WT t0, t1;
            t0 = S0[x]*b0 + S1[x]*b1;
            t1 = S0[x+1]*b0 + S1[x+1]*b1;
            dst[x] = castOp(t0); dst[x+1] = castOp(t1);
            t0 = S0[x+2]*b0 + S1[x+2]*b1;
            t1 = S0[x+3]*b0 + S1[x+3]*b1;
            dst[x+2] = castOp(t0); dst[x+3] = castOp(t1);
        }
        #endif
        for( ; x < width; x++ )
            dst[x] = castOp(S0[x]*b0 + S1[x]*b1);
    }
};

template<>
struct VResizeLinear<uchar, int, short, FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS*2>, VResizeLinearVec_32s8u>
{
    typedef uchar value_type;
    typedef int buf_type;
    typedef short alpha_type;

    void operator()(const buf_type** src, value_type* dst, const alpha_type* beta, int width ) const
    {
        alpha_type b0 = beta[0], b1 = beta[1];
        const buf_type *S0 = src[0], *S1 = src[1];
        VResizeLinearVec_32s8u vecOp;

        int x = vecOp((const uchar**)src, (uchar*)dst, (const uchar*)beta, width);
        #if CV_ENABLE_UNROLLED
        for( ; x <= width - 4; x += 4 )
        {
            dst[x+0] = uchar(( ((b0 * (S0[x+0] >> 4)) >> 16) + ((b1 * (S1[x+0] >> 4)) >> 16) + 2)>>2);
            dst[x+1] = uchar(( ((b0 * (S0[x+1] >> 4)) >> 16) + ((b1 * (S1[x+1] >> 4)) >> 16) + 2)>>2);
            dst[x+2] = uchar(( ((b0 * (S0[x+2] >> 4)) >> 16) + ((b1 * (S1[x+2] >> 4)) >> 16) + 2)>>2);
            dst[x+3] = uchar(( ((b0 * (S0[x+3] >> 4)) >> 16) + ((b1 * (S1[x+3] >> 4)) >> 16) + 2)>>2);
        }
        #endif
        for( ; x < width; x++ )
            dst[x] = uchar(( ((b0 * (S0[x] >> 4)) >> 16) + ((b1 * (S1[x] >> 4)) >> 16) + 2)>>2);
    }
};


template<typename T, typename WT, typename AT>
struct HResizeCubic
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const T** src, WT** dst, int count,
                    const int* xofs, const AT* alpha,
                    int swidth, int dwidth, int cn, int xmin, int xmax ) const
    {
        for( int k = 0; k < count; k++ )
        {
            const T *S = src[k];
            WT *D = dst[k];
            int dx = 0, limit = xmin;
            for(;;)
            {
                for( ; dx < limit; dx++, alpha += 4 )
                {
                    int j, sx = xofs[dx] - cn;
                    WT v = 0;
                    for( j = 0; j < 4; j++ )
                    {
                        int sxj = sx + j*cn;
                        if( (unsigned)sxj >= (unsigned)swidth )
                        {
                            while( sxj < 0 )
                                sxj += cn;
                            while( sxj >= swidth )
                                sxj -= cn;
                        }
                        v += S[sxj]*alpha[j];
                    }
                    D[dx] = v;
                }
                if( limit == dwidth )
                    break;
                for( ; dx < xmax; dx++, alpha += 4 )
                {
                    int sx = xofs[dx];
                    D[dx] = S[sx-cn]*alpha[0] + S[sx]*alpha[1] +
                        S[sx+cn]*alpha[2] + S[sx+cn*2]*alpha[3];
                }
                limit = dwidth;
            }
            alpha -= dwidth*4;
        }
    }
};


template<typename T, typename WT, typename AT, class CastOp, class VecOp>
struct VResizeCubic
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const WT** src, T* dst, const AT* beta, int width ) const
    {
        WT b0 = beta[0], b1 = beta[1], b2 = beta[2], b3 = beta[3];
        const WT *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        CastOp castOp;
        VecOp vecOp;

        int x = vecOp((const uchar**)src, (uchar*)dst, (const uchar*)beta, width);
        for( ; x < width; x++ )
            dst[x] = castOp(S0[x]*b0 + S1[x]*b1 + S2[x]*b2 + S3[x]*b3);
    }
};


template<typename T, typename WT, typename AT>
struct HResizeLanczos4
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const T** src, WT** dst, int count,
                    const int* xofs, const AT* alpha,
                    int swidth, int dwidth, int cn, int xmin, int xmax ) const
    {
        for( int k = 0; k < count; k++ )
        {
            const T *S = src[k];
            WT *D = dst[k];
            int dx = 0, limit = xmin;
            for(;;)
            {
                for( ; dx < limit; dx++, alpha += 8 )
                {
                    int j, sx = xofs[dx] - cn*3;
                    WT v = 0;
                    for( j = 0; j < 8; j++ )
                    {
                        int sxj = sx + j*cn;
                        if( (unsigned)sxj >= (unsigned)swidth )
                        {
                            while( sxj < 0 )
                                sxj += cn;
                            while( sxj >= swidth )
                                sxj -= cn;
                        }
                        v += S[sxj]*alpha[j];
                    }
                    D[dx] = v;
                }
                if( limit == dwidth )
                    break;
                for( ; dx < xmax; dx++, alpha += 8 )
                {
                    int sx = xofs[dx];
                    D[dx] = S[sx-cn*3]*alpha[0] + S[sx-cn*2]*alpha[1] +
                        S[sx-cn]*alpha[2] + S[sx]*alpha[3] +
                        S[sx+cn]*alpha[4] + S[sx+cn*2]*alpha[5] +
                        S[sx+cn*3]*alpha[6] + S[sx+cn*4]*alpha[7];
                }
                limit = dwidth;
            }
            alpha -= dwidth*8;
        }
    }
};


template<typename T, typename WT, typename AT, class CastOp, class VecOp>
struct VResizeLanczos4
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const WT** src, T* dst, const AT* beta, int width ) const
    {
        CastOp castOp;
        VecOp vecOp;
        int k, x = vecOp((const uchar**)src, (uchar*)dst, (const uchar*)beta, width);
        #if CV_ENABLE_UNROLLED
        for( ; x <= width - 4; x += 4 )
        {
            WT b = beta[0];
            const WT* S = src[0];
            WT s0 = S[x]*b, s1 = S[x+1]*b, s2 = S[x+2]*b, s3 = S[x+3]*b;

            for( k = 1; k < 8; k++ )
            {
                b = beta[k]; S = src[k];
                s0 += S[x]*b; s1 += S[x+1]*b;
                s2 += S[x+2]*b; s3 += S[x+3]*b;
            }

            dst[x] = castOp(s0); dst[x+1] = castOp(s1);
            dst[x+2] = castOp(s2); dst[x+3] = castOp(s3);
        }
        #endif
        for( ; x < width; x++ )
        {
            dst[x] = castOp(src[0][x]*beta[0] + src[1][x]*beta[1] +
                src[2][x]*beta[2] + src[3][x]*beta[3] + src[4][x]*beta[4] +
                src[5][x]*beta[5] + src[6][x]*beta[6] + src[7][x]*beta[7]);
        }
    }
};


static inline int clip(int x, int a, int b)
{
    return x >= a ? (x < b ? x : b-1) : a;
}

static const int MAX_ESIZE=16;

template <typename HResize, typename VResize>
class resizeGeneric_Invoker :
    public ParallelLoopBody
{
public:
    typedef typename HResize::value_type T;
    typedef typename HResize::buf_type WT;
    typedef typename HResize::alpha_type AT;

    resizeGeneric_Invoker(const Mat& _src, Mat &_dst, const int *_xofs, const int *_yofs,
        const AT* _alpha, const AT* __beta, const Size& _ssize, const Size &_dsize,
        int _ksize, int _xmin, int _xmax) :
        ParallelLoopBody(), src(_src), dst(_dst), xofs(_xofs), yofs(_yofs),
        alpha(_alpha), _beta(__beta), ssize(_ssize), dsize(_dsize),
        ksize(_ksize), xmin(_xmin), xmax(_xmax)
    {
        CV_Assert(ksize <= MAX_ESIZE);
    }

#if defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ == 8)
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Warray-bounds"
#endif
    virtual void operator() (const Range& range) const
    {
        int dy, cn = src.channels();
        HResize hresize;
        VResize vresize;

        int bufstep = (int)alignSize(dsize.width, 16);
        AutoBuffer<WT> _buffer(bufstep*ksize);
        const T* srows[MAX_ESIZE]={0};
        WT* rows[MAX_ESIZE]={0};
        int prev_sy[MAX_ESIZE];

        for(int k = 0; k < ksize; k++ )
        {
            prev_sy[k] = -1;
            rows[k] = (WT*)_buffer + bufstep*k;
        }

        const AT* beta = _beta + ksize * range.start;

        for( dy = range.start; dy < range.end; dy++, beta += ksize )
        {
            int sy0 = yofs[dy], k0=ksize, k1=0, ksize2 = ksize/2;

            for(int k = 0; k < ksize; k++ )
            {
                int sy = clip(sy0 - ksize2 + 1 + k, 0, ssize.height);
                for( k1 = std::max(k1, k); k1 < ksize; k1++ )
                {
                    if( sy == prev_sy[k1] ) // if the sy-th row has been computed already, reuse it.
                    {
                        if( k1 > k )
                            memcpy( rows[k], rows[k1], bufstep*sizeof(rows[0][0]) );
                        break;
                    }
                }
                if( k1 == ksize )
                    k0 = std::min(k0, k); // remember the first row that needs to be computed
                srows[k] = (T*)(src.data + src.step*sy);
                prev_sy[k] = sy;
            }

            if( k0 < ksize )
                hresize( (const T**)(srows + k0), (WT**)(rows + k0), ksize - k0, xofs, (const AT*)(alpha),
                        ssize.width, dsize.width, cn, xmin, xmax );
            vresize( (const WT**)rows, (T*)(dst.data + dst.step*dy), beta, dsize.width );
        }
    }
#if defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ == 8)
# pragma GCC diagnostic pop
#endif

private:
    Mat src;
    Mat dst;
    const int* xofs, *yofs;
    const AT* alpha, *_beta;
    Size ssize, dsize;
    const int ksize, xmin, xmax;

    resizeGeneric_Invoker& operator = (const resizeGeneric_Invoker&);
};

template<class HResize, class VResize>
static void resizeGeneric_( const Mat& src, Mat& dst,
                            const int* xofs, const void* _alpha,
                            const int* yofs, const void* _beta,
                            int xmin, int xmax, int ksize )
{
    typedef typename HResize::alpha_type AT;

    const AT* beta = (const AT*)_beta;
    Size ssize = src.size(), dsize = dst.size();
    int cn = src.channels();
    ssize.width *= cn;
    dsize.width *= cn;
    xmin *= cn;
    xmax *= cn;
    // image resize is a separable operation. In case of not too strong

    Range range(0, dsize.height);
    resizeGeneric_Invoker<HResize, VResize> invoker(src, dst, xofs, yofs, (const AT*)_alpha, beta,
        ssize, dsize, ksize, xmin, xmax);
    parallel_for_(range, invoker, dst.total()/(double)(1<<16));
}

template <typename T, typename WT>
struct ResizeAreaFastNoVec
{
    ResizeAreaFastNoVec(int, int) { }
    ResizeAreaFastNoVec(int, int, int, int) { }
    int operator() (const T*, T*, int) const
    { return 0; }
};

#if CV_SSE2
class ResizeAreaFastVec_SIMD_8u
{
public:
    ResizeAreaFastVec_SIMD_8u(int _cn, int _step) :
        cn(_cn), step(_step)
    {
        use_simd = checkHardwareSupport(CV_CPU_SSE2);
    }

    int operator() (const uchar* S, uchar* D, int w) const
    {
        if (!use_simd)
            return 0;

        int dx = 0;
        const uchar* S0 = S;
        const uchar* S1 = S0 + step;
        __m128i zero = _mm_setzero_si128();
        __m128i delta2 = _mm_set1_epi16(2);

        if (cn == 1)
        {
            __m128i masklow = _mm_set1_epi16(0x00ff);
            for ( ; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i s0 = _mm_add_epi16(_mm_srli_epi16(r0, 8), _mm_and_si128(r0, masklow));
                __m128i s1 = _mm_add_epi16(_mm_srli_epi16(r1, 8), _mm_and_si128(r1, masklow));
                s0 = _mm_add_epi16(_mm_add_epi16(s0, s1), delta2);
                s0 = _mm_packus_epi16(_mm_srli_epi16(s0, 2), zero);

                _mm_storel_epi64((__m128i*)D, s0);
            }
        }
        else if (cn == 3)
            for ( ; dx <= w - 11; dx += 6, S0 += 12, S1 += 12, D += 6)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i r0_16l = _mm_unpacklo_epi8(r0, zero);
                __m128i r0_16h = _mm_unpacklo_epi8(_mm_srli_si128(r0, 6), zero);
                __m128i r1_16l = _mm_unpacklo_epi8(r1, zero);
                __m128i r1_16h = _mm_unpacklo_epi8(_mm_srli_si128(r1, 6), zero);

                __m128i s0 = _mm_add_epi16(r0_16l, _mm_srli_si128(r0_16l, 6));
                __m128i s1 = _mm_add_epi16(r1_16l, _mm_srli_si128(r1_16l, 6));
                s0 = _mm_add_epi16(s1, _mm_add_epi16(s0, delta2));
                s0 = _mm_packus_epi16(_mm_srli_epi16(s0, 2), zero);
                _mm_storel_epi64((__m128i*)D, s0);

                s0 = _mm_add_epi16(r0_16h, _mm_srli_si128(r0_16h, 6));
                s1 = _mm_add_epi16(r1_16h, _mm_srli_si128(r1_16h, 6));
                s0 = _mm_add_epi16(s1, _mm_add_epi16(s0, delta2));
                s0 = _mm_packus_epi16(_mm_srli_epi16(s0, 2), zero);
                _mm_storel_epi64((__m128i*)(D+3), s0);
            }
        else
        {
            CV_Assert(cn == 4);
            int v[] = { 0, 0, -1, -1 };
            __m128i mask = _mm_loadu_si128((const __m128i*)v);

            for ( ; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i r0_16l = _mm_unpacklo_epi8(r0, zero);
                __m128i r0_16h = _mm_unpackhi_epi8(r0, zero);
                __m128i r1_16l = _mm_unpacklo_epi8(r1, zero);
                __m128i r1_16h = _mm_unpackhi_epi8(r1, zero);

                __m128i s0 = _mm_add_epi16(r0_16l, _mm_srli_si128(r0_16l, 8));
                __m128i s1 = _mm_add_epi16(r1_16l, _mm_srli_si128(r1_16l, 8));
                s0 = _mm_add_epi16(s1, _mm_add_epi16(s0, delta2));
                __m128i res0 = _mm_srli_epi16(s0, 2);

                s0 = _mm_add_epi16(r0_16h, _mm_srli_si128(r0_16h, 8));
                s1 = _mm_add_epi16(r1_16h, _mm_srli_si128(r1_16h, 8));
                s0 = _mm_add_epi16(s1, _mm_add_epi16(s0, delta2));
                __m128i res1 = _mm_srli_epi16(s0, 2);
                s0 = _mm_packus_epi16(_mm_or_si128(_mm_andnot_si128(mask, res0),
                                                   _mm_and_si128(mask, _mm_slli_si128(res1, 8))), zero);
                _mm_storel_epi64((__m128i*)(D), s0);
            }
        }

        return dx;
    }

private:
    int cn;
    bool use_simd;
    int step;
};

class ResizeAreaFastVec_SIMD_16u
{
public:
    ResizeAreaFastVec_SIMD_16u(int _cn, int _step) :
        cn(_cn), step(_step)
    {
        use_simd = checkHardwareSupport(CV_CPU_SSE2);
    }

    int operator() (const ushort* S, ushort* D, int w) const
    {
        if (!use_simd)
            return 0;

        int dx = 0;
        const ushort* S0 = (const ushort*)S;
        const ushort* S1 = (const ushort*)((const uchar*)(S) + step);
        __m128i masklow = _mm_set1_epi32(0x0000ffff);
        __m128i zero = _mm_setzero_si128();
        __m128i delta2 = _mm_set1_epi32(2);

#define _mm_packus_epi32(a, zero) _mm_packs_epi32(_mm_srai_epi32(_mm_slli_epi32(a, 16), 16), zero)

        if (cn == 1)
        {
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i s0 = _mm_add_epi32(_mm_srli_epi32(r0, 16), _mm_and_si128(r0, masklow));
                __m128i s1 = _mm_add_epi32(_mm_srli_epi32(r1, 16), _mm_and_si128(r1, masklow));
                s0 = _mm_add_epi32(_mm_add_epi32(s0, s1), delta2);
                s0 = _mm_srli_epi32(s0, 2);
                s0 = _mm_packus_epi32(s0, zero);

                _mm_storel_epi64((__m128i*)D, s0);
            }
        }
        else if (cn == 3)
            for ( ; dx <= w - 4; dx += 3, S0 += 6, S1 += 6, D += 3)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i r0_16l = _mm_unpacklo_epi16(r0, zero);
                __m128i r0_16h = _mm_unpacklo_epi16(_mm_srli_si128(r0, 6), zero);
                __m128i r1_16l = _mm_unpacklo_epi16(r1, zero);
                __m128i r1_16h = _mm_unpacklo_epi16(_mm_srli_si128(r1, 6), zero);

                __m128i s0 = _mm_add_epi32(r0_16l, r0_16h);
                __m128i s1 = _mm_add_epi32(r1_16l, r1_16h);
                s0 = _mm_add_epi32(delta2, _mm_add_epi32(s0, s1));
                s0 = _mm_packus_epi32(_mm_srli_epi32(s0, 2), zero);
                _mm_storel_epi64((__m128i*)D, s0);
            }
        else
        {
            CV_Assert(cn == 4);
            for ( ; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i r0_32l = _mm_unpacklo_epi16(r0, zero);
                __m128i r0_32h = _mm_unpackhi_epi16(r0, zero);
                __m128i r1_32l = _mm_unpacklo_epi16(r1, zero);
                __m128i r1_32h = _mm_unpackhi_epi16(r1, zero);

                __m128i s0 = _mm_add_epi32(r0_32l, r0_32h);
                __m128i s1 = _mm_add_epi32(r1_32l, r1_32h);
                s0 = _mm_add_epi32(s1, _mm_add_epi32(s0, delta2));
                s0 = _mm_packus_epi32(_mm_srli_epi32(s0, 2), zero);
                _mm_storel_epi64((__m128i*)D, s0);
            }
        }

#undef _mm_packus_epi32

        return dx;
    }

private:
    int cn;
    int step;
    bool use_simd;
};

#else
typedef ResizeAreaFastNoVec<uchar, uchar> ResizeAreaFastVec_SIMD_8u;
typedef ResizeAreaFastNoVec<ushort, ushort> ResizeAreaFastVec_SIMD_16u;
#endif

template<typename T, typename SIMDVecOp>
struct ResizeAreaFastVec
{
    ResizeAreaFastVec(int _scale_x, int _scale_y, int _cn, int _step) :
        scale_x(_scale_x), scale_y(_scale_y), cn(_cn), step(_step), vecOp(_cn, _step)
    {
        fast_mode = scale_x == 2 && scale_y == 2 && (cn == 1 || cn == 3 || cn == 4);
    }

    int operator() (const T* S, T* D, int w) const
    {
        if (!fast_mode)
            return 0;

        const T* nextS = (const T*)((const uchar*)S + step);
        int dx = vecOp(S, D, w);

        if (cn == 1)
            for( ; dx < w; ++dx )
            {
                int index = dx*2;
                D[dx] = (T)((S[index] + S[index+1] + nextS[index] + nextS[index+1] + 2) >> 2);
            }
        else if (cn == 3)
            for( ; dx < w; dx += 3 )
            {
                int index = dx*2;
                D[dx] = (T)((S[index] + S[index+3] + nextS[index] + nextS[index+3] + 2) >> 2);
                D[dx+1] = (T)((S[index+1] + S[index+4] + nextS[index+1] + nextS[index+4] + 2) >> 2);
                D[dx+2] = (T)((S[index+2] + S[index+5] + nextS[index+2] + nextS[index+5] + 2) >> 2);
            }
        else
            {
                CV_Assert(cn == 4);
                for( ; dx < w; dx += 4 )
                {
                    int index = dx*2;
                    D[dx] = (T)((S[index] + S[index+4] + nextS[index] + nextS[index+4] + 2) >> 2);
                    D[dx+1] = (T)((S[index+1] + S[index+5] + nextS[index+1] + nextS[index+5] + 2) >> 2);
                    D[dx+2] = (T)((S[index+2] + S[index+6] + nextS[index+2] + nextS[index+6] + 2) >> 2);
                    D[dx+3] = (T)((S[index+3] + S[index+7] + nextS[index+3] + nextS[index+7] + 2) >> 2);
                }
            }

        return dx;
    }

private:
    int scale_x, scale_y;
    int cn;
    bool fast_mode;
    int step;
    SIMDVecOp vecOp;
};

template <typename T, typename WT, typename VecOp>
class resizeAreaFast_Invoker :
    public ParallelLoopBody
{
public:
    resizeAreaFast_Invoker(const Mat &_src, Mat &_dst,
        int _scale_x, int _scale_y, const int* _ofs, const int* _xofs) :
        ParallelLoopBody(), src(_src), dst(_dst), scale_x(_scale_x),
        scale_y(_scale_y), ofs(_ofs), xofs(_xofs)
    {
    }

    virtual void operator() (const Range& range) const
    {
        Size ssize = src.size(), dsize = dst.size();
        int cn = src.channels();
        int area = scale_x*scale_y;
        float scale = 1.f/(area);
        int dwidth1 = (ssize.width/scale_x)*cn;
        dsize.width *= cn;
        ssize.width *= cn;
        int dy, dx, k = 0;

        VecOp vop(scale_x, scale_y, src.channels(), (int)src.step/*, area_ofs*/);

        for( dy = range.start; dy < range.end; dy++ )
        {
            T* D = (T*)(dst.data + dst.step*dy);
            int sy0 = dy*scale_y;
            int w = sy0 + scale_y <= ssize.height ? dwidth1 : 0;

            if( sy0 >= ssize.height )
            {
                for( dx = 0; dx < dsize.width; dx++ )
                    D[dx] = 0;
                continue;
            }

            dx = vop((const T*)(src.data + src.step * sy0), D, w);
            for( ; dx < w; dx++ )
            {
                const T* S = (const T*)(src.data + src.step * sy0) + xofs[dx];
                WT sum = 0;
                k = 0;
                #if CV_ENABLE_UNROLLED
                for( ; k <= area - 4; k += 4 )
                    sum += S[ofs[k]] + S[ofs[k+1]] + S[ofs[k+2]] + S[ofs[k+3]];
                #endif
                for( ; k < area; k++ )
                    sum += S[ofs[k]];

                D[dx] = saturate_cast<T>(sum * scale);
            }

            for( ; dx < dsize.width; dx++ )
            {
                WT sum = 0;
                int count = 0, sx0 = xofs[dx];
                if( sx0 >= ssize.width )
                    D[dx] = 0;

                for( int sy = 0; sy < scale_y; sy++ )
                {
                    if( sy0 + sy >= ssize.height )
                        break;
                    const T* S = (const T*)(src.data + src.step*(sy0 + sy)) + sx0;
                    for( int sx = 0; sx < scale_x*cn; sx += cn )
                    {
                        if( sx0 + sx >= ssize.width )
                            break;
                        sum += S[sx];
                        count++;
                    }
                }

                D[dx] = saturate_cast<T>((float)sum/count);
            }
        }
    }

private:
    Mat src;
    Mat dst;
    int scale_x, scale_y;
    const int *ofs, *xofs;
};

template<typename T, typename WT, typename VecOp>
static void resizeAreaFast_( const Mat& src, Mat& dst, const int* ofs, const int* xofs,
                             int scale_x, int scale_y )
{
    Range range(0, dst.rows);
    resizeAreaFast_Invoker<T, WT, VecOp> invoker(src, dst, scale_x,
        scale_y, ofs, xofs);
    parallel_for_(range, invoker, dst.total()/(double)(1<<16));
}

struct DecimateAlpha
{
    int si, di;
    float alpha;
};


template<typename T, typename WT> class ResizeArea_Invoker :
    public ParallelLoopBody
{
public:
    ResizeArea_Invoker( const Mat& _src, Mat& _dst,
                        const DecimateAlpha* _xtab, int _xtab_size,
                        const DecimateAlpha* _ytab, int _ytab_size,
                        const int* _tabofs )
    {
        src = &_src;
        dst = &_dst;
        xtab0 = _xtab;
        xtab_size0 = _xtab_size;
        ytab = _ytab;
        ytab_size = _ytab_size;
        tabofs = _tabofs;
    }

    virtual void operator() (const Range& range) const
    {
        Size dsize = dst->size();
        int cn = dst->channels();
        dsize.width *= cn;
        AutoBuffer<WT> _buffer(dsize.width*2);
        const DecimateAlpha* xtab = xtab0;
        int xtab_size = xtab_size0;
        WT *buf = _buffer, *sum = buf + dsize.width;
        int j_start = tabofs[range.start], j_end = tabofs[range.end], j, k, dx, prev_dy = ytab[j_start].di;

        for( dx = 0; dx < dsize.width; dx++ )
            sum[dx] = (WT)0;

        for( j = j_start; j < j_end; j++ )
        {
            WT beta = ytab[j].alpha;
            int dy = ytab[j].di;
            int sy = ytab[j].si;

            {
                const T* S = (const T*)(src->data + src->step*sy);
                for( dx = 0; dx < dsize.width; dx++ )
                    buf[dx] = (WT)0;

                if( cn == 1 )
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        buf[dxn] += S[xtab[k].si]*alpha;
                    }
                else if( cn == 2 )
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        WT t0 = buf[dxn] + S[sxn]*alpha;
                        WT t1 = buf[dxn+1] + S[sxn+1]*alpha;
                        buf[dxn] = t0; buf[dxn+1] = t1;
                    }
                else if( cn == 3 )
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        WT t0 = buf[dxn] + S[sxn]*alpha;
                        WT t1 = buf[dxn+1] + S[sxn+1]*alpha;
                        WT t2 = buf[dxn+2] + S[sxn+2]*alpha;
                        buf[dxn] = t0; buf[dxn+1] = t1; buf[dxn+2] = t2;
                    }
                else if( cn == 4 )
                {
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        WT t0 = buf[dxn] + S[sxn]*alpha;
                        WT t1 = buf[dxn+1] + S[sxn+1]*alpha;
                        buf[dxn] = t0; buf[dxn+1] = t1;
                        t0 = buf[dxn+2] + S[sxn+2]*alpha;
                        t1 = buf[dxn+3] + S[sxn+3]*alpha;
                        buf[dxn+2] = t0; buf[dxn+3] = t1;
                    }
                }
                else
                {
                    for( k = 0; k < xtab_size; k++ )
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        for( int c = 0; c < cn; c++ )
                            buf[dxn + c] += S[sxn + c]*alpha;
                    }
                }
            }

            if( dy != prev_dy )
            {
                T* D = (T*)(dst->data + dst->step*prev_dy);

                for( dx = 0; dx < dsize.width; dx++ )
                {
                    D[dx] = saturate_cast<T>(sum[dx]);
                    sum[dx] = beta*buf[dx];
                }
                prev_dy = dy;
            }
            else
            {
                for( dx = 0; dx < dsize.width; dx++ )
                    sum[dx] += beta*buf[dx];
            }
        }

        {
        T* D = (T*)(dst->data + dst->step*prev_dy);
        for( dx = 0; dx < dsize.width; dx++ )
            D[dx] = saturate_cast<T>(sum[dx]);
        }
    }

private:
    const Mat* src;
    Mat* dst;
    const DecimateAlpha* xtab0;
    const DecimateAlpha* ytab;
    int xtab_size0, ytab_size;
    const int* tabofs;
};


template <typename T, typename WT>
static void resizeArea_( const Mat& src, Mat& dst,
                         const DecimateAlpha* xtab, int xtab_size,
                         const DecimateAlpha* ytab, int ytab_size,
                         const int* tabofs )
{
    parallel_for_(Range(0, dst.rows),
                 ResizeArea_Invoker<T, WT>(src, dst, xtab, xtab_size, ytab, ytab_size, tabofs),
                 dst.total()/((double)(1 << 16)));
}


typedef void (*ResizeFunc)( const Mat& src, Mat& dst,
                            const int* xofs, const void* alpha,
                            const int* yofs, const void* beta,
                            int xmin, int xmax, int ksize );

typedef void (*ResizeAreaFastFunc)( const Mat& src, Mat& dst,
                                    const int* ofs, const int *xofs,
                                    int scale_x, int scale_y );

typedef void (*ResizeAreaFunc)( const Mat& src, Mat& dst,
                                const DecimateAlpha* xtab, int xtab_size,
                                const DecimateAlpha* ytab, int ytab_size,
                                const int* yofs);


static int computeResizeAreaTab( int ssize, int dsize, int cn, double scale, DecimateAlpha* tab )
{
    int k = 0;
    for(int dx = 0; dx < dsize; dx++ )
    {
        double fsx1 = dx * scale;
        double fsx2 = fsx1 + scale;
        double cellWidth = std::min(scale, ssize - fsx1);

        int sx1 = cvCeil(fsx1), sx2 = cvFloor(fsx2);

        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);

        if( sx1 - fsx1 > 1e-3 )
        {
            assert( k < ssize*2 );
            tab[k].di = dx * cn;
            tab[k].si = (sx1 - 1) * cn;
            tab[k++].alpha = (float)((sx1 - fsx1) / cellWidth);
        }

        for(int sx = sx1; sx < sx2; sx++ )
        {
            assert( k < ssize*2 );
            tab[k].di = dx * cn;
            tab[k].si = sx * cn;
            tab[k++].alpha = float(1.0 / cellWidth);
        }

        if( fsx2 - sx2 > 1e-3 )
        {
            assert( k < ssize*2 );
            tab[k].di = dx * cn;
            tab[k].si = sx2 * cn;
            tab[k++].alpha = (float)(std::min(std::min(fsx2 - sx2, 1.), cellWidth) / cellWidth);
        }
    }
    return k;
}

#define CHECK_IPP_STATUS(STATUS) if (STATUS < 0) { *ok = false; return; }

#define SET_IPP_RESIZE_LINEAR_FUNC_PTR(TYPE, CN) \
    func = (ippiResizeFunc)ippiResizeLinear_##TYPE##_##CN##R; \
    CHECK_IPP_STATUS(ippiResizeGetSize_##TYPE(srcSize, dstSize, (IppiInterpolationType)mode, 0, &specSize, &initSize));\
    specBuf.allocate(specSize);\
    pSpec = (uchar*)specBuf;\
    CHECK_IPP_STATUS(ippiResizeLinearInit_##TYPE(srcSize, dstSize, (IppiResizeSpec_32f*)pSpec));

#define SET_IPP_RESIZE_LINEAR_FUNC_64_PTR(TYPE, CN) \
    if (mode == (int)ippCubic) { *ok = false; return; } \
    func = (ippiResizeFunc)ippiResizeLinear_##TYPE##_##CN##R; \
    CHECK_IPP_STATUS(ippiResizeGetSize_##TYPE(srcSize, dstSize, (IppiInterpolationType)mode, 0, &specSize, &initSize));\
    specBuf.allocate(specSize);\
    pSpec = (uchar*)specBuf;\
    CHECK_IPP_STATUS(ippiResizeLinearInit_##TYPE(srcSize, dstSize, (IppiResizeSpec_64f*)pSpec));\
    getBufferSizeFunc = (ippiResizeGetBufferSize)ippiResizeGetBufferSize_##TYPE;\
    getSrcOffsetFunc =  (ippiResizeGetSrcOffset) ippiResizeGetSrcOffset_##TYPE;

#define SET_IPP_RESIZE_CUBIC_FUNC_PTR(TYPE, CN) \
    func = (ippiResizeFunc)ippiResizeCubic_##TYPE##_##CN##R; \
    CHECK_IPP_STATUS(ippiResizeGetSize_##TYPE(srcSize, dstSize, (IppiInterpolationType)mode, 0, &specSize, &initSize));\
    specBuf.allocate(specSize);\
    pSpec = (uchar*)specBuf;\
    AutoBuffer<uchar> buf(initSize);\
    uchar* pInit = (uchar*)buf;\
    CHECK_IPP_STATUS(ippiResizeCubicInit_##TYPE(srcSize, dstSize, 0.f, 0.75f, (IppiResizeSpec_32f*)pSpec, pInit));

#define SET_IPP_RESIZE_PTR(TYPE, CN) \
    if (mode == (int)ippLinear)     { SET_IPP_RESIZE_LINEAR_FUNC_PTR(TYPE, CN);} \
    else if (mode == (int)ippCubic) { SET_IPP_RESIZE_CUBIC_FUNC_PTR(TYPE, CN);} \
    else { *ok = false; return; } \
    getBufferSizeFunc = (ippiResizeGetBufferSize)ippiResizeGetBufferSize_##TYPE; \
    getSrcOffsetFunc =  (ippiResizeGetSrcOffset)ippiResizeGetSrcOffset_##TYPE;

#if IPP_VERSION_X100 >= 701
class IPPresizeInvoker :
    public ParallelLoopBody
{
public:
    IPPresizeInvoker(const Mat & _src, Mat & _dst, double _inv_scale_x, double _inv_scale_y, int _mode, bool *_ok) :
        ParallelLoopBody(), src(_src), dst(_dst), inv_scale_x(_inv_scale_x), inv_scale_y(_inv_scale_y), mode(_mode), ok(_ok)
    {
        *ok = true;
        IppiSize srcSize, dstSize;
        int type = src.type();
        int specSize = 0, initSize = 0;
        srcSize.width  = src.cols;
        srcSize.height = src.rows;
        dstSize.width  = dst.cols;
        dstSize.height = dst.rows;

        switch (type)
        {
#if 0 // disabled since it breaks tests for CascadeClassifier
            case CV_8UC1:  SET_IPP_RESIZE_PTR(8u,C1);  break;
            case CV_8UC3:  SET_IPP_RESIZE_PTR(8u,C3);  break;
            case CV_8UC4:  SET_IPP_RESIZE_PTR(8u,C4);  break;
#endif
            case CV_16UC1: SET_IPP_RESIZE_PTR(16u,C1); break;
            case CV_16UC3: SET_IPP_RESIZE_PTR(16u,C3); break;
            case CV_16UC4: SET_IPP_RESIZE_PTR(16u,C4); break;
            case CV_16SC1: SET_IPP_RESIZE_PTR(16s,C1); break;
            case CV_16SC3: SET_IPP_RESIZE_PTR(16s,C3); break;
            case CV_16SC4: SET_IPP_RESIZE_PTR(16s,C4); break;
            case CV_32FC1: SET_IPP_RESIZE_PTR(32f,C1); break;
            case CV_32FC3: SET_IPP_RESIZE_PTR(32f,C3); break;
            case CV_32FC4: SET_IPP_RESIZE_PTR(32f,C4); break;
            case CV_64FC1: SET_IPP_RESIZE_LINEAR_FUNC_64_PTR(64f,C1); break;
            case CV_64FC3: SET_IPP_RESIZE_LINEAR_FUNC_64_PTR(64f,C3); break;
            case CV_64FC4: SET_IPP_RESIZE_LINEAR_FUNC_64_PTR(64f,C4); break;
            default: { *ok = false; return; } break;
        }
    }

    ~IPPresizeInvoker()
    {
    }

    virtual void operator() (const Range& range) const
    {
        if (*ok == false)
          return;

        int cn = src.channels();
        int dsty = min(cvRound(range.start * inv_scale_y), dst.rows);
        int dstwidth  = min(cvRound(src.cols * inv_scale_x), dst.cols);
        int dstheight = min(cvRound(range.end * inv_scale_y), dst.rows);

        IppiPoint dstOffset = { 0, dsty }, srcOffset = {0, 0};
        IppiSize  dstSize   = { dstwidth, dstheight - dsty };
        int bufsize = 0, itemSize = (int)src.elemSize1();

        CHECK_IPP_STATUS(getBufferSizeFunc(pSpec, dstSize, cn, &bufsize));
        CHECK_IPP_STATUS(getSrcOffsetFunc(pSpec, dstOffset, &srcOffset));

        const Ipp8u* pSrc = (const Ipp8u*)src.data + (int)src.step[0] * srcOffset.y + srcOffset.x * cn * itemSize;
        Ipp8u* pDst = (Ipp8u*)dst.data + (int)dst.step[0] * dstOffset.y + dstOffset.x * cn * itemSize;

        AutoBuffer<uchar> buf(bufsize + 64);
        uchar* bufptr = alignPtr((uchar*)buf, 32);

        if( func( pSrc, (int)src.step[0], pDst, (int)dst.step[0], dstOffset, dstSize, ippBorderRepl, 0, pSpec, bufptr ) < 0 )
            *ok = false;
    }
private:
    const Mat & src;
    Mat & dst;
    double inv_scale_x;
    double inv_scale_y;
    void *pSpec;
    AutoBuffer<uchar>   specBuf;
    int mode;
    ippiResizeFunc func;
    ippiResizeGetBufferSize getBufferSizeFunc;
    ippiResizeGetSrcOffset getSrcOffsetFunc;
    bool *ok;
    const IPPresizeInvoker& operator= (const IPPresizeInvoker&);
};

#endif

#ifdef HAVE_OPENCL

static void ocl_computeResizeAreaTabs(int ssize, int dsize, double scale, int * const map_tab,
                                      float * const alpha_tab, int * const ofs_tab)
{
    int k = 0, dx = 0;
    for ( ; dx < dsize; dx++)
    {
        ofs_tab[dx] = k;

        double fsx1 = dx * scale;
        double fsx2 = fsx1 + scale;
        double cellWidth = std::min(scale, ssize - fsx1);

        int sx1 = cvCeil(fsx1), sx2 = cvFloor(fsx2);

        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);

        if (sx1 - fsx1 > 1e-3)
        {
            map_tab[k] = sx1 - 1;
            alpha_tab[k++] = (float)((sx1 - fsx1) / cellWidth);
        }

        for (int sx = sx1; sx < sx2; sx++)
        {
            map_tab[k] = sx;
            alpha_tab[k++] = float(1.0 / cellWidth);
        }

        if (fsx2 - sx2 > 1e-3)
        {
            map_tab[k] = sx2;
            alpha_tab[k++] = (float)(std::min(std::min(fsx2 - sx2, 1.), cellWidth) / cellWidth);
        }
    }
    ofs_tab[dx] = k;
}

static bool ocl_resize( InputArray _src, OutputArray _dst, Size dsize,
                        double fx, double fy, int interpolation)
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    double inv_fx = 1.0 / fx, inv_fy = 1.0 / fy;
    float inv_fxf = (float)inv_fx, inv_fyf = (float)inv_fy;
    int iscale_x = saturate_cast<int>(inv_fx), iscale_y = saturate_cast<int>(inv_fx);
    bool is_area_fast = std::abs(inv_fx - iscale_x) < DBL_EPSILON &&
        std::abs(inv_fy - iscale_y) < DBL_EPSILON;

    // in case of scale_x && scale_y is equal to 2
    // INTER_AREA (fast) also is equal to INTER_LINEAR
    if( interpolation == INTER_LINEAR && is_area_fast && iscale_x == 2 && iscale_y == 2 )
        /*interpolation = INTER_AREA*/(void)0; // INTER_AREA is slower

    if( !(cn <= 4 &&
           (interpolation == INTER_NEAREST || interpolation == INTER_LINEAR ||
            (interpolation == INTER_AREA && inv_fx >= 1 && inv_fy >= 1) )) )
        return false;

    UMat src = _src.getUMat();
    _dst.create(dsize, type);
    UMat dst = _dst.getUMat();

    Size ssize = src.size();
    ocl::Kernel k;
    size_t globalsize[] = { dst.cols, dst.rows };

    ocl::Image2D srcImage;

    // See if this could be done with a sampler.  We stick with integer
    // datatypes because the observed error is low.
    bool useSampler = (interpolation == INTER_LINEAR && ocl::Device::getDefault().imageSupport() &&
                       ocl::Image2D::canCreateAlias(src) && depth <= 4 &&
                       ocl::Image2D::isFormatSupported(depth, cn, true));
    if (useSampler)
    {
        int wdepth = std::max(depth, CV_32S);
        char buf[2][32];
        cv::String compileOpts = format("-D USE_SAMPLER -D depth=%d -D T=%s -D T1=%s "
                        "-D convertToDT=%s -D cn=%d",
                        depth, ocl::typeToStr(type), ocl::typeToStr(depth),
                        ocl::convertTypeStr(wdepth, depth, cn, buf[1]),
                        cn);
        k.create("resizeSampler", ocl::imgproc::resize_oclsrc, compileOpts);

        if (k.empty())
            useSampler = false;
        else
        {
            // Convert the input into an OpenCL image type, using normalized channel data types
            // and aliasing the UMat.
            srcImage = ocl::Image2D(src, true, true);
            k.args(srcImage, ocl::KernelArg::WriteOnly(dst),
                   (float)inv_fx, (float)inv_fy);
        }
    }

    if (interpolation == INTER_LINEAR && !useSampler)
    {
        char buf[2][32];

        // integer path is slower because of CPU part, so it's disabled
        if (depth == CV_8U && ((void)0, 0))
        {
            AutoBuffer<uchar> _buffer((dsize.width + dsize.height)*(sizeof(int) + sizeof(short)*2));
            int* xofs = (int*)(uchar*)_buffer, * yofs = xofs + dsize.width;
            short* ialpha = (short*)(yofs + dsize.height), * ibeta = ialpha + dsize.width*2;
            float fxx, fyy;
            int sx, sy;

            for (int dx = 0; dx < dsize.width; dx++)
            {
                fxx = (float)((dx+0.5)*inv_fx - 0.5);
                sx = cvFloor(fxx);
                fxx -= sx;

                if (sx < 0)
                    fxx = 0, sx = 0;

                if (sx >= ssize.width-1)
                    fxx = 0, sx = ssize.width-1;

                xofs[dx] = sx;
                ialpha[dx*2 + 0] = saturate_cast<short>((1.f - fxx) * INTER_RESIZE_COEF_SCALE);
                ialpha[dx*2 + 1] = saturate_cast<short>(fxx         * INTER_RESIZE_COEF_SCALE);
            }

            for (int dy = 0; dy < dsize.height; dy++)
            {
                fyy = (float)((dy+0.5)*inv_fy - 0.5);
                sy = cvFloor(fyy);
                fyy -= sy;

                yofs[dy] = sy;
                ibeta[dy*2 + 0] = saturate_cast<short>((1.f - fyy) * INTER_RESIZE_COEF_SCALE);
                ibeta[dy*2 + 1] = saturate_cast<short>(fyy         * INTER_RESIZE_COEF_SCALE);
            }

            int wdepth = std::max(depth, CV_32S), wtype = CV_MAKETYPE(wdepth, cn);
            UMat coeffs;
            Mat(1, static_cast<int>(_buffer.size()), CV_8UC1, (uchar *)_buffer).copyTo(coeffs);

            k.create("resizeLN", ocl::imgproc::resize_oclsrc,
                     format("-D INTER_LINEAR_INTEGER -D depth=%d -D T=%s -D T1=%s "
                            "-D WT=%s -D convertToWT=%s -D convertToDT=%s -D cn=%d "
                            "-D INTER_RESIZE_COEF_BITS=%d",
                            depth, ocl::typeToStr(type), ocl::typeToStr(depth), ocl::typeToStr(wtype),
                            ocl::convertTypeStr(depth, wdepth, cn, buf[0]),
                            ocl::convertTypeStr(wdepth, depth, cn, buf[1]),
                            cn, INTER_RESIZE_COEF_BITS));
            if (k.empty())
                return false;

            k.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnly(dst),
                   ocl::KernelArg::PtrReadOnly(coeffs));
        }
        else
        {
            int wdepth = std::max(depth, CV_32S), wtype = CV_MAKETYPE(wdepth, cn);
            k.create("resizeLN", ocl::imgproc::resize_oclsrc,
                     format("-D INTER_LINEAR -D depth=%d -D T=%s -D T1=%s "
                            "-D WT=%s -D convertToWT=%s -D convertToDT=%s -D cn=%d "
                            "-D INTER_RESIZE_COEF_BITS=%d",
                            depth, ocl::typeToStr(type), ocl::typeToStr(depth), ocl::typeToStr(wtype),
                            ocl::convertTypeStr(depth, wdepth, cn, buf[0]),
                            ocl::convertTypeStr(wdepth, depth, cn, buf[1]),
                            cn, INTER_RESIZE_COEF_BITS));
            if (k.empty())
                return false;

            k.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnly(dst),
                   (float)inv_fx, (float)inv_fy);
        }
    }
    else if (interpolation == INTER_NEAREST)
    {
        k.create("resizeNN", ocl::imgproc::resize_oclsrc,
                 format("-D INTER_NEAREST -D T=%s -D T1=%s -D cn=%d",
                        ocl::memopTypeToStr(type), ocl::memopTypeToStr(depth), cn));
        if (k.empty())
            return false;

        k.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnly(dst),
               (float)inv_fx, (float)inv_fy);
    }
    else if (interpolation == INTER_AREA)
    {
        int wdepth = std::max(depth, is_area_fast ? CV_32S : CV_32F);
        int wtype = CV_MAKE_TYPE(wdepth, cn);

        char cvt[2][40];
        String buildOption = format("-D INTER_AREA -D T=%s -D T1=%s -D WTV=%s -D convertToWTV=%s -D cn=%d",
                                    ocl::typeToStr(type), ocl::typeToStr(depth), ocl::typeToStr(wtype),
                                    ocl::convertTypeStr(depth, wdepth, cn, cvt[0]), cn);

        UMat alphaOcl, tabofsOcl, mapOcl;
        UMat dmap, smap;

        if (is_area_fast)
        {
            int wdepth2 = std::max(CV_32F, depth), wtype2 = CV_MAKE_TYPE(wdepth2, cn);
            buildOption = buildOption + format(" -D convertToT=%s -D WT2V=%s -D convertToWT2V=%s -D INTER_AREA_FAST"
                                                " -D XSCALE=%d -D YSCALE=%d -D SCALE=%ff",
                                                ocl::convertTypeStr(wdepth2, depth, cn, cvt[0]),
                                                ocl::typeToStr(wtype2), ocl::convertTypeStr(wdepth, wdepth2, cn, cvt[1]),
                                    iscale_x, iscale_y, 1.0f / (iscale_x * iscale_y));

            k.create("resizeAREA_FAST", ocl::imgproc::resize_oclsrc, buildOption);
            if (k.empty())
                return false;
        }
        else
        {
            buildOption = buildOption + format(" -D convertToT=%s", ocl::convertTypeStr(wdepth, depth, cn, cvt[0]));
            k.create("resizeAREA", ocl::imgproc::resize_oclsrc, buildOption);
            if (k.empty())
                return false;

            int xytab_size = (ssize.width + ssize.height) << 1;
            int tabofs_size = dsize.height + dsize.width + 2;

            AutoBuffer<int> _xymap_tab(xytab_size), _xyofs_tab(tabofs_size);
            AutoBuffer<float> _xyalpha_tab(xytab_size);
            int * xmap_tab = _xymap_tab, * ymap_tab = _xymap_tab + (ssize.width << 1);
            float * xalpha_tab = _xyalpha_tab, * yalpha_tab = _xyalpha_tab + (ssize.width << 1);
            int * xofs_tab = _xyofs_tab, * yofs_tab = _xyofs_tab + dsize.width + 1;

            ocl_computeResizeAreaTabs(ssize.width, dsize.width, inv_fx, xmap_tab, xalpha_tab, xofs_tab);
            ocl_computeResizeAreaTabs(ssize.height, dsize.height, inv_fy, ymap_tab, yalpha_tab, yofs_tab);

            // loading precomputed arrays to GPU
            Mat(1, xytab_size, CV_32FC1, (void *)_xyalpha_tab).copyTo(alphaOcl);
            Mat(1, xytab_size, CV_32SC1, (void *)_xymap_tab).copyTo(mapOcl);
            Mat(1, tabofs_size, CV_32SC1, (void *)_xyofs_tab).copyTo(tabofsOcl);
        }

        ocl::KernelArg srcarg = ocl::KernelArg::ReadOnly(src), dstarg = ocl::KernelArg::WriteOnly(dst);

        if (is_area_fast)
            k.args(srcarg, dstarg);
        else
            k.args(srcarg, dstarg, inv_fxf, inv_fyf, ocl::KernelArg::PtrReadOnly(tabofsOcl),
                   ocl::KernelArg::PtrReadOnly(mapOcl), ocl::KernelArg::PtrReadOnly(alphaOcl));

        return k.run(2, globalsize, NULL, false);
    }

    return k.run(2, globalsize, 0, false);
}

#endif

}

//////////////////////////////////////////////////////////////////////////////////////////

void cv::resize( InputArray _src, OutputArray _dst, Size dsize,
                 double inv_scale_x, double inv_scale_y, int interpolation )
{
    static ResizeFunc linear_tab[] =
    {
        resizeGeneric_<
            HResizeLinear<uchar, int, short,
                INTER_RESIZE_COEF_SCALE,
                HResizeLinearVec_8u32s>,
            VResizeLinear<uchar, int, short,
                FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS*2>,
                VResizeLinearVec_32s8u> >,
        0,
        resizeGeneric_<
            HResizeLinear<ushort, float, float, 1,
                HResizeLinearVec_16u32f>,
            VResizeLinear<ushort, float, float, Cast<float, ushort>,
                VResizeLinearVec_32f16u> >,
        resizeGeneric_<
            HResizeLinear<short, float, float, 1,
                HResizeLinearVec_16s32f>,
            VResizeLinear<short, float, float, Cast<float, short>,
                VResizeLinearVec_32f16s> >,
        0,
        resizeGeneric_<
            HResizeLinear<float, float, float, 1,
                HResizeLinearVec_32f>,
            VResizeLinear<float, float, float, Cast<float, float>,
                VResizeLinearVec_32f> >,
        resizeGeneric_<
            HResizeLinear<double, double, float, 1,
                HResizeNoVec>,
            VResizeLinear<double, double, float, Cast<double, double>,
                VResizeNoVec> >,
        0
    };

    static ResizeFunc cubic_tab[] =
    {
        resizeGeneric_<
            HResizeCubic<uchar, int, short>,
            VResizeCubic<uchar, int, short,
                FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS*2>,
                VResizeCubicVec_32s8u> >,
        0,
        resizeGeneric_<
            HResizeCubic<ushort, float, float>,
            VResizeCubic<ushort, float, float, Cast<float, ushort>,
            VResizeCubicVec_32f16u> >,
        resizeGeneric_<
            HResizeCubic<short, float, float>,
            VResizeCubic<short, float, float, Cast<float, short>,
            VResizeCubicVec_32f16s> >,
        0,
        resizeGeneric_<
            HResizeCubic<float, float, float>,
            VResizeCubic<float, float, float, Cast<float, float>,
            VResizeCubicVec_32f> >,
        resizeGeneric_<
            HResizeCubic<double, double, float>,
            VResizeCubic<double, double, float, Cast<double, double>,
            VResizeNoVec> >,
        0
    };

    static ResizeFunc lanczos4_tab[] =
    {
        resizeGeneric_<HResizeLanczos4<uchar, int, short>,
            VResizeLanczos4<uchar, int, short,
            FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS*2>,
            VResizeNoVec> >,
        0,
        resizeGeneric_<HResizeLanczos4<ushort, float, float>,
            VResizeLanczos4<ushort, float, float, Cast<float, ushort>,
            VResizeNoVec> >,
        resizeGeneric_<HResizeLanczos4<short, float, float>,
            VResizeLanczos4<short, float, float, Cast<float, short>,
            VResizeNoVec> >,
        0,
        resizeGeneric_<HResizeLanczos4<float, float, float>,
            VResizeLanczos4<float, float, float, Cast<float, float>,
            VResizeNoVec> >,
        resizeGeneric_<HResizeLanczos4<double, double, float>,
            VResizeLanczos4<double, double, float, Cast<double, double>,
            VResizeNoVec> >,
        0
    };

    static ResizeAreaFastFunc areafast_tab[] =
    {
        resizeAreaFast_<uchar, int, ResizeAreaFastVec<uchar, ResizeAreaFastVec_SIMD_8u> >,
        0,
        resizeAreaFast_<ushort, float, ResizeAreaFastVec<ushort, ResizeAreaFastVec_SIMD_16u> >,
        resizeAreaFast_<short, float, ResizeAreaFastVec<short, ResizeAreaFastNoVec<short, float> > >,
        0,
        resizeAreaFast_<float, float, ResizeAreaFastNoVec<float, float> >,
        resizeAreaFast_<double, double, ResizeAreaFastNoVec<double, double> >,
        0
    };

    static ResizeAreaFunc area_tab[] =
    {
        resizeArea_<uchar, float>, 0, resizeArea_<ushort, float>,
        resizeArea_<short, float>, 0, resizeArea_<float, float>,
        resizeArea_<double, double>, 0
    };

    Size ssize = _src.size();

    CV_Assert( ssize.area() > 0 );
    CV_Assert( dsize.area() > 0 || (inv_scale_x > 0 && inv_scale_y > 0) );
    if( dsize.area() == 0 )
    {
        dsize = Size(saturate_cast<int>(ssize.width*inv_scale_x),
                     saturate_cast<int>(ssize.height*inv_scale_y));
        CV_Assert( dsize.area() > 0 );
    }
    else
    {
        inv_scale_x = (double)dsize.width/ssize.width;
        inv_scale_y = (double)dsize.height/ssize.height;
    }

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_resize(_src, _dst, dsize, inv_scale_x, inv_scale_y, interpolation))

    Mat src = _src.getMat();
    _dst.create(dsize, src.type());
    Mat dst = _dst.getMat();

#ifdef HAVE_TEGRA_OPTIMIZATION
    if (tegra::resize(src, dst, (float)inv_scale_x, (float)inv_scale_y, interpolation))
        return;
#endif

    int type = src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    double scale_x = 1./inv_scale_x, scale_y = 1./inv_scale_y;
    int k, sx, sy, dx, dy;

    int iscale_x = saturate_cast<int>(scale_x);
    int iscale_y = saturate_cast<int>(scale_y);

    bool is_area_fast = std::abs(scale_x - iscale_x) < DBL_EPSILON &&
            std::abs(scale_y - iscale_y) < DBL_EPSILON;

#if IPP_VERSION_X100 >= 701
#define IPP_RESIZE_EPS 1e-10

    double ex = fabs((double)dsize.width / src.cols  - inv_scale_x) / inv_scale_x;
    double ey = fabs((double)dsize.height / src.rows - inv_scale_y) / inv_scale_y;

    if ( ((ex < IPP_RESIZE_EPS && ey < IPP_RESIZE_EPS && depth != CV_64F) || (ex == 0 && ey == 0 && depth == CV_64F)) &&
         (interpolation == INTER_LINEAR || interpolation == INTER_CUBIC) &&
         !(interpolation == INTER_LINEAR && is_area_fast && iscale_x == 2 && iscale_y == 2 && depth == CV_8U))
    {
        int mode = -1;
        if (interpolation == INTER_LINEAR && src.rows >= 2 && src.cols >= 2)
            mode = ippLinear;
        else if (interpolation == INTER_CUBIC && src.rows >= 4 && src.cols >= 4)
            mode = ippCubic;

        if( mode >= 0 && (cn == 1 || cn == 3 || cn == 4) &&
            (depth == CV_16U || depth == CV_16S || depth == CV_32F ||
            (depth == CV_64F && mode == ippLinear)))
        {
            bool ok = true;
            Range range(0, src.rows);
            IPPresizeInvoker invoker(src, dst, inv_scale_x, inv_scale_y, mode, &ok);
            parallel_for_(range, invoker, dst.total()/(double)(1<<16));
            if( ok )
                return;
            setIppErrorStatus();
        }
    }
#undef IPP_RESIZE_EPS
#endif

    if( interpolation == INTER_NEAREST )
    {
        resizeNN( src, dst, inv_scale_x, inv_scale_y );
        return;
    }

    {
        // in case of scale_x && scale_y is equal to 2
        // INTER_AREA (fast) also is equal to INTER_LINEAR
        if( interpolation == INTER_LINEAR && is_area_fast && iscale_x == 2 && iscale_y == 2 )
            interpolation = INTER_AREA;

        // true "area" interpolation is only implemented for the case (scale_x <= 1 && scale_y <= 1).
        // In other cases it is emulated using some variant of bilinear interpolation
        if( interpolation == INTER_AREA && scale_x >= 1 && scale_y >= 1 )
        {
            if( is_area_fast )
            {
                int area = iscale_x*iscale_y;
                size_t srcstep = src.step / src.elemSize1();
                AutoBuffer<int> _ofs(area + dsize.width*cn);
                int* ofs = _ofs;
                int* xofs = ofs + area;
                ResizeAreaFastFunc func = areafast_tab[depth];
                CV_Assert( func != 0 );

                for( sy = 0, k = 0; sy < iscale_y; sy++ )
                    for( sx = 0; sx < iscale_x; sx++ )
                        ofs[k++] = (int)(sy*srcstep + sx*cn);

                for( dx = 0; dx < dsize.width; dx++ )
                {
                    int j = dx * cn;
                    sx = iscale_x * j;
                    for( k = 0; k < cn; k++ )
                        xofs[j + k] = sx + k;
                }

                func( src, dst, ofs, xofs, iscale_x, iscale_y );
                return;
            }

            ResizeAreaFunc func = area_tab[depth];
            CV_Assert( func != 0 && cn <= 4 );

            AutoBuffer<DecimateAlpha> _xytab((ssize.width + ssize.height)*2);
            DecimateAlpha* xtab = _xytab, *ytab = xtab + ssize.width*2;

            int xtab_size = computeResizeAreaTab(ssize.width, dsize.width, cn, scale_x, xtab);
            int ytab_size = computeResizeAreaTab(ssize.height, dsize.height, 1, scale_y, ytab);

            AutoBuffer<int> _tabofs(dsize.height + 1);
            int* tabofs = _tabofs;
            for( k = 0, dy = 0; k < ytab_size; k++ )
            {
                if( k == 0 || ytab[k].di != ytab[k-1].di )
                {
                    assert( ytab[k].di == dy );
                    tabofs[dy++] = k;
                }
            }
            tabofs[dy] = ytab_size;

            func( src, dst, xtab, xtab_size, ytab, ytab_size, tabofs );
            return;
        }
    }

    int xmin = 0, xmax = dsize.width, width = dsize.width*cn;
    bool area_mode = interpolation == INTER_AREA;
    bool fixpt = depth == CV_8U;
    float fx, fy;
    ResizeFunc func=0;
    int ksize=0, ksize2;
    if( interpolation == INTER_CUBIC )
        ksize = 4, func = cubic_tab[depth];
    else if( interpolation == INTER_LANCZOS4 )
        ksize = 8, func = lanczos4_tab[depth];
    else if( interpolation == INTER_LINEAR || interpolation == INTER_AREA )
        ksize = 2, func = linear_tab[depth];
    else
        CV_Error( CV_StsBadArg, "Unknown interpolation method" );
    ksize2 = ksize/2;

    CV_Assert( func != 0 );

    AutoBuffer<uchar> _buffer((width + dsize.height)*(sizeof(int) + sizeof(float)*ksize));
    int* xofs = (int*)(uchar*)_buffer;
    int* yofs = xofs + width;
    float* alpha = (float*)(yofs + dsize.height);
    short* ialpha = (short*)alpha;
    float* beta = alpha + width*ksize;
    short* ibeta = ialpha + width*ksize;
    float cbuf[MAX_ESIZE];

    for( dx = 0; dx < dsize.width; dx++ )
    {
        if( !area_mode )
        {
            fx = (float)((dx+0.5)*scale_x - 0.5);
            sx = cvFloor(fx);
            fx -= sx;
        }
        else
        {
            sx = cvFloor(dx*scale_x);
            fx = (float)((dx+1) - (sx+1)*inv_scale_x);
            fx = fx <= 0 ? 0.f : fx - cvFloor(fx);
        }

        if( sx < ksize2-1 )
        {
            xmin = dx+1;
            if( sx < 0 && (interpolation != INTER_CUBIC && interpolation != INTER_LANCZOS4))
                fx = 0, sx = 0;
        }

        if( sx + ksize2 >= ssize.width )
        {
            xmax = std::min( xmax, dx );
            if( sx >= ssize.width-1 && (interpolation != INTER_CUBIC && interpolation != INTER_LANCZOS4))
                fx = 0, sx = ssize.width-1;
        }

        for( k = 0, sx *= cn; k < cn; k++ )
            xofs[dx*cn + k] = sx + k;

        if( interpolation == INTER_CUBIC )
            interpolateCubic( fx, cbuf );
        else if( interpolation == INTER_LANCZOS4 )
            interpolateLanczos4( fx, cbuf );
        else
        {
            cbuf[0] = 1.f - fx;
            cbuf[1] = fx;
        }
        if( fixpt )
        {
            for( k = 0; k < ksize; k++ )
                ialpha[dx*cn*ksize + k] = saturate_cast<short>(cbuf[k]*INTER_RESIZE_COEF_SCALE);
            for( ; k < cn*ksize; k++ )
                ialpha[dx*cn*ksize + k] = ialpha[dx*cn*ksize + k - ksize];
        }
        else
        {
            for( k = 0; k < ksize; k++ )
                alpha[dx*cn*ksize + k] = cbuf[k];
            for( ; k < cn*ksize; k++ )
                alpha[dx*cn*ksize + k] = alpha[dx*cn*ksize + k - ksize];
        }
    }

    for( dy = 0; dy < dsize.height; dy++ )
    {
        if( !area_mode )
        {
            fy = (float)((dy+0.5)*scale_y - 0.5);
            sy = cvFloor(fy);
            fy -= sy;
        }
        else
        {
            sy = cvFloor(dy*scale_y);
            fy = (float)((dy+1) - (sy+1)*inv_scale_y);
            fy = fy <= 0 ? 0.f : fy - cvFloor(fy);
        }

        yofs[dy] = sy;
        if( interpolation == INTER_CUBIC )
            interpolateCubic( fy, cbuf );
        else if( interpolation == INTER_LANCZOS4 )
            interpolateLanczos4( fy, cbuf );
        else
        {
            cbuf[0] = 1.f - fy;
            cbuf[1] = fy;
        }

        if( fixpt )
        {
            for( k = 0; k < ksize; k++ )
                ibeta[dy*ksize + k] = saturate_cast<short>(cbuf[k]*INTER_RESIZE_COEF_SCALE);
        }
        else
        {
            for( k = 0; k < ksize; k++ )
                beta[dy*ksize + k] = cbuf[k];
        }
    }

    func( src, dst, xofs, fixpt ? (void*)ialpha : (void*)alpha, yofs,
          fixpt ? (void*)ibeta : (void*)beta, xmin, xmax, ksize );
}


/****************************************************************************************\
*                       General warping (affine, perspective, remap)                     *
\****************************************************************************************/

namespace cv
{

template<typename T>
static void remapNearest( const Mat& _src, Mat& _dst, const Mat& _xy,
                          int borderType, const Scalar& _borderValue )
{
    Size ssize = _src.size(), dsize = _dst.size();
    int cn = _src.channels();
    const T* S0 = (const T*)_src.data;
    size_t sstep = _src.step/sizeof(S0[0]);
    Scalar_<T> cval(saturate_cast<T>(_borderValue[0]),
        saturate_cast<T>(_borderValue[1]),
        saturate_cast<T>(_borderValue[2]),
        saturate_cast<T>(_borderValue[3]));
    int dx, dy;

    unsigned width1 = ssize.width, height1 = ssize.height;

    if( _dst.isContinuous() && _xy.isContinuous() )
    {
        dsize.width *= dsize.height;
        dsize.height = 1;
    }

    for( dy = 0; dy < dsize.height; dy++ )
    {
        T* D = (T*)(_dst.data + _dst.step*dy);
        const short* XY = (const short*)(_xy.data + _xy.step*dy);

        if( cn == 1 )
        {
            for( dx = 0; dx < dsize.width; dx++ )
            {
                int sx = XY[dx*2], sy = XY[dx*2+1];
                if( (unsigned)sx < width1 && (unsigned)sy < height1 )
                    D[dx] = S0[sy*sstep + sx];
                else
                {
                    if( borderType == BORDER_REPLICATE )
                    {
                        sx = clip(sx, 0, ssize.width);
                        sy = clip(sy, 0, ssize.height);
                        D[dx] = S0[sy*sstep + sx];
                    }
                    else if( borderType == BORDER_CONSTANT )
                        D[dx] = cval[0];
                    else if( borderType != BORDER_TRANSPARENT )
                    {
                        sx = borderInterpolate(sx, ssize.width, borderType);
                        sy = borderInterpolate(sy, ssize.height, borderType);
                        D[dx] = S0[sy*sstep + sx];
                    }
                }
            }
        }
        else
        {
            for( dx = 0; dx < dsize.width; dx++, D += cn )
            {
                int sx = XY[dx*2], sy = XY[dx*2+1], k;
                const T *S;
                if( (unsigned)sx < width1 && (unsigned)sy < height1 )
                {
                    if( cn == 3 )
                    {
                        S = S0 + sy*sstep + sx*3;
                        D[0] = S[0], D[1] = S[1], D[2] = S[2];
                    }
                    else if( cn == 4 )
                    {
                        S = S0 + sy*sstep + sx*4;
                        D[0] = S[0], D[1] = S[1], D[2] = S[2], D[3] = S[3];
                    }
                    else
                    {
                        S = S0 + sy*sstep + sx*cn;
                        for( k = 0; k < cn; k++ )
                            D[k] = S[k];
                    }
                }
                else if( borderType != BORDER_TRANSPARENT )
                {
                    if( borderType == BORDER_REPLICATE )
                    {
                        sx = clip(sx, 0, ssize.width);
                        sy = clip(sy, 0, ssize.height);
                        S = S0 + sy*sstep + sx*cn;
                    }
                    else if( borderType == BORDER_CONSTANT )
                        S = &cval[0];
                    else
                    {
                        sx = borderInterpolate(sx, ssize.width, borderType);
                        sy = borderInterpolate(sy, ssize.height, borderType);
                        S = S0 + sy*sstep + sx*cn;
                    }
                    for( k = 0; k < cn; k++ )
                        D[k] = S[k];
                }
            }
        }
    }
}


struct RemapNoVec
{
    int operator()( const Mat&, void*, const short*, const ushort*,
                    const void*, int ) const { return 0; }
};

#if CV_SSE2

struct RemapVec_8u
{
    int operator()( const Mat& _src, void* _dst, const short* XY,
                    const ushort* FXY, const void* _wtab, int width ) const
    {
        int cn = _src.channels(), x = 0, sstep = (int)_src.step;

        if( (cn != 1 && cn != 3 && cn != 4) || !checkHardwareSupport(CV_CPU_SSE2) ||
            sstep > 0x8000 )
            return 0;

        const uchar *S0 = _src.data, *S1 = _src.data + _src.step;
        const short* wtab = cn == 1 ? (const short*)_wtab : &BilinearTab_iC4[0][0][0];
        uchar* D = (uchar*)_dst;
        __m128i delta = _mm_set1_epi32(INTER_REMAP_COEF_SCALE/2);
        __m128i xy2ofs = _mm_set1_epi32(cn + (sstep << 16));
        __m128i z = _mm_setzero_si128();
        int CV_DECL_ALIGNED(16) iofs0[4], iofs1[4];

        if( cn == 1 )
        {
            for( ; x <= width - 8; x += 8 )
            {
                __m128i xy0 = _mm_loadu_si128( (const __m128i*)(XY + x*2));
                __m128i xy1 = _mm_loadu_si128( (const __m128i*)(XY + x*2 + 8));
                __m128i v0, v1, v2, v3, a0, a1, b0, b1;
                unsigned i0, i1;

                xy0 = _mm_madd_epi16( xy0, xy2ofs );
                xy1 = _mm_madd_epi16( xy1, xy2ofs );
                _mm_store_si128( (__m128i*)iofs0, xy0 );
                _mm_store_si128( (__m128i*)iofs1, xy1 );

                i0 = *(ushort*)(S0 + iofs0[0]) + (*(ushort*)(S0 + iofs0[1]) << 16);
                i1 = *(ushort*)(S0 + iofs0[2]) + (*(ushort*)(S0 + iofs0[3]) << 16);
                v0 = _mm_unpacklo_epi32(_mm_cvtsi32_si128(i0), _mm_cvtsi32_si128(i1));
                i0 = *(ushort*)(S1 + iofs0[0]) + (*(ushort*)(S1 + iofs0[1]) << 16);
                i1 = *(ushort*)(S1 + iofs0[2]) + (*(ushort*)(S1 + iofs0[3]) << 16);
                v1 = _mm_unpacklo_epi32(_mm_cvtsi32_si128(i0), _mm_cvtsi32_si128(i1));
                v0 = _mm_unpacklo_epi8(v0, z);
                v1 = _mm_unpacklo_epi8(v1, z);

                a0 = _mm_unpacklo_epi32(_mm_loadl_epi64((__m128i*)(wtab+FXY[x]*4)),
                                        _mm_loadl_epi64((__m128i*)(wtab+FXY[x+1]*4)));
                a1 = _mm_unpacklo_epi32(_mm_loadl_epi64((__m128i*)(wtab+FXY[x+2]*4)),
                                        _mm_loadl_epi64((__m128i*)(wtab+FXY[x+3]*4)));
                b0 = _mm_unpacklo_epi64(a0, a1);
                b1 = _mm_unpackhi_epi64(a0, a1);
                v0 = _mm_madd_epi16(v0, b0);
                v1 = _mm_madd_epi16(v1, b1);
                v0 = _mm_add_epi32(_mm_add_epi32(v0, v1), delta);

                i0 = *(ushort*)(S0 + iofs1[0]) + (*(ushort*)(S0 + iofs1[1]) << 16);
                i1 = *(ushort*)(S0 + iofs1[2]) + (*(ushort*)(S0 + iofs1[3]) << 16);
                v2 = _mm_unpacklo_epi32(_mm_cvtsi32_si128(i0), _mm_cvtsi32_si128(i1));
                i0 = *(ushort*)(S1 + iofs1[0]) + (*(ushort*)(S1 + iofs1[1]) << 16);
                i1 = *(ushort*)(S1 + iofs1[2]) + (*(ushort*)(S1 + iofs1[3]) << 16);
                v3 = _mm_unpacklo_epi32(_mm_cvtsi32_si128(i0), _mm_cvtsi32_si128(i1));
                v2 = _mm_unpacklo_epi8(v2, z);
                v3 = _mm_unpacklo_epi8(v3, z);

                a0 = _mm_unpacklo_epi32(_mm_loadl_epi64((__m128i*)(wtab+FXY[x+4]*4)),
                                        _mm_loadl_epi64((__m128i*)(wtab+FXY[x+5]*4)));
                a1 = _mm_unpacklo_epi32(_mm_loadl_epi64((__m128i*)(wtab+FXY[x+6]*4)),
                                        _mm_loadl_epi64((__m128i*)(wtab+FXY[x+7]*4)));
                b0 = _mm_unpacklo_epi64(a0, a1);
                b1 = _mm_unpackhi_epi64(a0, a1);
                v2 = _mm_madd_epi16(v2, b0);
                v3 = _mm_madd_epi16(v3, b1);
                v2 = _mm_add_epi32(_mm_add_epi32(v2, v3), delta);

                v0 = _mm_srai_epi32(v0, INTER_REMAP_COEF_BITS);
                v2 = _mm_srai_epi32(v2, INTER_REMAP_COEF_BITS);
                v0 = _mm_packus_epi16(_mm_packs_epi32(v0, v2), z);
                _mm_storel_epi64( (__m128i*)(D + x), v0 );
            }
        }
        else if( cn == 3 )
        {
            for( ; x <= width - 5; x += 4, D += 12 )
            {
                __m128i xy0 = _mm_loadu_si128( (const __m128i*)(XY + x*2));
                __m128i u0, v0, u1, v1;

                xy0 = _mm_madd_epi16( xy0, xy2ofs );
                _mm_store_si128( (__m128i*)iofs0, xy0 );
                const __m128i *w0, *w1;
                w0 = (const __m128i*)(wtab + FXY[x]*16);
                w1 = (const __m128i*)(wtab + FXY[x+1]*16);

                u0 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)(S0 + iofs0[0])),
                                       _mm_cvtsi32_si128(*(int*)(S0 + iofs0[0] + 3)));
                v0 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)(S1 + iofs0[0])),
                                       _mm_cvtsi32_si128(*(int*)(S1 + iofs0[0] + 3)));
                u1 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)(S0 + iofs0[1])),
                                       _mm_cvtsi32_si128(*(int*)(S0 + iofs0[1] + 3)));
                v1 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)(S1 + iofs0[1])),
                                       _mm_cvtsi32_si128(*(int*)(S1 + iofs0[1] + 3)));
                u0 = _mm_unpacklo_epi8(u0, z);
                v0 = _mm_unpacklo_epi8(v0, z);
                u1 = _mm_unpacklo_epi8(u1, z);
                v1 = _mm_unpacklo_epi8(v1, z);
                u0 = _mm_add_epi32(_mm_madd_epi16(u0, w0[0]), _mm_madd_epi16(v0, w0[1]));
                u1 = _mm_add_epi32(_mm_madd_epi16(u1, w1[0]), _mm_madd_epi16(v1, w1[1]));
                u0 = _mm_srai_epi32(_mm_add_epi32(u0, delta), INTER_REMAP_COEF_BITS);
                u1 = _mm_srai_epi32(_mm_add_epi32(u1, delta), INTER_REMAP_COEF_BITS);
                u0 = _mm_slli_si128(u0, 4);
                u0 = _mm_packs_epi32(u0, u1);
                u0 = _mm_packus_epi16(u0, u0);
                _mm_storel_epi64((__m128i*)D, _mm_srli_si128(u0,1));

                w0 = (const __m128i*)(wtab + FXY[x+2]*16);
                w1 = (const __m128i*)(wtab + FXY[x+3]*16);

                u0 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)(S0 + iofs0[2])),
                                       _mm_cvtsi32_si128(*(int*)(S0 + iofs0[2] + 3)));
                v0 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)(S1 + iofs0[2])),
                                       _mm_cvtsi32_si128(*(int*)(S1 + iofs0[2] + 3)));
                u1 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)(S0 + iofs0[3])),
                                       _mm_cvtsi32_si128(*(int*)(S0 + iofs0[3] + 3)));
                v1 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)(S1 + iofs0[3])),
                                       _mm_cvtsi32_si128(*(int*)(S1 + iofs0[3] + 3)));
                u0 = _mm_unpacklo_epi8(u0, z);
                v0 = _mm_unpacklo_epi8(v0, z);
                u1 = _mm_unpacklo_epi8(u1, z);
                v1 = _mm_unpacklo_epi8(v1, z);
                u0 = _mm_add_epi32(_mm_madd_epi16(u0, w0[0]), _mm_madd_epi16(v0, w0[1]));
                u1 = _mm_add_epi32(_mm_madd_epi16(u1, w1[0]), _mm_madd_epi16(v1, w1[1]));
                u0 = _mm_srai_epi32(_mm_add_epi32(u0, delta), INTER_REMAP_COEF_BITS);
                u1 = _mm_srai_epi32(_mm_add_epi32(u1, delta), INTER_REMAP_COEF_BITS);
                u0 = _mm_slli_si128(u0, 4);
                u0 = _mm_packs_epi32(u0, u1);
                u0 = _mm_packus_epi16(u0, u0);
                _mm_storel_epi64((__m128i*)(D + 6), _mm_srli_si128(u0,1));
            }
        }
        else if( cn == 4 )
        {
            for( ; x <= width - 4; x += 4, D += 16 )
            {
                __m128i xy0 = _mm_loadu_si128( (const __m128i*)(XY + x*2));
                __m128i u0, v0, u1, v1;

                xy0 = _mm_madd_epi16( xy0, xy2ofs );
                _mm_store_si128( (__m128i*)iofs0, xy0 );
                const __m128i *w0, *w1;
                w0 = (const __m128i*)(wtab + FXY[x]*16);
                w1 = (const __m128i*)(wtab + FXY[x+1]*16);

                u0 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)(S0 + iofs0[0])),
                                       _mm_cvtsi32_si128(*(int*)(S0 + iofs0[0] + 4)));
                v0 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)(S1 + iofs0[0])),
                                       _mm_cvtsi32_si128(*(int*)(S1 + iofs0[0] + 4)));
                u1 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)(S0 + iofs0[1])),
                                       _mm_cvtsi32_si128(*(int*)(S0 + iofs0[1] + 4)));
                v1 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)(S1 + iofs0[1])),
                                       _mm_cvtsi32_si128(*(int*)(S1 + iofs0[1] + 4)));
                u0 = _mm_unpacklo_epi8(u0, z);
                v0 = _mm_unpacklo_epi8(v0, z);
                u1 = _mm_unpacklo_epi8(u1, z);
                v1 = _mm_unpacklo_epi8(v1, z);
                u0 = _mm_add_epi32(_mm_madd_epi16(u0, w0[0]), _mm_madd_epi16(v0, w0[1]));
                u1 = _mm_add_epi32(_mm_madd_epi16(u1, w1[0]), _mm_madd_epi16(v1, w1[1]));
                u0 = _mm_srai_epi32(_mm_add_epi32(u0, delta), INTER_REMAP_COEF_BITS);
                u1 = _mm_srai_epi32(_mm_add_epi32(u1, delta), INTER_REMAP_COEF_BITS);
                u0 = _mm_packs_epi32(u0, u1);
                u0 = _mm_packus_epi16(u0, u0);
                _mm_storel_epi64((__m128i*)D, u0);

                w0 = (const __m128i*)(wtab + FXY[x+2]*16);
                w1 = (const __m128i*)(wtab + FXY[x+3]*16);

                u0 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)(S0 + iofs0[2])),
                                       _mm_cvtsi32_si128(*(int*)(S0 + iofs0[2] + 4)));
                v0 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)(S1 + iofs0[2])),
                                       _mm_cvtsi32_si128(*(int*)(S1 + iofs0[2] + 4)));
                u1 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)(S0 + iofs0[3])),
                                       _mm_cvtsi32_si128(*(int*)(S0 + iofs0[3] + 4)));
                v1 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)(S1 + iofs0[3])),
                                       _mm_cvtsi32_si128(*(int*)(S1 + iofs0[3] + 4)));
                u0 = _mm_unpacklo_epi8(u0, z);
                v0 = _mm_unpacklo_epi8(v0, z);
                u1 = _mm_unpacklo_epi8(u1, z);
                v1 = _mm_unpacklo_epi8(v1, z);
                u0 = _mm_add_epi32(_mm_madd_epi16(u0, w0[0]), _mm_madd_epi16(v0, w0[1]));
                u1 = _mm_add_epi32(_mm_madd_epi16(u1, w1[0]), _mm_madd_epi16(v1, w1[1]));
                u0 = _mm_srai_epi32(_mm_add_epi32(u0, delta), INTER_REMAP_COEF_BITS);
                u1 = _mm_srai_epi32(_mm_add_epi32(u1, delta), INTER_REMAP_COEF_BITS);
                u0 = _mm_packs_epi32(u0, u1);
                u0 = _mm_packus_epi16(u0, u0);
                _mm_storel_epi64((__m128i*)(D + 8), u0);
            }
        }

        return x;
    }
};

#else

typedef RemapNoVec RemapVec_8u;

#endif


template<class CastOp, class VecOp, typename AT>
static void remapBilinear( const Mat& _src, Mat& _dst, const Mat& _xy,
                           const Mat& _fxy, const void* _wtab,
                           int borderType, const Scalar& _borderValue )
{
    typedef typename CastOp::rtype T;
    typedef typename CastOp::type1 WT;
    Size ssize = _src.size(), dsize = _dst.size();
    int cn = _src.channels();
    const AT* wtab = (const AT*)_wtab;
    const T* S0 = (const T*)_src.data;
    size_t sstep = _src.step/sizeof(S0[0]);
    Scalar_<T> cval(saturate_cast<T>(_borderValue[0]),
        saturate_cast<T>(_borderValue[1]),
        saturate_cast<T>(_borderValue[2]),
        saturate_cast<T>(_borderValue[3]));
    int dx, dy;
    CastOp castOp;
    VecOp vecOp;

    unsigned width1 = std::max(ssize.width-1, 0), height1 = std::max(ssize.height-1, 0);
    CV_Assert( cn <= 4 && ssize.area() > 0 );
#if CV_SSE2
    if( _src.type() == CV_8UC3 )
        width1 = std::max(ssize.width-2, 0);
#endif

    for( dy = 0; dy < dsize.height; dy++ )
    {
        T* D = (T*)(_dst.data + _dst.step*dy);
        const short* XY = (const short*)(_xy.data + _xy.step*dy);
        const ushort* FXY = (const ushort*)(_fxy.data + _fxy.step*dy);
        int X0 = 0;
        bool prevInlier = false;

        for( dx = 0; dx <= dsize.width; dx++ )
        {
            bool curInlier = dx < dsize.width ?
                (unsigned)XY[dx*2] < width1 &&
                (unsigned)XY[dx*2+1] < height1 : !prevInlier;
            if( curInlier == prevInlier )
                continue;

            int X1 = dx;
            dx = X0;
            X0 = X1;
            prevInlier = curInlier;

            if( !curInlier )
            {
                int len = vecOp( _src, D, XY + dx*2, FXY + dx, wtab, X1 - dx );
                D += len*cn;
                dx += len;

                if( cn == 1 )
                {
                    for( ; dx < X1; dx++, D++ )
                    {
                        int sx = XY[dx*2], sy = XY[dx*2+1];
                        const AT* w = wtab + FXY[dx]*4;
                        const T* S = S0 + sy*sstep + sx;
                        *D = castOp(WT(S[0]*w[0] + S[1]*w[1] + S[sstep]*w[2] + S[sstep+1]*w[3]));
                    }
                }
                else if( cn == 2 )
                    for( ; dx < X1; dx++, D += 2 )
                    {
                        int sx = XY[dx*2], sy = XY[dx*2+1];
                        const AT* w = wtab + FXY[dx]*4;
                        const T* S = S0 + sy*sstep + sx*2;
                        WT t0 = S[0]*w[0] + S[2]*w[1] + S[sstep]*w[2] + S[sstep+2]*w[3];
                        WT t1 = S[1]*w[0] + S[3]*w[1] + S[sstep+1]*w[2] + S[sstep+3]*w[3];
                        D[0] = castOp(t0); D[1] = castOp(t1);
                    }
                else if( cn == 3 )
                    for( ; dx < X1; dx++, D += 3 )
                    {
                        int sx = XY[dx*2], sy = XY[dx*2+1];
                        const AT* w = wtab + FXY[dx]*4;
                        const T* S = S0 + sy*sstep + sx*3;
                        WT t0 = S[0]*w[0] + S[3]*w[1] + S[sstep]*w[2] + S[sstep+3]*w[3];
                        WT t1 = S[1]*w[0] + S[4]*w[1] + S[sstep+1]*w[2] + S[sstep+4]*w[3];
                        WT t2 = S[2]*w[0] + S[5]*w[1] + S[sstep+2]*w[2] + S[sstep+5]*w[3];
                        D[0] = castOp(t0); D[1] = castOp(t1); D[2] = castOp(t2);
                    }
                else
                    for( ; dx < X1; dx++, D += 4 )
                    {
                        int sx = XY[dx*2], sy = XY[dx*2+1];
                        const AT* w = wtab + FXY[dx]*4;
                        const T* S = S0 + sy*sstep + sx*4;
                        WT t0 = S[0]*w[0] + S[4]*w[1] + S[sstep]*w[2] + S[sstep+4]*w[3];
                        WT t1 = S[1]*w[0] + S[5]*w[1] + S[sstep+1]*w[2] + S[sstep+5]*w[3];
                        D[0] = castOp(t0); D[1] = castOp(t1);
                        t0 = S[2]*w[0] + S[6]*w[1] + S[sstep+2]*w[2] + S[sstep+6]*w[3];
                        t1 = S[3]*w[0] + S[7]*w[1] + S[sstep+3]*w[2] + S[sstep+7]*w[3];
                        D[2] = castOp(t0); D[3] = castOp(t1);
                    }
            }
            else
            {
                if( borderType == BORDER_TRANSPARENT && cn != 3 )
                {
                    D += (X1 - dx)*cn;
                    dx = X1;
                    continue;
                }

                if( cn == 1 )
                    for( ; dx < X1; dx++, D++ )
                    {
                        int sx = XY[dx*2], sy = XY[dx*2+1];
                        if( borderType == BORDER_CONSTANT &&
                            (sx >= ssize.width || sx+1 < 0 ||
                             sy >= ssize.height || sy+1 < 0) )
                        {
                            D[0] = cval[0];
                        }
                        else
                        {
                            int sx0, sx1, sy0, sy1;
                            T v0, v1, v2, v3;
                            const AT* w = wtab + FXY[dx]*4;
                            if( borderType == BORDER_REPLICATE )
                            {
                                sx0 = clip(sx, 0, ssize.width);
                                sx1 = clip(sx+1, 0, ssize.width);
                                sy0 = clip(sy, 0, ssize.height);
                                sy1 = clip(sy+1, 0, ssize.height);
                                v0 = S0[sy0*sstep + sx0];
                                v1 = S0[sy0*sstep + sx1];
                                v2 = S0[sy1*sstep + sx0];
                                v3 = S0[sy1*sstep + sx1];
                            }
                            else
                            {
                                sx0 = borderInterpolate(sx, ssize.width, borderType);
                                sx1 = borderInterpolate(sx+1, ssize.width, borderType);
                                sy0 = borderInterpolate(sy, ssize.height, borderType);
                                sy1 = borderInterpolate(sy+1, ssize.height, borderType);
                                v0 = sx0 >= 0 && sy0 >= 0 ? S0[sy0*sstep + sx0] : cval[0];
                                v1 = sx1 >= 0 && sy0 >= 0 ? S0[sy0*sstep + sx1] : cval[0];
                                v2 = sx0 >= 0 && sy1 >= 0 ? S0[sy1*sstep + sx0] : cval[0];
                                v3 = sx1 >= 0 && sy1 >= 0 ? S0[sy1*sstep + sx1] : cval[0];
                            }
                            D[0] = castOp(WT(v0*w[0] + v1*w[1] + v2*w[2] + v3*w[3]));
                        }
                    }
                else
                    for( ; dx < X1; dx++, D += cn )
                    {
                        int sx = XY[dx*2], sy = XY[dx*2+1], k;
                        if( borderType == BORDER_CONSTANT &&
                            (sx >= ssize.width || sx+1 < 0 ||
                             sy >= ssize.height || sy+1 < 0) )
                        {
                            for( k = 0; k < cn; k++ )
                                D[k] = cval[k];
                        }
                        else
                        {
                            int sx0, sx1, sy0, sy1;
                            const T *v0, *v1, *v2, *v3;
                            const AT* w = wtab + FXY[dx]*4;
                            if( borderType == BORDER_REPLICATE )
                            {
                                sx0 = clip(sx, 0, ssize.width);
                                sx1 = clip(sx+1, 0, ssize.width);
                                sy0 = clip(sy, 0, ssize.height);
                                sy1 = clip(sy+1, 0, ssize.height);
                                v0 = S0 + sy0*sstep + sx0*cn;
                                v1 = S0 + sy0*sstep + sx1*cn;
                                v2 = S0 + sy1*sstep + sx0*cn;
                                v3 = S0 + sy1*sstep + sx1*cn;
                            }
                            else if( borderType == BORDER_TRANSPARENT &&
                                ((unsigned)sx >= (unsigned)(ssize.width-1) ||
                                (unsigned)sy >= (unsigned)(ssize.height-1)))
                                continue;
                            else
                            {
                                sx0 = borderInterpolate(sx, ssize.width, borderType);
                                sx1 = borderInterpolate(sx+1, ssize.width, borderType);
                                sy0 = borderInterpolate(sy, ssize.height, borderType);
                                sy1 = borderInterpolate(sy+1, ssize.height, borderType);
                                v0 = sx0 >= 0 && sy0 >= 0 ? S0 + sy0*sstep + sx0*cn : &cval[0];
                                v1 = sx1 >= 0 && sy0 >= 0 ? S0 + sy0*sstep + sx1*cn : &cval[0];
                                v2 = sx0 >= 0 && sy1 >= 0 ? S0 + sy1*sstep + sx0*cn : &cval[0];
                                v3 = sx1 >= 0 && sy1 >= 0 ? S0 + sy1*sstep + sx1*cn : &cval[0];
                            }
                            for( k = 0; k < cn; k++ )
                                D[k] = castOp(WT(v0[k]*w[0] + v1[k]*w[1] + v2[k]*w[2] + v3[k]*w[3]));
                        }
                    }
            }
        }
    }
}


template<class CastOp, typename AT, int ONE>
static void remapBicubic( const Mat& _src, Mat& _dst, const Mat& _xy,
                          const Mat& _fxy, const void* _wtab,
                          int borderType, const Scalar& _borderValue )
{
    typedef typename CastOp::rtype T;
    typedef typename CastOp::type1 WT;
    Size ssize = _src.size(), dsize = _dst.size();
    int cn = _src.channels();
    const AT* wtab = (const AT*)_wtab;
    const T* S0 = (const T*)_src.data;
    size_t sstep = _src.step/sizeof(S0[0]);
    Scalar_<T> cval(saturate_cast<T>(_borderValue[0]),
        saturate_cast<T>(_borderValue[1]),
        saturate_cast<T>(_borderValue[2]),
        saturate_cast<T>(_borderValue[3]));
    int dx, dy;
    CastOp castOp;
    int borderType1 = borderType != BORDER_TRANSPARENT ? borderType : BORDER_REFLECT_101;

    unsigned width1 = std::max(ssize.width-3, 0), height1 = std::max(ssize.height-3, 0);

    if( _dst.isContinuous() && _xy.isContinuous() && _fxy.isContinuous() )
    {
        dsize.width *= dsize.height;
        dsize.height = 1;
    }

    for( dy = 0; dy < dsize.height; dy++ )
    {
        T* D = (T*)(_dst.data + _dst.step*dy);
        const short* XY = (const short*)(_xy.data + _xy.step*dy);
        const ushort* FXY = (const ushort*)(_fxy.data + _fxy.step*dy);

        for( dx = 0; dx < dsize.width; dx++, D += cn )
        {
            int sx = XY[dx*2]-1, sy = XY[dx*2+1]-1;
            const AT* w = wtab + FXY[dx]*16;
            int i, k;
            if( (unsigned)sx < width1 && (unsigned)sy < height1 )
            {
                const T* S = S0 + sy*sstep + sx*cn;
                for( k = 0; k < cn; k++ )
                {
                    WT sum = S[0]*w[0] + S[cn]*w[1] + S[cn*2]*w[2] + S[cn*3]*w[3];
                    S += sstep;
                    sum += S[0]*w[4] + S[cn]*w[5] + S[cn*2]*w[6] + S[cn*3]*w[7];
                    S += sstep;
                    sum += S[0]*w[8] + S[cn]*w[9] + S[cn*2]*w[10] + S[cn*3]*w[11];
                    S += sstep;
                    sum += S[0]*w[12] + S[cn]*w[13] + S[cn*2]*w[14] + S[cn*3]*w[15];
                    S += 1 - sstep*3;
                    D[k] = castOp(sum);
                }
            }
            else
            {
                int x[4], y[4];
                if( borderType == BORDER_TRANSPARENT &&
                    ((unsigned)(sx+1) >= (unsigned)ssize.width ||
                    (unsigned)(sy+1) >= (unsigned)ssize.height) )
                    continue;

                if( borderType1 == BORDER_CONSTANT &&
                    (sx >= ssize.width || sx+4 <= 0 ||
                    sy >= ssize.height || sy+4 <= 0))
                {
                    for( k = 0; k < cn; k++ )
                        D[k] = cval[k];
                    continue;
                }

                for( i = 0; i < 4; i++ )
                {
                    x[i] = borderInterpolate(sx + i, ssize.width, borderType1)*cn;
                    y[i] = borderInterpolate(sy + i, ssize.height, borderType1);
                }

                for( k = 0; k < cn; k++, S0++, w -= 16 )
                {
                    WT cv = cval[k], sum = cv*ONE;
                    for( i = 0; i < 4; i++, w += 4 )
                    {
                        int yi = y[i];
                        const T* S = S0 + yi*sstep;
                        if( yi < 0 )
                            continue;
                        if( x[0] >= 0 )
                            sum += (S[x[0]] - cv)*w[0];
                        if( x[1] >= 0 )
                            sum += (S[x[1]] - cv)*w[1];
                        if( x[2] >= 0 )
                            sum += (S[x[2]] - cv)*w[2];
                        if( x[3] >= 0 )
                            sum += (S[x[3]] - cv)*w[3];
                    }
                    D[k] = castOp(sum);
                }
                S0 -= cn;
            }
        }
    }
}


template<class CastOp, typename AT, int ONE>
static void remapLanczos4( const Mat& _src, Mat& _dst, const Mat& _xy,
                           const Mat& _fxy, const void* _wtab,
                           int borderType, const Scalar& _borderValue )
{
    typedef typename CastOp::rtype T;
    typedef typename CastOp::type1 WT;
    Size ssize = _src.size(), dsize = _dst.size();
    int cn = _src.channels();
    const AT* wtab = (const AT*)_wtab;
    const T* S0 = (const T*)_src.data;
    size_t sstep = _src.step/sizeof(S0[0]);
    Scalar_<T> cval(saturate_cast<T>(_borderValue[0]),
        saturate_cast<T>(_borderValue[1]),
        saturate_cast<T>(_borderValue[2]),
        saturate_cast<T>(_borderValue[3]));
    int dx, dy;
    CastOp castOp;
    int borderType1 = borderType != BORDER_TRANSPARENT ? borderType : BORDER_REFLECT_101;

    unsigned width1 = std::max(ssize.width-7, 0), height1 = std::max(ssize.height-7, 0);

    if( _dst.isContinuous() && _xy.isContinuous() && _fxy.isContinuous() )
    {
        dsize.width *= dsize.height;
        dsize.height = 1;
    }

    for( dy = 0; dy < dsize.height; dy++ )
    {
        T* D = (T*)(_dst.data + _dst.step*dy);
        const short* XY = (const short*)(_xy.data + _xy.step*dy);
        const ushort* FXY = (const ushort*)(_fxy.data + _fxy.step*dy);

        for( dx = 0; dx < dsize.width; dx++, D += cn )
        {
            int sx = XY[dx*2]-3, sy = XY[dx*2+1]-3;
            const AT* w = wtab + FXY[dx]*64;
            const T* S = S0 + sy*sstep + sx*cn;
            int i, k;
            if( (unsigned)sx < width1 && (unsigned)sy < height1 )
            {
                for( k = 0; k < cn; k++ )
                {
                    WT sum = 0;
                    for( int r = 0; r < 8; r++, S += sstep, w += 8 )
                        sum += S[0]*w[0] + S[cn]*w[1] + S[cn*2]*w[2] + S[cn*3]*w[3] +
                            S[cn*4]*w[4] + S[cn*5]*w[5] + S[cn*6]*w[6] + S[cn*7]*w[7];
                    w -= 64;
                    S -= sstep*8 - 1;
                    D[k] = castOp(sum);
                }
            }
            else
            {
                int x[8], y[8];
                if( borderType == BORDER_TRANSPARENT &&
                    ((unsigned)(sx+3) >= (unsigned)ssize.width ||
                    (unsigned)(sy+3) >= (unsigned)ssize.height) )
                    continue;

                if( borderType1 == BORDER_CONSTANT &&
                    (sx >= ssize.width || sx+8 <= 0 ||
                    sy >= ssize.height || sy+8 <= 0))
                {
                    for( k = 0; k < cn; k++ )
                        D[k] = cval[k];
                    continue;
                }

                for( i = 0; i < 8; i++ )
                {
                    x[i] = borderInterpolate(sx + i, ssize.width, borderType1)*cn;
                    y[i] = borderInterpolate(sy + i, ssize.height, borderType1);
                }

                for( k = 0; k < cn; k++, S0++, w -= 64 )
                {
                    WT cv = cval[k], sum = cv*ONE;
                    for( i = 0; i < 8; i++, w += 8 )
                    {
                        int yi = y[i];
                        const T* S1 = S0 + yi*sstep;
                        if( yi < 0 )
                            continue;
                        if( x[0] >= 0 )
                            sum += (S1[x[0]] - cv)*w[0];
                        if( x[1] >= 0 )
                            sum += (S1[x[1]] - cv)*w[1];
                        if( x[2] >= 0 )
                            sum += (S1[x[2]] - cv)*w[2];
                        if( x[3] >= 0 )
                            sum += (S1[x[3]] - cv)*w[3];
                        if( x[4] >= 0 )
                            sum += (S1[x[4]] - cv)*w[4];
                        if( x[5] >= 0 )
                            sum += (S1[x[5]] - cv)*w[5];
                        if( x[6] >= 0 )
                            sum += (S1[x[6]] - cv)*w[6];
                        if( x[7] >= 0 )
                            sum += (S1[x[7]] - cv)*w[7];
                    }
                    D[k] = castOp(sum);
                }
                S0 -= cn;
            }
        }
    }
}


typedef void (*RemapNNFunc)(const Mat& _src, Mat& _dst, const Mat& _xy,
                            int borderType, const Scalar& _borderValue );

typedef void (*RemapFunc)(const Mat& _src, Mat& _dst, const Mat& _xy,
                          const Mat& _fxy, const void* _wtab,
                          int borderType, const Scalar& _borderValue);

class RemapInvoker :
    public ParallelLoopBody
{
public:
    RemapInvoker(const Mat& _src, Mat& _dst, const Mat *_m1,
                 const Mat *_m2, int _borderType, const Scalar &_borderValue,
                 int _planar_input, RemapNNFunc _nnfunc, RemapFunc _ifunc, const void *_ctab) :
        ParallelLoopBody(), src(&_src), dst(&_dst), m1(_m1), m2(_m2),
        borderType(_borderType), borderValue(_borderValue),
        planar_input(_planar_input), nnfunc(_nnfunc), ifunc(_ifunc), ctab(_ctab)
    {
    }

    virtual void operator() (const Range& range) const
    {
        int x, y, x1, y1;
        const int buf_size = 1 << 14;
        int brows0 = std::min(128, dst->rows), map_depth = m1->depth();
        int bcols0 = std::min(buf_size/brows0, dst->cols);
        brows0 = std::min(buf_size/bcols0, dst->rows);
    #if CV_SSE2
        bool useSIMD = checkHardwareSupport(CV_CPU_SSE2);
    #endif

        Mat _bufxy(brows0, bcols0, CV_16SC2), _bufa;
        if( !nnfunc )
            _bufa.create(brows0, bcols0, CV_16UC1);

        for( y = range.start; y < range.end; y += brows0 )
        {
            for( x = 0; x < dst->cols; x += bcols0 )
            {
                int brows = std::min(brows0, range.end - y);
                int bcols = std::min(bcols0, dst->cols - x);
                Mat dpart(*dst, Rect(x, y, bcols, brows));
                Mat bufxy(_bufxy, Rect(0, 0, bcols, brows));

                if( nnfunc )
                {
                    if( m1->type() == CV_16SC2 && !m2->data ) // the data is already in the right format
                        bufxy = (*m1)(Rect(x, y, bcols, brows));
                    else if( map_depth != CV_32F )
                    {
                        for( y1 = 0; y1 < brows; y1++ )
                        {
                            short* XY = (short*)(bufxy.data + bufxy.step*y1);
                            const short* sXY = (const short*)(m1->data + m1->step*(y+y1)) + x*2;
                            const ushort* sA = (const ushort*)(m2->data + m2->step*(y+y1)) + x;

                            for( x1 = 0; x1 < bcols; x1++ )
                            {
                                int a = sA[x1] & (INTER_TAB_SIZE2-1);
                                XY[x1*2] = sXY[x1*2] + NNDeltaTab_i[a][0];
                                XY[x1*2+1] = sXY[x1*2+1] + NNDeltaTab_i[a][1];
                            }
                        }
                    }
                    else if( !planar_input )
                        (*m1)(Rect(x, y, bcols, brows)).convertTo(bufxy, bufxy.depth());
                    else
                    {
                        for( y1 = 0; y1 < brows; y1++ )
                        {
                            short* XY = (short*)(bufxy.data + bufxy.step*y1);
                            const float* sX = (const float*)(m1->data + m1->step*(y+y1)) + x;
                            const float* sY = (const float*)(m2->data + m2->step*(y+y1)) + x;
                            x1 = 0;

                        #if CV_SSE2
                            if( useSIMD )
                            {
                                for( ; x1 <= bcols - 8; x1 += 8 )
                                {
                                    __m128 fx0 = _mm_loadu_ps(sX + x1);
                                    __m128 fx1 = _mm_loadu_ps(sX + x1 + 4);
                                    __m128 fy0 = _mm_loadu_ps(sY + x1);
                                    __m128 fy1 = _mm_loadu_ps(sY + x1 + 4);
                                    __m128i ix0 = _mm_cvtps_epi32(fx0);
                                    __m128i ix1 = _mm_cvtps_epi32(fx1);
                                    __m128i iy0 = _mm_cvtps_epi32(fy0);
                                    __m128i iy1 = _mm_cvtps_epi32(fy1);
                                    ix0 = _mm_packs_epi32(ix0, ix1);
                                    iy0 = _mm_packs_epi32(iy0, iy1);
                                    ix1 = _mm_unpacklo_epi16(ix0, iy0);
                                    iy1 = _mm_unpackhi_epi16(ix0, iy0);
                                    _mm_storeu_si128((__m128i*)(XY + x1*2), ix1);
                                    _mm_storeu_si128((__m128i*)(XY + x1*2 + 8), iy1);
                                }
                            }
                        #endif

                            for( ; x1 < bcols; x1++ )
                            {
                                XY[x1*2] = saturate_cast<short>(sX[x1]);
                                XY[x1*2+1] = saturate_cast<short>(sY[x1]);
                            }
                        }
                    }
                    nnfunc( *src, dpart, bufxy, borderType, borderValue );
                    continue;
                }

                Mat bufa(_bufa, Rect(0, 0, bcols, brows));
                for( y1 = 0; y1 < brows; y1++ )
                {
                    short* XY = (short*)(bufxy.data + bufxy.step*y1);
                    ushort* A = (ushort*)(bufa.data + bufa.step*y1);

                    if( m1->type() == CV_16SC2 && (m2->type() == CV_16UC1 || m2->type() == CV_16SC1) )
                    {
                        bufxy = (*m1)(Rect(x, y, bcols, brows));

                        const ushort* sA = (const ushort*)(m2->data + m2->step*(y+y1)) + x;
                        for( x1 = 0; x1 < bcols; x1++ )
                            A[x1] = (ushort)(sA[x1] & (INTER_TAB_SIZE2-1));
                    }
                    else if( planar_input )
                    {
                        const float* sX = (const float*)(m1->data + m1->step*(y+y1)) + x;
                        const float* sY = (const float*)(m2->data + m2->step*(y+y1)) + x;

                        x1 = 0;
                    #if CV_SSE2
                        if( useSIMD )
                        {
                            __m128 scale = _mm_set1_ps((float)INTER_TAB_SIZE);
                            __m128i mask = _mm_set1_epi32(INTER_TAB_SIZE-1);
                            for( ; x1 <= bcols - 8; x1 += 8 )
                            {
                                __m128 fx0 = _mm_loadu_ps(sX + x1);
                                __m128 fx1 = _mm_loadu_ps(sX + x1 + 4);
                                __m128 fy0 = _mm_loadu_ps(sY + x1);
                                __m128 fy1 = _mm_loadu_ps(sY + x1 + 4);
                                __m128i ix0 = _mm_cvtps_epi32(_mm_mul_ps(fx0, scale));
                                __m128i ix1 = _mm_cvtps_epi32(_mm_mul_ps(fx1, scale));
                                __m128i iy0 = _mm_cvtps_epi32(_mm_mul_ps(fy0, scale));
                                __m128i iy1 = _mm_cvtps_epi32(_mm_mul_ps(fy1, scale));
                                __m128i mx0 = _mm_and_si128(ix0, mask);
                                __m128i mx1 = _mm_and_si128(ix1, mask);
                                __m128i my0 = _mm_and_si128(iy0, mask);
                                __m128i my1 = _mm_and_si128(iy1, mask);
                                mx0 = _mm_packs_epi32(mx0, mx1);
                                my0 = _mm_packs_epi32(my0, my1);
                                my0 = _mm_slli_epi16(my0, INTER_BITS);
                                mx0 = _mm_or_si128(mx0, my0);
                                _mm_storeu_si128((__m128i*)(A + x1), mx0);
                                ix0 = _mm_srai_epi32(ix0, INTER_BITS);
                                ix1 = _mm_srai_epi32(ix1, INTER_BITS);
                                iy0 = _mm_srai_epi32(iy0, INTER_BITS);
                                iy1 = _mm_srai_epi32(iy1, INTER_BITS);
                                ix0 = _mm_packs_epi32(ix0, ix1);
                                iy0 = _mm_packs_epi32(iy0, iy1);
                                ix1 = _mm_unpacklo_epi16(ix0, iy0);
                                iy1 = _mm_unpackhi_epi16(ix0, iy0);
                                _mm_storeu_si128((__m128i*)(XY + x1*2), ix1);
                                _mm_storeu_si128((__m128i*)(XY + x1*2 + 8), iy1);
                            }
                        }
                    #endif

                        for( ; x1 < bcols; x1++ )
                        {
                            int sx = cvRound(sX[x1]*INTER_TAB_SIZE);
                            int sy = cvRound(sY[x1]*INTER_TAB_SIZE);
                            int v = (sy & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE-1));
                            XY[x1*2] = saturate_cast<short>(sx >> INTER_BITS);
                            XY[x1*2+1] = saturate_cast<short>(sy >> INTER_BITS);
                            A[x1] = (ushort)v;
                        }
                    }
                    else
                    {
                        const float* sXY = (const float*)(m1->data + m1->step*(y+y1)) + x*2;

                        for( x1 = 0; x1 < bcols; x1++ )
                        {
                            int sx = cvRound(sXY[x1*2]*INTER_TAB_SIZE);
                            int sy = cvRound(sXY[x1*2+1]*INTER_TAB_SIZE);
                            int v = (sy & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE-1));
                            XY[x1*2] = saturate_cast<short>(sx >> INTER_BITS);
                            XY[x1*2+1] = saturate_cast<short>(sy >> INTER_BITS);
                            A[x1] = (ushort)v;
                        }
                    }
                }
                ifunc(*src, dpart, bufxy, bufa, ctab, borderType, borderValue);
            }
        }
    }

private:
    const Mat* src;
    Mat* dst;
    const Mat *m1, *m2;
    int borderType;
    Scalar borderValue;
    int planar_input;
    RemapNNFunc nnfunc;
    RemapFunc ifunc;
    const void *ctab;
};

#ifdef HAVE_OPENCL

static bool ocl_remap(InputArray _src, OutputArray _dst, InputArray _map1, InputArray _map2,
                      int interpolation, int borderType, const Scalar& borderValue)
{
    int cn = _src.channels(), type = _src.type(), depth = _src.depth();

    if (borderType == BORDER_TRANSPARENT || !(interpolation == INTER_LINEAR || interpolation == INTER_NEAREST)
            || _map1.type() == CV_16SC1 || _map2.type() == CV_16SC1)
        return false;

    UMat src = _src.getUMat(), map1 = _map1.getUMat(), map2 = _map2.getUMat();

    if( (map1.type() == CV_16SC2 && (map2.type() == CV_16UC1 || map2.empty())) ||
        (map2.type() == CV_16SC2 && (map1.type() == CV_16UC1 || map1.empty())) )
    {
        if (map1.type() != CV_16SC2)
            std::swap(map1, map2);
    }
    else
        CV_Assert( map1.type() == CV_32FC2 || (map1.type() == CV_32FC1 && map2.type() == CV_32FC1) );

    _dst.create(map1.size(), type);
    UMat dst = _dst.getUMat();

    String kernelName = "remap";
    if (map1.type() == CV_32FC2 && map2.empty())
        kernelName += "_32FC2";
    else if (map1.type() == CV_16SC2)
    {
        kernelName += "_16SC2";
        if (!map2.empty())
            kernelName += "_16UC1";
    }
    else if (map1.type() == CV_32FC1 && map2.type() == CV_32FC1)
        kernelName += "_2_32FC1";
    else
        CV_Error(Error::StsBadArg, "Unsupported map types");

    static const char * const interMap[] = { "INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_LINEAR", "INTER_LANCZOS" };
    static const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", "BORDER_WRAP",
                           "BORDER_REFLECT_101", "BORDER_TRANSPARENT" };
    String buildOptions = format("-D %s -D %s -D T=%s", interMap[interpolation], borderMap[borderType], ocl::typeToStr(type));

    if (interpolation != INTER_NEAREST)
    {
        char cvt[3][40];
        int wdepth = std::max(CV_32F, dst.depth());
        buildOptions = buildOptions
                      + format(" -D WT=%s -D convertToT=%s -D convertToWT=%s"
                               " -D convertToWT2=%s -D WT2=%s",
                               ocl::typeToStr(CV_MAKE_TYPE(wdepth, cn)),
                               ocl::convertTypeStr(wdepth, depth, cn, cvt[0]),
                               ocl::convertTypeStr(depth, wdepth, cn, cvt[1]),
                               ocl::convertTypeStr(CV_32S, wdepth, 2, cvt[2]),
                               ocl::typeToStr(CV_MAKE_TYPE(wdepth, 2)));
    }
    int scalarcn = cn == 3 ? 4 : cn;
    int sctype = CV_MAKETYPE(depth, scalarcn);
    buildOptions += format(" -D T=%s -D T1=%s"
                           " -D cn=%d -D ST=%s",
                           ocl::typeToStr(type), ocl::typeToStr(depth),
                           cn, ocl::typeToStr(sctype));

    ocl::Kernel k(kernelName.c_str(), ocl::imgproc::remap_oclsrc, buildOptions);

    Mat scalar(1, 1, sctype, borderValue);
    ocl::KernelArg srcarg = ocl::KernelArg::ReadOnly(src), dstarg = ocl::KernelArg::WriteOnly(dst),
            map1arg = ocl::KernelArg::ReadOnlyNoSize(map1),
            scalararg = ocl::KernelArg::Constant((void*)scalar.data, scalar.elemSize());

    if (map2.empty())
        k.args(srcarg, dstarg, map1arg, scalararg);
    else
        k.args(srcarg, dstarg, map1arg, ocl::KernelArg::ReadOnlyNoSize(map2), scalararg);

    size_t globalThreads[2] = { dst.cols, dst.rows };
    return k.run(2, globalThreads, NULL, false);
}

#endif

#if IPP_VERSION_X100 >= 0 && !defined HAVE_IPP_ICV_ONLY && 0

typedef IppStatus (CV_STDCALL * ippiRemap)(const void * pSrc, IppiSize srcSize, int srcStep, IppiRect srcRoi,
                                           const Ipp32f* pxMap, int xMapStep, const Ipp32f* pyMap, int yMapStep,
                                           void * pDst, int dstStep, IppiSize dstRoiSize, int interpolation);

class IPPRemapInvoker :
        public ParallelLoopBody
{
public:
    IPPRemapInvoker(Mat & _src, Mat & _dst, Mat & _xmap, Mat & _ymap, ippiRemap _ippFunc,
                    int _ippInterpolation, int _borderType, const Scalar & _borderValue, bool * _ok) :
        ParallelLoopBody(), src(_src), dst(_dst), map1(_xmap), map2(_ymap), ippFunc(_ippFunc),
        ippInterpolation(_ippInterpolation), borderType(_borderType), borderValue(_borderValue), ok(_ok)
    {
        *ok = true;
    }

    virtual void operator() (const Range & range) const
    {
        IppiRect srcRoiRect = { 0, 0, src.cols, src.rows };
        Mat dstRoi = dst.rowRange(range);
        IppiSize dstRoiSize = ippiSize(dstRoi.size());
        int type = dst.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

        if (borderType == BORDER_CONSTANT &&
                !IPPSet(borderValue, dstRoi.data, (int)dstRoi.step, dstRoiSize, cn, depth))
        {
            *ok = false;
            return;
        }

        if (ippFunc(src.data, ippiSize(src.size()), (int)src.step, srcRoiRect,
                    (const Ipp32f *)map1.data, (int)map1.step, (const Ipp32f *)map2.data, (int)map2.step,
                    dstRoi.data, (int)dstRoi.step, dstRoiSize, ippInterpolation) < 0)
            *ok = false;
    }

private:
    Mat & src, & dst, & map1, & map2;
    ippiRemap ippFunc;
    int ippInterpolation, borderType;
    Scalar borderValue;
    bool * ok;
};

#endif

}

void cv::remap( InputArray _src, OutputArray _dst,
                InputArray _map1, InputArray _map2,
                int interpolation, int borderType, const Scalar& borderValue )
{
    static RemapNNFunc nn_tab[] =
    {
        remapNearest<uchar>, remapNearest<schar>, remapNearest<ushort>, remapNearest<short>,
        remapNearest<int>, remapNearest<float>, remapNearest<double>, 0
    };

    static RemapFunc linear_tab[] =
    {
        remapBilinear<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, RemapVec_8u, short>, 0,
        remapBilinear<Cast<float, ushort>, RemapNoVec, float>,
        remapBilinear<Cast<float, short>, RemapNoVec, float>, 0,
        remapBilinear<Cast<float, float>, RemapNoVec, float>,
        remapBilinear<Cast<double, double>, RemapNoVec, float>, 0
    };

    static RemapFunc cubic_tab[] =
    {
        remapBicubic<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, short, INTER_REMAP_COEF_SCALE>, 0,
        remapBicubic<Cast<float, ushort>, float, 1>,
        remapBicubic<Cast<float, short>, float, 1>, 0,
        remapBicubic<Cast<float, float>, float, 1>,
        remapBicubic<Cast<double, double>, float, 1>, 0
    };

    static RemapFunc lanczos4_tab[] =
    {
        remapLanczos4<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, short, INTER_REMAP_COEF_SCALE>, 0,
        remapLanczos4<Cast<float, ushort>, float, 1>,
        remapLanczos4<Cast<float, short>, float, 1>, 0,
        remapLanczos4<Cast<float, float>, float, 1>,
        remapLanczos4<Cast<double, double>, float, 1>, 0
    };

    CV_Assert( _map1.size().area() > 0 );
    CV_Assert( _map2.empty() || (_map2.size() == _map1.size()));

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_remap(_src, _dst, _map1, _map2, interpolation, borderType, borderValue))

    Mat src = _src.getMat(), map1 = _map1.getMat(), map2 = _map2.getMat();
    _dst.create( map1.size(), src.type() );
    Mat dst = _dst.getMat();
    if( dst.data == src.data )
        src = src.clone();

    if( interpolation == INTER_AREA )
        interpolation = INTER_LINEAR;

    int type = src.type(), depth = CV_MAT_DEPTH(type);

#if IPP_VERSION_X100 >= 0 && !defined HAVE_IPP_ICV_ONLY && 0
    if ((interpolation == INTER_LINEAR || interpolation == INTER_CUBIC || interpolation == INTER_NEAREST) &&
            map1.type() == CV_32FC1 && map2.type() == CV_32FC1 &&
            (borderType == BORDER_CONSTANT || borderType == BORDER_TRANSPARENT))
    {
        int ippInterpolation =
            interpolation == INTER_NEAREST ? IPPI_INTER_NN :
            interpolation == INTER_LINEAR ? IPPI_INTER_LINEAR : IPPI_INTER_CUBIC;

        ippiRemap ippFunc =
            type == CV_8UC1 ? (ippiRemap)ippiRemap_8u_C1R :
            type == CV_8UC3 ? (ippiRemap)ippiRemap_8u_C3R :
            type == CV_8UC4 ? (ippiRemap)ippiRemap_8u_C4R :
            type == CV_16UC1 ? (ippiRemap)ippiRemap_16u_C1R :
            type == CV_16UC3 ? (ippiRemap)ippiRemap_16u_C3R :
            type == CV_16UC4 ? (ippiRemap)ippiRemap_16u_C4R :
            type == CV_32FC1 ? (ippiRemap)ippiRemap_32f_C1R :
            type == CV_32FC3 ? (ippiRemap)ippiRemap_32f_C3R :
            type == CV_32FC4 ? (ippiRemap)ippiRemap_32f_C4R : 0;

        if (ippFunc)
        {
            bool ok;
            IPPRemapInvoker invoker(src, dst, map1, map2, ippFunc, ippInterpolation,
                                    borderType, borderValue, &ok);
            Range range(0, dst.rows);
            parallel_for_(range, invoker, dst.total() / (double)(1 << 16));

            if (ok)
                return;
            setIppErrorStatus();
        }
    }
#endif

    RemapNNFunc nnfunc = 0;
    RemapFunc ifunc = 0;
    const void* ctab = 0;
    bool fixpt = depth == CV_8U;
    bool planar_input = false;

    if( interpolation == INTER_NEAREST )
    {
        nnfunc = nn_tab[depth];
        CV_Assert( nnfunc != 0 );
    }
    else
    {
        if( interpolation == INTER_LINEAR )
            ifunc = linear_tab[depth];
        else if( interpolation == INTER_CUBIC )
            ifunc = cubic_tab[depth];
        else if( interpolation == INTER_LANCZOS4 )
            ifunc = lanczos4_tab[depth];
        else
            CV_Error( CV_StsBadArg, "Unknown interpolation method" );
        CV_Assert( ifunc != 0 );
        ctab = initInterTab2D( interpolation, fixpt );
    }

    const Mat *m1 = &map1, *m2 = &map2;

    if( (map1.type() == CV_16SC2 && (map2.type() == CV_16UC1 || map2.type() == CV_16SC1 || !map2.data)) ||
        (map2.type() == CV_16SC2 && (map1.type() == CV_16UC1 || map1.type() == CV_16SC1 || !map1.data)) )
    {
        if( map1.type() != CV_16SC2 )
            std::swap(m1, m2);
    }
    else
    {
        CV_Assert( ((map1.type() == CV_32FC2 || map1.type() == CV_16SC2) && !map2.data) ||
            (map1.type() == CV_32FC1 && map2.type() == CV_32FC1) );
        planar_input = map1.channels() == 1;
    }

    RemapInvoker invoker(src, dst, m1, m2,
                         borderType, borderValue, planar_input, nnfunc, ifunc,
                         ctab);
    parallel_for_(Range(0, dst.rows), invoker, dst.total()/(double)(1<<16));
}


void cv::convertMaps( InputArray _map1, InputArray _map2,
                      OutputArray _dstmap1, OutputArray _dstmap2,
                      int dstm1type, bool nninterpolate )
{
    Mat map1 = _map1.getMat(), map2 = _map2.getMat(), dstmap1, dstmap2;
    Size size = map1.size();
    const Mat *m1 = &map1, *m2 = &map2;
    int m1type = m1->type(), m2type = m2->type();

    CV_Assert( (m1type == CV_16SC2 && (nninterpolate || m2type == CV_16UC1 || m2type == CV_16SC1)) ||
               (m2type == CV_16SC2 && (nninterpolate || m1type == CV_16UC1 || m1type == CV_16SC1)) ||
               (m1type == CV_32FC1 && m2type == CV_32FC1) ||
               (m1type == CV_32FC2 && !m2->data) );

    if( m2type == CV_16SC2 )
    {
        std::swap( m1, m2 );
        std::swap( m1type, m2type );
    }

    if( dstm1type <= 0 )
        dstm1type = m1type == CV_16SC2 ? CV_32FC2 : CV_16SC2;
    CV_Assert( dstm1type == CV_16SC2 || dstm1type == CV_32FC1 || dstm1type == CV_32FC2 );
    _dstmap1.create( size, dstm1type );
    dstmap1 = _dstmap1.getMat();

    if( !nninterpolate && dstm1type != CV_32FC2 )
    {
        _dstmap2.create( size, dstm1type == CV_16SC2 ? CV_16UC1 : CV_32FC1 );
        dstmap2 = _dstmap2.getMat();
    }
    else
        _dstmap2.release();

    if( m1type == dstm1type || (nninterpolate &&
        ((m1type == CV_16SC2 && dstm1type == CV_32FC2) ||
        (m1type == CV_32FC2 && dstm1type == CV_16SC2))) )
    {
        m1->convertTo( dstmap1, dstmap1.type() );
        if( dstmap2.data && dstmap2.type() == m2->type() )
            m2->copyTo( dstmap2 );
        return;
    }

    if( m1type == CV_32FC1 && dstm1type == CV_32FC2 )
    {
        Mat vdata[] = { *m1, *m2 };
        merge( vdata, 2, dstmap1 );
        return;
    }

    if( m1type == CV_32FC2 && dstm1type == CV_32FC1 )
    {
        Mat mv[] = { dstmap1, dstmap2 };
        split( *m1, mv );
        return;
    }

    if( m1->isContinuous() && (!m2->data || m2->isContinuous()) &&
        dstmap1.isContinuous() && (!dstmap2.data || dstmap2.isContinuous()) )
    {
        size.width *= size.height;
        size.height = 1;
    }

    const float scale = 1.f/INTER_TAB_SIZE;
    int x, y;
    for( y = 0; y < size.height; y++ )
    {
        const float* src1f = (const float*)(m1->data + m1->step*y);
        const float* src2f = (const float*)(m2->data + m2->step*y);
        const short* src1 = (const short*)src1f;
        const ushort* src2 = (const ushort*)src2f;

        float* dst1f = (float*)(dstmap1.data + dstmap1.step*y);
        float* dst2f = (float*)(dstmap2.data + dstmap2.step*y);
        short* dst1 = (short*)dst1f;
        ushort* dst2 = (ushort*)dst2f;

        if( m1type == CV_32FC1 && dstm1type == CV_16SC2 )
        {
            if( nninterpolate )
                for( x = 0; x < size.width; x++ )
                {
                    dst1[x*2] = saturate_cast<short>(src1f[x]);
                    dst1[x*2+1] = saturate_cast<short>(src2f[x]);
                }
            else
                for( x = 0; x < size.width; x++ )
                {
                    int ix = saturate_cast<int>(src1f[x]*INTER_TAB_SIZE);
                    int iy = saturate_cast<int>(src2f[x]*INTER_TAB_SIZE);
                    dst1[x*2] = saturate_cast<short>(ix >> INTER_BITS);
                    dst1[x*2+1] = saturate_cast<short>(iy >> INTER_BITS);
                    dst2[x] = (ushort)((iy & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (ix & (INTER_TAB_SIZE-1)));
                }
        }
        else if( m1type == CV_32FC2 && dstm1type == CV_16SC2 )
        {
            if( nninterpolate )
                for( x = 0; x < size.width; x++ )
                {
                    dst1[x*2] = saturate_cast<short>(src1f[x*2]);
                    dst1[x*2+1] = saturate_cast<short>(src1f[x*2+1]);
                }
            else
                for( x = 0; x < size.width; x++ )
                {
                    int ix = saturate_cast<int>(src1f[x*2]*INTER_TAB_SIZE);
                    int iy = saturate_cast<int>(src1f[x*2+1]*INTER_TAB_SIZE);
                    dst1[x*2] = saturate_cast<short>(ix >> INTER_BITS);
                    dst1[x*2+1] = saturate_cast<short>(iy >> INTER_BITS);
                    dst2[x] = (ushort)((iy & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (ix & (INTER_TAB_SIZE-1)));
                }
        }
        else if( m1type == CV_16SC2 && dstm1type == CV_32FC1 )
        {
            for( x = 0; x < size.width; x++ )
            {
                int fxy = src2 ? src2[x] & (INTER_TAB_SIZE2-1) : 0;
                dst1f[x] = src1[x*2] + (fxy & (INTER_TAB_SIZE-1))*scale;
                dst2f[x] = src1[x*2+1] + (fxy >> INTER_BITS)*scale;
            }
        }
        else if( m1type == CV_16SC2 && dstm1type == CV_32FC2 )
        {
            for( x = 0; x < size.width; x++ )
            {
                int fxy = src2 ? src2[x] & (INTER_TAB_SIZE2-1): 0;
                dst1f[x*2] = src1[x*2] + (fxy & (INTER_TAB_SIZE-1))*scale;
                dst1f[x*2+1] = src1[x*2+1] + (fxy >> INTER_BITS)*scale;
            }
        }
        else
            CV_Error( CV_StsNotImplemented, "Unsupported combination of input/output matrices" );
    }
}


namespace cv
{

class WarpAffineInvoker :
    public ParallelLoopBody
{
public:
    WarpAffineInvoker(const Mat &_src, Mat &_dst, int _interpolation, int _borderType,
                      const Scalar &_borderValue, int *_adelta, int *_bdelta, double *_M) :
        ParallelLoopBody(), src(_src), dst(_dst), interpolation(_interpolation),
        borderType(_borderType), borderValue(_borderValue), adelta(_adelta), bdelta(_bdelta),
        M(_M)
    {
    }

    virtual void operator() (const Range& range) const
    {
        const int BLOCK_SZ = 64;
        short XY[BLOCK_SZ*BLOCK_SZ*2], A[BLOCK_SZ*BLOCK_SZ];
        const int AB_BITS = MAX(10, (int)INTER_BITS);
        const int AB_SCALE = 1 << AB_BITS;
        int round_delta = interpolation == INTER_NEAREST ? AB_SCALE/2 : AB_SCALE/INTER_TAB_SIZE/2, x, y, x1, y1;
    #if CV_SSE2
        bool useSIMD = checkHardwareSupport(CV_CPU_SSE2);
    #endif

        int bh0 = std::min(BLOCK_SZ/2, dst.rows);
        int bw0 = std::min(BLOCK_SZ*BLOCK_SZ/bh0, dst.cols);
        bh0 = std::min(BLOCK_SZ*BLOCK_SZ/bw0, dst.rows);

        for( y = range.start; y < range.end; y += bh0 )
        {
            for( x = 0; x < dst.cols; x += bw0 )
            {
                int bw = std::min( bw0, dst.cols - x);
                int bh = std::min( bh0, range.end - y);

                Mat _XY(bh, bw, CV_16SC2, XY), matA;
                Mat dpart(dst, Rect(x, y, bw, bh));

                for( y1 = 0; y1 < bh; y1++ )
                {
                    short* xy = XY + y1*bw*2;
                    int X0 = saturate_cast<int>((M[1]*(y + y1) + M[2])*AB_SCALE) + round_delta;
                    int Y0 = saturate_cast<int>((M[4]*(y + y1) + M[5])*AB_SCALE) + round_delta;

                    if( interpolation == INTER_NEAREST )
                        for( x1 = 0; x1 < bw; x1++ )
                        {
                            int X = (X0 + adelta[x+x1]) >> AB_BITS;
                            int Y = (Y0 + bdelta[x+x1]) >> AB_BITS;
                            xy[x1*2] = saturate_cast<short>(X);
                            xy[x1*2+1] = saturate_cast<short>(Y);
                        }
                    else
                    {
                        short* alpha = A + y1*bw;
                        x1 = 0;
                    #if CV_SSE2
                        if( useSIMD )
                        {
                            __m128i fxy_mask = _mm_set1_epi32(INTER_TAB_SIZE - 1);
                            __m128i XX = _mm_set1_epi32(X0), YY = _mm_set1_epi32(Y0);
                            for( ; x1 <= bw - 8; x1 += 8 )
                            {
                                __m128i tx0, tx1, ty0, ty1;
                                tx0 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(adelta + x + x1)), XX);
                                ty0 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(bdelta + x + x1)), YY);
                                tx1 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(adelta + x + x1 + 4)), XX);
                                ty1 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(bdelta + x + x1 + 4)), YY);

                                tx0 = _mm_srai_epi32(tx0, AB_BITS - INTER_BITS);
                                ty0 = _mm_srai_epi32(ty0, AB_BITS - INTER_BITS);
                                tx1 = _mm_srai_epi32(tx1, AB_BITS - INTER_BITS);
                                ty1 = _mm_srai_epi32(ty1, AB_BITS - INTER_BITS);

                                __m128i fx_ = _mm_packs_epi32(_mm_and_si128(tx0, fxy_mask),
                                                            _mm_and_si128(tx1, fxy_mask));
                                __m128i fy_ = _mm_packs_epi32(_mm_and_si128(ty0, fxy_mask),
                                                            _mm_and_si128(ty1, fxy_mask));
                                tx0 = _mm_packs_epi32(_mm_srai_epi32(tx0, INTER_BITS),
                                                            _mm_srai_epi32(tx1, INTER_BITS));
                                ty0 = _mm_packs_epi32(_mm_srai_epi32(ty0, INTER_BITS),
                                                    _mm_srai_epi32(ty1, INTER_BITS));
                                fx_ = _mm_adds_epi16(fx_, _mm_slli_epi16(fy_, INTER_BITS));

                                _mm_storeu_si128((__m128i*)(xy + x1*2), _mm_unpacklo_epi16(tx0, ty0));
                                _mm_storeu_si128((__m128i*)(xy + x1*2 + 8), _mm_unpackhi_epi16(tx0, ty0));
                                _mm_storeu_si128((__m128i*)(alpha + x1), fx_);
                            }
                        }
                    #endif
                        for( ; x1 < bw; x1++ )
                        {
                            int X = (X0 + adelta[x+x1]) >> (AB_BITS - INTER_BITS);
                            int Y = (Y0 + bdelta[x+x1]) >> (AB_BITS - INTER_BITS);
                            xy[x1*2] = saturate_cast<short>(X >> INTER_BITS);
                            xy[x1*2+1] = saturate_cast<short>(Y >> INTER_BITS);
                            alpha[x1] = (short)((Y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE +
                                    (X & (INTER_TAB_SIZE-1)));
                        }
                    }
                }

                if( interpolation == INTER_NEAREST )
                    remap( src, dpart, _XY, Mat(), interpolation, borderType, borderValue );
                else
                {
                    Mat _matA(bh, bw, CV_16U, A);
                    remap( src, dpart, _XY, _matA, interpolation, borderType, borderValue );
                }
            }
        }
    }

private:
    Mat src;
    Mat dst;
    int interpolation, borderType;
    Scalar borderValue;
    int *adelta, *bdelta;
    double *M;
};


#if defined (HAVE_IPP) && IPP_VERSION_MAJOR * 100 + IPP_VERSION_MINOR >= 801 && 0
class IPPWarpAffineInvoker :
    public ParallelLoopBody
{
public:
    IPPWarpAffineInvoker(Mat &_src, Mat &_dst, double (&_coeffs)[2][3], int &_interpolation, int _borderType,
                         const Scalar &_borderValue, ippiWarpAffineBackFunc _func, bool *_ok) :
        ParallelLoopBody(), src(_src), dst(_dst), mode(_interpolation), coeffs(_coeffs),
        borderType(_borderType), borderValue(_borderValue), func(_func), ok(_ok)
    {
        *ok = true;
    }

    virtual void operator() (const Range& range) const
    {
        IppiSize srcsize = { src.cols, src.rows };
        IppiRect srcroi = { 0, 0, src.cols, src.rows };
        IppiRect dstroi = { 0, range.start, dst.cols, range.end - range.start };
        int cnn = src.channels();
        if( borderType == BORDER_CONSTANT )
        {
            IppiSize setSize = { dst.cols, range.end - range.start };
            void *dataPointer = dst.data + dst.step[0] * range.start;
            if( !IPPSet( borderValue, dataPointer, (int)dst.step[0], setSize, cnn, src.depth() ) )
            {
                *ok = false;
                return;
            }
        }

        // Aug 2013: problem in IPP 7.1, 8.0 : sometimes function return ippStsCoeffErr
        IppStatus status = func( src.data, srcsize, (int)src.step[0], srcroi, dst.data,
                                (int)dst.step[0], dstroi, coeffs, mode );
        if( status < 0)
            *ok = false;
    }
private:
    Mat &src;
    Mat &dst;
    int mode;
    double (&coeffs)[2][3];
    int borderType;
    Scalar borderValue;
    ippiWarpAffineBackFunc func;
    bool *ok;
    const IPPWarpAffineInvoker& operator= (const IPPWarpAffineInvoker&);
};
#endif

#ifdef HAVE_OPENCL

enum { OCL_OP_PERSPECTIVE = 1, OCL_OP_AFFINE = 0 };

static bool ocl_warpTransform(InputArray _src, OutputArray _dst, InputArray _M0,
                              Size dsize, int flags, int borderType, const Scalar& borderValue,
                              int op_type)
{
    CV_Assert(op_type == OCL_OP_AFFINE || op_type == OCL_OP_PERSPECTIVE);

    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    double doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;

    int interpolation = flags & INTER_MAX;
    if( interpolation == INTER_AREA )
        interpolation = INTER_LINEAR;

    if ( !(borderType == cv::BORDER_CONSTANT &&
           (interpolation == cv::INTER_NEAREST || interpolation == cv::INTER_LINEAR || interpolation == cv::INTER_CUBIC)) ||
         (!doubleSupport && depth == CV_64F) || cn > 4)
        return false;

    const char * const interpolationMap[3] = { "NEAREST", "LINEAR", "CUBIC" };
    ocl::ProgramSource program = op_type == OCL_OP_AFFINE ?
                ocl::imgproc::warp_affine_oclsrc : ocl::imgproc::warp_perspective_oclsrc;
    const char * const kernelName = op_type == OCL_OP_AFFINE ? "warpAffine" : "warpPerspective";

    int scalarcn = cn == 3 ? 4 : cn;
    int wdepth = interpolation == INTER_NEAREST ? depth : std::max(CV_32S, depth);
    int sctype = CV_MAKETYPE(wdepth, scalarcn);

    ocl::Kernel k;
    String opts;
    if (interpolation == INTER_NEAREST)
    {
        opts = format("-D INTER_NEAREST -D T=%s%s -D T1=%s -D ST=%s -D cn=%d", ocl::typeToStr(type),
                      doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                      ocl::typeToStr(CV_MAT_DEPTH(type)),
                      ocl::typeToStr(sctype),
                      cn);
    }
    else
    {
        char cvt[2][50];
        opts = format("-D INTER_%s -D T=%s -D T1=%s -D ST=%s -D WT=%s -D depth=%d -D convertToWT=%s -D convertToT=%s%s -D cn=%d",
                      interpolationMap[interpolation], ocl::typeToStr(type),
                      ocl::typeToStr(CV_MAT_DEPTH(type)),
                      ocl::typeToStr(sctype),
                      ocl::typeToStr(CV_MAKE_TYPE(wdepth, cn)), depth,
                      ocl::convertTypeStr(depth, wdepth, cn, cvt[0]),
                      ocl::convertTypeStr(wdepth, depth, cn, cvt[1]),
                      doubleSupport ? " -D DOUBLE_SUPPORT" : "", cn);
    }

    k.create(kernelName, program, opts);
    if (k.empty())
        return false;

    double borderBuf[] = {0, 0, 0, 0};
    scalarToRawData(borderValue, borderBuf, sctype);

    UMat src = _src.getUMat(), M0;
    _dst.create( dsize.area() == 0 ? src.size() : dsize, src.type() );
    UMat dst = _dst.getUMat();

    double M[9];
    int matRows = (op_type == OCL_OP_AFFINE ? 2 : 3);
    Mat matM(matRows, 3, CV_64F, M), M1 = _M0.getMat();
    CV_Assert( (M1.type() == CV_32F || M1.type() == CV_64F) &&
               M1.rows == matRows && M1.cols == 3 );
    M1.convertTo(matM, matM.type());

    if( !(flags & WARP_INVERSE_MAP) )
    {
        if (op_type == OCL_OP_PERSPECTIVE)
            invert(matM, matM);
        else
        {
            double D = M[0]*M[4] - M[1]*M[3];
            D = D != 0 ? 1./D : 0;
            double A11 = M[4]*D, A22=M[0]*D;
            M[0] = A11; M[1] *= -D;
            M[3] *= -D; M[4] = A22;
            double b1 = -M[0]*M[2] - M[1]*M[5];
            double b2 = -M[3]*M[2] - M[4]*M[5];
            M[2] = b1; M[5] = b2;
        }
    }
    matM.convertTo(M0, doubleSupport ? CV_64F : CV_32F);

    k.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnly(dst), ocl::KernelArg::PtrReadOnly(M0),
           ocl::KernelArg(0, 0, 0, 0, borderBuf, CV_ELEM_SIZE(sctype)));

    size_t globalThreads[2] = { dst.cols, dst.rows };
    return k.run(2, globalThreads, NULL, false);
}

#endif

}


void cv::warpAffine( InputArray _src, OutputArray _dst,
                     InputArray _M0, Size dsize,
                     int flags, int borderType, const Scalar& borderValue )
{
    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_warpTransform(_src, _dst, _M0, dsize, flags, borderType,
                                 borderValue, OCL_OP_AFFINE))

    Mat src = _src.getMat(), M0 = _M0.getMat();
    _dst.create( dsize.area() == 0 ? src.size() : dsize, src.type() );
    Mat dst = _dst.getMat();
    CV_Assert( src.cols > 0 && src.rows > 0 );
    if( dst.data == src.data )
        src = src.clone();

    double M[6];
    Mat matM(2, 3, CV_64F, M);
    int interpolation = flags & INTER_MAX;
    if( interpolation == INTER_AREA )
        interpolation = INTER_LINEAR;

    CV_Assert( (M0.type() == CV_32F || M0.type() == CV_64F) && M0.rows == 2 && M0.cols == 3 );
    M0.convertTo(matM, matM.type());

#ifdef HAVE_TEGRA_OPTIMIZATION
    if( tegra::warpAffine(src, dst, M, flags, borderType, borderValue) )
        return;
#endif

    if( !(flags & WARP_INVERSE_MAP) )
    {
        double D = M[0]*M[4] - M[1]*M[3];
        D = D != 0 ? 1./D : 0;
        double A11 = M[4]*D, A22=M[0]*D;
        M[0] = A11; M[1] *= -D;
        M[3] *= -D; M[4] = A22;
        double b1 = -M[0]*M[2] - M[1]*M[5];
        double b2 = -M[3]*M[2] - M[4]*M[5];
        M[2] = b1; M[5] = b2;
    }

    int x;
    AutoBuffer<int> _abdelta(dst.cols*2);
    int* adelta = &_abdelta[0], *bdelta = adelta + dst.cols;
    const int AB_BITS = MAX(10, (int)INTER_BITS);
    const int AB_SCALE = 1 << AB_BITS;

#if defined (HAVE_IPP) && IPP_VERSION_MAJOR * 100 + IPP_VERSION_MINOR >= 801 && 0
    int type = src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    if( ( depth == CV_8U || depth == CV_16U || depth == CV_32F ) &&
       ( cn == 1 || cn == 3 || cn == 4 ) &&
       ( interpolation == INTER_NEAREST || interpolation == INTER_LINEAR || interpolation == INTER_CUBIC) &&
       ( borderType == cv::BORDER_TRANSPARENT || borderType == cv::BORDER_CONSTANT) )
    {
        ippiWarpAffineBackFunc ippFunc = 0;
        if ((flags & WARP_INVERSE_MAP) != 0)
        {
            ippFunc =
            type == CV_8UC1 ? (ippiWarpAffineBackFunc)ippiWarpAffineBack_8u_C1R :
            type == CV_8UC3 ? (ippiWarpAffineBackFunc)ippiWarpAffineBack_8u_C3R :
            type == CV_8UC4 ? (ippiWarpAffineBackFunc)ippiWarpAffineBack_8u_C4R :
            type == CV_16UC1 ? (ippiWarpAffineBackFunc)ippiWarpAffineBack_16u_C1R :
            type == CV_16UC3 ? (ippiWarpAffineBackFunc)ippiWarpAffineBack_16u_C3R :
            type == CV_16UC4 ? (ippiWarpAffineBackFunc)ippiWarpAffineBack_16u_C4R :
            type == CV_32FC1 ? (ippiWarpAffineBackFunc)ippiWarpAffineBack_32f_C1R :
            type == CV_32FC3 ? (ippiWarpAffineBackFunc)ippiWarpAffineBack_32f_C3R :
            type == CV_32FC4 ? (ippiWarpAffineBackFunc)ippiWarpAffineBack_32f_C4R :
            0;
        }
        else
        {
            ippFunc =
            type == CV_8UC1 ? (ippiWarpAffineBackFunc)ippiWarpAffine_8u_C1R :
            type == CV_8UC3 ? (ippiWarpAffineBackFunc)ippiWarpAffine_8u_C3R :
            type == CV_8UC4 ? (ippiWarpAffineBackFunc)ippiWarpAffine_8u_C4R :
            type == CV_16UC1 ? (ippiWarpAffineBackFunc)ippiWarpAffine_16u_C1R :
            type == CV_16UC3 ? (ippiWarpAffineBackFunc)ippiWarpAffine_16u_C3R :
            type == CV_16UC4 ? (ippiWarpAffineBackFunc)ippiWarpAffine_16u_C4R :
            type == CV_32FC1 ? (ippiWarpAffineBackFunc)ippiWarpAffine_32f_C1R :
            type == CV_32FC3 ? (ippiWarpAffineBackFunc)ippiWarpAffine_32f_C3R :
            type == CV_32FC4 ? (ippiWarpAffineBackFunc)ippiWarpAffine_32f_C4R :
            0;
        }
        int mode =
        interpolation == INTER_LINEAR ? IPPI_INTER_LINEAR :
        interpolation == INTER_NEAREST ? IPPI_INTER_NN :
        interpolation == INTER_CUBIC ? IPPI_INTER_CUBIC :
        0;
        CV_Assert(mode && ippFunc);

        double coeffs[2][3];
        for( int i = 0; i < 2; i++ )
            for( int j = 0; j < 3; j++ )
                coeffs[i][j] = matM.at<double>(i, j);

        bool ok;
        Range range(0, dst.rows);
        IPPWarpAffineInvoker invoker(src, dst, coeffs, mode, borderType, borderValue, ippFunc, &ok);
        parallel_for_(range, invoker, dst.total()/(double)(1<<16));
        if( ok )
            return;
        setIppErrorStatus();
    }
#endif

    for( x = 0; x < dst.cols; x++ )
    {
        adelta[x] = saturate_cast<int>(M[0]*x*AB_SCALE);
        bdelta[x] = saturate_cast<int>(M[3]*x*AB_SCALE);
    }

    Range range(0, dst.rows);
    WarpAffineInvoker invoker(src, dst, interpolation, borderType,
                              borderValue, adelta, bdelta, M);
    parallel_for_(range, invoker, dst.total()/(double)(1<<16));
}


namespace cv
{

class WarpPerspectiveInvoker :
    public ParallelLoopBody
{
public:
    WarpPerspectiveInvoker(const Mat &_src, Mat &_dst, double *_M, int _interpolation,
                           int _borderType, const Scalar &_borderValue) :
        ParallelLoopBody(), src(_src), dst(_dst), M(_M), interpolation(_interpolation),
        borderType(_borderType), borderValue(_borderValue)
    {
    }

    virtual void operator() (const Range& range) const
    {
        const int BLOCK_SZ = 32;
        short XY[BLOCK_SZ*BLOCK_SZ*2], A[BLOCK_SZ*BLOCK_SZ];
        int x, y, x1, y1, width = dst.cols, height = dst.rows;

        int bh0 = std::min(BLOCK_SZ/2, height);
        int bw0 = std::min(BLOCK_SZ*BLOCK_SZ/bh0, width);
        bh0 = std::min(BLOCK_SZ*BLOCK_SZ/bw0, height);

        for( y = range.start; y < range.end; y += bh0 )
        {
            for( x = 0; x < width; x += bw0 )
            {
                int bw = std::min( bw0, width - x);
                int bh = std::min( bh0, range.end - y); // height

                Mat _XY(bh, bw, CV_16SC2, XY), matA;
                Mat dpart(dst, Rect(x, y, bw, bh));

                for( y1 = 0; y1 < bh; y1++ )
                {
                    short* xy = XY + y1*bw*2;
                    double X0 = M[0]*x + M[1]*(y + y1) + M[2];
                    double Y0 = M[3]*x + M[4]*(y + y1) + M[5];
                    double W0 = M[6]*x + M[7]*(y + y1) + M[8];

                    if( interpolation == INTER_NEAREST )
                        for( x1 = 0; x1 < bw; x1++ )
                        {
                            double W = W0 + M[6]*x1;
                            W = W ? 1./W : 0;
                            double fX = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0]*x1)*W));
                            double fY = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3]*x1)*W));
                            int X = saturate_cast<int>(fX);
                            int Y = saturate_cast<int>(fY);

                            xy[x1*2] = saturate_cast<short>(X);
                            xy[x1*2+1] = saturate_cast<short>(Y);
                        }
                    else
                    {
                        short* alpha = A + y1*bw;
                        for( x1 = 0; x1 < bw; x1++ )
                        {
                            double W = W0 + M[6]*x1;
                            W = W ? INTER_TAB_SIZE/W : 0;
                            double fX = std::max((double)INT_MIN, std::min((double)INT_MAX, (X0 + M[0]*x1)*W));
                            double fY = std::max((double)INT_MIN, std::min((double)INT_MAX, (Y0 + M[3]*x1)*W));
                            int X = saturate_cast<int>(fX);
                            int Y = saturate_cast<int>(fY);

                            xy[x1*2] = saturate_cast<short>(X >> INTER_BITS);
                            xy[x1*2+1] = saturate_cast<short>(Y >> INTER_BITS);
                            alpha[x1] = (short)((Y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE +
                                                (X & (INTER_TAB_SIZE-1)));
                        }
                    }
                }

                if( interpolation == INTER_NEAREST )
                    remap( src, dpart, _XY, Mat(), interpolation, borderType, borderValue );
                else
                {
                    Mat _matA(bh, bw, CV_16U, A);
                    remap( src, dpart, _XY, _matA, interpolation, borderType, borderValue );
                }
            }
        }
    }

private:
    Mat src;
    Mat dst;
    double* M;
    int interpolation, borderType;
    Scalar borderValue;
};


#if defined (HAVE_IPP) && IPP_VERSION_MAJOR * 100 + IPP_VERSION_MINOR >= 801 && 0
class IPPWarpPerspectiveInvoker :
    public ParallelLoopBody
{
public:
    IPPWarpPerspectiveInvoker(Mat &_src, Mat &_dst, double (&_coeffs)[3][3], int &_interpolation,
                              int &_borderType, const Scalar &_borderValue, ippiWarpPerspectiveFunc _func, bool *_ok) :
        ParallelLoopBody(), src(_src), dst(_dst), mode(_interpolation), coeffs(_coeffs),
        borderType(_borderType), borderValue(_borderValue), func(_func), ok(_ok)
    {
        *ok = true;
    }

    virtual void operator() (const Range& range) const
    {
        IppiSize srcsize = {src.cols, src.rows};
        IppiRect srcroi = {0, 0, src.cols, src.rows};
        IppiRect dstroi = {0, range.start, dst.cols, range.end - range.start};
        int cnn = src.channels();

        if( borderType == BORDER_CONSTANT )
        {
            IppiSize setSize = {dst.cols, range.end - range.start};
            void *dataPointer = dst.data + dst.step[0] * range.start;
            if( !IPPSet( borderValue, dataPointer, (int)dst.step[0], setSize, cnn, src.depth() ) )
            {
                *ok = false;
                return;
            }
        }

        IppStatus status = func(src.data, srcsize, (int)src.step[0], srcroi, dst.data, (int)dst.step[0], dstroi, coeffs, mode);
        if (status != ippStsNoErr)
            *ok = false;
    }
private:
    Mat &src;
    Mat &dst;
    int mode;
    double (&coeffs)[3][3];
    int borderType;
    const Scalar borderValue;
    ippiWarpPerspectiveFunc func;
    bool *ok;

    const IPPWarpPerspectiveInvoker& operator= (const IPPWarpPerspectiveInvoker&);
};
#endif
}

void cv::warpPerspective( InputArray _src, OutputArray _dst, InputArray _M0,
                          Size dsize, int flags, int borderType, const Scalar& borderValue )
{
    CV_Assert( _src.total() > 0 );

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_warpTransform(_src, _dst, _M0, dsize, flags, borderType, borderValue,
                              OCL_OP_PERSPECTIVE))

    Mat src = _src.getMat(), M0 = _M0.getMat();
    _dst.create( dsize.area() == 0 ? src.size() : dsize, src.type() );
    Mat dst = _dst.getMat();

    if( dst.data == src.data )
        src = src.clone();

    double M[9];
    Mat matM(3, 3, CV_64F, M);
    int interpolation = flags & INTER_MAX;
    if( interpolation == INTER_AREA )
        interpolation = INTER_LINEAR;

    CV_Assert( (M0.type() == CV_32F || M0.type() == CV_64F) && M0.rows == 3 && M0.cols == 3 );
    M0.convertTo(matM, matM.type());

#ifdef HAVE_TEGRA_OPTIMIZATION
    if( tegra::warpPerspective(src, dst, M, flags, borderType, borderValue) )
        return;
#endif


#if defined (HAVE_IPP) && IPP_VERSION_MAJOR * 100 + IPP_VERSION_MINOR >= 801 && 0
    int type = src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    if( (depth == CV_8U || depth == CV_16U || depth == CV_32F) &&
       (cn == 1 || cn == 3 || cn == 4) &&
       ( borderType == cv::BORDER_TRANSPARENT || borderType == cv::BORDER_CONSTANT ) &&
       (interpolation == INTER_NEAREST || interpolation == INTER_LINEAR || interpolation == INTER_CUBIC))
    {
        ippiWarpPerspectiveFunc ippFunc = 0;
        if ((flags & WARP_INVERSE_MAP) != 0)
        {
            ippFunc = type == CV_8UC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveBack_8u_C1R :
            type == CV_8UC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveBack_8u_C3R :
            type == CV_8UC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveBack_8u_C4R :
            type == CV_16UC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveBack_16u_C1R :
            type == CV_16UC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveBack_16u_C3R :
            type == CV_16UC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveBack_16u_C4R :
            type == CV_32FC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveBack_32f_C1R :
            type == CV_32FC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveBack_32f_C3R :
            type == CV_32FC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspectiveBack_32f_C4R : 0;
        }
        else
        {
            ippFunc = type == CV_8UC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspective_8u_C1R :
            type == CV_8UC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspective_8u_C3R :
            type == CV_8UC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspective_8u_C4R :
            type == CV_16UC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspective_16u_C1R :
            type == CV_16UC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspective_16u_C3R :
            type == CV_16UC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspective_16u_C4R :
            type == CV_32FC1 ? (ippiWarpPerspectiveFunc)ippiWarpPerspective_32f_C1R :
            type == CV_32FC3 ? (ippiWarpPerspectiveFunc)ippiWarpPerspective_32f_C3R :
            type == CV_32FC4 ? (ippiWarpPerspectiveFunc)ippiWarpPerspective_32f_C4R : 0;
        }
        int mode =
        interpolation == INTER_NEAREST ? IPPI_INTER_NN :
        interpolation == INTER_LINEAR ? IPPI_INTER_LINEAR :
        interpolation == INTER_CUBIC ? IPPI_INTER_CUBIC : 0;
        CV_Assert(mode && ippFunc);

        double coeffs[3][3];
        for( int i = 0; i < 3; i++ )
            for( int j = 0; j < 3; j++ )
                coeffs[i][j] = matM.at<double>(i, j);

        bool ok;
        Range range(0, dst.rows);
        IPPWarpPerspectiveInvoker invoker(src, dst, coeffs, mode, borderType, borderValue, ippFunc, &ok);
        parallel_for_(range, invoker, dst.total()/(double)(1<<16));
        if( ok )
            return;
        setIppErrorStatus();
    }
#endif

    if( !(flags & WARP_INVERSE_MAP) )
        invert(matM, matM);

    Range range(0, dst.rows);
    WarpPerspectiveInvoker invoker(src, dst, M, interpolation, borderType, borderValue);
    parallel_for_(range, invoker, dst.total()/(double)(1<<16));
}


cv::Mat cv::getRotationMatrix2D( Point2f center, double angle, double scale )
{
    angle *= CV_PI/180;
    double alpha = cos(angle)*scale;
    double beta = sin(angle)*scale;

    Mat M(2, 3, CV_64F);
    double* m = (double*)M.data;

    m[0] = alpha;
    m[1] = beta;
    m[2] = (1-alpha)*center.x - beta*center.y;
    m[3] = -beta;
    m[4] = alpha;
    m[5] = beta*center.x + (1-alpha)*center.y;

    return M;
}

/* Calculates coefficients of perspective transformation
 * which maps (xi,yi) to (ui,vi), (i=1,2,3,4):
 *
 *      c00*xi + c01*yi + c02
 * ui = ---------------------
 *      c20*xi + c21*yi + c22
 *
 *      c10*xi + c11*yi + c12
 * vi = ---------------------
 *      c20*xi + c21*yi + c22
 *
 * Coefficients are calculated by solving linear system:
 * / x0 y0  1  0  0  0 -x0*u0 -y0*u0 \ /c00\ /u0\
 * | x1 y1  1  0  0  0 -x1*u1 -y1*u1 | |c01| |u1|
 * | x2 y2  1  0  0  0 -x2*u2 -y2*u2 | |c02| |u2|
 * | x3 y3  1  0  0  0 -x3*u3 -y3*u3 |.|c10|=|u3|,
 * |  0  0  0 x0 y0  1 -x0*v0 -y0*v0 | |c11| |v0|
 * |  0  0  0 x1 y1  1 -x1*v1 -y1*v1 | |c12| |v1|
 * |  0  0  0 x2 y2  1 -x2*v2 -y2*v2 | |c20| |v2|
 * \  0  0  0 x3 y3  1 -x3*v3 -y3*v3 / \c21/ \v3/
 *
 * where:
 *   cij - matrix coefficients, c22 = 1
 */
cv::Mat cv::getPerspectiveTransform( const Point2f src[], const Point2f dst[] )
{
    Mat M(3, 3, CV_64F), X(8, 1, CV_64F, M.data);
    double a[8][8], b[8];
    Mat A(8, 8, CV_64F, a), B(8, 1, CV_64F, b);

    for( int i = 0; i < 4; ++i )
    {
        a[i][0] = a[i+4][3] = src[i].x;
        a[i][1] = a[i+4][4] = src[i].y;
        a[i][2] = a[i+4][5] = 1;
        a[i][3] = a[i][4] = a[i][5] =
        a[i+4][0] = a[i+4][1] = a[i+4][2] = 0;
        a[i][6] = -src[i].x*dst[i].x;
        a[i][7] = -src[i].y*dst[i].x;
        a[i+4][6] = -src[i].x*dst[i].y;
        a[i+4][7] = -src[i].y*dst[i].y;
        b[i] = dst[i].x;
        b[i+4] = dst[i].y;
    }

    solve( A, B, X, DECOMP_SVD );
    ((double*)M.data)[8] = 1.;

    return M;
}

/* Calculates coefficients of affine transformation
 * which maps (xi,yi) to (ui,vi), (i=1,2,3):
 *
 * ui = c00*xi + c01*yi + c02
 *
 * vi = c10*xi + c11*yi + c12
 *
 * Coefficients are calculated by solving linear system:
 * / x0 y0  1  0  0  0 \ /c00\ /u0\
 * | x1 y1  1  0  0  0 | |c01| |u1|
 * | x2 y2  1  0  0  0 | |c02| |u2|
 * |  0  0  0 x0 y0  1 | |c10| |v0|
 * |  0  0  0 x1 y1  1 | |c11| |v1|
 * \  0  0  0 x2 y2  1 / |c12| |v2|
 *
 * where:
 *   cij - matrix coefficients
 */

cv::Mat cv::getAffineTransform( const Point2f src[], const Point2f dst[] )
{
    Mat M(2, 3, CV_64F), X(6, 1, CV_64F, M.data);
    double a[6*6], b[6];
    Mat A(6, 6, CV_64F, a), B(6, 1, CV_64F, b);

    for( int i = 0; i < 3; i++ )
    {
        int j = i*12;
        int k = i*12+6;
        a[j] = a[k+3] = src[i].x;
        a[j+1] = a[k+4] = src[i].y;
        a[j+2] = a[k+5] = 1;
        a[j+3] = a[j+4] = a[j+5] = 0;
        a[k] = a[k+1] = a[k+2] = 0;
        b[i*2] = dst[i].x;
        b[i*2+1] = dst[i].y;
    }

    solve( A, B, X );
    return M;
}

void cv::invertAffineTransform(InputArray _matM, OutputArray __iM)
{
    Mat matM = _matM.getMat();
    CV_Assert(matM.rows == 2 && matM.cols == 3);
    __iM.create(2, 3, matM.type());
    Mat _iM = __iM.getMat();

    if( matM.type() == CV_32F )
    {
        const float* M = (const float*)matM.data;
        float* iM = (float*)_iM.data;
        int step = (int)(matM.step/sizeof(M[0])), istep = (int)(_iM.step/sizeof(iM[0]));

        double D = M[0]*M[step+1] - M[1]*M[step];
        D = D != 0 ? 1./D : 0;
        double A11 = M[step+1]*D, A22 = M[0]*D, A12 = -M[1]*D, A21 = -M[step]*D;
        double b1 = -A11*M[2] - A12*M[step+2];
        double b2 = -A21*M[2] - A22*M[step+2];

        iM[0] = (float)A11; iM[1] = (float)A12; iM[2] = (float)b1;
        iM[istep] = (float)A21; iM[istep+1] = (float)A22; iM[istep+2] = (float)b2;
    }
    else if( matM.type() == CV_64F )
    {
        const double* M = (const double*)matM.data;
        double* iM = (double*)_iM.data;
        int step = (int)(matM.step/sizeof(M[0])), istep = (int)(_iM.step/sizeof(iM[0]));

        double D = M[0]*M[step+1] - M[1]*M[step];
        D = D != 0 ? 1./D : 0;
        double A11 = M[step+1]*D, A22 = M[0]*D, A12 = -M[1]*D, A21 = -M[step]*D;
        double b1 = -A11*M[2] - A12*M[step+2];
        double b2 = -A21*M[2] - A22*M[step+2];

        iM[0] = A11; iM[1] = A12; iM[2] = b1;
        iM[istep] = A21; iM[istep+1] = A22; iM[istep+2] = b2;
    }
    else
        CV_Error( CV_StsUnsupportedFormat, "" );
}

cv::Mat cv::getPerspectiveTransform(InputArray _src, InputArray _dst)
{
    Mat src = _src.getMat(), dst = _dst.getMat();
    CV_Assert(src.checkVector(2, CV_32F) == 4 && dst.checkVector(2, CV_32F) == 4);
    return getPerspectiveTransform((const Point2f*)src.data, (const Point2f*)dst.data);
}

cv::Mat cv::getAffineTransform(InputArray _src, InputArray _dst)
{
    Mat src = _src.getMat(), dst = _dst.getMat();
    CV_Assert(src.checkVector(2, CV_32F) == 3 && dst.checkVector(2, CV_32F) == 3);
    return getAffineTransform((const Point2f*)src.data, (const Point2f*)dst.data);
}

CV_IMPL void
cvResize( const CvArr* srcarr, CvArr* dstarr, int method )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src.type() == dst.type() );
    cv::resize( src, dst, dst.size(), (double)dst.cols/src.cols,
        (double)dst.rows/src.rows, method );
}


CV_IMPL void
cvWarpAffine( const CvArr* srcarr, CvArr* dstarr, const CvMat* marr,
              int flags, CvScalar fillval )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    cv::Mat matrix = cv::cvarrToMat(marr);
    CV_Assert( src.type() == dst.type() );
    cv::warpAffine( src, dst, matrix, dst.size(), flags,
        (flags & CV_WARP_FILL_OUTLIERS) ? cv::BORDER_CONSTANT : cv::BORDER_TRANSPARENT,
        fillval );
}

CV_IMPL void
cvWarpPerspective( const CvArr* srcarr, CvArr* dstarr, const CvMat* marr,
                   int flags, CvScalar fillval )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    cv::Mat matrix = cv::cvarrToMat(marr);
    CV_Assert( src.type() == dst.type() );
    cv::warpPerspective( src, dst, matrix, dst.size(), flags,
        (flags & CV_WARP_FILL_OUTLIERS) ? cv::BORDER_CONSTANT : cv::BORDER_TRANSPARENT,
        fillval );
}

CV_IMPL void
cvRemap( const CvArr* srcarr, CvArr* dstarr,
         const CvArr* _mapx, const CvArr* _mapy,
         int flags, CvScalar fillval )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), dst0 = dst;
    cv::Mat mapx = cv::cvarrToMat(_mapx), mapy = cv::cvarrToMat(_mapy);
    CV_Assert( src.type() == dst.type() && dst.size() == mapx.size() );
    cv::remap( src, dst, mapx, mapy, flags & cv::INTER_MAX,
        (flags & CV_WARP_FILL_OUTLIERS) ? cv::BORDER_CONSTANT : cv::BORDER_TRANSPARENT,
        fillval );
    CV_Assert( dst0.data == dst.data );
}


CV_IMPL CvMat*
cv2DRotationMatrix( CvPoint2D32f center, double angle,
                    double scale, CvMat* matrix )
{
    cv::Mat M0 = cv::cvarrToMat(matrix), M = cv::getRotationMatrix2D(center, angle, scale);
    CV_Assert( M.size() == M0.size() );
    M.convertTo(M0, M0.type());
    return matrix;
}


CV_IMPL CvMat*
cvGetPerspectiveTransform( const CvPoint2D32f* src,
                          const CvPoint2D32f* dst,
                          CvMat* matrix )
{
    cv::Mat M0 = cv::cvarrToMat(matrix),
        M = cv::getPerspectiveTransform((const cv::Point2f*)src, (const cv::Point2f*)dst);
    CV_Assert( M.size() == M0.size() );
    M.convertTo(M0, M0.type());
    return matrix;
}


CV_IMPL CvMat*
cvGetAffineTransform( const CvPoint2D32f* src,
                          const CvPoint2D32f* dst,
                          CvMat* matrix )
{
    cv::Mat M0 = cv::cvarrToMat(matrix),
        M = cv::getAffineTransform((const cv::Point2f*)src, (const cv::Point2f*)dst);
    CV_Assert( M.size() == M0.size() );
    M.convertTo(M0, M0.type());
    return matrix;
}


CV_IMPL void
cvConvertMaps( const CvArr* arr1, const CvArr* arr2, CvArr* dstarr1, CvArr* dstarr2 )
{
    cv::Mat map1 = cv::cvarrToMat(arr1), map2;
    cv::Mat dstmap1 = cv::cvarrToMat(dstarr1), dstmap2;

    if( arr2 )
        map2 = cv::cvarrToMat(arr2);
    if( dstarr2 )
    {
        dstmap2 = cv::cvarrToMat(dstarr2);
        if( dstmap2.type() == CV_16SC1 )
            dstmap2 = cv::Mat(dstmap2.size(), CV_16UC1, dstmap2.data, dstmap2.step);
    }

    cv::convertMaps( map1, map2, dstmap1, dstmap2, dstmap1.type(), false );
}

/****************************************************************************************\
*                                   Log-Polar Transform                                  *
\****************************************************************************************/

/* now it is done via Remap; more correct implementation should use
   some super-sampling technique outside of the "fovea" circle */
CV_IMPL void
cvLogPolar( const CvArr* srcarr, CvArr* dstarr,
            CvPoint2D32f center, double M, int flags )
{
    cv::Ptr<CvMat> mapx, mapy;

    CvMat srcstub, *src = cvGetMat(srcarr, &srcstub);
    CvMat dststub, *dst = cvGetMat(dstarr, &dststub);
    CvSize ssize, dsize;

    if( !CV_ARE_TYPES_EQ( src, dst ))
        CV_Error( CV_StsUnmatchedFormats, "" );

    if( M <= 0 )
        CV_Error( CV_StsOutOfRange, "M should be >0" );

    ssize = cvGetMatSize(src);
    dsize = cvGetMatSize(dst);

    mapx.reset(cvCreateMat( dsize.height, dsize.width, CV_32F ));
    mapy.reset(cvCreateMat( dsize.height, dsize.width, CV_32F ));

    if( !(flags & CV_WARP_INVERSE_MAP) )
    {
        int phi, rho;
        cv::AutoBuffer<double> _exp_tab(dsize.width);
        double* exp_tab = _exp_tab;

        for( rho = 0; rho < dst->width; rho++ )
            exp_tab[rho] = std::exp(rho/M);

        for( phi = 0; phi < dsize.height; phi++ )
        {
            double cp = cos(phi*2*CV_PI/dsize.height);
            double sp = sin(phi*2*CV_PI/dsize.height);
            float* mx = (float*)(mapx->data.ptr + phi*mapx->step);
            float* my = (float*)(mapy->data.ptr + phi*mapy->step);

            for( rho = 0; rho < dsize.width; rho++ )
            {
                double r = exp_tab[rho];
                double x = r*cp + center.x;
                double y = r*sp + center.y;

                mx[rho] = (float)x;
                my[rho] = (float)y;
            }
        }
    }
    else
    {
        int x, y;
        CvMat bufx, bufy, bufp, bufa;
        double ascale = ssize.height/(2*CV_PI);
        cv::AutoBuffer<float> _buf(4*dsize.width);
        float* buf = _buf;

        bufx = cvMat( 1, dsize.width, CV_32F, buf );
        bufy = cvMat( 1, dsize.width, CV_32F, buf + dsize.width );
        bufp = cvMat( 1, dsize.width, CV_32F, buf + dsize.width*2 );
        bufa = cvMat( 1, dsize.width, CV_32F, buf + dsize.width*3 );

        for( x = 0; x < dsize.width; x++ )
            bufx.data.fl[x] = (float)x - center.x;

        for( y = 0; y < dsize.height; y++ )
        {
            float* mx = (float*)(mapx->data.ptr + y*mapx->step);
            float* my = (float*)(mapy->data.ptr + y*mapy->step);

            for( x = 0; x < dsize.width; x++ )
                bufy.data.fl[x] = (float)y - center.y;

#if 1
            cvCartToPolar( &bufx, &bufy, &bufp, &bufa );

            for( x = 0; x < dsize.width; x++ )
                bufp.data.fl[x] += 1.f;

            cvLog( &bufp, &bufp );

            for( x = 0; x < dsize.width; x++ )
            {
                double rho = bufp.data.fl[x]*M;
                double phi = bufa.data.fl[x]*ascale;

                mx[x] = (float)rho;
                my[x] = (float)phi;
            }
#else
            for( x = 0; x < dsize.width; x++ )
            {
                double xx = bufx.data.fl[x];
                double yy = bufy.data.fl[x];

                double p = log(std::sqrt(xx*xx + yy*yy) + 1.)*M;
                double a = atan2(yy,xx);
                if( a < 0 )
                    a = 2*CV_PI + a;
                a *= ascale;

                mx[x] = (float)p;
                my[x] = (float)a;
            }
#endif
        }
    }

    cvRemap( src, dst, mapx, mapy, flags, cvScalarAll(0) );
}

void cv::logPolar( InputArray _src, OutputArray _dst,
                   Point2f center, double M, int flags )
{
    Mat src = _src.getMat();
    _dst.create( src.size(), src.type() );
    CvMat c_src = src, c_dst = _dst.getMat();
    cvLogPolar( &c_src, &c_dst, center, M, flags );
}

/****************************************************************************************
                                   Linear-Polar Transform
  J.L. Blanco, Apr 2009
 ****************************************************************************************/
CV_IMPL
void cvLinearPolar( const CvArr* srcarr, CvArr* dstarr,
            CvPoint2D32f center, double maxRadius, int flags )
{
    cv::Ptr<CvMat> mapx, mapy;

    CvMat srcstub, *src = (CvMat*)srcarr;
    CvMat dststub, *dst = (CvMat*)dstarr;
    CvSize ssize, dsize;

    src = cvGetMat( srcarr, &srcstub,0,0 );
    dst = cvGetMat( dstarr, &dststub,0,0 );

    if( !CV_ARE_TYPES_EQ( src, dst ))
        CV_Error( CV_StsUnmatchedFormats, "" );

    ssize.width = src->cols;
    ssize.height = src->rows;
    dsize.width = dst->cols;
    dsize.height = dst->rows;

    mapx.reset(cvCreateMat( dsize.height, dsize.width, CV_32F ));
    mapy.reset(cvCreateMat( dsize.height, dsize.width, CV_32F ));

    if( !(flags & CV_WARP_INVERSE_MAP) )
    {
        int phi, rho;

        for( phi = 0; phi < dsize.height; phi++ )
        {
            double cp = cos(phi*2*CV_PI/dsize.height);
            double sp = sin(phi*2*CV_PI/dsize.height);
            float* mx = (float*)(mapx->data.ptr + phi*mapx->step);
            float* my = (float*)(mapy->data.ptr + phi*mapy->step);

            for( rho = 0; rho < dsize.width; rho++ )
            {
                double r = maxRadius*(rho+1)/dsize.width;
                double x = r*cp + center.x;
                double y = r*sp + center.y;

                mx[rho] = (float)x;
                my[rho] = (float)y;
            }
        }
    }
    else
    {
        int x, y;
        CvMat bufx, bufy, bufp, bufa;
        const double ascale = ssize.height/(2*CV_PI);
        const double pscale = ssize.width/maxRadius;

        cv::AutoBuffer<float> _buf(4*dsize.width);
        float* buf = _buf;

        bufx = cvMat( 1, dsize.width, CV_32F, buf );
        bufy = cvMat( 1, dsize.width, CV_32F, buf + dsize.width );
        bufp = cvMat( 1, dsize.width, CV_32F, buf + dsize.width*2 );
        bufa = cvMat( 1, dsize.width, CV_32F, buf + dsize.width*3 );

        for( x = 0; x < dsize.width; x++ )
            bufx.data.fl[x] = (float)x - center.x;

        for( y = 0; y < dsize.height; y++ )
        {
            float* mx = (float*)(mapx->data.ptr + y*mapx->step);
            float* my = (float*)(mapy->data.ptr + y*mapy->step);

            for( x = 0; x < dsize.width; x++ )
                bufy.data.fl[x] = (float)y - center.y;

            cvCartToPolar( &bufx, &bufy, &bufp, &bufa, 0 );

            for( x = 0; x < dsize.width; x++ )
                bufp.data.fl[x] += 1.f;

            for( x = 0; x < dsize.width; x++ )
            {
                double rho = bufp.data.fl[x]*pscale;
                double phi = bufa.data.fl[x]*ascale;
                mx[x] = (float)rho;
                my[x] = (float)phi;
            }
        }
    }

    cvRemap( src, dst, mapx, mapy, flags, cvScalarAll(0) );
}

void cv::linearPolar( InputArray _src, OutputArray _dst,
                      Point2f center, double maxRadius, int flags )
{
    Mat src = _src.getMat();
    _dst.create( src.size(), src.type() );
    CvMat c_src = src, c_dst = _dst.getMat();
    cvLinearPolar( &c_src, &c_dst, center, maxRadius, flags );
}

/* End of file. */
