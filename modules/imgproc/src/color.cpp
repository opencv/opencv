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
// Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
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

/********************************* COPYRIGHT NOTICE *******************************\
  The function for RGB to Lab conversion is based on the MATLAB script
  RGB2Lab.m translated by Mark Ruzon from C code by Yossi Rubner, 23 September 1997.
  See the page [http://vision.stanford.edu/~ruzon/software/rgblab.html]
\**********************************************************************************/

/********************************* COPYRIGHT NOTICE *******************************\
  Original code for Bayer->BGR/RGB conversion is provided by Dirk Schaefer
  from MD-Mathematische Dienste GmbH. Below is the copyright notice:

    IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
    By downloading, copying, installing or using the software you agree
    to this license. If you do not agree to this license, do not download,
    install, copy or use the software.

    Contributors License Agreement:

      Copyright (c) 2002,
      MD-Mathematische Dienste GmbH
      Im Defdahl 5-10
      44141 Dortmund
      Germany
      www.md-it.de

    Redistribution and use in source and binary forms,
    with or without modification, are permitted provided
    that the following conditions are met:

    Redistributions of source code must retain
    the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
    The name of Contributor may not be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
    OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
    THE POSSIBILITY OF SUCH DAMAGE.
\**********************************************************************************/

#include "precomp.hpp"
#include "opencl_kernels.hpp"
#include <limits>

#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
#define MAX_IPP8u   255
#define MAX_IPP16u  65535
#define MAX_IPP32f  1.0
static IppStatus sts = ippInit();
#endif

namespace cv
{

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

// interpolates value of a function at x, 0 <= x <= n using a cubic spline.
template<typename _Tp> static inline _Tp splineInterpolate(_Tp x, const _Tp* tab, int n)
{
    // don't touch this function without urgent need - some versions of gcc fail to inline it correctly
    int ix = std::min(std::max(int(x), 0), n-1);
    x -= ix;
    tab += ix*4;
    return ((tab[3]*x + tab[2])*x + tab[1])*x + tab[0];
}


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

/*template<> struct ColorChannel<double>
{
    typedef double worktype_f;
    static double max() { return 1.; }
    static double half() { return 0.5; }
};*/


///////////////////////////// Top-level template function ////////////////////////////////

template <typename Cvt>
class CvtColorLoop_Invoker : public ParallelLoopBody
{
    typedef typename Cvt::channel_type _Tp;
public:

    CvtColorLoop_Invoker(const Mat& _src, Mat& _dst, const Cvt& _cvt) :
        ParallelLoopBody(), src(_src), dst(_dst), cvt(_cvt)
    {
    }

    virtual void operator()(const Range& range) const
    {
        const uchar* yS = src.ptr<uchar>(range.start);
        uchar* yD = dst.ptr<uchar>(range.start);

        for( int i = range.start; i < range.end; ++i, yS += src.step, yD += dst.step )
            cvt((const _Tp*)yS, (_Tp*)yD, src.cols);
    }

private:
    const Mat& src;
    Mat& dst;
    const Cvt& cvt;

    const CvtColorLoop_Invoker& operator= (const CvtColorLoop_Invoker&);
};

template <typename Cvt>
void CvtColorLoop(const Mat& src, Mat& dst, const Cvt& cvt)
{
    parallel_for_(Range(0, src.rows), CvtColorLoop_Invoker<Cvt>(src, dst, cvt), src.total()/(double)(1<<16) );
}

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
typedef IppStatus (CV_STDCALL* ippiReorderFunc)(const void *, int, void *, int, IppiSize, const int *);
typedef IppStatus (CV_STDCALL* ippiGeneralFunc)(const void *, int, void *, int, IppiSize);
typedef IppStatus (CV_STDCALL* ippiColor2GrayFunc)(const void *, int, void *, int, IppiSize, const Ipp32f *);

template <typename Cvt>
class CvtColorIPPLoop_Invoker : public ParallelLoopBody
{
public:

    CvtColorIPPLoop_Invoker(const Mat& _src, Mat& _dst, const Cvt& _cvt, bool *_ok) :
        ParallelLoopBody(), src(_src), dst(_dst), cvt(_cvt), ok(_ok)
    {
        *ok = true;
    }

    virtual void operator()(const Range& range) const
    {
        const void *yS = src.ptr<uchar>(range.start);
        void *yD = dst.ptr<uchar>(range.start);
        if( !cvt(yS, (int)src.step[0], yD, (int)dst.step[0], src.cols, range.end - range.start) )
            *ok = false;
    }

private:
    const Mat& src;
    Mat& dst;
    const Cvt& cvt;
    bool *ok;

    const CvtColorIPPLoop_Invoker& operator= (const CvtColorIPPLoop_Invoker&);
};

template <typename Cvt>
bool CvtColorIPPLoop(const Mat& src, Mat& dst, const Cvt& cvt)
{
    bool ok;
    parallel_for_(Range(0, src.rows), CvtColorIPPLoop_Invoker<Cvt>(src, dst, cvt, &ok), src.total()/(double)(1<<16) );
    return ok;
}

template <typename Cvt>
bool CvtColorIPPLoopCopy(Mat& src, Mat& dst, const Cvt& cvt)
{
    Mat temp;
    Mat &source = src;
    if( src.data == dst.data )
    {
        src.copyTo(temp);
        source = temp;
    }
    bool ok;
    parallel_for_(Range(0, source.rows), CvtColorIPPLoop_Invoker<Cvt>(source, dst, cvt, &ok), source.total()/(double)(1<<16) );
    return ok;
}

static IppStatus CV_STDCALL ippiSwapChannels_8u_C3C4Rf(const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int dstStep,
         IppiSize roiSize, const int *dstOrder)
{
    return ippiSwapChannels_8u_C3C4R(pSrc, srcStep, pDst, dstStep, roiSize, dstOrder, MAX_IPP8u);
}

static IppStatus CV_STDCALL ippiSwapChannels_16u_C3C4Rf(const Ipp16u* pSrc, int srcStep, Ipp16u* pDst, int dstStep,
         IppiSize roiSize, const int *dstOrder)
{
    return ippiSwapChannels_16u_C3C4R(pSrc, srcStep, pDst, dstStep, roiSize, dstOrder, MAX_IPP16u);
}

static IppStatus CV_STDCALL ippiSwapChannels_32f_C3C4Rf(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
         IppiSize roiSize, const int *dstOrder)
{
    return ippiSwapChannels_32f_C3C4R(pSrc, srcStep, pDst, dstStep, roiSize, dstOrder, MAX_IPP32f);
}

static ippiReorderFunc ippiSwapChannelsC3C4RTab[] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_C3C4Rf, 0, (ippiReorderFunc)ippiSwapChannels_16u_C3C4Rf, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_C3C4Rf, 0, 0
};

static ippiGeneralFunc ippiCopyAC4C3RTab[] =
{
    (ippiGeneralFunc)ippiCopy_8u_AC4C3R, 0, (ippiGeneralFunc)ippiCopy_16u_AC4C3R, 0,
    0, (ippiGeneralFunc)ippiCopy_32f_AC4C3R, 0, 0
};

static ippiReorderFunc ippiSwapChannelsC4C3RTab[] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_C4C3R, 0, (ippiReorderFunc)ippiSwapChannels_16u_C4C3R, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_C4C3R, 0, 0
};

static ippiReorderFunc ippiSwapChannelsC3RTab[] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_C3R, 0, (ippiReorderFunc)ippiSwapChannels_16u_C3R, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_C3R, 0, 0
};

static ippiReorderFunc ippiSwapChannelsC4RTab[] =
{
    (ippiReorderFunc)ippiSwapChannels_8u_AC4R, 0, (ippiReorderFunc)ippiSwapChannels_16u_AC4R, 0,
    0, (ippiReorderFunc)ippiSwapChannels_32f_AC4R, 0, 0
};

static ippiColor2GrayFunc ippiColor2GrayC3Tab[] =
{
    (ippiColor2GrayFunc)ippiColorToGray_8u_C3C1R, 0, (ippiColor2GrayFunc)ippiColorToGray_16u_C3C1R, 0,
    0, (ippiColor2GrayFunc)ippiColorToGray_32f_C3C1R, 0, 0
};

static ippiColor2GrayFunc ippiColor2GrayC4Tab[] =
{
    (ippiColor2GrayFunc)ippiColorToGray_8u_AC4C1R, 0, (ippiColor2GrayFunc)ippiColorToGray_16u_AC4C1R, 0,
    0, (ippiColor2GrayFunc)ippiColorToGray_32f_AC4C1R, 0, 0
};

static ippiGeneralFunc ippiRGB2GrayC3Tab[] =
{
    (ippiGeneralFunc)ippiRGBToGray_8u_C3C1R, 0, (ippiGeneralFunc)ippiRGBToGray_16u_C3C1R, 0,
    0, (ippiGeneralFunc)ippiRGBToGray_32f_C3C1R, 0, 0
};

static ippiGeneralFunc ippiRGB2GrayC4Tab[] =
{
    (ippiGeneralFunc)ippiRGBToGray_8u_AC4C1R, 0, (ippiGeneralFunc)ippiRGBToGray_16u_AC4C1R, 0,
    0, (ippiGeneralFunc)ippiRGBToGray_32f_AC4C1R, 0, 0
};

static ippiGeneralFunc ippiCopyP3C3RTab[] =
{
    (ippiGeneralFunc)ippiCopy_8u_P3C3R, 0, (ippiGeneralFunc)ippiCopy_16u_P3C3R, 0,
    0, (ippiGeneralFunc)ippiCopy_32f_P3C3R, 0, 0
};

static ippiGeneralFunc ippiRGB2XYZTab[] =
{
    (ippiGeneralFunc)ippiRGBToXYZ_8u_C3R, 0, (ippiGeneralFunc)ippiRGBToXYZ_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiRGBToXYZ_32f_C3R, 0, 0
};

static ippiGeneralFunc ippiXYZ2RGBTab[] =
{
    (ippiGeneralFunc)ippiXYZToRGB_8u_C3R, 0, (ippiGeneralFunc)ippiXYZToRGB_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiXYZToRGB_32f_C3R, 0, 0
};

static ippiGeneralFunc ippiRGB2HSVTab[] =
{
    (ippiGeneralFunc)ippiRGBToHSV_8u_C3R, 0, (ippiGeneralFunc)ippiRGBToHSV_16u_C3R, 0,
    0, 0, 0, 0
};

static ippiGeneralFunc ippiHSV2RGBTab[] =
{
    (ippiGeneralFunc)ippiHSVToRGB_8u_C3R, 0, (ippiGeneralFunc)ippiHSVToRGB_16u_C3R, 0,
    0, 0, 0, 0
};

static ippiGeneralFunc ippiRGB2HLSTab[] =
{
    (ippiGeneralFunc)ippiRGBToHLS_8u_C3R, 0, (ippiGeneralFunc)ippiRGBToHLS_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiRGBToHLS_32f_C3R, 0, 0
};

static ippiGeneralFunc ippiHLS2RGBTab[] =
{
    (ippiGeneralFunc)ippiHLSToRGB_8u_C3R, 0, (ippiGeneralFunc)ippiHLSToRGB_16u_C3R, 0,
    0, (ippiGeneralFunc)ippiHLSToRGB_32f_C3R, 0, 0
};

struct IPPGeneralFunctor
{
    IPPGeneralFunctor(ippiGeneralFunc _func) : func(_func){}
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        return func(src, srcStep, dst, dstStep, ippiSize(cols, rows)) >= 0;
    }
private:
    ippiGeneralFunc func;
};

struct IPPReorderFunctor
{
    IPPReorderFunctor(ippiReorderFunc _func, int _order0, int _order1, int _order2) : func(_func)
    {
        order[0] = _order0;
        order[1] = _order1;
        order[2] = _order2;
        order[3] = 3;
    }
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        return func(src, srcStep, dst, dstStep, ippiSize(cols, rows), order) >= 0;
    }
private:
    ippiReorderFunc func;
    int order[4];
};

struct IPPColor2GrayFunctor
{
    IPPColor2GrayFunctor(ippiColor2GrayFunc _func) : func(_func)
    {
        coeffs[0] = 0.114f;
        coeffs[1] = 0.587f;
        coeffs[2] = 0.299f;
    }
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        return func(src, srcStep, dst, dstStep, ippiSize(cols, rows), coeffs) >= 0;
    }
private:
    ippiColor2GrayFunc func;
    Ipp32f coeffs[3];
};

struct IPPGray2BGRFunctor
{
    IPPGray2BGRFunctor(ippiGeneralFunc _func) : func(_func){}
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        const void* srcarray[3] = { src, src, src };
        return func(srcarray, srcStep, dst, dstStep, ippiSize(cols, rows)) >= 0;
    }
private:
    ippiGeneralFunc func;
};

struct IPPGray2BGRAFunctor
{
    IPPGray2BGRAFunctor(ippiGeneralFunc _func1, ippiReorderFunc _func2, int _depth) : func1(_func1), func2(_func2), depth(_depth){}
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        const void* srcarray[3] = { src, src, src };
        Mat temp(rows, cols, CV_MAKETYPE(depth, 3));
        if(func1(srcarray, srcStep, temp.data, (int)temp.step[0], ippiSize(cols, rows)) < 0)
            return false;
        int order[4] = {0, 1, 2, 3};
        return func2(temp.data, (int)temp.step[0], dst, dstStep, ippiSize(cols, rows), order) >= 0;
    }
private:
    ippiGeneralFunc func1;
    ippiReorderFunc func2;
    int depth;
};

struct IPPReorderGeneralFunctor
{
    IPPReorderGeneralFunctor(ippiReorderFunc _func1, ippiGeneralFunc _func2, int _order0, int _order1, int _order2, int _depth) : func1(_func1), func2(_func2), depth(_depth)
    {
        order[0] = _order0;
        order[1] = _order1;
        order[2] = _order2;
        order[3] = 3;
    }
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        Mat temp;
        temp.create(rows, cols, CV_MAKETYPE(depth, 3));
        if(func1(src, srcStep, temp.data, (int)temp.step[0], ippiSize(cols, rows), order) < 0)
            return false;
        return func2(temp.data, (int)temp.step[0], dst, dstStep, ippiSize(cols, rows)) >= 0;
    }
private:
    ippiReorderFunc func1;
    ippiGeneralFunc func2;
    int order[4];
    int depth;
};

struct IPPGeneralReorderFunctor
{
    IPPGeneralReorderFunctor(ippiGeneralFunc _func1, ippiReorderFunc _func2, int _order0, int _order1, int _order2, int _depth) : func1(_func1), func2(_func2), depth(_depth)
    {
        order[0] = _order0;
        order[1] = _order1;
        order[2] = _order2;
        order[3] = 3;
    }
    bool operator()(const void *src, int srcStep, void *dst, int dstStep, int cols, int rows) const
    {
        Mat temp;
        temp.create(rows, cols, CV_MAKETYPE(depth, 3));
        if(func1(src, srcStep, temp.data, (int)temp.step[0], ippiSize(cols, rows)) < 0)
            return false;
        return func2(temp.data, (int)temp.step[0], dst, dstStep, ippiSize(cols, rows), order) >= 0;
    }
private:
    ippiGeneralFunc func1;
    ippiReorderFunc func2;
    int order[4];
    int depth;
};
#endif

////////////////// Various 3/4-channel to 3/4-channel RGB transformations /////////////////

template<typename _Tp> struct RGB2RGB
{
    typedef _Tp channel_type;

    RGB2RGB(int _srccn, int _dstcn, int _blueIdx) : srccn(_srccn), dstcn(_dstcn), blueIdx(_blueIdx) {}
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn, dcn = dstcn, bidx = blueIdx;
        if( dcn == 3 )
        {
            n *= 3;
            for( int i = 0; i < n; i += 3, src += scn )
            {
                _Tp t0 = src[bidx], t1 = src[1], t2 = src[bidx ^ 2];
                dst[i] = t0; dst[i+1] = t1; dst[i+2] = t2;
            }
        }
        else if( scn == 3 )
        {
            n *= 3;
            _Tp alpha = ColorChannel<_Tp>::max();
            for( int i = 0; i < n; i += 3, dst += 4 )
            {
                _Tp t0 = src[i], t1 = src[i+1], t2 = src[i+2];
                dst[bidx] = t0; dst[1] = t1; dst[bidx^2] = t2; dst[3] = alpha;
            }
        }
        else
        {
            n *= 4;
            for( int i = 0; i < n; i += 4 )
            {
                _Tp t0 = src[i], t1 = src[i+1], t2 = src[i+2], t3 = src[i+3];
                dst[i] = t2; dst[i+1] = t1; dst[i+2] = t0; dst[i+3] = t3;
            }
        }
    }

    int srccn, dstcn, blueIdx;
};

/////////// Transforming 16-bit (565 or 555) RGB to/from 24/32-bit (888[8]) RGB //////////

struct RGB5x52RGB
{
    typedef uchar channel_type;

    RGB5x52RGB(int _dstcn, int _blueIdx, int _greenBits)
        : dstcn(_dstcn), blueIdx(_blueIdx), greenBits(_greenBits) {}

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx;
        if( greenBits == 6 )
            for( int i = 0; i < n; i++, dst += dcn )
            {
                unsigned t = ((const ushort*)src)[i];
                dst[bidx] = (uchar)(t << 3);
                dst[1] = (uchar)((t >> 3) & ~3);
                dst[bidx ^ 2] = (uchar)((t >> 8) & ~7);
                if( dcn == 4 )
                    dst[3] = 255;
            }
        else
            for( int i = 0; i < n; i++, dst += dcn )
            {
                unsigned t = ((const ushort*)src)[i];
                dst[bidx] = (uchar)(t << 3);
                dst[1] = (uchar)((t >> 2) & ~7);
                dst[bidx ^ 2] = (uchar)((t >> 7) & ~7);
                if( dcn == 4 )
                    dst[3] = t & 0x8000 ? 255 : 0;
            }
    }

    int dstcn, blueIdx, greenBits;
};


struct RGB2RGB5x5
{
    typedef uchar channel_type;

    RGB2RGB5x5(int _srccn, int _blueIdx, int _greenBits)
        : srccn(_srccn), blueIdx(_blueIdx), greenBits(_greenBits) {}

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int scn = srccn, bidx = blueIdx;
        if( greenBits == 6 )
            for( int i = 0; i < n; i++, src += scn )
            {
                ((ushort*)dst)[i] = (ushort)((src[bidx] >> 3)|((src[1]&~3) << 3)|((src[bidx^2]&~7) << 8));
            }
        else if( scn == 3 )
            for( int i = 0; i < n; i++, src += 3 )
            {
                ((ushort*)dst)[i] = (ushort)((src[bidx] >> 3)|((src[1]&~7) << 2)|((src[bidx^2]&~7) << 7));
            }
        else
            for( int i = 0; i < n; i++, src += 4 )
            {
                ((ushort*)dst)[i] = (ushort)((src[bidx] >> 3)|((src[1]&~7) << 2)|
                    ((src[bidx^2]&~7) << 7)|(src[3] ? 0x8000 : 0));
            }
    }

    int srccn, blueIdx, greenBits;
};

///////////////////////////////// Color to/from Grayscale ////////////////////////////////

template<typename _Tp>
struct Gray2RGB
{
    typedef _Tp channel_type;

    Gray2RGB(int _dstcn) : dstcn(_dstcn) {}
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        if( dstcn == 3 )
            for( int i = 0; i < n; i++, dst += 3 )
            {
                dst[0] = dst[1] = dst[2] = src[i];
            }
        else
        {
            _Tp alpha = ColorChannel<_Tp>::max();
            for( int i = 0; i < n; i++, dst += 4 )
            {
                dst[0] = dst[1] = dst[2] = src[i];
                dst[3] = alpha;
            }
        }
    }

    int dstcn;
};


struct Gray2RGB5x5
{
    typedef uchar channel_type;

    Gray2RGB5x5(int _greenBits) : greenBits(_greenBits) {}
    void operator()(const uchar* src, uchar* dst, int n) const
    {
        if( greenBits == 6 )
            for( int i = 0; i < n; i++ )
            {
                int t = src[i];
                ((ushort*)dst)[i] = (ushort)((t >> 3)|((t & ~3) << 3)|((t & ~7) << 8));
            }
        else
            for( int i = 0; i < n; i++ )
            {
                int t = src[i] >> 3;
                ((ushort*)dst)[i] = (ushort)(t|(t << 5)|(t << 10));
            }
    }
    int greenBits;
};


#undef R2Y
#undef G2Y
#undef B2Y

enum
{
    yuv_shift = 14,
    xyz_shift = 12,
    R2Y = 4899,
    G2Y = 9617,
    B2Y = 1868,
    BLOCK_SIZE = 256
};


struct RGB5x52Gray
{
    typedef uchar channel_type;

    RGB5x52Gray(int _greenBits) : greenBits(_greenBits) {}
    void operator()(const uchar* src, uchar* dst, int n) const
    {
        if( greenBits == 6 )
            for( int i = 0; i < n; i++ )
            {
                int t = ((ushort*)src)[i];
                dst[i] = (uchar)CV_DESCALE(((t << 3) & 0xf8)*B2Y +
                                           ((t >> 3) & 0xfc)*G2Y +
                                           ((t >> 8) & 0xf8)*R2Y, yuv_shift);
            }
        else
            for( int i = 0; i < n; i++ )
            {
                int t = ((ushort*)src)[i];
                dst[i] = (uchar)CV_DESCALE(((t << 3) & 0xf8)*B2Y +
                                           ((t >> 2) & 0xf8)*G2Y +
                                           ((t >> 7) & 0xf8)*R2Y, yuv_shift);
            }
    }
    int greenBits;
};


template<typename _Tp> struct RGB2Gray
{
    typedef _Tp channel_type;

    RGB2Gray(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        static const float coeffs0[] = { 0.299f, 0.587f, 0.114f };
        memcpy( coeffs, _coeffs ? _coeffs : coeffs0, 3*sizeof(coeffs[0]) );
        if(blueIdx == 0)
            std::swap(coeffs[0], coeffs[2]);
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn;
        float cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];
        for(int i = 0; i < n; i++, src += scn)
            dst[i] = saturate_cast<_Tp>(src[0]*cb + src[1]*cg + src[2]*cr);
    }
    int srccn;
    float coeffs[3];
};


template<> struct RGB2Gray<uchar>
{
    typedef uchar channel_type;

    RGB2Gray(int _srccn, int blueIdx, const int* coeffs) : srccn(_srccn)
    {
        const int coeffs0[] = { R2Y, G2Y, B2Y };
        if(!coeffs) coeffs = coeffs0;

        int b = 0, g = 0, r = (1 << (yuv_shift-1));
        int db = coeffs[blueIdx^2], dg = coeffs[1], dr = coeffs[blueIdx];

        for( int i = 0; i < 256; i++, b += db, g += dg, r += dr )
        {
            tab[i] = b;
            tab[i+256] = g;
            tab[i+512] = r;
        }
    }
    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int scn = srccn;
        const int* _tab = tab;
        for(int i = 0; i < n; i++, src += scn)
            dst[i] = (uchar)((_tab[src[0]] + _tab[src[1]+256] + _tab[src[2]+512]) >> yuv_shift);
    }
    int srccn;
    int tab[256*3];
};


template<> struct RGB2Gray<ushort>
{
    typedef ushort channel_type;

    RGB2Gray(int _srccn, int blueIdx, const int* _coeffs) : srccn(_srccn)
    {
        static const int coeffs0[] = { R2Y, G2Y, B2Y };
        memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 3*sizeof(coeffs[0]));
        if( blueIdx == 0 )
            std::swap(coeffs[0], coeffs[2]);
    }

    void operator()(const ushort* src, ushort* dst, int n) const
    {
        int scn = srccn, cb = coeffs[0], cg = coeffs[1], cr = coeffs[2];
        for(int i = 0; i < n; i++, src += scn)
            dst[i] = (ushort)CV_DESCALE((unsigned)(src[0]*cb + src[1]*cg + src[2]*cr), yuv_shift);
    }
    int srccn;
    int coeffs[3];
};


///////////////////////////////////// RGB <-> YCrCb //////////////////////////////////////

template<typename _Tp> struct RGB2YCrCb_f
{
    typedef _Tp channel_type;

    RGB2YCrCb_f(int _srccn, int _blueIdx, const float* _coeffs) : srccn(_srccn), blueIdx(_blueIdx)
    {
        static const float coeffs0[] = {0.299f, 0.587f, 0.114f, 0.713f, 0.564f};
        memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 5*sizeof(coeffs[0]));
        if(blueIdx==0) std::swap(coeffs[0], coeffs[2]);
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn, bidx = blueIdx;
        const _Tp delta = ColorChannel<_Tp>::half();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        n *= 3;
        for(int i = 0; i < n; i += 3, src += scn)
        {
            _Tp Y = saturate_cast<_Tp>(src[0]*C0 + src[1]*C1 + src[2]*C2);
            _Tp Cr = saturate_cast<_Tp>((src[bidx^2] - Y)*C3 + delta);
            _Tp Cb = saturate_cast<_Tp>((src[bidx] - Y)*C4 + delta);
            dst[i] = Y; dst[i+1] = Cr; dst[i+2] = Cb;
        }
    }
    int srccn, blueIdx;
    float coeffs[5];
};


template<typename _Tp> struct RGB2YCrCb_i
{
    typedef _Tp channel_type;

    RGB2YCrCb_i(int _srccn, int _blueIdx, const int* _coeffs)
        : srccn(_srccn), blueIdx(_blueIdx)
    {
        static const int coeffs0[] = {R2Y, G2Y, B2Y, 11682, 9241};
        memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 5*sizeof(coeffs[0]));
        if(blueIdx==0) std::swap(coeffs[0], coeffs[2]);
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn, bidx = blueIdx;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3], C4 = coeffs[4];
        int delta = ColorChannel<_Tp>::half()*(1 << yuv_shift);
        n *= 3;
        for(int i = 0; i < n; i += 3, src += scn)
        {
            int Y = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, yuv_shift);
            int Cr = CV_DESCALE((src[bidx^2] - Y)*C3 + delta, yuv_shift);
            int Cb = CV_DESCALE((src[bidx] - Y)*C4 + delta, yuv_shift);
            dst[i] = saturate_cast<_Tp>(Y);
            dst[i+1] = saturate_cast<_Tp>(Cr);
            dst[i+2] = saturate_cast<_Tp>(Cb);
        }
    }
    int srccn, blueIdx;
    int coeffs[5];
};


template<typename _Tp> struct YCrCb2RGB_f
{
    typedef _Tp channel_type;

    YCrCb2RGB_f(int _dstcn, int _blueIdx, const float* _coeffs)
        : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        static const float coeffs0[] = {1.403f, -0.714f, -0.344f, 1.773f};
        memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 4*sizeof(coeffs[0]));
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx;
        const _Tp delta = ColorChannel<_Tp>::half(), alpha = ColorChannel<_Tp>::max();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];
        n *= 3;
        for(int i = 0; i < n; i += 3, dst += dcn)
        {
            _Tp Y = src[i];
            _Tp Cr = src[i+1];
            _Tp Cb = src[i+2];

            _Tp b = saturate_cast<_Tp>(Y + (Cb - delta)*C3);
            _Tp g = saturate_cast<_Tp>(Y + (Cb - delta)*C2 + (Cr - delta)*C1);
            _Tp r = saturate_cast<_Tp>(Y + (Cr - delta)*C0);

            dst[bidx] = b; dst[1] = g; dst[bidx^2] = r;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    float coeffs[4];
};


template<typename _Tp> struct YCrCb2RGB_i
{
    typedef _Tp channel_type;

    YCrCb2RGB_i(int _dstcn, int _blueIdx, const int* _coeffs)
        : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        static const int coeffs0[] = {22987, -11698, -5636, 29049};
        memcpy(coeffs, _coeffs ? _coeffs : coeffs0, 4*sizeof(coeffs[0]));
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int dcn = dstcn, bidx = blueIdx;
        const _Tp delta = ColorChannel<_Tp>::half(), alpha = ColorChannel<_Tp>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];
        n *= 3;
        for(int i = 0; i < n; i += 3, dst += dcn)
        {
            _Tp Y = src[i];
            _Tp Cr = src[i+1];
            _Tp Cb = src[i+2];

            int b = Y + CV_DESCALE((Cb - delta)*C3, yuv_shift);
            int g = Y + CV_DESCALE((Cb - delta)*C2 + (Cr - delta)*C1, yuv_shift);
            int r = Y + CV_DESCALE((Cr - delta)*C0, yuv_shift);

            dst[bidx] = saturate_cast<_Tp>(b);
            dst[1] = saturate_cast<_Tp>(g);
            dst[bidx^2] = saturate_cast<_Tp>(r);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    int coeffs[4];
};


////////////////////////////////////// RGB <-> XYZ ///////////////////////////////////////

static const float sRGB2XYZ_D65[] =
{
    0.412453f, 0.357580f, 0.180423f,
    0.212671f, 0.715160f, 0.072169f,
    0.019334f, 0.119193f, 0.950227f
};

static const float XYZ2sRGB_D65[] =
{
    3.240479f, -1.53715f, -0.498535f,
    -0.969256f, 1.875991f, 0.041556f,
    0.055648f, -0.204043f, 1.057311f
};

template<typename _Tp> struct RGB2XYZ_f
{
    typedef _Tp channel_type;

    RGB2XYZ_f(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        memcpy(coeffs, _coeffs ? _coeffs : sRGB2XYZ_D65, 9*sizeof(coeffs[0]));
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

        n *= 3;
        for(int i = 0; i < n; i += 3, src += scn)
        {
            _Tp X = saturate_cast<_Tp>(src[0]*C0 + src[1]*C1 + src[2]*C2);
            _Tp Y = saturate_cast<_Tp>(src[0]*C3 + src[1]*C4 + src[2]*C5);
            _Tp Z = saturate_cast<_Tp>(src[0]*C6 + src[1]*C7 + src[2]*C8);
            dst[i] = X; dst[i+1] = Y; dst[i+2] = Z;
        }
    }
    int srccn;
    float coeffs[9];
};


template<typename _Tp> struct RGB2XYZ_i
{
    typedef _Tp channel_type;

    RGB2XYZ_i(int _srccn, int blueIdx, const float* _coeffs) : srccn(_srccn)
    {
        static const int coeffs0[] =
        {
            1689,    1465,    739,
            871,     2929,    296,
            79,      488,     3892
        };
        for( int i = 0; i < 9; i++ )
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << xyz_shift)) : coeffs0[i];
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[2]);
            std::swap(coeffs[3], coeffs[5]);
            std::swap(coeffs[6], coeffs[8]);
        }
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int scn = srccn;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;
        for(int i = 0; i < n; i += 3, src += scn)
        {
            int X = CV_DESCALE(src[0]*C0 + src[1]*C1 + src[2]*C2, xyz_shift);
            int Y = CV_DESCALE(src[0]*C3 + src[1]*C4 + src[2]*C5, xyz_shift);
            int Z = CV_DESCALE(src[0]*C6 + src[1]*C7 + src[2]*C8, xyz_shift);
            dst[i] = saturate_cast<_Tp>(X); dst[i+1] = saturate_cast<_Tp>(Y);
            dst[i+2] = saturate_cast<_Tp>(Z);
        }
    }
    int srccn;
    int coeffs[9];
};


template<typename _Tp> struct XYZ2RGB_f
{
    typedef _Tp channel_type;

    XYZ2RGB_f(int _dstcn, int _blueIdx, const float* _coeffs)
    : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        memcpy(coeffs, _coeffs ? _coeffs : XYZ2sRGB_D65, 9*sizeof(coeffs[0]));
        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
    }

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int dcn = dstcn;
        _Tp alpha = ColorChannel<_Tp>::max();
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;
        for(int i = 0; i < n; i += 3, dst += dcn)
        {
            _Tp B = saturate_cast<_Tp>(src[i]*C0 + src[i+1]*C1 + src[i+2]*C2);
            _Tp G = saturate_cast<_Tp>(src[i]*C3 + src[i+1]*C4 + src[i+2]*C5);
            _Tp R = saturate_cast<_Tp>(src[i]*C6 + src[i+1]*C7 + src[i+2]*C8);
            dst[0] = B; dst[1] = G; dst[2] = R;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    float coeffs[9];
};


template<typename _Tp> struct XYZ2RGB_i
{
    typedef _Tp channel_type;

    XYZ2RGB_i(int _dstcn, int _blueIdx, const int* _coeffs)
    : dstcn(_dstcn), blueIdx(_blueIdx)
    {
        static const int coeffs0[] =
        {
            13273,  -6296,  -2042,
            -3970,   7684,    170,
              228,   -836,   4331
        };
        for(int i = 0; i < 9; i++)
            coeffs[i] = _coeffs ? cvRound(_coeffs[i]*(1 << xyz_shift)) : coeffs0[i];

        if(blueIdx == 0)
        {
            std::swap(coeffs[0], coeffs[6]);
            std::swap(coeffs[1], coeffs[7]);
            std::swap(coeffs[2], coeffs[8]);
        }
    }
    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        int dcn = dstcn;
        _Tp alpha = ColorChannel<_Tp>::max();
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;
        for(int i = 0; i < n; i += 3, dst += dcn)
        {
            int B = CV_DESCALE(src[i]*C0 + src[i+1]*C1 + src[i+2]*C2, xyz_shift);
            int G = CV_DESCALE(src[i]*C3 + src[i+1]*C4 + src[i+2]*C5, xyz_shift);
            int R = CV_DESCALE(src[i]*C6 + src[i+1]*C7 + src[i+2]*C8, xyz_shift);
            dst[0] = saturate_cast<_Tp>(B); dst[1] = saturate_cast<_Tp>(G);
            dst[2] = saturate_cast<_Tp>(R);
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }
    int dstcn, blueIdx;
    int coeffs[9];
};


////////////////////////////////////// RGB <-> HSV ///////////////////////////////////////


struct RGB2HSV_b
{
    typedef uchar channel_type;

    RGB2HSV_b(int _srccn, int _blueIdx, int _hrange)
    : srccn(_srccn), blueIdx(_blueIdx), hrange(_hrange)
    {
        CV_Assert( hrange == 180 || hrange == 256 );
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, bidx = blueIdx, scn = srccn;
        const int hsv_shift = 12;

        static int sdiv_table[256];
        static int hdiv_table180[256];
        static int hdiv_table256[256];
        static volatile bool initialized = false;

        int hr = hrange;
        const int* hdiv_table = hr == 180 ? hdiv_table180 : hdiv_table256;
        n *= 3;

        if( !initialized )
        {
            sdiv_table[0] = hdiv_table180[0] = hdiv_table256[0] = 0;
            for( i = 1; i < 256; i++ )
            {
                sdiv_table[i] = saturate_cast<int>((255 << hsv_shift)/(1.*i));
                hdiv_table180[i] = saturate_cast<int>((180 << hsv_shift)/(6.*i));
                hdiv_table256[i] = saturate_cast<int>((256 << hsv_shift)/(6.*i));
            }
            initialized = true;
        }

        for( i = 0; i < n; i += 3, src += scn )
        {
            int b = src[bidx], g = src[1], r = src[bidx^2];
            int h, s, v = b;
            int vmin = b, diff;
            int vr, vg;

            CV_CALC_MAX_8U( v, g );
            CV_CALC_MAX_8U( v, r );
            CV_CALC_MIN_8U( vmin, g );
            CV_CALC_MIN_8U( vmin, r );

            diff = v - vmin;
            vr = v == r ? -1 : 0;
            vg = v == g ? -1 : 0;

            s = (diff * sdiv_table[v] + (1 << (hsv_shift-1))) >> hsv_shift;
            h = (vr & (g - b)) +
                (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
            h = (h * hdiv_table[diff] + (1 << (hsv_shift-1))) >> hsv_shift;
            h += h < 0 ? hr : 0;

            dst[i] = saturate_cast<uchar>(h);
            dst[i+1] = (uchar)s;
            dst[i+2] = (uchar)v;
        }
    }

    int srccn, blueIdx, hrange;
};


struct RGB2HSV_f
{
    typedef float channel_type;

    RGB2HSV_f(int _srccn, int _blueIdx, float _hrange)
    : srccn(_srccn), blueIdx(_blueIdx), hrange(_hrange) {}

    void operator()(const float* src, float* dst, int n) const
    {
        int i, bidx = blueIdx, scn = srccn;
        float hscale = hrange*(1.f/360.f);
        n *= 3;

        for( i = 0; i < n; i += 3, src += scn )
        {
            float b = src[bidx], g = src[1], r = src[bidx^2];
            float h, s, v;

            float vmin, diff;

            v = vmin = r;
            if( v < g ) v = g;
            if( v < b ) v = b;
            if( vmin > g ) vmin = g;
            if( vmin > b ) vmin = b;

            diff = v - vmin;
            s = diff/(float)(fabs(v) + FLT_EPSILON);
            diff = (float)(60./(diff + FLT_EPSILON));
            if( v == r )
                h = (g - b)*diff;
            else if( v == g )
                h = (b - r)*diff + 120.f;
            else
                h = (r - g)*diff + 240.f;

            if( h < 0 ) h += 360.f;

            dst[i] = h*hscale;
            dst[i+1] = s;
            dst[i+2] = v;
        }
    }

    int srccn, blueIdx;
    float hrange;
};


struct HSV2RGB_f
{
    typedef float channel_type;

    HSV2RGB_f(int _dstcn, int _blueIdx, float _hrange)
    : dstcn(_dstcn), blueIdx(_blueIdx), hscale(6.f/_hrange) {}

    void operator()(const float* src, float* dst, int n) const
    {
        int i, bidx = blueIdx, dcn = dstcn;
        float _hscale = hscale;
        float alpha = ColorChannel<float>::max();
        n *= 3;

        for( i = 0; i < n; i += 3, dst += dcn )
        {
            float h = src[i], s = src[i+1], v = src[i+2];
            float b, g, r;

            if( s == 0 )
                b = g = r = v;
            else
            {
                static const int sector_data[][3]=
                    {{1,3,0}, {1,0,2}, {3,0,1}, {0,2,1}, {0,1,3}, {2,1,0}};
                float tab[4];
                int sector;
                h *= _hscale;
                if( h < 0 )
                    do h += 6; while( h < 0 );
                else if( h >= 6 )
                    do h -= 6; while( h >= 6 );
                sector = cvFloor(h);
                h -= sector;
                if( (unsigned)sector >= 6u )
                {
                    sector = 0;
                    h = 0.f;
                }

                tab[0] = v;
                tab[1] = v*(1.f - s);
                tab[2] = v*(1.f - s*h);
                tab[3] = v*(1.f - s*(1.f - h));

                b = tab[sector_data[sector][0]];
                g = tab[sector_data[sector][1]];
                r = tab[sector_data[sector][2]];
            }

            dst[bidx] = b;
            dst[1] = g;
            dst[bidx^2] = r;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn, blueIdx;
    float hscale;
};


struct HSV2RGB_b
{
    typedef uchar channel_type;

    HSV2RGB_b(int _dstcn, int _blueIdx, int _hrange)
    : dstcn(_dstcn), cvt(3, _blueIdx, (float)_hrange)
    {}

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
        float buf[3*BLOCK_SIZE];

        for( i = 0; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);

            for( j = 0; j < dn*3; j += 3 )
            {
                buf[j] = src[j];
                buf[j+1] = src[j+1]*(1.f/255.f);
                buf[j+2] = src[j+2]*(1.f/255.f);
            }
            cvt(buf, buf, dn);

            for( j = 0; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    int dstcn;
    HSV2RGB_f cvt;
};


///////////////////////////////////// RGB <-> HLS ////////////////////////////////////////

struct RGB2HLS_f
{
    typedef float channel_type;

    RGB2HLS_f(int _srccn, int _blueIdx, float _hrange)
    : srccn(_srccn), blueIdx(_blueIdx), hrange(_hrange) {}

    void operator()(const float* src, float* dst, int n) const
    {
        int i, bidx = blueIdx, scn = srccn;
        float hscale = hrange*(1.f/360.f);
        n *= 3;

        for( i = 0; i < n; i += 3, src += scn )
        {
            float b = src[bidx], g = src[1], r = src[bidx^2];
            float h = 0.f, s = 0.f, l;
            float vmin, vmax, diff;

            vmax = vmin = r;
            if( vmax < g ) vmax = g;
            if( vmax < b ) vmax = b;
            if( vmin > g ) vmin = g;
            if( vmin > b ) vmin = b;

            diff = vmax - vmin;
            l = (vmax + vmin)*0.5f;

            if( diff > FLT_EPSILON )
            {
                s = l < 0.5f ? diff/(vmax + vmin) : diff/(2 - vmax - vmin);
                diff = 60.f/diff;

                if( vmax == r )
                    h = (g - b)*diff;
                else if( vmax == g )
                    h = (b - r)*diff + 120.f;
                else
                    h = (r - g)*diff + 240.f;

                if( h < 0.f ) h += 360.f;
            }

            dst[i] = h*hscale;
            dst[i+1] = l;
            dst[i+2] = s;
        }
    }

    int srccn, blueIdx;
    float hrange;
};


struct RGB2HLS_b
{
    typedef uchar channel_type;

    RGB2HLS_b(int _srccn, int _blueIdx, int _hrange)
    : srccn(_srccn), cvt(3, _blueIdx, (float)_hrange) {}

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, scn = srccn;
        float buf[3*BLOCK_SIZE];

        for( i = 0; i < n; i += BLOCK_SIZE, dst += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);

            for( j = 0; j < dn*3; j += 3, src += scn )
            {
                buf[j] = src[0]*(1.f/255.f);
                buf[j+1] = src[1]*(1.f/255.f);
                buf[j+2] = src[2]*(1.f/255.f);
            }
            cvt(buf, buf, dn);

            for( j = 0; j < dn*3; j += 3 )
            {
                dst[j] = saturate_cast<uchar>(buf[j]);
                dst[j+1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[j+2] = saturate_cast<uchar>(buf[j+2]*255.f);
            }
        }
    }

    int srccn;
    RGB2HLS_f cvt;
};


struct HLS2RGB_f
{
    typedef float channel_type;

    HLS2RGB_f(int _dstcn, int _blueIdx, float _hrange)
    : dstcn(_dstcn), blueIdx(_blueIdx), hscale(6.f/_hrange) {}

    void operator()(const float* src, float* dst, int n) const
    {
        int i, bidx = blueIdx, dcn = dstcn;
        float _hscale = hscale;
        float alpha = ColorChannel<float>::max();
        n *= 3;

        for( i = 0; i < n; i += 3, dst += dcn )
        {
            float h = src[i], l = src[i+1], s = src[i+2];
            float b, g, r;

            if( s == 0 )
                b = g = r = l;
            else
            {
                static const int sector_data[][3]=
                {{1,3,0}, {1,0,2}, {3,0,1}, {0,2,1}, {0,1,3}, {2,1,0}};
                float tab[4];
                int sector;

                float p2 = l <= 0.5f ? l*(1 + s) : l + s - l*s;
                float p1 = 2*l - p2;

                h *= _hscale;
                if( h < 0 )
                    do h += 6; while( h < 0 );
                else if( h >= 6 )
                    do h -= 6; while( h >= 6 );

                assert( 0 <= h && h < 6 );
                sector = cvFloor(h);
                h -= sector;

                tab[0] = p2;
                tab[1] = p1;
                tab[2] = p1 + (p2 - p1)*(1-h);
                tab[3] = p1 + (p2 - p1)*h;

                b = tab[sector_data[sector][0]];
                g = tab[sector_data[sector][1]];
                r = tab[sector_data[sector][2]];
            }

            dst[bidx] = b;
            dst[1] = g;
            dst[bidx^2] = r;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn, blueIdx;
    float hscale;
};


struct HLS2RGB_b
{
    typedef uchar channel_type;

    HLS2RGB_b(int _dstcn, int _blueIdx, int _hrange)
    : dstcn(_dstcn), cvt(3, _blueIdx, (float)_hrange)
    {}

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
        float buf[3*BLOCK_SIZE];

        for( i = 0; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);

            for( j = 0; j < dn*3; j += 3 )
            {
                buf[j] = src[j];
                buf[j+1] = src[j+1]*(1.f/255.f);
                buf[j+2] = src[j+2]*(1.f/255.f);
            }
            cvt(buf, buf, dn);

            for( j = 0; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    int dstcn;
    HLS2RGB_f cvt;
};


///////////////////////////////////// RGB <-> L*a*b* /////////////////////////////////////

static const float D65[] = { 0.950456f, 1.f, 1.088754f };

enum { LAB_CBRT_TAB_SIZE = 1024, GAMMA_TAB_SIZE = 1024 };
static float LabCbrtTab[LAB_CBRT_TAB_SIZE*4];
static const float LabCbrtTabScale = LAB_CBRT_TAB_SIZE/1.5f;

static float sRGBGammaTab[GAMMA_TAB_SIZE*4], sRGBInvGammaTab[GAMMA_TAB_SIZE*4];
static const float GammaTabScale = (float)GAMMA_TAB_SIZE;

static ushort sRGBGammaTab_b[256], linearGammaTab_b[256];
#undef lab_shift
#define lab_shift xyz_shift
#define gamma_shift 3
#define lab_shift2 (lab_shift + gamma_shift)
#define LAB_CBRT_TAB_SIZE_B (256*3/2*(1<<gamma_shift))
static ushort LabCbrtTab_b[LAB_CBRT_TAB_SIZE_B];

static void initLabTabs()
{
    static bool initialized = false;
    if(!initialized)
    {
        float f[LAB_CBRT_TAB_SIZE+1], g[GAMMA_TAB_SIZE+1], ig[GAMMA_TAB_SIZE+1], scale = 1.f/LabCbrtTabScale;
        int i;
        for(i = 0; i <= LAB_CBRT_TAB_SIZE; i++)
        {
            float x = i*scale;
            f[i] = x < 0.008856f ? x*7.787f + 0.13793103448275862f : cvCbrt(x);
        }
        splineBuild(f, LAB_CBRT_TAB_SIZE, LabCbrtTab);

        scale = 1.f/GammaTabScale;
        for(i = 0; i <= GAMMA_TAB_SIZE; i++)
        {
            float x = i*scale;
            g[i] = x <= 0.04045f ? x*(1.f/12.92f) : (float)std::pow((double)(x + 0.055)*(1./1.055), 2.4);
            ig[i] = x <= 0.0031308 ? x*12.92f : (float)(1.055*std::pow((double)x, 1./2.4) - 0.055);
        }
        splineBuild(g, GAMMA_TAB_SIZE, sRGBGammaTab);
        splineBuild(ig, GAMMA_TAB_SIZE, sRGBInvGammaTab);

        for(i = 0; i < 256; i++)
        {
            float x = i*(1.f/255.f);
            sRGBGammaTab_b[i] = saturate_cast<ushort>(255.f*(1 << gamma_shift)*(x <= 0.04045f ? x*(1.f/12.92f) : (float)std::pow((double)(x + 0.055)*(1./1.055), 2.4)));
            linearGammaTab_b[i] = (ushort)(i*(1 << gamma_shift));
        }

        for(i = 0; i < LAB_CBRT_TAB_SIZE_B; i++)
        {
            float x = i*(1.f/(255.f*(1 << gamma_shift)));
            LabCbrtTab_b[i] = saturate_cast<ushort>((1 << lab_shift2)*(x < 0.008856f ? x*7.787f + 0.13793103448275862f : cvCbrt(x)));
        }
        initialized = true;
    }
}

struct RGB2Lab_b
{
    typedef uchar channel_type;

    RGB2Lab_b(int _srccn, int blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb)
    : srccn(_srccn), srgb(_srgb)
    {
        static volatile int _3 = 3;
        initLabTabs();

        if(!_coeffs) _coeffs = sRGB2XYZ_D65;
        if(!_whitept) _whitept = D65;
        float scale[] =
        {
            (1 << lab_shift)/_whitept[0],
            (float)(1 << lab_shift),
            (1 << lab_shift)/_whitept[2]
        };

        for( int i = 0; i < _3; i++ )
        {
            coeffs[i*3+(blueIdx^2)] = cvRound(_coeffs[i*3]*scale[i]);
            coeffs[i*3+1] = cvRound(_coeffs[i*3+1]*scale[i]);
            coeffs[i*3+blueIdx] = cvRound(_coeffs[i*3+2]*scale[i]);

            CV_Assert( coeffs[i] >= 0 && coeffs[i*3+1] >= 0 && coeffs[i*3+2] >= 0 &&
                      coeffs[i*3] + coeffs[i*3+1] + coeffs[i*3+2] < 2*(1 << lab_shift) );
        }
    }

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        const int Lscale = (116*255+50)/100;
        const int Lshift = -((16*255*(1 << lab_shift2) + 50)/100);
        const ushort* tab = srgb ? sRGBGammaTab_b : linearGammaTab_b;
        int i, scn = srccn;
        int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
            C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
            C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        for( i = 0; i < n; i += 3, src += scn )
        {
            int R = tab[src[0]], G = tab[src[1]], B = tab[src[2]];
            int fX = LabCbrtTab_b[CV_DESCALE(R*C0 + G*C1 + B*C2, lab_shift)];
            int fY = LabCbrtTab_b[CV_DESCALE(R*C3 + G*C4 + B*C5, lab_shift)];
            int fZ = LabCbrtTab_b[CV_DESCALE(R*C6 + G*C7 + B*C8, lab_shift)];

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
};


#define clip(value) \
    value < 0.0f ? 0.0f : value > 1.0f ? 1.0f : value;

struct RGB2Lab_f
{
    typedef float channel_type;

    RGB2Lab_f(int _srccn, int blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb)
    : srccn(_srccn), srgb(_srgb)
    {
        volatile int _3 = 3;
        initLabTabs();

        if (!_coeffs)
            _coeffs = sRGB2XYZ_D65;
        if (!_whitept)
            _whitept = D65;

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

    void operator()(const float* src, float* dst, int n) const
    {
        int i, scn = srccn;
        float gscale = GammaTabScale;
        const float* gammaTab = srgb ? sRGBGammaTab : 0;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        n *= 3;

        static const float _1_3 = 1.0f / 3.0f;
        static const float _a = 16.0f / 116.0f;
        for (i = 0; i < n; i += 3, src += scn )
        {
            float R = clip(src[0]);
            float G = clip(src[1]);
            float B = clip(src[2]);

//            CV_Assert(R >= 0.0f && R <= 1.0f);
//            CV_Assert(G >= 0.0f && G <= 1.0f);
//            CV_Assert(B >= 0.0f && B <= 1.0f);

            if (gammaTab)
            {
                R = splineInterpolate(R * gscale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G * gscale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B * gscale, gammaTab, GAMMA_TAB_SIZE);
            }
            float X = R*C0 + G*C1 + B*C2;
            float Y = R*C3 + G*C4 + B*C5;
            float Z = R*C6 + G*C7 + B*C8;

            float FX = X > 0.008856f ? std::pow(X, _1_3) : (7.787f * X + _a);
            float FY = Y > 0.008856f ? std::pow(Y, _1_3) : (7.787f * Y + _a);
            float FZ = Z > 0.008856f ? std::pow(Z, _1_3) : (7.787f * Z + _a);

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
};

struct Lab2RGB_f
{
    typedef float channel_type;

    Lab2RGB_f( int _dstcn, int blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb )
    : dstcn(_dstcn), srgb(_srgb), blueInd(blueIdx)
    {
        initLabTabs();

        if(!_coeffs)
            _coeffs = XYZ2sRGB_D65;
        if(!_whitept)
            _whitept = D65;

        for( int i = 0; i < 3; i++ )
        {
            coeffs[i+(blueIdx^2)*3] = _coeffs[i]*_whitept[i];
            coeffs[i+3] = _coeffs[i+3]*_whitept[i];
            coeffs[i+blueIdx*3] = _coeffs[i+6]*_whitept[i];
        }
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int i, dcn = dstcn;
        const float* gammaTab = srgb ? sRGBInvGammaTab : 0;
        float gscale = GammaTabScale;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
        C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
        C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        float alpha = ColorChannel<float>::max();
        n *= 3;

        static const float lThresh = 0.008856f * 903.3f;
        static const float fThresh = 7.787f * 0.008856f + 16.0f / 116.0f;
        for (i = 0; i < n; i += 3, dst += dcn)
        {
            float li = src[i];
            float ai = src[i + 1];
            float bi = src[i + 2];

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
            float ro = clip(C0 * x + C1 * y + C2 * z);
            float go = clip(C3 * x + C4 * y + C5 * z);
            float bo = clip(C6 * x + C7 * y + C8 * z);

//            CV_Assert(ro >= 0.0f && ro <= 1.0f);
//            CV_Assert(go >= 0.0f && go <= 1.0f);
//            CV_Assert(bo >= 0.0f && bo <= 1.0f);

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
    int blueInd;
};

#undef clip

struct Lab2RGB_b
{
    typedef uchar channel_type;

    Lab2RGB_b( int _dstcn, int blueIdx, const float* _coeffs,
               const float* _whitept, bool _srgb )
    : dstcn(_dstcn), cvt(3, blueIdx, _coeffs, _whitept, _srgb ) {}

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
        float buf[3*BLOCK_SIZE];

        for( i = 0; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);

            for( j = 0; j < dn*3; j += 3 )
            {
                buf[j] = src[j]*(100.f/255.f);
                buf[j+1] = (float)(src[j+1] - 128);
                buf[j+2] = (float)(src[j+2] - 128);
            }
            cvt(buf, buf, dn);

            for( j = 0; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    int dstcn;
    Lab2RGB_f cvt;
};


///////////////////////////////////// RGB <-> L*u*v* /////////////////////////////////////

struct RGB2Luv_f
{
    typedef float channel_type;

    RGB2Luv_f( int _srccn, int blueIdx, const float* _coeffs,
               const float* whitept, bool _srgb )
    : srccn(_srccn), srgb(_srgb)
    {
        volatile int i;
        initLabTabs();

        if(!_coeffs) _coeffs = sRGB2XYZ_D65;
        if(!whitept) whitept = D65;

        for( i = 0; i < 3; i++ )
        {
            coeffs[i*3] = _coeffs[i*3];
            coeffs[i*3+1] = _coeffs[i*3+1];
            coeffs[i*3+2] = _coeffs[i*3+2];
            if( blueIdx == 0 )
                std::swap(coeffs[i*3], coeffs[i*3+2]);
            CV_Assert( coeffs[i*3] >= 0 && coeffs[i*3+1] >= 0 && coeffs[i*3+2] >= 0 &&
                      coeffs[i*3] + coeffs[i*3+1] + coeffs[i*3+2] < 1.5f );
        }

        float d = 1.f/(whitept[0] + whitept[1]*15 + whitept[2]*3);
        un = 4*whitept[0]*d;
        vn = 9*whitept[1]*d;

        CV_Assert(whitept[1] == 1.f);
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int i, scn = srccn;
        float gscale = GammaTabScale;
        const float* gammaTab = srgb ? sRGBGammaTab : 0;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        float _un = 13*un, _vn = 13*vn;
        n *= 3;

        for( i = 0; i < n; i += 3, src += scn )
        {
            float R = src[0], G = src[1], B = src[2];
            if( gammaTab )
            {
                R = splineInterpolate(R*gscale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*gscale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*gscale, gammaTab, GAMMA_TAB_SIZE);
            }

            float X = R*C0 + G*C1 + B*C2;
            float Y = R*C3 + G*C4 + B*C5;
            float Z = R*C6 + G*C7 + B*C8;

            float L = splineInterpolate(Y*LabCbrtTabScale, LabCbrtTab, LAB_CBRT_TAB_SIZE);
            L = 116.f*L - 16.f;

            float d = (4*13) / std::max(X + 15 * Y + 3 * Z, FLT_EPSILON);
            float u = L*(X*d - _un);
            float v = L*((9*0.25f)*Y*d - _vn);

            dst[i] = L; dst[i+1] = u; dst[i+2] = v;
        }
    }

    int srccn;
    float coeffs[9], un, vn;
    bool srgb;
};


struct Luv2RGB_f
{
    typedef float channel_type;

    Luv2RGB_f( int _dstcn, int blueIdx, const float* _coeffs,
              const float* whitept, bool _srgb )
    : dstcn(_dstcn), srgb(_srgb)
    {
        initLabTabs();

        if(!_coeffs) _coeffs = XYZ2sRGB_D65;
        if(!whitept) whitept = D65;

        for( int i = 0; i < 3; i++ )
        {
            coeffs[i+(blueIdx^2)*3] = _coeffs[i];
            coeffs[i+3] = _coeffs[i+3];
            coeffs[i+blueIdx*3] = _coeffs[i+6];
        }

        float d = 1.f/(whitept[0] + whitept[1]*15 + whitept[2]*3);
        un = 4*whitept[0]*d;
        vn = 9*whitept[1]*d;

        CV_Assert(whitept[1] == 1.f);
    }

    void operator()(const float* src, float* dst, int n) const
    {
        int i, dcn = dstcn;
        const float* gammaTab = srgb ? sRGBInvGammaTab : 0;
        float gscale = GammaTabScale;
        float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
              C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
              C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
        float alpha = ColorChannel<float>::max();
        float _un = un, _vn = vn;
        n *= 3;

        for( i = 0; i < n; i += 3, dst += dcn )
        {
            float L = src[i], u = src[i+1], v = src[i+2], d, X, Y, Z;
            Y = (L + 16.f) * (1.f/116.f);
            Y = Y*Y*Y;
            d = (1.f/13.f)/L;
            u = u*d + _un;
            v = v*d + _vn;
            float iv = 1.f/v;
            X = 2.25f * u * Y * iv ;
            Z = (12 - 3 * u - 20 * v) * Y * 0.25f * iv;

            float R = X*C0 + Y*C1 + Z*C2;
            float G = X*C3 + Y*C4 + Z*C5;
            float B = X*C6 + Y*C7 + Z*C8;

            if( gammaTab )
            {
                R = splineInterpolate(R*gscale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*gscale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*gscale, gammaTab, GAMMA_TAB_SIZE);
            }

            dst[0] = R; dst[1] = G; dst[2] = B;
            if( dcn == 4 )
                dst[3] = alpha;
        }
    }

    int dstcn;
    float coeffs[9], un, vn;
    bool srgb;
};


struct RGB2Luv_b
{
    typedef uchar channel_type;

    RGB2Luv_b( int _srccn, int blueIdx, const float* _coeffs,
               const float* _whitept, bool _srgb )
    : srccn(_srccn), cvt(3, blueIdx, _coeffs, _whitept, _srgb) {}

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, scn = srccn;
        float buf[3*BLOCK_SIZE];

        for( i = 0; i < n; i += BLOCK_SIZE, dst += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);

            for( j = 0; j < dn*3; j += 3, src += scn )
            {
                buf[j] = src[0]*(1.f/255.f);
                buf[j+1] = (float)(src[1]*(1.f/255.f));
                buf[j+2] = (float)(src[2]*(1.f/255.f));
            }
            cvt(buf, buf, dn);

            for( j = 0; j < dn*3; j += 3 )
            {
                dst[j] = saturate_cast<uchar>(buf[j]*2.55f);
                dst[j+1] = saturate_cast<uchar>(buf[j+1]*0.72033898305084743f + 96.525423728813564f);
                dst[j+2] = saturate_cast<uchar>(buf[j+2]*0.99609375f + 139.453125f);
            }
        }
    }

    int srccn;
    RGB2Luv_f cvt;
};


struct Luv2RGB_b
{
    typedef uchar channel_type;

    Luv2RGB_b( int _dstcn, int blueIdx, const float* _coeffs,
               const float* _whitept, bool _srgb )
    : dstcn(_dstcn), cvt(3, blueIdx, _coeffs, _whitept, _srgb ) {}

    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, j, dcn = dstcn;
        uchar alpha = ColorChannel<uchar>::max();
        float buf[3*BLOCK_SIZE];

        for( i = 0; i < n; i += BLOCK_SIZE, src += BLOCK_SIZE*3 )
        {
            int dn = std::min(n - i, (int)BLOCK_SIZE);

            for( j = 0; j < dn*3; j += 3 )
            {
                buf[j] = src[j]*(100.f/255.f);
                buf[j+1] = (float)(src[j+1]*1.388235294117647f - 134.f);
                buf[j+2] = (float)(src[j+2]*1.003921568627451f - 140.f);
            }
            cvt(buf, buf, dn);

            for( j = 0; j < dn*3; j += 3, dst += dcn )
            {
                dst[0] = saturate_cast<uchar>(buf[j]*255.f);
                dst[1] = saturate_cast<uchar>(buf[j+1]*255.f);
                dst[2] = saturate_cast<uchar>(buf[j+2]*255.f);
                if( dcn == 4 )
                    dst[3] = alpha;
            }
        }
    }

    int dstcn;
    Luv2RGB_f cvt;
};


///////////////////////////////////// YUV420 -> RGB /////////////////////////////////////

const int ITUR_BT_601_CY = 1220542;
const int ITUR_BT_601_CUB = 2116026;
const int ITUR_BT_601_CUG = -409993;
const int ITUR_BT_601_CVG = -852492;
const int ITUR_BT_601_CVR = 1673527;
const int ITUR_BT_601_SHIFT = 20;

// Coefficients for RGB to YUV420p conversion
const int ITUR_BT_601_CRY =  269484;
const int ITUR_BT_601_CGY =  528482;
const int ITUR_BT_601_CBY =  102760;
const int ITUR_BT_601_CRU = -155188;
const int ITUR_BT_601_CGU = -305135;
const int ITUR_BT_601_CBU =  460324;
const int ITUR_BT_601_CGV = -385875;
const int ITUR_BT_601_CBV = -74448;

template<int bIdx, int uIdx>
struct YUV420sp2RGB888Invoker : ParallelLoopBody
{
    Mat* dst;
    const uchar* my1, *muv;
    int width, stride;

    YUV420sp2RGB888Invoker(Mat* _dst, int _stride, const uchar* _y1, const uchar* _uv)
        : dst(_dst), my1(_y1), muv(_uv), width(_dst->cols), stride(_stride) {}

    void operator()(const Range& range) const
    {
        int rangeBegin = range.start * 2;
        int rangeEnd = range.end * 2;

        //R = 1.164(Y - 16) + 1.596(V - 128)
        //G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
        //B = 1.164(Y - 16)                  + 2.018(U - 128)

        //R = (1220542(Y - 16) + 1673527(V - 128)                  + (1 << 19)) >> 20
        //G = (1220542(Y - 16) - 852492(V - 128) - 409993(U - 128) + (1 << 19)) >> 20
        //B = (1220542(Y - 16)                  + 2116026(U - 128) + (1 << 19)) >> 20

        const uchar* y1 = my1 + rangeBegin * stride, *uv = muv + rangeBegin * stride / 2;

#ifdef HAVE_TEGRA_OPTIMIZATION
        if(tegra::cvtYUV4202RGB(bIdx, uIdx, 3, y1, uv, stride, dst->ptr<uchar>(rangeBegin), dst->step, rangeEnd - rangeBegin, dst->cols))
            return;
#endif

        for (int j = rangeBegin; j < rangeEnd; j += 2, y1 += stride * 2, uv += stride)
        {
            uchar* row1 = dst->ptr<uchar>(j);
            uchar* row2 = dst->ptr<uchar>(j + 1);
            const uchar* y2 = y1 + stride;

            for (int i = 0; i < width; i += 2, row1 += 6, row2 += 6)
            {
                int u = int(uv[i + 0 + uIdx]) - 128;
                int v = int(uv[i + 1 - uIdx]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(y1[i]) - 16) * ITUR_BT_601_CY;
                row1[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row1[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row1[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);

                int y01 = std::max(0, int(y1[i + 1]) - 16) * ITUR_BT_601_CY;
                row1[5-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row1[4]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row1[3+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);

                int y10 = std::max(0, int(y2[i]) - 16) * ITUR_BT_601_CY;
                row2[2-bIdx] = saturate_cast<uchar>((y10 + ruv) >> ITUR_BT_601_SHIFT);
                row2[1]      = saturate_cast<uchar>((y10 + guv) >> ITUR_BT_601_SHIFT);
                row2[bIdx]   = saturate_cast<uchar>((y10 + buv) >> ITUR_BT_601_SHIFT);

                int y11 = std::max(0, int(y2[i + 1]) - 16) * ITUR_BT_601_CY;
                row2[5-bIdx] = saturate_cast<uchar>((y11 + ruv) >> ITUR_BT_601_SHIFT);
                row2[4]      = saturate_cast<uchar>((y11 + guv) >> ITUR_BT_601_SHIFT);
                row2[3+bIdx] = saturate_cast<uchar>((y11 + buv) >> ITUR_BT_601_SHIFT);
            }
        }
    }
};

template<int bIdx, int uIdx>
struct YUV420sp2RGBA8888Invoker : ParallelLoopBody
{
    Mat* dst;
    const uchar* my1, *muv;
    int width, stride;

    YUV420sp2RGBA8888Invoker(Mat* _dst, int _stride, const uchar* _y1, const uchar* _uv)
        : dst(_dst), my1(_y1), muv(_uv), width(_dst->cols), stride(_stride) {}

    void operator()(const Range& range) const
    {
        int rangeBegin = range.start * 2;
        int rangeEnd = range.end * 2;

        //R = 1.164(Y - 16) + 1.596(V - 128)
        //G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
        //B = 1.164(Y - 16)                  + 2.018(U - 128)

        //R = (1220542(Y - 16) + 1673527(V - 128)                  + (1 << 19)) >> 20
        //G = (1220542(Y - 16) - 852492(V - 128) - 409993(U - 128) + (1 << 19)) >> 20
        //B = (1220542(Y - 16)                  + 2116026(U - 128) + (1 << 19)) >> 20

        const uchar* y1 = my1 + rangeBegin * stride, *uv = muv + rangeBegin * stride / 2;

#ifdef HAVE_TEGRA_OPTIMIZATION
        if(tegra::cvtYUV4202RGB(bIdx, uIdx, 4, y1, uv, stride, dst->ptr<uchar>(rangeBegin), dst->step, rangeEnd - rangeBegin, dst->cols))
            return;
#endif

        for (int j = rangeBegin; j < rangeEnd; j += 2, y1 += stride * 2, uv += stride)
        {
            uchar* row1 = dst->ptr<uchar>(j);
            uchar* row2 = dst->ptr<uchar>(j + 1);
            const uchar* y2 = y1 + stride;

            for (int i = 0; i < width; i += 2, row1 += 8, row2 += 8)
            {
                int u = int(uv[i + 0 + uIdx]) - 128;
                int v = int(uv[i + 1 - uIdx]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(y1[i]) - 16) * ITUR_BT_601_CY;
                row1[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row1[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row1[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);
                row1[3]      = uchar(0xff);

                int y01 = std::max(0, int(y1[i + 1]) - 16) * ITUR_BT_601_CY;
                row1[6-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row1[5]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row1[4+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);
                row1[7]      = uchar(0xff);

                int y10 = std::max(0, int(y2[i]) - 16) * ITUR_BT_601_CY;
                row2[2-bIdx] = saturate_cast<uchar>((y10 + ruv) >> ITUR_BT_601_SHIFT);
                row2[1]      = saturate_cast<uchar>((y10 + guv) >> ITUR_BT_601_SHIFT);
                row2[bIdx]   = saturate_cast<uchar>((y10 + buv) >> ITUR_BT_601_SHIFT);
                row2[3]      = uchar(0xff);

                int y11 = std::max(0, int(y2[i + 1]) - 16) * ITUR_BT_601_CY;
                row2[6-bIdx] = saturate_cast<uchar>((y11 + ruv) >> ITUR_BT_601_SHIFT);
                row2[5]      = saturate_cast<uchar>((y11 + guv) >> ITUR_BT_601_SHIFT);
                row2[4+bIdx] = saturate_cast<uchar>((y11 + buv) >> ITUR_BT_601_SHIFT);
                row2[7]      = uchar(0xff);
            }
        }
    }
};

template<int bIdx>
struct YUV420p2RGB888Invoker : ParallelLoopBody
{
    Mat* dst;
    const uchar* my1, *mu, *mv;
    int width, stride;
    int ustepIdx, vstepIdx;

    YUV420p2RGB888Invoker(Mat* _dst, int _stride, const uchar* _y1, const uchar* _u, const uchar* _v, int _ustepIdx, int _vstepIdx)
        : dst(_dst), my1(_y1), mu(_u), mv(_v), width(_dst->cols), stride(_stride), ustepIdx(_ustepIdx), vstepIdx(_vstepIdx) {}

    void operator()(const Range& range) const
    {
        const int rangeBegin = range.start * 2;
        const int rangeEnd = range.end * 2;

        size_t uvsteps[2] = {width/2, stride - width/2};
        int usIdx = ustepIdx, vsIdx = vstepIdx;

        const uchar* y1 = my1 + rangeBegin * stride;
        const uchar* u1 = mu + (range.start / 2) * stride;
        const uchar* v1 = mv + (range.start / 2) * stride;

        if(range.start % 2 == 1)
        {
            u1 += uvsteps[(usIdx++) & 1];
            v1 += uvsteps[(vsIdx++) & 1];
        }

        for (int j = rangeBegin; j < rangeEnd; j += 2, y1 += stride * 2, u1 += uvsteps[(usIdx++) & 1], v1 += uvsteps[(vsIdx++) & 1])
        {
            uchar* row1 = dst->ptr<uchar>(j);
            uchar* row2 = dst->ptr<uchar>(j + 1);
            const uchar* y2 = y1 + stride;

            for (int i = 0; i < width / 2; i += 1, row1 += 6, row2 += 6)
            {
                int u = int(u1[i]) - 128;
                int v = int(v1[i]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(y1[2 * i]) - 16) * ITUR_BT_601_CY;
                row1[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row1[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row1[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);

                int y01 = std::max(0, int(y1[2 * i + 1]) - 16) * ITUR_BT_601_CY;
                row1[5-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row1[4]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row1[3+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);

                int y10 = std::max(0, int(y2[2 * i]) - 16) * ITUR_BT_601_CY;
                row2[2-bIdx] = saturate_cast<uchar>((y10 + ruv) >> ITUR_BT_601_SHIFT);
                row2[1]      = saturate_cast<uchar>((y10 + guv) >> ITUR_BT_601_SHIFT);
                row2[bIdx]   = saturate_cast<uchar>((y10 + buv) >> ITUR_BT_601_SHIFT);

                int y11 = std::max(0, int(y2[2 * i + 1]) - 16) * ITUR_BT_601_CY;
                row2[5-bIdx] = saturate_cast<uchar>((y11 + ruv) >> ITUR_BT_601_SHIFT);
                row2[4]      = saturate_cast<uchar>((y11 + guv) >> ITUR_BT_601_SHIFT);
                row2[3+bIdx] = saturate_cast<uchar>((y11 + buv) >> ITUR_BT_601_SHIFT);
            }
        }
    }
};

template<int bIdx>
struct YUV420p2RGBA8888Invoker : ParallelLoopBody
{
    Mat* dst;
    const uchar* my1, *mu, *mv;
    int width, stride;
    int ustepIdx, vstepIdx;

    YUV420p2RGBA8888Invoker(Mat* _dst, int _stride, const uchar* _y1, const uchar* _u, const uchar* _v, int _ustepIdx, int _vstepIdx)
        : dst(_dst), my1(_y1), mu(_u), mv(_v), width(_dst->cols), stride(_stride), ustepIdx(_ustepIdx), vstepIdx(_vstepIdx) {}

    void operator()(const Range& range) const
    {
        int rangeBegin = range.start * 2;
        int rangeEnd = range.end * 2;

        size_t uvsteps[2] = {width/2, stride - width/2};
        int usIdx = ustepIdx, vsIdx = vstepIdx;

        const uchar* y1 = my1 + rangeBegin * stride;
        const uchar* u1 = mu + (range.start / 2) * stride;
        const uchar* v1 = mv + (range.start / 2) * stride;

        if(range.start % 2 == 1)
        {
            u1 += uvsteps[(usIdx++) & 1];
            v1 += uvsteps[(vsIdx++) & 1];
        }

        for (int j = rangeBegin; j < rangeEnd; j += 2, y1 += stride * 2, u1 += uvsteps[(usIdx++) & 1], v1 += uvsteps[(vsIdx++) & 1])
        {
            uchar* row1 = dst->ptr<uchar>(j);
            uchar* row2 = dst->ptr<uchar>(j + 1);
            const uchar* y2 = y1 + stride;

            for (int i = 0; i < width / 2; i += 1, row1 += 8, row2 += 8)
            {
                int u = int(u1[i]) - 128;
                int v = int(v1[i]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(y1[2 * i]) - 16) * ITUR_BT_601_CY;
                row1[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row1[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row1[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);
                row1[3]      = uchar(0xff);

                int y01 = std::max(0, int(y1[2 * i + 1]) - 16) * ITUR_BT_601_CY;
                row1[6-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row1[5]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row1[4+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);
                row1[7]      = uchar(0xff);

                int y10 = std::max(0, int(y2[2 * i]) - 16) * ITUR_BT_601_CY;
                row2[2-bIdx] = saturate_cast<uchar>((y10 + ruv) >> ITUR_BT_601_SHIFT);
                row2[1]      = saturate_cast<uchar>((y10 + guv) >> ITUR_BT_601_SHIFT);
                row2[bIdx]   = saturate_cast<uchar>((y10 + buv) >> ITUR_BT_601_SHIFT);
                row2[3]      = uchar(0xff);

                int y11 = std::max(0, int(y2[2 * i + 1]) - 16) * ITUR_BT_601_CY;
                row2[6-bIdx] = saturate_cast<uchar>((y11 + ruv) >> ITUR_BT_601_SHIFT);
                row2[5]      = saturate_cast<uchar>((y11 + guv) >> ITUR_BT_601_SHIFT);
                row2[4+bIdx] = saturate_cast<uchar>((y11 + buv) >> ITUR_BT_601_SHIFT);
                row2[7]      = uchar(0xff);
            }
        }
    }
};

#define MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION (320*240)

template<int bIdx, int uIdx>
inline void cvtYUV420sp2RGB(Mat& _dst, int _stride, const uchar* _y1, const uchar* _uv)
{
    YUV420sp2RGB888Invoker<bIdx, uIdx> converter(&_dst, _stride, _y1,  _uv);
    if (_dst.total() >= MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION)
        parallel_for_(Range(0, _dst.rows/2), converter);
    else
        converter(Range(0, _dst.rows/2));
}

template<int bIdx, int uIdx>
inline void cvtYUV420sp2RGBA(Mat& _dst, int _stride, const uchar* _y1, const uchar* _uv)
{
    YUV420sp2RGBA8888Invoker<bIdx, uIdx> converter(&_dst, _stride, _y1,  _uv);
    if (_dst.total() >= MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION)
        parallel_for_(Range(0, _dst.rows/2), converter);
    else
        converter(Range(0, _dst.rows/2));
}

template<int bIdx>
inline void cvtYUV420p2RGB(Mat& _dst, int _stride, const uchar* _y1, const uchar* _u, const uchar* _v, int ustepIdx, int vstepIdx)
{
    YUV420p2RGB888Invoker<bIdx> converter(&_dst, _stride, _y1,  _u, _v, ustepIdx, vstepIdx);
    if (_dst.total() >= MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION)
        parallel_for_(Range(0, _dst.rows/2), converter);
    else
        converter(Range(0, _dst.rows/2));
}

template<int bIdx>
inline void cvtYUV420p2RGBA(Mat& _dst, int _stride, const uchar* _y1, const uchar* _u, const uchar* _v, int ustepIdx, int vstepIdx)
{
    YUV420p2RGBA8888Invoker<bIdx> converter(&_dst, _stride, _y1,  _u, _v, ustepIdx, vstepIdx);
    if (_dst.total() >= MIN_SIZE_FOR_PARALLEL_YUV420_CONVERSION)
        parallel_for_(Range(0, _dst.rows/2), converter);
    else
        converter(Range(0, _dst.rows/2));
}

///////////////////////////////////// RGB -> YUV420p /////////////////////////////////////

template<int bIdx>
struct RGB888toYUV420pInvoker: public ParallelLoopBody
{
    RGB888toYUV420pInvoker( const Mat& src, Mat* dst, const int uIdx )
        : src_(src),
          dst_(dst),
          uIdx_(uIdx) { }

    void operator()(const Range& rowRange) const
    {
        const int w = src_.cols;
        const int h = src_.rows;

        const int cn = src_.channels();
        for( int i = rowRange.start; i < rowRange.end; i++ )
        {
            const uchar* row0 = src_.ptr<uchar>(2 * i);
            const uchar* row1 = src_.ptr<uchar>(2 * i + 1);

            uchar* y = dst_->ptr<uchar>(2*i);
            uchar* u = dst_->ptr<uchar>(h + i/2) + (i % 2) * (w/2);
            uchar* v = dst_->ptr<uchar>(h + (i + h/2)/2) + ((i + h/2) % 2) * (w/2);
            if( uIdx_ == 2 ) std::swap(u, v);

            for( int j = 0, k = 0; j < w * cn; j += 2 * cn, k++ )
            {
                int r00 = row0[2-bIdx + j];      int g00 = row0[1 + j];      int b00 = row0[bIdx + j];
                int r01 = row0[2-bIdx + cn + j]; int g01 = row0[1 + cn + j]; int b01 = row0[bIdx + cn + j];
                int r10 = row1[2-bIdx + j];      int g10 = row1[1 + j];      int b10 = row1[bIdx + j];
                int r11 = row1[2-bIdx + cn + j]; int g11 = row1[1 + cn + j]; int b11 = row1[bIdx + cn + j];

                const int shifted16 = (16 << ITUR_BT_601_SHIFT);
                const int halfShift = (1 << (ITUR_BT_601_SHIFT - 1));
                int y00 = ITUR_BT_601_CRY * r00 + ITUR_BT_601_CGY * g00 + ITUR_BT_601_CBY * b00 + halfShift + shifted16;
                int y01 = ITUR_BT_601_CRY * r01 + ITUR_BT_601_CGY * g01 + ITUR_BT_601_CBY * b01 + halfShift + shifted16;
                int y10 = ITUR_BT_601_CRY * r10 + ITUR_BT_601_CGY * g10 + ITUR_BT_601_CBY * b10 + halfShift + shifted16;
                int y11 = ITUR_BT_601_CRY * r11 + ITUR_BT_601_CGY * g11 + ITUR_BT_601_CBY * b11 + halfShift + shifted16;

                y[2*k + 0]            = saturate_cast<uchar>(y00 >> ITUR_BT_601_SHIFT);
                y[2*k + 1]            = saturate_cast<uchar>(y01 >> ITUR_BT_601_SHIFT);
                y[2*k + dst_->step + 0] = saturate_cast<uchar>(y10 >> ITUR_BT_601_SHIFT);
                y[2*k + dst_->step + 1] = saturate_cast<uchar>(y11 >> ITUR_BT_601_SHIFT);

                const int shifted128 = (128 << ITUR_BT_601_SHIFT);
                int u00 = ITUR_BT_601_CRU * r00 + ITUR_BT_601_CGU * g00 + ITUR_BT_601_CBU * b00 + halfShift + shifted128;
                int v00 = ITUR_BT_601_CBU * r00 + ITUR_BT_601_CGV * g00 + ITUR_BT_601_CBV * b00 + halfShift + shifted128;

                u[k] = saturate_cast<uchar>(u00 >> ITUR_BT_601_SHIFT);
                v[k] = saturate_cast<uchar>(v00 >> ITUR_BT_601_SHIFT);
            }
        }
    }

    static bool isFit( const Mat& src )
    {
        return (src.total() >= 320*240);
    }

private:
    RGB888toYUV420pInvoker& operator=(const RGB888toYUV420pInvoker&);

    const Mat& src_;
    Mat* const dst_;
    const int uIdx_;
};

template<int bIdx, int uIdx>
static void cvtRGBtoYUV420p(const Mat& src, Mat& dst)
{
    RGB888toYUV420pInvoker<bIdx> colorConverter(src, &dst, uIdx);
    if( RGB888toYUV420pInvoker<bIdx>::isFit(src) )
        parallel_for_(Range(0, src.rows/2), colorConverter);
    else
        colorConverter(Range(0, src.rows/2));
}

///////////////////////////////////// YUV422 -> RGB /////////////////////////////////////

template<int bIdx, int uIdx, int yIdx>
struct YUV422toRGB888Invoker : ParallelLoopBody
{
    Mat* dst;
    const uchar* src;
    int width, stride;

    YUV422toRGB888Invoker(Mat* _dst, int _stride, const uchar* _yuv)
        : dst(_dst), src(_yuv), width(_dst->cols), stride(_stride) {}

    void operator()(const Range& range) const
    {
        int rangeBegin = range.start;
        int rangeEnd = range.end;

        const int uidx = 1 - yIdx + uIdx * 2;
        const int vidx = (2 + uidx) % 4;
        const uchar* yuv_src = src + rangeBegin * stride;

        for (int j = rangeBegin; j < rangeEnd; j++, yuv_src += stride)
        {
            uchar* row = dst->ptr<uchar>(j);

            for (int i = 0; i < 2 * width; i += 4, row += 6)
            {
                int u = int(yuv_src[i + uidx]) - 128;
                int v = int(yuv_src[i + vidx]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(yuv_src[i + yIdx]) - 16) * ITUR_BT_601_CY;
                row[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);

                int y01 = std::max(0, int(yuv_src[i + yIdx + 2]) - 16) * ITUR_BT_601_CY;
                row[5-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row[4]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row[3+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);
            }
        }
    }
};

template<int bIdx, int uIdx, int yIdx>
struct YUV422toRGBA8888Invoker : ParallelLoopBody
{
    Mat* dst;
    const uchar* src;
    int width, stride;

    YUV422toRGBA8888Invoker(Mat* _dst, int _stride, const uchar* _yuv)
        : dst(_dst), src(_yuv), width(_dst->cols), stride(_stride) {}

    void operator()(const Range& range) const
    {
        int rangeBegin = range.start;
        int rangeEnd = range.end;

        const int uidx = 1 - yIdx + uIdx * 2;
        const int vidx = (2 + uidx) % 4;
        const uchar* yuv_src = src + rangeBegin * stride;

        for (int j = rangeBegin; j < rangeEnd; j++, yuv_src += stride)
        {
            uchar* row = dst->ptr<uchar>(j);

            for (int i = 0; i < 2 * width; i += 4, row += 8)
            {
                int u = int(yuv_src[i + uidx]) - 128;
                int v = int(yuv_src[i + vidx]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = std::max(0, int(yuv_src[i + yIdx]) - 16) * ITUR_BT_601_CY;
                row[2-bIdx] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row[1]      = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row[bIdx]   = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);
                row[3]      = uchar(0xff);

                int y01 = std::max(0, int(yuv_src[i + yIdx + 2]) - 16) * ITUR_BT_601_CY;
                row[6-bIdx] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row[5]      = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row[4+bIdx] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);
                row[7]      = uchar(0xff);
            }
        }
    }
};

#define MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION (320*240)

template<int bIdx, int uIdx, int yIdx>
inline void cvtYUV422toRGB(Mat& _dst, int _stride, const uchar* _yuv)
{
    YUV422toRGB888Invoker<bIdx, uIdx, yIdx> converter(&_dst, _stride, _yuv);
    if (_dst.total() >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION)
        parallel_for_(Range(0, _dst.rows), converter);
    else
        converter(Range(0, _dst.rows));
}

template<int bIdx, int uIdx, int yIdx>
inline void cvtYUV422toRGBA(Mat& _dst, int _stride, const uchar* _yuv)
{
    YUV422toRGBA8888Invoker<bIdx, uIdx, yIdx> converter(&_dst, _stride, _yuv);
    if (_dst.total() >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION)
        parallel_for_(Range(0, _dst.rows), converter);
    else
        converter(Range(0, _dst.rows));
}

/////////////////////////// RGBA <-> mRGBA (alpha premultiplied) //////////////

template<typename _Tp>
struct RGBA2mRGBA
{
    typedef _Tp channel_type;

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        _Tp max_val  = ColorChannel<_Tp>::max();
        _Tp half_val = ColorChannel<_Tp>::half();
        for( int i = 0; i < n; i++ )
        {
            _Tp v0 = *src++;
            _Tp v1 = *src++;
            _Tp v2 = *src++;
            _Tp v3 = *src++;

            *dst++ = (v0 * v3 + half_val) / max_val;
            *dst++ = (v1 * v3 + half_val) / max_val;
            *dst++ = (v2 * v3 + half_val) / max_val;
            *dst++ = v3;
        }
    }
};


template<typename _Tp>
struct mRGBA2RGBA
{
    typedef _Tp channel_type;

    void operator()(const _Tp* src, _Tp* dst, int n) const
    {
        _Tp max_val = ColorChannel<_Tp>::max();
        for( int i = 0; i < n; i++ )
        {
            _Tp v0 = *src++;
            _Tp v1 = *src++;
            _Tp v2 = *src++;
            _Tp v3 = *src++;
            _Tp v3_half = v3 / 2;

            *dst++ = (v3==0)? 0 : (v0 * max_val + v3_half) / v3;
            *dst++ = (v3==0)? 0 : (v1 * max_val + v3_half) / v3;
            *dst++ = (v3==0)? 0 : (v2 * max_val + v3_half) / v3;
            *dst++ = v3;
        }
    }
};


static bool ocl_cvtColor( InputArray _src, OutputArray _dst, int code, int dcn )
{
    bool ok = true;
    UMat src = _src.getUMat(), dst;
    Size sz = src.size(), dstSz = sz;
    int scn = src.channels(), depth = src.depth(), bidx;
    size_t globalsize[] = { src.cols, src.rows };
    ocl::Kernel k;

    if(depth != CV_8U && depth != CV_16U && depth != CV_32F)
        return false;

    switch (code)
    {
    /*
     case COLOR_BGR2BGRA: case COLOR_RGB2BGRA: case COLOR_BGRA2BGR:
     case COLOR_RGBA2BGR: case COLOR_RGB2BGR: case COLOR_BGRA2RGBA:
     case COLOR_BGR2BGR565: case COLOR_BGR2BGR555: case COLOR_RGB2BGR565: case COLOR_RGB2BGR555:
     case COLOR_BGRA2BGR565: case COLOR_BGRA2BGR555: case COLOR_RGBA2BGR565: case COLOR_RGBA2BGR555:
     case COLOR_BGR5652BGR: case COLOR_BGR5552BGR: case COLOR_BGR5652RGB: case COLOR_BGR5552RGB:
     case COLOR_BGR5652BGRA: case COLOR_BGR5552BGRA: case COLOR_BGR5652RGBA: case COLOR_BGR5552RGBA:
     */
    case COLOR_BGR2GRAY:
    case COLOR_BGRA2GRAY:
    case COLOR_RGB2GRAY:
    case COLOR_RGBA2GRAY:
    {
        CV_Assert(scn == 3 || scn == 4);
        bidx = code == COLOR_BGR2GRAY || code == COLOR_BGRA2GRAY ? 0 : 2;
        dcn = 1;
        k.create("RGB2Gray", ocl::imgproc::cvtcolor_oclsrc,
                 format("-D depth=%d -D scn=%d -D dcn=1 -D bidx=%d", depth, scn, bidx));
        break;
    }
    case COLOR_GRAY2BGR:
    case COLOR_GRAY2BGRA:
    {
        CV_Assert(scn == 1);
        dcn = code == COLOR_GRAY2BGRA ? 4 : 3;
        k.create("Gray2RGB", ocl::imgproc::cvtcolor_oclsrc,
                 format("-D depth=%d -D scn=1 -D dcn=%d", depth, dcn));
        break;
    }
    case COLOR_BGR2YUV:
    case COLOR_RGB2YUV:
    {
        CV_Assert(scn == 3 || scn == 4);
        bidx = code == COLOR_RGB2YUV ? 0 : 2;
        dcn = 3;
        k.create("RGB2YUV", ocl::imgproc::cvtcolor_oclsrc,
                 format("-D depth=%d -D scn=%d -D dcn=3 -D bidx=%d", depth, scn, bidx));
        break;
    }
    case COLOR_YUV2BGR:
    case COLOR_YUV2RGB:
    {
        if(dcn < 0) dcn = 3;
        CV_Assert(dcn == 3 || dcn == 4);
        bidx = code == COLOR_YUV2RGB ? 0 : 2;
        k.create("YUV2RGB", ocl::imgproc::cvtcolor_oclsrc,
                 format("-D depth=%d -D scn=3 -D dcn=%d -D bidx=%d", depth, dcn, bidx));
        break;
    }
    case COLOR_YUV2RGB_NV12:
    case COLOR_YUV2BGR_NV12:
    case COLOR_YUV2RGBA_NV12:
    case COLOR_YUV2BGRA_NV12:
    {
        CV_Assert( scn == 1 );
        CV_Assert( sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U );
        dcn  = code == COLOR_YUV2BGRA_NV12 || code == COLOR_YUV2RGBA_NV12 ? 4 : 3;
        bidx = code == COLOR_YUV2BGRA_NV12 || code == COLOR_YUV2BGR_NV12 ? 0 : 2;

        dstSz = Size(sz.width, sz.height * 2 / 3);
        globalsize[0] = dstSz.height/2;
        globalsize[1] = dstSz.width/2;
        k.create("YUV2RGBA_NV12", ocl::imgproc::cvtcolor_oclsrc,
                 format("-D depth=0 -D scn=1 -D dcn=%d -D bidx=%d", dcn, bidx));
        break;
    }
    case COLOR_BGR2YCrCb:
    case COLOR_RGB2YCrCb:
    {
        CV_Assert(scn == 3 || scn == 4);
        bidx = code == COLOR_BGR2YCrCb ? 0 : 2;
        dcn = 3;
        k.create("RGB2YCrCb", ocl::imgproc::cvtcolor_oclsrc,
                 format("-D depth=%d -D scn=%d -D dcn=3 -D bidx=%d", depth, scn, bidx));
        break;
    }
    case COLOR_YCrCb2BGR:
    case COLOR_YCrCb2RGB:
        break;
    /*
     case COLOR_BGR5652GRAY: case COLOR_BGR5552GRAY:
     case COLOR_GRAY2BGR565: case COLOR_GRAY2BGR555:
     case COLOR_BGR2YCrCb: case COLOR_RGB2YCrCb:
     case COLOR_BGR2XYZ: case COLOR_RGB2XYZ:
     case COLOR_XYZ2BGR: case COLOR_XYZ2RGB:
     case COLOR_BGR2HSV: case COLOR_RGB2HSV: case COLOR_BGR2HSV_FULL: case COLOR_RGB2HSV_FULL:
     case COLOR_BGR2HLS: case COLOR_RGB2HLS: case COLOR_BGR2HLS_FULL: case COLOR_RGB2HLS_FULL:
     case COLOR_HSV2BGR: case COLOR_HSV2RGB: case COLOR_HSV2BGR_FULL: case COLOR_HSV2RGB_FULL:
     case COLOR_HLS2BGR: case COLOR_HLS2RGB: case COLOR_HLS2BGR_FULL: case COLOR_HLS2RGB_FULL:
     */
    default:
        ;
    }

    if( !k.empty() )
    {
        _dst.create(dstSz, CV_MAKETYPE(depth, dcn));
        dst = _dst.getUMat();
        k.args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::WriteOnly(dst));
        ok = k.run(2, globalsize, 0, false);
    }
    return ok;
}

}//namespace cv

//////////////////////////////////////////////////////////////////////////////////////////
//                                   The main function                                  //
//////////////////////////////////////////////////////////////////////////////////////////

void cv::cvtColor( InputArray _src, OutputArray _dst, int code, int dcn )
{
    bool use_opencl = ocl::useOpenCL() && _dst.kind() == _InputArray::UMAT;
    int stype = _src.type();
    int scn = CV_MAT_CN(stype), depth = CV_MAT_DEPTH(stype), bidx;

    if( use_opencl && ocl_cvtColor(_src, _dst, code, dcn) )
        return;

    Mat src = _src.getMat(), dst;
    Size sz = src.size();

    CV_Assert( depth == CV_8U || depth == CV_16U || depth == CV_32F );

    switch( code )
    {
        case CV_BGR2BGRA: case CV_RGB2BGRA: case CV_BGRA2BGR:
        case CV_RGBA2BGR: case CV_RGB2BGR: case CV_BGRA2RGBA:
            CV_Assert( scn == 3 || scn == 4 );
            dcn = code == CV_BGR2BGRA || code == CV_RGB2BGRA || code == CV_BGRA2RGBA ? 4 : 3;
            bidx = code == CV_BGR2BGRA || code == CV_BGRA2BGR ? 0 : 2;

            _dst.create( sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
            if( code == CV_BGR2BGRA || code == CV_RGB2RGBA)
            {
                if ( CvtColorIPPLoop(src, dst, IPPReorderFunctor(ippiSwapChannelsC3C4RTab[depth], 0, 1, 2)) )
                    return;
            }
            else if( code == CV_BGRA2BGR )
            {
                if ( CvtColorIPPLoop(src, dst, IPPGeneralFunctor(ippiCopyAC4C3RTab[depth])) )
                    return;
            }
            else if( code == CV_BGR2RGBA )
            {
                if( CvtColorIPPLoop(src, dst, IPPReorderFunctor(ippiSwapChannelsC3C4RTab[depth], 2, 1, 0)) )
                    return;
            }
            else if( code == CV_RGBA2BGR )
            {
                if( CvtColorIPPLoop(src, dst, IPPReorderFunctor(ippiSwapChannelsC4C3RTab[depth], 2, 1, 0)) )
                    return;
            }
            else if( code == CV_RGB2BGR )
            {
                if( CvtColorIPPLoopCopy(src, dst, IPPReorderFunctor(ippiSwapChannelsC3RTab[depth], 2, 1, 0)) )
                    return;
            }
            else if( code == CV_RGBA2BGRA )
            {
                if( CvtColorIPPLoopCopy(src, dst, IPPReorderFunctor(ippiSwapChannelsC4RTab[depth], 2, 1, 0)) )
                    return;
            }
#endif

            if( depth == CV_8U )
            {
#ifdef HAVE_TEGRA_OPTIMIZATION
                if(!tegra::cvtBGR2RGB(src, dst, bidx))
#endif
                    CvtColorLoop(src, dst, RGB2RGB<uchar>(scn, dcn, bidx));
            }
            else if( depth == CV_16U )
                CvtColorLoop(src, dst, RGB2RGB<ushort>(scn, dcn, bidx));
            else
                CvtColorLoop(src, dst, RGB2RGB<float>(scn, dcn, bidx));
            break;

        case CV_BGR2BGR565: case CV_BGR2BGR555: case CV_RGB2BGR565: case CV_RGB2BGR555:
        case CV_BGRA2BGR565: case CV_BGRA2BGR555: case CV_RGBA2BGR565: case CV_RGBA2BGR555:
            CV_Assert( (scn == 3 || scn == 4) && depth == CV_8U );
            _dst.create(sz, CV_8UC2);
            dst = _dst.getMat();

#ifdef HAVE_TEGRA_OPTIMIZATION
            if(code == CV_BGR2BGR565 || code == CV_BGRA2BGR565 || code == CV_RGB2BGR565  || code == CV_RGBA2BGR565)
                if(tegra::cvtRGB2RGB565(src, dst, code == CV_RGB2BGR565 || code == CV_RGBA2BGR565 ? 0 : 2))
                    break;
#endif

            CvtColorLoop(src, dst, RGB2RGB5x5(scn,
                      code == CV_BGR2BGR565 || code == CV_BGR2BGR555 ||
                      code == CV_BGRA2BGR565 || code == CV_BGRA2BGR555 ? 0 : 2,
                      code == CV_BGR2BGR565 || code == CV_RGB2BGR565 ||
                      code == CV_BGRA2BGR565 || code == CV_RGBA2BGR565 ? 6 : 5 // green bits
                                              ));
            break;

        case CV_BGR5652BGR: case CV_BGR5552BGR: case CV_BGR5652RGB: case CV_BGR5552RGB:
        case CV_BGR5652BGRA: case CV_BGR5552BGRA: case CV_BGR5652RGBA: case CV_BGR5552RGBA:
            if(dcn <= 0) dcn = (code==CV_BGR5652BGRA || code==CV_BGR5552BGRA || code==CV_BGR5652RGBA || code==CV_BGR5552RGBA) ? 4 : 3;
            CV_Assert( (dcn == 3 || dcn == 4) && scn == 2 && depth == CV_8U );
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();

            CvtColorLoop(src, dst, RGB5x52RGB(dcn,
                      code == CV_BGR5652BGR || code == CV_BGR5552BGR ||
                      code == CV_BGR5652BGRA || code == CV_BGR5552BGRA ? 0 : 2, // blue idx
                      code == CV_BGR5652BGR || code == CV_BGR5652RGB ||
                      code == CV_BGR5652BGRA || code == CV_BGR5652RGBA ? 6 : 5 // green bits
                      ));
            break;

        case CV_BGR2GRAY: case CV_BGRA2GRAY: case CV_RGB2GRAY: case CV_RGBA2GRAY:
            CV_Assert( scn == 3 || scn == 4 );
            _dst.create(sz, CV_MAKETYPE(depth, 1));
            dst = _dst.getMat();

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
            if( code == CV_BGR2GRAY )
            {
                if( CvtColorIPPLoop(src, dst, IPPColor2GrayFunctor(ippiColor2GrayC3Tab[depth])) )
                    return;
            }
            else if( code == CV_RGB2GRAY )
            {
                if( CvtColorIPPLoop(src, dst, IPPGeneralFunctor(ippiRGB2GrayC3Tab[depth])) )
                    return;
            }
            else if( code == CV_BGRA2GRAY )
            {
                if( CvtColorIPPLoop(src, dst, IPPColor2GrayFunctor(ippiColor2GrayC4Tab[depth])) )
                    return;
            }
            else if( code == CV_RGBA2GRAY )
            {
                if( CvtColorIPPLoop(src, dst, IPPGeneralFunctor(ippiRGB2GrayC4Tab[depth])) )
                    return;
            }
#endif

            bidx = code == CV_BGR2GRAY || code == CV_BGRA2GRAY ? 0 : 2;

            if( depth == CV_8U )
            {
#ifdef HAVE_TEGRA_OPTIMIZATION
                if(!tegra::cvtRGB2Gray(src, dst, bidx))
#endif
                CvtColorLoop(src, dst, RGB2Gray<uchar>(scn, bidx, 0));
            }
            else if( depth == CV_16U )
                CvtColorLoop(src, dst, RGB2Gray<ushort>(scn, bidx, 0));
            else
                CvtColorLoop(src, dst, RGB2Gray<float>(scn, bidx, 0));
            break;

        case CV_BGR5652GRAY: case CV_BGR5552GRAY:
            CV_Assert( scn == 2 && depth == CV_8U );
            _dst.create(sz, CV_8UC1);
            dst = _dst.getMat();

            CvtColorLoop(src, dst, RGB5x52Gray(code == CV_BGR5652GRAY ? 6 : 5));
            break;

        case CV_GRAY2BGR: case CV_GRAY2BGRA:
            if( dcn <= 0 ) dcn = (code==CV_GRAY2BGRA) ? 4 : 3;
            CV_Assert( scn == 1 && (dcn == 3 || dcn == 4));
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
            if( code == CV_GRAY2BGR )
            {
                if( CvtColorIPPLoop(src, dst, IPPGray2BGRFunctor(ippiCopyP3C3RTab[depth])) )
                    return;
            }
            else if( code == CV_GRAY2BGRA )
            {
                if( CvtColorIPPLoop(src, dst, IPPGray2BGRAFunctor(ippiCopyP3C3RTab[depth], ippiSwapChannelsC3C4RTab[depth], depth)) )
                    return;
            }
#endif


            if( depth == CV_8U )
            {
#ifdef HAVE_TEGRA_OPTIMIZATION
                if(!tegra::cvtGray2RGB(src, dst))
#endif
                CvtColorLoop(src, dst, Gray2RGB<uchar>(dcn));
            }
            else if( depth == CV_16U )
                CvtColorLoop(src, dst, Gray2RGB<ushort>(dcn));
            else
                CvtColorLoop(src, dst, Gray2RGB<float>(dcn));
            break;

        case CV_GRAY2BGR565: case CV_GRAY2BGR555:
            CV_Assert( scn == 1 && depth == CV_8U );
            _dst.create(sz, CV_8UC2);
            dst = _dst.getMat();

            CvtColorLoop(src, dst, Gray2RGB5x5(code == CV_GRAY2BGR565 ? 6 : 5));
            break;

        case CV_BGR2YCrCb: case CV_RGB2YCrCb:
        case CV_BGR2YUV: case CV_RGB2YUV:
            {
            CV_Assert( scn == 3 || scn == 4 );
            bidx = code == CV_BGR2YCrCb || code == CV_BGR2YUV ? 0 : 2;
            static const float yuv_f[] = { 0.114f, 0.587f, 0.299f, 0.492f, 0.877f };
            static const int yuv_i[] = { B2Y, G2Y, R2Y, 8061, 14369 };
            const float* coeffs_f = code == CV_BGR2YCrCb || code == CV_RGB2YCrCb ? 0 : yuv_f;
            const int* coeffs_i = code == CV_BGR2YCrCb || code == CV_RGB2YCrCb ? 0 : yuv_i;

            _dst.create(sz, CV_MAKETYPE(depth, 3));
            dst = _dst.getMat();

            if( depth == CV_8U )
            {
#ifdef HAVE_TEGRA_OPTIMIZATION
                if((code == CV_RGB2YCrCb || code == CV_BGR2YCrCb) && tegra::cvtRGB2YCrCb(src, dst, bidx))
                    break;
#endif
                CvtColorLoop(src, dst, RGB2YCrCb_i<uchar>(scn, bidx, coeffs_i));
            }
            else if( depth == CV_16U )
                CvtColorLoop(src, dst, RGB2YCrCb_i<ushort>(scn, bidx, coeffs_i));
            else
                CvtColorLoop(src, dst, RGB2YCrCb_f<float>(scn, bidx, coeffs_f));
            }
            break;

        case CV_YCrCb2BGR: case CV_YCrCb2RGB:
        case CV_YUV2BGR: case CV_YUV2RGB:
            {
            if( dcn <= 0 ) dcn = 3;
            CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) );
            bidx = code == CV_YCrCb2BGR || code == CV_YUV2BGR ? 0 : 2;
            static const float yuv_f[] = { 2.032f, -0.395f, -0.581f, 1.140f };
            static const int yuv_i[] = { 33292, -6472, -9519, 18678 };
            const float* coeffs_f = code == CV_YCrCb2BGR || code == CV_YCrCb2RGB ? 0 : yuv_f;
            const int* coeffs_i = code == CV_YCrCb2BGR || code == CV_YCrCb2RGB ? 0 : yuv_i;

            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();

            if( depth == CV_8U )
                CvtColorLoop(src, dst, YCrCb2RGB_i<uchar>(dcn, bidx, coeffs_i));
            else if( depth == CV_16U )
                CvtColorLoop(src, dst, YCrCb2RGB_i<ushort>(dcn, bidx, coeffs_i));
            else
                CvtColorLoop(src, dst, YCrCb2RGB_f<float>(dcn, bidx, coeffs_f));
            }
            break;

        case CV_BGR2XYZ: case CV_RGB2XYZ:
            CV_Assert( scn == 3 || scn == 4 );
            bidx = code == CV_BGR2XYZ ? 0 : 2;

            _dst.create(sz, CV_MAKETYPE(depth, 3));
            dst = _dst.getMat();

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
            if( code == CV_BGR2XYZ && scn == 3 )
            {
                if( CvtColorIPPLoopCopy(src, dst, IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth], ippiRGB2XYZTab[depth], 2, 1, 0, depth)) )
                    return;
            }
            else if( code == CV_BGR2XYZ && scn == 4 )
            {
                if( CvtColorIPPLoop(src, dst, IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2XYZTab[depth], 2, 1, 0, depth)) )
                    return;
            }
            else if( code == CV_RGB2XYZ && scn == 3 )
            {
                if( CvtColorIPPLoopCopy(src, dst, IPPGeneralFunctor(ippiRGB2XYZTab[depth])) )
                    return;
            }
            else if( code == CV_RGB2XYZ && scn == 4 )
            {
                if( CvtColorIPPLoop(src, dst, IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2XYZTab[depth], 0, 1, 2, depth)) )
                    return;
            }
#endif

            if( depth == CV_8U )
                CvtColorLoop(src, dst, RGB2XYZ_i<uchar>(scn, bidx, 0));
            else if( depth == CV_16U )
                CvtColorLoop(src, dst, RGB2XYZ_i<ushort>(scn, bidx, 0));
            else
                CvtColorLoop(src, dst, RGB2XYZ_f<float>(scn, bidx, 0));
            break;

        case CV_XYZ2BGR: case CV_XYZ2RGB:
            if( dcn <= 0 ) dcn = 3;
            CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) );
            bidx = code == CV_XYZ2BGR ? 0 : 2;

            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
            if( code == CV_XYZ2BGR && dcn == 3 )
            {
                if( CvtColorIPPLoopCopy(src, dst, IPPGeneralReorderFunctor(ippiXYZ2RGBTab[depth], ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                    return;
            }
            else if( code == CV_XYZ2BGR && dcn == 4 )
            {
                if( CvtColorIPPLoop(src, dst, IPPGeneralReorderFunctor(ippiXYZ2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                    return;
            }
            if( code == CV_XYZ2RGB && dcn == 3 )
            {
                if( CvtColorIPPLoopCopy(src, dst, IPPGeneralFunctor(ippiXYZ2RGBTab[depth])) )
                    return;
            }
            else if( code == CV_XYZ2RGB && dcn == 4 )
            {
                if( CvtColorIPPLoop(src, dst, IPPGeneralReorderFunctor(ippiXYZ2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                    return;
            }
#endif

            if( depth == CV_8U )
                CvtColorLoop(src, dst, XYZ2RGB_i<uchar>(dcn, bidx, 0));
            else if( depth == CV_16U )
                CvtColorLoop(src, dst, XYZ2RGB_i<ushort>(dcn, bidx, 0));
            else
                CvtColorLoop(src, dst, XYZ2RGB_f<float>(dcn, bidx, 0));
            break;

        case CV_BGR2HSV: case CV_RGB2HSV: case CV_BGR2HSV_FULL: case CV_RGB2HSV_FULL:
        case CV_BGR2HLS: case CV_RGB2HLS: case CV_BGR2HLS_FULL: case CV_RGB2HLS_FULL:
            {
            CV_Assert( (scn == 3 || scn == 4) && (depth == CV_8U || depth == CV_32F) );
            bidx = code == CV_BGR2HSV || code == CV_BGR2HLS ||
                code == CV_BGR2HSV_FULL || code == CV_BGR2HLS_FULL ? 0 : 2;
            int hrange = depth == CV_32F ? 360 : code == CV_BGR2HSV || code == CV_RGB2HSV ||
                code == CV_BGR2HLS || code == CV_RGB2HLS ? 180 : 256;

            _dst.create(sz, CV_MAKETYPE(depth, 3));
            dst = _dst.getMat();

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
            if( depth == CV_8U || depth == CV_16U )
            {
                if( code == CV_BGR2HSV_FULL && scn == 3 )
                {
                    if( CvtColorIPPLoopCopy(src, dst, IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth], ippiRGB2HSVTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( code == CV_BGR2HSV_FULL && scn == 4 )
                {
                    if( CvtColorIPPLoop(src, dst, IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HSVTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( code == CV_RGB2HSV_FULL && scn == 3 )
                {
                    if( CvtColorIPPLoopCopy(src, dst, IPPGeneralFunctor(ippiRGB2HSVTab[depth])) )
                        return;
                }
                else if( code == CV_RGB2HSV_FULL && scn == 4 )
                {
                    if( CvtColorIPPLoop(src, dst, IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HSVTab[depth], 0, 1, 2, depth)) )
                        return;
                }
                else if( code == CV_BGR2HLS_FULL && scn == 3 )
                {
                    if( CvtColorIPPLoopCopy(src, dst, IPPReorderGeneralFunctor(ippiSwapChannelsC3RTab[depth], ippiRGB2HLSTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( code == CV_BGR2HLS_FULL && scn == 4 )
                {
                    if( CvtColorIPPLoop(src, dst, IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HLSTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( code == CV_RGB2HLS_FULL && scn == 3 )
                {
                    if( CvtColorIPPLoopCopy(src, dst, IPPGeneralFunctor(ippiRGB2HLSTab[depth])) )
                        return;
                }
                else if( code == CV_RGB2HLS_FULL && scn == 4 )
                {
                    if( CvtColorIPPLoop(src, dst, IPPReorderGeneralFunctor(ippiSwapChannelsC4C3RTab[depth], ippiRGB2HLSTab[depth], 0, 1, 2, depth)) )
                        return;
                }
            }
#endif

            if( code == CV_BGR2HSV || code == CV_RGB2HSV ||
                code == CV_BGR2HSV_FULL || code == CV_RGB2HSV_FULL )
            {
#ifdef HAVE_TEGRA_OPTIMIZATION
                if(tegra::cvtRGB2HSV(src, dst, bidx, hrange))
                    break;
#endif
                if( depth == CV_8U )
                    CvtColorLoop(src, dst, RGB2HSV_b(scn, bidx, hrange));
                else
                    CvtColorLoop(src, dst, RGB2HSV_f(scn, bidx, (float)hrange));
            }
            else
            {
                if( depth == CV_8U )
                    CvtColorLoop(src, dst, RGB2HLS_b(scn, bidx, hrange));
                else
                    CvtColorLoop(src, dst, RGB2HLS_f(scn, bidx, (float)hrange));
            }
            }
            break;

        case CV_HSV2BGR: case CV_HSV2RGB: case CV_HSV2BGR_FULL: case CV_HSV2RGB_FULL:
        case CV_HLS2BGR: case CV_HLS2RGB: case CV_HLS2BGR_FULL: case CV_HLS2RGB_FULL:
            {
            if( dcn <= 0 ) dcn = 3;
            CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) && (depth == CV_8U || depth == CV_32F) );
            bidx = code == CV_HSV2BGR || code == CV_HLS2BGR ||
                code == CV_HSV2BGR_FULL || code == CV_HLS2BGR_FULL ? 0 : 2;
            int hrange = depth == CV_32F ? 360 : code == CV_HSV2BGR || code == CV_HSV2RGB ||
                code == CV_HLS2BGR || code == CV_HLS2RGB ? 180 : 255;

            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
            if( depth == CV_8U || depth == CV_16U )
            {
                if( code == CV_HSV2BGR_FULL && dcn == 3 )
                {
                    if( CvtColorIPPLoopCopy(src, dst, IPPGeneralReorderFunctor(ippiHSV2RGBTab[depth], ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( code == CV_HSV2BGR_FULL && dcn == 4 )
                {
                    if( CvtColorIPPLoop(src, dst, IPPGeneralReorderFunctor(ippiHSV2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( code == CV_HSV2RGB_FULL && dcn == 3 )
                {
                    if( CvtColorIPPLoopCopy(src, dst, IPPGeneralFunctor(ippiHSV2RGBTab[depth])) )
                        return;
                }
                else if( code == CV_HSV2RGB_FULL && dcn == 4 )
                {
                    if( CvtColorIPPLoop(src, dst, IPPGeneralReorderFunctor(ippiHSV2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                        return;
                }
                else if( code == CV_HLS2BGR_FULL && dcn == 3 )
                {
                    if( CvtColorIPPLoopCopy(src, dst, IPPGeneralReorderFunctor(ippiHLS2RGBTab[depth], ippiSwapChannelsC3RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( code == CV_HLS2BGR_FULL && dcn == 4 )
                {
                    if( CvtColorIPPLoop(src, dst, IPPGeneralReorderFunctor(ippiHLS2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 2, 1, 0, depth)) )
                        return;
                }
                else if( code == CV_HLS2RGB_FULL && dcn == 3 )
                {
                    if( CvtColorIPPLoopCopy(src, dst, IPPGeneralFunctor(ippiHLS2RGBTab[depth])) )
                        return;
                }
                else if( code == CV_HLS2RGB_FULL && dcn == 4 )
                {
                    if( CvtColorIPPLoop(src, dst, IPPGeneralReorderFunctor(ippiHLS2RGBTab[depth], ippiSwapChannelsC3C4RTab[depth], 0, 1, 2, depth)) )
                        return;
                }
            }
#endif

            if( code == CV_HSV2BGR || code == CV_HSV2RGB ||
                code == CV_HSV2BGR_FULL || code == CV_HSV2RGB_FULL )
            {
                if( depth == CV_8U )
                    CvtColorLoop(src, dst, HSV2RGB_b(dcn, bidx, hrange));
                else
                    CvtColorLoop(src, dst, HSV2RGB_f(dcn, bidx, (float)hrange));
            }
            else
            {
                if( depth == CV_8U )
                    CvtColorLoop(src, dst, HLS2RGB_b(dcn, bidx, hrange));
                else
                    CvtColorLoop(src, dst, HLS2RGB_f(dcn, bidx, (float)hrange));
            }
            }
            break;

        case CV_BGR2Lab: case CV_RGB2Lab: case CV_LBGR2Lab: case CV_LRGB2Lab:
        case CV_BGR2Luv: case CV_RGB2Luv: case CV_LBGR2Luv: case CV_LRGB2Luv:
            {
            CV_Assert( (scn == 3 || scn == 4) && (depth == CV_8U || depth == CV_32F) );
            bidx = code == CV_BGR2Lab || code == CV_BGR2Luv ||
                   code == CV_LBGR2Lab || code == CV_LBGR2Luv ? 0 : 2;
            bool srgb = code == CV_BGR2Lab || code == CV_RGB2Lab ||
                        code == CV_BGR2Luv || code == CV_RGB2Luv;

            _dst.create(sz, CV_MAKETYPE(depth, 3));
            dst = _dst.getMat();

            if( code == CV_BGR2Lab || code == CV_RGB2Lab ||
                code == CV_LBGR2Lab || code == CV_LRGB2Lab )
            {
                if( depth == CV_8U )
                    CvtColorLoop(src, dst, RGB2Lab_b(scn, bidx, 0, 0, srgb));
                else
                    CvtColorLoop(src, dst, RGB2Lab_f(scn, bidx, 0, 0, srgb));
            }
            else
            {
                if( depth == CV_8U )
                    CvtColorLoop(src, dst, RGB2Luv_b(scn, bidx, 0, 0, srgb));
                else
                    CvtColorLoop(src, dst, RGB2Luv_f(scn, bidx, 0, 0, srgb));
            }
            }
            break;

        case CV_Lab2BGR: case CV_Lab2RGB: case CV_Lab2LBGR: case CV_Lab2LRGB:
        case CV_Luv2BGR: case CV_Luv2RGB: case CV_Luv2LBGR: case CV_Luv2LRGB:
            {
            if( dcn <= 0 ) dcn = 3;
            CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) && (depth == CV_8U || depth == CV_32F) );
            bidx = code == CV_Lab2BGR || code == CV_Luv2BGR ||
                   code == CV_Lab2LBGR || code == CV_Luv2LBGR ? 0 : 2;
            bool srgb = code == CV_Lab2BGR || code == CV_Lab2RGB ||
                    code == CV_Luv2BGR || code == CV_Luv2RGB;

            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();

            if( code == CV_Lab2BGR || code == CV_Lab2RGB ||
                code == CV_Lab2LBGR || code == CV_Lab2LRGB )
            {
                if( depth == CV_8U )
                    CvtColorLoop(src, dst, Lab2RGB_b(dcn, bidx, 0, 0, srgb));
                else
                    CvtColorLoop(src, dst, Lab2RGB_f(dcn, bidx, 0, 0, srgb));
            }
            else
            {
                if( depth == CV_8U )
                    CvtColorLoop(src, dst, Luv2RGB_b(dcn, bidx, 0, 0, srgb));
                else
                    CvtColorLoop(src, dst, Luv2RGB_f(dcn, bidx, 0, 0, srgb));
            }
            }
            break;

        case CV_BayerBG2GRAY: case CV_BayerGB2GRAY: case CV_BayerRG2GRAY: case CV_BayerGR2GRAY:
        case CV_BayerBG2BGR: case CV_BayerGB2BGR: case CV_BayerRG2BGR: case CV_BayerGR2BGR:
        case CV_BayerBG2BGR_VNG: case CV_BayerGB2BGR_VNG: case CV_BayerRG2BGR_VNG: case CV_BayerGR2BGR_VNG:
        case CV_BayerBG2BGR_EA: case CV_BayerGB2BGR_EA: case CV_BayerRG2BGR_EA: case CV_BayerGR2BGR_EA:
            demosaicing(src, _dst, code, dcn);
            break;

        case CV_YUV2BGR_NV21:  case CV_YUV2RGB_NV21:  case CV_YUV2BGR_NV12:  case CV_YUV2RGB_NV12:
        case CV_YUV2BGRA_NV21: case CV_YUV2RGBA_NV21: case CV_YUV2BGRA_NV12: case CV_YUV2RGBA_NV12:
            {
                // http://www.fourcc.org/yuv.php#NV21 == yuv420sp -> a plane of 8 bit Y samples followed by an interleaved V/U plane containing 8 bit 2x2 subsampled chroma samples
                // http://www.fourcc.org/yuv.php#NV12 -> a plane of 8 bit Y samples followed by an interleaved U/V plane containing 8 bit 2x2 subsampled colour difference samples

                if (dcn <= 0) dcn = (code==CV_YUV420sp2BGRA || code==CV_YUV420sp2RGBA || code==CV_YUV2BGRA_NV12 || code==CV_YUV2RGBA_NV12) ? 4 : 3;
                const int bIdx = (code==CV_YUV2BGR_NV21 || code==CV_YUV2BGRA_NV21 || code==CV_YUV2BGR_NV12 || code==CV_YUV2BGRA_NV12) ? 0 : 2;
                const int uIdx = (code==CV_YUV2BGR_NV21 || code==CV_YUV2BGRA_NV21 || code==CV_YUV2RGB_NV21 || code==CV_YUV2RGBA_NV21) ? 1 : 0;

                CV_Assert( dcn == 3 || dcn == 4 );
                CV_Assert( sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U );

                Size dstSz(sz.width, sz.height * 2 / 3);
                _dst.create(dstSz, CV_MAKETYPE(depth, dcn));
                dst = _dst.getMat();

                int srcstep = (int)src.step;
                const uchar* y = src.ptr();
                const uchar* uv = y + srcstep * dstSz.height;

                switch(dcn*100 + bIdx * 10 + uIdx)
                {
                    case 300: cvtYUV420sp2RGB<0, 0> (dst, srcstep, y, uv); break;
                    case 301: cvtYUV420sp2RGB<0, 1> (dst, srcstep, y, uv); break;
                    case 320: cvtYUV420sp2RGB<2, 0> (dst, srcstep, y, uv); break;
                    case 321: cvtYUV420sp2RGB<2, 1> (dst, srcstep, y, uv); break;
                    case 400: cvtYUV420sp2RGBA<0, 0>(dst, srcstep, y, uv); break;
                    case 401: cvtYUV420sp2RGBA<0, 1>(dst, srcstep, y, uv); break;
                    case 420: cvtYUV420sp2RGBA<2, 0>(dst, srcstep, y, uv); break;
                    case 421: cvtYUV420sp2RGBA<2, 1>(dst, srcstep, y, uv); break;
                    default: CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" ); break;
                };
            }
            break;
        case CV_YUV2BGR_YV12: case CV_YUV2RGB_YV12: case CV_YUV2BGRA_YV12: case CV_YUV2RGBA_YV12:
        case CV_YUV2BGR_IYUV: case CV_YUV2RGB_IYUV: case CV_YUV2BGRA_IYUV: case CV_YUV2RGBA_IYUV:
            {
                //http://www.fourcc.org/yuv.php#YV12 == yuv420p -> It comprises an NxM Y plane followed by (N/2)x(M/2) V and U planes.
                //http://www.fourcc.org/yuv.php#IYUV == I420 -> It comprises an NxN Y plane followed by (N/2)x(N/2) U and V planes

                if (dcn <= 0) dcn = (code==CV_YUV2BGRA_YV12 || code==CV_YUV2RGBA_YV12 || code==CV_YUV2RGBA_IYUV || code==CV_YUV2BGRA_IYUV) ? 4 : 3;
                const int bIdx = (code==CV_YUV2BGR_YV12 || code==CV_YUV2BGRA_YV12 || code==CV_YUV2BGR_IYUV || code==CV_YUV2BGRA_IYUV) ? 0 : 2;
                const int uIdx  = (code==CV_YUV2BGR_YV12 || code==CV_YUV2RGB_YV12 || code==CV_YUV2BGRA_YV12 || code==CV_YUV2RGBA_YV12) ? 1 : 0;

                CV_Assert( dcn == 3 || dcn == 4 );
                CV_Assert( sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U );

                Size dstSz(sz.width, sz.height * 2 / 3);
                _dst.create(dstSz, CV_MAKETYPE(depth, dcn));
                dst = _dst.getMat();

                int srcstep = (int)src.step;
                const uchar* y = src.ptr();
                const uchar* u = y + srcstep * dstSz.height;
                const uchar* v = y + srcstep * (dstSz.height + dstSz.height/4) + (dstSz.width/2) * ((dstSz.height % 4)/2);

                int ustepIdx = 0;
                int vstepIdx = dstSz.height % 4 == 2 ? 1 : 0;

                if(uIdx == 1) { std::swap(u ,v), std::swap(ustepIdx, vstepIdx); };

                switch(dcn*10 + bIdx)
                {
                    case 30: cvtYUV420p2RGB<0>(dst, srcstep, y, u, v, ustepIdx, vstepIdx); break;
                    case 32: cvtYUV420p2RGB<2>(dst, srcstep, y, u, v, ustepIdx, vstepIdx); break;
                    case 40: cvtYUV420p2RGBA<0>(dst, srcstep, y, u, v, ustepIdx, vstepIdx); break;
                    case 42: cvtYUV420p2RGBA<2>(dst, srcstep, y, u, v, ustepIdx, vstepIdx); break;
                    default: CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" ); break;
                };
            }
            break;
        case CV_YUV2GRAY_420:
            {
                if (dcn <= 0) dcn = 1;

                CV_Assert( dcn == 1 );
                CV_Assert( sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U );

                Size dstSz(sz.width, sz.height * 2 / 3);
                _dst.create(dstSz, CV_MAKETYPE(depth, dcn));
                dst = _dst.getMat();

                src(Range(0, dstSz.height), Range::all()).copyTo(dst);
            }
            break;
        case CV_RGB2YUV_YV12: case CV_BGR2YUV_YV12: case CV_RGBA2YUV_YV12: case CV_BGRA2YUV_YV12:
        case CV_RGB2YUV_IYUV: case CV_BGR2YUV_IYUV: case CV_RGBA2YUV_IYUV: case CV_BGRA2YUV_IYUV:
            {
                if (dcn <= 0) dcn = 1;
                const int bIdx = (code == CV_BGR2YUV_IYUV || code == CV_BGRA2YUV_IYUV || code == CV_BGR2YUV_YV12 || code == CV_BGRA2YUV_YV12) ? 0 : 2;
                const int uIdx = (code == CV_BGR2YUV_IYUV || code == CV_BGRA2YUV_IYUV || code == CV_RGB2YUV_IYUV || code == CV_RGBA2YUV_IYUV) ? 1 : 2;

                CV_Assert( (scn == 3 || scn == 4) && depth == CV_8U );
                CV_Assert( dcn == 1 );
                CV_Assert( sz.width % 2 == 0 && sz.height % 2 == 0 );

                Size dstSz(sz.width, sz.height / 2 * 3);
                _dst.create(dstSz, CV_MAKETYPE(depth, dcn));
                dst = _dst.getMat();

                switch(bIdx + uIdx*10)
                {
                    case 10: cvtRGBtoYUV420p<0, 1>(src, dst); break;
                    case 12: cvtRGBtoYUV420p<2, 1>(src, dst); break;
                    case 20: cvtRGBtoYUV420p<0, 2>(src, dst); break;
                    case 22: cvtRGBtoYUV420p<2, 2>(src, dst); break;
                    default: CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" ); break;
                };
            }
            break;
        case CV_YUV2RGB_UYVY: case CV_YUV2BGR_UYVY: case CV_YUV2RGBA_UYVY: case CV_YUV2BGRA_UYVY:
        case CV_YUV2RGB_YUY2: case CV_YUV2BGR_YUY2: case CV_YUV2RGB_YVYU: case CV_YUV2BGR_YVYU:
        case CV_YUV2RGBA_YUY2: case CV_YUV2BGRA_YUY2: case CV_YUV2RGBA_YVYU: case CV_YUV2BGRA_YVYU:
            {
                //http://www.fourcc.org/yuv.php#UYVY
                //http://www.fourcc.org/yuv.php#YUY2
                //http://www.fourcc.org/yuv.php#YVYU

                if (dcn <= 0) dcn = (code==CV_YUV2RGBA_UYVY || code==CV_YUV2BGRA_UYVY || code==CV_YUV2RGBA_YUY2 || code==CV_YUV2BGRA_YUY2 || code==CV_YUV2RGBA_YVYU || code==CV_YUV2BGRA_YVYU) ? 4 : 3;
                const int bIdx = (code==CV_YUV2BGR_UYVY || code==CV_YUV2BGRA_UYVY || code==CV_YUV2BGR_YUY2 || code==CV_YUV2BGRA_YUY2 || code==CV_YUV2BGR_YVYU || code==CV_YUV2BGRA_YVYU) ? 0 : 2;
                const int ycn  = (code==CV_YUV2RGB_UYVY || code==CV_YUV2BGR_UYVY || code==CV_YUV2RGBA_UYVY || code==CV_YUV2BGRA_UYVY) ? 1 : 0;
                const int uIdx = (code==CV_YUV2RGB_YVYU || code==CV_YUV2BGR_YVYU || code==CV_YUV2RGBA_YVYU || code==CV_YUV2BGRA_YVYU) ? 1 : 0;

                CV_Assert( dcn == 3 || dcn == 4 );
                CV_Assert( scn == 2 && depth == CV_8U );

                _dst.create(sz, CV_8UC(dcn));
                dst = _dst.getMat();

                switch(dcn*1000 + bIdx*100 + uIdx*10 + ycn)
                {
                    case 3000: cvtYUV422toRGB<0,0,0>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 3001: cvtYUV422toRGB<0,0,1>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 3010: cvtYUV422toRGB<0,1,0>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 3011: cvtYUV422toRGB<0,1,1>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 3200: cvtYUV422toRGB<2,0,0>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 3201: cvtYUV422toRGB<2,0,1>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 3210: cvtYUV422toRGB<2,1,0>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 3211: cvtYUV422toRGB<2,1,1>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 4000: cvtYUV422toRGBA<0,0,0>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 4001: cvtYUV422toRGBA<0,0,1>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 4010: cvtYUV422toRGBA<0,1,0>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 4011: cvtYUV422toRGBA<0,1,1>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 4200: cvtYUV422toRGBA<2,0,0>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 4201: cvtYUV422toRGBA<2,0,1>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 4210: cvtYUV422toRGBA<2,1,0>(dst, (int)src.step, src.ptr<uchar>()); break;
                    case 4211: cvtYUV422toRGBA<2,1,1>(dst, (int)src.step, src.ptr<uchar>()); break;
                    default: CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" ); break;
                };
            }
            break;
        case CV_YUV2GRAY_UYVY: case CV_YUV2GRAY_YUY2:
            {
                if (dcn <= 0) dcn = 1;

                CV_Assert( dcn == 1 );
                CV_Assert( scn == 2 && depth == CV_8U );

                extractChannel(_src, _dst, code == CV_YUV2GRAY_UYVY ? 1 : 0);
            }
            break;
        case CV_RGBA2mRGBA:
            {
                if (dcn <= 0) dcn = 4;
                CV_Assert( scn == 4 && dcn == 4 );

                _dst.create(sz, CV_MAKETYPE(depth, dcn));
                dst = _dst.getMat();

                if( depth == CV_8U )
                {
                    CvtColorLoop(src, dst, RGBA2mRGBA<uchar>());
                } else {
                    CV_Error( CV_StsBadArg, "Unsupported image depth" );
                }
            }
            break;
        case CV_mRGBA2RGBA:
            {
                if (dcn <= 0) dcn = 4;
                CV_Assert( scn == 4 && dcn == 4 );

                _dst.create(sz, CV_MAKETYPE(depth, dcn));
                dst = _dst.getMat();

                if( depth == CV_8U )
                {
                    CvtColorLoop(src, dst, mRGBA2RGBA<uchar>());
                } else {
                    CV_Error( CV_StsBadArg, "Unsupported image depth" );
                }
            }
            break;
        default:
            CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" );
    }
}

CV_IMPL void
cvCvtColor( const CvArr* srcarr, CvArr* dstarr, int code )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst0 = cv::cvarrToMat(dstarr), dst = dst0;
    CV_Assert( src.depth() == dst.depth() );

    cv::cvtColor(src, dst, code, dst.channels());
    CV_Assert( dst.data == dst0.data );
}


/* End of file. */
