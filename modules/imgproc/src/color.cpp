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
#include <limits>

namespace cv
{
    
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

template<class Cvt> void CvtColorLoop(const Mat& srcmat, Mat& dstmat, const Cvt& cvt)
{
    typedef typename Cvt::channel_type _Tp;
    Size sz = srcmat.size();
    const uchar* src = srcmat.data;
    uchar* dst = dstmat.data;
    size_t srcstep = srcmat.step, dststep = dstmat.step;
    
    if( srcmat.isContinuous() && dstmat.isContinuous() )
    {
        sz.width *= sz.height;
        sz.height = 1;
    }    
    
    for( ; sz.height--; src += srcstep, dst += dststep )
        cvt((const _Tp*)src, (_Tp*)dst, sz.width);
}
    
    
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
    
    RGB2Gray<uchar>(int _srccn, int blueIdx, const int* coeffs) : srccn(_srccn)
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
    int srccn, blueIdx;
    int tab[256*3];
};

    
template<> struct RGB2Gray<ushort>
{
    typedef ushort channel_type;
    
    RGB2Gray<ushort>(int _srccn, int blueIdx, const int* _coeffs) : srccn(_srccn)
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
    : srccn(_srccn), blueIdx(_blueIdx), hrange(_hrange) {}
    
    void operator()(const uchar* src, uchar* dst, int n) const
    {
        int i, bidx = blueIdx, scn = srccn;
        const int hsv_shift = 12;
        
        static const int div_table[] = {
            0, 1044480, 522240, 348160, 261120, 208896, 174080, 149211,
            130560, 116053, 104448, 94953, 87040, 80345, 74606, 69632,
            65280, 61440, 58027, 54973, 52224, 49737, 47476, 45412,
            43520, 41779, 40172, 38684, 37303, 36017, 34816, 33693,
            32640, 31651, 30720, 29842, 29013, 28229, 27486, 26782,
            26112, 25475, 24869, 24290, 23738, 23211, 22706, 22223,
            21760, 21316, 20890, 20480, 20086, 19707, 19342, 18991,
            18651, 18324, 18008, 17703, 17408, 17123, 16846, 16579,
            16320, 16069, 15825, 15589, 15360, 15137, 14921, 14711,
            14507, 14308, 14115, 13926, 13743, 13565, 13391, 13221,
            13056, 12895, 12738, 12584, 12434, 12288, 12145, 12006,
            11869, 11736, 11605, 11478, 11353, 11231, 11111, 10995,
            10880, 10768, 10658, 10550, 10445, 10341, 10240, 10141,
            10043, 9947, 9854, 9761, 9671, 9582, 9495, 9410,
            9326, 9243, 9162, 9082, 9004, 8927, 8852, 8777,
            8704, 8632, 8561, 8492, 8423, 8356, 8290, 8224,
            8160, 8097, 8034, 7973, 7913, 7853, 7795, 7737,
            7680, 7624, 7569, 7514, 7461, 7408, 7355, 7304,
            7253, 7203, 7154, 7105, 7057, 7010, 6963, 6917,
            6872, 6827, 6782, 6739, 6695, 6653, 6611, 6569,
            6528, 6487, 6447, 6408, 6369, 6330, 6292, 6254,
            6217, 6180, 6144, 6108, 6073, 6037, 6003, 5968,
            5935, 5901, 5868, 5835, 5803, 5771, 5739, 5708,
            5677, 5646, 5615, 5585, 5556, 5526, 5497, 5468,
            5440, 5412, 5384, 5356, 5329, 5302, 5275, 5249,
            5222, 5196, 5171, 5145, 5120, 5095, 5070, 5046,
            5022, 4998, 4974, 4950, 4927, 4904, 4881, 4858,
            4836, 4813, 4791, 4769, 4748, 4726, 4705, 4684,
            4663, 4642, 4622, 4601, 4581, 4561, 4541, 4522,
            4502, 4483, 4464, 4445, 4426, 4407, 4389, 4370,
            4352, 4334, 4316, 4298, 4281, 4263, 4246, 4229,
            4212, 4195, 4178, 4161, 4145, 4128, 4112, 4096
        };
        int hr = hrange, hscale = hr == 180 ? 15 : 21;
        n *= 3;
        
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
            
            s = diff * div_table[v] >> hsv_shift;
            h = (vr & (g - b)) +
                (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
            h = (h * div_table[diff] * hscale + (1 << (hsv_shift + 6))) >> (7 + hsv_shift);
            h += h < 0 ? hr : 0;
            
            dst[i] = (uchar)h;
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
    : dstcn(_dstcn), cvt(3, _blueIdx, _hrange)
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
    : dstcn(_dstcn), cvt(3, _blueIdx, _hrange)
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
        float f[LAB_CBRT_TAB_SIZE+1], g[GAMMA_TAB_SIZE], ig[GAMMA_TAB_SIZE], scale = 1.f/LabCbrtTabScale;
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
            g[i] = x <= 0.04045f ? x*(1.f/12.92f) : (float)pow((double)(x + 0.055)*(1./1.055), 2.4);
            ig[i] = x <= 0.0031308 ? x*12.92f : (float)(1.055*pow((double)x, 1./2.4) - 0.055);
        }
        splineBuild(g, GAMMA_TAB_SIZE, sRGBGammaTab);
        splineBuild(ig, GAMMA_TAB_SIZE, sRGBInvGammaTab);
        
        for(i = 0; i < 256; i++)
        {
            float x = i*(1.f/255.f);
            sRGBGammaTab_b[i] = saturate_cast<ushort>(255.f*(1 << gamma_shift)*(x <= 0.04045f ? x*(1.f/12.92f) : (float)pow((double)(x + 0.055)*(1./1.055), 2.4)));
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
        initLabTabs();
        
        if(!_coeffs) _coeffs = sRGB2XYZ_D65;
        if(!_whitept) _whitept = D65;
        float scale[] =
        {
            (1 << lab_shift)/_whitept[0],
            (float)(1 << lab_shift),
            (1 << lab_shift)/_whitept[2]
        };
        
        for( int i = 0; i < 3; i++ )
        {
            coeffs[i*3+(blueIdx^2)] = cvRound(_coeffs[i*3]*scale[i]);
            coeffs[i*3+1] = cvRound(_coeffs[i*3+1]*scale[i]);
            coeffs[i*3+blueIdx] = cvRound(_coeffs[i*3+2]*scale[i]);
            CV_Assert( coeffs[i] >= 0 && coeffs[i+3] >= 0 && coeffs[i+6] >= 0 &&
                      coeffs[i] + coeffs[i+1] + coeffs[i+2] < 2*(1 << lab_shift) );
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
    
    
struct RGB2Lab_f
{
    typedef float channel_type;
    
    RGB2Lab_f(int _srccn, int blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb)
    : srccn(_srccn), srgb(_srgb)
    {
        initLabTabs();
        
        if(!_coeffs) _coeffs = sRGB2XYZ_D65;
        if(!_whitept) _whitept = D65;
        float scale[] = { LabCbrtTabScale/_whitept[0], LabCbrtTabScale, LabCbrtTabScale/_whitept[2] };
        
        for( int i = 0; i < 3; i++ )
        {
            coeffs[i*3+(blueIdx^2)] = _coeffs[i*3]*scale[i];
            coeffs[i*3+1] = _coeffs[i*3+1]*scale[i];
            coeffs[i*3+blueIdx] = _coeffs[i*3+2]*scale[i];
            CV_Assert( coeffs[i*3] >= 0 && coeffs[i*3+1] >= 0 && coeffs[i*3+2] >= 0 &&
                       coeffs[i*3] + coeffs[i*3+1] + coeffs[i*3+2] < 1.5f*LabCbrtTabScale );
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
        
        for( i = 0; i < n; i += 3, src += scn )
        {
            float R = src[0], G = src[1], B = src[2];
            if( gammaTab )
            {
                R = splineInterpolate(R*gscale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*gscale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*gscale, gammaTab, GAMMA_TAB_SIZE);
            }
            float fX = splineInterpolate(R*C0 + G*C1 + B*C2, LabCbrtTab, LAB_CBRT_TAB_SIZE); 
            float fY = splineInterpolate(R*C3 + G*C4 + B*C5, LabCbrtTab, LAB_CBRT_TAB_SIZE);
            float fZ = splineInterpolate(R*C6 + G*C7 + B*C8, LabCbrtTab, LAB_CBRT_TAB_SIZE);
            
            float L = 116.f*fY - 16.f;
            float a = 500.f*(fX - fY);
            float b = 200.f*(fY - fZ);
            
            dst[i] = L; dst[i+1] = a; dst[i+2] = b;
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
    : dstcn(_dstcn), srgb(_srgb)
    {
        initLabTabs();
        
        if(!_coeffs) _coeffs = XYZ2sRGB_D65;
        if(!_whitept) _whitept = D65;
        
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
        
        for( i = 0; i < n; i += 3, dst += dcn )
        {
            float L = src[i], a = src[i+1], b = src[i+2];
            float Y = (L + 16.f)*(1.f/116.f);
            float X = (Y + a*0.002f);
            float Z = (Y - b*0.005f);
            Y = Y*Y*Y;
            X = X*X*X;
            Z = Z*Z*Z;
            
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
    float coeffs[9];
    bool srgb;
};

    
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
        initLabTabs();
        
        if(!_coeffs) _coeffs = sRGB2XYZ_D65;
        if(!whitept) whitept = D65;
        
        for( int i = 0; i < 3; i++ )
        {
            coeffs[i*3+(blueIdx^2)] = _coeffs[i*3];
            coeffs[i*3+1] = _coeffs[i*3+1];
            coeffs[i*3+blueIdx] = _coeffs[i*3+2];
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
            float v = L*((9*0.25)*Y*d - _vn);
            
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
            Z = (12 - 3 * u - 20 * v) * Y * 0.25 * iv;                
                        
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

        
//////////////////////////// Bayer Pattern -> RGB conversion /////////////////////////////

static void Bayer2RGB_8u( const Mat& srcmat, Mat& dstmat, int code )
{
    const uchar* bayer0 = srcmat.data;
    int bayer_step = (int)srcmat.step;
    uchar* dst0 = dstmat.data;
    int dst_step = (int)dstmat.step;
    Size size = srcmat.size();
    int blue = code == CV_BayerBG2BGR || code == CV_BayerGB2BGR ? -1 : 1;
    int start_with_green = code == CV_BayerGB2BGR || code == CV_BayerGR2BGR;

    memset( dst0, 0, size.width*3*sizeof(dst0[0]) );
    memset( dst0 + (size.height - 1)*dst_step, 0, size.width*3*sizeof(dst0[0]) );
    dst0 += dst_step + 3 + 1;
    size.height -= 2;
    size.width -= 2;

    for( ; size.height-- > 0; bayer0 += bayer_step, dst0 += dst_step )
    {
        int t0, t1;
        const uchar* bayer = bayer0;
        uchar* dst = dst0;
        const uchar* bayer_end = bayer + size.width;

        dst[-4] = dst[-3] = dst[-2] = dst[size.width*3-1] =
            dst[size.width*3] = dst[size.width*3+1] = 0;

        if( size.width <= 0 )
            continue;

        if( start_with_green )
        {
            t0 = (bayer[1] + bayer[bayer_step*2+1] + 1) >> 1;
            t1 = (bayer[bayer_step] + bayer[bayer_step+2] + 1) >> 1;
            dst[-blue] = (uchar)t0;
            dst[0] = bayer[bayer_step+1];
            dst[blue] = (uchar)t1;
            bayer++;
            dst += 3;
        }

        if( blue > 0 )
        {
            for( ; bayer <= bayer_end - 2; bayer += 2, dst += 6 )
            {
                t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] +
                      bayer[bayer_step*2+2] + 2) >> 2;
                t1 = (bayer[1] + bayer[bayer_step] +
                      bayer[bayer_step+2] + bayer[bayer_step*2+1]+2) >> 2;
                dst[-1] = (uchar)t0;
                dst[0] = (uchar)t1;
                dst[1] = bayer[bayer_step+1];

                t0 = (bayer[2] + bayer[bayer_step*2+2] + 1) >> 1;
                t1 = (bayer[bayer_step+1] + bayer[bayer_step+3] + 1) >> 1;
                dst[2] = (uchar)t0;
                dst[3] = bayer[bayer_step+2];
                dst[4] = (uchar)t1;
            }
        }
        else
        {
            for( ; bayer <= bayer_end - 2; bayer += 2, dst += 6 )
            {
                t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] +
                      bayer[bayer_step*2+2] + 2) >> 2;
                t1 = (bayer[1] + bayer[bayer_step] +
                      bayer[bayer_step+2] + bayer[bayer_step*2+1]+2) >> 2;
                dst[1] = (uchar)t0;
                dst[0] = (uchar)t1;
                dst[-1] = bayer[bayer_step+1];

                t0 = (bayer[2] + bayer[bayer_step*2+2] + 1) >> 1;
                t1 = (bayer[bayer_step+1] + bayer[bayer_step+3] + 1) >> 1;
                dst[4] = (uchar)t0;
                dst[3] = bayer[bayer_step+2];
                dst[2] = (uchar)t1;
            }
        }

        if( bayer < bayer_end )
        {
            t0 = (bayer[0] + bayer[2] + bayer[bayer_step*2] +
                  bayer[bayer_step*2+2] + 2) >> 2;
            t1 = (bayer[1] + bayer[bayer_step] +
                  bayer[bayer_step+2] + bayer[bayer_step*2+1]+2) >> 2;
            dst[-blue] = (uchar)t0;
            dst[0] = (uchar)t1;
            dst[blue] = bayer[bayer_step+1];
            bayer++;
            dst += 3;
        }

        blue = -blue;
        start_with_green = !start_with_green;
    }
}

    
/////////////////// Demosaicing using Variable Number of Gradients ///////////////////////
    
static void Bayer2RGB_VNG_8u( const Mat& srcmat, Mat& dstmat, int code )
{
    const uchar* bayer = srcmat.data;
    int bstep = (int)srcmat.step;
    uchar* dst = dstmat.data;
    int dststep = (int)dstmat.step;
    Size size = srcmat.size();
    
    int blueIdx = code == CV_BayerBG2BGR_VNG || code == CV_BayerGB2BGR_VNG ? 0 : 2;
    bool greenCell0 = code != CV_BayerBG2BGR_VNG && code != CV_BayerRG2BGR_VNG;
    
    // for too small images use the simple interpolation algorithm
    if( MIN(size.width, size.height) < 8 )
    {
        Bayer2RGB_8u( srcmat, dstmat, code );
        return;
    }
    
    const int brows = 3, bcn = 7;
    int N = size.width, N2 = N*2, N3 = N*3, N4 = N*4, N5 = N*5, N6 = N*6, N7 = N*7;  
    int i, bufstep = N7*bcn;
    cv::AutoBuffer<ushort> _buf(bufstep*brows);
    ushort* buf = (ushort*)_buf;
    
    bayer += bstep*2;
    
#if CV_SSE2
    bool haveSSE = cv::checkHardwareSupport(CV_CPU_SSE2);
    #define _mm_absdiff_epu16(a,b) _mm_adds_epu16(_mm_subs_epu16(a, b), _mm_subs_epu16(b, a))
#endif
    
    for( int y = 2; y < size.height - 4; y++ )
    {
        uchar* dstrow = dst + dststep*y + 6;
        const uchar* srow;
        
        for( int dy = (y == 2 ? -1 : 1); dy <= 1; dy++ )
        {
            ushort* brow = buf + ((y + dy - 1)%brows)*bufstep + 1;
            srow = bayer + (y+dy)*bstep + 1;
            
            for( i = 0; i < bcn; i++ )
                brow[N*i-1] = brow[(N-2) + N*i] = 0;
            
            i = 1;
            
#if CV_SSE2
            if( haveSSE )
            {
                __m128i z = _mm_setzero_si128();
                for( ; i <= N-9; i += 8, srow += 8, brow += 8 )
                {
                    __m128i s1, s2, s3, s4, s6, s7, s8, s9;
                    
                    s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-1-bstep)),z);
                    s2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-bstep)),z);
                    s3 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+1-bstep)),z);
                    
                    s4 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-1)),z);
                    s6 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+1)),z);
                    
                    s7 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-1+bstep)),z);
                    s8 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+bstep)),z);
                    s9 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+1+bstep)),z);
                    
                    __m128i b0, b1, b2, b3, b4, b5, b6;
                    
                    b0 = _mm_adds_epu16(_mm_slli_epi16(_mm_absdiff_epu16(s2,s8),1),
                                        _mm_adds_epu16(_mm_absdiff_epu16(s1, s7),
                                                       _mm_absdiff_epu16(s3, s9)));
                    b1 = _mm_adds_epu16(_mm_slli_epi16(_mm_absdiff_epu16(s4,s6),1),
                                        _mm_adds_epu16(_mm_absdiff_epu16(s1, s3),
                                                       _mm_absdiff_epu16(s7, s9)));
                    b2 = _mm_slli_epi16(_mm_absdiff_epu16(s3,s7),1);
                    b3 = _mm_slli_epi16(_mm_absdiff_epu16(s1,s9),1);
                    
                    _mm_storeu_si128((__m128i*)brow, b0);
                    _mm_storeu_si128((__m128i*)(brow + N), b1);
                    _mm_storeu_si128((__m128i*)(brow + N2), b2);
                    _mm_storeu_si128((__m128i*)(brow + N3), b3);
                    
                    b4 = _mm_adds_epu16(b2,_mm_adds_epu16(_mm_absdiff_epu16(s2, s4),
                                                          _mm_absdiff_epu16(s6, s8)));
                    b5 = _mm_adds_epu16(b3,_mm_adds_epu16(_mm_absdiff_epu16(s2, s6),
                                                          _mm_absdiff_epu16(s4, s8)));
                    b6 = _mm_adds_epu16(_mm_adds_epu16(s2, s4), _mm_adds_epu16(s6, s8));
                    b6 = _mm_srli_epi16(b6, 1);
                    
                    _mm_storeu_si128((__m128i*)(brow + N4), b4);
                    _mm_storeu_si128((__m128i*)(brow + N5), b5);
                    _mm_storeu_si128((__m128i*)(brow + N6), b6);
                }
            }
#endif
            
            for( ; i < N-1; i++, srow++, brow++ )
            {
                brow[0] = (ushort)(std::abs(srow[-1-bstep] - srow[-1+bstep]) +
                                   std::abs(srow[-bstep] - srow[+bstep])*2 +
                                   std::abs(srow[1-bstep] - srow[1+bstep]));
                brow[N] = (ushort)(std::abs(srow[-1-bstep] - srow[1-bstep]) +
                                   std::abs(srow[-1] - srow[1])*2 +
                                   std::abs(srow[-1+bstep] - srow[1+bstep]));
                brow[N2] = (ushort)(std::abs(srow[+1-bstep] - srow[-1+bstep])*2);
                brow[N3] = (ushort)(std::abs(srow[-1-bstep] - srow[1+bstep])*2);
                brow[N4] = (ushort)(brow[N2] + std::abs(srow[-bstep] - srow[-1]) +
                                    std::abs(srow[+bstep] - srow[1]));
                brow[N5] = (ushort)(brow[N3] + std::abs(srow[-bstep] - srow[1]) +
                                    std::abs(srow[+bstep] - srow[-1]));
                brow[N6] = (ushort)((srow[-bstep] + srow[-1] + srow[1] + srow[+bstep])>>1);
            }
        }
        
        const ushort* brow0 = buf + ((y - 2) % brows)*bufstep + 2;
        const ushort* brow1 = buf + ((y - 1) % brows)*bufstep + 2;
        const ushort* brow2 = buf + (y % brows)*bufstep + 2;
        static const float scale[] = { 0.f, 0.5f, 0.25f, 0.1666666666667f, 0.125f, 0.1f, 0.08333333333f, 0.0714286f, 0.0625f };
        srow = bayer + y*bstep + 2;
        bool greenCell = greenCell0;
        
        i = 2;
#if CV_SSE2        
        int limit = !haveSSE ? N-2 : greenCell ? std::min(3, N-2) : 2;
#else
        int limit = N - 2;
#endif
        
        do
        {
            for( ; i < limit; i++, srow++, brow0++, brow1++, brow2++, dstrow += 3 )
            {
                int gradN = brow0[0] + brow1[0];
                int gradS = brow1[0] + brow2[0];
                int gradW = brow1[N-1] + brow1[N];
                int gradE = brow1[N] + brow1[N+1];
                int minGrad = std::min(std::min(std::min(gradN, gradS), gradW), gradE);
                int maxGrad = std::max(std::max(std::max(gradN, gradS), gradW), gradE);
                int R, G, B;
                
                if( !greenCell )
                {
                    int gradNE = brow0[N4+1] + brow1[N4];
                    int gradSW = brow1[N4] + brow2[N4-1];
                    int gradNW = brow0[N5-1] + brow1[N5];
                    int gradSE = brow1[N5] + brow2[N5+1];
                    
                    minGrad = std::min(std::min(std::min(std::min(minGrad, gradNE), gradSW), gradNW), gradSE);
                    maxGrad = std::max(std::max(std::max(std::max(maxGrad, gradNE), gradSW), gradNW), gradSE);
                    int T = minGrad + maxGrad/2;
                    
                    int Rs = 0, Gs = 0, Bs = 0, ng = 0;
                    if( gradN < T )
                    {
                        Rs += srow[-bstep*2] + srow[0];
                        Gs += srow[-bstep]*2;
                        Bs += srow[-bstep-1] + srow[-bstep+1];
                        ng++;
                    }
                    if( gradS < T )
                    {
                        Rs += srow[bstep*2] + srow[0];
                        Gs += srow[bstep]*2;
                        Bs += srow[bstep-1] + srow[bstep+1];
                        ng++;
                    }
                    if( gradW < T )
                    {
                        Rs += srow[-2] + srow[0];
                        Gs += srow[-1]*2;
                        Bs += srow[-bstep-1] + srow[bstep-1];
                        ng++;
                    }
                    if( gradE < T )
                    {
                        Rs += srow[2] + srow[0];
                        Gs += srow[1]*2;
                        Bs += srow[-bstep+1] + srow[bstep+1];
                        ng++;
                    }
                    if( gradNE < T )
                    {
                        Rs += srow[-bstep*2+2] + srow[0];
                        Gs += brow0[N6+1];
                        Bs += srow[-bstep+1]*2;
                        ng++;
                    }
                    if( gradSW < T )
                    {
                        Rs += srow[bstep*2-2] + srow[0];
                        Gs += brow2[N6-1];
                        Bs += srow[bstep-1]*2;
                        ng++;
                    }
                    if( gradNW < T )
                    {
                        Rs += srow[-bstep*2-2] + srow[0];
                        Gs += brow0[N6-1];
                        Bs += srow[-bstep+1]*2;
                        ng++;
                    }
                    if( gradSE < T )
                    {
                        Rs += srow[bstep*2+2] + srow[0];
                        Gs += brow2[N6+1];
                        Bs += srow[-bstep+1]*2;
                        ng++;
                    }
                    R = srow[0];
                    G = R + cvRound((Gs - Rs)*scale[ng]);
                    B = R + cvRound((Bs - Rs)*scale[ng]); 
                }
                else
                {
                    int gradNE = brow0[N2] + brow0[N2+1] + brow1[N2] + brow1[N2+1];
                    int gradSW = brow1[N2] + brow1[N2-1] + brow2[N2] + brow2[N2-1];
                    int gradNW = brow0[N3] + brow0[N3-1] + brow1[N3] + brow1[N3-1];
                    int gradSE = brow1[N3] + brow1[N3+1] + brow2[N3] + brow2[N3+1];
                    
                    minGrad = std::min(std::min(std::min(std::min(minGrad, gradNE), gradSW), gradNW), gradSE);
                    maxGrad = std::max(std::max(std::max(std::max(maxGrad, gradNE), gradSW), gradNW), gradSE);
                    int T = minGrad + maxGrad/2;
                    
                    int Rs = 0, Gs = 0, Bs = 0, ng = 0;
                    if( gradN < T )
                    {
                        Rs += srow[-bstep*2-1] + srow[-bstep*2+1];
                        Gs += srow[-bstep*2] + srow[0];
                        Bs += srow[-bstep]*2;
                        ng++;
                    }
                    if( gradS < T )
                    {
                        Rs += srow[bstep*2-1] + srow[bstep*2+1];
                        Gs += srow[bstep*2] + srow[0];
                        Bs += srow[bstep]*2;
                        ng++;
                    }
                    if( gradW < T )
                    {
                        Rs += srow[-1]*2;
                        Gs += srow[-2] + srow[0];
                        Bs += srow[-bstep-2]+srow[bstep-2];
                        ng++;
                    }
                    if( gradE < T )
                    {
                        Rs += srow[1]*2;
                        Gs += srow[2] + srow[0];
                        Bs += srow[-bstep+2]+srow[bstep+2];
                        ng++;
                    }
                    if( gradNE < T )
                    {
                        Rs += srow[-bstep*2+1] + srow[1];
                        Gs += srow[-bstep+1]*2;
                        Bs += srow[-bstep] + srow[-bstep+2];
                        ng++;
                    }
                    if( gradSW < T )
                    {
                        Rs += srow[bstep*2-1] + srow[-1];
                        Gs += srow[bstep-1]*2;
                        Bs += srow[bstep] + srow[bstep-2];
                        ng++;
                    }
                    if( gradNW < T )
                    {
                        Rs += srow[-bstep*2-1] + srow[-1];
                        Gs += srow[-bstep-1]*2;
                        Bs += srow[-bstep-2]+srow[-bstep];
                        ng++;
                    }
                    if( gradSE < T )
                    {
                        Rs += srow[bstep*2+1] + srow[1];
                        Gs += srow[bstep+1]*2;
                        Bs += srow[bstep+2]+srow[bstep];
                        ng++;
                    }
                    G = srow[0];
                    R = G + cvRound((Rs - Gs)*scale[ng]);
                    B = G + cvRound((Bs - Gs)*scale[ng]);
                }
                dstrow[blueIdx] = CV_CAST_8U(B);
                dstrow[1] = CV_CAST_8U(G);
                dstrow[blueIdx^2] = CV_CAST_8U(R);
                greenCell = !greenCell;
            }
            
#if CV_SSE2
            if( !haveSSE )
                break;
            
            __m128i emask = _mm_set1_epi32(0x0000ffff),
            omask = _mm_set1_epi32(0xffff0000),
            z = _mm_setzero_si128();
            __m128 _0_5 = _mm_set1_ps(0.5f);
            
            #define _mm_merge_epi16(a, b) \
                _mm_or_si128(_mm_and_si128(a, emask), _mm_and_si128(b, omask))
            #define _mm_cvtloepi16_ps(a) _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(a,a), 16))
            #define _mm_cvthiepi16_ps(a) _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(a,a), 16))
            
            // process 8 pixels at once
            for( ; i <= N - 10; i += 8, srow += 8, brow0 += 8, brow1 += 8, brow2 += 8 )
            {
                __m128i gradN, gradS, gradW, gradE, gradNE, gradSW, gradNW, gradSE;
                gradN = _mm_adds_epu16(_mm_loadu_si128((__m128i*)brow0),
                                       _mm_loadu_si128((__m128i*)brow1));
                gradS = _mm_adds_epu16(_mm_loadu_si128((__m128i*)brow1),
                                       _mm_loadu_si128((__m128i*)brow2));
                gradW = _mm_adds_epu16(_mm_loadu_si128((__m128i*)(brow1+N-1)),
                                       _mm_loadu_si128((__m128i*)(brow1+N)));
                gradE = _mm_adds_epu16(_mm_loadu_si128((__m128i*)(brow1+N+1)),
                                       _mm_loadu_si128((__m128i*)(brow1+N)));
                
                __m128i minGrad, maxGrad, T;
                minGrad = _mm_min_epi16(_mm_min_epi16(_mm_min_epi16(gradN, gradS), gradW), gradE);
                maxGrad = _mm_max_epi16(_mm_max_epi16(_mm_max_epi16(gradN, gradS), gradW), gradE);
                
                __m128i grad0, grad1;
                
                grad0 = _mm_adds_epu16(_mm_loadu_si128((__m128i*)(brow0+N4+1)),
                                       _mm_loadu_si128((__m128i*)(brow1+N4)));
                grad1 = _mm_adds_epu16(_mm_adds_epu16(_mm_loadu_si128((__m128i*)(brow0+N2)),
                                                      _mm_loadu_si128((__m128i*)(brow0+N2+1))),
                                       _mm_adds_epu16(_mm_loadu_si128((__m128i*)(brow1+N2)),
                                                      _mm_loadu_si128((__m128i*)(brow1+N2+1))));
                gradNE = _mm_srli_epi16(_mm_merge_epi16(grad0, grad1), 1);
                
                grad0 = _mm_adds_epu16(_mm_loadu_si128((__m128i*)(brow2+N4-1)),
                                       _mm_loadu_si128((__m128i*)(brow1+N4)));
                grad1 = _mm_adds_epu16(_mm_adds_epu16(_mm_loadu_si128((__m128i*)(brow2+N2)),
                                                      _mm_loadu_si128((__m128i*)(brow2+N2-1))),
                                       _mm_adds_epu16(_mm_loadu_si128((__m128i*)(brow1+N2)),
                                                      _mm_loadu_si128((__m128i*)(brow1+N2-1))));
                gradSW = _mm_srli_epi16(_mm_merge_epi16(grad0, grad1), 1);
                
                minGrad = _mm_min_epi16(_mm_min_epi16(minGrad, gradNE), gradSW);
                maxGrad = _mm_max_epi16(_mm_max_epi16(maxGrad, gradNE), gradSW);
                
                grad0 = _mm_adds_epu16(_mm_loadu_si128((__m128i*)(brow0+N5-1)),
                                       _mm_loadu_si128((__m128i*)(brow1+N5)));
                grad1 = _mm_adds_epu16(_mm_adds_epu16(_mm_loadu_si128((__m128i*)(brow0+N3)),
                                                      _mm_loadu_si128((__m128i*)(brow0+N3-1))),
                                       _mm_adds_epu16(_mm_loadu_si128((__m128i*)(brow1+N3)),
                                                      _mm_loadu_si128((__m128i*)(brow1+N3-1))));
                gradNW = _mm_srli_epi16(_mm_merge_epi16(grad0, grad1), 1);
                
                grad0 = _mm_adds_epu16(_mm_loadu_si128((__m128i*)(brow2+N5+1)),
                                       _mm_loadu_si128((__m128i*)(brow1+N5)));
                grad1 = _mm_adds_epu16(_mm_adds_epu16(_mm_loadu_si128((__m128i*)(brow2+N3)),
                                                      _mm_loadu_si128((__m128i*)(brow2+N3+1))),
                                       _mm_adds_epu16(_mm_loadu_si128((__m128i*)(brow1+N3)),
                                                      _mm_loadu_si128((__m128i*)(brow1+N3+1))));
                gradSE = _mm_srli_epi16(_mm_merge_epi16(grad0, grad1), 1);
                
                minGrad = _mm_min_epi16(_mm_min_epi16(minGrad, gradNW), gradSE);
                maxGrad = _mm_max_epi16(_mm_max_epi16(maxGrad, gradNW), gradSE);
                
                T = _mm_add_epi16(_mm_srli_epi16(maxGrad, 1), minGrad);
                __m128i RGs = z, GRs = z, Bs = z, ng = z, mask;
                
                __m128i t0, t1, x0, x1, x2, x3, x4, x5, x6, x7, x8,
                x9, x10, x11, x12, x13, x14, x15, x16;
                
                x0 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)srow), z);
                
                x1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-bstep-1)), z);
                x2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-bstep*2-1)), z);
                x3 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-bstep)), z);
                x4 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-bstep*2+1)), z);
                x5 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-bstep+1)), z);
                x6 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-bstep+2)), z);
                x7 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+1)), z);
                x8 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+bstep+2)), z);
                x9 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+bstep+1)), z);
                x10 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+bstep*2+1)), z);
                x11 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+bstep)), z);
                x12 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+bstep*2-1)), z);
                x13 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+bstep-1)), z);
                x14 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+bstep-2)), z);
                x15 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-1)), z);
                x16 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-bstep-2)), z);
                
                // gradN
                mask = _mm_cmpgt_epi16(T, gradN);
                ng = _mm_sub_epi16(ng, mask);
                
                t0 = _mm_slli_epi16(x3, 1);
                t1 = _mm_adds_epu16(_mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-bstep*2)), z), x0);
                
                RGs = _mm_adds_epu16(RGs, _mm_and_si128(t1, mask));
                GRs = _mm_adds_epu16(GRs, _mm_and_si128(_mm_merge_epi16(t0, _mm_adds_epu16(x2,x4)), mask));
                Bs = _mm_adds_epu16(Bs, _mm_and_si128(_mm_merge_epi16(_mm_adds_epu16(x1,x5), t0), mask));
                
                // gradNE
                mask = _mm_cmpgt_epi16(T, gradNE);
                ng = _mm_sub_epi16(ng, mask);
                
                t0 = _mm_slli_epi16(x5, 1);
                t1 = _mm_adds_epu16(_mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-bstep*2+2)), z), x0);
                
                RGs = _mm_adds_epu16(RGs, _mm_and_si128(_mm_merge_epi16(t1, t0), mask));
                GRs = _mm_adds_epu16(GRs, _mm_and_si128(_mm_merge_epi16(_mm_loadu_si128((__m128i*)(brow0+N6+1)),
                                                                        _mm_adds_epu16(x4,x7)), mask));
                Bs = _mm_adds_epu16(Bs, _mm_and_si128(_mm_merge_epi16(t0,_mm_adds_epu16(x3,x6)), mask));
                
                // gradE
                mask = _mm_cmpgt_epi16(T, gradE);
                ng = _mm_sub_epi16(ng, mask);
                
                t0 = _mm_slli_epi16(x7, 1);
                t1 = _mm_adds_epu16(_mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+2)), z), x0);
                
                RGs = _mm_adds_epu16(RGs, _mm_and_si128(t1, mask));
                GRs = _mm_adds_epu16(GRs, _mm_and_si128(t0, mask));
                Bs = _mm_adds_epu16(Bs, _mm_and_si128(_mm_merge_epi16(_mm_adds_epu16(x5,x9),
                                                                      _mm_adds_epu16(x6,x8)), mask));
                
                // gradSE
                mask = _mm_cmpgt_epi16(T, gradSE);
                ng = _mm_sub_epi16(ng, mask);
                
                t0 = _mm_slli_epi16(x9, 1);
                t1 = _mm_adds_epu16(_mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+bstep*2+2)), z), x0);
                
                RGs = _mm_adds_epu16(RGs, _mm_and_si128(_mm_merge_epi16(t1, t0), mask));
                GRs = _mm_adds_epu16(GRs, _mm_and_si128(_mm_merge_epi16(_mm_loadu_si128((__m128i*)(brow2+N6+1)),
                                                                        _mm_adds_epu16(x7,x10)), mask));
                Bs = _mm_adds_epu16(Bs, _mm_and_si128(_mm_merge_epi16(t0, _mm_adds_epu16(x8,x11)), mask));
                
                // gradS
                mask = _mm_cmpgt_epi16(T, gradS);
                ng = _mm_sub_epi16(ng, mask);
                
                t0 = _mm_slli_epi16(x11, 1);
                t1 = _mm_adds_epu16(_mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+bstep*2)), z), x0);
                
                RGs = _mm_adds_epu16(RGs, _mm_and_si128(t1, mask));
                GRs = _mm_adds_epu16(GRs, _mm_and_si128(_mm_merge_epi16(t0, _mm_adds_epu16(x10,x12)), mask));
                Bs = _mm_adds_epu16(Bs, _mm_and_si128(_mm_merge_epi16(_mm_adds_epu16(x9,x13), t0), mask));
                
                // gradSW
                mask = _mm_cmpgt_epi16(T, gradSW);
                ng = _mm_sub_epi16(ng, mask);
                
                t0 = _mm_slli_epi16(x13, 1);
                t1 = _mm_adds_epu16(_mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow+bstep*2-2)), z), x0);
                
                RGs = _mm_adds_epu16(RGs, _mm_and_si128(_mm_merge_epi16(t1, t0), mask));
                GRs = _mm_adds_epu16(GRs, _mm_and_si128(_mm_merge_epi16(_mm_loadu_si128((__m128i*)(brow2+N6-1)),
                                                                        _mm_adds_epu16(x12,x15)), mask));
                Bs = _mm_adds_epu16(Bs, _mm_and_si128(_mm_merge_epi16(t0,_mm_adds_epu16(x11,x14)), mask));
                
                // gradW
                mask = _mm_cmpgt_epi16(T, gradW);
                ng = _mm_sub_epi16(ng, mask);
                
                t0 = _mm_slli_epi16(x15, 1);
                t1 = _mm_adds_epu16(_mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-2)), z), x0);
                
                RGs = _mm_adds_epu16(RGs, _mm_and_si128(t1, mask));
                GRs = _mm_adds_epu16(GRs, _mm_and_si128(t0, mask));
                Bs = _mm_adds_epu16(Bs, _mm_and_si128(_mm_merge_epi16(_mm_adds_epu16(x1,x13),
                                                                      _mm_adds_epu16(x14,x16)), mask));
                
                // gradNW
                mask = _mm_cmpgt_epi16(T, gradNW);
                ng = _mm_sub_epi16(ng, mask);
                
                __m128 ngf0, ngf1;
                ngf0 = _mm_div_ps(_0_5, _mm_cvtloepi16_ps(ng));
                ngf1 = _mm_div_ps(_0_5, _mm_cvthiepi16_ps(ng));
                
                t0 = _mm_slli_epi16(x1, 1);
                t1 = _mm_adds_epu16(_mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow-bstep*2-2)), z), x0);
                
                RGs = _mm_adds_epu16(RGs, _mm_and_si128(_mm_merge_epi16(t1, t0), mask));
                GRs = _mm_adds_epu16(GRs, _mm_and_si128(_mm_merge_epi16(_mm_loadu_si128((__m128i*)(brow0+N6-1)),
                                                                        _mm_adds_epu16(x2,x15)), mask));
                Bs = _mm_adds_epu16(Bs, _mm_and_si128(_mm_merge_epi16(t0,_mm_adds_epu16(x3,x16)), mask));
                
                // now interpolate r, g & b
                t0 = _mm_sub_epi16(GRs, RGs);
                t1 = _mm_sub_epi16(Bs, RGs);
                
                t0 = _mm_add_epi16(x0, _mm_packs_epi32(
                                                       _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtloepi16_ps(t0), ngf0)),
                                                       _mm_cvtps_epi32(_mm_mul_ps(_mm_cvthiepi16_ps(t0), ngf1))));
                
                t1 = _mm_add_epi16(x0, _mm_packs_epi32(
                                                       _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtloepi16_ps(t1), ngf0)),
                                                       _mm_cvtps_epi32(_mm_mul_ps(_mm_cvthiepi16_ps(t1), ngf1))));
                
                x1 = _mm_merge_epi16(x0, t0);
                x2 = _mm_merge_epi16(t0, x0);
                
                uchar R[8], G[8], B[8];
                
                _mm_storel_epi64(blueIdx ? (__m128i*)B : (__m128i*)R, _mm_packus_epi16(x1, z));
                _mm_storel_epi64((__m128i*)G, _mm_packus_epi16(x2, z));
                _mm_storel_epi64(blueIdx ? (__m128i*)R : (__m128i*)B, _mm_packus_epi16(t1, z));
                
                for( int j = 0; j < 8; j++, dstrow += 3 )
                {
                    dstrow[0] = B[j]; dstrow[1] = G[j]; dstrow[2] = R[j];
                }
            }
#endif
            
            limit = N - 2;
        }
        while( i < N - 2 );
        
        for( i = 0; i < 6; i++ )
        {
            dst[dststep*y + 5 - i] = dst[dststep*y + 8 - i];
            dst[dststep*y + (N - 2)*3 + i] = dst[dststep*y + (N - 3)*3 + i];
        }
        
        greenCell0 = !greenCell0;
        blueIdx ^= 2;
    }
    
    for( i = 0; i < size.width*3; i++ )
    {
        dst[i] = dst[i + dststep] = dst[i + dststep*2];
        dst[i + dststep*(size.height-4)] =
        dst[i + dststep*(size.height-3)] =
        dst[i + dststep*(size.height-2)] =
        dst[i + dststep*(size.height-1)] = dst[i + dststep*(size.height-5)];
    }
}


//////////////////////////////////////////////////////////////////////////////////////////
//                                   The main function                                  // 
//////////////////////////////////////////////////////////////////////////////////////////

void cvtColor( const Mat& src, Mat& dst, int code, int dcn )
{
    Size sz = src.size();
    int scn = src.channels(), depth = src.depth(), bidx;
    
    CV_Assert( depth == CV_8U || depth == CV_16U || depth == CV_32F );
        
    switch( code )
    {
        case CV_BGR2BGRA: case CV_RGB2BGRA: case CV_BGRA2BGR:
        case CV_RGBA2BGR: case CV_RGB2BGR: case CV_BGRA2RGBA:
            CV_Assert( scn == 3 || scn == 4 );
            dcn = code == CV_BGR2BGRA || code == CV_RGB2BGRA || code == CV_BGRA2RGBA ? 4 : 3;
            bidx = code == CV_BGR2BGRA || code == CV_BGRA2BGR ? 0 : 2;
            
            dst.create( sz, CV_MAKETYPE(depth, dcn));
            if( depth == CV_8U )
                CvtColorLoop(src, dst, RGB2RGB<uchar>(scn, dcn, bidx));
            else if( depth == CV_16U )
                CvtColorLoop(src, dst, RGB2RGB<ushort>(scn, dcn, bidx));
            else
                CvtColorLoop(src, dst, RGB2RGB<float>(scn, dcn, bidx));
            break;
            
        case CV_BGR2BGR565: case CV_BGR2BGR555: case CV_RGB2BGR565: case CV_RGB2BGR555:
        case CV_BGRA2BGR565: case CV_BGRA2BGR555: case CV_RGBA2BGR565: case CV_RGBA2BGR555:
            CV_Assert( (scn == 3 || scn == 4) && depth == CV_8U );
            dst.create(sz, CV_8UC2);
        
            CvtColorLoop(src, dst, RGB2RGB5x5(scn,
                      code == CV_BGR2BGR565 || code == CV_BGR2BGR555 ||
                      code == CV_BGRA2BGR565 || code == CV_BGRA2BGR555 ? 0 : 2,
                      code == CV_BGR2BGR565 || code == CV_RGB2BGR565 ||
                      code == CV_BGRA2BGR565 || code == CV_RGBA2BGR565 ? 6 : 5 // green bits
                                              ));
            break;
        
        case CV_BGR5652BGR: case CV_BGR5552BGR: case CV_BGR5652RGB: case CV_BGR5552RGB:
        case CV_BGR5652BGRA: case CV_BGR5552BGRA: case CV_BGR5652RGBA: case CV_BGR5552RGBA:
            if(dcn <= 0) dcn = 3;
            CV_Assert( (dcn == 3 || dcn == 4) && scn == 2 && depth == CV_8U );
            dst.create(sz, CV_MAKETYPE(depth, dcn));
            
            CvtColorLoop(src, dst, RGB5x52RGB(dcn,
                      code == CV_BGR5652BGR || code == CV_BGR5552BGR ||
                      code == CV_BGR5652BGRA || code == CV_BGR5552BGRA ? 0 : 2, // blue idx
                      code == CV_BGR5652BGR || code == CV_BGR5652RGB ||
                      code == CV_BGR5652BGRA || code == CV_BGR5652RGBA ? 6 : 5 // green bits
                      ));
            break;
                    
        case CV_BGR2GRAY: case CV_BGRA2GRAY: case CV_RGB2GRAY: case CV_RGBA2GRAY:
            CV_Assert( scn == 3 || scn == 4 );
            dst.create(sz, CV_MAKETYPE(depth, 1));
            bidx = code == CV_BGR2GRAY || code == CV_BGRA2GRAY ? 0 : 2;
            
            if( depth == CV_8U )
                CvtColorLoop(src, dst, RGB2Gray<uchar>(scn, bidx, 0));
            else if( depth == CV_16U )
                CvtColorLoop(src, dst, RGB2Gray<ushort>(scn, bidx, 0));
            else
                CvtColorLoop(src, dst, RGB2Gray<float>(scn, bidx, 0));
            break;
        
        case CV_BGR5652GRAY: case CV_BGR5552GRAY:
            CV_Assert( scn == 2 && depth == CV_8U );
            dst.create(sz, CV_8UC1);
            CvtColorLoop(src, dst, RGB5x52Gray(code == CV_BGR5652GRAY ? 6 : 5));
            break;
        
        case CV_GRAY2BGR: case CV_GRAY2BGRA:
            if( dcn <= 0 ) dcn = 3;
            CV_Assert( scn == 1 && (dcn == 3 || dcn == 4));
            dst.create(sz, CV_MAKETYPE(depth, dcn));
            
            if( depth == CV_8U )
                CvtColorLoop(src, dst, Gray2RGB<uchar>(dcn));
            else if( depth == CV_16U )
                CvtColorLoop(src, dst, Gray2RGB<ushort>(dcn));
            else
                CvtColorLoop(src, dst, Gray2RGB<float>(dcn));
            break;
            
        case CV_GRAY2BGR565: case CV_GRAY2BGR555:
            CV_Assert( scn == 1 && depth == CV_8U );
            dst.create(sz, CV_8UC2);
            
            CvtColorLoop(src, dst, Gray2RGB5x5(code == CV_GRAY2BGR565 ? 6 : 5));
            break;
            
        case CV_BGR2YCrCb: case CV_RGB2YCrCb:
        case CV_BGR2YUV: case CV_RGB2YUV:
            {
            CV_Assert( scn == 3 || scn == 4 );
            bidx = code == CV_BGR2YCrCb || code == CV_RGB2YUV ? 0 : 2;
            static const float yuv_f[] = { 0.114f, 0.587f, 0.299f, 0.492f, 0.877f };
            static const int yuv_i[] = { B2Y, G2Y, R2Y, 8061, 14369 };
            const float* coeffs_f = code == CV_BGR2YCrCb || code == CV_RGB2YCrCb ? 0 : yuv_f;
            const int* coeffs_i = code == CV_BGR2YCrCb || code == CV_RGB2YCrCb ? 0 : yuv_i;
                
            dst.create(sz, CV_MAKETYPE(depth, 3));
            
            if( depth == CV_8U )
                CvtColorLoop(src, dst, RGB2YCrCb_i<uchar>(scn, bidx, coeffs_i));
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
            bidx = code == CV_YCrCb2BGR || code == CV_YUV2RGB ? 0 : 2;
            static const float yuv_f[] = { 2.032f, -0.395f, -0.581f, 1.140f };
            static const int yuv_i[] = { 33292, -6472, -9519, 18678 }; 
            const float* coeffs_f = code == CV_YCrCb2BGR || code == CV_YCrCb2RGB ? 0 : yuv_f;
            const int* coeffs_i = code == CV_YCrCb2BGR || code == CV_YCrCb2RGB ? 0 : yuv_i;
            
            dst.create(sz, CV_MAKETYPE(depth, dcn));
            
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
            
            dst.create(sz, CV_MAKETYPE(depth, 3));
            
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
            
            dst.create(sz, CV_MAKETYPE(depth, dcn));
            
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
                code == CV_BGR2HLS || code == CV_RGB2HLS ? 180 : 255;
            
            dst.create(sz, CV_MAKETYPE(depth, 3));
            
            if( code == CV_BGR2HSV || code == CV_RGB2HSV ||
                code == CV_BGR2HSV_FULL || code == CV_RGB2HSV_FULL )
            {
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
            
            dst.create(sz, CV_MAKETYPE(depth, dcn));
            
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
            
            dst.create(sz, CV_MAKETYPE(depth, 3));
            
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
            
            dst.create(sz, CV_MAKETYPE(depth, dcn));
            
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
            
        case CV_BayerBG2BGR: case CV_BayerGB2BGR: case CV_BayerRG2BGR: case CV_BayerGR2BGR:
        case CV_BayerBG2BGR_VNG: case CV_BayerGB2BGR_VNG: case CV_BayerRG2BGR_VNG: case CV_BayerGR2BGR_VNG:
            if(dcn <= 0) dcn = 3;
            CV_Assert( scn == 1 && dcn == 3 && depth == CV_8U );
            dst.create(sz, CV_8UC3);
            
            if( code == CV_BayerBG2BGR || code == CV_BayerGB2BGR ||
                code == CV_BayerRG2BGR || code == CV_BayerGR2BGR )
                Bayer2RGB_8u(src, dst, code);
            else
                Bayer2RGB_VNG_8u(src, dst, code);
            break;
        default:
            CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" );
    }
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


