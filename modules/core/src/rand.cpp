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
//  Filling CvMat/IplImage instances with random numbers
//
// */

#include "precomp.hpp"

namespace cv
{

///////////////////////////// Functions Declaration //////////////////////////////////////

/*
   Multiply-with-carry generator is used here:
   temp = ( A*X(n) + carry )
   X(n+1) = temp mod (2^32)
   carry = temp / (2^32)
*/

#define  RNG_NEXT(x)    ((uint64)(unsigned)(x)*CV_RNG_COEFF + ((x) >> 32))

/***************************************************************************************\
*                           Pseudo-Random Number Generators (PRNGs)                     *
\***************************************************************************************/

template<typename T> static void
randBits_( T* arr, int len, uint64* state, const Vec2i* p, bool small_flag )
{
    uint64 temp = *state;
    int i;

    if( !small_flag )
    {
        for( i = 0; i <= len - 4; i += 4 )
        {
            int t0, t1;

            temp = RNG_NEXT(temp);
            t0 = ((int)temp & p[i][0]) + p[i][1];
            temp = RNG_NEXT(temp);
            t1 = ((int)temp & p[i+1][0]) + p[i+1][1];
            arr[i] = saturate_cast<T>(t0);
            arr[i+1] = saturate_cast<T>(t1);

            temp = RNG_NEXT(temp);
            t0 = ((int)temp & p[i+2][0]) + p[i+2][1];
            temp = RNG_NEXT(temp);
            t1 = ((int)temp & p[i+3][0]) + p[i+3][1];
            arr[i+2] = saturate_cast<T>(t0);
            arr[i+3] = saturate_cast<T>(t1);
        }
    }
    else
    {
        for( i = 0; i <= len - 4; i += 4 )
        {
            int t0, t1, t;
            temp = RNG_NEXT(temp);
            t = (int)temp;
            t0 = (t & p[i][0]) + p[i][1];
            t1 = ((t >> 8) & p[i+1][0]) + p[i+1][1];
            arr[i] = saturate_cast<T>(t0);
            arr[i+1] = saturate_cast<T>(t1);

            t0 = ((t >> 16) & p[i+2][0]) + p[i+2][1];
            t1 = ((t >> 24) & p[i+3][0]) + p[i+3][1];
            arr[i+2] = saturate_cast<T>(t0);
            arr[i+3] = saturate_cast<T>(t1);
        }
    }

    for( ; i < len; i++ )
    {
        int t0;
        temp = RNG_NEXT(temp);

        t0 = ((int)temp & p[i][0]) + p[i][1];
        arr[i] = saturate_cast<T>(t0);
    }

    *state = temp;
}

struct DivStruct
{
    unsigned d;
    unsigned M;
    int sh1, sh2;
    int delta;
};

template<typename T> static void
randi_( T* arr, int len, uint64* state, const DivStruct* p )
{
    uint64 temp = *state;
    int i = 0;
    unsigned t0, t1, v0, v1;

    for( i = 0; i <= len - 4; i += 4 )
    {
        temp = RNG_NEXT(temp);
        t0 = (unsigned)temp;
        temp = RNG_NEXT(temp);
        t1 = (unsigned)temp;
        v0 = (unsigned)(((uint64)t0 * p[i].M) >> 32);
        v1 = (unsigned)(((uint64)t1 * p[i+1].M) >> 32);
        v0 = (v0 + ((t0 - v0) >> p[i].sh1)) >> p[i].sh2;
        v1 = (v1 + ((t1 - v1) >> p[i+1].sh1)) >> p[i+1].sh2;
        v0 = t0 - v0*p[i].d + p[i].delta;
        v1 = t1 - v1*p[i+1].d + p[i+1].delta;
        arr[i] = saturate_cast<T>((int)v0);
        arr[i+1] = saturate_cast<T>((int)v1);
        
        temp = RNG_NEXT(temp);
        t0 = (unsigned)temp;
        temp = RNG_NEXT(temp);
        t1 = (unsigned)temp;
        v0 = (unsigned)(((uint64)t0 * p[i+2].M) >> 32);
        v1 = (unsigned)(((uint64)t1 * p[i+3].M) >> 32);
        v0 = (v0 + ((t0 - v0) >> p[i+2].sh1)) >> p[i+2].sh2;
        v1 = (v1 + ((t1 - v1) >> p[i+3].sh1)) >> p[i+3].sh2;
        v0 = t0 - v0*p[i+2].d + p[i+2].delta;
        v1 = t1 - v1*p[i+3].d + p[i+3].delta;
        arr[i+2] = saturate_cast<T>((int)v0);
        arr[i+3] = saturate_cast<T>((int)v1);
    }

    for( ; i < len; i++ )
    {
        temp = RNG_NEXT(temp);
        t0 = (unsigned)temp;
        v0 = (unsigned)(((uint64)t0 * p[i].M) >> 32);
        v0 = (v0 + ((t0 - v0) >> p[i].sh1)) >> p[i].sh2;
        v0 = t0 - v0*p[i].d + p[i].delta;
        arr[i] = saturate_cast<T>((int)v0);
    }

    *state = temp;
}

    
#define DEF_RANDI_FUNC(suffix, type) \
static void randBits_##suffix(type* arr, int len, uint64* state, \
                              const Vec2i* p, bool small_flag) \
{ randBits_(arr, len, state, p, small_flag); } \
\
static void randi_##suffix(type* arr, int len, uint64* state, \
                           const DivStruct* p, bool ) \
{ randi_(arr, len, state, p); }

DEF_RANDI_FUNC(8u, uchar)
DEF_RANDI_FUNC(8s, schar)
DEF_RANDI_FUNC(16u, ushort)
DEF_RANDI_FUNC(16s, short)
DEF_RANDI_FUNC(32s, int)
    
static void randf_32f( float* arr, int len, uint64* state, const Vec2f* p, bool )
{
    uint64 temp = *state;
    int i;

    for( i = 0; i <= len - 4; i += 4 )
    {
        float f0, f1;

        temp = RNG_NEXT(temp);
        f0 = (int)temp*p[i][0] + p[i][1];
        temp = RNG_NEXT(temp);
        f1 = (int)temp*p[i+1][0] + p[i+1][1];
        arr[i] = f0; arr[i+1] = f1;

        temp = RNG_NEXT(temp);
        f0 = (int)temp*p[i+2][0] + p[i+2][1];
        temp = RNG_NEXT(temp);
        f1 = (int)temp*p[i+3][0] + p[i+3][1];
        arr[i+2] = f0; arr[i+3] = f1;
    }

    for( ; i < len; i++ )
    {
        temp = RNG_NEXT(temp);
        arr[i] = (int)temp*p[i][0] + p[i][1];
    }

    *state = temp;
}


static void
randf_64f( double* arr, int len, uint64* state, const Vec2d* p, bool )
{
    uint64 temp = *state;
    int64 v = 0;
    int i;

    for( i = 0; i <= len - 4; i += 4 )
    {
        double f0, f1;

        temp = RNG_NEXT(temp);
        v = (temp >> 32)|(temp << 32);
        f0 = v*p[i][0] + p[i][1];
        temp = RNG_NEXT(temp);
        v = (temp >> 32)|(temp << 32);
        f1 = v*p[i+1][0] + p[i+1][1];
        arr[i] = f0; arr[i+1] = f1;

        temp = RNG_NEXT(temp);
        v = (temp >> 32)|(temp << 32);
        f0 = v*p[i+2][0] + p[i+2][1];
        temp = RNG_NEXT(temp);
        v = (temp >> 32)|(temp << 32);
        f1 = v*p[i+3][0] + p[i+3][1];
        arr[i+2] = f0; arr[i+3] = f1;
    }

    for( ; i < len; i++ )
    {
        temp = RNG_NEXT(temp);
        v = (temp >> 32)|(temp << 32);
        arr[i] = v*p[i][0] + p[i][1];
    }

    *state = temp;
}

typedef void (*RandFunc)(uchar* arr, int len, uint64* state, const void* p, bool small_flag);    

    
static RandFunc randTab[][8] =
{
    {
        (RandFunc)randi_8u, (RandFunc)randi_8s, (RandFunc)randi_16u, (RandFunc)randi_16s,
        (RandFunc)randi_32s, (RandFunc)randf_32f, (RandFunc)randf_64f, 0
    },
    {
        (RandFunc)randBits_8u, (RandFunc)randBits_8s, (RandFunc)randBits_16u, (RandFunc)randBits_16s,
        (RandFunc)randBits_32s, 0, 0, 0
    }
};    
    
/*
   The code below implements the algorithm described in
   "The Ziggurat Method for Generating Random Variables"
   by Marsaglia and Tsang, Journal of Statistical Software.
*/
static void
randn_0_1_32f( float* arr, int len, uint64* state )
{
    const float r = 3.442620f; // The start of the right tail
    const float rng_flt = 2.3283064365386962890625e-10f; // 2^-32
    static unsigned kn[128];
    static float wn[128], fn[128];
    uint64 temp = *state;
    static bool initialized=false;
    int i;
    
    if( !initialized )
    {
        const double m1 = 2147483648.0;
        double dn = 3.442619855899, tn = dn, vn = 9.91256303526217e-3;
        
        // Set up the tables
        double q = vn/std::exp(-.5*dn*dn);
        kn[0] = (unsigned)((dn/q)*m1);
        kn[1] = 0;
        
        wn[0] = (float)(q/m1);
        wn[127] = (float)(dn/m1);
        
        fn[0] = 1.f;
        fn[127] = (float)std::exp(-.5*dn*dn);
        
        for(i=126;i>=1;i--)
        {
            dn = std::sqrt(-2.*std::log(vn/dn+std::exp(-.5*dn*dn)));
            kn[i+1] = (unsigned)((dn/tn)*m1);
            tn = dn;
            fn[i] = (float)std::exp(-.5*dn*dn);
            wn[i] = (float)(dn/m1);
        }
        initialized = true;
    }
    
    for( i = 0; i < len; i++ )
    {
        float x, y;
        for(;;)
        {
            int hz = (int)temp;
            temp = RNG_NEXT(temp);
            int iz = hz & 127;
            x = hz*wn[iz];
            if( (unsigned)std::abs(hz) < kn[iz] )
                break;
            if( iz == 0) // iz==0, handles the base strip
            {
                do
                {
                    x = (unsigned)temp*rng_flt;
                    temp = RNG_NEXT(temp);
                    y = (unsigned)temp*rng_flt;
                    temp = RNG_NEXT(temp);
                    x = (float)(-std::log(x+FLT_MIN)*0.2904764);
                    y = (float)-std::log(y+FLT_MIN);
                }	// .2904764 is 1/r
                while( y + y < x*x );
                x = hz > 0 ? r + x : -r - x;
                break;
            }
            // iz > 0, handle the wedges of other strips
            y = (unsigned)temp*rng_flt;
            temp = RNG_NEXT(temp);
            if( fn[iz] + y*(fn[iz - 1] - fn[iz]) < std::exp(-.5*x*x) )
                break;
        }
        arr[i] = x;
    }
    *state = temp;
}

    
double RNG::gaussian(double sigma)
{
    float temp;
    randn_0_1_32f( &temp, 1, &state );
    return temp*sigma;
}

    
template<typename T, typename PT> static void
randnScale_( const float* src, T* dst, int len, int cn, const PT* mean, const PT* stddev, bool stdmtx )
{
    int i, j, k;
    if( !stdmtx )
    {
        if( cn == 1 )
        {
            PT b = mean[0], a = stddev[0];
            for( i = 0; i < len; i++ )
                dst[i] = saturate_cast<T>(src[i]*a + b);
        }
        else
        {
            for( i = 0; i < len; i++, src += cn, dst += cn )
                for( k = 0; k < cn; k++ )
                    dst[k] = saturate_cast<T>(src[k]*stddev[k] + mean[k]);
        }
    }
    else
    {
        for( i = 0; i < len; i++, src += cn, dst += cn )
        {
            for( j = 0; j < cn; j++ )
            {
                PT s = mean[j];
                for( k = 0; k < cn; k++ )
                    s += src[k]*stddev[j*cn + k];
                dst[j] = saturate_cast<T>(s);
            }
        }
    }
}
    
static void randnScale_8u( const float* src, uchar* dst, int len, int cn,
                            const float* mean, const float* stddev, bool stdmtx )
{ randnScale_(src, dst, len, cn, mean, stddev, stdmtx); }

static void randnScale_8s( const float* src, schar* dst, int len, int cn,
                            const float* mean, const float* stddev, bool stdmtx )
{ randnScale_(src, dst, len, cn, mean, stddev, stdmtx); }

static void randnScale_16u( const float* src, ushort* dst, int len, int cn,
                             const float* mean, const float* stddev, bool stdmtx )
{ randnScale_(src, dst, len, cn, mean, stddev, stdmtx); }

static void randnScale_16s( const float* src, short* dst, int len, int cn,
                             const float* mean, const float* stddev, bool stdmtx )
{ randnScale_(src, dst, len, cn, mean, stddev, stdmtx); }

static void randnScale_32s( const float* src, int* dst, int len, int cn,
                             const float* mean, const float* stddev, bool stdmtx )
{ randnScale_(src, dst, len, cn, mean, stddev, stdmtx); }

static void randnScale_32f( const float* src, float* dst, int len, int cn,
                             const float* mean, const float* stddev, bool stdmtx )
{ randnScale_(src, dst, len, cn, mean, stddev, stdmtx); }

static void randnScale_64f( const float* src, double* dst, int len, int cn,
                             const double* mean, const double* stddev, bool stdmtx )
{ randnScale_(src, dst, len, cn, mean, stddev, stdmtx); }

typedef void (*RandnScaleFunc)(const float* src, uchar* dst, int len, int cn,
                               const uchar*, const uchar*, bool);

static RandnScaleFunc randnScaleTab[] =
{
    (RandnScaleFunc)randnScale_8u, (RandnScaleFunc)randnScale_8s, (RandnScaleFunc)randnScale_16u,
    (RandnScaleFunc)randnScale_16s, (RandnScaleFunc)randnScale_32s, (RandnScaleFunc)randnScale_32f,
    (RandnScaleFunc)randnScale_64f, 0 
};
    
void RNG::fill( InputOutputArray _mat, int disttype, InputArray _param1arg, InputArray _param2arg )
{
    Mat mat = _mat.getMat(), _param1 = _param1arg.getMat(), _param2 = _param2arg.getMat();
    int depth = mat.depth(), cn = mat.channels();
    AutoBuffer<double> _parambuf;
    int j, k, fast_int_mode = 0, smallFlag = 1;
    RandFunc func = 0;
    RandnScaleFunc scaleFunc = 0;
    
    CV_Assert(_param1.channels() == 1 && (_param1.rows == 1 || _param1.cols == 1) &&
              (_param1.rows + _param1.cols - 1 == cn ||
               (_param1.size() == Size(1, 4) && _param1.type() == CV_64F && cn <= 4)));
    CV_Assert( _param2.channels() == 1 &&
               (((_param2.rows == 1 || _param2.cols == 1) && 
                (_param2.rows + _param2.cols - 1 == cn ||
                (_param1.size() == Size(1, 4) && _param1.type() == CV_64F && cn <= 4))) ||
                (_param2.rows == cn && _param2.cols == cn && disttype == NORMAL)));
    
    Vec2i* ip = 0;
    Vec2d* dp = 0;
    Vec2f* fp = 0;
    DivStruct* ds = 0;
    uchar* mean = 0;
    uchar* stddev = 0;
    bool stdmtx = false;

    if( disttype == UNIFORM )
    {
        _parambuf.allocate(cn*8);
        double* parambuf = _parambuf;
        const double* p1 = (const double*)_param1.data;
        const double* p2 = (const double*)_param2.data;
        
        if( !_param1.isContinuous() || _param1.type() != CV_64F )
        {
            Mat tmp(_param1.size(), CV_64F, parambuf);
            _param1.convertTo(tmp, CV_64F);
            p1 = parambuf;
        }
        
        if( !_param2.isContinuous() || _param2.type() != CV_64F )
        {
            Mat tmp(_param2.size(), CV_64F, parambuf + cn);
            _param2.convertTo(tmp, CV_64F);
            p2 = parambuf + cn;
        }
        
        if( depth <= CV_32S )
        {
            ip = (Vec2i*)(parambuf + cn*2);
            for( j = 0, fast_int_mode = 1; j < cn; j++ )
            {
                double a = min(p1[j], p2[j]);
                double b = max(p1[j], p2[j]);
                ip[j][1] = cvCeil(a);
                int idiff = ip[j][0] = cvFloor(b) - ip[j][1] - 1;
                double diff = b - a;

                fast_int_mode &= diff <= 4294967296. && (idiff & (idiff+1)) == 0;
                if( fast_int_mode )
                    smallFlag &= idiff <= 255;
            }
            
            if( !fast_int_mode )
            {
                ds = (DivStruct*)(ip + cn);
                for( j = 0; j < cn; j++ )
                {
                    ds[j].delta = ip[j][1];
                    unsigned d = ds[j].d = (unsigned)(ip[j][0]+1);
                    int l = 0;
                    while(((uint64)1 << l) < d)
                        l++;
                    ds[j].M = (unsigned)(((uint64)1 << 32)*(((uint64)1 << l) - d)/d) + 1;
                    ds[j].sh1 = min(l, 1);
                    ds[j].sh2 = max(l - 1, 0);
                }            
            }
            
            func = randTab[fast_int_mode][depth];
        }
        else
        {
            double scale = depth == CV_64F ?
                5.4210108624275221700372640043497e-20 : // 2**-64
                2.3283064365386962890625e-10;           // 2**-32

            // for each channel i compute such dparam[0][i] & dparam[1][i],
            // so that a signed 32/64-bit integer X is transformed to
            // the range [param1.val[i], param2.val[i]) using
            // dparam[1][i]*X + dparam[0][i]
            if( depth == CV_32F )
            {
                fp = (Vec2f*)(parambuf + cn*2);
                for( j = 0; j < cn; j++ )
                {
                    fp[j][0] = (float)((p2[j] - p1[j])*scale);
                    fp[j][1] = (float)((p2[j] + p1[j])*0.5);
                }
            }
            else
            {
                dp = (Vec2d*)(parambuf + cn*2);
                for( j = 0; j < cn; j++ )
                {
                    dp[j][0] = ((p2[j] - p1[j])*scale);
                    dp[j][1] = ((p2[j] + p1[j])*0.5);
                }
            }
            
            func = randTab[0][depth];                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        }
        CV_Assert( func != 0 );
    }
    else if( disttype == CV_RAND_NORMAL )
    {
        _parambuf.allocate(_param1.total() + _param2.total());
        double* parambuf = _parambuf;
        
        int ptype = depth == CV_64F ? CV_64F : CV_32F;
        if( _param1.isContinuous() && _param1.type() == ptype )
            mean = _param1.data;
        else
        {
            Mat tmp(_param1.size(), ptype, parambuf);
            _param1.convertTo(tmp, ptype);
            mean = (uchar*)parambuf;
        }
        
        if( _param2.isContinuous() && _param2.type() == ptype )
            stddev = _param2.data;
        else
        {
            Mat tmp(_param2.size(), ptype, parambuf + cn);
            _param2.convertTo(tmp, ptype);
            stddev = (uchar*)(parambuf + cn);
        }
        
        stdmtx = _param2.rows == cn && _param2.cols == cn;
        scaleFunc = randnScaleTab[depth];
        CV_Assert( scaleFunc != 0 );
    }
    else
        CV_Error( CV_StsBadArg, "Unknown distribution type" );

    const Mat* arrays[] = {&mat, 0};
    uchar* ptr;
    NAryMatIterator it(arrays, &ptr);
    int total = (int)it.size, blockSize = std::min((BLOCK_SIZE + cn - 1)/cn, total);
    size_t esz = mat.elemSize();
    AutoBuffer<double> buf;
    uchar* param = 0;
    float* nbuf = 0;
    
    if( disttype == UNIFORM )
    {
        buf.allocate(blockSize*cn*4);
        param = (uchar*)(double*)buf;
        
        if( ip )
        {
            if( ds )
            {
                DivStruct* p = (DivStruct*)param;
                for( j = 0; j < blockSize*cn; j += cn )
                    for( k = 0; k < cn; k++ )
                        p[j + k] = ds[k];
            }
            else
            {
                Vec2i* p = (Vec2i*)param;
                for( j = 0; j < blockSize*cn; j += cn )
                    for( k = 0; k < cn; k++ )
                        p[j + k] = ip[k];
            }
        }
        else if( fp )
        {
            Vec2f* p = (Vec2f*)param;
            for( j = 0; j < blockSize*cn; j += cn )
                for( k = 0; k < cn; k++ )
                    p[j + k] = fp[k];
        }
        else
        {
            Vec2d* p = (Vec2d*)param;
            for( j = 0; j < blockSize*cn; j += cn )
                for( k = 0; k < cn; k++ )
                    p[j + k] = dp[k];
        }
    }
    else
    {
        buf.allocate((blockSize*cn+1)/2);
        nbuf = (float*)(double*)buf;
    }
    
    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int len = std::min(total - j, blockSize);
            
            if( disttype == CV_RAND_UNI )
                func( ptr, len*cn, &state, param, smallFlag != 0 );
            else
            {
                randn_0_1_32f(nbuf, len*cn, &state);
                scaleFunc(nbuf, ptr, len, cn, mean, stddev, stdmtx);
            }
            ptr += len*esz;
        }
    }
}

#ifdef WIN32
#ifdef WINCE
#	define TLS_OUT_OF_INDEXES ((DWORD)0xFFFFFFFF)
#endif
static DWORD tlsRNGKey = TLS_OUT_OF_INDEXES;

void deleteThreadRNGData()
{
    if( tlsRNGKey != TLS_OUT_OF_INDEXES )
        delete (RNG*)TlsGetValue( tlsRNGKey );
}

RNG& theRNG()
{
    if( tlsRNGKey == TLS_OUT_OF_INDEXES )
    {
        tlsRNGKey = TlsAlloc();
        CV_Assert(tlsRNGKey != TLS_OUT_OF_INDEXES);
    }
    RNG* rng = (RNG*)TlsGetValue( tlsRNGKey );
    if( !rng )
    {
        rng = new RNG;
        TlsSetValue( tlsRNGKey, rng );
    }
    return *rng;
}

#else

static pthread_key_t tlsRNGKey = 0;
static pthread_once_t tlsRNGKeyOnce = PTHREAD_ONCE_INIT;

static void deleteRNG(void* data)
{
    delete (RNG*)data;
}

static void makeRNGKey()
{
	int errcode = pthread_key_create(&tlsRNGKey, deleteRNG);
	CV_Assert(errcode == 0);
}

RNG& theRNG()
{
    pthread_once(&tlsRNGKeyOnce, makeRNGKey);
    RNG* rng = (RNG*)pthread_getspecific(tlsRNGKey);
    if( !rng )
    {
        rng = new RNG;
        pthread_setspecific(tlsRNGKey, rng);
    }
    return *rng;
}

#endif

}
    
void cv::randu(InputOutputArray dst, InputArray low, InputArray high)
{
    theRNG().fill(dst, RNG::UNIFORM, low, high);
}

void cv::randn(InputOutputArray dst, InputArray mean, InputArray stddev)
{
    theRNG().fill(dst, RNG::NORMAL, mean, stddev);
}    
 
namespace cv
{

template<typename T> static void
randShuffle_( Mat& _arr, RNG& rng, double iterFactor )
{
    int sz = _arr.rows*_arr.cols, iters = cvRound(iterFactor*sz);
    if( _arr.isContinuous() )
    {
        T* arr = (T*)_arr.data;
        for( int i = 0; i < iters; i++ )
        {
            int j = (unsigned)rng % sz, k = (unsigned)rng % sz;
            std::swap( arr[j], arr[k] );
        }
    }
    else
    {
        uchar* data = _arr.data;
        size_t step = _arr.step;
        int cols = _arr.cols;
        for( int i = 0; i < iters; i++ )
        {
            int j1 = (unsigned)rng % sz, k1 = (unsigned)rng % sz;
            int j0 = j1/cols, k0 = k1/cols;
            j1 -= j0*cols; k1 -= k0*cols;
            std::swap( ((T*)(data + step*j0))[j1], ((T*)(data + step*k0))[k1] );
        }
    }
}

typedef void (*RandShuffleFunc)( Mat& dst, RNG& rng, double iterFactor );

}
    
void cv::randShuffle( InputOutputArray _dst, double iterFactor, RNG* _rng )
{
    RandShuffleFunc tab[] =
    {
        0,
        randShuffle_<uchar>, // 1
        randShuffle_<ushort>, // 2
        randShuffle_<Vec<uchar,3> >, // 3
        randShuffle_<int>, // 4
        0,
        randShuffle_<Vec<ushort,3> >, // 6
        0,
        randShuffle_<Vec<int,2> >, // 8
        0, 0, 0,
        randShuffle_<Vec<int,3> >, // 12
        0, 0, 0,
        randShuffle_<Vec<int,4> >, // 16
        0, 0, 0, 0, 0, 0, 0,
        randShuffle_<Vec<int,6> >, // 24
        0, 0, 0, 0, 0, 0, 0,
        randShuffle_<Vec<int,8> > // 32
    };
    
    Mat dst = _dst.getMat();
    RNG& rng = _rng ? *_rng : theRNG();
    CV_Assert( dst.elemSize() <= 32 );
    RandShuffleFunc func = tab[dst.elemSize()];
    CV_Assert( func != 0 );
    func( dst, rng, iterFactor );
}

CV_IMPL void
cvRandArr( CvRNG* _rng, CvArr* arr, int disttype, CvScalar param1, CvScalar param2 )
{
    cv::Mat mat = cv::cvarrToMat(arr);
    // !!! this will only work for current 64-bit MWC RNG !!!
    cv::RNG& rng = _rng ? (cv::RNG&)*_rng : cv::theRNG();
    rng.fill(mat, disttype == CV_RAND_NORMAL ?
        cv::RNG::NORMAL : cv::RNG::UNIFORM, (cv::Scalar&)param1, (cv::Scalar&)param2 );
}

CV_IMPL void cvRandShuffle( CvArr* arr, CvRNG* _rng, double iter_factor )
{
    cv::Mat dst = cv::cvarrToMat(arr);
    cv::RNG& rng = _rng ? (cv::RNG&)*_rng : cv::theRNG();
    cv::randShuffle( dst, iter_factor, &rng );
}

/* End of file. */
