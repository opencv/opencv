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

namespace cv
{

///////////////////////////// Functions Declaration //////////////////////////////////////

#define CV_RNG_COEFF 4164903690U

/*
   Multiply-with-carry generator is used here:
   temp = ( A*X(n) + carry )
   X(n+1) = temp mod (2^32)
   carry = floor (temp / (2^32))
*/

#define  RNG_NEXT(x)    ((uint64)(unsigned)(x)*CV_RNG_COEFF + ((x) >> 32))
// make it jump-less
#define  CN_NEXT(k)     (((k) + 1) & (((k) >= cn) - 1))

enum
{
    RNG_FLAG_SMALL = 0x40000000,
    RNG_FLAG_STDMTX = 0x80000000
};

/***************************************************************************************\
*                           Pseudo-Random Number Generators (PRNGs)                     *
\***************************************************************************************/

template<typename T> static void
randBits_( T* arr, int len, int cn, uint64* state, const Vec2l* p, int flags )
{
    bool small_flag = (flags & RNG_FLAG_SMALL) != 0;
    uint64 temp = *state;
    int i, k = 0;
    len *= cn;
    --cn;

    if( !small_flag )
    {
        for( i = 0; i <= len - 4; i += 4 )
        {
            int64_t t0, t1;

            temp = RNG_NEXT(temp);
            t0 = ((int64_t)temp & p[k][0]) + p[k][1];
            k = CN_NEXT(k);
            temp = RNG_NEXT(temp);
            t1 = ((int64_t)temp & p[k][0]) + p[k][1];
            k = CN_NEXT(k);
            arr[i] = saturate_cast<T>(t0);
            arr[i+1] = saturate_cast<T>(t1);

            temp = RNG_NEXT(temp);
            t0 = ((int64_t)temp & p[k][0]) + p[k][1];
            k = CN_NEXT(k);
            temp = RNG_NEXT(temp);
            t1 = ((int64_t)temp & p[k][0]) + p[k][1];
            k = CN_NEXT(k);
            arr[i+2] = saturate_cast<T>(t0);
            arr[i+3] = saturate_cast<T>(t1);
        }
    }
    else
    {
        for( i = 0; i <= len - 4; i += 4 )
        {
            int64_t t0, t1, t;
            temp = RNG_NEXT(temp);
            t = temp;
            // p[i+...][0] is within 0..255 in this branch (small_flag==true),
            // so we don't need to do (t>>...)&255,
            // the upper bits will be cleaned with ... & p[i+...][0].
            t0 = (t & p[k][0]) + p[k][1];
            k = CN_NEXT(k);
            t1 = ((t >> 8) & p[k][0]) + p[k][1];
            k = CN_NEXT(k);
            arr[i] = saturate_cast<T>(t0);
            arr[i+1] = saturate_cast<T>(t1);

            t0 = ((t >> 16) & p[k][0]) + p[k][1];
            k = CN_NEXT(k);
            t1 = ((t >> 24) & p[k][0]) + p[k][1];
            k = CN_NEXT(k);
            arr[i+2] = saturate_cast<T>(t0);
            arr[i+3] = saturate_cast<T>(t1);
        }
    }

    for( ; i < len; i++ )
    {
        int64_t t0;
        temp = RNG_NEXT(temp);

        t0 = ((int64_t)temp & p[k][0]) + p[k][1];
        k = CN_NEXT(k);
        arr[i] = saturate_cast<T>(t0);
    }

    *state = temp;
}

struct DivStruct
{
    unsigned d;
    unsigned M;
    int sh1, sh2;
    int64_t delta;
    uint64_t diff;
};

template<typename T> static void
randi_( T* arr, int len, int cn, uint64* state, const DivStruct* p )
{
    uint64 temp = *state;
    int k = 0;
    len *= cn;
    cn--;
    for( int i = 0; i < len; i++ )
    {
        temp = RNG_NEXT(temp);
        unsigned t = (unsigned)temp;
        unsigned v = (unsigned)(((uint64)t * p[k].M) >> 32);
        v = (v + ((t - v) >> p[k].sh1)) >> p[k].sh2;
        int64_t res = (int64_t)(t - v*p[k].d) + p[k].delta;
        k = CN_NEXT(k);
        arr[i] = saturate_cast<T>(res);
    }
    *state = temp;
}

static void
randi_( int64_t* arr, int len, int cn, uint64* state, const DivStruct* p )
{
    uint64 temp = *state;
    int k = 0;
    len *= cn;
    cn--;
    for( int i = 0; i < len; i++ )
    {
        temp = RNG_NEXT(temp);
        unsigned t0 = (unsigned)temp;
        temp = RNG_NEXT(temp);
        unsigned t1 = (unsigned)temp;
        int64_t t = (int64_t)((((uint64_t)t0 << 32) | t1) % p[k].diff) + p[k].delta;
        k = CN_NEXT(k);
        arr[i] = t;
    }
    *state = temp;
}

static void
randi_( uint64_t* arr, int len, int cn, uint64* state, const DivStruct* p )
{
    uint64 temp = *state;
    int k = 0;
    len *= cn;
    cn--;
    for( int i = 0; i < len; i++ )
    {
        temp = RNG_NEXT(temp);
        unsigned t0 = (unsigned)temp;
        temp = RNG_NEXT(temp);
        unsigned t1 = (unsigned)temp;
        uint64_t t = (((uint64_t)t0 << 32) | t1) % p[k].diff;
        int64_t delta = p[k].delta;
        k = CN_NEXT(k);
        arr[i] = delta >= 0 || t >= (uint64_t)-delta ? t + (uint64_t)delta : 0;
    }
    *state = temp;
}

#define DEF_RANDI_FUNC(suffix, type) \
static void randBits_##suffix(type* arr, int len, int cn, uint64* state, \
                              const Vec2l* p, void*, int flags) \
{ randBits_(arr, len, cn, state, p, flags); } \
\
static void randi_##suffix(type* arr, int len, int cn, uint64* state, \
                           const DivStruct* p, void*, int) \
{ randi_(arr, len, cn, state, p); }

DEF_RANDI_FUNC(8u, uchar)
DEF_RANDI_FUNC(8b, bool)
DEF_RANDI_FUNC(8s, schar)
DEF_RANDI_FUNC(16u, ushort)
DEF_RANDI_FUNC(16s, short)
DEF_RANDI_FUNC(32u, unsigned)
DEF_RANDI_FUNC(32s, int)
DEF_RANDI_FUNC(64u, uint64_t)
DEF_RANDI_FUNC(64s, int64_t)

static void randf_16_or_32f( void* dst, int len_, int cn, uint64* state, const Vec2f* p, float* fbuf, int flags )
{
    int depth = CV_MAT_DEPTH(flags);
    uint64 temp = *state;
    int k = 0, len = len_*cn;
    float* arr = depth == CV_16F || depth == CV_16BF ? fbuf : (float*)dst;
    cn--;
    for( int i = 0; i < len; i++ )
    {
        int t = (int)(temp = RNG_NEXT(temp));
        arr[i] = (float)(t*p[k][0]);
        k = CN_NEXT(k);
    }
    *state = temp;
    hal::addRNGBias32f(arr, &p[0][0], len_, cn+1);
    if (depth == CV_16F)
        hal::cvt32f16f(fbuf, (hfloat*)dst, len);
    else if (depth == CV_16BF)
        hal::cvt32f16bf(fbuf, (bfloat*)dst, len);
}

static void
randf_64f( double* arr, int len_, int cn, uint64* state, const Vec2d* p, void*, int )
{
    uint64 temp = *state;
    int k = 0, len = len_*cn;
    cn--;
    for( int i = 0; i < len; i++ )
    {
        temp = RNG_NEXT(temp);
        int64_t v = (int64_t)((temp >> 32) | (temp << 32));
        arr[i] = v*p[k][0];
        k = CN_NEXT(k);
    }
    *state = temp;
    hal::addRNGBias64f(arr, &p[0][0], len_, cn+1);
}

typedef void (*RandFunc)(uchar* arr, int len, int cn, uint64* state,
                         const void* p, void* tempbuf, int flags);

static RandFunc randTab[CV_DEPTH_MAX][CV_DEPTH_MAX] =
{
    {
        (RandFunc)randi_8u, (RandFunc)randi_8s, (RandFunc)randi_16u,
        (RandFunc)randi_16s, (RandFunc)randi_32s, (RandFunc)randf_16_or_32f,
        (RandFunc)randf_64f, (RandFunc)randf_16_or_32f, (RandFunc)randf_16_or_32f,
        (RandFunc)randi_8b, (RandFunc)randi_64u, (RandFunc)randi_64s,
        (RandFunc)randi_32u, 0, 0, 0
    },
    {
        (RandFunc)randBits_8u, (RandFunc)randBits_8s, (RandFunc)randBits_16u,
        (RandFunc)randBits_16s, (RandFunc)randBits_32s, 0, 0, 0, 0,
        (RandFunc)randBits_8b, (RandFunc)randBits_64u, (RandFunc)randBits_64s,
        (RandFunc)randBits_32u, 0, 0, 0
    }
};

/*
   The code below implements the algorithm described in
   "The Ziggurat Method for Generating Random Variables"
   by George Marsaglia and Wai Wan Tsang, Journal of Statistical Software, 2007.
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
randnScale_(float* src, T* dst, int len, int cn,
            const PT* mean, const PT* stddev, int flags )
{
    bool stdmtx = (flags & RNG_FLAG_STDMTX) != 0;
    int i, j, k;
    if( !stdmtx || cn == 1 )
    {
        if( cn == 1 )
        {
            PT a = stddev[0], b = mean[0];
            for( i = 0; i < len; i++ )
                dst[i] = saturate_cast<T>(src[i]*a + b);
        }
        else
        {
            len *= cn;
            cn--;
            for( i = k = 0; i < len; i++ ) {
                dst[i] = saturate_cast<T>(src[i]*stddev[k] + mean[k]);
                k = CN_NEXT(k);
            }
        }
    }
    else
    {
        len *= cn;
        cn--;
        for( i = j = 0; i < len; i++ )
        {
            PT s = mean[j];
            int i0 = i - j;
            for( k = 0; k <= cn; k++ )
                s += src[i0 + k]*stddev[j*(cn+1) + k];
            dst[i] = saturate_cast<T>(s);
            j = CN_NEXT(j);
        }
    }
}

// special version for 16f, 16bf and 32f
static void
randnScale_16_or_32f(float* fbuf, float* dst, int len, int cn,
                     const float* mean, const float* stddev, int flags)
{
    bool stdmtx = (flags & RNG_FLAG_STDMTX) != 0;
    int depth = CV_MAT_DEPTH(flags);
    float* arr = depth == CV_16F || depth == CV_16BF ? fbuf : dst;
    int i, j, k;

    if( !stdmtx || cn == 1 )
    {
        if( cn == 1 )
        {
            float a = stddev[0], b = mean[0];
            for( i = 0; i < len; i++ )
                arr[i] = fbuf[i]*a + b;
        }
        else
        {
            len *= cn;
            cn--;
            for( i = k = 0; i < len; i++ ) {
                arr[i] = fbuf[i]*stddev[k] + mean[k];
                k = CN_NEXT(k);
            }
        }
    }
    else if( depth == CV_32F )
    {
        len *= cn;
        cn--;
        for( i = j = 0; i < len; i++ )
        {
            float s = mean[j];
            int i0 = i - j;
            for( k = 0; k <= cn; k++ )
                s += fbuf[i0 + k]*stddev[j*(cn+1) + k];
            dst[i] = s;
            j = CN_NEXT(j);
        }
    }
    else
    {
        float elembuf[CV_CN_MAX];
        len *= cn;
        for( i = 0; i < len; i += cn )
        {
            // since we process fbuf in-place,
            // we need to copy each cn-channel element
            // prior to matrix multiplication
            for (j = 0; j < cn; j++)
                elembuf[j] = fbuf[i + j];
            for (j = 0; j < cn; j++) {
                float s = mean[j];
                for( k = 0; k < cn; k++ )
                    s += elembuf[k]*stddev[j*cn + k];
                fbuf[i + j] = s;
            }
        }
    }
    if (depth == CV_16F)
        hal::cvt32f16f(fbuf, (hfloat*)dst, len);
    else if (depth == CV_16BF)
        hal::cvt32f16bf(fbuf, (bfloat*)dst, len);
}

#define DEF_RANDNSCALE_FUNC(suffix, T, PT) \
static void randnScale_##suffix( float* src, T* dst, int len, int cn, \
                                 const PT* mean, const PT* stddev, int flags ) \
{ randnScale_(src, dst, len, cn, mean, stddev, flags); }

DEF_RANDNSCALE_FUNC(8u, uchar, float)
DEF_RANDNSCALE_FUNC(8b, bool, float)
DEF_RANDNSCALE_FUNC(8s, schar, float)
DEF_RANDNSCALE_FUNC(16u, ushort, float)
DEF_RANDNSCALE_FUNC(16s, short, float)
DEF_RANDNSCALE_FUNC(32u, unsigned, float)
DEF_RANDNSCALE_FUNC(32s, int, float)
DEF_RANDNSCALE_FUNC(64u, uint64_t, double)
DEF_RANDNSCALE_FUNC(64s, int64_t, double)
DEF_RANDNSCALE_FUNC(64f, double, double)

typedef void (*RandnScaleFunc)(float* src, void* dst, int len, int cn,
                               const void* mean, const void* stddev, int flags);

static RandnScaleFunc randnScaleTab[CV_DEPTH_MAX] =
{
    (RandnScaleFunc)randnScale_8u, (RandnScaleFunc)randnScale_8s, (RandnScaleFunc)randnScale_16u,
    (RandnScaleFunc)randnScale_16s, (RandnScaleFunc)randnScale_32s, (RandnScaleFunc)randnScale_16_or_32f,
    (RandnScaleFunc)randnScale_64f, (RandnScaleFunc)randnScale_16_or_32f, (RandnScaleFunc)randnScale_16_or_32f,
    (RandnScaleFunc)randnScale_8b, (RandnScaleFunc)randnScale_64u, (RandnScaleFunc)randnScale_64s,
    (RandnScaleFunc)randnScale_32u, 0, 0, 0
};

void RNG::fill( InputOutputArray _mat, int disttype,
                InputArray _param1arg, InputArray _param2arg,
                bool saturateRange )
{
    CV_Assert(!_mat.empty());

    Mat mat = _mat.getMat(), _param1 = _param1arg.getMat(), _param2 = _param2arg.getMat();
    int j, depth = mat.depth(), cn = mat.channels();
    int esz1 = CV_ELEM_SIZE(depth);
    AutoBuffer<double> _parambuf;
    bool fast_int_mode = false;
    bool small_flag = false;
    RandFunc func = 0;
    RandnScaleFunc scaleFunc = 0;

    CV_Assert(_param1.channels() == 1 && (_param1.rows == 1 || _param1.cols == 1) &&
              (_param1.rows + _param1.cols - 1 == cn || _param1.rows + _param1.cols - 1 == 1 ||
               (_param1.size() == Size(1, 4) && _param1.type() == CV_64F && cn <= 4)));
    CV_Assert( _param2.channels() == 1 &&
               (((_param2.rows == 1 || _param2.cols == 1) &&
                (_param2.rows + _param2.cols - 1 == cn || _param2.rows + _param2.cols - 1 == 1 ||
                (_param1.size() == Size(1, 4) && _param1.type() == CV_64F && cn <= 4))) ||
                (_param2.rows == cn && _param2.cols == cn && disttype == NORMAL)));

    const void* uni_param = 0;
    uchar* mean = 0;
    uchar* stddev = 0;
    bool stdmtx = false;
    int n1 = (int)_param1.total();
    int n2 = (int)_param2.total();

    if( disttype == UNIFORM )
    {
        _parambuf.allocate(cn*(sizeof(DivStruct)+sizeof(double)-1)/sizeof(double) + cn*4);
        double* parambuf = _parambuf.data();
        double* p1 = _param1.ptr<double>();
        double* p2 = _param2.ptr<double>();

        if( !_param1.isContinuous() || _param1.type() != CV_64F || n1 != cn )
        {
            p1 = parambuf;
            Mat tmp(_param1.size(), CV_64F, p1);
            _param1.convertTo(tmp, CV_64F);
            for( j = n1; j < cn; j++ )
                p1[j] = p1[j-n1];
        }

        if( !_param2.isContinuous() || _param2.type() != CV_64F || n2 != cn )
        {
            p2 = parambuf + cn;
            Mat tmp(_param2.size(), CV_64F, p2);
            _param2.convertTo(tmp, CV_64F);
            for( j = n2; j < cn; j++ )
                p2[j] = p2[j-n2];
        }

        if( CV_IS_INT_TYPE(depth) )
        {
            Vec2l* ip = (Vec2l*)(parambuf + cn*2);
            CV_DbgCheckLT((size_t)(cn*4 - 1), _parambuf.size(), "");
            for( j = 0, fast_int_mode = true; j < cn; j++ )
            {
                double a = std::min(p1[j], p2[j]);
                double b = std::max(p1[j], p2[j]);
                if( saturateRange )
                {
                    a = std::max(a, depth == CV_8U || depth == CV_16U || depth == CV_32U ||
                                 depth == CV_64U || depth == CV_Bool ? 0. :
                                 depth == CV_8S ? -128. : depth == CV_16S ? -32768. :
                                 depth == CV_32S ? (double)INT_MIN : (double)INT64_MIN);
                    b = std::min(b, depth == CV_8U ? 256. : depth == CV_Bool ? 2. : depth == CV_16U ? 65536. :
                                 depth == CV_8S ? 128. : depth == CV_16S ? 32768. : depth == CV_32U ? (double)UINT_MAX :
                                 depth == CV_32S ? (double)INT_MAX : (double)INT64_MAX);
                }
                ip[j][1] = (int64_t)ceil(a);
                int64_t idiff = ip[j][0] = (int64_t)floor(b) - ip[j][1] - 1;
                if (idiff < 0)
                {
                    idiff = 0;
                    ip[j][0] = 0;
                }
                double diff = b - a;

                fast_int_mode = fast_int_mode && diff <= 4294967296. && (idiff & (idiff+1)) == 0;
                if( fast_int_mode )
                    small_flag = idiff <= 255;
                else
                {
                    int64_t minval = INT32_MIN/2, maxval = INT32_MAX;
                    if (depth == CV_64S || depth == CV_64U)
                    {
                        minval = INT64_MIN/2;
                        maxval = INT64_MAX;
                    }
                    if( diff > (double)maxval )
                        ip[j][0] = maxval;
                    if( a < (double)minval )
                        ip[j][1] = minval;
                }
            }

            uni_param = ip;
            if( !fast_int_mode )
            {
                DivStruct* ds = (DivStruct*)(ip + cn);
                CV_DbgCheckLE((void*)(ds + cn), (void*)(parambuf + _parambuf.size()), "Last byte check");
                for( j = 0; j < cn; j++ )
                {
                    ds[j].delta = ip[j][1];
                    ds[j].diff = ip[j][0];
                    if (depth != CV_64U && depth != CV_64S) {
                        unsigned d = ds[j].d = (unsigned)(ip[j][0]+1);
                        int l = 0;
                        while(((uint64)1 << l) < d)
                            l++;
                        ds[j].M = (unsigned)(((uint64)1 << 32)*(((uint64)1 << l) - d)/d) + 1;
                        ds[j].sh1 = std::min(l, 1);
                        ds[j].sh2 = std::max(l - 1, 0);
                    }
                }
                uni_param = ds;
            }

            func = randTab[fast_int_mode ? 1 : 0][depth];
        }
        else
        {
            double scale = depth == CV_64F ?
                5.4210108624275221700372640043497e-20 : // 2**-64
                2.3283064365386962890625e-10;           // 2**-32
            double maxdiff = saturateRange ? (double)FLT_MAX : DBL_MAX;

            // for each channel i compute such dparam[0][i] & dparam[1][i],
            // so that a signed 32/64-bit integer X is transformed to
            // the range [param1.val[i], param2.val[i]) using
            // dparam[0][i]*X + dparam[1][i]
            CV_DbgCheckLT((size_t)(cn*4 - 1), _parambuf.size(), "");
            if( depth != CV_64F )
            {
                Vec2f* fp = (Vec2f*)(parambuf + cn*2);
                for( j = 0; j < cn; j++ )
                {
                    fp[j][0] = (float)(std::min(maxdiff, p2[j] - p1[j])*scale);
                    fp[j][1] = (float)((p2[j] + p1[j])*0.5);
                }
                uni_param = fp;
            }
            else
            {
                Vec2d* dp = (Vec2d*)(parambuf + cn*2);
                for( j = 0; j < cn; j++ )
                {
                    dp[j][0] = std::min(DBL_MAX, p2[j] - p1[j])*scale;
                    dp[j][1] = ((p2[j] + p1[j])*0.5);
                }
                uni_param = dp;
            }

            func = randTab[0][depth];
        }
        CV_Assert( func != 0 );
    }
    else if( disttype == RNG::NORMAL )
    {
        _parambuf.allocate(MAX(n1, cn) + MAX(n2, cn));
        double* parambuf = _parambuf.data();

        int ptype = esz1 == 8 ? CV_64F : CV_32F;

        if( _param1.isContinuous() && _param1.type() == ptype && n1 >= cn)
            mean = _param1.ptr();
        else
        {
            Mat tmp(_param1.size(), ptype, parambuf);
            _param1.convertTo(tmp, ptype);
            mean = (uchar*)parambuf;
        }

        if( n1 < cn )
            for( j = n1*esz1; j < cn*esz1; j++ )
                mean[j] = mean[j - n1*esz1];

        if( _param2.isContinuous() && _param2.type() == ptype && n2 >= cn)
            stddev = _param2.ptr();
        else
        {
            Mat tmp(_param2.size(), ptype, parambuf + MAX(n1, cn));
            _param2.convertTo(tmp, ptype);
            stddev = (uchar*)(parambuf + MAX(n1, cn));
        }

        if( n2 < cn )
            for( j = n2*esz1; j < cn*esz1; j++ )
                stddev[j] = stddev[j - n2*esz1];

        stdmtx = _param2.rows == cn && _param2.cols == cn;
        scaleFunc = randnScaleTab[depth];
        CV_Assert( scaleFunc != 0 );
    }
    else
        CV_Error( cv::Error::StsBadArg, "Unknown distribution type" );

    const Mat* arrays[] = {&mat, 0};
    uchar* ptr = 0;
    NAryMatIterator it(arrays, &ptr, 1);
    float fbuf[BLOCK_SIZE + CV_CN_MAX];
    int total = (int)it.size;
    int blockSize = std::min((BLOCK_SIZE + cn - 1)/cn, total);
    size_t esz = (size_t)esz1*cn;
    int flags = mat.type();

    if( disttype == UNIFORM )
        flags |= (small_flag ? (int)RNG_FLAG_SMALL : 0);
    else
        flags |= (stdmtx ? (int)RNG_FLAG_STDMTX : 0);

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int len = std::min(total - j, blockSize);

            if( disttype == UNIFORM )
                func(ptr + j*esz, len, cn, &state, uni_param, fbuf, flags);
            else
            {
                randn_0_1_32f(fbuf, len*cn, &state);
                scaleFunc(fbuf, ptr + j*esz, len, cn, mean, stddev, flags);
            }
        }
    }
}

}

cv::RNG& cv::theRNG()
{
    return getCoreTlsData().rng;
}

void cv::setRNGSeed(int seed)
{
    theRNG() = RNG(static_cast<uint64>(seed));
}


void cv::randu(InputOutputArray dst, InputArray low, InputArray high)
{
    CV_INSTRUMENT_REGION();

    theRNG().fill(dst, RNG::UNIFORM, low, high);
}

void cv::randn(InputOutputArray dst, InputArray mean, InputArray stddev)
{
    CV_INSTRUMENT_REGION();

    theRNG().fill(dst, RNG::NORMAL, mean, stddev);
}

namespace cv
{

template<typename T> static void
randShuffle_( Mat& _arr, RNG& rng, double )
{
    unsigned sz = (unsigned)_arr.total();
    if( _arr.isContinuous() )
    {
        T* arr = _arr.ptr<T>();
        for( unsigned i = 0; i < sz; i++ )
        {
            unsigned j = (unsigned)rng % sz;
            std::swap( arr[j], arr[i] );
        }
    }
    else
    {
        CV_Assert( _arr.dims <= 2 );
        uchar* data = _arr.ptr();
        size_t step = _arr.step;
        int rows = _arr.rows;
        int cols = _arr.cols;
        for( int i0 = 0; i0 < rows; i0++ )
        {
            T* p = _arr.ptr<T>(i0);
            for( int j0 = 0; j0 < cols; j0++ )
            {
                unsigned k1 = (unsigned)rng % sz;
                int i1 = (int)(k1 / cols);
                int j1 = (int)(k1 - (unsigned)i1*(unsigned)cols);
                std::swap( p[j0], ((T*)(data + step*i1))[j1] );
            }
        }
    }
}

typedef void (*RandShuffleFunc)( Mat& dst, RNG& rng, double iterFactor );

}

void cv::randShuffle( InputOutputArray _dst, double iterFactor, RNG* _rng )
{
    CV_INSTRUMENT_REGION();

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


// Mersenne Twister random number generator.
// Inspired by http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/mt19937ar.c

/*
   A C-program for MT19937, with initialization improved 2002/1/26.
   Coded by Takuji Nishimura and Makoto Matsumoto.

   Before using, initialize the state by using init_genrand(seed)
   or init_by_array(init_key, key_length).

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote
        products derived from this software without specific prior written
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
*/

cv::RNG_MT19937::RNG_MT19937(unsigned s) { seed(s); }

cv::RNG_MT19937::RNG_MT19937() { seed(5489U); }

void cv::RNG_MT19937::seed(unsigned s)
{
    state[0]= s;
    for (mti = 1; mti < N; mti++)
    {
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        state[mti] = (1812433253U * (state[mti - 1] ^ (state[mti - 1] >> 30)) + mti);
    }
}

unsigned cv::RNG_MT19937::next()
{
    /* mag01[x] = x * MATRIX_A  for x=0,1 */
    static unsigned mag01[2] = { 0x0U, /*MATRIX_A*/ 0x9908b0dfU};

    const unsigned UPPER_MASK = 0x80000000U;
    const unsigned LOWER_MASK = 0x7fffffffU;

    /* generate N words at one time */
    if (mti >= N)
    {
        int kk = 0;

        for (; kk < N - M; ++kk)
        {
            unsigned y = (state[kk] & UPPER_MASK) | (state[kk + 1] & LOWER_MASK);
            state[kk] = state[kk + M] ^ (y >> 1) ^ mag01[y & 0x1U];
        }

        for (; kk < N - 1; ++kk)
        {
            unsigned y = (state[kk] & UPPER_MASK) | (state[kk + 1] & LOWER_MASK);
            state[kk] = state[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1U];
        }

        unsigned y = (state[N - 1] & UPPER_MASK) | (state[0] & LOWER_MASK);
        state[N - 1] = state[M - 1] ^ (y >> 1) ^ mag01[y & 0x1U];

        mti = 0;
    }

    unsigned y = state[mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y <<  7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);

    return y;
}

cv::RNG_MT19937::operator unsigned() { return next(); }

cv::RNG_MT19937::operator int() { return (int)next();}

cv::RNG_MT19937::operator float() { return next() * (1.f / 4294967296.f); }

cv::RNG_MT19937::operator double()
{
    unsigned a = next() >> 5;
    unsigned b = next() >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

int cv::RNG_MT19937::uniform(int a, int b) { return (int)(next() % (b - a) + a); }

float cv::RNG_MT19937::uniform(float a, float b) { return ((float)*this)*(b - a) + a; }

double cv::RNG_MT19937::uniform(double a, double b) { return ((double)*this)*(b - a) + a; }

unsigned cv::RNG_MT19937::operator ()(unsigned b) { return next() % b; }

unsigned cv::RNG_MT19937::operator ()() { return next(); }

/* End of file. */
