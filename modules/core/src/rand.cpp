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
    for( int i = 0; i < len; i++ )
    {
        temp = RNG_NEXT(temp);
        unsigned t = (unsigned)temp;
        unsigned v = (unsigned)(((uint64)t * p[i].M) >> 32);
        v = (v + ((t - v) >> p[i].sh1)) >> p[i].sh2;
        v = t - v*p[i].d + p[i].delta;
        arr[i] = saturate_cast<T>((int)v);
    }
    *state = temp;
}


#define DEF_RANDI_FUNC(suffix, type) \
static void randBits_##suffix(type* arr, int len, uint64* state, \
                              const Vec2i* p, void*, bool small_flag) \
{ randBits_(arr, len, state, p, small_flag); } \
\
static void randi_##suffix(type* arr, int len, uint64* state, \
                           const DivStruct* p, void*, bool ) \
{ randi_(arr, len, state, p); }

DEF_RANDI_FUNC(8u, uchar)
DEF_RANDI_FUNC(8s, schar)
DEF_RANDI_FUNC(16u, ushort)
DEF_RANDI_FUNC(16s, short)
DEF_RANDI_FUNC(32s, int)

static void randf_32f( float* arr, int len, uint64* state, const Vec2f* p, void*, bool )
{
    uint64 temp = *state;
    for( int i = 0; i < len; i++ )
    {
        int t = (int)(temp = RNG_NEXT(temp));
        arr[i] = (float)(t*p[i][0]);
    }
    *state = temp;

    // add bias separately to make the generated random numbers
    // more deterministic, independent of
    // architecture details (FMA instruction use etc.)
    hal::addRNGBias32f(arr, &p[0][0], len);
}

static void
randf_64f( double* arr, int len, uint64* state, const Vec2d* p, void*, bool )
{
    uint64 temp = *state;
    for( int i = 0; i < len; i++ )
    {
        temp = RNG_NEXT(temp);
        int64 v = (temp >> 32)|(temp << 32);
        arr[i] = v*p[i][0];
    }
    *state = temp;

    hal::addRNGBias64f(arr, &p[0][0], len);
}

static void randf_16f( hfloat* arr, int len, uint64* state, const Vec2f* p, float* fbuf, bool )
{
    uint64 temp = *state;
    for( int i = 0; i < len; i++ )
    {
        float f = (float)(int)(temp = RNG_NEXT(temp));
        fbuf[i] = f*p[i][0];
    }
    *state = temp;

    // add bias separately to make the generated random numbers
    // more deterministic, independent of
    // architecture details (FMA instruction use etc.)
    hal::addRNGBias32f(fbuf, &p[0][0], len);
    hal::cvt32f16f(fbuf, arr, len);
}

typedef void (*RandFunc)(uchar* arr, int len, uint64* state, const void* p, void* tempbuf, bool small_flag);


static RandFunc randTab[CV_DEPTH_MAX][CV_DEPTH_MAX] =
{
    {
        (RandFunc)randi_8u, (RandFunc)randi_8s, (RandFunc)randi_16u, (RandFunc)randi_16s,
        (RandFunc)randi_32s, (RandFunc)randf_32f, (RandFunc)randf_64f, (RandFunc)randf_16f
    },
    {
        (RandFunc)randBits_8u, (RandFunc)randBits_8s, (RandFunc)randBits_16u, (RandFunc)randBits_16s,
        (RandFunc)randBits_32s, 0, 0, 0
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

void RNG::fill( InputOutputArray _mat, int disttype,
                InputArray _param1arg, InputArray _param2arg, bool saturateRange )
{
    CV_Assert(!_mat.empty());

    Mat mat = _mat.getMat(), _param1 = _param1arg.getMat(), _param2 = _param2arg.getMat();
    int depth = mat.depth(), cn = mat.channels();
    AutoBuffer<double> _parambuf;
    int j, k;
    bool fast_int_mode = false;
    bool smallFlag = true;
    RandFunc func = 0;
    RandnScaleFunc scaleFunc = 0;

    CV_Assert(_param1.channels() == 1 && (_param1.rows == 1 || _param1.cols == 1) &&
              (_param1.rows + _param1.cols - 1 == cn || _param1.rows + _param1.cols - 1 == 1 ||
               (_param1.size() == Size(1, 4) && _param1.type() == CV_64F && cn <= 4)));
    CV_Assert( _param2.channels() == 1 &&
               (((_param2.rows == 1 || _param2.cols == 1) &&
                (_param2.rows + _param2.cols - 1 == cn || _param2.rows + _param2.cols - 1 == 1 ||
                (_param1.size() == Size(1, 4) && _param1.type() == CV_64F && cn <= 4))) ||
                (_param2.rows == cn && _param2.cols == cn && disttype == RNG::NORMAL)));

    Vec2i* ip = 0;
    Vec2d* dp = 0;
    Vec2f* fp = 0;
    DivStruct* ds = 0;
    uchar* mean = 0;
    uchar* stddev = 0;
    bool stdmtx = false;
    int n1 = (int)_param1.total();
    int n2 = (int)_param2.total();

    if( disttype == RNG::UNIFORM )
    {
        _parambuf.allocate(cn*8 + n1 + n2);
        double* parambuf = _parambuf.data();
        double* p1 = _param1.ptr<double>();
        double* p2 = _param2.ptr<double>();

        if( !_param1.isContinuous() || _param1.type() != CV_64F || n1 != cn )
        {
            Mat tmp(_param1.size(), CV_64F, parambuf);
            _param1.convertTo(tmp, CV_64F);
            p1 = parambuf;
            if( n1 < cn )
                for( j = n1; j < cn; j++ )
                    p1[j] = p1[j-n1];
        }

        if( !_param2.isContinuous() || _param2.type() != CV_64F || n2 != cn )
        {
            Mat tmp(_param2.size(), CV_64F, parambuf + cn);
            _param2.convertTo(tmp, CV_64F);
            p2 = parambuf + cn;
            if( n2 < cn )
                for( j = n2; j < cn; j++ )
                    p2[j] = p2[j-n2];
        }

        if( depth <= CV_32S )
        {
            ip = (Vec2i*)(parambuf + cn*2);
            for( j = 0, fast_int_mode = true; j < cn; j++ )
            {
                double a = std::min(p1[j], p2[j]);
                double b = std::max(p1[j], p2[j]);
                if( saturateRange )
                {
                    a = std::max(a, depth == CV_8U || depth == CV_16U ? 0. :
                            depth == CV_8S ? -128. : depth == CV_16S ? -32768. : (double)INT_MIN);
                    b = std::min(b, depth == CV_8U ? 256. : depth == CV_16U ? 65536. :
                            depth == CV_8S ? 128. : depth == CV_16S ? 32768. : (double)INT_MAX);
                }
                ip[j][1] = cvCeil(a);
                int idiff = ip[j][0] = cvFloor(b) - ip[j][1] - 1;
                if (idiff < 0)
                {
                    idiff = 0;
                    ip[j][0] = 0;
                }
                double diff = b - a;

                fast_int_mode = fast_int_mode && diff <= 4294967296. && (idiff & (idiff+1)) == 0;
                if( fast_int_mode )
                    smallFlag = smallFlag && (idiff <= 255);
                else
                {
                    if( diff > INT_MAX )
                        ip[j][0] = INT_MAX;
                    if( a < INT_MIN/2 )
                        ip[j][1] = INT_MIN/2;
                }
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
                    ds[j].sh1 = std::min(l, 1);
                    ds[j].sh2 = std::max(l - 1, 0);
                }
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
            if( depth != CV_64F )
            {
                fp = (Vec2f*)(parambuf + cn*2);
                for( j = 0; j < cn; j++ )
                {
                    fp[j][0] = (float)(std::min(maxdiff, p2[j] - p1[j])*scale);
                    fp[j][1] = (float)((p2[j] + p1[j])*0.5);
                }
            }
            else
            {
                dp = (Vec2d*)(parambuf + cn*2);
                for( j = 0; j < cn; j++ )
                {
                    dp[j][0] = std::min(DBL_MAX, p2[j] - p1[j])*scale;
                    dp[j][1] = ((p2[j] + p1[j])*0.5);
                }
            }

            func = randTab[0][depth];
        }
        CV_Assert( func != 0 );
    }
    else if( disttype == RNG::NORMAL )
    {
        _parambuf.allocate(MAX(n1, cn) + MAX(n2, cn));
        double* parambuf = _parambuf.data();

        int ptype = depth == CV_64F ? CV_64F : CV_32F;
        int esz = (int)CV_ELEM_SIZE(ptype);

        if( _param1.isContinuous() && _param1.type() == ptype && n1 >= cn)
            mean = _param1.ptr();
        else
        {
            Mat tmp(_param1.size(), ptype, parambuf);
            _param1.convertTo(tmp, ptype);
            mean = (uchar*)parambuf;
        }

        if( n1 < cn )
            for( j = n1*esz; j < cn*esz; j++ )
                mean[j] = mean[j - n1*esz];

        if( _param2.isContinuous() && _param2.type() == ptype && n2 >= cn)
            stddev = _param2.ptr();
        else
        {
            Mat tmp(_param2.size(), ptype, parambuf + MAX(n1, cn));
            _param2.convertTo(tmp, ptype);
            stddev = (uchar*)(parambuf + MAX(n1, cn));
        }

        if( n2 < cn )
            for( j = n2*esz; j < cn*esz; j++ )
                stddev[j] = stddev[j - n2*esz];

        stdmtx = _param2.rows == cn && _param2.cols == cn;
        scaleFunc = randnScaleTab[depth];
        CV_Assert( scaleFunc != 0 );
    }
    else
        CV_Error( cv::Error::StsBadArg, "Unknown distribution type" );

    const Mat* arrays[] = {&mat, 0};
    uchar* ptr;
    NAryMatIterator it(arrays, &ptr, 1);
    int total = (int)it.size, blockSize = std::min((BLOCK_SIZE + cn - 1)/cn, total);
    size_t esz = mat.elemSize();
    AutoBuffer<double> buf;
    uchar* param = 0;
    float* nbuf = 0;
    float* tmpbuf = 0;

    if( disttype == RNG::UNIFORM )
    {
        buf.allocate(blockSize*cn*4);
        param = (uchar*)(double*)buf.data();

        if( depth <= CV_32S )
        {
            if( !fast_int_mode )
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
        else if( depth != CV_64F )
        {
            Vec2f* p = (Vec2f*)param;
            for( j = 0; j < blockSize*cn; j += cn )
                for( k = 0; k < cn; k++ )
                    p[j + k] = fp[k];
            if( depth == CV_16F )
                tmpbuf = (float*)p + blockSize*cn*2;
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
        nbuf = (float*)(double*)buf.data();
    }

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int len = std::min(total - j, blockSize);

            if( disttype == RNG::UNIFORM )
                func( ptr, len*cn, &state, param, tmpbuf, smallFlag );
            else
            {
                randn_0_1_32f(nbuf, len*cn, &state);
                scaleFunc(nbuf, ptr, len, cn, mean, stddev, stdmtx);
            }
            ptr += len*esz;
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


#ifndef OPENCV_EXCLUDE_C_API

// Related with https://github.com/opencv/opencv/issues/26258
// To suppress cast-user-defined warning for casting CvRNG to cv::RNG& with GCC14.
// ( CvRNG is uint64, and cv::RNG has only status member which is uint64. )

#if defined(__GNUC__) && __GNUC__ >= 14
#define CV_IGNORE_CAST_USER_DEFINED_WARNING
#endif

CV_IMPL void
cvRandArr( CvRNG* _rng, CvArr* arr, int disttype, CvScalar param1, CvScalar param2 )
{
    cv::Mat mat = cv::cvarrToMat(arr);

#ifdef CV_IGNORE_CAST_USER_DEFINED_WARNING
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-user-defined"
#endif

    // !!! this will only work for current 64-bit MWC RNG !!!
    cv::RNG& rng = _rng ? (cv::RNG&)*_rng : cv::theRNG();

#ifdef CV_IGNORE_CAST_USER_DEFINED_WARNING
#pragma GCC diagnostic pop
#endif

    rng.fill(mat, disttype == CV_RAND_NORMAL ?
        cv::RNG::NORMAL : cv::RNG::UNIFORM, cv::Scalar(param1), cv::Scalar(param2) );
}

CV_IMPL void cvRandShuffle( CvArr* arr, CvRNG* _rng, double iter_factor )
{
    cv::Mat dst = cv::cvarrToMat(arr);

#ifdef CV_IGNORE_CAST_USER_DEFINED_WARNING
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-user-defined"
#endif

    cv::RNG& rng = _rng ? (cv::RNG&)*_rng : cv::theRNG();

#ifdef CV_IGNORE_CAST_USER_DEFINED_WARNING
#pragma GCC diagnostic pop
#endif

    cv::randShuffle( dst, iter_factor, &rng );
}

#ifdef CV_IGNORE_CAST_USER_DEFINED_WARNING
#undef CV_IGNORE_CAST_USER_DEFINED_WARNING
#endif

#endif  // OPENCV_EXCLUDE_C_API


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
