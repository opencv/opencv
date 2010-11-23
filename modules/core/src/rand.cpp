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

#define  RNG_NEXT(x)    ((uint64)(unsigned)(x)*RNG::A + ((x) >> 32))

/***************************************************************************************\
*                           Pseudo-Random Number Generators (PRNGs)                     *
\***************************************************************************************/

template<typename T> static void
RandBits_( Mat& _arr, uint64* state, const void* _param )
{
    uint64 temp = *state;
    const int* param = (const int*)_param;
    int small_flag = (param[12]|param[13]|param[14]|param[15]) <= 255;
    Size size = getContinuousSize(_arr,_arr.channels());

    for( int y = 0; y < size.height; y++ )
    {
        T* arr = (T*)(_arr.data + _arr.step*y);
        int i, k = 3;
        const int* p = param;

        if( !small_flag )
        {
            for( i = 0; i <= size.width - 4; i += 4 )
            {
                int t0, t1;

                temp = RNG_NEXT(temp);
                t0 = ((int)temp & p[i + 12]) + p[i];
                temp = RNG_NEXT(temp);
                t1 = ((int)temp & p[i + 13]) + p[i+1];
                arr[i] = saturate_cast<T>(t0);
                arr[i+1] = saturate_cast<T>(t1);

                temp = RNG_NEXT(temp);
                t0 = ((int)temp & p[i + 14]) + p[i+2];
                temp = RNG_NEXT(temp);
                t1 = ((int)temp & p[i + 15]) + p[i+3];
                arr[i+2] = saturate_cast<T>(t0);
                arr[i+3] = saturate_cast<T>(t1);

                if( !--k )
                {
                    k = 3;
                    p -= 12;
                }
            }
        }
        else
        {
            for( i = 0; i <= size.width - 4; i += 4 )
            {
                int t0, t1, t;

                temp = RNG_NEXT(temp);
                t = (int)temp;
                t0 = (t & p[i + 12]) + p[i];
                t1 = ((t >> 8) & p[i + 13]) + p[i+1];
                arr[i] = saturate_cast<T>(t0);
                arr[i+1] = saturate_cast<T>(t1);

                t0 = ((t >> 16) & p[i + 14]) + p[i + 2];
                t1 = ((t >> 24) & p[i + 15]) + p[i + 3];
                arr[i+2] = saturate_cast<T>(t0);
                arr[i+3] = saturate_cast<T>(t1);

                if( !--k )
                {
                    k = 3;
                    p -= 12;
                }
            }
        }

        for( ; i < size.width; i++ )
        {
            int t0;
            temp = RNG_NEXT(temp);

            t0 = ((int)temp & p[i + 12]) + p[i];
            arr[i] = saturate_cast<T>(t0);
        }
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
Randi_( Mat& _arr, uint64* state, const void* _param )
{
    uint64 temp = *state;
    const int* param = (const int*)_param;
    Size size = getContinuousSize(_arr,_arr.channels());
    int i, k, cn = _arr.channels();
    DivStruct ds[12];
    
    for( k = 0; k < cn; k++ )
    {
        ds[k].delta = param[k];
        ds[k].d = (unsigned)(param[k+12] - param[k]);
        int l = 0;
        while(((uint64)1 << l) < ds[k].d)
            l++;
        ds[k].M = (unsigned)(((uint64)1 << 32)*(((uint64)1 << l) - ds[k].d)/ds[k].d) + 1;
        ds[k].sh1 = min(l, 1);
        ds[k].sh2 = max(l - 1, 0);
    }
    
    for( ; k < 12; k++ )
        ds[k] = ds[k - cn];

    for( int y = 0; y < size.height; y++ )
    {
        T* arr = (T*)(_arr.data + _arr.step*y);
        const DivStruct* p = ds;
        unsigned t0, t1, v0, v1;

        for( i = 0, k = 3; i <= size.width - 4; i += 4 )
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

            if( !--k )
            {
                k = 3;
                p -= 12;
            }
        }

        for( ; i < size.width; i++ )
        {
            temp = RNG_NEXT(temp);
            t0 = (unsigned)temp;
            v0 = (unsigned)(((uint64)t0 * p[i].M) >> 32);
            v0 = (v0 + ((t0 - v0) >> p[i].sh1)) >> p[i].sh2;
            v0 = t0 - v0*p[i].d + p[i].delta;
            arr[i] = saturate_cast<T>((int)v0);
        }
    }

    *state = temp;
}


static void Randf_( Mat& _arr, uint64* state, const void* _param )
{
    uint64 temp = *state;
    const float* param = (const float*)_param;
    Size size = getContinuousSize(_arr,_arr.channels());

    for( int y = 0; y < size.height; y++ )
    {
        float* arr = (float*)(_arr.data + _arr.step*y);
        int i, k = 3;
        const float* p = param;
        for( i = 0; i <= size.width - 4; i += 4 )
        {
            float f0, f1;

            temp = RNG_NEXT(temp);
            f0 = (int)temp*p[i+12] + p[i];
            temp = RNG_NEXT(temp);
            f1 = (int)temp*p[i+13] + p[i+1];
            arr[i] = f0; arr[i+1] = f1;

            temp = RNG_NEXT(temp);
            f0 = (int)temp*p[i+14] + p[i+2];
            temp = RNG_NEXT(temp);
            f1 = (int)temp*p[i+15] + p[i+3];
            arr[i+2] = f0; arr[i+3] = f1;

            if( !--k )
            {
                k = 3;
                p -= 12;
            }
        }

        for( ; i < size.width; i++ )
        {
            temp = RNG_NEXT(temp);
            arr[i] = (int)temp*p[i+12] + p[i];
        }
    }

    *state = temp;
}


static void
Randd_( Mat& _arr, uint64* state, const void* _param )
{
    uint64 temp = *state;
    const double* param = (const double*)_param;
    Size size = getContinuousSize(_arr,_arr.channels());
    int64 v = 0;

    for( int y = 0; y < size.height; y++ )
    {
        double* arr = (double*)(_arr.data + _arr.step*y);
        int i, k = 3;
        const double* p = param;
        
        for( i = 0; i <= size.width - 4; i += 4 )
        {
            double f0, f1;

            temp = RNG_NEXT(temp);
            v = (temp >> 32)|(temp << 32);
            f0 = v*p[i+12] + p[i];
            temp = RNG_NEXT(temp);
            v = (temp >> 32)|(temp << 32);
            f1 = v*p[i+13] + p[i+1];
            arr[i] = f0; arr[i+1] = f1;

            temp = RNG_NEXT(temp);
            v = (temp >> 32)|(temp << 32);
            f0 = v*p[i+14] + p[i+2];
            temp = RNG_NEXT(temp);
            v = (temp >> 32)|(temp << 32);
            f1 = v*p[i+15] + p[i+3];
            arr[i+2] = f0; arr[i+3] = f1;

            if( !--k )
            {
                k = 3;
                p -= 12;
            }
        }

        for( ; i < size.width; i++ )
        {
            temp = RNG_NEXT(temp);
            v = (temp >> 32)|(temp << 32);
            arr[i] = v*p[i+12] + p[i];
        }
    }

    *state = temp;
}

   
/*
   The code below implements the algorithm described in
   "The Ziggurat Method for Generating Random Variables"
   by Marsaglia and Tsang, Journal of Statistical Software.
*/
static void
Randn_0_1_32f_C1R( float* arr, int len, uint64* state )
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
    Randn_0_1_32f_C1R( &temp, 1, &state );
    return temp*sigma;
}
    

template<typename T, typename PT> static void
Randn_( Mat& _arr, uint64* state, const void* _param )
{
    const int RAND_BUF_SIZE = 96;
    float buffer[RAND_BUF_SIZE];
    int pidx[RAND_BUF_SIZE];    
    const PT* param = (const PT*)_param;
    Size size = getContinuousSize(_arr, _arr.channels());
    
    int i, n = std::min(size.width, RAND_BUF_SIZE);
    for( i = 0; i < 12; i++ )
        pidx[i] = i;
    for( ; i < n; i++ )
        pidx[i] = pidx[i - 12];

    for( int y = 0; y < size.height; y++ )
    {
        T* arr = (T*)(_arr.data + _arr.step*y);
        int len = RAND_BUF_SIZE;
        for( i = 0; i < size.width; i += RAND_BUF_SIZE )
        {
            if( i + len > size.width )
                len = size.width - i;

            Randn_0_1_32f_C1R( buffer, len, state );

            for( int j = 0; j < len; j++ )
                arr[i+j] = saturate_cast<T>(buffer[j]*param[pidx[j]+12] + param[pidx[j]]);
        }
    }
}


typedef void (*RandFunc)(Mat& dst, uint64* state, const void* param);

void RNG::fill( Mat& mat, int disttype, const Scalar& param1, const Scalar& param2 )
{
    static RandFunc rngtab[3][8] =
    {
        {
        RandBits_<uchar>, 
        RandBits_<schar>,
        RandBits_<ushort>,
        RandBits_<short>,
        RandBits_<int>, 0, 0, 0},

        {Randi_<uchar>,
        Randi_<schar>,
        Randi_<ushort>,
        Randi_<short>,
        Randi_<int>,
        Randf_, Randd_, 0},

        {Randn_<uchar,float>,
        Randn_<schar,float>,
        Randn_<ushort,float>,
        Randn_<short,float>,
        Randn_<int,float>,
        Randn_<float,float>,
        Randn_<double,double>, 0}
    };
    
    int depth = mat.depth(), channels = mat.channels();
    double dparam[2][12];
    float fparam[2][12];
    int iparam[2][12];
    void* param = dparam;
    int i, fast_int_mode = 0;
    RandFunc func = 0;

    CV_Assert( channels <= 4 );

    if( disttype == UNIFORM )
    {
        if( depth <= CV_32S )
        {
            for( i = 0, fast_int_mode = 1; i < channels; i++ )
            {
                double a = min(param1.val[i], param2.val[i]);
                double b = max(param1.val[i], param2.val[i]);
                int t0 = iparam[0][i] = cvCeil(a);
                int t1 = iparam[1][i] = cvFloor(b);
                double diff = b - a;

                fast_int_mode &= diff <= 4294967296. && ((t1-t0) & (t1-t0-1)) == 0;
            }
            
            if( fast_int_mode )
            {
                for( i = 0; i < channels; i++ )
                    iparam[1][i] = iparam[1][i] > iparam[0][i] ? iparam[1][i] - iparam[0][i] - 1 : 0;
            }
                
            for( ; i < 12; i++ )
            {
                int t0 = iparam[0][i - channels];
                int t1 = iparam[1][i - channels];
                
                iparam[0][i] = t0;
                iparam[1][i] = t1;
            }
            
            func = rngtab[!fast_int_mode][depth];
            param = iparam;
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
            for( i = 0; i < channels; i++ )
            {
                double t0 = param1.val[i];
                double t1 = param2.val[i];
                dparam[0][i] = (t1 + t0)*0.5;
                dparam[1][i] = (t1 - t0)*scale;
            }
            
            func = rngtab[1][depth];
            param = dparam;
        }
    }
    else if( disttype == CV_RAND_NORMAL )
    {
        for( i = 0; i < channels; i++ )
        {
            double t0 = param1.val[i];
            double t1 = param2.val[i];

            dparam[0][i] = t0;
            dparam[1][i] = t1;
        }

        func = rngtab[2][depth];
        param = dparam;
    }
    else
        CV_Error( CV_StsBadArg, "Unknown distribution type" );

    if( param == dparam )
    {
        for( i = channels; i < 12; i++ )
        {
            double t0 = dparam[0][i - channels];
            double t1 = dparam[1][i - channels];

            dparam[0][i] = t0;
            dparam[1][i] = t1;
        }

        if( depth != CV_64F )
        {
            for( i = 0; i < 12; i++ )
            {
                fparam[0][i] = (float)dparam[0][i];
                fparam[1][i] = (float)dparam[1][i];
            }
            param = fparam;
        }
    }

    CV_Assert( func != 0);
    
    if( mat.dims > 2 )
    {
        const Mat* arrays[] = {&mat, 0};
        Mat planes[1];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            func( it.planes[0], &state, param );
    }
    else
        func( mat, &state, param );
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

static void deleteRNG(void* data)
{
    delete (RNG*)data;
}

RNG& theRNG()
{
    if( !tlsRNGKey )
    {
        int errcode = pthread_key_create(&tlsRNGKey, deleteRNG);
        CV_Assert(errcode == 0);
    }
    RNG* rng = (RNG*)pthread_getspecific(tlsRNGKey);
    if( !rng )
    {
        rng = new RNG;
        pthread_setspecific(tlsRNGKey, rng);
    }
    return *rng;
}

#endif

void randu(CV_OUT Mat& dst, const Scalar& low, const Scalar& high)
{
    theRNG().fill(dst, RNG::UNIFORM, low, high);
}

void randn(CV_OUT Mat& dst, const Scalar& mean, const Scalar& stddev)
{
    theRNG().fill(dst, RNG::NORMAL, mean, stddev);
}    
    
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

void randShuffle( Mat& dst, double iterFactor, RNG* _rng )
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

    RNG& rng = _rng ? *_rng : theRNG();
    CV_Assert( dst.elemSize() <= 32 );
    RandShuffleFunc func = tab[dst.elemSize()];
    CV_Assert( func != 0 );
    func( dst, rng, iterFactor );
}

}

CV_IMPL void
cvRandArr( CvRNG* _rng, CvArr* arr, int disttype, CvScalar param1, CvScalar param2 )
{
    cv::Mat mat = cv::cvarrToMat(arr);
    // !!! this will only work for current 64-bit MWC RNG !!!
    cv::RNG& rng = _rng ? (cv::RNG&)*_rng : cv::theRNG();
    rng.fill(mat, disttype == CV_RAND_NORMAL ?
        cv::RNG::NORMAL : cv::RNG::UNIFORM, param1, param2 );
}

CV_IMPL void cvRandShuffle( CvArr* arr, CvRNG* _rng, double iter_factor )
{
    cv::Mat dst = cv::cvarrToMat(arr);
    cv::RNG& rng = _rng ? (cv::RNG&)*_rng : cv::theRNG();
    cv::randShuffle( dst, iter_factor, &rng );
}

/* End of file. */
