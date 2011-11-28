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

template<typename T, int shift> struct FixPtCast
{
    typedef int type1;
    typedef T rtype;
    rtype operator ()(type1 arg) const { return (T)((arg + (1 << (shift-1))) >> shift); }
};

template<typename T, int shift> struct FltCast
{
    typedef T type1;
    typedef T rtype;
    rtype operator ()(type1 arg) const { return arg*(T)(1./(1 << shift)); }
};

template<typename T1, typename T2> struct NoVec
{
    int operator()(T1**, T2*, int, int) const { return 0; }
};

#if CV_SSE2

struct PyrDownVec_32s8u
{
    int operator()(int** src, uchar* dst, int, int width) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;
        
        int x = 0;
        const int *row0 = src[0], *row1 = src[1], *row2 = src[2], *row3 = src[3], *row4 = src[4];
        __m128i delta = _mm_set1_epi16(128);

        for( ; x <= width - 16; x += 16 )
        {
            __m128i r0, r1, r2, r3, r4, t0, t1;
            r0 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row0 + x)),
                                 _mm_load_si128((const __m128i*)(row0 + x + 4)));
            r1 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row1 + x)),
                                 _mm_load_si128((const __m128i*)(row1 + x + 4)));
            r2 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row2 + x)),
                                 _mm_load_si128((const __m128i*)(row2 + x + 4)));
            r3 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row3 + x)),
                                 _mm_load_si128((const __m128i*)(row3 + x + 4)));
            r4 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row4 + x)),
                                 _mm_load_si128((const __m128i*)(row4 + x + 4)));
            r0 = _mm_add_epi16(r0, r4);
            r1 = _mm_add_epi16(_mm_add_epi16(r1, r3), r2);
            r0 = _mm_add_epi16(r0, _mm_add_epi16(r2, r2));
            t0 = _mm_add_epi16(r0, _mm_slli_epi16(r1, 2));
            r0 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row0 + x + 8)),
                                 _mm_load_si128((const __m128i*)(row0 + x + 12)));
            r1 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row1 + x + 8)),
                                 _mm_load_si128((const __m128i*)(row1 + x + 12)));
            r2 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row2 + x + 8)),
                                 _mm_load_si128((const __m128i*)(row2 + x + 12)));
            r3 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row3 + x + 8)),
                                 _mm_load_si128((const __m128i*)(row3 + x + 12)));
            r4 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row4 + x + 8)),
                                 _mm_load_si128((const __m128i*)(row4 + x + 12)));
            r0 = _mm_add_epi16(r0, r4);
            r1 = _mm_add_epi16(_mm_add_epi16(r1, r3), r2);
            r0 = _mm_add_epi16(r0, _mm_add_epi16(r2, r2));
            t1 = _mm_add_epi16(r0, _mm_slli_epi16(r1, 2));
            t0 = _mm_srli_epi16(_mm_add_epi16(t0, delta), 8);
            t1 = _mm_srli_epi16(_mm_add_epi16(t1, delta), 8);
            _mm_storeu_si128((__m128i*)(dst + x), _mm_packus_epi16(t0, t1));
        }

        for( ; x <= width - 4; x += 4 )
        {
            __m128i r0, r1, r2, r3, r4, z = _mm_setzero_si128();
            r0 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row0 + x)), z);
            r1 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row1 + x)), z);
            r2 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row2 + x)), z);
            r3 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row3 + x)), z);
            r4 = _mm_packs_epi32(_mm_load_si128((const __m128i*)(row4 + x)), z);
            r0 = _mm_add_epi16(r0, r4);
            r1 = _mm_add_epi16(_mm_add_epi16(r1, r3), r2);
            r0 = _mm_add_epi16(r0, _mm_add_epi16(r2, r2));
            r0 = _mm_add_epi16(r0, _mm_slli_epi16(r1, 2));
            r0 = _mm_srli_epi16(_mm_add_epi16(r0, delta), 8);
            *(int*)(dst + x) = _mm_cvtsi128_si32(_mm_packus_epi16(r0, r0));
        }

        return x;
    }
};

struct PyrDownVec_32f
{
    int operator()(float** src, float* dst, int, int width) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE) )
            return 0;
        
        int x = 0;
        const float *row0 = src[0], *row1 = src[1], *row2 = src[2], *row3 = src[3], *row4 = src[4];
        __m128 _4 = _mm_set1_ps(4.f), _scale = _mm_set1_ps(1.f/256);
        for( ; x <= width - 8; x += 8 )
        {
            __m128 r0, r1, r2, r3, r4, t0, t1;
            r0 = _mm_load_ps(row0 + x);
            r1 = _mm_load_ps(row1 + x);
            r2 = _mm_load_ps(row2 + x);
            r3 = _mm_load_ps(row3 + x);
            r4 = _mm_load_ps(row4 + x);
            r0 = _mm_add_ps(r0, r4);
            r1 = _mm_add_ps(_mm_add_ps(r1, r3), r2);
            r0 = _mm_add_ps(r0, _mm_add_ps(r2, r2));
            t0 = _mm_add_ps(r0, _mm_mul_ps(r1, _4));

            r0 = _mm_load_ps(row0 + x + 4);
            r1 = _mm_load_ps(row1 + x + 4);
            r2 = _mm_load_ps(row2 + x + 4);
            r3 = _mm_load_ps(row3 + x + 4);
            r4 = _mm_load_ps(row4 + x + 4);
            r0 = _mm_add_ps(r0, r4);
            r1 = _mm_add_ps(_mm_add_ps(r1, r3), r2);
            r0 = _mm_add_ps(r0, _mm_add_ps(r2, r2));
            t1 = _mm_add_ps(r0, _mm_mul_ps(r1, _4));

            t0 = _mm_mul_ps(t0, _scale);
            t1 = _mm_mul_ps(t1, _scale);

            _mm_storeu_ps(dst + x, t0);
            _mm_storeu_ps(dst + x + 4, t1);
        }

        return x;
    }
};

#else

typedef NoVec<int, uchar> PyrDownVec_32s8u;
typedef NoVec<float, float> PyrDownVec_32f;

#endif

template<class CastOp, class VecOp> void
pyrDown_( const Mat& _src, Mat& _dst )
{
    const int PD_SZ = 5;
    typedef typename CastOp::type1 WT;
    typedef typename CastOp::rtype T;

    Size ssize = _src.size(), dsize = _dst.size();
    int cn = _src.channels();
    int bufstep = (int)alignSize(dsize.width*cn, 16);
    AutoBuffer<WT> _buf(bufstep*PD_SZ + 16);
    WT* buf = alignPtr((WT*)_buf, 16);
    int tabL[CV_CN_MAX*(PD_SZ+2)], tabR[CV_CN_MAX*(PD_SZ+2)];
    AutoBuffer<int> _tabM(dsize.width*cn);
    int* tabM = _tabM;
    WT* rows[PD_SZ];
    CastOp castOp;
    VecOp vecOp;

    CV_Assert( std::abs(dsize.width*2 - ssize.width) <= 2 &&
               std::abs(dsize.height*2 - ssize.height) <= 2 );
    int k, x, sy0 = -PD_SZ/2, sy = sy0, width0 = std::min((ssize.width-PD_SZ/2-1)/2 + 1, dsize.width);

    for( x = 0; x <= PD_SZ+1; x++ )
    {
        int sx0 = borderInterpolate(x - PD_SZ/2, ssize.width, BORDER_REFLECT_101)*cn;
        int sx1 = borderInterpolate(x + width0*2 - PD_SZ/2, ssize.width, BORDER_REFLECT_101)*cn;
        for( k = 0; k < cn; k++ )
        {
            tabL[x*cn + k] = sx0 + k;
            tabR[x*cn + k] = sx1 + k;
        }
    }
    
    ssize.width *= cn;
    dsize.width *= cn;
    width0 *= cn;

    for( x = 0; x < dsize.width; x++ )
        tabM[x] = (x/cn)*2*cn + x % cn;

    for( int y = 0; y < dsize.height; y++ )
    {
        T* dst = (T*)(_dst.data + _dst.step*y);
        WT *row0, *row1, *row2, *row3, *row4;

        // fill the ring buffer (horizontal convolution and decimation)
        for( ; sy <= y*2 + 2; sy++ )
        {
            WT* row = buf + ((sy - sy0) % PD_SZ)*bufstep;
            int _sy = borderInterpolate(sy, ssize.height, BORDER_REFLECT_101);
            const T* src = (const T*)(_src.data + _src.step*_sy);
            int limit = cn;
            const int* tab = tabL;

            for( x = 0;;)
            {
                for( ; x < limit; x++ )
                {
                    row[x] = src[tab[x+cn*2]]*6 + (src[tab[x+cn]] + src[tab[x+cn*3]])*4 +
                        src[tab[x]] + src[tab[x+cn*4]];
                }

                if( x == dsize.width )
                    break;

                if( cn == 1 )
                {
                    for( ; x < width0; x++ )
                        row[x] = src[x*2]*6 + (src[x*2 - 1] + src[x*2 + 1])*4 +
                            src[x*2 - 2] + src[x*2 + 2];
                }
                else if( cn == 3 )
                {
                    for( ; x < width0; x += 3 )
                    {
                        const T* s = src + x*2;
                        WT t0 = s[0]*6 + (s[-3] + s[3])*4 + s[-6] + s[6];
                        WT t1 = s[1]*6 + (s[-2] + s[4])*4 + s[-5] + s[7];
                        WT t2 = s[2]*6 + (s[-1] + s[5])*4 + s[-4] + s[8];
                        row[x] = t0; row[x+1] = t1; row[x+2] = t2;
                    }
                }
                else if( cn == 4 )
                {
                    for( ; x < width0; x += 4 )
                    {
                        const T* s = src + x*2;
                        WT t0 = s[0]*6 + (s[-4] + s[4])*4 + s[-8] + s[8];
                        WT t1 = s[1]*6 + (s[-3] + s[5])*4 + s[-7] + s[9];
                        row[x] = t0; row[x+1] = t1;
                        t0 = s[2]*6 + (s[-2] + s[6])*4 + s[-6] + s[10];
                        t1 = s[3]*6 + (s[-1] + s[7])*4 + s[-5] + s[11];
                        row[x+2] = t0; row[x+3] = t1;
                    }
                }
                else
                {
                    for( ; x < width0; x++ )
                    {
                        int sx = tabM[x];
                        row[x] = src[sx]*6 + (src[sx - cn] + src[sx + cn])*4 +
                            src[sx - cn*2] + src[sx + cn*2];
                    }
                }

                limit = dsize.width;
                tab = tabR - x;
            }
        }

        // do vertical convolution and decimation and write the result to the destination image
        for( k = 0; k < PD_SZ; k++ )
            rows[k] = buf + ((y*2 - PD_SZ/2 + k - sy0) % PD_SZ)*bufstep;
        row0 = rows[0]; row1 = rows[1]; row2 = rows[2]; row3 = rows[3]; row4 = rows[4];

        x = vecOp(rows, dst, (int)_dst.step, dsize.width);
        for( ; x < dsize.width; x++ )
            dst[x] = castOp(row2[x]*6 + (row1[x] + row3[x])*4 + row0[x] + row4[x]);
    }
}


template<class CastOp, class VecOp> void
pyrUp_( const Mat& _src, Mat& _dst )
{
    const int PU_SZ = 3;
    typedef typename CastOp::type1 WT;
    typedef typename CastOp::rtype T;

    Size ssize = _src.size(), dsize = _dst.size();
    int cn = _src.channels();
    int bufstep = (int)alignSize((dsize.width+1)*cn, 16);
    AutoBuffer<WT> _buf(bufstep*PU_SZ + 16);
    WT* buf = alignPtr((WT*)_buf, 16);
    AutoBuffer<int> _dtab(ssize.width*cn);
    int* dtab = _dtab;
    WT* rows[PU_SZ];
    CastOp castOp;
    VecOp vecOp;

    CV_Assert( std::abs(dsize.width - ssize.width*2) == dsize.width % 2 &&
               std::abs(dsize.height - ssize.height*2) == dsize.height % 2);
    int k, x, sy0 = -PU_SZ/2, sy = sy0, width0 = ssize.width - 1;

    ssize.width *= cn;
    dsize.width *= cn;
    width0 *= cn;

    for( x = 0; x < ssize.width; x++ )
        dtab[x] = (x/cn)*2*cn + x % cn;

    for( int y = 0; y < ssize.height; y++ )
    {
        T* dst0 = (T*)(_dst.data + _dst.step*y*2);
        T* dst1 = (T*)(_dst.data + _dst.step*(y*2+1));
        WT *row0, *row1, *row2;

        if( y*2+1 >= dsize.height )
            dst1 = dst0;

        // fill the ring buffer (horizontal convolution and decimation)
        for( ; sy <= y + 1; sy++ )
        {
            WT* row = buf + ((sy - sy0) % PU_SZ)*bufstep;
            int _sy = borderInterpolate(sy*2, dsize.height, BORDER_REFLECT_101)/2;
            const T* src = (const T*)(_src.data + _src.step*_sy);

            if( ssize.width == cn )
            {
                for( x = 0; x < cn; x++ )
                    row[x] = row[x + cn] = src[x]*8;
                continue;
            }

            for( x = 0; x < cn; x++ )
            {
                int dx = dtab[x];
                WT t0 = src[x]*6 + src[x + cn]*2;
                WT t1 = (src[x] + src[x + cn])*4;
                row[dx] = t0; row[dx + cn] = t1;
                dx = dtab[ssize.width - cn + x];
                int sx = ssize.width - cn + x;
                t0 = src[sx - cn] + src[sx]*7;
                t1 = src[sx]*8;
                row[dx] = t0; row[dx + cn] = t1;
            }

            for( x = cn; x < ssize.width - cn; x++ )
            {
                int dx = dtab[x];
                WT t0 = src[x-cn] + src[x]*6 + src[x+cn];
                WT t1 = (src[x] + src[x+cn])*4;
                row[dx] = t0;
                row[dx+cn] = t1;
            }
        }

        // do vertical convolution and decimation and write the result to the destination image
        for( k = 0; k < PU_SZ; k++ )
            rows[k] = buf + ((y - PU_SZ/2 + k - sy0) % PU_SZ)*bufstep;
        row0 = rows[0]; row1 = rows[1]; row2 = rows[2];

        x = vecOp(rows, dst0, (int)_dst.step, dsize.width);
        for( ; x < dsize.width; x++ )
        {
            T t1 = castOp((row1[x] + row2[x])*4);
            T t0 = castOp(row0[x] + row1[x]*6 + row2[x]);
            dst1[x] = t1; dst0[x] = t0;
        }
    }
}

typedef void (*PyrFunc)(const Mat&, Mat&);

}
    
void cv::pyrDown( InputArray _src, OutputArray _dst, const Size& _dsz )
{
    Mat src = _src.getMat();
    Size dsz = _dsz == Size() ? Size((src.cols + 1)/2, (src.rows + 1)/2) : _dsz;
    _dst.create( dsz, src.type() );
    Mat dst = _dst.getMat();

#ifdef HAVE_TEGRA_OPTIMIZATION
    if(tegra::pyrDown(src, dst))
        return;
#endif

    int depth = src.depth();
    PyrFunc func = 0;
    if( depth == CV_8U )
        func = pyrDown_<FixPtCast<uchar, 8>, PyrDownVec_32s8u>;
    else if( depth == CV_16S )
        func = pyrDown_<FixPtCast<short, 8>, NoVec<int, short> >;
    else if( depth == CV_16U )
        func = pyrDown_<FixPtCast<ushort, 8>, NoVec<int, ushort> >;
    else if( depth == CV_32F )
        func = pyrDown_<FltCast<float, 8>, PyrDownVec_32f>;
    else if( depth == CV_64F )
        func = pyrDown_<FltCast<double, 8>, NoVec<double, double> >;
    else
        CV_Error( CV_StsUnsupportedFormat, "" );

    func( src, dst );
}

void cv::pyrUp( InputArray _src, OutputArray _dst, const Size& _dsz )
{
    Mat src = _src.getMat();
    Size dsz = _dsz == Size() ? Size(src.cols*2, src.rows*2) : _dsz;
    _dst.create( dsz, src.type() );
    Mat dst = _dst.getMat();

#ifdef HAVE_TEGRA_OPTIMIZATION
    if(tegra::pyrUp(src, dst))
        return;
#endif

    int depth = src.depth();
    PyrFunc func = 0;
    if( depth == CV_8U )
        func = pyrUp_<FixPtCast<uchar, 6>, NoVec<int, uchar> >;
    else if( depth == CV_16S )
        func = pyrUp_<FixPtCast<short, 6>, NoVec<int, short> >;
    else if( depth == CV_16U )
        func = pyrUp_<FixPtCast<ushort, 6>, NoVec<int, ushort> >;
    else if( depth == CV_32F )
        func = pyrUp_<FltCast<float, 6>, NoVec<float, float> >;
    else if( depth == CV_64F )
        func = pyrUp_<FltCast<double, 6>, NoVec<double, double> >;
    else
        CV_Error( CV_StsUnsupportedFormat, "" );

    func( src, dst );
}

void cv::buildPyramid( InputArray _src, OutputArrayOfArrays _dst, int maxlevel )
{
    Mat src = _src.getMat();
    _dst.create( maxlevel + 1, 1, 0 );
    _dst.getMatRef(0) = src;
    for( int i = 1; i <= maxlevel; i++ )
        pyrDown( _dst.getMatRef(i-1), _dst.getMatRef(i) );
}

CV_IMPL void cvPyrDown( const void* srcarr, void* dstarr, int _filter )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( _filter == CV_GAUSSIAN_5x5 && src.type() == dst.type());
    cv::pyrDown( src, dst, dst.size() );
}

CV_IMPL void cvPyrUp( const void* srcarr, void* dstarr, int _filter )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( _filter == CV_GAUSSIAN_5x5 && src.type() == dst.type());
    cv::pyrUp( src, dst, dst.size() );
}


CV_IMPL void
cvReleasePyramid( CvMat*** _pyramid, int extra_layers )
{
    if( !_pyramid )
        CV_Error( CV_StsNullPtr, "" );
    
    if( *_pyramid )
        for( int i = 0; i <= extra_layers; i++ )
            cvReleaseMat( &(*_pyramid)[i] );
    
    cvFree( _pyramid );
}


CV_IMPL CvMat**
cvCreatePyramid( const CvArr* srcarr, int extra_layers, double rate,
                 const CvSize* layer_sizes, CvArr* bufarr,
                 int calc, int filter )
{
    const float eps = 0.1f;
    uchar* ptr = 0;

    CvMat stub, *src = cvGetMat( srcarr, &stub );

    if( extra_layers < 0 )
        CV_Error( CV_StsOutOfRange, "The number of extra layers must be non negative" );

    int i, layer_step, elem_size = CV_ELEM_SIZE(src->type);
    CvSize layer_size, size = cvGetMatSize(src);

    if( bufarr )
    {
        CvMat bstub, *buf;
        int bufsize = 0;

        buf = cvGetMat( bufarr, &bstub );
        bufsize = buf->rows*buf->cols*CV_ELEM_SIZE(buf->type);
        layer_size = size;
        for( i = 1; i <= extra_layers; i++ )
        {
            if( !layer_sizes )
            {
                layer_size.width = cvRound(layer_size.width*rate+eps);
                layer_size.height = cvRound(layer_size.height*rate+eps);
            }
            else
                layer_size = layer_sizes[i-1];
            layer_step = layer_size.width*elem_size;
            bufsize -= layer_step*layer_size.height;
        }

        if( bufsize < 0 )
            CV_Error( CV_StsOutOfRange, "The buffer is too small to fit the pyramid" );
        ptr = buf->data.ptr;
    }

    CvMat** pyramid = (CvMat**)cvAlloc( (extra_layers+1)*sizeof(pyramid[0]) );
    memset( pyramid, 0, (extra_layers+1)*sizeof(pyramid[0]) );

    pyramid[0] = cvCreateMatHeader( size.height, size.width, src->type );
    cvSetData( pyramid[0], src->data.ptr, src->step );
    layer_size = size;

    for( i = 1; i <= extra_layers; i++ )
    {
        if( !layer_sizes )
        {
            layer_size.width = cvRound(layer_size.width*rate + eps);
            layer_size.height = cvRound(layer_size.height*rate + eps);
        }
        else
            layer_size = layer_sizes[i];

        if( bufarr )
        {
            pyramid[i] = cvCreateMatHeader( layer_size.height, layer_size.width, src->type );
            layer_step = layer_size.width*elem_size;
            cvSetData( pyramid[i], ptr, layer_step );
            ptr += layer_step*layer_size.height;
        }
        else
            pyramid[i] = cvCreateMat( layer_size.height, layer_size.width, src->type );

        if( calc )
            cvPyrDown( pyramid[i-1], pyramid[i], filter );
            //cvResize( pyramid[i-1], pyramid[i], CV_INTER_LINEAR );
    }

    return pyramid;
}

/* End of file. */
