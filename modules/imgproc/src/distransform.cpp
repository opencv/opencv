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

#define ICV_DIST_SHIFT  16
#define ICV_INIT_DIST0  (INT_MAX >> 2)

static CvStatus
icvInitTopBottom( int* temp, int tempstep, CvSize size, int border )
{
    int i, j;
    for( i = 0; i < border; i++ )
    {
        int* ttop = (int*)(temp + i*tempstep);
        int* tbottom = (int*)(temp + (size.height + border*2 - i - 1)*tempstep);
        
        for( j = 0; j < size.width + border*2; j++ )
        {
            ttop[j] = ICV_INIT_DIST0;
            tbottom[j] = ICV_INIT_DIST0;
        }
    }

    return CV_OK;
}


static CvStatus CV_STDCALL
icvDistanceTransform_3x3_C1R( const uchar* src, int srcstep, int* temp,
        int step, float* dist, int dststep, CvSize size, const float* metrics )
{
    const int BORDER = 1;
    int i, j;
    const int HV_DIST = CV_FLT_TO_FIX( metrics[0], ICV_DIST_SHIFT );
    const int DIAG_DIST = CV_FLT_TO_FIX( metrics[1], ICV_DIST_SHIFT );
    const float scale = 1.f/(1 << ICV_DIST_SHIFT);

    srcstep /= sizeof(src[0]);
    step /= sizeof(temp[0]);
    dststep /= sizeof(dist[0]);

    icvInitTopBottom( temp, step, size, BORDER );

    // forward pass
    for( i = 0; i < size.height; i++ )
    {
        const uchar* s = src + i*srcstep;
        int* tmp = (int*)(temp + (i+BORDER)*step) + BORDER;

        for( j = 0; j < BORDER; j++ )
            tmp[-j-1] = tmp[size.width + j] = ICV_INIT_DIST0;
        
        for( j = 0; j < size.width; j++ )
        {
            if( !s[j] )
                tmp[j] = 0;
            else
            {
                int t0 = tmp[j-step-1] + DIAG_DIST;
                int t = tmp[j-step] + HV_DIST;
                if( t0 > t ) t0 = t;
                t = tmp[j-step+1] + DIAG_DIST;
                if( t0 > t ) t0 = t;
                t = tmp[j-1] + HV_DIST;
                if( t0 > t ) t0 = t;
                tmp[j] = t0;
            }
        }
    }

    // backward pass
    for( i = size.height - 1; i >= 0; i-- )
    {
        float* d = (float*)(dist + i*dststep);
        int* tmp = (int*)(temp + (i+BORDER)*step) + BORDER;
        
        for( j = size.width - 1; j >= 0; j-- )
        {
            int t0 = tmp[j];
            if( t0 > HV_DIST )
            {
                int t = tmp[j+step+1] + DIAG_DIST;
                if( t0 > t ) t0 = t;
                t = tmp[j+step] + HV_DIST;
                if( t0 > t ) t0 = t;
                t = tmp[j+step-1] + DIAG_DIST;
                if( t0 > t ) t0 = t;
                t = tmp[j+1] + HV_DIST;
                if( t0 > t ) t0 = t;
                tmp[j] = t0;
            }
            d[j] = (float)(t0 * scale);
        }
    }

    return CV_OK;
}


static CvStatus CV_STDCALL
icvDistanceTransform_5x5_C1R( const uchar* src, int srcstep, int* temp,
        int step, float* dist, int dststep, CvSize size, const float* metrics )
{
    const int BORDER = 2;
    int i, j;
    const int HV_DIST = CV_FLT_TO_FIX( metrics[0], ICV_DIST_SHIFT );
    const int DIAG_DIST = CV_FLT_TO_FIX( metrics[1], ICV_DIST_SHIFT );
    const int LONG_DIST = CV_FLT_TO_FIX( metrics[2], ICV_DIST_SHIFT );
    const float scale = 1.f/(1 << ICV_DIST_SHIFT);

    srcstep /= sizeof(src[0]);
    step /= sizeof(temp[0]);
    dststep /= sizeof(dist[0]);

    icvInitTopBottom( temp, step, size, BORDER );

    // forward pass
    for( i = 0; i < size.height; i++ )
    {
        const uchar* s = src + i*srcstep;
        int* tmp = (int*)(temp + (i+BORDER)*step) + BORDER;

        for( j = 0; j < BORDER; j++ )
            tmp[-j-1] = tmp[size.width + j] = ICV_INIT_DIST0;
        
        for( j = 0; j < size.width; j++ )
        {
            if( !s[j] )
                tmp[j] = 0;
            else
            {
                int t0 = tmp[j-step*2-1] + LONG_DIST;
                int t = tmp[j-step*2+1] + LONG_DIST;
                if( t0 > t ) t0 = t;
                t = tmp[j-step-2] + LONG_DIST;
                if( t0 > t ) t0 = t;
                t = tmp[j-step-1] + DIAG_DIST;
                if( t0 > t ) t0 = t;
                t = tmp[j-step] + HV_DIST;
                if( t0 > t ) t0 = t;
                t = tmp[j-step+1] + DIAG_DIST;
                if( t0 > t ) t0 = t;
                t = tmp[j-step+2] + LONG_DIST;
                if( t0 > t ) t0 = t;
                t = tmp[j-1] + HV_DIST;
                if( t0 > t ) t0 = t;
                tmp[j] = t0;
            }
        }
    }

    // backward pass
    for( i = size.height - 1; i >= 0; i-- )
    {
        float* d = (float*)(dist + i*dststep);
        int* tmp = (int*)(temp + (i+BORDER)*step) + BORDER;
        
        for( j = size.width - 1; j >= 0; j-- )
        {
            int t0 = tmp[j];
            if( t0 > HV_DIST )
            {
                int t = tmp[j+step*2+1] + LONG_DIST;
                if( t0 > t ) t0 = t;
                t = tmp[j+step*2-1] + LONG_DIST;
                if( t0 > t ) t0 = t;
                t = tmp[j+step+2] + LONG_DIST;
                if( t0 > t ) t0 = t;
                t = tmp[j+step+1] + DIAG_DIST;
                if( t0 > t ) t0 = t;
                t = tmp[j+step] + HV_DIST;
                if( t0 > t ) t0 = t;
                t = tmp[j+step-1] + DIAG_DIST;
                if( t0 > t ) t0 = t;
                t = tmp[j+step-2] + LONG_DIST;
                if( t0 > t ) t0 = t;
                t = tmp[j+1] + HV_DIST;
                if( t0 > t ) t0 = t;
                tmp[j] = t0;
            }
            d[j] = (float)(t0 * scale);
        }
    }

    return CV_OK;
}


static CvStatus CV_STDCALL
icvDistanceTransformEx_5x5_C1R( const uchar* src, int srcstep, int* temp,
                int step, float* dist, int dststep, int* labels, int lstep,
                CvSize size, const float* metrics )
{
    const int BORDER = 2;
    
    int i, j;
    const int HV_DIST = CV_FLT_TO_FIX( metrics[0], ICV_DIST_SHIFT );
    const int DIAG_DIST = CV_FLT_TO_FIX( metrics[1], ICV_DIST_SHIFT );
    const int LONG_DIST = CV_FLT_TO_FIX( metrics[2], ICV_DIST_SHIFT );
    const float scale = 1.f/(1 << ICV_DIST_SHIFT);

    srcstep /= sizeof(src[0]);
    step /= sizeof(temp[0]);
    dststep /= sizeof(dist[0]);
    lstep /= sizeof(labels[0]);

    icvInitTopBottom( temp, step, size, BORDER );

    // forward pass
    for( i = 0; i < size.height; i++ )
    {
        const uchar* s = src + i*srcstep;
        int* tmp = (int*)(temp + (i+BORDER)*step) + BORDER;
        int* lls = (int*)(labels + i*lstep);

        for( j = 0; j < BORDER; j++ )
            tmp[-j-1] = tmp[size.width + j] = ICV_INIT_DIST0;
        
        for( j = 0; j < size.width; j++ )
        {
            if( !s[j] )
            {
                tmp[j] = 0;
                //assert( lls[j] != 0 );
            }
            else
            {
                int t0 = ICV_INIT_DIST0, t;
                int l0 = 0;

                t = tmp[j-step*2-1] + LONG_DIST;
                if( t0 > t )
                {
                    t0 = t;
                    l0 = lls[j-lstep*2-1];
                }
                t = tmp[j-step*2+1] + LONG_DIST;
                if( t0 > t )
                {
                    t0 = t;
                    l0 = lls[j-lstep*2+1];
                }
                t = tmp[j-step-2] + LONG_DIST;
                if( t0 > t )
                {
                    t0 = t;
                    l0 = lls[j-lstep-2];
                }
                t = tmp[j-step-1] + DIAG_DIST;
                if( t0 > t )
                {
                    t0 = t;
                    l0 = lls[j-lstep-1];
                }
                t = tmp[j-step] + HV_DIST;
                if( t0 > t )
                {
                    t0 = t;
                    l0 = lls[j-lstep];
                }
                t = tmp[j-step+1] + DIAG_DIST;
                if( t0 > t )
                {
                    t0 = t;
                    l0 = lls[j-lstep+1];
                }
                t = tmp[j-step+2] + LONG_DIST;
                if( t0 > t )
                {
                    t0 = t;
                    l0 = lls[j-lstep+2];
                }
                t = tmp[j-1] + HV_DIST;
                if( t0 > t )
                {
                    t0 = t;
                    l0 = lls[j-1];
                }

                tmp[j] = t0;
                lls[j] = l0;
            }
        }
    }

    // backward pass
    for( i = size.height - 1; i >= 0; i-- )
    {
        float* d = (float*)(dist + i*dststep);
        int* tmp = (int*)(temp + (i+BORDER)*step) + BORDER;
        int* lls = (int*)(labels + i*lstep);
        
        for( j = size.width - 1; j >= 0; j-- )
        {
            int t0 = tmp[j];
            int l0 = lls[j];
            if( t0 > HV_DIST )
            {
                int t = tmp[j+step*2+1] + LONG_DIST;
                if( t0 > t )
                {
                    t0 = t;
                    l0 = lls[j+lstep*2+1];
                }
                t = tmp[j+step*2-1] + LONG_DIST;
                if( t0 > t )
                {
                    t0 = t;
                    l0 = lls[j+lstep*2-1];
                }
                t = tmp[j+step+2] + LONG_DIST;
                if( t0 > t )
                {
                    t0 = t;
                    l0 = lls[j+lstep+2];
                }
                t = tmp[j+step+1] + DIAG_DIST;
                if( t0 > t )
                {
                    t0 = t;
                    l0 = lls[j+lstep+1];
                }
                t = tmp[j+step] + HV_DIST;
                if( t0 > t )
                {
                    t0 = t;
                    l0 = lls[j+lstep];
                }
                t = tmp[j+step-1] + DIAG_DIST;
                if( t0 > t )
                {
                    t0 = t;
                    l0 = lls[j+lstep-1];
                }
                t = tmp[j+step-2] + LONG_DIST;
                if( t0 > t )
                {
                    t0 = t;
                    l0 = lls[j+lstep-2];
                }
                t = tmp[j+1] + HV_DIST;
                if( t0 > t )
                {
                    t0 = t;
                    l0 = lls[j+1];
                }
                tmp[j] = t0;
                lls[j] = l0;
            }
            d[j] = (float)(t0 * scale);
        }
    }

    return CV_OK;
}


static CvStatus
icvGetDistanceTransformMask( int maskType, float *metrics )
{
    if( !metrics )
        return CV_NULLPTR_ERR;

    switch (maskType)
    {
    case 30:
        metrics[0] = 1.0f;
        metrics[1] = 1.0f;
        break;

    case 31:
        metrics[0] = 1.0f;
        metrics[1] = 2.0f;
        break;

    case 32:
        metrics[0] = 0.955f;
        metrics[1] = 1.3693f;
        break;

    case 50:
        metrics[0] = 1.0f;
        metrics[1] = 1.0f;
        metrics[2] = 2.0f;
        break;

    case 51:
        metrics[0] = 1.0f;
        metrics[1] = 2.0f;
        metrics[2] = 3.0f;
        break;

    case 52:
        metrics[0] = 1.0f;
        metrics[1] = 1.4f;
        metrics[2] = 2.1969f;
        break;
    default:
        return CV_BADRANGE_ERR;
    }

    return CV_OK;
}

namespace cv
{

struct DTColumnInvoker
{
    DTColumnInvoker( const CvMat* _src, CvMat* _dst, const int* _sat_tab, const float* _sqr_tab)
    {
        src = _src;
        dst = _dst;
        sat_tab = _sat_tab + src->rows*2 + 1;
        sqr_tab = _sqr_tab;
    }
    
    void operator()( const BlockedRange& range ) const
    {
        int i, i1 = range.begin(), i2 = range.end();
        int m = src->rows;
        size_t sstep = src->step, dstep = dst->step/sizeof(float);
        AutoBuffer<int> _d(m);
        int* d = _d;
        
        for( i = i1; i < i2; i++ )
        {
            const uchar* sptr = src->data.ptr + i + (m-1)*sstep;
            float* dptr = dst->data.fl + i;
            int j, dist = m-1;
            
            for( j = m-1; j >= 0; j--, sptr -= sstep )
            {
                dist = (dist + 1) & (sptr[0] == 0 ? 0 : -1);
                d[j] = dist;
            }
            
            dist = m-1;
            for( j = 0; j < m; j++, dptr += dstep )
            {
                dist = dist + 1 - sat_tab[dist - d[j]];
                d[j] = dist;
                dptr[0] = sqr_tab[dist];
            }
        }
    }
    
    const CvMat* src;
    CvMat* dst;
    const int* sat_tab;
    const float* sqr_tab;
};
    
    
struct DTRowInvoker
{
    DTRowInvoker( CvMat* _dst, const float* _sqr_tab, const float* _inv_tab )
    {
        dst = _dst;
        sqr_tab = _sqr_tab;
        inv_tab = _inv_tab;
    }
    
    void operator()( const BlockedRange& range ) const
    {
        const float inf = 1e15f;
        int i, i1 = range.begin(), i2 = range.end();
        int n = dst->cols;
        AutoBuffer<uchar> _buf((n+2)*2*sizeof(float) + (n+2)*sizeof(int));
        float* f = (float*)(uchar*)_buf;
        float* z = f + n;
        int* v = alignPtr((int*)(z + n + 1), sizeof(int));
       
        for( i = i1; i < i2; i++ )
        {
            float* d = (float*)(dst->data.ptr + i*dst->step);
            int p, q, k;
            
            v[0] = 0;
            z[0] = -inf;
            z[1] = inf;
            f[0] = d[0];
            
            for( q = 1, k = 0; q < n; q++ )
            {
                float fq = d[q];
                f[q] = fq;
                
                for(;;k--)
                {
                    p = v[k];
                    float s = (fq + sqr_tab[q] - d[p] - sqr_tab[p])*inv_tab[q - p];
                    if( s > z[k] )
                    {
                        k++;
                        v[k] = q;
                        z[k] = s;
                        z[k+1] = inf;
                        break;
                    }
                }
            }
            
            for( q = 0, k = 0; q < n; q++ )
            {
                while( z[k+1] < q )
                    k++;
                p = v[k];
                d[q] = std::sqrt(sqr_tab[std::abs(q - p)] + f[p]);
            }
        }
    }
    
    CvMat* dst;
    const float* sqr_tab;
    const float* inv_tab;
};

}

static void
icvTrueDistTrans( const CvMat* src, CvMat* dst )
{
    const float inf = 1e15f;
    
    if( !CV_ARE_SIZES_EQ( src, dst ))
        CV_Error( CV_StsUnmatchedSizes, "" );

    if( CV_MAT_TYPE(src->type) != CV_8UC1 ||
        CV_MAT_TYPE(dst->type) != CV_32FC1 )
        CV_Error( CV_StsUnsupportedFormat,
        "The input image must have 8uC1 type and the output one must have 32fC1 type" );

    int i, m = src->rows, n = src->cols;

    cv::AutoBuffer<uchar> _buf(std::max(m*2*sizeof(float) + (m*3+1)*sizeof(int), n*2*sizeof(float)));
    // stage 1: compute 1d distance transform of each column
    float* sqr_tab = (float*)(uchar*)_buf;
    int* sat_tab = cv::alignPtr((int*)(sqr_tab + m*2), sizeof(int));
    int shift = m*2;

    for( i = 0; i < m; i++ )
        sqr_tab[i] = (float)(i*i);
    for( i = m; i < m*2; i++ )
        sqr_tab[i] = inf;
    for( i = 0; i < shift; i++ )
        sat_tab[i] = 0;
    for( ; i <= m*3; i++ )
        sat_tab[i] = i - shift;

    cv::parallel_for(cv::BlockedRange(0, n), cv::DTColumnInvoker(src, dst, sat_tab, sqr_tab)); 

    // stage 2: compute modified distance transform for each row
    float* inv_tab = sqr_tab + n;
    
    inv_tab[0] = sqr_tab[0] = 0.f;
    for( i = 1; i < n; i++ )
    {
        inv_tab[i] = (float)(0.5/i);
        sqr_tab[i] = (float)(i*i);
    }

    cv::parallel_for(cv::BlockedRange(0, m), cv::DTRowInvoker(dst, sqr_tab, inv_tab));
}


/*********************************** IPP functions *********************************/

typedef CvStatus (CV_STDCALL * CvIPPDistTransFunc)( const uchar* src, int srcstep,
                                                    void* dst, int dststep,
                                                    CvSize size, const void* metrics );

typedef CvStatus (CV_STDCALL * CvIPPDistTransFunc2)( uchar* src, int srcstep,
                                                     CvSize size, const int* metrics );

/***********************************************************************************/

typedef CvStatus (CV_STDCALL * CvDistTransFunc)( const uchar* src, int srcstep,
                                                 int* temp, int tempstep,
                                                 float* dst, int dststep,
                                                 CvSize size, const float* metrics );


/****************************************************************************************\
 Non-inplace and Inplace 8u->8u Distance Transform for CityBlock (a.k.a. L1) metric
 (C) 2006 by Jay Stavinzky.
\****************************************************************************************/

//BEGIN ATS ADDITION
/* 8-bit grayscale distance transform function */
static void
icvDistanceATS_L1_8u( const CvMat* src, CvMat* dst )
{
    int width = src->cols, height = src->rows;

    int a;
    uchar lut[256];
    int x, y;
    
    const uchar *sbase = src->data.ptr;
    uchar *dbase = dst->data.ptr;
    int srcstep = src->step;
    int dststep = dst->step;

    CV_Assert( CV_IS_MASK_ARR( src ) && CV_MAT_TYPE( dst->type ) == CV_8UC1 );
    CV_Assert( CV_ARE_SIZES_EQ( src, dst ));

    ////////////////////// forward scan ////////////////////////
    for( x = 0; x < 256; x++ )
        lut[x] = CV_CAST_8U(x+1);

    //init first pixel to max (we're going to be skipping it)
    dbase[0] = (uchar)(sbase[0] == 0 ? 0 : 255);

    //first row (scan west only, skip first pixel)
    for( x = 1; x < width; x++ )
        dbase[x] = (uchar)(sbase[x] == 0 ? 0 : lut[dbase[x-1]]);

    for( y = 1; y < height; y++ )
    {
        sbase += srcstep;
        dbase += dststep;

        //for left edge, scan north only
        a = sbase[0] == 0 ? 0 : lut[dbase[-dststep]];
        dbase[0] = (uchar)a;

        for( x = 1; x < width; x++ )
        {
            a = sbase[x] == 0 ? 0 : lut[MIN(a, dbase[x - dststep])];
            dbase[x] = (uchar)a;
        }
    }

    ////////////////////// backward scan ///////////////////////

    a = dbase[width-1];

    // do last row east pixel scan here (skip bottom right pixel)
    for( x = width - 2; x >= 0; x-- )
    {
        a = lut[a];
        dbase[x] = (uchar)(CV_CALC_MIN_8U(a, dbase[x]));
    }

    // right edge is the only error case
    for( y = height - 2; y >= 0; y-- )
    {
        dbase -= dststep;

        // do right edge
        a = lut[dbase[width-1+dststep]];
        dbase[width-1] = (uchar)(MIN(a, dbase[width-1]));

        for( x = width - 2; x >= 0; x-- )
        {
            int b = dbase[x+dststep];
            a = lut[MIN(a, b)];
            dbase[x] = (uchar)(MIN(a, dbase[x]));
        }
    }
}
//END ATS ADDITION


/* Wrapper function for distance transform group */
CV_IMPL void
cvDistTransform( const void* srcarr, void* dstarr,
                 int distType, int maskSize,
                 const float *mask,
                 void* labelsarr, int labelType )
{
    float _mask[5] = {0};
    CvMat srcstub, *src = (CvMat*)srcarr;
    CvMat dststub, *dst = (CvMat*)dstarr;
    CvMat lstub, *labels = (CvMat*)labelsarr;

    src = cvGetMat( src, &srcstub );
    dst = cvGetMat( dst, &dststub );

    if( !CV_IS_MASK_ARR( src ) || (CV_MAT_TYPE( dst->type ) != CV_32FC1 &&
        (CV_MAT_TYPE(dst->type) != CV_8UC1 || distType != CV_DIST_L1 || labels)) )
        CV_Error( CV_StsUnsupportedFormat,
        "source image must be 8uC1 and the distance map must be 32fC1 "
        "(or 8uC1 in case of simple L1 distance transform)" );

    if( !CV_ARE_SIZES_EQ( src, dst ))
        CV_Error( CV_StsUnmatchedSizes, "the source and the destination images must be of the same size" );

    if( maskSize != CV_DIST_MASK_3 && maskSize != CV_DIST_MASK_5 && maskSize != CV_DIST_MASK_PRECISE )
        CV_Error( CV_StsBadSize, "Mask size should be 3 or 5 or 0 (presize)" );

    if( distType == CV_DIST_C || distType == CV_DIST_L1 )
        maskSize = !labels ? CV_DIST_MASK_3 : CV_DIST_MASK_5;
    else if( distType == CV_DIST_L2 && labels )
        maskSize = CV_DIST_MASK_5;

    if( maskSize == CV_DIST_MASK_PRECISE )
    {
        icvTrueDistTrans( src, dst );
        return;
    }
    
    if( labels )
    {
        labels = cvGetMat( labels, &lstub );
        if( CV_MAT_TYPE( labels->type ) != CV_32SC1 )
            CV_Error( CV_StsUnsupportedFormat, "the output array of labels must be 32sC1" );

        if( !CV_ARE_SIZES_EQ( labels, dst ))
            CV_Error( CV_StsUnmatchedSizes, "the array of labels has a different size" );

        if( maskSize == CV_DIST_MASK_3 )
            CV_Error( CV_StsNotImplemented,
            "3x3 mask can not be used for \"labeled\" distance transform. Use 5x5 mask" );
    }

    if( distType == CV_DIST_C || distType == CV_DIST_L1 || distType == CV_DIST_L2 )
    {
        icvGetDistanceTransformMask( (distType == CV_DIST_C ? 0 :
            distType == CV_DIST_L1 ? 1 : 2) + maskSize*10, _mask );
    }
    else if( distType == CV_DIST_USER )
    {
        if( !mask )
            CV_Error( CV_StsNullPtr, "" );

        memcpy( _mask, mask, (maskSize/2 + 1)*sizeof(float));
    }

    CvSize size = cvGetMatSize(src);

    if( CV_MAT_TYPE(dst->type) == CV_8UC1 )
    {
        icvDistanceATS_L1_8u( src, dst );
    }
    else
    {
        int border = maskSize == CV_DIST_MASK_3 ? 1 : 2;
        cv::Ptr<CvMat> temp = cvCreateMat( size.height + border*2, size.width + border*2, CV_32SC1 );

        if( !labels )
        {
            CvDistTransFunc func = maskSize == CV_DIST_MASK_3 ?
                icvDistanceTransform_3x3_C1R :
                icvDistanceTransform_5x5_C1R;

            func( src->data.ptr, src->step, temp->data.i, temp->step,
                  dst->data.fl, dst->step, size, _mask );
        }
        else
        {
            cvZero( labels );
            
            if( labelType == CV_DIST_LABEL_CCOMP )
            {
                CvSeq *contours = 0;
                cv::Ptr<CvMemStorage> st = cvCreateMemStorage();
                cv::Ptr<CvMat> src_copy = cvCreateMat( size.height+border*2, size.width+border*2, src->type );
                cvCopyMakeBorder(src, src_copy, cvPoint(border, border), IPL_BORDER_CONSTANT, cvScalarAll(255));
                cvCmpS( src_copy, 0, src_copy, CV_CMP_EQ );
                cvFindContours( src_copy, st, &contours, sizeof(CvContour),
                               CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(-border, -border));
                
                for( int label = 1; contours != 0; contours = contours->h_next, label++ )
                {
                    CvScalar area_color = cvScalarAll(label);
                    cvDrawContours( labels, contours, area_color, area_color, -255, -1, 8 );
                }
            }
            else
            {
                int k = 1;
                for( int i = 0; i < src->rows; i++ )
                {
                    const uchar* srcptr = src->data.ptr + src->step*i;
                    int* labelptr = (int*)(labels->data.ptr + labels->step*i);
                    
                    for( int j = 0; j < src->cols; j++ )
                        if( srcptr[j] == 0 )
                            labelptr[j] = k++;
                }
            }

            icvDistanceTransformEx_5x5_C1R( src->data.ptr, src->step, temp->data.i, temp->step,
                        dst->data.fl, dst->step, labels->data.i, labels->step, size, _mask );
        }
    }
}

void cv::distanceTransform( InputArray _src, OutputArray _dst, OutputArray _labels,
                            int distanceType, int maskSize, int labelType )
{
    Mat src = _src.getMat();
    _dst.create(src.size(), CV_32F);
    _labels.create(src.size(), CV_32S);
    CvMat c_src = src, c_dst = _dst.getMat(), c_labels = _labels.getMat();
    cvDistTransform(&c_src, &c_dst, distanceType, maskSize, 0, &c_labels, labelType);
}

void cv::distanceTransform( InputArray _src, OutputArray _dst,
                            int distanceType, int maskSize )
{
    Mat src = _src.getMat();
    _dst.create(src.size(), CV_32F);
    Mat dst = _dst.getMat();
    CvMat c_src = src, c_dst = _dst.getMat();
    cvDistTransform(&c_src, &c_dst, distanceType, maskSize, 0, 0, -1);
}

/* End of file. */
