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

#ifdef HAVE_CONFIG_H
  #include "cvconfig.h"
#endif

#ifdef HAVE_MALLOC_H
  #include <malloc.h>
#endif

#ifdef HAVE_MEMORY_H
  #include <memory.h>
#endif

#ifdef _OPENMP
  #include <omp.h>
#endif /* _OPENMP */

#include <cstdio>
#include <cfloat>
#include <cmath>
#include <ctime>
#include <climits>

#include "_cvcommon.h"
#include "cvclassifier.h"

#ifdef _OPENMP
#include "omp.h"
#endif

#define CV_BOOST_IMPL

typedef struct CvValArray
{
    uchar* data;
    size_t step;
} CvValArray;

#define CMP_VALUES( idx1, idx2 )                                 \
    ( *( (float*) (aux->data + ((int) (idx1)) * aux->step ) ) <  \
      *( (float*) (aux->data + ((int) (idx2)) * aux->step ) ) )

CV_IMPLEMENT_QSORT_EX( icvSortIndexedValArray_16s, short, CMP_VALUES, CvValArray* )

CV_IMPLEMENT_QSORT_EX( icvSortIndexedValArray_32s, int,   CMP_VALUES, CvValArray* )

CV_IMPLEMENT_QSORT_EX( icvSortIndexedValArray_32f, float, CMP_VALUES, CvValArray* )

CV_BOOST_IMPL
void cvGetSortedIndices( CvMat* val, CvMat* idx, int sortcols )
{
    int idxtype = 0;
    size_t istep = 0;
    size_t jstep = 0;

    int i = 0;
    int j = 0;

    CvValArray va;

    CV_Assert( idx != NULL );
    CV_Assert( val != NULL );

    idxtype = CV_MAT_TYPE( idx->type );
    CV_Assert( idxtype == CV_16SC1 || idxtype == CV_32SC1 || idxtype == CV_32FC1 );
    CV_Assert( CV_MAT_TYPE( val->type ) == CV_32FC1 );
    if( sortcols )
    {
        CV_Assert( idx->rows == val->cols );
        CV_Assert( idx->cols == val->rows );
        istep = CV_ELEM_SIZE( val->type );
        jstep = val->step;
    }
    else
    {
        CV_Assert( idx->rows == val->rows );
        CV_Assert( idx->cols == val->cols );
        istep = val->step;
        jstep = CV_ELEM_SIZE( val->type );
    }

    va.data = val->data.ptr;
    va.step = jstep;
    switch( idxtype )
    {
        case CV_16SC1:
            for( i = 0; i < idx->rows; i++ )
            {
                for( j = 0; j < idx->cols; j++ )
                {
                    CV_MAT_ELEM( *idx, short, i, j ) = (short) j;
                }
                icvSortIndexedValArray_16s( (short*) (idx->data.ptr + (size_t)i * idx->step),
                                            idx->cols, &va );
                va.data += istep;
            }
            break;

        case CV_32SC1:
            for( i = 0; i < idx->rows; i++ )
            {
                for( j = 0; j < idx->cols; j++ )
                {
                    CV_MAT_ELEM( *idx, int, i, j ) = j;
                }
                icvSortIndexedValArray_32s( (int*) (idx->data.ptr + (size_t)i * idx->step),
                                            idx->cols, &va );
                va.data += istep;
            }
            break;

        case CV_32FC1:
            for( i = 0; i < idx->rows; i++ )
            {
                for( j = 0; j < idx->cols; j++ )
                {
                    CV_MAT_ELEM( *idx, float, i, j ) = (float) j;
                }
                icvSortIndexedValArray_32f( (float*) (idx->data.ptr + (size_t)i * idx->step),
                                            idx->cols, &va );
                va.data += istep;
            }
            break;

        default:
            assert( 0 );
            break;
    }
}

CV_BOOST_IMPL
void cvReleaseStumpClassifier( CvClassifier** classifier )
{
    cvFree( classifier );
    *classifier = 0;
}

CV_BOOST_IMPL
float cvEvalStumpClassifier( CvClassifier* classifier, CvMat* sample )
{
    assert( classifier != NULL );
    assert( sample != NULL );
    assert( CV_MAT_TYPE( sample->type ) == CV_32FC1 );
    
    if( (CV_MAT_ELEM( (*sample), float, 0,
            ((CvStumpClassifier*) classifier)->compidx )) <
        ((CvStumpClassifier*) classifier)->threshold ) 
        return ((CvStumpClassifier*) classifier)->left;
    return ((CvStumpClassifier*) classifier)->right;
}

#define ICV_DEF_FIND_STUMP_THRESHOLD( suffix, type, error )                              \
CV_BOOST_IMPL int icvFindStumpThreshold_##suffix(                                              \
        uchar* data, size_t datastep,                                                    \
        uchar* wdata, size_t wstep,                                                      \
        uchar* ydata, size_t ystep,                                                      \
        uchar* idxdata, size_t idxstep, int num,                                         \
        float* lerror,                                                                   \
        float* rerror,                                                                   \
        float* threshold, float* left, float* right,                                     \
        float* sumw, float* sumwy, float* sumwyy )                                       \
{                                                                                        \
    int found = 0;                                                                       \
    float wyl  = 0.0F;                                                                   \
    float wl   = 0.0F;                                                                   \
    float wyyl = 0.0F;                                                                   \
    float wyr  = 0.0F;                                                                   \
    float wr   = 0.0F;                                                                   \
                                                                                         \
    float curleft  = 0.0F;                                                               \
    float curright = 0.0F;                                                               \
    float* prevval = NULL;                                                               \
    float* curval  = NULL;                                                               \
    float curlerror = 0.0F;                                                              \
    float currerror = 0.0F;                                                              \
    float wposl;                                                                         \
    float wposr;                                                                         \
                                                                                         \
    int i = 0;                                                                           \
    int idx = 0;                                                                         \
                                                                                         \
    wposl = wposr = 0.0F;                                                                \
    if( *sumw == FLT_MAX )                                                               \
    {                                                                                    \
        /* calculate sums */                                                             \
        float *y = NULL;                                                                 \
        float *w = NULL;                                                                 \
        float wy = 0.0F;                                                                 \
                                                                                         \
        *sumw   = 0.0F;                                                                  \
        *sumwy  = 0.0F;                                                                  \
        *sumwyy = 0.0F;                                                                  \
        for( i = 0; i < num; i++ )                                                       \
        {                                                                                \
            idx = (int) ( *((type*) (idxdata + i*idxstep)) );                            \
            w = (float*) (wdata + idx * wstep);                                          \
            *sumw += *w;                                                                 \
            y = (float*) (ydata + idx * ystep);                                          \
            wy = (*w) * (*y);                                                            \
            *sumwy += wy;                                                                \
            *sumwyy += wy * (*y);                                                        \
        }                                                                                \
    }                                                                                    \
                                                                                         \
    for( i = 0; i < num; i++ )                                                           \
    {                                                                                    \
        idx = (int) ( *((type*) (idxdata + i*idxstep)) );                                \
        curval = (float*) (data + idx * datastep);                                       \
         /* for debug purpose */                                                         \
        if( i > 0 ) assert( (*prevval) <= (*curval) );                                   \
                                                                                         \
        wyr  = *sumwy - wyl;                                                             \
        wr   = *sumw  - wl;                                                              \
                                                                                         \
        if( wl > 0.0 ) curleft = wyl / wl;                                               \
        else curleft = 0.0F;                                                             \
                                                                                         \
        if( wr > 0.0 ) curright = wyr / wr;                                              \
        else curright = 0.0F;                                                            \
                                                                                         \
        error                                                                            \
                                                                                         \
        if( curlerror + currerror < (*lerror) + (*rerror) )                              \
        {                                                                                \
            (*lerror) = curlerror;                                                       \
            (*rerror) = currerror;                                                       \
            *threshold = *curval;                                                        \
            if( i > 0 ) {                                                                \
                *threshold = 0.5F * (*threshold + *prevval);                             \
            }                                                                            \
            *left  = curleft;                                                            \
            *right = curright;                                                           \
            found = 1;                                                                   \
        }                                                                                \
                                                                                         \
        do                                                                               \
        {                                                                                \
            wl  += *((float*) (wdata + idx * wstep));                                    \
            wyl += (*((float*) (wdata + idx * wstep)))                                   \
                * (*((float*) (ydata + idx * ystep)));                                   \
            wyyl += *((float*) (wdata + idx * wstep))                                    \
                * (*((float*) (ydata + idx * ystep)))                                    \
                * (*((float*) (ydata + idx * ystep)));                                   \
        }                                                                                \
        while( (++i) < num &&                                                            \
            ( *((float*) (data + (idx =                                                  \
                (int) ( *((type*) (idxdata + i*idxstep))) ) * datastep))                 \
                == *curval ) );                                                          \
        --i;                                                                             \
        prevval = curval;                                                                \
    } /* for each value */                                                               \
                                                                                         \
    return found;                                                                        \
}

/* misclassification error
 * err = MIN( wpos, wneg );
 */
#define ICV_DEF_FIND_STUMP_THRESHOLD_MISC( suffix, type )                                \
    ICV_DEF_FIND_STUMP_THRESHOLD( misc_##suffix, type,                                   \
        wposl = 0.5F * ( wl + wyl );                                                     \
        wposr = 0.5F * ( wr + wyr );                                                     \
        curleft = 0.5F * ( 1.0F + curleft );                                             \
        curright = 0.5F * ( 1.0F + curright );                                           \
        curlerror = MIN( wposl, wl - wposl );                                            \
        currerror = MIN( wposr, wr - wposr );                                            \
    )

/* gini error
 * err = 2 * wpos * wneg /(wpos + wneg)
 */
#define ICV_DEF_FIND_STUMP_THRESHOLD_GINI( suffix, type )                                \
    ICV_DEF_FIND_STUMP_THRESHOLD( gini_##suffix, type,                                   \
        wposl = 0.5F * ( wl + wyl );                                                     \
        wposr = 0.5F * ( wr + wyr );                                                     \
        curleft = 0.5F * ( 1.0F + curleft );                                             \
        curright = 0.5F * ( 1.0F + curright );                                           \
        curlerror = 2.0F * wposl * ( 1.0F - curleft );                                   \
        currerror = 2.0F * wposr * ( 1.0F - curright );                                  \
    )

#define CV_ENTROPY_THRESHOLD FLT_MIN

/* entropy error
 * err = - wpos * log(wpos / (wpos + wneg)) - wneg * log(wneg / (wpos + wneg))
 */
#define ICV_DEF_FIND_STUMP_THRESHOLD_ENTROPY( suffix, type )                             \
    ICV_DEF_FIND_STUMP_THRESHOLD( entropy_##suffix, type,                                \
        wposl = 0.5F * ( wl + wyl );                                                     \
        wposr = 0.5F * ( wr + wyr );                                                     \
        curleft = 0.5F * ( 1.0F + curleft );                                             \
        curright = 0.5F * ( 1.0F + curright );                                           \
        curlerror = currerror = 0.0F;                                                    \
        if( curleft > CV_ENTROPY_THRESHOLD )                                             \
            curlerror -= wposl * logf( curleft );                                        \
        if( curleft < 1.0F - CV_ENTROPY_THRESHOLD )                                      \
            curlerror -= (wl - wposl) * logf( 1.0F - curleft );                          \
                                                                                         \
        if( curright > CV_ENTROPY_THRESHOLD )                                            \
            currerror -= wposr * logf( curright );                                       \
        if( curright < 1.0F - CV_ENTROPY_THRESHOLD )                                     \
            currerror -= (wr - wposr) * logf( 1.0F - curright );                         \
    )

/* least sum of squares error */
#define ICV_DEF_FIND_STUMP_THRESHOLD_SQ( suffix, type )                                  \
    ICV_DEF_FIND_STUMP_THRESHOLD( sq_##suffix, type,                                     \
        /* calculate error (sum of squares)          */                                  \
        /* err = sum( w * (y - left(rigt)Val)^2 )    */                                  \
        curlerror = wyyl + curleft * curleft * wl - 2.0F * curleft * wyl;                \
        currerror = (*sumwyy) - wyyl + curright * curright * wr - 2.0F * curright * wyr; \
    )

ICV_DEF_FIND_STUMP_THRESHOLD_MISC( 16s, short )

ICV_DEF_FIND_STUMP_THRESHOLD_MISC( 32s, int )

ICV_DEF_FIND_STUMP_THRESHOLD_MISC( 32f, float )


ICV_DEF_FIND_STUMP_THRESHOLD_GINI( 16s, short )

ICV_DEF_FIND_STUMP_THRESHOLD_GINI( 32s, int )

ICV_DEF_FIND_STUMP_THRESHOLD_GINI( 32f, float )


ICV_DEF_FIND_STUMP_THRESHOLD_ENTROPY( 16s, short )

ICV_DEF_FIND_STUMP_THRESHOLD_ENTROPY( 32s, int )

ICV_DEF_FIND_STUMP_THRESHOLD_ENTROPY( 32f, float )


ICV_DEF_FIND_STUMP_THRESHOLD_SQ( 16s, short )

ICV_DEF_FIND_STUMP_THRESHOLD_SQ( 32s, int )

ICV_DEF_FIND_STUMP_THRESHOLD_SQ( 32f, float )

typedef int (*CvFindThresholdFunc)( uchar* data, size_t datastep,
                                    uchar* wdata, size_t wstep,
                                    uchar* ydata, size_t ystep,
                                    uchar* idxdata, size_t idxstep, int num,
                                    float* lerror,
                                    float* rerror,
                                    float* threshold, float* left, float* right,
                                    float* sumw, float* sumwy, float* sumwyy );

CvFindThresholdFunc findStumpThreshold_16s[4] = {
        icvFindStumpThreshold_misc_16s,
        icvFindStumpThreshold_gini_16s,
        icvFindStumpThreshold_entropy_16s,
        icvFindStumpThreshold_sq_16s
    };

CvFindThresholdFunc findStumpThreshold_32s[4] = {
        icvFindStumpThreshold_misc_32s,
        icvFindStumpThreshold_gini_32s,
        icvFindStumpThreshold_entropy_32s,
        icvFindStumpThreshold_sq_32s
    };

CvFindThresholdFunc findStumpThreshold_32f[4] = {
        icvFindStumpThreshold_misc_32f,
        icvFindStumpThreshold_gini_32f,
        icvFindStumpThreshold_entropy_32f,
        icvFindStumpThreshold_sq_32f
    };

CV_BOOST_IMPL
CvClassifier* cvCreateStumpClassifier( CvMat* trainData,
                      int flags,
                      CvMat* trainClasses,
                      CvMat* /*typeMask*/,
                      CvMat* missedMeasurementsMask,
                      CvMat* compIdx,
                      CvMat* sampleIdx,
                      CvMat* weights,
                      CvClassifierTrainParams* trainParams
                    )
{
    CvStumpClassifier* stump = NULL;
    int m = 0; /* number of samples */
    int n = 0; /* number of components */
    uchar* data = NULL;
    int cstep   = 0;
    int sstep   = 0;
    uchar* ydata = NULL;
    int ystep    = 0;
    uchar* idxdata = NULL;
    int idxstep    = 0;
    int l = 0; /* number of indices */     
    uchar* wdata = NULL;
    int wstep    = 0;

    int* idx = NULL;
    int i = 0;
    
    float sumw   = FLT_MAX;
    float sumwy  = FLT_MAX;
    float sumwyy = FLT_MAX;

    CV_Assert( trainData != NULL );
    CV_Assert( CV_MAT_TYPE( trainData->type ) == CV_32FC1 );
    CV_Assert( trainClasses != NULL );
    CV_Assert( CV_MAT_TYPE( trainClasses->type ) == CV_32FC1 );
    CV_Assert( missedMeasurementsMask == NULL );
    CV_Assert( compIdx == NULL );
    CV_Assert( weights != NULL );
    CV_Assert( CV_MAT_TYPE( weights->type ) == CV_32FC1 );
    CV_Assert( trainParams != NULL );

    data = trainData->data.ptr;
    if( CV_IS_ROW_SAMPLE( flags ) )
    {
        cstep = CV_ELEM_SIZE( trainData->type );
        sstep = trainData->step;
        m = trainData->rows;
        n = trainData->cols;
    }
    else
    {
        sstep = CV_ELEM_SIZE( trainData->type );
        cstep = trainData->step;
        m = trainData->cols;
        n = trainData->rows;
    }

    ydata = trainClasses->data.ptr;
    if( trainClasses->rows == 1 )
    {
        assert( trainClasses->cols == m );
        ystep = CV_ELEM_SIZE( trainClasses->type );
    }
    else
    {
        assert( trainClasses->rows == m );
        ystep = trainClasses->step;
    }

    wdata = weights->data.ptr;
    if( weights->rows == 1 )
    {
        assert( weights->cols == m );
        wstep = CV_ELEM_SIZE( weights->type );
    }
    else
    {
        assert( weights->rows == m );
        wstep = weights->step;
    }

    l = m;
    if( sampleIdx != NULL )
    {
        assert( CV_MAT_TYPE( sampleIdx->type ) == CV_32FC1 );

        idxdata = sampleIdx->data.ptr;
        if( sampleIdx->rows == 1 )
        {
            l = sampleIdx->cols;
            idxstep = CV_ELEM_SIZE( sampleIdx->type );
        }
        else
        {
            l = sampleIdx->rows;
            idxstep = sampleIdx->step;
        }
        assert( l <= m );
    }

    idx = (int*) cvAlloc( l * sizeof( int ) );
    stump = (CvStumpClassifier*) cvAlloc( sizeof( CvStumpClassifier) );

    /* START */
    memset( (void*) stump, 0, sizeof( CvStumpClassifier ) );

    stump->eval = cvEvalStumpClassifier;
    stump->tune = NULL;
    stump->save = NULL;
    stump->release = cvReleaseStumpClassifier;

    stump->lerror = FLT_MAX;
    stump->rerror = FLT_MAX;
    stump->left  = 0.0F;
    stump->right = 0.0F;

    /* copy indices */
    if( sampleIdx != NULL )
    {
        for( i = 0; i < l; i++ )
        {
            idx[i] = (int) *((float*) (idxdata + i*idxstep));
        }
    }
    else
    {
        for( i = 0; i < l; i++ )
        {
            idx[i] = i;
        }
    }

    for( i = 0; i < n; i++ )
    {
        CvValArray va;

        va.data = data + i * ((size_t) cstep);
        va.step = sstep;
        icvSortIndexedValArray_32s( idx, l, &va );
        if( findStumpThreshold_32s[(int) ((CvStumpTrainParams*) trainParams)->error]
              ( data + i * ((size_t) cstep), sstep,
                wdata, wstep, ydata, ystep, (uchar*) idx, sizeof( int ), l,
                &(stump->lerror), &(stump->rerror),
                &(stump->threshold), &(stump->left), &(stump->right), 
                &sumw, &sumwy, &sumwyy ) )
        {
            stump->compidx = i;
        }
    } /* for each component */

    /* END */

    cvFree( &idx );

    if( ((CvStumpTrainParams*) trainParams)->type == CV_CLASSIFICATION_CLASS )
    {
        stump->left = 2.0F * (stump->left >= 0.5F) - 1.0F;
        stump->right = 2.0F * (stump->right >= 0.5F) - 1.0F;
    }

    return (CvClassifier*) stump;
}

/*
 * cvCreateMTStumpClassifier
 *
 * Multithreaded stump classifier constructor
 * Includes huge train data support through callback function
 */
CV_BOOST_IMPL
CvClassifier* cvCreateMTStumpClassifier( CvMat* trainData,
                      int flags,
                      CvMat* trainClasses,
                      CvMat* /*typeMask*/,
                      CvMat* missedMeasurementsMask,
                      CvMat* compIdx,
                      CvMat* sampleIdx,
                      CvMat* weights,
                      CvClassifierTrainParams* trainParams )
{
    CvStumpClassifier* stump = NULL;
    int m = 0; /* number of samples */
    int n = 0; /* number of components */
    uchar* data = NULL;
    size_t cstep   = 0;
    size_t sstep   = 0;
    int    datan   = 0; /* num components */
    uchar* ydata = NULL;
    size_t ystep = 0;
    uchar* idxdata = NULL;
    size_t idxstep = 0;
    int    l = 0; /* number of indices */     
    uchar* wdata = NULL;
    size_t wstep = 0;

    uchar* sorteddata = NULL;
    int    sortedtype    = 0;
    size_t sortedcstep   = 0; /* component step */
    size_t sortedsstep   = 0; /* sample step */
    int    sortedn       = 0; /* num components */
    int    sortedm       = 0; /* num samples */

    char* filter = NULL;
    int i = 0;
    
    int compidx = 0;
    int stumperror;
    int portion;

    /* private variables */
    CvMat mat;
    CvValArray va;
    float lerror;
    float rerror;
    float left;
    float right;
    float threshold;
    int optcompidx;

    float sumw;
    float sumwy;
    float sumwyy;

    int t_compidx;
    int t_n;
    
    int ti;
    int tj;
    int tk;

    uchar* t_data;
    size_t t_cstep;
    size_t t_sstep;

    size_t matcstep;
    size_t matsstep;

    int* t_idx;
    /* end private variables */

    CV_Assert( trainParams != NULL );
    CV_Assert( trainClasses != NULL );
    CV_Assert( CV_MAT_TYPE( trainClasses->type ) == CV_32FC1 );
    CV_Assert( missedMeasurementsMask == NULL );
    CV_Assert( compIdx == NULL );

    stumperror = (int) ((CvMTStumpTrainParams*) trainParams)->error;

    ydata = trainClasses->data.ptr;
    if( trainClasses->rows == 1 )
    {
        m = trainClasses->cols;
        ystep = CV_ELEM_SIZE( trainClasses->type );
    }
    else
    {
        m = trainClasses->rows;
        ystep = trainClasses->step;
    }

    wdata = weights->data.ptr;
    if( weights->rows == 1 )
    {
        CV_Assert( weights->cols == m );
        wstep = CV_ELEM_SIZE( weights->type );
    }
    else
    {
        CV_Assert( weights->rows == m );
        wstep = weights->step;
    }

    if( ((CvMTStumpTrainParams*) trainParams)->sortedIdx != NULL )
    {
        sortedtype =
            CV_MAT_TYPE( ((CvMTStumpTrainParams*) trainParams)->sortedIdx->type );
        assert( sortedtype == CV_16SC1 || sortedtype == CV_32SC1
                || sortedtype == CV_32FC1 );
        sorteddata = ((CvMTStumpTrainParams*) trainParams)->sortedIdx->data.ptr;
        sortedsstep = CV_ELEM_SIZE( sortedtype );
        sortedcstep = ((CvMTStumpTrainParams*) trainParams)->sortedIdx->step;
        sortedn = ((CvMTStumpTrainParams*) trainParams)->sortedIdx->rows;
        sortedm = ((CvMTStumpTrainParams*) trainParams)->sortedIdx->cols;
    }

    if( trainData == NULL )
    {
        assert( ((CvMTStumpTrainParams*) trainParams)->getTrainData != NULL );
        n = ((CvMTStumpTrainParams*) trainParams)->numcomp;
        assert( n > 0 );
    }
    else
    {
        assert( CV_MAT_TYPE( trainData->type ) == CV_32FC1 );
        data = trainData->data.ptr;
        if( CV_IS_ROW_SAMPLE( flags ) )
        {
            cstep = CV_ELEM_SIZE( trainData->type );
            sstep = trainData->step;
            assert( m == trainData->rows );
            datan = n = trainData->cols;
        }
        else
        {
            sstep = CV_ELEM_SIZE( trainData->type );
            cstep = trainData->step;
            assert( m == trainData->cols );
            datan = n = trainData->rows;
        }
        if( ((CvMTStumpTrainParams*) trainParams)->getTrainData != NULL )
        {
            n = ((CvMTStumpTrainParams*) trainParams)->numcomp;
        }        
    }
    assert( datan <= n );

    if( sampleIdx != NULL )
    {
        assert( CV_MAT_TYPE( sampleIdx->type ) == CV_32FC1 );
        idxdata = sampleIdx->data.ptr;
        idxstep = ( sampleIdx->rows == 1 )
            ? CV_ELEM_SIZE( sampleIdx->type ) : sampleIdx->step;
        l = ( sampleIdx->rows == 1 ) ? sampleIdx->cols : sampleIdx->rows;

        if( sorteddata != NULL )
        {
            filter = (char*) cvAlloc( sizeof( char ) * m );
            memset( (void*) filter, 0, sizeof( char ) * m );
            for( i = 0; i < l; i++ )
            {
                filter[(int) *((float*) (idxdata + i * idxstep))] = (char) 1;
            }
        }
    }
    else
    {
        l = m;
    }

    stump = (CvStumpClassifier*) cvAlloc( sizeof( CvStumpClassifier) );

    /* START */
    memset( (void*) stump, 0, sizeof( CvStumpClassifier ) );

    portion = ((CvMTStumpTrainParams*)trainParams)->portion;
    
    if( portion < 1 )
    {
        /* auto portion */
        portion = n;
        #ifdef _OPENMP
        portion /= omp_get_max_threads();        
        #endif /* _OPENMP */        
    }

    stump->eval = cvEvalStumpClassifier;
    stump->tune = NULL;
    stump->save = NULL;
    stump->release = cvReleaseStumpClassifier;

    stump->lerror = FLT_MAX;
    stump->rerror = FLT_MAX;
    stump->left  = 0.0F;
    stump->right = 0.0F;

    compidx = 0;
    #ifdef _OPENMP
    #pragma omp parallel private(mat, va, lerror, rerror, left, right, threshold, \
                                 optcompidx, sumw, sumwy, sumwyy, t_compidx, t_n, \
                                 ti, tj, tk, t_data, t_cstep, t_sstep, matcstep,  \
                                 matsstep, t_idx)
    #endif /* _OPENMP */
    {
        lerror = FLT_MAX;
        rerror = FLT_MAX;
        left  = 0.0F;
        right = 0.0F;
        threshold = 0.0F;
        optcompidx = 0;

        sumw   = FLT_MAX;
        sumwy  = FLT_MAX;
        sumwyy = FLT_MAX;

        t_compidx = 0;
        t_n = 0;
        
        ti = 0;
        tj = 0;
        tk = 0;

        t_data = NULL;
        t_cstep = 0;
        t_sstep = 0;

        matcstep = 0;
        matsstep = 0;

        t_idx = NULL;

        mat.data.ptr = NULL;
        
        if( datan < n )
        {
            /* prepare matrix for callback */
            if( CV_IS_ROW_SAMPLE( flags ) )
            {
                mat = cvMat( m, portion, CV_32FC1, 0 );
                matcstep = CV_ELEM_SIZE( mat.type );
                matsstep = mat.step;
            }
            else
            {
                mat = cvMat( portion, m, CV_32FC1, 0 );
                matcstep = mat.step;
                matsstep = CV_ELEM_SIZE( mat.type );
            }
            mat.data.ptr = (uchar*) cvAlloc( sizeof( float ) * mat.rows * mat.cols );
        }

        if( filter != NULL || sortedn < n )
        {
            t_idx = (int*) cvAlloc( sizeof( int ) * m );
            if( sortedn == 0 || filter == NULL )
            {
                if( idxdata != NULL )
                {
                    for( ti = 0; ti < l; ti++ )
                    {
                        t_idx[ti] = (int) *((float*) (idxdata + ti * idxstep));
                    }
                }
                else
                {
                    for( ti = 0; ti < l; ti++ )
                    {
                        t_idx[ti] = ti;
                    }
                }                
            }
        }

        #ifdef _OPENMP
        #pragma omp critical(c_compidx)
        #endif /* _OPENMP */
        {
            t_compidx = compidx;
            compidx += portion;
        }
        while( t_compidx < n )
        {
            t_n = portion;
            if( t_compidx < datan )
            {
                t_n = ( t_n < (datan - t_compidx) ) ? t_n : (datan - t_compidx);
                t_data = data;
                t_cstep = cstep;
                t_sstep = sstep;
            }
            else
            {
                t_n = ( t_n < (n - t_compidx) ) ? t_n : (n - t_compidx);
                t_cstep = matcstep;
                t_sstep = matsstep;
                t_data = mat.data.ptr - t_compidx * ((size_t) t_cstep );

                /* calculate components */
                ((CvMTStumpTrainParams*)trainParams)->getTrainData( &mat,
                        sampleIdx, compIdx, t_compidx, t_n,
                        ((CvMTStumpTrainParams*)trainParams)->userdata );
            }

            if( sorteddata != NULL )
            {
                if( filter != NULL )
                {
                    /* have sorted indices and filter */
                    switch( sortedtype )
                    {
                        case CV_16SC1:
                            for( ti = t_compidx; ti < MIN( sortedn, t_compidx + t_n ); ti++ )
                            {
                                tk = 0;
                                for( tj = 0; tj < sortedm; tj++ )
                                {
                                    int curidx = (int) ( *((short*) (sorteddata
                                            + ti * sortedcstep + tj * sortedsstep)) );
                                    if( filter[curidx] != 0 )
                                    {
                                        t_idx[tk++] = curidx;
                                    }
                                }
                                if( findStumpThreshold_32s[stumperror]( 
                                        t_data + ti * t_cstep, t_sstep,
                                        wdata, wstep, ydata, ystep,
                                        (uchar*) t_idx, sizeof( int ), tk,
                                        &lerror, &rerror,
                                        &threshold, &left, &right, 
                                        &sumw, &sumwy, &sumwyy ) )
                                {
                                    optcompidx = ti;
                                }
                            }
                            break;
                        case CV_32SC1:
                            for( ti = t_compidx; ti < MIN( sortedn, t_compidx + t_n ); ti++ )
                            {
                                tk = 0;
                                for( tj = 0; tj < sortedm; tj++ )
                                {
                                    int curidx = (int) ( *((int*) (sorteddata
                                            + ti * sortedcstep + tj * sortedsstep)) );
                                    if( filter[curidx] != 0 )
                                    {
                                        t_idx[tk++] = curidx;
                                    }
                                }
                                if( findStumpThreshold_32s[stumperror]( 
                                        t_data + ti * t_cstep, t_sstep,
                                        wdata, wstep, ydata, ystep,
                                        (uchar*) t_idx, sizeof( int ), tk,
                                        &lerror, &rerror,
                                        &threshold, &left, &right, 
                                        &sumw, &sumwy, &sumwyy ) )
                                {
                                    optcompidx = ti;
                                }
                            }
                            break;
                        case CV_32FC1:
                            for( ti = t_compidx; ti < MIN( sortedn, t_compidx + t_n ); ti++ )
                            {
                                tk = 0;
                                for( tj = 0; tj < sortedm; tj++ )
                                {
                                    int curidx = (int) ( *((float*) (sorteddata
                                            + ti * sortedcstep + tj * sortedsstep)) );
                                    if( filter[curidx] != 0 )
                                    {
                                        t_idx[tk++] = curidx;
                                    }
                                }
                                if( findStumpThreshold_32s[stumperror]( 
                                        t_data + ti * t_cstep, t_sstep,
                                        wdata, wstep, ydata, ystep,
                                        (uchar*) t_idx, sizeof( int ), tk,
                                        &lerror, &rerror,
                                        &threshold, &left, &right, 
                                        &sumw, &sumwy, &sumwyy ) )
                                {
                                    optcompidx = ti;
                                }
                            }
                            break;
                        default:
                            assert( 0 );
                            break;
                    }
                }
                else
                {
                    /* have sorted indices */
                    switch( sortedtype )
                    {
                        case CV_16SC1:
                            for( ti = t_compidx; ti < MIN( sortedn, t_compidx + t_n ); ti++ )
                            {
                                if( findStumpThreshold_16s[stumperror]( 
                                        t_data + ti * t_cstep, t_sstep,
                                        wdata, wstep, ydata, ystep,
                                        sorteddata + ti * sortedcstep, sortedsstep, sortedm,
                                        &lerror, &rerror,
                                        &threshold, &left, &right, 
                                        &sumw, &sumwy, &sumwyy ) )
                                {
                                    optcompidx = ti;
                                }
                            }
                            break;
                        case CV_32SC1:
                            for( ti = t_compidx; ti < MIN( sortedn, t_compidx + t_n ); ti++ )
                            {
                                if( findStumpThreshold_32s[stumperror]( 
                                        t_data + ti * t_cstep, t_sstep,
                                        wdata, wstep, ydata, ystep,
                                        sorteddata + ti * sortedcstep, sortedsstep, sortedm,
                                        &lerror, &rerror,
                                        &threshold, &left, &right, 
                                        &sumw, &sumwy, &sumwyy ) )
                                {
                                    optcompidx = ti;
                                }
                            }
                            break;
                        case CV_32FC1:
                            for( ti = t_compidx; ti < MIN( sortedn, t_compidx + t_n ); ti++ )
                            {
                                if( findStumpThreshold_32f[stumperror]( 
                                        t_data + ti * t_cstep, t_sstep,
                                        wdata, wstep, ydata, ystep,
                                        sorteddata + ti * sortedcstep, sortedsstep, sortedm,
                                        &lerror, &rerror,
                                        &threshold, &left, &right, 
                                        &sumw, &sumwy, &sumwyy ) )
                                {
                                    optcompidx = ti;
                                }
                            }
                            break;
                        default:
                            assert( 0 );
                            break;
                    }
                }
            }

            ti = MAX( t_compidx, MIN( sortedn, t_compidx + t_n ) );
            for( ; ti < t_compidx + t_n; ti++ )
            {
                va.data = t_data + ti * t_cstep;
                va.step = t_sstep;
                icvSortIndexedValArray_32s( t_idx, l, &va );
                if( findStumpThreshold_32s[stumperror]( 
                        t_data + ti * t_cstep, t_sstep,
                        wdata, wstep, ydata, ystep,
                        (uchar*)t_idx, sizeof( int ), l,
                        &lerror, &rerror,
                        &threshold, &left, &right, 
                        &sumw, &sumwy, &sumwyy ) )
                {
                    optcompidx = ti;
                }
            }
            #ifdef _OPENMP
            #pragma omp critical(c_compidx)
            #endif /* _OPENMP */
            {
                t_compidx = compidx;
                compidx += portion;
            }
        } /* while have training data */

        /* get the best classifier */
        #ifdef _OPENMP
        #pragma omp critical(c_beststump)
        #endif /* _OPENMP */
        {
            if( lerror + rerror < stump->lerror + stump->rerror )
            {
                stump->lerror    = lerror;
                stump->rerror    = rerror;
                stump->compidx   = optcompidx;
                stump->threshold = threshold;
                stump->left      = left;
                stump->right     = right;
            }
        }

        /* free allocated memory */
        if( mat.data.ptr != NULL )
        {
            cvFree( &(mat.data.ptr) );
        }
        if( t_idx != NULL )
        {
            cvFree( &t_idx );
        }
    } /* end of parallel region */

    /* END */

    /* free allocated memory */
    if( filter != NULL )
    {
        cvFree( &filter );
    }

    if( ((CvMTStumpTrainParams*) trainParams)->type == CV_CLASSIFICATION_CLASS )
    {
        stump->left = 2.0F * (stump->left >= 0.5F) - 1.0F;
        stump->right = 2.0F * (stump->right >= 0.5F) - 1.0F;
    }

    return (CvClassifier*) stump;
}

CV_BOOST_IMPL
float cvEvalCARTClassifier( CvClassifier* classifier, CvMat* sample )
{
    CV_FUNCNAME( "cvEvalCARTClassifier" );

    int idx = 0;

    __BEGIN__;


    CV_ASSERT( classifier != NULL );
    CV_ASSERT( sample != NULL );
    CV_ASSERT( CV_MAT_TYPE( sample->type ) == CV_32FC1 );
    CV_ASSERT( sample->rows == 1 || sample->cols == 1 );

    if( sample->rows == 1 )
    {
        do
        {
            if( (CV_MAT_ELEM( (*sample), float, 0,
                    ((CvCARTClassifier*) classifier)->compidx[idx] )) <
                ((CvCARTClassifier*) classifier)->threshold[idx] ) 
            {
                idx = ((CvCARTClassifier*) classifier)->left[idx];
            }
            else
            {
                idx = ((CvCARTClassifier*) classifier)->right[idx];
            }
        } while( idx > 0 );
    }
    else
    {
        do
        {
            if( (CV_MAT_ELEM( (*sample), float,
                    ((CvCARTClassifier*) classifier)->compidx[idx], 0 )) <
                ((CvCARTClassifier*) classifier)->threshold[idx] ) 
            {
                idx = ((CvCARTClassifier*) classifier)->left[idx];
            }
            else
            {
                idx = ((CvCARTClassifier*) classifier)->right[idx];
            }
        } while( idx > 0 );
    } 

    __END__;

    return ((CvCARTClassifier*) classifier)->val[-idx];
}

CV_BOOST_IMPL
float cvEvalCARTClassifierIdx( CvClassifier* classifier, CvMat* sample )
{
    CV_FUNCNAME( "cvEvalCARTClassifierIdx" );

    int idx = 0;

    __BEGIN__;


    CV_ASSERT( classifier != NULL );
    CV_ASSERT( sample != NULL );
    CV_ASSERT( CV_MAT_TYPE( sample->type ) == CV_32FC1 );
    CV_ASSERT( sample->rows == 1 || sample->cols == 1 );

    if( sample->rows == 1 )
    {
        do
        {
            if( (CV_MAT_ELEM( (*sample), float, 0,
                    ((CvCARTClassifier*) classifier)->compidx[idx] )) <
                ((CvCARTClassifier*) classifier)->threshold[idx] ) 
            {
                idx = ((CvCARTClassifier*) classifier)->left[idx];
            }
            else
            {
                idx = ((CvCARTClassifier*) classifier)->right[idx];
            }
        } while( idx > 0 );
    }
    else
    {
        do
        {
            if( (CV_MAT_ELEM( (*sample), float,
                    ((CvCARTClassifier*) classifier)->compidx[idx], 0 )) <
                ((CvCARTClassifier*) classifier)->threshold[idx] ) 
            {
                idx = ((CvCARTClassifier*) classifier)->left[idx];
            }
            else
            {
                idx = ((CvCARTClassifier*) classifier)->right[idx];
            }
        } while( idx > 0 );
    } 

    __END__;

    return (float) (-idx);
}

CV_BOOST_IMPL
void cvReleaseCARTClassifier( CvClassifier** classifier )
{
    cvFree( classifier );
    *classifier = NULL;
}

void CV_CDECL icvDefaultSplitIdx_R( int compidx, float threshold,
                                    CvMat* idx, CvMat** left, CvMat** right,
                                    void* userdata )
{
    CvMat* trainData = (CvMat*) userdata;
    int i = 0;

    *left = cvCreateMat( 1, trainData->rows, CV_32FC1 );
    *right = cvCreateMat( 1, trainData->rows, CV_32FC1 );
    (*left)->cols = (*right)->cols = 0;
    if( idx == NULL )
    {
        for( i = 0; i < trainData->rows; i++ )
        {
            if( CV_MAT_ELEM( *trainData, float, i, compidx ) < threshold )
            {
                (*left)->data.fl[(*left)->cols++] = (float) i;
            }
            else
            {
                (*right)->data.fl[(*right)->cols++] = (float) i;
            }
        }
    }
    else
    {
        uchar* idxdata;
        int idxnum;
        int idxstep;
        int index;

        idxdata = idx->data.ptr;
        idxnum = (idx->rows == 1) ? idx->cols : idx->rows;
        idxstep = (idx->rows == 1) ? CV_ELEM_SIZE( idx->type ) : idx->step;
        for( i = 0; i < idxnum; i++ )
        {
            index = (int) *((float*) (idxdata + i * idxstep));
            if( CV_MAT_ELEM( *trainData, float, index, compidx ) < threshold )
            {
                (*left)->data.fl[(*left)->cols++] = (float) index;
            }
            else
            {
                (*right)->data.fl[(*right)->cols++] = (float) index;
            }
        }
    }
}

void CV_CDECL icvDefaultSplitIdx_C( int compidx, float threshold,
                                    CvMat* idx, CvMat** left, CvMat** right,
                                    void* userdata )
{
    CvMat* trainData = (CvMat*) userdata;
    int i = 0;

    *left = cvCreateMat( 1, trainData->cols, CV_32FC1 );
    *right = cvCreateMat( 1, trainData->cols, CV_32FC1 );
    (*left)->cols = (*right)->cols = 0;
    if( idx == NULL )
    {
        for( i = 0; i < trainData->cols; i++ )
        {
            if( CV_MAT_ELEM( *trainData, float, compidx, i ) < threshold )
            {
                (*left)->data.fl[(*left)->cols++] = (float) i;
            }
            else
            {
                (*right)->data.fl[(*right)->cols++] = (float) i;
            }
        }
    }
    else
    {
        uchar* idxdata;
        int idxnum;
        int idxstep;
        int index;

        idxdata = idx->data.ptr;
        idxnum = (idx->rows == 1) ? idx->cols : idx->rows;
        idxstep = (idx->rows == 1) ? CV_ELEM_SIZE( idx->type ) : idx->step;
        for( i = 0; i < idxnum; i++ )
        {
            index = (int) *((float*) (idxdata + i * idxstep));
            if( CV_MAT_ELEM( *trainData, float, compidx, index ) < threshold )
            {
                (*left)->data.fl[(*left)->cols++] = (float) index;
            }
            else
            {
                (*right)->data.fl[(*right)->cols++] = (float) index;
            }
        }
    }
}

/* internal structure used in CART creation */
typedef struct CvCARTNode
{
    CvMat* sampleIdx;
    CvStumpClassifier* stump;
    int parent;
    int leftflag;
    float errdrop;
} CvCARTNode;

CV_BOOST_IMPL
CvClassifier* cvCreateCARTClassifier( CvMat* trainData,
                     int flags,
                     CvMat* trainClasses,
                     CvMat* typeMask,
                     CvMat* missedMeasurementsMask,
                     CvMat* compIdx,
                     CvMat* sampleIdx,
                     CvMat* weights,
                     CvClassifierTrainParams* trainParams )
{
    CvCARTClassifier* cart = NULL;
    size_t datasize = 0;
    int count = 0;
    int i = 0;
    int j = 0;
    
    CvCARTNode* intnode = NULL;
    CvCARTNode* list = NULL;
    int listcount = 0;
    CvMat* lidx = NULL;
    CvMat* ridx = NULL;
    
    float maxerrdrop = 0.0F;
    int idx = 0;

    void (*splitIdxCallback)( int compidx, float threshold,
                              CvMat* idx, CvMat** left, CvMat** right,
                              void* userdata );
    void* userdata;

    count = ((CvCARTTrainParams*) trainParams)->count;
    
    assert( count > 0 );

    datasize = sizeof( *cart ) + (sizeof( float ) + 3 * sizeof( int )) * count + 
        sizeof( float ) * (count + 1);
    
    cart = (CvCARTClassifier*) cvAlloc( datasize );
    memset( cart, 0, datasize );
    
    cart->count = count;
    
    cart->eval = cvEvalCARTClassifier;
    cart->save = NULL;
    cart->release = cvReleaseCARTClassifier;

    cart->compidx = (int*) (cart + 1);
    cart->threshold = (float*) (cart->compidx + count);
    cart->left  = (int*) (cart->threshold + count);
    cart->right = (int*) (cart->left + count);
    cart->val = (float*) (cart->right + count);

    datasize = sizeof( CvCARTNode ) * (count + count);
    intnode = (CvCARTNode*) cvAlloc( datasize );
    memset( intnode, 0, datasize );
    list = (CvCARTNode*) (intnode + count);

    splitIdxCallback = ((CvCARTTrainParams*) trainParams)->splitIdx;
    userdata = ((CvCARTTrainParams*) trainParams)->userdata;
    if( splitIdxCallback == NULL )
    {
        splitIdxCallback = ( CV_IS_ROW_SAMPLE( flags ) )
            ? icvDefaultSplitIdx_R : icvDefaultSplitIdx_C;
        userdata = trainData;
    }

    /* create root of the tree */
    intnode[0].sampleIdx = sampleIdx;
    intnode[0].stump = (CvStumpClassifier*)
        ((CvCARTTrainParams*) trainParams)->stumpConstructor( trainData, flags,
            trainClasses, typeMask, missedMeasurementsMask, compIdx, sampleIdx, weights,
            ((CvCARTTrainParams*) trainParams)->stumpTrainParams );
    cart->left[0] = cart->right[0] = 0;

    /* build tree */
    listcount = 0;
    for( i = 1; i < count; i++ )
    {
        /* split last added node */
        splitIdxCallback( intnode[i-1].stump->compidx, intnode[i-1].stump->threshold,
            intnode[i-1].sampleIdx, &lidx, &ridx, userdata );
        
        if( intnode[i-1].stump->lerror != 0.0F )
        {
            list[listcount].sampleIdx = lidx;
            list[listcount].stump = (CvStumpClassifier*)
                ((CvCARTTrainParams*) trainParams)->stumpConstructor( trainData, flags,
                    trainClasses, typeMask, missedMeasurementsMask, compIdx,
                    list[listcount].sampleIdx,
                    weights, ((CvCARTTrainParams*) trainParams)->stumpTrainParams );
            list[listcount].errdrop = intnode[i-1].stump->lerror
                - (list[listcount].stump->lerror + list[listcount].stump->rerror);
            list[listcount].leftflag = 1;
            list[listcount].parent = i-1;
            listcount++;
        }
        else
        {
            cvReleaseMat( &lidx );
        }
        if( intnode[i-1].stump->rerror != 0.0F )
        {
            list[listcount].sampleIdx = ridx;
            list[listcount].stump = (CvStumpClassifier*)
                ((CvCARTTrainParams*) trainParams)->stumpConstructor( trainData, flags,
                    trainClasses, typeMask, missedMeasurementsMask, compIdx,
                    list[listcount].sampleIdx,
                    weights, ((CvCARTTrainParams*) trainParams)->stumpTrainParams );
            list[listcount].errdrop = intnode[i-1].stump->rerror
                - (list[listcount].stump->lerror + list[listcount].stump->rerror);
            list[listcount].leftflag = 0;
            list[listcount].parent = i-1;
            listcount++;
        }
        else
        {
            cvReleaseMat( &ridx );
        }
        
        if( listcount == 0 ) break;

        /* find the best node to be added to the tree */
        idx = 0;
        maxerrdrop = list[idx].errdrop;
        for( j = 1; j < listcount; j++ )
        {
            if( list[j].errdrop > maxerrdrop )
            {
                idx = j;
                maxerrdrop = list[j].errdrop;
            }
        }
        intnode[i] = list[idx];
        if( list[idx].leftflag )
        {
            cart->left[list[idx].parent] = i;
        }
        else
        {
            cart->right[list[idx].parent] = i;
        }
        if( idx != (listcount - 1) )
        {
            list[idx] = list[listcount - 1];
        }
        listcount--;
    }

    /* fill <cart> fields */
    j = 0;
    cart->count = 0;
    for( i = 0; i < count && (intnode[i].stump != NULL); i++ )
    {
        cart->count++;
        cart->compidx[i] = intnode[i].stump->compidx;
        cart->threshold[i] = intnode[i].stump->threshold;
        
        /* leaves */
        if( cart->left[i] <= 0 )
        {
            cart->left[i] = -j;
            cart->val[j] = intnode[i].stump->left;
            j++;
        }
        if( cart->right[i] <= 0 )
        {
            cart->right[i] = -j;
            cart->val[j] = intnode[i].stump->right;
            j++;
        }
    }
    
    /* CLEAN UP */
    for( i = 0; i < count && (intnode[i].stump != NULL); i++ )
    {
        intnode[i].stump->release( (CvClassifier**) &(intnode[i].stump) );
        if( i != 0 )
        {
            cvReleaseMat( &(intnode[i].sampleIdx) );
        }
    }
    for( i = 0; i < listcount; i++ )
    {
        list[i].stump->release( (CvClassifier**) &(list[i].stump) );
        cvReleaseMat( &(list[i].sampleIdx) );
    }
    
    cvFree( &intnode );

    return (CvClassifier*) cart;
}

/****************************************************************************************\
*                                        Boosting                                        *
\****************************************************************************************/

typedef struct CvBoostTrainer
{
    CvBoostType type;
    int count;             /* (idx) ? number_of_indices : number_of_samples */
    int* idx;
    float* F;
} CvBoostTrainer;

/*
 * cvBoostStartTraining, cvBoostNextWeakClassifier, cvBoostEndTraining
 *
 * These functions perform training of 2-class boosting classifier
 * using ANY appropriate weak classifier
 */

CV_BOOST_IMPL
CvBoostTrainer* icvBoostStartTraining( CvMat* trainClasses,
                                       CvMat* weakTrainVals,
                                       CvMat* /*weights*/,
                                       CvMat* sampleIdx,
                                       CvBoostType type )
{
    uchar* ydata;
    int ystep;
    int m;
    uchar* traindata;
    int trainstep;
    int trainnum;
    int i;
    int idx;

    size_t datasize;
    CvBoostTrainer* ptr;

    int idxnum;
    int idxstep;
    uchar* idxdata;

    assert( trainClasses != NULL );
    assert( CV_MAT_TYPE( trainClasses->type ) == CV_32FC1 );
    assert( weakTrainVals != NULL );
    assert( CV_MAT_TYPE( weakTrainVals->type ) == CV_32FC1 );

    CV_MAT2VEC( *trainClasses, ydata, ystep, m );
    CV_MAT2VEC( *weakTrainVals, traindata, trainstep, trainnum );

    assert( m == trainnum );

    idxnum = 0;
    idxstep = 0;
    idxdata = NULL;
    if( sampleIdx )
    {
        CV_MAT2VEC( *sampleIdx, idxdata, idxstep, idxnum );
    }
        
    datasize = sizeof( *ptr ) + sizeof( *ptr->idx ) * idxnum;
    ptr = (CvBoostTrainer*) cvAlloc( datasize );
    memset( ptr, 0, datasize );
    ptr->F = NULL;
    ptr->idx = NULL;

    ptr->count = m;
    ptr->type = type;
    
    if( idxnum > 0 )
    {
        CvScalar s;

        ptr->idx = (int*) (ptr + 1);
        ptr->count = idxnum;
        for( i = 0; i < ptr->count; i++ )
        {
            cvRawDataToScalar( idxdata + i*idxstep, CV_MAT_TYPE( sampleIdx->type ), &s );
            ptr->idx[i] = (int) s.val[0];
        }
    }
    for( i = 0; i < ptr->count; i++ )
    {
        idx = (ptr->idx) ? ptr->idx[i] : i;

        *((float*) (traindata + idx * trainstep)) = 
            2.0F * (*((float*) (ydata + idx * ystep))) - 1.0F;
    }

    return ptr;
}

/*
 *
 * Discrete AdaBoost functions
 *
 */
CV_BOOST_IMPL
float icvBoostNextWeakClassifierDAB( CvMat* weakEvalVals,
                                     CvMat* trainClasses,
                                     CvMat* /*weakTrainVals*/,
                                     CvMat* weights,
                                     CvBoostTrainer* trainer )
{
    uchar* evaldata;
    int evalstep;
    int m;
    uchar* ydata;
    int ystep;
    int ynum;
    uchar* wdata;
    int wstep;
    int wnum;

    float sumw;
    float err;
    int i;
    int idx;

    CV_Assert( weakEvalVals != NULL );
    CV_Assert( CV_MAT_TYPE( weakEvalVals->type ) == CV_32FC1 );
    CV_Assert( trainClasses != NULL );
    CV_Assert( CV_MAT_TYPE( trainClasses->type ) == CV_32FC1 );
    CV_Assert( weights != NULL );
    CV_Assert( CV_MAT_TYPE( weights ->type ) == CV_32FC1 );

    CV_MAT2VEC( *weakEvalVals, evaldata, evalstep, m );
    CV_MAT2VEC( *trainClasses, ydata, ystep, ynum );
    CV_MAT2VEC( *weights, wdata, wstep, wnum );

    assert( m == ynum );
    assert( m == wnum );

    sumw = 0.0F;
    err = 0.0F;
    for( i = 0; i < trainer->count; i++ )
    {
        idx = (trainer->idx) ? trainer->idx[i] : i;

        sumw += *((float*) (wdata + idx*wstep));
        err += (*((float*) (wdata + idx*wstep))) *
            ( (*((float*) (evaldata + idx*evalstep))) != 
                2.0F * (*((float*) (ydata + idx*ystep))) - 1.0F );
    }
    err /= sumw;
    err = -cvLogRatio( err );
    
    for( i = 0; i < trainer->count; i++ )
    {
        idx = (trainer->idx) ? trainer->idx[i] : i;

        *((float*) (wdata + idx*wstep)) *= expf( err * 
            ((*((float*) (evaldata + idx*evalstep))) != 
                2.0F * (*((float*) (ydata + idx*ystep))) - 1.0F) );
        sumw += *((float*) (wdata + idx*wstep));
    }
    for( i = 0; i < trainer->count; i++ )
    {
        idx = (trainer->idx) ? trainer->idx[i] : i;

        *((float*) (wdata + idx * wstep)) /= sumw;
    }
    
    return err;
}

/*
 *
 * Real AdaBoost functions
 *
 */
CV_BOOST_IMPL
float icvBoostNextWeakClassifierRAB( CvMat* weakEvalVals,
                                     CvMat* trainClasses,
                                     CvMat* /*weakTrainVals*/,
                                     CvMat* weights,
                                     CvBoostTrainer* trainer )
{
    uchar* evaldata;
    int evalstep;
    int m;
    uchar* ydata;
    int ystep;
    int ynum;
    uchar* wdata;
    int wstep;
    int wnum;

    float sumw;
    int i, idx;

    CV_Assert( weakEvalVals != NULL );
    CV_Assert( CV_MAT_TYPE( weakEvalVals->type ) == CV_32FC1 );
    CV_Assert( trainClasses != NULL );
    CV_Assert( CV_MAT_TYPE( trainClasses->type ) == CV_32FC1 );
    CV_Assert( weights != NULL );
    CV_Assert( CV_MAT_TYPE( weights ->type ) == CV_32FC1 );

    CV_MAT2VEC( *weakEvalVals, evaldata, evalstep, m );
    CV_MAT2VEC( *trainClasses, ydata, ystep, ynum );
    CV_MAT2VEC( *weights, wdata, wstep, wnum );

    CV_Assert( m == ynum );
    CV_Assert( m == wnum );


    sumw = 0.0F;
    for( i = 0; i < trainer->count; i++ )
    {
        idx = (trainer->idx) ? trainer->idx[i] : i;

        *((float*) (wdata + idx*wstep)) *= expf( (-(*((float*) (ydata + idx*ystep))) + 0.5F)
            * cvLogRatio( *((float*) (evaldata + idx*evalstep)) ) );
        sumw += *((float*) (wdata + idx*wstep));
    }
    for( i = 0; i < trainer->count; i++ )
    {
        idx = (trainer->idx) ? trainer->idx[i] : i;

        *((float*) (wdata + idx*wstep)) /= sumw;
    }
    
    return 1.0F;
}

/*
 *
 * LogitBoost functions
 *
 */
#define CV_LB_PROB_THRESH      0.01F
#define CV_LB_WEIGHT_THRESHOLD 0.0001F

CV_BOOST_IMPL
void icvResponsesAndWeightsLB( int num, uchar* wdata, int wstep,
                               uchar* ydata, int ystep,
                               uchar* fdata, int fstep,
                               uchar* traindata, int trainstep,
                               int* indices )
{
    int i, idx;
    float p;

    for( i = 0; i < num; i++ )
    {
        idx = (indices) ? indices[i] : i;

        p = 1.0F / (1.0F + expf( -(*((float*) (fdata + idx*fstep)))) );
        *((float*) (wdata + idx*wstep)) = MAX( p * (1.0F - p), CV_LB_WEIGHT_THRESHOLD );
        if( *((float*) (ydata + idx*ystep)) == 1.0F )
        {
            *((float*) (traindata + idx*trainstep)) = 
                1.0F / (MAX( p, CV_LB_PROB_THRESH ));
        }
        else
        {
            *((float*) (traindata + idx*trainstep)) = 
                -1.0F / (MAX( 1.0F - p, CV_LB_PROB_THRESH ));
        }
    }
}

CV_BOOST_IMPL
CvBoostTrainer* icvBoostStartTrainingLB( CvMat* trainClasses,
                                         CvMat* weakTrainVals,
                                         CvMat* weights,
                                         CvMat* sampleIdx,
                                         CvBoostType type )
{
    size_t datasize;
    CvBoostTrainer* ptr;

    uchar* ydata;
    int ystep;
    int m;
    uchar* traindata;
    int trainstep;
    int trainnum;
    uchar* wdata;
    int wstep;
    int wnum;
    int i;

    int idxnum;
    int idxstep;
    uchar* idxdata;

    assert( trainClasses != NULL );
    assert( CV_MAT_TYPE( trainClasses->type ) == CV_32FC1 );
    assert( weakTrainVals != NULL );
    assert( CV_MAT_TYPE( weakTrainVals->type ) == CV_32FC1 );
    assert( weights != NULL );
    assert( CV_MAT_TYPE( weights->type ) == CV_32FC1 );

    CV_MAT2VEC( *trainClasses, ydata, ystep, m );
    CV_MAT2VEC( *weakTrainVals, traindata, trainstep, trainnum );
    CV_MAT2VEC( *weights, wdata, wstep, wnum );

    assert( m == trainnum );
    assert( m == wnum );


    idxnum = 0;
    idxstep = 0;
    idxdata = NULL;
    if( sampleIdx )
    {
        CV_MAT2VEC( *sampleIdx, idxdata, idxstep, idxnum );
    }
        
    datasize = sizeof( *ptr ) + sizeof( *ptr->F ) * m + sizeof( *ptr->idx ) * idxnum;
    ptr = (CvBoostTrainer*) cvAlloc( datasize );
    memset( ptr, 0, datasize );
    ptr->F = (float*) (ptr + 1);
    ptr->idx = NULL;

    ptr->count = m;
    ptr->type = type;
    
    if( idxnum > 0 )
    {
        CvScalar s;

        ptr->idx = (int*) (ptr->F + m);
        ptr->count = idxnum;
        for( i = 0; i < ptr->count; i++ )
        {
            cvRawDataToScalar( idxdata + i*idxstep, CV_MAT_TYPE( sampleIdx->type ), &s );
            ptr->idx[i] = (int) s.val[0];
        }
    }

    for( i = 0; i < m; i++ )
    {
        ptr->F[i] = 0.0F;
    }

    icvResponsesAndWeightsLB( ptr->count, wdata, wstep, ydata, ystep,
                              (uchar*) ptr->F, sizeof( *ptr->F ),
                              traindata, trainstep, ptr->idx );

    return ptr;
}

CV_BOOST_IMPL
float icvBoostNextWeakClassifierLB( CvMat* weakEvalVals,
                                    CvMat* trainClasses,
                                    CvMat* weakTrainVals,
                                    CvMat* weights,
                                    CvBoostTrainer* trainer )
{
    uchar* evaldata;
    int evalstep;
    int m;
    uchar* ydata;
    int ystep;
    int ynum;
    uchar* traindata;
    int trainstep;
    int trainnum;
    uchar* wdata;
    int wstep;
    int wnum;
    int i, idx;

    assert( weakEvalVals != NULL );
    assert( CV_MAT_TYPE( weakEvalVals->type ) == CV_32FC1 );
    assert( trainClasses != NULL );
    assert( CV_MAT_TYPE( trainClasses->type ) == CV_32FC1 );
    assert( weakTrainVals != NULL );
    assert( CV_MAT_TYPE( weakTrainVals->type ) == CV_32FC1 );
    assert( weights != NULL );
    assert( CV_MAT_TYPE( weights ->type ) == CV_32FC1 );

    CV_MAT2VEC( *weakEvalVals, evaldata, evalstep, m );
    CV_MAT2VEC( *trainClasses, ydata, ystep, ynum );
    CV_MAT2VEC( *weakTrainVals, traindata, trainstep, trainnum );
    CV_MAT2VEC( *weights, wdata, wstep, wnum );

    assert( m == ynum );
    assert( m == wnum );
    assert( m == trainnum );
    //assert( m == trainer->count );

    for( i = 0; i < trainer->count; i++ )
    {
        idx = (trainer->idx) ? trainer->idx[i] : i;

        trainer->F[idx] += *((float*) (evaldata + idx * evalstep));
    }
    
    icvResponsesAndWeightsLB( trainer->count, wdata, wstep, ydata, ystep,
                              (uchar*) trainer->F, sizeof( *trainer->F ),
                              traindata, trainstep, trainer->idx );

    return 1.0F;
}

/*
 *
 * Gentle AdaBoost
 *
 */
CV_BOOST_IMPL
float icvBoostNextWeakClassifierGAB( CvMat* weakEvalVals,
                                     CvMat* trainClasses,
                                     CvMat* /*weakTrainVals*/,
                                     CvMat* weights,
                                     CvBoostTrainer* trainer )
{
    uchar* evaldata;
    int evalstep;
    int m;
    uchar* ydata;
    int ystep;
    int ynum;
    uchar* wdata;
    int wstep;
    int wnum;

    int i, idx;
    float sumw;

    CV_Assert( weakEvalVals != NULL );
    CV_Assert( CV_MAT_TYPE( weakEvalVals->type ) == CV_32FC1 );
    CV_Assert( trainClasses != NULL );
    CV_Assert( CV_MAT_TYPE( trainClasses->type ) == CV_32FC1 );
    CV_Assert( weights != NULL );
    CV_Assert( CV_MAT_TYPE( weights->type ) == CV_32FC1 );

    CV_MAT2VEC( *weakEvalVals, evaldata, evalstep, m );
    CV_MAT2VEC( *trainClasses, ydata, ystep, ynum );
    CV_MAT2VEC( *weights, wdata, wstep, wnum );

    assert( m == ynum );
    assert( m == wnum );

    sumw = 0.0F;
    for( i = 0; i < trainer->count; i++ )
    {
        idx = (trainer->idx) ? trainer->idx[i] : i;

        *((float*) (wdata + idx*wstep)) *= 
            expf( -(*((float*) (evaldata + idx*evalstep)))
                  * ( 2.0F * (*((float*) (ydata + idx*ystep))) - 1.0F ) );
        sumw += *((float*) (wdata + idx*wstep));
    }
    
    for( i = 0; i < trainer->count; i++ )
    {
        idx = (trainer->idx) ? trainer->idx[i] : i;

        *((float*) (wdata + idx*wstep)) /= sumw;
    }

    return 1.0F;
}

typedef CvBoostTrainer* (*CvBoostStartTraining)( CvMat* trainClasses,
                                                 CvMat* weakTrainVals,
                                                 CvMat* weights,
                                                 CvMat* sampleIdx,
                                                 CvBoostType type );

typedef float (*CvBoostNextWeakClassifier)( CvMat* weakEvalVals,
                                            CvMat* trainClasses,
                                            CvMat* weakTrainVals,
                                            CvMat* weights,
                                            CvBoostTrainer* data );

CvBoostStartTraining startTraining[4] = {
        icvBoostStartTraining,
        icvBoostStartTraining,
        icvBoostStartTrainingLB,
        icvBoostStartTraining
    };

CvBoostNextWeakClassifier nextWeakClassifier[4] = {
        icvBoostNextWeakClassifierDAB,
        icvBoostNextWeakClassifierRAB,
        icvBoostNextWeakClassifierLB,
        icvBoostNextWeakClassifierGAB
    };

/*
 *
 * Dispatchers
 *
 */
CV_BOOST_IMPL
CvBoostTrainer* cvBoostStartTraining( CvMat* trainClasses,
                                      CvMat* weakTrainVals,
                                      CvMat* weights,
                                      CvMat* sampleIdx,
                                      CvBoostType type )
{
    return startTraining[type]( trainClasses, weakTrainVals, weights, sampleIdx, type );
}

CV_BOOST_IMPL
void cvBoostEndTraining( CvBoostTrainer** trainer )
{
    cvFree( trainer );
    *trainer = NULL;
}

CV_BOOST_IMPL
float cvBoostNextWeakClassifier( CvMat* weakEvalVals,
                                 CvMat* trainClasses,
                                 CvMat* weakTrainVals,
                                 CvMat* weights,
                                 CvBoostTrainer* trainer )
{
    return nextWeakClassifier[trainer->type]( weakEvalVals, trainClasses,
        weakTrainVals, weights, trainer    );
}

/****************************************************************************************\
*                                    Boosted tree models                                 *
\****************************************************************************************/

typedef struct CvBtTrainer
{
    /* {{ external */    
    CvMat* trainData;
    int flags;
    
    CvMat* trainClasses;
    int m;
    uchar* ydata;
    int ystep;

    CvMat* sampleIdx;
    int numsamples;
    
    float param[2];
    CvBoostType type;
    int numclasses;
    /* }} external */

    CvMTStumpTrainParams stumpParams;
    CvCARTTrainParams  cartParams;

    float* f;          /* F_(m-1) */
    CvMat* y;          /* yhat    */
    CvMat* weights;
    CvBoostTrainer* boosttrainer;
} CvBtTrainer;

/*
 * cvBtStart, cvBtNext, cvBtEnd
 *
 * These functions perform iterative training of
 * 2-class (CV_DABCLASS - CV_GABCLASS, CV_L2CLASS), K-class (CV_LKCLASS) classifier
 * or fit regression model (CV_LSREG, CV_LADREG, CV_MREG)
 * using decision tree as a weak classifier.
 */

typedef void (*CvZeroApproxFunc)( float* approx, CvBtTrainer* trainer );

/* Mean zero approximation */
void icvZeroApproxMean( float* approx, CvBtTrainer* trainer )
{
    int i;
    int idx;

    approx[0] = 0.0F;
    for( i = 0; i < trainer->numsamples; i++ )
    {
        idx = icvGetIdxAt( trainer->sampleIdx, i );
        approx[0] += *((float*) (trainer->ydata + idx * trainer->ystep));
    }
    approx[0] /= (float) trainer->numsamples;
}

/*
 * Median zero approximation
 */
void icvZeroApproxMed( float* approx, CvBtTrainer* trainer )
{
    int i;
    int idx;

    for( i = 0; i < trainer->numsamples; i++ )
    {
        idx = icvGetIdxAt( trainer->sampleIdx, i );
        trainer->f[i] = *((float*) (trainer->ydata + idx * trainer->ystep));
    }
    
    icvSort_32f( trainer->f, trainer->numsamples, 0 );
    approx[0] = trainer->f[trainer->numsamples / 2];
}

/*
 * 0.5 * log( mean(y) / (1 - mean(y)) ) where y in {0, 1}
 */
void icvZeroApproxLog( float* approx, CvBtTrainer* trainer )
{
    float y_mean;

    icvZeroApproxMean( &y_mean, trainer );
    approx[0] = 0.5F * cvLogRatio( y_mean );
}

/*
 * 0 zero approximation
 */
void icvZeroApprox0( float* approx, CvBtTrainer* trainer )
{
    int i;

    for( i = 0; i < trainer->numclasses; i++ )
    {
        approx[i] = 0.0F;
    }
}

static CvZeroApproxFunc icvZeroApproxFunc[] =
{
    icvZeroApprox0,    /* CV_DABCLASS */
    icvZeroApprox0,    /* CV_RABCLASS */
    icvZeroApprox0,    /* CV_LBCLASS  */
    icvZeroApprox0,    /* CV_GABCLASS */
    icvZeroApproxLog,  /* CV_L2CLASS  */
    icvZeroApprox0,    /* CV_LKCLASS  */
    icvZeroApproxMean, /* CV_LSREG    */
    icvZeroApproxMed,  /* CV_LADREG   */
    icvZeroApproxMed,  /* CV_MREG     */
};

CV_BOOST_IMPL
void cvBtNext( CvCARTClassifier** trees, CvBtTrainer* trainer );

CV_BOOST_IMPL
CvBtTrainer* cvBtStart( CvCARTClassifier** trees,
                        CvMat* trainData,
                        int flags,
                        CvMat* trainClasses,
                        CvMat* sampleIdx,
                        int numsplits,
                        CvBoostType type,
                        int numclasses,
                        float* param )
{
    CvBtTrainer* ptr = 0;

    CV_FUNCNAME( "cvBtStart" );

    __BEGIN__;

    size_t data_size;
    float* zero_approx;
    int m;
    int i, j;
    
    if( trees == NULL )
    {
        CV_ERROR( CV_StsNullPtr, "Invalid trees parameter" );
    }
    
    if( type < CV_DABCLASS || type > CV_MREG ) 
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Unsupported type parameter" );
    }
    if( type == CV_LKCLASS )
    {
        CV_ASSERT( numclasses >= 2 );
    }
    else
    {
        numclasses = 1;
    }

    m = MAX( trainClasses->rows, trainClasses->cols );
    ptr = NULL;
    data_size = sizeof( *ptr );
    if( type > CV_GABCLASS )
    {
        data_size += m * numclasses * sizeof( *(ptr->f) );
    }
    CV_CALL( ptr = (CvBtTrainer*) cvAlloc( data_size ) );
    memset( ptr, 0, data_size );
    ptr->f = (float*) (ptr + 1);

    ptr->trainData = trainData;
    ptr->flags = flags;
    ptr->trainClasses = trainClasses;
    CV_MAT2VEC( *trainClasses, ptr->ydata, ptr->ystep, ptr->m );
    
    memset( &(ptr->cartParams), 0, sizeof( ptr->cartParams ) );
    memset( &(ptr->stumpParams), 0, sizeof( ptr->stumpParams ) );

    switch( type )
    {
        case CV_DABCLASS:
            ptr->stumpParams.error = CV_MISCLASSIFICATION;
            ptr->stumpParams.type  = CV_CLASSIFICATION_CLASS;
            break;
        case CV_RABCLASS:
            ptr->stumpParams.error = CV_GINI;
            ptr->stumpParams.type  = CV_CLASSIFICATION;
            break;
        default:
            ptr->stumpParams.error = CV_SQUARE;
            ptr->stumpParams.type  = CV_REGRESSION;
    }
    ptr->cartParams.count = numsplits;
    ptr->cartParams.stumpTrainParams = (CvClassifierTrainParams*) &(ptr->stumpParams);
    ptr->cartParams.stumpConstructor = cvCreateMTStumpClassifier;

    ptr->param[0] = param[0];
    ptr->param[1] = param[1];
    ptr->type = type;
    ptr->numclasses = numclasses;

    CV_CALL( ptr->y = cvCreateMat( 1, m, CV_32FC1 ) );
    ptr->sampleIdx = sampleIdx;
    ptr->numsamples = ( sampleIdx == NULL ) ? ptr->m
                             : MAX( sampleIdx->rows, sampleIdx->cols );
    
    ptr->weights = cvCreateMat( 1, m, CV_32FC1 );
    cvSet( ptr->weights, cvScalar( 1.0 ) );    
    
    if( type <= CV_GABCLASS )
    {
        ptr->boosttrainer = cvBoostStartTraining( ptr->trainClasses, ptr->y,
            ptr->weights, NULL, type );

        CV_CALL( cvBtNext( trees, ptr ) );
    }
    else
    {
        data_size = sizeof( *zero_approx ) * numclasses;
        CV_CALL( zero_approx = (float*) cvAlloc( data_size ) );
        icvZeroApproxFunc[type]( zero_approx, ptr );
        for( i = 0; i < m; i++ )
        {
            for( j = 0; j < numclasses; j++ )
            {
                ptr->f[i * numclasses + j] = zero_approx[j];
            }
        }

        CV_CALL( cvBtNext( trees, ptr ) );

        for( i = 0; i < numclasses; i++ )
        {
            for( j = 0; j <= trees[i]->count; j++ )
            {
                trees[i]->val[j] += zero_approx[i];
            }
        }    
        CV_CALL( cvFree( &zero_approx ) );
    }

    __END__;

    return ptr;
}

void icvBtNext_LSREG( CvCARTClassifier** trees, CvBtTrainer* trainer )
{
    int i;

    /* yhat_i = y_i - F_(m-1)(x_i) */
    for( i = 0; i < trainer->m; i++ )
    {
        trainer->y->data.fl[i] = 
            *((float*) (trainer->ydata + i * trainer->ystep)) - trainer->f[i];
    }

    trees[0] = (CvCARTClassifier*) cvCreateCARTClassifier( trainer->trainData,
        trainer->flags,
        trainer->y, NULL, NULL, NULL, trainer->sampleIdx, trainer->weights,
        (CvClassifierTrainParams*) &trainer->cartParams );
}


void icvBtNext_LADREG( CvCARTClassifier** trees, CvBtTrainer* trainer )
{
    CvCARTClassifier* ptr;
    int i, j;
    CvMat sample;
    int sample_step;
    uchar* sample_data;
    int index;
    
    int data_size;
    int* idx;
    float* resp;
    int respnum;
    float val;

    data_size = trainer->m * sizeof( *idx );
    idx = (int*) cvAlloc( data_size );
    data_size = trainer->m * sizeof( *resp );
    resp = (float*) cvAlloc( data_size );

    /* yhat_i = sign(y_i - F_(m-1)(x_i)) */
    for( i = 0; i < trainer->numsamples; i++ )
    {
        index = icvGetIdxAt( trainer->sampleIdx, i );
        trainer->y->data.fl[index] = (float)
             CV_SIGN( *((float*) (trainer->ydata + index * trainer->ystep))
                     - trainer->f[index] );
    }

    ptr = (CvCARTClassifier*) cvCreateCARTClassifier( trainer->trainData, trainer->flags,
        trainer->y, NULL, NULL, NULL, trainer->sampleIdx, trainer->weights,
        (CvClassifierTrainParams*) &trainer->cartParams );

    CV_GET_SAMPLE( *trainer->trainData, trainer->flags, 0, sample );
    CV_GET_SAMPLE_STEP( *trainer->trainData, trainer->flags, sample_step );
    sample_data = sample.data.ptr;
    for( i = 0; i < trainer->numsamples; i++ )
    {
        index = icvGetIdxAt( trainer->sampleIdx, i );
        sample.data.ptr = sample_data + index * sample_step;
        idx[index] = (int) cvEvalCARTClassifierIdx( (CvClassifier*) ptr, &sample );
    }
    for( j = 0; j <= ptr->count; j++ )
    {
        respnum = 0;
        for( i = 0; i < trainer->numsamples; i++ )
        {
            index = icvGetIdxAt( trainer->sampleIdx, i );
            if( idx[index] == j )
            {
                resp[respnum++] = *((float*) (trainer->ydata + index * trainer->ystep))
                                  - trainer->f[index];
            }
        }
        if( respnum > 0 )
        {
            icvSort_32f( resp, respnum, 0 );
            val = resp[respnum / 2];
        }
        else
        {
            val = 0.0F;
        }
        ptr->val[j] = val;
    }

    cvFree( &idx );
    cvFree( &resp );
    
    trees[0] = ptr;
}


void icvBtNext_MREG( CvCARTClassifier** trees, CvBtTrainer* trainer )
{
    CvCARTClassifier* ptr;
    int i, j;
    CvMat sample;
    int sample_step;
    uchar* sample_data;
    
    int data_size;
    int* idx;
    float* resid;
    float* resp;
    int respnum;
    float rhat;
    float val;
    float delta;
    int index;

    data_size = trainer->m * sizeof( *idx );
    idx = (int*) cvAlloc( data_size );
    data_size = trainer->m * sizeof( *resp );
    resp = (float*) cvAlloc( data_size );
    data_size = trainer->m * sizeof( *resid );
    resid = (float*) cvAlloc( data_size );

    /* resid_i = (y_i - F_(m-1)(x_i)) */
    for( i = 0; i < trainer->numsamples; i++ )
    {
        index = icvGetIdxAt( trainer->sampleIdx, i );
        resid[index] = *((float*) (trainer->ydata + index * trainer->ystep))
                       - trainer->f[index];
        /* for delta */
        resp[i] = (float) fabs( resid[index] );
    }
    
    /* delta = quantile_alpha{abs(resid_i)} */
    icvSort_32f( resp, trainer->numsamples, 0 );
    delta = resp[(int)(trainer->param[1] * (trainer->numsamples - 1))];

    /* yhat_i */
    for( i = 0; i < trainer->numsamples; i++ )
    {
        index = icvGetIdxAt( trainer->sampleIdx, i );
        trainer->y->data.fl[index] = MIN( delta, ((float) fabs( resid[index] )) ) *
                                 CV_SIGN( resid[index] );
    }
    
    ptr = (CvCARTClassifier*) cvCreateCARTClassifier( trainer->trainData, trainer->flags,
        trainer->y, NULL, NULL, NULL, trainer->sampleIdx, trainer->weights,
        (CvClassifierTrainParams*) &trainer->cartParams );

    CV_GET_SAMPLE( *trainer->trainData, trainer->flags, 0, sample );
    CV_GET_SAMPLE_STEP( *trainer->trainData, trainer->flags, sample_step );
    sample_data = sample.data.ptr;
    for( i = 0; i < trainer->numsamples; i++ )
    {
        index = icvGetIdxAt( trainer->sampleIdx, i );
        sample.data.ptr = sample_data + index * sample_step;
        idx[index] = (int) cvEvalCARTClassifierIdx( (CvClassifier*) ptr, &sample );
    }
    for( j = 0; j <= ptr->count; j++ )
    {
        respnum = 0;

        for( i = 0; i < trainer->numsamples; i++ )
        {
            index = icvGetIdxAt( trainer->sampleIdx, i );
            if( idx[index] == j )
            {
                resp[respnum++] = *((float*) (trainer->ydata + index * trainer->ystep))
                                  - trainer->f[index];
            }
        }
        if( respnum > 0 )
        {
            /* rhat = median(y_i - F_(m-1)(x_i)) */
            icvSort_32f( resp, respnum, 0 );
            rhat = resp[respnum / 2];
            
            /* val = sum{sign(r_i - rhat_i) * min(delta, abs(r_i - rhat_i)}
             * r_i = y_i - F_(m-1)(x_i)
             */
            val = 0.0F;
            for( i = 0; i < respnum; i++ )
            {
                val += CV_SIGN( resp[i] - rhat )
                       * MIN( delta, (float) fabs( resp[i] - rhat ) );
            }

            val = rhat + val / (float) respnum;
        }
        else
        {
            val = 0.0F;
        }

        ptr->val[j] = val;

    }

    cvFree( &resid );
    cvFree( &resp );
    cvFree( &idx );
    
    trees[0] = ptr;
}

//#define CV_VAL_MAX 1e304

//#define CV_LOG_VAL_MAX 700.0

#define CV_VAL_MAX 1e+8

#define CV_LOG_VAL_MAX 18.0

void icvBtNext_L2CLASS( CvCARTClassifier** trees, CvBtTrainer* trainer )
{
    CvCARTClassifier* ptr;
    int i, j;
    CvMat sample;
    int sample_step;
    uchar* sample_data;
    
    int data_size;
    int* idx;
    int respnum;
    float val;
    double val_f;

    float sum_weights;
    float* weights;
    float* sorted_weights;
    CvMat* trimmed_idx;
    CvMat* sample_idx;
    int index;
    int trimmed_num;

    data_size = trainer->m * sizeof( *idx );
    idx = (int*) cvAlloc( data_size );

    data_size = trainer->m * sizeof( *weights );
    weights = (float*) cvAlloc( data_size );
    data_size = trainer->m * sizeof( *sorted_weights );
    sorted_weights = (float*) cvAlloc( data_size );
    
    /* yhat_i = (4 * y_i - 2) / ( 1 + exp( (4 * y_i - 2) * F_(m-1)(x_i) ) ).
     *   y_i in {0, 1}
     */
    sum_weights = 0.0F;
    for( i = 0; i < trainer->numsamples; i++ )
    {
        index = icvGetIdxAt( trainer->sampleIdx, i );
        val = 4.0F * (*((float*) (trainer->ydata + index * trainer->ystep))) - 2.0F;
        val_f = val * trainer->f[index];
        val_f = ( val_f < CV_LOG_VAL_MAX ) ? exp( val_f ) : CV_LOG_VAL_MAX;
        val = (float) ( (double) val / ( 1.0 + val_f ) );
        trainer->y->data.fl[index] = val;
        val = (float) fabs( val );
        weights[index] = val * (2.0F - val);
        sorted_weights[i] = weights[index];
        sum_weights += sorted_weights[i];
    }
    
    trimmed_idx = NULL;
    sample_idx = trainer->sampleIdx;
    trimmed_num = trainer->numsamples;
    if( trainer->param[1] < 1.0F )
    {
        /* perform weight trimming */
        
        float threshold;
        int count;
        
        icvSort_32f( sorted_weights, trainer->numsamples, 0 );

        sum_weights *= (1.0F - trainer->param[1]);
        
        i = -1;
        do { sum_weights -= sorted_weights[++i]; }
        while( sum_weights > 0.0F && i < (trainer->numsamples - 1) );
        
        threshold = sorted_weights[i];

        while( i > 0 && sorted_weights[i-1] == threshold ) i--;

        if( i > 0 )
        {
            trimmed_num = trainer->numsamples - i;            
            trimmed_idx = cvCreateMat( 1, trimmed_num, CV_32FC1 );
            count = 0;
            for( i = 0; i < trainer->numsamples; i++ )
            {
                index = icvGetIdxAt( trainer->sampleIdx, i );
                if( weights[index] >= threshold )
                {
                    CV_MAT_ELEM( *trimmed_idx, float, 0, count ) = (float) index;
                    count++;
                }
            }
            
            assert( count == trimmed_num );

            sample_idx = trimmed_idx;

            printf( "Used samples %%: %g\n", 
                (float) trimmed_num / (float) trainer->numsamples * 100.0F );
        }
    }

    ptr = (CvCARTClassifier*) cvCreateCARTClassifier( trainer->trainData, trainer->flags,
        trainer->y, NULL, NULL, NULL, sample_idx, trainer->weights,
        (CvClassifierTrainParams*) &trainer->cartParams );

    CV_GET_SAMPLE( *trainer->trainData, trainer->flags, 0, sample );
    CV_GET_SAMPLE_STEP( *trainer->trainData, trainer->flags, sample_step );
    sample_data = sample.data.ptr;
    for( i = 0; i < trimmed_num; i++ )
    {
        index = icvGetIdxAt( sample_idx, i );
        sample.data.ptr = sample_data + index * sample_step;
        idx[index] = (int) cvEvalCARTClassifierIdx( (CvClassifier*) ptr, &sample );
    }
    for( j = 0; j <= ptr->count; j++ )
    {
        respnum = 0;
        val = 0.0F;
        sum_weights = 0.0F;
        for( i = 0; i < trimmed_num; i++ )
        {
            index = icvGetIdxAt( sample_idx, i );
            if( idx[index] == j )
            {
                val += trainer->y->data.fl[index];
                sum_weights += weights[index];
                respnum++;
            }
        }
        if( sum_weights > 0.0F )
        {
            val /= sum_weights;
        }
        else
        {
            val = 0.0F;
        }
        ptr->val[j] = val;
    }
    
    if( trimmed_idx != NULL ) cvReleaseMat( &trimmed_idx );
    cvFree( &sorted_weights );
    cvFree( &weights );
    cvFree( &idx );
    
    trees[0] = ptr;
}

void icvBtNext_LKCLASS( CvCARTClassifier** trees, CvBtTrainer* trainer )
{
    int i, j, k, kk, num;
    CvMat sample;
    int sample_step;
    uchar* sample_data;
    
    int data_size;
    int* idx;
    int respnum;
    float val;

    float sum_weights;
    float* weights;
    float* sorted_weights;
    CvMat* trimmed_idx;
    CvMat* sample_idx;
    int index;
    int trimmed_num;
    double sum_exp_f;
    double exp_f;
    double f_k;

    data_size = trainer->m * sizeof( *idx );
    idx = (int*) cvAlloc( data_size );
    data_size = trainer->m * sizeof( *weights );
    weights = (float*) cvAlloc( data_size );
    data_size = trainer->m * sizeof( *sorted_weights );
    sorted_weights = (float*) cvAlloc( data_size );
    trimmed_idx = cvCreateMat( 1, trainer->numsamples, CV_32FC1 );

    for( k = 0; k < trainer->numclasses; k++ )
    {
        /* yhat_i = y_i - p_k(x_i), y_i in {0, 1}      */
        /* p_k(x_i) = exp(f_k(x_i)) / (sum_exp_f(x_i)) */
        sum_weights = 0.0F;
        for( i = 0; i < trainer->numsamples; i++ )
        {
            index = icvGetIdxAt( trainer->sampleIdx, i );
            /* p_k(x_i) = 1 / (1 + sum(exp(f_kk(x_i) - f_k(x_i)))), kk != k */
            num = index * trainer->numclasses;
            f_k = (double) trainer->f[num + k];
            sum_exp_f = 1.0;
            for( kk = 0; kk < trainer->numclasses; kk++ )
            {
                if( kk == k ) continue;
                exp_f = (double) trainer->f[num + kk] - f_k;
                exp_f = (exp_f < CV_LOG_VAL_MAX) ? exp( exp_f ) : CV_VAL_MAX;
                if( exp_f == CV_VAL_MAX || exp_f >= (CV_VAL_MAX - sum_exp_f) )
                {
                    sum_exp_f = CV_VAL_MAX;
                    break;
                }
                sum_exp_f += exp_f;
            }

            val = (float) ( (*((float*) (trainer->ydata + index * trainer->ystep))) 
                            == (float) k );
            val -= (float) ( (sum_exp_f == CV_VAL_MAX) ? 0.0 : ( 1.0 / sum_exp_f ) );

            assert( val >= -1.0F );
            assert( val <= 1.0F );

            trainer->y->data.fl[index] = val;
            val = (float) fabs( val );
            weights[index] = val * (1.0F - val);
            sorted_weights[i] = weights[index];
            sum_weights += sorted_weights[i];
        }

        sample_idx = trainer->sampleIdx;
        trimmed_num = trainer->numsamples;
        if( trainer->param[1] < 1.0F )
        {
            /* perform weight trimming */
        
            float threshold;
            int count;
        
            icvSort_32f( sorted_weights, trainer->numsamples, 0 );

            sum_weights *= (1.0F - trainer->param[1]);
        
            i = -1;
            do { sum_weights -= sorted_weights[++i]; }
            while( sum_weights > 0.0F && i < (trainer->numsamples - 1) );
        
            threshold = sorted_weights[i];

            while( i > 0 && sorted_weights[i-1] == threshold ) i--;

            if( i > 0 )
            {
                trimmed_num = trainer->numsamples - i;            
                trimmed_idx->cols = trimmed_num;
                count = 0;
                for( i = 0; i < trainer->numsamples; i++ )
                {
                    index = icvGetIdxAt( trainer->sampleIdx, i );
                    if( weights[index] >= threshold )
                    {
                        CV_MAT_ELEM( *trimmed_idx, float, 0, count ) = (float) index;
                        count++;
                    }
                }
            
                assert( count == trimmed_num );

                sample_idx = trimmed_idx;

                printf( "k: %d Used samples %%: %g\n", k, 
                    (float) trimmed_num / (float) trainer->numsamples * 100.0F );
            }
        } /* weight trimming */

        trees[k] = (CvCARTClassifier*) cvCreateCARTClassifier( trainer->trainData,
            trainer->flags, trainer->y, NULL, NULL, NULL, sample_idx, trainer->weights,
            (CvClassifierTrainParams*) &trainer->cartParams );

        CV_GET_SAMPLE( *trainer->trainData, trainer->flags, 0, sample );
        CV_GET_SAMPLE_STEP( *trainer->trainData, trainer->flags, sample_step );
        sample_data = sample.data.ptr;
        for( i = 0; i < trimmed_num; i++ )
        {
            index = icvGetIdxAt( sample_idx, i );
            sample.data.ptr = sample_data + index * sample_step;
            idx[index] = (int) cvEvalCARTClassifierIdx( (CvClassifier*) trees[k],
                                                        &sample );
        }
        for( j = 0; j <= trees[k]->count; j++ )
        {
            respnum = 0;
            val = 0.0F;
            sum_weights = 0.0F;
            for( i = 0; i < trimmed_num; i++ )
            {
                index = icvGetIdxAt( sample_idx, i );
                if( idx[index] == j )
                {
                    val += trainer->y->data.fl[index];
                    sum_weights += weights[index];
                    respnum++;
                }
            }
            if( sum_weights > 0.0F )
            {
                val = ((float) (trainer->numclasses - 1)) * val /
                      ((float) (trainer->numclasses)) / sum_weights;
            }
            else
            {
                val = 0.0F;
            }
            trees[k]->val[j] = val;
        }
    } /* for each class */
    
    cvReleaseMat( &trimmed_idx );
    cvFree( &sorted_weights );
    cvFree( &weights );
    cvFree( &idx );
}


void icvBtNext_XXBCLASS( CvCARTClassifier** trees, CvBtTrainer* trainer )
{
    float alpha;
    int i;
    CvMat* weak_eval_vals;
    CvMat* sample_idx;
    int num_samples;
    CvMat sample;
    uchar* sample_data;
    int sample_step;

    weak_eval_vals = cvCreateMat( 1, trainer->m, CV_32FC1 );

    sample_idx = cvTrimWeights( trainer->weights, trainer->sampleIdx,
                                trainer->param[1] );
    num_samples = ( sample_idx == NULL )
        ? trainer->m : MAX( sample_idx->rows, sample_idx->cols );

    printf( "Used samples %%: %g\n", 
        (float) num_samples / (float) trainer->numsamples * 100.0F );

    trees[0] = (CvCARTClassifier*) cvCreateCARTClassifier( trainer->trainData,
        trainer->flags, trainer->y, NULL, NULL, NULL,
        sample_idx, trainer->weights,
        (CvClassifierTrainParams*) &trainer->cartParams );
    
    /* evaluate samples */
    CV_GET_SAMPLE( *trainer->trainData, trainer->flags, 0, sample );
    CV_GET_SAMPLE_STEP( *trainer->trainData, trainer->flags, sample_step );
    sample_data = sample.data.ptr;
    
    for( i = 0; i < trainer->m; i++ )
    {
        sample.data.ptr = sample_data + i * sample_step;
        weak_eval_vals->data.fl[i] = trees[0]->eval( (CvClassifier*) trees[0], &sample );
    }

    alpha = cvBoostNextWeakClassifier( weak_eval_vals, trainer->trainClasses,
        trainer->y, trainer->weights, trainer->boosttrainer );
    
    /* multiply tree by alpha */
    for( i = 0; i <= trees[0]->count; i++ )
    {
        trees[0]->val[i] *= alpha;
    }
    if( trainer->type == CV_RABCLASS )
    {
        for( i = 0; i <= trees[0]->count; i++ )
        {
            trees[0]->val[i] = cvLogRatio( trees[0]->val[i] );
        }
    }
    
    if( sample_idx != NULL && sample_idx != trainer->sampleIdx )
    {
        cvReleaseMat( &sample_idx );
    }
    cvReleaseMat( &weak_eval_vals );
}

typedef void (*CvBtNextFunc)( CvCARTClassifier** trees, CvBtTrainer* trainer );

static CvBtNextFunc icvBtNextFunc[] =
{
    icvBtNext_XXBCLASS,
    icvBtNext_XXBCLASS,
    icvBtNext_XXBCLASS,
    icvBtNext_XXBCLASS,
    icvBtNext_L2CLASS,
    icvBtNext_LKCLASS,
    icvBtNext_LSREG,
    icvBtNext_LADREG,
    icvBtNext_MREG
};

CV_BOOST_IMPL
void cvBtNext( CvCARTClassifier** trees, CvBtTrainer* trainer )
{
    int i, j;
    int index;
    CvMat sample;
    int sample_step;
    uchar* sample_data;

    icvBtNextFunc[trainer->type]( trees, trainer );        

    /* shrinkage */
    if( trainer->param[0] != 1.0F )
    {
        for( j = 0; j < trainer->numclasses; j++ )
        {
            for( i = 0; i <= trees[j]->count; i++ )
            {
                trees[j]->val[i] *= trainer->param[0];
            }
        }
    }

    if( trainer->type > CV_GABCLASS )
    {
        /* update F_(m-1) */
        CV_GET_SAMPLE( *(trainer->trainData), trainer->flags, 0, sample );
        CV_GET_SAMPLE_STEP( *(trainer->trainData), trainer->flags, sample_step );
        sample_data = sample.data.ptr;
        for( i = 0; i < trainer->numsamples; i++ )
        {
            index = icvGetIdxAt( trainer->sampleIdx, i );
            sample.data.ptr = sample_data + index * sample_step;
            for( j = 0; j < trainer->numclasses; j++ )
            {            
                trainer->f[index * trainer->numclasses + j] += 
                    trees[j]->eval( (CvClassifier*) (trees[j]), &sample );
            }
        }
    }
}

CV_BOOST_IMPL
void cvBtEnd( CvBtTrainer** trainer )
{
    CV_FUNCNAME( "cvBtEnd" );
    
    __BEGIN__;
    
    if( trainer == NULL || (*trainer) == NULL )
    {
        CV_ERROR( CV_StsNullPtr, "Invalid trainer parameter" );
    }
    
    if( (*trainer)->y != NULL )
    {
        CV_CALL( cvReleaseMat( &((*trainer)->y) ) );
    }
    if( (*trainer)->weights != NULL )
    {
        CV_CALL( cvReleaseMat( &((*trainer)->weights) ) );
    }
    if( (*trainer)->boosttrainer != NULL )
    {
        CV_CALL( cvBoostEndTraining( &((*trainer)->boosttrainer) ) );
    }
    CV_CALL( cvFree( trainer ) );

    __END__;
}

/****************************************************************************************\
*                         Boosted tree model as a classifier                             *
\****************************************************************************************/

CV_BOOST_IMPL
float cvEvalBtClassifier( CvClassifier* classifier, CvMat* sample )
{
    float val;

    CV_FUNCNAME( "cvEvalBtClassifier" );

    __BEGIN__;
    
    int i;

    val = 0.0F;
    if( CV_IS_TUNABLE( classifier->flags ) )
    {
        CvSeqReader reader;
        CvCARTClassifier* tree;

        CV_CALL( cvStartReadSeq( ((CvBtClassifier*) classifier)->seq, &reader ) );
        for( i = 0; i < ((CvBtClassifier*) classifier)->numiter; i++ )
        {
            CV_READ_SEQ_ELEM( tree, reader );
            val += tree->eval( (CvClassifier*) tree, sample );
        }
    }
    else
    {
        CvCARTClassifier** ptree;

        ptree = ((CvBtClassifier*) classifier)->trees;
        for( i = 0; i < ((CvBtClassifier*) classifier)->numiter; i++ )
        {
            val += (*ptree)->eval( (CvClassifier*) (*ptree), sample );
            ptree++;
        }
    }

    __END__;

    return val;
}

CV_BOOST_IMPL
float cvEvalBtClassifier2( CvClassifier* classifier, CvMat* sample )
{
    float val;

    CV_FUNCNAME( "cvEvalBtClassifier2" );

    __BEGIN__;
    
    CV_CALL( val = cvEvalBtClassifier( classifier, sample ) );

    __END__;

    return (float) (val >= 0.0F);
}

CV_BOOST_IMPL
float cvEvalBtClassifierK( CvClassifier* classifier, CvMat* sample )
{
    int cls = 0;

    CV_FUNCNAME( "cvEvalBtClassifierK" );

    __BEGIN__;
    
    int i, k;
    float max_val;
    int numclasses;

    float* vals;
    size_t data_size;

    numclasses = ((CvBtClassifier*) classifier)->numclasses;
    data_size = sizeof( *vals ) * numclasses;
    CV_CALL( vals = (float*) cvAlloc( data_size ) );
    memset( vals, 0, data_size );

    if( CV_IS_TUNABLE( classifier->flags ) )
    {
        CvSeqReader reader;
        CvCARTClassifier* tree;

        CV_CALL( cvStartReadSeq( ((CvBtClassifier*) classifier)->seq, &reader ) );
        for( i = 0; i < ((CvBtClassifier*) classifier)->numiter; i++ )
        {
            for( k = 0; k < numclasses; k++ )
            {
                CV_READ_SEQ_ELEM( tree, reader );
                vals[k] += tree->eval( (CvClassifier*) tree, sample );
            }
        }

    }
    else
    {
        CvCARTClassifier** ptree;

        ptree = ((CvBtClassifier*) classifier)->trees;
        for( i = 0; i < ((CvBtClassifier*) classifier)->numiter; i++ )
        {
            for( k = 0; k < numclasses; k++ )
            {
                vals[k] += (*ptree)->eval( (CvClassifier*) (*ptree), sample );
                ptree++;
            }
        }
    }

    max_val = vals[cls];
    for( k = 1; k < numclasses; k++ )
    {
        if( vals[k] > max_val )
        {
            max_val = vals[k];
            cls = k;
        }
    }

    CV_CALL( cvFree( &vals ) );

    __END__;

    return (float) cls;
}

typedef float (*CvEvalBtClassifier)( CvClassifier* classifier, CvMat* sample );

static CvEvalBtClassifier icvEvalBtClassifier[] =
{
    cvEvalBtClassifier2,
    cvEvalBtClassifier2,
    cvEvalBtClassifier2,
    cvEvalBtClassifier2,
    cvEvalBtClassifier2,
    cvEvalBtClassifierK,
    cvEvalBtClassifier,
    cvEvalBtClassifier,
    cvEvalBtClassifier
};

CV_BOOST_IMPL
int cvSaveBtClassifier( CvClassifier* classifier, const char* filename )
{
    CV_FUNCNAME( "cvSaveBtClassifier" );

    __BEGIN__;

    FILE* file;
    int i, j;
    CvSeqReader reader;
    memset(&reader, 0, sizeof(reader));
    CvCARTClassifier* tree;

    CV_ASSERT( classifier );
    CV_ASSERT( filename );
    
    if( !icvMkDir( filename ) || (file = fopen( filename, "w" )) == 0 )
    {
        CV_ERROR( CV_StsError, "Unable to create file" );
    }

    if( CV_IS_TUNABLE( classifier->flags ) )
    {
        CV_CALL( cvStartReadSeq( ((CvBtClassifier*) classifier)->seq, &reader ) );
    }
    fprintf( file, "%d %d\n%d\n%d\n", (int) ((CvBtClassifier*) classifier)->type,
                                      ((CvBtClassifier*) classifier)->numclasses,
                                      ((CvBtClassifier*) classifier)->numfeatures,
                                      ((CvBtClassifier*) classifier)->numiter );
    
    for( i = 0; i < ((CvBtClassifier*) classifier)->numclasses *
                    ((CvBtClassifier*) classifier)->numiter; i++ )
    {
        if( CV_IS_TUNABLE( classifier->flags ) )
        {
            CV_READ_SEQ_ELEM( tree, reader );
        }
        else
        {
            tree = ((CvBtClassifier*) classifier)->trees[i];
        }

        fprintf( file, "%d\n", tree->count );
        for( j = 0; j < tree->count; j++ )
        {
            fprintf( file, "%d %g %d %d\n", tree->compidx[j],
                                            tree->threshold[j],
                                            tree->left[j],
                                            tree->right[j] );
        }
        for( j = 0; j <= tree->count; j++ )
        {
            fprintf( file, "%g ", tree->val[j] );
        }
        fprintf( file, "\n" );
    }

    fclose( file );

    __END__;

    return 1;
}


CV_BOOST_IMPL
void cvReleaseBtClassifier( CvClassifier** ptr )
{
    CV_FUNCNAME( "cvReleaseBtClassifier" );

    __BEGIN__;

    int i;

    if( ptr == NULL || *ptr == NULL )
    {
        CV_ERROR( CV_StsNullPtr, "" );
    }
    if( CV_IS_TUNABLE( (*ptr)->flags ) )
    {
        CvSeqReader reader;
        CvCARTClassifier* tree;

        CV_CALL( cvStartReadSeq( ((CvBtClassifier*) *ptr)->seq, &reader ) );
        for( i = 0; i < ((CvBtClassifier*) *ptr)->numclasses *
                        ((CvBtClassifier*) *ptr)->numiter; i++ )
        {
            CV_READ_SEQ_ELEM( tree, reader );
            tree->release( (CvClassifier**) (&tree) );
        }
        CV_CALL( cvReleaseMemStorage( &(((CvBtClassifier*) *ptr)->seq->storage) ) );
    }
    else
    {
        CvCARTClassifier** ptree;

        ptree = ((CvBtClassifier*) *ptr)->trees;
        for( i = 0; i < ((CvBtClassifier*) *ptr)->numclasses *
                        ((CvBtClassifier*) *ptr)->numiter; i++ )
        {
            (*ptree)->release( (CvClassifier**) ptree );
            ptree++;
        }
    }

    CV_CALL( cvFree( ptr ) );
    *ptr = NULL;

    __END__;
}

void cvTuneBtClassifier( CvClassifier* classifier, CvMat*, int flags,
                         CvMat*, CvMat* , CvMat*, CvMat*, CvMat* )
{
    CV_FUNCNAME( "cvTuneBtClassifier" );

    __BEGIN__;

    size_t data_size;

    if( CV_IS_TUNABLE( flags ) )
    {
        if( !CV_IS_TUNABLE( classifier->flags ) )
        {
            CV_ERROR( CV_StsUnsupportedFormat,
                      "Classifier does not support tune function" );
        }
        else
        {
            /* tune classifier */
            CvCARTClassifier** trees;

            printf( "Iteration %d\n", ((CvBtClassifier*) classifier)->numiter + 1 );

            data_size = sizeof( *trees ) * ((CvBtClassifier*) classifier)->numclasses;
            CV_CALL( trees = (CvCARTClassifier**) cvAlloc( data_size ) );
            CV_CALL( cvBtNext( trees,
                (CvBtTrainer*) ((CvBtClassifier*) classifier)->trainer ) );
            CV_CALL( cvSeqPushMulti( ((CvBtClassifier*) classifier)->seq,
                trees, ((CvBtClassifier*) classifier)->numclasses ) );
            CV_CALL( cvFree( &trees ) );
            ((CvBtClassifier*) classifier)->numiter++;
        }
    }
    else
    {
        if( CV_IS_TUNABLE( classifier->flags ) )
        {
            /* convert */
            void* ptr;

            assert( ((CvBtClassifier*) classifier)->seq->total ==
                        ((CvBtClassifier*) classifier)->numiter *
                        ((CvBtClassifier*) classifier)->numclasses );

            data_size = sizeof( ((CvBtClassifier*) classifier)->trees[0] ) *
                ((CvBtClassifier*) classifier)->seq->total;
            CV_CALL( ptr = cvAlloc( data_size ) );
            CV_CALL( cvCvtSeqToArray( ((CvBtClassifier*) classifier)->seq, ptr ) );
            CV_CALL( cvReleaseMemStorage( 
                    &(((CvBtClassifier*) classifier)->seq->storage) ) );
            ((CvBtClassifier*) classifier)->trees = (CvCARTClassifier**) ptr;
            classifier->flags &= ~CV_TUNABLE;
            CV_CALL( cvBtEnd( (CvBtTrainer**)
                &(((CvBtClassifier*) classifier)->trainer )) );
            ((CvBtClassifier*) classifier)->trainer = NULL;
        }
    }

    __END__;
}

CvBtClassifier* icvAllocBtClassifier( CvBoostType type, int flags, int numclasses,
                                      int numiter )
{
    CvBtClassifier* ptr;
    size_t data_size;

    assert( numclasses >= 1 );
    assert( numiter >= 0 );
    assert( ( numclasses == 1 ) || (type == CV_LKCLASS) );

    data_size = sizeof( *ptr );
    ptr = (CvBtClassifier*) cvAlloc( data_size );
    memset( ptr, 0, data_size );

    if( CV_IS_TUNABLE( flags ) )
    {
        ptr->seq = cvCreateSeq( 0, sizeof( *(ptr->seq) ), sizeof( *(ptr->trees) ),
                                cvCreateMemStorage() );
        ptr->numiter = 0;
    }
    else
    {
        data_size = numclasses * numiter * sizeof( *(ptr->trees) );
        ptr->trees = (CvCARTClassifier**) cvAlloc( data_size );
        memset( ptr->trees, 0, data_size );

        ptr->numiter = numiter;
    }

    ptr->flags = flags;
    ptr->numclasses = numclasses;
    ptr->type = type;

    ptr->eval = icvEvalBtClassifier[(int) type];
    ptr->tune = cvTuneBtClassifier;
    ptr->save = cvSaveBtClassifier;
    ptr->release = cvReleaseBtClassifier;

    return ptr;
}

CV_BOOST_IMPL
CvClassifier* cvCreateBtClassifier( CvMat* trainData,
                                    int flags,
                                    CvMat* trainClasses,
                                    CvMat* typeMask,
                                    CvMat* missedMeasurementsMask,
                                    CvMat* compIdx,
                                    CvMat* sampleIdx,
                                    CvMat* weights,
                                    CvClassifierTrainParams* trainParams )
{
    CvBtClassifier* ptr = 0;

    CV_FUNCNAME( "cvCreateBtClassifier" );

    __BEGIN__;
    CvBoostType type;
    int num_classes;
    int num_iter;
    int i;
    CvCARTClassifier** trees;
    size_t data_size;

    CV_ASSERT( trainData != NULL );
    CV_ASSERT( trainClasses != NULL );
    CV_ASSERT( typeMask == NULL );
    CV_ASSERT( missedMeasurementsMask == NULL );
    CV_ASSERT( compIdx == NULL );
    CV_ASSERT( weights == NULL );
    CV_ASSERT( trainParams != NULL );

    type = ((CvBtClassifierTrainParams*) trainParams)->type;
    
    if( type >= CV_DABCLASS && type <= CV_GABCLASS && sampleIdx )
    {
        CV_ERROR( CV_StsBadArg, "Sample indices are not supported for this type" );
    }

    if( type == CV_LKCLASS )
    {
        double min_val;
        double max_val;

        cvMinMaxLoc( trainClasses, &min_val, &max_val );
        num_classes = (int) (max_val + 1.0);
        
        CV_ASSERT( num_classes >= 2 );
    }
    else
    {
        num_classes = 1;
    }
    num_iter = ((CvBtClassifierTrainParams*) trainParams)->numiter;
    
    CV_ASSERT( num_iter > 0 );

    ptr = icvAllocBtClassifier( type, CV_TUNABLE | flags, num_classes, num_iter );
    ptr->numfeatures = (CV_IS_ROW_SAMPLE( flags )) ? trainData->cols : trainData->rows;
    
    i = 0;

    printf( "Iteration %d\n", 1 );

    data_size = sizeof( *trees ) * ptr->numclasses;
    CV_CALL( trees = (CvCARTClassifier**) cvAlloc( data_size ) );

    CV_CALL( ptr->trainer = cvBtStart( trees, trainData, flags, trainClasses, sampleIdx,
        ((CvBtClassifierTrainParams*) trainParams)->numsplits, type, num_classes,
        &(((CvBtClassifierTrainParams*) trainParams)->param[0]) ) );

    CV_CALL( cvSeqPushMulti( ptr->seq, trees, ptr->numclasses ) );
    CV_CALL( cvFree( &trees ) );
    ptr->numiter++;
    
    for( i = 1; i < num_iter; i++ )
    {
        ptr->tune( (CvClassifier*) ptr, NULL, CV_TUNABLE, NULL, NULL, NULL, NULL, NULL );
    }
    if( !CV_IS_TUNABLE( flags ) )
    {
        /* convert */
        ptr->tune( (CvClassifier*) ptr, NULL, 0, NULL, NULL, NULL, NULL, NULL );
    }

    __END__;

    return (CvClassifier*) ptr;
}

CV_BOOST_IMPL
CvClassifier* cvCreateBtClassifierFromFile( const char* filename )
{
    CvBtClassifier* ptr = 0;

    CV_FUNCNAME( "cvCreateBtClassifierFromFile" );
    
    __BEGIN__;

    FILE* file;
    int i, j;
    int data_size;
    int num_classifiers;
    int num_features;
    int num_classes;
    int type;

    CV_ASSERT( filename != NULL );

    ptr = NULL;
    file = fopen( filename, "r" );
    if( !file )
    {
        CV_ERROR( CV_StsError, "Unable to open file" );
    }
    
    fscanf( file, "%d %d %d %d", &type, &num_classes, &num_features, &num_classifiers );

    CV_ASSERT( type >= (int) CV_DABCLASS && type <= (int) CV_MREG );
    CV_ASSERT( num_features > 0 );
    CV_ASSERT( num_classifiers > 0 );

    if( (CvBoostType) type != CV_LKCLASS )
    {
        num_classes = 1;
    }
    ptr = icvAllocBtClassifier( (CvBoostType) type, 0, num_classes, num_classifiers );
    ptr->numfeatures = num_features;
    
    for( i = 0; i < num_classes * num_classifiers; i++ )
    {
        int count;
        CvCARTClassifier* tree;

        fscanf( file, "%d", &count );

        data_size = sizeof( *tree )
            + count * ( sizeof( *(tree->compidx) ) + sizeof( *(tree->threshold) ) +
                        sizeof( *(tree->right) ) + sizeof( *(tree->left) ) )
            + (count + 1) * ( sizeof( *(tree->val) ) );
        CV_CALL( tree = (CvCARTClassifier*) cvAlloc( data_size ) );
        memset( tree, 0, data_size );
        tree->eval = cvEvalCARTClassifier;
        tree->tune = NULL;
        tree->save = NULL;
        tree->release = cvReleaseCARTClassifier;
        tree->compidx = (int*) ( tree + 1 );
        tree->threshold = (float*) ( tree->compidx + count );
        tree->left = (int*) ( tree->threshold + count );
        tree->right = (int*) ( tree->left + count );
        tree->val = (float*) ( tree->right + count );

        tree->count = count;
        for( j = 0; j < tree->count; j++ )
        {
            fscanf( file, "%d %g %d %d", &(tree->compidx[j]),
                                         &(tree->threshold[j]),
                                         &(tree->left[j]),
                                         &(tree->right[j]) );
        }
        for( j = 0; j <= tree->count; j++ )
        {
            fscanf( file, "%g", &(tree->val[j]) );
        }
        ptr->trees[i] = tree;
    }

    fclose( file );

    __END__;

    return (CvClassifier*) ptr;
}

/****************************************************************************************\
*                                    Utility functions                                   *
\****************************************************************************************/

CV_BOOST_IMPL
CvMat* cvTrimWeights( CvMat* weights, CvMat* idx, float factor )
{
    CvMat* ptr = 0;

    CV_FUNCNAME( "cvTrimWeights" );
    __BEGIN__;
    int i, index, num;
    float sum_weights;
    uchar* wdata;
    size_t wstep;
    int wnum;
    float threshold;
    int count;
    float* sorted_weights;

    CV_ASSERT( CV_MAT_TYPE( weights->type ) == CV_32FC1 );

    ptr = idx;
    sorted_weights = NULL;

    if( factor > 0.0F && factor < 1.0F )
    {
        size_t data_size;

        CV_MAT2VEC( *weights, wdata, wstep, wnum );
        num = ( idx == NULL ) ? wnum : MAX( idx->rows, idx->cols );

        data_size = num * sizeof( *sorted_weights );
        sorted_weights = (float*) cvAlloc( data_size );
        memset( sorted_weights, 0, data_size );

        sum_weights = 0.0F;
        for( i = 0; i < num; i++ )
        {
            index = icvGetIdxAt( idx, i );
            sorted_weights[i] = *((float*) (wdata + index * wstep));
            sum_weights += sorted_weights[i];
        }

        icvSort_32f( sorted_weights, num, 0 );

        sum_weights *= (1.0F - factor);

        i = -1;
        do { sum_weights -= sorted_weights[++i]; }
        while( sum_weights > 0.0F && i < (num - 1) );

        threshold = sorted_weights[i];

        while( i > 0 && sorted_weights[i-1] == threshold ) i--;

        if( i > 0 || ( idx != NULL && CV_MAT_TYPE( idx->type ) != CV_32FC1 ) )
        {
            CV_CALL( ptr = cvCreateMat( 1, num - i, CV_32FC1 ) );
            count = 0;
            for( i = 0; i < num; i++ )
            {
                index = icvGetIdxAt( idx, i );
                if( *((float*) (wdata + index * wstep)) >= threshold )
                {
                    CV_MAT_ELEM( *ptr, float, 0, count ) = (float) index;
                    count++;
                }
            }
        
            assert( count == ptr->cols );
        }
        cvFree( &sorted_weights );
    }

    __END__;

    return ptr;
}


CV_BOOST_IMPL
void cvReadTrainData( const char* filename, int flags,
                      CvMat** trainData,
                      CvMat** trainClasses )
{

    CV_FUNCNAME( "cvReadTrainData" );

    __BEGIN__;

    FILE* file;
    int m, n;
    int i, j;
    float val;

    if( filename == NULL )
    {
        CV_ERROR( CV_StsNullPtr, "filename must be specified" );
    }
    if( trainData == NULL )
    {
        CV_ERROR( CV_StsNullPtr, "trainData must be not NULL" );
    }
    if( trainClasses == NULL )
    {
        CV_ERROR( CV_StsNullPtr, "trainClasses must be not NULL" );
    }
    
    *trainData = NULL;
    *trainClasses = NULL;
    file = fopen( filename, "r" );
    if( !file )
    {
        CV_ERROR( CV_StsError, "Unable to open file" );
    }

    fscanf( file, "%d %d", &m, &n );

    if( CV_IS_ROW_SAMPLE( flags ) )
    {
        CV_CALL( *trainData = cvCreateMat( m, n, CV_32FC1 ) );
    }
    else
    {
        CV_CALL( *trainData = cvCreateMat( n, m, CV_32FC1 ) );
    }
    
    CV_CALL( *trainClasses = cvCreateMat( 1, m, CV_32FC1 ) );

    for( i = 0; i < m; i++ )
    {
        for( j = 0; j < n; j++ )
        {
            fscanf( file, "%f", &val );
            if( CV_IS_ROW_SAMPLE( flags ) )
            {
                CV_MAT_ELEM( **trainData, float, i, j ) = val;
            }
            else
            {
                CV_MAT_ELEM( **trainData, float, j, i ) = val;
            }
        }
        fscanf( file, "%f", &val );
        CV_MAT_ELEM( **trainClasses, float, 0, i ) = val;
    }

    fclose( file );

    __END__;
    
}

CV_BOOST_IMPL
void cvWriteTrainData( const char* filename, int flags,
                       CvMat* trainData, CvMat* trainClasses, CvMat* sampleIdx )
{
    CV_FUNCNAME( "cvWriteTrainData" );

    __BEGIN__;

    FILE* file;
    int m, n;
    int i, j;
    int clsrow;
    int count;
    int idx;
    CvScalar sc;

    if( filename == NULL )
    {
        CV_ERROR( CV_StsNullPtr, "filename must be specified" );
    }
    if( trainData == NULL || CV_MAT_TYPE( trainData->type ) != CV_32FC1 )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Invalid trainData" );
    }
    if( CV_IS_ROW_SAMPLE( flags ) )
    {
        m = trainData->rows;
        n = trainData->cols;
    }
    else
    {
        n = trainData->rows;
        m = trainData->cols;
    }
    if( trainClasses == NULL || CV_MAT_TYPE( trainClasses->type ) != CV_32FC1 ||
        MIN( trainClasses->rows, trainClasses->cols ) != 1 )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Invalid trainClasses" );
    }
    clsrow = (trainClasses->rows == 1);
    if( m != ( (clsrow) ? trainClasses->cols : trainClasses->rows ) )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Incorrect trainData and trainClasses sizes" );
    }
    
    if( sampleIdx != NULL )
    {
        count = (sampleIdx->rows == 1) ? sampleIdx->cols : sampleIdx->rows;
    }
    else
    {
        count = m;
    }
    

    file = fopen( filename, "w" );
    if( !file )
    {
        CV_ERROR( CV_StsError, "Unable to create file" );
    }

    fprintf( file, "%d %d\n", count, n );

    for( i = 0; i < count; i++ )
    {
        if( sampleIdx )
        {
            if( sampleIdx->rows == 1 )
            {
                sc = cvGet2D( sampleIdx, 0, i );
            }
            else
            {
                sc = cvGet2D( sampleIdx, i, 0 );
            }
            idx = (int) sc.val[0];
        }
        else
        {
            idx = i;
        }
        for( j = 0; j < n; j++ )
        {
            fprintf( file, "%g ", ( (CV_IS_ROW_SAMPLE( flags ))
                                    ? CV_MAT_ELEM( *trainData, float, idx, j ) 
                                    : CV_MAT_ELEM( *trainData, float, j, idx ) ) );
        }
        fprintf( file, "%g\n", ( (clsrow)
                                ? CV_MAT_ELEM( *trainClasses, float, 0, idx )
                                : CV_MAT_ELEM( *trainClasses, float, idx, 0 ) ) );
    }

    fclose( file );
    
    __END__;
}


#define ICV_RAND_SHUFFLE( suffix, type )                                                 \
void icvRandShuffle_##suffix( uchar* data, size_t step, int num )                        \
{                                                                                        \
    time_t seed;                                                                         \
    type tmp;                                                                            \
    int i;                                                                               \
    float rn;                                                                            \
                                                                                         \
    time( &seed );                                                                       \
    CvRNG state = cvRNG((int)seed);                                                      \
                                                                                         \
    for( i = 0; i < (num-1); i++ )                                                       \
    {                                                                                    \
        rn = ((float) cvRandInt( &state )) / (1.0F + UINT_MAX);                          \
        CV_SWAP( *((type*)(data + i * step)),                                            \
                 *((type*)(data + ( i + (int)( rn * (num - i ) ) )* step)),              \
                 tmp );                                                                  \
    }                                                                                    \
}

ICV_RAND_SHUFFLE( 8U, uchar )

ICV_RAND_SHUFFLE( 16S, short )

ICV_RAND_SHUFFLE( 32S, int )

ICV_RAND_SHUFFLE( 32F, float )

CV_BOOST_IMPL
void cvRandShuffleVec( CvMat* mat )
{
    CV_FUNCNAME( "cvRandShuffle" );

    __BEGIN__;

    uchar* data;
    size_t step;
    int num;

    if( (mat == NULL) || !CV_IS_MAT( mat ) || MIN( mat->rows, mat->cols ) != 1 )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "" );
    }

    CV_MAT2VEC( *mat, data, step, num );
    switch( CV_MAT_TYPE( mat->type ) )
    {
        case CV_8UC1:
            icvRandShuffle_8U( data, step, num);
            break;
        case CV_16SC1:
            icvRandShuffle_16S( data, step, num);
            break;
        case CV_32SC1:
            icvRandShuffle_32S( data, step, num);
            break;
        case CV_32FC1:
            icvRandShuffle_32F( data, step, num);
            break;
        default:
            CV_ERROR( CV_StsUnsupportedFormat, "" );
    }

    __END__;
}

/* End of file. */
