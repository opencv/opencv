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

/*
 * cvhaartraining.cpp
 *
 * training of cascade of boosted classifiers based on haar features
 */

#include "cvhaartraining.h"
#include "_cvhaartraining.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <ctype.h>

#include "highgui.h"

#ifdef CV_VERBOSE
#include <ctime>

#ifdef _WIN32
/* use clock() function insted of time() */
#define TIME( arg ) (((double) clock()) / CLOCKS_PER_SEC)
#else
#define TIME( arg ) (time( arg ))
#endif /* _WIN32 */

#endif /* CV_VERBOSE */

#if defined CV_OPENMP && (defined _MSC_VER || defined CV_ICC)
#define CV_OPENMP 1
#else
#undef CV_OPENMP
#endif

typedef struct CvBackgroundData
{
    int    count;
    char** filename;
    int    last;
    int    round;
    CvSize winsize;
} CvBackgroundData;

typedef struct CvBackgroundReader
{
    CvMat   src;
    CvMat   img;
    CvPoint offset;
    float   scale;
    float   scalefactor;
    float   stepfactor;
    CvPoint point;
} CvBackgroundReader;

/*
 * Background reader
 * Created in each thread
 */
CvBackgroundReader* cvbgreader = NULL;

#if defined CV_OPENMP
#pragma omp threadprivate(cvbgreader)
#endif

CvBackgroundData* cvbgdata = NULL;


/*
 * get sum image offsets for <rect> corner points 
 * step - row step (measured in image pixels!) of sum image
 */
#define CV_SUM_OFFSETS( p0, p1, p2, p3, rect, step )                      \
    /* (x, y) */                                                          \
    (p0) = (rect).x + (step) * (rect).y;                                  \
    /* (x + w, y) */                                                      \
    (p1) = (rect).x + (rect).width + (step) * (rect).y;                   \
    /* (x + w, y) */                                                      \
    (p2) = (rect).x + (step) * ((rect).y + (rect).height);                \
    /* (x + w, y + h) */                                                  \
    (p3) = (rect).x + (rect).width + (step) * ((rect).y + (rect).height);

/*
 * get tilted image offsets for <rect> corner points 
 * step - row step (measured in image pixels!) of tilted image
 */
#define CV_TILTED_OFFSETS( p0, p1, p2, p3, rect, step )                   \
    /* (x, y) */                                                          \
    (p0) = (rect).x + (step) * (rect).y;                                  \
    /* (x - h, y + h) */                                                  \
    (p1) = (rect).x - (rect).height + (step) * ((rect).y + (rect).height);\
    /* (x + w, y + w) */                                                  \
    (p2) = (rect).x + (rect).width + (step) * ((rect).y + (rect).width);  \
    /* (x + w - h, y + w + h) */                                          \
    (p3) = (rect).x + (rect).width - (rect).height                        \
           + (step) * ((rect).y + (rect).width + (rect).height);


/*
 * icvCreateIntHaarFeatures
 *
 * Create internal representation of haar features
 *
 * mode:
 *  0 - BASIC = Viola
 *  1 - CORE  = All upright
 *  2 - ALL   = All features
 */
static
CvIntHaarFeatures* icvCreateIntHaarFeatures( CvSize winsize,
                                             int mode,
                                             int symmetric )
{
    CvIntHaarFeatures* features = NULL;
    CvTHaarFeature haarFeature;
    
    CvMemStorage* storage = NULL;
    CvSeq* seq = NULL;
    CvSeqWriter writer;

    int s0 = 36; /* minimum total area size of basic haar feature     */
    int s1 = 12; /* minimum total area size of tilted haar features 2 */
    int s2 = 18; /* minimum total area size of tilted haar features 3 */
    int s3 = 24; /* minimum total area size of tilted haar features 4 */

    int x  = 0;
    int y  = 0;
    int dx = 0;
    int dy = 0;

    float factor = 1.0F;

    factor = ((float) winsize.width) * winsize.height / (24 * 24);
#if 0    
    s0 = (int) (s0 * factor);
    s1 = (int) (s1 * factor);
    s2 = (int) (s2 * factor);
    s3 = (int) (s3 * factor);
#else
    s0 = 1;
    s1 = 1;
    s2 = 1;
    s3 = 1;
#endif

    /* CV_VECTOR_CREATE( vec, CvIntHaarFeature, size, maxsize ) */
    storage = cvCreateMemStorage();
    cvStartWriteSeq( 0, sizeof( CvSeq ), sizeof( haarFeature ), storage, &writer );

    for( x = 0; x < winsize.width; x++ )
    {
        for( y = 0; y < winsize.height; y++ )
        {
            for( dx = 1; dx <= winsize.width; dx++ )
            {
                for( dy = 1; dy <= winsize.height; dy++ )
                {
                    // haar_x2
                    if ( (x+dx*2 <= winsize.width) && (y+dy <= winsize.height) ) {
                        if (dx*2*dy < s0) continue;
                        if (!symmetric || (x+x+dx*2 <=winsize.width)) {
                            haarFeature = cvHaarFeature( "haar_x2",
                                x,    y, dx*2, dy, -1,
                                x+dx, y, dx  , dy, +2 );
                            /* CV_VECTOR_PUSH( vec, CvIntHaarFeature, haarFeature, size, maxsize, step ) */
                            CV_WRITE_SEQ_ELEM( haarFeature, writer );
                        }
                    }

                    // haar_y2
                    if ( (x+dx <= winsize.width) && (y+dy*2 <= winsize.height) ) {
                        if (dx*2*dy < s0) continue;
                        if (!symmetric || (x+x+dx <= winsize.width)) {
                            haarFeature = cvHaarFeature( "haar_y2",
                                x, y,    dx, dy*2, -1,
                                x, y+dy, dx, dy,   +2 );
                            CV_WRITE_SEQ_ELEM( haarFeature, writer );
                        }
                    }

                    // haar_x3
                    if ( (x+dx*3 <= winsize.width) && (y+dy <= winsize.height) ) {
                        if (dx*3*dy < s0) continue;
                        if (!symmetric || (x+x+dx*3 <=winsize.width)) {
                            haarFeature = cvHaarFeature( "haar_x3",
                                x,    y, dx*3, dy, -1,
                                x+dx, y, dx,   dy, +3 );
                            CV_WRITE_SEQ_ELEM( haarFeature, writer );
                        }
                    }

                    // haar_y3
                    if ( (x+dx <= winsize.width) && (y+dy*3 <= winsize.height) ) {
                        if (dx*3*dy < s0) continue;
                        if (!symmetric || (x+x+dx <= winsize.width)) {
                            haarFeature = cvHaarFeature( "haar_y3",
                                x, y,    dx, dy*3, -1,
                                x, y+dy, dx, dy,   +3 );
                            CV_WRITE_SEQ_ELEM( haarFeature, writer );
                        }
                    }

                    if( mode != 0 /*BASIC*/ ) {
                        // haar_x4
                        if ( (x+dx*4 <= winsize.width) && (y+dy <= winsize.height) ) {
                            if (dx*4*dy < s0) continue;
                            if (!symmetric || (x+x+dx*4 <=winsize.width)) {
                                haarFeature = cvHaarFeature( "haar_x4",
                                    x,    y, dx*4, dy, -1,
                                    x+dx, y, dx*2, dy, +2 );
                                CV_WRITE_SEQ_ELEM( haarFeature, writer );
                            }
                        }
                            
                        // haar_y4
                        if ( (x+dx <= winsize.width ) && (y+dy*4 <= winsize.height) ) {
                            if (dx*4*dy < s0) continue;
                            if (!symmetric || (x+x+dx   <=winsize.width)) {
                                haarFeature = cvHaarFeature( "haar_y4",
                                    x, y,    dx, dy*4, -1,
                                    x, y+dy, dx, dy*2, +2 );
                                CV_WRITE_SEQ_ELEM( haarFeature, writer );
                            }
                        }
                    }

                    // x2_y2
                    if ( (x+dx*2 <= winsize.width) && (y+dy*2 <= winsize.height) ) {
                        if (dx*4*dy < s0) continue;
                        if (!symmetric || (x+x+dx*2 <=winsize.width)) {
                            haarFeature = cvHaarFeature( "haar_x2_y2",
                                x   , y,    dx*2, dy*2, -1,
                                x   , y   , dx  , dy,   +2,
                                x+dx, y+dy, dx  , dy,   +2 );
                            CV_WRITE_SEQ_ELEM( haarFeature, writer );
                        }
                    }

                    if (mode != 0 /*BASIC*/) {                
                        // point
                        if ( (x+dx*3 <= winsize.width) && (y+dy*3 <= winsize.height) ) {
                            if (dx*9*dy < s0) continue;
                            if (!symmetric || (x+x+dx*3 <=winsize.width))  {
                                haarFeature = cvHaarFeature( "haar_point",
                                    x   , y,    dx*3, dy*3, -1,
                                    x+dx, y+dy, dx  , dy  , +9);
                                CV_WRITE_SEQ_ELEM( haarFeature, writer );
                            }
                        }
                    }
                    
                    if (mode == 2 /*ALL*/) {                
                        // tilted haar_x2                                      (x, y, w, h, b, weight)
                        if ( (x+2*dx <= winsize.width) && (y+2*dx+dy <= winsize.height) && (x-dy>= 0) ) {
                            if (dx*2*dy < s1) continue;
                            
                            if (!symmetric || (x <= (winsize.width / 2) )) {
                                haarFeature = cvHaarFeature( "tilted_haar_x2",
                                    x, y, dx*2, dy, -1,
                                    x, y, dx  , dy, +2 );
                                CV_WRITE_SEQ_ELEM( haarFeature, writer );
                            }
                        }
                        
                        // tilted haar_y2                                      (x, y, w, h, b, weight)
                        if ( (x+dx <= winsize.width) && (y+dx+2*dy <= winsize.height) && (x-2*dy>= 0) ) {
                            if (dx*2*dy < s1) continue;
                            
                            if (!symmetric || (x <= (winsize.width / 2) )) {
                                haarFeature = cvHaarFeature( "tilted_haar_y2",
                                    x, y, dx, 2*dy, -1,
                                    x, y, dx,   dy, +2 );
                                CV_WRITE_SEQ_ELEM( haarFeature, writer );
                            }
                        }
                        
                        // tilted haar_x3                                   (x, y, w, h, b, weight)
                        if ( (x+3*dx <= winsize.width) && (y+3*dx+dy <= winsize.height) && (x-dy>= 0) ) {
                            if (dx*3*dy < s2) continue;
                            
                            if (!symmetric || (x <= (winsize.width / 2) )) {
                                haarFeature = cvHaarFeature( "tilted_haar_x3",
                                    x,    y,    dx*3, dy, -1,
                                    x+dx, y+dx, dx  , dy, +3 );
                                CV_WRITE_SEQ_ELEM( haarFeature, writer );
                            }
                        }
                        
                        // tilted haar_y3                                      (x, y, w, h, b, weight)
                        if ( (x+dx <= winsize.width) && (y+dx+3*dy <= winsize.height) && (x-3*dy>= 0) ) {
                            if (dx*3*dy < s2) continue;
                            
                            if (!symmetric || (x <= (winsize.width / 2) )) {
                                haarFeature = cvHaarFeature( "tilted_haar_y3",
                                    x,    y,    dx, 3*dy, -1,
                                    x-dy, y+dy, dx,   dy, +3 );
                                CV_WRITE_SEQ_ELEM( haarFeature, writer );
                            }
                        }
                        
                        
                        // tilted haar_x4                                   (x, y, w, h, b, weight)
                        if ( (x+4*dx <= winsize.width) && (y+4*dx+dy <= winsize.height) && (x-dy>= 0) ) {
                            if (dx*4*dy < s3) continue;
                            
                            if (!symmetric || (x <= (winsize.width / 2) )) {
                                haarFeature = cvHaarFeature( "tilted_haar_x4",


                                    x,    y,    dx*4, dy, -1,
                                    x+dx, y+dx, dx*2, dy, +2 );
                                CV_WRITE_SEQ_ELEM( haarFeature, writer );
                            }
                        }
                        
                        // tilted haar_y4                                      (x, y, w, h, b, weight)
                        if ( (x+dx <= winsize.width) && (y+dx+4*dy <= winsize.height) && (x-4*dy>= 0) ) {
                            if (dx*4*dy < s3) continue;
                            
                            if (!symmetric || (x <= (winsize.width / 2) )) {
                                haarFeature = cvHaarFeature( "tilted_haar_y4",
                                    x,    y,    dx, 4*dy, -1,
                                    x-dy, y+dy, dx, 2*dy, +2 );
                                CV_WRITE_SEQ_ELEM( haarFeature, writer );
                            }
                        }
                        

                        /*
                        
                          // tilted point
                          if ( (x+dx*3 <= winsize.width - 1) && (y+dy*3 <= winsize.height - 1) && (x-3*dy>= 0)) {
                          if (dx*9*dy < 36) continue;
                          if (!symmetric || (x <= (winsize.width / 2) ))  {
                            haarFeature = cvHaarFeature( "tilted_haar_point",
                                x, y,    dx*3, dy*3, -1,
                                x, y+dy, dx  , dy,   +9 );
                                CV_WRITE_SEQ_ELEM( haarFeature, writer );
                          }
                          }
                        */
                    }
                }
            }
        }
    }

    seq = cvEndWriteSeq( &writer );
    features = (CvIntHaarFeatures*) cvAlloc( sizeof( CvIntHaarFeatures ) +
        ( sizeof( CvTHaarFeature ) + sizeof( CvFastHaarFeature ) ) * seq->total );
    features->feature = (CvTHaarFeature*) (features + 1);
    features->fastfeature = (CvFastHaarFeature*) ( features->feature + seq->total );
    features->count = seq->total;
    features->winsize = winsize;
    cvCvtSeqToArray( seq, (CvArr*) features->feature );
    cvReleaseMemStorage( &storage );
    
    icvConvertToFastHaarFeature( features->feature, features->fastfeature,
                                 features->count, (winsize.width + 1) );
    
    return features;
}

static
void icvReleaseIntHaarFeatures( CvIntHaarFeatures** intHaarFeatures )
{
    if( intHaarFeatures != NULL && (*intHaarFeatures) != NULL )
    {
        cvFree( intHaarFeatures );
        (*intHaarFeatures) = NULL;
    }
}


void icvConvertToFastHaarFeature( CvTHaarFeature* haarFeature,
                                  CvFastHaarFeature* fastHaarFeature,
                                  int size, int step )
{
    int i = 0;
    int j = 0;

    for( i = 0; i < size; i++ )
    {
        fastHaarFeature[i].tilted = haarFeature[i].tilted;
        if( !fastHaarFeature[i].tilted )
        {
            for( j = 0; j < CV_HAAR_FEATURE_MAX; j++ )
            {
                fastHaarFeature[i].rect[j].weight = haarFeature[i].rect[j].weight;
                if( fastHaarFeature[i].rect[j].weight == 0.0F )
                {
                    break;
                }
                CV_SUM_OFFSETS( fastHaarFeature[i].rect[j].p0,
                                fastHaarFeature[i].rect[j].p1,
                                fastHaarFeature[i].rect[j].p2,
                                fastHaarFeature[i].rect[j].p3,
                                haarFeature[i].rect[j].r, step )
            }
            
        }
        else
        {
            for( j = 0; j < CV_HAAR_FEATURE_MAX; j++ )
            {
                fastHaarFeature[i].rect[j].weight = haarFeature[i].rect[j].weight;
                if( fastHaarFeature[i].rect[j].weight == 0.0F )
                {
                    break;
                }
                CV_TILTED_OFFSETS( fastHaarFeature[i].rect[j].p0,
                                   fastHaarFeature[i].rect[j].p1,
                                   fastHaarFeature[i].rect[j].p2,
                                   fastHaarFeature[i].rect[j].p3,
                                   haarFeature[i].rect[j].r, step )
            }
        }
    }
}


/*
 * icvCreateHaarTrainingData
 *
 * Create haar training data used in stage training
 */
static
CvHaarTrainigData* icvCreateHaarTrainingData( CvSize winsize, int maxnumsamples )
{
    CvHaarTrainigData* data;
    
    CV_FUNCNAME( "icvCreateHaarTrainingData" );
    
    __BEGIN__;

    data = NULL;
    uchar* ptr = NULL;
    size_t datasize = 0;
    
    datasize = sizeof( CvHaarTrainigData ) +
          /* sum and tilted */
        ( 2 * (winsize.width + 1) * (winsize.height + 1) * sizeof( sum_type ) +
          sizeof( float ) +      /* normfactor */
          sizeof( float ) +      /* cls */
          sizeof( float )        /* weight */
        ) * maxnumsamples;

    CV_CALL( data = (CvHaarTrainigData*) cvAlloc( datasize ) );
    memset( (void*)data, 0, datasize );
    data->maxnum = maxnumsamples;
    data->winsize = winsize;
    ptr = (uchar*)(data + 1);
    data->sum = cvMat( maxnumsamples, (winsize.width + 1) * (winsize.height + 1),
                       CV_SUM_MAT_TYPE, (void*) ptr );
    ptr += sizeof( sum_type ) * maxnumsamples * (winsize.width+1) * (winsize.height+1);
    data->tilted = cvMat( maxnumsamples, (winsize.width + 1) * (winsize.height + 1),
                       CV_SUM_MAT_TYPE, (void*) ptr );
    ptr += sizeof( sum_type ) * maxnumsamples * (winsize.width+1) * (winsize.height+1);
    data->normfactor = cvMat( 1, maxnumsamples, CV_32FC1, (void*) ptr );
    ptr += sizeof( float ) * maxnumsamples;
    data->cls = cvMat( 1, maxnumsamples, CV_32FC1, (void*) ptr );
    ptr += sizeof( float ) * maxnumsamples;
    data->weights = cvMat( 1, maxnumsamples, CV_32FC1, (void*) ptr );

    data->valcache = NULL;
    data->idxcache = NULL;

    __END__;

    return data;
}

static
void icvReleaseHaarTrainingDataCache( CvHaarTrainigData** haarTrainingData )
{
    if( haarTrainingData != NULL && (*haarTrainingData) != NULL )
    {
        if( (*haarTrainingData)->valcache != NULL )
        {
            cvReleaseMat( &(*haarTrainingData)->valcache );
            (*haarTrainingData)->valcache = NULL;
        }
        if( (*haarTrainingData)->idxcache != NULL )
        {
            cvReleaseMat( &(*haarTrainingData)->idxcache );
            (*haarTrainingData)->idxcache = NULL;
        }
    }
}

static
void icvReleaseHaarTrainingData( CvHaarTrainigData** haarTrainingData )
{
    if( haarTrainingData != NULL && (*haarTrainingData) != NULL )
    {
        icvReleaseHaarTrainingDataCache( haarTrainingData );

        cvFree( haarTrainingData );
    }
}

static
void icvGetTrainingDataCallback( CvMat* mat, CvMat* sampleIdx, CvMat*,
                                 int first, int num, void* userdata )
{
    int i = 0;
    int j = 0;
    float val = 0.0F;
    float normfactor = 0.0F;
    
    CvHaarTrainingData* training_data;
    CvIntHaarFeatures* haar_features;

#ifdef CV_COL_ARRANGEMENT
    assert( mat->rows >= num );
#else
    assert( mat->cols >= num );
#endif

    training_data = ((CvUserdata*) userdata)->trainingData;
    haar_features = ((CvUserdata*) userdata)->haarFeatures;
    if( sampleIdx == NULL )
    {
        int num_samples;

#ifdef CV_COL_ARRANGEMENT
        num_samples = mat->cols;
#else
        num_samples = mat->rows;
#endif
        for( i = 0; i < num_samples; i++ )
        {
            for( j = 0; j < num; j++ )
            {
                val = cvEvalFastHaarFeature(
                        ( haar_features->fastfeature
                            + first + j ),
                        (sum_type*) (training_data->sum.data.ptr
                            + i * training_data->sum.step),
                        (sum_type*) (training_data->tilted.data.ptr
                            + i * training_data->tilted.step) );
                normfactor = training_data->normfactor.data.fl[i];
                val = ( normfactor == 0.0F ) ? 0.0F : (val / normfactor);

#ifdef CV_COL_ARRANGEMENT
                CV_MAT_ELEM( *mat, float, j, i ) = val;
#else
                CV_MAT_ELEM( *mat, float, i, j ) = val;
#endif
            }
        }
    }
    else
    {
        uchar* idxdata = NULL;
        size_t step    = 0;
        int    numidx  = 0;
        int    idx     = 0;

        assert( CV_MAT_TYPE( sampleIdx->type ) == CV_32FC1 );

        idxdata = sampleIdx->data.ptr;
        if( sampleIdx->rows == 1 )
        {
            step = sizeof( float );
            numidx = sampleIdx->cols;
        }
        else
        {
            step = sampleIdx->step;
            numidx = sampleIdx->rows;
        }

        for( i = 0; i < numidx; i++ )
        {
            for( j = 0; j < num; j++ )
            {
                idx = (int)( *((float*) (idxdata + i * step)) );
                val = cvEvalFastHaarFeature(
                        ( haar_features->fastfeature
                            + first + j ),
                        (sum_type*) (training_data->sum.data.ptr
                            + idx * training_data->sum.step),
                        (sum_type*) (training_data->tilted.data.ptr
                            + idx * training_data->tilted.step) );
                normfactor = training_data->normfactor.data.fl[idx];
                val = ( normfactor == 0.0F ) ? 0.0F : (val / normfactor);

#ifdef CV_COL_ARRANGEMENT
                CV_MAT_ELEM( *mat, float, j, idx ) = val;
#else
                CV_MAT_ELEM( *mat, float, idx, j ) = val;
#endif

            }
        }
    }
#if 0 /*def CV_VERBOSE*/
    if( first % 5000 == 0 )
    {
        fprintf( stderr, "%3d%%\r", (int) (100.0 * first / 
            haar_features->count) );
        fflush( stderr );
    }
#endif /* CV_VERBOSE */
}

static
void icvPrecalculate( CvHaarTrainingData* data, CvIntHaarFeatures* haarFeatures,
                      int numprecalculated )
{
    CV_FUNCNAME( "icvPrecalculate" );

    __BEGIN__;

    icvReleaseHaarTrainingDataCache( &data );

    numprecalculated -= numprecalculated % CV_STUMP_TRAIN_PORTION;
    numprecalculated = MIN( numprecalculated, haarFeatures->count );

    if( numprecalculated > 0 )
    {
        //size_t datasize;
        int m;
        CvUserdata userdata;

        /* private variables */
        #ifdef CV_OPENMP
        CvMat t_data;
        CvMat t_idx;
        int first;
        int t_portion;
        int portion = CV_STUMP_TRAIN_PORTION;
        #endif /* CV_OPENMP */

        m = data->sum.rows;

#ifdef CV_COL_ARRANGEMENT
        CV_CALL( data->valcache = cvCreateMat( numprecalculated, m, CV_32FC1 ) );
#else
        CV_CALL( data->valcache = cvCreateMat( m, numprecalculated, CV_32FC1 ) );
#endif
        CV_CALL( data->idxcache = cvCreateMat( numprecalculated, m, CV_IDX_MAT_TYPE ) );

        userdata = cvUserdata( data, haarFeatures );

        #ifdef CV_OPENMP
        #pragma omp parallel for private(t_data, t_idx, first, t_portion)
        for( first = 0; first < numprecalculated; first += portion )
        {
            t_data = *data->valcache;
            t_idx = *data->idxcache;
            t_portion = MIN( portion, (numprecalculated - first) );
            
            /* indices */
            t_idx.rows = t_portion;
            t_idx.data.ptr = data->idxcache->data.ptr + first * ((size_t)t_idx.step);

            /* feature values */
#ifdef CV_COL_ARRANGEMENT
            t_data.rows = t_portion;
            t_data.data.ptr = data->valcache->data.ptr +
                first * ((size_t) t_data.step );
#else
            t_data.cols = t_portion;
            t_data.data.ptr = data->valcache->data.ptr +
                first * ((size_t) CV_ELEM_SIZE( t_data.type ));
#endif
            icvGetTrainingDataCallback( &t_data, NULL, NULL, first, t_portion,
                                        &userdata );
#ifdef CV_COL_ARRANGEMENT
            cvGetSortedIndices( &t_data, &t_idx, 0 );
#else
            cvGetSortedIndices( &t_data, &t_idx, 1 );
#endif

#ifdef CV_VERBOSE
            putc( '.', stderr );
            fflush( stderr );
#endif /* CV_VERBOSE */

        }

#ifdef CV_VERBOSE
        fprintf( stderr, "\n" );
        fflush( stderr );
#endif /* CV_VERBOSE */

        #else
        icvGetTrainingDataCallback( data->valcache, NULL, NULL, 0, numprecalculated,
                                    &userdata );
#ifdef CV_COL_ARRANGEMENT
        cvGetSortedIndices( data->valcache, data->idxcache, 0 );
#else
        cvGetSortedIndices( data->valcache, data->idxcache, 1 );
#endif
        #endif /* CV_OPENMP */
    }

    __END__;
}

static
void icvSplitIndicesCallback( int compidx, float threshold,
                              CvMat* idx, CvMat** left, CvMat** right,
                              void* userdata )
{
    CvHaarTrainingData* data;
    CvIntHaarFeatures* haar_features;
    int i;
    int m;
    CvFastHaarFeature* fastfeature;

    data = ((CvUserdata*) userdata)->trainingData;
    haar_features = ((CvUserdata*) userdata)->haarFeatures;
    fastfeature = &haar_features->fastfeature[compidx];

    m = data->sum.rows;
    *left = cvCreateMat( 1, m, CV_32FC1 );
    *right = cvCreateMat( 1, m, CV_32FC1 );
    (*left)->cols = (*right)->cols = 0;
    if( idx == NULL )
    {
        for( i = 0; i < m; i++ )
        {
            if( cvEvalFastHaarFeature( fastfeature,
                    (sum_type*) (data->sum.data.ptr + i * data->sum.step),
                    (sum_type*) (data->tilted.data.ptr + i * data->tilted.step) ) 
                < threshold * data->normfactor.data.fl[i] )
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
        int    idxnum;
        size_t idxstep;
        int    index;

        idxdata = idx->data.ptr;
        idxnum = (idx->rows == 1) ? idx->cols : idx->rows;
        idxstep = (idx->rows == 1) ? CV_ELEM_SIZE( idx->type ) : idx->step;
        for( i = 0; i < idxnum; i++ )
        {
            index = (int) *((float*) (idxdata + i * idxstep));
            if( cvEvalFastHaarFeature( fastfeature,
                    (sum_type*) (data->sum.data.ptr + index * data->sum.step),
                    (sum_type*) (data->tilted.data.ptr + index * data->tilted.step) ) 
                < threshold * data->normfactor.data.fl[index] )
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

/*
 * icvCreateCARTStageClassifier
 *
 * Create stage classifier with trees as weak classifiers
 * data             - haar training data. It must be created and filled before call
 * minhitrate       - desired min hit rate
 * maxfalsealarm    - desired max false alarm rate
 * symmetric        - if not 0 it is assumed that samples are vertically symmetric
 * numprecalculated - number of features that will be precalculated. Each precalculated
 *   feature need (number_of_samples*(sizeof( float ) + sizeof( short ))) bytes of memory
 * weightfraction   - weight trimming parameter
 * numsplits        - number of binary splits in each tree
 * boosttype        - type of applied boosting algorithm
 * stumperror       - type of used error if Discrete AdaBoost algorithm is applied
 * maxsplits        - maximum total number of splits in all weak classifiers.
 *   If it is not 0 then NULL returned if total number of splits exceeds <maxsplits>.
 */
static
CvIntHaarClassifier* icvCreateCARTStageClassifier( CvHaarTrainingData* data,
                                                   CvMat* sampleIdx,
                                                   CvIntHaarFeatures* haarFeatures,
                                                   float minhitrate,
                                                   float maxfalsealarm,
                                                   int   symmetric,
                                                   float weightfraction,
                                                   int numsplits,
                                                   CvBoostType boosttype,
                                                   CvStumpError stumperror,
                                                   int maxsplits )
{

#ifdef CV_COL_ARRANGEMENT
    int flags = CV_COL_SAMPLE;
#else
    int flags = CV_ROW_SAMPLE;
#endif

    CvStageHaarClassifier* stage = NULL;
    CvBoostTrainer* trainer;
    CvCARTClassifier* cart = NULL;
    CvCARTTrainParams trainParams;
    CvMTStumpTrainParams stumpTrainParams;
    //CvMat* trainData = NULL;
    //CvMat* sortedIdx = NULL;
    CvMat eval;
    int n = 0;
    int m = 0;
    int numpos = 0;
    int numneg = 0;
    int numfalse = 0;
    float sum_stage = 0.0F;
    float threshold = 0.0F;
    float falsealarm = 0.0F;
    
    //CvMat* sampleIdx = NULL;
    CvMat* trimmedIdx;
    //float* idxdata = NULL;
    //float* tempweights = NULL;
    //int    idxcount = 0;
    CvUserdata userdata;

    int i = 0;
    int j = 0;
    int idx;
    int numsamples;
    int numtrimmed;
    
    CvCARTHaarClassifier* classifier;
    CvSeq* seq = NULL;
    CvMemStorage* storage = NULL;
    CvMat* weakTrainVals;
    float alpha;
    float sumalpha;
    int num_splits; /* total number of splits in all weak classifiers */

#ifdef CV_VERBOSE
    printf( "+----+----+-+---------+---------+---------+---------+\n" );
    printf( "|  N |%%SMP|F|  ST.THR |    HR   |    FA   | EXP. ERR|\n" );
    printf( "+----+----+-+---------+---------+---------+---------+\n" );
#endif /* CV_VERBOSE */
    
    n = haarFeatures->count;
    m = data->sum.rows;
    numsamples = (sampleIdx) ? MAX( sampleIdx->rows, sampleIdx->cols ) : m;

    userdata = cvUserdata( data, haarFeatures );

    stumpTrainParams.type = ( boosttype == CV_DABCLASS )
        ? CV_CLASSIFICATION_CLASS : CV_REGRESSION;
    stumpTrainParams.error = ( boosttype == CV_LBCLASS || boosttype == CV_GABCLASS )
        ? CV_SQUARE : stumperror;
    stumpTrainParams.portion = CV_STUMP_TRAIN_PORTION;
    stumpTrainParams.getTrainData = icvGetTrainingDataCallback;
    stumpTrainParams.numcomp = n;
    stumpTrainParams.userdata = &userdata;
    stumpTrainParams.sortedIdx = data->idxcache;

    trainParams.count = numsplits;
    trainParams.stumpTrainParams = (CvClassifierTrainParams*) &stumpTrainParams;
    trainParams.stumpConstructor = cvCreateMTStumpClassifier;
    trainParams.splitIdx = icvSplitIndicesCallback;
    trainParams.userdata = &userdata;

    eval = cvMat( 1, m, CV_32FC1, cvAlloc( sizeof( float ) * m ) );
    
    storage = cvCreateMemStorage();
    seq = cvCreateSeq( 0, sizeof( *seq ), sizeof( classifier ), storage );

    weakTrainVals = cvCreateMat( 1, m, CV_32FC1 );
    trainer = cvBoostStartTraining( &data->cls, weakTrainVals, &data->weights,
                                    sampleIdx, boosttype );
    num_splits = 0;
    sumalpha = 0.0F;
    do
    {     

#ifdef CV_VERBOSE
        int v_wt = 0;
        int v_flipped = 0;
#endif /* CV_VERBOSE */

        trimmedIdx = cvTrimWeights( &data->weights, sampleIdx, weightfraction );
        numtrimmed = (trimmedIdx) ? MAX( trimmedIdx->rows, trimmedIdx->cols ) : m;

#ifdef CV_VERBOSE
        v_wt = 100 * numtrimmed / numsamples;
        v_flipped = 0;

#endif /* CV_VERBOSE */

        cart = (CvCARTClassifier*) cvCreateCARTClassifier( data->valcache,
                        flags,
                        weakTrainVals, 0, 0, 0, trimmedIdx,
                        &(data->weights),
                        (CvClassifierTrainParams*) &trainParams );

        classifier = (CvCARTHaarClassifier*) icvCreateCARTHaarClassifier( numsplits );
        icvInitCARTHaarClassifier( classifier, cart, haarFeatures );

        num_splits += classifier->count;

        cart->release( (CvClassifier**) &cart );
        
        if( symmetric && (seq->total % 2) )
        {
            float normfactor = 0.0F;
            CvStumpClassifier* stump;
            
            /* flip haar features */
            for( i = 0; i < classifier->count; i++ )
            {
                if( classifier->feature[i].desc[0] == 'h' )
                {
                    for( j = 0; j < CV_HAAR_FEATURE_MAX &&
                                    classifier->feature[i].rect[j].weight != 0.0F; j++ )
                    {
                        classifier->feature[i].rect[j].r.x = data->winsize.width - 
                            classifier->feature[i].rect[j].r.x -
                            classifier->feature[i].rect[j].r.width;                
                    }
                }
                else
                {
                    int tmp = 0;

                    /* (x,y) -> (24-x,y) */
                    /* w -> h; h -> w    */
                    for( j = 0; j < CV_HAAR_FEATURE_MAX &&
                                    classifier->feature[i].rect[j].weight != 0.0F; j++ )
                    {
                        classifier->feature[i].rect[j].r.x = data->winsize.width - 
                            classifier->feature[i].rect[j].r.x;
                        CV_SWAP( classifier->feature[i].rect[j].r.width,
                                 classifier->feature[i].rect[j].r.height, tmp );
                    }
                }
            }
            icvConvertToFastHaarFeature( classifier->feature,
                                         classifier->fastfeature,
                                         classifier->count, data->winsize.width + 1 );

            stumpTrainParams.getTrainData = NULL;
            stumpTrainParams.numcomp = 1;
            stumpTrainParams.userdata = NULL;
            stumpTrainParams.sortedIdx = NULL;

            for( i = 0; i < classifier->count; i++ )
            {
                for( j = 0; j < numtrimmed; j++ )
                {
                    idx = icvGetIdxAt( trimmedIdx, j );

                    eval.data.fl[idx] = cvEvalFastHaarFeature( &classifier->fastfeature[i],
                        (sum_type*) (data->sum.data.ptr + idx * data->sum.step),
                        (sum_type*) (data->tilted.data.ptr + idx * data->tilted.step) );
                    normfactor = data->normfactor.data.fl[idx];
                    eval.data.fl[idx] = ( normfactor == 0.0F )
                        ? 0.0F : (eval.data.fl[idx] / normfactor);
                }

                stump = (CvStumpClassifier*) trainParams.stumpConstructor( &eval,
                    CV_COL_SAMPLE,
                    weakTrainVals, 0, 0, 0, trimmedIdx,
                    &(data->weights),
                    trainParams.stumpTrainParams );
            
                classifier->threshold[i] = stump->threshold;
                if( classifier->left[i] <= 0 )
                {
                    classifier->val[-classifier->left[i]] = stump->left;
                }
                if( classifier->right[i] <= 0 )
                {
                    classifier->val[-classifier->right[i]] = stump->right;
                }

                stump->release( (CvClassifier**) &stump );        
                
            }

            stumpTrainParams.getTrainData = icvGetTrainingDataCallback;
            stumpTrainParams.numcomp = n;
            stumpTrainParams.userdata = &userdata;
            stumpTrainParams.sortedIdx = data->idxcache;

#ifdef CV_VERBOSE
            v_flipped = 1;
#endif /* CV_VERBOSE */

        } /* if symmetric */
        if( trimmedIdx != sampleIdx )
        {
            cvReleaseMat( &trimmedIdx );
            trimmedIdx = NULL;
        }
        
        for( i = 0; i < numsamples; i++ )
        {
            idx = icvGetIdxAt( sampleIdx, i );

            eval.data.fl[idx] = classifier->eval( (CvIntHaarClassifier*) classifier,
                (sum_type*) (data->sum.data.ptr + idx * data->sum.step),
                (sum_type*) (data->tilted.data.ptr + idx * data->tilted.step),
                data->normfactor.data.fl[idx] );
        }

        alpha = cvBoostNextWeakClassifier( &eval, &data->cls, weakTrainVals,
                                           &data->weights, trainer );
        sumalpha += alpha;
        
        for( i = 0; i <= classifier->count; i++ )
        {
            if( boosttype == CV_RABCLASS ) 
            {
                classifier->val[i] = cvLogRatio( classifier->val[i] );
            }
            classifier->val[i] *= alpha;
        }

        cvSeqPush( seq, (void*) &classifier );

        numpos = 0;
        for( i = 0; i < numsamples; i++ )
        {
            idx = icvGetIdxAt( sampleIdx, i );

            if( data->cls.data.fl[idx] == 1.0F )
            {
                eval.data.fl[numpos] = 0.0F;
                for( j = 0; j < seq->total; j++ )
                {
                    classifier = *((CvCARTHaarClassifier**) cvGetSeqElem( seq, j ));
                    eval.data.fl[numpos] += classifier->eval( 
                        (CvIntHaarClassifier*) classifier,
                        (sum_type*) (data->sum.data.ptr + idx * data->sum.step),
                        (sum_type*) (data->tilted.data.ptr + idx * data->tilted.step),
                        data->normfactor.data.fl[idx] );
                }
                /* eval.data.fl[numpos] = 2.0F * eval.data.fl[numpos] - seq->total; */
                numpos++;
            }
        }
        icvSort_32f( eval.data.fl, numpos, 0 );
        threshold = eval.data.fl[(int) ((1.0F - minhitrate) * numpos)];

        numneg = 0;
        numfalse = 0;
        for( i = 0; i < numsamples; i++ )
        {
            idx = icvGetIdxAt( sampleIdx, i );

            if( data->cls.data.fl[idx] == 0.0F )
            {
                numneg++;
                sum_stage = 0.0F;
                for( j = 0; j < seq->total; j++ )
                {
                   classifier = *((CvCARTHaarClassifier**) cvGetSeqElem( seq, j ));
                   sum_stage += classifier->eval( (CvIntHaarClassifier*) classifier,
                        (sum_type*) (data->sum.data.ptr + idx * data->sum.step),
                        (sum_type*) (data->tilted.data.ptr + idx * data->tilted.step),
                        data->normfactor.data.fl[idx] );
                }
                /* sum_stage = 2.0F * sum_stage - seq->total; */
                if( sum_stage >= (threshold - CV_THRESHOLD_EPS) )
                {
                    numfalse++;
                }
            }
        }
        falsealarm = ((float) numfalse) / ((float) numneg);

#ifdef CV_VERBOSE
        {
            float v_hitrate    = 0.0F;
            float v_falsealarm = 0.0F;
            /* expected error of stage classifier regardless threshold */
            float v_experr = 0.0F;

            for( i = 0; i < numsamples; i++ )
            {
                idx = icvGetIdxAt( sampleIdx, i );

                sum_stage = 0.0F;
                for( j = 0; j < seq->total; j++ )
                {
                    classifier = *((CvCARTHaarClassifier**) cvGetSeqElem( seq, j ));
                    sum_stage += classifier->eval( (CvIntHaarClassifier*) classifier,
                        (sum_type*) (data->sum.data.ptr + idx * data->sum.step),
                        (sum_type*) (data->tilted.data.ptr + idx * data->tilted.step),
                        data->normfactor.data.fl[idx] );
                }
                /* sum_stage = 2.0F * sum_stage - seq->total; */
                if( sum_stage >= (threshold - CV_THRESHOLD_EPS) )
                {
                    if( data->cls.data.fl[idx] == 1.0F )
                    {
                        v_hitrate += 1.0F;
                    }
                    else
                    {
                        v_falsealarm += 1.0F;
                    }
                }
                if( ( sum_stage >= 0.0F ) != (data->cls.data.fl[idx] == 1.0F) )
                {
                    v_experr += 1.0F;
                }
            }
            v_experr /= numsamples;
            printf( "|%4d|%3d%%|%c|%9f|%9f|%9f|%9f|\n",
                seq->total, v_wt, ( (v_flipped) ? '+' : '-' ),
                threshold, v_hitrate / numpos, v_falsealarm / numneg,
                v_experr );
            printf( "+----+----+-+---------+---------+---------+---------+\n" );
            fflush( stdout );
        }
#endif /* CV_VERBOSE */
        
    } while( falsealarm > maxfalsealarm && (!maxsplits || (num_splits < maxsplits) ) );
    cvBoostEndTraining( &trainer );

    if( falsealarm > maxfalsealarm )
    {
        stage = NULL;
    }
    else
    {
        stage = (CvStageHaarClassifier*) icvCreateStageHaarClassifier( seq->total,
                                                                       threshold );
        cvCvtSeqToArray( seq, (CvArr*) stage->classifier );
    }
    
    /* CLEANUP */
    cvReleaseMemStorage( &storage );
    cvReleaseMat( &weakTrainVals );
    cvFree( &(eval.data.ptr) );
    
    return (CvIntHaarClassifier*) stage;
}


static
CvBackgroundData* icvCreateBackgroundData( const char* filename, CvSize winsize )
{
    CvBackgroundData* data = NULL;

    const char* dir = NULL;    
    char full[PATH_MAX];
    char* imgfilename = NULL;
    size_t datasize = 0;
    int    count = 0;
    FILE*  input = NULL;
    char*  tmp   = NULL;
    int    len   = 0;

    assert( filename != NULL );
    
    dir = strrchr( filename, '\\' );
    if( dir == NULL )
    {
        dir = strrchr( filename, '/' );
    }
    if( dir == NULL )
    {
        imgfilename = &(full[0]);
    }
    else
    {
        strncpy( &(full[0]), filename, (dir - filename + 1) );
        imgfilename = &(full[(dir - filename + 1)]);
    }

    input = fopen( filename, "r" );
    if( input != NULL )
    {
        count = 0;
        datasize = 0;
        
        /* count */
        while( !feof( input ) )
        {
            *imgfilename = '\0';
            if( !fgets( imgfilename, PATH_MAX - (int)(imgfilename - full) - 1, input ))
                break;
            len = (int)strlen( imgfilename );
            for( ; len > 0 && isspace(imgfilename[len-1]); len-- )
                imgfilename[len-1] = '\0';
            if( len > 0 )
            {
                if( (*imgfilename) == '#' ) continue; /* comment */
                count++;
                datasize += sizeof( char ) * (strlen( &(full[0]) ) + 1);
            }
        }
        if( count > 0 )
        {
            //rewind( input );
            fseek( input, 0, SEEK_SET );
            datasize += sizeof( *data ) + sizeof( char* ) * count;
            data = (CvBackgroundData*) cvAlloc( datasize );
            memset( (void*) data, 0, datasize );
            data->count = count;
            data->filename = (char**) (data + 1);
            data->last = 0;
            data->round = 0;
            data->winsize = winsize;
            tmp = (char*) (data->filename + data->count);
            count = 0;
            while( !feof( input ) )
            {
                *imgfilename = '\0';
				if( !fgets( imgfilename, PATH_MAX - (int)(imgfilename - full) - 1, input ))
                    break;
                len = (int)strlen( imgfilename );
				if( len > 0 && imgfilename[len-1] == '\n' )
					imgfilename[len-1] = 0, len--;
                if( len > 0 )
                {
                    if( (*imgfilename) == '#' ) continue; /* comment */
                    data->filename[count++] = tmp;
                    strcpy( tmp, &(full[0]) );
                    tmp += strlen( &(full[0]) ) + 1;
                }
            }
        }
        fclose( input );
    }

    return data;
}

static
void icvReleaseBackgroundData( CvBackgroundData** data )
{
    assert( data != NULL && (*data) != NULL );

    cvFree( data );
}

static
CvBackgroundReader* icvCreateBackgroundReader()
{
    CvBackgroundReader* reader = NULL;

    reader = (CvBackgroundReader*) cvAlloc( sizeof( *reader ) );
    memset( (void*) reader, 0, sizeof( *reader ) );
    reader->src = cvMat( 0, 0, CV_8UC1, NULL );
    reader->img = cvMat( 0, 0, CV_8UC1, NULL );
    reader->offset = cvPoint( 0, 0 );
    reader->scale       = 1.0F;
    reader->scalefactor = 1.4142135623730950488016887242097F;
    reader->stepfactor  = 0.5F;
    reader->point = reader->offset;

    return reader;
}

static
void icvReleaseBackgroundReader( CvBackgroundReader** reader )
{
    assert( reader != NULL && (*reader) != NULL );

    if( (*reader)->src.data.ptr != NULL )
    {
        cvFree( &((*reader)->src.data.ptr) );
    }
    if( (*reader)->img.data.ptr != NULL )
    {
        cvFree( &((*reader)->img.data.ptr) );
    }

    cvFree( reader );
}

static
void icvGetNextFromBackgroundData( CvBackgroundData* data,
                                   CvBackgroundReader* reader )
{
    IplImage* img = NULL;
    size_t datasize = 0;
    int round = 0;
    int i = 0;
    CvPoint offset = cvPoint(0,0);

    assert( data != NULL && reader != NULL );

    if( reader->src.data.ptr != NULL )
    {
        cvFree( &(reader->src.data.ptr) );
        reader->src.data.ptr = NULL;
    }
    if( reader->img.data.ptr != NULL )
    {
        cvFree( &(reader->img.data.ptr) );
        reader->img.data.ptr = NULL;
    }

    #ifdef CV_OPENMP
    #pragma omp critical(c_background_data)
    #endif /* CV_OPENMP */
    {
        for( i = 0; i < data->count; i++ )
        {
            round = data->round;

//#ifdef CV_VERBOSE 
//            printf( "Open background image: %s\n", data->filename[data->last] );
//#endif /* CV_VERBOSE */
          
            data->last = rand() % data->count;
            data->last %= data->count;
            img = cvLoadImage( data->filename[data->last], 0 );
            if( !img )				
                continue;
            data->round += data->last / data->count;
            data->round = data->round % (data->winsize.width * data->winsize.height);

            offset.x = round % data->winsize.width;
            offset.y = round / data->winsize.width;

            offset.x = MIN( offset.x, img->width - data->winsize.width );
            offset.y = MIN( offset.y, img->height - data->winsize.height );
            
            if( img != NULL && img->depth == IPL_DEPTH_8U && img->nChannels == 1 &&
                offset.x >= 0 && offset.y >= 0 )
            {
                break;
            }
            if( img != NULL )
                cvReleaseImage( &img );
            img = NULL;
        }
    }
    if( img == NULL )
    {
        /* no appropriate image */

#ifdef CV_VERBOSE
        printf( "Invalid background description file.\n" );
#endif /* CV_VERBOSE */

        assert( 0 );
        exit( 1 );
    }
    datasize = sizeof( uchar ) * img->width * img->height;
    reader->src = cvMat( img->height, img->width, CV_8UC1, (void*) cvAlloc( datasize ) );
    cvCopy( img, &reader->src, NULL );
    cvReleaseImage( &img );
    img = NULL;

    //reader->offset.x = round % data->winsize.width;
    //reader->offset.y = round / data->winsize.width;
    reader->offset = offset;
    reader->point = reader->offset;
    reader->scale = MAX(
        ((float) data->winsize.width + reader->point.x) / ((float) reader->src.cols),
        ((float) data->winsize.height + reader->point.y) / ((float) reader->src.rows) );
    
    reader->img = cvMat( (int) (reader->scale * reader->src.rows + 0.5F),
                         (int) (reader->scale * reader->src.cols + 0.5F),
                          CV_8UC1, (void*) cvAlloc( datasize ) );
    cvResize( &(reader->src), &(reader->img) );
}


/*
 * icvGetBackgroundImage
 *
 * Get an image from background
 * <img> must be allocated and have size, previously passed to icvInitBackgroundReaders
 *
 * Usage example:
 * icvInitBackgroundReaders( "bg.txt", cvSize( 24, 24 ) );
 * ...
 * #pragma omp parallel
 * {
 *     ...
 *     icvGetBackgourndImage( cvbgdata, cvbgreader, img );
 *     ...
 * }
 * ...
 * icvDestroyBackgroundReaders();
 */
static
void icvGetBackgroundImage( CvBackgroundData* data,
                            CvBackgroundReader* reader,
                            CvMat* img )
{
    CvMat mat;

    assert( data != NULL && reader != NULL && img != NULL );
    assert( CV_MAT_TYPE( img->type ) == CV_8UC1 );
    assert( img->cols == data->winsize.width );
    assert( img->rows == data->winsize.height );

    if( reader->img.data.ptr == NULL )
    {
        icvGetNextFromBackgroundData( data, reader );
    }

    mat = cvMat( data->winsize.height, data->winsize.width, CV_8UC1 );
    cvSetData( &mat, (void*) (reader->img.data.ptr + reader->point.y * reader->img.step
                              + reader->point.x * sizeof( uchar )), reader->img.step );

    cvCopy( &mat, img, 0 );
    if( (int) ( reader->point.x + (1.0F + reader->stepfactor ) * data->winsize.width )
            < reader->img.cols )
    {
        reader->point.x += (int) (reader->stepfactor * data->winsize.width);
    }
    else
    {
        reader->point.x = reader->offset.x;
        if( (int) ( reader->point.y + (1.0F + reader->stepfactor ) * data->winsize.height )
                < reader->img.rows )
        {
            reader->point.y += (int) (reader->stepfactor * data->winsize.height);
        }
        else
        {
            reader->point.y = reader->offset.y;
            reader->scale *= reader->scalefactor;
            if( reader->scale <= 1.0F )
            {
                reader->img = cvMat( (int) (reader->scale * reader->src.rows),
                                     (int) (reader->scale * reader->src.cols),
                                      CV_8UC1, (void*) (reader->img.data.ptr) );
                cvResize( &(reader->src), &(reader->img) );
            }
            else
            {
                icvGetNextFromBackgroundData( data, reader );
            }
        }
    }
}


/*
 * icvInitBackgroundReaders
 *
 * Initialize background reading process.
 * <cvbgreader> and <cvbgdata> are initialized.
 * Must be called before any usage of background
 *
 * filename - name of background description file
 * winsize  - size of images will be obtained from background
 *
 * return 1 on success, 0 otherwise.
 */
static
int icvInitBackgroundReaders( const char* filename, CvSize winsize )
{
    if( cvbgdata == NULL && filename != NULL )
    {
        cvbgdata = icvCreateBackgroundData( filename, winsize );
    }

    if( cvbgdata )
    {

        #ifdef CV_OPENMP
        #pragma omp parallel
        #endif /* CV_OPENMP */
        {
            #ifdef CV_OPENMP
            #pragma omp critical(c_create_bg_data)
            #endif /* CV_OPENMP */
            {
                if( cvbgreader == NULL )
                {
                    cvbgreader = icvCreateBackgroundReader();
                }
            }
        }

    }

    return (cvbgdata != NULL);
}


/*
 * icvDestroyBackgroundReaders
 *
 * Finish backgournd reading process
 */
static
void icvDestroyBackgroundReaders()
{
    /* release background reader in each thread */
    #ifdef CV_OPENMP
    #pragma omp parallel
    #endif /* CV_OPENMP */
    {
        #ifdef CV_OPENMP
        #pragma omp critical(c_release_bg_data)
        #endif /* CV_OPENMP */
        {
            if( cvbgreader != NULL )
            {
                icvReleaseBackgroundReader( &cvbgreader );
                cvbgreader = NULL;
            }
        }
    }

    if( cvbgdata != NULL )
    {
        icvReleaseBackgroundData( &cvbgdata );
        cvbgdata = NULL;
    }
}


/*
 * icvGetAuxImages
 *
 * Get sum, tilted, sqsum images and calculate normalization factor
 * All images must be allocated.
 */
static
void icvGetAuxImages( CvMat* img, CvMat* sum, CvMat* tilted,
                      CvMat* sqsum, float* normfactor )
{
    CvRect normrect;
    int p0, p1, p2, p3;
    sum_type   valsum   = 0;
    sqsum_type valsqsum = 0;
    double area = 0.0;
    
    cvIntegral( img, sum, sqsum, tilted );
    normrect = cvRect( 1, 1, img->cols - 2, img->rows - 2 );
    CV_SUM_OFFSETS( p0, p1, p2, p3, normrect, img->cols + 1 )
    
    area = normrect.width * normrect.height;
    valsum = ((sum_type*) (sum->data.ptr))[p0] - ((sum_type*) (sum->data.ptr))[p1]
           - ((sum_type*) (sum->data.ptr))[p2] + ((sum_type*) (sum->data.ptr))[p3];
    valsqsum = ((sqsum_type*) (sqsum->data.ptr))[p0]
             - ((sqsum_type*) (sqsum->data.ptr))[p1]
             - ((sqsum_type*) (sqsum->data.ptr))[p2]
             + ((sqsum_type*) (sqsum->data.ptr))[p3];

    /* sqrt( valsqsum / area - ( valsum / are )^2 ) * area */
    (*normfactor) = (float) sqrt( (double) (area * valsqsum - (double)valsum * valsum) );
}


/* consumed counter */
typedef uint64 ccounter_t;

#define CCOUNTER_MAX CV_BIG_UINT(0xffffffffffffffff)
#define CCOUNTER_SET_ZERO(cc) ((cc) = 0)
#define CCOUNTER_INC(cc) ( (CCOUNTER_MAX > (cc) ) ? (++(cc)) : (CCOUNTER_MAX) )
#define CCOUNTER_ADD(cc0, cc1) ( ((CCOUNTER_MAX-(cc1)) > (cc0) ) ? ((cc0) += (cc1)) : ((cc0) = CCOUNTER_MAX) )
#define CCOUNTER_DIV(cc0, cc1) ( ((cc1) == 0) ? 0 : ( ((double)(cc0))/(double)(int64)(cc1) ) )



/*
 * icvGetHaarTrainingData
 *
 * Unified method that can now be used for vec file, bg file and bg vec file
 *
 * Fill <data> with samples, passed <cascade>
 */
static
int icvGetHaarTrainingData( CvHaarTrainingData* data, int first, int count,
                            CvIntHaarClassifier* cascade,
                            CvGetHaarTrainingDataCallback callback, void* userdata,
                            int* consumed, double* acceptance_ratio )
{
    int i = 0;
    ccounter_t getcount = 0;
    ccounter_t thread_getcount = 0;
    ccounter_t consumed_count; 
    ccounter_t thread_consumed_count;
    
    /* private variables */
    CvMat img;
    CvMat sum;
    CvMat tilted;
    CvMat sqsum;
    
    sum_type* sumdata;
    sum_type* tilteddata;
    float*    normfactor;
    
    /* end private variables */
    
    assert( data != NULL );
    assert( first + count <= data->maxnum );
    assert( cascade != NULL );
    assert( callback != NULL );
    
    // if( !cvbgdata ) return 0; this check needs to be done in the callback for BG
    
    CCOUNTER_SET_ZERO(getcount);
    CCOUNTER_SET_ZERO(thread_getcount);
    CCOUNTER_SET_ZERO(consumed_count);
    CCOUNTER_SET_ZERO(thread_consumed_count);

    #ifdef CV_OPENMP
    #pragma omp parallel private(img, sum, tilted, sqsum, sumdata, tilteddata, \
                                 normfactor, thread_consumed_count, thread_getcount)
    #endif /* CV_OPENMP */
    {
        sumdata    = NULL;
        tilteddata = NULL;
        normfactor = NULL;

        CCOUNTER_SET_ZERO(thread_getcount);
        CCOUNTER_SET_ZERO(thread_consumed_count);
        int ok = 1;

        img = cvMat( data->winsize.height, data->winsize.width, CV_8UC1,
            cvAlloc( sizeof( uchar ) * data->winsize.height * data->winsize.width ) );
        sum = cvMat( data->winsize.height + 1, data->winsize.width + 1,
                     CV_SUM_MAT_TYPE, NULL );
        tilted = cvMat( data->winsize.height + 1, data->winsize.width + 1,
                        CV_SUM_MAT_TYPE, NULL );
        sqsum = cvMat( data->winsize.height + 1, data->winsize.width + 1, CV_SQSUM_MAT_TYPE,
                       cvAlloc( sizeof( sqsum_type ) * (data->winsize.height + 1)
                                                     * (data->winsize.width + 1) ) );

        #ifdef CV_OPENMP
        #pragma omp for schedule(static, 1)
        #endif /* CV_OPENMP */
        for( i = first; (i < first + count); i++ )
        {
            if( !ok )
                continue;
            for( ; ; )
            {
                ok = callback( &img, userdata );
                if( !ok )
                    break;

                CCOUNTER_INC(thread_consumed_count);

                sumdata = (sum_type*) (data->sum.data.ptr + i * data->sum.step);
                tilteddata = (sum_type*) (data->tilted.data.ptr + i * data->tilted.step);
                normfactor = data->normfactor.data.fl + i;
                sum.data.ptr = (uchar*) sumdata;
                tilted.data.ptr = (uchar*) tilteddata;
                icvGetAuxImages( &img, &sum, &tilted, &sqsum, normfactor );            
                if( cascade->eval( cascade, sumdata, tilteddata, *normfactor ) != 0.0F )
                {
                    CCOUNTER_INC(thread_getcount);
                    break;
                }
            }
            
#ifdef CV_VERBOSE
            if( (i - first) % 500 == 0 )
            {
                fprintf( stderr, "%3d%%\r", (int) ( 100.0 * (i - first) / count ) );
                fflush( stderr );
            }
#endif /* CV_VERBOSE */
        }

        cvFree( &(img.data.ptr) );
        cvFree( &(sqsum.data.ptr) );

        #ifdef CV_OPENMP
        #pragma omp critical (c_consumed_count)
        #endif /* CV_OPENMP */
        {
            /* consumed_count += thread_consumed_count; */
            CCOUNTER_ADD(getcount, thread_getcount);
            CCOUNTER_ADD(consumed_count, thread_consumed_count);
        }
    } /* omp parallel */
    
    if( consumed != NULL )
    {
        *consumed = (int)consumed_count;
    }

    if( acceptance_ratio != NULL )
    {
        /* *acceptance_ratio = ((double) count) / consumed_count; */
        *acceptance_ratio = CCOUNTER_DIV(count, consumed_count);
    }
    
    return static_cast<int>(getcount);
}

/*
 * icvGetHaarTrainingDataFromBG
 *
 * Fill <data> with background samples, passed <cascade>
 * Background reading process must be initialized before call.
 */
//static
//int icvGetHaarTrainingDataFromBG( CvHaarTrainingData* data, int first, int count,
//                                  CvIntHaarClassifier* cascade, double* acceptance_ratio )
//{
//    int i = 0;
//    ccounter_t consumed_count;
//    ccounter_t thread_consumed_count;
//
//    /* private variables */
//    CvMat img;
//    CvMat sum;
//    CvMat tilted;
//    CvMat sqsum;
//
//    sum_type* sumdata;
//    sum_type* tilteddata;
//    float*    normfactor;
//
//    /* end private variables */
//
//    assert( data != NULL );
//    assert( first + count <= data->maxnum );
//    assert( cascade != NULL );
//
//    if( !cvbgdata ) return 0;
//
//    CCOUNTER_SET_ZERO(consumed_count);
//    CCOUNTER_SET_ZERO(thread_consumed_count);
//
//    #ifdef CV_OPENMP
//    #pragma omp parallel private(img, sum, tilted, sqsum, sumdata, tilteddata,
//                                 normfactor, thread_consumed_count)
//    #endif /* CV_OPENMP */
//    {
//        sumdata    = NULL;
//        tilteddata = NULL;
//        normfactor = NULL;
//
//        CCOUNTER_SET_ZERO(thread_consumed_count);
//
//        img = cvMat( data->winsize.height, data->winsize.width, CV_8UC1,
//            cvAlloc( sizeof( uchar ) * data->winsize.height * data->winsize.width ) );
//        sum = cvMat( data->winsize.height + 1, data->winsize.width + 1,
//                     CV_SUM_MAT_TYPE, NULL );
//        tilted = cvMat( data->winsize.height + 1, data->winsize.width + 1,
//                        CV_SUM_MAT_TYPE, NULL );
//        sqsum = cvMat( data->winsize.height + 1, data->winsize.width + 1,
//                       CV_SQSUM_MAT_TYPE,
//                       cvAlloc( sizeof( sqsum_type ) * (data->winsize.height + 1)
//                                                     * (data->winsize.width + 1) ) );
//        
//        #ifdef CV_OPENMP
//        #pragma omp for schedule(static, 1)
//        #endif /* CV_OPENMP */
//        for( i = first; i < first + count; i++ )
//        {
//            for( ; ; )
//            {
//                icvGetBackgroundImage( cvbgdata, cvbgreader, &img );
//                
//                CCOUNTER_INC(thread_consumed_count);
//
//                sumdata = (sum_type*) (data->sum.data.ptr + i * data->sum.step);
//                tilteddata = (sum_type*) (data->tilted.data.ptr + i * data->tilted.step);
//                normfactor = data->normfactor.data.fl + i;
//                sum.data.ptr = (uchar*) sumdata;
//                tilted.data.ptr = (uchar*) tilteddata;
//                icvGetAuxImages( &img, &sum, &tilted, &sqsum, normfactor );            
//                if( cascade->eval( cascade, sumdata, tilteddata, *normfactor ) != 0.0F )
//                {
//                    break;
//                }
//            }
//
//#ifdef CV_VERBOSE
//            if( (i - first) % 500 == 0 )
//            {
//                fprintf( stderr, "%3d%%\r", (int) ( 100.0 * (i - first) / count ) );
//                fflush( stderr );
//            }
//#endif /* CV_VERBOSE */
//            
//        }
//
//        cvFree( &(img.data.ptr) );
//        cvFree( &(sqsum.data.ptr) );
//
//        #ifdef CV_OPENMP
//        #pragma omp critical (c_consumed_count)
//        #endif /* CV_OPENMP */
//        {
//            /* consumed_count += thread_consumed_count; */
//            CCOUNTER_ADD(consumed_count, thread_consumed_count);
//        }
//    } /* omp parallel */
//
//    if( acceptance_ratio != NULL )
//    {
//        /* *acceptance_ratio = ((double) count) / consumed_count; */
//        *acceptance_ratio = CCOUNTER_DIV(count, consumed_count);
//    }
//    
//    return count;
//}

int icvGetHaarTraininDataFromVecCallback( CvMat* img, void* userdata )
{
    uchar tmp = 0;
    int r = 0;
    int c = 0;

    assert( img->rows * img->cols == ((CvVecFile*) userdata)->vecsize );
    
    fread( &tmp, sizeof( tmp ), 1, ((CvVecFile*) userdata)->input );
    fread( ((CvVecFile*) userdata)->vector, sizeof( short ),
           ((CvVecFile*) userdata)->vecsize, ((CvVecFile*) userdata)->input );
    
    if( feof( ((CvVecFile*) userdata)->input ) || 
        (((CvVecFile*) userdata)->last)++ >= ((CvVecFile*) userdata)->count )
    {
        return 0;
    }
    
    for( r = 0; r < img->rows; r++ )
    {
        for( c = 0; c < img->cols; c++ )
        {
            CV_MAT_ELEM( *img, uchar, r, c ) = 
                (uchar) ( ((CvVecFile*) userdata)->vector[r * img->cols + c] );
        }
    }

    return 1;
}

int icvGetHaarTrainingDataFromBGCallback ( CvMat* img, void* /*userdata*/ )
{
    if (! cvbgdata)
      return 0;
    
    if (! cvbgreader)
      return 0;
    
    // just in case icvGetBackgroundImage is not thread-safe ...
    #ifdef CV_OPENMP
    #pragma omp critical (get_background_image_callback)
    #endif /* CV_OPENMP */
    {
      icvGetBackgroundImage( cvbgdata, cvbgreader, img );
    }
    
    return 1;
}

/*
 * icvGetHaarTrainingDataFromVec
 * Get training data from .vec file
 */
static
int icvGetHaarTrainingDataFromVec( CvHaarTrainingData* data, int first, int count,                                   
                                   CvIntHaarClassifier* cascade,
                                   const char* filename,
                                   int* consumed )
{
    int getcount = 0;

    CV_FUNCNAME( "icvGetHaarTrainingDataFromVec" );

    __BEGIN__;

    CvVecFile file;
    short tmp = 0;    
    
    file.input = NULL;
    if( filename ) file.input = fopen( filename, "rb" );

    if( file.input != NULL )
    {
        fread( &file.count, sizeof( file.count ), 1, file.input );
        fread( &file.vecsize, sizeof( file.vecsize ), 1, file.input );
        fread( &tmp, sizeof( tmp ), 1, file.input );
        fread( &tmp, sizeof( tmp ), 1, file.input );
        if( !feof( file.input ) )
        {
            if( file.vecsize != data->winsize.width * data->winsize.height )
            {
                fclose( file.input );
                CV_ERROR( CV_StsError, "Vec file sample size mismatch" );
            }

            file.last = 0;
            file.vector = (short*) cvAlloc( sizeof( *file.vector ) * file.vecsize );
            getcount = icvGetHaarTrainingData( data, first, count, cascade,
                icvGetHaarTraininDataFromVecCallback, &file, consumed, NULL);
            cvFree( &file.vector );
        }
        fclose( file.input );
    }

    __END__;

    return getcount;
}

/*
 * icvGetHaarTrainingDataFromBG
 *
 * Fill <data> with background samples, passed <cascade>
 * Background reading process must be initialized before call, alternaly, a file
 * name to a vec file may be passed, a NULL filename indicates old behaviour
 */
static
int icvGetHaarTrainingDataFromBG( CvHaarTrainingData* data, int first, int count,
                                  CvIntHaarClassifier* cascade, double* acceptance_ratio, const char * filename = NULL )
{
    CV_FUNCNAME( "icvGetHaarTrainingDataFromBG" );

    __BEGIN__;

    if (filename)
    {
        CvVecFile file;
        short tmp = 0;    
        
        file.input = NULL;
        if( filename ) file.input = fopen( filename, "rb" );

        if( file.input != NULL )
        {
            fread( &file.count, sizeof( file.count ), 1, file.input );
            fread( &file.vecsize, sizeof( file.vecsize ), 1, file.input );
            fread( &tmp, sizeof( tmp ), 1, file.input );
            fread( &tmp, sizeof( tmp ), 1, file.input );
            if( !feof( file.input ) )
            {
                if( file.vecsize != data->winsize.width * data->winsize.height )
                {
                    fclose( file.input );
                    CV_ERROR( CV_StsError, "Vec file sample size mismatch" );
                }

                file.last = 0;
                file.vector = (short*) cvAlloc( sizeof( *file.vector ) * file.vecsize );
                icvGetHaarTrainingData( data, first, count, cascade,
                    icvGetHaarTraininDataFromVecCallback, &file, NULL, acceptance_ratio);
                cvFree( &file.vector );
            }
            fclose( file.input );
        }
    }
    else
    {
        icvGetHaarTrainingData( data, first, count, cascade,
            icvGetHaarTrainingDataFromBGCallback, NULL, NULL, acceptance_ratio);
    }

    __END__;

    return count;
}

void cvCreateCascadeClassifier( const char* dirname,
                                const char* vecfilename,
                                const char* bgfilename, 
                                int npos, int nneg, int nstages,
                                int numprecalculated,
                                int numsplits,
                                float minhitrate, float maxfalsealarm,
                                float weightfraction,
                                int mode, int symmetric,
                                int equalweights,
                                int winwidth, int winheight,
                                int boosttype, int stumperror )
{
    CvCascadeHaarClassifier* cascade = NULL;
    CvHaarTrainingData* data = NULL;
    CvIntHaarFeatures* haar_features;
    CvSize winsize;
    int i = 0;
    int j = 0;
    int poscount = 0;
    int negcount = 0;
    int consumed = 0;
    double false_alarm = 0;
    char stagename[PATH_MAX];
    float posweight = 1.0F;
    float negweight = 1.0F;
    FILE* file;

#ifdef CV_VERBOSE
    double proctime = 0.0F;
#endif /* CV_VERBOSE */

    assert( dirname != NULL );
    assert( bgfilename != NULL );
    assert( vecfilename != NULL );
    assert( nstages > 0 );

    winsize = cvSize( winwidth, winheight );

    cascade = (CvCascadeHaarClassifier*) icvCreateCascadeHaarClassifier( nstages );
    cascade->count = 0;
    
    if( icvInitBackgroundReaders( bgfilename, winsize ) )
    {
        data = icvCreateHaarTrainingData( winsize, npos + nneg );
        haar_features = icvCreateIntHaarFeatures( winsize, mode, symmetric );

#ifdef CV_VERBOSE
        printf("Number of features used : %d\n", haar_features->count);
#endif /* CV_VERBOSE */

        for( i = 0; i < nstages; i++, cascade->count++ )
        {
            sprintf( stagename, "%s%d/%s", dirname, i, CV_STAGE_CART_FILE_NAME );
            cascade->classifier[i] = 
                icvLoadCARTStageHaarClassifier( stagename, winsize.width + 1 );

            if( !icvMkDir( stagename ) )
            {

#ifdef CV_VERBOSE
                printf( "UNABLE TO CREATE DIRECTORY: %s\n", stagename );
#endif /* CV_VERBOSE */

                break;
            }
            if( cascade->classifier[i] != NULL )
            {

#ifdef CV_VERBOSE
                printf( "STAGE: %d LOADED.\n", i );
#endif /* CV_VERBOSE */

                continue;
            }

#ifdef CV_VERBOSE
            printf( "STAGE: %d\n", i );
#endif /* CV_VERBOSE */

            poscount = icvGetHaarTrainingDataFromVec( data, 0, npos,
                (CvIntHaarClassifier*) cascade, vecfilename, &consumed );
#ifdef CV_VERBOSE
            printf( "POS: %d %d %f\n", poscount, consumed,
                    ((float) poscount) / consumed );
#endif /* CV_VERBOSE */

            if( poscount <= 0 )
            {

#ifdef CV_VERBOSE
            printf( "UNABLE TO OBTAIN POS SAMPLES\n" );
#endif /* CV_VERBOSE */

                break;
            }

#ifdef CV_VERBOSE
            proctime = -TIME( 0 );
#endif /* CV_VERBOSE */

            negcount = icvGetHaarTrainingDataFromBG( data, poscount, nneg,
                (CvIntHaarClassifier*) cascade, &false_alarm );
#ifdef CV_VERBOSE
            printf( "NEG: %d %g\n", negcount, false_alarm );
            printf( "BACKGROUND PROCESSING TIME: %.2f\n",
                (proctime + TIME( 0 )) );
#endif /* CV_VERBOSE */

            if( negcount <= 0 )
            {

#ifdef CV_VERBOSE
            printf( "UNABLE TO OBTAIN NEG SAMPLES\n" );
#endif /* CV_VERBOSE */

                break;
            }

            data->sum.rows = data->tilted.rows = poscount + negcount;
            data->normfactor.cols = data->weights.cols = data->cls.cols =
                    poscount + negcount;
        
            posweight = (equalweights) ? 1.0F / (poscount + negcount) : (0.5F / poscount);
            negweight = (equalweights) ? 1.0F / (poscount + negcount) : (0.5F / negcount);
            for( j = 0; j < poscount; j++ )
            {
                data->weights.data.fl[j] = posweight;
                data->cls.data.fl[j] = 1.0F;

            }
            for( j = poscount; j < poscount + negcount; j++ )
            {
                data->weights.data.fl[j] = negweight;
                data->cls.data.fl[j] = 0.0F;
            }

#ifdef CV_VERBOSE
            proctime = -TIME( 0 );
#endif /* CV_VERBOSE */

            icvPrecalculate( data, haar_features, numprecalculated );

#ifdef CV_VERBOSE
            printf( "PRECALCULATION TIME: %.2f\n", (proctime + TIME( 0 )) );
#endif /* CV_VERBOSE */

#ifdef CV_VERBOSE
            proctime = -TIME( 0 );
#endif /* CV_VERBOSE */

            cascade->classifier[i] = icvCreateCARTStageClassifier(  data, NULL,
                haar_features, minhitrate, maxfalsealarm, symmetric, weightfraction,
                numsplits, (CvBoostType) boosttype, (CvStumpError) stumperror, 0 );

#ifdef CV_VERBOSE
            printf( "STAGE TRAINING TIME: %.2f\n", (proctime + TIME( 0 )) );
#endif /* CV_VERBOSE */

            file = fopen( stagename, "w" );
            if( file != NULL )
            {
                cascade->classifier[i]->save( 
                    (CvIntHaarClassifier*) cascade->classifier[i], file );
                fclose( file );
            }
            else
            {

#ifdef CV_VERBOSE
                printf( "FAILED TO SAVE STAGE CLASSIFIER IN FILE %s\n", stagename );
#endif /* CV_VERBOSE */

            }

        }
        icvReleaseIntHaarFeatures( &haar_features );
        icvReleaseHaarTrainingData( &data );

        if( i == nstages )
        {
            char xml_path[1024];
            int len = (int)strlen(dirname);
            CvHaarClassifierCascade* cascade = 0;
            strcpy( xml_path, dirname );
            if( xml_path[len-1] == '\\' || xml_path[len-1] == '/' )
                len--;
            strcpy( xml_path + len, ".xml" );
            cascade = cvLoadHaarClassifierCascade( dirname, cvSize(winwidth,winheight) );
            if( cascade )
                cvSave( xml_path, cascade );
            cvReleaseHaarClassifierCascade( &cascade );
        }
    }
    else
    {
#ifdef CV_VERBOSE
        printf( "FAILED TO INITIALIZE BACKGROUND READERS\n" );
#endif /* CV_VERBOSE */
    }
    
    /* CLEAN UP */
    icvDestroyBackgroundReaders();
    cascade->release( (CvIntHaarClassifier**) &cascade );
}

/* tree cascade classifier */

int icvNumSplits( CvStageHaarClassifier* stage )
{
    int i;
    int num;

    num = 0;
    for( i = 0; i < stage->count; i++ )
    {
        num += ((CvCARTHaarClassifier*) stage->classifier[i])->count;
    }

    return num;
}

void icvSetNumSamples( CvHaarTrainingData* training_data, int num )
{
    assert( num <= training_data->maxnum );

    training_data->sum.rows = training_data->tilted.rows = num;
    training_data->normfactor.cols = num;
    training_data->cls.cols = training_data->weights.cols = num;
}

void icvSetWeightsAndClasses( CvHaarTrainingData* training_data,
                              int num1, float weight1, float cls1,
                              int num2, float weight2, float cls2 )
{
    int j;

    assert( num1 + num2 <= training_data->maxnum );

    for( j = 0; j < num1; j++ )
    {
        training_data->weights.data.fl[j] = weight1;
        training_data->cls.data.fl[j] = cls1;
    }
    for( j = num1; j < num1 + num2; j++ )
    {
        training_data->weights.data.fl[j] = weight2;
        training_data->cls.data.fl[j] = cls2;
    }
}

CvMat* icvGetUsedValues( CvHaarTrainingData* training_data,
                         int start, int num,
                         CvIntHaarFeatures* haar_features,
                         CvStageHaarClassifier* stage )
{
    CvMat* ptr = NULL;
    CvMat* feature_idx = NULL;

    CV_FUNCNAME( "icvGetUsedValues" );

    __BEGIN__;

    int num_splits;
    int i, j;
    int r;
    int total, last;

    num_splits = icvNumSplits( stage );

    CV_CALL( feature_idx = cvCreateMat( 1, num_splits, CV_32SC1 ) );

    total = 0;
    for( i = 0; i < stage->count; i++ )
    {
        CvCARTHaarClassifier* cart;

        cart = (CvCARTHaarClassifier*) stage->classifier[i];
        for( j = 0; j < cart->count; j++ )
        {
            feature_idx->data.i[total++] = cart->compidx[j];
        }
    }
    icvSort_32s( feature_idx->data.i, total, 0 );

    last = 0;
    for( i = 1; i < total; i++ )
    {
        if( feature_idx->data.i[i] != feature_idx->data.i[last] )
        {
            feature_idx->data.i[++last] = feature_idx->data.i[i];
        }
    }
    total = last + 1;
    CV_CALL( ptr = cvCreateMat( num, total, CV_32FC1 ) );
    

    #ifdef CV_OPENMP
    #pragma omp parallel for
    #endif
    for( r = start; r < start + num; r++ )
    {
        int c;

        for( c = 0; c < total; c++ )
        {
            float val, normfactor;
            int fnum;

            fnum = feature_idx->data.i[c];

            val = cvEvalFastHaarFeature( haar_features->fastfeature + fnum,
                (sum_type*) (training_data->sum.data.ptr
                        + r * training_data->sum.step),
                (sum_type*) (training_data->tilted.data.ptr
                        + r * training_data->tilted.step) );
            normfactor = training_data->normfactor.data.fl[r];
            val = ( normfactor == 0.0F ) ? 0.0F : (val / normfactor);
            CV_MAT_ELEM( *ptr, float, r - start, c ) = val;
        }
    }

    __END__;

    cvReleaseMat( &feature_idx );

    return ptr;
}

/* possible split in the tree */
typedef struct CvSplit
{
    CvTreeCascadeNode* parent;
    CvTreeCascadeNode* single_cluster;
    CvTreeCascadeNode* multiple_clusters;
    int num_clusters;
    float single_multiple_ratio;

    struct CvSplit* next;
} CvSplit;


void cvCreateTreeCascadeClassifier( const char* dirname,
                                    const char* vecfilename,
                                    const char* bgfilename, 
                                    int npos, int nneg, int nstages,
                                    int numprecalculated,
                                    int numsplits,
                                    float minhitrate, float maxfalsealarm,
                                    float weightfraction,
                                    int mode, int symmetric,
                                    int equalweights,
                                    int winwidth, int winheight,
                                    int boosttype, int stumperror,
                                    int maxtreesplits, int minpos, bool bg_vecfile )
{
    CvTreeCascadeClassifier* tcc = NULL;
    CvIntHaarFeatures* haar_features = NULL;
    CvHaarTrainingData* training_data = NULL;
    CvMat* vals = NULL;
    CvMat* cluster_idx = NULL;
    CvMat* idx = NULL;
    CvMat* features_idx = NULL;

    CV_FUNCNAME( "cvCreateTreeCascadeClassifier" );

    __BEGIN__;

    int i, k;
    CvTreeCascadeNode* leaves;
    int best_num, cur_num;
    CvSize winsize;
    char stage_name[PATH_MAX];
    char buf[PATH_MAX];
    char* suffix;
    int total_splits;

    int poscount;
    int negcount;
    int consumed;
    double false_alarm;
    double proctime;

    int nleaves;
    double required_leaf_fa_rate;
    float neg_ratio;

    int max_clusters;

    max_clusters = CV_MAX_CLUSTERS;
    neg_ratio = (float) nneg / npos;

    nleaves = 1 + MAX( 0, maxtreesplits );
    required_leaf_fa_rate = pow( (double) maxfalsealarm, (double) nstages ) / nleaves;

    printf( "Required leaf false alarm rate: %g\n", required_leaf_fa_rate );

    total_splits = 0;

    winsize = cvSize( winwidth, winheight );

    CV_CALL( cluster_idx = cvCreateMat( 1, npos + nneg, CV_32SC1 ) );
    CV_CALL( idx = cvCreateMat( 1, npos + nneg, CV_32SC1 ) );

    CV_CALL( tcc = (CvTreeCascadeClassifier*)
        icvLoadTreeCascadeClassifier( dirname, winwidth + 1, &total_splits ) );
    CV_CALL( leaves = icvFindDeepestLeaves( tcc ) );

    CV_CALL( icvPrintTreeCascade( tcc->root ) );

    haar_features = icvCreateIntHaarFeatures( winsize, mode, symmetric );

    printf( "Number of features used : %d\n", haar_features->count );

    training_data = icvCreateHaarTrainingData( winsize, npos + nneg );

    sprintf( stage_name, "%s/", dirname );
    suffix = stage_name + strlen( stage_name );
    
    if (! bg_vecfile)
      if( !icvInitBackgroundReaders( bgfilename, winsize ) && nstages > 0 )
          CV_ERROR( CV_StsError, "Unable to read negative images" );
    
    if( nstages > 0 )
    {
        /* width-first search in the tree */
        do
        {
            CvSplit* first_split;
            CvSplit* last_split;
            CvSplit* cur_split;
            
            CvTreeCascadeNode* parent;
            CvTreeCascadeNode* cur_node;
            CvTreeCascadeNode* last_node;

            first_split = last_split = cur_split = NULL;
            parent = leaves;
            leaves = NULL;
            do
            {                
                int best_clusters; /* best selected number of clusters */
                float posweight, negweight;
                double leaf_fa_rate;

                if( parent ) sprintf( buf, "%d", parent->idx );
                else sprintf( buf, "NULL" );
                printf( "\nParent node: %s\n\n", buf );

                printf( "*** 1 cluster ***\n" );

                tcc->eval = icvEvalTreeCascadeClassifierFilter;
                /* find path from the root to the node <parent> */
                icvSetLeafNode( tcc, parent );

                /* load samples */
                consumed = 0;
                poscount = icvGetHaarTrainingDataFromVec( training_data, 0, npos,
                    (CvIntHaarClassifier*) tcc, vecfilename, &consumed );

                printf( "POS: %d %d %f\n", poscount, consumed, ((double) poscount)/consumed );

                if( poscount <= 0 )
                    CV_ERROR( CV_StsError, "Unable to obtain positive samples" );

                fflush( stdout );

                proctime = -TIME( 0 );

                nneg = (int) (neg_ratio * poscount);
                negcount = icvGetHaarTrainingDataFromBG( training_data, poscount, nneg,
                    (CvIntHaarClassifier*) tcc, &false_alarm, bg_vecfile ? bgfilename : NULL );
                printf( "NEG: %d %g\n", negcount, false_alarm );

                printf( "BACKGROUND PROCESSING TIME: %.2f\n", (proctime + TIME( 0 )) );

                if( negcount <= 0 )
                    CV_ERROR( CV_StsError, "Unable to obtain negative samples" );

                leaf_fa_rate = false_alarm;
                if( leaf_fa_rate <= required_leaf_fa_rate )
                {
                    printf( "Required leaf false alarm rate achieved. "
                            "Branch training terminated.\n" );
                }
                else if( nleaves == 1 && tcc->next_idx == nstages )
                {
                    printf( "Required number of stages achieved. "
                            "Branch training terminated.\n" );
                }
                else
                {
                    CvTreeCascadeNode* single_cluster;
                    CvTreeCascadeNode* multiple_clusters;
                    CvSplit* cur_split;
                    int single_num;

                    icvSetNumSamples( training_data, poscount + negcount );
                    posweight = (equalweights) ? 1.0F / (poscount + negcount) : (0.5F/poscount);
                    negweight = (equalweights) ? 1.0F / (poscount + negcount) : (0.5F/negcount);
                    icvSetWeightsAndClasses( training_data,
                        poscount, posweight, 1.0F, negcount, negweight, 0.0F );

                    fflush( stdout );

                    /* precalculate feature values */
                    proctime = -TIME( 0 );
                    icvPrecalculate( training_data, haar_features, numprecalculated );
                    printf( "Precalculation time: %.2f\n", (proctime + TIME( 0 )) );

                    /* train stage classifier using all positive samples */
                    CV_CALL( single_cluster = icvCreateTreeCascadeNode() );
                    fflush( stdout );

                    proctime = -TIME( 0 );
                    single_cluster->stage =
                        (CvStageHaarClassifier*) icvCreateCARTStageClassifier(
                            training_data, NULL, haar_features,
                            minhitrate, maxfalsealarm, symmetric,
                            weightfraction, numsplits, (CvBoostType) boosttype,
                            (CvStumpError) stumperror, 0 );
                    printf( "Stage training time: %.2f\n", (proctime + TIME( 0 )) );

                    single_num = icvNumSplits( single_cluster->stage );
                    best_num = single_num;
                    best_clusters = 1;
                    multiple_clusters = NULL;

                    printf( "Number of used features: %d\n", single_num );
                    
                    if( maxtreesplits >= 0 )
                    {
                        max_clusters = MIN( max_clusters, maxtreesplits - total_splits + 1 );
                    }

                    /* try clustering */
                    vals = NULL;
                    for( k = 2; k <= max_clusters; k++ )
                    {
                        int cluster;
                        int stop_clustering;

                        printf( "*** %d clusters ***\n", k );

                        /* check whether clusters are big enough */
                        stop_clustering = ( k * minpos > poscount );
                        if( !stop_clustering )
                        {
                            int num[CV_MAX_CLUSTERS];

                            if( k == 2 )
                            {
                                proctime = -TIME( 0 );
                                CV_CALL( vals = icvGetUsedValues( training_data, 0, poscount,
                                    haar_features, single_cluster->stage ) );
                                printf( "Getting values for clustering time: %.2f\n", (proctime + TIME(0)) );
                                printf( "Value matirx size: %d x %d\n", vals->rows, vals->cols );
                                fflush( stdout );

                                cluster_idx->cols = vals->rows;
                                for( i = 0; i < negcount; i++ ) idx->data.i[i] = poscount + i;
                            }

                            proctime = -TIME( 0 );

                            CV_CALL( cvKMeans2( vals, k, cluster_idx, CV_TERM_CRITERIA() ) );

                            printf( "Clustering time: %.2f\n", (proctime + TIME( 0 )) );

                            for( cluster = 0; cluster < k; cluster++ ) num[cluster] = 0;
                            for( i = 0; i < cluster_idx->cols; i++ )
                                num[cluster_idx->data.i[i]]++;
                            for( cluster = 0; cluster < k; cluster++ )
                            {
                                if( num[cluster] < minpos )
                                {
                                    stop_clustering = 1;
                                    break;
                                }
                            }
                        }

                        if( stop_clustering )
                        {
                            printf( "Clusters are too small. Clustering aborted.\n" );
                            break;
                        }
                        
                        cur_num = 0;
                        cur_node = last_node = NULL;
                        for( cluster = 0; (cluster < k) && (cur_num < best_num); cluster++ )
                        {
                            CvTreeCascadeNode* new_node;

                            int num_splits;
                            int last_pos;
                            int total_pos;

                            printf( "Cluster: %d\n", cluster );

                            last_pos = negcount;
                            for( i = 0; i < cluster_idx->cols; i++ )
                            {
                                if( cluster_idx->data.i[i] == cluster )
                                {
                                    idx->data.i[last_pos++] = i;
                                }
                            }
                            idx->cols = last_pos;

                            total_pos = idx->cols - negcount;
                            printf( "# pos: %d of %d. (%d%%)\n", total_pos, poscount,
                                100 * total_pos / poscount );

                            CV_CALL( new_node = icvCreateTreeCascadeNode() );
                            if( last_node ) last_node->next = new_node;
                            else cur_node = new_node;
                            last_node = new_node;

                            posweight = (equalweights)
                                ? 1.0F / (total_pos + negcount) : (0.5F / total_pos);
                            negweight = (equalweights)
                                ? 1.0F / (total_pos + negcount) : (0.5F / negcount);

                            icvSetWeightsAndClasses( training_data,
                                poscount, posweight, 1.0F, negcount, negweight, 0.0F );

                            /* CV_DEBUG_SAVE( idx ); */

                            fflush( stdout );

                            proctime = -TIME( 0 );
                            new_node->stage = (CvStageHaarClassifier*)
                                icvCreateCARTStageClassifier( training_data, idx, haar_features,
                                    minhitrate, maxfalsealarm, symmetric,
                                    weightfraction, numsplits, (CvBoostType) boosttype,
                                    (CvStumpError) stumperror, best_num - cur_num );
                            printf( "Stage training time: %.2f\n", (proctime + TIME( 0 )) );

                            if( !(new_node->stage) )
                            {
                                printf( "Stage training aborted.\n" );
                                cur_num = best_num + 1;
                            }
                            else
                            {
                                num_splits = icvNumSplits( new_node->stage );
                                cur_num += num_splits;

                                printf( "Number of used features: %d\n", num_splits );
                            }
                        } /* for each cluster */

                        if( cur_num < best_num )
                        {
                            icvReleaseTreeCascadeNodes( &multiple_clusters );
                            best_num = cur_num;
                            best_clusters = k;
                            multiple_clusters = cur_node;
                        }
                        else
                        {
                            icvReleaseTreeCascadeNodes( &cur_node );
                        }
                    } /* try different number of clusters */
                    cvReleaseMat( &vals );

                    CV_CALL( cur_split = (CvSplit*) cvAlloc( sizeof( *cur_split ) ) );
                    CV_ZERO_OBJ( cur_split );
                    
                    if( last_split ) last_split->next = cur_split;
                    else first_split = cur_split;
                    last_split = cur_split;

                    cur_split->single_cluster = single_cluster;
                    cur_split->multiple_clusters = multiple_clusters;
                    cur_split->num_clusters = best_clusters;
                    cur_split->parent = parent;
                    cur_split->single_multiple_ratio = (float) single_num / best_num;
                }

                if( parent ) parent = parent->next_same_level;
            } while( parent );

            /* choose which nodes should be splitted */
            do
            {
                float max_single_multiple_ratio;

                cur_split = NULL;
                max_single_multiple_ratio = 0.0F;
                last_split = first_split;
                while( last_split )
                {
                    if( last_split->single_cluster && last_split->multiple_clusters &&
                        last_split->single_multiple_ratio > max_single_multiple_ratio )
                    {
                        max_single_multiple_ratio = last_split->single_multiple_ratio;
                        cur_split = last_split;
                    }
                    last_split = last_split->next;
                }
                if( cur_split )
                {
                    if( maxtreesplits < 0 ||
                        cur_split->num_clusters <= maxtreesplits - total_splits + 1 )
                    {
                        cur_split->single_cluster = NULL;
                        total_splits += cur_split->num_clusters - 1;
                    }
                    else
                    {
                        icvReleaseTreeCascadeNodes( &(cur_split->multiple_clusters) );
                        cur_split->multiple_clusters = NULL;
                    }
                }
            } while( cur_split );

            /* attach new nodes to the tree */
            leaves = last_node = NULL;
            last_split = first_split;
            while( last_split )
            {
                cur_node = (last_split->multiple_clusters)
                    ? last_split->multiple_clusters : last_split->single_cluster;
                parent = last_split->parent;
                if( parent ) parent->child = cur_node;
                
                /* connect leaves via next_same_level and save them */
                for( ; cur_node; cur_node = cur_node->next )
                {
                    FILE* file;

                    if( last_node ) last_node->next_same_level = cur_node;
                    else leaves = cur_node;
                    last_node = cur_node;
                    cur_node->parent = parent;

                    cur_node->idx = tcc->next_idx;
                    tcc->next_idx++;
                    sprintf( suffix, "%d/%s", cur_node->idx, CV_STAGE_CART_FILE_NAME );
                    file = NULL;
                    if( icvMkDir( stage_name ) && (file = fopen( stage_name, "w" )) != 0 )
                    {
                        cur_node->stage->save( (CvIntHaarClassifier*) cur_node->stage, file );
                        fprintf( file, "\n%d\n%d\n",
                            ((parent) ? parent->idx : -1),
                            ((cur_node->next) ? tcc->next_idx : -1) );
                    }
                    else
                    {
                        printf( "Failed to save classifier into %s\n", stage_name );
                    }
                    if( file ) fclose( file );
                }

                if( parent ) sprintf( buf, "%d", parent->idx );
                else sprintf( buf, "NULL" );
                printf( "\nParent node: %s\n", buf );
                printf( "Chosen number of splits: %d\n\n", (last_split->multiple_clusters)
                    ? (last_split->num_clusters - 1) : 0 );
                
                cur_split = last_split;
                last_split = last_split->next;
                cvFree( &cur_split );
            } /* for each split point */

            printf( "Total number of splits: %d\n", total_splits );
            
            if( !(tcc->root) ) tcc->root = leaves;
            CV_CALL( icvPrintTreeCascade( tcc->root ) );

        } while( leaves );

        /* save the cascade to xml file */
        {
            char xml_path[1024];
            int len = (int)strlen(dirname);
            CvHaarClassifierCascade* cascade = 0;
            strcpy( xml_path, dirname );
            if( xml_path[len-1] == '\\' || xml_path[len-1] == '/' )
                len--;
            strcpy( xml_path + len, ".xml" );
            cascade = cvLoadHaarClassifierCascade( dirname, cvSize(winwidth,winheight) );
            if( cascade )
                cvSave( xml_path, cascade );
            cvReleaseHaarClassifierCascade( &cascade );
        }

    } /* if( nstages > 0 ) */

    /* check cascade performance */
    printf( "\nCascade performance\n" );

    tcc->eval = icvEvalTreeCascadeClassifier;

    /* load samples */
    consumed = 0;
    poscount = icvGetHaarTrainingDataFromVec( training_data, 0, npos,
        (CvIntHaarClassifier*) tcc, vecfilename, &consumed );

    printf( "POS: %d %d %f\n", poscount, consumed,
        (consumed > 0) ? (((float) poscount)/consumed) : 0 );

    if( poscount <= 0 )
        fprintf( stderr, "Warning: unable to obtain positive samples\n" );

    proctime = -TIME( 0 );

    negcount = icvGetHaarTrainingDataFromBG( training_data, poscount, nneg,
        (CvIntHaarClassifier*) tcc, &false_alarm, bg_vecfile ? bgfilename : NULL );

    printf( "NEG: %d %g\n", negcount, false_alarm );

    printf( "BACKGROUND PROCESSING TIME: %.2f\n", (proctime + TIME( 0 )) );

    if( negcount <= 0 )
        fprintf( stderr, "Warning: unable to obtain negative samples\n" );

    __END__;

    if (! bg_vecfile)
      icvDestroyBackgroundReaders();

    if( tcc ) tcc->release( (CvIntHaarClassifier**) &tcc );
    icvReleaseIntHaarFeatures( &haar_features );
    icvReleaseHaarTrainingData( &training_data );
    cvReleaseMat( &cluster_idx );
    cvReleaseMat( &idx );
    cvReleaseMat( &vals );
    cvReleaseMat( &features_idx );
}



void cvCreateTrainingSamples( const char* filename,
                              const char* imgfilename, int bgcolor, int bgthreshold,
                              const char* bgfilename, int count,
                              int invert, int maxintensitydev,
                              double maxxangle, double maxyangle, double maxzangle,
                              int showsamples,
                              int winwidth, int winheight )
{
    CvSampleDistortionData data;

    assert( filename != NULL );
    assert( imgfilename != NULL );

    if( !icvMkDir( filename ) )
    {
        fprintf( stderr, "Unable to create output file: %s\n", filename );
        return;
    }
    if( icvStartSampleDistortion( imgfilename, bgcolor, bgthreshold, &data ) )
    {
        FILE* output = NULL;

        output = fopen( filename, "wb" );
        if( output != NULL )
        {
            int hasbg;
            int i;
            CvMat sample;
            int inverse;

            hasbg = 0;
            hasbg = (bgfilename != NULL && icvInitBackgroundReaders( bgfilename,
                     cvSize( winwidth,winheight ) ) );

            sample = cvMat( winheight, winwidth, CV_8UC1, cvAlloc( sizeof( uchar ) *
                            winheight * winwidth ) );

            icvWriteVecHeader( output, count, sample.cols, sample.rows );

            if( showsamples )
            {
                cvNamedWindow( "Sample", CV_WINDOW_AUTOSIZE );
            }

            inverse = invert;
            for( i = 0; i < count; i++ )
            {
                if( hasbg )
                {
                    icvGetBackgroundImage( cvbgdata, cvbgreader, &sample );
                }
                else
                {
                    cvSet( &sample, cvScalar( bgcolor ) );
                }

                if( invert == CV_RANDOM_INVERT )
                {
                    inverse = (rand() > (RAND_MAX/2));
                }
                icvPlaceDistortedSample( &sample, inverse, maxintensitydev,
                    maxxangle, maxyangle, maxzangle, 
                    0   /* nonzero means placing image without cut offs */,
                    0.0 /* nozero adds random shifting                  */,
                    0.0 /* nozero adds random scaling                   */,
                    &data );

                if( showsamples )
                {
                    cvShowImage( "Sample", &sample );
                    if( cvWaitKey( 0 ) == 27 )
                    {
                        showsamples = 0;
                    }
                }

                icvWriteVecSample( output, &sample );

#ifdef CV_VERBOSE
                if( i % 500 == 0 )
                {
                    printf( "\r%3d%%", 100 * i / count );
                }
#endif /* CV_VERBOSE */
            }
            icvDestroyBackgroundReaders();
            cvFree( &(sample.data.ptr) );
            fclose( output );
        } /* if( output != NULL ) */
        
        icvEndSampleDistortion( &data );
    }
    
#ifdef CV_VERBOSE
    printf( "\r      \r" );
#endif /* CV_VERBOSE */ 

}

#define CV_INFO_FILENAME "info.dat"


void cvCreateTestSamples( const char* infoname,
                          const char* imgfilename, int bgcolor, int bgthreshold,
                          const char* bgfilename, int count,
                          int invert, int maxintensitydev,
                          double maxxangle, double maxyangle, double maxzangle,
                          int showsamples,
                          int winwidth, int winheight )
{
    CvSampleDistortionData data;

    assert( infoname != NULL );
    assert( imgfilename != NULL );
    assert( bgfilename != NULL );

    if( !icvMkDir( infoname ) )
    {

#if CV_VERBOSE
        fprintf( stderr, "Unable to create directory hierarchy: %s\n", infoname );
#endif /* CV_VERBOSE */

        return;
    }
    if( icvStartSampleDistortion( imgfilename, bgcolor, bgthreshold, &data ) )
    {
        char fullname[PATH_MAX];
        char* filename;
        CvMat win;
        FILE* info;

        if( icvInitBackgroundReaders( bgfilename, cvSize( 10, 10 ) ) )
        {
            int i;
            int x, y, width, height;
            float scale;
            float maxscale;
            int inverse;

            if( showsamples )
            {
                cvNamedWindow( "Image", CV_WINDOW_AUTOSIZE );
            }
            
            info = fopen( infoname, "w" );
            strcpy( fullname, infoname );
            filename = strrchr( fullname, '\\' );
            if( filename == NULL )
            {
                filename = strrchr( fullname, '/' );
            }
            if( filename == NULL )
            {
                filename = fullname;
            }
            else
            {
                filename++;
            }

            count = MIN( count, cvbgdata->count );
            inverse = invert;
            for( i = 0; i < count; i++ )
            {
                icvGetNextFromBackgroundData( cvbgdata, cvbgreader );
                
                maxscale = MIN( 0.7F * cvbgreader->src.cols / winwidth,
                                   0.7F * cvbgreader->src.rows / winheight );
                if( maxscale < 1.0F ) continue;

                scale = (maxscale - 1.0F) * rand() / RAND_MAX + 1.0F;
                width = (int) (scale * winwidth);
                height = (int) (scale * winheight);
                x = (int) ((0.1+0.8 * rand()/RAND_MAX) * (cvbgreader->src.cols - width));
                y = (int) ((0.1+0.8 * rand()/RAND_MAX) * (cvbgreader->src.rows - height));

                cvGetSubArr( &cvbgreader->src, &win, cvRect( x, y ,width, height ) );
                if( invert == CV_RANDOM_INVERT )
                {
                    inverse = (rand() > (RAND_MAX/2));
                }
                icvPlaceDistortedSample( &win, inverse, maxintensitydev,
                                         maxxangle, maxyangle, maxzangle, 
                                         1, 0.0, 0.0, &data );
                
                
                sprintf( filename, "%04d_%04d_%04d_%04d_%04d.jpg",
                         (i + 1), x, y, width, height );
                
                if( info ) 
                {
                    fprintf( info, "%s %d %d %d %d %d\n",
                        filename, 1, x, y, width, height );
                }

                cvSaveImage( fullname, &cvbgreader->src );
                if( showsamples )
                {
                    cvShowImage( "Image", &cvbgreader->src );
                    if( cvWaitKey( 0 ) == 27 )
                    {
                        showsamples = 0;
                    }
                }
            }
            if( info ) fclose( info );
            icvDestroyBackgroundReaders();
        }
        icvEndSampleDistortion( &data );
    }
}


/* End of file. */
