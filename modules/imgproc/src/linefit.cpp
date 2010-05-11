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

static const double eps = 1e-6;

static CvStatus
icvFitLine2D_wods( CvPoint2D32f * points, int _count, float *weights, float *line )
{
    double x = 0, y = 0, x2 = 0, y2 = 0, xy = 0, w = 0;
    double dx2, dy2, dxy;
    int i;
    int count = _count;
    float t;

    /* Calculating the average of x and y... */

    if( weights == 0 )
    {
        for( i = 0; i < count; i += 1 )
        {
            x += points[i].x;
            y += points[i].y;
            x2 += points[i].x * points[i].x;
            y2 += points[i].y * points[i].y;
            xy += points[i].x * points[i].y;
        }
        w = (float) count;
    }
    else
    {
        for( i = 0; i < count; i += 1 )
        {
            x += weights[i] * points[i].x;
            y += weights[i] * points[i].y;
            x2 += weights[i] * points[i].x * points[i].x;
            y2 += weights[i] * points[i].y * points[i].y;
            xy += weights[i] * points[i].x * points[i].y;
            w += weights[i];
        }
    }

    x /= w;
    y /= w;
    x2 /= w;
    y2 /= w;
    xy /= w;

    dx2 = x2 - x * x;
    dy2 = y2 - y * y;
    dxy = xy - x * y;

    t = (float) atan2( 2 * dxy, dx2 - dy2 ) / 2;
    line[0] = (float) cos( t );
    line[1] = (float) sin( t );

    line[2] = (float) x;
    line[3] = (float) y;

    return CV_NO_ERR;
}

static CvStatus
icvFitLine3D_wods( CvPoint3D32f * points, int count, float *weights, float *line )
{
    int i;
    float w0 = 0;
    float x0 = 0, y0 = 0, z0 = 0;
    float x2 = 0, y2 = 0, z2 = 0, xy = 0, yz = 0, xz = 0;
    float dx2, dy2, dz2, dxy, dxz, dyz;
    float *v;
    float n;
    float det[9], evc[9], evl[3];

    memset( evl, 0, 3*sizeof(evl[0]));
    memset( evc, 0, 9*sizeof(evl[0]));

    if( weights )
    {
        for( i = 0; i < count; i++ )
        {
            float x = points[i].x;
            float y = points[i].y;
            float z = points[i].z;
            float w = weights[i];


            x2 += x * x * w;
            xy += x * y * w;
            xz += x * z * w;
            y2 += y * y * w;
            yz += y * z * w;
            z2 += z * z * w;
            x0 += x * w;
            y0 += y * w;
            z0 += z * w;
            w0 += w;
        }
    }
    else
    {
        for( i = 0; i < count; i++ )
        {
            float x = points[i].x;
            float y = points[i].y;
            float z = points[i].z;

            x2 += x * x;
            xy += x * y;
            xz += x * z;
            y2 += y * y;
            yz += y * z;
            z2 += z * z;
            x0 += x;
            y0 += y;
            z0 += z;
        }
        w0 = (float) count;
    }

    x2 /= w0;
    xy /= w0;
    xz /= w0;
    y2 /= w0;
    yz /= w0;
    z2 /= w0;

    x0 /= w0;
    y0 /= w0;
    z0 /= w0;

    dx2 = x2 - x0 * x0;
    dxy = xy - x0 * y0;
    dxz = xz - x0 * z0;
    dy2 = y2 - y0 * y0;
    dyz = yz - y0 * z0;
    dz2 = z2 - z0 * z0;

    det[0] = dz2 + dy2;
    det[1] = -dxy;
    det[2] = -dxz;
    det[3] = det[1];
    det[4] = dx2 + dz2;
    det[5] = -dyz;
    det[6] = det[2];
    det[7] = det[5];
    det[8] = dy2 + dx2;

    /* Searching for a eigenvector of det corresponding to the minimal eigenvalue */
#if 1
    {
    CvMat _det = cvMat( 3, 3, CV_32F, det );
    CvMat _evc = cvMat( 3, 3, CV_32F, evc );
    CvMat _evl = cvMat( 3, 1, CV_32F, evl );
    cvEigenVV( &_det, &_evc, &_evl, 0 );
    i = evl[0] < evl[1] ? (evl[0] < evl[2] ? 0 : 2) : (evl[1] < evl[2] ? 1 : 2);
    }
#else
    {
        CvMat _det = cvMat( 3, 3, CV_32F, det );
        CvMat _evc = cvMat( 3, 3, CV_32F, evc );
        CvMat _evl = cvMat( 1, 3, CV_32F, evl );

        cvSVD( &_det, &_evl, &_evc, 0, CV_SVD_MODIFY_A+CV_SVD_U_T );
    }
    i = 2;
#endif
    v = &evc[i * 3];
    n = (float) sqrt( (double)v[0] * v[0] + (double)v[1] * v[1] + (double)v[2] * v[2] );
    n = (float)MAX(n, eps);
    line[0] = v[0] / n;
    line[1] = v[1] / n;
    line[2] = v[2] / n;
    line[3] = x0;
    line[4] = y0;
    line[5] = z0;

    return CV_NO_ERR;
}

static double
icvCalcDist2D( CvPoint2D32f * points, int count, float *_line, float *dist )
{
    int j;
    float px = _line[2], py = _line[3];
    float nx = _line[1], ny = -_line[0];
    double sum_dist = 0.;

    for( j = 0; j < count; j++ )
    {
        float x, y;

        x = points[j].x - px;
        y = points[j].y - py;

        dist[j] = (float) fabs( nx * x + ny * y );
        sum_dist += dist[j];
    }

    return sum_dist;
}

static double
icvCalcDist3D( CvPoint3D32f * points, int count, float *_line, float *dist )
{
    int j;
    float px = _line[3], py = _line[4], pz = _line[5];
    float vx = _line[0], vy = _line[1], vz = _line[2];
    double sum_dist = 0.;

    for( j = 0; j < count; j++ )
    {
        float x, y, z;
        double p1, p2, p3;

        x = points[j].x - px;
        y = points[j].y - py;
        z = points[j].z - pz;

        p1 = vy * z - vz * y;
        p2 = vz * x - vx * z;
        p3 = vx * y - vy * x;

        dist[j] = (float) sqrt( p1*p1 + p2*p2 + p3*p3 );
        sum_dist += dist[j];
    }

    return sum_dist;
}

static void
icvWeightL1( float *d, int count, float *w )
{
    int i;

    for( i = 0; i < count; i++ )
    {
        double t = fabs( (double) d[i] );
        w[i] = (float)(1. / MAX(t, eps));
    }
}

static void
icvWeightL12( float *d, int count, float *w )
{
    int i;

    for( i = 0; i < count; i++ )
    {
        w[i] = 1.0f / (float) sqrt( 1 + (double) (d[i] * d[i] * 0.5) );
    }
}


static void
icvWeightHuber( float *d, int count, float *w, float _c )
{
    int i;
    const float c = _c <= 0 ? 1.345f : _c;

    for( i = 0; i < count; i++ )
    {
        if( d[i] < c )
            w[i] = 1.0f;
        else
            w[i] = c/d[i];
    }
}


static void
icvWeightFair( float *d, int count, float *w, float _c )
{
    int i;
    const float c = _c == 0 ? 1 / 1.3998f : 1 / _c;

    for( i = 0; i < count; i++ )
    {
        w[i] = 1 / (1 + d[i] * c);
    }
}

static void
icvWeightWelsch( float *d, int count, float *w, float _c )
{
    int i;
    const float c = _c == 0 ? 1 / 2.9846f : 1 / _c;

    for( i = 0; i < count; i++ )
    {
        w[i] = (float) exp( -d[i] * d[i] * c * c );
    }
}


/* Takes an array of 2D points, type of distance (including user-defined
distance specified by callbacks, fills the array of four floats with line
parameters A, B, C, D, where (A, B) is the normalized direction vector,
(C, D) is the point that belongs to the line. */

static CvStatus  icvFitLine2D( CvPoint2D32f * points, int count, int dist,
                               float _param, float reps, float aeps, float *line )
{
    double EPS = count*FLT_EPSILON;
    void (*calc_weights) (float *, int, float *) = 0;
    void (*calc_weights_param) (float *, int, float *, float) = 0;
    float *w;                   /* weights */
    float *r;                   /* square distances */
    int i, j, k;
    float _line[6], _lineprev[6];
    float rdelta = reps != 0 ? reps : 1.0f;
    float adelta = aeps != 0 ? aeps : 0.01f;
    double min_err = DBL_MAX, err = 0;
    CvRNG rng = cvRNG(-1);

    memset( line, 0, 4*sizeof(line[0]) );

    switch (dist)
    {
    case CV_DIST_L2:
        return icvFitLine2D_wods( points, count, 0, line );

    case CV_DIST_L1:
        calc_weights = icvWeightL1;
        break;

    case CV_DIST_L12:
        calc_weights = icvWeightL12;
        break;

    case CV_DIST_FAIR:
        calc_weights_param = icvWeightFair;
        break;

    case CV_DIST_WELSCH:
        calc_weights_param = icvWeightWelsch;
        break;

    case CV_DIST_HUBER:
        calc_weights_param = icvWeightHuber;
        break;

    /*case CV_DIST_USER:
        calc_weights = (void ( * )(float *, int, float *)) _PFP.fp;
        break;*/

    default:
        return CV_BADFACTOR_ERR;
    }

    w = (float *) cvAlloc( count * sizeof( float ));
    r = (float *) cvAlloc( count * sizeof( float ));

    for( k = 0; k < 20; k++ )
    {
        int first = 1;
        for( i = 0; i < count; i++ )
            w[i] = 0.f;
        
        for( i = 0; i < MIN(count,10); )
        {
            j = cvRandInt(&rng) % count;
            if( w[j] < FLT_EPSILON )
            {
                w[j] = 1.f;
                i++;
            }
        }

        icvFitLine2D_wods( points, count, w, _line );
        for( i = 0; i < 30; i++ )
        {
            double sum_w = 0;

            if( first )
            {
                first = 0;
            }
            else
            {
                double t = _line[0] * _lineprev[0] + _line[1] * _lineprev[1];
                t = MAX(t,-1.);
                t = MIN(t,1.);
                if( fabs(acos(t)) < adelta )
                {
                    float x, y, d;

                    x = (float) fabs( _line[2] - _lineprev[2] );
                    y = (float) fabs( _line[3] - _lineprev[3] );

                    d = x > y ? x : y;
                    if( d < rdelta )
                        break;
                }
            }
            /* calculate distances */
            err = icvCalcDist2D( points, count, _line, r );
            if( err < EPS )
                break;

            /* calculate weights */
            if( calc_weights )
                calc_weights( r, count, w );
            else
                calc_weights_param( r, count, w, _param );

            for( j = 0; j < count; j++ )
                sum_w += w[j];

            if( fabs(sum_w) > FLT_EPSILON )
            {
                sum_w = 1./sum_w;
                for( j = 0; j < count; j++ )
                    w[j] = (float)(w[j]*sum_w);
            }
            else
            {
                for( j = 0; j < count; j++ )
                    w[j] = 1.f;
            }

            /* save the line parameters */
            memcpy( _lineprev, _line, 4 * sizeof( float ));

            /* Run again... */
            icvFitLine2D_wods( points, count, w, _line );
        }

        if( err < min_err )
        {
            min_err = err;
            memcpy( line, _line, 4 * sizeof(line[0]));
            if( err < EPS )
                break;
        }
    }

    cvFree( &w );
    cvFree( &r );
    return CV_OK;
}


/* Takes an array of 3D points, type of distance (including user-defined
distance specified by callbacks, fills the array of four floats with line
parameters A, B, C, D, E, F, where (A, B, C) is the normalized direction vector,
(D, E, F) is the point that belongs to the line. */

static CvStatus
icvFitLine3D( CvPoint3D32f * points, int count, int dist,
              float _param, float reps, float aeps, float *line )
{
    double EPS = count*FLT_EPSILON;
    void (*calc_weights) (float *, int, float *) = 0;
    void (*calc_weights_param) (float *, int, float *, float) = 0;
    float *w;                   /* weights */
    float *r;                   /* square distances */
    int i, j, k;
    float _line[6], _lineprev[6];
    float rdelta = reps != 0 ? reps : 1.0f;
    float adelta = aeps != 0 ? aeps : 0.01f;
    double min_err = DBL_MAX, err = 0;
    CvRNG rng = cvRNG(-1);

    memset( line, 0, 6*sizeof(line[0]) );

    switch (dist)
    {
    case CV_DIST_L2:
        return icvFitLine3D_wods( points, count, 0, line );

    case CV_DIST_L1:
        calc_weights = icvWeightL1;
        break;

    case CV_DIST_L12:
        calc_weights = icvWeightL12;
        break;

    case CV_DIST_FAIR:
        calc_weights_param = icvWeightFair;
        break;

    case CV_DIST_WELSCH:
        calc_weights_param = icvWeightWelsch;
        break;

    case CV_DIST_HUBER:
        calc_weights_param = icvWeightHuber;
        break;

    /*case CV_DIST_USER:
        _PFP.p = param;
        calc_weights = (void ( * )(float *, int, float *)) _PFP.fp;
        break;*/

    default:
        return CV_BADFACTOR_ERR;
    }

    w = (float *) cvAlloc( count * sizeof( float ));
    r = (float *) cvAlloc( count * sizeof( float ));

    for( k = 0; k < 20; k++ )
    {
        int first = 1;
        for( i = 0; i < count; i++ )
            w[i] = 0.f;
        
        for( i = 0; i < MIN(count,10); )
        {
            j = cvRandInt(&rng) % count;
            if( w[j] < FLT_EPSILON )
            {
                w[j] = 1.f;
                i++;
            }
        }

        icvFitLine3D_wods( points, count, w, _line );
        for( i = 0; i < 30; i++ )
        {
            double sum_w = 0;

            if( first )
            {
                first = 0;
            }
            else
            {
                double t = _line[0] * _lineprev[0] + _line[1] * _lineprev[1] + _line[2] * _lineprev[2];
                t = MAX(t,-1.);
                t = MIN(t,1.);
                if( fabs(acos(t)) < adelta )
                {
                    float x, y, z, ax, ay, az, dx, dy, dz, d;

                    x = _line[3] - _lineprev[3];
                    y = _line[4] - _lineprev[4];
                    z = _line[5] - _lineprev[5];
                    ax = _line[0] - _lineprev[0];
                    ay = _line[1] - _lineprev[1];
                    az = _line[2] - _lineprev[2];
                    dx = (float) fabs( y * az - z * ay );
                    dy = (float) fabs( z * ax - x * az );
                    dz = (float) fabs( x * ay - y * ax );

                    d = dx > dy ? (dx > dz ? dx : dz) : (dy > dz ? dy : dz);
                    if( d < rdelta )
                        break;
                }
            }
            /* calculate distances */
            if( icvCalcDist3D( points, count, _line, r ) < FLT_EPSILON*count )
                break;

            /* calculate weights */
            if( calc_weights )
                calc_weights( r, count, w );
            else
                calc_weights_param( r, count, w, _param );

            for( j = 0; j < count; j++ )
                sum_w += w[j];

            if( fabs(sum_w) > FLT_EPSILON )
            {
                sum_w = 1./sum_w;
                for( j = 0; j < count; j++ )
                    w[j] = (float)(w[j]*sum_w);
            }
            else
            {
                for( j = 0; j < count; j++ )
                    w[j] = 1.f;
            }

            /* save the line parameters */
            memcpy( _lineprev, _line, 6 * sizeof( float ));

            /* Run again... */
            icvFitLine3D_wods( points, count, w, _line );
        }

        if( err < min_err )
        {
            min_err = err;
            memcpy( line, _line, 6 * sizeof(line[0]));
            if( err < EPS )
                break;
        }
    }

    // Return...
    cvFree( &w );
    cvFree( &r );
    return CV_OK;
}


CV_IMPL void
cvFitLine( const CvArr* array, int dist, double param,
           double reps, double aeps, float *line )
{
    cv::AutoBuffer<schar> buffer;

    schar* points = 0;
    union { CvContour contour; CvSeq seq; } header;
    CvSeqBlock block;
    CvSeq* ptseq = (CvSeq*)array;
    int type;

    if( !line )
        CV_Error( CV_StsNullPtr, "NULL pointer to line parameters" );

    if( CV_IS_SEQ(ptseq) )
    {
        type = CV_SEQ_ELTYPE(ptseq);
        if( ptseq->total == 0 )
            CV_Error( CV_StsBadSize, "The sequence has no points" );
        if( (type!=CV_32FC2 && type!=CV_32FC3 && type!=CV_32SC2 && type!=CV_32SC3) ||
            CV_ELEM_SIZE(type) != ptseq->elem_size )
            CV_Error( CV_StsUnsupportedFormat,
                "Input sequence must consist of 2d points or 3d points" );
    }
    else
    {
        CvMat* mat = (CvMat*)array;
        type = CV_MAT_TYPE(mat->type);
        if( !CV_IS_MAT(mat))
            CV_Error( CV_StsBadArg, "Input array is not a sequence nor matrix" );

        if( !CV_IS_MAT_CONT(mat->type) ||
            (type!=CV_32FC2 && type!=CV_32FC3 && type!=CV_32SC2 && type!=CV_32SC3) ||
            (mat->width != 1 && mat->height != 1))
            CV_Error( CV_StsBadArg,
            "Input array must be 1d continuous array of 2d or 3d points" );

        ptseq = cvMakeSeqHeaderForArray(
            CV_SEQ_KIND_GENERIC|type, sizeof(CvContour), CV_ELEM_SIZE(type), mat->data.ptr,
            mat->width + mat->height - 1, &header.seq, &block );
    }

    if( reps < 0 || aeps < 0 )
        CV_Error( CV_StsOutOfRange, "Both reps and aeps must be non-negative" );

    if( CV_MAT_DEPTH(type) == CV_32F && ptseq->first->next == ptseq->first )
    {
        /* no need to copy data in this case */
        points = ptseq->first->data;
    }
    else
    {
        buffer.allocate(ptseq->total*CV_ELEM_SIZE(type));
        points = buffer;
        cvCvtSeqToArray( ptseq, points, CV_WHOLE_SEQ );

        if( CV_MAT_DEPTH(type) != CV_32F )
        {
            int i, total = ptseq->total*CV_MAT_CN(type);
            assert( CV_MAT_DEPTH(type) == CV_32S );

            for( i = 0; i < total; i++ )
                ((float*)points)[i] = (float)((int*)points)[i];
        }
    }

    if( dist == CV_DIST_USER )
        CV_Error( CV_StsBadArg, "User-defined distance is not allowed" );

    if( CV_MAT_CN(type) == 2 )
    {
        IPPI_CALL( icvFitLine2D( (CvPoint2D32f*)points, ptseq->total,
                                 dist, (float)param, (float)reps, (float)aeps, line ));
    }
    else
    {
        IPPI_CALL( icvFitLine3D( (CvPoint3D32f*)points, ptseq->total,
                                 dist, (float)param, (float)reps, (float)aeps, line ));
    }
}

/* End of file. */
