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

namespace cv
{

static const double eps = 1e-6;

static int fitLineOriginIndex(const float* weights, int count)
{
    // Use a point that contributes to the fit when initializing from a subset.
    int i = 0;
    while (weights && i < count - 1 && weights[i] == 0)
        ++i;
    return i;
}

static void fitLine2D_wods(const Point2f* points, int count, const float* weights, float* line)
{
    CV_Assert(count > 0);
    const Point2d origin(points[fitLineOriginIndex(weights, count)]);
    Point2d mean;
    double sumWeight = 0;
    for (int i = 0; i < count; ++i)
    {
        const double w = weights ? weights[i] : 1.0;
        mean += w * (Point2d(points[i]) - origin);
        sumWeight += w;
    }
    mean *= 1.0 / sumWeight;

    double xx = 0, xy = 0, yy = 0;
    for (int i = 0; i < count; ++i)
    {
        const double w = weights ? weights[i] : 1.0;
        const Point2d p = (Point2d(points[i]) - origin) - mean;
        xx += w * (p.x * p.x);
        xy += w * (p.x * p.y);
        yy += w * (p.y * p.y);
    }

    const double angle = 0.5 * atan2(2 * xy, xx - yy);
    const Point2d center = origin + mean;
    line[0] = static_cast<float>(cos(angle));
    line[1] = static_cast<float>(sin(angle));
    line[2] = static_cast<float>(center.x);
    line[3] = static_cast<float>(center.y);
}

static void fitLine3D_wods(const Point3f* points, int count, const float* weights, float* line)
{
    CV_Assert(count > 0);
    const Point3d origin(points[fitLineOriginIndex(weights, count)]);
    Point3d mean;
    double sumWeight = 0;
    for (int i = 0; i < count; ++i)
    {
        const double w = weights ? weights[i] : 1.0;
        mean += w * (Point3d(points[i]) - origin);
        sumWeight += w;
    }
    mean *= 1.0 / sumWeight;

    // Form products after centering to avoid subtracting nearly equal raw moments.
    double xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
    for (int i = 0; i < count; ++i)
    {
        const double w = weights ? weights[i] : 1.0;
        const Point3d p = (Point3d(points[i]) - origin) - mean;
        xx += w * (p.x * p.x);
        xy += w * (p.x * p.y);
        xz += w * (p.x * p.z);
        yy += w * (p.y * p.y);
        yz += w * (p.y * p.z);
        zz += w * (p.z * p.z);
    }

    // The smallest eigenvector of trace(scatter)*I - scatter minimizes squared line distance.
    const Matx33d distanceMatrix(yy + zz, -xy, -xz,
                                -xy, xx + zz, -yz,
                                -xz, -yz, xx + yy);
    // The Jacobi fallback uses an absolute tolerance for off-diagonal entries.
    const double scale = std::max(xx, std::max(yy, zz));
    Vec3d values;
    Matx33d vectors = Matx33d::zeros();
    eigen(distanceMatrix * (scale > 0 ? 1.0 / scale : 1.0), values, vectors);

    const Point3d center = origin + mean;
    line[0] = static_cast<float>(vectors(2, 0));
    line[1] = static_cast<float>(vectors(2, 1));
    line[2] = static_cast<float>(vectors(2, 2));
    line[3] = static_cast<float>(center.x);
    line[4] = static_cast<float>(center.y);
    line[5] = static_cast<float>(center.z);
}

static double calcDist2D( const Point2f* points, int count, float *_line, float *dist )
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

static double calcDist3D( const Point3f* points, int count, float *_line, float *dist )
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

        dist[j] = (float) std::sqrt( p1*p1 + p2*p2 + p3*p3 );
        sum_dist += dist[j];
    }

    return sum_dist;
}

static void weightL1( float *d, int count, float *w )
{
    int i;

    for( i = 0; i < count; i++ )
    {
        double t = fabs( (double) d[i] );
        w[i] = (float)(1. / MAX(t, eps));
    }
}

static void weightL12( float *d, int count, float *w )
{
    int i;

    for( i = 0; i < count; i++ )
    {
        w[i] = 1.0f / (float) std::sqrt( 1 + (double) (d[i] * d[i] * 0.5) );
    }
}


static void weightHuber( float *d, int count, float *w, float _c )
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


static void weightFair( float *d, int count, float *w, float _c )
{
    int i;
    const float c = _c == 0 ? 1 / 1.3998f : 1 / _c;

    for( i = 0; i < count; i++ )
    {
        w[i] = 1 / (1 + d[i] * c);
    }
}

static void weightWelsch( float *d, int count, float *w, float _c )
{
    int i;
    const float c = _c == 0 ? 1 / 2.9846f : 1 / _c;

    for( i = 0; i < count; i++ )
    {
        w[i] = (float) std::exp( -d[i] * d[i] * c * c );
    }
}


/* Takes an array of 2D points, type of distance (including user-defined
 distance specified by callbacks, fills the array of four floats with line
 parameters A, B, C, D, where (A, B) is the normalized direction vector,
 (C, D) is the point that belongs to the line. */

static void fitLine2D( const Point2f * points, int count, int dist,
                      float _param, float reps, float aeps, float *line )
{
    double EPS = count*FLT_EPSILON;
    void (*calc_weights) (float *, int, float *) = 0;
    void (*calc_weights_param) (float *, int, float *, float) = 0;
    int i, j, k;
    float _line[4], _lineprev[4];
    float rdelta = reps != 0 ? reps : 1.0f;
    float adelta = aeps != 0 ? aeps : 0.01f;
    double min_err = DBL_MAX, err = 0;
    RNG rng((uint64)-1);

    memset( line, 0, 4*sizeof(line[0]) );

    switch (dist)
    {
    case cv::DIST_L2:
        return fitLine2D_wods( points, count, 0, line );

    case cv::DIST_L1:
        calc_weights = weightL1;
        break;

    case cv::DIST_L12:
        calc_weights = weightL12;
        break;

    case cv::DIST_FAIR:
        calc_weights_param = weightFair;
        break;

    case cv::DIST_WELSCH:
        calc_weights_param = weightWelsch;
        break;

    case cv::DIST_HUBER:
        calc_weights_param = weightHuber;
        break;

    /*case DIST_USER:
     calc_weights = (void ( * )(float *, int, float *)) _PFP.fp;
     break;*/
    default:
        CV_Error(cv::Error::StsBadArg, "Unknown distance type");
    }

    AutoBuffer<float> wr(count*2);
    float *w = wr.data(), *r = w + count;

    for( k = 0; k < 20; k++ )
    {
        int first = 1;
        for( i = 0; i < count; i++ )
            w[i] = 0.f;

        for( i = 0; i < MIN(count,10); )
        {
            j = rng.uniform(0, count);
            if( w[j] < FLT_EPSILON )
            {
                w[j] = 1.f;
                i++;
            }
        }

        fitLine2D_wods( points, count, w, _line );
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
            err = calcDist2D( points, count, _line, r );

            if (err < min_err)
            {
                min_err = err;
                memcpy(line, _line, 4 * sizeof(line[0]));
                if (err < EPS)
                    break;
            }

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
            fitLine2D_wods( points, count, w, _line );
        }

        if( err < min_err )
        {
            min_err = err;
            memcpy( line, _line, 4 * sizeof(line[0]));
            if( err < EPS )
                break;
        }
    }
}


/* Takes an array of 3D points, type of distance (including user-defined
 distance specified by callbacks, fills the array of four floats with line
 parameters A, B, C, D, E, F, where (A, B, C) is the normalized direction vector,
 (D, E, F) is the point that belongs to the line. */
static void fitLine3D( Point3f * points, int count, int dist,
                       float _param, float reps, float aeps, float *line )
{
    double EPS = count*FLT_EPSILON;
    void (*calc_weights) (float *, int, float *) = 0;
    void (*calc_weights_param) (float *, int, float *, float) = 0;
    int i, j, k;
    float _line[6]={0,0,0,0,0,0}, _lineprev[6]={0,0,0,0,0,0};
    float rdelta = reps != 0 ? reps : 1.0f;
    float adelta = aeps != 0 ? aeps : 0.01f;
    double min_err = DBL_MAX, err = 0;
    RNG rng((uint64)-1);

    switch (dist)
    {
    case cv::DIST_L2:
        return fitLine3D_wods( points, count, 0, line );

    case cv::DIST_L1:
        calc_weights = weightL1;
        break;

    case cv::DIST_L12:
        calc_weights = weightL12;
        break;

    case cv::DIST_FAIR:
        calc_weights_param = weightFair;
        break;

    case cv::DIST_WELSCH:
        calc_weights_param = weightWelsch;
        break;

    case cv::DIST_HUBER:
        calc_weights_param = weightHuber;
        break;

    default:
        CV_Error(cv::Error::StsBadArg, "Unknown distance");
    }

    AutoBuffer<float> buf(count*2);
    float *w = buf.data(), *r = w + count;

    for( k = 0; k < 20; k++ )
    {
        int first = 1;
        for( i = 0; i < count; i++ )
            w[i] = 0.f;

        for( i = 0; i < MIN(count,10); )
        {
            j = rng.uniform(0, count);
            if( w[j] < FLT_EPSILON )
            {
                w[j] = 1.f;
                i++;
            }
        }

        fitLine3D_wods( points, count, w, _line );
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
            err = calcDist3D( points, count, _line, r );
            if (err < min_err)
            {
                min_err = err;
                memcpy(line, _line, 6 * sizeof(line[0]));
                if (err < EPS)
                    break;
            }

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
            fitLine3D_wods( points, count, w, _line );
        }

        if( err < min_err )
        {
            min_err = err;
            memcpy( line, _line, 6 * sizeof(line[0]));
            if( err < EPS )
                break;
        }
    }
}

}

void cv::fitLine( InputArray _points, OutputArray _line, int distType,
                 double param, double reps, double aeps )
{
    CV_INSTRUMENT_REGION();

    Mat points = _points.getMat();

    float linebuf[6]={0.f};
    int npoints2 = points.checkVector(2, -1, false);
    int npoints3 = points.checkVector(3, -1, false);

    CV_Assert( npoints2 >= 0 || npoints3 >= 0 );

    if( points.depth() != CV_32F || !points.isContinuous() )
    {
        Mat temp;
        points.convertTo(temp, CV_32F);
        points = temp;
    }

    if( npoints2 >= 0 )
        fitLine2D( points.ptr<Point2f>(), npoints2, distType,
                   (float)param, (float)reps, (float)aeps, linebuf);
    else
        fitLine3D( points.ptr<Point3f>(), npoints3, distType,
                   (float)param, (float)reps, (float)aeps, linebuf);

    Mat(npoints2 >= 0 ? 4 : 6, 1, CV_32F, linebuf).copyTo(_line);
}


CV_IMPL void
cvFitLine( const CvArr* array, int dist, double param,
           double reps, double aeps, float *line )
{
    CV_Assert(line != 0);

    cv::AutoBuffer<double> buf;
    cv::Mat points = cv::cvarrToMat(array, false, false, 0, &buf);
    cv::Mat linemat(points.checkVector(2) >= 0 ? 4 : 6, 1, CV_32F, line);

    cv::fitLine(points, linemat, dist, param, reps, aeps);
}

/* End of file. */
