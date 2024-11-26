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
#include <cstring>
#include <ctime>

#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#include <direct.h>
#endif /* _WIN32 */

#include "utility.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

#if defined __GNUC__ && __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif

using namespace cv;

#ifndef PATH_MAX
#define PATH_MAX 512
#endif /* PATH_MAX */

#define __BEGIN__ __CV_BEGIN__
#define __END__  __CV_END__
#define EXIT __CV_EXIT__

static int icvMkDir( const char* filename )
{
    char path[PATH_MAX+1];
    char* p;
    int pos;

#ifdef _WIN32
    struct _stat st;
#else /* _WIN32 */
    struct stat st;
    mode_t mode;

    mode = 0755;
#endif /* _WIN32 */

    path[0] = '\0';
    strncat( path, filename, PATH_MAX );

    p = path;
    for( ; ; )
    {
        pos = (int)strcspn( p, "/\\" );

        if( pos == (int) strlen( p ) ) break;
        if( pos != 0 )
        {
            p[pos] = '\0';

#ifdef _WIN32
            if( p[pos-1] != ':' )
            {
                if( _stat( path, &st ) != 0 )
                {
                    if( _mkdir( path ) != 0 ) return 0;
                }
            }
#else /* _WIN32 */
            if( stat( path, &st ) != 0 )
            {
                if( mkdir( path, mode ) != 0 ) return 0;
            }
#endif /* _WIN32 */
        }

        p[pos] = '/';

        p += pos + 1;
    }

    return 1;
}

static void icvWriteVecHeader( FILE* file, int count, int width, int height )
{
    int vecsize;
    short tmp;

    /* number of samples */
    fwrite( &count, sizeof( count ), 1, file );
    /* vector size */
    vecsize = width * height;
    fwrite( &vecsize, sizeof( vecsize ), 1, file );
    /* min/max values */
    tmp = 0;
    fwrite( &tmp, sizeof( tmp ), 1, file );
    fwrite( &tmp, sizeof( tmp ), 1, file );
}

static void icvWriteVecSample( FILE* file, Mat sample )
{
    uchar chartmp = 0;
    fwrite( &chartmp, sizeof( chartmp ), 1, file );
    for(int r = 0; r < sample.rows; r++ )
    {
        for(int c = 0; c < sample.cols; c++ )
        {
            short tmp = sample.at<uchar>(r,c);
            fwrite( &tmp, sizeof( tmp ), 1, file );
        }
    }
}

/* Calculates coefficients of perspective transformation
 * which maps <quad> into rectangle ((0,0), (w,0), (w,h), (h,0)):
 *
 *      c00*xi + c01*yi + c02
 * ui = ---------------------
 *      c20*xi + c21*yi + c22
 *
 *      c10*xi + c11*yi + c12
 * vi = ---------------------
 *      c20*xi + c21*yi + c22
 *
 * Coefficients are calculated by solving linear system:
 * / x0 y0  1  0  0  0 -x0*u0 -y0*u0 \ /c00\ /u0\
 * | x1 y1  1  0  0  0 -x1*u1 -y1*u1 | |c01| |u1|
 * | x2 y2  1  0  0  0 -x2*u2 -y2*u2 | |c02| |u2|
 * | x3 y3  1  0  0  0 -x3*u3 -y3*u3 |.|c10|=|u3|,
 * |  0  0  0 x0 y0  1 -x0*v0 -y0*v0 | |c11| |v0|
 * |  0  0  0 x1 y1  1 -x1*v1 -y1*v1 | |c12| |v1|
 * |  0  0  0 x2 y2  1 -x2*v2 -y2*v2 | |c20| |v2|
 * \  0  0  0 x3 y3  1 -x3*v3 -y3*v3 / \c21/ \v3/
 *
 * where:
 *   (xi, yi) = (quad[i][0], quad[i][1])
 *        cij - coeffs[i][j], coeffs[2][2] = 1
 *   (ui, vi) - rectangle vertices
 */
static void cvGetPerspectiveTransform( Size src_size, double quad[4][2], double coeffs[3][3] )
{
    double a[8][8];
    double b[8];

    Mat A( 8, 8, CV_64FC1, a );
    Mat B( 8, 1, CV_64FC1, b );
    Mat X( 8, 1, CV_64FC1, coeffs );

    int i;
    for( i = 0; i < 4; ++i )
    {
        a[i][0] = quad[i][0]; a[i][1] = quad[i][1]; a[i][2] = 1;
        a[i][3] = a[i][4] = a[i][5] = a[i][6] = a[i][7] = 0;
        b[i] = 0;
    }
    for( i = 4; i < 8; ++i )
    {
        a[i][3] = quad[i-4][0]; a[i][4] = quad[i-4][1]; a[i][5] = 1;
        a[i][0] = a[i][1] = a[i][2] = a[i][6] = a[i][7] = 0;
        b[i] = 0;
    }

    int u = src_size.width - 1;
    int v = src_size.height - 1;

    a[1][6] = -quad[1][0] * u; a[1][7] = -quad[1][1] * u;
    a[2][6] = -quad[2][0] * u; a[2][7] = -quad[2][1] * u;
    b[1] = b[2] = u;

    a[6][6] = -quad[2][0] * v; a[6][7] = -quad[2][1] * v;
    a[7][6] = -quad[3][0] * v; a[7][7] = -quad[3][1] * v;
    b[6] = b[7] = v;

    solve( A, B, X );

    coeffs[2][2] = 1;
}

/* Warps source into destination by a perspective transform */
static void cvWarpPerspective( Mat src, Mat dst, double quad[4][2] )
{
    int fill_value = 0;

    double c[3][3]; /* transformation coefficients */
    double q[4][2]; /* rearranged quad */

    int left = 0;
    int right = 0;
    int next_right = 0;
    int next_left = 0;
    double y_min = 0;
    double y_max = 0;
    double k_left, b_left, k_right, b_right;

    double d = 0;
    int direction = 0;
    int i;

    if( src.type() != CV_8UC1 || src.dims != 2 )
    {
        CV_Error( Error::StsBadArg,
            "Source must be two-dimensional array of CV_8UC1 type." );
    }
    if( dst.type() != CV_8UC1 || dst.dims != 2 )
    {
        CV_Error( Error::StsBadArg,
            "Destination must be two-dimensional array of CV_8UC1 type." );
    }

    cvGetPerspectiveTransform( src.size(), quad, c );

    /* if direction > 0 then vertices in quad follow in a CW direction,
       otherwise they follow in a CCW direction */
    direction = 0;
    for( i = 0; i < 4; ++i )
    {
        int ni = i + 1; if( ni == 4 ) ni = 0;
        int pi = i - 1; if( pi == -1 ) pi = 3;

        d = (quad[i][0] - quad[pi][0])*(quad[ni][1] - quad[i][1]) -
            (quad[i][1] - quad[pi][1])*(quad[ni][0] - quad[i][0]);
        int cur_direction = d > 0 ? 1 : d < 0 ? -1 : 0;
        if( direction == 0 )
        {
            direction = cur_direction;
        }
        else if( direction * cur_direction < 0 )
        {
            direction = 0;
            break;
        }
    }
    if( direction == 0 )
    {
        CV_Error(Error::StsBadArg, "Quadrangle is nonconvex or degenerated." );
    }

    /* <left> is the index of the topmost quad vertice
       if there are two such vertices <left> is the leftmost one */
    left = 0;
    for( i = 1; i < 4; ++i )
    {
        if( (quad[i][1] < quad[left][1]) ||
            ((quad[i][1] == quad[left][1]) && (quad[i][0] < quad[left][0])) )
        {
            left = i;
        }
    }
    /* rearrange <quad> vertices in such way that they follow in a CW
       direction and the first vertice is the topmost one and put them
       into <q> */
    if( direction > 0 )
    {
        for( i = left; i < 4; ++i )
        {
            q[i-left][0] = quad[i][0];
            q[i-left][1] = quad[i][1];
        }
        for( i = 0; i < left; ++i )
        {
            q[4-left+i][0] = quad[i][0];
            q[4-left+i][1] = quad[i][1];
        }
    }
    else
    {
        for( i = left; i >= 0; --i )
        {
            q[left-i][0] = quad[i][0];
            q[left-i][1] = quad[i][1];
        }
        for( i = 3; i > left; --i )
        {
            q[4+left-i][0] = quad[i][0];
            q[4+left-i][1] = quad[i][1];
        }
    }

    left = right = 0;
    /* if there are two topmost points, <right> is the index of the rightmost one
       otherwise <right> */
    if( q[left][1] == q[left+1][1] )
    {
        right = 1;
    }

    /* <next_left> follows <left> in a CCW direction */
    next_left = 3;
    /* <next_right> follows <right> in a CW direction */
    next_right = right + 1;

    /* subtraction of 1 prevents skipping of the first row */
    y_min = q[left][1] - 1;

    /* left edge equation: y = k_left * x + b_left */
    k_left = (q[left][0] - q[next_left][0]) /
               (q[left][1] - q[next_left][1]);
    b_left = (q[left][1] * q[next_left][0] -
               q[left][0] * q[next_left][1]) /
                 (q[left][1] - q[next_left][1]);

    /* right edge equation: y = k_right * x + b_right */
    k_right = (q[right][0] - q[next_right][0]) /
               (q[right][1] - q[next_right][1]);
    b_right = (q[right][1] * q[next_right][0] -
               q[right][0] * q[next_right][1]) /
                 (q[right][1] - q[next_right][1]);

    for(;;)
    {
        int x, y;

        y_max = MIN( q[next_left][1], q[next_right][1] );

        int iy_min = MAX( cvRound(y_min), 0 ) + 1;
        int iy_max = MIN( cvRound(y_max), dst.rows - 1 );

        double x_min = k_left * iy_min + b_left;
        double x_max = k_right * iy_min + b_right;

        /* walk through the destination quadrangle row by row */
        for( y = iy_min; y <= iy_max; ++y )
        {
            int ix_min = MAX( cvRound( x_min ), 0 );
            int ix_max = MIN( cvRound( x_max ), dst.cols - 1 );

            for( x = ix_min; x <= ix_max; ++x )
            {
                /* calculate coordinates of the corresponding source array point */
                double div = (c[2][0] * x + c[2][1] * y + c[2][2]);
                double src_x = (c[0][0] * x + c[0][1] * y + c[0][2]) / div;
                double src_y = (c[1][0] * x + c[1][1] * y + c[1][2]) / div;

                int isrc_x = cvFloor( src_x );
                int isrc_y = cvFloor( src_y );
                double delta_x = src_x - isrc_x;
                double delta_y = src_y - isrc_y;

                int i00, i10, i01, i11;
                i00 = i10 = i01 = i11 = (int) fill_value;

                /* linear interpolation using 2x2 neighborhood */
                if( isrc_x >= 0 && isrc_x < src.cols &&
                    isrc_y >= 0 && isrc_y < src.rows )
                {
                    i00 = src.at<uchar>(isrc_y, isrc_x);
                }
                if( isrc_x >= -1 && isrc_x + 1 < src.cols &&
                    isrc_y >= 0 && isrc_y < src.rows )
                {
                    i10 = src.at<uchar>(isrc_y, isrc_x + 1);
                }
                if( isrc_x >= 0 && isrc_x < src.cols &&
                    isrc_y >= -1 && isrc_y + 1 < src.rows )
                {
                    i01 = src.at<uchar>(isrc_y + 1, isrc_x);
                }
                if( isrc_x >= -1 && isrc_x + 1 < src.cols &&
                    isrc_y >= -1 && isrc_y + 1 < src.rows )
                {
                    i11 = src.at<uchar>(isrc_y + 1, isrc_x + 1);
                }

                double i0 = i00 + (i10 - i00)*delta_x;
                double i1 = i01 + (i11 - i01)*delta_x;

                dst.at<uchar>(y, x) = (uchar) (i0 + (i1 - i0)*delta_y);
            }
            x_min += k_left;
            x_max += k_right;
        }

        if( (next_left == next_right) ||
            (next_left+1 == next_right && q[next_left][1] == q[next_right][1]) )
        {
            break;
        }

        if( y_max == q[next_left][1] )
        {
            left = next_left;
            next_left = left - 1;

            k_left = (q[left][0] - q[next_left][0]) /
                       (q[left][1] - q[next_left][1]);
            b_left = (q[left][1] * q[next_left][0] -
                       q[left][0] * q[next_left][1]) /
                         (q[left][1] - q[next_left][1]);
        }
        if( y_max == q[next_right][1] )
        {
            right = next_right;
            next_right = right + 1;

            k_right = (q[right][0] - q[next_right][0]) /
                       (q[right][1] - q[next_right][1]);
            b_right = (q[right][1] * q[next_right][0] -
                       q[right][0] * q[next_right][1]) /
                         (q[right][1] - q[next_right][1]);
        }
        y_min = y_max;
    }
}

static
void icvRandomQuad( int width, int height, double quad[4][2],
                    double maxxangle,
                    double maxyangle,
                    double maxzangle )
{
    double distfactor = 3.0;
    double distfactor2 = 1.0;

    double halfw, halfh;
    int i;

    double rotVectData[3];
    double vectData[3];
    double rotMatData[9];

    double d;

    Mat rotVect( 3, 1, CV_64FC1, &rotVectData[0] );
    Mat rotMat( 3, 3, CV_64FC1, &rotMatData[0] );
    Mat vect( 3, 1, CV_64FC1, &vectData[0] );

    rotVectData[0] = theRNG().uniform( -maxxangle, maxxangle );
    rotVectData[1] = ( maxyangle - fabs( rotVectData[0] ) ) * theRNG().uniform( -1.0, 1.0 );
    rotVectData[2] = theRNG().uniform( -maxzangle, maxzangle );
    d = ( distfactor + distfactor2 * theRNG().uniform( -1.0, 1.0 ) ) * width;

    Rodrigues( rotVect, rotMat );

    halfw = 0.5 * width;
    halfh = 0.5 * height;

    quad[0][0] = -halfw;
    quad[0][1] = -halfh;
    quad[1][0] =  halfw;
    quad[1][1] = -halfh;
    quad[2][0] =  halfw;
    quad[2][1] =  halfh;
    quad[3][0] = -halfw;
    quad[3][1] =  halfh;

    for( i = 0; i < 4; i++ )
    {
        rotVectData[0] = quad[i][0];
        rotVectData[1] = quad[i][1];
        rotVectData[2] = 0.0;
        gemm(rotMat, rotVect, 1., Mat(), 1., vect);
        quad[i][0] = vectData[0] * d / (d + vectData[2]) + halfw;
        quad[i][1] = vectData[1] * d / (d + vectData[2]) + halfh;
    }
}


typedef struct CvSampleDistortionData
{
    Mat src;
    Mat erode;
    Mat dilate;
    Mat mask;
    Mat img;
    Mat maskimg;
    int dx;
    int dy;
    int bgcolor;
} CvSampleDistortionData;

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
    Size winsize;
} CvBackgroundData;

typedef struct CvBackgroundReader
{
    Mat   src;
    Mat   img;
    Point offset;
    float   scale;
    float   scalefactor;
    float   stepfactor;
    Point point;
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

static int icvStartSampleDistortion( const char* imgfilename, int bgcolor, int bgthreshold,
                              CvSampleDistortionData* data )
{
    memset( data, 0, sizeof( *data ) );
    data->src = imread( imgfilename, IMREAD_GRAYSCALE );
    if( !(data->src.empty()) && data->src.type() == CV_8UC1 )
    {
        int r, c;

        data->dx = data->src.cols / 2;
        data->dy = data->src.rows / 2;
        data->bgcolor = bgcolor;

        data->mask = data->src.clone();
        data->erode = data->src.clone();
        data->dilate = data->src.clone();

        /* make mask image */
        for( r = 0; r < data->mask.rows; r++ )
        {
            for( c = 0; c < data->mask.cols; c++ )
            {
                uchar& pmask = data->mask.at<uchar>(r, c);
                if( bgcolor - bgthreshold <= (int)pmask &&
                    (int)pmask <= bgcolor + bgthreshold )
                {
                    pmask = (uchar) 0;
                }
                else
                {
                    pmask = (uchar) 255;
                }
            }
        }

        /* extend borders of source image */
        erode( data->src, data->erode, Mat() );
        dilate( data->src, data->dilate, Mat() );
        for( r = 0; r < data->mask.rows; r++ )
        {
            for( c = 0; c < data->mask.cols; c++ )
            {
                uchar& pmask = data->mask.at<uchar>(r, c);
                if( pmask == 0 )
                {
                    uchar& psrc = data->src.at<uchar>(r, c);
                    uchar& perode = data->erode.at<uchar>(r, c);
                    uchar& pdilate = data->dilate.at<uchar>(r, c);
                    uchar de = (uchar)(bgcolor - perode);
                    uchar dd = (uchar)(pdilate - bgcolor);
                    if( de >= dd && de > bgthreshold )
                    {
                        psrc = perode;
                    }
                    if( dd > de && dd > bgthreshold )
                    {
                        psrc = pdilate;
                    }
                }
            }
        }

        data->img = Mat(Size( data->src.cols + 2 * data->dx, data->src.rows + 2 * data->dy ), CV_8UC1);
        data->maskimg = Mat(Size(data->src.cols + 2 * data->dx, data->src.rows + 2 * data->dy), CV_8UC1);

        return 1;
    }

    return 0;
}

static
void icvPlaceDistortedSample( Mat background,
                              int inverse, int maxintensitydev,
                              double maxxangle, double maxyangle, double maxzangle,
                              int inscribe, double maxshiftf, double maxscalef,
                              CvSampleDistortionData* data )
{
    double quad[4][2];
    int r, c;
    int forecolordev;
    float scale;

    Rect cr;
    Rect roi;

    double xshift, yshift, randscale;

    icvRandomQuad( data->src.cols, data->src.rows, quad,
                   maxxangle, maxyangle, maxzangle );
    quad[0][0] += (double) data->dx;
    quad[0][1] += (double) data->dy;
    quad[1][0] += (double) data->dx;
    quad[1][1] += (double) data->dy;
    quad[2][0] += (double) data->dx;
    quad[2][1] += (double) data->dy;
    quad[3][0] += (double) data->dx;
    quad[3][1] += (double) data->dy;

    data->img = data->bgcolor;
    data->maskimg = 0;

    cvWarpPerspective( data->src, data->img, quad );
    cvWarpPerspective( data->mask, data->maskimg, quad );

    GaussianBlur( data->maskimg, data->maskimg, Size(3, 3), 0, 0 );

    cr.x = data->dx;
    cr.y = data->dy;
    cr.width = data->src.cols;
    cr.height = data->src.rows;

    if( inscribe )
    {
        /* quad's circumscribing rectangle */
        cr.x = (int) MIN( quad[0][0], quad[3][0] );
        cr.y = (int) MIN( quad[0][1], quad[1][1] );
        cr.width  = (int) (MAX( quad[1][0], quad[2][0] ) + 0.5F ) - cr.x;
        cr.height = (int) (MAX( quad[2][1], quad[3][1] ) + 0.5F ) - cr.y;
    }

    xshift = theRNG().uniform( 0., maxshiftf );
    yshift = theRNG().uniform( 0., maxshiftf );

    cr.x -= (int) ( xshift * cr.width  );
    cr.y -= (int) ( yshift * cr.height );
    cr.width  = (int) ((1.0 + maxshiftf) * cr.width );
    cr.height = (int) ((1.0 + maxshiftf) * cr.height);

    randscale = theRNG().uniform( 0., maxscalef );
    cr.x -= (int) ( 0.5 * randscale * cr.width  );
    cr.y -= (int) ( 0.5 * randscale * cr.height );
    cr.width  = (int) ((1.0 + randscale) * cr.width );
    cr.height = (int) ((1.0 + randscale) * cr.height);

    scale = MAX( ((float) cr.width) / background.cols, ((float) cr.height) / background.rows );

    roi.x = (int) (-0.5F * (scale * background.cols - cr.width) + cr.x);
    roi.y = (int) (-0.5F * (scale * background.rows - cr.height) + cr.y);
    roi.width  = (int) (scale * background.cols);
    roi.height = (int) (scale * background.rows);

    Mat img( background.size(), CV_8UC1 );
    Mat maskimg( background.size(), CV_8UC1 );

    resize( data->img(roi & Rect(Point(0,0), data->img.size())), img, img.size(), 0, 0, INTER_LINEAR_EXACT);
    resize( data->maskimg(roi & Rect(Point(0, 0), data->maskimg.size())), maskimg, maskimg.size(), 0, 0, INTER_LINEAR_EXACT);

    forecolordev = theRNG().uniform( -maxintensitydev, maxintensitydev );

    for( r = 0; r < img.rows; r++ )
    {
        for( c = 0; c < img.cols; c++ )
        {
            uchar& pbg = background.at<uchar>(r, c);
            uchar& palpha = maskimg.at<uchar>(r, c);
            uchar chartmp = (uchar) MAX( 0, MIN( 255, forecolordev + img.at<uchar>(r, c)) );
            if( inverse )
            {
                chartmp ^= 0xFF;
            }
            pbg = (uchar) ((chartmp*palpha + (255 - palpha)*pbg) / 255);
        }
    }
}

static
CvBackgroundData* icvCreateBackgroundData( const char* filename, Size winsize )
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

    CV_Assert( filename != NULL );

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
            fseek( input, 0, SEEK_SET );
            datasize += sizeof( *data ) + sizeof( char* ) * count;
            data = (CvBackgroundData*) fastMalloc( datasize );
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
CvBackgroundReader* icvCreateBackgroundReader()
{
    CvBackgroundReader* reader = NULL;

    reader = new CvBackgroundReader;
    reader->scale       = 1.0F;
    reader->scalefactor = 1.4142135623730950488016887242097F;
    reader->stepfactor  = 0.5F;

    return reader;
}

static
void icvGetNextFromBackgroundData( CvBackgroundData* data,
                                   CvBackgroundReader* reader )
{
    Mat img;
    int round = 0;
    int i = 0;
    Point offset;

    CV_Assert( data != NULL && reader != NULL );

    #ifdef CV_OPENMP
    #pragma omp critical(c_background_data)
    #endif /* CV_OPENMP */
    {
        for( i = 0; i < data->count; i++ )
        {
            round = data->round;

            data->last = theRNG().uniform( 0, RAND_MAX ) % data->count;

#ifdef CV_VERBOSE
            printf( "Open background image: %s\n", data->filename[data->last] );
#endif /* CV_VERBOSE */

            img = imread( data->filename[data->last], IMREAD_GRAYSCALE );
            if( img.empty() )
                continue;
            data->round += data->last / data->count;
            data->round = data->round % (data->winsize.width * data->winsize.height);

            offset.x = round % data->winsize.width;
            offset.y = round / data->winsize.width;

            offset.x = MIN( offset.x, img.cols - data->winsize.width );
            offset.y = MIN( offset.y, img.rows - data->winsize.height );

            if( !img.empty() && img.type() == CV_8UC1 && offset.x >= 0 && offset.y >= 0 )
            {
                break;
            }
            img = Mat();
        }
    }
    if( img.empty() )
    {
        /* no appropriate image */

#ifdef CV_VERBOSE
        printf( "Invalid background description file.\n" );
#endif /* CV_VERBOSE */

        CV_Assert( 0 );
        exit( 1 );
    }

    reader->src = img;


    reader->offset = offset;
    reader->point = reader->offset;
    reader->scale = MAX(
        ((float) data->winsize.width + reader->point.x) / ((float) reader->src.cols),
        ((float) data->winsize.height + reader->point.y) / ((float) reader->src.rows) );

    resize( reader->src, reader->img,
            Size((int)(reader->scale * reader->src.cols + 0.5F), (int)(reader->scale * reader->src.rows + 0.5F)), 0, 0, INTER_LINEAR_EXACT);
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
 *     icvGetBackgroundImage( cvbgdata, cvbgreader, img );
 *     ...
 * }
 * ...
 * icvDestroyBackgroundReaders();
 */
static
void icvGetBackgroundImage( CvBackgroundData* data,
                            CvBackgroundReader* reader,
                            Mat& img )
{
    CV_Assert( data != NULL && reader != NULL );

    if( reader->img.empty() )
    {
        icvGetNextFromBackgroundData( data, reader );
    }

    img = reader->img(Rect(reader->point.x, reader->point.y, data->winsize.width, data->winsize.height)).clone();

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
                resize(reader->src, reader->img,
                       Size((int)(reader->scale * reader->src.cols), (int)(reader->scale * reader->src.rows)), 0, 0, INTER_LINEAR_EXACT);
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
static int icvInitBackgroundReaders( const char* filename, Size winsize )
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
 * Finish background reading process
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
                delete cvbgreader;
                cvbgreader = NULL;
            }
        }
    }

    if( cvbgdata != NULL )
    {
        fastFree(cvbgdata);
        cvbgdata = NULL;
    }
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

    CV_Assert( filename != NULL );
    CV_Assert( imgfilename != NULL );

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
            int i;
            int inverse;

            const int hasbg = (bgfilename != NULL && icvInitBackgroundReaders( bgfilename,
                     Size( winwidth,winheight ) ) );

            Mat sample( winheight, winwidth, CV_8UC1 );

            icvWriteVecHeader( output, count, sample.cols, sample.rows );

            if( showsamples )
            {
                namedWindow( "Sample", WINDOW_AUTOSIZE );
            }

            inverse = invert;
            for( i = 0; i < count; i++ )
            {
                if( hasbg )
                {
                    icvGetBackgroundImage( cvbgdata, cvbgreader, sample );
                }
                else
                {
                    sample = bgcolor;
                }

                if( invert == CV_RANDOM_INVERT )
                {
                    inverse = theRNG().uniform( 0, 2 );
                }
                icvPlaceDistortedSample( sample, inverse, maxintensitydev,
                    maxxangle, maxyangle, maxzangle,
                    0   /* nonzero means placing image without cut offs */,
                    0.0 /* nonzero adds random shifting                  */,
                    0.0 /* nonzero adds random scaling                   */,
                    &data );

                if( showsamples )
                {
                    imshow( "Sample", sample );
                    if( (waitKey( 0 ) & 0xFF) == 27 )
                    {
                        showsamples = 0;
                    }
                }

                icvWriteVecSample( output, sample );

#ifdef CV_VERBOSE
                if( i % 500 == 0 )
                {
                    printf( "\r%3d%%", 100 * i / count );
                }
#endif /* CV_VERBOSE */
            }
            icvDestroyBackgroundReaders();
            fclose( output );
        } /* if( output != NULL ) */
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
                          int winwidth, int winheight, double maxscale )
{
    CvSampleDistortionData data;

    CV_Assert( infoname != NULL );
    CV_Assert( imgfilename != NULL );
    CV_Assert( bgfilename != NULL );

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
        FILE* info;

        if( icvInitBackgroundReaders( bgfilename, Size( 10, 10 ) ) )
        {
            int i;
            int x, y, width, height;
            float scale;
            int inverse;

            if( showsamples )
            {
                namedWindow( "Image", WINDOW_AUTOSIZE );
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
                filename++; // get basename after last path delimiter
            }

            count = MIN( count, cvbgdata->count );
            inverse = invert;
            for( i = 0; i < count; i++ )
            {
                icvGetNextFromBackgroundData( cvbgdata, cvbgreader );
                if( maxscale < 0.0 )
                {
                    maxscale = MIN( 0.7F * cvbgreader->src.cols / winwidth,
                                   0.7F * cvbgreader->src.rows / winheight );
                }

                if( maxscale < 1.0F ) continue;

                scale = theRNG().uniform( 1.0F, (float)maxscale );

                width = (int) (scale * winwidth);
                height = (int) (scale * winheight);
                x = (int) ( theRNG().uniform( 0.1, 0.8 ) * (cvbgreader->src.cols - width));
                y = (int) ( theRNG().uniform( 0.1, 0.8 ) * (cvbgreader->src.rows - height));

                if( invert == CV_RANDOM_INVERT )
                {
                    inverse = theRNG().uniform( 0, 2 );
                }
                icvPlaceDistortedSample( cvbgreader->src(Rect(x, y, width, height)), inverse, maxintensitydev,
                                         maxxangle, maxyangle, maxzangle,
                                         1, 0.0, 0.0, &data );


                snprintf( filename, sizeof(fullname) - (filename - fullname), "%04d_%04d_%04d_%04d_%04d.jpg",
                          (i + 1), x, y, width, height );

                if( info )
                {
                    fprintf( info, "%s %d %d %d %d %d\n",
                        filename, 1, x, y, width, height );
                }

                imwrite( fullname, cvbgreader->src );
                if( showsamples )
                {
                    imshow( "Image", cvbgreader->src );
                    if( (waitKey( 0 ) & 0xFF) == 27 )
                    {
                        showsamples = 0;
                    }
                }
            }
            if( info ) fclose( info );
            icvDestroyBackgroundReaders();
        }
    }
}


int cvCreateTrainingSamplesFromInfo( const char* infoname, const char* vecfilename,
                                     int num,
                                     int showsamples,
                                     int winwidth, int winheight )
{
    char fullname[PATH_MAX];
    char* filename;

    FILE* info;
    FILE* vec;
    int line;
    int error;
    int i;
    int x, y, width, height;
    int total;

    CV_Assert( infoname != NULL );
    CV_Assert( vecfilename != NULL );

    total = 0;
    if( !icvMkDir( vecfilename ) )
    {

#if CV_VERBOSE
        fprintf( stderr, "Unable to create directory hierarchy: %s\n", vecfilename );
#endif /* CV_VERBOSE */

        return total;
    }

    info = fopen( infoname, "r" );
    if( info == NULL )
    {

#if CV_VERBOSE
        fprintf( stderr, "Unable to open file: %s\n", infoname );
#endif /* CV_VERBOSE */

        return total;
    }

    vec = fopen( vecfilename, "wb" );
    if( vec == NULL )
    {

#if CV_VERBOSE
        fprintf( stderr, "Unable to open file: %s\n", vecfilename );
#endif /* CV_VERBOSE */

        fclose( info );

        return total;
    }

    icvWriteVecHeader( vec, num, winwidth, winheight );

    if( showsamples )
    {
        namedWindow( "Sample", WINDOW_AUTOSIZE );
    }

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

    for( line = 1, error = 0, total = 0; total < num ;line++ )
    {
        Mat src;
        int count;

        if(fscanf(info, "%s %d", filename, &count) == 2)
        {
            src = imread( fullname, IMREAD_GRAYSCALE );
            if(src.empty())
            {

#if CV_VERBOSE
                fprintf( stderr, "Unable to open image: %s\n", fullname );
#endif /* CV_VERBOSE */

            }
        }
        for( i = 0; (i < count) && (total < num); i++, total++ )
        {
            error = ( fscanf( info, "%d %d %d %d", &x, &y, &width, &height ) != 4 );
            if( error ) break;
            Mat sample;
            resize( src(Rect(x, y, width, height)), sample, Size(winwidth, winheight), 0, 0,
                    width >= winwidth && height >= winheight ? INTER_AREA : INTER_LINEAR_EXACT );

            if( showsamples )
            {
                imshow( "Sample", sample );
                if( (waitKey( 0 ) & 0xFF) == 27 )
                {
                    showsamples = 0;
                }
            }
            icvWriteVecSample( vec, sample );
        }

        if( error )
        {

#if CV_VERBOSE
            fprintf( stderr, "%s(%d) : parse error", infoname, line );
#endif /* CV_VERBOSE */

            break;
        }
    }

    fclose( vec );
    fclose( info );

    return total;
}

typedef struct CvVecFile
{
    FILE*  input;
    int    count;
    int    vecsize;
    int    last;
} CvVecFile;

static
int icvGetTraininDataFromVec( Mat& img, CvVecFile& userdata )
{
    AutoBuffer<short> vector(userdata.vecsize);
    uchar tmp = 0;
    int r = 0;
    int c = 0;

    CV_Assert( img.rows * img.cols == userdata.vecsize );

    size_t elements_read = fread( &tmp, sizeof( tmp ), 1, userdata.input );
    CV_Assert(elements_read == 1);
    elements_read = fread(vector.data(), sizeof(short), userdata.vecsize, userdata.input);
    CV_Assert(elements_read == (size_t)userdata.vecsize);

    if( feof( userdata.input ) || userdata.last++ >= userdata.count )
    {
        return 0;
    }

    for( r = 0; r < img.rows; r++ )
    {
        for( c = 0; c < img.cols; c++ )
        {
            img.at<uchar>(r, c) = (uchar) ( vector[r * img.cols + c] );
        }
    }

    return 1;
}
void cvShowVecSamples( const char* filename, int winwidth, int winheight,
                       double scale )
{
    CvVecFile file;
    short tmp;
    int i;

    tmp = 0;
    file.input = fopen( filename, "rb" );

    if( file.input != NULL )
    {
        size_t elements_read1 = fread( &file.count, sizeof( file.count ), 1, file.input );
        size_t elements_read2 = fread( &file.vecsize, sizeof( file.vecsize ), 1, file.input );
        size_t elements_read3 = fread( &tmp, sizeof( tmp ), 1, file.input );
        size_t elements_read4 = fread( &tmp, sizeof( tmp ), 1, file.input );
        CV_Assert(elements_read1 == 1 && elements_read2 == 1 && elements_read3 == 1 && elements_read4 == 1);

        if( file.vecsize != winwidth * winheight )
        {
            int guessed_w = 0;
            int guessed_h = 0;

            fprintf( stderr, "Warning: specified sample width=%d and height=%d "
                "does not correspond to .vec file vector size=%d.\n",
                winwidth, winheight, file.vecsize );
            if( file.vecsize > 0 )
            {
                guessed_w = cvFloor( sqrt( (float) file.vecsize ) );
                if( guessed_w > 0 )
                {
                    guessed_h = file.vecsize / guessed_w;
                }
            }

            if( guessed_w <= 0 || guessed_h <= 0 || guessed_w * guessed_h != file.vecsize)
            {
                fprintf( stderr, "Error: failed to guess sample width and height\n" );
                fclose( file.input );

                return;
            }
            else
            {
                winwidth = guessed_w;
                winheight = guessed_h;
                fprintf( stderr, "Guessed width=%d, guessed height=%d\n",
                    winwidth, winheight );
            }
        }

        if( !feof( file.input ) && scale > 0 )
        {
            file.last = 0;
            namedWindow( "Sample", WINDOW_AUTOSIZE );
            for( i = 0; i < file.count; i++ )
            {
                Mat sample(winheight, winwidth, CV_8UC1);
                icvGetTraininDataFromVec( sample, file );
                if( scale != 1.0 )
                    resize( sample, sample,
                            Size(MAX(1, cvCeil(scale * winwidth)), MAX(1, cvCeil(scale * winheight))), 0, 0, INTER_LINEAR_EXACT);
                imshow( "Sample", sample );
                if( waitKey( 0 ) == 27 ) break;
            }
        }
        fclose( file.input );
    }
}
