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

/****************************************************************************************\
     Implementation of SIFT taken from http://blogs.oregonstate.edu/hess/code/sift/
\****************************************************************************************/

//    Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
//    All rights reserved.

//    The following patent has been issued for methods embodied in this
//    software: "Method and apparatus for identifying scale invariant features
//    in an image and use of same for locating an object in an image," David
//    G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application
//    filed March 8, 1999. Asignee: The University of British Columbia. For
//    further details, contact David Lowe (lowe@cs.ubc.ca) or the
//    University-Industry Liaison Office of the University of British
//    Columbia.

//    Note that restrictions imposed by this patent (and possibly others)
//    exist independently of and may be in conflict with the freedoms granted
//    in this license, which refers to copyright of the program, not patents
//    for any methods that it implements.  Both copyright and patent law must
//    be obeyed to legally use and redistribute this program and it is not the
//    purpose of this license to induce you to infringe any patents or other
//    property right claims or to contest validity of any such claims.  If you
//    redistribute or use the program, then this license merely protects you
//    from committing copyright infringement.  It does not protect you from
//    committing patent infringement.  So, before you do anything with this
//    program, make sure that you have permission to do so not merely in terms
//    of copyright, but also in terms of patent law.

//    Please note that this license is not to be understood as a guarantee
//    either.  If you use the program according to this license, but in
//    conflict with patent law, it does not mean that the licensor will refund
//    you for any losses that you incur if you are sued for your patent
//    infringement.

//    Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are
//    met:
//        * Redistributions of source code must retain the above copyright and
//          patent notices, this list of conditions and the following
//          disclaimer.
//        * Redistributions in binary form must reproduce the above copyright
//          notice, this list of conditions and the following disclaimer in
//          the documentation and/or other materials provided with the
//          distribution.
//        * Neither the name of Oregon State University nor the names of its
//          contributors may be used to endorse or promote products derived
//          from this software without specific prior written permission.

//    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
//    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//    HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "precomp.hpp"

const double a_180divPI = 180./CV_PI;
const double a_PIdiv180 = CV_PI/180.;

static inline float getOctaveSamplingPeriod( int o )
{
  return (o >= 0) ? (1 << o) : 1.0f / (1 << -o) ;
}

/*
  Interpolates a histogram peak from left, center, and right values
*/
#define interp_hist_peak( l, c, r ) ( 0.5 * ((l)-(r)) / ((l) - 2.0*(c) + (r)) )


// utils.h
/**
   A function to get a pixel value from a 32-bit floating-point image.

   @param img an image
   @param r row
   @param c column
   @return Returns the value of the pixel at (\a r, \a c) in \a img
*/
static inline float pixval32f( IplImage* img, int r, int c )
{
  return ( (float*)(img->imageData + img->widthStep*r) )[c];
}


// imgfeatures.h

/** holds feature data relevant to detection */
struct detection_data
{
    int r; // row
    int c; // col
    int octv;
    int intvl;
    double subintvl;
    double scl_octv;
};

/** max feature descriptor length */
#define FEATURE_MAX_D 128

/**
   Structure to represent an affine invariant image feature.  The fields
   x, y, a, b, c represent the affine region around the feature:

   a(x-u)(x-u) + 2b(x-u)(y-v) + c(y-v)(y-v) = 1
*/
struct feature
{
    double x;                      /**< x coord */
    double y;                      /**< y coord */

    double scl;                    /**< scale of a Lowe-style feature */
    double ori;                    /**< orientation of a Lowe-style feature */

    int d;                         /**< descriptor length */
    double descr[FEATURE_MAX_D];   /**< descriptor */

    detection_data* feature_data;            /**< user-definable data */
};

/******************************* Defs and macros *****************************/

/** default number of sampled intervals per octave */
#define SIFT_INTVLS 3

/** default sigma for initial gaussian smoothing */
#define SIFT_SIGMA 1.6

/** default threshold on keypoint contrast |D(x)| */
#define SIFT_CONTR_THR 0.04

/** default threshold on keypoint ratio of principle curvatures */
#define SIFT_CURV_THR 10

/** double image size before pyramid construction? */
#define SIFT_IMG_DBL 1

/** default width of descriptor histogram array */
#define SIFT_DESCR_WIDTH 4

/** default number of bins per histogram in descriptor array */
#define SIFT_DESCR_HIST_BINS 8

/* assumed gaussian blur for input image */
#define SIFT_INIT_SIGMA 0.5

/* width of border in which to ignore keypoints */
#define SIFT_IMG_BORDER 5

/* maximum steps of keypoint interpolation before failure */
#define SIFT_MAX_INTERP_STEPS 5

/* default number of bins in histogram for orientation assignment */
#define SIFT_ORI_HIST_BINS 36

/* determines gaussian sigma for orientation assignment */
#define SIFT_ORI_SIG_FCTR 1.5

/* determines the radius of the region used in orientation assignment */
#define SIFT_ORI_RADIUS 3.0 * SIFT_ORI_SIG_FCTR

/* number of passes of orientation histogram smoothing */
#define SIFT_ORI_SMOOTH_PASSES 2

/* orientation magnitude relative to max that results in new feature */
#define SIFT_ORI_PEAK_RATIO 0.8

/* determines the size of a single descriptor orientation histogram */
#define SIFT_DESCR_SCL_FCTR 3.0

/* threshold on magnitude of elements of descriptor vector */
#define SIFT_DESCR_MAG_THR 0.2

/* factor used to convert floating-point descriptor to unsigned char */
#define SIFT_INT_DESCR_FCTR 512.0

/**************************************************************************/

/*
  Converts an image to 32-bit grayscale

  @param img a 3-channel 8-bit color (BGR) or 8-bit gray image

  @return Returns a 32-bit grayscale image
*/
static IplImage* convert_to_gray32( IplImage* img )
{
  IplImage* gray8, * gray32;

  gray32 = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
  if( img->nChannels == 1 )
    gray8 = (IplImage*)cvClone( img );
  else
    {
      gray8 = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );
      cvCvtColor( img, gray8, CV_BGR2GRAY );
    }
  cvConvertScale( gray8, gray32, 1.0 / 255.0, 0 );

  cvReleaseImage( &gray8 );
  return gray32;
}

/*
  Converts an image to 8-bit grayscale and Gaussian-smooths it.  The image is
  optionally doubled in size prior to smoothing.

  @param img input image
  @param img_dbl if true, image is doubled in size prior to smoothing
  @param sigma total std of Gaussian smoothing
*/
static IplImage* create_init_img( IplImage* img, int img_dbl, double sigma )
{
  IplImage* gray, * dbl;
  double sig_diff;

  gray = convert_to_gray32( img );
  if( img_dbl )
    {
      sig_diff = sqrt( sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4 );
      dbl = cvCreateImage( cvSize( img->width*2, img->height*2 ),
                           IPL_DEPTH_32F, 1 );
      cvResize( gray, dbl, CV_INTER_CUBIC );
      cvSmooth( dbl, dbl, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff );
      cvReleaseImage( &gray );
      return dbl;
    }
  else
    {
      sig_diff = sqrt( sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA );
      cvSmooth( gray, gray, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff );
      return gray;
    }
}

/*
  Downsamples an image to a quarter of its size (half in each dimension)
  using nearest-neighbor interpolation

  @param img an image

  @return Returns an image whose dimensions are half those of img
*/
static IplImage* downsample( IplImage* img )
{
  IplImage* smaller = cvCreateImage( cvSize(img->width / 2, img->height / 2),
                                     img->depth, img->nChannels );
  cvResize( img, smaller, CV_INTER_NN );

  return smaller;
}

/*
  Builds Gaussian scale space pyramid from an image

  @param base base image of the pyramid
  @param octvs number of octaves of scale space
  @param intvls number of intervals per octave
  @param sigma amount of Gaussian smoothing per octave

  @return Returns a Gaussian scale space pyramid as an octvs x (intvls + 3)
    array
*/
static IplImage*** build_gauss_pyr( IplImage* base, int octvs,
                             int intvls, double sigma )
{
  IplImage*** gauss_pyr;
  const int _intvls = intvls;
  double sig[_intvls+3], sig_total, sig_prev, k;
  int i, o;

  gauss_pyr = (IplImage***)calloc( octvs, sizeof( IplImage** ) );
  for( i = 0; i < octvs; i++ )
    gauss_pyr[i] = (IplImage**)calloc( intvls + 3, sizeof( IplImage *) );

  /*
    precompute Gaussian sigmas using the following formula:

    \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
  */
  sig[0] = sigma;
  k = pow( 2.0, 1.0 / intvls );
  for( i = 1; i < intvls + 3; i++ )
    {
      sig_prev = pow( k, i - 1 ) * sigma;
      sig_total = sig_prev * k;
      sig[i] = sqrt( sig_total * sig_total - sig_prev * sig_prev );
    }

  for( o = 0; o < octvs; o++ )
    for( i = 0; i < intvls + 3; i++ )
      {
        if( o == 0  &&  i == 0 )
          gauss_pyr[o][i] = cvCloneImage(base);

        /* base of new octvave is halved image from end of previous octave */
        else if( i == 0 )
          gauss_pyr[o][i] = downsample( gauss_pyr[o-1][intvls] );

        /* blur the current octave's last image to create the next one */
        else
          {
            gauss_pyr[o][i] = cvCreateImage( cvGetSize(gauss_pyr[o][i-1]),
                                             IPL_DEPTH_32F, 1 );
            cvSmooth( gauss_pyr[o][i-1], gauss_pyr[o][i],
                      CV_GAUSSIAN, 0, 0, sig[i], sig[i] );
          }
      }

  return gauss_pyr;
}

/*
  Builds a difference of Gaussians scale space pyramid by subtracting adjacent
  intervals of a Gaussian pyramid

  @param gauss_pyr Gaussian scale-space pyramid
  @param octvs number of octaves of scale space
  @param intvls number of intervals per octave

  @return Returns a difference of Gaussians scale space pyramid as an
    octvs x (intvls + 2) array
*/
static IplImage*** build_dog_pyr( IplImage*** gauss_pyr, int octvs, int intvls )
{
  IplImage*** dog_pyr;
  int i, o;

  dog_pyr = (IplImage***)calloc( octvs, sizeof( IplImage** ) );
  for( i = 0; i < octvs; i++ )
    dog_pyr[i] = (IplImage**)calloc( intvls + 2, sizeof(IplImage*) );

  for( o = 0; o < octvs; o++ )
    for( i = 0; i < intvls + 2; i++ )
      {
        dog_pyr[o][i] = cvCreateImage( cvGetSize(gauss_pyr[o][i]),
                                       IPL_DEPTH_32F, 1 );
        cvSub( gauss_pyr[o][i+1], gauss_pyr[o][i], dog_pyr[o][i], NULL );
      }

  return dog_pyr;
}

/*
  Determines whether a pixel is a scale-space extremum by comparing it to it's
  3x3x3 pixel neighborhood.

  @param dog_pyr DoG scale space pyramid
  @param octv pixel's scale space octave
  @param intvl pixel's within-octave interval
  @param r pixel's image row
  @param c pixel's image col

  @return Returns 1 if the specified pixel is an extremum (max or min) among
    it's 3x3x3 pixel neighborhood.
*/
static int is_extremum( IplImage*** dog_pyr, int octv, int intvl, int r, int c )
{
  double val = pixval32f( dog_pyr[octv][intvl], r, c );
  int i, j, k;

  /* check for maximum */
  if( val > 0 )
    {
      for( i = -1; i <= 1; i++ )
        for( j = -1; j <= 1; j++ )
          for( k = -1; k <= 1; k++ )
            if( val < pixval32f( dog_pyr[octv][intvl+i], r + j, c + k ) )
              return 0;
    }

  /* check for minimum */
  else
    {
      for( i = -1; i <= 1; i++ )
        for( j = -1; j <= 1; j++ )
          for( k = -1; k <= 1; k++ )
            if( val > pixval32f( dog_pyr[octv][intvl+i], r + j, c + k ) )
              return 0;
    }

  return 1;
}

/*
  Computes the partial derivatives in x, y, and scale of a pixel in the DoG
  scale space pyramid.

  @param dog_pyr DoG scale space pyramid
  @param octv pixel's octave in dog_pyr
  @param intvl pixel's interval in octv
  @param r pixel's image row
  @param c pixel's image col

  @return Returns the vector of partial derivatives for pixel I
    { dI/dx, dI/dy, dI/ds }^T as a CvMat*
*/
static CvMat* deriv_3D( IplImage*** dog_pyr, int octv, int intvl, int r, int c )
{
  CvMat* dI;
  double dx, dy, ds;

  dx = ( pixval32f( dog_pyr[octv][intvl], r, c+1 ) -
         pixval32f( dog_pyr[octv][intvl], r, c-1 ) ) / 2.0;
  dy = ( pixval32f( dog_pyr[octv][intvl], r+1, c ) -
         pixval32f( dog_pyr[octv][intvl], r-1, c ) ) / 2.0;
  ds = ( pixval32f( dog_pyr[octv][intvl+1], r, c ) -
         pixval32f( dog_pyr[octv][intvl-1], r, c ) ) / 2.0;

  dI = cvCreateMat( 3, 1, CV_64FC1 );
  cvmSet( dI, 0, 0, dx );
  cvmSet( dI, 1, 0, dy );
  cvmSet( dI, 2, 0, ds );

  return dI;
}

/*
  Computes the 3D Hessian matrix for a pixel in the DoG scale space pyramid.

  @param dog_pyr DoG scale space pyramid
  @param octv pixel's octave in dog_pyr
  @param intvl pixel's interval in octv
  @param r pixel's image row
  @param c pixel's image col

  @return Returns the Hessian matrix (below) for pixel I as a CvMat*

  / Ixx  Ixy  Ixs \ <BR>
  | Ixy  Iyy  Iys | <BR>
  \ Ixs  Iys  Iss /
*/
static CvMat* hessian_3D( IplImage*** dog_pyr, int octv, int intvl, int r,
                          int c )
{
  CvMat* H;
  double v, dxx, dyy, dss, dxy, dxs, dys;

  v = pixval32f( dog_pyr[octv][intvl], r, c );
  dxx = ( pixval32f( dog_pyr[octv][intvl], r, c+1 ) +
          pixval32f( dog_pyr[octv][intvl], r, c-1 ) - 2 * v );
  dyy = ( pixval32f( dog_pyr[octv][intvl], r+1, c ) +
          pixval32f( dog_pyr[octv][intvl], r-1, c ) - 2 * v );
  dss = ( pixval32f( dog_pyr[octv][intvl+1], r, c ) +
          pixval32f( dog_pyr[octv][intvl-1], r, c ) - 2 * v );
  dxy = ( pixval32f( dog_pyr[octv][intvl], r+1, c+1 ) -
          pixval32f( dog_pyr[octv][intvl], r+1, c-1 ) -
          pixval32f( dog_pyr[octv][intvl], r-1, c+1 ) +
          pixval32f( dog_pyr[octv][intvl], r-1, c-1 ) ) / 4.0;
  dxs = ( pixval32f( dog_pyr[octv][intvl+1], r, c+1 ) -
          pixval32f( dog_pyr[octv][intvl+1], r, c-1 ) -
          pixval32f( dog_pyr[octv][intvl-1], r, c+1 ) +
          pixval32f( dog_pyr[octv][intvl-1], r, c-1 ) ) / 4.0;
  dys = ( pixval32f( dog_pyr[octv][intvl+1], r+1, c ) -
          pixval32f( dog_pyr[octv][intvl+1], r-1, c ) -
          pixval32f( dog_pyr[octv][intvl-1], r+1, c ) +
          pixval32f( dog_pyr[octv][intvl-1], r-1, c ) ) / 4.0;

  H = cvCreateMat( 3, 3, CV_64FC1 );
  cvmSet( H, 0, 0, dxx );
  cvmSet( H, 0, 1, dxy );
  cvmSet( H, 0, 2, dxs );
  cvmSet( H, 1, 0, dxy );
  cvmSet( H, 1, 1, dyy );
  cvmSet( H, 1, 2, dys );
  cvmSet( H, 2, 0, dxs );
  cvmSet( H, 2, 1, dys );
  cvmSet( H, 2, 2, dss );

  return H;
}

/*
  Performs one step of extremum interpolation.  Based on Eqn. (3) in Lowe's
  paper.

  @param dog_pyr difference of Gaussians scale space pyramid
  @param octv octave of scale space
  @param intvl interval being interpolated
  @param r row being interpolated
  @param c column being interpolated
  @param xi output as interpolated subpixel increment to interval
  @param xr output as interpolated subpixel increment to row
  @param xc output as interpolated subpixel increment to col
*/

static void interp_step( IplImage*** dog_pyr, int octv, int intvl, int r, int c,
                         double* xi, double* xr, double* xc )
{
  CvMat* dD, * H, * H_inv, X;
  double x[3] = { 0 };

  dD = deriv_3D( dog_pyr, octv, intvl, r, c );
  H = hessian_3D( dog_pyr, octv, intvl, r, c );
  H_inv = cvCreateMat( 3, 3, CV_64FC1 );
  cvInvert( H, H_inv, CV_SVD );
  cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
  cvGEMM( H_inv, dD, -1, NULL, 0, &X, 0 );

  cvReleaseMat( &dD );
  cvReleaseMat( &H );
  cvReleaseMat( &H_inv );

  *xi = x[2];
  *xr = x[1];
  *xc = x[0];
}

/*
  Calculates interpolated pixel contrast.  Based on Eqn. (3) in Lowe's
  paper.

  @param dog_pyr difference of Gaussians scale space pyramid
  @param octv octave of scale space
  @param intvl within-octave interval
  @param r pixel row
  @param c pixel column
  @param xi interpolated subpixel increment to interval
  @param xr interpolated subpixel increment to row
  @param xc interpolated subpixel increment to col

  @param Returns interpolated contrast.
*/
static double interp_contr( IplImage*** dog_pyr, int octv, int intvl, int r,
                            int c, double xi, double xr, double xc )
{
  CvMat* dD, X, T;
  double t[1], x[3] = { xc, xr, xi };

  cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
  cvInitMatHeader( &T, 1, 1, CV_64FC1, t, CV_AUTOSTEP );
  dD = deriv_3D( dog_pyr, octv, intvl, r, c );
  cvGEMM( dD, &X, 1, NULL, 0, &T,  CV_GEMM_A_T );
  cvReleaseMat( &dD );

  return pixval32f( dog_pyr[octv][intvl], r, c ) + t[0] * 0.5;
}

/*
  Allocates and initializes a new feature

  @return Returns a pointer to the new feature
*/
static struct feature* new_feature( void )
{
  struct feature* feat;
  struct detection_data* ddata;

  feat = (feature*) malloc( sizeof( struct feature ) );
  memset( feat, 0, sizeof( struct feature ) );
  ddata = (detection_data*) malloc( sizeof( struct detection_data ) );
  memset( ddata, 0, sizeof( struct detection_data ) );
  feat->feature_data = ddata;

  return feat;
}

/*
  Interpolates a scale-space extremum's location and scale to subpixel
  accuracy to form an image feature.  Rejects features with low contrast.
  Based on Section 4 of Lowe's paper.

  @param dog_pyr DoG scale space pyramid
  @param octv feature's octave of scale space
  @param intvl feature's within-octave interval
  @param r feature's image row
  @param c feature's image column
  @param intvls total intervals per octave
  @param contr_thr threshold on feature contrast

  @return Returns the feature resulting from interpolation of the given
    parameters or NULL if the given location could not be interpolated or
    if contrast at the interpolated loation was too low.  If a feature is
    returned, its scale, orientation, and descriptor are yet to be determined.
*/
static struct feature* interp_extremum( IplImage*** dog_pyr, int octv,
                                        int intvl, int r, int c, int intvls,
                                        double contr_thr )
{
  struct feature* feat;
  struct detection_data* ddata;
  double xi, xr, xc, contr;
  int i = 0;

  while( i < SIFT_MAX_INTERP_STEPS )
    {
      interp_step( dog_pyr, octv, intvl, r, c, &xi, &xr, &xc );
      if( std::abs( xi ) < 0.5  &&  std::abs( xr ) < 0.5  &&  std::abs( xc ) < 0.5 )
        break;

      c += cvRound( xc );
      r += cvRound( xr );
      intvl += cvRound( xi );

      if( intvl < 1  ||
          intvl > intvls  ||
          c < SIFT_IMG_BORDER  ||
          r < SIFT_IMG_BORDER  ||
          c >= dog_pyr[octv][0]->width - SIFT_IMG_BORDER  ||
          r >= dog_pyr[octv][0]->height - SIFT_IMG_BORDER )
        {
          return NULL;
        }

      i++;
    }

  /* ensure convergence of interpolation */
  if( i >= SIFT_MAX_INTERP_STEPS )
    return NULL;

  contr = interp_contr( dog_pyr, octv, intvl, r, c, xi, xr, xc );
  if( std::abs( contr ) < contr_thr / intvls )
    return NULL;

  feat = new_feature();
  ddata = feat->feature_data;
  feat->x = ( c + xc ) * pow( 2.0, octv );
  feat->y = ( r + xr ) * pow( 2.0, octv );
  ddata->r = r;
  ddata->c = c;
  ddata->octv = octv;
  ddata->intvl = intvl;
  ddata->subintvl = xi;

  return feat;
}

/*
  Determines whether a feature is too edge like to be stable by computing the
  ratio of principal curvatures at that feature.  Based on Section 4.1 of
  Lowe's paper.

  @param dog_img image from the DoG pyramid in which feature was detected
  @param r feature row
  @param c feature col
  @param curv_thr high threshold on ratio of principal curvatures

  @return Returns 0 if the feature at (r,c) in dog_img is sufficiently
    corner-like or 1 otherwise.
*/
static int is_too_edge_like( IplImage* dog_img, int r, int c, int curv_thr )
{
  double d, dxx, dyy, dxy, tr, det;

  /* principal curvatures are computed using the trace and det of Hessian */
  d = pixval32f(dog_img, r, c);
  dxx = pixval32f( dog_img, r, c+1 ) + pixval32f( dog_img, r, c-1 ) - 2 * d;
  dyy = pixval32f( dog_img, r+1, c ) + pixval32f( dog_img, r-1, c ) - 2 * d;
  dxy = ( pixval32f(dog_img, r+1, c+1) - pixval32f(dog_img, r+1, c-1) -
          pixval32f(dog_img, r-1, c+1) + pixval32f(dog_img, r-1, c-1) ) / 4.0;
  tr = dxx + dyy;
  det = dxx * dyy - dxy * dxy;

  /* negative determinant -> curvatures have different signs; reject feature */
  if( det <= 0 )
    return 1;

  if( tr * tr / det < ( curv_thr + 1.0 )*( curv_thr + 1.0 ) / curv_thr )
    return 0;
  return 1;
}

/*
  Detects features at extrema in DoG scale space.  Bad features are discarded
  based on contrast and ratio of principal curvatures.

  @param dog_pyr DoG scale space pyramid
  @param octvs octaves of scale space represented by dog_pyr
  @param intvls intervals per octave
  @param contr_thr low threshold on feature contrast
  @param curv_thr high threshold on feature ratio of principal curvatures
  @param storage memory storage in which to store detected features

  @return Returns an array of detected features whose scales, orientations,
    and descriptors are yet to be determined.
*/
static CvSeq* scale_space_extrema( IplImage*** dog_pyr, int octvs, int intvls,
                                   double contr_thr, int curv_thr,
                                   CvMemStorage* storage )
{
  CvSeq* features;
  double prelim_contr_thr = 0.5 * contr_thr / intvls;
  struct feature* feat;
  struct detection_data* ddata;
  int o, i, r, c;

  features = cvCreateSeq( 0, sizeof(CvSeq), sizeof(struct feature), storage );
  for( o = 0; o < octvs; o++ )
    for( i = 1; i <= intvls; i++ )
      for(r = SIFT_IMG_BORDER; r < dog_pyr[o][0]->height-SIFT_IMG_BORDER; r++)
        for(c = SIFT_IMG_BORDER; c < dog_pyr[o][0]->width-SIFT_IMG_BORDER; c++)
          /* perform preliminary check on contrast */
          if( std::abs( pixval32f( dog_pyr[o][i], r, c ) ) > prelim_contr_thr )
            if( is_extremum( dog_pyr, o, i, r, c ) )
              {
                feat = interp_extremum(dog_pyr, o, i, r, c, intvls, contr_thr);
                if( feat )
                  {
                    ddata = feat->feature_data;
                    if( ! is_too_edge_like( dog_pyr[ddata->octv][ddata->intvl],
                                            ddata->r, ddata->c, curv_thr ) )
                      {
                        cvSeqPush( features, feat );
                      }
                    else
                      free( ddata );
                    free( feat );
                  }
              }

  return features;
}

/*
  Calculates characteristic scale for each feature in an array.

  @param features array of features
  @param sigma amount of Gaussian smoothing per octave of scale space
  @param intvls intervals per octave of scale space
*/
static void calc_feature_scales( CvSeq* features, double sigma, int intvls )
{
  struct feature* feat;
  struct detection_data* ddata;
  double intvl;
  int i, n;

  n = features->total;
  for( i = 0; i < n; i++ )
    {
      feat = CV_GET_SEQ_ELEM( struct feature, features, i );
      ddata = feat->feature_data;
      intvl = ddata->intvl + ddata->subintvl;
      feat->scl = sigma * pow( 2.0, ddata->octv + intvl / intvls );
      ddata->scl_octv = sigma * pow( 2.0, intvl / intvls );
    }
}



/*
  Halves feature coordinates and scale in case the input image was doubled
  prior to scale space construction.

  @param features array of features
*/
static void adjust_for_img_dbl( CvSeq* features )
{
  struct feature* feat;
  int i, n;

  n = features->total;
  for( i = 0; i < n; i++ )
    {
      feat = CV_GET_SEQ_ELEM( struct feature, features, i );
      feat->x /= 2.0;
      feat->y /= 2.0;
      feat->scl /= 2.0;
    }
}

/*
  Calculates the gradient magnitude and orientation at a given pixel.

  @param img image
  @param r pixel row
  @param c pixel col
  @param mag output as gradient magnitude at pixel (r,c)
  @param ori output as gradient orientation at pixel (r,c)

  @return Returns 1 if the specified pixel is a valid one and sets mag and
    ori accordingly; otherwise returns 0
*/
static int calc_grad_mag_ori( IplImage* img, int r, int c, double* mag,
                              double* ori )
{
  double dx, dy;

  if( r > 0  &&  r < img->height - 1  &&  c > 0  &&  c < img->width - 1 )
    {
      dx = pixval32f( img, r, c+1 ) - pixval32f( img, r, c-1 );
      dy = pixval32f( img, r-1, c ) - pixval32f( img, r+1, c );
      *mag = sqrt( dx*dx + dy*dy );
      *ori = atan2( dy, dx );
      return 1;
    }

  else
    return 0;
}

/*
  Computes a gradient orientation histogram at a specified pixel.

  @param img image
  @param r pixel row
  @param c pixel col
  @param n number of histogram bins
  @param rad radius of region over which histogram is computed
  @param sigma std for Gaussian weighting of histogram entries

  @return Returns an n-element array containing an orientation histogram
    representing orientations between 0 and 2 PI.
*/
static double* ori_hist( IplImage* img, int r, int c, int n, int rad,
                         double sigma )
{
  double* hist;
  double mag, ori, w, exp_denom, PI2 = CV_PI * 2.0;
  int bin, i, j;

  hist = (double*) calloc( n, sizeof( double ) );
  exp_denom = 2.0 * sigma * sigma;
  for( i = -rad; i <= rad; i++ )
    for( j = -rad; j <= rad; j++ )
      if( calc_grad_mag_ori( img, r + i, c + j, &mag, &ori ) )
        {
          w = exp( -( i*i + j*j ) / exp_denom );
          bin = cvRound( n * ( ori + CV_PI ) / PI2 );
          bin = ( bin < n )? bin : 0;
          hist[bin] += w * mag;
        }

  return hist;
}

/*
  Gaussian smooths an orientation histogram.

  @param hist an orientation histogram
  @param n number of bins
*/
static void smooth_ori_hist( double* hist, int n )
{
  double prev, tmp, h0 = hist[0];
  int i;

  prev = hist[n-1];
  for( i = 0; i < n; i++ )
    {
      tmp = hist[i];
      hist[i] = 0.25 * prev + 0.5 * hist[i] +
        0.25 * ( ( i+1 == n )? h0 : hist[i+1] );
      prev = tmp;
    }
}

/*
  Finds the magnitude of the dominant orientation in a histogram

  @param hist an orientation histogram
  @param n number of bins

  @return Returns the value of the largest bin in hist
*/
static double dominant_ori( double* hist, int n )
{
  double omax;
  int maxbin, i;

  omax = hist[0];
  maxbin = 0;
  for( i = 1; i < n; i++ )
    if( hist[i] > omax )
      {
        omax = hist[i];
        maxbin = i;
      }
  return omax;
}

/*
  Makes a deep copy of a feature

  @param feat feature to be cloned

  @return Returns a deep copy of feat
*/
static struct feature* clone_feature( struct feature* feat )
{
  struct feature* new_feat;
  struct detection_data* ddata;

  new_feat = new_feature();
  ddata = new_feat->feature_data;
  memcpy( new_feat, feat, sizeof( struct feature ) );
  memcpy( ddata, feat->feature_data, sizeof( struct detection_data ) );
  new_feat->feature_data = ddata;

  return new_feat;
}

/*
  Adds features to an array for every orientation in a histogram greater than
  a specified threshold.

  @param features new features are added to the end of this array
  @param hist orientation histogram
  @param n number of bins in hist
  @param mag_thr new features are added for entries in hist greater than this
  @param feat new features are clones of this with different orientations
*/
static void add_good_ori_features( CvSeq* features, double* hist, int n,
                                   double mag_thr, struct feature* feat )
{
  struct feature* new_feat;
  double bin, PI2 = CV_PI * 2.0;
  int l, r, i;

  for( i = 0; i < n; i++ )
    {
      l = ( i == 0 )? n - 1 : i-1;
      r = ( i + 1 ) % n;

      if( hist[i] > hist[l]  &&  hist[i] > hist[r]  &&  hist[i] >= mag_thr )
        {
          bin = i + interp_hist_peak( hist[l], hist[i], hist[r] );
          bin = ( bin < 0 )? n + bin : ( bin >= n )? bin - n : bin;
          new_feat = clone_feature( feat );
          new_feat->ori = ( ( PI2 * bin ) / n ) - CV_PI;
          cvSeqPush( features, new_feat );
          free( new_feat );
        }
    }
}

/*
  Computes a canonical orientation for each image feature in an array.  Based
  on Section 5 of Lowe's paper.  This function adds features to the array when
  there is more than one dominant orientation at a given feature location.

  @param features an array of image features
  @param gauss_pyr Gaussian scale space pyramid
*/
static void calc_feature_oris( CvSeq* features, IplImage*** gauss_pyr )
{
  struct feature* feat;
  struct detection_data* ddata;
  double* hist;
  double omax;
  int i, j, n = features->total;

  for( i = 0; i < n; i++ )
    {
      feat = (feature*) malloc( sizeof( struct feature ) );
      cvSeqPopFront( features, feat );
      ddata = feat->feature_data;
      hist = ori_hist( gauss_pyr[ddata->octv][ddata->intvl],
                       ddata->r, ddata->c, SIFT_ORI_HIST_BINS,
                       cvRound( SIFT_ORI_RADIUS * ddata->scl_octv ),
                       SIFT_ORI_SIG_FCTR * ddata->scl_octv );
      for( j = 0; j < SIFT_ORI_SMOOTH_PASSES; j++ )
        smooth_ori_hist( hist, SIFT_ORI_HIST_BINS );
      omax = dominant_ori( hist, SIFT_ORI_HIST_BINS );
      add_good_ori_features( features, hist, SIFT_ORI_HIST_BINS,
                             omax * SIFT_ORI_PEAK_RATIO, feat );
      free( ddata );
      free( feat );
      free( hist );
    }
}

/*
  Interpolates an entry into the array of orientation histograms that form
  the feature descriptor.

  @param hist 2D array of orientation histograms
  @param rbin sub-bin row coordinate of entry
  @param cbin sub-bin column coordinate of entry
  @param obin sub-bin orientation coordinate of entry
  @param mag size of entry
  @param d width of 2D array of orientation histograms
  @param n number of bins per orientation histogram
*/
static void interp_hist_entry( double*** hist, double rbin, double cbin,
                               double obin, double mag, int d, int n )
{
  double d_r, d_c, d_o, v_r, v_c, v_o;
  double** row, * h;
  int r0, c0, o0, rb, cb, ob, r, c, o;

  r0 = cvFloor( rbin );
  c0 = cvFloor( cbin );
  o0 = cvFloor( obin );
  d_r = rbin - r0;
  d_c = cbin - c0;
  d_o = obin - o0;

  /*
    The entry is distributed into up to 8 bins.  Each entry into a bin
    is multiplied by a weight of 1 - d for each dimension, where d is the
    distance from the center value of the bin measured in bin units.
  */
  for( r = 0; r <= 1; r++ )
    {
      rb = r0 + r;
      if( rb >= 0  &&  rb < d )
        {
          v_r = mag * ( ( r == 0 )? 1.0 - d_r : d_r );
          row = hist[rb];
          for( c = 0; c <= 1; c++ )
            {
              cb = c0 + c;
              if( cb >= 0  &&  cb < d )
                {
                  v_c = v_r * ( ( c == 0 )? 1.0 - d_c : d_c );
                  h = row[cb];
                  for( o = 0; o <= 1; o++ )
                    {
                      ob = ( o0 + o ) % n;
                      v_o = v_c * ( ( o == 0 )? 1.0 - d_o : d_o );
                      h[ob] += v_o;
                    }
                }
            }
        }
    }
}

/*
  Computes the 2D array of orientation histograms that form the feature
  descriptor.  Based on Section 6.1 of Lowe's paper.

  @param img image used in descriptor computation
  @param r row coord of center of orientation histogram array
  @param c column coord of center of orientation histogram array
  @param ori canonical orientation of feature whose descr is being computed
  @param scl scale relative to img of feature whose descr is being computed
  @param d width of 2d array of orientation histograms
  @param n bins per orientation histogram

  @return Returns a d x d array of n-bin orientation histograms.
*/
static double*** descr_hist( IplImage* img, int r, int c, double ori,
                             double scl, int d, int n )
{
  double*** hist;
  double cos_t, sin_t, hist_width, exp_denom, r_rot, c_rot, grad_mag,
    grad_ori, w, rbin, cbin, obin, bins_per_rad, PI2 = 2.0 * CV_PI;
  int radius, i, j;

  hist = (double***) calloc( d, sizeof( double** ) );
  for( i = 0; i < d; i++ )
    {
      hist[i] = (double**) calloc( d, sizeof( double* ) );
      for( j = 0; j < d; j++ )
        hist[i][j] = (double*) calloc( n, sizeof( double ) );
    }

  cos_t = cos( ori );
  sin_t = sin( ori );
  bins_per_rad = n / PI2;
  exp_denom = d * d * 0.5;
  hist_width = SIFT_DESCR_SCL_FCTR * scl;
  radius = hist_width * sqrt(2) * ( d + 1.0 ) * 0.5 + 0.5;
  for( i = -radius; i <= radius; i++ )
    for( j = -radius; j <= radius; j++ )
      {
        /*
          Calculate sample's histogram array coords rotated relative to ori.
          Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
          r_rot = 1.5) have full weight placed in row 1 after interpolation.
        */
        c_rot = ( j * cos_t - i * sin_t ) / hist_width;
        r_rot = ( j * sin_t + i * cos_t ) / hist_width;
        rbin = r_rot + d / 2 - 0.5;
        cbin = c_rot + d / 2 - 0.5;

        if( rbin > -1.0  &&  rbin < d  &&  cbin > -1.0  &&  cbin < d )
          if( calc_grad_mag_ori( img, r + i, c + j, &grad_mag, &grad_ori ))
            {
              grad_ori -= ori;
              while( grad_ori < 0.0 )
                grad_ori += PI2;
              while( grad_ori >= PI2 )
                grad_ori -= PI2;

              obin = grad_ori * bins_per_rad;
              w = exp( -(c_rot * c_rot + r_rot * r_rot) / exp_denom );
              interp_hist_entry( hist, rbin, cbin, obin, grad_mag * w, d, n );
            }
      }

  return hist;
}

/*
  Normalizes a feature's descriptor vector to unitl length

  @param feat feature
*/
static void normalize_descr( struct feature* feat )
{
  double cur, len_inv, len_sq = 0.0;
  int i, d = feat->d;

  for( i = 0; i < d; i++ )
    {
      cur = feat->descr[i];
      len_sq += cur*cur;
    }
  len_inv = 1.0 / sqrt( len_sq );
  for( i = 0; i < d; i++ )
    feat->descr[i] *= len_inv;
}

/*
  Converts the 2D array of orientation histograms into a feature's descriptor
  vector.

  @param hist 2D array of orientation histograms
  @param d width of hist
  @param n bins per histogram
  @param feat feature into which to store descriptor
*/
static void hist_to_descr( double*** hist, int d, int n, struct feature* feat )
{
  int int_val, i, r, c, o, k = 0;

  for( r = 0; r < d; r++ )
    for( c = 0; c < d; c++ )
      for( o = 0; o < n; o++ )
        feat->descr[k++] = hist[r][c][o];

  feat->d = k;
  normalize_descr( feat );
  for( i = 0; i < k; i++ )
    if( feat->descr[i] > SIFT_DESCR_MAG_THR )
      feat->descr[i] = SIFT_DESCR_MAG_THR;
  normalize_descr( feat );

  /* convert floating-point descriptor to integer valued descriptor */
  for( i = 0; i < k; i++ )
    {
      int_val = SIFT_INT_DESCR_FCTR * feat->descr[i];
      feat->descr[i] = MIN( 255, int_val );
    }
}

/*
  Compares features for a decreasing-scale ordering.  Intended for use with
  CvSeqSort

  @param feat1 first feature
  @param feat2 second feature
  @param param unused

  @return Returns 1 if feat1's scale is greater than feat2's, -1 if vice versa,
    and 0 if their scales are equal
*/
static int feature_cmp( void* feat1, void* feat2, void* param )
{
  struct feature* f1 = (struct feature*) feat1;
  struct feature* f2 = (struct feature*) feat2;

  if( f1->scl < f2->scl )
    return 1;
  if( f1->scl > f2->scl )
    return -1;
  return 0;
}

/*
  De-allocates memory held by a descriptor histogram

  @param hist pointer to a 2D array of orientation histograms
  @param d width of hist
*/
static void release_descr_hist( double**** hist, int d )
{
  int i, j;

  for( i = 0; i < d; i++)
    {
      for( j = 0; j < d; j++ )
        free( (*hist)[i][j] );
      free( (*hist)[i] );
    }
  free( *hist );
  *hist = NULL;
}


/*
  De-allocates memory held by a scale space pyramid

  @param pyr scale space pyramid
  @param octvs number of octaves of scale space
  @param n number of images per octave
*/
static void release_pyr( IplImage**** pyr, int octvs, int n )
{
  int i, j;
  for( i = 0; i < octvs; i++ )
    {
      for( j = 0; j < n; j++ )
        cvReleaseImage( &(*pyr)[i][j] );
      free( (*pyr)[i] );
    }
  free( *pyr );
  *pyr = NULL;
}

/*
  Computes feature descriptors for features in an array.  Based on Section 6
  of Lowe's paper.

  @param features array of features
  @param gauss_pyr Gaussian scale space pyramid
  @param d width of 2D array of orientation histograms
  @param n number of bins per orientation histogram
*/
static void compute_descriptors( CvSeq* features, IplImage*** gauss_pyr, int d,
                                 int n )
{
  struct feature* feat;
  struct detection_data* ddata;
  double*** hist;
  int i, k = features->total;

  for( i = 0; i < k; i++ )
    {
      feat = CV_GET_SEQ_ELEM( struct feature, features, i );
      ddata = feat->feature_data;
      hist = descr_hist( gauss_pyr[ddata->octv][ddata->intvl], ddata->r,
                         ddata->c, feat->ori, ddata->scl_octv, d, n );
      hist_to_descr( hist, d, n, feat );
      release_descr_hist( &hist, d );
    }
}

/***** some auxilary stucture (there is not it in original implementation) *******/

struct ImagePyrData
{
    ImagePyrData( IplImage* img, int octvs, int intvls, double _sigma, int img_dbl )
    {
        if( ! img )
          CV_Error( CV_StsBadArg, "NULL image pointer" );

        /* build scale space pyramid; smallest dimension of top level is ~4 pixels */
        init_img = create_init_img( img, img_dbl, _sigma );

        int max_octvs = log( MIN( init_img->width, init_img->height ) ) / log(2) - 2;
        octvs = std::max( std::min( octvs, max_octvs ), 1 );

        gauss_pyr = build_gauss_pyr( init_img, octvs, intvls, _sigma );
        dog_pyr = build_dog_pyr( gauss_pyr, octvs, intvls );

        octaves = octvs;
        intervals = intvls;
        sigma = _sigma;
        is_img_dbl = img_dbl != 0 ? true : false;
    }

    virtual ~ImagePyrData()
    {
        cvReleaseImage( &init_img );
        release_pyr( &gauss_pyr, octaves, intervals + 3 );
        release_pyr( &dog_pyr, octaves, intervals + 2 );
    }

    IplImage* init_img;
    IplImage*** gauss_pyr, *** dog_pyr;

    int octaves, intervals;
    double sigma;

    bool is_img_dbl;
};

void release_features( struct feature** feat, int count )
{
    for( int i = 0; i < count; i++ )
    {
        free( (*feat)[i].feature_data );
        (*feat)[i].feature_data = NULL;
    }
    free( *feat );
}

void compute_features( const ImagePyrData* imgPyrData, struct feature** feat, int& count,
                       double contr_thr, int curv_thr )
{
    CvMemStorage* storage;
    CvSeq* features;

    storage = cvCreateMemStorage( 0 );
    features = scale_space_extrema( imgPyrData->dog_pyr, imgPyrData->octaves, imgPyrData->intervals,
                                    contr_thr, curv_thr, storage );

    calc_feature_scales( features, imgPyrData->sigma, imgPyrData->intervals );
    if( imgPyrData->is_img_dbl )
      adjust_for_img_dbl( features );
    calc_feature_oris( features, imgPyrData->gauss_pyr );

    /* sort features by decreasing scale and move from CvSeq to array */
    cvSeqSort( features, (CvCmpFunc)feature_cmp, NULL );
    int n = features->total;
    *feat = (feature*)calloc( n, sizeof(struct feature) );
    *feat = (feature*)cvCvtSeqToArray( features, *feat, CV_WHOLE_SEQ );

    cvReleaseMemStorage( &storage );

    count = n;
}

/****************************************************************************************\
  2.) wrapper of Rob Hess`s SIFT
\****************************************************************************************/

using namespace cv;

SIFT::CommonParams::CommonParams() :
        nOctaves(DEFAULT_NOCTAVES), nOctaveLayers(DEFAULT_NOCTAVE_LAYERS),
        firstOctave(DEFAULT_FIRST_OCTAVE), angleMode(FIRST_ANGLE)
{}

SIFT::CommonParams::CommonParams( int _nOctaves, int _nOctaveLayers, int /*_firstOctave*/, int /*_angleMode*/ ) :
        nOctaves(_nOctaves), nOctaveLayers(_nOctaveLayers),
        firstOctave(-1/*_firstOctave*/), angleMode(FIRST_ANGLE/*_angleMode*/)
{}

SIFT::DetectorParams::DetectorParams() :
        threshold(GET_DEFAULT_THRESHOLD()), edgeThreshold(GET_DEFAULT_EDGE_THRESHOLD())
{}

SIFT::DetectorParams::DetectorParams( double _threshold, double _edgeThreshold ) :
        threshold(_threshold), edgeThreshold(_edgeThreshold)
{}

SIFT::DescriptorParams::DescriptorParams() :
        magnification(GET_DEFAULT_MAGNIFICATION()), isNormalize(DEFAULT_IS_NORMALIZE),
        recalculateAngles(true)
{}

SIFT::DescriptorParams::DescriptorParams( double _magnification, bool /*_isNormalize*/, bool _recalculateAngles ) :
        magnification(_magnification), isNormalize(true/*_isNormalize*/),
        recalculateAngles(_recalculateAngles)
{}

SIFT::DescriptorParams::DescriptorParams( bool _recalculateAngles )
    : magnification(GET_DEFAULT_MAGNIFICATION()), isNormalize(true),recalculateAngles(_recalculateAngles)
{}

SIFT::SIFT()
{}

SIFT::SIFT( double _threshold, double _edgeThreshold, int _nOctaves,
            int _nOctaveLayers, int _firstOctave, int _angleMode )
{
    detectorParams = DetectorParams(_threshold, _edgeThreshold);
    commParams = CommonParams(_nOctaves, _nOctaveLayers, _firstOctave, _angleMode);
}

SIFT::SIFT( double _magnification, bool _isNormalize, bool _recalculateAngles, int _nOctaves,
            int _nOctaveLayers, int _firstOctave, int _angleMode )
{
    descriptorParams = DescriptorParams(_magnification, _isNormalize, _recalculateAngles);
    commParams = CommonParams(_nOctaves, _nOctaveLayers, _firstOctave, _angleMode);
}

SIFT::SIFT( const CommonParams& _commParams,
            const DetectorParams& _detectorParams,
            const DescriptorParams& _descriptorParams )
{
    commParams = _commParams;
    detectorParams = _detectorParams;
    descriptorParams = _descriptorParams;
}

int SIFT::descriptorSize() const
{
    return DescriptorParams::DESCRIPTOR_SIZE;
}

SIFT::CommonParams SIFT::getCommonParams () const
{
    return commParams;
}

SIFT::DetectorParams SIFT::getDetectorParams () const
{
    return detectorParams;
}

SIFT::DescriptorParams SIFT::getDescriptorParams () const
{
    return descriptorParams;
}

struct SiftParams
{
    SiftParams( int _O, int _S )
    {
        O = _O;
        S = _S;

        sigma0 = 1.6 * powf(2.0f, 1.0f / S ) ;

        omin = -1;
        smin = -1;
        smax = S + 1;
    }

    int O;
    int S;

    double sigma0;

    int omin;
    int smin;
    int smax;
};

inline KeyPoint featureToKeyPoint( const feature& feat )
{
    float size = feat.scl * SIFT::DescriptorParams::GET_DEFAULT_MAGNIFICATION() * 4; // 4==NBP
    float angle = feat.ori * a_180divPI;
    return KeyPoint( feat.x, feat.y, size, angle, 0, feat.feature_data->octv, 0 );
}

static void fillFeatureData( feature& feat, const SiftParams& params )
{

  /*
    The formula linking the keypoint scale sigma to the octave and
    scale index is

    (1) sigma(o,s) = sigma0 2^(o+s/S)

    for which

    (2) o + s/S = log2 sigma/sigma0 == phi.

    In addition to the scale index s (which can be fractional due to
    scale interpolation) a keypoint has an integer scale index is too
    (which is the index of the scale level where it was detected in
    the DoG scale space). We have the constraints:

    - o and is are integer

    - is is in the range [smin+1, smax-2  ]

    - o  is in the range [omin,   omin+O-1]

    - is = rand(s) most of the times (but not always, due to the way s
      is obtained by quadratic interpolation of the DoG scale space).

    Depending on the values of smin and smax, often (2) has multiple
    solutions is,o that satisfy all constraints.  In this case we
    choose the one with biggest index o (this saves a bit of
    computation).

    DETERMINING THE OCTAVE INDEX O

    From (2) we have o = phi - s/S and we want to pick the biggest
    possible index o in the feasible range. This corresponds to
    selecting the smallest possible index s. We write s = is + ds
    where in most cases |ds|<.5 (but in general |ds|<1). So we have

       o = phi - s/S,   s = is + ds ,   |ds| < .5 (or |ds| < 1).

    Since is is in the range [smin+1,smax-2], s is in the range
    [smin+.5,smax-1.5] (or [smin,smax-1]), the number o is an integer
    in the range phi+[-smax+1.5,-smin-.5] (or
    phi+[-smax+1,-smin]). Thus the maximum value of o is obtained for
    o = floor(phi-smin-.5) (or o = floor(phi-smin)).

    Finally o is clamped to make sure it is contained in the feasible
    range.

    DETERMINING THE SCALE INDEXES S AND IS

    Given o we can derive is by writing (2) as

      s = is + ds = S(phi - o).

    We then take is = round(s) and clamp its value to be in the
    feasible range.
  */

  double sigma = feat.scl;
  double x = feat.x;
  double y = feat.y;

  int o, ix, iy, is;
  float s, phi;

  phi = log2( sigma / params.sigma0 ) ;
  o = std::floor( phi -  (float(params.smin)+.5)/params.S );
  o = std::min(o, params.omin+params.O-1);
  o = std::max(o, params.omin);
  s = params.S * (phi - o);

  is = int(s + 0.5);
  is = std::min(is, params.smax - 2);
  is = std::max(is, params.smin + 1);

  float per = getOctaveSamplingPeriod(o) ;
  ix = int(x / per + 0.5) ;
  iy = int(y / per + 0.5) ;


  detection_data* ddata = feat.feature_data;

  ddata->r = iy;
  ddata->c = ix;

  ddata->octv = o + 1;
  ddata->intvl = is + 1;

  ddata->subintvl = s - is;
  ddata->scl_octv = params.sigma0 * pow(2.0, s / params.S);
}

inline void keyPointToFeature( const KeyPoint& keypoint, feature& feat, const SiftParams& params )
{
    feat.x = keypoint.pt.x;
    feat.y = keypoint.pt.y;

    feat.scl = keypoint.size / (SIFT::DescriptorParams::GET_DEFAULT_MAGNIFICATION()*4); // 4==NBP
    feat.ori = keypoint.angle * a_PIdiv180;

    feat.feature_data = (detection_data*) calloc( 1, sizeof( detection_data ) );
    fillFeatureData( feat, params );
}

// detectors
void SIFT::operator()(const Mat& image, const Mat& mask,
                      vector<KeyPoint>& keypoints) const
{
    if( image.empty() || image.type() != CV_8UC1 )
        CV_Error( CV_StsBadArg, "image is empty or has incorrect type (!=CV_8UC1)" );

    if( !mask.empty() && mask.type() != CV_8UC1 )
        CV_Error( CV_StsBadArg, "mask has incorrect type (!=CV_8UC1)" );

    Mat subImage, subMask;
    Rect brect( 0, 0, image.cols, image.rows );
    if( mask.empty() )
    {
        subImage = image;
    }
    else
    {
        vector<Point> points;
        points.reserve( image.rows * image.cols );
        for( int y = 0; y < mask.rows; y++ )
        {
            for( int x = 0; x < mask.cols; x++ )
            {
                if( mask.at<uchar>(y,x) )
                    points.push_back( cv::Point(x,y) );
            }
        }
        brect = cv::boundingRect( points );

        if( brect.x == 0 && brect.y == 0 && brect.width == mask.cols && brect.height == mask.rows )
        {
            subImage = image;
        }
        else
        {
            subImage = image( brect );
            subMask = mask( brect );
        }
    }

    Mat fimg;
    subImage.convertTo( fimg, CV_32FC1 );

    // compute features
    IplImage img = fimg;
    struct feature* features;

    ImagePyrData pyrImages( &img, commParams.nOctaves, commParams.nOctaveLayers, SIFT_SIGMA, SIFT_IMG_DBL );

    int feature_count = 0;
    compute_features( &pyrImages, &features, feature_count, detectorParams.threshold, detectorParams.edgeThreshold );

    // convert to KeyPoint structure
    keypoints.resize( feature_count );
    for( int i = 0; i < feature_count; i++ )
    {
        keypoints[i] = featureToKeyPoint( features[i] );
    }
    release_features( &features, feature_count );

    KeyPointsFilter::removeDuplicated( keypoints );

    if( !subMask.empty() )
    {
        // filter points by subMask and convert the points coordinates from subImage size to image size
        KeyPointsFilter::runByPixelsMask( keypoints, subMask );
        int dx = brect.x, dy = brect.y;
        for( vector<KeyPoint>::iterator it = keypoints.begin(); it != keypoints.end(); ++it )
        {
            it->pt.x += dx;
            it->pt.y += dy;
        }
    }
}

// descriptors
void SIFT::operator()(const Mat& image, const Mat& mask,
                      vector<KeyPoint>& keypoints,
                      Mat& descriptors,
                      bool useProvidedKeypoints) const
{
    if( image.empty() || image.type() != CV_8UC1 )
        CV_Error( CV_StsBadArg, "img is empty or has incorrect type" );

    Mat fimg;
    image.convertTo(fimg, CV_32FC1/*, 1.0/255.0*/);

    if( !useProvidedKeypoints )
        (*this)(image, mask, keypoints);
    else
    {
        // filter keypoints by mask
        KeyPointsFilter::runByPixelsMask( keypoints, mask );
    }

    IplImage img = fimg;
    ImagePyrData pyrImages( &img, commParams.nOctaves, commParams.nOctaveLayers, SIFT_SIGMA, SIFT_IMG_DBL );

    // Calculate orientation of features.
    // Note: calc_feature_oris() duplicates the points with several dominant orientations.
    // So if keypoints was detected by Sift feature detector then some points will be
    // duplicated twice.
    CvMemStorage* storage = cvCreateMemStorage( 0 );
    CvSeq* featuresSeq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(struct feature), storage );

    if( descriptorParams.recalculateAngles )
    {
        for( size_t i = 0; i < keypoints.size(); i++ )
        {
            feature* ft = (feature*) calloc( 1, sizeof( struct feature ) );
            keyPointToFeature( keypoints[i], *ft, SiftParams( commParams.nOctaves, commParams.nOctaveLayers ) );
            cvSeqPush( featuresSeq, ft );
        }
        calc_feature_oris( featuresSeq, pyrImages.gauss_pyr );

        keypoints.resize( featuresSeq->total );
        for( int i = 0; i < featuresSeq->total; i++ )
        {
            feature * ft = CV_GET_SEQ_ELEM( feature, featuresSeq, i );
            keypoints[i] = featureToKeyPoint( *ft );
        }

        // Remove duplicated keypoints.
        KeyPointsFilter::removeDuplicated( keypoints );

        // Compute descriptors.
        cvSeqRemoveSlice( featuresSeq, cvSlice(0, featuresSeq->total) );
    }

    for( size_t i = 0; i < keypoints.size(); i++ )
    {
        feature* ft = (feature*) calloc( 1, sizeof( struct feature ) );
        keyPointToFeature( keypoints[i], *ft, SiftParams( commParams.nOctaves, commParams.nOctaveLayers ) );
        cvSeqPush( featuresSeq, ft );
    }
    compute_descriptors( featuresSeq, pyrImages.gauss_pyr, SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS );
    CV_DbgAssert( (int)keypoints.size() == featuresSeq->total );
    // TODO check that keypoint fiels is the same as before compute_descriptors()

    descriptors.create( featuresSeq->total, SIFT::DescriptorParams::DESCRIPTOR_SIZE, CV_32FC1 );
    for( int i = 0; i < featuresSeq->total; i++ )
    {
        float* rowPtr = descriptors.ptr<float>(i);
        feature * featurePtr = CV_GET_SEQ_ELEM( feature, featuresSeq, i );
        CV_Assert( featurePtr );
        double* desc = featurePtr->descr;
        for( int j = 0; j < descriptors.cols; j++ )
        {
            rowPtr[j] = (float)desc[j];
        }
    }

    cvReleaseMemStorage( &storage );
}
