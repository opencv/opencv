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

#include "cvtest.h"

#if 0 /* avoid this while a substitution for IPL DCT is not ready */

#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <float.h>

static char* funcs[] =
{
    "cvImgToObs_DCT"
};

static char *test_desc[] =
{
    "Comparing against IPL DCT"
};

/* actual parameters */
static int min_img_size, max_img_size;
static int max_dct_size;
static int base_iters;

static int init_hmm_obs_params = 0;

static const int img8u_range = 256;

static void read_hmm_obs_params( void )
{
    if( !init_hmm_obs_params )
    {
        /* read tests params */
        trsiRead( &min_img_size, "10", "Minimal width or height of image" );
        trsiRead( &max_img_size, "300", "Maximal width or height of image" );
        trsiRead( &max_dct_size, "24", "Maximum DCT size" );
        trsiRead( &base_iters, "100", "Base number of iterations" );

        init_hmm_obs_params = 1;
    }
}


static  CvSize  hmm_obs_dct_get_size( IplImage* img, CvSize dctSize, CvSize delta )
{
    CvSize result;
    CvRect roi = cvGetImageROI( img );

    result.width = (roi.width - dctSize.width + delta.width) / delta.width;
    result.height = (roi.height - dctSize.height + delta.height) / delta.height;

    return result;
}


static  void hmm_obs_dct_etalon( IplImage* img, char* obs, CvSize dctSize,
                                 CvSize obsSize, CvSize delta )
{
    IplImage  *src = cvCreateImage( dctSize, IPL_DEPTH_8U, 1 );
    IplImage  *dst = cvCreateImage( dctSize, IPL_DEPTH_32F, 1 );

    CvSize result = hmm_obs_dct_get_size( img, dctSize, delta );
    
    int  x, y, j;
    int  obs_step = obsSize.width*sizeof(float);
    
    result.width *= delta.width;
    result.height *= delta.height;

    for( y = 0; y < result.height; y += delta.height )
        for( x = 0; x < result.width; x += delta.width )
        {
            cvSetImageROI( img, cvRect( x, y, dctSize.width, dctSize.height ));
            cvCopy( img, src );
            iplDCT2D( src, dst, IPL_DCT_Forward );
            for( j = 0; j < obsSize.height; j++ )
            {
                memcpy( obs, dst->imageData + dst->widthStep*j, obs_step );
                obs += obs_step;
            }
        }

    cvReleaseImage( &src );
    cvReleaseImage( &dst );
}



/* ///////////////////// moments_test ///////////////////////// */
static int hmm_dct_test( void )
{
    const double success_error_level = 1.1;

    int   seed = atsGetSeed();
    int   code = TRS_OK;
    const int max_obs_size = 8;

    /* position where the maximum error occured */
    int   i, merr_iter = 0;

    /* test parameters */
    double  max_err = 0.;

    IplImage  *img = 0;
    IplImage  *obs = 0;
    IplImage  *obs2 = 0;
    AtsRandState rng_state;
    CvSize  obs_size;

    atsRandInit( &rng_state, 0, img8u_range, seed );

    read_hmm_obs_params();

    img = cvCreateImage( cvSize( max_img_size, max_img_size ), IPL_DEPTH_8U, 1 );
    obs_size.height = max_img_size; 
    obs_size.width = obs_size.height*64;
    obs = cvCreateImage( obs_size, IPL_DEPTH_32F, 1 );
    obs2 = cvCreateImage( obs_size, IPL_DEPTH_32F, 1 );

    for( i = 0; i < base_iters; i++ )
    {
        CvSize size;
        CvSize dctSize, obsSize, delta, result;
        double err = 0;

        size.width = atsRandPlain32s( &rng_state ) %
                     (max_img_size - min_img_size + 1) + min_img_size;
        size.height = atsRandPlain32s( &rng_state ) %
                     (max_img_size - min_img_size + 1) + min_img_size;

        dctSize.width = atsRandPlain32s( &rng_state ) % (max_dct_size - 1) + 2;
        if( dctSize.width > size.width )
            dctSize.width = size.width;
        dctSize.height = atsRandPlain32s( &rng_state ) % (max_dct_size - 1) + 2;
        if( dctSize.height > size.height )
            dctSize.height = size.height;

        obsSize.width = atsRandPlain32s( &rng_state ) % max_obs_size + 1;
        if( obsSize.width > dctSize.width )
            obsSize.width = dctSize.width;
        obsSize.height = atsRandPlain32s( &rng_state ) % max_obs_size + 1;
        if( obsSize.height > dctSize.height )
            obsSize.height = dctSize.height;

        delta.width = atsRandPlain32s( &rng_state ) % dctSize.width + 1;
        delta.height = atsRandPlain32s( &rng_state ) % dctSize.height + 1;

        cvSetImageROI( img, cvRect( 0, 0, size.width, size.height ));

        result = hmm_obs_dct_get_size( img, dctSize, delta );

        atsFillRandomImageEx( img, &rng_state );

        OPENCV_CALL( cvImgToObs_DCT( img, (float*)(obs->imageData), dctSize, obsSize, delta ));

        hmm_obs_dct_etalon( img, obs2->imageData, dctSize, obsSize, delta );

        obs->width = obs2->width = result.width*obsSize.width*obsSize.height;
        obs->height = obs2->height = result.height;
        obs->widthStep = obs2->widthStep = obs->width*sizeof(float);

        assert( obs->roi == 0 && obs2->roi == 0 );

        err = cvNorm( obs, obs2, CV_C );

        obs->width = obs2->width = max_img_size;
        obs->height = obs2->height = max_img_size;
        obs->widthStep = obs2->widthStep = obs->width*sizeof(float);

        if( err > max_err )
        {
            merr_iter = i;
            max_err = err;
            if( max_err > success_error_level )
                goto test_exit;
        }
    }

test_exit:

    cvReleaseImage( &img );
    cvReleaseImage( &obs );
    cvReleaseImage( &obs2 );

    if( code == TRS_OK )
    {
        trsWrite( ATS_LST, "Max err is %g at iter = %d, seed = %08x",
                           max_err, merr_iter, seed );

        return max_err <= success_error_level ?
            trsResult( TRS_OK, "No errors" ) :
            trsResult( TRS_FAIL, "Bad accuracy" );
    }
    /*else
    {
        trsWrite( ATS_LST, "Fatal error at iter = %d, seed = %08x", i, seed );
        return trsResult( TRS_FAIL, "Function returns error code" );
    }*/
}


void InitAImageToHMMObs( void )
{
    /* Register test functions */

    trsReg( funcs[0], test_desc[0], atsAlgoClass, hmm_dct_test );

} /* InitAMoments */

#endif

/* End of file. */
