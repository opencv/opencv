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

#if 0
CvStatus
icvFetchLine8uC3R( uchar * src, int src_step,
                   uchar * dst, int *dst_num, CvSize src_size, CvPoint start, CvPoint end )
{
    int i;
    int dx = end.x - start.x, dy = end.y - start.y;
    int err;

    if( !src || !dst || (src_size.width | src_size.height) < 0 ||
        src_step < src_size.width * 3 ||
        (unsigned) start.x >= (unsigned) src_size.width ||
        (unsigned) start.y >= (unsigned) src_size.height ||
        (unsigned) end.x >= (unsigned) src_size.width ||
        (unsigned) end.y >= (unsigned) src_size.height )
        return CV_BADFACTOR_ERR;

    if( dx < 0 )
    {
        dx = -dx;
        dy = -dy;
        start.x = end.x;
        start.y = end.y;
    }

    src += start.y * src_step + start.x * 3;

    i = dy >> 31;
    dy = (dy ^ i) - i;
    src_step = (src_step ^ i) - i;

    if( dx > dy )
    {
        if( dst_num )
        {
            if( *dst_num <= dx )
                return CV_BADSIZE_ERR;
            *dst_num = dx + 1;
        }
        err = dx;
        dx += dx;
        dy += dy;
        for( i = dx; i >= 0; i -= 2, dst += 3 )
        {
            int mask = (err -= dy) < 0 ? -1 : 0;

            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];

            err += dx & mask;
            src += (src_step & mask) + 3;
        }
    }
    else
    {
        if( dst_num )
        {
            if( *dst_num <= dy )
                return CV_BADSIZE_ERR;
            *dst_num = dy + 1;
        }
        err = dy;
        dx += dx;
        dy += dy;
        for( i = dy; i >= 0; i -= 2, dst += 3 )
        {
            int mask = (err -= dx) < 0 ? -1 : 0;

            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];

            err += dy & mask;
            src += src_step + (mask & 3);
        }
    }
    return CV_NO_ERR;
}

CvStatus
icvDrawLine8uC3R( uchar * src, int src_num,
                  uchar * dst, int dst_step, CvSize dst_size, CvPoint start, CvPoint end )
{
    int i;
    int dx = end.x - start.x, dy = end.y - start.y;
    int err;

    if( !src || !dst || (dst_size.width | dst_size.height) < 0 ||
        dst_step < dst_size.width * 3 ||
        (unsigned) start.x >= (unsigned) dst_size.width ||
        (unsigned) start.y >= (unsigned) dst_size.height ||
        (unsigned) end.x >= (unsigned) dst_size.width ||
        (unsigned) end.y >= (unsigned) dst_size.height )
        return CV_BADFACTOR_ERR;

    if( dx < 0 )
    {
        dx = -dx;
        dy = -dy;
        start.x = end.x;
        start.y = end.y;
    }

    dst += start.y * dst_step + start.x * 3;

    i = dy >> 31;
    dy = (dy ^ i) - i;
    dst_step = (dst_step ^ i) - i;

    if( dx > dy )
    {
        if( (unsigned) (src_num - 1) < (unsigned) dx )
            return CV_BADSIZE_ERR;
        err = dx;
        dx += dx;
        dy += dy;
        for( i = dx; i >= 0; i -= 2, src += 3 )
        {
            int mask = (err -= dy) < 0 ? -1 : 0;

            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            err += dx & mask;
            dst += (dst_step & mask) + 3;
        }
    }
    else
    {
        if( (unsigned) (src_num - 1) < (unsigned) dy )
            return CV_BADSIZE_ERR;
        err = dy;
        dx += dx;
        dy += dy;
        for( i = dy; i >= 0; i -= 2, src += 3 )
        {
            int mask = (err -= dx) < 0 ? -1 : 0;

            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            err += dy & mask;
            dst += dst_step + (mask & 3);
        }
    }
    return CV_NO_ERR;
}
#endif

/*======================================================================================*/

static CvStatus
icvPreWarpImage8uC3R( int numLines,     /* number of scanlines   */
                      uchar * src,      /* source image          */
                      int src_step,     /* line step         */
                      uchar * dst,      /* dest buffers          */
                      int *dst_nums,    /* lens of buffer        */
                      CvSize src_size,  /* image size in pixels */
                      int *scanlines )  /* scanlines array       */
{
    int k;
    CvPoint start;
    CvPoint end;
    int curr;
    int curr_dst;
    CvMat mat;

    curr = 0;
    curr_dst = 0;

    cvInitMatHeader( &mat, src_size.height, src_size.width, CV_8UC3, src, src_step );

    for( k = 0; k < numLines; k++ )
    {
        start.x = scanlines[curr++];
        start.y = scanlines[curr++];

        end.x = scanlines[curr++];
        end.y = scanlines[curr++];

#ifdef _DEBUG
        {
        CvLineIterator iterator;
        assert( cvInitLineIterator( &mat, start, end, &iterator, 8 ) == dst_nums[k] );
        }
#endif
        cvSampleLine( &mat, start, end, dst + curr_dst, 8 );
        curr_dst += dst_nums[k] * 3;

    }

    return CV_NO_ERR;
}


/*======================================================================================*/

static CvStatus
icvPostWarpImage8uC3R( int numLines,    /* number of scanlines  */
                       uchar * src,     /* source buffers       */
                       int *src_nums,   /* lens of buffers      */
                       uchar * dst,     /* dest image           */
                       int dst_step,    /* dest image step      */
                       CvSize dst_size, /* dest image size      */
                       int *scanlines ) /* scanline             */
{
    int i, k;
    CvPoint start;
    CvPoint end;
    int curr;
    int src_num;
    int curr_src;
    CvMat mat;
    CvLineIterator iterator;

    curr = 0;
    curr_src = 0;

    cvInitMatHeader( &mat, dst_size.height, dst_size.width, CV_8UC3, dst, dst_step );

    for( k = 0; k < numLines; k++ )
    {
        start.x = scanlines[curr++];
        start.y = scanlines[curr++];

        end.x = scanlines[curr++];
        end.y = scanlines[curr++];

        src_num = src_nums[k];

        if( cvInitLineIterator( &mat, start, end, &iterator, 8 ) != src_num )
        {
            assert(0);
            return CV_NOTDEFINED_ERR;
        }

        for( i = 0; i < src_num; i++ )
        {
            memcpy( iterator.ptr, src + curr_src, 3 );
            CV_NEXT_LINE_POINT( iterator );
            curr_src += 3;
        }

#if 0
        err = icvDrawLine8uC3R( src + curr_src, /* sourse buffer    */
                                src_num,        /* len of buffer    */
                                dst,    /* dest image       */
                                dst_step,       /* dest image step  */
                                dst_size,       /* dest image size  */
                                start,  /* start point      */
                                end );  /* end point        */
        curr_src += src_num * 3;
#endif
    }

    return CV_NO_ERR;

}


/*======================================================================================*/

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name:    icvDeleteMoire8uC3R
//    Purpose:
//      Function deletes moire - replaces black uncovered pixels with their neighboors.
//    Context:
//    Parameters:
//      img       - image data
//      img_step  - distance between lines in bytes
//      img_size  - width and height of the image in pixels
//    Returns:
//      CV_NO_ERR if all Ok or error code
//    Notes:
//F*/
static CvStatus
icvDeleteMoire8u( uchar * img, int img_step, CvSize img_size, int cn )
{
    int x, y;
    uchar *src = img, *dst = img + img_step;

    if( !img || img_size.width <= 0 || img_size.height <= 0 || img_step < img_size.width * 3 )
        return CV_BADFACTOR_ERR;

    img_size.width *= cn;

    for( y = 1; y < img_size.height; y++, src = dst, dst += img_step )
    {
        switch( cn )
        {
        case 1:
            for( x = 0; x < img_size.width; x++ )
            {
                if( dst[x] == 0 )
                    dst[x] = src[x];
            }
            break;
        case 3:
            for( x = 0; x < img_size.width; x += 3 )
            {
                if( dst[x] == 0 && dst[x + 1] == 0 && dst[x + 2] == 0 )
                {
                    dst[x] = src[x];
                    dst[x + 1] = src[x + 1];
                    dst[x + 2] = src[x + 2];
                }
            }
            break;
        default:
            assert(0);
            break;
        }
    }

    return CV_NO_ERR;
}


/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: cvDeleteMoire
//    Purpose: The functions delete moire on the image after ViewMorphing
//    Context:
//    Parameters:  img        - image on which will delete moire
//
//    Notes:
//F*/
CV_IMPL void
cvDeleteMoire( IplImage * img )
{
    uchar *img_data = 0;
    int img_step = 0;
    CvSize img_size;

    CV_FUNCNAME( "cvDeleteMoire" );

    __BEGIN__;

    cvGetImageRawData( img, &img_data, &img_step, &img_size );

    if( img->nChannels != 1 && img->nChannels != 3 )
        CV_ERROR( CV_BadNumChannels, "Source image must have 3 channel." );
    if( img->depth != IPL_DEPTH_8U )
        CV_ERROR( CV_BadDepth, "Channel depth of source image must be 8." );

    CV_CALL( icvDeleteMoire8u( img_data, img_step, img_size, img->nChannels ));

    __END__;

}


/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: cvPreWarpImage
//    Purpose: The functions warp image for next stage of ViewMorphing
//    Context:
//    Parameters:  img        - initial image (in the beginning)
//
//    Notes:
//F*/
CV_IMPL void
cvPreWarpImage( int numLines,   /* number of scanlines */
                IplImage * img, /* Source Image       */
                uchar * dst,    /* dest buffers       */
                int *dst_nums,  /* lens of buffer     */
                int *scanlines /* scanlines array    */  )
{
    uchar *img_data = 0;
    int img_step = 0;
    CvSize img_size;

    CV_FUNCNAME( "cvPreWarpImage" );

    __BEGIN__;

    cvGetImageRawData( img, &img_data, &img_step, &img_size );

    if( img->nChannels != 3 )
        CV_ERROR( CV_BadNumChannels, "Source image must have 3 channel." );
    if( img->depth != IPL_DEPTH_8U )
        CV_ERROR( CV_BadDepth, "Channel depth of image must be 8." );

    CV_CALL( icvPreWarpImage8uC3R( numLines,    /* number of scanlines  */
                                   img_data,    /* source image         */
                                   img_step,    /* line step            */
                                   dst, /* dest buffers         */
                                   dst_nums,    /* lens of buffer       */
                                   img_size,    /* image size in pixels */
                                   scanlines /* scanlines array      */  ));

    __END__;

}


/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: cvPostWarpImage
//    Purpose: The functions postwarp the image after morphing
//    Context:
//    Parameters:  img        - initial image (in the beginning)
//
//    Notes:
//F*/
CV_IMPL void
cvPostWarpImage( int numLines,  /* number of scanlines  */
                 uchar * src,   /* source buffers       */
                 int *src_nums, /* lens of buffers      */
                 IplImage * img,        /* dest image           */
                 int *scanlines /* scanline             */  )
{
    uchar *img_data = 0;
    int img_step = 0;
    CvSize img_size;

    CV_FUNCNAME( "cvPostWarpImage" );

    __BEGIN__;

    cvGetImageRawData( img, &img_data, &img_step, &img_size );

    if( img->nChannels != 3 )
        CV_ERROR( CV_BadNumChannels, "Source image must have 3 channel." );
    if( img->depth != IPL_DEPTH_8U )
        CV_ERROR( CV_BadDepth, "Channel depth of image must be 8." );

    CV_CALL( icvPostWarpImage8uC3R( numLines,   /* number of scanlines   */
                                    src,        /* source buffers       */
                                    src_nums,   /* lens of buffers      */
                                    img_data,   /* dest image           */
                                    img_step,   /* dest image step      */
                                    img_size,   /* dest image size      */
                                    scanlines /* scanline             */  ));

    __END__;
}

/* End of file */
