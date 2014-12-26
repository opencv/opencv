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

// Authors: Konstantin Dols <dols@ient.rwth-aachen.de>
//          Mark Asbach <asbach@ient.rwth-aachen.de>
//
//          Institute of Communications Engineering
//          RWTH Aachen University


#include "precomp.hpp"

// required to enable some functions used here...
#define XINE_ENABLE_EXPERIMENTAL_FEATURES

#include <cassert>

extern "C"
{
#include <xine.h>
    //#include <xine/xineutils.h>

    // forward declaration from <xine/xineutils.h>
    const char *xine_get_homedir( void );
}

typedef struct CvCaptureAVI_XINE
{
    /// method call table
    xine_t * xine;
    xine_stream_t * stream;
    xine_video_port_t * vo_port;

    /// frame returned by xine_get_next_video_frame()
    xine_video_frame_t xine_frame;

    IplImage	* yuv_frame;
    IplImage	* bgr_frame;

    /// image dimansions of the input stream.
    CvSize	size;

    /// framenumber of the last frame received from xine_get_next_video_frame().
    /// note: always keep this value updated !!!!
    int	frame_number;

    /// framerate of the opened stream
    double	frame_rate;

    /// duration of a frame in stream
    double	frame_duration;

    /// indicated if input is seekable
    bool	seekable;

}
CvCaptureAVI_XINE;


// 4:2:2 interleaved -> BGR
static void icvYUY2toBGR( CvCaptureAVI_XINE * capture )
{
    uint8_t * v	= capture->xine_frame.data;
    int offset;
    for ( int y = 0; y < capture->yuv_frame->height; y++ )
    {
        offset	= y * capture->yuv_frame->widthStep;

        for ( int x = 0; x < capture->yuv_frame->width; x++, offset += 3 )
        {
            capture->yuv_frame->imageData[ offset + 1 ] = v[ 3 ];
            capture->yuv_frame->imageData[ offset + 2 ] = v[ 1 ];
            if ( x & 1 )
            {
                capture->yuv_frame->imageData[ offset ] = v[ 2 ];
                v += 4;
            }
            else
            {
                capture->yuv_frame->imageData[ offset ] = v[ 0 ];
            }
        }
    }

    // convert to BGR
    cvCvtColor( capture->yuv_frame, capture->bgr_frame, CV_YCrCb2BGR );
}


// 4:2:0 planary -> BGR
static void icvYV12toBGR( CvCaptureAVI_XINE * capture )
{
    IplImage * yuv	= capture->yuv_frame;
    int	w_Y	= capture->size.width;
    int	h_Y	= capture->size.height;

    int	w_UV	= w_Y >> 1;

    int	size_Y	= w_Y * h_Y;
    int	size_UV	= size_Y / 4;

    int	line	= yuv->widthStep;

    uint8_t * addr_Y = capture->xine_frame.data;
    uint8_t * addr_U = addr_Y + size_Y;
    uint8_t * addr_V = addr_U + size_UV;

    // YYYY..UU.VV. -> BGRBGRBGR...
    for ( int y = 0; y < h_Y; y++ )
    {
        int offset = y * line;
        for ( int x = 0; x < w_Y; x++, offset += 3 )
        {
            /*
            if ( x&1 )
            {
                addr_U++; addr_V++;
            }
            */
            int one_zero = x & 1;
            addr_U += one_zero;
            addr_V += one_zero;

            yuv->imageData[ offset ] = *( addr_Y++ );
            yuv->imageData[ offset + 1 ] = *addr_U;
            yuv->imageData[ offset + 2 ] = *addr_V;
        }

        if ( y & 1 )
        {
            addr_U -= w_UV;
            addr_V -= w_UV;
        }
    }

    /* convert to BGR */
    cvCvtColor( capture->yuv_frame, capture->bgr_frame, CV_YCrCb2BGR );
}

static void icvCloseAVI_XINE( CvCaptureAVI_XINE* capture )
{
    xine_free_video_frame( capture->vo_port, &capture->xine_frame );

    if ( capture->yuv_frame ) cvReleaseImage( &capture->yuv_frame );
    if ( capture->bgr_frame ) cvReleaseImage( &capture->bgr_frame );

    xine_close( capture->stream );
    //	xine_dispose( capture->stream );

    if ( capture->vo_port ) xine_close_video_driver( capture->xine, capture->vo_port );

    xine_exit( capture->xine );
}


/**
 * CHECKS IF THE STREAM IN * capture IS SEEKABLE.
**/
static void icvCheckSeekAVI_XINE( CvCaptureAVI_XINE * capture )
{
    OPENCV_ASSERT ( capture,                        "icvCheckSeekAVI_XINE( CvCaptureAVI_XINE* )", "illegal capture");
    OPENCV_ASSERT ( capture->stream,
                        "icvCheckSeekAVI_XINE( CvCaptureAVI_XINE* )", "illegal capture->stream");
    OPENCV_ASSERT ( capture->vo_port,
                        "icvCheckSeekAVI_XINE( CvCaptureAVI_XINE* )", "illegal capture->vo_port");

#ifndef NDEBUG
    fprintf( stderr, "(DEBUG) icvCheckSeekAVI_XINE ... start\n" );
#endif

    // temp. frame for testing.
    xine_video_frame_t tmp;
    // try to seek to a future frame...
    xine_play( capture->stream, 0, 300 ); /* 300msec */
    // try to receive the frame...
    xine_get_next_video_frame( capture->vo_port, &tmp );
    // if the framenumber is still 0, we can't use the xine seek functionality
    capture->seekable = ( tmp.frame_number != 0 );
    // reset stream
    xine_play( capture->stream, 0, 0 );
    // release xine_frame
    xine_free_video_frame( capture->vo_port, &tmp );

#ifndef NDEBUG
    if ( capture->seekable )
        fprintf( stderr, "(DEBUG) icvCheckSeekAVI_XINE: Input is seekable, using XINE seek implementation.\n" );
    else
        fprintf( stderr, "(DEBUG) icvCheckSeekAVI_XINE: Input is NOT seekable, using fallback function.\n" );

    fprintf( stderr, "(DEBUG) icvCheckSeekAVI_XINE ... end\n" );
#endif
}


static int icvOpenAVI_XINE( CvCaptureAVI_XINE* capture, const char* filename )
{
#ifndef NDEBUG
    fprintf( stderr, "(DEBUG) icvOpenAVI_XINE ... start\n" );
#endif

    char configfile[ 2048 ];

    capture->xine = xine_new();
    sprintf( configfile, "%s%s", xine_get_homedir(), "/.xine/config" );

    xine_config_load( capture->xine, configfile );
    xine_init( capture->xine );

    xine_engine_set_param( capture->xine, 0, 0 );
    capture->vo_port = xine_new_framegrab_video_port( capture->xine );
    if ( capture->vo_port == NULL )
    {
        printf( "(ERROR)icvOpenAVI_XINE(): Unable to initialize video driver.\n" );
        return 0;
    }

    capture->stream = xine_stream_new( capture->xine, NULL, capture->vo_port );

    if ( !xine_open( capture->stream, filename ) )
    {
        printf( "(ERROR)icvOpenAVI_XINE(): Unable to open source '%s'\n", filename );
        return 0;
    }
    // reset stream...
    xine_play( capture->stream, 0, 0 );


    // initialize some internals...
    capture->frame_number = 0;

    if ( !xine_get_next_video_frame( capture->vo_port, &capture->xine_frame ) )
    {
#ifndef NDEBUG
        fprintf( stderr, "(DEBUG) icvOpenAVI_XINE ... failed!\n" );
#endif
        return 0;
    }

    capture->size = cvSize( capture->xine_frame.width, capture->xine_frame.height );
    capture->yuv_frame = cvCreateImage( capture->size, IPL_DEPTH_8U, 3 );
    capture->bgr_frame = cvCreateImage( capture->size, IPL_DEPTH_8U, 3 );

    xine_free_video_frame( capture->vo_port, &capture->xine_frame );
    capture->xine_frame.data[ 0 ] = 0;

    icvCheckSeekAVI_XINE( capture );

    capture->frame_duration = xine_get_stream_info( capture->stream, XINE_STREAM_INFO_FRAME_DURATION ) / 90.;
    capture->frame_rate = 1000 / capture->frame_duration;

#ifndef NDEBUG
    fprintf( stderr, "(DEBUG) frame_duration = %f, framerate = %f\n", capture->frame_duration, capture->frame_rate );
#endif

    OPENCV_ASSERT ( capture->yuv_frame,
                        "icvOpenAVI_XINE( CvCaptureAVI_XINE *, const char *)", "couldn't create yuv frame");

    OPENCV_ASSERT ( capture->bgr_frame,
                        "icvOpenAVI_XINE( CvCaptureAVI_XINE *, const char *)", "couldn't create bgr frame");

#ifndef NDEBUG
    fprintf( stderr, "(DEBUG) icvOpenAVI_XINE ... end\n" );
#endif
    return 1;
}


static int icvGrabFrameAVI_XINE( CvCaptureAVI_XINE* capture )
{
#ifndef NDEBUG
    fprintf( stderr, "(DEBUG) icvGrabFrameAVI_XINE ... start\n" );
#endif

    OPENCV_ASSERT ( capture,
                        "icvGrabFrameAVI_XINE( CvCaptureAVI_XINE * )", "illegal capture");
    OPENCV_ASSERT ( capture->vo_port,
                        "icvGrabFrameAVI_XINE( CvCaptureAVI_XINE * )", "illegal capture->vo_port");

    int res = xine_get_next_video_frame( capture->vo_port, &capture->xine_frame );

    /* always keep internal framenumber updated !!! */
    if ( res ) capture->frame_number++;

#ifndef NDEBUG
    fprintf( stderr, "(DEBUG) icvGrabFrameAVI_XINE ... end\n" );
#endif
    return res;
}


static const IplImage* icvRetrieveFrameAVI_XINE( CvCaptureAVI_XINE* capture, int )
{
#ifndef NDEBUG
    fprintf( stderr, "(DEBUG) icvRetrieveFrameAVI_XINE ... start\n" );
#endif

    OPENCV_ASSERT ( capture,
                        "icvRetrieveFrameAVI_XINE( CvCaptureAVI_XINE * )", "illegal capture");
    OPENCV_ASSERT ( capture->stream,
                        "icvRetrieveFrameAVI_XINE( CvCaptureAVI_XINE * )", "illegal capture->stream");
    OPENCV_ASSERT ( capture->vo_port,
                        "icvRetrieveFrameAVI_XINE( CvCaptureAVI_XINE * )", "illegal capture->vo_port");

    /* no frame grabbed yet? so let's do it now! */
    int res = 0;
    if ( capture->xine_frame.data == 0 )
    {
        res = icvGrabFrameAVI_XINE( capture );
    }
    else
    {
        res = 1;
    }

    if ( res )
    {
        switch ( capture->xine_frame.colorspace )
        {
                case XINE_IMGFMT_YV12: icvYV12toBGR( capture );
#ifndef NDEBUG
                printf( "(DEBUG)icvRetrieveFrameAVI_XINE: converted YV12 to BGR.\n" );
#endif
                break;

                case XINE_IMGFMT_YUY2: icvYUY2toBGR( capture );
#ifndef NDEBUG
                printf( "(DEBUG)icvRetrieveFrameAVI_XINE: converted YUY2 to BGR.\n" );
#endif
                break;
                case XINE_IMGFMT_XVMC: printf( "(ERROR)icvRetrieveFrameAVI_XINE: XVMC format not supported!\n" );
                break;

                case XINE_IMGFMT_XXMC: printf( "(ERROR)icvRetrieveFrameAVI_XINE: XXMC format not supported!\n" );
                break;

                default: printf( "(ERROR)icvRetrieveFrameAVI_XINE: unknown color/pixel format!\n" );
        }

        /* always release last xine_frame, not needed anymore, but store its frame_number in *capture ! */
        xine_free_video_frame( capture->vo_port, &capture->xine_frame );
        capture->xine_frame.data = 0;

#ifndef NDEBUG
        fprintf( stderr, "(DEBUG) icvRetrieveFrameAVI_XINE ... end\n" );
#endif
        return capture->bgr_frame;
    }

#ifndef NDEBUG
    fprintf( stderr, "(DEBUG) icvRetrieveFrameAVI_XINE ... failed!\n" );
#endif
    return 0;
}


/**
 * THIS FUNCTION IS A FALLBACK FUNCTION FOR THE CASE THAT THE XINE SEEK IMPLEMENTATION
 * DOESN'T WORK WITH THE ACTUAL INPUT. THIS FUNCTION IS ONLY USED IN THE CASE OF AN EMERGENCY,
 * BECAUSE IT IS VERY SLOW !
**/
static int icvOldSeekFrameAVI_XINE( CvCaptureAVI_XINE* capture, int f )
{
#ifndef NDEBUG
    fprintf( stderr, "(DEBUG) icvOldSeekFrameAVI_XINE ... start\n" );
#endif

    OPENCV_ASSERT ( capture,
                        "icvRetricvOldSeekFrameAVI_XINE( CvCaptureAVI_XINE *, int )", "illegal capture");
    OPENCV_ASSERT ( capture->stream,
                        "icvOldSeekFrameAVI_XINE( CvCaptureAVI_XINE *, int )", "illegal capture->stream");
    OPENCV_ASSERT ( capture->vo_port,
                        "icvOldSeekFrameAVI_XINE( CvCaptureAVI_XINE *, int )", "illegal capture->vo_port");

// not needed tnx to asserts...
    // we need a valid capture context and it's stream to seek through
//	if ( !capture || !capture->stream ) return 0;

    // no need to seek if we are already there...
    if ( f == capture->frame_number )
    {
#ifndef NDEBUG
        fprintf( stderr, "(DEBUG) icvOldSeekFrameAVI_XINE ... end\n" );
#endif
        return 1;
    }
    // if the requested position is behind out actual position,
    // we just need to read the remaining amount of frames until we are there.
    else if ( f > capture->frame_number )
    {
        for ( ;capture->frame_number < f;capture->frame_number++ )
            /// un-increment framenumber grabbing failed
            if ( !xine_get_next_video_frame( capture->vo_port, &capture->xine_frame ) )
            {
                capture->frame_number--;
                break;
            }
            else
            {
                xine_free_video_frame( capture->vo_port, &capture->xine_frame );
            }
    }
    // otherwise we need to reset the stream and
    // start reading frames from the beginning.
    else // f < capture->frame_number
    {
        /// reset stream, should also work with non-seekable input
        xine_play( capture->stream, 0, 0 );
        /// read frames until we are at the requested frame
        for ( capture->frame_number = 0; capture->frame_number < f; capture->frame_number++ )
            /// un-increment last framenumber if grabbing failed
            if ( !xine_get_next_video_frame( capture->vo_port, &capture->xine_frame ) )
            {
                capture->frame_number--;
                break;
            }
            else
            {
                xine_free_video_frame( capture->vo_port, &capture->xine_frame );
            }
    }


#ifndef NDEBUG
    fprintf( stderr, "(DEBUG) icvOldSeekFrameAVI_XINE ... end\n" );
#endif
    return ( f == capture->frame_number ) ? 1 : 0;
}


static int icvSeekFrameAVI_XINE( CvCaptureAVI_XINE* capture, int f )
{
#ifndef NDEBUG
    fprintf( stderr, "(DEBUG) icvSeekFrameAVI_XINE ... start\n" );
#endif

    OPENCV_ASSERT ( capture,
                        "icvSeekFrameAVI_XINE( CvCaptureAVI_XINE *, int )", "illegal capture");
    OPENCV_ASSERT ( capture->stream,
                        "icvSeekFrameAVI_XINE( CvCaptureAVI_XINE *, int )", "illegal capture->stream");
    OPENCV_ASSERT ( capture->vo_port,
                        "icvSeekFrameAVI_XINE( CvCaptureAVI_XINE *, int )", "illegal capture->vo_port");

// not needed tnx to asserts...
    // we need a valid capture context and it's stream to seek through
//	if ( !capture || !capture->stream ) return 0;

    if ( capture->seekable )
    {

        /// use xinelib's seek functionality
        int new_time = ( int ) ( ( f + 1 ) * ( float ) capture->frame_duration );

#ifndef NDEBUG
        fprintf( stderr, "(DEBUG) calling xine_play()" );
#endif
        if ( xine_play( capture->stream, 0, new_time ) )
        {
#ifndef NDEBUG
            fprintf( stderr, "ok\n" );
            fprintf( stderr, "(DEBUG) icvSeekFrameAVI_XINE ... end\n" );
#endif
            capture->frame_number = f;
            return 1;
        }
        else
        {
#ifndef NDEBUG
            fprintf( stderr, "failed\n" );
            fprintf( stderr, "(DEBUG) icvSeekFrameAVI_XINE ... failed\n" );
#endif
            return 0;
        }
    }
    else
    {
#ifndef NDEBUG
        fprintf( stderr, "(DEBUG) icvSeekFrameAVI_XINE ... end\n" );
#endif
        return icvOldSeekFrameAVI_XINE( capture, f );
    }
}


static int icvSeekTimeAVI_XINE( CvCaptureAVI_XINE* capture, int t )
{
#ifndef NDEBUG
    fprintf( stderr, "(DEBUG) icvSeekTimeAVI_XINE ... start\n" );
#endif

    OPENCV_ASSERT ( capture,
                        "icvSeekTimeAVI_XINE( CvCaptureAVI_XINE *, int )", "illegal capture");
    OPENCV_ASSERT ( capture->stream,
                        "icvSeekTimeAVI_XINE( CvCaptureAVI_XINE *, int )", "illegal capture->stream");
    OPENCV_ASSERT ( capture->vo_port,
                        "icvSeekTimeAVI_XINE( CvCaptureAVI_XINE *, int )", "illegal capture->vo_port");

#ifndef NDEBUG
    fprintf( stderr, "(DEBUG) icvSeekTimeAVI_XINE ... start\n" );
#endif

// not needed tnx to asserts...
    // we need a valid capture context and it's stream to seek through
//	if ( !capture || !capture->stream ) return 0;

    if ( capture->seekable )
    {
        /// use xinelib's seek functionality
        if ( xine_play( capture->stream, 0, t ) )
        {
            capture->frame_number = ( int ) ( ( float ) t * capture->frame_rate / 1000 );
#ifndef NDEBUG
            fprintf( stderr, "(DEBUG) icvSeekFrameAVI_XINE ... end\n" );
#endif
            return 1;
        }
        else
        {
#ifndef NDEBUG
            fprintf( stderr, "(DEBUG) icvSeekFrameAVI_XINE ... failed!\n" );
#endif
            return 0;
        }
    }
    else
    {
        int new_frame = ( int ) ( ( float ) t * capture->frame_rate / 1000 );
#ifndef NDEBUG
        fprintf( stderr, "(DEBUG) icvSeekFrameAVI_XINE ....end\n" );
#endif
        return icvOldSeekFrameAVI_XINE( capture, new_frame );
    }
}


static int icvSeekRatioAVI_XINE( CvCaptureAVI_XINE* capture, double ratio )
{
#ifndef NDEBUG
    fprintf( stderr, "(DEBUG) icvSeekRatioAVI_XINE ... start\n" );
#endif

    OPENCV_ASSERT ( capture,
                        "icvSeekRatioAVI_XINE( CvCaptureAVI_XINE *, double )", "illegal capture");
    OPENCV_ASSERT ( capture->stream,
                        "icvSeekRatioAVI_XINE( CvCaptureAVI_XINE *, double )", "illegal capture->stream");
    OPENCV_ASSERT ( capture->vo_port,
                        "icvSeekRatioAVI_XINE( CvCaptureAVI_XINE *, double )", "illegal capture->vo_port");

// not needed tnx to asserts...
    // we need a valid capture context and it's stream to seek through
//	if ( !capture || !capture->stream ) return 0;

    /// ratio must be [0..1]
    if ( ratio > 1 || ratio < 0 ) return 0;

    if ( capture->seekable )
    {
    // TODO: FIX IT, DOESN'T WORK PROPERLY, YET...!
        int pos_t, pos_l, length;
        xine_get_pos_length( capture->stream, &pos_l, &pos_t, &length );
        fprintf( stderr, "ratio on GetProperty(): %d\n", pos_l );

        /// use xinelib's seek functionality
        if ( xine_play( capture->stream, (int)(ratio*(float)length), 0 ) )
        {
            capture->frame_number = ( int ) ( ratio*length / capture->frame_duration );
        }
        else
        {
#ifndef NDEBUG
            fprintf( stderr, "(DEBUG) icvSeekRatioAVI_XINE ... failed!\n" );
#endif
            return 0;
        }
    }
    else
    {
        /// TODO: fill it !
        fprintf( stderr, "icvSeekRatioAVI_XINE(): Seek not supported by stream !\n" );
        fprintf( stderr, "icvSeekRatioAVI_XINE(): (seek in stream with NO seek support NOT implemented...yet!)\n" );
#ifndef NDEBUG
        fprintf( stderr, "(DEBUG) icvSeekRatioAVI_XINE ... failed!\n" );
#endif
        return 0;
    }

#ifndef NDEBUG
    fprintf( stderr, "(DEBUG) icvSeekRatioAVI_XINE ... end!\n" );
#endif
    return 1;
}


static double icvGetPropertyAVI_XINE( CvCaptureAVI_XINE* capture, int property_id )
{
#ifndef NDEBUG
    fprintf( stderr, "(DEBUG) icvGetPropertyAVI_XINE ... start\n" );
#endif

    OPENCV_ASSERT ( capture,
                        "icvGetPropertyAVI_XINE( CvCaptureAVI_XINE *, int )", "illegal capture");
    OPENCV_ASSERT ( capture->stream,
                        "icvGetPropertyAVI_XINE( CvCaptureAVI_XINE *, int )", "illegal capture->stream");
    OPENCV_ASSERT ( capture->vo_port,
                        "icvGetPropertyAVI_XINE( CvCaptureAVI_XINE *, int )", "illegal capture->vo_port");
    OPENCV_ASSERT ( capture->xine,
                        "icvGetPropertyAVI_XINE( CvCaptureAVI_XINE *, int )", "illegal capture->xine");
    OPENCV_ASSERT ( capture->bgr_frame,
                        "icvGetPropertyAVI_XINE( CvCaptureAVI_XINE *, int )", "illegal capture->bgr_frame");

// not needed tnx to asserts...
    // we need a valid capture context and it's stream to seek through
//	if ( !capture || !capture->stream || !capture->bgr_frame || !capture->xine || !capture->vo_port ) return 0

    int pos_t, pos_l, length;
    xine_get_pos_length( capture->stream, &pos_l, &pos_t, &length );
    fprintf( stderr, "ratio on GetProperty(): %i\n", pos_l );

    switch ( property_id )
    {
            /// return actual position in msec
            case CV_CAP_PROP_POS_MSEC:
            if ( !capture->seekable )
            {
                fprintf( stderr, "(ERROR) GetPropertyAVI_XINE(CV_CAP_PROP_POS_MSEC:\n" );
                fprintf( stderr, "	Stream is NOT seekable, so position info may NOT be valid !!\n" );
            }
            return pos_t;

            /// return actual frame number
            case CV_CAP_PROP_POS_FRAMES:
            /// we insist the capture->frame_number to be remain updated !!!!
            return capture->frame_number;

            /// return actual position ratio in the range [0..1] depending on
            /// the total length of the stream and the actual position
            case CV_CAP_PROP_POS_AVI_RATIO:
            if ( !capture->seekable )
            {
                fprintf( stderr, "(ERROR) GetPropertyAVI_XINE(CV_CAP_PROP_POS_AVI_RATIO:\n" );
                fprintf( stderr, "	Stream is NOT seekable, so ratio info may NOT be valid !!\n" );
            }
            if ( length == 0 ) break;
            else return pos_l / 65535;


            /// return width of image source
            case CV_CAP_PROP_FRAME_WIDTH:
            return capture->size.width;

            /// return height of image source
            case CV_CAP_PROP_FRAME_HEIGHT:
            return capture->size.height;

            /// return framerate of stream
            case CV_CAP_PROP_FPS:
            if ( !capture->seekable )
            {
                fprintf( stderr, "(ERROR) GetPropertyAVI_XINE(CV_CAP_PROP_FPS:\n" );
                fprintf( stderr, "	Stream is NOT seekable, so FPS info may NOT be valid !!\n" );
            }
            return capture->frame_rate;

            /// return four-character-code (FOURCC) of source's codec
            case CV_CAP_PROP_FOURCC:
            return ( double ) xine_get_stream_info( capture->stream, XINE_STREAM_INFO_VIDEO_FOURCC );
    }

#ifndef NDEBUG
    fprintf( stderr, "(DEBUG) icvGetPropertyAVI_XINE ... failed!\n" );
#endif

    return 0;
}


static int icvSetPropertyAVI_XINE( CvCaptureAVI_XINE* capture,
                                   int property_id, double value )
{
#ifndef NDEBUG
    fprintf( stderr, "(DEBUG) icvSetPropertyAVI_XINE ... start\n" );
#endif

    OPENCV_ASSERT ( capture,
                        "icvSetPropertyAVI_XINE( CvCaptureAVI_XINE *, int, double )", "illegal capture");
    OPENCV_ASSERT ( capture->stream,
                        "icvGetPropericvSetPropertyAVI_XINE( CvCaptureAVI_XINE *, int )", "illegal capture->stream");
    OPENCV_ASSERT ( capture->vo_port,
                        "icvSetPropertyAVI_XINE( CvCaptureAVI_XINE *, int, double )", "illegal capture->vo_port");

// not needed tnx to asserts...
    // we need a valid capture context and it's stream to seek through
//	if ( !capture || !capture->stream || !capture->bgr_frame || !capture->xine || !capture->vo_port ) return 0

#ifndef NDEBUG
    fprintf( stderr, "(DEBUG) icvSetPropertyAVI_XINE: seeking to value %f ... ", value );
#endif

    switch ( property_id )
    {
            /// set (seek to) position in msec
            case CV_CAP_PROP_POS_MSEC:
            return icvSeekTimeAVI_XINE( capture, ( int ) value );

            /// set (seek to) frame number
            case CV_CAP_PROP_POS_FRAMES:
            return icvSeekFrameAVI_XINE( capture, ( int ) value );

            /// set (seek to) position ratio in the range [0..1] depending on
            /// the total length of the stream and the actual position
            case CV_CAP_PROP_POS_AVI_RATIO:
            return icvSeekRatioAVI_XINE( capture, value );

            default:
#ifndef NDEBUG
            fprintf( stderr, "(DEBUG) icvSetPropertyAVI_XINE ... failed!\n" );
#endif

            return 0;
    }
}


static CvCaptureAVI_XINE* icvCaptureFromFile_XINE( const char* filename )
{
    // construct capture struct
    CvCaptureAVI_XINE * capture = ( CvCaptureAVI_XINE* ) cvAlloc ( sizeof ( CvCaptureAVI_XINE ) );
    memset( capture, 0, sizeof ( CvCaptureAVI_XINE ) );

    // initialize XINE
    if ( !icvOpenAVI_XINE( capture, filename ) )
        return 0;

    OPENCV_ASSERT ( capture,
                        "cvCaptureFromFile_XINE( const char * )", "couldn't create capture");

    return capture;

}



class CvCaptureAVI_XINE_CPP : public CvCapture
{
public:
    CvCaptureAVI_XINE_CPP() { captureXINE = 0; }
    virtual ~CvCaptureAVI_XINE_CPP() { close(); }

    virtual bool open( const char* filename );
    virtual void close();

    virtual double getProperty(int) const;
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
protected:

    CvCaptureAVI_XINE* captureXINE;
};

bool CvCaptureAVI_XINE_CPP::open( const char* filename )
{
    close();
    captureXINE = icvCaptureFromFile_XINE(filename);
    return captureXINE != 0;
}

void CvCaptureAVI_XINE_CPP::close()
{
    if( captureXINE )
    {
        icvCloseAVI_XINE( captureXINE );
        cvFree( &captureXINE );
    }
}

bool CvCaptureAVI_XINE_CPP::grabFrame()
{
    return captureXINE ? icvGrabFrameAVI_XINE( captureXINE ) != 0 : false;
}

IplImage* CvCaptureAVI_XINE_CPP::retrieveFrame(int)
{
    return captureXINE ? (IplImage*)icvRetrieveFrameAVI_XINE( captureXINE, 0 ) : 0;
}

double CvCaptureAVI_XINE_CPP::getProperty( int propId ) const
{
    return captureXINE ? icvGetPropertyAVI_XINE( captureXINE, propId ) : 0;
}

bool CvCaptureAVI_XINE_CPP::setProperty( int propId, double value )
{
    return captureXINE ? icvSetPropertyAVI_XINE( captureXINE, propId, value ) != 0 : false;
}

CvCapture* cvCreateFileCapture_XINE(const char* filename)
{
    CvCaptureAVI_XINE_CPP* capture = new CvCaptureAVI_XINE_CPP;

    if( capture->open(filename))
        return capture;

    delete capture;
    return 0;
}


#undef NDEBUG
