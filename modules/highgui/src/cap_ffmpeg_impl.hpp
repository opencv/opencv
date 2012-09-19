/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#include "cap_ffmpeg_api.hpp"
#include <assert.h>
#include <algorithm>
#include <limits>

#if defined _MSC_VER && _MSC_VER >= 1200
#pragma warning( disable: 4244 4510 4512 4610 )
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "ffmpeg_codecs.hpp"

#include <libavutil/mathematics.h>

#ifdef WIN32
  #define HAVE_FFMPEG_SWSCALE 1
  #include <libavcodec/avcodec.h>
  #include <libswscale/swscale.h>
#else

#ifndef HAVE_FFMPEG_SWSCALE
    #error "libswscale is necessary to build the newer OpenCV ffmpeg wrapper"
#endif

// if the header path is not specified explicitly, let's deduce it
#if !defined HAVE_FFMPEG_AVCODEC_H && !defined HAVE_LIBAVCODEC_AVCODEC_H

#if defined(HAVE_GENTOO_FFMPEG)
  #define HAVE_LIBAVCODEC_AVCODEC_H 1
  #if defined(HAVE_FFMPEG_SWSCALE)
    #define HAVE_LIBSWSCALE_SWSCALE_H 1
  #endif
#elif defined HAVE_FFMPEG
  #define HAVE_FFMPEG_AVCODEC_H 1
  #if defined(HAVE_FFMPEG_SWSCALE)
    #define HAVE_FFMPEG_SWSCALE_H 1
  #endif
#endif

#endif

#if defined(HAVE_FFMPEG_AVCODEC_H)
  #include <ffmpeg/avcodec.h>
#endif
#if defined(HAVE_FFMPEG_SWSCALE_H)
  #include <ffmpeg/swscale.h>
#endif

#if defined(HAVE_LIBAVCODEC_AVCODEC_H)
  #include <libavcodec/avcodec.h>
#endif
#if defined(HAVE_LIBSWSCALE_SWSCALE_H)
  #include <libswscale/swscale.h>
#endif

#endif

#ifdef __cplusplus
}
#endif

#if defined _MSC_VER && _MSC_VER >= 1200
#pragma warning( default: 4244 4510 4512 4610 )
#endif

#ifdef NDEBUG
#define CV_WARN(message)
#else
#define CV_WARN(message) fprintf(stderr, "warning: %s (%s:%d)\n", message, __FILE__, __LINE__)
#endif

/* PIX_FMT_RGBA32 macro changed in newer ffmpeg versions */
#ifndef PIX_FMT_RGBA32
#define PIX_FMT_RGBA32 PIX_FMT_RGB32
#endif

#define CALC_FFMPEG_VERSION(a,b,c) ( a<<16 | b<<8 | c )

#if defined WIN32 || defined _WIN32
    #include <windows.h>
#elif defined __linux__ || defined __APPLE__
    #include <unistd.h>
    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/sysctl.h>
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#if defined(__APPLE__)
#define AV_NOPTS_VALUE_ ((int64_t)0x8000000000000000LL)
#else
#define AV_NOPTS_VALUE_ ((int64_t)AV_NOPTS_VALUE)
#endif

#ifndef AVERROR_EOF
#define AVERROR_EOF (-MKTAG( 'E','O','F',' '))
#endif

static int get_number_of_cpus(void)
{
#if LIBAVFORMAT_BUILD < CALC_FFMPEG_VERSION(52, 111, 0)
    return 1;
#elif defined WIN32 || defined _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo( &sysinfo );

    return (int)sysinfo.dwNumberOfProcessors;
#elif defined __linux__
    return (int)sysconf( _SC_NPROCESSORS_ONLN );
#elif defined __APPLE__
    int numCPU=0;
    int mib[4];
    size_t len = sizeof(numCPU);

    // set the mib for hw.ncpu
    mib[0] = CTL_HW;
    mib[1] = HW_AVAILCPU;  // alternatively, try HW_NCPU;

    // get the number of CPUs from the system
    sysctl(mib, 2, &numCPU, &len, NULL, 0);

    if( numCPU < 1 )
    {
        mib[1] = HW_NCPU;
        sysctl( mib, 2, &numCPU, &len, NULL, 0 );

        if( numCPU < 1 )
            numCPU = 1;
    }

    return (int)numCPU;
#else
    return 1;
#endif
}


struct Image_FFMPEG
{
    unsigned char* data;
    int step;
    int width;
    int height;
    int cn;
};


inline void _opencv_ffmpeg_free(void** ptr)
{
    if(*ptr) free(*ptr);
    *ptr = 0;
}


struct CvCapture_FFMPEG
{
    bool open( const char* filename );
    void close();

    double getProperty(int);
    bool setProperty(int, double);
    bool grabFrame();
    bool retrieveFrame(int, unsigned char** data, int* step, int* width, int* height, int* cn);

    void init();

    void    seek(int64_t frame_number);
    void    seek(double sec);
    bool    slowSeek( int framenumber );

    int64_t get_total_frames();
    double  get_duration_sec();
    double  get_fps();
    int     get_bitrate();

    double  r2d(AVRational r);
    int64_t dts_to_frame_number(int64_t dts);
    double  dts_to_sec(int64_t dts);

    AVFormatContext * ic;
    AVCodec         * avcodec;
    int               video_stream;
    AVStream        * video_st;
    AVFrame         * picture;
    AVFrame           rgb_picture;
    int64_t           picture_pts;

    AVPacket          packet;
    Image_FFMPEG      frame;
    struct SwsContext *img_convert_ctx;

    int64_t frame_number, first_frame_number;

    double eps_zero;
/*
   'filename' contains the filename of the videosource,
   'filename==NULL' indicates that ffmpeg's seek support works
   for the particular file.
   'filename!=NULL' indicates that the slow fallback function is used for seeking,
   and so the filename is needed to reopen the file on backward seeking.
*/
    char              * filename;
};

void CvCapture_FFMPEG::init()
{
    ic = 0;
    video_stream = -1;
    video_st = 0;
    picture = 0;
    picture_pts = AV_NOPTS_VALUE_;
    first_frame_number = -1;
    memset( &rgb_picture, 0, sizeof(rgb_picture) );
    memset( &frame, 0, sizeof(frame) );
    filename = 0;
    memset(&packet, 0, sizeof(packet));
    av_init_packet(&packet);
    img_convert_ctx = 0;

    avcodec = 0;
    frame_number = 0;
    eps_zero = 0.000025;
}


void CvCapture_FFMPEG::close()
{
    if( img_convert_ctx )
    {
        sws_freeContext(img_convert_ctx);
        img_convert_ctx = 0;
    }

    if( picture )
        av_free(picture);

    if( video_st )
    {
#if LIBAVFORMAT_BUILD > 4628
        avcodec_close( video_st->codec );

#else
        avcodec_close( &(video_st->codec) );

#endif
        video_st = NULL;
    }

    if( ic )
    {
#if LIBAVFORMAT_BUILD < CALC_FFMPEG_VERSION(53, 24, 2)
        av_close_input_file(ic);
#else
        avformat_close_input(&ic);
#endif

        ic = NULL;
    }

    if( rgb_picture.data[0] )
    {
        free( rgb_picture.data[0] );
        rgb_picture.data[0] = 0;
    }

    // free last packet if exist
    if (packet.data) {
        av_free_packet (&packet);
        packet.data = NULL;
    }

    init();
}


#ifndef AVSEEK_FLAG_FRAME
#define AVSEEK_FLAG_FRAME 0
#endif
#ifndef AVSEEK_FLAG_ANY
#define AVSEEK_FLAG_ANY 1
#endif

static void icvInitFFMPEG_internal()
{
    static volatile bool initialized = false;
    if( !initialized )
    {
    #if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 13, 0)
        avformat_network_init();
    #endif

        /* register all codecs, demux and protocols */
        av_register_all();

        av_log_set_level(AV_LOG_ERROR);

        initialized = true;
    }
}

bool CvCapture_FFMPEG::open( const char* _filename )
{
    icvInitFFMPEG_internal();

    unsigned i;
    bool valid = false;

    close();

#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(52, 111, 0)
    int err = avformat_open_input(&ic, _filename, NULL, NULL);
#else
    int err = av_open_input_file(&ic, _filename, NULL, 0, NULL);
#endif

    if (err < 0) {
        CV_WARN("Error opening file");
        goto exit_func;
    }
    err =
#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 3, 0)
    avformat_find_stream_info(ic, NULL);
#else
    av_find_stream_info(ic);
#endif
    if (err < 0) {
        CV_WARN("Could not find codec parameters");
        goto exit_func;
    }
    for(i = 0; i < ic->nb_streams; i++)
    {
#if LIBAVFORMAT_BUILD > 4628
        AVCodecContext *enc = ic->streams[i]->codec;
#else
        AVCodecContext *enc = &ic->streams[i]->codec;
#endif

#ifdef FF_API_THREAD_INIT
        avcodec_thread_init(enc, get_number_of_cpus());
#else
        enc->thread_count = get_number_of_cpus();
#endif

#if LIBAVFORMAT_BUILD < CALC_FFMPEG_VERSION(53, 2, 0)
#define AVMEDIA_TYPE_VIDEO CODEC_TYPE_VIDEO
#endif

        if( AVMEDIA_TYPE_VIDEO == enc->codec_type && video_stream < 0) {
            AVCodec *codec = avcodec_find_decoder(enc->codec_id);
            if (!codec ||
#if LIBAVCODEC_VERSION_INT >= ((53<<16)+(8<<8)+0)
                avcodec_open2(enc, codec, NULL)
#else
                avcodec_open(enc, codec)
#endif
                < 0) goto exit_func;

            video_stream = i;
            video_st = ic->streams[i];
            picture = avcodec_alloc_frame();

            rgb_picture.data[0] = (uint8_t*)malloc(
                    avpicture_get_size( PIX_FMT_BGR24,
                                        enc->width, enc->height ));
            avpicture_fill( (AVPicture*)&rgb_picture, rgb_picture.data[0],
                            PIX_FMT_BGR24, enc->width, enc->height );

            frame.width = enc->width;
            frame.height = enc->height;
            frame.cn = 3;
            frame.step = rgb_picture.linesize[0];
            frame.data = rgb_picture.data[0];
            break;
        }
    }

    if(video_stream >= 0) valid = true;

exit_func:

    if( !valid )
        close();

    return valid;
}


bool CvCapture_FFMPEG::grabFrame()
{
    bool valid = false;
    int got_picture;

    int count_errs = 0;
    const int max_number_of_attempts = 1 << 16;

    if( !ic || !video_st )  return false;

    if( ic->streams[video_stream]->nb_frames > 0 &&
        frame_number > ic->streams[video_stream]->nb_frames )
        return false;

    av_free_packet (&packet);

    picture_pts = AV_NOPTS_VALUE_;

    // get the next frame
    while (!valid)
    {
        int ret = av_read_frame(ic, &packet);
        if (ret == AVERROR(EAGAIN)) continue;

        /* else if (ret < 0) break; */

        if( packet.stream_index != video_stream )
        {
            av_free_packet (&packet);
            count_errs++;
            if (count_errs > max_number_of_attempts)
                break;
            continue;
        }

        // Decode video frame
        #if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 2, 0)
            avcodec_decode_video2(video_st->codec, picture, &got_picture, &packet);
        #elif LIBAVFORMAT_BUILD > 4628
                avcodec_decode_video(video_st->codec,
                                     picture, &got_picture,
                                     packet.data, packet.size);
        #else
                avcodec_decode_video(&video_st->codec,
                                     picture, &got_picture,
                                     packet.data, packet.size);
        #endif

        // Did we get a video frame?
        if(got_picture)
        {
            //picture_pts = picture->best_effort_timestamp;
            if( picture_pts == AV_NOPTS_VALUE_ )
                picture_pts = packet.pts != AV_NOPTS_VALUE_ && packet.pts != 0 ? packet.pts : packet.dts;
            frame_number++;
            valid = true;
        }
        else
        {
            count_errs++;
            if (count_errs > max_number_of_attempts)
                break;
        }

        av_free_packet (&packet);
    }

    if( valid && first_frame_number < 0 )
        first_frame_number = dts_to_frame_number(picture_pts);

    // return if we have a new picture or not
    return valid;
}


bool CvCapture_FFMPEG::retrieveFrame(int, unsigned char** data, int* step, int* width, int* height, int* cn)
{
    if( !video_st || !picture->data[0] )
        return false;

    avpicture_fill((AVPicture*)&rgb_picture, rgb_picture.data[0], PIX_FMT_RGB24,
                   video_st->codec->width, video_st->codec->height);

    if( img_convert_ctx == NULL ||
        frame.width != video_st->codec->width ||
        frame.height != video_st->codec->height )
    {
        if( img_convert_ctx )
            sws_freeContext(img_convert_ctx);

        frame.width = video_st->codec->width;
        frame.height = video_st->codec->height;

        img_convert_ctx = sws_getCachedContext(
                NULL,
                video_st->codec->width, video_st->codec->height,
                video_st->codec->pix_fmt,
                video_st->codec->width, video_st->codec->height,
                PIX_FMT_BGR24,
                SWS_BICUBIC,
                NULL, NULL, NULL
                );

        if (img_convert_ctx == NULL)
            return false;//CV_Error(0, "Cannot initialize the conversion context!");
    }

    sws_scale(
            img_convert_ctx,
            picture->data,
            picture->linesize,
            0, video_st->codec->height,
            rgb_picture.data,
            rgb_picture.linesize
            );

    *data = frame.data;
    *step = frame.step;
    *width = frame.width;
    *height = frame.height;
    *cn = frame.cn;

    return true;
}


double CvCapture_FFMPEG::getProperty( int property_id )
{
    if( !video_st ) return 0;

    switch( property_id )
    {
    case CV_FFMPEG_CAP_PROP_POS_MSEC:
        return 1000.0*(double)frame_number/get_fps();
    case CV_FFMPEG_CAP_PROP_POS_FRAMES:
        return (double)frame_number;
    case CV_FFMPEG_CAP_PROP_POS_AVI_RATIO:
        return r2d(ic->streams[video_stream]->time_base);
    case CV_FFMPEG_CAP_PROP_FRAME_COUNT:
        return (double)get_total_frames();
    case CV_FFMPEG_CAP_PROP_FRAME_WIDTH:
        return (double)frame.width;
    case CV_FFMPEG_CAP_PROP_FRAME_HEIGHT:
        return (double)frame.height;
    case CV_FFMPEG_CAP_PROP_FPS:
#if LIBAVCODEC_BUILD > 4753
        return av_q2d(video_st->r_frame_rate);
#else
        return (double)video_st->codec.frame_rate
                / (double)video_st->codec.frame_rate_base;
#endif
    case CV_FFMPEG_CAP_PROP_FOURCC:
#if LIBAVFORMAT_BUILD > 4628
        return (double)video_st->codec->codec_tag;
#else
        return (double)video_st->codec.codec_tag;
#endif
    default:
        break;
    }

    return 0;
}

double CvCapture_FFMPEG::r2d(AVRational r)
{
    return r.num == 0 || r.den == 0 ? 0. : (double)r.num / (double)r.den;
}

double CvCapture_FFMPEG::get_duration_sec()
{
    double sec = (double)ic->duration / (double)AV_TIME_BASE;

    if (sec < eps_zero)
    {
        sec = (double)ic->streams[video_stream]->duration * r2d(ic->streams[video_stream]->time_base);
    }

    if (sec < eps_zero)
    {
        sec = (double)ic->streams[video_stream]->duration * r2d(ic->streams[video_stream]->time_base);
    }

    return sec;
}

int CvCapture_FFMPEG::get_bitrate()
{
    return ic->bit_rate;
}

double CvCapture_FFMPEG::get_fps()
{
    double fps = r2d(ic->streams[video_stream]->r_frame_rate);

#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(52, 111, 0)
    if (fps < eps_zero)
    {
        fps = r2d(ic->streams[video_stream]->avg_frame_rate);
    }
#endif

    if (fps < eps_zero)
    {
        fps = 1.0 / r2d(ic->streams[video_stream]->codec->time_base);
    }

    return fps;
}

int64_t CvCapture_FFMPEG::get_total_frames()
{
    int64_t nbf = ic->streams[video_stream]->nb_frames;

    if (nbf == 0)
    {
        nbf = (int64_t)floor(get_duration_sec() * get_fps() + 0.5);
    }
    return nbf;
}

int64_t CvCapture_FFMPEG::dts_to_frame_number(int64_t dts)
{
    double sec = dts_to_sec(dts);
    return (int64_t)(get_fps() * sec + 0.5);
}

double CvCapture_FFMPEG::dts_to_sec(int64_t dts)
{
    return (double)(dts - ic->streams[video_stream]->start_time) *
        r2d(ic->streams[video_stream]->time_base);
}

void CvCapture_FFMPEG::seek(int64_t _frame_number)
{
    _frame_number = std::min(_frame_number, get_total_frames());
    int delta = 16;

    // if we have not grabbed a single frame before first seek, let's read the first frame
    // and get some valuable information during the process
    if( first_frame_number < 0 && get_total_frames() > 1 )
        grabFrame();

    for(;;)
    {
        int64_t _frame_number_temp = std::max(_frame_number-delta, (int64_t)0);
        double sec = (double)_frame_number_temp / get_fps();
        int64_t time_stamp = ic->streams[video_stream]->start_time;
        double  time_base  = r2d(ic->streams[video_stream]->time_base);
        time_stamp += (int64_t)(sec / time_base + 0.5);
        if (get_total_frames() > 1) av_seek_frame(ic, video_stream, time_stamp, AVSEEK_FLAG_BACKWARD);
        avcodec_flush_buffers(ic->streams[video_stream]->codec);
        if( _frame_number > 0 )
        {
            grabFrame();

            if( _frame_number > 1 )
            {
                frame_number = dts_to_frame_number(picture_pts) - first_frame_number;
                //printf("_frame_number = %d, frame_number = %d, delta = %d\n",
                //       (int)_frame_number, (int)frame_number, delta);

                if( frame_number < 0 || frame_number > _frame_number-1 )
                {
                    if( _frame_number_temp == 0 || delta >= INT_MAX/4 )
                        break;
                    delta = delta < 16 ? delta*2 : delta*3/2;
                    continue;
                }
                while( frame_number < _frame_number-1 )
                {
                    if(!grabFrame())
                        break;
                }
                frame_number++;
                break;
            }
            else
            {
                frame_number = 1;
                break;
            }
        }
        else
        {
            frame_number = 0;
            break;
        }
    }
}

void CvCapture_FFMPEG::seek(double sec)
{
    seek((int64_t)(sec * get_fps() + 0.5));
}

bool CvCapture_FFMPEG::setProperty( int property_id, double value )
{
    if( !video_st ) return false;

    switch( property_id )
    {
    case CV_FFMPEG_CAP_PROP_POS_MSEC:
    case CV_FFMPEG_CAP_PROP_POS_FRAMES:
    case CV_FFMPEG_CAP_PROP_POS_AVI_RATIO:
        {
            switch( property_id )
            {
            case CV_FFMPEG_CAP_PROP_POS_FRAMES:
                seek((int64_t)value);
                break;

            case CV_FFMPEG_CAP_PROP_POS_MSEC:
                seek(value/1000.0);
                break;

            case CV_FFMPEG_CAP_PROP_POS_AVI_RATIO:
                seek((int64_t)(value*ic->duration));
                break;
            }

            picture_pts=(int64_t)value;
        }
        break;
    default:
        return false;
    }

    return true;
}


///////////////// FFMPEG CvVideoWriter implementation //////////////////////////
struct CvVideoWriter_FFMPEG
{
    bool open( const char* filename, int fourcc,
               double fps, int width, int height, bool isColor );
    void close();
    bool writeFrame( const unsigned char* data, int step, int width, int height, int cn, int origin );

    void init();

    AVOutputFormat  * fmt;
    AVFormatContext * oc;
    uint8_t         * outbuf;
    uint32_t          outbuf_size;
    FILE            * outfile;
    AVFrame         * picture;
    AVFrame         * input_picture;
    uint8_t         * picbuf;
    AVStream        * video_st;
    int               input_pix_fmt;
    Image_FFMPEG      temp_image;
    int               frame_width, frame_height;
    bool              ok;
    struct SwsContext *img_convert_ctx;
};

static const char * icvFFMPEGErrStr(int err)
{
#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 2, 0)
    switch(err) {
    case AVERROR_BSF_NOT_FOUND:
        return "Bitstream filter not found";
    case AVERROR_DECODER_NOT_FOUND:
        return "Decoder not found";
    case AVERROR_DEMUXER_NOT_FOUND:
        return "Demuxer not found";
    case AVERROR_ENCODER_NOT_FOUND:
        return "Encoder not found";
    case AVERROR_EOF:
        return "End of file";
    case AVERROR_EXIT:
        return "Immediate exit was requested; the called function should not be restarted";
    case AVERROR_FILTER_NOT_FOUND:
        return "Filter not found";
    case AVERROR_INVALIDDATA:
        return "Invalid data found when processing input";
    case AVERROR_MUXER_NOT_FOUND:
        return "Muxer not found";
    case AVERROR_OPTION_NOT_FOUND:
        return "Option not found";
    case AVERROR_PATCHWELCOME:
        return "Not yet implemented in FFmpeg, patches welcome";
    case AVERROR_PROTOCOL_NOT_FOUND:
        return "Protocol not found";
    case AVERROR_STREAM_NOT_FOUND:
        return "Stream not found";
    default:
        break;
    }
#else
    switch(err) {
    case AVERROR_NUMEXPECTED:
        return "Incorrect filename syntax";
    case AVERROR_INVALIDDATA:
        return "Invalid data in header";
    case AVERROR_NOFMT:
        return "Unknown format";
    case AVERROR_IO:
        return "I/O error occurred";
    case AVERROR_NOMEM:
        return "Memory allocation error";
    default:
        break;
    }
#endif

    return "Unspecified error";
}

/* function internal to FFMPEG (libavformat/riff.c) to lookup codec id by fourcc tag*/
extern "C" {
    enum CodecID codec_get_bmp_id(unsigned int tag);
}

void CvVideoWriter_FFMPEG::init()
{
    fmt = 0;
    oc = 0;
    outbuf = 0;
    outbuf_size = 0;
    outfile = 0;
    picture = 0;
    input_picture = 0;
    picbuf = 0;
    video_st = 0;
    input_pix_fmt = 0;
    memset(&temp_image, 0, sizeof(temp_image));
    img_convert_ctx = 0;
    frame_width = frame_height = 0;
    ok = false;
}

/**
 * the following function is a modified version of code
 * found in ffmpeg-0.4.9-pre1/output_example.c
 */
static AVFrame * icv_alloc_picture_FFMPEG(int pix_fmt, int width, int height, bool alloc)
{
    AVFrame * picture;
    uint8_t * picture_buf;
    int size;

    picture = avcodec_alloc_frame();
    if (!picture)
        return NULL;
    size = avpicture_get_size( (PixelFormat) pix_fmt, width, height);
    if(alloc){
        picture_buf = (uint8_t *) malloc(size);
        if (!picture_buf)
        {
            av_free(picture);
            return NULL;
        }
        avpicture_fill((AVPicture *)picture, picture_buf,
                       (PixelFormat) pix_fmt, width, height);
    }
    else {
    }
    return picture;
}

/* add a video output stream to the container */
static AVStream *icv_add_video_stream_FFMPEG(AVFormatContext *oc,
                                             CodecID codec_id,
                                             int w, int h, int bitrate,
                                             double fps, int pixel_format)
{
    AVCodecContext *c;
    AVStream *st;
    int frame_rate, frame_rate_base;
    AVCodec *codec;

#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 10, 0)
    st = avformat_new_stream(oc, 0);
#else
    st = av_new_stream(oc, 0);
#endif

    if (!st) {
        CV_WARN("Could not allocate stream");
        return NULL;
    }

#if LIBAVFORMAT_BUILD > 4628
    c = st->codec;
#else
    c = &(st->codec);
#endif

#if LIBAVFORMAT_BUILD > 4621
    c->codec_id = av_guess_codec(oc->oformat, NULL, oc->filename, NULL, AVMEDIA_TYPE_VIDEO);
#else
    c->codec_id = oc->oformat->video_codec;
#endif

    if(codec_id != CODEC_ID_NONE){
        c->codec_id = codec_id;
    }

    //if(codec_tag) c->codec_tag=codec_tag;
    codec = avcodec_find_encoder(c->codec_id);

    c->codec_type = AVMEDIA_TYPE_VIDEO;

    /* put sample parameters */
    int64_t lbit_rate = (int64_t)bitrate;
    lbit_rate += (bitrate / 2);
    lbit_rate = std::min(lbit_rate, (int64_t)INT_MAX);
    c->bit_rate = lbit_rate;

    // took advice from
    // http://ffmpeg-users.933282.n4.nabble.com/warning-clipping-1-dct-coefficients-to-127-127-td934297.html
    c->qmin = 3;

    /* resolution must be a multiple of two */
    c->width = w;
    c->height = h;

    /* time base: this is the fundamental unit of time (in seconds) in terms
       of which frame timestamps are represented. for fixed-fps content,
       timebase should be 1/framerate and timestamp increments should be
       identically 1. */
    frame_rate=(int)(fps+0.5);
    frame_rate_base=1;
    while (fabs((double)frame_rate/frame_rate_base) - fps > 0.001){
        frame_rate_base*=10;
        frame_rate=(int)(fps*frame_rate_base + 0.5);
    }
#if LIBAVFORMAT_BUILD > 4752
    c->time_base.den = frame_rate;
    c->time_base.num = frame_rate_base;
    /* adjust time base for supported framerates */
    if(codec && codec->supported_framerates){
        const AVRational *p= codec->supported_framerates;
        AVRational req = {frame_rate, frame_rate_base};
        const AVRational *best=NULL;
        AVRational best_error= {INT_MAX, 1};
        for(; p->den!=0; p++){
            AVRational error= av_sub_q(req, *p);
            if(error.num <0) error.num *= -1;
            if(av_cmp_q(error, best_error) < 0){
                best_error= error;
                best= p;
            }
        }
        c->time_base.den= best->num;
        c->time_base.num= best->den;
    }
#else
    c->frame_rate = frame_rate;
    c->frame_rate_base = frame_rate_base;
#endif

    c->gop_size = 12; /* emit one intra frame every twelve frames at most */
    c->pix_fmt = (PixelFormat) pixel_format;

    if (c->codec_id == CODEC_ID_MPEG2VIDEO) {
        c->max_b_frames = 2;
    }
    if (c->codec_id == CODEC_ID_MPEG1VIDEO || c->codec_id == CODEC_ID_MSMPEG4V3){
        /* needed to avoid using macroblocks in which some coeffs overflow
           this doesnt happen with normal video, it just happens here as the
           motion of the chroma plane doesnt match the luma plane */
        /* avoid FFMPEG warning 'clipping 1 dct coefficients...' */
        c->mb_decision=2;
    }
#if LIBAVCODEC_VERSION_INT>0x000409
    // some formats want stream headers to be seperate
    if(oc->oformat->flags & AVFMT_GLOBALHEADER)
    {
        c->flags |= CODEC_FLAG_GLOBAL_HEADER;
    }
#endif

    return st;
}

static const int OPENCV_NO_FRAMES_WRITTEN_CODE = 1000;

static int icv_av_write_frame_FFMPEG( AVFormatContext * oc, AVStream * video_st, uint8_t * outbuf, uint32_t outbuf_size, AVFrame * picture )
{
#if LIBAVFORMAT_BUILD > 4628
    AVCodecContext * c = video_st->codec;
#else
    AVCodecContext * c = &(video_st->codec);
#endif
    int out_size;
    int ret = 0;

    if (oc->oformat->flags & AVFMT_RAWPICTURE) {
        /* raw video case. The API will change slightly in the near
           futur for that */
        AVPacket pkt;
        av_init_packet(&pkt);

#ifndef PKT_FLAG_KEY
#define PKT_FLAG_KEY AV_PKT_FLAG_KEY
#endif

        pkt.flags |= PKT_FLAG_KEY;
        pkt.stream_index= video_st->index;
        pkt.data= (uint8_t *)picture;
        pkt.size= sizeof(AVPicture);

        ret = av_write_frame(oc, &pkt);
    } else {
        /* encode the image */
        out_size = avcodec_encode_video(c, outbuf, outbuf_size, picture);
        /* if zero size, it means the image was buffered */
        if (out_size > 0) {
            AVPacket pkt;
            av_init_packet(&pkt);

#if LIBAVFORMAT_BUILD > 4752
            if(c->coded_frame->pts != (int64_t)AV_NOPTS_VALUE)
                pkt.pts = av_rescale_q(c->coded_frame->pts, c->time_base, video_st->time_base);
#else
            pkt.pts = c->coded_frame->pts;
#endif
            if(c->coded_frame->key_frame)
                pkt.flags |= PKT_FLAG_KEY;
            pkt.stream_index= video_st->index;
            pkt.data= outbuf;
            pkt.size= out_size;

            /* write the compressed frame in the media file */
            ret = av_write_frame(oc, &pkt);
        } else {
            ret = OPENCV_NO_FRAMES_WRITTEN_CODE;
        }
    }
    return ret;
}

/// write a frame with FFMPEG
bool CvVideoWriter_FFMPEG::writeFrame( const unsigned char* data, int step, int width, int height, int cn, int origin )
{
    bool ret = false;

    if( (width & -2) != frame_width || (height & -2) != frame_height || !data )
        return false;
    width = frame_width;
    height = frame_height;

    // typecast from opaque data type to implemented struct
#if LIBAVFORMAT_BUILD > 4628
    AVCodecContext *c = video_st->codec;
#else
    AVCodecContext *c = &(video_st->codec);
#endif

#if LIBAVFORMAT_BUILD < 5231
    // It is not needed in the latest versions of the ffmpeg
    if( c->codec_id == CODEC_ID_RAWVIDEO && origin != 1 )
    {
        if( !temp_image.data )
        {
            temp_image.step = (width*cn + 3) & -4;
            temp_image.width = width;
            temp_image.height = height;
            temp_image.cn = cn;
            temp_image.data = (unsigned char*)malloc(temp_image.step*temp_image.height);
        }
        for( int y = 0; y < height; y++ )
            memcpy(temp_image.data + y*temp_image.step, data + (height-1-y)*step, width*cn);
        data = temp_image.data;
        step = temp_image.step;
    }
#else
    if( width*cn != step )
    {
        if( !temp_image.data )
        {
            temp_image.step = width*cn;
            temp_image.width = width;
            temp_image.height = height;
            temp_image.cn = cn;
            temp_image.data = (unsigned char*)malloc(temp_image.step*temp_image.height);
        }
        if (origin == 1)
            for( int y = 0; y < height; y++ )
                memcpy(temp_image.data + y*temp_image.step, data + (height-1-y)*step, temp_image.step);
        else
            for( int y = 0; y < height; y++ )
                memcpy(temp_image.data + y*temp_image.step, data + y*step, temp_image.step);
        data = temp_image.data;
        step = temp_image.step;
    }
#endif

    // check parameters
    if (input_pix_fmt == PIX_FMT_BGR24) {
        if (cn != 3) {
            return false;
        }
    }
    else if (input_pix_fmt == PIX_FMT_GRAY8) {
        if (cn != 1) {
            return false;
        }
    }
    else {
        assert(false);
    }

    if ( c->pix_fmt != input_pix_fmt ) {
        assert( input_picture );
        // let input_picture point to the raw data buffer of 'image'
        avpicture_fill((AVPicture *)input_picture, (uint8_t *) data,
                       (PixelFormat)input_pix_fmt, width, height);

        if( !img_convert_ctx )
        {
            img_convert_ctx = sws_getContext(width,
                                             height,
                                             (PixelFormat)input_pix_fmt,
                                             c->width,
                                             c->height,
                                             c->pix_fmt,
                                             SWS_BICUBIC,
                                             NULL, NULL, NULL);
            if( !img_convert_ctx )
                return false;
        }

        if ( sws_scale(img_convert_ctx, input_picture->data,
                       input_picture->linesize, 0,
                       height,
                       picture->data, picture->linesize) < 0 )
            return false;
    }
    else{
        avpicture_fill((AVPicture *)picture, (uint8_t *) data,
                       (PixelFormat)input_pix_fmt, width, height);
    }

    ret = icv_av_write_frame_FFMPEG( oc, video_st, outbuf, outbuf_size, picture) >= 0;

    return ret;
}

/// close video output stream and free associated memory
void CvVideoWriter_FFMPEG::close()
{
    unsigned i;

    // nothing to do if already released
    if ( !picture )
        return;

    /* no more frame to compress. The codec has a latency of a few
       frames if using B frames, so we get the last frames by
       passing the same picture again */
    // TODO -- do we need to account for latency here?

    /* write the trailer, if any */
    if(ok && oc)
    {
        if( (oc->oformat->flags & AVFMT_RAWPICTURE) == 0 )
        {
            for(;;)
            {
                int ret = icv_av_write_frame_FFMPEG( oc, video_st, outbuf, outbuf_size, NULL);
                if( ret == OPENCV_NO_FRAMES_WRITTEN_CODE || ret < 0 )
                    break;
            }
        }
        av_write_trailer(oc);
    }

    if( img_convert_ctx )
    {
        sws_freeContext(img_convert_ctx);
        img_convert_ctx = 0;
    }

    // free pictures
#if LIBAVFORMAT_BUILD > 4628
    if( video_st->codec->pix_fmt != input_pix_fmt)
#else
    if( video_st->codec.pix_fmt != input_pix_fmt)
#endif
    {
        if(picture->data[0])
            free(picture->data[0]);
        picture->data[0] = 0;
    }
    av_free(picture);

    if (input_picture)
        av_free(input_picture);

    /* close codec */
#if LIBAVFORMAT_BUILD > 4628
    avcodec_close(video_st->codec);
#else
    avcodec_close(&(video_st->codec));
#endif

    av_free(outbuf);

    /* free the streams */
    for(i = 0; i < oc->nb_streams; i++)
    {
        av_freep(&oc->streams[i]->codec);
        av_freep(&oc->streams[i]);
    }

    if (!(fmt->flags & AVFMT_NOFILE))
    {
        /* close the output file */

#if LIBAVCODEC_VERSION_INT < ((52<<16)+(123<<8)+0)
#if LIBAVCODEC_VERSION_INT >= ((51<<16)+(49<<8)+0)
        url_fclose(oc->pb);
#else
        url_fclose(&oc->pb);
#endif
#else
        avio_close(oc->pb);
#endif

    }

    /* free the stream */
    av_free(oc);

    if( temp_image.data )
    {
        free(temp_image.data);
        temp_image.data = 0;
    }

    init();
}

/// Create a video writer object that uses FFMPEG
bool CvVideoWriter_FFMPEG::open( const char * filename, int fourcc,
                                 double fps, int width, int height, bool is_color )
{
    icvInitFFMPEG_internal();

    CodecID codec_id = CODEC_ID_NONE;
    int err, codec_pix_fmt;
    double bitrate_scale = 1;

    close();

    // check arguments
    if( !filename )
        return false;
    if(fps <= 0)
        return false;

    // we allow frames of odd width or height, but in this case we truncate
    // the rightmost column/the bottom row. Probably, this should be handled more elegantly,
    // but some internal functions inside FFMPEG swscale require even width/height.
    width &= -2;
    height &= -2;
    if( width <= 0 || height <= 0 )
        return false;

    /* auto detect the output format from the name and fourcc code. */

#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 2, 0)
    fmt = av_guess_format(NULL, filename, NULL);
#else
    fmt = guess_format(NULL, filename, NULL);
#endif

    if (!fmt)
        return false;

    /* determine optimal pixel format */
    if (is_color) {
        input_pix_fmt = PIX_FMT_BGR24;
    }
    else {
        input_pix_fmt = PIX_FMT_GRAY8;
    }

    /* Lookup codec_id for given fourcc */
#if LIBAVCODEC_VERSION_INT<((51<<16)+(49<<8)+0)
    if( (codec_id = codec_get_bmp_id( fourcc )) == CODEC_ID_NONE )
        return false;
#else
    const struct AVCodecTag * tags[] = { codec_bmp_tags, NULL};
    if( (codec_id = av_codec_get_id(tags, fourcc)) == CODEC_ID_NONE )
        return false;
#endif

    // alloc memory for context
#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 2, 0)
    oc = avformat_alloc_context();
#else
    oc = av_alloc_format_context();
#endif
    assert (oc);

    /* set file name */
    oc->oformat = fmt;
    snprintf(oc->filename, sizeof(oc->filename), "%s", filename);

    /* set some options */
    oc->max_delay = (int)(0.7*AV_TIME_BASE);  /* This reduces buffer underrun warnings with MPEG */

    // set a few optimal pixel formats for lossless codecs of interest..
    switch (codec_id) {
#if LIBAVCODEC_VERSION_INT>((50<<16)+(1<<8)+0)
    case CODEC_ID_JPEGLS:
        // BGR24 or GRAY8 depending on is_color...
        codec_pix_fmt = input_pix_fmt;
        break;
#endif
    case CODEC_ID_HUFFYUV:
        codec_pix_fmt = PIX_FMT_YUV422P;
        break;
    case CODEC_ID_MJPEG:
    case CODEC_ID_LJPEG:
        codec_pix_fmt = PIX_FMT_YUVJ420P;
        bitrate_scale = 3;
        break;
    case CODEC_ID_RAWVIDEO:
        codec_pix_fmt = input_pix_fmt == PIX_FMT_GRAY8 ||
                        input_pix_fmt == PIX_FMT_GRAY16LE ||
                        input_pix_fmt == PIX_FMT_GRAY16BE ? input_pix_fmt : PIX_FMT_YUV420P;
        break;
    default:
        // good for lossy formats, MPEG, etc.
        codec_pix_fmt = PIX_FMT_YUV420P;
        break;
    }

    double bitrate = MIN(bitrate_scale*fps*width*height, (double)INT_MAX/2);

    // TODO -- safe to ignore output audio stream?
    video_st = icv_add_video_stream_FFMPEG(oc, codec_id,
                                           width, height, (int)(bitrate + 0.5),
                                           fps, codec_pix_fmt);

    /* set the output parameters (must be done even if no
   parameters). */
#if LIBAVFORMAT_BUILD < CALC_FFMPEG_VERSION(53, 2, 0)
    if (av_set_parameters(oc, NULL) < 0) {
        return false;
    }
#endif

#if 0
#if FF_API_DUMP_FORMAT
    dump_format(oc, 0, filename, 1);
#else
    av_dump_format(oc, 0, filename, 1);
#endif
#endif

    /* now that all the parameters are set, we can open the audio and
     video codecs and allocate the necessary encode buffers */
    if (!video_st){
        return false;
    }

    AVCodec *codec;
    AVCodecContext *c;

#if LIBAVFORMAT_BUILD > 4628
    c = (video_st->codec);
#else
    c = &(video_st->codec);
#endif

    c->codec_tag = fourcc;
    /* find the video encoder */
    codec = avcodec_find_encoder(c->codec_id);
    if (!codec) {
        fprintf(stderr, "Could not find encoder for codec id %d: %s", c->codec_id, icvFFMPEGErrStr(
        #if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 2, 0)
                AVERROR_ENCODER_NOT_FOUND
        #else
                -1
        #endif
                ));
        return false;
    }

    int64_t lbit_rate = (int64_t)c->bit_rate;
    lbit_rate += (bitrate / 2);
    lbit_rate = std::min(lbit_rate, (int64_t)INT_MAX);
    c->bit_rate_tolerance = (int)lbit_rate;
    c->bit_rate = (int)lbit_rate;

    /* open the codec */
    if ((err=
#if LIBAVCODEC_VERSION_INT >= ((53<<16)+(8<<8)+0)
         avcodec_open2(c, codec, NULL)
#else
         avcodec_open(c, codec)
#endif
         ) < 0) {
        fprintf(stderr, "Could not open codec '%s': %s", codec->name, icvFFMPEGErrStr(err));
        return false;
    }

    outbuf = NULL;

    if (!(oc->oformat->flags & AVFMT_RAWPICTURE)) {
        /* allocate output buffer */
        /* assume we will never get codec output with more than 4 bytes per pixel... */
        outbuf_size = width*height*4;
        outbuf = (uint8_t *) av_malloc(outbuf_size);
    }

    bool need_color_convert;
    need_color_convert = (c->pix_fmt != input_pix_fmt);

    /* allocate the encoded raw picture */
    picture = icv_alloc_picture_FFMPEG(c->pix_fmt, c->width, c->height, need_color_convert);
    if (!picture) {
        return false;
    }

    /* if the output format is not our input format, then a temporary
   picture of the input format is needed too. It is then converted
   to the required output format */
    input_picture = NULL;
    if ( need_color_convert ) {
        input_picture = icv_alloc_picture_FFMPEG(input_pix_fmt, c->width, c->height, false);
        if (!input_picture) {
            return false;
        }
    }

    /* open the output file, if needed */
    if (!(fmt->flags & AVFMT_NOFILE)) {
#if LIBAVFORMAT_BUILD < CALC_FFMPEG_VERSION(53, 2, 0)
        if (url_fopen(&oc->pb, filename, URL_WRONLY) < 0)
#else
            if (avio_open(&oc->pb, filename, AVIO_FLAG_WRITE) < 0)
#endif
            {
            return false;
        }
    }

#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(52, 111, 0)
    /* write the stream header, if any */
    err=avformat_write_header(oc, NULL);
#else
    err=av_write_header( oc );
#endif

    if(err < 0)
    {
        close();
        remove(filename);
        return false;
    }
    frame_width = width;
    frame_height = height;
    ok = true;
    return true;
}



CvCapture_FFMPEG* cvCreateFileCapture_FFMPEG( const char* filename )
{
    CvCapture_FFMPEG* capture = (CvCapture_FFMPEG*)malloc(sizeof(*capture));
    capture->init();
    if( capture->open( filename ))
        return capture;
    capture->close();
    free(capture);
    return 0;
}


void cvReleaseCapture_FFMPEG(CvCapture_FFMPEG** capture)
{
    if( capture && *capture )
    {
        (*capture)->close();
        free(*capture);
        *capture = 0;
    }
}

int cvSetCaptureProperty_FFMPEG(CvCapture_FFMPEG* capture, int prop_id, double value)
{
    return capture->setProperty(prop_id, value);
}

double cvGetCaptureProperty_FFMPEG(CvCapture_FFMPEG* capture, int prop_id)
{
    return capture->getProperty(prop_id);
}

int cvGrabFrame_FFMPEG(CvCapture_FFMPEG* capture)
{
    return capture->grabFrame();
}

int cvRetrieveFrame_FFMPEG(CvCapture_FFMPEG* capture, unsigned char** data, int* step, int* width, int* height, int* cn)
{
    return capture->retrieveFrame(0, data, step, width, height, cn);
}

CvVideoWriter_FFMPEG* cvCreateVideoWriter_FFMPEG( const char* filename, int fourcc, double fps,
                                                  int width, int height, int isColor )
{
    CvVideoWriter_FFMPEG* writer = (CvVideoWriter_FFMPEG*)malloc(sizeof(*writer));
    writer->init();
    if( writer->open( filename, fourcc, fps, width, height, isColor != 0 ))
        return writer;
    writer->close();
    free(writer);
    return 0;
}


void cvReleaseVideoWriter_FFMPEG( CvVideoWriter_FFMPEG** writer )
{
    if( writer && *writer )
    {
        (*writer)->close();
        free(*writer);
        *writer = 0;
    }
}


int cvWriteFrame_FFMPEG( CvVideoWriter_FFMPEG* writer,
                         const unsigned char* data, int step,
                         int width, int height, int cn, int origin)
{
    return writer->writeFrame(data, step, width, height, cn, origin);
}



/*
 * For CUDA encoder
 */

struct OutputMediaStream_FFMPEG
{
    bool open(const char* fileName, int width, int height, double fps);
    void close();

    void write(unsigned char* data, int size, int keyFrame);

    // add a video output stream to the container
    static AVStream* addVideoStream(AVFormatContext *oc, CodecID codec_id, int w, int h, int bitrate, double fps, PixelFormat pixel_format);

    AVOutputFormat* fmt_;
    AVFormatContext* oc_;
    AVStream* video_st_;
};

void OutputMediaStream_FFMPEG::close()
{
    // no more frame to compress. The codec has a latency of a few
    // frames if using B frames, so we get the last frames by
    // passing the same picture again

    // TODO -- do we need to account for latency here?

    if (oc_)
    {
        // write the trailer, if any
        av_write_trailer(oc_);

        // free the streams
        for (unsigned int i = 0; i < oc_->nb_streams; ++i)
        {
            av_freep(&oc_->streams[i]->codec);
            av_freep(&oc_->streams[i]);
        }

        if (!(fmt_->flags & AVFMT_NOFILE) && oc_->pb)
        {
            // close the output file

            #if LIBAVCODEC_VERSION_INT < ((52<<16)+(123<<8)+0)
                #if LIBAVCODEC_VERSION_INT >= ((51<<16)+(49<<8)+0)
                    url_fclose(oc_->pb);
                #else
                    url_fclose(&oc_->pb);
                #endif
            #else
                avio_close(oc_->pb);
            #endif
        }

        // free the stream
        av_free(oc_);
    }
}

AVStream* OutputMediaStream_FFMPEG::addVideoStream(AVFormatContext *oc, CodecID codec_id, int w, int h, int bitrate, double fps, PixelFormat pixel_format)
{
    #if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 10, 0)
        AVStream* st = avformat_new_stream(oc, 0);
    #else
        AVStream* st = av_new_stream(oc, 0);
    #endif
    if (!st)
        return 0;

    #if LIBAVFORMAT_BUILD > 4628
        AVCodecContext* c = st->codec;
    #else
        AVCodecContext* c = &(st->codec);
    #endif

    c->codec_id = codec_id;
    c->codec_type = AVMEDIA_TYPE_VIDEO;

    // put sample parameters
    unsigned long long lbit_rate = static_cast<unsigned long long>(bitrate);
    lbit_rate += (bitrate / 4);
    lbit_rate = std::min(lbit_rate, static_cast<unsigned long long>(std::numeric_limits<int>::max()));
    c->bit_rate = bitrate;

    // took advice from
    // http://ffmpeg-users.933282.n4.nabble.com/warning-clipping-1-dct-coefficients-to-127-127-td934297.html
    c->qmin = 3;

    // resolution must be a multiple of two
    c->width = w;
    c->height = h;

    AVCodec* codec = avcodec_find_encoder(c->codec_id);

    // time base: this is the fundamental unit of time (in seconds) in terms
    // of which frame timestamps are represented. for fixed-fps content,
    // timebase should be 1/framerate and timestamp increments should be
    // identically 1

    int frame_rate = static_cast<int>(fps+0.5);
    int frame_rate_base = 1;
    while (fabs(static_cast<double>(frame_rate)/frame_rate_base) - fps > 0.001)
    {
        frame_rate_base *= 10;
        frame_rate = static_cast<int>(fps*frame_rate_base + 0.5);
    }
    c->time_base.den = frame_rate;
    c->time_base.num = frame_rate_base;

    #if LIBAVFORMAT_BUILD > 4752
        // adjust time base for supported framerates
        if (codec && codec->supported_framerates)
        {
            AVRational req = {frame_rate, frame_rate_base};
            const AVRational* best = NULL;
            AVRational best_error = {INT_MAX, 1};

            for (const AVRational* p = codec->supported_framerates; p->den!=0; ++p)
            {
                AVRational error = av_sub_q(req, *p);

                if (error.num < 0)
                    error.num *= -1;

                if (av_cmp_q(error, best_error) < 0)
                {
                    best_error= error;
                    best= p;
                }
            }

            c->time_base.den= best->num;
            c->time_base.num= best->den;
        }
    #endif

    c->gop_size = 12; // emit one intra frame every twelve frames at most
    c->pix_fmt = pixel_format;

    if (c->codec_id == CODEC_ID_MPEG2VIDEO)
        c->max_b_frames = 2;

    if (c->codec_id == CODEC_ID_MPEG1VIDEO || c->codec_id == CODEC_ID_MSMPEG4V3)
    {
        // needed to avoid using macroblocks in which some coeffs overflow
        // this doesnt happen with normal video, it just happens here as the
        // motion of the chroma plane doesnt match the luma plane

        // avoid FFMPEG warning 'clipping 1 dct coefficients...'

        c->mb_decision = 2;
    }

    #if LIBAVCODEC_VERSION_INT > 0x000409
        // some formats want stream headers to be seperate
        if (oc->oformat->flags & AVFMT_GLOBALHEADER)
        {
            c->flags |= CODEC_FLAG_GLOBAL_HEADER;
        }
    #endif

    return st;
}

bool OutputMediaStream_FFMPEG::open(const char* fileName, int width, int height, double fps)
{
    fmt_ = 0;
    oc_ = 0;
    video_st_ = 0;

    // tell FFMPEG to register codecs
    av_register_all();

    av_log_set_level(AV_LOG_ERROR);

    // auto detect the output format from the name and fourcc code
    #if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 2, 0)
        fmt_ = av_guess_format(NULL, fileName, NULL);
    #else
        fmt_ = guess_format(NULL, fileName, NULL);
    #endif
    if (!fmt_)
        return false;

    CodecID codec_id = CODEC_ID_H264;

    // alloc memory for context
    #if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 2, 0)
        oc_ = avformat_alloc_context();
    #else
        oc_ = av_alloc_format_context();
    #endif
    if (!oc_)
        return false;

    // set some options
    oc_->oformat = fmt_;
    snprintf(oc_->filename, sizeof(oc_->filename), "%s", fileName);

    oc_->max_delay = (int)(0.7 * AV_TIME_BASE); // This reduces buffer underrun warnings with MPEG

    // set a few optimal pixel formats for lossless codecs of interest..
    PixelFormat codec_pix_fmt = PIX_FMT_YUV420P;
    int bitrate_scale = 64;

    // TODO -- safe to ignore output audio stream?
    video_st_ = addVideoStream(oc_, codec_id, width, height, width * height * bitrate_scale, fps, codec_pix_fmt);
    if (!video_st_)
        return false;

    // set the output parameters (must be done even if no parameters)
    #if LIBAVFORMAT_BUILD < CALC_FFMPEG_VERSION(53, 2, 0)
        if (av_set_parameters(oc_, NULL) < 0)
            return false;
    #endif

    // now that all the parameters are set, we can open the audio and
    // video codecs and allocate the necessary encode buffers

    #if LIBAVFORMAT_BUILD > 4628
        AVCodecContext* c = (video_st_->codec);
    #else
        AVCodecContext* c = &(video_st_->codec);
    #endif

    c->codec_tag = MKTAG('H', '2', '6', '4');
    c->bit_rate_tolerance = c->bit_rate;

    // open the output file, if needed
    if (!(fmt_->flags & AVFMT_NOFILE))
    {
        #if LIBAVFORMAT_BUILD < CALC_FFMPEG_VERSION(53, 2, 0)
            int err = url_fopen(&oc_->pb, fileName, URL_WRONLY);
        #else
            int err = avio_open(&oc_->pb, fileName, AVIO_FLAG_WRITE);
        #endif

        if (err != 0)
            return false;
    }

    // write the stream header, if any
    #if LIBAVFORMAT_BUILD < CALC_FFMPEG_VERSION(53, 2, 0)
        av_write_header(oc_);
    #else
        avformat_write_header(oc_, NULL);
    #endif

    return true;
}

void OutputMediaStream_FFMPEG::write(unsigned char* data, int size, int keyFrame)
{
    // if zero size, it means the image was buffered
    if (size > 0)
    {
        AVPacket pkt;
        av_init_packet(&pkt);

        if (keyFrame)
            pkt.flags |= PKT_FLAG_KEY;

        pkt.stream_index = video_st_->index;
        pkt.data = data;
        pkt.size = size;

        // write the compressed frame in the media file
        av_write_frame(oc_, &pkt);
    }
}

struct OutputMediaStream_FFMPEG* create_OutputMediaStream_FFMPEG(const char* fileName, int width, int height, double fps)
{
    OutputMediaStream_FFMPEG* stream = (OutputMediaStream_FFMPEG*) malloc(sizeof(OutputMediaStream_FFMPEG));

    if (stream->open(fileName, width, height, fps))
        return stream;

    stream->close();
    free(stream);

    return 0;
}

void release_OutputMediaStream_FFMPEG(struct OutputMediaStream_FFMPEG* stream)
{
    stream->close();
    free(stream);
}

void write_OutputMediaStream_FFMPEG(struct OutputMediaStream_FFMPEG* stream, unsigned char* data, int size, int keyFrame)
{
    stream->write(data, size, keyFrame);
}

/*
 * For CUDA decoder
 */

enum
{
    VideoCodec_MPEG1 = 0,
    VideoCodec_MPEG2,
    VideoCodec_MPEG4,
    VideoCodec_VC1,
    VideoCodec_H264,
    VideoCodec_JPEG,
    VideoCodec_H264_SVC,
    VideoCodec_H264_MVC,

    // Uncompressed YUV
    VideoCodec_YUV420 = (('I'<<24)|('Y'<<16)|('U'<<8)|('V')),   // Y,U,V (4:2:0)
    VideoCodec_YV12   = (('Y'<<24)|('V'<<16)|('1'<<8)|('2')),   // Y,V,U (4:2:0)
    VideoCodec_NV12   = (('N'<<24)|('V'<<16)|('1'<<8)|('2')),   // Y,UV  (4:2:0)
    VideoCodec_YUYV   = (('Y'<<24)|('U'<<16)|('Y'<<8)|('V')),   // YUYV/YUY2 (4:2:2)
    VideoCodec_UYVY   = (('U'<<24)|('Y'<<16)|('V'<<8)|('Y')),   // UYVY (4:2:2)
};

enum
{
    VideoChromaFormat_Monochrome = 0,
    VideoChromaFormat_YUV420,
    VideoChromaFormat_YUV422,
    VideoChromaFormat_YUV444,
};

struct InputMediaStream_FFMPEG
{
public:
    bool open(const char* fileName, int* codec, int* chroma_format, int* width, int* height);
    void close();

    bool read(unsigned char** data, int* size, int* endOfFile);

private:
    InputMediaStream_FFMPEG(const InputMediaStream_FFMPEG&);
    InputMediaStream_FFMPEG& operator =(const InputMediaStream_FFMPEG&);

    AVFormatContext* ctx_;
    int video_stream_id_;
    AVPacket pkt_;
};

bool InputMediaStream_FFMPEG::open(const char* fileName, int* codec, int* chroma_format, int* width, int* height)
{
    int err;

    ctx_ = 0;
    video_stream_id_ = -1;
    memset(&pkt_, 0, sizeof(AVPacket));

    #if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 13, 0)
        avformat_network_init();
    #endif

    // register all codecs, demux and protocols
    av_register_all();

    av_log_set_level(AV_LOG_ERROR);

    #if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 6, 0)
        err = avformat_open_input(&ctx_, fileName, 0, 0);
    #else
        err = av_open_input_file(&ctx_, fileName, 0, 0, 0);
    #endif
    if (err < 0)
        return false;

    #if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 3, 0)
        err = avformat_find_stream_info(ctx_, 0);
    #else
        err = av_find_stream_info(ctx_);
    #endif
    if (err < 0)
        return false;

    for (unsigned int i = 0; i < ctx_->nb_streams; ++i)
    {
        #if LIBAVFORMAT_BUILD > 4628
            AVCodecContext *enc = ctx_->streams[i]->codec;
        #else
            AVCodecContext *enc = &ctx_->streams[i]->codec;
        #endif

        if (enc->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            video_stream_id_ = static_cast<int>(i);

            switch (enc->codec_id)
            {
            case CODEC_ID_MPEG1VIDEO:
                *codec = ::VideoCodec_MPEG1;
                break;

            case CODEC_ID_MPEG2VIDEO:
                *codec = ::VideoCodec_MPEG2;
                break;

            case CODEC_ID_MPEG4:
                *codec = ::VideoCodec_MPEG4;
                break;

            case CODEC_ID_VC1:
                *codec = ::VideoCodec_VC1;
                break;

            case CODEC_ID_H264:
                *codec = ::VideoCodec_H264;
                break;

            default:
                return false;
            };

            switch (enc->pix_fmt)
            {
            case PIX_FMT_YUV420P:
                *chroma_format = ::VideoChromaFormat_YUV420;
                break;

            case PIX_FMT_YUV422P:
                *chroma_format = ::VideoChromaFormat_YUV422;
                break;

            case PIX_FMT_YUV444P:
                *chroma_format = ::VideoChromaFormat_YUV444;
                break;

            default:
                return false;
            }

            *width = enc->coded_width;
            *height = enc->coded_height;

            break;
        }
    }

    if (video_stream_id_ < 0)
        return false;

    av_init_packet(&pkt_);

    return true;
}

void InputMediaStream_FFMPEG::close()
{
    if (ctx_)
    {
        #if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 24, 2)
            avformat_close_input(&ctx_);
        #else
            av_close_input_file(ctx_);
        #endif
    }

    // free last packet if exist
    if (pkt_.data)
        av_free_packet(&pkt_);
}

bool InputMediaStream_FFMPEG::read(unsigned char** data, int* size, int* endOfFile)
{
    // free last packet if exist
    if (pkt_.data)
        av_free_packet(&pkt_);

    // get the next frame
    for (;;)
    {
        int ret = av_read_frame(ctx_, &pkt_);

        if (ret == AVERROR(EAGAIN))
            continue;

        if (ret < 0)
        {
            if (ret == AVERROR_EOF)
                *endOfFile = true;
            return false;
        }

        if (pkt_.stream_index != video_stream_id_)
        {
            av_free_packet(&pkt_);
            continue;
        }

        break;
    }

    *data = pkt_.data;
    *size = pkt_.size;
    *endOfFile = false;

    return true;
}

InputMediaStream_FFMPEG* create_InputMediaStream_FFMPEG(const char* fileName, int* codec, int* chroma_format, int* width, int* height)
{
    InputMediaStream_FFMPEG* stream = (InputMediaStream_FFMPEG*) malloc(sizeof(InputMediaStream_FFMPEG));

    if (stream && stream->open(fileName, codec, chroma_format, width, height))
        return stream;

    stream->close();
    free(stream);

    return 0;
}

void release_InputMediaStream_FFMPEG(InputMediaStream_FFMPEG* stream)
{
    stream->close();
    free(stream);
}

int read_InputMediaStream_FFMPEG(InputMediaStream_FFMPEG* stream, unsigned char** data, int* size, int* endOfFile)
{
    return stream->read(data, size, endOfFile);
}
