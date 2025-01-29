/*M//////////////////////////////////////////////////////////
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
#include <xine.h>
#include <xine/xineutils.h>

using namespace cv;

class XINECapture : public IVideoCapture
{
    // method call table
    xine_t *xine;
    xine_stream_t *stream;
    xine_video_port_t *vo_port;
    xine_video_frame_t xine_frame;
    Size size;
    int frame_number;
    double frame_rate; // fps
    double frame_duration; // ms
    bool seekable;

  public:
    XINECapture()
        : xine(0), stream(0), vo_port(0), frame_number(-1), frame_rate(0.), frame_duration(0.),
          seekable(false)
    {
        xine_video_frame_t z = {};
        xine_frame = z;
    }

    ~XINECapture() { close(); }

    bool isOpened() const CV_OVERRIDE { return xine && stream; }

    int getCaptureDomain() CV_OVERRIDE { return CAP_XINE; }

    void close()
    {
        if (vo_port && xine_frame.data)
        {
            xine_free_video_frame(vo_port, &xine_frame);
        }
        if (stream)
        {
            xine_close(stream);
            stream = 0;
        }
        if (vo_port)
        {
            xine_close_video_driver(xine, vo_port);
            vo_port = 0;
        }
        if (xine)
        {
            xine_exit(xine);
            xine = 0;
        }
    }

    bool open(const char *filename)
    {
        CV_Assert_N(!xine, !stream, !vo_port);
        char configfile[2048] = {0};

        xine = xine_new();
        sprintf(configfile, "%s%s", xine_get_homedir(), "/.xine/config");
        xine_config_load(xine, configfile);
        xine_init(xine);
        xine_engine_set_param(xine, 0, 0);

        vo_port = xine_new_framegrab_video_port(xine);
        if (!vo_port)
            return false;

        stream = xine_stream_new(xine, NULL, vo_port);
        if (!xine_open(stream, filename))
            return false;

        // reset stream...
        if (!xine_play(stream, 0, 0))
            return false;

        // initialize some internals...
        frame_number = 0;


        if ( !xine_get_next_video_frame( vo_port, &xine_frame ) )
            return false;

        size = Size( xine_frame.width, xine_frame.height );

        xine_free_video_frame( vo_port, &xine_frame );
        xine_frame.data = 0;

        {
            xine_video_frame_t tmp;
            if (!xine_play( stream, 0, 300 )) /* 300msec */
                return false;
            if (!xine_get_next_video_frame( vo_port, &tmp ))
                return false;
            seekable = ( tmp.frame_number != 0 );
            xine_free_video_frame( vo_port, &tmp );
            if (!xine_play( stream, 0, 0 ))
                return false;
        }

        frame_duration = xine_get_stream_info( stream, XINE_STREAM_INFO_FRAME_DURATION ) / 90.;
        frame_rate = frame_duration > 0 ? 1000 / frame_duration : 0.;
        return true;
    }

    bool grabFrame() CV_OVERRIDE
    {
        CV_Assert(vo_port);
        bool res = xine_get_next_video_frame(vo_port, &xine_frame);
        if (res)
            frame_number++;
        return res;
    }

    bool retrieveFrame(int, OutputArray out) CV_OVERRIDE
    {
        CV_Assert(stream);
        CV_Assert(vo_port);

        if (xine_frame.data == 0)
            return false;

        bool res = false;
        Mat frame_bgr;

        switch (xine_frame.colorspace)
        {
        case XINE_IMGFMT_YV12: // actual format seems to be I420 (or IYUV)
        {
            Mat frame(Size(xine_frame.width, xine_frame.height * 3 / 2), CV_8UC1, xine_frame.data);
            cv::cvtColor(frame, out, cv::COLOR_YUV2BGR_I420);
            res = true;
        }
        break;

        case XINE_IMGFMT_YUY2:
        {
            Mat frame(Size(xine_frame.width, xine_frame.height), CV_8UC2, xine_frame.data);
            cv::cvtColor(frame, out, cv::COLOR_YUV2BGR_YUY2);
            res = true;
        }
        break;

        default:
            break;
        }

        // always release last xine_frame, not needed anymore
        xine_free_video_frame(vo_port, &xine_frame);
        xine_frame.data = 0;
        return res;
    }

    double getProperty(int property_id) const CV_OVERRIDE
    {
        CV_Assert_N(xine, vo_port, stream);
        int pos_t, pos_l, length;
        bool res = (bool)xine_get_pos_length(stream, &pos_l, &pos_t, &length);

        switch (property_id)
        {
        case CV_CAP_PROP_POS_MSEC: return res ? pos_t : 0;
        case CV_CAP_PROP_POS_FRAMES: return frame_number;
        case CV_CAP_PROP_POS_AVI_RATIO: return length && res ? pos_l / 65535.0 : 0.0;
        case CV_CAP_PROP_FRAME_WIDTH: return size.width;
        case CV_CAP_PROP_FRAME_HEIGHT: return size.height;
        case CV_CAP_PROP_FPS: return frame_rate;
        case CV_CAP_PROP_FOURCC: return (double)xine_get_stream_info(stream, XINE_STREAM_INFO_VIDEO_FOURCC);
        }
        return -1.0;
    }

    bool setProperty(int property_id, double value) CV_OVERRIDE
    {
        CV_Assert(stream);
        CV_Assert(vo_port);
        switch (property_id)
        {
        case CV_CAP_PROP_POS_MSEC: return seekTime((int)value);
        case CV_CAP_PROP_POS_FRAMES: return seekFrame((int)value);
        case CV_CAP_PROP_POS_AVI_RATIO: return seekRatio(value);
        default: return false;
        }
    }

protected:
    bool oldSeekFrame(int f)
    {
        CV_Assert_N(xine, vo_port, stream);
        // no need to seek if we are already there...
        if (f == frame_number)
        {
            return true;
        }
        else if (f > frame_number)
        {
            // if the requested position is behind out actual position,
            // we just need to read the remaining amount of frames until we are there.
            for (; frame_number < f; frame_number++)
            {
                // un-increment framenumber grabbing failed
                if (!xine_get_next_video_frame(vo_port, &xine_frame))
                {
                    frame_number--;
                    break;
                }
                else
                {
                    xine_free_video_frame(vo_port, &xine_frame);
                }
            }
        }
        else // f < frame_number
        {
            // otherwise we need to reset the stream and
            // start reading frames from the beginning.
            // reset stream, should also work with non-seekable input
            xine_play(stream, 0, 0);
            // read frames until we are at the requested frame
            for (frame_number = 0; frame_number < f; frame_number++)
            {
                // un-increment last framenumber if grabbing failed
                if (!xine_get_next_video_frame(vo_port, &xine_frame))
                {
                    frame_number--;
                    break;
                }
                else
                {
                    xine_free_video_frame(vo_port, &xine_frame);
                }
            }
        }
        return f == frame_number;
    }

    bool seekFrame(int f)
    {
        CV_Assert_N(xine, vo_port, stream);
        if (seekable)
        {
            int new_time = (int)((f + 1) * (float)frame_duration);
            if (xine_play(stream, 0, new_time))
            {
                frame_number = f;
                return true;
            }
        }
        else
        {
            return oldSeekFrame(f);
        }
        return false;
    }

    bool seekTime(int t)
    {
        CV_Assert_N(xine, vo_port, stream);
        if (seekable)
        {
            if (xine_play(stream, 0, t))
            {
                frame_number = (int)((double)t * frame_rate / 1000);
                return true;
            }
        }
        else
        {
            int new_frame = (int)((double)t * frame_rate / 1000);
            return oldSeekFrame(new_frame);
        }
        return false;
    }

    bool seekRatio(double ratio)
    {
        CV_Assert_N(xine, vo_port, stream);
        if (ratio > 1 || ratio < 0)
            return false;
        if (seekable)
        {
            // TODO: FIX IT, DOESN'T WORK PROPERLY, YET...!
            int pos_t, pos_l, length;
            bool res = (bool)xine_get_pos_length(stream, &pos_l, &pos_t, &length);
            if (res && xine_play(stream, (int)(ratio * (double)length), 0))
            {
                frame_number = (int)(ratio * length / frame_duration);
                return true;
            }
        }
        return false;
    }
};

Ptr<IVideoCapture> cv::createXINECapture(const char *filename)
{
    Ptr<XINECapture> res = makePtr<XINECapture>();
    if (res && res->open(filename))
        return res;
    return Ptr<IVideoCapture>();
}
