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

#ifdef HAVE_DC1394_2

#include <unistd.h>
#include <stdint.h>
#include <sys/select.h>
#include <dc1394/dc1394.h>
#include <stdlib.h>
#include <string.h>

static dc1394error_t adaptBufferStereoLocal(dc1394video_frame_t *in, dc1394video_frame_t *out)
{
    uint32_t bpp;

    // buffer position is not changed. Size is boubled in Y
    out->size[0] = in->size[0];
    out->size[1] = in->size[1] * 2;
    out->position[0] = in->position[0];
    out->position[1] = in->position[1];

    // color coding is set to mono8 or raw8.
    switch (in->color_coding)
    {
    case DC1394_COLOR_CODING_RAW16:
        out->color_coding = DC1394_COLOR_CODING_RAW8;
        break;
    case DC1394_COLOR_CODING_MONO16:
    case DC1394_COLOR_CODING_YUV422:
        out->color_coding = DC1394_COLOR_CODING_MONO8;
        break;
    default:
        return DC1394_INVALID_COLOR_CODING;
    }

    // keep the color filter value in all cases. if the format is not raw it will not be further used anyway
    out->color_filter = in->color_filter;

    // the output YUV byte order must be already set if the buffer is YUV422 at the output
    // if the output is not YUV we don't care about this field.
    // Hence nothing to do.
    // we always convert to 8bits (at this point) we can safely set this value to 8.
    out->data_depth = 8;

    // don't know what to do with stride... >>>> TODO: STRIDE SHOULD BE TAKEN INTO ACCOUNT... <<<<
    // out->stride=??
    // the video mode should not change. Color coding and other stuff can be accessed in specific fields of this struct
    out->video_mode = in->video_mode;

    // padding is kept:
    out->padding_bytes = in->padding_bytes;

    // image bytes changes:    >>>> TODO: STRIDE SHOULD BE TAKEN INTO ACCOUNT... <<<<
    dc1394_get_color_coding_bit_size(out->color_coding, &bpp);
    out->image_bytes = (out->size[0] * out->size[1] * bpp) / 8;

    // total is image_bytes + padding_bytes
    out->total_bytes = out->image_bytes + out->padding_bytes;

    // bytes-per-packet and packets_per_frame are internal data that can be kept as is.
    out->packet_size  = in->packet_size;
    out->packets_per_frame = in->packets_per_frame;

    // timestamp, frame_behind, id and camera are copied too:
    out->timestamp = in->timestamp;
    out->frames_behind = in->frames_behind;
    out->camera = in->camera;
    out->id = in->id;

    // verify memory allocation:
    if (out->total_bytes > out->allocated_image_bytes)
    {
        free(out->image);
        out->image = (uint8_t*)malloc(out->total_bytes * sizeof(uint8_t));
        out->allocated_image_bytes = out->total_bytes;
    }

    // Copy padding bytes:
    memcpy(&(out->image[out->image_bytes]), &(in->image[in->image_bytes]), out->padding_bytes);
    out->little_endian = DC1394_FALSE; // not used before 1.32 is out.
    out->data_in_padding = DC1394_FALSE; // not used before 1.32 is out.
    return DC1394_SUCCESS;
}

static dc1394error_t dc1394_deinterlace_stereo_frames_fixed(dc1394video_frame_t *in,
    dc1394video_frame_t *out, dc1394stereo_method_t method)
{
    dc1394error_t err;

    if((in->color_coding == DC1394_COLOR_CODING_RAW16) ||
       (in->color_coding == DC1394_COLOR_CODING_MONO16) ||
       (in->color_coding == DC1394_COLOR_CODING_YUV422))
    {
        switch (method)
        {

        case DC1394_STEREO_METHOD_INTERLACED:
            err = adaptBufferStereoLocal(in, out);
//FIXED by AB:
//          dc1394_deinterlace_stereo(in->image, out->image, in->size[0], in->size[1]);
            dc1394_deinterlace_stereo(in->image, out->image, out->size[0], out->size[1]);
            break;

        case DC1394_STEREO_METHOD_FIELD:
            err = adaptBufferStereoLocal(in, out);
            memcpy(out->image, in->image, out->image_bytes);
            break;
        }

        return DC1394_INVALID_STEREO_METHOD;
    }
    else
        return DC1394_FUNCTION_NOT_SUPPORTED;
}

static uint32_t getControlRegister(dc1394camera_t *camera, uint64_t offset)
{
    uint32_t value = 0;
    dc1394error_t err = dc1394_get_control_register(camera, offset, &value);

    assert(err == DC1394_SUCCESS);
    return err == DC1394_SUCCESS ? value : 0xffffffff;
}

struct CvDC1394
{
    CvDC1394();
    ~CvDC1394();

    dc1394_t* dc;
    fd_set camFds;
};

CvDC1394::CvDC1394()
{
    dc = dc1394_new();
    FD_ZERO(&camFds);
}

CvDC1394::~CvDC1394()
{
    if (dc)
        dc1394_free(dc);
    dc = 0;
}

static CvDC1394 dc1394;

class CvCaptureCAM_DC1394_v2_CPP : public CvCapture
{
public:
    CvCaptureCAM_DC1394_v2_CPP();
    virtual ~CvCaptureCAM_DC1394_v2_CPP()
    {
        close();
    }

    virtual bool open(int index);
    virtual void close();

    virtual double getProperty(int);
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
	virtual int getCaptureDomain() { return CV_CAP_DC1394; } // Return the type of the capture object: CV_CAP_VFW, etc...
	

protected:
    virtual bool startCapture();
    virtual bool getVidereCalibrationInfo( char* buf, int bufSize );
    virtual bool initVidereRectifyMaps( const char* info, IplImage* ml[2], IplImage* mr[2] );

    uint64_t guid;
    dc1394camera_t* dcCam;
    int isoSpeed;
    int videoMode;
    int frameWidth, frameHeight;
    double fps;
    int nDMABufs;
    bool started;

    enum { VIDERE = 0x5505 };

    int cameraId;
    bool colorStereo;
    dc1394bayer_method_t bayer;
    dc1394color_filter_t bayerFilter;

    enum { NIMG = 2 };
    IplImage *img[NIMG];
    dc1394video_frame_t* frameC;
    int nimages;

    bool rectify;
    bool init_rectify;
    IplImage *maps[NIMG][2];
};

CvCaptureCAM_DC1394_v2_CPP::CvCaptureCAM_DC1394_v2_CPP()
{
    guid = 0;
    dcCam = 0;
    isoSpeed = 400;
    fps = 15;
    nDMABufs = 8;
    started = false;
    cameraId = 0;
    colorStereo = false;
    bayer = DC1394_BAYER_METHOD_BILINEAR;
    bayerFilter = DC1394_COLOR_FILTER_GRBG;
    frameWidth = 640;
    frameHeight = 480;

    for (int i = 0; i < NIMG; i++)
        img[i] = maps[i][0] = maps[i][1] = 0;
    frameC = 0;
    nimages = 1;
    rectify = false;
}


bool CvCaptureCAM_DC1394_v2_CPP::startCapture()
{
    int i;
    int code = 0;
    if (!dcCam)
        return false;
    if (isoSpeed > 0)
    {
        code = dc1394_video_set_iso_speed(dcCam,
                                          isoSpeed <= 100 ? DC1394_ISO_SPEED_100 :
                                          isoSpeed <= 200 ? DC1394_ISO_SPEED_200 :
                                          isoSpeed <= 400 ? DC1394_ISO_SPEED_400 :
                                          isoSpeed <= 800 ? DC1394_ISO_SPEED_800 :
                                          isoSpeed == 1600 ? DC1394_ISO_SPEED_1600 :
                                          DC1394_ISO_SPEED_3200);
    }

    if (frameWidth > 0 || frameHeight > 0)
    {
        dc1394video_mode_t bestMode = (dc1394video_mode_t) - 1;
        dc1394video_modes_t videoModes;
        dc1394_video_get_supported_modes(dcCam, &videoModes);
        for (i = 0; i < (int)videoModes.num; i++)
        {
            dc1394video_mode_t mode = videoModes.modes[i];
			if (mode >= DC1394_VIDEO_MODE_FORMAT7_MIN && mode <= DC1394_VIDEO_MODE_FORMAT7_MAX)
				continue;
            int pref = -1;
            dc1394color_coding_t colorCoding;
            dc1394_get_color_coding_from_video_mode(dcCam, mode, &colorCoding);

            uint32_t width, height;
            dc1394_get_image_size_from_video_mode(dcCam, mode, &width, &height);
            if ((int)width == frameWidth || (int)height == frameHeight)
            {
                if (colorCoding == DC1394_COLOR_CODING_RGB8 ||
                        colorCoding == DC1394_COLOR_CODING_RGB16 ||
                        colorCoding == DC1394_COLOR_CODING_RAW8 ||
                        colorCoding == DC1394_COLOR_CODING_RAW16)
                {
                    bestMode = mode;
                    break;
                }

                if (colorCoding == DC1394_COLOR_CODING_YUV411 ||
                        colorCoding == DC1394_COLOR_CODING_YUV422 ||
                        (colorCoding == DC1394_COLOR_CODING_YUV444 &&
                        pref < 1))
                {
                    bestMode = mode;
                    pref = 1;
                }

                if (colorCoding == DC1394_COLOR_CODING_MONO8 ||
                        (colorCoding == DC1394_COLOR_CODING_MONO16 &&
                        pref < 0))
                {
                    bestMode = mode;
                    pref = 0;
                }
            }
        }
        if ((int)bestMode >= 0)
            code = dc1394_video_set_mode(dcCam, bestMode);
    }

    if (fps > 0)
    {
        dc1394video_mode_t mode;
        dc1394framerates_t framerates;
        double minDiff = DBL_MAX;
        dc1394framerate_t bestFps = (dc1394framerate_t) - 1;

        dc1394_video_get_mode(dcCam, &mode);
        dc1394_video_get_supported_framerates(dcCam, mode, &framerates);

        for (i = 0; i < (int)framerates.num; i++)
        {
            dc1394framerate_t ifps = framerates.framerates[i];
            double fps1 = (1 << (ifps - DC1394_FRAMERATE_1_875)) * 1.875;
            double diff = fabs(fps1 - fps);
            if (diff < minDiff)
            {
                minDiff = diff;
                bestFps = ifps;
            }
        }
        if ((int)bestFps >= 0)
            code = dc1394_video_set_framerate(dcCam, bestFps);
    }

    if (cameraId == VIDERE)
    {
        bayerFilter = DC1394_COLOR_FILTER_GBRG;
        nimages = 2;
        uint32_t value = 0;
        dc1394_get_control_register(dcCam, 0x50c, &value);
        colorStereo = (value & 0x80000000) != 0;
    }

    code = dc1394_capture_setup(dcCam, nDMABufs, DC1394_CAPTURE_FLAGS_DEFAULT);
    if (code >= 0)
    {
        FD_SET(dc1394_capture_get_fileno(dcCam), &dc1394.camFds);
        dc1394_video_set_transmission(dcCam, DC1394_ON);
        if (cameraId == VIDERE)
        {
            enum { PROC_MODE_OFF, PROC_MODE_NONE, PROC_MODE_TEST, PROC_MODE_RECTIFIED, PROC_MODE_DISPARITY, PROC_MODE_DISPARITY_RAW };
            int procMode = PROC_MODE_RECTIFIED;
            usleep(100000);
            uint32_t qval1 = 0x08000000 | (0x90 << 16) | ((procMode & 0x7) << 16);
            uint32_t qval2 = 0x08000000 | (0x9C << 16);
            dc1394_set_control_register(dcCam, 0xFF000, qval1);
            dc1394_set_control_register(dcCam, 0xFF000, qval2);
        }
        started = true;
    }

    return code >= 0;
}

bool CvCaptureCAM_DC1394_v2_CPP::open(int index)
{
    bool result = false;
    dc1394camera_list_t* cameraList = 0;
    dc1394error_t err;

    close();

    if (!dc1394.dc)
        goto _exit_;

    err = dc1394_camera_enumerate(dc1394.dc, &cameraList);
    if (err < 0 || !cameraList || (unsigned)index >= (unsigned)cameraList->num)
        goto _exit_;

    guid = cameraList->ids[index].guid;
    dcCam = dc1394_camera_new(dc1394.dc, guid);
    if (!dcCam)
        goto _exit_;

    cameraId = dcCam->vendor_id;
    result = true;

_exit_:
    if (cameraList)
        dc1394_camera_free_list(cameraList);

    return result;
}

void CvCaptureCAM_DC1394_v2_CPP::close()
{
    if (dcCam)
    {
        if (FD_ISSET(dc1394_capture_get_fileno(dcCam), &dc1394.camFds))
            FD_CLR(dc1394_capture_get_fileno(dcCam), &dc1394.camFds);
        dc1394_video_set_transmission(dcCam, DC1394_OFF);
        dc1394_capture_stop(dcCam);
        dc1394_camera_free(dcCam);
        dcCam = 0;
        started = false;
    }

    for (int i = 0; i < NIMG; i++)
    {
        cvReleaseImage(&img[i]);
        cvReleaseImage(&maps[i][0]);
        cvReleaseImage(&maps[i][1]);
    }
    if (frameC)
    {
        if (frameC->image)
            free(frameC->image);
        free(frameC);
        frameC = 0;
    }
}


bool CvCaptureCAM_DC1394_v2_CPP::grabFrame()
{
    dc1394capture_policy_t policy = DC1394_CAPTURE_POLICY_WAIT;
    bool code = false, isColor;
    dc1394video_frame_t *dcFrame = 0, *fs = 0;
    int i, nch;

    if (!dcCam || (!started && !startCapture()))
        return false;

    dc1394_capture_dequeue(dcCam, policy, &dcFrame);

    if (!dcFrame)
        return false;

    if (/*dcFrame->frames_behind > 1 ||*/ dc1394_capture_is_frame_corrupt(dcCam, dcFrame) == DC1394_TRUE)
    {
        goto _exit_;
    }

    isColor = dcFrame->color_coding != DC1394_COLOR_CODING_MONO8 &&
              dcFrame->color_coding != DC1394_COLOR_CODING_MONO16 &&
              dcFrame->color_coding != DC1394_COLOR_CODING_MONO16S;

    if (nimages == 2)
    {
        fs = (dc1394video_frame_t*)calloc(1, sizeof(*fs));

        //dc1394_deinterlace_stereo_frames(dcFrame, fs, DC1394_STEREO_METHOD_INTERLACED);
        dc1394_deinterlace_stereo_frames_fixed(dcFrame, fs, DC1394_STEREO_METHOD_INTERLACED);

        dc1394_capture_enqueue(dcCam, dcFrame); // release the captured frame as soon as possible
        dcFrame = 0;
        if (!fs->image)
            goto _exit_;
        isColor = colorStereo;
    }
    nch = isColor ? 3 : 1;

    for (i = 0; i < nimages; i++)
    {
        IplImage fhdr;
        dc1394video_frame_t f = fs ? *fs : *dcFrame, *fc = &f;
        f.size[1] /= nimages;
        f.image += f.size[0] * f.size[1] * i; // TODO: make it more universal
        if (isColor)
        {
            if (!frameC)
                frameC = (dc1394video_frame_t*)calloc(1, sizeof(*frameC));
            frameC->color_coding = nch == 3 ? DC1394_COLOR_CODING_RGB8 : DC1394_COLOR_CODING_MONO8;
            if (nimages == 1)
            {
                dc1394_convert_frames(&f, frameC);
                dc1394_capture_enqueue(dcCam, dcFrame);
                dcFrame = 0;
            }
            else
            {
                f.color_filter = bayerFilter;
                dc1394_debayer_frames(&f, frameC, bayer);
            }
            fc = frameC;
        }
        if (!img[i])
            img[i] = cvCreateImage(cvSize(fc->size[0], fc->size[1]), 8, nch);
        cvInitImageHeader(&fhdr, cvSize(fc->size[0], fc->size[1]), 8, nch);
        cvSetData(&fhdr, fc->image, fc->size[0]*nch);

	// Swap R&B channels:
	if (nch==3)
		cvConvertImage(&fhdr,&fhdr,CV_CVTIMG_SWAP_RB);

        if( rectify && cameraId == VIDERE && nimages == 2 )
        {
            if( !maps[0][0] || maps[0][0]->width != img[i]->width || maps[0][0]->height != img[i]->height )
            {
                CvSize size = cvGetSize(img[i]);
                cvReleaseImage(&maps[0][0]);
                cvReleaseImage(&maps[0][1]);
                cvReleaseImage(&maps[1][0]);
                cvReleaseImage(&maps[1][1]);
                maps[0][0] = cvCreateImage(size, IPL_DEPTH_16S, 2);
                maps[0][1] = cvCreateImage(size, IPL_DEPTH_16S, 1);
                maps[1][0] = cvCreateImage(size, IPL_DEPTH_16S, 2);
                maps[1][1] = cvCreateImage(size, IPL_DEPTH_16S, 1);
                char buf[4*4096];
                if( getVidereCalibrationInfo( buf, (int)sizeof(buf) ) &&
                    initVidereRectifyMaps( buf, maps[0], maps[1] ))
                    ;
                else
                    rectify = false;
            }
            cvRemap(&fhdr, img[i], maps[i][0], maps[i][1]);
        }
        else
            cvCopy(&fhdr, img[i]);
    }

    code = true;

_exit_:
    if (dcFrame)
        dc1394_capture_enqueue(dcCam, dcFrame);
    if (fs)
    {
        if (fs->image)
            free(fs->image);
        free(fs);
    }

    return code;
}

IplImage* CvCaptureCAM_DC1394_v2_CPP::retrieveFrame(int idx)
{
    return 0 <= idx && idx < nimages ? img[idx] : 0;
}

double CvCaptureCAM_DC1394_v2_CPP::getProperty(int propId)
{
    switch (propId)
    {
    case CV_CAP_PROP_FRAME_WIDTH:
        return frameWidth ? frameWidth : frameHeight*4 / 3;
    case CV_CAP_PROP_FRAME_HEIGHT:
        return frameHeight ? frameHeight : frameWidth*3 / 4;
    case CV_CAP_PROP_FPS:
        return fps;
    case CV_CAP_PROP_RECTIFICATION:
        return rectify ? 1 : 0;
//    case CV_CAP_PROP_BRIGHTNESS :
//    case CV_CAP_PROP_CONTRAST :
//    case CV_CAP_PROP_WHITE_BALANCE :
    default:
        ;
    }
    return 0;
}

bool CvCaptureCAM_DC1394_v2_CPP::setProperty(int propId, double value)
{
    switch (propId)
    {
    case CV_CAP_PROP_FRAME_WIDTH:
        if(started)
			return false;
        frameWidth = cvRound(value);
        frameHeight = 0;
        break;
    case CV_CAP_PROP_FRAME_HEIGHT:
    	if(started)
			return false;
        frameWidth = 0;
        frameHeight = cvRound(value);
        break;
    case CV_CAP_PROP_FPS:
    	if(started)
			return false;
        fps = value;
        break;
    case CV_CAP_PROP_RECTIFICATION:
        if( cameraId != VIDERE )
            return false;
        rectify = fabs(value) > FLT_EPSILON;
        break;
    default:
        return false;
    }
    return true;
}


bool CvCaptureCAM_DC1394_v2_CPP::getVidereCalibrationInfo( char* buf, int bufSize )
{
    int pos;

    for( pos = 0; pos < bufSize - 4; pos += 4 )
    {
        uint32_t quad = getControlRegister(dcCam, 0xF0800 + pos);
        if( quad == 0 || quad == 0xffffffff )
            break;
        buf[pos] = (uchar)(quad >> 24);
        buf[pos+1] = (uchar)(quad >> 16);
        buf[pos+2] = (uchar)(quad >> 8);
        buf[pos+3] = (uchar)(quad);
    }

    if( pos == 0 )
        return false;

    buf[pos] = '\0';
    return true;
}


bool CvCaptureCAM_DC1394_v2_CPP::initVidereRectifyMaps( const char* info,
    IplImage* ml[2], IplImage* mr[2] )
{
    float identity_data[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    CvMat l_rect = cvMat(3, 3, CV_32F, identity_data), r_rect = l_rect;
    float l_intrinsic_data[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float r_intrinsic_data[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    CvMat l_intrinsic = cvMat(3, 3, CV_32F, l_intrinsic_data);
    CvMat r_intrinsic = cvMat(3, 3, CV_32F, r_intrinsic_data);
    float l_distortion_data[] = {0,0,0,0,0}, r_distortion_data[] = {0,0,0,0,0};
    CvMat l_distortion = cvMat(1, 5, CV_32F, l_distortion_data);
    CvMat r_distortion = cvMat(1, 5, CV_32F, r_distortion_data);
    IplImage* mx = cvCreateImage(cvGetSize(ml[0]), IPL_DEPTH_32F, 1);
    IplImage* my = cvCreateImage(cvGetSize(ml[0]), IPL_DEPTH_32F, 1);
    int k, j;

    for( k = 0; k < 2; k++ )
    {
        const char* section_name = k == 0 ? "[left_camera]" : "[right_camera]";
        static const char* param_names[] = { "f ", "fy", "Cx", "Cy" "kappa1", "kappa2", "tau1", "tau2", "kappa3", 0 };
        const char* section_start = strstr( info, section_name );
        CvMat* intrinsic = k == 0 ? &l_intrinsic : &r_intrinsic;
        CvMat* distortion = k == 0 ? &l_distortion : &r_distortion;
        CvMat* rectification = k == 0 ? &l_rect : &r_rect;
        IplImage** dst = k == 0 ? ml : mr;
        if( !section_start )
            break;
        section_start += strlen(section_name);
        for( j = 0; param_names[j] != 0; j++ )
        {
            const char* param_value_start = strstr(section_start, param_names[j]);
            float val=0;
            if(!param_value_start)
                break;
            sscanf(param_value_start + strlen(param_names[j]), "%f", &val);
            if( j < 4 )
                intrinsic->data.fl[j == 0 ? 0 : j == 1 ? 4 : j == 2 ? 2 : 5] = val;
            else
                distortion->data.fl[j - 4] = val;
        }
        if( param_names[j] != 0 )
            break;

        // some sanity check for the principal point
        if( fabs(mx->width*0.5 - intrinsic->data.fl[2]) > mx->width*0.1 ||
            fabs(my->height*0.5 - intrinsic->data.fl[5]) > my->height*0.1 )
        {
            cvScale( &intrinsic, &intrinsic, 0.5 ); // try the corrected intrinsic matrix for 2x lower resolution
            if( fabs(mx->width*0.5 - intrinsic->data.fl[2]) > mx->width*0.05 ||
                fabs(my->height*0.5 - intrinsic->data.fl[5]) > my->height*0.05 )
                cvScale( &intrinsic, &intrinsic, 2 ); // revert it back if the new variant is not much better
            intrinsic->data.fl[8] = 1;
        }

        cvInitUndistortRectifyMap( intrinsic, distortion,
                    rectification, intrinsic, mx, my );
        cvConvertMaps( mx, my, dst[0], dst[1] );
    }

    cvReleaseImage( &mx );
    cvReleaseImage( &my );
    return k >= 2;
}


CvCapture* cvCreateCameraCapture_DC1394_2(int index)
{
    CvCaptureCAM_DC1394_v2_CPP* capture = new CvCaptureCAM_DC1394_v2_CPP;

    if (capture->open(index))
        return capture;

    delete capture;
    return 0;
}

#endif
