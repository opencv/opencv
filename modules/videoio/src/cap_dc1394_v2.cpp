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
#include "cap_interface.hpp"

#ifdef HAVE_DC1394_2

#include <unistd.h>
#include <stdint.h>
#ifdef _WIN32
  // On Windows, we have no sys/select.h, but we need to pick up
  // select() which is in winsock2.
  #ifndef __SYS_SELECT_H__
    #define __SYS_SELECT_H__ 1
    #include <winsock2.h>
  #endif
#else
  #include <sys/select.h>
#endif /*_WIN32*/
#include <dc1394/dc1394.h>
#include <stdlib.h>
#include <string.h>

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

static CvDC1394& getDC1394()
{
    static CvDC1394 dc1394;
    return dc1394;
}

class CvCaptureCAM_DC1394_v2_CPP : public CvCapture
{
public:
    static int dc1394properties[CV_CAP_PROP_MAX_DC1394];
    CvCaptureCAM_DC1394_v2_CPP();
    virtual ~CvCaptureCAM_DC1394_v2_CPP()
    {
        close();
    }

    virtual bool open(int index);
    virtual void close();

    virtual double getProperty(int) const CV_OVERRIDE;
    virtual bool setProperty(int, double) CV_OVERRIDE;
    virtual bool grabFrame() CV_OVERRIDE;
    virtual IplImage* retrieveFrame(int) CV_OVERRIDE;
    virtual int getCaptureDomain() CV_OVERRIDE { return CV_CAP_DC1394; }


protected:
    virtual bool startCapture();

    uint64_t guid;
    dc1394camera_t* dcCam;
    int isoSpeed;
    int videoMode;
    int frameWidth, frameHeight;
    double fps;
    int nDMABufs;
    bool started;
    int userMode;

    enum { VIDERE = 0x5505 };

    int cameraId;
    bool colorStereo;
    dc1394bayer_method_t bayer;
    dc1394color_filter_t bayerFilter;

    enum { NIMG = 2 };
    IplImage *img[NIMG];
    dc1394video_frame_t* frameC;
    int nimages;

    dc1394featureset_t feature_set;
};
//mapping CV_CAP_PROP_ to DC1394_FEATUREs
int CvCaptureCAM_DC1394_v2_CPP::dc1394properties[CV_CAP_PROP_MAX_DC1394] = {
-1, //no corresponding feature for CV_CAP_PROP_POS_MSEC
-1,-1,-1,-1,
DC1394_FEATURE_FRAME_RATE, //CV_CAP_PROP_FPS - fps can be set for format 7 only!
-1,-1,-1,-1,
DC1394_FEATURE_BRIGHTNESS, //CV_CAP_PROP_BRIGHTNESS 10
-1,
DC1394_FEATURE_SATURATION, //CV_CAP_PROP_SATURATION
DC1394_FEATURE_HUE,
DC1394_FEATURE_GAIN,
DC1394_FEATURE_SHUTTER, //CV_CAP_PROP_EXPOSURE
-1, //CV_CAP_PROP_CONVERT_RGB
DC1394_FEATURE_WHITE_BALANCE, //corresponds to CV_CAP_PROP_WHITE_BALANCE_BLUE_U and CV_CAP_PROP_WHITE_BALANCE_RED_V, see set function to check these props are set
-1,-1,
DC1394_FEATURE_SHARPNESS, //20
DC1394_FEATURE_EXPOSURE, //CV_CAP_PROP_AUTO_EXPOSURE - this is auto exposure according to the IIDC standard
DC1394_FEATURE_GAMMA, //CV_CAP_PROP_GAMMA
DC1394_FEATURE_TEMPERATURE, //CV_CAP_PROP_TEMPERATURE
DC1394_FEATURE_TRIGGER, //CV_CAP_PROP_TRIGGER
DC1394_FEATURE_TRIGGER_DELAY, //CV_CAP_PROP_TRIGGER_DELAY
DC1394_FEATURE_WHITE_BALANCE, //CV_CAP_PROP_WHITE_BALANCE_RED_V
DC1394_FEATURE_ZOOM, //CV_CAP_PROP_ZOOM
DC1394_FEATURE_FOCUS, //CV_CAP_PROP_FOCUS
-1 //CV_CAP_PROP_GUID
};
CvCaptureCAM_DC1394_v2_CPP::CvCaptureCAM_DC1394_v2_CPP()
{
    guid = 0;
    dcCam = 0;
    isoSpeed = 400;
    fps = 15;
    // Reset the value here to 1 in order to ensure only a single frame is stored in the buffer!
    nDMABufs = 8;
    started = false;
    cameraId = 0;
    colorStereo = false;
    bayer = DC1394_BAYER_METHOD_BILINEAR;
    bayerFilter = DC1394_COLOR_FILTER_GRBG;
    frameWidth = 640;
    frameHeight = 480;

    for (int i = 0; i < NIMG; i++)
        img[i] = 0;
    frameC = 0;
    nimages = 1;
    userMode = -1;
}


bool CvCaptureCAM_DC1394_v2_CPP::startCapture()
{
    int i;
    int code = 0;
    if (!dcCam)
        return false;
    if (isoSpeed > 0)
    {
        // if capable set operation mode to 1394b for iso speeds above 400
        if (isoSpeed > 400 && dcCam->bmode_capable == DC1394_TRUE)
        {
            dc1394_video_set_operation_mode(dcCam, DC1394_OPERATION_MODE_1394B);
        }
        code = dc1394_video_set_iso_speed(dcCam,
                                          isoSpeed <= 100 ? DC1394_ISO_SPEED_100 :
                                          isoSpeed <= 200 ? DC1394_ISO_SPEED_200 :
                                          isoSpeed <= 400 ? DC1394_ISO_SPEED_400 :
                                          isoSpeed <= 800 ? DC1394_ISO_SPEED_800 :
                                          isoSpeed == 1600 ? DC1394_ISO_SPEED_1600 :
                                          DC1394_ISO_SPEED_3200);
    }

    // should a specific mode be used
    if (userMode >= 0)

    {
        dc1394video_mode_t wantedMode;
        dc1394video_modes_t videoModes;
        dc1394_video_get_supported_modes(dcCam, &videoModes);

        //set mode from number, for example the second supported mode, i.e userMode = 1

        if (userMode < (int)videoModes.num)
        {
            wantedMode = videoModes.modes[userMode];
        }

        //set modes directly from DC134 constants (from dc1394video_mode_t)
        else if ((userMode >= DC1394_VIDEO_MODE_MIN) && (userMode <= DC1394_VIDEO_MODE_MAX ))
        {
            //search for wanted mode, to check if camera supports it
            int j = 0;
            while ((j< (int)videoModes.num) && videoModes.modes[j]!=userMode)
            {
                j++;
            }

            if ((int)videoModes.modes[j]==userMode)
            {
                wantedMode = videoModes.modes[j];
            }
            else
            {
                userMode = -1;  // wanted mode not supported, search for best mode
            }
        }
        else
        {
            userMode = -1;      // wanted mode not supported, search for best mode
        }
        //if userMode is available: set it and update size
        if (userMode != -1)
        {
            code = dc1394_video_set_mode(dcCam, wantedMode);
            uint32_t width, height;
            dc1394_get_image_size_from_video_mode(dcCam, wantedMode, &width, &height);
            frameWidth  = (int)width;
            frameHeight = (int)height;
        }
    }

    if (userMode == -1 && (frameWidth > 0 || frameHeight > 0))
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
                        colorCoding == DC1394_COLOR_CODING_RAW8)
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
                    break;
                }

                if (colorCoding == DC1394_COLOR_CODING_MONO8)
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
        FD_SET(dc1394_capture_get_fileno(dcCam), &getDC1394().camFds);
        dc1394_video_set_transmission(dcCam, DC1394_ON);
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

    if (!getDC1394().dc)
        goto _exit_;

    err = dc1394_camera_enumerate(getDC1394().dc, &cameraList);
    if (err < 0 || !cameraList || (unsigned)index >= (unsigned)cameraList->num)
        goto _exit_;

    guid = cameraList->ids[index].guid;
    dcCam = dc1394_camera_new(getDC1394().dc, guid);
    if (!dcCam)
        goto _exit_;

    cameraId = dcCam->vendor_id;
    //get all features
    if (dc1394_feature_get_all(dcCam,&feature_set) == DC1394_SUCCESS)
        result = true;
    else
        result = false;

_exit_:
    if (cameraList)
        dc1394_camera_free_list(cameraList);

    return result;
}

void CvCaptureCAM_DC1394_v2_CPP::close()
{
    if (dcCam)
    {
        // check for fileno valid before using
        int fileno=dc1394_capture_get_fileno(dcCam);

        if (fileno>=0 && FD_ISSET(fileno, &getDC1394().camFds))
            FD_CLR(fileno, &getDC1394().camFds);
        dc1394_video_set_transmission(dcCam, DC1394_OFF);
        dc1394_capture_stop(dcCam);
        dc1394_camera_free(dcCam);
        dcCam = 0;
        started = false;
    }

    for (int i = 0; i < NIMG; i++)
    {
        cvReleaseImage(&img[i]);
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
        dc1394_deinterlace_stereo_frames(dcFrame, fs, DC1394_STEREO_METHOD_INTERLACED);
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
        {
            cv::Mat tmp = cv::cvarrToMat(&fhdr);
            cv::cvtColor(tmp, tmp, cv::COLOR_RGB2BGR, tmp.channels());
        }

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

double CvCaptureCAM_DC1394_v2_CPP::getProperty(int propId) const
{
    // Simulate mutable (C++11-like) member variable
    dc1394featureset_t& fs = const_cast<dc1394featureset_t&>(feature_set);

    switch (propId)
    {
    case CV_CAP_PROP_FRAME_WIDTH:
        return frameWidth ? frameWidth : frameHeight*4 / 3;
    case CV_CAP_PROP_FRAME_HEIGHT:
        return frameHeight ? frameHeight : frameWidth*3 / 4;
    case CV_CAP_PROP_FPS:
        return fps;
    case CV_CAP_PROP_RECTIFICATION:
        CV_LOG_WARNING(NULL, "cap_dc1394: rectification support has been removed from videoio module");
        return 0;
    case CV_CAP_PROP_WHITE_BALANCE_BLUE_U:
        if (dc1394_feature_whitebalance_get_value(dcCam,
                                                  &fs.feature[DC1394_FEATURE_WHITE_BALANCE-DC1394_FEATURE_MIN].BU_value,
                                                  &fs.feature[DC1394_FEATURE_WHITE_BALANCE-DC1394_FEATURE_MIN].RV_value) == DC1394_SUCCESS)
        return feature_set.feature[DC1394_FEATURE_WHITE_BALANCE-DC1394_FEATURE_MIN].BU_value;
        break;
    case CV_CAP_PROP_WHITE_BALANCE_RED_V:
        if (dc1394_feature_whitebalance_get_value(dcCam,
                                                  &fs.feature[DC1394_FEATURE_WHITE_BALANCE-DC1394_FEATURE_MIN].BU_value,
                                                  &fs.feature[DC1394_FEATURE_WHITE_BALANCE-DC1394_FEATURE_MIN].RV_value) == DC1394_SUCCESS)
        return feature_set.feature[DC1394_FEATURE_WHITE_BALANCE-DC1394_FEATURE_MIN].RV_value;
        break;
    case CV_CAP_PROP_GUID:
        //the least 32 bits are enough to identify the camera
        return (double) (guid & 0x00000000FFFFFFFF);
        break;
    case CV_CAP_PROP_MODE:
        return (double) userMode;
        break;
    case CV_CAP_PROP_ISO_SPEED:
        return (double) isoSpeed;
    case CV_CAP_PROP_BUFFERSIZE:
        return (double) nDMABufs;
    default:
        if (propId<CV_CAP_PROP_MAX_DC1394 && dc1394properties[propId]!=-1
            && dcCam)
            //&& feature_set.feature[dc1394properties[propId]-DC1394_FEATURE_MIN].on_off_capable)
            if (dc1394_feature_get_value(dcCam,(dc1394feature_t)dc1394properties[propId],
                &fs.feature[dc1394properties[propId]-DC1394_FEATURE_MIN].value) == DC1394_SUCCESS)
              return feature_set.feature[dc1394properties[propId]-DC1394_FEATURE_MIN].value;
    }
    return -1; // the value of the feature can be 0, so returning 0 as an error is wrong
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
        CV_LOG_WARNING(NULL, "cap_dc1394: rectification support has been removed from videoio module");
        return false;
    case CV_CAP_PROP_MODE:
        if(started)
          return false;
        userMode = cvRound(value);
        break;
    case CV_CAP_PROP_ISO_SPEED:
        if(started)
          return false;
        isoSpeed = cvRound(value);
        break;
    case CV_CAP_PROP_BUFFERSIZE:
        if(started)
            return false;
        nDMABufs = value;
        break;
        //The code below is based on coriander, callbacks.c:795, refer to case RANGE_MENU_MAN :
         default:
             if (propId<CV_CAP_PROP_MAX_DC1394 && dc1394properties[propId]!=-1
                 && dcCam)
             {
                 //get the corresponding feature from property-id
                 dc1394feature_info_t *act_feature = &feature_set.feature[dc1394properties[propId]-DC1394_FEATURE_MIN];

                 if (cvRound(value) == CV_CAP_PROP_DC1394_OFF)
                 {
                     if (  (act_feature->on_off_capable)
                           && (dc1394_feature_set_power(dcCam, act_feature->id, DC1394_OFF) == DC1394_SUCCESS))
                     {
                         act_feature->is_on=DC1394_OFF;
                         return true;
                     }
                     return false;
                 }
                 //try to turn the feature ON, feature can be ON and at the same time it can be not capable to change state to OFF
                 if ( (act_feature->is_on == DC1394_OFF) && (act_feature->on_off_capable == DC1394_TRUE))
                 {
                     if (dc1394_feature_set_power(dcCam, act_feature->id, DC1394_ON) == DC1394_SUCCESS)
                       feature_set.feature[dc1394properties[propId]-DC1394_FEATURE_MIN].is_on=DC1394_ON;
                 }
                 //turn off absolute mode - the actual value will be stored in the value field,
                 //otherwise it would be stored into CSR (control and status register) absolute value
                 if (act_feature->absolute_capable
                     && dc1394_feature_set_absolute_control(dcCam, act_feature->id, DC1394_OFF) !=DC1394_SUCCESS)
                     return false;
                 else
                     act_feature->abs_control=DC1394_OFF;
                 //set AUTO
                 if (cvRound(value) == CV_CAP_PROP_DC1394_MODE_AUTO)
                 {
                     if (dc1394_feature_set_mode(dcCam, act_feature->id, DC1394_FEATURE_MODE_AUTO)!=DC1394_SUCCESS)
                         return false;
                     act_feature->current_mode=DC1394_FEATURE_MODE_AUTO;
                     return true;
                 }
                 //set ONE PUSH
                 if (cvRound(value) == CV_CAP_PROP_DC1394_MODE_ONE_PUSH_AUTO)
                 {
                     //have to set to manual first, otherwise one push will be ignored (AVT  manual 4.3.0 p. 115)
                     if (dc1394_feature_set_mode(dcCam, act_feature->id, DC1394_FEATURE_MODE_ONE_PUSH_AUTO)!=DC1394_SUCCESS)
                         return false;
                     //will change to
                     act_feature->current_mode=DC1394_FEATURE_MODE_ONE_PUSH_AUTO;
                     return true;
                 }
                 //set the feature to MANUAL mode,
                 if (dc1394_feature_set_mode(dcCam, act_feature->id, DC1394_FEATURE_MODE_MANUAL)!=DC1394_SUCCESS)
                     return false;
                 else
                     act_feature->current_mode=DC1394_FEATURE_MODE_MANUAL;
                 // if property is one of the white balance features treat it in different way
                 if (propId == CV_CAP_PROP_WHITE_BALANCE_BLUE_U)
                 {
                     if (dc1394_feature_whitebalance_set_value(dcCam,cvRound(value), act_feature->RV_value)!=DC1394_SUCCESS)
                         return false;
                     else
                     {
                         act_feature->BU_value = cvRound(value);
                         return true;
                     }
                 }
                 if (propId == CV_CAP_PROP_WHITE_BALANCE_RED_V)
                 {
                     if (dc1394_feature_whitebalance_set_value(dcCam, act_feature->BU_value, cvRound(value))!=DC1394_SUCCESS)
                         return false;
                     else
                     {
                         act_feature->RV_value = cvRound(value);
                         return true;
                     }
                 }

                 //first: check boundaries
                 if (value < act_feature->min)
                 {
                     value = act_feature->min;
                 }
                 else if (value > act_feature->max)
                 {
                     value = act_feature->max;
                 }

                 if (dc1394_feature_set_value(dcCam, act_feature->id, cvRound(value)) == DC1394_SUCCESS)
                 {
                     act_feature->value = value;
                     return true;
                 }
             }
             return false;
    }
    return true;
}


cv::Ptr<cv::IVideoCapture> cv::create_DC1394_capture(int index)
{
    CvCaptureCAM_DC1394_v2_CPP* capture = new CvCaptureCAM_DC1394_v2_CPP;
    if (capture->open(index))
        return cv::makePtr<cv::LegacyCapture>(capture);
    delete capture;
    return 0;
}

#endif
