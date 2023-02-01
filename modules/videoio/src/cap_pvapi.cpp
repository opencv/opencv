////////////////////////////////////////////////////////////////////////////////////////
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
//

//
// The code has been contributed by Justin G. Eskesen on 2010 Jan
//

#include "precomp.hpp"

#ifdef HAVE_PVAPI
#if !defined _WIN32 && !defined _LINUX
#define _LINUX
#endif

#if defined(_x64) || defined (__x86_64) || defined (_M_X64)
#define _x64 1
#elif defined(_x86) || defined(__i386) || defined (_M_IX86)
#define _x86 1
#endif

#include <PvApi.h>
#ifdef _WIN32
#  include <io.h>
#else
#  include <time.h>
#  include <unistd.h>
#endif

//#include <arpa/inet.h>

#define MAX_CAMERAS 10

/********************* Capturing video from camera via PvAPI *********************/

class CvCaptureCAM_PvAPI : public CvCapture
{
public:
    CvCaptureCAM_PvAPI();
    virtual ~CvCaptureCAM_PvAPI()
    {
        close();
    }

    virtual bool open( int index );
    virtual void close();
    virtual double getProperty(int) const CV_OVERRIDE;
    virtual bool setProperty(int, double) CV_OVERRIDE;
    virtual bool grabFrame() CV_OVERRIDE;
    virtual IplImage* retrieveFrame(int) CV_OVERRIDE;
    virtual int getCaptureDomain() CV_OVERRIDE
    {
        return CV_CAP_PVAPI;
    }

protected:
#ifndef _WIN32
    virtual void Sleep(unsigned int time);
#endif

    void stopCapture();
    bool startCapture();
    bool resizeCaptureFrame (int frameWidth, int frameHeight);

    typedef struct
    {
        unsigned long   UID;
        tPvHandle       Handle;
        tPvFrame        Frame;
    } tCamera;

    IplImage *frame;
    tCamera  Camera;
    tPvErr   Errcode;
};


CvCaptureCAM_PvAPI::CvCaptureCAM_PvAPI()
{
    frame = NULL;
    memset(&this->Camera, 0, sizeof(this->Camera));
}

#ifndef _WIN32
void CvCaptureCAM_PvAPI::Sleep(unsigned int time)
{
    struct timespec t,r;

    t.tv_sec    = time / 1000;
    t.tv_nsec   = (time % 1000) * 1000000;

    while(nanosleep(&t,&r)==-1)
        t = r;
}
#endif

void CvCaptureCAM_PvAPI::close()
{
    // Stop the acquisition & free the camera
    stopCapture();
    PvCameraClose(Camera.Handle);
    PvUnInitialize();
}

// Initialize camera input
bool CvCaptureCAM_PvAPI::open( int index )
{
    tPvCameraInfo cameraList[MAX_CAMERAS];

    tPvCameraInfo  camInfo;
    tPvIpSettings ipSettings;


    if (PvInitialize()) {
    }
    //return false;

    Sleep(1000);

    //close();

    int numCameras=PvCameraList(cameraList, MAX_CAMERAS, NULL);

    if (numCameras <= 0 || index >= numCameras)
        return false;

    Camera.UID = cameraList[index].UniqueId;

    if (!PvCameraInfo(Camera.UID,&camInfo) && !PvCameraIpSettingsGet(Camera.UID,&ipSettings))
    {
        /*
        struct in_addr addr;
        addr.s_addr = ipSettings.CurrentIpAddress;
        printf("Current address:\t%s\n",inet_ntoa(addr));
        addr.s_addr = ipSettings.CurrentIpSubnet;
        printf("Current subnet:\t\t%s\n",inet_ntoa(addr));
        addr.s_addr = ipSettings.CurrentIpGateway;
        printf("Current gateway:\t%s\n",inet_ntoa(addr));
        */
    }
    else
    {
        fprintf(stderr,"ERROR: could not retrieve camera IP settings.\n");
        return false;
    }


    if (PvCameraOpen(Camera.UID, ePvAccessMaster, &(Camera.Handle))==ePvErrSuccess)
    {
        tPvUint32 frameWidth, frameHeight;
        unsigned long maxSize;

        PvAttrUint32Get(Camera.Handle, "Width", &frameWidth);
        PvAttrUint32Get(Camera.Handle, "Height", &frameHeight);

        // Determine the maximum packet size supported by the system (ethernet adapter)
        // and then configure the camera to use this value.  If the system's NIC only supports
        // an MTU of 1500 or lower, this will automatically configure an MTU of 1500.
        // 8228 is the optimal size described by the API in order to enable jumbo frames

        maxSize = 8228;
        //PvAttrUint32Get(Camera.Handle,"PacketSize",&maxSize);
        if (PvCaptureAdjustPacketSize(Camera.Handle,maxSize)!=ePvErrSuccess)
            return false;

        resizeCaptureFrame(frameWidth, frameHeight);

        return startCapture();

    }
    fprintf(stderr,"Error cannot open camera\n");
    return false;

}

bool CvCaptureCAM_PvAPI::grabFrame()
{
    //if(Camera.Frame.Status != ePvErrUnplugged && Camera.Frame.Status != ePvErrCancelled)
    return PvCaptureQueueFrame(Camera.Handle, &(Camera.Frame), NULL) == ePvErrSuccess;
}


IplImage* CvCaptureCAM_PvAPI::retrieveFrame(int)
{
    if (PvCaptureWaitForFrameDone(Camera.Handle, &(Camera.Frame), 1000) == ePvErrSuccess)
    {
        return frame;
    }
    else return NULL;
}

double CvCaptureCAM_PvAPI::getProperty( int property_id ) const
{
    tPvUint32 nTemp;

    switch ( property_id )
    {
    case CV_CAP_PROP_FRAME_WIDTH:
        PvAttrUint32Get(Camera.Handle, "Width", &nTemp);
        return (double)nTemp;
    case CV_CAP_PROP_FRAME_HEIGHT:
        PvAttrUint32Get(Camera.Handle, "Height", &nTemp);
        return (double)nTemp;
    case CV_CAP_PROP_EXPOSURE:
        PvAttrUint32Get(Camera.Handle,"ExposureValue",&nTemp);
        return (double)nTemp;
    case CV_CAP_PROP_FPS:
        tPvFloat32 nfTemp;
        PvAttrFloat32Get(Camera.Handle, "StatFrameRate", &nfTemp);
        return (double)nfTemp;
    case CV_CAP_PROP_PVAPI_MULTICASTIP:
        char mEnable[2];
        char mIp[11];
        PvAttrEnumGet(Camera.Handle,"MulticastEnable",mEnable,sizeof(mEnable),NULL);
        if (strcmp(mEnable, "Off") == 0)
        {
            return -1;
        }
        else
        {
            long int ip;
            int a,b,c,d;
            PvAttrStringGet(Camera.Handle, "MulticastIPAddress",mIp,sizeof(mIp),NULL);
            sscanf(mIp, "%d.%d.%d.%d", &a, &b, &c, &d); ip = ((a*256 + b)*256 + c)*256 + d;
            return (double)ip;
        }
    case CV_CAP_PROP_GAIN:
        PvAttrUint32Get(Camera.Handle, "GainValue", &nTemp);
        return (double)nTemp;
    case CV_CAP_PROP_PVAPI_FRAMESTARTTRIGGERMODE:
        char triggerMode[256];
        PvAttrEnumGet(Camera.Handle, "FrameStartTriggerMode", triggerMode, 256, NULL);
        if (strcmp(triggerMode, "Freerun")==0)
            return 0.0;
        else if (strcmp(triggerMode, "SyncIn1")==0)
            return 1.0;
        else if (strcmp(triggerMode, "SyncIn2")==0)
            return 2.0;
        else if (strcmp(triggerMode, "FixedRate")==0)
            return 3.0;
        else if (strcmp(triggerMode, "Software")==0)
            return 4.0;
        else
            return -1.0;
    case CV_CAP_PROP_PVAPI_DECIMATIONHORIZONTAL:
        PvAttrUint32Get(Camera.Handle, "DecimationHorizontal", &nTemp);
        return (double)nTemp;
    case CV_CAP_PROP_PVAPI_DECIMATIONVERTICAL:
        PvAttrUint32Get(Camera.Handle, "DecimationVertical", &nTemp);
        return (double)nTemp;
    case CV_CAP_PROP_PVAPI_BINNINGX:
        PvAttrUint32Get(Camera.Handle,"BinningX",&nTemp);
        return (double)nTemp;
    case CV_CAP_PROP_PVAPI_BINNINGY:
        PvAttrUint32Get(Camera.Handle,"BinningY",&nTemp);
        return (double)nTemp;
    case CV_CAP_PROP_PVAPI_PIXELFORMAT:
        char pixelFormat[256];
        PvAttrEnumGet(Camera.Handle, "PixelFormat", pixelFormat,256,NULL);
        if (strcmp(pixelFormat, "Mono8")==0)
            return 1.0;
        else if (strcmp(pixelFormat, "Mono16")==0)
            return 2.0;
        else if (strcmp(pixelFormat, "Bayer8")==0)
            return 3.0;
        else if (strcmp(pixelFormat, "Bayer16")==0)
            return 4.0;
        else if (strcmp(pixelFormat, "Rgb24")==0)
            return 5.0;
        else if (strcmp(pixelFormat, "Bgr24")==0)
            return 6.0;
        else if (strcmp(pixelFormat, "Rgba32")==0)
            return 7.0;
        else if (strcmp(pixelFormat, "Bgra32")==0)
            return 8.0;
    }
    return -1.0;
}

bool CvCaptureCAM_PvAPI::setProperty( int property_id, double value )
{
    tPvErr error;

    switch ( property_id )
    {
    case CV_CAP_PROP_FRAME_WIDTH:
    {
        tPvUint32 currHeight;

        PvAttrUint32Get(Camera.Handle, "Height", &currHeight);

        stopCapture();
        // Reallocate Frames
        if (!resizeCaptureFrame(value, currHeight))
        {
            startCapture();
            return false;
        }

        startCapture();

        break;
    }
    case CV_CAP_PROP_FRAME_HEIGHT:
    {
        tPvUint32 currWidth;

        PvAttrUint32Get(Camera.Handle, "Width", &currWidth);

        stopCapture();

        // Reallocate Frames
        if (!resizeCaptureFrame(currWidth, value))
        {
            startCapture();
            return false;
        }

        startCapture();

        break;
    }
    case CV_CAP_PROP_EXPOSURE:
        if ((PvAttrUint32Set(Camera.Handle,"ExposureValue",(tPvUint32)value)==ePvErrSuccess))
            break;
        else
            return false;
    case CV_CAP_PROP_PVAPI_MULTICASTIP:
        if (value==-1)
        {
            if ((PvAttrEnumSet(Camera.Handle,"MulticastEnable", "Off")==ePvErrSuccess))
                break;
            else
                return false;
        }
        else
        {
            cv::String ip=cv::format("%d.%d.%d.%d", ((unsigned int)value>>24)&255, ((unsigned int)value>>16)&255, ((unsigned int)value>>8)&255, (unsigned int)value&255);
            if ((PvAttrEnumSet(Camera.Handle,"MulticastEnable", "On")==ePvErrSuccess) &&
                (PvAttrStringSet(Camera.Handle, "MulticastIPAddress", ip.c_str())==ePvErrSuccess))
                break;
            else
                return false;
        }
    case CV_CAP_PROP_GAIN:
        if (PvAttrUint32Set(Camera.Handle,"GainValue",(tPvUint32)value)!=ePvErrSuccess)
        {
            return false;
        }
        break;
    case CV_CAP_PROP_PVAPI_FRAMESTARTTRIGGERMODE:
        if (value==0)
            error = PvAttrEnumSet(Camera.Handle, "FrameStartTriggerMode", "Freerun");
        else if (value==1)
            error = PvAttrEnumSet(Camera.Handle, "FrameStartTriggerMode", "SyncIn1");
        else if (value==2)
            error = PvAttrEnumSet(Camera.Handle, "FrameStartTriggerMode", "SyncIn2");
        else if (value==3)
            error = PvAttrEnumSet(Camera.Handle, "FrameStartTriggerMode", "FixedRate");
        else if (value==4)
            error = PvAttrEnumSet(Camera.Handle, "FrameStartTriggerMode", "Software");
        else
            error = ePvErrOutOfRange;
        if(error==ePvErrSuccess)
            break;
        else
            return false;
    case CV_CAP_PROP_PVAPI_DECIMATIONHORIZONTAL:
        if (value >= 1 && value <= 8)
            error = PvAttrUint32Set(Camera.Handle, "DecimationHorizontal", value);
        else
            error = ePvErrOutOfRange;
        if(error==ePvErrSuccess)
            break;
        else
            return false;
    case CV_CAP_PROP_PVAPI_DECIMATIONVERTICAL:
        if (value >= 1 && value <= 8)
            error = PvAttrUint32Set(Camera.Handle, "DecimationVertical", value);
        else
            error = ePvErrOutOfRange;
        if(error==ePvErrSuccess)
            break;
        else
            return false;
    case CV_CAP_PROP_PVAPI_BINNINGX:
        error = PvAttrUint32Set(Camera.Handle, "BinningX", value);
        if(error==ePvErrSuccess)
            break;
        else
            return false;
    case CV_CAP_PROP_PVAPI_BINNINGY:
        error = PvAttrUint32Set(Camera.Handle, "BinningY", value);
        if(error==ePvErrSuccess)
            break;
        else
            return false;
    case CV_CAP_PROP_PVAPI_PIXELFORMAT:
        {
            cv::String pixelFormat;

            if (value==1)
                pixelFormat = "Mono8";
            else if (value==2)
                pixelFormat = "Mono16";
            else if (value==3)
                pixelFormat = "Bayer8";
            else if (value==4)
                pixelFormat = "Bayer16";
            else if (value==5)
                pixelFormat = "Rgb24";
            else if (value==6)
                pixelFormat = "Bgr24";
            else if (value==7)
                pixelFormat = "Rgba32";
            else if (value==8)
                pixelFormat = "Bgra32";
            else
                return false;

            if ((PvAttrEnumSet(Camera.Handle,"PixelFormat", pixelFormat.c_str())==ePvErrSuccess))
            {
                tPvUint32 currWidth;
                tPvUint32 currHeight;

                PvAttrUint32Get(Camera.Handle, "Width", &currWidth);
                PvAttrUint32Get(Camera.Handle, "Height", &currHeight);

                stopCapture();
                // Reallocate Frames
                if (!resizeCaptureFrame(currWidth, currHeight))
                {
                    startCapture();
                    return false;
                }

                startCapture();
                return true;
            }
            else
                return false;
        }
    default:
        return false;
    }
    return true;
}

void CvCaptureCAM_PvAPI::stopCapture()
{
    PvCommandRun(Camera.Handle, "AcquisitionStop");
    PvCaptureEnd(Camera.Handle);
}

bool CvCaptureCAM_PvAPI::startCapture()
{
    // Start the camera
    PvCaptureStart(Camera.Handle);

    // Set the camera to capture continuously
    if(PvAttrEnumSet(Camera.Handle, "AcquisitionMode", "Continuous")!= ePvErrSuccess)
    {
        fprintf(stderr,"Could not set PvAPI Acquisition Mode\n");
        return false;
    }

    if(PvCommandRun(Camera.Handle, "AcquisitionStart")!= ePvErrSuccess)
    {
        fprintf(stderr,"Could not start PvAPI acquisition\n");
        return false;
    }

    if(PvAttrEnumSet(Camera.Handle, "FrameStartTriggerMode", "Freerun")!= ePvErrSuccess)
    {
        fprintf(stderr,"Error setting PvAPI trigger to \"Freerun\"");
        return false;
    }

    return true;
}

bool CvCaptureCAM_PvAPI::resizeCaptureFrame (int frameWidth, int frameHeight)
{
    char pixelFormat[256];
    tPvUint32 frameSize;
    tPvUint32 sensorHeight;
    tPvUint32 sensorWidth;

    if (frame)
    {
        cvReleaseImage(&frame);
        frame = NULL;
    }

    if (PvAttrUint32Get(Camera.Handle, "SensorWidth", &sensorWidth) != ePvErrSuccess)
    {
        return false;
    }

    if (PvAttrUint32Get(Camera.Handle, "SensorHeight", &sensorHeight) != ePvErrSuccess)
    {
        return false;
    }

    // Cap out of bounds widths to the max supported by the sensor
    if ((frameWidth < 0) || ((tPvUint32)frameWidth > sensorWidth))
    {
        frameWidth = sensorWidth;
    }

    if ((frameHeight < 0) || ((tPvUint32)frameHeight > sensorHeight))
    {
        frameHeight = sensorHeight;
    }


    if (PvAttrUint32Set(Camera.Handle, "Height", frameHeight) != ePvErrSuccess)
    {
        return false;
    }

    if (PvAttrUint32Set(Camera.Handle, "Width", frameWidth) != ePvErrSuccess)
    {
        return false;
    }

    PvAttrEnumGet(Camera.Handle, "PixelFormat", pixelFormat,256,NULL);
    PvAttrUint32Get(Camera.Handle, "TotalBytesPerFrame", &frameSize);


    if ( (strcmp(pixelFormat, "Mono8")==0) || (strcmp(pixelFormat, "Bayer8")==0) )
    {
        frame = cvCreateImage(cvSize((int)frameWidth, (int)frameHeight), IPL_DEPTH_8U, 1);
        frame->widthStep = (int)frameWidth;
        Camera.Frame.ImageBufferSize = frameSize;
        Camera.Frame.ImageBuffer = frame->imageData;
    }
    else if ( (strcmp(pixelFormat, "Mono16")==0) || (strcmp(pixelFormat, "Bayer16")==0) )
    {
        frame = cvCreateImage(cvSize((int)frameWidth, (int)frameHeight), IPL_DEPTH_16U, 1);
        frame->widthStep = (int)frameWidth*2;
        Camera.Frame.ImageBufferSize = frameSize;
        Camera.Frame.ImageBuffer = frame->imageData;
    }
    else if ( (strcmp(pixelFormat, "Rgb24")==0) || (strcmp(pixelFormat, "Bgr24")==0) )
    {
        frame = cvCreateImage(cvSize((int)frameWidth, (int)frameHeight), IPL_DEPTH_8U, 3);
        frame->widthStep = (int)frameWidth*3;
        Camera.Frame.ImageBufferSize = frameSize;
        Camera.Frame.ImageBuffer = frame->imageData;
    }
    else if ( (strcmp(pixelFormat, "Rgba32")==0) || (strcmp(pixelFormat, "Bgra32")==0) )
    {
        frame = cvCreateImage(cvSize((int)frameWidth, (int)frameHeight), IPL_DEPTH_8U, 4);
        frame->widthStep = (int)frameWidth*4;
        Camera.Frame.ImageBufferSize = frameSize;
        Camera.Frame.ImageBuffer = frame->imageData;
    }
    else
        return false;

    return true;
}

CvCapture* cvCreateCameraCapture_PvAPI( int index )
{
    CvCaptureCAM_PvAPI* capture = new CvCaptureCAM_PvAPI;

    if ( capture->open( index ))
        return capture;

    delete capture;
    return NULL;
}
#endif
