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
#if !defined WIN32 && !defined _WIN32 && !defined _LINUX
#define _LINUX
#endif

#if defined(_x64) || defined (__x86_64) || defined (_M_X64)
#define _x64 1
#elif defined(_x86) || defined(__i386) || defined (_M_IX86)
#define _x86 1
#endif

#include <PvApi.h>
#ifdef WIN32
#  include <io.h>
#else
#  include <unistd.h>
#endif

#include <string>
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
    virtual double getProperty(int);
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
    virtual int getCaptureDomain()
    {
        return CV_CAP_PVAPI;
    }

protected:
#ifndef WIN32
    virtual void Sleep(unsigned int time);
#endif

    typedef struct
    {
        unsigned long   UID;
        tPvHandle       Handle;
        tPvFrame        Frame;
    } tCamera;

    IplImage *frame;
    IplImage *grayframe;
    tCamera  Camera;
    tPvErr   Errcode;
    bool monocrome;
};


CvCaptureCAM_PvAPI::CvCaptureCAM_PvAPI()
{
    monocrome=false;
    memset(&this->Camera, 0, sizeof(this->Camera));
}

#ifndef WIN32
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
    PvCommandRun(Camera.Handle, "AcquisitionStop");
    PvCaptureEnd(Camera.Handle);
    PvCameraClose(Camera.Handle);
    PvUnInitialize();
}

// Initialize camera input
bool CvCaptureCAM_PvAPI::open( int )
{
    tPvCameraInfo cameraList[MAX_CAMERAS];
    tPvCameraInfo  camInfo;
    tPvIpSettings ipSettings;

    // Initialization parameters [500 x 10 ms = 5000 ms timeout]
    int initializeTimeOut = 500;

    // Disregard any errors, since this might be called several times and only needs to be called once or it will return an
    // Important when wanting to use more than 1 AVT camera at the same time
    PvInitialize();

    while((!PvCameraCount())  && (initializeTimeOut--))
        Sleep(10);

    if (!initializeTimeOut){
        fprintf(stderr,"ERROR: camera intialisation timeout [5000ms].\n");
        return false;
    }

    unsigned int numCameras = PvCameraList(cameraList, MAX_CAMERAS, NULL);

    // If no cameras are found
    if(!numCameras)
    {
        fprintf(stderr, "ERROR: No cameras found.\n");
        return false;
    }

    // Try opening the cameras in the list, one-by-one until a camera that is not used is found
    unsigned int findNewCamera;
    for(findNewCamera=0; findNewCamera<numCameras; findNewCamera++)
    {
        Camera.UID = cameraList[findNewCamera].UniqueId;
        if(PvCameraOpen(Camera.UID, ePvAccessMaster, &(Camera.Handle))==ePvErrSuccess)
            break;
    }

    if(findNewCamera == numCameras)
    {
        fprintf(stderr, "Could not find a new camera to connect to.\n");
        return false;
    }

    if(PvCameraIpSettingsGet(Camera.UID,&ipSettings)==ePvErrNotFound)
    {
        fprintf(stderr, "The specified camera UID %lu could not be found, PvCameraIpSettingsGet().\n", Camera.UID);
        return false;
    }

    if(PvCameraInfo(Camera.UID,&camInfo)==ePvErrNotFound)
    {
        fprintf(stderr, "The specified camera UID %lu could not be found, PvCameraInfo().\n", Camera.UID);
        return false;
    }

    tPvUint32 frameWidth, frameHeight, frameSize;
    char pixelFormat[256];
    PvAttrUint32Get(Camera.Handle, "TotalBytesPerFrame", &frameSize);
    PvAttrUint32Get(Camera.Handle, "Width", &frameWidth);
    PvAttrUint32Get(Camera.Handle, "Height", &frameHeight);
    PvAttrEnumGet(Camera.Handle, "PixelFormat", pixelFormat,256,NULL);

    // Start the camera
    PvCaptureStart(Camera.Handle);

    // Set the camera explicitly to capture data frames continuously
    if(PvAttrEnumSet(Camera.Handle, "AcquisitionMode", "Continuous")!= ePvErrSuccess)
    {
        fprintf(stderr,"Could not set Acquisition Mode\n");
        return false;
    }

    if(PvCommandRun(Camera.Handle, "AcquisitionStart")!= ePvErrSuccess)
    {
        fprintf(stderr,"Could not start acquisition\n");
        return false;
    }

    if(PvAttrEnumSet(Camera.Handle, "FrameStartTriggerMode", "Freerun")!= ePvErrSuccess)
    {
        fprintf(stderr,"Error setting trigger to \"Freerun\"");
        return false;
    }

    // Settings depending on the pixelformat
    // This works for all AVT camera models that use the PvAPI interface
    if (strcmp(pixelFormat, "Mono8")==0) {
        monocrome = true;
        grayframe = cvCreateImage(cvSize((int)frameWidth, (int)frameHeight), IPL_DEPTH_8U, 1);
        grayframe->widthStep = (int)frameWidth;
        Camera.Frame.ImageBufferSize = frameSize;
        Camera.Frame.ImageBuffer = grayframe->imageData;
    }
    else if (strcmp(pixelFormat, "Mono16")==0) {
        monocrome = true;
        grayframe = cvCreateImage(cvSize((int)frameWidth, (int)frameHeight), IPL_DEPTH_16U, 1);
        grayframe->widthStep = (int)frameWidth*2;
        Camera.Frame.ImageBufferSize = frameSize;
        Camera.Frame.ImageBuffer = grayframe->imageData;
    }
    else if (strcmp(pixelFormat, "Bgr24")==0) {
        monocrome = false;
        frame = cvCreateImage(cvSize((int)frameWidth, (int)frameHeight), IPL_DEPTH_8U, 3);
        frame->widthStep = (int)frameWidth*3;
        Camera.Frame.ImageBufferSize = frameSize;
        Camera.Frame.ImageBuffer = frame->imageData;
    }
    else{
        fprintf(stderr, "Pixel format %s not supported; only Mono8, Mono16 and Bgr24 are currently supported.\n", pixelFormat);
        return false;
    }

    return true;
}

bool CvCaptureCAM_PvAPI::grabFrame()
{
    return PvCaptureQueueFrame(Camera.Handle, &(Camera.Frame), NULL) == ePvErrSuccess;
}


IplImage* CvCaptureCAM_PvAPI::retrieveFrame(int)
{

    if (PvCaptureWaitForFrameDone(Camera.Handle, &(Camera.Frame), 1000) == ePvErrSuccess) {
        if (!monocrome) {
            return frame;
        }
        return grayframe;
    }
    else return NULL;
}

double CvCaptureCAM_PvAPI::getProperty( int property_id )
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
        if (strcmp(mEnable, "Off") == 0) {
            return -1;
        }
        else {
            long int ip;
            int a,b,c,d;
            PvAttrStringGet(Camera.Handle, "MulticastIPAddress",mIp,sizeof(mIp),NULL);
            sscanf(mIp, "%d.%d.%d.%d", &a, &b, &c, &d); ip = ((a*256 + b)*256 + c)*256 + d;
            return (double)ip;
        }
    }
    return -1.0;
}

bool CvCaptureCAM_PvAPI::setProperty( int property_id, double value )
{
    switch ( property_id )
    {
    /*  TODO: Camera works, but IplImage must be modified for the new size
    case CV_CAP_PROP_FRAME_WIDTH:
        PvAttrUint32Set(Camera.Handle, "Width", (tPvUint32)value);
        break;
    case CV_CAP_PROP_FRAME_HEIGHT:
        PvAttrUint32Set(Camera.Handle, "Heigth", (tPvUint32)value);
        break;
    */
    case CV_CAP_PROP_MONOCROME:
        if (value==1) {
            char pixelFormat[256];
            PvAttrEnumGet(Camera.Handle, "PixelFormat", pixelFormat,256,NULL);
            if ((strcmp(pixelFormat, "Mono8")==0) || strcmp(pixelFormat, "Mono16")==0) {
                monocrome=true;
            }
            else
                return false;
        }
        else
            monocrome=false;
        break;
    case CV_CAP_PROP_EXPOSURE:
        if ((PvAttrUint32Set(Camera.Handle,"ExposureValue",(tPvUint32)value)==ePvErrSuccess))
        break;
        else
        return false;
    case CV_CAP_PROP_PVAPI_MULTICASTIP:

            if (value==-1) {
        if ((PvAttrEnumSet(Camera.Handle,"MulticastEnable", "Off")==ePvErrSuccess))
            break;
        else
            return false;
        }
        else {
        std::string ip=cv::format("%d.%d.%d.%d", ((int)value>>24)&255, ((int)value>>16)&255, ((int)value>>8)&255, (int)value&255);
        if ((PvAttrEnumSet(Camera.Handle,"MulticastEnable", "On")==ePvErrSuccess) &&
        (PvAttrStringSet(Camera.Handle, "MulticastIPAddress", ip.c_str())==ePvErrSuccess))
            break;
        else
            return false;
        }
    default:
        return false;
    }
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
