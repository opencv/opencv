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

#if defined(_x64) || defined (__x86_64) || defined (_WIN64)
#define _x64 1
#elif defined(_x86) || defined(__i386) || defined (_WIN32)
#define _x86 1
#endif

#include <PvApi.h>
#include <unistd.h>
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
	virtual void Sleep(unsigned int time);

    typedef struct
    {
        unsigned long   UID;
        tPvHandle       Handle;
        tPvFrame        Frame;


    } tCamera;
    IplImage *frame;
    IplImage *grayframe;
    tCamera Camera;
    tPvErr          Errcode;
    bool monocrome;


};


CvCaptureCAM_PvAPI::CvCaptureCAM_PvAPI()
{

}
void CvCaptureCAM_PvAPI::Sleep(unsigned int time)
{
    struct timespec t,r;
    
    t.tv_sec    = time / 1000;
    t.tv_nsec   = (time % 1000) * 1000000;    
    
    while(nanosleep(&t,&r)==-1)
        t = r;
}
void CvCaptureCAM_PvAPI::close()
{
	// Stop the acquisition & free the camera
	PvCommandRun(Camera.Handle, "AcquisitionStop");
	PvCaptureEnd(Camera.Handle);
	PvCameraClose(Camera.Handle);	
}

// Initialize camera input
bool CvCaptureCAM_PvAPI::open( int index )
{
    tPvCameraInfo cameraList[MAX_CAMERAS];
    
    tPvCameraInfo  camInfo;
    tPvIpSettings ipSettings;

 
    if (PvInitialize())
        return false;
        
    Sleep(1000);

    //close();
    
    int numCameras=PvCameraList(cameraList, MAX_CAMERAS, NULL);

    if (numCameras <= 0 || index >= numCameras)
        return false;

    Camera.UID = cameraList[index].UniqueId;    

    if (!PvCameraInfo(Camera.UID,&camInfo) && !PvCameraIpSettingsGet(Camera.UID,&ipSettings)) {
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
	else {
		fprintf(stderr,"ERROR: could not retrieve camera IP settings.\n");
		return false;
	}	


    if (PvCameraOpen(Camera.UID, ePvAccessMaster, &(Camera.Handle))==ePvErrSuccess)
    {
    
        //Set Pixel Format to BRG24 to follow conventions 
        /*Errcode = PvAttrEnumSet(Camera.Handle, "PixelFormat", "Bgr24");
        if (Errcode != ePvErrSuccess)
        {
            fprintf(stderr, "PvAPI: couldn't set PixelFormat to Bgr24\n");
            return NULL;
        }
        */
        tPvUint32 frameWidth, frameHeight, frameSize, maxSize;
		char pixelFormat[256];
        PvAttrUint32Get(Camera.Handle, "TotalBytesPerFrame", &frameSize);
        PvAttrUint32Get(Camera.Handle, "Width", &frameWidth);
        PvAttrUint32Get(Camera.Handle, "Height", &frameHeight);
        //PvAttrEnumGet(Camera.Handle, "pixelFormat", pixelFormat,256,NULL);
        maxSize = 8228;
        PvAttrUint32Get(Camera.Handle,"PacketSize",&maxSize);
        PvCaptureAdjustPacketSize(Camera.Handle,maxSize);
        //printf ("Pixel Format %s  %d %d\n ", pixelFormat,frameWidth,frameHeight);
        if (strncmp(pixelFormat, "Mono8",NULL)==0) {
				grayframe = cvCreateImage(cvSize((int)frameWidth, (int)frameHeight), IPL_DEPTH_8U, 1);
			    grayframe->widthStep = (int)frameWidth;
			    frame = cvCreateImage(cvSize((int)frameWidth, (int)frameHeight), IPL_DEPTH_8U, 3);
				frame->widthStep = (int)frameWidth*3;		 
				Camera.Frame.ImageBufferSize = frameSize;
				Camera.Frame.ImageBuffer = grayframe->imageData;   
		}	    
		else if (strncmp(pixelFormat, "Mono16",NULL)==0) {
				grayframe = cvCreateImage(cvSize((int)frameWidth, (int)frameHeight), IPL_DEPTH_16U, 1);
			    grayframe->widthStep = (int)frameWidth;	
			    frame = cvCreateImage(cvSize((int)frameWidth, (int)frameHeight), IPL_DEPTH_16U, 3);
				frame->widthStep = (int)frameWidth*3;
				Camera.Frame.ImageBufferSize = frameSize;
				Camera.Frame.ImageBuffer = grayframe->imageData;
		}	  
		else if	(strncmp(pixelFormat, "Bgr24",NULL)==0) {
				frame = cvCreateImage(cvSize((int)frameWidth, (int)frameHeight), IPL_DEPTH_8U, 3);
				frame->widthStep = (int)frameWidth*3;
				Camera.Frame.ImageBufferSize = frameSize;
				Camera.Frame.ImageBuffer = frame->imageData;
		}		
		else
				return false;
        // Start the camera
        PvCaptureStart(Camera.Handle);

        // Set the camera to capture continuously
        if(PvAttrEnumSet(Camera.Handle, "AcquisitionMode", "Continuous")!= ePvErrSuccess)
        {
            fprintf(stderr,"Could not set Prosilica Acquisition Mode\n");
            return false;
        }
        
        if(PvCommandRun(Camera.Handle, "AcquisitionStart")!= ePvErrSuccess)
        {
            fprintf(stderr,"Could not start Prosilica acquisition\n");
            return false;
        }
        
        if(PvAttrEnumSet(Camera.Handle, "FrameStartTriggerMode", "Freerun")!= ePvErrSuccess)
        {
            fprintf(stderr,"Error setting Prosilica trigger to \"Freerun\"");
            return false;
        }
        
        return true;
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

    if (PvCaptureWaitForFrameDone(Camera.Handle, &(Camera.Frame), 1000) == ePvErrSuccess) {
		if (!monocrome)
			cvMerge(grayframe,grayframe,grayframe,NULL,frame); 
		return frame;
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
		char pixelFormat[256];
        PvAttrEnumGet(Camera.Handle, "PixelFormat", pixelFormat,256,NULL);
        if ((strncmp(pixelFormat, "Mono8",NULL)==0) || strncmp(pixelFormat, "Mono16",NULL)==0) {
			monocrome=true;
			break;
		}	
		else
			return false;
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

#ifdef _MSC_VER
#pragma comment(lib, "PvAPI.lib")
#endif

#endif
