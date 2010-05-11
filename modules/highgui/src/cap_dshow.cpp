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

#ifdef HAVE_VIDEOINPUT
#include "videoinput.h"

/********************* Capturing video from camera via VFW *********************/

class CvCaptureCAM_DShow : public CvCapture
{
public:
    CvCaptureCAM_DShow();
    virtual ~CvCaptureCAM_DShow() { close(); }

    virtual bool open( int index );
    virtual void close();
    virtual double getProperty(int);
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
	virtual int getCaptureDomain() { return CV_CAP_DSHOW; } // Return the type of the capture object: CV_CAP_VFW, etc...

protected:
    void init();

    int index;
    IplImage* frame;
    static videoInput VI;
};


struct SuppressVideoInputMessages
{
    SuppressVideoInputMessages() { videoInput::setVerbose(false); }
};

static SuppressVideoInputMessages do_it;
videoInput CvCaptureCAM_DShow::VI;

CvCaptureCAM_DShow::CvCaptureCAM_DShow()
{
    index = -1;
    frame = 0;
}

void CvCaptureCAM_DShow::close()
{
    if( index >= 0 )
    {
        VI.stopDevice(index);
        index = -1;
        cvReleaseImage(&frame);
    }
}

// Initialize camera input
bool CvCaptureCAM_DShow::open( int _index )
{
	int try_index = _index;
	int devices = 0;

    close();
    devices = VI.listDevices(true);
    if (devices == 0)
    	return false;
    try_index = try_index < 0 ? 0 : (try_index > devices-1 ? devices-1 : try_index);
    VI.setupDevice(try_index);
    if( !VI.isDeviceSetup(try_index) )
        return false;
    index = try_index;
    return true;
}

bool CvCaptureCAM_DShow::grabFrame()
{
    return true;
}


IplImage* CvCaptureCAM_DShow::retrieveFrame(int)
{
    if( !frame || VI.getWidth(index) != frame->width || VI.getHeight(index) != frame->height )
    {
        if (frame)
            cvReleaseImage( &frame );
        int w = VI.getWidth(index), h = VI.getHeight(index);
        frame = cvCreateImage( cvSize(w,h), 8, 3 );
    }

    VI.getPixels( index, (uchar*)frame->imageData, false, true );
    return frame;
}

double CvCaptureCAM_DShow::getProperty( int property_id )
{
    switch( property_id )
    {
    case CV_CAP_PROP_FRAME_WIDTH:
        return VI.getWidth(index);
    case CV_CAP_PROP_FRAME_HEIGHT:
        return VI.getHeight(index);
    case CV_CAP_PROP_FOURCC:
        return 0;
    }
    return 0;
}

bool CvCaptureCAM_DShow::setProperty( int property_id, double value )
{
    int width = 0, height = 0;

    switch( property_id )
    {
    case CV_CAP_PROP_FRAME_WIDTH:
        width = cvRound(value);
        height = width*3/4;
        break;
    case CV_CAP_PROP_FRAME_HEIGHT:
        height = cvRound(value);
        width = height*4/3;
    default:
        return false;
    }

    if( width != VI.getWidth(index) || height != VI.getHeight(index) )
    {
        VI.stopDevice(index);
        VI.setupDevice(index, width, height);
    }
    return VI.isDeviceSetup(index);
}


CvCapture* cvCreateCameraCapture_DShow( int index )
{
    CvCaptureCAM_DShow* capture = new CvCaptureCAM_DShow;

    if( capture->open( index ))
        return capture;

    delete capture;
    return 0;
}

#ifdef _MSC_VER
#pragma comment(lib, "videoInput.lib")
#endif

#endif
