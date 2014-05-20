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
#include <DeepSeaIF.h>

#if _MSC_VER >= 1200
    #pragma comment(lib,"DeepSeaIF.lib")
#endif


/****************** Capturing video from TYZX stereo camera  *******************/
/** Initially developed by Roman Stanchak rstanchak@yahoo.com                  */

class CvCaptureCAM_TYZX : public CvCapture
{
public:
    CvCaptureCAM_TYZX() { index = -1; image = 0; }
    virtual ~CvCaptureCAM_TYZX() { close(); }

    virtual bool open( int _index );
    virtual void close();
    bool isOpened() { return index >= 0; }

    virtual double getProperty(int);
    virtual bool setProperty(int, double) { return false; }
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
    virtual int getCaptureDomain() { return CV_CAP_TYZX; } // Return the type of the capture object: CV_CAP_VFW, etc...

protected:
    virtual bool allocateImage();

    int index;
    IplImage* image;
}
CvCaptureCAM_TYZX;

DeepSeaIF * g_tyzx_camera   = 0;
int         g_tyzx_refcount = 0;

bool CvCaptureCAM_TYZX::open( int _index )
{
    close();

    if(!g_tyzx_camera){
        g_tyzx_camera = new DeepSeaIF;
        if(!g_tyzx_camera) return false;

        if(!g_tyzx_camera->initializeSettings(NULL)){
            delete g_tyzx_camera;
            return false;
        }

        // set initial sensor mode
        // TODO is g_tyzx_camera redundant?
        g_tyzx_camera->setSensorMode(g_tyzx_camera->getSensorMode());

        // mm's
        g_tyzx_camera->setZUnits((int) 1000);

        g_tyzx_camera->enableLeftColor(true);
        g_tyzx_camera->setColorMode(DeepSeaIF::BGRcolor);
        g_tyzx_camera->setDoIntensityCrop(true);
        g_tyzx_camera->enable8bitImages(true);
        if(!g_tyzx_camera->startCapture()){
            return false;
        }
        g_tyzx_refcount++;
    }
    index = _index;
    return true;
}

void CvCaptureCAM_TYZX::close()
{
    if( isOpened() )
    {
        cvReleaseImage( &image );
        g_tyzx_refcount--;
        if(g_tyzx_refcount==0){
            delete g_tyzx_camera;
        }
    }
}

bool CvCaptureCAM_TYZX::grabFrame()
{
    return isOpened() && g_tyzx_camera && g_tyzx_camera->grab();
}

bool CvCaptureCAM_TYZX::allocateImage()
{
    int depth, nch;
    CvSize size;

    // assume we want to resize
    cvReleaseImage(&image);

    // figure out size depending on index provided
    switch(index){
        case CV_TYZX_RIGHT:
            size = cvSize(g_tyzx_camera->intensityWidth(), g_tyzx_camera->intensityHeight());
            depth = 8;
            nch = 1;
            break;
        case CV_TYZX_Z:
            size = cvSize(g_tyzx_camera->zWidth(), g_tyzx_camera->zHeight());
            depth = IPL_DEPTH_16S;
            nch = 1;
            break;
        case CV_TYZX_LEFT:
        default:
            size = cvSize(g_tyzx_camera->intensityWidth(), g_tyzx_camera->intensityHeight());
            depth = 8;
            nch = 1;
            break;
    }
    image = cvCreateImage(size, depth, nch);
    return image != 0;
}

/// Copy 'grabbed' image into capture buffer and return it.
IplImage * CvCaptureCAM_TYZX::retrieveFrame(int)
{
    if(!isOpened() || !g_tyzx_camera) return 0;

    if(!image && !allocateImage())
        return 0;

    // copy camera image into buffer.
    // tempting to reference TYZX memory directly to avoid copying.
    switch (index)
    {
        case CV_TYZX_RIGHT:
            memcpy(image->imageData, g_tyzx_camera->getRImage(), image->imageSize);
            break;
        case CV_TYZX_Z:
            memcpy(image->imageData, g_tyzx_camera->getZImage(), image->imageSize);
            break;
        case CV_TYZX_LEFT:
        default:
            memcpy(image->imageData, g_tyzx_camera->getLImage(), image->imageSize);
            break;
    }

    return image;
}

double CvCaptureCAM_TYZX::getProperty(int property_id)
{
    CvSize size;
    switch(capture->index)
    {
        case CV_TYZX_LEFT:
            size = cvSize(g_tyzx_camera->intensityWidth(), g_tyzx_camera->intensityHeight());
            break;
        case CV_TYZX_RIGHT:
            size = cvSize(g_tyzx_camera->intensityWidth(), g_tyzx_camera->intensityHeight());
            break;
        case CV_TYZX_Z:
            size = cvSize(g_tyzx_camera->zWidth(), g_tyzx_camera->zHeight());
            break;
        default:
            size = cvSize(0,0);
    }

    switch( property_id )
    {
        case CV_CAP_PROP_FRAME_WIDTH:
            return size.width;
        case CV_CAP_PROP_FRAME_HEIGHT:
            return size.height;
    }

    return 0;
}

bool CvCaptureCAM_TYZX::setProperty( int, double )
{
    return false;
}

CvCapture * cvCreateCameraCapture_TYZX (int index)
{
    CvCaptureCAM_TYZX * capture = new CvCaptureCAM_TYZX;
    if( capture->open(index) )
        return capture;

    delete capture;
    return 0;
}
