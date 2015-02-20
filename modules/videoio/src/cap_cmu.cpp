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

#ifdef WIN32

/****************** Capturing video from camera via CMU lib *******************/

#ifdef HAVE_CMU1394

// This firewire capability added by Philip Gruebele (pgruebele@cox.net).
// For this to work you need to install the CMU firewire DCAM drivers,
// located at http://www-2.cs.cmu.edu/~iwan/1394/.
#include "1394camera.h"

class CvCaptureCAM_CMU : public CvCapture
{
public:
    CvCaptureCAM_CMU()
    {
        index = -1;
        image = 0;
    }

    virtual ~CvCaptureCAM_CMU()
    {
        close();
    }

    virtual bool open(int cameraId);
    virtual void close();
    virtual double getProperty(int) const;
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);

protected:
    C1394Camera* camera();
    CvSize getSize();
    int getDepth();
    int getNChannels();

    bool setVideoSize(int, int);
    bool setMode(int mode);
    bool setFrameRate(int rate);
    bool setFormat(int format);

    int  fps;    // 0-5
    int  mode;   // 0-7
    int  format; // 0-2, 7 ?
    int  index;
    IplImage* image;
};

// CMU 1394 camera stuff.
// This firewire capability added by Philip Gruebele (pgruebele@cox.net)
// and modified by Roman Stanchak (rstanchak@yahoo.com).
// For this to work you need to install the CMU firewire DCAM drivers,
// located at http://www-2.cs.cmu.edu/~iwan/1394/.
#define CMU_MAX_CAMERAS     20
int             CMU_numCameras = 0;
int             CMU_numActiveCameras = 0;
bool            CMU_useCameraFlags[CMU_MAX_CAMERAS];
C1394Camera     *CMU_theCamera = 0;

// stupid defines for mode, format, FPS
#define CV_CAP_IEEE1394_FPS_1_875 0
#define CV_CAP_IEEE1394_FPS_3_75 1
#define CV_CAP_IEEE1394_FPS_7_5 2
#define CV_CAP_IEEE1394_FPS_15 3
#define CV_CAP_IEEE1394_FPS_30 4
#define CV_CAP_IEEE1394_FPS_60 5

// index by size, color
#define CV_CAP_IEEE1394_COLOR_MONO 0
#define CV_CAP_IEEE1394_COLOR_MONO16 1
#define CV_CAP_IEEE1394_COLOR_YUV444 2
#define CV_CAP_IEEE1394_COLOR_YUV422 3
#define CV_CAP_IEEE1394_COLOR_YUV411 4
#define CV_CAP_IEEE1394_COLOR_RGB 5

#define CV_CAP_IEEE1394_SIZE_160X120 0
#define CV_CAP_IEEE1394_SIZE_320X240 1
#define CV_CAP_IEEE1394_SIZE_640X480 2
#define CV_CAP_IEEE1394_SIZE_800X600 3
#define CV_CAP_IEEE1394_SIZE_1024X768 4
#define CV_CAP_IEEE1394_SIZE_1280X960 5
#define CV_CAP_IEEE1394_SIZE_1600X1200 6

// given color, size, output format
// 1 16  444 422 411 RGB
static char CV_CAP_IEEE1394_FORMAT[7][6] =
{
    {-1, -1,  0, -1, -1, -1}, // 160x120
    {-1, -1, -1,  0, -1, -1}, // 320x240
    { 0,  0, -1,  0,  0,  0}, // 640x480
    { 1,  1, -1,  1, -1,  1}, // 800x600
    { 1,  1, -1,  1, -1,  1}, // 1024x768
    { 2,  2, -1,  2, -1,  2}, // 1280x960
    { 2,  2, -1,  2, -1,  2}  // 1600x1200
};

// given color, size, output corresponding mode
static char CV_CAP_IEEE1394_MODE[7][6] =
{
    {-1, -1,  0, -1, -1, -1}, // 160x120
    {-1, -1, -1,  1, -1, -1}, // 320x240
    { 5,  6, -1,  3,  2,  4}, // 640x480
    { 2,  6, -1,  0, -1,  1}, // 800x600
    { 5,  7, -1,  3, -1,  4}, // 1024x768
    { 2,  6, -1,  0, -1,  1}, // 1280x960
    { 5,  7, -1,  3, -1,  4}  // 1600x1200
};

// given format, mode, return COLOR
static char CV_CAP_IEEE1394_COLOR[2][8] =
{
    {
    CV_CAP_IEEE1394_COLOR_YUV444,
    CV_CAP_IEEE1394_COLOR_YUV422,
    CV_CAP_IEEE1394_COLOR_YUV411,
    CV_CAP_IEEE1394_COLOR_YUV422,
    CV_CAP_IEEE1394_COLOR_RGB,
    CV_CAP_IEEE1394_COLOR_MONO,
    CV_CAP_IEEE1394_COLOR_MONO16
    },
    {
    CV_CAP_IEEE1394_COLOR_YUV422,
    CV_CAP_IEEE1394_COLOR_RGB,
    CV_CAP_IEEE1394_COLOR_MONO,
    CV_CAP_IEEE1394_COLOR_YUV422,
    CV_CAP_IEEE1394_COLOR_RGB,
    CV_CAP_IEEE1394_COLOR_MONO,
    CV_CAP_IEEE1394_COLOR_MONO16,
    CV_CAP_IEEE1394_COLOR_MONO16
    }
};

// convert frame rate to suitable enum
/*static int icvFrameRateToIndex_CMU(double framerate){
    if(framerate > 30)       return CV_CAP_IEEE1394_FPS_60;
    else if(framerate > 15)  return CV_CAP_IEEE1394_FPS_30;
    else if(framerate > 7.5) return CV_CAP_IEEE1394_FPS_15;
    else if(framerate > 3.75) return CV_CAP_IEEE1394_FPS_7_5;
    else if(framerate > 1.875) return CV_CAP_IEEE1394_FPS_3_75;
    return CV_CAP_IEEE1394_FPS_1_875;
}*/

#if _MSC_VER >= 1200
#pragma comment(lib,"1394camera.lib")
#endif

C1394Camera* CvCaptureCAM_CMU::camera()
{
    return CMU_theCamera && index >= 0 ? &CMU_theCamera[index] : 0;
}

// return the size of the image
CvSize CvCaptureCAM_CMU::getSize()
{
    C1394Camera* cmucam = camera();
    unsigned long width = 0, height = 0;
    cmucam->GetVideoFrameDimensions( &width, &height );
    return cvSize((int)width, (int)height);
}

// return the opencv depth flag corresponding to the camera format
int CvCaptureCAM_CMU::getDepth()
{
    C1394Camera* cmucam = camera();
    int format = cmucam->GetVideoFormat();
    int mode = cmucam->GetVideoMode();

    // TODO
    if( format==7 ) {
        assert(0);
        return 1;
    }
    // irrelvant to depth
    if( format > 1 )
        format = 1;

    if( CV_CAP_IEEE1394_COLOR[format][mode]==CV_CAP_IEEE1394_COLOR_MONO16 )
        return IPL_DEPTH_16S;

    return IPL_DEPTH_8U;
}

// return the number of channels for camera
int CvCaptureCAM_CMU::getNChannels()
{
    C1394Camera* cmucam = camera();
    int format = cmucam->GetVideoFormat();
    int mode = cmucam->GetVideoMode();

    if( format==7 ){
        assert(0);
        return 1;
    }

    // irrelvant to nchannels
    if( format > 1 )
        format = 1;

    switch(CV_CAP_IEEE1394_COLOR[format][mode]){
    case CV_CAP_IEEE1394_COLOR_RGB:
        return 3;
    case CV_CAP_IEEE1394_COLOR_MONO:
    case CV_CAP_IEEE1394_COLOR_MONO16:
        return 1;
    case CV_CAP_IEEE1394_COLOR_YUV422:
    case CV_CAP_IEEE1394_COLOR_YUV444:
    case CV_CAP_IEEE1394_COLOR_YUV411:
        return 3;
    default:
        ;
    }
    return -1;
}

bool CvCaptureCAM_CMU::open( int _index )
{
    close();

    // if first time, then allocate all available cameras
    if( CMU_numCameras == 0 )
    {
        CMU_numActiveCameras = 0;
        CMU_theCamera = new C1394Camera[CMU_MAX_CAMERAS];

        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // create all cameras
        try
        {
            // create camera0
            if( CMU_theCamera[0].CheckLink() != CAM_SUCCESS )
                throw 1;

            // we have one pin per camera
            CMU_numCameras = CMU_theCamera[0].GetNumberCameras();

            // allocate remaining cameras
            for(int i = 1; i < CMU_numCameras && i<CMU_MAX_CAMERAS; i++ )
            {
                CMU_useCameraFlags[i] = false;
                if (CMU_theCamera[i].CheckLink() != CAM_SUCCESS)
                    throw 1;
            }
        }
        catch (...)
        {
            // free any allocated cameras
            // ...
            CMU_numCameras = 0;
            return false;
        }
    }

    try
    {
        CvSize size;

        // pick first unused camera
        if(_index==-1){
            for(int i = 0; i < CMU_numCameras; i++ )
            {
                if( !CMU_useCameraFlags[i] ){
                    _index = i;
                    break;
                }
            }
        }

        // no empty camera found
        if (_index==-1)
            throw 1;

        if (CMU_theCamera[_index].SelectCamera(_index) != CAM_SUCCESS)
            throw 2;

        if (CMU_theCamera[_index].InitCamera() != CAM_SUCCESS)
            throw 3;

        // set initial format -- try to pick best frame rate first, then color, then size
        bool found_format = false;
        for (int rate=5; rate>=0 && !found_format; rate--)
        {
            for (int color=CV_CAP_IEEE1394_COLOR_RGB; color>=0 && !found_format; color--)
            {
                for (int size=CV_CAP_IEEE1394_SIZE_1600X1200; size>=0 && !found_format; size--)
                {
                    int format = CV_CAP_IEEE1394_FORMAT[size][color];
                    int mode = CV_CAP_IEEE1394_MODE[size][color];
                    if (format!=-1 && mode!=-1 &&
                        CMU_theCamera[_index].HasVideoFrameRate(format,mode,rate))
                    {
                        CMU_theCamera[_index].SetVideoFormat(format);
                        CMU_theCamera[_index].SetVideoMode(mode);
                        CMU_theCamera[_index].SetVideoFrameRate(rate);
                        found_format = (CMU_theCamera[_index].StartImageAcquisition() == CAM_SUCCESS);
                    }
                }
            }
        }

        // try format 7
        if(!found_format){
            CMU_theCamera[_index].SetVideoFormat(7);
            CMU_theCamera[_index].SetVideoMode(0);
            if(CMU_theCamera[_index].StartImageAcquisition() != CAM_SUCCESS){
                // no format found
                throw 9;
            }
        }

        index = _index;
        size = getSize();
        // allocate image frame
        image = cvCreateImage( size, 8, 3 );
        cvZero(image);

        // successfully activated camera
        CMU_numActiveCameras++;
        CMU_useCameraFlags[_index] = true;
    }
    catch ( int )
    {
        return false;
    }

    return true;
}

void CvCaptureCAM_CMU::close()
{
    C1394Camera* cmucam = camera();
    if( cmucam )
    {
        cvReleaseImage( &image );
        cmucam->StopImageAcquisition();
        CMU_useCameraFlags[index] = false;
        index = -1;

        if( --CMU_numActiveCameras == 0 )
        {
            delete[] CMU_theCamera;
            CMU_theCamera = 0;
            CMU_numCameras = 0;
        }
    }
}


bool CvCaptureCAM_CMU::grabFrame()
{
    C1394Camera* cmucam = camera();
    return cmucam ? cmucam->AcquireImage() == CAM_SUCCESS : false;
}

/*static void swapRedBlue(IplImage * im)
{
    uchar * ptr = (uchar *) im->imageData;
    uchar t;
    for(int i=0; i<im->height; i++){
        ptr = (uchar *) im->imageData+im->widthStep*i;
        for(int j=0; j<im->width; j++){
            t = ptr[0];
            ptr[0] = ptr[2];
            ptr[2] = t;
            ptr+=3;
        }
    }
}*/

IplImage* CvCaptureCAM_CMU::retrieveFrame(int)
{
    C1394Camera* cmucam = camera();
    if( !cmucam )
        return 0;
    cmucam->getRGB((uchar*)image->imageData, image->imageSize);
    cvConvertImage( image, image, CV_CVTIMG_SWAP_RB );
    return image;
}


double CvCaptureCAM_CMU::getProperty( int property_id ) const
{
    C1394Camera* cmucam = camera();
    if( !cmucam )
        return 0;
    switch( property_id )
    {
    case CV_CAP_PROP_FRAME_WIDTH:
        return image->width;
    case CV_CAP_PROP_FRAME_HEIGHT:
        return image->height;
    case CV_CAP_PROP_FPS:
        return cmucam->GetVideoFrameRate();
    case CV_CAP_PROP_MODE:
        return cmucam->GetVideoMode();
    case CV_CAP_PROP_FORMAT:
        return cmucam->GetVideoFormat();
    }
    return 0;
}

bool CvCaptureCAM_CMU::setVideoSize(int, int)
{
    return false;
}

bool CvCaptureCAM_CMU::setMode(int mode)
{
    int format;
    C1394Camera* cmucam = camera();
    if( !cmucam )
        return false;
    format = cmucam->GetVideoFormat();
    if( mode < 0 || mode > 7 || !cmucam->HasVideoMode(format, mode))
        return false;
    cmucam->StopImageAcquisition();
    cmucam->SetVideoMode(mode);
    cmucam->StartImageAcquisition();
    return true;
}

bool CvCaptureCAM_CMU::setFrameRate(int rate)
{
    int format, mode;
    C1394Camera* cmucam = camera();
    if( !cmucam )
        return false;
    mode = cmucam->GetVideoMode();
    format = cmucam->GetVideoFormat();
    if( rate < 0 || rate > 5 || !cmucam->HasVideoFrameRate(format, mode, rate) )
        return false;
    cmucam->StopImageAcquisition();
    cmucam->SetVideoFrameRate(rate);
    cmucam->StartImageAcquisition();
    return true;
}

bool CvCaptureCAM_CMU::setFormat(int format)
{
    C1394Camera* cmucam = camera();
    if( !cmucam )
        return false;
    if( format < 0 || format > 2 || !cmucam->HasVideoFormat(format) )
        return false;
    cmucam->StopImageAcquisition();
    cmucam->SetVideoFormat(format);
    cmucam->StartImageAcquisition();
    return true;
}

bool CvCaptureCAM_CMU::setProperty( int property_id, double value )
{
    bool retval = false;
    int ival = cvRound(value);
    C1394Camera* cmucam = camera();
    if( !cmucam )
        return false;

    switch (property_id) {
        case CV_CAP_PROP_FRAME_WIDTH:
        case CV_CAP_PROP_FRAME_HEIGHT:
            {
                int width, height;
                if (property_id == CV_CAP_PROP_FRAME_WIDTH)
                {
                    width = ival;
                    height = width*3/4;
                }
                else {
                    height = ival;
                    width = height*4/3;
                }
                retval = setVideoSize(width, height);
            }
            break;
        case CV_CAP_PROP_FPS:
            retval = setFrameRate(ival);
            break;
        case CV_CAP_PROP_MODE:
            retval = setMode(ival);
            break;
        case CV_CAP_PROP_FORMAT:
            retval = setFormat(ival);
            break;
    }

    // resize image if its not the right size anymore
    CvSize size = getSize();
    if( !image || image->width != size.width || image->height != size.height )
    {
        cvReleaseImage( &image );
        image = cvCreateImage( size, 8, 3 );
    }
    return retval;
}

CvCapture * cvCreateCameraCapture_CMU (int index)
{
    CvCaptureCAM_CMU* capture = new CvCaptureCAM_CMU;
    if( capture->open(index) )
        return capture;
    delete capture;
    return 0;
}

#endif // CMU
#endif // WIN32
