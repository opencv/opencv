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
#include "mil.h"

#if _MSC_VER >= 1200
  #pragma warning( disable: 4711 )
  #pragma comment(lib,"mil.lib")
  #pragma comment(lib,"milmet2.lib")
#endif

#if defined WIN64 && defined EM64T && defined _MSC_VER && !defined __ICL
  #pragma optimize("",off)
#endif

/********************* Capturing video from camera via MIL *********************/

struct
{
    MIL_ID MilApplication;
    int MilUser;
} g_Mil = {0,0}; //global structure for handling MIL application

class CvCaptureCAM_MIL : public CvCapture
{
public:
    CvCaptureCAM_MIL() { init(); }
    virtual ~CvCaptureCAM_MIL() { close(); }

    virtual bool open( int index );
    virtual void close();

    virtual double getProperty(int);
    virtual bool setProperty(int, double) { return false; }
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
	virtual int getCaptureDomain() { return CV_CAP_MIL; } // Return the type of the capture object: CV_CAP_VFW, etc...

protected:
    void init();

    MIL_ID
        MilSystem,       /* System identifier.       */
        MilDisplay,      /* Display identifier.      */
        MilDigitizer,    /* Digitizer identifier.    */
        MilImage;        /* Image buffer identifier. */
    IplImage* rgb_frame;
};


void CvCaptureCAM_MIL::init()
{
    MilSystem = MilDisplay = MilDigitizer = MilImage = M_NULL;
    rgb_frame = 0;
}

// Initialize camera input
bool CvCaptureCAM_MIL::open( int wIndex )
{
    close();

    if( g_Mil.MilApplication == M_NULL )
    {
        assert(g_Mil.MilUser == 0);
        MappAlloc(M_DEFAULT, &(g_Mil.MilApplication) );
        g_Mil.MilUser = 1;
    }
    else
    {
        assert(g_Mil.MilUser>0);
        g_Mil.MilUser++;
    }

    int dev_table[16] = { M_DEV0, M_DEV1, M_DEV2, M_DEV3,
        M_DEV4, M_DEV5, M_DEV6, M_DEV7,
        M_DEV8, M_DEV9, M_DEV10, M_DEV11,
        M_DEV12, M_DEV13, M_DEV14, M_DEV15 };

    //set default window size
    int w = 320;
    int h = 240;

    for( ; wIndex < 16; wIndex++ )
    {
        MsysAlloc( M_SYSTEM_SETUP, //we use default system,
                                   //if this does not work
                                   //try to define exact board
                                   //e.g.M_SYSTEM_METEOR,M_SYSTEM_METEOR_II...
                   dev_table[wIndex],
                   M_DEFAULT,
                   &MilSystem );

        if( MilSystem != M_NULL )
            break;
    }
    if( MilSystem != M_NULL )
    {
        MdigAlloc(MilSystem,M_DEFAULT,
                  M_CAMERA_SETUP, //default. May be M_NTSC or other
                  M_DEFAULT,&MilDigitizer);

        rgb_frame = cvCreateImage(cvSize(w,h), IPL_DEPTH_8U, 3 );
        MdigControl(MilDigitizer, M_GRAB_SCALE,  1.0 / 2);

        /*below line enables getting image vertical orientation
         consistent with VFW but it introduces some image corruption
         on MeteorII, so we left the image as is*/
        //MdigControl(MilDigitizer, M_GRAB_DIRECTION_Y, M_REVERSE );

        MilImage = MbufAllocColor(MilSystem, 3, w, h,
            8+M_UNSIGNED, M_IMAGE + M_GRAB, M_NULL);
    }

    return MilSystem != M_NULL;
}

void CvCaptureCAM_MIL::close( CvCaptureCAM_MIL* capture )
{
    if( MilSystem != M_NULL )
    {
        MdigFree( MilDigitizer );
        MbufFree( MilImage );
        MsysFree( MilSystem );
        cvReleaseImage(&rgb_frame );

        g_Mil.MilUser--;
        if(!g_Mil.MilUser)
            MappFree(g_Mil.MilApplication);

        MilSystem = M_NULL;
        MilDigitizer = M_NULL;
    }
}


bool CvCaptureCAM_MIL::grabFrame()
{
    if( MilSystem )
    {
        MdigGrab(MilDigitizer, MilImage);
        return true;
    }
    return false;
}


IplImage* CvCaptureCAM_MIL::retrieveFrame(int)
{
    MbufGetColor(MilImage, M_BGR24+M_PACKED, M_ALL_BAND, (void*)(rgb_frame->imageData));
    return rgb_frame;
}

double CvCaptureCAM_MIL::getProperty( int property_id )
{
    switch( property_id )
    {
	case CV_CAP_PROP_FRAME_WIDTH:
        return rgb_frame ? rgb_frame->width : 0;
	case CV_CAP_PROP_FRAME_HEIGHT:
		return rgb_frame ? rgb_frame->height : 0;
    }
    return 0;
}

bool CvCaptureCAM_MIL::setProperty( int, double )
{
    return false;
}


CvCapture* cvCreateCameraCapture_MIL( int index )
{
	CvCaptureCAM_MIL* capture = new CvCaptureCAM_MIL;

    if( capture->open( index ))
        return capture;

    delete capture;
    return 0;
}
