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
// The code has been contributed by Arkadiusz Raj on 2016 Oct
//

#include "precomp.hpp"

#ifdef HAVE_ARAVIS_API
#if !defined WIN32 && !defined _WIN32 && !defined _LINUX
#define _LINUX
#endif

#if defined(_x64) || defined (__x86_64) || defined (_M_X64)
#define _x64 1
#elif defined(_x86) || defined(__i386) || defined (_M_IX86)
#define _x86 1
#endif

#include <arv.h>
#ifdef WIN32
#  include <io.h>
#else
#  include <time.h>
#  include <unistd.h>
#endif

#define MAX_CAMERAS 10

/********************* Capturing video from camera via Aravis *********************/

class CvCaptureCAM_Aravis : public CvCapture
{
public:
    CvCaptureCAM_Aravis();
    virtual ~CvCaptureCAM_Aravis()
    {
        close();
    }

    virtual bool open(int);
    virtual void close();
    virtual double getProperty(int) const;
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
    virtual int getCaptureDomain()
    {
        return CV_CAP_ARAVIS;
    }

protected:
	bool create(int);
	bool init();
	
    void stopCapture();
    bool startCapture();
    
    bool getDeviceNameById(int id, std::string &device);

	ArvCamera       *camera;                // Camera to control.
	ArvStream       *stream;                // Object for video stream reception.
	void			*framebuffer;			// 
	unsigned int    payload;                // Width x height x Pixel width.
	
	int             widthMin;               // Camera sensor minium width.
	int             widthMax;               // Camera sensor maximum width.
	int             heightMin;              // Camera sensor minium height.
	int             heightMax;              // Camera sensor maximum height.
	double          fpsMin;     	        // Camera minium fps.
	double          fpsMax; 	            // Camera maximum fps.
	double          gainMin;                // Camera minimum gain.
	double          gainMax;                // Camera maximum gain.
	double          exposureMin;            // Camera's minimum exposure time.
	double          exposureMax;            // Camera's maximum exposure time.
	
	int 			num_buffers;			// number of payload transmission buffers
    
    IplImage *frame;
};


CvCaptureCAM_Aravis::CvCaptureCAM_Aravis()
{
    camera = NULL;
    stream = NULL;
    framebuffer = NULL;
    num_buffers = 50;
    
    frame = NULL;
}

void CvCaptureCAM_Aravis::close()
{
    stopCapture();
}

bool CvCaptureCAM_Aravis::getDeviceNameById(int id, std::string &device)
{
	arv_update_device_list();

	int n_devices = arv_get_n_devices();

	for(int i = 0; i< n_devices; i++){
		if(id == i){
			device = arv_get_device_id(i);
			return true;
		}
	}

	return false;
}

bool CvCaptureCAM_Aravis::create( int index )
{
	std::string deviceName;
	if(!getDeviceNameById(index, deviceName))
		return false;

	return NULL != (camera = arv_camera_new(deviceName.c_str()));
}

bool CvCaptureCAM_Aravis::init() 
{
	if( (stream = arv_camera_create_stream(camera, NULL, NULL)) ) {
		g_object_set(stream,
			"socket-buffer", ARV_GV_STREAM_SOCKET_BUFFER_AUTO,
			"socket-buffer-size", 0, NULL);
		g_object_set(stream,
			"packet-resend", ARV_GV_STREAM_PACKET_RESEND_NEVER, NULL);
		g_object_set(stream,
			"packet-timeout", (unsigned)40000,
			"frame-retention", (unsigned) 200000, NULL);

		for (int i = 0; i < num_buffers; i++)
			arv_stream_push_buffer(stream, arv_buffer_new(payload, NULL));

		return true;
	}

	return false;
}

// Initialize camera input
bool CvCaptureCAM_Aravis::open( int index )
{
	if( create( index) ) {
		// fetch basic properties
		payload = arv_camera_get_payload (camera);
		arv_camera_get_width_bounds(camera, &widthMin, &widthMax);
		arv_camera_get_height_bounds(camera, &heightMin, &heightMax);
		arv_camera_get_frame_rate_bounds(camera, &fpsMin, &fpsMax);
		arv_camera_get_gain_bounds (camera, &gainMin, &gainMax);
		arv_camera_get_exposure_time_bounds (camera, &exposureMin, &exposureMax);

		// enforce mono 8 format
		arv_camera_set_pixel_format(camera, ARV_PIXEL_FORMAT_MONO_8);
		
		// init communication
		init();
		
		return startCapture();
	}
	return false;
}

bool CvCaptureCAM_Aravis::grabFrame()
{
	ArvBuffer *arv_buffer = arv_stream_timeout_pop_buffer(stream, 2000000); //us
	if(arv_buffer != NULL) {
		if(arv_buffer_get_status(arv_buffer) == ARV_BUFFER_STATUS_SUCCESS) {
			size_t buffer_size;
			framebuffer = (void*)arv_buffer_get_data (arv_buffer, &buffer_size);
		}
		arv_stream_push_buffer(stream, arv_buffer);
		return true;
	}
	framebuffer = NULL;
	return false;
}

IplImage* CvCaptureCAM_Aravis::retrieveFrame(int)
{
	if(framebuffer) {
		IplImage src;
		cvInitImageHeader( &src, cvSize( widthMax, heightMax ),
						   IPL_DEPTH_8U, 1, IPL_ORIGIN_TL, 4 );

		cvSetData( &src, framebuffer, src.widthStep );
		if( !frame || frame->width != src.width || frame->height != src.height ) {
			cvReleaseImage( &frame );
			frame = cvCreateImage( cvGetSize(&src), 8, 1 );
		} 
		cvCopy(&src, frame);
		
		return frame;
	}
	return NULL;
}

double CvCaptureCAM_Aravis::getProperty( int property_id ) const
{
	switch ( property_id ) {
		case CV_CAP_PROP_EXPOSURE:
			return arv_camera_get_exposure_time(camera);
        
		case CV_CAP_PROP_FPS:
			return arv_camera_get_frame_rate(camera);

		case CV_CAP_PROP_GAIN:
			return arv_camera_get_gain(camera);
    }
    return -1.0;
}

bool CvCaptureCAM_Aravis::setProperty( int property_id, double value )
{
	switch ( property_id ) {
		case CV_CAP_PROP_EXPOSURE:
			arv_camera_set_exposure_time(camera, value);
			break;
        
		case CV_CAP_PROP_FPS:
			arv_camera_set_frame_rate(camera, value);
			break;

		case CV_CAP_PROP_GAIN:
			arv_camera_set_gain(camera, value);
			break;
    }
    return -1.0 != getProperty( property_id );
}

void CvCaptureCAM_Aravis::stopCapture()
{
	arv_camera_stop_acquisition(camera);
	
	g_object_unref(stream);
	stream = NULL;

	g_object_unref(camera);
}

bool CvCaptureCAM_Aravis::startCapture()
{
	arv_camera_set_acquisition_mode(camera, ARV_ACQUISITION_MODE_CONTINUOUS);
    arv_device_set_string_feature_value(arv_camera_get_device (camera), "TriggerMode" , "Off");
    arv_camera_start_acquisition(camera);
    
    return true;
}

CvCapture* cvCreateCameraCapture_Aravis( int index )
{
    CvCaptureCAM_Aravis* capture = new CvCaptureCAM_Aravis;

    if ( capture->open( index ))
        return capture;

    delete capture;
    return NULL;
}
#endif
