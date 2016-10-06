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

    virtual bool open( int index );
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
	bool init();
	
    void stopCapture();
    bool startCapture();
    
    bool getDeviceNameById(int id, std::string &device);

    typedef struct
    {
		ArvCamera       *camera;                // Camera to control.
		ArvStream       *stream;                // Object for video stream reception.
		void			*framebuffer;			// 
		unsigned int    payload;                // Width x height.
		
		int             mWidth;                 // Camera sensor's width.
		int             mHeight;                // Camera sensor's height.
		double          gainMin;                // Camera minimum gain.
		double          gainMax;                // Camera maximum gain.
		double          exposureMin;            // Camera's minimum exposure time.
		double          exposureMax;            // Camera's maximum exposure time.
    } tCamera;
    
    IplImage *frame;
	tCamera Device;
};


CvCaptureCAM_Aravis::CvCaptureCAM_Aravis()
{
    frame = NULL;
    memset(&this->Device, 0, sizeof(this->Device));
}

void CvCaptureCAM_Aravis::close()
{
    // Stop the acquisition & free the camera
    stopCapture();
}
#include <iostream>
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

bool CvCaptureCAM_Aravis::init() 
{
	// Create a new stream object. Open stream on Device.
	Device.stream = arv_camera_create_stream(Device.camera, NULL, NULL);

	if(Device.stream) {
		g_object_set(Device.stream,
			// ARV_GV_STREAM_SOCKET_BUFFER_FIXED : socket buffer is set to a given fixed value.
			// ARV_GV_STREAM_SOCKET_BUFFER_AUTO: socket buffer is set with respect to the payload size.
			"socket-buffer", ARV_GV_STREAM_SOCKET_BUFFER_AUTO,
			// Socket buffer size, in bytes.
			// Allowed values: >= G_MAXULONG
			// Default value: 0
			"socket-buffer-size", 0, NULL);

		// # packet-resend : Enables or disables the packet resend mechanism

		// If packet resend is disabled and a packet has been lost during transmission,
		// the grab result for the returned buffer holding the image will indicate that
		// the grab failed and the image will be incomplete.
		//
		// If packet resend is enabled and a packet has been lost during transmission,
		// a request is sent to the Device. If the camera still has the packet in its
		// buffer, it will resend the packet. If there are several lost packets in a
		// row, the resend requests will be combined.

		g_object_set(Device.stream,
			// ARV_GV_STREAM_PACKET_RESEND_NEVER: never request a packet resend
			// ARV_GV_STREAM_PACKET_RESEND_ALWAYS: request a packet resend if a packet was missing
			// Default value: ARV_GV_STREAM_PACKET_RESEND_ALWAYS
			"packet-resend", ARV_GV_STREAM_PACKET_RESEND_NEVER, NULL);

		// # packet-timeout

		// The Packet Timeout parameter defines how long (in milliseconds) we will wait for
		// the next expected packet before it initiates a resend request.

		// Packet timeout, in µs.
		// Allowed values: [1000,10000000]
		// Default value: 40000
		
		// The Frame Retention parameter sets the timeout (in milliseconds) for the
		// frame retention timer. Whenever detection of the leader is made for a frame,
		// the frame retention timer starts. The timer resets after each packet in the
		// frame is received and will timeout after the last packet is received. If the
		// timer times out at any time before the last packet is received, the buffer for
		// the frame will be released and will be indicated as an unsuccessful grab.

		// Packet retention, in µs.
		// Allowed values: [1000,10000000]
		// Default value: 200000			
		g_object_set(Device.stream,
			"packet-timeout",/* (unsigned) arv_option_packet_timeout * 1000*/(unsigned)40000,
			"frame-retention", /*(unsigned) arv_option_frame_retention * 1000*/(unsigned) 200000,NULL);

		// Push 50 buffer in the stream input buffer queue.
		for (int i = 0; i < 50; i++)
			arv_stream_push_buffer(Device.stream, arv_buffer_new(Device.payload, NULL));

		return true;
	}

	return false;
}

// Initialize camera input
bool CvCaptureCAM_Aravis::open( int index )
{
	std::string deviceName;

	if(!getDeviceNameById(index, deviceName))
		return false;

std::cout << deviceName << std::endl;
	Device.camera = arv_camera_new(deviceName.c_str());
	if(Device.camera) {
std::cout << "camera defined" << std::endl;		
		// init communication
		init();
std::cout << "camera initialized" << std::endl;				
		// fetch basic properties
		Device.payload = arv_camera_get_payload (Device.camera);
std::cout << Device.payload << std::endl;						
		arv_camera_get_sensor_size(Device.camera, &Device.mWidth, &Device.mHeight);
std::cout << Device.mWidth << 'x' << Device.mHeight << std::endl;						

		arv_camera_get_exposure_time_bounds (Device.camera, &Device.exposureMin, &Device.exposureMax);
std::cout << Device.exposureMin << " - " << Device.exposureMax << std::endl;								
		arv_camera_get_gain_bounds (Device.camera, &Device.gainMin, &Device.gainMax);
std::cout << Device.gainMin << " - " << Device.gainMax << std::endl;			

		arv_camera_set_pixel_format(Device.camera, ARV_PIXEL_FORMAT_MONO_8);
	}
	return false;
}

bool CvCaptureCAM_Aravis::grabFrame()
{
	ArvBuffer *arv_buffer = arv_stream_timeout_pop_buffer(Device.stream, 2000000); //us
	if(arv_buffer != NULL) {
		if(arv_buffer_get_status(arv_buffer) == ARV_BUFFER_STATUS_SUCCESS) {
			size_t buffer_size;
			Device.framebuffer = (IplImage*) arv_buffer_get_data (arv_buffer, &buffer_size);
			 
			return true;
		}
	}
	
	Device.framebuffer = NULL;
	return false;
}

IplImage* CvCaptureCAM_Aravis::retrieveFrame(int)
{
	if(Device.framebuffer) {
		IplImage src;
		cvInitImageHeader( &src, cvSize( Device.mWidth, Device.mHeight ),
						   IPL_DEPTH_8U, 1, IPL_ORIGIN_TL, 4 );
		cvSetData( &src, Device.framebuffer, src.widthStep );
		if( !frame || frame->width != src.width || frame->height != src.height ) {
			cvReleaseImage( &frame );
			frame = cvCreateImage( cvGetSize(&src), 8, 1 );
			
			return frame;
		}
	}
	return NULL;
}

double CvCaptureCAM_Aravis::getProperty( int property_id ) const
{
	switch ( property_id ) {
		case CV_CAP_PROP_EXPOSURE:
			return arv_camera_get_exposure_time(Device.camera);
        
		case CV_CAP_PROP_FPS:
			return arv_camera_get_frame_rate(Device.camera);

		case CV_CAP_PROP_GAIN:
			return arv_camera_get_gain(Device.camera);
			
		case CV_CAP_PROP_PVAPI_PIXELFORMAT:
			int pixFormat = arv_camera_get_pixel_format(Device.camera);
			if (pixFormat == ARV_PIXEL_FORMAT_MONO_8)
				return 1.0;
			else if (pixFormat == ARV_PIXEL_FORMAT_MONO_12)
				return 2.0;
    }
    return -1.0;
}

bool CvCaptureCAM_Aravis::setProperty( int property_id, double value )
{
	switch ( property_id ) {
		case CV_CAP_PROP_EXPOSURE:
			arv_camera_set_exposure_time(Device.camera, value);
			break;
        
		case CV_CAP_PROP_FPS:
			arv_camera_set_frame_rate(Device.camera, value);
			break;

		case CV_CAP_PROP_GAIN:
			arv_camera_set_gain(Device.camera, value);
			break;
			
		case CV_CAP_PROP_PVAPI_PIXELFORMAT:
			if (value==1) arv_camera_set_pixel_format(Device.camera, ARV_PIXEL_FORMAT_MONO_8);
			else if (value==2) arv_camera_set_pixel_format(Device.camera, ARV_PIXEL_FORMAT_MONO_12);
			break;
    }
    return -1.0 != getProperty( property_id );
}

void CvCaptureCAM_Aravis::stopCapture()
{
	arv_camera_stop_acquisition(Device.camera);
	
	g_object_unref(Device.stream);
	Device.stream = NULL;

	g_object_unref(Device.camera);
	
std::cout << "camera stopped" << std::endl;			
}

bool CvCaptureCAM_Aravis::startCapture()
{
	arv_camera_set_acquisition_mode(Device.camera, ARV_ACQUISITION_MODE_CONTINUOUS);
    arv_device_set_string_feature_value(arv_camera_get_device (Device.camera), "TriggerMode" , "Off");
    arv_camera_start_acquisition(Device.camera);
std::cout << "camera started" << std::endl;		
    
    
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
