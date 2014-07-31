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
// Copyright (C) 2008, Xavier Delacour, all rights reserved.
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

// 2008-04-27 Xavier Delacour <xavier.delacour@gmail.com>

#include "precomp.hpp"
#include <unistd.h>
#include <unicap.h>
extern "C" {
#include <ucil.h>
}

#ifdef NDEBUG
#define CV_WARN(message)
#else
#define CV_WARN(message) fprintf(stderr, "warning: %s (%s:%d)\n", message, __FILE__, __LINE__)
#endif

struct CvCapture_Unicap : public CvCapture
{
  CvCapture_Unicap() { init(); }
  virtual ~CvCapture_Unicap() { close(); }

  virtual bool open( int index );
  virtual void close();

  virtual double getProperty(int);
  virtual bool setProperty(int, double);
  virtual bool grabFrame();
  virtual IplImage* retrieveFrame(int);
  virtual int getCaptureDomain() { return CV_CAP_UNICAP; } // Return the type of the capture object: CV_CAP_VFW, etc...

  bool shutdownDevice();
  bool initDevice();

  void init()
  {
    device_initialized = false;
    desired_format = 0;
    desired_size = cvSize(0,0);
    convert_rgb = false;

    handle = 0;
    memset( &device, 0, sizeof(device) );
    memset( &format_spec, 0, sizeof(format_spec) );
    memset( &format, 0, sizeof(format) );
    memset( &raw_buffer, 0, sizeof(raw_buffer) );
    memset( &buffer, 0, sizeof(buffer) );

    raw_frame = frame = 0;
  }

  bool device_initialized;

  int desired_device;
  int desired_format;
  CvSize desired_size;
  bool convert_rgb;

  unicap_handle_t handle;
  unicap_device_t device;
  unicap_format_t format_spec;
  unicap_format_t format;
  unicap_data_buffer_t raw_buffer;
  unicap_data_buffer_t buffer;

  IplImage *raw_frame;
  IplImage *frame;
};

bool CvCapture_Unicap::shutdownDevice() {
  bool result = false;
  CV_FUNCNAME("CvCapture_Unicap::shutdownDevice");
  __BEGIN__;

  if (!SUCCESS(unicap_stop_capture(handle)))
    CV_ERROR(CV_StsError, "unicap: failed to stop capture on device\n");

  if (!SUCCESS(unicap_close(handle)))
    CV_ERROR(CV_StsError, "unicap: failed to close the device\n");

  cvReleaseImage(&raw_frame);
  cvReleaseImage(&frame);

  device_initialized = false;

  result = true;
  __END__;
  return result;
}

bool CvCapture_Unicap::initDevice() {
  bool result = false;
  CV_FUNCNAME("CvCapture_Unicap::initDevice");
  __BEGIN__;

  if (device_initialized && !shutdownDevice())
    return false;

  if(!SUCCESS(unicap_enumerate_devices(NULL, &device, desired_device)))
    CV_ERROR(CV_StsError, "unicap: failed to get info for device\n");

  if(!SUCCESS(unicap_open( &handle, &device)))
    CV_ERROR(CV_StsError, "unicap: failed to open device\n");

  unicap_void_format(&format_spec);

  if (!SUCCESS(unicap_enumerate_formats(handle, &format_spec, &format, desired_format))) {
    shutdownDevice();
    CV_ERROR(CV_StsError, "unicap: failed to get video format\n");
  }

  int i;
  if (format.sizes)
  {
      for (i = format.size_count - 1; i > 0; i--)
        if (format.sizes[i].width == desired_size.width &&
        format.sizes[i].height == desired_size.height)
          break;
      format.size.width = format.sizes[i].width;
      format.size.height = format.sizes[i].height;
  }

  if (!SUCCESS(unicap_set_format(handle, &format))) {
    shutdownDevice();
    CV_ERROR(CV_StsError, "unicap: failed to set video format\n");
  }

  memset(&raw_buffer, 0x0, sizeof(unicap_data_buffer_t));
  raw_frame = cvCreateImage(cvSize(format.size.width,
                    format.size.height),
                  8, format.bpp / 8);
  memcpy(&raw_buffer.format, &format, sizeof(raw_buffer.format));
  raw_buffer.data = (unsigned char*)raw_frame->imageData;
  raw_buffer.buffer_size = format.size.width *
    format.size.height * format.bpp / 8;

  memset(&buffer, 0x0, sizeof(unicap_data_buffer_t));
  memcpy(&buffer.format, &format, sizeof(buffer.format));

  buffer.format.fourcc = UCIL_FOURCC('B','G','R','3');
  buffer.format.bpp = 24;
  // * todo support greyscale output
  //    buffer.format.fourcc = UCIL_FOURCC('G','R','E','Y');
  //    buffer.format.bpp = 8;

  frame = cvCreateImage(cvSize(buffer.format.size.width,
                    buffer.format.size.height),
                  8, buffer.format.bpp / 8);
  buffer.data = (unsigned char*)frame->imageData;
  buffer.buffer_size = buffer.format.size.width *
    buffer.format.size.height * buffer.format.bpp / 8;

  if(!SUCCESS(unicap_start_capture(handle))) {
    shutdownDevice();
    CV_ERROR(CV_StsError, "unicap: failed to start capture on device\n");
  }

  device_initialized = true;
  result = true;
  __END__;
  return result;
}

void CvCapture_Unicap::close() {
  if(device_initialized)
    shutdownDevice();
}

bool CvCapture_Unicap::grabFrame() {
  bool result = false;

  CV_FUNCNAME("CvCapture_Unicap::grabFrame");
  __BEGIN__;

  unicap_data_buffer_t *returned_buffer;

  int retry_count = 100;

  while (retry_count--) {
    if(!SUCCESS(unicap_queue_buffer(handle, &raw_buffer)))
      CV_ERROR(CV_StsError, "unicap: failed to queue a buffer on device\n");

    if(SUCCESS(unicap_wait_buffer(handle, &returned_buffer)))
    {
      result = true;
      EXIT;
    }

    CV_WARN("unicap: failed to wait for buffer on device\n");
    usleep(100 * 1000);
  }

  __END__;
  return result;
}

IplImage * CvCapture_Unicap::retrieveFrame(int) {
  if (convert_rgb) {
    ucil_convert_buffer(&buffer, &raw_buffer);
    return frame;
  }
  return raw_frame;
}

double CvCapture_Unicap::getProperty(int id) {
  switch (id) {
  case CV_CAP_PROP_POS_MSEC: break;
  case CV_CAP_PROP_POS_FRAMES: break;
  case CV_CAP_PROP_POS_AVI_RATIO: break;
  case CV_CAP_PROP_FRAME_WIDTH:
    return desired_size.width;
  case CV_CAP_PROP_FRAME_HEIGHT:
    return desired_size.height;
  case CV_CAP_PROP_FPS: break;
  case CV_CAP_PROP_FOURCC: break;
  case CV_CAP_PROP_FRAME_COUNT: break;
  case CV_CAP_PROP_FORMAT:
    return desired_format;
  case CV_CAP_PROP_MODE: break;
  case CV_CAP_PROP_BRIGHTNESS: break;
  case CV_CAP_PROP_CONTRAST: break;
  case CV_CAP_PROP_SATURATION: break;
  case CV_CAP_PROP_HUE: break;
  case CV_CAP_PROP_GAIN: break;
  case CV_CAP_PROP_CONVERT_RGB:
    return convert_rgb;
  }

  return 0;
}

bool CvCapture_Unicap::setProperty(int id, double value) {
  bool reinit = false;

  switch (id) {
  case CV_CAP_PROP_POS_MSEC: break;
  case CV_CAP_PROP_POS_FRAMES: break;
  case CV_CAP_PROP_POS_AVI_RATIO: break;
  case CV_CAP_PROP_FRAME_WIDTH:
    desired_size.width = (int)value;
    reinit = true;
    break;
  case CV_CAP_PROP_FRAME_HEIGHT:
    desired_size.height = (int)value;
    reinit = true;
    break;
  case CV_CAP_PROP_FPS: break;
  case CV_CAP_PROP_FOURCC: break;
  case CV_CAP_PROP_FRAME_COUNT: break;
  case CV_CAP_PROP_FORMAT:
    desired_format = id;
    reinit = true;
    break;
  case CV_CAP_PROP_MODE: break;
  case CV_CAP_PROP_BRIGHTNESS: break;
  case CV_CAP_PROP_CONTRAST: break;
  case CV_CAP_PROP_SATURATION: break;
  case CV_CAP_PROP_HUE: break;
  case CV_CAP_PROP_GAIN: break;
  case CV_CAP_PROP_CONVERT_RGB:
    convert_rgb = value != 0;
    break;
  }

  if (reinit && !initDevice())
    return false;

  return true;
}

bool CvCapture_Unicap::open(int index)
{
  close();
  device_initialized = false;

  desired_device = index < 0 ? 0 : index;
  desired_format = 0;
  desired_size = cvSize(320, 240);
  convert_rgb = true;

  return initDevice();
}


CvCapture * cvCreateCameraCapture_Unicap(const int index)
{
  CvCapture_Unicap *cap = new CvCapture_Unicap;
  if( cap->open(index) )
    return cap;
  delete cap;
  return 0;
}
