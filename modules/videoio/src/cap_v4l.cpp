/* This is the contributed code:

File:             cvcap_v4l.cpp
Current Location: ../opencv-0.9.6/otherlibs/videoio

Original Version: 2003-03-12  Magnus Lundin lundin@mlu.mine.nu
Original Comments:

ML:This set of files adds support for firevre and usb cameras.
First it tries to install a firewire camera,
if that fails it tries a v4l/USB camera
It has been tested with the motempl sample program

First Patch:  August 24, 2004 Travis Wood   TravisOCV@tkwood.com
For Release:  OpenCV-Linux Beta4  opencv-0.9.6
Tested On:    LMLBT44 with 8 video inputs
Problems?     Post your questions at answers.opencv.org,
              Report bugs at code.opencv.org,
              Submit your fixes at https://github.com/opencv/opencv/
Patched Comments:

TW: The cv cam utils that came with the initial release of OpenCV for LINUX Beta4
were not working.  I have rewritten them so they work for me. At the same time, trying
to keep the original code as ML wrote it as unchanged as possible.  No one likes to debug
someone elses code, so I resisted changes as much as possible.  I have tried to keep the
same "ideas" where applicable, that is, where I could figure out what the previous author
intended. Some areas I just could not help myself and had to "spiffy-it-up" my way.

These drivers should work with other V4L frame capture cards other then my bttv
driven frame capture card.

Re Written driver for standard V4L mode. Tested using LMLBT44 video capture card.
Standard bttv drivers are on the LMLBT44 with up to 8 Inputs.

This utility was written with the help of the document:
http://pages.cpsc.ucalgary.ca/~sayles/VFL_HowTo
as a general guide for interfacing into the V4l standard.

Made the index value passed for icvOpenCAM_V4L(index) be the number of the
video device source in the /dev tree. The -1 uses original /dev/video.

Index  Device
  0    /dev/video0
  1    /dev/video1
  2    /dev/video2
  3    /dev/video3
  ...
  7    /dev/video7
with
  -1   /dev/video

TW: You can select any video source, but this package was limited from the start to only
ONE camera opened at any ONE time.
This is an original program limitation.
If you are interested, I will make my version available to other OpenCV users.  The big
difference in mine is you may pass the camera number as part of the cv argument, but this
convention is non standard for current OpenCV calls and the camera number is not currently
passed into the called routine.

Second Patch:   August 28, 2004 Sfuncia Fabio fiblan@yahoo.it
For Release:  OpenCV-Linux Beta4 Opencv-0.9.6

FS: this patch fix not sequential index of device (unplugged device), and real numCameras.
    for -1 index (icvOpenCAM_V4L) i dont use /dev/video but real device available, because
    if /dev/video is a link to /dev/video0 and i unplugged device on /dev/video0, /dev/video
    is a bad link. I search the first available device with indexList.

Third Patch:   December 9, 2004 Frederic Devernay Frederic.Devernay@inria.fr
For Release:  OpenCV-Linux Beta4 Opencv-0.9.6

[FD] I modified the following:
 - handle YUV420P, YUV420, and YUV411P palettes (for many webcams) without using floating-point
 - cvGrabFrame should not wait for the end of the first frame, and should return quickly
   (see videoio doc)
 - cvRetrieveFrame should in turn wait for the end of frame capture, and should not
   trigger the capture of the next frame (the user choses when to do it using GrabFrame)
   To get the old behavior, re-call cvRetrieveFrame just after cvGrabFrame.
 - having global bufferIndex and FirstCapture variables makes the code non-reentrant
 (e.g. when using several cameras), put these in the CvCapture struct.
 - according to V4L HowTo, incrementing the buffer index must be done before VIDIOCMCAPTURE.
 - the VID_TYPE_SCALES stuff from V4L HowTo is wrong: image size can be changed
   even if the hardware does not support scaling (e.g. webcams can have several
   resolutions available). Just don't try to set the size at 640x480 if the hardware supports
   scaling: open with the default (probably best) image size, and let the user scale it
   using SetProperty.
 - image size can be changed by two subsequent calls to SetProperty (for width and height)
 - bug fix: if the image size changes, realloc the new image only when it is grabbed
 - issue errors only when necessary, fix error message formatting.

Fourth Patch: Sept 7, 2005 Csaba Kertesz sign@freemail.hu
For Release:  OpenCV-Linux Beta5 OpenCV-0.9.7

I modified the following:
  - Additional Video4Linux2 support :)
  - Use mmap functions (v4l2)
  - New methods are internal:
    try_palette_v4l2 -> rewrite try_palette for v4l2
    mainloop_v4l2, read_image_v4l2 -> this methods are moved from official v4l2 capture.c example
    try_init_v4l -> device v4l initialisation
    try_init_v4l2 -> device v4l2 initialisation
    autosetup_capture_mode_v4l -> autodetect capture modes for v4l
    autosetup_capture_mode_v4l2 -> autodetect capture modes for v4l2
  - Modifications are according with Video4Linux old codes
  - Video4Linux handling is automatically if it does not recognize a Video4Linux2 device
  - Tested successfully with Logitech Quickcam Express (V4L), Creative Vista (V4L) and Genius VideoCam Notebook (V4L2)
  - Correct source lines with compiler warning messages
  - Information message from v4l/v4l2 detection

Fifth Patch: Sept 7, 2005 Csaba Kertesz sign@freemail.hu
For Release:  OpenCV-Linux Beta5 OpenCV-0.9.7

I modified the following:
  - SN9C10x chip based webcams support
  - New methods are internal:
    bayer2rgb24, sonix_decompress -> decoder routines for SN9C10x decoding from Takafumi Mizuno <taka-qce@ls-a.jp> with his pleasure :)
  - Tested successfully with Genius VideoCam Notebook (V4L2)

Sixth Patch: Sept 10, 2005 Csaba Kertesz sign@freemail.hu
For Release:  OpenCV-Linux Beta5 OpenCV-0.9.7

I added the following:
  - Add capture control support (hue, saturation, brightness, contrast, gain)
  - Get and change V4L capture controls (hue, saturation, brightness, contrast)
  - New method is internal:
    icvSetControl -> set capture controls
  - Tested successfully with Creative Vista (V4L)

Seventh Patch: Sept 10, 2005 Csaba Kertesz sign@freemail.hu
For Release:  OpenCV-Linux Beta5 OpenCV-0.9.7

I added the following:
  - Detect, get and change V4L2 capture controls (hue, saturation, brightness, contrast, gain)
  - New methods are internal:
    v4l2_scan_controls_enumerate_menu, v4l2_scan_controls -> detect capture control intervals
  - Tested successfully with Genius VideoCam Notebook (V4L2)

8th patch: Jan 5, 2006, Olivier.Bornet@idiap.ch
Add support of V4L2_PIX_FMT_YUYV and V4L2_PIX_FMT_MJPEG.
With this patch, new webcams of Logitech, like QuickCam Fusion works.
Note: For use these webcams, look at the UVC driver at
http://linux-uvc.berlios.de/

9th patch: Mar 4, 2006, Olivier.Bornet@idiap.ch
- try V4L2 before V4L, because some devices are V4L2 by default,
  but they try to implement the V4L compatibility layer.
  So, I think this is better to support V4L2 before V4L.
- better separation between V4L2 and V4L initialization. (this was needed to support
  some drivers working, but not fully with V4L2. (so, we do not know when we
  need to switch from V4L2 to V4L.

10th patch: July 02, 2008, Mikhail Afanasyev fopencv@theamk.com
Fix reliability problems with high-resolution UVC cameras on linux
the symptoms were damaged image and 'Corrupt JPEG data: premature end of data segment' on stderr
- V4L_ABORT_BADJPEG detects JPEG warnings and turns them into errors, so bad images
  could be filtered out
- USE_TEMP_BUFFER fixes the main problem (improper buffer management) and
  prevents bad images in the first place

11th patch: April 2, 2013, Forrest Reiling forrest.reiling@gmail.com
Added v4l2 support for getting capture property CV_CAP_PROP_POS_MSEC.
Returns the millisecond timestamp of the last frame grabbed or 0 if no frames have been grabbed
Used to successfully synchonize 2 Logitech C310 USB webcams to within 16 ms of one another


make & enjoy!

*/

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

#if !defined WIN32 && (defined HAVE_CAMV4L2 || defined HAVE_VIDEOIO)

#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/mman.h>

#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/ioctl.h>

#ifdef HAVE_CAMV4L2
#include <asm/types.h>          /* for videodev2.h */
#include <linux/videodev2.h>
#endif

#ifdef HAVE_VIDEOIO
// NetBSD compability layer with V4L2
#include <sys/videoio.h>
#endif

/* Defaults - If your board can do better, set it here.  Set for the most common type inputs. */
#define DEFAULT_V4L_WIDTH  640
#define DEFAULT_V4L_HEIGHT 480
#define DEFAULT_V4L_FPS 30

#define CHANNEL_NUMBER 1
#define MAX_CAMERAS 8


// default and maximum number of V4L buffers, not including last, 'special' buffer
#define MAX_V4L_BUFFERS 10
#define DEFAULT_V4L_BUFFERS 4

// if enabled, then bad JPEG warnings become errors and cause NULL returned instead of image
#define V4L_ABORT_BADJPEG

#define MAX_DEVICE_DRIVER_NAME 80

namespace cv {

/* Device Capture Objects */
/* V4L2 structure */
struct buffer
{
  void *  start;
  size_t  length;
};

static unsigned int n_buffers = 0;

struct CvCaptureCAM_V4L : public CvCapture
{
    int deviceHandle;
    int bufferIndex;
    int FirstCapture;
    String deviceName;

    char *memoryMap;
    IplImage frame;

   __u32 palette;
   int width, height;
   __u32 fps;
   bool convert_rgb;
   bool frame_allocated;

   /* V4L2 variables */
   buffer buffers[MAX_V4L_BUFFERS + 1];
   v4l2_capability cap;
   v4l2_input inp;
   v4l2_format form;
   v4l2_crop crop;
   v4l2_cropcap cropcap;
   v4l2_requestbuffers req;
   v4l2_buf_type type;
   v4l2_queryctrl queryctrl;

   timeval timestamp;

   /* V4L2 control variables */
   Range focus, brightness, contrast, saturation, hue, gain, exposure;

   bool open(int _index);
   bool open(const char* deviceName);

   virtual double getProperty(int) const;
   virtual bool setProperty(int, double);
   virtual bool grabFrame();
   virtual IplImage* retrieveFrame(int);

   Range getRange(int property_id) const {
       switch (property_id) {
       case CV_CAP_PROP_BRIGHTNESS:
           return brightness;
       case CV_CAP_PROP_CONTRAST:
           return contrast;
       case CV_CAP_PROP_SATURATION:
           return saturation;
       case CV_CAP_PROP_HUE:
           return hue;
       case CV_CAP_PROP_GAIN:
           return gain;
       case CV_CAP_PROP_EXPOSURE:
           return exposure;
       case CV_CAP_PROP_FOCUS:
           return focus;
       case CV_CAP_PROP_AUTOFOCUS:
           return Range(0, 1);
       case CV_CAP_PROP_AUTO_EXPOSURE:
           return Range(0, 4);
       default:
           return Range(0, 255);
       }
   }

   virtual ~CvCaptureCAM_V4L();
};

static void icvCloseCAM_V4L( CvCaptureCAM_V4L* capture );

static bool icvGrabFrameCAM_V4L( CvCaptureCAM_V4L* capture );
static IplImage* icvRetrieveFrameCAM_V4L( CvCaptureCAM_V4L* capture, int );

static double icvGetPropertyCAM_V4L( const CvCaptureCAM_V4L* capture, int property_id );
static int    icvSetPropertyCAM_V4L( CvCaptureCAM_V4L* capture, int property_id, double value );

/***********************   Implementations  ***************************************/

static int numCameras = 0;
static int indexList = 0;

CvCaptureCAM_V4L::~CvCaptureCAM_V4L() {
    icvCloseCAM_V4L(this);
}

/* Simple test program: Find number of Video Sources available.
   Start from 0 and go to MAX_CAMERAS while checking for the device with that name.
   If it fails on the first attempt of /dev/video0, then check if /dev/video is valid.
   Returns the global numCameras with the correct value (we hope) */

static void icvInitCapture_V4L() {
   int deviceHandle;
   int CameraNumber;
   char deviceName[MAX_DEVICE_DRIVER_NAME];

   CameraNumber = 0;
   while(CameraNumber < MAX_CAMERAS) {
      /* Print the CameraNumber at the end of the string with a width of one character */
      sprintf(deviceName, "/dev/video%1d", CameraNumber);
      /* Test using an open to see if this new device name really does exists. */
      deviceHandle = open(deviceName, O_RDONLY);
      if (deviceHandle != -1) {
         /* This device does indeed exist - add it to the total so far */
    // add indexList
    indexList|=(1 << CameraNumber);
        numCameras++;
    }
    if (deviceHandle != -1)
      close(deviceHandle);
    /* Set up to test the next /dev/video source in line */
    CameraNumber++;
   } /* End while */

}; /* End icvInitCapture_V4L */

static bool try_palette_v4l2(CvCaptureCAM_V4L* capture)
{
  capture->form = v4l2_format();
  capture->form.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  capture->form.fmt.pix.pixelformat = capture->palette;
  capture->form.fmt.pix.field       = V4L2_FIELD_ANY;
  capture->form.fmt.pix.width       = capture->width;
  capture->form.fmt.pix.height      = capture->height;

  if (-1 == ioctl (capture->deviceHandle, VIDIOC_S_FMT, &capture->form))
      return false;

  return capture->palette == capture->form.fmt.pix.pixelformat;
}

static int try_init_v4l2(CvCaptureCAM_V4L* capture, const char *deviceName)
{
  // Test device for V4L2 compability
  // Return value:
  // -1 then unable to open device
  //  0 then detected nothing
  //  1 then V4L2 device

  int deviceIndex;

  /* Open and test V4L2 device */
  capture->deviceHandle = open (deviceName, O_RDWR /* required */ | O_NONBLOCK, 0);
  if (-1 == capture->deviceHandle)
  {
#ifndef NDEBUG
    fprintf(stderr, "(DEBUG) try_init_v4l2 open \"%s\": %s\n", deviceName, strerror(errno));
#endif
    icvCloseCAM_V4L(capture);
    return -1;
  }

  capture->cap = v4l2_capability();
  if (-1 == ioctl (capture->deviceHandle, VIDIOC_QUERYCAP, &capture->cap))
  {
#ifndef NDEBUG
    fprintf(stderr, "(DEBUG) try_init_v4l2 VIDIOC_QUERYCAP \"%s\": %s\n", deviceName, strerror(errno));
#endif
    icvCloseCAM_V4L(capture);
    return 0;
  }

  /* Query channels number */
  if (-1 == ioctl (capture->deviceHandle, VIDIOC_G_INPUT, &deviceIndex))
  {
#ifndef NDEBUG
    fprintf(stderr, "(DEBUG) try_init_v4l2 VIDIOC_G_INPUT \"%s\": %s\n", deviceName, strerror(errno));
#endif
    icvCloseCAM_V4L(capture);
    return 0;
  }

  /* Query information about current input */
  capture->inp = v4l2_input();
  capture->inp.index = deviceIndex;
  if (-1 == ioctl (capture->deviceHandle, VIDIOC_ENUMINPUT, &capture->inp))
  {
#ifndef NDEBUG
    fprintf(stderr, "(DEBUG) try_init_v4l2 VIDIOC_ENUMINPUT \"%s\": %s\n", deviceName, strerror(errno));
#endif
    icvCloseCAM_V4L(capture);
    return 0;
  }

  return 1;

}

static int autosetup_capture_mode_v4l2(CvCaptureCAM_V4L* capture) {
    //in case palette is already set and works, no need to setup.
    if(capture->palette != 0 and try_palette_v4l2(capture)){
        return 0;
    }
    __u32 try_order[] = {
            V4L2_PIX_FMT_BGR24,
            V4L2_PIX_FMT_YVU420,
            V4L2_PIX_FMT_YUV411P,
#ifdef HAVE_JPEG
            V4L2_PIX_FMT_MJPEG,
            V4L2_PIX_FMT_JPEG,
#endif
            V4L2_PIX_FMT_YUYV,
            V4L2_PIX_FMT_UYVY,
            V4L2_PIX_FMT_SN9C10X,
            V4L2_PIX_FMT_SBGGR8,
            V4L2_PIX_FMT_SGBRG8,
            V4L2_PIX_FMT_RGB24,
            V4L2_PIX_FMT_Y16
    };

    for (size_t i = 0; i < sizeof(try_order) / sizeof(__u32); i++) {
        capture->palette = try_order[i];
        if (try_palette_v4l2(capture)) {
            return 0;
        }
    }

    fprintf(stderr,
            "VIDEOIO ERROR: V4L2: Pixel format of incoming image is unsupported by OpenCV\n");
    icvCloseCAM_V4L(capture);
    return -1;
}

static void v4l2_control_range(CvCaptureCAM_V4L* cap, __u32 id)
{
    cap->queryctrl= v4l2_queryctrl();
    cap->queryctrl.id = id;

    if(0 != ioctl(cap->deviceHandle, VIDIOC_QUERYCTRL, &cap->queryctrl))
    {
        if (errno != EINVAL)
            perror ("VIDIOC_QUERYCTRL");
        return;
    }

    if (cap->queryctrl.flags & V4L2_CTRL_FLAG_DISABLED)
        return;

    Range range(cap->queryctrl.minimum, cap->queryctrl.maximum);

    switch(cap->queryctrl.id) {
    case V4L2_CID_BRIGHTNESS:
        cap->brightness = range;
        break;
    case V4L2_CID_CONTRAST:
        cap->contrast = range;
        break;
    case V4L2_CID_SATURATION:
        cap->saturation = range;
        break;
    case V4L2_CID_HUE:
        cap->hue = range;
        break;
    case V4L2_CID_GAIN:
        cap->gain = range;
        break;
    case V4L2_CID_EXPOSURE_ABSOLUTE:
        cap->exposure = range;
        break;
    case V4L2_CID_FOCUS_ABSOLUTE:
        cap->focus = range;
        break;
    }
}

static void v4l2_scan_controls(CvCaptureCAM_V4L* capture)
{

  __u32 ctrl_id;

  for (ctrl_id = V4L2_CID_BASE; ctrl_id < V4L2_CID_LASTP1; ctrl_id++)
  {
      v4l2_control_range(capture, ctrl_id);
  }

  for (ctrl_id = V4L2_CID_PRIVATE_BASE;;ctrl_id++)
  {
      v4l2_control_range(capture, ctrl_id);

      if (errno == EINVAL)
        break;
  }

  v4l2_control_range(capture, V4L2_CID_FOCUS_ABSOLUTE);
}

static int v4l2_set_fps(CvCaptureCAM_V4L* capture) {
    v4l2_streamparm setfps = v4l2_streamparm();
    setfps.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    setfps.parm.capture.timeperframe.numerator = 1;
    setfps.parm.capture.timeperframe.denominator = capture->fps;
    return ioctl (capture->deviceHandle, VIDIOC_S_PARM, &setfps);
}

static int v4l2_num_channels(__u32 palette) {
    switch(palette) {
    case V4L2_PIX_FMT_YVU420:
    case V4L2_PIX_FMT_MJPEG:
    case V4L2_PIX_FMT_JPEG:
    case V4L2_PIX_FMT_Y16:
        return 1;
    case V4L2_PIX_FMT_YUYV:
    case V4L2_PIX_FMT_UYVY:
        return 2;
    case V4L2_PIX_FMT_BGR24:
    case V4L2_PIX_FMT_RGB24:
        return 3;
    default:
        return 0;
    }
}

static void v4l2_create_frame(CvCaptureCAM_V4L *capture) {
    CvSize size(capture->form.fmt.pix.width, capture->form.fmt.pix.height);
    int channels = 3;
    int depth = IPL_DEPTH_8U;

    if (!capture->convert_rgb) {
        channels = v4l2_num_channels(capture->palette);

        switch(capture->palette) {
        case V4L2_PIX_FMT_MJPEG:
        case V4L2_PIX_FMT_JPEG:
            size = CvSize(capture->buffers[capture->bufferIndex].length, 1);
            break;
        case V4L2_PIX_FMT_YVU420:
            size.height = size.height * 3 / 2; // "1.5" channels
            break;
        case V4L2_PIX_FMT_Y16:
            if(!capture->convert_rgb){
                depth = IPL_DEPTH_16U;
            }
            break;
        }
    }

    /* Set up Image data */
    cvInitImageHeader(&capture->frame, size, depth, channels);

    /* Allocate space for pixelformat we convert to.
     * If we do not convert frame is just points to the buffer
     */
    if(capture->convert_rgb) {
        capture->frame.imageData = (char*)cvAlloc(capture->frame.imageSize);
    }

    capture->frame_allocated = capture->convert_rgb;
}

static int _capture_V4L2 (CvCaptureCAM_V4L *capture)
{
   const char* deviceName = capture->deviceName.c_str();
   if (try_init_v4l2(capture, deviceName) != 1) {
       /* init of the v4l2 device is not OK */
       return -1;
   }

   /* V4L2 control variables are zero (memset above) */

   /* Scan V4L2 controls */
   v4l2_scan_controls(capture);

   if ((capture->cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) == 0) {
      /* Nope. */
      fprintf( stderr, "VIDEOIO ERROR: V4L2: device %s is unable to capture video memory.\n",deviceName);
      icvCloseCAM_V4L(capture);
      return -1;
   }

   /* The following code sets the CHANNEL_NUMBER of the video input.  Some video sources
   have sub "Channel Numbers".  For a typical V4L TV capture card, this is usually 1.
   I myself am using a simple NTSC video input capture card that uses the value of 1.
   If you are not in North America or have a different video standard, you WILL have to change
   the following settings and recompile/reinstall.  This set of settings is based on
   the most commonly encountered input video source types (like my bttv card) */

   if(capture->inp.index > 0) {
       capture->inp = v4l2_input();
       capture->inp.index = CHANNEL_NUMBER;
       /* Set only channel number to CHANNEL_NUMBER */
       /* V4L2 have a status field from selected video mode */
       if (-1 == ioctl (capture->deviceHandle, VIDIOC_ENUMINPUT, &capture->inp))
       {
         fprintf (stderr, "VIDEOIO ERROR: V4L2: Aren't able to set channel number\n");
         icvCloseCAM_V4L (capture);
         return -1;
       }
   } /* End if */

   /* Find Window info */
   capture->form = v4l2_format();
   capture->form.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

   if (-1 == ioctl (capture->deviceHandle, VIDIOC_G_FMT, &capture->form)) {
       fprintf( stderr, "VIDEOIO ERROR: V4L2: Could not obtain specifics of capture window.\n\n");
       icvCloseCAM_V4L(capture);
       return -1;
   }

   if (autosetup_capture_mode_v4l2(capture) == -1)
       return -1;

   /* try to set framerate */
   v4l2_set_fps(capture);

   unsigned int min;

   /* Buggy driver paranoia. */
   min = capture->form.fmt.pix.width * 2;

   if (capture->form.fmt.pix.bytesperline < min)
       capture->form.fmt.pix.bytesperline = min;

   min = capture->form.fmt.pix.bytesperline * capture->form.fmt.pix.height;

   if (capture->form.fmt.pix.sizeimage < min)
       capture->form.fmt.pix.sizeimage = min;

   capture->req = v4l2_requestbuffers();

   unsigned int buffer_number = DEFAULT_V4L_BUFFERS;

   try_again:

   capture->req.count = buffer_number;
   capture->req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
   capture->req.memory = V4L2_MEMORY_MMAP;

   if (-1 == ioctl (capture->deviceHandle, VIDIOC_REQBUFS, &capture->req))
   {
       if (EINVAL == errno)
       {
         fprintf (stderr, "%s does not support memory mapping\n", deviceName);
       } else {
         perror ("VIDIOC_REQBUFS");
       }
       /* free capture, and returns an error code */
       icvCloseCAM_V4L (capture);
       return -1;
   }

   if (capture->req.count < buffer_number)
   {
       if (buffer_number == 1)
       {
           fprintf (stderr, "Insufficient buffer memory on %s\n", deviceName);

           /* free capture, and returns an error code */
           icvCloseCAM_V4L (capture);
           return -1;
       } else {
         buffer_number--;
     fprintf (stderr, "Insufficient buffer memory on %s -- decreaseing buffers\n", deviceName);

     goto try_again;
       }
   }

   for (n_buffers = 0; n_buffers < capture->req.count; ++n_buffers)
   {
       v4l2_buffer buf = v4l2_buffer();
       buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
       buf.memory = V4L2_MEMORY_MMAP;
       buf.index = n_buffers;

       if (-1 == ioctl (capture->deviceHandle, VIDIOC_QUERYBUF, &buf)) {
           perror ("VIDIOC_QUERYBUF");

           /* free capture, and returns an error code */
           icvCloseCAM_V4L (capture);
           return -1;
       }

       capture->buffers[n_buffers].length = buf.length;
       capture->buffers[n_buffers].start =
         mmap (NULL /* start anywhere */,
               buf.length,
               PROT_READ | PROT_WRITE /* required */,
               MAP_SHARED /* recommended */,
               capture->deviceHandle, buf.m.offset);

       if (MAP_FAILED == capture->buffers[n_buffers].start) {
           perror ("mmap");

           /* free capture, and returns an error code */
           icvCloseCAM_V4L (capture);
           return -1;
       }

       if (n_buffers == 0) {
     capture->buffers[MAX_V4L_BUFFERS].start = malloc( buf.length );
     capture->buffers[MAX_V4L_BUFFERS].length = buf.length;
       }
   }

   v4l2_create_frame(capture);

   // reinitialize buffers
   capture->FirstCapture = 1;

   return 1;
}; /* End _capture_V4L2 */

/**
 * some properties can not be changed while the device is in streaming mode.
 * this method closes and re-opens the device to re-start the stream.
 * this also causes buffers to be reallocated if the frame size was changed.
 */
static bool v4l2_reset( CvCaptureCAM_V4L* capture) {
    String deviceName = capture->deviceName;
    icvCloseCAM_V4L(capture);
    capture->deviceName = deviceName;
    return _capture_V4L2(capture) == 1;
}

bool CvCaptureCAM_V4L::open(int _index)
{
   int autoindex = 0;
   char _deviceName[MAX_DEVICE_DRIVER_NAME];

   if (!numCameras)
      icvInitCapture_V4L(); /* Havent called icvInitCapture yet - do it now! */
   if (!numCameras)
     return false; /* Are there any /dev/video input sources? */

   //search index in indexList
   if ( (_index>-1) && ! ((1 << _index) & indexList) )
   {
     fprintf( stderr, "VIDEOIO ERROR: V4L: index %d is not correct!\n",_index);
     return false; /* Did someone ask for not correct video source number? */
   }

   /* Select camera, or rather, V4L video source */
   if (_index<0) { // Asking for the first device available
     for (; autoindex<MAX_CAMERAS;autoindex++)
    if (indexList & (1<<autoindex))
        break;
     if (autoindex==MAX_CAMERAS)
    return false;
     _index=autoindex;
     autoindex++;// i can recall icvOpenCAM_V4l with index=-1 for next camera
   }

   /* Print the CameraNumber at the end of the string with a width of one character */
   sprintf(_deviceName, "/dev/video%1d", _index);
   return open(_deviceName);
}

bool CvCaptureCAM_V4L::open(const char* _deviceName)
{
    FirstCapture = 1;
    width = DEFAULT_V4L_WIDTH;
    height = DEFAULT_V4L_HEIGHT;
    fps = DEFAULT_V4L_FPS;
    convert_rgb = true;
    deviceName = _deviceName;

    return _capture_V4L2(this) == 1;
}

static int read_frame_v4l2(CvCaptureCAM_V4L* capture) {
    v4l2_buffer buf = v4l2_buffer();

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (-1 == ioctl (capture->deviceHandle, VIDIOC_DQBUF, &buf)) {
        switch (errno) {
        case EAGAIN:
            return 0;

        case EIO:
        if (!(buf.flags & (V4L2_BUF_FLAG_QUEUED | V4L2_BUF_FLAG_DONE)))
        {
          if (ioctl(capture->deviceHandle, VIDIOC_QBUF, &buf) == -1)
          {
            return 0;
          }
        }
        return 0;

        default:
            /* display the error and stop processing */
            perror ("VIDIOC_DQBUF");
            return -1;
        }
   }

   assert(buf.index < capture->req.count);

   memcpy(capture->buffers[MAX_V4L_BUFFERS].start,
      capture->buffers[buf.index].start,
      capture->buffers[MAX_V4L_BUFFERS].length );
   capture->bufferIndex = MAX_V4L_BUFFERS;
   //printf("got data in buff %d, len=%d, flags=0x%X, seq=%d, used=%d)\n",
   //	  buf.index, buf.length, buf.flags, buf.sequence, buf.bytesused);

   if (-1 == ioctl (capture->deviceHandle, VIDIOC_QBUF, &buf))
       perror ("VIDIOC_QBUF");

   //set timestamp in capture struct to be timestamp of most recent frame
   capture->timestamp = buf.timestamp;

   return 1;
}

static int mainloop_v4l2(CvCaptureCAM_V4L* capture) {
    unsigned int count;

    count = 1;

    while (count-- > 0) {
        for (;;) {
            fd_set fds;
            struct timeval tv;
            int r;

            FD_ZERO (&fds);
            FD_SET (capture->deviceHandle, &fds);

            /* Timeout. */
            tv.tv_sec = 10;
            tv.tv_usec = 0;

            r = select (capture->deviceHandle+1, &fds, NULL, NULL, &tv);

            if (-1 == r) {
                if (EINTR == errno)
                    continue;

                perror ("select");
            }

            if (0 == r) {
                fprintf (stderr, "select timeout\n");

                /* end the infinite loop */
                break;
            }

            int returnCode = read_frame_v4l2 (capture);
            if(returnCode == -1)
                return -1;
            if(returnCode == 1)
                break;
        }
    }
    return 0;
}

static bool icvGrabFrameCAM_V4L(CvCaptureCAM_V4L* capture) {
   if (capture->FirstCapture) {
      /* Some general initialization must take place the first time through */

      /* This is just a technicality, but all buffers must be filled up before any
         staggered SYNC is applied.  SO, filler up. (see V4L HowTo) */

      {

        for (capture->bufferIndex = 0;
             capture->bufferIndex < ((int)capture->req.count);
             ++capture->bufferIndex)
        {

          v4l2_buffer buf = v4l2_buffer();

          buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
          buf.memory      = V4L2_MEMORY_MMAP;
          buf.index       = (unsigned long)capture->bufferIndex;

          if (-1 == ioctl (capture->deviceHandle, VIDIOC_QBUF, &buf)) {
              perror ("VIDIOC_QBUF");
              return false;
          }
        }

        /* enable the streaming */
        capture->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (-1 == ioctl (capture->deviceHandle, VIDIOC_STREAMON,
                          &capture->type)) {
            /* error enabling the stream */
            perror ("VIDIOC_STREAMON");
            return false;
        }
      }

#if defined(V4L_ABORT_BADJPEG)
        // skip first frame. it is often bad -- this is unnotied in traditional apps,
        //  but could be fatal if bad jpeg is enabled
        if(mainloop_v4l2(capture) == -1)
                return false;
#endif

      /* preparation is ok */
      capture->FirstCapture = 0;
   }

   if(mainloop_v4l2(capture) == -1) return false;

   return true;
}

/*
 * Turn a YUV4:2:0 block into an RGB block
 *
 * Video4Linux seems to use the blue, green, red channel
 * order convention-- rgb[0] is blue, rgb[1] is green, rgb[2] is red.
 *
 * Color space conversion coefficients taken from the excellent
 * http://www.inforamp.net/~poynton/ColorFAQ.html
 * In his terminology, this is a CCIR 601.1 YCbCr -> RGB.
 * Y values are given for all 4 pixels, but the U (Pb)
 * and V (Pr) are assumed constant over the 2x2 block.
 *
 * To avoid floating point arithmetic, the color conversion
 * coefficients are scaled into 16.16 fixed-point integers.
 * They were determined as follows:
 *
 *  double brightness = 1.0;  (0->black; 1->full scale)
 *  double saturation = 1.0;  (0->greyscale; 1->full color)
 *  double fixScale = brightness * 256 * 256;
 *  int rvScale = (int)(1.402 * saturation * fixScale);
 *  int guScale = (int)(-0.344136 * saturation * fixScale);
 *  int gvScale = (int)(-0.714136 * saturation * fixScale);
 *  int buScale = (int)(1.772 * saturation * fixScale);
 *  int yScale = (int)(fixScale);
 */

/* LIMIT: convert a 16.16 fixed-point value to a byte, with clipping. */
#define LIMIT(x) ((x)>0xffffff?0xff: ((x)<=0xffff?0:((x)>>16)))

static inline void
move_411_block(int yTL, int yTR, int yBL, int yBR, int u, int v,
               int /*rowPixels*/, unsigned char * rgb)
{
    const int rvScale = 91881;
    const int guScale = -22553;
    const int gvScale = -46801;
    const int buScale = 116129;
    const int yScale  = 65536;
    int r, g, b;

    g = guScale * u + gvScale * v;
//  if (force_rgb) {
//      r = buScale * u;
//      b = rvScale * v;
//  } else {
        r = rvScale * v;
        b = buScale * u;
//  }

    yTL *= yScale; yTR *= yScale;
    yBL *= yScale; yBR *= yScale;

    /* Write out top two first pixels */
    rgb[0] = LIMIT(b+yTL); rgb[1] = LIMIT(g+yTL);
    rgb[2] = LIMIT(r+yTL);

    rgb[3] = LIMIT(b+yTR); rgb[4] = LIMIT(g+yTR);
    rgb[5] = LIMIT(r+yTR);

    /* Write out top two last pixels */
    rgb += 6;
    rgb[0] = LIMIT(b+yBL); rgb[1] = LIMIT(g+yBL);
    rgb[2] = LIMIT(r+yBL);

    rgb[3] = LIMIT(b+yBR); rgb[4] = LIMIT(g+yBR);
    rgb[5] = LIMIT(r+yBR);
}

/* Converts from planar YUV420P to RGB24. */
static inline void
yuv420p_to_rgb24(int width, int height, uchar* src, uchar* dst)
{
    cvtColor(Mat(height * 3 / 2, width, CV_8U, src), Mat(height, width, CV_8UC3, dst),
             COLOR_YUV2BGR_YV12);
}

// Consider a YUV411P image of 8x2 pixels.
//
// A plane of Y values as before.
//
// A plane of U values    1       2
//                        3       4
//
// A plane of V values    1       2
//                        3       4
//
// The U1/V1 samples correspond to the ABCD pixels.
//     U2/V2 samples correspond to the EFGH pixels.
//
/* Converts from planar YUV411P to RGB24. */
/* [FD] untested... */
static void
yuv411p_to_rgb24(int width, int height,
           unsigned char *pIn0, unsigned char *pOut0)
{
    const int numpix = width * height;
    const int bytes = 24 >> 3;
    int i, j, y00, y01, y10, y11, u, v;
    unsigned char *pY = pIn0;
    unsigned char *pU = pY + numpix;
    unsigned char *pV = pU + numpix / 4;
    unsigned char *pOut = pOut0;

    for (j = 0; j <= height; j++) {
        for (i = 0; i <= width - 4; i += 4) {
            y00 = *pY;
            y01 = *(pY + 1);
            y10 = *(pY + 2);
            y11 = *(pY + 3);
            u = (*pU++) - 128;
            v = (*pV++) - 128;

            move_411_block(y00, y01, y10, y11, u, v,
                       width, pOut);

            pY += 4;
            pOut += 4 * bytes;

        }
    }
}

/* convert from 4:2:2 YUYV interlaced to RGB24 */
static void
yuyv_to_rgb24(int width, int height, unsigned char* src, unsigned char* dst) {
    cvtColor(Mat(height, width, CV_8UC2, src), Mat(height, width, CV_8UC3, dst),
             COLOR_YUV2BGR_YUYV);
}

static inline void
uyvy_to_rgb24 (int width, int height, unsigned char *src, unsigned char *dst)
{
    cvtColor(Mat(height, width, CV_8UC2, src), Mat(height, width, CV_8UC3, dst),
             COLOR_YUV2BGR_UYVY);
}

static inline void
y16_to_rgb24 (int width, int height, unsigned char* src, unsigned char* dst)
{
    Mat gray8;
    Mat(height, width, CV_16UC1, src).convertTo(gray8, CV_8U, 0.00390625);
    cvtColor(gray8,Mat(height, width, CV_8UC3, dst),COLOR_GRAY2BGR);
}

#ifdef HAVE_JPEG

/* convert from mjpeg to rgb24 */
static bool
mjpeg_to_rgb24(int width, int height, unsigned char* src, int length, IplImage* dst) {
    Mat temp = cvarrToMat(dst);
    imdecode(Mat(1, length, CV_8U, src), IMREAD_COLOR, &temp);
    return temp.data && temp.cols == width && temp.rows == height;
}

#endif

/*
 * BAYER2RGB24 ROUTINE TAKEN FROM:
 *
 * Sonix SN9C10x based webcam basic I/F routines
 * Takafumi Mizuno <taka-qce@ls-a.jp>
 *
 */
static void bayer2rgb24(long int WIDTH, long int HEIGHT, unsigned char *src, unsigned char *dst)
{
    long int i;
    unsigned char *rawpt, *scanpt;
    long int size;

    rawpt = src;
    scanpt = dst;
    size = WIDTH*HEIGHT;

    for ( i = 0; i < size; i++ ) {
  if ( (i/WIDTH) % 2 == 0 ) {
      if ( (i % 2) == 0 ) {
    /* B */
    if ( (i > WIDTH) && ((i % WIDTH) > 0) ) {
        *scanpt++ = (*(rawpt-WIDTH-1)+*(rawpt-WIDTH+1)+
         *(rawpt+WIDTH-1)+*(rawpt+WIDTH+1))/4;  /* R */
        *scanpt++ = (*(rawpt-1)+*(rawpt+1)+
         *(rawpt+WIDTH)+*(rawpt-WIDTH))/4;      /* G */
        *scanpt++ = *rawpt;                                     /* B */
    } else {
        /* first line or left column */
        *scanpt++ = *(rawpt+WIDTH+1);           /* R */
        *scanpt++ = (*(rawpt+1)+*(rawpt+WIDTH))/2;      /* G */
        *scanpt++ = *rawpt;                             /* B */
    }
      } else {
    /* (B)G */
    if ( (i > WIDTH) && ((i % WIDTH) < (WIDTH-1)) ) {
        *scanpt++ = (*(rawpt+WIDTH)+*(rawpt-WIDTH))/2;  /* R */
        *scanpt++ = *rawpt;                                     /* G */
        *scanpt++ = (*(rawpt-1)+*(rawpt+1))/2;          /* B */
    } else {
        /* first line or right column */
        *scanpt++ = *(rawpt+WIDTH);     /* R */
        *scanpt++ = *rawpt;             /* G */
        *scanpt++ = *(rawpt-1); /* B */
    }
      }
  } else {
      if ( (i % 2) == 0 ) {
    /* G(R) */
    if ( (i < (WIDTH*(HEIGHT-1))) && ((i % WIDTH) > 0) ) {
        *scanpt++ = (*(rawpt-1)+*(rawpt+1))/2;          /* R */
        *scanpt++ = *rawpt;                                     /* G */
        *scanpt++ = (*(rawpt+WIDTH)+*(rawpt-WIDTH))/2;  /* B */
    } else {
        /* bottom line or left column */
        *scanpt++ = *(rawpt+1);         /* R */
        *scanpt++ = *rawpt;                     /* G */
        *scanpt++ = *(rawpt-WIDTH);             /* B */
    }
      } else {
    /* R */
    if ( i < (WIDTH*(HEIGHT-1)) && ((i % WIDTH) < (WIDTH-1)) ) {
        *scanpt++ = *rawpt;                                     /* R */
        *scanpt++ = (*(rawpt-1)+*(rawpt+1)+
         *(rawpt-WIDTH)+*(rawpt+WIDTH))/4;      /* G */
        *scanpt++ = (*(rawpt-WIDTH-1)+*(rawpt-WIDTH+1)+
         *(rawpt+WIDTH-1)+*(rawpt+WIDTH+1))/4;  /* B */
    } else {
        /* bottom line or right column */
        *scanpt++ = *rawpt;                             /* R */
        *scanpt++ = (*(rawpt-1)+*(rawpt-WIDTH))/2;      /* G */
        *scanpt++ = *(rawpt-WIDTH-1);           /* B */
    }
      }
  }
  rawpt++;
    }

}

// SGBRG to RGB24
// for some reason, red and blue needs to be swapped
// at least for  046d:092f Logitech, Inc. QuickCam Express Plus to work
//see: http://www.siliconimaging.com/RGB%20Bayer.htm
//and 4.6 at http://tldp.org/HOWTO/html_single/libdc1394-HOWTO/
static void sgbrg2rgb24(long int WIDTH, long int HEIGHT, unsigned char *src, unsigned char *dst)
{
    long int i;
    unsigned char *rawpt, *scanpt;
    long int size;

    rawpt = src;
    scanpt = dst;
    size = WIDTH*HEIGHT;

    for ( i = 0; i < size; i++ )
    {
        if ( (i/WIDTH) % 2 == 0 ) //even row
        {
            if ( (i % 2) == 0 ) //even pixel
            {
                if ( (i > WIDTH) && ((i % WIDTH) > 0) )
                {
                    *scanpt++ = (*(rawpt-1)+*(rawpt+1))/2;       /* R */
                    *scanpt++ = *(rawpt);                        /* G */
                    *scanpt++ = (*(rawpt-WIDTH) + *(rawpt+WIDTH))/2;      /* B */
                } else
                {
                  /* first line or left column */

                  *scanpt++ = *(rawpt+1);           /* R */
                  *scanpt++ = *(rawpt);             /* G */
                  *scanpt++ =  *(rawpt+WIDTH);      /* B */
                }
            } else //odd pixel
            {
                if ( (i > WIDTH) && ((i % WIDTH) < (WIDTH-1)) )
                {
                    *scanpt++ = *(rawpt);       /* R */
                    *scanpt++ = (*(rawpt-1)+*(rawpt+1)+*(rawpt-WIDTH)+*(rawpt+WIDTH))/4; /* G */
                    *scanpt++ = (*(rawpt-WIDTH-1) + *(rawpt-WIDTH+1) + *(rawpt+WIDTH-1) + *(rawpt+WIDTH+1))/4;      /* B */
                } else
                {
                    /* first line or right column */

                    *scanpt++ = *(rawpt);       /* R */
                    *scanpt++ = (*(rawpt-1)+*(rawpt+WIDTH))/2; /* G */
                    *scanpt++ = *(rawpt+WIDTH-1);      /* B */
                }
            }
        } else
        { //odd row
            if ( (i % 2) == 0 ) //even pixel
            {
                if ( (i < (WIDTH*(HEIGHT-1))) && ((i % WIDTH) > 0) )
                {
                    *scanpt++ =  (*(rawpt-WIDTH-1)+*(rawpt-WIDTH+1)+*(rawpt+WIDTH-1)+*(rawpt+WIDTH+1))/4;          /* R */
                    *scanpt++ =  (*(rawpt-1)+*(rawpt+1)+*(rawpt-WIDTH)+*(rawpt+WIDTH))/4;      /* G */
                    *scanpt++ =  *(rawpt); /* B */
                } else
                {
                    /* bottom line or left column */

                    *scanpt++ =  *(rawpt-WIDTH+1);          /* R */
                    *scanpt++ =  (*(rawpt+1)+*(rawpt-WIDTH))/2;      /* G */
                    *scanpt++ =  *(rawpt); /* B */
                }
            } else
            { //odd pixel
                if ( i < (WIDTH*(HEIGHT-1)) && ((i % WIDTH) < (WIDTH-1)) )
                {
                    *scanpt++ = (*(rawpt-WIDTH)+*(rawpt+WIDTH))/2;  /* R */
                    *scanpt++ = *(rawpt);      /* G */
                    *scanpt++ = (*(rawpt-1)+*(rawpt+1))/2; /* B */
                } else
                {
                    /* bottom line or right column */

                    *scanpt++ = (*(rawpt-WIDTH));  /* R */
                    *scanpt++ = *(rawpt);      /* G */
                    *scanpt++ = (*(rawpt-1)); /* B */
                }
            }
        }
        rawpt++;
    }
}

static inline void
rgb24_to_rgb24 (int width, int height, unsigned char *src, unsigned char *dst)
{
    cvtColor(Mat(height, width, CV_8UC3, src), Mat(height, width, CV_8UC3, dst), COLOR_RGB2BGR);
}

#define CLAMP(x)        ((x)<0?0:((x)>255)?255:(x))

typedef struct {
  int is_abs;
  int len;
  int val;
} code_table_t;


/* local storage */
static code_table_t table[256];
static int init_done = 0;


/*
  sonix_decompress_init
  =====================
    pre-calculates a locally stored table for efficient huffman-decoding.

  Each entry at index x in the table represents the codeword
  present at the MSB of byte x.

*/
static void sonix_decompress_init(void)
{
  int i;
  int is_abs, val, len;

  for (i = 0; i < 256; i++) {
    is_abs = 0;
    val = 0;
    len = 0;
    if ((i & 0x80) == 0) {
      /* code 0 */
      val = 0;
      len = 1;
    }
    else if ((i & 0xE0) == 0x80) {
      /* code 100 */
      val = +4;
      len = 3;
    }
    else if ((i & 0xE0) == 0xA0) {
      /* code 101 */
      val = -4;
      len = 3;
    }
    else if ((i & 0xF0) == 0xD0) {
      /* code 1101 */
      val = +11;
      len = 4;
    }
    else if ((i & 0xF0) == 0xF0) {
      /* code 1111 */
      val = -11;
      len = 4;
    }
    else if ((i & 0xF8) == 0xC8) {
      /* code 11001 */
      val = +20;
      len = 5;
    }
    else if ((i & 0xFC) == 0xC0) {
      /* code 110000 */
      val = -20;
      len = 6;
    }
    else if ((i & 0xFC) == 0xC4) {
      /* code 110001xx: unknown */
      val = 0;
      len = 8;
    }
    else if ((i & 0xF0) == 0xE0) {
      /* code 1110xxxx */
      is_abs = 1;
      val = (i & 0x0F) << 4;
      len = 8;
    }
    table[i].is_abs = is_abs;
    table[i].val = val;
    table[i].len = len;
  }

  init_done = 1;
}


/*
  sonix_decompress
  ================
    decompresses an image encoded by a SN9C101 camera controller chip.

  IN    width
    height
    inp         pointer to compressed frame (with header already stripped)
  OUT   outp    pointer to decompressed frame

  Returns 0 if the operation was successful.
  Returns <0 if operation failed.

*/
static int sonix_decompress(int width, int height, unsigned char *inp, unsigned char *outp)
{
  int row, col;
  int val;
  int bitpos;
  unsigned char code;
  unsigned char *addr;

  if (!init_done) {
    /* do sonix_decompress_init first! */
    return -1;
  }

  bitpos = 0;
  for (row = 0; row < height; row++) {

    col = 0;



    /* first two pixels in first two rows are stored as raw 8-bit */
    if (row < 2) {
      addr = inp + (bitpos >> 3);
      code = (addr[0] << (bitpos & 7)) | (addr[1] >> (8 - (bitpos & 7)));
      bitpos += 8;
      *outp++ = code;

      addr = inp + (bitpos >> 3);
      code = (addr[0] << (bitpos & 7)) | (addr[1] >> (8 - (bitpos & 7)));
      bitpos += 8;
      *outp++ = code;

      col += 2;
    }

    while (col < width) {
      /* get bitcode from bitstream */
      addr = inp + (bitpos >> 3);
      code = (addr[0] << (bitpos & 7)) | (addr[1] >> (8 - (bitpos & 7)));

      /* update bit position */
      bitpos += table[code].len;

      /* calculate pixel value */
      val = table[code].val;
      if (!table[code].is_abs) {
        /* value is relative to top and left pixel */
        if (col < 2) {
          /* left column: relative to top pixel */
          val += outp[-2*width];
        }
        else if (row < 2) {
          /* top row: relative to left pixel */
          val += outp[-2];
        }
        else {
          /* main area: average of left pixel and top pixel */
          val += (outp[-2] + outp[-2*width]) / 2;
        }
      }

      /* store pixel */
      *outp++ = CLAMP(val);
      col++;
    }
  }

  return 0;
}

static IplImage* icvRetrieveFrameCAM_V4L( CvCaptureCAM_V4L* capture, int) {
    /* Now get what has already been captured as a IplImage return */
    // we need memory iff convert_rgb is true
    bool recreate_frame = capture->frame_allocated != capture->convert_rgb;

    if (!capture->convert_rgb) {
        // for mjpeg streams the size might change in between, so we have to change the header
        recreate_frame += capture->frame.imageSize != (int)capture->buffers[capture->bufferIndex].length;
    }

    if(recreate_frame) {
        // printf("realloc %d %zu\n", capture->frame.imageSize, capture->buffers[capture->bufferIndex].length);
        if(capture->frame_allocated)
            cvFree(&capture->frame.imageData);
        v4l2_create_frame(capture);
    }

    if(!capture->convert_rgb) {
        capture->frame.imageData = (char*)capture->buffers[capture->bufferIndex].start;
        return &capture->frame;
    }

    switch (capture->palette)
    {
    case V4L2_PIX_FMT_BGR24:
        memcpy((char *)capture->frame.imageData,
               (char *)capture->buffers[capture->bufferIndex].start,
               capture->frame.imageSize);
        break;

    case V4L2_PIX_FMT_YVU420:
        yuv420p_to_rgb24(capture->form.fmt.pix.width,
                 capture->form.fmt.pix.height,
                 (unsigned char*)(capture->buffers[capture->bufferIndex].start),
                 (unsigned char*)capture->frame.imageData);
        break;

    case V4L2_PIX_FMT_YUV411P:
        yuv411p_to_rgb24(capture->form.fmt.pix.width,
                 capture->form.fmt.pix.height,
                 (unsigned char*)(capture->buffers[capture->bufferIndex].start),
                 (unsigned char*)capture->frame.imageData);
        break;
#ifdef HAVE_JPEG
    case V4L2_PIX_FMT_MJPEG:
    case V4L2_PIX_FMT_JPEG:
        if (!mjpeg_to_rgb24(capture->form.fmt.pix.width,
                    capture->form.fmt.pix.height,
                    (unsigned char*)(capture->buffers[capture->bufferIndex]
                             .start),
                    capture->buffers[capture->bufferIndex].length,
                    &capture->frame))
          return 0;
        break;
#endif

    case V4L2_PIX_FMT_YUYV:
        yuyv_to_rgb24(capture->form.fmt.pix.width,
                  capture->form.fmt.pix.height,
                  (unsigned char*)(capture->buffers[capture->bufferIndex].start),
                  (unsigned char*)capture->frame.imageData);
        break;
    case V4L2_PIX_FMT_UYVY:
        uyvy_to_rgb24(capture->form.fmt.pix.width,
                  capture->form.fmt.pix.height,
                  (unsigned char*)(capture->buffers[capture->bufferIndex].start),
                  (unsigned char*)capture->frame.imageData);
        break;
    case V4L2_PIX_FMT_SBGGR8:
        bayer2rgb24(capture->form.fmt.pix.width,
                capture->form.fmt.pix.height,
                (unsigned char*)capture->buffers[capture->bufferIndex].start,
                (unsigned char*)capture->frame.imageData);
        break;

    case V4L2_PIX_FMT_SN9C10X:
        sonix_decompress_init();
        sonix_decompress(capture->form.fmt.pix.width,
                 capture->form.fmt.pix.height,
                 (unsigned char*)capture->buffers[capture->bufferIndex].start,
                 (unsigned char*)capture->buffers[(capture->bufferIndex+1) % capture->req.count].start);

        bayer2rgb24(capture->form.fmt.pix.width,
                capture->form.fmt.pix.height,
                (unsigned char*)capture->buffers[(capture->bufferIndex+1) % capture->req.count].start,
                (unsigned char*)capture->frame.imageData);
        break;

    case V4L2_PIX_FMT_SGBRG8:
        sgbrg2rgb24(capture->form.fmt.pix.width,
                capture->form.fmt.pix.height,
                (unsigned char*)capture->buffers[(capture->bufferIndex+1) % capture->req.count].start,
                (unsigned char*)capture->frame.imageData);
        break;
    case V4L2_PIX_FMT_RGB24:
        rgb24_to_rgb24(capture->form.fmt.pix.width,
                capture->form.fmt.pix.height,
                (unsigned char*)capture->buffers[(capture->bufferIndex+1) % capture->req.count].start,
                (unsigned char*)capture->frame.imageData);
        break;
    case V4L2_PIX_FMT_Y16:
        if(capture->convert_rgb){
            y16_to_rgb24(capture->form.fmt.pix.width,
                         capture->form.fmt.pix.height,
                         (unsigned char*)capture->buffers[capture->bufferIndex].start,
                         (unsigned char*)capture->frame.imageData);
        }else{
            memcpy((char *)capture->frame.imageData,
                   (char *)capture->buffers[capture->bufferIndex].start,
                   capture->frame.imageSize);
        }
        break;
    }

    return(&capture->frame);
}

static inline __u32 capPropertyToV4L2(int prop) {
    switch (prop) {
    case CV_CAP_PROP_BRIGHTNESS:
        return V4L2_CID_BRIGHTNESS;
    case CV_CAP_PROP_CONTRAST:
        return V4L2_CID_CONTRAST;
    case CV_CAP_PROP_SATURATION:
        return V4L2_CID_SATURATION;
    case CV_CAP_PROP_HUE:
        return V4L2_CID_HUE;
    case CV_CAP_PROP_GAIN:
        return V4L2_CID_GAIN;
    case CV_CAP_PROP_AUTO_EXPOSURE:
        return V4L2_CID_EXPOSURE_AUTO;
    case CV_CAP_PROP_EXPOSURE:
        return V4L2_CID_EXPOSURE_ABSOLUTE;
    case CV_CAP_PROP_AUTOFOCUS:
        return V4L2_CID_FOCUS_AUTO;
    case CV_CAP_PROP_FOCUS:
        return V4L2_CID_FOCUS_ABSOLUTE;
    default:
        return -1;
    }
}

static double icvGetPropertyCAM_V4L (const CvCaptureCAM_V4L* capture,
                                     int property_id ) {
  {
      v4l2_format form;
      memset(&form, 0, sizeof(v4l2_format));
      form.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      if (-1 == ioctl (capture->deviceHandle, VIDIOC_G_FMT, &form)) {
          /* display an error message, and return an error code */
          perror ("VIDIOC_G_FMT");
          return -1;
      }

      switch (property_id) {
      case CV_CAP_PROP_FRAME_WIDTH:
          return form.fmt.pix.width;
      case CV_CAP_PROP_FRAME_HEIGHT:
          return form.fmt.pix.height;
      case CV_CAP_PROP_FOURCC:
      case CV_CAP_PROP_MODE:
          return capture->palette;
      case CV_CAP_PROP_FORMAT:
          return CV_MAKETYPE(CV_8U, capture->frame.nChannels);
      case CV_CAP_PROP_CONVERT_RGB:
          return capture->convert_rgb;
      }

      if(property_id == CV_CAP_PROP_FPS) {
          v4l2_streamparm sp = v4l2_streamparm();
          sp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
          if (ioctl(capture->deviceHandle, VIDIOC_G_PARM, &sp) < 0){
              fprintf(stderr, "VIDEOIO ERROR: V4L: Unable to get camera FPS\n");
              return -1;
          }

          return sp.parm.capture.timeperframe.denominator / (double)sp.parm.capture.timeperframe.numerator;
      }

      /* initialize the control structure */

      if(property_id == CV_CAP_PROP_POS_MSEC) {
          if (capture->FirstCapture) {
            return 0;
          } else {
            return 1000 * capture->timestamp.tv_sec + ((double) capture->timestamp.tv_usec) / 1000;
          }
      }

      __u32 v4l2id = capPropertyToV4L2(property_id);

      if(v4l2id == __u32(-1)) {
          fprintf(stderr,
                  "VIDEOIO ERROR: V4L2: getting property #%d is not supported\n",
                  property_id);
          return -1;
      }

      v4l2_control control = {v4l2id, 0};

      if (-1 == ioctl (capture->deviceHandle, VIDIOC_G_CTRL,
                        &control)) {

          fprintf( stderr, "VIDEOIO ERROR: V4L2: ");
          switch (property_id) {
          case CV_CAP_PROP_BRIGHTNESS:
              fprintf (stderr, "Brightness");
              break;
          case CV_CAP_PROP_CONTRAST:
              fprintf (stderr, "Contrast");
              break;
          case CV_CAP_PROP_SATURATION:
              fprintf (stderr, "Saturation");
              break;
          case CV_CAP_PROP_HUE:
              fprintf (stderr, "Hue");
              break;
          case CV_CAP_PROP_GAIN:
              fprintf (stderr, "Gain");
              break;
          case CV_CAP_PROP_AUTO_EXPOSURE:
              fprintf (stderr, "Auto Exposure");
              break;
          case CV_CAP_PROP_EXPOSURE:
              fprintf (stderr, "Exposure");
              break;
          case CV_CAP_PROP_AUTOFOCUS:
              fprintf (stderr, "Autofocus");
              break;
          case CV_CAP_PROP_FOCUS:
              fprintf (stderr, "Focus");
              break;
          }
          fprintf (stderr, " is not supported by your device\n");

          return -1;
      }

      /* get the min/max values */
      Range range = capture->getRange(property_id);

      /* all was OK, so convert to 0.0 - 1.0 range, and return the value */
      return ((double)control.value - range.start) / range.size();

  }
};

static bool icvSetControl (CvCaptureCAM_V4L* capture,
                          int property_id, double value) {

  /* limitation of the input value */
  if (value < 0.0) {
    value = 0.0;
  } else if (value > 1.0) {
    value = 1.0;
  }

    /* initialisations */
    __u32 v4l2id = capPropertyToV4L2(property_id);

    if(v4l2id == __u32(-1)) {
        fprintf(stderr,
                "VIDEOIO ERROR: V4L2: setting property #%d is not supported\n",
                property_id);
        return -1;
    }

    /* get the min/max values */
    Range range = capture->getRange(property_id);

    /* scale the value we want to set */
    value = value * range.size() + range.start;

    /* set which control we want to set */
    v4l2_control control = {v4l2id, int(value)};

    /* The driver may clamp the value or return ERANGE, ignored here */
    if (-1 == ioctl(capture->deviceHandle, VIDIOC_S_CTRL, &control) && errno != ERANGE) {
        perror ("VIDIOC_S_CTRL");
        return false;
    }

    if(control.id == V4L2_CID_EXPOSURE_AUTO && control.value == V4L2_EXPOSURE_MANUAL) {
        // update the control range for expose after disabling autoexposure
        // as it is not read correctly at startup
        // TODO check this again as it might be fixed with Linux 4.5
        v4l2_control_range(capture, V4L2_CID_EXPOSURE_ABSOLUTE);
    }

    /* all was OK */
    return true;
}

static int icvSetPropertyCAM_V4L( CvCaptureCAM_V4L* capture,
                                  int property_id, double value ){
    static int width = 0, height = 0;
    bool retval = false;
    bool possible;

    /* two subsequent calls setting WIDTH and HEIGHT will change
       the video size */

    switch (property_id) {
    case CV_CAP_PROP_FRAME_WIDTH:
        width = cvRound(value);
        retval = width != 0;
        if(width !=0 && height != 0) {
            capture->width = width;
            capture->height = height;
            retval = v4l2_reset(capture);
            width = height = 0;
        }
        break;
    case CV_CAP_PROP_FRAME_HEIGHT:
        height = cvRound(value);
        retval = height != 0;
        if(width !=0 && height != 0) {
            capture->width = width;
            capture->height = height;
            retval = v4l2_reset(capture);
            width = height = 0;
        }
        break;
    case CV_CAP_PROP_FPS:
        capture->fps = value;
        retval = v4l2_reset(capture);
        break;
    case CV_CAP_PROP_CONVERT_RGB:
        // returns "0" for formats we do not know how to map to IplImage
        possible = v4l2_num_channels(capture->palette);
        capture->convert_rgb = bool(value) && possible;
        retval = possible || !bool(value);
        break;
    case CV_CAP_PROP_FOURCC:
        {
            __u32 old_palette = capture->palette;
            __u32 new_palette = static_cast<__u32>(value);
            capture->palette = new_palette;
            if (v4l2_reset(capture)) {
                retval = true;
            } else {
                capture->palette = old_palette;
                v4l2_reset(capture);
                retval = false;
            }
        }
        break;
    default:
        retval = icvSetControl(capture, property_id, value);
        break;
    }

    /* return the the status */
    return retval;
}

static void icvCloseCAM_V4L( CvCaptureCAM_V4L* capture ){
   /* Deallocate space - Hopefully, no leaks */

   if (!capture->deviceName.empty())
   {
       if (capture->deviceHandle != -1)
       {
           capture->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
           if (-1 == ioctl(capture->deviceHandle, VIDIOC_STREAMOFF, &capture->type)) {
               perror ("Unable to stop the stream");
       }

       for (unsigned int n_buffers_ = 0; n_buffers_ < capture->req.count; ++n_buffers_)
       {
           if (-1 == munmap (capture->buffers[n_buffers_].start, capture->buffers[n_buffers_].length)) {
               perror ("munmap");
           }
       }

       if (capture->buffers[MAX_V4L_BUFFERS].start)
       {
           free(capture->buffers[MAX_V4L_BUFFERS].start);
           capture->buffers[MAX_V4L_BUFFERS].start = 0;
       }
     }

     if (capture->deviceHandle != -1)
       close(capture->deviceHandle);

     if (capture->frame_allocated && capture->frame.imageData)
         cvFree(&capture->frame.imageData);

     capture->deviceName.clear(); // flag that the capture is closed
   }
};

bool CvCaptureCAM_V4L::grabFrame()
{
    return icvGrabFrameCAM_V4L( this );
}

IplImage* CvCaptureCAM_V4L::retrieveFrame(int)
{
    return icvRetrieveFrameCAM_V4L( this, 0 );
}

double CvCaptureCAM_V4L::getProperty( int propId ) const
{
    return icvGetPropertyCAM_V4L( this, propId );
}

bool CvCaptureCAM_V4L::setProperty( int propId, double value )
{
    return icvSetPropertyCAM_V4L( this, propId, value );
}

} // end namespace cv

CvCapture* cvCreateCameraCapture_V4L( int index )
{
    cv::CvCaptureCAM_V4L* capture = new cv::CvCaptureCAM_V4L();

    if(capture->open(index))
        return capture;

    delete capture;
    return NULL;
}

CvCapture* cvCreateCameraCapture_V4L( const char * deviceName )
{
    cv::CvCaptureCAM_V4L* capture = new cv::CvCaptureCAM_V4L();

    if(capture->open( deviceName ))
        return capture;

    delete capture;
    return NULL;
}

#endif
