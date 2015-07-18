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
              Submit your fixes at https://github.com/Itseez/opencv/
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

11th patch: Apr 13, 2010, Filipe Almeida filipe.almeida@ist.utl.pt
- Tries to setup all properties first through v4l2_ioctl call.
- Allows setting up all Video4Linux properties through cvSetCaptureProperty instead of only CV_CAP_PROP_BRIGHTNESS, CV_CAP_PROP_CONTRAST, CV_CAP_PROP_SATURATION, CV_CAP_PROP_HUE, CV_CAP_PROP_GAIN and CV_CAP_PROP_EXPOSURE.

12th patch: Apr 16, 2010, Filipe Almeida filipe.almeida@ist.utl.pt
- CvCaptureCAM_V4L structure cleanup (no longer needs <PROPERTY>_{min,max,} variables)
- Introduction of v4l2_ctrl_range - minimum and maximum allowed values for v4l controls
- Allows setting up all Video4Linux properties through cvSetCaptureProperty using input values between 0.0 and 1.0
- Gets v4l properties first through v4l2_ioctl call (ignores capture->is_v4l2_device)
- cvGetCaptureProperty adjusted to support the changes
- Returns device properties to initial values after device closes

13th patch: Apr 27, 2010, Filipe Almeida filipe.almeida@ist.utl.pt
- Solved problem mmaping the device using uvcvideo driver (use o v4l2_mmap instead of mmap)
make & enjoy!

14th patch: May 10, 2010, Filipe Almeida filipe.almeida@ist.utl.pt
- Bug #142: Solved/Workaround "setting frame width and height does not work"
  There was a problem setting up the size when the input is a v4l2 device
  The workaround closes the camera and reopens it with the new definition
  Planning for future rewrite of this whole library (July/August 2010)

15th patch: May 12, 2010, Filipe Almeida filipe.almeida@ist.utl.pt
- Broken compile of library (include "_videoio.h")

16th patch: Dec 16, 2014, Joseph Howse josephhowse@nummist.com
- Allow getting/setting CV_CAP_PROP_MODE. These values are supported:
    - CV_CAP_MODE_BGR  : BGR24 (default)
    - CV_CAP_MODE_RGB  : RGB24
    - CV_CAP_MODE_GRAY : Y8, extracted from YUV420
- Tested successfully on these cameras:
    - PlayStation 3 Eye
    - Logitech C920
    - Odroid USB-CAM 720P

17th patch: May 9, 2015, Matt Sandler
 added supported for CV_CAP_PROP_POS_MSEC, CV_CAP_PROP_POS_FRAMES, CV_CAP_PROP_FPS

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

#if !defined WIN32 && defined HAVE_LIBV4L

#define CLEAR(x) memset (&(x), 0, sizeof (x))

#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <string.h>
#include <stdlib.h>
#include <asm/types.h>          /* for videodev2.h */
#include <assert.h>
#include <sys/stat.h>
#include <sys/ioctl.h>

#ifdef HAVE_CAMV4L
#include <linux/videodev.h>
#endif
#ifdef HAVE_CAMV4L2
#include <linux/videodev2.h>
#endif

#include <libv4l1.h>
#include <libv4l2.h>

/* Defaults - If your board can do better, set it here.  Set for the most common type inputs. */
#define DEFAULT_V4L_WIDTH  640
#define DEFAULT_V4L_HEIGHT 480

#define CHANNEL_NUMBER 1
#define MAX_CAMERAS 8


// default and maximum number of V4L buffers, not including last, 'special' buffer
#define MAX_V4L_BUFFERS 10
#define DEFAULT_V4L_BUFFERS 4

// if enabled, copies data from the buffer. this uses a bit more memory,
//  but much more reliable for some UVC cameras
#define USE_TEMP_BUFFER

#define MAX_DEVICE_DRIVER_NAME 80

/* Device Capture Objects */
/* V4L2 structure */
struct buffer
{
  void *  start;
  size_t  length;
};
static unsigned int n_buffers = 0;

/* TODO: Dilemas: */
/* TODO: Consider drop the use of this data structure and perform ioctl to obtain needed values */
/* TODO: Consider at program exit return controls to the initial values - See v4l2_free_ranges function */
/* TODO: Consider at program exit reset the device to default values - See v4l2_free_ranges function */
typedef struct v4l2_ctrl_range {
  __u32 ctrl_id;
  __s32 initial_value;
  __s32 current_value;
  __s32 minimum;
  __s32 maximum;
  __s32 default_value;
} v4l2_ctrl_range;

typedef struct CvCaptureCAM_V4L
{
    char* deviceName;
    int deviceHandle;
    int bufferIndex;
    int FirstCapture;

    int width; int height;
    int mode;

    struct video_capability capability;
    struct video_window     captureWindow;
    struct video_picture    imageProperties;
    struct video_mbuf       memoryBuffer;
    struct video_mmap       *mmaps;
    char *memoryMap;
    IplImage frame;

   /* V4L2 variables */
   buffer buffers[MAX_V4L_BUFFERS + 1];
   struct v4l2_capability cap;
   struct v4l2_input inp;
   struct v4l2_format form;
   struct v4l2_crop crop;
   struct v4l2_cropcap cropcap;
   struct v4l2_requestbuffers req;
   struct v4l2_jpegcompression compr;
   struct v4l2_control control;
   enum v4l2_buf_type type;
   struct v4l2_queryctrl queryctrl;

   struct timeval timestamp;

   /** value set the buffer of V4L*/
   int sequence;

   /* V4L2 control variables */
   v4l2_ctrl_range** v4l2_ctrl_ranges;
   int v4l2_ctrl_count;

   int is_v4l2_device;
}
CvCaptureCAM_V4L;

static void icvCloseCAM_V4L( CvCaptureCAM_V4L* capture );

static int icvGrabFrameCAM_V4L( CvCaptureCAM_V4L* capture );
static IplImage* icvRetrieveFrameCAM_V4L( CvCaptureCAM_V4L* capture, int );
CvCapture* cvCreateCameraCapture_V4L( int index );

static double icvGetPropertyCAM_V4L( CvCaptureCAM_V4L* capture, int property_id );
static int    icvSetPropertyCAM_V4L( CvCaptureCAM_V4L* capture, int property_id, double value );

static int icvSetVideoSize( CvCaptureCAM_V4L* capture, int w, int h);

/***********************   Implementations  ***************************************/

static int numCameras = 0;
static int indexList = 0;

// IOCTL handling for V4L2
#ifdef HAVE_IOCTL_ULONG
static int xioctl( int fd, unsigned long request, void *arg)
#else
static int xioctl( int fd, int request, void *arg)
#endif
{

  int r;


  do r = v4l2_ioctl (fd, request, arg);
  while (-1 == r && EINTR == errno);

  return r;

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


static int try_init_v4l(CvCaptureCAM_V4L* capture, char *deviceName)

{

  // if detect = -1 then unable to open device
  // if detect = 0 then detected nothing
  // if detect = 1 then V4L device
  int detect = 0;


  // Test device for V4L compability

  /* Test using an open to see if this new device name really does exists. */
  /* No matter what the name - it still must be opened! */
  capture->deviceHandle = v4l1_open(deviceName, O_RDWR);


  if (capture->deviceHandle == 0)
  {
    detect = -1;

    icvCloseCAM_V4L(capture);
  }

  if (detect == 0)
  {
    /* Query the newly opened device for its capabilities */
    if (v4l1_ioctl(capture->deviceHandle, VIDIOCGCAP, &capture->capability) < 0)
    {
      detect = 0;

      icvCloseCAM_V4L(capture);
    }
      else
    {
      detect = 1;
    }
  }

  return detect;

}


static int try_init_v4l2(CvCaptureCAM_V4L* capture, char *deviceName)
{

  // if detect = -1 then unable to open device
  // if detect = 0 then detected nothing
  // if detect = 1 then V4L2 device
  int detect = 0;


  // Test device for V4L2 compability

  /* Open and test V4L2 device */
  capture->deviceHandle = v4l2_open (deviceName, O_RDWR /* required */ | O_NONBLOCK, 0);



  if (capture->deviceHandle == 0)
  {
    detect = -1;

    icvCloseCAM_V4L(capture);
  }

  if (detect == 0)
  {
    CLEAR (capture->cap);
    if (-1 == xioctl (capture->deviceHandle, VIDIOC_QUERYCAP, &capture->cap))
    {
      detect = 0;

      icvCloseCAM_V4L(capture);
    }
      else
    {
      CLEAR (capture->capability);
      capture->capability.type = capture->cap.capabilities;

      /* Query channels number */
      if (-1 != xioctl (capture->deviceHandle, VIDIOC_G_INPUT, &capture->capability.channels))
      {
        detect = 1;
      }
    }
  }

  return detect;

}


static void v4l2_free_ranges(CvCaptureCAM_V4L* capture) {
  int i;
  if (capture->v4l2_ctrl_ranges != NULL) {
    for (i = 0; i < capture->v4l2_ctrl_count; i++) {
      /* Return device to initial values: */
      /* double value = (capture->v4l2_ctrl_ranges[i]->initial_value == 0)?0.0:((float)capture->v4l2_ctrl_ranges[i]->initial_value - capture->v4l2_ctrl_ranges[i]->minimum) / (capture->v4l2_ctrl_ranges[i]->maximum - capture->v4l2_ctrl_ranges[i]->minimum); */
      /* Return device to default values: */
      /* double value = (capture->v4l2_ctrl_ranges[i]->default_value == 0)?0.0:((float)capture->v4l2_ctrl_ranges[i]->default_value - capture->v4l2_ctrl_ranges[i]->minimum + 1) / (capture->v4l2_ctrl_ranges[i]->maximum - capture->v4l2_ctrl_ranges[i]->minimum); */

      /* icvSetPropertyCAM_V4L(capture, capture->v4l2_ctrl_ranges[i]->ctrl_id, value); */
      free(capture->v4l2_ctrl_ranges[i]);
    }
  }
  free(capture->v4l2_ctrl_ranges);
  capture->v4l2_ctrl_count  = 0;
  capture->v4l2_ctrl_ranges = NULL;
}

static void v4l2_add_ctrl_range(CvCaptureCAM_V4L* capture, v4l2_control* ctrl) {
  v4l2_ctrl_range* range    = (v4l2_ctrl_range*)malloc(sizeof(v4l2_ctrl_range));
  range->ctrl_id            = ctrl->id;
  range->initial_value      = ctrl->value;
  range->current_value      = ctrl->value;
  range->minimum            = capture->queryctrl.minimum;
  range->maximum            = capture->queryctrl.maximum;
  range->default_value      = capture->queryctrl.default_value;
  capture->v4l2_ctrl_ranges[capture->v4l2_ctrl_count] = range;
  capture->v4l2_ctrl_count += 1;
  capture->v4l2_ctrl_ranges = (v4l2_ctrl_range**)realloc((v4l2_ctrl_range**)capture->v4l2_ctrl_ranges, (capture->v4l2_ctrl_count + 1) * sizeof(v4l2_ctrl_range*));
}

static int v4l2_get_ctrl_default(CvCaptureCAM_V4L* capture, __u32 id) {
  int i;
  for (i = 0; i < capture->v4l2_ctrl_count; i++) {
    if (id == capture->v4l2_ctrl_ranges[i]->ctrl_id) {
      return capture->v4l2_ctrl_ranges[i]->default_value;
    }
  }
  return -1;
}

static int v4l2_get_ctrl_min(CvCaptureCAM_V4L* capture, __u32 id) {
  int i;
  for (i = 0; i < capture->v4l2_ctrl_count; i++) {
    if (id == capture->v4l2_ctrl_ranges[i]->ctrl_id) {
      return capture->v4l2_ctrl_ranges[i]->minimum;
    }
  }
  return -1;
}

static int v4l2_get_ctrl_max(CvCaptureCAM_V4L* capture, __u32 id) {
  int i;
  for (i = 0; i < capture->v4l2_ctrl_count; i++) {
    if (id == capture->v4l2_ctrl_ranges[i]->ctrl_id) {
      return capture->v4l2_ctrl_ranges[i]->maximum;
    }
  }
  return -1;
}


static void v4l2_scan_controls(CvCaptureCAM_V4L* capture) {

  __u32 ctrl_id;
  struct v4l2_control c;
  if (capture->v4l2_ctrl_ranges != NULL) {
    v4l2_free_ranges(capture);
  }
  capture->v4l2_ctrl_ranges = (v4l2_ctrl_range**)malloc(sizeof(v4l2_ctrl_range*));
#ifdef V4L2_CTRL_FLAG_NEXT_CTRL
  /* Try the extended control API first */
  capture->queryctrl.id      = V4L2_CTRL_FLAG_NEXT_CTRL;
  if(0 == v4l2_ioctl (capture->deviceHandle, VIDIOC_QUERYCTRL, &capture->queryctrl)) {
    do {
      c.id = capture->queryctrl.id;
      capture->queryctrl.id |= V4L2_CTRL_FLAG_NEXT_CTRL;
      if(capture->queryctrl.flags & V4L2_CTRL_FLAG_DISABLED) {
        continue;
      }
      if(capture->queryctrl.type != V4L2_CTRL_TYPE_INTEGER &&
         capture->queryctrl.type != V4L2_CTRL_TYPE_BOOLEAN &&
         capture->queryctrl.type != V4L2_CTRL_TYPE_MENU) {
        continue;
      }
      if(v4l2_ioctl(capture->deviceHandle, VIDIOC_G_CTRL, &c) == 0) {
        v4l2_add_ctrl_range(capture, &c);
      }

    } while(0 == v4l2_ioctl (capture->deviceHandle, VIDIOC_QUERYCTRL, &capture->queryctrl));
  } else
#endif
  {
    /* Check all the standard controls */
    for(ctrl_id=V4L2_CID_BASE; ctrl_id<V4L2_CID_LASTP1; ctrl_id++) {
      capture->queryctrl.id = ctrl_id;
      if(v4l2_ioctl(capture->deviceHandle, VIDIOC_QUERYCTRL, &capture->queryctrl) == 0) {
        if(capture->queryctrl.flags & V4L2_CTRL_FLAG_DISABLED) {
          continue;
        }
        if(capture->queryctrl.type != V4L2_CTRL_TYPE_INTEGER &&
           capture->queryctrl.type != V4L2_CTRL_TYPE_BOOLEAN &&
           capture->queryctrl.type != V4L2_CTRL_TYPE_MENU) {
          continue;
        }
        c.id = ctrl_id;

        if(v4l2_ioctl(capture->deviceHandle, VIDIOC_G_CTRL, &c) == 0) {
          v4l2_add_ctrl_range(capture, &c);
        }
      }
    }

    /* Check any custom controls */
    for(ctrl_id=V4L2_CID_PRIVATE_BASE; ; ctrl_id++) {
      capture->queryctrl.id = ctrl_id;
      if(v4l2_ioctl(capture->deviceHandle, VIDIOC_QUERYCTRL, &capture->queryctrl) == 0) {
        if(capture->queryctrl.flags & V4L2_CTRL_FLAG_DISABLED) {
          continue;
        }


        if(capture->queryctrl.type != V4L2_CTRL_TYPE_INTEGER &&
           capture->queryctrl.type != V4L2_CTRL_TYPE_BOOLEAN &&
           capture->queryctrl.type != V4L2_CTRL_TYPE_MENU) {
           continue;
        }

        c.id = ctrl_id;

        if(v4l2_ioctl(capture->deviceHandle, VIDIOC_G_CTRL, &c) == 0) {
          v4l2_add_ctrl_range(capture, &c);
        }
      } else {
        break;
      }
    }
  }
}

static inline int channels_for_mode(int mode)
{
    switch(mode) {
    case CV_CAP_MODE_GRAY:
        return 1;
    case CV_CAP_MODE_YUYV:
        return 2;
    default:
        return 3;
    }
}

static int _capture_V4L2 (CvCaptureCAM_V4L *capture, char *deviceName)
{
   int detect_v4l2 = 0;

   capture->deviceName = strdup(deviceName);

   detect_v4l2 = try_init_v4l2(capture, deviceName);

   if (detect_v4l2 != 1) {
       /* init of the v4l2 device is not OK */
       return -1;
   }

   /* starting from here, we assume we are in V4L2 mode */
   capture->is_v4l2_device = 1;

   capture->v4l2_ctrl_ranges = NULL;
   capture->v4l2_ctrl_count = 0;

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
       CLEAR (capture->inp);
       capture->inp.index = CHANNEL_NUMBER;
       /* Set only channel number to CHANNEL_NUMBER */
       /* V4L2 have a status field from selected video mode */
       if (-1 == xioctl (capture->deviceHandle, VIDIOC_ENUMINPUT, &capture->inp))
       {
         fprintf (stderr, "VIDEOIO ERROR: V4L2: Aren't able to set channel number\n");
         icvCloseCAM_V4L (capture);
         return -1;
       }
   } /* End if */

   /* Find Window info */
   CLEAR (capture->form);
   capture->form.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

   if (-1 == xioctl (capture->deviceHandle, VIDIOC_G_FMT, &capture->form)) {
       fprintf( stderr, "VIDEOIO ERROR: V4L2: Could not obtain specifics of capture window.\n\n");
       icvCloseCAM_V4L(capture);
       return -1;
   }

  /* libv4l will convert from any format to V4L2_PIX_FMT_BGR24,
     V4L2_PIX_FMT_RGV24, or V4L2_PIX_FMT_YUV420 */
  unsigned int requestedPixelFormat;
  switch (capture->mode) {
  case CV_CAP_MODE_RGB:
    requestedPixelFormat = V4L2_PIX_FMT_RGB24;
    break;
  case CV_CAP_MODE_GRAY:
    requestedPixelFormat = V4L2_PIX_FMT_YUV420;
    break;
  case CV_CAP_MODE_YUYV:
    requestedPixelFormat = V4L2_PIX_FMT_YUYV;
    break;
  default:
    requestedPixelFormat = V4L2_PIX_FMT_BGR24;
    break;
  }
  CLEAR (capture->form);
  capture->form.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  capture->form.fmt.pix.pixelformat = requestedPixelFormat;
  capture->form.fmt.pix.field       = V4L2_FIELD_ANY;
  capture->form.fmt.pix.width       = capture->width;
  capture->form.fmt.pix.height      = capture->height;

  if (-1 == xioctl (capture->deviceHandle, VIDIOC_S_FMT, &capture->form)) {
      fprintf(stderr, "VIDEOIO ERROR: libv4l unable to ioctl S_FMT\n");
      return -1;
  }

  if (requestedPixelFormat != capture->form.fmt.pix.pixelformat) {
      fprintf( stderr, "VIDEOIO ERROR: libv4l unable convert to requested pixfmt\n");
      return -1;
  }

   /* icvSetVideoSize(capture, DEFAULT_V4L_WIDTH, DEFAULT_V4L_HEIGHT); */

   unsigned int min;

   /* Buggy driver paranoia. */
   min = capture->form.fmt.pix.width * 2;

   if (capture->form.fmt.pix.bytesperline < min)
       capture->form.fmt.pix.bytesperline = min;

   min = capture->form.fmt.pix.bytesperline * capture->form.fmt.pix.height;

   if (capture->form.fmt.pix.sizeimage < min)
       capture->form.fmt.pix.sizeimage = min;

   CLEAR (capture->req);

   unsigned int buffer_number = DEFAULT_V4L_BUFFERS;

   try_again:

   capture->req.count = buffer_number;
   capture->req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
   capture->req.memory = V4L2_MEMORY_MMAP;

   if (-1 == xioctl (capture->deviceHandle, VIDIOC_REQBUFS, &capture->req))
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
       struct v4l2_buffer buf;

       CLEAR (buf);

       buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
       buf.memory = V4L2_MEMORY_MMAP;
       buf.index = n_buffers;

       if (-1 == xioctl (capture->deviceHandle, VIDIOC_QUERYBUF, &buf)) {
           perror ("VIDIOC_QUERYBUF");

           /* free capture, and returns an error code */
           icvCloseCAM_V4L (capture);
           return -1;
       }

       capture->buffers[n_buffers].length = buf.length;
       capture->buffers[n_buffers].start =
         v4l2_mmap (NULL /* start anywhere */,
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

#ifdef USE_TEMP_BUFFER
       if (n_buffers == 0) {
           if (capture->buffers[MAX_V4L_BUFFERS].start) {
               free(capture->buffers[MAX_V4L_BUFFERS].start);
               capture->buffers[MAX_V4L_BUFFERS].start = NULL;
       }

           capture->buffers[MAX_V4L_BUFFERS].start = malloc(buf.length);
           capture->buffers[MAX_V4L_BUFFERS].length = buf.length;
       };
#endif
   }

   /* Set up Image data */
   cvInitImageHeader( &capture->frame,
                      cvSize( capture->captureWindow.width,
                              capture->captureWindow.height ),
                      IPL_DEPTH_8U, channels_for_mode(capture->mode),
                      IPL_ORIGIN_TL, 4 );
   /* Allocate space for RGBA data */
   capture->frame.imageData = (char *)cvAlloc(capture->frame.imageSize);

   return 1;
}; /* End _capture_V4L2 */


static int _capture_V4L (CvCaptureCAM_V4L *capture, char *deviceName)
{
   int detect_v4l = 0;

   detect_v4l = try_init_v4l(capture, deviceName);

   if (detect_v4l == -1)
   {
     fprintf (stderr, "VIDEOIO ERROR: V4L"
              ": device %s: Unable to open for READ ONLY\n", deviceName);

     return -1;
   }

   if (detect_v4l <= 0)
   {
     fprintf (stderr, "VIDEOIO ERROR: V4L"
              ": device %s: Unable to query number of channels\n", deviceName);

     return -1;
   }

   {
     if ((capture->capability.type & VID_TYPE_CAPTURE) == 0) {
       /* Nope. */
       fprintf( stderr, "VIDEOIO ERROR: V4L: "
                "device %s is unable to capture video memory.\n",deviceName);
       icvCloseCAM_V4L(capture);
       return -1;
     }

   }


   /* The following code sets the CHANNEL_NUMBER of the video input.  Some video sources
   have sub "Channel Numbers".  For a typical V4L TV capture card, this is usually 1.
   I myself am using a simple NTSC video input capture card that uses the value of 1.
   If you are not in North America or have a different video standard, you WILL have to change
   the following settings and recompile/reinstall.  This set of settings is based on
   the most commonly encountered input video source types (like my bttv card) */

   {

     if(capture->capability.channels>0) {

       struct video_channel selectedChannel;

       selectedChannel.channel=CHANNEL_NUMBER;
       if (v4l1_ioctl(capture->deviceHandle, VIDIOCGCHAN , &selectedChannel) != -1) {
          /* set the video mode to ( VIDEO_MODE_PAL, VIDEO_MODE_NTSC, VIDEO_MODE_SECAM) */
//           selectedChannel.norm = VIDEO_MODE_NTSC;
          if (v4l1_ioctl(capture->deviceHandle, VIDIOCSCHAN , &selectedChannel) == -1) {
             /* Could not set selected channel - Oh well */
             //printf("\n%d, %s not NTSC capable.\n",selectedChannel.channel, selectedChannel.name);
          } /* End if */
       } /* End if */
     } /* End if */

   }

   {

     if(v4l1_ioctl(capture->deviceHandle, VIDIOCGWIN, &capture->captureWindow) == -1) {
       fprintf( stderr, "VIDEOIO ERROR: V4L: "
                "Could not obtain specifics of capture window.\n\n");
       icvCloseCAM_V4L(capture);
       return -1;
     }

   }

   {
      if(v4l1_ioctl(capture->deviceHandle, VIDIOCGPICT, &capture->imageProperties) < 0) {
         fprintf( stderr, "VIDEOIO ERROR: V4L: Unable to determine size of incoming image\n");
         icvCloseCAM_V4L(capture);
         return -1;
      }

      int requestedVideoPalette;
      int depth;
      switch (capture->mode) {
      case CV_CAP_MODE_GRAY:
        requestedVideoPalette = VIDEO_PALETTE_YUV420;
        depth = 8;
        break;
      case CV_CAP_MODE_YUYV:
        requestedVideoPalette = VIDEO_PALETTE_YUYV;
        depth = 16;
        break;
      default:
        requestedVideoPalette = VIDEO_PALETTE_RGB24;
        depth = 24;
        break;
      }
      capture->imageProperties.depth = depth;
      capture->imageProperties.palette = requestedVideoPalette;
      if (v4l1_ioctl(capture->deviceHandle, VIDIOCSPICT, &capture->imageProperties) < 0) {
        fprintf( stderr, "VIDEOIO ERROR: libv4l unable to ioctl VIDIOCSPICT\n\n");
        icvCloseCAM_V4L(capture);
        return -1;
      }
      if (v4l1_ioctl(capture->deviceHandle, VIDIOCGPICT, &capture->imageProperties) < 0) {
        fprintf( stderr, "VIDEOIO ERROR: libv4l unable to ioctl VIDIOCGPICT\n\n");
        icvCloseCAM_V4L(capture);
        return -1;
      }
      if (capture->imageProperties.palette != requestedVideoPalette) {
        fprintf( stderr, "VIDEOIO ERROR: libv4l unable convert to requested pixfmt\n\n");
        icvCloseCAM_V4L(capture);
        return -1;
      }

   }

   {

     v4l1_ioctl(capture->deviceHandle, VIDIOCGMBUF, &capture->memoryBuffer);
     capture->memoryMap  = (char *)v4l1_mmap(0,
                                   capture->memoryBuffer.size,
                                   PROT_READ | PROT_WRITE,
                                   MAP_SHARED,
                                   capture->deviceHandle,
                                   0);
     if (capture->memoryMap == MAP_FAILED) {
        fprintf( stderr, "VIDEOIO ERROR: V4L: Mapping Memmory from video source error: %s\n", strerror(errno));
        icvCloseCAM_V4L(capture);
        return -1;
     }

     /* Set up video_mmap structure pointing to this memory mapped area so each image may be
        retrieved from an index value */
     capture->mmaps = (struct video_mmap *)
                 (malloc(capture->memoryBuffer.frames * sizeof(struct video_mmap)));
     if (!capture->mmaps) {
        fprintf( stderr, "VIDEOIO ERROR: V4L: Could not memory map video frames.\n");
        icvCloseCAM_V4L(capture);
        return -1;
     }

   }

   /* Set up Image data */
   cvInitImageHeader( &capture->frame,
                      cvSize( capture->captureWindow.width,
                              capture->captureWindow.height ),
                      IPL_DEPTH_8U, channels_for_mode(capture->mode),
                      IPL_ORIGIN_TL, 4 );
   /* Allocate space for RGBA data */
   capture->frame.imageData = (char *)cvAlloc(capture->frame.imageSize);

   return 1;
}; /* End _capture_V4L */

static CvCaptureCAM_V4L * icvCaptureFromCAM_V4L (int index)
{
   static int autoindex;
   autoindex = 0;

   char deviceName[MAX_DEVICE_DRIVER_NAME];

   if (!numCameras)
      icvInitCapture_V4L(); /* Havent called icvInitCapture yet - do it now! */
   if (!numCameras)
     return NULL; /* Are there any /dev/video input sources? */

   //search index in indexList
   if ( (index>-1) && ! ((1 << index) & indexList) )
   {
     fprintf( stderr, "VIDEOIO ERROR: V4L: index %d is not correct!\n",index);
     return NULL; /* Did someone ask for not correct video source number? */
   }
   /* Allocate memory for this humongus CvCaptureCAM_V4L structure that contains ALL
      the handles for V4L processing */
   CvCaptureCAM_V4L * capture = (CvCaptureCAM_V4L*)cvAlloc(sizeof(CvCaptureCAM_V4L));
   if (!capture) {
      fprintf( stderr, "VIDEOIO ERROR: V4L: Could not allocate memory for capture process.\n");
      return NULL;
   }

#ifdef USE_TEMP_BUFFER
   capture->buffers[MAX_V4L_BUFFERS].start = NULL;
#endif

   /* Select camera, or rather, V4L video source */
   if (index<0) { // Asking for the first device available
     for (; autoindex<MAX_CAMERAS;autoindex++)
    if (indexList & (1<<autoindex))
        break;
     if (autoindex==MAX_CAMERAS)
    return NULL;
     index=autoindex;
     autoindex++;// i can recall icvOpenCAM_V4l with index=-1 for next camera
   }
   /* Print the CameraNumber at the end of the string with a width of one character */
   sprintf(deviceName, "/dev/video%1d", index);

   /* w/o memset some parts  arent initialized - AKA: Fill it with zeros so it is clean */
   memset(capture,0,sizeof(CvCaptureCAM_V4L));
   /* Present the routines needed for V4L funtionality.  They are inserted as part of
      the standard set of cv calls promoting transparency.  "Vector Table" insertion. */
   capture->FirstCapture = 1;

   /* set the default size */
   capture->width  = DEFAULT_V4L_WIDTH;
   capture->height = DEFAULT_V4L_HEIGHT;

   if (_capture_V4L2 (capture, deviceName) == -1) {
       icvCloseCAM_V4L(capture);
       capture->is_v4l2_device = 0;
       if (_capture_V4L (capture, deviceName) == -1) {
           icvCloseCAM_V4L(capture);
           return NULL;
       }
   } else {
       capture->is_v4l2_device = 1;
   }

   return capture;
}; /* End icvOpenCAM_V4L */

#ifdef HAVE_CAMV4L2

static int read_frame_v4l2(CvCaptureCAM_V4L* capture) {
    struct v4l2_buffer buf;

    CLEAR (buf);

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl (capture->deviceHandle, VIDIOC_DQBUF, &buf)) {
        switch (errno) {
        case EAGAIN:
            return 0;

        case EIO:
            /* Could ignore EIO, see spec. */

            /* fall through */

        default:
            /* display the error and stop processing */
            perror ("VIDIOC_DQBUF");
            return 1;
        }
   }

   assert(buf.index < capture->req.count);

#ifdef USE_TEMP_BUFFER
   memcpy(capture->buffers[MAX_V4L_BUFFERS].start,
    capture->buffers[buf.index].start,
    capture->buffers[MAX_V4L_BUFFERS].length );
   capture->bufferIndex = MAX_V4L_BUFFERS;
   //printf("got data in buff %d, len=%d, flags=0x%X, seq=%d, used=%d)\n",
   //   buf.index, buf.length, buf.flags, buf.sequence, buf.bytesused);
#else
   capture->bufferIndex = buf.index;
#endif

   if (-1 == xioctl (capture->deviceHandle, VIDIOC_QBUF, &buf))
       perror ("VIDIOC_QBUF");

   //set timestamp in capture struct to be timestamp of most recent frame
   /** where timestamps refer to the instant the field or frame was received by the driver, not the capture time*/
   capture->timestamp = buf.timestamp;   //printf( "timestamp update done \n");
   capture->sequence = buf.sequence;

   return 1;
}

static void mainloop_v4l2(CvCaptureCAM_V4L* capture) {
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

            if (read_frame_v4l2 (capture))
                break;
        }
    }
}

static int icvGrabFrameCAM_V4L(CvCaptureCAM_V4L* capture) {

   if (capture->FirstCapture) {
      /* Some general initialization must take place the first time through */

      /* This is just a technicality, but all buffers must be filled up before any
         staggered SYNC is applied.  SO, filler up. (see V4L HowTo) */

      if (capture->is_v4l2_device == 1)
      {

        for (capture->bufferIndex = 0;
             capture->bufferIndex < ((int)capture->req.count);
             ++capture->bufferIndex)
        {

          struct v4l2_buffer buf;

          CLEAR (buf);

          buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
          buf.memory      = V4L2_MEMORY_MMAP;
          buf.index       = (unsigned long)capture->bufferIndex;

          if (-1 == xioctl (capture->deviceHandle, VIDIOC_QBUF, &buf)) {
              perror ("VIDIOC_QBUF");
              return 0;
          }
        }

        /* enable the streaming */
        capture->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (-1 == xioctl (capture->deviceHandle, VIDIOC_STREAMON,
                          &capture->type)) {
            /* error enabling the stream */
            perror ("VIDIOC_STREAMON");
            return 0;
        }
      } else
      {

        for (capture->bufferIndex = 0;
         capture->bufferIndex < (capture->memoryBuffer.frames-1);
         ++capture->bufferIndex) {

          capture->mmaps[capture->bufferIndex].frame  = capture->bufferIndex;
          capture->mmaps[capture->bufferIndex].width  = capture->captureWindow.width;
          capture->mmaps[capture->bufferIndex].height = capture->captureWindow.height;
          capture->mmaps[capture->bufferIndex].format = capture->imageProperties.palette;

          if (v4l1_ioctl(capture->deviceHandle, VIDIOCMCAPTURE, &capture->mmaps[capture->bufferIndex]) == -1) {
            fprintf( stderr, "VIDEOIO ERROR: V4L: Initial Capture Error: Unable to load initial memory buffers.\n");
            return 0;
          }
        }

      }

      /* preparation is ok */
      capture->FirstCapture = 0;
   }

   if (capture->is_v4l2_device == 1)
   {

     mainloop_v4l2(capture);

   } else
   {

   capture->mmaps[capture->bufferIndex].frame  = capture->bufferIndex;
   capture->mmaps[capture->bufferIndex].width  = capture->captureWindow.width;
   capture->mmaps[capture->bufferIndex].height = capture->captureWindow.height;
   capture->mmaps[capture->bufferIndex].format = capture->imageProperties.palette;

   if (v4l1_ioctl (capture->deviceHandle, VIDIOCMCAPTURE,
           &capture->mmaps[capture->bufferIndex]) == -1) {
      /* capture is on the way, so just exit */
      return 1;
   }

     ++capture->bufferIndex;
     if (capture->bufferIndex == capture->memoryBuffer.frames) {
        capture->bufferIndex = 0;
     }

   }

   return(1);
}

static IplImage* icvRetrieveFrameCAM_V4L( CvCaptureCAM_V4L* capture, int) {

  if (capture->is_v4l2_device == 0)
  {

    /* [FD] this really belongs here */
    if (v4l1_ioctl(capture->deviceHandle, VIDIOCSYNC, &capture->mmaps[capture->bufferIndex].frame) == -1) {
      fprintf( stderr, "VIDEOIO ERROR: V4L: Could not SYNC to video stream. %s\n", strerror(errno));
    }

  }

   /* Now get what has already been captured as a IplImage return */

   /* First, reallocate imageData if the frame size changed */

  if (capture->is_v4l2_device == 1)
  {

    if(((unsigned long)capture->frame.width != capture->form.fmt.pix.width)
       || ((unsigned long)capture->frame.height != capture->form.fmt.pix.height)) {
        cvFree(&capture->frame.imageData);
        cvInitImageHeader( &capture->frame,
                           cvSize( capture->form.fmt.pix.width,
                                   capture->form.fmt.pix.height ),
                           IPL_DEPTH_8U, channels_for_mode(capture->mode),
                           IPL_ORIGIN_TL, 4 );
       capture->frame.imageData = (char *)cvAlloc(capture->frame.imageSize);
    }

  } else
  {

    if((capture->frame.width != capture->mmaps[capture->bufferIndex].width)
      || (capture->frame.height != capture->mmaps[capture->bufferIndex].height)) {
       cvFree(&capture->frame.imageData);
       cvInitImageHeader( &capture->frame,
                          cvSize( capture->captureWindow.width,
                                  capture->captureWindow.height ),
                          IPL_DEPTH_8U, channels_for_mode(capture->mode),
                          IPL_ORIGIN_TL, 4 );
       capture->frame.imageData = (char *)cvAlloc(capture->frame.imageSize);
    }

  }

  if (capture->is_v4l2_device == 1)
  {

    if(capture->buffers[capture->bufferIndex].start){
      memcpy((char *)capture->frame.imageData,
         (char *)capture->buffers[capture->bufferIndex].start,
         capture->frame.imageSize);
    }

  } else
#endif /* HAVE_CAMV4L2 */
  {

    switch(capture->imageProperties.palette) {
      case VIDEO_PALETTE_RGB24:
      case VIDEO_PALETTE_YUV420:
      case VIDEO_PALETTE_YUYV:
        memcpy((char *)capture->frame.imageData,
           (char *)(capture->memoryMap + capture->memoryBuffer.offsets[capture->bufferIndex]),
           capture->frame.imageSize);
        break;
      default:
        fprintf( stderr,
                 "VIDEOIO ERROR: V4L: Cannot convert from palette %d to mode %d\n",
                 capture->imageProperties.palette,
                 capture->mode);
        return 0;
    }

  }

   return(&capture->frame);
}

/* TODO: review this adaptation */
static double icvGetPropertyCAM_V4L (CvCaptureCAM_V4L* capture,
                                     int property_id ) {
  char name[32];
  int is_v4l2_device = 0;
      /* initialize the control structure */
  switch (property_id) {
    case CV_CAP_PROP_FRAME_WIDTH:
    case CV_CAP_PROP_FRAME_HEIGHT:
      CLEAR (capture->form);
      capture->form.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      if (-1 == xioctl (capture->deviceHandle, VIDIOC_G_FMT, &capture->form)) {
          /* display an error message, and return an error code */
          perror ("VIDIOC_G_FMT");
        if (v4l1_ioctl (capture->deviceHandle, VIDIOCGWIN, &capture->captureWindow) < 0) {
          fprintf (stderr, " ERROR: V4L: Unable to determine size of incoming image\n");
          icvCloseCAM_V4L(capture);
          return -1;
        } else {
          int retval = (property_id == CV_CAP_PROP_FRAME_WIDTH)?capture->captureWindow.width:capture->captureWindow.height;
          return retval / 0xFFFF;
        }
      }
      return (property_id == CV_CAP_PROP_FRAME_WIDTH)?capture->form.fmt.pix.width:capture->form.fmt.pix.height;

    case CV_CAP_PROP_POS_MSEC:
        if (capture->FirstCapture) {
            return 0;
        } else {
            //would be maximally numerically stable to cast to convert as bits, but would also be counterintuitive to decode
            return 1000 * capture->timestamp.tv_sec + ((double) capture->timestamp.tv_usec) / 1000;
        }
        break;

    case CV_CAP_PROP_POS_FRAMES:
        return capture->sequence;
        break;

    case CV_CAP_PROP_FPS: {
        struct v4l2_streamparm sp;
        memset (&sp, 0, sizeof(struct v4l2_streamparm));
        sp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl (capture->deviceHandle, VIDIOC_G_PARM, &sp) < 0){
            fprintf(stderr, "VIDEOIO ERROR: V4L: Unable to get camera FPS\n");
            return (double) -1;
        }

        // this is the captureable, not per say what you'll get..
        double framesPerSec = sp.parm.capture.timeperframe.denominator / (double)  sp.parm.capture.timeperframe.numerator ;
        return framesPerSec;
    }
    break;


    case CV_CAP_PROP_MODE:
      return capture->mode;
      break;
    case CV_CAP_PROP_BRIGHTNESS:
      sprintf(name, "Brightness");
      capture->control.id = V4L2_CID_BRIGHTNESS;
      break;
    case CV_CAP_PROP_CONTRAST:
      sprintf(name, "Contrast");
      capture->control.id = V4L2_CID_CONTRAST;
      break;
    case CV_CAP_PROP_SATURATION:
      sprintf(name, "Saturation");
      capture->control.id = V4L2_CID_SATURATION;
      break;
    case CV_CAP_PROP_HUE:
      sprintf(name, "Hue");
      capture->control.id = V4L2_CID_HUE;
      break;
    case CV_CAP_PROP_GAIN:
      sprintf(name, "Gain");
      capture->control.id = V4L2_CID_GAIN;
      break;
    case CV_CAP_PROP_EXPOSURE:
      sprintf(name, "Exposure");
      capture->control.id = V4L2_CID_EXPOSURE;
      break;
    case CV_CAP_PROP_FOCUS: {
      struct v4l2_control c;
      int v4l2_min;
      int v4l2_max;
      //we need to make sure that the autofocus is switch off, if available.
      capture->control.id = V4L2_CID_FOCUS_AUTO;
      v4l2_min = v4l2_get_ctrl_min(capture, capture->control.id);
      v4l2_max = v4l2_get_ctrl_max(capture, capture->control.id);
      if ( !((v4l2_min == -1) && (v4l2_max == -1)) ) {
        //autofocus capability is supported, switch it off.
        c.id    = capture->control.id;
        c.value = 0;//off
        if( v4l2_ioctl(capture->deviceHandle, VIDIOC_S_CTRL, &c) != 0 ){
          if (errno != ERANGE) {
            fprintf(stderr, "VIDEOIO ERROR: V4L2: Failed to set control \"%d\"(FOCUS_AUTO): %s (value %d)\n", c.id, strerror(errno), c.value);
            return -1;
          }
        }
      }//lack of support should not be considerred an error.

      sprintf(name, "Focus");
      capture->control.id = V4L2_CID_FOCUS_ABSOLUTE;
      break;
    }
    default:
      sprintf(name, "<unknown property string>");
      capture->control.id = property_id;
  }

  if(v4l2_ioctl(capture->deviceHandle, VIDIOC_G_CTRL, &capture->control) == 0) {
    /* all went well */
    is_v4l2_device = 1;
  } else {
    fprintf(stderr, "VIDEOIO ERROR: V4L2: Unable to get property %s(%u) - %s\n", name, capture->control.id, strerror(errno));
  }

  if (is_v4l2_device == 1) {
      /* get the min/max values */
      int v4l2_min = v4l2_get_ctrl_min(capture, capture->control.id);
      int v4l2_max = v4l2_get_ctrl_max(capture, capture->control.id);

      if ((v4l2_min == -1) && (v4l2_max == -1)) {
        fprintf(stderr, "VIDEOIO ERROR: V4L2: Property %s(%u) not supported by device\n", name, property_id);
        return -1;
      }

      /* all was OK, so convert to 0.0 - 1.0 range, and return the value */
      return ((float)capture->control.value - v4l2_min) / (v4l2_max - v4l2_min);

  } else {
    /* TODO: review this section */
    int retval = -1;

    switch (property_id) {
      case CV_CAP_PROP_BRIGHTNESS:
        retval = capture->imageProperties.brightness;
        break;
      case CV_CAP_PROP_CONTRAST:
        retval = capture->imageProperties.contrast;
        break;
      case CV_CAP_PROP_SATURATION:
        retval = capture->imageProperties.colour;
        break;
      case CV_CAP_PROP_HUE:
        retval = capture->imageProperties.hue;
        break;
      case CV_CAP_PROP_GAIN:
        fprintf(stderr, "VIDEOIO ERROR: V4L: Gain control in V4L is not supported\n");
        return -1;
        break;
      case CV_CAP_PROP_EXPOSURE:
        fprintf(stderr, "VIDEOIO ERROR: V4L: Exposure control in V4L is not supported\n");
        return -1;
        break;
    }

    if (retval == -1) {
      /* there was a problem */
      return -1;
    }
    /* all was OK, so convert to 0.0 - 1.0 range, and return the value */
    return float (retval) / 0xFFFF;
  }
}

static int icvSetVideoSize( CvCaptureCAM_V4L* capture, int w, int h) {

  if (capture->is_v4l2_device == 1)
  {
    char deviceName[MAX_DEVICE_DRIVER_NAME];
    sprintf(deviceName, "%s", capture->deviceName);
    icvCloseCAM_V4L(capture);
    _capture_V4L2(capture, deviceName);

    int cropHeight;
    int cropWidth;
    switch (capture->mode) {
    case CV_CAP_MODE_GRAY:
      cropHeight = h*8;
      cropWidth = w*8;
      break;
    case CV_CAP_MODE_YUYV:
      cropHeight = h*16;
      cropWidth = w*16;
      break;
    default:
      cropHeight = h*24;
      cropWidth = w*24;
      break;
    }
    CLEAR (capture->crop);
    capture->crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    capture->crop.c.left       = 0;
    capture->crop.c.top        = 0;
    capture->crop.c.height     = cropHeight;
    capture->crop.c.width      = cropWidth;

    /* set the crop area, but don't exit if the device don't support croping */
    xioctl (capture->deviceHandle, VIDIOC_S_CROP, &capture->crop);

    CLEAR (capture->form);
    capture->form.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    /* read the current setting, mainly to retreive the pixelformat information */
    xioctl (capture->deviceHandle, VIDIOC_G_FMT, &capture->form);

    /* set the values we want to change */
    capture->form.fmt.pix.width = w;
    capture->form.fmt.pix.height = h;
    capture->form.fmt.win.chromakey = 0;
    capture->form.fmt.win.field = V4L2_FIELD_ANY;
    capture->form.fmt.win.clips = 0;
    capture->form.fmt.win.clipcount = 0;
    capture->form.fmt.pix.field = V4L2_FIELD_ANY;

    /* ask the device to change the size
     * don't test if the set of the size is ok, because some device
     * don't allow changing the size, and we will get the real size
     * later */
    xioctl (capture->deviceHandle, VIDIOC_S_FMT, &capture->form);

    /* try to set framerate to 30 fps */

    struct v4l2_streamparm setfps;
    memset (&setfps, 0, sizeof(struct v4l2_streamparm));

    setfps.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    setfps.parm.capture.timeperframe.numerator = 1;
    setfps.parm.capture.timeperframe.denominator = 30;

    xioctl (capture->deviceHandle, VIDIOC_S_PARM, &setfps);


    /* we need to re-initialize some things, like buffers, because the size has
     * changed */
    capture->FirstCapture = 1;

    /* Get window info again, to get the real value */
    if (-1 == xioctl (capture->deviceHandle, VIDIOC_G_FMT, &capture->form))
    {
      fprintf(stderr, "VIDEOIO ERROR: V4L/V4L2: Could not obtain specifics of capture window.\n\n");

      icvCloseCAM_V4L(capture);

      return 0;
    }

    return 0;

  } else
  {

    if (capture==0) return 0;
     if (w>capture->capability.maxwidth) {
       w=capture->capability.maxwidth;
     }
     if (h>capture->capability.maxheight) {
       h=capture->capability.maxheight;
     }

     capture->captureWindow.width=w;
     capture->captureWindow.height=h;

     if (ioctl(capture->deviceHandle, VIDIOCSWIN, &capture->captureWindow) < 0) {
       icvCloseCAM_V4L(capture);
       return 0;
     }

     if (ioctl(capture->deviceHandle, VIDIOCGWIN, &capture->captureWindow) < 0) {
       icvCloseCAM_V4L(capture);
       return 0;
     }

     capture->FirstCapture = 1;

  }

  return 0;

}

static int icvSetControl (CvCaptureCAM_V4L* capture, int property_id, double value) {
  struct v4l2_control c;
  __s32 ctrl_value;
  char name[32];
  int is_v4l2  = 1;
  int v4l2_min = 0;
  int v4l2_max = 255;
  if (capture->v4l2_ctrl_ranges == NULL) {
    v4l2_scan_controls(capture);
  }

  CLEAR (capture->control);
  CLEAR (capture->queryctrl);

  /* get current values */
  switch (property_id) {
    case CV_CAP_PROP_BRIGHTNESS:
      sprintf(name, "Brightness");
      capture->control.id = V4L2_CID_BRIGHTNESS;
      break;
    case CV_CAP_PROP_CONTRAST:
      sprintf(name, "Contrast");
      capture->control.id = V4L2_CID_CONTRAST;
      break;
    case CV_CAP_PROP_SATURATION:
      sprintf(name, "Saturation");
      capture->control.id = V4L2_CID_SATURATION;
      break;
    case CV_CAP_PROP_HUE:
      sprintf(name, "Hue");
      capture->control.id = V4L2_CID_HUE;
      break;
    case CV_CAP_PROP_GAIN:
      sprintf(name, "Gain");
      capture->control.id = V4L2_CID_GAIN;
      break;
    case CV_CAP_PROP_EXPOSURE:
      sprintf(name, "Exposure");
      capture->control.id = V4L2_CID_EXPOSURE;
      break;
    case CV_CAP_PROP_FOCUS:
      //we need to make sure that the autofocus is switch off, if available.
      capture->control.id = V4L2_CID_FOCUS_AUTO;
      v4l2_min = v4l2_get_ctrl_min(capture, capture->control.id);
      v4l2_max = v4l2_get_ctrl_max(capture, capture->control.id);
      if ( !((v4l2_min == -1) && (v4l2_max == -1)) ) {
        //autofocus capability is supported, switch it off.
        c.id    = capture->control.id;
        c.value = 0;//off
        if( v4l2_ioctl(capture->deviceHandle, VIDIOC_S_CTRL, &c) != 0 ){
          if (errno != ERANGE) {
            fprintf(stderr, "VIDEOIO ERROR: V4L2: Failed to set control \"%d\"(FOCUS_AUTO): %s (value %d)\n", c.id, strerror(errno), c.value);
            return -1;
          }
        }
      }//lack of support should not be considerred an error.

      //now set the manual focus
      sprintf(name, "Focus");
      capture->control.id = V4L2_CID_FOCUS_ABSOLUTE;
      break;
    default:
      sprintf(name, "<unknown property string>");
      capture->control.id = property_id;
  }

  v4l2_min = v4l2_get_ctrl_min(capture, capture->control.id);
  v4l2_max = v4l2_get_ctrl_max(capture, capture->control.id);

  if ((v4l2_min == -1) && (v4l2_max == -1)) {
    fprintf(stderr, "VIDEOIO ERROR: V4L: Property %s(%u) not supported by device\n", name, property_id);
    return -1;
  }

  if(v4l2_ioctl(capture->deviceHandle, VIDIOC_G_CTRL, &capture->control) == 0) {
    /* all went well */
  } else {
    fprintf(stderr, "VIDEOIO ERROR: V4L2: Unable to get property %s(%u) - %s\n", name, capture->control.id, strerror(errno));
  }

  if (v4l2_max != 0) {
    double val = value;
    if (value < 0.0) {
      val = 0.0;
    } else if (value > 1.0) {
      val = 1.0;
    }
    ctrl_value = val * (double)(v4l2_max - v4l2_min) + v4l2_min;
  } else {
    ctrl_value = v4l2_get_ctrl_default(capture, capture->control.id) * (double)(v4l2_max - v4l2_min) + v4l2_min;
  }

  /* try and set value as if it was a v4l2 device */
  c.id    = capture->control.id;
  c.value = ctrl_value;
  if (v4l2_ioctl(capture->deviceHandle, VIDIOC_S_CTRL, &c) != 0) {
    /* The driver may clamp the value or return ERANGE, ignored here */
    if (errno != ERANGE) {
      fprintf(stderr, "VIDEOIO ERROR: V4L2: Failed to set control \"%d\": %s (value %d)\n", c.id, strerror(errno), c.value);
      is_v4l2 = 0;
    } else {
      return 0;
    }
  } else {
    return 0;
  }

  if (is_v4l2 == 0) { /* use v4l1_ioctl */
    fprintf(stderr, "VIDEOIO WARNING: Setting property %u through v4l2 failed. Trying with v4l1.\n", c.id);
    int v4l_value;
    /* scale the value to the wanted integer one */
    v4l_value = (int)(0xFFFF * value);

    switch (property_id) {
      case CV_CAP_PROP_BRIGHTNESS:
        capture->imageProperties.brightness = v4l_value;
        break;
      case CV_CAP_PROP_CONTRAST:
        capture->imageProperties.contrast = v4l_value;
        break;
      case CV_CAP_PROP_SATURATION:
        capture->imageProperties.colour = v4l_value;
        break;
      case CV_CAP_PROP_HUE:
        capture->imageProperties.hue = v4l_value;
        break;
      case CV_CAP_PROP_GAIN:
          fprintf(stderr, "VIDEOIO ERROR: V4L: Gain control in V4L is not supported\n");
        return -1;
    case CV_CAP_PROP_EXPOSURE:
        fprintf(stderr, "VIDEOIO ERROR: V4L: Exposure control in V4L is not supported\n");
        return -1;
    default:
        fprintf(stderr, "VIDEOIO ERROR: V4L: property #%d is not supported\n", property_id);
        return -1;
    }

    if (v4l1_ioctl(capture->deviceHandle, VIDIOCSPICT, &capture->imageProperties) < 0){
      fprintf(stderr, "VIDEOIO ERROR: V4L: Unable to set video informations\n");
      icvCloseCAM_V4L(capture);
      return -1;
    }
  }

  /* all was OK */
  return 0;
}

static int icvSetPropertyCAM_V4L(CvCaptureCAM_V4L* capture, int property_id, double value){
    static int width = 0, height = 0;
    int retval;

    /* initialization */
    retval = 0;

    /* two subsequent calls setting WIDTH and HEIGHT will change
       the video size */
    /* the first one will return an error, though. */

    switch (property_id) {
    case CV_CAP_PROP_FRAME_WIDTH:
        width = cvRound(value);
        capture->width = width;
        if(width !=0 && height != 0) {
            retval = icvSetVideoSize( capture, width, height);
            width = height = 0;
        }
        break;
    case CV_CAP_PROP_FRAME_HEIGHT:
        height = cvRound(value);
        capture->height = height;
        if(width !=0 && height != 0) {
            retval = icvSetVideoSize( capture, width, height);
            width = height = 0;
        }
        break;
    case CV_CAP_PROP_MODE:
        int mode;
        mode = cvRound(value);
        if (capture->mode != mode) {
            switch (mode) {
            case CV_CAP_MODE_BGR:
            case CV_CAP_MODE_RGB:
            case CV_CAP_MODE_GRAY:
            case CV_CAP_MODE_YUYV:
                capture->mode = mode;
                /* recreate the capture buffer for the same output resolution
                   but a different pixel format */
                retval = icvSetVideoSize(capture, capture->width, capture->height);
                break;
            default:
                fprintf(stderr, "VIDEOIO ERROR: V4L/V4L2: Unsupported mode: %d\n", mode);
                retval=0;
                break;
            }
        }
        break;
    case CV_CAP_PROP_FPS:
        struct v4l2_streamparm setfps;
        memset (&setfps, 0, sizeof(struct v4l2_streamparm));
        setfps.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        setfps.parm.capture.timeperframe.numerator = 1;
        setfps.parm.capture.timeperframe.denominator = value;
        if (xioctl (capture->deviceHandle, VIDIOC_S_PARM, &setfps) < 0){
            fprintf(stderr, "VIDEOIO ERROR: V4L: Unable to set camera FPS\n");
            retval=0;
        }
        break;
    default:
        retval = icvSetControl(capture, property_id, value);
    }

    /* return the the status */
    return retval;
}

static void icvCloseCAM_V4L( CvCaptureCAM_V4L* capture ){
   /* Deallocate space - Hopefully, no leaks */
   if (capture) {
     v4l2_free_ranges(capture);
     if (capture->is_v4l2_device == 0) {
       if (capture->mmaps) {
         free(capture->mmaps);
       }
       if (capture->memoryMap) {
         v4l1_munmap(capture->memoryMap, capture->memoryBuffer.size);
       }
       if (capture->deviceHandle != -1) {
         v4l1_close(capture->deviceHandle);
       }
     } else {
       capture->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
       if (xioctl(capture->deviceHandle, VIDIOC_STREAMOFF, &capture->type) < 0) {
         perror ("Unable to stop the stream.");
       }
       for (unsigned int n_buffers2 = 0; n_buffers2 < capture->req.count; ++n_buffers2) {
         if (-1 == v4l2_munmap (capture->buffers[n_buffers2].start, capture->buffers[n_buffers2].length)) {
           perror ("munmap");
         }
       }

       if (capture->deviceHandle != -1) {
         v4l2_close(capture->deviceHandle);
       }
     }

     if (capture->frame.imageData)
       cvFree(&capture->frame.imageData);

#ifdef USE_TEMP_BUFFER
     if (capture->buffers[MAX_V4L_BUFFERS].start) {
       free(capture->buffers[MAX_V4L_BUFFERS].start);
       capture->buffers[MAX_V4L_BUFFERS].start = NULL;
     }
#endif

     free(capture->deviceName);
     capture->deviceName = NULL;
     //v4l2_free_ranges(capture);
     //cvFree((void **)capture);
   }
};


class CvCaptureCAM_V4L_CPP : CvCapture
{
public:
    CvCaptureCAM_V4L_CPP() { captureV4L = 0; }
    virtual ~CvCaptureCAM_V4L_CPP() { close(); }

    virtual bool open( int index );
    virtual void close();

    virtual double getProperty(int) const;
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
protected:

    CvCaptureCAM_V4L* captureV4L;
};

bool CvCaptureCAM_V4L_CPP::open( int index )
{
    close();
    captureV4L = icvCaptureFromCAM_V4L(index);
    return captureV4L != 0;
}

void CvCaptureCAM_V4L_CPP::close()
{
    if( captureV4L )
    {
        icvCloseCAM_V4L( captureV4L );
        cvFree( &captureV4L );
    }
}

bool CvCaptureCAM_V4L_CPP::grabFrame()
{
    return captureV4L ? icvGrabFrameCAM_V4L( captureV4L ) != 0 : false;
}

IplImage* CvCaptureCAM_V4L_CPP::retrieveFrame(int)
{
    return captureV4L ? icvRetrieveFrameCAM_V4L( captureV4L, 0 ) : 0;
}

double CvCaptureCAM_V4L_CPP::getProperty( int propId ) const
{
    return captureV4L ? icvGetPropertyCAM_V4L( captureV4L, propId ) : 0.0;
}

bool CvCaptureCAM_V4L_CPP::setProperty( int propId, double value )
{
    return captureV4L ? icvSetPropertyCAM_V4L( captureV4L, propId, value ) != 0 : false;
}

CvCapture* cvCreateCameraCapture_V4L( int index )
{
    CvCaptureCAM_V4L_CPP* capture = new CvCaptureCAM_V4L_CPP;

    if( capture->open( index ))
        return (CvCapture*)capture;

    delete capture;
    return 0;
}

#endif
