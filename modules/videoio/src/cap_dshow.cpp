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

#if defined _WIN32 && defined HAVE_DSHOW
#include "cap_dshow.hpp"

/*
   DirectShow-based Video Capturing module is based on
   videoInput library by Theodore Watson:
   http://muonics.net/school/spring05/videoInput/

   Below is the original copyright
*/

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//THE SOFTWARE.

//////////////////////////////////////////////////////////
//Written by Theodore Watson - theo.watson@gmail.com    //
//Do whatever you want with this code but if you find   //
//a bug or make an improvement I would love to know!    //
//                                                      //
//Warning This code is experimental                     //
//use at your own risk :)                               //
//////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
/*                     Shoutouts

Thanks to:

           Dillip Kumar Kara for crossbar code.
           Zachary Lieberman for getting me into this stuff
           and for being so generous with time and code.
           The guys at Potion Design for helping me with VC++
           Josh Fisher for being a serious C++ nerd :)
           Golan Levin for helping me debug the strangest
           and slowest bug in the world!

           And all the people using this library who send in
           bugs, suggestions and improvements who keep me working on
           the next version - yeah thanks a lot ;)

*/
/////////////////////////////////////////////////////////

#if defined _MSC_VER && _MSC_VER >= 100
//'sprintf': name was marked as #pragma deprecated
#pragma warning(disable: 4995)
#endif

#ifdef __MINGW32__
// MinGW does not understand COM interfaces
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

#include <tchar.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <wchar.h>

#include <vector>

//Include Directshow stuff here so we don't worry about needing all the h files.
#define NO_DSHOW_STRSAFE
#include "DShow.h"
#include "strmif.h"
#include "Aviriff.h"
#include "dvdmedia.h"
#include "bdaiface.h"

//for threading
#include <process.h>

//this is for TryEnterCriticalSection
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x400
#endif


/*
MEDIASUBTYPE_I420 : TGUID ='{30323449-0000-0010-8000-00AA00389B71}';
MEDIASUBTYPE_Y800 : TGUID ='{30303859-0000-0010-8000-00AA00389B71}';
MEDIASUBTYPE_Y8   : TGUID ='{20203859-0000-0010-8000-00AA00389B71}';
MEDIASUBTYPE_Y160 : TGUID ='{30363159-0000-0010-8000-00AA00389B71}';
MEDIASUBTYPE_YV16 : TGUID ='{32315659-0000-0010-8000-00AA00389B71}';
MEDIASUBTYPE_Y422 : TGUID ='{32323459-0000-0010-8000-00AA00389B71}';
MEDIASUBTYPE_GREY : TGUID ='{59455247-0000-0010-8000-00AA00389B71}';
*/

#include <initguid.h>

DEFINE_GUID(MEDIASUBTYPE_GREY, 0x59455247, 0x0000, 0x0010, 0x80, 0x00,
    0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);
DEFINE_GUID(MEDIASUBTYPE_Y8, 0x20203859, 0x0000, 0x0010, 0x80, 0x00,
    0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);
DEFINE_GUID(MEDIASUBTYPE_Y800, 0x30303859, 0x0000, 0x0010, 0x80, 0x00,
    0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);
DEFINE_GUID(MEDIASUBTYPE_Y16, 0x20363159, 0x0000, 0x0010, 0x80, 0x00,
    0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);
DEFINE_GUID(MEDIASUBTYPE_BY8, 0x20385942, 0x0000, 0x0010, 0x80, 0x00,
    0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

DEFINE_GUID(CLSID_CaptureGraphBuilder2,0xbf87b6e1,0x8c27,0x11d0,0xb3,0xf0,0x00,0xaa,0x00,0x37,0x61,0xc5);
DEFINE_GUID(CLSID_FilterGraph,0xe436ebb3,0x524f,0x11ce,0x9f,0x53,0x00,0x20,0xaf,0x0b,0xa7,0x70);
DEFINE_GUID(CLSID_NullRenderer,0xc1f400a4,0x3f08,0x11d3,0x9f,0x0b,0x00,0x60,0x08,0x03,0x9e,0x37);
DEFINE_GUID(CLSID_SampleGrabber,0xc1f400a0,0x3f08,0x11d3,0x9f,0x0b,0x00,0x60,0x08,0x03,0x9e,0x37);
DEFINE_GUID(CLSID_SystemDeviceEnum,0x62be5d10,0x60eb,0x11d0,0xbd,0x3b,0x00,0xa0,0xc9,0x11,0xce,0x86);
DEFINE_GUID(CLSID_VideoInputDeviceCategory,0x860bb310,0x5d01,0x11d0,0xbd,0x3b,0x00,0xa0,0xc9,0x11,0xce,0x86);
DEFINE_GUID(FORMAT_VideoInfo,0x05589f80,0xc356,0x11ce,0xbf,0x01,0x00,0xaa,0x00,0x55,0x59,0x5a);
DEFINE_GUID(IID_IAMAnalogVideoDecoder,0xc6e13350,0x30ac,0x11d0,0xa1,0x8c,0x00,0xa0,0xc9,0x11,0x89,0x56);
DEFINE_GUID(IID_IAMCameraControl,0xc6e13370,0x30ac,0x11d0,0xa1,0x8c,0x00,0xa0,0xc9,0x11,0x89,0x56);
DEFINE_GUID(IID_IAMCrossbar,0xc6e13380,0x30ac,0x11d0,0xa1,0x8c,0x00,0xa0,0xc9,0x11,0x89,0x56);
DEFINE_GUID(IID_IAMStreamConfig,0xc6e13340,0x30ac,0x11d0,0xa1,0x8c,0x00,0xa0,0xc9,0x11,0x89,0x56);
DEFINE_GUID(IID_IAMVideoProcAmp,0xc6e13360,0x30ac,0x11d0,0xa1,0x8c,0x00,0xa0,0xc9,0x11,0x89,0x56);
DEFINE_GUID(IID_IBaseFilter,0x56a86895,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
DEFINE_GUID(IID_ICaptureGraphBuilder2,0x93e5a4e0,0x2d50,0x11d2,0xab,0xfa,0x00,0xa0,0xc9,0xc6,0xe3,0x8d);
DEFINE_GUID(IID_ICreateDevEnum,0x29840822,0x5b84,0x11d0,0xbd,0x3b,0x00,0xa0,0xc9,0x11,0xce,0x86);
DEFINE_GUID(IID_IGraphBuilder,0x56a868a9,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
DEFINE_GUID(IID_IMPEG2PIDMap,0xafb6c2a1,0x2c41,0x11d3,0x8a,0x60,0x00,0x00,0xf8,0x1e,0x0e,0x4a);
DEFINE_GUID(IID_IMediaControl,0x56a868b1,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
DEFINE_GUID(IID_IMediaEventEx, 0x56a868c0,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
DEFINE_GUID(IID_IMediaFilter,0x56a86899,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
DEFINE_GUID(IID_ISampleGrabber,0x6b652fff,0x11fe,0x4fce,0x92,0xad,0x02,0x66,0xb5,0xd7,0xc7,0x8f);
DEFINE_GUID(LOOK_UPSTREAM_ONLY,0xac798be0,0x98e3,0x11d1,0xb3,0xf1,0x00,0xaa,0x00,0x37,0x61,0xc5);
DEFINE_GUID(MEDIASUBTYPE_AYUV,0x56555941,0x0000,0x0010,0x80,0x00,0x00,0xaa,0x00,0x38,0x9b,0x71);
DEFINE_GUID(MEDIASUBTYPE_IYUV,0x56555949,0x0000,0x0010,0x80,0x00,0x00,0xaa,0x00,0x38,0x9b,0x71);
DEFINE_GUID(MEDIASUBTYPE_RGB24,0xe436eb7d,0x524f,0x11ce,0x9f,0x53,0x00,0x20,0xaf,0x0b,0xa7,0x70);
DEFINE_GUID(MEDIASUBTYPE_RGB32,0xe436eb7e,0x524f,0x11ce,0x9f,0x53,0x00,0x20,0xaf,0x0b,0xa7,0x70);
DEFINE_GUID(MEDIASUBTYPE_RGB555,0xe436eb7c,0x524f,0x11ce,0x9f,0x53,0x00,0x20,0xaf,0x0b,0xa7,0x70);
DEFINE_GUID(MEDIASUBTYPE_RGB565,0xe436eb7b,0x524f,0x11ce,0x9f,0x53,0x00,0x20,0xaf,0x0b,0xa7,0x70);
DEFINE_GUID(MEDIASUBTYPE_I420,0x30323449,0x0000,0x0010,0x80,0x00,0x00,0xaa,0x00,0x38,0x9b,0x71);
DEFINE_GUID(MEDIASUBTYPE_UYVY,0x59565955,0x0000,0x0010,0x80,0x00,0x00,0xaa,0x00,0x38,0x9b,0x71);
DEFINE_GUID(MEDIASUBTYPE_Y211,0x31313259,0x0000,0x0010,0x80,0x00,0x00,0xaa,0x00,0x38,0x9b,0x71);
DEFINE_GUID(MEDIASUBTYPE_Y411,0x31313459,0x0000,0x0010,0x80,0x00,0x00,0xaa,0x00,0x38,0x9b,0x71);
DEFINE_GUID(MEDIASUBTYPE_Y41P,0x50313459,0x0000,0x0010,0x80,0x00,0x00,0xaa,0x00,0x38,0x9b,0x71);
DEFINE_GUID(MEDIASUBTYPE_YUY2,0x32595559,0x0000,0x0010,0x80,0x00,0x00,0xaa,0x00,0x38,0x9b,0x71);
DEFINE_GUID(MEDIASUBTYPE_YUYV,0x56595559,0x0000,0x0010,0x80,0x00,0x00,0xaa,0x00,0x38,0x9b,0x71);
DEFINE_GUID(MEDIASUBTYPE_YV12,0x32315659,0x0000,0x0010,0x80,0x00,0x00,0xaa,0x00,0x38,0x9b,0x71);
DEFINE_GUID(MEDIASUBTYPE_YVU9,0x39555659,0x0000,0x0010,0x80,0x00,0x00,0xaa,0x00,0x38,0x9b,0x71);
DEFINE_GUID(MEDIASUBTYPE_YVYU,0x55595659,0x0000,0x0010,0x80,0x00,0x00,0xaa,0x00,0x38,0x9b,0x71);
DEFINE_GUID(MEDIASUBTYPE_MJPG,0x47504A4D, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71); // MGB
DEFINE_GUID(MEDIATYPE_Interleaved,0x73766169,0x0000,0x0010,0x80,0x00,0x00,0xaa,0x00,0x38,0x9b,0x71);
DEFINE_GUID(MEDIATYPE_Video,0x73646976,0x0000,0x0010,0x80,0x00,0x00,0xaa,0x00,0x38,0x9b,0x71);
DEFINE_GUID(PIN_CATEGORY_CAPTURE,0xfb6c4281,0x0353,0x11d1,0x90,0x5f,0x00,0x00,0xc0,0xcc,0x16,0xba);
DEFINE_GUID(PIN_CATEGORY_PREVIEW,0xfb6c4282,0x0353,0x11d1,0x90,0x5f,0x00,0x00,0xc0,0xcc,0x16,0xba);

interface ISampleGrabberCB : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE SampleCB(
        double SampleTime,
        IMediaSample *pSample) = 0;

    virtual HRESULT STDMETHODCALLTYPE BufferCB(
        double SampleTime,
        BYTE *pBuffer,
        LONG BufferLen) = 0;
};

interface ISampleGrabber : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE SetOneShot(
        BOOL OneShot) = 0;

    virtual HRESULT STDMETHODCALLTYPE SetMediaType(
        const AM_MEDIA_TYPE *pType) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetConnectedMediaType(
        AM_MEDIA_TYPE *pType) = 0;

    virtual HRESULT STDMETHODCALLTYPE SetBufferSamples(
        BOOL BufferThem) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetCurrentBuffer(
        LONG *pBufferSize,
        LONG *pBuffer) = 0;

    virtual HRESULT STDMETHODCALLTYPE GetCurrentSample(
        IMediaSample **ppSample) = 0;

    virtual HRESULT STDMETHODCALLTYPE SetCallback(
        ISampleGrabberCB *pCallback,
        LONG WhichMethodToCallback) = 0;
};

#ifndef HEADER
#define HEADER(p) (&(((VIDEOINFOHEADER*)(p))->bmiHeader))
#endif

//Example Usage
/*
    //create a videoInput object
    videoInput VI;

    //Prints out a list of available devices and returns num of devices found
    int numDevices = VI.listDevices();

    int device1 = 0;  //this could be any deviceID that shows up in listDevices
    int device2 = 1;  //this could be any deviceID that shows up in listDevices

    //if you want to capture at a different frame rate (default is 30)
    //specify it here, you are not guaranteed to get this fps though.
    //VI.setIdealFramerate(dev, 60);

    //setup the first device - there are a number of options:

    VI.setupDevice(device1);                           //setup the first device with the default settings
    //VI.setupDevice(device1, VI_COMPOSITE);           //or setup device with specific connection type
    //VI.setupDevice(device1, 320, 240);               //or setup device with specified video size
    //VI.setupDevice(device1, 320, 240, VI_COMPOSITE); //or setup device with video size and connection type

    //VI.setFormat(device1, VI_NTSC_M);                //if your card doesn't remember what format it should be
                                                       //call this with the appropriate format listed above
                                                       //NOTE: must be called after setupDevice!

    //optionally setup a second (or third, fourth ...) device - same options as above
    VI.setupDevice(device2);

    //As requested width and height can not always be accommodated
    //make sure to check the size once the device is setup

    int width   = VI.getWidth(device1);
    int height  = VI.getHeight(device1);
    int size    = VI.getSize(device1);

    unsigned char * yourBuffer1 = new unsigned char[size];
    unsigned char * yourBuffer2 = new unsigned char[size];

    //to get the data from the device first check if the data is new
    if(VI.isFrameNew(device1)){
        VI.getPixels(device1, yourBuffer1, false, false);   //fills pixels as a BGR (for openCV) unsigned char array - no flipping
        VI.getPixels(device1, yourBuffer2, true, true);     //fills pixels as a RGB (for openGL) unsigned char array - flipping!
    }

    //same applies to device2 etc

    //to get a settings dialog for the device
    VI.showSettingsWindow(device1);


    //Shut down devices properly
    VI.stopDevice(device1);
    VI.stopDevice(device2);
*/


//////////////////////////////////////   VARS AND DEFS   //////////////////////////////////


//STUFF YOU CAN CHANGE

#ifdef _DEBUG
#include <strsafe.h>

static void DebugPrintOut(const char *format, ...)
{
    static int gs_verbose = -1;
    if (gs_verbose < 0)
    {
        // Fetch initial debug state from environment - defaults to disabled
        const char* s = getenv("OPENCV_DSHOW_DEBUG");
        gs_verbose = s != NULL && atoi(s) != 0;
    }


    if (gs_verbose)
    {
        va_list args;
        va_start(args, format);
        if( ::IsDebuggerPresent() )
        {
            CHAR szMsg[512];
            ::StringCbVPrintfA(szMsg, sizeof(szMsg), format, args);
            ::OutputDebugStringA(szMsg);
        }
        else
        {
            vprintf(format, args);
        }
        va_end (args);
    }
}
#else
#define DebugPrintOut(...) void()
#endif

//if you need VI to use multi threaded com
//#define VI_COM_MULTI_THREADED

//STUFF YOU DON'T CHANGE

//videoInput defines
#define VI_VERSION      0.1995
#define VI_MAX_CAMERAS  20
#define VI_NUM_TYPES    22 //MGB
#define VI_NUM_FORMATS  18 //DON'T TOUCH

//defines for setPhyCon - tuner is not as well supported as composite and s-video
#define VI_COMPOSITE 0
#define VI_S_VIDEO   1
#define VI_TUNER     2
#define VI_USB       3
#define VI_1394      4

//defines for formats
#define VI_NTSC_M   0
#define VI_PAL_B    1
#define VI_PAL_D    2
#define VI_PAL_G    3
#define VI_PAL_H    4
#define VI_PAL_I    5
#define VI_PAL_M    6
#define VI_PAL_N    7
#define VI_PAL_NC   8
#define VI_SECAM_B  9
#define VI_SECAM_D  10
#define VI_SECAM_G  11
#define VI_SECAM_H  12
#define VI_SECAM_K  13
#define VI_SECAM_K1 14
#define VI_SECAM_L  15
#define VI_NTSC_M_J 16
#define VI_NTSC_433 17


//allows us to directShow classes here with the includes in the cpp
struct ICaptureGraphBuilder2;
struct IGraphBuilder;
struct IBaseFilter;
struct IAMCrossbar;
struct IMediaControl;
struct ISampleGrabber;
struct IMediaEventEx;
struct IAMStreamConfig;
struct _AMMediaType;
class SampleGrabberCallback;
typedef _AMMediaType AM_MEDIA_TYPE;

//keeps track of how many instances of VI are being used
//don't touch
//static int comInitCount = 0;


////////////////////////////////////////   VIDEO DEVICE   ///////////////////////////////////

class videoDevice{


    public:

        videoDevice();
        void setSize(int w, int h);
        void NukeDownstream(IBaseFilter *pBF);
        void destroyGraph();
        ~videoDevice();

        int videoSize;
        int width;
        int height;

        int tryWidth;
        int tryHeight;
        GUID tryVideoType;

        ICaptureGraphBuilder2 *pCaptureGraph;    // Capture graph builder object
        IGraphBuilder *pGraph;                    // Graph builder object
        IMediaControl *pControl;                // Media control object
        IBaseFilter *pVideoInputFilter;          // Video Capture filter
        IBaseFilter *pGrabberF;
        IBaseFilter * pDestFilter;
        IAMStreamConfig *streamConf;
        ISampleGrabber * pGrabber;                // Grabs frame
        AM_MEDIA_TYPE * pAmMediaType;

        IMediaEventEx * pMediaEvent;

        GUID videoType;
        long formatType;

        SampleGrabberCallback * sgCallback;

        bool tryDiffSize;
        bool useCrossbar;
        bool readyToCapture;
        bool sizeSet;
        bool setupStarted;
        bool specificFormat;
        bool autoReconnect;
        bool convertRGB;
        int  nFramesForReconnect;
        unsigned long nFramesRunning;
        int  connection;
        int  storeConn;
        int  myID;
        long requestedFrameTime; //ie fps

        LONG volatile property_window_count;

        char  nDeviceName[255];
        WCHAR wDeviceName[255];

        unsigned char * pixels;
        char * pBuffer;

};

//////////////////////////////////////   VIDEO INPUT   /////////////////////////////////////
class videoInput{

    public:
        videoInput();
        ~videoInput();

        //Functions in rough order they should be used.
        static int listDevices(bool silent = false);

        //needs to be called after listDevices - otherwise returns NULL
        static char * getDeviceName(int deviceID);

        //choose to use callback based capture - or single threaded
        void setUseCallback(bool useCallback);

        //call before setupDevice
        //directshow will try and get the closest possible framerate to what is requested
        void setIdealFramerate(int deviceID, int idealFramerate);

        //some devices will stop delivering frames after a while - this method gives you the option to try and reconnect
        //to a device if videoInput detects that a device has stopped delivering frames.
        //you MUST CALL isFrameNew every app loop for this to have any effect
        void setAutoReconnectOnFreeze(int deviceNumber, bool doReconnect, int numMissedFramesBeforeReconnect);

        //Choose one of these five to setup your device
        bool setupDevice(int deviceID);
        bool setupDevice(int deviceID, int w, int h);
        bool setupDeviceFourcc(int deviceID, int w, int h,int fourcc);

        //These two are only for capture cards
        //USB and Firewire cameras shouldn't specify connection
        bool setupDevice(int deviceID, int connection);
        bool setupDevice(int deviceID, int w, int h, int connection);

        bool setFourcc(int deviceNumber, int fourcc);

        //If you need to you can set your NTSC/PAL/SECAM
        //preference here. if it is available it will be used.
        //see #defines above for available formats - eg VI_NTSC_M or VI_PAL_B
        //should be called after setupDevice
        //can be called multiple times
        bool setFormat(int deviceNumber, int format);

        //Tells you when a new frame has arrived - you should call this if you have specified setAutoReconnectOnFreeze to true
        bool isFrameNew(int deviceID);

        bool isDeviceSetup(int deviceID) const;

        //Returns the pixels - flipRedAndBlue toggles RGB/BGR flipping - and you can flip the image too
        unsigned char * getPixels(int deviceID, bool flipRedAndBlue = true, bool flipImage = false);

        //Or pass in a buffer for getPixels to fill returns true if successful.
        bool getPixels(int id, unsigned char * pixels, bool flipRedAndBlue = true, bool flipImage = false);

        //Launches a pop up settings window
        //For some reason in GLUT you have to call it twice each time.
        bool showSettingsWindow(int deviceID);

        //Manual control over settings thanks.....
        //These are experimental for now.
        bool setVideoSettingFilter(int deviceID, long Property, long lValue, long Flags = 0, bool useDefaultValue = false);
        bool setVideoSettingFilterPct(int deviceID, long Property, float pctValue, long Flags = 0);
        bool getVideoSettingFilter(int deviceID, long Property, long &min, long &max, long &SteppingDelta, long &currentValue, long &flags, long &defaultValue);

        bool setVideoSettingCamera(int deviceID, long Property, long lValue, long Flags = 0, bool useDefaultValue = false);
        bool setVideoSettingCameraPct(int deviceID, long Property, float pctValue, long Flags = 0);
        bool getVideoSettingCamera(int deviceID, long Property, long &min, long &max, long &SteppingDelta, long &currentValue, long &flags, long &defaultValue);

        //bool setVideoSettingCam(int deviceID, long Property, long lValue, long Flags = NULL, bool useDefaultValue = false);

        //get width, height and number of pixels
        int  getWidth(int deviceID) const;
        int  getHeight(int deviceID) const;
        int  getSize(int deviceID) const;
        int  getFourcc(int deviceID) const;
        double getFPS(int deviceID) const;

        int getChannel(int deviceID) const;

        // RGB conversion setting
        bool getConvertRGB(int deviceID);
        bool setConvertRGB(int deviceID, bool enable);

        //completely stops and frees a device
        void stopDevice(int deviceID);

        //as above but then sets it up with same settings
        bool restartDevice(int deviceID);

        //number of devices available
        int  devicesFound;

        // mapping from OpenCV CV_CAP_PROP to videoinput/dshow properties
        int getVideoPropertyFromCV(int cv_property);
        int getCameraPropertyFromCV(int cv_property);

        bool isDeviceDisconnected(int deviceID);

        int property_window_count(int device_idx);

        GUID getMediasubtype(int deviceID);

    private:
        void setPhyCon(int deviceID, int conn);
        void setAttemptCaptureSize(int deviceID, int w, int h,GUID mediaType=MEDIASUBTYPE_RGB24);
        bool setup(int deviceID);
        void processPixels(unsigned char * src, unsigned char * dst, int width, int height, bool bRGB, bool bFlip, int bytesperpixel = 3);
        int  start(int deviceID, videoDevice * VD);
        int  getDeviceCount();
        void getMediaSubtypeAsString(GUID type, char * typeAsString);
        GUID *getMediaSubtypeFromFourcc(int fourcc);
        int   getFourccFromMediaSubtype(GUID type) const;

        void getVideoPropertyAsString(int prop, char * propertyAsString);
        void getCameraPropertyAsString(int prop, char * propertyAsString);

        HRESULT getDevice(IBaseFilter **pSrcFilter, int deviceID, WCHAR * wDeviceName, char * nDeviceName);
        static HRESULT ShowFilterPropertyPages(IBaseFilter *pFilter);
        static HRESULT ShowStreamPropertyPages(IAMStreamConfig  *pStream);

        HRESULT SaveGraphFile(IGraphBuilder *pGraph, WCHAR *wszPath);
        HRESULT routeCrossbar(ICaptureGraphBuilder2 **ppBuild, IBaseFilter **pVidInFilter, int conType, GUID captureMode);

        //don't touch
        static bool comInit();
        static bool comUnInit();

        int  connection;
        int  callbackSetCount;
        bool bCallback;

        GUID CAPTURE_MODE;

        //Extra video subtypes
        // GUID MEDIASUBTYPE_Y800;
        // GUID MEDIASUBTYPE_Y8;
        // GUID MEDIASUBTYPE_GREY;

        videoDevice * VDList[VI_MAX_CAMERAS];
        GUID mediaSubtypes[VI_NUM_TYPES];
        long formatTypes[VI_NUM_FORMATS];

        static void __cdecl basicThread(void * objPtr);

        static char deviceNames[VI_MAX_CAMERAS][255];
};

///////////////////////////  HANDY FUNCTIONS  /////////////////////////////

//Included by e-con
//Checks whether the current formattype is single byte format
//Eg: MEDIASUBTYPE_Y800, MEDIASUBTYPE_Y8, MEDIASUBTYPE_GREY
static bool checkSingleByteFormat(GUID formatType)
{

    if (formatType == MEDIASUBTYPE_Y800 ||
        formatType == MEDIASUBTYPE_Y8 ||
        formatType == MEDIASUBTYPE_GREY)
    {
        return true;
    }
    else
    {
        return false;
    }
}

static void MyFreeMediaType(AM_MEDIA_TYPE& mt){
    if (mt.cbFormat != 0)
    {
        CoTaskMemFree((PVOID)mt.pbFormat);
        mt.cbFormat = 0;
        mt.pbFormat = NULL;
    }
    if (mt.pUnk != NULL)
    {
        // Unnecessary because pUnk should not be used, but safest.
        mt.pUnk->Release();
        mt.pUnk = NULL;
    }
}

static void MyDeleteMediaType(AM_MEDIA_TYPE *pmt)
{
    if (pmt != NULL)
    {
        MyFreeMediaType(*pmt);
        CoTaskMemFree(pmt);
    }
}

//////////////////////////////  CALLBACK  ////////////////////////////////

//Callback class
class SampleGrabberCallback : public ISampleGrabberCB{
public:

    //------------------------------------------------
    SampleGrabberCallback(){
        InitializeCriticalSection(&critSection);
        freezeCheck = 0;


        bufferSetup         = false;
        newFrame            = false;
        latestBufferLength  = 0;

        hEvent = CreateEvent(NULL, true, false, NULL);
        pixels = 0;
        ptrBuffer = 0;
        numBytes = 0;
    }


    //------------------------------------------------
    virtual ~SampleGrabberCallback(){
        ptrBuffer = NULL;
        DeleteCriticalSection(&critSection);
        CloseHandle(hEvent);
        if(bufferSetup){
            delete[] pixels;
        }
    }


    //------------------------------------------------
    bool setupBuffer(int numBytesIn){
        if(bufferSetup){
            return false;
        }else{
            numBytes            = numBytesIn;
            pixels              = new unsigned char[numBytes];
            bufferSetup         = true;
            newFrame            = false;
            latestBufferLength  = 0;
        }
        return true;
    }


    //------------------------------------------------
    STDMETHODIMP_(ULONG) AddRef() { return 1; }
    STDMETHODIMP_(ULONG) Release() { return 2; }


    //------------------------------------------------
    STDMETHODIMP QueryInterface(REFIID, void **ppvObject){
        *ppvObject = static_cast<ISampleGrabberCB*>(this);
        return S_OK;
    }


    //This method is meant to have less overhead
    //------------------------------------------------
    STDMETHODIMP SampleCB(double , IMediaSample *pSample){
        if(WaitForSingleObject(hEvent, 0) == WAIT_OBJECT_0) return S_OK;

        HRESULT hr = pSample->GetPointer(&ptrBuffer);

        if(hr == S_OK){
            latestBufferLength = pSample->GetActualDataLength();
              if(latestBufferLength == numBytes){
                EnterCriticalSection(&critSection);
                      memcpy(pixels, ptrBuffer, latestBufferLength);
                    newFrame    = true;
                    freezeCheck = 1;
                LeaveCriticalSection(&critSection);
                SetEvent(hEvent);
            }else{
                DebugPrintOut("ERROR: SampleCB() - buffer sizes do not match\n");
            }
        }

        return S_OK;
    }


    //This method is meant to have more overhead
    STDMETHODIMP BufferCB(double, BYTE *, long){
        return E_NOTIMPL;
    }

    int freezeCheck;

    int latestBufferLength;
    int numBytes;
    bool newFrame;
    bool bufferSetup;
    unsigned char * pixels;
    unsigned char * ptrBuffer;
    CRITICAL_SECTION critSection;
    HANDLE hEvent;
};


//////////////////////////////  VIDEO DEVICE  ////////////////////////////////

// ----------------------------------------------------------------------
//    Should this class also be the callback?
//
// ----------------------------------------------------------------------

videoDevice::videoDevice(){

     pCaptureGraph      = NULL;    // Capture graph builder object
     pGraph             = NULL;    // Graph builder object
     pControl           = NULL;    // Media control object
     pVideoInputFilter  = NULL; // Video Capture filter
     pGrabber           = NULL; // Grabs frame
     pDestFilter        = NULL; // Null Renderer Filter
     pGrabberF          = NULL; // Grabber Filter
     pMediaEvent        = NULL;
     streamConf         = NULL;
     pAmMediaType       = NULL;

     //This is our callback class that processes the frame.
     sgCallback           = new SampleGrabberCallback();
     sgCallback->newFrame = false;

     //Default values for capture type
     videoType          = MEDIASUBTYPE_RGB24;
     connection         = PhysConn_Video_Composite;
     storeConn          = 0;

     videoSize          = 0;
     width              = 0;
     height             = 0;

     tryWidth           = 640;
     tryHeight          = 480;
     tryVideoType = MEDIASUBTYPE_RGB24;
     nFramesForReconnect= 10000;
     nFramesRunning     = 0;
     myID               = -1;

     tryDiffSize        = true;
     useCrossbar        = false;
     readyToCapture     = false;
     sizeSet            = false;
     setupStarted       = false;
     specificFormat     = false;
     autoReconnect      = false;
     convertRGB         = true;
     requestedFrameTime = -1;

     pBuffer = 0;
     pixels = 0;
     formatType = 0;

     property_window_count = 0;

     memset(wDeviceName, 0, sizeof(WCHAR) * 255);
     memset(nDeviceName, 0, sizeof(char) * 255);

}


// ----------------------------------------------------------------------
//    The only place we are doing new
//
// ----------------------------------------------------------------------

void videoDevice::setSize(int w, int h){
    if(sizeSet){
        DebugPrintOut("SETUP: Error device size should not be set more than once\n");
    }
    else
    {
        width               = w;
        height              = h;

        if (checkSingleByteFormat(pAmMediaType->subtype))
        {
            videoSize      = w * h;
        }
        else if (pAmMediaType->subtype == MEDIASUBTYPE_Y16)
        {
            videoSize      = w * h * 2;
        }
        else
        {
            videoSize      = w * h * 3;
        }

        sizeSet             = true;
        pixels              = new unsigned char[videoSize];
        pBuffer             = new char[videoSize];

        memset(pixels, 0 , videoSize);
        sgCallback->setupBuffer(videoSize);

    }
}


// ----------------------------------------------------------------------
//    Borrowed from the SDK, use it to take apart the graph from
//  the capture device downstream to the null renderer
// ----------------------------------------------------------------------

void videoDevice::NukeDownstream(IBaseFilter *pBF){
    IPin *pP, *pTo;
    ULONG u;
    IEnumPins *pins = NULL;
    PIN_INFO pininfo;
    HRESULT hr = pBF->EnumPins(&pins);
    if (hr != S_OK || !pins)
        return;
    pins->Reset();
    while (hr == NOERROR)
    {
        hr = pins->Next(1, &pP, &u);
        if (hr == S_OK && pP)
        {
            pP->ConnectedTo(&pTo);
            if (pTo)
            {
                hr = pTo->QueryPinInfo(&pininfo);
                if (hr == NOERROR)
                {
                    if (pininfo.dir == PINDIR_INPUT)
                    {
                        NukeDownstream(pininfo.pFilter);
                        pGraph->Disconnect(pTo);
                        pGraph->Disconnect(pP);
                        pGraph->RemoveFilter(pininfo.pFilter);
                    }
                    pininfo.pFilter->Release();
                    pininfo.pFilter = NULL;
                }
                pTo->Release();
            }
            pP->Release();
        }
    }
    pins->Release();
}


// ----------------------------------------------------------------------
//    Also from SDK
// ----------------------------------------------------------------------

void videoDevice::destroyGraph(){
    HRESULT hr = 0;

    int i = 0;
    while (hr == NOERROR)
    {
        IEnumFilters * pEnum = 0;
        ULONG cFetched = 0;

        // We must get the enumerator again every time because removing a filter from the graph
        // invalidates the enumerator. We always get only the first filter from each enumerator.
        hr = pGraph->EnumFilters(&pEnum);
        if (FAILED(hr)) { DebugPrintOut("SETUP: pGraph->EnumFilters() failed.\n"); return; }

        IBaseFilter * pFilter = NULL;
        if (pEnum->Next(1, &pFilter, &cFetched) == S_OK)
        {
            FILTER_INFO FilterInfo;
            memset(&FilterInfo, 0, sizeof(FilterInfo));
            hr = pFilter->QueryFilterInfo(&FilterInfo);
            FilterInfo.pGraph->Release();

            int count = 0;
            char buffer[255];
            memset(buffer, 0, 255 * sizeof(char));

            while( FilterInfo.achName[count] != 0x00 )
            {
                buffer[count] = (char)FilterInfo.achName[count];
                count++;
            }

            DebugPrintOut("SETUP: removing filter %s...\n", buffer);
            hr = pGraph->RemoveFilter(pFilter);
            if (FAILED(hr)) { DebugPrintOut("SETUP: pGraph->RemoveFilter() failed.\n"); return; }
            DebugPrintOut("SETUP: filter removed %s\n",buffer);

            pFilter->Release();
            pFilter = NULL;
        }
        pEnum->Release();
        pEnum = NULL;

        if (cFetched == 0)
            break;
        i++;
    }

 return;
}


// ----------------------------------------------------------------------
// Our deconstructor, attempts to tear down graph and release filters etc
// Does checking to make sure it only is freeing if it needs to
// Probably could be a lot cleaner! :)
// ----------------------------------------------------------------------

videoDevice::~videoDevice(){

    if(setupStarted){ DebugPrintOut("\nSETUP: Disconnecting device %i\n", myID); }
    else{
        if(sgCallback){
            sgCallback->Release();
            delete sgCallback;
        }
        return;
    }

    HRESULT HR = NOERROR;

    //Check to see if the graph is running, if so stop it.
    if( (pControl) )
    {
        HR = pControl->Pause();
        if (FAILED(HR)) DebugPrintOut("ERROR - Could not pause pControl\n");

        HR = pControl->Stop();
        if (FAILED(HR)) DebugPrintOut("ERROR - Could not stop pControl\n");
    }

    //Stop the callback and free it
    if( (sgCallback) && (pGrabber) )
    {
        pGrabber->SetCallback(NULL, 1);
        DebugPrintOut("SETUP: freeing Grabber Callback\n");
        sgCallback->Release();

        //delete our pixels
        if(sizeSet){
             delete[] pixels;
             delete[] pBuffer;
        }

        delete sgCallback;
    }

    //Disconnect filters from capture device
    if( (pVideoInputFilter) )NukeDownstream(pVideoInputFilter);

    //Release and zero pointers to our filters etc
    if( (pDestFilter) ){        DebugPrintOut("SETUP: freeing Renderer\n");
                                (pDestFilter)->Release();
                                (pDestFilter) = 0;
    }
    if( (pVideoInputFilter) ){  DebugPrintOut("SETUP: freeing Capture Source\n");
                                (pVideoInputFilter)->Release();
                                (pVideoInputFilter) = 0;
    }
    if( (pGrabberF) ){          DebugPrintOut("SETUP: freeing Grabber Filter\n");
                                (pGrabberF)->Release();
                                (pGrabberF) = 0;
    }
    if( (pGrabber) ){           DebugPrintOut("SETUP: freeing Grabber\n");
                                (pGrabber)->Release();
                                (pGrabber) = 0;
    }
    if( (pControl) ){           DebugPrintOut("SETUP: freeing Control\n");
                                (pControl)->Release();
                                (pControl) = 0;
    }
    if( (pMediaEvent) ){        DebugPrintOut("SETUP: freeing Media Event\n");
                                (pMediaEvent)->Release();
                                (pMediaEvent) = 0;
    }
    if( (streamConf) ){         DebugPrintOut("SETUP: freeing Stream\n");
                                (streamConf)->Release();
                                (streamConf) = 0;
    }

    if( (pAmMediaType) ){       DebugPrintOut("SETUP: freeing Media Type\n");
                                MyDeleteMediaType(pAmMediaType);
    }

    if((pMediaEvent)){
            DebugPrintOut("SETUP: freeing Media Event\n");
            (pMediaEvent)->Release();
            (pMediaEvent) = 0;
    }

    //Destroy the graph
    if( (pGraph) )destroyGraph();

    //Release and zero our capture graph and our main graph
    if( (pCaptureGraph) ){      DebugPrintOut("SETUP: freeing Capture Graph\n");
                                (pCaptureGraph)->Release();
                                (pCaptureGraph) = 0;
    }
    if( (pGraph) ){             DebugPrintOut("SETUP: freeing Main Graph\n");
                                (pGraph)->Release();
                                (pGraph) = 0;
    }

    DebugPrintOut("SETUP: Device %i disconnected and freed\n\n",myID);
}


//////////////////////////////  VIDEO INPUT  ////////////////////////////////
////////////////////////////  PUBLIC METHODS  ///////////////////////////////


// ----------------------------------------------------------------------
// Constructor - creates instances of videoDevice and adds the various
// media subtypes to check.
// ----------------------------------------------------------------------

videoInput::videoInput(){
    //start com
    comInit();

    devicesFound         = 0;
    callbackSetCount     = 0;
    bCallback            = true;

    connection = PhysConn_Video_Composite;
    CAPTURE_MODE = PIN_CATEGORY_PREVIEW;

    //setup a max no of device objects
    for(int i=0; i<VI_MAX_CAMERAS; i++)  VDList[i] = new videoDevice();

    DebugPrintOut("\n***** VIDEOINPUT LIBRARY - %2.04f - TFW07 *****\n\n",VI_VERSION);

    //added for the pixelink firewire camera
    // MEDIASUBTYPE_Y800 = (GUID)FOURCCMap(FCC('Y800'));
    // MEDIASUBTYPE_Y8   = (GUID)FOURCCMap(FCC('Y8'));
    // MEDIASUBTYPE_GREY = (GUID)FOURCCMap(FCC('GREY'));

    //The video types we support
    //in order of preference

    mediaSubtypes[0]     = MEDIASUBTYPE_RGB24;
    mediaSubtypes[1]     = MEDIASUBTYPE_RGB32;
    mediaSubtypes[2]     = MEDIASUBTYPE_RGB555;
    mediaSubtypes[3]     = MEDIASUBTYPE_RGB565;
    mediaSubtypes[4]     = MEDIASUBTYPE_YUY2;
    mediaSubtypes[5]     = MEDIASUBTYPE_YVYU;
    mediaSubtypes[6]     = MEDIASUBTYPE_YUYV;
    mediaSubtypes[7]     = MEDIASUBTYPE_IYUV;
    mediaSubtypes[8]     = MEDIASUBTYPE_UYVY;
    mediaSubtypes[9]     = MEDIASUBTYPE_YV12;
    mediaSubtypes[10]    = MEDIASUBTYPE_YVU9;
    mediaSubtypes[11]    = MEDIASUBTYPE_Y411;
    mediaSubtypes[12]    = MEDIASUBTYPE_Y41P;
    mediaSubtypes[13]    = MEDIASUBTYPE_Y211;
    mediaSubtypes[14]    = MEDIASUBTYPE_AYUV;
    mediaSubtypes[15]    = MEDIASUBTYPE_MJPG; // MGB

    //non standard
    mediaSubtypes[16]    = MEDIASUBTYPE_Y800;
    mediaSubtypes[17]    = MEDIASUBTYPE_Y8;
    mediaSubtypes[18]    = MEDIASUBTYPE_GREY;
    mediaSubtypes[19]    = MEDIASUBTYPE_I420;
    mediaSubtypes[20] = MEDIASUBTYPE_BY8;
    mediaSubtypes[21] = MEDIASUBTYPE_Y16;

    //The video formats we support
    formatTypes[VI_NTSC_M]      = AnalogVideo_NTSC_M;
    formatTypes[VI_NTSC_M_J]    = AnalogVideo_NTSC_M_J;
    formatTypes[VI_NTSC_433]    = AnalogVideo_NTSC_433;

    formatTypes[VI_PAL_B]       = AnalogVideo_PAL_B;
    formatTypes[VI_PAL_D]       = AnalogVideo_PAL_D;
    formatTypes[VI_PAL_G]       = AnalogVideo_PAL_G;
    formatTypes[VI_PAL_H]       = AnalogVideo_PAL_H;
    formatTypes[VI_PAL_I]       = AnalogVideo_PAL_I;
    formatTypes[VI_PAL_M]       = AnalogVideo_PAL_M;
    formatTypes[VI_PAL_N]       = AnalogVideo_PAL_N;
    formatTypes[VI_PAL_NC]      = AnalogVideo_PAL_N_COMBO;

    formatTypes[VI_SECAM_B]     = AnalogVideo_SECAM_B;
    formatTypes[VI_SECAM_D]     = AnalogVideo_SECAM_D;
    formatTypes[VI_SECAM_G]     = AnalogVideo_SECAM_G;
    formatTypes[VI_SECAM_H]     = AnalogVideo_SECAM_H;
    formatTypes[VI_SECAM_K]     = AnalogVideo_SECAM_K;
    formatTypes[VI_SECAM_K1]    = AnalogVideo_SECAM_K1;
    formatTypes[VI_SECAM_L]     = AnalogVideo_SECAM_L;

}

// ----------------------------------------------------------------------
// change to use callback or regular capture
// callback tells you when a new frame has arrived
// but non-callback won't - but is single threaded
// ----------------------------------------------------------------------
void videoInput::setUseCallback(bool useCallback){
    if(callbackSetCount == 0){
        bCallback = useCallback;
        callbackSetCount = 1;
    }else{
        DebugPrintOut("ERROR: setUseCallback can only be called before setup\n");
    }
}

// ----------------------------------------------------------------------
// Set the requested framerate - no guarantee you will get this
//
// ----------------------------------------------------------------------

void videoInput::setIdealFramerate(int deviceNumber, int idealFramerate){
    if(deviceNumber >= VI_MAX_CAMERAS || VDList[deviceNumber]->readyToCapture) return;

    if( idealFramerate > 0 ){
        VDList[deviceNumber]->requestedFrameTime = (unsigned long)(10000000 / idealFramerate);
    }
}


// ----------------------------------------------------------------------
// Set the requested framerate - no guarantee you will get this
//
// ----------------------------------------------------------------------

void videoInput::setAutoReconnectOnFreeze(int deviceNumber, bool doReconnect, int numMissedFramesBeforeReconnect){
    if(deviceNumber >= VI_MAX_CAMERAS) return;

    VDList[deviceNumber]->autoReconnect            = doReconnect;
    VDList[deviceNumber]->nFramesForReconnect    = numMissedFramesBeforeReconnect;

}


// ----------------------------------------------------------------------
// Setup a device with the default settings
//
// ----------------------------------------------------------------------

bool videoInput::setupDevice(int deviceNumber){
    if(deviceNumber >= VI_MAX_CAMERAS || VDList[deviceNumber]->readyToCapture) return false;

    if(setup(deviceNumber))return true;
    return false;
}


// ----------------------------------------------------------------------
// Setup a device with the default size but specify input type
//
// ----------------------------------------------------------------------

bool videoInput::setupDevice(int deviceNumber, int _connection){
    if(deviceNumber >= VI_MAX_CAMERAS || VDList[deviceNumber]->readyToCapture) return false;

    setPhyCon(deviceNumber, _connection);
    if(setup(deviceNumber))return true;
    return false;
}


// ----------------------------------------------------------------------
// Setup a device with the default connection but specify size
//
// ----------------------------------------------------------------------

bool videoInput::setupDevice(int deviceNumber, int w, int h){
    if(deviceNumber >= VI_MAX_CAMERAS || VDList[deviceNumber]->readyToCapture) return false;

    setAttemptCaptureSize(deviceNumber,w,h);
    if(setup(deviceNumber))return true;
    return false;
}

// ----------------------------------------------------------------------
// Setup a device with the default connection but specify size and image format
//
// Note:
// Need a new name for this since signature clashes with ",int connection)"
// ----------------------------------------------------------------------

bool videoInput::setupDeviceFourcc(int deviceNumber, int w, int h,int fourcc){
    if(deviceNumber >= VI_MAX_CAMERAS || VDList[deviceNumber]->readyToCapture) return false;

    if ( fourcc != -1 ) {
        GUID *mediaType = getMediaSubtypeFromFourcc(fourcc);
        if ( mediaType ) {
            setAttemptCaptureSize(deviceNumber,w,h,*mediaType);
        } else {
            DebugPrintOut("SETUP: Unknown GUID \n");
            return false;
        }
    } else {
        setAttemptCaptureSize(deviceNumber,w,h);
    }
    if(setup(deviceNumber))return true;
    return false;
}


// ----------------------------------------------------------------------
// Setup a device with specific size and connection
//
// ----------------------------------------------------------------------

bool videoInput::setupDevice(int deviceNumber, int w, int h, int _connection){
    if(deviceNumber >= VI_MAX_CAMERAS || VDList[deviceNumber]->readyToCapture) return false;

    setAttemptCaptureSize(deviceNumber,w,h);
    setPhyCon(deviceNumber, _connection);
    if(setup(deviceNumber))return true;
    return false;
}


// ----------------------------------------------------------------------
// Setup the default video format of the device
// Must be called after setup!
// See #define formats in header file (eg VI_NTSC_M )
//
// ----------------------------------------------------------------------

bool videoInput::setFormat(int deviceNumber, int format){
    if(deviceNumber >= VI_MAX_CAMERAS || !VDList[deviceNumber]->readyToCapture) return false;

    bool returnVal = false;

    if(format >= 0 && format < VI_NUM_FORMATS){
        VDList[deviceNumber]->formatType = formatTypes[format];
        VDList[deviceNumber]->specificFormat = true;

        if(VDList[deviceNumber]->specificFormat){

            HRESULT hr = getDevice(&VDList[deviceNumber]->pVideoInputFilter, deviceNumber, VDList[deviceNumber]->wDeviceName, VDList[deviceNumber]->nDeviceName);
            if(hr != S_OK){
                return false;
            }

            IAMAnalogVideoDecoder *pVideoDec = NULL;
               hr = VDList[deviceNumber]->pCaptureGraph->FindInterface(NULL, &MEDIATYPE_Video, VDList[deviceNumber]->pVideoInputFilter, IID_IAMAnalogVideoDecoder, (void **)&pVideoDec);


            //in case the settings window some how freed them first
            if(VDList[deviceNumber]->pVideoInputFilter)VDList[deviceNumber]->pVideoInputFilter->Release();
            if(VDList[deviceNumber]->pVideoInputFilter)VDList[deviceNumber]->pVideoInputFilter = NULL;


            if(FAILED(hr)){
                DebugPrintOut("SETUP: couldn't set requested format\n");
            }else{
                long lValue = 0;
                hr = pVideoDec->get_AvailableTVFormats(&lValue);
                if( SUCCEEDED(hr) && (lValue & VDList[deviceNumber]->formatType) )
                   {
                       hr = pVideoDec->put_TVFormat(VDList[deviceNumber]->formatType);
                    if( FAILED(hr) ){
                        DebugPrintOut("SETUP: couldn't set requested format\n");
                    }else{
                        returnVal = true;
                    }
                   }

                pVideoDec->Release();
                pVideoDec = NULL;
            }
        }
    }

    return returnVal;
}

// ----------------------------------------------------------------------
// Our static function for returning device names - thanks Peter!
// Must call listDevices first.
//
// ----------------------------------------------------------------------
char videoInput::deviceNames[VI_MAX_CAMERAS][255]={{0}};

char * videoInput::getDeviceName(int deviceID){
    if( deviceID >= VI_MAX_CAMERAS ){
        return NULL;
    }
    return deviceNames[deviceID];
}


// ----------------------------------------------------------------------
// Our static function for finding num devices available etc
//
// ----------------------------------------------------------------------

int videoInput::listDevices(bool silent){

    //COM Library Initialization
    comInit();

    if(!silent) DebugPrintOut("\nVIDEOINPUT SPY MODE!\n\n");


    ICreateDevEnum *pDevEnum = NULL;
    IEnumMoniker *pEnum = NULL;
    int deviceCounter = 0;

    HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL,
        CLSCTX_INPROC_SERVER, IID_ICreateDevEnum,
        reinterpret_cast<void**>(&pDevEnum));


    if (SUCCEEDED(hr))
    {
        // Create an enumerator for the video capture category.
        hr = pDevEnum->CreateClassEnumerator(
            CLSID_VideoInputDeviceCategory,
            &pEnum, 0);

       if(hr == S_OK){

            if(!silent) DebugPrintOut("SETUP: Looking For Capture Devices\n");
            IMoniker *pMoniker = NULL;

            while (pEnum->Next(1, &pMoniker, NULL) == S_OK){

                IPropertyBag *pPropBag;
                hr = pMoniker->BindToStorage(0, 0, IID_IPropertyBag,
                    (void**)(&pPropBag));

                if (FAILED(hr)){
                    pMoniker->Release();
                    continue;  // Skip this one, maybe the next one will work.
                }


                 // Find the description or friendly name.
                VARIANT varName;
                VariantInit(&varName);
                hr = pPropBag->Read(L"Description", &varName, 0);

                if (FAILED(hr)) hr = pPropBag->Read(L"FriendlyName", &varName, 0);

                if (SUCCEEDED(hr)){

                    hr = pPropBag->Read(L"FriendlyName", &varName, 0);

                    int count = 0;
                    int maxLen = sizeof(deviceNames[0])/sizeof(deviceNames[0][0]) - 2;
                    while( varName.bstrVal[count] != 0x00 && count < maxLen) {
                        deviceNames[deviceCounter][count] = (char)varName.bstrVal[count];
                        count++;
                    }
                    deviceNames[deviceCounter][count] = 0;

                    if(!silent) DebugPrintOut("SETUP: %i) %s\n",deviceCounter, deviceNames[deviceCounter]);
                }

                pPropBag->Release();
                pPropBag = NULL;

                pMoniker->Release();
                pMoniker = NULL;

                deviceCounter++;
            }

            pDevEnum->Release();
            pDevEnum = NULL;

            pEnum->Release();
            pEnum = NULL;
        }

         if(!silent) DebugPrintOut("SETUP: %i Device(s) found\n\n", deviceCounter);
    }

    comUnInit();

    return deviceCounter;
}


// ----------------------------------------------------------------------
//
//
// ----------------------------------------------------------------------

int videoInput::getWidth(int id) const
{
    if(isDeviceSetup(id))
    {
        return VDList[id] ->width;
    }

    return 0;

}


// ----------------------------------------------------------------------
//
//
// ----------------------------------------------------------------------

int videoInput::getHeight(int id) const
{
    if(isDeviceSetup(id))
    {
        return VDList[id] ->height;
    }

    return 0;

}

// ----------------------------------------------------------------------
//
//
// ----------------------------------------------------------------------
int videoInput::getFourcc(int id) const
{
    if(isDeviceSetup(id))
    {
        return getFourccFromMediaSubtype(VDList[id]->videoType);
    }

    return 0;

}

double videoInput::getFPS(int id) const
{
    if(isDeviceSetup(id))
    {
        double frameTime= VDList[id]->requestedFrameTime;
        if (frameTime>0) {
            return (10000000.0 / frameTime);
        }
    }

    return 0;

}

int videoInput::getChannel(int deviceID) const
{
    if (!isDeviceSetup(deviceID))
        return 0;
    return VDList[deviceID]->storeConn;
}

// ----------------------------------------------------------------------
//
//
// ----------------------------------------------------------------------

int videoInput::getSize(int id) const
{
    if(isDeviceSetup(id))
    {
        return VDList[id] ->videoSize;
    }

    return 0;

}

// ----------------------------------------------------------------------
//
//
// ----------------------------------------------------------------------

bool videoInput::getConvertRGB(int id)
{
    if (isDeviceSetup(id))
    {
        return VDList[id]->convertRGB;
    }
    else
    {
        return false;
    }

}

bool videoInput::setConvertRGB(int id, bool enable)
{
    if (isDeviceSetup(id))
    {
        VDList[id]->convertRGB = enable;
        return true;
    }
    else
    {
        return false;
    }
}


// ----------------------------------------------------------------------
// Uses a supplied buffer
// ----------------------------------------------------------------------

bool videoInput::getPixels(int id, unsigned char * dstBuffer, bool flipRedAndBlue, bool flipImage){

    bool success = false;

    if(isDeviceSetup(id)){
        if(bCallback){
            //callback capture

            DWORD result = WaitForSingleObject(VDList[id]->sgCallback->hEvent, 1000);
            if( result != WAIT_OBJECT_0) return false;

            //double paranoia - mutexing with both event and critical section
            EnterCriticalSection(&VDList[id]->sgCallback->critSection);

                unsigned char * src = VDList[id]->sgCallback->pixels;
                unsigned char * dst = dstBuffer;
                int height             = VDList[id]->height;
                int width              = VDList[id]->width;

                // Conditional processing for 8/16-bit images (e-Con systems)
                if (checkSingleByteFormat(VDList[id]->pAmMediaType->subtype))
                {
                    memcpy(dst, src, width * height);
                }
                else if (VDList[id]->pAmMediaType->subtype == MEDIASUBTYPE_Y16)
                {
                    if (!VDList[id]->convertRGB) {
                        memcpy(dst, src, width * height * 2);
                    }
                    else {
                        processPixels(src, dst, width, height, flipRedAndBlue, flipImage, 2);
                    }
                }
                else
                {
                    processPixels(src, dst, width, height, flipRedAndBlue, flipImage);
                }
                VDList[id]->sgCallback->newFrame = false;

            LeaveCriticalSection(&VDList[id]->sgCallback->critSection);

            ResetEvent(VDList[id]->sgCallback->hEvent);

            success = true;

        }
        else{
            //regular capture method
            long bufferSize = VDList[id]->videoSize;
            HRESULT hr = VDList[id]->pGrabber->GetCurrentBuffer(&bufferSize, (long *)VDList[id]->pBuffer);
            if(hr==S_OK){
                int numBytes = VDList[id]->videoSize;
                if (numBytes == bufferSize){

                    unsigned char * src = (unsigned char * )VDList[id]->pBuffer;
                    unsigned char * dst = dstBuffer;
                    int height             = VDList[id]->height;
                    int width             = VDList[id]->width;

                    processPixels(src, dst, width, height, flipRedAndBlue, flipImage);
                    success = true;
                }else{
                    DebugPrintOut("ERROR: GetPixels() - bufferSizes do not match!\n");
                }
            }else{
                DebugPrintOut("ERROR: GetPixels() - Unable to grab frame for device %i\n", id);
            }
        }
    }

    return success;
}


// ----------------------------------------------------------------------
// Returns a buffer
// ----------------------------------------------------------------------
unsigned char * videoInput::getPixels(int id, bool flipRedAndBlue, bool flipImage){

    if(isDeviceSetup(id)){
           getPixels(id, VDList[id]->pixels, flipRedAndBlue, flipImage);
    }

    return VDList[id]->pixels;
}



// ----------------------------------------------------------------------
//
//
// ----------------------------------------------------------------------
bool videoInput::isFrameNew(int id){
    if(!isDeviceSetup(id)) return false;
    if(!bCallback)return true;

    bool result = false;
    bool freeze = false;

    //again super paranoia!
    EnterCriticalSection(&VDList[id]->sgCallback->critSection);
        result = VDList[id]->sgCallback->newFrame;

        //we need to give it some time at the beginning to start up so lets check after 400 frames
        if(VDList[id]->nFramesRunning > 400 && VDList[id]->sgCallback->freezeCheck > VDList[id]->nFramesForReconnect ){
            freeze = true;
        }

        //we increment the freezeCheck var here - the callback resets it to 1
        //so as long as the callback is running this var should never get too high.
        //if the callback is not running then this number will get high and trigger the freeze action below
        VDList[id]->sgCallback->freezeCheck++;
    LeaveCriticalSection(&VDList[id]->sgCallback->critSection);

    VDList[id]->nFramesRunning++;

    if(freeze && VDList[id]->autoReconnect){
        DebugPrintOut("ERROR: Device seems frozen - attempting to reconnect\n");
        if( !restartDevice(VDList[id]->myID) ){
            DebugPrintOut("ERROR: Unable to reconnect to device\n");
        }else{
            DebugPrintOut("SUCCESS: Able to reconnect to device\n");
        }
    }

    return result;
}


// ----------------------------------------------------------------------
//
//
// ----------------------------------------------------------------------

bool videoInput::isDeviceSetup(int id) const
{
    if(id>=0 && id<devicesFound && VDList[id]->readyToCapture)return true;
    else return false;
}


// ----------------------------------------------------------------------
// Gives us a little pop up window to adjust settings
// We do this in a separate thread now!
// ----------------------------------------------------------------------



void __cdecl videoInput::basicThread(void* ptr)
{
    videoDevice* dev = (videoDevice*) ptr;
    IBaseFilter* filter = dev->pVideoInputFilter;

    (void) ShowFilterPropertyPages(filter);
    (void) InterlockedDecrement(&dev->property_window_count);
}

bool videoInput::showSettingsWindow(int id){
    if(isDeviceSetup(id)){
        //HANDLE myTempThread;

        //we reconnect to the device as we have freed our reference to it
        //why have we freed our reference? because there seemed to be an issue
        //with some mpeg devices if we didn't

        // XXX TODO compare fourcc for mpeg devices? is this comment still valid? -sh 20180104

        videoDevice* dev = VDList[id];
        HRESULT hr = getDevice(&dev->pVideoInputFilter, id, dev->wDeviceName, dev->nDeviceName);
        if(hr == S_OK)
        {
            // it's pointless to keep the filter around. it crashes in
            // pGraph or ISampleGrabber anyway
            //dev->pVideoInputFilter->AddRef();

            int new_window_count = InterlockedIncrement(&dev->property_window_count);
            // don't open multiple property windows at a time.
            // will cause the camera to confuse itself anyway.
            if (new_window_count == 1)
            {
                _beginthread(basicThread, 0, dev);
                return true;
            }
            else
                (void) InterlockedDecrement(&dev->property_window_count);
        }
    }
    return false;
}


// Set a video signal setting using IAMVideoProcAmp
bool videoInput::getVideoSettingFilter(int deviceID, long Property, long &min, long &max, long &SteppingDelta, long &currentValue, long &flags, long &defaultValue){
    if( !isDeviceSetup(deviceID) )return false;

    HRESULT hr;
    //bool isSuccessful = false;

    videoDevice * VD = VDList[deviceID];

    hr = getDevice(&VD->pVideoInputFilter, deviceID, VD->wDeviceName, VD->nDeviceName);
    if (FAILED(hr)){
        DebugPrintOut("setVideoSetting - getDevice Error\n");
        return false;
    }

    IAMVideoProcAmp *pAMVideoProcAmp = NULL;

    hr = VD->pVideoInputFilter->QueryInterface(IID_IAMVideoProcAmp, (void**)&pAMVideoProcAmp);
    if(FAILED(hr) || !pAMVideoProcAmp){
        DebugPrintOut("setVideoSetting - QueryInterface Error\n");
#if 0
        if(VD->pVideoInputFilter)VD->pVideoInputFilter->Release();
        if(VD->pVideoInputFilter)VD->pVideoInputFilter = NULL;
#endif
        return false;
    }

    char propStr[16];
    getVideoPropertyAsString(Property,propStr);

    DebugPrintOut("Setting video setting %s.\n", propStr);

    //both GetRange() and Get() will fail if the device doesn't support this property
    hr = pAMVideoProcAmp->GetRange(Property, &min, &max, &SteppingDelta, &defaultValue, &flags);
    if (SUCCEEDED(hr))
    {
        DebugPrintOut("Range for video setting %s: Min:%ld Max:%ld SteppingDelta:%ld Default:%ld Flags:%ld\n", propStr, min, max, SteppingDelta, defaultValue, flags);
        hr = pAMVideoProcAmp->Get(Property, &currentValue, &flags);
    }

    pAMVideoProcAmp->Release();
#if 0
    if(VD->pVideoInputFilter)VD->pVideoInputFilter->Release();
    if(VD->pVideoInputFilter)VD->pVideoInputFilter = NULL;
#endif

    return SUCCEEDED(hr);
}


// Set a video signal setting using IAMVideoProcAmp
bool videoInput::setVideoSettingFilterPct(int deviceID, long Property, float pctValue, long Flags){
    if( !isDeviceSetup(deviceID) )return false;

    long min, max, currentValue, flags, defaultValue, stepAmnt;

    if( !getVideoSettingFilter(deviceID, Property, min, max, stepAmnt, currentValue, flags, defaultValue) )return false;

    if(pctValue > 1.0)pctValue = 1.0;
    else if(pctValue < 0)pctValue = 0.0;

    float range = (float)max - (float)min;
    if(range <= 0)return false;
    if(stepAmnt == 0) return false;

    long value     = (long)( (float)min + range * pctValue );
    long rasterValue = value;

    //if the range is the stepAmnt then it is just a switch
    //so we either set the value to low or high
    if( range == stepAmnt ){
        if( pctValue < 0.5)rasterValue = min;
        else rasterValue = max;
    }else{
        //we need to rasterize the value to the stepping amnt
        long mod         = value % stepAmnt;
        float halfStep     = (float)stepAmnt * 0.5f;
        if( mod < halfStep ) rasterValue -= mod;
        else rasterValue += stepAmnt - mod;
        DebugPrintOut("RASTER - pctValue is %f - value is %li - step is %li - mod is %li - rasterValue is %li\n", pctValue, value, stepAmnt, mod, rasterValue);
    }

    return setVideoSettingFilter(deviceID, Property, rasterValue, Flags, false);
}


// Set a video signal setting using IAMVideoProcAmp
bool videoInput::setVideoSettingFilter(int deviceID, long Property, long lValue, long Flags, bool useDefaultValue){
    if( !isDeviceSetup(deviceID) )return false;

    HRESULT hr;
    //bool isSuccessful = false;

    char propStr[16];
    getVideoPropertyAsString(Property,propStr);

    videoDevice * VD = VDList[deviceID];

    hr = getDevice(&VD->pVideoInputFilter, deviceID, VD->wDeviceName, VD->nDeviceName);
    if (FAILED(hr)){
        DebugPrintOut("setVideoSetting - getDevice Error\n");
        return false;
    }

    IAMVideoProcAmp *pAMVideoProcAmp = NULL;

    hr = VD->pVideoInputFilter->QueryInterface(IID_IAMVideoProcAmp, (void**)&pAMVideoProcAmp);
    if(FAILED(hr)){
        DebugPrintOut("setVideoSetting - QueryInterface Error\n");
#if 0
        if(VD->pVideoInputFilter)VD->pVideoInputFilter->Release();
        if(VD->pVideoInputFilter)VD->pVideoInputFilter = NULL;
#endif
        return false;
    }

    DebugPrintOut("Setting video setting %s.\n", propStr);
    long CurrVal, Min, Max, SteppingDelta, Default, CapsFlags, AvailableCapsFlags = 0;


    pAMVideoProcAmp->GetRange(Property, &Min, &Max, &SteppingDelta, &Default, &AvailableCapsFlags);
    DebugPrintOut("Range for video setting %s: Min:%ld Max:%ld SteppingDelta:%ld Default:%ld Flags:%ld\n", propStr, Min, Max, SteppingDelta, Default, AvailableCapsFlags);
    pAMVideoProcAmp->Get(Property, &CurrVal, &CapsFlags);

    DebugPrintOut("Current value: %ld Flags %ld (%s)\n", CurrVal, CapsFlags, (CapsFlags == 1 ? "Auto" : (CapsFlags == 2 ? "Manual" : "Unknown")));

    if (useDefaultValue) {
        hr = pAMVideoProcAmp->Set(Property, Default, VideoProcAmp_Flags_Auto);
    }
    else{
        // Perhaps add a check that lValue and Flags are within the range acquired from GetRange above
        hr = pAMVideoProcAmp->Set(Property, lValue, Flags);
    }

    if(pAMVideoProcAmp)pAMVideoProcAmp->Release();
#if 0
    if(VD->pVideoInputFilter)VD->pVideoInputFilter->Release();
    if(VD->pVideoInputFilter)VD->pVideoInputFilter = NULL;
#endif

    return SUCCEEDED(hr);

}


bool videoInput::setVideoSettingCameraPct(int deviceID, long Property, float pctValue, long Flags){
    if( !isDeviceSetup(deviceID) )return false;

    long min, max, currentValue, flags, defaultValue, stepAmnt;

    if( !getVideoSettingCamera(deviceID, Property, min, max, stepAmnt, currentValue, flags, defaultValue) )return false;

    if(pctValue > 1.0)pctValue = 1.0;
    else if(pctValue < 0)pctValue = 0.0;

    float range = (float)max - (float)min;
    if(range <= 0)return false;
    if(stepAmnt == 0) return false;

    long value     = (long)( (float)min + range * pctValue );
    long rasterValue = value;

    //if the range is the stepAmnt then it is just a switch
    //so we either set the value to low or high
    if( range == stepAmnt ){
        if( pctValue < 0.5)rasterValue = min;
        else rasterValue = max;
    }else{
        //we need to rasterize the value to the stepping amnt
        long mod         = value % stepAmnt;
        float halfStep     = (float)stepAmnt * 0.5f;
        if( mod < halfStep ) rasterValue -= mod;
        else rasterValue += stepAmnt - mod;
        DebugPrintOut("RASTER - pctValue is %f - value is %li - step is %li - mod is %li - rasterValue is %li\n", pctValue, value, stepAmnt, mod, rasterValue);
    }

    return setVideoSettingCamera(deviceID, Property, rasterValue, Flags, false);
}


bool videoInput::setVideoSettingCamera(int deviceID, long Property, long lValue, long Flags, bool useDefaultValue){
    IAMCameraControl *pIAMCameraControl;
    if(isDeviceSetup(deviceID))
    {
        HRESULT hr;
        hr = getDevice(&VDList[deviceID]->pVideoInputFilter, deviceID, VDList[deviceID]->wDeviceName, VDList[deviceID]->nDeviceName);

        char propStr[16];
        getCameraPropertyAsString(Property,propStr);

        DebugPrintOut("Setting video setting %s.\n", propStr);
        hr = VDList[deviceID]->pVideoInputFilter->QueryInterface(IID_IAMCameraControl, (void**)&pIAMCameraControl);
        if (FAILED(hr)) {
            DebugPrintOut("Error\n");
#if 0
            if(VDList[deviceID]->pVideoInputFilter)VDList[deviceID]->pVideoInputFilter->Release();
            if(VDList[deviceID]->pVideoInputFilter)VDList[deviceID]->pVideoInputFilter = NULL;
#endif
            return false;
        }
        else
        {
            long CurrVal, Min, Max, SteppingDelta, Default, CapsFlags, AvailableCapsFlags;
            pIAMCameraControl->GetRange(Property, &Min, &Max, &SteppingDelta, &Default, &AvailableCapsFlags);
            DebugPrintOut("Range for video setting %s: Min:%ld Max:%ld SteppingDelta:%ld Default:%ld Flags:%ld\n", propStr, Min, Max, SteppingDelta, Default, AvailableCapsFlags);
            pIAMCameraControl->Get(Property, &CurrVal, &CapsFlags);
            DebugPrintOut("Current value: %ld Flags %ld (%s)\n", CurrVal, CapsFlags, (CapsFlags == 1 ? "Auto" : (CapsFlags == 2 ? "Manual" : "Unknown")));
            if (useDefaultValue) {
                hr = pIAMCameraControl->Set(Property, Default, CameraControl_Flags_Auto);
            }
            else
            {
                // Perhaps add a check that lValue and Flags are within the range acquired from GetRange above
                hr = pIAMCameraControl->Set(Property, lValue, Flags);
            }
            pIAMCameraControl->Release();
#if 0
            if(VDList[deviceID]->pVideoInputFilter)VDList[deviceID]->pVideoInputFilter->Release();
            if(VDList[deviceID]->pVideoInputFilter)VDList[deviceID]->pVideoInputFilter = NULL;
#endif
            return SUCCEEDED(hr);
        }
    }
    return false;
}



bool videoInput::getVideoSettingCamera(int deviceID, long Property, long &min, long &max, long &SteppingDelta, long &currentValue, long &flags, long &defaultValue){
    if( !isDeviceSetup(deviceID) )return false;

    HRESULT hr;
    //bool isSuccessful = false;

    videoDevice * VD = VDList[deviceID];

    hr = getDevice(&VD->pVideoInputFilter, deviceID, VD->wDeviceName, VD->nDeviceName);
    if (FAILED(hr)){
        DebugPrintOut("setVideoSetting - getDevice Error\n");
        return false;
    }

    IAMCameraControl *pIAMCameraControl = NULL;

    hr = VD->pVideoInputFilter->QueryInterface(IID_IAMCameraControl, (void**)&pIAMCameraControl);
    if(FAILED(hr) || !pIAMCameraControl){
        DebugPrintOut("setVideoSetting - QueryInterface Error\n");
#if 0
        if(VD->pVideoInputFilter)VD->pVideoInputFilter->Release();
        if(VD->pVideoInputFilter)VD->pVideoInputFilter = NULL;
#endif
        return false;
    }

    char propStr[16];
    getCameraPropertyAsString(Property,propStr);
    DebugPrintOut("Setting video setting %s.\n", propStr);

    //both GetRange() and Get() will fail if the device doesn't support this property
    hr = pIAMCameraControl->GetRange(Property, &min, &max, &SteppingDelta, &defaultValue, &flags);
    if (SUCCEEDED(hr))
    {
        DebugPrintOut("Range for video setting %s: Min:%ld Max:%ld SteppingDelta:%ld Default:%ld Flags:%ld\n", propStr, min, max, SteppingDelta, defaultValue, flags);
        hr = pIAMCameraControl->Get(Property, &currentValue, &flags);
    }

    pIAMCameraControl->Release();
#if 0
    if(VD->pVideoInputFilter)VD->pVideoInputFilter->Release();
    if(VD->pVideoInputFilter)VD->pVideoInputFilter = NULL;
#endif

    return SUCCEEDED(hr);
}


// ----------------------------------------------------------------------
// Shutsdown the device, deletes the object and creates a new object
// so it is ready to be setup again
// ----------------------------------------------------------------------

void videoInput::stopDevice(int id){
    if(id < VI_MAX_CAMERAS)
    {
        delete VDList[id];
        VDList[id] = new videoDevice();
    }

}

// ----------------------------------------------------------------------
// Restarts the device with the same settings it was using
//
// ----------------------------------------------------------------------

bool videoInput::restartDevice(int id){
    if(isDeviceSetup(id))
    {
        int conn         = VDList[id]->storeConn;
        int tmpW           = VDList[id]->width;
        int tmpH           = VDList[id]->height;

        bool bFormat    = VDList[id]->specificFormat;
        long format     = VDList[id]->formatType;

        int nReconnect    = VDList[id]->nFramesForReconnect;
        bool bReconnect = VDList[id]->autoReconnect;

        unsigned long avgFrameTime = VDList[id]->requestedFrameTime;

        stopDevice(id);

        //set our fps if needed
        if( avgFrameTime != (unsigned long)-1){
            VDList[id]->requestedFrameTime = avgFrameTime;
        }

        if( setupDevice(id, tmpW, tmpH, conn) ){
            //reapply the format - ntsc / pal etc
            if( bFormat ){
                setFormat(id, format);
            }
            if( bReconnect ){
                setAutoReconnectOnFreeze(id, true, nReconnect);
            }
            return true;
        }
    }
    return false;
}

// ----------------------------------------------------------------------
// Shuts down all devices, deletes objects and unitializes com if needed
//
// ----------------------------------------------------------------------
videoInput::~videoInput(){

    for(int i = 0; i < VI_MAX_CAMERAS; i++)
    {
        delete VDList[i];
    }
    //Uninitialize com
    comUnInit();
}


//////////////////////////////  VIDEO INPUT  ////////////////////////////////
////////////////////////////  PRIVATE METHODS  //////////////////////////////

// ----------------------------------------------------------------------
// We only should init com if it hasn't been done so by our apps thread
// Use a static counter to keep track of other times it has been inited
// (do we need to worry about multithreaded apps?)
// ----------------------------------------------------------------------

bool videoInput::comInit(){
    /*HRESULT hr = NOERROR;

    //no need for us to start com more than once
    if(comInitCount == 0 ){

        // Initialize the COM library.
        //CoInitializeEx so videoInput can run in another thread
    #ifdef VI_COM_MULTI_THREADED
        hr = CoInitializeEx(NULL,COINIT_MULTITHREADED);
    #else
        hr = CoInitialize(NULL);
    #endif
        //this is the only case where there might be a problem
        //if another library has started com as single threaded
        //and we need it multi-threaded - send warning but don't fail
        if( hr == RPC_E_CHANGED_MODE){
             DebugPrintOut("SETUP - COM already setup - threaded VI might not be possible\n");
        }
    }

    comInitCount++;*/
    return true;
}


// ----------------------------------------------------------------------
// Same as above but to uninitialize com, decreases counter and frees com
// if no one else is using it
// ----------------------------------------------------------------------

bool videoInput::comUnInit(){
    /*if(comInitCount > 0)comInitCount--;        //decrease the count of instances using com

       if(comInitCount == 0){
           CoUninitialize();    //if there are no instances left - uninitialize com
        return true;
    }

    return false;*/
    return true;
}


// ----------------------------------------------------------------------
// This is the size we ask for - we might not get it though :)
//
// ----------------------------------------------------------------------

void videoInput::setAttemptCaptureSize(int id, int w, int h,GUID mediaType){

    VDList[id]->tryWidth    = w;
    VDList[id]->tryHeight   = h;
    VDList[id]->tryDiffSize = true;
    VDList[id]->tryVideoType = mediaType;

}

// ----------------------------------------------------------------------
// Set the connection type
// (maybe move to private?)
// ----------------------------------------------------------------------

void videoInput::setPhyCon(int id, int conn){

    switch(conn){

        case 0:
            VDList[id]->connection = PhysConn_Video_Composite;
            break;
        case 1:
            VDList[id]->connection = PhysConn_Video_SVideo;
            break;
        case 2:
            VDList[id]->connection = PhysConn_Video_Tuner;
            break;
        case 3:
            VDList[id]->connection = PhysConn_Video_USB;
            break;
        case 4:
            VDList[id]->connection = PhysConn_Video_1394;
            break;
        default:
            return; //if it is not these types don't set crossbar
        break;
    }

    VDList[id]->storeConn    = conn;
    VDList[id]->useCrossbar    = true;
}


// ----------------------------------------------------------------------
// Check that we are not trying to setup a non-existent device
// Then start the graph building!
// ----------------------------------------------------------------------

bool videoInput::setup(int deviceNumber){
    devicesFound = getDeviceCount();

     if(deviceNumber>devicesFound-1)
    {
        DebugPrintOut("SETUP: device[%i] not found - you have %i devices available\n", deviceNumber, devicesFound);
        if(devicesFound>=0) DebugPrintOut("SETUP: this means that the last device you can use is device[%i]\n",  devicesFound-1);
        return false;
    }

    if(VDList[deviceNumber]->readyToCapture)
    {
        DebugPrintOut("SETUP: can't setup, device %i is currently being used\n",VDList[deviceNumber]->myID);
        return false;
    }

    HRESULT hr = start(deviceNumber, VDList[deviceNumber]);
    if(hr == S_OK)return true;
    else return false;
}


// ----------------------------------------------------------------------
// Does both vertical buffer flipping and bgr to rgb swapping
// You have any combination of those.
// ----------------------------------------------------------------------

void videoInput::processPixels(unsigned char * src, unsigned char * dst, int width, int height, bool bRGB, bool bFlip, int bytesperpixel){

    int widthInBytes = width * bytesperpixel;
    int numBytes = widthInBytes * height;

    if (bytesperpixel == 2) {
        for (int i = 0; i < width*height; i++) {
            if (bytesperpixel == 2) {
                *dst = (uint8_t) (*((uint16_t*) src) >> 8);
                dst++;

                *dst = (uint8_t) (*((uint16_t*)src) >> 8);
                dst++;

                *dst = (uint8_t) (*((uint16_t*)src) >> 8);
                dst++;

                src += 2;
            }
        }
    }
    else
    {
        if(!bRGB){

            //int x = 0;
            //int y = 0;

            if(bFlip){
                for(int y = 0; y < height; y++){
                    memcpy(dst + (y * widthInBytes), src + ( (height -y -1) * widthInBytes), widthInBytes);
                }

            }else{
                memcpy(dst, src, numBytes);
            }
        }else{
            if(bFlip){

                int x = 0;
                int y = (height - 1) * widthInBytes;
                src += y;

                for(int i = 0; i < numBytes; i+=3){
                    if(x >= width){
                        x = 0;
                        src -= widthInBytes*2;
                    }

                    *dst = *(src+2);
                    dst++;

                    *dst = *(src+1);
                    dst++;

                    *dst = *src;
                    dst++;

                    src+=3;
                    x++;
                }
            }
            else{
                for(int i = 0; i < numBytes; i+=3){
                    *dst = *(src+2);
                    dst++;

                    *dst = *(src+1);
                    dst++;

                    *dst = *src;
                    dst++;

                    src+=3;
                }
            }
        }
    }
}


//------------------------------------------------------------------------------------------
void videoInput::getMediaSubtypeAsString(GUID type, char * typeAsString){

    char tmpStr[8];
    if( type == MEDIASUBTYPE_RGB24)     sprintf(tmpStr, "RGB24");
    else if(type == MEDIASUBTYPE_RGB32) sprintf(tmpStr, "RGB32");
    else if(type == MEDIASUBTYPE_RGB555)sprintf(tmpStr, "RGB555");
    else if(type == MEDIASUBTYPE_RGB565)sprintf(tmpStr, "RGB565");
    else if(type == MEDIASUBTYPE_YUY2)  sprintf(tmpStr, "YUY2");
    else if(type == MEDIASUBTYPE_YVYU)  sprintf(tmpStr, "YVYU");
    else if(type == MEDIASUBTYPE_YUYV)  sprintf(tmpStr, "YUYV");
    else if(type == MEDIASUBTYPE_IYUV)  sprintf(tmpStr, "IYUV");
    else if(type == MEDIASUBTYPE_UYVY)  sprintf(tmpStr, "UYVY");
    else if(type == MEDIASUBTYPE_YV12)  sprintf(tmpStr, "YV12");
    else if(type == MEDIASUBTYPE_YVU9)  sprintf(tmpStr, "YVU9");
    else if(type == MEDIASUBTYPE_Y411)  sprintf(tmpStr, "Y411");
    else if(type == MEDIASUBTYPE_Y41P)  sprintf(tmpStr, "Y41P");
    else if(type == MEDIASUBTYPE_Y211)  sprintf(tmpStr, "Y211");
    else if(type == MEDIASUBTYPE_AYUV)  sprintf(tmpStr, "AYUV");
    else if(type == MEDIASUBTYPE_MJPG)  sprintf(tmpStr, "MJPG");
    else if(type == MEDIASUBTYPE_Y800)  sprintf(tmpStr, "Y800");
    else if(type == MEDIASUBTYPE_Y8)    sprintf(tmpStr, "Y8");
    else if(type == MEDIASUBTYPE_GREY)  sprintf(tmpStr, "GREY");
    else if(type == MEDIASUBTYPE_I420)  sprintf(tmpStr, "I420");
    else if (type == MEDIASUBTYPE_BY8)  sprintf(tmpStr, "BY8");
    else if (type == MEDIASUBTYPE_Y16)  sprintf(tmpStr, "Y16");
    else sprintf(tmpStr, "OTHER");

    memcpy(typeAsString, tmpStr, sizeof(char)*8);
}

int videoInput::getFourccFromMediaSubtype(GUID type) const
{
    return type.Data1;
}

GUID *videoInput::getMediaSubtypeFromFourcc(int fourcc){

    for (int i=0;i<VI_NUM_TYPES;i++) {
        if ( (unsigned long)(unsigned)fourcc == mediaSubtypes[i].Data1 ) {
            return &mediaSubtypes[i];
        }
    }

    return NULL;
}


void videoInput::getVideoPropertyAsString(int prop, char * propertyAsString){

    char tmpStr[16];

    if ( prop==VideoProcAmp_Brightness) sprintf(tmpStr, "Brightness");
    else if ( prop==VideoProcAmp_Contrast) sprintf(tmpStr, "Contrast");
    else if ( prop==VideoProcAmp_Saturation) sprintf(tmpStr, "Saturation");
    else if ( prop==VideoProcAmp_Hue) sprintf(tmpStr, "Hue");
    else if ( prop==VideoProcAmp_Gain) sprintf(tmpStr, "Gain");
    else if ( prop==VideoProcAmp_Gamma) sprintf(tmpStr, "Gamma");
    else if ( prop==VideoProcAmp_ColorEnable) sprintf(tmpStr, "ColorEnable");
    else if ( prop==VideoProcAmp_Sharpness) sprintf(tmpStr, "Sharpness");
    else sprintf(tmpStr, "%u",prop);

    memcpy(propertyAsString, tmpStr, sizeof(char)*16);
}


int videoInput::getVideoPropertyFromCV(int cv_property){
    // see VideoProcAmpProperty in strmif.h
    switch (cv_property) {
        case CV_CAP_PROP_BRIGHTNESS:
            return VideoProcAmp_Brightness;

        case CV_CAP_PROP_CONTRAST:
            return VideoProcAmp_Contrast;

        case CV_CAP_PROP_HUE:
            return VideoProcAmp_Hue;

        case CV_CAP_PROP_SATURATION:
            return VideoProcAmp_Saturation;

        case CV_CAP_PROP_SHARPNESS:
            return VideoProcAmp_Sharpness;

        case CV_CAP_PROP_GAMMA:
            return VideoProcAmp_Gamma;

        case CV_CAP_PROP_MONOCHROME:
            return VideoProcAmp_ColorEnable;

        case CV_CAP_PROP_WHITE_BALANCE_BLUE_U:
            return VideoProcAmp_WhiteBalance;

        case  CV_CAP_PROP_BACKLIGHT:
            return VideoProcAmp_BacklightCompensation;

        case CV_CAP_PROP_GAIN:
            return VideoProcAmp_Gain;
    }
    return -1;
}

int videoInput::getCameraPropertyFromCV(int cv_property){

    // see CameraControlProperty in strmif.h
    switch (cv_property) {
        case CV_CAP_PROP_PAN:
            return CameraControl_Pan;

        case CV_CAP_PROP_TILT:
            return CameraControl_Tilt;

        case CV_CAP_PROP_ROLL:
            return CameraControl_Roll;

        case CV_CAP_PROP_ZOOM:
            return CameraControl_Zoom;

        case CV_CAP_PROP_EXPOSURE:
            return CameraControl_Exposure;

        case CV_CAP_PROP_IRIS:
            return CameraControl_Iris;

        case CV_CAP_PROP_FOCUS:
            return CameraControl_Focus;
    }
    return -1;
}

bool videoInput::isDeviceDisconnected(int deviceNumber)
{
    if (!isDeviceSetup(deviceNumber)) return true;
    long evCode;
    LONG_PTR param1, param2;
    bool disconnected = false;

    while (S_OK == VDList[deviceNumber]->pMediaEvent->GetEvent(&evCode, &param1, &param2, 0))
    {
        DebugPrintOut("Event: Code: %#04x Params: %d, %d\n", evCode, param1, param2);

        VDList[deviceNumber]->pMediaEvent->FreeEventParams(evCode, param1, param2);
        if (evCode == EC_DEVICE_LOST)
        {
           DebugPrintOut("ERROR: Device disconnected\n");
           disconnected = true;
        }
    }
    return disconnected;
}

void videoInput::getCameraPropertyAsString(int prop, char * propertyAsString){

    char tmpStr[16];

    if ( prop==CameraControl_Pan) sprintf(tmpStr, "Pan");
    else if ( prop==CameraControl_Tilt) sprintf(tmpStr, "Tilt");
    else if ( prop==CameraControl_Roll) sprintf(tmpStr, "Roll");
    else if ( prop==CameraControl_Zoom) sprintf(tmpStr, "Zoom");
    else if ( prop==CameraControl_Exposure) sprintf(tmpStr, "Exposure");
    else if ( prop==CameraControl_Iris) sprintf(tmpStr, "Iris");
    else if ( prop==CameraControl_Focus) sprintf(tmpStr, "Focus");
    else sprintf(tmpStr, "%u",prop);

    memcpy(propertyAsString, tmpStr, sizeof(char)*16);
}

GUID videoInput::getMediasubtype(int deviceID)
{
    return VDList[deviceID]->pAmMediaType->subtype;
}

//-------------------------------------------------------------------------------------------
static void findClosestSizeAndSubtype(videoDevice * VD, int widthIn, int heightIn, int &widthOut, int &heightOut, GUID & mediatypeOut){
    HRESULT hr;

    //find perfect match or closest size
    int nearW                = 9999999;
    int nearH                = 9999999;
    //bool foundClosestMatch     = true;

    int iCount = 0;
    int iSize = 0;
    hr = VD->streamConf->GetNumberOfCapabilities(&iCount, &iSize);

    if (iSize == sizeof(VIDEO_STREAM_CONFIG_CAPS))
    {
        //For each format type RGB24 YUV2 etc
        for (int iFormat = 0; iFormat < iCount; iFormat++)
        {
            VIDEO_STREAM_CONFIG_CAPS scc;
            AM_MEDIA_TYPE *pmtConfig;
            hr =  VD->streamConf->GetStreamCaps(iFormat, &pmtConfig, (BYTE*)&scc);

            if (SUCCEEDED(hr)){

                //his is how many diff sizes are available for the format
                int stepX = scc.OutputGranularityX;
                int stepY = scc.OutputGranularityY;

                   int tempW = 999999;
                   int tempH = 999999;

                   //Don't want to get stuck in a loop
                   if(stepX < 1 || stepY < 1) continue;

                   //DebugPrintOut("min is %i %i max is %i %i - res is %i %i\n", scc.MinOutputSize.cx, scc.MinOutputSize.cy,  scc.MaxOutputSize.cx,  scc.MaxOutputSize.cy, stepX, stepY);
                   //DebugPrintOut("min frame duration is %i  max duration is %i\n", scc.MinFrameInterval, scc.MaxFrameInterval);

                   bool exactMatch     = false;
                   bool exactMatchX    = false;
                bool exactMatchY    = false;

                for(int x = scc.MinOutputSize.cx; x <= scc.MaxOutputSize.cx; x+= stepX){
                    //If we find an exact match
                    if( widthIn == x ){
                        exactMatchX = true;
                        tempW = x;
                    }
                    //Otherwise lets find the closest match based on width
                    else if( abs(widthIn-x) < abs(widthIn-tempW) ){
                        tempW = x;
                    }
                }

                for(int y = scc.MinOutputSize.cy; y <= scc.MaxOutputSize.cy; y+= stepY){
                    //If we find an exact match
                    if( heightIn == y){
                        exactMatchY = true;
                        tempH = y;
                    }
                    //Otherwise lets find the closest match based on height
                    else if( abs(heightIn-y) < abs(heightIn-tempH) ){
                        tempH = y;
                    }
                }

                //see if we have an exact match!
                if(exactMatchX && exactMatchY){
                    //foundClosestMatch = false;
                    exactMatch = true;

                    widthOut        = widthIn;
                    heightOut        = heightIn;
                    mediatypeOut    = pmtConfig->subtype;
                }

                  //otherwise lets see if this filters closest size is the closest
                  //available. the closest size is determined by the sum difference
                //of the widths and heights
                  else if( abs(widthIn - tempW) + abs(heightIn - tempH)  < abs(widthIn - nearW) + abs(heightIn - nearH) )
                  {
                      nearW = tempW;
                      nearH = tempH;

                    widthOut        = nearW;
                    heightOut        = nearH;
                    mediatypeOut    = pmtConfig->subtype;
                  }

                MyDeleteMediaType(pmtConfig);

                //If we have found an exact match no need to search anymore
                if(exactMatch)break;
            }
         }
    }

}


//---------------------------------------------------------------------------------------------------
static bool setSizeAndSubtype(videoDevice * VD, int attemptWidth, int attemptHeight, GUID mediatype){
    VIDEOINFOHEADER *pVih =  reinterpret_cast<VIDEOINFOHEADER*>(VD->pAmMediaType->pbFormat);

    //store current size
    //int tmpWidth  = HEADER(pVih)->biWidth;
    //int tmpHeight = HEADER(pVih)->biHeight;
    AM_MEDIA_TYPE * tmpType = NULL;

    HRESULT    hr = VD->streamConf->GetFormat(&tmpType);
    if(hr != S_OK)return false;

    //set new size:
    //width and height
    HEADER(pVih)->biWidth  = attemptWidth;
    HEADER(pVih)->biHeight = attemptHeight;

    VD->pAmMediaType->formattype = FORMAT_VideoInfo;
    VD->pAmMediaType->majortype  = MEDIATYPE_Video;
    VD->pAmMediaType->subtype     = mediatype;

    //buffer size
    if (mediatype == MEDIASUBTYPE_RGB24){
        VD->pAmMediaType->lSampleSize = attemptWidth*attemptHeight * 3;
    }
    else if ((mediatype == MEDIASUBTYPE_YUY2) || (mediatype == MEDIASUBTYPE_YVYU) ||
        (mediatype == MEDIASUBTYPE_UYVY)){

        VD->pAmMediaType->lSampleSize = attemptWidth*attemptHeight * 2;
    }
    else{
        VD->pAmMediaType->lSampleSize = 0;
    }

    //set fps if requested
    if( VD->requestedFrameTime != -1){
        pVih->AvgTimePerFrame = VD->requestedFrameTime;
    }

    //okay lets try new size
    hr = VD->streamConf->SetFormat(VD->pAmMediaType);
    if(hr == S_OK){
        if( tmpType != NULL )MyDeleteMediaType(tmpType);
        return true;
    }else{
        VD->streamConf->SetFormat(tmpType);
        if( tmpType != NULL )MyDeleteMediaType(tmpType);
    }

    return false;
}

// ----------------------------------------------------------------------
// Where all the work happens!
// Attempts to build a graph for the specified device
// ----------------------------------------------------------------------

int videoInput::start(int deviceID, videoDevice *VD){

    HRESULT hr             = NOERROR;
    VD->myID             = deviceID;
    VD->setupStarted    = true;
    CAPTURE_MODE           = PIN_CATEGORY_CAPTURE; //Don't worry - it ends up being preview (which is faster)
    callbackSetCount     = 1;  //make sure callback method is not changed after setup called

    DebugPrintOut("SETUP: Setting up device %i\n",deviceID);

    // CREATE THE GRAPH BUILDER //
    // Create the filter graph manager and query for interfaces.
    hr = CoCreateInstance(CLSID_CaptureGraphBuilder2, NULL, CLSCTX_INPROC_SERVER, IID_ICaptureGraphBuilder2, (void **)&VD->pCaptureGraph);
    if (FAILED(hr))    // FAILED is a macro that tests the return value
    {
        DebugPrintOut("ERROR - Could not create the Filter Graph Manager\n");
        return hr;
    }

    //FILTER GRAPH MANAGER//
    // Create the Filter Graph Manager.
    hr = CoCreateInstance(CLSID_FilterGraph, 0, CLSCTX_INPROC_SERVER,IID_IGraphBuilder, (void**)&VD->pGraph);
    if (FAILED(hr))
    {
        DebugPrintOut("ERROR - Could not add the graph builder!\n");
        stopDevice(deviceID);
        return hr;
    }

    //MEDIA EVENT//
    //Used to obtain event when capture device is disconnected
    hr = VD->pGraph->QueryInterface(IID_IMediaEventEx, (void**)&VD->pMediaEvent);
    if (FAILED(hr))
    {
        DebugPrintOut("ERROR - Could not create media event object\n");
        stopDevice(deviceID);
        return hr;
    }

    //SET THE FILTERGRAPH//
    hr = VD->pCaptureGraph->SetFiltergraph(VD->pGraph);
    if (FAILED(hr))
    {
        DebugPrintOut("ERROR - Could not set filtergraph\n");
        stopDevice(deviceID);
        return hr;
    }

    //MEDIA CONTROL (START/STOPS STREAM)//
    // Using QueryInterface on the graph builder,
    // Get the Media Control object.
    hr = VD->pGraph->QueryInterface(IID_IMediaControl, (void **)&VD->pControl);
    if (FAILED(hr))
    {
        DebugPrintOut("ERROR - Could not create the Media Control object\n");
        stopDevice(deviceID);
        return hr;
    }


    //FIND VIDEO DEVICE AND ADD TO GRAPH//
    //gets the device specified by the second argument.
    hr = getDevice(&VD->pVideoInputFilter, deviceID, VD->wDeviceName, VD->nDeviceName);

    if (SUCCEEDED(hr)){
        DebugPrintOut("SETUP: %s\n", VD->nDeviceName);
        hr = VD->pGraph->AddFilter(VD->pVideoInputFilter, VD->wDeviceName);
    }else{
        DebugPrintOut("ERROR - Could not find specified video device\n");
        stopDevice(deviceID);
        return hr;
    }

    //LOOK FOR PREVIEW PIN IF THERE IS NONE THEN WE USE CAPTURE PIN AND THEN SMART TEE TO PREVIEW
    IAMStreamConfig *streamConfTest = NULL;
    hr = VD->pCaptureGraph->FindInterface(&PIN_CATEGORY_PREVIEW, &MEDIATYPE_Video, VD->pVideoInputFilter, IID_IAMStreamConfig, (void **)&streamConfTest);
    if(FAILED(hr)){
        DebugPrintOut("SETUP: Couldn't find preview pin using SmartTee\n");
    }else{
         CAPTURE_MODE = PIN_CATEGORY_PREVIEW;
         streamConfTest->Release();
         streamConfTest = NULL;
    }

    //CROSSBAR (SELECT PHYSICAL INPUT TYPE)//
    //my own function that checks to see if the device can support a crossbar and if so it routes it.
    //webcams tend not to have a crossbar so this function will also detect a webcams and not apply the crossbar
    if(VD->useCrossbar)
    {
        DebugPrintOut("SETUP: Checking crossbar\n");
        routeCrossbar(&VD->pCaptureGraph, &VD->pVideoInputFilter, VD->connection, CAPTURE_MODE);
    }


    //we do this because webcams don't have a preview mode
    hr = VD->pCaptureGraph->FindInterface(&CAPTURE_MODE, &MEDIATYPE_Video, VD->pVideoInputFilter, IID_IAMStreamConfig, (void **)&VD->streamConf);
    if(FAILED(hr) || !VD->streamConf){
        DebugPrintOut("ERROR: Couldn't config the stream!\n");
        stopDevice(deviceID);
        return hr;
    }

    //NOW LETS DEAL WITH GETTING THE RIGHT SIZE
    hr = VD->streamConf->GetFormat(&VD->pAmMediaType);
    if(FAILED(hr)){
        DebugPrintOut("ERROR: Couldn't getFormat for pAmMediaType!\n");
        stopDevice(deviceID);
        return hr;
    }

    VIDEOINFOHEADER *pVih =  reinterpret_cast<VIDEOINFOHEADER*>(VD->pAmMediaType->pbFormat);
    int currentWidth    =  HEADER(pVih)->biWidth;
    int currentHeight    =  HEADER(pVih)->biHeight;

    bool customSize = VD->tryDiffSize;

    bool foundSize  = false;

    if(customSize){
        DebugPrintOut("SETUP: Default Format is set to %ix%i\n", currentWidth, currentHeight);

        char guidStr[8];
            // try specified format and size
            getMediaSubtypeAsString(VD->tryVideoType, guidStr);
            DebugPrintOut("SETUP: trying specified format %s @ %ix%i\n", guidStr, VD->tryWidth, VD->tryHeight);

            if( setSizeAndSubtype(VD, VD->tryWidth, VD->tryHeight, VD->tryVideoType) ){
                VD->setSize(VD->tryWidth, VD->tryHeight);
                VD->videoType = VD->tryVideoType;
                foundSize = true;
            } else {
                // try specified size with all formats
                for(int i = 0; i < VI_NUM_TYPES; i++){

                    getMediaSubtypeAsString(mediaSubtypes[i], guidStr);

                    DebugPrintOut("SETUP: trying format %s @ %ix%i\n", guidStr, VD->tryWidth, VD->tryHeight);
                    if( setSizeAndSubtype(VD, VD->tryWidth, VD->tryHeight, mediaSubtypes[i]) ){
                        VD->setSize(VD->tryWidth, VD->tryHeight);
                        VD->videoType = mediaSubtypes[i];
                        foundSize = true;
                        break;
                    }
                }
            }


        //if we didn't find the requested size - lets try and find the closest matching size
        if( foundSize == false ){
            DebugPrintOut("SETUP: couldn't find requested size - searching for closest matching size\n");

            int closestWidth        = -1;
            int closestHeight        = -1;
            GUID newMediaSubtype;

            findClosestSizeAndSubtype(VD, VD->tryWidth, VD->tryHeight, closestWidth, closestHeight, newMediaSubtype);

            if( closestWidth != -1 && closestHeight != -1){
                getMediaSubtypeAsString(newMediaSubtype, guidStr);

                DebugPrintOut("SETUP: closest supported size is %s @ %i %i\n", guidStr, closestWidth, closestHeight);
                if( setSizeAndSubtype(VD, closestWidth, closestHeight, newMediaSubtype) ){
                    VD->setSize(closestWidth, closestHeight);
                    foundSize = true;
                }
            }
        }
    }

    //if we didn't specify a custom size or if we did but couldn't find it lets setup with the default settings
    if(customSize == false || foundSize == false){
        if( VD->requestedFrameTime != -1 ){
            pVih->AvgTimePerFrame  = VD->requestedFrameTime;
            hr = VD->streamConf->SetFormat(VD->pAmMediaType);
        }
        VD->setSize(currentWidth, currentHeight);
    }

    //SAMPLE GRABBER (ALLOWS US TO GRAB THE BUFFER)//
    // Create the Sample Grabber.
    hr = CoCreateInstance(CLSID_SampleGrabber, NULL, CLSCTX_INPROC_SERVER,IID_IBaseFilter, (void**)&VD->pGrabberF);
    if (FAILED(hr)){
        DebugPrintOut("Could not Create Sample Grabber - CoCreateInstance()\n");
        stopDevice(deviceID);
        return hr;
    }

    hr = VD->pGraph->AddFilter(VD->pGrabberF, L"Sample Grabber");
    if (FAILED(hr)){
        DebugPrintOut("Could not add Sample Grabber - AddFilter()\n");
        stopDevice(deviceID);
        return hr;
    }

    hr = VD->pGrabberF->QueryInterface(IID_ISampleGrabber, (void**)&VD->pGrabber);
    if (FAILED(hr)){
        DebugPrintOut("ERROR: Could not query SampleGrabber\n");
        stopDevice(deviceID);
        return hr;
    }


    //Set Params - One Shot should be false unless you want to capture just one buffer
    hr = VD->pGrabber->SetOneShot(FALSE);
    if(bCallback){
        hr = VD->pGrabber->SetBufferSamples(FALSE);
    }else{
        hr = VD->pGrabber->SetBufferSamples(TRUE);
    }

    if(bCallback){
        //Tell the grabber to use our callback function - 0 is for SampleCB and 1 for BufferCB
        //We use SampleCB
        hr = VD->pGrabber->SetCallback(VD->sgCallback, 0);
        if (FAILED(hr)){
            DebugPrintOut("ERROR: problem setting callback\n");
            stopDevice(deviceID);
            return hr;
        }else{
            DebugPrintOut("SETUP: Capture callback set\n");
        }
    }

    //MEDIA CONVERSION
    //Get video properties from the stream's mediatype and apply to the grabber (otherwise we don't get an RGB image)
    //zero the media type - lets try this :) - maybe this works?
    AM_MEDIA_TYPE mt;
    ZeroMemory(&mt,sizeof(AM_MEDIA_TYPE));

    mt.majortype     = MEDIATYPE_Video;

    // Disable format conversion if using 8/16-bit data (e-Con systems)
    if (checkSingleByteFormat(VD->pAmMediaType->subtype) || (VD->pAmMediaType->subtype == MEDIASUBTYPE_Y16)) {
        DebugPrintOut("SETUP: Not converting frames to RGB.\n");
        mt.subtype = VD->pAmMediaType->subtype;
    }
    else
    {
        DebugPrintOut("SETUP: Converting frames to RGB.\n");
        mt.subtype = MEDIASUBTYPE_RGB24;	//Making it RGB24, does conversion from YUV to RGB
    }
    mt.formattype     = FORMAT_VideoInfo;

    //VD->pAmMediaType->subtype = VD->videoType;
    hr = VD->pGrabber->SetMediaType(&mt);

    //lets try freeing our stream conf here too
    //this will fail if the device is already running
    VD->streamConf->Release();
    VD->streamConf = NULL;


    //NULL RENDERER//
    //used to give the video stream somewhere to go to.
    hr = CoCreateInstance(CLSID_NullRenderer, NULL, CLSCTX_INPROC_SERVER, IID_IBaseFilter, (void**)(&VD->pDestFilter));
    if (FAILED(hr)){
        DebugPrintOut("ERROR: Could not create filter - NullRenderer\n");
        stopDevice(deviceID);
        return hr;
    }

    hr = VD->pGraph->AddFilter(VD->pDestFilter, L"NullRenderer");
    if (FAILED(hr)){
        DebugPrintOut("ERROR: Could not add filter - NullRenderer\n");
        stopDevice(deviceID);
        return hr;
    }

    //RENDER STREAM//
    //This is where the stream gets put together.
    hr = VD->pCaptureGraph->RenderStream(&PIN_CATEGORY_PREVIEW, &MEDIATYPE_Video, VD->pVideoInputFilter, VD->pGrabberF, VD->pDestFilter);

    if (FAILED(hr)){
        DebugPrintOut("ERROR: Could not connect pins - RenderStream()\n");
        stopDevice(deviceID);
        return hr;
    }


    //EXP - lets try setting the sync source to null - and make it run as fast as possible
    {
        IMediaFilter *pMediaFilter = 0;
        hr = VD->pGraph->QueryInterface(IID_IMediaFilter, (void**)&pMediaFilter);
        if (FAILED(hr)){
            DebugPrintOut("ERROR: Could not get IID_IMediaFilter interface\n");
        }else{
            pMediaFilter->SetSyncSource(NULL);
            pMediaFilter->Release();
        }
    }


    //LETS RUN THE STREAM!
    hr = VD->pControl->Run();

    if (FAILED(hr)){
         DebugPrintOut("ERROR: Could not start graph\n");
         stopDevice(deviceID);
         return hr;
    }


    //MAKE SURE THE DEVICE IS SENDING VIDEO BEFORE WE FINISH
    if(!bCallback){

        long bufferSize = VD->videoSize;

        while( hr != S_OK){
            hr = VD->pGrabber->GetCurrentBuffer(&bufferSize, (long *)VD->pBuffer);
            Sleep(10);
        }

    }

    DebugPrintOut("SETUP: Device is setup and ready to capture.\n\n");
    VD->readyToCapture = true;

    // check for optional saving the direct show graph to a file
    const char* graph_filename = getenv("OPENCV_DSHOW_SAVEGRAPH_FILENAME");
    if (graph_filename) {
        size_t filename_len = strlen(graph_filename);
        std::vector<WCHAR> wfilename(filename_len + 1);
        size_t len = mbstowcs(&wfilename[0], graph_filename, filename_len  + 1);
        CV_Assert(len == filename_len);

        HRESULT res = SaveGraphFile(VD->pGraph, &wfilename[0]);
        if (SUCCEEDED(res)) {
            DebugPrintOut("Saved DSHOW graph to %s\n", graph_filename);
        } else {
            DebugPrintOut("Failed to save DSHOW graph to %s\n", graph_filename);
        }
    }

    //Release filters - seen someone else do this
    //looks like it solved the freezes

    //if we release this then we don't have access to the settings
    //we release our video input filter but then reconnect with it
    //each time we need to use it
#if 0
    VD->pVideoInputFilter->Release();
    VD->pVideoInputFilter = NULL;
#endif

    VD->pGrabberF->Release();
    VD->pGrabberF = NULL;

    VD->pDestFilter->Release();
    VD->pDestFilter = NULL;

    return S_OK;
}


// ----------------------------------------------------------------------
// Returns number of good devices
//
// ----------------------------------------------------------------------

int videoInput::getDeviceCount(){


    ICreateDevEnum *pDevEnum = NULL;
    IEnumMoniker *pEnum = NULL;
    int deviceCounter = 0;

    HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL,
        CLSCTX_INPROC_SERVER, IID_ICreateDevEnum,
        reinterpret_cast<void**>(&pDevEnum));


    if (SUCCEEDED(hr))
    {
        // Create an enumerator for the video capture category.
        hr = pDevEnum->CreateClassEnumerator(
            CLSID_VideoInputDeviceCategory,
            &pEnum, 0);

       if(hr == S_OK){
            IMoniker *pMoniker = NULL;
            while (pEnum->Next(1, &pMoniker, NULL) == S_OK){

                IPropertyBag *pPropBag;
                hr = pMoniker->BindToStorage(0, 0, IID_IPropertyBag,
                    (void**)(&pPropBag));

                if (FAILED(hr)){
                    pMoniker->Release();
                    continue;  // Skip this one, maybe the next one will work.
                }

                pPropBag->Release();
                pPropBag = NULL;

                pMoniker->Release();
                pMoniker = NULL;

                deviceCounter++;
            }

            pEnum->Release();
            pEnum = NULL;
        }

        pDevEnum->Release();
        pDevEnum = NULL;
    }
    return deviceCounter;
}


// ----------------------------------------------------------------------
// Do we need this?
//
// Enumerate all of the video input devices
// Return the filter with a matching friendly name
// ----------------------------------------------------------------------

HRESULT videoInput::getDevice(IBaseFilter** gottaFilter, int deviceId, WCHAR * wDeviceName, char * nDeviceName){
    BOOL done = false;
    int deviceCounter = 0;

    // Create the System Device Enumerator.
    ICreateDevEnum *pSysDevEnum = NULL;
    HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL, CLSCTX_INPROC_SERVER, IID_ICreateDevEnum, (void **)&pSysDevEnum);
    if (FAILED(hr))
    {
        return hr;
    }

    // Obtain a class enumerator for the video input category.
    IEnumMoniker *pEnumCat = NULL;
    hr = pSysDevEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &pEnumCat, 0);

    if (hr == S_OK)
    {
        // Enumerate the monikers.
        IMoniker *pMoniker = NULL;
        ULONG cFetched;
        while ((!done) && (pEnumCat->Next(1, &pMoniker, &cFetched) == S_OK))
        {
            if(deviceCounter == deviceId)
            {
                // Bind the first moniker to an object
                IPropertyBag *pPropBag;
                hr = pMoniker->BindToStorage(0, 0, IID_IPropertyBag, (void **)&pPropBag);
                if (SUCCEEDED(hr))
                {
                    // To retrieve the filter's friendly name, do the following:
                    VARIANT varName;
                    VariantInit(&varName);
                    hr = pPropBag->Read(L"FriendlyName", &varName, 0);
                    if (SUCCEEDED(hr))
                    {

                        //copy the name to nDeviceName & wDeviceName
                        int count = 0;
                        while( varName.bstrVal[count] != 0x00 ) {
                               wDeviceName[count] = varName.bstrVal[count];
                               nDeviceName[count] = (char)varName.bstrVal[count];
                               count++;
                         }

                        // reuse existing filter due to webcam problems
                        if (*gottaFilter)
                            hr = S_OK;
                        else
                            // We found it, so send it back to the caller
                            hr = pMoniker->BindToObject(NULL, NULL, IID_IBaseFilter, (void**)gottaFilter);
                        done = true;
                    }
                    VariantClear(&varName);
                    pPropBag->Release();
                    pPropBag = NULL;
                    pMoniker->Release();
                    pMoniker = NULL;
                }
            }
            else
            {
                // cleaning for the case when this isn't the device we are looking for
                pMoniker->Release();
                pMoniker = NULL;
            }
            deviceCounter++;
        }
        pEnumCat->Release();
        pEnumCat = NULL;
    }
    pSysDevEnum->Release();
    pSysDevEnum = NULL;

    if (done) {
        return hr;    // found it, return native error
    } else {
        return VFW_E_NOT_FOUND;    // didn't find it error
    }
}


// ----------------------------------------------------------------------
// Show the property pages for a filter
// This is stolen from the DX9 SDK
// ----------------------------------------------------------------------

HRESULT videoInput::ShowFilterPropertyPages(IBaseFilter *pFilter){

    ISpecifyPropertyPages *pProp;

    HRESULT hr = pFilter->QueryInterface(IID_ISpecifyPropertyPages, (void **)&pProp);
    if (SUCCEEDED(hr))
    {
        // Get the filter's name and IUnknown pointer.
        FILTER_INFO FilterInfo;
        hr = pFilter->QueryFilterInfo(&FilterInfo);
        IUnknown *pFilterUnk;
        pFilter->QueryInterface(IID_IUnknown, (void **)&pFilterUnk);

        // Show the page.
        CAUUID caGUID;
        pProp->GetPages(&caGUID);
        pProp->Release();
        OleCreatePropertyFrame(
            NULL,                   // Parent window
            0, 0,                   // Reserved
            FilterInfo.achName,     // Caption for the dialog box
            1,                      // Number of objects (just the filter)
            &pFilterUnk,            // Array of object pointers.
            caGUID.cElems,          // Number of property pages
            caGUID.pElems,          // Array of property page CLSIDs
            0,                      // Locale identifier
            0, NULL                 // Reserved
        );

        // Clean up.
        if(pFilterUnk)pFilterUnk->Release();
        if(FilterInfo.pGraph)FilterInfo.pGraph->Release();
        CoTaskMemFree(caGUID.pElems);
    }
    return hr;
}

HRESULT videoInput::ShowStreamPropertyPages(IAMStreamConfig  * /*pStream*/){

    HRESULT hr             = NOERROR;
    return hr;
}

// ----------------------------------------------------------------------
// This code was also brazenly stolen from the DX9 SDK
// Pass it a file name in wszPath, and it will save the filter graph to that file.
// ----------------------------------------------------------------------

HRESULT videoInput::SaveGraphFile(IGraphBuilder *pGraph, WCHAR *wszPath) {
    const WCHAR wszStreamName[] = L"ActiveMovieGraph";
    HRESULT hr;
    IStorage *pStorage = NULL;

    // First, create a document file which will hold the GRF file
    hr = StgCreateDocfile(
        wszPath,
        STGM_CREATE | STGM_TRANSACTED | STGM_READWRITE | STGM_SHARE_EXCLUSIVE,
        0, &pStorage);
    if(FAILED(hr))
    {
        return hr;
    }

    // Next, create a stream to store.
    IStream *pStream;
    hr = pStorage->CreateStream(
        wszStreamName,
        STGM_WRITE | STGM_CREATE | STGM_SHARE_EXCLUSIVE,
        0, 0, &pStream);
    if (FAILED(hr))
    {
        pStorage->Release();
        return hr;
    }

    // The IPersistStream converts a stream into a persistent object.
    IPersistStream *pPersist = NULL;
    pGraph->QueryInterface(IID_IPersistStream, reinterpret_cast<void**>(&pPersist));
    hr = pPersist->Save(pStream, TRUE);
    pStream->Release();
    pPersist->Release();
    if (SUCCEEDED(hr))
    {
        hr = pStorage->Commit(STGC_DEFAULT);
    }
    pStorage->Release();
    return hr;
}


// ----------------------------------------------------------------------
// For changing the input types
//
// ----------------------------------------------------------------------

HRESULT videoInput::routeCrossbar(ICaptureGraphBuilder2 **ppBuild, IBaseFilter **pVidInFilter, int conType, GUID captureMode){

    //create local ICaptureGraphBuilder2
    ICaptureGraphBuilder2 *pBuild = NULL;
     pBuild = *ppBuild;

     //create local IBaseFilter
     IBaseFilter *pVidFilter = NULL;
     pVidFilter = * pVidInFilter;

    // Search upstream for a crossbar.
    IAMCrossbar *pXBar1 = NULL;
    HRESULT hr = pBuild->FindInterface(&LOOK_UPSTREAM_ONLY, NULL, pVidFilter,
            IID_IAMCrossbar, (void**)&pXBar1);
    if (SUCCEEDED(hr) && pXBar1)
    {

        bool foundDevice = false;

        DebugPrintOut("SETUP: You are not a webcam! Setting Crossbar\n");
        pXBar1->Release();

        IAMCrossbar *Crossbar;
        hr = pBuild->FindInterface(&captureMode, &MEDIATYPE_Interleaved, pVidFilter, IID_IAMCrossbar, (void **)&Crossbar);

        if(hr != NOERROR){
            hr = pBuild->FindInterface(&captureMode, &MEDIATYPE_Video, pVidFilter, IID_IAMCrossbar, (void **)&Crossbar);
        }

        LONG lInpin, lOutpin;
        hr = Crossbar->get_PinCounts(&lOutpin , &lInpin);

        BOOL iPin=TRUE; LONG pIndex=0 , pRIndex=0 , pType=0;

        while( pIndex < lInpin)
        {
            hr = Crossbar->get_CrossbarPinInfo( iPin , pIndex , &pRIndex , &pType);

            if( pType == conType){
                    DebugPrintOut("SETUP: Found Physical Interface");

                    switch(conType){

                        case PhysConn_Video_Composite:
                            DebugPrintOut(" - Composite\n");
                            break;
                        case PhysConn_Video_SVideo:
                            DebugPrintOut(" - S-Video\n");
                            break;
                        case PhysConn_Video_Tuner:
                            DebugPrintOut(" - Tuner\n");
                            break;
                        case PhysConn_Video_USB:
                            DebugPrintOut(" - USB\n");
                            break;
                        case PhysConn_Video_1394:
                            DebugPrintOut(" - Firewire\n");
                            break;
                    }

                foundDevice = true;
                break;
            }
            pIndex++;

        }

        if(foundDevice){
            BOOL OPin=FALSE; LONG pOIndex=0 , pORIndex=0 , pOType=0;
            while( pOIndex < lOutpin)
            {
                hr = Crossbar->get_CrossbarPinInfo( OPin , pOIndex , &pORIndex , &pOType);
                if( pOType == PhysConn_Video_VideoDecoder)
                    break;
            }
            Crossbar->Route(pOIndex,pIndex);
        }else{
            DebugPrintOut("SETUP: Didn't find specified Physical Connection type. Using Default.\n");
        }

        //we only free the crossbar when we close or restart the device
        //we were getting a crash otherwise
        //if(Crossbar)Crossbar->Release();
        //if(Crossbar)Crossbar = NULL;
    }else{
        DebugPrintOut("SETUP: You are a webcam or snazzy firewire cam! No Crossbar needed\n");
        return hr;
    }

    return hr;
}

int videoInput::property_window_count(int idx)
{
    if (isDeviceSetup(idx))
        return (int)InterlockedCompareExchange(&VDList[idx]->property_window_count, 0L, 0L);

    return 0;
}

namespace cv
{
videoInput VideoCapture_DShow::g_VI;

VideoCapture_DShow::VideoCapture_DShow(int index)
    : m_index(-1)
    , m_width(-1)
    , m_height(-1)
    , m_fourcc(-1)
    , m_widthSet(-1)
    , m_heightSet(-1)
{
    CoInitialize(0);
    open(index);
}
VideoCapture_DShow::~VideoCapture_DShow()
{
    close();
    CoUninitialize();
}

double VideoCapture_DShow::getProperty(int propIdx) const
{

    long min_value, max_value, stepping_delta, current_value, flags, defaultValue;

    switch (propIdx)
    {
    // image format properties
    case CV_CAP_PROP_FRAME_WIDTH:
        return g_VI.getWidth(m_index);
    case CV_CAP_PROP_FRAME_HEIGHT:
        return g_VI.getHeight(m_index);
    case CV_CAP_PROP_FOURCC:
        return g_VI.getFourcc(m_index);
    case CV_CAP_PROP_FPS:
        return g_VI.getFPS(m_index);
    case CV_CAP_PROP_CONVERT_RGB:
        return g_VI.getConvertRGB(m_index);
    case CAP_PROP_CHANNEL:
        return g_VI.getChannel(m_index);
    case CV_CAP_PROP_AUTOFOCUS:
      // Flags indicate whether or not autofocus is enabled
      if (g_VI.getVideoSettingCamera(m_index, CameraControl_Focus, min_value, max_value, stepping_delta, current_value, flags, defaultValue))
        return (double)flags;
      break;

    // video filter properties
    case CV_CAP_PROP_BRIGHTNESS:
    case CV_CAP_PROP_CONTRAST:
    case CV_CAP_PROP_HUE:
    case CV_CAP_PROP_SATURATION:
    case CV_CAP_PROP_SHARPNESS:
    case CV_CAP_PROP_GAMMA:
    case CV_CAP_PROP_MONOCHROME:
    case CV_CAP_PROP_WHITE_BALANCE_BLUE_U:
    case CV_CAP_PROP_BACKLIGHT:
    case CV_CAP_PROP_GAIN:
        if (g_VI.getVideoSettingFilter(m_index, g_VI.getVideoPropertyFromCV(propIdx), min_value, max_value, stepping_delta, current_value, flags, defaultValue))
            return (double)current_value;
        break;

    // camera properties
    case CV_CAP_PROP_PAN:
    case CV_CAP_PROP_TILT:
    case CV_CAP_PROP_ROLL:
    case CV_CAP_PROP_ZOOM:
    case CV_CAP_PROP_EXPOSURE:
    case CV_CAP_PROP_IRIS:
    case CV_CAP_PROP_FOCUS:
        if (g_VI.getVideoSettingCamera(m_index, g_VI.getCameraPropertyFromCV(propIdx), min_value, max_value, stepping_delta, current_value, flags, defaultValue))
            return (double)current_value;
        break;
    case CV_CAP_PROP_SETTINGS:
        return g_VI.property_window_count(m_index);
    default:
        break;
    }
    // unknown parameter or value not available
    return -1;
}
bool VideoCapture_DShow::setProperty(int propIdx, double propVal)
{
    // image capture properties
    bool handled = false;
    switch (propIdx)
    {
    case CV_CAP_PROP_FRAME_WIDTH:
        m_width = cvRound(propVal);
        handled = true;
        break;

    case CV_CAP_PROP_FRAME_HEIGHT:
        m_height = cvRound(propVal);
        handled = true;
        break;

    case CV_CAP_PROP_FOURCC:
        m_fourcc = (int)(unsigned long)(propVal);
        m_width = (int)getProperty(CAP_PROP_FRAME_WIDTH);
        m_height = (int)getProperty(CAP_PROP_FRAME_HEIGHT);

        if (-1 == m_fourcc)
        {
            // following cvCreateVideo usage will pop up caprturepindialog here if fourcc=-1
            // TODO - how to create a capture pin dialog
        }
        else
        {
            handled = true;
        }

        break;

    case CV_CAP_PROP_FPS:
    {
        int fps = cvRound(propVal);
        if (fps != g_VI.getFPS(m_index))
        {
            g_VI.stopDevice(m_index);
            g_VI.setIdealFramerate(m_index, fps);
            if (m_widthSet > 0 && m_heightSet > 0)
                g_VI.setupDevice(m_index, m_widthSet, m_heightSet);
            else
                g_VI.setupDevice(m_index);
        }
        return g_VI.isDeviceSetup(m_index);
    }

    case CV_CAP_PROP_AUTOFOCUS:
    {
        // Flags are required to toggle autofocus or not, but the setProperty interface does not support multiple parameters
        bool enabled = cvRound(propVal) == 1;
        long minFocus, maxFocus, delta, currentFocus, flags, defaultValue;
        if (!g_VI.getVideoSettingCamera(m_index, CameraControl_Focus, minFocus, maxFocus, delta, currentFocus, flags, defaultValue))
        {
            return false;
        }
        return g_VI.setVideoSettingCamera(m_index, CameraControl_Focus, currentFocus, enabled ? CameraControl_Flags_Auto | CameraControl_Flags_Manual : CameraControl_Flags_Manual, enabled ? true : false);
    }

    case CV_CAP_PROP_CONVERT_RGB:
    {
        return g_VI.setConvertRGB(m_index, cvRound(propVal) == 1);
    }

    }

    if (handled)
    {
        // a stream setting
        if (m_width > 0 && m_height > 0)
        {
            if (m_width != g_VI.getWidth(m_index) || m_height != g_VI.getHeight(m_index) || m_fourcc != g_VI.getFourcc(m_index) )
            {
                int fps = static_cast<int>(g_VI.getFPS(m_index));
                g_VI.stopDevice(m_index);
                g_VI.setIdealFramerate(m_index, fps);
                g_VI.setupDeviceFourcc(m_index, m_width, m_height, m_fourcc);
            }

            bool success = g_VI.isDeviceSetup(m_index);
            if (success)
            {
                DebugPrintOut("SETUP: Updated FourCC\n");
                m_widthSet = m_width;
                m_heightSet = m_height;
                m_width = m_height = m_fourcc = -1;
            }
            else {
                DebugPrintOut("SETUP: Couldn't update FourCC\n");
            }
            return success;
        }
        return true;
    }

    //video Filter properties
    switch (propIdx)
    {
    case CV_CAP_PROP_BRIGHTNESS:
    case CV_CAP_PROP_CONTRAST:
    case CV_CAP_PROP_HUE:
    case CV_CAP_PROP_SATURATION:
    case CV_CAP_PROP_SHARPNESS:
    case CV_CAP_PROP_GAMMA:
    case CV_CAP_PROP_MONOCHROME:
    case CV_CAP_PROP_WHITE_BALANCE_BLUE_U:
    case CV_CAP_PROP_BACKLIGHT:
    case CV_CAP_PROP_GAIN:
        return g_VI.setVideoSettingFilter(m_index, g_VI.getVideoPropertyFromCV(propIdx), (long)propVal);
    }

    //camera properties
    switch (propIdx)
    {
    case CV_CAP_PROP_PAN:
    case CV_CAP_PROP_TILT:
    case CV_CAP_PROP_ROLL:
    case CV_CAP_PROP_ZOOM:
    case CV_CAP_PROP_EXPOSURE:
    case CV_CAP_PROP_IRIS:
    case CV_CAP_PROP_FOCUS:
        return g_VI.setVideoSettingCamera(m_index, g_VI.getCameraPropertyFromCV(propIdx), (long)propVal);
    // show video/camera filter dialog
    case CV_CAP_PROP_SETTINGS:
        return g_VI.showSettingsWindow(m_index);
    }

    return false;
}

bool VideoCapture_DShow::grabFrame()
{
    return !g_VI.isDeviceDisconnected(m_index);
}
bool VideoCapture_DShow::retrieveFrame(int, OutputArray frame)
{
    int w = g_VI.getWidth(m_index), h = g_VI.getHeight(m_index);
    bool convertRGB = g_VI.getConvertRGB(m_index);

    // Set suitable output matrix type (e-Con systems)
    if (checkSingleByteFormat(g_VI.getMediasubtype(m_index))){
        frame.create(Size(w, h), CV_8UC1);
    } else if (g_VI.getMediasubtype(m_index) == MEDIASUBTYPE_Y16 && !convertRGB) {
        frame.create(Size(w, h), CV_16UC1);
    } else {
        frame.create(Size(w, h), CV_8UC3);
    }

    cv::Mat mat = frame.getMat();
    return g_VI.getPixels(m_index, mat.ptr(), false, true );
}
int VideoCapture_DShow::getCaptureDomain()
{
    return CV_CAP_DSHOW;
}
bool VideoCapture_DShow::isOpened() const
{
    return (-1 != m_index);
}

void VideoCapture_DShow::open(int index)
{
    close();
    int devices = g_VI.listDevices(true);
    if (0 == devices)
        return;
    if (index < 0 || index > devices-1)
        return;
    g_VI.setupDevice(index);
    if (!g_VI.isDeviceSetup(index))
        return;
    m_index = index;
}

void VideoCapture_DShow::close()
{
    if (m_index >= 0)
    {
        g_VI.stopDevice(m_index);
        m_index = -1;
    }
    m_widthSet = m_heightSet = m_width = m_height = -1;
}

}

#endif
