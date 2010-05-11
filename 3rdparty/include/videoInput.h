#ifndef _VIDEOINPUT
#define _VIDEOINPUT

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
//														//
//Warning This code is experimental 					//
//use at your own risk :)								//
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



#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <wchar.h>

//this is for TryEnterCriticalSection
#ifndef _WIN32_WINNT
	#   define _WIN32_WINNT 0x400
#endif
#include <windows.h>


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
	
	VI.setupDevice(device1); 						  //setup the first device with the default settings
	//VI.setupDevice(device1, VI_COMPOSITE); 			  //or setup device with specific connection type
	//VI.setupDevice(device1, 320, 240);				  //or setup device with specified video size
	//VI.setupDevice(device1, 320, 240, VI_COMPOSITE);  //or setup device with video size and connection type

	//VI.setFormat(device1, VI_NTSC_M);					//if your card doesn't remember what format it should be
														//call this with the appropriate format listed above
														//NOTE: must be called after setupDevice!
	
	//optionally setup a second (or third, fourth ...) device - same options as above
	VI.setupDevice(device2); 						  

	//As requested width and height can not always be accomodated
	//make sure to check the size once the device is setup

	int width 	= VI.getWidth(device1);
	int height 	= VI.getHeight(device1);
	int size	= VI.getSize(device1);
	
	unsigned char * yourBuffer1 = new unsigned char[size];
	unsigned char * yourBuffer2 = new unsigned char[size];
	
	//to get the data from the device first check if the data is new
	if(VI.isFrameNew(device1)){
		VI.getPixels(device1, yourBuffer1, false, false);	//fills pixels as a BGR (for openCV) unsigned char array - no flipping
		VI.getPixels(device1, yourBuffer2, true, true); 	//fills pixels as a RGB (for openGL) unsigned char array - flipping!
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

//change for verbose debug info
static bool verbose = true;

//if you need VI to use multi threaded com
//#define VI_COM_MULTI_THREADED

//STUFF YOU DON'T CHANGE

//videoInput defines
#define VI_VERSION	 0.1995
#define VI_MAX_CAMERAS  20
#define VI_NUM_TYPES    18 //DON'T TOUCH
#define VI_NUM_FORMATS  18 //DON'T TOUCH

//defines for setPhyCon - tuner is not as well supported as composite and s-video 
#define VI_COMPOSITE 0
#define VI_S_VIDEO   1
#define VI_TUNER     2
#define VI_USB       3
#define VI_1394		 4

//defines for formats
#define VI_NTSC_M	0
#define VI_PAL_B	1
#define VI_PAL_D	2
#define VI_PAL_G	3
#define VI_PAL_H	4
#define VI_PAL_I	5
#define VI_PAL_M	6
#define VI_PAL_N	7
#define VI_PAL_NC	8
#define VI_SECAM_B	9
#define VI_SECAM_D	10
#define VI_SECAM_G	11
#define VI_SECAM_H	12
#define VI_SECAM_K	13
#define VI_SECAM_K1	14
#define VI_SECAM_L	15
#define VI_NTSC_M_J	16
#define VI_NTSC_433	17


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
static int comInitCount = 0;


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
		
		ICaptureGraphBuilder2 *pCaptureGraph;	// Capture graph builder object
		IGraphBuilder *pGraph;					// Graph builder object
	    IMediaControl *pControl;				// Media control object
		IBaseFilter *pVideoInputFilter;  		// Video Capture filter
		IBaseFilter *pGrabberF;
		IBaseFilter * pDestFilter;
		IAMStreamConfig *streamConf;
		ISampleGrabber * pGrabber;    			// Grabs frame
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
		int  nFramesForReconnect;
		unsigned long nFramesRunning;
		int  connection;
		int	 storeConn;
		int  myID;
		long requestedFrameTime; //ie fps
		
		char 	nDeviceName[255];
		WCHAR 	wDeviceName[255];
		
		unsigned char * pixels;
		char * pBuffer;

};




//////////////////////////////////////   VIDEO INPUT   /////////////////////////////////////



class videoInput{

	public:
		videoInput();
		~videoInput();
				
		//turns off console messages - default is to print messages
		static void setVerbose(bool _verbose);
		
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
		
		//Choose one of these four to setup your device
		bool setupDevice(int deviceID);
		bool setupDevice(int deviceID, int w, int h);

		//These two are only for capture cards
		//USB and Firewire cameras souldn't specify connection 
		bool setupDevice(int deviceID, int connection);	
		bool setupDevice(int deviceID, int w, int h, int connection); 
		
		//If you need to you can set your NTSC/PAL/SECAM
		//preference here. if it is available it will be used.
		//see #defines above for available formats - eg VI_NTSC_M or VI_PAL_B
		//should be called after setupDevice
		//can be called multiple times
		bool setFormat(int deviceNumber, int format);	
				
		//Tells you when a new frame has arrived - you should call this if you have specified setAutoReconnectOnFreeze to true
		bool isFrameNew(int deviceID); 
		
		bool isDeviceSetup(int deviceID);
		    
		//Returns the pixels - flipRedAndBlue toggles RGB/BGR flipping - and you can flip the image too
		unsigned char * getPixels(int deviceID, bool flipRedAndBlue = true, bool flipImage = false);
		
		//Or pass in a buffer for getPixels to fill returns true if successful.
		bool getPixels(int id, unsigned char * pixels, bool flipRedAndBlue = true, bool flipImage = false);
		
		//Launches a pop up settings window
		//For some reason in GLUT you have to call it twice each time. 
		void showSettingsWindow(int deviceID);
		
		//Manual control over settings thanks..... 
		//These are experimental for now.
		bool setVideoSettingFilter(int deviceID, long Property, long lValue, long Flags = NULL, bool useDefaultValue = false);
		bool setVideoSettingFilterPct(int deviceID, long Property, float pctValue, long Flags = NULL);
		bool getVideoSettingFilter(int deviceID, long Property, long &min, long &max, long &SteppingDelta, long &currentValue, long &flags, long &defaultValue);

		bool setVideoSettingCamera(int deviceID, long Property, long lValue, long Flags = NULL, bool useDefaultValue = false);
		bool setVideoSettingCameraPct(int deviceID, long Property, float pctValue, long Flags = NULL);
		bool getVideoSettingCamera(int deviceID, long Property, long &min, long &max, long &SteppingDelta, long &currentValue, long &flags, long &defaultValue);

		//bool setVideoSettingCam(int deviceID, long Property, long lValue, long Flags = NULL, bool useDefaultValue = false);

		//get width, height and number of pixels
		int  getWidth(int deviceID);
		int  getHeight(int deviceID);
		int  getSize(int deviceID);
		
		//completely stops and frees a device
		void stopDevice(int deviceID);
		
		//as above but then sets it up with same settings
		bool restartDevice(int deviceID);
		
		//number of devices available
		int  devicesFound;
		
		long propBrightness;
		long propContrast;
		long propHue;
		long propSaturation;
		long propSharpness;
		long propGamma;
		long propColorEnable;
		long propWhiteBalance;
		long propBacklightCompensation;
		long propGain;

		long propPan;
		long propTilt;
		long propRoll;
		long propZoom;
		long propExposure;
		long propIris;
		long propFocus;
				
		
	private:		
		void setPhyCon(int deviceID, int conn);                   
		void setAttemptCaptureSize(int deviceID, int w, int h);   
		bool setup(int deviceID);
		void processPixels(unsigned char * src, unsigned char * dst, int width, int height, bool bRGB, bool bFlip);
		int  start(int deviceID, videoDevice * VD);                   
		int  getDeviceCount();
		void getMediaSubtypeAsString(GUID type, char * typeAsString);
		
		HRESULT getDevice(IBaseFilter **pSrcFilter, int deviceID, WCHAR * wDeviceName, char * nDeviceName);
		static HRESULT ShowFilterPropertyPages(IBaseFilter *pFilter);
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
		GUID MEDIASUBTYPE_Y800;
		GUID MEDIASUBTYPE_Y8;
		GUID MEDIASUBTYPE_GREY;

		videoDevice * VDList[VI_MAX_CAMERAS];
		GUID mediaSubtypes[VI_NUM_TYPES];
		long formatTypes[VI_NUM_FORMATS];

		static void __cdecl basicThread(void * objPtr);

		static char deviceNames[VI_MAX_CAMERAS][255];

}; 
  
 #endif
