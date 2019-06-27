// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "RtGVApi.h"

#ifndef CAP_RTX64GIGE
#define CAP_RTX64GIGE
#ifdef HAVE_RTX64_GIGE

#define MAX_DRIVER_EVENT_HANDLES            3
#define GIGEV_EVENT_0_DRIVERSTOPPED         0
#define GIGEV_EVENT_1_GRAB_STARTED          1
#define GIGEV_EVENT_2_GRAB_COMPLETE         2

#define CAMERA_PORT                         8888
#define FRAMERATE                           5
#define MAX_BYTES_PER_PIXEL                 4
#define GRAB_TIMEOUT                        1000
#define DISCOVERY_PORT                      3500
#define CONTROL_PORT                        3501
#define STREAM_PORT                         4500
#define HEARTBEAT_TIMEOUT_MS                10000
#define HEARTBEATS_PER_PERIOD               3

#pragma pack(push, 1)
#pragma pack(show)


#pragma pack(pop)
#pragma pack(show)

namespace cv
{

//
//  Gets the offset of the Image Data.  This is for use when sharing Images between RTSS and Windows, using RtGVComm.lib.
//
unsigned int getImageDataOffset(void);

class Rtx64GigECapture : public IVideoCapture
{
public:
    //
    // Rtx64 GigE Capture Class constructor
    //
    Rtx64GigECapture();

    //
    // Rtx64 GigE Capture Class constructor calls open with given index
    //
    Rtx64GigECapture(int index);

    //
    // Rtx64GigE Capture Class Destructor
    //
    ~Rtx64GigECapture()
    {
        close();
    };

    //
    // Initializes the Rtx64GigECapture object. Opens the connection with the camera. This is called by cvCreateCameraCapture_Rtx64GigE.
    //
    //  int index                       - Index of the camera to be opened
    //
    virtual bool open(int index);

    //
    // Closes the connection with the camera.  This is called by both the Destructor and release().
    //
    virtual void close();

    //
    // Grabs a frame from the camera
    //
    virtual bool grabFrame();

    //
    // Retrieves a frame which was grabbed through grabFrame().
    //
    //  int flag                        - Currently performs no function
    //  OutputArray image               - The buffer given by the customer which will receive the image
    //
    virtual bool retrieveFrame(int flag, OutputArray image);

    //
    // Checks whether or not the Rtx64GigECapture object has been opened
    //
    virtual bool isOpened() const;

    //
    // Gets the capture domain of the Rtx64GigECapture object (CAP_RTX64_GIGE).
    //
    virtual int getCaptureDomain();

private:
    DWORD                   numCamerasDiscovered = 0;           // Holds the number of cameras discovered on the network
    PRTGV_CAMERA_INFOA      pCameraInfos = NULL;                // Pointer to RTGV_CAMERA_INFO for receiving camera info from RtGVEnumerateCameras
    ACQUISITIONMODE         acquisitionMode;                    // Holds the acquisition mode being used by the camera
    PRTGV_FRAME             pActiveFrame;                       // Pointer to the RTGV_FRAME which will receive frames from RtGVGrabFrame
    int                     openedCameraIndex = -1;             // The index in the camera list of the camera which has been opened.  -1 indicates that no camera has been opened.
    uint32_t                interfaceIPAddress = 0;
};

//
//  Capture class for use with RTX64 GigE
//
class CvCaptureCAM_Rtx64GigE : public CvCapture
{
public:

    //
    //  Constructor for RTX64 GigE Capture Class.  This is called by cvCreateCameraCapture_Rtx64GigE.
    //
    CvCaptureCAM_Rtx64GigE();

    //
    //  Destructor for RTX64 GigE Capture Class
    //
    virtual ~CvCaptureCAM_Rtx64GigE()
    {
        close();
    }

    //
    //  Opens the connection with the camera.  This is called by cvCreateCameraCapture_Rtx64GigE.
    //
    virtual bool open(int index);

    //
    //  Closes the connection with the camera.  This is called by both the Destructor and release().
    //
    virtual void close();

#ifdef NOT_READY_YET
        virtual double getProperty(int) const;
        virtual bool setProperty(int, double);
#endif // NOT_READY_YET

    //
    //  Grabs a frame from the camera
    //
    virtual bool grabFrame();

    //
    //  Retrieves the grabbed frame from the camera and returns it in the IplImage format.
    //
    //  @param index the index of the camera from which the frame should be retrieved
    //
    virtual IplImage* retrieveFrame(int index);

    //
    //  Releases the resources held by the RTX64 GigE Capture class.
    //
    void release(void);

    uint32_t                interfaceIPAddress = 0;             //  The IP Address of the RTX64 NIC

protected:
    //  Image Bits per pixel
    const uint32_t IMAGE_BITS_PER_PIXEL = 16;
    //  Image Height
    const uint32_t IMAGE_HEIGHT = 1200;
    // Image Width
    const uint32_t IMAGE_WIDTH = 1920;
    // Image Number of Channels (3 for RGB/BGR/etc.)
    const uint32_t IMAGE_NUM_CHANNELS = 3;

    DWORD                   numCamerasDiscovered = 0;           // The number of cameras discovered
    bool                    streamStarted = false;              // Tracks whether or not a stream has been started
    int                     openedCameraIndex = -1;             // Index into camera list for open camera -1 indicates no camera has been opened
    PRTGV_FRAME             pActiveFrame = NULL;                // The Active Frame
    IplImage                *pIplImageColorConverted;           // Output IplImage with color conversion
    cv::Mat                 mImageSrc;
    cv::Mat                 mImageDest;
    PRTGV_CAMERA_INFOA      pCameraInfo = NULL;                 // Pointer to an RTGV_CAMERA_INFO structure for receiving camera info from RtGVEnumerateCameras
    ACQUISITIONMODE         acquisitionMode;
};
}

#endif  // HAVE_RTX64_GIGE
#endif // CAP_RTX64GIGE