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
    int                     openedCameraHandle = -1;            // The handle of the camera which has been opened.  -1 indicates that no camera has been opened.
    uint32_t                interfaceIPAddress = 0;
};
}

#endif  // HAVE_RTX64_GIGE
#endif // CAP_RTX64GIGE