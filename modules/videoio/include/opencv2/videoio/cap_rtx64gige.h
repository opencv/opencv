// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef CAP_RTX64GIGE
#define CAP_RTX64GIGE
#ifdef HAVE_RTX64_GIGE

#include "RTX_GigEV_Camera_Controller.h"

#define MAX_DRIVER_EVENT_HANDLES            3
#define GIGEV_EVENT_0_DRIVERSTOPPED         0
#define GIGEV_EVENT_1_GRAB_STARTED          1
#define GIGEV_EVENT_2_GRAB_COMPLETE         2

#define CAMERA_PORT                         8888
#define FRAMERATE                           5
#define MAX_BYTES_PER_PIXEL                 4
#define GRAB_TIMEOUT                        1000

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

    //
    //  Converts a MAC Address to a string.
    //
    //  @param pStrMac pointer to the resultant string
    //  @param pMacAddr pointer to the MAC Address to be converted
    //
    void macToString(char * pStrMAC, unsigned char *pMacAddr);

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

#ifdef NOT_READY_YET
        bool CvCaptureCAM_Rtx64GigE::retrieveFrameEx(int index, RTX_Image ** pImageObj);

        void ReturnCameraList(void);
#endif // NOT_READY_YET

        // The grabbed frame, which is set once grabFrame() returns (will point to the same address as m_pActiveFrame)
        RTX_Image *pGrabbedFrameObj = NULL;

        // Pointer to the Camera Info for the GigE camera being used.
        RTX_GigEV_Camera_Info *CI_Ptr = NULL;

        //  The IP Address of the RTX64 NIC
        uint32_t m_IP_Address_Ours = 0;

        //  The Camera Controller object for the GigE camera being used.
        RTX_GigEV_Camera_Controller *m_Camera_Controller = NULL;

    protected:

        //  Image Bits per pixel
        const uint32_t IMAGE_BITS_PER_PIXEL = 16;
        //  Image Height
        const uint32_t IMAGE_HEIGHT = 1200;
        // Image Width
        const uint32_t IMAGE_WIDTH = 1920;
        // Image Number of Channels (3 for RGB/BGR/etc.)
        const uint32_t IMAGE_NUM_CHANNELS = 3;

        //  Initializes m_Camera_Controller.  Called by the Constructor.
        HRESULT InitializeCameraController(uint32_t IP_ADDRESS_OURS);

        //  Index into camera list for open camera -1 indicates no camera has been opened
        int m_OpenCameraIndex = -1;

        // The Active Frame
        RTX_Image *m_pActiveFrame = NULL;
    };
}

#endif  // HAVE_RTX64_GIGE
#endif // CAP_RTX64GIGE