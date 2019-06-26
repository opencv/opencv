// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// $$$NOW: OpenCV code must change per RtGigE changes (special focus on bitsPerPixel vs pixelDepth)

#ifdef HAVE_RTX64_GIGE

#include "precomp.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include <SDKDDKVer.h>

// for RTTCPIP Support
//#include <winsock2.h>
//#include <ws2tcpip.h>
#include <windows.h>
#include <tchar.h>
#include <rtapi.h>    // RTX64 APIs that can be used in real-time and Windows applications.
#include <stdint.h>

#ifdef UNDER_RTSS
#include <rtssapi.h>  // RTX64 APIs that can only be used in real-time applications.
#include <rtnapi.h>
#endif // UNDER_RTSS

//#include "rtGigEDev.h"
#include "RtGVDataTypes.h"

#include "cap_rtx64gige.h"

namespace cv
{

Ptr<IVideoCapture> createCameraCapture_Rtx64GigE(int index)
{
    return makePtr<Rtx64GigECapture>(index);
}

unsigned int getImageDataOffset(void)
{
    unsigned int structureSize = sizeof(RTGV_FRAME) + sizeof(IplImage);
    return(ROUNDUP(structureSize, NODE_ALIGNMENT_SIZE));
}

Rtx64GigECapture::Rtx64GigECapture()
{
    // Determine IP Addresses to be used to communicate with the camera (some of this could also be done later)
    char *NIC_Name = "RtGigE_01_Nic";
    void *ndp = RtnGetDevicePtr(NIC_Name, DEVICE_NAME);
    interfaceIPAddress = htonl(RtnGetIpAddress(ndp));

    //Initialize RtGigEVision
    if (!RtGVInitialize(INADDR_ANY, DISCOVERY_PORT))
    {
        throw "Error initializing RtGigE Vision!\n";
    }

    // Enumerate Cameras
    DWORD numElements = 0;
    RtGVEnumerateCamerasA(NULL, numElements, &numCamerasDiscovered);

    numElements = numCamerasDiscovered;

    pCameraInfos = new RTGV_CAMERA_INFOA[numElements];
    for (int i = 0; i < numElements; i++)
    {
        pCameraInfos[i].size = sizeof(RTGV_CAMERA_INFOA);
    }

    if (!RtGVEnumerateCamerasA(pCameraInfos, numElements, &numCamerasDiscovered))
    {
        throw "Camera enumeration failed!\n";
    }

    for (uint32_t i = 0; i < numCamerasDiscovered; i++)
    {
        RtPrintf("Camera %u:\n", i);
        RtPrintf("\t handle: %u\n", pCameraInfos[i].cameraHandle);
        RtPrintf("\t manufacturer name: %s\n", pCameraInfos[i].manufacturerName);
        RtPrintf("\t model name: %s\n", pCameraInfos[i].modelName);
        RtPrintf("\t device version: %s\n", pCameraInfos[i].deviceVersion);
    }
}

Rtx64GigECapture::Rtx64GigECapture(int index)
    : Rtx64GigECapture()
{
    this->open(index);
}

bool Rtx64GigECapture::open(int index)
{
    uint32_t maxWidth = 0, maxHeight = 0, width = 0, height = 0, numChannels = 0, pixelDepth = 0, payloadSize = 0;
    GVSP_PIXEL_TYPE pixelFormat;

    // Initialize camera control
    if (!RtGVInitCameraControl(0, interfaceIPAddress, CONTROL_PORT + index, HEARTBEAT_TIMEOUT_MS, HEARTBEATS_PER_PERIOD, TRUE))
    {
        RtPrintf("RtGVInitCameraControl Failed! Error: %d\n", GetLastError());
        return false;
    }

    // Get current camera settings
    if (!RtGVGetCameraMaxWidth(index, &maxWidth))
    {
        RtPrintf("RtGVGetCameraMaxWidth Failed! Error: %d\n", GetLastError());
        RtGVCloseCameraControl(index);
        return false;
    }

    if (!RtGVGetCameraMaxHeight(index, &maxHeight))
    {
        RtPrintf("RtGVGetCameraMaxHeight Failed! Error: %d\n", GetLastError());
        RtGVCloseCameraControl(index);
        return false;
    }

    if (!RtGVGetCameraWidth(index, &width))
    {
        RtPrintf("RtGVGetCameraWidth Failed! Error: %d\n", GetLastError());
        RtGVCloseCameraControl(index);
        return false;
    }
    if (!RtGVGetCameraHeight(index, &height))
    {
        RtPrintf("RtGVGetCameraHeight Failed! Error: %d\n", GetLastError());
        RtGVCloseCameraControl(index);
        return false;
    }

    if (!RtGVGetCameraNumberOfChannels(index, &numChannels))
    {
        RtPrintf("RtGVGetCameraNumberOfChannels Failed! Error: %d\n", GetLastError());
        RtGVCloseCameraControl(index);
        return false;
    }

    if (!RtGVGetCameraPixelDepth(index, &pixelDepth))
    {
        RtPrintf("RtGVGetCameraPixelDepth Failed! Error: %d\n", GetLastError());
        RtGVCloseCameraControl(index);
        return false;
    }

    if (!RtGVGetCameraPixelFormat(index, &pixelFormat))
    {
        RtPrintf("RtGVGetCameraPixelFormat Failed! Error: %d\n", GetLastError());
        RtGVCloseCameraControl(index);
        return false;
    }

    if (!RtGVGetCameraAcquisitionMode(index, &acquisitionMode))
    {
        RtPrintf("RtGVGetCameraAcquisitionMode failed on camera %d. Error: %d\n", index, GetLastError());
        RtGVCloseCameraControl(index);
        return false;
    }

    if (invalidAcquisitionMode == acquisitionMode)
    {
        RtPrintf("Invalid acquisition mode configured on camera %d!\n", index);
        RtGVCloseCameraControl(index);
        return false;
    }

    if (!RtGVGetCameraPayloadSize(index, &payloadSize))
    {
        RtPrintf("RtGVGetCameraPayloadSize failed on camera %d. Error: %d\n", index, GetLastError());
        return false;
    }
    
    // Allocate/Initialize RTGV_FRAME buffer using the camera settings
    pActiveFrame = (PRTGV_FRAME)malloc(sizeof(RTGV_FRAME) + payloadSize);
    pActiveFrame->size = sizeof(RTGV_FRAME);
    pActiveFrame->pixelDepth = 0;
    pActiveFrame->pixelFormat = 0;
    pActiveFrame->numberOfChannels = 0;
    pActiveFrame->offsetX = 0;
    pActiveFrame->offsetY = 0;
    pActiveFrame->width = 0;
    pActiveFrame->height = 0;
    pActiveFrame->paddingX = 0;
    pActiveFrame->paddingY = 0;
    pActiveFrame->frameSize = 0;
    pActiveFrame->offsetToImageData = sizeof(RTGV_FRAME) + sizeof(IplImage);
    pActiveFrame->pCustomMetaData = ((char *)pActiveFrame) + sizeof(RTGV_FRAME);
    pActiveFrame->imageData = ((char *)pActiveFrame) + pActiveFrame->offsetToImageData;

    // Start the stream
    if (!RtGVStartStream(index, interfaceIPAddress, STREAM_PORT + index, 50))
    {
        free(pActiveFrame);
        RtGVCloseCameraControl(index);
        return false;
    }

    openedCameraIndex = index;

    return true;
}

void Rtx64GigECapture::close()
{
    // Free the RTGV_FRAME buffer
    if (NULL != pActiveFrame)
    {
        free(pActiveFrame);
    }

    // Check if we need to close the stream and control channels
    if (-1 != openedCameraIndex)
    {
        // Close the stream
        RtGVCloseStream(openedCameraIndex);

        // Close Camera Control
        RtGVCloseCameraControl(openedCameraIndex);
    }

    // Close RtGigEVision
    RtGVClose();
}

bool Rtx64GigECapture::grabFrame()
{
    // Check if camera is configured for trigger mode, and if it is, send a trigger
    if (softwareTriggeredAcquisitionMode == acquisitionMode)
    {
        if (!RtGVSendTrigger(openedCameraIndex))
        {
            return false;
        }
    }

    // Attempt to grab a frame using RtGVGrabFrame
    int errorCount = 0;
    while (!RtGVGrabFrame(openedCameraIndex, pActiveFrame, GRAB_TIMEOUT))
    {
        errorCount++;
        // If we are grabbing faster than the camera is sending images, then we will hit RTGV_ERROR_NO_IMAGE
        DWORD error = GetLastError();
        if (RTGV_ERROR_NO_IMAGE != error)
        {
            RtPrintf("RtGVGrabFrame error: %d\n", GetLastError());
            return false;
        }

        if (errorCount >= 200)
        {
            return false;
        }
    }

    return true;
}

bool Rtx64GigECapture::retrieveFrame(int flag, OutputArray image)
{
    int         colorConversionCode = 0;    // The color conversion code used if we need to convert the color of the image
    int			inputArrayType = 0;         // The array type of the source image (what was received from the camera)
    int			outputArrayType = 0;        // The array type of the the output image (what will be returned to the caller)
    int         inputDepth = 0;             // The depth of the source image
    int         outputDepth = 0;            // The depth of the output image

    // Determine if we need to convert the color of the image, and if so what conversion code to use
    switch (pActiveFrame->pixelFormat)
    {
    case GVSP_PIX_BAYBG8:
    case GVSP_PIX_BAYBG10:
    case GVSP_PIX_BAYBG12:
    case GVSP_PIX_BAYBG10_PACKED:
    case GVSP_PIX_BAYBG12_PACKED:
    case GVSP_PIX_BAYBG16:
        colorConversionCode = CV_BayerBG2BGR;
        break;

    case GVSP_PIX_BAYGB8:
    case GVSP_PIX_BAYGB10:
    case GVSP_PIX_BAYGB12:
    case GVSP_PIX_BAYGB10_PACKED:
    case GVSP_PIX_BAYGB12_PACKED:
    case GVSP_PIX_BAYGB16:
        colorConversionCode = CV_BayerGB2BGR;
        break;

    case  GVSP_PIX_BAYRG8:
    case  GVSP_PIX_BAYRG10:
    case  GVSP_PIX_BAYRG12:
    case  GVSP_PIX_BAYRG10_PACKED:
    case  GVSP_PIX_BAYRG12_PACKED:
    case  GVSP_PIX_BAYRG16:
        colorConversionCode = CV_BayerRG2BGR;
        break;

    case GVSP_PIX_BAYGR16:
    case GVSP_PIX_BAYGR12_PACKED:
    case GVSP_PIX_BAYGR10_PACKED:
    case GVSP_PIX_BAYGR12:
    case GVSP_PIX_BAYGR10:
    case GVSP_PIX_BAYGR8:
        colorConversionCode = CV_BayerGR2BGR;
        break;

    default:
        colorConversionCode = 0;
    }

    // Determine the input and output array types
    switch (pActiveFrame->pixelFormat)
    {
    case GVSP_PIX_BAYBG8:
    case GVSP_PIX_BAYGR8:
    case GVSP_PIX_BAYGB8:
    case  GVSP_PIX_BAYRG8:
        inputArrayType = CV_8UC1;
        outputArrayType = CV_8UC3;
        break;
    case GVSP_PIX_BAYBG16:
    case GVSP_PIX_BAYGB16:
    case GVSP_PIX_BAYRG16:
    case GVSP_PIX_BAYGR16:
    case GVSP_PIX_BAYGR10:
    case GVSP_PIX_BAYRG10:
    case GVSP_PIX_BAYGB10:
    case GVSP_PIX_BAYBG10:
    case GVSP_PIX_BAYGR12:
    case GVSP_PIX_BAYRG12:
    case GVSP_PIX_BAYGB12:
    case GVSP_PIX_BAYBG12:
    case GVSP_PIX_BAYGR10_PACKED:
    case GVSP_PIX_BAYRG10_PACKED:
    case GVSP_PIX_BAYGB10_PACKED:
    case GVSP_PIX_BAYBG10_PACKED:
    case GVSP_PIX_BAYGR12_PACKED:
    case GVSP_PIX_BAYRG12_PACKED:
    case GVSP_PIX_BAYGB12_PACKED:
    case GVSP_PIX_BAYBG12_PACKED:
        inputArrayType = CV_16UC1;
        outputArrayType = CV_16UC3;
        break;
    default:
        inputArrayType = 0;
        outputArrayType = 0;
    }

    // Place the source image into a Mat object.
    cv::Mat mImageSrc = cv::Mat(pActiveFrame->height, pActiveFrame->width, inputArrayType, pActiveFrame->imageData);

    if (colorConversionCode &&  outputArrayType)
    {
        // Convert color
        cv::cvtColor(mImageSrc, image, colorConversionCode);
    }
    else
    {
        mImageSrc.copyTo(image);
    }

    return true;
}

int Rtx64GigECapture::getCaptureDomain()
{
    return CAP_RTX64_GIGE;
}

bool Rtx64GigECapture::isOpened() const
{
    return (-1 != openedCameraIndex);
}

}
#endif  // HAVE_RTX64_GIGE