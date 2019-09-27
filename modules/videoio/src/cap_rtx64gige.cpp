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
    if (!RtGVInitialize())
    {
        throw "Error initializing RtGigE Vision!\n";
    }

    // Enumerate Cameras
    DWORD numElements = 0;
    RtGVEnumerateCamerasA(NULL, numElements, &numCamerasDiscovered, INADDR_ANY, DISCOVERY_PORT);

    numElements = numCamerasDiscovered;

        pCameraInfos = new RTGV_CAMERA_INFOA[numElements];
        for (unsigned int i = 0; i < numElements; i++)
        {
            pCameraInfos[i].size = sizeof(RTGV_CAMERA_INFOA);
        }

        if (!RtGVEnumerateCamerasA(pCameraInfos, numElements, &numCamerasDiscovered, INADDR_ANY, DISCOVERY_PORT))
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
        if (!this->open(index))
        {
            throw "Rtx64GigECapture::open failed!\n";
        }
    }

bool Rtx64GigECapture::open(int index)
{
    uint32_t maxWidth = 0, maxHeight = 0, width = 0, height = 0, numChannels = 0, pixelDepth = 0, payloadSize = 0;
    GVSP_PIXEL_TYPE pixelFormat;

        // Initialize camera control
        if (!RtGVInitCameraControl(pCameraInfos[index].cameraHandle, interfaceIPAddress, CONTROL_PORT, HEARTBEAT_TIMEOUT_MS, HEARTBEATS_PER_PERIOD, TRUE))
        {
            RtPrintf("RtGVInitCameraControl Failed! Error: %d\n", GetLastError());
            return false;
        }

        // Get current camera settings
        if (!RtGVGetFrameMaxWidth(index, &maxWidth))
        {
            RtPrintf("RtGVGetFrameMaxWidth Failed! Error: %d\n", GetLastError());
            RtGVCloseCameraControl(index);
            return false;
        }

        if (!RtGVGetFrameMaxHeight(index, &maxHeight))
        {
            RtPrintf("RtGVGetFrameMaxHeight Failed! Error: %d\n", GetLastError());
            RtGVCloseCameraControl(index);
            return false;
        }

        if (!RtGVGetFrameWidth(index, &width))
        {
            RtPrintf("RtGVGetFrameWidth Failed! Error: %d\n", GetLastError());
            RtGVCloseCameraControl(index);
            return false;
        }
        if (!RtGVGetFrameHeight(index, &height))
        {
            RtPrintf("RtGVGetFrameHeight Failed! Error: %d\n", GetLastError());
            RtGVCloseCameraControl(index);
            return false;
        }

        if (!RtGVGetFrameNumberOfChannels(index, &numChannels))
        {
            RtPrintf("RtGVGetFrameNumberOfChannels Failed! Error: %d\n", GetLastError());
            RtGVCloseCameraControl(index);
            return false;
        }

        if (!RtGVGetFramePixelDepth(index, &pixelDepth))
        {
            RtPrintf("RtGVGetFramePixelDepth Failed! Error: %d\n", GetLastError());
            RtGVCloseCameraControl(index);
            return false;
        }

        if (!RtGVGetFramePixelFormat(index, &pixelFormat))
        {
            RtPrintf("RtGVGetFramePixelFormat Failed! Error: %d\n", GetLastError());
            RtGVCloseCameraControl(index);
            return false;
        }

        if (!RtGVGetAcquisitionMode(index, &acquisitionMode))
        {
            RtPrintf("RtGVGetAcquisitionMode failed on camera %d. Error: %d\n", index, GetLastError());
            RtGVCloseCameraControl(index);
            return false;
        }

        if (invalidAcquisitionMode == acquisitionMode)
        {
            RtPrintf("Invalid acquisition mode configured on camera %d!\n", index);
            RtGVCloseCameraControl(index);
            return false;
        }

        if (!RtGVGetPayloadSize(index, &payloadSize))
        {
            RtPrintf("RtGVGetPayloadSize failed on camera %d. Error: %d\n", index, GetLastError());
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
        pActiveFrame->imageData = ((char *)pActiveFrame) + pActiveFrame->offsetToImageData;

        HANDLE streamErrorEvent;
        // Start the stream
        if (!RtGVStartStream(index, interfaceIPAddress, STREAM_PORT + index, 10, &streamErrorEvent))
        {
            free(pActiveFrame);
            RtGVCloseCameraControl(index);
            return false;
        }

        openedCameraHandle = pCameraInfos[index].cameraHandle;

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
        if (-1 != openedCameraHandle)
        {
            // Close the stream
            RtGVCloseStream(openedCameraHandle);

            // Close Camera Control
            RtGVCloseCameraControl(openedCameraHandle);
        }

    // Close RtGigEVision
    RtGVClose();
}

    bool Rtx64GigECapture::grabFrame()
    {
        // Check if camera is configured for trigger mode, and if it is, send a trigger
        if (softwareTriggeredAcquisitionMode == acquisitionMode)
        {
            if (!RtGVSendTrigger(openedCameraHandle))
            {
                return false;
            }
        }

        // Attempt to grab a frame using RtGVGrabFrame
        if (!RtGVGrabFrame(openedCameraHandle, pActiveFrame, 1000))
        {
            // If we are grabbing faster than the camera is sending images, then we will hit RTGV_ERROR_NO_IMAGE
            DWORD error = GetLastError();
            if (RTGV_ERROR_NO_IMAGE == error)
            {
                // Wait for the camera to catch up
                if (!RtGVWaitForFrame(openedCameraHandle, 1000))
                {
                    RtPrintf("A new frame was not detected after 1 second.  Error: %d\n", error);
                }

                if (!RtGVGrabFrame(openedCameraHandle, pActiveFrame, 1000))
                {
                    RtPrintf("Error in RtGVGrabFrame: %d\n", error);
                }
            }
            else
            {
                RtPrintf("Error in RtGVGrabFrame: %d\n", error);
            }
        }

    return true;
}

    bool Rtx64GigECapture::retrieveFrame(int flag, OutputArray image)
    {
        int         colorConversionCode = 0;    // The color conversion code used if we need to convert the color of the image
        int			inputArrayType = 0;         // The array type of the source image (what was received from the camera)
        int			outputArrayType = 0;        // The array type of the the output image (what will be returned to the caller)
        int         outputDepth = 8;            // The depth of the output image
        bool        packedConversion = false;

        // Currently, this code converts all color formats to either Mono 8 bit or RGB 24 bit, depending on the input color format.  However, this is easily modifiable to output
        // whatever color format is preferable.

        // Determine if we need to convert the color of the image, and if so what conversion code to use
        switch (pActiveFrame->pixelFormat)
        {
        case GVSP_PIX_BAYBG8:
        case GVSP_PIX_BAYBG10:
        case GVSP_PIX_BAYBG12:
        case GVSP_PIX_BAYBG10_PACKED:
        case GVSP_PIX_BAYBG12_PACKED:
        case GVSP_PIX_BAYBG16:
            colorConversionCode = CV_BayerBG2RGB;
            break;

        case GVSP_PIX_BAYGB8:
        case GVSP_PIX_BAYGB10:
        case GVSP_PIX_BAYGB12:
        case GVSP_PIX_BAYGB10_PACKED:
        case GVSP_PIX_BAYGB12_PACKED:
        case GVSP_PIX_BAYGB16:
            colorConversionCode = CV_BayerGB2RGB;
            break;

        case  GVSP_PIX_BAYRG8:
        case  GVSP_PIX_BAYRG10:
        case  GVSP_PIX_BAYRG12:
        case  GVSP_PIX_BAYRG10_PACKED:
        case  GVSP_PIX_BAYRG12_PACKED:
        case  GVSP_PIX_BAYRG16:
            colorConversionCode = CV_BayerRG2RGB;
            break;

        case GVSP_PIX_BAYGR16:
        case GVSP_PIX_BAYGR12_PACKED:
        case GVSP_PIX_BAYGR10_PACKED:
        case GVSP_PIX_BAYGR12:
        case GVSP_PIX_BAYGR10:
        case GVSP_PIX_BAYGR8:
            colorConversionCode = CV_BayerGR2RGB;
            break;
        case GVSP_PIX_RGBA8_PACKED:
            colorConversionCode = CV_RGBA2RGB;
            break;
        case GVSP_PIX_BGRA8_PACKED:
            colorConversionCode = CV_BGRA2RGB;
            break;
        case GVSP_PIX_YUV411_PACKED:
        case GVSP_PIX_YUV422_PACKED:
            colorConversionCode = CV_YUV2BGR_UYVY;  // There may be an issue
            break;
        case GVSP_PIX_YUV422_YUYV_PACKED:
            colorConversionCode = CV_YUV2BGR_YUYV;
            break;
        case GVSP_PIX_YUV444_PACKED:
            colorConversionCode = CV_YUV2RGB;
            break;
        case GVSP_PIX_BGR8_PACKED:
        case GVSP_PIX_BGR10_PACKED:
        case GVSP_PIX_BGR12_PACKED:
            colorConversionCode = CV_BGR2RGB;
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
            inputArrayType = CV_16UC1;
            outputArrayType = CV_8UC3;
            break;
        case GVSP_PIX_BAYGR10_PACKED:
        case GVSP_PIX_BAYRG10_PACKED:
        case GVSP_PIX_BAYGB10_PACKED:
        case GVSP_PIX_BAYBG10_PACKED:
        case GVSP_PIX_BAYGR12_PACKED:
        case GVSP_PIX_BAYRG12_PACKED:
        case GVSP_PIX_BAYGB12_PACKED:
        case GVSP_PIX_BAYBG12_PACKED:
            packedConversion = true;
            inputArrayType = CV_16UC1;
            outputArrayType = CV_8UC3;
            break;
        case GVSP_PIX_MONO10_PACKED:
        case GVSP_PIX_MONO12_PACKED:
            packedConversion = true;
            inputArrayType = CV_16UC1;
            outputArrayType = CV_8UC1;
            break;
        case GVSP_PIX_MONO8:
        case GVSP_PIX_MONO8_SIGNED:
            inputArrayType = CV_8UC1;
            outputArrayType = CV_8UC1;
            break;
        case GVSP_PIX_MONO10:
        case GVSP_PIX_MONO12:
        case GVSP_PIX_MONO16:
            inputArrayType = CV_16UC1;
            outputArrayType = CV_8UC1;
            break;
        case GVSP_PIX_RGBA8_PACKED:
        case GVSP_PIX_BGRA8_PACKED:
            inputArrayType = CV_8UC4;
            outputArrayType = CV_8UC3;
            break;
        case GVSP_PIX_RGB10V1_PACKED:
        case GVSP_PIX_RGB10V2_PACKED:
        case GVSP_PIX_RGB10_PACKED:
        case GVSP_PIX_BGR10_PACKED:
        case GVSP_PIX_RGB12_PACKED:
        case GVSP_PIX_BGR12_PACKED:
        case GVSP_PIX_RGB16:
            inputArrayType = CV_16UC3;
            outputArrayType = CV_8UC3;
            break;
        case GVSP_PIX_YUV411_PACKED:
        case GVSP_PIX_YUV422_PACKED:
        case GVSP_PIX_YUV422_YUYV_PACKED:
            inputArrayType = CV_8UC2;
            outputArrayType = CV_8UC3;
            break;
        case GVSP_PIX_YUV444_PACKED:
            inputArrayType = CV_8UC3;
            outputArrayType = CV_8UC3;
            break;
        case GVSP_PIX_BGR8_PACKED:
            inputArrayType = CV_8UC3;
            outputArrayType = CV_8UC3;
            break;
        default:
            inputArrayType = CV_8UC3;
            outputArrayType = CV_8UC3;
        }

        cv::Mat mImageSrc;
        // Place the source image into a Mat object.
        if (packedConversion)
        {
            mImageSrc = cv::Mat(pActiveFrame->height, pActiveFrame->width, inputArrayType);
            for (int dstIndex = 0, srcIndex = 0; srcIndex < pActiveFrame->frameSize; srcIndex = srcIndex + 3)
            {
                mImageSrc.data[dstIndex + 1] = 0x0F & (pActiveFrame->imageData[srcIndex] >> 4);
                mImageSrc.data[dstIndex] = (0x0F & pActiveFrame->imageData[srcIndex + 1]) | (0xF0 & (pActiveFrame->imageData[srcIndex] << 4));
                mImageSrc.data[dstIndex + 3] = 0x0F & (pActiveFrame->imageData[srcIndex + 2] >> 4);
                mImageSrc.data[dstIndex + 2] = (0x0F & (pActiveFrame->imageData[srcIndex + 1] >> 4)) | (0xF0 & (pActiveFrame->imageData[srcIndex + 2] << 4));

                dstIndex = dstIndex + 4;
            }
        }
        else
        {
            mImageSrc = cv::Mat(pActiveFrame->height, pActiveFrame->width, inputArrayType, pActiveFrame->imageData);
        }

        if (colorConversionCode)
        {
            // Convert color
            cv::cvtColor(mImageSrc, mImageSrc, colorConversionCode);
        }

        if (outputArrayType != inputArrayType)
        {
            // Convert image to the correct type
            double alpha = std::pow(2.0, outputDepth) / std::pow(2.0, pActiveFrame->pixelDepth);
            mImageSrc.convertTo(mImageSrc, outputArrayType, alpha);
        }

        mImageSrc.copyTo(image);

    return true;
}

int Rtx64GigECapture::getCaptureDomain()
{
    return CAP_RTX64_GIGE;
}

    bool Rtx64GigECapture::isOpened() const
    {
        return (-1 != openedCameraHandle);
    }

}
#endif  // HAVE_RTX64_GIGE