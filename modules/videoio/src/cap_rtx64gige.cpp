// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#ifdef HAVE_RTX64_GIGE

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
#include "opencv2/videoio/cap_rtx64gige_RtImage.h"

#include "RTX_GigEV_DataTypes.h"
#include "opencv2/videoio/cap_rtx64gige.h"

CvCapture* cvCreateCameraCapture_Rtx64GigE(int index)
{
    cv::CvCaptureCAM_Rtx64GigE* capture = new cv::CvCaptureCAM_Rtx64GigE;

    if (capture->open(index))
    {
        return capture;
    }

    delete capture;
    return NULL;
}

namespace cv
{

    void macToString(char * pStrMAC, unsigned char *pMacAddr)
    {
        sprintf(pStrMAC, "%02x:%02x:%02x:%02x:%02x:%02x", *(pMacAddr + 5), *(pMacAddr + 4),
            *(pMacAddr + 3), *(pMacAddr + 2), *(pMacAddr + 1), *(pMacAddr));
    }

    unsigned int getImageDataOffset(void)
    {
        unsigned int structureSize = sizeof(RTX_Image) + sizeof(IplImage);
        return(ROUNDUP(structureSize, NODE_ALIGNMENT_SIZE));
    }

    CvCaptureCAM_Rtx64GigE::CvCaptureCAM_Rtx64GigE()
    {
        RtPrintf("CvCaptureCAM_Rtx64GigE::CvCaptureCAM_Rtx64GigE()\n");


        char *NIC_Name = "RtGigE_01_Nic";
        void *ndp = RtnGetDevicePtr(NIC_Name, DEVICE_NAME);
        m_IP_Address_Ours = htonl(RtnGetIpAddress(ndp));

        InitializeCameraController(m_IP_Address_Ours);
    }

#ifdef NOT_READY_YET
    void CvCaptureCAM_Rtx64GigE::ReturnCameraList(void)
    {

    }
#endif // NOT_READY_YET

    HRESULT CvCaptureCAM_Rtx64GigE::InitializeCameraController(uint32_t IP_ADDRESS_OURS)
    {
        m_IP_Address_Ours = IP_ADDRESS_OURS;

        if (m_Camera_Controller == NULL)
        {
            m_Camera_Controller = new RTX_GigEV_Camera_Controller();
            m_Camera_Controller->Initialize(NULL, 0);

            Sleep(10000);
        }
        // cameras should be discovered by now

        return S_OK;
    }


    bool CvCaptureCAM_Rtx64GigE::open(int index)
    {
        m_OpenCameraIndex = index;

        CI_Ptr = NULL;

        if (m_Camera_Controller != NULL)
        {
            CI_Ptr = m_Camera_Controller->Get_Camera_Class(m_OpenCameraIndex);

            if (CI_Ptr != NULL)
            {
                CI_Ptr->m_OpenedCameraIndex = index;

                m_Camera_Controller->Start_Camera(CI_Ptr->m_ID, 0, NULL, m_IP_Address_Ours, 0, CAMERA_PORT, NULL,
                    CI_Ptr->m_Width_Max, CI_Ptr->m_Height_Max, softwareTriggeredAcquisitionMode, FRAMERATE);

                int maxImageSize, bytesPerPixel;

                // we should use TotalBytesPerFrame.  Todo add m_TotalBytesPerFrame to CI_Ptr
                bytesPerPixel = MAX_BYTES_PER_PIXEL;  // assume worse case
                maxImageSize = CI_Ptr->m_Width_Max * CI_Ptr->m_Height_Max * bytesPerPixel;

                RtPrintf("m_Camera_Controller->Start_Camera returned OK. manufacturer_name=%s\n",
                    CI_Ptr->Discovery_Info.manufacturer_name);
                RtPrintf("m_Camera_Controller->Start_Camera returned OK. m_Width_Max=%d\n", CI_Ptr->m_Width_Max);
                RtPrintf("m_Camera_Controller->Start_Camera returned OK. m_Height_Max=%d\n", CI_Ptr->m_Height_Max);
                RtPrintf("m_Camera_Controller->Start_Camera returned OK. m_Width=%d\n", CI_Ptr->m_Width);
                RtPrintf("m_Camera_Controller->Start_Camera returned OK. m_Height=%d\n", CI_Ptr->m_Height);


                RtPrintf("m_Camera_Controller->Start_Camera returned OK. maxImageSize=%d\n", maxImageSize);

                uint32_t metaDataSize = getImageDataOffset();
                uint32_t imageSize = (IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_NUM_CHANNELS * IMAGE_BITS_PER_PIXEL) / 8;

                m_pActiveFrame = (RTX_Image*)malloc(metaDataSize + imageSize);

                //Initialize m_pActiveFrame fields
                m_pActiveFrame->bitsPerPixel = IMAGE_BITS_PER_PIXEL;
                m_pActiveFrame->pixelFormat = 0;
                m_pActiveFrame->numberOfChannels = IMAGE_NUM_CHANNELS;
                m_pActiveFrame->offset_X = 0;
                m_pActiveFrame->offset_Y = 0;
                m_pActiveFrame->Width = IMAGE_WIDTH;
                m_pActiveFrame->Height = IMAGE_HEIGHT;
                m_pActiveFrame->padding_X = 0;
                m_pActiveFrame->padding_Y = 0;
                m_pActiveFrame->Image_Size = (m_pActiveFrame->Height * m_pActiveFrame->Width * m_pActiveFrame->numberOfChannels * m_pActiveFrame->bitsPerPixel) / 8;
                m_pActiveFrame->offsetTo_Image_Data = getImageDataOffset();
                m_pActiveFrame->pCustomMetaData = ((char *)m_pActiveFrame) + sizeof(RTX_Image);
                m_pActiveFrame->Image_Data = ((char *)m_pActiveFrame) + m_pActiveFrame->offsetTo_Image_Data;

            }


        }

        return CI_Ptr != NULL;
    }

    void CvCaptureCAM_Rtx64GigE::release()
    {
        close();
    }

    void CvCaptureCAM_Rtx64GigE::close(void)
    {
        if (m_Camera_Controller != NULL)
        {
            m_Camera_Controller->Close();
        }

        CI_Ptr = NULL;
    }

    bool CvCaptureCAM_Rtx64GigE::grabFrame(void)
    {
        if (CI_Ptr != NULL)
        {
            if (S_OK == CI_Ptr->Do_Grab(m_pActiveFrame))
            {
                pGrabbedFrameObj = CI_Ptr->RetrieveGrab(GRAB_TIMEOUT);

                if (pGrabbedFrameObj == NULL)
                {
                    return false;
                }

                if (pGrabbedFrameObj->Image_Data == NULL)
                {
                    RtPrintf("CvCaptureCAM_Rtx64GigE::grabFrame bad data\n");
                    return false;
                }

                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }

#ifdef NOT_READY_YET
    bool CvCaptureCAM_Rtx64GigE::retrieveFrameEx(int index, RTX_Image ** pImageObj)
    {
        return false;
    }
#endif // NOT_READY_YET

    IplImage * CvCaptureCAM_Rtx64GigE::retrieveFrame(int index)
    {
        IplImage	*pIplImageFrameObj;
        int			colorConversionCode;
        int			inputArrayType = 0;  // ie CV_8UC1
        int			outputArrayType = 0;  // ie CV_8UC1
        int			pixelDepth;   // holds some value like IPL_DEPTH_8U

        if (pGrabbedFrameObj == NULL)
        {
            RtPrintf("CvCaptureCAM_Rtx64GigE::retrieveFrame error, no frame has been grabbed yet!  Call grab() first.\n");
            return NULL;
        }

        pIplImageFrameObj = (IplImage *)pGrabbedFrameObj->pCustomMetaData;

        pixelDepth = pGrabbedFrameObj->bitsPerPixel; // like IPL_DEPTH_8U

        cvInitImageHeader(pIplImageFrameObj, cvSize(pGrabbedFrameObj->Width, pGrabbedFrameObj->Height),
            pixelDepth, pGrabbedFrameObj->numberOfChannels, IPL_ORIGIN_TL, MAX_BYTES_PER_PIXEL);
        pIplImageFrameObj->imageData = pGrabbedFrameObj->Image_Data;
        int widthstep = (pixelDepth / 8) * pGrabbedFrameObj->Width * pGrabbedFrameObj->numberOfChannels;
        cvSetData(pIplImageFrameObj, pGrabbedFrameObj->Image_Data, widthstep);

        // Setup conversion code to BGR
        switch (pGrabbedFrameObj->pixelFormat)  // m_PixelFormat
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

        switch (pGrabbedFrameObj->pixelFormat)  // m_PixelFormat
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
        case  GVSP_PIX_BAYRG16:
        case GVSP_PIX_BAYGR16:
            inputArrayType = CV_16UC1;
            outputArrayType = CV_16UC3;
            break;

        default:
            inputArrayType = 0;
            outputArrayType = 0;
        }

        if (colorConversionCode &&  outputArrayType)
        {
            // convert bayer to BGR
            pGrabbedFrameObj->numberOfChannels = IMAGE_NUM_CHANNELS;
            pGrabbedFrameObj->CvArrayType = outputArrayType;


            cv::Mat mImageSrc(pIplImageFrameObj->height, pIplImageFrameObj->width, inputArrayType,
                pIplImageFrameObj->imageData, pIplImageFrameObj->widthStep);
            cv::Mat mImageDest(pIplImageFrameObj->height, pIplImageFrameObj->width, outputArrayType);

            cv::cvtColor(mImageSrc, mImageDest, colorConversionCode);


            cvInitImageHeader(pIplImageFrameObj, cvSize(mImageDest.cols, mImageDest.rows), IPL_DEPTH_8U,
                IMAGE_NUM_CHANNELS, IPL_ORIGIN_TL, MAX_BYTES_PER_PIXEL);
            widthstep = (pixelDepth / 8) * mImageDest.cols * IMAGE_NUM_CHANNELS;

            pIplImageFrameObj->imageData = pGrabbedFrameObj->Image_Data;
            memcpy(pIplImageFrameObj->imageData, mImageDest.data, (mImageDest.total() * mImageDest.elemSize()));
            cvSetData(pIplImageFrameObj, pGrabbedFrameObj->Image_Data, widthstep);


        }
        else
        {
            cvInitImageHeader(pIplImageFrameObj, cvSize(pGrabbedFrameObj->Width, pGrabbedFrameObj->Height),
                pixelDepth, pGrabbedFrameObj->numberOfChannels, IPL_ORIGIN_TL, MAX_BYTES_PER_PIXEL);
            pGrabbedFrameObj->CvArrayType = CV_MAKETYPE(pixelDepth, pGrabbedFrameObj->numberOfChannels);
            pIplImageFrameObj->imageData = pGrabbedFrameObj->Image_Data;
            widthstep = (pixelDepth / 8) * pGrabbedFrameObj->Width * pGrabbedFrameObj->numberOfChannels;
            cvSetData(pIplImageFrameObj, pGrabbedFrameObj->Image_Data, widthstep);
        }

        return  pIplImageFrameObj;
    }

}
#endif  // HAVE_RTX64_GIGE