////////////////////////////////////////////////////////////////////////////////////////
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
//

//
// The code has been contributed by Vladimir N. Litvinenko on 2012 Jul
// mailto:vladimir.litvinenko@codepaint.ru
//

#include "precomp.hpp"
#include <GigEVisionSDK.h>
#include <GigEVisionSDK.cpp>

#ifdef WIN32
#include <io.h>
#else
#include <stdio.h>
#endif

#ifdef NDEBUG
#define CV_WARN(message)
#else
#define CV_WARN(message) fprintf(stderr, "warning: %s (%s:%d)\n", message, __FILE__, __LINE__)
#endif

#define QTGIG_HEARTBEAT_TIME (12000.0)
#define QTGIG_MAX_WAIT_TIME (2.0)
#define QTGIG_IMG_WAIT_TIME (3.0)

/*----------------------------------------------------------------------------*/
/**
  \internal
  \fn bool wrprInitGigEVisionAPI();
  \brief Wrapper to GigEVisionAPI function gige::InitGigEVisionAPI ()
  \return true -- success
  See \a wrprExitGigEVisionAPI

*/
bool
wrprInitGigEVisionAPI()
{
  CV_FUNCNAME("wrprInitGigEVisionAPI");
  __BEGIN__;

  try {
    gige::InitGigEVisionAPI ();
  } catch(...) {
    CV_ERROR(CV_StsError, "GigEVisionAPI: initialization (InitGigEVisionAPI()) failed.\n");
  }
  __END__;
  return true;
}

/*----------------------------------------------------------------------------*/
/**
  \internal
  \fn void wrprExitGigEVisionAPI()
  \brief Wrapper to GigEVisionAPI function gige::ExitGigEVisionAPI ()
  \return true -- success
  See \a wrprInitGigEVisionAPI

*/
bool
wrprExitGigEVisionAPI()
{
  CV_FUNCNAME("wrprExitGigEVisionAPI");
  __BEGIN__;

  try {
    gige::ExitGigEVisionAPI ();
  } catch(...) {
    CV_ERROR(CV_StsError, "GigEVisionAPI: finalization (ExitGigEVisionAPI()) failed.\n");
    return false;
  }
  __END__;
  return true;
}


/*----------------------------------------------------------------------------*/
/**
  \internal
  \fn gige::IGigEVisionAPI wrprGetGigEVisionAPI()
  \brief Wrapper to GigEVisionAPI function gige::GetGigEVisionAPI ()
  \return item of gige::IGigEVisionAPI type
  See \a wrprInitGigEVisionAPI, \a gige::IGigEVisionAPI
*/
gige::IGigEVisionAPI
wrprGetGigEVisionAPI()
{

  gige::IGigEVisionAPI b_ret = 0;

  CV_FUNCNAME("wrprGetGigEVisionAPI");
  __BEGIN__;

  try {
    b_ret = gige::GetGigEVisionAPI ();
  } catch(...) {
    CV_ERROR(CV_StsError, "GigEVisionAPI: API instance (from GetGigEVisionAPI()) failed.\n");
  }

  __END__;

  return b_ret;
}


/*----------------------------------------------------------------------------*/
/**
  \internal
  \fn bool wrprUnregisterCallback( const gige::IGigEVisionAPI* api, gige::ICallbackEvent* eventHandler)
  \brief Wrapper to GigEVisionAPI function
  \param api
  \param eventHandler
  \return true - succsess, else - false
  See \a wrprInitGigEVisionAPI, \a gige::IGigEVisionAPI

*/
bool
wrprUnregisterCallback( const gige::IGigEVisionAPI* api, gige::ICallbackEvent* eventHandler)
{
  bool b_ret = api != NULL;

  if(b_ret) b_ret = api->IsValid ();

  CV_FUNCNAME("wrprUnregisterCallback");
  __BEGIN__;

  if(b_ret)
  {
    if(eventHandler != NULL)
    {
      try {
        b_ret = ((gige::IGigEVisionAPIInterface*)api)->UnregisterCallback (eventHandler);
      } catch(...) {
        CV_ERROR(CV_StsError, "GigEVisionAPI: API unregister callback function (from UnregisterCallback()) failed.\n");
        b_ret = false;
      }
    }
  }
  __END__;

  return (b_ret);
}


/*----------------------------------------------------------------------------*/
/**
  \internal
  \fn bool wrprDeviceIsConnect( gige::IDevice& device )
  \brief Wrapper to GigEVisionAPI function IDevice::IsConnected()
  \param device - selected device
  \return true - device connected
*/
bool
wrprDeviceIsConnect( gige::IDevice& device )
{
  bool b_ret = device != NULL;

  CV_FUNCNAME("wrprDeviceIsConnect");
  __BEGIN__;

  if(b_ret)
  {
    try {
      b_ret = device->IsConnected ();
    } catch (...) {
      CV_ERROR(CV_StsError, "GigEVisionAPI: API device connection state (from IsConnected()) failed.\n");
      b_ret = false;
    }
  }
  __END__;

  return (b_ret);
}


/*----------------------------------------------------------------------------*/
/**
  \internal
  \fn bool wrprDeviceIsValid( gige::IDevice& device )
  \brief Wrapper to GigEVisionAPI function IDevice::Connect()
  \param device - selected device
  \return true - device valid

*/
bool
wrprDeviceIsValid( gige::IDevice& device )
{
  bool b_ret = device != NULL;

  CV_FUNCNAME("wrprDeviceIsConnect");
  __BEGIN__;

  if(b_ret)
  {
    try {
      b_ret = device.IsValid ();
    } catch (...) {
      CV_ERROR(CV_StsError, "GigEVisionAPI: API device validation state (from IsValid()) failed.\n");
      b_ret = false;
    }
  }
  __END__;

  return (b_ret);
}


/*----------------------------------------------------------------------------*/
/**
  \internal
  \fn bool wrprDeviceDisconnect ( gige::IDevice& device )
  \brief Wrapper to GigEVisionAPI function IDevice::Disconnect()
  \param device - selected device
  \return true - device valid

*/
bool
wrprDeviceDisconnect ( gige::IDevice& device )
{
  bool b_ret = device != NULL;

  CV_FUNCNAME("wrprDeviceDisconnect");
  __BEGIN__;

  if(b_ret)
  {
    try {
      device->Disconnect ();
    } catch (...) {
      CV_ERROR(CV_StsError, "GigEVisionAPI: API device disconnect (from Disconnect()) failed.\n");
      b_ret = false;
    }
  }

  __END__;

  return (b_ret);
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
/**
  \internal
  \class CvCaptureCAM_Giganetix
  \brief Capturing video from camera via Smartec Giganetix (use GigEVisualSDK library).
*/

class CvCaptureCAM_Giganetix : public CvCapture
{
  public:
    CvCaptureCAM_Giganetix();
    virtual ~CvCaptureCAM_Giganetix();

    virtual bool open( int index );
    virtual void close();
    virtual double getProperty(int);
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
    virtual int getCaptureDomain()
    {
        return CV_CAP_GIGANETIX;
    }

    bool  start ();
    bool  stop ();

  protected:

    void  init ();
    void  grabImage ();

    gige::IGigEVisionAPI  m_api;
    bool                  m_api_on;
    gige::IDevice         m_device;
    bool                  m_active;

    IplImage* m_raw_image;
    UINT32    m_rawImagePixelType;
    bool      m_monocrome;

};
/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void
CvCaptureCAM_Giganetix::init ()
{
  m_monocrome = m_active = m_api_on = false;
  m_api = 0;
  m_device = 0;
  m_raw_image = 0;
  m_rawImagePixelType = 0;
}

/*----------------------------------------------------------------------------*/
CvCaptureCAM_Giganetix::CvCaptureCAM_Giganetix()
{
  init ();

  m_api_on = wrprInitGigEVisionAPI ();

  if(m_api_on)
  {
    if((m_api = wrprGetGigEVisionAPI ()) != NULL)
    {
      m_api->SetHeartbeatTime (QTGIG_HEARTBEAT_TIME);
    }
  }
}

/*----------------------------------------------------------------------------*/
CvCaptureCAM_Giganetix::~CvCaptureCAM_Giganetix()
{
  close();
}
/*----------------------------------------------------------------------------*/
void
CvCaptureCAM_Giganetix::close()
{
  stop ();

  (void)wrprDeviceDisconnect(m_device);

  (void)wrprExitGigEVisionAPI ();

  if(m_raw_image) cvReleaseImageHeader(&m_raw_image);

  init ();
}

/*----------------------------------------------------------------------------*/
bool
CvCaptureCAM_Giganetix::open( int index )
{
  bool b_ret = m_api_on;

  CV_FUNCNAME("CvCaptureCAM_Giganetix::open");
  __BEGIN__;

  if(b_ret)
    b_ret = m_api.IsValid ();

  if(b_ret )
  {
    m_api->FindAllDevices (QTGIG_MAX_WAIT_TIME);

    //TODO - serch device as DevicesList member
    gige::DevicesList DevicesList = m_api->GetAllDevices ();

    m_device = 0;
    b_ret = false;

    for (int i = 0; i < (int) DevicesList.size() && !b_ret; i++)
    {
      if((b_ret = i == index))
      {
        m_device = DevicesList[i];
        b_ret = m_device->Connect ();

        if(b_ret)
        {
          b_ret =
                m_device->SetStringNodeValue("AcquisitionStatusSelector", "AcquisitionActive")
                &&
                m_device->SetStringNodeValue ("TriggerMode", "Off")
                &&
                m_device->SetStringNodeValue ("AcquisitionMode", "Continuous")
                &&
                m_device->SetIntegerNodeValue ("AcquisitionFrameCount", 20)
                ;
        }
      }
    } // for
  }

  if(!b_ret)
  {
    CV_ERROR(CV_StsError, "Giganetix: Error cannot find camera\n");
    close ();
  } else {
    start ();
  }

  __END__;

  return b_ret;
}

/*----------------------------------------------------------------------------*/
void
CvCaptureCAM_Giganetix::grabImage ()
{
  CV_FUNCNAME("CvCaptureCAM_Giganetix::grabImage");
  __BEGIN__;

  if(wrprDeviceIsValid(m_device) && wrprDeviceIsConnect(m_device))
  {
    if(!m_device->IsBufferEmpty ())
    {
      gige::IImageInfo imageInfo;
      m_device->GetImageInfo (&imageInfo);
      assert(imageInfo.IsValid());

      if (m_device->GetPendingImagesCount() ==  1)
      {
        UINT32 newPixelType;
        UINT32 newWidth, newHeight;

        imageInfo->GetPixelType(newPixelType);
        imageInfo->GetSize(newWidth, newHeight);

        //TODO - validation of image exists
        bool b_validation = m_raw_image != NULL;
        if(b_validation)
        {
          b_validation =
                  m_raw_image->imageSize == (int)(imageInfo->GetRawDataSize ())
                  &&
                  m_rawImagePixelType == newPixelType;
        } else {
          if(m_raw_image) cvReleaseImageHeader(&m_raw_image);
        }

        m_rawImagePixelType = newPixelType;
        m_monocrome = GvspGetBitsPerPixel((GVSP_PIXEL_TYPES)newPixelType) == IPL_DEPTH_8U;

        try {
          if (m_monocrome)
          {
            //TODO - For Mono & Color BayerRGB raw pixel types
            if (!b_validation)
            {
              m_raw_image = cvCreateImageHeader (cvSize((int)newWidth, (int)newHeight),IPL_DEPTH_8U,1);
              m_raw_image->origin = IPL_ORIGIN_TL;
              m_raw_image->dataOrder =  IPL_DATA_ORDER_PIXEL;
              m_raw_image->widthStep = newWidth;
            }
            // Copy image.
            // ::memcpy(m_raw_image->imageData, imageInfo->GetRawData (), imageInfo->GetRawDataSize ());

            //TODO - Set pointer to image !
            m_raw_image->imageData = (char*)(imageInfo->GetRawData ());
          }

          if (!m_monocrome && newPixelType == GVSP_PIX_RGB8_PACKED)
          {
            //TODO - 24 bit RGB color image.
            if (!b_validation)
            {
              m_raw_image = cvCreateImageHeader (cvSize((int)newWidth, (int)newHeight), IPL_DEPTH_32F, 3);
              m_raw_image->origin = IPL_ORIGIN_TL;
              m_raw_image->dataOrder =  IPL_DATA_ORDER_PIXEL;
              m_raw_image->widthStep = newWidth * 3;
            }
            m_raw_image->imageData = (char*)(imageInfo->GetRawData ());
          }
        } catch (...) {
          CV_ERROR(CV_StsError, "Giganetix: failed to queue a buffer on device\n");
          close ();
        }
      } else {
        //TODO - all other pixel types
        m_raw_image = 0;
        CV_WARN("Giganetix: Undefined image pixel type\n");
      }
      m_device->PopImage (imageInfo);
      m_device->ClearImageBuffer ();
    }
  }

  __END__;
}

/*----------------------------------------------------------------------------*/
bool
CvCaptureCAM_Giganetix::start ()
{
  CV_FUNCNAME("CvCaptureCAM_Giganetix::start");
  __BEGIN__;

  m_active = wrprDeviceIsValid(m_device) && wrprDeviceIsConnect(m_device);

  if(m_active)
  {
    (void)m_device->SetIntegerNodeValue("TLParamsLocked", 1);
    (void)m_device->CommandNodeExecute("AcquisitionStart");
    m_active = m_device->GetBooleanNodeValue("AcquisitionStatus", m_active);
  }

  if(!m_active)
  {
    CV_ERROR(CV_StsError, "Giganetix: Cannot open camera\n");
    close ();
  }

  __END__;

  return m_active;
}

/*----------------------------------------------------------------------------*/
bool
CvCaptureCAM_Giganetix::stop ()
{
  if (!m_active) return true;

  CV_FUNCNAME("CvCaptureCAM_Giganetix::stop");
  __BEGIN__;

  if(wrprDeviceIsValid(m_device) && wrprDeviceIsConnect(m_device))
  {
    (void)m_device->GetBooleanNodeValue("AcquisitionStatus", m_active);

    if(m_active)
    {
      (void)m_device->CommandNodeExecute("AcquisitionStop");
      (void)m_device->SetIntegerNodeValue("TLParamsLocked", 0);
      m_device->ClearImageBuffer ();
      (void)m_device->GetBooleanNodeValue("AcquisitionStatus", m_active);
    }
  }

  if(m_active)
  {
    CV_ERROR(CV_StsError, "Giganetix: Improper closure of the camera\n");
    close ();
  }
  __END__;

  return !m_active;
}

/*----------------------------------------------------------------------------*/
bool
CvCaptureCAM_Giganetix::grabFrame()
{
  bool b_ret =
            wrprDeviceIsValid(m_device)
            &&
            wrprDeviceIsConnect(m_device);

  if(b_ret) grabImage ();

  return b_ret;
}


/*----------------------------------------------------------------------------*/
IplImage*
CvCaptureCAM_Giganetix::retrieveFrame(int)
{
  return (
        wrprDeviceIsValid(m_device) && wrprDeviceIsConnect(m_device) ?
          m_raw_image :
          NULL
  );
}

/*----------------------------------------------------------------------------*/
double
CvCaptureCAM_Giganetix::getProperty( int property_id )
{
  double d_ret = -1.0;
  INT64 i;

  if(wrprDeviceIsConnect(m_device))
  {
    switch ( property_id )
    {
      case CV_CAP_PROP_FRAME_WIDTH:
        m_device->GetIntegerNodeValue ("Width", i);
        d_ret = i;
        break;
      case CV_CAP_PROP_FRAME_HEIGHT:
        m_device->GetIntegerNodeValue ("Height", i);
        d_ret = i;
        break;
      case CV_CAP_PROP_GIGA_FRAME_OFFSET_X:
        m_device->GetIntegerNodeValue ("OffsetX", i);
        d_ret = i;
        break;
      case CV_CAP_PROP_GIGA_FRAME_OFFSET_Y:
        m_device->GetIntegerNodeValue ("OffsetY", i);
        d_ret = i;
        break;
      case CV_CAP_PROP_GIGA_FRAME_WIDTH_MAX:
        m_device->GetIntegerNodeValue ("WidthMax", i);
        d_ret = i;
        break;
      case CV_CAP_PROP_GIGA_FRAME_HEIGH_MAX:
        m_device->GetIntegerNodeValue ("HeightMax", i);
        d_ret = i;
        break;
      case CV_CAP_PROP_GIGA_FRAME_SENS_WIDTH:
        m_device->GetIntegerNodeValue ("SensorWidth", i);
        d_ret = i;
        break;
      case CV_CAP_PROP_GIGA_FRAME_SENS_HEIGH:
        m_device->GetIntegerNodeValue ("SensorHeight", i);
        d_ret = i;
        break;
      case CV_CAP_PROP_FRAME_COUNT:
        m_device->GetIntegerNodeValue ("AcquisitionFrameCount", i);
        d_ret = i;
        break;
      case CV_CAP_PROP_EXPOSURE:
        m_device->GetFloatNodeValue ("ExposureTime",d_ret);
        break;
      case CV_CAP_PROP_GAIN :
        m_device->GetFloatNodeValue ("Gain",d_ret);
        break;
      case CV_CAP_PROP_TRIGGER :
        bool b;
        m_device->GetBooleanNodeValue ("TriggerMode",b);
        d_ret = (double)b;
        break;
      case CV_CAP_PROP_TRIGGER_DELAY :
        m_device->GetFloatNodeValue ("TriggerDelay",d_ret);
        break;
      default : ;
    }
  }

  return d_ret;
}

/*----------------------------------------------------------------------------*/
bool
CvCaptureCAM_Giganetix::setProperty( int property_id, double value )
{
  bool b_ret = wrprDeviceIsConnect(m_device);

  if(b_ret)
  {
    bool b_val = m_active;

    switch ( property_id )
    {
      case CV_CAP_PROP_FRAME_WIDTH:
        stop ();
        b_ret = m_device->SetIntegerNodeValue ("Width", (INT64)value);
        if(b_val) start ();
        break;
      case CV_CAP_PROP_GIGA_FRAME_WIDTH_MAX:
        stop ();
        b_ret = m_device->SetIntegerNodeValue ("WidthMax", (INT64)value);
        if(b_val) start ();
        break;
      case CV_CAP_PROP_GIGA_FRAME_SENS_WIDTH:
        stop ();
        b_ret = m_device->SetIntegerNodeValue ("SensorWidth", (INT64)value);
        if(b_val) start ();
        break;
      case CV_CAP_PROP_FRAME_HEIGHT:
        stop ();
        b_ret = m_device->SetIntegerNodeValue ("Height", (INT64)value);
        if(b_val) start ();
        break;
      case CV_CAP_PROP_GIGA_FRAME_HEIGH_MAX:
        stop ();
        b_ret = m_device->SetIntegerNodeValue ("HeightMax", (INT64)value);
        if(b_val) start ();
        break;
      case CV_CAP_PROP_GIGA_FRAME_SENS_HEIGH:
        stop ();
        b_ret = m_device->SetIntegerNodeValue ("SensorHeight", (INT64)value);
        if(b_val) start ();
        break;
      case CV_CAP_PROP_GIGA_FRAME_OFFSET_X: {
        INT64 w, wmax, val = (INT64)value;
        if((b_ret = m_device->GetIntegerNodeValue ("Width", w)))
          if((b_ret = m_device->GetIntegerNodeValue ("WidthMax", wmax)))
            b_ret = m_device->SetIntegerNodeValue ("OffsetX", val w > wmax ? wmax - w : val);
      } break;
      case CV_CAP_PROP_GIGA_FRAME_OFFSET_Y: {
        INT64 h, hmax, val = (INT64)value;
        if((b_ret = m_device->GetIntegerNodeValue ("Height", h)))
          if((b_ret = m_device->GetIntegerNodeValue ("HeightMax", hmax)))
            b_ret = m_device->SetIntegerNodeValue ("OffsetY", val h > hmax ? hmax - h : val);
        b_ret = m_device->SetIntegerNodeValue ("OffsetY", (INT64)value);
      }
        break;
      case CV_CAP_PROP_EXPOSURE:
        b_ret = m_device->SetFloatNodeValue ("ExposureTime",value);
        break;
      case CV_CAP_PROP_GAIN :
        b_ret = m_device->SetFloatNodeValue ("Gain",value);
          break;
      case CV_CAP_PROP_TRIGGER :
        b_ret = m_device->SetBooleanNodeValue ("TriggerMode",(bool)value);
        break;
      case CV_CAP_PROP_TRIGGER_DELAY :
        stop ();
        b_ret = m_device->SetFloatNodeValue ("TriggerDelay",value);
        if(b_val) start ();
        break;
    default:
        b_ret = false;
    }
  }

  return b_ret;
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
CvCapture*
cvCreateCameraCapture_Giganetix( int index )
{
    CvCaptureCAM_Giganetix* capture = new CvCaptureCAM_Giganetix;

    if (!(capture->open( index )))
    {
      delete capture;
      capture = NULL;
    }

    return ((CvCapture*)capture);
}

/*----------------------------------------------------------------------------*/
