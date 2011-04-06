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

#ifdef HAVE_ANDROID_NATIVE_CAMERA

#include <opencv2/imgproc/imgproc.hpp>
#include <pthread.h>
#include <android/log.h>
#include "camera_activity.h"

#define LOG_TAG "CV_CAP"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))

class HighguiAndroidCameraActivity;

class CvCapture_Android : public CvCapture
{
public:
    CvCapture_Android();
    virtual ~CvCapture_Android();

    virtual double getProperty(int propIdx);
    virtual bool setProperty(int probIdx, double propVal);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int outputType);
    virtual int getCaptureDomain() { return CV_CAP_ANDROID; }

    bool isOpened() const;

protected:
    struct OutputMap
    {
    public:
        cv::Mat mat;
        IplImage* getIplImagePtr();
    private:
        IplImage iplHeader;
    };

    CameraActivity* m_activity;

private:
    bool m_isOpened;

    OutputMap *m_frameYUV;
    OutputMap *m_frameYUVnext;
    OutputMap m_frameGray;
    OutputMap m_frameColor;
    bool m_hasGray;
    bool m_hasColor;

    pthread_mutex_t m_nextFrameMutex;
    pthread_cond_t m_nextFrameCond;
    volatile bool m_waitingNextFrame;

    int m_framesGrabbed;

    friend class HighguiAndroidCameraActivity;

    void onFrame(const void* buffer, int bufferSize);

    void convertBufferToYUV(const void* buffer, int size, int width, int height);
    static bool convertYUVToGrey(const cv::Mat& yuv, cv::Mat& resmat);
    static bool convertYUVToColor(const cv::Mat& yuv, cv::Mat& resmat);
};


class HighguiAndroidCameraActivity : public CameraActivity
{
  public:
    HighguiAndroidCameraActivity(CvCapture_Android* capture)
    {
      m_capture = capture;
      m_framesReceived = 0;
    }

    virtual bool onFrameBuffer(void* buffer, int bufferSize)
    {
      LOGD("buffer addr:%p size:%d",buffer, bufferSize);
      if(isConnected() && buffer != 0 && bufferSize > 0)
      {
	m_framesReceived++;
	if (m_capture->m_waitingNextFrame)
	{
	  m_capture->onFrame(buffer, bufferSize);
	  pthread_mutex_lock(&m_capture->m_nextFrameMutex);
	  m_capture->m_waitingNextFrame = false;//set flag that no more frames required at this moment
	  pthread_cond_broadcast(&m_capture->m_nextFrameCond);
	  pthread_mutex_unlock(&m_capture->m_nextFrameMutex);
	}
	return true;
      }
      return false;
    }

    void LogFramesRate()
    {
      LOGI("FRAMES received: %d  grabbed: %d", m_framesReceived, m_capture->m_framesGrabbed);
    }

  private:
    CvCapture_Android* m_capture;
    int m_framesReceived;
};

IplImage* CvCapture_Android::OutputMap::getIplImagePtr()
{
    if( mat.empty() )
        return 0;

    iplHeader = IplImage(mat);
    return &iplHeader;
}

bool CvCapture_Android::isOpened() const
{
    return m_isOpened;
}

CvCapture_Android::CvCapture_Android()
{
  //defaults
  m_activity = 0;
  m_isOpened = false;
  m_frameYUV = 0;
  m_frameYUVnext = 0;
  m_hasGray = false;
  m_hasColor = false;
  m_waitingNextFrame = false;
  m_framesGrabbed = 0;

  //try connect to camera
  m_activity = new HighguiAndroidCameraActivity(this);

  if (m_activity == 0) return;
  pthread_mutex_init(&m_nextFrameMutex, NULL);
  pthread_cond_init (&m_nextFrameCond, NULL);

  CameraActivity::ErrorCode errcode = m_activity->connect();
  if(errcode == CameraActivity::NO_ERROR)
  {
    m_isOpened = true;
    m_frameYUV = new OutputMap();
    m_frameYUVnext = new OutputMap();
  }
  else
  {
    LOGE("Native_camera returned opening error: %d", errcode);
    delete m_activity;
    m_activity = 0;
  }
}

CvCapture_Android::~CvCapture_Android()
{
  if (m_activity)
  {
    ((HighguiAndroidCameraActivity*)m_activity)->LogFramesRate();

    //m_activity->disconnect() will be automatically called inside destructor;
    delete m_activity;
    delete m_frameYUV;
    delete m_frameYUVnext;
    m_activity = 0;
    m_frameYUV = 0;
    m_frameYUVnext = 0;
    
    pthread_mutex_destroy(&m_nextFrameMutex);
    pthread_cond_destroy(&m_nextFrameCond);
  }
}

double CvCapture_Android::getProperty( int propIdx )
{
  switch ( propIdx )
  {
    case CV_CAP_PROP_FRAME_WIDTH:
      return (double)CameraActivity::getFrameWidth();
    case CV_CAP_PROP_FRAME_HEIGHT:
      return (double)CameraActivity::getFrameHeight();
    default:
      CV_Error( CV_StsError, "Failed attempt to GET unsupported camera property." );
      break;
  }
  return -1.0;
}

bool CvCapture_Android::setProperty( int propIdx, double propValue )
{
  bool res = false;
  if( isOpened() )
  {
    switch ( propIdx )
    {
      default:
	CV_Error( CV_StsError, "Failed attempt to SET unsupported camera property." );
	break;
    }
  }

  return res;
}

bool CvCapture_Android::grabFrame()
{
  if( !isOpened() )
    return false;

  pthread_mutex_lock(&m_nextFrameMutex);
  m_waitingNextFrame = true;
  pthread_cond_wait(&m_nextFrameCond, &m_nextFrameMutex);
  pthread_mutex_unlock(&m_nextFrameMutex);
  m_framesGrabbed++;
  return true;
}

void CvCapture_Android::onFrame(const void* buffer, int bufferSize)
{
   LOGD("Buffer available: %p + %d", buffer, bufferSize);

   convertBufferToYUV(buffer, bufferSize, CameraActivity::getFrameWidth(), CameraActivity::getFrameHeight());

   //swap current and new frames
   OutputMap* tmp = m_frameYUV;
   m_frameYUV = m_frameYUVnext;
   m_frameYUVnext = tmp;

   //discard cached frames
   m_hasGray = false;
   m_hasColor = false;
}

IplImage* CvCapture_Android::retrieveFrame( int outputType )
{
  IplImage* image = 0;
  if (0 != m_frameYUV && !m_frameYUV->mat.empty())
  {
    switch(outputType)
    {
      case CV_CAP_ANDROID_YUV_FRAME:
	image = m_frameYUV->getIplImagePtr();
	break;
      case CV_CAP_ANDROID_GREY_FRAME:
	if (!m_hasGray)
	  if (!(m_hasGray = convertYUVToGrey(m_frameYUV->mat, m_frameGray.mat)))
	    image = 0;
	image = m_frameGray.getIplImagePtr();
	break;
      case CV_CAP_ANDROID_COLOR_FRAME:
	if (!m_hasColor)
	  if (!(m_hasColor = convertYUVToColor(m_frameYUV->mat, m_frameColor.mat)))
	    image = 0;
	image = m_frameColor.getIplImagePtr();
	break;
      default:
	LOGE("Unsupported frame output format: %d", outputType);
	image = 0;
	break;
    }
  }
  return image;
}


void CvCapture_Android::convertBufferToYUV(const void* buffer, int size, int width, int height)
{
  cv::Size buffSize(width, height + (height / 2));
  if (buffSize.area() != size)
  {
    LOGE("ERROR convertBufferToYuv_Mat: width=%d, height=%d, buffSize=%d x %d, buffSize.area()=%d, size=%d",
	 width, height, buffSize.width, buffSize.height, buffSize.area(), size);
	 
    return;
  }

  m_frameYUVnext->mat.create(buffSize, CV_8UC1);
  uchar* matBuff = m_frameYUVnext->mat.ptr<uchar> (0);
  memcpy(matBuff, buffer, size);
}

bool CvCapture_Android::convertYUVToGrey(const cv::Mat& yuv, cv::Mat& resmat)
{
  if (yuv.empty())
    return false;

  resmat = yuv(cv::Range(0, yuv.rows * (2.0f / 3)), cv::Range::all());

  return !resmat.empty();
}

bool CvCapture_Android::convertYUVToColor(const cv::Mat& yuv, cv::Mat& resmat)
{
  if (yuv.empty())
    return false;

  cv::cvtColor(yuv, resmat, CV_YUV2RGB);
  return !resmat.empty();
}


CvCapture* cvCreateCameraCapture_Android( int /*index*/ )
{
    CvCapture_Android* capture = new CvCapture_Android();

    if( capture->isOpened() )
        return capture;

    delete capture;
    return 0;
}

#endif
