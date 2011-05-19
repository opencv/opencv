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

#if !defined(LOGD) && !defined(LOGI) && !defined(LOGE)
#define LOG_TAG "CV_CAP"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))
#endif

class HighguiAndroidCameraActivity;

class CvCapture_Android : public CvCapture
{
public:
    CvCapture_Android(int);
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

    //raw from camera
    int m_width;
    int m_height;
    unsigned char *m_frameYUV420i;
    unsigned char *m_frameYUV420inext;

    void setFrame(const void* buffer, int bufferSize);

private:
    bool m_isOpened;
    bool m_CameraParamsChanged;

    //frames counter for statistics
    int m_framesGrabbed;

    //cached converted frames
    OutputMap m_frameGray;
    OutputMap m_frameColor;
    bool m_hasGray;
    bool m_hasColor;

    enum CvCapture_Android_DataState {
	    CVCAPTURE_ANDROID_STATE_NO_FRAME=0,
	    CVCAPTURE_ANDROID_STATE_HAS_NEW_FRAME_UNGRABBED,
	    CVCAPTURE_ANDROID_STATE_HAS_FRAME_GRABBED
    };
    volatile CvCapture_Android_DataState m_dataState;

    //synchronization
    pthread_mutex_t m_nextFrameMutex;
    pthread_cond_t m_nextFrameCond;
    volatile bool m_waitingNextFrame;
    volatile bool m_shouldAutoGrab;

    void prepareCacheForYUV420i(int width, int height);
    static bool convertYUV420i2Grey(int width, int height, const unsigned char* yuv, cv::Mat& resmat);
    static bool convertYUV420i2BGR888(int width, int height, const unsigned char* yuv, cv::Mat& resmat, bool inRGBorder);

    friend class HighguiAndroidCameraActivity;
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
        if(isConnected() && buffer != 0 && bufferSize > 0)
        {
            m_framesReceived++;
            if (m_capture->m_waitingNextFrame || m_capture->m_shouldAutoGrab)
            {
                pthread_mutex_lock(&m_capture->m_nextFrameMutex);

                m_capture->setFrame(buffer, bufferSize);

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

CvCapture_Android::CvCapture_Android(int cameraId)
{
    //defaults
    m_width               = 0;
    m_height              = 0;
    m_activity            = 0;
    m_isOpened            = false;
    m_frameYUV420i        = 0;
    m_frameYUV420inext    = 0;
    m_hasGray             = false;
    m_hasColor            = false;
    m_dataState           = CVCAPTURE_ANDROID_STATE_NO_FRAME;
    m_waitingNextFrame    = false;
    m_shouldAutoGrab      = false;
    m_framesGrabbed       = 0;
    m_CameraParamsChanged = false;

    //try connect to camera
    m_activity = new HighguiAndroidCameraActivity(this);

    if (m_activity == 0) return;

    pthread_mutex_init(&m_nextFrameMutex, NULL);
    pthread_cond_init (&m_nextFrameCond,  NULL);

    CameraActivity::ErrorCode errcode = m_activity->connect(cameraId);

    if(errcode == CameraActivity::NO_ERROR)
        m_isOpened = true;
    else
    {
        LOGE("Native_camera returned opening error: %d", errcode);
        delete m_activity;
        m_activity = 0;
    }
}

bool CvCapture_Android::isOpened() const
{
    return m_isOpened;
}

CvCapture_Android::~CvCapture_Android()
{
    if (m_activity)
    {
        ((HighguiAndroidCameraActivity*)m_activity)->LogFramesRate();


	pthread_mutex_lock(&m_nextFrameMutex);

	unsigned char *tmp1=m_frameYUV420i;
	unsigned char *tmp2=m_frameYUV420inext;
        m_frameYUV420i = 0;
        m_frameYUV420inext = 0;
        delete tmp1;
        delete tmp2;

	m_dataState=CVCAPTURE_ANDROID_STATE_NO_FRAME;
	pthread_cond_broadcast(&m_nextFrameCond);

	pthread_mutex_unlock(&m_nextFrameMutex);

        //m_activity->disconnect() will be automatically called inside destructor;
        delete m_activity;
        m_activity = 0;

        pthread_mutex_destroy(&m_nextFrameMutex);
        pthread_cond_destroy(&m_nextFrameCond);
    }
}

double CvCapture_Android::getProperty( int propIdx )
{
  switch ( propIdx )
    {
    case CV_CAP_PROP_FRAME_WIDTH:
        return (double)m_activity->getFrameWidth();
    case CV_CAP_PROP_FRAME_HEIGHT:
        return (double)m_activity->getFrameHeight();

    case CV_CAP_PROP_SUPPORTED_PREVIEW_SIZES_STRING:
	return (double)m_activity->getProperty(ANDROID_CAMERA_PROPERTY_SUPPORTED_PREVIEW_SIZES_STRING);
    default:
        CV_Error( CV_StsOutOfRange, "Failed attempt to GET unsupported camera property." );
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
        case CV_CAP_PROP_FRAME_WIDTH:
            m_activity->setProperty(ANDROID_CAMERA_PROPERTY_FRAMEWIDTH, propValue);
            break;
        case CV_CAP_PROP_FRAME_HEIGHT:
            m_activity->setProperty(ANDROID_CAMERA_PROPERTY_FRAMEHEIGHT, propValue);
            break;

        case CV_CAP_PROP_AUTOGRAB:
	    m_shouldAutoGrab=(propValue != 0);
            break;

        default:
            CV_Error( CV_StsOutOfRange, "Failed attempt to SET unsupported camera property." );
	    return false;
        }

	if (propIdx != CV_CAP_PROP_AUTOGRAB) {// property for highgui class CvCapture_Android only
		m_CameraParamsChanged = true;
	}
	res = true;
    }

    return res;
}

bool CvCapture_Android::grabFrame()
{
    if( !isOpened() ) {
	    LOGE("CvCapture_Android::grabFrame(): camera is not opened");
	    return false;
    }

    bool res=false;
    pthread_mutex_lock(&m_nextFrameMutex);
    if (m_CameraParamsChanged)
    {
        m_activity->applyProperties();
        m_CameraParamsChanged = false;
	m_dataState= CVCAPTURE_ANDROID_STATE_NO_FRAME;//we will wait new frame
    }

    if (m_dataState!=CVCAPTURE_ANDROID_STATE_HAS_NEW_FRAME_UNGRABBED) {
	    m_waitingNextFrame = true;
	    pthread_cond_wait(&m_nextFrameCond, &m_nextFrameMutex);
    }

    if (m_dataState == CVCAPTURE_ANDROID_STATE_HAS_NEW_FRAME_UNGRABBED) {
	    LOGD("CvCapture_Android::grabFrame: get new frame");
	    //swap current and new frames
	    unsigned char* tmp = m_frameYUV420i;
	    m_frameYUV420i = m_frameYUV420inext;
	    m_frameYUV420inext = tmp;

	    //discard cached frames
	    m_hasGray = false;
	    m_hasColor = false;

	    m_dataState=CVCAPTURE_ANDROID_STATE_HAS_FRAME_GRABBED;
	    m_framesGrabbed++;

	    res=true;
    } else {
	    LOGE("CvCapture_Android::grabFrame: NO new frame");
    }


    int res_unlock=pthread_mutex_unlock(&m_nextFrameMutex);
    if (res_unlock) {
	    LOGE("Error in CvCapture_Android::grabFrame: pthread_mutex_unlock returned %d --- probably, this object has been destroyed", res_unlock);
	    return false;
    }

    return res;
}

IplImage* CvCapture_Android::retrieveFrame( int outputType )
{
    IplImage* image = NULL;

    unsigned char *current_frameYUV420i=m_frameYUV420i;
    //Attention! all the operations in this function below should occupy less time than the period between two frames from camera
    if (NULL != current_frameYUV420i)
    {
        switch(outputType)
        {
        case CV_CAP_ANDROID_COLOR_FRAME:
            if (!m_hasColor)
                if (!(m_hasColor = convertYUV420i2BGR888(m_width, m_height, current_frameYUV420i, m_frameColor.mat, false)))
                    return NULL;
            image = m_frameColor.getIplImagePtr();
            break;
        case CV_CAP_ANDROID_GREY_FRAME:
            if (!m_hasGray)
                if (!(m_hasGray = convertYUV420i2Grey(m_width, m_height, current_frameYUV420i, m_frameGray.mat)))
                    return NULL;
            image = m_frameGray.getIplImagePtr();
            break;
        case CV_CAP_ANDROID_COLOR_FRAME_RGB:
            if (!m_hasColor)
                if (!(m_hasColor = convertYUV420i2BGR888(m_width, m_height, current_frameYUV420i, m_frameColor.mat, true)))
                    return NULL;
            image = m_frameColor.getIplImagePtr();
            break;
        default:
            LOGE("Unsupported frame output format: %d", outputType);
            CV_Error( CV_StsOutOfRange, "Output frame format is not supported." );
            image = NULL;
            break;
        }
    }
    return image;
}

//Attention: this method should be called inside pthread_mutex_lock(m_nextFrameMutex) only
void CvCapture_Android::setFrame(const void* buffer, int bufferSize)
{
    int width = m_activity->getFrameWidth();
    int height = m_activity->getFrameHeight();
    int expectedSize = (width * height * 3) >> 1;

    if ( expectedSize != bufferSize)
    {
        LOGE("ERROR reading YUV420i buffer: width=%d, height=%d, size=%d, receivedSize=%d", width, height, expectedSize, bufferSize);
        return;
    }

    //allocate memory if needed
    prepareCacheForYUV420i(width, height);

    //copy data
    memcpy(m_frameYUV420inext, buffer, bufferSize);
    LOGD("CvCapture_Android::setFrame -- memcpy is done");

#if 0 //moved this part of code into grabFrame
    //swap current and new frames
    unsigned char* tmp = m_frameYUV420i;
    m_frameYUV420i = m_frameYUV420inext;
    m_frameYUV420inext = tmp;

    //discard cached frames
    m_hasGray = false;
    m_hasColor = false;
#endif

    m_dataState = CVCAPTURE_ANDROID_STATE_HAS_NEW_FRAME_UNGRABBED;
    m_waitingNextFrame = false;//set flag that no more frames required at this moment
}

//Attention: this method should be called inside pthread_mutex_lock(m_nextFrameMutex) only
void CvCapture_Android::prepareCacheForYUV420i(int width, int height)
{
    if (width != m_width || height != m_height)
    {
	LOGD("CvCapture_Android::prepareCacheForYUV420i: Changing size of buffers: from width=%d height=%d to width=%d height=%d", m_width, m_height, width, height);
        m_width = width;
        m_height = height;
        unsigned char *tmp = m_frameYUV420inext;
        m_frameYUV420inext = new unsigned char [width * height * 3 / 2];
	if (tmp != NULL) {
		delete[] tmp;
	}

        tmp = m_frameYUV420i;
        m_frameYUV420i = new unsigned char [width * height * 3 / 2];
	if (tmp != NULL) {
		delete[] tmp;
	}
    }
}

inline unsigned char clamp(int value)
{
    if (value <= 0)
        return 0;
    if (value >= 255)
        return (unsigned char)255;
    return (unsigned char)value;
}


bool CvCapture_Android::convertYUV420i2Grey(int width, int height, const unsigned char* yuv, cv::Mat& resmat)
{
    if (yuv == 0) return false;

    resmat.create(height, width, CV_8UC1);
    unsigned char* matBuff = resmat.ptr<unsigned char> (0);
    memcpy(matBuff, yuv, width * height);
    return !resmat.empty();
}

#ifndef HAVE_TEGRA_OPTIMIZATION
template<int R>
struct YUV420i2BGR888Invoker
{
	cv::Mat& dst;
	unsigned char* my1, *muv;
	int width;

	YUV420i2BGR888Invoker(cv::Mat& _dst, int _width, unsigned char* _y1, unsigned char* _uv)
		: dst(_dst), my1(_y1), muv(_uv), width(_width) {}

	void operator()(const cv::BlockedRange& range) const
    	{
		//B = 1.164(Y - 16)                  + 2.018(U - 128)
		//G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
		//R = 1.164(Y - 16) + 1.596(V - 128)

		unsigned char* y1 = my1 + range.begin() * width, *uv = muv + range.begin() * width / 2;

		for (int j = range.begin(); j < range.end(); j+=2, y1+=width*2, uv+=width)
		{
			unsigned char* row1 = dst.ptr<unsigned char>(j);
			unsigned char* row2 = dst.ptr<unsigned char>(j+1);
			unsigned char* y2 = y1 + width;

			for(int i = 0; i < width; i+=2,row1+=6,row2+=6)
			{
			    int cr = uv[i] - 128;
			    int cb = uv[i+1] - 128;

			    int ruv = 409 * cr + 128;
			    int guv = 128 - 100 * cb - 208 * cr;
			    int buv = 516 * cb + 128;

			    int y00 = (y1[i] - 16) * 298;
			    row1[0+R] = clamp((y00 + buv) >> 8);
			    row1[1] = clamp((y00 + guv) >> 8);
			    row1[2-R] = clamp((y00 + ruv) >> 8);

			    int y01 = (y1[i+1] - 16) * 298;
			    row1[3+R] = clamp((y01 + buv) >> 8);
			    row1[4] = clamp((y01 + guv) >> 8);
			    row1[5-R] = clamp((y01 + ruv) >> 8);

			    int y10 = (y2[i] - 16) * 298;
			    row2[0+R] = clamp((y10 + buv) >> 8);
			    row2[1] = clamp((y10 + guv) >> 8);
			    row2[2-R] = clamp((y10 + ruv) >> 8);

			    int y11 = (y2[i+1] - 16) * 298;
			    row2[3+R] = clamp((y11 + buv) >> 8);
			    row2[4] = clamp((y11 + guv) >> 8);
			    row2[5-R] = clamp((y11 + ruv) >> 8);
			}
		}
	}
};
#endif

bool CvCapture_Android::convertYUV420i2BGR888(int width, int height, const unsigned char* yuv, cv::Mat& resmat, bool inRGBorder)
{
    if (yuv == 0) return false;
    CV_Assert(width % 2 == 0 && height % 2 == 0);

    resmat.create(height, width, CV_8UC3);

    unsigned char* y1 = (unsigned char*)yuv;
    unsigned char* uv = y1 + width * height;

#ifdef HAVE_TEGRA_OPTIMIZATION
    cv::parallel_for(cv::BlockedRange(0, height, 2), tegra::YUV420i2BGR888Invoker(resmat, width, y1, uv, inRGBorder));
#else
    if (inRGBorder)
        cv::parallel_for(cv::BlockedRange(0, height, 2), YUV420i2BGR888Invoker<2>(resmat, width, y1, uv));
    else
        cv::parallel_for(cv::BlockedRange(0, height, 2), YUV420i2BGR888Invoker<0>(resmat, width, y1, uv));
#endif

    return !resmat.empty();
}

CvCapture* cvCreateCameraCapture_Android( int cameraId )
{
    CvCapture_Android* capture = new CvCapture_Android(cameraId);

    if( capture->isOpened() )
        return capture;

    delete capture;
    return 0;
}

#endif
