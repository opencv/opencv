#include "precomp.hpp"

#ifdef HAVE_INTELPERC

#include "pxcsession.h"
#include "pxcsmartptr.h"
#include "pxccapture.h"

class CvIntelPerCStreamBase
{
protected:
    struct FrameInternal
    {
        IplImage* retrieveFrame()
        {
            if (m_mat.empty())
                return NULL;
            m_iplHeader = IplImage(m_mat);
            return &m_iplHeader;
        }
        cv::Mat m_mat;
    private:
        IplImage m_iplHeader;
    };
public:
    CvIntelPerCStreamBase()
        : m_profileIdx(-1)
        , m_frameIdx(0)
        , m_timeStampStartNS(0)
    {
    }
    virtual ~CvIntelPerCStreamBase()
    {
    }

    bool isValid()
    {
        return (m_device.IsValid() && m_stream.IsValid());
    }
    bool grabFrame()
    {
        if (!m_stream.IsValid())
            return false;
        if (-1 == m_profileIdx)
        {
            if (!setProperty(CV_CAP_PROP_INTELPERC_PROFILE_IDX, 0))
                return false;
        }
        PXCSmartPtr<PXCImage> pxcImage; PXCSmartSP sp;
        if (PXC_STATUS_NO_ERROR > m_stream->ReadStreamAsync(&pxcImage, &sp))
            return false;
        if (PXC_STATUS_NO_ERROR > sp->Synchronize())
            return false;
        if (0 == m_timeStampStartNS)
            m_timeStampStartNS = pxcImage->QueryTimeStamp();
        m_timeStamp = (double)((pxcImage->QueryTimeStamp() - m_timeStampStartNS) / 10000);
        m_frameIdx++;
        return prepareIplImage(pxcImage);
    }
    int getProfileIDX() const
    {
        return m_profileIdx;
    }
public:
    virtual bool initStream(PXCSession *session)            = 0;
    virtual double getProperty(int propIdx)
    {
        double ret = 0.0;
        switch (propIdx)
        {
        case CV_CAP_PROP_INTELPERC_PROFILE_COUNT:
            ret = (double)m_profiles.size();
            break;
        case CV_CAP_PROP_FRAME_WIDTH :
            if ((0 <= m_profileIdx) && (m_profileIdx < m_profiles.size()))
                ret = (double)m_profiles[m_profileIdx].imageInfo.width;
            break;
        case CV_CAP_PROP_FRAME_HEIGHT :
            if ((0 <= m_profileIdx) && (m_profileIdx < m_profiles.size()))
                ret = (double)m_profiles[m_profileIdx].imageInfo.height;
            break;
        case CV_CAP_PROP_FPS :
            if ((0 <= m_profileIdx) && (m_profileIdx < m_profiles.size()))
            {
                ret = ((double)m_profiles[m_profileIdx].frameRateMin.numerator / (double)m_profiles[m_profileIdx].frameRateMin.denominator
                        + (double)m_profiles[m_profileIdx].frameRateMax.numerator / (double)m_profiles[m_profileIdx].frameRateMax.denominator) / 2.0;
            }
            break;
        case CV_CAP_PROP_POS_FRAMES:
            ret  = (double)m_frameIdx;
            break;
        case CV_CAP_PROP_POS_MSEC:
            ret  = m_timeStamp;
            break;
        };
        return ret;
    }
    virtual bool setProperty(int propIdx, double propVal)
    {
        bool isSet = false;
        switch (propIdx)
        {
        case CV_CAP_PROP_INTELPERC_PROFILE_IDX:
            {
                int propValInt = (int)propVal;
                if ((0 <= propValInt) && (propValInt < m_profiles.size()))
                {
                    if (m_profileIdx != propValInt)
                    {
                        m_profileIdx = propValInt;
                        if (m_stream.IsValid())
                            m_stream->SetProfile(&m_profiles[m_profileIdx]);
                        m_frameIdx = 0;
                        m_timeStampStartNS = 0;
                    }
                    isSet = true;
                }
            }
            break;
        };
        return isSet;
    }
protected:
    PXCSmartPtr<PXCCapture::Device> m_device;
    bool initDevice(PXCSession *session)
    {
        if (NULL == session)
            return false;

        pxcStatus sts = PXC_STATUS_NO_ERROR;
        PXCSession::ImplDesc templat;
        memset(&templat,0,sizeof(templat));
        templat.group   = PXCSession::IMPL_GROUP_SENSOR;
        templat.subgroup= PXCSession::IMPL_SUBGROUP_VIDEO_CAPTURE;

        for (int modidx = 0; PXC_STATUS_NO_ERROR <= sts; modidx++)
        {
            PXCSession::ImplDesc desc;
            sts = session->QueryImpl(&templat, modidx, &desc);
            if (PXC_STATUS_NO_ERROR > sts)
                break;

            PXCSmartPtr<PXCCapture> capture;
            sts = session->CreateImpl<PXCCapture>(&desc, &capture);
            if (!capture.IsValid())
                continue;

            /* enumerate devices */
            for (int devidx = 0; PXC_STATUS_NO_ERROR <= sts; devidx++)
            {
                PXCSmartPtr<PXCCapture::Device> device;
                sts = capture->CreateDevice(devidx, &device);
                if (PXC_STATUS_NO_ERROR <= sts)
                {
                    m_device = device.ReleasePtr();
                    return true;
                }
            }
        }
        return false;
    }

    PXCSmartPtr<PXCCapture::VideoStream> m_stream;
    void initStreamImpl(PXCImage::ImageType type)
    {
        if (!m_device.IsValid())
            return;

        pxcStatus sts = PXC_STATUS_NO_ERROR;
        /* enumerate streams */
        for (int streamidx = 0; PXC_STATUS_NO_ERROR <= sts; streamidx++)
        {
            PXCCapture::Device::StreamInfo sinfo;
            sts = m_device->QueryStream(streamidx, &sinfo);
            if (PXC_STATUS_NO_ERROR > sts)
                break;
            if (PXCCapture::VideoStream::CUID != sinfo.cuid)
                continue;
            if (type != sinfo.imageType)
                continue;

            sts = m_device->CreateStream<PXCCapture::VideoStream>(streamidx, &m_stream);
            if (PXC_STATUS_NO_ERROR == sts)
                break;
            m_stream.ReleaseRef();
        }
    }
protected:
    std::vector<PXCCapture::VideoStream::ProfileInfo> m_profiles;
    int m_profileIdx;
    int m_frameIdx;
    pxcU64 m_timeStampStartNS;
    double m_timeStamp;

    virtual bool validProfile(const PXCCapture::VideoStream::ProfileInfo& /*pinfo*/)
    {
        return true;
    }
    void enumProfiles()
    {
        m_profiles.clear();
        if (!m_stream.IsValid())
            return;
        pxcStatus sts = PXC_STATUS_NO_ERROR;
        for (int profidx = 0; PXC_STATUS_NO_ERROR <= sts; profidx++)
        {
            PXCCapture::VideoStream::ProfileInfo pinfo;
            sts = m_stream->QueryProfile(profidx, &pinfo);
            if (PXC_STATUS_NO_ERROR > sts)
                break;
            if (validProfile(pinfo))
                m_profiles.push_back(pinfo);
        }
    }
    virtual bool prepareIplImage(PXCImage *pxcImage) = 0;
};

class CvIntelPerCStreamImage
    : public CvIntelPerCStreamBase
{
public:
    CvIntelPerCStreamImage()
    {
    }
    virtual ~CvIntelPerCStreamImage()
    {
    }

    virtual bool initStream(PXCSession *session)
    {
        if (!initDevice(session))
            return false;
        initStreamImpl(PXCImage::IMAGE_TYPE_COLOR);
        if (!m_stream.IsValid())
            return false;
        enumProfiles();
        return true;
    }
    virtual double getProperty(int propIdx)
    {
        switch (propIdx)
        {
        case CV_CAP_PROP_BRIGHTNESS:
            {
                if (!m_device.IsValid())
                    return 0.0;
                float fret = 0.0f;
                if (PXC_STATUS_NO_ERROR == m_device->QueryProperty(PXCCapture::Device::PROPERTY_COLOR_BRIGHTNESS, &fret))
                    return (double)fret;
                return 0.0;
            }
            break;
        case CV_CAP_PROP_CONTRAST:
            {
                if (!m_device.IsValid())
                    return 0.0;
                float fret = 0.0f;
                if (PXC_STATUS_NO_ERROR == m_device->QueryProperty(PXCCapture::Device::PROPERTY_COLOR_CONTRAST, &fret))
                    return (double)fret;
                return 0.0;
            }
            break;
        case CV_CAP_PROP_SATURATION:
            {
                if (!m_device.IsValid())
                    return 0.0;
                float fret = 0.0f;
                if (PXC_STATUS_NO_ERROR == m_device->QueryProperty(PXCCapture::Device::PROPERTY_COLOR_SATURATION, &fret))
                    return (double)fret;
                return 0.0;
            }
            break;
        case CV_CAP_PROP_HUE:
            {
                if (!m_device.IsValid())
                    return 0.0;
                float fret = 0.0f;
                if (PXC_STATUS_NO_ERROR == m_device->QueryProperty(PXCCapture::Device::PROPERTY_COLOR_HUE, &fret))
                    return (double)fret;
                return 0.0;
            }
            break;
        case CV_CAP_PROP_GAMMA:
            {
                if (!m_device.IsValid())
                    return 0.0;
                float fret = 0.0f;
                if (PXC_STATUS_NO_ERROR == m_device->QueryProperty(PXCCapture::Device::PROPERTY_COLOR_GAMMA, &fret))
                    return (double)fret;
                return 0.0;
            }
            break;
        case CV_CAP_PROP_SHARPNESS:
            {
                if (!m_device.IsValid())
                    return 0.0;
                float fret = 0.0f;
                if (PXC_STATUS_NO_ERROR == m_device->QueryProperty(PXCCapture::Device::PROPERTY_COLOR_SHARPNESS, &fret))
                    return (double)fret;
                return 0.0;
            }
            break;
        case CV_CAP_PROP_GAIN:
            {
                if (!m_device.IsValid())
                    return 0.0;
                float fret = 0.0f;
                if (PXC_STATUS_NO_ERROR == m_device->QueryProperty(PXCCapture::Device::PROPERTY_COLOR_GAIN, &fret))
                    return (double)fret;
                return 0.0;
            }
            break;
        case CV_CAP_PROP_BACKLIGHT:
            {
                if (!m_device.IsValid())
                    return 0.0;
                float fret = 0.0f;
                if (PXC_STATUS_NO_ERROR == m_device->QueryProperty(PXCCapture::Device::PROPERTY_COLOR_BACK_LIGHT_COMPENSATION, &fret))
                    return (double)fret;
                return 0.0;
            }
            break;
        case CV_CAP_PROP_EXPOSURE:
            {
                if (!m_device.IsValid())
                    return 0.0;
                float fret = 0.0f;
                if (PXC_STATUS_NO_ERROR == m_device->QueryProperty(PXCCapture::Device::PROPERTY_COLOR_EXPOSURE, &fret))
                    return (double)fret;
                return 0.0;
            }
            break;
        //Add image stream specific properties
        }
        return CvIntelPerCStreamBase::getProperty(propIdx);
    }
    virtual bool setProperty(int propIdx, double propVal)
    {
        switch (propIdx)
        {
        case CV_CAP_PROP_BRIGHTNESS:
            {
                if (!m_device.IsValid())
                    return false;
                return (PXC_STATUS_NO_ERROR == m_device->SetProperty(PXCCapture::Device::PROPERTY_COLOR_BRIGHTNESS, (float)propVal));
            }
            break;
        case CV_CAP_PROP_CONTRAST:
            {
                if (!m_device.IsValid())
                    return false;
                return (PXC_STATUS_NO_ERROR == m_device->SetProperty(PXCCapture::Device::PROPERTY_COLOR_CONTRAST, (float)propVal));
            }
            break;
        case CV_CAP_PROP_SATURATION:
            {
                if (!m_device.IsValid())
                    return false;
                return (PXC_STATUS_NO_ERROR == m_device->SetProperty(PXCCapture::Device::PROPERTY_COLOR_SATURATION, (float)propVal));
            }
            break;
        case CV_CAP_PROP_HUE:
            {
                if (!m_device.IsValid())
                    return false;
                return (PXC_STATUS_NO_ERROR == m_device->SetProperty(PXCCapture::Device::PROPERTY_COLOR_HUE, (float)propVal));
            }
            break;
        case CV_CAP_PROP_GAMMA:
            {
                if (!m_device.IsValid())
                    return false;
                return (PXC_STATUS_NO_ERROR == m_device->SetProperty(PXCCapture::Device::PROPERTY_COLOR_GAMMA, (float)propVal));
            }
            break;
        case CV_CAP_PROP_SHARPNESS:
            {
                if (!m_device.IsValid())
                    return false;
                return (PXC_STATUS_NO_ERROR == m_device->SetProperty(PXCCapture::Device::PROPERTY_COLOR_SHARPNESS, (float)propVal));
            }
            break;
        case CV_CAP_PROP_GAIN:
            {
                if (!m_device.IsValid())
                    return false;
                return (PXC_STATUS_NO_ERROR == m_device->SetProperty(PXCCapture::Device::PROPERTY_COLOR_GAIN, (float)propVal));
            }
            break;
        case CV_CAP_PROP_BACKLIGHT:
            {
                if (!m_device.IsValid())
                    return false;
                return (PXC_STATUS_NO_ERROR == m_device->SetProperty(PXCCapture::Device::PROPERTY_COLOR_BACK_LIGHT_COMPENSATION, (float)propVal));
            }
            break;
        case CV_CAP_PROP_EXPOSURE:
            {
                if (!m_device.IsValid())
                    return false;
                return (PXC_STATUS_NO_ERROR == m_device->SetProperty(PXCCapture::Device::PROPERTY_COLOR_EXPOSURE, (float)propVal));
            }
            break;
        //Add image stream specific properties
        }
        return CvIntelPerCStreamBase::setProperty(propIdx, propVal);
    }
public:
    IplImage* retrieveFrame()
    {
        return m_frame.retrieveFrame();
    }
protected:
    FrameInternal m_frame;
    bool prepareIplImage(PXCImage *pxcImage)
    {
        if (NULL == pxcImage)
            return false;
        PXCImage::ImageInfo info;
        pxcImage->QueryInfo(&info);

        PXCImage::ImageData data;
        pxcImage->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::COLOR_FORMAT_RGB24, &data);

        if (PXCImage::SURFACE_TYPE_SYSTEM_MEMORY != data.type)
            return false;

        cv::Mat temp(info.height, info.width, CV_8UC3, data.planes[0], data.pitches[0]);
        temp.copyTo(m_frame.m_mat);

        pxcImage->ReleaseAccess(&data);
        return true;
    }
};

class CvIntelPerCStreamDepth
    : public CvIntelPerCStreamBase
{
public:
    CvIntelPerCStreamDepth()
    {
    }
    virtual ~CvIntelPerCStreamDepth()
    {
    }

    virtual bool initStream(PXCSession *session)
    {
        if (!initDevice(session))
            return false;
        initStreamImpl(PXCImage::IMAGE_TYPE_DEPTH);
        if (!m_stream.IsValid())
            return false;
        enumProfiles();
        return true;
    }
    virtual double getProperty(int propIdx)
    {
        switch (propIdx)
        {
        case CV_CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE:
            {
                if (!m_device.IsValid())
                    return 0.0;
                float fret = 0.0f;
                if (PXC_STATUS_NO_ERROR == m_device->QueryProperty(PXCCapture::Device::PROPERTY_DEPTH_LOW_CONFIDENCE_VALUE, &fret))
                    return (double)fret;
                return 0.0;
            }
            break;
        case CV_CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE:
            {
                if (!m_device.IsValid())
                    return 0.0;
                float fret = 0.0f;
                if (PXC_STATUS_NO_ERROR == m_device->QueryProperty(PXCCapture::Device::PROPERTY_DEPTH_SATURATION_VALUE, &fret))
                    return (double)fret;
                return 0.0;
            }
            break;
        case CV_CAP_PROP_INTELPERC_DEPTH_CONFIDENCE_THRESHOLD:
            {
                if (!m_device.IsValid())
                    return 0.0;
                float fret = 0.0f;
                if (PXC_STATUS_NO_ERROR == m_device->QueryProperty(PXCCapture::Device::PROPERTY_DEPTH_CONFIDENCE_THRESHOLD, &fret))
                    return (double)fret;
                return 0.0;
            }
            break;
        case CV_CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_HORZ:
            {
                if (!m_device.IsValid())
                    return 0.0f;
                PXCPointF32 ptf;
                if (PXC_STATUS_NO_ERROR == m_device->QueryPropertyAsPoint(PXCCapture::Device::PROPERTY_DEPTH_FOCAL_LENGTH, &ptf))
                    return (double)ptf.x;
                return 0.0;
            }
            break;
        case CV_CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_VERT:
            {
                if (!m_device.IsValid())
                    return 0.0f;
                PXCPointF32 ptf;
                if (PXC_STATUS_NO_ERROR == m_device->QueryPropertyAsPoint(PXCCapture::Device::PROPERTY_DEPTH_FOCAL_LENGTH, &ptf))
                    return (double)ptf.y;
                return 0.0;
            }
            break;
            //Add depth stream sepcific properties
        }
        return CvIntelPerCStreamBase::getProperty(propIdx);
    }
    virtual bool setProperty(int propIdx, double propVal)
    {
        switch (propIdx)
        {
        case CV_CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE:
            {
                if (!m_device.IsValid())
                    return false;
                return (PXC_STATUS_NO_ERROR == m_device->SetProperty(PXCCapture::Device::PROPERTY_DEPTH_LOW_CONFIDENCE_VALUE, (float)propVal));
            }
            break;
        case CV_CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE:
            {
                if (!m_device.IsValid())
                    return false;
                return (PXC_STATUS_NO_ERROR == m_device->SetProperty(PXCCapture::Device::PROPERTY_DEPTH_SATURATION_VALUE, (float)propVal));
            }
            break;
        case CV_CAP_PROP_INTELPERC_DEPTH_CONFIDENCE_THRESHOLD:
            {
                if (!m_device.IsValid())
                    return false;
                return (PXC_STATUS_NO_ERROR == m_device->SetProperty(PXCCapture::Device::PROPERTY_DEPTH_CONFIDENCE_THRESHOLD, (float)propVal));
            }
            break;
        //Add depth stream sepcific properties
        }
        return CvIntelPerCStreamBase::setProperty(propIdx, propVal);
    }
public:
    IplImage* retrieveDepthFrame()
    {
        return m_frameDepth.retrieveFrame();
    }
    IplImage* retrieveIRFrame()
    {
        return m_frameIR.retrieveFrame();
    }
    IplImage* retrieveUVFrame()
    {
        return m_frameUV.retrieveFrame();
    }
protected:
    virtual bool validProfile(const PXCCapture::VideoStream::ProfileInfo& pinfo)
    {
        return (PXCImage::COLOR_FORMAT_DEPTH == pinfo.imageInfo.format);
    }
protected:
    FrameInternal m_frameDepth;
    FrameInternal m_frameIR;
    FrameInternal m_frameUV;

    bool prepareIplImage(PXCImage *pxcImage)
    {
        if (NULL == pxcImage)
            return false;
        PXCImage::ImageInfo info;
        pxcImage->QueryInfo(&info);

        PXCImage::ImageData data;
        pxcImage->AcquireAccess(PXCImage::ACCESS_READ, &data);

        if (PXCImage::SURFACE_TYPE_SYSTEM_MEMORY != data.type)
            return false;

        if (PXCImage::COLOR_FORMAT_DEPTH != data.format)
            return false;

        {
            cv::Mat temp(info.height, info.width, CV_16SC1, data.planes[0], data.pitches[0]);
            temp.copyTo(m_frameDepth.m_mat);
        }
        {
            cv::Mat temp(info.height, info.width, CV_16SC1, data.planes[1], data.pitches[1]);
            temp.copyTo(m_frameIR.m_mat);
        }
        {
            cv::Mat temp(info.height, info.width, CV_32FC2, data.planes[2], data.pitches[2]);
            temp.copyTo(m_frameUV.m_mat);
        }

        pxcImage->ReleaseAccess(&data);
        return true;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class CvCapture_IntelPerC : public CvCapture
{
public:
    CvCapture_IntelPerC(int /*index*/)
        : m_contextOpened(false)
    {
        pxcStatus sts = PXCSession_Create(&m_session);
        if (PXC_STATUS_NO_ERROR > sts)
            return;
        m_contextOpened = m_imageStream.initStream(m_session);
        m_contextOpened &= m_depthStream.initStream(m_session);
    }
    virtual ~CvCapture_IntelPerC(){}

    virtual double getProperty(int propIdx)
    {
        double propValue = 0;
        int purePropIdx = propIdx & ~CV_CAP_INTELPERC_GENERATORS_MASK;
        if (CV_CAP_INTELPERC_IMAGE_GENERATOR == (propIdx & CV_CAP_INTELPERC_GENERATORS_MASK))
        {
            propValue = m_imageStream.getProperty(purePropIdx);
        }
        else if (CV_CAP_INTELPERC_DEPTH_GENERATOR == (propIdx & CV_CAP_INTELPERC_GENERATORS_MASK))
        {
            propValue = m_depthStream.getProperty(purePropIdx);
        }
        else
        {
            propValue = m_depthStream.getProperty(purePropIdx);
        }
        return propValue;
    }
    virtual bool setProperty(int propIdx, double propVal)
    {
        bool isSet = false;
        int purePropIdx = propIdx & ~CV_CAP_INTELPERC_GENERATORS_MASK;
        if (CV_CAP_INTELPERC_IMAGE_GENERATOR == (propIdx & CV_CAP_INTELPERC_GENERATORS_MASK))
        {
            isSet = m_imageStream.setProperty(purePropIdx, propVal);
        }
        else if (CV_CAP_INTELPERC_DEPTH_GENERATOR == (propIdx & CV_CAP_INTELPERC_GENERATORS_MASK))
        {
            isSet = m_depthStream.setProperty(purePropIdx, propVal);
        }
        else
        {
            isSet = m_depthStream.setProperty(purePropIdx, propVal);
        }
        return isSet;
    }

    bool grabFrame()
    {
        if (!isOpened())
            return false;

        bool isGrabbed = false;
        if (m_depthStream.isValid())
            isGrabbed = m_depthStream.grabFrame();
        if ((m_imageStream.isValid()) && (-1 != m_imageStream.getProfileIDX()))
            isGrabbed &= m_imageStream.grabFrame();

        return isGrabbed;
    }

    virtual IplImage* retrieveFrame(int outputType)
    {
        IplImage* image = 0;
        switch (outputType)
        {
        case CV_CAP_INTELPERC_DEPTH_MAP:
            image = m_depthStream.retrieveDepthFrame();
            break;
        case CV_CAP_INTELPERC_UVDEPTH_MAP:
            image = m_depthStream.retrieveUVFrame();
            break;
        case CV_CAP_INTELPERC_IR_MAP:
            image = m_depthStream.retrieveIRFrame();
            break;
        case CV_CAP_INTELPERC_IMAGE:
            image = m_imageStream.retrieveFrame();
            break;
        }
        CV_Assert(NULL != image);
        return image;
    }

    bool isOpened() const
    {
        return m_contextOpened;
    }
protected:
    bool m_contextOpened;

    PXCSmartPtr<PXCSession> m_session;
    CvIntelPerCStreamImage m_imageStream;
    CvIntelPerCStreamDepth m_depthStream;
};


CvCapture* cvCreateCameraCapture_IntelPerC(int index)
{
    CvCapture_IntelPerC* capture = new CvCapture_IntelPerC(index);

    if( capture->isOpened() )
        return capture;

    delete capture;
    return 0;
}


#endif //HAVE_INTELPERC
