#ifdef HAVE_INTELPERC

#include "cap_intelperc.hpp"

namespace cv
{

///////////////// IntelPerCStreamBase //////////////////

IntelPerCStreamBase::IntelPerCStreamBase()
    : m_profileIdx(-1)
    , m_frameIdx(0)
    , m_timeStampStartNS(0)
{
}
IntelPerCStreamBase::~IntelPerCStreamBase()
{
}

bool IntelPerCStreamBase::isValid()
{
    return (m_device.IsValid() && m_stream.IsValid());
}
bool IntelPerCStreamBase::grabFrame()
{
    if (!m_stream.IsValid())
        return false;
    if (-1 == m_profileIdx)
    {
        if (!setProperty(CV_CAP_PROP_INTELPERC_PROFILE_IDX, 0))
            return false;
    }
    PXCSmartSP sp;
    m_pxcImage.ReleaseRef();
    if (PXC_STATUS_NO_ERROR > m_stream->ReadStreamAsync(&m_pxcImage, &sp))
        return false;
    if (PXC_STATUS_NO_ERROR > sp->Synchronize())
        return false;
    if (0 == m_timeStampStartNS)
        m_timeStampStartNS = m_pxcImage->QueryTimeStamp();
    m_timeStamp = (double)((m_pxcImage->QueryTimeStamp() - m_timeStampStartNS) / 10000);
    m_frameIdx++;
    return true;
}
int IntelPerCStreamBase::getProfileIDX() const
{
    return m_profileIdx;
}
double IntelPerCStreamBase::getProperty(int propIdx) const
{
    double ret = -1.0;
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
bool IntelPerCStreamBase::setProperty(int propIdx, double propVal)
{
    bool isSet = false;
    switch (propIdx)
    {
    case CV_CAP_PROP_INTELPERC_PROFILE_IDX:
        {
            int propValInt = (int)propVal;
            if (0 > propValInt)
            {
                m_profileIdx = propValInt;
                isSet = true;
            }
            else if (propValInt < m_profiles.size())
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
bool IntelPerCStreamBase::initDevice(PXCSession *session)
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

void IntelPerCStreamBase::initStreamImpl(PXCImage::ImageType type)
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
bool IntelPerCStreamBase::validProfile(const PXCCapture::VideoStream::ProfileInfo& /*pinfo*/)
{
    return true;
}
void IntelPerCStreamBase::enumProfiles()
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

///////////////// IntelPerCStreamImage //////////////////

IntelPerCStreamImage::IntelPerCStreamImage()
{
}
IntelPerCStreamImage::~IntelPerCStreamImage()
{
}

bool IntelPerCStreamImage::initStream(PXCSession *session)
{
    if (!initDevice(session))
        return false;
    initStreamImpl(PXCImage::IMAGE_TYPE_COLOR);
    if (!m_stream.IsValid())
        return false;
    enumProfiles();
    return true;
}
double IntelPerCStreamImage::getProperty(int propIdx) const
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
    return IntelPerCStreamBase::getProperty(propIdx);
}
bool IntelPerCStreamImage::setProperty(int propIdx, double propVal)
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
    return IntelPerCStreamBase::setProperty(propIdx, propVal);
}
bool IntelPerCStreamImage::retrieveAsOutputArray(cv::OutputArray image)
{
    if (!m_pxcImage.IsValid())
        return false;
    PXCImage::ImageInfo info;
    m_pxcImage->QueryInfo(&info);

    PXCImage::ImageData data;
    m_pxcImage->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::COLOR_FORMAT_RGB24, &data);

    if (PXCImage::SURFACE_TYPE_SYSTEM_MEMORY != data.type)
        return false;

    cv::Mat temp(info.height, info.width, CV_8UC3, data.planes[0], data.pitches[0]);
    temp.copyTo(image);

    m_pxcImage->ReleaseAccess(&data);
    return true;
}

///////////////// IntelPerCStreamDepth //////////////////

IntelPerCStreamDepth::IntelPerCStreamDepth()
{
}
IntelPerCStreamDepth::~IntelPerCStreamDepth()
{
}

bool IntelPerCStreamDepth::initStream(PXCSession *session)
{
    if (!initDevice(session))
        return false;
    initStreamImpl(PXCImage::IMAGE_TYPE_DEPTH);
    if (!m_stream.IsValid())
        return false;
    enumProfiles();
    return true;
}
double IntelPerCStreamDepth::getProperty(int propIdx) const
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
    return IntelPerCStreamBase::getProperty(propIdx);
}
bool IntelPerCStreamDepth::setProperty(int propIdx, double propVal)
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
    return IntelPerCStreamBase::setProperty(propIdx, propVal);
}
bool IntelPerCStreamDepth::retrieveDepthAsOutputArray(cv::OutputArray image)
{
    return retrieveFrame(CV_16SC1, 0, image);
}
bool IntelPerCStreamDepth::retrieveIRAsOutputArray(cv::OutputArray image)
{
    return retrieveFrame(CV_16SC1, 1, image);
}
bool IntelPerCStreamDepth::retrieveUVAsOutputArray(cv::OutputArray image)
{
    return retrieveFrame(CV_32FC2, 2, image);
}
bool IntelPerCStreamDepth::validProfile(const PXCCapture::VideoStream::ProfileInfo& pinfo)
{
    return (PXCImage::COLOR_FORMAT_DEPTH == pinfo.imageInfo.format);
}
bool IntelPerCStreamDepth::retrieveFrame(int type, int planeIdx, cv::OutputArray frame)
{
    if (!m_pxcImage.IsValid())
        return false;
    PXCImage::ImageInfo info;
    m_pxcImage->QueryInfo(&info);

    PXCImage::ImageData data;
    m_pxcImage->AcquireAccess(PXCImage::ACCESS_READ, &data);

    if (PXCImage::SURFACE_TYPE_SYSTEM_MEMORY != data.type)
        return false;

    cv::Mat temp(info.height, info.width, type, data.planes[planeIdx], data.pitches[planeIdx]);
    temp.copyTo(frame);

    m_pxcImage->ReleaseAccess(&data);
    return true;
}

///////////////// VideoCapture_IntelPerC //////////////////

VideoCapture_IntelPerC::VideoCapture_IntelPerC()
    : m_contextOpened(false)
{
    pxcStatus sts = PXCSession_Create(&m_session);
    if (PXC_STATUS_NO_ERROR > sts)
        return;
    m_contextOpened = m_imageStream.initStream(m_session);
    m_contextOpened &= m_depthStream.initStream(m_session);
}
VideoCapture_IntelPerC::~VideoCapture_IntelPerC(){}

double VideoCapture_IntelPerC::getProperty(int propIdx) const
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
bool VideoCapture_IntelPerC::setProperty(int propIdx, double propVal)
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

bool VideoCapture_IntelPerC::grabFrame()
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
bool VideoCapture_IntelPerC::retrieveFrame(int outputType, cv::OutputArray frame)
{
    switch (outputType)
    {
    case CV_CAP_INTELPERC_DEPTH_MAP:
        return m_depthStream.retrieveDepthAsOutputArray(frame);
    case CV_CAP_INTELPERC_UVDEPTH_MAP:
        return m_depthStream.retrieveUVAsOutputArray(frame);
    case CV_CAP_INTELPERC_IR_MAP:
        return m_depthStream.retrieveIRAsOutputArray(frame);
    case CV_CAP_INTELPERC_IMAGE:
        return m_imageStream.retrieveAsOutputArray(frame);
    }
    return false;
}
int VideoCapture_IntelPerC::getCaptureDomain()
{
    return CV_CAP_INTELPERC;
}

bool VideoCapture_IntelPerC::isOpened() const
{
    return m_contextOpened;
}

}

#endif //HAVE_INTELPERC
