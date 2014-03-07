/*M///////////////////////////////////////////////////////////////////////////////////////
//
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
//M*/

#ifndef _CAP_INTELPERC_HPP_
#define _CAP_INTELPERC_HPP_

#include "precomp.hpp"

#ifdef HAVE_INTELPERC

#include "pxcsession.h"
#include "pxcsmartptr.h"
#include "pxccapture.h"

namespace cv
{

class IntelPerCStreamBase
{
public:
    IntelPerCStreamBase();
    virtual ~IntelPerCStreamBase();

    bool isValid();
    bool grabFrame();
    int getProfileIDX() const;
public:
    virtual bool initStream(PXCSession *session)            = 0;
    virtual double getProperty(int propIdx);
    virtual bool setProperty(int propIdx, double propVal);
protected:
    PXCSmartPtr<PXCCapture::Device> m_device;
    bool initDevice(PXCSession *session);

    PXCSmartPtr<PXCCapture::VideoStream> m_stream;
    void initStreamImpl(PXCImage::ImageType type);
protected:
    std::vector<PXCCapture::VideoStream::ProfileInfo> m_profiles;
    int m_profileIdx;
    int m_frameIdx;
    pxcU64 m_timeStampStartNS;
    double m_timeStamp;
    PXCSmartPtr<PXCImage> m_pxcImage;

    virtual bool validProfile(const PXCCapture::VideoStream::ProfileInfo& /*pinfo*/);
    void enumProfiles();
};

class IntelPerCStreamImage
    : public IntelPerCStreamBase
{
public:
    IntelPerCStreamImage();
    virtual ~IntelPerCStreamImage();

    virtual bool initStream(PXCSession *session);
    virtual double getProperty(int propIdx);
    virtual bool setProperty(int propIdx, double propVal);
public:
    bool retrieveAsOutputArray(OutputArray image);
};

class IntelPerCStreamDepth
    : public IntelPerCStreamBase
{
public:
    IntelPerCStreamDepth();
    virtual ~IntelPerCStreamDepth();

    virtual bool initStream(PXCSession *session);
    virtual double getProperty(int propIdx);
    virtual bool setProperty(int propIdx, double propVal);
public:
    bool retrieveDepthAsOutputArray(OutputArray image);
    bool retrieveIRAsOutputArray(OutputArray image);
    bool retrieveUVAsOutputArray(OutputArray image);
protected:
    virtual bool validProfile(const PXCCapture::VideoStream::ProfileInfo& pinfo);
protected:
    bool retriveFrame(int type, int planeIdx, OutputArray frame);
};

class VideoCapture_IntelPerC : public IVideoCapture
{
public:
    VideoCapture_IntelPerC();
    virtual ~VideoCapture_IntelPerC();

    virtual double getProperty(int propIdx);
    virtual bool setProperty(int propIdx, double propVal);

    virtual bool grabFrame();
    virtual bool retrieveFrame(int outputType, OutputArray frame);
    virtual int getCaptureDomain();
    bool isOpened() const;
protected:
    bool m_contextOpened;

    PXCSmartPtr<PXCSession> m_session;
    IntelPerCStreamImage m_imageStream;
    IntelPerCStreamDepth m_depthStream;
};

}

#endif //HAVE_INTELPERC
#endif //_CAP_INTELPERC_HPP_