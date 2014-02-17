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