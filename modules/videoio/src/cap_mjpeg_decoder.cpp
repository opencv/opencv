/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015, OpenCV Foundation, all rights reserved.
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
#include "opencv2/videoio/container_avi.private.hpp"

namespace cv
{

class MotionJpegCapture: public IVideoCapture
{
public:
    virtual ~MotionJpegCapture() CV_OVERRIDE;
    virtual double getProperty(int) const CV_OVERRIDE;
    virtual bool setProperty(int, double) CV_OVERRIDE;
    virtual bool grabFrame() CV_OVERRIDE;
    virtual bool retrieveFrame(int, OutputArray) CV_OVERRIDE;
    virtual bool isOpened() const CV_OVERRIDE;
    virtual int getCaptureDomain() CV_OVERRIDE { return CAP_OPENCV_MJPEG; }
    MotionJpegCapture(const String&);

    bool open(const String&);
    void close();
protected:

    inline uint64_t getFramePos() const;

    Ptr<AVIReadContainer> m_avi_container;
    bool             m_is_first_frame;
    frame_list       m_mjpeg_frames;

    frame_iterator   m_frame_iterator;
    Mat              m_current_frame;

    //frame width/height and fps could be different for
    //each frame/stream. At the moment we suppose that they
    //stays the same within single avi file.
    uint32_t         m_frame_width;
    uint32_t         m_frame_height;
    double           m_fps;
};

uint64_t MotionJpegCapture::getFramePos() const
{
    if(m_is_first_frame)
        return 0;

    if(m_frame_iterator == m_mjpeg_frames.end())
        return m_mjpeg_frames.size();

    return m_frame_iterator - m_mjpeg_frames.begin() + 1;
}

bool MotionJpegCapture::setProperty(int property, double value)
{
    if(property == CAP_PROP_POS_FRAMES)
    {
        if(int(value) == 0)
        {
            m_is_first_frame = true;
            m_frame_iterator = m_mjpeg_frames.end();
            return true;
        }
        else if(m_mjpeg_frames.size() > value)
        {
            m_frame_iterator = m_mjpeg_frames.begin() + int(value - 1);
            m_is_first_frame = false;
            return true;
        }
    }

    return false;
}

double MotionJpegCapture::getProperty(int property) const
{
    switch(property)
    {
        case CAP_PROP_POS_FRAMES:
            return (double)getFramePos();
        case CAP_PROP_POS_AVI_RATIO:
            return double(getFramePos())/m_mjpeg_frames.size();
        case CAP_PROP_FRAME_WIDTH:
            return (double)m_frame_width;
        case CAP_PROP_FRAME_HEIGHT:
            return (double)m_frame_height;
        case CAP_PROP_FPS:
            return m_fps;
        case CAP_PROP_FOURCC:
            return (double)CV_FOURCC('M','J','P','G');
        case CAP_PROP_FRAME_COUNT:
            return (double)m_mjpeg_frames.size();
        case CAP_PROP_FORMAT:
            return 0;
        default:
            return 0;
    }
}

bool MotionJpegCapture::grabFrame()
{
    if(isOpened())
    {
        if(m_is_first_frame)
        {
            m_is_first_frame = false;
            m_frame_iterator = m_mjpeg_frames.begin();
        }
        else
        {
            if (m_frame_iterator == m_mjpeg_frames.end())
                return false;

            ++m_frame_iterator;
        }
    }

    return m_frame_iterator != m_mjpeg_frames.end();
}

bool MotionJpegCapture::retrieveFrame(int, OutputArray output_frame)
{
    if(m_frame_iterator != m_mjpeg_frames.end())
    {
        std::vector<char> data = m_avi_container->readFrame(m_frame_iterator);

        if(data.size())
        {
            m_current_frame = imdecode(data, IMREAD_ANYDEPTH | IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION);
        }

        m_current_frame.copyTo(output_frame);

        return true;
    }

    return false;
}

MotionJpegCapture::~MotionJpegCapture()
{
    close();
}

MotionJpegCapture::MotionJpegCapture(const String& filename)
{
    m_avi_container = makePtr<AVIReadContainer>();
    m_avi_container->initStream(filename);
    open(filename);
}

bool MotionJpegCapture::isOpened() const
{
    return m_mjpeg_frames.size() > 0;
}

void MotionJpegCapture::close()
{
    m_avi_container->close();
    m_frame_iterator = m_mjpeg_frames.end();
}

bool MotionJpegCapture::open(const String& filename)
{
    close();

    m_avi_container = makePtr<AVIReadContainer>();
    m_avi_container->initStream(filename);

    m_frame_iterator = m_mjpeg_frames.end();
    m_is_first_frame = true;

    if(!m_avi_container->parseRiff(m_mjpeg_frames))
    {
        close();
    } else
    {
        m_frame_width = m_avi_container->getWidth();
        m_frame_height = m_avi_container->getHeight();
        m_fps = m_avi_container->getFps();
    }

    return isOpened();
}

Ptr<IVideoCapture> createMotionJpegCapture(const String& filename)
{
    Ptr<MotionJpegCapture> mjdecoder(new MotionJpegCapture(filename));
    if( mjdecoder->isOpened() )
        return mjdecoder;
    return Ptr<MotionJpegCapture>();
}

}
