// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CAP_MFX_WRITER_HPP
#define CAP_MFX_WRITER_HPP

#include "precomp.hpp"

class MFXVideoSession;
class Plugin;
class DeviceHandler;
class WriteBitstream;
class SurfacePool;
class MFXVideoDECODE;
class MFXVideoENCODE;

class VideoWriter_IntelMFX : public cv::IVideoWriter
{
public:
    VideoWriter_IntelMFX(const cv::String &filename, int _fourcc, double fps, cv::Size frameSize, bool isColor);
    virtual ~VideoWriter_IntelMFX();
    virtual double getProperty(int) const;
    virtual bool setProperty(int, double);
    virtual bool isOpened() const;
    virtual void write(cv::InputArray input);
    static cv::Ptr<VideoWriter_IntelMFX> create(const cv::String& filename, int _fourcc, double fps, cv::Size frameSize, bool isColor);

    virtual int getCaptureDomain() const { return cv::CAP_INTEL_MFX; }
protected:
    bool write_one(cv::InputArray bgr);

private:
    VideoWriter_IntelMFX(const VideoWriter_IntelMFX &);
    VideoWriter_IntelMFX & operator=(const VideoWriter_IntelMFX &);

private:
    MFXVideoSession *session;
    Plugin *plugin;
    DeviceHandler *deviceHandler;
    WriteBitstream *bs;
    MFXVideoENCODE *encoder;
    SurfacePool *pool;
    void *outSurface;
    cv::Size frameSize;
    bool good;
};

#endif // CAP_MFX_WRITER_HPP
