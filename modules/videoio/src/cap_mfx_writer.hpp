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
    ~VideoWriter_IntelMFX() CV_OVERRIDE;
    double getProperty(int) const CV_OVERRIDE;
    bool setProperty(int, double) CV_OVERRIDE;
    bool isOpened() const CV_OVERRIDE;
    void write(cv::InputArray input) CV_OVERRIDE;
    int getCaptureDomain() const CV_OVERRIDE { return cv::CAP_INTEL_MFX; }
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
