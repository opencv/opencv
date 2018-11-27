// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CAP_MFX_HPP
#define CAP_MFX_HPP

#include "precomp.hpp"


class MFXVideoSession;
class Plugin;
class DeviceHandler;
class ReadBitstream;
class SurfacePool;
class MFXVideoDECODE;

class VideoCapture_IntelMFX : public cv::IVideoCapture
{
public:
    VideoCapture_IntelMFX(const cv::String &filename);
    virtual ~VideoCapture_IntelMFX();
    virtual double getProperty(int) const CV_OVERRIDE;
    virtual bool setProperty(int, double) CV_OVERRIDE;
    virtual bool grabFrame() CV_OVERRIDE;
    virtual bool retrieveFrame(int, cv::OutputArray out) CV_OVERRIDE;
    virtual bool isOpened() const CV_OVERRIDE;
    virtual int getCaptureDomain() CV_OVERRIDE;
private:
    MFXVideoSession *session;
    Plugin *plugin;
    DeviceHandler *deviceHandler;
    ReadBitstream *bs;
    MFXVideoDECODE *decoder;
    SurfacePool *pool;
    void *outSurface;
    bool good;
};


#endif
