// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef CAP_INTERFACE_HPP
#define CAP_INTERFACE_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/videoio.hpp"
#include "opencv2/videoio/videoio_c.h"

//===================================================

// Legacy structs

struct CvCapture
{
    virtual ~CvCapture() {}
    virtual double getProperty(int) const { return 0; }
    virtual bool setProperty(int, double) { return 0; }
    virtual bool grabFrame() { return true; }
    virtual IplImage* retrieveFrame(int) { return 0; }
    virtual int getCaptureDomain() { return cv::CAP_ANY; } // Return the type of the capture object: CAP_DSHOW, etc...
};

struct CvVideoWriter
{
    virtual ~CvVideoWriter() {}
    virtual bool writeFrame(const IplImage*) { return false; }
    virtual int getCaptureDomain() const { return cv::CAP_ANY; } // Return the type of the capture object: CAP_FFMPEG, etc...
};

//===================================================

// Modern classes

namespace cv
{

class IVideoCapture
{
public:
    virtual ~IVideoCapture() {}
    virtual double getProperty(int) const { return 0; }
    virtual bool setProperty(int, double) { return false; }
    virtual bool grabFrame() = 0;
    virtual bool retrieveFrame(int, OutputArray) = 0;
    virtual bool isOpened() const = 0;
    virtual int getCaptureDomain() { return CAP_ANY; } // Return the type of the capture object: CAP_DSHOW, etc...
};

class IVideoWriter
{
public:
    virtual ~IVideoWriter() {}
    virtual double getProperty(int) const { return 0; }
    virtual bool setProperty(int, double) { return false; }
    virtual bool isOpened() const = 0;
    virtual void write(InputArray) = 0;
    virtual int getCaptureDomain() const { return cv::CAP_ANY; } // Return the type of the capture object: CAP_FFMPEG, etc...
};

namespace internal {
class VideoCapturePrivateAccessor
{
public:
    static
    IVideoCapture* getIVideoCapture(const VideoCapture& cap) { return cap.icap.get(); }
};
} // namespace

//===================================================

// Wrapper

class LegacyCapture : public IVideoCapture
{
private:
    CvCapture * cap;
    LegacyCapture(const LegacyCapture &);
    LegacyCapture& operator=(const LegacyCapture &);
public:
    LegacyCapture(CvCapture * cap_) : cap(cap_) {}
    ~LegacyCapture()
    {
        cvReleaseCapture(&cap);
    }
    double getProperty(int propId) const CV_OVERRIDE
    {
        return cap ? cap->getProperty(propId) : 0;
    }
    bool setProperty(int propId, double value) CV_OVERRIDE
    {
        return cvSetCaptureProperty(cap, propId, value) != 0;
    }
    bool grabFrame() CV_OVERRIDE
    {
        return cap ? cvGrabFrame(cap) != 0 : false;
    }
    bool retrieveFrame(int channel, OutputArray image) CV_OVERRIDE
    {
        IplImage* _img = cvRetrieveFrame(cap, channel);
        if( !_img )
        {
            image.release();
            return false;
        }
        if(_img->origin == IPL_ORIGIN_TL)
        {
            cv::cvarrToMat(_img).copyTo(image);
        }
        else
        {
            Mat temp = cv::cvarrToMat(_img);
            flip(temp, image, 0);
        }
        return true;
    }
    bool isOpened() const CV_OVERRIDE
    {
        return cap != 0;  // legacy interface doesn't support closed files
    }
    int getCaptureDomain() CV_OVERRIDE
    {
        return cap ? cap->getCaptureDomain() : 0;
    }

    CvCapture* getCvCapture() const { return cap; }
};

class LegacyWriter : public IVideoWriter
{
private:
    CvVideoWriter * writer;
    LegacyWriter(const LegacyWriter &);
    LegacyWriter& operator=(const LegacyWriter &);
public:
    LegacyWriter(CvVideoWriter * wri_) : writer(wri_)
    {}
    ~LegacyWriter()
    {
        cvReleaseVideoWriter(&writer);
    }
    double getProperty(int) const CV_OVERRIDE
    {
        return 0.;
    }
    bool setProperty(int, double) CV_OVERRIDE
    {
        return false;
    }
    bool isOpened() const CV_OVERRIDE
    {
        return writer != NULL;
    }
    void write(InputArray image) CV_OVERRIDE
    {
        IplImage _img = cvIplImage(image.getMat());
        cvWriteFrame(writer, &_img);
    }
    int getCaptureDomain() const CV_OVERRIDE
    {
        return writer ? writer->getCaptureDomain() : 0;
    }
};

//==================================================================================================

Ptr<IVideoCapture> cvCreateFileCapture_FFMPEG_proxy(const std::string &filename);
Ptr<IVideoWriter> cvCreateVideoWriter_FFMPEG_proxy(const std::string& filename, int fourcc, double fps, const Size &frameSize, bool isColor);

Ptr<IVideoCapture> createGStreamerCapture_file(const std::string& filename);
Ptr<IVideoCapture> createGStreamerCapture_cam(int index);
Ptr<IVideoWriter> create_GStreamer_writer(const std::string& filename, int fourcc, double fps, const Size &frameSize, bool isColor);

Ptr<IVideoCapture> create_MFX_capture(const std::string &filename);
Ptr<IVideoWriter> create_MFX_writer(const std::string &filename, int _fourcc, double fps, const Size &frameSize, bool isColor);

Ptr<IVideoCapture> create_AVFoundation_capture_file(const std::string &filename);
Ptr<IVideoCapture> create_AVFoundation_capture_cam(int index);
Ptr<IVideoWriter> create_AVFoundation_writer(const std::string& filename, int fourcc, double fps, const Size &frameSize, bool isColor);

Ptr<IVideoCapture> create_WRT_capture(int device);

Ptr<IVideoCapture> cvCreateCapture_MSMF(int index);
Ptr<IVideoCapture> cvCreateCapture_MSMF(const std::string& filename);
Ptr<IVideoWriter> cvCreateVideoWriter_MSMF(const std::string& filename, int fourcc, double fps, const Size &frameSize, bool is_color);

Ptr<IVideoCapture> create_DShow_capture(int index);

Ptr<IVideoCapture> create_V4L_capture_cam(int index);
Ptr<IVideoCapture> create_V4L_capture_file(const std::string &filename);

Ptr<IVideoCapture> create_OpenNI2_capture_cam( int index );
Ptr<IVideoCapture> create_OpenNI2_capture_file( const std::string &filename );

Ptr<IVideoCapture> create_Images_capture(const std::string &filename);
Ptr<IVideoWriter> create_Images_writer(const std::string &filename, int fourcc, double fps, const Size &frameSize, bool iscolor);

Ptr<IVideoCapture> create_DC1394_capture(int index);

Ptr<IVideoCapture> create_RealSense_capture(int index);

Ptr<IVideoCapture> create_PvAPI_capture( int index );

Ptr<IVideoCapture> create_XIMEA_capture_cam( int index );
Ptr<IVideoCapture> create_XIMEA_capture_file( const std::string &serialNumber );

Ptr<IVideoCapture> create_Aravis_capture( int index );

Ptr<IVideoCapture> createMotionJpegCapture(const std::string& filename);
Ptr<IVideoWriter> createMotionJpegWriter(const std::string &filename, int fourcc, double fps, const Size &frameSize, bool iscolor);

Ptr<IVideoCapture> createGPhoto2Capture(int index);
Ptr<IVideoCapture> createGPhoto2Capture(const std::string& deviceName);

Ptr<IVideoCapture> createXINECapture(const std::string &filename);

Ptr<IVideoCapture> createAndroidCapture_file(const std::string &filename);

bool VideoCapture_V4L_waitAny(
        const std::vector<VideoCapture>& streams,
        CV_OUT std::vector<int>& ready,
        int64 timeoutNs);

} // cv::

#endif // CAP_INTERFACE_HPP
