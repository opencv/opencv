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
#include "cap_interface.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#ifdef HAVE_OPENNI2

#include <queue>

#ifndef i386
#  define i386 0
#endif
#ifndef __arm__
#  define __arm__ 0
#endif
#ifndef _ARC
#  define _ARC 0
#endif
#ifndef __APPLE__
#  define __APPLE__ 0
#endif

#define CV_STREAM_TIMEOUT 2000

#define CV_DEPTH_STREAM     0
#define CV_COLOR_STREAM     1
#define CV_IR_STREAM        2
#define CV_MAX_NUM_STREAMS  3

#include "OpenNI.h"
#include "PS1080.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static cv::Mutex initOpenNI2Mutex;

struct OpenNI2Initializer
{
public:
    static void init()
    {
        cv::AutoLock al(initOpenNI2Mutex);
        static OpenNI2Initializer initializer;
    }

private:
    OpenNI2Initializer()
    {
        // Initialize and configure the context.
        openni::Status status = openni::OpenNI::initialize();
        if (status != openni::STATUS_OK)
        {
            CV_Error(CV_StsError, std::string("Failed to initialize:") + openni::OpenNI::getExtendedError());
        }
    }

    ~OpenNI2Initializer()
    {
        openni::OpenNI::shutdown();
    }
};

class CvCapture_OpenNI2 : public CvCapture
{
public:
    enum { DEVICE_DEFAULT=0, DEVICE_MS_KINECT=0, DEVICE_ASUS_XTION=1, DEVICE_MAX=1 };

    static const int INVALID_PIXEL_VAL = 0;
    static const int INVALID_COORDINATE_VAL = 0;

    static const int DEFAULT_MAX_BUFFER_SIZE = 2;
    static const int DEFAULT_IS_CIRCLE_BUFFER = 0;
    static const int DEFAULT_MAX_TIME_DURATION = 20;

    CvCapture_OpenNI2(int index = 0);
    CvCapture_OpenNI2(const char * filename);
    virtual ~CvCapture_OpenNI2();

    virtual double getProperty(int propIdx) const CV_OVERRIDE;
    virtual bool setProperty(int probIdx, double propVal) CV_OVERRIDE;
    virtual bool grabFrame() CV_OVERRIDE;
    virtual IplImage* retrieveFrame(int outputType) CV_OVERRIDE;

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

    static const int outputMapsTypesCount = 8;

    static openni::VideoMode defaultStreamOutputMode(int stream);

    CvCapture_OpenNI2(int index, const char * filename);

    IplImage* retrieveDepthMap();
    IplImage* retrievePointCloudMap();
    IplImage* retrieveDisparityMap();
    IplImage* retrieveDisparityMap_32F();
    IplImage* retrieveValidDepthMask();
    IplImage* retrieveBGRImage();
    IplImage* retrieveGrayImage();
    IplImage* retrieveIrImage();

    void toggleStream(int stream, bool toggle);
    void readCamerasParams();

    double getDepthGeneratorProperty(int propIdx) const;
    bool setDepthGeneratorProperty(int propIdx, double propVal);
    double getImageGeneratorProperty(int propIdx) const;
    bool setImageGeneratorProperty(int propIdx, double propVal);
    double getIrGeneratorProperty(int propIdx) const;
    bool setIrGeneratorProperty(int propIdx, double propVal);
    double getCommonProperty(int propIdx) const;
    bool setCommonProperty(int propIdx, double propVal);

    // OpenNI context
    openni::Device device;
    bool isContextOpened;

    // Data generators with its metadata
    std::vector<openni::VideoStream> streams;
    std::vector<openni::VideoFrameRef> streamFrames;
    std::vector<cv::Mat> streamImages;

    int maxBufferSize, maxTimeDuration; // for approx sync
    bool isCircleBuffer;
    //cv::Ptr<ApproximateSyncGrabber> approxSyncGrabber;

    // Cameras settings:
    // TODO find in OpenNI function to convert z->disparity and remove fields "baseline" and depthFocalLength_VGA
    // Distance between IR projector and IR camera (in meters)
    double baseline;
    // Focal length for the IR camera in VGA resolution (in pixels)
    int depthFocalLength_VGA;

    // The value for shadow (occluded pixels)
    int shadowValue;
    // The value for pixels without a valid disparity measurement
    int noSampleValue;

    std::vector<OutputMap> outputMaps;
};

IplImage* CvCapture_OpenNI2::OutputMap::getIplImagePtr()
{
    if( mat.empty() )
        return 0;

    iplHeader = cvIplImage(mat);
    return &iplHeader;
}

bool CvCapture_OpenNI2::isOpened() const
{
    return isContextOpened;
}

openni::VideoMode CvCapture_OpenNI2::defaultStreamOutputMode(int stream)
{
    openni::VideoMode mode;
    mode.setResolution(640, 480);
    mode.setFps(30);
    switch (stream)
    {
    case CV_DEPTH_STREAM:
        mode.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);
        break;
    case CV_COLOR_STREAM:
        mode.setPixelFormat(openni::PIXEL_FORMAT_RGB888);
        break;
    case CV_IR_STREAM:
        mode.setPixelFormat(openni::PIXEL_FORMAT_GRAY16);
        break;
    }
    return mode;
}


CvCapture_OpenNI2::CvCapture_OpenNI2(int index) :
    CvCapture_OpenNI2(index, nullptr)
{ }

CvCapture_OpenNI2::CvCapture_OpenNI2(const char * filename) :
    CvCapture_OpenNI2(-1, filename)
{ }

CvCapture_OpenNI2::CvCapture_OpenNI2(int index, const char * filename) :
    device(),
    isContextOpened(false),
    streams(CV_MAX_NUM_STREAMS),
    streamFrames(CV_MAX_NUM_STREAMS),
    streamImages(CV_MAX_NUM_STREAMS),
    maxBufferSize(DEFAULT_MAX_BUFFER_SIZE),
    maxTimeDuration(DEFAULT_MAX_TIME_DURATION),
    isCircleBuffer(DEFAULT_IS_CIRCLE_BUFFER),
    baseline(0),
    depthFocalLength_VGA(0),
    shadowValue(0),
    noSampleValue(0),
    outputMaps(outputMapsTypesCount)
{
    // Initialize and configure the context.
    OpenNI2Initializer::init();

    const char* deviceURI = openni::ANY_DEVICE;
    bool needColor = true;
    bool needIR = true;
    if (index >= 0)
    {
        int deviceType = DEVICE_DEFAULT;
        if (index >= 10)
        {
            deviceType = index / 10;
            index %= 10;
        }
        // Asus XTION and Occipital Structure Sensor do not have an image generator
        needColor = (deviceType != DEVICE_ASUS_XTION);

        // find appropriate device URI
        openni::Array<openni::DeviceInfo> ldevs;
        if (index > 0)
        {
            openni::OpenNI::enumerateDevices(&ldevs);
            if (index < ldevs.getSize())
                deviceURI = ldevs[index].getUri();
            else
            {
                CV_Error(CV_StsError, "OpenCVKinect2: Device index exceeds the number of available OpenNI devices");
            }
        }
    }
    else
    {
        deviceURI = filename;
    }

    openni::Status status;
    status = device.open(deviceURI);
    if (status != openni::STATUS_OK)
    {
        CV_Error(CV_StsError, std::string("OpenCVKinect2: Failed to open device: ") + openni::OpenNI::getExtendedError());
    }

    toggleStream(CV_DEPTH_STREAM, true);
    if (needColor)
        toggleStream(CV_COLOR_STREAM, true);
    if (needIR)
        toggleStream(CV_IR_STREAM, true);

    setProperty(CV_CAP_PROP_OPENNI_REGISTRATION, 1.0);

    // default for Kinect2 camera
    setProperty(CV_CAP_PROP_OPENNI2_MIRROR, 0.0);

    isContextOpened = true;
}

CvCapture_OpenNI2::~CvCapture_OpenNI2()
{
    for (size_t i = 0; i < streams.size(); ++i)
    {
        streamFrames[i].release();
        streams[i].stop();
        streams[i].destroy();
    }
    device.close();
}

void CvCapture_OpenNI2::toggleStream(int stream, bool toggle)
{
    openni::Status status;

    // for logging
    static const std::string stream_names[CV_MAX_NUM_STREAMS] = {
        "depth",
        "color",
        "IR"
    };

    static const openni::SensorType stream_sensor_types[CV_MAX_NUM_STREAMS] = {
        openni::SENSOR_DEPTH,
        openni::SENSOR_COLOR,
        openni::SENSOR_IR
    };

    if (toggle) // want to open stream
    {
        // already opened
        if (streams[stream].isValid())
            return;

        // open stream
        status = streams[stream].create(device, stream_sensor_types[stream]);
        if (status == openni::STATUS_OK)
        {
            // try to set up default stream mode (if available)
            const openni::Array<openni::VideoMode>& vm = streams[stream].getSensorInfo().getSupportedVideoModes();
            openni::VideoMode dm = defaultStreamOutputMode(stream);
            for (int i = 0; i < vm.getSize(); i++)
            {
                if (vm[i].getPixelFormat() == dm.getPixelFormat() &&
                    vm[i].getResolutionX() == dm.getResolutionX() &&
                    vm[i].getResolutionY() == dm.getResolutionY() &&
                    vm[i].getFps() == dm.getFps())
                {
                    status = streams[stream].setVideoMode(defaultStreamOutputMode(stream));
                    if (status != openni::STATUS_OK)
                    {
                        streams[stream].destroy();
                        CV_Error(CV_StsError, std::string("OpenCVKinect2 : Couldn't set ") +
                                 stream_names[stream] + std::string(" stream output mode: ") +
                                 std::string(openni::OpenNI::getExtendedError()));
                    }
                }
            }

            // start stream
            status = streams[stream].start();
            if (status != openni::STATUS_OK)
            {
                streams[stream].destroy();
                CV_Error(CV_StsError, std::string("CvCapture_OpenNI2::CvCapture_OpenNI2 : Couldn't start ") +
                         stream_names[stream] + std::string(" stream: ") +
                         std::string(openni::OpenNI::getExtendedError()));
            }
        }
        else
        {
            CV_Error(CV_StsError, std::string("CvCapture_OpenNI2::CvCapture_OpenNI2 : Couldn't find ") +
                     stream_names[stream] + " stream: " +
                     std::string(openni::OpenNI::getExtendedError()));
        }
    }
    else if (streams[stream].isValid()) // want to close stream
    {
        //FIX for libfreenect2
        //which stops the whole device when stopping only one stream

        //streams[stream].stop();
        //streams[stream].destroy();
    }
}


void CvCapture_OpenNI2::readCamerasParams()
{
    double pixelSize = 0;
    if (streams[CV_DEPTH_STREAM].getProperty<double>(XN_STREAM_PROPERTY_ZERO_PLANE_PIXEL_SIZE, &pixelSize) != openni::STATUS_OK)
    {
        CV_Error(CV_StsError, "CvCapture_OpenNI2::readCamerasParams : Could not read pixel size!" +
                              std::string(openni::OpenNI::getExtendedError()));
    }

    // pixel size @ VGA = pixel size @ SXGA x 2
    pixelSize *= 2.0; // in mm

    // focal length of IR camera in pixels for VGA resolution
    unsigned long long zeroPlaneDistance; // in mm
    if (streams[CV_DEPTH_STREAM].getProperty(XN_STREAM_PROPERTY_ZERO_PLANE_DISTANCE, &zeroPlaneDistance) != openni::STATUS_OK)
    {
        CV_Error(CV_StsError, "CvCapture_OpenNI2::readCamerasParams : Could not read virtual plane distance!" +
                              std::string(openni::OpenNI::getExtendedError()));
    }

    if (streams[CV_DEPTH_STREAM].getProperty<double>(XN_STREAM_PROPERTY_EMITTER_DCMOS_DISTANCE, &baseline) != openni::STATUS_OK)
    {
        CV_Error(CV_StsError, "CvCapture_OpenNI2::readCamerasParams : Could not read base line!" +
                              std::string(openni::OpenNI::getExtendedError()));
    }

    // baseline from cm -> mm
    baseline *= 10;

    // focal length from mm -> pixels (valid for 640x480)
    depthFocalLength_VGA = (int)((double)zeroPlaneDistance / (double)pixelSize);
}

double CvCapture_OpenNI2::getProperty( int propIdx ) const
{
    double propValue = 0;

    if( isOpened() )
    {
        int purePropIdx = propIdx & ~CV_CAP_OPENNI_GENERATORS_MASK;

        if( (propIdx & CV_CAP_OPENNI_GENERATORS_MASK) == CV_CAP_OPENNI_IMAGE_GENERATOR )
        {
            propValue = getImageGeneratorProperty( purePropIdx );
        }
        else if( (propIdx & CV_CAP_OPENNI_GENERATORS_MASK) == CV_CAP_OPENNI_DEPTH_GENERATOR )
        {
            propValue = getDepthGeneratorProperty( purePropIdx );
        }
        else if ((propIdx & CV_CAP_OPENNI_GENERATORS_MASK) == CV_CAP_OPENNI_IR_GENERATOR)
        {
            propValue = getIrGeneratorProperty(purePropIdx);
        }
        else
        {
            propValue = getCommonProperty( purePropIdx );
        }
    }

    return propValue;
}

bool CvCapture_OpenNI2::setProperty( int propIdx, double propValue )
{
    bool isSet = false;
    if( isOpened() )
    {
        int purePropIdx = propIdx & ~CV_CAP_OPENNI_GENERATORS_MASK;

        if( (propIdx & CV_CAP_OPENNI_GENERATORS_MASK) == CV_CAP_OPENNI_IMAGE_GENERATOR )
        {
            isSet = setImageGeneratorProperty( purePropIdx, propValue );
        }
        else if( (propIdx & CV_CAP_OPENNI_GENERATORS_MASK) == CV_CAP_OPENNI_DEPTH_GENERATOR )
        {
            isSet = setDepthGeneratorProperty( purePropIdx, propValue );
        }
        else if ((propIdx & CV_CAP_OPENNI_GENERATORS_MASK) == CV_CAP_OPENNI_IR_GENERATOR)
        {
            isSet = setIrGeneratorProperty(purePropIdx, propValue);
        }
        else
        {
            isSet = setCommonProperty( purePropIdx, propValue );
        }
    }

    return isSet;
}

double CvCapture_OpenNI2::getCommonProperty( int propIdx ) const
{
    double propValue = 0;

    switch( propIdx )
    {
    case CV_CAP_PROP_FRAME_WIDTH :
    case CV_CAP_PROP_FRAME_HEIGHT :
    case CV_CAP_PROP_FPS :
    case CV_CAP_PROP_OPENNI_FRAME_MAX_DEPTH :
    case CV_CAP_PROP_OPENNI_BASELINE :
    case CV_CAP_PROP_OPENNI_FOCAL_LENGTH :
    case CV_CAP_PROP_OPENNI_REGISTRATION :
        propValue = getDepthGeneratorProperty( propIdx );
        break;
    case CV_CAP_PROP_OPENNI2_SYNC :
        propValue = const_cast<CvCapture_OpenNI2 *>(this)->device.getDepthColorSyncEnabled();
        break;
    case CV_CAP_PROP_OPENNI2_MIRROR:
    {
        bool isMirroring = false;
        for (int i = 0; i < CV_MAX_NUM_STREAMS; ++i)
            isMirroring |= streams[i].getMirroringEnabled();
        propValue = isMirroring ? 1.0 : 0.0;
        break;
    }
    default :
        CV_Error( CV_StsBadArg, cv::format("Such parameter (propIdx=%d) isn't supported for getting.", propIdx) );
    }

    return propValue;
}

bool CvCapture_OpenNI2::setCommonProperty( int propIdx, double propValue )
{
    bool isSet = false;

    switch( propIdx )
    {
    case CV_CAP_PROP_OPENNI2_MIRROR:
    {
        bool mirror = propValue > 0.0 ? true : false;
        for (int i = 0; i < CV_MAX_NUM_STREAMS; ++i)
        {
            if (streams[i].isValid())
                isSet |= streams[i].setMirroringEnabled(mirror) == openni::STATUS_OK;
        }
    }
        break;
    // There is a set of properties that correspond to depth generator by default
    // (is they are pass without particular generator flag).
    case CV_CAP_PROP_OPENNI_REGISTRATION:
        isSet = setDepthGeneratorProperty(propIdx, propValue);
        break;
    case CV_CAP_PROP_OPENNI2_SYNC:
        isSet = device.setDepthColorSyncEnabled(propValue > 0.0) == openni::STATUS_OK;
        break;

    case CV_CAP_PROP_FRAME_WIDTH:
    case CV_CAP_PROP_FRAME_HEIGHT:
    case CV_CAP_PROP_AUTOFOCUS:
        isSet = false;
        break;

    default:
        CV_Error(CV_StsBadArg, cv::format("Such parameter (propIdx=%d) isn't supported for setting.", propIdx));
    }

    return isSet;
}

double CvCapture_OpenNI2::getDepthGeneratorProperty( int propIdx ) const
{
    double propValue = 0;
    if( !streams[CV_DEPTH_STREAM].isValid() )
        return propValue;

    openni::VideoMode mode;

    switch( propIdx )
    {
    case CV_CAP_PROP_OPENNI_GENERATOR_PRESENT:
        propValue = streams[CV_DEPTH_STREAM].isValid();
        break;
    case CV_CAP_PROP_FRAME_WIDTH :
        propValue = streams[CV_DEPTH_STREAM].getVideoMode().getResolutionX();
        break;
    case CV_CAP_PROP_FRAME_HEIGHT :
            propValue = streams[CV_DEPTH_STREAM].getVideoMode().getResolutionY();
        break;
    case CV_CAP_PROP_FPS :
        mode = streams[CV_DEPTH_STREAM].getVideoMode();
        propValue = mode.getFps();
        break;
    case CV_CAP_PROP_OPENNI_FRAME_MAX_DEPTH :
        propValue = streams[CV_DEPTH_STREAM].getMaxPixelValue();
        break;
    case CV_CAP_PROP_OPENNI_BASELINE :
        if(baseline <= 0)
            const_cast<CvCapture_OpenNI2*>(this)->readCamerasParams();
        propValue = baseline;
        break;
    case CV_CAP_PROP_OPENNI_FOCAL_LENGTH :
        if(depthFocalLength_VGA <= 0)
            const_cast<CvCapture_OpenNI2*>(this)->readCamerasParams();
        propValue = (double)depthFocalLength_VGA;
        break;
    case CV_CAP_PROP_OPENNI_REGISTRATION :
        propValue = device.getImageRegistrationMode();
        break;
    case CV_CAP_PROP_POS_MSEC :
        propValue = (double)streamFrames[CV_DEPTH_STREAM].getTimestamp();
        break;
    case CV_CAP_PROP_POS_FRAMES :
        propValue = streamFrames[CV_DEPTH_STREAM].getFrameIndex();
        break;
    default :
        CV_Error( CV_StsBadArg, cv::format("Depth generator does not support such parameter (propIdx=%d) for getting.", propIdx) );
    }

    return propValue;
}

bool CvCapture_OpenNI2::setDepthGeneratorProperty( int propIdx, double propValue )
{
    bool isSet = false;

    switch( propIdx )
    {
    case CV_CAP_PROP_OPENNI_GENERATOR_PRESENT:
        if (isContextOpened)
        {
            toggleStream(CV_DEPTH_STREAM, propValue > 0.0);
            isSet = true;
        }
        break;
    case CV_CAP_PROP_OPENNI_REGISTRATION:
        {
            CV_Assert(streams[CV_DEPTH_STREAM].isValid());
            if( propValue != 0.0 ) // "on"
            {
                // if there isn't image generator (i.e. ASUS XtionPro doesn't have it)
                // then the property isn't available
                if ( streams[CV_COLOR_STREAM].isValid() )
                {
                    openni::ImageRegistrationMode mode = propValue != 0.0 ? openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR : openni::IMAGE_REGISTRATION_OFF;
                    if( device.getImageRegistrationMode() != mode )
                    {
                        if (device.isImageRegistrationModeSupported(mode))
                        {
                            openni::Status status = device.setImageRegistrationMode(mode);
                            if( status != openni::STATUS_OK )
                                CV_Error(CV_StsError, std::string("CvCapture_OpenNI2::setDepthGeneratorProperty: ") +
                                         std::string(openni::OpenNI::getExtendedError()));
                            else
                                isSet = true;
                        }
                        else
                            CV_Error(CV_StsError, "CvCapture_OpenNI2::setDepthGeneratorProperty: Unsupported viewpoint.");
                    }
                    else
                        isSet = true;
                }
            }
            else // "off"
            {
                openni::Status status = device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_OFF);
                if( status != openni::STATUS_OK )
                    CV_Error(CV_StsError, std::string("CvCapture_OpenNI2::setDepthGeneratorProperty: ") +
                             std::string(openni::OpenNI::getExtendedError()));
                else
                    isSet = true;
            }
        }
        break;
    default:
        CV_Error( CV_StsBadArg, cv::format("Depth generator does not support such parameter (propIdx=%d) for setting.", propIdx) );
    }

    return isSet;
}

double CvCapture_OpenNI2::getImageGeneratorProperty( int propIdx ) const
{
    double propValue = 0.;
    if( !streams[CV_COLOR_STREAM].isValid() )
        return propValue;

    openni::VideoMode mode;
    switch( propIdx )
    {
    case CV_CAP_PROP_OPENNI_GENERATOR_PRESENT:
        propValue = streams[CV_COLOR_STREAM].isValid();
        break;
    case CV_CAP_PROP_FRAME_WIDTH :
            propValue = streams[CV_COLOR_STREAM].getVideoMode().getResolutionX();
        break;
    case CV_CAP_PROP_FRAME_HEIGHT :
            propValue = streams[CV_COLOR_STREAM].getVideoMode().getResolutionY();
        break;
    case CV_CAP_PROP_FPS :
            propValue = streams[CV_COLOR_STREAM].getVideoMode().getFps();
        break;
    case CV_CAP_PROP_POS_MSEC :
        propValue = (double)streamFrames[CV_COLOR_STREAM].getTimestamp();
        break;
    case CV_CAP_PROP_POS_FRAMES :
        propValue = (double)streamFrames[CV_COLOR_STREAM].getFrameIndex();
        break;
    default :
        CV_Error( CV_StsBadArg, cv::format("Image generator does not support such parameter (propIdx=%d) for getting.", propIdx) );
    }

    return propValue;
}

bool CvCapture_OpenNI2::setImageGeneratorProperty(int propIdx, double propValue)
{
    bool isSet = false;

        switch( propIdx )
        {
        case CV_CAP_PROP_OPENNI_GENERATOR_PRESENT:
            if (isContextOpened)
            {
                toggleStream(CV_COLOR_STREAM, propValue > 0.0);
                isSet = true;
            }
            break;
        case CV_CAP_PROP_OPENNI_OUTPUT_MODE :
        {
            if (!streams[CV_COLOR_STREAM].isValid())
                return isSet;
            openni::VideoMode mode = streams[CV_COLOR_STREAM].getVideoMode();

            switch( cvRound(propValue) )
            {
            case CV_CAP_OPENNI_VGA_30HZ :
                mode.setResolution(640,480);
                mode.setFps(30);
                break;
            case CV_CAP_OPENNI_SXGA_15HZ :
                mode.setResolution(1280, 960);
                mode.setFps(15);
                break;
            case CV_CAP_OPENNI_SXGA_30HZ :
                mode.setResolution(1280, 960);
                mode.setFps(30);
                break;
            case CV_CAP_OPENNI_QVGA_30HZ :
                mode.setResolution(320, 240);
                mode.setFps(30);
                 break;
            case CV_CAP_OPENNI_QVGA_60HZ :
                mode.setResolution(320, 240);
                mode.setFps(60);
                 break;
            default :
                CV_Error( CV_StsBadArg, "Unsupported image generator output mode.");
            }

            openni::Status status = streams[CV_COLOR_STREAM].setVideoMode( mode );
            if( status != openni::STATUS_OK )
                CV_Error(CV_StsError, std::string("CvCapture_OpenNI2::setImageGeneratorProperty: ") +
                         std::string(openni::OpenNI::getExtendedError()));
            else
                isSet = true;
            break;
        }
        default:
            CV_Error( CV_StsBadArg, cv::format("Image generator does not support such parameter (propIdx=%d) for setting.", propIdx) );
        }

    return isSet;
}

double CvCapture_OpenNI2::getIrGeneratorProperty(int propIdx) const
{
    double propValue = 0.;
    if (!streams[CV_IR_STREAM].isValid())
        return propValue;

    openni::VideoMode mode;
    switch (propIdx)
    {
    case CV_CAP_PROP_OPENNI_GENERATOR_PRESENT:
        propValue = streams[CV_IR_STREAM].isValid();
        break;
    case CV_CAP_PROP_FRAME_WIDTH:
        propValue = streams[CV_IR_STREAM].getVideoMode().getResolutionX();
        break;
    case CV_CAP_PROP_FRAME_HEIGHT:
        propValue = streams[CV_IR_STREAM].getVideoMode().getResolutionY();
        break;
    case CV_CAP_PROP_FPS:
        propValue = streams[CV_IR_STREAM].getVideoMode().getFps();
        break;
    case CV_CAP_PROP_POS_MSEC:
        propValue = (double)streamFrames[CV_IR_STREAM].getTimestamp();
        break;
    case CV_CAP_PROP_POS_FRAMES:
        propValue = (double)streamFrames[CV_IR_STREAM].getFrameIndex();
        break;
    default:
        CV_Error(CV_StsBadArg, cv::format("Image generator does not support such parameter (propIdx=%d) for getting.", propIdx));
    }

    return propValue;
}

bool CvCapture_OpenNI2::setIrGeneratorProperty(int propIdx, double propValue)
{
    bool isSet = false;

    switch (propIdx)
    {
    case CV_CAP_PROP_OPENNI_GENERATOR_PRESENT:
        if (isContextOpened)
        {
            toggleStream(CV_IR_STREAM, propValue > 0.0);
            isSet = true;
        }
        break;
    case CV_CAP_PROP_OPENNI_OUTPUT_MODE:
    {
        if (!streams[CV_IR_STREAM].isValid())
            return isSet;
        openni::VideoMode mode = streams[CV_IR_STREAM].getVideoMode();

        switch (cvRound(propValue))
        {
        case CV_CAP_OPENNI_VGA_30HZ:
            mode.setResolution(640, 480);
            mode.setFps(30);
            break;
        case CV_CAP_OPENNI_SXGA_15HZ:
            mode.setResolution(1280, 960);
            mode.setFps(15);
            break;
        case CV_CAP_OPENNI_SXGA_30HZ:
            mode.setResolution(1280, 960);
            mode.setFps(30);
            break;
        case CV_CAP_OPENNI_QVGA_30HZ:
            mode.setResolution(320, 240);
            mode.setFps(30);
            break;
        case CV_CAP_OPENNI_QVGA_60HZ:
            mode.setResolution(320, 240);
            mode.setFps(60);
            break;
        default:
            CV_Error(CV_StsBadArg, "Unsupported image generator output mode.");
        }

        openni::Status status = streams[CV_IR_STREAM].setVideoMode(mode);
        if (status != openni::STATUS_OK)
            CV_Error(CV_StsError, std::string("CvCapture_OpenNI2::setImageGeneratorProperty: ") +
                     std::string(openni::OpenNI::getExtendedError()));
        else
            isSet = true;
        break;
    }
    default:
        CV_Error(CV_StsBadArg, cv::format("Image generator does not support such parameter (propIdx=%d) for setting.", propIdx));
    }

    return isSet;
}

bool CvCapture_OpenNI2::grabFrame()
{
    if( !isOpened() )
        return false;

    bool isGrabbed = false;

    int numActiveStreams = 0;
    openni::VideoStream* streamPtrs[CV_MAX_NUM_STREAMS];
    for (int i = 0; i < CV_MAX_NUM_STREAMS; ++i) {
        streamPtrs[numActiveStreams++] = &streams[i];
    }

    int currentStream;
    openni::Status status = openni::OpenNI::waitForAnyStream(streamPtrs, numActiveStreams, &currentStream, CV_STREAM_TIMEOUT);
    if( status != openni::STATUS_OK )
        return false;

    for (int i = 0; i < CV_MAX_NUM_STREAMS; ++i)
    {
        if (streams[i].isValid())
            streams[i].readFrame(&streamFrames[i]);
    }
    isGrabbed = true;

    return isGrabbed;
}

inline void getDepthMapFromMetaData(const openni::VideoFrameRef& depthMetaData, cv::Mat& depthMap, int noSampleValue, int shadowValue)
{
    depthMap.create(depthMetaData.getHeight(), depthMetaData.getWidth(), CV_16UC1);
    depthMap.data = (uchar*)depthMetaData.getData();

    cv::Mat badMask = (depthMap == (double)noSampleValue) | (depthMap == (double)shadowValue) | (depthMap == 0);

    // mask the pixels with invalid depth
    depthMap.setTo( cv::Scalar::all( CvCapture_OpenNI2::INVALID_PIXEL_VAL ), badMask );
}

IplImage* CvCapture_OpenNI2::retrieveDepthMap()
{
    if( !streamFrames[CV_DEPTH_STREAM].isValid() )
        return 0;

    getDepthMapFromMetaData(streamFrames[CV_DEPTH_STREAM], outputMaps[CV_CAP_OPENNI_DEPTH_MAP].mat, noSampleValue, shadowValue );

    return outputMaps[CV_CAP_OPENNI_DEPTH_MAP].getIplImagePtr();
}

IplImage* CvCapture_OpenNI2::retrievePointCloudMap()
{
    if( !streamFrames[CV_DEPTH_STREAM].isValid() )
        return 0;

    cv::Mat depthImg;
    getDepthMapFromMetaData(streamFrames[CV_DEPTH_STREAM], depthImg, noSampleValue, shadowValue);

    const int badPoint = INVALID_PIXEL_VAL;
    const float badCoord = INVALID_COORDINATE_VAL;
    int cols = streamFrames[CV_DEPTH_STREAM].getWidth(), rows = streamFrames[CV_DEPTH_STREAM].getHeight();
    cv::Mat pointCloud_XYZ( rows, cols, CV_32FC3, cv::Scalar::all(badPoint) );

    float worldX, worldY, worldZ;
    for( int y = 0; y < rows; y++ )
    {
        for (int x = 0; x < cols; x++)
        {
            openni::CoordinateConverter::convertDepthToWorld(streams[CV_DEPTH_STREAM], x, y, depthImg.at<unsigned short>(y, x), &worldX, &worldY, &worldZ);

            if (depthImg.at<unsigned short>(y, x) == badPoint) // not valid
                pointCloud_XYZ.at<cv::Point3f>(y, x) = cv::Point3f(badCoord, badCoord, badCoord);
            else
            {
                pointCloud_XYZ.at<cv::Point3f>(y, x) = cv::Point3f(worldX*0.001f, worldY*0.001f, worldZ*0.001f); // from mm to meters
            }
        }
    }

    outputMaps[CV_CAP_OPENNI_POINT_CLOUD_MAP].mat = pointCloud_XYZ;

    return outputMaps[CV_CAP_OPENNI_POINT_CLOUD_MAP].getIplImagePtr();
}

static void computeDisparity_32F( const openni::VideoFrameRef& depthMetaData, cv::Mat& disp, double baseline, int F, int noSampleValue, int shadowValue)
{
    cv::Mat depth;
    getDepthMapFromMetaData( depthMetaData, depth, noSampleValue, shadowValue );
    CV_Assert( depth.type() == CV_16UC1 );

    // disparity = baseline * F / z;

    float mult = (float)(baseline /*mm*/ * F /*pixels*/);

    disp.create( depth.size(), CV_32FC1);
    disp = cv::Scalar::all( CvCapture_OpenNI2::INVALID_PIXEL_VAL );
    for( int y = 0; y < disp.rows; y++ )
    {
        for( int x = 0; x < disp.cols; x++ )
        {
            unsigned short curDepth = depth.at<unsigned short>(y,x);
            if( curDepth != CvCapture_OpenNI2::INVALID_PIXEL_VAL )
                disp.at<float>(y,x) = mult / curDepth;
        }
    }
}

IplImage* CvCapture_OpenNI2::retrieveDisparityMap()
{
    if (!streamFrames[CV_DEPTH_STREAM].isValid())
        return 0;

    readCamerasParams();

    cv::Mat disp32;
    computeDisparity_32F(streamFrames[CV_DEPTH_STREAM], disp32, baseline, depthFocalLength_VGA, noSampleValue, shadowValue);

    disp32.convertTo(outputMaps[CV_CAP_OPENNI_DISPARITY_MAP].mat, CV_8UC1);

    return outputMaps[CV_CAP_OPENNI_DISPARITY_MAP].getIplImagePtr();
}

IplImage* CvCapture_OpenNI2::retrieveDisparityMap_32F()
{
    if (!streamFrames[CV_DEPTH_STREAM].isValid())
        return 0;

    readCamerasParams();

    computeDisparity_32F(streamFrames[CV_DEPTH_STREAM], outputMaps[CV_CAP_OPENNI_DISPARITY_MAP_32F].mat, baseline, depthFocalLength_VGA, noSampleValue, shadowValue);

    return outputMaps[CV_CAP_OPENNI_DISPARITY_MAP_32F].getIplImagePtr();
}

IplImage* CvCapture_OpenNI2::retrieveValidDepthMask()
{
    if (!streamFrames[CV_DEPTH_STREAM].isValid())
        return 0;

    cv::Mat d;
    getDepthMapFromMetaData(streamFrames[CV_DEPTH_STREAM], d, noSampleValue, shadowValue);

    outputMaps[CV_CAP_OPENNI_VALID_DEPTH_MASK].mat = d != CvCapture_OpenNI2::INVALID_PIXEL_VAL;

    return outputMaps[CV_CAP_OPENNI_VALID_DEPTH_MASK].getIplImagePtr();
}

inline void getBGRImageFromMetaData( const openni::VideoFrameRef& imageMetaData, cv::Mat& bgrImage )
{
   cv::Mat bufferImage;
   if( imageMetaData.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_RGB888 )
        CV_Error( CV_StsUnsupportedFormat, "Unsupported format of grabbed image." );

   bgrImage.create(imageMetaData.getHeight(), imageMetaData.getWidth(), CV_8UC3);
   bufferImage.create(imageMetaData.getHeight(), imageMetaData.getWidth(), CV_8UC3);
   bufferImage.data = (uchar*)imageMetaData.getData();

   cv::cvtColor(bufferImage, bgrImage, cv::COLOR_RGB2BGR);
}

inline void getGrayImageFromMetaData(const openni::VideoFrameRef& imageMetaData, cv::Mat& grayImage)
{
    if (imageMetaData.getVideoMode().getPixelFormat() == openni::PIXEL_FORMAT_GRAY8)
    {
        grayImage.create(imageMetaData.getHeight(), imageMetaData.getWidth(), CV_8UC1);
        grayImage.data = (uchar*)imageMetaData.getData();
    }
    else if (imageMetaData.getVideoMode().getPixelFormat() == openni::PIXEL_FORMAT_GRAY16)
    {
        grayImage.create(imageMetaData.getHeight(), imageMetaData.getWidth(), CV_16UC1);
        grayImage.data = (uchar*)imageMetaData.getData();
    }
    else
    {
        CV_Error(CV_StsUnsupportedFormat, "Unsupported format of grabbed image.");
    }
}

IplImage* CvCapture_OpenNI2::retrieveBGRImage()
{
    if( !streamFrames[CV_COLOR_STREAM].isValid() )
        return 0;

    getBGRImageFromMetaData(streamFrames[CV_COLOR_STREAM], outputMaps[CV_CAP_OPENNI_BGR_IMAGE].mat );

    return outputMaps[CV_CAP_OPENNI_BGR_IMAGE].getIplImagePtr();
}

IplImage* CvCapture_OpenNI2::retrieveGrayImage()
{
    if (!streamFrames[CV_COLOR_STREAM].isValid())
        return 0;

    CV_Assert(streamFrames[CV_COLOR_STREAM].getVideoMode().getPixelFormat() == openni::PIXEL_FORMAT_RGB888); // RGB

    cv::Mat rgbImage;
    getBGRImageFromMetaData(streamFrames[CV_COLOR_STREAM], rgbImage);
    cv::cvtColor( rgbImage, outputMaps[CV_CAP_OPENNI_GRAY_IMAGE].mat, CV_BGR2GRAY );

    return outputMaps[CV_CAP_OPENNI_GRAY_IMAGE].getIplImagePtr();
}

IplImage* CvCapture_OpenNI2::retrieveIrImage()
{
    if (!streamFrames[CV_IR_STREAM].isValid())
        return 0;

    getGrayImageFromMetaData(streamFrames[CV_IR_STREAM], outputMaps[CV_CAP_OPENNI_IR_IMAGE].mat);

    return outputMaps[CV_CAP_OPENNI_IR_IMAGE].getIplImagePtr();
}

IplImage* CvCapture_OpenNI2::retrieveFrame( int outputType )
{
    IplImage* image = 0;
    CV_Assert( outputType < outputMapsTypesCount && outputType >= 0);

    if( outputType == CV_CAP_OPENNI_DEPTH_MAP )
    {
        image = retrieveDepthMap();
    }
    else if( outputType == CV_CAP_OPENNI_POINT_CLOUD_MAP )
    {
        image = retrievePointCloudMap();
    }
    else if( outputType == CV_CAP_OPENNI_DISPARITY_MAP )
    {
        image = retrieveDisparityMap();
    }
    else if( outputType == CV_CAP_OPENNI_DISPARITY_MAP_32F )
    {
        image = retrieveDisparityMap_32F();
    }
    else if( outputType == CV_CAP_OPENNI_VALID_DEPTH_MASK )
    {
        image = retrieveValidDepthMask();
    }
    else if( outputType == CV_CAP_OPENNI_BGR_IMAGE )
    {
        image = retrieveBGRImage();
    }
    else if( outputType == CV_CAP_OPENNI_GRAY_IMAGE )
    {
        image = retrieveGrayImage();
    }
    else if( outputType == CV_CAP_OPENNI_IR_IMAGE )
    {
        image = retrieveIrImage();
    }

    return image;
}

cv::Ptr<cv::IVideoCapture> cv::create_OpenNI2_capture_cam( int index )
{
    CvCapture_OpenNI2* capture = new CvCapture_OpenNI2( index );

    if( capture->isOpened() )
        return cv::makePtr<cv::LegacyCapture>(capture);

    delete capture;
    return 0;
}

cv::Ptr<cv::IVideoCapture> cv::create_OpenNI2_capture_file( const std::string &filename )
{
    CvCapture_OpenNI2* capture = new CvCapture_OpenNI2( filename.c_str() );

    if( capture->isOpened() )
        return cv::makePtr<cv::LegacyCapture>(capture);

    delete capture;
    return 0;
}

#endif
