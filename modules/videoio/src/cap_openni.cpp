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
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#ifdef HAVE_OPENNI

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

#include "XnCppWrapper.h"

const cv::String XMLConfig =
"<OpenNI>"
        "<Licenses>"
        "<License vendor=\"PrimeSense\" key=\"0KOIk2JeIBYClPWVnMoRKn5cdY4=\"/>"
        "</Licenses>"
        "<Log writeToConsole=\"false\" writeToFile=\"false\">"
                "<LogLevel value=\"3\"/>"
                "<Masks>"
                        "<Mask name=\"ALL\" on=\"true\"/>"
                "</Masks>"
                "<Dumps>"
                "</Dumps>"
        "</Log>"
        "<ProductionNodes>"
                "<Node type=\"Image\" name=\"Image1\" stopOnError=\"false\">"
                        "<Configuration>"
                                "<MapOutputMode xRes=\"640\" yRes=\"480\" FPS=\"30\"/>"
                                "<Mirror on=\"false\"/>"
                        "</Configuration>"
                "</Node> "
                "<Node type=\"Depth\" name=\"Depth1\">"
                        "<Configuration>"
                                "<MapOutputMode xRes=\"640\" yRes=\"480\" FPS=\"30\"/>"
                                "<Mirror on=\"false\"/>"
                        "</Configuration>"
                "</Node>"
        "</ProductionNodes>"
"</OpenNI>\n";

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ApproximateSyncGrabber
{
public:
    ApproximateSyncGrabber( xn::Context &_context,
                            xn::DepthGenerator &_depthGenerator,
                            xn::ImageGenerator &_imageGenerator,
                            int _maxBufferSize, bool _isCircleBuffer, int _maxTimeDuration ) :
        context(_context), depthGenerator(_depthGenerator), imageGenerator(_imageGenerator),
        maxBufferSize(_maxBufferSize), isCircleBuffer(_isCircleBuffer), maxTimeDuration(_maxTimeDuration)
    {

        CV_Assert( depthGenerator.IsValid() );
        CV_Assert( imageGenerator.IsValid() );
    }

    void setMaxBufferSize( int _maxBufferSize )
    {
        maxBufferSize = _maxBufferSize;
    }
    inline int getMaxBufferSize() const { return maxBufferSize; }

    void setIsCircleBuffer( bool _isCircleBuffer ) { isCircleBuffer = _isCircleBuffer; }
    bool getIsCircleBuffer() const { return isCircleBuffer; }

    void setMaxTimeDuration( int _maxTimeDuration ) {  maxTimeDuration = _maxTimeDuration; }
    int getMaxTimeDuration() const { return maxTimeDuration; }

    bool grab( xn::DepthMetaData& depthMetaData,
               xn::ImageMetaData& imageMetaData )
    {
        CV_Assert( task );


        while( task->grab(depthMetaData, imageMetaData) == false )
        {
            task->spin();
        }
        return true;

    }

    void start()
    {
        CV_Assert( depthGenerator.IsValid() );
        CV_Assert( imageGenerator.IsValid() );
        task.reset( new ApproximateSynchronizer( *this ) );
    }

    void finish()
    {
        task.release();
    }

    bool isRun() const { return task != 0; }

    xn::Context &context;
    xn::DepthGenerator &depthGenerator;
    xn::ImageGenerator &imageGenerator;

private:
    ApproximateSyncGrabber(const ApproximateSyncGrabber&);
    ApproximateSyncGrabber& operator=(const ApproximateSyncGrabber&);

    int maxBufferSize;
    bool isCircleBuffer;
    int maxTimeDuration;

    class ApproximateSynchronizerBase
    {
    public:
        ApproximateSynchronizerBase( ApproximateSyncGrabber& _approxSyncGrabber ) :
            approxSyncGrabber(_approxSyncGrabber), isDepthFilled(false), isImageFilled(false)
        {}

        virtual ~ApproximateSynchronizerBase() {}

        virtual bool isSpinContinue() const = 0;
        virtual void pushDepthMetaData( xn::DepthMetaData& depthMetaData ) = 0;
        virtual void pushImageMetaData( xn::ImageMetaData& imageMetaData ) = 0;
        virtual bool popDepthMetaData( xn::DepthMetaData& depthMetaData ) = 0;
        virtual bool popImageMetaData( xn::ImageMetaData& imageMetaData ) = 0;

        void spin()
        {
            while(isSpinContinue() == true)
            {
                XnStatus status = approxSyncGrabber.context.WaitAnyUpdateAll();
                if( status != XN_STATUS_OK )
                    continue;

                //xn::DepthMetaData depth;
                //xn::ImageMetaData image;
                approxSyncGrabber.depthGenerator.GetMetaData(depth);
                approxSyncGrabber.imageGenerator.GetMetaData(image);

                if( depth.Data() && depth.IsDataNew() )
                    pushDepthMetaData( depth );

                if( image.Data() && image.IsDataNew() )
                    pushImageMetaData( image );
            }
        }

        virtual bool grab( xn::DepthMetaData& depthMetaData,
                           xn::ImageMetaData& imageMetaData )
        {
            for(;;)
            {
                if( !isDepthFilled )
                    isDepthFilled = popDepthMetaData(depth);
                if( !isImageFilled )
                    isImageFilled = popImageMetaData(image);

                if( !isDepthFilled || !isImageFilled )
                    break;

                double timeDiff = 1e-3 * std::abs(static_cast<double>(depth.Timestamp()) - static_cast<double>(image.Timestamp()));

                if( timeDiff <= approxSyncGrabber.maxTimeDuration )
                {
                    depthMetaData.InitFrom(depth);
                    imageMetaData.InitFrom(image);
                    isDepthFilled = isImageFilled = false;
                    return true;
                }
                else
                {
                    if( depth.Timestamp() < image.Timestamp() )
                        isDepthFilled = false;
                    else
                        isImageFilled = false;
                }
            }

            return false;
        }

    protected:
        ApproximateSyncGrabber& approxSyncGrabber;
        xn::DepthMetaData depth;
        xn::ImageMetaData image;
        bool isDepthFilled;
        bool isImageFilled;
    };

    // If there isn't TBB the synchronization will be executed in the main thread.
    class ApproximateSynchronizer: public ApproximateSynchronizerBase
    {
    public:
        ApproximateSynchronizer( ApproximateSyncGrabber& _approxSyncGrabber ) :
            ApproximateSynchronizerBase(_approxSyncGrabber)
        {}

        virtual bool isSpinContinue() const CV_OVERRIDE
        {
            int maxBufferSize = approxSyncGrabber.getMaxBufferSize();
            return (maxBufferSize <= 0) || (static_cast<int>(depthQueue.size()) < maxBufferSize &&
                                           static_cast<int>(imageQueue.size()) < maxBufferSize); // "<" to may push
        }

        virtual inline void pushDepthMetaData( xn::DepthMetaData& depthMetaData ) CV_OVERRIDE
        {
            cv::Ptr<xn::DepthMetaData> depthPtr = cv::makePtr<xn::DepthMetaData>();
            depthPtr->CopyFrom(depthMetaData);
            depthQueue.push(depthPtr);
        }
        virtual inline void pushImageMetaData( xn::ImageMetaData& imageMetaData ) CV_OVERRIDE
        {
            cv::Ptr<xn::ImageMetaData> imagePtr = cv::makePtr<xn::ImageMetaData>();
            imagePtr->CopyFrom(imageMetaData);
            imageQueue.push(imagePtr);
        }
        virtual inline bool popDepthMetaData( xn::DepthMetaData& depthMetaData ) CV_OVERRIDE
        {
            if( depthQueue.empty() )
                return false;

            depthMetaData.CopyFrom(*depthQueue.front());
            depthQueue.pop();
            return true;
        }
        virtual inline bool popImageMetaData( xn::ImageMetaData& imageMetaData ) CV_OVERRIDE
        {
            if( imageQueue.empty() )
                return false;

            imageMetaData.CopyFrom(*imageQueue.front());
            imageQueue.pop();
            return true;
        }

    private:
        std::queue<cv::Ptr<xn::DepthMetaData> > depthQueue;
        std::queue<cv::Ptr<xn::ImageMetaData> > imageQueue;
    };

    cv::Ptr<ApproximateSynchronizer> task;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class CvCapture_OpenNI : public CvCapture
{
public:
    enum { DEVICE_DEFAULT=0, DEVICE_MS_KINECT=0, DEVICE_ASUS_XTION=1, DEVICE_MAX=1 };

    static const int INVALID_PIXEL_VAL = 0;
    static const int INVALID_COORDINATE_VAL = 0;

    static const int DEFAULT_MAX_BUFFER_SIZE = 2;
    static const int DEFAULT_IS_CIRCLE_BUFFER = 0;
    static const int DEFAULT_MAX_TIME_DURATION = 20;

    CvCapture_OpenNI(int index=0);
    CvCapture_OpenNI(const char * filename);
    virtual ~CvCapture_OpenNI();

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

    static const int outputMapsTypesCount = 7;

    static XnMapOutputMode defaultMapOutputMode();

    IplImage* retrieveDepthMap();
    IplImage* retrievePointCloudMap();
    IplImage* retrieveDisparityMap();
    IplImage* retrieveDisparityMap_32F();
    IplImage* retrieveValidDepthMask();
    IplImage* retrieveBGRImage();
    IplImage* retrieveGrayImage();

    bool readCamerasParams();

    double getDepthGeneratorProperty(int propIdx) const;
    bool setDepthGeneratorProperty(int propIdx, double propVal);
    double getImageGeneratorProperty(int propIdx) const;
    bool setImageGeneratorProperty(int propIdx, double propVal);
    double getCommonProperty(int propIdx) const;
    bool setCommonProperty(int propIdx, double propVal);

    // OpenNI context
    xn::Context context;
    bool isContextOpened;

    xn::ProductionNode productionNode;

    // Data generators with its metadata
    xn::DepthGenerator depthGenerator;
    xn::DepthMetaData  depthMetaData;

    xn::ImageGenerator imageGenerator;
    xn::ImageMetaData  imageMetaData;

    int maxBufferSize, maxTimeDuration; // for approx sync
    bool isCircleBuffer;
    cv::Ptr<ApproximateSyncGrabber> approxSyncGrabber;

    // Cameras settings:
    // TODO find in OpenNI function to convert z->disparity and remove fields "baseline" and depthFocalLength_VGA
    // Distance between IR projector and IR camera (in meters)
    XnDouble baseline;
    // Focal length for the IR camera in VGA resolution (in pixels)
    XnUInt64 depthFocalLength_VGA;

    // The value for shadow (occluded pixels)
    XnUInt64 shadowValue;
    // The value for pixels without a valid disparity measurement
    XnUInt64 noSampleValue;

    std::vector<OutputMap> outputMaps;
};

IplImage* CvCapture_OpenNI::OutputMap::getIplImagePtr()
{
    if( mat.empty() )
        return 0;

    iplHeader = cvIplImage(mat);
    return &iplHeader;
}

bool CvCapture_OpenNI::isOpened() const
{
    return isContextOpened;
}

XnMapOutputMode CvCapture_OpenNI::defaultMapOutputMode()
{
    XnMapOutputMode mode;
    mode.nXRes = XN_VGA_X_RES;
    mode.nYRes = XN_VGA_Y_RES;
    mode.nFPS  = 30;
    return mode;
}

CvCapture_OpenNI::CvCapture_OpenNI( int index )
{
    int deviceType = DEVICE_DEFAULT;
    XnStatus status;

    isContextOpened = false;
    maxBufferSize = DEFAULT_MAX_BUFFER_SIZE;
    isCircleBuffer = DEFAULT_IS_CIRCLE_BUFFER;
    maxTimeDuration = DEFAULT_MAX_TIME_DURATION;

    if( index >= 10 )
    {
        deviceType = index / 10;
        index %= 10;
    }

    if( deviceType > DEVICE_MAX )
        return;

    // Initialize and configure the context.
    status = context.Init();
    if( status != XN_STATUS_OK )
    {
        fprintf(stderr, "CvCapture_OpenNI::CvCapture_OpenNI : Failed to initialize the context: %s\n", xnGetStatusString(status));
        return;
    }

    // Find devices
    xn::NodeInfoList devicesList;
    status = context.EnumerateProductionTrees( XN_NODE_TYPE_DEVICE, NULL, devicesList, 0 );
    if( status != XN_STATUS_OK )
    {
        fprintf(stderr, "CvCapture_OpenNI::CvCapture_OpenNI : Failed to enumerate production trees: %s\n", xnGetStatusString(status));
        return;
    }

    // Chose device according to index
    xn::NodeInfoList::Iterator it = devicesList.Begin();
    for( int i = 0; i < index && it!=devicesList.End(); ++i ) it++;
        if ( it == devicesList.End() )
        {
            fprintf(stderr, "CvCapture_OpenNI::CvCapture_OpenNI : Failed device with index %d\n", index);
            return;
        }

    xn::NodeInfo deviceNode = *it;
    status = context.CreateProductionTree( deviceNode, productionNode );
    if( status != XN_STATUS_OK )
    {
        fprintf(stderr, "CvCapture_OpenNI::CvCapture_OpenNI : Failed to create production tree: %s\n", xnGetStatusString(status));
        return;
    }

    xn::ScriptNode scriptNode;
    status = context.RunXmlScript( XMLConfig.c_str(), scriptNode );
    if( status != XN_STATUS_OK )
    {
        fprintf(stderr, "CvCapture_OpenNI::CvCapture_OpenNI : Failed to run xml script: %s\n", xnGetStatusString(status));
        return;
    }

    // Associate generators with context.
    // enumerate the nodes to find if depth generator is present
    xn::NodeInfoList depthList;
    status = context.EnumerateExistingNodes( depthList, XN_NODE_TYPE_DEPTH );
    if( status != XN_STATUS_OK )
    {
        fprintf(stderr, "CvCapture_OpenNI::CvCapture_OpenNI : Failed to enumerate depth generators: %s\n", xnGetStatusString(status));
        return;
    }
    if( depthList.IsEmpty() )
    {
        fprintf(stderr, "CvCapture_OpenNI::CvCapture_OpenNI : The device doesn't have depth generator. Such devices aren't supported now.\n");
        return;
    }
    status = depthGenerator.Create( context );
    if( status != XN_STATUS_OK )
    {
        fprintf(stderr, "CvCapture_OpenNI::CvCapture_OpenNI : Failed to create depth generator: %s\n", xnGetStatusString(status));
        return;
    }

    // enumerate the nodes to find if image generator is present
    xn::NodeInfoList imageList;
    status = context.EnumerateExistingNodes( imageList, XN_NODE_TYPE_IMAGE );
    if( status != XN_STATUS_OK )
    {
        fprintf(stderr, "CvCapture_OpenNI::CvCapture_OpenNI : Failed to enumerate image generators: %s\n", xnGetStatusString(status));
        return;
    }

    if( !imageList.IsEmpty() )
    {
        status = imageGenerator.Create( context );
        if( status != XN_STATUS_OK )
        {
            fprintf(stderr, "CvCapture_OpenNI::CvCapture_OpenNI : Failed to create image generator: %s\n", xnGetStatusString(status));
            return;
        }
    }

    // Set map output mode.
    if( depthGenerator.IsValid() )
    {
        CV_DbgAssert( depthGenerator.SetMapOutputMode(defaultMapOutputMode()) == XN_STATUS_OK ); // xn::DepthGenerator supports VGA only! (Jan 2011)
    }
    if( imageGenerator.IsValid() )
    {
        CV_DbgAssert( imageGenerator.SetMapOutputMode(defaultMapOutputMode()) == XN_STATUS_OK );
    }

    if( deviceType == DEVICE_ASUS_XTION )
    {
        //ps/asus specific
        imageGenerator.SetIntProperty("InputFormat", 1 /*XN_IO_IMAGE_FORMAT_YUV422*/);
        imageGenerator.SetPixelFormat(XN_PIXEL_FORMAT_RGB24);
        depthGenerator.SetIntProperty("RegistrationType", 1 /*XN_PROCESSING_HARDWARE*/);
    }

    //  Start generating data.
    status = context.StartGeneratingAll();
    if( status != XN_STATUS_OK )
    {
        fprintf(stderr, "CvCapture_OpenNI::CvCapture_OpenNI : Failed to start generating OpenNI data: %s\n", xnGetStatusString(status));
        return;
    }

    if( !readCamerasParams() )
    {
        fprintf(stderr, "CvCapture_OpenNI::CvCapture_OpenNI : Could not read cameras parameters\n");
        return;
    }

    outputMaps.resize( outputMapsTypesCount );

    isContextOpened = true;

    setProperty(CV_CAP_PROP_OPENNI_REGISTRATION, 1.0);
}

CvCapture_OpenNI::CvCapture_OpenNI(const char * filename)
{
    XnStatus status;

    isContextOpened = false;
    maxBufferSize = DEFAULT_MAX_BUFFER_SIZE;
    isCircleBuffer = DEFAULT_IS_CIRCLE_BUFFER;
    maxTimeDuration = DEFAULT_MAX_TIME_DURATION;

    // Initialize and configure the context.
    status = context.Init();
    if( status != XN_STATUS_OK )
    {
        fprintf(stderr, "CvCapture_OpenNI::CvCapture_OpenNI : Failed to initialize the context: %s\n", xnGetStatusString(status));
        return;
    }

    // Open file
    status = context.OpenFileRecording( filename, productionNode );
    if( status != XN_STATUS_OK )
    {
        fprintf(stderr, "CvCapture_OpenNI::CvCapture_OpenNI : Failed to open input file (%s): %s\n", filename, xnGetStatusString(status));
        return;
    }

    context.FindExistingNode( XN_NODE_TYPE_DEPTH, depthGenerator );
    context.FindExistingNode( XN_NODE_TYPE_IMAGE, imageGenerator );

    if( !readCamerasParams() )
    {
        fprintf(stderr, "CvCapture_OpenNI::CvCapture_OpenNI : Could not read cameras parameters\n");
        return;
    }

    outputMaps.resize( outputMapsTypesCount );

    isContextOpened = true;
}

CvCapture_OpenNI::~CvCapture_OpenNI()
{
    context.StopGeneratingAll();
    context.Release();
}

bool CvCapture_OpenNI::readCamerasParams()
{
    XnDouble pixelSize = 0;
    if( depthGenerator.GetRealProperty( "ZPPS", pixelSize ) != XN_STATUS_OK )
    {
        fprintf(stderr, "CvCapture_OpenNI::readCamerasParams : Could not read pixel size!\n");
        return false;
    }

    // pixel size @ VGA = pixel size @ SXGA x 2
    pixelSize *= 2.0; // in mm

    // focal length of IR camera in pixels for VGA resolution
    XnUInt64 zeroPlanDistance; // in mm
    if( depthGenerator.GetIntProperty( "ZPD", zeroPlanDistance ) != XN_STATUS_OK )
    {
        fprintf(stderr, "CvCapture_OpenNI::readCamerasParams : Could not read virtual plane distance!\n");
        return false;
    }

    if( depthGenerator.GetRealProperty( "LDDIS", baseline ) != XN_STATUS_OK )
    {
        fprintf(stderr, "CvCapture_OpenNI::readCamerasParams : Could not read base line!\n");
        return false;
    }

    // baseline from cm -> mm
    baseline *= 10;

    // focal length from mm -> pixels (valid for 640x480)
    depthFocalLength_VGA = (XnUInt64)((double)zeroPlanDistance / (double)pixelSize);

    if( depthGenerator.GetIntProperty( "ShadowValue", shadowValue ) != XN_STATUS_OK )
    {
        fprintf(stderr, "CvCapture_OpenNI::readCamerasParams : Could not read property \"ShadowValue\"!\n");
        return false;
    }

    if( depthGenerator.GetIntProperty("NoSampleValue", noSampleValue ) != XN_STATUS_OK )
    {
        fprintf(stderr, "CvCapture_OpenNI::readCamerasParams : Could not read property \"NoSampleValue\"!\n");
        return false;
    }

    return true;
}

double CvCapture_OpenNI::getProperty( int propIdx ) const
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
        else
        {
            propValue = getCommonProperty( purePropIdx );
        }
    }

    return propValue;
}

bool CvCapture_OpenNI::setProperty( int propIdx, double propValue )
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
        else
        {
            isSet = setCommonProperty( purePropIdx, propValue );
        }
    }

    return isSet;
}

double CvCapture_OpenNI::getCommonProperty( int propIdx ) const
{
    double propValue = 0;

    switch( propIdx )
    {
    // There is a set of properties that correspond to depth generator by default
    // (is they are pass without particular generator flag). Two reasons of this:
    // 1) We can assume that depth generator is the main one for depth sensor.
    // 2) In the initial vertions of OpenNI integration to OpenCV the value of
    //    flag CV_CAP_OPENNI_DEPTH_GENERATOR was 0 (it isn't zero now).
    case CV_CAP_PROP_OPENNI_GENERATOR_PRESENT :
    case CV_CAP_PROP_FRAME_WIDTH :
    case CV_CAP_PROP_FRAME_HEIGHT :
    case CV_CAP_PROP_FPS :
    case CV_CAP_PROP_OPENNI_FRAME_MAX_DEPTH :
    case CV_CAP_PROP_OPENNI_BASELINE :
    case CV_CAP_PROP_OPENNI_FOCAL_LENGTH :
    case CV_CAP_PROP_OPENNI_REGISTRATION :
        propValue = getDepthGeneratorProperty( propIdx );
        break;
    case CV_CAP_PROP_OPENNI_APPROX_FRAME_SYNC :
        propValue = !approxSyncGrabber.empty() && approxSyncGrabber->isRun() ? 1. : 0.;
        break;
    case CV_CAP_PROP_OPENNI_MAX_BUFFER_SIZE :
        propValue = maxBufferSize;
        break;
    case CV_CAP_PROP_OPENNI_CIRCLE_BUFFER :
        propValue = isCircleBuffer ? 1. : 0.;
        break;
    case CV_CAP_PROP_OPENNI_MAX_TIME_DURATION :
        propValue = maxTimeDuration;
        break;
    default :
        CV_Error( CV_StsBadArg, cv::format("Such parameter (propIdx=%d) isn't supported for getting.\n", propIdx) );
    }

    return propValue;
}

bool CvCapture_OpenNI::setCommonProperty( int propIdx, double propValue )
{
    bool isSet = false;

    switch( propIdx )
    {
    // There is a set of properties that correspond to depth generator by default
    // (is they are pass without particular generator flag).
    case CV_CAP_PROP_OPENNI_REGISTRATION:
        isSet = setDepthGeneratorProperty( propIdx, propValue );
        break;
    case CV_CAP_PROP_OPENNI_APPROX_FRAME_SYNC :
        if( propValue && depthGenerator.IsValid() && imageGenerator.IsValid() )
        {
            // start synchronization
            if( approxSyncGrabber.empty() )
            {
                approxSyncGrabber.reset(new ApproximateSyncGrabber( context, depthGenerator, imageGenerator, maxBufferSize, isCircleBuffer, maxTimeDuration ));
            }
            else
            {
                approxSyncGrabber->finish();

                // update params
                approxSyncGrabber->setMaxBufferSize(maxBufferSize);
                approxSyncGrabber->setIsCircleBuffer(isCircleBuffer);
                approxSyncGrabber->setMaxTimeDuration(maxTimeDuration);
            }
            approxSyncGrabber->start();
        }
        else if( !propValue && !approxSyncGrabber.empty() )
        {
            // finish synchronization
            approxSyncGrabber->finish();
        }
        break;
    case CV_CAP_PROP_OPENNI_MAX_BUFFER_SIZE :
        maxBufferSize = cvRound(propValue);
        if( !approxSyncGrabber.empty() )
            approxSyncGrabber->setMaxBufferSize(maxBufferSize);
        break;
    case CV_CAP_PROP_OPENNI_CIRCLE_BUFFER :
        if( !approxSyncGrabber.empty() )
            approxSyncGrabber->setIsCircleBuffer(isCircleBuffer);
        break;
    case CV_CAP_PROP_OPENNI_MAX_TIME_DURATION :
        maxTimeDuration = cvRound(propValue);
        if( !approxSyncGrabber.empty() )
            approxSyncGrabber->setMaxTimeDuration(maxTimeDuration);
        break;
    default:
        CV_Error( CV_StsBadArg, cv::format("Such parameter (propIdx=%d) isn't supported for setting.\n", propIdx) );
    }

    return isSet;
}

double CvCapture_OpenNI::getDepthGeneratorProperty( int propIdx ) const
{
    double propValue = 0;
    if( !depthGenerator.IsValid() )
        return propValue;

    XnMapOutputMode mode;

    switch( propIdx )
    {
    case CV_CAP_PROP_OPENNI_GENERATOR_PRESENT :
        CV_DbgAssert( depthGenerator.IsValid() );
        propValue = 1.;
        break;
    case CV_CAP_PROP_FRAME_WIDTH :
        if( depthGenerator.GetMapOutputMode(mode) == XN_STATUS_OK )
            propValue = mode.nXRes;
        break;
    case CV_CAP_PROP_FRAME_HEIGHT :
        if( depthGenerator.GetMapOutputMode(mode) == XN_STATUS_OK )
            propValue = mode.nYRes;
        break;
    case CV_CAP_PROP_FPS :
        if( depthGenerator.GetMapOutputMode(mode) == XN_STATUS_OK )
            propValue = mode.nFPS;
        break;
    case CV_CAP_PROP_OPENNI_FRAME_MAX_DEPTH :
        propValue = depthGenerator.GetDeviceMaxDepth();
        break;
    case CV_CAP_PROP_OPENNI_BASELINE :
        propValue = baseline;
        break;
    case CV_CAP_PROP_OPENNI_FOCAL_LENGTH :
        propValue = (double)depthFocalLength_VGA;
        break;
    case CV_CAP_PROP_OPENNI_REGISTRATION :
        propValue = depthGenerator.GetAlternativeViewPointCap().IsViewPointAs(const_cast<CvCapture_OpenNI *>(this)->imageGenerator) ? 1.0 : 0.0;
        break;
    case CV_CAP_PROP_POS_MSEC :
        propValue = (double)depthGenerator.GetTimestamp();
        break;
    case CV_CAP_PROP_POS_FRAMES :
        propValue = depthGenerator.GetFrameID();
        break;
    default :
        CV_Error( CV_StsBadArg, cv::format("Depth generator does not support such parameter (propIdx=%d) for getting.\n", propIdx) );
    }

    return propValue;
}

bool CvCapture_OpenNI::setDepthGeneratorProperty( int propIdx, double propValue )
{
    bool isSet = false;

    CV_Assert( depthGenerator.IsValid() );

    switch( propIdx )
    {
    case CV_CAP_PROP_OPENNI_REGISTRATION:
        {
            if( propValue != 0.0 ) // "on"
            {
                // if there isn't image generator (i.e. ASUS XtionPro doesn't have it)
                // then the property isn't available
                if( imageGenerator.IsValid() )
                {
                    if( !depthGenerator.GetAlternativeViewPointCap().IsViewPointAs(imageGenerator) )
                    {
                        if( depthGenerator.GetAlternativeViewPointCap().IsViewPointSupported(imageGenerator) )
                        {
                            XnStatus status = depthGenerator.GetAlternativeViewPointCap().SetViewPoint(imageGenerator);
                            if( status != XN_STATUS_OK )
                                fprintf(stderr, "CvCapture_OpenNI::setDepthGeneratorProperty : %s\n", xnGetStatusString(status));
                            else
                                isSet = true;
                        }
                        else
                            fprintf(stderr, "CvCapture_OpenNI::setDepthGeneratorProperty : Unsupported viewpoint.\n");
                    }
                    else
                        isSet = true;
                }
            }
            else // "off"
            {
                XnStatus status = depthGenerator.GetAlternativeViewPointCap().ResetViewPoint();
                if( status != XN_STATUS_OK )
                    fprintf(stderr, "CvCapture_OpenNI::setDepthGeneratorProperty : %s\n", xnGetStatusString(status));
                else
                    isSet = true;
            }
        }
        break;
    default:
        CV_Error( CV_StsBadArg, cv::format("Depth generator does not support such parameter (propIdx=%d) for setting.\n", propIdx) );
    }

    return isSet;
}

double CvCapture_OpenNI::getImageGeneratorProperty( int propIdx ) const
{
    double propValue = 0.;
    if( !imageGenerator.IsValid() )
        return propValue;

    XnMapOutputMode mode;
    switch( propIdx )
    {
    case CV_CAP_PROP_OPENNI_GENERATOR_PRESENT :
        CV_DbgAssert( imageGenerator.IsValid() );
        propValue = 1.;
        break;
    case CV_CAP_PROP_FRAME_WIDTH :
        if( imageGenerator.GetMapOutputMode(mode) == XN_STATUS_OK )
            propValue = mode.nXRes;
        break;
    case CV_CAP_PROP_FRAME_HEIGHT :
        if( imageGenerator.GetMapOutputMode(mode) == XN_STATUS_OK )
            propValue = mode.nYRes;
        break;
    case CV_CAP_PROP_FPS :
        if( imageGenerator.GetMapOutputMode(mode) == XN_STATUS_OK )
            propValue = mode.nFPS;
        break;
    case CV_CAP_PROP_POS_MSEC :
        propValue = (double)imageGenerator.GetTimestamp();
        break;
    case CV_CAP_PROP_POS_FRAMES :
        propValue = (double)imageGenerator.GetFrameID();
        break;
    default :
        CV_Error( CV_StsBadArg, cv::format("Image generator does not support such parameter (propIdx=%d) for getting.\n", propIdx) );
    }

    return propValue;
}

bool CvCapture_OpenNI::setImageGeneratorProperty( int propIdx, double propValue )
{
    bool isSet = false;
    if( !imageGenerator.IsValid() )
        return isSet;

    switch( propIdx )
    {
    case CV_CAP_PROP_OPENNI_OUTPUT_MODE :
    {
        XnMapOutputMode mode;

        switch( cvRound(propValue) )
        {
        case CV_CAP_OPENNI_VGA_30HZ :
            mode.nXRes = XN_VGA_X_RES;
            mode.nYRes = XN_VGA_Y_RES;
            mode.nFPS = 30;
            break;
        case CV_CAP_OPENNI_SXGA_15HZ :
            mode.nXRes = XN_SXGA_X_RES;
            mode.nYRes = XN_SXGA_Y_RES;
            mode.nFPS = 15;
            break;
        case CV_CAP_OPENNI_SXGA_30HZ :
            mode.nXRes = XN_SXGA_X_RES;
            mode.nYRes = XN_SXGA_Y_RES;
            mode.nFPS = 30;
            break;
        case CV_CAP_OPENNI_QVGA_30HZ :
             mode.nXRes = XN_QVGA_X_RES;
             mode.nYRes = XN_QVGA_Y_RES;
             mode.nFPS = 30;
             break;
        case CV_CAP_OPENNI_QVGA_60HZ :
             mode.nXRes = XN_QVGA_X_RES;
             mode.nYRes = XN_QVGA_Y_RES;
             mode.nFPS = 60;
             break;
        default :
            CV_Error( CV_StsBadArg, "Unsupported image generator output mode.\n");
        }

        XnStatus status = imageGenerator.SetMapOutputMode( mode );
        if( status != XN_STATUS_OK )
            fprintf(stderr, "CvCapture_OpenNI::setImageGeneratorProperty : %s\n", xnGetStatusString(status));
        else
            isSet = true;
        break;
    }
    default:
        CV_Error( CV_StsBadArg, cv::format("Image generator does not support such parameter (propIdx=%d) for setting.\n", propIdx) );
    }

    return isSet;
}

bool CvCapture_OpenNI::grabFrame()
{
    if( !isOpened() )
        return false;

    bool isGrabbed = false;
    if( !approxSyncGrabber.empty() && approxSyncGrabber->isRun() )
    {
        isGrabbed = approxSyncGrabber->grab( depthMetaData, imageMetaData );
    }
    else
    {
        XnStatus status = context.WaitAndUpdateAll();
        if( status != XN_STATUS_OK )
            return false;

        if( depthGenerator.IsValid() )
            depthGenerator.GetMetaData( depthMetaData );
        if( imageGenerator.IsValid() )
            imageGenerator.GetMetaData( imageMetaData );
        isGrabbed = true;
    }

    return isGrabbed;
}

inline void getDepthMapFromMetaData( const xn::DepthMetaData& depthMetaData, cv::Mat& depthMap, XnUInt64 noSampleValue, XnUInt64 shadowValue )
{
    int cols = depthMetaData.XRes();
    int rows = depthMetaData.YRes();

    depthMap.create( rows, cols, CV_16UC1 );

    const XnDepthPixel* pDepthMap = depthMetaData.Data();

    // CV_Assert( sizeof(unsigned short) == sizeof(XnDepthPixel) );
    memcpy( depthMap.data, pDepthMap, cols*rows*sizeof(XnDepthPixel) );

    cv::Mat badMask = (depthMap == (double)noSampleValue) | (depthMap == (double)shadowValue) | (depthMap == 0);

    // mask the pixels with invalid depth
    depthMap.setTo( cv::Scalar::all( CvCapture_OpenNI::INVALID_PIXEL_VAL ), badMask );
}

IplImage* CvCapture_OpenNI::retrieveDepthMap()
{
    if( !depthMetaData.Data() )
        return 0;

    getDepthMapFromMetaData( depthMetaData, outputMaps[CV_CAP_OPENNI_DEPTH_MAP].mat, noSampleValue, shadowValue );

    return outputMaps[CV_CAP_OPENNI_DEPTH_MAP].getIplImagePtr();
}

IplImage* CvCapture_OpenNI::retrievePointCloudMap()
{
    if( !depthMetaData.Data() )
        return 0;

    cv::Mat depth;
    getDepthMapFromMetaData( depthMetaData, depth, noSampleValue, shadowValue );

    const int badPoint = INVALID_PIXEL_VAL;
    const float badCoord = INVALID_COORDINATE_VAL;
    int cols = depthMetaData.XRes(), rows = depthMetaData.YRes();
    cv::Mat pointCloud_XYZ( rows, cols, CV_32FC3, cv::Scalar::all(badPoint) );

    std::vector<XnPoint3D> proj(cols*rows);
    std::vector<XnPoint3D> real(cols*rows);
    for( int y = 0; y < rows; y++ )
    {
        for( int x = 0; x < cols; x++ )
        {
            int ind = y*cols+x;
            proj[ind].X = (float)x;
            proj[ind].Y = (float)y;
            proj[ind].Z = depth.at<unsigned short>(y, x);
        }
    }
    depthGenerator.ConvertProjectiveToRealWorld(cols*rows, &proj.front(), &real.front());

    for( int y = 0; y < rows; y++ )
    {
        for( int x = 0; x < cols; x++ )
        {
            // Check for invalid measurements
            if( depth.at<unsigned short>(y, x) == badPoint ) // not valid
                pointCloud_XYZ.at<cv::Point3f>(y,x) = cv::Point3f( badCoord, badCoord, badCoord );
            else
            {
                int ind = y*cols+x;
                pointCloud_XYZ.at<cv::Point3f>(y,x) = cv::Point3f( real[ind].X*0.001f, real[ind].Y*0.001f, real[ind].Z*0.001f); // from mm to meters
            }
        }
    }

    outputMaps[CV_CAP_OPENNI_POINT_CLOUD_MAP].mat = pointCloud_XYZ;

    return outputMaps[CV_CAP_OPENNI_POINT_CLOUD_MAP].getIplImagePtr();
}

static void computeDisparity_32F( const xn::DepthMetaData& depthMetaData, cv::Mat& disp, XnDouble baseline, XnUInt64 F,
                           XnUInt64 noSampleValue, XnUInt64 shadowValue )
{
    cv::Mat depth;
    getDepthMapFromMetaData( depthMetaData, depth, noSampleValue, shadowValue );
    CV_Assert( depth.type() == CV_16UC1 );


    // disparity = baseline * F / z;

    float mult = (float)(baseline /*mm*/ * F /*pixels*/);

    disp.create( depth.size(), CV_32FC1);
    disp = cv::Scalar::all( CvCapture_OpenNI::INVALID_PIXEL_VAL );
    for( int y = 0; y < disp.rows; y++ )
    {
        for( int x = 0; x < disp.cols; x++ )
        {
            unsigned short curDepth = depth.at<unsigned short>(y,x);
            if( curDepth != CvCapture_OpenNI::INVALID_PIXEL_VAL )
                disp.at<float>(y,x) = mult / curDepth;
        }
    }
}

IplImage* CvCapture_OpenNI::retrieveDisparityMap()
{
    if( !depthMetaData.Data() )
        return 0;

    cv::Mat disp32;
    computeDisparity_32F( depthMetaData, disp32, baseline, depthFocalLength_VGA, noSampleValue, shadowValue );

    disp32.convertTo( outputMaps[CV_CAP_OPENNI_DISPARITY_MAP].mat, CV_8UC1 );

    return outputMaps[CV_CAP_OPENNI_DISPARITY_MAP].getIplImagePtr();
}

IplImage* CvCapture_OpenNI::retrieveDisparityMap_32F()
{
    if( !depthMetaData.Data() )
        return 0;

    computeDisparity_32F( depthMetaData, outputMaps[CV_CAP_OPENNI_DISPARITY_MAP_32F].mat, baseline, depthFocalLength_VGA, noSampleValue, shadowValue );

    return outputMaps[CV_CAP_OPENNI_DISPARITY_MAP_32F].getIplImagePtr();
}

IplImage* CvCapture_OpenNI::retrieveValidDepthMask()
{
    if( !depthMetaData.Data() )
        return 0;

    cv::Mat depth;
    getDepthMapFromMetaData( depthMetaData, depth, noSampleValue, shadowValue );

    outputMaps[CV_CAP_OPENNI_VALID_DEPTH_MASK].mat = depth != CvCapture_OpenNI::INVALID_PIXEL_VAL;

    return outputMaps[CV_CAP_OPENNI_VALID_DEPTH_MASK].getIplImagePtr();
}

inline void getBGRImageFromMetaData( const xn::ImageMetaData& imageMetaData, cv::Mat& bgrImage )
{
    if( imageMetaData.PixelFormat() != XN_PIXEL_FORMAT_RGB24 )
        CV_Error( CV_StsUnsupportedFormat, "Unsupported format of grabbed image\n" );

    cv::Mat rgbImage( imageMetaData.YRes(), imageMetaData.XRes(), CV_8UC3 );
    const XnRGB24Pixel* pRgbImage = imageMetaData.RGB24Data();

    // CV_Assert( 3*sizeof(uchar) == sizeof(XnRGB24Pixel) );
    memcpy( rgbImage.data, pRgbImage, rgbImage.total()*sizeof(XnRGB24Pixel) );
    cv::cvtColor( rgbImage, bgrImage, CV_RGB2BGR );
}

IplImage* CvCapture_OpenNI::retrieveBGRImage()
{
    if( !imageMetaData.Data() )
        return 0;

    getBGRImageFromMetaData( imageMetaData, outputMaps[CV_CAP_OPENNI_BGR_IMAGE].mat );

    return outputMaps[CV_CAP_OPENNI_BGR_IMAGE].getIplImagePtr();
}

IplImage* CvCapture_OpenNI::retrieveGrayImage()
{
    if( !imageMetaData.Data() )
        return 0;

    CV_Assert( imageMetaData.BytesPerPixel() == 3 ); // RGB

    cv::Mat rgbImage;
    getBGRImageFromMetaData( imageMetaData, rgbImage );
    cv::cvtColor( rgbImage, outputMaps[CV_CAP_OPENNI_GRAY_IMAGE].mat, CV_BGR2GRAY );

    return outputMaps[CV_CAP_OPENNI_GRAY_IMAGE].getIplImagePtr();
}

IplImage* CvCapture_OpenNI::retrieveFrame( int outputType )
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

    return image;
}


CvCapture* cvCreateCameraCapture_OpenNI( int index )
{
    CvCapture_OpenNI* capture = new CvCapture_OpenNI( index );

    if( capture->isOpened() )
        return capture;

    delete capture;
    return 0;
}

CvCapture* cvCreateFileCapture_OpenNI( const char* filename )
{
    CvCapture_OpenNI* capture = new CvCapture_OpenNI( filename );

    if( capture->isOpened() )
        return capture;

    delete capture;
    return 0;
}

#endif
