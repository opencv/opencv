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
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#ifdef HAVE_OPENNI

#include <iostream>
#include "XnCppWrapper.h"

const std::string XMLConfig =
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

class CvCapture_OpenNI : public CvCapture
{
public:
    static const int INVALID_PIXEL_VAL = 0;
    static const int INVALID_COORDINATE_VAL = 0;

    CvCapture_OpenNI(int index=0);
    CvCapture_OpenNI(const char * filename);
    virtual ~CvCapture_OpenNI();

    virtual double getProperty(int propIdx);
    virtual bool setProperty(int probIdx, double propVal);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int outputType);

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

    IplImage* retrieveDepthMap();
    IplImage* retrievePointCloudMap();
    IplImage* retrieveDisparityMap();
    IplImage* retrieveDisparityMap_32F();
    IplImage* retrieveValidDepthMask();
    IplImage* retrieveBGRImage();
    IplImage* retrieveGrayImage();

    bool readCamerasParams();

    double getDepthGeneratorProperty(int propIdx);
    bool setDepthGeneratorProperty(int propIdx, double propVal);
    double getImageGeneratorProperty(int propIdx);
    bool setImageGeneratorProperty(int propIdx, double propVal);

    // OpenNI context
    xn::Context context;
    bool isContextOpened;
    
    // Data generators with its metadata
    xn::DepthGenerator depthGenerator;
    xn::DepthMetaData  depthMetaData;

    xn::ImageGenerator imageGenerator;
    xn::ImageMetaData  imageMetaData;

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

    iplHeader = IplImage(mat);
    return &iplHeader;
}

bool CvCapture_OpenNI::isOpened() const
{
    return isContextOpened;
}

XnMapOutputMode defaultMapOutputMode()
{
    XnMapOutputMode mode;
    mode.nXRes = XN_VGA_X_RES;
    mode.nYRes = XN_VGA_Y_RES;
    mode.nFPS  = 30;
    return mode;
}

CvCapture_OpenNI::CvCapture_OpenNI( int index )
{
    isContextOpened = false;

    // Initialize and configure the context.
    if( context.Init() == XN_STATUS_OK )
    {
        // Find devices
        xn::NodeInfoList devicesList;
        XnStatus status = context.EnumerateProductionTrees( XN_NODE_TYPE_DEVICE, NULL, devicesList, 0 );
        if( status != XN_STATUS_OK )
        {
            std::cerr << "CvCapture_OpenNI::CvCapture_OpenNI : Failed to enumerate production trees: "
                      << std::string(xnGetStatusString(status)) << std::endl;
            return;
        }

        // Chose device according to index
        xn::NodeInfoList::Iterator it = devicesList.Begin();
        for( int i = 0; i < index; ++i ) it++;

        xn::NodeInfo deviceNode = *it;
        status = context.CreateProductionTree( deviceNode );
        if( status != XN_STATUS_OK )
        {
            std::cerr << "CvCapture_OpenNI::CvCapture_OpenNI : Failed to create production tree: "
                      << std::string(xnGetStatusString(status)) << std::endl;
            return;
        }

        status = context.RunXmlScript( XMLConfig.c_str() );
        if( status != XN_STATUS_OK )
        {
            std::cerr << "CvCapture_OpenNI::CvCapture_OpenNI : Failed to run xml script: "
                      << std::string(xnGetStatusString(status)) << std::endl;
            return;
        }

        // Associate generators with context.
        // enumerate the nodes to find if depth generator is present
        xn::NodeInfoList depthList;
        status = context.EnumerateExistingNodes( depthList, XN_NODE_TYPE_DEPTH );
        if( status != XN_STATUS_OK )
        {
            std::cerr << "CvCapture_OpenNI::CvCapture_OpenNI : Failed to enumerate depth generators: "
                      << std::string(xnGetStatusString(status)) << std::endl;
            return;
        }
        if( depthList.IsEmpty() )
        {
            std::cerr << "CvCapture_OpenNI::CvCapture_OpenNI : The device doesn't have depth generator. Such devices aren't supported now." << std::endl;
            return;
        }
        status = depthGenerator.Create( context );
        if( status != XN_STATUS_OK )
        {
            std::cerr << "CvCapture_OpenNI::CvCapture_OpenNI : Failed to create depth generator: "
                      << std::string(xnGetStatusString(status)) << std::endl;
            return;
        }

        // enumerate the nodes to find if image generator is present
        xn::NodeInfoList imageList;
        status = context.EnumerateExistingNodes( imageList, XN_NODE_TYPE_IMAGE );
        if( status != XN_STATUS_OK )
        {
            std::cerr << "CvCapture_OpenNI::CvCapture_OpenNI : Failed to enumerate image generators: "
                      << std::string(xnGetStatusString(status)) << std::endl;
            return;
        }

        if( !imageList.IsEmpty() )
        {
            status = imageGenerator.Create( context );
            if( status != XN_STATUS_OK )
            {
                std::cerr << "CvCapture_OpenNI::CvCapture_OpenNI : Failed to create image generator: "
                          <<  std::string(xnGetStatusString(status)) << std::endl;
                return;
            }
        }

        // Set map output mode.
        if( depthGenerator.IsValid() )
            CV_DbgAssert( depthGenerator.SetMapOutputMode(defaultMapOutputMode()) == XN_STATUS_OK ); // xn::DepthGenerator supports VGA only! (Jan 2011)
        if( imageGenerator.IsValid() )
            CV_DbgAssert( imageGenerator.SetMapOutputMode(defaultMapOutputMode()) == XN_STATUS_OK );

        //  Start generating data.
        status = context.StartGeneratingAll();
        if( status != XN_STATUS_OK )
        {
            std::cerr << "CvCapture_OpenNI::CvCapture_OpenNI : Failed to start generating OpenNI data: "
                      << std::string(xnGetStatusString(status)) << std::endl;
            return;
        }

        if( !readCamerasParams() )
        {
            std::cerr << "CvCapture_OpenNI::CvCapture_OpenNI : Could not read cameras parameters" << std::endl;
            return;
        }

        outputMaps.resize( outputMapsTypesCount );

        isContextOpened = true;
    }

    setProperty(CV_CAP_PROP_OPENNI_REGISTRATION, 1.0);
}

CvCapture_OpenNI::CvCapture_OpenNI(const char * filename)
{
    CV_Assert(0);
}

CvCapture_OpenNI::~CvCapture_OpenNI()
{
    context.StopGeneratingAll();
    context.Shutdown();
}

bool CvCapture_OpenNI::readCamerasParams()
{
    XnDouble pixelSize = 0;
    if( depthGenerator.GetRealProperty( "ZPPS", pixelSize ) != XN_STATUS_OK )
    {
        std::cerr << "CvCapture_OpenNI::readCamerasParams : Could not read pixel size!" << std::endl;
        return false;
    }

    // pixel size @ VGA = pixel size @ SXGA x 2
    pixelSize *= 2.0; // in mm

    // focal length of IR camera in pixels for VGA resolution
    XnUInt64 zeroPlanDistance; // in mm
    if( depthGenerator.GetIntProperty( "ZPD", zeroPlanDistance ) != XN_STATUS_OK )
    {
        std::cerr << "CvCapture_OpenNI::readCamerasParams : Could not read virtual plane distance!" << std::endl;
        return false;
    }

    if( depthGenerator.GetRealProperty( "LDDIS", baseline ) != XN_STATUS_OK )
    {
        std::cerr << "CvCapture_OpenNI::readCamerasParams : Could not read base line!" << std::endl;
        return false;
    }

    // baseline from cm -> mm
    baseline *= 10;

    // focal length from mm -> pixels (valid for 640x480)
    depthFocalLength_VGA = (XnUInt64)((double)zeroPlanDistance / (double)pixelSize);

    if( depthGenerator.GetIntProperty( "ShadowValue", shadowValue ) != XN_STATUS_OK )
    {
        std::cerr << "CvCapture_OpenNI::readCamerasParams : Could not read property \"ShadowValue\"!" << std::endl;
        return false;
    }

    if( depthGenerator.GetIntProperty("NoSampleValue", noSampleValue ) != XN_STATUS_OK )
    {
        std::cerr << "CvCapture_OpenNI::readCamerasParams : Could not read property \"NoSampleValue\"!" <<std::endl;
        return false;
    }

    return true;
}

double CvCapture_OpenNI::getProperty( int propIdx )
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
            CV_Error( CV_StsError, "Unsupported generator prefix!" );
        }
    }

    return propValue;
}

bool CvCapture_OpenNI::setProperty( int propIdx, double propValue )
{
    bool res = false;
    if( isOpened() )
    {
        int purePropIdx = propIdx & ~CV_CAP_OPENNI_GENERATORS_MASK;

        if( (propIdx & CV_CAP_OPENNI_GENERATORS_MASK) == CV_CAP_OPENNI_IMAGE_GENERATOR )
        {
            res = setImageGeneratorProperty( purePropIdx, propValue );
        }
        else if( (propIdx & CV_CAP_OPENNI_GENERATORS_MASK) == CV_CAP_OPENNI_DEPTH_GENERATOR )
        {
            res = setDepthGeneratorProperty( purePropIdx, propValue );
        }
        else
        {
            CV_Error( CV_StsError, "Unsupported generator prefix!" );
        }
    }

    return res;
}

double CvCapture_OpenNI::getDepthGeneratorProperty( int propIdx )
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
        propValue = depthGenerator.GetAlternativeViewPointCap().IsViewPointAs(imageGenerator) ? 1.0 : 0.0;
    default :
    {
        std::stringstream ss;
        ss << "Depth generator does not support such parameter (propIdx=" << propIdx << ") for getting.\n";
        CV_Error( CV_StsBadArg, ss.str().c_str() );
    }
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
                // then the property isn't avaliable
                if( imageGenerator.IsValid() )
                {
                    if( !depthGenerator.GetAlternativeViewPointCap().IsViewPointAs(imageGenerator) )
                    {
                        if( depthGenerator.GetAlternativeViewPointCap().IsViewPointSupported(imageGenerator) )
                        {
                            XnStatus status = depthGenerator.GetAlternativeViewPointCap().SetViewPoint(imageGenerator);
                            if( status != XN_STATUS_OK )
                                std::cerr << "CvCapture_OpenNI::setDepthGeneratorProperty : " << xnGetStatusString(status) << std::endl;
                            else
                                isSet = true;
                        }
                        else
                            std::cerr << "CvCapture_OpenNI::setDepthGeneratorProperty : Unsupported viewpoint." << std::endl;
                    }
                    else
                        isSet = true;
                }
            }
            else // "off"
            {
                XnStatus status = depthGenerator.GetAlternativeViewPointCap().ResetViewPoint();
                if( status != XN_STATUS_OK )
                    std::cerr << "CvCapture_OpenNI::setDepthGeneratorProperty : " << xnGetStatusString(status) << std::endl;
                else
                    isSet = true;
            }
        }
        break;
    default:
    {
        std::stringstream ss;
        ss << "Depth generator does not support such parameter (propIdx=" << propIdx << ") for setting.\n";
        CV_Error( CV_StsBadArg, ss.str().c_str() );
    }
    }

    return isSet;
}

double CvCapture_OpenNI::getImageGeneratorProperty( int propIdx )
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
    default :
    {
        std::stringstream ss;
        ss << "Image generator does not support such parameter (propIdx=" << propIdx << ") for getting.\n";
        CV_Error( CV_StsBadArg, ss.str().c_str() );
    }
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
        default :
            CV_Error( CV_StsBadArg, "Unsupported image generator output mode.\n");
        }

        XnStatus status = imageGenerator.SetMapOutputMode( mode );
        if( status != XN_STATUS_OK )
            std::cerr << "CvCapture_OpenNI::setImageGeneratorProperty : " << xnGetStatusString(status) << std::endl;
        else
            isSet = true;
        break;
    }
    default:
    {
        std::stringstream ss;
        ss << "Image generator does not support such parameter (propIdx=" << propIdx << ") for setting.\n";
        CV_Error( CV_StsBadArg, ss.str().c_str() );
    }
    }

    return isSet;
}

bool CvCapture_OpenNI::grabFrame()
{
    if( !isOpened() )
        return false;

    XnStatus status = context.WaitAndUpdateAll();
    if( status != XN_STATUS_OK )
        return false;

    if( depthGenerator.IsValid() )
        depthGenerator.GetMetaData( depthMetaData );
    if( imageGenerator.IsValid() )
        imageGenerator.GetMetaData( imageMetaData );

    return true;
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
    if( depthMetaData.XRes() <= 0 || depthMetaData.YRes() <= 0 )
        return 0;

    getDepthMapFromMetaData( depthMetaData, outputMaps[CV_CAP_OPENNI_DEPTH_MAP].mat, noSampleValue, shadowValue );

    return outputMaps[CV_CAP_OPENNI_DEPTH_MAP].getIplImagePtr();
}

IplImage* CvCapture_OpenNI::retrievePointCloudMap()
{
    int cols = depthMetaData.XRes(), rows = depthMetaData.YRes();
    if( cols <= 0 || rows <= 0 )
        return 0;

    cv::Mat depth;
    getDepthMapFromMetaData( depthMetaData, depth, noSampleValue, shadowValue );

    const int badPoint = INVALID_PIXEL_VAL;
    const float badCoord = INVALID_COORDINATE_VAL;
    cv::Mat pointCloud_XYZ( rows, cols, CV_32FC3, cv::Scalar::all(badPoint) );

    cv::Ptr<XnPoint3D> proj = new XnPoint3D[cols*rows];
    cv::Ptr<XnPoint3D> real = new XnPoint3D[cols*rows];
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
    depthGenerator.ConvertProjectiveToRealWorld(cols*rows, proj, real);

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

void computeDisparity_32F( const xn::DepthMetaData& depthMetaData, cv::Mat& disp, XnDouble baseline, XnUInt64 F,
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
    if( depthMetaData.XRes() <= 0 || depthMetaData.YRes() <= 0 )
        return 0;

    cv::Mat disp32;
    computeDisparity_32F( depthMetaData, disp32, baseline, depthFocalLength_VGA, noSampleValue, shadowValue );

    disp32.convertTo( outputMaps[CV_CAP_OPENNI_DISPARITY_MAP].mat, CV_8UC1 );

    return outputMaps[CV_CAP_OPENNI_DISPARITY_MAP].getIplImagePtr();
}

IplImage* CvCapture_OpenNI::retrieveDisparityMap_32F()
{
    if( depthMetaData.XRes() <= 0 || depthMetaData.YRes() <= 0 )
        return 0;

    computeDisparity_32F( depthMetaData, outputMaps[CV_CAP_OPENNI_DISPARITY_MAP_32F].mat, baseline, depthFocalLength_VGA, noSampleValue, shadowValue );

    return outputMaps[CV_CAP_OPENNI_DISPARITY_MAP_32F].getIplImagePtr();
}

IplImage* CvCapture_OpenNI::retrieveValidDepthMask()
{
    if( depthMetaData.XRes() <= 0 || depthMetaData.YRes() <= 0 )
        return 0;

    cv::Mat depth;
    getDepthMapFromMetaData( depthMetaData, depth, noSampleValue, shadowValue );

    outputMaps[CV_CAP_OPENNI_VALID_DEPTH_MASK].mat = depth != CvCapture_OpenNI::INVALID_PIXEL_VAL;

    return outputMaps[CV_CAP_OPENNI_VALID_DEPTH_MASK].getIplImagePtr();
}

inline void getBGRImageFromMetaData( const xn::ImageMetaData& imageMetaData, cv::Mat& bgrImage )
{
    int cols = imageMetaData.XRes();
    int rows = imageMetaData.YRes();

    cv::Mat rgbImage( rows, cols, CV_8UC3 );

    const XnRGB24Pixel* pRgbImage = imageMetaData.RGB24Data();

    // CV_Assert( 3*sizeof(uchar) == sizeof(XnRGB24Pixel) );
    memcpy( rgbImage.data, pRgbImage, cols*rows*sizeof(XnRGB24Pixel) );
    cv::cvtColor( rgbImage, bgrImage, CV_RGB2BGR );
}

IplImage* CvCapture_OpenNI::retrieveBGRImage()
{
    if( imageMetaData.XRes() <= 0 || imageMetaData.YRes() <= 0 )
        return 0;

    getBGRImageFromMetaData( imageMetaData, outputMaps[CV_CAP_OPENNI_BGR_IMAGE].mat );

    return outputMaps[CV_CAP_OPENNI_BGR_IMAGE].getIplImagePtr();
}

IplImage* CvCapture_OpenNI::retrieveGrayImage()
{
    if( imageMetaData.XRes() <= 0 || imageMetaData.YRes() <= 0 )
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
