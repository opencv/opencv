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
#include "opencv2/imgproc/imgproc.hpp"
#ifdef HAVE_OPENNI

#define HACK_WITH_XML

#ifdef HACK_WITH_XML
#include <iostream>
#include <fstream>
#endif

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
                "<Node type=\"Image\" name=\"Image1\">"
                        "<Configuration>"
                                "<MapOutputMode xRes=\"640\" yRes=\"480\" FPS=\"30\"/>"
                                "<Mirror on=\"true\"/>"
                        "</Configuration>"
                "</Node> "
                "<Node type=\"Depth\" name=\"Depth1\">"
                        "<Configuration>"
                                "<MapOutputMode xRes=\"640\" yRes=\"480\" FPS=\"30\"/>"
                                "<Mirror on=\"true\"/>"
                        "</Configuration>"
                "</Node>"
        "</ProductionNodes>"
"</OpenNI>";

class CvCapture_OpenNI : public CvCapture
{
public:
    CvCapture_OpenNI();
    virtual ~CvCapture_OpenNI();

    virtual double getProperty(int);
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);

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

    static const int outputTypesCount = 7;

    static const unsigned short badDepth = 0;
    static const unsigned int badDisparity = 0;

    IplImage* retrieveDepthMap();
    IplImage* retrievePointCloudMap(); 
    IplImage* retrieveDisparityMap();
    IplImage* retrieveDisparityMap_32F();
    IplImage* retrieveValidDepthMask();
    IplImage* retrieveBGRImage();
    IplImage* retrieveGrayImage();

    void readCamerasParams();

    double getDepthGeneratorProperty(int);
    bool setDepthGeneratorProperty(int, double);
    double getImageGeneratorProperty(int);
    bool setImageGeneratorProperty(int, double);

    // OpenNI context
    xn::Context context;
    bool m_isOpened;

    // Data generators with its metadata
    xn::DepthGenerator depthGenerator;
    xn::DepthMetaData  depthMetaData;
    XnMapOutputMode depthOutputMode;

    xn::ImageGenerator imageGenerator;
    xn::ImageMetaData  imageMetaData;
    XnMapOutputMode imageOutputMode;

    // Cameras settings:
#if 1
    // Distance between IR projector and IR camera (in meters)
    XnDouble baseline;
    // Focal length for the IR camera in VGA resolution (in pixels)
    XnUInt64 depthFocalLength_VGA;
#endif
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
    return m_isOpened;
}

CvCapture_OpenNI::CvCapture_OpenNI()
{
    XnStatus status = XN_STATUS_OK;

    // Initialize image output modes (VGA_30HZ by default).
    depthOutputMode.nXRes = imageOutputMode.nXRes = XN_VGA_X_RES;
    depthOutputMode.nYRes = imageOutputMode.nYRes = XN_VGA_Y_RES;
    depthOutputMode.nFPS = imageOutputMode.nFPS = 30;

    m_isOpened = false;

    // Initialize and configure the context.
    if( context.Init() == XN_STATUS_OK )
    {
#ifdef HACK_WITH_XML
        // Write configuration to the temporary file.
        // This is a hack, because there is a bug in RunXmlScript().
        // TODO: remove hack when bug in RunXmlScript() will be fixed.
        char xmlFilename[100];
        tmpnam( xmlFilename );
        std::ofstream outfile( xmlFilename );
        outfile.write( XMLConfig.c_str(), XMLConfig.length() );
        outfile.close();

        status = context.RunXmlScriptFromFile( xmlFilename );

        // Remove temporary configuration file.
        remove( xmlFilename );
#else
        status = context.RunXmlScript( XMLConfig.c_str() );
#endif
        m_isOpened = ( status == XN_STATUS_OK );
    }

    if( m_isOpened )
    {
        // Associate generators with context.
        status = depthGenerator.Create( context );
        if( status != XN_STATUS_OK )
            CV_Error(CV_StsError, ("Failed to create depth generator: " + std::string(xnGetStatusString(status))).c_str() );
        imageGenerator.Create( context );
        if( status != XN_STATUS_OK )
            CV_Error(CV_StsError, ("Failed to create image generator: " + std::string(xnGetStatusString(status))).c_str() );

        // Set map output mode.
        CV_Assert( depthGenerator.SetMapOutputMode( depthOutputMode ) == XN_STATUS_OK ); // xn::DepthGenerator supports VGA only! (Jan 2011)
        CV_Assert( imageGenerator.SetMapOutputMode( imageOutputMode ) == XN_STATUS_OK );

        //  Start generating data.
        status = context.StartGeneratingAll();
        if( status != XN_STATUS_OK )
            CV_Error(CV_StsError, ("Failed to start generating OpenNI data: " + std::string(xnGetStatusString(status))).c_str() );

        readCamerasParams();

        outputMaps.resize( outputTypesCount );
    }
}

CvCapture_OpenNI::~CvCapture_OpenNI()
{
    context.StopGeneratingAll();
    context.Shutdown();
}

void CvCapture_OpenNI::readCamerasParams()
{
#if 1
    XnDouble pixelSize = 0;
    if( depthGenerator.GetRealProperty( "ZPPS", pixelSize ) != XN_STATUS_OK )
        CV_Error( CV_StsError, "Could not read pixel size!" );

    // pixel size @ VGA = pixel size @ SXGA x 2
    pixelSize *= 2.0; // in mm

    // focal length of IR camera in pixels for VGA resolution
    XnUInt64 zpd; // in mm
    if( depthGenerator.GetIntProperty( "ZPD", zpd ) != XN_STATUS_OK )
        CV_Error( CV_StsError, "Could not read virtual plane distance!" );

    if( depthGenerator.GetRealProperty( "LDDIS", baseline ) != XN_STATUS_OK )
        CV_Error( CV_StsError, "Could not read base line!" );

    // baseline from cm -> mm
    baseline *= 10;

    // focal length from mm -> pixels (valid for 640x480)
    depthFocalLength_VGA = (XnUInt64)((double)zpd / (double)pixelSize);
#endif

    if( depthGenerator.GetIntProperty( "ShadowValue", shadowValue ) != XN_STATUS_OK )
        CV_Error( CV_StsError, "Could not read shadow value!" );

    if( depthGenerator.GetIntProperty("NoSampleValue", noSampleValue ) != XN_STATUS_OK )
        CV_Error( CV_StsError, "Could not read no sample value!" );
}

double CvCapture_OpenNI::getProperty( int propIdx )
{
    double propValue = -1;

    if( isOpened() )
    {
        if( propIdx & OPENNI_IMAGE_GENERATOR )
        {
            propValue = getImageGeneratorProperty( propIdx ^ OPENNI_IMAGE_GENERATOR );
        }
        else // depth generator (by default, OPENNI_DEPTH_GENERATOR == 0)
        {
            propValue = getDepthGeneratorProperty( propIdx /*^ OPENNI_DEPTH_GENERATOR*/ );
        }
    }

    return propValue;
}

bool CvCapture_OpenNI::setProperty( int propIdx, double propValue )
{
    bool res = false;
    if( isOpened() )
    {
        if( propIdx & OPENNI_IMAGE_GENERATOR )
        {
            res = setImageGeneratorProperty( propIdx ^ OPENNI_IMAGE_GENERATOR, propValue );
        }
        else // depth generator (by default, OPENNI_DEPTH_GENERATOR == 0)
        {
            res = setDepthGeneratorProperty( propIdx /*^ OPENNI_DEPTH_GENERATOR*/, propValue );
        }
    }

    return false;
}

double CvCapture_OpenNI::getDepthGeneratorProperty( int propIdx )
{
    CV_Assert( depthGenerator.IsValid() );

    double res = -1;
    switch( propIdx )
    {
    case CV_CAP_PROP_FRAME_WIDTH :
        res = depthOutputMode.nXRes;
        break;
    case CV_CAP_PROP_FRAME_HEIGHT :
        res = depthOutputMode.nYRes;
        break;
    case CV_CAP_PROP_FPS :
        res = depthOutputMode.nFPS;
        break;
    case OPENNI_FRAME_MAX_DEPTH :
        res = depthGenerator.GetDeviceMaxDepth();
        break;
    case OPENNI_BASELINE :
        res = baseline;
        break;
    case OPENNI_FOCAL_LENGTH :
        res = depthFocalLength_VGA;
        break;
    default :
        CV_Error( CV_StsBadArg, "Depth generator does not support such parameter for getting.\n");
    }

    return res;
}

bool CvCapture_OpenNI::setDepthGeneratorProperty( int propIdx, double propValue )
{
    CV_Assert( depthGenerator.IsValid() );
    CV_Error( CV_StsBadArg, "Depth generator does not support such parameter for setting.\n");
}

double CvCapture_OpenNI::getImageGeneratorProperty( int propIdx )
{
    CV_Assert( imageGenerator.IsValid() );

    double res = -1;
    switch( propIdx )
    {
    case CV_CAP_PROP_FRAME_WIDTH :
        res = imageOutputMode.nXRes;
        break;
    case CV_CAP_PROP_FRAME_HEIGHT :
        res = imageOutputMode.nYRes;
        break;
    case CV_CAP_PROP_FPS :
        res = imageOutputMode.nFPS;
        break;
    default :
        CV_Error( CV_StsBadArg, "Image generator does not support such parameter for getting.\n");
    }

    return res;
}

bool CvCapture_OpenNI::setImageGeneratorProperty( int propIdx, double propValue )
{
    bool res = false;

    CV_Assert( imageGenerator.IsValid() );
    XnMapOutputMode newImageOutputMode = imageOutputMode;
    switch( propIdx )
    {
    case OPENNI_OUTPUT_MODE :
        switch( cvRound(propValue) )
        {
        case OPENNI_VGA_30HZ :
            newImageOutputMode.nXRes = XN_VGA_X_RES;
            newImageOutputMode.nYRes = XN_VGA_Y_RES;
            newImageOutputMode.nFPS = 30;
            break;
        case OPENNI_SXGA_15HZ :
            newImageOutputMode.nXRes = XN_SXGA_X_RES;
            newImageOutputMode.nYRes = XN_SXGA_Y_RES;
            newImageOutputMode.nFPS = 15;
            break;
        default :
            CV_Error( CV_StsBadArg, "Unsupported image generator output mode.\n");
        }
        break;

   default:
        CV_Error( CV_StsBadArg, "Image generator does not support such parameter for setting.\n");
    }

    if( imageGenerator.SetMapOutputMode( newImageOutputMode ) == XN_STATUS_OK )
    {
        imageOutputMode = newImageOutputMode;
        res = true;
    }

    return res;
}

bool CvCapture_OpenNI::grabFrame()
{
    if( !isOpened() )
        return false;

    XnStatus status = context.WaitAnyUpdateAll();
    if( status != XN_STATUS_OK )
        return false;

    depthGenerator.GetMetaData( depthMetaData );
    imageGenerator.GetMetaData( imageMetaData );
    return true;
}

inline void getDepthMapFromMetaData( const xn::DepthMetaData& depthMetaData, cv::Mat& depthMap, XnUInt64 noSampleValue, XnUInt64 shadowValue, unsigned short badDepth )
{
    int cols = depthMetaData.XRes();
    int rows = depthMetaData.YRes();

    depthMap.create( rows, cols, CV_16UC1 );

    const XnDepthPixel* pDepthMap = depthMetaData.Data();

    // CV_Assert( sizeof(unsigned short) == sizeof(XnDepthPixel) );
    memcpy( depthMap.data, pDepthMap, cols*rows*sizeof(XnDepthPixel) );

    cv::Mat badMask = (depthMap == noSampleValue) | (depthMap == shadowValue) | (depthMap == 0);

    // mask the pixels with invalid depth
    depthMap.setTo( cv::Scalar::all( badDepth ), badMask );
}

IplImage* CvCapture_OpenNI::retrieveDepthMap()
{
    if( depthMetaData.XRes() <= 0 || depthMetaData.YRes() <= 0 )
        return 0;

    getDepthMapFromMetaData( depthMetaData, outputMaps[OPENNI_DEPTH_MAP].mat, noSampleValue, shadowValue, badDepth );

    return outputMaps[OPENNI_DEPTH_MAP].getIplImagePtr();
}

IplImage* CvCapture_OpenNI::retrievePointCloudMap()
{
    int cols = depthMetaData.XRes(), rows = depthMetaData.YRes();
    if( cols <= 0 || rows <= 0 )
        return 0;

#if 0
    // X = (x - centerX) * depth / F[in pixels]
    // Y = (y - centerY) * depth / F[in pixels]
    // Z = depth
    // Multiply by 0.001 to convert from mm in meters.


    float mult = 0.001f / depthFocalLength_VGA;
    int centerX = cols >> 1;
    int centerY = rows >> 1;
#endif


    cv::Mat depth;
    getDepthMapFromMetaData( depthMetaData, depth, noSampleValue, shadowValue, badDepth );

    const float badPoint = 0;
    cv::Mat XYZ( rows, cols, CV_32FC3, cv::Scalar::all(badPoint) );

    for( int y = 0; y < rows; y++ )
    {
        for( int x = 0; x < cols; x++ )
        {

            unsigned short d = depth.at<unsigned short>(y, x);

            // Check for invalid measurements
            if( d == badDepth ) // not valid
                continue;
#if 0
            // Fill in XYZ
            cv::Point3f point3D;
            point3D.x = (x - centerX) * d * mult;
            point3D.y = (y - centerY) * d * mult;
            point3D.z = d * 0.001f;

            XYZ.at<cv::Point3f>(y,x) = point3D;
#else
            XnPoint3D proj, real;
            proj.X = x;
            proj.Y = y;
            proj.Z = d;
            depthGenerator.ConvertProjectiveToRealWorld(1, &proj, &real);
            XYZ.at<cv::Point3f>(y,x) = cv::Point3f( real.X*0.001f, real.Y*0.001f, real.Z*0.001f); // from mm to meters
#endif
        }
    }

    outputMaps[OPENNI_POINT_CLOUD_MAP].mat = XYZ;

    return outputMaps[OPENNI_POINT_CLOUD_MAP].getIplImagePtr();
}

void computeDisparity_32F( const xn::DepthMetaData& depthMetaData, cv::Mat& disp, XnDouble baseline, XnUInt64 F, 
                           XnUInt64 noSampleValue, XnUInt64 shadowValue, 
                           short badDepth, unsigned int badDisparity )
{
    cv::Mat depth;
    getDepthMapFromMetaData( depthMetaData, depth, noSampleValue, shadowValue, badDepth );
    CV_Assert( depth.type() == CV_16UC1 );


    // disparity = baseline * F / z;

    float mult = baseline /*mm*/ * F /*pixels*/;
    
    disp.create( depth.size(), CV_32FC1);
    disp = cv::Scalar::all(badDisparity);
    for( int y = 0; y < disp.rows; y++ )
    {
        for( int x = 0; x < disp.cols; x++ )
        {
            unsigned short curDepth = depth.at<unsigned short>(y,x);
            if( curDepth != badDepth )
                disp.at<float>(y,x) = mult / curDepth;
        }
    }
}

IplImage* CvCapture_OpenNI::retrieveDisparityMap()
{
    if( depthMetaData.XRes() <= 0 || depthMetaData.YRes() <= 0 )
        return 0;

    cv::Mat disp32;
    computeDisparity_32F( depthMetaData, disp32, baseline, depthFocalLength_VGA,
                          noSampleValue, shadowValue, badDepth, badDisparity );

    disp32.convertTo( outputMaps[OPENNI_DISPARITY_MAP].mat, CV_8UC1 );
    
    return outputMaps[OPENNI_DISPARITY_MAP].getIplImagePtr();
}

IplImage* CvCapture_OpenNI::retrieveDisparityMap_32F()
{
    if( depthMetaData.XRes() <= 0 || depthMetaData.YRes() <= 0 )
        return 0;

    computeDisparity_32F( depthMetaData, outputMaps[OPENNI_DISPARITY_MAP_32F].mat, baseline, depthFocalLength_VGA, 
                          noSampleValue, shadowValue, badDepth, badDisparity );

    return outputMaps[OPENNI_DISPARITY_MAP_32F].getIplImagePtr();
}

IplImage* CvCapture_OpenNI::retrieveValidDepthMask()
{
    if( depthMetaData.XRes() <= 0 || depthMetaData.YRes() <= 0 )
        return 0;

    cv::Mat depth;
    getDepthMapFromMetaData( depthMetaData, depth, noSampleValue, shadowValue, badDepth );

    outputMaps[OPENNI_VALID_DEPTH_MASK].mat = depth != badDepth;
    
    return outputMaps[OPENNI_VALID_DEPTH_MASK].getIplImagePtr();
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

    getBGRImageFromMetaData( imageMetaData, outputMaps[OPENNI_BGR_IMAGE].mat );

    return outputMaps[OPENNI_BGR_IMAGE].getIplImagePtr();
}

IplImage* CvCapture_OpenNI::retrieveGrayImage()
{
    if( imageMetaData.XRes() <= 0 || imageMetaData.YRes() <= 0 )
        return 0;

    CV_Assert( imageMetaData.BytesPerPixel() == 3 ); // RGB

    cv::Mat rgbImage;
    getBGRImageFromMetaData( imageMetaData, rgbImage );
    cv::cvtColor( rgbImage, outputMaps[OPENNI_GRAY_IMAGE].mat, CV_BGR2GRAY );

    return outputMaps[OPENNI_GRAY_IMAGE].getIplImagePtr();
}

IplImage* CvCapture_OpenNI::retrieveFrame( int dataType )
{
    IplImage* image = 0;
    CV_Assert( dataType < outputTypesCount && dataType >= 0);

    if( dataType == OPENNI_DEPTH_MAP )
    {
        image = retrieveDepthMap();
    }
    else if( dataType == OPENNI_POINT_CLOUD_MAP )
    {
        image = retrievePointCloudMap();
    }
    else if( dataType == OPENNI_DISPARITY_MAP )
    {
        image = retrieveDisparityMap();
    }
    else if( dataType == OPENNI_DISPARITY_MAP_32F )
    {
        image = retrieveDisparityMap_32F();
    }
    else if( dataType == OPENNI_VALID_DEPTH_MASK )
    {
        image = retrieveValidDepthMask();
    }
    else if( dataType == OPENNI_BGR_IMAGE )
    {
        image = retrieveBGRImage();
    }
    else if( dataType == OPENNI_GRAY_IMAGE )
    {
        image = retrieveGrayImage();
    }

    return image;
}


CvCapture* cvCreateCameraCapture_OpenNI( int /*index*/ )
{
    // TODO devices enumeration (if several Kinects)
    CvCapture_OpenNI* capture = new CvCapture_OpenNI();

    if( capture->isOpened() )
        return capture;

    delete capture;
    return 0;
}

#endif
