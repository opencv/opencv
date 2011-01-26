#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

#define COLORIZED_DISP              1
#define IMAGE_GENERATOR_VGA_30HZ    1
#define FIXED_MAX_DISP              0

void colorizeDisparity( const Mat& gray, Mat& rgb, double maxDisp=-1.f, float S=1.f, float V=1.f )
{
    CV_Assert( !gray.empty() );
    CV_Assert( gray.type() == CV_8UC1 );

    if( maxDisp <= 0 )
    {
        maxDisp = 0;
        minMaxLoc( gray, 0, &maxDisp );
    }

    rgb.create( gray.size(), CV_8UC3 );
    for( int y = 0; y < gray.rows; y++ )
    {
        for( int x = 0; x < gray.cols; x++ )
        {
            uchar d = gray.at<uchar>(y,x);
            unsigned int H = ((uchar)maxDisp - d) * 240 / (uchar)maxDisp;

            unsigned int hi = (H/60) % 6;
            float f = H/60.f - H/60;
            float p = V * (1 - S);
            float q = V * (1 - f * S);
            float t = V * (1 - (1 - f) * S);

            Point3f res;
            
            if( hi == 0 ) //R = V,	G = t,	B = p
                res = Point3f( p, t, V );
            if( hi == 1 ) // R = q,	G = V,	B = p
                res = Point3f( p, V, q );
            if( hi == 2 ) // R = p,	G = V,	B = t
                res = Point3f( t, V, p );
            if( hi == 3 ) // R = p,	G = q,	B = V
                res = Point3f( V, q, p );
            if( hi == 4 ) // R = t,	G = p,	B = V
                res = Point3f( V, p, t );
            if( hi == 5 ) // R = V,	G = p,	B = q
                res = Point3f( q, p, V );

            uchar b = (uchar)(std::max(0.f, std::min (res.x, 1.f)) * 255.f);
            uchar g = (uchar)(std::max(0.f, std::min (res.y, 1.f)) * 255.f);
            uchar r = (uchar)(std::max(0.f, std::min (res.z, 1.f)) * 255.f);

            rgb.at<Point3_<uchar> >(y,x) = Point3_<uchar>(b, g, r);     
        }
    }
}

void help()
{
	cout << "\nThis program demonstrates usage of Kinect sensor.\n"
			"The user gets some of the supported output images.\n" 
            "\nAll supported output map types:\n"
            "1.) Data given from depth generator\n"
            "   OPENNI_DEPTH_MAP            - depth values in mm (CV_16UC1)\n"
            "   OPENNI_POINT_CLOUD_MAP      - XYZ in meters (CV_32FC3)\n"
            "   OPENNI_DISPARITY_MAP        - disparity in pixels (CV_8UC1)\n"
            "   OPENNI_DISPARITY_MAP_32F    - disparity in pixels (CV_32FC1)\n"
            "   OPENNI_VALID_DEPTH_MASK     - mask of valid pixels (not ocluded, not shaded etc.) (CV_8UC1)\n"
            "2.) Data given from RGB image generator\n"
            "   OPENNI_BGR_IMAGE            - color image (CV_8UC3)\n"
            "   OPENNI_GRAY_IMAGE           - gray image (CV_8UC1)\n" 
         << endl;
}

float getMaxDisparity( VideoCapture& capture )
{
#if FIXED_MAX_DISP
    const int minDistance = 400; // mm
    float b = capture.get( OPENNI_DEPTH_GENERATOR_BASELINE ); // mm
    float F = capture.get( OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH ); // pixels
    return b * F / minDistance;
#else
    return -1;
#endif
}

/*
 * To work with Kinect the user must install OpenNI library and PrimeSensorModule for OpenNI and
 * configure OpenCV with WITH_OPENNI flag is ON (using CMake).
 */
int main()
{
    help();

    cout << "Kinect opening ..." << endl;
    VideoCapture capture(0); // or CV_CAP_OPENNI
    cout << "done." << endl;

    if( !capture.isOpened() )
    {
        cout << "Can not open a capture object." << endl;
        return -1;
    }

#if IMAGE_GENERATOR_VGA_30HZ
    capture.set( OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, OPENNI_VGA_30HZ ); // default
#else
    capture.set( OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, OPENNI_SXGA_15HZ );
#endif

    // Print some avalible Kinect settings.
    cout << "\nDepth generator output mode:" << endl <<
            "FRAME_WIDTH    " << capture.get( CV_CAP_PROP_FRAME_WIDTH ) << endl <<
            "FRAME_HEIGHT   " << capture.get( CV_CAP_PROP_FRAME_HEIGHT ) << endl <<
            "FRAME_MAX_DEPTH    " << capture.get( OPENNI_FRAME_MAX_DEPTH ) << " mm" << endl <<
            "FPS    " << capture.get( CV_CAP_PROP_FPS ) << endl;

    cout << "\nImage generator output mode:" << endl <<
            "FRAME_WIDTH    " << capture.get( OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_FRAME_WIDTH ) << endl <<
            "FRAME_HEIGHT   " << capture.get( OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_FRAME_HEIGHT ) << endl <<
            "FPS    " << capture.get( OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_FPS ) << endl;

    for(;;)
    {
        Mat depthMap;
        Mat validDepthMap;
        Mat disparityMap;
        Mat bgrImage;
        Mat grayImage;

        if( !capture.grab() )
        {
            cout << "Can not grab images." << endl;
            return -1;
        }
        else
        {
            if( capture.retrieve( depthMap, OPENNI_DEPTH_MAP ) )
            {
                const float scaleFactor = 0.05f;
                Mat show; depthMap.convertTo( show, CV_8UC1, scaleFactor );
                imshow( "depth map", show );
            }

            if( capture.retrieve( disparityMap, OPENNI_DISPARITY_MAP ) )
            {
#if COLORIZED_DISP // colorized disparity for more visibility
                Mat colorDisparityMap;
                colorizeDisparity( disparityMap, colorDisparityMap, getMaxDisparity( capture ) );
                Mat validColorDisparityMap;
                colorDisparityMap.copyTo( validColorDisparityMap, disparityMap != OPENNI_BAD_DISP_VAL );
                imshow( "colorized disparity map", validColorDisparityMap );
#else // original disparity
                imshow( "original disparity map", disparityMap );
#endif
            }

            if( capture.retrieve( validDepthMap, OPENNI_VALID_DEPTH_MASK ) )
                imshow( "valid depth mask", validDepthMap );

            if( capture.retrieve( bgrImage, OPENNI_BGR_IMAGE ) )
                imshow( "rgb image", bgrImage );

            if( capture.retrieve( grayImage, OPENNI_GRAY_IMAGE ) )
                imshow( "gray image", grayImage );
        }

        if( waitKey( 30 ) >= 0 )
            break;
    }

    return 0;
}
