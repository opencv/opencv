#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

void help()
{
        cout << "\nThis program demonstrates usage of depth sensors (Kinect, XtionPRO,...).\n"
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
    rgb = Scalar::all(0);
    if( maxDisp < 1 )
        return;

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

float getMaxDisparity( VideoCapture& capture )
{
    const int minDistance = 400; // mm
    float b = (float)capture.get( CV_CAP_OPENNI_DEPTH_GENERATOR_BASELINE ); // mm
    float F = (float)capture.get( CV_CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH ); // pixels
    return b * F / minDistance;
}

void printCommandLineParams()
{
    cout << "-cd       Colorized disparity? (0 or 1; 1 by default) Ignored if disparity map is not selected to show." << endl;
    cout << "-fmd      Fixed max disparity? (0 or 1; 0 by default) Ignored if disparity map is not colorized (-cd 0)." << endl;
    cout << "-mode     image mode: resolution and fps, supported three values:  0 - CV_CAP_OPENNI_VGA_30HZ, 1 - CV_CAP_OPENNI_SXGA_15HZ," << endl;
    cout << "          2 - CV_CAP_OPENNI_SXGA_30HZ (0 by default). Ignored if rgb image or gray image are not selected to show." << endl;
    cout << "-m        Mask to set which output images are need. It is a string of size 5. Each element of this is '0' or '1' and" << endl;
    cout << "          determine: is depth map, disparity map, valid pixels mask, rgb image, gray image need or not (correspondently)?" << endl ;
    cout << "          By default -m 01010 i.e. disparity map and rgb image will be shown." << endl ;
    cout << "-r        Filename of .oni video file. The data will grabbed from it." << endl ;
}

void parseCommandLine( int argc, char* argv[], bool& isColorizeDisp, bool& isFixedMaxDisp, int& imageMode, bool retrievedImageFlags[],
                       string& filename, bool& isFileReading )
{
    // set defaut values
    isColorizeDisp = true;
    isFixedMaxDisp = false;
    imageMode = 0;

    retrievedImageFlags[0] = false;
    retrievedImageFlags[1] = true;
    retrievedImageFlags[2] = false;
    retrievedImageFlags[3] = true;
    retrievedImageFlags[4] = false;

    filename.clear();
    isFileReading = false;

    if( argc == 1 )
    {
        help();
    }
    else
    {
        for( int i = 1; i < argc; i++ )
        {
            if( !strcmp( argv[i], "--help" ) || !strcmp( argv[i], "-h" ) )
            {
                printCommandLineParams();
                exit(0);
            }
            else if( !strcmp( argv[i], "-cd" ) )
            {
                isColorizeDisp = atoi(argv[++i]) == 0 ? false : true;
            }
            else if( !strcmp( argv[i], "-fmd" ) )
            {
                isFixedMaxDisp = atoi(argv[++i]) == 0 ? false : true;
            }
            else if( !strcmp( argv[i], "-mode" ) )
            {
                imageMode = atoi(argv[++i]);
            }
            else if( !strcmp( argv[i], "-m" ) )
            {
                string mask( argv[++i] );
                if( mask.size() != 5)
                    CV_Error( CV_StsBadArg, "Incorrect length of -m argument string" );
                int val = atoi(mask.c_str());

                int l = 100000, r = 10000, sum = 0;
                for( int i = 0; i < 5; i++ )
                {
                    retrievedImageFlags[i] = ((val % l) / r ) == 0 ? false : true;
                    l /= 10; r /= 10;
                    if( retrievedImageFlags[i] ) sum++;
                }

                if( sum == 0 )
                {
                    cout << "No one output image is selected." << endl;
                    exit(0);
                }
            }
            else if( !strcmp( argv[i], "-r" ) )
            {
                filename = argv[++i];
                isFileReading = true;
            }
            else
            {
                cout << "Unsupported command line argument: " << argv[i] << "." << endl;
                exit(-1);
            }
        }
    }
}

/*
 * To work with Kinect or XtionPRO the user must install OpenNI library and PrimeSensorModule for OpenNI and
 * configure OpenCV with WITH_OPENNI flag is ON (using CMake).
 */
int main( int argc, char* argv[] )
{
    bool isColorizeDisp, isFixedMaxDisp;
    int imageMode;
    bool retrievedImageFlags[5];
    string filename;
    bool isVideoReading;
    parseCommandLine( argc, argv, isColorizeDisp, isFixedMaxDisp, imageMode, retrievedImageFlags, filename, isVideoReading );

    cout << "Device opening ..." << endl;
    VideoCapture capture;
    if( isVideoReading )
        capture.open( filename );
    else
        capture.open( CV_CAP_OPENNI );

    cout << "done." << endl;

    if( !capture.isOpened() )
    {
        cout << "Can not open a capture object." << endl;
        return -1;
    }

    if( !isVideoReading )
    {
        bool modeRes=false;
        switch ( imageMode )
        {
            case 0:
                modeRes = capture.set( CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_VGA_30HZ );
                break;
            case 1:
                modeRes = capture.set( CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_SXGA_15HZ );
                break;
            case 2:
                modeRes = capture.set( CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_SXGA_30HZ );
                break;
            default:
                CV_Error( CV_StsBadArg, "Unsupported image mode property.\n");
        }
        if (!modeRes)
            cout << "\nThis image mode is not supported by the device, the default value (CV_CAP_OPENNI_SXGA_15HZ) will be used.\n" << endl;
    }

    // Print some avalible device settings.
    cout << "\nDepth generator output mode:" << endl <<
            "FRAME_WIDTH      " << capture.get( CV_CAP_PROP_FRAME_WIDTH ) << endl <<
            "FRAME_HEIGHT     " << capture.get( CV_CAP_PROP_FRAME_HEIGHT ) << endl <<
            "FRAME_MAX_DEPTH  " << capture.get( CV_CAP_PROP_OPENNI_FRAME_MAX_DEPTH ) << " mm" << endl <<
            "FPS              " << capture.get( CV_CAP_PROP_FPS ) << endl <<
            "REGISTRATION     " << capture.get( CV_CAP_PROP_OPENNI_REGISTRATION ) << endl;
    if( capture.get( CV_CAP_OPENNI_IMAGE_GENERATOR_PRESENT ) )
    {
        cout <<
            "\nImage generator output mode:" << endl <<
            "FRAME_WIDTH   " << capture.get( CV_CAP_OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_FRAME_WIDTH ) << endl <<
            "FRAME_HEIGHT  " << capture.get( CV_CAP_OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_FRAME_HEIGHT ) << endl <<
            "FPS           " << capture.get( CV_CAP_OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_FPS ) << endl;
    }
    else
    {
        cout << "\nDevice doesn't contain image generator." << endl;
        if (!retrievedImageFlags[0] && !retrievedImageFlags[1] && !retrievedImageFlags[2])
            return 0;
    }

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
            if( retrievedImageFlags[0] && capture.retrieve( depthMap, CV_CAP_OPENNI_DEPTH_MAP ) )
            {
                const float scaleFactor = 0.05f;
                Mat show; depthMap.convertTo( show, CV_8UC1, scaleFactor );
                imshow( "depth map", show );
            }

            if( retrievedImageFlags[1] && capture.retrieve( disparityMap, CV_CAP_OPENNI_DISPARITY_MAP ) )
            {
                if( isColorizeDisp )
                {
                    Mat colorDisparityMap;
                    colorizeDisparity( disparityMap, colorDisparityMap, isFixedMaxDisp ? getMaxDisparity(capture) : -1 );
                    Mat validColorDisparityMap;
                    colorDisparityMap.copyTo( validColorDisparityMap, disparityMap != 0 );
                    imshow( "colorized disparity map", validColorDisparityMap );
                }
                else
                {
                    imshow( "original disparity map", disparityMap );
                }
            }

            if( retrievedImageFlags[2] && capture.retrieve( validDepthMap, CV_CAP_OPENNI_VALID_DEPTH_MASK ) )
                imshow( "valid depth mask", validDepthMap );

            if( retrievedImageFlags[3] && capture.retrieve( bgrImage, CV_CAP_OPENNI_BGR_IMAGE ) )
                imshow( "rgb image", bgrImage );

            if( retrievedImageFlags[4] && capture.retrieve( grayImage, CV_CAP_OPENNI_GRAY_IMAGE ) )
                imshow( "gray image", grayImage );
        }

        if( waitKey( 30 ) >= 0 )
            break;
    }

    return 0;
}
