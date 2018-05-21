#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static void help()
{
        cout << "\nThis program demonstrates usage of depth sensors (Kinect, XtionPRO,...).\n"
                        "The user gets some of the supported output images.\n"
            "\nAll supported output map types:\n"
            "1.) Data given from depth generator\n"
            "   CAP_OPENNI_DEPTH_MAP            - depth values in mm (CV_16UC1)\n"
            "   CAP_OPENNI_POINT_CLOUD_MAP      - XYZ in meters (CV_32FC3)\n"
            "   CAP_OPENNI_DISPARITY_MAP        - disparity in pixels (CV_8UC1)\n"
            "   CAP_OPENNI_DISPARITY_MAP_32F    - disparity in pixels (CV_32FC1)\n"
            "   CAP_OPENNI_VALID_DEPTH_MASK     - mask of valid pixels (not ocluded, not shaded etc.) (CV_8UC1)\n"
            "2.) Data given from RGB image generator\n"
            "   CAP_OPENNI_BGR_IMAGE            - color image (CV_8UC3)\n"
            "   CAP_OPENNI_GRAY_IMAGE           - gray image (CV_8UC1)\n"
            "2.) Data given from IR image generator\n"
            "   CAP_OPENNI_IR_IMAGE             - gray image (CV_16UC1)\n"
         << endl;
}

static void colorizeDisparity( const Mat& gray, Mat& rgb, double maxDisp=-1.f)
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

    Mat tmp;
    convertScaleAbs(gray, tmp, 255.f / maxDisp);
    applyColorMap(tmp, rgb, COLORMAP_JET);
}

static float getMaxDisparity( VideoCapture& capture )
{
    const int minDistance = 400; // mm
    float b = (float)capture.get( CAP_OPENNI_DEPTH_GENERATOR_BASELINE ); // mm
    float F = (float)capture.get( CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH ); // pixels
    return b * F / minDistance;
}

static void printCommandLineParams()
{
    cout << "-cd=       Colorized disparity? (0 or 1; 1 by default) Ignored if disparity map is not selected to show." << endl;
    cout << "-fmd=      Fixed max disparity? (0 or 1; 0 by default) Ignored if disparity map is not colorized (-cd 0)." << endl;
    cout << "-mode=     image mode: resolution and fps, supported three values:  0 - CAP_OPENNI_VGA_30HZ, 1 - CAP_OPENNI_SXGA_15HZ," << endl;
    cout << "          2 - CAP_OPENNI_SXGA_30HZ (0 by default). Ignored if rgb image or gray image are not selected to show." << endl;
    cout << "-m=        Mask to set which output images are need. It is a string of size 5. Each element of this is '0' or '1' and" << endl;
    cout << "          determine: is depth map, disparity map, valid pixels mask, rgb image, gray image need or not (correspondently), ir image" << endl ;
    cout << "          By default -m=010100 i.e. disparity map and rgb image will be shown." << endl ;
    cout << "-r=        Filename of .oni video file. The data will grabbed from it." << endl ;
}

static void parseCommandLine( int argc, char* argv[], bool& isColorizeDisp, bool& isFixedMaxDisp, int& imageMode, bool retrievedImageFlags[],
                       string& filename, bool& isFileReading )
{
    filename.clear();
    cv::CommandLineParser parser(argc, argv, "{h help||}{cd|1|}{fmd|0|}{mode|-1|}{m|010100|}{r||}");
    if (parser.has("h"))
    {
        help();
        printCommandLineParams();
        exit(0);
    }
    isColorizeDisp = (parser.get<int>("cd") != 0);
    isFixedMaxDisp = (parser.get<int>("fmd") != 0);
    imageMode = parser.get<int>("mode");
    int flags = parser.get<int>("m");
    isFileReading = parser.has("r");
    if (isFileReading)
        filename = parser.get<string>("r");
    if (!parser.check())
    {
        parser.printErrors();
        help();
        exit(-1);
    }
    if (flags % 1000000 == 0)
    {
        cout << "No one output image is selected." << endl;
        exit(0);
    }
    for (int i = 0; i < 6; i++)
    {
        retrievedImageFlags[5 - i] = (flags % 10 != 0);
        flags /= 10;
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
    bool retrievedImageFlags[6];
    string filename;
    bool isVideoReading;
    parseCommandLine( argc, argv, isColorizeDisp, isFixedMaxDisp, imageMode, retrievedImageFlags, filename, isVideoReading );

    cout << "Device opening ..." << endl;
    VideoCapture capture;
    if( isVideoReading )
        capture.open( filename );
    else
    {
        capture.open( CAP_OPENNI2 );
        if( !capture.isOpened() )
            capture.open( CAP_OPENNI );
    }

    cout << "done." << endl;

    if( !capture.isOpened() )
    {
        cout << "Can not open a capture object." << endl;
        return -1;
    }

    if( !isVideoReading && imageMode >= 0 )
    {
        bool modeRes=false;
        switch ( imageMode )
        {
            case 0:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_VGA_30HZ );
                break;
            case 1:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_SXGA_15HZ );
                break;
            case 2:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_SXGA_30HZ );
                break;
                //The following modes are only supported by the Xtion Pro Live
            case 3:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_QVGA_30HZ );
                break;
            case 4:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_QVGA_60HZ );
                break;
            default:
                CV_Error( Error::StsBadArg, "Unsupported image mode property.\n");
        }
        if (!modeRes)
            cout << "\nThis image mode is not supported by the device, the default value (CV_CAP_OPENNI_SXGA_15HZ) will be used.\n" << endl;
    }

    // turn on depth, color and IR if needed
    if (retrievedImageFlags[0] || retrievedImageFlags[1] || retrievedImageFlags[2])
        capture.set(CAP_OPENNI_DEPTH_GENERATOR_PRESENT, true);
    else
        capture.set(CAP_OPENNI_DEPTH_GENERATOR_PRESENT, false);
    if (retrievedImageFlags[3] || retrievedImageFlags[4])
        capture.set(CAP_OPENNI_IMAGE_GENERATOR_PRESENT, true);
    else
        capture.set(CAP_OPENNI_IMAGE_GENERATOR_PRESENT, false);
    if (retrievedImageFlags[5])
        capture.set(CAP_OPENNI_IR_GENERATOR_PRESENT, true);
    else
        capture.set(CAP_OPENNI_IR_GENERATOR_PRESENT, false);

    // Print some avalible device settings.
    if (capture.get(CAP_OPENNI_DEPTH_GENERATOR_PRESENT))
    {
        cout << "\nDepth generator output mode:" << endl <<
            "FRAME_WIDTH      " << capture.get(CAP_PROP_FRAME_WIDTH) << endl <<
            "FRAME_HEIGHT     " << capture.get(CAP_PROP_FRAME_HEIGHT) << endl <<
            "FRAME_MAX_DEPTH  " << capture.get(CAP_PROP_OPENNI_FRAME_MAX_DEPTH) << " mm" << endl <<
            "FPS              " << capture.get(CAP_PROP_FPS) << endl <<
            "REGISTRATION     " << capture.get(CAP_PROP_OPENNI_REGISTRATION) << endl;
    }
    else
    {
        cout << "\nDevice doesn't contain depth generator or it is not selected." << endl;
    }

    if( capture.get( CAP_OPENNI_IMAGE_GENERATOR_PRESENT ) )
    {
        cout <<
            "\nImage generator output mode:" << endl <<
            "FRAME_WIDTH   " << capture.get( CAP_OPENNI_IMAGE_GENERATOR+CAP_PROP_FRAME_WIDTH ) << endl <<
            "FRAME_HEIGHT  " << capture.get( CAP_OPENNI_IMAGE_GENERATOR+CAP_PROP_FRAME_HEIGHT ) << endl <<
            "FPS           " << capture.get( CAP_OPENNI_IMAGE_GENERATOR+CAP_PROP_FPS ) << endl;
    }
    else
    {
        cout << "\nDevice doesn't contain image generator or it is not selected." << endl;
    }

    if( capture.get(CAP_OPENNI_IR_GENERATOR_PRESENT) )
    {
        cout <<
            "\nIR generator output mode:" << endl <<
            "FRAME_WIDTH   " << capture.get(CAP_OPENNI_IR_GENERATOR + CAP_PROP_FRAME_WIDTH) << endl <<
            "FRAME_HEIGHT  " << capture.get(CAP_OPENNI_IR_GENERATOR + CAP_PROP_FRAME_HEIGHT) << endl <<
            "FPS           " << capture.get(CAP_OPENNI_IR_GENERATOR + CAP_PROP_FPS) << endl;
    }
    else
    {
        cout << "\nDevice doesn't contain IR generator or it is not selected." << endl;
    }

    for(;;)
    {
        Mat depthMap;
        Mat validDepthMap;
        Mat disparityMap;
        Mat bgrImage;
        Mat grayImage;
        Mat irImage;

        if( !capture.grab() )
        {
            cout << "Can not grab images." << endl;
            return -1;
        }
        else
        {
            if( retrievedImageFlags[0] && capture.retrieve( depthMap, CAP_OPENNI_DEPTH_MAP ) )
            {
                const float scaleFactor = 0.05f;
                Mat show; depthMap.convertTo( show, CV_8UC1, scaleFactor );
                imshow( "depth map", show );
            }

            if( retrievedImageFlags[1] && capture.retrieve( disparityMap, CAP_OPENNI_DISPARITY_MAP ) )
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

            if( retrievedImageFlags[2] && capture.retrieve( validDepthMap, CAP_OPENNI_VALID_DEPTH_MASK ) )
                imshow( "valid depth mask", validDepthMap );

            if( retrievedImageFlags[3] && capture.retrieve( bgrImage, CAP_OPENNI_BGR_IMAGE ) )
                imshow( "rgb image", bgrImage );

            if( retrievedImageFlags[4] && capture.retrieve( grayImage, CAP_OPENNI_GRAY_IMAGE ) )
                imshow( "gray image", grayImage );

            if( retrievedImageFlags[5] && capture.retrieve( irImage, CAP_OPENNI_IR_IMAGE ) )
            {
                Mat ir8;
                irImage.convertTo(ir8, CV_8U, 256.0 / 3500, 0.0);
                imshow("IR image", ir8);
            }
        }

        if( waitKey( 30 ) >= 0 )
            break;
    }

    return 0;
}
