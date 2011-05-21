// WARNING: this sample is under construction! Use it on your own risk.

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void help()
{
    cout << "\nThis program demonstrates the multi cascade recognizer. It is a generalization of facedetect sample.\n\n"
            "Usage: ./multicascadeclassifier \n"
               "   --cascade1=<cascade_path> this is the primary trained classifier such as frontal face\n"
               "   [--cascade2=[this an optional secondary classifier such as profile face or eyes]]\n"
               "   input video or image\n\n"
            "Example: ./multicascadeclassifier --cascade1=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --cascade2=\"../../data/haarcascades/haarcascade_eye.xml\"\n\n"
            "Using OpenCV version " << CV_VERSION << endl << endl;
}

void DetectAndDraw(Mat& img, CascadeClassifier& cascade);

String cascadeName = "../../data/haarcascades/haarcascade_frontalface_alt.xml";
String nestedCascadeName = "../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

int main( int argc, const char** argv )
{
    CvCapture* capture = 0;
    Mat frame, image;

    if (argc == 0)
    {
        help();
        return 0;
    }

    const String cascadeOpt = "--cascade1=";
    size_t cascadeOptLen = cascadeOpt.length();
    string inputName;
    for( int i = 1; i < argc; i++ )
    {
        cout << "Processing argument #" << i << ": " <<  argv[i] << endl;
        if( cascadeOpt.compare( 0, cascadeOptLen, argv[i], cascadeOptLen ) == 0 )
        {
            cascadeName.assign( argv[i] + cascadeOptLen );
            cout << "  from which we have cascadeName= " << cascadeName << endl;
        }
        else if( argv[i][0] == '-' )
        {
            cerr << "WARNING: Unknown option " << argv[i] << endl;
        }
        else
            inputName.assign( argv[i] );
    }

    CascadeClassifier cascade;
    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load cascade classifier \"" << cascadeName << "\"" << endl;
        help();

        return -1;
    }

    if( inputName.size() )
    {
        image = imread( inputName, 1 );
        if( image.empty() )
        {
            capture = cvCaptureFromAVI( inputName.c_str() );
            if( !capture )
                cout << "Capture from AVI don't work" << endl;
        }
    }
    else
    {
        cout << "Please provide input file." << endl;
        return -1;
    }

    cvNamedWindow( "result", 1 );

    if( capture )
    {
        for(;;)
        {
            IplImage* iplImg = cvQueryFrame( capture );
            frame = iplImg;
            if( frame.empty() )
                break;

            DetectAndDraw( frame, cascade );

            if( waitKey( 10 ) >= 0 )
                goto _cleanup_;
        }

        waitKey(0);
_cleanup_:
        cvReleaseCapture( &capture );
    }
    else if( !image.empty() )
    {
        DetectAndDraw( image, cascade );
        waitKey(0);
    }
    else
    {
        cout << "Please provide correct input file." << endl;
    }

    cvDestroyWindow("result");

    return 0;
}

void DetectAndDraw( Mat& img, CascadeClassifier& cascade)
{
    int i = 0;
    double t = 0;
    vector<Rect> faces;
    const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    Mat gray;
    Mat frame( cvRound(img.rows), cvRound(img.cols), CV_8UC1 );

    cvtColor( img, gray, CV_BGR2GRAY );
    resize( gray, frame, frame.size(), 0, 0, INTER_LINEAR );
    equalizeHist( frame, frame );

    t = (double)cvGetTickCount();
    cascade.detectMultiScale( frame, faces,
        1.1, 2, 0
        //|CV_HAAR_FIND_BIGGEST_OBJECT
        //|CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE
        ,
        Size(30, 30) );
    t = (double)cvGetTickCount() - t;
    printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );

    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
        Point center;
        Scalar color = colors[i%8];
        int radius;
        center.x = cvRound(r->x + r->width*0.5);
        center.y = cvRound(r->y + r->height*0.5);
        radius = (int)(cvRound(r->width + r->height)*0.25);
        circle( img, center, radius, color, 3, 8, 0 );
    }

    cv::imshow( "result", img );
}
