//This sample is inherited from facedetect.cpp in smaple/c

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ocl/ocl.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

static void help()
{
    cout << "\nThis program demonstrates the cascade recognizer.\n"
        "This classifier can recognize many ~rigid objects, it's most known use is for faces.\n"
        "Usage:\n"
        "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
        "   [--scale=<image scale greater or equal to 1, try 1.3 for example>\n"
        "   [filename|camera_index]\n\n"
        "see facedetect.cmd for one call:\n"
        "./facedetect --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --scale=1.3 \n"
        "Hit any key to quit.\n"
        "Using OpenCV version " << CV_VERSION << "\n" << endl;
}
struct getRect { Rect operator ()(const CvAvgComp& e) const { return e.rect; } };
void detectAndDraw( Mat& img,
    cv::ocl::OclCascadeClassifier& cascade, CascadeClassifier& nestedCascade,
    double scale);

String cascadeName = "../../../data/haarcascades/haarcascade_frontalface_alt.xml";

int main( int argc, const char** argv )
{
    CvCapture* capture = 0;
    Mat frame, frameCopy, image;
    const String scaleOpt = "--scale=";
    size_t scaleOptLen = scaleOpt.length();
    const String cascadeOpt = "--cascade=";
    size_t cascadeOptLen = cascadeOpt.length();
    String inputName;

    help();
    cv::ocl::OclCascadeClassifier cascade;
    CascadeClassifier  nestedCascade;
    double scale = 1;

    for( int i = 1; i < argc; i++ )
    {
        cout << "Processing " << i << " " <<  argv[i] << endl;
        if( cascadeOpt.compare( 0, cascadeOptLen, argv[i], cascadeOptLen ) == 0 )
        {
            cascadeName.assign( argv[i] + cascadeOptLen );
            cout << "  from which we have cascadeName= " << cascadeName << endl;
        }
        else if( scaleOpt.compare( 0, scaleOptLen, argv[i], scaleOptLen ) == 0 )
        {
            if( !sscanf( argv[i] + scaleOpt.length(), "%lf", &scale ) || scale < 1 )
                scale = 1;
            cout << " from which we read scale = " << scale << endl;
        }
        else if( argv[i][0] == '-' )
        {
            cerr << "WARNING: Unknown option %s" << argv[i] << endl;
        }
        else
            inputName.assign( argv[i] );
    }

    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        cerr << "Usage: facedetect [--cascade=<cascade_path>]\n"
            "   [--scale[=<image scale>\n"
            "   [filename|camera_index]\n" << endl ;
        return -1;
    }

    if( inputName.empty() || (isdigit(inputName.c_str()[0]) && inputName.c_str()[1] == '\0') )
    {
        capture = cvCaptureFromCAM( inputName.empty() ? 0 : inputName.c_str()[0] - '0' );
        int c = inputName.empty() ? 0 : inputName.c_str()[0] - '0' ;
        if(!capture) cout << "Capture from CAM " <<  c << " didn't work" << endl;
    }
    else if( inputName.size() )
    {
        image = imread( inputName, 1 );
        if( image.empty() )
        {
            capture = cvCaptureFromAVI( inputName.c_str() );
            if(!capture) cout << "Capture from AVI didn't work" << endl;
        }
    }
    else
    {
        image = imread( "lena.jpg", 1 );
        if(image.empty()) cout << "Couldn't read lena.jpg" << endl;
    }

    cvNamedWindow( "result", 1 );
    std::vector<cv::ocl::Info> oclinfo;
    int devnums = cv::ocl::getDevice(oclinfo);
    if(devnums<1)
    {
        std::cout << "no device found\n";
        return -1;
    }
    //if you want to use undefault device, set it here
    //setDevice(oclinfo[0]);
    //setBinpath(CLBINPATH);
    if( capture )
    {
        cout << "In capture ..." << endl;
        for(;;)
        {
            IplImage* iplImg = cvQueryFrame( capture );
            frame = iplImg;
            if( frame.empty() )
                break;
            if( iplImg->origin == IPL_ORIGIN_TL )
                frame.copyTo( frameCopy );
            else
                flip( frame, frameCopy, 0 );

            detectAndDraw( frameCopy, cascade, nestedCascade, scale );

            if( waitKey( 10 ) >= 0 )
                goto _cleanup_;
        }

        waitKey(0);

_cleanup_:
        cvReleaseCapture( &capture );
    }
    else
    {
        cout << "In image read" << endl;
        if( !image.empty() )
        {
            detectAndDraw( image, cascade, nestedCascade, scale );
            waitKey(0);
        }
        else if( !inputName.empty() )
        {
            /* assume it is a text file containing the
            list of the image filenames to be processed - one per line */
            FILE* f = fopen( inputName.c_str(), "rt" );
            if( f )
            {
                char buf[1000+1];
                while( fgets( buf, 1000, f ) )
                {
                    int len = (int)strlen(buf), c;
                    while( len > 0 && isspace(buf[len-1]) )
                        len--;
                    buf[len] = '\0';
                    cout << "file " << buf << endl;
                    image = imread( buf, 1 );
                    if( !image.empty() )
                    {
                        detectAndDraw( image, cascade, nestedCascade, scale );
                        c = waitKey(0);
                        if( c == 27 || c == 'q' || c == 'Q' )
                            break;
                    }
                    else
                    {
                        cerr << "Aw snap, couldn't read image " << buf << endl;
                    }
                }
                fclose(f);
            }
        }
    }

    cvDestroyWindow("result");

    return 0;
}

void detectAndDraw( Mat& img,
    cv::ocl::OclCascadeClassifier& cascade, CascadeClassifier&,
    double scale)
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
    cv::ocl::oclMat image(img);
    cv::ocl::oclMat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

    cv::ocl::cvtColor( image, gray, CV_BGR2GRAY );
    cv::ocl::resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    cv::ocl::equalizeHist( smallImg, smallImg );

    CvSeq* _objects;
    MemStorage storage(cvCreateMemStorage(0));
    t = (double)cvGetTickCount();
    _objects = cascade.oclHaarDetectObjects( smallImg, storage, 1.1,
        3, 0
        |CV_HAAR_SCALE_IMAGE
        , Size(30,30), Size(0, 0) );
    vector<CvAvgComp> vecAvgComp;
    Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
    faces.resize(vecAvgComp.size());
    std::transform(vecAvgComp.begin(), vecAvgComp.end(), faces.begin(), getRect());
    t = (double)cvGetTickCount() - t;
    printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
        Mat smallImgROI;
        Point center;
        Scalar color = colors[i%8];
        int radius;
        center.x = cvRound((r->x + r->width*0.5)*scale);
        center.y = cvRound((r->y + r->height*0.5)*scale);
        radius = cvRound((r->width + r->height)*0.25*scale);
        circle( img, center, radius, color, 3, 8, 0 );
    }
    cv::imshow( "result", img );
}
