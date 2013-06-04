#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ocl/ocl.hpp"

#include "opencv2/highgui/highgui_c.h"

#include <iostream>
#include <stdio.h>

int main( int, const char** ) { return 0; }

#if 0

using namespace std;
using namespace cv;
#define LOOP_NUM 10

const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;

int64 work_begin = 0;
int64 work_end = 0;

static void workBegin()
{
    work_begin = getTickCount();
}
static void workEnd()
{
    work_end += (getTickCount() - work_begin);
}


static double getTime(){
    return work_end /((double)cvGetTickFrequency() * 1000.);
}

void detect( Mat& img, vector<Rect>& faces,
    cv::ocl::OclCascadeClassifierBuf& cascade,
    double scale, bool calTime);

void detectCPU( Mat& img, vector<Rect>& faces,
    CascadeClassifier& cascade,
    double scale, bool calTime);

void Draw(Mat& img, vector<Rect>& faces, double scale);

// This function test if gpu_rst matches cpu_rst.
// If the two vectors are not equal, it will return the difference in vector size
// Else if will return (total diff of each cpu and gpu rects covered pixels)/(total cpu rects covered pixels)
double checkRectSimilarity(Size sz, std::vector<Rect>& cpu_rst, std::vector<Rect>& gpu_rst);

int main( int argc, const char** argv )
{
    const char* keys =
        "{ h | help       | false       | print help message }"
        "{ i | input      |             | specify input image }"
        "{ t | template   | ../../../data/haarcascades/haarcascade_frontalface_alt.xml  | specify template file }"
        "{ c | scale      |   1.0       | scale image }"
        "{ s | use_cpu    | false       | use cpu or gpu to process the image }";

    CommandLineParser cmd(argc, argv, keys);
    if (cmd.get<bool>("help"))
    {
        cout << "Avaible options:" << endl;
        cmd.printParams();
        return 0;
    }
    CvCapture* capture = 0;
    Mat frame, frameCopy, image;

    bool useCPU = cmd.get<bool>("s");
    string inputName = cmd.get<string>("i");
    string cascadeName = cmd.get<string>("t");
    double scale = cmd.get<double>("c");
    cv::ocl::OclCascadeClassifierBuf cascade;
    CascadeClassifier  cpu_cascade;

    if( !cascade.load( cascadeName ) || !cpu_cascade.load(cascadeName) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }

    if( inputName.empty() )
    {
        capture = cvCaptureFromCAM(0);
        if(!capture)
            cout << "Capture from CAM 0 didn't work" << endl;
    }
    else if( inputName.size() )
    {
        image = imread( inputName, 1 );
        if( image.empty() )
        {
            capture = cvCaptureFromAVI( inputName.c_str() );
            if(!capture)
                cout << "Capture from AVI didn't work" << endl;
            return -1;
        }
    }
    else
    {
        image = imread( "lena.jpg", 1 );
        if(image.empty())
            cout << "Couldn't read lena.jpg" << endl;
        return -1;
    }

    cvNamedWindow( "result", 1 );
    std::vector<cv::ocl::Info> oclinfo;
    int devnums = cv::ocl::getDevice(oclinfo);
    if( devnums < 1 )
    {
        std::cout << "no device found\n";
        return -1;
    }
    //if you want to use undefault device, set it here
    //setDevice(oclinfo[0]);
    ocl::setBinpath("./");
    if( capture )
    {
        cout << "In capture ..." << endl;
        for(;;)
        {
            IplImage* iplImg = cvQueryFrame( capture );
            frame = cv::cvarrToMat(iplImg);
            vector<Rect> faces;
            if( frame.empty() )
                break;
            if( iplImg->origin == IPL_ORIGIN_TL )
                frame.copyTo( frameCopy );
            else
                flip( frame, frameCopy, 0 );
            if(useCPU){
                detectCPU(frameCopy, faces, cpu_cascade, scale, false);
            }
            else{
                detect(frameCopy, faces, cascade, scale, false);
            }
            Draw(frameCopy, faces, scale);
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
        vector<Rect> faces;
        vector<Rect> ref_rst;
        double accuracy = 0.;
        for(int i = 0; i <= LOOP_NUM;i ++)
        {
            cout << "loop" << i << endl;
            if(useCPU){
                detectCPU(image, faces, cpu_cascade, scale, i==0?false:true);
            }
            else{
                detect(image, faces, cascade, scale, i==0?false:true);
                if(i == 0){
                    detectCPU(image, ref_rst, cpu_cascade, scale, false);
                    accuracy = checkRectSimilarity(image.size(), ref_rst, faces);
                }
            }
            if (i == LOOP_NUM)
            {
                if (useCPU)
                    cout << "average CPU time (noCamera) : ";
                else
                    cout << "average GPU time (noCamera) : ";
                cout << getTime() / LOOP_NUM << " ms" << endl;
                cout << "accuracy value: " << accuracy <<endl;
            }
        }
        Draw(image, faces, scale);
        waitKey(0);
    }

    cvDestroyWindow("result");

    return 0;
}

void detect( Mat& img, vector<Rect>& faces,
    cv::ocl::OclCascadeClassifierBuf& cascade,
    double scale, bool calTime)
{
    cv::ocl::oclMat image(img);
    cv::ocl::oclMat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    if(calTime) workBegin();
    cv::ocl::cvtColor( image, gray, COLOR_BGR2GRAY );
    cv::ocl::resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    cv::ocl::equalizeHist( smallImg, smallImg );

    cascade.detectMultiScale( smallImg, faces, 1.1,
        3, 0
        |CV_HAAR_SCALE_IMAGE
        , Size(30,30), Size(0, 0) );
    if(calTime) workEnd();
}

void detectCPU( Mat& img, vector<Rect>& faces,
    CascadeClassifier& cascade,
    double scale, bool calTime)
{
    if(calTime) workBegin();
    Mat cpu_gray, cpu_smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    cvtColor(img, cpu_gray, CV_BGR2GRAY);
    resize(cpu_gray, cpu_smallImg, cpu_smallImg.size(), 0, 0, INTER_LINEAR);
    equalizeHist(cpu_smallImg, cpu_smallImg);
    cascade.detectMultiScale(cpu_smallImg, faces, 1.1,
        3, 0 | CV_HAAR_SCALE_IMAGE,
        Size(30, 30), Size(0, 0));
    if(calTime) workEnd();
}

void Draw(Mat& img, vector<Rect>& faces, double scale)
{
    int i = 0;
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
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

double checkRectSimilarity(Size sz, std::vector<Rect>& ob1, std::vector<Rect>& ob2)
{
    double final_test_result = 0.0;
    size_t sz1 = ob1.size();
    size_t sz2 = ob2.size();

    if(sz1 != sz2)
        return sz1 > sz2 ? (double)(sz1 - sz2) : (double)(sz2 - sz1);
    else
    {
        cv::Mat cpu_result(sz, CV_8UC1);
        cpu_result.setTo(0);

        for(vector<Rect>::const_iterator r = ob1.begin(); r != ob1.end(); r++)
        {
            cv::Mat cpu_result_roi(cpu_result, *r);
            cpu_result_roi.setTo(1);
            cpu_result.copyTo(cpu_result);
        }
        int cpu_area = cv::countNonZero(cpu_result > 0);

        cv::Mat gpu_result(sz, CV_8UC1);
        gpu_result.setTo(0);
        for(vector<Rect>::const_iterator r2 = ob2.begin(); r2 != ob2.end(); r2++)
        {
            cv::Mat gpu_result_roi(gpu_result, *r2);
            gpu_result_roi.setTo(1);
            gpu_result.copyTo(gpu_result);
        }

        cv::Mat result_;
        multiply(cpu_result, gpu_result, result_);
        int result = cv::countNonZero(result_ > 0);

        final_test_result = 1.0 - (double)result/(double)cpu_area;
    }
    return final_test_result;
}
#endif
