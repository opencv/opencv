#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ocl/ocl.hpp"

#include "opencv2/highgui/highgui_c.h"

#include <iostream>
#include <stdio.h>


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
                                  CV_RGB(255,0,255)
                                } ;


int64 work_begin = 0;
int64 work_end = 0;
string outputName;

static void workBegin()
{
    work_begin = getTickCount();
}
static void workEnd()
{
    work_end += (getTickCount() - work_begin);
}

static double getTime()
{
    return work_end /((double)cvGetTickFrequency() * 1000.);
}

void detect( Mat& img, vector<Rect>& faces,
             ocl::OclCascadeClassifier& cascade,
             double scale, bool calTime);


void detectCPU( Mat& img, vector<Rect>& faces,
                CascadeClassifier& cascade,
                double scale, bool calTime);

void Draw(Mat& img, vector<Rect>& faces, double scale);


// This function test if gpu_rst matches cpu_rst.
// If the two vectors are not equal, it will return the difference in vector size
// Else if will return (total diff of each cpu and gpu rects covered pixels)/(total cpu rects covered pixels)
double checkRectSimilarity(Size sz, vector<Rect>& cpu_rst, vector<Rect>& gpu_rst);


int main( int argc, const char** argv )
{
    const char* keys =
        "{ h  help       | false       | print help message }"
        "{ i  input      |             | specify input image }"
        "{ t  template   | haarcascade_frontalface_alt.xml |"
        " specify template file path }"
        "{ c  scale      |   1.0       | scale image }"
        "{ s  use_cpu    | false       | use cpu or gpu to process the image }"
        "{ o  output     | facedetect_output.jpg  |"
        " specify output image save path(only works when input is images) }";

    CommandLineParser cmd(argc, argv, keys);
    if (cmd.get<bool>("help"))
    {
        cout << "Avaible options:" << endl;
        return 0;
    }
    CvCapture* capture = 0;
    Mat frame, frameCopy, image;

    bool useCPU = cmd.get<bool>("s");
    string inputName = cmd.get<string>("i");
    outputName = cmd.get<string>("o");
    string cascadeName = cmd.get<string>("t");
    double scale = cmd.get<double>("c");
    ocl::OclCascadeClassifier cascade;
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
    vector<ocl::Info> oclinfo;
    int devnums = ocl::getDevice(oclinfo);
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
            if(useCPU)
            {
                detectCPU(frameCopy, faces, cpu_cascade, scale, false);
            }
            else
            {
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
        for(int i = 0; i <= LOOP_NUM; i ++)
        {
            cout << "loop" << i << endl;
            if(useCPU)
            {
                detectCPU(image, faces, cpu_cascade, scale, i==0?false:true);
            }
            else
            {
                detect(image, faces, cascade, scale, i==0?false:true);
                if(i == 0)
                {
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
             ocl::OclCascadeClassifier& cascade,
             double scale, bool calTime)
{
    ocl::oclMat image(img);
    ocl::oclMat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    if(calTime) workBegin();
    ocl::cvtColor( image, gray, COLOR_BGR2GRAY );
    ocl::resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    ocl::equalizeHist( smallImg, smallImg );

    cascade.detectMultiScale( smallImg, faces, 1.1,
                              3, 0
                              |CASCADE_SCALE_IMAGE
                              , Size(30,30), Size(0, 0) );
    if(calTime) workEnd();
}

void detectCPU( Mat& img, vector<Rect>& faces,
                CascadeClassifier& cascade,
                double scale, bool calTime)
{
    if(calTime) workBegin();
    Mat cpu_gray, cpu_smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    cvtColor(img, cpu_gray, COLOR_BGR2GRAY);
    resize(cpu_gray, cpu_smallImg, cpu_smallImg.size(), 0, 0, INTER_LINEAR);
    equalizeHist(cpu_smallImg, cpu_smallImg);
    cascade.detectMultiScale(cpu_smallImg, faces, 1.1,
                             3, 0 | CASCADE_SCALE_IMAGE,
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
    imwrite( outputName, img );
    if(abs(scale-1.0)>.001)
    {
        resize(img, img, Size((int)(img.cols/scale), (int)(img.rows/scale)));
    }
    imshow( "result", img );
    
}


double checkRectSimilarity(Size sz, vector<Rect>& ob1, vector<Rect>& ob2)
{
    double final_test_result = 0.0;
    size_t sz1 = ob1.size();
    size_t sz2 = ob2.size();

    if(sz1 != sz2)
    {
        return sz1 > sz2 ? (double)(sz1 - sz2) : (double)(sz2 - sz1);
    }
    else
    {
        if(sz1==0 && sz2==0)
            return 0;
        Mat cpu_result(sz, CV_8UC1);
        cpu_result.setTo(0);

        for(vector<Rect>::const_iterator r = ob1.begin(); r != ob1.end(); r++)
        {
            Mat cpu_result_roi(cpu_result, *r);
            cpu_result_roi.setTo(1);
            cpu_result.copyTo(cpu_result);
        }
        int cpu_area = countNonZero(cpu_result > 0);


        Mat gpu_result(sz, CV_8UC1);
        gpu_result.setTo(0);
        for(vector<Rect>::const_iterator r2 = ob2.begin(); r2 != ob2.end(); r2++)
        {
            cv::Mat gpu_result_roi(gpu_result, *r2);
            gpu_result_roi.setTo(1);
            gpu_result.copyTo(gpu_result);
        }

        Mat result_;
        multiply(cpu_result, gpu_result, result_);
        int result = countNonZero(result_ > 0);
        if(cpu_area!=0 && result!=0)
            final_test_result = 1.0 - (double)result/(double)cpu_area;
        else if(cpu_area==0 && result!=0)
            final_test_result = -1;
    }
    return final_test_result;
}