#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ocl/ocl.hpp"

#include "opencv2/highgui/highgui_c.h"

#include <iostream>
#include <stdio.h>

#if defined(_MSC_VER) && (_MSC_VER >= 1700)
    # include <thread>
#endif

using namespace std;
using namespace cv;
#define LOOP_NUM 1

///////////////////////////single-threading faces detecting///////////////////////////////

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
string inputName, outputName, cascadeName;

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


static void detect( Mat& img, vector<Rect>& faces,
             ocl::OclCascadeClassifier& cascade,
             double scale);


static void detectCPU( Mat& img, vector<Rect>& faces,
                CascadeClassifier& cascade,
                double scale);

static void Draw(Mat& img, vector<Rect>& faces, double scale);


// This function test if gpu_rst matches cpu_rst.
// If the two vectors are not equal, it will return the difference in vector size
// Else if will return (total diff of each cpu and gpu rects covered pixels)/(total cpu rects covered pixels)
double checkRectSimilarity(Size sz, vector<Rect>& cpu_rst, vector<Rect>& gpu_rst);

static int facedetect_one_thread(bool useCPU, double scale )
{
    CvCapture* capture = 0;
    Mat frame, frameCopy0, frameCopy, image;

    ocl::OclCascadeClassifier cascade;
    CascadeClassifier  cpu_cascade;

    if( !cascade.load( cascadeName ) || !cpu_cascade.load(cascadeName) )
    {
        cout << "ERROR: Could not load classifier cascade: " << cascadeName << endl;
        return EXIT_FAILURE;
    }

    if( inputName.empty() )
    {
        capture = cvCaptureFromCAM(0);
        if(!capture)
            cout << "Capture from CAM 0 didn't work" << endl;
    }
    else
    {
        image = imread( inputName, CV_LOAD_IMAGE_COLOR );
        if( image.empty() )
        {
            capture = cvCaptureFromAVI( inputName.c_str() );
            if(!capture)
                cout << "Capture from AVI didn't work" << endl;
            return EXIT_FAILURE;
        }
    }

    cvNamedWindow( "result", 1 );
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
                frame.copyTo( frameCopy0 );
            else
                flip( frame, frameCopy0, 0 );
            if( scale == 1)
                frameCopy0.copyTo(frameCopy);
            else
                resize(frameCopy0, frameCopy, Size(), 1./scale, 1./scale, INTER_LINEAR);

            work_end = 0;
            if(useCPU)
                detectCPU(frameCopy, faces, cpu_cascade, 1);
            else
                detect(frameCopy, faces, cascade, 1);

            Draw(frameCopy, faces, 1);
            if( waitKey( 10 ) >= 0 )
                break;
        }
        cvReleaseCapture( &capture );
    }
    else
    {
        cout << "In image read" << endl;
        vector<Rect> faces;
        vector<Rect> ref_rst;
        double accuracy = 0.;
        detectCPU(image, ref_rst, cpu_cascade, scale);
        work_end = 0;

        for(int i = 0; i <= LOOP_NUM; i ++)
        {
            cout << "loop" << i << endl;
            if(useCPU)
                detectCPU(image, faces, cpu_cascade, scale);
            else
            {
                detect(image, faces, cascade, scale);
                if(i == 0)
                {
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
    std::cout<< "single-threaded sample has finished" <<std::endl;
    return 0;
}

///////////////////////////////////////detectfaces with multithreading////////////////////////////////////////////
#if defined(_MSC_VER) && (_MSC_VER >= 1700)

#define MAX_THREADS 10

static void detectFaces(std::string fileName)
{
    ocl::OclCascadeClassifier cascade;
    if(!cascade.load(cascadeName))
    {
        std::cout << "ERROR: Could not load classifier cascade: " << cascadeName << std::endl;
        return;
    }

    Mat img = imread(fileName, CV_LOAD_IMAGE_COLOR);
    if (img.empty())
    {
        std::cout << "cann't open file " + fileName <<std::endl;
        return;
    }

    ocl::oclMat d_img;
    d_img.upload(img);

    std::vector<Rect> oclfaces;
    cascade.detectMultiScale(d_img, oclfaces,  1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30), Size(0, 0));

    for(unsigned int i = 0; i<oclfaces.size(); i++)
        rectangle(img, Point(oclfaces[i].x, oclfaces[i].y), Point(oclfaces[i].x + oclfaces[i].width, oclfaces[i].y + oclfaces[i].height), colors[i%8], 3);

    std::string::size_type pos = outputName.rfind('.');
    std::string outputNameTid = outputName + '-' + std::to_string(_threadid);
    if(pos == std::string::npos)
    {
        std::cout << "Invalid output file name: " << outputName << std::endl;
    }
    else
    {
        outputNameTid = outputName.substr(0, pos) + "_" + std::to_string(_threadid) + outputName.substr(pos);
        imwrite(outputNameTid, img);
    }
    imshow(outputNameTid, img);
    waitKey(0);
}

static void facedetect_multithreading(int nthreads)
{
    int thread_number = MAX_THREADS < nthreads ? MAX_THREADS : nthreads;
    std::vector<std::thread> threads;
    for(int i = 0; i<thread_number; i++)
        threads.push_back(std::thread(detectFaces, inputName));
    for(int i = 0; i<thread_number; i++)
        threads[i].join();
}
#endif

int main( int argc, const char** argv )
{

    const char* keys =
        "{ h help       | false       | print help message }"
        "{ i input      |             | specify input image }"
        "{ t template   | haarcascade_frontalface_alt.xml |"
        " specify template file path }"
        "{ c scale      |   1.0       | scale image }"
        "{ s use_cpu    | false       | use cpu or gpu to process the image }"
        "{ o output     | facedetect_output.jpg  |"
        " specify output image save path(only works when input is images) }"
        "{ n thread_num |      1      | set number of threads >= 1 }";

    CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help"))
    {
        cout << "Usage : facedetect [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printMessage();
        return EXIT_SUCCESS;
    }
    bool useCPU = cmd.get<bool>("s");
    inputName = cmd.get<string>("i");
    outputName = cmd.get<string>("o");
    cascadeName = cmd.get<string>("t");
    double scale = cmd.get<double>("c");
    int n = cmd.get<int>("n");

    if(n > 1)
    {
#if defined(_MSC_VER) && (_MSC_VER >= 1700)
            std::cout<<"multi-threaded sample is running" <<std::endl;
            facedetect_multithreading(n);
            std::cout<<"multi-threaded sample has finished" <<std::endl;
            return 0;
#else
            std::cout << "std::thread is not supported, running a single-threaded version" << std::endl;
#endif
    }
    if (n<0)
        std::cout<<"incorrect number of threads:" << n << ", running a single-threaded version" <<std::endl;
    else
        std::cout<<"single-threaded sample is running" <<std::endl;
    return facedetect_one_thread(useCPU, scale);

}

void detect( Mat& img, vector<Rect>& faces,
             ocl::OclCascadeClassifier& cascade,
             double scale)
{
    ocl::oclMat image(img);
    ocl::oclMat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    workBegin();
    ocl::cvtColor( image, gray, COLOR_BGR2GRAY );
    ocl::resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    ocl::equalizeHist( smallImg, smallImg );

    cascade.detectMultiScale( smallImg, faces, 1.1,
                              3, 0
                              |CASCADE_SCALE_IMAGE
                              , Size(30,30), Size(0, 0) );
    workEnd();
}

void detectCPU( Mat& img, vector<Rect>& faces,
                CascadeClassifier& cascade,
                double scale)
{
    workBegin();
    Mat cpu_gray, cpu_smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    cvtColor(img, cpu_gray, COLOR_BGR2GRAY);
    resize(cpu_gray, cpu_smallImg, cpu_smallImg.size(), 0, 0, INTER_LINEAR);
    equalizeHist(cpu_smallImg, cpu_smallImg);
    cascade.detectMultiScale(cpu_smallImg, faces, 1.1,
                             3, 0 | CASCADE_SCALE_IMAGE,
                             Size(30, 30), Size(0, 0));
    workEnd();
}


void Draw(Mat& img, vector<Rect>& faces, double scale)
{
    int i = 0;
    putText(img, format("fps: %.1f", 1000./getTime()), Point(450, 50),
            FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0), 3);
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
    //imwrite( outputName, img );
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
