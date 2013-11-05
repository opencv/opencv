#include <iostream>
#include <vector>
#include <iomanip>

#include "opencv2/core/utility.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ocl/ocl.hpp"
#include "opencv2/video/video.hpp"

using namespace std;
using namespace cv;
using namespace cv::ocl;

typedef unsigned char uchar;
#define LOOP_NUM 10
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
static double getTime()
{
    return work_end * 1000. / getTickFrequency();
}

template <typename T> inline T clamp (T x, T a, T b)
{
    return ((x) > (a) ? ((x) < (b) ? (x) : (b)) : (a));
}

template <typename T> inline T mapValue(T x, T a, T b, T c, T d)
{
    x = clamp(x, a, b);
    return c + (d - c) * (x - a) / (b - a);
}

static void getFlowField(const Mat& u, const Mat& v, Mat& flowField)
{
    float maxDisplacement = 1.0f;

    for (int i = 0; i < u.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);

        for (int j = 0; j < u.cols; ++j)
        {
            float d = max(fabsf(ptr_u[j]), fabsf(ptr_v[j]));

            if (d > maxDisplacement)
                maxDisplacement = d;
        }
    }

    flowField.create(u.size(), CV_8UC4);

    for (int i = 0; i < flowField.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);


        Vec4b* row = flowField.ptr<Vec4b>(i);

        for (int j = 0; j < flowField.cols; ++j)
        {
            row[j][0] = 0;
            row[j][1] = static_cast<unsigned char> (mapValue (-ptr_v[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][2] = static_cast<unsigned char> (mapValue ( ptr_u[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][3] = 255;
        }
    }
}


int main(int argc, const char* argv[])
{
    const char* keys =
        "{ h   | help       | false           | print help message }"
        "{ l   | left       |                 | specify left image }"
        "{ r   | right      |                 | specify right image }"
        "{ o   | output     | tvl1_output.jpg | specify output save path }"
        "{ c   | camera     | 0               | enable camera capturing }"
        "{ s   | use_cpu    | false           | use cpu or gpu to process the image }"
        "{ v   | video      |                 | use video as input }";

    CommandLineParser cmd(argc, argv, keys);

    if (cmd.get<bool>("help"))
    {
        cout << "Usage: pyrlk_optical_flow [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printMessage();
        return EXIT_SUCCESS;
    }

    string fname0 = cmd.get<string>("l");
    string fname1 = cmd.get<string>("r");
    string vdofile = cmd.get<string>("v");
    string outpath = cmd.get<string>("o");
    bool useCPU = cmd.get<bool>("s");
    bool useCamera = cmd.get<bool>("c");
    int inputName = cmd.get<int>("c");

    Mat frame0 = imread(fname0, cv::IMREAD_GRAYSCALE);
    Mat frame1 = imread(fname1, cv::IMREAD_GRAYSCALE);
    cv::Ptr<cv::DenseOpticalFlow> alg = cv::createOptFlow_DualTVL1();
    cv::ocl::OpticalFlowDual_TVL1_OCL d_alg;

    Mat flow, show_flow;
    Mat flow_vec[2];
    if (frame0.empty() || frame1.empty())
        useCamera = true;

    if (useCamera)
    {
        VideoCapture capture;
        Mat frame, frameCopy;
        Mat frame0Gray, frame1Gray;
        Mat ptr0, ptr1;

        if(vdofile.empty())
            capture.open( inputName );
        else
            capture.open(vdofile.c_str());

        if(!capture.isOpened())
        {
            if(vdofile.empty())
                cout << "Capture from CAM " << inputName << " didn't work" << endl;
            else
                cout << "Capture from file " << vdofile << " failed" <<endl;
            goto nocamera;
        }

        cout << "In capture ..." << endl;
        for(int i = 0;; i++)
        {
            if( !capture.read(frame) )
                break;

            if (i == 0)
            {
                frame.copyTo( frame0 );
                cvtColor(frame0, frame0Gray, COLOR_BGR2GRAY);
            }
            else
            {
                if (i%2 == 1)
                {
                    frame.copyTo(frame1);
                    cvtColor(frame1, frame1Gray, COLOR_BGR2GRAY);
                    ptr0 = frame0Gray;
                    ptr1 = frame1Gray;
                }
                else
                {
                    frame.copyTo(frame0);
                    cvtColor(frame0, frame0Gray, COLOR_BGR2GRAY);
                    ptr0 = frame1Gray;
                    ptr1 = frame0Gray;
                }

                if (useCPU)
                {
                    alg->calc(ptr0, ptr1, flow);
                    split(flow, flow_vec);
                }
                else
                {
                    oclMat d_flowx, d_flowy;
                    d_alg(oclMat(ptr0), oclMat(ptr1), d_flowx, d_flowy);
                    d_flowx.download(flow_vec[0]);
                    d_flowy.download(flow_vec[1]);
                }
                if (i%2 == 1)
                    frame1.copyTo(frameCopy);
                else
                    frame0.copyTo(frameCopy);
                getFlowField(flow_vec[0], flow_vec[1], show_flow);
                imshow("tvl1 optical flow field", show_flow);
            }

            if( waitKey( 10 ) >= 0 )
                break;
        }

        capture.release();
    }
    else
    {
nocamera:
        oclMat d_flowx, d_flowy;
        for(int i = 0; i <= LOOP_NUM; i ++)
        {
            cout << "loop" << i << endl;

            if (i > 0) workBegin();
            if (useCPU)
            {
                alg->calc(frame0, frame1, flow);
                split(flow, flow_vec);
            }
            else
            {
                d_alg(oclMat(frame0), oclMat(frame1), d_flowx, d_flowy);
                d_flowx.download(flow_vec[0]);
                d_flowy.download(flow_vec[1]);
            }
            if (i > 0 && i <= LOOP_NUM)
                workEnd();

            if (i == LOOP_NUM)
            {
                if (useCPU)
                    cout << "average CPU time (noCamera) : ";
                else
                    cout << "average GPU time (noCamera) : ";
                cout << getTime() / LOOP_NUM << " ms" << endl;

                getFlowField(flow_vec[0], flow_vec[1], show_flow);
                imshow("PyrLK [Sparse]", show_flow);
                imwrite(outpath, show_flow);
            }
        }
    }

    waitKey();

    return EXIT_SUCCESS;
}
