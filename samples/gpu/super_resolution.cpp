#include <iostream>
#include <iomanip>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/superres/superres.hpp"
#include "opencv2/superres/optical_flow.hpp"
#include "opencv2/opencv_modules.hpp"

#if defined(HAVE_OPENCV_OCL)
#include "opencv2/ocl/ocl.hpp"
#endif

using namespace std;
using namespace cv;
using namespace cv::superres;

#define MEASURE_TIME(op) \
    { \
        TickMeter tm; \
        tm.start(); \
        op; \
        tm.stop(); \
        cout << tm.getTimeSec() << " sec" << endl; \
    }

static Ptr<DenseOpticalFlowExt> createOptFlow(const string& name, bool useGpu)
{
    if (name == "farneback")
    {
        if (useGpu)
            return createOptFlow_Farneback_GPU();
        else
            return createOptFlow_Farneback();
    }
    else if (name == "simple")
        return createOptFlow_Simple();
    else if (name == "tvl1")
    {
        if (useGpu)
            return createOptFlow_DualTVL1_GPU();
        else
            return createOptFlow_DualTVL1();
    }
    else if (name == "brox")
        return createOptFlow_Brox_GPU();
    else if (name == "pyrlk")
        return createOptFlow_PyrLK_GPU();
    else
    {
        cerr << "Incorrect Optical Flow algorithm - " << name << endl;
        exit(-1);
    }
}
#if defined(HAVE_OPENCV_OCL)
static Ptr<DenseOpticalFlowExt> createOptFlow(const string& name)
{
    if (name == "farneback")
    {
        printf("farneback has not been implemented!\n");
        return NULL;
        //return createOptFlow_Farneback_GPU();
    }
    else if (name == "simple")
    {
        printf("simple has not been implemented!\n");
        return NULL;
        //return createOptFlow_Simple();
    }
    else if (name == "tvl1")
        return createOptFlow_DualTVL1_OCL();
    else if (name == "brox")
    {
        printf("simple has not been implemented!\n");
        return NULL;
        //return createOptFlow_Brox_OCL();
    }
    else if (name == "pyrlk")
        return createOptFlow_PyrLK_OCL();
    else
    {
        cerr << "Incorrect Optical Flow algorithm - " << name << endl;
    }
    return 0;
}
#endif
int main(int argc, const char* argv[])
{
    CommandLineParser cmd(argc, argv,
        "{ v   | video      |           | Input video }"
        "{ o   | output     |           | Output video }"
        "{ s   | scale      | 4         | Scale factor }"
        "{ i   | iterations | 180       | Iteration count }"
        "{ t   | temporal   | 4         | Radius of the temporal search area }"
        "{ f   | flow       | farneback | Optical flow algorithm (farneback, simple, tvl1, brox, pyrlk) }"
        "{ gpu | gpu        | false     | Use GPU }"
        "{ h   | help       | false     | Print help message }"
        "{ ocl | ocl        | false     | Use OCL }"
    );

    if (cmd.get<bool>("help"))
    {
        cout << "This sample demonstrates Super Resolution algorithms for video sequence" << endl;
        cmd.printParams();
        return 0;
    }

    const string inputVideoName = cmd.get<string>("video");
    const string outputVideoName = cmd.get<string>("output");
    const int scale = cmd.get<int>("scale");
    const int iterations = cmd.get<int>("iterations");
    const int temporalAreaRadius = cmd.get<int>("temporal");
    const string optFlow = cmd.get<string>("flow");
    const bool useGpu = cmd.get<bool>("gpu");
    const bool useOcl = cmd.get<bool>("ocl");

#ifndef HAVE_OPENCV_OCL
    if(useOcl)
    {
        {
            cout<<"OPENCL is not compiled\n";
            return 0;
        }
    }
#endif
#if defined(HAVE_OPENCV_OCL)
    std::vector<cv::ocl::Info>info;
    if(useGpu)
    {
        CV_Assert(!useOcl);
        info.clear();
    }
    
    if(useOcl)
    {
        CV_Assert(!useGpu);
        cv::ocl::getDevice(info);
    }
#endif
    Ptr<SuperResolution> superRes;
    if (useGpu)
        superRes = createSuperResolution_BTVL1_GPU();
#if defined(HAVE_OPENCV_OCL)
    else if(useOcl)
        superRes = createSuperResolution_BTVL1_OCL();
#endif
    else
        superRes = createSuperResolution_BTVL1();

    superRes->set("scale", scale);
    superRes->set("iterations", iterations);
    superRes->set("temporalAreaRadius", temporalAreaRadius);

#if defined(HAVE_OPENCV_OCL)
    if(useOcl)
    {
        Ptr<DenseOpticalFlowExt> of = createOptFlow(optFlow);
        if (of.empty())
            exit(-1);
        
        superRes->set("opticalFlow", of);
    }
    else
#endif
    {
        Ptr<DenseOpticalFlowExt> of = createOptFlow(optFlow, useGpu);
        
        if (of.empty())
            exit(-1);
        superRes->set("opticalFlow", of);
    }

    Ptr<FrameSource> frameSource;
    if (useGpu)
    {
        // Try to use gpu Video Decoding
        try
        {
            frameSource = createFrameSource_Video_GPU(inputVideoName);
            Mat frame;
            frameSource->nextFrame(frame);
        }
        catch (const cv::Exception&)
        {
            frameSource.release();
        }
    }
    if (frameSource.empty())
        frameSource = createFrameSource_Video(inputVideoName);

    // skip first frame, it is usually corrupted
    {
        Mat frame;
        frameSource->nextFrame(frame);
        cout << "Input           : " << inputVideoName << " " << frame.size() << endl;
        cout << "Scale factor    : " << scale << endl;
        cout << "Iterations      : " << iterations << endl;
        cout << "Temporal radius : " << temporalAreaRadius << endl;
        cout << "Optical Flow    : " << optFlow << endl;
#if defined(HAVE_OPENCV_OCL)
        cout << "Mode            : " << (useGpu ? "GPU" : useOcl? "OCL" : "CPU") << endl;
#else
        cout << "Mode            : " << (useGpu ? "GPU" : "CPU") << endl;
#endif
    }

    superRes->setInput(frameSource);

    VideoWriter writer;

    for (int i = 0;; ++i)
    {
        cout << '[' << setw(3) << i << "] : ";
        Mat result;

#if defined(HAVE_OPENCV_OCL)
        cv::ocl::oclMat result_;

        if(useOcl)
        {
            MEASURE_TIME(superRes->nextFrame(result_));
        }
        else
#endif
        {
            MEASURE_TIME(superRes->nextFrame(result));
        }

#ifdef HAVE_OPENCV_OCL
        if(useOcl)
        {
            if(!result_.empty())
            {
                Mat temp_res;
                result_.download(result);
            }
        }
#endif
        if (result.empty())
            break;

        imshow("Super Resolution", result);

        if (waitKey(1000) > 0)
            break;

        if (!outputVideoName.empty())
        {
            if (!writer.isOpened())
                writer.open(outputVideoName, CV_FOURCC('X', 'V', 'I', 'D'), 25.0, result.size());
            writer << result;
        }
    }

    return 0;
}
