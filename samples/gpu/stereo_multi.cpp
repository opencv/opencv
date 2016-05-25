// This sample demonstrates working on one piece of data using two GPUs.
// It splits input into two parts and processes them separately on different GPUs.

#ifdef WIN32
    #define NOMINMAX
    #include <windows.h>
#else
    #include <pthread.h>
    #include <unistd.h>
#endif

#include <iostream>
#include <iomanip>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudastereo.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

///////////////////////////////////////////////////////////
// Thread
// OS-specific wrappers for multi-threading

#ifdef WIN32
class Thread
{
    struct UserData
    {
        void (*func)(void* userData);
        void* param;
    };

    static DWORD WINAPI WinThreadFunction(LPVOID lpParam)
    {
        UserData* userData = static_cast<UserData*>(lpParam);

        userData->func(userData->param);

        return 0;
    }

    UserData userData_;
    HANDLE thread_;
    DWORD threadId_;

public:
    Thread(void (*func)(void* userData), void* userData)
    {
        userData_.func = func;
        userData_.param = userData;

        thread_ = CreateThread(
            NULL,                   // default security attributes
            0,                      // use default stack size
            WinThreadFunction,      // thread function name
            &userData_,             // argument to thread function
            0,                      // use default creation flags
            &threadId_);            // returns the thread identifier
    }

    ~Thread()
    {
        CloseHandle(thread_);
    }

    void wait()
    {
        WaitForSingleObject(thread_, INFINITE);
    }
};
#else
class Thread
{
    struct UserData
    {
        void (*func)(void* userData);
        void* param;
    };

    static void* PThreadFunction(void* lpParam)
    {
        UserData* userData = static_cast<UserData*>(lpParam);

        userData->func(userData->param);

        return 0;
    }

    pthread_t thread_;
    UserData userData_;

public:
    Thread(void (*func)(void* userData), void* userData)
    {
        userData_.func = func;
        userData_.param = userData;

        pthread_create(&thread_, NULL, PThreadFunction, &userData_);
    }

    ~Thread()
    {
        pthread_detach(thread_);
    }

    void wait()
    {
        pthread_join(thread_, NULL);
    }
};
#endif

///////////////////////////////////////////////////////////
// StereoSingleGpu
// Run Stereo algorithm on single GPU

class StereoSingleGpu
{
public:
    explicit StereoSingleGpu(int deviceId = 0);
    ~StereoSingleGpu();

    void compute(const Mat& leftFrame, const Mat& rightFrame, Mat& disparity);

private:
    int deviceId_;
    GpuMat d_leftFrame;
    GpuMat d_rightFrame;
    GpuMat d_disparity;
    Ptr<cuda::StereoBM> d_alg;
};

StereoSingleGpu::StereoSingleGpu(int deviceId) : deviceId_(deviceId)
{
    cuda::setDevice(deviceId_);
    d_alg = cuda::createStereoBM(256);
}

StereoSingleGpu::~StereoSingleGpu()
{
    cuda::setDevice(deviceId_);
    d_leftFrame.release();
    d_rightFrame.release();
    d_disparity.release();
    d_alg.release();
}

void StereoSingleGpu::compute(const Mat& leftFrame, const Mat& rightFrame, Mat& disparity)
{
    cuda::setDevice(deviceId_);
    d_leftFrame.upload(leftFrame);
    d_rightFrame.upload(rightFrame);
    d_alg->compute(d_leftFrame, d_rightFrame, d_disparity);
    d_disparity.download(disparity);
}

///////////////////////////////////////////////////////////
// StereoMultiGpuThread
// Run Stereo algorithm on two GPUs using different host threads

class StereoMultiGpuThread
{
public:
    StereoMultiGpuThread();
    ~StereoMultiGpuThread();

    void compute(const Mat& leftFrame, const Mat& rightFrame, Mat& disparity);

private:
    GpuMat d_leftFrames[2];
    GpuMat d_rightFrames[2];
    GpuMat d_disparities[2];
    Ptr<cuda::StereoBM> d_algs[2];

    struct StereoLaunchData
    {
        int deviceId;
        Mat leftFrame;
        Mat rightFrame;
        Mat disparity;
        GpuMat* d_leftFrame;
        GpuMat* d_rightFrame;
        GpuMat* d_disparity;
        Ptr<cuda::StereoBM> d_alg;
    };

    static void launchGpuStereoAlg(void* userData);
};

StereoMultiGpuThread::StereoMultiGpuThread()
{
    cuda::setDevice(0);
    d_algs[0] = cuda::createStereoBM(256);

    cuda::setDevice(1);
    d_algs[1] = cuda::createStereoBM(256);
}

StereoMultiGpuThread::~StereoMultiGpuThread()
{
    cuda::setDevice(0);
    d_leftFrames[0].release();
    d_rightFrames[0].release();
    d_disparities[0].release();
    d_algs[0].release();

    cuda::setDevice(1);
    d_leftFrames[1].release();
    d_rightFrames[1].release();
    d_disparities[1].release();
    d_algs[1].release();
}

void StereoMultiGpuThread::compute(const Mat& leftFrame, const Mat& rightFrame, Mat& disparity)
{
    disparity.create(leftFrame.size(), CV_8UC1);

    // Split input data onto two parts for each GPUs.
    // We add small border for each part,
    // because original algorithm doesn't calculate disparity on image borders.
    // With such padding we will get output in the middle of final result.

    StereoLaunchData launchDatas[2];

    launchDatas[0].deviceId = 0;
    launchDatas[0].leftFrame = leftFrame.rowRange(0, leftFrame.rows / 2 + 32);
    launchDatas[0].rightFrame = rightFrame.rowRange(0, rightFrame.rows / 2 + 32);
    launchDatas[0].disparity = disparity.rowRange(0, leftFrame.rows / 2);
    launchDatas[0].d_leftFrame = &d_leftFrames[0];
    launchDatas[0].d_rightFrame = &d_rightFrames[0];
    launchDatas[0].d_disparity = &d_disparities[0];
    launchDatas[0].d_alg = d_algs[0];

    launchDatas[1].deviceId = 1;
    launchDatas[1].leftFrame = leftFrame.rowRange(leftFrame.rows / 2 - 32, leftFrame.rows);
    launchDatas[1].rightFrame = rightFrame.rowRange(leftFrame.rows / 2 - 32, leftFrame.rows);
    launchDatas[1].disparity = disparity.rowRange(leftFrame.rows / 2, leftFrame.rows);
    launchDatas[1].d_leftFrame = &d_leftFrames[1];
    launchDatas[1].d_rightFrame = &d_rightFrames[1];
    launchDatas[1].d_disparity = &d_disparities[1];
    launchDatas[1].d_alg = d_algs[1];

    Thread thread0(launchGpuStereoAlg, &launchDatas[0]);
    Thread thread1(launchGpuStereoAlg, &launchDatas[1]);

    thread0.wait();
    thread1.wait();
}

void StereoMultiGpuThread::launchGpuStereoAlg(void* userData)
{
    StereoLaunchData* data = static_cast<StereoLaunchData*>(userData);

    cuda::setDevice(data->deviceId);
    data->d_leftFrame->upload(data->leftFrame);
    data->d_rightFrame->upload(data->rightFrame);
    data->d_alg->compute(*data->d_leftFrame, *data->d_rightFrame, *data->d_disparity);

    if (data->deviceId == 0)
        data->d_disparity->rowRange(0, data->d_disparity->rows - 32).download(data->disparity);
    else
        data->d_disparity->rowRange(32, data->d_disparity->rows).download(data->disparity);
}

///////////////////////////////////////////////////////////
// StereoMultiGpuStream
// Run Stereo algorithm on two GPUs from single host thread using async API

class StereoMultiGpuStream
{
public:
    StereoMultiGpuStream();
    ~StereoMultiGpuStream();

    void compute(const HostMem& leftFrame, const HostMem& rightFrame, HostMem& disparity);

private:
    GpuMat d_leftFrames[2];
    GpuMat d_rightFrames[2];
    GpuMat d_disparities[2];
    Ptr<cuda::StereoBM> d_algs[2];
    Ptr<Stream> streams[2];
};

StereoMultiGpuStream::StereoMultiGpuStream()
{
    cuda::setDevice(0);
    d_algs[0] = cuda::createStereoBM(256);
    streams[0] = makePtr<Stream>();

    cuda::setDevice(1);
    d_algs[1] = cuda::createStereoBM(256);
    streams[1] = makePtr<Stream>();
}

StereoMultiGpuStream::~StereoMultiGpuStream()
{
    cuda::setDevice(0);
    d_leftFrames[0].release();
    d_rightFrames[0].release();
    d_disparities[0].release();
    d_algs[0].release();
    streams[0].release();

    cuda::setDevice(1);
    d_leftFrames[1].release();
    d_rightFrames[1].release();
    d_disparities[1].release();
    d_algs[1].release();
    streams[1].release();
}

void StereoMultiGpuStream::compute(const HostMem& leftFrame, const HostMem& rightFrame, HostMem& disparity)
{
    disparity.create(leftFrame.size(), CV_8UC1);

    // Split input data onto two parts for each GPUs.
    // We add small border for each part,
    // because original algorithm doesn't calculate disparity on image borders.
    // With such padding we will get output in the middle of final result.

    Mat leftFrameHdr = leftFrame.createMatHeader();
    Mat rightFrameHdr = rightFrame.createMatHeader();
    Mat disparityHdr = disparity.createMatHeader();
    Mat disparityPart0 = disparityHdr.rowRange(0, leftFrame.rows / 2);
    Mat disparityPart1 = disparityHdr.rowRange(leftFrame.rows / 2, leftFrame.rows);

    cuda::setDevice(0);
    d_leftFrames[0].upload(leftFrameHdr.rowRange(0, leftFrame.rows / 2 + 32), *streams[0]);
    d_rightFrames[0].upload(rightFrameHdr.rowRange(0, leftFrame.rows / 2 + 32), *streams[0]);
    d_algs[0]->compute(d_leftFrames[0], d_rightFrames[0], d_disparities[0], *streams[0]);
    d_disparities[0].rowRange(0, leftFrame.rows / 2).download(disparityPart0, *streams[0]);

    cuda::setDevice(1);
    d_leftFrames[1].upload(leftFrameHdr.rowRange(leftFrame.rows / 2 - 32, leftFrame.rows), *streams[1]);
    d_rightFrames[1].upload(rightFrameHdr.rowRange(leftFrame.rows / 2 - 32, leftFrame.rows), *streams[1]);
    d_algs[1]->compute(d_leftFrames[1], d_rightFrames[1], d_disparities[1], *streams[1]);
    d_disparities[1].rowRange(32, d_disparities[1].rows).download(disparityPart1, *streams[1]);

    cuda::setDevice(0);
    streams[0]->waitForCompletion();

    cuda::setDevice(1);
    streams[1]->waitForCompletion();
}

///////////////////////////////////////////////////////////
// main

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        cerr << "Usage: stereo_multi_gpu <left_video> <right_video>" << endl;
        return -1;
    }

    const int numDevices = getCudaEnabledDeviceCount();
    if (numDevices != 2)
    {
        cerr << "Two GPUs are required" << endl;
        return -1;
    }

    for (int i = 0; i < numDevices; ++i)
    {
        DeviceInfo devInfo(i);
        if (!devInfo.isCompatible())
        {
            cerr << "CUDA module was't built for GPU #" << i << " ("
                 << devInfo.name() << ", CC " << devInfo.majorVersion()
                 << devInfo.minorVersion() << endl;
            return -1;
        }

        printShortCudaDeviceInfo(i);
    }

    VideoCapture leftVideo(argv[1]);
    VideoCapture rightVideo(argv[2]);

    if (!leftVideo.isOpened())
    {
         cerr << "Can't open " << argv[1] << " video file" << endl;
         return -1;
    }

    if (!rightVideo.isOpened())
    {
         cerr << "Can't open " << argv[2] << " video file" << endl;
         return -1;
    }

    cout << endl;
    cout << "This sample demonstrates working on one piece of data using two GPUs." << endl;
    cout << "It splits input into two parts and processes them separately on different GPUs." << endl;
    cout << endl;

    Mat leftFrame, rightFrame;
    HostMem leftGrayFrame, rightGrayFrame;

    StereoSingleGpu gpu0Alg(0);
    StereoSingleGpu gpu1Alg(1);
    StereoMultiGpuThread multiThreadAlg;
    StereoMultiGpuStream multiStreamAlg;

    Mat disparityGpu0;
    Mat disparityGpu1;
    Mat disparityMultiThread;
    HostMem disparityMultiStream;

    Mat disparityGpu0Show;
    Mat disparityGpu1Show;
    Mat disparityMultiThreadShow;
    Mat disparityMultiStreamShow;

    TickMeter tm;

    cout << "-------------------------------------------------------------------" << endl;
    cout << "| Frame | GPU 0 ms | GPU 1 ms | Multi Thread ms | Multi Stream ms |" << endl;
    cout << "-------------------------------------------------------------------" << endl;

    for (int i = 0;; ++i)
    {
        leftVideo >> leftFrame;
        rightVideo >> rightFrame;

        if (leftFrame.empty() || rightFrame.empty())
            break;

        if (leftFrame.size() != rightFrame.size())
        {
            cerr << "Frames have different sizes" << endl;
            return -1;
        }

        leftGrayFrame.create(leftFrame.size(), CV_8UC1);
        rightGrayFrame.create(leftFrame.size(), CV_8UC1);

        cvtColor(leftFrame, leftGrayFrame.createMatHeader(), COLOR_BGR2GRAY);
        cvtColor(rightFrame, rightGrayFrame.createMatHeader(), COLOR_BGR2GRAY);

        tm.reset(); tm.start();
        gpu0Alg.compute(leftGrayFrame.createMatHeader(), rightGrayFrame.createMatHeader(),
                        disparityGpu0);
        tm.stop();

        const double gpu0Time = tm.getTimeMilli();

        tm.reset(); tm.start();
        gpu1Alg.compute(leftGrayFrame.createMatHeader(), rightGrayFrame.createMatHeader(),
                        disparityGpu1);
        tm.stop();

        const double gpu1Time = tm.getTimeMilli();

        tm.reset(); tm.start();
        multiThreadAlg.compute(leftGrayFrame.createMatHeader(), rightGrayFrame.createMatHeader(),
                               disparityMultiThread);
        tm.stop();

        const double multiThreadTime = tm.getTimeMilli();

        tm.reset(); tm.start();
        multiStreamAlg.compute(leftGrayFrame, rightGrayFrame, disparityMultiStream);
        tm.stop();

        const double multiStreamTime = tm.getTimeMilli();

        cout << "| " << setw(5) << i << " | "
             << setw(8) << setprecision(1) << fixed << gpu0Time << " | "
             << setw(8) << setprecision(1) << fixed << gpu1Time << " | "
             << setw(15) << setprecision(1) << fixed << multiThreadTime << " | "
             << setw(15) << setprecision(1) << fixed << multiStreamTime << " |" << endl;

        resize(disparityGpu0, disparityGpu0Show, Size(1024, 768), 0, 0, INTER_AREA);
        resize(disparityGpu1, disparityGpu1Show, Size(1024, 768), 0, 0, INTER_AREA);
        resize(disparityMultiThread, disparityMultiThreadShow, Size(1024, 768), 0, 0, INTER_AREA);
        resize(disparityMultiStream.createMatHeader(), disparityMultiStreamShow, Size(1024, 768), 0, 0, INTER_AREA);

        imshow("disparityGpu0", disparityGpu0Show);
        imshow("disparityGpu1", disparityGpu1Show);
        imshow("disparityMultiThread", disparityMultiThreadShow);
        imshow("disparityMultiStream", disparityMultiStreamShow);

        const int key = waitKey(30) & 0xff;
        if (key == 27)
            break;
    }

    cout << "-------------------------------------------------------------------" << endl;

    return 0;
}
