#include <iostream>
#include <iomanip>
#include <stdexcept>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "utility_lib/utility_lib.h"

#define PARAM_OFFSET "--offset"

using namespace std;
using namespace cv;
using namespace cv::gpu;

class App : public BaseApp
{
public:
    App() : use_gpu(true), match_confidence(10), the_same_video_offset(1) {}

protected:
    void process();
    bool parseCmdArgs(int& i, int argc, const char* argv[]);
    bool processKey(int key);
    void printHelp();

private:
    bool use_gpu;
    int match_confidence;
    int the_same_video_offset;
    Ptr<PairFrameSource> source_;
};

void App::process()
{
    if (sources.size() == 1)
        source_ = PairFrameSource::get(sources[0], the_same_video_offset);
    else if (sources.size() == 2)
        source_ = PairFrameSource::get(sources[0], sources[1]);
    else
    {
        cout << "Loading default images..." << endl;
        sources.resize(2);
        sources[0] = new ImageSource("data/matching/t34mA.JPG");
        sources[1] = new ImageSource("data/matching/t34mB.JPG");

        source_ = PairFrameSource::get(sources[0], sources[1]);
    }

    cout << "\nControls:" << endl;
    cout << "  space - change CPU/GPU mode" << endl;
    cout << "  a/s - increase/decrease match confidence\n" << endl;

    Mat h_img1, h_img2, h_img1_gray, h_img2_gray;
    GpuMat d_img1_gray, d_img2_gray;

    ORB orb_cpu(1000);
    ORB_GPU orb_gpu(1000);

    vector<KeyPoint> keypoints1_cpu, keypoints2_cpu;
    Mat descriptors1_cpu, descriptors2_cpu;
    GpuMat keypoints1_gpu, keypoints2_gpu;
    GpuMat descriptors1_gpu, descriptors2_gpu;

    BFMatcher matcher_cpu(NORM_HAMMING);
    BruteForceMatcher_GPU<Hamming> matcher_gpu;
    GpuMat trainIdx, distance, allDist;
    vector< vector<DMatch> > matches;
    vector<DMatch> good_matches;

    double total_fps = 0;

    while (!exited)
    {
        int64 start = getTickCount();

        source_->next(h_img1, h_img2);

        makeGray(h_img1, h_img1_gray);
        makeGray(h_img2, h_img2_gray);

        if (use_gpu)
        {
            d_img1_gray.upload(h_img1_gray);
            d_img2_gray.upload(h_img2_gray);
        }

        int64 proc_start = getTickCount();
        
        int64 orb_start = getTickCount();

        if (use_gpu)
        {
            orb_gpu(d_img1_gray, GpuMat(), keypoints1_gpu, descriptors1_gpu);
            orb_gpu(d_img2_gray, GpuMat(), keypoints2_gpu, descriptors2_gpu);
        }
        else
        {
            orb_cpu(h_img1_gray, Mat(), keypoints1_cpu, descriptors1_cpu);
            orb_cpu(h_img2_gray, Mat(), keypoints2_cpu, descriptors2_cpu);
        }

        double orb_fps = getTickFrequency()  / (getTickCount() - orb_start);
        
        int64 match_start = getTickCount();

        if (use_gpu)
        {
            matcher_gpu.knnMatchSingle(descriptors1_gpu, descriptors2_gpu, trainIdx, distance, allDist, 2);
        }
        else
        {
            matcher_cpu.knnMatch(descriptors1_cpu, descriptors2_cpu, matches, 2);
        }

        double match_fps = getTickFrequency()  / (getTickCount() - match_start);

        if (use_gpu)
        {
            matcher_gpu.knnMatchDownload(trainIdx, distance, matches);
        }

        good_matches.clear();
        good_matches.reserve(matches.size());

        for (size_t i = 0; i < matches.size(); ++i)
        {
            if (matches[i].size() < 2)
                continue;

            const DMatch &m1 = matches[i][0];
            const DMatch &m2 = matches[i][1];

            if (abs(m1.distance - m2.distance) > match_confidence)
                good_matches.push_back(m1);
        }

        double proc_fps = getTickFrequency()  / (getTickCount() - proc_start);

        if (use_gpu)
        {
            orb_gpu.downloadKeyPoints(keypoints1_gpu, keypoints1_cpu);
            orb_gpu.downloadKeyPoints(keypoints2_gpu, keypoints2_cpu);
        }

        Mat dst;
        drawMatches(h_img1, keypoints1_cpu, h_img2, keypoints2_cpu, good_matches, dst, Scalar(255, 0, 0, 255), Scalar(0, 0, 255, 255));

        stringstream msg; msg << "Total FPS : " << setprecision(4) << total_fps;
        printText(dst, msg.str(), 0);

        msg.str(""); msg << "Processing FPS : " << setprecision(4) << proc_fps;
        printText(dst, msg.str(), 1);

        msg.str(""); msg << "ORB FPS : " << setprecision(4) << orb_fps;
        printText(dst, msg.str(), 2);

        msg.str(""); msg << "Match FPS : " << setprecision(4) << match_fps;
        printText(dst, msg.str(), 3);

        printText(dst, use_gpu ? "Mode : GPU" : "Mode : CPU", 4);

        imshow("orb_demo", dst);
        processKey(waitKey(3));

        total_fps = getTickFrequency()  / (getTickCount() - proc_start);
    }
}

bool App::parseCmdArgs(int& i, int argc, const char* argv[])
{
    string arg(argv[i]);

    if (arg == PARAM_OFFSET)
    {
        ++i;

        if (i >= argc)
        {
            ostringstream msg;
            msg << "Missing value after " << PARAM_OFFSET;
            throw runtime_error(msg.str());
        }

        the_same_video_offset = atoi(argv[i]);
    }
    else
        return false;

    return true;
}

bool App::processKey(int key)
{
    if (BaseApp::processKey(key))
        return true;

    switch (toupper(key & 0xff))
    {
    case 32 /*space*/:
        use_gpu = !use_gpu;
        cout << "Use gpu = " << use_gpu << endl;
        break;

    case 'A':
        ++match_confidence;
        cout << "match_confidence = " << match_confidence << endl;
        break;

    case 'S':
        --match_confidence;
        match_confidence = max(match_confidence, 0);
        cout << "match_confidence = " << match_confidence << endl;
        break;

    default:
        return false;
    }

    return true;
}

void App::printHelp()
{
    cout << "This program demonstrates using ORB_GPU features detector, descriptor extractor and BruteForceMatcher_GPU" << endl;
    cout << "Usage: demo_surf <frames_source1> [<frames_source2>]" << endl;
    cout << '\t' << setw(15) << PARAM_OFFSET << " - set frames offset for the duplicate video source" << endl;
    BaseApp::printHelp();
}

RUN_APP(App)
