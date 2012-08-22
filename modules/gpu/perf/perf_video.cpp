#include "perf_precomp.hpp"

using namespace std;
using namespace testing;

namespace cv
{
    template<> void Ptr<CvBGStatModel>::delete_obj()
    {
        cvReleaseBGStatModel(&obj);
    }
}

namespace {

//////////////////////////////////////////////////////
// BroxOpticalFlow

typedef pair<string, string> pair_string;

DEF_PARAM_TEST_1(ImagePair, pair_string);

PERF_TEST_P(ImagePair, Video_BroxOpticalFlow, Values<pair_string>(make_pair("gpu/opticalflow/frame0.png", "gpu/opticalflow/frame1.png")))
{
    declare.time(10);

    cv::Mat frame0 = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage(GetParam().second, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    frame0.convertTo(frame0, CV_32FC1, 1.0 / 255.0);
    frame1.convertTo(frame1, CV_32FC1, 1.0 / 255.0);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_frame0(frame0);
        cv::gpu::GpuMat d_frame1(frame1);
        cv::gpu::GpuMat d_u;
        cv::gpu::GpuMat d_v;

        cv::gpu::BroxOpticalFlow d_flow(0.197f /*alpha*/, 50.0f /*gamma*/, 0.8f /*scale_factor*/,
                                        10 /*inner_iterations*/, 77 /*outer_iterations*/, 10 /*solver_iterations*/);

        d_flow(d_frame0, d_frame1, d_u, d_v);

        TEST_CYCLE()
        {
            d_flow(d_frame0, d_frame1, d_u, d_v);
        }
    }
    else
    {
        FAIL();
    }
}

//////////////////////////////////////////////////////
// InterpolateFrames

PERF_TEST_P(ImagePair, Video_InterpolateFrames, Values<pair_string>(make_pair("gpu/opticalflow/frame0.png", "gpu/opticalflow/frame1.png")))
{
    cv::Mat frame0 = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage(GetParam().second, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    frame0.convertTo(frame0, CV_32FC1, 1.0 / 255.0);
    frame1.convertTo(frame1, CV_32FC1, 1.0 / 255.0);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_frame0(frame0);
        cv::gpu::GpuMat d_frame1(frame1);
        cv::gpu::GpuMat d_fu, d_fv;
        cv::gpu::GpuMat d_bu, d_bv;

        cv::gpu::BroxOpticalFlow d_flow(0.197f /*alpha*/, 50.0f /*gamma*/, 0.8f /*scale_factor*/,
                                        10 /*inner_iterations*/, 77 /*outer_iterations*/, 10 /*solver_iterations*/);

        d_flow(d_frame0, d_frame1, d_fu, d_fv);
        d_flow(d_frame1, d_frame0, d_bu, d_bv);

        cv::gpu::GpuMat d_newFrame;
        cv::gpu::GpuMat d_buf;

        cv::gpu::interpolateFrames(d_frame0, d_frame1, d_fu, d_fv, d_bu, d_bv, 0.5f, d_newFrame, d_buf);

        TEST_CYCLE()
        {
            cv::gpu::interpolateFrames(d_frame0, d_frame1, d_fu, d_fv, d_bu, d_bv, 0.5f, d_newFrame, d_buf);
        }
    }
    else
    {
        FAIL();
    }
}

//////////////////////////////////////////////////////
// CreateOpticalFlowNeedleMap

PERF_TEST_P(ImagePair, Video_CreateOpticalFlowNeedleMap, Values<pair_string>(make_pair("gpu/opticalflow/frame0.png", "gpu/opticalflow/frame1.png")))
{
    cv::Mat frame0 = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage(GetParam().second, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    frame0.convertTo(frame0, CV_32FC1, 1.0 / 255.0);
    frame1.convertTo(frame1, CV_32FC1, 1.0 / 255.0);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_frame0(frame0);
        cv::gpu::GpuMat d_frame1(frame1);
        cv::gpu::GpuMat d_u;
        cv::gpu::GpuMat d_v;

        cv::gpu::BroxOpticalFlow d_flow(0.197f /*alpha*/, 50.0f /*gamma*/, 0.8f /*scale_factor*/,
                                        10 /*inner_iterations*/, 77 /*outer_iterations*/, 10 /*solver_iterations*/);

        d_flow(d_frame0, d_frame1, d_u, d_v);

        cv::gpu::GpuMat d_vertex, d_colors;

        cv::gpu::createOpticalFlowNeedleMap(d_u, d_v, d_vertex, d_colors);

        TEST_CYCLE()
        {
            cv::gpu::createOpticalFlowNeedleMap(d_u, d_v, d_vertex, d_colors);
        }
    }
    else
    {
        FAIL();
    }
}

//////////////////////////////////////////////////////
// GoodFeaturesToTrack

DEF_PARAM_TEST(Image_MinDistance, string, double);

PERF_TEST_P(Image_MinDistance, Video_GoodFeaturesToTrack, Combine(Values<string>("gpu/perf/aloe.jpg"), Values(0.0, 3.0)))
{
    string fileName = GET_PARAM(0);
    double minDistance = GET_PARAM(1);

    cv::Mat image = readImage(fileName, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    if (runOnGpu)
    {
        cv::gpu::GoodFeaturesToTrackDetector_GPU d_detector(8000, 0.01, minDistance);

        cv::gpu::GpuMat d_image(image);
        cv::gpu::GpuMat d_pts;

        d_detector(d_image, d_pts);

        TEST_CYCLE()
        {
            d_detector(d_image, d_pts);
        }
    }
    else
    {
        cv::Mat pts;

        cv::goodFeaturesToTrack(image, pts, 8000, 0.01, minDistance);

        TEST_CYCLE()
        {
            cv::goodFeaturesToTrack(image, pts, 8000, 0.01, minDistance);
        }
    }
}

//////////////////////////////////////////////////////
// PyrLKOpticalFlowSparse

DEF_PARAM_TEST(ImagePair_Gray_NPts_WinSz_Levels_Iters, pair_string, bool, int, int, int, int);

PERF_TEST_P(ImagePair_Gray_NPts_WinSz_Levels_Iters, Video_PyrLKOpticalFlowSparse, Combine(
    Values<pair_string>(make_pair("gpu/opticalflow/frame0.png", "gpu/opticalflow/frame1.png")),
    Bool(),
    Values(1000, 2000, 4000, 8000),
    Values(9, 13, 17, 21),
    Values(1, 2, 3),
    Values(1, 10, 30)))
{
    declare.time(20.0);

    pair_string imagePair = GET_PARAM(0);
    bool useGray = GET_PARAM(1);
    int points = GET_PARAM(2);
    int winSize = GET_PARAM(3);
    int levels = GET_PARAM(4);
    int iters = GET_PARAM(5);

    cv::Mat frame0 = readImage(imagePair.first, useGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage(imagePair.second, useGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    ASSERT_FALSE(frame1.empty());

    cv::Mat gray_frame;
    if (useGray)
        gray_frame = frame0;
    else
        cv::cvtColor(frame0, gray_frame, cv::COLOR_BGR2GRAY);

    cv::Mat pts;
    cv::goodFeaturesToTrack(gray_frame, pts, points, 0.01, 0.0);

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_pts(pts);

        cv::gpu::PyrLKOpticalFlow d_pyrLK;
        d_pyrLK.winSize = cv::Size(winSize, winSize);
        d_pyrLK.maxLevel = levels - 1;
        d_pyrLK.iters = iters;

        cv::gpu::GpuMat d_frame0(frame0);
        cv::gpu::GpuMat d_frame1(frame1);
        cv::gpu::GpuMat d_nextPts;
        cv::gpu::GpuMat d_status;

        d_pyrLK.sparse(d_frame0, d_frame1, d_pts, d_nextPts, d_status);

        TEST_CYCLE()
        {
            d_pyrLK.sparse(d_frame0, d_frame1, d_pts, d_nextPts, d_status);
        }
    }
    else
    {
        cv::Mat nextPts;
        cv::Mat status;

        cv::calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts, status, cv::noArray(),
                                 cv::Size(winSize, winSize), levels - 1,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, iters, 0.01));

        TEST_CYCLE()
        {
            cv::calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts, status, cv::noArray(),
                                     cv::Size(winSize, winSize), levels - 1,
                                     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, iters, 0.01));
        }
    }
}

//////////////////////////////////////////////////////
// PyrLKOpticalFlowDense

DEF_PARAM_TEST(ImagePair_WinSz_Levels_Iters, pair_string, int, int, int);

PERF_TEST_P(ImagePair_WinSz_Levels_Iters, Video_PyrLKOpticalFlowDense, Combine(
    Values<pair_string>(make_pair("gpu/opticalflow/frame0.png", "gpu/opticalflow/frame1.png")),
    Values(3, 5, 7, 9, 13, 17, 21),
    Values(1, 2, 3),
    Values(1, 10)))
{
    declare.time(30);

    pair_string imagePair = GET_PARAM(0);
    int winSize = GET_PARAM(1);
    int levels = GET_PARAM(2);
    int iters = GET_PARAM(3);

    cv::Mat frame0 = readImage(imagePair.first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage(imagePair.second, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_frame0(frame0);
        cv::gpu::GpuMat d_frame1(frame1);
        cv::gpu::GpuMat d_u;
        cv::gpu::GpuMat d_v;

        cv::gpu::PyrLKOpticalFlow d_pyrLK;
        d_pyrLK.winSize = cv::Size(winSize, winSize);
        d_pyrLK.maxLevel = levels - 1;
        d_pyrLK.iters = iters;

        d_pyrLK.dense(d_frame0, d_frame1, d_u, d_v);

        TEST_CYCLE()
        {
            d_pyrLK.dense(d_frame0, d_frame1, d_u, d_v);
        }
    }
    else
    {
        FAIL();
    }
}

//////////////////////////////////////////////////////
// FarnebackOpticalFlow

PERF_TEST_P(ImagePair, Video_FarnebackOpticalFlow, Values<pair_string>(make_pair("gpu/opticalflow/frame0.png", "gpu/opticalflow/frame1.png")))
{
    declare.time(10);

    cv::Mat frame0 = readImage(GetParam().first, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage(GetParam().second, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    int numLevels = 5;
    double pyrScale = 0.5;
    int winSize = 13;
    int numIters = 10;
    int polyN = 5;
    double polySigma = 1.1;
    int flags = 0;

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_frame0(frame0);
        cv::gpu::GpuMat d_frame1(frame1);
        cv::gpu::GpuMat d_u;
        cv::gpu::GpuMat d_v;

        cv::gpu::FarnebackOpticalFlow d_farneback;
        d_farneback.numLevels = numLevels;
        d_farneback.pyrScale = pyrScale;
        d_farneback.winSize = winSize;
        d_farneback.numIters = numIters;
        d_farneback.polyN = polyN;
        d_farneback.polySigma = polySigma;
        d_farneback.flags = flags;

        d_farneback(d_frame0, d_frame1, d_u, d_v);

        TEST_CYCLE()
        {
            d_farneback(d_frame0, d_frame1, d_u, d_v);
        }
    }
    else
    {
        cv::Mat flow;

        cv::calcOpticalFlowFarneback(frame0, frame1, flow, pyrScale, numLevels, winSize, numIters, polyN, polySigma, flags);

        TEST_CYCLE()
        {
            cv::calcOpticalFlowFarneback(frame0, frame1, flow, pyrScale, numLevels, winSize, numIters, polyN, polySigma, flags);
        }
    }
}

//////////////////////////////////////////////////////
// FGDStatModel

DEF_PARAM_TEST_1(Video, string);

PERF_TEST_P(Video, Video_FGDStatModel, Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"))
{
    declare.time(60);

    string inputFile = perf::TestBase::getDataPath(GetParam());

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;
    cap >> frame;
    ASSERT_FALSE(frame.empty());

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_frame(frame);

        cv::gpu::FGDStatModel d_model(4);
        d_model.create(d_frame);

        for (int i = 0; i < 10; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            d_frame.upload(frame);

            startTimer(); next();
            d_model.update(d_frame);
            stopTimer();
        }
    }
    else
    {
        IplImage ipl_frame = frame;
        cv::Ptr<CvBGStatModel> model(cvCreateFGDStatModel(&ipl_frame));

        for (int i = 0; i < 10; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            ipl_frame = frame;

            startTimer(); next();
            cvUpdateBGStatModel(&ipl_frame, model);
            stopTimer();
        }
    }
}

//////////////////////////////////////////////////////
// MOG

DEF_PARAM_TEST(Video_Cn_LearningRate, string, int, double);

PERF_TEST_P(Video_Cn_LearningRate, Video_MOG, Combine(Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"), Values(1, 3, 4), Values(0.0, 0.01)))
{
    string inputFile = perf::TestBase::getDataPath(GET_PARAM(0));
    int cn = GET_PARAM(1);
    float learningRate = static_cast<float>(GET_PARAM(2));

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;

    cap >> frame;
    ASSERT_FALSE(frame.empty());

    if (cn != 3)
    {
        cv::Mat temp;
        if (cn == 1)
            cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
        else
            cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
        cv::swap(temp, frame);
    }

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_frame(frame);
        cv::gpu::MOG_GPU d_mog;
        cv::gpu::GpuMat d_foreground;

        d_mog(d_frame, d_foreground, learningRate);

        for (int i = 0; i < 10; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            if (cn != 3)
            {
                cv::Mat temp;
                if (cn == 1)
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
                else
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
                cv::swap(temp, frame);
            }

            d_frame.upload(frame);

            startTimer(); next();
            d_mog(d_frame, d_foreground, learningRate);
            stopTimer();
        }
    }
    else
    {
        cv::BackgroundSubtractorMOG mog;
        cv::Mat foreground;

        mog(frame, foreground, learningRate);

        for (int i = 0; i < 10; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            if (cn != 3)
            {
                cv::Mat temp;
                if (cn == 1)
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
                else
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
                cv::swap(temp, frame);
            }

            startTimer(); next();
            mog(frame, foreground, learningRate);
            stopTimer();
        }
    }
}

//////////////////////////////////////////////////////
// MOG2

DEF_PARAM_TEST(Video_Cn, string, int);

PERF_TEST_P(Video_Cn, Video_MOG2, Combine(Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"), Values(1, 3, 4)))
{
    string inputFile = perf::TestBase::getDataPath(GET_PARAM(0));
    int cn = GET_PARAM(1);

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;

    cap >> frame;
    ASSERT_FALSE(frame.empty());

    if (cn != 3)
    {
        cv::Mat temp;
        if (cn == 1)
            cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
        else
            cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
        cv::swap(temp, frame);
    }

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_frame(frame);
        cv::gpu::MOG2_GPU d_mog2;
        cv::gpu::GpuMat d_foreground;

        d_mog2(d_frame, d_foreground);

        for (int i = 0; i < 10; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            if (cn != 3)
            {
                cv::Mat temp;
                if (cn == 1)
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
                else
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
                cv::swap(temp, frame);
            }

            d_frame.upload(frame);

            startTimer(); next();
            d_mog2(d_frame, d_foreground);
            stopTimer();
        }
    }
    else
    {
        cv::BackgroundSubtractorMOG2 mog2;
        cv::Mat foreground;

        mog2(frame, foreground);

        for (int i = 0; i < 10; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            if (cn != 3)
            {
                cv::Mat temp;
                if (cn == 1)
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
                else
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
                cv::swap(temp, frame);
            }

            startTimer(); next();
            mog2(frame, foreground);
            stopTimer();
        }
    }
}

//////////////////////////////////////////////////////
// MOG2GetBackgroundImage

PERF_TEST_P(Video_Cn, Video_MOG2GetBackgroundImage, Combine(Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"), Values(1, 3, 4)))
{
    string inputFile = perf::TestBase::getDataPath(GET_PARAM(0));
    int cn = GET_PARAM(1);

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_frame;
        cv::gpu::MOG2_GPU d_mog2;
        cv::gpu::GpuMat d_foreground;

        for (int i = 0; i < 10; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            if (cn != 3)
            {
                cv::Mat temp;
                if (cn == 1)
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
                else
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
                cv::swap(temp, frame);
            }

            d_frame.upload(frame);

            d_mog2(d_frame, d_foreground);
        }

        cv::gpu::GpuMat d_background;
        d_mog2.getBackgroundImage(d_background);

        TEST_CYCLE()
        {
            d_mog2.getBackgroundImage(d_background);
        }
    }
    else
    {
        cv::BackgroundSubtractorMOG2 mog2;
        cv::Mat foreground;

        for (int i = 0; i < 10; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            if (cn != 3)
            {
                cv::Mat temp;
                if (cn == 1)
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
                else
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
                cv::swap(temp, frame);
            }

            mog2(frame, foreground);
        }

        cv::Mat background;
        mog2.getBackgroundImage(background);

        TEST_CYCLE()
        {
            mog2.getBackgroundImage(background);
        }
    }
}

//////////////////////////////////////////////////////
// VIBE

PERF_TEST_P(Video_Cn, Video_VIBE, Combine(Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"), Values(1, 3, 4)))
{
    string inputFile = perf::TestBase::getDataPath(GET_PARAM(0));
    int cn = GET_PARAM(1);

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;
    cap >> frame;
    ASSERT_FALSE(frame.empty());

    if (cn != 3)
    {
        cv::Mat temp;
        if (cn == 1)
            cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
        else
            cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
        cv::swap(temp, frame);
    }

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_frame(frame);
        cv::gpu::VIBE_GPU d_vibe;
        cv::gpu::GpuMat d_foreground;

        d_vibe(d_frame, d_foreground);

        for (int i = 0; i < 10; ++i)
        {
            cap >> frame;
            ASSERT_FALSE(frame.empty());

            if (cn != 3)
            {
                cv::Mat temp;
                if (cn == 1)
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
                else
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
                cv::swap(temp, frame);
            }

            d_frame.upload(frame);

            startTimer(); next();
            d_vibe(d_frame, d_foreground);
            stopTimer();
        }
    }
    else
    {
        FAIL();
    }
}

//////////////////////////////////////////////////////
// GMG

DEF_PARAM_TEST(Video_Cn_MaxFeatures, string, int, int);

PERF_TEST_P(Video_Cn_MaxFeatures, Video_GMG, Combine(Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"), Values(1, 3, 4), Values(20, 40, 60)))
{
    std::string inputFile = perf::TestBase::getDataPath(GET_PARAM(0));
    int cn = GET_PARAM(1);
    int maxFeatures = GET_PARAM(2);

    cv::VideoCapture cap(inputFile);
    ASSERT_TRUE(cap.isOpened());

    cv::Mat frame;
    cap >> frame;
    ASSERT_FALSE(frame.empty());

    if (cn != 3)
    {
        cv::Mat temp;
        if (cn == 1)
            cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
        else
            cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
        cv::swap(temp, frame);
    }

    if (runOnGpu)
    {
        cv::gpu::GpuMat d_frame(frame);
        cv::gpu::GpuMat d_fgmask;

        cv::gpu::GMG_GPU d_gmg;
        d_gmg.maxFeatures = maxFeatures;

        d_gmg(d_frame, d_fgmask);

        for (int i = 0; i < 150; ++i)
        {
            cap >> frame;
            if (frame.empty())
            {
                cap.open(inputFile);
                cap >> frame;
            }

            if (cn != 3)
            {
                cv::Mat temp;
                if (cn == 1)
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
                else
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
                cv::swap(temp, frame);
            }

            d_frame.upload(frame);

            startTimer(); next();
            d_gmg(d_frame, d_fgmask);
            stopTimer();
        }
    }
    else
    {
        cv::Mat fgmask;
        cv::Mat zeros(frame.size(), CV_8UC1, cv::Scalar::all(0));

        cv::BackgroundSubtractorGMG gmg;
        gmg.set("maxFeatures", maxFeatures);
        gmg.initialize(frame.size(), 0.0, 255.0);

        gmg(frame, fgmask);

        for (int i = 0; i < 150; ++i)
        {
            cap >> frame;
            if (frame.empty())
            {
                cap.open(inputFile);
                cap >> frame;
            }

            if (cn != 3)
            {
                cv::Mat temp;
                if (cn == 1)
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2GRAY);
                else
                    cv::cvtColor(frame, temp, cv::COLOR_BGR2BGRA);
                cv::swap(temp, frame);
            }

            startTimer(); next();
            gmg(frame, fgmask);
            stopTimer();
        }
    }
}

//////////////////////////////////////////////////////
// VideoWriter

PERF_TEST_P(Video, Video_VideoWriter, Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"))
{
    declare.time(30);

    string inputFile = perf::TestBase::getDataPath(GetParam());
    string outputFile = cv::tempfile(".avi");

    const double FPS = 25.0;

    cv::VideoCapture reader(inputFile);
    ASSERT_TRUE( reader.isOpened() );

    cv::Mat frame;

    if (runOnGpu)
    {
        cv::gpu::VideoWriter_GPU d_writer;

        cv::gpu::GpuMat d_frame;

        for (int i = 0; i < 10; ++i)
        {
            reader >> frame;
            ASSERT_FALSE(frame.empty());

            d_frame.upload(frame);

            if (!d_writer.isOpened())
                d_writer.open(outputFile, frame.size(), FPS);

            startTimer(); next();
            d_writer.write(d_frame);
            stopTimer();
        }
    }
    else
    {
        cv::VideoWriter writer;

        for (int i = 0; i < 10; ++i)
        {
            reader >> frame;
            ASSERT_FALSE(frame.empty());

            if (!writer.isOpened())
                writer.open(outputFile, CV_FOURCC('X', 'V', 'I', 'D'), FPS, frame.size());

            startTimer(); next();
            writer.write(frame);
            stopTimer();
        }
    }
}

//////////////////////////////////////////////////////
// VideoReader

PERF_TEST_P(Video, Video_VideoReader, Values("gpu/video/768x576.avi", "gpu/video/1920x1080.avi"))
{
    declare.time(20);

    string inputFile = perf::TestBase::getDataPath(GetParam());

    if (runOnGpu)
    {
        cv::gpu::VideoReader_GPU d_reader(inputFile);
        ASSERT_TRUE( d_reader.isOpened() );

        cv::gpu::GpuMat d_frame;

        d_reader.read(d_frame);

        TEST_CYCLE_N(10)
        {
            d_reader.read(d_frame);
        }
    }
    else
    {
        cv::VideoCapture reader(inputFile);
        ASSERT_TRUE( reader.isOpened() );

        cv::Mat frame;

        reader >> frame;

        TEST_CYCLE_N(10)
        {
            reader >> frame;
        }
    }
}

} // namespace
