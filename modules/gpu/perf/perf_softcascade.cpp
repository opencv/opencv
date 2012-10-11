#include "perf_precomp.hpp"

#define GPU_PERF_TEST_P(fixture, name, params)  \
    class fixture##_##name : public fixture {\
     public:\
      fixture##_##name() {}\
     protected:\
             virtual void __cpu();\
        virtual void __gpu();\
      virtual void PerfTestBody();\
    };\
    TEST_P(fixture##_##name, name /*perf*/){ RunPerfTestBody(); if (runOnGpu) __gpu(); else __cpu();}\
    INSTANTIATE_TEST_CASE_P(/*none*/, fixture##_##name, params);\
    void fixture##_##name::PerfTestBody()

#define RUN_CPU(fixture, name)\
    void fixture##_##name::__cpu()

#define RUN_GPU(fixture, name)\
    void fixture##_##name::__gpu()

#define FAIL_NO_CPU(fixture, name)\
void fixture##_##name::__cpu() { FAIL() << "No such CPU implementation analogy";}


typedef std::tr1::tuple<std::string, std::string> fixture_t;
typedef perf::TestBaseWithParam<fixture_t> SoftCascadeTest;

GPU_PERF_TEST_P(SoftCascadeTest, detect,
    testing::Combine(
        testing::Values(std::string("cv/cascadeandhog/sc_cvpr_2012_to_opencv.xml")),
        testing::Values(std::string("cv/cascadeandhog/bahnhof/image_00000000_0.png"))))
{ }

RUN_GPU(SoftCascadeTest, detect)
{
    cv::Mat cpu = readImage (GET_PARAM(1));
    ASSERT_FALSE(cpu.empty());
    cv::gpu::GpuMat colored(cpu);

    cv::gpu::SoftCascade cascade;
    ASSERT_TRUE(cascade.load(perf::TestBase::getDataPath(GET_PARAM(0))));

    cv::gpu::GpuMat objectBoxes(1, 16384, CV_8UC1), rois(cascade.getRoiSize(), CV_8UC1), trois;
    rois.setTo(1);
    cv::gpu::transpose(rois, trois);

    cv::gpu::GpuMat curr = objectBoxes;
    cascade.detectMultiScale(colored, trois, curr);

    TEST_CYCLE()
    {
        curr = objectBoxes;
        cascade.detectMultiScale(colored, trois, curr);
    }
}

RUN_CPU(SoftCascadeTest, detect)
{
    cv::Mat colored = readImage(GET_PARAM(1));
    ASSERT_FALSE(colored.empty());

    cv::SoftCascade cascade;
    ASSERT_TRUE(cascade.load(getDataPath(GET_PARAM(0))));

    std::vector<cv::Rect> rois;

    typedef cv::SoftCascade::Detection Detection;
    std::vector<Detection>objectBoxes;
    cascade.detectMultiScale(colored, rois, objectBoxes);

    TEST_CYCLE()
    {
        cascade.detectMultiScale(colored, rois, objectBoxes);
    }
}

static cv::Rect getFromTable(int idx)
{
    static const cv::Rect rois[] =
    {
        cv::Rect( 65,  20,  35, 80),
        cv::Rect( 95,  35,  45, 40),
        cv::Rect( 45,  35,  45, 40),
        cv::Rect( 25,  27,  50, 45),
        cv::Rect(100,  50,  45, 40),

        cv::Rect( 60,  30,  45, 40),
        cv::Rect( 40,  55,  50, 40),
        cv::Rect( 48,  37,  72, 80),
        cv::Rect( 48,  32,  85, 58),
        cv::Rect( 48,   0,  32, 27)
    };

    return rois[idx];
}

typedef std::tr1::tuple<std::string, std::string, int> roi_fixture_t;
typedef perf::TestBaseWithParam<roi_fixture_t> SoftCascadeTestRoi;

GPU_PERF_TEST_P(SoftCascadeTestRoi, detectInRoi,
    testing::Combine(
        testing::Values(std::string("cv/cascadeandhog/sc_cvpr_2012_to_opencv.xml")),
        testing::Values(std::string("cv/cascadeandhog/bahnhof/image_00000000_0.png")),
        testing::Range(0, 5)))
{}

RUN_GPU(SoftCascadeTestRoi, detectInRoi)
{
    cv::Mat cpu = readImage (GET_PARAM(1));
    ASSERT_FALSE(cpu.empty());
    cv::gpu::GpuMat colored(cpu);

    cv::gpu::SoftCascade cascade;
    ASSERT_TRUE(cascade.load(perf::TestBase::getDataPath(GET_PARAM(0))));

    cv::gpu::GpuMat objectBoxes(1, 16384 * 20, CV_8UC1), rois(cascade.getRoiSize(), CV_8UC1);
    rois.setTo(0);

    int nroi = GET_PARAM(2);
    cv::RNG rng;
    for (int i = 0; i < nroi; ++i)
    {
        cv::Rect r = getFromTable(rng(10));
        cv::gpu::GpuMat sub(rois, r);
        sub.setTo(1);
    }

    cv::gpu::GpuMat trois;
    cv::gpu::transpose(rois, trois);

    cv::gpu::GpuMat curr = objectBoxes;
    cascade.detectMultiScale(colored, trois, curr);

    TEST_CYCLE()
    {
        curr = objectBoxes;
        cascade.detectMultiScale(colored, trois, curr);
    }
}

FAIL_NO_CPU(SoftCascadeTestRoi, detectInRoi)


GPU_PERF_TEST_P(SoftCascadeTestRoi, detectEachRoi,
    testing::Combine(
        testing::Values(std::string("cv/cascadeandhog/sc_cvpr_2012_to_opencv.xml")),
        testing::Values(std::string("cv/cascadeandhog/bahnhof/image_00000000_0.png")),
        testing::Range(0, 10)))
{}

RUN_GPU(SoftCascadeTestRoi, detectEachRoi)
{
    cv::Mat cpu = readImage (GET_PARAM(1));
    ASSERT_FALSE(cpu.empty());
    cv::gpu::GpuMat colored(cpu);

    cv::gpu::SoftCascade cascade;
    ASSERT_TRUE(cascade.load(perf::TestBase::getDataPath(GET_PARAM(0))));

    cv::gpu::GpuMat objectBoxes(1, 16384 * 20, CV_8UC1), rois(cascade.getRoiSize(), CV_8UC1);
    rois.setTo(0);

    int idx = GET_PARAM(2);
    cv::Rect r = getFromTable(idx);
    cv::gpu::GpuMat sub(rois, r);
    sub.setTo(1);

    cv::gpu::GpuMat curr = objectBoxes;
    cv::gpu::GpuMat trois;
    cv::gpu::transpose(rois, trois);

    cascade.detectMultiScale(colored, trois, curr);

    TEST_CYCLE()
    {
        curr = objectBoxes;
        cascade.detectMultiScale(colored, trois, curr);
    }
}

FAIL_NO_CPU(SoftCascadeTestRoi, detectEachRoi)