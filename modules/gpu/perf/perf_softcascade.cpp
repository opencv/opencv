#include "perf_precomp.hpp"

#define SC_PERF_TEST_P(fixture, name, params)  \
    class fixture##_##name : public fixture {\
     public:\
      fixture##_##name() {}\
     protected:\
        virtual void __cpu();\
        virtual void __gpu();\
      virtual void PerfTestBody();\
    };\
    TEST_P(fixture##_##name, name /*perf*/){ RunPerfTestBody(); }\
    INSTANTIATE_TEST_CASE_P(/*none*/, fixture##_##name, params);\
    void fixture##_##name::PerfTestBody() { if (PERF_RUN_GPU()) __gpu(); else __cpu(); }

#define RUN_CPU(fixture, name)\
    void fixture##_##name::__cpu()

#define RUN_GPU(fixture, name)\
    void fixture##_##name::__gpu()

#define NO_CPU(fixture, name)\
void fixture##_##name::__cpu() { FAIL() << "No such CPU implementation analogy";}

namespace {
    struct DetectionLess
    {
        bool operator()(const cv::gpu::SCascade::Detection& a,
            const cv::gpu::SCascade::Detection& b) const
        {
            if (a.x != b.x)      return a.x < b.x;
            else if (a.y != b.y) return a.y < b.y;
            else if (a.w != b.w) return a.w < b.w;
            else return a.h < b.h;
        }
    };

    cv::Mat sortDetections(cv::gpu::GpuMat& objects)
    {
        cv::Mat detections(objects);

        typedef cv::gpu::SCascade::Detection Detection;
        Detection* begin = (Detection*)(detections.ptr<char>(0));
        Detection* end = (Detection*)(detections.ptr<char>(0) + detections.cols);
        std::sort(begin, end, DetectionLess());

        return detections;
    }
}


typedef std::tr1::tuple<std::string, std::string> fixture_t;
typedef perf::TestBaseWithParam<fixture_t> SCascadeTest;

SC_PERF_TEST_P(SCascadeTest, detect,
    testing::Combine(
        testing::Values(std::string("cv/cascadeandhog/cascades/inria_caltech-17.01.2013.xml"),
                        std::string("cv/cascadeandhog/cascades/sc_cvpr_2012_to_opencv_new_format.xml")),
        testing::Values(std::string("cv/cascadeandhog/images/image_00000000_0.png"))))

RUN_GPU(SCascadeTest, detect)
{
    cv::Mat cpu = readImage (GET_PARAM(1));
    ASSERT_FALSE(cpu.empty());
    cv::gpu::GpuMat colored(cpu);

    cv::gpu::SCascade cascade;

    cv::FileStorage fs(perf::TestBase::getDataPath(GET_PARAM(0)), cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());

    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    cv::gpu::GpuMat objectBoxes(1, 10000 * sizeof(cv::gpu::SCascade::Detection), CV_8UC1), rois(colored.size(), CV_8UC1);
    rois.setTo(1);

    cascade.detect(colored, rois, objectBoxes);

    TEST_CYCLE()
    {
        cascade.detect(colored, rois, objectBoxes);
    }

    SANITY_CHECK(sortDetections(objectBoxes));
}

NO_CPU(SCascadeTest, detect)

static cv::Rect getFromTable(int idx)
{
    static const cv::Rect rois[] =
    {
        cv::Rect( 65 * 4,  20 * 4,  35 * 4, 80 * 4),
        cv::Rect( 95 * 4,  35 * 4,  45 * 4, 40 * 4),
        cv::Rect( 45 * 4,  35 * 4,  45 * 4, 40 * 4),
        cv::Rect( 25 * 4,  27 * 4,  50 * 4, 45 * 4),
        cv::Rect(100 * 4,  50 * 4,  45 * 4, 40 * 4),

        cv::Rect( 60 * 4,  30 * 4,  45 * 4, 40 * 4),
        cv::Rect( 40 * 4,  55 * 4,  50 * 4, 40 * 4),
        cv::Rect( 48 * 4,  37 * 4,  72 * 4, 80 * 4),
        cv::Rect( 48 * 4,  32 * 4,  85 * 4, 58 * 4),
        cv::Rect( 48 * 4,   0 * 4,  32 * 4, 27 * 4)
    };

    return rois[idx];
}

typedef std::tr1::tuple<std::string, std::string, int> roi_fixture_t;
typedef perf::TestBaseWithParam<roi_fixture_t> SCascadeTestRoi;

SC_PERF_TEST_P(SCascadeTestRoi, detectInRoi,
    testing::Combine(
        testing::Values(std::string("cv/cascadeandhog/cascades/inria_caltech-17.01.2013.xml"),
                        std::string("cv/cascadeandhog/cascades/sc_cvpr_2012_to_opencv_new_format.xml")),
        testing::Values(std::string("cv/cascadeandhog/images/image_00000000_0.png")),
        testing::Range(0, 5)))

RUN_GPU(SCascadeTestRoi, detectInRoi)
{
    cv::Mat cpu = readImage (GET_PARAM(1));
    ASSERT_FALSE(cpu.empty());
    cv::gpu::GpuMat colored(cpu);

    cv::gpu::SCascade cascade;

    cv::FileStorage fs(perf::TestBase::getDataPath(GET_PARAM(0)), cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());

    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    cv::gpu::GpuMat objectBoxes(1, 16384 * 20, CV_8UC1), rois(colored.size(), CV_8UC1);
    rois.setTo(0);

    int nroi = GET_PARAM(2);
    cv::RNG rng;
    for (int i = 0; i < nroi; ++i)
    {
        cv::Rect r = getFromTable(rng(10));
        cv::gpu::GpuMat sub(rois, r);
        sub.setTo(1);
    }

    cascade.detect(colored, rois, objectBoxes);

    TEST_CYCLE()
    {
        cascade.detect(colored, rois, objectBoxes);
    }

    SANITY_CHECK(sortDetections(objectBoxes));
}

NO_CPU(SCascadeTestRoi, detectInRoi)


SC_PERF_TEST_P(SCascadeTestRoi, detectEachRoi,
    testing::Combine(
        testing::Values(std::string("cv/cascadeandhog/cascades/inria_caltech-17.01.2013.xml"),
                        std::string("cv/cascadeandhog/cascades/sc_cvpr_2012_to_opencv_new_format.xml")),
        testing::Values(std::string("cv/cascadeandhog/images/image_00000000_0.png")),
        testing::Range(0, 10)))

RUN_GPU(SCascadeTestRoi, detectEachRoi)
{
    cv::Mat cpu = readImage (GET_PARAM(1));
    ASSERT_FALSE(cpu.empty());
    cv::gpu::GpuMat colored(cpu);

    cv::gpu::SCascade cascade;

    cv::FileStorage fs(perf::TestBase::getDataPath(GET_PARAM(0)), cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());

    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    cv::gpu::GpuMat objectBoxes(1, 16384 * 20, CV_8UC1), rois(colored.size(), CV_8UC1);
    rois.setTo(0);

    int idx = GET_PARAM(2);
    cv::Rect r = getFromTable(idx);
    cv::gpu::GpuMat sub(rois, r);
    sub.setTo(1);

    cascade.detect(colored, rois, objectBoxes);

    TEST_CYCLE()
    {
        cascade.detect(colored, rois, objectBoxes);
    }

    SANITY_CHECK(sortDetections(objectBoxes));
}

NO_CPU(SCascadeTestRoi, detectEachRoi)

SC_PERF_TEST_P(SCascadeTest, detectStream,
    testing::Combine(
        testing::Values(std::string("cv/cascadeandhog/cascades/inria_caltech-17.01.2013.xml"),
                        std::string("cv/cascadeandhog/cascades/sc_cvpr_2012_to_opencv_new_format.xml")),
        testing::Values(std::string("cv/cascadeandhog/images/image_00000000_0.png"))))

RUN_GPU(SCascadeTest, detectStream)
{
    cv::Mat cpu = readImage (GET_PARAM(1));
    ASSERT_FALSE(cpu.empty());
    cv::gpu::GpuMat colored(cpu);

    cv::gpu::SCascade cascade;

    cv::FileStorage fs(perf::TestBase::getDataPath(GET_PARAM(0)), cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());

    ASSERT_TRUE(cascade.load(fs.getFirstTopLevelNode()));

    cv::gpu::GpuMat objectBoxes(1, 10000 * sizeof(cv::gpu::SCascade::Detection), CV_8UC1), rois(colored.size(), CV_8UC1);
    rois.setTo(1);

    cv::gpu::Stream s;

    cascade.detect(colored, rois, objectBoxes, s);

    TEST_CYCLE()
    {
        cascade.detect(colored, rois, objectBoxes, s);
    }

    s.waitForCompletion();
    SANITY_CHECK(sortDetections(objectBoxes));
}

NO_CPU(SCascadeTest, detectStream)

#undef SC_PERF_TEST_P