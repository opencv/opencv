#include "perf_precomp.hpp"

#define GPU_SANITY_CHECK(dmat, ...) \
    do{ \
        cv::Mat d##dmat(dmat); \
        SANITY_CHECK(d##dmat, ## __VA_ARGS__); \
    } while(0)

#define CPU_SANITY_CHECK(cmat, ...) \
    do{ \
        SANITY_CHECK(cmat, ## __VA_ARGS__); \
    } while(0)

typedef perf::TestBaseWithParam<std::string> VideoFile;

PERF_TEST_P(VideoFile, SuperResolution_BTVL1, testing::Values(std::string("superres/car.avi")))
{
    declare.time(5 * 60);

    const std::string inputVideoName = perf::TestBase::getDataPath(GetParam());

    if (PERF_RUN_GPU())
    {
        cv::Ptr<cv::superres::SuperResolution> superRes = cv::superres::createSuperResolution_BTVL1_GPU();

        superRes->setInput(cv::superres::createFrameSource_Video(inputVideoName));

        cv::gpu::GpuMat d_dst;
        superRes->nextFrame(d_dst);

        TEST_CYCLE_N(10) superRes->nextFrame(d_dst);

        GPU_SANITY_CHECK(d_dst);
    }
    else
    {
        cv::Ptr<cv::superres::SuperResolution> superRes = cv::superres::createSuperResolution_BTVL1();

        superRes->setInput(cv::superres::createFrameSource_Video(inputVideoName));

        cv::Mat dst;
        superRes->nextFrame(dst);

        TEST_CYCLE_N(10) superRes->nextFrame(dst);

        CPU_SANITY_CHECK(dst);
    }
}
