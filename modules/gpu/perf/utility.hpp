#ifndef __OPENCV_PERF_GPU_UTILITY_HPP__
#define __OPENCV_PERF_GPU_UTILITY_HPP__

#include "opencv2/core/core.hpp"
#include "opencv2/core/gpumat.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ts/ts_perf.hpp"

void fillRandom(cv::Mat& m, double a = 0.0, double b = 255.0);
cv::Mat readImage(const std::string& fileName, int flags = cv::IMREAD_COLOR);

using perf::MatType;
using perf::MatDepth;

CV_ENUM(BorderMode, cv::BORDER_REFLECT101, cv::BORDER_REPLICATE, cv::BORDER_CONSTANT, cv::BORDER_REFLECT, cv::BORDER_WRAP)
#define ALL_BORDER_MODES testing::ValuesIn(BorderMode::all())

CV_ENUM(Interpolation, cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC, cv::INTER_AREA)
#define ALL_INTERPOLATIONS testing::ValuesIn(Interpolation::all())
CV_ENUM(NormType, cv::NORM_INF, cv::NORM_L1, cv::NORM_L2, cv::NORM_HAMMING, cv::NORM_MINMAX)

const int Gray = 1, TwoChannel = 2, BGR = 3, BGRA = 4;
CV_ENUM(MatCn, Gray, TwoChannel, BGR, BGRA)
#define GPU_CHANNELS_1_3_4 testing::Values(Gray, BGR, BGRA)
#define GPU_CHANNELS_1_3 testing::Values(Gray, BGR)

struct CvtColorInfo
{
    int scn;
    int dcn;
    int code;

    explicit CvtColorInfo(int scn_=0, int dcn_=0, int code_=0) : scn(scn_), dcn(dcn_), code(code_) {}
};
void PrintTo(const CvtColorInfo& info, std::ostream* os);

#define GET_PARAM(k) std::tr1::get< k >(GetParam())

#define DEF_PARAM_TEST(name, ...) typedef ::perf::TestBaseWithParam< std::tr1::tuple< __VA_ARGS__ > > name
#define DEF_PARAM_TEST_1(name, param_type) typedef ::perf::TestBaseWithParam< param_type > name

DEF_PARAM_TEST_1(Sz, cv::Size);
typedef perf::Size_MatType Sz_Type;
DEF_PARAM_TEST(Sz_Depth, cv::Size, MatDepth);
DEF_PARAM_TEST(Sz_Depth_Cn, cv::Size, MatDepth, MatCn);

#define GPU_TYPICAL_MAT_SIZES testing::Values(perf::sz720p, perf::szSXGA, perf::sz1080p)

#define GPU_SANITY_CHECK(dmat, ...) \
    do{ \
        cv::Mat d##dmat(dmat); \
        SANITY_CHECK(d##dmat, ## __VA_ARGS__); \
    } while(0)

#define CPU_SANITY_CHECK(cmat, ...) \
    do{ \
        SANITY_CHECK(cmat, ## __VA_ARGS__); \
    } while(0)

#define GPU_SANITY_CHECK_KEYPOINTS(alg, dmat, ...)                                          \
    do{                                                                                     \
        cv::Mat d##dmat(dmat);                                                              \
        cv::Mat __pt_x      = d##dmat.row(cv::gpu::alg##_GPU::X_ROW);                       \
        cv::Mat __pt_y      = d##dmat.row(cv::gpu::alg##_GPU::Y_ROW);                       \
        cv::Mat __angle     = d##dmat.row(cv::gpu::alg##_GPU::ANGLE_ROW);                   \
        cv::Mat __octave    = d##dmat.row(cv::gpu::alg##_GPU::OCTAVE_ROW);                               \
        cv::Mat __size      = d##dmat.row(cv::gpu::alg##_GPU::SIZE_ROW);                                 \
        ::perf::Regression::add(this, std::string(#dmat) + "-pt-x-row",     __pt_x,     ## __VA_ARGS__); \
        ::perf::Regression::add(this, std::string(#dmat) + "-pt-y-row",     __pt_y,     ## __VA_ARGS__); \
        ::perf::Regression::add(this, std::string(#dmat) + "-angle-row",    __angle,    ## __VA_ARGS__); \
        ::perf::Regression::add(this, std::string(#dmat) + "octave-row",    __octave,   ## __VA_ARGS__); \
        ::perf::Regression::add(this, std::string(#dmat) + "-pt-size-row",  __size,     ## __VA_ARGS__); \
    } while(0)

#define GPU_SANITY_CHECK_RESPONSE(alg, dmat, ...) \
    do{                                                                                     \
        cv::Mat d##dmat(dmat);                                                              \
        cv::Mat __response  = d##dmat.row(cv::gpu::alg##_GPU::RESPONSE_ROW);                \
        ::perf::Regression::add(this, std::string(#dmat) + "-response-row", __response, ## __VA_ARGS__); \
    } while(0)

#define FAIL_NO_CPU()   FAIL() << "No such CPU implementation analogy"

#endif // __OPENCV_PERF_GPU_UTILITY_HPP__
