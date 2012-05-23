#ifndef __OPENCV_PERF_GPU_UTILITY_HPP__
#define __OPENCV_PERF_GPU_UTILITY_HPP__

void fill(cv::Mat& m, double a, double b);

using perf::MatType;
using perf::MatDepth;

CV_ENUM(BorderMode, cv::BORDER_REFLECT101, cv::BORDER_REPLICATE, cv::BORDER_CONSTANT, cv::BORDER_REFLECT, cv::BORDER_WRAP)
CV_ENUM(Interpolation, cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC)
CV_ENUM(NormType, cv::NORM_INF, cv::NORM_L1, cv::NORM_L2)

struct CvtColorInfo
{
    int scn;
    int dcn;
    int code;

    explicit CvtColorInfo(int scn_=0, int dcn_=0, int code_=0) : scn(scn_), dcn(dcn_), code(code_) {}
};

void PrintTo(const CvtColorInfo& info, std::ostream* os);

#define IMPLEMENT_PARAM_CLASS(name, type) \
    class name \
    { \
    public: \
        name ( type arg = type ()) : val_(arg) {} \
        operator type () const {return val_;} \
    private: \
        type val_; \
    }; \
    inline void PrintTo( name param, std::ostream* os) \
    { \
        *os << #name <<  "(" << testing::PrintToString(static_cast< type >(param)) << ")"; \
    }

IMPLEMENT_PARAM_CLASS(Channels, int)

namespace cv { namespace gpu
{
    void PrintTo(const cv::gpu::DeviceInfo& info, std::ostream* os);
}}

#define GPU_PERF_TEST(name, ...) \
    struct name : perf::TestBaseWithParam< std::tr1::tuple< __VA_ARGS__ > > \
    { \
    public: \
        name() {} \
    protected: \
        void PerfTestBody(); \
    }; \
    TEST_P(name, perf){ RunPerfTestBody(); } \
    void name :: PerfTestBody()

#define GPU_PERF_TEST_1(name, param_type) \
    struct name : perf::TestBaseWithParam< param_type > \
    { \
    public: \
        name() {} \
    protected: \
        void PerfTestBody(); \
    }; \
    TEST_P(name, perf){ RunPerfTestBody(); } \
    void name :: PerfTestBody()

#define GPU_TYPICAL_MAT_SIZES testing::Values(perf::szSXGA, perf::sz1080p, cv::Size(1800, 1500))

cv::Mat readImage(const std::string& fileName, int flags = cv::IMREAD_COLOR);

const std::vector<cv::gpu::DeviceInfo>& devices();

#define ALL_DEVICES testing::ValuesIn(devices())

#define GET_PARAM(k) std::tr1::get< k >(GetParam())

#endif // __OPENCV_PERF_GPU_UTILITY_HPP__
