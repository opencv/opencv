#ifndef __OPENCV_PERF_GPU_UTILITY_HPP__
#define __OPENCV_PERF_GPU_UTILITY_HPP__

void fill(cv::Mat& m, double a, double b);

enum {HORIZONTAL_AXIS = 0, VERTICAL_AXIS = 1, BOTH_AXIS = -1};

using perf::MatType;
using perf::MatDepth;

CV_ENUM(BorderMode, cv::BORDER_REFLECT101, cv::BORDER_REPLICATE, cv::BORDER_CONSTANT, cv::BORDER_REFLECT, cv::BORDER_WRAP)
CV_ENUM(FlipCode, HORIZONTAL_AXIS, VERTICAL_AXIS, BOTH_AXIS)
CV_ENUM(Interpolation, cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC)
CV_ENUM(MatchMethod, cv::TM_SQDIFF, cv::TM_SQDIFF_NORMED, cv::TM_CCORR, cv::TM_CCORR_NORMED, cv::TM_CCOEFF, cv::TM_CCOEFF_NORMED)
CV_ENUM(NormType, cv::NORM_INF, cv::NORM_L1, cv::NORM_L2)
CV_ENUM(AlphaOp, cv::gpu::ALPHA_OVER, cv::gpu::ALPHA_IN, cv::gpu::ALPHA_OUT, cv::gpu::ALPHA_ATOP, cv::gpu::ALPHA_XOR, cv::gpu::ALPHA_PLUS, cv::gpu::ALPHA_OVER_PREMUL, cv::gpu::ALPHA_IN_PREMUL, cv::gpu::ALPHA_OUT_PREMUL, cv::gpu::ALPHA_ATOP_PREMUL, cv::gpu::ALPHA_XOR_PREMUL, cv::gpu::ALPHA_PLUS_PREMUL, cv::gpu::ALPHA_PREMUL)

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

struct CvtColorInfo
{
    int scn;
    int dcn;
    int code;

    explicit CvtColorInfo(int scn_=0, int dcn_=0, int code_=0) : scn(scn_), dcn(dcn_), code(code_) {}
};

void PrintTo(const CvtColorInfo& info, std::ostream* os);

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

bool supportFeature(const cv::gpu::DeviceInfo& info, cv::gpu::FeatureSet feature);

const std::vector<cv::gpu::DeviceInfo>& devices();

std::vector<cv::gpu::DeviceInfo> devices(cv::gpu::FeatureSet feature);

#define ALL_DEVICES testing::ValuesIn(devices())
#define DEVICES(feature) testing::ValuesIn(devices(feature))

#define GET_PARAM(k) std::tr1::get< k >(GetParam())

#endif // __OPENCV_PERF_GPU_UTILITY_HPP__
