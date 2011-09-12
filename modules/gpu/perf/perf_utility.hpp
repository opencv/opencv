#ifndef __OPENCV_PERF_GPU_UTILITY_HPP__
#define __OPENCV_PERF_GPU_UTILITY_HPP__

#include <iosfwd>
#include "opencv2/ts/ts.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace perf;

enum {HORIZONTAL_AXIS = 0, VERTICAL_AXIS = 1, BOTH_AXIS = -1};

CV_ENUM(MorphOp, MORPH_ERODE, MORPH_DILATE)
CV_ENUM(BorderMode, BORDER_REFLECT101, BORDER_REPLICATE, BORDER_CONSTANT, BORDER_REFLECT, BORDER_WRAP)
CV_ENUM(FlipCode, HORIZONTAL_AXIS, VERTICAL_AXIS, BOTH_AXIS)
CV_ENUM(CmpOp, CMP_EQ, CMP_GT, CMP_GE, CMP_LT, CMP_LE, CMP_NE)
CV_ENUM(Interpolation, INTER_NEAREST, INTER_LINEAR, INTER_CUBIC)
CV_ENUM(MatchMethod, TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_CCOEFF, TM_CCOEFF_NORMED)
CV_ENUM(NormType, NORM_INF, NORM_L1, NORM_L2)

struct CvtColorInfo
{
    int scn;
    int dcn;
    int code;

    explicit CvtColorInfo(int scn_=0, int dcn_=0, int code_=0) : scn(scn_), dcn(dcn_), code(code_) {}
};

typedef TestBaseWithParam<DeviceInfo> DevInfo;
typedef TestBaseWithParam< std::tr1::tuple<DeviceInfo, Size> > DevInfo_Size;
typedef TestBaseWithParam< std::tr1::tuple<DeviceInfo, MatType> > DevInfo_MatType;
typedef TestBaseWithParam< std::tr1::tuple<DeviceInfo, Size, MatType> > DevInfo_Size_MatType;
typedef TestBaseWithParam< std::tr1::tuple<DeviceInfo, Size, MatType, MatType> > DevInfo_Size_MatType_MatType;
typedef TestBaseWithParam< std::tr1::tuple<DeviceInfo, Size, MatType, int> > DevInfo_Size_MatType_KernelSize;
typedef TestBaseWithParam< std::tr1::tuple<DeviceInfo, Size, MatType, MorphOp, int> > DevInfo_Size_MatType_MorphOp_KernelSize;
typedef TestBaseWithParam< std::tr1::tuple<DeviceInfo, Size, MatType, int, BorderMode> > DevInfo_Size_MatType_KernelSize_BorderMode;
typedef TestBaseWithParam< std::tr1::tuple<DeviceInfo, Size, MatType, FlipCode> > DevInfo_Size_MatType_FlipCode;
typedef TestBaseWithParam< std::tr1::tuple<DeviceInfo, Size, MatType, CmpOp> > DevInfo_Size_MatType_CmpOp;
typedef TestBaseWithParam< std::tr1::tuple<DeviceInfo, Size, MatType, Interpolation> > DevInfo_Size_MatType_Interpolation;
typedef TestBaseWithParam< std::tr1::tuple<DeviceInfo, Size, MatType, Interpolation, double> > DevInfo_Size_MatType_Interpolation_SizeCoeff;
typedef TestBaseWithParam< std::tr1::tuple<DeviceInfo, Size, MatType, Interpolation, BorderMode> > DevInfo_Size_MatType_Interpolation_BorderMode;
typedef TestBaseWithParam< std::tr1::tuple<DeviceInfo, Size, MatType, CvtColorInfo> > DevInfo_Size_MatType_CvtColorInfo;
typedef TestBaseWithParam< std::tr1::tuple<DeviceInfo, Size, MatType, MatchMethod> > DevInfo_Size_MatType_MatchMethod;
typedef TestBaseWithParam< std::tr1::tuple<DeviceInfo, Size, NormType> > DevInfo_Size_NormType;
typedef TestBaseWithParam< std::tr1::tuple<DeviceInfo, Size, MatType, NormType> > DevInfo_Size_MatType_NormType;
typedef TestBaseWithParam< std::tr1::tuple<DeviceInfo, int> > DevInfo_DescSize;
typedef TestBaseWithParam< std::tr1::tuple<DeviceInfo, int, int> > DevInfo_K_DescSize;

const cv::Size sz1800x1500 = cv::Size(1800, 1500);
const cv::Size sz4700x3000 = cv::Size(4700, 3000);

#define GPU_TYPICAL_MAT_SIZES szXGA, szSXGA, sz720p, sz1080p, sz1800x1500, sz4700x3000

//! read image from testdata folder.
Mat readImage(const string& fileName, int flags = CV_LOAD_IMAGE_COLOR);

//! return true if device supports specified feature and gpu module was built with support the feature.
bool supportFeature(const cv::gpu::DeviceInfo& info, cv::gpu::FeatureSet feature);

//! return all devices compatible with current gpu module build.
const std::vector<cv::gpu::DeviceInfo>& devices();
//! return all devices compatible with current gpu module build which support specified feature.
std::vector<cv::gpu::DeviceInfo> devices(cv::gpu::FeatureSet feature);

namespace cv
{
    namespace gpu
    {
        void PrintTo(const DeviceInfo& info, ::std::ostream* os);
    }
}

void PrintTo(const CvtColorInfo& info, ::std::ostream* os);

#endif // __OPENCV_PERF_GPU_UTILITY_HPP__
