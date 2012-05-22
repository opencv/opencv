#include "perf_cpu_precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

void fill(Mat& m, double a, double b)
{
    RNG rng(123456789);
    rng.fill(m, RNG::UNIFORM, a, b);
}

void PrintTo(const CvtColorInfo& info, ostream* os)
{
    static const char* str[] =
    {
        "BGR2BGRA",
        "BGRA2BGR",
        "BGR2RGBA",
        "RGBA2BGR",
        "BGR2RGB",
        "BGRA2RGBA",

        "BGR2GRAY",
        "RGB2GRAY",
        "GRAY2BGR",
        "GRAY2BGRA",
        "BGRA2GRAY",
        "RGBA2GRAY",

        "BGR2BGR565",
        "RGB2BGR565",
        "BGR5652BGR",
        "BGR5652RGB",
        "BGRA2BGR565",
        "RGBA2BGR565",
        "BGR5652BGRA",
        "BGR5652RGBA",

        "GRAY2BGR565",
        "BGR5652GRAY",

        "BGR2BGR555",
        "RGB2BGR555",
        "BGR5552BGR",
        "BGR5552RGB",
        "BGRA2BGR555",
        "RGBA2BGR555",
        "BGR5552BGRA",
        "BGR5552RGBA",

        "GRAY2BGR555",
        "BGR5552GRAY",

        "BGR2XYZ",
        "RGB2XYZ",
        "XYZ2BGR",
        "XYZ2RGB",

        "BGR2YCrCb",
        "RGB2YCrCb",
        "YCrCb2BGR",
        "YCrCb2RGB",

        "BGR2HSV",
        "RGB2HSV",

        0,
        0,

        0,
        0,

        0,
        0,
        0,
        0,

        0,
        0,

        "BGR2HLS",
        "RGB2HLS",

        "HSV2BGR",
        "HSV2RGB",

        0,
        0,
        0,
        0,

        "HLS2BGR",
        "HLS2RGB",

        0,
        0,
        0,
        0,

        "BGR2HSV_FULL",
        "RGB2HSV_FULL",
        "BGR2HLS_FULL",
        "RGB2HLS_FULL",

        "HSV2BGR_FULL",
        "HSV2RGB_FULL",
        "HLS2BGR_FULL",
        "HLS2RGB_FULL",

        0,
        0,
        0,
        0,

        0,
        0,
        0,
        0,

        "BGR2YUV",
        "RGB2YUV",
        "YUV2BGR",
        "YUV2RGB",

        0,
        0,
        0,
        0,

        0,
        0,
        0,
        0
    };

    *os << str[info.code];
}

void cv::gpu::PrintTo(const DeviceInfo& info, ostream* os)
{
    *os << info.name();
}

Mat readImage(const string& fileName, int flags)
{
    return imread(perf::TestBase::getDataPath(fileName), flags);
}

bool supportFeature(const DeviceInfo& info, FeatureSet feature)
{
    return TargetArchs::builtWith(feature) && info.supports(feature);
}

const vector<DeviceInfo>& devices()
{
    static vector<DeviceInfo> devs;
    static bool first = true;

    if (first)
    {
        int deviceCount = getCudaEnabledDeviceCount();

        devs.reserve(deviceCount);

        for (int i = 0; i < deviceCount; ++i)
        {
            DeviceInfo info(i);
            if (info.isCompatible())
                devs.push_back(info);
        }

        first = false;
    }

    return devs;
}

vector<DeviceInfo> devices(FeatureSet feature)
{
    const vector<DeviceInfo>& d = devices();

    vector<DeviceInfo> devs_filtered;

    if (TargetArchs::builtWith(feature))
    {
        devs_filtered.reserve(d.size());

        for (size_t i = 0, size = d.size(); i < size; ++i)
        {
            const DeviceInfo& info = d[i];

            if (info.supports(feature))
                devs_filtered.push_back(info);
        }
    }

    return devs_filtered;
}


