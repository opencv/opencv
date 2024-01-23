// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "perf_precomp.hpp"

namespace opencv_test
{

// that was easier than using CV_ENUM() macro
namespace
{
using namespace cv;
struct ShadingTypeEnum
{
    static const std::array<ShadingType, 3> vals;
    static const std::array<std::string, 3> svals;

    ShadingTypeEnum(ShadingType v = ShadingType::White) : val(v) {}
    operator ShadingType() const { return val; }
    void PrintTo(std::ostream *os) const
    {
        int v = int(val);
        if (v >= 0 && v < (int)vals.size())
        {
            *os << svals[v];
        }
        else
        {
            *os << "UNKNOWN";
        }
    }
    static ::testing::internal::ParamGenerator<ShadingTypeEnum> all()
    {
        return ::testing::Values(ShadingTypeEnum(vals[0]),
                                 ShadingTypeEnum(vals[1]),
                                 ShadingTypeEnum(vals[2]));
    }

private:
    ShadingType val;
};

const std::array<ShadingType, 3> ShadingTypeEnum::vals{ ShadingType::White,
                                                        ShadingType::Flat,
                                                        ShadingType::Shaded
                                                      };
const std::array<std::string, 3> ShadingTypeEnum::svals{ std::string("White"),
                                                         std::string("Flat"),
                                                         std::string("Shaded")
                                                       };

static inline void PrintTo(const ShadingTypeEnum &t, std::ostream *os) { t.PrintTo(os); }
}

enum class Outputs
{
    DepthOnly  = 0,
    ColorOnly  = 1,
    DepthColor = 2,
};

// that was easier than using CV_ENUM() macro
namespace
{
using namespace cv;
struct OutputsEnum
{
    static const std::array<Outputs, 3> vals;
    static const std::array<std::string, 3> svals;

    OutputsEnum(Outputs v = Outputs::DepthColor) : val(v) {}
    operator Outputs() const { return val; }
    void PrintTo(std::ostream *os) const
    {
        int v = int(val);
        if (v >= 0 && v < (int)vals.size())
        {
            *os << svals[v];
        }
        else
        {
            *os << "UNKNOWN";
        }
    }
    static ::testing::internal::ParamGenerator<OutputsEnum> all()
    {
        return ::testing::Values(OutputsEnum(vals[0]),
                                 OutputsEnum(vals[1]),
                                 OutputsEnum(vals[2]));
    }

private:
    Outputs val;
};

const std::array<Outputs, 3> OutputsEnum::vals{ Outputs::DepthOnly,
                                                Outputs::ColorOnly,
                                                Outputs::DepthColor
                                              };
const std::array<std::string, 3> OutputsEnum::svals{ std::string("DepthOnly"),
                                                     std::string("ColorOnly"),
                                                     std::string("DepthColor")
                                                   };

static inline void PrintTo(const OutputsEnum &t, std::ostream *os) { t.PrintTo(os); }
}

static Matx44f lookAtMatrixCal(const Vec3f& position, const Vec3f& lookat, const Vec3f& upVector)
{
    Vec3f w = cv::normalize(position - lookat);
    Vec3f u = cv::normalize(upVector.cross(w));

    Vec3f v = w.cross(u);

    Matx44f res(u[0], u[1], u[2],   0,
                v[0], v[1], v[2],   0,
                w[0], w[1], w[2],   0,
                   0,    0,    0,   1.f);

    Matx44f translate(1.f,   0,   0, -position[0],
                        0, 1.f,   0, -position[1],
                        0,   0, 1.f, -position[2],
                        0,   0,   0,          1.0f);
    res = res * translate;

    return res;
}

template<typename T>
std::string printEnum(T v)
{
    std::string s;
    std::stringstream ss;
    v.PrintTo(&ss);
    ss >> s;
    return s;
}

// resolution, shading type, outputs needed
typedef perf::TestBaseWithParam<std::tuple<std::tuple<int, int>, ShadingTypeEnum, OutputsEnum>> RenderingTest;

PERF_TEST_P(RenderingTest, rasterizeTriangles, ::testing::Combine(
    ::testing::Values(std::make_tuple(1920, 1080), std::make_tuple(1024, 768), std::make_tuple(640, 480)),
    ShadingTypeEnum::all(),
    OutputsEnum::all()
    ))
{
    auto t = GetParam();
    auto wh = std::get<0>(t);
    int width = std::get<0>(wh);
    int height = std::get<1>(wh);
    auto shadingType = std::get<1>(t);
    auto outputs = std::get<2>(t);

    string objectPath = findDataFile("rendering/spot.obj");

    Vec3f position = Vec3f( 2.4f, 0.7f, 1.2f);
    Vec3f lookat   = Vec3f( 0.0f, 0.0f, 0.3f);
    Vec3f upVector = Vec3f( 0.0f, 1.0f, 0.0f);

    double fovy = 45.0;

    std::vector<Vec3f> vertices;
    std::vector<Vec3i> indices;
    std::vector<Vec3f> colors;
    std::vector<vector<int>> indvec;
    loadMesh(objectPath, vertices, colors, indvec);
    for (const auto &vec : indvec)
    {
        indices.push_back({vec[0], vec[1], vec[2]});
    }

    if (shadingType == ShadingType::White)
    {
        colors.clear();
    }
    else
    {
        for (auto &color : colors)
        {
            color = Vec3f(abs(color[0]), abs(color[1]), abs(color[2]));
        }
    }

    float zNear = 0.1f, zFar = 50.f;

    Matx44f cameraPose = lookAtMatrixCal(position, lookat, upVector);
    float fovYradians = fovy * (float)(CV_PI / 180.0);
    RasterizeSettings settings = RasterizeSettings().setCullingMode(CullingMode::CW).setShadingType(shadingType);

    Mat depth_buf, color_buf;
    while (next())
    {
        // Prefilled to measure pure rendering time w/o allocation and clear
        depth_buf = Mat(height, width, CV_32F, zFar);
        color_buf = Mat(height, width, CV_32FC3, Scalar::all(0));

        startTimer();
        cv::triangleRasterize(vertices, indices, colors, cameraPose, fovYradians, zNear, zFar, settings,
                              (outputs != Outputs::ColorOnly) ? depth_buf : noArray(),
                              (outputs != Outputs::DepthOnly) ? color_buf : noArray());
        stopTimer();
    }

    if (debugLevel > 0)
    {
        cvtColor(color_buf, color_buf, cv::COLOR_RGB2BGR);
        cv::flip(color_buf, color_buf, 0);
        depth_buf.convertTo(depth_buf, CV_16U, 1000.0);
        cv::flip(depth_buf, depth_buf, 0);

        std::string shadingName = printEnum(shadingType);
        std::string suffix = cv::format("%dx%d_%s", width, height, shadingName.c_str());

        imwrite("perf_color_image_" + suffix + ".png", color_buf * 255.f);
        imwrite("perf_depth_image_" + suffix + ".png", depth_buf);
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
