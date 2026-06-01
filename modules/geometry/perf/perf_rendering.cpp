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
    static const std::array<TriangleShadingType, 3> vals;
    static const std::array<std::string, 3> svals;

    ShadingTypeEnum(TriangleShadingType v = RASTERIZE_SHADING_WHITE) : val(v) {}
    operator TriangleShadingType() const { return val; }
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
    TriangleShadingType val;
};

const std::array<TriangleShadingType, 3> ShadingTypeEnum::vals
{
    RASTERIZE_SHADING_WHITE,
    RASTERIZE_SHADING_FLAT,
    RASTERIZE_SHADING_SHADED
};
const std::array<std::string, 3> ShadingTypeEnum::svals
{
    std::string("White"),
    std::string("Flat"),
    std::string("Shaded")
};

static inline void PrintTo(const ShadingTypeEnum &t, std::ostream *os) { t.PrintTo(os); }


using namespace cv;
struct GlCompatibleModeEnum
{
    static const std::array<TriangleGlCompatibleMode, 2> vals;
    static const std::array<std::string, 2> svals;

    GlCompatibleModeEnum(TriangleGlCompatibleMode v = RASTERIZE_COMPAT_DISABLED) : val(v) {}
    operator TriangleGlCompatibleMode() const { return val; }
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
    static ::testing::internal::ParamGenerator<GlCompatibleModeEnum> all()
    {
        return ::testing::Values(GlCompatibleModeEnum(vals[0]),
                                 GlCompatibleModeEnum(vals[1]));
    }

private:
    TriangleGlCompatibleMode val;
};

const std::array<TriangleGlCompatibleMode, 2> GlCompatibleModeEnum::vals
{
    RASTERIZE_COMPAT_DISABLED,
    RASTERIZE_COMPAT_INVDEPTH,
};
const std::array<std::string, 2> GlCompatibleModeEnum::svals
{
    std::string("Disabled"),
    std::string("InvertedDepth"),
};

static inline void PrintTo(const GlCompatibleModeEnum &t, std::ostream *os) { t.PrintTo(os); }
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

const std::array<Outputs, 3> OutputsEnum::vals
{
    Outputs::DepthOnly,
    Outputs::ColorOnly,
    Outputs::DepthColor
};
const std::array<std::string, 3> OutputsEnum::svals
{
    std::string("DepthOnly"),
    std::string("ColorOnly"),
    std::string("DepthColor")
};

static inline void PrintTo(const OutputsEnum &t, std::ostream *os) { t.PrintTo(os); }
}

static Matx44d lookAtMatrixCal(const Vec3d& position, const Vec3d& lookat, const Vec3d& upVector)
{
    Vec3d w = cv::normalize(position - lookat);
    Vec3d u = cv::normalize(upVector.cross(w));

    Vec3d v = w.cross(u);

    Matx44d res(u[0], u[1], u[2],   0,
                v[0], v[1], v[2],   0,
                w[0], w[1], w[2],   0,
                   0,    0,    0,   1.0);

    Matx44d translate(1.0,   0,   0, -position[0],
                        0, 1.0,   0, -position[1],
                        0,   0, 1.0, -position[2],
                        0,   0,   0,          1.0);
    res = res * translate;

    return res;
}


static void generateNormals(const std::vector<Vec3f>& points, const std::vector<std::vector<int>>& indices,
                            std::vector<Vec3f>& normals)
{
    std::vector<std::vector<Vec3f>> preNormals(points.size(), std::vector<Vec3f>());

    for (const auto& tri : indices)
    {
        Vec3f p0 = points[tri[0]];
        Vec3f p1 = points[tri[1]];
        Vec3f p2 = points[tri[2]];

        Vec3f cross = cv::normalize((p1 - p0).cross(p2 - p0));
        for (int i = 0; i < 3; i++)
        {
            preNormals[tri[i]].push_back(cross);
        }
    }

    normals.reserve(points.size());
    for (const auto& pn : preNormals)
    {
        Vec3f sum { };
        for (const auto& n : pn)
        {
            sum += n;
        }
        normals.push_back(cv::normalize(sum));
    }
}

// load model once and keep it in static memory
static void getModelOnce(const std::string& objectPath, std::vector<Vec3f>& vertices,
                         std::vector<Vec3i>& indices, std::vector<Vec3f>& colors)
{
    static bool load = false;
    static std::vector<Vec3f> vert, col;
    static std::vector<Vec3i> ind;

    if (!load)
    {
        std::vector<vector<int>> indvec;
        // using per-vertex normals as colors
        loadMesh(objectPath, vert, indvec);
        generateNormals(vert, indvec, col);

        for (const auto &vec : indvec)
        {
            ind.push_back({vec[0], vec[1], vec[2]});
        }

        for (auto &color : col)
        {
            color = Vec3f(abs(color[0]), abs(color[1]), abs(color[2]));
        }

        load = true;
    }

    vertices = vert;
    colors = col;
    indices = ind;
}

template<typename T>
std::string printEnum(T v)
{
    std::ostringstream ss;
    v.PrintTo(&ss);
    return ss.str();
}

// resolution, shading type, outputs needed
typedef perf::TestBaseWithParam<std::tuple<std::tuple<int, int>, ShadingTypeEnum, OutputsEnum, GlCompatibleModeEnum>> RenderingTest;

PERF_TEST_P(RenderingTest, rasterizeTriangles, ::testing::Combine(
    ::testing::Values(std::make_tuple(1920, 1080), std::make_tuple(1024, 768), std::make_tuple(640, 480)),
    ShadingTypeEnum::all(),
    OutputsEnum::all(),
    GlCompatibleModeEnum::all()
    ))
{
    auto t = GetParam();
    auto wh = std::get<0>(t);
    int width = std::get<0>(wh);
    int height = std::get<1>(wh);
    auto shadingType = std::get<1>(t);
    auto outputs = std::get<2>(t);
    auto glCompatibleMode = std::get<3>(t);

    string objectPath = findDataFile("viz/dragon.ply");

    Vec3f position = Vec3d( 1.9, 0.4, 1.3);
    Vec3f lookat   = Vec3d( 0.0, 0.0, 0.0);
    Vec3f upVector = Vec3d( 0.0, 1.0, 0.0);

    double fovy = 45.0;

    std::vector<Vec3f> vertices;
    std::vector<Vec3i> indices;
    std::vector<Vec3f> colors;

    getModelOnce(objectPath, vertices, indices, colors);
    if (shadingType != RASTERIZE_SHADING_WHITE)
    {
        // let vertices be in BGR format to avoid later color conversions
        // mixChannels does not support in-place operation
        cv::mixChannels(Mat(colors).clone(), colors, {0, 2, 1, 1, 2, 0});
    }

    double zNear = 0.1, zFar = 50.0;

    Matx44d cameraPose = lookAtMatrixCal(position, lookat, upVector);
    double fovYradians = fovy * (CV_PI / 180.0);
    TriangleRasterizeSettings settings;
    settings.setCullingMode(RASTERIZE_CULLING_CW)
            .setShadingType(shadingType)
            .setGlCompatibleMode(glCompatibleMode);

    Mat depth_buf, color_buf;
    while (next())
    {
        // Prefilled to measure pure rendering time w/o allocation and clear
        float zMax = (glCompatibleMode == RASTERIZE_COMPAT_INVDEPTH) ? 1.f : (float)zFar;
        depth_buf = Mat(height, width, CV_32F, zMax);
        color_buf = Mat(height, width, CV_32FC3, Scalar::all(0));

        startTimer();
        if (outputs == Outputs::ColorOnly)
        {
            cv::triangleRasterizeColor(vertices, indices, colors, color_buf, cameraPose,
                                       fovYradians, zNear, zFar, settings);
        }
        else if (outputs == Outputs::DepthOnly)
        {
            cv::triangleRasterizeDepth(vertices, indices, depth_buf, cameraPose,
                                       fovYradians, zNear, zFar, settings);
        }
        else // Outputs::DepthColor
        {
            cv::triangleRasterize(vertices, indices, colors, color_buf, depth_buf,
                                  cameraPose, fovYradians, zNear, zFar, settings);
        }
        stopTimer();
    }

    if (debugLevel > 0)
    {
        depth_buf.convertTo(depth_buf, CV_16U, 1000.0);

        std::string shadingName = printEnum(shadingType);
        std::string suffix = cv::format("%dx%d_%s", width, height, shadingName.c_str());

        imwrite("perf_color_image_" + suffix + ".png", color_buf * 255.f);
        imwrite("perf_depth_image_" + suffix + ".png", depth_buf);
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
