#include "perf_precomp.hpp"

namespace opencv_test
{

Matx43f makeCamMatrix_TODO_rewrite_it_later(Vec3f position, Vec3f lookat, Vec3f upVector, double fovy, double zNear, double zFar);
Matx43f makeCamMatrix_TODO_rewrite_it_later(Vec3f position, Vec3f lookat, Vec3f upVector, double fovy, double zNear, double zFar)
{
    Matx43f m;
    m(0, 0) = position(0); m(0, 1) = position(1); m(0, 2) = position(2);
    m(1, 0) = lookat  (0); m(1, 1) = lookat  (1); m(1, 2) = lookat  (2);
    m(2, 0) = upVector(0); m(2, 1) = upVector(1); m(2, 2) = upVector(2);
    m(3, 0) = fovy;        m(3, 1) = zNear;       m(3, 3) = zFar;

    return m;
}

enum class ShadingType
{
    White = 0,
    Flat = 1,
    Shaded = 2
};

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
            if (v >= 0 && v < 5)
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
            if (v >= 0 && v < 5)
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

    Vec3f position = Vec3f(20.0, 60.0, -40.0);
    Vec3f lookat   = Vec3f( 0.0,  0.0,   0.0);
    Vec3f upVector = Vec3f( 0.0,  1.0,   0.0);

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
            color = Vec3f(abs(color[0]), abs(color[1]), abs(color[2])) * 255.0f;
        }
    }

    double zNear = 0.1, zFar = 50;

    Mat cameraMatrix = Mat(makeCamMatrix_TODO_rewrite_it_later(position, lookat, upVector, fovy, zNear, zFar));

    Mat depth_buf, color_buf;
    while (next())
    {
        // Prefilled to measure pure rendering time w/o allocation and clear
        depth_buf = Mat(height, width, CV_32F, zFar);
        color_buf = Mat(height, width, CV_32FC3, Scalar(0.0, 0.0, 0.0));

        startTimer();
        cv::triangleRasterize(vertices, indices, colors, cameraMatrix, width, height,
                              (shadingType == ShadingType::Shaded),
                              (outputs != Outputs::ColorOnly) ? depth_buf : noArray(),
                              (outputs != Outputs::DepthOnly) ? color_buf : noArray());
        stopTimer();
    }

    if (debugLevel > 0)
    {
        color_buf.convertTo(color_buf, CV_8UC3, 1.0f);
        cvtColor(color_buf, color_buf, cv::COLOR_RGB2BGR);
        cv::flip(color_buf, color_buf, 0);
        depth_buf.convertTo(depth_buf, CV_8UC1, 1.0);
        cv::flip(depth_buf, depth_buf, 0);

        std::string shadingName;
        {
            std::stringstream ss;
            shadingType.PrintTo(&ss);
            ss >> shadingName;
        }
        std::string widthStr  = std::to_string(width);
        std::string heightStr = std::to_string(height);
        std::string suffix = widthStr + "x" + heightStr + "_" + shadingName;

        imwrite("color_image_" + suffix + ".png", color_buf);
        imwrite("depth_image_" + suffix + ".png", depth_buf);
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
