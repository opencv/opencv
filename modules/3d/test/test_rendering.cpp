#include "test_precomp.hpp"

namespace opencv_test { namespace {
using namespace cv;

enum class CullingMode
{
    None,
    CW,
    CCW
};

// that was easier than using CV_ENUM() macro
namespace
{
    using namespace cv;
    struct CullingModeEnum
    {
        static const std::array<CullingMode, 3> vals;
        static const std::array<std::string, 3> svals;

        CullingModeEnum(CullingMode v = CullingMode::None) : val(v) {}
        operator CullingMode() const { return val; }
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
        static ::testing::internal::ParamGenerator<CullingModeEnum> all()
        {
            return ::testing::Values(CullingModeEnum(vals[0]),
                                     CullingModeEnum(vals[1]),
                                     CullingModeEnum(vals[2]));
        }

    private:
        CullingMode val;
    };

    const std::array<CullingMode, 3> CullingModeEnum::vals{ CullingMode::None,
                                                            CullingMode::CW,
                                                            CullingMode::CCW
                                                          };
    const std::array<std::string, 3> CullingModeEnum::svals{ std::string("None"),
                                                             std::string("CW"),
                                                             std::string("CCW")
                                                           };

    static inline void PrintTo(const CullingModeEnum &t, std::ostream *os) { t.PrintTo(os); }
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


enum class ModelType
{
    Empty = 0,
    File = 1,
    Clipping = 2,
    Color = 3,
    Centered = 4
};

// that was easier than using CV_ENUM() macro
namespace
{
    using namespace cv;
    struct ModelTypeEnum
    {
        static const std::array<ModelType, 5> vals;
        static const std::array<std::string, 5> svals;

        ModelTypeEnum(ModelType v = ModelType::Empty) : val(v) {}
        operator ModelType() const { return val; }
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
        static ::testing::internal::ParamGenerator<ModelTypeEnum> all()
        {
            return ::testing::Values(ModelTypeEnum(vals[0]),
                                     ModelTypeEnum(vals[1]),
                                     ModelTypeEnum(vals[2]),
                                     ModelTypeEnum(vals[3]),
                                     ModelTypeEnum(vals[4]));
        }

    private:
        ModelType val;
    };

    const std::array<ModelType, 5> ModelTypeEnum::vals{ ModelType::Empty,
                                                        ModelType::File,
                                                        ModelType::Clipping,
                                                        ModelType::Color,
                                                        ModelType::Centered };
    const std::array<std::string, 5> ModelTypeEnum::svals{ std::string("Empty"),
                                                           std::string("File"),
                                                           std::string("Clipping"),
                                                           std::string("Color"),
                                                           std::string("Centered") };

    static inline void PrintTo(const ModelTypeEnum &t, std::ostream *os) { t.PrintTo(os); }
}

class ModelData
{
public:
    ModelData(ModelType type = ModelType::Empty)
    {
        switch (type)
        {
        case ModelType::Empty:
        {
            position = Vec3f(0.0, 0.0,  0.0);
            lookat   = Vec3f(0.0, 0.0,  0.0);
            upVector = Vec3f(0.0, 1.0,  0.0);

            fovy = 45.0;

            vertices = std::vector<Vec3f>(4, {2.0f, 0, -2.0f});
            colors   = std::vector<Vec3f>(4, {0, 0, 1.0f});
            indices = { };
        }
        break;
        case ModelType::File:
        {
            string objectPath = findDataFile("rendering/spot.obj");

            position = Vec3f( 2.4, 0.7, 1.2);
            lookat   = Vec3f( 0.0, 0.0, 0.3);
            upVector = Vec3f( 0.0, 1.0, 0.0);

            fovy = 45.0;

            std::vector<vector<int>> indvec;
            // using per-vertex normals as colors
            loadMesh(objectPath, vertices, colors, indvec);
            for (const auto &vec : indvec)
            {
                indices.push_back({vec[0], vec[1], vec[2]});
            }

            for (auto &color : colors)
            {
                color = Vec3f(abs(color[0]), abs(color[1]), abs(color[2]));
            }
        }
        break;
        case ModelType::Clipping:
        {
            position = Vec3f(0.0, 0.0, 5.0);
            lookat   = Vec3f(0.0, 0.0, 0.0);
            upVector = Vec3f(0.0, 1.0, 0.0);

            fovy = 45.0;

            vertices =
            {
                { 2.0,  0.0, -2.0}, { 0.0, -6.0, -2.0}, {-2.0,  0.0, -2.0},
                { 3.5, -1.0, -5.0}, { 2.5, -2.5, -5.0}, {-1.0,  1.0, -5.0},
                {-6.5, -1.0, -3.0}, {-2.5, -2.0, -3.0}, { 1.0,  1.0, -5.0},
            };

            indices = { {0, 1, 2}, {3, 4, 5}, {6, 7, 8} };

            Vec3f col1(217.0, 238.0, 185.0);
            Vec3f col2(185.0, 217.0, 238.0);
            Vec3f col3(150.0,  10.0, 238.0);

            col1 *= (1.f / 255.f);
            col2 *= (1.f / 255.f);
            col3 *= (1.f / 255.f);

            colors =
            {
                col1, col2, col3,
                col2, col3, col1,
                col3, col1, col2,
            };
        }
        break;
        case ModelType::Centered:
        {
            position = Vec3f(0.0, 0.0, 5.0);
            lookat   = Vec3f(0.0, 0.0, 0.0);
            upVector = Vec3f(0.0, 1.0, 0.0);

            fovy = 45.0;

            vertices =
            {
                { 2.0,  0.0, -2.0}, { 0.0, -2.0, -2.0}, {-2.0,  0.0, -2.0},
                { 3.5, -1.0, -5.0}, { 2.5, -1.5, -5.0}, {-1.0,  0.5, -5.0},
            };

            indices = { {0, 1, 2}, {3, 4, 5} };

            Vec3f col1(217.0, 238.0, 185.0);
            Vec3f col2(185.0, 217.0, 238.0);

            col1 *= (1.f / 255.f);
            col2 *= (1.f / 255.f);

            colors =
            {
                col1, col2, col1,
                col2, col1, col2,
            };
        }
        break;
        case ModelType::Color:
        {
            position = Vec3f(0.0, 0.0, 5.0);
            lookat   = Vec3f(0.0, 0.0, 0.0);
            upVector = Vec3f(0.0, 1.0, 0.0);

            fovy = 60.0;

            vertices =
            {
                { 2.0,  0.0, -2.0},
                { 0.0,  2.0, -3.0},
                {-2.0,  0.0, -2.0},
                { 0.0, -2.0,  1.0},
            };

            indices = { {0, 1, 2}, {0, 2, 3} };

            colors =
            {
                { 0.0f, 0.0f, 1.0f},
                { 0.0f, 1.0f, 0.0f},
                { 1.0f, 0.0f, 0.0f},
                { 0.0f, 1.0f, 0.0f},
            };
        }
        break;

        default:
            break;
        }
    }

    Vec3f position;
    Vec3f lookat;
    Vec3f upVector;

    double fovy;

    std::vector<Vec3f> vertices;
    std::vector<Vec3i> indices;
    std::vector<Vec3f> colors;
};


void compareDepth(const cv::Mat& gt, const cv::Mat& mat, float zFar, float scale,
                  float maskThreshold, float normInfThreshold, float normL2Threshold)
{
    ASSERT_EQ(gt.type(), CV_16UC1);
    ASSERT_EQ(mat.type(), CV_16UC1);
    ASSERT_EQ(gt.size(), mat.size());

    Mat gtMask  = gt  < zFar*scale;
    Mat matMask = mat < zFar*scale;

    Mat diffMask = gtMask != matMask;
    int nzDepthDiff = cv::countNonZero(diffMask);
    EXPECT_LE(nzDepthDiff, maskThreshold);

    float normInfDepth = cv::norm(gt, mat, cv::NORM_INF, gtMask & matMask);
    EXPECT_LE(normInfDepth, normInfThreshold);
    float normL2Depth =  cv::norm(gt, mat, cv::NORM_L2, gtMask & matMask);
    EXPECT_LE(normL2Depth, normL2Threshold);
}


void compareRGB(const cv::Mat& gt, const cv::Mat& mat, float normInfThreshold, float normL2Threshold)
{
    ASSERT_EQ(gt.type(), CV_32FC3);
    ASSERT_EQ(mat.type(), CV_32FC3);
    ASSERT_EQ(gt.size(), mat.size());

    float normInfRgb = cv::norm(gt, mat, cv::NORM_INF);
    EXPECT_LE(normInfRgb, normInfThreshold);
    float normL2Rgb = cv::norm(gt, mat, cv::NORM_L2);
    EXPECT_LE(normL2Rgb, normL2Threshold);
}


// resolution, shading type, culling mode, model type, float type, index type
class RenderingTest : public ::testing::TestWithParam<
    std::tuple<std::tuple<int, int>, ShadingTypeEnum, CullingModeEnum, ModelTypeEnum, MatDepth, MatDepth>>
{
protected:
    void SetUp() override
    {
        auto t = GetParam();
        auto wh = std::get<0>(t);
        width = std::get<0>(wh);
        height = std::get<1>(wh);
        shadingType = std::get<1>(t);
        cullingMode = std::get<2>(t);
        modelType = std::get<3>(t);
        modelData = ModelData(modelType);
        ftype = std::get<4>(t);
        itype = std::get<5>(t);

        zNear = 0.1, zFar = 50.0;
        depthScale = 1000.0;

        depth_buf = Mat(height, width, ftype, zFar);
        color_buf = Mat(height, width, CV_MAKETYPE(ftype, 3), Scalar::all(0));

        cameraMatrix = Mat(makeCamMatrix_TODO_rewrite_it_later(modelData.position, modelData.lookat, modelData.upVector, modelData.fovy, zNear, zFar));

        verts = Mat(modelData.vertices);
        verts.convertTo(verts, ftype);

        if (shadingType != ShadingType::White)
        {
            colors = Mat(modelData.colors);
            colors.convertTo(colors, ftype);
        }

        indices = Mat(modelData.indices);
        if (itype != CV_32S)
        {
            indices.convertTo(indices, itype);
        }

        int cullIdx = (cullingMode == CullingMode::None) ? 0 :
                      ((cullingMode == CullingMode::CW) ? 1 :
                      ((cullingMode == CullingMode::CCW) ? 2 : -1));
        triangleRasterize(verts, indices, colors, cameraMatrix, width, height,
                          (shadingType == ShadingType::Shaded), cullIdx, depth_buf, color_buf);
    }

    Matx43f makeCamMatrix_TODO_rewrite_it_later(Vec3f position, Vec3f lookat, Vec3f upVector, double fovy, double znear, double zfar)
    {
        Matx43f m;
        m(0, 0) = position(0); m(0, 1) = position(1); m(0, 2) = position(2);
        m(1, 0) = lookat  (0); m(1, 1) = lookat  (1); m(1, 2) = lookat  (2);
        m(2, 0) = upVector(0); m(2, 1) = upVector(1); m(2, 2) = upVector(2);
        m(3, 0) = fovy;        m(3, 1) = znear;       m(3, 2) = zfar;

        return m;
    }

public:
    int width, height;
    double zNear, zFar, depthScale;

    Mat depth_buf, color_buf;

    Mat verts, colors, indices;
    Mat cameraMatrix;

    ModelData modelData;
    ModelTypeEnum modelType;
    ShadingTypeEnum shadingType;
    CullingModeEnum cullingMode;
    int ftype, itype;
};


TEST_P(RenderingTest, noArrays)
{
    Mat depthOnly, colorOnly;
    int cullIdx = (cullingMode == CullingMode::None) ? 0 :
                      ((cullingMode == CullingMode::CW) ? 1 :
                      ((cullingMode == CullingMode::CCW) ? 2 : -1));
    triangleRasterize(verts, indices, colors, cameraMatrix, width, height,
                      (shadingType == ShadingType::Shaded), cullIdx, depthOnly, cv::noArray());
    triangleRasterize(verts, indices, colors, cameraMatrix, width, height,
                      (shadingType == ShadingType::Shaded), cullIdx, cv::noArray(), colorOnly);

    compareRGB(color_buf, colorOnly, 0, 0);
    compareDepth(depth_buf, depthOnly, zFar, depthScale, 0, 0, 0);
}


TEST_P(RenderingTest, accuracy)
{
    cvtColor(color_buf, color_buf, cv::COLOR_RGB2BGR);
    cv::flip(color_buf, color_buf, 0);
    depth_buf.convertTo(depth_buf, CV_16U, depthScale);
    cv::flip(depth_buf, depth_buf, 0);

    if (modelType == ModelType::Empty)
    {
        std::vector<Mat> channels(3);
        split(color_buf, channels);

        for (int i = 0; i < 3; i++)
        {
            EXPECT_EQ(countNonZero(channels[i]), 0);
        }
    }
    else
    {
        std::string path = findDataDirectory("rendering");

        std::string modelName;
        {
            std::stringstream ss;
            modelType.PrintTo(&ss);
            ss >> modelName;
        }
        std::string shadingName;
        {
            std::stringstream ss;
            shadingType.PrintTo(&ss);
            ss >> shadingName;
        }
        std::string cullingName;
        {
            std::stringstream ss;
            cullingMode.PrintTo(&ss);
            ss >> cullingName;
        }

        //TODO: cv::format()
        std::string widthStr  = std::to_string(width);
        std::string heightStr = std::to_string(height);

        std::string suffix = modelName + "_" + widthStr + "x" + heightStr+"_Cull" + cullingName;
        std::string gtPathColor = path + "/example_image_" + suffix + "_" + shadingName + ".png";
        std::string gtPathDepth = path + "/depth_image_"   + suffix + ".png";

        Mat groundTruthColor = imread(gtPathColor);
        groundTruthColor.convertTo(groundTruthColor, CV_32F);
        compareRGB(groundTruthColor, color_buf, 0, 0);

        Mat groundTruthDepth = imread(gtPathDepth, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
        compareDepth(groundTruthDepth, depth_buf, zFar, depthScale, 0, 0, 0);

        // add --test_debug to output resulting images
        if (debugLevel > 0)
        {
            std::string outColorPath = "color_image_" + suffix + "_" + shadingName + ".png";
            std::string outDepthPath = "depth_image_" + suffix + "_" + shadingName + ".png";
            imwrite(outColorPath, color_buf * 255.f);
            imwrite(outDepthPath, depth_buf);
        }
    }
}


// drawing model as a whole or as two halves should give the same result
TEST_P(RenderingTest, keepDrawnData)
{
    if (modelType != ModelType::Empty)
    {
        // should be initialized inside rasterization
        Mat depth_buf2, color_buf2;

        Mat idx1, idx2;
        int nTriangles = indices.total();
        idx1 = indices.reshape(3, 1)(Range::all(), Range(0, nTriangles / 2));
        idx2 = indices.reshape(3, 1)(Range::all(), Range(nTriangles / 2, nTriangles));

        int cullIdx = (cullingMode == CullingMode::None) ? 0 :
                      ((cullingMode == CullingMode::CW) ? 1 :
                      ((cullingMode == CullingMode::CCW) ? 2 : -1));
        triangleRasterize(verts, idx1, colors, cameraMatrix, width, height, (shadingType == ShadingType::Shaded), cullIdx, depth_buf2, color_buf2);
        triangleRasterize(verts, idx2, colors, cameraMatrix, width, height, (shadingType == ShadingType::Shaded), cullIdx, depth_buf2, color_buf2);

        compareRGB(color_buf, color_buf2, 0, 0);
        compareDepth(depth_buf, depth_buf2, zFar, depthScale, 0, 0, 0);
    }
}

INSTANTIATE_TEST_CASE_P(Rendering, RenderingTest, ::testing::Combine(
    ::testing::Values(std::make_tuple(700, 700), std::make_tuple(640, 480)),
    ShadingTypeEnum::all(),
    CullingModeEnum::all(),
    ModelTypeEnum::all(),
    // float type
    //::testing::Values(CV_32F, CV_64F), // not supported yet
    ::testing::Values(CV_32F), // float type
    // index type
    //::testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32U, CV_32S) // not supported yet
    ::testing::Values(CV_32S) // not supported yet
));

}
}
