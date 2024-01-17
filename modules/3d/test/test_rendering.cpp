// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace opencv_test { namespace {
using namespace cv;

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
        if (v >= 0 && v < (int)vals.size())
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
        if (v >= 0 && v < (int)vals.size())
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

    Mat jointMask = gtMask & matMask;
    int nzJointMask = cv::countNonZero(jointMask);
    float normInfDepth = cv::norm(gt, mat, cv::NORM_INF, jointMask);
    EXPECT_LE(normInfDepth, normInfThreshold);
    float normL2Depth = nzJointMask ? cv::norm(gt, mat, cv::NORM_L2, jointMask) / nzJointMask : 0;
    EXPECT_LE(normL2Depth, normL2Threshold);
}


void compareRGB(const cv::Mat& gt, const cv::Mat& mat, float normInfThreshold, float normL2Threshold)
{
    ASSERT_EQ(gt.type(), CV_32FC3);
    ASSERT_EQ(mat.type(), CV_32FC3);
    ASSERT_EQ(gt.size(), mat.size());

    float normInfRgb = cv::norm(gt, mat, cv::NORM_INF);
    EXPECT_LE(normInfRgb, normInfThreshold);
    float normL2Rgb = cv::norm(gt, mat, cv::NORM_L2) / gt.total();
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

        cameraPose = lookAtMatrixCal(modelData.position, modelData.lookat, modelData.upVector);
        fovYradians = modelData.fovy / 180.f * CV_PI;

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

        settings = RasterizeSettings().setCullingMode(cullingMode).setShadingType(shadingType);

        triangleRasterize(verts, indices, colors, cameraPose, fovYradians, zNear, zFar,
                          width, height, settings, depth_buf, color_buf);
    }

public:
    int width, height;
    double zNear, zFar, depthScale;

    Mat depth_buf, color_buf;

    Mat verts, colors, indices;
    Matx44f cameraPose;
    double fovYradians;
    RasterizeSettings settings;

    ModelData modelData;
    ModelTypeEnum modelType;
    ShadingTypeEnum shadingType;
    CullingModeEnum cullingMode;
    int ftype, itype;
};


TEST_P(RenderingTest, noArrays)
{
    Mat depthOnly, colorOnly;

    // cameraPose can also be double, checking it
    triangleRasterize(verts, indices, colors, Matx44d(cameraPose), fovYradians, zNear, zFar,
                      width, height, settings, depthOnly, cv::noArray());
    triangleRasterize(verts, indices, colors, Matx44d(cameraPose), fovYradians, zNear, zFar,
                      width, height, settings, cv::noArray(), colorOnly);

    compareRGB(color_buf, colorOnly, 1, 0.00134);
    depth_buf.convertTo(depth_buf, CV_16U, depthScale);
    depthOnly.convertTo(depthOnly, CV_16U, depthScale);
    compareDepth(depth_buf, depthOnly, zFar, depthScale, 0, 0, 0);
}


TEST_P(RenderingTest, accuracy)
{
    cvtColor(color_buf, color_buf, cv::COLOR_RGB2BGR);
    depth_buf.convertTo(depth_buf, CV_16U, depthScale);

    if (modelType == ModelType::Empty ||
       (modelType == ModelType::Centered && cullingMode == CullingMode::CW) ||
       (modelType == ModelType::Color    && cullingMode == CullingMode::CCW))
    {
        std::vector<Mat> channels(3);
        split(color_buf, channels);

        for (int i = 0; i < 3; i++)
        {
            EXPECT_EQ(countNonZero(channels[i]), 0);
        }

        Mat depthDiff;
        absdiff(depth_buf, Scalar(zFar * depthScale), depthDiff);
        float sumDepthDiff = sum(depthDiff)[0];
        EXPECT_EQ(sumDepthDiff, 0);
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

        std::string suffix = cv::format("%s_%dx%d_Cull%s", modelName.c_str(), width, height, cullingName.c_str());
        std::string gtPathColor = path + "/example_image_" + suffix + "_" + shadingName + ".png";
        std::string gtPathDepth = path + "/depth_image_"   + suffix + ".png";

        Mat groundTruthColor = imread(gtPathColor);
        groundTruthColor.convertTo(groundTruthColor, CV_32F, (1.f / 255.f));
        compareRGB(groundTruthColor, color_buf, 1, 3.04e-05);

        Mat groundTruthDepth = imread(gtPathDepth, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
        compareDepth(groundTruthDepth, depth_buf, zFar, depthScale, 29, 485, 0.00681);

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

        triangleRasterize(verts, idx1, colors, cameraPose, fovYradians, zNear, zFar, width, height, settings, depth_buf2, color_buf2);
        triangleRasterize(verts, idx2, colors, cameraPose, fovYradians, zNear, zFar, width, height, settings, depth_buf2, color_buf2);

        compareRGB(color_buf, color_buf2, 0, 0);
        depth_buf.convertTo(depth_buf, CV_16U, depthScale);
        depth_buf2.convertTo(depth_buf2, CV_16U, depthScale);
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
