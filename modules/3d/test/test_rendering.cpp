#include "test_precomp.hpp"

namespace opencv_test { namespace {
using namespace cv;

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
            colors   = std::vector<Vec3f>(4, {0, 0, 255.0f});
            indices = { };
        }
        break;
        case ModelType::File:
        {
            string objectPath = "../../../opencv_extra/opencv_extra/testdata/rendering/model/spot.obj";

            position = Vec3f(20.0, 60.0,  40.0);
            lookat   = Vec3f( 0.0,  0.0,   0.0);
            upVector = Vec3f( 0.0,  1.0,   0.0);

            fovy = 45.0;

            std::vector<vector<int>> indvec;
            loadMesh(objectPath, vertices, colors, indvec);
            for (const auto &vec : indvec)
            {
                indices.push_back({vec[0], vec[1], vec[2]});
            }

            for (auto &color : colors)
            {
                color = Vec3f(abs(color[0]), abs(color[1]), abs(color[2])) * 255.0f;
            }
        }
        break;
        case ModelType::Clipping:
        {
            position = Vec3f(0.0, 0.0,  5.0);
            lookat   = Vec3f(0.0, 0.0,  0.0);
            upVector = Vec3f(0.0, 1.0,  0.0);

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
            colors =
            {
                col1, col1, col1,
                col2, col2, col2,
                col3, col3, col3,
            };
        }
        break;
        case ModelType::Centered:
        {
            position = Vec3f(0.0, 0.0,  5.0);
            lookat   = Vec3f(0.0, 0.0,  0.0);
            upVector = Vec3f(0.0, 1.0,  0.0);

            fovy = 45.0;

            vertices =
            {
                { 2.0,  0.0, -2.0}, { 0.0, -2.0, -2.0}, {-2.0,  0.0, -2.0},
                { 3.5, -1.0, -5.0}, { 2.5, -1.5, -5.0}, {-1.0,  0.5, -5.0},
            };

            indices = { {0, 1, 2}, {3, 4, 5} };

            Vec3f col1(217.0, 238.0, 185.0);
            Vec3f col2(185.0, 217.0, 238.0);
            colors =
            {
                col1, col1, col1,
                col2, col2, col2,
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
                {  0.0f,   0.0f, 255.0f},
                {  0.0f, 255.0f,   0.0f},
                {255.0f,   0.0f,   0.0f},
                {  0.0f, 255.0f,   0.0f},
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


void AssertMatsEqual(const cv::Mat& mat1, const cv::Mat& mat2, int threshold)
{
    ASSERT_EQ(mat1.size(), mat2.size());
    ASSERT_EQ(mat1.type(), mat2.type());

    // Check if the matrices have the same content
    cv::Mat diff;
    cv::compare(mat1, mat2, diff, cv::CMP_NE);
    int nonZeroElements = cv::countNonZero(diff);

    EXPECT_LT(nonZeroElements, threshold);
}


// resolution, shading type, model type, float type, index type
class RenderingTest : public ::testing::TestWithParam<std::tuple<std::tuple<int, int>, ShadingTypeEnum, ModelTypeEnum, int, int>>
{
protected:
    void SetUp() override
    {
        auto t = GetParam();
        auto wh = std::get<0>(t);
        width = std::get<0>(wh);
        height = std::get<1>(wh);
        shadingType = std::get<1>(t);
        modelType = std::get<2>(t);
        modelData = ModelData(modelType);
        ftype = std::get<3>(t);
        itype = std::get<4>(t);

        zNear = 0.1, zFar = 50;

        depth_buf = Mat(height, width, ftype, zFar);
        color_buf = Mat(height, width, CV_MAKETYPE(ftype, 3), Scalar(0.0, 0.0, 0.0));

        cameraMatrix = Mat(makeCamMatrix_TODO_rewrite_it_later(modelData.position, modelData.lookat, modelData.upVector, modelData.fovy, zNear, zFar));

        verts = Mat(modelData.vertices);
        verts.convertTo(verts, ftype);

        if (shadingType != ShadingType::White)
        {
            colors = Mat(modelData.vertices);
            colors.convertTo(colors, ftype);
        }

        indices = Mat(modelData.indices);
        if (itype != CV_32S)
        {
            indices.convertTo(indices, itype);
        }

        triangleRasterize(verts, indices, colors, cameraMatrix, width, height,
                          (shadingType == ShadingType::Shaded), depth_buf, color_buf);
    }

    Matx43f makeCamMatrix_TODO_rewrite_it_later(Vec3f position, Vec3f lookat, Vec3f upVector, double fovy, double znear, double zfar)
    {
        Matx43f m;
        m(0, 0) = position(0); m(0, 1) = position(1); m(0, 2) = position(2);
        m(1, 0) = lookat  (0); m(1, 1) = lookat  (1); m(1, 2) = lookat  (2);
        m(2, 0) = upVector(0); m(2, 1) = upVector(1); m(2, 2) = upVector(2);
        m(3, 0) = fovy;        m(3, 1) = znear;       m(3, 3) = zfar;

        return m;
    }

public:
    int width, height;
    double zNear, zFar;

    Mat depth_buf, color_buf;

    Mat verts, colors, indices;
    Mat cameraMatrix;

    ModelData modelData;
    ModelTypeEnum modelType;
    ShadingTypeEnum shadingType;
    int ftype, itype;
};


TEST_P(RenderingTest, noArrays)
{
    Mat depthOnly, colorOnly;
    triangleRasterize(verts, indices, colors, cameraMatrix, width, height,
                      (shadingType == ShadingType::Shaded), depthOnly, cv::noArray());
    triangleRasterize(verts, indices, colors, cameraMatrix, width, height,
                      (shadingType == ShadingType::Shaded), cv::noArray(), colorOnly);

    // TODO: tune this threshold
    AssertMatsEqual(color_buf, colorOnly, 1000);
    // TODO: tune this threshold
    AssertMatsEqual(depth_buf, depthOnly, 1000);
}


TEST_P(RenderingTest, accuracy)
{
    color_buf.convertTo(color_buf, CV_8UC3, 1.0f);
    cvtColor(color_buf, color_buf, cv::COLOR_RGB2BGR);
    cv::flip(color_buf, color_buf, 0);
    depth_buf.convertTo(depth_buf, CV_8UC1, 1.0);
    cv::flip(depth_buf, depth_buf, 0);

    if (modelType == ModelType::Empty)
    {
        std::vector<Mat> channels(3);
        split(color_buf, channels);

        for (int i = 0; i < 3; i++)
        {
            ASSERT_EQ(countNonZero(channels[i]), 0);
        }
    }
    else
    {
        std::string path = "../../../opencv_extra/opencv_extra/testdata/rendering/";

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

        std::string widthStr  = std::to_string(width);
        std::string heightStr = std::to_string(height);

        std::string suffix = modelName + "_" + widthStr + "x" + heightStr;
        std::string gtPathColor = path + "/example_image_" + suffix + "_" + shadingName + ".png";
        std::string gtPathDepth = path + "/depth_image_"   + suffix + ".png";

        Mat groundTruthColor = imread(gtPathColor);
        //TODO: tune this threshold
        AssertMatsEqual(color_buf, groundTruthColor, 1000);

        Mat groundTruthDepth = imread(gtPathDepth, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH) * (1. / 1000.0);
        //TODO: tune this threshold
        AssertMatsEqual(depth_buf, groundTruthDepth, 1000);

        if (debugLevel > 0)
        {
            std::string outColorPath = "color_image_" + suffix + "_" + shadingName + ".png";
            std::string outDepthPath = "depth_image_" + suffix + "_" + shadingName + ".png";
            imwrite(outColorPath, color_buf);
            imwrite(outDepthPath, depth_buf);
        }
    }
}


// drawing model as a whole or as two halves should give the same result
TEST_P(RenderingTest, keepDrawnData)
{
    // should be initialized inside rasterization
    Mat depth_buf2, color_buf2;

    Mat idx1, idx2;
    int nverts = indices.total();
    idx1 = indices.reshape(3, 1)(Range::all(), Range(0, nverts/2));
    idx2 = indices.reshape(3, 1)(Range::all(), Range(nverts/2, nverts));
    idx2 = idx2 - Scalar(nverts/2, nverts/2, nverts/2);

    Mat verts1, verts2, colors1, colors2;
    verts1 = verts.reshape(3, 1)(Range::all(), Range(0, nverts/2));
    verts2 = verts.reshape(3, 1)(Range::all(), Range(nverts/2, nverts));
    colors1 = colors.reshape(3, 1)(Range::all(), Range(0, nverts/2));
    colors2 = colors.reshape(3, 1)(Range::all(), Range(nverts/2, nverts));

    triangleRasterize(verts1, idx1, colors1, cameraMatrix, width, height, (shadingType == ShadingType::Shaded), depth_buf2, color_buf2);
    triangleRasterize(verts2, idx2, colors2, cameraMatrix, width, height, (shadingType == ShadingType::Shaded), depth_buf2, color_buf2);

    //TODO: tune this threshold
    AssertMatsEqual(color_buf, color_buf2, 1000);

    //TODO: tune this threshold
    AssertMatsEqual(depth_buf, depth_buf2, 1000);
}

INSTANTIATE_TEST_CASE_P(Rendering, RenderingTest, ::testing::Combine(
    ::testing::Values(std::make_tuple(700, 700), std::make_tuple(640, 480)),
    ShadingTypeEnum::all(),
    ModelTypeEnum::all(),
    ::testing::Values(CV_32F, CV_64F), // float type
    ::testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32U, CV_32S) // index type
));

}
}
