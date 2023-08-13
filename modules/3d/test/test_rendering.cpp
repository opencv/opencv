#include "test_precomp.hpp"

namespace opencv_test { namespace {
using namespace cv;

void AssertMatsEqual(const cv::Mat& mat1, const cv::Mat& mat2) {
    ASSERT_EQ(mat1.size(), mat2.size()); // Check if sizes are equal
    ASSERT_EQ(mat1.type(), mat2.type()); // Check if types are equal

    // Check if the matrices have the same content
    cv::Mat diff;
    cv::compare(mat1, mat2, diff, cv::CMP_NE);
    int nonZeroElements = cv::countNonZero(diff.reshape(1));

    EXPECT_LT(nonZeroElements, 1000);
}

class RenderingTest : public ::testing::TestWithParam<std::tuple<int, int>>
{
protected:
    void SetUp() override
    {
        auto t = GetParam();
        width = std::get<0>(t), height = std::get<1>(t);

        position = Vec3f(0.0, 0.0, 5.0);
        lookat = Vec3f(0.0, 0.0, 0.0);
        upVector = Vec3f(0.0, 1.0, 0.0);

        fovy = 45.0;
        zNear = 0.1, zFar = 50;
        shadingMode = false;
    }

public:
    int width, height;
    Vec3f position;
    Vec3f lookat;
    Vec3f upVector;
    float zNear, zFar, fovy;
    bool shadingMode;
};

TEST_P(RenderingTest, depthRenderingTest)
{
    Mat depth_buf(height, width, CV_32F, zFar);
    Mat color_buf(height, width, CV_32FC3, Scalar(0.0, 0.0, 0.0));
    //position = Vec3f(0.0, 0.5, 5.0);

    std::vector <Vec3f> vertices = {
        Vec3f(2.0, 0, -2.0),
        Vec3f(0, -2, -2),
        Vec3f(-2, 0, -2),
        Vec3f(3.5, -1, -5),
        Vec3f(2.5, -1.5, -5),
        Vec3f(-1, 0.5, -5)
    };

    std::vector<Vec3i> indices = {
        Vec3i(0, 1, 2),
        Vec3i(3, 4, 5)
    };

    std::vector<Vec3f> colors = {
        Vec3f(217.0, 238.0, 185.0),
        Vec3f(217.0, 238.0, 185.0),
        Vec3f(217.0, 238.0, 185.0),
        Vec3f(185.0, 217.0, 238.0),
        Vec3f(185.0, 217.0, 238.0),
        Vec3f(185.0, 217.0, 238.0)
    };

    Mat cameraMatrix(4, 3, CV_32F);

    cameraMatrix.row(0) = Mat(position).t();
    cameraMatrix.row(1) = Mat(lookat).t();
    cameraMatrix.row(2) = Mat(upVector).t();
    cameraMatrix.row(3) = Mat(Vec3f(fovy, zNear, zFar)).t();

    triangleRasterize(vertices, indices, colors, cameraMatrix, width, height,
        shadingMode, depth_buf, color_buf);

    color_buf.convertTo(color_buf, CV_8UC3, 1.0f);
    cvtColor(color_buf, color_buf, cv::COLOR_RGB2BGR);
    cv::flip(color_buf, color_buf, 0);

    depth_buf.convertTo(depth_buf, CV_8UC1, 1.0);
    cv::flip(depth_buf, depth_buf, 0);

    if (width == 700)
    {
        if (debugLevel > 0)
        {
            Mat groundTruth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/example_image_depth_1.png");
            AssertMatsEqual(color_buf, groundTruth);

            Mat groundTruthDepth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/depth_image_depth_1.png", cv::IMREAD_GRAYSCALE);
            AssertMatsEqual(depth_buf, groundTruthDepth);

            imwrite("temp_image.png", color_buf);
            imwrite("constant_image_depth", depth_buf);
        }
        else
        {
            Mat groundTruth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/example_image_depth_1.png");
            AssertMatsEqual(color_buf, groundTruth);

            Mat groundTruthDepth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/depth_image_depth_1.png", cv::IMREAD_GRAYSCALE);
            AssertMatsEqual(depth_buf, groundTruthDepth);
        }
    }
    else
    {
        if (debugLevel > 0)
        {
            Mat groundTruth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/example_image_depth_2.png");
            AssertMatsEqual(color_buf, groundTruth);

            Mat groundTruthDepth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/depth_image_depth_2.png", cv::IMREAD_GRAYSCALE);
            AssertMatsEqual(depth_buf, groundTruthDepth);

            imwrite("temp_image_cam.png", color_buf);
            imwrite("constant_image_depth_cam", depth_buf);
        }
        else
        {
            Mat groundTruth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/example_image_depth_2.png");
            AssertMatsEqual(color_buf, groundTruth);

            Mat groundTruthDepth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/depth_image_depth_2.png", cv::IMREAD_GRAYSCALE);
            AssertMatsEqual(depth_buf, groundTruthDepth);
        }
    }
    
}

TEST_P(RenderingTest, clippingTest)
{
    Mat depth_buf(height, width, CV_32F, zFar);
    Mat color_buf(height, width, CV_32FC3, Scalar(0.0, 0.0, 0.0));

    std::vector <Vec3f> vertices = {
        Vec3f(2.0, 0, -2.0),
        Vec3f(0, -6, -2),
        Vec3f(-2, 0, -2),
        Vec3f(3.5, -1, -5),
        Vec3f(2.5, -2.5, -5),
        Vec3f(-1, 1, -5),
        Vec3f(-6.5, -1, -3),
        Vec3f(-2.5, -2, -3),
        Vec3f(1, 1, -5)
    };

    std::vector<Vec3i> indices = {
        Vec3i(0, 1, 2),
        Vec3i(3, 4, 5),
        Vec3i(6, 7, 8)
    };

    std::vector<Vec3f> colors = {
        Vec3f(217.0, 238.0, 185.0),
        Vec3f(217.0, 238.0, 185.0),
        Vec3f(217.0, 238.0, 185.0),
        Vec3f(185.0, 217.0, 238.0),
        Vec3f(185.0, 217.0, 238.0),
        Vec3f(185.0, 217.0, 238.0),
        Vec3f(150.0, 10.0, 238.0),
        Vec3f(150.0, 10.0, 238.0),
        Vec3f(150.0, 10.0, 238.0)
    };

    Mat cameraMatrix(4, 3, CV_32F);

    cameraMatrix.row(0) = Mat(position).t();
    cameraMatrix.row(1) = Mat(lookat).t();
    cameraMatrix.row(2) = Mat(upVector).t();
    cameraMatrix.row(3) = Mat(Vec3f(fovy, zNear, zFar)).t();

    triangleRasterize(vertices, indices, colors, cameraMatrix, width, height,
        shadingMode, depth_buf, color_buf);

    color_buf.convertTo(color_buf, CV_8UC3, 1.0f);
    cvtColor(color_buf, color_buf, cv::COLOR_RGB2BGR);
    cv::flip(color_buf, color_buf, 0);

    depth_buf.convertTo(depth_buf, CV_8UC1, 1.0);
    cv::flip(depth_buf, depth_buf, 0);

    if (width == 700)
    {
        if (debugLevel > 0)
        {
            Mat groundTruth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/example_image_clipping_1.png");
            AssertMatsEqual(color_buf, groundTruth);

            Mat groundTruthDepth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/depth_image_clipping_1.png", cv::IMREAD_GRAYSCALE);
            AssertMatsEqual(depth_buf, groundTruthDepth);

            imwrite("temp_image_clipping.png", color_buf);
            imwrite("clipping_image_depth_cam.png", depth_buf);
        }
        else
        {
            Mat groundTruth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/example_image_clipping_1.png");
            AssertMatsEqual(color_buf, groundTruth);

            Mat groundTruthDepth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/depth_image_clipping_1.png", cv::IMREAD_GRAYSCALE);
            AssertMatsEqual(depth_buf, groundTruthDepth);
        }
    }
    else
    {
        if (debugLevel > 0)
        {
            Mat groundTruth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/example_image_clipping_2.png");
            AssertMatsEqual(color_buf, groundTruth);

            Mat groundTruthDepth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/depth_image_clipping_2.png", cv::IMREAD_GRAYSCALE);
            AssertMatsEqual(depth_buf, groundTruthDepth);

            imwrite("temp_image_clipping_cam.png", color_buf);
            imwrite("clipping_image_depth_cam.png", depth_buf);
        }
        else
        {
            Mat groundTruth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/example_image_clipping_2.png");
            AssertMatsEqual(color_buf, groundTruth);

            Mat groundTruthDepth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/depth_image_clipping_2.png", cv::IMREAD_GRAYSCALE);
            AssertMatsEqual(depth_buf, groundTruthDepth);
        }
    }
}

TEST_P(RenderingTest, colorRenderingTest)
{
    Mat depth_buf(height, width, CV_32F, zFar);
    Mat color_buf(height, width, CV_32FC3, Scalar(0.0, 0.0, 0.0));

    shadingMode = false;
    position = Vec3f(0, 0, 5.0);
    lookat = Vec3f(0.0, 0.0, 0.0);
    fovy = 60.0;

    Mat cameraMatrix(4, 3, CV_32F);

    cameraMatrix.row(0) = Mat(position).t();
    cameraMatrix.row(1) = Mat(lookat).t();
    cameraMatrix.row(2) = Mat(upVector).t();
    cameraMatrix.row(3) = Mat(Vec3f(fovy, zNear, zFar)).t();

    std::vector <Vec3f> vertices = {
        Vec3f(2.0, 0, -2.0),
        //Vec3f(0, 2, -2),
        Vec3f(0, 2, -3),
        Vec3f(-2, 0, -2),
        Vec3f(0, -2, 1)
        //Vec3f(0, -2, -2)
    };

    std::vector<Vec3i> indices = {
        Vec3i(0, 1, 2),
        Vec3i(0, 2, 3)
    };

    std::vector<Vec3f> colors = {
        Vec3f(0.0f, 0.0f, 255.0f),  Vec3f(0.0f, 255.0f, 0.0f),  Vec3f(255.0f, 0.0f, 0.0f), Vec3f(0.0f, 255.0f, 0.0f)
    };

    triangleRasterize(vertices, indices, colors, cameraMatrix, width, height,
        shadingMode, depth_buf, color_buf);

    color_buf.convertTo(color_buf, CV_8UC3, 1.0f);
    cvtColor(color_buf, color_buf, cv::COLOR_RGB2BGR);
    cv::flip(color_buf, color_buf, 0);

    depth_buf.convertTo(depth_buf, CV_8UC1, 1.0);
    cv::flip(depth_buf, depth_buf, 0);

    if (width == 700)
    {
        if (debugLevel > 0)
        {
            Mat groundTruth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/example_image_color_1.png");
            AssertMatsEqual(color_buf, groundTruth);

            Mat groundTruthDepth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/depth_image_color_1.png", cv::IMREAD_GRAYSCALE);
            AssertMatsEqual(depth_buf, groundTruthDepth);

            imwrite("temp_image_color.png", color_buf);
            imwrite("color_image_depth.png", depth_buf);
        }
        else
        {
            Mat groundTruth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/example_image_color_1.png");
            AssertMatsEqual(color_buf, groundTruth);

            Mat groundTruthDepth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/depth_image_color_1.png", cv::IMREAD_GRAYSCALE);
            AssertMatsEqual(depth_buf, groundTruthDepth);
        }
    }
    else
    {
        if (debugLevel > 0)
        {
            Mat groundTruth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/example_image_color_2.png");
            AssertMatsEqual(color_buf, groundTruth);

            Mat groundTruthDepth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/depth_image_color_2.png", cv::IMREAD_GRAYSCALE);
            AssertMatsEqual(depth_buf, groundTruthDepth);

            imwrite("temp_image_color_cam.png", color_buf);
            imwrite("color_image_depth.png", depth_buf);
        }
        else
        {
            Mat groundTruth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/example_image_color_2.png");
            AssertMatsEqual(color_buf, groundTruth);

            Mat groundTruthDepth = imread("../../../opencv_extra/opencv_extra/testdata/rendering/depth_image_color_2.png", cv::IMREAD_GRAYSCALE);
            AssertMatsEqual(depth_buf, groundTruthDepth);
        }
    }
}

TEST_P(RenderingTest, emptyIndiceTest)
{
    Mat depth_buf(height, width, CV_32F, zFar);
    Mat color_buf(height, width, CV_32FC3, Scalar(0.0, 0.0, 0.0));

    std::vector <Vec3f> vertices;
    std::vector<Vec3i> indices;
    std::vector<Vec3f> colors;

    Mat cameraMatrix(4, 3, CV_32F);

    cameraMatrix.row(0) = Mat(position).t();
    cameraMatrix.row(1) = Mat(lookat).t();
    cameraMatrix.row(2) = Mat(upVector).t();
    cameraMatrix.row(3) = Mat(Vec3f(fovy, zNear, zFar)).t();

    triangleRasterize(vertices, indices, colors, cameraMatrix, width, height,
        shadingMode, depth_buf, color_buf);

    color_buf.convertTo(color_buf, CV_8UC3, 1.0f);
    std::vector<Mat> channels(3);
    split(color_buf, channels);

    for (int i = 0; i < 3; i++)
    {
        ASSERT_EQ(countNonZero(channels[i]), 0);
    }
}

INSTANTIATE_TEST_CASE_P(Rendering, RenderingTest, ::testing::Values(
    std::make_tuple(700, 700),
    std::make_tuple(640, 480)
));

}
}
