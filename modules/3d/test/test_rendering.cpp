#include "test_precomp.hpp"

namespace opencv_test { namespace {
using namespace cv;

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
        isConstant = false;
    }

public:
    int width, height;
    Vec3f position;
    Vec3f lookat;
    Vec3f upVector;
    float zNear, zFar, fovy;
    bool isConstant;
};

TEST_P(RenderingTest, depthRenderingTest)
{
    std::vector<float> depth_buf;
    std::vector<Vec3f> color_buf;

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

    triangleRasterize(vertices, indices, colors, position, lookat, upVector, fovy, zNear, zFar, width, height,
        isConstant, depth_buf, color_buf);

    Mat image(height, width, CV_32FC3, color_buf.data());
    image.convertTo(image, CV_8UC3, 1.0f);
    cvtColor(image, image, cv::COLOR_RGB2BGR);

    if (width == 700)
        imwrite("temp_image.png", image);
    else
        imwrite("temp_image_cam.png", image);
}

TEST_P(RenderingTest, depthPlaneRenderingTest)
{
    std::vector<float> depth_buf;
    std::vector<Vec3f> color_buf;

    std::vector <Vec3f> vertices = {
        Vec3f(3.5, -1, 0.0),
        Vec3f(2.5, -1.5, 3.0),
        Vec3f(-1, 0.5, 6.0)
    };

    std::vector<Vec3i> indices = {
        Vec3i(0, 1, 2)
    };

    std::vector<Vec3f> colors = {
        Vec3f(217.0, 238.0, 185.0),
        Vec3f(217.0, 238.0, 185.0),
        Vec3f(217.0, 238.0, 185.0)
    };

    triangleRasterize(vertices, indices, colors, position, lookat, upVector, fovy, zNear, zFar, width, height,
        isConstant, depth_buf, color_buf);

    Mat image(height, width, CV_32FC3, color_buf.data());
    image.convertTo(image, CV_8UC3, 1.0f);
    cvtColor(image, image, cv::COLOR_RGB2BGR);

    if (width == 700)
        imwrite("temp_image_plane.png", image);
    else
        imwrite("temp_image_plane_cam.png", image);
}

TEST_P(RenderingTest, clippingTest)
{
    std::vector<float> depth_buf;
    std::vector<Vec3f> color_buf;

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

    triangleRasterize(vertices, indices, colors, position, lookat, upVector, fovy, zNear, zFar, width, height,
        isConstant, depth_buf, color_buf);

    Mat image(height, width, CV_32FC3, color_buf.data());
    image.convertTo(image, CV_8UC3, 1.0f);
    cvtColor(image, image, cv::COLOR_RGB2BGR);
    if(width == 700)
        imwrite("temp_multiple_image.png", image);
    else
        imwrite("temp_multiple_image_cam.png", image);
}

TEST_P(RenderingTest, colorRenderingTest)
{
    std::vector<float> depth_buf;
    std::vector<Vec3f> color_buf;
    isConstant = false;
    position = Vec3f(0.0, 0.5, 5.0);
    fovy = 60.0;

    std::vector <Vec3f> vertices = {
        Vec3f(2.0, 0, -2.0),
        Vec3f(0, 2, -3),
        Vec3f(-2, 0, -2),
        Vec3f(0, -2, -1)
    };

    std::vector<Vec3i> indices = {
        Vec3i(0, 1, 2),
        Vec3i(0, 2, 3)
    };

    std::vector<Vec3f> colors = {
        Vec3f(0.0f, 0.0f, 255.0f),  Vec3f(0.0f, 255.0f, 0.0f),  Vec3f(255.0f, 0.0f, 0.0f), Vec3f(0.0f, 255.0f, 0.0f)
    };

    triangleRasterize(vertices, indices, colors, position, lookat, upVector, fovy, zNear, zFar, width, height,
        isConstant, depth_buf, color_buf);

    Scalar blue_color(255, 0, 0);
    /*bool flag = false;
    for (auto color : color_buf)
    {
        if (color[0] != 0.0 || color[1] != 0.0 || color[2] != 0.0)
            flag = true;
    }*/
    Mat image(height, width, CV_32FC3, color_buf.data());
    //Mat image(width, height, CV_32FC3, blue_color);
    image.convertTo(image, CV_8UC3, 1.0f);
    cvtColor(image, image, cv::COLOR_RGB2BGR);

    if (width == 700)
        imwrite("temp_image_color.png", image);
    else
        imwrite("temp_image_color_cam.png", image);
}

TEST_P(RenderingTest, emptyIndiceTest)
{
    std::vector<float> depth_buf;
    std::vector<Vec3f> color_buf;

    std::vector <Vec3f> vertices = {};
    std::vector<Vec3i> indices = {};
    std::vector<Vec3f> colors = {};

    triangleRasterize(vertices, indices, colors, position, lookat, upVector, fovy, zNear, zFar, width, height,
        isConstant, depth_buf, color_buf);

    Mat image(height, width, CV_32FC3, color_buf.data());
    image.convertTo(image, CV_8UC3, 1.0f);
    cvtColor(image, image, cv::COLOR_RGB2BGR);

    if (width == 700)
        imwrite("temp_empty_set_rendering.png", image);
    else
        imwrite("temp_empty_set_rendering_cam.png", image);
}

INSTANTIATE_TEST_CASE_P(Rendering, RenderingTest, ::testing::Values(
    std::make_tuple(700, 700),
    std::make_tuple(640, 480)
));

}
}
