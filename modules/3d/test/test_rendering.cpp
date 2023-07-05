#include "test_precomp.hpp"

namespace opencv_test { namespace {
using namespace cv;

class RenderingTest : public testing::Test
{
protected:
    void SetUp() override
    {
        width = 700, height = 700;

        position = Vec3f(0.0, 0.0, 6.0);
        lookat = Vec3f(0.0, 0.0, 1.0);
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

TEST_F(RenderingTest, depthRenderingTest)
{
    std::vector<float> depth_buf;
    std::vector<Vec3f> color_buf;

    std::vector <Vec3f> vertices = {
        Vec3f(2.0, 0, -1.0),
        Vec3f(0, 2, -1),
        Vec3f(-2, 0, -1),
        Vec3f(3.5, -1, -4),
        Vec3f(2.5, 1.5, -4),
        Vec3f(-1, 0.5, -4)
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

    Mat image(width, height, CV_32FC3, color_buf.data());
    image.convertTo(image, CV_8UC3, 1.0f);
    cvtColor(image, image, cv::COLOR_RGB2BGR);
    imwrite("temp_image.png", image);
}

TEST_F(RenderingTest, colorRenderingTest)
{
    std::vector<float> depth_buf;
    std::vector<Vec3f> color_buf;
    isConstant = false;

    std::vector <Vec3f> vertices = {
        Vec3f(-0.5f, -0.5f, 1.5f), Vec3f(-0.5f, 0.5f, 1.5f), Vec3f(0.5f, 0.5f, 1.5f), Vec3f(0.5f, -0.5f, 1.5f),
        Vec3f(-0.5f, -0.5f, 0.5f), Vec3f(-0.5f, 0.5f, 0.5f), Vec3f(0.5f, 0.5f, 0.5f), Vec3f(0.5f, -0.5f, 0.5f),
        Vec3f(-0.5f, -0.5f, 1.5f), Vec3f(-0.5f, 0.5f, 1.5f), Vec3f(-0.5f, 0.5f, 0.5f), Vec3f(-0.5f, -0.5f, 0.5f),
        Vec3f(0.5f, -0.5f, 1.5f), Vec3f(0.5f, 0.5f, 1.5f), Vec3f(0.5f, 0.5f, 0.5f), Vec3f(0.5f, -0.5f, 0.5f),
        Vec3f(0.5f, 0.5f, 1.5f), Vec3f(-0.5f, 0.5f, 1.5f), Vec3f(-0.5f, 0.5f, 0.5f), Vec3f(0.5f, 0.5f, 0.5f),
        Vec3f(0.5f, -0.5f, 1.5f), Vec3f(-0.5f, -0.5f, 1.5f), Vec3f(-0.5f, -0.5f, 0.5f), Vec3f(0.5f, -0.5f, 0.5f)
    };

    std::vector<Vec3i> indices = {
        Vec3i(0, 1, 2), Vec3i(0, 2, 3), Vec3i(4, 5, 6), Vec3i(4, 6, 7),
        Vec3i(8, 9, 10), Vec3i(8, 10, 11), Vec3i(12, 13, 14), Vec3i(12, 14, 15),
        Vec3i(16, 17, 18), Vec3i(16, 18, 19), Vec3i(20, 21, 22), Vec3i(20, 22, 23)
    };

    std::vector<Vec3f> colors = {
        Vec3f(0.0f, 0.0f, 255.0f),  Vec3f(0.0f, 0.0f, 255.0f),  Vec3f(0.0f, 0.0f, 255.0f),  Vec3f(0.0f, 0.0f, 255.0f),
        Vec3f(0.0f, 0.0f, -255.0f), Vec3f(0.0f, 0.0f, -255.0f), Vec3f(0.0f, 0.0f, -255.0f), Vec3f(0.0f, 0.0f, -255.0f),
        Vec3f(-255.0f, 0.0f, 0.0f), Vec3f(-255.0f, 0.0f, 0.0f), Vec3f(-255.0f, 0.0f, 0.0f), Vec3f(-255.0f, 0.0f, 0.0f),
        Vec3f(255.0f, 0.0f, 0.0f), Vec3f(255.0f, 0.0f, 0.0f), Vec3f(255.0f, 0.0f, 0.0f), Vec3f(255.0f, 0.0f, 0.0f),
        Vec3f(0.0f, 255.0f, 0.0f), Vec3f(0.0f, 255.0f, 0.0f), Vec3f(0.0f, 255.0f, 0.0f), Vec3f(0.0f, 255.0f, 0.0f),
        Vec3f(0.0f, -255.0f, 0.0f), Vec3f(0.0f, -255.0f, 0.0f), Vec3f(0.0f, -255.0f, 0.0f), Vec3f(0.0f, -255.0f, 0.0f)
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
    Mat image(width, height, CV_32FC3, color_buf.data());
    //Mat image(width, height, CV_32FC3, blue_color);
    image.convertTo(image, CV_8UC3, 1.0f);
    cvtColor(image, image, cv::COLOR_RGB2BGR);
    imwrite("temp_image_color.png", image);
}


}
}
