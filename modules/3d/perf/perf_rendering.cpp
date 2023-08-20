#include "perf_precomp.hpp"

namespace opencv_test
{
    using namespace perf;
    typedef perf::TestBaseWithParam<std::tuple<int, int>> RenderingTest;

    PERF_TEST_P(RenderingTest, depthRenderingPerfTest, testing::Values(
        std::make_tuple(700, 700),
        std::make_tuple(640, 480)
    ))
    {
        int width, height;
        Vec3f position;
        Vec3f lookat;
        Vec3f upVector;
        float zNear, zFar, fovy;
        bool shadingMode;

        auto t = GetParam();
        width = std::get<0>(t), height = std::get<1>(t);

        position = Vec3f(0.0, 0.0, 5.0);
        lookat = Vec3f(0.0, 0.0, 0.0);
        upVector = Vec3f(0.0, 1.0, 0.0);

        fovy = 45.0;
        zNear = 0.1, zFar = 50;
        shadingMode = false;

        Mat depth_buf(height, width, CV_32F, zFar);
        Mat color_buf(height, width, CV_32FC3, Scalar(0.0, 0.0, 0.0));

        std::vector <Vec3f> vertices = {
        Vec3f(2.0, 0, -2.0), Vec3f(0, -2, -2.0), Vec3f(-2, 0, -2.0),
        Vec3f(3.5, -1, -5.0), Vec3f(2.5, -1.5, -5.0), Vec3f(-1, 0.5, -5.0)
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

        TEST_CYCLE()
        {
            cv::triangleRasterize(vertices, indices, colors, cameraMatrix, width, height,
                shadingMode, depth_buf, color_buf);
        }

        SANITY_CHECK_NOTHING();
    }

    PERF_TEST_P(RenderingTest, colorRenderingPerfTest, testing::Values(
        std::make_tuple(700, 700),
        std::make_tuple(640, 480)
    ))
    {
        int width, height;
        Vec3f position;
        Vec3f lookat;
        Vec3f upVector;
        float zNear, zFar, fovy;
        bool shadingMode;

        auto t = GetParam();
        width = std::get<0>(t), height = std::get<1>(t);

        position = Vec3f(0.0, 0.0, 5.0);
        lookat = Vec3f(0.0, 0.0, 0.0);
        upVector = Vec3f(0.0, 1.0, 0.0);

        fovy = 60.0;
        zNear = 0.1, zFar = 50;
        shadingMode = false;

        Mat depth_buf(height, width, CV_32F, zFar);
        Mat color_buf(height, width, CV_32FC3, Scalar(0.0, 0.0, 0.0));

        std::vector <Vec3f> vertices = {
            Vec3f(2.0, 0, -2.0), Vec3f(0, 2, -3),
            Vec3f(-2, 0, -2), Vec3f(0, -2, 1)
        };

        std::vector<Vec3i> indices = {
            Vec3i(0, 1, 2), Vec3i(0, 2, 3)
        };

        std::vector<Vec3f> colors = {
            Vec3f(0.0f, 0.0f, 255.0f), Vec3f(0.0f, 255.0f, 0.0f),
            Vec3f(255.0f, 0.0f, 0.0f), Vec3f(0.0f, 255.0f, 0.0f)
        };

        Mat cameraMatrix(4, 3, CV_32F);

        cameraMatrix.row(0) = Mat(position).t();
        cameraMatrix.row(1) = Mat(lookat).t();
        cameraMatrix.row(2) = Mat(upVector).t();
        cameraMatrix.row(3) = Mat(Vec3f(fovy, zNear, zFar)).t();

        TEST_CYCLE()
        {
            cv::triangleRasterize(vertices, indices, colors, cameraMatrix, width, height,
                shadingMode, depth_buf, color_buf);
        }

        SANITY_CHECK_NOTHING();
    }

    PERF_TEST_P(RenderingTest, clippingRenderingPerfTest, testing::Values(
        std::make_tuple(700, 700),
        std::make_tuple(640, 480)
    ))
    {
        int width, height;
        Vec3f position;
        Vec3f lookat;
        Vec3f upVector;
        float zNear, zFar, fovy;
        bool shadingMode;

        auto t = GetParam();
        width = std::get<0>(t), height = std::get<1>(t);

        position = Vec3f(0.0, 0.0, 5.0);
        lookat = Vec3f(0.0, 0.0, 0.0);
        upVector = Vec3f(0.0, 1.0, 0.0);

        fovy = 45.0;
        zNear = 0.1, zFar = 50;
        shadingMode = false;

        Mat depth_buf(height, width, CV_32F, zFar);
        Mat color_buf(height, width, CV_32FC3, Scalar(0.0, 0.0, 0.0));

        std::vector <Vec3f> vertices = {
            Vec3f(2.0, 0, -2.0), Vec3f(0, -6, -2), Vec3f(-2, 0, -2),
            Vec3f(3.5, -1, -5), Vec3f(2.5, -2.5, -5), Vec3f(-1, 1, -5),
            Vec3f(-6.5, -1, -3), Vec3f(-2.5, -2, -3), Vec3f(1, 1, -5)
        };

        std::vector<Vec3i> indices = {
            Vec3i(0, 1, 2), Vec3i(3, 4, 5), Vec3i(6, 7, 8)
        };

        std::vector<Vec3f> colors = {
            Vec3f(217.0, 238.0, 185.0), Vec3f(217.0, 238.0, 185.0), Vec3f(217.0, 238.0, 185.0),
            Vec3f(185.0, 217.0, 238.0), Vec3f(185.0, 217.0, 238.0), Vec3f(185.0, 217.0, 238.0),
            Vec3f(150.0, 10.0, 238.0), Vec3f(150.0, 10.0, 238.0), Vec3f(150.0, 10.0, 238.0)
        };

        Mat cameraMatrix(4, 3, CV_32F);

        cameraMatrix.row(0) = Mat(position).t();
        cameraMatrix.row(1) = Mat(lookat).t();
        cameraMatrix.row(2) = Mat(upVector).t();
        cameraMatrix.row(3) = Mat(Vec3f(fovy, zNear, zFar)).t();

        TEST_CYCLE()
        {
            cv::triangleRasterize(vertices, indices, colors, cameraMatrix, width, height,
                shadingMode, depth_buf, color_buf);
        }

        SANITY_CHECK_NOTHING();
    }

    PERF_TEST_P(RenderingTest, emptyIndicePerfTest, testing::Values(
        std::make_tuple(700, 700),
        std::make_tuple(640, 480)
    ))
    {
        int width, height;
        Vec3f position;
        Vec3f lookat;
        Vec3f upVector;
        float zNear, zFar, fovy;
        bool shadingMode;

        auto t = GetParam();
        width = std::get<0>(t), height = std::get<1>(t);

        position = Vec3f(0.0, 0.0, 5.0);
        lookat = Vec3f(0.0, 0.0, 0.0);
        upVector = Vec3f(0.0, 1.0, 0.0);

        fovy = 45.0;
        zNear = 0.1, zFar = 50;
        shadingMode = false;

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

        TEST_CYCLE()
        {
            cv::triangleRasterize(vertices, indices, colors, cameraMatrix, width, height,
                shadingMode, depth_buf, color_buf);
        }

        SANITY_CHECK_NOTHING();

    }
} // namespace
