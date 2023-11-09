#include "perf_precomp.hpp"

namespace opencv_test
{
    using namespace perf;
    typedef perf::TestBaseWithParam<cv::Size> RenderingTest;

    PERF_TEST_P(RenderingTest, rasterizeTriangles, testing::Values({700, 700}, {640, 480}))
    {
        auto t = GetParam();
        int width = t.width, height = t.height;

        Vec3f position = Vec3f(0.0, 0.0, 5.0);
        Vec3f lookat   = Vec3f(0.0, 0.0, 0.0);
        Vec3f upVector = Vec3f(0.0, 1.0, 0.0);

        double fovy;
        if (depth || clipping)
        {
            fovy = 45.0;
        }
        else if (color)
        {
            fovy = 60.0;
        }
        double zNear = 0.1, zFar = 50.0;
        //TODO: check also this
        bool shadingMode = false;

        std::vector<Vec3f> vertices;
        std::vector<Vec3i> indices;
        std::vector<Vec3f> colors;
        if (depth)
        {
            vertices =
            {
                { 2.0,  0.0, -2.0}, { 0.0, -2.0, -2.0}, {-2.0,  0.0, -2.0},
                { 3.5, -1.0, -5.0}, { 2.5, -1.5, -5.0}, {-1.0,  0.5, -5.0},
            };
            //TODO: value types and ranges
            Vec3f col1(217.0, 238.0, 185.0);
            Vec3f col2(185.0, 217.0, 238.0);
            colors = { col1, col1, col1, col2, col2, col2 };

            indices = { {0, 1, 2}, {3, 4, 5} };
        }
        else if (color)
        {
            vertices =
            {
                { 2.0,  0.0, -2.0},
                { 0.0,  2.0, -3.0},
                {-2.0,  0.0, -2.0},
                { 0.0, -2.0,  1.0},
            };
            //TODO: value types and ranges
            colors =
            {
                {  0.0f,   0.0f, 255.0f},
                {  0.0f, 255.0f,   0.0f},
                {255.0f,   0.0f,   0.0f},
                {  0.0f, 255.0f,   0.0f},
            };

            indices = { {0, 1, 2}, {0, 2, 3} };
        }
        else if (clipping)
        {
            vertices =
            {
                { 2.0,  0.0, -2.0}, { 0.0, -6.0, -2.0}, {-2.0,  0.0, -2.0},
                { 3.5, -1.0, -5.0}, { 2.5, -2.5, -5.0}, {-1.0,  1.0, -5.0},
                {-6.5, -1.0, -3.0}, {-2.5, -2.0, -3.0}, { 1.0,  1.0, -5.0},
            };
            //TODO: value types and ranges
            Vec3f col1(217.0, 238.0, 185.0);
            Vec3f col2(185.0, 217.0, 238.0);
            Vec3f col3(150.0,  10.0, 238.0);
            colors =
            {
                col1, col1, col1,
                col2, col2, col2,
                col3, col3, col3,
            };

            indices = { {0, 1, 2}, {3, 4, 5}, {6, 7, 8} };
        }

        Mat cameraMatrix(4, 3, CV_32F);

        cameraMatrix.row(0) = Mat(position).t();
        cameraMatrix.row(1) = Mat(lookat).t();
        cameraMatrix.row(2) = Mat(upVector).t();
        cameraMatrix.row(3) = Mat(Vec3f(fovy, zNear, zFar)).t();

        while(next())
        {
            // Prefilled to measure pure rendering time w/o allocation and clear
            Mat depth_buf(height, width, CV_32F, zFar);
            Mat color_buf(height, width, CV_32FC3, Scalar(0.0, 0.0, 0.0));

            startTimer();
            //TODO: no color for depth test
            cv::triangleRasterize(vertices, indices, colors, cameraMatrix, width, height,
                shadingMode, depth_buf, color_buf);
            stopTimer();
        }

        SANITY_CHECK_NOTHING();
    }
} // namespace
