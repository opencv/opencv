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

typedef perf::TestBaseWithParam<std::tuple<std::tuple<int, int>, bool, bool>> RenderingTest;

PERF_TEST_P(RenderingTest, rasterizeTriangles, ::testing::Combine(
    ::testing::Values(std::make_tuple(1920, 1080), std::make_tuple(1024, 768), std::make_tuple(640, 480)),
    ::testing::Bool(), // shading
    ::testing::Bool() // have colors
    ))
{
    auto t = GetParam();
    auto wh = std::get<0>(t);
    int width = std::get<0>(wh);
    int height = std::get<1>(wh);
    bool shadingMode = std::get<1>(t);
    bool haveColors = std::get<2>(t);

    string objectPath = findDataFile("rendering/model/spot.obj");

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

    if (haveColors)
    {
        for (auto &color : colors)
        {
            color = Vec3f(abs(color[0]), abs(color[1]), abs(color[2])) * 255.0f;
        }
    }
    else
    {
        colors.clear();
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
                              shadingMode, depth_buf, haveColors ? color_buf : noArray());
        stopTimer();
    }

    if (debugLevel > 0)
    {
        color_buf.convertTo(color_buf, CV_8UC3, 1.0f);
        cvtColor(color_buf, color_buf, cv::COLOR_RGB2BGR);
        cv::flip(color_buf, color_buf, 0);
        depth_buf.convertTo(depth_buf, CV_8UC1, 1.0);
        cv::flip(depth_buf, depth_buf, 0);

        std::string suffix = std::to_string(width) + "x" + std::to_string(height) + " ";
        suffix += (shadingMode ? "shaded" : "flat");
        suffix += (haveColors ? "colored" : "white");

        imwrite("color_image_" + suffix + ".png", color_buf);
        imwrite("depth_image_" + suffix + ".png", depth_buf);
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
