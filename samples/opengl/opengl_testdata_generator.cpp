#include <iostream>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN 1
#define NOMINMAX 1
#include <windows.h>
#endif

#if defined(__APPLE__)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "opencv2/core.hpp"
#include "opencv2/core/opengl.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

struct DrawData
{
    ogl::Arrays arr;
    ogl::Buffer indices;
};

void draw(void* userdata);

void draw(void* userdata)
{
    DrawData* data = static_cast<DrawData*>(userdata);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    ogl::render(data->arr, data->indices, ogl::TRIANGLES);
}

int main(int argc, char* argv[])
{
    cv::CommandLineParser parser(argc, argv,
            "{ help h usage ? |      | show this message }"
            "{ width          | 700  | resulting image width }"
            "{ height         | 700  | resulting image height }"
            
    );
    parser.about("This app is used to generate test data for triangleRasterize() function");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    parser.get<string>("colorFname");
    parser.get<string>("depthFname");
    parser.get<string>("clip");
    int win_width  = parser.get<int>("width");
    int win_height = parser.get<int>("height");

    namedWindow("OpenGL", WINDOW_OPENGL);
    resizeWindow("OpenGL", win_width, win_height);

    Mat_<Vec3f> vertex;
    Mat_<Vec4f> colors;
    Mat_<int> indices;
    double fovy = 0.0;
    std::string fname;
    if (clipping)
    {
        vertex = {
            { 2.0,  0.0, -2.0}, { 0.0  -6.0, -2.0}, {-2.0,  0.0, -2.0},
            { 3.5, -1.0, -5.0}, { 2.5, -2.5, -5.0}, {-1.0,  1.0, -5.0},
            {-6.5, -1.0, -3.0}, {-2.5, -2.0, -3.0}, { 1.0,  1.0, -5.0},
        };
        Vec4f col1(0.725f, 0.933f, 0.851f, 1.0f);
        Vec4f col2(0.933f, 0.851f, 0.725f, 1.0f);
        Vec4f col3(0.933f, 0.039f, 0.588f, 1.0f);
        colors = {
            col1, col1, col1,
            col2, col2, col2,
            col3, col3, col3,
        };
        indices = {0, 1, 2, 3, 4, 5, 6, 7, 8};

        fovy = 45.0;
        fname = "example_image_clipping.png";
    }
    else if (color)
    {
        vertex = {
            { 2.0,  0.0, -2.0},
            { 0.0,  2.0, -3.0},
            {-2.0,  0.0, -2.0},
            { 0.0, -2.0,  1.0},
        };
        colors = {
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f},
            {0.0f, 1.0f, 0.0f},
        };
        indices = {0, 1, 2, 0, 2, 3};

        fovy = 60.0;
        fname = "example_image_color.png";
    }
    else if (depth)
    {
        vertex = {
            { 2.0,  0.0, -2.0}, { 0.0, -2.0, -2.0}, {-2.0,  0.0, -2.0},
            { 3.5, -1.0, -5.0}, { 2.5, -1.5, -5.0}, {-1.0,  0.5, -5.0},
        };
        Vec4f col1(0.851f, 0.933f, 0.725f, 1.0f);
        Vec4f col2(0.725f, 0.851f, 0.933f, 1.0f);
        colors = {
            col1, col1, col1,
            col2, col2, col2
        };
        indices = { 0, 1, 2, 3, 4, 5 };

        fovy = 45.0;
        fname = "example_image_depth.png";
    }
    else
    {
        std::cout << "Wrong mode: " << mode << std::endl;
        return -1;
    }

    DrawData data;

    data.arr.setVertexArray(vertex);
    data.arr.setColorArray(colors);
    data.indices.copyFrom(indices);

    float zNear = 0.1, zFar = 50;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(fovy, (double)win_width / win_height, zNear, zFar);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0);

    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);

    setOpenGlDrawCallback("OpenGL", draw, &data);

    for (;;)
    {
        updateWindow("OpenGL");

        cv::Mat colorData(win_height, win_width, CV_8UC3, Scalar());
        glReadPixels(0, 0, win_width, win_height, GL_RGB, GL_UNSIGNED_BYTE, colorData.data());
        cv::cvtColor(colorData, colorData, cv::COLOR_RGB2BGR);
        cv::flip(colorData, colorData, 0);
        cv::imwrite(colorFname, colorData);

        if(depth)
        {
            cv::Mat depthMat(win_height, win_width, CV_32F);
            glReadPixels(0, 0, win_width, win_height, GL_DEPTH_COMPONENT, GL_FLOAT, depthMat.ptr());
            for (auto it = depthMat.begin<float>(); it != depthMat.end<float>(); ++it)
            {
                *it = zNear * zFar / ((*it) * (zNear - zFar) + zFar);
            }
            cv::flip(depthMat, depthMat, 0);
            cv::imwrite(depthFname, depthMat);
        }

        char key = (char)waitKey(40);
        if (key == 27)
            break;
    }

    setOpenGlDrawCallback("OpenGL", 0, 0);
    destroyAllWindows();

    return 0;
}
