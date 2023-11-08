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

//TODO: CommandLineParser
// This app is used to generate test data for triangleRasterize() function

int main(int argc, char* argv[])
{
    int win_width = 700;
    int win_height = 700;

    if (argc != 4)
    {
        std::cout << "Wrong input for the demo. please input again" << std::endl;
        return 0;
    }
    std::string testMode = argv[1];
    win_width = std::atoi(argv[2]), win_height = std::atoi(argv[3]);

    namedWindow("OpenGL", WINDOW_OPENGL);
    resizeWindow("OpenGL", win_width, win_height);

    Mat_<Vec3f> vertex;
    Mat_<Vec4f> colors;
    Mat_<int> indices;
    double fovy = 0.0;
    std::string fname;
    if (clipping)
    {
        vertex << Vec3f(2.0, 0, -2.0), Vec3f(0, -6, -2), Vec3f(-2, 0, -2),
                Vec3f(3.5, -1, -5),  Vec3f(2.5, -2.5, -5), Vec3f(-1, 1, -5),
                Vec3f(-6.5, -1, -3), Vec3f(-2.5, -2, -3), Vec3f(1, 1, -5);
        colors << Vec4f(0.725f, 0.933f, 0.851f, 1.0f), Vec4f(0.725f, 0.933f, 0.851f, 1.0f), Vec4f(0.725f, 0.933f, 0.851f, 1.0f),
        Vec4f(0.933f, 0.851f, 0.725f, 1.0f), Vec4f(0.933f, 0.851f, 0.725f, 1.0f), Vec4f(0.933f, 0.851f, 0.725f, 1.0f),
        Vec4f(0.933f, 0.039f, 0.588f, 1.0f), Vec4f(0.933f, 0.039f, 0.588f, 1.0f), Vec4f(0.933f, 0.039f, 0.588f, 1.0f);

        indices << 0, 1, 2, 3, 4, 5, 6, 7, 8;
        fovy = 45.0;
        fname = "example_image_clipping.png";
    }
    else if (color)
    {
        vertex << Vec3f(2.0, 0, -2.0), Vec3f(0, 2, -3),
                Vec3f(-2, 0, -2), Vec3f(0, -2, 1.0);
        colors << Vec3f(1.0f, 0.0f, 0.0f), Vec3f(0.0f, 1.0f, 0.0f), Vec3f(0.0f, 0.0f, 1.0f), Vec3f(0.0f, 1.0f, 0.0f);

        indices << 0, 1, 2, 0, 2, 3;

        fovy = 60.0;
        fname = "example_image_color.png";
    }
    else if (depth)
    {
        vertex << Vec3f(2.0, 0, -2.0), Vec3f(0, -2, -2),
                  Vec3f(-2, 0, -2), Vec3f(3.5, -1, -5),
                  Vec3f(2.5, -1.5, -5), Vec3f(-1, 0.5, -5);

        indices << 0, 1, 2, 3, 4, 5;

        colors << Vec4f(0.851f, 0.933f, 0.725f, 1.0f), Vec4f(0.851f, 0.933f, 0.725f, 1.0f),
                  Vec4f(0.851f, 0.933f, 0.725f, 1.0f), Vec4f(0.725f, 0.851f, 0.933f, 1.0f),
                  Vec4f(0.725f, 0.851f, 0.933f, 1.0f), Vec4f(0.725f, 0.851f, 0.933f, 1.0f);

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

        std::vector<uint8_t> pixels(win_width * win_height * 3);
        glReadPixels(0, 0, win_width, win_height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
        cv::Mat image(win_height, win_width, CV_8UC3, pixels.data());
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        cv::flip(image, image, 0);
        cv::imwrite(fname, image);

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
