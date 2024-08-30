#include <iostream>

#include <epoxy/gl.h>

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
#include "opencv2/core/cuda.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

const int win_width = 800;
const int win_height = 640;

struct DrawData
{
    GLuint vao, vbo, textureID;
    ogl::Program program;
};

static cv::Mat rot(float angle)
{
    cv::Mat R_y = (cv::Mat_<float>(4,4) <<
        cos(angle), 0, sin(angle), 0,
        0, 1, 0, 0,
        -sin(angle), 0, cos(angle), 0,
        0, 0, 0, 1);

    return R_y;
}

static void draw(void* userdata) {
    DrawData* data = static_cast<DrawData*>(userdata);
    static float angle = 0.0f;
    angle += 1.f;

    cv::Mat trans = rot(CV_PI * angle / 360.f);

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(data->program.getProgram());
    glUniformMatrix4fv(glGetUniformLocation(data->program.getProgram(), "transform"), 1, GL_FALSE, trans.ptr<float>());
    glBindTexture(GL_TEXTURE_2D, data->textureID);
    glBindVertexArray(data->vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

int main(int argc, char* argv[])
{
    string filename;
    if (argc < 2)
    {
        cout << "Usage: " << argv[0] << " image" << endl;
        filename = "baboon.jpg";
    }
    else
        filename = argv[1];

    Mat img = imread(samples::findFile(filename));
    if (img.empty())
    {
        cerr << "Can't open image " << filename << endl;
        return -1;
    }
    flip(img, img, 0);

    namedWindow("OpenGL", WINDOW_OPENGL);
    resizeWindow("OpenGL", win_width, win_height);

    DrawData data;

    glEnable(GL_DEPTH_TEST);
    data.program.attachDefaultShaders();

    GLfloat vertices[] = {
            // Positions        // Texture Coords
            1.0f,  1.0f, 0.0f,  1.0f, 1.0f,   // Top Right
            1.0f, -1.0f, 0.0f,  1.0f, 0.0f,   // Bottom Right
            -1.0f,  1.0f, 0.0f,  0.0f, 1.0f,   // Top Left
            -1.0f, -1.0f, 0.0f,  0.0f, 0.0f    // Bottom Left
    };

    glGenVertexArrays(1, &data.vao);
    glGenBuffers(1, &data.vbo);
    glBindVertexArray(data.vao);
    glBindBuffer(GL_ARRAY_BUFFER, data.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);
    // Texture Coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0); // Unbind VAO


//        Image to texture
    glGenTextures(1, &data.textureID);
    glBindTexture(GL_TEXTURE_2D, data.textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, img.data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    setOpenGlDrawCallback("OpenGL", draw, &data);

    for (;;)
    {
        updateWindow("OpenGL");
        char key = (char)waitKey(40);
        if (key == 27)
            break;
    }

    setOpenGlDrawCallback("OpenGL", 0, 0);
    destroyAllWindows();
    return 0;
}
