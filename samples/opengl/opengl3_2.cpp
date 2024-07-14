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
    GLuint vao, vbo, program, textureID;
};

cv::Mat rot(float angle)
{
    cv::Mat R_y = (cv::Mat_<float>(4,4) <<
        cos(angle), 0, sin(angle), 0,
        0, 1, 0, 0,
        -sin(angle), 0, cos(angle), 0,
        0, 0, 0, 1);

    return R_y;
}

static GLuint create_shader(const char* source, GLenum type) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    return shader;
}

void draw(void* userdata){
    DrawData* data = static_cast<DrawData*>(userdata);
    static float angle = 0.0f;
    angle += 1.f;

    cv::Mat trans = rot(CV_PI * angle / 360.f);

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(data->program);
    glUniformMatrix4fv(glGetUniformLocation(data->program, "transform"), 1, GL_FALSE, trans.ptr<float>());
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
        filename = "HappyFish.jpg";
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
    const char *vertex_shader_source =
            "#version 330 core\n"
            "layout (location = 0) in vec3 position;\n"
            "layout (location = 1) in vec2 texCoord;\n"
            "out vec2 TexCoord;\n"
            "uniform mat4 transform;\n"
            "void main() {\n"
            "   gl_Position = transform * vec4(position, 1.0);\n"
            "   TexCoord = texCoord;\n"
            "}\n";
    const char *fragment_shader_source =
            "#version 330 core\n"
            "in vec2 TexCoord;\n"
            "out vec4 color;\n"
            "uniform sampler2D ourTexture;\n"
            "void main() {\n"
            "   color = texture(ourTexture, TexCoord);\n"
            "}\n";
    data.program = glCreateProgram();
    GLuint vertex_shader = create_shader(vertex_shader_source, GL_VERTEX_SHADER);
    GLuint fragment_shader = create_shader(fragment_shader_source, GL_FRAGMENT_SHADER);
    glAttachShader(data.program, vertex_shader);
    glAttachShader(data.program, fragment_shader);
    glLinkProgram(data.program);
    glUseProgram(data.program);

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
