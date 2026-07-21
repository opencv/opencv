#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/core/opengl.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

const int win_width = 800;
const int win_height = 640;

// All GL objects live for the lifetime of the window's GL context.
// autoRelease is left at its default (false) so they are reclaimed with the
// context on shutdown; calling glDelete* from a destructor after the context
// is gone would raise a GL error (see cv::ogl wrapper docs).
struct DrawData
{
    ogl::Program program;
    ogl::Buffer vbo;          // referenced by the VertexArray; must outlive it
    ogl::VertexArray vao;
    ogl::Texture2D tex;
    int transformLoc;
};

static cv::Matx44f rot(float angle)
{
    // Row-major; ogl::Program::setUniformMat4x4 uploads with transpose enabled.
    return cv::Matx44f(
        cos(angle), 0, sin(angle), 0,
        0, 1, 0, 0,
        -sin(angle), 0, cos(angle), 0,
        0, 0, 0, 1);
}

static void draw(void* userdata) {
    DrawData* data = static_cast<DrawData*>(userdata);
    static float angle = 0.0f;
    angle += 1.f;

    cv::Matx44f trans = rot(CV_PI * angle / 360.f);

    ogl::clearColor(Scalar::all(0));
    ogl::clearDepth(1.0f);

    data->program.bind();
    ogl::Program::setUniformMat4x4(data->transformLoc, trans);
    data->tex.bind();
    data->vao.bind();
    ogl::drawArrays(0, 4, ogl::TRIANGLE_STRIP);
    ogl::VertexArray::unbind();
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

    ogl::enable(ogl::DEPTH_TEST);

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

    // Compile shaders and link the program via the ogl wrappers.
    ogl::Shader vertex_shader(vertex_shader_source, ogl::Shader::VERTEX);
    ogl::Shader fragment_shader(fragment_shader_source, ogl::Shader::FRAGMENT);
    data.program = ogl::Program(vertex_shader, fragment_shader);
    data.transformLoc = data.program.getUniformLocation("transform");

    const float vertices[] = {
            // Positions        // Texture Coords
            1.0f,  1.0f, 0.0f,  1.0f, 1.0f,   // Top Right
            1.0f, -1.0f, 0.0f,  1.0f, 0.0f,   // Bottom Right
            -1.0f,  1.0f, 0.0f,  0.0f, 1.0f,   // Top Left
            -1.0f, -1.0f, 0.0f,  0.0f, 0.0f    // Bottom Left
    };

    // Upload the interleaved vertex data into a GL buffer.
    Mat vertexMat(4, 5, CV_32F, (void*)vertices);
    data.vbo = ogl::Buffer(vertexMat, ogl::Buffer::ARRAY_BUFFER);

    // Position attribute (location 0): 3 floats, no offset.
    ogl::Attribute posAttr;
    posAttr.buffer_ = data.vbo;
    posAttr.stride_ = 5 * sizeof(float);
    posAttr.offset_ = 0;
    posAttr.size_ = 3;
    posAttr.type_ = ogl::Attribute::FLOAT;
    posAttr.integer_ = false;
    posAttr.normalized_ = false;
    posAttr.shader_loc_ = 0;

    // Texture coord attribute (location 1): 2 floats, offset past the 3 position floats.
    ogl::Attribute texAttr;
    texAttr.buffer_ = data.vbo;
    texAttr.stride_ = 5 * sizeof(float);
    texAttr.offset_ = 3 * sizeof(float);
    texAttr.size_ = 2;
    texAttr.type_ = ogl::Attribute::FLOAT;
    texAttr.integer_ = false;
    texAttr.normalized_ = false;
    texAttr.shader_loc_ = 1;

    data.vao = ogl::VertexArray({posAttr, texAttr});

    // Image to texture.
    data.tex = ogl::Texture2D(img);

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
