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
#include "opencv2/core/cuda.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

const int win_width = 700;
const int win_height = 700;

struct DrawData
{
	ogl::Arrays arr;
	ogl::Buffer indices;
};

void draw(void* userdata);

void draw(void* userdata)
{
	DrawData* data = static_cast<DrawData*>(userdata);

	ogl::render(data->arr, data->indices, ogl::TRIANGLES);
}

int main()
{
	namedWindow("OpenGL", WINDOW_OPENGL);
	resizeWindow("OpenGL", win_width, win_height);

	Mat_<Vec3f> vertex(1, 6);
	vertex << Vec3f(2.0, 0, -2.0), Vec3f(0, -2, -2),
		Vec3f(-2, 0, -2), Vec3f(3.5, -1, -5),
		Vec3f(2.5, -1.5, -5), Vec3f(-1, 0.5, -5);

	Mat_<int> indices(1, 6);
	indices << 0, 1, 2, 3, 4, 5;

	Mat_<Vec4f> colors(1, 6);
	colors << Vec4f(0.725f, 0.933f, 0.851f, 1.0f), Vec4f(0.725f, 0.933f, 0.851f, 1.0f),
		Vec4f(0.725f, 0.933f, 0.851f, 1.0f), Vec4f(0.933f, 0.851f, 0.725f, 1.0f),
		Vec4f(0.933f, 0.851f, 0.725f, 1.0f), Vec4f(0.933f, 0.851f, 0.725f, 1.0f);

	DrawData data;

	data.arr.setVertexArray(vertex);
	data.arr.setColorArray(colors);
	data.indices.copyFrom(indices);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (double)win_width / win_height, 0.1, 50);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0);

	glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);

	setOpenGlDrawCallback("OpenGL", draw, &data);

	for (;;)
	{
        std::vector<uint8_t> pixels(win_width * win_height * 3);
        glReadPixels(0, 0, win_width, win_height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
        cv::Mat image(win_height, win_width, CV_8UC3, pixels.data());
        cv::flip(image, image, 0);

        cv::imwrite("example_image_depth.png", image);
		updateWindow("OpenGL");
		char key = (char)waitKey(40);
		if (key == 27)
			break;
	}

	setOpenGlDrawCallback("OpenGL", 0, 0);
	destroyAllWindows();

	return 0;
}
