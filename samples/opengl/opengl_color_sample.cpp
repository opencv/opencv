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

	Mat_<Vec3f> vertex(1, 4);
	vertex << Vec3f(2.0, 0, -2.0), Vec3f(0, 2, -3),
              Vec3f(-2, 0, -2), Vec3f(0, -2, -1);

	Mat_<int> indices(1, 6);
	indices << 0, 1, 2, 0, 2, 3;

	Mat_<Vec3f> colors(1, 4);
	colors << Vec3f(0.0f, 0.0f, 255.0f), Vec3f(0.0f, 255.0f, 0.0f), Vec3f(255.0f, 0.0f, 0.0f), Vec3f(0.0f, 255.0f, 0.0f);

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
