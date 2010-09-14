#ifndef GLCAMERA_H_
#define GLCAMERA_H_
#include <opencv2/core/core.hpp>

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#include "image_pool.h"
class glcamera {
	Mat nimg;
	bool newimage;
	GLuint textureID;

	GLuint gProgram;
	GLuint gvPositionHandle;

	GLuint gvTexCoordHandle;
	GLuint gvSamplerHandle;

public:

	glcamera();
	~glcamera();
	void init(int width, int height);
	void step();

	void drawMatToGL(int idx, image_pool* pool);
	void setTextureImage(Ptr<Mat> img);

private:
	GLuint createSimpleTexture2D(GLuint _textureid, GLubyte* pixels, int width,
			int height, int channels);
	GLuint loadShader(GLenum shaderType, const char* pSource);
	GLuint
			createProgram(const char* pVertexSource,
					const char* pFragmentSource);
	bool setupGraphics(int w, int h);
	void renderFrame();
};
#endif
