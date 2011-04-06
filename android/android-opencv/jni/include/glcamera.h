#ifndef GLCAMERA_H_
#define GLCAMERA_H_
#include <opencv2/core/core.hpp>

#ifdef __ANDROID__
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "image_pool.h"

class glcamera
{
public:

  glcamera();
  ~glcamera();
  void init(int width, int height);
  void step();

  void drawMatToGL(int idx, image_pool* pool);
  void drawMatToGL(const cv::Mat& img);
  void setTextureImage(const cv::Mat& img);

  void clear();

private:
  GLuint createSimpleTexture2D(GLuint _textureid, GLubyte* pixels, int width, int height, int channels);
  GLuint loadShader(GLenum shaderType, const char* pSource);
  GLuint
  createProgram(const char* pVertexSource, const char* pFragmentSource);
  bool setupGraphics(int w, int h);
  void renderFrame();
  cv::Mat nimg;
  bool newimage;
  GLuint textureID;

  GLuint gProgram;
  GLuint gvPositionHandle;

  GLuint gvTexCoordHandle;
  GLuint gvSamplerHandle;
  float img_w, img_h;
};
#endif
