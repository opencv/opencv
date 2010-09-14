/*
 * Copyright (C) 2009 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// OpenGL ES 2.0 code

#include <jni.h>

#include <android/log.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "glcamera.h"
#include "image_pool.h"
using namespace cv;
#define  LOG_TAG    "libandroid-opencv"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

static void printGLString(const char *name, GLenum s) {
	const char *v = (const char *) glGetString(s);
	LOGI("GL %s = %s\n", name, v);
}

static void checkGlError(const char* op) {
	for (GLint error = glGetError(); error; error = glGetError()) {
		LOGI("after %s() glError (0x%x)\n", op, error);
	}
}

static const char gVertexShader[] = "attribute vec4 a_position;   \n"
	"attribute vec2 a_texCoord;   \n"
	"varying vec2 v_texCoord;     \n"
	"void main()                  \n"
	"{                            \n"
	"   gl_Position = a_position; \n"
	"   v_texCoord = a_texCoord;  \n"
	"}                            \n";

static const char gFragmentShader[] =
		"precision mediump float;                            \n"
			"varying vec2 v_texCoord;                            \n"
			"uniform sampler2D s_texture;                        \n"
			"void main()                                         \n"
			"{                                                   \n"
			"  gl_FragColor = texture2D( s_texture, v_texCoord );\n"
			"}                                                   \n";

const GLfloat gTriangleVertices[] = { 0.0f, 0.5f, -0.5f, -0.5f, 0.5f, -0.5f };
GLubyte testpixels[4 * 3] = { 255, 0, 0, // Red
		0, 255, 0, // Green
		0, 0, 255, // Blue
		255, 255, 0 // Yellow
		};

GLuint glcamera::createSimpleTexture2D(GLuint _textureid, GLubyte* pixels,
		int width, int height, int channels) {

	// Bind the texture
	glActiveTexture(GL_TEXTURE0);
	checkGlError("glActiveTexture");
	// Bind the texture object
	glBindTexture(GL_TEXTURE_2D, _textureid);
	checkGlError("glBindTexture");

	GLenum format;
	switch (channels) {
	case 3:
		format = GL_RGB;
		break;
	case 1:
		format = GL_LUMINANCE;
		break;
	case 4:
		format = GL_RGBA;
		break;
	}
	// Load the texture
	glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format,
			GL_UNSIGNED_BYTE, pixels);

	checkGlError("glTexImage2D");
	// Set the filtering mode
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

	return _textureid;

}

GLuint glcamera::loadShader(GLenum shaderType, const char* pSource) {
	GLuint shader = glCreateShader(shaderType);
	if (shader) {
		glShaderSource(shader, 1, &pSource, NULL);
		glCompileShader(shader);
		GLint compiled = 0;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
		if (!compiled) {
			GLint infoLen = 0;
			glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
			if (infoLen) {
				char* buf = (char*) malloc(infoLen);
				if (buf) {
					glGetShaderInfoLog(shader, infoLen, NULL, buf);
					LOGE("Could not compile shader %d:\n%s\n",
							shaderType, buf);
					free(buf);
				}
				glDeleteShader(shader);
				shader = 0;
			}
		}
	}
	return shader;
}

GLuint glcamera::createProgram(const char* pVertexSource,
		const char* pFragmentSource) {
	GLuint vertexShader = loadShader(GL_VERTEX_SHADER, pVertexSource);
	if (!vertexShader) {
		return 0;
	}

	GLuint pixelShader = loadShader(GL_FRAGMENT_SHADER, pFragmentSource);
	if (!pixelShader) {
		return 0;
	}

	GLuint program = glCreateProgram();
	if (program) {
		glAttachShader(program, vertexShader);
		checkGlError("glAttachShader");
		glAttachShader(program, pixelShader);
		checkGlError("glAttachShader");
		glLinkProgram(program);
		GLint linkStatus = GL_FALSE;
		glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
		if (linkStatus != GL_TRUE) {
			GLint bufLength = 0;
			glGetProgramiv(program, GL_INFO_LOG_LENGTH, &bufLength);
			if (bufLength) {
				char* buf = (char*) malloc(bufLength);
				if (buf) {
					glGetProgramInfoLog(program, bufLength, NULL, buf);
					LOGE("Could not link program:\n%s\n", buf);
					free(buf);
				}
			}
			glDeleteProgram(program);
			program = 0;
		}
	}
	return program;
}

//GLuint textureID;

bool glcamera::setupGraphics(int w, int h) {
	printGLString("Version", GL_VERSION);
	printGLString("Vendor", GL_VENDOR);
	printGLString("Renderer", GL_RENDERER);
	printGLString("Extensions", GL_EXTENSIONS);

	LOGI("setupGraphics(%d, %d)", w, h);
	gProgram = createProgram(gVertexShader, gFragmentShader);
	if (!gProgram) {
		LOGE("Could not create program.");
		return false;
	}
	gvPositionHandle = glGetAttribLocation(gProgram, "a_position");
	gvTexCoordHandle = glGetAttribLocation(gProgram, "a_texCoord");

	gvSamplerHandle = glGetAttribLocation(gProgram, "s_texture");

	// Use tightly packed data
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	// Generate a texture object
	glGenTextures(1, &textureID);
	textureID = createSimpleTexture2D(textureID, testpixels, 2, 2, 3);

	checkGlError("glGetAttribLocation");
	LOGI("glGetAttribLocation(\"vPosition\") = %d\n",
			gvPositionHandle);

	glViewport(0, 0, w, h);
	checkGlError("glViewport");
	return true;
}

void glcamera::renderFrame() {

	GLfloat vVertices[] = { -1.0f, 1.0f, 0.0f, // Position 0
			0.0f, 0.0f, // TexCoord 0
			-1.0f, -1.0f, 0.0f, // Position 1
			0.0f, 1.0f, // TexCoord 1
			1.0f, -1.0f, 0.0f, // Position 2
			1.0f, 1.0f, // TexCoord 2
			1.0f, 1.0f, 0.0f, // Position 3
			1.0f, 0.0f // TexCoord 3
			};
	GLushort indices[] = { 0, 1, 2, 0, 2, 3 };
	GLsizei stride = 5 * sizeof(GLfloat); // 3 for position, 2 for texture

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	checkGlError("glClearColor");

	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	checkGlError("glClear");

	glUseProgram(gProgram);
	checkGlError("glUseProgram");

	// Load the vertex position
	glVertexAttribPointer(gvPositionHandle, 3, GL_FLOAT, GL_FALSE, stride,
			vVertices);
	// Load the texture coordinate
	glVertexAttribPointer(gvTexCoordHandle, 2, GL_FLOAT, GL_FALSE, stride,
			&vVertices[3]);

	glEnableVertexAttribArray(gvPositionHandle);
	glEnableVertexAttribArray(gvTexCoordHandle);

	// Bind the texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, textureID);

	// Set the sampler texture unit to 0
	glUniform1i(gvSamplerHandle, 0);

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, indices);

	//checkGlError("glVertexAttribPointer");
	//glEnableVertexAttribArray(gvPositionHandle);
	//checkGlError("glEnableVertexAttribArray");
	//glDrawArrays(GL_TRIANGLES, 0, 3);
	//checkGlError("glDrawArrays");
}

void glcamera::init(int width, int height) {
	newimage = false;
	nimg = Mat();
	setupGraphics(width, height);

}

void glcamera::step() {
	if (newimage && !nimg.empty()) {

		textureID = createSimpleTexture2D(textureID,
				nimg.ptr<unsigned char> (0), nimg.rows, nimg.cols,
				nimg.channels());
		newimage = false;
	}
	renderFrame();

}

void glcamera::setTextureImage(Ptr<Mat> img) {

	//int p2 = (int)(std::log(img->size().width)/0.69315);
	int sz = 256;//std::pow(2,p2);
	Size size(sz, sz);

	resize(*img, nimg, size,cv::INTER_NEAREST);

	newimage = true;

}

void glcamera::drawMatToGL(int idx, image_pool* pool) {

	Ptr<Mat> img = pool->getImage(idx);

	if (img.empty())
		return; //no image at input_idx!

	setTextureImage(img);

}

glcamera::glcamera():newimage(false) {
	LOGI("glcamera constructor");
}
glcamera::~glcamera() {
	LOGI("glcamera destructor");
}


