// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef __OPENCV_FFMPEG_LEGACY_API_H__
#define __OPENCV_FFMPEG_LEGACY_API_H__

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef OPENCV_FFMPEG_API
#if defined(__OPENCV_BUILD)
#   define OPENCV_FFMPEG_API
#elif defined _WIN32
#   define OPENCV_FFMPEG_API __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#   define OPENCV_FFMPEG_API __attribute__ ((visibility ("default")))
#else
#   define OPENCV_FFMPEG_API
#endif
#endif

typedef struct CvCapture_FFMPEG CvCapture_FFMPEG;
typedef struct CvVideoWriter_FFMPEG CvVideoWriter_FFMPEG;

//OPENCV_FFMPEG_API struct CvCapture_FFMPEG* cvCreateFileCapture_FFMPEG(const char* filename);
OPENCV_FFMPEG_API int cvSetCaptureProperty_FFMPEG(struct CvCapture_FFMPEG* cap,
                                                  int prop, double value);
OPENCV_FFMPEG_API double cvGetCaptureProperty_FFMPEG(struct CvCapture_FFMPEG* cap, int prop);
OPENCV_FFMPEG_API int cvGrabFrame_FFMPEG(struct CvCapture_FFMPEG* cap);
OPENCV_FFMPEG_API int cvRetrieveFrame_FFMPEG(struct CvCapture_FFMPEG* capture, unsigned char** data,
                                             int* step, int* width, int* height, int* cn);
OPENCV_FFMPEG_API void cvReleaseCapture_FFMPEG(struct CvCapture_FFMPEG** cap);

OPENCV_FFMPEG_API struct CvVideoWriter_FFMPEG* cvCreateVideoWriter_FFMPEG(const char* filename,
            int fourcc, double fps, int width, int height, int isColor );
OPENCV_FFMPEG_API int cvWriteFrame_FFMPEG(struct CvVideoWriter_FFMPEG* writer, const unsigned char* data,
                                          int step, int width, int height, int cn, int origin);
OPENCV_FFMPEG_API void cvReleaseVideoWriter_FFMPEG(struct CvVideoWriter_FFMPEG** writer);

#ifdef __cplusplus
}
#endif

#endif  // __OPENCV_FFMPEG_LEGACY_API_H__
