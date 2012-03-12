#ifndef __OPENCV_FFMPEG_H__
#define __OPENCV_FFMPEG_H__

#ifdef __cplusplus
extern "C"
{
#endif

#if defined WIN32 || defined _WIN32
#define OPENCV_FFMPEG_API __declspec(dllexport)
#else
#define OPENCV_FFMPEG_API
#endif

enum
{
    CV_FFMPEG_CAP_PROP_POS_MSEC=0,
    CV_FFMPEG_CAP_PROP_POS_FRAMES=1,
    CV_FFMPEG_CAP_PROP_POS_AVI_RATIO=2,
    CV_FFMPEG_CAP_PROP_FRAME_WIDTH=3,
    CV_FFMPEG_CAP_PROP_FRAME_HEIGHT=4,
    CV_FFMPEG_CAP_PROP_FPS=5,
    CV_FFMPEG_CAP_PROP_FOURCC=6,
    CV_FFMPEG_CAP_PROP_FRAME_COUNT=7
};


OPENCV_FFMPEG_API struct CvCapture_FFMPEG* cvCreateFileCapture_FFMPEG(const char* filename);
OPENCV_FFMPEG_API struct CvCapture_FFMPEG_2* cvCreateFileCapture_FFMPEG_2(const char* filename);
OPENCV_FFMPEG_API int cvSetCaptureProperty_FFMPEG(struct CvCapture_FFMPEG* cap,
                                                  int prop, double value);
OPENCV_FFMPEG_API int cvSetCaptureProperty_FFMPEG_2(struct CvCapture_FFMPEG_2* cap,
                                                    int prop, double value);
OPENCV_FFMPEG_API double cvGetCaptureProperty_FFMPEG(struct CvCapture_FFMPEG* cap, int prop);
OPENCV_FFMPEG_API double cvGetCaptureProperty_FFMPEG_2(struct CvCapture_FFMPEG_2* cap, int prop);
OPENCV_FFMPEG_API int cvGrabFrame_FFMPEG(struct CvCapture_FFMPEG* cap);
OPENCV_FFMPEG_API int cvGrabFrame_FFMPEG_2(struct CvCapture_FFMPEG_2* cap);
OPENCV_FFMPEG_API int cvRetrieveFrame_FFMPEG(struct CvCapture_FFMPEG* capture, unsigned char** data,
                                             int* step, int* width, int* height, int* cn);
OPENCV_FFMPEG_API int cvRetrieveFrame_FFMPEG_2(struct CvCapture_FFMPEG_2* capture, unsigned char** data,
                                             int* step, int* width, int* height, int* cn);
OPENCV_FFMPEG_API void cvReleaseCapture_FFMPEG(struct CvCapture_FFMPEG** cap);
OPENCV_FFMPEG_API void cvReleaseCapture_FFMPEG_2(struct CvCapture_FFMPEG_2** cap);
OPENCV_FFMPEG_API struct CvVideoWriter_FFMPEG* cvCreateVideoWriter_FFMPEG(const char* filename,
            int fourcc, double fps, int width, int height, int isColor );
OPENCV_FFMPEG_API struct CvVideoWriter_FFMPEG_2* cvCreateVideoWriter_FFMPEG_2(const char* filename,
            int fourcc, double fps, int width, int height, int isColor );

OPENCV_FFMPEG_API int cvWriteFrame_FFMPEG(struct CvVideoWriter_FFMPEG* writer, const unsigned char* data,
                                          int step, int width, int height, int cn, int origin);

OPENCV_FFMPEG_API void cvReleaseVideoWriter_FFMPEG(struct CvVideoWriter_FFMPEG** writer);

typedef void* (*CvCreateFileCapture_Plugin)( const char* filename );
typedef void* (*CvCreateCameraCapture_Plugin)( int index );
typedef int (*CvGrabFrame_Plugin)( void* capture_handle );
typedef int (*CvRetrieveFrame_Plugin)( void* capture_handle, unsigned char** data, int* step,
                                       int* width, int* height, int* cn );
typedef int (*CvSetCaptureProperty_Plugin)( void* capture_handle, int prop_id, double value );
typedef double (*CvGetCaptureProperty_Plugin)( void* capture_handle, int prop_id );
typedef void (*CvReleaseCapture_Plugin)( void** capture_handle );
typedef void* (*CvCreateVideoWriter_Plugin)( const char* filename, int fourcc,
                                             double fps, int width, int height, int iscolor );
typedef int (*CvWriteFrame_Plugin)( void* writer_handle, const unsigned char* data, int step,
                                    int width, int height, int cn, int origin);
typedef void (*CvReleaseVideoWriter_Plugin)( void** writer );

#ifdef __cplusplus
}
#endif

#endif

