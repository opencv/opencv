#ifndef __OPENCV_FFMPEG_H__
#define __OPENCV_FFMPEG_H__

#ifdef __cplusplus
extern "C"
{
#endif

#if defined _WIN32
#   define OPENCV_FFMPEG_API __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#   define OPENCV_FFMPEG_API __attribute__ ((visibility ("default")))
#else
#   define OPENCV_FFMPEG_API
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
    CV_FFMPEG_CAP_PROP_FRAME_COUNT=7,
    CV_FFMPEG_CAP_PROP_FORMAT=8,
    CV_FFMPEG_CAP_PROP_SAR_NUM=40,
    CV_FFMPEG_CAP_PROP_SAR_DEN=41,
    CV_FFMPEG_CAP_PROP_CODEC_PIXEL_FORMAT=46,
    CV_FFMPEG_CAP_PROP_BITRATE=47
};

typedef struct CvCapture_FFMPEG CvCapture_FFMPEG;
typedef struct CvVideoWriter_FFMPEG CvVideoWriter_FFMPEG;

OPENCV_FFMPEG_API struct CvCapture_FFMPEG* cvCreateFileCapture_FFMPEG(const char* filename);
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

typedef CvCapture_FFMPEG* (*CvCreateFileCapture_Plugin)( const char* filename );
typedef CvCapture_FFMPEG* (*CvCreateCameraCapture_Plugin)( int index );
typedef int (*CvGrabFrame_Plugin)( CvCapture_FFMPEG* capture_handle );
typedef int (*CvRetrieveFrame_Plugin)( CvCapture_FFMPEG* capture_handle, unsigned char** data, int* step,
                                       int* width, int* height, int* cn );
typedef int (*CvSetCaptureProperty_Plugin)( CvCapture_FFMPEG* capture_handle, int prop_id, double value );
typedef double (*CvGetCaptureProperty_Plugin)( CvCapture_FFMPEG* capture_handle, int prop_id );
typedef void (*CvReleaseCapture_Plugin)( CvCapture_FFMPEG** capture_handle );
typedef CvVideoWriter_FFMPEG* (*CvCreateVideoWriter_Plugin)( const char* filename, int fourcc,
                                             double fps, int width, int height, int iscolor );
typedef int (*CvWriteFrame_Plugin)( CvVideoWriter_FFMPEG* writer_handle, const unsigned char* data, int step,
                                    int width, int height, int cn, int origin);
typedef void (*CvReleaseVideoWriter_Plugin)( CvVideoWriter_FFMPEG** writer );

/*
 * For CUDA encoder
 */

OPENCV_FFMPEG_API struct OutputMediaStream_FFMPEG* create_OutputMediaStream_FFMPEG(const char* fileName, int width, int height, double fps);
OPENCV_FFMPEG_API void release_OutputMediaStream_FFMPEG(struct OutputMediaStream_FFMPEG* stream);
OPENCV_FFMPEG_API void write_OutputMediaStream_FFMPEG(struct OutputMediaStream_FFMPEG* stream, unsigned char* data, int size, int keyFrame);

typedef struct OutputMediaStream_FFMPEG* (*Create_OutputMediaStream_FFMPEG_Plugin)(const char* fileName, int width, int height, double fps);
typedef void (*Release_OutputMediaStream_FFMPEG_Plugin)(struct OutputMediaStream_FFMPEG* stream);
typedef void (*Write_OutputMediaStream_FFMPEG_Plugin)(struct OutputMediaStream_FFMPEG* stream, unsigned char* data, int size, int keyFrame);

/*
 * For CUDA decoder
 */

OPENCV_FFMPEG_API struct InputMediaStream_FFMPEG* create_InputMediaStream_FFMPEG(const char* fileName, int* codec, int* chroma_format, int* width, int* height);
OPENCV_FFMPEG_API void release_InputMediaStream_FFMPEG(struct InputMediaStream_FFMPEG* stream);
OPENCV_FFMPEG_API int read_InputMediaStream_FFMPEG(struct InputMediaStream_FFMPEG* stream, unsigned char** data, int* size, int* endOfFile);

typedef struct InputMediaStream_FFMPEG* (*Create_InputMediaStream_FFMPEG_Plugin)(const char* fileName, int* codec, int* chroma_format, int* width, int* height);
typedef void (*Release_InputMediaStream_FFMPEG_Plugin)(struct InputMediaStream_FFMPEG* stream);
typedef int (*Read_InputMediaStream_FFMPEG_Plugin)(struct InputMediaStream_FFMPEG* stream, unsigned char** data, int* size, int* endOfFile);

#ifdef __cplusplus
}
#endif

#endif
