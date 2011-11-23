/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_HIGHGUI_HPP__
#define __OPENCV_HIGHGUI_HPP__

#include "opencv2/core/core.hpp"
#include "opencv2/core/gpumat.hpp"
#include "opencv2/highgui/highgui_c.h"

#ifdef __cplusplus

struct CvCapture;
struct CvVideoWriter;

namespace cv
{

enum { 
    // Flags for namedWindow
    WINDOW_NORMAL   = CV_WINDOW_NORMAL,   // the user can resize the window (no constraint) / also use to switch a fullscreen window to a normal size
    WINDOW_AUTOSIZE = CV_WINDOW_AUTOSIZE, // the user cannot resize the window, the size is constrainted by the image displayed
    WINDOW_OPENGL   = CV_WINDOW_OPENGL,   // window with opengl support

    // Flags for set / getWindowProperty
    WND_PROP_FULLSCREEN   = CV_WND_PROP_FULLSCREEN,  // fullscreen property
    WND_PROP_AUTOSIZE     = CV_WND_PROP_AUTOSIZE,    // autosize property
    WND_PROP_ASPECT_RATIO = CV_WND_PROP_ASPECTRATIO, // window's aspect ration
    WND_PROP_OPENGL       = CV_WND_PROP_OPENGL       // opengl support
};

CV_EXPORTS_W void namedWindow(const string& winname, int flags = WINDOW_AUTOSIZE);
CV_EXPORTS_W void destroyWindow(const string& winname);
CV_EXPORTS_W void destroyAllWindows();

CV_EXPORTS_W int startWindowThread();

CV_EXPORTS_W int waitKey(int delay = 0);

CV_EXPORTS_W void imshow(const string& winname, InputArray mat);

CV_EXPORTS_W void resizeWindow(const string& winname, int width, int height);
CV_EXPORTS_W void moveWindow(const string& winname, int x, int y);

CV_EXPORTS_W void setWindowProperty(const string& winname, int prop_id, double prop_value);//YV
CV_EXPORTS_W double getWindowProperty(const string& winname, int prop_id);//YV

enum
{
    EVENT_MOUSEMOVE      =0,
    EVENT_LBUTTONDOWN    =1,
    EVENT_RBUTTONDOWN    =2,
    EVENT_MBUTTONDOWN    =3,
    EVENT_LBUTTONUP      =4,
    EVENT_RBUTTONUP      =5,
    EVENT_MBUTTONUP      =6,
    EVENT_LBUTTONDBLCLK  =7,
    EVENT_RBUTTONDBLCLK  =8,
    EVENT_MBUTTONDBLCLK  =9
};

enum
{
    EVENT_FLAG_LBUTTON   =1,
    EVENT_FLAG_RBUTTON   =2,
    EVENT_FLAG_MBUTTON   =4,
    EVENT_FLAG_CTRLKEY   =8,
    EVENT_FLAG_SHIFTKEY  =16,
    EVENT_FLAG_ALTKEY    =32
};
        
typedef void (*MouseCallback)(int event, int x, int y, int flags, void* userdata);

//! assigns callback for mouse events
CV_EXPORTS void setMouseCallback(const string& winname, MouseCallback onMouse, void* userdata = 0);


typedef void (CV_CDECL *TrackbarCallback)(int pos, void* userdata);

CV_EXPORTS int createTrackbar(const string& trackbarname, const string& winname,
                              int* value, int count,
                              TrackbarCallback onChange = 0,
                              void* userdata = 0);

CV_EXPORTS_W int getTrackbarPos(const string& trackbarname, const string& winname);
CV_EXPORTS_W void setTrackbarPos(const string& trackbarname, const string& winname, int pos);

// OpenGL support

typedef void (CV_CDECL *OpenGLCallback)(void* userdata);
CV_EXPORTS void createOpenGLCallback(const string& winname, OpenGLCallback onOpenGlDraw, void* userdata = 0);

typedef void (CV_CDECL *OpenGlDrawCallback)(void* userdata);
static inline void setOpenGlDrawCallback(const string& winname, OpenGlDrawCallback onOpenGlDraw, void* userdata = 0)
{
    createOpenGLCallback(winname, onOpenGlDraw, userdata);
}

CV_EXPORTS void setOpenGlContext(const string& winname);

CV_EXPORTS void updateWindow(const string& winname);

CV_EXPORTS void imshow(const string& winname, const gpu::GlTexture& tex);
CV_EXPORTS void imshow(const string& winname, const gpu::GlBuffer& buf);
CV_EXPORTS void imshow(const string& winname, const gpu::GpuMat& d_mat);

CV_EXPORTS void pointCloudShow(const string& winname, const gpu::GlCamera& camera, const gpu::GlArrays& arr);
CV_EXPORTS void pointCloudShow(const string& winname, const gpu::GlCamera& camera, const gpu::GlBuffer& points, 
                               const gpu::GlBuffer& colors = gpu::GlBuffer(gpu::GlBuffer::ARRAY_BUFFER));
CV_EXPORTS void pointCloudShow(const string& winname, const gpu::GlCamera& camera, const gpu::GpuMat& points, 
                               const gpu::GpuMat& colors = gpu::GpuMat());
CV_EXPORTS void pointCloudShow(const string& winname, const gpu::GlCamera& camera, InputArray points, 
                               InputArray colors = noArray());

//Only for Qt

CV_EXPORTS CvFont fontQt(const string& nameFont, int pointSize=-1,
                         Scalar color=Scalar::all(0), int weight=CV_FONT_NORMAL,
                         int style=CV_STYLE_NORMAL, int spacing=0);
CV_EXPORTS void addText( const Mat& img, const string& text, Point org, CvFont font);

CV_EXPORTS void displayOverlay(const string& winname, const string& text, int delayms);
CV_EXPORTS void displayStatusBar(const string& winname, const string& text, int delayms);

CV_EXPORTS void saveWindowParameters(const string& windowName);
CV_EXPORTS void loadWindowParameters(const string& windowName);
CV_EXPORTS  int startLoop(int (*pt2Func)(int argc, char *argv[]), int argc, char* argv[]);
CV_EXPORTS  void stopLoop();

typedef void (CV_CDECL *ButtonCallback)(int state, void* userdata);
CV_EXPORTS int createButton( const string& bar_name, ButtonCallback on_change,
                             void* userdata=NULL, int type=CV_PUSH_BUTTON,
                             bool initial_button_state=0);

//-------------------------

enum
{
    // 8bit, color or not
    IMREAD_UNCHANGED  =-1,
    // 8bit, gray
    IMREAD_GRAYSCALE  =0,
    // ?, color
    IMREAD_COLOR      =1,
    // any depth, ?
    IMREAD_ANYDEPTH   =2,
    // ?, any color
    IMREAD_ANYCOLOR   =4
};
    
enum
{
    IMWRITE_JPEG_QUALITY =1,
    IMWRITE_PNG_COMPRESSION =16,
    IMWRITE_PNG_STRATEGY =17,
    IMWRITE_PNG_STRATEGY_DEFAULT =0,
    IMWRITE_PNG_STRATEGY_FILTERED =1,
    IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY =2,
    IMWRITE_PNG_STRATEGY_RLE =3,
    IMWRITE_PNG_STRATEGY_FIXED =4,
    IMWRITE_PXM_BINARY =32
};
    
CV_EXPORTS_W Mat imread( const string& filename, int flags=1 );
CV_EXPORTS_W bool imwrite( const string& filename, InputArray img,
              const vector<int>& params=vector<int>());
CV_EXPORTS_W Mat imdecode( InputArray buf, int flags );
CV_EXPORTS_W bool imencode( const string& ext, InputArray img,
                            CV_OUT vector<uchar>& buf,
                            const vector<int>& params=vector<int>());

#ifndef CV_NO_VIDEO_CAPTURE_CPP_API

template<> void CV_EXPORTS Ptr<CvCapture>::delete_obj();
template<> void CV_EXPORTS Ptr<CvVideoWriter>::delete_obj();

class CV_EXPORTS_W VideoCapture
{
public:
    CV_WRAP VideoCapture();
    CV_WRAP VideoCapture(const string& filename);
    CV_WRAP VideoCapture(int device);
    
    virtual ~VideoCapture();
    CV_WRAP virtual bool open(const string& filename);
    CV_WRAP virtual bool open(int device);
    CV_WRAP virtual bool isOpened() const;
    CV_WRAP virtual void release();
    
    CV_WRAP virtual bool grab();
    CV_WRAP virtual bool retrieve(CV_OUT Mat& image, int channel=0);
    virtual VideoCapture& operator >> (CV_OUT Mat& image);
    CV_WRAP virtual bool read(CV_OUT Mat& image);
    
    CV_WRAP virtual bool set(int propId, double value);
    CV_WRAP virtual double get(int propId);
    
protected:
    Ptr<CvCapture> cap;
};

    
class CV_EXPORTS_W VideoWriter
{
public:    
    CV_WRAP VideoWriter();
    CV_WRAP VideoWriter(const string& filename, int fourcc, double fps,
                Size frameSize, bool isColor=true);
    
    virtual ~VideoWriter();
    CV_WRAP virtual bool open(const string& filename, int fourcc, double fps,
                      Size frameSize, bool isColor=true);
    CV_WRAP virtual bool isOpened() const;
    virtual VideoWriter& operator << (const Mat& image);
    CV_WRAP virtual void write(const Mat& image);
    
protected:
    Ptr<CvVideoWriter> writer;
};

#endif

}

#endif

#endif
