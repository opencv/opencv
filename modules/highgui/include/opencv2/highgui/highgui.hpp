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
#include "opencv2/highgui/highgui_c.h"

#ifdef __cplusplus

struct CvCapture;
struct CvVideoWriter;

namespace cv
{

enum { WINDOW_AUTOSIZE=1 };

CV_EXPORTS void namedWindow( const string& winname, int flags CV_DEFAULT(WINDOW_AUTOSIZE) );
CV_EXPORTS void destroyWindow( const string& winname );
CV_EXPORTS int startWindowThread();

CV_EXPORTS void setWindowProperty(const string& winname, int prop_id, double prop_value);//YV
CV_EXPORTS double getWindowProperty(const string& winname, int prop_id);//YV


//Only for Qt
//------------------------
CV_EXPORTS CvFont fontQt(const string& nameFont, int pointSize CV_DEFAULT(-1), Scalar color CV_DEFAULT(Scalar::all(0)), int weight CV_DEFAULT(CV_FONT_NORMAL),  int style CV_DEFAULT(CV_STYLE_NORMAL), int spacing CV_DEFAULT(0));
CV_EXPORTS void addText( const Mat& img, const char* text, Point org, CvFont font);

CV_EXPORTS void displayOverlay(const string& winname, const string& text, int delayms);
CV_EXPORTS void displayStatusBar(const string& winname, const string& text, int delayms);

typedef void (CV_CDECL *OpenGLCallback)(void* userdata);
CV_EXPORTS void createOpenGLCallback(const string& winname, CvOpenGLCallback callbackOpenGL, void* userdata CV_DEFAULT(0));

CV_EXPORTS void saveWindowParameters(const string& windowName);
CV_EXPORTS void loadWindowParameters(const string& windowName);
CV_EXPORTS  int startLoop(int (*pt2Func)(int argc, char *argv[]), int argc, char* argv[]);
CV_EXPORTS  void stopLoop();

typedef void (CV_CDECL *ButtonCallback)(int state, void* userdata);
CV_EXPORTS int createButton( const string& bar_name, ButtonCallback on_change , void* userdata CV_DEFAULT(NULL), int type CV_DEFAULT(CV_PUSH_BUTTON), bool initial_button_state CV_DEFAULT(0));
//-------------------------

CV_EXPORTS void imshow( const string& winname, const Mat& mat );

typedef void (CV_CDECL *TrackbarCallback)(int pos, void* userdata);

CV_EXPORTS int createTrackbar( const string& trackbarname, const string& winname,
                               int* value, int count,
                               TrackbarCallback onChange CV_DEFAULT(0),
                               void* userdata CV_DEFAULT(0));

CV_EXPORTS int getTrackbarPos( const string& trackbarname, const string& winname );
CV_EXPORTS void setTrackbarPos( const string& trackbarname, const string& winname, int pos );

typedef void (*MouseCallback )(int event, int x, int y, int flags, void* param);

//! assigns callback for mouse events
CV_EXPORTS void setMouseCallback( const string& windowName, MouseCallback onMouse, void* param=0);
    
CV_EXPORTS Mat imread( const string& filename, int flags=1 );
CV_EXPORTS bool imwrite( const string& filename, const Mat& img,
              const vector<int>& params=vector<int>());
CV_EXPORTS Mat imdecode( const Mat& buf, int flags );
CV_EXPORTS bool imencode( const string& ext, const Mat& img,
                          vector<uchar>& buf,
                          const vector<int>& params=vector<int>());

CV_EXPORTS int waitKey(int delay=0);

#ifndef CV_NO_VIDEO_CAPTURE_CPP_API

template<> void CV_EXPORTS Ptr<CvCapture>::delete_obj();
template<> void CV_EXPORTS Ptr<CvVideoWriter>::delete_obj();

class CV_EXPORTS VideoCapture
{
public:
    VideoCapture();
    VideoCapture(const string& filename);
    VideoCapture(int device);
    
    virtual ~VideoCapture();
    virtual bool open(const string& filename);
    virtual bool open(int device);
    virtual bool isOpened() const;
    virtual void release();
    
    virtual bool grab();
    virtual bool retrieve(Mat& image, int channel=0);
    virtual VideoCapture& operator >> (Mat& image);
    
    virtual bool set(int propId, double value);
    virtual double get(int propId);
    
protected:
    Ptr<CvCapture> cap;
};

    
class CV_EXPORTS VideoWriter
{
public:    
    VideoWriter();
    VideoWriter(const string& filename, int fourcc, double fps, Size frameSize, bool isColor=true);
    
    virtual ~VideoWriter();
    virtual bool open(const string& filename, int fourcc, double fps, Size frameSize, bool isColor=true);
    virtual bool isOpened() const;
    virtual VideoWriter& operator << (const Mat& image);
    
protected:
    Ptr<CvVideoWriter> writer;
};

#endif

}

#endif

#endif
