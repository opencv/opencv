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

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"


///////////////////////// graphical user interface //////////////////////////
namespace cv
{

// Flags for namedWindow
enum { WINDOW_NORMAL     = 0x00000000, // the user can resize the window (no constraint) / also use to switch a fullscreen window to a normal size
       WINDOW_AUTOSIZE   = 0x00000001, // the user cannot resize the window, the size is constrainted by the image displayed
       WINDOW_OPENGL     = 0x00001000, // window with opengl support

       WINDOW_FULLSCREEN = 1,          // change the window to fullscreen
       WINDOW_FREERATIO  = 0x00000100, // the image expends as much as it can (no ratio constraint)
       WINDOW_KEEPRATIO  = 0x00000000  // the ratio of the image is respected
     };

// Flags for set / getWindowProperty
enum { WND_PROP_FULLSCREEN   = 0, // fullscreen property    (can be WINDOW_NORMAL or WINDOW_FULLSCREEN)
       WND_PROP_AUTOSIZE     = 1, // autosize property      (can be WINDOW_NORMAL or WINDOW_AUTOSIZE)
       WND_PROP_ASPECT_RATIO = 2, // window's aspect ration (can be set to WINDOW_FREERATIO or WINDOW_KEEPRATIO);
       WND_PROP_OPENGL       = 3  // opengl support
     };

enum { EVENT_MOUSEMOVE      = 0,
       EVENT_LBUTTONDOWN    = 1,
       EVENT_RBUTTONDOWN    = 2,
       EVENT_MBUTTONDOWN    = 3,
       EVENT_LBUTTONUP      = 4,
       EVENT_RBUTTONUP      = 5,
       EVENT_MBUTTONUP      = 6,
       EVENT_LBUTTONDBLCLK  = 7,
       EVENT_RBUTTONDBLCLK  = 8,
       EVENT_MBUTTONDBLCLK  = 9,
       EVENT_MOUSEWHEEL     = 10,
       EVENT_MOUSEHWHEEL    = 11
     };

enum { EVENT_FLAG_LBUTTON   = 1,
       EVENT_FLAG_RBUTTON   = 2,
       EVENT_FLAG_MBUTTON   = 4,
       EVENT_FLAG_CTRLKEY   = 8,
       EVENT_FLAG_SHIFTKEY  = 16,
       EVENT_FLAG_ALTKEY    = 32
     };

// Qt font
enum {  QT_FONT_LIGHT           = 25, //QFont::Light,
        QT_FONT_NORMAL          = 50, //QFont::Normal,
        QT_FONT_DEMIBOLD        = 63, //QFont::DemiBold,
        QT_FONT_BOLD            = 75, //QFont::Bold,
        QT_FONT_BLACK           = 87  //QFont::Black
     };

// Qt font style
enum {  QT_STYLE_NORMAL         = 0, //QFont::StyleNormal,
        QT_STYLE_ITALIC         = 1, //QFont::StyleItalic,
        QT_STYLE_OBLIQUE        = 2  //QFont::StyleOblique
     };

// Qt "button" type
enum { QT_PUSH_BUTTON = 0,
       QT_CHECKBOX    = 1,
       QT_RADIOBOX    = 2
     };


typedef void (*MouseCallback)(int event, int x, int y, int flags, void* userdata);
typedef void (*TrackbarCallback)(int pos, void* userdata);
typedef void (*OpenGlDrawCallback)(void* userdata);
typedef void (*ButtonCallback)(int state, void* userdata);


CV_EXPORTS_W void namedWindow(const String& winname, int flags = WINDOW_AUTOSIZE);

CV_EXPORTS_W void destroyWindow(const String& winname);

CV_EXPORTS_W void destroyAllWindows();

CV_EXPORTS_W int startWindowThread();

CV_EXPORTS_W int waitKey(int delay = 0);

CV_EXPORTS_W void imshow(const String& winname, InputArray mat);

CV_EXPORTS_W void resizeWindow(const String& winname, int width, int height);

CV_EXPORTS_W void moveWindow(const String& winname, int x, int y);

CV_EXPORTS_W void setWindowProperty(const String& winname, int prop_id, double prop_value);

CV_EXPORTS_W double getWindowProperty(const String& winname, int prop_id);

//! assigns callback for mouse events
CV_EXPORTS void setMouseCallback(const String& winname, MouseCallback onMouse, void* userdata = 0);

CV_EXPORTS int getMouseWheelDelta(int flags);

CV_EXPORTS int createTrackbar(const String& trackbarname, const String& winname,
                              int* value, int count,
                              TrackbarCallback onChange = 0,
                              void* userdata = 0);

CV_EXPORTS_W int getTrackbarPos(const String& trackbarname, const String& winname);

CV_EXPORTS_W void setTrackbarPos(const String& trackbarname, const String& winname, int pos);


// OpenGL support
CV_EXPORTS void imshow(const String& winname, const ogl::Texture2D& tex);

CV_EXPORTS void setOpenGlDrawCallback(const String& winname, OpenGlDrawCallback onOpenGlDraw, void* userdata = 0);

CV_EXPORTS void setOpenGlContext(const String& winname);

CV_EXPORTS void updateWindow(const String& winname);


// Only for Qt

struct QtFont
{
    const char* nameFont;  // Qt: nameFont
    Scalar      color;     // Qt: ColorFont -> cvScalar(blue_component, green_component, red\_component[, alpha_component])
    int         font_face; // Qt: bool italic
    const int*  ascii;     // font data and metrics
    const int*  greek;
    const int*  cyrillic;
    float       hscale, vscale;
    float       shear;     // slope coefficient: 0 - normal, >0 - italic
    int         thickness; // Qt: weight
    float       dx;        // horizontal interval between letters
    int         line_type; // Qt: PointSize
};

CV_EXPORTS QtFont fontQt(const String& nameFont, int pointSize = -1,
                         Scalar color = Scalar::all(0), int weight = QT_FONT_NORMAL,
                         int style = QT_STYLE_NORMAL, int spacing = 0);

CV_EXPORTS void addText( const Mat& img, const String& text, Point org, const QtFont& font);

CV_EXPORTS void displayOverlay(const String& winname, const String& text, int delayms = 0);

CV_EXPORTS void displayStatusBar(const String& winname, const String& text, int delayms = 0);

CV_EXPORTS void saveWindowParameters(const String& windowName);

CV_EXPORTS void loadWindowParameters(const String& windowName);

CV_EXPORTS  int startLoop(int (*pt2Func)(int argc, char *argv[]), int argc, char* argv[]);

CV_EXPORTS  void stopLoop();

CV_EXPORTS int createButton( const String& bar_name, ButtonCallback on_change,
                             void* userdata = 0, int type = QT_PUSH_BUTTON,
                             bool initial_button_state = false);

} // cv
#endif
