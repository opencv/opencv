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

/**
@defgroup highgui High-level GUI

While OpenCV was designed for use in full-scale applications and can be used within functionally
rich UI frameworks (such as Qt\*, WinForms\*, or Cocoa\*) or without any UI at all, sometimes there
it is required to try functionality quickly and visualize the results. This is what the HighGUI
module has been designed for.

It provides easy interface to:

-   Create and manipulate windows that can display images and "remember" their content (no need to
    handle repaint events from OS).
-   Add trackbars to the windows, handle simple mouse events as well as keyboard commands.

@{
    @defgroup highgui_opengl OpenGL support
    @defgroup highgui_qt Qt New Functions

    ![image](pics/qtgui.png)

    This figure explains new functionality implemented with Qt\* GUI. The new GUI provides a statusbar,
    a toolbar, and a control panel. The control panel can have trackbars and buttonbars attached to it.
    If you cannot see the control panel, press Ctrl+P or right-click any Qt window and select **Display
    properties window**.

    -   To attach a trackbar, the window name parameter must be NULL.

    -   To attach a buttonbar, a button must be created. If the last bar attached to the control panel
        is a buttonbar, the new button is added to the right of the last button. If the last bar
        attached to the control panel is a trackbar, or the control panel is empty, a new buttonbar is
        created. Then, a new button is attached to it.

    See below the example used to generate the figure:
    @code
        int main(int argc, char *argv[])
            int value = 50;
            int value2 = 0;

            cvNamedWindow("main1",CV_WINDOW_NORMAL);
            cvNamedWindow("main2",CV_WINDOW_AUTOSIZE | CV_GUI_NORMAL);

            cvCreateTrackbar( "track1", "main1", &value, 255,  NULL);//OK tested
            char* nameb1 = "button1";
            char* nameb2 = "button2";
            cvCreateButton(nameb1,callbackButton,nameb1,CV_CHECKBOX,1);

            cvCreateButton(nameb2,callbackButton,nameb2,CV_CHECKBOX,0);
            cvCreateTrackbar( "track2", NULL, &value2, 255, NULL);
            cvCreateButton("button5",callbackButton1,NULL,CV_RADIOBOX,0);
            cvCreateButton("button6",callbackButton2,NULL,CV_RADIOBOX,1);

            cvSetMouseCallback( "main2",on_mouse,NULL );

            IplImage* img1 = cvLoadImage("files/flower.jpg");
            IplImage* img2 = cvCreateImage(cvGetSize(img1),8,3);
            CvCapture* video = cvCaptureFromFile("files/hockey.avi");
            IplImage* img3 = cvCreateImage(cvGetSize(cvQueryFrame(video)),8,3);

            while(cvWaitKey(33) != 27)
            {
                cvAddS(img1,cvScalarAll(value),img2);
                cvAddS(cvQueryFrame(video),cvScalarAll(value2),img3);
                cvShowImage("main1",img2);
                cvShowImage("main2",img3);
            }

            cvDestroyAllWindows();
            cvReleaseImage(&img1);
            cvReleaseImage(&img2);
            cvReleaseImage(&img3);
            cvReleaseCapture(&video);
            return 0;
        }
    @endcode


    @defgroup highgui_winrt WinRT support

    This figure explains new functionality implemented with WinRT GUI. The new GUI provides an Image control,
    and a slider panel. Slider panel holds trackbars attached to it.

    Sliders are attached below the image control. Every new slider is added below the previous one.

    See below the example used to generate the figure:
    @code
        void sample_app::MainPage::ShowWindow()
        {
            static cv::String windowName("sample");
            cv::winrt_initContainer(this->cvContainer);
            cv::namedWindow(windowName); // not required

            cv::Mat image = cv::imread("Assets/sample.jpg");
            cv::Mat converted = cv::Mat(image.rows, image.cols, CV_8UC4);
            cvtColor(image, converted, CV_BGR2BGRA);
            cv::imshow(windowName, converted); // this will create window if it hasn't been created before

            int state = 42;
            cv::TrackbarCallback callback = [](int pos, void* userdata)
            {
                if (pos == 0) {
                    cv::destroyWindow(windowName);
                }
            };
            cv::TrackbarCallback callbackTwin = [](int pos, void* userdata)
            {
                if (pos >= 70) {
                    cv::destroyAllWindows();
                }
            };
            cv::createTrackbar("Sample trackbar", windowName, &state, 100, callback);
            cv::createTrackbar("Twin brother", windowName, &state, 100, callbackTwin);
        }
    @endcode

    @defgroup highgui_c C API
@}
*/

///////////////////////// graphical user interface //////////////////////////
namespace cv
{

//! @addtogroup highgui
//! @{

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

/** @brief Creates a window.

@param winname Name of the window in the window caption that may be used as a window identifier.
@param flags Flags of the window. The supported flags are:
> -   **WINDOW_NORMAL** If this is set, the user can resize the window (no constraint).
> -   **WINDOW_AUTOSIZE** If this is set, the window size is automatically adjusted to fit the
>     displayed image (see imshow ), and you cannot change the window size manually.
> -   **WINDOW_OPENGL** If this is set, the window will be created with OpenGL support.

The function namedWindow creates a window that can be used as a placeholder for images and
trackbars. Created windows are referred to by their names.

If a window with the same name already exists, the function does nothing.

You can call destroyWindow or destroyAllWindows to close the window and de-allocate any associated
memory usage. For a simple program, you do not really have to call these functions because all the
resources and windows of the application are closed automatically by the operating system upon exit.

@note

Qt backend supports additional flags:
 -   **CV_WINDOW_NORMAL or CV_WINDOW_AUTOSIZE:** CV_WINDOW_NORMAL enables you to resize the
     window, whereas CV_WINDOW_AUTOSIZE adjusts automatically the window size to fit the
     displayed image (see imshow ), and you cannot change the window size manually.
 -   **CV_WINDOW_FREERATIO or CV_WINDOW_KEEPRATIO:** CV_WINDOW_FREERATIO adjusts the image
     with no respect to its ratio, whereas CV_WINDOW_KEEPRATIO keeps the image ratio.
 -   **CV_GUI_NORMAL or CV_GUI_EXPANDED:** CV_GUI_NORMAL is the old way to draw the window
     without statusbar and toolbar, whereas CV_GUI_EXPANDED is a new enhanced GUI.
By default, flags == CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED
 */
CV_EXPORTS_W void namedWindow(const String& winname, int flags = WINDOW_AUTOSIZE);

/** @brief Destroys a window.

@param winname Name of the window to be destroyed.

The function destroyWindow destroys the window with the given name.
 */
CV_EXPORTS_W void destroyWindow(const String& winname);

/** @brief Destroys all of the HighGUI windows.

The function destroyAllWindows destroys all of the opened HighGUI windows.
 */
CV_EXPORTS_W void destroyAllWindows();

CV_EXPORTS_W int startWindowThread();

/** @brief Waits for a pressed key.

@param delay Delay in milliseconds. 0 is the special value that means "forever".

The function waitKey waits for a key event infinitely (when \f$\texttt{delay}\leq 0\f$ ) or for delay
milliseconds, when it is positive. Since the OS has a minimum time between switching threads, the
function will not wait exactly delay ms, it will wait at least delay ms, depending on what else is
running on your computer at that time. It returns the code of the pressed key or -1 if no key was
pressed before the specified time had elapsed.

@note

This function is the only method in HighGUI that can fetch and handle events, so it needs to be
called periodically for normal event processing unless HighGUI is used within an environment that
takes care of event processing.

@note

The function only works if there is at least one HighGUI window created and the window is active.
If there are several HighGUI windows, any of them can be active.
 */
CV_EXPORTS_W int waitKey(int delay = 0);

/** @brief Displays an image in the specified window.

@param winname Name of the window.
@param mat Image to be shown.

The function imshow displays an image in the specified window. If the window was created with the
CV_WINDOW_AUTOSIZE flag, the image is shown with its original size, however it is still limited by the screen resolution.
Otherwise, the image is scaled to fit the window. The function may scale the image, depending on its depth:

-   If the image is 8-bit unsigned, it is displayed as is.
-   If the image is 16-bit unsigned or 32-bit integer, the pixels are divided by 256. That is, the
    value range [0,255\*256] is mapped to [0,255].
-   If the image is 32-bit floating-point, the pixel values are multiplied by 255. That is, the
    value range [0,1] is mapped to [0,255].

If window was created with OpenGL support, imshow also support ogl::Buffer , ogl::Texture2D and
cuda::GpuMat as input.

If the window was not created before this function, it is assumed creating a window with CV_WINDOW_AUTOSIZE.

If you need to show an image that is bigger than the screen resolution, you will need to call namedWindow("", WINDOW_NORMAL) before the imshow.

@note This function should be followed by waitKey function which displays the image for specified
milliseconds. Otherwise, it won't display the image. For example, waitKey(0) will display the window
infinitely until any keypress (it is suitable for image display). waitKey(25) will display a frame
for 25 ms, after which display will be automatically closed. (If you put it in a loop to read
videos, it will display the video frame-by-frame)

@note

[Windows Backend Only] Pressing Ctrl+C will copy the image to the clipboard.

 */
CV_EXPORTS_W void imshow(const String& winname, InputArray mat);

/** @brief Resizes window to the specified size

@param winname Window name
@param width The new window width
@param height The new window height

@note

-   The specified window size is for the image area. Toolbars are not counted.
-   Only windows created without CV_WINDOW_AUTOSIZE flag can be resized.
 */
CV_EXPORTS_W void resizeWindow(const String& winname, int width, int height);

/** @brief Moves window to the specified position

@param winname Window name
@param x The new x-coordinate of the window
@param y The new y-coordinate of the window
 */
CV_EXPORTS_W void moveWindow(const String& winname, int x, int y);

/** @brief Changes parameters of a window dynamically.

@param winname Name of the window.
@param prop_id Window property to edit. The following operation flags are available:
 -   **CV_WND_PROP_FULLSCREEN** Change if the window is fullscreen ( CV_WINDOW_NORMAL or
     CV_WINDOW_FULLSCREEN ).
 -   **CV_WND_PROP_AUTOSIZE** Change if the window is resizable (CV_WINDOW_NORMAL or
     CV_WINDOW_AUTOSIZE ).
 -   **CV_WND_PROP_ASPECTRATIO** Change if the aspect ratio of the image is preserved (
     CV_WINDOW_FREERATIO or CV_WINDOW_KEEPRATIO ).
@param prop_value New value of the window property. The following operation flags are available:
 -   **CV_WINDOW_NORMAL** Change the window to normal size or make the window resizable.
 -   **CV_WINDOW_AUTOSIZE** Constrain the size by the displayed image. The window is not
     resizable.
 -   **CV_WINDOW_FULLSCREEN** Change the window to fullscreen.
 -   **CV_WINDOW_FREERATIO** Make the window resizable without any ratio constraints.
 -   **CV_WINDOW_KEEPRATIO** Make the window resizable, but preserve the proportions of the
     displayed image.

The function setWindowProperty enables changing properties of a window.
 */
CV_EXPORTS_W void setWindowProperty(const String& winname, int prop_id, double prop_value);

/** @brief Updates window title
*/
CV_EXPORTS_W void setWindowTitle(const String& winname, const String& title);

/** @brief Provides parameters of a window.

@param winname Name of the window.
@param prop_id Window property to retrieve. The following operation flags are available:
 -   **CV_WND_PROP_FULLSCREEN** Change if the window is fullscreen ( CV_WINDOW_NORMAL or
     CV_WINDOW_FULLSCREEN ).
 -   **CV_WND_PROP_AUTOSIZE** Change if the window is resizable (CV_WINDOW_NORMAL or
     CV_WINDOW_AUTOSIZE ).
 -   **CV_WND_PROP_ASPECTRATIO** Change if the aspect ratio of the image is preserved
     (CV_WINDOW_FREERATIO or CV_WINDOW_KEEPRATIO ).

See setWindowProperty to know the meaning of the returned values.

The function getWindowProperty returns properties of a window.
 */
CV_EXPORTS_W double getWindowProperty(const String& winname, int prop_id);

/** @brief Sets mouse handler for the specified window

@param winname Window name
@param onMouse Mouse callback. See OpenCV samples, such as
<https://github.com/Itseez/opencv/tree/master/samples/cpp/ffilldemo.cpp>, on how to specify and
use the callback.
@param userdata The optional parameter passed to the callback.
 */
CV_EXPORTS void setMouseCallback(const String& winname, MouseCallback onMouse, void* userdata = 0);

/** @brief Gets the mouse-wheel motion delta, when handling mouse-wheel events EVENT_MOUSEWHEEL and
EVENT_MOUSEHWHEEL.

@param flags The mouse callback flags parameter.

For regular mice with a scroll-wheel, delta will be a multiple of 120. The value 120 corresponds to
a one notch rotation of the wheel or the threshold for action to be taken and one such action should
occur for each delta. Some high-precision mice with higher-resolution freely-rotating wheels may
generate smaller values.

For EVENT_MOUSEWHEEL positive and negative values mean forward and backward scrolling,
respectively. For EVENT_MOUSEHWHEEL, where available, positive and negative values mean right and
left scrolling, respectively.

With the C API, the macro CV_GET_WHEEL_DELTA(flags) can be used alternatively.

@note

Mouse-wheel events are currently supported only on Windows.
 */
CV_EXPORTS int getMouseWheelDelta(int flags);

/** @brief Creates a trackbar and attaches it to the specified window.

@param trackbarname Name of the created trackbar.
@param winname Name of the window that will be used as a parent of the created trackbar.
@param value Optional pointer to an integer variable whose value reflects the position of the
slider. Upon creation, the slider position is defined by this variable.
@param count Maximal position of the slider. The minimal position is always 0.
@param onChange Pointer to the function to be called every time the slider changes position. This
function should be prototyped as void Foo(int,void\*); , where the first parameter is the trackbar
position and the second parameter is the user data (see the next parameter). If the callback is
the NULL pointer, no callbacks are called, but only value is updated.
@param userdata User data that is passed as is to the callback. It can be used to handle trackbar
events without using global variables.

The function createTrackbar creates a trackbar (a slider or range control) with the specified name
and range, assigns a variable value to be a position synchronized with the trackbar and specifies
the callback function onChange to be called on the trackbar position change. The created trackbar is
displayed in the specified window winname.

@note

**[Qt Backend Only]** winname can be empty (or NULL) if the trackbar should be attached to the
control panel.

Clicking the label of each trackbar enables editing the trackbar values manually.

@note

-   An example of using the trackbar functionality can be found at
    opencv_source_code/samples/cpp/connected_components.cpp
 */
CV_EXPORTS int createTrackbar(const String& trackbarname, const String& winname,
                              int* value, int count,
                              TrackbarCallback onChange = 0,
                              void* userdata = 0);

/** @brief Returns the trackbar position.

@param trackbarname Name of the trackbar.
@param winname Name of the window that is the parent of the trackbar.

The function returns the current position of the specified trackbar.

@note

**[Qt Backend Only]** winname can be empty (or NULL) if the trackbar is attached to the control
panel.

 */
CV_EXPORTS_W int getTrackbarPos(const String& trackbarname, const String& winname);

/** @brief Sets the trackbar position.

@param trackbarname Name of the trackbar.
@param winname Name of the window that is the parent of trackbar.
@param pos New position.

The function sets the position of the specified trackbar in the specified window.

@note

**[Qt Backend Only]** winname can be empty (or NULL) if the trackbar is attached to the control
panel.
 */
CV_EXPORTS_W void setTrackbarPos(const String& trackbarname, const String& winname, int pos);

/** @brief Sets the trackbar maximum position.

@param trackbarname Name of the trackbar.
@param winname Name of the window that is the parent of trackbar.
@param maxval New maximum position.

The function sets the maximum position of the specified trackbar in the specified window.

@note

**[Qt Backend Only]** winname can be empty (or NULL) if the trackbar is attached to the control
panel.
 */
CV_EXPORTS_W void setTrackbarMax(const String& trackbarname, const String& winname, int maxval);

//! @addtogroup highgui_opengl OpenGL support
//! @{

CV_EXPORTS void imshow(const String& winname, const ogl::Texture2D& tex);

/** @brief Sets a callback function to be called to draw on top of displayed image.

@param winname Name of the window.
@param onOpenGlDraw Pointer to the function to be called every frame. This function should be
prototyped as void Foo(void\*) .
@param userdata Pointer passed to the callback function. *(Optional)*

The function setOpenGlDrawCallback can be used to draw 3D data on the window. See the example of
callback function below: :
@code
    void on_opengl(void* param)
    {
        glLoadIdentity();

        glTranslated(0.0, 0.0, -1.0);

        glRotatef( 55, 1, 0, 0 );
        glRotatef( 45, 0, 1, 0 );
        glRotatef( 0, 0, 0, 1 );

        static const int coords[6][4][3] = {
            { { +1, -1, -1 }, { -1, -1, -1 }, { -1, +1, -1 }, { +1, +1, -1 } },
            { { +1, +1, -1 }, { -1, +1, -1 }, { -1, +1, +1 }, { +1, +1, +1 } },
            { { +1, -1, +1 }, { +1, -1, -1 }, { +1, +1, -1 }, { +1, +1, +1 } },
            { { -1, -1, -1 }, { -1, -1, +1 }, { -1, +1, +1 }, { -1, +1, -1 } },
            { { +1, -1, +1 }, { -1, -1, +1 }, { -1, -1, -1 }, { +1, -1, -1 } },
            { { -1, -1, +1 }, { +1, -1, +1 }, { +1, +1, +1 }, { -1, +1, +1 } }
        };

        for (int i = 0; i < 6; ++i) {
                    glColor3ub( i*20, 100+i*10, i*42 );
                    glBegin(GL_QUADS);
                    for (int j = 0; j < 4; ++j) {
                            glVertex3d(0.2 * coords[i][j][0], 0.2 * coords[i][j][1], 0.2 * coords[i][j][2]);
                    }
                    glEnd();
        }
    }
@endcode
 */
CV_EXPORTS void setOpenGlDrawCallback(const String& winname, OpenGlDrawCallback onOpenGlDraw, void* userdata = 0);

/** @brief Sets the specified window as current OpenGL context.

@param winname Window name
 */
CV_EXPORTS void setOpenGlContext(const String& winname);

/** @brief Force window to redraw its context and call draw callback ( setOpenGlDrawCallback ).

@param winname Window name
 */
CV_EXPORTS void updateWindow(const String& winname);

//! @} highgui_opengl

//! @addtogroup highgui_qt
//! @{
// Only for Qt

struct QtFont
{
    const char* nameFont;  // Qt: nameFont
    Scalar      color;     // Qt: ColorFont -> cvScalar(blue_component, green_component, red_component[, alpha_component])
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

/** @brief Creates the font to draw a text on an image.

@param nameFont Name of the font. The name should match the name of a system font (such as
*Times*). If the font is not found, a default one is used.
@param pointSize Size of the font. If not specified, equal zero or negative, the point size of the
font is set to a system-dependent default value. Generally, this is 12 points.
@param color Color of the font in BGRA where A = 255 is fully transparent. Use the macro CV _ RGB
for simplicity.
@param weight Font weight. The following operation flags are available:
 -   **CV_FONT_LIGHT** Weight of 25
 -   **CV_FONT_NORMAL** Weight of 50
 -   **CV_FONT_DEMIBOLD** Weight of 63
 -   **CV_FONT_BOLD** Weight of 75
 -   **CV_FONT_BLACK** Weight of 87

 You can also specify a positive integer for better control.
@param style Font style. The following operation flags are available:
 -   **CV_STYLE_NORMAL** Normal font
 -   **CV_STYLE_ITALIC** Italic font
 -   **CV_STYLE_OBLIQUE** Oblique font
@param spacing Spacing between characters. It can be negative or positive.

The function fontQt creates a CvFont object. This CvFont is not compatible with putText .

A basic usage of this function is the following: :
@code
    CvFont font = fontQt(''Times'');
    addText( img1, ``Hello World !'', Point(50,50), font);
@endcode
 */
CV_EXPORTS QtFont fontQt(const String& nameFont, int pointSize = -1,
                         Scalar color = Scalar::all(0), int weight = QT_FONT_NORMAL,
                         int style = QT_STYLE_NORMAL, int spacing = 0);

/** @brief Creates the font to draw a text on an image.

@param img 8-bit 3-channel image where the text should be drawn.
@param text Text to write on an image.
@param org Point(x,y) where the text should start on an image.
@param font Font to use to draw a text.

The function addText draws *text* on an image *img* using a specific font *font* (see example fontQt
)
 */
CV_EXPORTS void addText( const Mat& img, const String& text, Point org, const QtFont& font);

/** @brief Displays a text on a window image as an overlay for a specified duration.

@param winname Name of the window.
@param text Overlay text to write on a window image.
@param delayms The period (in milliseconds), during which the overlay text is displayed. If this
function is called before the previous overlay text timed out, the timer is restarted and the text
is updated. If this value is zero, the text never disappears.

The function displayOverlay displays useful information/tips on top of the window for a certain
amount of time *delayms*. The function does not modify the image, displayed in the window, that is,
after the specified delay the original content of the window is restored.
 */
CV_EXPORTS void displayOverlay(const String& winname, const String& text, int delayms = 0);

/** @brief Displays a text on the window statusbar during the specified period of time.

@param winname Name of the window.
@param text Text to write on the window statusbar.
@param delayms Duration (in milliseconds) to display the text. If this function is called before
the previous text timed out, the timer is restarted and the text is updated. If this value is
zero, the text never disappears.

The function displayOverlay displays useful information/tips on top of the window for a certain
amount of time *delayms* . This information is displayed on the window statusbar (the window must be
created with the CV_GUI_EXPANDED flags).
 */
CV_EXPORTS void displayStatusBar(const String& winname, const String& text, int delayms = 0);

/** @brief Saves parameters of the specified window.

@param windowName Name of the window.

The function saveWindowParameters saves size, location, flags, trackbars value, zoom and panning
location of the window window_name .
 */
CV_EXPORTS void saveWindowParameters(const String& windowName);

/** @brief Loads parameters of the specified window.

@param windowName Name of the window.

The function loadWindowParameters loads size, location, flags, trackbars value, zoom and panning
location of the window window_name .
 */
CV_EXPORTS void loadWindowParameters(const String& windowName);

CV_EXPORTS  int startLoop(int (*pt2Func)(int argc, char *argv[]), int argc, char* argv[]);

CV_EXPORTS  void stopLoop();

/** @brief Attaches a button to the control panel.

@param  bar_name
   Name of the button.
@param on_change Pointer to the function to be called every time the button changes its state.
This function should be prototyped as void Foo(int state,\*void); . *state* is the current state
of the button. It could be -1 for a push button, 0 or 1 for a check/radio box button.
@param userdata Pointer passed to the callback function.
@param type Optional type of the button.
 -   **CV_PUSH_BUTTON** Push button
 -   **CV_CHECKBOX** Checkbox button
 -   **CV_RADIOBOX** Radiobox button. The radiobox on the same buttonbar (same line) are
     exclusive, that is only one can be selected at a time.
@param initial_button_state Default state of the button. Use for checkbox and radiobox. Its
value could be 0 or 1. *(Optional)*

The function createButton attaches a button to the control panel. Each button is added to a
buttonbar to the right of the last button. A new buttonbar is created if nothing was attached to the
control panel before, or if the last element attached to the control panel was a trackbar.

See below various examples of the createButton function call: :
@code
    createButton(NULL,callbackButton);//create a push button "button 0", that will call callbackButton.
    createButton("button2",callbackButton,NULL,CV_CHECKBOX,0);
    createButton("button3",callbackButton,&value);
    createButton("button5",callbackButton1,NULL,CV_RADIOBOX);
    createButton("button6",callbackButton2,NULL,CV_PUSH_BUTTON,1);
@endcode
*/
CV_EXPORTS int createButton( const String& bar_name, ButtonCallback on_change,
                             void* userdata = 0, int type = QT_PUSH_BUTTON,
                             bool initial_button_state = false);

//! @} highgui_qt

//! @} highgui

} // cv

#ifndef DISABLE_OPENCV_24_COMPATIBILITY
#include "opencv2/highgui/highgui_c.h"
#endif

#endif
