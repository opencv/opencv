// highgui to XAML bridge for OpenCV

// Copyright (c) Microsoft Open Technologies, Inc.
// All rights reserved.
//
// (3 - clause BSD License)
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
// following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
// following disclaimer in the documentation and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
// promote products derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "precomp.hpp"

#include <map>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <opencv2\highgui.hpp>
#include <opencv2\highgui\highgui_winrt.hpp>
#include "window_winrt_bridge.hpp"

#define CV_WINRT_NO_GUI_ERROR( funcname )       \
{                                               \
    cvError( CV_StsNotImplemented, funcname,    \
    "The function is not implemented. ",        \
    __FILE__, __LINE__ );                       \
}

#define CV_ERROR( Code, Msg )                                       \
{                                                                   \
    cvError( (Code), cvFuncName, Msg, __FILE__, __LINE__ );         \
};

/********************************** WinRT Specific API Implementation ******************************************/

// Initializes or overrides container contents with default XAML markup structure
void cv::winrt_initContainer(::Windows::UI::Xaml::Controls::Panel^ _container)
{
    HighguiBridge::getInstance().setContainer(_container);
}

/********************************** API Implementation *********************************************************/

CV_IMPL void cvShowImage(const char* name, const CvArr* arr)
{
    CV_FUNCNAME("cvShowImage");

    __BEGIN__;

    CvMat stub, *image;

    if (!name)
        CV_ERROR(CV_StsNullPtr, "NULL name");

    CvWindow* window = HighguiBridge::getInstance().namedWindow(name);

    if (!window || !arr)
        return;

    CV_CALL(image = cvGetMat(arr, &stub));

    //TODO: use approach from window_w32.cpp or cv::Mat(.., .., CV_8UC4)
    //      and cvtColor(.., .., CV_BGR2BGRA) to convert image here
    //      than beforehand.

    window->updateImage(image);
    HighguiBridge::getInstance().showWindow(window);

    __END__;
}

CV_IMPL int cvNamedWindow(const char* name, int flags)
{
    CV_FUNCNAME("cvNamedWindow");

    if (!name)
        CV_ERROR(CV_StsNullPtr, "NULL name");

    HighguiBridge::getInstance().namedWindow(name);

    return CV_OK;
}

CV_IMPL void cvDestroyWindow(const char* name)
{
    CV_FUNCNAME("cvDestroyWindow");

    if (!name)
        CV_ERROR(CV_StsNullPtr, "NULL name string");

    HighguiBridge::getInstance().destroyWindow(name);
}

CV_IMPL void cvDestroyAllWindows()
{
    HighguiBridge::getInstance().destroyAllWindows();
}

CV_IMPL int cvCreateTrackbar2(const char* trackbar_name, const char* window_name,
    int* val, int count, CvTrackbarCallback2 on_notify, void* userdata)
{
    CV_FUNCNAME("cvCreateTrackbar2");

    int pos = 0;

    if (!window_name || !trackbar_name)
        CV_ERROR(CV_StsNullPtr, "NULL window or trackbar name");

    if (count < 0)
        CV_ERROR(CV_StsOutOfRange, "Bad trackbar max value");

    CvWindow* window = HighguiBridge::getInstance().namedWindow(window_name);

    if (!window)
    {
        CV_ERROR(CV_StsNullPtr, "NULL window");
    }

    window->createSlider(trackbar_name, val, count, on_notify, userdata);

    return CV_OK;
}

CV_IMPL void cvSetTrackbarPos(const char* trackbar_name, const char* window_name, int pos)
{
    CV_FUNCNAME("cvSetTrackbarPos");

    CvTrackbar* trackbar = 0;

    if (trackbar_name == 0 || window_name == 0)
        CV_ERROR(CV_StsNullPtr, "NULL trackbar or window name");

    CvWindow* window = HighguiBridge::getInstance().findWindowByName(window_name);
    if (window)
        trackbar = window->findTrackbarByName(trackbar_name);

    if (trackbar)
        trackbar->setPosition(pos);
}

CV_IMPL void cvSetTrackbarMax(const char* trackbar_name, const char* window_name, int maxval)
{
    CV_FUNCNAME("cvSetTrackbarMax");

    if (maxval >= 0)
    {
        if (trackbar_name == 0 || window_name == 0)
            CV_ERROR(CV_StsNullPtr, "NULL trackbar or window name");

        CvTrackbar* trackbar = HighguiBridge::getInstance().findTrackbarByName(trackbar_name, window_name);

        if (trackbar)
            trackbar->setMaxPosition(maxval);
    }
}

CV_IMPL void cvSetTrackbarMin(const char* trackbar_name, const char* window_name, int minval)
{
    CV_FUNCNAME("cvSetTrackbarMin");

    if (minval >= 0)
    {
        if (trackbar_name == 0 || window_name == 0)
            CV_ERROR(CV_StsNullPtr, "NULL trackbar or window name");

        CvTrackbar* trackbar = HighguiBridge::getInstance().findTrackbarByName(trackbar_name, window_name);

        if (trackbar)
            trackbar->setMinPosition(minval);
    }
}

CV_IMPL int cvGetTrackbarPos(const char* trackbar_name, const char* window_name)
{
    int pos = -1;

    CV_FUNCNAME("cvGetTrackbarPos");

    if (trackbar_name == 0 || window_name == 0)
        CV_ERROR(CV_StsNullPtr, "NULL trackbar or window name");

    CvTrackbar* trackbar = HighguiBridge::getInstance().findTrackbarByName(trackbar_name, window_name);

    if (trackbar)
        pos = trackbar->getPosition();

    return pos;
}

/********************************** Not YET implemented API ****************************************************/

CV_IMPL int cvWaitKey(int delay)
{
    CV_WINRT_NO_GUI_ERROR("cvWaitKey");

    // see https://msdn.microsoft.com/en-us/library/windows/desktop/ms724411(v=vs.85).aspx
    int time0 = GetTickCount64();

    for (;;)
    {
        CvWindow* window;

        if (delay <= 0)
        {
            // TODO: implement appropriate logic here
        }
    }
}

CV_IMPL void cvSetMouseCallback(const char* window_name, CvMouseCallback on_mouse, void* param)
{
    CV_WINRT_NO_GUI_ERROR("cvSetMouseCallback");

    CV_FUNCNAME("cvSetMouseCallback");

    if (!window_name)
        CV_ERROR(CV_StsNullPtr, "NULL window name");

    CvWindow* window = HighguiBridge::getInstance().findWindowByName(window_name);
    if (!window)
        return;

    // TODO: implement appropriate logic here
}

/********************************** Disabled or not supported API **********************************************/

CV_IMPL void cvMoveWindow(const char* name, int x, int y)
{
    CV_WINRT_NO_GUI_ERROR("cvMoveWindow");
}

CV_IMPL void cvResizeWindow(const char* name, int width, int height)
{
    CV_WINRT_NO_GUI_ERROR("cvResizeWindow");
}

CV_IMPL int cvInitSystem(int, char**)
{
    CV_WINRT_NO_GUI_ERROR("cvInitSystem");
    return CV_StsNotImplemented;
}

CV_IMPL void* cvGetWindowHandle(const char*)
{
    CV_WINRT_NO_GUI_ERROR("cvGetWindowHandle");
    return (void*) CV_StsNotImplemented;
}

CV_IMPL const char* cvGetWindowName(void*)
{
    CV_WINRT_NO_GUI_ERROR("cvGetWindowName");
    return (const char*) CV_StsNotImplemented;
}

void cvSetModeWindow_WinRT(const char* name, double prop_value) {
    CV_WINRT_NO_GUI_ERROR("cvSetModeWindow");
}

double cvGetModeWindow_WinRT(const char* name) {
    CV_WINRT_NO_GUI_ERROR("cvGetModeWindow");
    return CV_StsNotImplemented;
}

CV_IMPL int cvStartWindowThread() {
    CV_WINRT_NO_GUI_ERROR("cvStartWindowThread");
    return CV_StsNotImplemented;
}
