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

#define CV_WINRT_NO_GUI_ERROR( funcname )  CV_Error( cv::Error::StsNotImplemented, "The function is not implemented"))

/********************************** WinRT Specific API Implementation ******************************************/

// Initializes or overrides container contents with default XAML markup structure
void cv::winrt_initContainer(::Windows::UI::Xaml::Controls::Panel^ _container)
{
    HighguiBridge::getInstance().setContainer(_container);
}

/********************************** API Implementation *********************************************************/

void showImageImpl(const char* name, const CvArr* arr)
{
    CV_FUNCNAME("showImageImpl");

    __BEGIN__;

    CvMat stub, *image;

    if (!name)
        CV_Error(cv::Error::StsNullPtr, "NULL name");

    CvWindow* window = HighguiBridge::getInstance().namedWindow(name);

    if (!window || !arr)
        return;

    CV_CALL(image = cvGetMat(arr, &stub));

    //TODO: use approach from window_w32.cpp or cv::Mat(.., .., CV_8UC4)
    //      and cvtColor(.., .., cv::COLOR_BGR2BGRA) to convert image here
    //      than beforehand.

    window->updateImage(image);
    HighguiBridge::getInstance().showWindow(window);

    __END__;
}

int namedWindowImpl(const char* name, int flags)
{
    CV_FUNCNAME("namedWindowImpl");

    if (!name)
        CV_Error(cv::Error::StsNullPtr, "NULL name");

    HighguiBridge::getInstance().namedWindow(name);

    return CV_OK;
}

void destroyWindowImpl(const char* name)
{
    CV_FUNCNAME("destroyWindowImpl");

    if (!name)
        CV_Error(cv::Error::StsNullPtr, "NULL name string");

    HighguiBridge::getInstance().destroyWindow(name);
}

void destroyAllWindowsImpl()
{
    HighguiBridge::getInstance().destroyAllWindows();
}

int createTrackbar2Impl(const char* trackbar_name, const char* window_name,
    int* val, int count, CvTrackbarCallback2 on_notify, void* userdata)
{
    CV_FUNCNAME("createTrackbar2Impl");

    int pos = 0;

    if (!window_name || !trackbar_name)
        CV_Error(cv::Error::StsNullPtr, "NULL window or trackbar name");

    if (count < 0)
        CV_Error(cv::Error::StsOutOfRange, "Bad trackbar max value");

    CvWindow* window = HighguiBridge::getInstance().namedWindow(window_name);

    if (!window)
    {
        CV_Error(cv::Error::StsNullPtr, "NULL window");
    }

    window->createSlider(trackbar_name, val, count, on_notify, userdata);

    return CV_OK;
}

void setTrackbarPosImpl(const char* trackbar_name, const char* window_name, int pos)
{
    CV_FUNCNAME("setTrackbarPosImpl");

    CvTrackbar* trackbar = 0;

    if (trackbar_name == 0 || window_name == 0)
        CV_Error(cv::Error::StsNullPtr, "NULL trackbar or window name");

    CvWindow* window = HighguiBridge::getInstance().findWindowByName(window_name);
    if (window)
        trackbar = window->findTrackbarByName(trackbar_name);

    if (trackbar)
        trackbar->setPosition(pos);
}

void setTrackbarMaxImpl(const char* trackbar_name, const char* window_name, int maxval)
{
    CV_FUNCNAME("setTrackbarMaxImpl");

    if (maxval >= 0)
    {
        if (trackbar_name == 0 || window_name == 0)
            CV_Error(cv::Error::StsNullPtr, "NULL trackbar or window name");

        CvTrackbar* trackbar = HighguiBridge::getInstance().findTrackbarByName(trackbar_name, window_name);

        if (trackbar)
            trackbar->setMaxPosition(maxval);
    }
}

void setTrackbarMinImpl(const char* trackbar_name, const char* window_name, int minval)
{
    CV_FUNCNAME("setTrackbarMinImpl");

    if (minval >= 0)
    {
        if (trackbar_name == 0 || window_name == 0)
            CV_Error(cv::Error::StsNullPtr, "NULL trackbar or window name");

        CvTrackbar* trackbar = HighguiBridge::getInstance().findTrackbarByName(trackbar_name, window_name);

        if (trackbar)
            trackbar->setMinPosition(minval);
    }
}

int getTrackbarPosImpl(const char* trackbar_name, const char* window_name)
{
    int pos = -1;

    CV_FUNCNAME("getTrackbarPosImpl");

    if (trackbar_name == 0 || window_name == 0)
        CV_Error(cv::Error::StsNullPtr, "NULL trackbar or window name");

    CvTrackbar* trackbar = HighguiBridge::getInstance().findTrackbarByName(trackbar_name, window_name);

    if (trackbar)
        pos = trackbar->getPosition();

    return pos;
}

/********************************** Not YET implemented API ****************************************************/

int waitKeyImpl(int delay)
{
    CV_WINRT_NO_GUI_ERROR("waitKeyImpl");

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

void setMouseCallbackImpl(const char* window_name, CvMouseCallback on_mouse, void* param)
{
    CV_WINRT_NO_GUI_ERROR("setMouseCallbackImpl");

    CV_FUNCNAME("setMouseCallbackImpl");

    if (!window_name)
        CV_Error(cv::Error::StsNullPtr, "NULL window name");

    CvWindow* window = HighguiBridge::getInstance().findWindowByName(window_name);
    if (!window)
        return;

    // TODO: implement appropriate logic here
}

/********************************** Disabled or not supported API **********************************************/

void moveWindowImpl(const char* name, int x, int y)
{
    CV_WINRT_NO_GUI_ERROR("moveWindowImpl");
}

void resizeWindowImpl(const char* name, int width, int height)
{
    CV_WINRT_NO_GUI_ERROR("resizeWindowImpl");
}

void cvSetModeWindow_WinRT(const char* name, double prop_value) {
    CV_WINRT_NO_GUI_ERROR("cvSetModeWindow");
}

double cvGetModeWindow_WinRT(const char* name) {
    CV_WINRT_NO_GUI_ERROR("cvGetModeWindow");
    return cv::Error::StsNotImplemented;
}
