// Video support for Windows Runtime

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

#include <ppl.h>
#include <functional>
#include <concrt.h>
#include <agile.h>
#include "opencv2/core/cvdef.h"

namespace cv
{

//! @addtogroup videoio_winrt
//! @{

enum {
    OPEN_CAMERA = 300,
    CLOSE_CAMERA,
    UPDATE_IMAGE_ELEMENT,
    SHOW_TRACKBAR
};

/********************************** WinRT API ************************************************/

template <typename ...Args>
CV_EXPORTS void winrt_startMessageLoop(std::function<void(Args...)>&& callback, Args... args);

template <typename ...Args>
CV_EXPORTS void winrt_startMessageLoop(void callback(Args...), Args... args);

/** @brief
@note
    Starts (1) frame-grabbing loop and (2) message loop
    1. Function passed as an argument must implement common OCV reading frames
       pattern (see cv::VideoCapture documentation) AND call cv::winrt_imgshow().
    2. Message processing loop required to overcome WinRT container and type
       conversion restrictions. OCV provides default implementation
       Here is how the class can be used:
@code
    void cvMain()
    {
        Mat frame;
        VideoCapture cam;
        cam.open(0);

        while (1)
        {
            cam >> frame;

            // don't reprocess the same frame again
            if (!cam.grab()) continue;

            // your processing logic goes here

            // obligatory step to get XAML image component updated
            winrt_imshow();
        }
    }

    MainPage::MainPage()
    {
        InitializeComponent();

        cv::winrt_setFrameContainer(cvImage);
        cv::winrt_startMessageLoop(cvMain);
    }
@endcode
*/
template
CV_EXPORTS void winrt_startMessageLoop(void callback(void));

/** @brief
@note
    Must be called from WinRT specific callback to handle image grabber state.
    Here is how the class can be used:
@code
    MainPage::MainPage()
    {
        // ...
        Window::Current->VisibilityChanged += ref new Windows::UI::Xaml::WindowVisibilityChangedEventHandler(this, &Application::MainPage::OnVisibilityChanged);
        // ...
    }

    void Application::MainPage::OnVisibilityChanged(Platform::Object ^sender,
        Windows::UI::Core::VisibilityChangedEventArgs ^e)
    {
        cv::winrt_onVisibilityChanged(e->Visible);
    }
@endcode
*/
CV_EXPORTS void winrt_onVisibilityChanged(bool visible);

/** @brief
@note
    Must be called to assign WinRT control holding image you're working with.
    Code sample is available for winrt_startMessageLoop().
*/
CV_EXPORTS void winrt_setFrameContainer(::Windows::UI::Xaml::Controls::Image^ image);

/** @brief
@note
    Must be called to update attached image source.
    Code sample is available for winrt_startMessageLoop().
*/
CV_EXPORTS void winrt_imshow();

//! @} videoio_winrt

} // cv
