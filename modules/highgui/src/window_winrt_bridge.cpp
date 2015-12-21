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

#include "opencv2\highgui\highgui_winrt.hpp"
#include "window_winrt_bridge.hpp"

#include <collection.h>
#include <Robuffer.h>   // Windows::Storage::Streams::IBufferByteAccess

using namespace Microsoft::WRL;             // ComPtr
using namespace Windows::Storage::Streams;  // IBuffer
using namespace Windows::UI::Xaml;
using namespace Windows::UI::Xaml::Controls;
using namespace Windows::UI::Xaml::Media::Imaging;

using namespace ::std;

/***************************** Constants ****************************************/

// Default markup for the container content allowing for proper components placement
const Platform::String^ CvWindow::markupContent =
"<Page \n" \
"    xmlns = \"http://schemas.microsoft.com/winfx/2006/xaml/presentation\"\n" \
"    xmlns:x = \"http://schemas.microsoft.com/winfx/2006/xaml\" >\n" \
"    <StackPanel Name=\"Container\" Orientation=\"Vertical\" Width=\"Auto\" Height=\"Auto\" HorizontalAlignment=\"Left\" VerticalAlignment=\"Top\" Visibility=\"Visible\">\n" \
"        <Image Name=\"cvImage\" Width=\"Auto\" Height=\"Auto\" Margin=\"10\" HorizontalAlignment=\"Left\" VerticalAlignment=\"Top\" Visibility=\"Visible\"/>\n" \
"        <StackPanel Name=\"cvTrackbar\" Height=\"Auto\" Width=\"Auto\" Orientation=\"Vertical\" Visibility=\"Visible\"/>\n" \
"        <StackPanel Name=\"cvButton\" Height=\"Auto\" Width=\"Auto\" Orientation=\"Horizontal\" Visibility=\"Visible\"/>\n" \
"    </StackPanel>\n" \
"</Page>";

const double CvWindow::sliderDefaultWidth = 100;

/***************************** HighguiBridge class ******************************/

HighguiBridge& HighguiBridge::getInstance()
{
    static HighguiBridge instance;
    return instance;
}

void HighguiBridge::setContainer(Windows::UI::Xaml::Controls::Panel^ container)
{
    this->container = container;
}

CvWindow* HighguiBridge::findWindowByName(cv::String name)
{
    auto search = windowsMap->find(name);
    if (search != windowsMap->end()) {
        return search->second;
    }

    return nullptr;
}

CvTrackbar* HighguiBridge::findTrackbarByName(cv::String trackbar_name, cv::String window_name)
{
    CvWindow* window = findWindowByName(window_name);

    if (window)
        return window->findTrackbarByName(trackbar_name);

    return nullptr;
}

Platform::String^ HighguiBridge::convertString(cv::String name)
{
    auto data = name.c_str();
    int bufferSize = MultiByteToWideChar(CP_UTF8, 0, data, -1, nullptr, 0);
    auto wide = std::make_unique<wchar_t[]>(bufferSize);
    if (0 == MultiByteToWideChar(CP_UTF8, 0, data, -1, wide.get(), bufferSize))
        return nullptr;

    std::wstring* stdStr = new std::wstring(wide.get());
    return ref new Platform::String(stdStr->c_str());
}

void HighguiBridge::cleanContainer()
{
    container->Children->Clear();
}

void HighguiBridge::showWindow(CvWindow* window)
{
    currentWindow = window;
    cleanContainer();
    HighguiBridge::getInstance().container->Children->Append(window->getPage());
}

CvWindow* HighguiBridge::namedWindow(cv::String name) {

    CvWindow* window = HighguiBridge::getInstance().findWindowByName(name.c_str());
    if (!window)
    {
        window = createWindow(name);
    }

    return window;
}

void HighguiBridge::destroyWindow(cv::String name)
{
    auto window = windowsMap->find(name);
    if (window != windowsMap->end())
    {
        // Check if deleted window is the one currently displayed
        // and clear container if this is the case
        if (window->second == currentWindow)
        {
            cleanContainer();
        }

        windowsMap->erase(window);
    }
}

void HighguiBridge::destroyAllWindows()
{
    cleanContainer();
    windowsMap->clear();
}

CvWindow* HighguiBridge::createWindow(cv::String name)
{
    CvWindow* window = new CvWindow(name);
    windowsMap->insert(std::pair<cv::String, CvWindow*>(name, window));

    return window;
}

/***************************** CvTrackbar class *********************************/

CvTrackbar::CvTrackbar(cv::String name, Slider^ slider, CvWindow* parent) : name(name), slider(slider), parent(parent) {}

CvTrackbar::~CvTrackbar() {}

void CvTrackbar::setPosition(double pos)
{
    if (pos < 0)
        pos = 0;

    if (pos > slider->Maximum)
        pos = slider->Maximum;

    slider->Value = pos;
}

void CvTrackbar::setMaxPosition(double pos)
{
    //slider->Minimum is initialized with 0
    if (pos < slider->Minimum)
        pos = slider->Minimum;

    slider->Maximum = pos;
}

void CvTrackbar::setMinPosition(double pos)
{
    if (pos < 0)
        pos = 0;
    //Min is always less than Max.
    if (pos > slider->Maximum)
        pos = slider->Maximum;
    slider->Minimum = pos;
}

void CvTrackbar::setSlider(Slider^ slider) {
    if (slider)
        this->slider = slider;
}

double CvTrackbar::getPosition()
{
    return slider->Value;
}

double CvTrackbar::getMaxPosition()
{
    return slider->Maximum;
}

double CvTrackbar::getMinPosition()
{
    return slider->Minimum;
}

Slider^ CvTrackbar::getSlider()
{
    return slider;
}

/***************************** CvWindow class ***********************************/

CvWindow::CvWindow(cv::String name, int flags) : name(name)
{
    this->page = (Page^)Windows::UI::Xaml::Markup::XamlReader::Load(const_cast<Platform::String^>(markupContent));
    this->sliderMap = new std::map<cv::String, CvTrackbar*>();

    sliderPanel = (Panel^)page->FindName("cvTrackbar");
    imageControl = (Image^)page->FindName("cvImage");
    buttonPanel = (Panel^)page->FindName("cvButton");

    // Required to adapt controls to the size of the image.
    // System calculates image control width first, after that we can
    // update other controls
    imageControl->Loaded += ref new Windows::UI::Xaml::RoutedEventHandler(
        [=](Platform::Object^ sender,
        Windows::UI::Xaml::RoutedEventArgs^ e)
    {
        // Need to update sliders with appropriate width
        for (auto iter = sliderMap->begin(); iter != sliderMap->end(); ++iter) {
            iter->second->getSlider()->Width = imageControl->ActualWidth;
        }

        // Need to update buttons with appropriate width
        // TODO: implement when adding buttons
    });

}

CvWindow::~CvWindow() {}

void CvWindow::createSlider(cv::String name, int* val, int count, CvTrackbarCallback2 on_notify, void* userdata)
{
    CvTrackbar* trackbar = findTrackbarByName(name);

    // Creating slider if name is new or reusing the existing one
    Slider^ slider = !trackbar ? ref new Slider() : trackbar->getSlider();

    slider->Header = HighguiBridge::getInstance().convertString(name);

    // Making slider the same size as the image control or setting minimal size.
    // This is added to cover potential edge cases because:
    //   1. Fist clause will not be true until the second call to any container-updating API
    //      e.g. cv::createTrackbar, cv:imshow or cv::namedWindow
    //   2. Second clause will work but should be immediately overridden by Image->Loaded callback,
    //      see CvWindow ctor.
    if (this->imageControl->ActualWidth > 0) {
        // One would use double.NaN for auto-stretching but there is no such constant in C++/CX
        // see https://msdn.microsoft.com/en-us/library/windows/apps/windows.ui.xaml.frameworkelement.width
        slider->Width = this->imageControl->ActualWidth;
    } else {
        // This value would never be used/seen on the screen unless there is something wrong with the image.
        // Although this code actually gets called, slider width will be overridden in the callback after
        // Image control is loaded. See callback implementation in CvWindow ctor.
        slider->Width = sliderDefaultWidth;
    }
    slider->Value = *val;
    slider->Maximum = count;
    slider->Visibility = Windows::UI::Xaml::Visibility::Visible;
    slider->Margin = Windows::UI::Xaml::ThicknessHelper::FromLengths(10, 10, 10, 0);
    slider->HorizontalAlignment = Windows::UI::Xaml::HorizontalAlignment::Left;

    if (!trackbar)
    {
        if (!sliderPanel) return;

        // Adding slider to the list for current window
        CvTrackbar* trackbar = new CvTrackbar(name, slider, this);
        trackbar->callback = on_notify;
        slider->ValueChanged +=
            ref new Controls::Primitives::RangeBaseValueChangedEventHandler(
            [=](Platform::Object^ sender,
                Windows::UI::Xaml::Controls::Primitives::RangeBaseValueChangedEventArgs^ e)
            {
                Slider^ slider = (Slider^)sender;
                trackbar->callback(slider->Value, nullptr);
            });
        this->sliderMap->insert(std::pair<cv::String, CvTrackbar*>(name, trackbar));

        // Adding slider to the window
        sliderPanel->Children->Append(slider);
    }
}

CvTrackbar* CvWindow::findTrackbarByName(cv::String name)
{
    auto search = sliderMap->find(name);
    if (search != sliderMap->end()) {
        return search->second;
    }

    return nullptr;
}

void CvWindow::updateImage(CvMat* src)
{
    if (!imageControl) return;

    this->imageData = src;
    this->imageWidth = src->width;

    // Create the WriteableBitmap
    WriteableBitmap^ bitmap = ref new WriteableBitmap(src->cols, src->rows);

    // Get access to the pixels
    IBuffer^ buffer = bitmap->PixelBuffer;
    unsigned char* dstPixels;

    // Obtain IBufferByteAccess
    ComPtr<IBufferByteAccess> pBufferByteAccess;
    ComPtr<IInspectable> pBuffer((IInspectable*)buffer);
    pBuffer.As(&pBufferByteAccess);

    // Get pointer to pixel bytes
    pBufferByteAccess->Buffer(&dstPixels);
    memcpy(dstPixels, src->data.ptr, CV_ELEM_SIZE(src->type) * src->cols*src->rows);

    // Set the bitmap to the Image element
    imageControl->Source = bitmap;
}

Page^ CvWindow::getPage()
{
    return page;
}

//TODO: prototype, not in use yet
void CvWindow::createButton(cv::String name)
{
    if (!buttonPanel) return;

    Button^ b = ref new Button();
    b->Content = HighguiBridge::getInstance().convertString(name);
    b->Width = 260;
    b->Height = 80;
    b->Click += ref new Windows::UI::Xaml::RoutedEventHandler(
        [=](Platform::Object^ sender,
            Windows::UI::Xaml::RoutedEventArgs^ e)
    {
        Button^ button = (Button^)sender;
        // TODO: more logic here...
    });

    buttonPanel->Children->Append(b);
}

// end