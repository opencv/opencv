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

#pragma once

#include <map>
#include <opencv2\core.hpp>

using namespace Windows::UI::Xaml::Controls;

class CvWindow;
class CvTrackbar;

class HighguiBridge
{
public:

    /** @brief Instantiates a Highgui singleton (Meyers type).

    The function Instantiates a Highgui singleton (Meyers type) and returns reference to that instance.
    */
    static HighguiBridge& getInstance();

    /** @brief Finds window by name and returns the reference to it.

    @param name Name of the window.

    The function finds window by name and returns the reference to it. Returns nullptr
    if window with specified name is not found or name argument is null.
    */
    CvWindow*           findWindowByName(cv::String name);

    /** @brief Returns reference to the trackbar(slider) registered within window with a provided name.

    @param name Name of the window.

    The function returns reference to the trackbar(slider) registered within window with a provided name.
    Returns nullptr if trackbar with specified name is not found or window reference is nullptr.
    */
    CvTrackbar*         findTrackbarByName(cv::String trackbarName, cv::String windowName);

    /** @brief Converts cv::String to Platform::String.

    @param name String to convert.

    The function converts cv::String to Platform::String.
    Returns nullptr if conversion fails.
    */
    Platform::String^   convertString(cv::String name);

    /** @brief Creates window if there is no window with this name, otherwise returns existing window.

    @param name Window name.

    The function creates window if there is no window with this name, otherwise returns existing window.
    */
    CvWindow*           namedWindow(cv::String name);

    /** @brief Shows provided window.

    The function shows provided window: makes provided window current, removes current container
    contents and shows current window by putting it as a container content.
    */
    void                showWindow(CvWindow* window);

    /** @brief Destroys window if there exists window with this name, otherwise does nothing.

    @param name Window name.

    The function destroys window if there exists window with this name, otherwise does nothing.
    If window being destroyed is the current one, it will be hidden by clearing the window container.
    */
    void                destroyWindow(cv::String name);

    /** @brief Destroys all windows.

    The function destroys all windows.
    */
    void                destroyAllWindows();

    /** @brief Assigns container used to display windows.

    @param _container Container reference.

    The function assigns container used to display windows.
    */
    void                setContainer(Windows::UI::Xaml::Controls::Panel^ _container);

private:

    // Meyers singleton
    HighguiBridge(const HighguiBridge &);
    HighguiBridge() {
        windowsMap = new std::map<cv::String, CvWindow*>();
    };

    /** @brief Creates window if there is no window with this name.

    @param name Window name.

    The function creates window if there is no window with this name.
    */
    CvWindow*           createWindow(cv::String name);

    /** @brief Cleans current container contents.

    The function cleans current container contents.
    */
    void                cleanContainer();

    // see https://msdn.microsoft.com/en-US/library/windows/apps/xaml/hh700103.aspx
    // see https://msdn.microsoft.com/ru-ru/library/windows.foundation.collections.aspx
    std::map<cv::String, CvWindow*>*    windowsMap;
    CvWindow*                           currentWindow;

    // Holds current container/content to manipulate with
    Windows::UI::Xaml::Controls::Panel^ container;
};

class CvTrackbar
{
public:
    CvTrackbar(cv::String name, Slider^ slider, CvWindow* parent);
    ~CvTrackbar();

    double  getPosition();
    void    setPosition(double pos);
    double  getMaxPosition();
    void    setMaxPosition(double pos);
    double  getMinPosition();
    void    setMinPosition(double pos);
    Slider^ getSlider();
    void    setSlider(Slider^ pos);

    CvTrackbarCallback2 callback;

private:
    cv::String  name;
    Slider^     slider;
    CvWindow*   parent;
};

class CvWindow
{
public:
    CvWindow(cv::String name, int flag = CV_WINDOW_NORMAL);
    ~CvWindow();

    /** @brief NOTE: prototype.

    Should create button if there is no button with this name already.
    */
    void            createButton(cv::String name);

    /** @brief Creates slider if there is no slider with this name already.

    The function creates slider if there is no slider with this name already OR resets
    provided values for the existing one.
    */
    void            createSlider(cv::String name, int* val, int count, CvTrackbarCallback2 on_notify, void* userdata);

    /** @brief Updates window image.

    @param src Image data object reference.

    The function updates window image. If argument is null or image control is not found - does nothing.
    */
    void            updateImage(CvMat* arr);

    /** @brief Returns reference to the trackbar(slider) registered within provided window.

    @param name Name of the window.

    The function returns reference to the trackbar(slider) registered within provided window.
    Returns nullptr if trackbar with specified name is not found or window reference is nullptr.
    */
    CvTrackbar*     findTrackbarByName(cv::String name);
    Page^           getPage();

private:
    cv::String name;

    // Holds image data in CV format
    CvMat* imageData;

    // Map of all sliders assigned to this window
    std::map<cv::String, CvTrackbar*>*  sliderMap;

    // Window contents holder
    Page^ page;

    // Image control displayed by this window
    Image^ imageControl;

    // Container for sliders
    Panel^ sliderPanel;

    // Container for buttons
    // TODO: prototype, not available via API
    Panel^ buttonPanel;

    // Holds image width to arrange other UI elements.
    // Required since imageData->width value gets recalculated when processing
    int imageWidth;

    // Default markup for the container content allowing for proper components placement
    static const Platform::String^ markupContent;

    // Default Slider size, fallback solution for unexpected edge cases
    static const double sliderDefaultWidth;
};
