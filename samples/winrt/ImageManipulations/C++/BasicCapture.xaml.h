//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

//
// BasicCapture.xaml.h
// Declaration of the BasicCapture class
//

#pragma once

#include "pch.h"
#include "BasicCapture.g.h"
#include "MainPage.xaml.h"

using namespace Windows::UI::Xaml;
using namespace Windows::UI::Xaml::Controls;
using namespace Windows::Graphics::Display;
using namespace Windows::UI::ViewManagement;
using namespace Windows::Devices::Enumeration;
#define VIDEO_FILE_NAME "video.mp4"
#define PHOTO_FILE_NAME "photo.jpg"
namespace SDKSample
{
    namespace MediaCapture
    {
        /// <summary>
        /// An empty page that can be used on its own or navigated to within a Frame.
        /// </summary>
    	[Windows::Foundation::Metadata::WebHostHidden]
        public ref class BasicCapture sealed
        {
        public:
            BasicCapture();
    
        protected:
            virtual void OnNavigatedTo(Windows::UI::Xaml::Navigation::NavigationEventArgs^ e) override;
            virtual void OnNavigatedFrom(Windows::UI::Xaml::Navigation::NavigationEventArgs^ e) override;
    
        private:
            MainPage^ rootPage;
            void ScenarioInit();
            void ScenarioReset();
    
            void Suspending(Object^ sender, Windows::ApplicationModel::SuspendingEventArgs^ e);
            void Resuming(Object^ sender, Object^ e);
    
            void btnStartDevice_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
            void SoundLevelChanged(Object^ sender, Object^ e);
            void RecordLimitationExceeded(Windows::Media::Capture::MediaCapture ^ mediaCapture);
            void Failed(Windows::Media::Capture::MediaCapture ^ mediaCapture, Windows::Media::Capture::MediaCaptureFailedEventArgs ^ args);
    
    
            void btnStartPreview_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
    
            void btnStartStopRecord_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
    
            void btnTakePhoto_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
    
            void SetupVideoDeviceControl(Windows::Media::Devices::MediaDeviceControl^ videoDeviceControl, Slider^ slider);
            void sldBrightness_ValueChanged(Platform::Object^ sender, Windows::UI::Xaml::Controls::Primitives::RangeBaseValueChangedEventArgs^ e);
            void sldContrast_ValueChanged(Platform::Object^ sender, Windows::UI::Xaml::Controls::Primitives::RangeBaseValueChangedEventArgs^ e);
    
            void ShowStatusMessage(Platform::String^ text);
            void ShowExceptionMessage(Platform::Exception^ ex);
    
            void EnableButton(bool enabled, Platform::String ^name);
            void SwitchRecordButtonContent();
    
            Platform::Agile<Windows::Media::Capture::MediaCapture> m_mediaCaptureMgr;
            Windows::Storage::StorageFile^ m_photoStorageFile;
            Windows::Storage::StorageFile^ m_recordStorageFile;
            bool m_bRecording;
            bool m_bEffectAdded;
            bool m_bSuspended;
            bool m_bPreviewing;
            Windows::UI::Xaml::WindowVisibilityChangedEventHandler ^m_visbilityHandler;
            Windows::Foundation::EventRegistrationToken m_eventRegistrationToken;
            bool m_currentScenarioLoaded;
        };
    }
}
