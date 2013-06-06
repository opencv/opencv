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
// AudioCapture.xaml.h
// Declaration of the AudioCapture class
//

#pragma once

#include "pch.h"
#include "AudioCapture.g.h"
#include "MainPage.xaml.h"

#define AUDIO_FILE_NAME "audio.mp4"

namespace SDKSample
{
    namespace MediaCapture
    {
        /// <summary>
        /// An empty page that can be used on its own or navigated to within a Frame.
        /// </summary>
    	[Windows::Foundation::Metadata::WebHostHidden]
        public ref class AudioCapture sealed
        {
        public:
            AudioCapture();
    
        protected:
            virtual void OnNavigatedTo(Windows::UI::Xaml::Navigation::NavigationEventArgs^ e) override;
            virtual void OnNavigatedFrom(Windows::UI::Xaml::Navigation::NavigationEventArgs^ e) override;
        private:
            MainPage^ rootPage;
    
            void ScenarioInit();
            void ScenarioReset();
    
            void SoundLevelChanged(Object^ sender, Object^ e);
            void RecordLimitationExceeded(Windows::Media::Capture::MediaCapture ^ mediaCapture);
            void Failed(Windows::Media::Capture::MediaCapture ^ mediaCapture, Windows::Media::Capture::MediaCaptureFailedEventArgs ^ args);
    
            void btnStartDevice_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
    
            void btnStartPreview_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
    
            void btnStartStopRecord_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
    
            void ShowStatusMessage(Platform::String^ text);
            void ShowExceptionMessage(Platform::Exception^ ex);
    
            void EnableButton(bool enabled, Platform::String ^name);
            void SwitchRecordButtonContent();
    
            Platform::Agile<Windows::Media::Capture::MediaCapture> m_mediaCaptureMgr;
            Windows::Storage::StorageFile^ m_photoStorageFile;
            Windows::Storage::StorageFile^ m_recordStorageFile;
            bool m_bRecording;
            bool m_bSuspended;
            Windows::Foundation::EventRegistrationToken m_eventRegistrationToken;
        };
    }
}
