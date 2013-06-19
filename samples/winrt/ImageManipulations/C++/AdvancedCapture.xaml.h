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
// AdvancedCapture.xaml.h
// Declaration of the AdvancedCapture class
//

#pragma once

#include "pch.h"
#include "AdvancedCapture.g.h"
#include "MainPage.xaml.h"
#include <ppl.h>

#define VIDEO_FILE_NAME "video.mp4"
#define PHOTO_FILE_NAME "photo.jpg"
#define TEMP_PHOTO_FILE_NAME "photoTmp.jpg"

using namespace concurrency;
using namespace Windows::Devices::Enumeration;

namespace SDKSample
{
    namespace MediaCapture
    {
        /// <summary>
        /// An empty page that can be used on its own or navigated to within a Frame.
        /// </summary>
        [Windows::Foundation::Metadata::WebHostHidden]
        public ref class AdvancedCapture sealed
        {
        public:
            AdvancedCapture();

        protected:
            virtual void OnNavigatedTo(Windows::UI::Xaml::Navigation::NavigationEventArgs^ e) override;
            virtual void OnNavigatedFrom(Windows::UI::Xaml::Navigation::NavigationEventArgs^ e) override;

        private:
            MainPage^ rootPage;
            void ScenarioInit();
            void ScenarioReset();

            void Failed(Windows::Media::Capture::MediaCapture ^ mediaCapture, Windows::Media::Capture::MediaCaptureFailedEventArgs ^ args);

            void btnStartDevice_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);

            void btnStartPreview_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);

            void lstEnumedDevices_SelectionChanged(Platform::Object^ sender, Windows::UI::Xaml::Controls::SelectionChangedEventArgs^ e);
            void EnumerateWebcamsAsync();

            void AddEffectToImageStream();

            void ShowStatusMessage(Platform::String^ text);
            void ShowExceptionMessage(Platform::Exception^ ex);

            void EnableButton(bool enabled, Platform::String ^name);

            task<Windows::Storage::StorageFile^> ReencodePhotoAsync(
                Windows::Storage::StorageFile ^tempStorageFile,
                Windows::Storage::FileProperties::PhotoOrientation photoRotation);
            Windows::Storage::FileProperties::PhotoOrientation GetCurrentPhotoRotation();
            void PrepareForVideoRecording();
            void DisplayProperties_OrientationChanged(Platform::Object^ sender);
            Windows::Storage::FileProperties::PhotoOrientation PhotoRotationLookup(
                Windows::Graphics::Display::DisplayOrientations displayOrientation, bool counterclockwise);
            Windows::Media::Capture::VideoRotation VideoRotationLookup(
                Windows::Graphics::Display::DisplayOrientations displayOrientation, bool counterclockwise);

            Platform::Agile<Windows::Media::Capture::MediaCapture> m_mediaCaptureMgr;
            Windows::Storage::StorageFile^ m_recordStorageFile;
            bool m_bRecording;
            bool m_bEffectAdded;
            bool m_bEffectAddedToRecord;
            bool m_bEffectAddedToPhoto;
            bool m_bSuspended;
            bool m_bPreviewing;
            DeviceInformationCollection^ m_devInfoCollection;
            Windows::Foundation::EventRegistrationToken m_eventRegistrationToken;
            bool m_bRotateVideoOnOrientationChange;
            bool m_bReversePreviewRotation;
            Windows::Foundation::EventRegistrationToken m_orientationChangedEventToken;
            void Button_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
        };
    }
}
