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
// BasicCapture.xaml.cpp
// Implementation of the BasicCapture class
//

#include "pch.h"
#include "BasicCapture.xaml.h"
#include "ppl.h"

using namespace Windows::System;
using namespace Windows::Foundation;
using namespace Platform;
using namespace Windows::UI;
using namespace Windows::UI::Core;
using namespace Windows::UI::Xaml;
using namespace Windows::UI::Xaml::Controls;
using namespace Windows::UI::Xaml::Navigation;
using namespace Windows::UI::Xaml::Data;
using namespace Windows::UI::Xaml::Media;
using namespace Windows::Storage;
using namespace Windows::Media::MediaProperties;
using namespace Windows::Storage::Streams;
using namespace Windows::System;
using namespace Windows::UI::Xaml::Media::Imaging;

using namespace SDKSample::MediaCapture;
using namespace concurrency;


BasicCapture::BasicCapture()
{
    InitializeComponent();
    ScenarioInit();
}

/// <summary>
/// Invoked when this page is about to be displayed in a Frame.
/// </summary>
/// <param name="e">Event data that describes how this page was reached.  The Parameter
/// property is typically used to configure the page.</param>
void BasicCapture::OnNavigatedTo(NavigationEventArgs^ e)
{
    // A pointer back to the main page.  This is needed if you want to call methods in MainPage such
    // as NotifyUser()
    rootPage = MainPage::Current;
    m_eventRegistrationToken = Windows::Media::MediaControl::SoundLevelChanged += ref new EventHandler<Object^>(this, &BasicCapture::SoundLevelChanged);
}

void BasicCapture::OnNavigatedFrom(NavigationEventArgs^ e)
{
    // A pointer back to the main page.  This is needed if you want to call methods in MainPage such
    // as NotifyUser()

    Windows::Media::MediaControl::SoundLevelChanged -= m_eventRegistrationToken;
    m_currentScenarioLoaded = false;
}


void  BasicCapture::ScenarioInit()
{
    try
    {
        btnStartDevice1->IsEnabled = true;
        btnStartPreview1->IsEnabled = false;
        btnStartStopRecord1->IsEnabled = false;
        m_bRecording = false;
        m_bPreviewing = false;
        btnStartStopRecord1->Content = "StartRecord";
        btnTakePhoto1->IsEnabled = false;
        previewElement1->Source = nullptr;
        playbackElement1->Source = nullptr;
        imageElement1->Source= nullptr;
        sldBrightness->IsEnabled = false;
        sldContrast->IsEnabled = false;
        m_bSuspended = false;
        previewCanvas1->Visibility = Windows::UI::Xaml::Visibility::Collapsed;

    }
    catch (Exception ^e)
    {
        ShowExceptionMessage(e);
    }

}

void BasicCapture::ScenarioReset()
{
    previewCanvas1->Visibility = Windows::UI::Xaml::Visibility::Collapsed;
    ScenarioInit();
}

void BasicCapture::SoundLevelChanged(Object^ sender, Object^ e)
{
    create_task(Dispatcher->RunAsync(Windows::UI::Core::CoreDispatcherPriority::High, ref new Windows::UI::Core::DispatchedHandler([this]()
    {    
        if(Windows::Media::MediaControl::SoundLevel != Windows::Media::SoundLevel::Muted)
        {
            ScenarioReset();
        }
        else
        {
            if (m_bRecording)
            {
                ShowStatusMessage("Stopping Record on invisibility");

                create_task(m_mediaCaptureMgr->StopRecordAsync()).then([this](task<void> recordTask)
                {
                    m_bRecording = false;
                });
            }
            if (m_bPreviewing)
            {
                ShowStatusMessage("Stopping Preview on invisibility");

                create_task(m_mediaCaptureMgr->StopPreviewAsync()).then([this](task<void> previewTask)
                {
                    try
                    {
                        previewTask.get();
                        m_bPreviewing = false;
                    }
                    catch (Exception ^e)
                    {
                        ShowExceptionMessage(e);
                    }
                });
            }
        }
    })));
}

void BasicCapture::RecordLimitationExceeded(Windows::Media::Capture::MediaCapture ^currentCaptureObject)
{
    try
    {
        if (m_bRecording)
        {
            create_task(Dispatcher->RunAsync(Windows::UI::Core::CoreDispatcherPriority::High, ref new Windows::UI::Core::DispatchedHandler([this](){
                try
                {
                    ShowStatusMessage("Stopping Record on exceeding max record duration");
                    EnableButton(false, "StartStopRecord");
                    create_task(m_mediaCaptureMgr->StopRecordAsync()).then([this](task<void> recordTask)
                    {
                        try
                        {
                            recordTask.get();
                            m_bRecording = false;
                            SwitchRecordButtonContent();
                            EnableButton(true, "StartStopRecord");
                            ShowStatusMessage("Stopped record on exceeding max record duration:" + m_recordStorageFile->Path);
                        }
                        catch (Exception ^e)
                        {
                            ShowExceptionMessage(e);
                            m_bRecording = false;
                            SwitchRecordButtonContent();
                            EnableButton(true, "StartStopRecord");
                        }
                    });

                }
                catch (Exception ^e)
                {
                    m_bRecording = false;
                    SwitchRecordButtonContent();
                    EnableButton(true, "StartStopRecord");
                    ShowExceptionMessage(e);
                }

            })));
        }
    }
    catch (Exception ^e)
    {
        m_bRecording = false;
        SwitchRecordButtonContent();
        EnableButton(true, "StartStopRecord");
        ShowExceptionMessage(e);
    }
}

void BasicCapture::Failed(Windows::Media::Capture::MediaCapture ^currentCaptureObject, Windows::Media::Capture::MediaCaptureFailedEventArgs^ currentFailure)
{
    String ^message = "Fatal error: " + currentFailure->Message;
    create_task(Dispatcher->RunAsync(Windows::UI::Core::CoreDispatcherPriority::High,
        ref new Windows::UI::Core::DispatchedHandler([this, message]()
    {
        ShowStatusMessage(message);
    })));
}

void BasicCapture::btnStartDevice_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    try
    {
        EnableButton(false, "StartDevice");
        ShowStatusMessage("Starting device");
        auto mediaCapture = ref new Windows::Media::Capture::MediaCapture();
        m_mediaCaptureMgr = mediaCapture;
        create_task(mediaCapture->InitializeAsync()).then([this](task<void> initTask)
        {
            try
            {
                initTask.get();

                auto mediaCapture = m_mediaCaptureMgr.Get();
                EnableButton(true, "StartPreview");
                EnableButton(true, "StartStopRecord");
                EnableButton(true, "TakePhoto");
                ShowStatusMessage("Device initialized successful");
                mediaCapture->RecordLimitationExceeded += ref new Windows::Media::Capture::RecordLimitationExceededEventHandler(this, &BasicCapture::RecordLimitationExceeded);
                mediaCapture->Failed += ref new Windows::Media::Capture::MediaCaptureFailedEventHandler(this, &BasicCapture::Failed);
            }
            catch (Exception ^ e)
            {
                ShowExceptionMessage(e);
            }
        }
        );
    }
    catch (Platform::Exception^ e)
    {
        ShowExceptionMessage(e);
    }
}

void BasicCapture::btnStartPreview_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    m_bPreviewing = false;
    try
    {
        ShowStatusMessage("Starting preview");
        EnableButton(false, "StartPreview");
        auto mediaCapture = m_mediaCaptureMgr.Get();

        previewCanvas1->Visibility = Windows::UI::Xaml::Visibility::Visible;
        previewElement1->Source = mediaCapture;
        create_task(mediaCapture->StartPreviewAsync()).then([this](task<void> previewTask)
        {
            try
            {
                previewTask.get();
                auto mediaCapture = m_mediaCaptureMgr.Get();
                m_bPreviewing = true;
                ShowStatusMessage("Start preview successful");
                if(mediaCapture->VideoDeviceController->Brightness)
                {
                    SetupVideoDeviceControl(mediaCapture->VideoDeviceController->Brightness, sldBrightness);
                }
                if(mediaCapture->VideoDeviceController->Contrast)
                {                
                    SetupVideoDeviceControl(mediaCapture->VideoDeviceController->Contrast, sldContrast);
                }

            }catch (Exception ^e)
            {
                ShowExceptionMessage(e);
            }
        });
    }
    catch (Platform::Exception^ e)
    {
        m_bPreviewing = false;
        previewElement1->Source = nullptr;
        EnableButton(true, "StartPreview");
        ShowExceptionMessage(e);
    }
}

void BasicCapture::btnTakePhoto_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    try
    {
        ShowStatusMessage("Taking photo");
        EnableButton(false, "TakePhoto");

        task<StorageFile^>(KnownFolders::PicturesLibrary->CreateFileAsync(PHOTO_FILE_NAME, Windows::Storage::CreationCollisionOption::GenerateUniqueName)).then([this](task<StorageFile^> getFileTask) 
        {
            try
            {
                this->m_photoStorageFile = getFileTask.get();
                ShowStatusMessage("Create photo file successful");
                ImageEncodingProperties^ imageProperties = ImageEncodingProperties::CreateJpeg();

                create_task(m_mediaCaptureMgr->CapturePhotoToStorageFileAsync(imageProperties, this->m_photoStorageFile)).then([this](task<void> photoTask)
                {
                    try
                    {
                        photoTask.get();
                        EnableButton(true, "TakePhoto");
                        ShowStatusMessage("Photo taken");

                        task<IRandomAccessStream^>(this->m_photoStorageFile->OpenAsync(FileAccessMode::Read)).then([this](task<IRandomAccessStream^> getStreamTask)
                        {
                            try
                            {
                                auto photoStream = getStreamTask.get();
                                ShowStatusMessage("File open successful");
                                auto bmpimg = ref new BitmapImage();

                                bmpimg->SetSource(photoStream);
                                imageElement1->Source = bmpimg;
                            }
                            catch (Exception^ e)
                            {
                                ShowExceptionMessage(e);
                                EnableButton(true, "TakePhoto");
                            }
                        });
                    }
                    catch (Platform::Exception ^ e)
                    {
                        ShowExceptionMessage(e);
                        EnableButton(true, "TakePhoto");
                    }
                });
            }
            catch (Exception^ e)
            {
                ShowExceptionMessage(e);
                EnableButton(true, "TakePhoto");
            }
        });
    }
    catch (Platform::Exception^ e)
    {
        ShowExceptionMessage(e);
        EnableButton(true, "TakePhoto");
    }
}

void BasicCapture::btnStartStopRecord_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    try
    {
        String ^fileName;
        EnableButton(false, "StartStopRecord");

        if (!m_bRecording)
        {
            ShowStatusMessage("Starting Record");

            fileName = VIDEO_FILE_NAME;

            task<StorageFile^>(KnownFolders::VideosLibrary->CreateFileAsync(fileName,Windows::Storage::CreationCollisionOption::GenerateUniqueName )).then([this](task<StorageFile^> fileTask)
            {
                try
                {
                    this->m_recordStorageFile = fileTask.get();
                    ShowStatusMessage("Create record file successful");

                    MediaEncodingProfile^ recordProfile= nullptr;
                    recordProfile = MediaEncodingProfile::CreateMp4(Windows::Media::MediaProperties::VideoEncodingQuality::Auto);

                    create_task(m_mediaCaptureMgr->StartRecordToStorageFileAsync(recordProfile, this->m_recordStorageFile)).then([this](task<void> recordTask)
                    {
                        try
                        {
                            recordTask.get();
                            m_bRecording = true;
                            SwitchRecordButtonContent();
                            EnableButton(true, "StartStopRecord");

                            ShowStatusMessage("Start Record successful");
                        }
                        catch (Exception ^e)
                        {
                            ShowExceptionMessage(e);
                        }
                    });
                }
                catch (Exception ^e)
                {
                    m_bRecording = false;
                    SwitchRecordButtonContent();
                    EnableButton(true, "StartStopRecord");
                    ShowExceptionMessage(e);
                }
            }
            );
        }
        else
        {
            ShowStatusMessage("Stopping Record");

            create_task(m_mediaCaptureMgr->StopRecordAsync()).then([this](task<void> recordTask)
            {
                try
                {
                    recordTask.get();
                    m_bRecording = false;
                    EnableButton(true, "StartStopRecord");
                    SwitchRecordButtonContent();

                    ShowStatusMessage("Stop record successful");
                    if (!m_bSuspended)
                    {
                        task<IRandomAccessStream^>(this->m_recordStorageFile->OpenAsync(FileAccessMode::Read)).then([this](task<IRandomAccessStream^> streamTask)
                        {
                            try
                            {
                                auto stream = streamTask.get();
                                ShowStatusMessage("Record file opened");
                                ShowStatusMessage(this->m_recordStorageFile->Path);
                                playbackElement1->AutoPlay = true;
                                playbackElement1->SetSource(stream, this->m_recordStorageFile->FileType);
                                playbackElement1->Play();
                            }
                            catch (Exception ^e)
                            {
                                ShowExceptionMessage(e);
                                m_bRecording = false;
                                EnableButton(true, "StartStopRecord");
                                SwitchRecordButtonContent();
                            }
                        });
                    }
                }
                catch (Exception ^e)
                {
                    m_bRecording = false;
                    EnableButton(true, "StartStopRecord");
                    SwitchRecordButtonContent();
                    ShowExceptionMessage(e);
                }
            });
        }
    }
    catch (Platform::Exception^ e)
    {
        EnableButton(true, "StartStopRecord");
        ShowExceptionMessage(e);
        SwitchRecordButtonContent();
        m_bRecording = false;
    }
}

void BasicCapture::SetupVideoDeviceControl(Windows::Media::Devices::MediaDeviceControl^ videoDeviceControl, Slider^ slider)
{
    try
    {		
        if ((videoDeviceControl->Capabilities)->Supported)
        {
            slider->IsEnabled = true;
            slider->Maximum = videoDeviceControl->Capabilities->Max;
            slider->Minimum = videoDeviceControl->Capabilities->Min;
            slider->StepFrequency = videoDeviceControl->Capabilities->Step;
            double controlValue = 0;
            if (videoDeviceControl->TryGetValue(&controlValue))
            {
                slider->Value = controlValue;
            }
        }
        else
        {
            slider->IsEnabled = false;
        }
    }
    catch (Platform::Exception^ e)
    {
        ShowExceptionMessage(e);
    }
}

// VideoDeviceControllers
void BasicCapture::sldBrightness_ValueChanged(Platform::Object^ sender, Windows::UI::Xaml::Controls::Primitives::RangeBaseValueChangedEventArgs^ e)
{
    bool succeeded = m_mediaCaptureMgr->VideoDeviceController->Brightness->TrySetValue(sldBrightness->Value);
    if (!succeeded)
    {
        ShowStatusMessage("Set Brightness failed");
    }
}

void BasicCapture::sldContrast_ValueChanged(Platform::Object^ sender, Windows::UI::Xaml::Controls::Primitives::RangeBaseValueChangedEventArgs ^e)
{
    bool succeeded = m_mediaCaptureMgr->VideoDeviceController->Contrast->TrySetValue(sldContrast->Value);
    if (!succeeded)
    {
        ShowStatusMessage("Set Contrast failed");
    }
}

void BasicCapture::ShowStatusMessage(Platform::String^ text)
{
    rootPage->NotifyUser(text, NotifyType::StatusMessage);
}

void BasicCapture::ShowExceptionMessage(Platform::Exception^ ex)
{
    rootPage->NotifyUser(ex->Message, NotifyType::ErrorMessage);
}

void BasicCapture::SwitchRecordButtonContent()
{
    if (m_bRecording)
    {
        btnStartStopRecord1->Content="StopRecord";
    }
    else
    {
        btnStartStopRecord1->Content="StartRecord";
    }
}
void BasicCapture::EnableButton(bool enabled, String^ name)
{
    if (name->Equals("StartDevice"))
    {
        btnStartDevice1->IsEnabled = enabled;
    }
    else if (name->Equals("StartPreview"))
    {
        btnStartPreview1->IsEnabled = enabled;
    }
    else if (name->Equals("StartStopRecord"))
    {
        btnStartStopRecord1->IsEnabled = enabled;
    }
    else if (name->Equals("TakePhoto"))
    {
        btnTakePhoto1->IsEnabled = enabled;
    }
}

