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
// AdvancedCapture.xaml.cpp
// Implementation of the AdvancedCapture class
//

#include "pch.h"
#include "AdvancedCapture.xaml.h"

using namespace SDKSample::MediaCapture;

using namespace Windows::UI::Xaml;
using namespace Windows::UI::Xaml::Navigation;
using namespace Windows::UI::Xaml::Data;
using namespace Windows::System;
using namespace Windows::Foundation;
using namespace Platform;
using namespace Windows::UI;
using namespace Windows::UI::Core;
using namespace Windows::UI::Xaml;
using namespace Windows::UI::Xaml::Controls;
using namespace Windows::UI::Xaml::Data;
using namespace Windows::UI::Xaml::Media;
using namespace Windows::Storage;
using namespace Windows::Media::MediaProperties;
using namespace Windows::Storage::Streams;
using namespace Windows::System;
using namespace Windows::UI::Xaml::Media::Imaging;
using namespace Windows::Devices::Enumeration;

ref class ReencodeState sealed
{
public:
    ReencodeState()
    {
    }

    virtual ~ReencodeState()
    {
        if (InputStream != nullptr)
        {
            delete InputStream;
        }
        if (OutputStream != nullptr)
        {
            delete OutputStream;
        }
    }

internal:
    Windows::Storage::Streams::IRandomAccessStream ^InputStream;
    Windows::Storage::Streams::IRandomAccessStream ^OutputStream;
    Windows::Storage::StorageFile ^PhotoStorage;
    Windows::Graphics::Imaging::BitmapDecoder ^Decoder;
    Windows::Graphics::Imaging::BitmapEncoder ^Encoder;
};

AdvancedCapture::AdvancedCapture()
{
    InitializeComponent();
    ScenarioInit();
}

/// <summary>
/// Invoked when this page is about to be displayed in a Frame.
/// </summary>
/// <param name="e">Event data that describes how this page was reached.  The Parameter
/// property is typically used to configure the page.</param>
void AdvancedCapture::OnNavigatedTo(NavigationEventArgs^ e)
{
    // A pointer back to the main page.  This is needed if you want to call methods in MainPage such
    // as NotifyUser()
    rootPage = MainPage::Current;
    m_eventRegistrationToken = Windows::Media::MediaControl::SoundLevelChanged += ref new EventHandler<Object^>(this, &AdvancedCapture::SoundLevelChanged);

    m_orientationChangedEventToken = Windows::Graphics::Display::DisplayProperties::OrientationChanged += ref new Windows::Graphics::Display::DisplayPropertiesEventHandler(this, &AdvancedCapture::DisplayProperties_OrientationChanged);
}

void AdvancedCapture::OnNavigatedFrom(NavigationEventArgs^ e)
{
    Windows::Media::MediaControl::SoundLevelChanged -= m_eventRegistrationToken;
    Windows::Graphics::Display::DisplayProperties::OrientationChanged  -= m_orientationChangedEventToken;
}

void  AdvancedCapture::ScenarioInit()
{
    rootPage = MainPage::Current;
    btnStartDevice2->IsEnabled = true;
    btnStartPreview2->IsEnabled = false;
    btnStartStopRecord2->IsEnabled = false;
    m_bRecording = false;
    m_bPreviewing = false;
    m_bEffectAdded = false;
    btnStartStopRecord2->Content = "StartRecord";
    btnTakePhoto2->IsEnabled = false;
    previewElement2->Source = nullptr;
    playbackElement2->Source = nullptr;
    imageElement2->Source= nullptr;
    ShowStatusMessage("");
    chkAddRemoveEffect->IsChecked = false;
    chkAddRemoveEffect->IsEnabled = false;
    previewCanvas2->Visibility = Windows::UI::Xaml::Visibility::Collapsed;
    EnumerateWebcamsAsync();
    m_bSuspended = false;
}

void AdvancedCapture::ScenarioReset()
{
    previewCanvas2->Visibility = Windows::UI::Xaml::Visibility::Collapsed;
    ScenarioInit();
}

void AdvancedCapture::SoundLevelChanged(Object^ sender, Object^ e)
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
                    try
                    {
                        recordTask.get();
                        m_bRecording = false;
                    }
                    catch (Exception ^e)
                    {
                        ShowExceptionMessage(e);
                    }
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

                    }catch (Exception ^e)
                    {
                        ShowExceptionMessage(e);
                    }
                });
            }
        }
    })));
}

void AdvancedCapture::RecordLimitationExceeded(Windows::Media::Capture::MediaCapture ^currentCaptureObject)
{
    try
    {
        if (m_bRecording)
        {
            create_task(Dispatcher->RunAsync(Windows::UI::Core::CoreDispatcherPriority::High, ref new Windows::UI::Core::DispatchedHandler([this]()
            {
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

void AdvancedCapture::Failed(Windows::Media::Capture::MediaCapture ^currentCaptureObject, Windows::Media::Capture::MediaCaptureFailedEventArgs^ currentFailure)
{
    String ^message = "Fatal error" + currentFailure->Message;
    create_task(Dispatcher->RunAsync(Windows::UI::Core::CoreDispatcherPriority::High, 
        ref new Windows::UI::Core::DispatchedHandler([this, message]()
    {
        ShowStatusMessage(message);
    })));
}

void AdvancedCapture::btnStartDevice_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    try
    {
        EnableButton(false, "StartDevice");
        ShowStatusMessage("Starting device");
        auto mediaCapture = ref new Windows::Media::Capture::MediaCapture();
        m_mediaCaptureMgr = mediaCapture;
        auto settings = ref new Windows::Media::Capture::MediaCaptureInitializationSettings();
        auto chosenDevInfo = m_devInfoCollection->GetAt(EnumedDeviceList2->SelectedIndex);
        settings->VideoDeviceId = chosenDevInfo->Id;
        if (chosenDevInfo->EnclosureLocation != nullptr && chosenDevInfo->EnclosureLocation->Panel == Windows::Devices::Enumeration::Panel::Back)
        {
            m_bRotateVideoOnOrientationChange = true;
            m_bReversePreviewRotation = false;
        }
        else if (chosenDevInfo->EnclosureLocation != nullptr && chosenDevInfo->EnclosureLocation->Panel == Windows::Devices::Enumeration::Panel::Front)
        {
            m_bRotateVideoOnOrientationChange = true;
            m_bReversePreviewRotation = true;
        }
        else
        {
            m_bRotateVideoOnOrientationChange = false;
        }

        create_task(mediaCapture->InitializeAsync(settings)).then([this](task<void> initTask)
        {
            try
            {
                initTask.get();

                auto mediaCapture =  m_mediaCaptureMgr.Get();

                DisplayProperties_OrientationChanged(nullptr);

                EnableButton(true, "StartPreview");
                EnableButton(true, "StartStopRecord");
                EnableButton(true, "TakePhoto");
                ShowStatusMessage("Device initialized successful");
                chkAddRemoveEffect->IsEnabled = true;
                mediaCapture->RecordLimitationExceeded += ref new Windows::Media::Capture::RecordLimitationExceededEventHandler(this, &AdvancedCapture::RecordLimitationExceeded);
                mediaCapture->Failed += ref new Windows::Media::Capture::MediaCaptureFailedEventHandler(this, &AdvancedCapture::Failed);
            }
            catch (Exception ^ e)
            {
                ShowExceptionMessage(e);
            }
        });
    }
    catch (Platform::Exception^ e)
    {
        ShowExceptionMessage(e);
    }
}

void AdvancedCapture::btnStartPreview_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    m_bPreviewing = false;
    try
    {
        ShowStatusMessage("Starting preview");
        EnableButton(false, "StartPreview");

        auto mediaCapture = m_mediaCaptureMgr.Get();
        previewCanvas2->Visibility = Windows::UI::Xaml::Visibility::Visible;
        previewElement2->Source = mediaCapture;
        create_task(mediaCapture->StartPreviewAsync()).then([this](task<void> previewTask)
        {
            try
            {
                previewTask.get();
                m_bPreviewing = true;
                ShowStatusMessage("Start preview successful");
            }
            catch (Exception ^e)
            {
                ShowExceptionMessage(e);
            }
        });
    }
    catch (Platform::Exception^ e)
    {
        m_bPreviewing = false;
        previewElement2->Source = nullptr;
        EnableButton(true, "StartPreview");
        ShowExceptionMessage(e);
    }
}

void AdvancedCapture::btnTakePhoto_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    try
    {
        ShowStatusMessage("Taking photo");
        EnableButton(false, "TakePhoto");
        auto currentRotation = GetCurrentPhotoRotation();

        task<StorageFile^>(KnownFolders::PicturesLibrary->CreateFileAsync(TEMP_PHOTO_FILE_NAME, Windows::Storage::CreationCollisionOption::GenerateUniqueName)).then([this, currentRotation](task<StorageFile^> getFileTask) 
        {
            try
            {
                auto tempPhotoStorageFile = getFileTask.get();
                ShowStatusMessage("Create photo file successful");
                ImageEncodingProperties^ imageProperties = ImageEncodingProperties::CreateJpeg();

                create_task(m_mediaCaptureMgr->CapturePhotoToStorageFileAsync(imageProperties, tempPhotoStorageFile)).then([this,tempPhotoStorageFile,currentRotation](task<void> photoTask)
                {
                    try
                    {
                        photoTask.get();

                        ReencodePhotoAsync(tempPhotoStorageFile, currentRotation).then([this] (task<StorageFile^> reencodeImageTask)
                        {
                            try
                            {
                                auto photoStorageFile = reencodeImageTask.get();

                                EnableButton(true, "TakePhoto");
                                ShowStatusMessage("Photo taken");

                                task<IRandomAccessStream^>(photoStorageFile->OpenAsync(FileAccessMode::Read)).then([this](task<IRandomAccessStream^> getStreamTask)
                                {
                                    try
                                    {
                                        auto photoStream = getStreamTask.get();
                                        ShowStatusMessage("File open successful");
                                        auto bmpimg = ref new BitmapImage();

                                        bmpimg->SetSource(photoStream);
                                        imageElement2->Source = bmpimg;
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

void AdvancedCapture::btnStartStopRecord_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    try
    {
        String ^fileName;
        EnableButton(false, "StartStopRecord");

        if (!m_bRecording)
        {
            ShowStatusMessage("Starting Record");

            fileName = VIDEO_FILE_NAME;

            PrepareForVideoRecording();

            task<StorageFile^>(KnownFolders::VideosLibrary->CreateFileAsync(fileName, Windows::Storage::CreationCollisionOption::GenerateUniqueName)).then([this](task<StorageFile^> fileTask)
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
                            m_bRecording = true;
                            SwitchRecordButtonContent();
                            EnableButton(true, "StartStopRecord");
                            ShowExceptionMessage(e);
                        }
                    });
                }
                catch (Exception ^e)
                {
                    m_bRecording = false;
                    EnableButton(true, "StartStopRecord");
                    ShowExceptionMessage(e);
                }
            });
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
                                playbackElement2->AutoPlay = true;
                                playbackElement2->SetSource(stream, this->m_recordStorageFile->FileType);
                                playbackElement2->Play();
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
        m_bRecording = false;
        SwitchRecordButtonContent();
    }
}
void AdvancedCapture::lstEnumedDevices_SelectionChanged(Platform::Object^ sender, Windows::UI::Xaml::Controls::SelectionChangedEventArgs^ e)
{
     if ( m_bPreviewing )
     {
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
 
    btnStartDevice2->IsEnabled = true;
    btnStartPreview2->IsEnabled = false;
    btnStartStopRecord2->IsEnabled = false;
    m_bRecording = false;
    btnStartStopRecord2->Content = "StartRecord";
    btnTakePhoto2->IsEnabled = false;
    previewElement2->Source = nullptr;
    playbackElement2->Source = nullptr;
    imageElement2->Source= nullptr;
    chkAddRemoveEffect->IsEnabled = false;
    chkAddRemoveEffect->IsChecked = false;
    m_bEffectAdded = false;
    m_bEffectAddedToRecord = false;
    m_bEffectAddedToPhoto = false;
    ShowStatusMessage("");
}

void AdvancedCapture::EnumerateWebcamsAsync()
{
    try
    {
        ShowStatusMessage("Enumerating Webcams...");
        m_devInfoCollection = nullptr;

        EnumedDeviceList2->Items->Clear();

        task<DeviceInformationCollection^>(DeviceInformation::FindAllAsync(DeviceClass::VideoCapture)).then([this](task<DeviceInformationCollection^> findTask)
        {
            try
            {
                m_devInfoCollection = findTask.get();
                if (m_devInfoCollection == nullptr || m_devInfoCollection->Size == 0)
                {
                    ShowStatusMessage("No WebCams found.");
                }
                else
                {
                    for(unsigned int i = 0; i < m_devInfoCollection->Size; i++)
                    {
                        auto devInfo = m_devInfoCollection->GetAt(i);
                        EnumedDeviceList2->Items->Append(devInfo->Name);
                    }
                    EnumedDeviceList2->SelectedIndex = 0;
                    ShowStatusMessage("Enumerating Webcams completed successfully.");
                    btnStartDevice2->IsEnabled = true;
                }
            }
            catch (Exception ^e)
            {
                ShowExceptionMessage(e);
            }
        });
    }
    catch (Platform::Exception^ e)
    {
        ShowExceptionMessage(e);
    }
}

void AdvancedCapture::AddEffectToImageStream()
{    
    auto mediaCapture = m_mediaCaptureMgr.Get();
    Windows::Media::Capture::VideoDeviceCharacteristic charecteristic = mediaCapture->MediaCaptureSettings->VideoDeviceCharacteristic;

    if((charecteristic != Windows::Media::Capture::VideoDeviceCharacteristic::AllStreamsIdentical) &&
        (charecteristic != Windows::Media::Capture::VideoDeviceCharacteristic::PreviewPhotoStreamsIdentical) && 
        (charecteristic != Windows::Media::Capture::VideoDeviceCharacteristic::RecordPhotoStreamsIdentical))
    {
        Windows::Media::MediaProperties::IMediaEncodingProperties ^props = mediaCapture->VideoDeviceController->GetMediaStreamProperties(Windows::Media::Capture::MediaStreamType::Photo);
        if(props->Type->Equals("Image"))
        {
            //Switch to a video media type instead since we cant add an effect to a image media type
            Windows::Foundation::Collections::IVectorView<Windows::Media::MediaProperties::IMediaEncodingProperties^>^ supportedPropsList = mediaCapture->VideoDeviceController->GetAvailableMediaStreamProperties(Windows::Media::Capture::MediaStreamType::Photo);
            {
                unsigned int i = 0;
                while (i<  supportedPropsList->Size)
                {
                    Windows::Media::MediaProperties::IMediaEncodingProperties^ props = supportedPropsList->GetAt(i);

                    String^ s = props->Type;
                    if(props->Type->Equals("Video"))
                    {                                                    
                        task<void>(mediaCapture->VideoDeviceController->SetMediaStreamPropertiesAsync(Windows::Media::Capture::MediaStreamType::Photo,props)).then([this](task<void> changeTypeTask)
                        {
                            try
                            {
                                changeTypeTask.get();
                                ShowStatusMessage("Change type on photo stream successful");
                                //Now add the effect on the image pin
                                task<void>(m_mediaCaptureMgr->AddEffectAsync(Windows::Media::Capture::MediaStreamType::Photo,"GrayscaleTransform.GrayscaleEffect", nullptr)).then([this](task<void> effectTask3)
                                {
                                    try
                                    {
                                        effectTask3.get();
                                        m_bEffectAddedToPhoto = true;
                                        ShowStatusMessage("Adding effect to photo stream successful");                                                                    
                                        chkAddRemoveEffect->IsEnabled = true;

                                    }
                                    catch(Exception ^e)
                                    {
                                        ShowExceptionMessage(e);
                                        chkAddRemoveEffect->IsEnabled = true;
                                        chkAddRemoveEffect->IsChecked = false;
                                    }
                                });

                            }
                            catch(Exception ^e)
                            {
                                ShowExceptionMessage(e);
                                chkAddRemoveEffect->IsEnabled = true;
                                chkAddRemoveEffect->IsChecked = false;																	

                            }

                        });
                        break;

                    }
                    i++;
                }
            }
        }
        else
        {
            //Add the effect to the image pin if the type is already "Video"
            task<void>(mediaCapture->AddEffectAsync(Windows::Media::Capture::MediaStreamType::Photo,"GrayscaleTransform.GrayscaleEffect", nullptr)).then([this](task<void> effectTask3)
            {
                try
                {
                    effectTask3.get();
                    m_bEffectAddedToPhoto = true;
                    ShowStatusMessage("Adding effect to photo stream successful");
                    chkAddRemoveEffect->IsEnabled = true;

                }
                catch(Exception ^e)
                {
                    ShowExceptionMessage(e);
                    chkAddRemoveEffect->IsEnabled = true;
                    chkAddRemoveEffect->IsChecked = false;
                }
            });
        }
    }
}



void AdvancedCapture::chkAddRemoveEffect_Checked(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    try
    {
        chkAddRemoveEffect->IsEnabled = false;
        m_bEffectAdded = true;
        create_task(m_mediaCaptureMgr->AddEffectAsync(Windows::Media::Capture::MediaStreamType::VideoPreview,"GrayscaleTransform.GrayscaleEffect", nullptr)).then([this](task<void> effectTask)
        {
            try
            {
                effectTask.get();

                auto mediaCapture = m_mediaCaptureMgr.Get();
                Windows::Media::Capture::VideoDeviceCharacteristic charecteristic = mediaCapture->MediaCaptureSettings->VideoDeviceCharacteristic;

                ShowStatusMessage("Add effect successful to preview stream successful");                
                if((charecteristic != Windows::Media::Capture::VideoDeviceCharacteristic::AllStreamsIdentical) && 
                    (charecteristic != Windows::Media::Capture::VideoDeviceCharacteristic::PreviewRecordStreamsIdentical))
                {
                    Windows::Media::MediaProperties::IMediaEncodingProperties ^props = mediaCapture->VideoDeviceController->GetMediaStreamProperties(Windows::Media::Capture::MediaStreamType::VideoRecord);
                    Windows::Media::MediaProperties::VideoEncodingProperties ^videoEncodingProperties  = static_cast<Windows::Media::MediaProperties::VideoEncodingProperties ^>(props);
                    if(!videoEncodingProperties->Subtype->Equals("H264")) //Cant add an effect to an H264 stream
                    {
                        task<void>(mediaCapture->AddEffectAsync(Windows::Media::Capture::MediaStreamType::VideoRecord,"GrayscaleTransform.GrayscaleEffect", nullptr)).then([this](task<void> effectTask2)
                        {
                            try
                            {
                                effectTask2.get();
                                ShowStatusMessage("Add effect successful to record stream successful");
                                m_bEffectAddedToRecord = true;
                                AddEffectToImageStream();
                                chkAddRemoveEffect->IsEnabled = true;
                            } 
                            catch(Exception ^e)
                            {
                                ShowExceptionMessage(e);
                                chkAddRemoveEffect->IsEnabled = true;
                                chkAddRemoveEffect->IsChecked = false;
                            }
                        }); 						
                    }
                    else
                    {
                        AddEffectToImageStream();
                        chkAddRemoveEffect->IsEnabled = true;
                    }

                }                
                else
                {
                    AddEffectToImageStream();
                    chkAddRemoveEffect->IsEnabled = true;
                }
            }
            catch (Exception ^e)
            {
                ShowExceptionMessage(e);
                chkAddRemoveEffect->IsEnabled = true;
                chkAddRemoveEffect->IsChecked = false;
            }
        });
    }
    catch (Platform::Exception ^e)
    {
        ShowExceptionMessage(e);
        chkAddRemoveEffect->IsEnabled = true;
        chkAddRemoveEffect->IsChecked = false;
    }
}

void AdvancedCapture::chkAddRemoveEffect_Unchecked(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    try
    {
        chkAddRemoveEffect->IsEnabled = false;
        m_bEffectAdded = false;
        create_task(m_mediaCaptureMgr->ClearEffectsAsync(Windows::Media::Capture::MediaStreamType::VideoPreview)).then([this](task<void> effectTask)
        {
            try
            {
                effectTask.get();
                ShowStatusMessage("Remove effect from video preview stream successful");
                if(m_bEffectAddedToRecord)
                {
                    task<void>(m_mediaCaptureMgr->ClearEffectsAsync(Windows::Media::Capture::MediaStreamType::VideoRecord)).then([this](task<void> effectTask)
                    {
                        try
                        {
                            effectTask.get();
                            ShowStatusMessage("Remove effect from video record stream successful");
                            m_bEffectAddedToRecord = false;
                            if(m_bEffectAddedToPhoto)
                            {
                                task<void>(m_mediaCaptureMgr->ClearEffectsAsync(Windows::Media::Capture::MediaStreamType::Photo)).then([this](task<void> effectTask)
                                {
                                    try
                                    {
                                        effectTask.get();
                                        ShowStatusMessage("Remove effect from Photo stream successful");
                                        m_bEffectAddedToPhoto = false;

                                    }
                                    catch(Exception ^e)
                                    {
                                        ShowExceptionMessage(e);
                                        chkAddRemoveEffect->IsEnabled = true;
                                        chkAddRemoveEffect->IsChecked = true;				
                                    }

                                });
                            }
                            else
                            {
                            }
                            chkAddRemoveEffect->IsEnabled = true;
                        }
                        catch(Exception ^e)
                        {
                            ShowExceptionMessage(e);
                            chkAddRemoveEffect->IsEnabled = true;
                            chkAddRemoveEffect->IsChecked = true;				

                        }

                    });

                }
                else if(m_bEffectAddedToPhoto)
                {
                    task<void>(m_mediaCaptureMgr->ClearEffectsAsync(Windows::Media::Capture::MediaStreamType::Photo)).then([this](task<void> effectTask)
                    {
                        try
                        {
                            effectTask.get();
                            ShowStatusMessage("Remove effect from Photo stream successful");
                            m_bEffectAddedToPhoto = false;

                        }
                        catch(Exception ^e)
                        {
                            ShowExceptionMessage(e);
                            chkAddRemoveEffect->IsEnabled = true;
                            chkAddRemoveEffect->IsChecked = true;				
                        }

                    });
                }
                else
                {
                    chkAddRemoveEffect->IsEnabled = true;
                    chkAddRemoveEffect->IsChecked = true;	
                }
            }
            catch (Exception ^e)
            {
                ShowExceptionMessage(e);
                chkAddRemoveEffect->IsEnabled = true;
                chkAddRemoveEffect->IsChecked = true;
            }
        });
    }
    catch (Platform::Exception ^e)
    {
        ShowExceptionMessage(e);
        chkAddRemoveEffect->IsEnabled = true;
        chkAddRemoveEffect->IsChecked = true;
    }
}

void AdvancedCapture::ShowStatusMessage(Platform::String^ text)
{
    rootPage->NotifyUser(text, NotifyType::StatusMessage);
}

void AdvancedCapture::ShowExceptionMessage(Platform::Exception^ ex)
{
    rootPage->NotifyUser(ex->Message, NotifyType::ErrorMessage);
}

void AdvancedCapture::SwitchRecordButtonContent()
{
    if (m_bRecording)
    {
        btnStartStopRecord2->Content="StopRecord";
    }
    else
    {
        btnStartStopRecord2->Content="StartRecord";
    }
}

void AdvancedCapture::EnableButton(bool enabled, String^ name)
{
    if (name->Equals("StartDevice"))
    {
        btnStartDevice2->IsEnabled = enabled;
    }
    else if (name->Equals("StartPreview"))
    {
        btnStartPreview2->IsEnabled = enabled;
    }
    else if (name->Equals("StartStopRecord"))
    {
        btnStartStopRecord2->IsEnabled = enabled;
    }
    else if (name->Equals("TakePhoto"))
    {
        btnTakePhoto2->IsEnabled = enabled;
    }
}

task<Windows::Storage::StorageFile^> AdvancedCapture::ReencodePhotoAsync(
    Windows::Storage::StorageFile ^tempStorageFile,
    Windows::Storage::FileProperties::PhotoOrientation photoRotation)
{
    ReencodeState ^state = ref new ReencodeState();

    return create_task(tempStorageFile->OpenAsync(Windows::Storage::FileAccessMode::Read)).then([state](Windows::Storage::Streams::IRandomAccessStream ^stream)
    {
        state->InputStream = stream;
        return Windows::Graphics::Imaging::BitmapDecoder::CreateAsync(state->InputStream);
    }).then([state](Windows::Graphics::Imaging::BitmapDecoder ^decoder)
    {
        state->Decoder = decoder;
        return Windows::Storage::KnownFolders::PicturesLibrary->CreateFileAsync(PHOTO_FILE_NAME, Windows::Storage::CreationCollisionOption::GenerateUniqueName);
    }).then([state](Windows::Storage::StorageFile ^storageFile)
    {
        state->PhotoStorage = storageFile;
        return state->PhotoStorage->OpenAsync(Windows::Storage::FileAccessMode::ReadWrite);
    }).then([state](Windows::Storage::Streams::IRandomAccessStream ^stream)
    {
        state->OutputStream = stream;
        state->OutputStream->Size = 0;
        return Windows::Graphics::Imaging::BitmapEncoder::CreateForTranscodingAsync(state->OutputStream, state->Decoder);
    }).then([state, photoRotation](Windows::Graphics::Imaging::BitmapEncoder ^encoder)
    {
        state->Encoder = encoder;
        auto properties = ref new Windows::Graphics::Imaging::BitmapPropertySet();
        properties->Insert("System.Photo.Orientation",
            ref new Windows::Graphics::Imaging::BitmapTypedValue((unsigned short)photoRotation, Windows::Foundation::PropertyType::UInt16));
        return create_task(state->Encoder->BitmapProperties->SetPropertiesAsync(properties));
    }).then([state]()
    {
        return state->Encoder->FlushAsync();
    }).then([tempStorageFile, state](task<void> previousTask)
    {
        auto result = state->PhotoStorage;
        delete state;

        tempStorageFile->DeleteAsync(Windows::Storage::StorageDeleteOption::PermanentDelete);

        previousTask.get();

        return result;
    });
}

Windows::Storage::FileProperties::PhotoOrientation AdvancedCapture::GetCurrentPhotoRotation()
{
    bool counterclockwiseRotation = m_bReversePreviewRotation;

    if (m_bRotateVideoOnOrientationChange)
    {
        return PhotoRotationLookup(Windows::Graphics::Display::DisplayProperties::CurrentOrientation, counterclockwiseRotation);
    }
    else
    {
        return Windows::Storage::FileProperties::PhotoOrientation::Normal;
    }
}

void AdvancedCapture::PrepareForVideoRecording()
{
    Windows::Media::Capture::MediaCapture ^mediaCapture = m_mediaCaptureMgr.Get();
    if (mediaCapture == nullptr)
    {
        return;
    }

    bool counterclockwiseRotation = m_bReversePreviewRotation;

    if (m_bRotateVideoOnOrientationChange)
    {
        mediaCapture->SetRecordRotation(VideoRotationLookup(Windows::Graphics::Display::DisplayProperties::CurrentOrientation, counterclockwiseRotation));
    }
    else
    {
        mediaCapture->SetRecordRotation(Windows::Media::Capture::VideoRotation::None);
    }
}

void AdvancedCapture::DisplayProperties_OrientationChanged(Platform::Object^ sender)
{
    Windows::Media::Capture::MediaCapture ^mediaCapture = m_mediaCaptureMgr.Get();
    if (mediaCapture == nullptr)
    {
        return;
    }

    bool previewMirroring = mediaCapture->GetPreviewMirroring();
    bool counterclockwiseRotation = (previewMirroring && !m_bReversePreviewRotation) ||
        (!previewMirroring && m_bReversePreviewRotation);

    if (m_bRotateVideoOnOrientationChange)
    {
        mediaCapture->SetPreviewRotation(VideoRotationLookup(Windows::Graphics::Display::DisplayProperties::CurrentOrientation, counterclockwiseRotation));
    }
    else
    {
        mediaCapture->SetPreviewRotation(Windows::Media::Capture::VideoRotation::None);
    }
}

Windows::Storage::FileProperties::PhotoOrientation AdvancedCapture::PhotoRotationLookup(
    Windows::Graphics::Display::DisplayOrientations displayOrientation, bool counterclockwise)
{
    switch (displayOrientation)
    {
    case Windows::Graphics::Display::DisplayOrientations::Landscape:
        return Windows::Storage::FileProperties::PhotoOrientation::Normal;

    case Windows::Graphics::Display::DisplayOrientations::Portrait:
        return (counterclockwise) ? Windows::Storage::FileProperties::PhotoOrientation::Rotate270:
            Windows::Storage::FileProperties::PhotoOrientation::Rotate90;

    case Windows::Graphics::Display::DisplayOrientations::LandscapeFlipped:
        return Windows::Storage::FileProperties::PhotoOrientation::Rotate180;

    case Windows::Graphics::Display::DisplayOrientations::PortraitFlipped:
        return (counterclockwise) ? Windows::Storage::FileProperties::PhotoOrientation::Rotate90 :
            Windows::Storage::FileProperties::PhotoOrientation::Rotate270;

    default:
        return Windows::Storage::FileProperties::PhotoOrientation::Unspecified;
    }
}

Windows::Media::Capture::VideoRotation AdvancedCapture::VideoRotationLookup(
    Windows::Graphics::Display::DisplayOrientations displayOrientation, bool counterclockwise)
{
    switch (displayOrientation)
    {
    case Windows::Graphics::Display::DisplayOrientations::Landscape:
        return Windows::Media::Capture::VideoRotation::None;

    case Windows::Graphics::Display::DisplayOrientations::Portrait:
        return (counterclockwise) ? Windows::Media::Capture::VideoRotation::Clockwise270Degrees :
            Windows::Media::Capture::VideoRotation::Clockwise90Degrees;

    case Windows::Graphics::Display::DisplayOrientations::LandscapeFlipped:
        return Windows::Media::Capture::VideoRotation::Clockwise180Degrees;

    case Windows::Graphics::Display::DisplayOrientations::PortraitFlipped:
        return (counterclockwise) ? Windows::Media::Capture::VideoRotation::Clockwise90Degrees:
            Windows::Media::Capture::VideoRotation::Clockwise270Degrees ;

    default:
        return Windows::Media::Capture::VideoRotation::None;
    }
}

