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
using namespace Windows::Foundation::Collections;
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
    m_bRecording = false;
    m_bPreviewing = false;
    m_bEffectAdded = false;
    previewElement2->Source = nullptr;
    ShowStatusMessage("");
    EffectTypeCombo->IsEnabled = false;
    previewCanvas2->Visibility = Windows::UI::Xaml::Visibility::Collapsed;
    EnumerateWebcamsAsync();
    m_bSuspended = false;
}

void AdvancedCapture::ScenarioReset()
{
    previewCanvas2->Visibility = Windows::UI::Xaml::Visibility::Collapsed;
    ScenarioInit();
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
                EffectTypeCombo->IsEnabled = true;
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
    m_bRecording = false;
    previewElement2->Source = nullptr;
    EffectTypeCombo->IsEnabled = false;
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
                while (i < supportedPropsList->Size)
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
                                task<void>(m_mediaCaptureMgr->AddEffectAsync(Windows::Media::Capture::MediaStreamType::Photo,"OcvTransform.OcvImageManipulations", nullptr)).then([this](task<void> effectTask3)
                                {
                                    try
                                    {
                                        effectTask3.get();
                                        m_bEffectAddedToPhoto = true;
                                        ShowStatusMessage("Adding effect to photo stream successful");
                                        EffectTypeCombo->IsEnabled = true;

                                    }
                                    catch(Exception ^e)
                                    {
                                        ShowExceptionMessage(e);
                                        EffectTypeCombo->IsEnabled = true;
                                    }
                                });

                            }
                            catch(Exception ^e)
                            {
                                ShowExceptionMessage(e);
                                EffectTypeCombo->IsEnabled = true;
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
            task<void>(mediaCapture->AddEffectAsync(Windows::Media::Capture::MediaStreamType::Photo,"OcvTransform.OcvImageManipulations", nullptr)).then([this](task<void> effectTask3)
            {
                try
                {
                    effectTask3.get();
                    m_bEffectAddedToPhoto = true;
                    ShowStatusMessage("Adding effect to photo stream successful");
                    EffectTypeCombo->IsEnabled = true;

                }
                catch(Exception ^e)
                {
                    ShowExceptionMessage(e);
                    EffectTypeCombo->IsEnabled = true;
                }
            });
        }
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

void SDKSample::MediaCapture::AdvancedCapture::Button_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    try
    {
        create_task(m_mediaCaptureMgr->ClearEffectsAsync(Windows::Media::Capture::MediaStreamType::VideoPreview)).then([this](task<void> cleanTask)
        {
            m_bEffectAdded = true;
            int index = EffectTypeCombo->SelectedIndex;
            PropertySet^ props = ref new PropertySet();
            props->Insert(L"{698649BE-8EAE-4551-A4CB-3EC98FBD3D86}", index);
            create_task(m_mediaCaptureMgr->AddEffectAsync(Windows::Media::Capture::MediaStreamType::VideoPreview,"OcvTransform.OcvImageManipulations", props)).then([this](task<void> effectTask)
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
                            task<void>(mediaCapture->AddEffectAsync(Windows::Media::Capture::MediaStreamType::VideoRecord,"OcvTransform.OcvImageManipulations", nullptr)).then([this](task<void> effectTask2)
                            {
                                try
                                {
                                    effectTask2.get();
                                    ShowStatusMessage("Add effect successful to record stream successful");
                                    m_bEffectAddedToRecord = true;
                                    AddEffectToImageStream();
                                    EffectTypeCombo->IsEnabled = true;
                                }
                                catch(Exception ^e)
                                {
                                    ShowExceptionMessage(e);
                                    EffectTypeCombo->IsEnabled = true;
                                }
                            });
                        }
                        else
                        {
                            AddEffectToImageStream();
                            EffectTypeCombo->IsEnabled = true;
                        }

                    }
                    else
                    {
                        AddEffectToImageStream();
                        EffectTypeCombo->IsEnabled = true;
                    }
                }
                catch (Exception ^e)
                {
                    ShowExceptionMessage(e);
                    EffectTypeCombo->IsEnabled = true;
                }
            });
        });
    }
    catch (Platform::Exception ^e)
    {
        ShowExceptionMessage(e);
        EffectTypeCombo->IsEnabled = true;
    }
}
