#ifdef HAVE_WINRT
#define ICustomStreamSink StreamSink
#include "ppltasks_winrt.h"
#else
EXTERN_C const IID IID_ICustomStreamSink;

class DECLSPEC_UUID("4F8A1939-2FD3-46DB-AE70-DB7E0DD79B73") DECLSPEC_NOVTABLE ICustomStreamSink : public IUnknown
{
public:
    virtual HRESULT Initialize() = 0;
    virtual HRESULT Shutdown() = 0;
    virtual HRESULT Start(MFTIME start) = 0;
    virtual HRESULT Pause() = 0;
    virtual HRESULT Restart() = 0;
    virtual HRESULT Stop() = 0;
};
#endif

#define MF_PROP_SAMPLEGRABBERCALLBACK L"samplegrabbercallback"
#define MF_PROP_VIDTYPE L"vidtype"
#define MF_PROP_VIDENCPROPS L"videncprops"

#include <initguid.h>

// MF_MEDIASINK_SAMPLEGRABBERCALLBACK: {26957AA7-AFF4-464c-BB8B-07BA65CE11DF}
// Type: IUnknown*
DEFINE_GUID(MF_MEDIASINK_SAMPLEGRABBERCALLBACK,
            0x26957aa7, 0xaff4, 0x464c, 0xbb, 0x8b, 0x7, 0xba, 0x65, 0xce, 0x11, 0xdf);

// {4BD133CC-EB9B-496E-8865-0813BFBC6FAA}
DEFINE_GUID(MF_STREAMSINK_ID, 0x4bd133cc, 0xeb9b, 0x496e, 0x88, 0x65, 0x8, 0x13, 0xbf, 0xbc, 0x6f, 0xaa);

// {C9E22A8C-6A50-4D78-9183-0834A02A3780}
DEFINE_GUID(MF_STREAMSINK_MEDIASINKINTERFACE,
    0xc9e22a8c, 0x6a50, 0x4d78, 0x91, 0x83, 0x8, 0x34, 0xa0, 0x2a, 0x37, 0x80);

// {DABD13AB-26B7-47C2-97C1-4B04C187B838}
DEFINE_GUID(MF_MEDIASINK_PREFERREDTYPE,
    0xdabd13ab, 0x26b7, 0x47c2, 0x97, 0xc1, 0x4b, 0x4, 0xc1, 0x87, 0xb8, 0x38);

#include <utility>
#ifdef _UNICODE
#define MAKE_MAP(e) std::map<e, std::wstring>
#define MAKE_ENUM(e) std::pair<e, std::wstring>
#define MAKE_ENUM_PAIR(e, str) std::pair<e, std::wstring>(str, L#str)
#else
#define MAKE_MAP(e) std::map<e, std::string>
#define MAKE_ENUM(e) std::pair<e, std::string>
#define MAKE_ENUM_PAIR(e, str) std::pair<e, std::string>(str, #str)
#endif

MAKE_ENUM(MediaEventType) MediaEventTypePairs[] = {
    MAKE_ENUM_PAIR(MediaEventType, MEUnknown),
    MAKE_ENUM_PAIR(MediaEventType, MEError),
    MAKE_ENUM_PAIR(MediaEventType, MEExtendedType),
    MAKE_ENUM_PAIR(MediaEventType, MENonFatalError),
    MAKE_ENUM_PAIR(MediaEventType, MEGenericV1Anchor),
    MAKE_ENUM_PAIR(MediaEventType, MESessionUnknown),
    MAKE_ENUM_PAIR(MediaEventType, MESessionTopologySet),
    MAKE_ENUM_PAIR(MediaEventType, MESessionTopologiesCleared),
    MAKE_ENUM_PAIR(MediaEventType, MESessionStarted),
    MAKE_ENUM_PAIR(MediaEventType, MESessionPaused),
    MAKE_ENUM_PAIR(MediaEventType, MESessionStopped),
    MAKE_ENUM_PAIR(MediaEventType, MESessionClosed),
    MAKE_ENUM_PAIR(MediaEventType, MESessionEnded),
    MAKE_ENUM_PAIR(MediaEventType, MESessionRateChanged),
    MAKE_ENUM_PAIR(MediaEventType, MESessionScrubSampleComplete),
    MAKE_ENUM_PAIR(MediaEventType, MESessionCapabilitiesChanged),
    MAKE_ENUM_PAIR(MediaEventType, MESessionTopologyStatus),
    MAKE_ENUM_PAIR(MediaEventType, MESessionNotifyPresentationTime),
    MAKE_ENUM_PAIR(MediaEventType, MENewPresentation),
    MAKE_ENUM_PAIR(MediaEventType, MELicenseAcquisitionStart),
    MAKE_ENUM_PAIR(MediaEventType, MELicenseAcquisitionCompleted),
    MAKE_ENUM_PAIR(MediaEventType, MEIndividualizationStart),
    MAKE_ENUM_PAIR(MediaEventType, MEIndividualizationCompleted),
    MAKE_ENUM_PAIR(MediaEventType, MEEnablerProgress),
    MAKE_ENUM_PAIR(MediaEventType, MEEnablerCompleted),
    MAKE_ENUM_PAIR(MediaEventType, MEPolicyError),
    MAKE_ENUM_PAIR(MediaEventType, MEPolicyReport),
    MAKE_ENUM_PAIR(MediaEventType, MEBufferingStarted),
    MAKE_ENUM_PAIR(MediaEventType, MEBufferingStopped),
    MAKE_ENUM_PAIR(MediaEventType, MEConnectStart),
    MAKE_ENUM_PAIR(MediaEventType, MEConnectEnd),
    MAKE_ENUM_PAIR(MediaEventType, MEReconnectStart),
    MAKE_ENUM_PAIR(MediaEventType, MEReconnectEnd),
    MAKE_ENUM_PAIR(MediaEventType, MERendererEvent),
    MAKE_ENUM_PAIR(MediaEventType, MESessionStreamSinkFormatChanged),
    MAKE_ENUM_PAIR(MediaEventType, MESessionV1Anchor),
    MAKE_ENUM_PAIR(MediaEventType, MESourceUnknown),
    MAKE_ENUM_PAIR(MediaEventType, MESourceStarted),
    MAKE_ENUM_PAIR(MediaEventType, MEStreamStarted),
    MAKE_ENUM_PAIR(MediaEventType, MESourceSeeked),
    MAKE_ENUM_PAIR(MediaEventType, MEStreamSeeked),
    MAKE_ENUM_PAIR(MediaEventType, MENewStream),
    MAKE_ENUM_PAIR(MediaEventType, MEUpdatedStream),
    MAKE_ENUM_PAIR(MediaEventType, MESourceStopped),
    MAKE_ENUM_PAIR(MediaEventType, MEStreamStopped),
    MAKE_ENUM_PAIR(MediaEventType, MESourcePaused),
    MAKE_ENUM_PAIR(MediaEventType, MEStreamPaused),
    MAKE_ENUM_PAIR(MediaEventType, MEEndOfPresentation),
    MAKE_ENUM_PAIR(MediaEventType, MEEndOfStream),
    MAKE_ENUM_PAIR(MediaEventType, MEMediaSample),
    MAKE_ENUM_PAIR(MediaEventType, MEStreamTick),
    MAKE_ENUM_PAIR(MediaEventType, MEStreamThinMode),
    MAKE_ENUM_PAIR(MediaEventType, MEStreamFormatChanged),
    MAKE_ENUM_PAIR(MediaEventType, MESourceRateChanged),
    MAKE_ENUM_PAIR(MediaEventType, MEEndOfPresentationSegment),
    MAKE_ENUM_PAIR(MediaEventType, MESourceCharacteristicsChanged),
    MAKE_ENUM_PAIR(MediaEventType, MESourceRateChangeRequested),
    MAKE_ENUM_PAIR(MediaEventType, MESourceMetadataChanged),
    MAKE_ENUM_PAIR(MediaEventType, MESequencerSourceTopologyUpdated),
    MAKE_ENUM_PAIR(MediaEventType, MESourceV1Anchor),
    MAKE_ENUM_PAIR(MediaEventType, MESinkUnknown),
    MAKE_ENUM_PAIR(MediaEventType, MEStreamSinkStarted),
    MAKE_ENUM_PAIR(MediaEventType, MEStreamSinkStopped),
    MAKE_ENUM_PAIR(MediaEventType, MEStreamSinkPaused),
    MAKE_ENUM_PAIR(MediaEventType, MEStreamSinkRateChanged),
    MAKE_ENUM_PAIR(MediaEventType, MEStreamSinkRequestSample),
    MAKE_ENUM_PAIR(MediaEventType, MEStreamSinkMarker),
    MAKE_ENUM_PAIR(MediaEventType, MEStreamSinkPrerolled),
    MAKE_ENUM_PAIR(MediaEventType, MEStreamSinkScrubSampleComplete),
    MAKE_ENUM_PAIR(MediaEventType, MEStreamSinkFormatChanged),
    MAKE_ENUM_PAIR(MediaEventType, MEStreamSinkDeviceChanged),
    MAKE_ENUM_PAIR(MediaEventType, MEQualityNotify),
    MAKE_ENUM_PAIR(MediaEventType, MESinkInvalidated),
    MAKE_ENUM_PAIR(MediaEventType, MEAudioSessionNameChanged),
    MAKE_ENUM_PAIR(MediaEventType, MEAudioSessionVolumeChanged),
    MAKE_ENUM_PAIR(MediaEventType, MEAudioSessionDeviceRemoved),
    MAKE_ENUM_PAIR(MediaEventType, MEAudioSessionServerShutdown),
    MAKE_ENUM_PAIR(MediaEventType, MEAudioSessionGroupingParamChanged),
    MAKE_ENUM_PAIR(MediaEventType, MEAudioSessionIconChanged),
    MAKE_ENUM_PAIR(MediaEventType, MEAudioSessionFormatChanged),
    MAKE_ENUM_PAIR(MediaEventType, MEAudioSessionDisconnected),
    MAKE_ENUM_PAIR(MediaEventType, MEAudioSessionExclusiveModeOverride),
    MAKE_ENUM_PAIR(MediaEventType, MESinkV1Anchor),
    MAKE_ENUM_PAIR(MediaEventType, MECaptureAudioSessionVolumeChanged),
    MAKE_ENUM_PAIR(MediaEventType, MECaptureAudioSessionDeviceRemoved),
    MAKE_ENUM_PAIR(MediaEventType, MECaptureAudioSessionFormatChanged),
    MAKE_ENUM_PAIR(MediaEventType, MECaptureAudioSessionDisconnected),
    MAKE_ENUM_PAIR(MediaEventType, MECaptureAudioSessionExclusiveModeOverride),
    MAKE_ENUM_PAIR(MediaEventType, MECaptureAudioSessionServerShutdown),
    MAKE_ENUM_PAIR(MediaEventType, MESinkV2Anchor),
    MAKE_ENUM_PAIR(MediaEventType, METrustUnknown),
    MAKE_ENUM_PAIR(MediaEventType, MEPolicyChanged),
    MAKE_ENUM_PAIR(MediaEventType, MEContentProtectionMessage),
    MAKE_ENUM_PAIR(MediaEventType, MEPolicySet),
    MAKE_ENUM_PAIR(MediaEventType, METrustV1Anchor),
    MAKE_ENUM_PAIR(MediaEventType, MEWMDRMLicenseBackupCompleted),
    MAKE_ENUM_PAIR(MediaEventType, MEWMDRMLicenseBackupProgress),
    MAKE_ENUM_PAIR(MediaEventType, MEWMDRMLicenseRestoreCompleted),
    MAKE_ENUM_PAIR(MediaEventType, MEWMDRMLicenseRestoreProgress),
    MAKE_ENUM_PAIR(MediaEventType, MEWMDRMLicenseAcquisitionCompleted),
    MAKE_ENUM_PAIR(MediaEventType, MEWMDRMIndividualizationCompleted),
    MAKE_ENUM_PAIR(MediaEventType, MEWMDRMIndividualizationProgress),
    MAKE_ENUM_PAIR(MediaEventType, MEWMDRMProximityCompleted),
    MAKE_ENUM_PAIR(MediaEventType, MEWMDRMLicenseStoreCleaned),
    MAKE_ENUM_PAIR(MediaEventType, MEWMDRMRevocationDownloadCompleted),
    MAKE_ENUM_PAIR(MediaEventType, MEWMDRMV1Anchor),
    MAKE_ENUM_PAIR(MediaEventType, METransformUnknown),
    MAKE_ENUM_PAIR(MediaEventType, METransformNeedInput),
    MAKE_ENUM_PAIR(MediaEventType, METransformHaveOutput),
    MAKE_ENUM_PAIR(MediaEventType, METransformDrainComplete),
    MAKE_ENUM_PAIR(MediaEventType, METransformMarker),
    MAKE_ENUM_PAIR(MediaEventType, MEByteStreamCharacteristicsChanged),
    MAKE_ENUM_PAIR(MediaEventType, MEVideoCaptureDeviceRemoved),
    MAKE_ENUM_PAIR(MediaEventType, MEVideoCaptureDevicePreempted),
    MAKE_ENUM_PAIR(MediaEventType, MEReservedMax)
};
MAKE_MAP(MediaEventType) MediaEventTypeMap(MediaEventTypePairs, MediaEventTypePairs + sizeof(MediaEventTypePairs) / sizeof(MediaEventTypePairs[0]));

MAKE_ENUM(MFSTREAMSINK_MARKER_TYPE) StreamSinkMarkerTypePairs[] = {
    MAKE_ENUM_PAIR(MFSTREAMSINK_MARKER_TYPE, MFSTREAMSINK_MARKER_DEFAULT),
    MAKE_ENUM_PAIR(MFSTREAMSINK_MARKER_TYPE, MFSTREAMSINK_MARKER_ENDOFSEGMENT),
    MAKE_ENUM_PAIR(MFSTREAMSINK_MARKER_TYPE, MFSTREAMSINK_MARKER_TICK),
    MAKE_ENUM_PAIR(MFSTREAMSINK_MARKER_TYPE, MFSTREAMSINK_MARKER_EVENT)
};
MAKE_MAP(MFSTREAMSINK_MARKER_TYPE) StreamSinkMarkerTypeMap(StreamSinkMarkerTypePairs, StreamSinkMarkerTypePairs + sizeof(StreamSinkMarkerTypePairs) / sizeof(StreamSinkMarkerTypePairs[0]));

#ifdef HAVE_WINRT

#ifdef __cplusplus_winrt
#define _Object Platform::Object^
#define _ObjectObj Platform::Object^
#define _String Platform::String^
#define _StringObj Platform::String^
#define _StringReference ref new Platform::String
#define _DeviceInformationCollection Windows::Devices::Enumeration::DeviceInformationCollection
#define _MediaCapture Windows::Media::Capture::MediaCapture
#define _MediaCaptureVideoPreview Windows::Media::Capture::MediaCapture
#define _MediaCaptureInitializationSettings Windows::Media::Capture::MediaCaptureInitializationSettings
#define _VideoDeviceController Windows::Media::Devices::VideoDeviceController
#define _MediaDeviceController Windows::Media::Devices::VideoDeviceController
#define _MediaEncodingProperties Windows::Media::MediaProperties::IMediaEncodingProperties
#define _VideoEncodingProperties Windows::Media::MediaProperties::VideoEncodingProperties
#define _MediaStreamType Windows::Media::Capture::MediaStreamType
#define _AsyncAction Windows::Foundation::IAsyncAction
#define _AsyncOperation Windows::Foundation::IAsyncOperation
#define _DeviceClass Windows::Devices::Enumeration::DeviceClass
#define _IDeviceInformation Windows::Devices::Enumeration::DeviceInformation
#define _DeviceInformation Windows::Devices::Enumeration::DeviceInformation
#define _DeviceInformationStatics Windows::Devices::Enumeration::DeviceInformation
#define _MediaEncodingProfile Windows::Media::MediaProperties::MediaEncodingProfile
#define _StreamingCaptureMode Windows::Media::Capture::StreamingCaptureMode
#define _PropertySet Windows::Foundation::Collections::PropertySet
#define _Map Windows::Foundation::Collections::PropertySet
#define _PropertyValueStatics Windows::Foundation::PropertyValue
#define _VectorView Windows::Foundation::Collections::IVectorView
#define _StartPreviewToCustomSinkIdAsync StartPreviewToCustomSinkAsync
#define _InitializeWithSettingsAsync InitializeAsync
#define _FindAllAsyncDeviceClass FindAllAsync
#define _MediaExtension Windows::Media::IMediaExtension
#define _ContextCallback Concurrency::details::_ContextCallback
#define BEGIN_CALL_IN_CONTEXT(hr, var, ...) hr = S_OK;\
var._CallInContext([__VA_ARGS__]() {
#define END_CALL_IN_CONTEXT(hr) if (FAILED(hr)) throw Platform::Exception::CreateException(hr);\
});
#define DO_ACTION_SYNCHRONOUSLY(hr, action, ctxt) hr = S_OK;\
CCompletionHandler::PerformActionSynchronously(reinterpret_cast<Windows::Foundation::IAsyncAction^>(action), ctxt)
#define DO_OPERATION_SYNCHRONOUSLY_VECTOR(hr, action, ctxt, pResult, vectortype, elementtype, _type) hr = S_OK;\
pResult = CCompletionHandler::PerformSynchronously<_type^>(reinterpret_cast<Windows::Foundation::IAsyncOperation<_type^>^>(action), ctxt)
#define BEGIN_CREATE_ASYNC(...) reinterpret_cast<ABI::Windows::Foundation::IAsyncAction*>(Concurrency::create_async([__VA_ARGS__]() {
#define END_CREATE_ASYNC(hr) if (FAILED(hr)) throw Platform::Exception::CreateException(hr);\
}))
#define DEFINE_TASK Concurrency::task
#define CREATE_TASK Concurrency::create_task
#define CREATE_OR_CONTINUE_TASK(_task, rettype, func) _task = (_task == Concurrency::task<rettype>()) ? Concurrency::create_task(func) : _task.then([func](rettype) -> rettype { return func(); });
#define MAKE_WRL_OBJ(x) x^
#define MAKE_WRL_REF(x) x^
#define MAKE_WRL_AGILE_REF(x) Platform::Agile<x^>
#define RELEASE_AGILE_WRL(x) x = nullptr;
#define RELEASE_WRL(x) x = nullptr;
#define GET_WRL_OBJ_FROM_REF(objtype, obj, orig, hr) objtype^ obj = orig;\
hr = S_OK;
#define GET_WRL_OBJ_FROM_OBJ(objtype, obj, orig, hr) objtype^ obj = orig;\
hr = S_OK;
#define WRL_ENUM_GET(obj, prefix, prop) obj::##prop
#define WRL_PROP_GET(obj, prop, arg, hr) arg = obj->##prop;\
hr = S_OK;
#define WRL_PROP_PUT(obj, prop, arg, hr) obj->##prop = arg;\
hr = S_OK;
#define WRL_METHOD_BASE(obj, method, ret, hr) ret = obj->##method();\
hr = S_OK;
#define WRL_METHOD(obj, method, ret, hr, ...) ret = obj->##method(__VA_ARGS__);\
hr = S_OK;
#define REF_WRL_OBJ(obj) &obj
#define DEREF_WRL_OBJ(obj) obj
#define DEREF_AGILE_WRL_OBJ(obj) obj.Get()
#define DEREF_AS_NATIVE_WRL_OBJ(type, obj) reinterpret_cast<type*>(obj)
#define PREPARE_TRANSFER_WRL_OBJ(obj) obj
#define ACTIVATE_OBJ(rtclass, objtype, obj, hr) MAKE_WRL_OBJ(objtype) obj = ref new objtype();\
hr = S_OK;
#define ACTIVATE_STATIC_OBJ(rtclass, objtype, obj, hr) objtype obj;\
hr = S_OK;
#else
#define _Object IInspectable*
#define _ObjectObj Microsoft::WRL::ComPtr<IInspectable>
#define _String HSTRING
#define _StringObj Microsoft::WRL::Wrappers::HString
#define _StringReference Microsoft::WRL::Wrappers::HStringReference
#define _DeviceInformationCollection ABI::Windows::Devices::Enumeration::DeviceInformationCollection
#define _MediaCapture ABI::Windows::Media::Capture::IMediaCapture
#define _MediaCaptureVideoPreview ABI::Windows::Media::Capture::IMediaCaptureVideoPreview
#define _MediaCaptureInitializationSettings ABI::Windows::Media::Capture::IMediaCaptureInitializationSettings
#define _VideoDeviceController ABI::Windows::Media::Devices::IVideoDeviceController
#define _MediaDeviceController ABI::Windows::Media::Devices::IMediaDeviceController
#define _MediaEncodingProperties ABI::Windows::Media::MediaProperties::IMediaEncodingProperties
#define _VideoEncodingProperties ABI::Windows::Media::MediaProperties::IVideoEncodingProperties
#define _MediaStreamType ABI::Windows::Media::Capture::MediaStreamType
#define _AsyncAction ABI::Windows::Foundation::IAsyncAction
#define _AsyncOperation ABI::Windows::Foundation::IAsyncOperation
#define _DeviceClass ABI::Windows::Devices::Enumeration::DeviceClass
#define _IDeviceInformation ABI::Windows::Devices::Enumeration::IDeviceInformation
#define _DeviceInformation ABI::Windows::Devices::Enumeration::DeviceInformation
#define _DeviceInformationStatics ABI::Windows::Devices::Enumeration::IDeviceInformationStatics
#define _MediaEncodingProfile ABI::Windows::Media::MediaProperties::IMediaEncodingProfile
#define _StreamingCaptureMode ABI::Windows::Media::Capture::StreamingCaptureMode
#define _PropertySet ABI::Windows::Foundation::Collections::IPropertySet
#define _Map ABI::Windows::Foundation::Collections::IMap<HSTRING, IInspectable *>
#define _PropertyValueStatics ABI::Windows::Foundation::IPropertyValueStatics
#define _VectorView ABI::Windows::Foundation::Collections::IVectorView
#define _StartPreviewToCustomSinkIdAsync StartPreviewToCustomSinkIdAsync
#define _InitializeWithSettingsAsync InitializeWithSettingsAsync
#define _FindAllAsyncDeviceClass FindAllAsyncDeviceClass
#define _MediaExtension ABI::Windows::Media::IMediaExtension
#define _ContextCallback Concurrency_winrt::details::_ContextCallback
#define BEGIN_CALL_IN_CONTEXT(hr, var, ...) hr = var._CallInContext([__VA_ARGS__]() -> HRESULT {
#define END_CALL_IN_CONTEXT(hr) return hr;\
});
#define DO_ACTION_SYNCHRONOUSLY(hr, action, ctxt) hr = CCompletionHandler<ABI::Windows::Foundation::IAsyncActionCompletedHandler, ABI::Windows::Foundation::IAsyncAction>::PerformActionSynchronously(action, ctxt)
#define DO_OPERATION_SYNCHRONOUSLY_VECTOR(hr, action, ctxt, pResult, vectortype, elementtype, _type) hr = CCompletionHandler<ABI::Windows::Foundation::IAsyncOperationCompletedHandler<_type*>, ABI::Windows::Foundation::IAsyncOperation<_type*>>::PerformSynchronously<vectortype<elementtype*>*>(action, ctxt, pResult.GetAddressOf())
#define BEGIN_CREATE_ASYNC(...) Concurrency_winrt::create_async([__VA_ARGS__]() -> HRESULT {
#define END_CREATE_ASYNC(hr) return hr;\
})
#define DEFINE_TASK Concurrency_winrt::task
#define CREATE_TASK Concurrency_winrt::create_task
#define CREATE_OR_CONTINUE_TASK(_task, rettype, func) _task = (_task == Concurrency_winrt::task<rettype>()) ? Concurrency_winrt::create_task(func) : _task.then([func](rettype) -> rettype { return func(); });
#define MAKE_WRL_OBJ(x) Microsoft::WRL::ComPtr<x>
#define MAKE_WRL_REF(x) x*
#define MAKE_WRL_AGILE_REF(x) x*
#define RELEASE_AGILE_WRL(x) if (x) { (x)->Release(); x = nullptr; }
#define RELEASE_WRL(x) if (x) { (x)->Release(); x = nullptr; }
#define GET_WRL_OBJ_FROM_REF(objtype, obj, orig, hr) Microsoft::WRL::ComPtr<objtype> obj;\
hr = orig->QueryInterface(__uuidof(objtype), &obj);
#define GET_WRL_OBJ_FROM_OBJ(objtype, obj, orig, hr) Microsoft::WRL::ComPtr<objtype> obj;\
hr = orig.As(&obj);
#define WRL_ENUM_GET(obj, prefix, prop) obj::prefix##_##prop
#define WRL_PROP_GET(obj, prop, arg, hr) hr = obj->get_##prop(&arg);
#define WRL_PROP_PUT(obj, prop, arg, hr) hr = obj->put_##prop(arg);
#define WRL_METHOD_BASE(obj, method, ret, hr) hr = obj->##method(&ret);
#define WRL_METHOD(obj, method, ret, hr, ...) hr = obj->##method(__VA_ARGS__, &ret);
#define REF_WRL_OBJ(obj) obj.GetAddressOf()
#define DEREF_WRL_OBJ(obj) obj.Get()
#define DEREF_AGILE_WRL_OBJ(obj) obj
#define DEREF_AS_NATIVE_WRL_OBJ(type, obj) obj.Get()
#define PREPARE_TRANSFER_WRL_OBJ(obj) obj.Detach()
#define ACTIVATE_OBJ(rtclass, objtype, obj, hr) MAKE_WRL_OBJ(objtype) obj;\
{\
    Microsoft::WRL::ComPtr<IActivationFactory> objFactory;\
    hr = Windows::Foundation::GetActivationFactory(Microsoft::WRL::Wrappers::HStringReference(rtclass).Get(), objFactory.ReleaseAndGetAddressOf());\
    if (SUCCEEDED(hr)) {\
        Microsoft::WRL::ComPtr<IInspectable> pInsp;\
        hr = objFactory->ActivateInstance(pInsp.GetAddressOf());\
        if (SUCCEEDED(hr)) hr = pInsp.As(&obj);\
    }\
}
#define ACTIVATE_STATIC_OBJ(rtclass, objtype, obj, hr) objtype obj;\
{\
    Microsoft::WRL::ComPtr<IActivationFactory> objFactory;\
    hr = Windows::Foundation::GetActivationFactory(Microsoft::WRL::Wrappers::HStringReference(rtclass).Get(), objFactory.ReleaseAndGetAddressOf());\
    if (SUCCEEDED(hr)) {\
        if (SUCCEEDED(hr)) hr = objFactory.As(&obj);\
    }\
}
#endif

#define GET_CURRENT_CONTEXT _ContextCallback::_CaptureCurrent()
#define SAVE_CURRENT_CONTEXT(var) _ContextCallback var = GET_CURRENT_CONTEXT

#ifdef __cplusplus_winrt
ref class CCompletionHandler sealed
#else
template <typename TCompletionHandler, typename TAction>
class CCompletionHandler
    : public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags< Microsoft::WRL::RuntimeClassType::ClassicCom>,
    TCompletionHandler, IAgileObject, FtmBase>
#endif
{
    MixInHelper()
#ifndef __cplusplus_winrt
public:
    CCompletionHandler() {}

    STDMETHODIMP Invoke(TAction* /*asyncInfo*/, AsyncStatus /*asyncStatus*/)
    {
        m_Event.set();
        return S_OK;
    }
    void wait() { m_Event.wait(); }
#endif
#ifdef __cplusplus_winrt
internal:
    template <typename TResult>
    static TResult PerformSynchronously(Windows::Foundation::IAsyncOperation<TResult>^ asyncOp, _ContextCallback context)
    {
        TResult pResult;
        context._CallInContext([asyncOp, &pResult]() { Concurrency::task<TResult> asyncTask = Concurrency::task<TResult>(asyncOp); pResult = asyncTask.get(); });
        return pResult;
#else
    template <typename TResult>
    static HRESULT PerformSynchronously(TAction* asyncOp, _ContextCallback context, TResult* pResult)
    {
        HRESULT hr;
        ComPtr<CCompletionHandler<TCompletionHandler, TAction>> completeHandler = Microsoft::WRL::Make<CCompletionHandler<TCompletionHandler, TAction>>();
        hr = context._CallInContext([&asyncOp, &completeHandler]() -> HRESULT {
            HRESULT hr = asyncOp->put_Completed(completeHandler.Get());
            if (FAILED(hr)) asyncOp->Release();
            return hr;
        });
        if (SUCCEEDED(hr))
            completeHandler->wait();
        else
            return hr;
        hr = context._CallInContext([&asyncOp, &pResult]() -> HRESULT {
            HRESULT hr = asyncOp->GetResults(pResult);
            asyncOp->Release();
            return hr;
        });
        return hr;
#endif
    }

#ifdef __cplusplus_winrt
    static void PerformActionSynchronously(Windows::Foundation::IAsyncAction^ asyncOp, _ContextCallback context)
    {
        context._CallInContext([asyncOp](){ Concurrency::task<void>(asyncOp).get(); });
#else
    static HRESULT PerformActionSynchronously(TAction* asyncOp, _ContextCallback context)
    {
        HRESULT hr;
        ComPtr<CCompletionHandler<TCompletionHandler, TAction>> completeHandler = Microsoft::WRL::Make<CCompletionHandler<TCompletionHandler, TAction>>();
        hr = context._CallInContext([&asyncOp, &completeHandler]() -> HRESULT {
            HRESULT hr = asyncOp->put_Completed(completeHandler.Get());
            if (FAILED(hr)) asyncOp->Release();
            return hr;
        });
        if (SUCCEEDED(hr))
            completeHandler->wait();
        else
            return hr;
        hr = context._CallInContext([&asyncOp]() -> HRESULT {
            HRESULT hr = asyncOp->GetResults();
            asyncOp->Release();
            return hr;
        });
        return hr;
#endif
    }
#ifndef __cplusplus_winrt
private:
    Concurrency::event m_Event;
#endif
};

#ifndef __cplusplus_winrt

// Helpers for create_async validation
//
// A parameter lambda taking no arguments is valid
template<typename _Ty>
static auto _IsValidCreateAsync(_Ty _Param, int, int, int, int) -> typename decltype(_Param(), std::true_type());

// A parameter lambda taking an cancellation_token argument is valid
template<typename _Ty>
static auto _IsValidCreateAsync(_Ty _Param, int, int, int, ...) -> typename decltype(_Param(cancellation_token::none()), std::true_type());

// A parameter lambda taking a progress report argument is valid
template<typename _Ty>
static auto _IsValidCreateAsync(_Ty _Param, int, int, ...) -> typename decltype(_Param(details::_ProgressReporterCtorArgType()), std::true_type());

// A parameter lambda taking a progress report and a cancellation_token argument is valid
template<typename _Ty>
static auto _IsValidCreateAsync(_Ty _Param, int, ...) -> typename decltype(_Param(details::_ProgressReporterCtorArgType(), cancellation_token::none()), std::true_type());

// All else is invalid
template<typename _Ty>
static std::false_type _IsValidCreateAsync(_Ty _Param, ...);

//for task specific architecture
//could add a CancelPending which is set when Cancel is called, return as Cancel when get_Status is called and set when a task_canceled exception is thrown

extern const __declspec(selectany) WCHAR RuntimeClass_CV_CAsyncAction[] = L"cv.CAsyncAction";

template<typename _Function>
class CAsyncAction
    : public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags< Microsoft::WRL::RuntimeClassType::WinRt>,
    Microsoft::WRL::Implements<ABI::Windows::Foundation::IAsyncAction>, Microsoft::WRL::AsyncBase<ABI::Windows::Foundation::IAsyncActionCompletedHandler >>
{
    InspectableClass(RuntimeClass_CV_CAsyncAction, BaseTrust)
public:
    STDMETHOD(RuntimeClassInitialize)() { return S_OK; }
    virtual ~CAsyncAction() {}
    CAsyncAction(const _Function &_Func) : _M_func(_Func) {
        Start();
    }
    void _SetTaskCreationAddressHint(void* _SourceAddressHint)
    {
        if (!(std::is_same<Concurrency_winrt::details::_TaskTypeTraits<Concurrency_winrt::task<HRESULT>>::_AsyncKind, Concurrency_winrt::details::_TypeSelectorAsyncTask>::value))
        {
            // Overwrite the creation address with the return address of create_async unless the
            // lambda returned a task. If the create async lambda returns a task, that task is reused and
            // we want to preserve its creation address hint.
            _M_task._SetTaskCreationAddressHint(_SourceAddressHint);
        }
    }
    HRESULT STDMETHODCALLTYPE put_Completed(
        /* [in] */ __RPC__in_opt ABI::Windows::Foundation::IAsyncActionCompletedHandler *handler)
    {
        HRESULT hr;
        if (SUCCEEDED(hr = PutOnComplete(handler)) && cCallbackMade_ == 0) {
            //okay to use default implementation even for the callback as already running in context
            //otherwise check for the alternate case and save the context
            _M_completeDelegateContext = _ContextCallback::_CaptureCurrent();
        }
        return hr;
    }
    HRESULT STDMETHODCALLTYPE get_Completed(
        /* [out][retval] */ __RPC__deref_out_opt ABI::Windows::Foundation::IAsyncActionCompletedHandler **handler) {
        if (!handler) return E_POINTER;
        return GetOnComplete(handler);
    }
    HRESULT STDMETHODCALLTYPE GetResults(void) {
        HRESULT hr = CheckValidStateForResultsCall();
        if (SUCCEEDED(hr)) {
            _M_task.get();
        }
        return hr;
    }
    HRESULT OnStart() {
        _M_task = Concurrency_winrt::task<HRESULT>(_M_func, _M_cts.get_token());
        AddRef();
        _M_task.then([this](Concurrency_winrt::task<HRESULT> _Antecedent) {
            try {
                HRESULT hr = _Antecedent.get();
                if (FAILED(hr)) TryTransitionToError(hr);
            }
            catch (Concurrency::task_canceled&){
            }
            catch (...) {
                TryTransitionToError(E_FAIL);
            }
            _FireCompletion();
            Release();
        });
        return S_OK;
    }
    void OnClose() {}
    void OnCancel() { _M_cts.cancel(); }

protected:
    //modified for _CallInContext to support UI STA thread
    //can wrap the base clase implementation or duplicate it but must use get_Completed to fetch the private member variable
    virtual void _FireCompletion()
    {
        AddRef();
        _M_completeDelegateContext._CallInContext([this]() -> HRESULT {
            FireCompletion();
            Release();
            return S_OK;
        });
    }
private:

    _Function _M_func;
    Concurrency_winrt::task<HRESULT> _M_task;
    Concurrency::cancellation_token_source _M_cts;
    _ContextCallback        _M_completeDelegateContext;
};

template<typename _Function>
__declspec(noinline)
CAsyncAction<_Function>* create_async(const _Function& _Func)
{
    static_assert(std::is_same<decltype(_IsValidCreateAsync(_Func, 0, 0, 0, 0)), std::true_type>::value,
        "argument to create_async must be a callable object taking zero, one or two arguments");
    CAsyncAction<_Function>* action = Microsoft::WRL::Make<CAsyncAction<_Function>>(_Func).Detach();
    action->_SetTaskCreationAddressHint(_ReturnAddress());
    return action;
}
#endif

EXTERN_C const IID IID_IMedCapFailHandler;

class DECLSPEC_UUID("CE22BEDB-0B3C-4BE0-BE8F-E53AB457EA2C") DECLSPEC_NOVTABLE IMedCapFailHandler : public IUnknown
{
public:
    virtual HRESULT AddHandler(ABI::Windows::Media::Capture::IMediaCapture* pMedCap) = 0;
    virtual HRESULT RemoveHandler(ABI::Windows::Media::Capture::IMediaCapture* pMedCap) = 0;
};

template<typename _Function>
class MediaCaptureFailedHandler :
    public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags< Microsoft::WRL::RuntimeClassType::ClassicCom>,
    IMedCapFailHandler, ABI::Windows::Media::Capture::IMediaCaptureFailedEventHandler, IAgileObject, FtmBase>
{
public:
    MediaCaptureFailedHandler(const _Function &_Func) : _M_func(_Func) { m_cookie.value = 0; }
    HRESULT AddHandler(ABI::Windows::Media::Capture::IMediaCapture* pMedCap)
    {
        return pMedCap->add_Failed(this, &m_cookie);
    }
    HRESULT RemoveHandler(ABI::Windows::Media::Capture::IMediaCapture* pMedCap)
    {
        return pMedCap->remove_Failed(m_cookie);
    }
    HRESULT STDMETHODCALLTYPE Invoke(
        ABI::Windows::Media::Capture::IMediaCapture *sender,
        ABI::Windows::Media::Capture::IMediaCaptureFailedEventArgs *errorEventArgs)
    {
        (void)sender;
        (void)errorEventArgs;
        AddRef();
        _M_func();
        Release();
        return S_OK;
    }

private:
    _Function _M_func;
    EventRegistrationToken m_cookie;
};

template<typename _Function>
__declspec(noinline)
MediaCaptureFailedHandler<_Function>* create_medcapfailedhandler(const _Function& _Func)
{
    return Microsoft::WRL::Make<MediaCaptureFailedHandler<_Function>>(_Func).Detach();
}

#endif

template <class TBase=IMFAttributes>
class CBaseAttributes : public TBase
{
protected:
    // This version of the constructor does not initialize the
    // attribute store. The derived class must call Initialize() in
    // its own constructor.
    CBaseAttributes()
    {
    }

    // This version of the constructor initializes the attribute
    // store, but the derived class must pass an HRESULT parameter
    // to the constructor.

    CBaseAttributes(HRESULT& hr, UINT32 cInitialSize = 0)
    {
        hr = Initialize(cInitialSize);
    }

    // The next version of the constructor uses a caller-provided
    // implementation of IMFAttributes.

    // (Sometimes you want to delegate IMFAttributes calls to some
    // other object that implements IMFAttributes, rather than using
    // MFCreateAttributes.)

    CBaseAttributes(HRESULT& hr, IUnknown *pUnk)
    {
        hr = Initialize(pUnk);
    }

    virtual ~CBaseAttributes()
    {
    }

    // Initializes the object by creating the standard Media Foundation attribute store.
    HRESULT Initialize(UINT32 cInitialSize = 0)
    {
        if (_spAttributes.Get() == nullptr)
        {
            return MFCreateAttributes(&_spAttributes, cInitialSize);
        }
        else
        {
            return S_OK;
        }
    }

    // Initializes this object from a caller-provided attribute store.
    // pUnk: Pointer to an object that exposes IMFAttributes.
    HRESULT Initialize(IUnknown *pUnk)
    {
        if (_spAttributes)
        {
            _spAttributes.Reset();
            _spAttributes = nullptr;
        }


        return pUnk->QueryInterface(IID_PPV_ARGS(&_spAttributes));
    }

public:

    // IMFAttributes methods

    STDMETHODIMP GetItem(REFGUID guidKey, PROPVARIANT* pValue)
    {
        assert(_spAttributes);
        return _spAttributes->GetItem(guidKey, pValue);
    }

    STDMETHODIMP GetItemType(REFGUID guidKey, MF_ATTRIBUTE_TYPE* pType)
    {
        assert(_spAttributes);
        return _spAttributes->GetItemType(guidKey, pType);
    }

    STDMETHODIMP CompareItem(REFGUID guidKey, REFPROPVARIANT Value, BOOL* pbResult)
    {
        assert(_spAttributes);
        return _spAttributes->CompareItem(guidKey, Value, pbResult);
    }

    STDMETHODIMP Compare(
        IMFAttributes* pTheirs,
        MF_ATTRIBUTES_MATCH_TYPE MatchType,
        BOOL* pbResult
        )
    {
        assert(_spAttributes);
        return _spAttributes->Compare(pTheirs, MatchType, pbResult);
    }

    STDMETHODIMP GetUINT32(REFGUID guidKey, UINT32* punValue)
    {
        assert(_spAttributes);
        return _spAttributes->GetUINT32(guidKey, punValue);
    }

    STDMETHODIMP GetUINT64(REFGUID guidKey, UINT64* punValue)
    {
        assert(_spAttributes);
        return _spAttributes->GetUINT64(guidKey, punValue);
    }

    STDMETHODIMP GetDouble(REFGUID guidKey, double* pfValue)
    {
        assert(_spAttributes);
        return _spAttributes->GetDouble(guidKey, pfValue);
    }

    STDMETHODIMP GetGUID(REFGUID guidKey, GUID* pguidValue)
    {
        assert(_spAttributes);
        return _spAttributes->GetGUID(guidKey, pguidValue);
    }

    STDMETHODIMP GetStringLength(REFGUID guidKey, UINT32* pcchLength)
    {
        assert(_spAttributes);
        return _spAttributes->GetStringLength(guidKey, pcchLength);
    }

    STDMETHODIMP GetString(REFGUID guidKey, LPWSTR pwszValue, UINT32 cchBufSize, UINT32* pcchLength)
    {
        assert(_spAttributes);
        return _spAttributes->GetString(guidKey, pwszValue, cchBufSize, pcchLength);
    }

    STDMETHODIMP GetAllocatedString(REFGUID guidKey, LPWSTR* ppwszValue, UINT32* pcchLength)
    {
        assert(_spAttributes);
        return _spAttributes->GetAllocatedString(guidKey, ppwszValue, pcchLength);
    }

    STDMETHODIMP GetBlobSize(REFGUID guidKey, UINT32* pcbBlobSize)
    {
        assert(_spAttributes);
        return _spAttributes->GetBlobSize(guidKey, pcbBlobSize);
    }

    STDMETHODIMP GetBlob(REFGUID guidKey, UINT8* pBuf, UINT32 cbBufSize, UINT32* pcbBlobSize)
    {
        assert(_spAttributes);
        return _spAttributes->GetBlob(guidKey, pBuf, cbBufSize, pcbBlobSize);
    }

    STDMETHODIMP GetAllocatedBlob(REFGUID guidKey, UINT8** ppBuf, UINT32* pcbSize)
    {
        assert(_spAttributes);
        return _spAttributes->GetAllocatedBlob(guidKey, ppBuf, pcbSize);
    }

    STDMETHODIMP GetUnknown(REFGUID guidKey, REFIID riid, LPVOID* ppv)
    {
        assert(_spAttributes);
        return _spAttributes->GetUnknown(guidKey, riid, ppv);
    }

    STDMETHODIMP SetItem(REFGUID guidKey, REFPROPVARIANT Value)
    {
        assert(_spAttributes);
        return _spAttributes->SetItem(guidKey, Value);
    }

    STDMETHODIMP DeleteItem(REFGUID guidKey)
    {
        assert(_spAttributes);
        return _spAttributes->DeleteItem(guidKey);
    }

    STDMETHODIMP DeleteAllItems()
    {
        assert(_spAttributes);
        return _spAttributes->DeleteAllItems();
    }

    STDMETHODIMP SetUINT32(REFGUID guidKey, UINT32 unValue)
    {
        assert(_spAttributes);
        return _spAttributes->SetUINT32(guidKey, unValue);
    }

    STDMETHODIMP SetUINT64(REFGUID guidKey,UINT64 unValue)
    {
        assert(_spAttributes);
        return _spAttributes->SetUINT64(guidKey, unValue);
    }

    STDMETHODIMP SetDouble(REFGUID guidKey, double fValue)
    {
        assert(_spAttributes);
        return _spAttributes->SetDouble(guidKey, fValue);
    }

    STDMETHODIMP SetGUID(REFGUID guidKey, REFGUID guidValue)
    {
        assert(_spAttributes);
        return _spAttributes->SetGUID(guidKey, guidValue);
    }

    STDMETHODIMP SetString(REFGUID guidKey, LPCWSTR wszValue)
    {
        assert(_spAttributes);
        return _spAttributes->SetString(guidKey, wszValue);
    }

    STDMETHODIMP SetBlob(REFGUID guidKey, const UINT8* pBuf, UINT32 cbBufSize)
    {
        assert(_spAttributes);
        return _spAttributes->SetBlob(guidKey, pBuf, cbBufSize);
    }

    STDMETHODIMP SetUnknown(REFGUID guidKey, IUnknown* pUnknown)
    {
        assert(_spAttributes);
        return _spAttributes->SetUnknown(guidKey, pUnknown);
    }

    STDMETHODIMP LockStore()
    {
        assert(_spAttributes);
        return _spAttributes->LockStore();
    }

    STDMETHODIMP UnlockStore()
    {
        assert(_spAttributes);
        return _spAttributes->UnlockStore();
    }

    STDMETHODIMP GetCount(UINT32* pcItems)
    {
        assert(_spAttributes);
        return _spAttributes->GetCount(pcItems);
    }

    STDMETHODIMP GetItemByIndex(UINT32 unIndex, GUID* pguidKey, PROPVARIANT* pValue)
    {
        assert(_spAttributes);
        return _spAttributes->GetItemByIndex(unIndex, pguidKey, pValue);
    }

    STDMETHODIMP CopyAllItems(IMFAttributes* pDest)
    {
        assert(_spAttributes);
        return _spAttributes->CopyAllItems(pDest);
    }

    // Helper functions

    HRESULT SerializeToStream(DWORD dwOptions, IStream* pStm)
        // dwOptions: Flags from MF_ATTRIBUTE_SERIALIZE_OPTIONS
    {
        assert(_spAttributes);
        return MFSerializeAttributesToStream(_spAttributes.Get(), dwOptions, pStm);
    }

    HRESULT DeserializeFromStream(DWORD dwOptions, IStream* pStm)
    {
        assert(_spAttributes);
        return MFDeserializeAttributesFromStream(_spAttributes.Get(), dwOptions, pStm);
    }

    // SerializeToBlob: Stores the attributes in a byte array.
    //
    // ppBuf: Receives a pointer to the byte array.
    // pcbSize: Receives the size of the byte array.
    //
    // The caller must free the array using CoTaskMemFree.
    HRESULT SerializeToBlob(UINT8 **ppBuffer, UINT *pcbSize)
    {
        assert(_spAttributes);

        if (ppBuffer == NULL)
        {
            return E_POINTER;
        }
        if (pcbSize == NULL)
        {
            return E_POINTER;
        }

        HRESULT hr = S_OK;
        UINT32 cbSize = 0;
        BYTE *pBuffer = NULL;

        CHECK_HR(hr = MFGetAttributesAsBlobSize(_spAttributes.Get(), &cbSize));

        pBuffer = (BYTE*)CoTaskMemAlloc(cbSize);
        if (pBuffer == NULL)
        {
            CHECK_HR(hr = E_OUTOFMEMORY);
        }

        CHECK_HR(hr = MFGetAttributesAsBlob(_spAttributes.Get(), pBuffer, cbSize));

        *ppBuffer = pBuffer;
        *pcbSize = cbSize;

done:
        if (FAILED(hr))
        {
            *ppBuffer = NULL;
            *pcbSize = 0;
            CoTaskMemFree(pBuffer);
        }
        return hr;
    }

    HRESULT DeserializeFromBlob(const UINT8* pBuffer, UINT cbSize)
    {
        assert(_spAttributes);
        return MFInitAttributesFromBlob(_spAttributes.Get(), pBuffer, cbSize);
    }

    HRESULT GetRatio(REFGUID guidKey, UINT32* pnNumerator, UINT32* punDenominator)
    {
        assert(_spAttributes);
        return MFGetAttributeRatio(_spAttributes.Get(), guidKey, pnNumerator, punDenominator);
    }

    HRESULT SetRatio(REFGUID guidKey, UINT32 unNumerator, UINT32 unDenominator)
    {
        assert(_spAttributes);
        return MFSetAttributeRatio(_spAttributes.Get(), guidKey, unNumerator, unDenominator);
    }

    // Gets an attribute whose value represents the size of something (eg a video frame).
    HRESULT GetSize(REFGUID guidKey, UINT32* punWidth, UINT32* punHeight)
    {
        assert(_spAttributes);
        return MFGetAttributeSize(_spAttributes.Get(), guidKey, punWidth, punHeight);
    }

    // Sets an attribute whose value represents the size of something (eg a video frame).
    HRESULT SetSize(REFGUID guidKey, UINT32 unWidth, UINT32 unHeight)
    {
        assert(_spAttributes);
        return MFSetAttributeSize (_spAttributes.Get(), guidKey, unWidth, unHeight);
    }

protected:
    ComPtr<IMFAttributes> _spAttributes;
};

class StreamSink :
#ifdef HAVE_WINRT
    public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags< Microsoft::WRL::RuntimeClassType::ClassicCom>,
    IMFStreamSink,
    IMFMediaEventGenerator,
    IMFMediaTypeHandler,
    CBaseAttributes<> >
#else
    public IMFStreamSink,
    public IMFMediaTypeHandler,
    public CBaseAttributes<>,
    public ICustomStreamSink
#endif
{
public:
    // IUnknown methods
    STDMETHOD(QueryInterface)(REFIID riid, _Outptr_result_nullonfailure_ void **ppv)
    {
        if (ppv == nullptr) {
            return E_POINTER;
        }
        (*ppv) = nullptr;
        HRESULT hr = S_OK;
        if (riid == IID_IMarshal) {
            return MarshalQI(riid, ppv);
        } else {
#ifdef HAVE_WINRT
            hr = RuntimeClassT::QueryInterface(riid, ppv);
#else
            if (riid == IID_IUnknown || riid == IID_IMFStreamSink) {
                *ppv = static_cast<IMFStreamSink*>(this);
                AddRef();
            } else if (riid == IID_IMFMediaEventGenerator) {
                *ppv = static_cast<IMFMediaEventGenerator*>(this);
                AddRef();
            } else if (riid == IID_IMFMediaTypeHandler) {
                *ppv = static_cast<IMFMediaTypeHandler*>(this);
                AddRef();
            } else if (riid == IID_IMFAttributes) {
                *ppv = static_cast<IMFAttributes*>(this);
                AddRef();
            } else if (riid == IID_ICustomStreamSink) {
                *ppv = static_cast<ICustomStreamSink*>(this);
                AddRef();
            } else
                hr = E_NOINTERFACE;
#endif
        }

        return hr;
    }

#ifdef HAVE_WINRT
    STDMETHOD(RuntimeClassInitialize)() { return S_OK; }
#else
    ULONG AddRef()
    {
        return InterlockedIncrement(&m_cRef);
    }
    ULONG Release()
    {
        ULONG cRef = InterlockedDecrement(&m_cRef);
        if (cRef == 0)
        {
            delete this;
        }
        return cRef;
    }
#endif
    HRESULT MarshalQI(REFIID riid, LPVOID* ppv)
    {
        HRESULT hr = S_OK;
        if (m_spFTM == nullptr) {
            EnterCriticalSection(&m_critSec);
            if (m_spFTM == nullptr) {
                hr = CoCreateFreeThreadedMarshaler(static_cast<IMFStreamSink*>(this), &m_spFTM);
            }
            LeaveCriticalSection(&m_critSec);
        }

        if (SUCCEEDED(hr)) {
            if (m_spFTM == nullptr) {
                hr = E_UNEXPECTED;
            }
            else {
                hr = m_spFTM.Get()->QueryInterface(riid, ppv);
            }
        }
        return hr;
    }
    enum State
    {
        State_TypeNotSet = 0,    // No media type is set
        State_Ready,             // Media type is set, Start has never been called.
        State_Started,
        State_Stopped,
        State_Paused,
        State_Count              // Number of states
    };
    StreamSink() : m_IsShutdown(false),
        m_StartTime(0), m_fGetStartTimeFromSample(false), m_fWaitingForFirstSample(false),
        m_state(State_TypeNotSet), m_pParent(nullptr),
        m_imageWidthInPixels(0), m_imageHeightInPixels(0) {
#ifdef HAVE_WINRT
        m_token.value = 0;
#else
        m_bConnected = false;
#endif
        InitializeCriticalSectionEx(&m_critSec, 3000, 0);
        ZeroMemory(&m_guiCurrentSubtype, sizeof(m_guiCurrentSubtype));
        CBaseAttributes::Initialize(0U);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"StreamSink::StreamSink\n");
    }
    virtual ~StreamSink() {
        DeleteCriticalSection(&m_critSec);
        assert(m_IsShutdown);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"StreamSink::~StreamSink\n");
    }

    HRESULT Initialize()
    {
        HRESULT hr;
        // Create the event queue helper.
        hr = MFCreateEventQueue(&m_spEventQueue);
        if (SUCCEEDED(hr))
        {
            ComPtr<IMFMediaSink> pMedSink;
            hr = CBaseAttributes<>::GetUnknown(MF_STREAMSINK_MEDIASINKINTERFACE, __uuidof(IMFMediaSink), (LPVOID*)pMedSink.GetAddressOf());
            assert(pMedSink.Get() != NULL);
            if (SUCCEEDED(hr)) {
                hr = pMedSink.Get()->QueryInterface(IID_PPV_ARGS(&m_pParent));
            }
        }
        return hr;
    }

    HRESULT CheckShutdown() const
    {
        if (m_IsShutdown)
        {
            return MF_E_SHUTDOWN;
        }
        else
        {
            return S_OK;
        }
    }
    // Called when the presentation clock starts.
    HRESULT Start(MFTIME start)
    {
        EnterCriticalSection(&m_critSec);

        HRESULT hr = S_OK;

        if (m_state != State_TypeNotSet) {
            if (start != PRESENTATION_CURRENT_POSITION)
            {
                m_StartTime = start;        // Cache the start time.
                m_fGetStartTimeFromSample = false;
            }
            else
            {
                m_fGetStartTimeFromSample = true;
            }
            m_state = State_Started;
            GUID guiMajorType;
            m_fWaitingForFirstSample = SUCCEEDED(m_spCurrentType->GetMajorType(&guiMajorType)) && (guiMajorType == MFMediaType_Video);
            hr = QueueEvent(MEStreamSinkStarted, GUID_NULL, hr, NULL);
            if (SUCCEEDED(hr)) {
                hr = QueueEvent(MEStreamSinkRequestSample, GUID_NULL, hr, NULL);
            }
        }
        else hr = MF_E_NOT_INITIALIZED;
        LeaveCriticalSection(&m_critSec);
        return hr;
    }

    // Called when the presentation clock pauses.
    HRESULT Pause()
    {
        EnterCriticalSection(&m_critSec);

        HRESULT hr = S_OK;

        if (m_state != State_Stopped && m_state != State_TypeNotSet) {
            m_state = State_Paused;
            hr = QueueEvent(MEStreamSinkPaused, GUID_NULL, hr, NULL);
        } else if (hr == State_TypeNotSet)
            hr = MF_E_NOT_INITIALIZED;
        else
            hr = MF_E_INVALIDREQUEST;
        LeaveCriticalSection(&m_critSec);
        return hr;
    }
    // Called when the presentation clock restarts.
    HRESULT Restart()
    {
        EnterCriticalSection(&m_critSec);

        HRESULT hr = S_OK;

        if (m_state == State_Paused) {
            m_state = State_Started;
            hr = QueueEvent(MEStreamSinkStarted, GUID_NULL, hr, NULL);
            if (SUCCEEDED(hr)) {
                hr = QueueEvent(MEStreamSinkRequestSample, GUID_NULL, hr, NULL);
            }
        } else if (hr == State_TypeNotSet)
            hr = MF_E_NOT_INITIALIZED;
        else
            hr = MF_E_INVALIDREQUEST;
        LeaveCriticalSection(&m_critSec);
        return hr;
    }
    // Called when the presentation clock stops.
    HRESULT Stop()
    {
        EnterCriticalSection(&m_critSec);

        HRESULT hr = S_OK;
        if (m_state != State_TypeNotSet) {
            m_state = State_Stopped;
            hr = QueueEvent(MEStreamSinkStopped, GUID_NULL, hr, NULL);
        }
        else hr = MF_E_NOT_INITIALIZED;
        LeaveCriticalSection(&m_critSec);
        return hr;
    }

    // Shuts down the stream sink.
    HRESULT Shutdown()
    {
        ComPtr<IMFSampleGrabberSinkCallback> pSampleCallback;
        HRESULT hr = S_OK;
        assert(!m_IsShutdown);
        hr = m_pParent->GetUnknown(MF_MEDIASINK_SAMPLEGRABBERCALLBACK, IID_IMFSampleGrabberSinkCallback, (LPVOID*)pSampleCallback.GetAddressOf());
        if (SUCCEEDED(hr)) {
            hr = pSampleCallback->OnShutdown();
        }

        if (m_spEventQueue) {
            hr = m_spEventQueue->Shutdown();
        }
        if (m_pParent)
            m_pParent->Release();
        m_spCurrentType.Reset();
        m_IsShutdown = TRUE;

        return hr;
    }

    //IMFStreamSink
    HRESULT STDMETHODCALLTYPE GetMediaSink(
    /* [out] */ __RPC__deref_out_opt IMFMediaSink **ppMediaSink) {
        if (ppMediaSink == NULL)
        {
            return E_INVALIDARG;
        }

        EnterCriticalSection(&m_critSec);

        HRESULT hr = CheckShutdown();

        if (SUCCEEDED(hr))
        {
            ComPtr<IMFMediaSink> pMedSink;
            hr = CBaseAttributes<>::GetUnknown(MF_STREAMSINK_MEDIASINKINTERFACE, __uuidof(IMFMediaSink), (LPVOID*)pMedSink.GetAddressOf());
            if (SUCCEEDED(hr)) {
                *ppMediaSink = pMedSink.Detach();
            }
        }

        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"StreamSink::GetMediaSink: HRESULT=%i\n", hr);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE GetIdentifier(
        /* [out] */ __RPC__out DWORD *pdwIdentifier) {
        if (pdwIdentifier == NULL)
        {
            return E_INVALIDARG;
        }

        EnterCriticalSection(&m_critSec);

        HRESULT hr = CheckShutdown();

        if (SUCCEEDED(hr))
        {
            hr = GetUINT32(MF_STREAMSINK_ID, (UINT32*)pdwIdentifier);
        }

        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"StreamSink::GetIdentifier: HRESULT=%i\n", hr);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE GetMediaTypeHandler(
        /* [out] */ __RPC__deref_out_opt IMFMediaTypeHandler **ppHandler) {
        if (ppHandler == NULL)
        {
            return E_INVALIDARG;
        }

        EnterCriticalSection(&m_critSec);

        HRESULT hr = CheckShutdown();

        // This stream object acts as its own type handler, so we QI ourselves.
        if (SUCCEEDED(hr))
        {
            hr = QueryInterface(IID_IMFMediaTypeHandler, (void**)ppHandler);
        }

        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"StreamSink::GetMediaTypeHandler: HRESULT=%i\n", hr);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE ProcessSample(IMFSample *pSample) {
        ComPtr<IMFMediaBuffer> pInput;
        ComPtr<IMFSampleGrabberSinkCallback> pSampleCallback;
        BYTE *pSrc = NULL;          // Source buffer.
        // Stride if the buffer does not support IMF2DBuffer
        LONGLONG hnsTime = 0;
        LONGLONG hnsDuration = 0;
        DWORD cbMaxLength;
        DWORD cbCurrentLength = 0;
        GUID guidMajorType;
        if (pSample == NULL)
        {
            return E_INVALIDARG;
        }
        HRESULT hr = S_OK;

        EnterCriticalSection(&m_critSec);

        if (m_state != State_Started && m_state != State_Paused) {
            if (m_state == State_TypeNotSet)
                hr = MF_E_NOT_INITIALIZED;
            else
                hr = MF_E_INVALIDREQUEST;
        }
        if (SUCCEEDED(hr))
            hr = CheckShutdown();
        if (SUCCEEDED(hr)) {
            hr = pSample->ConvertToContiguousBuffer(&pInput);
            if (SUCCEEDED(hr)) {
                hr = pSample->GetSampleTime(&hnsTime);
            }
            if (SUCCEEDED(hr)) {
                hr = pSample->GetSampleDuration(&hnsDuration);
            }
            if (SUCCEEDED(hr)) {
                hr = GetMajorType(&guidMajorType);
            }
            if (SUCCEEDED(hr)) {
                hr = m_pParent->GetUnknown(MF_MEDIASINK_SAMPLEGRABBERCALLBACK, IID_IMFSampleGrabberSinkCallback, (LPVOID*)pSampleCallback.GetAddressOf());
            }
            if (SUCCEEDED(hr)) {
                hr = pInput->Lock(&pSrc, &cbMaxLength, &cbCurrentLength);
            }
            if (SUCCEEDED(hr)) {
                hr = pSampleCallback->OnProcessSample(guidMajorType, 0, hnsTime, hnsDuration, pSrc, cbCurrentLength);
                pInput->Unlock();
            }
            if (SUCCEEDED(hr)) {
                hr = QueueEvent(MEStreamSinkRequestSample, GUID_NULL, S_OK, NULL);
            }
        }
        LeaveCriticalSection(&m_critSec);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE PlaceMarker(
        /* [in] */ MFSTREAMSINK_MARKER_TYPE eMarkerType,
        /* [in] */ __RPC__in const PROPVARIANT * /*pvarMarkerValue*/,
        /* [in] */ __RPC__in const PROPVARIANT * /*pvarContextValue*/) {
        EnterCriticalSection(&m_critSec);

        HRESULT hr = S_OK;
        if (m_state == State_TypeNotSet)
            hr = MF_E_NOT_INITIALIZED;

        if (SUCCEEDED(hr))
            hr = CheckShutdown();

        if (SUCCEEDED(hr))
        {
            hr = QueueEvent(MEStreamSinkRequestSample, GUID_NULL, S_OK, NULL);
        }

        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"StreamSink::PlaceMarker: HRESULT=%i %s\n", hr, StreamSinkMarkerTypeMap.at(eMarkerType).c_str());
        return hr;
    }

    HRESULT STDMETHODCALLTYPE Flush(void) {
        EnterCriticalSection(&m_critSec);

        HRESULT hr = CheckShutdown();

        if (SUCCEEDED(hr))
        {
        }

        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"StreamSink::Flush: HRESULT=%i\n", hr);
        return hr;
    }

    //IMFMediaEventGenerator
    HRESULT STDMETHODCALLTYPE GetEvent(
        DWORD dwFlags, IMFMediaEvent **ppEvent) {
        // NOTE:
        // GetEvent can block indefinitely, so we don't hold the lock.
        // This requires some juggling with the event queue pointer.

        HRESULT hr = S_OK;

        ComPtr<IMFMediaEventQueue> pQueue;

        {
            EnterCriticalSection(&m_critSec);

            // Check shutdown
            hr = CheckShutdown();

            // Get the pointer to the event queue.
            if (SUCCEEDED(hr))
            {
                pQueue = m_spEventQueue.Get();
            }
            LeaveCriticalSection(&m_critSec);
        }

        // Now get the event.
        if (SUCCEEDED(hr))
        {
            hr = pQueue->GetEvent(dwFlags, ppEvent);
        }
        MediaEventType meType = MEUnknown;
        if (SUCCEEDED(hr) && SUCCEEDED((*ppEvent)->GetType(&meType)) && meType == MEStreamSinkStopped) {
        }
        HRESULT hrStatus = S_OK;
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        if (SUCCEEDED(hr))
            hr = (*ppEvent)->GetStatus(&hrStatus);
        if (SUCCEEDED(hr))
            DPO->printOut(L"StreamSink::GetEvent: HRESULT=%i %s\n", hrStatus, MediaEventTypeMap.at(meType).c_str());
        else
            DPO->printOut(L"StreamSink::GetEvent: HRESULT=%i\n", hr);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE BeginGetEvent(
        IMFAsyncCallback *pCallback, IUnknown *punkState) {
        HRESULT hr = S_OK;

        EnterCriticalSection(&m_critSec);

        hr = CheckShutdown();

        if (SUCCEEDED(hr))
        {
            hr = m_spEventQueue->BeginGetEvent(pCallback, punkState);
        }
        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"StreamSink::BeginGetEvent: HRESULT=%i\n", hr);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE EndGetEvent(
        IMFAsyncResult *pResult, IMFMediaEvent **ppEvent) {
        HRESULT hr = S_OK;

        EnterCriticalSection(&m_critSec);

        hr = CheckShutdown();

        if (SUCCEEDED(hr))
        {
            hr = m_spEventQueue->EndGetEvent(pResult, ppEvent);
        }

        MediaEventType meType = MEUnknown;
        if (SUCCEEDED(hr) && SUCCEEDED((*ppEvent)->GetType(&meType)) && meType == MEStreamSinkStopped) {
        }

        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        HRESULT hrStatus = S_OK;
        if (SUCCEEDED(hr))
            hr = (*ppEvent)->GetStatus(&hrStatus);
        if (SUCCEEDED(hr))
            DPO->printOut(L"StreamSink::EndGetEvent: HRESULT=%i %s\n", hrStatus, MediaEventTypeMap.at(meType).c_str());
        else
            DPO->printOut(L"StreamSink::EndGetEvent: HRESULT=%i\n", hr);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE QueueEvent(
        MediaEventType met, REFGUID guidExtendedType,
        HRESULT hrStatus, const PROPVARIANT *pvValue) {
        HRESULT hr = S_OK;

        EnterCriticalSection(&m_critSec);

        hr = CheckShutdown();

        if (SUCCEEDED(hr))
        {
            hr = m_spEventQueue->QueueEventParamVar(met, guidExtendedType, hrStatus, pvValue);
        }

        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"StreamSink::QueueEvent: HRESULT=%i %s\n", hrStatus, MediaEventTypeMap.at(met).c_str());
        DPO->printOut(L"StreamSink::QueueEvent: HRESULT=%i\n", hr);
        return hr;
    }

    /// IMFMediaTypeHandler methods

    // Check if a media type is supported.
    STDMETHODIMP IsMediaTypeSupported(
        /* [in] */ IMFMediaType *pMediaType,
        /* [out] */ IMFMediaType **ppMediaType)
    {
        if (pMediaType == nullptr)
        {
            return E_INVALIDARG;
        }

        EnterCriticalSection(&m_critSec);

        GUID majorType = GUID_NULL;

        HRESULT hr = CheckShutdown();

        if (SUCCEEDED(hr))
        {
            hr = pMediaType->GetGUID(MF_MT_MAJOR_TYPE, &majorType);
        }

        // First make sure it's video or audio type.
        if (SUCCEEDED(hr))
        {
            if (majorType != MFMediaType_Video && majorType != MFMediaType_Audio)
            {
                hr = MF_E_INVALIDTYPE;
            }
        }

        if (SUCCEEDED(hr) && m_spCurrentType != nullptr)
        {
            GUID guiNewSubtype;
            if (FAILED(pMediaType->GetGUID(MF_MT_SUBTYPE, &guiNewSubtype)) ||
                guiNewSubtype != m_guiCurrentSubtype)
            {
                hr = MF_E_INVALIDTYPE;
            }
        }
        if (ppMediaType)
        {
            *ppMediaType = nullptr;
        }

        if (ppMediaType && SUCCEEDED(hr)) {
            ComPtr<IMFMediaType> pType;
            hr = MFCreateMediaType(ppMediaType);
            if (SUCCEEDED(hr)) {
                hr = m_pParent->GetUnknown(MF_MEDIASINK_PREFERREDTYPE, __uuidof(IMFMediaType), (LPVOID*)&pType);
            }
            if (SUCCEEDED(hr)) {
                hr = pType->LockStore();
            }
            bool bLocked = false;
            if (SUCCEEDED(hr)) {
                bLocked = true;
                UINT32 uiCount;
                UINT32 uiTotal;
                hr = pType->GetCount(&uiTotal);
                for (uiCount = 0; SUCCEEDED(hr) && uiCount < uiTotal; uiCount++) {
                    GUID guid;
                    PROPVARIANT propval;
                    hr = pType->GetItemByIndex(uiCount, &guid, &propval);
                    if (SUCCEEDED(hr) && (guid == MF_MT_FRAME_SIZE || guid == MF_MT_MAJOR_TYPE || guid == MF_MT_PIXEL_ASPECT_RATIO ||
                        guid == MF_MT_ALL_SAMPLES_INDEPENDENT || guid == MF_MT_INTERLACE_MODE || guid == MF_MT_SUBTYPE)) {
                        hr = (*ppMediaType)->SetItem(guid, propval);
                        PropVariantClear(&propval);
                    }
                }
            }
            if (bLocked) {
                hr = pType->UnlockStore();
            }
        }
        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"StreamSink::IsMediaTypeSupported: HRESULT=%i\n", hr);
        return hr;
    }


    // Return the number of preferred media types.
    STDMETHODIMP GetMediaTypeCount(DWORD *pdwTypeCount)
    {
        if (pdwTypeCount == nullptr)
        {
            return E_INVALIDARG;
        }

        EnterCriticalSection(&m_critSec);

        HRESULT hr = CheckShutdown();

        if (SUCCEEDED(hr))
        {
            // We've got only one media type
            *pdwTypeCount = 1;
        }

        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"StreamSink::GetMediaTypeCount: HRESULT=%i\n", hr);
        return hr;
    }


    // Return a preferred media type by index.
    STDMETHODIMP GetMediaTypeByIndex(
        /* [in] */ DWORD dwIndex,
        /* [out] */ IMFMediaType **ppType)
    {
        if (ppType == NULL) {
            return E_INVALIDARG;
        }

        EnterCriticalSection(&m_critSec);

        HRESULT hr = CheckShutdown();

        if (dwIndex > 0)
        {
            hr = MF_E_NO_MORE_TYPES;
        } else {
            //return preferred type based on media capture library 6 elements preferred preview type
            //hr = m_spCurrentType.CopyTo(ppType);
            if (SUCCEEDED(hr)) {
                ComPtr<IMFMediaType> pType;
                hr = MFCreateMediaType(ppType);
                if (SUCCEEDED(hr)) {
                    hr = m_pParent->GetUnknown(MF_MEDIASINK_PREFERREDTYPE, __uuidof(IMFMediaType), (LPVOID*)&pType);
                }
                if (SUCCEEDED(hr)) {
                    hr = pType->LockStore();
                }
                bool bLocked = false;
                if (SUCCEEDED(hr)) {
                    bLocked = true;
                    UINT32 uiCount;
                    UINT32 uiTotal;
                    hr = pType->GetCount(&uiTotal);
                    for (uiCount = 0; SUCCEEDED(hr) && uiCount < uiTotal; uiCount++) {
                        GUID guid;
                        PROPVARIANT propval;
                        hr = pType->GetItemByIndex(uiCount, &guid, &propval);
                        if (SUCCEEDED(hr) && (guid == MF_MT_FRAME_SIZE || guid == MF_MT_MAJOR_TYPE || guid == MF_MT_PIXEL_ASPECT_RATIO ||
                            guid == MF_MT_ALL_SAMPLES_INDEPENDENT || guid == MF_MT_INTERLACE_MODE || guid == MF_MT_SUBTYPE)) {
                            hr = (*ppType)->SetItem(guid, propval);
                            PropVariantClear(&propval);
                        }
                    }
                }
                if (bLocked) {
                    hr = pType->UnlockStore();
                }
            }
        }

        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"StreamSink::GetMediaTypeByIndex: HRESULT=%i\n", hr);
        return hr;
    }


    // Set the current media type.
    STDMETHODIMP SetCurrentMediaType(IMFMediaType *pMediaType)
    {
        if (pMediaType == NULL) {
            return E_INVALIDARG;
        }
        EnterCriticalSection(&m_critSec);

        HRESULT hr = S_OK;
        if (m_state != State_TypeNotSet && m_state != State_Ready)
            hr = MF_E_INVALIDREQUEST;
        if (SUCCEEDED(hr))
            hr = CheckShutdown();

        // We don't allow format changes after streaming starts.

        // We set media type already
        if (m_state >= State_Ready)
        {
            if (SUCCEEDED(hr))
            {
                hr = IsMediaTypeSupported(pMediaType, NULL);
            }
        }

        if (SUCCEEDED(hr))
        {
            GUID guiMajorType;
            pMediaType->GetMajorType(&guiMajorType);

            hr = MFCreateMediaType(m_spCurrentType.ReleaseAndGetAddressOf());
            if (SUCCEEDED(hr))
            {
                hr = pMediaType->CopyAllItems(m_spCurrentType.Get());
            }
            if (SUCCEEDED(hr))
            {
                hr = m_spCurrentType->GetGUID(MF_MT_SUBTYPE, &m_guiCurrentSubtype);
            }
            if (SUCCEEDED(hr)) {
                hr = MFGetAttributeSize(m_spCurrentType.Get(), MF_MT_FRAME_SIZE, &m_imageWidthInPixels, &m_imageHeightInPixels);
            }
            if (SUCCEEDED(hr))
            {
                m_state = State_Ready;
            }
        }

        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"StreamSink::SetCurrentMediaType: HRESULT=%i\n", hr);
        return hr;
    }

    // Return the current media type, if any.
    STDMETHODIMP GetCurrentMediaType(IMFMediaType **ppMediaType)
    {
        if (ppMediaType == NULL) {
            return E_INVALIDARG;
        }

        EnterCriticalSection(&m_critSec);

        HRESULT hr = CheckShutdown();

        if (SUCCEEDED(hr)) {
            if (m_spCurrentType == nullptr) {
                hr = MF_E_NOT_INITIALIZED;
            }
        }

        if (SUCCEEDED(hr)) {
            hr = m_spCurrentType.CopyTo(ppMediaType);
        }

        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"StreamSink::GetCurrentMediaType: HRESULT=%i\n", hr);
        return hr;
    }


    // Return the major type GUID.
    STDMETHODIMP GetMajorType(GUID *pguidMajorType)
    {
        HRESULT hr;
        if (pguidMajorType == nullptr) {
            return E_INVALIDARG;
        }

        ComPtr<IMFMediaType> pType;
        hr = m_pParent->GetUnknown(MF_MEDIASINK_PREFERREDTYPE, __uuidof(IMFMediaType), (LPVOID*)&pType);
        if (SUCCEEDED(hr)) {
            hr = pType->GetMajorType(pguidMajorType);
        }
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"StreamSink::GetMajorType: HRESULT=%i\n", hr);
        return hr;
    }
private:
#ifdef HAVE_WINRT
    EventRegistrationToken m_token;
#else
    bool m_bConnected;
#endif

    bool m_IsShutdown;                // Flag to indicate if Shutdown() method was called.
    CRITICAL_SECTION m_critSec;
#ifndef HAVE_WINRT
    long m_cRef;
#endif
    IMFAttributes*        m_pParent;
    ComPtr<IMFMediaType>        m_spCurrentType;
    ComPtr<IMFMediaEventQueue>  m_spEventQueue;              // Event queue

    ComPtr<IUnknown>            m_spFTM;
    State                       m_state;
    bool                        m_fGetStartTimeFromSample;
    bool                        m_fWaitingForFirstSample;
    MFTIME                      m_StartTime;                 // Presentation time when the clock started.
    GUID                        m_guiCurrentSubtype;
    UINT32                      m_imageWidthInPixels;
    UINT32                      m_imageHeightInPixels;
};

// Notes:
//
// The List class template implements a simple double-linked list.
// It uses STL's copy semantics.

// There are two versions of the Clear() method:
//  Clear(void) clears the list w/out cleaning up the object.
//  Clear(FN fn) takes a functor object that releases the objects, if they need cleanup.

// The List class supports enumeration. Example of usage:
//
// List<T>::POSIITON pos = list.GetFrontPosition();
// while (pos != list.GetEndPosition())
// {
//     T item;
//     hr = list.GetItemPos(&item);
//     pos = list.Next(pos);
// }

// The ComPtrList class template derives from List<> and implements a list of COM pointers.

template <class T>
struct NoOp
{
    void operator()(T& /*t*/)
    {
    }
};

template <class T>
class List
{
protected:

    // Nodes in the linked list
    struct Node
    {
        Node *prev;
        Node *next;
        T    item;

        Node() : prev(nullptr), next(nullptr)
        {
        }

        Node(T item) : prev(nullptr), next(nullptr)
        {
            this->item = item;
        }

        T Item() const { return item; }
    };

public:

    // Object for enumerating the list.
    class POSITION
    {
        friend class List<T>;

    public:
        POSITION() : pNode(nullptr)
        {
        }

        bool operator==(const POSITION &p) const
        {
            return pNode == p.pNode;
        }

        bool operator!=(const POSITION &p) const
        {
            return pNode != p.pNode;
        }

    private:
        const Node *pNode;

        POSITION(Node *p) : pNode(p)
        {
        }
    };

protected:
    Node    m_anchor;  // Anchor node for the linked list.
    DWORD   m_count;   // Number of items in the list.

    Node* Front() const
    {
        return m_anchor.next;
    }

    Node* Back() const
    {
        return m_anchor.prev;
    }

    virtual HRESULT InsertAfter(T item, Node *pBefore)
    {
        if (pBefore == nullptr)
        {
            return E_POINTER;
        }

        Node *pNode = new Node(item);
        if (pNode == nullptr)
        {
            return E_OUTOFMEMORY;
        }

        Node *pAfter = pBefore->next;

        pBefore->next = pNode;
        pAfter->prev = pNode;

        pNode->prev = pBefore;
        pNode->next = pAfter;

        m_count++;

        return S_OK;
    }

    virtual HRESULT GetItem(const Node *pNode, T* ppItem)
    {
        if (pNode == nullptr || ppItem == nullptr)
        {
            return E_POINTER;
        }

        *ppItem = pNode->item;
        return S_OK;
    }

    // RemoveItem:
    // Removes a node and optionally returns the item.
    // ppItem can be nullptr.
    virtual HRESULT RemoveItem(Node *pNode, T *ppItem)
    {
        if (pNode == nullptr)
        {
            return E_POINTER;
        }

        assert(pNode != &m_anchor); // We should never try to remove the anchor node.
        if (pNode == &m_anchor)
        {
            return E_INVALIDARG;
        }


        T item;

        // The next node's previous is this node's previous.
        pNode->next->prev = pNode->prev;

        // The previous node's next is this node's next.
        pNode->prev->next = pNode->next;

        item = pNode->item;
        delete pNode;

        m_count--;

        if (ppItem)
        {
            *ppItem = item;
        }

        return S_OK;
    }

public:

    List()
    {
        m_anchor.next = &m_anchor;
        m_anchor.prev = &m_anchor;

        m_count = 0;
    }

    virtual ~List()
    {
        Clear();
    }

    // Insertion functions
    HRESULT InsertBack(T item)
    {
        return InsertAfter(item, m_anchor.prev);
    }


    HRESULT InsertFront(T item)
    {
        return InsertAfter(item, &m_anchor);
    }

    HRESULT InsertPos(POSITION pos, T item)
    {
        if (pos.pNode == nullptr)
        {
            return InsertBack(item);
        }

        return InsertAfter(item, pos.pNode->prev);
    }

    // RemoveBack: Removes the tail of the list and returns the value.
    // ppItem can be nullptr if you don't want the item back. (But the method does not release the item.)
    HRESULT RemoveBack(T *ppItem)
    {
        if (IsEmpty())
        {
            return E_FAIL;
        }
        else
        {
            return RemoveItem(Back(), ppItem);
        }
    }

    // RemoveFront: Removes the head of the list and returns the value.
    // ppItem can be nullptr if you don't want the item back. (But the method does not release the item.)
    HRESULT RemoveFront(T *ppItem)
    {
        if (IsEmpty())
        {
            return E_FAIL;
        }
        else
        {
            return RemoveItem(Front(), ppItem);
        }
    }

    // GetBack: Gets the tail item.
    HRESULT GetBack(T *ppItem)
    {
        if (IsEmpty())
        {
            return E_FAIL;
        }
        else
        {
            return GetItem(Back(), ppItem);
        }
    }

    // GetFront: Gets the front item.
    HRESULT GetFront(T *ppItem)
    {
        if (IsEmpty())
        {
            return E_FAIL;
        }
        else
        {
            return GetItem(Front(), ppItem);
        }
    }


    // GetCount: Returns the number of items in the list.
    DWORD GetCount() const { return m_count; }

    bool IsEmpty() const
    {
        return (GetCount() == 0);
    }

    // Clear: Takes a functor object whose operator()
    // frees the object on the list.
    template <class FN>
    void Clear(FN& clear_fn)
    {
        Node *n = m_anchor.next;

        // Delete the nodes
        while (n != &m_anchor)
        {
            clear_fn(n->item);

            Node *tmp = n->next;
            delete n;
            n = tmp;
        }

        // Reset the anchor to point at itself
        m_anchor.next = &m_anchor;
        m_anchor.prev = &m_anchor;

        m_count = 0;
    }

    // Clear: Clears the list. (Does not delete or release the list items.)
    virtual void Clear()
    {
        NoOp<T> clearOp;
        Clear<>(clearOp);
    }


    // Enumerator functions

    POSITION FrontPosition()
    {
        if (IsEmpty())
        {
            return POSITION(nullptr);
        }
        else
        {
            return POSITION(Front());
        }
    }

    POSITION EndPosition() const
    {
        return POSITION();
    }

    HRESULT GetItemPos(POSITION pos, T *ppItem)
    {
        if (pos.pNode)
        {
            return GetItem(pos.pNode, ppItem);
        }
        else
        {
            return E_FAIL;
        }
    }

    POSITION Next(const POSITION pos)
    {
        if (pos.pNode && (pos.pNode->next != &m_anchor))
        {
            return POSITION(pos.pNode->next);
        }
        else
        {
            return POSITION(nullptr);
        }
    }

    // Remove an item at a position.
    // The item is returns in ppItem, unless ppItem is nullptr.
    // NOTE: This method invalidates the POSITION object.
    HRESULT Remove(POSITION& pos, T *ppItem)
    {
        if (pos.pNode)
        {
            // Remove const-ness temporarily...
            Node *pNode = const_cast<Node*>(pos.pNode);

            pos = POSITION();

            return RemoveItem(pNode, ppItem);
        }
        else
        {
            return E_INVALIDARG;
        }
    }

};



// Typical functors for Clear method.

// ComAutoRelease: Releases COM pointers.
// MemDelete: Deletes pointers to new'd memory.

class ComAutoRelease
{
public:
    void operator()(IUnknown *p)
    {
        if (p)
        {
            p->Release();
        }
    }
};

class MemDelete
{
public:
    void operator()(void *p)
    {
        if (p)
        {
            delete p;
        }
    }
};


// ComPtrList class
// Derived class that makes it safer to store COM pointers in the List<> class.
// It automatically AddRef's the pointers that are inserted onto the list
// (unless the insertion method fails).
//
// T must be a COM interface type.
// example: ComPtrList<IUnknown>
//
// NULLABLE: If true, client can insert nullptr pointers. This means GetItem can
// succeed but return a nullptr pointer. By default, the list does not allow nullptr
// pointers.

template <class T, bool NULLABLE = FALSE>
class ComPtrList : public List<T*>
{
public:

    typedef T* Ptr;

    void Clear()
    {
        ComAutoRelease car;
        List<Ptr>::Clear(car);
    }

    ~ComPtrList()
    {
        Clear();
    }

protected:
    HRESULT InsertAfter(Ptr item, Node *pBefore)
    {
        // Do not allow nullptr item pointers unless NULLABLE is true.
        if (item == nullptr && !NULLABLE)
        {
            return E_POINTER;
        }

        if (item)
        {
            item->AddRef();
        }

        HRESULT hr = List<Ptr>::InsertAfter(item, pBefore);
        if (FAILED(hr) && item != nullptr)
        {
            item->Release();
        }
        return hr;
    }

    HRESULT GetItem(const Node *pNode, Ptr* ppItem)
    {
        Ptr pItem = nullptr;

        // The base class gives us the pointer without AddRef'ing it.
        // If we return the pointer to the caller, we must AddRef().
        HRESULT hr = List<Ptr>::GetItem(pNode, &pItem);
        if (SUCCEEDED(hr))
        {
            assert(pItem || NULLABLE);
            if (pItem)
            {
                *ppItem = pItem;
                (*ppItem)->AddRef();
            }
        }
        return hr;
    }

    HRESULT RemoveItem(Node *pNode, Ptr *ppItem)
    {
        // ppItem can be nullptr, but we need to get the
        // item so that we can release it.

        // If ppItem is not nullptr, we will AddRef it on the way out.

        Ptr pItem = nullptr;

        HRESULT hr = List<Ptr>::RemoveItem(pNode, &pItem);

        if (SUCCEEDED(hr))
        {
            assert(pItem || NULLABLE);
            if (ppItem && pItem)
            {
                *ppItem = pItem;
                (*ppItem)->AddRef();
            }

            if (pItem)
            {
                pItem->Release();
                pItem = nullptr;
            }
        }

        return hr;
    }
};

/* Be sure to declare webcam device capability in manifest
  For better media capture support, add the following snippet with correct module name to the project manifest
    (videoio needs DLL activation class factoryentry points):
  <Extensions>
    <Extension Category="windows.activatableClass.inProcessServer">
      <InProcessServer>
        <Path>modulename</Path>
        <ActivatableClass ActivatableClassId="cv.MediaSink" ThreadingModel="both" />
      </InProcessServer>
    </Extension>
  </Extensions>*/

extern const __declspec(selectany) WCHAR RuntimeClass_CV_MediaSink[] = L"cv.MediaSink";

class MediaSink :
#ifdef HAVE_WINRT
    public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags< Microsoft::WRL::RuntimeClassType::WinRtClassicComMix >,
    Microsoft::WRL::Implements<ABI::Windows::Media::IMediaExtension>,
    IMFMediaSink,
    IMFClockStateSink,
    FtmBase,
    CBaseAttributes<>>
#else
    public IMFMediaSink, public IMFClockStateSink, public CBaseAttributes<>
#endif
{
#ifdef HAVE_WINRT
    InspectableClass(RuntimeClass_CV_MediaSink, BaseTrust)
public:
#else
public:
    ULONG AddRef()
    {
        return InterlockedIncrement(&m_cRef);
    }
    ULONG Release()
    {
        ULONG cRef = InterlockedDecrement(&m_cRef);
        if (cRef == 0)
        {
            delete this;
        }
        return cRef;
    }
    STDMETHOD(QueryInterface)(REFIID riid, _Outptr_result_nullonfailure_ void **ppv)
    {
        if (ppv == nullptr) {
            return E_POINTER;
        }
        (*ppv) = nullptr;
        HRESULT hr = S_OK;
        if (riid == IID_IUnknown ||
            riid == IID_IMFMediaSink) {
            (*ppv) = static_cast<IMFMediaSink*>(this);
            AddRef();
        } else if (riid == IID_IMFClockStateSink) {
            (*ppv) = static_cast<IMFClockStateSink*>(this);
            AddRef();
        } else if (riid == IID_IMFAttributes) {
            (*ppv) = static_cast<IMFAttributes*>(this);
            AddRef();
        } else {
            hr = E_NOINTERFACE;
        }

        return hr;
    }
#endif
    MediaSink() : m_IsShutdown(false), m_llStartTime(0) {
        CBaseAttributes<>::Initialize(0U);
        InitializeCriticalSectionEx(&m_critSec, 3000, 0);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"MediaSink::MediaSink\n");
    }

    virtual ~MediaSink() {
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"MediaSink::~MediaSink\n");
        DeleteCriticalSection(&m_critSec);
        assert(m_IsShutdown);
    }
    HRESULT CheckShutdown() const
    {
        if (m_IsShutdown)
        {
            return MF_E_SHUTDOWN;
        }
        else
        {
            return S_OK;
        }
    }
#ifdef HAVE_WINRT
    STDMETHODIMP SetProperties(ABI::Windows::Foundation::Collections::IPropertySet *pConfiguration)
    {
        HRESULT hr = S_OK;
        if (pConfiguration) {
            Microsoft::WRL::ComPtr<IInspectable> spInsp;
            Microsoft::WRL::ComPtr<ABI::Windows::Foundation::Collections::IMap<HSTRING, IInspectable *>> spSetting;
            Microsoft::WRL::ComPtr<ABI::Windows::Foundation::IPropertyValue> spPropVal;
            ComPtr<ABI::Windows::Media::MediaProperties::IMediaEncodingProperties> pMedEncProps;
            UINT32 uiType = ABI::Windows::Media::Capture::MediaStreamType_VideoPreview;

            hr = pConfiguration->QueryInterface(IID_PPV_ARGS(&spSetting));
            if (FAILED(hr)) {
                hr = E_FAIL;
            }

            if (SUCCEEDED(hr)) {
                hr = spSetting->Lookup(HStringReference(MF_PROP_SAMPLEGRABBERCALLBACK).Get(), spInsp.ReleaseAndGetAddressOf());
                if (FAILED(hr)) {
                    hr = E_INVALIDARG;
                }
                if (SUCCEEDED(hr)) {
                    hr = SetUnknown(MF_MEDIASINK_SAMPLEGRABBERCALLBACK, spInsp.Get());
                }
            }
            if (SUCCEEDED(hr)) {
                hr = spSetting->Lookup(HStringReference(MF_PROP_VIDTYPE).Get(), spInsp.ReleaseAndGetAddressOf());
                if (FAILED(hr)) {
                    hr = E_INVALIDARG;
                }
                if (SUCCEEDED(hr)) {
                    if (SUCCEEDED(hr = spInsp.As(&spPropVal))) {
                        hr = spPropVal->GetUInt32(&uiType);
                    }
                }
            }
            if (SUCCEEDED(hr)) {
                hr = spSetting->Lookup(HStringReference(MF_PROP_VIDENCPROPS).Get(), spInsp.ReleaseAndGetAddressOf());
                if (FAILED(hr)) {
                    hr = E_INVALIDARG;
                }
                if (SUCCEEDED(hr)) {
                    hr = spInsp.As(&pMedEncProps);
                }
            }
            if (SUCCEEDED(hr)) {
                hr = SetMediaStreamProperties((ABI::Windows::Media::Capture::MediaStreamType)uiType, pMedEncProps.Get());
            }
        }

        return hr;
    }
    static DWORD GetStreamId(ABI::Windows::Media::Capture::MediaStreamType mediaStreamType)
    {
        return 3 - mediaStreamType;
    }
    static HRESULT AddAttribute(_In_ GUID guidKey, _In_ ABI::Windows::Foundation::IPropertyValue *pValue, _In_ IMFAttributes* pAttr)
    {
        HRESULT hr = S_OK;
        PROPVARIANT var;
        ABI::Windows::Foundation::PropertyType type;
        hr = pValue->get_Type(&type);
        ZeroMemory(&var, sizeof(var));

        if (SUCCEEDED(hr))
        {
            switch (type)
            {
            case ABI::Windows::Foundation::PropertyType_UInt8Array:
            {
                                                                      UINT32 cbBlob;
                                                                      BYTE *pbBlog = nullptr;
                                                                      hr = pValue->GetUInt8Array(&cbBlob, &pbBlog);
                                                                      if (SUCCEEDED(hr))
                                                                      {
                                                                          if (pbBlog == nullptr)
                                                                          {
                                                                              hr = E_INVALIDARG;
                                                                          }
                                                                          else
                                                                          {
                                                                              hr = pAttr->SetBlob(guidKey, pbBlog, cbBlob);
                                                                          }
                                                                      }
                                                                      CoTaskMemFree(pbBlog);
            }
                break;

            case ABI::Windows::Foundation::PropertyType_Double:
            {
                                                                  DOUBLE value;
                                                                  hr = pValue->GetDouble(&value);
                                                                  if (SUCCEEDED(hr))
                                                                  {
                                                                      hr = pAttr->SetDouble(guidKey, value);
                                                                  }
            }
                break;

            case ABI::Windows::Foundation::PropertyType_Guid:
            {
                                                                GUID value;
                                                                hr = pValue->GetGuid(&value);
                                                                if (SUCCEEDED(hr))
                                                                {
                                                                    hr = pAttr->SetGUID(guidKey, value);
                                                                }
            }
                break;

            case ABI::Windows::Foundation::PropertyType_String:
            {
                                                                  HSTRING value;
                                                                  hr = pValue->GetString(&value);
                                                                  if (SUCCEEDED(hr))
                                                                  {
                                                                      UINT32 len = 0;
                                                                      LPCWSTR szValue = WindowsGetStringRawBuffer(value, &len);
                                                                      hr = pAttr->SetString(guidKey, szValue);
                                                                      WindowsDeleteString(value);
                                                                  }
            }
                break;

            case ABI::Windows::Foundation::PropertyType_UInt32:
            {
                                                                  UINT32 value;
                                                                  hr = pValue->GetUInt32(&value);
                                                                  if (SUCCEEDED(hr))
                                                                  {
                                                                      pAttr->SetUINT32(guidKey, value);
                                                                  }
            }
                break;

            case ABI::Windows::Foundation::PropertyType_UInt64:
            {
                                                                  UINT64 value;
                                                                  hr = pValue->GetUInt64(&value);
                                                                  if (SUCCEEDED(hr))
                                                                  {
                                                                      hr = pAttr->SetUINT64(guidKey, value);
                                                                  }
            }
                break;

            case ABI::Windows::Foundation::PropertyType_Inspectable:
            {
                                                                       ComPtr<IInspectable> value;
                                                                       hr = TYPE_E_TYPEMISMATCH;
                                                                       if (SUCCEEDED(hr))
                                                                       {
                                                                           pAttr->SetUnknown(guidKey, value.Get());
                                                                       }
            }
                break;

                // ignore unknown values
            }
        }

        return hr;
    }
    static HRESULT ConvertPropertiesToMediaType(_In_ ABI::Windows::Media::MediaProperties::IMediaEncodingProperties *pMEP, _Outptr_ IMFMediaType **ppMT)
    {
        HRESULT hr = S_OK;
        ComPtr<IMFMediaType> spMT;
        ComPtr<ABI::Windows::Foundation::Collections::IMap<GUID, IInspectable*>> spMap;
        ComPtr<ABI::Windows::Foundation::Collections::IIterable<ABI::Windows::Foundation::Collections::IKeyValuePair<GUID, IInspectable*>*>> spIterable;
        ComPtr<ABI::Windows::Foundation::Collections::IIterator<ABI::Windows::Foundation::Collections::IKeyValuePair<GUID, IInspectable*>*>> spIterator;

        if (pMEP == nullptr || ppMT == nullptr)
        {
            return E_INVALIDARG;
        }
        *ppMT = nullptr;

        hr = pMEP->get_Properties(&spMap);

        if (SUCCEEDED(hr))
        {
            hr = spMap.As(&spIterable);
        }
        if (SUCCEEDED(hr))
        {
            hr = spIterable->First(&spIterator);
        }
        if (SUCCEEDED(hr))
        {
            MFCreateMediaType(spMT.ReleaseAndGetAddressOf());
        }

        boolean hasCurrent = false;
        if (SUCCEEDED(hr))
        {
            hr = spIterator->get_HasCurrent(&hasCurrent);
        }

        while (hasCurrent)
        {
            ComPtr<ABI::Windows::Foundation::Collections::IKeyValuePair<GUID, IInspectable*> > spKeyValuePair;
            ComPtr<IInspectable> spValue;
            ComPtr<ABI::Windows::Foundation::IPropertyValue> spPropValue;
            GUID guidKey;

            hr = spIterator->get_Current(&spKeyValuePair);
            if (FAILED(hr))
            {
                break;
            }
            hr = spKeyValuePair->get_Key(&guidKey);
            if (FAILED(hr))
            {
                break;
            }
            hr = spKeyValuePair->get_Value(&spValue);
            if (FAILED(hr))
            {
                break;
            }
            hr = spValue.As(&spPropValue);
            if (FAILED(hr))
            {
                break;
            }
            hr = AddAttribute(guidKey, spPropValue.Get(), spMT.Get());
            if (FAILED(hr))
            {
                break;
            }

            hr = spIterator->MoveNext(&hasCurrent);
            if (FAILED(hr))
            {
                break;
            }
        }


        if (SUCCEEDED(hr))
        {
            ComPtr<IInspectable> spValue;
            ComPtr<ABI::Windows::Foundation::IPropertyValue> spPropValue;
            GUID guiMajorType;

            hr = spMap->Lookup(MF_MT_MAJOR_TYPE, spValue.GetAddressOf());

            if (SUCCEEDED(hr))
            {
                hr = spValue.As(&spPropValue);
            }
            if (SUCCEEDED(hr))
            {
                hr = spPropValue->GetGuid(&guiMajorType);
            }
            if (SUCCEEDED(hr))
            {
                if (guiMajorType != MFMediaType_Video && guiMajorType != MFMediaType_Audio)
                {
                    hr = E_UNEXPECTED;
                }
            }
        }

        if (SUCCEEDED(hr))
        {
            *ppMT = spMT.Detach();
        }

        return hr;
    }
    HRESULT SetMediaStreamProperties(
        ABI::Windows::Media::Capture::MediaStreamType MediaStreamType,
        _In_opt_ ABI::Windows::Media::MediaProperties::IMediaEncodingProperties *mediaEncodingProperties)
    {
        HRESULT hr = S_OK;
        ComPtr<IMFMediaType> spMediaType;

        if (MediaStreamType != ABI::Windows::Media::Capture::MediaStreamType_VideoPreview &&
            MediaStreamType != ABI::Windows::Media::Capture::MediaStreamType_VideoRecord &&
            MediaStreamType != ABI::Windows::Media::Capture::MediaStreamType_Audio)
        {
            return E_INVALIDARG;
        }

        RemoveStreamSink(GetStreamId(MediaStreamType));

        if (mediaEncodingProperties != nullptr)
        {
            ComPtr<IMFStreamSink> spStreamSink;
            hr = ConvertPropertiesToMediaType(mediaEncodingProperties, &spMediaType);
            if (SUCCEEDED(hr))
            {
                hr = AddStreamSink(GetStreamId(MediaStreamType), nullptr, spStreamSink.GetAddressOf());
            }
            if (SUCCEEDED(hr)) {
                hr = SetUnknown(MF_MEDIASINK_PREFERREDTYPE, spMediaType.Detach());
            }
        }

        return hr;
    }
#endif
    //IMFMediaSink
    HRESULT STDMETHODCALLTYPE GetCharacteristics(
        /* [out] */ __RPC__out DWORD *pdwCharacteristics) {
        HRESULT hr;
        if (pdwCharacteristics == NULL) return E_INVALIDARG;
        EnterCriticalSection(&m_critSec);
        if (SUCCEEDED(hr = CheckShutdown())) {
            *pdwCharacteristics = MEDIASINK_FIXED_STREAMS;
        }
        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"MediaSink::GetCharacteristics: HRESULT=%i\n", hr);
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE AddStreamSink(
        DWORD dwStreamSinkIdentifier, IMFMediaType * /*pMediaType*/, IMFStreamSink **ppStreamSink) {
        ComPtr<IMFStreamSink> spMFStream;
        ComPtr<ICustomStreamSink> pStream;
        EnterCriticalSection(&m_critSec);
        HRESULT hr = CheckShutdown();

        if (SUCCEEDED(hr))
        {
            hr = GetStreamSinkById(dwStreamSinkIdentifier, &spMFStream);
        }

        if (SUCCEEDED(hr))
        {
            hr = MF_E_STREAMSINK_EXISTS;
        }
        else
        {
            hr = S_OK;
        }

        if (SUCCEEDED(hr))
        {
#ifdef HAVE_WINRT
            pStream = Microsoft::WRL::Make<StreamSink>();
            if (pStream == nullptr) {
                hr = E_OUTOFMEMORY;
            }
            if (SUCCEEDED(hr))
                hr = pStream.As<IMFStreamSink>(&spMFStream);
#else
            StreamSink* pSink = new StreamSink();
            if (pSink) {
                hr = pSink->QueryInterface(IID_IMFStreamSink, (void**)spMFStream.GetAddressOf());
                if (SUCCEEDED(hr)) {
                    hr = spMFStream.As(&pStream);
                }
                if (FAILED(hr)) delete pSink;
            }
#endif
        }

        // Initialize the stream.
        ComPtr<IMFAttributes> pAttr;
        if (SUCCEEDED(hr)) {
            hr = pStream.As(&pAttr);
        }
        if (SUCCEEDED(hr)) {
            hr = pAttr->SetUINT32(MF_STREAMSINK_ID, dwStreamSinkIdentifier);
            if (SUCCEEDED(hr)) {
                hr = pAttr->SetUnknown(MF_STREAMSINK_MEDIASINKINTERFACE, (IMFMediaSink*)this);
            }
        }
        if (SUCCEEDED(hr)) {
            hr = pStream->Initialize();
        }

        if (SUCCEEDED(hr))
        {
            ComPtrList<IMFStreamSink>::POSITION pos = m_streams.FrontPosition();
            ComPtrList<IMFStreamSink>::POSITION posEnd = m_streams.EndPosition();

            // Insert in proper position
            for (; pos != posEnd; pos = m_streams.Next(pos))
            {
                DWORD dwCurrId;
                ComPtr<IMFStreamSink> spCurr;
                hr = m_streams.GetItemPos(pos, &spCurr);
                if (FAILED(hr))
                {
                    break;
                }
                hr = spCurr->GetIdentifier(&dwCurrId);
                if (FAILED(hr))
                {
                    break;
                }

                if (dwCurrId > dwStreamSinkIdentifier)
                {
                    break;
                }
            }

            if (SUCCEEDED(hr))
            {
                hr = m_streams.InsertPos(pos, spMFStream.Get());
            }
        }

        if (SUCCEEDED(hr))
        {
            *ppStreamSink = spMFStream.Detach();
        }
        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"MediaSink::AddStreamSink: HRESULT=%i\n", hr);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE RemoveStreamSink(DWORD dwStreamSinkIdentifier) {
        EnterCriticalSection(&m_critSec);
        HRESULT hr = CheckShutdown();
        ComPtrList<IMFStreamSink>::POSITION pos = m_streams.FrontPosition();
        ComPtrList<IMFStreamSink>::POSITION endPos = m_streams.EndPosition();
        ComPtr<IMFStreamSink> spStream;

        if (SUCCEEDED(hr))
        {
            for (; pos != endPos; pos = m_streams.Next(pos))
            {
                hr = m_streams.GetItemPos(pos, &spStream);
                DWORD dwId;

                if (FAILED(hr))
                {
                    break;
                }

                hr = spStream->GetIdentifier(&dwId);
                if (FAILED(hr) || dwId == dwStreamSinkIdentifier)
                {
                    break;
                }
            }

            if (pos == endPos)
            {
                hr = MF_E_INVALIDSTREAMNUMBER;
            }
        }

        if (SUCCEEDED(hr))
        {
            hr = m_streams.Remove(pos, nullptr);
            static_cast<StreamSink *>(spStream.Get())->Shutdown();
        }
        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"MediaSink::RemoveStreamSink: HRESULT=%i\n", hr);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE GetStreamSinkCount(DWORD *pStreamSinkCount) {
        if (pStreamSinkCount == NULL)
        {
            return E_INVALIDARG;
        }

        EnterCriticalSection(&m_critSec);

        HRESULT hr = CheckShutdown();

        if (SUCCEEDED(hr))
        {
            *pStreamSinkCount = m_streams.GetCount();
        }

        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"MediaSink::GetStreamSinkCount: HRESULT=%i\n", hr);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE GetStreamSinkByIndex(
        DWORD dwIndex, IMFStreamSink **ppStreamSink) {
        if (ppStreamSink == NULL)
        {
            return E_INVALIDARG;
        }

        ComPtr<IMFStreamSink> spStream;
        EnterCriticalSection(&m_critSec);
        DWORD cStreams = m_streams.GetCount();

        if (dwIndex >= cStreams)
        {
            return MF_E_INVALIDINDEX;
        }

        HRESULT hr = CheckShutdown();

        if (SUCCEEDED(hr))
        {
            ComPtrList<IMFStreamSink>::POSITION pos = m_streams.FrontPosition();
            ComPtrList<IMFStreamSink>::POSITION endPos = m_streams.EndPosition();
            DWORD dwCurrent = 0;

            for (; pos != endPos && dwCurrent < dwIndex; pos = m_streams.Next(pos), ++dwCurrent)
            {
                // Just move to proper position
            }

            if (pos == endPos)
            {
                hr = MF_E_UNEXPECTED;
            }
            else
            {
                hr = m_streams.GetItemPos(pos, &spStream);
            }
        }

        if (SUCCEEDED(hr))
        {
            *ppStreamSink = spStream.Detach();
        }
        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"MediaSink::GetStreamSinkByIndex: HRESULT=%i\n", hr);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE GetStreamSinkById(
        DWORD dwStreamSinkIdentifier, IMFStreamSink **ppStreamSink) {
        if (ppStreamSink == NULL)
        {
            return E_INVALIDARG;
        }

        EnterCriticalSection(&m_critSec);
        HRESULT hr = CheckShutdown();
        ComPtr<IMFStreamSink> spResult;

        if (SUCCEEDED(hr))
        {
            ComPtrList<IMFStreamSink>::POSITION pos = m_streams.FrontPosition();
            ComPtrList<IMFStreamSink>::POSITION endPos = m_streams.EndPosition();

            for (; pos != endPos; pos = m_streams.Next(pos))
            {
                ComPtr<IMFStreamSink> spStream;
                hr = m_streams.GetItemPos(pos, &spStream);
                DWORD dwId;

                if (FAILED(hr))
                {
                    break;
                }

                hr = spStream->GetIdentifier(&dwId);
                if (FAILED(hr))
                {
                    break;
                }
                else if (dwId == dwStreamSinkIdentifier)
                {
                    spResult = spStream;
                    break;
                }
            }

            if (pos == endPos)
            {
                hr = MF_E_INVALIDSTREAMNUMBER;
            }
        }

        if (SUCCEEDED(hr))
        {
            assert(spResult);
            *ppStreamSink = spResult.Detach();
        }
        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"MediaSink::GetStreamSinkById: HRESULT=%i\n", hr);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE SetPresentationClock(
        IMFPresentationClock *pPresentationClock) {
        EnterCriticalSection(&m_critSec);

        HRESULT hr = CheckShutdown();

        // If we already have a clock, remove ourselves from that clock's
        // state notifications.
        if (SUCCEEDED(hr)) {
            if (m_spClock) {
                hr = m_spClock->RemoveClockStateSink(this);
            }
        }

        // Register ourselves to get state notifications from the new clock.
        if (SUCCEEDED(hr)) {
            if (pPresentationClock) {
                hr = pPresentationClock->AddClockStateSink(this);
            }
        }

        ComPtr<IMFSampleGrabberSinkCallback> pSampleCallback;
        if (SUCCEEDED(hr)) {
            // Release the pointer to the old clock.
            // Store the pointer to the new clock.
            m_spClock = pPresentationClock;
            hr = GetUnknown(MF_MEDIASINK_SAMPLEGRABBERCALLBACK, IID_IMFSampleGrabberSinkCallback, (LPVOID*)pSampleCallback.GetAddressOf());
        }
        LeaveCriticalSection(&m_critSec);
        if (SUCCEEDED(hr))
            hr = pSampleCallback->OnSetPresentationClock(pPresentationClock);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"MediaSink::SetPresentationClock: HRESULT=%i\n", hr);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE GetPresentationClock(
        IMFPresentationClock **ppPresentationClock) {
        if (ppPresentationClock == NULL) {
            return E_INVALIDARG;
        }

        EnterCriticalSection(&m_critSec);

        HRESULT hr = CheckShutdown();

        if (SUCCEEDED(hr)) {
            if (m_spClock == NULL) {
                hr = MF_E_NO_CLOCK; // There is no presentation clock.
            } else {
                // Return the pointer to the caller.
                hr = m_spClock.CopyTo(ppPresentationClock);
            }
        }
        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"MediaSink::GetPresentationClock: HRESULT=%i\n", hr);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE Shutdown(void) {
        EnterCriticalSection(&m_critSec);

        HRESULT hr = CheckShutdown();

        if (SUCCEEDED(hr)) {
            ForEach(m_streams, ShutdownFunc());
            m_streams.Clear();
            m_spClock.ReleaseAndGetAddressOf();

            hr = DeleteItem(MF_MEDIASINK_PREFERREDTYPE);
            m_IsShutdown = true;
        }

        LeaveCriticalSection(&m_critSec);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"MediaSink::Shutdown: HRESULT=%i\n", hr);
        return hr;
    }
    class ShutdownFunc
    {
    public:
        HRESULT operator()(IMFStreamSink *pStream) const
        {
            static_cast<StreamSink *>(pStream)->Shutdown();
            return S_OK;
        }
    };

    class StartFunc
    {
    public:
        StartFunc(LONGLONG llStartTime)
            : _llStartTime(llStartTime)
        {
        }

        HRESULT operator()(IMFStreamSink *pStream) const
        {
            return static_cast<StreamSink *>(pStream)->Start(_llStartTime);
        }

        LONGLONG _llStartTime;
    };

    class StopFunc
    {
    public:
        HRESULT operator()(IMFStreamSink *pStream) const
        {
            return static_cast<StreamSink *>(pStream)->Stop();
        }
    };

    template <class T, class TFunc>
    HRESULT ForEach(ComPtrList<T> &col, TFunc fn)
    {
        ComPtrList<T>::POSITION pos = col.FrontPosition();
        ComPtrList<T>::POSITION endPos = col.EndPosition();
        HRESULT hr = S_OK;

        for (; pos != endPos; pos = col.Next(pos))
        {
            ComPtr<T> spStream;

            hr = col.GetItemPos(pos, &spStream);
            if (FAILED(hr))
            {
                break;
            }

            hr = fn(spStream.Get());
        }

        return hr;
    }
    //IMFClockStateSink
    HRESULT STDMETHODCALLTYPE OnClockStart(
        MFTIME hnsSystemTime,
        LONGLONG llClockStartOffset) {
        EnterCriticalSection(&m_critSec);
        HRESULT hr = CheckShutdown();

        if (SUCCEEDED(hr))
        {
            // Start each stream.
            m_llStartTime = llClockStartOffset;
            hr = ForEach(m_streams, StartFunc(llClockStartOffset));
        }
        ComPtr<IMFSampleGrabberSinkCallback> pSampleCallback;
        if (SUCCEEDED(hr))
            hr = GetUnknown(MF_MEDIASINK_SAMPLEGRABBERCALLBACK, IID_IMFSampleGrabberSinkCallback, (LPVOID*)pSampleCallback.GetAddressOf());
        LeaveCriticalSection(&m_critSec);
        if (SUCCEEDED(hr))
            hr = pSampleCallback->OnClockStart(hnsSystemTime, llClockStartOffset);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"MediaSink::OnClockStart: HRESULT=%i\n", hr);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE OnClockStop(
        MFTIME hnsSystemTime) {
        EnterCriticalSection(&m_critSec);
        HRESULT hr = CheckShutdown();

        if (SUCCEEDED(hr))
        {
            // Stop each stream
            hr = ForEach(m_streams, StopFunc());
        }
        ComPtr<IMFSampleGrabberSinkCallback> pSampleCallback;
        if (SUCCEEDED(hr))
            hr = GetUnknown(MF_MEDIASINK_SAMPLEGRABBERCALLBACK, IID_IMFSampleGrabberSinkCallback, (LPVOID*)pSampleCallback.GetAddressOf());
        LeaveCriticalSection(&m_critSec);
        if (SUCCEEDED(hr))
            hr = pSampleCallback->OnClockStop(hnsSystemTime);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"MediaSink::OnClockStop: HRESULT=%i\n", hr);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE OnClockPause(
        MFTIME hnsSystemTime) {
        HRESULT hr;
        ComPtr<IMFSampleGrabberSinkCallback> pSampleCallback;
        hr = GetUnknown(MF_MEDIASINK_SAMPLEGRABBERCALLBACK, IID_IMFSampleGrabberSinkCallback, (LPVOID*)pSampleCallback.GetAddressOf());
        if (SUCCEEDED(hr))
            hr = pSampleCallback->OnClockPause(hnsSystemTime);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"MediaSink::OnClockPause: HRESULT=%i\n", hr);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE OnClockRestart(
        MFTIME hnsSystemTime) {
           HRESULT hr;
        ComPtr<IMFSampleGrabberSinkCallback> pSampleCallback;
        hr = GetUnknown(MF_MEDIASINK_SAMPLEGRABBERCALLBACK, IID_IMFSampleGrabberSinkCallback, (LPVOID*)pSampleCallback.GetAddressOf());
        if (SUCCEEDED(hr))
            hr = pSampleCallback->OnClockRestart(hnsSystemTime);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"MediaSink::OnClockRestart: HRESULT=%i\n", hr);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE OnClockSetRate(
        MFTIME hnsSystemTime,
        float flRate) {
           HRESULT hr;
        ComPtr<IMFSampleGrabberSinkCallback> pSampleCallback;
        hr = GetUnknown(MF_MEDIASINK_SAMPLEGRABBERCALLBACK, IID_IMFSampleGrabberSinkCallback, (LPVOID*)pSampleCallback.GetAddressOf());
        if (SUCCEEDED(hr))
            hr = pSampleCallback->OnClockSetRate(hnsSystemTime, flRate);
        DebugPrintOut *DPO = &DebugPrintOut::getInstance();
        DPO->printOut(L"MediaSink::OnClockSetRate: HRESULT=%i\n", hr);
        return hr;
    }
private:
#ifndef HAVE_WINRT
    long m_cRef;
#endif
    CRITICAL_SECTION            m_critSec;
    bool                        m_IsShutdown;
    ComPtrList<IMFStreamSink>    m_streams;
    ComPtr<IMFPresentationClock>    m_spClock;
    LONGLONG                        m_llStartTime;
};

#ifdef HAVE_WINRT
ActivatableClass(MediaSink);
#endif
