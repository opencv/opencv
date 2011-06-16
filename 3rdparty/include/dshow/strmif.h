/**
 * This file has no copyright assigned and is placed in the Public Domain.
 * This file is part of the w64 mingw-runtime package.
 * No warranty is given; refer to the file DISCLAIMER.PD within this package.
 */
#ifndef __REQUIRED_RPCNDR_H_VERSION__
#define __REQUIRED_RPCNDR_H_VERSION__ 475
#endif

#include "rpc.h"
#include "rpcndr.h"

#ifndef __RPCNDR_H_VERSION__
#error This stub requires an updated version of <rpcndr.h>
#endif

#ifndef COM_NO_WINDOWS_H
#include "windows.h"
#include "ole2.h"
#endif

#ifndef __strmif_h__
#define __strmif_h__

#ifndef __ICreateDevEnum_FWD_DEFINED__
#define __ICreateDevEnum_FWD_DEFINED__
typedef struct ICreateDevEnum ICreateDevEnum;
#endif

#ifndef __IPin_FWD_DEFINED__
#define __IPin_FWD_DEFINED__
typedef struct IPin IPin;
#endif

#ifndef __IEnumPins_FWD_DEFINED__
#define __IEnumPins_FWD_DEFINED__
typedef struct IEnumPins IEnumPins;
#endif

#ifndef __IEnumMediaTypes_FWD_DEFINED__
#define __IEnumMediaTypes_FWD_DEFINED__
typedef struct IEnumMediaTypes IEnumMediaTypes;
#endif

#ifndef __IFilterGraph_FWD_DEFINED__
#define __IFilterGraph_FWD_DEFINED__
typedef struct IFilterGraph IFilterGraph;
#endif

#ifndef __IEnumFilters_FWD_DEFINED__
#define __IEnumFilters_FWD_DEFINED__
typedef struct IEnumFilters IEnumFilters;
#endif

#ifndef __IMediaFilter_FWD_DEFINED__
#define __IMediaFilter_FWD_DEFINED__
typedef struct IMediaFilter IMediaFilter;
#endif

#ifndef __IBaseFilter_FWD_DEFINED__
#define __IBaseFilter_FWD_DEFINED__
typedef struct IBaseFilter IBaseFilter;
#endif

#ifndef __IReferenceClock_FWD_DEFINED__
#define __IReferenceClock_FWD_DEFINED__
typedef struct IReferenceClock IReferenceClock;
#endif

#ifndef __IReferenceClock2_FWD_DEFINED__
#define __IReferenceClock2_FWD_DEFINED__
typedef struct IReferenceClock2 IReferenceClock2;
#endif

#ifndef __IMediaSample_FWD_DEFINED__
#define __IMediaSample_FWD_DEFINED__
typedef struct IMediaSample IMediaSample;
#endif

#ifndef __IMediaSample2_FWD_DEFINED__
#define __IMediaSample2_FWD_DEFINED__
typedef struct IMediaSample2 IMediaSample2;
#endif

#ifndef __IMemAllocator_FWD_DEFINED__
#define __IMemAllocator_FWD_DEFINED__
typedef struct IMemAllocator IMemAllocator;
#endif

#ifndef __IMemAllocatorCallbackTemp_FWD_DEFINED__
#define __IMemAllocatorCallbackTemp_FWD_DEFINED__
typedef struct IMemAllocatorCallbackTemp IMemAllocatorCallbackTemp;
#endif

#ifndef __IMemAllocatorNotifyCallbackTemp_FWD_DEFINED__
#define __IMemAllocatorNotifyCallbackTemp_FWD_DEFINED__
typedef struct IMemAllocatorNotifyCallbackTemp IMemAllocatorNotifyCallbackTemp;
#endif

#ifndef __IMemInputPin_FWD_DEFINED__
#define __IMemInputPin_FWD_DEFINED__
typedef struct IMemInputPin IMemInputPin;
#endif

#ifndef __IAMovieSetup_FWD_DEFINED__
#define __IAMovieSetup_FWD_DEFINED__
typedef struct IAMovieSetup IAMovieSetup;
#endif

#ifndef __IMediaSeeking_FWD_DEFINED__
#define __IMediaSeeking_FWD_DEFINED__
typedef struct IMediaSeeking IMediaSeeking;
#endif

#ifndef __IEnumRegFilters_FWD_DEFINED__
#define __IEnumRegFilters_FWD_DEFINED__
typedef struct IEnumRegFilters IEnumRegFilters;
#endif

#ifndef __IFilterMapper_FWD_DEFINED__
#define __IFilterMapper_FWD_DEFINED__
typedef struct IFilterMapper IFilterMapper;
#endif

#ifndef __IFilterMapper2_FWD_DEFINED__
#define __IFilterMapper2_FWD_DEFINED__
typedef struct IFilterMapper2 IFilterMapper2;
#endif

#ifndef __IFilterMapper3_FWD_DEFINED__
#define __IFilterMapper3_FWD_DEFINED__
typedef struct IFilterMapper3 IFilterMapper3;
#endif

#ifndef __IQualityControl_FWD_DEFINED__
#define __IQualityControl_FWD_DEFINED__
typedef struct IQualityControl IQualityControl;
#endif

#ifndef __IOverlayNotify_FWD_DEFINED__
#define __IOverlayNotify_FWD_DEFINED__
typedef struct IOverlayNotify IOverlayNotify;
#endif

#ifndef __IOverlayNotify2_FWD_DEFINED__
#define __IOverlayNotify2_FWD_DEFINED__
typedef struct IOverlayNotify2 IOverlayNotify2;
#endif

#ifndef __IOverlay_FWD_DEFINED__
#define __IOverlay_FWD_DEFINED__
typedef struct IOverlay IOverlay;
#endif

#ifndef __IMediaEventSink_FWD_DEFINED__
#define __IMediaEventSink_FWD_DEFINED__
typedef struct IMediaEventSink IMediaEventSink;
#endif

#ifndef __IFileSourceFilter_FWD_DEFINED__
#define __IFileSourceFilter_FWD_DEFINED__
typedef struct IFileSourceFilter IFileSourceFilter;
#endif

#ifndef __IFileSinkFilter_FWD_DEFINED__
#define __IFileSinkFilter_FWD_DEFINED__
typedef struct IFileSinkFilter IFileSinkFilter;
#endif

#ifndef __IFileSinkFilter2_FWD_DEFINED__
#define __IFileSinkFilter2_FWD_DEFINED__
typedef struct IFileSinkFilter2 IFileSinkFilter2;
#endif

#ifndef __IGraphBuilder_FWD_DEFINED__
#define __IGraphBuilder_FWD_DEFINED__
typedef struct IGraphBuilder IGraphBuilder;
#endif

#ifndef __ICaptureGraphBuilder_FWD_DEFINED__
#define __ICaptureGraphBuilder_FWD_DEFINED__
typedef struct ICaptureGraphBuilder ICaptureGraphBuilder;
#endif

#ifndef __IAMCopyCaptureFileProgress_FWD_DEFINED__
#define __IAMCopyCaptureFileProgress_FWD_DEFINED__
typedef struct IAMCopyCaptureFileProgress IAMCopyCaptureFileProgress;
#endif

#ifndef __ICaptureGraphBuilder2_FWD_DEFINED__
#define __ICaptureGraphBuilder2_FWD_DEFINED__
typedef struct ICaptureGraphBuilder2 ICaptureGraphBuilder2;
#endif

#ifndef __IFilterGraph2_FWD_DEFINED__
#define __IFilterGraph2_FWD_DEFINED__
typedef struct IFilterGraph2 IFilterGraph2;
#endif

#ifndef __IStreamBuilder_FWD_DEFINED__
#define __IStreamBuilder_FWD_DEFINED__
typedef struct IStreamBuilder IStreamBuilder;
#endif

#ifndef __IAsyncReader_FWD_DEFINED__
#define __IAsyncReader_FWD_DEFINED__
typedef struct IAsyncReader IAsyncReader;
#endif

#ifndef __IGraphVersion_FWD_DEFINED__
#define __IGraphVersion_FWD_DEFINED__
typedef struct IGraphVersion IGraphVersion;
#endif

#ifndef __IResourceConsumer_FWD_DEFINED__
#define __IResourceConsumer_FWD_DEFINED__
typedef struct IResourceConsumer IResourceConsumer;
#endif

#ifndef __IResourceManager_FWD_DEFINED__
#define __IResourceManager_FWD_DEFINED__
typedef struct IResourceManager IResourceManager;
#endif

#ifndef __IDistributorNotify_FWD_DEFINED__
#define __IDistributorNotify_FWD_DEFINED__
typedef struct IDistributorNotify IDistributorNotify;
#endif

#ifndef __IAMStreamControl_FWD_DEFINED__
#define __IAMStreamControl_FWD_DEFINED__
typedef struct IAMStreamControl IAMStreamControl;
#endif

#ifndef __ISeekingPassThru_FWD_DEFINED__
#define __ISeekingPassThru_FWD_DEFINED__
typedef struct ISeekingPassThru ISeekingPassThru;
#endif

#ifndef __IAMStreamConfig_FWD_DEFINED__
#define __IAMStreamConfig_FWD_DEFINED__
typedef struct IAMStreamConfig IAMStreamConfig;
#endif

#ifndef __IConfigInterleaving_FWD_DEFINED__
#define __IConfigInterleaving_FWD_DEFINED__
typedef struct IConfigInterleaving IConfigInterleaving;
#endif

#ifndef __IConfigAviMux_FWD_DEFINED__
#define __IConfigAviMux_FWD_DEFINED__
typedef struct IConfigAviMux IConfigAviMux;
#endif

#ifndef __IAMVideoCompression_FWD_DEFINED__
#define __IAMVideoCompression_FWD_DEFINED__
typedef struct IAMVideoCompression IAMVideoCompression;
#endif

#ifndef __IAMVfwCaptureDialogs_FWD_DEFINED__
#define __IAMVfwCaptureDialogs_FWD_DEFINED__
typedef struct IAMVfwCaptureDialogs IAMVfwCaptureDialogs;
#endif

#ifndef __IAMVfwCompressDialogs_FWD_DEFINED__
#define __IAMVfwCompressDialogs_FWD_DEFINED__
typedef struct IAMVfwCompressDialogs IAMVfwCompressDialogs;
#endif

#ifndef __IAMDroppedFrames_FWD_DEFINED__
#define __IAMDroppedFrames_FWD_DEFINED__
typedef struct IAMDroppedFrames IAMDroppedFrames;
#endif

#ifndef __IAMAudioInputMixer_FWD_DEFINED__
#define __IAMAudioInputMixer_FWD_DEFINED__
typedef struct IAMAudioInputMixer IAMAudioInputMixer;
#endif

#ifndef __IAMBufferNegotiation_FWD_DEFINED__
#define __IAMBufferNegotiation_FWD_DEFINED__
typedef struct IAMBufferNegotiation IAMBufferNegotiation;
#endif

#ifndef __IAMAnalogVideoDecoder_FWD_DEFINED__
#define __IAMAnalogVideoDecoder_FWD_DEFINED__
typedef struct IAMAnalogVideoDecoder IAMAnalogVideoDecoder;
#endif

#ifndef __IAMVideoProcAmp_FWD_DEFINED__
#define __IAMVideoProcAmp_FWD_DEFINED__
typedef struct IAMVideoProcAmp IAMVideoProcAmp;
#endif

#ifndef __IAMCameraControl_FWD_DEFINED__
#define __IAMCameraControl_FWD_DEFINED__
typedef struct IAMCameraControl IAMCameraControl;
#endif

#ifndef __IAMVideoControl_FWD_DEFINED__
#define __IAMVideoControl_FWD_DEFINED__
typedef struct IAMVideoControl IAMVideoControl;
#endif

#ifndef __IAMCrossbar_FWD_DEFINED__
#define __IAMCrossbar_FWD_DEFINED__
typedef struct IAMCrossbar IAMCrossbar;
#endif

#ifndef __IAMTuner_FWD_DEFINED__
#define __IAMTuner_FWD_DEFINED__
typedef struct IAMTuner IAMTuner;
#endif

#ifndef __IAMTunerNotification_FWD_DEFINED__
#define __IAMTunerNotification_FWD_DEFINED__
typedef struct IAMTunerNotification IAMTunerNotification;
#endif

#ifndef __IAMTVTuner_FWD_DEFINED__
#define __IAMTVTuner_FWD_DEFINED__
typedef struct IAMTVTuner IAMTVTuner;
#endif

#ifndef __IBPCSatelliteTuner_FWD_DEFINED__
#define __IBPCSatelliteTuner_FWD_DEFINED__
typedef struct IBPCSatelliteTuner IBPCSatelliteTuner;
#endif

#ifndef __IAMTVAudio_FWD_DEFINED__
#define __IAMTVAudio_FWD_DEFINED__
typedef struct IAMTVAudio IAMTVAudio;
#endif

#ifndef __IAMTVAudioNotification_FWD_DEFINED__
#define __IAMTVAudioNotification_FWD_DEFINED__
typedef struct IAMTVAudioNotification IAMTVAudioNotification;
#endif

#ifndef __IAMAnalogVideoEncoder_FWD_DEFINED__
#define __IAMAnalogVideoEncoder_FWD_DEFINED__
typedef struct IAMAnalogVideoEncoder IAMAnalogVideoEncoder;
#endif

#ifndef __IKsPropertySet_FWD_DEFINED__
#define __IKsPropertySet_FWD_DEFINED__
typedef struct IKsPropertySet IKsPropertySet;
#endif

#ifndef __IMediaPropertyBag_FWD_DEFINED__
#define __IMediaPropertyBag_FWD_DEFINED__
typedef struct IMediaPropertyBag IMediaPropertyBag;
#endif

#ifndef __IPersistMediaPropertyBag_FWD_DEFINED__
#define __IPersistMediaPropertyBag_FWD_DEFINED__
typedef struct IPersistMediaPropertyBag IPersistMediaPropertyBag;
#endif

#ifndef __IAMPhysicalPinInfo_FWD_DEFINED__
#define __IAMPhysicalPinInfo_FWD_DEFINED__
typedef struct IAMPhysicalPinInfo IAMPhysicalPinInfo;
#endif

#ifndef __IAMExtDevice_FWD_DEFINED__
#define __IAMExtDevice_FWD_DEFINED__
typedef struct IAMExtDevice IAMExtDevice;
#endif

#ifndef __IAMExtTransport_FWD_DEFINED__
#define __IAMExtTransport_FWD_DEFINED__
typedef struct IAMExtTransport IAMExtTransport;
#endif

#ifndef __IAMTimecodeReader_FWD_DEFINED__
#define __IAMTimecodeReader_FWD_DEFINED__
typedef struct IAMTimecodeReader IAMTimecodeReader;
#endif

#ifndef __IAMTimecodeGenerator_FWD_DEFINED__
#define __IAMTimecodeGenerator_FWD_DEFINED__
typedef struct IAMTimecodeGenerator IAMTimecodeGenerator;
#endif

#ifndef __IAMTimecodeDisplay_FWD_DEFINED__
#define __IAMTimecodeDisplay_FWD_DEFINED__
typedef struct IAMTimecodeDisplay IAMTimecodeDisplay;
#endif

#ifndef __IAMDevMemoryAllocator_FWD_DEFINED__
#define __IAMDevMemoryAllocator_FWD_DEFINED__
typedef struct IAMDevMemoryAllocator IAMDevMemoryAllocator;
#endif

#ifndef __IAMDevMemoryControl_FWD_DEFINED__
#define __IAMDevMemoryControl_FWD_DEFINED__
typedef struct IAMDevMemoryControl IAMDevMemoryControl;
#endif

#ifndef __IAMStreamSelect_FWD_DEFINED__
#define __IAMStreamSelect_FWD_DEFINED__
typedef struct IAMStreamSelect IAMStreamSelect;
#endif

#ifndef __IAMResourceControl_FWD_DEFINED__
#define __IAMResourceControl_FWD_DEFINED__
typedef struct IAMResourceControl IAMResourceControl;
#endif

#ifndef __IAMClockAdjust_FWD_DEFINED__
#define __IAMClockAdjust_FWD_DEFINED__
typedef struct IAMClockAdjust IAMClockAdjust;
#endif

#ifndef __IAMFilterMiscFlags_FWD_DEFINED__
#define __IAMFilterMiscFlags_FWD_DEFINED__
typedef struct IAMFilterMiscFlags IAMFilterMiscFlags;
#endif

#ifndef __IDrawVideoImage_FWD_DEFINED__
#define __IDrawVideoImage_FWD_DEFINED__
typedef struct IDrawVideoImage IDrawVideoImage;
#endif

#ifndef __IDecimateVideoImage_FWD_DEFINED__
#define __IDecimateVideoImage_FWD_DEFINED__
typedef struct IDecimateVideoImage IDecimateVideoImage;
#endif

#ifndef __IAMVideoDecimationProperties_FWD_DEFINED__
#define __IAMVideoDecimationProperties_FWD_DEFINED__
typedef struct IAMVideoDecimationProperties IAMVideoDecimationProperties;
#endif

#ifndef __IVideoFrameStep_FWD_DEFINED__
#define __IVideoFrameStep_FWD_DEFINED__
typedef struct IVideoFrameStep IVideoFrameStep;
#endif

#ifndef __IAMLatency_FWD_DEFINED__
#define __IAMLatency_FWD_DEFINED__
typedef struct IAMLatency IAMLatency;
#endif

#ifndef __IAMPushSource_FWD_DEFINED__
#define __IAMPushSource_FWD_DEFINED__
typedef struct IAMPushSource IAMPushSource;
#endif

#ifndef __IAMDeviceRemoval_FWD_DEFINED__
#define __IAMDeviceRemoval_FWD_DEFINED__
typedef struct IAMDeviceRemoval IAMDeviceRemoval;
#endif

#ifndef __IDVEnc_FWD_DEFINED__
#define __IDVEnc_FWD_DEFINED__
typedef struct IDVEnc IDVEnc;
#endif

#ifndef __IIPDVDec_FWD_DEFINED__
#define __IIPDVDec_FWD_DEFINED__
typedef struct IIPDVDec IIPDVDec;
#endif

#ifndef __IDVRGB219_FWD_DEFINED__
#define __IDVRGB219_FWD_DEFINED__
typedef struct IDVRGB219 IDVRGB219;
#endif

#ifndef __IDVSplitter_FWD_DEFINED__
#define __IDVSplitter_FWD_DEFINED__
typedef struct IDVSplitter IDVSplitter;
#endif

#ifndef __IAMAudioRendererStats_FWD_DEFINED__
#define __IAMAudioRendererStats_FWD_DEFINED__
typedef struct IAMAudioRendererStats IAMAudioRendererStats;
#endif

#ifndef __IAMGraphStreams_FWD_DEFINED__
#define __IAMGraphStreams_FWD_DEFINED__
typedef struct IAMGraphStreams IAMGraphStreams;
#endif

#ifndef __IAMOverlayFX_FWD_DEFINED__
#define __IAMOverlayFX_FWD_DEFINED__
typedef struct IAMOverlayFX IAMOverlayFX;
#endif

#ifndef __IAMOpenProgress_FWD_DEFINED__
#define __IAMOpenProgress_FWD_DEFINED__
typedef struct IAMOpenProgress IAMOpenProgress;
#endif

#ifndef __IMpeg2Demultiplexer_FWD_DEFINED__
#define __IMpeg2Demultiplexer_FWD_DEFINED__
typedef struct IMpeg2Demultiplexer IMpeg2Demultiplexer;
#endif

#ifndef __IEnumStreamIdMap_FWD_DEFINED__
#define __IEnumStreamIdMap_FWD_DEFINED__
typedef struct IEnumStreamIdMap IEnumStreamIdMap;
#endif

#ifndef __IMPEG2StreamIdMap_FWD_DEFINED__
#define __IMPEG2StreamIdMap_FWD_DEFINED__
typedef struct IMPEG2StreamIdMap IMPEG2StreamIdMap;
#endif

#ifndef __IRegisterServiceProvider_FWD_DEFINED__
#define __IRegisterServiceProvider_FWD_DEFINED__
typedef struct IRegisterServiceProvider IRegisterServiceProvider;
#endif

#ifndef __IAMClockSlave_FWD_DEFINED__
#define __IAMClockSlave_FWD_DEFINED__
typedef struct IAMClockSlave IAMClockSlave;
#endif

#ifndef __IAMGraphBuilderCallback_FWD_DEFINED__
#define __IAMGraphBuilderCallback_FWD_DEFINED__
typedef struct IAMGraphBuilderCallback IAMGraphBuilderCallback;
#endif

#ifndef __ICodecAPI_FWD_DEFINED__
#define __ICodecAPI_FWD_DEFINED__
typedef struct ICodecAPI ICodecAPI;
#endif

#ifndef __IGetCapabilitiesKey_FWD_DEFINED__
#define __IGetCapabilitiesKey_FWD_DEFINED__
typedef struct IGetCapabilitiesKey IGetCapabilitiesKey;
#endif

#ifndef __IEncoderAPI_FWD_DEFINED__
#define __IEncoderAPI_FWD_DEFINED__
typedef struct IEncoderAPI IEncoderAPI;
#endif

#ifndef __IVideoEncoder_FWD_DEFINED__
#define __IVideoEncoder_FWD_DEFINED__
typedef struct IVideoEncoder IVideoEncoder;
#endif

#ifndef __IAMDecoderCaps_FWD_DEFINED__
#define __IAMDecoderCaps_FWD_DEFINED__
typedef struct IAMDecoderCaps IAMDecoderCaps;
#endif

#ifndef __IAMCertifiedOutputProtection_FWD_DEFINED__
#define __IAMCertifiedOutputProtection_FWD_DEFINED__
typedef struct IAMCertifiedOutputProtection IAMCertifiedOutputProtection;
#endif

#ifndef __IDvdControl_FWD_DEFINED__
#define __IDvdControl_FWD_DEFINED__
typedef struct IDvdControl IDvdControl;
#endif

#ifndef __IDvdInfo_FWD_DEFINED__
#define __IDvdInfo_FWD_DEFINED__
typedef struct IDvdInfo IDvdInfo;
#endif

#ifndef __IDvdCmd_FWD_DEFINED__
#define __IDvdCmd_FWD_DEFINED__
typedef struct IDvdCmd IDvdCmd;
#endif

#ifndef __IDvdState_FWD_DEFINED__
#define __IDvdState_FWD_DEFINED__
typedef struct IDvdState IDvdState;
#endif

#ifndef __IDvdControl2_FWD_DEFINED__
#define __IDvdControl2_FWD_DEFINED__
typedef struct IDvdControl2 IDvdControl2;
#endif

#ifndef __IDvdInfo2_FWD_DEFINED__
#define __IDvdInfo2_FWD_DEFINED__
typedef struct IDvdInfo2 IDvdInfo2;
#endif

#ifndef __IDvdGraphBuilder_FWD_DEFINED__
#define __IDvdGraphBuilder_FWD_DEFINED__
typedef struct IDvdGraphBuilder IDvdGraphBuilder;
#endif

#ifndef __IDDrawExclModeVideo_FWD_DEFINED__
#define __IDDrawExclModeVideo_FWD_DEFINED__
typedef struct IDDrawExclModeVideo IDDrawExclModeVideo;
#endif

#ifndef __IDDrawExclModeVideoCallback_FWD_DEFINED__
#define __IDDrawExclModeVideoCallback_FWD_DEFINED__
typedef struct IDDrawExclModeVideoCallback IDDrawExclModeVideoCallback;
#endif

#ifndef __IPinConnection_FWD_DEFINED__
#define __IPinConnection_FWD_DEFINED__
typedef struct IPinConnection IPinConnection;
#endif

#ifndef __IPinFlowControl_FWD_DEFINED__
#define __IPinFlowControl_FWD_DEFINED__
typedef struct IPinFlowControl IPinFlowControl;
#endif

#ifndef __IGraphConfig_FWD_DEFINED__
#define __IGraphConfig_FWD_DEFINED__
typedef struct IGraphConfig IGraphConfig;
#endif

#ifndef __IGraphConfigCallback_FWD_DEFINED__
#define __IGraphConfigCallback_FWD_DEFINED__
typedef struct IGraphConfigCallback IGraphConfigCallback;
#endif

#ifndef __IFilterChain_FWD_DEFINED__
#define __IFilterChain_FWD_DEFINED__
typedef struct IFilterChain IFilterChain;
#endif

#ifndef __IVMRImagePresenter_FWD_DEFINED__
#define __IVMRImagePresenter_FWD_DEFINED__
typedef struct IVMRImagePresenter IVMRImagePresenter;
#endif

#ifndef __IVMRSurfaceAllocator_FWD_DEFINED__
#define __IVMRSurfaceAllocator_FWD_DEFINED__
typedef struct IVMRSurfaceAllocator IVMRSurfaceAllocator;
#endif

#ifndef __IVMRSurfaceAllocatorNotify_FWD_DEFINED__
#define __IVMRSurfaceAllocatorNotify_FWD_DEFINED__
typedef struct IVMRSurfaceAllocatorNotify IVMRSurfaceAllocatorNotify;
#endif

#ifndef __IVMRWindowlessControl_FWD_DEFINED__
#define __IVMRWindowlessControl_FWD_DEFINED__
typedef struct IVMRWindowlessControl IVMRWindowlessControl;
#endif

#ifndef __IVMRMixerControl_FWD_DEFINED__
#define __IVMRMixerControl_FWD_DEFINED__
typedef struct IVMRMixerControl IVMRMixerControl;
#endif

#ifndef __IVMRMonitorConfig_FWD_DEFINED__
#define __IVMRMonitorConfig_FWD_DEFINED__
typedef struct IVMRMonitorConfig IVMRMonitorConfig;
#endif

#ifndef __IVMRFilterConfig_FWD_DEFINED__
#define __IVMRFilterConfig_FWD_DEFINED__
typedef struct IVMRFilterConfig IVMRFilterConfig;
#endif

#ifndef __IVMRAspectRatioControl_FWD_DEFINED__
#define __IVMRAspectRatioControl_FWD_DEFINED__
typedef struct IVMRAspectRatioControl IVMRAspectRatioControl;
#endif

#ifndef __IVMRDeinterlaceControl_FWD_DEFINED__
#define __IVMRDeinterlaceControl_FWD_DEFINED__
typedef struct IVMRDeinterlaceControl IVMRDeinterlaceControl;
#endif

#ifndef __IVMRMixerBitmap_FWD_DEFINED__
#define __IVMRMixerBitmap_FWD_DEFINED__
typedef struct IVMRMixerBitmap IVMRMixerBitmap;
#endif

#ifndef __IVMRImageCompositor_FWD_DEFINED__
#define __IVMRImageCompositor_FWD_DEFINED__
typedef struct IVMRImageCompositor IVMRImageCompositor;
#endif

#ifndef __IVMRVideoStreamControl_FWD_DEFINED__
#define __IVMRVideoStreamControl_FWD_DEFINED__
typedef struct IVMRVideoStreamControl IVMRVideoStreamControl;
#endif

#ifndef __IVMRSurface_FWD_DEFINED__
#define __IVMRSurface_FWD_DEFINED__
typedef struct IVMRSurface IVMRSurface;
#endif

#ifndef __IVMRImagePresenterConfig_FWD_DEFINED__
#define __IVMRImagePresenterConfig_FWD_DEFINED__
typedef struct IVMRImagePresenterConfig IVMRImagePresenterConfig;
#endif

#ifndef __IVMRImagePresenterExclModeConfig_FWD_DEFINED__
#define __IVMRImagePresenterExclModeConfig_FWD_DEFINED__
typedef struct IVMRImagePresenterExclModeConfig IVMRImagePresenterExclModeConfig;
#endif

#ifndef __IVPManager_FWD_DEFINED__
#define __IVPManager_FWD_DEFINED__
typedef struct IVPManager IVPManager;
#endif

#include "unknwn.h"
#include "objidl.h"
#include "oaidl.h"
#include "ocidl.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __MIDL_user_allocate_free_DEFINED__
#define __MIDL_user_allocate_free_DEFINED__
  void *__RPC_API MIDL_user_allocate(size_t);
  void __RPC_API MIDL_user_free(void *);
#endif

#define CDEF_CLASS_DEFAULT 0x0001
#define CDEF_BYPASS_CLASS_MANAGER 0x0002
#define CDEF_MERIT_ABOVE_DO_NOT_USE 0x0008
#define CDEF_DEVMON_CMGR_DEVICE 0x0010
#define CDEF_DEVMON_DMO 0x0020
#define CDEF_DEVMON_PNP_DEVICE 0x0040
#define CDEF_DEVMON_FILTER 0x0080
#define CDEF_DEVMON_SELECTIVE_MASK 0x00f0

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0000_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0000_v0_0_s_ifspec;
#ifndef __ICreateDevEnum_INTERFACE_DEFINED__
#define __ICreateDevEnum_INTERFACE_DEFINED__
  EXTERN_C const IID IID_ICreateDevEnum;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct ICreateDevEnum : public IUnknown {
  public:
    virtual HRESULT WINAPI CreateClassEnumerator(REFCLSID clsidDeviceClass,IEnumMoniker **ppEnumMoniker,DWORD dwFlags) = 0;
  };
#else
  typedef struct ICreateDevEnumVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(ICreateDevEnum *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(ICreateDevEnum *This);
      ULONG (WINAPI *Release)(ICreateDevEnum *This);
      HRESULT (WINAPI *CreateClassEnumerator)(ICreateDevEnum *This,REFCLSID clsidDeviceClass,IEnumMoniker **ppEnumMoniker,DWORD dwFlags);
    END_INTERFACE
  } ICreateDevEnumVtbl;
  struct ICreateDevEnum {
    CONST_VTBL struct ICreateDevEnumVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define ICreateDevEnum_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define ICreateDevEnum_AddRef(This) (This)->lpVtbl->AddRef(This)
#define ICreateDevEnum_Release(This) (This)->lpVtbl->Release(This)
#define ICreateDevEnum_CreateClassEnumerator(This,clsidDeviceClass,ppEnumMoniker,dwFlags) (This)->lpVtbl->CreateClassEnumerator(This,clsidDeviceClass,ppEnumMoniker,dwFlags)
#endif
#endif
  HRESULT WINAPI ICreateDevEnum_CreateClassEnumerator_Proxy(ICreateDevEnum *This,REFCLSID clsidDeviceClass,IEnumMoniker **ppEnumMoniker,DWORD dwFlags);
  void __RPC_STUB ICreateDevEnum_CreateClassEnumerator_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#define CHARS_IN_GUID 39
  typedef struct _AMMediaType {
    GUID majortype;
    GUID subtype;
    WINBOOL bFixedSizeSamples;
    WINBOOL bTemporalCompression;
    ULONG lSampleSize;
    GUID formattype;
    IUnknown *pUnk;
    ULONG cbFormat;
    BYTE *pbFormat;
  } AM_MEDIA_TYPE;

  typedef enum _PinDirection {
    PINDIR_INPUT = 0,PINDIR_OUTPUT = PINDIR_INPUT + 1
  } PIN_DIRECTION;

#define MAX_PIN_NAME 128
#define MAX_FILTER_NAME 128

#ifndef __REFERENCE_TIME_DEFINED
#define __REFERENCE_TIME_DEFINED
typedef LONGLONG REFERENCE_TIME;
#endif /*__REFERENCE_TIME_DEFINED*/

  typedef double REFTIME;
  typedef DWORD_PTR HSEMAPHORE;
  typedef DWORD_PTR HEVENT;

  typedef struct _AllocatorProperties {
    long cBuffers;
    long cbBuffer;
    long cbAlign;
    long cbPrefix;
  } ALLOCATOR_PROPERTIES;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0117_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0117_v0_0_s_ifspec;
#ifndef __IPin_INTERFACE_DEFINED__
#define __IPin_INTERFACE_DEFINED__
  typedef struct _PinInfo {
    IBaseFilter *pFilter;
    PIN_DIRECTION dir;
    WCHAR achName[128];
  } PIN_INFO;

  EXTERN_C const IID IID_IPin;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IPin : public IUnknown {
  public:
    virtual HRESULT WINAPI Connect(IPin *pReceivePin,const AM_MEDIA_TYPE *pmt) = 0;
    virtual HRESULT WINAPI ReceiveConnection(IPin *pConnector,const AM_MEDIA_TYPE *pmt) = 0;
    virtual HRESULT WINAPI Disconnect(void) = 0;
    virtual HRESULT WINAPI ConnectedTo(IPin **pPin) = 0;
    virtual HRESULT WINAPI ConnectionMediaType(AM_MEDIA_TYPE *pmt) = 0;
    virtual HRESULT WINAPI QueryPinInfo(PIN_INFO *pInfo) = 0;
    virtual HRESULT WINAPI QueryDirection(PIN_DIRECTION *pPinDir) = 0;
    virtual HRESULT WINAPI QueryId(LPWSTR *Id) = 0;
    virtual HRESULT WINAPI QueryAccept(const AM_MEDIA_TYPE *pmt) = 0;
    virtual HRESULT WINAPI EnumMediaTypes(IEnumMediaTypes **ppEnum) = 0;
    virtual HRESULT WINAPI QueryInternalConnections(IPin **apPin,ULONG *nPin) = 0;
    virtual HRESULT WINAPI EndOfStream(void) = 0;
    virtual HRESULT WINAPI BeginFlush(void) = 0;
    virtual HRESULT WINAPI EndFlush(void) = 0;
    virtual HRESULT WINAPI NewSegment(REFERENCE_TIME tStart,REFERENCE_TIME tStop,double dRate) = 0;
  };
#else
  typedef struct IPinVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IPin *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IPin *This);
      ULONG (WINAPI *Release)(IPin *This);
      HRESULT (WINAPI *Connect)(IPin *This,IPin *pReceivePin,const AM_MEDIA_TYPE *pmt);
      HRESULT (WINAPI *ReceiveConnection)(IPin *This,IPin *pConnector,const AM_MEDIA_TYPE *pmt);
      HRESULT (WINAPI *Disconnect)(IPin *This);
      HRESULT (WINAPI *ConnectedTo)(IPin *This,IPin **pPin);
      HRESULT (WINAPI *ConnectionMediaType)(IPin *This,AM_MEDIA_TYPE *pmt);
      HRESULT (WINAPI *QueryPinInfo)(IPin *This,PIN_INFO *pInfo);
      HRESULT (WINAPI *QueryDirection)(IPin *This,PIN_DIRECTION *pPinDir);
      HRESULT (WINAPI *QueryId)(IPin *This,LPWSTR *Id);
      HRESULT (WINAPI *QueryAccept)(IPin *This,const AM_MEDIA_TYPE *pmt);
      HRESULT (WINAPI *EnumMediaTypes)(IPin *This,IEnumMediaTypes **ppEnum);
      HRESULT (WINAPI *QueryInternalConnections)(IPin *This,IPin **apPin,ULONG *nPin);
      HRESULT (WINAPI *EndOfStream)(IPin *This);
      HRESULT (WINAPI *BeginFlush)(IPin *This);
      HRESULT (WINAPI *EndFlush)(IPin *This);
      HRESULT (WINAPI *NewSegment)(IPin *This,REFERENCE_TIME tStart,REFERENCE_TIME tStop,double dRate);
    END_INTERFACE
  } IPinVtbl;
  struct IPin {
    CONST_VTBL struct IPinVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IPin_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IPin_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IPin_Release(This) (This)->lpVtbl->Release(This)
#define IPin_Connect(This,pReceivePin,pmt) (This)->lpVtbl->Connect(This,pReceivePin,pmt)
#define IPin_ReceiveConnection(This,pConnector,pmt) (This)->lpVtbl->ReceiveConnection(This,pConnector,pmt)
#define IPin_Disconnect(This) (This)->lpVtbl->Disconnect(This)
#define IPin_ConnectedTo(This,pPin) (This)->lpVtbl->ConnectedTo(This,pPin)
#define IPin_ConnectionMediaType(This,pmt) (This)->lpVtbl->ConnectionMediaType(This,pmt)
#define IPin_QueryPinInfo(This,pInfo) (This)->lpVtbl->QueryPinInfo(This,pInfo)
#define IPin_QueryDirection(This,pPinDir) (This)->lpVtbl->QueryDirection(This,pPinDir)
#define IPin_QueryId(This,Id) (This)->lpVtbl->QueryId(This,Id)
#define IPin_QueryAccept(This,pmt) (This)->lpVtbl->QueryAccept(This,pmt)
#define IPin_EnumMediaTypes(This,ppEnum) (This)->lpVtbl->EnumMediaTypes(This,ppEnum)
#define IPin_QueryInternalConnections(This,apPin,nPin) (This)->lpVtbl->QueryInternalConnections(This,apPin,nPin)
#define IPin_EndOfStream(This) (This)->lpVtbl->EndOfStream(This)
#define IPin_BeginFlush(This) (This)->lpVtbl->BeginFlush(This)
#define IPin_EndFlush(This) (This)->lpVtbl->EndFlush(This)
#define IPin_NewSegment(This,tStart,tStop,dRate) (This)->lpVtbl->NewSegment(This,tStart,tStop,dRate)
#endif
#endif
  HRESULT WINAPI IPin_Connect_Proxy(IPin *This,IPin *pReceivePin,const AM_MEDIA_TYPE *pmt);
  void __RPC_STUB IPin_Connect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPin_ReceiveConnection_Proxy(IPin *This,IPin *pConnector,const AM_MEDIA_TYPE *pmt);
  void __RPC_STUB IPin_ReceiveConnection_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPin_Disconnect_Proxy(IPin *This);
  void __RPC_STUB IPin_Disconnect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPin_ConnectedTo_Proxy(IPin *This,IPin **pPin);
  void __RPC_STUB IPin_ConnectedTo_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPin_ConnectionMediaType_Proxy(IPin *This,AM_MEDIA_TYPE *pmt);
  void __RPC_STUB IPin_ConnectionMediaType_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPin_QueryPinInfo_Proxy(IPin *This,PIN_INFO *pInfo);
  void __RPC_STUB IPin_QueryPinInfo_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPin_QueryDirection_Proxy(IPin *This,PIN_DIRECTION *pPinDir);
  void __RPC_STUB IPin_QueryDirection_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPin_QueryId_Proxy(IPin *This,LPWSTR *Id);
  void __RPC_STUB IPin_QueryId_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPin_QueryAccept_Proxy(IPin *This,const AM_MEDIA_TYPE *pmt);
  void __RPC_STUB IPin_QueryAccept_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPin_EnumMediaTypes_Proxy(IPin *This,IEnumMediaTypes **ppEnum);
  void __RPC_STUB IPin_EnumMediaTypes_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPin_QueryInternalConnections_Proxy(IPin *This,IPin **apPin,ULONG *nPin);
  void __RPC_STUB IPin_QueryInternalConnections_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPin_EndOfStream_Proxy(IPin *This);
  void __RPC_STUB IPin_EndOfStream_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPin_BeginFlush_Proxy(IPin *This);
  void __RPC_STUB IPin_BeginFlush_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPin_EndFlush_Proxy(IPin *This);
  void __RPC_STUB IPin_EndFlush_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPin_NewSegment_Proxy(IPin *This,REFERENCE_TIME tStart,REFERENCE_TIME tStop,double dRate);
  void __RPC_STUB IPin_NewSegment_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IPin *PPIN;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0118_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0118_v0_0_s_ifspec;
#ifndef __IEnumPins_INTERFACE_DEFINED__
#define __IEnumPins_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IEnumPins;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IEnumPins : public IUnknown {
  public:
    virtual HRESULT WINAPI Next(ULONG cPins,IPin **ppPins,ULONG *pcFetched) = 0;
    virtual HRESULT WINAPI Skip(ULONG cPins) = 0;
    virtual HRESULT WINAPI Reset(void) = 0;
    virtual HRESULT WINAPI Clone(IEnumPins **ppEnum) = 0;
  };
#else
  typedef struct IEnumPinsVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IEnumPins *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IEnumPins *This);
      ULONG (WINAPI *Release)(IEnumPins *This);
      HRESULT (WINAPI *Next)(IEnumPins *This,ULONG cPins,IPin **ppPins,ULONG *pcFetched);
      HRESULT (WINAPI *Skip)(IEnumPins *This,ULONG cPins);
      HRESULT (WINAPI *Reset)(IEnumPins *This);
      HRESULT (WINAPI *Clone)(IEnumPins *This,IEnumPins **ppEnum);
    END_INTERFACE
  } IEnumPinsVtbl;
  struct IEnumPins {
    CONST_VTBL struct IEnumPinsVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IEnumPins_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IEnumPins_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IEnumPins_Release(This) (This)->lpVtbl->Release(This)
#define IEnumPins_Next(This,cPins,ppPins,pcFetched) (This)->lpVtbl->Next(This,cPins,ppPins,pcFetched)
#define IEnumPins_Skip(This,cPins) (This)->lpVtbl->Skip(This,cPins)
#define IEnumPins_Reset(This) (This)->lpVtbl->Reset(This)
#define IEnumPins_Clone(This,ppEnum) (This)->lpVtbl->Clone(This,ppEnum)
#endif
#endif
  HRESULT WINAPI IEnumPins_Next_Proxy(IEnumPins *This,ULONG cPins,IPin **ppPins,ULONG *pcFetched);
  void __RPC_STUB IEnumPins_Next_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEnumPins_Skip_Proxy(IEnumPins *This,ULONG cPins);
  void __RPC_STUB IEnumPins_Skip_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEnumPins_Reset_Proxy(IEnumPins *This);
  void __RPC_STUB IEnumPins_Reset_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEnumPins_Clone_Proxy(IEnumPins *This,IEnumPins **ppEnum);
  void __RPC_STUB IEnumPins_Clone_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IEnumPins *PENUMPINS;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0119_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0119_v0_0_s_ifspec;
#ifndef __IEnumMediaTypes_INTERFACE_DEFINED__
#define __IEnumMediaTypes_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IEnumMediaTypes;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IEnumMediaTypes : public IUnknown {
  public:
    virtual HRESULT WINAPI Next(ULONG cMediaTypes,AM_MEDIA_TYPE **ppMediaTypes,ULONG *pcFetched) = 0;
    virtual HRESULT WINAPI Skip(ULONG cMediaTypes) = 0;
    virtual HRESULT WINAPI Reset(void) = 0;
    virtual HRESULT WINAPI Clone(IEnumMediaTypes **ppEnum) = 0;
  };
#else
  typedef struct IEnumMediaTypesVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IEnumMediaTypes *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IEnumMediaTypes *This);
      ULONG (WINAPI *Release)(IEnumMediaTypes *This);
      HRESULT (WINAPI *Next)(IEnumMediaTypes *This,ULONG cMediaTypes,AM_MEDIA_TYPE **ppMediaTypes,ULONG *pcFetched);
      HRESULT (WINAPI *Skip)(IEnumMediaTypes *This,ULONG cMediaTypes);
      HRESULT (WINAPI *Reset)(IEnumMediaTypes *This);
      HRESULT (WINAPI *Clone)(IEnumMediaTypes *This,IEnumMediaTypes **ppEnum);
    END_INTERFACE
  } IEnumMediaTypesVtbl;
  struct IEnumMediaTypes {
    CONST_VTBL struct IEnumMediaTypesVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IEnumMediaTypes_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IEnumMediaTypes_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IEnumMediaTypes_Release(This) (This)->lpVtbl->Release(This)
#define IEnumMediaTypes_Next(This,cMediaTypes,ppMediaTypes,pcFetched) (This)->lpVtbl->Next(This,cMediaTypes,ppMediaTypes,pcFetched)
#define IEnumMediaTypes_Skip(This,cMediaTypes) (This)->lpVtbl->Skip(This,cMediaTypes)
#define IEnumMediaTypes_Reset(This) (This)->lpVtbl->Reset(This)
#define IEnumMediaTypes_Clone(This,ppEnum) (This)->lpVtbl->Clone(This,ppEnum)
#endif
#endif
  HRESULT WINAPI IEnumMediaTypes_Next_Proxy(IEnumMediaTypes *This,ULONG cMediaTypes,AM_MEDIA_TYPE **ppMediaTypes,ULONG *pcFetched);
  void __RPC_STUB IEnumMediaTypes_Next_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEnumMediaTypes_Skip_Proxy(IEnumMediaTypes *This,ULONG cMediaTypes);
  void __RPC_STUB IEnumMediaTypes_Skip_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEnumMediaTypes_Reset_Proxy(IEnumMediaTypes *This);
  void __RPC_STUB IEnumMediaTypes_Reset_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEnumMediaTypes_Clone_Proxy(IEnumMediaTypes *This,IEnumMediaTypes **ppEnum);
  void __RPC_STUB IEnumMediaTypes_Clone_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IEnumMediaTypes *PENUMMEDIATYPES;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0120_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0120_v0_0_s_ifspec;
#ifndef __IFilterGraph_INTERFACE_DEFINED__
#define __IFilterGraph_INTERFACE_DEFINED__

  EXTERN_C const IID IID_IFilterGraph;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IFilterGraph : public IUnknown {
  public:
    virtual HRESULT WINAPI AddFilter(IBaseFilter *pFilter,LPCWSTR pName) = 0;
    virtual HRESULT WINAPI RemoveFilter(IBaseFilter *pFilter) = 0;
    virtual HRESULT WINAPI EnumFilters(IEnumFilters **ppEnum) = 0;
    virtual HRESULT WINAPI FindFilterByName(LPCWSTR pName,IBaseFilter **ppFilter) = 0;
    virtual HRESULT WINAPI ConnectDirect(IPin *ppinOut,IPin *ppinIn,const AM_MEDIA_TYPE *pmt) = 0;
    virtual HRESULT WINAPI Reconnect(IPin *ppin) = 0;
    virtual HRESULT WINAPI Disconnect(IPin *ppin) = 0;
    virtual HRESULT WINAPI SetDefaultSyncSource(void) = 0;
  };
#else
  typedef struct IFilterGraphVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IFilterGraph *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IFilterGraph *This);
      ULONG (WINAPI *Release)(IFilterGraph *This);
      HRESULT (WINAPI *AddFilter)(IFilterGraph *This,IBaseFilter *pFilter,LPCWSTR pName);
      HRESULT (WINAPI *RemoveFilter)(IFilterGraph *This,IBaseFilter *pFilter);
      HRESULT (WINAPI *EnumFilters)(IFilterGraph *This,IEnumFilters **ppEnum);
      HRESULT (WINAPI *FindFilterByName)(IFilterGraph *This,LPCWSTR pName,IBaseFilter **ppFilter);
      HRESULT (WINAPI *ConnectDirect)(IFilterGraph *This,IPin *ppinOut,IPin *ppinIn,const AM_MEDIA_TYPE *pmt);
      HRESULT (WINAPI *Reconnect)(IFilterGraph *This,IPin *ppin);
      HRESULT (WINAPI *Disconnect)(IFilterGraph *This,IPin *ppin);
      HRESULT (WINAPI *SetDefaultSyncSource)(IFilterGraph *This);
    END_INTERFACE
  } IFilterGraphVtbl;
  struct IFilterGraph {
    CONST_VTBL struct IFilterGraphVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IFilterGraph_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IFilterGraph_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IFilterGraph_Release(This) (This)->lpVtbl->Release(This)
#define IFilterGraph_AddFilter(This,pFilter,pName) (This)->lpVtbl->AddFilter(This,pFilter,pName)
#define IFilterGraph_RemoveFilter(This,pFilter) (This)->lpVtbl->RemoveFilter(This,pFilter)
#define IFilterGraph_EnumFilters(This,ppEnum) (This)->lpVtbl->EnumFilters(This,ppEnum)
#define IFilterGraph_FindFilterByName(This,pName,ppFilter) (This)->lpVtbl->FindFilterByName(This,pName,ppFilter)
#define IFilterGraph_ConnectDirect(This,ppinOut,ppinIn,pmt) (This)->lpVtbl->ConnectDirect(This,ppinOut,ppinIn,pmt)
#define IFilterGraph_Reconnect(This,ppin) (This)->lpVtbl->Reconnect(This,ppin)
#define IFilterGraph_Disconnect(This,ppin) (This)->lpVtbl->Disconnect(This,ppin)
#define IFilterGraph_SetDefaultSyncSource(This) (This)->lpVtbl->SetDefaultSyncSource(This)
#endif
#endif
  HRESULT WINAPI IFilterGraph_AddFilter_Proxy(IFilterGraph *This,IBaseFilter *pFilter,LPCWSTR pName);
  void __RPC_STUB IFilterGraph_AddFilter_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterGraph_RemoveFilter_Proxy(IFilterGraph *This,IBaseFilter *pFilter);
  void __RPC_STUB IFilterGraph_RemoveFilter_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterGraph_EnumFilters_Proxy(IFilterGraph *This,IEnumFilters **ppEnum);
  void __RPC_STUB IFilterGraph_EnumFilters_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterGraph_FindFilterByName_Proxy(IFilterGraph *This,LPCWSTR pName,IBaseFilter **ppFilter);
  void __RPC_STUB IFilterGraph_FindFilterByName_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterGraph_ConnectDirect_Proxy(IFilterGraph *This,IPin *ppinOut,IPin *ppinIn,const AM_MEDIA_TYPE *pmt);
  void __RPC_STUB IFilterGraph_ConnectDirect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterGraph_Reconnect_Proxy(IFilterGraph *This,IPin *ppin);
  void __RPC_STUB IFilterGraph_Reconnect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterGraph_Disconnect_Proxy(IFilterGraph *This,IPin *ppin);
  void __RPC_STUB IFilterGraph_Disconnect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterGraph_SetDefaultSyncSource_Proxy(IFilterGraph *This);
  void __RPC_STUB IFilterGraph_SetDefaultSyncSource_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IFilterGraph *PFILTERGRAPH;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0121_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0121_v0_0_s_ifspec;
#ifndef __IEnumFilters_INTERFACE_DEFINED__
#define __IEnumFilters_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IEnumFilters;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IEnumFilters : public IUnknown {
  public:
    virtual HRESULT WINAPI Next(ULONG cFilters,IBaseFilter **ppFilter,ULONG *pcFetched) = 0;
    virtual HRESULT WINAPI Skip(ULONG cFilters) = 0;
    virtual HRESULT WINAPI Reset(void) = 0;
    virtual HRESULT WINAPI Clone(IEnumFilters **ppEnum) = 0;
  };
#else
  typedef struct IEnumFiltersVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IEnumFilters *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IEnumFilters *This);
      ULONG (WINAPI *Release)(IEnumFilters *This);
      HRESULT (WINAPI *Next)(IEnumFilters *This,ULONG cFilters,IBaseFilter **ppFilter,ULONG *pcFetched);
      HRESULT (WINAPI *Skip)(IEnumFilters *This,ULONG cFilters);
      HRESULT (WINAPI *Reset)(IEnumFilters *This);
      HRESULT (WINAPI *Clone)(IEnumFilters *This,IEnumFilters **ppEnum);
    END_INTERFACE
  } IEnumFiltersVtbl;
  struct IEnumFilters {
    CONST_VTBL struct IEnumFiltersVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IEnumFilters_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IEnumFilters_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IEnumFilters_Release(This) (This)->lpVtbl->Release(This)
#define IEnumFilters_Next(This,cFilters,ppFilter,pcFetched) (This)->lpVtbl->Next(This,cFilters,ppFilter,pcFetched)
#define IEnumFilters_Skip(This,cFilters) (This)->lpVtbl->Skip(This,cFilters)
#define IEnumFilters_Reset(This) (This)->lpVtbl->Reset(This)
#define IEnumFilters_Clone(This,ppEnum) (This)->lpVtbl->Clone(This,ppEnum)
#endif
#endif
  HRESULT WINAPI IEnumFilters_Next_Proxy(IEnumFilters *This,ULONG cFilters,IBaseFilter **ppFilter,ULONG *pcFetched);
  void __RPC_STUB IEnumFilters_Next_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEnumFilters_Skip_Proxy(IEnumFilters *This,ULONG cFilters);
  void __RPC_STUB IEnumFilters_Skip_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEnumFilters_Reset_Proxy(IEnumFilters *This);
  void __RPC_STUB IEnumFilters_Reset_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEnumFilters_Clone_Proxy(IEnumFilters *This,IEnumFilters **ppEnum);
  void __RPC_STUB IEnumFilters_Clone_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IEnumFilters *PENUMFILTERS;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0122_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0122_v0_0_s_ifspec;
#ifndef __IMediaFilter_INTERFACE_DEFINED__
#define __IMediaFilter_INTERFACE_DEFINED__
  typedef enum _FilterState {
    State_Stopped = 0,State_Paused,State_Running
  } FILTER_STATE;

  EXTERN_C const IID IID_IMediaFilter;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IMediaFilter : public IPersist {
  public:
    virtual HRESULT WINAPI Stop(void) = 0;
    virtual HRESULT WINAPI Pause(void) = 0;
    virtual HRESULT WINAPI Run(REFERENCE_TIME tStart) = 0;
    virtual HRESULT WINAPI GetState(DWORD dwMilliSecsTimeout,FILTER_STATE *State) = 0;
    virtual HRESULT WINAPI SetSyncSource(IReferenceClock *pClock) = 0;
    virtual HRESULT WINAPI GetSyncSource(IReferenceClock **pClock) = 0;
  };
#else
  typedef struct IMediaFilterVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IMediaFilter *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IMediaFilter *This);
      ULONG (WINAPI *Release)(IMediaFilter *This);
      HRESULT (WINAPI *GetClassID)(IMediaFilter *This,CLSID *pClassID);
      HRESULT (WINAPI *Stop)(IMediaFilter *This);
      HRESULT (WINAPI *Pause)(IMediaFilter *This);
      HRESULT (WINAPI *Run)(IMediaFilter *This,REFERENCE_TIME tStart);
      HRESULT (WINAPI *GetState)(IMediaFilter *This,DWORD dwMilliSecsTimeout,FILTER_STATE *State);
      HRESULT (WINAPI *SetSyncSource)(IMediaFilter *This,IReferenceClock *pClock);
      HRESULT (WINAPI *GetSyncSource)(IMediaFilter *This,IReferenceClock **pClock);
    END_INTERFACE
  } IMediaFilterVtbl;
  struct IMediaFilter {
    CONST_VTBL struct IMediaFilterVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IMediaFilter_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMediaFilter_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMediaFilter_Release(This) (This)->lpVtbl->Release(This)
#define IMediaFilter_GetClassID(This,pClassID) (This)->lpVtbl->GetClassID(This,pClassID)
#define IMediaFilter_Stop(This) (This)->lpVtbl->Stop(This)
#define IMediaFilter_Pause(This) (This)->lpVtbl->Pause(This)
#define IMediaFilter_Run(This,tStart) (This)->lpVtbl->Run(This,tStart)
#define IMediaFilter_GetState(This,dwMilliSecsTimeout,State) (This)->lpVtbl->GetState(This,dwMilliSecsTimeout,State)
#define IMediaFilter_SetSyncSource(This,pClock) (This)->lpVtbl->SetSyncSource(This,pClock)
#define IMediaFilter_GetSyncSource(This,pClock) (This)->lpVtbl->GetSyncSource(This,pClock)
#endif
#endif
  HRESULT WINAPI IMediaFilter_Stop_Proxy(IMediaFilter *This);
  void __RPC_STUB IMediaFilter_Stop_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaFilter_Pause_Proxy(IMediaFilter *This);
  void __RPC_STUB IMediaFilter_Pause_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaFilter_Run_Proxy(IMediaFilter *This,REFERENCE_TIME tStart);
  void __RPC_STUB IMediaFilter_Run_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaFilter_GetState_Proxy(IMediaFilter *This,DWORD dwMilliSecsTimeout,FILTER_STATE *State);
  void __RPC_STUB IMediaFilter_GetState_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaFilter_SetSyncSource_Proxy(IMediaFilter *This,IReferenceClock *pClock);
  void __RPC_STUB IMediaFilter_SetSyncSource_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaFilter_GetSyncSource_Proxy(IMediaFilter *This,IReferenceClock **pClock);
  void __RPC_STUB IMediaFilter_GetSyncSource_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IMediaFilter *PMEDIAFILTER;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0123_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0123_v0_0_s_ifspec;
#ifndef __IBaseFilter_INTERFACE_DEFINED__
#define __IBaseFilter_INTERFACE_DEFINED__

  typedef struct _FilterInfo {
    WCHAR achName[128];
    IFilterGraph *pGraph;
  } FILTER_INFO;

  EXTERN_C const IID IID_IBaseFilter;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IBaseFilter : public IMediaFilter {
  public:
    virtual HRESULT WINAPI EnumPins(IEnumPins **ppEnum) = 0;
    virtual HRESULT WINAPI FindPin(LPCWSTR Id,IPin **ppPin) = 0;
    virtual HRESULT WINAPI QueryFilterInfo(FILTER_INFO *pInfo) = 0;
    virtual HRESULT WINAPI JoinFilterGraph(IFilterGraph *pGraph,LPCWSTR pName) = 0;
    virtual HRESULT WINAPI QueryVendorInfo(LPWSTR *pVendorInfo) = 0;
  };
#else
  typedef struct IBaseFilterVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IBaseFilter *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IBaseFilter *This);
      ULONG (WINAPI *Release)(IBaseFilter *This);
      HRESULT (WINAPI *GetClassID)(IBaseFilter *This,CLSID *pClassID);
      HRESULT (WINAPI *Stop)(IBaseFilter *This);
      HRESULT (WINAPI *Pause)(IBaseFilter *This);
      HRESULT (WINAPI *Run)(IBaseFilter *This,REFERENCE_TIME tStart);
      HRESULT (WINAPI *GetState)(IBaseFilter *This,DWORD dwMilliSecsTimeout,FILTER_STATE *State);
      HRESULT (WINAPI *SetSyncSource)(IBaseFilter *This,IReferenceClock *pClock);
      HRESULT (WINAPI *GetSyncSource)(IBaseFilter *This,IReferenceClock **pClock);
      HRESULT (WINAPI *EnumPins)(IBaseFilter *This,IEnumPins **ppEnum);
      HRESULT (WINAPI *FindPin)(IBaseFilter *This,LPCWSTR Id,IPin **ppPin);
      HRESULT (WINAPI *QueryFilterInfo)(IBaseFilter *This,FILTER_INFO *pInfo);
      HRESULT (WINAPI *JoinFilterGraph)(IBaseFilter *This,IFilterGraph *pGraph,LPCWSTR pName);
      HRESULT (WINAPI *QueryVendorInfo)(IBaseFilter *This,LPWSTR *pVendorInfo);
    END_INTERFACE
  } IBaseFilterVtbl;
  struct IBaseFilter {
    CONST_VTBL struct IBaseFilterVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IBaseFilter_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IBaseFilter_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IBaseFilter_Release(This) (This)->lpVtbl->Release(This)
#define IBaseFilter_GetClassID(This,pClassID) (This)->lpVtbl->GetClassID(This,pClassID)
#define IBaseFilter_Stop(This) (This)->lpVtbl->Stop(This)
#define IBaseFilter_Pause(This) (This)->lpVtbl->Pause(This)
#define IBaseFilter_Run(This,tStart) (This)->lpVtbl->Run(This,tStart)
#define IBaseFilter_GetState(This,dwMilliSecsTimeout,State) (This)->lpVtbl->GetState(This,dwMilliSecsTimeout,State)
#define IBaseFilter_SetSyncSource(This,pClock) (This)->lpVtbl->SetSyncSource(This,pClock)
#define IBaseFilter_GetSyncSource(This,pClock) (This)->lpVtbl->GetSyncSource(This,pClock)
#define IBaseFilter_EnumPins(This,ppEnum) (This)->lpVtbl->EnumPins(This,ppEnum)
#define IBaseFilter_FindPin(This,Id,ppPin) (This)->lpVtbl->FindPin(This,Id,ppPin)
#define IBaseFilter_QueryFilterInfo(This,pInfo) (This)->lpVtbl->QueryFilterInfo(This,pInfo)
#define IBaseFilter_JoinFilterGraph(This,pGraph,pName) (This)->lpVtbl->JoinFilterGraph(This,pGraph,pName)
#define IBaseFilter_QueryVendorInfo(This,pVendorInfo) (This)->lpVtbl->QueryVendorInfo(This,pVendorInfo)
#endif
#endif
  HRESULT WINAPI IBaseFilter_EnumPins_Proxy(IBaseFilter *This,IEnumPins **ppEnum);
  void __RPC_STUB IBaseFilter_EnumPins_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBaseFilter_FindPin_Proxy(IBaseFilter *This,LPCWSTR Id,IPin **ppPin);
  void __RPC_STUB IBaseFilter_FindPin_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBaseFilter_QueryFilterInfo_Proxy(IBaseFilter *This,FILTER_INFO *pInfo);
  void __RPC_STUB IBaseFilter_QueryFilterInfo_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBaseFilter_JoinFilterGraph_Proxy(IBaseFilter *This,IFilterGraph *pGraph,LPCWSTR pName);
  void __RPC_STUB IBaseFilter_JoinFilterGraph_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBaseFilter_QueryVendorInfo_Proxy(IBaseFilter *This,LPWSTR *pVendorInfo);
  void __RPC_STUB IBaseFilter_QueryVendorInfo_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IBaseFilter *PFILTER;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0124_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0124_v0_0_s_ifspec;
#ifndef __IReferenceClock_INTERFACE_DEFINED__
#define __IReferenceClock_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IReferenceClock;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IReferenceClock : public IUnknown {
  public:
    virtual HRESULT WINAPI GetTime(REFERENCE_TIME *pTime) = 0;
    virtual HRESULT WINAPI AdviseTime(REFERENCE_TIME baseTime,REFERENCE_TIME streamTime,HEVENT hEvent,DWORD_PTR *pdwAdviseCookie) = 0;
    virtual HRESULT WINAPI AdvisePeriodic(REFERENCE_TIME startTime,REFERENCE_TIME periodTime,HSEMAPHORE hSemaphore,DWORD_PTR *pdwAdviseCookie) = 0;
    virtual HRESULT WINAPI Unadvise(DWORD_PTR dwAdviseCookie) = 0;
  };
#else
  typedef struct IReferenceClockVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IReferenceClock *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IReferenceClock *This);
      ULONG (WINAPI *Release)(IReferenceClock *This);
      HRESULT (WINAPI *GetTime)(IReferenceClock *This,REFERENCE_TIME *pTime);
      HRESULT (WINAPI *AdviseTime)(IReferenceClock *This,REFERENCE_TIME baseTime,REFERENCE_TIME streamTime,HEVENT hEvent,DWORD_PTR *pdwAdviseCookie);
      HRESULT (WINAPI *AdvisePeriodic)(IReferenceClock *This,REFERENCE_TIME startTime,REFERENCE_TIME periodTime,HSEMAPHORE hSemaphore,DWORD_PTR *pdwAdviseCookie);
      HRESULT (WINAPI *Unadvise)(IReferenceClock *This,DWORD_PTR dwAdviseCookie);
    END_INTERFACE
  } IReferenceClockVtbl;
  struct IReferenceClock {
    CONST_VTBL struct IReferenceClockVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IReferenceClock_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IReferenceClock_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IReferenceClock_Release(This) (This)->lpVtbl->Release(This)
#define IReferenceClock_GetTime(This,pTime) (This)->lpVtbl->GetTime(This,pTime)
#define IReferenceClock_AdviseTime(This,baseTime,streamTime,hEvent,pdwAdviseCookie) (This)->lpVtbl->AdviseTime(This,baseTime,streamTime,hEvent,pdwAdviseCookie)
#define IReferenceClock_AdvisePeriodic(This,startTime,periodTime,hSemaphore,pdwAdviseCookie) (This)->lpVtbl->AdvisePeriodic(This,startTime,periodTime,hSemaphore,pdwAdviseCookie)
#define IReferenceClock_Unadvise(This,dwAdviseCookie) (This)->lpVtbl->Unadvise(This,dwAdviseCookie)
#endif
#endif
  HRESULT WINAPI IReferenceClock_GetTime_Proxy(IReferenceClock *This,REFERENCE_TIME *pTime);
  void __RPC_STUB IReferenceClock_GetTime_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IReferenceClock_AdviseTime_Proxy(IReferenceClock *This,REFERENCE_TIME baseTime,REFERENCE_TIME streamTime,HEVENT hEvent,DWORD_PTR *pdwAdviseCookie);
  void __RPC_STUB IReferenceClock_AdviseTime_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IReferenceClock_AdvisePeriodic_Proxy(IReferenceClock *This,REFERENCE_TIME startTime,REFERENCE_TIME periodTime,HSEMAPHORE hSemaphore,DWORD_PTR *pdwAdviseCookie);
  void __RPC_STUB IReferenceClock_AdvisePeriodic_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IReferenceClock_Unadvise_Proxy(IReferenceClock *This,DWORD_PTR dwAdviseCookie);
  void __RPC_STUB IReferenceClock_Unadvise_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IReferenceClock *PREFERENCECLOCK;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0125_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0125_v0_0_s_ifspec;
#ifndef __IReferenceClock2_INTERFACE_DEFINED__
#define __IReferenceClock2_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IReferenceClock2;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IReferenceClock2 : public IReferenceClock {
  };
#else
  typedef struct IReferenceClock2Vtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IReferenceClock2 *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IReferenceClock2 *This);
      ULONG (WINAPI *Release)(IReferenceClock2 *This);
      HRESULT (WINAPI *GetTime)(IReferenceClock2 *This,REFERENCE_TIME *pTime);
      HRESULT (WINAPI *AdviseTime)(IReferenceClock2 *This,REFERENCE_TIME baseTime,REFERENCE_TIME streamTime,HEVENT hEvent,DWORD_PTR *pdwAdviseCookie);
      HRESULT (WINAPI *AdvisePeriodic)(IReferenceClock2 *This,REFERENCE_TIME startTime,REFERENCE_TIME periodTime,HSEMAPHORE hSemaphore,DWORD_PTR *pdwAdviseCookie);
      HRESULT (WINAPI *Unadvise)(IReferenceClock2 *This,DWORD_PTR dwAdviseCookie);
    END_INTERFACE
  } IReferenceClock2Vtbl;
  struct IReferenceClock2 {
    CONST_VTBL struct IReferenceClock2Vtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IReferenceClock2_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IReferenceClock2_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IReferenceClock2_Release(This) (This)->lpVtbl->Release(This)
#define IReferenceClock2_GetTime(This,pTime) (This)->lpVtbl->GetTime(This,pTime)
#define IReferenceClock2_AdviseTime(This,baseTime,streamTime,hEvent,pdwAdviseCookie) (This)->lpVtbl->AdviseTime(This,baseTime,streamTime,hEvent,pdwAdviseCookie)
#define IReferenceClock2_AdvisePeriodic(This,startTime,periodTime,hSemaphore,pdwAdviseCookie) (This)->lpVtbl->AdvisePeriodic(This,startTime,periodTime,hSemaphore,pdwAdviseCookie)
#define IReferenceClock2_Unadvise(This,dwAdviseCookie) (This)->lpVtbl->Unadvise(This,dwAdviseCookie)
#endif
#endif
#endif

  typedef IReferenceClock2 *PREFERENCECLOCK2;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0126_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0126_v0_0_s_ifspec;
#ifndef __IMediaSample_INTERFACE_DEFINED__
#define __IMediaSample_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IMediaSample;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IMediaSample : public IUnknown {
  public:
    virtual HRESULT WINAPI GetPointer(BYTE **ppBuffer) = 0;
    virtual long WINAPI GetSize(void) = 0;
    virtual HRESULT WINAPI GetTime(REFERENCE_TIME *pTimeStart,REFERENCE_TIME *pTimeEnd) = 0;
    virtual HRESULT WINAPI SetTime(REFERENCE_TIME *pTimeStart,REFERENCE_TIME *pTimeEnd) = 0;
    virtual HRESULT WINAPI IsSyncPoint(void) = 0;
    virtual HRESULT WINAPI SetSyncPoint(WINBOOL bIsSyncPoint) = 0;
    virtual HRESULT WINAPI IsPreroll(void) = 0;
    virtual HRESULT WINAPI SetPreroll(WINBOOL bIsPreroll) = 0;
    virtual long WINAPI GetActualDataLength(void) = 0;
    virtual HRESULT WINAPI SetActualDataLength(long __MIDL_0010) = 0;
    virtual HRESULT WINAPI GetMediaType(AM_MEDIA_TYPE **ppMediaType) = 0;
    virtual HRESULT WINAPI SetMediaType(AM_MEDIA_TYPE *pMediaType) = 0;
    virtual HRESULT WINAPI IsDiscontinuity(void) = 0;
    virtual HRESULT WINAPI SetDiscontinuity(WINBOOL bDiscontinuity) = 0;
    virtual HRESULT WINAPI GetMediaTime(LONGLONG *pTimeStart,LONGLONG *pTimeEnd) = 0;
    virtual HRESULT WINAPI SetMediaTime(LONGLONG *pTimeStart,LONGLONG *pTimeEnd) = 0;
  };
#else
  typedef struct IMediaSampleVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IMediaSample *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IMediaSample *This);
      ULONG (WINAPI *Release)(IMediaSample *This);
      HRESULT (WINAPI *GetPointer)(IMediaSample *This,BYTE **ppBuffer);
      long (WINAPI *GetSize)(IMediaSample *This);
      HRESULT (WINAPI *GetTime)(IMediaSample *This,REFERENCE_TIME *pTimeStart,REFERENCE_TIME *pTimeEnd);
      HRESULT (WINAPI *SetTime)(IMediaSample *This,REFERENCE_TIME *pTimeStart,REFERENCE_TIME *pTimeEnd);
      HRESULT (WINAPI *IsSyncPoint)(IMediaSample *This);
      HRESULT (WINAPI *SetSyncPoint)(IMediaSample *This,WINBOOL bIsSyncPoint);
      HRESULT (WINAPI *IsPreroll)(IMediaSample *This);
      HRESULT (WINAPI *SetPreroll)(IMediaSample *This,WINBOOL bIsPreroll);
      long (WINAPI *GetActualDataLength)(IMediaSample *This);
      HRESULT (WINAPI *SetActualDataLength)(IMediaSample *This,long __MIDL_0010);
      HRESULT (WINAPI *GetMediaType)(IMediaSample *This,AM_MEDIA_TYPE **ppMediaType);
      HRESULT (WINAPI *SetMediaType)(IMediaSample *This,AM_MEDIA_TYPE *pMediaType);
      HRESULT (WINAPI *IsDiscontinuity)(IMediaSample *This);
      HRESULT (WINAPI *SetDiscontinuity)(IMediaSample *This,WINBOOL bDiscontinuity);
      HRESULT (WINAPI *GetMediaTime)(IMediaSample *This,LONGLONG *pTimeStart,LONGLONG *pTimeEnd);
      HRESULT (WINAPI *SetMediaTime)(IMediaSample *This,LONGLONG *pTimeStart,LONGLONG *pTimeEnd);
    END_INTERFACE
  } IMediaSampleVtbl;
  struct IMediaSample {
    CONST_VTBL struct IMediaSampleVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IMediaSample_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMediaSample_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMediaSample_Release(This) (This)->lpVtbl->Release(This)
#define IMediaSample_GetPointer(This,ppBuffer) (This)->lpVtbl->GetPointer(This,ppBuffer)
#define IMediaSample_GetSize(This) (This)->lpVtbl->GetSize(This)
#define IMediaSample_GetTime(This,pTimeStart,pTimeEnd) (This)->lpVtbl->GetTime(This,pTimeStart,pTimeEnd)
#define IMediaSample_SetTime(This,pTimeStart,pTimeEnd) (This)->lpVtbl->SetTime(This,pTimeStart,pTimeEnd)
#define IMediaSample_IsSyncPoint(This) (This)->lpVtbl->IsSyncPoint(This)
#define IMediaSample_SetSyncPoint(This,bIsSyncPoint) (This)->lpVtbl->SetSyncPoint(This,bIsSyncPoint)
#define IMediaSample_IsPreroll(This) (This)->lpVtbl->IsPreroll(This)
#define IMediaSample_SetPreroll(This,bIsPreroll) (This)->lpVtbl->SetPreroll(This,bIsPreroll)
#define IMediaSample_GetActualDataLength(This) (This)->lpVtbl->GetActualDataLength(This)
#define IMediaSample_SetActualDataLength(This,__MIDL_0010) (This)->lpVtbl->SetActualDataLength(This,__MIDL_0010)
#define IMediaSample_GetMediaType(This,ppMediaType) (This)->lpVtbl->GetMediaType(This,ppMediaType)
#define IMediaSample_SetMediaType(This,pMediaType) (This)->lpVtbl->SetMediaType(This,pMediaType)
#define IMediaSample_IsDiscontinuity(This) (This)->lpVtbl->IsDiscontinuity(This)
#define IMediaSample_SetDiscontinuity(This,bDiscontinuity) (This)->lpVtbl->SetDiscontinuity(This,bDiscontinuity)
#define IMediaSample_GetMediaTime(This,pTimeStart,pTimeEnd) (This)->lpVtbl->GetMediaTime(This,pTimeStart,pTimeEnd)
#define IMediaSample_SetMediaTime(This,pTimeStart,pTimeEnd) (This)->lpVtbl->SetMediaTime(This,pTimeStart,pTimeEnd)
#endif
#endif
  HRESULT WINAPI IMediaSample_GetPointer_Proxy(IMediaSample *This,BYTE **ppBuffer);
  void __RPC_STUB IMediaSample_GetPointer_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  long WINAPI IMediaSample_GetSize_Proxy(IMediaSample *This);
  void __RPC_STUB IMediaSample_GetSize_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSample_GetTime_Proxy(IMediaSample *This,REFERENCE_TIME *pTimeStart,REFERENCE_TIME *pTimeEnd);
  void __RPC_STUB IMediaSample_GetTime_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSample_SetTime_Proxy(IMediaSample *This,REFERENCE_TIME *pTimeStart,REFERENCE_TIME *pTimeEnd);
  void __RPC_STUB IMediaSample_SetTime_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSample_IsSyncPoint_Proxy(IMediaSample *This);
  void __RPC_STUB IMediaSample_IsSyncPoint_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSample_SetSyncPoint_Proxy(IMediaSample *This,WINBOOL bIsSyncPoint);
  void __RPC_STUB IMediaSample_SetSyncPoint_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSample_IsPreroll_Proxy(IMediaSample *This);
  void __RPC_STUB IMediaSample_IsPreroll_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSample_SetPreroll_Proxy(IMediaSample *This,WINBOOL bIsPreroll);
  void __RPC_STUB IMediaSample_SetPreroll_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  long WINAPI IMediaSample_GetActualDataLength_Proxy(IMediaSample *This);
  void __RPC_STUB IMediaSample_GetActualDataLength_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSample_SetActualDataLength_Proxy(IMediaSample *This,long __MIDL_0010);
  void __RPC_STUB IMediaSample_SetActualDataLength_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSample_GetMediaType_Proxy(IMediaSample *This,AM_MEDIA_TYPE **ppMediaType);
  void __RPC_STUB IMediaSample_GetMediaType_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSample_SetMediaType_Proxy(IMediaSample *This,AM_MEDIA_TYPE *pMediaType);
  void __RPC_STUB IMediaSample_SetMediaType_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSample_IsDiscontinuity_Proxy(IMediaSample *This);
  void __RPC_STUB IMediaSample_IsDiscontinuity_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSample_SetDiscontinuity_Proxy(IMediaSample *This,WINBOOL bDiscontinuity);
  void __RPC_STUB IMediaSample_SetDiscontinuity_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSample_GetMediaTime_Proxy(IMediaSample *This,LONGLONG *pTimeStart,LONGLONG *pTimeEnd);
  void __RPC_STUB IMediaSample_GetMediaTime_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSample_SetMediaTime_Proxy(IMediaSample *This,LONGLONG *pTimeStart,LONGLONG *pTimeEnd);
  void __RPC_STUB IMediaSample_SetMediaTime_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IMediaSample *PMEDIASAMPLE;

  enum tagAM_SAMPLE_PROPERTY_FLAGS {
    AM_SAMPLE_SPLICEPOINT = 0x1,AM_SAMPLE_PREROLL = 0x2,AM_SAMPLE_DATADISCONTINUITY = 0x4,AM_SAMPLE_TYPECHANGED = 0x8,AM_SAMPLE_TIMEVALID = 0x10,
    AM_SAMPLE_TIMEDISCONTINUITY = 0x40,AM_SAMPLE_FLUSH_ON_PAUSE = 0x80,AM_SAMPLE_STOPVALID = 0x100,AM_SAMPLE_ENDOFSTREAM = 0x200,AM_STREAM_MEDIA = 0,
    AM_STREAM_CONTROL = 1
  };
  typedef struct tagAM_SAMPLE2_PROPERTIES {
    DWORD cbData;
    DWORD dwTypeSpecificFlags;
    DWORD dwSampleFlags;
    LONG lActual;
    REFERENCE_TIME tStart;
    REFERENCE_TIME tStop;
    DWORD dwStreamId;
    AM_MEDIA_TYPE *pMediaType;
    BYTE *pbBuffer;
    LONG cbBuffer;
  } AM_SAMPLE2_PROPERTIES;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0127_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0127_v0_0_s_ifspec;
#ifndef __IMediaSample2_INTERFACE_DEFINED__
#define __IMediaSample2_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IMediaSample2;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IMediaSample2 : public IMediaSample {
  public:
    virtual HRESULT WINAPI GetProperties(DWORD cbProperties,BYTE *pbProperties) = 0;
    virtual HRESULT WINAPI SetProperties(DWORD cbProperties,const BYTE *pbProperties) = 0;
  };
#else
  typedef struct IMediaSample2Vtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IMediaSample2 *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IMediaSample2 *This);
      ULONG (WINAPI *Release)(IMediaSample2 *This);
      HRESULT (WINAPI *GetPointer)(IMediaSample2 *This,BYTE **ppBuffer);
      long (WINAPI *GetSize)(IMediaSample2 *This);
      HRESULT (WINAPI *GetTime)(IMediaSample2 *This,REFERENCE_TIME *pTimeStart,REFERENCE_TIME *pTimeEnd);
      HRESULT (WINAPI *SetTime)(IMediaSample2 *This,REFERENCE_TIME *pTimeStart,REFERENCE_TIME *pTimeEnd);
      HRESULT (WINAPI *IsSyncPoint)(IMediaSample2 *This);
      HRESULT (WINAPI *SetSyncPoint)(IMediaSample2 *This,WINBOOL bIsSyncPoint);
      HRESULT (WINAPI *IsPreroll)(IMediaSample2 *This);
      HRESULT (WINAPI *SetPreroll)(IMediaSample2 *This,WINBOOL bIsPreroll);
      long (WINAPI *GetActualDataLength)(IMediaSample2 *This);
      HRESULT (WINAPI *SetActualDataLength)(IMediaSample2 *This,long __MIDL_0010);
      HRESULT (WINAPI *GetMediaType)(IMediaSample2 *This,AM_MEDIA_TYPE **ppMediaType);
      HRESULT (WINAPI *SetMediaType)(IMediaSample2 *This,AM_MEDIA_TYPE *pMediaType);
      HRESULT (WINAPI *IsDiscontinuity)(IMediaSample2 *This);
      HRESULT (WINAPI *SetDiscontinuity)(IMediaSample2 *This,WINBOOL bDiscontinuity);
      HRESULT (WINAPI *GetMediaTime)(IMediaSample2 *This,LONGLONG *pTimeStart,LONGLONG *pTimeEnd);
      HRESULT (WINAPI *SetMediaTime)(IMediaSample2 *This,LONGLONG *pTimeStart,LONGLONG *pTimeEnd);
      HRESULT (WINAPI *GetProperties)(IMediaSample2 *This,DWORD cbProperties,BYTE *pbProperties);
      HRESULT (WINAPI *SetProperties)(IMediaSample2 *This,DWORD cbProperties,const BYTE *pbProperties);
    END_INTERFACE
  } IMediaSample2Vtbl;
  struct IMediaSample2 {
    CONST_VTBL struct IMediaSample2Vtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IMediaSample2_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMediaSample2_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMediaSample2_Release(This) (This)->lpVtbl->Release(This)
#define IMediaSample2_GetPointer(This,ppBuffer) (This)->lpVtbl->GetPointer(This,ppBuffer)
#define IMediaSample2_GetSize(This) (This)->lpVtbl->GetSize(This)
#define IMediaSample2_GetTime(This,pTimeStart,pTimeEnd) (This)->lpVtbl->GetTime(This,pTimeStart,pTimeEnd)
#define IMediaSample2_SetTime(This,pTimeStart,pTimeEnd) (This)->lpVtbl->SetTime(This,pTimeStart,pTimeEnd)
#define IMediaSample2_IsSyncPoint(This) (This)->lpVtbl->IsSyncPoint(This)
#define IMediaSample2_SetSyncPoint(This,bIsSyncPoint) (This)->lpVtbl->SetSyncPoint(This,bIsSyncPoint)
#define IMediaSample2_IsPreroll(This) (This)->lpVtbl->IsPreroll(This)
#define IMediaSample2_SetPreroll(This,bIsPreroll) (This)->lpVtbl->SetPreroll(This,bIsPreroll)
#define IMediaSample2_GetActualDataLength(This) (This)->lpVtbl->GetActualDataLength(This)
#define IMediaSample2_SetActualDataLength(This,__MIDL_0010) (This)->lpVtbl->SetActualDataLength(This,__MIDL_0010)
#define IMediaSample2_GetMediaType(This,ppMediaType) (This)->lpVtbl->GetMediaType(This,ppMediaType)
#define IMediaSample2_SetMediaType(This,pMediaType) (This)->lpVtbl->SetMediaType(This,pMediaType)
#define IMediaSample2_IsDiscontinuity(This) (This)->lpVtbl->IsDiscontinuity(This)
#define IMediaSample2_SetDiscontinuity(This,bDiscontinuity) (This)->lpVtbl->SetDiscontinuity(This,bDiscontinuity)
#define IMediaSample2_GetMediaTime(This,pTimeStart,pTimeEnd) (This)->lpVtbl->GetMediaTime(This,pTimeStart,pTimeEnd)
#define IMediaSample2_SetMediaTime(This,pTimeStart,pTimeEnd) (This)->lpVtbl->SetMediaTime(This,pTimeStart,pTimeEnd)
#define IMediaSample2_GetProperties(This,cbProperties,pbProperties) (This)->lpVtbl->GetProperties(This,cbProperties,pbProperties)
#define IMediaSample2_SetProperties(This,cbProperties,pbProperties) (This)->lpVtbl->SetProperties(This,cbProperties,pbProperties)
#endif
#endif
  HRESULT WINAPI IMediaSample2_GetProperties_Proxy(IMediaSample2 *This,DWORD cbProperties,BYTE *pbProperties);
  void __RPC_STUB IMediaSample2_GetProperties_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSample2_SetProperties_Proxy(IMediaSample2 *This,DWORD cbProperties,const BYTE *pbProperties);
  void __RPC_STUB IMediaSample2_SetProperties_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IMediaSample2 *PMEDIASAMPLE2;

#define AM_GBF_PREVFRAMESKIPPED 1
#define AM_GBF_NOTASYNCPOINT 2
#define AM_GBF_NOWAIT 4
#define AM_GBF_NODDSURFACELOCK 8

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0128_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0128_v0_0_s_ifspec;
#ifndef __IMemAllocator_INTERFACE_DEFINED__
#define __IMemAllocator_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IMemAllocator;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IMemAllocator : public IUnknown {
  public:
    virtual HRESULT WINAPI SetProperties(ALLOCATOR_PROPERTIES *pRequest,ALLOCATOR_PROPERTIES *pActual) = 0;
    virtual HRESULT WINAPI GetProperties(ALLOCATOR_PROPERTIES *pProps) = 0;
    virtual HRESULT WINAPI Commit(void) = 0;
    virtual HRESULT WINAPI Decommit(void) = 0;
    virtual HRESULT WINAPI GetBuffer(IMediaSample **ppBuffer,REFERENCE_TIME *pStartTime,REFERENCE_TIME *pEndTime,DWORD dwFlags) = 0;
    virtual HRESULT WINAPI ReleaseBuffer(IMediaSample *pBuffer) = 0;
  };
#else
  typedef struct IMemAllocatorVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IMemAllocator *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IMemAllocator *This);
      ULONG (WINAPI *Release)(IMemAllocator *This);
      HRESULT (WINAPI *SetProperties)(IMemAllocator *This,ALLOCATOR_PROPERTIES *pRequest,ALLOCATOR_PROPERTIES *pActual);
      HRESULT (WINAPI *GetProperties)(IMemAllocator *This,ALLOCATOR_PROPERTIES *pProps);
      HRESULT (WINAPI *Commit)(IMemAllocator *This);
      HRESULT (WINAPI *Decommit)(IMemAllocator *This);
      HRESULT (WINAPI *GetBuffer)(IMemAllocator *This,IMediaSample **ppBuffer,REFERENCE_TIME *pStartTime,REFERENCE_TIME *pEndTime,DWORD dwFlags);
      HRESULT (WINAPI *ReleaseBuffer)(IMemAllocator *This,IMediaSample *pBuffer);
    END_INTERFACE
  } IMemAllocatorVtbl;
  struct IMemAllocator {
    CONST_VTBL struct IMemAllocatorVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IMemAllocator_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMemAllocator_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMemAllocator_Release(This) (This)->lpVtbl->Release(This)
#define IMemAllocator_SetProperties(This,pRequest,pActual) (This)->lpVtbl->SetProperties(This,pRequest,pActual)
#define IMemAllocator_GetProperties(This,pProps) (This)->lpVtbl->GetProperties(This,pProps)
#define IMemAllocator_Commit(This) (This)->lpVtbl->Commit(This)
#define IMemAllocator_Decommit(This) (This)->lpVtbl->Decommit(This)
#define IMemAllocator_GetBuffer(This,ppBuffer,pStartTime,pEndTime,dwFlags) (This)->lpVtbl->GetBuffer(This,ppBuffer,pStartTime,pEndTime,dwFlags)
#define IMemAllocator_ReleaseBuffer(This,pBuffer) (This)->lpVtbl->ReleaseBuffer(This,pBuffer)
#endif
#endif
  HRESULT WINAPI IMemAllocator_SetProperties_Proxy(IMemAllocator *This,ALLOCATOR_PROPERTIES *pRequest,ALLOCATOR_PROPERTIES *pActual);
  void __RPC_STUB IMemAllocator_SetProperties_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMemAllocator_GetProperties_Proxy(IMemAllocator *This,ALLOCATOR_PROPERTIES *pProps);
  void __RPC_STUB IMemAllocator_GetProperties_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMemAllocator_Commit_Proxy(IMemAllocator *This);
  void __RPC_STUB IMemAllocator_Commit_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMemAllocator_Decommit_Proxy(IMemAllocator *This);
  void __RPC_STUB IMemAllocator_Decommit_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMemAllocator_GetBuffer_Proxy(IMemAllocator *This,IMediaSample **ppBuffer,REFERENCE_TIME *pStartTime,REFERENCE_TIME *pEndTime,DWORD dwFlags);
  void __RPC_STUB IMemAllocator_GetBuffer_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMemAllocator_ReleaseBuffer_Proxy(IMemAllocator *This,IMediaSample *pBuffer);
  void __RPC_STUB IMemAllocator_ReleaseBuffer_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IMemAllocator *PMEMALLOCATOR;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0129_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0129_v0_0_s_ifspec;
#ifndef __IMemAllocatorCallbackTemp_INTERFACE_DEFINED__
#define __IMemAllocatorCallbackTemp_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IMemAllocatorCallbackTemp;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IMemAllocatorCallbackTemp : public IMemAllocator {
  public:
    virtual HRESULT WINAPI SetNotify(IMemAllocatorNotifyCallbackTemp *pNotify) = 0;
    virtual HRESULT WINAPI GetFreeCount(LONG *plBuffersFree) = 0;
  };
#else
  typedef struct IMemAllocatorCallbackTempVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IMemAllocatorCallbackTemp *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IMemAllocatorCallbackTemp *This);
      ULONG (WINAPI *Release)(IMemAllocatorCallbackTemp *This);
      HRESULT (WINAPI *SetProperties)(IMemAllocatorCallbackTemp *This,ALLOCATOR_PROPERTIES *pRequest,ALLOCATOR_PROPERTIES *pActual);
      HRESULT (WINAPI *GetProperties)(IMemAllocatorCallbackTemp *This,ALLOCATOR_PROPERTIES *pProps);
      HRESULT (WINAPI *Commit)(IMemAllocatorCallbackTemp *This);
      HRESULT (WINAPI *Decommit)(IMemAllocatorCallbackTemp *This);
      HRESULT (WINAPI *GetBuffer)(IMemAllocatorCallbackTemp *This,IMediaSample **ppBuffer,REFERENCE_TIME *pStartTime,REFERENCE_TIME *pEndTime,DWORD dwFlags);
      HRESULT (WINAPI *ReleaseBuffer)(IMemAllocatorCallbackTemp *This,IMediaSample *pBuffer);
      HRESULT (WINAPI *SetNotify)(IMemAllocatorCallbackTemp *This,IMemAllocatorNotifyCallbackTemp *pNotify);
      HRESULT (WINAPI *GetFreeCount)(IMemAllocatorCallbackTemp *This,LONG *plBuffersFree);
    END_INTERFACE
  } IMemAllocatorCallbackTempVtbl;
  struct IMemAllocatorCallbackTemp {
    CONST_VTBL struct IMemAllocatorCallbackTempVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IMemAllocatorCallbackTemp_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMemAllocatorCallbackTemp_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMemAllocatorCallbackTemp_Release(This) (This)->lpVtbl->Release(This)
#define IMemAllocatorCallbackTemp_SetProperties(This,pRequest,pActual) (This)->lpVtbl->SetProperties(This,pRequest,pActual)
#define IMemAllocatorCallbackTemp_GetProperties(This,pProps) (This)->lpVtbl->GetProperties(This,pProps)
#define IMemAllocatorCallbackTemp_Commit(This) (This)->lpVtbl->Commit(This)
#define IMemAllocatorCallbackTemp_Decommit(This) (This)->lpVtbl->Decommit(This)
#define IMemAllocatorCallbackTemp_GetBuffer(This,ppBuffer,pStartTime,pEndTime,dwFlags) (This)->lpVtbl->GetBuffer(This,ppBuffer,pStartTime,pEndTime,dwFlags)
#define IMemAllocatorCallbackTemp_ReleaseBuffer(This,pBuffer) (This)->lpVtbl->ReleaseBuffer(This,pBuffer)
#define IMemAllocatorCallbackTemp_SetNotify(This,pNotify) (This)->lpVtbl->SetNotify(This,pNotify)
#define IMemAllocatorCallbackTemp_GetFreeCount(This,plBuffersFree) (This)->lpVtbl->GetFreeCount(This,plBuffersFree)
#endif
#endif
  HRESULT WINAPI IMemAllocatorCallbackTemp_SetNotify_Proxy(IMemAllocatorCallbackTemp *This,IMemAllocatorNotifyCallbackTemp *pNotify);
  void __RPC_STUB IMemAllocatorCallbackTemp_SetNotify_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMemAllocatorCallbackTemp_GetFreeCount_Proxy(IMemAllocatorCallbackTemp *This,LONG *plBuffersFree);
  void __RPC_STUB IMemAllocatorCallbackTemp_GetFreeCount_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IMemAllocatorNotifyCallbackTemp_INTERFACE_DEFINED__
#define __IMemAllocatorNotifyCallbackTemp_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IMemAllocatorNotifyCallbackTemp;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IMemAllocatorNotifyCallbackTemp : public IUnknown {
  public:
    virtual HRESULT WINAPI NotifyRelease(void) = 0;
  };
#else
  typedef struct IMemAllocatorNotifyCallbackTempVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IMemAllocatorNotifyCallbackTemp *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IMemAllocatorNotifyCallbackTemp *This);
      ULONG (WINAPI *Release)(IMemAllocatorNotifyCallbackTemp *This);
      HRESULT (WINAPI *NotifyRelease)(IMemAllocatorNotifyCallbackTemp *This);
    END_INTERFACE
  } IMemAllocatorNotifyCallbackTempVtbl;
  struct IMemAllocatorNotifyCallbackTemp {
    CONST_VTBL struct IMemAllocatorNotifyCallbackTempVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IMemAllocatorNotifyCallbackTemp_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMemAllocatorNotifyCallbackTemp_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMemAllocatorNotifyCallbackTemp_Release(This) (This)->lpVtbl->Release(This)
#define IMemAllocatorNotifyCallbackTemp_NotifyRelease(This) (This)->lpVtbl->NotifyRelease(This)
#endif
#endif
  HRESULT WINAPI IMemAllocatorNotifyCallbackTemp_NotifyRelease_Proxy(IMemAllocatorNotifyCallbackTemp *This);
  void __RPC_STUB IMemAllocatorNotifyCallbackTemp_NotifyRelease_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IMemInputPin_INTERFACE_DEFINED__
#define __IMemInputPin_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IMemInputPin;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IMemInputPin : public IUnknown {
  public:
    virtual HRESULT WINAPI GetAllocator(IMemAllocator **ppAllocator) = 0;
    virtual HRESULT WINAPI NotifyAllocator(IMemAllocator *pAllocator,WINBOOL bReadOnly) = 0;
    virtual HRESULT WINAPI GetAllocatorRequirements(ALLOCATOR_PROPERTIES *pProps) = 0;
    virtual HRESULT WINAPI Receive(IMediaSample *pSample) = 0;
    virtual HRESULT WINAPI ReceiveMultiple(IMediaSample **pSamples,long nSamples,long *nSamplesProcessed) = 0;
    virtual HRESULT WINAPI ReceiveCanBlock(void) = 0;
  };
#else
  typedef struct IMemInputPinVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IMemInputPin *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IMemInputPin *This);
      ULONG (WINAPI *Release)(IMemInputPin *This);
      HRESULT (WINAPI *GetAllocator)(IMemInputPin *This,IMemAllocator **ppAllocator);
      HRESULT (WINAPI *NotifyAllocator)(IMemInputPin *This,IMemAllocator *pAllocator,WINBOOL bReadOnly);
      HRESULT (WINAPI *GetAllocatorRequirements)(IMemInputPin *This,ALLOCATOR_PROPERTIES *pProps);
      HRESULT (WINAPI *Receive)(IMemInputPin *This,IMediaSample *pSample);
      HRESULT (WINAPI *ReceiveMultiple)(IMemInputPin *This,IMediaSample **pSamples,long nSamples,long *nSamplesProcessed);
      HRESULT (WINAPI *ReceiveCanBlock)(IMemInputPin *This);
    END_INTERFACE
  } IMemInputPinVtbl;
  struct IMemInputPin {
    CONST_VTBL struct IMemInputPinVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IMemInputPin_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMemInputPin_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMemInputPin_Release(This) (This)->lpVtbl->Release(This)
#define IMemInputPin_GetAllocator(This,ppAllocator) (This)->lpVtbl->GetAllocator(This,ppAllocator)
#define IMemInputPin_NotifyAllocator(This,pAllocator,bReadOnly) (This)->lpVtbl->NotifyAllocator(This,pAllocator,bReadOnly)
#define IMemInputPin_GetAllocatorRequirements(This,pProps) (This)->lpVtbl->GetAllocatorRequirements(This,pProps)
#define IMemInputPin_Receive(This,pSample) (This)->lpVtbl->Receive(This,pSample)
#define IMemInputPin_ReceiveMultiple(This,pSamples,nSamples,nSamplesProcessed) (This)->lpVtbl->ReceiveMultiple(This,pSamples,nSamples,nSamplesProcessed)
#define IMemInputPin_ReceiveCanBlock(This) (This)->lpVtbl->ReceiveCanBlock(This)
#endif
#endif
  HRESULT WINAPI IMemInputPin_GetAllocator_Proxy(IMemInputPin *This,IMemAllocator **ppAllocator);
  void __RPC_STUB IMemInputPin_GetAllocator_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMemInputPin_NotifyAllocator_Proxy(IMemInputPin *This,IMemAllocator *pAllocator,WINBOOL bReadOnly);
  void __RPC_STUB IMemInputPin_NotifyAllocator_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMemInputPin_GetAllocatorRequirements_Proxy(IMemInputPin *This,ALLOCATOR_PROPERTIES *pProps);
  void __RPC_STUB IMemInputPin_GetAllocatorRequirements_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMemInputPin_Receive_Proxy(IMemInputPin *This,IMediaSample *pSample);
  void __RPC_STUB IMemInputPin_Receive_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMemInputPin_ReceiveMultiple_Proxy(IMemInputPin *This,IMediaSample **pSamples,long nSamples,long *nSamplesProcessed);
  void __RPC_STUB IMemInputPin_ReceiveMultiple_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMemInputPin_ReceiveCanBlock_Proxy(IMemInputPin *This);
  void __RPC_STUB IMemInputPin_ReceiveCanBlock_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IMemInputPin *PMEMINPUTPIN;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0132_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0132_v0_0_s_ifspec;
#ifndef __IAMovieSetup_INTERFACE_DEFINED__
#define __IAMovieSetup_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMovieSetup;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMovieSetup : public IUnknown {
  public:
    virtual HRESULT WINAPI Register(void) = 0;
    virtual HRESULT WINAPI Unregister(void) = 0;
  };
#else
  typedef struct IAMovieSetupVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMovieSetup *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMovieSetup *This);
      ULONG (WINAPI *Release)(IAMovieSetup *This);
      HRESULT (WINAPI *Register)(IAMovieSetup *This);
      HRESULT (WINAPI *Unregister)(IAMovieSetup *This);
    END_INTERFACE
  } IAMovieSetupVtbl;
  struct IAMovieSetup {
    CONST_VTBL struct IAMovieSetupVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMovieSetup_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMovieSetup_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMovieSetup_Release(This) (This)->lpVtbl->Release(This)
#define IAMovieSetup_Register(This) (This)->lpVtbl->Register(This)
#define IAMovieSetup_Unregister(This) (This)->lpVtbl->Unregister(This)
#endif
#endif
  HRESULT WINAPI IAMovieSetup_Register_Proxy(IAMovieSetup *This);
  void __RPC_STUB IAMovieSetup_Register_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMovieSetup_Unregister_Proxy(IAMovieSetup *This);
  void __RPC_STUB IAMovieSetup_Unregister_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IAMovieSetup *PAMOVIESETUP;

  typedef enum AM_SEEKING_SeekingFlags {
    AM_SEEKING_NoPositioning = 0,AM_SEEKING_AbsolutePositioning = 0x1,AM_SEEKING_RelativePositioning = 0x2,AM_SEEKING_IncrementalPositioning = 0x3,
    AM_SEEKING_PositioningBitsMask = 0x3,AM_SEEKING_SeekToKeyFrame = 0x4,AM_SEEKING_ReturnTime = 0x8,AM_SEEKING_Segment = 0x10,AM_SEEKING_NoFlush = 0x20
  } AM_SEEKING_SEEKING_FLAGS;

  typedef enum AM_SEEKING_SeekingCapabilities {
    AM_SEEKING_CanSeekAbsolute = 0x1,AM_SEEKING_CanSeekForwards = 0x2,AM_SEEKING_CanSeekBackwards = 0x4,AM_SEEKING_CanGetCurrentPos = 0x8,
    AM_SEEKING_CanGetStopPos = 0x10,AM_SEEKING_CanGetDuration = 0x20,AM_SEEKING_CanPlayBackwards = 0x40,AM_SEEKING_CanDoSegments = 0x80,
    AM_SEEKING_Source = 0x100
  } AM_SEEKING_SEEKING_CAPABILITIES;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0133_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0133_v0_0_s_ifspec;
#ifndef __IMediaSeeking_INTERFACE_DEFINED__
#define __IMediaSeeking_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IMediaSeeking;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IMediaSeeking : public IUnknown {
  public:
    virtual HRESULT WINAPI GetCapabilities(DWORD *pCapabilities) = 0;
    virtual HRESULT WINAPI CheckCapabilities(DWORD *pCapabilities) = 0;
    virtual HRESULT WINAPI IsFormatSupported(const GUID *pFormat) = 0;
    virtual HRESULT WINAPI QueryPreferredFormat(GUID *pFormat) = 0;
    virtual HRESULT WINAPI GetTimeFormat(GUID *pFormat) = 0;
    virtual HRESULT WINAPI IsUsingTimeFormat(const GUID *pFormat) = 0;
    virtual HRESULT WINAPI SetTimeFormat(const GUID *pFormat) = 0;
    virtual HRESULT WINAPI GetDuration(LONGLONG *pDuration) = 0;
    virtual HRESULT WINAPI GetStopPosition(LONGLONG *pStop) = 0;
    virtual HRESULT WINAPI GetCurrentPosition(LONGLONG *pCurrent) = 0;
    virtual HRESULT WINAPI ConvertTimeFormat(LONGLONG *pTarget,const GUID *pTargetFormat,LONGLONG Source,const GUID *pSourceFormat) = 0;
    virtual HRESULT WINAPI SetPositions(LONGLONG *pCurrent,DWORD dwCurrentFlags,LONGLONG *pStop,DWORD dwStopFlags) = 0;
    virtual HRESULT WINAPI GetPositions(LONGLONG *pCurrent,LONGLONG *pStop) = 0;
    virtual HRESULT WINAPI GetAvailable(LONGLONG *pEarliest,LONGLONG *pLatest) = 0;
    virtual HRESULT WINAPI SetRate(double dRate) = 0;
    virtual HRESULT WINAPI GetRate(double *pdRate) = 0;
    virtual HRESULT WINAPI GetPreroll(LONGLONG *pllPreroll) = 0;
  };
#else
  typedef struct IMediaSeekingVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IMediaSeeking *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IMediaSeeking *This);
      ULONG (WINAPI *Release)(IMediaSeeking *This);
      HRESULT (WINAPI *GetCapabilities)(IMediaSeeking *This,DWORD *pCapabilities);
      HRESULT (WINAPI *CheckCapabilities)(IMediaSeeking *This,DWORD *pCapabilities);
      HRESULT (WINAPI *IsFormatSupported)(IMediaSeeking *This,const GUID *pFormat);
      HRESULT (WINAPI *QueryPreferredFormat)(IMediaSeeking *This,GUID *pFormat);
      HRESULT (WINAPI *GetTimeFormat)(IMediaSeeking *This,GUID *pFormat);
      HRESULT (WINAPI *IsUsingTimeFormat)(IMediaSeeking *This,const GUID *pFormat);
      HRESULT (WINAPI *SetTimeFormat)(IMediaSeeking *This,const GUID *pFormat);
      HRESULT (WINAPI *GetDuration)(IMediaSeeking *This,LONGLONG *pDuration);
      HRESULT (WINAPI *GetStopPosition)(IMediaSeeking *This,LONGLONG *pStop);
      HRESULT (WINAPI *GetCurrentPosition)(IMediaSeeking *This,LONGLONG *pCurrent);
      HRESULT (WINAPI *ConvertTimeFormat)(IMediaSeeking *This,LONGLONG *pTarget,const GUID *pTargetFormat,LONGLONG Source,const GUID *pSourceFormat);
      HRESULT (WINAPI *SetPositions)(IMediaSeeking *This,LONGLONG *pCurrent,DWORD dwCurrentFlags,LONGLONG *pStop,DWORD dwStopFlags);
      HRESULT (WINAPI *GetPositions)(IMediaSeeking *This,LONGLONG *pCurrent,LONGLONG *pStop);
      HRESULT (WINAPI *GetAvailable)(IMediaSeeking *This,LONGLONG *pEarliest,LONGLONG *pLatest);
      HRESULT (WINAPI *SetRate)(IMediaSeeking *This,double dRate);
      HRESULT (WINAPI *GetRate)(IMediaSeeking *This,double *pdRate);
      HRESULT (WINAPI *GetPreroll)(IMediaSeeking *This,LONGLONG *pllPreroll);
    END_INTERFACE
  } IMediaSeekingVtbl;
  struct IMediaSeeking {
    CONST_VTBL struct IMediaSeekingVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IMediaSeeking_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMediaSeeking_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMediaSeeking_Release(This) (This)->lpVtbl->Release(This)
#define IMediaSeeking_GetCapabilities(This,pCapabilities) (This)->lpVtbl->GetCapabilities(This,pCapabilities)
#define IMediaSeeking_CheckCapabilities(This,pCapabilities) (This)->lpVtbl->CheckCapabilities(This,pCapabilities)
#define IMediaSeeking_IsFormatSupported(This,pFormat) (This)->lpVtbl->IsFormatSupported(This,pFormat)
#define IMediaSeeking_QueryPreferredFormat(This,pFormat) (This)->lpVtbl->QueryPreferredFormat(This,pFormat)
#define IMediaSeeking_GetTimeFormat(This,pFormat) (This)->lpVtbl->GetTimeFormat(This,pFormat)
#define IMediaSeeking_IsUsingTimeFormat(This,pFormat) (This)->lpVtbl->IsUsingTimeFormat(This,pFormat)
#define IMediaSeeking_SetTimeFormat(This,pFormat) (This)->lpVtbl->SetTimeFormat(This,pFormat)
#define IMediaSeeking_GetDuration(This,pDuration) (This)->lpVtbl->GetDuration(This,pDuration)
#define IMediaSeeking_GetStopPosition(This,pStop) (This)->lpVtbl->GetStopPosition(This,pStop)
#define IMediaSeeking_GetCurrentPosition(This,pCurrent) (This)->lpVtbl->GetCurrentPosition(This,pCurrent)
#define IMediaSeeking_ConvertTimeFormat(This,pTarget,pTargetFormat,Source,pSourceFormat) (This)->lpVtbl->ConvertTimeFormat(This,pTarget,pTargetFormat,Source,pSourceFormat)
#define IMediaSeeking_SetPositions(This,pCurrent,dwCurrentFlags,pStop,dwStopFlags) (This)->lpVtbl->SetPositions(This,pCurrent,dwCurrentFlags,pStop,dwStopFlags)
#define IMediaSeeking_GetPositions(This,pCurrent,pStop) (This)->lpVtbl->GetPositions(This,pCurrent,pStop)
#define IMediaSeeking_GetAvailable(This,pEarliest,pLatest) (This)->lpVtbl->GetAvailable(This,pEarliest,pLatest)
#define IMediaSeeking_SetRate(This,dRate) (This)->lpVtbl->SetRate(This,dRate)
#define IMediaSeeking_GetRate(This,pdRate) (This)->lpVtbl->GetRate(This,pdRate)
#define IMediaSeeking_GetPreroll(This,pllPreroll) (This)->lpVtbl->GetPreroll(This,pllPreroll)
#endif
#endif
  HRESULT WINAPI IMediaSeeking_GetCapabilities_Proxy(IMediaSeeking *This,DWORD *pCapabilities);
  void __RPC_STUB IMediaSeeking_GetCapabilities_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSeeking_CheckCapabilities_Proxy(IMediaSeeking *This,DWORD *pCapabilities);
  void __RPC_STUB IMediaSeeking_CheckCapabilities_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSeeking_IsFormatSupported_Proxy(IMediaSeeking *This,const GUID *pFormat);
  void __RPC_STUB IMediaSeeking_IsFormatSupported_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSeeking_QueryPreferredFormat_Proxy(IMediaSeeking *This,GUID *pFormat);
  void __RPC_STUB IMediaSeeking_QueryPreferredFormat_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSeeking_GetTimeFormat_Proxy(IMediaSeeking *This,GUID *pFormat);
  void __RPC_STUB IMediaSeeking_GetTimeFormat_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSeeking_IsUsingTimeFormat_Proxy(IMediaSeeking *This,const GUID *pFormat);
  void __RPC_STUB IMediaSeeking_IsUsingTimeFormat_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSeeking_SetTimeFormat_Proxy(IMediaSeeking *This,const GUID *pFormat);
  void __RPC_STUB IMediaSeeking_SetTimeFormat_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSeeking_GetDuration_Proxy(IMediaSeeking *This,LONGLONG *pDuration);
  void __RPC_STUB IMediaSeeking_GetDuration_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSeeking_GetStopPosition_Proxy(IMediaSeeking *This,LONGLONG *pStop);
  void __RPC_STUB IMediaSeeking_GetStopPosition_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSeeking_GetCurrentPosition_Proxy(IMediaSeeking *This,LONGLONG *pCurrent);
  void __RPC_STUB IMediaSeeking_GetCurrentPosition_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSeeking_ConvertTimeFormat_Proxy(IMediaSeeking *This,LONGLONG *pTarget,const GUID *pTargetFormat,LONGLONG Source,const GUID *pSourceFormat);
  void __RPC_STUB IMediaSeeking_ConvertTimeFormat_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSeeking_SetPositions_Proxy(IMediaSeeking *This,LONGLONG *pCurrent,DWORD dwCurrentFlags,LONGLONG *pStop,DWORD dwStopFlags);
  void __RPC_STUB IMediaSeeking_SetPositions_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSeeking_GetPositions_Proxy(IMediaSeeking *This,LONGLONG *pCurrent,LONGLONG *pStop);
  void __RPC_STUB IMediaSeeking_GetPositions_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSeeking_GetAvailable_Proxy(IMediaSeeking *This,LONGLONG *pEarliest,LONGLONG *pLatest);
  void __RPC_STUB IMediaSeeking_GetAvailable_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSeeking_SetRate_Proxy(IMediaSeeking *This,double dRate);
  void __RPC_STUB IMediaSeeking_SetRate_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSeeking_GetRate_Proxy(IMediaSeeking *This,double *pdRate);
  void __RPC_STUB IMediaSeeking_GetRate_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaSeeking_GetPreroll_Proxy(IMediaSeeking *This,LONGLONG *pllPreroll);
  void __RPC_STUB IMediaSeeking_GetPreroll_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IMediaSeeking *PMEDIASEEKING;

  enum tagAM_MEDIAEVENT_FLAGS {
    AM_MEDIAEVENT_NONOTIFY = 0x01
  };

  typedef struct __MIDL___MIDL_itf_strmif_0134_0001 {
    CLSID Clsid;
    LPWSTR Name;
  } REGFILTER;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0134_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0134_v0_0_s_ifspec;
#ifndef __IEnumRegFilters_INTERFACE_DEFINED__
#define __IEnumRegFilters_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IEnumRegFilters;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IEnumRegFilters : public IUnknown {
  public:
    virtual HRESULT WINAPI Next(ULONG cFilters,REGFILTER **apRegFilter,ULONG *pcFetched) = 0;
    virtual HRESULT WINAPI Skip(ULONG cFilters) = 0;
    virtual HRESULT WINAPI Reset(void) = 0;
    virtual HRESULT WINAPI Clone(IEnumRegFilters **ppEnum) = 0;
  };
#else
  typedef struct IEnumRegFiltersVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IEnumRegFilters *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IEnumRegFilters *This);
      ULONG (WINAPI *Release)(IEnumRegFilters *This);
      HRESULT (WINAPI *Next)(IEnumRegFilters *This,ULONG cFilters,REGFILTER **apRegFilter,ULONG *pcFetched);
      HRESULT (WINAPI *Skip)(IEnumRegFilters *This,ULONG cFilters);
      HRESULT (WINAPI *Reset)(IEnumRegFilters *This);
      HRESULT (WINAPI *Clone)(IEnumRegFilters *This,IEnumRegFilters **ppEnum);
    END_INTERFACE
  } IEnumRegFiltersVtbl;
  struct IEnumRegFilters {
    CONST_VTBL struct IEnumRegFiltersVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IEnumRegFilters_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IEnumRegFilters_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IEnumRegFilters_Release(This) (This)->lpVtbl->Release(This)
#define IEnumRegFilters_Next(This,cFilters,apRegFilter,pcFetched) (This)->lpVtbl->Next(This,cFilters,apRegFilter,pcFetched)
#define IEnumRegFilters_Skip(This,cFilters) (This)->lpVtbl->Skip(This,cFilters)
#define IEnumRegFilters_Reset(This) (This)->lpVtbl->Reset(This)
#define IEnumRegFilters_Clone(This,ppEnum) (This)->lpVtbl->Clone(This,ppEnum)
#endif
#endif
  HRESULT WINAPI IEnumRegFilters_Next_Proxy(IEnumRegFilters *This,ULONG cFilters,REGFILTER **apRegFilter,ULONG *pcFetched);
  void __RPC_STUB IEnumRegFilters_Next_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEnumRegFilters_Skip_Proxy(IEnumRegFilters *This,ULONG cFilters);
  void __RPC_STUB IEnumRegFilters_Skip_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEnumRegFilters_Reset_Proxy(IEnumRegFilters *This);
  void __RPC_STUB IEnumRegFilters_Reset_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEnumRegFilters_Clone_Proxy(IEnumRegFilters *This,IEnumRegFilters **ppEnum);
  void __RPC_STUB IEnumRegFilters_Clone_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IEnumRegFilters *PENUMREGFILTERS;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0136_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0136_v0_0_s_ifspec;
#ifndef __IFilterMapper_INTERFACE_DEFINED__
#define __IFilterMapper_INTERFACE_DEFINED__
  enum __MIDL_IFilterMapper_0001 {
    MERIT_PREFERRED = 0x800000,MERIT_NORMAL = 0x600000,MERIT_UNLIKELY = 0x400000,MERIT_DO_NOT_USE = 0x200000,MERIT_SW_COMPRESSOR = 0x100000,
    MERIT_HW_COMPRESSOR = 0x100050
  };
  EXTERN_C const IID IID_IFilterMapper;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IFilterMapper : public IUnknown {
  public:
    virtual HRESULT WINAPI RegisterFilter(CLSID clsid,LPCWSTR Name,DWORD dwMerit) = 0;
    virtual HRESULT WINAPI RegisterFilterInstance(CLSID clsid,LPCWSTR Name,CLSID *MRId) = 0;
    virtual HRESULT WINAPI RegisterPin(CLSID Filter,LPCWSTR Name,WINBOOL bRendered,WINBOOL bOutput,WINBOOL bZero,WINBOOL bMany,CLSID ConnectsToFilter,LPCWSTR ConnectsToPin) = 0;
    virtual HRESULT WINAPI RegisterPinType(CLSID clsFilter,LPCWSTR strName,CLSID clsMajorType,CLSID clsSubType) = 0;
    virtual HRESULT WINAPI UnregisterFilter(CLSID Filter) = 0;
    virtual HRESULT WINAPI UnregisterFilterInstance(CLSID MRId) = 0;
    virtual HRESULT WINAPI UnregisterPin(CLSID Filter,LPCWSTR Name) = 0;
    virtual HRESULT WINAPI EnumMatchingFilters(IEnumRegFilters **ppEnum,DWORD dwMerit,WINBOOL bInputNeeded,CLSID clsInMaj,CLSID clsInSub,WINBOOL bRender,WINBOOL bOututNeeded,CLSID clsOutMaj,CLSID clsOutSub) = 0;
  };
#else
  typedef struct IFilterMapperVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IFilterMapper *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IFilterMapper *This);
      ULONG (WINAPI *Release)(IFilterMapper *This);
      HRESULT (WINAPI *RegisterFilter)(IFilterMapper *This,CLSID clsid,LPCWSTR Name,DWORD dwMerit);
      HRESULT (WINAPI *RegisterFilterInstance)(IFilterMapper *This,CLSID clsid,LPCWSTR Name,CLSID *MRId);
      HRESULT (WINAPI *RegisterPin)(IFilterMapper *This,CLSID Filter,LPCWSTR Name,WINBOOL bRendered,WINBOOL bOutput,WINBOOL bZero,WINBOOL bMany,CLSID ConnectsToFilter,LPCWSTR ConnectsToPin);
      HRESULT (WINAPI *RegisterPinType)(IFilterMapper *This,CLSID clsFilter,LPCWSTR strName,CLSID clsMajorType,CLSID clsSubType);
      HRESULT (WINAPI *UnregisterFilter)(IFilterMapper *This,CLSID Filter);
      HRESULT (WINAPI *UnregisterFilterInstance)(IFilterMapper *This,CLSID MRId);
      HRESULT (WINAPI *UnregisterPin)(IFilterMapper *This,CLSID Filter,LPCWSTR Name);
      HRESULT (WINAPI *EnumMatchingFilters)(IFilterMapper *This,IEnumRegFilters **ppEnum,DWORD dwMerit,WINBOOL bInputNeeded,CLSID clsInMaj,CLSID clsInSub,WINBOOL bRender,WINBOOL bOututNeeded,CLSID clsOutMaj,CLSID clsOutSub);
    END_INTERFACE
  } IFilterMapperVtbl;
  struct IFilterMapper {
    CONST_VTBL struct IFilterMapperVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IFilterMapper_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IFilterMapper_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IFilterMapper_Release(This) (This)->lpVtbl->Release(This)
#define IFilterMapper_RegisterFilter(This,clsid,Name,dwMerit) (This)->lpVtbl->RegisterFilter(This,clsid,Name,dwMerit)
#define IFilterMapper_RegisterFilterInstance(This,clsid,Name,MRId) (This)->lpVtbl->RegisterFilterInstance(This,clsid,Name,MRId)
#define IFilterMapper_RegisterPin(This,Filter,Name,bRendered,bOutput,bZero,bMany,ConnectsToFilter,ConnectsToPin) (This)->lpVtbl->RegisterPin(This,Filter,Name,bRendered,bOutput,bZero,bMany,ConnectsToFilter,ConnectsToPin)
#define IFilterMapper_RegisterPinType(This,clsFilter,strName,clsMajorType,clsSubType) (This)->lpVtbl->RegisterPinType(This,clsFilter,strName,clsMajorType,clsSubType)
#define IFilterMapper_UnregisterFilter(This,Filter) (This)->lpVtbl->UnregisterFilter(This,Filter)
#define IFilterMapper_UnregisterFilterInstance(This,MRId) (This)->lpVtbl->UnregisterFilterInstance(This,MRId)
#define IFilterMapper_UnregisterPin(This,Filter,Name) (This)->lpVtbl->UnregisterPin(This,Filter,Name)
#define IFilterMapper_EnumMatchingFilters(This,ppEnum,dwMerit,bInputNeeded,clsInMaj,clsInSub,bRender,bOututNeeded,clsOutMaj,clsOutSub) (This)->lpVtbl->EnumMatchingFilters(This,ppEnum,dwMerit,bInputNeeded,clsInMaj,clsInSub,bRender,bOututNeeded,clsOutMaj,clsOutSub)
#endif
#endif
  HRESULT WINAPI IFilterMapper_RegisterFilter_Proxy(IFilterMapper *This,CLSID clsid,LPCWSTR Name,DWORD dwMerit);
  void __RPC_STUB IFilterMapper_RegisterFilter_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterMapper_RegisterFilterInstance_Proxy(IFilterMapper *This,CLSID clsid,LPCWSTR Name,CLSID *MRId);
  void __RPC_STUB IFilterMapper_RegisterFilterInstance_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterMapper_RegisterPin_Proxy(IFilterMapper *This,CLSID Filter,LPCWSTR Name,WINBOOL bRendered,WINBOOL bOutput,WINBOOL bZero,WINBOOL bMany,CLSID ConnectsToFilter,LPCWSTR ConnectsToPin);
  void __RPC_STUB IFilterMapper_RegisterPin_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterMapper_RegisterPinType_Proxy(IFilterMapper *This,CLSID clsFilter,LPCWSTR strName,CLSID clsMajorType,CLSID clsSubType);
  void __RPC_STUB IFilterMapper_RegisterPinType_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterMapper_UnregisterFilter_Proxy(IFilterMapper *This,CLSID Filter);
  void __RPC_STUB IFilterMapper_UnregisterFilter_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterMapper_UnregisterFilterInstance_Proxy(IFilterMapper *This,CLSID MRId);
  void __RPC_STUB IFilterMapper_UnregisterFilterInstance_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterMapper_UnregisterPin_Proxy(IFilterMapper *This,CLSID Filter,LPCWSTR Name);
  void __RPC_STUB IFilterMapper_UnregisterPin_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterMapper_EnumMatchingFilters_Proxy(IFilterMapper *This,IEnumRegFilters **ppEnum,DWORD dwMerit,WINBOOL bInputNeeded,CLSID clsInMaj,CLSID clsInSub,WINBOOL bRender,WINBOOL bOututNeeded,CLSID clsOutMaj,CLSID clsOutSub);
  void __RPC_STUB IFilterMapper_EnumMatchingFilters_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef struct __MIDL___MIDL_itf_strmif_0138_0001 {
    const CLSID *clsMajorType;
    const CLSID *clsMinorType;
  } REGPINTYPES;

  typedef struct __MIDL___MIDL_itf_strmif_0138_0002 {
    LPWSTR strName;
    WINBOOL bRendered;
    WINBOOL bOutput;
    WINBOOL bZero;
    WINBOOL bMany;
    const CLSID *clsConnectsToFilter;
    const WCHAR *strConnectsToPin;
    UINT nMediaTypes;
    const REGPINTYPES *lpMediaType;
  } REGFILTERPINS;

  typedef struct __MIDL___MIDL_itf_strmif_0138_0003 {
    CLSID clsMedium;
    DWORD dw1;
    DWORD dw2;
  } REGPINMEDIUM;

  enum __MIDL___MIDL_itf_strmif_0138_0004 {
    REG_PINFLAG_B_ZERO = 0x1,REG_PINFLAG_B_RENDERER = 0x2,REG_PINFLAG_B_MANY = 0x4,REG_PINFLAG_B_OUTPUT = 0x8
  };
  typedef struct __MIDL___MIDL_itf_strmif_0138_0005 {
    DWORD dwFlags;
    UINT cInstances;
    UINT nMediaTypes;
    const REGPINTYPES *lpMediaType;
    UINT nMediums;
    const REGPINMEDIUM *lpMedium;
    const CLSID *clsPinCategory;
  } REGFILTERPINS2;

  typedef struct __MIDL___MIDL_itf_strmif_0138_0006 {
    DWORD dwVersion;
    DWORD dwMerit;
    union {
      struct {
	ULONG cPins;
	const REGFILTERPINS *rgPins;
      };
      struct {
	ULONG cPins2;
	const REGFILTERPINS2 *rgPins2;
      };
    };
  } REGFILTER2;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0138_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0138_v0_0_s_ifspec;
#ifndef __IFilterMapper2_INTERFACE_DEFINED__
#define __IFilterMapper2_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IFilterMapper2;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IFilterMapper2 : public IUnknown {
  public:
    virtual HRESULT WINAPI CreateCategory(REFCLSID clsidCategory,DWORD dwCategoryMerit,LPCWSTR Description) = 0;
    virtual HRESULT WINAPI UnregisterFilter(const CLSID *pclsidCategory,const OLECHAR *szInstance,REFCLSID Filter) = 0;
    virtual HRESULT WINAPI RegisterFilter(REFCLSID clsidFilter,LPCWSTR Name,IMoniker **ppMoniker,const CLSID *pclsidCategory,const OLECHAR *szInstance,const REGFILTER2 *prf2) = 0;
    virtual HRESULT WINAPI EnumMatchingFilters(IEnumMoniker **ppEnum,DWORD dwFlags,WINBOOL bExactMatch,DWORD dwMerit,WINBOOL bInputNeeded,DWORD cInputTypes,const GUID *pInputTypes,const REGPINMEDIUM *pMedIn,const CLSID *pPinCategoryIn,WINBOOL bRender,WINBOOL bOutputNeeded,DWORD cOutputTypes,const GUID *pOutputTypes,const REGPINMEDIUM *pMedOut,const CLSID *pPinCategoryOut) = 0;
  };
#else
  typedef struct IFilterMapper2Vtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IFilterMapper2 *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IFilterMapper2 *This);
      ULONG (WINAPI *Release)(IFilterMapper2 *This);
      HRESULT (WINAPI *CreateCategory)(IFilterMapper2 *This,REFCLSID clsidCategory,DWORD dwCategoryMerit,LPCWSTR Description);
      HRESULT (WINAPI *UnregisterFilter)(IFilterMapper2 *This,const CLSID *pclsidCategory,const OLECHAR *szInstance,REFCLSID Filter);
      HRESULT (WINAPI *RegisterFilter)(IFilterMapper2 *This,REFCLSID clsidFilter,LPCWSTR Name,IMoniker **ppMoniker,const CLSID *pclsidCategory,const OLECHAR *szInstance,const REGFILTER2 *prf2);
      HRESULT (WINAPI *EnumMatchingFilters)(IFilterMapper2 *This,IEnumMoniker **ppEnum,DWORD dwFlags,WINBOOL bExactMatch,DWORD dwMerit,WINBOOL bInputNeeded,DWORD cInputTypes,const GUID *pInputTypes,const REGPINMEDIUM *pMedIn,const CLSID *pPinCategoryIn,WINBOOL bRender,WINBOOL bOutputNeeded,DWORD cOutputTypes,const GUID *pOutputTypes,const REGPINMEDIUM *pMedOut,const CLSID *pPinCategoryOut);
    END_INTERFACE
  } IFilterMapper2Vtbl;
  struct IFilterMapper2 {
    CONST_VTBL struct IFilterMapper2Vtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IFilterMapper2_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IFilterMapper2_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IFilterMapper2_Release(This) (This)->lpVtbl->Release(This)
#define IFilterMapper2_CreateCategory(This,clsidCategory,dwCategoryMerit,Description) (This)->lpVtbl->CreateCategory(This,clsidCategory,dwCategoryMerit,Description)
#define IFilterMapper2_UnregisterFilter(This,pclsidCategory,szInstance,Filter) (This)->lpVtbl->UnregisterFilter(This,pclsidCategory,szInstance,Filter)
#define IFilterMapper2_RegisterFilter(This,clsidFilter,Name,ppMoniker,pclsidCategory,szInstance,prf2) (This)->lpVtbl->RegisterFilter(This,clsidFilter,Name,ppMoniker,pclsidCategory,szInstance,prf2)
#define IFilterMapper2_EnumMatchingFilters(This,ppEnum,dwFlags,bExactMatch,dwMerit,bInputNeeded,cInputTypes,pInputTypes,pMedIn,pPinCategoryIn,bRender,bOutputNeeded,cOutputTypes,pOutputTypes,pMedOut,pPinCategoryOut) (This)->lpVtbl->EnumMatchingFilters(This,ppEnum,dwFlags,bExactMatch,dwMerit,bInputNeeded,cInputTypes,pInputTypes,pMedIn,pPinCategoryIn,bRender,bOutputNeeded,cOutputTypes,pOutputTypes,pMedOut,pPinCategoryOut)
#endif
#endif
  HRESULT WINAPI IFilterMapper2_CreateCategory_Proxy(IFilterMapper2 *This,REFCLSID clsidCategory,DWORD dwCategoryMerit,LPCWSTR Description);
  void __RPC_STUB IFilterMapper2_CreateCategory_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterMapper2_UnregisterFilter_Proxy(IFilterMapper2 *This,const CLSID *pclsidCategory,const OLECHAR *szInstance,REFCLSID Filter);
  void __RPC_STUB IFilterMapper2_UnregisterFilter_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterMapper2_RegisterFilter_Proxy(IFilterMapper2 *This,REFCLSID clsidFilter,LPCWSTR Name,IMoniker **ppMoniker,const CLSID *pclsidCategory,const OLECHAR *szInstance,const REGFILTER2 *prf2);
  void __RPC_STUB IFilterMapper2_RegisterFilter_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterMapper2_EnumMatchingFilters_Proxy(IFilterMapper2 *This,IEnumMoniker **ppEnum,DWORD dwFlags,WINBOOL bExactMatch,DWORD dwMerit,WINBOOL bInputNeeded,DWORD cInputTypes,const GUID *pInputTypes,const REGPINMEDIUM *pMedIn,const CLSID *pPinCategoryIn,WINBOOL bRender,WINBOOL bOutputNeeded,DWORD cOutputTypes,const GUID *pOutputTypes,const REGPINMEDIUM *pMedOut,const CLSID *pPinCategoryOut);
  void __RPC_STUB IFilterMapper2_EnumMatchingFilters_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IFilterMapper3_INTERFACE_DEFINED__
#define __IFilterMapper3_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IFilterMapper3;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IFilterMapper3 : public IFilterMapper2 {
  public:
    virtual HRESULT WINAPI GetICreateDevEnum(ICreateDevEnum **ppEnum) = 0;
  };
#else
  typedef struct IFilterMapper3Vtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IFilterMapper3 *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IFilterMapper3 *This);
      ULONG (WINAPI *Release)(IFilterMapper3 *This);
      HRESULT (WINAPI *CreateCategory)(IFilterMapper3 *This,REFCLSID clsidCategory,DWORD dwCategoryMerit,LPCWSTR Description);
      HRESULT (WINAPI *UnregisterFilter)(IFilterMapper3 *This,const CLSID *pclsidCategory,const OLECHAR *szInstance,REFCLSID Filter);
      HRESULT (WINAPI *RegisterFilter)(IFilterMapper3 *This,REFCLSID clsidFilter,LPCWSTR Name,IMoniker **ppMoniker,const CLSID *pclsidCategory,const OLECHAR *szInstance,const REGFILTER2 *prf2);
      HRESULT (WINAPI *EnumMatchingFilters)(IFilterMapper3 *This,IEnumMoniker **ppEnum,DWORD dwFlags,WINBOOL bExactMatch,DWORD dwMerit,WINBOOL bInputNeeded,DWORD cInputTypes,const GUID *pInputTypes,const REGPINMEDIUM *pMedIn,const CLSID *pPinCategoryIn,WINBOOL bRender,WINBOOL bOutputNeeded,DWORD cOutputTypes,const GUID *pOutputTypes,const REGPINMEDIUM *pMedOut,const CLSID *pPinCategoryOut);
      HRESULT (WINAPI *GetICreateDevEnum)(IFilterMapper3 *This,ICreateDevEnum **ppEnum);
    END_INTERFACE
  } IFilterMapper3Vtbl;
  struct IFilterMapper3 {
    CONST_VTBL struct IFilterMapper3Vtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IFilterMapper3_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IFilterMapper3_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IFilterMapper3_Release(This) (This)->lpVtbl->Release(This)
#define IFilterMapper3_CreateCategory(This,clsidCategory,dwCategoryMerit,Description) (This)->lpVtbl->CreateCategory(This,clsidCategory,dwCategoryMerit,Description)
#define IFilterMapper3_UnregisterFilter(This,pclsidCategory,szInstance,Filter) (This)->lpVtbl->UnregisterFilter(This,pclsidCategory,szInstance,Filter)
#define IFilterMapper3_RegisterFilter(This,clsidFilter,Name,ppMoniker,pclsidCategory,szInstance,prf2) (This)->lpVtbl->RegisterFilter(This,clsidFilter,Name,ppMoniker,pclsidCategory,szInstance,prf2)
#define IFilterMapper3_EnumMatchingFilters(This,ppEnum,dwFlags,bExactMatch,dwMerit,bInputNeeded,cInputTypes,pInputTypes,pMedIn,pPinCategoryIn,bRender,bOutputNeeded,cOutputTypes,pOutputTypes,pMedOut,pPinCategoryOut) (This)->lpVtbl->EnumMatchingFilters(This,ppEnum,dwFlags,bExactMatch,dwMerit,bInputNeeded,cInputTypes,pInputTypes,pMedIn,pPinCategoryIn,bRender,bOutputNeeded,cOutputTypes,pOutputTypes,pMedOut,pPinCategoryOut)
#define IFilterMapper3_GetICreateDevEnum(This,ppEnum) (This)->lpVtbl->GetICreateDevEnum(This,ppEnum)
#endif
#endif
  HRESULT WINAPI IFilterMapper3_GetICreateDevEnum_Proxy(IFilterMapper3 *This,ICreateDevEnum **ppEnum);
  void __RPC_STUB IFilterMapper3_GetICreateDevEnum_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef enum tagQualityMessageType {
    Famine = 0,Flood = Famine + 1
  } QualityMessageType;

  typedef struct tagQuality {
    QualityMessageType Type;
    long Proportion;
    REFERENCE_TIME Late;
    REFERENCE_TIME TimeStamp;
  } Quality;

  typedef IQualityControl *PQUALITYCONTROL;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0141_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0141_v0_0_s_ifspec;
#ifndef __IQualityControl_INTERFACE_DEFINED__
#define __IQualityControl_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IQualityControl;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IQualityControl : public IUnknown {
  public:
    virtual HRESULT WINAPI Notify(IBaseFilter *pSelf,Quality q) = 0;
    virtual HRESULT WINAPI SetSink(IQualityControl *piqc) = 0;
  };
#else
  typedef struct IQualityControlVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IQualityControl *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IQualityControl *This);
      ULONG (WINAPI *Release)(IQualityControl *This);
      HRESULT (WINAPI *Notify)(IQualityControl *This,IBaseFilter *pSelf,Quality q);
      HRESULT (WINAPI *SetSink)(IQualityControl *This,IQualityControl *piqc);
    END_INTERFACE
  } IQualityControlVtbl;
  struct IQualityControl {
    CONST_VTBL struct IQualityControlVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IQualityControl_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IQualityControl_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IQualityControl_Release(This) (This)->lpVtbl->Release(This)
#define IQualityControl_Notify(This,pSelf,q) (This)->lpVtbl->Notify(This,pSelf,q)
#define IQualityControl_SetSink(This,piqc) (This)->lpVtbl->SetSink(This,piqc)
#endif
#endif
  HRESULT WINAPI IQualityControl_Notify_Proxy(IQualityControl *This,IBaseFilter *pSelf,Quality q);
  void __RPC_STUB IQualityControl_Notify_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IQualityControl_SetSink_Proxy(IQualityControl *This,IQualityControl *piqc);
  void __RPC_STUB IQualityControl_SetSink_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  enum __MIDL___MIDL_itf_strmif_0142_0001 {
    CK_NOCOLORKEY = 0,CK_INDEX = 0x1,CK_RGB = 0x2
  };
  typedef struct tagCOLORKEY {
    DWORD KeyType;
    DWORD PaletteIndex;
    COLORREF LowColorValue;
    COLORREF HighColorValue;
  } COLORKEY;

  enum __MIDL___MIDL_itf_strmif_0142_0002 {
    ADVISE_NONE = 0,ADVISE_CLIPPING = 0x1,ADVISE_PALETTE = 0x2,ADVISE_COLORKEY = 0x4,ADVISE_POSITION = 0x8,ADVISE_DISPLAY_CHANGE = 0x10
  };
#define ADVISE_ALL (ADVISE_CLIPPING | ADVISE_PALETTE | ADVISE_COLORKEY | ADVISE_POSITION)
#define ADVISE_ALL2 (ADVISE_ALL | ADVISE_DISPLAY_CHANGE)

#ifndef _WINGDI_
  typedef struct _RGNDATAHEADER {
    DWORD dwSize;
    DWORD iType;
    DWORD nCount;
    DWORD nRgnSize;
    RECT rcBound;
  } RGNDATAHEADER;

  typedef struct _RGNDATA {
    RGNDATAHEADER rdh;
    char Buffer[1];
  } RGNDATA;
#endif

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0142_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0142_v0_0_s_ifspec;
#ifndef __IOverlayNotify_INTERFACE_DEFINED__
#define __IOverlayNotify_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IOverlayNotify;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IOverlayNotify : public IUnknown {
  public:
    virtual HRESULT WINAPI OnPaletteChange(DWORD dwColors,const PALETTEENTRY *pPalette) = 0;
    virtual HRESULT WINAPI OnClipChange(const RECT *pSourceRect,const RECT *pDestinationRect,const RGNDATA *pRgnData) = 0;
    virtual HRESULT WINAPI OnColorKeyChange(const COLORKEY *pColorKey) = 0;
    virtual HRESULT WINAPI OnPositionChange(const RECT *pSourceRect,const RECT *pDestinationRect) = 0;
  };
#else
  typedef struct IOverlayNotifyVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IOverlayNotify *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IOverlayNotify *This);
      ULONG (WINAPI *Release)(IOverlayNotify *This);
      HRESULT (WINAPI *OnPaletteChange)(IOverlayNotify *This,DWORD dwColors,const PALETTEENTRY *pPalette);
      HRESULT (WINAPI *OnClipChange)(IOverlayNotify *This,const RECT *pSourceRect,const RECT *pDestinationRect,const RGNDATA *pRgnData);
      HRESULT (WINAPI *OnColorKeyChange)(IOverlayNotify *This,const COLORKEY *pColorKey);
      HRESULT (WINAPI *OnPositionChange)(IOverlayNotify *This,const RECT *pSourceRect,const RECT *pDestinationRect);
    END_INTERFACE
  } IOverlayNotifyVtbl;
  struct IOverlayNotify {
    CONST_VTBL struct IOverlayNotifyVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IOverlayNotify_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IOverlayNotify_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IOverlayNotify_Release(This) (This)->lpVtbl->Release(This)
#define IOverlayNotify_OnPaletteChange(This,dwColors,pPalette) (This)->lpVtbl->OnPaletteChange(This,dwColors,pPalette)
#define IOverlayNotify_OnClipChange(This,pSourceRect,pDestinationRect,pRgnData) (This)->lpVtbl->OnClipChange(This,pSourceRect,pDestinationRect,pRgnData)
#define IOverlayNotify_OnColorKeyChange(This,pColorKey) (This)->lpVtbl->OnColorKeyChange(This,pColorKey)
#define IOverlayNotify_OnPositionChange(This,pSourceRect,pDestinationRect) (This)->lpVtbl->OnPositionChange(This,pSourceRect,pDestinationRect)
#endif
#endif
  HRESULT WINAPI IOverlayNotify_OnPaletteChange_Proxy(IOverlayNotify *This,DWORD dwColors,const PALETTEENTRY *pPalette);
  void __RPC_STUB IOverlayNotify_OnPaletteChange_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IOverlayNotify_OnClipChange_Proxy(IOverlayNotify *This,const RECT *pSourceRect,const RECT *pDestinationRect,const RGNDATA *pRgnData);
  void __RPC_STUB IOverlayNotify_OnClipChange_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IOverlayNotify_OnColorKeyChange_Proxy(IOverlayNotify *This,const COLORKEY *pColorKey);
  void __RPC_STUB IOverlayNotify_OnColorKeyChange_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IOverlayNotify_OnPositionChange_Proxy(IOverlayNotify *This,const RECT *pSourceRect,const RECT *pDestinationRect);
  void __RPC_STUB IOverlayNotify_OnPositionChange_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IOverlayNotify *POVERLAYNOTIFY;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0143_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0143_v0_0_s_ifspec;
#ifndef __IOverlayNotify2_INTERFACE_DEFINED__
#define __IOverlayNotify2_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IOverlayNotify2;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IOverlayNotify2 : public IOverlayNotify {
  public:
    virtual HRESULT WINAPI OnDisplayChange(HMONITOR hMonitor) = 0;
  };
#else
  typedef struct IOverlayNotify2Vtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IOverlayNotify2 *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IOverlayNotify2 *This);
      ULONG (WINAPI *Release)(IOverlayNotify2 *This);
      HRESULT (WINAPI *OnPaletteChange)(IOverlayNotify2 *This,DWORD dwColors,const PALETTEENTRY *pPalette);
      HRESULT (WINAPI *OnClipChange)(IOverlayNotify2 *This,const RECT *pSourceRect,const RECT *pDestinationRect,const RGNDATA *pRgnData);
      HRESULT (WINAPI *OnColorKeyChange)(IOverlayNotify2 *This,const COLORKEY *pColorKey);
      HRESULT (WINAPI *OnPositionChange)(IOverlayNotify2 *This,const RECT *pSourceRect,const RECT *pDestinationRect);
      HRESULT (WINAPI *OnDisplayChange)(IOverlayNotify2 *This,HMONITOR hMonitor);
    END_INTERFACE
  } IOverlayNotify2Vtbl;
  struct IOverlayNotify2 {
    CONST_VTBL struct IOverlayNotify2Vtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IOverlayNotify2_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IOverlayNotify2_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IOverlayNotify2_Release(This) (This)->lpVtbl->Release(This)
#define IOverlayNotify2_OnPaletteChange(This,dwColors,pPalette) (This)->lpVtbl->OnPaletteChange(This,dwColors,pPalette)
#define IOverlayNotify2_OnClipChange(This,pSourceRect,pDestinationRect,pRgnData) (This)->lpVtbl->OnClipChange(This,pSourceRect,pDestinationRect,pRgnData)
#define IOverlayNotify2_OnColorKeyChange(This,pColorKey) (This)->lpVtbl->OnColorKeyChange(This,pColorKey)
#define IOverlayNotify2_OnPositionChange(This,pSourceRect,pDestinationRect) (This)->lpVtbl->OnPositionChange(This,pSourceRect,pDestinationRect)
#define IOverlayNotify2_OnDisplayChange(This,hMonitor) (This)->lpVtbl->OnDisplayChange(This,hMonitor)
#endif
#endif
  HRESULT WINAPI IOverlayNotify2_OnDisplayChange_Proxy(IOverlayNotify2 *This,HMONITOR hMonitor);
  void __RPC_STUB IOverlayNotify2_OnDisplayChange_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IOverlayNotify2 *POVERLAYNOTIFY2;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0144_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0144_v0_0_s_ifspec;
#ifndef __IOverlay_INTERFACE_DEFINED__
#define __IOverlay_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IOverlay;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IOverlay : public IUnknown {
  public:
    virtual HRESULT WINAPI GetPalette(DWORD *pdwColors,PALETTEENTRY **ppPalette) = 0;
    virtual HRESULT WINAPI SetPalette(DWORD dwColors,PALETTEENTRY *pPalette) = 0;
    virtual HRESULT WINAPI GetDefaultColorKey(COLORKEY *pColorKey) = 0;
    virtual HRESULT WINAPI GetColorKey(COLORKEY *pColorKey) = 0;
    virtual HRESULT WINAPI SetColorKey(COLORKEY *pColorKey) = 0;
    virtual HRESULT WINAPI GetWindowHandle(HWND *pHwnd) = 0;
    virtual HRESULT WINAPI GetClipList(RECT *pSourceRect,RECT *pDestinationRect,RGNDATA **ppRgnData) = 0;
    virtual HRESULT WINAPI GetVideoPosition(RECT *pSourceRect,RECT *pDestinationRect) = 0;
    virtual HRESULT WINAPI Advise(IOverlayNotify *pOverlayNotify,DWORD dwInterests) = 0;
    virtual HRESULT WINAPI Unadvise(void) = 0;
  };
#else
  typedef struct IOverlayVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IOverlay *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IOverlay *This);
      ULONG (WINAPI *Release)(IOverlay *This);
      HRESULT (WINAPI *GetPalette)(IOverlay *This,DWORD *pdwColors,PALETTEENTRY **ppPalette);
      HRESULT (WINAPI *SetPalette)(IOverlay *This,DWORD dwColors,PALETTEENTRY *pPalette);
      HRESULT (WINAPI *GetDefaultColorKey)(IOverlay *This,COLORKEY *pColorKey);
      HRESULT (WINAPI *GetColorKey)(IOverlay *This,COLORKEY *pColorKey);
      HRESULT (WINAPI *SetColorKey)(IOverlay *This,COLORKEY *pColorKey);
      HRESULT (WINAPI *GetWindowHandle)(IOverlay *This,HWND *pHwnd);
      HRESULT (WINAPI *GetClipList)(IOverlay *This,RECT *pSourceRect,RECT *pDestinationRect,RGNDATA **ppRgnData);
      HRESULT (WINAPI *GetVideoPosition)(IOverlay *This,RECT *pSourceRect,RECT *pDestinationRect);
      HRESULT (WINAPI *Advise)(IOverlay *This,IOverlayNotify *pOverlayNotify,DWORD dwInterests);
      HRESULT (WINAPI *Unadvise)(IOverlay *This);
    END_INTERFACE
  } IOverlayVtbl;
  struct IOverlay {
    CONST_VTBL struct IOverlayVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IOverlay_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IOverlay_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IOverlay_Release(This) (This)->lpVtbl->Release(This)
#define IOverlay_GetPalette(This,pdwColors,ppPalette) (This)->lpVtbl->GetPalette(This,pdwColors,ppPalette)
#define IOverlay_SetPalette(This,dwColors,pPalette) (This)->lpVtbl->SetPalette(This,dwColors,pPalette)
#define IOverlay_GetDefaultColorKey(This,pColorKey) (This)->lpVtbl->GetDefaultColorKey(This,pColorKey)
#define IOverlay_GetColorKey(This,pColorKey) (This)->lpVtbl->GetColorKey(This,pColorKey)
#define IOverlay_SetColorKey(This,pColorKey) (This)->lpVtbl->SetColorKey(This,pColorKey)
#define IOverlay_GetWindowHandle(This,pHwnd) (This)->lpVtbl->GetWindowHandle(This,pHwnd)
#define IOverlay_GetClipList(This,pSourceRect,pDestinationRect,ppRgnData) (This)->lpVtbl->GetClipList(This,pSourceRect,pDestinationRect,ppRgnData)
#define IOverlay_GetVideoPosition(This,pSourceRect,pDestinationRect) (This)->lpVtbl->GetVideoPosition(This,pSourceRect,pDestinationRect)
#define IOverlay_Advise(This,pOverlayNotify,dwInterests) (This)->lpVtbl->Advise(This,pOverlayNotify,dwInterests)
#define IOverlay_Unadvise(This) (This)->lpVtbl->Unadvise(This)
#endif
#endif
  HRESULT WINAPI IOverlay_GetPalette_Proxy(IOverlay *This,DWORD *pdwColors,PALETTEENTRY **ppPalette);
  void __RPC_STUB IOverlay_GetPalette_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IOverlay_SetPalette_Proxy(IOverlay *This,DWORD dwColors,PALETTEENTRY *pPalette);
  void __RPC_STUB IOverlay_SetPalette_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IOverlay_GetDefaultColorKey_Proxy(IOverlay *This,COLORKEY *pColorKey);
  void __RPC_STUB IOverlay_GetDefaultColorKey_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IOverlay_GetColorKey_Proxy(IOverlay *This,COLORKEY *pColorKey);
  void __RPC_STUB IOverlay_GetColorKey_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IOverlay_SetColorKey_Proxy(IOverlay *This,COLORKEY *pColorKey);
  void __RPC_STUB IOverlay_SetColorKey_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IOverlay_GetWindowHandle_Proxy(IOverlay *This,HWND *pHwnd);
  void __RPC_STUB IOverlay_GetWindowHandle_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IOverlay_GetClipList_Proxy(IOverlay *This,RECT *pSourceRect,RECT *pDestinationRect,RGNDATA **ppRgnData);
  void __RPC_STUB IOverlay_GetClipList_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IOverlay_GetVideoPosition_Proxy(IOverlay *This,RECT *pSourceRect,RECT *pDestinationRect);
  void __RPC_STUB IOverlay_GetVideoPosition_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IOverlay_Advise_Proxy(IOverlay *This,IOverlayNotify *pOverlayNotify,DWORD dwInterests);
  void __RPC_STUB IOverlay_Advise_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IOverlay_Unadvise_Proxy(IOverlay *This);
  void __RPC_STUB IOverlay_Unadvise_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IOverlay *POVERLAY;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0145_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0145_v0_0_s_ifspec;
#ifndef __IMediaEventSink_INTERFACE_DEFINED__
#define __IMediaEventSink_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IMediaEventSink;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IMediaEventSink : public IUnknown {
  public:
    virtual HRESULT WINAPI Notify(long EventCode,LONG_PTR EventParam1,LONG_PTR EventParam2) = 0;
  };
#else
  typedef struct IMediaEventSinkVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IMediaEventSink *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IMediaEventSink *This);
      ULONG (WINAPI *Release)(IMediaEventSink *This);
      HRESULT (WINAPI *Notify)(IMediaEventSink *This,long EventCode,LONG_PTR EventParam1,LONG_PTR EventParam2);
    END_INTERFACE
  } IMediaEventSinkVtbl;
  struct IMediaEventSink {
    CONST_VTBL struct IMediaEventSinkVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IMediaEventSink_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMediaEventSink_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMediaEventSink_Release(This) (This)->lpVtbl->Release(This)
#define IMediaEventSink_Notify(This,EventCode,EventParam1,EventParam2) (This)->lpVtbl->Notify(This,EventCode,EventParam1,EventParam2)
#endif
#endif
  HRESULT WINAPI IMediaEventSink_Notify_Proxy(IMediaEventSink *This,long EventCode,LONG_PTR EventParam1,LONG_PTR EventParam2);
  void __RPC_STUB IMediaEventSink_Notify_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IMediaEventSink *PMEDIAEVENTSINK;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0146_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0146_v0_0_s_ifspec;
#ifndef __IFileSourceFilter_INTERFACE_DEFINED__
#define __IFileSourceFilter_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IFileSourceFilter;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IFileSourceFilter : public IUnknown {
  public:
    virtual HRESULT WINAPI Load(LPCOLESTR pszFileName,const AM_MEDIA_TYPE *pmt) = 0;
    virtual HRESULT WINAPI GetCurFile(LPOLESTR *ppszFileName,AM_MEDIA_TYPE *pmt) = 0;
  };
#else
  typedef struct IFileSourceFilterVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IFileSourceFilter *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IFileSourceFilter *This);
      ULONG (WINAPI *Release)(IFileSourceFilter *This);
      HRESULT (WINAPI *Load)(IFileSourceFilter *This,LPCOLESTR pszFileName,const AM_MEDIA_TYPE *pmt);
      HRESULT (WINAPI *GetCurFile)(IFileSourceFilter *This,LPOLESTR *ppszFileName,AM_MEDIA_TYPE *pmt);
    END_INTERFACE
  } IFileSourceFilterVtbl;
  struct IFileSourceFilter {
    CONST_VTBL struct IFileSourceFilterVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IFileSourceFilter_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IFileSourceFilter_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IFileSourceFilter_Release(This) (This)->lpVtbl->Release(This)
#define IFileSourceFilter_Load(This,pszFileName,pmt) (This)->lpVtbl->Load(This,pszFileName,pmt)
#define IFileSourceFilter_GetCurFile(This,ppszFileName,pmt) (This)->lpVtbl->GetCurFile(This,ppszFileName,pmt)
#endif
#endif
  HRESULT WINAPI IFileSourceFilter_Load_Proxy(IFileSourceFilter *This,LPCOLESTR pszFileName,const AM_MEDIA_TYPE *pmt);
  void __RPC_STUB IFileSourceFilter_Load_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFileSourceFilter_GetCurFile_Proxy(IFileSourceFilter *This,LPOLESTR *ppszFileName,AM_MEDIA_TYPE *pmt);
  void __RPC_STUB IFileSourceFilter_GetCurFile_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IFileSourceFilter *PFILTERFILESOURCE;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0147_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0147_v0_0_s_ifspec;
#ifndef __IFileSinkFilter_INTERFACE_DEFINED__
#define __IFileSinkFilter_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IFileSinkFilter;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IFileSinkFilter : public IUnknown {
  public:
    virtual HRESULT WINAPI SetFileName(LPCOLESTR pszFileName,const AM_MEDIA_TYPE *pmt) = 0;
    virtual HRESULT WINAPI GetCurFile(LPOLESTR *ppszFileName,AM_MEDIA_TYPE *pmt) = 0;
  };
#else
  typedef struct IFileSinkFilterVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IFileSinkFilter *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IFileSinkFilter *This);
      ULONG (WINAPI *Release)(IFileSinkFilter *This);
      HRESULT (WINAPI *SetFileName)(IFileSinkFilter *This,LPCOLESTR pszFileName,const AM_MEDIA_TYPE *pmt);
      HRESULT (WINAPI *GetCurFile)(IFileSinkFilter *This,LPOLESTR *ppszFileName,AM_MEDIA_TYPE *pmt);
    END_INTERFACE
  } IFileSinkFilterVtbl;
  struct IFileSinkFilter {
    CONST_VTBL struct IFileSinkFilterVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IFileSinkFilter_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IFileSinkFilter_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IFileSinkFilter_Release(This) (This)->lpVtbl->Release(This)
#define IFileSinkFilter_SetFileName(This,pszFileName,pmt) (This)->lpVtbl->SetFileName(This,pszFileName,pmt)
#define IFileSinkFilter_GetCurFile(This,ppszFileName,pmt) (This)->lpVtbl->GetCurFile(This,ppszFileName,pmt)
#endif
#endif
  HRESULT WINAPI IFileSinkFilter_SetFileName_Proxy(IFileSinkFilter *This,LPCOLESTR pszFileName,const AM_MEDIA_TYPE *pmt);
  void __RPC_STUB IFileSinkFilter_SetFileName_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFileSinkFilter_GetCurFile_Proxy(IFileSinkFilter *This,LPOLESTR *ppszFileName,AM_MEDIA_TYPE *pmt);
  void __RPC_STUB IFileSinkFilter_GetCurFile_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IFileSinkFilter *PFILTERFILESINK;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0148_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0148_v0_0_s_ifspec;
#ifndef __IFileSinkFilter2_INTERFACE_DEFINED__
#define __IFileSinkFilter2_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IFileSinkFilter2;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IFileSinkFilter2 : public IFileSinkFilter {
  public:
    virtual HRESULT WINAPI SetMode(DWORD dwFlags) = 0;
    virtual HRESULT WINAPI GetMode(DWORD *pdwFlags) = 0;
  };
#else
  typedef struct IFileSinkFilter2Vtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IFileSinkFilter2 *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IFileSinkFilter2 *This);
      ULONG (WINAPI *Release)(IFileSinkFilter2 *This);
      HRESULT (WINAPI *SetFileName)(IFileSinkFilter2 *This,LPCOLESTR pszFileName,const AM_MEDIA_TYPE *pmt);
      HRESULT (WINAPI *GetCurFile)(IFileSinkFilter2 *This,LPOLESTR *ppszFileName,AM_MEDIA_TYPE *pmt);
      HRESULT (WINAPI *SetMode)(IFileSinkFilter2 *This,DWORD dwFlags);
      HRESULT (WINAPI *GetMode)(IFileSinkFilter2 *This,DWORD *pdwFlags);
    END_INTERFACE
  } IFileSinkFilter2Vtbl;
  struct IFileSinkFilter2 {
    CONST_VTBL struct IFileSinkFilter2Vtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IFileSinkFilter2_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IFileSinkFilter2_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IFileSinkFilter2_Release(This) (This)->lpVtbl->Release(This)
#define IFileSinkFilter2_SetFileName(This,pszFileName,pmt) (This)->lpVtbl->SetFileName(This,pszFileName,pmt)
#define IFileSinkFilter2_GetCurFile(This,ppszFileName,pmt) (This)->lpVtbl->GetCurFile(This,ppszFileName,pmt)
#define IFileSinkFilter2_SetMode(This,dwFlags) (This)->lpVtbl->SetMode(This,dwFlags)
#define IFileSinkFilter2_GetMode(This,pdwFlags) (This)->lpVtbl->GetMode(This,pdwFlags)
#endif
#endif
  HRESULT WINAPI IFileSinkFilter2_SetMode_Proxy(IFileSinkFilter2 *This,DWORD dwFlags);
  void __RPC_STUB IFileSinkFilter2_SetMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFileSinkFilter2_GetMode_Proxy(IFileSinkFilter2 *This,DWORD *pdwFlags);
  void __RPC_STUB IFileSinkFilter2_GetMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IFileSinkFilter2 *PFILESINKFILTER2;

  typedef enum __MIDL___MIDL_itf_strmif_0149_0001 {
    AM_FILE_OVERWRITE = 0x1
  } AM_FILESINK_FLAGS;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0149_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0149_v0_0_s_ifspec;
#ifndef __IGraphBuilder_INTERFACE_DEFINED__
#define __IGraphBuilder_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IGraphBuilder;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IGraphBuilder : public IFilterGraph {
  public:
    virtual HRESULT WINAPI Connect(IPin *ppinOut,IPin *ppinIn) = 0;
    virtual HRESULT WINAPI Render(IPin *ppinOut) = 0;
    virtual HRESULT WINAPI RenderFile(LPCWSTR lpcwstrFile,LPCWSTR lpcwstrPlayList) = 0;
    virtual HRESULT WINAPI AddSourceFilter(LPCWSTR lpcwstrFileName,LPCWSTR lpcwstrFilterName,IBaseFilter **ppFilter) = 0;
    virtual HRESULT WINAPI SetLogFile(DWORD_PTR hFile) = 0;
    virtual HRESULT WINAPI Abort(void) = 0;
    virtual HRESULT WINAPI ShouldOperationContinue(void) = 0;
  };
#else
  typedef struct IGraphBuilderVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IGraphBuilder *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IGraphBuilder *This);
      ULONG (WINAPI *Release)(IGraphBuilder *This);
      HRESULT (WINAPI *AddFilter)(IGraphBuilder *This,IBaseFilter *pFilter,LPCWSTR pName);
      HRESULT (WINAPI *RemoveFilter)(IGraphBuilder *This,IBaseFilter *pFilter);
      HRESULT (WINAPI *EnumFilters)(IGraphBuilder *This,IEnumFilters **ppEnum);
      HRESULT (WINAPI *FindFilterByName)(IGraphBuilder *This,LPCWSTR pName,IBaseFilter **ppFilter);
      HRESULT (WINAPI *ConnectDirect)(IGraphBuilder *This,IPin *ppinOut,IPin *ppinIn,const AM_MEDIA_TYPE *pmt);
      HRESULT (WINAPI *Reconnect)(IGraphBuilder *This,IPin *ppin);
      HRESULT (WINAPI *Disconnect)(IGraphBuilder *This,IPin *ppin);
      HRESULT (WINAPI *SetDefaultSyncSource)(IGraphBuilder *This);
      HRESULT (WINAPI *Connect)(IGraphBuilder *This,IPin *ppinOut,IPin *ppinIn);
      HRESULT (WINAPI *Render)(IGraphBuilder *This,IPin *ppinOut);
      HRESULT (WINAPI *RenderFile)(IGraphBuilder *This,LPCWSTR lpcwstrFile,LPCWSTR lpcwstrPlayList);
      HRESULT (WINAPI *AddSourceFilter)(IGraphBuilder *This,LPCWSTR lpcwstrFileName,LPCWSTR lpcwstrFilterName,IBaseFilter **ppFilter);
      HRESULT (WINAPI *SetLogFile)(IGraphBuilder *This,DWORD_PTR hFile);
      HRESULT (WINAPI *Abort)(IGraphBuilder *This);
      HRESULT (WINAPI *ShouldOperationContinue)(IGraphBuilder *This);
    END_INTERFACE
  } IGraphBuilderVtbl;
  struct IGraphBuilder {
    CONST_VTBL struct IGraphBuilderVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IGraphBuilder_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IGraphBuilder_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IGraphBuilder_Release(This) (This)->lpVtbl->Release(This)
#define IGraphBuilder_AddFilter(This,pFilter,pName) (This)->lpVtbl->AddFilter(This,pFilter,pName)
#define IGraphBuilder_RemoveFilter(This,pFilter) (This)->lpVtbl->RemoveFilter(This,pFilter)
#define IGraphBuilder_EnumFilters(This,ppEnum) (This)->lpVtbl->EnumFilters(This,ppEnum)
#define IGraphBuilder_FindFilterByName(This,pName,ppFilter) (This)->lpVtbl->FindFilterByName(This,pName,ppFilter)
#define IGraphBuilder_ConnectDirect(This,ppinOut,ppinIn,pmt) (This)->lpVtbl->ConnectDirect(This,ppinOut,ppinIn,pmt)
#define IGraphBuilder_Reconnect(This,ppin) (This)->lpVtbl->Reconnect(This,ppin)
#define IGraphBuilder_Disconnect(This,ppin) (This)->lpVtbl->Disconnect(This,ppin)
#define IGraphBuilder_SetDefaultSyncSource(This) (This)->lpVtbl->SetDefaultSyncSource(This)
#define IGraphBuilder_Connect(This,ppinOut,ppinIn) (This)->lpVtbl->Connect(This,ppinOut,ppinIn)
#define IGraphBuilder_Render(This,ppinOut) (This)->lpVtbl->Render(This,ppinOut)
#define IGraphBuilder_RenderFile(This,lpcwstrFile,lpcwstrPlayList) (This)->lpVtbl->RenderFile(This,lpcwstrFile,lpcwstrPlayList)
#define IGraphBuilder_AddSourceFilter(This,lpcwstrFileName,lpcwstrFilterName,ppFilter) (This)->lpVtbl->AddSourceFilter(This,lpcwstrFileName,lpcwstrFilterName,ppFilter)
#define IGraphBuilder_SetLogFile(This,hFile) (This)->lpVtbl->SetLogFile(This,hFile)
#define IGraphBuilder_Abort(This) (This)->lpVtbl->Abort(This)
#define IGraphBuilder_ShouldOperationContinue(This) (This)->lpVtbl->ShouldOperationContinue(This)
#endif
#endif
  HRESULT WINAPI IGraphBuilder_Connect_Proxy(IGraphBuilder *This,IPin *ppinOut,IPin *ppinIn);
  void __RPC_STUB IGraphBuilder_Connect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IGraphBuilder_Render_Proxy(IGraphBuilder *This,IPin *ppinOut);
  void __RPC_STUB IGraphBuilder_Render_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IGraphBuilder_RenderFile_Proxy(IGraphBuilder *This,LPCWSTR lpcwstrFile,LPCWSTR lpcwstrPlayList);
  void __RPC_STUB IGraphBuilder_RenderFile_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IGraphBuilder_AddSourceFilter_Proxy(IGraphBuilder *This,LPCWSTR lpcwstrFileName,LPCWSTR lpcwstrFilterName,IBaseFilter **ppFilter);
  void __RPC_STUB IGraphBuilder_AddSourceFilter_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IGraphBuilder_SetLogFile_Proxy(IGraphBuilder *This,DWORD_PTR hFile);
  void __RPC_STUB IGraphBuilder_SetLogFile_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IGraphBuilder_Abort_Proxy(IGraphBuilder *This);
  void __RPC_STUB IGraphBuilder_Abort_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IGraphBuilder_ShouldOperationContinue_Proxy(IGraphBuilder *This);
  void __RPC_STUB IGraphBuilder_ShouldOperationContinue_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __ICaptureGraphBuilder_INTERFACE_DEFINED__
#define __ICaptureGraphBuilder_INTERFACE_DEFINED__
  EXTERN_C const IID IID_ICaptureGraphBuilder;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct ICaptureGraphBuilder : public IUnknown {
  public:
    virtual HRESULT WINAPI SetFiltergraph(IGraphBuilder *pfg) = 0;
    virtual HRESULT WINAPI GetFiltergraph(IGraphBuilder **ppfg) = 0;
    virtual HRESULT WINAPI SetOutputFileName(const GUID *pType,LPCOLESTR lpstrFile,IBaseFilter **ppf,IFileSinkFilter **ppSink) = 0;
    virtual HRESULT WINAPI FindInterface(const GUID *pCategory,IBaseFilter *pf,REFIID riid,void **ppint) = 0;
    virtual HRESULT WINAPI RenderStream(const GUID *pCategory,IUnknown *pSource,IBaseFilter *pfCompressor,IBaseFilter *pfRenderer) = 0;
    virtual HRESULT WINAPI ControlStream(const GUID *pCategory,IBaseFilter *pFilter,REFERENCE_TIME *pstart,REFERENCE_TIME *pstop,WORD wStartCookie,WORD wStopCookie) = 0;
    virtual HRESULT WINAPI AllocCapFile(LPCOLESTR lpstr,DWORDLONG dwlSize) = 0;
    virtual HRESULT WINAPI CopyCaptureFile(LPOLESTR lpwstrOld,LPOLESTR lpwstrNew,int fAllowEscAbort,IAMCopyCaptureFileProgress *pCallback) = 0;
  };
#else
  typedef struct ICaptureGraphBuilderVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(ICaptureGraphBuilder *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(ICaptureGraphBuilder *This);
      ULONG (WINAPI *Release)(ICaptureGraphBuilder *This);
      HRESULT (WINAPI *SetFiltergraph)(ICaptureGraphBuilder *This,IGraphBuilder *pfg);
      HRESULT (WINAPI *GetFiltergraph)(ICaptureGraphBuilder *This,IGraphBuilder **ppfg);
      HRESULT (WINAPI *SetOutputFileName)(ICaptureGraphBuilder *This,const GUID *pType,LPCOLESTR lpstrFile,IBaseFilter **ppf,IFileSinkFilter **ppSink);
      HRESULT (WINAPI *FindInterface)(ICaptureGraphBuilder *This,const GUID *pCategory,IBaseFilter *pf,REFIID riid,void **ppint);
      HRESULT (WINAPI *RenderStream)(ICaptureGraphBuilder *This,const GUID *pCategory,IUnknown *pSource,IBaseFilter *pfCompressor,IBaseFilter *pfRenderer);
      HRESULT (WINAPI *ControlStream)(ICaptureGraphBuilder *This,const GUID *pCategory,IBaseFilter *pFilter,REFERENCE_TIME *pstart,REFERENCE_TIME *pstop,WORD wStartCookie,WORD wStopCookie);
      HRESULT (WINAPI *AllocCapFile)(ICaptureGraphBuilder *This,LPCOLESTR lpstr,DWORDLONG dwlSize);
      HRESULT (WINAPI *CopyCaptureFile)(ICaptureGraphBuilder *This,LPOLESTR lpwstrOld,LPOLESTR lpwstrNew,int fAllowEscAbort,IAMCopyCaptureFileProgress *pCallback);
    END_INTERFACE
  } ICaptureGraphBuilderVtbl;
  struct ICaptureGraphBuilder {
    CONST_VTBL struct ICaptureGraphBuilderVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define ICaptureGraphBuilder_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define ICaptureGraphBuilder_AddRef(This) (This)->lpVtbl->AddRef(This)
#define ICaptureGraphBuilder_Release(This) (This)->lpVtbl->Release(This)
#define ICaptureGraphBuilder_SetFiltergraph(This,pfg) (This)->lpVtbl->SetFiltergraph(This,pfg)
#define ICaptureGraphBuilder_GetFiltergraph(This,ppfg) (This)->lpVtbl->GetFiltergraph(This,ppfg)
#define ICaptureGraphBuilder_SetOutputFileName(This,pType,lpstrFile,ppf,ppSink) (This)->lpVtbl->SetOutputFileName(This,pType,lpstrFile,ppf,ppSink)
#define ICaptureGraphBuilder_FindInterface(This,pCategory,pf,riid,ppint) (This)->lpVtbl->FindInterface(This,pCategory,pf,riid,ppint)
#define ICaptureGraphBuilder_RenderStream(This,pCategory,pSource,pfCompressor,pfRenderer) (This)->lpVtbl->RenderStream(This,pCategory,pSource,pfCompressor,pfRenderer)
#define ICaptureGraphBuilder_ControlStream(This,pCategory,pFilter,pstart,pstop,wStartCookie,wStopCookie) (This)->lpVtbl->ControlStream(This,pCategory,pFilter,pstart,pstop,wStartCookie,wStopCookie)
#define ICaptureGraphBuilder_AllocCapFile(This,lpstr,dwlSize) (This)->lpVtbl->AllocCapFile(This,lpstr,dwlSize)
#define ICaptureGraphBuilder_CopyCaptureFile(This,lpwstrOld,lpwstrNew,fAllowEscAbort,pCallback) (This)->lpVtbl->CopyCaptureFile(This,lpwstrOld,lpwstrNew,fAllowEscAbort,pCallback)
#endif
#endif
  HRESULT WINAPI ICaptureGraphBuilder_SetFiltergraph_Proxy(ICaptureGraphBuilder *This,IGraphBuilder *pfg);
  void __RPC_STUB ICaptureGraphBuilder_SetFiltergraph_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICaptureGraphBuilder_GetFiltergraph_Proxy(ICaptureGraphBuilder *This,IGraphBuilder **ppfg);
  void __RPC_STUB ICaptureGraphBuilder_GetFiltergraph_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICaptureGraphBuilder_SetOutputFileName_Proxy(ICaptureGraphBuilder *This,const GUID *pType,LPCOLESTR lpstrFile,IBaseFilter **ppf,IFileSinkFilter **ppSink);
  void __RPC_STUB ICaptureGraphBuilder_SetOutputFileName_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICaptureGraphBuilder_RemoteFindInterface_Proxy(ICaptureGraphBuilder *This,const GUID *pCategory,IBaseFilter *pf,REFIID riid,IUnknown **ppint);
  void __RPC_STUB ICaptureGraphBuilder_RemoteFindInterface_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICaptureGraphBuilder_RenderStream_Proxy(ICaptureGraphBuilder *This,const GUID *pCategory,IUnknown *pSource,IBaseFilter *pfCompressor,IBaseFilter *pfRenderer);
  void __RPC_STUB ICaptureGraphBuilder_RenderStream_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICaptureGraphBuilder_ControlStream_Proxy(ICaptureGraphBuilder *This,const GUID *pCategory,IBaseFilter *pFilter,REFERENCE_TIME *pstart,REFERENCE_TIME *pstop,WORD wStartCookie,WORD wStopCookie);
  void __RPC_STUB ICaptureGraphBuilder_ControlStream_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICaptureGraphBuilder_AllocCapFile_Proxy(ICaptureGraphBuilder *This,LPCOLESTR lpstr,DWORDLONG dwlSize);
  void __RPC_STUB ICaptureGraphBuilder_AllocCapFile_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICaptureGraphBuilder_CopyCaptureFile_Proxy(ICaptureGraphBuilder *This,LPOLESTR lpwstrOld,LPOLESTR lpwstrNew,int fAllowEscAbort,IAMCopyCaptureFileProgress *pCallback);
  void __RPC_STUB ICaptureGraphBuilder_CopyCaptureFile_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IAMCopyCaptureFileProgress_INTERFACE_DEFINED__
#define __IAMCopyCaptureFileProgress_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMCopyCaptureFileProgress;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMCopyCaptureFileProgress : public IUnknown {
  public:
    virtual HRESULT WINAPI Progress(int iProgress) = 0;
  };
#else
  typedef struct IAMCopyCaptureFileProgressVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMCopyCaptureFileProgress *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMCopyCaptureFileProgress *This);
      ULONG (WINAPI *Release)(IAMCopyCaptureFileProgress *This);
      HRESULT (WINAPI *Progress)(IAMCopyCaptureFileProgress *This,int iProgress);
    END_INTERFACE
  } IAMCopyCaptureFileProgressVtbl;
  struct IAMCopyCaptureFileProgress {
    CONST_VTBL struct IAMCopyCaptureFileProgressVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMCopyCaptureFileProgress_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMCopyCaptureFileProgress_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMCopyCaptureFileProgress_Release(This) (This)->lpVtbl->Release(This)
#define IAMCopyCaptureFileProgress_Progress(This,iProgress) (This)->lpVtbl->Progress(This,iProgress)
#endif
#endif
  HRESULT WINAPI IAMCopyCaptureFileProgress_Progress_Proxy(IAMCopyCaptureFileProgress *This,int iProgress);
  void __RPC_STUB IAMCopyCaptureFileProgress_Progress_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __ICaptureGraphBuilder2_INTERFACE_DEFINED__
#define __ICaptureGraphBuilder2_INTERFACE_DEFINED__
  EXTERN_C const IID IID_ICaptureGraphBuilder2;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct ICaptureGraphBuilder2 : public IUnknown {
  public:
    virtual HRESULT WINAPI SetFiltergraph(IGraphBuilder *pfg) = 0;
    virtual HRESULT WINAPI GetFiltergraph(IGraphBuilder **ppfg) = 0;
    virtual HRESULT WINAPI SetOutputFileName(const GUID *pType,LPCOLESTR lpstrFile,IBaseFilter **ppf,IFileSinkFilter **ppSink) = 0;
    virtual HRESULT WINAPI FindInterface(const GUID *pCategory,const GUID *pType,IBaseFilter *pf,REFIID riid,void **ppint) = 0;
    virtual HRESULT WINAPI RenderStream(const GUID *pCategory,const GUID *pType,IUnknown *pSource,IBaseFilter *pfCompressor,IBaseFilter *pfRenderer) = 0;
    virtual HRESULT WINAPI ControlStream(const GUID *pCategory,const GUID *pType,IBaseFilter *pFilter,REFERENCE_TIME *pstart,REFERENCE_TIME *pstop,WORD wStartCookie,WORD wStopCookie) = 0;
    virtual HRESULT WINAPI AllocCapFile(LPCOLESTR lpstr,DWORDLONG dwlSize) = 0;
    virtual HRESULT WINAPI CopyCaptureFile(LPOLESTR lpwstrOld,LPOLESTR lpwstrNew,int fAllowEscAbort,IAMCopyCaptureFileProgress *pCallback) = 0;
    virtual HRESULT WINAPI FindPin(IUnknown *pSource,PIN_DIRECTION pindir,const GUID *pCategory,const GUID *pType,WINBOOL fUnconnected,int num,IPin **ppPin) = 0;
  };
#else
  typedef struct ICaptureGraphBuilder2Vtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(ICaptureGraphBuilder2 *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(ICaptureGraphBuilder2 *This);
      ULONG (WINAPI *Release)(ICaptureGraphBuilder2 *This);
      HRESULT (WINAPI *SetFiltergraph)(ICaptureGraphBuilder2 *This,IGraphBuilder *pfg);
      HRESULT (WINAPI *GetFiltergraph)(ICaptureGraphBuilder2 *This,IGraphBuilder **ppfg);
      HRESULT (WINAPI *SetOutputFileName)(ICaptureGraphBuilder2 *This,const GUID *pType,LPCOLESTR lpstrFile,IBaseFilter **ppf,IFileSinkFilter **ppSink);
      HRESULT (WINAPI *FindInterface)(ICaptureGraphBuilder2 *This,const GUID *pCategory,const GUID *pType,IBaseFilter *pf,REFIID riid,void **ppint);
      HRESULT (WINAPI *RenderStream)(ICaptureGraphBuilder2 *This,const GUID *pCategory,const GUID *pType,IUnknown *pSource,IBaseFilter *pfCompressor,IBaseFilter *pfRenderer);
      HRESULT (WINAPI *ControlStream)(ICaptureGraphBuilder2 *This,const GUID *pCategory,const GUID *pType,IBaseFilter *pFilter,REFERENCE_TIME *pstart,REFERENCE_TIME *pstop,WORD wStartCookie,WORD wStopCookie);
      HRESULT (WINAPI *AllocCapFile)(ICaptureGraphBuilder2 *This,LPCOLESTR lpstr,DWORDLONG dwlSize);
      HRESULT (WINAPI *CopyCaptureFile)(ICaptureGraphBuilder2 *This,LPOLESTR lpwstrOld,LPOLESTR lpwstrNew,int fAllowEscAbort,IAMCopyCaptureFileProgress *pCallback);
      HRESULT (WINAPI *FindPin)(ICaptureGraphBuilder2 *This,IUnknown *pSource,PIN_DIRECTION pindir,const GUID *pCategory,const GUID *pType,WINBOOL fUnconnected,int num,IPin **ppPin);
    END_INTERFACE
  } ICaptureGraphBuilder2Vtbl;
  struct ICaptureGraphBuilder2 {
    CONST_VTBL struct ICaptureGraphBuilder2Vtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define ICaptureGraphBuilder2_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define ICaptureGraphBuilder2_AddRef(This) (This)->lpVtbl->AddRef(This)
#define ICaptureGraphBuilder2_Release(This) (This)->lpVtbl->Release(This)
#define ICaptureGraphBuilder2_SetFiltergraph(This,pfg) (This)->lpVtbl->SetFiltergraph(This,pfg)
#define ICaptureGraphBuilder2_GetFiltergraph(This,ppfg) (This)->lpVtbl->GetFiltergraph(This,ppfg)
#define ICaptureGraphBuilder2_SetOutputFileName(This,pType,lpstrFile,ppf,ppSink) (This)->lpVtbl->SetOutputFileName(This,pType,lpstrFile,ppf,ppSink)
#define ICaptureGraphBuilder2_FindInterface(This,pCategory,pType,pf,riid,ppint) (This)->lpVtbl->FindInterface(This,pCategory,pType,pf,riid,ppint)
#define ICaptureGraphBuilder2_RenderStream(This,pCategory,pType,pSource,pfCompressor,pfRenderer) (This)->lpVtbl->RenderStream(This,pCategory,pType,pSource,pfCompressor,pfRenderer)
#define ICaptureGraphBuilder2_ControlStream(This,pCategory,pType,pFilter,pstart,pstop,wStartCookie,wStopCookie) (This)->lpVtbl->ControlStream(This,pCategory,pType,pFilter,pstart,pstop,wStartCookie,wStopCookie)
#define ICaptureGraphBuilder2_AllocCapFile(This,lpstr,dwlSize) (This)->lpVtbl->AllocCapFile(This,lpstr,dwlSize)
#define ICaptureGraphBuilder2_CopyCaptureFile(This,lpwstrOld,lpwstrNew,fAllowEscAbort,pCallback) (This)->lpVtbl->CopyCaptureFile(This,lpwstrOld,lpwstrNew,fAllowEscAbort,pCallback)
#define ICaptureGraphBuilder2_FindPin(This,pSource,pindir,pCategory,pType,fUnconnected,num,ppPin) (This)->lpVtbl->FindPin(This,pSource,pindir,pCategory,pType,fUnconnected,num,ppPin)
#endif
#endif
  HRESULT WINAPI ICaptureGraphBuilder2_SetFiltergraph_Proxy(ICaptureGraphBuilder2 *This,IGraphBuilder *pfg);
  void __RPC_STUB ICaptureGraphBuilder2_SetFiltergraph_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICaptureGraphBuilder2_GetFiltergraph_Proxy(ICaptureGraphBuilder2 *This,IGraphBuilder **ppfg);
  void __RPC_STUB ICaptureGraphBuilder2_GetFiltergraph_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICaptureGraphBuilder2_SetOutputFileName_Proxy(ICaptureGraphBuilder2 *This,const GUID *pType,LPCOLESTR lpstrFile,IBaseFilter **ppf,IFileSinkFilter **ppSink);
  void __RPC_STUB ICaptureGraphBuilder2_SetOutputFileName_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICaptureGraphBuilder2_RemoteFindInterface_Proxy(ICaptureGraphBuilder2 *This,const GUID *pCategory,const GUID *pType,IBaseFilter *pf,REFIID riid,IUnknown **ppint);
  void __RPC_STUB ICaptureGraphBuilder2_RemoteFindInterface_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICaptureGraphBuilder2_RenderStream_Proxy(ICaptureGraphBuilder2 *This,const GUID *pCategory,const GUID *pType,IUnknown *pSource,IBaseFilter *pfCompressor,IBaseFilter *pfRenderer);
  void __RPC_STUB ICaptureGraphBuilder2_RenderStream_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICaptureGraphBuilder2_ControlStream_Proxy(ICaptureGraphBuilder2 *This,const GUID *pCategory,const GUID *pType,IBaseFilter *pFilter,REFERENCE_TIME *pstart,REFERENCE_TIME *pstop,WORD wStartCookie,WORD wStopCookie);
  void __RPC_STUB ICaptureGraphBuilder2_ControlStream_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICaptureGraphBuilder2_AllocCapFile_Proxy(ICaptureGraphBuilder2 *This,LPCOLESTR lpstr,DWORDLONG dwlSize);
  void __RPC_STUB ICaptureGraphBuilder2_AllocCapFile_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICaptureGraphBuilder2_CopyCaptureFile_Proxy(ICaptureGraphBuilder2 *This,LPOLESTR lpwstrOld,LPOLESTR lpwstrNew,int fAllowEscAbort,IAMCopyCaptureFileProgress *pCallback);
  void __RPC_STUB ICaptureGraphBuilder2_CopyCaptureFile_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICaptureGraphBuilder2_FindPin_Proxy(ICaptureGraphBuilder2 *This,IUnknown *pSource,PIN_DIRECTION pindir,const GUID *pCategory,const GUID *pType,WINBOOL fUnconnected,int num,IPin **ppPin);
  void __RPC_STUB ICaptureGraphBuilder2_FindPin_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  enum _AM_RENSDEREXFLAGS {
    AM_RENDEREX_RENDERTOEXISTINGRENDERERS = 0x1
  };

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0153_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0153_v0_0_s_ifspec;
#ifndef __IFilterGraph2_INTERFACE_DEFINED__
#define __IFilterGraph2_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IFilterGraph2;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IFilterGraph2 : public IGraphBuilder {
  public:
    virtual HRESULT WINAPI AddSourceFilterForMoniker(IMoniker *pMoniker,IBindCtx *pCtx,LPCWSTR lpcwstrFilterName,IBaseFilter **ppFilter) = 0;
    virtual HRESULT WINAPI ReconnectEx(IPin *ppin,const AM_MEDIA_TYPE *pmt) = 0;
    virtual HRESULT WINAPI RenderEx(IPin *pPinOut,DWORD dwFlags,DWORD *pvContext) = 0;
  };
#else
  typedef struct IFilterGraph2Vtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IFilterGraph2 *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IFilterGraph2 *This);
      ULONG (WINAPI *Release)(IFilterGraph2 *This);
      HRESULT (WINAPI *AddFilter)(IFilterGraph2 *This,IBaseFilter *pFilter,LPCWSTR pName);
      HRESULT (WINAPI *RemoveFilter)(IFilterGraph2 *This,IBaseFilter *pFilter);
      HRESULT (WINAPI *EnumFilters)(IFilterGraph2 *This,IEnumFilters **ppEnum);
      HRESULT (WINAPI *FindFilterByName)(IFilterGraph2 *This,LPCWSTR pName,IBaseFilter **ppFilter);
      HRESULT (WINAPI *ConnectDirect)(IFilterGraph2 *This,IPin *ppinOut,IPin *ppinIn,const AM_MEDIA_TYPE *pmt);
      HRESULT (WINAPI *Reconnect)(IFilterGraph2 *This,IPin *ppin);
      HRESULT (WINAPI *Disconnect)(IFilterGraph2 *This,IPin *ppin);
      HRESULT (WINAPI *SetDefaultSyncSource)(IFilterGraph2 *This);
      HRESULT (WINAPI *Connect)(IFilterGraph2 *This,IPin *ppinOut,IPin *ppinIn);
      HRESULT (WINAPI *Render)(IFilterGraph2 *This,IPin *ppinOut);
      HRESULT (WINAPI *RenderFile)(IFilterGraph2 *This,LPCWSTR lpcwstrFile,LPCWSTR lpcwstrPlayList);
      HRESULT (WINAPI *AddSourceFilter)(IFilterGraph2 *This,LPCWSTR lpcwstrFileName,LPCWSTR lpcwstrFilterName,IBaseFilter **ppFilter);
      HRESULT (WINAPI *SetLogFile)(IFilterGraph2 *This,DWORD_PTR hFile);
      HRESULT (WINAPI *Abort)(IFilterGraph2 *This);
      HRESULT (WINAPI *ShouldOperationContinue)(IFilterGraph2 *This);
      HRESULT (WINAPI *AddSourceFilterForMoniker)(IFilterGraph2 *This,IMoniker *pMoniker,IBindCtx *pCtx,LPCWSTR lpcwstrFilterName,IBaseFilter **ppFilter);
      HRESULT (WINAPI *ReconnectEx)(IFilterGraph2 *This,IPin *ppin,const AM_MEDIA_TYPE *pmt);
      HRESULT (WINAPI *RenderEx)(IFilterGraph2 *This,IPin *pPinOut,DWORD dwFlags,DWORD *pvContext);
    END_INTERFACE
  } IFilterGraph2Vtbl;
  struct IFilterGraph2 {
    CONST_VTBL struct IFilterGraph2Vtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IFilterGraph2_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IFilterGraph2_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IFilterGraph2_Release(This) (This)->lpVtbl->Release(This)
#define IFilterGraph2_AddFilter(This,pFilter,pName) (This)->lpVtbl->AddFilter(This,pFilter,pName)
#define IFilterGraph2_RemoveFilter(This,pFilter) (This)->lpVtbl->RemoveFilter(This,pFilter)
#define IFilterGraph2_EnumFilters(This,ppEnum) (This)->lpVtbl->EnumFilters(This,ppEnum)
#define IFilterGraph2_FindFilterByName(This,pName,ppFilter) (This)->lpVtbl->FindFilterByName(This,pName,ppFilter)
#define IFilterGraph2_ConnectDirect(This,ppinOut,ppinIn,pmt) (This)->lpVtbl->ConnectDirect(This,ppinOut,ppinIn,pmt)
#define IFilterGraph2_Reconnect(This,ppin) (This)->lpVtbl->Reconnect(This,ppin)
#define IFilterGraph2_Disconnect(This,ppin) (This)->lpVtbl->Disconnect(This,ppin)
#define IFilterGraph2_SetDefaultSyncSource(This) (This)->lpVtbl->SetDefaultSyncSource(This)
#define IFilterGraph2_Connect(This,ppinOut,ppinIn) (This)->lpVtbl->Connect(This,ppinOut,ppinIn)
#define IFilterGraph2_Render(This,ppinOut) (This)->lpVtbl->Render(This,ppinOut)
#define IFilterGraph2_RenderFile(This,lpcwstrFile,lpcwstrPlayList) (This)->lpVtbl->RenderFile(This,lpcwstrFile,lpcwstrPlayList)
#define IFilterGraph2_AddSourceFilter(This,lpcwstrFileName,lpcwstrFilterName,ppFilter) (This)->lpVtbl->AddSourceFilter(This,lpcwstrFileName,lpcwstrFilterName,ppFilter)
#define IFilterGraph2_SetLogFile(This,hFile) (This)->lpVtbl->SetLogFile(This,hFile)
#define IFilterGraph2_Abort(This) (This)->lpVtbl->Abort(This)
#define IFilterGraph2_ShouldOperationContinue(This) (This)->lpVtbl->ShouldOperationContinue(This)
#define IFilterGraph2_AddSourceFilterForMoniker(This,pMoniker,pCtx,lpcwstrFilterName,ppFilter) (This)->lpVtbl->AddSourceFilterForMoniker(This,pMoniker,pCtx,lpcwstrFilterName,ppFilter)
#define IFilterGraph2_ReconnectEx(This,ppin,pmt) (This)->lpVtbl->ReconnectEx(This,ppin,pmt)
#define IFilterGraph2_RenderEx(This,pPinOut,dwFlags,pvContext) (This)->lpVtbl->RenderEx(This,pPinOut,dwFlags,pvContext)
#endif
#endif
  HRESULT WINAPI IFilterGraph2_AddSourceFilterForMoniker_Proxy(IFilterGraph2 *This,IMoniker *pMoniker,IBindCtx *pCtx,LPCWSTR lpcwstrFilterName,IBaseFilter **ppFilter);
  void __RPC_STUB IFilterGraph2_AddSourceFilterForMoniker_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterGraph2_ReconnectEx_Proxy(IFilterGraph2 *This,IPin *ppin,const AM_MEDIA_TYPE *pmt);
  void __RPC_STUB IFilterGraph2_ReconnectEx_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterGraph2_RenderEx_Proxy(IFilterGraph2 *This,IPin *pPinOut,DWORD dwFlags,DWORD *pvContext);
  void __RPC_STUB IFilterGraph2_RenderEx_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IStreamBuilder_INTERFACE_DEFINED__
#define __IStreamBuilder_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IStreamBuilder;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IStreamBuilder : public IUnknown {
  public:
    virtual HRESULT WINAPI Render(IPin *ppinOut,IGraphBuilder *pGraph) = 0;
    virtual HRESULT WINAPI Backout(IPin *ppinOut,IGraphBuilder *pGraph) = 0;
  };
#else
  typedef struct IStreamBuilderVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IStreamBuilder *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IStreamBuilder *This);
      ULONG (WINAPI *Release)(IStreamBuilder *This);
      HRESULT (WINAPI *Render)(IStreamBuilder *This,IPin *ppinOut,IGraphBuilder *pGraph);
      HRESULT (WINAPI *Backout)(IStreamBuilder *This,IPin *ppinOut,IGraphBuilder *pGraph);
    END_INTERFACE
  } IStreamBuilderVtbl;
  struct IStreamBuilder {
    CONST_VTBL struct IStreamBuilderVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IStreamBuilder_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IStreamBuilder_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IStreamBuilder_Release(This) (This)->lpVtbl->Release(This)
#define IStreamBuilder_Render(This,ppinOut,pGraph) (This)->lpVtbl->Render(This,ppinOut,pGraph)
#define IStreamBuilder_Backout(This,ppinOut,pGraph) (This)->lpVtbl->Backout(This,ppinOut,pGraph)
#endif
#endif
  HRESULT WINAPI IStreamBuilder_Render_Proxy(IStreamBuilder *This,IPin *ppinOut,IGraphBuilder *pGraph);
  void __RPC_STUB IStreamBuilder_Render_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IStreamBuilder_Backout_Proxy(IStreamBuilder *This,IPin *ppinOut,IGraphBuilder *pGraph);
  void __RPC_STUB IStreamBuilder_Backout_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IAsyncReader_INTERFACE_DEFINED__
#define __IAsyncReader_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAsyncReader;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAsyncReader : public IUnknown {
  public:
    virtual HRESULT WINAPI RequestAllocator(IMemAllocator *pPreferred,ALLOCATOR_PROPERTIES *pProps,IMemAllocator **ppActual) = 0;
    virtual HRESULT WINAPI Request(IMediaSample *pSample,DWORD_PTR dwUser) = 0;
    virtual HRESULT WINAPI WaitForNext(DWORD dwTimeout,IMediaSample **ppSample,DWORD_PTR *pdwUser) = 0;
    virtual HRESULT WINAPI SyncReadAligned(IMediaSample *pSample) = 0;
    virtual HRESULT WINAPI SyncRead(LONGLONG llPosition,LONG lLength,BYTE *pBuffer) = 0;
    virtual HRESULT WINAPI Length(LONGLONG *pTotal,LONGLONG *pAvailable) = 0;
    virtual HRESULT WINAPI BeginFlush(void) = 0;
    virtual HRESULT WINAPI EndFlush(void) = 0;
  };
#else
  typedef struct IAsyncReaderVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAsyncReader *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAsyncReader *This);
      ULONG (WINAPI *Release)(IAsyncReader *This);
      HRESULT (WINAPI *RequestAllocator)(IAsyncReader *This,IMemAllocator *pPreferred,ALLOCATOR_PROPERTIES *pProps,IMemAllocator **ppActual);
      HRESULT (WINAPI *Request)(IAsyncReader *This,IMediaSample *pSample,DWORD_PTR dwUser);
      HRESULT (WINAPI *WaitForNext)(IAsyncReader *This,DWORD dwTimeout,IMediaSample **ppSample,DWORD_PTR *pdwUser);
      HRESULT (WINAPI *SyncReadAligned)(IAsyncReader *This,IMediaSample *pSample);
      HRESULT (WINAPI *SyncRead)(IAsyncReader *This,LONGLONG llPosition,LONG lLength,BYTE *pBuffer);
      HRESULT (WINAPI *Length)(IAsyncReader *This,LONGLONG *pTotal,LONGLONG *pAvailable);
      HRESULT (WINAPI *BeginFlush)(IAsyncReader *This);
      HRESULT (WINAPI *EndFlush)(IAsyncReader *This);
    END_INTERFACE
  } IAsyncReaderVtbl;
  struct IAsyncReader {
    CONST_VTBL struct IAsyncReaderVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAsyncReader_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAsyncReader_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAsyncReader_Release(This) (This)->lpVtbl->Release(This)
#define IAsyncReader_RequestAllocator(This,pPreferred,pProps,ppActual) (This)->lpVtbl->RequestAllocator(This,pPreferred,pProps,ppActual)
#define IAsyncReader_Request(This,pSample,dwUser) (This)->lpVtbl->Request(This,pSample,dwUser)
#define IAsyncReader_WaitForNext(This,dwTimeout,ppSample,pdwUser) (This)->lpVtbl->WaitForNext(This,dwTimeout,ppSample,pdwUser)
#define IAsyncReader_SyncReadAligned(This,pSample) (This)->lpVtbl->SyncReadAligned(This,pSample)
#define IAsyncReader_SyncRead(This,llPosition,lLength,pBuffer) (This)->lpVtbl->SyncRead(This,llPosition,lLength,pBuffer)
#define IAsyncReader_Length(This,pTotal,pAvailable) (This)->lpVtbl->Length(This,pTotal,pAvailable)
#define IAsyncReader_BeginFlush(This) (This)->lpVtbl->BeginFlush(This)
#define IAsyncReader_EndFlush(This) (This)->lpVtbl->EndFlush(This)
#endif
#endif
  HRESULT WINAPI IAsyncReader_RequestAllocator_Proxy(IAsyncReader *This,IMemAllocator *pPreferred,ALLOCATOR_PROPERTIES *pProps,IMemAllocator **ppActual);
  void __RPC_STUB IAsyncReader_RequestAllocator_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAsyncReader_Request_Proxy(IAsyncReader *This,IMediaSample *pSample,DWORD_PTR dwUser);
  void __RPC_STUB IAsyncReader_Request_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAsyncReader_WaitForNext_Proxy(IAsyncReader *This,DWORD dwTimeout,IMediaSample **ppSample,DWORD_PTR *pdwUser);
  void __RPC_STUB IAsyncReader_WaitForNext_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAsyncReader_SyncReadAligned_Proxy(IAsyncReader *This,IMediaSample *pSample);
  void __RPC_STUB IAsyncReader_SyncReadAligned_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAsyncReader_SyncRead_Proxy(IAsyncReader *This,LONGLONG llPosition,LONG lLength,BYTE *pBuffer);
  void __RPC_STUB IAsyncReader_SyncRead_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAsyncReader_Length_Proxy(IAsyncReader *This,LONGLONG *pTotal,LONGLONG *pAvailable);
  void __RPC_STUB IAsyncReader_Length_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAsyncReader_BeginFlush_Proxy(IAsyncReader *This);
  void __RPC_STUB IAsyncReader_BeginFlush_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAsyncReader_EndFlush_Proxy(IAsyncReader *This);
  void __RPC_STUB IAsyncReader_EndFlush_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IGraphVersion_INTERFACE_DEFINED__
#define __IGraphVersion_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IGraphVersion;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IGraphVersion : public IUnknown {
  public:
    virtual HRESULT WINAPI QueryVersion(LONG *pVersion) = 0;
  };
#else
  typedef struct IGraphVersionVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IGraphVersion *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IGraphVersion *This);
      ULONG (WINAPI *Release)(IGraphVersion *This);
      HRESULT (WINAPI *QueryVersion)(IGraphVersion *This,LONG *pVersion);
    END_INTERFACE
  } IGraphVersionVtbl;
  struct IGraphVersion {
    CONST_VTBL struct IGraphVersionVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IGraphVersion_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IGraphVersion_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IGraphVersion_Release(This) (This)->lpVtbl->Release(This)
#define IGraphVersion_QueryVersion(This,pVersion) (This)->lpVtbl->QueryVersion(This,pVersion)
#endif
#endif
  HRESULT WINAPI IGraphVersion_QueryVersion_Proxy(IGraphVersion *This,LONG *pVersion);
  void __RPC_STUB IGraphVersion_QueryVersion_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IResourceConsumer_INTERFACE_DEFINED__
#define __IResourceConsumer_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IResourceConsumer;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IResourceConsumer : public IUnknown {
  public:
    virtual HRESULT WINAPI AcquireResource(LONG idResource) = 0;
    virtual HRESULT WINAPI ReleaseResource(LONG idResource) = 0;
  };
#else
  typedef struct IResourceConsumerVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IResourceConsumer *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IResourceConsumer *This);
      ULONG (WINAPI *Release)(IResourceConsumer *This);
      HRESULT (WINAPI *AcquireResource)(IResourceConsumer *This,LONG idResource);
      HRESULT (WINAPI *ReleaseResource)(IResourceConsumer *This,LONG idResource);
    END_INTERFACE
  } IResourceConsumerVtbl;
  struct IResourceConsumer {
    CONST_VTBL struct IResourceConsumerVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IResourceConsumer_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IResourceConsumer_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IResourceConsumer_Release(This) (This)->lpVtbl->Release(This)
#define IResourceConsumer_AcquireResource(This,idResource) (This)->lpVtbl->AcquireResource(This,idResource)
#define IResourceConsumer_ReleaseResource(This,idResource) (This)->lpVtbl->ReleaseResource(This,idResource)
#endif
#endif
  HRESULT WINAPI IResourceConsumer_AcquireResource_Proxy(IResourceConsumer *This,LONG idResource);
  void __RPC_STUB IResourceConsumer_AcquireResource_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IResourceConsumer_ReleaseResource_Proxy(IResourceConsumer *This,LONG idResource);
  void __RPC_STUB IResourceConsumer_ReleaseResource_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IResourceManager_INTERFACE_DEFINED__
#define __IResourceManager_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IResourceManager;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IResourceManager : public IUnknown {
  public:
    virtual HRESULT WINAPI Register(LPCWSTR pName,LONG cResource,LONG *plToken) = 0;
    virtual HRESULT WINAPI RegisterGroup(LPCWSTR pName,LONG cResource,LONG *palTokens,LONG *plToken) = 0;
    virtual HRESULT WINAPI RequestResource(LONG idResource,IUnknown *pFocusObject,IResourceConsumer *pConsumer) = 0;
    virtual HRESULT WINAPI NotifyAcquire(LONG idResource,IResourceConsumer *pConsumer,HRESULT hr) = 0;
    virtual HRESULT WINAPI NotifyRelease(LONG idResource,IResourceConsumer *pConsumer,WINBOOL bStillWant) = 0;
    virtual HRESULT WINAPI CancelRequest(LONG idResource,IResourceConsumer *pConsumer) = 0;
    virtual HRESULT WINAPI SetFocus(IUnknown *pFocusObject) = 0;
    virtual HRESULT WINAPI ReleaseFocus(IUnknown *pFocusObject) = 0;
  };
#else
  typedef struct IResourceManagerVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IResourceManager *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IResourceManager *This);
      ULONG (WINAPI *Release)(IResourceManager *This);
      HRESULT (WINAPI *Register)(IResourceManager *This,LPCWSTR pName,LONG cResource,LONG *plToken);
      HRESULT (WINAPI *RegisterGroup)(IResourceManager *This,LPCWSTR pName,LONG cResource,LONG *palTokens,LONG *plToken);
      HRESULT (WINAPI *RequestResource)(IResourceManager *This,LONG idResource,IUnknown *pFocusObject,IResourceConsumer *pConsumer);
      HRESULT (WINAPI *NotifyAcquire)(IResourceManager *This,LONG idResource,IResourceConsumer *pConsumer,HRESULT hr);
      HRESULT (WINAPI *NotifyRelease)(IResourceManager *This,LONG idResource,IResourceConsumer *pConsumer,WINBOOL bStillWant);
      HRESULT (WINAPI *CancelRequest)(IResourceManager *This,LONG idResource,IResourceConsumer *pConsumer);
      HRESULT (WINAPI *SetFocus)(IResourceManager *This,IUnknown *pFocusObject);
      HRESULT (WINAPI *ReleaseFocus)(IResourceManager *This,IUnknown *pFocusObject);
    END_INTERFACE
  } IResourceManagerVtbl;
  struct IResourceManager {
    CONST_VTBL struct IResourceManagerVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IResourceManager_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IResourceManager_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IResourceManager_Release(This) (This)->lpVtbl->Release(This)
#define IResourceManager_Register(This,pName,cResource,plToken) (This)->lpVtbl->Register(This,pName,cResource,plToken)
#define IResourceManager_RegisterGroup(This,pName,cResource,palTokens,plToken) (This)->lpVtbl->RegisterGroup(This,pName,cResource,palTokens,plToken)
#define IResourceManager_RequestResource(This,idResource,pFocusObject,pConsumer) (This)->lpVtbl->RequestResource(This,idResource,pFocusObject,pConsumer)
#define IResourceManager_NotifyAcquire(This,idResource,pConsumer,hr) (This)->lpVtbl->NotifyAcquire(This,idResource,pConsumer,hr)
#define IResourceManager_NotifyRelease(This,idResource,pConsumer,bStillWant) (This)->lpVtbl->NotifyRelease(This,idResource,pConsumer,bStillWant)
#define IResourceManager_CancelRequest(This,idResource,pConsumer) (This)->lpVtbl->CancelRequest(This,idResource,pConsumer)
#define IResourceManager_SetFocus(This,pFocusObject) (This)->lpVtbl->SetFocus(This,pFocusObject)
#define IResourceManager_ReleaseFocus(This,pFocusObject) (This)->lpVtbl->ReleaseFocus(This,pFocusObject)
#endif
#endif
  HRESULT WINAPI IResourceManager_Register_Proxy(IResourceManager *This,LPCWSTR pName,LONG cResource,LONG *plToken);
  void __RPC_STUB IResourceManager_Register_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IResourceManager_RegisterGroup_Proxy(IResourceManager *This,LPCWSTR pName,LONG cResource,LONG *palTokens,LONG *plToken);
  void __RPC_STUB IResourceManager_RegisterGroup_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IResourceManager_RequestResource_Proxy(IResourceManager *This,LONG idResource,IUnknown *pFocusObject,IResourceConsumer *pConsumer);
  void __RPC_STUB IResourceManager_RequestResource_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IResourceManager_NotifyAcquire_Proxy(IResourceManager *This,LONG idResource,IResourceConsumer *pConsumer,HRESULT hr);
  void __RPC_STUB IResourceManager_NotifyAcquire_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IResourceManager_NotifyRelease_Proxy(IResourceManager *This,LONG idResource,IResourceConsumer *pConsumer,WINBOOL bStillWant);
  void __RPC_STUB IResourceManager_NotifyRelease_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IResourceManager_CancelRequest_Proxy(IResourceManager *This,LONG idResource,IResourceConsumer *pConsumer);
  void __RPC_STUB IResourceManager_CancelRequest_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IResourceManager_SetFocus_Proxy(IResourceManager *This,IUnknown *pFocusObject);
  void __RPC_STUB IResourceManager_SetFocus_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IResourceManager_ReleaseFocus_Proxy(IResourceManager *This,IUnknown *pFocusObject);
  void __RPC_STUB IResourceManager_ReleaseFocus_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IDistributorNotify_INTERFACE_DEFINED__
#define __IDistributorNotify_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IDistributorNotify;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IDistributorNotify : public IUnknown {
  public:
    virtual HRESULT WINAPI Stop(void) = 0;
    virtual HRESULT WINAPI Pause(void) = 0;
    virtual HRESULT WINAPI Run(REFERENCE_TIME tStart) = 0;
    virtual HRESULT WINAPI SetSyncSource(IReferenceClock *pClock) = 0;
    virtual HRESULT WINAPI NotifyGraphChange(void) = 0;
  };
#else
  typedef struct IDistributorNotifyVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IDistributorNotify *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IDistributorNotify *This);
      ULONG (WINAPI *Release)(IDistributorNotify *This);
      HRESULT (WINAPI *Stop)(IDistributorNotify *This);
      HRESULT (WINAPI *Pause)(IDistributorNotify *This);
      HRESULT (WINAPI *Run)(IDistributorNotify *This,REFERENCE_TIME tStart);
      HRESULT (WINAPI *SetSyncSource)(IDistributorNotify *This,IReferenceClock *pClock);
      HRESULT (WINAPI *NotifyGraphChange)(IDistributorNotify *This);
    END_INTERFACE
  } IDistributorNotifyVtbl;
  struct IDistributorNotify {
    CONST_VTBL struct IDistributorNotifyVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IDistributorNotify_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDistributorNotify_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDistributorNotify_Release(This) (This)->lpVtbl->Release(This)
#define IDistributorNotify_Stop(This) (This)->lpVtbl->Stop(This)
#define IDistributorNotify_Pause(This) (This)->lpVtbl->Pause(This)
#define IDistributorNotify_Run(This,tStart) (This)->lpVtbl->Run(This,tStart)
#define IDistributorNotify_SetSyncSource(This,pClock) (This)->lpVtbl->SetSyncSource(This,pClock)
#define IDistributorNotify_NotifyGraphChange(This) (This)->lpVtbl->NotifyGraphChange(This)
#endif
#endif
  HRESULT WINAPI IDistributorNotify_Stop_Proxy(IDistributorNotify *This);
  void __RPC_STUB IDistributorNotify_Stop_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDistributorNotify_Pause_Proxy(IDistributorNotify *This);
  void __RPC_STUB IDistributorNotify_Pause_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDistributorNotify_Run_Proxy(IDistributorNotify *This,REFERENCE_TIME tStart);
  void __RPC_STUB IDistributorNotify_Run_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDistributorNotify_SetSyncSource_Proxy(IDistributorNotify *This,IReferenceClock *pClock);
  void __RPC_STUB IDistributorNotify_SetSyncSource_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDistributorNotify_NotifyGraphChange_Proxy(IDistributorNotify *This);
  void __RPC_STUB IDistributorNotify_NotifyGraphChange_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef enum __MIDL___MIDL_itf_strmif_0160_0001 {
    AM_STREAM_INFO_START_DEFINED = 0x1,AM_STREAM_INFO_STOP_DEFINED = 0x2,AM_STREAM_INFO_DISCARDING = 0x4,AM_STREAM_INFO_STOP_SEND_EXTRA = 0x10
  } AM_STREAM_INFO_FLAGS;

  typedef struct __MIDL___MIDL_itf_strmif_0160_0002 {
    REFERENCE_TIME tStart;
    REFERENCE_TIME tStop;
    DWORD dwStartCookie;
    DWORD dwStopCookie;
    DWORD dwFlags;
  } AM_STREAM_INFO;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0160_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0160_v0_0_s_ifspec;
#ifndef __IAMStreamControl_INTERFACE_DEFINED__
#define __IAMStreamControl_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMStreamControl;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMStreamControl : public IUnknown {
  public:
    virtual HRESULT WINAPI StartAt(const REFERENCE_TIME *ptStart,DWORD dwCookie) = 0;
    virtual HRESULT WINAPI StopAt(const REFERENCE_TIME *ptStop,WINBOOL bSendExtra,DWORD dwCookie) = 0;
    virtual HRESULT WINAPI GetInfo(AM_STREAM_INFO *pInfo) = 0;
  };
#else
  typedef struct IAMStreamControlVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMStreamControl *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMStreamControl *This);
      ULONG (WINAPI *Release)(IAMStreamControl *This);
      HRESULT (WINAPI *StartAt)(IAMStreamControl *This,const REFERENCE_TIME *ptStart,DWORD dwCookie);
      HRESULT (WINAPI *StopAt)(IAMStreamControl *This,const REFERENCE_TIME *ptStop,WINBOOL bSendExtra,DWORD dwCookie);
      HRESULT (WINAPI *GetInfo)(IAMStreamControl *This,AM_STREAM_INFO *pInfo);
    END_INTERFACE
  } IAMStreamControlVtbl;
  struct IAMStreamControl {
    CONST_VTBL struct IAMStreamControlVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMStreamControl_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMStreamControl_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMStreamControl_Release(This) (This)->lpVtbl->Release(This)
#define IAMStreamControl_StartAt(This,ptStart,dwCookie) (This)->lpVtbl->StartAt(This,ptStart,dwCookie)
#define IAMStreamControl_StopAt(This,ptStop,bSendExtra,dwCookie) (This)->lpVtbl->StopAt(This,ptStop,bSendExtra,dwCookie)
#define IAMStreamControl_GetInfo(This,pInfo) (This)->lpVtbl->GetInfo(This,pInfo)
#endif
#endif
  HRESULT WINAPI IAMStreamControl_StartAt_Proxy(IAMStreamControl *This,const REFERENCE_TIME *ptStart,DWORD dwCookie);
  void __RPC_STUB IAMStreamControl_StartAt_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMStreamControl_StopAt_Proxy(IAMStreamControl *This,const REFERENCE_TIME *ptStop,WINBOOL bSendExtra,DWORD dwCookie);
  void __RPC_STUB IAMStreamControl_StopAt_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMStreamControl_GetInfo_Proxy(IAMStreamControl *This,AM_STREAM_INFO *pInfo);
  void __RPC_STUB IAMStreamControl_GetInfo_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __ISeekingPassThru_INTERFACE_DEFINED__
#define __ISeekingPassThru_INTERFACE_DEFINED__
  EXTERN_C const IID IID_ISeekingPassThru;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct ISeekingPassThru : public IUnknown {
  public:
    virtual HRESULT WINAPI Init(WINBOOL bSupportRendering,IPin *pPin) = 0;
  };
#else
  typedef struct ISeekingPassThruVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(ISeekingPassThru *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(ISeekingPassThru *This);
      ULONG (WINAPI *Release)(ISeekingPassThru *This);
      HRESULT (WINAPI *Init)(ISeekingPassThru *This,WINBOOL bSupportRendering,IPin *pPin);
    END_INTERFACE
  } ISeekingPassThruVtbl;
  struct ISeekingPassThru {
    CONST_VTBL struct ISeekingPassThruVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define ISeekingPassThru_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define ISeekingPassThru_AddRef(This) (This)->lpVtbl->AddRef(This)
#define ISeekingPassThru_Release(This) (This)->lpVtbl->Release(This)
#define ISeekingPassThru_Init(This,bSupportRendering,pPin) (This)->lpVtbl->Init(This,bSupportRendering,pPin)
#endif
#endif
  HRESULT WINAPI ISeekingPassThru_Init_Proxy(ISeekingPassThru *This,WINBOOL bSupportRendering,IPin *pPin);
  void __RPC_STUB ISeekingPassThru_Init_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IAMStreamConfig_INTERFACE_DEFINED__
#define __IAMStreamConfig_INTERFACE_DEFINED__
  typedef struct _VIDEO_STREAM_CONFIG_CAPS {
    GUID guid;
    ULONG VideoStandard;
    SIZE InputSize;
    SIZE MinCroppingSize;
    SIZE MaxCroppingSize;
    int CropGranularityX;
    int CropGranularityY;
    int CropAlignX;
    int CropAlignY;
    SIZE MinOutputSize;
    SIZE MaxOutputSize;
    int OutputGranularityX;
    int OutputGranularityY;
    int StretchTapsX;
    int StretchTapsY;
    int ShrinkTapsX;
    int ShrinkTapsY;
    LONGLONG MinFrameInterval;
    LONGLONG MaxFrameInterval;
    LONG MinBitsPerSecond;
    LONG MaxBitsPerSecond;
  } VIDEO_STREAM_CONFIG_CAPS;

  typedef struct _AUDIO_STREAM_CONFIG_CAPS {
    GUID guid;
    ULONG MinimumChannels;
    ULONG MaximumChannels;
    ULONG ChannelsGranularity;
    ULONG MinimumBitsPerSample;
    ULONG MaximumBitsPerSample;
    ULONG BitsPerSampleGranularity;
    ULONG MinimumSampleFrequency;
    ULONG MaximumSampleFrequency;
    ULONG SampleFrequencyGranularity;
  } AUDIO_STREAM_CONFIG_CAPS;

  EXTERN_C const IID IID_IAMStreamConfig;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMStreamConfig : public IUnknown {
  public:
    virtual HRESULT WINAPI SetFormat(AM_MEDIA_TYPE *pmt) = 0;
    virtual HRESULT WINAPI GetFormat(AM_MEDIA_TYPE **ppmt) = 0;
    virtual HRESULT WINAPI GetNumberOfCapabilities(int *piCount,int *piSize) = 0;
    virtual HRESULT WINAPI GetStreamCaps(int iIndex,AM_MEDIA_TYPE **ppmt,BYTE *pSCC) = 0;
  };
#else
  typedef struct IAMStreamConfigVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMStreamConfig *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMStreamConfig *This);
      ULONG (WINAPI *Release)(IAMStreamConfig *This);
      HRESULT (WINAPI *SetFormat)(IAMStreamConfig *This,AM_MEDIA_TYPE *pmt);
      HRESULT (WINAPI *GetFormat)(IAMStreamConfig *This,AM_MEDIA_TYPE **ppmt);
      HRESULT (WINAPI *GetNumberOfCapabilities)(IAMStreamConfig *This,int *piCount,int *piSize);
      HRESULT (WINAPI *GetStreamCaps)(IAMStreamConfig *This,int iIndex,AM_MEDIA_TYPE **ppmt,BYTE *pSCC);
    END_INTERFACE
  } IAMStreamConfigVtbl;
  struct IAMStreamConfig {
    CONST_VTBL struct IAMStreamConfigVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMStreamConfig_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMStreamConfig_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMStreamConfig_Release(This) (This)->lpVtbl->Release(This)
#define IAMStreamConfig_SetFormat(This,pmt) (This)->lpVtbl->SetFormat(This,pmt)
#define IAMStreamConfig_GetFormat(This,ppmt) (This)->lpVtbl->GetFormat(This,ppmt)
#define IAMStreamConfig_GetNumberOfCapabilities(This,piCount,piSize) (This)->lpVtbl->GetNumberOfCapabilities(This,piCount,piSize)
#define IAMStreamConfig_GetStreamCaps(This,iIndex,ppmt,pSCC) (This)->lpVtbl->GetStreamCaps(This,iIndex,ppmt,pSCC)
#endif
#endif
  HRESULT WINAPI IAMStreamConfig_SetFormat_Proxy(IAMStreamConfig *This,AM_MEDIA_TYPE *pmt);
  void __RPC_STUB IAMStreamConfig_SetFormat_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMStreamConfig_GetFormat_Proxy(IAMStreamConfig *This,AM_MEDIA_TYPE **ppmt);
  void __RPC_STUB IAMStreamConfig_GetFormat_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMStreamConfig_GetNumberOfCapabilities_Proxy(IAMStreamConfig *This,int *piCount,int *piSize);
  void __RPC_STUB IAMStreamConfig_GetNumberOfCapabilities_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMStreamConfig_GetStreamCaps_Proxy(IAMStreamConfig *This,int iIndex,AM_MEDIA_TYPE **ppmt,BYTE *pSCC);
  void __RPC_STUB IAMStreamConfig_GetStreamCaps_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IConfigInterleaving_INTERFACE_DEFINED__
#define __IConfigInterleaving_INTERFACE_DEFINED__
  typedef enum __MIDL_IConfigInterleaving_0001 {
    INTERLEAVE_NONE = 0,
    INTERLEAVE_CAPTURE,INTERLEAVE_FULL,INTERLEAVE_NONE_BUFFERED
  } InterleavingMode;

  EXTERN_C const IID IID_IConfigInterleaving;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IConfigInterleaving : public IUnknown {
  public:
    virtual HRESULT WINAPI put_Mode(InterleavingMode mode) = 0;
    virtual HRESULT WINAPI get_Mode(InterleavingMode *pMode) = 0;
    virtual HRESULT WINAPI put_Interleaving(const REFERENCE_TIME *prtInterleave,const REFERENCE_TIME *prtPreroll) = 0;
    virtual HRESULT WINAPI get_Interleaving(REFERENCE_TIME *prtInterleave,REFERENCE_TIME *prtPreroll) = 0;
  };
#else
  typedef struct IConfigInterleavingVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IConfigInterleaving *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IConfigInterleaving *This);
      ULONG (WINAPI *Release)(IConfigInterleaving *This);
      HRESULT (WINAPI *put_Mode)(IConfigInterleaving *This,InterleavingMode mode);
      HRESULT (WINAPI *get_Mode)(IConfigInterleaving *This,InterleavingMode *pMode);
      HRESULT (WINAPI *put_Interleaving)(IConfigInterleaving *This,const REFERENCE_TIME *prtInterleave,const REFERENCE_TIME *prtPreroll);
      HRESULT (WINAPI *get_Interleaving)(IConfigInterleaving *This,REFERENCE_TIME *prtInterleave,REFERENCE_TIME *prtPreroll);
    END_INTERFACE
  } IConfigInterleavingVtbl;
  struct IConfigInterleaving {
    CONST_VTBL struct IConfigInterleavingVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IConfigInterleaving_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IConfigInterleaving_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IConfigInterleaving_Release(This) (This)->lpVtbl->Release(This)
#define IConfigInterleaving_put_Mode(This,mode) (This)->lpVtbl->put_Mode(This,mode)
#define IConfigInterleaving_get_Mode(This,pMode) (This)->lpVtbl->get_Mode(This,pMode)
#define IConfigInterleaving_put_Interleaving(This,prtInterleave,prtPreroll) (This)->lpVtbl->put_Interleaving(This,prtInterleave,prtPreroll)
#define IConfigInterleaving_get_Interleaving(This,prtInterleave,prtPreroll) (This)->lpVtbl->get_Interleaving(This,prtInterleave,prtPreroll)
#endif
#endif
  HRESULT WINAPI IConfigInterleaving_put_Mode_Proxy(IConfigInterleaving *This,InterleavingMode mode);
  void __RPC_STUB IConfigInterleaving_put_Mode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IConfigInterleaving_get_Mode_Proxy(IConfigInterleaving *This,InterleavingMode *pMode);
  void __RPC_STUB IConfigInterleaving_get_Mode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IConfigInterleaving_put_Interleaving_Proxy(IConfigInterleaving *This,const REFERENCE_TIME *prtInterleave,const REFERENCE_TIME *prtPreroll);
  void __RPC_STUB IConfigInterleaving_put_Interleaving_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IConfigInterleaving_get_Interleaving_Proxy(IConfigInterleaving *This,REFERENCE_TIME *prtInterleave,REFERENCE_TIME *prtPreroll);
  void __RPC_STUB IConfigInterleaving_get_Interleaving_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IConfigAviMux_INTERFACE_DEFINED__
#define __IConfigAviMux_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IConfigAviMux;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IConfigAviMux : public IUnknown {
  public:
    virtual HRESULT WINAPI SetMasterStream(LONG iStream) = 0;
    virtual HRESULT WINAPI GetMasterStream(LONG *pStream) = 0;
    virtual HRESULT WINAPI SetOutputCompatibilityIndex(WINBOOL fOldIndex) = 0;
    virtual HRESULT WINAPI GetOutputCompatibilityIndex(WINBOOL *pfOldIndex) = 0;
  };
#else
  typedef struct IConfigAviMuxVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IConfigAviMux *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IConfigAviMux *This);
      ULONG (WINAPI *Release)(IConfigAviMux *This);
      HRESULT (WINAPI *SetMasterStream)(IConfigAviMux *This,LONG iStream);
      HRESULT (WINAPI *GetMasterStream)(IConfigAviMux *This,LONG *pStream);
      HRESULT (WINAPI *SetOutputCompatibilityIndex)(IConfigAviMux *This,WINBOOL fOldIndex);
      HRESULT (WINAPI *GetOutputCompatibilityIndex)(IConfigAviMux *This,WINBOOL *pfOldIndex);
    END_INTERFACE
  } IConfigAviMuxVtbl;
  struct IConfigAviMux {
    CONST_VTBL struct IConfigAviMuxVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IConfigAviMux_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IConfigAviMux_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IConfigAviMux_Release(This) (This)->lpVtbl->Release(This)
#define IConfigAviMux_SetMasterStream(This,iStream) (This)->lpVtbl->SetMasterStream(This,iStream)
#define IConfigAviMux_GetMasterStream(This,pStream) (This)->lpVtbl->GetMasterStream(This,pStream)
#define IConfigAviMux_SetOutputCompatibilityIndex(This,fOldIndex) (This)->lpVtbl->SetOutputCompatibilityIndex(This,fOldIndex)
#define IConfigAviMux_GetOutputCompatibilityIndex(This,pfOldIndex) (This)->lpVtbl->GetOutputCompatibilityIndex(This,pfOldIndex)
#endif
#endif
  HRESULT WINAPI IConfigAviMux_SetMasterStream_Proxy(IConfigAviMux *This,LONG iStream);
  void __RPC_STUB IConfigAviMux_SetMasterStream_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IConfigAviMux_GetMasterStream_Proxy(IConfigAviMux *This,LONG *pStream);
  void __RPC_STUB IConfigAviMux_GetMasterStream_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IConfigAviMux_SetOutputCompatibilityIndex_Proxy(IConfigAviMux *This,WINBOOL fOldIndex);
  void __RPC_STUB IConfigAviMux_SetOutputCompatibilityIndex_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IConfigAviMux_GetOutputCompatibilityIndex_Proxy(IConfigAviMux *This,WINBOOL *pfOldIndex);
  void __RPC_STUB IConfigAviMux_GetOutputCompatibilityIndex_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef enum __MIDL___MIDL_itf_strmif_0167_0001 {
    CompressionCaps_CanQuality = 0x1,CompressionCaps_CanCrunch = 0x2,CompressionCaps_CanKeyFrame = 0x4,CompressionCaps_CanBFrame = 0x8,
    CompressionCaps_CanWindow = 0x10
  } CompressionCaps;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0167_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0167_v0_0_s_ifspec;
#ifndef __IAMVideoCompression_INTERFACE_DEFINED__
#define __IAMVideoCompression_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMVideoCompression;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMVideoCompression : public IUnknown {
  public:
    virtual HRESULT WINAPI put_KeyFrameRate(long KeyFrameRate) = 0;
    virtual HRESULT WINAPI get_KeyFrameRate(long *pKeyFrameRate) = 0;
    virtual HRESULT WINAPI put_PFramesPerKeyFrame(long PFramesPerKeyFrame) = 0;
    virtual HRESULT WINAPI get_PFramesPerKeyFrame(long *pPFramesPerKeyFrame) = 0;
    virtual HRESULT WINAPI put_Quality(double Quality) = 0;
    virtual HRESULT WINAPI get_Quality(double *pQuality) = 0;
    virtual HRESULT WINAPI put_WindowSize(DWORDLONG WindowSize) = 0;
    virtual HRESULT WINAPI get_WindowSize(DWORDLONG *pWindowSize) = 0;
    virtual HRESULT WINAPI GetInfo(WCHAR *pszVersion,int *pcbVersion,LPWSTR pszDescription,int *pcbDescription,long *pDefaultKeyFrameRate,long *pDefaultPFramesPerKey,double *pDefaultQuality,long *pCapabilities) = 0;
    virtual HRESULT WINAPI OverrideKeyFrame(long FrameNumber) = 0;
    virtual HRESULT WINAPI OverrideFrameSize(long FrameNumber,long Size) = 0;
  };
#else
  typedef struct IAMVideoCompressionVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMVideoCompression *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMVideoCompression *This);
      ULONG (WINAPI *Release)(IAMVideoCompression *This);
      HRESULT (WINAPI *put_KeyFrameRate)(IAMVideoCompression *This,long KeyFrameRate);
      HRESULT (WINAPI *get_KeyFrameRate)(IAMVideoCompression *This,long *pKeyFrameRate);
      HRESULT (WINAPI *put_PFramesPerKeyFrame)(IAMVideoCompression *This,long PFramesPerKeyFrame);
      HRESULT (WINAPI *get_PFramesPerKeyFrame)(IAMVideoCompression *This,long *pPFramesPerKeyFrame);
      HRESULT (WINAPI *put_Quality)(IAMVideoCompression *This,double Quality);
      HRESULT (WINAPI *get_Quality)(IAMVideoCompression *This,double *pQuality);
      HRESULT (WINAPI *put_WindowSize)(IAMVideoCompression *This,DWORDLONG WindowSize);
      HRESULT (WINAPI *get_WindowSize)(IAMVideoCompression *This,DWORDLONG *pWindowSize);
      HRESULT (WINAPI *GetInfo)(IAMVideoCompression *This,WCHAR *pszVersion,int *pcbVersion,LPWSTR pszDescription,int *pcbDescription,long *pDefaultKeyFrameRate,long *pDefaultPFramesPerKey,double *pDefaultQuality,long *pCapabilities);
      HRESULT (WINAPI *OverrideKeyFrame)(IAMVideoCompression *This,long FrameNumber);
      HRESULT (WINAPI *OverrideFrameSize)(IAMVideoCompression *This,long FrameNumber,long Size);
    END_INTERFACE
  } IAMVideoCompressionVtbl;
  struct IAMVideoCompression {
    CONST_VTBL struct IAMVideoCompressionVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMVideoCompression_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMVideoCompression_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMVideoCompression_Release(This) (This)->lpVtbl->Release(This)
#define IAMVideoCompression_put_KeyFrameRate(This,KeyFrameRate) (This)->lpVtbl->put_KeyFrameRate(This,KeyFrameRate)
#define IAMVideoCompression_get_KeyFrameRate(This,pKeyFrameRate) (This)->lpVtbl->get_KeyFrameRate(This,pKeyFrameRate)
#define IAMVideoCompression_put_PFramesPerKeyFrame(This,PFramesPerKeyFrame) (This)->lpVtbl->put_PFramesPerKeyFrame(This,PFramesPerKeyFrame)
#define IAMVideoCompression_get_PFramesPerKeyFrame(This,pPFramesPerKeyFrame) (This)->lpVtbl->get_PFramesPerKeyFrame(This,pPFramesPerKeyFrame)
#define IAMVideoCompression_put_Quality(This,Quality) (This)->lpVtbl->put_Quality(This,Quality)
#define IAMVideoCompression_get_Quality(This,pQuality) (This)->lpVtbl->get_Quality(This,pQuality)
#define IAMVideoCompression_put_WindowSize(This,WindowSize) (This)->lpVtbl->put_WindowSize(This,WindowSize)
#define IAMVideoCompression_get_WindowSize(This,pWindowSize) (This)->lpVtbl->get_WindowSize(This,pWindowSize)
#define IAMVideoCompression_GetInfo(This,pszVersion,pcbVersion,pszDescription,pcbDescription,pDefaultKeyFrameRate,pDefaultPFramesPerKey,pDefaultQuality,pCapabilities) (This)->lpVtbl->GetInfo(This,pszVersion,pcbVersion,pszDescription,pcbDescription,pDefaultKeyFrameRate,pDefaultPFramesPerKey,pDefaultQuality,pCapabilities)
#define IAMVideoCompression_OverrideKeyFrame(This,FrameNumber) (This)->lpVtbl->OverrideKeyFrame(This,FrameNumber)
#define IAMVideoCompression_OverrideFrameSize(This,FrameNumber,Size) (This)->lpVtbl->OverrideFrameSize(This,FrameNumber,Size)
#endif
#endif
  HRESULT WINAPI IAMVideoCompression_put_KeyFrameRate_Proxy(IAMVideoCompression *This,long KeyFrameRate);
  void __RPC_STUB IAMVideoCompression_put_KeyFrameRate_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVideoCompression_get_KeyFrameRate_Proxy(IAMVideoCompression *This,long *pKeyFrameRate);
  void __RPC_STUB IAMVideoCompression_get_KeyFrameRate_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVideoCompression_put_PFramesPerKeyFrame_Proxy(IAMVideoCompression *This,long PFramesPerKeyFrame);
  void __RPC_STUB IAMVideoCompression_put_PFramesPerKeyFrame_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVideoCompression_get_PFramesPerKeyFrame_Proxy(IAMVideoCompression *This,long *pPFramesPerKeyFrame);
  void __RPC_STUB IAMVideoCompression_get_PFramesPerKeyFrame_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVideoCompression_put_Quality_Proxy(IAMVideoCompression *This,double Quality);
  void __RPC_STUB IAMVideoCompression_put_Quality_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVideoCompression_get_Quality_Proxy(IAMVideoCompression *This,double *pQuality);
  void __RPC_STUB IAMVideoCompression_get_Quality_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVideoCompression_put_WindowSize_Proxy(IAMVideoCompression *This,DWORDLONG WindowSize);
  void __RPC_STUB IAMVideoCompression_put_WindowSize_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVideoCompression_get_WindowSize_Proxy(IAMVideoCompression *This,DWORDLONG *pWindowSize);
  void __RPC_STUB IAMVideoCompression_get_WindowSize_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVideoCompression_GetInfo_Proxy(IAMVideoCompression *This,WCHAR *pszVersion,int *pcbVersion,LPWSTR pszDescription,int *pcbDescription,long *pDefaultKeyFrameRate,long *pDefaultPFramesPerKey,double *pDefaultQuality,long *pCapabilities);
  void __RPC_STUB IAMVideoCompression_GetInfo_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVideoCompression_OverrideKeyFrame_Proxy(IAMVideoCompression *This,long FrameNumber);
  void __RPC_STUB IAMVideoCompression_OverrideKeyFrame_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVideoCompression_OverrideFrameSize_Proxy(IAMVideoCompression *This,long FrameNumber,long Size);
  void __RPC_STUB IAMVideoCompression_OverrideFrameSize_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef enum __MIDL___MIDL_itf_strmif_0168_0001 {
    VfwCaptureDialog_Source = 0x1,VfwCaptureDialog_Format = 0x2,VfwCaptureDialog_Display = 0x4
  } VfwCaptureDialogs;

  typedef enum __MIDL___MIDL_itf_strmif_0168_0002 {
    VfwCompressDialog_Config = 0x1,VfwCompressDialog_About = 0x2,VfwCompressDialog_QueryConfig = 0x4,VfwCompressDialog_QueryAbout = 0x8
  } VfwCompressDialogs;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0168_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0168_v0_0_s_ifspec;
#ifndef __IAMVfwCaptureDialogs_INTERFACE_DEFINED__
#define __IAMVfwCaptureDialogs_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMVfwCaptureDialogs;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMVfwCaptureDialogs : public IUnknown {
  public:
    virtual HRESULT WINAPI HasDialog(int iDialog) = 0;
    virtual HRESULT WINAPI ShowDialog(int iDialog,HWND hwnd) = 0;
    virtual HRESULT WINAPI SendDriverMessage(int iDialog,int uMsg,long dw1,long dw2) = 0;
  };
#else
  typedef struct IAMVfwCaptureDialogsVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMVfwCaptureDialogs *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMVfwCaptureDialogs *This);
      ULONG (WINAPI *Release)(IAMVfwCaptureDialogs *This);
      HRESULT (WINAPI *HasDialog)(IAMVfwCaptureDialogs *This,int iDialog);
      HRESULT (WINAPI *ShowDialog)(IAMVfwCaptureDialogs *This,int iDialog,HWND hwnd);
      HRESULT (WINAPI *SendDriverMessage)(IAMVfwCaptureDialogs *This,int iDialog,int uMsg,long dw1,long dw2);
    END_INTERFACE
  } IAMVfwCaptureDialogsVtbl;
  struct IAMVfwCaptureDialogs {
    CONST_VTBL struct IAMVfwCaptureDialogsVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMVfwCaptureDialogs_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMVfwCaptureDialogs_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMVfwCaptureDialogs_Release(This) (This)->lpVtbl->Release(This)
#define IAMVfwCaptureDialogs_HasDialog(This,iDialog) (This)->lpVtbl->HasDialog(This,iDialog)
#define IAMVfwCaptureDialogs_ShowDialog(This,iDialog,hwnd) (This)->lpVtbl->ShowDialog(This,iDialog,hwnd)
#define IAMVfwCaptureDialogs_SendDriverMessage(This,iDialog,uMsg,dw1,dw2) (This)->lpVtbl->SendDriverMessage(This,iDialog,uMsg,dw1,dw2)
#endif
#endif
  HRESULT WINAPI IAMVfwCaptureDialogs_HasDialog_Proxy(IAMVfwCaptureDialogs *This,int iDialog);
  void __RPC_STUB IAMVfwCaptureDialogs_HasDialog_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVfwCaptureDialogs_ShowDialog_Proxy(IAMVfwCaptureDialogs *This,int iDialog,HWND hwnd);
  void __RPC_STUB IAMVfwCaptureDialogs_ShowDialog_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVfwCaptureDialogs_SendDriverMessage_Proxy(IAMVfwCaptureDialogs *This,int iDialog,int uMsg,long dw1,long dw2);
  void __RPC_STUB IAMVfwCaptureDialogs_SendDriverMessage_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IAMVfwCompressDialogs_INTERFACE_DEFINED__
#define __IAMVfwCompressDialogs_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMVfwCompressDialogs;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMVfwCompressDialogs : public IUnknown {
  public:
    virtual HRESULT WINAPI ShowDialog(int iDialog,HWND hwnd) = 0;
    virtual HRESULT WINAPI GetState(LPVOID pState,int *pcbState) = 0;
    virtual HRESULT WINAPI SetState(LPVOID pState,int cbState) = 0;
    virtual HRESULT WINAPI SendDriverMessage(int uMsg,long dw1,long dw2) = 0;
  };
#else
  typedef struct IAMVfwCompressDialogsVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMVfwCompressDialogs *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMVfwCompressDialogs *This);
      ULONG (WINAPI *Release)(IAMVfwCompressDialogs *This);
      HRESULT (WINAPI *ShowDialog)(IAMVfwCompressDialogs *This,int iDialog,HWND hwnd);
      HRESULT (WINAPI *GetState)(IAMVfwCompressDialogs *This,LPVOID pState,int *pcbState);
      HRESULT (WINAPI *SetState)(IAMVfwCompressDialogs *This,LPVOID pState,int cbState);
      HRESULT (WINAPI *SendDriverMessage)(IAMVfwCompressDialogs *This,int uMsg,long dw1,long dw2);
    END_INTERFACE
  } IAMVfwCompressDialogsVtbl;
  struct IAMVfwCompressDialogs {
    CONST_VTBL struct IAMVfwCompressDialogsVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMVfwCompressDialogs_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMVfwCompressDialogs_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMVfwCompressDialogs_Release(This) (This)->lpVtbl->Release(This)
#define IAMVfwCompressDialogs_ShowDialog(This,iDialog,hwnd) (This)->lpVtbl->ShowDialog(This,iDialog,hwnd)
#define IAMVfwCompressDialogs_GetState(This,pState,pcbState) (This)->lpVtbl->GetState(This,pState,pcbState)
#define IAMVfwCompressDialogs_SetState(This,pState,cbState) (This)->lpVtbl->SetState(This,pState,cbState)
#define IAMVfwCompressDialogs_SendDriverMessage(This,uMsg,dw1,dw2) (This)->lpVtbl->SendDriverMessage(This,uMsg,dw1,dw2)
#endif
#endif
  HRESULT WINAPI IAMVfwCompressDialogs_ShowDialog_Proxy(IAMVfwCompressDialogs *This,int iDialog,HWND hwnd);
  void __RPC_STUB IAMVfwCompressDialogs_ShowDialog_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVfwCompressDialogs_GetState_Proxy(IAMVfwCompressDialogs *This,LPVOID pState,int *pcbState);
  void __RPC_STUB IAMVfwCompressDialogs_GetState_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVfwCompressDialogs_SetState_Proxy(IAMVfwCompressDialogs *This,LPVOID pState,int cbState);
  void __RPC_STUB IAMVfwCompressDialogs_SetState_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVfwCompressDialogs_SendDriverMessage_Proxy(IAMVfwCompressDialogs *This,int uMsg,long dw1,long dw2);
  void __RPC_STUB IAMVfwCompressDialogs_SendDriverMessage_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IAMDroppedFrames_INTERFACE_DEFINED__
#define __IAMDroppedFrames_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMDroppedFrames;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMDroppedFrames : public IUnknown {
  public:
    virtual HRESULT WINAPI GetNumDropped(long *plDropped) = 0;
    virtual HRESULT WINAPI GetNumNotDropped(long *plNotDropped) = 0;
    virtual HRESULT WINAPI GetDroppedInfo(long lSize,long *plArray,long *plNumCopied) = 0;
    virtual HRESULT WINAPI GetAverageFrameSize(long *plAverageSize) = 0;
  };
#else
  typedef struct IAMDroppedFramesVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMDroppedFrames *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMDroppedFrames *This);
      ULONG (WINAPI *Release)(IAMDroppedFrames *This);
      HRESULT (WINAPI *GetNumDropped)(IAMDroppedFrames *This,long *plDropped);
      HRESULT (WINAPI *GetNumNotDropped)(IAMDroppedFrames *This,long *plNotDropped);
      HRESULT (WINAPI *GetDroppedInfo)(IAMDroppedFrames *This,long lSize,long *plArray,long *plNumCopied);
      HRESULT (WINAPI *GetAverageFrameSize)(IAMDroppedFrames *This,long *plAverageSize);
    END_INTERFACE
  } IAMDroppedFramesVtbl;
  struct IAMDroppedFrames {
    CONST_VTBL struct IAMDroppedFramesVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMDroppedFrames_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMDroppedFrames_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMDroppedFrames_Release(This) (This)->lpVtbl->Release(This)
#define IAMDroppedFrames_GetNumDropped(This,plDropped) (This)->lpVtbl->GetNumDropped(This,plDropped)
#define IAMDroppedFrames_GetNumNotDropped(This,plNotDropped) (This)->lpVtbl->GetNumNotDropped(This,plNotDropped)
#define IAMDroppedFrames_GetDroppedInfo(This,lSize,plArray,plNumCopied) (This)->lpVtbl->GetDroppedInfo(This,lSize,plArray,plNumCopied)
#define IAMDroppedFrames_GetAverageFrameSize(This,plAverageSize) (This)->lpVtbl->GetAverageFrameSize(This,plAverageSize)
#endif
#endif
  HRESULT WINAPI IAMDroppedFrames_GetNumDropped_Proxy(IAMDroppedFrames *This,long *plDropped);
  void __RPC_STUB IAMDroppedFrames_GetNumDropped_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMDroppedFrames_GetNumNotDropped_Proxy(IAMDroppedFrames *This,long *plNotDropped);
  void __RPC_STUB IAMDroppedFrames_GetNumNotDropped_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMDroppedFrames_GetDroppedInfo_Proxy(IAMDroppedFrames *This,long lSize,long *plArray,long *plNumCopied);
  void __RPC_STUB IAMDroppedFrames_GetDroppedInfo_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMDroppedFrames_GetAverageFrameSize_Proxy(IAMDroppedFrames *This,long *plAverageSize);
  void __RPC_STUB IAMDroppedFrames_GetAverageFrameSize_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#define AMF_AUTOMATICGAIN -1.0

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0171_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0171_v0_0_s_ifspec;
#ifndef __IAMAudioInputMixer_INTERFACE_DEFINED__
#define __IAMAudioInputMixer_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMAudioInputMixer;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMAudioInputMixer : public IUnknown {
  public:
    virtual HRESULT WINAPI put_Enable(WINBOOL fEnable) = 0;
    virtual HRESULT WINAPI get_Enable(WINBOOL *pfEnable) = 0;
    virtual HRESULT WINAPI put_Mono(WINBOOL fMono) = 0;
    virtual HRESULT WINAPI get_Mono(WINBOOL *pfMono) = 0;
    virtual HRESULT WINAPI put_MixLevel(double Level) = 0;
    virtual HRESULT WINAPI get_MixLevel(double *pLevel) = 0;
    virtual HRESULT WINAPI put_Pan(double Pan) = 0;
    virtual HRESULT WINAPI get_Pan(double *pPan) = 0;
    virtual HRESULT WINAPI put_Loudness(WINBOOL fLoudness) = 0;
    virtual HRESULT WINAPI get_Loudness(WINBOOL *pfLoudness) = 0;
    virtual HRESULT WINAPI put_Treble(double Treble) = 0;
    virtual HRESULT WINAPI get_Treble(double *pTreble) = 0;
    virtual HRESULT WINAPI get_TrebleRange(double *pRange) = 0;
    virtual HRESULT WINAPI put_Bass(double Bass) = 0;
    virtual HRESULT WINAPI get_Bass(double *pBass) = 0;
    virtual HRESULT WINAPI get_BassRange(double *pRange) = 0;
  };
#else
  typedef struct IAMAudioInputMixerVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMAudioInputMixer *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMAudioInputMixer *This);
      ULONG (WINAPI *Release)(IAMAudioInputMixer *This);
      HRESULT (WINAPI *put_Enable)(IAMAudioInputMixer *This,WINBOOL fEnable);
      HRESULT (WINAPI *get_Enable)(IAMAudioInputMixer *This,WINBOOL *pfEnable);
      HRESULT (WINAPI *put_Mono)(IAMAudioInputMixer *This,WINBOOL fMono);
      HRESULT (WINAPI *get_Mono)(IAMAudioInputMixer *This,WINBOOL *pfMono);
      HRESULT (WINAPI *put_MixLevel)(IAMAudioInputMixer *This,double Level);
      HRESULT (WINAPI *get_MixLevel)(IAMAudioInputMixer *This,double *pLevel);
      HRESULT (WINAPI *put_Pan)(IAMAudioInputMixer *This,double Pan);
      HRESULT (WINAPI *get_Pan)(IAMAudioInputMixer *This,double *pPan);
      HRESULT (WINAPI *put_Loudness)(IAMAudioInputMixer *This,WINBOOL fLoudness);
      HRESULT (WINAPI *get_Loudness)(IAMAudioInputMixer *This,WINBOOL *pfLoudness);
      HRESULT (WINAPI *put_Treble)(IAMAudioInputMixer *This,double Treble);
      HRESULT (WINAPI *get_Treble)(IAMAudioInputMixer *This,double *pTreble);
      HRESULT (WINAPI *get_TrebleRange)(IAMAudioInputMixer *This,double *pRange);
      HRESULT (WINAPI *put_Bass)(IAMAudioInputMixer *This,double Bass);
      HRESULT (WINAPI *get_Bass)(IAMAudioInputMixer *This,double *pBass);
      HRESULT (WINAPI *get_BassRange)(IAMAudioInputMixer *This,double *pRange);
    END_INTERFACE
  } IAMAudioInputMixerVtbl;
  struct IAMAudioInputMixer {
    CONST_VTBL struct IAMAudioInputMixerVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMAudioInputMixer_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMAudioInputMixer_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMAudioInputMixer_Release(This) (This)->lpVtbl->Release(This)
#define IAMAudioInputMixer_put_Enable(This,fEnable) (This)->lpVtbl->put_Enable(This,fEnable)
#define IAMAudioInputMixer_get_Enable(This,pfEnable) (This)->lpVtbl->get_Enable(This,pfEnable)
#define IAMAudioInputMixer_put_Mono(This,fMono) (This)->lpVtbl->put_Mono(This,fMono)
#define IAMAudioInputMixer_get_Mono(This,pfMono) (This)->lpVtbl->get_Mono(This,pfMono)
#define IAMAudioInputMixer_put_MixLevel(This,Level) (This)->lpVtbl->put_MixLevel(This,Level)
#define IAMAudioInputMixer_get_MixLevel(This,pLevel) (This)->lpVtbl->get_MixLevel(This,pLevel)
#define IAMAudioInputMixer_put_Pan(This,Pan) (This)->lpVtbl->put_Pan(This,Pan)
#define IAMAudioInputMixer_get_Pan(This,pPan) (This)->lpVtbl->get_Pan(This,pPan)
#define IAMAudioInputMixer_put_Loudness(This,fLoudness) (This)->lpVtbl->put_Loudness(This,fLoudness)
#define IAMAudioInputMixer_get_Loudness(This,pfLoudness) (This)->lpVtbl->get_Loudness(This,pfLoudness)
#define IAMAudioInputMixer_put_Treble(This,Treble) (This)->lpVtbl->put_Treble(This,Treble)
#define IAMAudioInputMixer_get_Treble(This,pTreble) (This)->lpVtbl->get_Treble(This,pTreble)
#define IAMAudioInputMixer_get_TrebleRange(This,pRange) (This)->lpVtbl->get_TrebleRange(This,pRange)
#define IAMAudioInputMixer_put_Bass(This,Bass) (This)->lpVtbl->put_Bass(This,Bass)
#define IAMAudioInputMixer_get_Bass(This,pBass) (This)->lpVtbl->get_Bass(This,pBass)
#define IAMAudioInputMixer_get_BassRange(This,pRange) (This)->lpVtbl->get_BassRange(This,pRange)
#endif
#endif
  HRESULT WINAPI IAMAudioInputMixer_put_Enable_Proxy(IAMAudioInputMixer *This,WINBOOL fEnable);
  void __RPC_STUB IAMAudioInputMixer_put_Enable_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAudioInputMixer_get_Enable_Proxy(IAMAudioInputMixer *This,WINBOOL *pfEnable);
  void __RPC_STUB IAMAudioInputMixer_get_Enable_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAudioInputMixer_put_Mono_Proxy(IAMAudioInputMixer *This,WINBOOL fMono);
  void __RPC_STUB IAMAudioInputMixer_put_Mono_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAudioInputMixer_get_Mono_Proxy(IAMAudioInputMixer *This,WINBOOL *pfMono);
  void __RPC_STUB IAMAudioInputMixer_get_Mono_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAudioInputMixer_put_MixLevel_Proxy(IAMAudioInputMixer *This,double Level);
  void __RPC_STUB IAMAudioInputMixer_put_MixLevel_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAudioInputMixer_get_MixLevel_Proxy(IAMAudioInputMixer *This,double *pLevel);
  void __RPC_STUB IAMAudioInputMixer_get_MixLevel_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAudioInputMixer_put_Pan_Proxy(IAMAudioInputMixer *This,double Pan);
  void __RPC_STUB IAMAudioInputMixer_put_Pan_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAudioInputMixer_get_Pan_Proxy(IAMAudioInputMixer *This,double *pPan);
  void __RPC_STUB IAMAudioInputMixer_get_Pan_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAudioInputMixer_put_Loudness_Proxy(IAMAudioInputMixer *This,WINBOOL fLoudness);
  void __RPC_STUB IAMAudioInputMixer_put_Loudness_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAudioInputMixer_get_Loudness_Proxy(IAMAudioInputMixer *This,WINBOOL *pfLoudness);
  void __RPC_STUB IAMAudioInputMixer_get_Loudness_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAudioInputMixer_put_Treble_Proxy(IAMAudioInputMixer *This,double Treble);
  void __RPC_STUB IAMAudioInputMixer_put_Treble_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAudioInputMixer_get_Treble_Proxy(IAMAudioInputMixer *This,double *pTreble);
  void __RPC_STUB IAMAudioInputMixer_get_Treble_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAudioInputMixer_get_TrebleRange_Proxy(IAMAudioInputMixer *This,double *pRange);
  void __RPC_STUB IAMAudioInputMixer_get_TrebleRange_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAudioInputMixer_put_Bass_Proxy(IAMAudioInputMixer *This,double Bass);
  void __RPC_STUB IAMAudioInputMixer_put_Bass_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAudioInputMixer_get_Bass_Proxy(IAMAudioInputMixer *This,double *pBass);
  void __RPC_STUB IAMAudioInputMixer_get_Bass_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAudioInputMixer_get_BassRange_Proxy(IAMAudioInputMixer *This,double *pRange);
  void __RPC_STUB IAMAudioInputMixer_get_BassRange_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IAMBufferNegotiation_INTERFACE_DEFINED__
#define __IAMBufferNegotiation_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMBufferNegotiation;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMBufferNegotiation : public IUnknown {
  public:
    virtual HRESULT WINAPI SuggestAllocatorProperties(const ALLOCATOR_PROPERTIES *pprop) = 0;
    virtual HRESULT WINAPI GetAllocatorProperties(ALLOCATOR_PROPERTIES *pprop) = 0;
  };
#else
  typedef struct IAMBufferNegotiationVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMBufferNegotiation *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMBufferNegotiation *This);
      ULONG (WINAPI *Release)(IAMBufferNegotiation *This);
      HRESULT (WINAPI *SuggestAllocatorProperties)(IAMBufferNegotiation *This,const ALLOCATOR_PROPERTIES *pprop);
      HRESULT (WINAPI *GetAllocatorProperties)(IAMBufferNegotiation *This,ALLOCATOR_PROPERTIES *pprop);
    END_INTERFACE
  } IAMBufferNegotiationVtbl;
  struct IAMBufferNegotiation {
    CONST_VTBL struct IAMBufferNegotiationVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMBufferNegotiation_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMBufferNegotiation_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMBufferNegotiation_Release(This) (This)->lpVtbl->Release(This)
#define IAMBufferNegotiation_SuggestAllocatorProperties(This,pprop) (This)->lpVtbl->SuggestAllocatorProperties(This,pprop)
#define IAMBufferNegotiation_GetAllocatorProperties(This,pprop) (This)->lpVtbl->GetAllocatorProperties(This,pprop)
#endif
#endif
  HRESULT WINAPI IAMBufferNegotiation_SuggestAllocatorProperties_Proxy(IAMBufferNegotiation *This,const ALLOCATOR_PROPERTIES *pprop);
  void __RPC_STUB IAMBufferNegotiation_SuggestAllocatorProperties_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMBufferNegotiation_GetAllocatorProperties_Proxy(IAMBufferNegotiation *This,ALLOCATOR_PROPERTIES *pprop);
  void __RPC_STUB IAMBufferNegotiation_GetAllocatorProperties_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef enum tagAnalogVideoStandard {
    AnalogVideo_None = 0,AnalogVideo_NTSC_M = 0x1,AnalogVideo_NTSC_M_J = 0x2,AnalogVideo_NTSC_433 = 0x4,AnalogVideo_PAL_B = 0x10,
    AnalogVideo_PAL_D = 0x20,AnalogVideo_PAL_G = 0x40,AnalogVideo_PAL_H = 0x80,AnalogVideo_PAL_I = 0x100,AnalogVideo_PAL_M = 0x200,
    AnalogVideo_PAL_N = 0x400,AnalogVideo_PAL_60 = 0x800,AnalogVideo_SECAM_B = 0x1000,AnalogVideo_SECAM_D = 0x2000,AnalogVideo_SECAM_G = 0x4000,
    AnalogVideo_SECAM_H = 0x8000,AnalogVideo_SECAM_K = 0x10000,AnalogVideo_SECAM_K1 = 0x20000,AnalogVideo_SECAM_L = 0x40000,AnalogVideo_SECAM_L1 = 0x80000,
    AnalogVideo_PAL_N_COMBO = 0x100000,AnalogVideoMask_MCE_NTSC = AnalogVideo_NTSC_M | AnalogVideo_NTSC_M_J | AnalogVideo_NTSC_433 | AnalogVideo_PAL_M | AnalogVideo_PAL_N | AnalogVideo_PAL_60 | AnalogVideo_PAL_N_COMBO,AnalogVideoMask_MCE_PAL = AnalogVideo_PAL_B | AnalogVideo_PAL_D | AnalogVideo_PAL_G | AnalogVideo_PAL_H | AnalogVideo_PAL_I,AnalogVideoMask_MCE_SECAM = AnalogVideo_SECAM_B | AnalogVideo_SECAM_D | AnalogVideo_SECAM_G | AnalogVideo_SECAM_H | AnalogVideo_SECAM_K | AnalogVideo_SECAM_K1 | AnalogVideo_SECAM_L | AnalogVideo_SECAM_L1
  } AnalogVideoStandard;

  typedef enum tagTunerInputType {
    TunerInputCable = 0,TunerInputAntenna = TunerInputCable + 1
  } TunerInputType;

#define AnalogVideo_NTSC_Mask 0x00000007
#define AnalogVideo_PAL_Mask 0x00100FF0
#define AnalogVideo_SECAM_Mask 0x000FF000

  typedef enum __MIDL___MIDL_itf_strmif_0173_0001 {
    VideoCopyProtectionMacrovisionBasic = 0,VideoCopyProtectionMacrovisionCBI = VideoCopyProtectionMacrovisionBasic + 1
  } VideoCopyProtectionType;

  typedef enum tagPhysicalConnectorType {
    PhysConn_Video_Tuner = 1,
    PhysConn_Video_Composite,PhysConn_Video_SVideo,PhysConn_Video_RGB,
    PhysConn_Video_YRYBY,PhysConn_Video_SerialDigital,PhysConn_Video_ParallelDigital,
    PhysConn_Video_SCSI,PhysConn_Video_AUX,PhysConn_Video_1394,PhysConn_Video_USB,
    PhysConn_Video_VideoDecoder,PhysConn_Video_VideoEncoder,PhysConn_Video_SCART,PhysConn_Video_Black,
    PhysConn_Audio_Tuner = 0x1000,PhysConn_Audio_Line = 0x1001,PhysConn_Audio_Mic = 0x1002,
    PhysConn_Audio_AESDigital = 0x1003,PhysConn_Audio_SPDIFDigital = 0x1004,
    PhysConn_Audio_SCSI = 0x1005,PhysConn_Audio_AUX = 0x1006,PhysConn_Audio_1394 = 0x1007,
    PhysConn_Audio_USB = 0x1008,PhysConn_Audio_AudioDecoder = 0x1009
  } PhysicalConnectorType;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0173_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0173_v0_0_s_ifspec;
#ifndef __IAMAnalogVideoDecoder_INTERFACE_DEFINED__
#define __IAMAnalogVideoDecoder_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMAnalogVideoDecoder;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMAnalogVideoDecoder : public IUnknown {
  public:
    virtual HRESULT WINAPI get_AvailableTVFormats(long *lAnalogVideoStandard) = 0;
    virtual HRESULT WINAPI put_TVFormat(long lAnalogVideoStandard) = 0;
    virtual HRESULT WINAPI get_TVFormat(long *plAnalogVideoStandard) = 0;
    virtual HRESULT WINAPI get_HorizontalLocked(long *plLocked) = 0;
    virtual HRESULT WINAPI put_VCRHorizontalLocking(long lVCRHorizontalLocking) = 0;
    virtual HRESULT WINAPI get_VCRHorizontalLocking(long *plVCRHorizontalLocking) = 0;
    virtual HRESULT WINAPI get_NumberOfLines(long *plNumberOfLines) = 0;
    virtual HRESULT WINAPI put_OutputEnable(long lOutputEnable) = 0;
    virtual HRESULT WINAPI get_OutputEnable(long *plOutputEnable) = 0;
  };
#else
  typedef struct IAMAnalogVideoDecoderVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMAnalogVideoDecoder *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMAnalogVideoDecoder *This);
      ULONG (WINAPI *Release)(IAMAnalogVideoDecoder *This);
      HRESULT (WINAPI *get_AvailableTVFormats)(IAMAnalogVideoDecoder *This,long *lAnalogVideoStandard);
      HRESULT (WINAPI *put_TVFormat)(IAMAnalogVideoDecoder *This,long lAnalogVideoStandard);
      HRESULT (WINAPI *get_TVFormat)(IAMAnalogVideoDecoder *This,long *plAnalogVideoStandard);
      HRESULT (WINAPI *get_HorizontalLocked)(IAMAnalogVideoDecoder *This,long *plLocked);
      HRESULT (WINAPI *put_VCRHorizontalLocking)(IAMAnalogVideoDecoder *This,long lVCRHorizontalLocking);
      HRESULT (WINAPI *get_VCRHorizontalLocking)(IAMAnalogVideoDecoder *This,long *plVCRHorizontalLocking);
      HRESULT (WINAPI *get_NumberOfLines)(IAMAnalogVideoDecoder *This,long *plNumberOfLines);
      HRESULT (WINAPI *put_OutputEnable)(IAMAnalogVideoDecoder *This,long lOutputEnable);
      HRESULT (WINAPI *get_OutputEnable)(IAMAnalogVideoDecoder *This,long *plOutputEnable);
    END_INTERFACE
  } IAMAnalogVideoDecoderVtbl;
  struct IAMAnalogVideoDecoder {
    CONST_VTBL struct IAMAnalogVideoDecoderVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMAnalogVideoDecoder_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMAnalogVideoDecoder_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMAnalogVideoDecoder_Release(This) (This)->lpVtbl->Release(This)
#define IAMAnalogVideoDecoder_get_AvailableTVFormats(This,lAnalogVideoStandard) (This)->lpVtbl->get_AvailableTVFormats(This,lAnalogVideoStandard)
#define IAMAnalogVideoDecoder_put_TVFormat(This,lAnalogVideoStandard) (This)->lpVtbl->put_TVFormat(This,lAnalogVideoStandard)
#define IAMAnalogVideoDecoder_get_TVFormat(This,plAnalogVideoStandard) (This)->lpVtbl->get_TVFormat(This,plAnalogVideoStandard)
#define IAMAnalogVideoDecoder_get_HorizontalLocked(This,plLocked) (This)->lpVtbl->get_HorizontalLocked(This,plLocked)
#define IAMAnalogVideoDecoder_put_VCRHorizontalLocking(This,lVCRHorizontalLocking) (This)->lpVtbl->put_VCRHorizontalLocking(This,lVCRHorizontalLocking)
#define IAMAnalogVideoDecoder_get_VCRHorizontalLocking(This,plVCRHorizontalLocking) (This)->lpVtbl->get_VCRHorizontalLocking(This,plVCRHorizontalLocking)
#define IAMAnalogVideoDecoder_get_NumberOfLines(This,plNumberOfLines) (This)->lpVtbl->get_NumberOfLines(This,plNumberOfLines)
#define IAMAnalogVideoDecoder_put_OutputEnable(This,lOutputEnable) (This)->lpVtbl->put_OutputEnable(This,lOutputEnable)
#define IAMAnalogVideoDecoder_get_OutputEnable(This,plOutputEnable) (This)->lpVtbl->get_OutputEnable(This,plOutputEnable)
#endif
#endif
  HRESULT WINAPI IAMAnalogVideoDecoder_get_AvailableTVFormats_Proxy(IAMAnalogVideoDecoder *This,long *lAnalogVideoStandard);
  void __RPC_STUB IAMAnalogVideoDecoder_get_AvailableTVFormats_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAnalogVideoDecoder_put_TVFormat_Proxy(IAMAnalogVideoDecoder *This,long lAnalogVideoStandard);
  void __RPC_STUB IAMAnalogVideoDecoder_put_TVFormat_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAnalogVideoDecoder_get_TVFormat_Proxy(IAMAnalogVideoDecoder *This,long *plAnalogVideoStandard);
  void __RPC_STUB IAMAnalogVideoDecoder_get_TVFormat_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAnalogVideoDecoder_get_HorizontalLocked_Proxy(IAMAnalogVideoDecoder *This,long *plLocked);
  void __RPC_STUB IAMAnalogVideoDecoder_get_HorizontalLocked_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAnalogVideoDecoder_put_VCRHorizontalLocking_Proxy(IAMAnalogVideoDecoder *This,long lVCRHorizontalLocking);
  void __RPC_STUB IAMAnalogVideoDecoder_put_VCRHorizontalLocking_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAnalogVideoDecoder_get_VCRHorizontalLocking_Proxy(IAMAnalogVideoDecoder *This,long *plVCRHorizontalLocking);
  void __RPC_STUB IAMAnalogVideoDecoder_get_VCRHorizontalLocking_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAnalogVideoDecoder_get_NumberOfLines_Proxy(IAMAnalogVideoDecoder *This,long *plNumberOfLines);
  void __RPC_STUB IAMAnalogVideoDecoder_get_NumberOfLines_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAnalogVideoDecoder_put_OutputEnable_Proxy(IAMAnalogVideoDecoder *This,long lOutputEnable);
  void __RPC_STUB IAMAnalogVideoDecoder_put_OutputEnable_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAnalogVideoDecoder_get_OutputEnable_Proxy(IAMAnalogVideoDecoder *This,long *plOutputEnable);
  void __RPC_STUB IAMAnalogVideoDecoder_get_OutputEnable_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef enum tagVideoProcAmpProperty {
    VideoProcAmp_Brightness = 0,
    VideoProcAmp_Contrast,VideoProcAmp_Hue,VideoProcAmp_Saturation,VideoProcAmp_Sharpness,
    VideoProcAmp_Gamma,VideoProcAmp_ColorEnable,VideoProcAmp_WhiteBalance,
    VideoProcAmp_BacklightCompensation,VideoProcAmp_Gain
  } VideoProcAmpProperty;

  typedef enum tagVideoProcAmpFlags {
    VideoProcAmp_Flags_Auto = 0x1,VideoProcAmp_Flags_Manual = 0x2
  } VideoProcAmpFlags;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0174_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0174_v0_0_s_ifspec;
#ifndef __IAMVideoProcAmp_INTERFACE_DEFINED__
#define __IAMVideoProcAmp_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMVideoProcAmp;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMVideoProcAmp : public IUnknown {
  public:
    virtual HRESULT WINAPI GetRange(long Property,long *pMin,long *pMax,long *pSteppingDelta,long *pDefault,long *pCapsFlags) = 0;
    virtual HRESULT WINAPI Set(long Property,long lValue,long Flags) = 0;
    virtual HRESULT WINAPI Get(long Property,long *lValue,long *Flags) = 0;
  };
#else
  typedef struct IAMVideoProcAmpVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMVideoProcAmp *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMVideoProcAmp *This);
      ULONG (WINAPI *Release)(IAMVideoProcAmp *This);
      HRESULT (WINAPI *GetRange)(IAMVideoProcAmp *This,long Property,long *pMin,long *pMax,long *pSteppingDelta,long *pDefault,long *pCapsFlags);
      HRESULT (WINAPI *Set)(IAMVideoProcAmp *This,long Property,long lValue,long Flags);
      HRESULT (WINAPI *Get)(IAMVideoProcAmp *This,long Property,long *lValue,long *Flags);
    END_INTERFACE
  } IAMVideoProcAmpVtbl;
  struct IAMVideoProcAmp {
    CONST_VTBL struct IAMVideoProcAmpVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMVideoProcAmp_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMVideoProcAmp_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMVideoProcAmp_Release(This) (This)->lpVtbl->Release(This)
#define IAMVideoProcAmp_GetRange(This,Property,pMin,pMax,pSteppingDelta,pDefault,pCapsFlags) (This)->lpVtbl->GetRange(This,Property,pMin,pMax,pSteppingDelta,pDefault,pCapsFlags)
#define IAMVideoProcAmp_Set(This,Property,lValue,Flags) (This)->lpVtbl->Set(This,Property,lValue,Flags)
#define IAMVideoProcAmp_Get(This,Property,lValue,Flags) (This)->lpVtbl->Get(This,Property,lValue,Flags)
#endif
#endif
  HRESULT WINAPI IAMVideoProcAmp_GetRange_Proxy(IAMVideoProcAmp *This,long Property,long *pMin,long *pMax,long *pSteppingDelta,long *pDefault,long *pCapsFlags);
  void __RPC_STUB IAMVideoProcAmp_GetRange_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVideoProcAmp_Set_Proxy(IAMVideoProcAmp *This,long Property,long lValue,long Flags);
  void __RPC_STUB IAMVideoProcAmp_Set_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVideoProcAmp_Get_Proxy(IAMVideoProcAmp *This,long Property,long *lValue,long *Flags);
  void __RPC_STUB IAMVideoProcAmp_Get_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef enum tagCameraControlProperty {
    CameraControl_Pan = 0,
    CameraControl_Tilt,CameraControl_Roll,CameraControl_Zoom,CameraControl_Exposure,
    CameraControl_Iris,CameraControl_Focus
  } CameraControlProperty;

  typedef enum tagCameraControlFlags {
    CameraControl_Flags_Auto = 0x1,CameraControl_Flags_Manual = 0x2
  } CameraControlFlags;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0175_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0175_v0_0_s_ifspec;
#ifndef __IAMCameraControl_INTERFACE_DEFINED__
#define __IAMCameraControl_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMCameraControl;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMCameraControl : public IUnknown {
  public:
    virtual HRESULT WINAPI GetRange(long Property,long *pMin,long *pMax,long *pSteppingDelta,long *pDefault,long *pCapsFlags) = 0;
    virtual HRESULT WINAPI Set(long Property,long lValue,long Flags) = 0;
    virtual HRESULT WINAPI Get(long Property,long *lValue,long *Flags) = 0;
  };
#else
  typedef struct IAMCameraControlVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMCameraControl *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMCameraControl *This);
      ULONG (WINAPI *Release)(IAMCameraControl *This);
      HRESULT (WINAPI *GetRange)(IAMCameraControl *This,long Property,long *pMin,long *pMax,long *pSteppingDelta,long *pDefault,long *pCapsFlags);
      HRESULT (WINAPI *Set)(IAMCameraControl *This,long Property,long lValue,long Flags);
      HRESULT (WINAPI *Get)(IAMCameraControl *This,long Property,long *lValue,long *Flags);
    END_INTERFACE
  } IAMCameraControlVtbl;
  struct IAMCameraControl {
    CONST_VTBL struct IAMCameraControlVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMCameraControl_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMCameraControl_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMCameraControl_Release(This) (This)->lpVtbl->Release(This)
#define IAMCameraControl_GetRange(This,Property,pMin,pMax,pSteppingDelta,pDefault,pCapsFlags) (This)->lpVtbl->GetRange(This,Property,pMin,pMax,pSteppingDelta,pDefault,pCapsFlags)
#define IAMCameraControl_Set(This,Property,lValue,Flags) (This)->lpVtbl->Set(This,Property,lValue,Flags)
#define IAMCameraControl_Get(This,Property,lValue,Flags) (This)->lpVtbl->Get(This,Property,lValue,Flags)
#endif
#endif
  HRESULT WINAPI IAMCameraControl_GetRange_Proxy(IAMCameraControl *This,long Property,long *pMin,long *pMax,long *pSteppingDelta,long *pDefault,long *pCapsFlags);
  void __RPC_STUB IAMCameraControl_GetRange_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMCameraControl_Set_Proxy(IAMCameraControl *This,long Property,long lValue,long Flags);
  void __RPC_STUB IAMCameraControl_Set_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMCameraControl_Get_Proxy(IAMCameraControl *This,long Property,long *lValue,long *Flags);
  void __RPC_STUB IAMCameraControl_Get_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef enum tagVideoControlFlags {
    VideoControlFlag_FlipHorizontal = 0x1,VideoControlFlag_FlipVertical = 0x2,VideoControlFlag_ExternalTriggerEnable = 0x4,VideoControlFlag_Trigger = 0x8
  } VideoControlFlags;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0176_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0176_v0_0_s_ifspec;
#ifndef __IAMVideoControl_INTERFACE_DEFINED__
#define __IAMVideoControl_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMVideoControl;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMVideoControl : public IUnknown {
  public:
    virtual HRESULT WINAPI GetCaps(IPin *pPin,long *pCapsFlags) = 0;
    virtual HRESULT WINAPI SetMode(IPin *pPin,long Mode) = 0;
    virtual HRESULT WINAPI GetMode(IPin *pPin,long *Mode) = 0;
    virtual HRESULT WINAPI GetCurrentActualFrameRate(IPin *pPin,LONGLONG *ActualFrameRate) = 0;
    virtual HRESULT WINAPI GetMaxAvailableFrameRate(IPin *pPin,long iIndex,SIZE Dimensions,LONGLONG *MaxAvailableFrameRate) = 0;
    virtual HRESULT WINAPI GetFrameRateList(IPin *pPin,long iIndex,SIZE Dimensions,long *ListSize,LONGLONG **FrameRates) = 0;
  };
#else
  typedef struct IAMVideoControlVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMVideoControl *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMVideoControl *This);
      ULONG (WINAPI *Release)(IAMVideoControl *This);
      HRESULT (WINAPI *GetCaps)(IAMVideoControl *This,IPin *pPin,long *pCapsFlags);
      HRESULT (WINAPI *SetMode)(IAMVideoControl *This,IPin *pPin,long Mode);
      HRESULT (WINAPI *GetMode)(IAMVideoControl *This,IPin *pPin,long *Mode);
      HRESULT (WINAPI *GetCurrentActualFrameRate)(IAMVideoControl *This,IPin *pPin,LONGLONG *ActualFrameRate);
      HRESULT (WINAPI *GetMaxAvailableFrameRate)(IAMVideoControl *This,IPin *pPin,long iIndex,SIZE Dimensions,LONGLONG *MaxAvailableFrameRate);
      HRESULT (WINAPI *GetFrameRateList)(IAMVideoControl *This,IPin *pPin,long iIndex,SIZE Dimensions,long *ListSize,LONGLONG **FrameRates);
    END_INTERFACE
  } IAMVideoControlVtbl;
  struct IAMVideoControl {
    CONST_VTBL struct IAMVideoControlVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMVideoControl_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMVideoControl_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMVideoControl_Release(This) (This)->lpVtbl->Release(This)
#define IAMVideoControl_GetCaps(This,pPin,pCapsFlags) (This)->lpVtbl->GetCaps(This,pPin,pCapsFlags)
#define IAMVideoControl_SetMode(This,pPin,Mode) (This)->lpVtbl->SetMode(This,pPin,Mode)
#define IAMVideoControl_GetMode(This,pPin,Mode) (This)->lpVtbl->GetMode(This,pPin,Mode)
#define IAMVideoControl_GetCurrentActualFrameRate(This,pPin,ActualFrameRate) (This)->lpVtbl->GetCurrentActualFrameRate(This,pPin,ActualFrameRate)
#define IAMVideoControl_GetMaxAvailableFrameRate(This,pPin,iIndex,Dimensions,MaxAvailableFrameRate) (This)->lpVtbl->GetMaxAvailableFrameRate(This,pPin,iIndex,Dimensions,MaxAvailableFrameRate)
#define IAMVideoControl_GetFrameRateList(This,pPin,iIndex,Dimensions,ListSize,FrameRates) (This)->lpVtbl->GetFrameRateList(This,pPin,iIndex,Dimensions,ListSize,FrameRates)
#endif
#endif
  HRESULT WINAPI IAMVideoControl_GetCaps_Proxy(IAMVideoControl *This,IPin *pPin,long *pCapsFlags);
  void __RPC_STUB IAMVideoControl_GetCaps_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVideoControl_SetMode_Proxy(IAMVideoControl *This,IPin *pPin,long Mode);
  void __RPC_STUB IAMVideoControl_SetMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVideoControl_GetMode_Proxy(IAMVideoControl *This,IPin *pPin,long *Mode);
  void __RPC_STUB IAMVideoControl_GetMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVideoControl_GetCurrentActualFrameRate_Proxy(IAMVideoControl *This,IPin *pPin,LONGLONG *ActualFrameRate);
  void __RPC_STUB IAMVideoControl_GetCurrentActualFrameRate_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVideoControl_GetMaxAvailableFrameRate_Proxy(IAMVideoControl *This,IPin *pPin,long iIndex,SIZE Dimensions,LONGLONG *MaxAvailableFrameRate);
  void __RPC_STUB IAMVideoControl_GetMaxAvailableFrameRate_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVideoControl_GetFrameRateList_Proxy(IAMVideoControl *This,IPin *pPin,long iIndex,SIZE Dimensions,long *ListSize,LONGLONG **FrameRates);
  void __RPC_STUB IAMVideoControl_GetFrameRateList_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IAMCrossbar_INTERFACE_DEFINED__
#define __IAMCrossbar_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMCrossbar;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMCrossbar : public IUnknown {
  public:
    virtual HRESULT WINAPI get_PinCounts(long *OutputPinCount,long *InputPinCount) = 0;
    virtual HRESULT WINAPI CanRoute(long OutputPinIndex,long InputPinIndex) = 0;
    virtual HRESULT WINAPI Route(long OutputPinIndex,long InputPinIndex) = 0;
    virtual HRESULT WINAPI get_IsRoutedTo(long OutputPinIndex,long *InputPinIndex) = 0;
    virtual HRESULT WINAPI get_CrossbarPinInfo(WINBOOL IsInputPin,long PinIndex,long *PinIndexRelated,long *PhysicalType) = 0;
  };
#else
  typedef struct IAMCrossbarVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMCrossbar *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMCrossbar *This);
      ULONG (WINAPI *Release)(IAMCrossbar *This);
      HRESULT (WINAPI *get_PinCounts)(IAMCrossbar *This,long *OutputPinCount,long *InputPinCount);
      HRESULT (WINAPI *CanRoute)(IAMCrossbar *This,long OutputPinIndex,long InputPinIndex);
      HRESULT (WINAPI *Route)(IAMCrossbar *This,long OutputPinIndex,long InputPinIndex);
      HRESULT (WINAPI *get_IsRoutedTo)(IAMCrossbar *This,long OutputPinIndex,long *InputPinIndex);
      HRESULT (WINAPI *get_CrossbarPinInfo)(IAMCrossbar *This,WINBOOL IsInputPin,long PinIndex,long *PinIndexRelated,long *PhysicalType);
    END_INTERFACE
  } IAMCrossbarVtbl;
  struct IAMCrossbar {
    CONST_VTBL struct IAMCrossbarVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMCrossbar_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMCrossbar_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMCrossbar_Release(This) (This)->lpVtbl->Release(This)
#define IAMCrossbar_get_PinCounts(This,OutputPinCount,InputPinCount) (This)->lpVtbl->get_PinCounts(This,OutputPinCount,InputPinCount)
#define IAMCrossbar_CanRoute(This,OutputPinIndex,InputPinIndex) (This)->lpVtbl->CanRoute(This,OutputPinIndex,InputPinIndex)
#define IAMCrossbar_Route(This,OutputPinIndex,InputPinIndex) (This)->lpVtbl->Route(This,OutputPinIndex,InputPinIndex)
#define IAMCrossbar_get_IsRoutedTo(This,OutputPinIndex,InputPinIndex) (This)->lpVtbl->get_IsRoutedTo(This,OutputPinIndex,InputPinIndex)
#define IAMCrossbar_get_CrossbarPinInfo(This,IsInputPin,PinIndex,PinIndexRelated,PhysicalType) (This)->lpVtbl->get_CrossbarPinInfo(This,IsInputPin,PinIndex,PinIndexRelated,PhysicalType)
#endif
#endif
  HRESULT WINAPI IAMCrossbar_get_PinCounts_Proxy(IAMCrossbar *This,long *OutputPinCount,long *InputPinCount);
  void __RPC_STUB IAMCrossbar_get_PinCounts_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMCrossbar_CanRoute_Proxy(IAMCrossbar *This,long OutputPinIndex,long InputPinIndex);
  void __RPC_STUB IAMCrossbar_CanRoute_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMCrossbar_Route_Proxy(IAMCrossbar *This,long OutputPinIndex,long InputPinIndex);
  void __RPC_STUB IAMCrossbar_Route_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMCrossbar_get_IsRoutedTo_Proxy(IAMCrossbar *This,long OutputPinIndex,long *InputPinIndex);
  void __RPC_STUB IAMCrossbar_get_IsRoutedTo_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMCrossbar_get_CrossbarPinInfo_Proxy(IAMCrossbar *This,WINBOOL IsInputPin,long PinIndex,long *PinIndexRelated,long *PhysicalType);
  void __RPC_STUB IAMCrossbar_get_CrossbarPinInfo_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef enum tagAMTunerSubChannel {
    AMTUNER_SUBCHAN_NO_TUNE = -2,AMTUNER_SUBCHAN_DEFAULT = -1
  } AMTunerSubChannel;

  typedef enum tagAMTunerSignalStrength {
    AMTUNER_HASNOSIGNALSTRENGTH = -1,AMTUNER_NOSIGNAL = 0,AMTUNER_SIGNALPRESENT = 1
  } AMTunerSignalStrength;

  typedef enum tagAMTunerModeType {
    AMTUNER_MODE_DEFAULT = 0,AMTUNER_MODE_TV = 0x1,AMTUNER_MODE_FM_RADIO = 0x2,AMTUNER_MODE_AM_RADIO = 0x4,AMTUNER_MODE_DSS = 0x8
  } AMTunerModeType;

  typedef enum tagAMTunerEventType {
    AMTUNER_EVENT_CHANGED = 0x1
  } AMTunerEventType;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0178_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0178_v0_0_s_ifspec;
#ifndef __IAMTuner_INTERFACE_DEFINED__
#define __IAMTuner_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMTuner;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMTuner : public IUnknown {
  public:
    virtual HRESULT WINAPI put_Channel(long lChannel,long lVideoSubChannel,long lAudioSubChannel) = 0;
    virtual HRESULT WINAPI get_Channel(long *plChannel,long *plVideoSubChannel,long *plAudioSubChannel) = 0;
    virtual HRESULT WINAPI ChannelMinMax(long *lChannelMin,long *lChannelMax) = 0;
    virtual HRESULT WINAPI put_CountryCode(long lCountryCode) = 0;
    virtual HRESULT WINAPI get_CountryCode(long *plCountryCode) = 0;
    virtual HRESULT WINAPI put_TuningSpace(long lTuningSpace) = 0;
    virtual HRESULT WINAPI get_TuningSpace(long *plTuningSpace) = 0;
    virtual HRESULT WINAPI Logon(HANDLE hCurrentUser) = 0;
    virtual HRESULT WINAPI Logout(void) = 0;
    virtual HRESULT WINAPI SignalPresent(long *plSignalStrength) = 0;
    virtual HRESULT WINAPI put_Mode(AMTunerModeType lMode) = 0;
    virtual HRESULT WINAPI get_Mode(AMTunerModeType *plMode) = 0;
    virtual HRESULT WINAPI GetAvailableModes(long *plModes) = 0;
    virtual HRESULT WINAPI RegisterNotificationCallBack(IAMTunerNotification *pNotify,long lEvents) = 0;
    virtual HRESULT WINAPI UnRegisterNotificationCallBack(IAMTunerNotification *pNotify) = 0;
  };
#else
  typedef struct IAMTunerVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMTuner *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMTuner *This);
      ULONG (WINAPI *Release)(IAMTuner *This);
      HRESULT (WINAPI *put_Channel)(IAMTuner *This,long lChannel,long lVideoSubChannel,long lAudioSubChannel);
      HRESULT (WINAPI *get_Channel)(IAMTuner *This,long *plChannel,long *plVideoSubChannel,long *plAudioSubChannel);
      HRESULT (WINAPI *ChannelMinMax)(IAMTuner *This,long *lChannelMin,long *lChannelMax);
      HRESULT (WINAPI *put_CountryCode)(IAMTuner *This,long lCountryCode);
      HRESULT (WINAPI *get_CountryCode)(IAMTuner *This,long *plCountryCode);
      HRESULT (WINAPI *put_TuningSpace)(IAMTuner *This,long lTuningSpace);
      HRESULT (WINAPI *get_TuningSpace)(IAMTuner *This,long *plTuningSpace);
      HRESULT (WINAPI *Logon)(IAMTuner *This,HANDLE hCurrentUser);
      HRESULT (WINAPI *Logout)(IAMTuner *This);
      HRESULT (WINAPI *SignalPresent)(IAMTuner *This,long *plSignalStrength);
      HRESULT (WINAPI *put_Mode)(IAMTuner *This,AMTunerModeType lMode);
      HRESULT (WINAPI *get_Mode)(IAMTuner *This,AMTunerModeType *plMode);
      HRESULT (WINAPI *GetAvailableModes)(IAMTuner *This,long *plModes);
      HRESULT (WINAPI *RegisterNotificationCallBack)(IAMTuner *This,IAMTunerNotification *pNotify,long lEvents);
      HRESULT (WINAPI *UnRegisterNotificationCallBack)(IAMTuner *This,IAMTunerNotification *pNotify);
    END_INTERFACE
  } IAMTunerVtbl;
  struct IAMTuner {
    CONST_VTBL struct IAMTunerVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMTuner_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMTuner_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMTuner_Release(This) (This)->lpVtbl->Release(This)
#define IAMTuner_put_Channel(This,lChannel,lVideoSubChannel,lAudioSubChannel) (This)->lpVtbl->put_Channel(This,lChannel,lVideoSubChannel,lAudioSubChannel)
#define IAMTuner_get_Channel(This,plChannel,plVideoSubChannel,plAudioSubChannel) (This)->lpVtbl->get_Channel(This,plChannel,plVideoSubChannel,plAudioSubChannel)
#define IAMTuner_ChannelMinMax(This,lChannelMin,lChannelMax) (This)->lpVtbl->ChannelMinMax(This,lChannelMin,lChannelMax)
#define IAMTuner_put_CountryCode(This,lCountryCode) (This)->lpVtbl->put_CountryCode(This,lCountryCode)
#define IAMTuner_get_CountryCode(This,plCountryCode) (This)->lpVtbl->get_CountryCode(This,plCountryCode)
#define IAMTuner_put_TuningSpace(This,lTuningSpace) (This)->lpVtbl->put_TuningSpace(This,lTuningSpace)
#define IAMTuner_get_TuningSpace(This,plTuningSpace) (This)->lpVtbl->get_TuningSpace(This,plTuningSpace)
#define IAMTuner_Logon(This,hCurrentUser) (This)->lpVtbl->Logon(This,hCurrentUser)
#define IAMTuner_Logout(This) (This)->lpVtbl->Logout(This)
#define IAMTuner_SignalPresent(This,plSignalStrength) (This)->lpVtbl->SignalPresent(This,plSignalStrength)
#define IAMTuner_put_Mode(This,lMode) (This)->lpVtbl->put_Mode(This,lMode)
#define IAMTuner_get_Mode(This,plMode) (This)->lpVtbl->get_Mode(This,plMode)
#define IAMTuner_GetAvailableModes(This,plModes) (This)->lpVtbl->GetAvailableModes(This,plModes)
#define IAMTuner_RegisterNotificationCallBack(This,pNotify,lEvents) (This)->lpVtbl->RegisterNotificationCallBack(This,pNotify,lEvents)
#define IAMTuner_UnRegisterNotificationCallBack(This,pNotify) (This)->lpVtbl->UnRegisterNotificationCallBack(This,pNotify)
#endif
#endif
  HRESULT WINAPI IAMTuner_put_Channel_Proxy(IAMTuner *This,long lChannel,long lVideoSubChannel,long lAudioSubChannel);
  void __RPC_STUB IAMTuner_put_Channel_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTuner_get_Channel_Proxy(IAMTuner *This,long *plChannel,long *plVideoSubChannel,long *plAudioSubChannel);
  void __RPC_STUB IAMTuner_get_Channel_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTuner_ChannelMinMax_Proxy(IAMTuner *This,long *lChannelMin,long *lChannelMax);
  void __RPC_STUB IAMTuner_ChannelMinMax_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTuner_put_CountryCode_Proxy(IAMTuner *This,long lCountryCode);
  void __RPC_STUB IAMTuner_put_CountryCode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTuner_get_CountryCode_Proxy(IAMTuner *This,long *plCountryCode);
  void __RPC_STUB IAMTuner_get_CountryCode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTuner_put_TuningSpace_Proxy(IAMTuner *This,long lTuningSpace);
  void __RPC_STUB IAMTuner_put_TuningSpace_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTuner_get_TuningSpace_Proxy(IAMTuner *This,long *plTuningSpace);
  void __RPC_STUB IAMTuner_get_TuningSpace_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTuner_Logon_Proxy(IAMTuner *This,HANDLE hCurrentUser);
  void __RPC_STUB IAMTuner_Logon_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTuner_Logout_Proxy(IAMTuner *This);
  void __RPC_STUB IAMTuner_Logout_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTuner_SignalPresent_Proxy(IAMTuner *This,long *plSignalStrength);
  void __RPC_STUB IAMTuner_SignalPresent_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTuner_put_Mode_Proxy(IAMTuner *This,AMTunerModeType lMode);
  void __RPC_STUB IAMTuner_put_Mode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTuner_get_Mode_Proxy(IAMTuner *This,AMTunerModeType *plMode);
  void __RPC_STUB IAMTuner_get_Mode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTuner_GetAvailableModes_Proxy(IAMTuner *This,long *plModes);
  void __RPC_STUB IAMTuner_GetAvailableModes_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTuner_RegisterNotificationCallBack_Proxy(IAMTuner *This,IAMTunerNotification *pNotify,long lEvents);
  void __RPC_STUB IAMTuner_RegisterNotificationCallBack_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTuner_UnRegisterNotificationCallBack_Proxy(IAMTuner *This,IAMTunerNotification *pNotify);
  void __RPC_STUB IAMTuner_UnRegisterNotificationCallBack_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IAMTunerNotification_INTERFACE_DEFINED__
#define __IAMTunerNotification_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMTunerNotification;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMTunerNotification : public IUnknown {
  public:
    virtual HRESULT WINAPI OnEvent(AMTunerEventType Event) = 0;
  };
#else
  typedef struct IAMTunerNotificationVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMTunerNotification *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMTunerNotification *This);
      ULONG (WINAPI *Release)(IAMTunerNotification *This);
      HRESULT (WINAPI *OnEvent)(IAMTunerNotification *This,AMTunerEventType Event);
    END_INTERFACE
  } IAMTunerNotificationVtbl;
  struct IAMTunerNotification {
    CONST_VTBL struct IAMTunerNotificationVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMTunerNotification_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMTunerNotification_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMTunerNotification_Release(This) (This)->lpVtbl->Release(This)
#define IAMTunerNotification_OnEvent(This,Event) (This)->lpVtbl->OnEvent(This,Event)
#endif
#endif
  HRESULT WINAPI IAMTunerNotification_OnEvent_Proxy(IAMTunerNotification *This,AMTunerEventType Event);
  void __RPC_STUB IAMTunerNotification_OnEvent_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IAMTVTuner_INTERFACE_DEFINED__
#define __IAMTVTuner_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMTVTuner;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMTVTuner : public IAMTuner {
  public:
    virtual HRESULT WINAPI get_AvailableTVFormats(long *lAnalogVideoStandard) = 0;
    virtual HRESULT WINAPI get_TVFormat(long *plAnalogVideoStandard) = 0;
    virtual HRESULT WINAPI AutoTune(long lChannel,long *plFoundSignal) = 0;
    virtual HRESULT WINAPI StoreAutoTune(void) = 0;
    virtual HRESULT WINAPI get_NumInputConnections(long *plNumInputConnections) = 0;
    virtual HRESULT WINAPI put_InputType(long lIndex,TunerInputType InputType) = 0;
    virtual HRESULT WINAPI get_InputType(long lIndex,TunerInputType *pInputType) = 0;
    virtual HRESULT WINAPI put_ConnectInput(long lIndex) = 0;
    virtual HRESULT WINAPI get_ConnectInput(long *plIndex) = 0;
    virtual HRESULT WINAPI get_VideoFrequency(long *lFreq) = 0;
    virtual HRESULT WINAPI get_AudioFrequency(long *lFreq) = 0;
  };
#else
  typedef struct IAMTVTunerVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMTVTuner *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMTVTuner *This);
      ULONG (WINAPI *Release)(IAMTVTuner *This);
      HRESULT (WINAPI *put_Channel)(IAMTVTuner *This,long lChannel,long lVideoSubChannel,long lAudioSubChannel);
      HRESULT (WINAPI *get_Channel)(IAMTVTuner *This,long *plChannel,long *plVideoSubChannel,long *plAudioSubChannel);
      HRESULT (WINAPI *ChannelMinMax)(IAMTVTuner *This,long *lChannelMin,long *lChannelMax);
      HRESULT (WINAPI *put_CountryCode)(IAMTVTuner *This,long lCountryCode);
      HRESULT (WINAPI *get_CountryCode)(IAMTVTuner *This,long *plCountryCode);
      HRESULT (WINAPI *put_TuningSpace)(IAMTVTuner *This,long lTuningSpace);
      HRESULT (WINAPI *get_TuningSpace)(IAMTVTuner *This,long *plTuningSpace);
      HRESULT (WINAPI *Logon)(IAMTVTuner *This,HANDLE hCurrentUser);
      HRESULT (WINAPI *Logout)(IAMTVTuner *This);
      HRESULT (WINAPI *SignalPresent)(IAMTVTuner *This,long *plSignalStrength);
      HRESULT (WINAPI *put_Mode)(IAMTVTuner *This,AMTunerModeType lMode);
      HRESULT (WINAPI *get_Mode)(IAMTVTuner *This,AMTunerModeType *plMode);
      HRESULT (WINAPI *GetAvailableModes)(IAMTVTuner *This,long *plModes);
      HRESULT (WINAPI *RegisterNotificationCallBack)(IAMTVTuner *This,IAMTunerNotification *pNotify,long lEvents);
      HRESULT (WINAPI *UnRegisterNotificationCallBack)(IAMTVTuner *This,IAMTunerNotification *pNotify);
      HRESULT (WINAPI *get_AvailableTVFormats)(IAMTVTuner *This,long *lAnalogVideoStandard);
      HRESULT (WINAPI *get_TVFormat)(IAMTVTuner *This,long *plAnalogVideoStandard);
      HRESULT (WINAPI *AutoTune)(IAMTVTuner *This,long lChannel,long *plFoundSignal);
      HRESULT (WINAPI *StoreAutoTune)(IAMTVTuner *This);
      HRESULT (WINAPI *get_NumInputConnections)(IAMTVTuner *This,long *plNumInputConnections);
      HRESULT (WINAPI *put_InputType)(IAMTVTuner *This,long lIndex,TunerInputType InputType);
      HRESULT (WINAPI *get_InputType)(IAMTVTuner *This,long lIndex,TunerInputType *pInputType);
      HRESULT (WINAPI *put_ConnectInput)(IAMTVTuner *This,long lIndex);
      HRESULT (WINAPI *get_ConnectInput)(IAMTVTuner *This,long *plIndex);
      HRESULT (WINAPI *get_VideoFrequency)(IAMTVTuner *This,long *lFreq);
      HRESULT (WINAPI *get_AudioFrequency)(IAMTVTuner *This,long *lFreq);
    END_INTERFACE
  } IAMTVTunerVtbl;
  struct IAMTVTuner {
    CONST_VTBL struct IAMTVTunerVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMTVTuner_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMTVTuner_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMTVTuner_Release(This) (This)->lpVtbl->Release(This)
#define IAMTVTuner_put_Channel(This,lChannel,lVideoSubChannel,lAudioSubChannel) (This)->lpVtbl->put_Channel(This,lChannel,lVideoSubChannel,lAudioSubChannel)
#define IAMTVTuner_get_Channel(This,plChannel,plVideoSubChannel,plAudioSubChannel) (This)->lpVtbl->get_Channel(This,plChannel,plVideoSubChannel,plAudioSubChannel)
#define IAMTVTuner_ChannelMinMax(This,lChannelMin,lChannelMax) (This)->lpVtbl->ChannelMinMax(This,lChannelMin,lChannelMax)
#define IAMTVTuner_put_CountryCode(This,lCountryCode) (This)->lpVtbl->put_CountryCode(This,lCountryCode)
#define IAMTVTuner_get_CountryCode(This,plCountryCode) (This)->lpVtbl->get_CountryCode(This,plCountryCode)
#define IAMTVTuner_put_TuningSpace(This,lTuningSpace) (This)->lpVtbl->put_TuningSpace(This,lTuningSpace)
#define IAMTVTuner_get_TuningSpace(This,plTuningSpace) (This)->lpVtbl->get_TuningSpace(This,plTuningSpace)
#define IAMTVTuner_Logon(This,hCurrentUser) (This)->lpVtbl->Logon(This,hCurrentUser)
#define IAMTVTuner_Logout(This) (This)->lpVtbl->Logout(This)
#define IAMTVTuner_SignalPresent(This,plSignalStrength) (This)->lpVtbl->SignalPresent(This,plSignalStrength)
#define IAMTVTuner_put_Mode(This,lMode) (This)->lpVtbl->put_Mode(This,lMode)
#define IAMTVTuner_get_Mode(This,plMode) (This)->lpVtbl->get_Mode(This,plMode)
#define IAMTVTuner_GetAvailableModes(This,plModes) (This)->lpVtbl->GetAvailableModes(This,plModes)
#define IAMTVTuner_RegisterNotificationCallBack(This,pNotify,lEvents) (This)->lpVtbl->RegisterNotificationCallBack(This,pNotify,lEvents)
#define IAMTVTuner_UnRegisterNotificationCallBack(This,pNotify) (This)->lpVtbl->UnRegisterNotificationCallBack(This,pNotify)
#define IAMTVTuner_get_AvailableTVFormats(This,lAnalogVideoStandard) (This)->lpVtbl->get_AvailableTVFormats(This,lAnalogVideoStandard)
#define IAMTVTuner_get_TVFormat(This,plAnalogVideoStandard) (This)->lpVtbl->get_TVFormat(This,plAnalogVideoStandard)
#define IAMTVTuner_AutoTune(This,lChannel,plFoundSignal) (This)->lpVtbl->AutoTune(This,lChannel,plFoundSignal)
#define IAMTVTuner_StoreAutoTune(This) (This)->lpVtbl->StoreAutoTune(This)
#define IAMTVTuner_get_NumInputConnections(This,plNumInputConnections) (This)->lpVtbl->get_NumInputConnections(This,plNumInputConnections)
#define IAMTVTuner_put_InputType(This,lIndex,InputType) (This)->lpVtbl->put_InputType(This,lIndex,InputType)
#define IAMTVTuner_get_InputType(This,lIndex,pInputType) (This)->lpVtbl->get_InputType(This,lIndex,pInputType)
#define IAMTVTuner_put_ConnectInput(This,lIndex) (This)->lpVtbl->put_ConnectInput(This,lIndex)
#define IAMTVTuner_get_ConnectInput(This,plIndex) (This)->lpVtbl->get_ConnectInput(This,plIndex)
#define IAMTVTuner_get_VideoFrequency(This,lFreq) (This)->lpVtbl->get_VideoFrequency(This,lFreq)
#define IAMTVTuner_get_AudioFrequency(This,lFreq) (This)->lpVtbl->get_AudioFrequency(This,lFreq)
#endif
#endif
  HRESULT WINAPI IAMTVTuner_get_AvailableTVFormats_Proxy(IAMTVTuner *This,long *lAnalogVideoStandard);
  void __RPC_STUB IAMTVTuner_get_AvailableTVFormats_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTVTuner_get_TVFormat_Proxy(IAMTVTuner *This,long *plAnalogVideoStandard);
  void __RPC_STUB IAMTVTuner_get_TVFormat_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTVTuner_AutoTune_Proxy(IAMTVTuner *This,long lChannel,long *plFoundSignal);
  void __RPC_STUB IAMTVTuner_AutoTune_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTVTuner_StoreAutoTune_Proxy(IAMTVTuner *This);
  void __RPC_STUB IAMTVTuner_StoreAutoTune_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTVTuner_get_NumInputConnections_Proxy(IAMTVTuner *This,long *plNumInputConnections);
  void __RPC_STUB IAMTVTuner_get_NumInputConnections_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTVTuner_put_InputType_Proxy(IAMTVTuner *This,long lIndex,TunerInputType InputType);
  void __RPC_STUB IAMTVTuner_put_InputType_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTVTuner_get_InputType_Proxy(IAMTVTuner *This,long lIndex,TunerInputType *pInputType);
  void __RPC_STUB IAMTVTuner_get_InputType_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTVTuner_put_ConnectInput_Proxy(IAMTVTuner *This,long lIndex);
  void __RPC_STUB IAMTVTuner_put_ConnectInput_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTVTuner_get_ConnectInput_Proxy(IAMTVTuner *This,long *plIndex);
  void __RPC_STUB IAMTVTuner_get_ConnectInput_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTVTuner_get_VideoFrequency_Proxy(IAMTVTuner *This,long *lFreq);
  void __RPC_STUB IAMTVTuner_get_VideoFrequency_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTVTuner_get_AudioFrequency_Proxy(IAMTVTuner *This,long *lFreq);
  void __RPC_STUB IAMTVTuner_get_AudioFrequency_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IBPCSatelliteTuner_INTERFACE_DEFINED__
#define __IBPCSatelliteTuner_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IBPCSatelliteTuner;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IBPCSatelliteTuner : public IAMTuner {
  public:
    virtual HRESULT WINAPI get_DefaultSubChannelTypes(long *plDefaultVideoType,long *plDefaultAudioType) = 0;
    virtual HRESULT WINAPI put_DefaultSubChannelTypes(long lDefaultVideoType,long lDefaultAudioType) = 0;
    virtual HRESULT WINAPI IsTapingPermitted(void) = 0;
  };
#else
  typedef struct IBPCSatelliteTunerVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IBPCSatelliteTuner *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IBPCSatelliteTuner *This);
      ULONG (WINAPI *Release)(IBPCSatelliteTuner *This);
      HRESULT (WINAPI *put_Channel)(IBPCSatelliteTuner *This,long lChannel,long lVideoSubChannel,long lAudioSubChannel);
      HRESULT (WINAPI *get_Channel)(IBPCSatelliteTuner *This,long *plChannel,long *plVideoSubChannel,long *plAudioSubChannel);
      HRESULT (WINAPI *ChannelMinMax)(IBPCSatelliteTuner *This,long *lChannelMin,long *lChannelMax);
      HRESULT (WINAPI *put_CountryCode)(IBPCSatelliteTuner *This,long lCountryCode);
      HRESULT (WINAPI *get_CountryCode)(IBPCSatelliteTuner *This,long *plCountryCode);
      HRESULT (WINAPI *put_TuningSpace)(IBPCSatelliteTuner *This,long lTuningSpace);
      HRESULT (WINAPI *get_TuningSpace)(IBPCSatelliteTuner *This,long *plTuningSpace);
      HRESULT (WINAPI *Logon)(IBPCSatelliteTuner *This,HANDLE hCurrentUser);
      HRESULT (WINAPI *Logout)(IBPCSatelliteTuner *This);
      HRESULT (WINAPI *SignalPresent)(IBPCSatelliteTuner *This,long *plSignalStrength);
      HRESULT (WINAPI *put_Mode)(IBPCSatelliteTuner *This,AMTunerModeType lMode);
      HRESULT (WINAPI *get_Mode)(IBPCSatelliteTuner *This,AMTunerModeType *plMode);
      HRESULT (WINAPI *GetAvailableModes)(IBPCSatelliteTuner *This,long *plModes);
      HRESULT (WINAPI *RegisterNotificationCallBack)(IBPCSatelliteTuner *This,IAMTunerNotification *pNotify,long lEvents);
      HRESULT (WINAPI *UnRegisterNotificationCallBack)(IBPCSatelliteTuner *This,IAMTunerNotification *pNotify);
      HRESULT (WINAPI *get_DefaultSubChannelTypes)(IBPCSatelliteTuner *This,long *plDefaultVideoType,long *plDefaultAudioType);
      HRESULT (WINAPI *put_DefaultSubChannelTypes)(IBPCSatelliteTuner *This,long lDefaultVideoType,long lDefaultAudioType);
      HRESULT (WINAPI *IsTapingPermitted)(IBPCSatelliteTuner *This);
    END_INTERFACE
  } IBPCSatelliteTunerVtbl;
  struct IBPCSatelliteTuner {
    CONST_VTBL struct IBPCSatelliteTunerVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IBPCSatelliteTuner_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IBPCSatelliteTuner_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IBPCSatelliteTuner_Release(This) (This)->lpVtbl->Release(This)
#define IBPCSatelliteTuner_put_Channel(This,lChannel,lVideoSubChannel,lAudioSubChannel) (This)->lpVtbl->put_Channel(This,lChannel,lVideoSubChannel,lAudioSubChannel)
#define IBPCSatelliteTuner_get_Channel(This,plChannel,plVideoSubChannel,plAudioSubChannel) (This)->lpVtbl->get_Channel(This,plChannel,plVideoSubChannel,plAudioSubChannel)
#define IBPCSatelliteTuner_ChannelMinMax(This,lChannelMin,lChannelMax) (This)->lpVtbl->ChannelMinMax(This,lChannelMin,lChannelMax)
#define IBPCSatelliteTuner_put_CountryCode(This,lCountryCode) (This)->lpVtbl->put_CountryCode(This,lCountryCode)
#define IBPCSatelliteTuner_get_CountryCode(This,plCountryCode) (This)->lpVtbl->get_CountryCode(This,plCountryCode)
#define IBPCSatelliteTuner_put_TuningSpace(This,lTuningSpace) (This)->lpVtbl->put_TuningSpace(This,lTuningSpace)
#define IBPCSatelliteTuner_get_TuningSpace(This,plTuningSpace) (This)->lpVtbl->get_TuningSpace(This,plTuningSpace)
#define IBPCSatelliteTuner_Logon(This,hCurrentUser) (This)->lpVtbl->Logon(This,hCurrentUser)
#define IBPCSatelliteTuner_Logout(This) (This)->lpVtbl->Logout(This)
#define IBPCSatelliteTuner_SignalPresent(This,plSignalStrength) (This)->lpVtbl->SignalPresent(This,plSignalStrength)
#define IBPCSatelliteTuner_put_Mode(This,lMode) (This)->lpVtbl->put_Mode(This,lMode)
#define IBPCSatelliteTuner_get_Mode(This,plMode) (This)->lpVtbl->get_Mode(This,plMode)
#define IBPCSatelliteTuner_GetAvailableModes(This,plModes) (This)->lpVtbl->GetAvailableModes(This,plModes)
#define IBPCSatelliteTuner_RegisterNotificationCallBack(This,pNotify,lEvents) (This)->lpVtbl->RegisterNotificationCallBack(This,pNotify,lEvents)
#define IBPCSatelliteTuner_UnRegisterNotificationCallBack(This,pNotify) (This)->lpVtbl->UnRegisterNotificationCallBack(This,pNotify)
#define IBPCSatelliteTuner_get_DefaultSubChannelTypes(This,plDefaultVideoType,plDefaultAudioType) (This)->lpVtbl->get_DefaultSubChannelTypes(This,plDefaultVideoType,plDefaultAudioType)
#define IBPCSatelliteTuner_put_DefaultSubChannelTypes(This,lDefaultVideoType,lDefaultAudioType) (This)->lpVtbl->put_DefaultSubChannelTypes(This,lDefaultVideoType,lDefaultAudioType)
#define IBPCSatelliteTuner_IsTapingPermitted(This) (This)->lpVtbl->IsTapingPermitted(This)
#endif
#endif
  HRESULT WINAPI IBPCSatelliteTuner_get_DefaultSubChannelTypes_Proxy(IBPCSatelliteTuner *This,long *plDefaultVideoType,long *plDefaultAudioType);
  void __RPC_STUB IBPCSatelliteTuner_get_DefaultSubChannelTypes_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBPCSatelliteTuner_put_DefaultSubChannelTypes_Proxy(IBPCSatelliteTuner *This,long lDefaultVideoType,long lDefaultAudioType);
  void __RPC_STUB IBPCSatelliteTuner_put_DefaultSubChannelTypes_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBPCSatelliteTuner_IsTapingPermitted_Proxy(IBPCSatelliteTuner *This);
  void __RPC_STUB IBPCSatelliteTuner_IsTapingPermitted_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef enum tagTVAudioMode {
    AMTVAUDIO_MODE_MONO = 0x1,AMTVAUDIO_MODE_STEREO = 0x2,AMTVAUDIO_MODE_LANG_A = 0x10,AMTVAUDIO_MODE_LANG_B = 0x20,AMTVAUDIO_MODE_LANG_C = 0x40
  } TVAudioMode;

  typedef enum tagAMTVAudioEventType {
    AMTVAUDIO_EVENT_CHANGED = 0x1
  } AMTVAudioEventType;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0182_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0182_v0_0_s_ifspec;
#ifndef __IAMTVAudio_INTERFACE_DEFINED__
#define __IAMTVAudio_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMTVAudio;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMTVAudio : public IUnknown {
  public:
    virtual HRESULT WINAPI GetHardwareSupportedTVAudioModes(long *plModes) = 0;
    virtual HRESULT WINAPI GetAvailableTVAudioModes(long *plModes) = 0;
    virtual HRESULT WINAPI get_TVAudioMode(long *plMode) = 0;
    virtual HRESULT WINAPI put_TVAudioMode(long lMode) = 0;
    virtual HRESULT WINAPI RegisterNotificationCallBack(IAMTunerNotification *pNotify,long lEvents) = 0;
    virtual HRESULT WINAPI UnRegisterNotificationCallBack(IAMTunerNotification *pNotify) = 0;
  };
#else
  typedef struct IAMTVAudioVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMTVAudio *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMTVAudio *This);
      ULONG (WINAPI *Release)(IAMTVAudio *This);
      HRESULT (WINAPI *GetHardwareSupportedTVAudioModes)(IAMTVAudio *This,long *plModes);
      HRESULT (WINAPI *GetAvailableTVAudioModes)(IAMTVAudio *This,long *plModes);
      HRESULT (WINAPI *get_TVAudioMode)(IAMTVAudio *This,long *plMode);
      HRESULT (WINAPI *put_TVAudioMode)(IAMTVAudio *This,long lMode);
      HRESULT (WINAPI *RegisterNotificationCallBack)(IAMTVAudio *This,IAMTunerNotification *pNotify,long lEvents);
      HRESULT (WINAPI *UnRegisterNotificationCallBack)(IAMTVAudio *This,IAMTunerNotification *pNotify);
    END_INTERFACE
  } IAMTVAudioVtbl;
  struct IAMTVAudio {
    CONST_VTBL struct IAMTVAudioVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMTVAudio_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMTVAudio_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMTVAudio_Release(This) (This)->lpVtbl->Release(This)
#define IAMTVAudio_GetHardwareSupportedTVAudioModes(This,plModes) (This)->lpVtbl->GetHardwareSupportedTVAudioModes(This,plModes)
#define IAMTVAudio_GetAvailableTVAudioModes(This,plModes) (This)->lpVtbl->GetAvailableTVAudioModes(This,plModes)
#define IAMTVAudio_get_TVAudioMode(This,plMode) (This)->lpVtbl->get_TVAudioMode(This,plMode)
#define IAMTVAudio_put_TVAudioMode(This,lMode) (This)->lpVtbl->put_TVAudioMode(This,lMode)
#define IAMTVAudio_RegisterNotificationCallBack(This,pNotify,lEvents) (This)->lpVtbl->RegisterNotificationCallBack(This,pNotify,lEvents)
#define IAMTVAudio_UnRegisterNotificationCallBack(This,pNotify) (This)->lpVtbl->UnRegisterNotificationCallBack(This,pNotify)
#endif
#endif
  HRESULT WINAPI IAMTVAudio_GetHardwareSupportedTVAudioModes_Proxy(IAMTVAudio *This,long *plModes);
  void __RPC_STUB IAMTVAudio_GetHardwareSupportedTVAudioModes_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTVAudio_GetAvailableTVAudioModes_Proxy(IAMTVAudio *This,long *plModes);
  void __RPC_STUB IAMTVAudio_GetAvailableTVAudioModes_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTVAudio_get_TVAudioMode_Proxy(IAMTVAudio *This,long *plMode);
  void __RPC_STUB IAMTVAudio_get_TVAudioMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTVAudio_put_TVAudioMode_Proxy(IAMTVAudio *This,long lMode);
  void __RPC_STUB IAMTVAudio_put_TVAudioMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTVAudio_RegisterNotificationCallBack_Proxy(IAMTVAudio *This,IAMTunerNotification *pNotify,long lEvents);
  void __RPC_STUB IAMTVAudio_RegisterNotificationCallBack_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTVAudio_UnRegisterNotificationCallBack_Proxy(IAMTVAudio *This,IAMTunerNotification *pNotify);
  void __RPC_STUB IAMTVAudio_UnRegisterNotificationCallBack_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IAMTVAudioNotification_INTERFACE_DEFINED__
#define __IAMTVAudioNotification_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMTVAudioNotification;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMTVAudioNotification : public IUnknown {
  public:
    virtual HRESULT WINAPI OnEvent(AMTVAudioEventType Event) = 0;
  };
#else
  typedef struct IAMTVAudioNotificationVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMTVAudioNotification *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMTVAudioNotification *This);
      ULONG (WINAPI *Release)(IAMTVAudioNotification *This);
      HRESULT (WINAPI *OnEvent)(IAMTVAudioNotification *This,AMTVAudioEventType Event);
    END_INTERFACE
  } IAMTVAudioNotificationVtbl;
  struct IAMTVAudioNotification {
    CONST_VTBL struct IAMTVAudioNotificationVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMTVAudioNotification_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMTVAudioNotification_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMTVAudioNotification_Release(This) (This)->lpVtbl->Release(This)
#define IAMTVAudioNotification_OnEvent(This,Event) (This)->lpVtbl->OnEvent(This,Event)
#endif
#endif
  HRESULT WINAPI IAMTVAudioNotification_OnEvent_Proxy(IAMTVAudioNotification *This,AMTVAudioEventType Event);
  void __RPC_STUB IAMTVAudioNotification_OnEvent_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IAMAnalogVideoEncoder_INTERFACE_DEFINED__
#define __IAMAnalogVideoEncoder_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMAnalogVideoEncoder;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMAnalogVideoEncoder : public IUnknown {
  public:
    virtual HRESULT WINAPI get_AvailableTVFormats(long *lAnalogVideoStandard) = 0;
    virtual HRESULT WINAPI put_TVFormat(long lAnalogVideoStandard) = 0;
    virtual HRESULT WINAPI get_TVFormat(long *plAnalogVideoStandard) = 0;
    virtual HRESULT WINAPI put_CopyProtection(long lVideoCopyProtection) = 0;
    virtual HRESULT WINAPI get_CopyProtection(long *lVideoCopyProtection) = 0;
    virtual HRESULT WINAPI put_CCEnable(long lCCEnable) = 0;
    virtual HRESULT WINAPI get_CCEnable(long *lCCEnable) = 0;
  };
#else
  typedef struct IAMAnalogVideoEncoderVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMAnalogVideoEncoder *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMAnalogVideoEncoder *This);
      ULONG (WINAPI *Release)(IAMAnalogVideoEncoder *This);
      HRESULT (WINAPI *get_AvailableTVFormats)(IAMAnalogVideoEncoder *This,long *lAnalogVideoStandard);
      HRESULT (WINAPI *put_TVFormat)(IAMAnalogVideoEncoder *This,long lAnalogVideoStandard);
      HRESULT (WINAPI *get_TVFormat)(IAMAnalogVideoEncoder *This,long *plAnalogVideoStandard);
      HRESULT (WINAPI *put_CopyProtection)(IAMAnalogVideoEncoder *This,long lVideoCopyProtection);
      HRESULT (WINAPI *get_CopyProtection)(IAMAnalogVideoEncoder *This,long *lVideoCopyProtection);
      HRESULT (WINAPI *put_CCEnable)(IAMAnalogVideoEncoder *This,long lCCEnable);
      HRESULT (WINAPI *get_CCEnable)(IAMAnalogVideoEncoder *This,long *lCCEnable);
    END_INTERFACE
  } IAMAnalogVideoEncoderVtbl;
  struct IAMAnalogVideoEncoder {
    CONST_VTBL struct IAMAnalogVideoEncoderVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMAnalogVideoEncoder_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMAnalogVideoEncoder_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMAnalogVideoEncoder_Release(This) (This)->lpVtbl->Release(This)
#define IAMAnalogVideoEncoder_get_AvailableTVFormats(This,lAnalogVideoStandard) (This)->lpVtbl->get_AvailableTVFormats(This,lAnalogVideoStandard)
#define IAMAnalogVideoEncoder_put_TVFormat(This,lAnalogVideoStandard) (This)->lpVtbl->put_TVFormat(This,lAnalogVideoStandard)
#define IAMAnalogVideoEncoder_get_TVFormat(This,plAnalogVideoStandard) (This)->lpVtbl->get_TVFormat(This,plAnalogVideoStandard)
#define IAMAnalogVideoEncoder_put_CopyProtection(This,lVideoCopyProtection) (This)->lpVtbl->put_CopyProtection(This,lVideoCopyProtection)
#define IAMAnalogVideoEncoder_get_CopyProtection(This,lVideoCopyProtection) (This)->lpVtbl->get_CopyProtection(This,lVideoCopyProtection)
#define IAMAnalogVideoEncoder_put_CCEnable(This,lCCEnable) (This)->lpVtbl->put_CCEnable(This,lCCEnable)
#define IAMAnalogVideoEncoder_get_CCEnable(This,lCCEnable) (This)->lpVtbl->get_CCEnable(This,lCCEnable)
#endif
#endif
  HRESULT WINAPI IAMAnalogVideoEncoder_get_AvailableTVFormats_Proxy(IAMAnalogVideoEncoder *This,long *lAnalogVideoStandard);
  void __RPC_STUB IAMAnalogVideoEncoder_get_AvailableTVFormats_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAnalogVideoEncoder_put_TVFormat_Proxy(IAMAnalogVideoEncoder *This,long lAnalogVideoStandard);
  void __RPC_STUB IAMAnalogVideoEncoder_put_TVFormat_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAnalogVideoEncoder_get_TVFormat_Proxy(IAMAnalogVideoEncoder *This,long *plAnalogVideoStandard);
  void __RPC_STUB IAMAnalogVideoEncoder_get_TVFormat_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAnalogVideoEncoder_put_CopyProtection_Proxy(IAMAnalogVideoEncoder *This,long lVideoCopyProtection);
  void __RPC_STUB IAMAnalogVideoEncoder_put_CopyProtection_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAnalogVideoEncoder_get_CopyProtection_Proxy(IAMAnalogVideoEncoder *This,long *lVideoCopyProtection);
  void __RPC_STUB IAMAnalogVideoEncoder_get_CopyProtection_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAnalogVideoEncoder_put_CCEnable_Proxy(IAMAnalogVideoEncoder *This,long lCCEnable);
  void __RPC_STUB IAMAnalogVideoEncoder_put_CCEnable_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMAnalogVideoEncoder_get_CCEnable_Proxy(IAMAnalogVideoEncoder *This,long *lCCEnable);
  void __RPC_STUB IAMAnalogVideoEncoder_get_CCEnable_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef enum __MIDL___MIDL_itf_strmif_0185_0001 {
    AMPROPERTY_PIN_CATEGORY = 0,AMPROPERTY_PIN_MEDIUM = AMPROPERTY_PIN_CATEGORY + 1
  } AMPROPERTY_PIN;

#ifndef _IKsPropertySet_
#define _IKsPropertySet_
#define KSPROPERTY_SUPPORT_GET 1
#define KSPROPERTY_SUPPORT_SET 2

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0185_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0185_v0_0_s_ifspec;
#ifndef __IKsPropertySet_INTERFACE_DEFINED__
#define __IKsPropertySet_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IKsPropertySet;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IKsPropertySet : public IUnknown {
  public:
    virtual HRESULT WINAPI Set(REFGUID guidPropSet,DWORD dwPropID,LPVOID pInstanceData,DWORD cbInstanceData,LPVOID pPropData,DWORD cbPropData) = 0;
    virtual HRESULT WINAPI Get(REFGUID guidPropSet,DWORD dwPropID,LPVOID pInstanceData,DWORD cbInstanceData,LPVOID pPropData,DWORD cbPropData,DWORD *pcbReturned) = 0;
    virtual HRESULT WINAPI QuerySupported(REFGUID guidPropSet,DWORD dwPropID,DWORD *pTypeSupport) = 0;
  };
#else
  typedef struct IKsPropertySetVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IKsPropertySet *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IKsPropertySet *This);
      ULONG (WINAPI *Release)(IKsPropertySet *This);
      HRESULT (WINAPI *Set)(IKsPropertySet *This,REFGUID guidPropSet,DWORD dwPropID,LPVOID pInstanceData,DWORD cbInstanceData,LPVOID pPropData,DWORD cbPropData);
      HRESULT (WINAPI *Get)(IKsPropertySet *This,REFGUID guidPropSet,DWORD dwPropID,LPVOID pInstanceData,DWORD cbInstanceData,LPVOID pPropData,DWORD cbPropData,DWORD *pcbReturned);
      HRESULT (WINAPI *QuerySupported)(IKsPropertySet *This,REFGUID guidPropSet,DWORD dwPropID,DWORD *pTypeSupport);
    END_INTERFACE
  } IKsPropertySetVtbl;
  struct IKsPropertySet {
    CONST_VTBL struct IKsPropertySetVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IKsPropertySet_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IKsPropertySet_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IKsPropertySet_Release(This) (This)->lpVtbl->Release(This)
#define IKsPropertySet_Set(This,guidPropSet,dwPropID,pInstanceData,cbInstanceData,pPropData,cbPropData) (This)->lpVtbl->Set(This,guidPropSet,dwPropID,pInstanceData,cbInstanceData,pPropData,cbPropData)
#define IKsPropertySet_Get(This,guidPropSet,dwPropID,pInstanceData,cbInstanceData,pPropData,cbPropData,pcbReturned) (This)->lpVtbl->Get(This,guidPropSet,dwPropID,pInstanceData,cbInstanceData,pPropData,cbPropData,pcbReturned)
#define IKsPropertySet_QuerySupported(This,guidPropSet,dwPropID,pTypeSupport) (This)->lpVtbl->QuerySupported(This,guidPropSet,dwPropID,pTypeSupport)
#endif
#endif
  HRESULT WINAPI IKsPropertySet_RemoteSet_Proxy(IKsPropertySet *This,REFGUID guidPropSet,DWORD dwPropID,byte *pInstanceData,DWORD cbInstanceData,byte *pPropData,DWORD cbPropData);
  void __RPC_STUB IKsPropertySet_RemoteSet_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IKsPropertySet_RemoteGet_Proxy(IKsPropertySet *This,REFGUID guidPropSet,DWORD dwPropID,byte *pInstanceData,DWORD cbInstanceData,byte *pPropData,DWORD cbPropData,DWORD *pcbReturned);
  void __RPC_STUB IKsPropertySet_RemoteGet_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IKsPropertySet_QuerySupported_Proxy(IKsPropertySet *This,REFGUID guidPropSet,DWORD dwPropID,DWORD *pTypeSupport);
  void __RPC_STUB IKsPropertySet_QuerySupported_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif
#endif

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0186_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0186_v0_0_s_ifspec;
#ifndef __IMediaPropertyBag_INTERFACE_DEFINED__
#define __IMediaPropertyBag_INTERFACE_DEFINED__
  typedef IMediaPropertyBag *LPMEDIAPROPERTYBAG;

  EXTERN_C const IID IID_IMediaPropertyBag;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IMediaPropertyBag : public IPropertyBag {
  public:
    virtual HRESULT WINAPI EnumProperty(ULONG iProperty,VARIANT *pvarPropertyName,VARIANT *pvarPropertyValue) = 0;
  };
#else
  typedef struct IMediaPropertyBagVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IMediaPropertyBag *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IMediaPropertyBag *This);
      ULONG (WINAPI *Release)(IMediaPropertyBag *This);
      HRESULT (WINAPI *Read)(IMediaPropertyBag *This,LPCOLESTR pszPropName,VARIANT *pVar,IErrorLog *pErrorLog);
      HRESULT (WINAPI *Write)(IMediaPropertyBag *This,LPCOLESTR pszPropName,VARIANT *pVar);
      HRESULT (WINAPI *EnumProperty)(IMediaPropertyBag *This,ULONG iProperty,VARIANT *pvarPropertyName,VARIANT *pvarPropertyValue);
    END_INTERFACE
  } IMediaPropertyBagVtbl;
  struct IMediaPropertyBag {
    CONST_VTBL struct IMediaPropertyBagVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IMediaPropertyBag_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMediaPropertyBag_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMediaPropertyBag_Release(This) (This)->lpVtbl->Release(This)
#define IMediaPropertyBag_Read(This,pszPropName,pVar,pErrorLog) (This)->lpVtbl->Read(This,pszPropName,pVar,pErrorLog)
#define IMediaPropertyBag_Write(This,pszPropName,pVar) (This)->lpVtbl->Write(This,pszPropName,pVar)
#define IMediaPropertyBag_EnumProperty(This,iProperty,pvarPropertyName,pvarPropertyValue) (This)->lpVtbl->EnumProperty(This,iProperty,pvarPropertyName,pvarPropertyValue)
#endif
#endif
  HRESULT WINAPI IMediaPropertyBag_EnumProperty_Proxy(IMediaPropertyBag *This,ULONG iProperty,VARIANT *pvarPropertyName,VARIANT *pvarPropertyValue);
  void __RPC_STUB IMediaPropertyBag_EnumProperty_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IPersistMediaPropertyBag_INTERFACE_DEFINED__
#define __IPersistMediaPropertyBag_INTERFACE_DEFINED__
  typedef IPersistMediaPropertyBag *LPPERSISTMEDIAPROPERTYBAG;

  EXTERN_C const IID IID_IPersistMediaPropertyBag;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IPersistMediaPropertyBag : public IPersist {
  public:
    virtual HRESULT WINAPI InitNew(void) = 0;
    virtual HRESULT WINAPI Load(IMediaPropertyBag *pPropBag,IErrorLog *pErrorLog) = 0;
    virtual HRESULT WINAPI Save(IMediaPropertyBag *pPropBag,WINBOOL fClearDirty,WINBOOL fSaveAllProperties) = 0;
  };
#else
  typedef struct IPersistMediaPropertyBagVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IPersistMediaPropertyBag *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IPersistMediaPropertyBag *This);
      ULONG (WINAPI *Release)(IPersistMediaPropertyBag *This);
      HRESULT (WINAPI *GetClassID)(IPersistMediaPropertyBag *This,CLSID *pClassID);
      HRESULT (WINAPI *InitNew)(IPersistMediaPropertyBag *This);
      HRESULT (WINAPI *Load)(IPersistMediaPropertyBag *This,IMediaPropertyBag *pPropBag,IErrorLog *pErrorLog);
      HRESULT (WINAPI *Save)(IPersistMediaPropertyBag *This,IMediaPropertyBag *pPropBag,WINBOOL fClearDirty,WINBOOL fSaveAllProperties);
    END_INTERFACE
  } IPersistMediaPropertyBagVtbl;
  struct IPersistMediaPropertyBag {
    CONST_VTBL struct IPersistMediaPropertyBagVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IPersistMediaPropertyBag_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IPersistMediaPropertyBag_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IPersistMediaPropertyBag_Release(This) (This)->lpVtbl->Release(This)
#define IPersistMediaPropertyBag_GetClassID(This,pClassID) (This)->lpVtbl->GetClassID(This,pClassID)
#define IPersistMediaPropertyBag_InitNew(This) (This)->lpVtbl->InitNew(This)
#define IPersistMediaPropertyBag_Load(This,pPropBag,pErrorLog) (This)->lpVtbl->Load(This,pPropBag,pErrorLog)
#define IPersistMediaPropertyBag_Save(This,pPropBag,fClearDirty,fSaveAllProperties) (This)->lpVtbl->Save(This,pPropBag,fClearDirty,fSaveAllProperties)
#endif
#endif
  HRESULT WINAPI IPersistMediaPropertyBag_InitNew_Proxy(IPersistMediaPropertyBag *This);
  void __RPC_STUB IPersistMediaPropertyBag_InitNew_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPersistMediaPropertyBag_Load_Proxy(IPersistMediaPropertyBag *This,IMediaPropertyBag *pPropBag,IErrorLog *pErrorLog);
  void __RPC_STUB IPersistMediaPropertyBag_Load_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPersistMediaPropertyBag_Save_Proxy(IPersistMediaPropertyBag *This,IMediaPropertyBag *pPropBag,WINBOOL fClearDirty,WINBOOL fSaveAllProperties);
  void __RPC_STUB IPersistMediaPropertyBag_Save_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IAMPhysicalPinInfo_INTERFACE_DEFINED__
#define __IAMPhysicalPinInfo_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMPhysicalPinInfo;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMPhysicalPinInfo : public IUnknown {
  public:
    virtual HRESULT WINAPI GetPhysicalType(long *pType,LPOLESTR *ppszType) = 0;
  };
#else
  typedef struct IAMPhysicalPinInfoVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMPhysicalPinInfo *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMPhysicalPinInfo *This);
      ULONG (WINAPI *Release)(IAMPhysicalPinInfo *This);
      HRESULT (WINAPI *GetPhysicalType)(IAMPhysicalPinInfo *This,long *pType,LPOLESTR *ppszType);
    END_INTERFACE
  } IAMPhysicalPinInfoVtbl;
  struct IAMPhysicalPinInfo {
    CONST_VTBL struct IAMPhysicalPinInfoVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMPhysicalPinInfo_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMPhysicalPinInfo_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMPhysicalPinInfo_Release(This) (This)->lpVtbl->Release(This)
#define IAMPhysicalPinInfo_GetPhysicalType(This,pType,ppszType) (This)->lpVtbl->GetPhysicalType(This,pType,ppszType)
#endif
#endif
  HRESULT WINAPI IAMPhysicalPinInfo_GetPhysicalType_Proxy(IAMPhysicalPinInfo *This,long *pType,LPOLESTR *ppszType);
  void __RPC_STUB IAMPhysicalPinInfo_GetPhysicalType_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IAMPhysicalPinInfo *PAMPHYSICALPININFO;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0338_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0338_v0_0_s_ifspec;
#ifndef __IAMExtDevice_INTERFACE_DEFINED__
#define __IAMExtDevice_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMExtDevice;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMExtDevice : public IUnknown {
  public:
    virtual HRESULT WINAPI GetCapability(long Capability,long *pValue,double *pdblValue) = 0;
    virtual HRESULT WINAPI get_ExternalDeviceID(LPOLESTR *ppszData) = 0;
    virtual HRESULT WINAPI get_ExternalDeviceVersion(LPOLESTR *ppszData) = 0;
    virtual HRESULT WINAPI put_DevicePower(long PowerMode) = 0;
    virtual HRESULT WINAPI get_DevicePower(long *pPowerMode) = 0;
    virtual HRESULT WINAPI Calibrate(HEVENT hEvent,long Mode,long *pStatus) = 0;
    virtual HRESULT WINAPI put_DevicePort(long DevicePort) = 0;
    virtual HRESULT WINAPI get_DevicePort(long *pDevicePort) = 0;
  };
#else
  typedef struct IAMExtDeviceVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMExtDevice *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMExtDevice *This);
      ULONG (WINAPI *Release)(IAMExtDevice *This);
      HRESULT (WINAPI *GetCapability)(IAMExtDevice *This,long Capability,long *pValue,double *pdblValue);
      HRESULT (WINAPI *get_ExternalDeviceID)(IAMExtDevice *This,LPOLESTR *ppszData);
      HRESULT (WINAPI *get_ExternalDeviceVersion)(IAMExtDevice *This,LPOLESTR *ppszData);
      HRESULT (WINAPI *put_DevicePower)(IAMExtDevice *This,long PowerMode);
      HRESULT (WINAPI *get_DevicePower)(IAMExtDevice *This,long *pPowerMode);
      HRESULT (WINAPI *Calibrate)(IAMExtDevice *This,HEVENT hEvent,long Mode,long *pStatus);
      HRESULT (WINAPI *put_DevicePort)(IAMExtDevice *This,long DevicePort);
      HRESULT (WINAPI *get_DevicePort)(IAMExtDevice *This,long *pDevicePort);
    END_INTERFACE
  } IAMExtDeviceVtbl;
  struct IAMExtDevice {
    CONST_VTBL struct IAMExtDeviceVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMExtDevice_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMExtDevice_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMExtDevice_Release(This) (This)->lpVtbl->Release(This)
#define IAMExtDevice_GetCapability(This,Capability,pValue,pdblValue) (This)->lpVtbl->GetCapability(This,Capability,pValue,pdblValue)
#define IAMExtDevice_get_ExternalDeviceID(This,ppszData) (This)->lpVtbl->get_ExternalDeviceID(This,ppszData)
#define IAMExtDevice_get_ExternalDeviceVersion(This,ppszData) (This)->lpVtbl->get_ExternalDeviceVersion(This,ppszData)
#define IAMExtDevice_put_DevicePower(This,PowerMode) (This)->lpVtbl->put_DevicePower(This,PowerMode)
#define IAMExtDevice_get_DevicePower(This,pPowerMode) (This)->lpVtbl->get_DevicePower(This,pPowerMode)
#define IAMExtDevice_Calibrate(This,hEvent,Mode,pStatus) (This)->lpVtbl->Calibrate(This,hEvent,Mode,pStatus)
#define IAMExtDevice_put_DevicePort(This,DevicePort) (This)->lpVtbl->put_DevicePort(This,DevicePort)
#define IAMExtDevice_get_DevicePort(This,pDevicePort) (This)->lpVtbl->get_DevicePort(This,pDevicePort)
#endif
#endif
  HRESULT WINAPI IAMExtDevice_GetCapability_Proxy(IAMExtDevice *This,long Capability,long *pValue,double *pdblValue);
  void __RPC_STUB IAMExtDevice_GetCapability_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtDevice_get_ExternalDeviceID_Proxy(IAMExtDevice *This,LPOLESTR *ppszData);
  void __RPC_STUB IAMExtDevice_get_ExternalDeviceID_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtDevice_get_ExternalDeviceVersion_Proxy(IAMExtDevice *This,LPOLESTR *ppszData);
  void __RPC_STUB IAMExtDevice_get_ExternalDeviceVersion_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtDevice_put_DevicePower_Proxy(IAMExtDevice *This,long PowerMode);
  void __RPC_STUB IAMExtDevice_put_DevicePower_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtDevice_get_DevicePower_Proxy(IAMExtDevice *This,long *pPowerMode);
  void __RPC_STUB IAMExtDevice_get_DevicePower_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtDevice_Calibrate_Proxy(IAMExtDevice *This,HEVENT hEvent,long Mode,long *pStatus);
  void __RPC_STUB IAMExtDevice_Calibrate_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtDevice_put_DevicePort_Proxy(IAMExtDevice *This,long DevicePort);
  void __RPC_STUB IAMExtDevice_put_DevicePort_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtDevice_get_DevicePort_Proxy(IAMExtDevice *This,long *pDevicePort);
  void __RPC_STUB IAMExtDevice_get_DevicePort_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IAMExtDevice *PEXTDEVICE;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0339_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0339_v0_0_s_ifspec;
#ifndef __IAMExtTransport_INTERFACE_DEFINED__
#define __IAMExtTransport_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMExtTransport;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMExtTransport : public IUnknown {
  public:
    virtual HRESULT WINAPI GetCapability(long Capability,long *pValue,double *pdblValue) = 0;
    virtual HRESULT WINAPI put_MediaState(long State) = 0;
    virtual HRESULT WINAPI get_MediaState(long *pState) = 0;
    virtual HRESULT WINAPI put_LocalControl(long State) = 0;
    virtual HRESULT WINAPI get_LocalControl(long *pState) = 0;
    virtual HRESULT WINAPI GetStatus(long StatusItem,long *pValue) = 0;
    virtual HRESULT WINAPI GetTransportBasicParameters(long Param,long *pValue,LPOLESTR *ppszData) = 0;
    virtual HRESULT WINAPI SetTransportBasicParameters(long Param,long Value,LPCOLESTR pszData) = 0;
    virtual HRESULT WINAPI GetTransportVideoParameters(long Param,long *pValue) = 0;
    virtual HRESULT WINAPI SetTransportVideoParameters(long Param,long Value) = 0;
    virtual HRESULT WINAPI GetTransportAudioParameters(long Param,long *pValue) = 0;
    virtual HRESULT WINAPI SetTransportAudioParameters(long Param,long Value) = 0;
    virtual HRESULT WINAPI put_Mode(long Mode) = 0;
    virtual HRESULT WINAPI get_Mode(long *pMode) = 0;
    virtual HRESULT WINAPI put_Rate(double dblRate) = 0;
    virtual HRESULT WINAPI get_Rate(double *pdblRate) = 0;
    virtual HRESULT WINAPI GetChase(long *pEnabled,long *pOffset,HEVENT *phEvent) = 0;
    virtual HRESULT WINAPI SetChase(long Enable,long Offset,HEVENT hEvent) = 0;
    virtual HRESULT WINAPI GetBump(long *pSpeed,long *pDuration) = 0;
    virtual HRESULT WINAPI SetBump(long Speed,long Duration) = 0;
    virtual HRESULT WINAPI get_AntiClogControl(long *pEnabled) = 0;
    virtual HRESULT WINAPI put_AntiClogControl(long Enable) = 0;
    virtual HRESULT WINAPI GetEditPropertySet(long EditID,long *pState) = 0;
    virtual HRESULT WINAPI SetEditPropertySet(long *pEditID,long State) = 0;
    virtual HRESULT WINAPI GetEditProperty(long EditID,long Param,long *pValue) = 0;
    virtual HRESULT WINAPI SetEditProperty(long EditID,long Param,long Value) = 0;
    virtual HRESULT WINAPI get_EditStart(long *pValue) = 0;
    virtual HRESULT WINAPI put_EditStart(long Value) = 0;
  };
#else
  typedef struct IAMExtTransportVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMExtTransport *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMExtTransport *This);
      ULONG (WINAPI *Release)(IAMExtTransport *This);
      HRESULT (WINAPI *GetCapability)(IAMExtTransport *This,long Capability,long *pValue,double *pdblValue);
      HRESULT (WINAPI *put_MediaState)(IAMExtTransport *This,long State);
      HRESULT (WINAPI *get_MediaState)(IAMExtTransport *This,long *pState);
      HRESULT (WINAPI *put_LocalControl)(IAMExtTransport *This,long State);
      HRESULT (WINAPI *get_LocalControl)(IAMExtTransport *This,long *pState);
      HRESULT (WINAPI *GetStatus)(IAMExtTransport *This,long StatusItem,long *pValue);
      HRESULT (WINAPI *GetTransportBasicParameters)(IAMExtTransport *This,long Param,long *pValue,LPOLESTR *ppszData);
      HRESULT (WINAPI *SetTransportBasicParameters)(IAMExtTransport *This,long Param,long Value,LPCOLESTR pszData);
      HRESULT (WINAPI *GetTransportVideoParameters)(IAMExtTransport *This,long Param,long *pValue);
      HRESULT (WINAPI *SetTransportVideoParameters)(IAMExtTransport *This,long Param,long Value);
      HRESULT (WINAPI *GetTransportAudioParameters)(IAMExtTransport *This,long Param,long *pValue);
      HRESULT (WINAPI *SetTransportAudioParameters)(IAMExtTransport *This,long Param,long Value);
      HRESULT (WINAPI *put_Mode)(IAMExtTransport *This,long Mode);
      HRESULT (WINAPI *get_Mode)(IAMExtTransport *This,long *pMode);
      HRESULT (WINAPI *put_Rate)(IAMExtTransport *This,double dblRate);
      HRESULT (WINAPI *get_Rate)(IAMExtTransport *This,double *pdblRate);
      HRESULT (WINAPI *GetChase)(IAMExtTransport *This,long *pEnabled,long *pOffset,HEVENT *phEvent);
      HRESULT (WINAPI *SetChase)(IAMExtTransport *This,long Enable,long Offset,HEVENT hEvent);
      HRESULT (WINAPI *GetBump)(IAMExtTransport *This,long *pSpeed,long *pDuration);
      HRESULT (WINAPI *SetBump)(IAMExtTransport *This,long Speed,long Duration);
      HRESULT (WINAPI *get_AntiClogControl)(IAMExtTransport *This,long *pEnabled);
      HRESULT (WINAPI *put_AntiClogControl)(IAMExtTransport *This,long Enable);
      HRESULT (WINAPI *GetEditPropertySet)(IAMExtTransport *This,long EditID,long *pState);
      HRESULT (WINAPI *SetEditPropertySet)(IAMExtTransport *This,long *pEditID,long State);
      HRESULT (WINAPI *GetEditProperty)(IAMExtTransport *This,long EditID,long Param,long *pValue);
      HRESULT (WINAPI *SetEditProperty)(IAMExtTransport *This,long EditID,long Param,long Value);
      HRESULT (WINAPI *get_EditStart)(IAMExtTransport *This,long *pValue);
      HRESULT (WINAPI *put_EditStart)(IAMExtTransport *This,long Value);
    END_INTERFACE
  } IAMExtTransportVtbl;
  struct IAMExtTransport {
    CONST_VTBL struct IAMExtTransportVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMExtTransport_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMExtTransport_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMExtTransport_Release(This) (This)->lpVtbl->Release(This)
#define IAMExtTransport_GetCapability(This,Capability,pValue,pdblValue) (This)->lpVtbl->GetCapability(This,Capability,pValue,pdblValue)
#define IAMExtTransport_put_MediaState(This,State) (This)->lpVtbl->put_MediaState(This,State)
#define IAMExtTransport_get_MediaState(This,pState) (This)->lpVtbl->get_MediaState(This,pState)
#define IAMExtTransport_put_LocalControl(This,State) (This)->lpVtbl->put_LocalControl(This,State)
#define IAMExtTransport_get_LocalControl(This,pState) (This)->lpVtbl->get_LocalControl(This,pState)
#define IAMExtTransport_GetStatus(This,StatusItem,pValue) (This)->lpVtbl->GetStatus(This,StatusItem,pValue)
#define IAMExtTransport_GetTransportBasicParameters(This,Param,pValue,ppszData) (This)->lpVtbl->GetTransportBasicParameters(This,Param,pValue,ppszData)
#define IAMExtTransport_SetTransportBasicParameters(This,Param,Value,pszData) (This)->lpVtbl->SetTransportBasicParameters(This,Param,Value,pszData)
#define IAMExtTransport_GetTransportVideoParameters(This,Param,pValue) (This)->lpVtbl->GetTransportVideoParameters(This,Param,pValue)
#define IAMExtTransport_SetTransportVideoParameters(This,Param,Value) (This)->lpVtbl->SetTransportVideoParameters(This,Param,Value)
#define IAMExtTransport_GetTransportAudioParameters(This,Param,pValue) (This)->lpVtbl->GetTransportAudioParameters(This,Param,pValue)
#define IAMExtTransport_SetTransportAudioParameters(This,Param,Value) (This)->lpVtbl->SetTransportAudioParameters(This,Param,Value)
#define IAMExtTransport_put_Mode(This,Mode) (This)->lpVtbl->put_Mode(This,Mode)
#define IAMExtTransport_get_Mode(This,pMode) (This)->lpVtbl->get_Mode(This,pMode)
#define IAMExtTransport_put_Rate(This,dblRate) (This)->lpVtbl->put_Rate(This,dblRate)
#define IAMExtTransport_get_Rate(This,pdblRate) (This)->lpVtbl->get_Rate(This,pdblRate)
#define IAMExtTransport_GetChase(This,pEnabled,pOffset,phEvent) (This)->lpVtbl->GetChase(This,pEnabled,pOffset,phEvent)
#define IAMExtTransport_SetChase(This,Enable,Offset,hEvent) (This)->lpVtbl->SetChase(This,Enable,Offset,hEvent)
#define IAMExtTransport_GetBump(This,pSpeed,pDuration) (This)->lpVtbl->GetBump(This,pSpeed,pDuration)
#define IAMExtTransport_SetBump(This,Speed,Duration) (This)->lpVtbl->SetBump(This,Speed,Duration)
#define IAMExtTransport_get_AntiClogControl(This,pEnabled) (This)->lpVtbl->get_AntiClogControl(This,pEnabled)
#define IAMExtTransport_put_AntiClogControl(This,Enable) (This)->lpVtbl->put_AntiClogControl(This,Enable)
#define IAMExtTransport_GetEditPropertySet(This,EditID,pState) (This)->lpVtbl->GetEditPropertySet(This,EditID,pState)
#define IAMExtTransport_SetEditPropertySet(This,pEditID,State) (This)->lpVtbl->SetEditPropertySet(This,pEditID,State)
#define IAMExtTransport_GetEditProperty(This,EditID,Param,pValue) (This)->lpVtbl->GetEditProperty(This,EditID,Param,pValue)
#define IAMExtTransport_SetEditProperty(This,EditID,Param,Value) (This)->lpVtbl->SetEditProperty(This,EditID,Param,Value)
#define IAMExtTransport_get_EditStart(This,pValue) (This)->lpVtbl->get_EditStart(This,pValue)
#define IAMExtTransport_put_EditStart(This,Value) (This)->lpVtbl->put_EditStart(This,Value)
#endif
#endif
  HRESULT WINAPI IAMExtTransport_GetCapability_Proxy(IAMExtTransport *This,long Capability,long *pValue,double *pdblValue);
  void __RPC_STUB IAMExtTransport_GetCapability_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_put_MediaState_Proxy(IAMExtTransport *This,long State);
  void __RPC_STUB IAMExtTransport_put_MediaState_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_get_MediaState_Proxy(IAMExtTransport *This,long *pState);
  void __RPC_STUB IAMExtTransport_get_MediaState_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_put_LocalControl_Proxy(IAMExtTransport *This,long State);
  void __RPC_STUB IAMExtTransport_put_LocalControl_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_get_LocalControl_Proxy(IAMExtTransport *This,long *pState);
  void __RPC_STUB IAMExtTransport_get_LocalControl_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_GetStatus_Proxy(IAMExtTransport *This,long StatusItem,long *pValue);
  void __RPC_STUB IAMExtTransport_GetStatus_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_GetTransportBasicParameters_Proxy(IAMExtTransport *This,long Param,long *pValue,LPOLESTR *ppszData);
  void __RPC_STUB IAMExtTransport_GetTransportBasicParameters_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_SetTransportBasicParameters_Proxy(IAMExtTransport *This,long Param,long Value,LPCOLESTR pszData);
  void __RPC_STUB IAMExtTransport_SetTransportBasicParameters_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_GetTransportVideoParameters_Proxy(IAMExtTransport *This,long Param,long *pValue);
  void __RPC_STUB IAMExtTransport_GetTransportVideoParameters_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_SetTransportVideoParameters_Proxy(IAMExtTransport *This,long Param,long Value);
  void __RPC_STUB IAMExtTransport_SetTransportVideoParameters_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_GetTransportAudioParameters_Proxy(IAMExtTransport *This,long Param,long *pValue);
  void __RPC_STUB IAMExtTransport_GetTransportAudioParameters_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_SetTransportAudioParameters_Proxy(IAMExtTransport *This,long Param,long Value);
  void __RPC_STUB IAMExtTransport_SetTransportAudioParameters_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_put_Mode_Proxy(IAMExtTransport *This,long Mode);
  void __RPC_STUB IAMExtTransport_put_Mode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_get_Mode_Proxy(IAMExtTransport *This,long *pMode);
  void __RPC_STUB IAMExtTransport_get_Mode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_put_Rate_Proxy(IAMExtTransport *This,double dblRate);
  void __RPC_STUB IAMExtTransport_put_Rate_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_get_Rate_Proxy(IAMExtTransport *This,double *pdblRate);
  void __RPC_STUB IAMExtTransport_get_Rate_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_GetChase_Proxy(IAMExtTransport *This,long *pEnabled,long *pOffset,HEVENT *phEvent);
  void __RPC_STUB IAMExtTransport_GetChase_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_SetChase_Proxy(IAMExtTransport *This,long Enable,long Offset,HEVENT hEvent);
  void __RPC_STUB IAMExtTransport_SetChase_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_GetBump_Proxy(IAMExtTransport *This,long *pSpeed,long *pDuration);
  void __RPC_STUB IAMExtTransport_GetBump_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_SetBump_Proxy(IAMExtTransport *This,long Speed,long Duration);
  void __RPC_STUB IAMExtTransport_SetBump_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_get_AntiClogControl_Proxy(IAMExtTransport *This,long *pEnabled);
  void __RPC_STUB IAMExtTransport_get_AntiClogControl_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_put_AntiClogControl_Proxy(IAMExtTransport *This,long Enable);
  void __RPC_STUB IAMExtTransport_put_AntiClogControl_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_GetEditPropertySet_Proxy(IAMExtTransport *This,long EditID,long *pState);
  void __RPC_STUB IAMExtTransport_GetEditPropertySet_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_SetEditPropertySet_Proxy(IAMExtTransport *This,long *pEditID,long State);
  void __RPC_STUB IAMExtTransport_SetEditPropertySet_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_GetEditProperty_Proxy(IAMExtTransport *This,long EditID,long Param,long *pValue);
  void __RPC_STUB IAMExtTransport_GetEditProperty_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_SetEditProperty_Proxy(IAMExtTransport *This,long EditID,long Param,long Value);
  void __RPC_STUB IAMExtTransport_SetEditProperty_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_get_EditStart_Proxy(IAMExtTransport *This,long *pValue);
  void __RPC_STUB IAMExtTransport_get_EditStart_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMExtTransport_put_EditStart_Proxy(IAMExtTransport *This,long Value);
  void __RPC_STUB IAMExtTransport_put_EditStart_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IAMExtTransport *PIAMEXTTRANSPORT;

#ifndef TIMECODE_DEFINED
#define TIMECODE_DEFINED
  typedef union _timecode {
    struct {
      WORD wFrameRate;
      WORD wFrameFract;
      DWORD dwFrames;
    };
    DWORDLONG qw;
  } TIMECODE;
#endif

  typedef TIMECODE *PTIMECODE;

  typedef struct tagTIMECODE_SAMPLE {
    LONGLONG qwTick;
    TIMECODE timecode;
    DWORD dwUser;
    DWORD dwFlags;
  } TIMECODE_SAMPLE;

  typedef TIMECODE_SAMPLE *PTIMECODE_SAMPLE;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0340_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0340_v0_0_s_ifspec;
#ifndef __IAMTimecodeReader_INTERFACE_DEFINED__
#define __IAMTimecodeReader_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMTimecodeReader;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMTimecodeReader : public IUnknown {
  public:
    virtual HRESULT WINAPI GetTCRMode(long Param,long *pValue) = 0;
    virtual HRESULT WINAPI SetTCRMode(long Param,long Value) = 0;
    virtual HRESULT WINAPI put_VITCLine(long Line) = 0;
    virtual HRESULT WINAPI get_VITCLine(long *pLine) = 0;
    virtual HRESULT WINAPI GetTimecode(PTIMECODE_SAMPLE pTimecodeSample) = 0;
  };
#else
  typedef struct IAMTimecodeReaderVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMTimecodeReader *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMTimecodeReader *This);
      ULONG (WINAPI *Release)(IAMTimecodeReader *This);
      HRESULT (WINAPI *GetTCRMode)(IAMTimecodeReader *This,long Param,long *pValue);
      HRESULT (WINAPI *SetTCRMode)(IAMTimecodeReader *This,long Param,long Value);
      HRESULT (WINAPI *put_VITCLine)(IAMTimecodeReader *This,long Line);
      HRESULT (WINAPI *get_VITCLine)(IAMTimecodeReader *This,long *pLine);
      HRESULT (WINAPI *GetTimecode)(IAMTimecodeReader *This,PTIMECODE_SAMPLE pTimecodeSample);
    END_INTERFACE
  } IAMTimecodeReaderVtbl;
  struct IAMTimecodeReader {
    CONST_VTBL struct IAMTimecodeReaderVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMTimecodeReader_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMTimecodeReader_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMTimecodeReader_Release(This) (This)->lpVtbl->Release(This)
#define IAMTimecodeReader_GetTCRMode(This,Param,pValue) (This)->lpVtbl->GetTCRMode(This,Param,pValue)
#define IAMTimecodeReader_SetTCRMode(This,Param,Value) (This)->lpVtbl->SetTCRMode(This,Param,Value)
#define IAMTimecodeReader_put_VITCLine(This,Line) (This)->lpVtbl->put_VITCLine(This,Line)
#define IAMTimecodeReader_get_VITCLine(This,pLine) (This)->lpVtbl->get_VITCLine(This,pLine)
#define IAMTimecodeReader_GetTimecode(This,pTimecodeSample) (This)->lpVtbl->GetTimecode(This,pTimecodeSample)
#endif
#endif
  HRESULT WINAPI IAMTimecodeReader_GetTCRMode_Proxy(IAMTimecodeReader *This,long Param,long *pValue);
  void __RPC_STUB IAMTimecodeReader_GetTCRMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTimecodeReader_SetTCRMode_Proxy(IAMTimecodeReader *This,long Param,long Value);
  void __RPC_STUB IAMTimecodeReader_SetTCRMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTimecodeReader_put_VITCLine_Proxy(IAMTimecodeReader *This,long Line);
  void __RPC_STUB IAMTimecodeReader_put_VITCLine_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTimecodeReader_get_VITCLine_Proxy(IAMTimecodeReader *This,long *pLine);
  void __RPC_STUB IAMTimecodeReader_get_VITCLine_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTimecodeReader_GetTimecode_Proxy(IAMTimecodeReader *This,PTIMECODE_SAMPLE pTimecodeSample);
  void __RPC_STUB IAMTimecodeReader_GetTimecode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IAMTimecodeReader *PIAMTIMECODEREADER;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0341_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0341_v0_0_s_ifspec;
#ifndef __IAMTimecodeGenerator_INTERFACE_DEFINED__
#define __IAMTimecodeGenerator_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMTimecodeGenerator;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMTimecodeGenerator : public IUnknown {
  public:
    virtual HRESULT WINAPI GetTCGMode(long Param,long *pValue) = 0;
    virtual HRESULT WINAPI SetTCGMode(long Param,long Value) = 0;
    virtual HRESULT WINAPI put_VITCLine(long Line) = 0;
    virtual HRESULT WINAPI get_VITCLine(long *pLine) = 0;
    virtual HRESULT WINAPI SetTimecode(PTIMECODE_SAMPLE pTimecodeSample) = 0;
    virtual HRESULT WINAPI GetTimecode(PTIMECODE_SAMPLE pTimecodeSample) = 0;
  };
#else
  typedef struct IAMTimecodeGeneratorVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMTimecodeGenerator *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMTimecodeGenerator *This);
      ULONG (WINAPI *Release)(IAMTimecodeGenerator *This);
      HRESULT (WINAPI *GetTCGMode)(IAMTimecodeGenerator *This,long Param,long *pValue);
      HRESULT (WINAPI *SetTCGMode)(IAMTimecodeGenerator *This,long Param,long Value);
      HRESULT (WINAPI *put_VITCLine)(IAMTimecodeGenerator *This,long Line);
      HRESULT (WINAPI *get_VITCLine)(IAMTimecodeGenerator *This,long *pLine);
      HRESULT (WINAPI *SetTimecode)(IAMTimecodeGenerator *This,PTIMECODE_SAMPLE pTimecodeSample);
      HRESULT (WINAPI *GetTimecode)(IAMTimecodeGenerator *This,PTIMECODE_SAMPLE pTimecodeSample);
    END_INTERFACE
  } IAMTimecodeGeneratorVtbl;
  struct IAMTimecodeGenerator {
    CONST_VTBL struct IAMTimecodeGeneratorVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMTimecodeGenerator_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMTimecodeGenerator_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMTimecodeGenerator_Release(This) (This)->lpVtbl->Release(This)
#define IAMTimecodeGenerator_GetTCGMode(This,Param,pValue) (This)->lpVtbl->GetTCGMode(This,Param,pValue)
#define IAMTimecodeGenerator_SetTCGMode(This,Param,Value) (This)->lpVtbl->SetTCGMode(This,Param,Value)
#define IAMTimecodeGenerator_put_VITCLine(This,Line) (This)->lpVtbl->put_VITCLine(This,Line)
#define IAMTimecodeGenerator_get_VITCLine(This,pLine) (This)->lpVtbl->get_VITCLine(This,pLine)
#define IAMTimecodeGenerator_SetTimecode(This,pTimecodeSample) (This)->lpVtbl->SetTimecode(This,pTimecodeSample)
#define IAMTimecodeGenerator_GetTimecode(This,pTimecodeSample) (This)->lpVtbl->GetTimecode(This,pTimecodeSample)
#endif
#endif
  HRESULT WINAPI IAMTimecodeGenerator_GetTCGMode_Proxy(IAMTimecodeGenerator *This,long Param,long *pValue);
  void __RPC_STUB IAMTimecodeGenerator_GetTCGMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTimecodeGenerator_SetTCGMode_Proxy(IAMTimecodeGenerator *This,long Param,long Value);
  void __RPC_STUB IAMTimecodeGenerator_SetTCGMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTimecodeGenerator_put_VITCLine_Proxy(IAMTimecodeGenerator *This,long Line);
  void __RPC_STUB IAMTimecodeGenerator_put_VITCLine_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTimecodeGenerator_get_VITCLine_Proxy(IAMTimecodeGenerator *This,long *pLine);
  void __RPC_STUB IAMTimecodeGenerator_get_VITCLine_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTimecodeGenerator_SetTimecode_Proxy(IAMTimecodeGenerator *This,PTIMECODE_SAMPLE pTimecodeSample);
  void __RPC_STUB IAMTimecodeGenerator_SetTimecode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTimecodeGenerator_GetTimecode_Proxy(IAMTimecodeGenerator *This,PTIMECODE_SAMPLE pTimecodeSample);
  void __RPC_STUB IAMTimecodeGenerator_GetTimecode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IAMTimecodeGenerator *PIAMTIMECODEGENERATOR;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0342_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0342_v0_0_s_ifspec;
#ifndef __IAMTimecodeDisplay_INTERFACE_DEFINED__
#define __IAMTimecodeDisplay_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMTimecodeDisplay;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMTimecodeDisplay : public IUnknown {
  public:
    virtual HRESULT WINAPI GetTCDisplayEnable(long *pState) = 0;
    virtual HRESULT WINAPI SetTCDisplayEnable(long State) = 0;
    virtual HRESULT WINAPI GetTCDisplay(long Param,long *pValue) = 0;
    virtual HRESULT WINAPI SetTCDisplay(long Param,long Value) = 0;
  };
#else
  typedef struct IAMTimecodeDisplayVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMTimecodeDisplay *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMTimecodeDisplay *This);
      ULONG (WINAPI *Release)(IAMTimecodeDisplay *This);
      HRESULT (WINAPI *GetTCDisplayEnable)(IAMTimecodeDisplay *This,long *pState);
      HRESULT (WINAPI *SetTCDisplayEnable)(IAMTimecodeDisplay *This,long State);
      HRESULT (WINAPI *GetTCDisplay)(IAMTimecodeDisplay *This,long Param,long *pValue);
      HRESULT (WINAPI *SetTCDisplay)(IAMTimecodeDisplay *This,long Param,long Value);
    END_INTERFACE
  } IAMTimecodeDisplayVtbl;
  struct IAMTimecodeDisplay {
    CONST_VTBL struct IAMTimecodeDisplayVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMTimecodeDisplay_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMTimecodeDisplay_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMTimecodeDisplay_Release(This) (This)->lpVtbl->Release(This)
#define IAMTimecodeDisplay_GetTCDisplayEnable(This,pState) (This)->lpVtbl->GetTCDisplayEnable(This,pState)
#define IAMTimecodeDisplay_SetTCDisplayEnable(This,State) (This)->lpVtbl->SetTCDisplayEnable(This,State)
#define IAMTimecodeDisplay_GetTCDisplay(This,Param,pValue) (This)->lpVtbl->GetTCDisplay(This,Param,pValue)
#define IAMTimecodeDisplay_SetTCDisplay(This,Param,Value) (This)->lpVtbl->SetTCDisplay(This,Param,Value)
#endif
#endif
  HRESULT WINAPI IAMTimecodeDisplay_GetTCDisplayEnable_Proxy(IAMTimecodeDisplay *This,long *pState);
  void __RPC_STUB IAMTimecodeDisplay_GetTCDisplayEnable_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTimecodeDisplay_SetTCDisplayEnable_Proxy(IAMTimecodeDisplay *This,long State);
  void __RPC_STUB IAMTimecodeDisplay_SetTCDisplayEnable_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTimecodeDisplay_GetTCDisplay_Proxy(IAMTimecodeDisplay *This,long Param,long *pValue);
  void __RPC_STUB IAMTimecodeDisplay_GetTCDisplay_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMTimecodeDisplay_SetTCDisplay_Proxy(IAMTimecodeDisplay *This,long Param,long Value);
  void __RPC_STUB IAMTimecodeDisplay_SetTCDisplay_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IAMTimecodeDisplay *PIAMTIMECODEDISPLAY;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0343_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0343_v0_0_s_ifspec;
#ifndef __IAMDevMemoryAllocator_INTERFACE_DEFINED__
#define __IAMDevMemoryAllocator_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMDevMemoryAllocator;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMDevMemoryAllocator : public IUnknown {
  public:
    virtual HRESULT WINAPI GetInfo(DWORD *pdwcbTotalFree,DWORD *pdwcbLargestFree,DWORD *pdwcbTotalMemory,DWORD *pdwcbMinimumChunk) = 0;
    virtual HRESULT WINAPI CheckMemory(const BYTE *pBuffer) = 0;
    virtual HRESULT WINAPI Alloc(BYTE **ppBuffer,DWORD *pdwcbBuffer) = 0;
    virtual HRESULT WINAPI Free(BYTE *pBuffer) = 0;
    virtual HRESULT WINAPI GetDevMemoryObject(IUnknown **ppUnkInnner,IUnknown *pUnkOuter) = 0;
  };
#else
  typedef struct IAMDevMemoryAllocatorVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMDevMemoryAllocator *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMDevMemoryAllocator *This);
      ULONG (WINAPI *Release)(IAMDevMemoryAllocator *This);
      HRESULT (WINAPI *GetInfo)(IAMDevMemoryAllocator *This,DWORD *pdwcbTotalFree,DWORD *pdwcbLargestFree,DWORD *pdwcbTotalMemory,DWORD *pdwcbMinimumChunk);
      HRESULT (WINAPI *CheckMemory)(IAMDevMemoryAllocator *This,const BYTE *pBuffer);
      HRESULT (WINAPI *Alloc)(IAMDevMemoryAllocator *This,BYTE **ppBuffer,DWORD *pdwcbBuffer);
      HRESULT (WINAPI *Free)(IAMDevMemoryAllocator *This,BYTE *pBuffer);
      HRESULT (WINAPI *GetDevMemoryObject)(IAMDevMemoryAllocator *This,IUnknown **ppUnkInnner,IUnknown *pUnkOuter);
    END_INTERFACE
  } IAMDevMemoryAllocatorVtbl;
  struct IAMDevMemoryAllocator {
    CONST_VTBL struct IAMDevMemoryAllocatorVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMDevMemoryAllocator_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMDevMemoryAllocator_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMDevMemoryAllocator_Release(This) (This)->lpVtbl->Release(This)
#define IAMDevMemoryAllocator_GetInfo(This,pdwcbTotalFree,pdwcbLargestFree,pdwcbTotalMemory,pdwcbMinimumChunk) (This)->lpVtbl->GetInfo(This,pdwcbTotalFree,pdwcbLargestFree,pdwcbTotalMemory,pdwcbMinimumChunk)
#define IAMDevMemoryAllocator_CheckMemory(This,pBuffer) (This)->lpVtbl->CheckMemory(This,pBuffer)
#define IAMDevMemoryAllocator_Alloc(This,ppBuffer,pdwcbBuffer) (This)->lpVtbl->Alloc(This,ppBuffer,pdwcbBuffer)
#define IAMDevMemoryAllocator_Free(This,pBuffer) (This)->lpVtbl->Free(This,pBuffer)
#define IAMDevMemoryAllocator_GetDevMemoryObject(This,ppUnkInnner,pUnkOuter) (This)->lpVtbl->GetDevMemoryObject(This,ppUnkInnner,pUnkOuter)
#endif
#endif
  HRESULT WINAPI IAMDevMemoryAllocator_GetInfo_Proxy(IAMDevMemoryAllocator *This,DWORD *pdwcbTotalFree,DWORD *pdwcbLargestFree,DWORD *pdwcbTotalMemory,DWORD *pdwcbMinimumChunk);
  void __RPC_STUB IAMDevMemoryAllocator_GetInfo_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMDevMemoryAllocator_CheckMemory_Proxy(IAMDevMemoryAllocator *This,const BYTE *pBuffer);
  void __RPC_STUB IAMDevMemoryAllocator_CheckMemory_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMDevMemoryAllocator_Alloc_Proxy(IAMDevMemoryAllocator *This,BYTE **ppBuffer,DWORD *pdwcbBuffer);
  void __RPC_STUB IAMDevMemoryAllocator_Alloc_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMDevMemoryAllocator_Free_Proxy(IAMDevMemoryAllocator *This,BYTE *pBuffer);
  void __RPC_STUB IAMDevMemoryAllocator_Free_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMDevMemoryAllocator_GetDevMemoryObject_Proxy(IAMDevMemoryAllocator *This,IUnknown **ppUnkInnner,IUnknown *pUnkOuter);
  void __RPC_STUB IAMDevMemoryAllocator_GetDevMemoryObject_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IAMDevMemoryAllocator *PAMDEVMEMORYALLOCATOR;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0344_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0344_v0_0_s_ifspec;
#ifndef __IAMDevMemoryControl_INTERFACE_DEFINED__
#define __IAMDevMemoryControl_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMDevMemoryControl;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMDevMemoryControl : public IUnknown {
  public:
    virtual HRESULT WINAPI QueryWriteSync(void) = 0;
    virtual HRESULT WINAPI WriteSync(void) = 0;
    virtual HRESULT WINAPI GetDevId(DWORD *pdwDevId) = 0;
  };
#else
  typedef struct IAMDevMemoryControlVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMDevMemoryControl *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMDevMemoryControl *This);
      ULONG (WINAPI *Release)(IAMDevMemoryControl *This);
      HRESULT (WINAPI *QueryWriteSync)(IAMDevMemoryControl *This);
      HRESULT (WINAPI *WriteSync)(IAMDevMemoryControl *This);
      HRESULT (WINAPI *GetDevId)(IAMDevMemoryControl *This,DWORD *pdwDevId);
    END_INTERFACE
  } IAMDevMemoryControlVtbl;
  struct IAMDevMemoryControl {
    CONST_VTBL struct IAMDevMemoryControlVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMDevMemoryControl_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMDevMemoryControl_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMDevMemoryControl_Release(This) (This)->lpVtbl->Release(This)
#define IAMDevMemoryControl_QueryWriteSync(This) (This)->lpVtbl->QueryWriteSync(This)
#define IAMDevMemoryControl_WriteSync(This) (This)->lpVtbl->WriteSync(This)
#define IAMDevMemoryControl_GetDevId(This,pdwDevId) (This)->lpVtbl->GetDevId(This,pdwDevId)
#endif
#endif
  HRESULT WINAPI IAMDevMemoryControl_QueryWriteSync_Proxy(IAMDevMemoryControl *This);
  void __RPC_STUB IAMDevMemoryControl_QueryWriteSync_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMDevMemoryControl_WriteSync_Proxy(IAMDevMemoryControl *This);
  void __RPC_STUB IAMDevMemoryControl_WriteSync_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMDevMemoryControl_GetDevId_Proxy(IAMDevMemoryControl *This,DWORD *pdwDevId);
  void __RPC_STUB IAMDevMemoryControl_GetDevId_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IAMDevMemoryControl *PAMDEVMEMORYCONTROL;

  enum _AMSTREAMSELECTINFOFLAGS {
    AMSTREAMSELECTINFO_ENABLED = 0x1,AMSTREAMSELECTINFO_EXCLUSIVE = 0x2
  };

  enum _AMSTREAMSELECTENABLEFLAGS {
    AMSTREAMSELECTENABLE_ENABLE = 0x1,AMSTREAMSELECTENABLE_ENABLEALL = 0x2
  };

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0345_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0345_v0_0_s_ifspec;
#ifndef __IAMStreamSelect_INTERFACE_DEFINED__
#define __IAMStreamSelect_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMStreamSelect;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMStreamSelect : public IUnknown {
  public:
    virtual HRESULT WINAPI Count(DWORD *pcStreams) = 0;
    virtual HRESULT WINAPI Info(long lIndex,AM_MEDIA_TYPE **ppmt,DWORD *pdwFlags,LCID *plcid,DWORD *pdwGroup,WCHAR **ppszName,IUnknown **ppObject,IUnknown **ppUnk) = 0;
    virtual HRESULT WINAPI Enable(long lIndex,DWORD dwFlags) = 0;
  };
#else
  typedef struct IAMStreamSelectVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMStreamSelect *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMStreamSelect *This);
      ULONG (WINAPI *Release)(IAMStreamSelect *This);
      HRESULT (WINAPI *Count)(IAMStreamSelect *This,DWORD *pcStreams);
      HRESULT (WINAPI *Info)(IAMStreamSelect *This,long lIndex,AM_MEDIA_TYPE **ppmt,DWORD *pdwFlags,LCID *plcid,DWORD *pdwGroup,WCHAR **ppszName,IUnknown **ppObject,IUnknown **ppUnk);
      HRESULT (WINAPI *Enable)(IAMStreamSelect *This,long lIndex,DWORD dwFlags);
    END_INTERFACE
  } IAMStreamSelectVtbl;
  struct IAMStreamSelect {
    CONST_VTBL struct IAMStreamSelectVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMStreamSelect_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMStreamSelect_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMStreamSelect_Release(This) (This)->lpVtbl->Release(This)
#define IAMStreamSelect_Count(This,pcStreams) (This)->lpVtbl->Count(This,pcStreams)
#define IAMStreamSelect_Info(This,lIndex,ppmt,pdwFlags,plcid,pdwGroup,ppszName,ppObject,ppUnk) (This)->lpVtbl->Info(This,lIndex,ppmt,pdwFlags,plcid,pdwGroup,ppszName,ppObject,ppUnk)
#define IAMStreamSelect_Enable(This,lIndex,dwFlags) (This)->lpVtbl->Enable(This,lIndex,dwFlags)
#endif
#endif
  HRESULT WINAPI IAMStreamSelect_Count_Proxy(IAMStreamSelect *This,DWORD *pcStreams);
  void __RPC_STUB IAMStreamSelect_Count_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMStreamSelect_Info_Proxy(IAMStreamSelect *This,long lIndex,AM_MEDIA_TYPE **ppmt,DWORD *pdwFlags,LCID *plcid,DWORD *pdwGroup,WCHAR **ppszName,IUnknown **ppObject,IUnknown **ppUnk);
  void __RPC_STUB IAMStreamSelect_Info_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMStreamSelect_Enable_Proxy(IAMStreamSelect *This,long lIndex,DWORD dwFlags);
  void __RPC_STUB IAMStreamSelect_Enable_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef IAMStreamSelect *PAMSTREAMSELECT;

  enum _AMRESCTL_RESERVEFLAGS {
    AMRESCTL_RESERVEFLAGS_RESERVE = 0,AMRESCTL_RESERVEFLAGS_UNRESERVE = 0x1
  };

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0346_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0346_v0_0_s_ifspec;
#ifndef __IAMResourceControl_INTERFACE_DEFINED__
#define __IAMResourceControl_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMResourceControl;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMResourceControl : public IUnknown {
  public:
    virtual HRESULT WINAPI Reserve(DWORD dwFlags,PVOID pvReserved) = 0;
  };
#else
  typedef struct IAMResourceControlVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMResourceControl *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMResourceControl *This);
      ULONG (WINAPI *Release)(IAMResourceControl *This);
      HRESULT (WINAPI *Reserve)(IAMResourceControl *This,DWORD dwFlags,PVOID pvReserved);
    END_INTERFACE
  } IAMResourceControlVtbl;
  struct IAMResourceControl {
    CONST_VTBL struct IAMResourceControlVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMResourceControl_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMResourceControl_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMResourceControl_Release(This) (This)->lpVtbl->Release(This)
#define IAMResourceControl_Reserve(This,dwFlags,pvReserved) (This)->lpVtbl->Reserve(This,dwFlags,pvReserved)
#endif
#endif
  HRESULT WINAPI IAMResourceControl_Reserve_Proxy(IAMResourceControl *This,DWORD dwFlags,PVOID pvReserved);
  void __RPC_STUB IAMResourceControl_Reserve_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IAMClockAdjust_INTERFACE_DEFINED__
#define __IAMClockAdjust_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMClockAdjust;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMClockAdjust : public IUnknown {
  public:
    virtual HRESULT WINAPI SetClockDelta(REFERENCE_TIME rtDelta) = 0;
  };
#else
  typedef struct IAMClockAdjustVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMClockAdjust *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMClockAdjust *This);
      ULONG (WINAPI *Release)(IAMClockAdjust *This);
      HRESULT (WINAPI *SetClockDelta)(IAMClockAdjust *This,REFERENCE_TIME rtDelta);
    END_INTERFACE
  } IAMClockAdjustVtbl;
  struct IAMClockAdjust {
    CONST_VTBL struct IAMClockAdjustVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMClockAdjust_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMClockAdjust_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMClockAdjust_Release(This) (This)->lpVtbl->Release(This)
#define IAMClockAdjust_SetClockDelta(This,rtDelta) (This)->lpVtbl->SetClockDelta(This,rtDelta)
#endif
#endif
  HRESULT WINAPI IAMClockAdjust_SetClockDelta_Proxy(IAMClockAdjust *This,REFERENCE_TIME rtDelta);
  void __RPC_STUB IAMClockAdjust_SetClockDelta_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  enum _AM_FILTER_MISC_FLAGS {
    AM_FILTER_MISC_FLAGS_IS_RENDERER = 0x1,AM_FILTER_MISC_FLAGS_IS_SOURCE = 0x2
  };

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0348_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0348_v0_0_s_ifspec;
#ifndef __IAMFilterMiscFlags_INTERFACE_DEFINED__
#define __IAMFilterMiscFlags_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMFilterMiscFlags;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMFilterMiscFlags : public IUnknown {
  public:
    virtual ULONG WINAPI GetMiscFlags(void) = 0;
  };
#else
  typedef struct IAMFilterMiscFlagsVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMFilterMiscFlags *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMFilterMiscFlags *This);
      ULONG (WINAPI *Release)(IAMFilterMiscFlags *This);
      ULONG (WINAPI *GetMiscFlags)(IAMFilterMiscFlags *This);
    END_INTERFACE
  } IAMFilterMiscFlagsVtbl;
  struct IAMFilterMiscFlags {
    CONST_VTBL struct IAMFilterMiscFlagsVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMFilterMiscFlags_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMFilterMiscFlags_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMFilterMiscFlags_Release(This) (This)->lpVtbl->Release(This)
#define IAMFilterMiscFlags_GetMiscFlags(This) (This)->lpVtbl->GetMiscFlags(This)
#endif
#endif
  ULONG WINAPI IAMFilterMiscFlags_GetMiscFlags_Proxy(IAMFilterMiscFlags *This);
  void __RPC_STUB IAMFilterMiscFlags_GetMiscFlags_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IDrawVideoImage_INTERFACE_DEFINED__
#define __IDrawVideoImage_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IDrawVideoImage;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IDrawVideoImage : public IUnknown {
  public:
    virtual HRESULT WINAPI DrawVideoImageBegin(void) = 0;
    virtual HRESULT WINAPI DrawVideoImageEnd(void) = 0;
    virtual HRESULT WINAPI DrawVideoImageDraw(HDC hdc,LPRECT lprcSrc,LPRECT lprcDst) = 0;
  };
#else
  typedef struct IDrawVideoImageVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IDrawVideoImage *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IDrawVideoImage *This);
      ULONG (WINAPI *Release)(IDrawVideoImage *This);
      HRESULT (WINAPI *DrawVideoImageBegin)(IDrawVideoImage *This);
      HRESULT (WINAPI *DrawVideoImageEnd)(IDrawVideoImage *This);
      HRESULT (WINAPI *DrawVideoImageDraw)(IDrawVideoImage *This,HDC hdc,LPRECT lprcSrc,LPRECT lprcDst);
    END_INTERFACE
  } IDrawVideoImageVtbl;
  struct IDrawVideoImage {
    CONST_VTBL struct IDrawVideoImageVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IDrawVideoImage_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDrawVideoImage_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDrawVideoImage_Release(This) (This)->lpVtbl->Release(This)
#define IDrawVideoImage_DrawVideoImageBegin(This) (This)->lpVtbl->DrawVideoImageBegin(This)
#define IDrawVideoImage_DrawVideoImageEnd(This) (This)->lpVtbl->DrawVideoImageEnd(This)
#define IDrawVideoImage_DrawVideoImageDraw(This,hdc,lprcSrc,lprcDst) (This)->lpVtbl->DrawVideoImageDraw(This,hdc,lprcSrc,lprcDst)
#endif
#endif
  HRESULT WINAPI IDrawVideoImage_DrawVideoImageBegin_Proxy(IDrawVideoImage *This);
  void __RPC_STUB IDrawVideoImage_DrawVideoImageBegin_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDrawVideoImage_DrawVideoImageEnd_Proxy(IDrawVideoImage *This);
  void __RPC_STUB IDrawVideoImage_DrawVideoImageEnd_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDrawVideoImage_DrawVideoImageDraw_Proxy(IDrawVideoImage *This,HDC hdc,LPRECT lprcSrc,LPRECT lprcDst);
  void __RPC_STUB IDrawVideoImage_DrawVideoImageDraw_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IDecimateVideoImage_INTERFACE_DEFINED__
#define __IDecimateVideoImage_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IDecimateVideoImage;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IDecimateVideoImage : public IUnknown {
  public:
    virtual HRESULT WINAPI SetDecimationImageSize(long lWidth,long lHeight) = 0;
    virtual HRESULT WINAPI ResetDecimationImageSize(void) = 0;
  };
#else
  typedef struct IDecimateVideoImageVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IDecimateVideoImage *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IDecimateVideoImage *This);
      ULONG (WINAPI *Release)(IDecimateVideoImage *This);
      HRESULT (WINAPI *SetDecimationImageSize)(IDecimateVideoImage *This,long lWidth,long lHeight);
      HRESULT (WINAPI *ResetDecimationImageSize)(IDecimateVideoImage *This);
    END_INTERFACE
  } IDecimateVideoImageVtbl;
  struct IDecimateVideoImage {
    CONST_VTBL struct IDecimateVideoImageVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IDecimateVideoImage_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDecimateVideoImage_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDecimateVideoImage_Release(This) (This)->lpVtbl->Release(This)
#define IDecimateVideoImage_SetDecimationImageSize(This,lWidth,lHeight) (This)->lpVtbl->SetDecimationImageSize(This,lWidth,lHeight)
#define IDecimateVideoImage_ResetDecimationImageSize(This) (This)->lpVtbl->ResetDecimationImageSize(This)
#endif
#endif
  HRESULT WINAPI IDecimateVideoImage_SetDecimationImageSize_Proxy(IDecimateVideoImage *This,long lWidth,long lHeight);
  void __RPC_STUB IDecimateVideoImage_SetDecimationImageSize_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDecimateVideoImage_ResetDecimationImageSize_Proxy(IDecimateVideoImage *This);
  void __RPC_STUB IDecimateVideoImage_ResetDecimationImageSize_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef enum _DECIMATION_USAGE {
    DECIMATION_LEGACY = 0,
    DECIMATION_USE_DECODER_ONLY,DECIMATION_USE_VIDEOPORT_ONLY,DECIMATION_USE_OVERLAY_ONLY,
    DECIMATION_DEFAULT
  } DECIMATION_USAGE;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0351_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0351_v0_0_s_ifspec;
#ifndef __IAMVideoDecimationProperties_INTERFACE_DEFINED__
#define __IAMVideoDecimationProperties_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMVideoDecimationProperties;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMVideoDecimationProperties : public IUnknown {
  public:
    virtual HRESULT WINAPI QueryDecimationUsage(DECIMATION_USAGE *lpUsage) = 0;
    virtual HRESULT WINAPI SetDecimationUsage(DECIMATION_USAGE Usage) = 0;
  };
#else
  typedef struct IAMVideoDecimationPropertiesVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMVideoDecimationProperties *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMVideoDecimationProperties *This);
      ULONG (WINAPI *Release)(IAMVideoDecimationProperties *This);
      HRESULT (WINAPI *QueryDecimationUsage)(IAMVideoDecimationProperties *This,DECIMATION_USAGE *lpUsage);
      HRESULT (WINAPI *SetDecimationUsage)(IAMVideoDecimationProperties *This,DECIMATION_USAGE Usage);
    END_INTERFACE
  } IAMVideoDecimationPropertiesVtbl;
  struct IAMVideoDecimationProperties {
    CONST_VTBL struct IAMVideoDecimationPropertiesVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMVideoDecimationProperties_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMVideoDecimationProperties_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMVideoDecimationProperties_Release(This) (This)->lpVtbl->Release(This)
#define IAMVideoDecimationProperties_QueryDecimationUsage(This,lpUsage) (This)->lpVtbl->QueryDecimationUsage(This,lpUsage)
#define IAMVideoDecimationProperties_SetDecimationUsage(This,Usage) (This)->lpVtbl->SetDecimationUsage(This,Usage)
#endif
#endif
  HRESULT WINAPI IAMVideoDecimationProperties_QueryDecimationUsage_Proxy(IAMVideoDecimationProperties *This,DECIMATION_USAGE *lpUsage);
  void __RPC_STUB IAMVideoDecimationProperties_QueryDecimationUsage_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMVideoDecimationProperties_SetDecimationUsage_Proxy(IAMVideoDecimationProperties *This,DECIMATION_USAGE Usage);
  void __RPC_STUB IAMVideoDecimationProperties_SetDecimationUsage_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IVideoFrameStep_INTERFACE_DEFINED__
#define __IVideoFrameStep_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IVideoFrameStep;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IVideoFrameStep : public IUnknown {
  public:
    virtual HRESULT WINAPI Step(DWORD dwFrames,IUnknown *pStepObject) = 0;
    virtual HRESULT WINAPI CanStep(long bMultiple,IUnknown *pStepObject) = 0;
    virtual HRESULT WINAPI CancelStep(void) = 0;
  };
#else
  typedef struct IVideoFrameStepVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IVideoFrameStep *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IVideoFrameStep *This);
      ULONG (WINAPI *Release)(IVideoFrameStep *This);
      HRESULT (WINAPI *Step)(IVideoFrameStep *This,DWORD dwFrames,IUnknown *pStepObject);
      HRESULT (WINAPI *CanStep)(IVideoFrameStep *This,long bMultiple,IUnknown *pStepObject);
      HRESULT (WINAPI *CancelStep)(IVideoFrameStep *This);
    END_INTERFACE
  } IVideoFrameStepVtbl;
  struct IVideoFrameStep {
    CONST_VTBL struct IVideoFrameStepVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IVideoFrameStep_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IVideoFrameStep_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IVideoFrameStep_Release(This) (This)->lpVtbl->Release(This)
#define IVideoFrameStep_Step(This,dwFrames,pStepObject) (This)->lpVtbl->Step(This,dwFrames,pStepObject)
#define IVideoFrameStep_CanStep(This,bMultiple,pStepObject) (This)->lpVtbl->CanStep(This,bMultiple,pStepObject)
#define IVideoFrameStep_CancelStep(This) (This)->lpVtbl->CancelStep(This)
#endif
#endif
  HRESULT WINAPI IVideoFrameStep_Step_Proxy(IVideoFrameStep *This,DWORD dwFrames,IUnknown *pStepObject);
  void __RPC_STUB IVideoFrameStep_Step_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoFrameStep_CanStep_Proxy(IVideoFrameStep *This,long bMultiple,IUnknown *pStepObject);
  void __RPC_STUB IVideoFrameStep_CanStep_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoFrameStep_CancelStep_Proxy(IVideoFrameStep *This);
  void __RPC_STUB IVideoFrameStep_CancelStep_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  enum _AM_PUSHSOURCE_FLAGS {
    AM_PUSHSOURCECAPS_INTERNAL_RM = 0x1,AM_PUSHSOURCECAPS_NOT_LIVE = 0x2,AM_PUSHSOURCECAPS_PRIVATE_CLOCK = 0x4,
    AM_PUSHSOURCEREQS_USE_STREAM_CLOCK = 0x10000,AM_PUSHSOURCEREQS_USE_CLOCK_CHAIN = 0x20000
  };

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0353_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0353_v0_0_s_ifspec;
#ifndef __IAMLatency_INTERFACE_DEFINED__
#define __IAMLatency_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMLatency;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMLatency : public IUnknown {
  public:
    virtual HRESULT WINAPI GetLatency(REFERENCE_TIME *prtLatency) = 0;
  };
#else
  typedef struct IAMLatencyVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMLatency *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMLatency *This);
      ULONG (WINAPI *Release)(IAMLatency *This);
      HRESULT (WINAPI *GetLatency)(IAMLatency *This,REFERENCE_TIME *prtLatency);
    END_INTERFACE
  } IAMLatencyVtbl;
  struct IAMLatency {
    CONST_VTBL struct IAMLatencyVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMLatency_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMLatency_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMLatency_Release(This) (This)->lpVtbl->Release(This)
#define IAMLatency_GetLatency(This,prtLatency) (This)->lpVtbl->GetLatency(This,prtLatency)
#endif
#endif
  HRESULT WINAPI IAMLatency_GetLatency_Proxy(IAMLatency *This,REFERENCE_TIME *prtLatency);
  void __RPC_STUB IAMLatency_GetLatency_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IAMPushSource_INTERFACE_DEFINED__
#define __IAMPushSource_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMPushSource;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMPushSource : public IAMLatency {
  public:
    virtual HRESULT WINAPI GetPushSourceFlags(ULONG *pFlags) = 0;
    virtual HRESULT WINAPI SetPushSourceFlags(ULONG Flags) = 0;
    virtual HRESULT WINAPI SetStreamOffset(REFERENCE_TIME rtOffset) = 0;
    virtual HRESULT WINAPI GetStreamOffset(REFERENCE_TIME *prtOffset) = 0;
    virtual HRESULT WINAPI GetMaxStreamOffset(REFERENCE_TIME *prtMaxOffset) = 0;
    virtual HRESULT WINAPI SetMaxStreamOffset(REFERENCE_TIME rtMaxOffset) = 0;
  };
#else
  typedef struct IAMPushSourceVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMPushSource *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMPushSource *This);
      ULONG (WINAPI *Release)(IAMPushSource *This);
      HRESULT (WINAPI *GetLatency)(IAMPushSource *This,REFERENCE_TIME *prtLatency);
      HRESULT (WINAPI *GetPushSourceFlags)(IAMPushSource *This,ULONG *pFlags);
      HRESULT (WINAPI *SetPushSourceFlags)(IAMPushSource *This,ULONG Flags);
      HRESULT (WINAPI *SetStreamOffset)(IAMPushSource *This,REFERENCE_TIME rtOffset);
      HRESULT (WINAPI *GetStreamOffset)(IAMPushSource *This,REFERENCE_TIME *prtOffset);
      HRESULT (WINAPI *GetMaxStreamOffset)(IAMPushSource *This,REFERENCE_TIME *prtMaxOffset);
      HRESULT (WINAPI *SetMaxStreamOffset)(IAMPushSource *This,REFERENCE_TIME rtMaxOffset);
    END_INTERFACE
  } IAMPushSourceVtbl;
  struct IAMPushSource {
    CONST_VTBL struct IAMPushSourceVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMPushSource_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMPushSource_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMPushSource_Release(This) (This)->lpVtbl->Release(This)
#define IAMPushSource_GetLatency(This,prtLatency) (This)->lpVtbl->GetLatency(This,prtLatency)
#define IAMPushSource_GetPushSourceFlags(This,pFlags) (This)->lpVtbl->GetPushSourceFlags(This,pFlags)
#define IAMPushSource_SetPushSourceFlags(This,Flags) (This)->lpVtbl->SetPushSourceFlags(This,Flags)
#define IAMPushSource_SetStreamOffset(This,rtOffset) (This)->lpVtbl->SetStreamOffset(This,rtOffset)
#define IAMPushSource_GetStreamOffset(This,prtOffset) (This)->lpVtbl->GetStreamOffset(This,prtOffset)
#define IAMPushSource_GetMaxStreamOffset(This,prtMaxOffset) (This)->lpVtbl->GetMaxStreamOffset(This,prtMaxOffset)
#define IAMPushSource_SetMaxStreamOffset(This,rtMaxOffset) (This)->lpVtbl->SetMaxStreamOffset(This,rtMaxOffset)
#endif
#endif
  HRESULT WINAPI IAMPushSource_GetPushSourceFlags_Proxy(IAMPushSource *This,ULONG *pFlags);
  void __RPC_STUB IAMPushSource_GetPushSourceFlags_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMPushSource_SetPushSourceFlags_Proxy(IAMPushSource *This,ULONG Flags);
  void __RPC_STUB IAMPushSource_SetPushSourceFlags_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMPushSource_SetStreamOffset_Proxy(IAMPushSource *This,REFERENCE_TIME rtOffset);
  void __RPC_STUB IAMPushSource_SetStreamOffset_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMPushSource_GetStreamOffset_Proxy(IAMPushSource *This,REFERENCE_TIME *prtOffset);
  void __RPC_STUB IAMPushSource_GetStreamOffset_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMPushSource_GetMaxStreamOffset_Proxy(IAMPushSource *This,REFERENCE_TIME *prtMaxOffset);
  void __RPC_STUB IAMPushSource_GetMaxStreamOffset_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMPushSource_SetMaxStreamOffset_Proxy(IAMPushSource *This,REFERENCE_TIME rtMaxOffset);
  void __RPC_STUB IAMPushSource_SetMaxStreamOffset_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IAMDeviceRemoval_INTERFACE_DEFINED__
#define __IAMDeviceRemoval_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMDeviceRemoval;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMDeviceRemoval : public IUnknown {
  public:
    virtual HRESULT WINAPI DeviceInfo(CLSID *pclsidInterfaceClass,WCHAR **pwszSymbolicLink) = 0;
    virtual HRESULT WINAPI Reassociate(void) = 0;
    virtual HRESULT WINAPI Disassociate(void) = 0;
  };
#else
  typedef struct IAMDeviceRemovalVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMDeviceRemoval *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMDeviceRemoval *This);
      ULONG (WINAPI *Release)(IAMDeviceRemoval *This);
      HRESULT (WINAPI *DeviceInfo)(IAMDeviceRemoval *This,CLSID *pclsidInterfaceClass,WCHAR **pwszSymbolicLink);
      HRESULT (WINAPI *Reassociate)(IAMDeviceRemoval *This);
      HRESULT (WINAPI *Disassociate)(IAMDeviceRemoval *This);
    END_INTERFACE
  } IAMDeviceRemovalVtbl;
  struct IAMDeviceRemoval {
    CONST_VTBL struct IAMDeviceRemovalVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMDeviceRemoval_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMDeviceRemoval_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMDeviceRemoval_Release(This) (This)->lpVtbl->Release(This)
#define IAMDeviceRemoval_DeviceInfo(This,pclsidInterfaceClass,pwszSymbolicLink) (This)->lpVtbl->DeviceInfo(This,pclsidInterfaceClass,pwszSymbolicLink)
#define IAMDeviceRemoval_Reassociate(This) (This)->lpVtbl->Reassociate(This)
#define IAMDeviceRemoval_Disassociate(This) (This)->lpVtbl->Disassociate(This)
#endif
#endif
  HRESULT WINAPI IAMDeviceRemoval_DeviceInfo_Proxy(IAMDeviceRemoval *This,CLSID *pclsidInterfaceClass,WCHAR **pwszSymbolicLink);
  void __RPC_STUB IAMDeviceRemoval_DeviceInfo_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMDeviceRemoval_Reassociate_Proxy(IAMDeviceRemoval *This);
  void __RPC_STUB IAMDeviceRemoval_Reassociate_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMDeviceRemoval_Disassociate_Proxy(IAMDeviceRemoval *This);
  void __RPC_STUB IAMDeviceRemoval_Disassociate_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef struct __MIDL___MIDL_itf_strmif_0355_0001 {
    DWORD dwDVAAuxSrc;
    DWORD dwDVAAuxCtl;
    DWORD dwDVAAuxSrc1;
    DWORD dwDVAAuxCtl1;
    DWORD dwDVVAuxSrc;
    DWORD dwDVVAuxCtl;
    DWORD dwDVReserved[2];
  } DVINFO;

  typedef struct __MIDL___MIDL_itf_strmif_0355_0001 *PDVINFO;

  enum _DVENCODERRESOLUTION {
    DVENCODERRESOLUTION_720x480 = 2012,DVENCODERRESOLUTION_360x240 = 2013,DVENCODERRESOLUTION_180x120 = 2014,DVENCODERRESOLUTION_88x60 = 2015
  };

  enum _DVENCODERVIDEOFORMAT {
    DVENCODERVIDEOFORMAT_NTSC = 2000,DVENCODERVIDEOFORMAT_PAL = 2001
  };

  enum _DVENCODERFORMAT {
    DVENCODERFORMAT_DVSD = 2007,DVENCODERFORMAT_DVHD = 2008,DVENCODERFORMAT_DVSL = 2009
  };

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0355_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0355_v0_0_s_ifspec;
#ifndef __IDVEnc_INTERFACE_DEFINED__
#define __IDVEnc_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IDVEnc;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IDVEnc : public IUnknown {
  public:
    virtual HRESULT WINAPI get_IFormatResolution(int *VideoFormat,int *DVFormat,int *Resolution,BYTE fDVInfo,DVINFO *sDVInfo) = 0;
    virtual HRESULT WINAPI put_IFormatResolution(int VideoFormat,int DVFormat,int Resolution,BYTE fDVInfo,DVINFO *sDVInfo) = 0;
  };
#else
  typedef struct IDVEncVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IDVEnc *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IDVEnc *This);
      ULONG (WINAPI *Release)(IDVEnc *This);
      HRESULT (WINAPI *get_IFormatResolution)(IDVEnc *This,int *VideoFormat,int *DVFormat,int *Resolution,BYTE fDVInfo,DVINFO *sDVInfo);
      HRESULT (WINAPI *put_IFormatResolution)(IDVEnc *This,int VideoFormat,int DVFormat,int Resolution,BYTE fDVInfo,DVINFO *sDVInfo);
    END_INTERFACE
  } IDVEncVtbl;
  struct IDVEnc {
    CONST_VTBL struct IDVEncVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IDVEnc_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDVEnc_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDVEnc_Release(This) (This)->lpVtbl->Release(This)
#define IDVEnc_get_IFormatResolution(This,VideoFormat,DVFormat,Resolution,fDVInfo,sDVInfo) (This)->lpVtbl->get_IFormatResolution(This,VideoFormat,DVFormat,Resolution,fDVInfo,sDVInfo)
#define IDVEnc_put_IFormatResolution(This,VideoFormat,DVFormat,Resolution,fDVInfo,sDVInfo) (This)->lpVtbl->put_IFormatResolution(This,VideoFormat,DVFormat,Resolution,fDVInfo,sDVInfo)
#endif
#endif
  HRESULT WINAPI IDVEnc_get_IFormatResolution_Proxy(IDVEnc *This,int *VideoFormat,int *DVFormat,int *Resolution,BYTE fDVInfo,DVINFO *sDVInfo);
  void __RPC_STUB IDVEnc_get_IFormatResolution_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDVEnc_put_IFormatResolution_Proxy(IDVEnc *This,int VideoFormat,int DVFormat,int Resolution,BYTE fDVInfo,DVINFO *sDVInfo);
  void __RPC_STUB IDVEnc_put_IFormatResolution_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  enum _DVDECODERRESOLUTION {
    DVDECODERRESOLUTION_720x480 = 1000,DVDECODERRESOLUTION_360x240 = 1001,DVDECODERRESOLUTION_180x120 = 1002,DVDECODERRESOLUTION_88x60 = 1003
  };

  enum _DVRESOLUTION {
    DVRESOLUTION_FULL = 1000,DVRESOLUTION_HALF = 1001,DVRESOLUTION_QUARTER = 1002,DVRESOLUTION_DC = 1003
  };

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0356_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0356_v0_0_s_ifspec;
#ifndef __IIPDVDec_INTERFACE_DEFINED__
#define __IIPDVDec_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IIPDVDec;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IIPDVDec : public IUnknown {
  public:
    virtual HRESULT WINAPI get_IPDisplay(int *displayPix) = 0;
    virtual HRESULT WINAPI put_IPDisplay(int displayPix) = 0;
  };
#else
  typedef struct IIPDVDecVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IIPDVDec *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IIPDVDec *This);
      ULONG (WINAPI *Release)(IIPDVDec *This);
      HRESULT (WINAPI *get_IPDisplay)(IIPDVDec *This,int *displayPix);
      HRESULT (WINAPI *put_IPDisplay)(IIPDVDec *This,int displayPix);
    END_INTERFACE
  } IIPDVDecVtbl;
  struct IIPDVDec {
    CONST_VTBL struct IIPDVDecVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IIPDVDec_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IIPDVDec_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IIPDVDec_Release(This) (This)->lpVtbl->Release(This)
#define IIPDVDec_get_IPDisplay(This,displayPix) (This)->lpVtbl->get_IPDisplay(This,displayPix)
#define IIPDVDec_put_IPDisplay(This,displayPix) (This)->lpVtbl->put_IPDisplay(This,displayPix)
#endif
#endif
  HRESULT WINAPI IIPDVDec_get_IPDisplay_Proxy(IIPDVDec *This,int *displayPix);
  void __RPC_STUB IIPDVDec_get_IPDisplay_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IIPDVDec_put_IPDisplay_Proxy(IIPDVDec *This,int displayPix);
  void __RPC_STUB IIPDVDec_put_IPDisplay_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IDVRGB219_INTERFACE_DEFINED__
#define __IDVRGB219_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IDVRGB219;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IDVRGB219 : public IUnknown {
  public:
    virtual HRESULT WINAPI SetRGB219(WINBOOL bState) = 0;
  };
#else
  typedef struct IDVRGB219Vtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IDVRGB219 *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IDVRGB219 *This);
      ULONG (WINAPI *Release)(IDVRGB219 *This);
      HRESULT (WINAPI *SetRGB219)(IDVRGB219 *This,WINBOOL bState);
    END_INTERFACE
  } IDVRGB219Vtbl;
  struct IDVRGB219 {
    CONST_VTBL struct IDVRGB219Vtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IDVRGB219_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDVRGB219_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDVRGB219_Release(This) (This)->lpVtbl->Release(This)
#define IDVRGB219_SetRGB219(This,bState) (This)->lpVtbl->SetRGB219(This,bState)
#endif
#endif
  HRESULT WINAPI IDVRGB219_SetRGB219_Proxy(IDVRGB219 *This,WINBOOL bState);
  void __RPC_STUB IDVRGB219_SetRGB219_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IDVSplitter_INTERFACE_DEFINED__
#define __IDVSplitter_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IDVSplitter;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IDVSplitter : public IUnknown {
  public:
    virtual HRESULT WINAPI DiscardAlternateVideoFrames(int nDiscard) = 0;
  };
#else
  typedef struct IDVSplitterVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IDVSplitter *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IDVSplitter *This);
      ULONG (WINAPI *Release)(IDVSplitter *This);
      HRESULT (WINAPI *DiscardAlternateVideoFrames)(IDVSplitter *This,int nDiscard);
    END_INTERFACE
  } IDVSplitterVtbl;
  struct IDVSplitter {
    CONST_VTBL struct IDVSplitterVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IDVSplitter_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDVSplitter_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDVSplitter_Release(This) (This)->lpVtbl->Release(This)
#define IDVSplitter_DiscardAlternateVideoFrames(This,nDiscard) (This)->lpVtbl->DiscardAlternateVideoFrames(This,nDiscard)
#endif
#endif
  HRESULT WINAPI IDVSplitter_DiscardAlternateVideoFrames_Proxy(IDVSplitter *This,int nDiscard);
  void __RPC_STUB IDVSplitter_DiscardAlternateVideoFrames_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  enum _AM_AUDIO_RENDERER_STAT_PARAM {
    AM_AUDREND_STAT_PARAM_BREAK_COUNT = 1,
    AM_AUDREND_STAT_PARAM_SLAVE_MODE,AM_AUDREND_STAT_PARAM_SILENCE_DUR,
    AM_AUDREND_STAT_PARAM_LAST_BUFFER_DUR,AM_AUDREND_STAT_PARAM_DISCONTINUITIES,
    AM_AUDREND_STAT_PARAM_SLAVE_RATE,AM_AUDREND_STAT_PARAM_SLAVE_DROPWRITE_DUR,
    AM_AUDREND_STAT_PARAM_SLAVE_HIGHLOWERROR,AM_AUDREND_STAT_PARAM_SLAVE_LASTHIGHLOWERROR,
    AM_AUDREND_STAT_PARAM_SLAVE_ACCUMERROR,AM_AUDREND_STAT_PARAM_BUFFERFULLNESS,
    AM_AUDREND_STAT_PARAM_JITTER
  };

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0359_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0359_v0_0_s_ifspec;
#ifndef __IAMAudioRendererStats_INTERFACE_DEFINED__
#define __IAMAudioRendererStats_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMAudioRendererStats;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMAudioRendererStats : public IUnknown {
  public:
    virtual HRESULT WINAPI GetStatParam(DWORD dwParam,DWORD *pdwParam1,DWORD *pdwParam2) = 0;
  };
#else
  typedef struct IAMAudioRendererStatsVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMAudioRendererStats *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMAudioRendererStats *This);
      ULONG (WINAPI *Release)(IAMAudioRendererStats *This);
      HRESULT (WINAPI *GetStatParam)(IAMAudioRendererStats *This,DWORD dwParam,DWORD *pdwParam1,DWORD *pdwParam2);
    END_INTERFACE
  } IAMAudioRendererStatsVtbl;
  struct IAMAudioRendererStats {
    CONST_VTBL struct IAMAudioRendererStatsVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMAudioRendererStats_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMAudioRendererStats_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMAudioRendererStats_Release(This) (This)->lpVtbl->Release(This)
#define IAMAudioRendererStats_GetStatParam(This,dwParam,pdwParam1,pdwParam2) (This)->lpVtbl->GetStatParam(This,dwParam,pdwParam1,pdwParam2)
#endif
#endif
  HRESULT WINAPI IAMAudioRendererStats_GetStatParam_Proxy(IAMAudioRendererStats *This,DWORD dwParam,DWORD *pdwParam1,DWORD *pdwParam2);
  void __RPC_STUB IAMAudioRendererStats_GetStatParam_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  enum _AM_INTF_SEARCH_FLAGS {
    AM_INTF_SEARCH_INPUT_PIN = 0x1,AM_INTF_SEARCH_OUTPUT_PIN = 0x2,AM_INTF_SEARCH_FILTER = 0x4
  };

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0361_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0361_v0_0_s_ifspec;
#ifndef __IAMGraphStreams_INTERFACE_DEFINED__
#define __IAMGraphStreams_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMGraphStreams;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMGraphStreams : public IUnknown {
  public:
    virtual HRESULT WINAPI FindUpstreamInterface(IPin *pPin,REFIID riid,void **ppvInterface,DWORD dwFlags) = 0;
    virtual HRESULT WINAPI SyncUsingStreamOffset(WINBOOL bUseStreamOffset) = 0;
    virtual HRESULT WINAPI SetMaxGraphLatency(REFERENCE_TIME rtMaxGraphLatency) = 0;
  };
#else
  typedef struct IAMGraphStreamsVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMGraphStreams *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMGraphStreams *This);
      ULONG (WINAPI *Release)(IAMGraphStreams *This);
      HRESULT (WINAPI *FindUpstreamInterface)(IAMGraphStreams *This,IPin *pPin,REFIID riid,void **ppvInterface,DWORD dwFlags);
      HRESULT (WINAPI *SyncUsingStreamOffset)(IAMGraphStreams *This,WINBOOL bUseStreamOffset);
      HRESULT (WINAPI *SetMaxGraphLatency)(IAMGraphStreams *This,REFERENCE_TIME rtMaxGraphLatency);
    END_INTERFACE
  } IAMGraphStreamsVtbl;
  struct IAMGraphStreams {
    CONST_VTBL struct IAMGraphStreamsVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMGraphStreams_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMGraphStreams_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMGraphStreams_Release(This) (This)->lpVtbl->Release(This)
#define IAMGraphStreams_FindUpstreamInterface(This,pPin,riid,ppvInterface,dwFlags) (This)->lpVtbl->FindUpstreamInterface(This,pPin,riid,ppvInterface,dwFlags)
#define IAMGraphStreams_SyncUsingStreamOffset(This,bUseStreamOffset) (This)->lpVtbl->SyncUsingStreamOffset(This,bUseStreamOffset)
#define IAMGraphStreams_SetMaxGraphLatency(This,rtMaxGraphLatency) (This)->lpVtbl->SetMaxGraphLatency(This,rtMaxGraphLatency)
#endif
#endif
  HRESULT WINAPI IAMGraphStreams_FindUpstreamInterface_Proxy(IAMGraphStreams *This,IPin *pPin,REFIID riid,void **ppvInterface,DWORD dwFlags);
  void __RPC_STUB IAMGraphStreams_FindUpstreamInterface_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMGraphStreams_SyncUsingStreamOffset_Proxy(IAMGraphStreams *This,WINBOOL bUseStreamOffset);
  void __RPC_STUB IAMGraphStreams_SyncUsingStreamOffset_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMGraphStreams_SetMaxGraphLatency_Proxy(IAMGraphStreams *This,REFERENCE_TIME rtMaxGraphLatency);
  void __RPC_STUB IAMGraphStreams_SetMaxGraphLatency_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  enum AMOVERLAYFX {
    AMOVERFX_NOFX = 0,AMOVERFX_MIRRORLEFTRIGHT = 0x2,AMOVERFX_MIRRORUPDOWN = 0x4,AMOVERFX_DEINTERLACE = 0x8
  };

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0362_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0362_v0_0_s_ifspec;
#ifndef __IAMOverlayFX_INTERFACE_DEFINED__
#define __IAMOverlayFX_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMOverlayFX;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMOverlayFX : public IUnknown {
  public:
    virtual HRESULT WINAPI QueryOverlayFXCaps(DWORD *lpdwOverlayFXCaps) = 0;
    virtual HRESULT WINAPI SetOverlayFX(DWORD dwOverlayFX) = 0;
    virtual HRESULT WINAPI GetOverlayFX(DWORD *lpdwOverlayFX) = 0;
  };
#else
  typedef struct IAMOverlayFXVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMOverlayFX *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMOverlayFX *This);
      ULONG (WINAPI *Release)(IAMOverlayFX *This);
      HRESULT (WINAPI *QueryOverlayFXCaps)(IAMOverlayFX *This,DWORD *lpdwOverlayFXCaps);
      HRESULT (WINAPI *SetOverlayFX)(IAMOverlayFX *This,DWORD dwOverlayFX);
      HRESULT (WINAPI *GetOverlayFX)(IAMOverlayFX *This,DWORD *lpdwOverlayFX);
    END_INTERFACE
  } IAMOverlayFXVtbl;
  struct IAMOverlayFX {
    CONST_VTBL struct IAMOverlayFXVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMOverlayFX_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMOverlayFX_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMOverlayFX_Release(This) (This)->lpVtbl->Release(This)
#define IAMOverlayFX_QueryOverlayFXCaps(This,lpdwOverlayFXCaps) (This)->lpVtbl->QueryOverlayFXCaps(This,lpdwOverlayFXCaps)
#define IAMOverlayFX_SetOverlayFX(This,dwOverlayFX) (This)->lpVtbl->SetOverlayFX(This,dwOverlayFX)
#define IAMOverlayFX_GetOverlayFX(This,lpdwOverlayFX) (This)->lpVtbl->GetOverlayFX(This,lpdwOverlayFX)
#endif
#endif
  HRESULT WINAPI IAMOverlayFX_QueryOverlayFXCaps_Proxy(IAMOverlayFX *This,DWORD *lpdwOverlayFXCaps);
  void __RPC_STUB IAMOverlayFX_QueryOverlayFXCaps_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMOverlayFX_SetOverlayFX_Proxy(IAMOverlayFX *This,DWORD dwOverlayFX);
  void __RPC_STUB IAMOverlayFX_SetOverlayFX_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMOverlayFX_GetOverlayFX_Proxy(IAMOverlayFX *This,DWORD *lpdwOverlayFX);
  void __RPC_STUB IAMOverlayFX_GetOverlayFX_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IAMOpenProgress_INTERFACE_DEFINED__
#define __IAMOpenProgress_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMOpenProgress;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMOpenProgress : public IUnknown {
  public:
    virtual HRESULT WINAPI QueryProgress(LONGLONG *pllTotal,LONGLONG *pllCurrent) = 0;
    virtual HRESULT WINAPI AbortOperation(void) = 0;
  };
#else
  typedef struct IAMOpenProgressVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMOpenProgress *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMOpenProgress *This);
      ULONG (WINAPI *Release)(IAMOpenProgress *This);
      HRESULT (WINAPI *QueryProgress)(IAMOpenProgress *This,LONGLONG *pllTotal,LONGLONG *pllCurrent);
      HRESULT (WINAPI *AbortOperation)(IAMOpenProgress *This);
    END_INTERFACE
  } IAMOpenProgressVtbl;
  struct IAMOpenProgress {
    CONST_VTBL struct IAMOpenProgressVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMOpenProgress_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMOpenProgress_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMOpenProgress_Release(This) (This)->lpVtbl->Release(This)
#define IAMOpenProgress_QueryProgress(This,pllTotal,pllCurrent) (This)->lpVtbl->QueryProgress(This,pllTotal,pllCurrent)
#define IAMOpenProgress_AbortOperation(This) (This)->lpVtbl->AbortOperation(This)
#endif
#endif
  HRESULT WINAPI IAMOpenProgress_QueryProgress_Proxy(IAMOpenProgress *This,LONGLONG *pllTotal,LONGLONG *pllCurrent);
  void __RPC_STUB IAMOpenProgress_QueryProgress_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMOpenProgress_AbortOperation_Proxy(IAMOpenProgress *This);
  void __RPC_STUB IAMOpenProgress_AbortOperation_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IMpeg2Demultiplexer_INTERFACE_DEFINED__
#define __IMpeg2Demultiplexer_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IMpeg2Demultiplexer;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IMpeg2Demultiplexer : public IUnknown {
  public:
    virtual HRESULT WINAPI CreateOutputPin(AM_MEDIA_TYPE *pMediaType,LPWSTR pszPinName,IPin **ppIPin) = 0;
    virtual HRESULT WINAPI SetOutputPinMediaType(LPWSTR pszPinName,AM_MEDIA_TYPE *pMediaType) = 0;
    virtual HRESULT WINAPI DeleteOutputPin(LPWSTR pszPinName) = 0;
  };
#else
  typedef struct IMpeg2DemultiplexerVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IMpeg2Demultiplexer *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IMpeg2Demultiplexer *This);
      ULONG (WINAPI *Release)(IMpeg2Demultiplexer *This);
      HRESULT (WINAPI *CreateOutputPin)(IMpeg2Demultiplexer *This,AM_MEDIA_TYPE *pMediaType,LPWSTR pszPinName,IPin **ppIPin);
      HRESULT (WINAPI *SetOutputPinMediaType)(IMpeg2Demultiplexer *This,LPWSTR pszPinName,AM_MEDIA_TYPE *pMediaType);
      HRESULT (WINAPI *DeleteOutputPin)(IMpeg2Demultiplexer *This,LPWSTR pszPinName);
    END_INTERFACE
  } IMpeg2DemultiplexerVtbl;
  struct IMpeg2Demultiplexer {
    CONST_VTBL struct IMpeg2DemultiplexerVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IMpeg2Demultiplexer_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMpeg2Demultiplexer_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMpeg2Demultiplexer_Release(This) (This)->lpVtbl->Release(This)
#define IMpeg2Demultiplexer_CreateOutputPin(This,pMediaType,pszPinName,ppIPin) (This)->lpVtbl->CreateOutputPin(This,pMediaType,pszPinName,ppIPin)
#define IMpeg2Demultiplexer_SetOutputPinMediaType(This,pszPinName,pMediaType) (This)->lpVtbl->SetOutputPinMediaType(This,pszPinName,pMediaType)
#define IMpeg2Demultiplexer_DeleteOutputPin(This,pszPinName) (This)->lpVtbl->DeleteOutputPin(This,pszPinName)
#endif
#endif
  HRESULT WINAPI IMpeg2Demultiplexer_CreateOutputPin_Proxy(IMpeg2Demultiplexer *This,AM_MEDIA_TYPE *pMediaType,LPWSTR pszPinName,IPin **ppIPin);
  void __RPC_STUB IMpeg2Demultiplexer_CreateOutputPin_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMpeg2Demultiplexer_SetOutputPinMediaType_Proxy(IMpeg2Demultiplexer *This,LPWSTR pszPinName,AM_MEDIA_TYPE *pMediaType);
  void __RPC_STUB IMpeg2Demultiplexer_SetOutputPinMediaType_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMpeg2Demultiplexer_DeleteOutputPin_Proxy(IMpeg2Demultiplexer *This,LPWSTR pszPinName);
  void __RPC_STUB IMpeg2Demultiplexer_DeleteOutputPin_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#define MPEG2_PROGRAM_STREAM_MAP 0x00000000
#define MPEG2_PROGRAM_ELEMENTARY_STREAM 0x00000001
#define MPEG2_PROGRAM_DIRECTORY_PES_PACKET 0x00000002
#define MPEG2_PROGRAM_PACK_HEADER 0x00000003
#define MPEG2_PROGRAM_PES_STREAM 0x00000004
#define MPEG2_PROGRAM_SYSTEM_HEADER 0x00000005
#define SUBSTREAM_FILTER_VAL_NONE 0x10000000

  typedef struct __MIDL___MIDL_itf_strmif_0365_0001 {
    ULONG stream_id;
    DWORD dwMediaSampleContent;
    ULONG ulSubstreamFilterValue;
    int iDataOffset;
  } STREAM_ID_MAP;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0365_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0365_v0_0_s_ifspec;
#ifndef __IEnumStreamIdMap_INTERFACE_DEFINED__
#define __IEnumStreamIdMap_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IEnumStreamIdMap;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IEnumStreamIdMap : public IUnknown {
  public:
    virtual HRESULT WINAPI Next(ULONG cRequest,STREAM_ID_MAP *pStreamIdMap,ULONG *pcReceived) = 0;
    virtual HRESULT WINAPI Skip(ULONG cRecords) = 0;
    virtual HRESULT WINAPI Reset(void) = 0;
    virtual HRESULT WINAPI Clone(IEnumStreamIdMap **ppIEnumStreamIdMap) = 0;
  };
#else
  typedef struct IEnumStreamIdMapVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IEnumStreamIdMap *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IEnumStreamIdMap *This);
      ULONG (WINAPI *Release)(IEnumStreamIdMap *This);
      HRESULT (WINAPI *Next)(IEnumStreamIdMap *This,ULONG cRequest,STREAM_ID_MAP *pStreamIdMap,ULONG *pcReceived);
      HRESULT (WINAPI *Skip)(IEnumStreamIdMap *This,ULONG cRecords);
      HRESULT (WINAPI *Reset)(IEnumStreamIdMap *This);
      HRESULT (WINAPI *Clone)(IEnumStreamIdMap *This,IEnumStreamIdMap **ppIEnumStreamIdMap);
    END_INTERFACE
  } IEnumStreamIdMapVtbl;
  struct IEnumStreamIdMap {
    CONST_VTBL struct IEnumStreamIdMapVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IEnumStreamIdMap_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IEnumStreamIdMap_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IEnumStreamIdMap_Release(This) (This)->lpVtbl->Release(This)
#define IEnumStreamIdMap_Next(This,cRequest,pStreamIdMap,pcReceived) (This)->lpVtbl->Next(This,cRequest,pStreamIdMap,pcReceived)
#define IEnumStreamIdMap_Skip(This,cRecords) (This)->lpVtbl->Skip(This,cRecords)
#define IEnumStreamIdMap_Reset(This) (This)->lpVtbl->Reset(This)
#define IEnumStreamIdMap_Clone(This,ppIEnumStreamIdMap) (This)->lpVtbl->Clone(This,ppIEnumStreamIdMap)
#endif
#endif
  HRESULT WINAPI IEnumStreamIdMap_Next_Proxy(IEnumStreamIdMap *This,ULONG cRequest,STREAM_ID_MAP *pStreamIdMap,ULONG *pcReceived);
  void __RPC_STUB IEnumStreamIdMap_Next_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEnumStreamIdMap_Skip_Proxy(IEnumStreamIdMap *This,ULONG cRecords);
  void __RPC_STUB IEnumStreamIdMap_Skip_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEnumStreamIdMap_Reset_Proxy(IEnumStreamIdMap *This);
  void __RPC_STUB IEnumStreamIdMap_Reset_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEnumStreamIdMap_Clone_Proxy(IEnumStreamIdMap *This,IEnumStreamIdMap **ppIEnumStreamIdMap);
  void __RPC_STUB IEnumStreamIdMap_Clone_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IMPEG2StreamIdMap_INTERFACE_DEFINED__
#define __IMPEG2StreamIdMap_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IMPEG2StreamIdMap;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IMPEG2StreamIdMap : public IUnknown {
  public:
    virtual HRESULT WINAPI MapStreamId(ULONG ulStreamId,DWORD MediaSampleContent,ULONG ulSubstreamFilterValue,int iDataOffset) = 0;
    virtual HRESULT WINAPI UnmapStreamId(ULONG culStreamId,ULONG *pulStreamId) = 0;
    virtual HRESULT WINAPI EnumStreamIdMap(IEnumStreamIdMap **ppIEnumStreamIdMap) = 0;
  };
#else
  typedef struct IMPEG2StreamIdMapVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IMPEG2StreamIdMap *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IMPEG2StreamIdMap *This);
      ULONG (WINAPI *Release)(IMPEG2StreamIdMap *This);
      HRESULT (WINAPI *MapStreamId)(IMPEG2StreamIdMap *This,ULONG ulStreamId,DWORD MediaSampleContent,ULONG ulSubstreamFilterValue,int iDataOffset);
      HRESULT (WINAPI *UnmapStreamId)(IMPEG2StreamIdMap *This,ULONG culStreamId,ULONG *pulStreamId);
      HRESULT (WINAPI *EnumStreamIdMap)(IMPEG2StreamIdMap *This,IEnumStreamIdMap **ppIEnumStreamIdMap);
    END_INTERFACE
  } IMPEG2StreamIdMapVtbl;
  struct IMPEG2StreamIdMap {
    CONST_VTBL struct IMPEG2StreamIdMapVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IMPEG2StreamIdMap_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMPEG2StreamIdMap_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMPEG2StreamIdMap_Release(This) (This)->lpVtbl->Release(This)
#define IMPEG2StreamIdMap_MapStreamId(This,ulStreamId,MediaSampleContent,ulSubstreamFilterValue,iDataOffset) (This)->lpVtbl->MapStreamId(This,ulStreamId,MediaSampleContent,ulSubstreamFilterValue,iDataOffset)
#define IMPEG2StreamIdMap_UnmapStreamId(This,culStreamId,pulStreamId) (This)->lpVtbl->UnmapStreamId(This,culStreamId,pulStreamId)
#define IMPEG2StreamIdMap_EnumStreamIdMap(This,ppIEnumStreamIdMap) (This)->lpVtbl->EnumStreamIdMap(This,ppIEnumStreamIdMap)
#endif
#endif
  HRESULT WINAPI IMPEG2StreamIdMap_MapStreamId_Proxy(IMPEG2StreamIdMap *This,ULONG ulStreamId,DWORD MediaSampleContent,ULONG ulSubstreamFilterValue,int iDataOffset);
  void __RPC_STUB IMPEG2StreamIdMap_MapStreamId_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMPEG2StreamIdMap_UnmapStreamId_Proxy(IMPEG2StreamIdMap *This,ULONG culStreamId,ULONG *pulStreamId);
  void __RPC_STUB IMPEG2StreamIdMap_UnmapStreamId_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMPEG2StreamIdMap_EnumStreamIdMap_Proxy(IMPEG2StreamIdMap *This,IEnumStreamIdMap **ppIEnumStreamIdMap);
  void __RPC_STUB IMPEG2StreamIdMap_EnumStreamIdMap_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IRegisterServiceProvider_INTERFACE_DEFINED__
#define __IRegisterServiceProvider_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IRegisterServiceProvider;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IRegisterServiceProvider : public IUnknown {
  public:
    virtual HRESULT WINAPI RegisterService(REFGUID guidService,IUnknown *pUnkObject) = 0;
  };
#else
  typedef struct IRegisterServiceProviderVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IRegisterServiceProvider *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IRegisterServiceProvider *This);
      ULONG (WINAPI *Release)(IRegisterServiceProvider *This);
      HRESULT (WINAPI *RegisterService)(IRegisterServiceProvider *This,REFGUID guidService,IUnknown *pUnkObject);
    END_INTERFACE
  } IRegisterServiceProviderVtbl;
  struct IRegisterServiceProvider {
    CONST_VTBL struct IRegisterServiceProviderVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IRegisterServiceProvider_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IRegisterServiceProvider_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IRegisterServiceProvider_Release(This) (This)->lpVtbl->Release(This)
#define IRegisterServiceProvider_RegisterService(This,guidService,pUnkObject) (This)->lpVtbl->RegisterService(This,guidService,pUnkObject)
#endif
#endif
  HRESULT WINAPI IRegisterServiceProvider_RegisterService_Proxy(IRegisterServiceProvider *This,REFGUID guidService,IUnknown *pUnkObject);
  void __RPC_STUB IRegisterServiceProvider_RegisterService_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IAMClockSlave_INTERFACE_DEFINED__
#define __IAMClockSlave_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMClockSlave;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMClockSlave : public IUnknown {
  public:
    virtual HRESULT WINAPI SetErrorTolerance(DWORD dwTolerance) = 0;
    virtual HRESULT WINAPI GetErrorTolerance(DWORD *pdwTolerance) = 0;
  };
#else
  typedef struct IAMClockSlaveVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMClockSlave *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMClockSlave *This);
      ULONG (WINAPI *Release)(IAMClockSlave *This);
      HRESULT (WINAPI *SetErrorTolerance)(IAMClockSlave *This,DWORD dwTolerance);
      HRESULT (WINAPI *GetErrorTolerance)(IAMClockSlave *This,DWORD *pdwTolerance);
    END_INTERFACE
  } IAMClockSlaveVtbl;
  struct IAMClockSlave {
    CONST_VTBL struct IAMClockSlaveVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMClockSlave_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMClockSlave_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMClockSlave_Release(This) (This)->lpVtbl->Release(This)
#define IAMClockSlave_SetErrorTolerance(This,dwTolerance) (This)->lpVtbl->SetErrorTolerance(This,dwTolerance)
#define IAMClockSlave_GetErrorTolerance(This,pdwTolerance) (This)->lpVtbl->GetErrorTolerance(This,pdwTolerance)
#endif
#endif
  HRESULT WINAPI IAMClockSlave_SetErrorTolerance_Proxy(IAMClockSlave *This,DWORD dwTolerance);
  void __RPC_STUB IAMClockSlave_SetErrorTolerance_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMClockSlave_GetErrorTolerance_Proxy(IAMClockSlave *This,DWORD *pdwTolerance);
  void __RPC_STUB IAMClockSlave_GetErrorTolerance_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IAMGraphBuilderCallback_INTERFACE_DEFINED__
#define __IAMGraphBuilderCallback_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMGraphBuilderCallback;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMGraphBuilderCallback : public IUnknown {
  public:
    virtual HRESULT WINAPI SelectedFilter(IMoniker *pMon) = 0;
    virtual HRESULT WINAPI CreatedFilter(IBaseFilter *pFil) = 0;
  };
#else
  typedef struct IAMGraphBuilderCallbackVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMGraphBuilderCallback *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMGraphBuilderCallback *This);
      ULONG (WINAPI *Release)(IAMGraphBuilderCallback *This);
      HRESULT (WINAPI *SelectedFilter)(IAMGraphBuilderCallback *This,IMoniker *pMon);
      HRESULT (WINAPI *CreatedFilter)(IAMGraphBuilderCallback *This,IBaseFilter *pFil);
    END_INTERFACE
  } IAMGraphBuilderCallbackVtbl;
  struct IAMGraphBuilderCallback {
    CONST_VTBL struct IAMGraphBuilderCallbackVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMGraphBuilderCallback_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMGraphBuilderCallback_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMGraphBuilderCallback_Release(This) (This)->lpVtbl->Release(This)
#define IAMGraphBuilderCallback_SelectedFilter(This,pMon) (This)->lpVtbl->SelectedFilter(This,pMon)
#define IAMGraphBuilderCallback_CreatedFilter(This,pFil) (This)->lpVtbl->CreatedFilter(This,pFil)
#endif
#endif
  HRESULT WINAPI IAMGraphBuilderCallback_SelectedFilter_Proxy(IAMGraphBuilderCallback *This,IMoniker *pMon);
  void __RPC_STUB IAMGraphBuilderCallback_SelectedFilter_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMGraphBuilderCallback_CreatedFilter_Proxy(IAMGraphBuilderCallback *This,IBaseFilter *pFil);
  void __RPC_STUB IAMGraphBuilderCallback_CreatedFilter_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifdef __cplusplus
#ifndef _IAMFilterGraphCallback_
#define _IAMFilterGraphCallback_
  //EXTERN_GUID(IID_IAMFilterGraphCallback,0x56a868fd,0x0ad4,0x11ce,0xb0,0xa3,0x0,0x20,0xaf,0x0b,0xa7,0x70);
  struct IAMFilterGraphCallback : public IUnknown {
    virtual HRESULT UnableToRender(IPin *pPin) = 0;
  };
#endif
#endif
  struct CodecAPIEventData {
    GUID guid;
    DWORD dataLength;
    DWORD reserved[3];
  };

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0370_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0370_v0_0_s_ifspec;
#ifndef __ICodecAPI_INTERFACE_DEFINED__
#define __ICodecAPI_INTERFACE_DEFINED__
  EXTERN_C const IID IID_ICodecAPI;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct ICodecAPI : public IUnknown {
  public:
    virtual HRESULT WINAPI IsSupported(const GUID *Api) = 0;
    virtual HRESULT WINAPI IsModifiable(const GUID *Api) = 0;
    virtual HRESULT WINAPI GetParameterRange(const GUID *Api,VARIANT *ValueMin,VARIANT *ValueMax,VARIANT *SteppingDelta) = 0;
    virtual HRESULT WINAPI GetParameterValues(const GUID *Api,VARIANT **Values,ULONG *ValuesCount) = 0;
    virtual HRESULT WINAPI GetDefaultValue(const GUID *Api,VARIANT *Value) = 0;
    virtual HRESULT WINAPI GetValue(const GUID *Api,VARIANT *Value) = 0;
    virtual HRESULT WINAPI SetValue(const GUID *Api,VARIANT *Value) = 0;
    virtual HRESULT WINAPI RegisterForEvent(const GUID *Api,LONG_PTR userData) = 0;
    virtual HRESULT WINAPI UnregisterForEvent(const GUID *Api) = 0;
    virtual HRESULT WINAPI SetAllDefaults(void) = 0;
    virtual HRESULT WINAPI SetValueWithNotify(const GUID *Api,VARIANT *Value,GUID **ChangedParam,ULONG *ChangedParamCount) = 0;
    virtual HRESULT WINAPI SetAllDefaultsWithNotify(GUID **ChangedParam,ULONG *ChangedParamCount) = 0;
    virtual HRESULT WINAPI GetAllSettings(IStream *__MIDL_0016) = 0;
    virtual HRESULT WINAPI SetAllSettings(IStream *__MIDL_0017) = 0;
    virtual HRESULT WINAPI SetAllSettingsWithNotify(IStream *__MIDL_0018,GUID **ChangedParam,ULONG *ChangedParamCount) = 0;
  };
#else
  typedef struct ICodecAPIVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(ICodecAPI *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(ICodecAPI *This);
      ULONG (WINAPI *Release)(ICodecAPI *This);
      HRESULT (WINAPI *IsSupported)(ICodecAPI *This,const GUID *Api);
      HRESULT (WINAPI *IsModifiable)(ICodecAPI *This,const GUID *Api);
      HRESULT (WINAPI *GetParameterRange)(ICodecAPI *This,const GUID *Api,VARIANT *ValueMin,VARIANT *ValueMax,VARIANT *SteppingDelta);
      HRESULT (WINAPI *GetParameterValues)(ICodecAPI *This,const GUID *Api,VARIANT **Values,ULONG *ValuesCount);
      HRESULT (WINAPI *GetDefaultValue)(ICodecAPI *This,const GUID *Api,VARIANT *Value);
      HRESULT (WINAPI *GetValue)(ICodecAPI *This,const GUID *Api,VARIANT *Value);
      HRESULT (WINAPI *SetValue)(ICodecAPI *This,const GUID *Api,VARIANT *Value);
      HRESULT (WINAPI *RegisterForEvent)(ICodecAPI *This,const GUID *Api,LONG_PTR userData);
      HRESULT (WINAPI *UnregisterForEvent)(ICodecAPI *This,const GUID *Api);
      HRESULT (WINAPI *SetAllDefaults)(ICodecAPI *This);
      HRESULT (WINAPI *SetValueWithNotify)(ICodecAPI *This,const GUID *Api,VARIANT *Value,GUID **ChangedParam,ULONG *ChangedParamCount);
      HRESULT (WINAPI *SetAllDefaultsWithNotify)(ICodecAPI *This,GUID **ChangedParam,ULONG *ChangedParamCount);
      HRESULT (WINAPI *GetAllSettings)(ICodecAPI *This,IStream *__MIDL_0016);
      HRESULT (WINAPI *SetAllSettings)(ICodecAPI *This,IStream *__MIDL_0017);
      HRESULT (WINAPI *SetAllSettingsWithNotify)(ICodecAPI *This,IStream *__MIDL_0018,GUID **ChangedParam,ULONG *ChangedParamCount);
    END_INTERFACE
  } ICodecAPIVtbl;
  struct ICodecAPI {
    CONST_VTBL struct ICodecAPIVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define ICodecAPI_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define ICodecAPI_AddRef(This) (This)->lpVtbl->AddRef(This)
#define ICodecAPI_Release(This) (This)->lpVtbl->Release(This)
#define ICodecAPI_IsSupported(This,Api) (This)->lpVtbl->IsSupported(This,Api)
#define ICodecAPI_IsModifiable(This,Api) (This)->lpVtbl->IsModifiable(This,Api)
#define ICodecAPI_GetParameterRange(This,Api,ValueMin,ValueMax,SteppingDelta) (This)->lpVtbl->GetParameterRange(This,Api,ValueMin,ValueMax,SteppingDelta)
#define ICodecAPI_GetParameterValues(This,Api,Values,ValuesCount) (This)->lpVtbl->GetParameterValues(This,Api,Values,ValuesCount)
#define ICodecAPI_GetDefaultValue(This,Api,Value) (This)->lpVtbl->GetDefaultValue(This,Api,Value)
#define ICodecAPI_GetValue(This,Api,Value) (This)->lpVtbl->GetValue(This,Api,Value)
#define ICodecAPI_SetValue(This,Api,Value) (This)->lpVtbl->SetValue(This,Api,Value)
#define ICodecAPI_RegisterForEvent(This,Api,userData) (This)->lpVtbl->RegisterForEvent(This,Api,userData)
#define ICodecAPI_UnregisterForEvent(This,Api) (This)->lpVtbl->UnregisterForEvent(This,Api)
#define ICodecAPI_SetAllDefaults(This) (This)->lpVtbl->SetAllDefaults(This)
#define ICodecAPI_SetValueWithNotify(This,Api,Value,ChangedParam,ChangedParamCount) (This)->lpVtbl->SetValueWithNotify(This,Api,Value,ChangedParam,ChangedParamCount)
#define ICodecAPI_SetAllDefaultsWithNotify(This,ChangedParam,ChangedParamCount) (This)->lpVtbl->SetAllDefaultsWithNotify(This,ChangedParam,ChangedParamCount)
#define ICodecAPI_GetAllSettings(This,__MIDL_0016) (This)->lpVtbl->GetAllSettings(This,__MIDL_0016)
#define ICodecAPI_SetAllSettings(This,__MIDL_0017) (This)->lpVtbl->SetAllSettings(This,__MIDL_0017)
#define ICodecAPI_SetAllSettingsWithNotify(This,__MIDL_0018,ChangedParam,ChangedParamCount) (This)->lpVtbl->SetAllSettingsWithNotify(This,__MIDL_0018,ChangedParam,ChangedParamCount)
#endif
#endif
  HRESULT WINAPI ICodecAPI_IsSupported_Proxy(ICodecAPI *This,const GUID *Api);
  void __RPC_STUB ICodecAPI_IsSupported_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICodecAPI_IsModifiable_Proxy(ICodecAPI *This,const GUID *Api);
  void __RPC_STUB ICodecAPI_IsModifiable_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICodecAPI_GetParameterRange_Proxy(ICodecAPI *This,const GUID *Api,VARIANT *ValueMin,VARIANT *ValueMax,VARIANT *SteppingDelta);
  void __RPC_STUB ICodecAPI_GetParameterRange_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICodecAPI_GetParameterValues_Proxy(ICodecAPI *This,const GUID *Api,VARIANT **Values,ULONG *ValuesCount);
  void __RPC_STUB ICodecAPI_GetParameterValues_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICodecAPI_GetDefaultValue_Proxy(ICodecAPI *This,const GUID *Api,VARIANT *Value);
  void __RPC_STUB ICodecAPI_GetDefaultValue_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICodecAPI_GetValue_Proxy(ICodecAPI *This,const GUID *Api,VARIANT *Value);
  void __RPC_STUB ICodecAPI_GetValue_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICodecAPI_SetValue_Proxy(ICodecAPI *This,const GUID *Api,VARIANT *Value);
  void __RPC_STUB ICodecAPI_SetValue_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICodecAPI_RegisterForEvent_Proxy(ICodecAPI *This,const GUID *Api,LONG_PTR userData);
  void __RPC_STUB ICodecAPI_RegisterForEvent_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICodecAPI_UnregisterForEvent_Proxy(ICodecAPI *This,const GUID *Api);
  void __RPC_STUB ICodecAPI_UnregisterForEvent_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICodecAPI_SetAllDefaults_Proxy(ICodecAPI *This);
  void __RPC_STUB ICodecAPI_SetAllDefaults_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICodecAPI_SetValueWithNotify_Proxy(ICodecAPI *This,const GUID *Api,VARIANT *Value,GUID **ChangedParam,ULONG *ChangedParamCount);
  void __RPC_STUB ICodecAPI_SetValueWithNotify_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICodecAPI_SetAllDefaultsWithNotify_Proxy(ICodecAPI *This,GUID **ChangedParam,ULONG *ChangedParamCount);
  void __RPC_STUB ICodecAPI_SetAllDefaultsWithNotify_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICodecAPI_GetAllSettings_Proxy(ICodecAPI *This,IStream *__MIDL_0016);
  void __RPC_STUB ICodecAPI_GetAllSettings_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICodecAPI_SetAllSettings_Proxy(ICodecAPI *This,IStream *__MIDL_0017);
  void __RPC_STUB ICodecAPI_SetAllSettings_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI ICodecAPI_SetAllSettingsWithNotify_Proxy(ICodecAPI *This,IStream *__MIDL_0018,GUID **ChangedParam,ULONG *ChangedParamCount);
  void __RPC_STUB ICodecAPI_SetAllSettingsWithNotify_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IGetCapabilitiesKey_INTERFACE_DEFINED__
#define __IGetCapabilitiesKey_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IGetCapabilitiesKey;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IGetCapabilitiesKey : public IUnknown {
  public:
    virtual HRESULT WINAPI GetCapabilitiesKey(HKEY *pHKey) = 0;
  };
#else
  typedef struct IGetCapabilitiesKeyVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IGetCapabilitiesKey *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IGetCapabilitiesKey *This);
      ULONG (WINAPI *Release)(IGetCapabilitiesKey *This);
      HRESULT (WINAPI *GetCapabilitiesKey)(IGetCapabilitiesKey *This,HKEY *pHKey);
    END_INTERFACE
  } IGetCapabilitiesKeyVtbl;
  struct IGetCapabilitiesKey {
    CONST_VTBL struct IGetCapabilitiesKeyVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IGetCapabilitiesKey_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IGetCapabilitiesKey_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IGetCapabilitiesKey_Release(This) (This)->lpVtbl->Release(This)
#define IGetCapabilitiesKey_GetCapabilitiesKey(This,pHKey) (This)->lpVtbl->GetCapabilitiesKey(This,pHKey)
#endif
#endif
  HRESULT WINAPI IGetCapabilitiesKey_GetCapabilitiesKey_Proxy(IGetCapabilitiesKey *This,HKEY *pHKey);
  void __RPC_STUB IGetCapabilitiesKey_GetCapabilitiesKey_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IEncoderAPI_INTERFACE_DEFINED__
#define __IEncoderAPI_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IEncoderAPI;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IEncoderAPI : public IUnknown {
  public:
    virtual HRESULT WINAPI IsSupported(const GUID *Api) = 0;
    virtual HRESULT WINAPI IsAvailable(const GUID *Api) = 0;
    virtual HRESULT WINAPI GetParameterRange(const GUID *Api,VARIANT *ValueMin,VARIANT *ValueMax,VARIANT *SteppingDelta) = 0;
    virtual HRESULT WINAPI GetParameterValues(const GUID *Api,VARIANT **Values,ULONG *ValuesCount) = 0;
    virtual HRESULT WINAPI GetDefaultValue(const GUID *Api,VARIANT *Value) = 0;
    virtual HRESULT WINAPI GetValue(const GUID *Api,VARIANT *Value) = 0;
    virtual HRESULT WINAPI SetValue(const GUID *Api,VARIANT *Value) = 0;
  };
#else
  typedef struct IEncoderAPIVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IEncoderAPI *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IEncoderAPI *This);
      ULONG (WINAPI *Release)(IEncoderAPI *This);
      HRESULT (WINAPI *IsSupported)(IEncoderAPI *This,const GUID *Api);
      HRESULT (WINAPI *IsAvailable)(IEncoderAPI *This,const GUID *Api);
      HRESULT (WINAPI *GetParameterRange)(IEncoderAPI *This,const GUID *Api,VARIANT *ValueMin,VARIANT *ValueMax,VARIANT *SteppingDelta);
      HRESULT (WINAPI *GetParameterValues)(IEncoderAPI *This,const GUID *Api,VARIANT **Values,ULONG *ValuesCount);
      HRESULT (WINAPI *GetDefaultValue)(IEncoderAPI *This,const GUID *Api,VARIANT *Value);
      HRESULT (WINAPI *GetValue)(IEncoderAPI *This,const GUID *Api,VARIANT *Value);
      HRESULT (WINAPI *SetValue)(IEncoderAPI *This,const GUID *Api,VARIANT *Value);
    END_INTERFACE
  } IEncoderAPIVtbl;
  struct IEncoderAPI {
    CONST_VTBL struct IEncoderAPIVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IEncoderAPI_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IEncoderAPI_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IEncoderAPI_Release(This) (This)->lpVtbl->Release(This)
#define IEncoderAPI_IsSupported(This,Api) (This)->lpVtbl->IsSupported(This,Api)
#define IEncoderAPI_IsAvailable(This,Api) (This)->lpVtbl->IsAvailable(This,Api)
#define IEncoderAPI_GetParameterRange(This,Api,ValueMin,ValueMax,SteppingDelta) (This)->lpVtbl->GetParameterRange(This,Api,ValueMin,ValueMax,SteppingDelta)
#define IEncoderAPI_GetParameterValues(This,Api,Values,ValuesCount) (This)->lpVtbl->GetParameterValues(This,Api,Values,ValuesCount)
#define IEncoderAPI_GetDefaultValue(This,Api,Value) (This)->lpVtbl->GetDefaultValue(This,Api,Value)
#define IEncoderAPI_GetValue(This,Api,Value) (This)->lpVtbl->GetValue(This,Api,Value)
#define IEncoderAPI_SetValue(This,Api,Value) (This)->lpVtbl->SetValue(This,Api,Value)
#endif
#endif
  HRESULT WINAPI IEncoderAPI_IsSupported_Proxy(IEncoderAPI *This,const GUID *Api);
  void __RPC_STUB IEncoderAPI_IsSupported_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEncoderAPI_IsAvailable_Proxy(IEncoderAPI *This,const GUID *Api);
  void __RPC_STUB IEncoderAPI_IsAvailable_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEncoderAPI_GetParameterRange_Proxy(IEncoderAPI *This,const GUID *Api,VARIANT *ValueMin,VARIANT *ValueMax,VARIANT *SteppingDelta);
  void __RPC_STUB IEncoderAPI_GetParameterRange_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEncoderAPI_GetParameterValues_Proxy(IEncoderAPI *This,const GUID *Api,VARIANT **Values,ULONG *ValuesCount);
  void __RPC_STUB IEncoderAPI_GetParameterValues_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEncoderAPI_GetDefaultValue_Proxy(IEncoderAPI *This,const GUID *Api,VARIANT *Value);
  void __RPC_STUB IEncoderAPI_GetDefaultValue_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEncoderAPI_GetValue_Proxy(IEncoderAPI *This,const GUID *Api,VARIANT *Value);
  void __RPC_STUB IEncoderAPI_GetValue_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IEncoderAPI_SetValue_Proxy(IEncoderAPI *This,const GUID *Api,VARIANT *Value);
  void __RPC_STUB IEncoderAPI_SetValue_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IVideoEncoder_INTERFACE_DEFINED__
#define __IVideoEncoder_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IVideoEncoder;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IVideoEncoder : public IEncoderAPI {
  public:
  };
#else
  typedef struct IVideoEncoderVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IVideoEncoder *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IVideoEncoder *This);
      ULONG (WINAPI *Release)(IVideoEncoder *This);
      HRESULT (WINAPI *IsSupported)(IVideoEncoder *This,const GUID *Api);
      HRESULT (WINAPI *IsAvailable)(IVideoEncoder *This,const GUID *Api);
      HRESULT (WINAPI *GetParameterRange)(IVideoEncoder *This,const GUID *Api,VARIANT *ValueMin,VARIANT *ValueMax,VARIANT *SteppingDelta);
      HRESULT (WINAPI *GetParameterValues)(IVideoEncoder *This,const GUID *Api,VARIANT **Values,ULONG *ValuesCount);
      HRESULT (WINAPI *GetDefaultValue)(IVideoEncoder *This,const GUID *Api,VARIANT *Value);
      HRESULT (WINAPI *GetValue)(IVideoEncoder *This,const GUID *Api,VARIANT *Value);
      HRESULT (WINAPI *SetValue)(IVideoEncoder *This,const GUID *Api,VARIANT *Value);
    END_INTERFACE
  } IVideoEncoderVtbl;
  struct IVideoEncoder {
    CONST_VTBL struct IVideoEncoderVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IVideoEncoder_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IVideoEncoder_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IVideoEncoder_Release(This) (This)->lpVtbl->Release(This)
#define IVideoEncoder_IsSupported(This,Api) (This)->lpVtbl->IsSupported(This,Api)
#define IVideoEncoder_IsAvailable(This,Api) (This)->lpVtbl->IsAvailable(This,Api)
#define IVideoEncoder_GetParameterRange(This,Api,ValueMin,ValueMax,SteppingDelta) (This)->lpVtbl->GetParameterRange(This,Api,ValueMin,ValueMax,SteppingDelta)
#define IVideoEncoder_GetParameterValues(This,Api,Values,ValuesCount) (This)->lpVtbl->GetParameterValues(This,Api,Values,ValuesCount)
#define IVideoEncoder_GetDefaultValue(This,Api,Value) (This)->lpVtbl->GetDefaultValue(This,Api,Value)
#define IVideoEncoder_GetValue(This,Api,Value) (This)->lpVtbl->GetValue(This,Api,Value)
#define IVideoEncoder_SetValue(This,Api,Value) (This)->lpVtbl->SetValue(This,Api,Value)
#endif
#endif
#endif

#ifndef __ENCODER_API_DEFINES__
#define __ENCODER_API_DEFINES__
  typedef enum __MIDL___MIDL_itf_strmif_0374_0001 {
    ConstantBitRate = 0,
    VariableBitRateAverage,VariableBitRatePeak
  } VIDEOENCODER_BITRATE_MODE;
#endif
#define AM_GETDECODERCAP_QUERY_VMR_SUPPORT 0x00000001
#define VMR_NOTSUPPORTED 0x00000000
#define VMR_SUPPORTED 0x00000001
#define AM_QUERY_DECODER_VMR_SUPPORT 0x00000001
#define AM_QUERY_DECODER_DXVA_1_SUPPORT 0x00000002
#define AM_QUERY_DECODER_DVD_SUPPORT 0x00000003
#define AM_QUERY_DECODER_ATSC_SD_SUPPORT 0x00000004
#define AM_QUERY_DECODER_ATSC_HD_SUPPORT 0x00000005
#define AM_GETDECODERCAP_QUERY_VMR9_SUPPORT 0x00000006
#define DECODER_CAP_NOTSUPPORTED 0x00000000
#define DECODER_CAP_SUPPORTED 0x00000001

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0374_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0374_v0_0_s_ifspec;
#ifndef __IAMDecoderCaps_INTERFACE_DEFINED__
#define __IAMDecoderCaps_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMDecoderCaps;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMDecoderCaps : public IUnknown {
  public:
    virtual HRESULT WINAPI GetDecoderCaps(DWORD dwCapIndex,DWORD *lpdwCap) = 0;
  };
#else
  typedef struct IAMDecoderCapsVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMDecoderCaps *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMDecoderCaps *This);
      ULONG (WINAPI *Release)(IAMDecoderCaps *This);
      HRESULT (WINAPI *GetDecoderCaps)(IAMDecoderCaps *This,DWORD dwCapIndex,DWORD *lpdwCap);
    END_INTERFACE
  } IAMDecoderCapsVtbl;
  struct IAMDecoderCaps {
    CONST_VTBL struct IAMDecoderCapsVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMDecoderCaps_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMDecoderCaps_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMDecoderCaps_Release(This) (This)->lpVtbl->Release(This)
#define IAMDecoderCaps_GetDecoderCaps(This,dwCapIndex,lpdwCap) (This)->lpVtbl->GetDecoderCaps(This,dwCapIndex,lpdwCap)
#endif
#endif
  HRESULT WINAPI IAMDecoderCaps_GetDecoderCaps_Proxy(IAMDecoderCaps *This,DWORD dwCapIndex,DWORD *lpdwCap);
  void __RPC_STUB IAMDecoderCaps_GetDecoderCaps_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef struct _AMCOPPSignature {
    BYTE Signature[256];
  } AMCOPPSignature;

  typedef struct _AMCOPPCommand {
    GUID macKDI;
    GUID guidCommandID;
    DWORD dwSequence;
    DWORD cbSizeData;
    BYTE CommandData[4056];
  } AMCOPPCommand;

  typedef struct _AMCOPPCommand *LPAMCOPPCommand;

  typedef struct _AMCOPPStatusInput {
    GUID rApp;
    GUID guidStatusRequestID;
    DWORD dwSequence;
    DWORD cbSizeData;
    BYTE StatusData[4056];
  } AMCOPPStatusInput;

  typedef struct _AMCOPPStatusInput *LPAMCOPPStatusInput;

  typedef struct _AMCOPPStatusOutput {
    GUID macKDI;
    DWORD cbSizeData;
    BYTE COPPStatus[4076];
  } AMCOPPStatusOutput;

  typedef struct _AMCOPPStatusOutput *LPAMCOPPStatusOutput;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0375_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0375_v0_0_s_ifspec;
#ifndef __IAMCertifiedOutputProtection_INTERFACE_DEFINED__
#define __IAMCertifiedOutputProtection_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IAMCertifiedOutputProtection;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMCertifiedOutputProtection : public IUnknown {
  public:
    virtual HRESULT WINAPI KeyExchange(GUID *pRandom,BYTE **VarLenCertGH,DWORD *pdwLengthCertGH) = 0;
    virtual HRESULT WINAPI SessionSequenceStart(AMCOPPSignature *pSig) = 0;
    virtual HRESULT WINAPI ProtectionCommand(const AMCOPPCommand *cmd) = 0;
    virtual HRESULT WINAPI ProtectionStatus(const AMCOPPStatusInput *pStatusInput,AMCOPPStatusOutput *pStatusOutput) = 0;
  };
#else
  typedef struct IAMCertifiedOutputProtectionVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMCertifiedOutputProtection *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMCertifiedOutputProtection *This);
      ULONG (WINAPI *Release)(IAMCertifiedOutputProtection *This);
      HRESULT (WINAPI *KeyExchange)(IAMCertifiedOutputProtection *This,GUID *pRandom,BYTE **VarLenCertGH,DWORD *pdwLengthCertGH);
      HRESULT (WINAPI *SessionSequenceStart)(IAMCertifiedOutputProtection *This,AMCOPPSignature *pSig);
      HRESULT (WINAPI *ProtectionCommand)(IAMCertifiedOutputProtection *This,const AMCOPPCommand *cmd);
      HRESULT (WINAPI *ProtectionStatus)(IAMCertifiedOutputProtection *This,const AMCOPPStatusInput *pStatusInput,AMCOPPStatusOutput *pStatusOutput);
    END_INTERFACE
  } IAMCertifiedOutputProtectionVtbl;
  struct IAMCertifiedOutputProtection {
    CONST_VTBL struct IAMCertifiedOutputProtectionVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMCertifiedOutputProtection_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMCertifiedOutputProtection_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMCertifiedOutputProtection_Release(This) (This)->lpVtbl->Release(This)
#define IAMCertifiedOutputProtection_KeyExchange(This,pRandom,VarLenCertGH,pdwLengthCertGH) (This)->lpVtbl->KeyExchange(This,pRandom,VarLenCertGH,pdwLengthCertGH)
#define IAMCertifiedOutputProtection_SessionSequenceStart(This,pSig) (This)->lpVtbl->SessionSequenceStart(This,pSig)
#define IAMCertifiedOutputProtection_ProtectionCommand(This,cmd) (This)->lpVtbl->ProtectionCommand(This,cmd)
#define IAMCertifiedOutputProtection_ProtectionStatus(This,pStatusInput,pStatusOutput) (This)->lpVtbl->ProtectionStatus(This,pStatusInput,pStatusOutput)
#endif
#endif
  HRESULT WINAPI IAMCertifiedOutputProtection_KeyExchange_Proxy(IAMCertifiedOutputProtection *This,GUID *pRandom,BYTE **VarLenCertGH,DWORD *pdwLengthCertGH);
  void __RPC_STUB IAMCertifiedOutputProtection_KeyExchange_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMCertifiedOutputProtection_SessionSequenceStart_Proxy(IAMCertifiedOutputProtection *This,AMCOPPSignature *pSig);
  void __RPC_STUB IAMCertifiedOutputProtection_SessionSequenceStart_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMCertifiedOutputProtection_ProtectionCommand_Proxy(IAMCertifiedOutputProtection *This,const AMCOPPCommand *cmd);
  void __RPC_STUB IAMCertifiedOutputProtection_ProtectionCommand_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMCertifiedOutputProtection_ProtectionStatus_Proxy(IAMCertifiedOutputProtection *This,const AMCOPPStatusInput *pStatusInput,AMCOPPStatusOutput *pStatusOutput);
  void __RPC_STUB IAMCertifiedOutputProtection_ProtectionStatus_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#include <dshow/ddraw.h>

  typedef enum tagDVD_DOMAIN {
    DVD_DOMAIN_FirstPlay = 1,
    DVD_DOMAIN_VideoManagerMenu,DVD_DOMAIN_VideoTitleSetMenu,DVD_DOMAIN_Title,
    DVD_DOMAIN_Stop
  } DVD_DOMAIN;

  typedef enum tagDVD_MENU_ID {
    DVD_MENU_Title = 2,DVD_MENU_Root = 3,DVD_MENU_Subpicture = 4,DVD_MENU_Audio = 5,
    DVD_MENU_Angle = 6,DVD_MENU_Chapter = 7
  } DVD_MENU_ID;

  typedef enum tagDVD_DISC_SIDE {
    DVD_SIDE_A = 1,DVD_SIDE_B = 2
  } DVD_DISC_SIDE;

  typedef enum tagDVD_PREFERRED_DISPLAY_MODE {
    DISPLAY_CONTENT_DEFAULT = 0,DISPLAY_16x9 = 1,DISPLAY_4x3_PANSCAN_PREFERRED = 2,DISPLAY_4x3_LETTERBOX_PREFERRED = 3
  } DVD_PREFERRED_DISPLAY_MODE;

  typedef WORD DVD_REGISTER;
  typedef DVD_REGISTER GPRMARRAY[16];
  typedef DVD_REGISTER SPRMARRAY[24];

  typedef struct tagDVD_ATR {
    ULONG ulCAT;
    BYTE pbATRI[768];
  } DVD_ATR;

  typedef BYTE DVD_VideoATR[2];
  typedef BYTE DVD_AudioATR[8];
  typedef BYTE DVD_SubpictureATR[6];

  typedef enum tagDVD_FRAMERATE {
    DVD_FPS_25 = 1,DVD_FPS_30NonDrop = 3
  } DVD_FRAMERATE;

  typedef struct tagDVD_TIMECODE {
    ULONG Hours1 :4;
    ULONG Hours10 :4;
    ULONG Minutes1 :4;
    ULONG Minutes10:4;
    ULONG Seconds1 :4;
    ULONG Seconds10:4;
    ULONG Frames1 :4;
    ULONG Frames10 :2;
    ULONG FrameRateCode: 2;
  } DVD_TIMECODE;

  typedef enum tagDVD_TIMECODE_FLAGS {
    DVD_TC_FLAG_25fps = 0x1,DVD_TC_FLAG_30fps = 0x2,DVD_TC_FLAG_DropFrame = 0x4,DVD_TC_FLAG_Interpolated = 0x8
  } DVD_TIMECODE_FLAGS;

  typedef struct tagDVD_HMSF_TIMECODE {
    BYTE bHours;
    BYTE bMinutes;
    BYTE bSeconds;
    BYTE bFrames;
  } DVD_HMSF_TIMECODE;

  typedef struct tagDVD_PLAYBACK_LOCATION2 {
    ULONG TitleNum;
    ULONG ChapterNum;
    DVD_HMSF_TIMECODE TimeCode;
    ULONG TimeCodeFlags;
  } DVD_PLAYBACK_LOCATION2;

  typedef struct tagDVD_PLAYBACK_LOCATION {
    ULONG TitleNum;
    ULONG ChapterNum;
    ULONG TimeCode;
  } DVD_PLAYBACK_LOCATION;

  typedef DWORD VALID_UOP_SOMTHING_OR_OTHER;

  typedef enum __MIDL___MIDL_itf_strmif_0376_0001 {
    UOP_FLAG_Play_Title_Or_AtTime = 0x1,UOP_FLAG_Play_Chapter = 0x2,UOP_FLAG_Play_Title = 0x4,UOP_FLAG_Stop = 0x8,UOP_FLAG_ReturnFromSubMenu = 0x10,
    UOP_FLAG_Play_Chapter_Or_AtTime = 0x20,UOP_FLAG_PlayPrev_Or_Replay_Chapter = 0x40,UOP_FLAG_PlayNext_Chapter = 0x80,UOP_FLAG_Play_Forwards = 0x100,
    UOP_FLAG_Play_Backwards = 0x200,UOP_FLAG_ShowMenu_Title = 0x400,UOP_FLAG_ShowMenu_Root = 0x800,UOP_FLAG_ShowMenu_SubPic = 0x1000,
    UOP_FLAG_ShowMenu_Audio = 0x2000,UOP_FLAG_ShowMenu_Angle = 0x4000,UOP_FLAG_ShowMenu_Chapter = 0x8000,UOP_FLAG_Resume = 0x10000,
    UOP_FLAG_Select_Or_Activate_Button = 0x20000,UOP_FLAG_Still_Off = 0x40000,UOP_FLAG_Pause_On = 0x80000,UOP_FLAG_Select_Audio_Stream = 0x100000,
    UOP_FLAG_Select_SubPic_Stream = 0x200000,UOP_FLAG_Select_Angle = 0x400000,UOP_FLAG_Select_Karaoke_Audio_Presentation_Mode = 0x800000,
    UOP_FLAG_Select_Video_Mode_Preference = 0x1000000
  } VALID_UOP_FLAG;

  typedef enum __MIDL___MIDL_itf_strmif_0376_0002 {
    DVD_CMD_FLAG_None = 0,DVD_CMD_FLAG_Flush = 0x1,DVD_CMD_FLAG_SendEvents = 0x2,DVD_CMD_FLAG_Block = 0x4,DVD_CMD_FLAG_StartWhenRendered = 0x8,
    DVD_CMD_FLAG_EndAfterRendered = 0x10
  } DVD_CMD_FLAGS;

  typedef enum __MIDL___MIDL_itf_strmif_0376_0003 {
    DVD_ResetOnStop = 1,DVD_NotifyParentalLevelChange = 2,DVD_HMSF_TimeCodeEvents = 3,DVD_AudioDuringFFwdRew = 4
  } DVD_OPTION_FLAG;

  typedef enum __MIDL___MIDL_itf_strmif_0376_0004 {
    DVD_Relative_Upper = 1,DVD_Relative_Lower = 2,DVD_Relative_Left = 3,DVD_Relative_Right = 4
  } DVD_RELATIVE_BUTTON;

  typedef enum tagDVD_PARENTAL_LEVEL {
    DVD_PARENTAL_LEVEL_8 = 0x8000,DVD_PARENTAL_LEVEL_7 = 0x4000,DVD_PARENTAL_LEVEL_6 = 0x2000,DVD_PARENTAL_LEVEL_5 = 0x1000,
    DVD_PARENTAL_LEVEL_4 = 0x800,DVD_PARENTAL_LEVEL_3 = 0x400,DVD_PARENTAL_LEVEL_2 = 0x200,DVD_PARENTAL_LEVEL_1 = 0x100
  } DVD_PARENTAL_LEVEL;

  typedef enum tagDVD_AUDIO_LANG_EXT {
    DVD_AUD_EXT_NotSpecified = 0,DVD_AUD_EXT_Captions = 1,DVD_AUD_EXT_VisuallyImpaired = 2,DVD_AUD_EXT_DirectorComments1 = 3,
    DVD_AUD_EXT_DirectorComments2 = 4
  } DVD_AUDIO_LANG_EXT;

  typedef enum tagDVD_SUBPICTURE_LANG_EXT {
    DVD_SP_EXT_NotSpecified = 0,DVD_SP_EXT_Caption_Normal = 1,DVD_SP_EXT_Caption_Big = 2,DVD_SP_EXT_Caption_Children = 3,DVD_SP_EXT_CC_Normal = 5,
    DVD_SP_EXT_CC_Big = 6,DVD_SP_EXT_CC_Children = 7,DVD_SP_EXT_Forced = 9,DVD_SP_EXT_DirectorComments_Normal = 13,DVD_SP_EXT_DirectorComments_Big = 14,
    DVD_SP_EXT_DirectorComments_Children = 15
  } DVD_SUBPICTURE_LANG_EXT;

  typedef enum tagDVD_AUDIO_APPMODE {
    DVD_AudioMode_None = 0,DVD_AudioMode_Karaoke = 1,DVD_AudioMode_Surround = 2,DVD_AudioMode_Other = 3
  } DVD_AUDIO_APPMODE;

  typedef enum tagDVD_AUDIO_FORMAT {
    DVD_AudioFormat_AC3 = 0,DVD_AudioFormat_MPEG1 = 1,DVD_AudioFormat_MPEG1_DRC = 2,DVD_AudioFormat_MPEG2 = 3,DVD_AudioFormat_MPEG2_DRC = 4,
    DVD_AudioFormat_LPCM = 5,DVD_AudioFormat_DTS = 6,DVD_AudioFormat_SDDS = 7,DVD_AudioFormat_Other = 8
  } DVD_AUDIO_FORMAT;

  typedef enum tagDVD_KARAOKE_DOWNMIX {
    DVD_Mix_0to0 = 0x1,DVD_Mix_1to0 = 0x2,DVD_Mix_2to0 = 0x4,DVD_Mix_3to0 = 0x8,DVD_Mix_4to0 = 0x10,DVD_Mix_Lto0 = 0x20,DVD_Mix_Rto0 = 0x40,
    DVD_Mix_0to1 = 0x100,DVD_Mix_1to1 = 0x200,DVD_Mix_2to1 = 0x400,DVD_Mix_3to1 = 0x800,DVD_Mix_4to1 = 0x1000,DVD_Mix_Lto1 = 0x2000,
    DVD_Mix_Rto1 = 0x4000
  } DVD_KARAOKE_DOWNMIX;

  typedef struct tagDVD_AudioAttributes {
    DVD_AUDIO_APPMODE AppMode;
    BYTE AppModeData;
    DVD_AUDIO_FORMAT AudioFormat;
    LCID Language;
    DVD_AUDIO_LANG_EXT LanguageExtension;
    WINBOOL fHasMultichannelInfo;
    DWORD dwFrequency;
    BYTE bQuantization;
    BYTE bNumberOfChannels;
    DWORD dwReserved[2];
  } DVD_AudioAttributes;

  typedef struct tagDVD_MUA_MixingInfo {
    WINBOOL fMixTo0;
    WINBOOL fMixTo1;
    WINBOOL fMix0InPhase;
    WINBOOL fMix1InPhase;
    DWORD dwSpeakerPosition;
  } DVD_MUA_MixingInfo;

  typedef struct tagDVD_MUA_Coeff {
    double log2_alpha;
    double log2_beta;
  } DVD_MUA_Coeff;

  typedef struct tagDVD_MultichannelAudioAttributes {
    DVD_MUA_MixingInfo Info[8];
    DVD_MUA_Coeff Coeff[8];
  } DVD_MultichannelAudioAttributes;

  typedef enum tagDVD_KARAOKE_CONTENTS {
    DVD_Karaoke_GuideVocal1 = 0x1,DVD_Karaoke_GuideVocal2 = 0x2,DVD_Karaoke_GuideMelody1 = 0x4,DVD_Karaoke_GuideMelody2 = 0x8,
    DVD_Karaoke_GuideMelodyA = 0x10,DVD_Karaoke_GuideMelodyB = 0x20,DVD_Karaoke_SoundEffectA = 0x40,DVD_Karaoke_SoundEffectB = 0x80
  } DVD_KARAOKE_CONTENTS;

  typedef enum tagDVD_KARAOKE_ASSIGNMENT {
    DVD_Assignment_reserved0 = 0,DVD_Assignment_reserved1 = 1,DVD_Assignment_LR = 2,DVD_Assignment_LRM = 3,DVD_Assignment_LR1 = 4,
    DVD_Assignment_LRM1 = 5,DVD_Assignment_LR12 = 6,DVD_Assignment_LRM12 = 7
  } DVD_KARAOKE_ASSIGNMENT;

  typedef struct tagDVD_KaraokeAttributes {
    BYTE bVersion;
    WINBOOL fMasterOfCeremoniesInGuideVocal1;
    WINBOOL fDuet;
    DVD_KARAOKE_ASSIGNMENT ChannelAssignment;
    WORD wChannelContents[8];
  } DVD_KaraokeAttributes;

  typedef enum tagDVD_VIDEO_COMPRESSION {
    DVD_VideoCompression_Other = 0,DVD_VideoCompression_MPEG1 = 1,DVD_VideoCompression_MPEG2 = 2
  } DVD_VIDEO_COMPRESSION;

  typedef struct tagDVD_VideoAttributes {
    WINBOOL fPanscanPermitted;
    WINBOOL fLetterboxPermitted;
    ULONG ulAspectX;
    ULONG ulAspectY;
    ULONG ulFrameRate;
    ULONG ulFrameHeight;
    DVD_VIDEO_COMPRESSION Compression;
    WINBOOL fLine21Field1InGOP;
    WINBOOL fLine21Field2InGOP;
    ULONG ulSourceResolutionX;
    ULONG ulSourceResolutionY;
    WINBOOL fIsSourceLetterboxed;
    WINBOOL fIsFilmMode;
  } DVD_VideoAttributes;

  typedef enum tagDVD_SUBPICTURE_TYPE {
    DVD_SPType_NotSpecified = 0,DVD_SPType_Language = 1,DVD_SPType_Other = 2
  } DVD_SUBPICTURE_TYPE;

  typedef enum tagDVD_SUBPICTURE_CODING {
    DVD_SPCoding_RunLength = 0,DVD_SPCoding_Extended = 1,DVD_SPCoding_Other = 2
  } DVD_SUBPICTURE_CODING;

  typedef struct tagDVD_SubpictureAttributes {
    DVD_SUBPICTURE_TYPE Type;
    DVD_SUBPICTURE_CODING CodingMode;
    LCID Language;
    DVD_SUBPICTURE_LANG_EXT LanguageExtension;
  } DVD_SubpictureAttributes;

  typedef enum tagDVD_TITLE_APPMODE {
    DVD_AppMode_Not_Specified = 0,DVD_AppMode_Karaoke = 1,DVD_AppMode_Other = 3
  } DVD_TITLE_APPMODE;

  typedef struct tagDVD_TitleMainAttributes {
    DVD_TITLE_APPMODE AppMode;
    DVD_VideoAttributes VideoAttributes;
    ULONG ulNumberOfAudioStreams;
    DVD_AudioAttributes AudioAttributes[8];
    DVD_MultichannelAudioAttributes MultichannelAudioAttributes[8];
    ULONG ulNumberOfSubpictureStreams;
    DVD_SubpictureAttributes SubpictureAttributes[32];
  } DVD_TitleAttributes;

  typedef struct tagDVD_MenuAttributes {
    WINBOOL fCompatibleRegion[8];
    DVD_VideoAttributes VideoAttributes;
    WINBOOL fAudioPresent;
    DVD_AudioAttributes AudioAttributes;
    WINBOOL fSubpicturePresent;
    DVD_SubpictureAttributes SubpictureAttributes;
  } DVD_MenuAttributes;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0376_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0376_v0_0_s_ifspec;
#ifndef __IDvdControl_INTERFACE_DEFINED__
#define __IDvdControl_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IDvdControl;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IDvdControl : public IUnknown {
  public:
    virtual HRESULT WINAPI TitlePlay(ULONG ulTitle) = 0;
    virtual HRESULT WINAPI ChapterPlay(ULONG ulTitle,ULONG ulChapter) = 0;
    virtual HRESULT WINAPI TimePlay(ULONG ulTitle,ULONG bcdTime) = 0;
    virtual HRESULT WINAPI StopForResume(void) = 0;
    virtual HRESULT WINAPI GoUp(void) = 0;
    virtual HRESULT WINAPI TimeSearch(ULONG bcdTime) = 0;
    virtual HRESULT WINAPI ChapterSearch(ULONG ulChapter) = 0;
    virtual HRESULT WINAPI PrevPGSearch(void) = 0;
    virtual HRESULT WINAPI TopPGSearch(void) = 0;
    virtual HRESULT WINAPI NextPGSearch(void) = 0;
    virtual HRESULT WINAPI ForwardScan(double dwSpeed) = 0;
    virtual HRESULT WINAPI BackwardScan(double dwSpeed) = 0;
    virtual HRESULT WINAPI MenuCall(DVD_MENU_ID MenuID) = 0;
    virtual HRESULT WINAPI Resume(void) = 0;
    virtual HRESULT WINAPI UpperButtonSelect(void) = 0;
    virtual HRESULT WINAPI LowerButtonSelect(void) = 0;
    virtual HRESULT WINAPI LeftButtonSelect(void) = 0;
    virtual HRESULT WINAPI RightButtonSelect(void) = 0;
    virtual HRESULT WINAPI ButtonActivate(void) = 0;
    virtual HRESULT WINAPI ButtonSelectAndActivate(ULONG ulButton) = 0;
    virtual HRESULT WINAPI StillOff(void) = 0;
    virtual HRESULT WINAPI PauseOn(void) = 0;
    virtual HRESULT WINAPI PauseOff(void) = 0;
    virtual HRESULT WINAPI MenuLanguageSelect(LCID Language) = 0;
    virtual HRESULT WINAPI AudioStreamChange(ULONG ulAudio) = 0;
    virtual HRESULT WINAPI SubpictureStreamChange(ULONG ulSubPicture,WINBOOL bDisplay) = 0;
    virtual HRESULT WINAPI AngleChange(ULONG ulAngle) = 0;
    virtual HRESULT WINAPI ParentalLevelSelect(ULONG ulParentalLevel) = 0;
    virtual HRESULT WINAPI ParentalCountrySelect(WORD wCountry) = 0;
    virtual HRESULT WINAPI KaraokeAudioPresentationModeChange(ULONG ulMode) = 0;
    virtual HRESULT WINAPI VideoModePreferrence(ULONG ulPreferredDisplayMode) = 0;
    virtual HRESULT WINAPI SetRoot(LPCWSTR pszPath) = 0;
    virtual HRESULT WINAPI MouseActivate(POINT point) = 0;
    virtual HRESULT WINAPI MouseSelect(POINT point) = 0;
    virtual HRESULT WINAPI ChapterPlayAutoStop(ULONG ulTitle,ULONG ulChapter,ULONG ulChaptersToPlay) = 0;
  };
#else
  typedef struct IDvdControlVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IDvdControl *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IDvdControl *This);
      ULONG (WINAPI *Release)(IDvdControl *This);
      HRESULT (WINAPI *TitlePlay)(IDvdControl *This,ULONG ulTitle);
      HRESULT (WINAPI *ChapterPlay)(IDvdControl *This,ULONG ulTitle,ULONG ulChapter);
      HRESULT (WINAPI *TimePlay)(IDvdControl *This,ULONG ulTitle,ULONG bcdTime);
      HRESULT (WINAPI *StopForResume)(IDvdControl *This);
      HRESULT (WINAPI *GoUp)(IDvdControl *This);
      HRESULT (WINAPI *TimeSearch)(IDvdControl *This,ULONG bcdTime);
      HRESULT (WINAPI *ChapterSearch)(IDvdControl *This,ULONG ulChapter);
      HRESULT (WINAPI *PrevPGSearch)(IDvdControl *This);
      HRESULT (WINAPI *TopPGSearch)(IDvdControl *This);
      HRESULT (WINAPI *NextPGSearch)(IDvdControl *This);
      HRESULT (WINAPI *ForwardScan)(IDvdControl *This,double dwSpeed);
      HRESULT (WINAPI *BackwardScan)(IDvdControl *This,double dwSpeed);
      HRESULT (WINAPI *MenuCall)(IDvdControl *This,DVD_MENU_ID MenuID);
      HRESULT (WINAPI *Resume)(IDvdControl *This);
      HRESULT (WINAPI *UpperButtonSelect)(IDvdControl *This);
      HRESULT (WINAPI *LowerButtonSelect)(IDvdControl *This);
      HRESULT (WINAPI *LeftButtonSelect)(IDvdControl *This);
      HRESULT (WINAPI *RightButtonSelect)(IDvdControl *This);
      HRESULT (WINAPI *ButtonActivate)(IDvdControl *This);
      HRESULT (WINAPI *ButtonSelectAndActivate)(IDvdControl *This,ULONG ulButton);
      HRESULT (WINAPI *StillOff)(IDvdControl *This);
      HRESULT (WINAPI *PauseOn)(IDvdControl *This);
      HRESULT (WINAPI *PauseOff)(IDvdControl *This);
      HRESULT (WINAPI *MenuLanguageSelect)(IDvdControl *This,LCID Language);
      HRESULT (WINAPI *AudioStreamChange)(IDvdControl *This,ULONG ulAudio);
      HRESULT (WINAPI *SubpictureStreamChange)(IDvdControl *This,ULONG ulSubPicture,WINBOOL bDisplay);
      HRESULT (WINAPI *AngleChange)(IDvdControl *This,ULONG ulAngle);
      HRESULT (WINAPI *ParentalLevelSelect)(IDvdControl *This,ULONG ulParentalLevel);
      HRESULT (WINAPI *ParentalCountrySelect)(IDvdControl *This,WORD wCountry);
      HRESULT (WINAPI *KaraokeAudioPresentationModeChange)(IDvdControl *This,ULONG ulMode);
      HRESULT (WINAPI *VideoModePreferrence)(IDvdControl *This,ULONG ulPreferredDisplayMode);
      HRESULT (WINAPI *SetRoot)(IDvdControl *This,LPCWSTR pszPath);
      HRESULT (WINAPI *MouseActivate)(IDvdControl *This,POINT point);
      HRESULT (WINAPI *MouseSelect)(IDvdControl *This,POINT point);
      HRESULT (WINAPI *ChapterPlayAutoStop)(IDvdControl *This,ULONG ulTitle,ULONG ulChapter,ULONG ulChaptersToPlay);
    END_INTERFACE
  } IDvdControlVtbl;
  struct IDvdControl {
    CONST_VTBL struct IDvdControlVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IDvdControl_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDvdControl_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDvdControl_Release(This) (This)->lpVtbl->Release(This)
#define IDvdControl_TitlePlay(This,ulTitle) (This)->lpVtbl->TitlePlay(This,ulTitle)
#define IDvdControl_ChapterPlay(This,ulTitle,ulChapter) (This)->lpVtbl->ChapterPlay(This,ulTitle,ulChapter)
#define IDvdControl_TimePlay(This,ulTitle,bcdTime) (This)->lpVtbl->TimePlay(This,ulTitle,bcdTime)
#define IDvdControl_StopForResume(This) (This)->lpVtbl->StopForResume(This)
#define IDvdControl_GoUp(This) (This)->lpVtbl->GoUp(This)
#define IDvdControl_TimeSearch(This,bcdTime) (This)->lpVtbl->TimeSearch(This,bcdTime)
#define IDvdControl_ChapterSearch(This,ulChapter) (This)->lpVtbl->ChapterSearch(This,ulChapter)
#define IDvdControl_PrevPGSearch(This) (This)->lpVtbl->PrevPGSearch(This)
#define IDvdControl_TopPGSearch(This) (This)->lpVtbl->TopPGSearch(This)
#define IDvdControl_NextPGSearch(This) (This)->lpVtbl->NextPGSearch(This)
#define IDvdControl_ForwardScan(This,dwSpeed) (This)->lpVtbl->ForwardScan(This,dwSpeed)
#define IDvdControl_BackwardScan(This,dwSpeed) (This)->lpVtbl->BackwardScan(This,dwSpeed)
#define IDvdControl_MenuCall(This,MenuID) (This)->lpVtbl->MenuCall(This,MenuID)
#define IDvdControl_Resume(This) (This)->lpVtbl->Resume(This)
#define IDvdControl_UpperButtonSelect(This) (This)->lpVtbl->UpperButtonSelect(This)
#define IDvdControl_LowerButtonSelect(This) (This)->lpVtbl->LowerButtonSelect(This)
#define IDvdControl_LeftButtonSelect(This) (This)->lpVtbl->LeftButtonSelect(This)
#define IDvdControl_RightButtonSelect(This) (This)->lpVtbl->RightButtonSelect(This)
#define IDvdControl_ButtonActivate(This) (This)->lpVtbl->ButtonActivate(This)
#define IDvdControl_ButtonSelectAndActivate(This,ulButton) (This)->lpVtbl->ButtonSelectAndActivate(This,ulButton)
#define IDvdControl_StillOff(This) (This)->lpVtbl->StillOff(This)
#define IDvdControl_PauseOn(This) (This)->lpVtbl->PauseOn(This)
#define IDvdControl_PauseOff(This) (This)->lpVtbl->PauseOff(This)
#define IDvdControl_MenuLanguageSelect(This,Language) (This)->lpVtbl->MenuLanguageSelect(This,Language)
#define IDvdControl_AudioStreamChange(This,ulAudio) (This)->lpVtbl->AudioStreamChange(This,ulAudio)
#define IDvdControl_SubpictureStreamChange(This,ulSubPicture,bDisplay) (This)->lpVtbl->SubpictureStreamChange(This,ulSubPicture,bDisplay)
#define IDvdControl_AngleChange(This,ulAngle) (This)->lpVtbl->AngleChange(This,ulAngle)
#define IDvdControl_ParentalLevelSelect(This,ulParentalLevel) (This)->lpVtbl->ParentalLevelSelect(This,ulParentalLevel)
#define IDvdControl_ParentalCountrySelect(This,wCountry) (This)->lpVtbl->ParentalCountrySelect(This,wCountry)
#define IDvdControl_KaraokeAudioPresentationModeChange(This,ulMode) (This)->lpVtbl->KaraokeAudioPresentationModeChange(This,ulMode)
#define IDvdControl_VideoModePreferrence(This,ulPreferredDisplayMode) (This)->lpVtbl->VideoModePreferrence(This,ulPreferredDisplayMode)
#define IDvdControl_SetRoot(This,pszPath) (This)->lpVtbl->SetRoot(This,pszPath)
#define IDvdControl_MouseActivate(This,point) (This)->lpVtbl->MouseActivate(This,point)
#define IDvdControl_MouseSelect(This,point) (This)->lpVtbl->MouseSelect(This,point)
#define IDvdControl_ChapterPlayAutoStop(This,ulTitle,ulChapter,ulChaptersToPlay) (This)->lpVtbl->ChapterPlayAutoStop(This,ulTitle,ulChapter,ulChaptersToPlay)
#endif
#endif
  HRESULT WINAPI IDvdControl_TitlePlay_Proxy(IDvdControl *This,ULONG ulTitle);
  void __RPC_STUB IDvdControl_TitlePlay_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_ChapterPlay_Proxy(IDvdControl *This,ULONG ulTitle,ULONG ulChapter);
  void __RPC_STUB IDvdControl_ChapterPlay_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_TimePlay_Proxy(IDvdControl *This,ULONG ulTitle,ULONG bcdTime);
  void __RPC_STUB IDvdControl_TimePlay_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_StopForResume_Proxy(IDvdControl *This);
  void __RPC_STUB IDvdControl_StopForResume_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_GoUp_Proxy(IDvdControl *This);
  void __RPC_STUB IDvdControl_GoUp_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_TimeSearch_Proxy(IDvdControl *This,ULONG bcdTime);
  void __RPC_STUB IDvdControl_TimeSearch_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_ChapterSearch_Proxy(IDvdControl *This,ULONG ulChapter);
  void __RPC_STUB IDvdControl_ChapterSearch_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_PrevPGSearch_Proxy(IDvdControl *This);
  void __RPC_STUB IDvdControl_PrevPGSearch_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_TopPGSearch_Proxy(IDvdControl *This);
  void __RPC_STUB IDvdControl_TopPGSearch_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_NextPGSearch_Proxy(IDvdControl *This);
  void __RPC_STUB IDvdControl_NextPGSearch_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_ForwardScan_Proxy(IDvdControl *This,double dwSpeed);
  void __RPC_STUB IDvdControl_ForwardScan_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_BackwardScan_Proxy(IDvdControl *This,double dwSpeed);
  void __RPC_STUB IDvdControl_BackwardScan_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_MenuCall_Proxy(IDvdControl *This,DVD_MENU_ID MenuID);
  void __RPC_STUB IDvdControl_MenuCall_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_Resume_Proxy(IDvdControl *This);
  void __RPC_STUB IDvdControl_Resume_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_UpperButtonSelect_Proxy(IDvdControl *This);
  void __RPC_STUB IDvdControl_UpperButtonSelect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_LowerButtonSelect_Proxy(IDvdControl *This);
  void __RPC_STUB IDvdControl_LowerButtonSelect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_LeftButtonSelect_Proxy(IDvdControl *This);
  void __RPC_STUB IDvdControl_LeftButtonSelect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_RightButtonSelect_Proxy(IDvdControl *This);
  void __RPC_STUB IDvdControl_RightButtonSelect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_ButtonActivate_Proxy(IDvdControl *This);
  void __RPC_STUB IDvdControl_ButtonActivate_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_ButtonSelectAndActivate_Proxy(IDvdControl *This,ULONG ulButton);
  void __RPC_STUB IDvdControl_ButtonSelectAndActivate_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_StillOff_Proxy(IDvdControl *This);
  void __RPC_STUB IDvdControl_StillOff_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_PauseOn_Proxy(IDvdControl *This);
  void __RPC_STUB IDvdControl_PauseOn_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_PauseOff_Proxy(IDvdControl *This);
  void __RPC_STUB IDvdControl_PauseOff_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_MenuLanguageSelect_Proxy(IDvdControl *This,LCID Language);
  void __RPC_STUB IDvdControl_MenuLanguageSelect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_AudioStreamChange_Proxy(IDvdControl *This,ULONG ulAudio);
  void __RPC_STUB IDvdControl_AudioStreamChange_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_SubpictureStreamChange_Proxy(IDvdControl *This,ULONG ulSubPicture,WINBOOL bDisplay);
  void __RPC_STUB IDvdControl_SubpictureStreamChange_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_AngleChange_Proxy(IDvdControl *This,ULONG ulAngle);
  void __RPC_STUB IDvdControl_AngleChange_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_ParentalLevelSelect_Proxy(IDvdControl *This,ULONG ulParentalLevel);
  void __RPC_STUB IDvdControl_ParentalLevelSelect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_ParentalCountrySelect_Proxy(IDvdControl *This,WORD wCountry);
  void __RPC_STUB IDvdControl_ParentalCountrySelect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_KaraokeAudioPresentationModeChange_Proxy(IDvdControl *This,ULONG ulMode);
  void __RPC_STUB IDvdControl_KaraokeAudioPresentationModeChange_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_VideoModePreferrence_Proxy(IDvdControl *This,ULONG ulPreferredDisplayMode);
  void __RPC_STUB IDvdControl_VideoModePreferrence_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_SetRoot_Proxy(IDvdControl *This,LPCWSTR pszPath);
  void __RPC_STUB IDvdControl_SetRoot_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_MouseActivate_Proxy(IDvdControl *This,POINT point);
  void __RPC_STUB IDvdControl_MouseActivate_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_MouseSelect_Proxy(IDvdControl *This,POINT point);
  void __RPC_STUB IDvdControl_MouseSelect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl_ChapterPlayAutoStop_Proxy(IDvdControl *This,ULONG ulTitle,ULONG ulChapter,ULONG ulChaptersToPlay);
  void __RPC_STUB IDvdControl_ChapterPlayAutoStop_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IDvdInfo_INTERFACE_DEFINED__
#define __IDvdInfo_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IDvdInfo;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IDvdInfo : public IUnknown {
  public:
    virtual HRESULT WINAPI GetCurrentDomain(DVD_DOMAIN *pDomain) = 0;
    virtual HRESULT WINAPI GetCurrentLocation(DVD_PLAYBACK_LOCATION *pLocation) = 0;
    virtual HRESULT WINAPI GetTotalTitleTime(ULONG *pulTotalTime) = 0;
    virtual HRESULT WINAPI GetCurrentButton(ULONG *pulButtonsAvailable,ULONG *pulCurrentButton) = 0;
    virtual HRESULT WINAPI GetCurrentAngle(ULONG *pulAnglesAvailable,ULONG *pulCurrentAngle) = 0;
    virtual HRESULT WINAPI GetCurrentAudio(ULONG *pulStreamsAvailable,ULONG *pulCurrentStream) = 0;
    virtual HRESULT WINAPI GetCurrentSubpicture(ULONG *pulStreamsAvailable,ULONG *pulCurrentStream,WINBOOL *pIsDisabled) = 0;
    virtual HRESULT WINAPI GetCurrentUOPS(VALID_UOP_SOMTHING_OR_OTHER *pUOP) = 0;
    virtual HRESULT WINAPI GetAllSPRMs(SPRMARRAY *pRegisterArray) = 0;
    virtual HRESULT WINAPI GetAllGPRMs(GPRMARRAY *pRegisterArray) = 0;
    virtual HRESULT WINAPI GetAudioLanguage(ULONG ulStream,LCID *pLanguage) = 0;
    virtual HRESULT WINAPI GetSubpictureLanguage(ULONG ulStream,LCID *pLanguage) = 0;
    virtual HRESULT WINAPI GetTitleAttributes(ULONG ulTitle,DVD_ATR *pATR) = 0;
    virtual HRESULT WINAPI GetVMGAttributes(DVD_ATR *pATR) = 0;
    virtual HRESULT WINAPI GetCurrentVideoAttributes(DVD_VideoATR *pATR) = 0;
    virtual HRESULT WINAPI GetCurrentAudioAttributes(DVD_AudioATR *pATR) = 0;
    virtual HRESULT WINAPI GetCurrentSubpictureAttributes(DVD_SubpictureATR *pATR) = 0;
    virtual HRESULT WINAPI GetCurrentVolumeInfo(ULONG *pulNumOfVol,ULONG *pulThisVolNum,DVD_DISC_SIDE *pSide,ULONG *pulNumOfTitles) = 0;
    virtual HRESULT WINAPI GetDVDTextInfo(BYTE *pTextManager,ULONG ulBufSize,ULONG *pulActualSize) = 0;
    virtual HRESULT WINAPI GetPlayerParentalLevel(ULONG *pulParentalLevel,ULONG *pulCountryCode) = 0;
    virtual HRESULT WINAPI GetNumberOfChapters(ULONG ulTitle,ULONG *pulNumberOfChapters) = 0;
    virtual HRESULT WINAPI GetTitleParentalLevels(ULONG ulTitle,ULONG *pulParentalLevels) = 0;
    virtual HRESULT WINAPI GetRoot(LPSTR pRoot,ULONG ulBufSize,ULONG *pulActualSize) = 0;
  };
#else
  typedef struct IDvdInfoVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IDvdInfo *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IDvdInfo *This);
      ULONG (WINAPI *Release)(IDvdInfo *This);
      HRESULT (WINAPI *GetCurrentDomain)(IDvdInfo *This,DVD_DOMAIN *pDomain);
      HRESULT (WINAPI *GetCurrentLocation)(IDvdInfo *This,DVD_PLAYBACK_LOCATION *pLocation);
      HRESULT (WINAPI *GetTotalTitleTime)(IDvdInfo *This,ULONG *pulTotalTime);
      HRESULT (WINAPI *GetCurrentButton)(IDvdInfo *This,ULONG *pulButtonsAvailable,ULONG *pulCurrentButton);
      HRESULT (WINAPI *GetCurrentAngle)(IDvdInfo *This,ULONG *pulAnglesAvailable,ULONG *pulCurrentAngle);
      HRESULT (WINAPI *GetCurrentAudio)(IDvdInfo *This,ULONG *pulStreamsAvailable,ULONG *pulCurrentStream);
      HRESULT (WINAPI *GetCurrentSubpicture)(IDvdInfo *This,ULONG *pulStreamsAvailable,ULONG *pulCurrentStream,WINBOOL *pIsDisabled);
      HRESULT (WINAPI *GetCurrentUOPS)(IDvdInfo *This,VALID_UOP_SOMTHING_OR_OTHER *pUOP);
      HRESULT (WINAPI *GetAllSPRMs)(IDvdInfo *This,SPRMARRAY *pRegisterArray);
      HRESULT (WINAPI *GetAllGPRMs)(IDvdInfo *This,GPRMARRAY *pRegisterArray);
      HRESULT (WINAPI *GetAudioLanguage)(IDvdInfo *This,ULONG ulStream,LCID *pLanguage);
      HRESULT (WINAPI *GetSubpictureLanguage)(IDvdInfo *This,ULONG ulStream,LCID *pLanguage);
      HRESULT (WINAPI *GetTitleAttributes)(IDvdInfo *This,ULONG ulTitle,DVD_ATR *pATR);
      HRESULT (WINAPI *GetVMGAttributes)(IDvdInfo *This,DVD_ATR *pATR);
      HRESULT (WINAPI *GetCurrentVideoAttributes)(IDvdInfo *This,DVD_VideoATR *pATR);
      HRESULT (WINAPI *GetCurrentAudioAttributes)(IDvdInfo *This,DVD_AudioATR *pATR);
      HRESULT (WINAPI *GetCurrentSubpictureAttributes)(IDvdInfo *This,DVD_SubpictureATR *pATR);
      HRESULT (WINAPI *GetCurrentVolumeInfo)(IDvdInfo *This,ULONG *pulNumOfVol,ULONG *pulThisVolNum,DVD_DISC_SIDE *pSide,ULONG *pulNumOfTitles);
      HRESULT (WINAPI *GetDVDTextInfo)(IDvdInfo *This,BYTE *pTextManager,ULONG ulBufSize,ULONG *pulActualSize);
      HRESULT (WINAPI *GetPlayerParentalLevel)(IDvdInfo *This,ULONG *pulParentalLevel,ULONG *pulCountryCode);
      HRESULT (WINAPI *GetNumberOfChapters)(IDvdInfo *This,ULONG ulTitle,ULONG *pulNumberOfChapters);
      HRESULT (WINAPI *GetTitleParentalLevels)(IDvdInfo *This,ULONG ulTitle,ULONG *pulParentalLevels);
      HRESULT (WINAPI *GetRoot)(IDvdInfo *This,LPSTR pRoot,ULONG ulBufSize,ULONG *pulActualSize);
    END_INTERFACE
  } IDvdInfoVtbl;
  struct IDvdInfo {
    CONST_VTBL struct IDvdInfoVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IDvdInfo_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDvdInfo_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDvdInfo_Release(This) (This)->lpVtbl->Release(This)
#define IDvdInfo_GetCurrentDomain(This,pDomain) (This)->lpVtbl->GetCurrentDomain(This,pDomain)
#define IDvdInfo_GetCurrentLocation(This,pLocation) (This)->lpVtbl->GetCurrentLocation(This,pLocation)
#define IDvdInfo_GetTotalTitleTime(This,pulTotalTime) (This)->lpVtbl->GetTotalTitleTime(This,pulTotalTime)
#define IDvdInfo_GetCurrentButton(This,pulButtonsAvailable,pulCurrentButton) (This)->lpVtbl->GetCurrentButton(This,pulButtonsAvailable,pulCurrentButton)
#define IDvdInfo_GetCurrentAngle(This,pulAnglesAvailable,pulCurrentAngle) (This)->lpVtbl->GetCurrentAngle(This,pulAnglesAvailable,pulCurrentAngle)
#define IDvdInfo_GetCurrentAudio(This,pulStreamsAvailable,pulCurrentStream) (This)->lpVtbl->GetCurrentAudio(This,pulStreamsAvailable,pulCurrentStream)
#define IDvdInfo_GetCurrentSubpicture(This,pulStreamsAvailable,pulCurrentStream,pIsDisabled) (This)->lpVtbl->GetCurrentSubpicture(This,pulStreamsAvailable,pulCurrentStream,pIsDisabled)
#define IDvdInfo_GetCurrentUOPS(This,pUOP) (This)->lpVtbl->GetCurrentUOPS(This,pUOP)
#define IDvdInfo_GetAllSPRMs(This,pRegisterArray) (This)->lpVtbl->GetAllSPRMs(This,pRegisterArray)
#define IDvdInfo_GetAllGPRMs(This,pRegisterArray) (This)->lpVtbl->GetAllGPRMs(This,pRegisterArray)
#define IDvdInfo_GetAudioLanguage(This,ulStream,pLanguage) (This)->lpVtbl->GetAudioLanguage(This,ulStream,pLanguage)
#define IDvdInfo_GetSubpictureLanguage(This,ulStream,pLanguage) (This)->lpVtbl->GetSubpictureLanguage(This,ulStream,pLanguage)
#define IDvdInfo_GetTitleAttributes(This,ulTitle,pATR) (This)->lpVtbl->GetTitleAttributes(This,ulTitle,pATR)
#define IDvdInfo_GetVMGAttributes(This,pATR) (This)->lpVtbl->GetVMGAttributes(This,pATR)
#define IDvdInfo_GetCurrentVideoAttributes(This,pATR) (This)->lpVtbl->GetCurrentVideoAttributes(This,pATR)
#define IDvdInfo_GetCurrentAudioAttributes(This,pATR) (This)->lpVtbl->GetCurrentAudioAttributes(This,pATR)
#define IDvdInfo_GetCurrentSubpictureAttributes(This,pATR) (This)->lpVtbl->GetCurrentSubpictureAttributes(This,pATR)
#define IDvdInfo_GetCurrentVolumeInfo(This,pulNumOfVol,pulThisVolNum,pSide,pulNumOfTitles) (This)->lpVtbl->GetCurrentVolumeInfo(This,pulNumOfVol,pulThisVolNum,pSide,pulNumOfTitles)
#define IDvdInfo_GetDVDTextInfo(This,pTextManager,ulBufSize,pulActualSize) (This)->lpVtbl->GetDVDTextInfo(This,pTextManager,ulBufSize,pulActualSize)
#define IDvdInfo_GetPlayerParentalLevel(This,pulParentalLevel,pulCountryCode) (This)->lpVtbl->GetPlayerParentalLevel(This,pulParentalLevel,pulCountryCode)
#define IDvdInfo_GetNumberOfChapters(This,ulTitle,pulNumberOfChapters) (This)->lpVtbl->GetNumberOfChapters(This,ulTitle,pulNumberOfChapters)
#define IDvdInfo_GetTitleParentalLevels(This,ulTitle,pulParentalLevels) (This)->lpVtbl->GetTitleParentalLevels(This,ulTitle,pulParentalLevels)
#define IDvdInfo_GetRoot(This,pRoot,ulBufSize,pulActualSize) (This)->lpVtbl->GetRoot(This,pRoot,ulBufSize,pulActualSize)
#endif
#endif
  HRESULT WINAPI IDvdInfo_GetCurrentDomain_Proxy(IDvdInfo *This,DVD_DOMAIN *pDomain);
  void __RPC_STUB IDvdInfo_GetCurrentDomain_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetCurrentLocation_Proxy(IDvdInfo *This,DVD_PLAYBACK_LOCATION *pLocation);
  void __RPC_STUB IDvdInfo_GetCurrentLocation_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetTotalTitleTime_Proxy(IDvdInfo *This,ULONG *pulTotalTime);
  void __RPC_STUB IDvdInfo_GetTotalTitleTime_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetCurrentButton_Proxy(IDvdInfo *This,ULONG *pulButtonsAvailable,ULONG *pulCurrentButton);
  void __RPC_STUB IDvdInfo_GetCurrentButton_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetCurrentAngle_Proxy(IDvdInfo *This,ULONG *pulAnglesAvailable,ULONG *pulCurrentAngle);
  void __RPC_STUB IDvdInfo_GetCurrentAngle_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetCurrentAudio_Proxy(IDvdInfo *This,ULONG *pulStreamsAvailable,ULONG *pulCurrentStream);
  void __RPC_STUB IDvdInfo_GetCurrentAudio_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetCurrentSubpicture_Proxy(IDvdInfo *This,ULONG *pulStreamsAvailable,ULONG *pulCurrentStream,WINBOOL *pIsDisabled);
  void __RPC_STUB IDvdInfo_GetCurrentSubpicture_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetCurrentUOPS_Proxy(IDvdInfo *This,VALID_UOP_SOMTHING_OR_OTHER *pUOP);
  void __RPC_STUB IDvdInfo_GetCurrentUOPS_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetAllSPRMs_Proxy(IDvdInfo *This,SPRMARRAY *pRegisterArray);
  void __RPC_STUB IDvdInfo_GetAllSPRMs_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetAllGPRMs_Proxy(IDvdInfo *This,GPRMARRAY *pRegisterArray);
  void __RPC_STUB IDvdInfo_GetAllGPRMs_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetAudioLanguage_Proxy(IDvdInfo *This,ULONG ulStream,LCID *pLanguage);
  void __RPC_STUB IDvdInfo_GetAudioLanguage_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetSubpictureLanguage_Proxy(IDvdInfo *This,ULONG ulStream,LCID *pLanguage);
  void __RPC_STUB IDvdInfo_GetSubpictureLanguage_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetTitleAttributes_Proxy(IDvdInfo *This,ULONG ulTitle,DVD_ATR *pATR);
  void __RPC_STUB IDvdInfo_GetTitleAttributes_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetVMGAttributes_Proxy(IDvdInfo *This,DVD_ATR *pATR);
  void __RPC_STUB IDvdInfo_GetVMGAttributes_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetCurrentVideoAttributes_Proxy(IDvdInfo *This,DVD_VideoATR *pATR);
  void __RPC_STUB IDvdInfo_GetCurrentVideoAttributes_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetCurrentAudioAttributes_Proxy(IDvdInfo *This,DVD_AudioATR *pATR);
  void __RPC_STUB IDvdInfo_GetCurrentAudioAttributes_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetCurrentSubpictureAttributes_Proxy(IDvdInfo *This,DVD_SubpictureATR *pATR);
  void __RPC_STUB IDvdInfo_GetCurrentSubpictureAttributes_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetCurrentVolumeInfo_Proxy(IDvdInfo *This,ULONG *pulNumOfVol,ULONG *pulThisVolNum,DVD_DISC_SIDE *pSide,ULONG *pulNumOfTitles);
  void __RPC_STUB IDvdInfo_GetCurrentVolumeInfo_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetDVDTextInfo_Proxy(IDvdInfo *This,BYTE *pTextManager,ULONG ulBufSize,ULONG *pulActualSize);
  void __RPC_STUB IDvdInfo_GetDVDTextInfo_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetPlayerParentalLevel_Proxy(IDvdInfo *This,ULONG *pulParentalLevel,ULONG *pulCountryCode);
  void __RPC_STUB IDvdInfo_GetPlayerParentalLevel_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetNumberOfChapters_Proxy(IDvdInfo *This,ULONG ulTitle,ULONG *pulNumberOfChapters);
  void __RPC_STUB IDvdInfo_GetNumberOfChapters_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetTitleParentalLevels_Proxy(IDvdInfo *This,ULONG ulTitle,ULONG *pulParentalLevels);
  void __RPC_STUB IDvdInfo_GetTitleParentalLevels_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo_GetRoot_Proxy(IDvdInfo *This,LPSTR pRoot,ULONG ulBufSize,ULONG *pulActualSize);
  void __RPC_STUB IDvdInfo_GetRoot_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IDvdCmd_INTERFACE_DEFINED__
#define __IDvdCmd_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IDvdCmd;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IDvdCmd : public IUnknown {
  public:
    virtual HRESULT WINAPI WaitForStart(void) = 0;
    virtual HRESULT WINAPI WaitForEnd(void) = 0;
  };
#else
  typedef struct IDvdCmdVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IDvdCmd *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IDvdCmd *This);
      ULONG (WINAPI *Release)(IDvdCmd *This);
      HRESULT (WINAPI *WaitForStart)(IDvdCmd *This);
      HRESULT (WINAPI *WaitForEnd)(IDvdCmd *This);
    END_INTERFACE
  } IDvdCmdVtbl;
  struct IDvdCmd {
    CONST_VTBL struct IDvdCmdVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IDvdCmd_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDvdCmd_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDvdCmd_Release(This) (This)->lpVtbl->Release(This)
#define IDvdCmd_WaitForStart(This) (This)->lpVtbl->WaitForStart(This)
#define IDvdCmd_WaitForEnd(This) (This)->lpVtbl->WaitForEnd(This)
#endif
#endif
  HRESULT WINAPI IDvdCmd_WaitForStart_Proxy(IDvdCmd *This);
  void __RPC_STUB IDvdCmd_WaitForStart_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdCmd_WaitForEnd_Proxy(IDvdCmd *This);
  void __RPC_STUB IDvdCmd_WaitForEnd_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IDvdState_INTERFACE_DEFINED__
#define __IDvdState_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IDvdState;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IDvdState : public IUnknown {
  public:
    virtual HRESULT WINAPI GetDiscID(ULONGLONG *pullUniqueID) = 0;
    virtual HRESULT WINAPI GetParentalLevel(ULONG *pulParentalLevel) = 0;
  };
#else
  typedef struct IDvdStateVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IDvdState *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IDvdState *This);
      ULONG (WINAPI *Release)(IDvdState *This);
      HRESULT (WINAPI *GetDiscID)(IDvdState *This,ULONGLONG *pullUniqueID);
      HRESULT (WINAPI *GetParentalLevel)(IDvdState *This,ULONG *pulParentalLevel);
    END_INTERFACE
  } IDvdStateVtbl;
  struct IDvdState {
    CONST_VTBL struct IDvdStateVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IDvdState_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDvdState_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDvdState_Release(This) (This)->lpVtbl->Release(This)
#define IDvdState_GetDiscID(This,pullUniqueID) (This)->lpVtbl->GetDiscID(This,pullUniqueID)
#define IDvdState_GetParentalLevel(This,pulParentalLevel) (This)->lpVtbl->GetParentalLevel(This,pulParentalLevel)
#endif
#endif
  HRESULT WINAPI IDvdState_GetDiscID_Proxy(IDvdState *This,ULONGLONG *pullUniqueID);
  void __RPC_STUB IDvdState_GetDiscID_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdState_GetParentalLevel_Proxy(IDvdState *This,ULONG *pulParentalLevel);
  void __RPC_STUB IDvdState_GetParentalLevel_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IDvdControl2_INTERFACE_DEFINED__
#define __IDvdControl2_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IDvdControl2;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IDvdControl2 : public IUnknown {
  public:
    virtual HRESULT WINAPI PlayTitle(ULONG ulTitle,DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI PlayChapterInTitle(ULONG ulTitle,ULONG ulChapter,DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI PlayAtTimeInTitle(ULONG ulTitle,DVD_HMSF_TIMECODE *pStartTime,DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI Stop(void) = 0;
    virtual HRESULT WINAPI ReturnFromSubmenu(DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI PlayAtTime(DVD_HMSF_TIMECODE *pTime,DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI PlayChapter(ULONG ulChapter,DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI PlayPrevChapter(DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI ReplayChapter(DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI PlayNextChapter(DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI PlayForwards(double dSpeed,DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI PlayBackwards(double dSpeed,DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI ShowMenu(DVD_MENU_ID MenuID,DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI Resume(DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI SelectRelativeButton(DVD_RELATIVE_BUTTON buttonDir) = 0;
    virtual HRESULT WINAPI ActivateButton(void) = 0;
    virtual HRESULT WINAPI SelectButton(ULONG ulButton) = 0;
    virtual HRESULT WINAPI SelectAndActivateButton(ULONG ulButton) = 0;
    virtual HRESULT WINAPI StillOff(void) = 0;
    virtual HRESULT WINAPI Pause(WINBOOL bState) = 0;
    virtual HRESULT WINAPI SelectAudioStream(ULONG ulAudio,DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI SelectSubpictureStream(ULONG ulSubPicture,DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI SetSubpictureState(WINBOOL bState,DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI SelectAngle(ULONG ulAngle,DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI SelectParentalLevel(ULONG ulParentalLevel) = 0;
    virtual HRESULT WINAPI SelectParentalCountry(BYTE bCountry[2]) = 0;
    virtual HRESULT WINAPI SelectKaraokeAudioPresentationMode(ULONG ulMode) = 0;
    virtual HRESULT WINAPI SelectVideoModePreference(ULONG ulPreferredDisplayMode) = 0;
    virtual HRESULT WINAPI SetDVDDirectory(LPCWSTR pszwPath) = 0;
    virtual HRESULT WINAPI ActivateAtPosition(POINT point) = 0;
    virtual HRESULT WINAPI SelectAtPosition(POINT point) = 0;
    virtual HRESULT WINAPI PlayChaptersAutoStop(ULONG ulTitle,ULONG ulChapter,ULONG ulChaptersToPlay,DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI AcceptParentalLevelChange(WINBOOL bAccept) = 0;
    virtual HRESULT WINAPI SetOption(DVD_OPTION_FLAG flag,WINBOOL fState) = 0;
    virtual HRESULT WINAPI SetState(IDvdState *pState,DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI PlayPeriodInTitleAutoStop(ULONG ulTitle,DVD_HMSF_TIMECODE *pStartTime,DVD_HMSF_TIMECODE *pEndTime,DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI SetGPRM(ULONG ulIndex,WORD wValue,DWORD dwFlags,IDvdCmd **ppCmd) = 0;
    virtual HRESULT WINAPI SelectDefaultMenuLanguage(LCID Language) = 0;
    virtual HRESULT WINAPI SelectDefaultAudioLanguage(LCID Language,DVD_AUDIO_LANG_EXT audioExtension) = 0;
    virtual HRESULT WINAPI SelectDefaultSubpictureLanguage(LCID Language,DVD_SUBPICTURE_LANG_EXT subpictureExtension) = 0;
  };
#else
  typedef struct IDvdControl2Vtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IDvdControl2 *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IDvdControl2 *This);
      ULONG (WINAPI *Release)(IDvdControl2 *This);
      HRESULT (WINAPI *PlayTitle)(IDvdControl2 *This,ULONG ulTitle,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *PlayChapterInTitle)(IDvdControl2 *This,ULONG ulTitle,ULONG ulChapter,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *PlayAtTimeInTitle)(IDvdControl2 *This,ULONG ulTitle,DVD_HMSF_TIMECODE *pStartTime,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *Stop)(IDvdControl2 *This);
      HRESULT (WINAPI *ReturnFromSubmenu)(IDvdControl2 *This,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *PlayAtTime)(IDvdControl2 *This,DVD_HMSF_TIMECODE *pTime,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *PlayChapter)(IDvdControl2 *This,ULONG ulChapter,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *PlayPrevChapter)(IDvdControl2 *This,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *ReplayChapter)(IDvdControl2 *This,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *PlayNextChapter)(IDvdControl2 *This,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *PlayForwards)(IDvdControl2 *This,double dSpeed,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *PlayBackwards)(IDvdControl2 *This,double dSpeed,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *ShowMenu)(IDvdControl2 *This,DVD_MENU_ID MenuID,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *Resume)(IDvdControl2 *This,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *SelectRelativeButton)(IDvdControl2 *This,DVD_RELATIVE_BUTTON buttonDir);
      HRESULT (WINAPI *ActivateButton)(IDvdControl2 *This);
      HRESULT (WINAPI *SelectButton)(IDvdControl2 *This,ULONG ulButton);
      HRESULT (WINAPI *SelectAndActivateButton)(IDvdControl2 *This,ULONG ulButton);
      HRESULT (WINAPI *StillOff)(IDvdControl2 *This);
      HRESULT (WINAPI *Pause)(IDvdControl2 *This,WINBOOL bState);
      HRESULT (WINAPI *SelectAudioStream)(IDvdControl2 *This,ULONG ulAudio,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *SelectSubpictureStream)(IDvdControl2 *This,ULONG ulSubPicture,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *SetSubpictureState)(IDvdControl2 *This,WINBOOL bState,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *SelectAngle)(IDvdControl2 *This,ULONG ulAngle,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *SelectParentalLevel)(IDvdControl2 *This,ULONG ulParentalLevel);
      HRESULT (WINAPI *SelectParentalCountry)(IDvdControl2 *This,BYTE bCountry[2]);
      HRESULT (WINAPI *SelectKaraokeAudioPresentationMode)(IDvdControl2 *This,ULONG ulMode);
      HRESULT (WINAPI *SelectVideoModePreference)(IDvdControl2 *This,ULONG ulPreferredDisplayMode);
      HRESULT (WINAPI *SetDVDDirectory)(IDvdControl2 *This,LPCWSTR pszwPath);
      HRESULT (WINAPI *ActivateAtPosition)(IDvdControl2 *This,POINT point);
      HRESULT (WINAPI *SelectAtPosition)(IDvdControl2 *This,POINT point);
      HRESULT (WINAPI *PlayChaptersAutoStop)(IDvdControl2 *This,ULONG ulTitle,ULONG ulChapter,ULONG ulChaptersToPlay,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *AcceptParentalLevelChange)(IDvdControl2 *This,WINBOOL bAccept);
      HRESULT (WINAPI *SetOption)(IDvdControl2 *This,DVD_OPTION_FLAG flag,WINBOOL fState);
      HRESULT (WINAPI *SetState)(IDvdControl2 *This,IDvdState *pState,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *PlayPeriodInTitleAutoStop)(IDvdControl2 *This,ULONG ulTitle,DVD_HMSF_TIMECODE *pStartTime,DVD_HMSF_TIMECODE *pEndTime,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *SetGPRM)(IDvdControl2 *This,ULONG ulIndex,WORD wValue,DWORD dwFlags,IDvdCmd **ppCmd);
      HRESULT (WINAPI *SelectDefaultMenuLanguage)(IDvdControl2 *This,LCID Language);
      HRESULT (WINAPI *SelectDefaultAudioLanguage)(IDvdControl2 *This,LCID Language,DVD_AUDIO_LANG_EXT audioExtension);
      HRESULT (WINAPI *SelectDefaultSubpictureLanguage)(IDvdControl2 *This,LCID Language,DVD_SUBPICTURE_LANG_EXT subpictureExtension);
    END_INTERFACE
  } IDvdControl2Vtbl;
  struct IDvdControl2 {
    CONST_VTBL struct IDvdControl2Vtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IDvdControl2_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDvdControl2_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDvdControl2_Release(This) (This)->lpVtbl->Release(This)
#define IDvdControl2_PlayTitle(This,ulTitle,dwFlags,ppCmd) (This)->lpVtbl->PlayTitle(This,ulTitle,dwFlags,ppCmd)
#define IDvdControl2_PlayChapterInTitle(This,ulTitle,ulChapter,dwFlags,ppCmd) (This)->lpVtbl->PlayChapterInTitle(This,ulTitle,ulChapter,dwFlags,ppCmd)
#define IDvdControl2_PlayAtTimeInTitle(This,ulTitle,pStartTime,dwFlags,ppCmd) (This)->lpVtbl->PlayAtTimeInTitle(This,ulTitle,pStartTime,dwFlags,ppCmd)
#define IDvdControl2_Stop(This) (This)->lpVtbl->Stop(This)
#define IDvdControl2_ReturnFromSubmenu(This,dwFlags,ppCmd) (This)->lpVtbl->ReturnFromSubmenu(This,dwFlags,ppCmd)
#define IDvdControl2_PlayAtTime(This,pTime,dwFlags,ppCmd) (This)->lpVtbl->PlayAtTime(This,pTime,dwFlags,ppCmd)
#define IDvdControl2_PlayChapter(This,ulChapter,dwFlags,ppCmd) (This)->lpVtbl->PlayChapter(This,ulChapter,dwFlags,ppCmd)
#define IDvdControl2_PlayPrevChapter(This,dwFlags,ppCmd) (This)->lpVtbl->PlayPrevChapter(This,dwFlags,ppCmd)
#define IDvdControl2_ReplayChapter(This,dwFlags,ppCmd) (This)->lpVtbl->ReplayChapter(This,dwFlags,ppCmd)
#define IDvdControl2_PlayNextChapter(This,dwFlags,ppCmd) (This)->lpVtbl->PlayNextChapter(This,dwFlags,ppCmd)
#define IDvdControl2_PlayForwards(This,dSpeed,dwFlags,ppCmd) (This)->lpVtbl->PlayForwards(This,dSpeed,dwFlags,ppCmd)
#define IDvdControl2_PlayBackwards(This,dSpeed,dwFlags,ppCmd) (This)->lpVtbl->PlayBackwards(This,dSpeed,dwFlags,ppCmd)
#define IDvdControl2_ShowMenu(This,MenuID,dwFlags,ppCmd) (This)->lpVtbl->ShowMenu(This,MenuID,dwFlags,ppCmd)
#define IDvdControl2_Resume(This,dwFlags,ppCmd) (This)->lpVtbl->Resume(This,dwFlags,ppCmd)
#define IDvdControl2_SelectRelativeButton(This,buttonDir) (This)->lpVtbl->SelectRelativeButton(This,buttonDir)
#define IDvdControl2_ActivateButton(This) (This)->lpVtbl->ActivateButton(This)
#define IDvdControl2_SelectButton(This,ulButton) (This)->lpVtbl->SelectButton(This,ulButton)
#define IDvdControl2_SelectAndActivateButton(This,ulButton) (This)->lpVtbl->SelectAndActivateButton(This,ulButton)
#define IDvdControl2_StillOff(This) (This)->lpVtbl->StillOff(This)
#define IDvdControl2_Pause(This,bState) (This)->lpVtbl->Pause(This,bState)
#define IDvdControl2_SelectAudioStream(This,ulAudio,dwFlags,ppCmd) (This)->lpVtbl->SelectAudioStream(This,ulAudio,dwFlags,ppCmd)
#define IDvdControl2_SelectSubpictureStream(This,ulSubPicture,dwFlags,ppCmd) (This)->lpVtbl->SelectSubpictureStream(This,ulSubPicture,dwFlags,ppCmd)
#define IDvdControl2_SetSubpictureState(This,bState,dwFlags,ppCmd) (This)->lpVtbl->SetSubpictureState(This,bState,dwFlags,ppCmd)
#define IDvdControl2_SelectAngle(This,ulAngle,dwFlags,ppCmd) (This)->lpVtbl->SelectAngle(This,ulAngle,dwFlags,ppCmd)
#define IDvdControl2_SelectParentalLevel(This,ulParentalLevel) (This)->lpVtbl->SelectParentalLevel(This,ulParentalLevel)
#define IDvdControl2_SelectParentalCountry(This,bCountry) (This)->lpVtbl->SelectParentalCountry(This,bCountry)
#define IDvdControl2_SelectKaraokeAudioPresentationMode(This,ulMode) (This)->lpVtbl->SelectKaraokeAudioPresentationMode(This,ulMode)
#define IDvdControl2_SelectVideoModePreference(This,ulPreferredDisplayMode) (This)->lpVtbl->SelectVideoModePreference(This,ulPreferredDisplayMode)
#define IDvdControl2_SetDVDDirectory(This,pszwPath) (This)->lpVtbl->SetDVDDirectory(This,pszwPath)
#define IDvdControl2_ActivateAtPosition(This,point) (This)->lpVtbl->ActivateAtPosition(This,point)
#define IDvdControl2_SelectAtPosition(This,point) (This)->lpVtbl->SelectAtPosition(This,point)
#define IDvdControl2_PlayChaptersAutoStop(This,ulTitle,ulChapter,ulChaptersToPlay,dwFlags,ppCmd) (This)->lpVtbl->PlayChaptersAutoStop(This,ulTitle,ulChapter,ulChaptersToPlay,dwFlags,ppCmd)
#define IDvdControl2_AcceptParentalLevelChange(This,bAccept) (This)->lpVtbl->AcceptParentalLevelChange(This,bAccept)
#define IDvdControl2_SetOption(This,flag,fState) (This)->lpVtbl->SetOption(This,flag,fState)
#define IDvdControl2_SetState(This,pState,dwFlags,ppCmd) (This)->lpVtbl->SetState(This,pState,dwFlags,ppCmd)
#define IDvdControl2_PlayPeriodInTitleAutoStop(This,ulTitle,pStartTime,pEndTime,dwFlags,ppCmd) (This)->lpVtbl->PlayPeriodInTitleAutoStop(This,ulTitle,pStartTime,pEndTime,dwFlags,ppCmd)
#define IDvdControl2_SetGPRM(This,ulIndex,wValue,dwFlags,ppCmd) (This)->lpVtbl->SetGPRM(This,ulIndex,wValue,dwFlags,ppCmd)
#define IDvdControl2_SelectDefaultMenuLanguage(This,Language) (This)->lpVtbl->SelectDefaultMenuLanguage(This,Language)
#define IDvdControl2_SelectDefaultAudioLanguage(This,Language,audioExtension) (This)->lpVtbl->SelectDefaultAudioLanguage(This,Language,audioExtension)
#define IDvdControl2_SelectDefaultSubpictureLanguage(This,Language,subpictureExtension) (This)->lpVtbl->SelectDefaultSubpictureLanguage(This,Language,subpictureExtension)
#endif
#endif
  HRESULT WINAPI IDvdControl2_PlayTitle_Proxy(IDvdControl2 *This,ULONG ulTitle,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_PlayTitle_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_PlayChapterInTitle_Proxy(IDvdControl2 *This,ULONG ulTitle,ULONG ulChapter,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_PlayChapterInTitle_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_PlayAtTimeInTitle_Proxy(IDvdControl2 *This,ULONG ulTitle,DVD_HMSF_TIMECODE *pStartTime,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_PlayAtTimeInTitle_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_Stop_Proxy(IDvdControl2 *This);
  void __RPC_STUB IDvdControl2_Stop_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_ReturnFromSubmenu_Proxy(IDvdControl2 *This,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_ReturnFromSubmenu_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_PlayAtTime_Proxy(IDvdControl2 *This,DVD_HMSF_TIMECODE *pTime,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_PlayAtTime_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_PlayChapter_Proxy(IDvdControl2 *This,ULONG ulChapter,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_PlayChapter_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_PlayPrevChapter_Proxy(IDvdControl2 *This,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_PlayPrevChapter_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_ReplayChapter_Proxy(IDvdControl2 *This,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_ReplayChapter_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_PlayNextChapter_Proxy(IDvdControl2 *This,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_PlayNextChapter_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_PlayForwards_Proxy(IDvdControl2 *This,double dSpeed,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_PlayForwards_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_PlayBackwards_Proxy(IDvdControl2 *This,double dSpeed,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_PlayBackwards_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_ShowMenu_Proxy(IDvdControl2 *This,DVD_MENU_ID MenuID,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_ShowMenu_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_Resume_Proxy(IDvdControl2 *This,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_Resume_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_SelectRelativeButton_Proxy(IDvdControl2 *This,DVD_RELATIVE_BUTTON buttonDir);
  void __RPC_STUB IDvdControl2_SelectRelativeButton_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_ActivateButton_Proxy(IDvdControl2 *This);
  void __RPC_STUB IDvdControl2_ActivateButton_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_SelectButton_Proxy(IDvdControl2 *This,ULONG ulButton);
  void __RPC_STUB IDvdControl2_SelectButton_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_SelectAndActivateButton_Proxy(IDvdControl2 *This,ULONG ulButton);
  void __RPC_STUB IDvdControl2_SelectAndActivateButton_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_StillOff_Proxy(IDvdControl2 *This);
  void __RPC_STUB IDvdControl2_StillOff_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_Pause_Proxy(IDvdControl2 *This,WINBOOL bState);
  void __RPC_STUB IDvdControl2_Pause_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_SelectAudioStream_Proxy(IDvdControl2 *This,ULONG ulAudio,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_SelectAudioStream_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_SelectSubpictureStream_Proxy(IDvdControl2 *This,ULONG ulSubPicture,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_SelectSubpictureStream_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_SetSubpictureState_Proxy(IDvdControl2 *This,WINBOOL bState,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_SetSubpictureState_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_SelectAngle_Proxy(IDvdControl2 *This,ULONG ulAngle,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_SelectAngle_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_SelectParentalLevel_Proxy(IDvdControl2 *This,ULONG ulParentalLevel);
  void __RPC_STUB IDvdControl2_SelectParentalLevel_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_SelectParentalCountry_Proxy(IDvdControl2 *This,BYTE bCountry[2]);
  void __RPC_STUB IDvdControl2_SelectParentalCountry_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_SelectKaraokeAudioPresentationMode_Proxy(IDvdControl2 *This,ULONG ulMode);
  void __RPC_STUB IDvdControl2_SelectKaraokeAudioPresentationMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_SelectVideoModePreference_Proxy(IDvdControl2 *This,ULONG ulPreferredDisplayMode);
  void __RPC_STUB IDvdControl2_SelectVideoModePreference_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_SetDVDDirectory_Proxy(IDvdControl2 *This,LPCWSTR pszwPath);
  void __RPC_STUB IDvdControl2_SetDVDDirectory_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_ActivateAtPosition_Proxy(IDvdControl2 *This,POINT point);
  void __RPC_STUB IDvdControl2_ActivateAtPosition_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_SelectAtPosition_Proxy(IDvdControl2 *This,POINT point);
  void __RPC_STUB IDvdControl2_SelectAtPosition_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_PlayChaptersAutoStop_Proxy(IDvdControl2 *This,ULONG ulTitle,ULONG ulChapter,ULONG ulChaptersToPlay,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_PlayChaptersAutoStop_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_AcceptParentalLevelChange_Proxy(IDvdControl2 *This,WINBOOL bAccept);
  void __RPC_STUB IDvdControl2_AcceptParentalLevelChange_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_SetOption_Proxy(IDvdControl2 *This,DVD_OPTION_FLAG flag,WINBOOL fState);
  void __RPC_STUB IDvdControl2_SetOption_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_SetState_Proxy(IDvdControl2 *This,IDvdState *pState,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_SetState_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_PlayPeriodInTitleAutoStop_Proxy(IDvdControl2 *This,ULONG ulTitle,DVD_HMSF_TIMECODE *pStartTime,DVD_HMSF_TIMECODE *pEndTime,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_PlayPeriodInTitleAutoStop_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_SetGPRM_Proxy(IDvdControl2 *This,ULONG ulIndex,WORD wValue,DWORD dwFlags,IDvdCmd **ppCmd);
  void __RPC_STUB IDvdControl2_SetGPRM_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_SelectDefaultMenuLanguage_Proxy(IDvdControl2 *This,LCID Language);
  void __RPC_STUB IDvdControl2_SelectDefaultMenuLanguage_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_SelectDefaultAudioLanguage_Proxy(IDvdControl2 *This,LCID Language,DVD_AUDIO_LANG_EXT audioExtension);
  void __RPC_STUB IDvdControl2_SelectDefaultAudioLanguage_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdControl2_SelectDefaultSubpictureLanguage_Proxy(IDvdControl2 *This,LCID Language,DVD_SUBPICTURE_LANG_EXT subpictureExtension);
  void __RPC_STUB IDvdControl2_SelectDefaultSubpictureLanguage_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  enum DVD_TextStringType {
    DVD_Struct_Volume = 0x1,DVD_Struct_Title = 0x2,DVD_Struct_ParentalID = 0x3,DVD_Struct_PartOfTitle = 0x4,DVD_Struct_Cell = 0x5,
    DVD_Stream_Audio = 0x10,DVD_Stream_Subpicture = 0x11,DVD_Stream_Angle = 0x12,DVD_Channel_Audio = 0x20,DVD_General_Name = 0x30,
    DVD_General_Comments = 0x31,DVD_Title_Series = 0x38,DVD_Title_Movie = 0x39,DVD_Title_Video = 0x3a,DVD_Title_Album = 0x3b,DVD_Title_Song = 0x3c,
    DVD_Title_Other = 0x3f,DVD_Title_Sub_Series = 0x40,DVD_Title_Sub_Movie = 0x41,DVD_Title_Sub_Video = 0x42,DVD_Title_Sub_Album = 0x43,
    DVD_Title_Sub_Song = 0x44,DVD_Title_Sub_Other = 0x47,DVD_Title_Orig_Series = 0x48,DVD_Title_Orig_Movie = 0x49,DVD_Title_Orig_Video = 0x4a,
    DVD_Title_Orig_Album = 0x4b,DVD_Title_Orig_Song = 0x4c,DVD_Title_Orig_Other = 0x4f,DVD_Other_Scene = 0x50,DVD_Other_Cut = 0x51,DVD_Other_Take = 0x52
  };

  enum DVD_TextCharSet {
    DVD_CharSet_Unicode = 0,DVD_CharSet_ISO646 = 1,DVD_CharSet_JIS_Roman_Kanji = 2,DVD_CharSet_ISO8859_1 = 3,
    DVD_CharSet_ShiftJIS_Kanji_Roman_Katakana = 4
  };
#define DVD_TITLE_MENU 0x000
#define DVD_STREAM_DATA_CURRENT 0x800
#define DVD_STREAM_DATA_VMGM 0x400
#define DVD_STREAM_DATA_VTSM 0x401
#define DVD_DEFAULT_AUDIO_STREAM 0x0f

  typedef struct tagDVD_DECODER_CAPS {
    DWORD dwSize;
    DWORD dwAudioCaps;
    double dFwdMaxRateVideo;
    double dFwdMaxRateAudio;
    double dFwdMaxRateSP;
    double dBwdMaxRateVideo;
    double dBwdMaxRateAudio;
    double dBwdMaxRateSP;
    DWORD dwRes1;
    DWORD dwRes2;
    DWORD dwRes3;
    DWORD dwRes4;
  } DVD_DECODER_CAPS;

#define DVD_AUDIO_CAPS_AC3 0x00000001
#define DVD_AUDIO_CAPS_MPEG2 0x00000002
#define DVD_AUDIO_CAPS_LPCM 0x00000004
#define DVD_AUDIO_CAPS_DTS 0x00000008
#define DVD_AUDIO_CAPS_SDDS 0x00000010

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0387_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0387_v0_0_s_ifspec;
#ifndef __IDvdInfo2_INTERFACE_DEFINED__
#define __IDvdInfo2_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IDvdInfo2;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IDvdInfo2 : public IUnknown {
  public:
    virtual HRESULT WINAPI GetCurrentDomain(DVD_DOMAIN *pDomain) = 0;
    virtual HRESULT WINAPI GetCurrentLocation(DVD_PLAYBACK_LOCATION2 *pLocation) = 0;
    virtual HRESULT WINAPI GetTotalTitleTime(DVD_HMSF_TIMECODE *pTotalTime,ULONG *ulTimeCodeFlags) = 0;
    virtual HRESULT WINAPI GetCurrentButton(ULONG *pulButtonsAvailable,ULONG *pulCurrentButton) = 0;
    virtual HRESULT WINAPI GetCurrentAngle(ULONG *pulAnglesAvailable,ULONG *pulCurrentAngle) = 0;
    virtual HRESULT WINAPI GetCurrentAudio(ULONG *pulStreamsAvailable,ULONG *pulCurrentStream) = 0;
    virtual HRESULT WINAPI GetCurrentSubpicture(ULONG *pulStreamsAvailable,ULONG *pulCurrentStream,WINBOOL *pbIsDisabled) = 0;
    virtual HRESULT WINAPI GetCurrentUOPS(ULONG *pulUOPs) = 0;
    virtual HRESULT WINAPI GetAllSPRMs(SPRMARRAY *pRegisterArray) = 0;
    virtual HRESULT WINAPI GetAllGPRMs(GPRMARRAY *pRegisterArray) = 0;
    virtual HRESULT WINAPI GetAudioLanguage(ULONG ulStream,LCID *pLanguage) = 0;
    virtual HRESULT WINAPI GetSubpictureLanguage(ULONG ulStream,LCID *pLanguage) = 0;
    virtual HRESULT WINAPI GetTitleAttributes(ULONG ulTitle,DVD_MenuAttributes *pMenu,DVD_TitleAttributes *pTitle) = 0;
    virtual HRESULT WINAPI GetVMGAttributes(DVD_MenuAttributes *pATR) = 0;
    virtual HRESULT WINAPI GetCurrentVideoAttributes(DVD_VideoAttributes *pATR) = 0;
    virtual HRESULT WINAPI GetAudioAttributes(ULONG ulStream,DVD_AudioAttributes *pATR) = 0;
    virtual HRESULT WINAPI GetKaraokeAttributes(ULONG ulStream,DVD_KaraokeAttributes *pAttributes) = 0;
    virtual HRESULT WINAPI GetSubpictureAttributes(ULONG ulStream,DVD_SubpictureAttributes *pATR) = 0;
    virtual HRESULT WINAPI GetDVDVolumeInfo(ULONG *pulNumOfVolumes,ULONG *pulVolume,DVD_DISC_SIDE *pSide,ULONG *pulNumOfTitles) = 0;
    virtual HRESULT WINAPI GetDVDTextNumberOfLanguages(ULONG *pulNumOfLangs) = 0;
    virtual HRESULT WINAPI GetDVDTextLanguageInfo(ULONG ulLangIndex,ULONG *pulNumOfStrings,LCID *pLangCode,enum DVD_TextCharSet *pbCharacterSet) = 0;
    virtual HRESULT WINAPI GetDVDTextStringAsNative(ULONG ulLangIndex,ULONG ulStringIndex,BYTE *pbBuffer,ULONG ulMaxBufferSize,ULONG *pulActualSize,enum DVD_TextStringType *pType) = 0;
    virtual HRESULT WINAPI GetDVDTextStringAsUnicode(ULONG ulLangIndex,ULONG ulStringIndex,WCHAR *pchwBuffer,ULONG ulMaxBufferSize,ULONG *pulActualSize,enum DVD_TextStringType *pType) = 0;
    virtual HRESULT WINAPI GetPlayerParentalLevel(ULONG *pulParentalLevel,BYTE pbCountryCode[2]) = 0;
    virtual HRESULT WINAPI GetNumberOfChapters(ULONG ulTitle,ULONG *pulNumOfChapters) = 0;
    virtual HRESULT WINAPI GetTitleParentalLevels(ULONG ulTitle,ULONG *pulParentalLevels) = 0;
    virtual HRESULT WINAPI GetDVDDirectory(LPWSTR pszwPath,ULONG ulMaxSize,ULONG *pulActualSize) = 0;
    virtual HRESULT WINAPI IsAudioStreamEnabled(ULONG ulStreamNum,WINBOOL *pbEnabled) = 0;
    virtual HRESULT WINAPI GetDiscID(LPCWSTR pszwPath,ULONGLONG *pullDiscID) = 0;
    virtual HRESULT WINAPI GetState(IDvdState **pStateData) = 0;
    virtual HRESULT WINAPI GetMenuLanguages(LCID *pLanguages,ULONG ulMaxLanguages,ULONG *pulActualLanguages) = 0;
    virtual HRESULT WINAPI GetButtonAtPosition(POINT point,ULONG *pulButtonIndex) = 0;
    virtual HRESULT WINAPI GetCmdFromEvent(LONG_PTR lParam1,IDvdCmd **pCmdObj) = 0;
    virtual HRESULT WINAPI GetDefaultMenuLanguage(LCID *pLanguage) = 0;
    virtual HRESULT WINAPI GetDefaultAudioLanguage(LCID *pLanguage,DVD_AUDIO_LANG_EXT *pAudioExtension) = 0;
    virtual HRESULT WINAPI GetDefaultSubpictureLanguage(LCID *pLanguage,DVD_SUBPICTURE_LANG_EXT *pSubpictureExtension) = 0;
    virtual HRESULT WINAPI GetDecoderCaps(DVD_DECODER_CAPS *pCaps) = 0;
    virtual HRESULT WINAPI GetButtonRect(ULONG ulButton,RECT *pRect) = 0;
    virtual HRESULT WINAPI IsSubpictureStreamEnabled(ULONG ulStreamNum,WINBOOL *pbEnabled) = 0;
  };
#else
  typedef struct IDvdInfo2Vtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IDvdInfo2 *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IDvdInfo2 *This);
      ULONG (WINAPI *Release)(IDvdInfo2 *This);
      HRESULT (WINAPI *GetCurrentDomain)(IDvdInfo2 *This,DVD_DOMAIN *pDomain);
      HRESULT (WINAPI *GetCurrentLocation)(IDvdInfo2 *This,DVD_PLAYBACK_LOCATION2 *pLocation);
      HRESULT (WINAPI *GetTotalTitleTime)(IDvdInfo2 *This,DVD_HMSF_TIMECODE *pTotalTime,ULONG *ulTimeCodeFlags);
      HRESULT (WINAPI *GetCurrentButton)(IDvdInfo2 *This,ULONG *pulButtonsAvailable,ULONG *pulCurrentButton);
      HRESULT (WINAPI *GetCurrentAngle)(IDvdInfo2 *This,ULONG *pulAnglesAvailable,ULONG *pulCurrentAngle);
      HRESULT (WINAPI *GetCurrentAudio)(IDvdInfo2 *This,ULONG *pulStreamsAvailable,ULONG *pulCurrentStream);
      HRESULT (WINAPI *GetCurrentSubpicture)(IDvdInfo2 *This,ULONG *pulStreamsAvailable,ULONG *pulCurrentStream,WINBOOL *pbIsDisabled);
      HRESULT (WINAPI *GetCurrentUOPS)(IDvdInfo2 *This,ULONG *pulUOPs);
      HRESULT (WINAPI *GetAllSPRMs)(IDvdInfo2 *This,SPRMARRAY *pRegisterArray);
      HRESULT (WINAPI *GetAllGPRMs)(IDvdInfo2 *This,GPRMARRAY *pRegisterArray);
      HRESULT (WINAPI *GetAudioLanguage)(IDvdInfo2 *This,ULONG ulStream,LCID *pLanguage);
      HRESULT (WINAPI *GetSubpictureLanguage)(IDvdInfo2 *This,ULONG ulStream,LCID *pLanguage);
      HRESULT (WINAPI *GetTitleAttributes)(IDvdInfo2 *This,ULONG ulTitle,DVD_MenuAttributes *pMenu,DVD_TitleAttributes *pTitle);
      HRESULT (WINAPI *GetVMGAttributes)(IDvdInfo2 *This,DVD_MenuAttributes *pATR);
      HRESULT (WINAPI *GetCurrentVideoAttributes)(IDvdInfo2 *This,DVD_VideoAttributes *pATR);
      HRESULT (WINAPI *GetAudioAttributes)(IDvdInfo2 *This,ULONG ulStream,DVD_AudioAttributes *pATR);
      HRESULT (WINAPI *GetKaraokeAttributes)(IDvdInfo2 *This,ULONG ulStream,DVD_KaraokeAttributes *pAttributes);
      HRESULT (WINAPI *GetSubpictureAttributes)(IDvdInfo2 *This,ULONG ulStream,DVD_SubpictureAttributes *pATR);
      HRESULT (WINAPI *GetDVDVolumeInfo)(IDvdInfo2 *This,ULONG *pulNumOfVolumes,ULONG *pulVolume,DVD_DISC_SIDE *pSide,ULONG *pulNumOfTitles);
      HRESULT (WINAPI *GetDVDTextNumberOfLanguages)(IDvdInfo2 *This,ULONG *pulNumOfLangs);
      HRESULT (WINAPI *GetDVDTextLanguageInfo)(IDvdInfo2 *This,ULONG ulLangIndex,ULONG *pulNumOfStrings,LCID *pLangCode,enum DVD_TextCharSet *pbCharacterSet);
      HRESULT (WINAPI *GetDVDTextStringAsNative)(IDvdInfo2 *This,ULONG ulLangIndex,ULONG ulStringIndex,BYTE *pbBuffer,ULONG ulMaxBufferSize,ULONG *pulActualSize,enum DVD_TextStringType *pType);
      HRESULT (WINAPI *GetDVDTextStringAsUnicode)(IDvdInfo2 *This,ULONG ulLangIndex,ULONG ulStringIndex,WCHAR *pchwBuffer,ULONG ulMaxBufferSize,ULONG *pulActualSize,enum DVD_TextStringType *pType);
      HRESULT (WINAPI *GetPlayerParentalLevel)(IDvdInfo2 *This,ULONG *pulParentalLevel,BYTE pbCountryCode[2]);
      HRESULT (WINAPI *GetNumberOfChapters)(IDvdInfo2 *This,ULONG ulTitle,ULONG *pulNumOfChapters);
      HRESULT (WINAPI *GetTitleParentalLevels)(IDvdInfo2 *This,ULONG ulTitle,ULONG *pulParentalLevels);
      HRESULT (WINAPI *GetDVDDirectory)(IDvdInfo2 *This,LPWSTR pszwPath,ULONG ulMaxSize,ULONG *pulActualSize);
      HRESULT (WINAPI *IsAudioStreamEnabled)(IDvdInfo2 *This,ULONG ulStreamNum,WINBOOL *pbEnabled);
      HRESULT (WINAPI *GetDiscID)(IDvdInfo2 *This,LPCWSTR pszwPath,ULONGLONG *pullDiscID);
      HRESULT (WINAPI *GetState)(IDvdInfo2 *This,IDvdState **pStateData);
      HRESULT (WINAPI *GetMenuLanguages)(IDvdInfo2 *This,LCID *pLanguages,ULONG ulMaxLanguages,ULONG *pulActualLanguages);
      HRESULT (WINAPI *GetButtonAtPosition)(IDvdInfo2 *This,POINT point,ULONG *pulButtonIndex);
      HRESULT (WINAPI *GetCmdFromEvent)(IDvdInfo2 *This,LONG_PTR lParam1,IDvdCmd **pCmdObj);
      HRESULT (WINAPI *GetDefaultMenuLanguage)(IDvdInfo2 *This,LCID *pLanguage);
      HRESULT (WINAPI *GetDefaultAudioLanguage)(IDvdInfo2 *This,LCID *pLanguage,DVD_AUDIO_LANG_EXT *pAudioExtension);
      HRESULT (WINAPI *GetDefaultSubpictureLanguage)(IDvdInfo2 *This,LCID *pLanguage,DVD_SUBPICTURE_LANG_EXT *pSubpictureExtension);
      HRESULT (WINAPI *GetDecoderCaps)(IDvdInfo2 *This,DVD_DECODER_CAPS *pCaps);
      HRESULT (WINAPI *GetButtonRect)(IDvdInfo2 *This,ULONG ulButton,RECT *pRect);
      HRESULT (WINAPI *IsSubpictureStreamEnabled)(IDvdInfo2 *This,ULONG ulStreamNum,WINBOOL *pbEnabled);
    END_INTERFACE
  } IDvdInfo2Vtbl;
  struct IDvdInfo2 {
    CONST_VTBL struct IDvdInfo2Vtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IDvdInfo2_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDvdInfo2_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDvdInfo2_Release(This) (This)->lpVtbl->Release(This)
#define IDvdInfo2_GetCurrentDomain(This,pDomain) (This)->lpVtbl->GetCurrentDomain(This,pDomain)
#define IDvdInfo2_GetCurrentLocation(This,pLocation) (This)->lpVtbl->GetCurrentLocation(This,pLocation)
#define IDvdInfo2_GetTotalTitleTime(This,pTotalTime,ulTimeCodeFlags) (This)->lpVtbl->GetTotalTitleTime(This,pTotalTime,ulTimeCodeFlags)
#define IDvdInfo2_GetCurrentButton(This,pulButtonsAvailable,pulCurrentButton) (This)->lpVtbl->GetCurrentButton(This,pulButtonsAvailable,pulCurrentButton)
#define IDvdInfo2_GetCurrentAngle(This,pulAnglesAvailable,pulCurrentAngle) (This)->lpVtbl->GetCurrentAngle(This,pulAnglesAvailable,pulCurrentAngle)
#define IDvdInfo2_GetCurrentAudio(This,pulStreamsAvailable,pulCurrentStream) (This)->lpVtbl->GetCurrentAudio(This,pulStreamsAvailable,pulCurrentStream)
#define IDvdInfo2_GetCurrentSubpicture(This,pulStreamsAvailable,pulCurrentStream,pbIsDisabled) (This)->lpVtbl->GetCurrentSubpicture(This,pulStreamsAvailable,pulCurrentStream,pbIsDisabled)
#define IDvdInfo2_GetCurrentUOPS(This,pulUOPs) (This)->lpVtbl->GetCurrentUOPS(This,pulUOPs)
#define IDvdInfo2_GetAllSPRMs(This,pRegisterArray) (This)->lpVtbl->GetAllSPRMs(This,pRegisterArray)
#define IDvdInfo2_GetAllGPRMs(This,pRegisterArray) (This)->lpVtbl->GetAllGPRMs(This,pRegisterArray)
#define IDvdInfo2_GetAudioLanguage(This,ulStream,pLanguage) (This)->lpVtbl->GetAudioLanguage(This,ulStream,pLanguage)
#define IDvdInfo2_GetSubpictureLanguage(This,ulStream,pLanguage) (This)->lpVtbl->GetSubpictureLanguage(This,ulStream,pLanguage)
#define IDvdInfo2_GetTitleAttributes(This,ulTitle,pMenu,pTitle) (This)->lpVtbl->GetTitleAttributes(This,ulTitle,pMenu,pTitle)
#define IDvdInfo2_GetVMGAttributes(This,pATR) (This)->lpVtbl->GetVMGAttributes(This,pATR)
#define IDvdInfo2_GetCurrentVideoAttributes(This,pATR) (This)->lpVtbl->GetCurrentVideoAttributes(This,pATR)
#define IDvdInfo2_GetAudioAttributes(This,ulStream,pATR) (This)->lpVtbl->GetAudioAttributes(This,ulStream,pATR)
#define IDvdInfo2_GetKaraokeAttributes(This,ulStream,pAttributes) (This)->lpVtbl->GetKaraokeAttributes(This,ulStream,pAttributes)
#define IDvdInfo2_GetSubpictureAttributes(This,ulStream,pATR) (This)->lpVtbl->GetSubpictureAttributes(This,ulStream,pATR)
#define IDvdInfo2_GetDVDVolumeInfo(This,pulNumOfVolumes,pulVolume,pSide,pulNumOfTitles) (This)->lpVtbl->GetDVDVolumeInfo(This,pulNumOfVolumes,pulVolume,pSide,pulNumOfTitles)
#define IDvdInfo2_GetDVDTextNumberOfLanguages(This,pulNumOfLangs) (This)->lpVtbl->GetDVDTextNumberOfLanguages(This,pulNumOfLangs)
#define IDvdInfo2_GetDVDTextLanguageInfo(This,ulLangIndex,pulNumOfStrings,pLangCode,pbCharacterSet) (This)->lpVtbl->GetDVDTextLanguageInfo(This,ulLangIndex,pulNumOfStrings,pLangCode,pbCharacterSet)
#define IDvdInfo2_GetDVDTextStringAsNative(This,ulLangIndex,ulStringIndex,pbBuffer,ulMaxBufferSize,pulActualSize,pType) (This)->lpVtbl->GetDVDTextStringAsNative(This,ulLangIndex,ulStringIndex,pbBuffer,ulMaxBufferSize,pulActualSize,pType)
#define IDvdInfo2_GetDVDTextStringAsUnicode(This,ulLangIndex,ulStringIndex,pchwBuffer,ulMaxBufferSize,pulActualSize,pType) (This)->lpVtbl->GetDVDTextStringAsUnicode(This,ulLangIndex,ulStringIndex,pchwBuffer,ulMaxBufferSize,pulActualSize,pType)
#define IDvdInfo2_GetPlayerParentalLevel(This,pulParentalLevel,pbCountryCode) (This)->lpVtbl->GetPlayerParentalLevel(This,pulParentalLevel,pbCountryCode)
#define IDvdInfo2_GetNumberOfChapters(This,ulTitle,pulNumOfChapters) (This)->lpVtbl->GetNumberOfChapters(This,ulTitle,pulNumOfChapters)
#define IDvdInfo2_GetTitleParentalLevels(This,ulTitle,pulParentalLevels) (This)->lpVtbl->GetTitleParentalLevels(This,ulTitle,pulParentalLevels)
#define IDvdInfo2_GetDVDDirectory(This,pszwPath,ulMaxSize,pulActualSize) (This)->lpVtbl->GetDVDDirectory(This,pszwPath,ulMaxSize,pulActualSize)
#define IDvdInfo2_IsAudioStreamEnabled(This,ulStreamNum,pbEnabled) (This)->lpVtbl->IsAudioStreamEnabled(This,ulStreamNum,pbEnabled)
#define IDvdInfo2_GetDiscID(This,pszwPath,pullDiscID) (This)->lpVtbl->GetDiscID(This,pszwPath,pullDiscID)
#define IDvdInfo2_GetState(This,pStateData) (This)->lpVtbl->GetState(This,pStateData)
#define IDvdInfo2_GetMenuLanguages(This,pLanguages,ulMaxLanguages,pulActualLanguages) (This)->lpVtbl->GetMenuLanguages(This,pLanguages,ulMaxLanguages,pulActualLanguages)
#define IDvdInfo2_GetButtonAtPosition(This,point,pulButtonIndex) (This)->lpVtbl->GetButtonAtPosition(This,point,pulButtonIndex)
#define IDvdInfo2_GetCmdFromEvent(This,lParam1,pCmdObj) (This)->lpVtbl->GetCmdFromEvent(This,lParam1,pCmdObj)
#define IDvdInfo2_GetDefaultMenuLanguage(This,pLanguage) (This)->lpVtbl->GetDefaultMenuLanguage(This,pLanguage)
#define IDvdInfo2_GetDefaultAudioLanguage(This,pLanguage,pAudioExtension) (This)->lpVtbl->GetDefaultAudioLanguage(This,pLanguage,pAudioExtension)
#define IDvdInfo2_GetDefaultSubpictureLanguage(This,pLanguage,pSubpictureExtension) (This)->lpVtbl->GetDefaultSubpictureLanguage(This,pLanguage,pSubpictureExtension)
#define IDvdInfo2_GetDecoderCaps(This,pCaps) (This)->lpVtbl->GetDecoderCaps(This,pCaps)
#define IDvdInfo2_GetButtonRect(This,ulButton,pRect) (This)->lpVtbl->GetButtonRect(This,ulButton,pRect)
#define IDvdInfo2_IsSubpictureStreamEnabled(This,ulStreamNum,pbEnabled) (This)->lpVtbl->IsSubpictureStreamEnabled(This,ulStreamNum,pbEnabled)
#endif
#endif
  HRESULT WINAPI IDvdInfo2_GetCurrentDomain_Proxy(IDvdInfo2 *This,DVD_DOMAIN *pDomain);
  void __RPC_STUB IDvdInfo2_GetCurrentDomain_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetCurrentLocation_Proxy(IDvdInfo2 *This,DVD_PLAYBACK_LOCATION2 *pLocation);
  void __RPC_STUB IDvdInfo2_GetCurrentLocation_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetTotalTitleTime_Proxy(IDvdInfo2 *This,DVD_HMSF_TIMECODE *pTotalTime,ULONG *ulTimeCodeFlags);
  void __RPC_STUB IDvdInfo2_GetTotalTitleTime_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetCurrentButton_Proxy(IDvdInfo2 *This,ULONG *pulButtonsAvailable,ULONG *pulCurrentButton);
  void __RPC_STUB IDvdInfo2_GetCurrentButton_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetCurrentAngle_Proxy(IDvdInfo2 *This,ULONG *pulAnglesAvailable,ULONG *pulCurrentAngle);
  void __RPC_STUB IDvdInfo2_GetCurrentAngle_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetCurrentAudio_Proxy(IDvdInfo2 *This,ULONG *pulStreamsAvailable,ULONG *pulCurrentStream);
  void __RPC_STUB IDvdInfo2_GetCurrentAudio_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetCurrentSubpicture_Proxy(IDvdInfo2 *This,ULONG *pulStreamsAvailable,ULONG *pulCurrentStream,WINBOOL *pbIsDisabled);
  void __RPC_STUB IDvdInfo2_GetCurrentSubpicture_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetCurrentUOPS_Proxy(IDvdInfo2 *This,ULONG *pulUOPs);
  void __RPC_STUB IDvdInfo2_GetCurrentUOPS_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetAllSPRMs_Proxy(IDvdInfo2 *This,SPRMARRAY *pRegisterArray);
  void __RPC_STUB IDvdInfo2_GetAllSPRMs_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetAllGPRMs_Proxy(IDvdInfo2 *This,GPRMARRAY *pRegisterArray);
  void __RPC_STUB IDvdInfo2_GetAllGPRMs_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetAudioLanguage_Proxy(IDvdInfo2 *This,ULONG ulStream,LCID *pLanguage);
  void __RPC_STUB IDvdInfo2_GetAudioLanguage_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetSubpictureLanguage_Proxy(IDvdInfo2 *This,ULONG ulStream,LCID *pLanguage);
  void __RPC_STUB IDvdInfo2_GetSubpictureLanguage_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetTitleAttributes_Proxy(IDvdInfo2 *This,ULONG ulTitle,DVD_MenuAttributes *pMenu,DVD_TitleAttributes *pTitle);
  void __RPC_STUB IDvdInfo2_GetTitleAttributes_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetVMGAttributes_Proxy(IDvdInfo2 *This,DVD_MenuAttributes *pATR);
  void __RPC_STUB IDvdInfo2_GetVMGAttributes_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetCurrentVideoAttributes_Proxy(IDvdInfo2 *This,DVD_VideoAttributes *pATR);
  void __RPC_STUB IDvdInfo2_GetCurrentVideoAttributes_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetAudioAttributes_Proxy(IDvdInfo2 *This,ULONG ulStream,DVD_AudioAttributes *pATR);
  void __RPC_STUB IDvdInfo2_GetAudioAttributes_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetKaraokeAttributes_Proxy(IDvdInfo2 *This,ULONG ulStream,DVD_KaraokeAttributes *pAttributes);
  void __RPC_STUB IDvdInfo2_GetKaraokeAttributes_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetSubpictureAttributes_Proxy(IDvdInfo2 *This,ULONG ulStream,DVD_SubpictureAttributes *pATR);
  void __RPC_STUB IDvdInfo2_GetSubpictureAttributes_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetDVDVolumeInfo_Proxy(IDvdInfo2 *This,ULONG *pulNumOfVolumes,ULONG *pulVolume,DVD_DISC_SIDE *pSide,ULONG *pulNumOfTitles);
  void __RPC_STUB IDvdInfo2_GetDVDVolumeInfo_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetDVDTextNumberOfLanguages_Proxy(IDvdInfo2 *This,ULONG *pulNumOfLangs);
  void __RPC_STUB IDvdInfo2_GetDVDTextNumberOfLanguages_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetDVDTextLanguageInfo_Proxy(IDvdInfo2 *This,ULONG ulLangIndex,ULONG *pulNumOfStrings,LCID *pLangCode,enum DVD_TextCharSet *pbCharacterSet);
  void __RPC_STUB IDvdInfo2_GetDVDTextLanguageInfo_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetDVDTextStringAsNative_Proxy(IDvdInfo2 *This,ULONG ulLangIndex,ULONG ulStringIndex,BYTE *pbBuffer,ULONG ulMaxBufferSize,ULONG *pulActualSize,enum DVD_TextStringType *pType);
  void __RPC_STUB IDvdInfo2_GetDVDTextStringAsNative_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetDVDTextStringAsUnicode_Proxy(IDvdInfo2 *This,ULONG ulLangIndex,ULONG ulStringIndex,WCHAR *pchwBuffer,ULONG ulMaxBufferSize,ULONG *pulActualSize,enum DVD_TextStringType *pType);
  void __RPC_STUB IDvdInfo2_GetDVDTextStringAsUnicode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetPlayerParentalLevel_Proxy(IDvdInfo2 *This,ULONG *pulParentalLevel,BYTE pbCountryCode[2]);
  void __RPC_STUB IDvdInfo2_GetPlayerParentalLevel_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetNumberOfChapters_Proxy(IDvdInfo2 *This,ULONG ulTitle,ULONG *pulNumOfChapters);
  void __RPC_STUB IDvdInfo2_GetNumberOfChapters_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetTitleParentalLevels_Proxy(IDvdInfo2 *This,ULONG ulTitle,ULONG *pulParentalLevels);
  void __RPC_STUB IDvdInfo2_GetTitleParentalLevels_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetDVDDirectory_Proxy(IDvdInfo2 *This,LPWSTR pszwPath,ULONG ulMaxSize,ULONG *pulActualSize);
  void __RPC_STUB IDvdInfo2_GetDVDDirectory_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_IsAudioStreamEnabled_Proxy(IDvdInfo2 *This,ULONG ulStreamNum,WINBOOL *pbEnabled);
  void __RPC_STUB IDvdInfo2_IsAudioStreamEnabled_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetDiscID_Proxy(IDvdInfo2 *This,LPCWSTR pszwPath,ULONGLONG *pullDiscID);
  void __RPC_STUB IDvdInfo2_GetDiscID_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetState_Proxy(IDvdInfo2 *This,IDvdState **pStateData);
  void __RPC_STUB IDvdInfo2_GetState_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetMenuLanguages_Proxy(IDvdInfo2 *This,LCID *pLanguages,ULONG ulMaxLanguages,ULONG *pulActualLanguages);
  void __RPC_STUB IDvdInfo2_GetMenuLanguages_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetButtonAtPosition_Proxy(IDvdInfo2 *This,POINT point,ULONG *pulButtonIndex);
  void __RPC_STUB IDvdInfo2_GetButtonAtPosition_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetCmdFromEvent_Proxy(IDvdInfo2 *This,LONG_PTR lParam1,IDvdCmd **pCmdObj);
  void __RPC_STUB IDvdInfo2_GetCmdFromEvent_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetDefaultMenuLanguage_Proxy(IDvdInfo2 *This,LCID *pLanguage);
  void __RPC_STUB IDvdInfo2_GetDefaultMenuLanguage_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetDefaultAudioLanguage_Proxy(IDvdInfo2 *This,LCID *pLanguage,DVD_AUDIO_LANG_EXT *pAudioExtension);
  void __RPC_STUB IDvdInfo2_GetDefaultAudioLanguage_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetDefaultSubpictureLanguage_Proxy(IDvdInfo2 *This,LCID *pLanguage,DVD_SUBPICTURE_LANG_EXT *pSubpictureExtension);
  void __RPC_STUB IDvdInfo2_GetDefaultSubpictureLanguage_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetDecoderCaps_Proxy(IDvdInfo2 *This,DVD_DECODER_CAPS *pCaps);
  void __RPC_STUB IDvdInfo2_GetDecoderCaps_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_GetButtonRect_Proxy(IDvdInfo2 *This,ULONG ulButton,RECT *pRect);
  void __RPC_STUB IDvdInfo2_GetButtonRect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdInfo2_IsSubpictureStreamEnabled_Proxy(IDvdInfo2 *This,ULONG ulStreamNum,WINBOOL *pbEnabled);
  void __RPC_STUB IDvdInfo2_IsSubpictureStreamEnabled_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef enum _AM_DVD_GRAPH_FLAGS {
    AM_DVD_HWDEC_PREFER = 0x1,AM_DVD_HWDEC_ONLY = 0x2,AM_DVD_SWDEC_PREFER = 0x4,AM_DVD_SWDEC_ONLY = 0x8,AM_DVD_NOVPE = 0x100,
    AM_DVD_VMR9_ONLY = 0x800
  } AM_DVD_GRAPH_FLAGS;

  typedef enum _AM_DVD_STREAM_FLAGS {
    AM_DVD_STREAM_VIDEO = 0x1,AM_DVD_STREAM_AUDIO = 0x2,AM_DVD_STREAM_SUBPIC = 0x4
  } AM_DVD_STREAM_FLAGS;

  typedef struct __MIDL___MIDL_itf_strmif_0389_0001 {
    HRESULT hrVPEStatus;
    WINBOOL bDvdVolInvalid;
    WINBOOL bDvdVolUnknown;
    WINBOOL bNoLine21In;
    WINBOOL bNoLine21Out;
    int iNumStreams;
    int iNumStreamsFailed;
    DWORD dwFailedStreamsFlag;
  } AM_DVD_RENDERSTATUS;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0389_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0389_v0_0_s_ifspec;
#ifndef __IDvdGraphBuilder_INTERFACE_DEFINED__
#define __IDvdGraphBuilder_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IDvdGraphBuilder;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IDvdGraphBuilder : public IUnknown {
  public:
    virtual HRESULT WINAPI GetFiltergraph(IGraphBuilder **ppGB) = 0;
    virtual HRESULT WINAPI GetDvdInterface(REFIID riid,void **ppvIF) = 0;
    virtual HRESULT WINAPI RenderDvdVideoVolume(LPCWSTR lpcwszPathName,DWORD dwFlags,AM_DVD_RENDERSTATUS *pStatus) = 0;
  };
#else
  typedef struct IDvdGraphBuilderVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IDvdGraphBuilder *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IDvdGraphBuilder *This);
      ULONG (WINAPI *Release)(IDvdGraphBuilder *This);
      HRESULT (WINAPI *GetFiltergraph)(IDvdGraphBuilder *This,IGraphBuilder **ppGB);
      HRESULT (WINAPI *GetDvdInterface)(IDvdGraphBuilder *This,REFIID riid,void **ppvIF);
      HRESULT (WINAPI *RenderDvdVideoVolume)(IDvdGraphBuilder *This,LPCWSTR lpcwszPathName,DWORD dwFlags,AM_DVD_RENDERSTATUS *pStatus);
    END_INTERFACE
  } IDvdGraphBuilderVtbl;
  struct IDvdGraphBuilder {
    CONST_VTBL struct IDvdGraphBuilderVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IDvdGraphBuilder_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDvdGraphBuilder_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDvdGraphBuilder_Release(This) (This)->lpVtbl->Release(This)
#define IDvdGraphBuilder_GetFiltergraph(This,ppGB) (This)->lpVtbl->GetFiltergraph(This,ppGB)
#define IDvdGraphBuilder_GetDvdInterface(This,riid,ppvIF) (This)->lpVtbl->GetDvdInterface(This,riid,ppvIF)
#define IDvdGraphBuilder_RenderDvdVideoVolume(This,lpcwszPathName,dwFlags,pStatus) (This)->lpVtbl->RenderDvdVideoVolume(This,lpcwszPathName,dwFlags,pStatus)
#endif
#endif
  HRESULT WINAPI IDvdGraphBuilder_GetFiltergraph_Proxy(IDvdGraphBuilder *This,IGraphBuilder **ppGB);
  void __RPC_STUB IDvdGraphBuilder_GetFiltergraph_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdGraphBuilder_GetDvdInterface_Proxy(IDvdGraphBuilder *This,REFIID riid,void **ppvIF);
  void __RPC_STUB IDvdGraphBuilder_GetDvdInterface_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDvdGraphBuilder_RenderDvdVideoVolume_Proxy(IDvdGraphBuilder *This,LPCWSTR lpcwszPathName,DWORD dwFlags,AM_DVD_RENDERSTATUS *pStatus);
  void __RPC_STUB IDvdGraphBuilder_RenderDvdVideoVolume_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IDDrawExclModeVideo_INTERFACE_DEFINED__
#define __IDDrawExclModeVideo_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IDDrawExclModeVideo;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IDDrawExclModeVideo : public IUnknown {
  public:
    virtual HRESULT WINAPI SetDDrawObject(IDirectDraw *pDDrawObject) = 0;
    virtual HRESULT WINAPI GetDDrawObject(IDirectDraw **ppDDrawObject,WINBOOL *pbUsingExternal) = 0;
    virtual HRESULT WINAPI SetDDrawSurface(IDirectDrawSurface *pDDrawSurface) = 0;
    virtual HRESULT WINAPI GetDDrawSurface(IDirectDrawSurface **ppDDrawSurface,WINBOOL *pbUsingExternal) = 0;
    virtual HRESULT WINAPI SetDrawParameters(const RECT *prcSource,const RECT *prcTarget) = 0;
    virtual HRESULT WINAPI GetNativeVideoProps(DWORD *pdwVideoWidth,DWORD *pdwVideoHeight,DWORD *pdwPictAspectRatioX,DWORD *pdwPictAspectRatioY) = 0;
    virtual HRESULT WINAPI SetCallbackInterface(IDDrawExclModeVideoCallback *pCallback,DWORD dwFlags) = 0;
  };
#else
  typedef struct IDDrawExclModeVideoVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IDDrawExclModeVideo *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IDDrawExclModeVideo *This);
      ULONG (WINAPI *Release)(IDDrawExclModeVideo *This);
      HRESULT (WINAPI *SetDDrawObject)(IDDrawExclModeVideo *This,IDirectDraw *pDDrawObject);
      HRESULT (WINAPI *GetDDrawObject)(IDDrawExclModeVideo *This,IDirectDraw **ppDDrawObject,WINBOOL *pbUsingExternal);
      HRESULT (WINAPI *SetDDrawSurface)(IDDrawExclModeVideo *This,IDirectDrawSurface *pDDrawSurface);
      HRESULT (WINAPI *GetDDrawSurface)(IDDrawExclModeVideo *This,IDirectDrawSurface **ppDDrawSurface,WINBOOL *pbUsingExternal);
      HRESULT (WINAPI *SetDrawParameters)(IDDrawExclModeVideo *This,const RECT *prcSource,const RECT *prcTarget);
      HRESULT (WINAPI *GetNativeVideoProps)(IDDrawExclModeVideo *This,DWORD *pdwVideoWidth,DWORD *pdwVideoHeight,DWORD *pdwPictAspectRatioX,DWORD *pdwPictAspectRatioY);
      HRESULT (WINAPI *SetCallbackInterface)(IDDrawExclModeVideo *This,IDDrawExclModeVideoCallback *pCallback,DWORD dwFlags);
    END_INTERFACE
  } IDDrawExclModeVideoVtbl;
  struct IDDrawExclModeVideo {
    CONST_VTBL struct IDDrawExclModeVideoVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IDDrawExclModeVideo_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDDrawExclModeVideo_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDDrawExclModeVideo_Release(This) (This)->lpVtbl->Release(This)
#define IDDrawExclModeVideo_SetDDrawObject(This,pDDrawObject) (This)->lpVtbl->SetDDrawObject(This,pDDrawObject)
#define IDDrawExclModeVideo_GetDDrawObject(This,ppDDrawObject,pbUsingExternal) (This)->lpVtbl->GetDDrawObject(This,ppDDrawObject,pbUsingExternal)
#define IDDrawExclModeVideo_SetDDrawSurface(This,pDDrawSurface) (This)->lpVtbl->SetDDrawSurface(This,pDDrawSurface)
#define IDDrawExclModeVideo_GetDDrawSurface(This,ppDDrawSurface,pbUsingExternal) (This)->lpVtbl->GetDDrawSurface(This,ppDDrawSurface,pbUsingExternal)
#define IDDrawExclModeVideo_SetDrawParameters(This,prcSource,prcTarget) (This)->lpVtbl->SetDrawParameters(This,prcSource,prcTarget)
#define IDDrawExclModeVideo_GetNativeVideoProps(This,pdwVideoWidth,pdwVideoHeight,pdwPictAspectRatioX,pdwPictAspectRatioY) (This)->lpVtbl->GetNativeVideoProps(This,pdwVideoWidth,pdwVideoHeight,pdwPictAspectRatioX,pdwPictAspectRatioY)
#define IDDrawExclModeVideo_SetCallbackInterface(This,pCallback,dwFlags) (This)->lpVtbl->SetCallbackInterface(This,pCallback,dwFlags)
#endif
#endif
  HRESULT WINAPI IDDrawExclModeVideo_SetDDrawObject_Proxy(IDDrawExclModeVideo *This,IDirectDraw *pDDrawObject);
  void __RPC_STUB IDDrawExclModeVideo_SetDDrawObject_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDDrawExclModeVideo_GetDDrawObject_Proxy(IDDrawExclModeVideo *This,IDirectDraw **ppDDrawObject,WINBOOL *pbUsingExternal);
  void __RPC_STUB IDDrawExclModeVideo_GetDDrawObject_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDDrawExclModeVideo_SetDDrawSurface_Proxy(IDDrawExclModeVideo *This,IDirectDrawSurface *pDDrawSurface);
  void __RPC_STUB IDDrawExclModeVideo_SetDDrawSurface_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDDrawExclModeVideo_GetDDrawSurface_Proxy(IDDrawExclModeVideo *This,IDirectDrawSurface **ppDDrawSurface,WINBOOL *pbUsingExternal);
  void __RPC_STUB IDDrawExclModeVideo_GetDDrawSurface_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDDrawExclModeVideo_SetDrawParameters_Proxy(IDDrawExclModeVideo *This,const RECT *prcSource,const RECT *prcTarget);
  void __RPC_STUB IDDrawExclModeVideo_SetDrawParameters_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDDrawExclModeVideo_GetNativeVideoProps_Proxy(IDDrawExclModeVideo *This,DWORD *pdwVideoWidth,DWORD *pdwVideoHeight,DWORD *pdwPictAspectRatioX,DWORD *pdwPictAspectRatioY);
  void __RPC_STUB IDDrawExclModeVideo_GetNativeVideoProps_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDDrawExclModeVideo_SetCallbackInterface_Proxy(IDDrawExclModeVideo *This,IDDrawExclModeVideoCallback *pCallback,DWORD dwFlags);
  void __RPC_STUB IDDrawExclModeVideo_SetCallbackInterface_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  enum _AM_OVERLAY_NOTIFY_FLAGS {
    AM_OVERLAY_NOTIFY_VISIBLE_CHANGE = 0x1,AM_OVERLAY_NOTIFY_SOURCE_CHANGE = 0x2,AM_OVERLAY_NOTIFY_DEST_CHANGE = 0x4
  };

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0391_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0391_v0_0_s_ifspec;
#ifndef __IDDrawExclModeVideoCallback_INTERFACE_DEFINED__
#define __IDDrawExclModeVideoCallback_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IDDrawExclModeVideoCallback;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IDDrawExclModeVideoCallback : public IUnknown {
  public:
    virtual HRESULT WINAPI OnUpdateOverlay(WINBOOL bBefore,DWORD dwFlags,WINBOOL bOldVisible,const RECT *prcOldSrc,const RECT *prcOldDest,WINBOOL bNewVisible,const RECT *prcNewSrc,const RECT *prcNewDest) = 0;
    virtual HRESULT WINAPI OnUpdateColorKey(const COLORKEY *pKey,DWORD dwColor) = 0;
    virtual HRESULT WINAPI OnUpdateSize(DWORD dwWidth,DWORD dwHeight,DWORD dwARWidth,DWORD dwARHeight) = 0;
  };
#else
  typedef struct IDDrawExclModeVideoCallbackVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IDDrawExclModeVideoCallback *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IDDrawExclModeVideoCallback *This);
      ULONG (WINAPI *Release)(IDDrawExclModeVideoCallback *This);
      HRESULT (WINAPI *OnUpdateOverlay)(IDDrawExclModeVideoCallback *This,WINBOOL bBefore,DWORD dwFlags,WINBOOL bOldVisible,const RECT *prcOldSrc,const RECT *prcOldDest,WINBOOL bNewVisible,const RECT *prcNewSrc,const RECT *prcNewDest);
      HRESULT (WINAPI *OnUpdateColorKey)(IDDrawExclModeVideoCallback *This,const COLORKEY *pKey,DWORD dwColor);
      HRESULT (WINAPI *OnUpdateSize)(IDDrawExclModeVideoCallback *This,DWORD dwWidth,DWORD dwHeight,DWORD dwARWidth,DWORD dwARHeight);
    END_INTERFACE
  } IDDrawExclModeVideoCallbackVtbl;
  struct IDDrawExclModeVideoCallback {
    CONST_VTBL struct IDDrawExclModeVideoCallbackVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IDDrawExclModeVideoCallback_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDDrawExclModeVideoCallback_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDDrawExclModeVideoCallback_Release(This) (This)->lpVtbl->Release(This)
#define IDDrawExclModeVideoCallback_OnUpdateOverlay(This,bBefore,dwFlags,bOldVisible,prcOldSrc,prcOldDest,bNewVisible,prcNewSrc,prcNewDest) (This)->lpVtbl->OnUpdateOverlay(This,bBefore,dwFlags,bOldVisible,prcOldSrc,prcOldDest,bNewVisible,prcNewSrc,prcNewDest)
#define IDDrawExclModeVideoCallback_OnUpdateColorKey(This,pKey,dwColor) (This)->lpVtbl->OnUpdateColorKey(This,pKey,dwColor)
#define IDDrawExclModeVideoCallback_OnUpdateSize(This,dwWidth,dwHeight,dwARWidth,dwARHeight) (This)->lpVtbl->OnUpdateSize(This,dwWidth,dwHeight,dwARWidth,dwARHeight)
#endif
#endif
  HRESULT WINAPI IDDrawExclModeVideoCallback_OnUpdateOverlay_Proxy(IDDrawExclModeVideoCallback *This,WINBOOL bBefore,DWORD dwFlags,WINBOOL bOldVisible,const RECT *prcOldSrc,const RECT *prcOldDest,WINBOOL bNewVisible,const RECT *prcNewSrc,const RECT *prcNewDest);
  void __RPC_STUB IDDrawExclModeVideoCallback_OnUpdateOverlay_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDDrawExclModeVideoCallback_OnUpdateColorKey_Proxy(IDDrawExclModeVideoCallback *This,const COLORKEY *pKey,DWORD dwColor);
  void __RPC_STUB IDDrawExclModeVideoCallback_OnUpdateColorKey_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDDrawExclModeVideoCallback_OnUpdateSize_Proxy(IDDrawExclModeVideoCallback *This,DWORD dwWidth,DWORD dwHeight,DWORD dwARWidth,DWORD dwARHeight);
  void __RPC_STUB IDDrawExclModeVideoCallback_OnUpdateSize_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0392_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0392_v0_0_s_ifspec;
#ifndef __IPinConnection_INTERFACE_DEFINED__
#define __IPinConnection_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IPinConnection;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IPinConnection : public IUnknown {
  public:
    virtual HRESULT WINAPI DynamicQueryAccept(const AM_MEDIA_TYPE *pmt) = 0;
    virtual HRESULT WINAPI NotifyEndOfStream(HANDLE hNotifyEvent) = 0;
    virtual HRESULT WINAPI IsEndPin(void) = 0;
    virtual HRESULT WINAPI DynamicDisconnect(void) = 0;
  };
#else
  typedef struct IPinConnectionVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IPinConnection *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IPinConnection *This);
      ULONG (WINAPI *Release)(IPinConnection *This);
      HRESULT (WINAPI *DynamicQueryAccept)(IPinConnection *This,const AM_MEDIA_TYPE *pmt);
      HRESULT (WINAPI *NotifyEndOfStream)(IPinConnection *This,HANDLE hNotifyEvent);
      HRESULT (WINAPI *IsEndPin)(IPinConnection *This);
      HRESULT (WINAPI *DynamicDisconnect)(IPinConnection *This);
    END_INTERFACE
  } IPinConnectionVtbl;
  struct IPinConnection {
    CONST_VTBL struct IPinConnectionVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IPinConnection_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IPinConnection_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IPinConnection_Release(This) (This)->lpVtbl->Release(This)
#define IPinConnection_DynamicQueryAccept(This,pmt) (This)->lpVtbl->DynamicQueryAccept(This,pmt)
#define IPinConnection_NotifyEndOfStream(This,hNotifyEvent) (This)->lpVtbl->NotifyEndOfStream(This,hNotifyEvent)
#define IPinConnection_IsEndPin(This) (This)->lpVtbl->IsEndPin(This)
#define IPinConnection_DynamicDisconnect(This) (This)->lpVtbl->DynamicDisconnect(This)
#endif
#endif
  HRESULT WINAPI IPinConnection_DynamicQueryAccept_Proxy(IPinConnection *This,const AM_MEDIA_TYPE *pmt);
  void __RPC_STUB IPinConnection_DynamicQueryAccept_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPinConnection_NotifyEndOfStream_Proxy(IPinConnection *This,HANDLE hNotifyEvent);
  void __RPC_STUB IPinConnection_NotifyEndOfStream_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPinConnection_IsEndPin_Proxy(IPinConnection *This);
  void __RPC_STUB IPinConnection_IsEndPin_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPinConnection_DynamicDisconnect_Proxy(IPinConnection *This);
  void __RPC_STUB IPinConnection_DynamicDisconnect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IPinFlowControl_INTERFACE_DEFINED__
#define __IPinFlowControl_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IPinFlowControl;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IPinFlowControl : public IUnknown {
  public:
    virtual HRESULT WINAPI Block(DWORD dwBlockFlags,HANDLE hEvent) = 0;
  };
#else
  typedef struct IPinFlowControlVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IPinFlowControl *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IPinFlowControl *This);
      ULONG (WINAPI *Release)(IPinFlowControl *This);
      HRESULT (WINAPI *Block)(IPinFlowControl *This,DWORD dwBlockFlags,HANDLE hEvent);
    END_INTERFACE
  } IPinFlowControlVtbl;
  struct IPinFlowControl {
    CONST_VTBL struct IPinFlowControlVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IPinFlowControl_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IPinFlowControl_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IPinFlowControl_Release(This) (This)->lpVtbl->Release(This)
#define IPinFlowControl_Block(This,dwBlockFlags,hEvent) (This)->lpVtbl->Block(This,dwBlockFlags,hEvent)
#endif
#endif
  HRESULT WINAPI IPinFlowControl_Block_Proxy(IPinFlowControl *This,DWORD dwBlockFlags,HANDLE hEvent);
  void __RPC_STUB IPinFlowControl_Block_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  enum _AM_PIN_FLOW_CONTROL_BLOCK_FLAGS {
    AM_PIN_FLOW_CONTROL_BLOCK = 0x1
  };
  typedef enum _AM_GRAPH_CONFIG_RECONNECT_FLAGS {
    AM_GRAPH_CONFIG_RECONNECT_DIRECTCONNECT = 0x1,AM_GRAPH_CONFIG_RECONNECT_CACHE_REMOVED_FILTERS = 0x2,
    AM_GRAPH_CONFIG_RECONNECT_USE_ONLY_CACHED_FILTERS = 0x4
  } AM_GRAPH_CONFIG_RECONNECT_FLAGS;

  enum _REM_FILTER_FLAGS {
    REMFILTERF_LEAVECONNECTED = 0x1
  };

  typedef enum _AM_FILTER_FLAGS {
    AM_FILTER_FLAGS_REMOVABLE = 0x1
  } AM_FILTER_FLAGS;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0394_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0394_v0_0_s_ifspec;
#ifndef __IGraphConfig_INTERFACE_DEFINED__
#define __IGraphConfig_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IGraphConfig;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IGraphConfig : public IUnknown {
  public:
    virtual HRESULT WINAPI Reconnect(IPin *pOutputPin,IPin *pInputPin,const AM_MEDIA_TYPE *pmtFirstConnection,IBaseFilter *pUsingFilter,HANDLE hAbortEvent,DWORD dwFlags) = 0;
    virtual HRESULT WINAPI Reconfigure(IGraphConfigCallback *pCallback,PVOID pvContext,DWORD dwFlags,HANDLE hAbortEvent) = 0;
    virtual HRESULT WINAPI AddFilterToCache(IBaseFilter *pFilter) = 0;
    virtual HRESULT WINAPI EnumCacheFilter(IEnumFilters **pEnum) = 0;
    virtual HRESULT WINAPI RemoveFilterFromCache(IBaseFilter *pFilter) = 0;
    virtual HRESULT WINAPI GetStartTime(REFERENCE_TIME *prtStart) = 0;
    virtual HRESULT WINAPI PushThroughData(IPin *pOutputPin,IPinConnection *pConnection,HANDLE hEventAbort) = 0;
    virtual HRESULT WINAPI SetFilterFlags(IBaseFilter *pFilter,DWORD dwFlags) = 0;
    virtual HRESULT WINAPI GetFilterFlags(IBaseFilter *pFilter,DWORD *pdwFlags) = 0;
    virtual HRESULT WINAPI RemoveFilterEx(IBaseFilter *pFilter,DWORD Flags) = 0;
  };
#else
  typedef struct IGraphConfigVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IGraphConfig *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IGraphConfig *This);
      ULONG (WINAPI *Release)(IGraphConfig *This);
      HRESULT (WINAPI *Reconnect)(IGraphConfig *This,IPin *pOutputPin,IPin *pInputPin,const AM_MEDIA_TYPE *pmtFirstConnection,IBaseFilter *pUsingFilter,HANDLE hAbortEvent,DWORD dwFlags);
      HRESULT (WINAPI *Reconfigure)(IGraphConfig *This,IGraphConfigCallback *pCallback,PVOID pvContext,DWORD dwFlags,HANDLE hAbortEvent);
      HRESULT (WINAPI *AddFilterToCache)(IGraphConfig *This,IBaseFilter *pFilter);
      HRESULT (WINAPI *EnumCacheFilter)(IGraphConfig *This,IEnumFilters **pEnum);
      HRESULT (WINAPI *RemoveFilterFromCache)(IGraphConfig *This,IBaseFilter *pFilter);
      HRESULT (WINAPI *GetStartTime)(IGraphConfig *This,REFERENCE_TIME *prtStart);
      HRESULT (WINAPI *PushThroughData)(IGraphConfig *This,IPin *pOutputPin,IPinConnection *pConnection,HANDLE hEventAbort);
      HRESULT (WINAPI *SetFilterFlags)(IGraphConfig *This,IBaseFilter *pFilter,DWORD dwFlags);
      HRESULT (WINAPI *GetFilterFlags)(IGraphConfig *This,IBaseFilter *pFilter,DWORD *pdwFlags);
      HRESULT (WINAPI *RemoveFilterEx)(IGraphConfig *This,IBaseFilter *pFilter,DWORD Flags);
    END_INTERFACE
  } IGraphConfigVtbl;
  struct IGraphConfig {
    CONST_VTBL struct IGraphConfigVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IGraphConfig_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IGraphConfig_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IGraphConfig_Release(This) (This)->lpVtbl->Release(This)
#define IGraphConfig_Reconnect(This,pOutputPin,pInputPin,pmtFirstConnection,pUsingFilter,hAbortEvent,dwFlags) (This)->lpVtbl->Reconnect(This,pOutputPin,pInputPin,pmtFirstConnection,pUsingFilter,hAbortEvent,dwFlags)
#define IGraphConfig_Reconfigure(This,pCallback,pvContext,dwFlags,hAbortEvent) (This)->lpVtbl->Reconfigure(This,pCallback,pvContext,dwFlags,hAbortEvent)
#define IGraphConfig_AddFilterToCache(This,pFilter) (This)->lpVtbl->AddFilterToCache(This,pFilter)
#define IGraphConfig_EnumCacheFilter(This,pEnum) (This)->lpVtbl->EnumCacheFilter(This,pEnum)
#define IGraphConfig_RemoveFilterFromCache(This,pFilter) (This)->lpVtbl->RemoveFilterFromCache(This,pFilter)
#define IGraphConfig_GetStartTime(This,prtStart) (This)->lpVtbl->GetStartTime(This,prtStart)
#define IGraphConfig_PushThroughData(This,pOutputPin,pConnection,hEventAbort) (This)->lpVtbl->PushThroughData(This,pOutputPin,pConnection,hEventAbort)
#define IGraphConfig_SetFilterFlags(This,pFilter,dwFlags) (This)->lpVtbl->SetFilterFlags(This,pFilter,dwFlags)
#define IGraphConfig_GetFilterFlags(This,pFilter,pdwFlags) (This)->lpVtbl->GetFilterFlags(This,pFilter,pdwFlags)
#define IGraphConfig_RemoveFilterEx(This,pFilter,Flags) (This)->lpVtbl->RemoveFilterEx(This,pFilter,Flags)
#endif
#endif
  HRESULT WINAPI IGraphConfig_Reconnect_Proxy(IGraphConfig *This,IPin *pOutputPin,IPin *pInputPin,const AM_MEDIA_TYPE *pmtFirstConnection,IBaseFilter *pUsingFilter,HANDLE hAbortEvent,DWORD dwFlags);
  void __RPC_STUB IGraphConfig_Reconnect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IGraphConfig_Reconfigure_Proxy(IGraphConfig *This,IGraphConfigCallback *pCallback,PVOID pvContext,DWORD dwFlags,HANDLE hAbortEvent);
  void __RPC_STUB IGraphConfig_Reconfigure_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IGraphConfig_AddFilterToCache_Proxy(IGraphConfig *This,IBaseFilter *pFilter);
  void __RPC_STUB IGraphConfig_AddFilterToCache_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IGraphConfig_EnumCacheFilter_Proxy(IGraphConfig *This,IEnumFilters **pEnum);
  void __RPC_STUB IGraphConfig_EnumCacheFilter_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IGraphConfig_RemoveFilterFromCache_Proxy(IGraphConfig *This,IBaseFilter *pFilter);
  void __RPC_STUB IGraphConfig_RemoveFilterFromCache_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IGraphConfig_GetStartTime_Proxy(IGraphConfig *This,REFERENCE_TIME *prtStart);
  void __RPC_STUB IGraphConfig_GetStartTime_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IGraphConfig_PushThroughData_Proxy(IGraphConfig *This,IPin *pOutputPin,IPinConnection *pConnection,HANDLE hEventAbort);
  void __RPC_STUB IGraphConfig_PushThroughData_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IGraphConfig_SetFilterFlags_Proxy(IGraphConfig *This,IBaseFilter *pFilter,DWORD dwFlags);
  void __RPC_STUB IGraphConfig_SetFilterFlags_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IGraphConfig_GetFilterFlags_Proxy(IGraphConfig *This,IBaseFilter *pFilter,DWORD *pdwFlags);
  void __RPC_STUB IGraphConfig_GetFilterFlags_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IGraphConfig_RemoveFilterEx_Proxy(IGraphConfig *This,IBaseFilter *pFilter,DWORD Flags);
  void __RPC_STUB IGraphConfig_RemoveFilterEx_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IGraphConfigCallback_INTERFACE_DEFINED__
#define __IGraphConfigCallback_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IGraphConfigCallback;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IGraphConfigCallback : public IUnknown {
  public:
    virtual HRESULT WINAPI Reconfigure(PVOID pvContext,DWORD dwFlags) = 0;
  };
#else
  typedef struct IGraphConfigCallbackVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IGraphConfigCallback *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IGraphConfigCallback *This);
      ULONG (WINAPI *Release)(IGraphConfigCallback *This);
      HRESULT (WINAPI *Reconfigure)(IGraphConfigCallback *This,PVOID pvContext,DWORD dwFlags);
    END_INTERFACE
  } IGraphConfigCallbackVtbl;
  struct IGraphConfigCallback {
    CONST_VTBL struct IGraphConfigCallbackVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IGraphConfigCallback_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IGraphConfigCallback_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IGraphConfigCallback_Release(This) (This)->lpVtbl->Release(This)
#define IGraphConfigCallback_Reconfigure(This,pvContext,dwFlags) (This)->lpVtbl->Reconfigure(This,pvContext,dwFlags)
#endif
#endif
  HRESULT WINAPI IGraphConfigCallback_Reconfigure_Proxy(IGraphConfigCallback *This,PVOID pvContext,DWORD dwFlags);
  void __RPC_STUB IGraphConfigCallback_Reconfigure_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IFilterChain_INTERFACE_DEFINED__
#define __IFilterChain_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IFilterChain;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IFilterChain : public IUnknown {
  public:
    virtual HRESULT WINAPI StartChain(IBaseFilter *pStartFilter,IBaseFilter *pEndFilter) = 0;
    virtual HRESULT WINAPI PauseChain(IBaseFilter *pStartFilter,IBaseFilter *pEndFilter) = 0;
    virtual HRESULT WINAPI StopChain(IBaseFilter *pStartFilter,IBaseFilter *pEndFilter) = 0;
    virtual HRESULT WINAPI RemoveChain(IBaseFilter *pStartFilter,IBaseFilter *pEndFilter) = 0;
  };
#else
  typedef struct IFilterChainVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IFilterChain *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IFilterChain *This);
      ULONG (WINAPI *Release)(IFilterChain *This);
      HRESULT (WINAPI *StartChain)(IFilterChain *This,IBaseFilter *pStartFilter,IBaseFilter *pEndFilter);
      HRESULT (WINAPI *PauseChain)(IFilterChain *This,IBaseFilter *pStartFilter,IBaseFilter *pEndFilter);
      HRESULT (WINAPI *StopChain)(IFilterChain *This,IBaseFilter *pStartFilter,IBaseFilter *pEndFilter);
      HRESULT (WINAPI *RemoveChain)(IFilterChain *This,IBaseFilter *pStartFilter,IBaseFilter *pEndFilter);
    END_INTERFACE
  } IFilterChainVtbl;
  struct IFilterChain {
    CONST_VTBL struct IFilterChainVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IFilterChain_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IFilterChain_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IFilterChain_Release(This) (This)->lpVtbl->Release(This)
#define IFilterChain_StartChain(This,pStartFilter,pEndFilter) (This)->lpVtbl->StartChain(This,pStartFilter,pEndFilter)
#define IFilterChain_PauseChain(This,pStartFilter,pEndFilter) (This)->lpVtbl->PauseChain(This,pStartFilter,pEndFilter)
#define IFilterChain_StopChain(This,pStartFilter,pEndFilter) (This)->lpVtbl->StopChain(This,pStartFilter,pEndFilter)
#define IFilterChain_RemoveChain(This,pStartFilter,pEndFilter) (This)->lpVtbl->RemoveChain(This,pStartFilter,pEndFilter)
#endif
#endif
  HRESULT WINAPI IFilterChain_StartChain_Proxy(IFilterChain *This,IBaseFilter *pStartFilter,IBaseFilter *pEndFilter);
  void __RPC_STUB IFilterChain_StartChain_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterChain_PauseChain_Proxy(IFilterChain *This,IBaseFilter *pStartFilter,IBaseFilter *pEndFilter);
  void __RPC_STUB IFilterChain_PauseChain_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterChain_StopChain_Proxy(IFilterChain *This,IBaseFilter *pStartFilter,IBaseFilter *pEndFilter);
  void __RPC_STUB IFilterChain_StopChain_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterChain_RemoveChain_Proxy(IFilterChain *This,IBaseFilter *pStartFilter,IBaseFilter *pEndFilter);
  void __RPC_STUB IFilterChain_RemoveChain_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifdef MINGW_HAS_DDRAW_H
#include <dshow/ddraw.h>
#endif

  typedef enum __MIDL___MIDL_itf_strmif_0397_0002 {
    VMRSample_SyncPoint = 0x1,VMRSample_Preroll = 0x2,VMRSample_Discontinuity = 0x4,VMRSample_TimeValid = 0x8,VMRSample_SrcDstRectsValid = 0x10
  } VMRPresentationFlags;

  typedef struct tagVMRPRESENTATIONINFO {
    DWORD dwFlags;
    LPDIRECTDRAWSURFACE7 lpSurf;
    REFERENCE_TIME rtStart;
    REFERENCE_TIME rtEnd;
    SIZE szAspectRatio;
    RECT rcSrc;
    RECT rcDst;
    DWORD dwTypeSpecificFlags;
    DWORD dwInterlaceFlags;
  } VMRPRESENTATIONINFO;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0397_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0397_v0_0_s_ifspec;
#ifndef __IVMRImagePresenter_INTERFACE_DEFINED__
#define __IVMRImagePresenter_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IVMRImagePresenter;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IVMRImagePresenter : public IUnknown {
  public:
    virtual HRESULT WINAPI StartPresenting(DWORD_PTR dwUserID) = 0;
    virtual HRESULT WINAPI StopPresenting(DWORD_PTR dwUserID) = 0;
    virtual HRESULT WINAPI PresentImage(DWORD_PTR dwUserID,VMRPRESENTATIONINFO *lpPresInfo) = 0;
  };
#else
  typedef struct IVMRImagePresenterVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IVMRImagePresenter *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IVMRImagePresenter *This);
      ULONG (WINAPI *Release)(IVMRImagePresenter *This);
      HRESULT (WINAPI *StartPresenting)(IVMRImagePresenter *This,DWORD_PTR dwUserID);
      HRESULT (WINAPI *StopPresenting)(IVMRImagePresenter *This,DWORD_PTR dwUserID);
      HRESULT (WINAPI *PresentImage)(IVMRImagePresenter *This,DWORD_PTR dwUserID,VMRPRESENTATIONINFO *lpPresInfo);
    END_INTERFACE
  } IVMRImagePresenterVtbl;
  struct IVMRImagePresenter {
    CONST_VTBL struct IVMRImagePresenterVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IVMRImagePresenter_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IVMRImagePresenter_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IVMRImagePresenter_Release(This) (This)->lpVtbl->Release(This)
#define IVMRImagePresenter_StartPresenting(This,dwUserID) (This)->lpVtbl->StartPresenting(This,dwUserID)
#define IVMRImagePresenter_StopPresenting(This,dwUserID) (This)->lpVtbl->StopPresenting(This,dwUserID)
#define IVMRImagePresenter_PresentImage(This,dwUserID,lpPresInfo) (This)->lpVtbl->PresentImage(This,dwUserID,lpPresInfo)
#endif
#endif
  HRESULT WINAPI IVMRImagePresenter_StartPresenting_Proxy(IVMRImagePresenter *This,DWORD_PTR dwUserID);
  void __RPC_STUB IVMRImagePresenter_StartPresenting_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRImagePresenter_StopPresenting_Proxy(IVMRImagePresenter *This,DWORD_PTR dwUserID);
  void __RPC_STUB IVMRImagePresenter_StopPresenting_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRImagePresenter_PresentImage_Proxy(IVMRImagePresenter *This,DWORD_PTR dwUserID,VMRPRESENTATIONINFO *lpPresInfo);
  void __RPC_STUB IVMRImagePresenter_PresentImage_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef enum __MIDL___MIDL_itf_strmif_0398_0001 {
    AMAP_PIXELFORMAT_VALID = 0x1,AMAP_3D_TARGET = 0x2,AMAP_ALLOW_SYSMEM = 0x4,AMAP_FORCE_SYSMEM = 0x8,AMAP_DIRECTED_FLIP = 0x10,AMAP_DXVA_TARGET = 0x20
  } VMRSurfaceAllocationFlags;

  typedef struct tagVMRALLOCATIONINFO {
    DWORD dwFlags;
    LPBITMAPINFOHEADER lpHdr;
    LPDDPIXELFORMAT lpPixFmt;
    SIZE szAspectRatio;
    DWORD dwMinBuffers;
    DWORD dwMaxBuffers;
    DWORD dwInterlaceFlags;
    SIZE szNativeSize;
  } VMRALLOCATIONINFO;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0398_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0398_v0_0_s_ifspec;
#ifndef __IVMRSurfaceAllocator_INTERFACE_DEFINED__
#define __IVMRSurfaceAllocator_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IVMRSurfaceAllocator;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IVMRSurfaceAllocator : public IUnknown {
  public:
    virtual HRESULT WINAPI AllocateSurface(DWORD_PTR dwUserID,VMRALLOCATIONINFO *lpAllocInfo,DWORD *lpdwActualBuffers,LPDIRECTDRAWSURFACE7 *lplpSurface) = 0;
    virtual HRESULT WINAPI FreeSurface(DWORD_PTR dwID) = 0;
    virtual HRESULT WINAPI PrepareSurface(DWORD_PTR dwUserID,LPDIRECTDRAWSURFACE7 lpSurface,DWORD dwSurfaceFlags) = 0;
    virtual HRESULT WINAPI AdviseNotify(IVMRSurfaceAllocatorNotify *lpIVMRSurfAllocNotify) = 0;
  };
#else
  typedef struct IVMRSurfaceAllocatorVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IVMRSurfaceAllocator *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IVMRSurfaceAllocator *This);
      ULONG (WINAPI *Release)(IVMRSurfaceAllocator *This);
      HRESULT (WINAPI *AllocateSurface)(IVMRSurfaceAllocator *This,DWORD_PTR dwUserID,VMRALLOCATIONINFO *lpAllocInfo,DWORD *lpdwActualBuffers,LPDIRECTDRAWSURFACE7 *lplpSurface);
      HRESULT (WINAPI *FreeSurface)(IVMRSurfaceAllocator *This,DWORD_PTR dwID);
      HRESULT (WINAPI *PrepareSurface)(IVMRSurfaceAllocator *This,DWORD_PTR dwUserID,LPDIRECTDRAWSURFACE7 lpSurface,DWORD dwSurfaceFlags);
      HRESULT (WINAPI *AdviseNotify)(IVMRSurfaceAllocator *This,IVMRSurfaceAllocatorNotify *lpIVMRSurfAllocNotify);
    END_INTERFACE
  } IVMRSurfaceAllocatorVtbl;
  struct IVMRSurfaceAllocator {
    CONST_VTBL struct IVMRSurfaceAllocatorVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IVMRSurfaceAllocator_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IVMRSurfaceAllocator_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IVMRSurfaceAllocator_Release(This) (This)->lpVtbl->Release(This)
#define IVMRSurfaceAllocator_AllocateSurface(This,dwUserID,lpAllocInfo,lpdwActualBuffers,lplpSurface) (This)->lpVtbl->AllocateSurface(This,dwUserID,lpAllocInfo,lpdwActualBuffers,lplpSurface)
#define IVMRSurfaceAllocator_FreeSurface(This,dwID) (This)->lpVtbl->FreeSurface(This,dwID)
#define IVMRSurfaceAllocator_PrepareSurface(This,dwUserID,lpSurface,dwSurfaceFlags) (This)->lpVtbl->PrepareSurface(This,dwUserID,lpSurface,dwSurfaceFlags)
#define IVMRSurfaceAllocator_AdviseNotify(This,lpIVMRSurfAllocNotify) (This)->lpVtbl->AdviseNotify(This,lpIVMRSurfAllocNotify)
#endif
#endif
  HRESULT WINAPI IVMRSurfaceAllocator_AllocateSurface_Proxy(IVMRSurfaceAllocator *This,DWORD_PTR dwUserID,VMRALLOCATIONINFO *lpAllocInfo,DWORD *lpdwActualBuffers,LPDIRECTDRAWSURFACE7 *lplpSurface);
  void __RPC_STUB IVMRSurfaceAllocator_AllocateSurface_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRSurfaceAllocator_FreeSurface_Proxy(IVMRSurfaceAllocator *This,DWORD_PTR dwID);
  void __RPC_STUB IVMRSurfaceAllocator_FreeSurface_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRSurfaceAllocator_PrepareSurface_Proxy(IVMRSurfaceAllocator *This,DWORD_PTR dwUserID,LPDIRECTDRAWSURFACE7 lpSurface,DWORD dwSurfaceFlags);
  void __RPC_STUB IVMRSurfaceAllocator_PrepareSurface_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRSurfaceAllocator_AdviseNotify_Proxy(IVMRSurfaceAllocator *This,IVMRSurfaceAllocatorNotify *lpIVMRSurfAllocNotify);
  void __RPC_STUB IVMRSurfaceAllocator_AdviseNotify_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IVMRSurfaceAllocatorNotify_INTERFACE_DEFINED__
#define __IVMRSurfaceAllocatorNotify_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IVMRSurfaceAllocatorNotify;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IVMRSurfaceAllocatorNotify : public IUnknown {
  public:
    virtual HRESULT WINAPI AdviseSurfaceAllocator(DWORD_PTR dwUserID,IVMRSurfaceAllocator *lpIVRMSurfaceAllocator) = 0;
    virtual HRESULT WINAPI SetDDrawDevice(LPDIRECTDRAW7 lpDDrawDevice,HMONITOR hMonitor) = 0;
    virtual HRESULT WINAPI ChangeDDrawDevice(LPDIRECTDRAW7 lpDDrawDevice,HMONITOR hMonitor) = 0;
    virtual HRESULT WINAPI RestoreDDrawSurfaces(void) = 0;
    virtual HRESULT WINAPI NotifyEvent(LONG EventCode,LONG_PTR Param1,LONG_PTR Param2) = 0;
    virtual HRESULT WINAPI SetBorderColor(COLORREF clrBorder) = 0;
  };
#else
  typedef struct IVMRSurfaceAllocatorNotifyVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IVMRSurfaceAllocatorNotify *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IVMRSurfaceAllocatorNotify *This);
      ULONG (WINAPI *Release)(IVMRSurfaceAllocatorNotify *This);
      HRESULT (WINAPI *AdviseSurfaceAllocator)(IVMRSurfaceAllocatorNotify *This,DWORD_PTR dwUserID,IVMRSurfaceAllocator *lpIVRMSurfaceAllocator);
      HRESULT (WINAPI *SetDDrawDevice)(IVMRSurfaceAllocatorNotify *This,LPDIRECTDRAW7 lpDDrawDevice,HMONITOR hMonitor);
      HRESULT (WINAPI *ChangeDDrawDevice)(IVMRSurfaceAllocatorNotify *This,LPDIRECTDRAW7 lpDDrawDevice,HMONITOR hMonitor);
      HRESULT (WINAPI *RestoreDDrawSurfaces)(IVMRSurfaceAllocatorNotify *This);
      HRESULT (WINAPI *NotifyEvent)(IVMRSurfaceAllocatorNotify *This,LONG EventCode,LONG_PTR Param1,LONG_PTR Param2);
      HRESULT (WINAPI *SetBorderColor)(IVMRSurfaceAllocatorNotify *This,COLORREF clrBorder);
    END_INTERFACE
  } IVMRSurfaceAllocatorNotifyVtbl;
  struct IVMRSurfaceAllocatorNotify {
    CONST_VTBL struct IVMRSurfaceAllocatorNotifyVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IVMRSurfaceAllocatorNotify_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IVMRSurfaceAllocatorNotify_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IVMRSurfaceAllocatorNotify_Release(This) (This)->lpVtbl->Release(This)
#define IVMRSurfaceAllocatorNotify_AdviseSurfaceAllocator(This,dwUserID,lpIVRMSurfaceAllocator) (This)->lpVtbl->AdviseSurfaceAllocator(This,dwUserID,lpIVRMSurfaceAllocator)
#define IVMRSurfaceAllocatorNotify_SetDDrawDevice(This,lpDDrawDevice,hMonitor) (This)->lpVtbl->SetDDrawDevice(This,lpDDrawDevice,hMonitor)
#define IVMRSurfaceAllocatorNotify_ChangeDDrawDevice(This,lpDDrawDevice,hMonitor) (This)->lpVtbl->ChangeDDrawDevice(This,lpDDrawDevice,hMonitor)
#define IVMRSurfaceAllocatorNotify_RestoreDDrawSurfaces(This) (This)->lpVtbl->RestoreDDrawSurfaces(This)
#define IVMRSurfaceAllocatorNotify_NotifyEvent(This,EventCode,Param1,Param2) (This)->lpVtbl->NotifyEvent(This,EventCode,Param1,Param2)
#define IVMRSurfaceAllocatorNotify_SetBorderColor(This,clrBorder) (This)->lpVtbl->SetBorderColor(This,clrBorder)
#endif
#endif
  HRESULT WINAPI IVMRSurfaceAllocatorNotify_AdviseSurfaceAllocator_Proxy(IVMRSurfaceAllocatorNotify *This,DWORD_PTR dwUserID,IVMRSurfaceAllocator *lpIVRMSurfaceAllocator);
  void __RPC_STUB IVMRSurfaceAllocatorNotify_AdviseSurfaceAllocator_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRSurfaceAllocatorNotify_SetDDrawDevice_Proxy(IVMRSurfaceAllocatorNotify *This,LPDIRECTDRAW7 lpDDrawDevice,HMONITOR hMonitor);
  void __RPC_STUB IVMRSurfaceAllocatorNotify_SetDDrawDevice_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRSurfaceAllocatorNotify_ChangeDDrawDevice_Proxy(IVMRSurfaceAllocatorNotify *This,LPDIRECTDRAW7 lpDDrawDevice,HMONITOR hMonitor);
  void __RPC_STUB IVMRSurfaceAllocatorNotify_ChangeDDrawDevice_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRSurfaceAllocatorNotify_RestoreDDrawSurfaces_Proxy(IVMRSurfaceAllocatorNotify *This);
  void __RPC_STUB IVMRSurfaceAllocatorNotify_RestoreDDrawSurfaces_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRSurfaceAllocatorNotify_NotifyEvent_Proxy(IVMRSurfaceAllocatorNotify *This,LONG EventCode,LONG_PTR Param1,LONG_PTR Param2);
  void __RPC_STUB IVMRSurfaceAllocatorNotify_NotifyEvent_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRSurfaceAllocatorNotify_SetBorderColor_Proxy(IVMRSurfaceAllocatorNotify *This,COLORREF clrBorder);
  void __RPC_STUB IVMRSurfaceAllocatorNotify_SetBorderColor_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef enum __MIDL___MIDL_itf_strmif_0400_0001 {
    VMR_ARMODE_NONE = 0,VMR_ARMODE_LETTER_BOX = VMR_ARMODE_NONE + 1
  } VMR_ASPECT_RATIO_MODE;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0400_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0400_v0_0_s_ifspec;
#ifndef __IVMRWindowlessControl_INTERFACE_DEFINED__
#define __IVMRWindowlessControl_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IVMRWindowlessControl;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IVMRWindowlessControl : public IUnknown {
  public:
    virtual HRESULT WINAPI GetNativeVideoSize(LONG *lpWidth,LONG *lpHeight,LONG *lpARWidth,LONG *lpARHeight) = 0;
    virtual HRESULT WINAPI GetMinIdealVideoSize(LONG *lpWidth,LONG *lpHeight) = 0;
    virtual HRESULT WINAPI GetMaxIdealVideoSize(LONG *lpWidth,LONG *lpHeight) = 0;
    virtual HRESULT WINAPI SetVideoPosition(const LPRECT lpSRCRect,const LPRECT lpDSTRect) = 0;
    virtual HRESULT WINAPI GetVideoPosition(LPRECT lpSRCRect,LPRECT lpDSTRect) = 0;
    virtual HRESULT WINAPI GetAspectRatioMode(DWORD *lpAspectRatioMode) = 0;
    virtual HRESULT WINAPI SetAspectRatioMode(DWORD AspectRatioMode) = 0;
    virtual HRESULT WINAPI SetVideoClippingWindow(HWND hwnd) = 0;
    virtual HRESULT WINAPI RepaintVideo(HWND hwnd,HDC hdc) = 0;
    virtual HRESULT WINAPI DisplayModeChanged(void) = 0;
    virtual HRESULT WINAPI GetCurrentImage(BYTE **lpDib) = 0;
    virtual HRESULT WINAPI SetBorderColor(COLORREF Clr) = 0;
    virtual HRESULT WINAPI GetBorderColor(COLORREF *lpClr) = 0;
    virtual HRESULT WINAPI SetColorKey(COLORREF Clr) = 0;
    virtual HRESULT WINAPI GetColorKey(COLORREF *lpClr) = 0;
  };
#else
  typedef struct IVMRWindowlessControlVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IVMRWindowlessControl *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IVMRWindowlessControl *This);
      ULONG (WINAPI *Release)(IVMRWindowlessControl *This);
      HRESULT (WINAPI *GetNativeVideoSize)(IVMRWindowlessControl *This,LONG *lpWidth,LONG *lpHeight,LONG *lpARWidth,LONG *lpARHeight);
      HRESULT (WINAPI *GetMinIdealVideoSize)(IVMRWindowlessControl *This,LONG *lpWidth,LONG *lpHeight);
      HRESULT (WINAPI *GetMaxIdealVideoSize)(IVMRWindowlessControl *This,LONG *lpWidth,LONG *lpHeight);
      HRESULT (WINAPI *SetVideoPosition)(IVMRWindowlessControl *This,const LPRECT lpSRCRect,const LPRECT lpDSTRect);
      HRESULT (WINAPI *GetVideoPosition)(IVMRWindowlessControl *This,LPRECT lpSRCRect,LPRECT lpDSTRect);
      HRESULT (WINAPI *GetAspectRatioMode)(IVMRWindowlessControl *This,DWORD *lpAspectRatioMode);
      HRESULT (WINAPI *SetAspectRatioMode)(IVMRWindowlessControl *This,DWORD AspectRatioMode);
      HRESULT (WINAPI *SetVideoClippingWindow)(IVMRWindowlessControl *This,HWND hwnd);
      HRESULT (WINAPI *RepaintVideo)(IVMRWindowlessControl *This,HWND hwnd,HDC hdc);
      HRESULT (WINAPI *DisplayModeChanged)(IVMRWindowlessControl *This);
      HRESULT (WINAPI *GetCurrentImage)(IVMRWindowlessControl *This,BYTE **lpDib);
      HRESULT (WINAPI *SetBorderColor)(IVMRWindowlessControl *This,COLORREF Clr);
      HRESULT (WINAPI *GetBorderColor)(IVMRWindowlessControl *This,COLORREF *lpClr);
      HRESULT (WINAPI *SetColorKey)(IVMRWindowlessControl *This,COLORREF Clr);
      HRESULT (WINAPI *GetColorKey)(IVMRWindowlessControl *This,COLORREF *lpClr);
    END_INTERFACE
  } IVMRWindowlessControlVtbl;
  struct IVMRWindowlessControl {
    CONST_VTBL struct IVMRWindowlessControlVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IVMRWindowlessControl_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IVMRWindowlessControl_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IVMRWindowlessControl_Release(This) (This)->lpVtbl->Release(This)
#define IVMRWindowlessControl_GetNativeVideoSize(This,lpWidth,lpHeight,lpARWidth,lpARHeight) (This)->lpVtbl->GetNativeVideoSize(This,lpWidth,lpHeight,lpARWidth,lpARHeight)
#define IVMRWindowlessControl_GetMinIdealVideoSize(This,lpWidth,lpHeight) (This)->lpVtbl->GetMinIdealVideoSize(This,lpWidth,lpHeight)
#define IVMRWindowlessControl_GetMaxIdealVideoSize(This,lpWidth,lpHeight) (This)->lpVtbl->GetMaxIdealVideoSize(This,lpWidth,lpHeight)
#define IVMRWindowlessControl_SetVideoPosition(This,lpSRCRect,lpDSTRect) (This)->lpVtbl->SetVideoPosition(This,lpSRCRect,lpDSTRect)
#define IVMRWindowlessControl_GetVideoPosition(This,lpSRCRect,lpDSTRect) (This)->lpVtbl->GetVideoPosition(This,lpSRCRect,lpDSTRect)
#define IVMRWindowlessControl_GetAspectRatioMode(This,lpAspectRatioMode) (This)->lpVtbl->GetAspectRatioMode(This,lpAspectRatioMode)
#define IVMRWindowlessControl_SetAspectRatioMode(This,AspectRatioMode) (This)->lpVtbl->SetAspectRatioMode(This,AspectRatioMode)
#define IVMRWindowlessControl_SetVideoClippingWindow(This,hwnd) (This)->lpVtbl->SetVideoClippingWindow(This,hwnd)
#define IVMRWindowlessControl_RepaintVideo(This,hwnd,hdc) (This)->lpVtbl->RepaintVideo(This,hwnd,hdc)
#define IVMRWindowlessControl_DisplayModeChanged(This) (This)->lpVtbl->DisplayModeChanged(This)
#define IVMRWindowlessControl_GetCurrentImage(This,lpDib) (This)->lpVtbl->GetCurrentImage(This,lpDib)
#define IVMRWindowlessControl_SetBorderColor(This,Clr) (This)->lpVtbl->SetBorderColor(This,Clr)
#define IVMRWindowlessControl_GetBorderColor(This,lpClr) (This)->lpVtbl->GetBorderColor(This,lpClr)
#define IVMRWindowlessControl_SetColorKey(This,Clr) (This)->lpVtbl->SetColorKey(This,Clr)
#define IVMRWindowlessControl_GetColorKey(This,lpClr) (This)->lpVtbl->GetColorKey(This,lpClr)
#endif
#endif
  HRESULT WINAPI IVMRWindowlessControl_GetNativeVideoSize_Proxy(IVMRWindowlessControl *This,LONG *lpWidth,LONG *lpHeight,LONG *lpARWidth,LONG *lpARHeight);
  void __RPC_STUB IVMRWindowlessControl_GetNativeVideoSize_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRWindowlessControl_GetMinIdealVideoSize_Proxy(IVMRWindowlessControl *This,LONG *lpWidth,LONG *lpHeight);
  void __RPC_STUB IVMRWindowlessControl_GetMinIdealVideoSize_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRWindowlessControl_GetMaxIdealVideoSize_Proxy(IVMRWindowlessControl *This,LONG *lpWidth,LONG *lpHeight);
  void __RPC_STUB IVMRWindowlessControl_GetMaxIdealVideoSize_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRWindowlessControl_SetVideoPosition_Proxy(IVMRWindowlessControl *This,const LPRECT lpSRCRect,const LPRECT lpDSTRect);
  void __RPC_STUB IVMRWindowlessControl_SetVideoPosition_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRWindowlessControl_GetVideoPosition_Proxy(IVMRWindowlessControl *This,LPRECT lpSRCRect,LPRECT lpDSTRect);
  void __RPC_STUB IVMRWindowlessControl_GetVideoPosition_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRWindowlessControl_GetAspectRatioMode_Proxy(IVMRWindowlessControl *This,DWORD *lpAspectRatioMode);
  void __RPC_STUB IVMRWindowlessControl_GetAspectRatioMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRWindowlessControl_SetAspectRatioMode_Proxy(IVMRWindowlessControl *This,DWORD AspectRatioMode);
  void __RPC_STUB IVMRWindowlessControl_SetAspectRatioMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRWindowlessControl_SetVideoClippingWindow_Proxy(IVMRWindowlessControl *This,HWND hwnd);
  void __RPC_STUB IVMRWindowlessControl_SetVideoClippingWindow_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRWindowlessControl_RepaintVideo_Proxy(IVMRWindowlessControl *This,HWND hwnd,HDC hdc);
  void __RPC_STUB IVMRWindowlessControl_RepaintVideo_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRWindowlessControl_DisplayModeChanged_Proxy(IVMRWindowlessControl *This);
  void __RPC_STUB IVMRWindowlessControl_DisplayModeChanged_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRWindowlessControl_GetCurrentImage_Proxy(IVMRWindowlessControl *This,BYTE **lpDib);
  void __RPC_STUB IVMRWindowlessControl_GetCurrentImage_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRWindowlessControl_SetBorderColor_Proxy(IVMRWindowlessControl *This,COLORREF Clr);
  void __RPC_STUB IVMRWindowlessControl_SetBorderColor_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRWindowlessControl_GetBorderColor_Proxy(IVMRWindowlessControl *This,COLORREF *lpClr);
  void __RPC_STUB IVMRWindowlessControl_GetBorderColor_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRWindowlessControl_SetColorKey_Proxy(IVMRWindowlessControl *This,COLORREF Clr);
  void __RPC_STUB IVMRWindowlessControl_SetColorKey_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRWindowlessControl_GetColorKey_Proxy(IVMRWindowlessControl *This,COLORREF *lpClr);
  void __RPC_STUB IVMRWindowlessControl_GetColorKey_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef enum __MIDL___MIDL_itf_strmif_0401_0001 {
    MixerPref_NoDecimation = 0x1,MixerPref_DecimateOutput = 0x2,MixerPref_ARAdjustXorY = 0x4,MixerPref_DecimationReserved = 0x8,
    MixerPref_DecimateMask = 0xf,MixerPref_BiLinearFiltering = 0x10,MixerPref_PointFiltering = 0x20,MixerPref_FilteringMask = 0xf0,
    MixerPref_RenderTargetRGB = 0x100,MixerPref_RenderTargetYUV = 0x1000,MixerPref_RenderTargetYUV420 = 0x200,MixerPref_RenderTargetYUV422 = 0x400,
    MixerPref_RenderTargetYUV444 = 0x800,MixerPref_RenderTargetReserved = 0xe000,MixerPref_RenderTargetMask = 0xff00,
    MixerPref_DynamicSwitchToBOB = 0x10000,MixerPref_DynamicDecimateBy2 = 0x20000,MixerPref_DynamicReserved = 0xc0000,
    MixerPref_DynamicMask = 0xf0000
  } VMRMixerPrefs;

  typedef struct _NORMALIZEDRECT {
    float left;
    float top;
    float right;
    float bottom;
  } NORMALIZEDRECT;

  typedef struct _NORMALIZEDRECT *PNORMALIZEDRECT;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0401_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0401_v0_0_s_ifspec;
#ifndef __IVMRMixerControl_INTERFACE_DEFINED__
#define __IVMRMixerControl_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IVMRMixerControl;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IVMRMixerControl : public IUnknown {
  public:
    virtual HRESULT WINAPI SetAlpha(DWORD dwStreamID,float Alpha) = 0;
    virtual HRESULT WINAPI GetAlpha(DWORD dwStreamID,float *pAlpha) = 0;
    virtual HRESULT WINAPI SetZOrder(DWORD dwStreamID,DWORD dwZ) = 0;
    virtual HRESULT WINAPI GetZOrder(DWORD dwStreamID,DWORD *pZ) = 0;
    virtual HRESULT WINAPI SetOutputRect(DWORD dwStreamID,const NORMALIZEDRECT *pRect) = 0;
    virtual HRESULT WINAPI GetOutputRect(DWORD dwStreamID,NORMALIZEDRECT *pRect) = 0;
    virtual HRESULT WINAPI SetBackgroundClr(COLORREF ClrBkg) = 0;
    virtual HRESULT WINAPI GetBackgroundClr(COLORREF *lpClrBkg) = 0;
    virtual HRESULT WINAPI SetMixingPrefs(DWORD dwMixerPrefs) = 0;
    virtual HRESULT WINAPI GetMixingPrefs(DWORD *pdwMixerPrefs) = 0;
  };
#else
  typedef struct IVMRMixerControlVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IVMRMixerControl *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IVMRMixerControl *This);
      ULONG (WINAPI *Release)(IVMRMixerControl *This);
      HRESULT (WINAPI *SetAlpha)(IVMRMixerControl *This,DWORD dwStreamID,float Alpha);
      HRESULT (WINAPI *GetAlpha)(IVMRMixerControl *This,DWORD dwStreamID,float *pAlpha);
      HRESULT (WINAPI *SetZOrder)(IVMRMixerControl *This,DWORD dwStreamID,DWORD dwZ);
      HRESULT (WINAPI *GetZOrder)(IVMRMixerControl *This,DWORD dwStreamID,DWORD *pZ);
      HRESULT (WINAPI *SetOutputRect)(IVMRMixerControl *This,DWORD dwStreamID,const NORMALIZEDRECT *pRect);
      HRESULT (WINAPI *GetOutputRect)(IVMRMixerControl *This,DWORD dwStreamID,NORMALIZEDRECT *pRect);
      HRESULT (WINAPI *SetBackgroundClr)(IVMRMixerControl *This,COLORREF ClrBkg);
      HRESULT (WINAPI *GetBackgroundClr)(IVMRMixerControl *This,COLORREF *lpClrBkg);
      HRESULT (WINAPI *SetMixingPrefs)(IVMRMixerControl *This,DWORD dwMixerPrefs);
      HRESULT (WINAPI *GetMixingPrefs)(IVMRMixerControl *This,DWORD *pdwMixerPrefs);
    END_INTERFACE
  } IVMRMixerControlVtbl;
  struct IVMRMixerControl {
    CONST_VTBL struct IVMRMixerControlVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IVMRMixerControl_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IVMRMixerControl_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IVMRMixerControl_Release(This) (This)->lpVtbl->Release(This)
#define IVMRMixerControl_SetAlpha(This,dwStreamID,Alpha) (This)->lpVtbl->SetAlpha(This,dwStreamID,Alpha)
#define IVMRMixerControl_GetAlpha(This,dwStreamID,pAlpha) (This)->lpVtbl->GetAlpha(This,dwStreamID,pAlpha)
#define IVMRMixerControl_SetZOrder(This,dwStreamID,dwZ) (This)->lpVtbl->SetZOrder(This,dwStreamID,dwZ)
#define IVMRMixerControl_GetZOrder(This,dwStreamID,pZ) (This)->lpVtbl->GetZOrder(This,dwStreamID,pZ)
#define IVMRMixerControl_SetOutputRect(This,dwStreamID,pRect) (This)->lpVtbl->SetOutputRect(This,dwStreamID,pRect)
#define IVMRMixerControl_GetOutputRect(This,dwStreamID,pRect) (This)->lpVtbl->GetOutputRect(This,dwStreamID,pRect)
#define IVMRMixerControl_SetBackgroundClr(This,ClrBkg) (This)->lpVtbl->SetBackgroundClr(This,ClrBkg)
#define IVMRMixerControl_GetBackgroundClr(This,lpClrBkg) (This)->lpVtbl->GetBackgroundClr(This,lpClrBkg)
#define IVMRMixerControl_SetMixingPrefs(This,dwMixerPrefs) (This)->lpVtbl->SetMixingPrefs(This,dwMixerPrefs)
#define IVMRMixerControl_GetMixingPrefs(This,pdwMixerPrefs) (This)->lpVtbl->GetMixingPrefs(This,pdwMixerPrefs)
#endif
#endif
  HRESULT WINAPI IVMRMixerControl_SetAlpha_Proxy(IVMRMixerControl *This,DWORD dwStreamID,float Alpha);
  void __RPC_STUB IVMRMixerControl_SetAlpha_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRMixerControl_GetAlpha_Proxy(IVMRMixerControl *This,DWORD dwStreamID,float *pAlpha);
  void __RPC_STUB IVMRMixerControl_GetAlpha_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRMixerControl_SetZOrder_Proxy(IVMRMixerControl *This,DWORD dwStreamID,DWORD dwZ);
  void __RPC_STUB IVMRMixerControl_SetZOrder_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRMixerControl_GetZOrder_Proxy(IVMRMixerControl *This,DWORD dwStreamID,DWORD *pZ);
  void __RPC_STUB IVMRMixerControl_GetZOrder_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRMixerControl_SetOutputRect_Proxy(IVMRMixerControl *This,DWORD dwStreamID,const NORMALIZEDRECT *pRect);
  void __RPC_STUB IVMRMixerControl_SetOutputRect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRMixerControl_GetOutputRect_Proxy(IVMRMixerControl *This,DWORD dwStreamID,NORMALIZEDRECT *pRect);
  void __RPC_STUB IVMRMixerControl_GetOutputRect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRMixerControl_SetBackgroundClr_Proxy(IVMRMixerControl *This,COLORREF ClrBkg);
  void __RPC_STUB IVMRMixerControl_SetBackgroundClr_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRMixerControl_GetBackgroundClr_Proxy(IVMRMixerControl *This,COLORREF *lpClrBkg);
  void __RPC_STUB IVMRMixerControl_GetBackgroundClr_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRMixerControl_SetMixingPrefs_Proxy(IVMRMixerControl *This,DWORD dwMixerPrefs);
  void __RPC_STUB IVMRMixerControl_SetMixingPrefs_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRMixerControl_GetMixingPrefs_Proxy(IVMRMixerControl *This,DWORD *pdwMixerPrefs);

  void __RPC_STUB IVMRMixerControl_GetMixingPrefs_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifdef __cplusplus
  typedef struct tagVMRGUID {
    ::GUID *pGUID;
    ::GUID GUID;
  } VMRGUID;
#else
  typedef struct tagVMRGUID {
    GUID *pGUID;
    GUID GUID;
  } VMRGUID;
#endif

  typedef struct tagVMRMONITORINFO {
    VMRGUID guid;
    RECT rcMonitor;
    HMONITOR hMon;
    DWORD dwFlags;
    wchar_t szDevice[32];
    wchar_t szDescription[256];
    LARGE_INTEGER liDriverVersion;
    DWORD dwVendorId;
    DWORD dwDeviceId;
    DWORD dwSubSysId;
    DWORD dwRevision;
  } VMRMONITORINFO;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0402_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0402_v0_0_s_ifspec;
#ifndef __IVMRMonitorConfig_INTERFACE_DEFINED__
#define __IVMRMonitorConfig_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IVMRMonitorConfig;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IVMRMonitorConfig : public IUnknown {
  public:
    virtual HRESULT WINAPI SetMonitor(const VMRGUID *pGUID) = 0;
    virtual HRESULT WINAPI GetMonitor(VMRGUID *pGUID) = 0;
    virtual HRESULT WINAPI SetDefaultMonitor(const VMRGUID *pGUID) = 0;
    virtual HRESULT WINAPI GetDefaultMonitor(VMRGUID *pGUID) = 0;
    virtual HRESULT WINAPI GetAvailableMonitors(VMRMONITORINFO *pInfo,DWORD dwMaxInfoArraySize,DWORD *pdwNumDevices) = 0;
  };
#else
  typedef struct IVMRMonitorConfigVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IVMRMonitorConfig *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IVMRMonitorConfig *This);
      ULONG (WINAPI *Release)(IVMRMonitorConfig *This);
      HRESULT (WINAPI *SetMonitor)(IVMRMonitorConfig *This,const VMRGUID *pGUID);
      HRESULT (WINAPI *GetMonitor)(IVMRMonitorConfig *This,VMRGUID *pGUID);
      HRESULT (WINAPI *SetDefaultMonitor)(IVMRMonitorConfig *This,const VMRGUID *pGUID);
      HRESULT (WINAPI *GetDefaultMonitor)(IVMRMonitorConfig *This,VMRGUID *pGUID);
      HRESULT (WINAPI *GetAvailableMonitors)(IVMRMonitorConfig *This,VMRMONITORINFO *pInfo,DWORD dwMaxInfoArraySize,DWORD *pdwNumDevices);
    END_INTERFACE
  } IVMRMonitorConfigVtbl;
  struct IVMRMonitorConfig {
    CONST_VTBL struct IVMRMonitorConfigVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IVMRMonitorConfig_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IVMRMonitorConfig_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IVMRMonitorConfig_Release(This) (This)->lpVtbl->Release(This)
#define IVMRMonitorConfig_SetMonitor(This,pGUID) (This)->lpVtbl->SetMonitor(This,pGUID)
#define IVMRMonitorConfig_GetMonitor(This,pGUID) (This)->lpVtbl->GetMonitor(This,pGUID)
#define IVMRMonitorConfig_SetDefaultMonitor(This,pGUID) (This)->lpVtbl->SetDefaultMonitor(This,pGUID)
#define IVMRMonitorConfig_GetDefaultMonitor(This,pGUID) (This)->lpVtbl->GetDefaultMonitor(This,pGUID)
#define IVMRMonitorConfig_GetAvailableMonitors(This,pInfo,dwMaxInfoArraySize,pdwNumDevices) (This)->lpVtbl->GetAvailableMonitors(This,pInfo,dwMaxInfoArraySize,pdwNumDevices)
#endif
#endif
  HRESULT WINAPI IVMRMonitorConfig_SetMonitor_Proxy(IVMRMonitorConfig *This,const VMRGUID *pGUID);
  void __RPC_STUB IVMRMonitorConfig_SetMonitor_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRMonitorConfig_GetMonitor_Proxy(IVMRMonitorConfig *This,VMRGUID *pGUID);
  void __RPC_STUB IVMRMonitorConfig_GetMonitor_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRMonitorConfig_SetDefaultMonitor_Proxy(IVMRMonitorConfig *This,const VMRGUID *pGUID);
  void __RPC_STUB IVMRMonitorConfig_SetDefaultMonitor_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRMonitorConfig_GetDefaultMonitor_Proxy(IVMRMonitorConfig *This,VMRGUID *pGUID);
  void __RPC_STUB IVMRMonitorConfig_GetDefaultMonitor_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRMonitorConfig_GetAvailableMonitors_Proxy(IVMRMonitorConfig *This,VMRMONITORINFO *pInfo,DWORD dwMaxInfoArraySize,DWORD *pdwNumDevices);
  void __RPC_STUB IVMRMonitorConfig_GetAvailableMonitors_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef enum __MIDL___MIDL_itf_strmif_0403_0001 {
    RenderPrefs_RestrictToInitialMonitor = 0,RenderPrefs_ForceOffscreen = 0x1,RenderPrefs_ForceOverlays = 0x2,RenderPrefs_AllowOverlays = 0,
    RenderPrefs_AllowOffscreen = 0,RenderPrefs_DoNotRenderColorKeyAndBorder = 0x8,RenderPrefs_Reserved = 0x10,RenderPrefs_PreferAGPMemWhenMixing = 0x20,
    RenderPrefs_Mask = 0x3f
  } VMRRenderPrefs;

  typedef enum __MIDL___MIDL_itf_strmif_0403_0002 {
    VMRMode_Windowed = 0x1,VMRMode_Windowless = 0x2,VMRMode_Renderless = 0x4,VMRMode_Mask = 0x7
  } VMRMode;

  enum __MIDL___MIDL_itf_strmif_0403_0003 {
    MAX_NUMBER_OF_STREAMS = 16
  };

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0403_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0403_v0_0_s_ifspec;
#ifndef __IVMRFilterConfig_INTERFACE_DEFINED__
#define __IVMRFilterConfig_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IVMRFilterConfig;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IVMRFilterConfig : public IUnknown {
  public:
    virtual HRESULT WINAPI SetImageCompositor(IVMRImageCompositor *lpVMRImgCompositor) = 0;
    virtual HRESULT WINAPI SetNumberOfStreams(DWORD dwMaxStreams) = 0;
    virtual HRESULT WINAPI GetNumberOfStreams(DWORD *pdwMaxStreams) = 0;
    virtual HRESULT WINAPI SetRenderingPrefs(DWORD dwRenderFlags) = 0;
    virtual HRESULT WINAPI GetRenderingPrefs(DWORD *pdwRenderFlags) = 0;
    virtual HRESULT WINAPI SetRenderingMode(DWORD Mode) = 0;
    virtual HRESULT WINAPI GetRenderingMode(DWORD *pMode) = 0;
  };
#else
  typedef struct IVMRFilterConfigVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IVMRFilterConfig *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IVMRFilterConfig *This);
      ULONG (WINAPI *Release)(IVMRFilterConfig *This);
      HRESULT (WINAPI *SetImageCompositor)(IVMRFilterConfig *This,IVMRImageCompositor *lpVMRImgCompositor);
      HRESULT (WINAPI *SetNumberOfStreams)(IVMRFilterConfig *This,DWORD dwMaxStreams);
      HRESULT (WINAPI *GetNumberOfStreams)(IVMRFilterConfig *This,DWORD *pdwMaxStreams);
      HRESULT (WINAPI *SetRenderingPrefs)(IVMRFilterConfig *This,DWORD dwRenderFlags);
      HRESULT (WINAPI *GetRenderingPrefs)(IVMRFilterConfig *This,DWORD *pdwRenderFlags);
      HRESULT (WINAPI *SetRenderingMode)(IVMRFilterConfig *This,DWORD Mode);
      HRESULT (WINAPI *GetRenderingMode)(IVMRFilterConfig *This,DWORD *pMode);
    END_INTERFACE
  } IVMRFilterConfigVtbl;
  struct IVMRFilterConfig {
    CONST_VTBL struct IVMRFilterConfigVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IVMRFilterConfig_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IVMRFilterConfig_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IVMRFilterConfig_Release(This) (This)->lpVtbl->Release(This)
#define IVMRFilterConfig_SetImageCompositor(This,lpVMRImgCompositor) (This)->lpVtbl->SetImageCompositor(This,lpVMRImgCompositor)
#define IVMRFilterConfig_SetNumberOfStreams(This,dwMaxStreams) (This)->lpVtbl->SetNumberOfStreams(This,dwMaxStreams)
#define IVMRFilterConfig_GetNumberOfStreams(This,pdwMaxStreams) (This)->lpVtbl->GetNumberOfStreams(This,pdwMaxStreams)
#define IVMRFilterConfig_SetRenderingPrefs(This,dwRenderFlags) (This)->lpVtbl->SetRenderingPrefs(This,dwRenderFlags)
#define IVMRFilterConfig_GetRenderingPrefs(This,pdwRenderFlags) (This)->lpVtbl->GetRenderingPrefs(This,pdwRenderFlags)
#define IVMRFilterConfig_SetRenderingMode(This,Mode) (This)->lpVtbl->SetRenderingMode(This,Mode)
#define IVMRFilterConfig_GetRenderingMode(This,pMode) (This)->lpVtbl->GetRenderingMode(This,pMode)
#endif
#endif
  HRESULT WINAPI IVMRFilterConfig_SetImageCompositor_Proxy(IVMRFilterConfig *This,IVMRImageCompositor *lpVMRImgCompositor);
  void __RPC_STUB IVMRFilterConfig_SetImageCompositor_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRFilterConfig_SetNumberOfStreams_Proxy(IVMRFilterConfig *This,DWORD dwMaxStreams);
  void __RPC_STUB IVMRFilterConfig_SetNumberOfStreams_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRFilterConfig_GetNumberOfStreams_Proxy(IVMRFilterConfig *This,DWORD *pdwMaxStreams);
  void __RPC_STUB IVMRFilterConfig_GetNumberOfStreams_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRFilterConfig_SetRenderingPrefs_Proxy(IVMRFilterConfig *This,DWORD dwRenderFlags);
  void __RPC_STUB IVMRFilterConfig_SetRenderingPrefs_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRFilterConfig_GetRenderingPrefs_Proxy(IVMRFilterConfig *This,DWORD *pdwRenderFlags);
  void __RPC_STUB IVMRFilterConfig_GetRenderingPrefs_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRFilterConfig_SetRenderingMode_Proxy(IVMRFilterConfig *This,DWORD Mode);
  void __RPC_STUB IVMRFilterConfig_SetRenderingMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRFilterConfig_GetRenderingMode_Proxy(IVMRFilterConfig *This,DWORD *pMode);
  void __RPC_STUB IVMRFilterConfig_GetRenderingMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IVMRAspectRatioControl_INTERFACE_DEFINED__
#define __IVMRAspectRatioControl_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IVMRAspectRatioControl;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IVMRAspectRatioControl : public IUnknown {
  public:
    virtual HRESULT WINAPI GetAspectRatioMode(LPDWORD lpdwARMode) = 0;
    virtual HRESULT WINAPI SetAspectRatioMode(DWORD dwARMode) = 0;
  };
#else
  typedef struct IVMRAspectRatioControlVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IVMRAspectRatioControl *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IVMRAspectRatioControl *This);
      ULONG (WINAPI *Release)(IVMRAspectRatioControl *This);
      HRESULT (WINAPI *GetAspectRatioMode)(IVMRAspectRatioControl *This,LPDWORD lpdwARMode);
      HRESULT (WINAPI *SetAspectRatioMode)(IVMRAspectRatioControl *This,DWORD dwARMode);
    END_INTERFACE
  } IVMRAspectRatioControlVtbl;
  struct IVMRAspectRatioControl {
    CONST_VTBL struct IVMRAspectRatioControlVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IVMRAspectRatioControl_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IVMRAspectRatioControl_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IVMRAspectRatioControl_Release(This) (This)->lpVtbl->Release(This)
#define IVMRAspectRatioControl_GetAspectRatioMode(This,lpdwARMode) (This)->lpVtbl->GetAspectRatioMode(This,lpdwARMode)
#define IVMRAspectRatioControl_SetAspectRatioMode(This,dwARMode) (This)->lpVtbl->SetAspectRatioMode(This,dwARMode)
#endif
#endif
  HRESULT WINAPI IVMRAspectRatioControl_GetAspectRatioMode_Proxy(IVMRAspectRatioControl *This,LPDWORD lpdwARMode);
  void __RPC_STUB IVMRAspectRatioControl_GetAspectRatioMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRAspectRatioControl_SetAspectRatioMode_Proxy(IVMRAspectRatioControl *This,DWORD dwARMode);
  void __RPC_STUB IVMRAspectRatioControl_SetAspectRatioMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef enum __MIDL___MIDL_itf_strmif_0405_0001 {
    DeinterlacePref_NextBest = 0x1,DeinterlacePref_BOB = 0x2,DeinterlacePref_Weave = 0x4,DeinterlacePref_Mask = 0x7
  } VMRDeinterlacePrefs;

  typedef enum __MIDL___MIDL_itf_strmif_0405_0002 {
    DeinterlaceTech_Unknown = 0,DeinterlaceTech_BOBLineReplicate = 0x1,DeinterlaceTech_BOBVerticalStretch = 0x2,DeinterlaceTech_MedianFiltering = 0x4,
    DeinterlaceTech_EdgeFiltering = 0x10,DeinterlaceTech_FieldAdaptive = 0x20,DeinterlaceTech_PixelAdaptive = 0x40,
    DeinterlaceTech_MotionVectorSteered = 0x80
  } VMRDeinterlaceTech;

  typedef struct _VMRFrequency {
    DWORD dwNumerator;
    DWORD dwDenominator;
  } VMRFrequency;

  typedef struct _VMRVideoDesc {
    DWORD dwSize;
    DWORD dwSampleWidth;
    DWORD dwSampleHeight;
    WINBOOL SingleFieldPerSample;
    DWORD dwFourCC;
    VMRFrequency InputSampleFreq;
    VMRFrequency OutputFrameFreq;
  } VMRVideoDesc;

  typedef struct _VMRDeinterlaceCaps {
    DWORD dwSize;
    DWORD dwNumPreviousOutputFrames;
    DWORD dwNumForwardRefSamples;
    DWORD dwNumBackwardRefSamples;
    VMRDeinterlaceTech DeinterlaceTechnology;
  } VMRDeinterlaceCaps;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0405_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0405_v0_0_s_ifspec;
#ifndef __IVMRDeinterlaceControl_INTERFACE_DEFINED__
#define __IVMRDeinterlaceControl_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IVMRDeinterlaceControl;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IVMRDeinterlaceControl : public IUnknown {
  public:
    virtual HRESULT WINAPI GetNumberOfDeinterlaceModes(VMRVideoDesc *lpVideoDescription,LPDWORD lpdwNumDeinterlaceModes,LPGUID lpDeinterlaceModes) = 0;
    virtual HRESULT WINAPI GetDeinterlaceModeCaps(LPGUID lpDeinterlaceMode,VMRVideoDesc *lpVideoDescription,VMRDeinterlaceCaps *lpDeinterlaceCaps) = 0;
    virtual HRESULT WINAPI GetDeinterlaceMode(DWORD dwStreamID,LPGUID lpDeinterlaceMode) = 0;
    virtual HRESULT WINAPI SetDeinterlaceMode(DWORD dwStreamID,LPGUID lpDeinterlaceMode) = 0;
    virtual HRESULT WINAPI GetDeinterlacePrefs(LPDWORD lpdwDeinterlacePrefs) = 0;
    virtual HRESULT WINAPI SetDeinterlacePrefs(DWORD dwDeinterlacePrefs) = 0;
    virtual HRESULT WINAPI GetActualDeinterlaceMode(DWORD dwStreamID,LPGUID lpDeinterlaceMode) = 0;
  };
#else
  typedef struct IVMRDeinterlaceControlVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IVMRDeinterlaceControl *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IVMRDeinterlaceControl *This);
      ULONG (WINAPI *Release)(IVMRDeinterlaceControl *This);
      HRESULT (WINAPI *GetNumberOfDeinterlaceModes)(IVMRDeinterlaceControl *This,VMRVideoDesc *lpVideoDescription,LPDWORD lpdwNumDeinterlaceModes,LPGUID lpDeinterlaceModes);
      HRESULT (WINAPI *GetDeinterlaceModeCaps)(IVMRDeinterlaceControl *This,LPGUID lpDeinterlaceMode,VMRVideoDesc *lpVideoDescription,VMRDeinterlaceCaps *lpDeinterlaceCaps);
      HRESULT (WINAPI *GetDeinterlaceMode)(IVMRDeinterlaceControl *This,DWORD dwStreamID,LPGUID lpDeinterlaceMode);
      HRESULT (WINAPI *SetDeinterlaceMode)(IVMRDeinterlaceControl *This,DWORD dwStreamID,LPGUID lpDeinterlaceMode);
      HRESULT (WINAPI *GetDeinterlacePrefs)(IVMRDeinterlaceControl *This,LPDWORD lpdwDeinterlacePrefs);
      HRESULT (WINAPI *SetDeinterlacePrefs)(IVMRDeinterlaceControl *This,DWORD dwDeinterlacePrefs);
      HRESULT (WINAPI *GetActualDeinterlaceMode)(IVMRDeinterlaceControl *This,DWORD dwStreamID,LPGUID lpDeinterlaceMode);
    END_INTERFACE
  } IVMRDeinterlaceControlVtbl;
  struct IVMRDeinterlaceControl {
    CONST_VTBL struct IVMRDeinterlaceControlVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IVMRDeinterlaceControl_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IVMRDeinterlaceControl_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IVMRDeinterlaceControl_Release(This) (This)->lpVtbl->Release(This)
#define IVMRDeinterlaceControl_GetNumberOfDeinterlaceModes(This,lpVideoDescription,lpdwNumDeinterlaceModes,lpDeinterlaceModes) (This)->lpVtbl->GetNumberOfDeinterlaceModes(This,lpVideoDescription,lpdwNumDeinterlaceModes,lpDeinterlaceModes)
#define IVMRDeinterlaceControl_GetDeinterlaceModeCaps(This,lpDeinterlaceMode,lpVideoDescription,lpDeinterlaceCaps) (This)->lpVtbl->GetDeinterlaceModeCaps(This,lpDeinterlaceMode,lpVideoDescription,lpDeinterlaceCaps)
#define IVMRDeinterlaceControl_GetDeinterlaceMode(This,dwStreamID,lpDeinterlaceMode) (This)->lpVtbl->GetDeinterlaceMode(This,dwStreamID,lpDeinterlaceMode)
#define IVMRDeinterlaceControl_SetDeinterlaceMode(This,dwStreamID,lpDeinterlaceMode) (This)->lpVtbl->SetDeinterlaceMode(This,dwStreamID,lpDeinterlaceMode)
#define IVMRDeinterlaceControl_GetDeinterlacePrefs(This,lpdwDeinterlacePrefs) (This)->lpVtbl->GetDeinterlacePrefs(This,lpdwDeinterlacePrefs)
#define IVMRDeinterlaceControl_SetDeinterlacePrefs(This,dwDeinterlacePrefs) (This)->lpVtbl->SetDeinterlacePrefs(This,dwDeinterlacePrefs)
#define IVMRDeinterlaceControl_GetActualDeinterlaceMode(This,dwStreamID,lpDeinterlaceMode) (This)->lpVtbl->GetActualDeinterlaceMode(This,dwStreamID,lpDeinterlaceMode)
#endif
#endif
  HRESULT WINAPI IVMRDeinterlaceControl_GetNumberOfDeinterlaceModes_Proxy(IVMRDeinterlaceControl *This,VMRVideoDesc *lpVideoDescription,LPDWORD lpdwNumDeinterlaceModes,LPGUID lpDeinterlaceModes);
  void __RPC_STUB IVMRDeinterlaceControl_GetNumberOfDeinterlaceModes_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRDeinterlaceControl_GetDeinterlaceModeCaps_Proxy(IVMRDeinterlaceControl *This,LPGUID lpDeinterlaceMode,VMRVideoDesc *lpVideoDescription,VMRDeinterlaceCaps *lpDeinterlaceCaps);
  void __RPC_STUB IVMRDeinterlaceControl_GetDeinterlaceModeCaps_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRDeinterlaceControl_GetDeinterlaceMode_Proxy(IVMRDeinterlaceControl *This,DWORD dwStreamID,LPGUID lpDeinterlaceMode);
  void __RPC_STUB IVMRDeinterlaceControl_GetDeinterlaceMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRDeinterlaceControl_SetDeinterlaceMode_Proxy(IVMRDeinterlaceControl *This,DWORD dwStreamID,LPGUID lpDeinterlaceMode);
  void __RPC_STUB IVMRDeinterlaceControl_SetDeinterlaceMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRDeinterlaceControl_GetDeinterlacePrefs_Proxy(IVMRDeinterlaceControl *This,LPDWORD lpdwDeinterlacePrefs);
  void __RPC_STUB IVMRDeinterlaceControl_GetDeinterlacePrefs_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRDeinterlaceControl_SetDeinterlacePrefs_Proxy(IVMRDeinterlaceControl *This,DWORD dwDeinterlacePrefs);
  void __RPC_STUB IVMRDeinterlaceControl_SetDeinterlacePrefs_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRDeinterlaceControl_GetActualDeinterlaceMode_Proxy(IVMRDeinterlaceControl *This,DWORD dwStreamID,LPGUID lpDeinterlaceMode);
  void __RPC_STUB IVMRDeinterlaceControl_GetActualDeinterlaceMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef struct _VMRALPHABITMAP {
    DWORD dwFlags;
    HDC hdc;
    LPDIRECTDRAWSURFACE7 pDDS;
    RECT rSrc;
    NORMALIZEDRECT rDest;
    FLOAT fAlpha;
    COLORREF clrSrcKey;
  } VMRALPHABITMAP;

  typedef struct _VMRALPHABITMAP *PVMRALPHABITMAP;

#define VMRBITMAP_DISABLE 0x00000001
#define VMRBITMAP_HDC 0x00000002
#define VMRBITMAP_ENTIREDDS 0x00000004
#define VMRBITMAP_SRCCOLORKEY 0x00000008
#define VMRBITMAP_SRCRECT 0x00000010

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0406_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0406_v0_0_s_ifspec;
#ifndef __IVMRMixerBitmap_INTERFACE_DEFINED__
#define __IVMRMixerBitmap_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IVMRMixerBitmap;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IVMRMixerBitmap : public IUnknown {
  public:
    virtual HRESULT WINAPI SetAlphaBitmap(const VMRALPHABITMAP *pBmpParms) = 0;
    virtual HRESULT WINAPI UpdateAlphaBitmapParameters(PVMRALPHABITMAP pBmpParms) = 0;
    virtual HRESULT WINAPI GetAlphaBitmapParameters(PVMRALPHABITMAP pBmpParms) = 0;
  };
#else
  typedef struct IVMRMixerBitmapVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IVMRMixerBitmap *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IVMRMixerBitmap *This);
      ULONG (WINAPI *Release)(IVMRMixerBitmap *This);
      HRESULT (WINAPI *SetAlphaBitmap)(IVMRMixerBitmap *This,const VMRALPHABITMAP *pBmpParms);
      HRESULT (WINAPI *UpdateAlphaBitmapParameters)(IVMRMixerBitmap *This,PVMRALPHABITMAP pBmpParms);
      HRESULT (WINAPI *GetAlphaBitmapParameters)(IVMRMixerBitmap *This,PVMRALPHABITMAP pBmpParms);
    END_INTERFACE
  } IVMRMixerBitmapVtbl;
  struct IVMRMixerBitmap {
    CONST_VTBL struct IVMRMixerBitmapVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IVMRMixerBitmap_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IVMRMixerBitmap_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IVMRMixerBitmap_Release(This) (This)->lpVtbl->Release(This)
#define IVMRMixerBitmap_SetAlphaBitmap(This,pBmpParms) (This)->lpVtbl->SetAlphaBitmap(This,pBmpParms)
#define IVMRMixerBitmap_UpdateAlphaBitmapParameters(This,pBmpParms) (This)->lpVtbl->UpdateAlphaBitmapParameters(This,pBmpParms)
#define IVMRMixerBitmap_GetAlphaBitmapParameters(This,pBmpParms) (This)->lpVtbl->GetAlphaBitmapParameters(This,pBmpParms)
#endif
#endif
  HRESULT WINAPI IVMRMixerBitmap_SetAlphaBitmap_Proxy(IVMRMixerBitmap *This,const VMRALPHABITMAP *pBmpParms);
  void __RPC_STUB IVMRMixerBitmap_SetAlphaBitmap_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRMixerBitmap_UpdateAlphaBitmapParameters_Proxy(IVMRMixerBitmap *This,PVMRALPHABITMAP pBmpParms);
  void __RPC_STUB IVMRMixerBitmap_UpdateAlphaBitmapParameters_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRMixerBitmap_GetAlphaBitmapParameters_Proxy(IVMRMixerBitmap *This,PVMRALPHABITMAP pBmpParms);
  void __RPC_STUB IVMRMixerBitmap_GetAlphaBitmapParameters_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  typedef struct _VMRVIDEOSTREAMINFO {
    LPDIRECTDRAWSURFACE7 pddsVideoSurface;
    DWORD dwWidth;
    DWORD dwHeight;
    DWORD dwStrmID;
    FLOAT fAlpha;
    DDCOLORKEY ddClrKey;
    NORMALIZEDRECT rNormal;
  } VMRVIDEOSTREAMINFO;

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0407_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0407_v0_0_s_ifspec;
#ifndef __IVMRImageCompositor_INTERFACE_DEFINED__
#define __IVMRImageCompositor_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IVMRImageCompositor;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IVMRImageCompositor : public IUnknown {
  public:
    virtual HRESULT WINAPI InitCompositionTarget(IUnknown *pD3DDevice,LPDIRECTDRAWSURFACE7 pddsRenderTarget) = 0;
    virtual HRESULT WINAPI TermCompositionTarget(IUnknown *pD3DDevice,LPDIRECTDRAWSURFACE7 pddsRenderTarget) = 0;
    virtual HRESULT WINAPI SetStreamMediaType(DWORD dwStrmID,AM_MEDIA_TYPE *pmt,WINBOOL fTexture) = 0;
    virtual HRESULT WINAPI CompositeImage(IUnknown *pD3DDevice,LPDIRECTDRAWSURFACE7 pddsRenderTarget,AM_MEDIA_TYPE *pmtRenderTarget,REFERENCE_TIME rtStart,REFERENCE_TIME rtEnd,DWORD dwClrBkGnd,VMRVIDEOSTREAMINFO *pVideoStreamInfo,UINT cStreams) = 0;
  };
#else
  typedef struct IVMRImageCompositorVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IVMRImageCompositor *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IVMRImageCompositor *This);
      ULONG (WINAPI *Release)(IVMRImageCompositor *This);
      HRESULT (WINAPI *InitCompositionTarget)(IVMRImageCompositor *This,IUnknown *pD3DDevice,LPDIRECTDRAWSURFACE7 pddsRenderTarget);
      HRESULT (WINAPI *TermCompositionTarget)(IVMRImageCompositor *This,IUnknown *pD3DDevice,LPDIRECTDRAWSURFACE7 pddsRenderTarget);
      HRESULT (WINAPI *SetStreamMediaType)(IVMRImageCompositor *This,DWORD dwStrmID,AM_MEDIA_TYPE *pmt,WINBOOL fTexture);
      HRESULT (WINAPI *CompositeImage)(IVMRImageCompositor *This,IUnknown *pD3DDevice,LPDIRECTDRAWSURFACE7 pddsRenderTarget,AM_MEDIA_TYPE *pmtRenderTarget,REFERENCE_TIME rtStart,REFERENCE_TIME rtEnd,DWORD dwClrBkGnd,VMRVIDEOSTREAMINFO *pVideoStreamInfo,UINT cStreams);
    END_INTERFACE
  } IVMRImageCompositorVtbl;
  struct IVMRImageCompositor {
    CONST_VTBL struct IVMRImageCompositorVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IVMRImageCompositor_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IVMRImageCompositor_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IVMRImageCompositor_Release(This) (This)->lpVtbl->Release(This)
#define IVMRImageCompositor_InitCompositionTarget(This,pD3DDevice,pddsRenderTarget) (This)->lpVtbl->InitCompositionTarget(This,pD3DDevice,pddsRenderTarget)
#define IVMRImageCompositor_TermCompositionTarget(This,pD3DDevice,pddsRenderTarget) (This)->lpVtbl->TermCompositionTarget(This,pD3DDevice,pddsRenderTarget)
#define IVMRImageCompositor_SetStreamMediaType(This,dwStrmID,pmt,fTexture) (This)->lpVtbl->SetStreamMediaType(This,dwStrmID,pmt,fTexture)
#define IVMRImageCompositor_CompositeImage(This,pD3DDevice,pddsRenderTarget,pmtRenderTarget,rtStart,rtEnd,dwClrBkGnd,pVideoStreamInfo,cStreams) (This)->lpVtbl->CompositeImage(This,pD3DDevice,pddsRenderTarget,pmtRenderTarget,rtStart,rtEnd,dwClrBkGnd,pVideoStreamInfo,cStreams)
#endif
#endif
  HRESULT WINAPI IVMRImageCompositor_InitCompositionTarget_Proxy(IVMRImageCompositor *This,IUnknown *pD3DDevice,LPDIRECTDRAWSURFACE7 pddsRenderTarget);
  void __RPC_STUB IVMRImageCompositor_InitCompositionTarget_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRImageCompositor_TermCompositionTarget_Proxy(IVMRImageCompositor *This,IUnknown *pD3DDevice,LPDIRECTDRAWSURFACE7 pddsRenderTarget);
  void __RPC_STUB IVMRImageCompositor_TermCompositionTarget_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRImageCompositor_SetStreamMediaType_Proxy(IVMRImageCompositor *This,DWORD dwStrmID,AM_MEDIA_TYPE *pmt,WINBOOL fTexture);
  void __RPC_STUB IVMRImageCompositor_SetStreamMediaType_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRImageCompositor_CompositeImage_Proxy(IVMRImageCompositor *This,IUnknown *pD3DDevice,LPDIRECTDRAWSURFACE7 pddsRenderTarget,AM_MEDIA_TYPE *pmtRenderTarget,REFERENCE_TIME rtStart,REFERENCE_TIME rtEnd,DWORD dwClrBkGnd,VMRVIDEOSTREAMINFO *pVideoStreamInfo,UINT cStreams);
  void __RPC_STUB IVMRImageCompositor_CompositeImage_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IVMRVideoStreamControl_INTERFACE_DEFINED__
#define __IVMRVideoStreamControl_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IVMRVideoStreamControl;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IVMRVideoStreamControl : public IUnknown {
  public:
    virtual HRESULT WINAPI SetColorKey(LPDDCOLORKEY lpClrKey) = 0;
    virtual HRESULT WINAPI GetColorKey(LPDDCOLORKEY lpClrKey) = 0;
    virtual HRESULT WINAPI SetStreamActiveState(WINBOOL fActive) = 0;
    virtual HRESULT WINAPI GetStreamActiveState(WINBOOL *lpfActive) = 0;
  };
#else
  typedef struct IVMRVideoStreamControlVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IVMRVideoStreamControl *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IVMRVideoStreamControl *This);
      ULONG (WINAPI *Release)(IVMRVideoStreamControl *This);
      HRESULT (WINAPI *SetColorKey)(IVMRVideoStreamControl *This,LPDDCOLORKEY lpClrKey);
      HRESULT (WINAPI *GetColorKey)(IVMRVideoStreamControl *This,LPDDCOLORKEY lpClrKey);
      HRESULT (WINAPI *SetStreamActiveState)(IVMRVideoStreamControl *This,WINBOOL fActive);
      HRESULT (WINAPI *GetStreamActiveState)(IVMRVideoStreamControl *This,WINBOOL *lpfActive);
    END_INTERFACE
  } IVMRVideoStreamControlVtbl;
  struct IVMRVideoStreamControl {
    CONST_VTBL struct IVMRVideoStreamControlVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IVMRVideoStreamControl_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IVMRVideoStreamControl_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IVMRVideoStreamControl_Release(This) (This)->lpVtbl->Release(This)
#define IVMRVideoStreamControl_SetColorKey(This,lpClrKey) (This)->lpVtbl->SetColorKey(This,lpClrKey)
#define IVMRVideoStreamControl_GetColorKey(This,lpClrKey) (This)->lpVtbl->GetColorKey(This,lpClrKey)
#define IVMRVideoStreamControl_SetStreamActiveState(This,fActive) (This)->lpVtbl->SetStreamActiveState(This,fActive)
#define IVMRVideoStreamControl_GetStreamActiveState(This,lpfActive) (This)->lpVtbl->GetStreamActiveState(This,lpfActive)
#endif
#endif
  HRESULT WINAPI IVMRVideoStreamControl_SetColorKey_Proxy(IVMRVideoStreamControl *This,LPDDCOLORKEY lpClrKey);
  void __RPC_STUB IVMRVideoStreamControl_SetColorKey_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRVideoStreamControl_GetColorKey_Proxy(IVMRVideoStreamControl *This,LPDDCOLORKEY lpClrKey);
  void __RPC_STUB IVMRVideoStreamControl_GetColorKey_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRVideoStreamControl_SetStreamActiveState_Proxy(IVMRVideoStreamControl *This,WINBOOL fActive);
  void __RPC_STUB IVMRVideoStreamControl_SetStreamActiveState_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRVideoStreamControl_GetStreamActiveState_Proxy(IVMRVideoStreamControl *This,WINBOOL *lpfActive);
  void __RPC_STUB IVMRVideoStreamControl_GetStreamActiveState_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IVMRSurface_INTERFACE_DEFINED__
#define __IVMRSurface_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IVMRSurface;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IVMRSurface : public IUnknown {
  public:
    virtual HRESULT WINAPI IsSurfaceLocked(void) = 0;
    virtual HRESULT WINAPI LockSurface(BYTE **lpSurface) = 0;
    virtual HRESULT WINAPI UnlockSurface(void) = 0;
    virtual HRESULT WINAPI GetSurface(LPDIRECTDRAWSURFACE7 *lplpSurface) = 0;
  };
#else
  typedef struct IVMRSurfaceVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IVMRSurface *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IVMRSurface *This);
      ULONG (WINAPI *Release)(IVMRSurface *This);
      HRESULT (WINAPI *IsSurfaceLocked)(IVMRSurface *This);
      HRESULT (WINAPI *LockSurface)(IVMRSurface *This,BYTE **lpSurface);
      HRESULT (WINAPI *UnlockSurface)(IVMRSurface *This);
      HRESULT (WINAPI *GetSurface)(IVMRSurface *This,LPDIRECTDRAWSURFACE7 *lplpSurface);
    END_INTERFACE
  } IVMRSurfaceVtbl;
  struct IVMRSurface {
    CONST_VTBL struct IVMRSurfaceVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IVMRSurface_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IVMRSurface_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IVMRSurface_Release(This) (This)->lpVtbl->Release(This)
#define IVMRSurface_IsSurfaceLocked(This) (This)->lpVtbl->IsSurfaceLocked(This)
#define IVMRSurface_LockSurface(This,lpSurface) (This)->lpVtbl->LockSurface(This,lpSurface)
#define IVMRSurface_UnlockSurface(This) (This)->lpVtbl->UnlockSurface(This)
#define IVMRSurface_GetSurface(This,lplpSurface) (This)->lpVtbl->GetSurface(This,lplpSurface)
#endif
#endif
  HRESULT WINAPI IVMRSurface_IsSurfaceLocked_Proxy(IVMRSurface *This);
  void __RPC_STUB IVMRSurface_IsSurfaceLocked_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRSurface_LockSurface_Proxy(IVMRSurface *This,BYTE **lpSurface);
  void __RPC_STUB IVMRSurface_LockSurface_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRSurface_UnlockSurface_Proxy(IVMRSurface *This);
  void __RPC_STUB IVMRSurface_UnlockSurface_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRSurface_GetSurface_Proxy(IVMRSurface *This,LPDIRECTDRAWSURFACE7 *lplpSurface);
  void __RPC_STUB IVMRSurface_GetSurface_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IVMRImagePresenterConfig_INTERFACE_DEFINED__
#define __IVMRImagePresenterConfig_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IVMRImagePresenterConfig;

#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IVMRImagePresenterConfig : public IUnknown {
  public:
    virtual HRESULT WINAPI SetRenderingPrefs(DWORD dwRenderFlags) = 0;
    virtual HRESULT WINAPI GetRenderingPrefs(DWORD *dwRenderFlags) = 0;
  };
#else
  typedef struct IVMRImagePresenterConfigVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IVMRImagePresenterConfig *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IVMRImagePresenterConfig *This);
      ULONG (WINAPI *Release)(IVMRImagePresenterConfig *This);
      HRESULT (WINAPI *SetRenderingPrefs)(IVMRImagePresenterConfig *This,DWORD dwRenderFlags);
      HRESULT (WINAPI *GetRenderingPrefs)(IVMRImagePresenterConfig *This,DWORD *dwRenderFlags);
    END_INTERFACE
  } IVMRImagePresenterConfigVtbl;
  struct IVMRImagePresenterConfig {
    CONST_VTBL struct IVMRImagePresenterConfigVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IVMRImagePresenterConfig_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IVMRImagePresenterConfig_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IVMRImagePresenterConfig_Release(This) (This)->lpVtbl->Release(This)
#define IVMRImagePresenterConfig_SetRenderingPrefs(This,dwRenderFlags) (This)->lpVtbl->SetRenderingPrefs(This,dwRenderFlags)
#define IVMRImagePresenterConfig_GetRenderingPrefs(This,dwRenderFlags) (This)->lpVtbl->GetRenderingPrefs(This,dwRenderFlags)
#endif
#endif
  HRESULT WINAPI IVMRImagePresenterConfig_SetRenderingPrefs_Proxy(IVMRImagePresenterConfig *This,DWORD dwRenderFlags);
  void __RPC_STUB IVMRImagePresenterConfig_SetRenderingPrefs_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRImagePresenterConfig_GetRenderingPrefs_Proxy(IVMRImagePresenterConfig *This,DWORD *dwRenderFlags);
  void __RPC_STUB IVMRImagePresenterConfig_GetRenderingPrefs_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IVMRImagePresenterExclModeConfig_INTERFACE_DEFINED__
#define __IVMRImagePresenterExclModeConfig_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IVMRImagePresenterExclModeConfig;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IVMRImagePresenterExclModeConfig : public IVMRImagePresenterConfig {
  public:
    virtual HRESULT WINAPI SetXlcModeDDObjAndPrimarySurface(LPDIRECTDRAW7 lpDDObj,LPDIRECTDRAWSURFACE7 lpPrimarySurf) = 0;
    virtual HRESULT WINAPI GetXlcModeDDObjAndPrimarySurface(LPDIRECTDRAW7 *lpDDObj,LPDIRECTDRAWSURFACE7 *lpPrimarySurf) = 0;
  };
#else
  typedef struct IVMRImagePresenterExclModeConfigVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IVMRImagePresenterExclModeConfig *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IVMRImagePresenterExclModeConfig *This);
      ULONG (WINAPI *Release)(IVMRImagePresenterExclModeConfig *This);
      HRESULT (WINAPI *SetRenderingPrefs)(IVMRImagePresenterExclModeConfig *This,DWORD dwRenderFlags);
      HRESULT (WINAPI *GetRenderingPrefs)(IVMRImagePresenterExclModeConfig *This,DWORD *dwRenderFlags);
      HRESULT (WINAPI *SetXlcModeDDObjAndPrimarySurface)(IVMRImagePresenterExclModeConfig *This,LPDIRECTDRAW7 lpDDObj,LPDIRECTDRAWSURFACE7 lpPrimarySurf);
      HRESULT (WINAPI *GetXlcModeDDObjAndPrimarySurface)(IVMRImagePresenterExclModeConfig *This,LPDIRECTDRAW7 *lpDDObj,LPDIRECTDRAWSURFACE7 *lpPrimarySurf);
    END_INTERFACE
  } IVMRImagePresenterExclModeConfigVtbl;
  struct IVMRImagePresenterExclModeConfig {
    CONST_VTBL struct IVMRImagePresenterExclModeConfigVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IVMRImagePresenterExclModeConfig_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IVMRImagePresenterExclModeConfig_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IVMRImagePresenterExclModeConfig_Release(This) (This)->lpVtbl->Release(This)
#define IVMRImagePresenterExclModeConfig_SetRenderingPrefs(This,dwRenderFlags) (This)->lpVtbl->SetRenderingPrefs(This,dwRenderFlags)
#define IVMRImagePresenterExclModeConfig_GetRenderingPrefs(This,dwRenderFlags) (This)->lpVtbl->GetRenderingPrefs(This,dwRenderFlags)
#define IVMRImagePresenterExclModeConfig_SetXlcModeDDObjAndPrimarySurface(This,lpDDObj,lpPrimarySurf) (This)->lpVtbl->SetXlcModeDDObjAndPrimarySurface(This,lpDDObj,lpPrimarySurf)
#define IVMRImagePresenterExclModeConfig_GetXlcModeDDObjAndPrimarySurface(This,lpDDObj,lpPrimarySurf) (This)->lpVtbl->GetXlcModeDDObjAndPrimarySurface(This,lpDDObj,lpPrimarySurf)
#endif
#endif
  HRESULT WINAPI IVMRImagePresenterExclModeConfig_SetXlcModeDDObjAndPrimarySurface_Proxy(IVMRImagePresenterExclModeConfig *This,LPDIRECTDRAW7 lpDDObj,LPDIRECTDRAWSURFACE7 lpPrimarySurf);
  void __RPC_STUB IVMRImagePresenterExclModeConfig_SetXlcModeDDObjAndPrimarySurface_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVMRImagePresenterExclModeConfig_GetXlcModeDDObjAndPrimarySurface_Proxy(IVMRImagePresenterExclModeConfig *This,LPDIRECTDRAW7 *lpDDObj,LPDIRECTDRAWSURFACE7 *lpPrimarySurf);
  void __RPC_STUB IVMRImagePresenterExclModeConfig_GetXlcModeDDObjAndPrimarySurface_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IVPManager_INTERFACE_DEFINED__
#define __IVPManager_INTERFACE_DEFINED__
  EXTERN_C const IID IID_IVPManager;
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IVPManager : public IUnknown {
  public:
    virtual HRESULT WINAPI SetVideoPortIndex(DWORD dwVideoPortIndex) = 0;
    virtual HRESULT WINAPI GetVideoPortIndex(DWORD *pdwVideoPortIndex) = 0;
  };
#else
  typedef struct IVPManagerVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IVPManager *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IVPManager *This);
      ULONG (WINAPI *Release)(IVPManager *This);
      HRESULT (WINAPI *SetVideoPortIndex)(IVPManager *This,DWORD dwVideoPortIndex);
      HRESULT (WINAPI *GetVideoPortIndex)(IVPManager *This,DWORD *pdwVideoPortIndex);
    END_INTERFACE
  } IVPManagerVtbl;
  struct IVPManager {
    CONST_VTBL struct IVPManagerVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IVPManager_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IVPManager_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IVPManager_Release(This) (This)->lpVtbl->Release(This)
#define IVPManager_SetVideoPortIndex(This,dwVideoPortIndex) (This)->lpVtbl->SetVideoPortIndex(This,dwVideoPortIndex)
#define IVPManager_GetVideoPortIndex(This,pdwVideoPortIndex) (This)->lpVtbl->GetVideoPortIndex(This,pdwVideoPortIndex)
#endif
#endif
  HRESULT WINAPI IVPManager_SetVideoPortIndex_Proxy(IVPManager *This,DWORD dwVideoPortIndex);
  void __RPC_STUB IVPManager_SetVideoPortIndex_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVPManager_GetVideoPortIndex_Proxy(IVPManager *This,DWORD *pdwVideoPortIndex);
  void __RPC_STUB IVPManager_GetVideoPortIndex_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  extern RPC_IF_HANDLE __MIDL_itf_strmif_0413_v0_0_c_ifspec;
  extern RPC_IF_HANDLE __MIDL_itf_strmif_0413_v0_0_s_ifspec;

  unsigned long __RPC_API VARIANT_UserSize(unsigned long *,unsigned long,VARIANT *);
  unsigned char *__RPC_API VARIANT_UserMarshal(unsigned long *,unsigned char *,VARIANT *);
  unsigned char *__RPC_API VARIANT_UserUnmarshal(unsigned long *,unsigned char *,VARIANT *);
  void __RPC_API VARIANT_UserFree(unsigned long *,VARIANT *);

  HRESULT WINAPI ICaptureGraphBuilder_FindInterface_Proxy(ICaptureGraphBuilder *This,const GUID *pCategory,IBaseFilter *pf,REFIID riid,void **ppint);
  HRESULT WINAPI ICaptureGraphBuilder_FindInterface_Stub(ICaptureGraphBuilder *This,const GUID *pCategory,IBaseFilter *pf,REFIID riid,IUnknown **ppint);
  HRESULT WINAPI ICaptureGraphBuilder2_FindInterface_Proxy(ICaptureGraphBuilder2 *This,const GUID *pCategory,const GUID *pType,IBaseFilter *pf,REFIID riid,void **ppint);
  HRESULT WINAPI ICaptureGraphBuilder2_FindInterface_Stub(ICaptureGraphBuilder2 *This,const GUID *pCategory,const GUID *pType,IBaseFilter *pf,REFIID riid,IUnknown **ppint);
  HRESULT WINAPI IKsPropertySet_Set_Proxy(IKsPropertySet *This,REFGUID guidPropSet,DWORD dwPropID,LPVOID pInstanceData,DWORD cbInstanceData,LPVOID pPropData,DWORD cbPropData);
  HRESULT WINAPI IKsPropertySet_Set_Stub(IKsPropertySet *This,REFGUID guidPropSet,DWORD dwPropID,byte *pInstanceData,DWORD cbInstanceData,byte *pPropData,DWORD cbPropData);
  HRESULT WINAPI IKsPropertySet_Get_Proxy(IKsPropertySet *This,REFGUID guidPropSet,DWORD dwPropID,LPVOID pInstanceData,DWORD cbInstanceData,LPVOID pPropData,DWORD cbPropData,DWORD *pcbReturned);
  HRESULT WINAPI IKsPropertySet_Get_Stub(IKsPropertySet *This,REFGUID guidPropSet,DWORD dwPropID,byte *pInstanceData,DWORD cbInstanceData,byte *pPropData,DWORD cbPropData,DWORD *pcbReturned);

#ifdef __cplusplus
}
#endif
#endif
