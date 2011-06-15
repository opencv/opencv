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
#error this stub requires an updated version of <rpcndr.h>
#endif

#ifndef __control_h__
#define __control_h__

#ifndef __IAMCollection_FWD_DEFINED__
#define __IAMCollection_FWD_DEFINED__
typedef struct IAMCollection IAMCollection;
#endif

#ifndef __IMediaControl_FWD_DEFINED__
#define __IMediaControl_FWD_DEFINED__
typedef struct IMediaControl IMediaControl;
#endif

#ifndef __IMediaEvent_FWD_DEFINED__
#define __IMediaEvent_FWD_DEFINED__
typedef struct IMediaEvent IMediaEvent;
#endif

#ifndef __IMediaEventEx_FWD_DEFINED__
#define __IMediaEventEx_FWD_DEFINED__
typedef struct IMediaEventEx IMediaEventEx;
#endif

#ifndef __IMediaPosition_FWD_DEFINED__
#define __IMediaPosition_FWD_DEFINED__
typedef struct IMediaPosition IMediaPosition;
#endif

#ifndef __IBasicAudio_FWD_DEFINED__
#define __IBasicAudio_FWD_DEFINED__
typedef struct IBasicAudio IBasicAudio;
#endif

#ifndef __IVideoWindow_FWD_DEFINED__
#define __IVideoWindow_FWD_DEFINED__
typedef struct IVideoWindow IVideoWindow;
#endif

#ifndef __IBasicVideo_FWD_DEFINED__
#define __IBasicVideo_FWD_DEFINED__
typedef struct IBasicVideo IBasicVideo;
#endif

#ifndef __IBasicVideo2_FWD_DEFINED__
#define __IBasicVideo2_FWD_DEFINED__
typedef struct IBasicVideo2 IBasicVideo2;
#endif

#ifndef __IDeferredCommand_FWD_DEFINED__
#define __IDeferredCommand_FWD_DEFINED__
typedef struct IDeferredCommand IDeferredCommand;
#endif

#ifndef __IQueueCommand_FWD_DEFINED__
#define __IQueueCommand_FWD_DEFINED__
typedef struct IQueueCommand IQueueCommand;
#endif

#ifndef __FilgraphManager_FWD_DEFINED__
#define __FilgraphManager_FWD_DEFINED__

#ifdef __cplusplus
typedef class FilgraphManager FilgraphManager;
#else
typedef struct FilgraphManager FilgraphManager;
#endif
#endif

#ifndef __IFilterInfo_FWD_DEFINED__
#define __IFilterInfo_FWD_DEFINED__
typedef struct IFilterInfo IFilterInfo;
#endif

#ifndef __IRegFilterInfo_FWD_DEFINED__
#define __IRegFilterInfo_FWD_DEFINED__
typedef struct IRegFilterInfo IRegFilterInfo;
#endif

#ifndef __IMediaTypeInfo_FWD_DEFINED__
#define __IMediaTypeInfo_FWD_DEFINED__
typedef struct IMediaTypeInfo IMediaTypeInfo;
#endif

#ifndef __IPinInfo_FWD_DEFINED__
#define __IPinInfo_FWD_DEFINED__
typedef struct IPinInfo IPinInfo;
#endif

#ifndef __IAMStats_FWD_DEFINED__
#define __IAMStats_FWD_DEFINED__
typedef struct IAMStats IAMStats;
#endif

#ifdef __cplusplus
extern "C"{
#endif

#ifndef __MIDL_user_allocate_free_DEFINED__
#define __MIDL_user_allocate_free_DEFINED__
  void *__RPC_API MIDL_user_allocate(size_t);
  void __RPC_API MIDL_user_free(void *);
#endif

#ifndef __QuartzTypeLib_LIBRARY_DEFINED__
#define __QuartzTypeLib_LIBRARY_DEFINED__
  typedef double REFTIME;
  typedef LONG_PTR OAEVENT;
  typedef LONG_PTR OAHWND;
  typedef long OAFilterState;

  DEFINE_GUID(LIBID_QuartzTypeLib,0x56a868b0,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
#ifndef __IAMCollection_INTERFACE_DEFINED__
#define __IAMCollection_INTERFACE_DEFINED__
  DEFINE_GUID(IID_IAMCollection,0x56a868b9,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMCollection : public IDispatch {
  public:
    virtual HRESULT WINAPI get_Count(LONG *plCount) = 0;
    virtual HRESULT WINAPI Item(long lItem,IUnknown **ppUnk) = 0;
    virtual HRESULT WINAPI get__NewEnum(IUnknown **ppUnk) = 0;
  };
#else
  typedef struct IAMCollectionVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMCollection *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMCollection *This);
      ULONG (WINAPI *Release)(IAMCollection *This);
      HRESULT (WINAPI *GetTypeInfoCount)(IAMCollection *This,UINT *pctinfo);
      HRESULT (WINAPI *GetTypeInfo)(IAMCollection *This,UINT iTInfo,LCID lcid,ITypeInfo **ppTInfo);
      HRESULT (WINAPI *GetIDsOfNames)(IAMCollection *This,REFIID riid,LPOLESTR *rgszNames,UINT cNames,LCID lcid,DISPID *rgDispId);
      HRESULT (WINAPI *Invoke)(IAMCollection *This,DISPID dispIdMember,REFIID riid,LCID lcid,WORD wFlags,DISPPARAMS *pDispParams,VARIANT *pVarResult,EXCEPINFO *pExcepInfo,UINT *puArgErr);
      HRESULT (WINAPI *get_Count)(IAMCollection *This,LONG *plCount);
      HRESULT (WINAPI *Item)(IAMCollection *This,long lItem,IUnknown **ppUnk);
      HRESULT (WINAPI *get__NewEnum)(IAMCollection *This,IUnknown **ppUnk);
    END_INTERFACE
  } IAMCollectionVtbl;
  struct IAMCollection {
    CONST_VTBL struct IAMCollectionVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMCollection_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMCollection_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMCollection_Release(This) (This)->lpVtbl->Release(This)
#define IAMCollection_GetTypeInfoCount(This,pctinfo) (This)->lpVtbl->GetTypeInfoCount(This,pctinfo)
#define IAMCollection_GetTypeInfo(This,iTInfo,lcid,ppTInfo) (This)->lpVtbl->GetTypeInfo(This,iTInfo,lcid,ppTInfo)
#define IAMCollection_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) (This)->lpVtbl->GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)
#define IAMCollection_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) (This)->lpVtbl->Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)
#define IAMCollection_get_Count(This,plCount) (This)->lpVtbl->get_Count(This,plCount)
#define IAMCollection_Item(This,lItem,ppUnk) (This)->lpVtbl->Item(This,lItem,ppUnk)
#define IAMCollection_get__NewEnum(This,ppUnk) (This)->lpVtbl->get__NewEnum(This,ppUnk)
#endif
#endif
  HRESULT WINAPI IAMCollection_get_Count_Proxy(IAMCollection *This,LONG *plCount);
  void __RPC_STUB IAMCollection_get_Count_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMCollection_Item_Proxy(IAMCollection *This,long lItem,IUnknown **ppUnk);
  void __RPC_STUB IAMCollection_Item_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMCollection_get__NewEnum_Proxy(IAMCollection *This,IUnknown **ppUnk);
  void __RPC_STUB IAMCollection_get__NewEnum_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IMediaControl_INTERFACE_DEFINED__
#define __IMediaControl_INTERFACE_DEFINED__
  DEFINE_GUID(IID_IMediaControl,0x56a868b1,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IMediaControl : public IDispatch {
  public:
    virtual HRESULT WINAPI Run(void) = 0;
    virtual HRESULT WINAPI Pause(void) = 0;
    virtual HRESULT WINAPI Stop(void) = 0;
    virtual HRESULT WINAPI GetState(LONG msTimeout,OAFilterState *pfs) = 0;
    virtual HRESULT WINAPI RenderFile(BSTR strFilename) = 0;
    virtual HRESULT WINAPI AddSourceFilter(BSTR strFilename,IDispatch **ppUnk) = 0;
    virtual HRESULT WINAPI get_FilterCollection(IDispatch **ppUnk) = 0;
    virtual HRESULT WINAPI get_RegFilterCollection(IDispatch **ppUnk) = 0;
    virtual HRESULT WINAPI StopWhenReady(void) = 0;
  };
#else
  typedef struct IMediaControlVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IMediaControl *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IMediaControl *This);
      ULONG (WINAPI *Release)(IMediaControl *This);
      HRESULT (WINAPI *GetTypeInfoCount)(IMediaControl *This,UINT *pctinfo);
      HRESULT (WINAPI *GetTypeInfo)(IMediaControl *This,UINT iTInfo,LCID lcid,ITypeInfo **ppTInfo);
      HRESULT (WINAPI *GetIDsOfNames)(IMediaControl *This,REFIID riid,LPOLESTR *rgszNames,UINT cNames,LCID lcid,DISPID *rgDispId);
      HRESULT (WINAPI *Invoke)(IMediaControl *This,DISPID dispIdMember,REFIID riid,LCID lcid,WORD wFlags,DISPPARAMS *pDispParams,VARIANT *pVarResult,EXCEPINFO *pExcepInfo,UINT *puArgErr);
      HRESULT (WINAPI *Run)(IMediaControl *This);
      HRESULT (WINAPI *Pause)(IMediaControl *This);
      HRESULT (WINAPI *Stop)(IMediaControl *This);
      HRESULT (WINAPI *GetState)(IMediaControl *This,LONG msTimeout,OAFilterState *pfs);
      HRESULT (WINAPI *RenderFile)(IMediaControl *This,BSTR strFilename);
      HRESULT (WINAPI *AddSourceFilter)(IMediaControl *This,BSTR strFilename,IDispatch **ppUnk);
      HRESULT (WINAPI *get_FilterCollection)(IMediaControl *This,IDispatch **ppUnk);
      HRESULT (WINAPI *get_RegFilterCollection)(IMediaControl *This,IDispatch **ppUnk);
      HRESULT (WINAPI *StopWhenReady)(IMediaControl *This);
    END_INTERFACE
  } IMediaControlVtbl;
  struct IMediaControl {
    CONST_VTBL struct IMediaControlVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IMediaControl_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMediaControl_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMediaControl_Release(This) (This)->lpVtbl->Release(This)
#define IMediaControl_GetTypeInfoCount(This,pctinfo) (This)->lpVtbl->GetTypeInfoCount(This,pctinfo)
#define IMediaControl_GetTypeInfo(This,iTInfo,lcid,ppTInfo) (This)->lpVtbl->GetTypeInfo(This,iTInfo,lcid,ppTInfo)
#define IMediaControl_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) (This)->lpVtbl->GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)
#define IMediaControl_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) (This)->lpVtbl->Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)
#define IMediaControl_Run(This) (This)->lpVtbl->Run(This)
#define IMediaControl_Pause(This) (This)->lpVtbl->Pause(This)
#define IMediaControl_Stop(This) (This)->lpVtbl->Stop(This)
#define IMediaControl_GetState(This,msTimeout,pfs) (This)->lpVtbl->GetState(This,msTimeout,pfs)
#define IMediaControl_RenderFile(This,strFilename) (This)->lpVtbl->RenderFile(This,strFilename)
#define IMediaControl_AddSourceFilter(This,strFilename,ppUnk) (This)->lpVtbl->AddSourceFilter(This,strFilename,ppUnk)
#define IMediaControl_get_FilterCollection(This,ppUnk) (This)->lpVtbl->get_FilterCollection(This,ppUnk)
#define IMediaControl_get_RegFilterCollection(This,ppUnk) (This)->lpVtbl->get_RegFilterCollection(This,ppUnk)
#define IMediaControl_StopWhenReady(This) (This)->lpVtbl->StopWhenReady(This)
#endif
#endif
  HRESULT WINAPI IMediaControl_Run_Proxy(IMediaControl *This);
  void __RPC_STUB IMediaControl_Run_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaControl_Pause_Proxy(IMediaControl *This);
  void __RPC_STUB IMediaControl_Pause_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaControl_Stop_Proxy(IMediaControl *This);
  void __RPC_STUB IMediaControl_Stop_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaControl_GetState_Proxy(IMediaControl *This,LONG msTimeout,OAFilterState *pfs);
  void __RPC_STUB IMediaControl_GetState_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaControl_RenderFile_Proxy(IMediaControl *This,BSTR strFilename);
  void __RPC_STUB IMediaControl_RenderFile_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaControl_AddSourceFilter_Proxy(IMediaControl *This,BSTR strFilename,IDispatch **ppUnk);
  void __RPC_STUB IMediaControl_AddSourceFilter_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaControl_get_FilterCollection_Proxy(IMediaControl *This,IDispatch **ppUnk);
  void __RPC_STUB IMediaControl_get_FilterCollection_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaControl_get_RegFilterCollection_Proxy(IMediaControl *This,IDispatch **ppUnk);
  void __RPC_STUB IMediaControl_get_RegFilterCollection_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaControl_StopWhenReady_Proxy(IMediaControl *This);
  void __RPC_STUB IMediaControl_StopWhenReady_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IMediaEvent_INTERFACE_DEFINED__
#define __IMediaEvent_INTERFACE_DEFINED__
  DEFINE_GUID(IID_IMediaEvent,0x56a868b6,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IMediaEvent : public IDispatch {
  public:
    virtual HRESULT WINAPI GetEventHandle(OAEVENT *hEvent) = 0;
    virtual HRESULT WINAPI GetEvent(long *lEventCode,LONG_PTR *lParam1,LONG_PTR *lParam2,long msTimeout) = 0;
    virtual HRESULT WINAPI WaitForCompletion(long msTimeout,long *pEvCode) = 0;
    virtual HRESULT WINAPI CancelDefaultHandling(long lEvCode) = 0;
    virtual HRESULT WINAPI RestoreDefaultHandling(long lEvCode) = 0;
    virtual HRESULT WINAPI FreeEventParams(long lEvCode,LONG_PTR lParam1,LONG_PTR lParam2) = 0;
  };
#else
  typedef struct IMediaEventVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IMediaEvent *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IMediaEvent *This);
      ULONG (WINAPI *Release)(IMediaEvent *This);
      HRESULT (WINAPI *GetTypeInfoCount)(IMediaEvent *This,UINT *pctinfo);
      HRESULT (WINAPI *GetTypeInfo)(IMediaEvent *This,UINT iTInfo,LCID lcid,ITypeInfo **ppTInfo);
      HRESULT (WINAPI *GetIDsOfNames)(IMediaEvent *This,REFIID riid,LPOLESTR *rgszNames,UINT cNames,LCID lcid,DISPID *rgDispId);
      HRESULT (WINAPI *Invoke)(IMediaEvent *This,DISPID dispIdMember,REFIID riid,LCID lcid,WORD wFlags,DISPPARAMS *pDispParams,VARIANT *pVarResult,EXCEPINFO *pExcepInfo,UINT *puArgErr);
      HRESULT (WINAPI *GetEventHandle)(IMediaEvent *This,OAEVENT *hEvent);
      HRESULT (WINAPI *GetEvent)(IMediaEvent *This,long *lEventCode,LONG_PTR *lParam1,LONG_PTR *lParam2,long msTimeout);
      HRESULT (WINAPI *WaitForCompletion)(IMediaEvent *This,long msTimeout,long *pEvCode);
      HRESULT (WINAPI *CancelDefaultHandling)(IMediaEvent *This,long lEvCode);
      HRESULT (WINAPI *RestoreDefaultHandling)(IMediaEvent *This,long lEvCode);
      HRESULT (WINAPI *FreeEventParams)(IMediaEvent *This,long lEvCode,LONG_PTR lParam1,LONG_PTR lParam2);
    END_INTERFACE
  } IMediaEventVtbl;
  struct IMediaEvent {
    CONST_VTBL struct IMediaEventVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IMediaEvent_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMediaEvent_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMediaEvent_Release(This) (This)->lpVtbl->Release(This)
#define IMediaEvent_GetTypeInfoCount(This,pctinfo) (This)->lpVtbl->GetTypeInfoCount(This,pctinfo)
#define IMediaEvent_GetTypeInfo(This,iTInfo,lcid,ppTInfo) (This)->lpVtbl->GetTypeInfo(This,iTInfo,lcid,ppTInfo)
#define IMediaEvent_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) (This)->lpVtbl->GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)
#define IMediaEvent_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) (This)->lpVtbl->Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)
#define IMediaEvent_GetEventHandle(This,hEvent) (This)->lpVtbl->GetEventHandle(This,hEvent)
#define IMediaEvent_GetEvent(This,lEventCode,lParam1,lParam2,msTimeout) (This)->lpVtbl->GetEvent(This,lEventCode,lParam1,lParam2,msTimeout)
#define IMediaEvent_WaitForCompletion(This,msTimeout,pEvCode) (This)->lpVtbl->WaitForCompletion(This,msTimeout,pEvCode)
#define IMediaEvent_CancelDefaultHandling(This,lEvCode) (This)->lpVtbl->CancelDefaultHandling(This,lEvCode)
#define IMediaEvent_RestoreDefaultHandling(This,lEvCode) (This)->lpVtbl->RestoreDefaultHandling(This,lEvCode)
#define IMediaEvent_FreeEventParams(This,lEvCode,lParam1,lParam2) (This)->lpVtbl->FreeEventParams(This,lEvCode,lParam1,lParam2)
#endif
#endif
  HRESULT WINAPI IMediaEvent_GetEventHandle_Proxy(IMediaEvent *This,OAEVENT *hEvent);
  void __RPC_STUB IMediaEvent_GetEventHandle_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaEvent_GetEvent_Proxy(IMediaEvent *This,long *lEventCode,LONG_PTR *lParam1,LONG_PTR *lParam2,long msTimeout);
  void __RPC_STUB IMediaEvent_GetEvent_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaEvent_WaitForCompletion_Proxy(IMediaEvent *This,long msTimeout,long *pEvCode);
  void __RPC_STUB IMediaEvent_WaitForCompletion_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaEvent_CancelDefaultHandling_Proxy(IMediaEvent *This,long lEvCode);
  void __RPC_STUB IMediaEvent_CancelDefaultHandling_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaEvent_RestoreDefaultHandling_Proxy(IMediaEvent *This,long lEvCode);
  void __RPC_STUB IMediaEvent_RestoreDefaultHandling_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaEvent_FreeEventParams_Proxy(IMediaEvent *This,long lEvCode,LONG_PTR lParam1,LONG_PTR lParam2);
  void __RPC_STUB IMediaEvent_FreeEventParams_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IMediaEventEx_INTERFACE_DEFINED__
#define __IMediaEventEx_INTERFACE_DEFINED__
  DEFINE_GUID(IID_IMediaEventEx,0x56a868c0,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IMediaEventEx : public IMediaEvent {
  public:
    virtual HRESULT WINAPI SetNotifyWindow(OAHWND hwnd,long lMsg,LONG_PTR lInstanceData) = 0;
    virtual HRESULT WINAPI SetNotifyFlags(long lNoNotifyFlags) = 0;
    virtual HRESULT WINAPI GetNotifyFlags(long *lplNoNotifyFlags) = 0;
  };
#else
  typedef struct IMediaEventExVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IMediaEventEx *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IMediaEventEx *This);
      ULONG (WINAPI *Release)(IMediaEventEx *This);
      HRESULT (WINAPI *GetTypeInfoCount)(IMediaEventEx *This,UINT *pctinfo);
      HRESULT (WINAPI *GetTypeInfo)(IMediaEventEx *This,UINT iTInfo,LCID lcid,ITypeInfo **ppTInfo);
      HRESULT (WINAPI *GetIDsOfNames)(IMediaEventEx *This,REFIID riid,LPOLESTR *rgszNames,UINT cNames,LCID lcid,DISPID *rgDispId);
      HRESULT (WINAPI *Invoke)(IMediaEventEx *This,DISPID dispIdMember,REFIID riid,LCID lcid,WORD wFlags,DISPPARAMS *pDispParams,VARIANT *pVarResult,EXCEPINFO *pExcepInfo,UINT *puArgErr);
      HRESULT (WINAPI *GetEventHandle)(IMediaEventEx *This,OAEVENT *hEvent);
      HRESULT (WINAPI *GetEvent)(IMediaEventEx *This,long *lEventCode,LONG_PTR *lParam1,LONG_PTR *lParam2,long msTimeout);
      HRESULT (WINAPI *WaitForCompletion)(IMediaEventEx *This,long msTimeout,long *pEvCode);
      HRESULT (WINAPI *CancelDefaultHandling)(IMediaEventEx *This,long lEvCode);
      HRESULT (WINAPI *RestoreDefaultHandling)(IMediaEventEx *This,long lEvCode);
      HRESULT (WINAPI *FreeEventParams)(IMediaEventEx *This,long lEvCode,LONG_PTR lParam1,LONG_PTR lParam2);
      HRESULT (WINAPI *SetNotifyWindow)(IMediaEventEx *This,OAHWND hwnd,long lMsg,LONG_PTR lInstanceData);
      HRESULT (WINAPI *SetNotifyFlags)(IMediaEventEx *This,long lNoNotifyFlags);
      HRESULT (WINAPI *GetNotifyFlags)(IMediaEventEx *This,long *lplNoNotifyFlags);
    END_INTERFACE
  } IMediaEventExVtbl;
  struct IMediaEventEx {
    CONST_VTBL struct IMediaEventExVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IMediaEventEx_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMediaEventEx_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMediaEventEx_Release(This) (This)->lpVtbl->Release(This)
#define IMediaEventEx_GetTypeInfoCount(This,pctinfo) (This)->lpVtbl->GetTypeInfoCount(This,pctinfo)
#define IMediaEventEx_GetTypeInfo(This,iTInfo,lcid,ppTInfo) (This)->lpVtbl->GetTypeInfo(This,iTInfo,lcid,ppTInfo)
#define IMediaEventEx_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) (This)->lpVtbl->GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)
#define IMediaEventEx_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) (This)->lpVtbl->Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)
#define IMediaEventEx_GetEventHandle(This,hEvent) (This)->lpVtbl->GetEventHandle(This,hEvent)
#define IMediaEventEx_GetEvent(This,lEventCode,lParam1,lParam2,msTimeout) (This)->lpVtbl->GetEvent(This,lEventCode,lParam1,lParam2,msTimeout)
#define IMediaEventEx_WaitForCompletion(This,msTimeout,pEvCode) (This)->lpVtbl->WaitForCompletion(This,msTimeout,pEvCode)
#define IMediaEventEx_CancelDefaultHandling(This,lEvCode) (This)->lpVtbl->CancelDefaultHandling(This,lEvCode)
#define IMediaEventEx_RestoreDefaultHandling(This,lEvCode) (This)->lpVtbl->RestoreDefaultHandling(This,lEvCode)
#define IMediaEventEx_FreeEventParams(This,lEvCode,lParam1,lParam2) (This)->lpVtbl->FreeEventParams(This,lEvCode,lParam1,lParam2)
#define IMediaEventEx_SetNotifyWindow(This,hwnd,lMsg,lInstanceData) (This)->lpVtbl->SetNotifyWindow(This,hwnd,lMsg,lInstanceData)
#define IMediaEventEx_SetNotifyFlags(This,lNoNotifyFlags) (This)->lpVtbl->SetNotifyFlags(This,lNoNotifyFlags)
#define IMediaEventEx_GetNotifyFlags(This,lplNoNotifyFlags) (This)->lpVtbl->GetNotifyFlags(This,lplNoNotifyFlags)
#endif
#endif
  HRESULT WINAPI IMediaEventEx_SetNotifyWindow_Proxy(IMediaEventEx *This,OAHWND hwnd,long lMsg,LONG_PTR lInstanceData);
  void __RPC_STUB IMediaEventEx_SetNotifyWindow_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaEventEx_SetNotifyFlags_Proxy(IMediaEventEx *This,long lNoNotifyFlags);
  void __RPC_STUB IMediaEventEx_SetNotifyFlags_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaEventEx_GetNotifyFlags_Proxy(IMediaEventEx *This,long *lplNoNotifyFlags);
  void __RPC_STUB IMediaEventEx_GetNotifyFlags_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IMediaPosition_INTERFACE_DEFINED__
#define __IMediaPosition_INTERFACE_DEFINED__
  DEFINE_GUID(IID_IMediaPosition,0x56a868b2,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IMediaPosition : public IDispatch {
  public:
    virtual HRESULT WINAPI get_Duration(REFTIME *plength) = 0;
    virtual HRESULT WINAPI put_CurrentPosition(REFTIME llTime) = 0;
    virtual HRESULT WINAPI get_CurrentPosition(REFTIME *pllTime) = 0;
    virtual HRESULT WINAPI get_StopTime(REFTIME *pllTime) = 0;
    virtual HRESULT WINAPI put_StopTime(REFTIME llTime) = 0;
    virtual HRESULT WINAPI get_PrerollTime(REFTIME *pllTime) = 0;
    virtual HRESULT WINAPI put_PrerollTime(REFTIME llTime) = 0;
    virtual HRESULT WINAPI put_Rate(double dRate) = 0;
    virtual HRESULT WINAPI get_Rate(double *pdRate) = 0;
    virtual HRESULT WINAPI CanSeekForward(LONG *pCanSeekForward) = 0;
    virtual HRESULT WINAPI CanSeekBackward(LONG *pCanSeekBackward) = 0;
  };
#else
  typedef struct IMediaPositionVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IMediaPosition *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IMediaPosition *This);
      ULONG (WINAPI *Release)(IMediaPosition *This);
      HRESULT (WINAPI *GetTypeInfoCount)(IMediaPosition *This,UINT *pctinfo);
      HRESULT (WINAPI *GetTypeInfo)(IMediaPosition *This,UINT iTInfo,LCID lcid,ITypeInfo **ppTInfo);
      HRESULT (WINAPI *GetIDsOfNames)(IMediaPosition *This,REFIID riid,LPOLESTR *rgszNames,UINT cNames,LCID lcid,DISPID *rgDispId);
      HRESULT (WINAPI *Invoke)(IMediaPosition *This,DISPID dispIdMember,REFIID riid,LCID lcid,WORD wFlags,DISPPARAMS *pDispParams,VARIANT *pVarResult,EXCEPINFO *pExcepInfo,UINT *puArgErr);
      HRESULT (WINAPI *get_Duration)(IMediaPosition *This,REFTIME *plength);
      HRESULT (WINAPI *put_CurrentPosition)(IMediaPosition *This,REFTIME llTime);
      HRESULT (WINAPI *get_CurrentPosition)(IMediaPosition *This,REFTIME *pllTime);
      HRESULT (WINAPI *get_StopTime)(IMediaPosition *This,REFTIME *pllTime);
      HRESULT (WINAPI *put_StopTime)(IMediaPosition *This,REFTIME llTime);
      HRESULT (WINAPI *get_PrerollTime)(IMediaPosition *This,REFTIME *pllTime);
      HRESULT (WINAPI *put_PrerollTime)(IMediaPosition *This,REFTIME llTime);
      HRESULT (WINAPI *put_Rate)(IMediaPosition *This,double dRate);
      HRESULT (WINAPI *get_Rate)(IMediaPosition *This,double *pdRate);
      HRESULT (WINAPI *CanSeekForward)(IMediaPosition *This,LONG *pCanSeekForward);
      HRESULT (WINAPI *CanSeekBackward)(IMediaPosition *This,LONG *pCanSeekBackward);
    END_INTERFACE
  } IMediaPositionVtbl;
  struct IMediaPosition {
    CONST_VTBL struct IMediaPositionVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IMediaPosition_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMediaPosition_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMediaPosition_Release(This) (This)->lpVtbl->Release(This)
#define IMediaPosition_GetTypeInfoCount(This,pctinfo) (This)->lpVtbl->GetTypeInfoCount(This,pctinfo)
#define IMediaPosition_GetTypeInfo(This,iTInfo,lcid,ppTInfo) (This)->lpVtbl->GetTypeInfo(This,iTInfo,lcid,ppTInfo)
#define IMediaPosition_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) (This)->lpVtbl->GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)
#define IMediaPosition_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) (This)->lpVtbl->Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)
#define IMediaPosition_get_Duration(This,plength) (This)->lpVtbl->get_Duration(This,plength)
#define IMediaPosition_put_CurrentPosition(This,llTime) (This)->lpVtbl->put_CurrentPosition(This,llTime)
#define IMediaPosition_get_CurrentPosition(This,pllTime) (This)->lpVtbl->get_CurrentPosition(This,pllTime)
#define IMediaPosition_get_StopTime(This,pllTime) (This)->lpVtbl->get_StopTime(This,pllTime)
#define IMediaPosition_put_StopTime(This,llTime) (This)->lpVtbl->put_StopTime(This,llTime)
#define IMediaPosition_get_PrerollTime(This,pllTime) (This)->lpVtbl->get_PrerollTime(This,pllTime)
#define IMediaPosition_put_PrerollTime(This,llTime) (This)->lpVtbl->put_PrerollTime(This,llTime)
#define IMediaPosition_put_Rate(This,dRate) (This)->lpVtbl->put_Rate(This,dRate)
#define IMediaPosition_get_Rate(This,pdRate) (This)->lpVtbl->get_Rate(This,pdRate)
#define IMediaPosition_CanSeekForward(This,pCanSeekForward) (This)->lpVtbl->CanSeekForward(This,pCanSeekForward)
#define IMediaPosition_CanSeekBackward(This,pCanSeekBackward) (This)->lpVtbl->CanSeekBackward(This,pCanSeekBackward)
#endif
#endif
  HRESULT WINAPI IMediaPosition_get_Duration_Proxy(IMediaPosition *This,REFTIME *plength);
  void __RPC_STUB IMediaPosition_get_Duration_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaPosition_put_CurrentPosition_Proxy(IMediaPosition *This,REFTIME llTime);
  void __RPC_STUB IMediaPosition_put_CurrentPosition_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaPosition_get_CurrentPosition_Proxy(IMediaPosition *This,REFTIME *pllTime);
  void __RPC_STUB IMediaPosition_get_CurrentPosition_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaPosition_get_StopTime_Proxy(IMediaPosition *This,REFTIME *pllTime);
  void __RPC_STUB IMediaPosition_get_StopTime_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaPosition_put_StopTime_Proxy(IMediaPosition *This,REFTIME llTime);
  void __RPC_STUB IMediaPosition_put_StopTime_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaPosition_get_PrerollTime_Proxy(IMediaPosition *This,REFTIME *pllTime);
  void __RPC_STUB IMediaPosition_get_PrerollTime_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaPosition_put_PrerollTime_Proxy(IMediaPosition *This,REFTIME llTime);
  void __RPC_STUB IMediaPosition_put_PrerollTime_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaPosition_put_Rate_Proxy(IMediaPosition *This,double dRate);
  void __RPC_STUB IMediaPosition_put_Rate_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaPosition_get_Rate_Proxy(IMediaPosition *This,double *pdRate);
  void __RPC_STUB IMediaPosition_get_Rate_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaPosition_CanSeekForward_Proxy(IMediaPosition *This,LONG *pCanSeekForward);
  void __RPC_STUB IMediaPosition_CanSeekForward_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaPosition_CanSeekBackward_Proxy(IMediaPosition *This,LONG *pCanSeekBackward);
  void __RPC_STUB IMediaPosition_CanSeekBackward_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IBasicAudio_INTERFACE_DEFINED__
#define __IBasicAudio_INTERFACE_DEFINED__
  DEFINE_GUID(IID_IBasicAudio,0x56a868b3,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IBasicAudio : public IDispatch {
  public:
    virtual HRESULT WINAPI put_Volume(long lVolume) = 0;
    virtual HRESULT WINAPI get_Volume(long *plVolume) = 0;
    virtual HRESULT WINAPI put_Balance(long lBalance) = 0;
    virtual HRESULT WINAPI get_Balance(long *plBalance) = 0;
  };
#else
  typedef struct IBasicAudioVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IBasicAudio *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IBasicAudio *This);
      ULONG (WINAPI *Release)(IBasicAudio *This);
      HRESULT (WINAPI *GetTypeInfoCount)(IBasicAudio *This,UINT *pctinfo);
      HRESULT (WINAPI *GetTypeInfo)(IBasicAudio *This,UINT iTInfo,LCID lcid,ITypeInfo **ppTInfo);
      HRESULT (WINAPI *GetIDsOfNames)(IBasicAudio *This,REFIID riid,LPOLESTR *rgszNames,UINT cNames,LCID lcid,DISPID *rgDispId);
      HRESULT (WINAPI *Invoke)(IBasicAudio *This,DISPID dispIdMember,REFIID riid,LCID lcid,WORD wFlags,DISPPARAMS *pDispParams,VARIANT *pVarResult,EXCEPINFO *pExcepInfo,UINT *puArgErr);
      HRESULT (WINAPI *put_Volume)(IBasicAudio *This,long lVolume);
      HRESULT (WINAPI *get_Volume)(IBasicAudio *This,long *plVolume);
      HRESULT (WINAPI *put_Balance)(IBasicAudio *This,long lBalance);
      HRESULT (WINAPI *get_Balance)(IBasicAudio *This,long *plBalance);
    END_INTERFACE
  } IBasicAudioVtbl;
  struct IBasicAudio {
    CONST_VTBL struct IBasicAudioVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IBasicAudio_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IBasicAudio_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IBasicAudio_Release(This) (This)->lpVtbl->Release(This)
#define IBasicAudio_GetTypeInfoCount(This,pctinfo) (This)->lpVtbl->GetTypeInfoCount(This,pctinfo)
#define IBasicAudio_GetTypeInfo(This,iTInfo,lcid,ppTInfo) (This)->lpVtbl->GetTypeInfo(This,iTInfo,lcid,ppTInfo)
#define IBasicAudio_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) (This)->lpVtbl->GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)
#define IBasicAudio_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) (This)->lpVtbl->Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)
#define IBasicAudio_put_Volume(This,lVolume) (This)->lpVtbl->put_Volume(This,lVolume)
#define IBasicAudio_get_Volume(This,plVolume) (This)->lpVtbl->get_Volume(This,plVolume)
#define IBasicAudio_put_Balance(This,lBalance) (This)->lpVtbl->put_Balance(This,lBalance)
#define IBasicAudio_get_Balance(This,plBalance) (This)->lpVtbl->get_Balance(This,plBalance)
#endif
#endif
  HRESULT WINAPI IBasicAudio_put_Volume_Proxy(IBasicAudio *This,long lVolume);
  void __RPC_STUB IBasicAudio_put_Volume_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicAudio_get_Volume_Proxy(IBasicAudio *This,long *plVolume);
  void __RPC_STUB IBasicAudio_get_Volume_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicAudio_put_Balance_Proxy(IBasicAudio *This,long lBalance);
  void __RPC_STUB IBasicAudio_put_Balance_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicAudio_get_Balance_Proxy(IBasicAudio *This,long *plBalance);
  void __RPC_STUB IBasicAudio_get_Balance_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IVideoWindow_INTERFACE_DEFINED__
#define __IVideoWindow_INTERFACE_DEFINED__
  DEFINE_GUID(IID_IVideoWindow,0x56a868b4,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IVideoWindow : public IDispatch {
  public:
    virtual HRESULT WINAPI put_Caption(BSTR strCaption) = 0;
    virtual HRESULT WINAPI get_Caption(BSTR *strCaption) = 0;
    virtual HRESULT WINAPI put_WindowStyle(long WindowStyle) = 0;
    virtual HRESULT WINAPI get_WindowStyle(long *WindowStyle) = 0;
    virtual HRESULT WINAPI put_WindowStyleEx(long WindowStyleEx) = 0;
    virtual HRESULT WINAPI get_WindowStyleEx(long *WindowStyleEx) = 0;
    virtual HRESULT WINAPI put_AutoShow(long AutoShow) = 0;
    virtual HRESULT WINAPI get_AutoShow(long *AutoShow) = 0;
    virtual HRESULT WINAPI put_WindowState(long WindowState) = 0;
    virtual HRESULT WINAPI get_WindowState(long *WindowState) = 0;
    virtual HRESULT WINAPI put_BackgroundPalette(long BackgroundPalette) = 0;
    virtual HRESULT WINAPI get_BackgroundPalette(long *pBackgroundPalette) = 0;
    virtual HRESULT WINAPI put_Visible(long Visible) = 0;
    virtual HRESULT WINAPI get_Visible(long *pVisible) = 0;
    virtual HRESULT WINAPI put_Left(long Left) = 0;
    virtual HRESULT WINAPI get_Left(long *pLeft) = 0;
    virtual HRESULT WINAPI put_Width(long Width) = 0;
    virtual HRESULT WINAPI get_Width(long *pWidth) = 0;
    virtual HRESULT WINAPI put_Top(long Top) = 0;
    virtual HRESULT WINAPI get_Top(long *pTop) = 0;
    virtual HRESULT WINAPI put_Height(long Height) = 0;
    virtual HRESULT WINAPI get_Height(long *pHeight) = 0;
    virtual HRESULT WINAPI put_Owner(OAHWND Owner) = 0;
    virtual HRESULT WINAPI get_Owner(OAHWND *Owner) = 0;
    virtual HRESULT WINAPI put_MessageDrain(OAHWND Drain) = 0;
    virtual HRESULT WINAPI get_MessageDrain(OAHWND *Drain) = 0;
    virtual HRESULT WINAPI get_BorderColor(long *Color) = 0;
    virtual HRESULT WINAPI put_BorderColor(long Color) = 0;
    virtual HRESULT WINAPI get_FullScreenMode(long *FullScreenMode) = 0;
    virtual HRESULT WINAPI put_FullScreenMode(long FullScreenMode) = 0;
    virtual HRESULT WINAPI SetWindowForeground(long Focus) = 0;
    virtual HRESULT WINAPI NotifyOwnerMessage(OAHWND hwnd,long uMsg,LONG_PTR wParam,LONG_PTR lParam) = 0;
    virtual HRESULT WINAPI SetWindowPosition(long Left,long Top,long Width,long Height) = 0;
    virtual HRESULT WINAPI GetWindowPosition(long *pLeft,long *pTop,long *pWidth,long *pHeight) = 0;
    virtual HRESULT WINAPI GetMinIdealImageSize(long *pWidth,long *pHeight) = 0;
    virtual HRESULT WINAPI GetMaxIdealImageSize(long *pWidth,long *pHeight) = 0;
    virtual HRESULT WINAPI GetRestorePosition(long *pLeft,long *pTop,long *pWidth,long *pHeight) = 0;
    virtual HRESULT WINAPI HideCursor(long HideCursor) = 0;
    virtual HRESULT WINAPI IsCursorHidden(long *CursorHidden) = 0;
  };
#else
  typedef struct IVideoWindowVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IVideoWindow *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IVideoWindow *This);
      ULONG (WINAPI *Release)(IVideoWindow *This);
      HRESULT (WINAPI *GetTypeInfoCount)(IVideoWindow *This,UINT *pctinfo);
      HRESULT (WINAPI *GetTypeInfo)(IVideoWindow *This,UINT iTInfo,LCID lcid,ITypeInfo **ppTInfo);
      HRESULT (WINAPI *GetIDsOfNames)(IVideoWindow *This,REFIID riid,LPOLESTR *rgszNames,UINT cNames,LCID lcid,DISPID *rgDispId);
      HRESULT (WINAPI *Invoke)(IVideoWindow *This,DISPID dispIdMember,REFIID riid,LCID lcid,WORD wFlags,DISPPARAMS *pDispParams,VARIANT *pVarResult,EXCEPINFO *pExcepInfo,UINT *puArgErr);
      HRESULT (WINAPI *put_Caption)(IVideoWindow *This,BSTR strCaption);
      HRESULT (WINAPI *get_Caption)(IVideoWindow *This,BSTR *strCaption);
      HRESULT (WINAPI *put_WindowStyle)(IVideoWindow *This,long WindowStyle);
      HRESULT (WINAPI *get_WindowStyle)(IVideoWindow *This,long *WindowStyle);
      HRESULT (WINAPI *put_WindowStyleEx)(IVideoWindow *This,long WindowStyleEx);
      HRESULT (WINAPI *get_WindowStyleEx)(IVideoWindow *This,long *WindowStyleEx);
      HRESULT (WINAPI *put_AutoShow)(IVideoWindow *This,long AutoShow);
      HRESULT (WINAPI *get_AutoShow)(IVideoWindow *This,long *AutoShow);
      HRESULT (WINAPI *put_WindowState)(IVideoWindow *This,long WindowState);
      HRESULT (WINAPI *get_WindowState)(IVideoWindow *This,long *WindowState);
      HRESULT (WINAPI *put_BackgroundPalette)(IVideoWindow *This,long BackgroundPalette);
      HRESULT (WINAPI *get_BackgroundPalette)(IVideoWindow *This,long *pBackgroundPalette);
      HRESULT (WINAPI *put_Visible)(IVideoWindow *This,long Visible);
      HRESULT (WINAPI *get_Visible)(IVideoWindow *This,long *pVisible);
      HRESULT (WINAPI *put_Left)(IVideoWindow *This,long Left);
      HRESULT (WINAPI *get_Left)(IVideoWindow *This,long *pLeft);
      HRESULT (WINAPI *put_Width)(IVideoWindow *This,long Width);
      HRESULT (WINAPI *get_Width)(IVideoWindow *This,long *pWidth);
      HRESULT (WINAPI *put_Top)(IVideoWindow *This,long Top);
      HRESULT (WINAPI *get_Top)(IVideoWindow *This,long *pTop);
      HRESULT (WINAPI *put_Height)(IVideoWindow *This,long Height);
      HRESULT (WINAPI *get_Height)(IVideoWindow *This,long *pHeight);
      HRESULT (WINAPI *put_Owner)(IVideoWindow *This,OAHWND Owner);
      HRESULT (WINAPI *get_Owner)(IVideoWindow *This,OAHWND *Owner);
      HRESULT (WINAPI *put_MessageDrain)(IVideoWindow *This,OAHWND Drain);
      HRESULT (WINAPI *get_MessageDrain)(IVideoWindow *This,OAHWND *Drain);
      HRESULT (WINAPI *get_BorderColor)(IVideoWindow *This,long *Color);
      HRESULT (WINAPI *put_BorderColor)(IVideoWindow *This,long Color);
      HRESULT (WINAPI *get_FullScreenMode)(IVideoWindow *This,long *FullScreenMode);
      HRESULT (WINAPI *put_FullScreenMode)(IVideoWindow *This,long FullScreenMode);
      HRESULT (WINAPI *SetWindowForeground)(IVideoWindow *This,long Focus);
      HRESULT (WINAPI *NotifyOwnerMessage)(IVideoWindow *This,OAHWND hwnd,long uMsg,LONG_PTR wParam,LONG_PTR lParam);
      HRESULT (WINAPI *SetWindowPosition)(IVideoWindow *This,long Left,long Top,long Width,long Height);
      HRESULT (WINAPI *GetWindowPosition)(IVideoWindow *This,long *pLeft,long *pTop,long *pWidth,long *pHeight);
      HRESULT (WINAPI *GetMinIdealImageSize)(IVideoWindow *This,long *pWidth,long *pHeight);
      HRESULT (WINAPI *GetMaxIdealImageSize)(IVideoWindow *This,long *pWidth,long *pHeight);
      HRESULT (WINAPI *GetRestorePosition)(IVideoWindow *This,long *pLeft,long *pTop,long *pWidth,long *pHeight);
      HRESULT (WINAPI *HideCursor)(IVideoWindow *This,long HideCursor);
      HRESULT (WINAPI *IsCursorHidden)(IVideoWindow *This,long *CursorHidden);
    END_INTERFACE
  } IVideoWindowVtbl;
  struct IVideoWindow {
    CONST_VTBL struct IVideoWindowVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IVideoWindow_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IVideoWindow_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IVideoWindow_Release(This) (This)->lpVtbl->Release(This)
#define IVideoWindow_GetTypeInfoCount(This,pctinfo) (This)->lpVtbl->GetTypeInfoCount(This,pctinfo)
#define IVideoWindow_GetTypeInfo(This,iTInfo,lcid,ppTInfo) (This)->lpVtbl->GetTypeInfo(This,iTInfo,lcid,ppTInfo)
#define IVideoWindow_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) (This)->lpVtbl->GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)
#define IVideoWindow_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) (This)->lpVtbl->Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)
#define IVideoWindow_put_Caption(This,strCaption) (This)->lpVtbl->put_Caption(This,strCaption)
#define IVideoWindow_get_Caption(This,strCaption) (This)->lpVtbl->get_Caption(This,strCaption)
#define IVideoWindow_put_WindowStyle(This,WindowStyle) (This)->lpVtbl->put_WindowStyle(This,WindowStyle)
#define IVideoWindow_get_WindowStyle(This,WindowStyle) (This)->lpVtbl->get_WindowStyle(This,WindowStyle)
#define IVideoWindow_put_WindowStyleEx(This,WindowStyleEx) (This)->lpVtbl->put_WindowStyleEx(This,WindowStyleEx)
#define IVideoWindow_get_WindowStyleEx(This,WindowStyleEx) (This)->lpVtbl->get_WindowStyleEx(This,WindowStyleEx)
#define IVideoWindow_put_AutoShow(This,AutoShow) (This)->lpVtbl->put_AutoShow(This,AutoShow)
#define IVideoWindow_get_AutoShow(This,AutoShow) (This)->lpVtbl->get_AutoShow(This,AutoShow)
#define IVideoWindow_put_WindowState(This,WindowState) (This)->lpVtbl->put_WindowState(This,WindowState)
#define IVideoWindow_get_WindowState(This,WindowState) (This)->lpVtbl->get_WindowState(This,WindowState)
#define IVideoWindow_put_BackgroundPalette(This,BackgroundPalette) (This)->lpVtbl->put_BackgroundPalette(This,BackgroundPalette)
#define IVideoWindow_get_BackgroundPalette(This,pBackgroundPalette) (This)->lpVtbl->get_BackgroundPalette(This,pBackgroundPalette)
#define IVideoWindow_put_Visible(This,Visible) (This)->lpVtbl->put_Visible(This,Visible)
#define IVideoWindow_get_Visible(This,pVisible) (This)->lpVtbl->get_Visible(This,pVisible)
#define IVideoWindow_put_Left(This,Left) (This)->lpVtbl->put_Left(This,Left)
#define IVideoWindow_get_Left(This,pLeft) (This)->lpVtbl->get_Left(This,pLeft)
#define IVideoWindow_put_Width(This,Width) (This)->lpVtbl->put_Width(This,Width)
#define IVideoWindow_get_Width(This,pWidth) (This)->lpVtbl->get_Width(This,pWidth)
#define IVideoWindow_put_Top(This,Top) (This)->lpVtbl->put_Top(This,Top)
#define IVideoWindow_get_Top(This,pTop) (This)->lpVtbl->get_Top(This,pTop)
#define IVideoWindow_put_Height(This,Height) (This)->lpVtbl->put_Height(This,Height)
#define IVideoWindow_get_Height(This,pHeight) (This)->lpVtbl->get_Height(This,pHeight)
#define IVideoWindow_put_Owner(This,Owner) (This)->lpVtbl->put_Owner(This,Owner)
#define IVideoWindow_get_Owner(This,Owner) (This)->lpVtbl->get_Owner(This,Owner)
#define IVideoWindow_put_MessageDrain(This,Drain) (This)->lpVtbl->put_MessageDrain(This,Drain)
#define IVideoWindow_get_MessageDrain(This,Drain) (This)->lpVtbl->get_MessageDrain(This,Drain)
#define IVideoWindow_get_BorderColor(This,Color) (This)->lpVtbl->get_BorderColor(This,Color)
#define IVideoWindow_put_BorderColor(This,Color) (This)->lpVtbl->put_BorderColor(This,Color)
#define IVideoWindow_get_FullScreenMode(This,FullScreenMode) (This)->lpVtbl->get_FullScreenMode(This,FullScreenMode)
#define IVideoWindow_put_FullScreenMode(This,FullScreenMode) (This)->lpVtbl->put_FullScreenMode(This,FullScreenMode)
#define IVideoWindow_SetWindowForeground(This,Focus) (This)->lpVtbl->SetWindowForeground(This,Focus)
#define IVideoWindow_NotifyOwnerMessage(This,hwnd,uMsg,wParam,lParam) (This)->lpVtbl->NotifyOwnerMessage(This,hwnd,uMsg,wParam,lParam)
#define IVideoWindow_SetWindowPosition(This,Left,Top,Width,Height) (This)->lpVtbl->SetWindowPosition(This,Left,Top,Width,Height)
#define IVideoWindow_GetWindowPosition(This,pLeft,pTop,pWidth,pHeight) (This)->lpVtbl->GetWindowPosition(This,pLeft,pTop,pWidth,pHeight)
#define IVideoWindow_GetMinIdealImageSize(This,pWidth,pHeight) (This)->lpVtbl->GetMinIdealImageSize(This,pWidth,pHeight)
#define IVideoWindow_GetMaxIdealImageSize(This,pWidth,pHeight) (This)->lpVtbl->GetMaxIdealImageSize(This,pWidth,pHeight)
#define IVideoWindow_GetRestorePosition(This,pLeft,pTop,pWidth,pHeight) (This)->lpVtbl->GetRestorePosition(This,pLeft,pTop,pWidth,pHeight)
#define IVideoWindow_HideCursor(This,HideCursor) (This)->lpVtbl->HideCursor(This,HideCursor)
#define IVideoWindow_IsCursorHidden(This,CursorHidden) (This)->lpVtbl->IsCursorHidden(This,CursorHidden)
#endif
#endif
  HRESULT WINAPI IVideoWindow_put_Caption_Proxy(IVideoWindow *This,BSTR strCaption);
  void __RPC_STUB IVideoWindow_put_Caption_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_get_Caption_Proxy(IVideoWindow *This,BSTR *strCaption);
  void __RPC_STUB IVideoWindow_get_Caption_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_put_WindowStyle_Proxy(IVideoWindow *This,long WindowStyle);
  void __RPC_STUB IVideoWindow_put_WindowStyle_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_get_WindowStyle_Proxy(IVideoWindow *This,long *WindowStyle);
  void __RPC_STUB IVideoWindow_get_WindowStyle_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_put_WindowStyleEx_Proxy(IVideoWindow *This,long WindowStyleEx);
  void __RPC_STUB IVideoWindow_put_WindowStyleEx_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_get_WindowStyleEx_Proxy(IVideoWindow *This,long *WindowStyleEx);
  void __RPC_STUB IVideoWindow_get_WindowStyleEx_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_put_AutoShow_Proxy(IVideoWindow *This,long AutoShow);
  void __RPC_STUB IVideoWindow_put_AutoShow_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_get_AutoShow_Proxy(IVideoWindow *This,long *AutoShow);
  void __RPC_STUB IVideoWindow_get_AutoShow_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_put_WindowState_Proxy(IVideoWindow *This,long WindowState);
  void __RPC_STUB IVideoWindow_put_WindowState_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_get_WindowState_Proxy(IVideoWindow *This,long *WindowState);
  void __RPC_STUB IVideoWindow_get_WindowState_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_put_BackgroundPalette_Proxy(IVideoWindow *This,long BackgroundPalette);
  void __RPC_STUB IVideoWindow_put_BackgroundPalette_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_get_BackgroundPalette_Proxy(IVideoWindow *This,long *pBackgroundPalette);
  void __RPC_STUB IVideoWindow_get_BackgroundPalette_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_put_Visible_Proxy(IVideoWindow *This,long Visible);
  void __RPC_STUB IVideoWindow_put_Visible_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_get_Visible_Proxy(IVideoWindow *This,long *pVisible);
  void __RPC_STUB IVideoWindow_get_Visible_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_put_Left_Proxy(IVideoWindow *This,long Left);
  void __RPC_STUB IVideoWindow_put_Left_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_get_Left_Proxy(IVideoWindow *This,long *pLeft);
  void __RPC_STUB IVideoWindow_get_Left_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_put_Width_Proxy(IVideoWindow *This,long Width);
  void __RPC_STUB IVideoWindow_put_Width_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_get_Width_Proxy(IVideoWindow *This,long *pWidth);
  void __RPC_STUB IVideoWindow_get_Width_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_put_Top_Proxy(IVideoWindow *This,long Top);
  void __RPC_STUB IVideoWindow_put_Top_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_get_Top_Proxy(IVideoWindow *This,long *pTop);
  void __RPC_STUB IVideoWindow_get_Top_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_put_Height_Proxy(IVideoWindow *This,long Height);
  void __RPC_STUB IVideoWindow_put_Height_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_get_Height_Proxy(IVideoWindow *This,long *pHeight);
  void __RPC_STUB IVideoWindow_get_Height_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_put_Owner_Proxy(IVideoWindow *This,OAHWND Owner);
  void __RPC_STUB IVideoWindow_put_Owner_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_get_Owner_Proxy(IVideoWindow *This,OAHWND *Owner);
  void __RPC_STUB IVideoWindow_get_Owner_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_put_MessageDrain_Proxy(IVideoWindow *This,OAHWND Drain);
  void __RPC_STUB IVideoWindow_put_MessageDrain_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_get_MessageDrain_Proxy(IVideoWindow *This,OAHWND *Drain);
  void __RPC_STUB IVideoWindow_get_MessageDrain_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_get_BorderColor_Proxy(IVideoWindow *This,long *Color);
  void __RPC_STUB IVideoWindow_get_BorderColor_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_put_BorderColor_Proxy(IVideoWindow *This,long Color);
  void __RPC_STUB IVideoWindow_put_BorderColor_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_get_FullScreenMode_Proxy(IVideoWindow *This,long *FullScreenMode);
  void __RPC_STUB IVideoWindow_get_FullScreenMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_put_FullScreenMode_Proxy(IVideoWindow *This,long FullScreenMode);
  void __RPC_STUB IVideoWindow_put_FullScreenMode_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_SetWindowForeground_Proxy(IVideoWindow *This,long Focus);
  void __RPC_STUB IVideoWindow_SetWindowForeground_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_NotifyOwnerMessage_Proxy(IVideoWindow *This,OAHWND hwnd,long uMsg,LONG_PTR wParam,LONG_PTR lParam);
  void __RPC_STUB IVideoWindow_NotifyOwnerMessage_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_SetWindowPosition_Proxy(IVideoWindow *This,long Left,long Top,long Width,long Height);
  void __RPC_STUB IVideoWindow_SetWindowPosition_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_GetWindowPosition_Proxy(IVideoWindow *This,long *pLeft,long *pTop,long *pWidth,long *pHeight);
  void __RPC_STUB IVideoWindow_GetWindowPosition_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_GetMinIdealImageSize_Proxy(IVideoWindow *This,long *pWidth,long *pHeight);
  void __RPC_STUB IVideoWindow_GetMinIdealImageSize_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_GetMaxIdealImageSize_Proxy(IVideoWindow *This,long *pWidth,long *pHeight);
  void __RPC_STUB IVideoWindow_GetMaxIdealImageSize_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_GetRestorePosition_Proxy(IVideoWindow *This,long *pLeft,long *pTop,long *pWidth,long *pHeight);
  void __RPC_STUB IVideoWindow_GetRestorePosition_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_HideCursor_Proxy(IVideoWindow *This,long HideCursor);
  void __RPC_STUB IVideoWindow_HideCursor_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IVideoWindow_IsCursorHidden_Proxy(IVideoWindow *This,long *CursorHidden);
  void __RPC_STUB IVideoWindow_IsCursorHidden_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IBasicVideo_INTERFACE_DEFINED__
#define __IBasicVideo_INTERFACE_DEFINED__
  DEFINE_GUID(IID_IBasicVideo,0x56a868b5,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IBasicVideo : public IDispatch {
  public:
    virtual HRESULT WINAPI get_AvgTimePerFrame(REFTIME *pAvgTimePerFrame) = 0;
    virtual HRESULT WINAPI get_BitRate(long *pBitRate) = 0;
    virtual HRESULT WINAPI get_BitErrorRate(long *pBitErrorRate) = 0;
    virtual HRESULT WINAPI get_VideoWidth(long *pVideoWidth) = 0;
    virtual HRESULT WINAPI get_VideoHeight(long *pVideoHeight) = 0;
    virtual HRESULT WINAPI put_SourceLeft(long SourceLeft) = 0;
    virtual HRESULT WINAPI get_SourceLeft(long *pSourceLeft) = 0;
    virtual HRESULT WINAPI put_SourceWidth(long SourceWidth) = 0;
    virtual HRESULT WINAPI get_SourceWidth(long *pSourceWidth) = 0;
    virtual HRESULT WINAPI put_SourceTop(long SourceTop) = 0;
    virtual HRESULT WINAPI get_SourceTop(long *pSourceTop) = 0;
    virtual HRESULT WINAPI put_SourceHeight(long SourceHeight) = 0;
    virtual HRESULT WINAPI get_SourceHeight(long *pSourceHeight) = 0;
    virtual HRESULT WINAPI put_DestinationLeft(long DestinationLeft) = 0;
    virtual HRESULT WINAPI get_DestinationLeft(long *pDestinationLeft) = 0;
    virtual HRESULT WINAPI put_DestinationWidth(long DestinationWidth) = 0;
    virtual HRESULT WINAPI get_DestinationWidth(long *pDestinationWidth) = 0;
    virtual HRESULT WINAPI put_DestinationTop(long DestinationTop) = 0;
    virtual HRESULT WINAPI get_DestinationTop(long *pDestinationTop) = 0;
    virtual HRESULT WINAPI put_DestinationHeight(long DestinationHeight) = 0;
    virtual HRESULT WINAPI get_DestinationHeight(long *pDestinationHeight) = 0;
    virtual HRESULT WINAPI SetSourcePosition(long Left,long Top,long Width,long Height) = 0;
    virtual HRESULT WINAPI GetSourcePosition(long *pLeft,long *pTop,long *pWidth,long *pHeight) = 0;
    virtual HRESULT WINAPI SetDefaultSourcePosition(void) = 0;
    virtual HRESULT WINAPI SetDestinationPosition(long Left,long Top,long Width,long Height) = 0;
    virtual HRESULT WINAPI GetDestinationPosition(long *pLeft,long *pTop,long *pWidth,long *pHeight) = 0;
    virtual HRESULT WINAPI SetDefaultDestinationPosition(void) = 0;
    virtual HRESULT WINAPI GetVideoSize(long *pWidth,long *pHeight) = 0;
    virtual HRESULT WINAPI GetVideoPaletteEntries(long StartIndex,long Entries,long *pRetrieved,long *pPalette) = 0;
    virtual HRESULT WINAPI GetCurrentImage(long *pBufferSize,long *pDIBImage) = 0;
    virtual HRESULT WINAPI IsUsingDefaultSource(void) = 0;
    virtual HRESULT WINAPI IsUsingDefaultDestination(void) = 0;
  };
#else
  typedef struct IBasicVideoVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IBasicVideo *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IBasicVideo *This);
      ULONG (WINAPI *Release)(IBasicVideo *This);
      HRESULT (WINAPI *GetTypeInfoCount)(IBasicVideo *This,UINT *pctinfo);
      HRESULT (WINAPI *GetTypeInfo)(IBasicVideo *This,UINT iTInfo,LCID lcid,ITypeInfo **ppTInfo);
      HRESULT (WINAPI *GetIDsOfNames)(IBasicVideo *This,REFIID riid,LPOLESTR *rgszNames,UINT cNames,LCID lcid,DISPID *rgDispId);
      HRESULT (WINAPI *Invoke)(IBasicVideo *This,DISPID dispIdMember,REFIID riid,LCID lcid,WORD wFlags,DISPPARAMS *pDispParams,VARIANT *pVarResult,EXCEPINFO *pExcepInfo,UINT *puArgErr);
      HRESULT (WINAPI *get_AvgTimePerFrame)(IBasicVideo *This,REFTIME *pAvgTimePerFrame);
      HRESULT (WINAPI *get_BitRate)(IBasicVideo *This,long *pBitRate);
      HRESULT (WINAPI *get_BitErrorRate)(IBasicVideo *This,long *pBitErrorRate);
      HRESULT (WINAPI *get_VideoWidth)(IBasicVideo *This,long *pVideoWidth);
      HRESULT (WINAPI *get_VideoHeight)(IBasicVideo *This,long *pVideoHeight);
      HRESULT (WINAPI *put_SourceLeft)(IBasicVideo *This,long SourceLeft);
      HRESULT (WINAPI *get_SourceLeft)(IBasicVideo *This,long *pSourceLeft);
      HRESULT (WINAPI *put_SourceWidth)(IBasicVideo *This,long SourceWidth);
      HRESULT (WINAPI *get_SourceWidth)(IBasicVideo *This,long *pSourceWidth);
      HRESULT (WINAPI *put_SourceTop)(IBasicVideo *This,long SourceTop);
      HRESULT (WINAPI *get_SourceTop)(IBasicVideo *This,long *pSourceTop);
      HRESULT (WINAPI *put_SourceHeight)(IBasicVideo *This,long SourceHeight);
      HRESULT (WINAPI *get_SourceHeight)(IBasicVideo *This,long *pSourceHeight);
      HRESULT (WINAPI *put_DestinationLeft)(IBasicVideo *This,long DestinationLeft);
      HRESULT (WINAPI *get_DestinationLeft)(IBasicVideo *This,long *pDestinationLeft);
      HRESULT (WINAPI *put_DestinationWidth)(IBasicVideo *This,long DestinationWidth);
      HRESULT (WINAPI *get_DestinationWidth)(IBasicVideo *This,long *pDestinationWidth);
      HRESULT (WINAPI *put_DestinationTop)(IBasicVideo *This,long DestinationTop);
      HRESULT (WINAPI *get_DestinationTop)(IBasicVideo *This,long *pDestinationTop);
      HRESULT (WINAPI *put_DestinationHeight)(IBasicVideo *This,long DestinationHeight);
      HRESULT (WINAPI *get_DestinationHeight)(IBasicVideo *This,long *pDestinationHeight);
      HRESULT (WINAPI *SetSourcePosition)(IBasicVideo *This,long Left,long Top,long Width,long Height);
      HRESULT (WINAPI *GetSourcePosition)(IBasicVideo *This,long *pLeft,long *pTop,long *pWidth,long *pHeight);
      HRESULT (WINAPI *SetDefaultSourcePosition)(IBasicVideo *This);
      HRESULT (WINAPI *SetDestinationPosition)(IBasicVideo *This,long Left,long Top,long Width,long Height);
      HRESULT (WINAPI *GetDestinationPosition)(IBasicVideo *This,long *pLeft,long *pTop,long *pWidth,long *pHeight);
      HRESULT (WINAPI *SetDefaultDestinationPosition)(IBasicVideo *This);
      HRESULT (WINAPI *GetVideoSize)(IBasicVideo *This,long *pWidth,long *pHeight);
      HRESULT (WINAPI *GetVideoPaletteEntries)(IBasicVideo *This,long StartIndex,long Entries,long *pRetrieved,long *pPalette);
      HRESULT (WINAPI *GetCurrentImage)(IBasicVideo *This,long *pBufferSize,long *pDIBImage);
      HRESULT (WINAPI *IsUsingDefaultSource)(IBasicVideo *This);
      HRESULT (WINAPI *IsUsingDefaultDestination)(IBasicVideo *This);
    END_INTERFACE
  } IBasicVideoVtbl;
  struct IBasicVideo {
    CONST_VTBL struct IBasicVideoVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IBasicVideo_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IBasicVideo_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IBasicVideo_Release(This) (This)->lpVtbl->Release(This)
#define IBasicVideo_GetTypeInfoCount(This,pctinfo) (This)->lpVtbl->GetTypeInfoCount(This,pctinfo)
#define IBasicVideo_GetTypeInfo(This,iTInfo,lcid,ppTInfo) (This)->lpVtbl->GetTypeInfo(This,iTInfo,lcid,ppTInfo)
#define IBasicVideo_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) (This)->lpVtbl->GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)
#define IBasicVideo_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) (This)->lpVtbl->Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)
#define IBasicVideo_get_AvgTimePerFrame(This,pAvgTimePerFrame) (This)->lpVtbl->get_AvgTimePerFrame(This,pAvgTimePerFrame)
#define IBasicVideo_get_BitRate(This,pBitRate) (This)->lpVtbl->get_BitRate(This,pBitRate)
#define IBasicVideo_get_BitErrorRate(This,pBitErrorRate) (This)->lpVtbl->get_BitErrorRate(This,pBitErrorRate)
#define IBasicVideo_get_VideoWidth(This,pVideoWidth) (This)->lpVtbl->get_VideoWidth(This,pVideoWidth)
#define IBasicVideo_get_VideoHeight(This,pVideoHeight) (This)->lpVtbl->get_VideoHeight(This,pVideoHeight)
#define IBasicVideo_put_SourceLeft(This,SourceLeft) (This)->lpVtbl->put_SourceLeft(This,SourceLeft)
#define IBasicVideo_get_SourceLeft(This,pSourceLeft) (This)->lpVtbl->get_SourceLeft(This,pSourceLeft)
#define IBasicVideo_put_SourceWidth(This,SourceWidth) (This)->lpVtbl->put_SourceWidth(This,SourceWidth)
#define IBasicVideo_get_SourceWidth(This,pSourceWidth) (This)->lpVtbl->get_SourceWidth(This,pSourceWidth)
#define IBasicVideo_put_SourceTop(This,SourceTop) (This)->lpVtbl->put_SourceTop(This,SourceTop)
#define IBasicVideo_get_SourceTop(This,pSourceTop) (This)->lpVtbl->get_SourceTop(This,pSourceTop)
#define IBasicVideo_put_SourceHeight(This,SourceHeight) (This)->lpVtbl->put_SourceHeight(This,SourceHeight)
#define IBasicVideo_get_SourceHeight(This,pSourceHeight) (This)->lpVtbl->get_SourceHeight(This,pSourceHeight)
#define IBasicVideo_put_DestinationLeft(This,DestinationLeft) (This)->lpVtbl->put_DestinationLeft(This,DestinationLeft)
#define IBasicVideo_get_DestinationLeft(This,pDestinationLeft) (This)->lpVtbl->get_DestinationLeft(This,pDestinationLeft)
#define IBasicVideo_put_DestinationWidth(This,DestinationWidth) (This)->lpVtbl->put_DestinationWidth(This,DestinationWidth)
#define IBasicVideo_get_DestinationWidth(This,pDestinationWidth) (This)->lpVtbl->get_DestinationWidth(This,pDestinationWidth)
#define IBasicVideo_put_DestinationTop(This,DestinationTop) (This)->lpVtbl->put_DestinationTop(This,DestinationTop)
#define IBasicVideo_get_DestinationTop(This,pDestinationTop) (This)->lpVtbl->get_DestinationTop(This,pDestinationTop)
#define IBasicVideo_put_DestinationHeight(This,DestinationHeight) (This)->lpVtbl->put_DestinationHeight(This,DestinationHeight)
#define IBasicVideo_get_DestinationHeight(This,pDestinationHeight) (This)->lpVtbl->get_DestinationHeight(This,pDestinationHeight)
#define IBasicVideo_SetSourcePosition(This,Left,Top,Width,Height) (This)->lpVtbl->SetSourcePosition(This,Left,Top,Width,Height)
#define IBasicVideo_GetSourcePosition(This,pLeft,pTop,pWidth,pHeight) (This)->lpVtbl->GetSourcePosition(This,pLeft,pTop,pWidth,pHeight)
#define IBasicVideo_SetDefaultSourcePosition(This) (This)->lpVtbl->SetDefaultSourcePosition(This)
#define IBasicVideo_SetDestinationPosition(This,Left,Top,Width,Height) (This)->lpVtbl->SetDestinationPosition(This,Left,Top,Width,Height)
#define IBasicVideo_GetDestinationPosition(This,pLeft,pTop,pWidth,pHeight) (This)->lpVtbl->GetDestinationPosition(This,pLeft,pTop,pWidth,pHeight)
#define IBasicVideo_SetDefaultDestinationPosition(This) (This)->lpVtbl->SetDefaultDestinationPosition(This)
#define IBasicVideo_GetVideoSize(This,pWidth,pHeight) (This)->lpVtbl->GetVideoSize(This,pWidth,pHeight)
#define IBasicVideo_GetVideoPaletteEntries(This,StartIndex,Entries,pRetrieved,pPalette) (This)->lpVtbl->GetVideoPaletteEntries(This,StartIndex,Entries,pRetrieved,pPalette)
#define IBasicVideo_GetCurrentImage(This,pBufferSize,pDIBImage) (This)->lpVtbl->GetCurrentImage(This,pBufferSize,pDIBImage)
#define IBasicVideo_IsUsingDefaultSource(This) (This)->lpVtbl->IsUsingDefaultSource(This)
#define IBasicVideo_IsUsingDefaultDestination(This) (This)->lpVtbl->IsUsingDefaultDestination(This)
#endif
#endif
  HRESULT WINAPI IBasicVideo_get_AvgTimePerFrame_Proxy(IBasicVideo *This,REFTIME *pAvgTimePerFrame);
  void __RPC_STUB IBasicVideo_get_AvgTimePerFrame_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_get_BitRate_Proxy(IBasicVideo *This,long *pBitRate);
  void __RPC_STUB IBasicVideo_get_BitRate_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_get_BitErrorRate_Proxy(IBasicVideo *This,long *pBitErrorRate);
  void __RPC_STUB IBasicVideo_get_BitErrorRate_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_get_VideoWidth_Proxy(IBasicVideo *This,long *pVideoWidth);
  void __RPC_STUB IBasicVideo_get_VideoWidth_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_get_VideoHeight_Proxy(IBasicVideo *This,long *pVideoHeight);
  void __RPC_STUB IBasicVideo_get_VideoHeight_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_put_SourceLeft_Proxy(IBasicVideo *This,long SourceLeft);
  void __RPC_STUB IBasicVideo_put_SourceLeft_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_get_SourceLeft_Proxy(IBasicVideo *This,long *pSourceLeft);
  void __RPC_STUB IBasicVideo_get_SourceLeft_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_put_SourceWidth_Proxy(IBasicVideo *This,long SourceWidth);
  void __RPC_STUB IBasicVideo_put_SourceWidth_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_get_SourceWidth_Proxy(IBasicVideo *This,long *pSourceWidth);
  void __RPC_STUB IBasicVideo_get_SourceWidth_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_put_SourceTop_Proxy(IBasicVideo *This,long SourceTop);
  void __RPC_STUB IBasicVideo_put_SourceTop_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_get_SourceTop_Proxy(IBasicVideo *This,long *pSourceTop);
  void __RPC_STUB IBasicVideo_get_SourceTop_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_put_SourceHeight_Proxy(IBasicVideo *This,long SourceHeight);
  void __RPC_STUB IBasicVideo_put_SourceHeight_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_get_SourceHeight_Proxy(IBasicVideo *This,long *pSourceHeight);
  void __RPC_STUB IBasicVideo_get_SourceHeight_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_put_DestinationLeft_Proxy(IBasicVideo *This,long DestinationLeft);
  void __RPC_STUB IBasicVideo_put_DestinationLeft_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_get_DestinationLeft_Proxy(IBasicVideo *This,long *pDestinationLeft);
  void __RPC_STUB IBasicVideo_get_DestinationLeft_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_put_DestinationWidth_Proxy(IBasicVideo *This,long DestinationWidth);
  void __RPC_STUB IBasicVideo_put_DestinationWidth_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_get_DestinationWidth_Proxy(IBasicVideo *This,long *pDestinationWidth);
  void __RPC_STUB IBasicVideo_get_DestinationWidth_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_put_DestinationTop_Proxy(IBasicVideo *This,long DestinationTop);
  void __RPC_STUB IBasicVideo_put_DestinationTop_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_get_DestinationTop_Proxy(IBasicVideo *This,long *pDestinationTop);
  void __RPC_STUB IBasicVideo_get_DestinationTop_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_put_DestinationHeight_Proxy(IBasicVideo *This,long DestinationHeight);
  void __RPC_STUB IBasicVideo_put_DestinationHeight_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_get_DestinationHeight_Proxy(IBasicVideo *This,long *pDestinationHeight);
  void __RPC_STUB IBasicVideo_get_DestinationHeight_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_SetSourcePosition_Proxy(IBasicVideo *This,long Left,long Top,long Width,long Height);
  void __RPC_STUB IBasicVideo_SetSourcePosition_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_GetSourcePosition_Proxy(IBasicVideo *This,long *pLeft,long *pTop,long *pWidth,long *pHeight);
  void __RPC_STUB IBasicVideo_GetSourcePosition_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_SetDefaultSourcePosition_Proxy(IBasicVideo *This);
  void __RPC_STUB IBasicVideo_SetDefaultSourcePosition_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_SetDestinationPosition_Proxy(IBasicVideo *This,long Left,long Top,long Width,long Height);
  void __RPC_STUB IBasicVideo_SetDestinationPosition_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_GetDestinationPosition_Proxy(IBasicVideo *This,long *pLeft,long *pTop,long *pWidth,long *pHeight);
  void __RPC_STUB IBasicVideo_GetDestinationPosition_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_SetDefaultDestinationPosition_Proxy(IBasicVideo *This);
  void __RPC_STUB IBasicVideo_SetDefaultDestinationPosition_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_GetVideoSize_Proxy(IBasicVideo *This,long *pWidth,long *pHeight);
  void __RPC_STUB IBasicVideo_GetVideoSize_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_GetVideoPaletteEntries_Proxy(IBasicVideo *This,long StartIndex,long Entries,long *pRetrieved,long *pPalette);
  void __RPC_STUB IBasicVideo_GetVideoPaletteEntries_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_GetCurrentImage_Proxy(IBasicVideo *This,long *pBufferSize,long *pDIBImage);
  void __RPC_STUB IBasicVideo_GetCurrentImage_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_IsUsingDefaultSource_Proxy(IBasicVideo *This);
  void __RPC_STUB IBasicVideo_IsUsingDefaultSource_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IBasicVideo_IsUsingDefaultDestination_Proxy(IBasicVideo *This);
  void __RPC_STUB IBasicVideo_IsUsingDefaultDestination_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IBasicVideo2_INTERFACE_DEFINED__
#define __IBasicVideo2_INTERFACE_DEFINED__
  DEFINE_GUID(IID_IBasicVideo2,0x329bb360,0xf6ea,0x11d1,0x90,0x38,0x00,0xa0,0xc9,0x69,0x72,0x98);
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IBasicVideo2 : public IBasicVideo {
  public:
    virtual HRESULT WINAPI GetPreferredAspectRatio(long *plAspectX,long *plAspectY) = 0;
  };
#else
  typedef struct IBasicVideo2Vtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IBasicVideo2 *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IBasicVideo2 *This);
      ULONG (WINAPI *Release)(IBasicVideo2 *This);
      HRESULT (WINAPI *GetTypeInfoCount)(IBasicVideo2 *This,UINT *pctinfo);
      HRESULT (WINAPI *GetTypeInfo)(IBasicVideo2 *This,UINT iTInfo,LCID lcid,ITypeInfo **ppTInfo);
      HRESULT (WINAPI *GetIDsOfNames)(IBasicVideo2 *This,REFIID riid,LPOLESTR *rgszNames,UINT cNames,LCID lcid,DISPID *rgDispId);
      HRESULT (WINAPI *Invoke)(IBasicVideo2 *This,DISPID dispIdMember,REFIID riid,LCID lcid,WORD wFlags,DISPPARAMS *pDispParams,VARIANT *pVarResult,EXCEPINFO *pExcepInfo,UINT *puArgErr);
      HRESULT (WINAPI *get_AvgTimePerFrame)(IBasicVideo2 *This,REFTIME *pAvgTimePerFrame);
      HRESULT (WINAPI *get_BitRate)(IBasicVideo2 *This,long *pBitRate);
      HRESULT (WINAPI *get_BitErrorRate)(IBasicVideo2 *This,long *pBitErrorRate);
      HRESULT (WINAPI *get_VideoWidth)(IBasicVideo2 *This,long *pVideoWidth);
      HRESULT (WINAPI *get_VideoHeight)(IBasicVideo2 *This,long *pVideoHeight);
      HRESULT (WINAPI *put_SourceLeft)(IBasicVideo2 *This,long SourceLeft);
      HRESULT (WINAPI *get_SourceLeft)(IBasicVideo2 *This,long *pSourceLeft);
      HRESULT (WINAPI *put_SourceWidth)(IBasicVideo2 *This,long SourceWidth);
      HRESULT (WINAPI *get_SourceWidth)(IBasicVideo2 *This,long *pSourceWidth);
      HRESULT (WINAPI *put_SourceTop)(IBasicVideo2 *This,long SourceTop);
      HRESULT (WINAPI *get_SourceTop)(IBasicVideo2 *This,long *pSourceTop);
      HRESULT (WINAPI *put_SourceHeight)(IBasicVideo2 *This,long SourceHeight);
      HRESULT (WINAPI *get_SourceHeight)(IBasicVideo2 *This,long *pSourceHeight);
      HRESULT (WINAPI *put_DestinationLeft)(IBasicVideo2 *This,long DestinationLeft);
      HRESULT (WINAPI *get_DestinationLeft)(IBasicVideo2 *This,long *pDestinationLeft);
      HRESULT (WINAPI *put_DestinationWidth)(IBasicVideo2 *This,long DestinationWidth);
      HRESULT (WINAPI *get_DestinationWidth)(IBasicVideo2 *This,long *pDestinationWidth);
      HRESULT (WINAPI *put_DestinationTop)(IBasicVideo2 *This,long DestinationTop);
      HRESULT (WINAPI *get_DestinationTop)(IBasicVideo2 *This,long *pDestinationTop);
      HRESULT (WINAPI *put_DestinationHeight)(IBasicVideo2 *This,long DestinationHeight);
      HRESULT (WINAPI *get_DestinationHeight)(IBasicVideo2 *This,long *pDestinationHeight);
      HRESULT (WINAPI *SetSourcePosition)(IBasicVideo2 *This,long Left,long Top,long Width,long Height);
      HRESULT (WINAPI *GetSourcePosition)(IBasicVideo2 *This,long *pLeft,long *pTop,long *pWidth,long *pHeight);
      HRESULT (WINAPI *SetDefaultSourcePosition)(IBasicVideo2 *This);
      HRESULT (WINAPI *SetDestinationPosition)(IBasicVideo2 *This,long Left,long Top,long Width,long Height);
      HRESULT (WINAPI *GetDestinationPosition)(IBasicVideo2 *This,long *pLeft,long *pTop,long *pWidth,long *pHeight);
      HRESULT (WINAPI *SetDefaultDestinationPosition)(IBasicVideo2 *This);
      HRESULT (WINAPI *GetVideoSize)(IBasicVideo2 *This,long *pWidth,long *pHeight);
      HRESULT (WINAPI *GetVideoPaletteEntries)(IBasicVideo2 *This,long StartIndex,long Entries,long *pRetrieved,long *pPalette);
      HRESULT (WINAPI *GetCurrentImage)(IBasicVideo2 *This,long *pBufferSize,long *pDIBImage);
      HRESULT (WINAPI *IsUsingDefaultSource)(IBasicVideo2 *This);
      HRESULT (WINAPI *IsUsingDefaultDestination)(IBasicVideo2 *This);
      HRESULT (WINAPI *GetPreferredAspectRatio)(IBasicVideo2 *This,long *plAspectX,long *plAspectY);
    END_INTERFACE
  } IBasicVideo2Vtbl;
  struct IBasicVideo2 {
    CONST_VTBL struct IBasicVideo2Vtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IBasicVideo2_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IBasicVideo2_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IBasicVideo2_Release(This) (This)->lpVtbl->Release(This)
#define IBasicVideo2_GetTypeInfoCount(This,pctinfo) (This)->lpVtbl->GetTypeInfoCount(This,pctinfo)
#define IBasicVideo2_GetTypeInfo(This,iTInfo,lcid,ppTInfo) (This)->lpVtbl->GetTypeInfo(This,iTInfo,lcid,ppTInfo)
#define IBasicVideo2_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) (This)->lpVtbl->GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)
#define IBasicVideo2_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) (This)->lpVtbl->Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)
#define IBasicVideo2_get_AvgTimePerFrame(This,pAvgTimePerFrame) (This)->lpVtbl->get_AvgTimePerFrame(This,pAvgTimePerFrame)
#define IBasicVideo2_get_BitRate(This,pBitRate) (This)->lpVtbl->get_BitRate(This,pBitRate)
#define IBasicVideo2_get_BitErrorRate(This,pBitErrorRate) (This)->lpVtbl->get_BitErrorRate(This,pBitErrorRate)
#define IBasicVideo2_get_VideoWidth(This,pVideoWidth) (This)->lpVtbl->get_VideoWidth(This,pVideoWidth)
#define IBasicVideo2_get_VideoHeight(This,pVideoHeight) (This)->lpVtbl->get_VideoHeight(This,pVideoHeight)
#define IBasicVideo2_put_SourceLeft(This,SourceLeft) (This)->lpVtbl->put_SourceLeft(This,SourceLeft)
#define IBasicVideo2_get_SourceLeft(This,pSourceLeft) (This)->lpVtbl->get_SourceLeft(This,pSourceLeft)
#define IBasicVideo2_put_SourceWidth(This,SourceWidth) (This)->lpVtbl->put_SourceWidth(This,SourceWidth)
#define IBasicVideo2_get_SourceWidth(This,pSourceWidth) (This)->lpVtbl->get_SourceWidth(This,pSourceWidth)
#define IBasicVideo2_put_SourceTop(This,SourceTop) (This)->lpVtbl->put_SourceTop(This,SourceTop)
#define IBasicVideo2_get_SourceTop(This,pSourceTop) (This)->lpVtbl->get_SourceTop(This,pSourceTop)
#define IBasicVideo2_put_SourceHeight(This,SourceHeight) (This)->lpVtbl->put_SourceHeight(This,SourceHeight)
#define IBasicVideo2_get_SourceHeight(This,pSourceHeight) (This)->lpVtbl->get_SourceHeight(This,pSourceHeight)
#define IBasicVideo2_put_DestinationLeft(This,DestinationLeft) (This)->lpVtbl->put_DestinationLeft(This,DestinationLeft)
#define IBasicVideo2_get_DestinationLeft(This,pDestinationLeft) (This)->lpVtbl->get_DestinationLeft(This,pDestinationLeft)
#define IBasicVideo2_put_DestinationWidth(This,DestinationWidth) (This)->lpVtbl->put_DestinationWidth(This,DestinationWidth)
#define IBasicVideo2_get_DestinationWidth(This,pDestinationWidth) (This)->lpVtbl->get_DestinationWidth(This,pDestinationWidth)
#define IBasicVideo2_put_DestinationTop(This,DestinationTop) (This)->lpVtbl->put_DestinationTop(This,DestinationTop)
#define IBasicVideo2_get_DestinationTop(This,pDestinationTop) (This)->lpVtbl->get_DestinationTop(This,pDestinationTop)
#define IBasicVideo2_put_DestinationHeight(This,DestinationHeight) (This)->lpVtbl->put_DestinationHeight(This,DestinationHeight)
#define IBasicVideo2_get_DestinationHeight(This,pDestinationHeight) (This)->lpVtbl->get_DestinationHeight(This,pDestinationHeight)
#define IBasicVideo2_SetSourcePosition(This,Left,Top,Width,Height) (This)->lpVtbl->SetSourcePosition(This,Left,Top,Width,Height)
#define IBasicVideo2_GetSourcePosition(This,pLeft,pTop,pWidth,pHeight) (This)->lpVtbl->GetSourcePosition(This,pLeft,pTop,pWidth,pHeight)
#define IBasicVideo2_SetDefaultSourcePosition(This) (This)->lpVtbl->SetDefaultSourcePosition(This)
#define IBasicVideo2_SetDestinationPosition(This,Left,Top,Width,Height) (This)->lpVtbl->SetDestinationPosition(This,Left,Top,Width,Height)
#define IBasicVideo2_GetDestinationPosition(This,pLeft,pTop,pWidth,pHeight) (This)->lpVtbl->GetDestinationPosition(This,pLeft,pTop,pWidth,pHeight)
#define IBasicVideo2_SetDefaultDestinationPosition(This) (This)->lpVtbl->SetDefaultDestinationPosition(This)
#define IBasicVideo2_GetVideoSize(This,pWidth,pHeight) (This)->lpVtbl->GetVideoSize(This,pWidth,pHeight)
#define IBasicVideo2_GetVideoPaletteEntries(This,StartIndex,Entries,pRetrieved,pPalette) (This)->lpVtbl->GetVideoPaletteEntries(This,StartIndex,Entries,pRetrieved,pPalette)
#define IBasicVideo2_GetCurrentImage(This,pBufferSize,pDIBImage) (This)->lpVtbl->GetCurrentImage(This,pBufferSize,pDIBImage)
#define IBasicVideo2_IsUsingDefaultSource(This) (This)->lpVtbl->IsUsingDefaultSource(This)
#define IBasicVideo2_IsUsingDefaultDestination(This) (This)->lpVtbl->IsUsingDefaultDestination(This)
#define IBasicVideo2_GetPreferredAspectRatio(This,plAspectX,plAspectY) (This)->lpVtbl->GetPreferredAspectRatio(This,plAspectX,plAspectY)
#endif
#endif
  HRESULT WINAPI IBasicVideo2_GetPreferredAspectRatio_Proxy(IBasicVideo2 *This,long *plAspectX,long *plAspectY);
  void __RPC_STUB IBasicVideo2_GetPreferredAspectRatio_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IDeferredCommand_INTERFACE_DEFINED__
#define __IDeferredCommand_INTERFACE_DEFINED__
  DEFINE_GUID(IID_IDeferredCommand,0x56a868b8,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IDeferredCommand : public IUnknown {
  public:
    virtual HRESULT WINAPI Cancel(void) = 0;
    virtual HRESULT WINAPI Confidence(LONG *pConfidence) = 0;
    virtual HRESULT WINAPI Postpone(REFTIME newtime) = 0;
    virtual HRESULT WINAPI GetHResult(HRESULT *phrResult) = 0;
  };
#else
  typedef struct IDeferredCommandVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IDeferredCommand *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IDeferredCommand *This);
      ULONG (WINAPI *Release)(IDeferredCommand *This);
      HRESULT (WINAPI *Cancel)(IDeferredCommand *This);
      HRESULT (WINAPI *Confidence)(IDeferredCommand *This,LONG *pConfidence);
      HRESULT (WINAPI *Postpone)(IDeferredCommand *This,REFTIME newtime);
      HRESULT (WINAPI *GetHResult)(IDeferredCommand *This,HRESULT *phrResult);
    END_INTERFACE
  } IDeferredCommandVtbl;
  struct IDeferredCommand {
    CONST_VTBL struct IDeferredCommandVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IDeferredCommand_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IDeferredCommand_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IDeferredCommand_Release(This) (This)->lpVtbl->Release(This)
#define IDeferredCommand_Cancel(This) (This)->lpVtbl->Cancel(This)
#define IDeferredCommand_Confidence(This,pConfidence) (This)->lpVtbl->Confidence(This,pConfidence)
#define IDeferredCommand_Postpone(This,newtime) (This)->lpVtbl->Postpone(This,newtime)
#define IDeferredCommand_GetHResult(This,phrResult) (This)->lpVtbl->GetHResult(This,phrResult)
#endif
#endif
  HRESULT WINAPI IDeferredCommand_Cancel_Proxy(IDeferredCommand *This);
  void __RPC_STUB IDeferredCommand_Cancel_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDeferredCommand_Confidence_Proxy(IDeferredCommand *This,LONG *pConfidence);
  void __RPC_STUB IDeferredCommand_Confidence_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDeferredCommand_Postpone_Proxy(IDeferredCommand *This,REFTIME newtime);
  void __RPC_STUB IDeferredCommand_Postpone_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IDeferredCommand_GetHResult_Proxy(IDeferredCommand *This,HRESULT *phrResult);
  void __RPC_STUB IDeferredCommand_GetHResult_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IQueueCommand_INTERFACE_DEFINED__
#define __IQueueCommand_INTERFACE_DEFINED__
  DEFINE_GUID(IID_IQueueCommand,0x56a868b7,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IQueueCommand : public IUnknown {
  public:
    virtual HRESULT WINAPI InvokeAtStreamTime(IDeferredCommand **pCmd,REFTIME time,GUID *iid,long dispidMethod,short wFlags,long cArgs,VARIANT *pDispParams,VARIANT *pvarResult,short *puArgErr) = 0;
    virtual HRESULT WINAPI InvokeAtPresentationTime(IDeferredCommand **pCmd,REFTIME time,GUID *iid,long dispidMethod,short wFlags,long cArgs,VARIANT *pDispParams,VARIANT *pvarResult,short *puArgErr) = 0;
  };
#else
  typedef struct IQueueCommandVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IQueueCommand *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IQueueCommand *This);
      ULONG (WINAPI *Release)(IQueueCommand *This);
      HRESULT (WINAPI *InvokeAtStreamTime)(IQueueCommand *This,IDeferredCommand **pCmd,REFTIME time,GUID *iid,long dispidMethod,short wFlags,long cArgs,VARIANT *pDispParams,VARIANT *pvarResult,short *puArgErr);
      HRESULT (WINAPI *InvokeAtPresentationTime)(IQueueCommand *This,IDeferredCommand **pCmd,REFTIME time,GUID *iid,long dispidMethod,short wFlags,long cArgs,VARIANT *pDispParams,VARIANT *pvarResult,short *puArgErr);
    END_INTERFACE
  } IQueueCommandVtbl;
  struct IQueueCommand {
    CONST_VTBL struct IQueueCommandVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IQueueCommand_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IQueueCommand_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IQueueCommand_Release(This) (This)->lpVtbl->Release(This)
#define IQueueCommand_InvokeAtStreamTime(This,pCmd,time,iid,dispidMethod,wFlags,cArgs,pDispParams,pvarResult,puArgErr) (This)->lpVtbl->InvokeAtStreamTime(This,pCmd,time,iid,dispidMethod,wFlags,cArgs,pDispParams,pvarResult,puArgErr)
#define IQueueCommand_InvokeAtPresentationTime(This,pCmd,time,iid,dispidMethod,wFlags,cArgs,pDispParams,pvarResult,puArgErr) (This)->lpVtbl->InvokeAtPresentationTime(This,pCmd,time,iid,dispidMethod,wFlags,cArgs,pDispParams,pvarResult,puArgErr)
#endif
#endif
  HRESULT WINAPI IQueueCommand_InvokeAtStreamTime_Proxy(IQueueCommand *This,IDeferredCommand **pCmd,REFTIME time,GUID *iid,long dispidMethod,short wFlags,long cArgs,VARIANT *pDispParams,VARIANT *pvarResult,short *puArgErr);
  void __RPC_STUB IQueueCommand_InvokeAtStreamTime_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IQueueCommand_InvokeAtPresentationTime_Proxy(IQueueCommand *This,IDeferredCommand **pCmd,REFTIME time,GUID *iid,long dispidMethod,short wFlags,long cArgs,VARIANT *pDispParams,VARIANT *pvarResult,short *puArgErr);
  void __RPC_STUB IQueueCommand_InvokeAtPresentationTime_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

  DEFINE_GUID(CLSID_FilgraphManager,0xe436ebb3,0x524f,0x11ce,0x9f,0x53,0x00,0x20,0xaf,0x0b,0xa7,0x70);
#ifdef __cplusplus
  class FilgraphManager;
#endif

#ifndef __IFilterInfo_INTERFACE_DEFINED__
#define __IFilterInfo_INTERFACE_DEFINED__
  DEFINE_GUID(IID_IFilterInfo,0x56a868ba,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IFilterInfo : public IDispatch {
  public:
    virtual HRESULT WINAPI FindPin(BSTR strPinID,IDispatch **ppUnk) = 0;
    virtual HRESULT WINAPI get_Name(BSTR *strName) = 0;
    virtual HRESULT WINAPI get_VendorInfo(BSTR *strVendorInfo) = 0;
    virtual HRESULT WINAPI get_Filter(IUnknown **ppUnk) = 0;
    virtual HRESULT WINAPI get_Pins(IDispatch **ppUnk) = 0;
    virtual HRESULT WINAPI get_IsFileSource(LONG *pbIsSource) = 0;
    virtual HRESULT WINAPI get_Filename(BSTR *pstrFilename) = 0;
    virtual HRESULT WINAPI put_Filename(BSTR strFilename) = 0;
  };
#else
  typedef struct IFilterInfoVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IFilterInfo *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IFilterInfo *This);
      ULONG (WINAPI *Release)(IFilterInfo *This);
      HRESULT (WINAPI *GetTypeInfoCount)(IFilterInfo *This,UINT *pctinfo);
      HRESULT (WINAPI *GetTypeInfo)(IFilterInfo *This,UINT iTInfo,LCID lcid,ITypeInfo **ppTInfo);
      HRESULT (WINAPI *GetIDsOfNames)(IFilterInfo *This,REFIID riid,LPOLESTR *rgszNames,UINT cNames,LCID lcid,DISPID *rgDispId);
      HRESULT (WINAPI *Invoke)(IFilterInfo *This,DISPID dispIdMember,REFIID riid,LCID lcid,WORD wFlags,DISPPARAMS *pDispParams,VARIANT *pVarResult,EXCEPINFO *pExcepInfo,UINT *puArgErr);
      HRESULT (WINAPI *FindPin)(IFilterInfo *This,BSTR strPinID,IDispatch **ppUnk);
      HRESULT (WINAPI *get_Name)(IFilterInfo *This,BSTR *strName);
      HRESULT (WINAPI *get_VendorInfo)(IFilterInfo *This,BSTR *strVendorInfo);
      HRESULT (WINAPI *get_Filter)(IFilterInfo *This,IUnknown **ppUnk);
      HRESULT (WINAPI *get_Pins)(IFilterInfo *This,IDispatch **ppUnk);
      HRESULT (WINAPI *get_IsFileSource)(IFilterInfo *This,LONG *pbIsSource);
      HRESULT (WINAPI *get_Filename)(IFilterInfo *This,BSTR *pstrFilename);
      HRESULT (WINAPI *put_Filename)(IFilterInfo *This,BSTR strFilename);
    END_INTERFACE
  } IFilterInfoVtbl;
  struct IFilterInfo {
    CONST_VTBL struct IFilterInfoVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IFilterInfo_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IFilterInfo_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IFilterInfo_Release(This) (This)->lpVtbl->Release(This)
#define IFilterInfo_GetTypeInfoCount(This,pctinfo) (This)->lpVtbl->GetTypeInfoCount(This,pctinfo)
#define IFilterInfo_GetTypeInfo(This,iTInfo,lcid,ppTInfo) (This)->lpVtbl->GetTypeInfo(This,iTInfo,lcid,ppTInfo)
#define IFilterInfo_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) (This)->lpVtbl->GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)
#define IFilterInfo_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) (This)->lpVtbl->Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)
#define IFilterInfo_FindPin(This,strPinID,ppUnk) (This)->lpVtbl->FindPin(This,strPinID,ppUnk)
#define IFilterInfo_get_Name(This,strName) (This)->lpVtbl->get_Name(This,strName)
#define IFilterInfo_get_VendorInfo(This,strVendorInfo) (This)->lpVtbl->get_VendorInfo(This,strVendorInfo)
#define IFilterInfo_get_Filter(This,ppUnk) (This)->lpVtbl->get_Filter(This,ppUnk)
#define IFilterInfo_get_Pins(This,ppUnk) (This)->lpVtbl->get_Pins(This,ppUnk)
#define IFilterInfo_get_IsFileSource(This,pbIsSource) (This)->lpVtbl->get_IsFileSource(This,pbIsSource)
#define IFilterInfo_get_Filename(This,pstrFilename) (This)->lpVtbl->get_Filename(This,pstrFilename)
#define IFilterInfo_put_Filename(This,strFilename) (This)->lpVtbl->put_Filename(This,strFilename)
#endif
#endif
  HRESULT WINAPI IFilterInfo_FindPin_Proxy(IFilterInfo *This,BSTR strPinID,IDispatch **ppUnk);
  void __RPC_STUB IFilterInfo_FindPin_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterInfo_get_Name_Proxy(IFilterInfo *This,BSTR *strName);
  void __RPC_STUB IFilterInfo_get_Name_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterInfo_get_VendorInfo_Proxy(IFilterInfo *This,BSTR *strVendorInfo);
  void __RPC_STUB IFilterInfo_get_VendorInfo_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterInfo_get_Filter_Proxy(IFilterInfo *This,IUnknown **ppUnk);
  void __RPC_STUB IFilterInfo_get_Filter_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterInfo_get_Pins_Proxy(IFilterInfo *This,IDispatch **ppUnk);
  void __RPC_STUB IFilterInfo_get_Pins_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterInfo_get_IsFileSource_Proxy(IFilterInfo *This,LONG *pbIsSource);
  void __RPC_STUB IFilterInfo_get_IsFileSource_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterInfo_get_Filename_Proxy(IFilterInfo *This,BSTR *pstrFilename);
  void __RPC_STUB IFilterInfo_get_Filename_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IFilterInfo_put_Filename_Proxy(IFilterInfo *This,BSTR strFilename);
  void __RPC_STUB IFilterInfo_put_Filename_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IRegFilterInfo_INTERFACE_DEFINED__
#define __IRegFilterInfo_INTERFACE_DEFINED__
  DEFINE_GUID(IID_IRegFilterInfo,0x56a868bb,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IRegFilterInfo : public IDispatch {
  public:
    virtual HRESULT WINAPI get_Name(BSTR *strName) = 0;
    virtual HRESULT WINAPI Filter(IDispatch **ppUnk) = 0;
  };
#else
  typedef struct IRegFilterInfoVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IRegFilterInfo *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IRegFilterInfo *This);
      ULONG (WINAPI *Release)(IRegFilterInfo *This);
      HRESULT (WINAPI *GetTypeInfoCount)(IRegFilterInfo *This,UINT *pctinfo);
      HRESULT (WINAPI *GetTypeInfo)(IRegFilterInfo *This,UINT iTInfo,LCID lcid,ITypeInfo **ppTInfo);
      HRESULT (WINAPI *GetIDsOfNames)(IRegFilterInfo *This,REFIID riid,LPOLESTR *rgszNames,UINT cNames,LCID lcid,DISPID *rgDispId);
      HRESULT (WINAPI *Invoke)(IRegFilterInfo *This,DISPID dispIdMember,REFIID riid,LCID lcid,WORD wFlags,DISPPARAMS *pDispParams,VARIANT *pVarResult,EXCEPINFO *pExcepInfo,UINT *puArgErr);
      HRESULT (WINAPI *get_Name)(IRegFilterInfo *This,BSTR *strName);
      HRESULT (WINAPI *Filter)(IRegFilterInfo *This,IDispatch **ppUnk);
    END_INTERFACE
  } IRegFilterInfoVtbl;
  struct IRegFilterInfo {
    CONST_VTBL struct IRegFilterInfoVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IRegFilterInfo_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IRegFilterInfo_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IRegFilterInfo_Release(This) (This)->lpVtbl->Release(This)
#define IRegFilterInfo_GetTypeInfoCount(This,pctinfo) (This)->lpVtbl->GetTypeInfoCount(This,pctinfo)
#define IRegFilterInfo_GetTypeInfo(This,iTInfo,lcid,ppTInfo) (This)->lpVtbl->GetTypeInfo(This,iTInfo,lcid,ppTInfo)
#define IRegFilterInfo_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) (This)->lpVtbl->GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)
#define IRegFilterInfo_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) (This)->lpVtbl->Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)
#define IRegFilterInfo_get_Name(This,strName) (This)->lpVtbl->get_Name(This,strName)
#define IRegFilterInfo_Filter(This,ppUnk) (This)->lpVtbl->Filter(This,ppUnk)
#endif
#endif
  HRESULT WINAPI IRegFilterInfo_get_Name_Proxy(IRegFilterInfo *This,BSTR *strName);
  void __RPC_STUB IRegFilterInfo_get_Name_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IRegFilterInfo_Filter_Proxy(IRegFilterInfo *This,IDispatch **ppUnk);
  void __RPC_STUB IRegFilterInfo_Filter_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IMediaTypeInfo_INTERFACE_DEFINED__
#define __IMediaTypeInfo_INTERFACE_DEFINED__
  DEFINE_GUID(IID_IMediaTypeInfo,0x56a868bc,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IMediaTypeInfo : public IDispatch {
  public:
    virtual HRESULT WINAPI get_Type(BSTR *strType) = 0;
    virtual HRESULT WINAPI get_Subtype(BSTR *strType) = 0;
  };
#else
  typedef struct IMediaTypeInfoVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IMediaTypeInfo *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IMediaTypeInfo *This);
      ULONG (WINAPI *Release)(IMediaTypeInfo *This);
      HRESULT (WINAPI *GetTypeInfoCount)(IMediaTypeInfo *This,UINT *pctinfo);
      HRESULT (WINAPI *GetTypeInfo)(IMediaTypeInfo *This,UINT iTInfo,LCID lcid,ITypeInfo **ppTInfo);
      HRESULT (WINAPI *GetIDsOfNames)(IMediaTypeInfo *This,REFIID riid,LPOLESTR *rgszNames,UINT cNames,LCID lcid,DISPID *rgDispId);
      HRESULT (WINAPI *Invoke)(IMediaTypeInfo *This,DISPID dispIdMember,REFIID riid,LCID lcid,WORD wFlags,DISPPARAMS *pDispParams,VARIANT *pVarResult,EXCEPINFO *pExcepInfo,UINT *puArgErr);
      HRESULT (WINAPI *get_Type)(IMediaTypeInfo *This,BSTR *strType);
      HRESULT (WINAPI *get_Subtype)(IMediaTypeInfo *This,BSTR *strType);
    END_INTERFACE
  } IMediaTypeInfoVtbl;
  struct IMediaTypeInfo {
    CONST_VTBL struct IMediaTypeInfoVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IMediaTypeInfo_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IMediaTypeInfo_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IMediaTypeInfo_Release(This) (This)->lpVtbl->Release(This)
#define IMediaTypeInfo_GetTypeInfoCount(This,pctinfo) (This)->lpVtbl->GetTypeInfoCount(This,pctinfo)
#define IMediaTypeInfo_GetTypeInfo(This,iTInfo,lcid,ppTInfo) (This)->lpVtbl->GetTypeInfo(This,iTInfo,lcid,ppTInfo)
#define IMediaTypeInfo_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) (This)->lpVtbl->GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)
#define IMediaTypeInfo_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) (This)->lpVtbl->Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)
#define IMediaTypeInfo_get_Type(This,strType) (This)->lpVtbl->get_Type(This,strType)
#define IMediaTypeInfo_get_Subtype(This,strType) (This)->lpVtbl->get_Subtype(This,strType)
#endif
#endif
  HRESULT WINAPI IMediaTypeInfo_get_Type_Proxy(IMediaTypeInfo *This,BSTR *strType);
  void __RPC_STUB IMediaTypeInfo_get_Type_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IMediaTypeInfo_get_Subtype_Proxy(IMediaTypeInfo *This,BSTR *strType);
  void __RPC_STUB IMediaTypeInfo_get_Subtype_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IPinInfo_INTERFACE_DEFINED__
#define __IPinInfo_INTERFACE_DEFINED__
  DEFINE_GUID(IID_IPinInfo,0x56a868bd,0x0ad4,0x11ce,0xb0,0x3a,0x00,0x20,0xaf,0x0b,0xa7,0x70);
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IPinInfo : public IDispatch {
  public:
    virtual HRESULT WINAPI get_Pin(IUnknown **ppUnk) = 0;
    virtual HRESULT WINAPI get_ConnectedTo(IDispatch **ppUnk) = 0;
    virtual HRESULT WINAPI get_ConnectionMediaType(IDispatch **ppUnk) = 0;
    virtual HRESULT WINAPI get_FilterInfo(IDispatch **ppUnk) = 0;
    virtual HRESULT WINAPI get_Name(BSTR *ppUnk) = 0;
    virtual HRESULT WINAPI get_Direction(LONG *ppDirection) = 0;
    virtual HRESULT WINAPI get_PinID(BSTR *strPinID) = 0;
    virtual HRESULT WINAPI get_MediaTypes(IDispatch **ppUnk) = 0;
    virtual HRESULT WINAPI Connect(IUnknown *pPin) = 0;
    virtual HRESULT WINAPI ConnectDirect(IUnknown *pPin) = 0;
    virtual HRESULT WINAPI ConnectWithType(IUnknown *pPin,IDispatch *pMediaType) = 0;
    virtual HRESULT WINAPI Disconnect(void) = 0;
    virtual HRESULT WINAPI Render(void) = 0;
  };
#else
  typedef struct IPinInfoVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IPinInfo *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IPinInfo *This);
      ULONG (WINAPI *Release)(IPinInfo *This);
      HRESULT (WINAPI *GetTypeInfoCount)(IPinInfo *This,UINT *pctinfo);
      HRESULT (WINAPI *GetTypeInfo)(IPinInfo *This,UINT iTInfo,LCID lcid,ITypeInfo **ppTInfo);
      HRESULT (WINAPI *GetIDsOfNames)(IPinInfo *This,REFIID riid,LPOLESTR *rgszNames,UINT cNames,LCID lcid,DISPID *rgDispId);
      HRESULT (WINAPI *Invoke)(IPinInfo *This,DISPID dispIdMember,REFIID riid,LCID lcid,WORD wFlags,DISPPARAMS *pDispParams,VARIANT *pVarResult,EXCEPINFO *pExcepInfo,UINT *puArgErr);
      HRESULT (WINAPI *get_Pin)(IPinInfo *This,IUnknown **ppUnk);
      HRESULT (WINAPI *get_ConnectedTo)(IPinInfo *This,IDispatch **ppUnk);
      HRESULT (WINAPI *get_ConnectionMediaType)(IPinInfo *This,IDispatch **ppUnk);
      HRESULT (WINAPI *get_FilterInfo)(IPinInfo *This,IDispatch **ppUnk);
      HRESULT (WINAPI *get_Name)(IPinInfo *This,BSTR *ppUnk);
      HRESULT (WINAPI *get_Direction)(IPinInfo *This,LONG *ppDirection);
      HRESULT (WINAPI *get_PinID)(IPinInfo *This,BSTR *strPinID);
      HRESULT (WINAPI *get_MediaTypes)(IPinInfo *This,IDispatch **ppUnk);
      HRESULT (WINAPI *Connect)(IPinInfo *This,IUnknown *pPin);
      HRESULT (WINAPI *ConnectDirect)(IPinInfo *This,IUnknown *pPin);
      HRESULT (WINAPI *ConnectWithType)(IPinInfo *This,IUnknown *pPin,IDispatch *pMediaType);
      HRESULT (WINAPI *Disconnect)(IPinInfo *This);
      HRESULT (WINAPI *Render)(IPinInfo *This);
    END_INTERFACE
  } IPinInfoVtbl;
  struct IPinInfo {
    CONST_VTBL struct IPinInfoVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IPinInfo_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IPinInfo_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IPinInfo_Release(This) (This)->lpVtbl->Release(This)
#define IPinInfo_GetTypeInfoCount(This,pctinfo) (This)->lpVtbl->GetTypeInfoCount(This,pctinfo)
#define IPinInfo_GetTypeInfo(This,iTInfo,lcid,ppTInfo) (This)->lpVtbl->GetTypeInfo(This,iTInfo,lcid,ppTInfo)
#define IPinInfo_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) (This)->lpVtbl->GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)
#define IPinInfo_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) (This)->lpVtbl->Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)
#define IPinInfo_get_Pin(This,ppUnk) (This)->lpVtbl->get_Pin(This,ppUnk)
#define IPinInfo_get_ConnectedTo(This,ppUnk) (This)->lpVtbl->get_ConnectedTo(This,ppUnk)
#define IPinInfo_get_ConnectionMediaType(This,ppUnk) (This)->lpVtbl->get_ConnectionMediaType(This,ppUnk)
#define IPinInfo_get_FilterInfo(This,ppUnk) (This)->lpVtbl->get_FilterInfo(This,ppUnk)
#define IPinInfo_get_Name(This,ppUnk) (This)->lpVtbl->get_Name(This,ppUnk)
#define IPinInfo_get_Direction(This,ppDirection) (This)->lpVtbl->get_Direction(This,ppDirection)
#define IPinInfo_get_PinID(This,strPinID) (This)->lpVtbl->get_PinID(This,strPinID)
#define IPinInfo_get_MediaTypes(This,ppUnk) (This)->lpVtbl->get_MediaTypes(This,ppUnk)
#define IPinInfo_Connect(This,pPin) (This)->lpVtbl->Connect(This,pPin)
#define IPinInfo_ConnectDirect(This,pPin) (This)->lpVtbl->ConnectDirect(This,pPin)
#define IPinInfo_ConnectWithType(This,pPin,pMediaType) (This)->lpVtbl->ConnectWithType(This,pPin,pMediaType)
#define IPinInfo_Disconnect(This) (This)->lpVtbl->Disconnect(This)
#define IPinInfo_Render(This) (This)->lpVtbl->Render(This)
#endif
#endif
  HRESULT WINAPI IPinInfo_get_Pin_Proxy(IPinInfo *This,IUnknown **ppUnk);
  void __RPC_STUB IPinInfo_get_Pin_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPinInfo_get_ConnectedTo_Proxy(IPinInfo *This,IDispatch **ppUnk);
  void __RPC_STUB IPinInfo_get_ConnectedTo_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPinInfo_get_ConnectionMediaType_Proxy(IPinInfo *This,IDispatch **ppUnk);
  void __RPC_STUB IPinInfo_get_ConnectionMediaType_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPinInfo_get_FilterInfo_Proxy(IPinInfo *This,IDispatch **ppUnk);
  void __RPC_STUB IPinInfo_get_FilterInfo_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPinInfo_get_Name_Proxy(IPinInfo *This,BSTR *ppUnk);
  void __RPC_STUB IPinInfo_get_Name_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPinInfo_get_Direction_Proxy(IPinInfo *This,LONG *ppDirection);
  void __RPC_STUB IPinInfo_get_Direction_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPinInfo_get_PinID_Proxy(IPinInfo *This,BSTR *strPinID);
  void __RPC_STUB IPinInfo_get_PinID_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPinInfo_get_MediaTypes_Proxy(IPinInfo *This,IDispatch **ppUnk);
  void __RPC_STUB IPinInfo_get_MediaTypes_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPinInfo_Connect_Proxy(IPinInfo *This,IUnknown *pPin);
  void __RPC_STUB IPinInfo_Connect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPinInfo_ConnectDirect_Proxy(IPinInfo *This,IUnknown *pPin);
  void __RPC_STUB IPinInfo_ConnectDirect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPinInfo_ConnectWithType_Proxy(IPinInfo *This,IUnknown *pPin,IDispatch *pMediaType);
  void __RPC_STUB IPinInfo_ConnectWithType_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPinInfo_Disconnect_Proxy(IPinInfo *This);
  void __RPC_STUB IPinInfo_Disconnect_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IPinInfo_Render_Proxy(IPinInfo *This);
  void __RPC_STUB IPinInfo_Render_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif

#ifndef __IAMStats_INTERFACE_DEFINED__
#define __IAMStats_INTERFACE_DEFINED__
  DEFINE_GUID(IID_IAMStats,0xbc9bcf80,0xdcd2,0x11d2,0xab,0xf6,0x00,0xa0,0xc9,0x05,0xf3,0x75);
#if defined(__cplusplus) && !defined(CINTERFACE)
  struct IAMStats : public IDispatch {
  public:
    virtual HRESULT WINAPI Reset(void) = 0;
    virtual HRESULT WINAPI get_Count(LONG *plCount) = 0;
    virtual HRESULT WINAPI GetValueByIndex(long lIndex,BSTR *szName,long *lCount,double *dLast,double *dAverage,double *dStdDev,double *dMin,double *dMax) = 0;
    virtual HRESULT WINAPI GetValueByName(BSTR szName,long *lIndex,long *lCount,double *dLast,double *dAverage,double *dStdDev,double *dMin,double *dMax) = 0;
    virtual HRESULT WINAPI GetIndex(BSTR szName,long lCreate,long *plIndex) = 0;
    virtual HRESULT WINAPI AddValue(long lIndex,double dValue) = 0;
  };
#else
  typedef struct IAMStatsVtbl {
    BEGIN_INTERFACE
      HRESULT (WINAPI *QueryInterface)(IAMStats *This,REFIID riid,void **ppvObject);
      ULONG (WINAPI *AddRef)(IAMStats *This);
      ULONG (WINAPI *Release)(IAMStats *This);
      HRESULT (WINAPI *GetTypeInfoCount)(IAMStats *This,UINT *pctinfo);
      HRESULT (WINAPI *GetTypeInfo)(IAMStats *This,UINT iTInfo,LCID lcid,ITypeInfo **ppTInfo);
      HRESULT (WINAPI *GetIDsOfNames)(IAMStats *This,REFIID riid,LPOLESTR *rgszNames,UINT cNames,LCID lcid,DISPID *rgDispId);
      HRESULT (WINAPI *Invoke)(IAMStats *This,DISPID dispIdMember,REFIID riid,LCID lcid,WORD wFlags,DISPPARAMS *pDispParams,VARIANT *pVarResult,EXCEPINFO *pExcepInfo,UINT *puArgErr);
      HRESULT (WINAPI *Reset)(IAMStats *This);
      HRESULT (WINAPI *get_Count)(IAMStats *This,LONG *plCount);
      HRESULT (WINAPI *GetValueByIndex)(IAMStats *This,long lIndex,BSTR *szName,long *lCount,double *dLast,double *dAverage,double *dStdDev,double *dMin,double *dMax);
      HRESULT (WINAPI *GetValueByName)(IAMStats *This,BSTR szName,long *lIndex,long *lCount,double *dLast,double *dAverage,double *dStdDev,double *dMin,double *dMax);
      HRESULT (WINAPI *GetIndex)(IAMStats *This,BSTR szName,long lCreate,long *plIndex);
      HRESULT (WINAPI *AddValue)(IAMStats *This,long lIndex,double dValue);
    END_INTERFACE
  } IAMStatsVtbl;
  struct IAMStats {
    CONST_VTBL struct IAMStatsVtbl *lpVtbl;
  };
#ifdef COBJMACROS
#define IAMStats_QueryInterface(This,riid,ppvObject) (This)->lpVtbl->QueryInterface(This,riid,ppvObject)
#define IAMStats_AddRef(This) (This)->lpVtbl->AddRef(This)
#define IAMStats_Release(This) (This)->lpVtbl->Release(This)
#define IAMStats_GetTypeInfoCount(This,pctinfo) (This)->lpVtbl->GetTypeInfoCount(This,pctinfo)
#define IAMStats_GetTypeInfo(This,iTInfo,lcid,ppTInfo) (This)->lpVtbl->GetTypeInfo(This,iTInfo,lcid,ppTInfo)
#define IAMStats_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) (This)->lpVtbl->GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)
#define IAMStats_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) (This)->lpVtbl->Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)
#define IAMStats_Reset(This) (This)->lpVtbl->Reset(This)
#define IAMStats_get_Count(This,plCount) (This)->lpVtbl->get_Count(This,plCount)
#define IAMStats_GetValueByIndex(This,lIndex,szName,lCount,dLast,dAverage,dStdDev,dMin,dMax) (This)->lpVtbl->GetValueByIndex(This,lIndex,szName,lCount,dLast,dAverage,dStdDev,dMin,dMax)
#define IAMStats_GetValueByName(This,szName,lIndex,lCount,dLast,dAverage,dStdDev,dMin,dMax) (This)->lpVtbl->GetValueByName(This,szName,lIndex,lCount,dLast,dAverage,dStdDev,dMin,dMax)
#define IAMStats_GetIndex(This,szName,lCreate,plIndex) (This)->lpVtbl->GetIndex(This,szName,lCreate,plIndex)
#define IAMStats_AddValue(This,lIndex,dValue) (This)->lpVtbl->AddValue(This,lIndex,dValue)
#endif
#endif

  HRESULT WINAPI IAMStats_Reset_Proxy(IAMStats *This);
  void __RPC_STUB IAMStats_Reset_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMStats_get_Count_Proxy(IAMStats *This,LONG *plCount);
  void __RPC_STUB IAMStats_get_Count_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMStats_GetValueByIndex_Proxy(IAMStats *This,long lIndex,BSTR *szName,long *lCount,double *dLast,double *dAverage,double *dStdDev,double *dMin,double *dMax);
  void __RPC_STUB IAMStats_GetValueByIndex_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMStats_GetValueByName_Proxy(IAMStats *This,BSTR szName,long *lIndex,long *lCount,double *dLast,double *dAverage,double *dStdDev,double *dMin,double *dMax);
  void __RPC_STUB IAMStats_GetValueByName_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMStats_GetIndex_Proxy(IAMStats *This,BSTR szName,long lCreate,long *plIndex);
  void __RPC_STUB IAMStats_GetIndex_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
  HRESULT WINAPI IAMStats_AddValue_Proxy(IAMStats *This,long lIndex,double dValue);
  void __RPC_STUB IAMStats_AddValue_Stub(IRpcStubBuffer *This,IRpcChannelBuffer *_pRpcChannelBuffer,PRPC_MESSAGE _pRpcMessage,DWORD *_pdwStubPhase);
#endif
#endif

#ifdef __cplusplus
}
#endif
#endif
