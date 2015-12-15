/***
* ==++==
*
* Copyright (c) Microsoft Corporation. All rights reserved.
*
* ==--==
* =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
*
* ppltasks.cpp
*
* =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
****/

#if _MSC_VER >= 1900
#include <pplinterface.h>
#include <windows.h>

#ifdef _CRT_APP
#include <wrl.h>
#include <ctxtcall.h>
#include <windows.foundation.diagnostics.h>
#include <ehdata.h>
#include <exception>
#include <trnsctrl.h>
#include <RestrictedErrorInfo.h>
#include <mutex>
#endif

#include <crtdbg.h>
#include <crtdefs.h>
#include <ppltasks_winrt.h>

// This IID is exported by ole32.dll; we cannot depend on ole32.dll on OneCore.
static GUID const Local_IID_ICallbackWithNoReentrancyToApplicationSTA =
{
    0x0A299774, 0x3E4E, 0xFC42, { 0x1D, 0x9D, 0x72, 0xCE, 0xE1, 0x05, 0xCA, 0x57 }
};

extern "C" void *__GetPlatformExceptionInfo(int *);

// Introduce stacktrace API for Debug CRT_APP
#if defined(_CRT_APP) && defined(_DEBUG)
extern "C" NTSYSAPI WORD NTAPI
RtlCaptureStackBackTrace (
    _In_ DWORD FramesToSkip,
    _In_ DWORD FramesToCapture,
    _Out_writes_to_(FramesToCapture, return) PVOID * BackTrace,
    _Out_opt_ PDWORD BackTraceHash
    );
#endif

namespace Concurrency_winrt
{

namespace details
{
void __cdecl _ReportUnobservedException()
{

#if (defined(_M_IX86) || defined(_M_X64)) && !(defined(_CRT_APP) || defined(_KERNELX))
    if (IsProcessorFeaturePresent(PF_FASTFAIL_AVAILABLE))
#endif
        __fastfail(FAST_FAIL_INVALID_ARG);

   std::terminate();
}

namespace platform
{

long __cdecl GetCurrentThreadId()
{
    return static_cast<long>(::GetCurrentThreadId());
}

/// <summary>
///     CRT CaptureStackBackTrace API wrapper
/// </summary>
size_t __cdecl CaptureCallstack(void **stackData, size_t skipFrames, size_t captureFrames)
{
    size_t capturedFrames = 0;
    // RtlCaptureSTackBackTrace is not available in MSDK, so we only call it under Desktop or _DEBUG MSDK.
    //  For MSDK unsupported version, we will return zero frame number.
#if !defined(_CRT_APP) || defined(_DEBUG)
    capturedFrames = RtlCaptureStackBackTrace(static_cast<DWORD>(skipFrames + 1), static_cast<DWORD>(captureFrames), stackData, nullptr);
#else
    (stackData); (skipFrames); (captureFrames);
#endif
    return capturedFrames;
}

static unsigned int s_asyncId = 0;

unsigned int __cdecl GetNextAsyncId()
{
    //
    // ASYNC TODO: Determine the requirements on the domain uniqueness of this value.  C++ / C# / WRL are all supposed to produce "unique" IDs and
    // there is no common broker.
    //
    return static_cast<unsigned int>(::_InterlockedIncrement(reinterpret_cast<volatile LONG *>(&s_asyncId)));
}

} // namespace Concurrency_winrt::details::platform

#if defined(_CRT_APP) && !defined(_KERNELX)
        using namespace ABI::Windows::Foundation;
        using namespace ABI::Windows::Foundation::Diagnostics;
        using namespace Microsoft::WRL;
        using namespace Microsoft::WRL::Wrappers;


        class AsyncCausalityTracer
        {
            IAsyncCausalityTracerStatics * m_causalityAPIs;
            std::once_flag m_stateFlag;
            bool m_isSupported;
        public:
            IAsyncCausalityTracerStatics * get() const
            {
                return m_causalityAPIs;
            }

            AsyncCausalityTracer() : m_causalityAPIs(nullptr), m_isSupported(false)
            {
            }

            void release()
            {
                if (m_causalityAPIs)
                {
                    APTTYPE aptType;
          APTTYPEQUALIFIER aptTypeQualifier;
          if (CoGetApartmentType(&aptType, &aptTypeQualifier) == S_OK)
          {
            // Release causality APIs only if current apartment is still RoInitialized
            m_causalityAPIs->Release();
            m_causalityAPIs = nullptr;
            m_isSupported = false;
          }
                }
            }

            bool isCausalitySupported()
            {
                std::call_once(m_stateFlag, [this] {
                    ComPtr<IAsyncCausalityTracerStatics> causalityAPIs;
                    if (SUCCEEDED(GetActivationFactory(HStringReference(RuntimeClass_Windows_Foundation_Diagnostics_AsyncCausalityTracer).Get(), &causalityAPIs)))
                    {
                        m_causalityAPIs = causalityAPIs.Detach();
                        m_isSupported = true;
                    }
                });
                return m_isSupported;
            }
        } asyncCausalityTracer;

        // GUID used for identifying causality logs from PPLTask
        const GUID PPLTaskCausalityPlatformID = { 0x7A76B220, 0xA758, 0x4E6E, 0xB0, 0xE0, 0xD7, 0xC6, 0xD7, 0x4A, 0x88, 0xFE };

        void __thiscall _TaskEventLogger::_LogScheduleTask(bool _isContinuation)
        {
            if (asyncCausalityTracer.isCausalitySupported())
            {
                asyncCausalityTracer.get()->TraceOperationCreation(CausalityTraceLevel_Required, CausalitySource_Library, PPLTaskCausalityPlatformID,
                    reinterpret_cast<unsigned long long>(_M_task),
                    HStringReference(_isContinuation ? L"Concurrency::PPLTask::ScheduleContinuationTask" : L"Concurrency::PPLTask::ScheduleTask").Get(), 0);
                _M_scheduled = true;
            }
        }
        void __thiscall _TaskEventLogger::_LogTaskCompleted()
        {
            if (_M_scheduled)
            {
                AsyncStatus status;
                if (_M_task->_IsCompleted())
                    status = AsyncStatus::Completed;
                else if (_M_task->_HasUserException())
                    status = AsyncStatus::Error;
                else
                    status = AsyncStatus::Canceled;

                if (asyncCausalityTracer.isCausalitySupported())
                {
                    asyncCausalityTracer.get()->TraceOperationCompletion(CausalityTraceLevel_Required, CausalitySource_Library, PPLTaskCausalityPlatformID,
                        reinterpret_cast<unsigned long long>(_M_task), status);
                }
            }
        }

        void __thiscall _TaskEventLogger::_LogCancelTask()
        {
            if (asyncCausalityTracer.isCausalitySupported())
            {
                asyncCausalityTracer.get()->TraceOperationRelation(CausalityTraceLevel_Important, CausalitySource_Library, PPLTaskCausalityPlatformID,
                    reinterpret_cast<unsigned long long>(_M_task), CausalityRelation_Cancel);
            }
        }

        void __thiscall _TaskEventLogger::_LogTaskExecutionCompleted()
        {
            if (asyncCausalityTracer.isCausalitySupported())
            {
                asyncCausalityTracer.get()->TraceSynchronousWorkCompletion(CausalityTraceLevel_Required, CausalitySource_Library,
                    CausalitySynchronousWork_CompletionNotification);
            }
        }

        void __thiscall _TaskEventLogger::_LogWorkItemStarted()
        {
            if (asyncCausalityTracer.isCausalitySupported())
            {
                asyncCausalityTracer.get()->TraceSynchronousWorkStart(CausalityTraceLevel_Required, CausalitySource_Library, PPLTaskCausalityPlatformID,
                    reinterpret_cast<unsigned long long>(_M_task), CausalitySynchronousWork_Execution );
            }
        }

        void __thiscall _TaskEventLogger::_LogWorkItemCompleted()
        {
            if (asyncCausalityTracer.isCausalitySupported())
            {
                asyncCausalityTracer.get()->TraceSynchronousWorkCompletion(CausalityTraceLevel_Required, CausalitySource_Library, CausalitySynchronousWork_Execution);

                asyncCausalityTracer.get()->TraceSynchronousWorkStart(CausalityTraceLevel_Required, CausalitySource_Library, PPLTaskCausalityPlatformID,
                    reinterpret_cast<unsigned long long>(_M_task), CausalitySynchronousWork_CompletionNotification);
                _M_taskPostEventStarted = true;
            }
        }

        static HRESULT __stdcall _PPLTaskContextCallbackBridge(ComCallData *_PParam)
        {
            auto pFunc = static_cast<std::function<void()> *>(_PParam->pUserDefined);
            (*pFunc)();
            return S_OK;
        }
        HRESULT __thiscall _ContextCallback::_CallInContext(_CallbackFunction _Func, bool _IgnoreDisconnect) const
        {
            if (!_HasCapturedContext())
            {
                _Func();
            }
            else
            {
                ComCallData callData;
                ZeroMemory(&callData, sizeof(callData));
                callData.pUserDefined = reinterpret_cast<void *>(&_Func);

                HRESULT hresult = static_cast<IContextCallback *>(_M_context._M_pContextCallback)->ContextCallback(&_PPLTaskContextCallbackBridge, &callData, Local_IID_ICallbackWithNoReentrancyToApplicationSTA, 5, nullptr);
                if (FAILED(hresult) && !(_IgnoreDisconnect && _IsHRCOMDisconnected(hresult)))
                {
                    throw std::runtime_error("Context callback failed.");
          return hresult;
                }
            }
      return S_OK;
        }
        void __thiscall _ContextCallback::_Capture()
        {
            HRESULT _Hr = CoGetObjectContext(__uuidof(IContextCallback), &_M_context._M_pContextCallback);
            if (FAILED(_Hr))
            {
                _M_context._M_pContextCallback = nullptr;
            }
        }

        void __thiscall _ContextCallback::_Reset()
        {
            if (_M_context._M_captureMethod != _S_captureDeferred && _M_context._M_pContextCallback != nullptr)
            {
                static_cast<IContextCallback *>(_M_context._M_pContextCallback)->Release();
            }
        }

        void __thiscall _ContextCallback::_Assign(void *_PContextCallback)
        {
            _M_context._M_pContextCallback = _PContextCallback;
            if (_M_context._M_captureMethod != _S_captureDeferred && _M_context._M_pContextCallback != nullptr)
            {
                static_cast<IContextCallback*>(_M_context._M_pContextCallback)->AddRef();
            }
        }

        bool __cdecl _ContextCallback::_IsCurrentOriginSTA()
        {
            APTTYPE _AptType;
            APTTYPEQUALIFIER _AptTypeQualifier;

            HRESULT hr = CoGetApartmentType(&_AptType, &_AptTypeQualifier);
            if (SUCCEEDED(hr))
            {
                // We determine the origin of a task continuation by looking at where .then is called, so we can tell whether
                // to need to marshal the continuation back to the originating apartment. If an STA thread is in executing in
                // a neutral aparment when it schedules a continuation, we will not marshal continuations back to the STA,
                // since variables used within a neutral apartment are expected to be apartment neutral.
                switch (_AptType)
                {
                case APTTYPE_MAINSTA:
                case APTTYPE_STA:
                    return true;
                default:
                    break;
                }
            }
            return false;
        }



        void __thiscall _ExceptionHolder::ReportUnhandledError()
        {
            if (_M_stdException)
            {
                try
                {
                    std::rethrow_exception(_M_stdException);
                }
                catch (...)
                {
                    int isBadAlloc;
                    WINRTEXCEPTIONINFO * info = static_cast<WINRTEXCEPTIONINFO*>(__GetPlatformExceptionInfo(&isBadAlloc));
                    // If it is a winrt exception
                    if (!isBadAlloc && info != nullptr)
                    {
                        IRestrictedErrorInfo *restrictedInfo(static_cast<IRestrictedErrorInfo*>(info->restrictedInfo));
                        if (restrictedInfo != nullptr)
                        {
                            #ifndef _M_ARM64 // CRT_REFACTOR TODO
                            ::RoReportUnhandledError(restrictedInfo);
                            #endif
                            // IRestrictedErrorInfo should be put back
                            // to make sure that RoFailFastWithErrorContext can extract the information
                            ::SetRestrictedErrorInfo(restrictedInfo);
                        }
                        ::RoFailFastWithErrorContext(info->hr);
                    }
                }
            }
        }

        bool __cdecl _Task_impl_base::_IsNonBlockingThread()
        {
            APTTYPE _AptType;
            APTTYPEQUALIFIER _AptTypeQualifier;

            HRESULT hr = CoGetApartmentType(&_AptType, &_AptTypeQualifier);
            //
            // If it failed, it's not a Windows Runtime/COM initialized thread. This is not a failure.
            //
            if (SUCCEEDED(hr))
            {
                switch (_AptType)
                {
                case APTTYPE_STA:
                case APTTYPE_MAINSTA:
                    return true;
                    break;
                case APTTYPE_NA:
                    switch (_AptTypeQualifier)
                    {
                        // A thread executing in a neutral apartment is either STA or MTA. To find out if this thread is allowed
                        // to wait, we check the app qualifier. If it is an STA thread executing in a neutral apartment, waiting
                        // is illegal, because the thread is responsible for pumping messages and waiting on a task could take the
                        // thread out of circulation for a while.
                    case APTTYPEQUALIFIER_NA_ON_STA:
                    case APTTYPEQUALIFIER_NA_ON_MAINSTA:
                        return true;
                        break;
                    }
                    break;
                }
            }
            return false;
        }
#else
        void __thiscall _TaskEventLogger::_LogScheduleTask(bool)
        {
        }
        void __thiscall _TaskEventLogger::_LogTaskCompleted()
        {
        }

        void __thiscall _TaskEventLogger::_LogCancelTask()
        {
        }

        void __thiscall _TaskEventLogger::_LogTaskExecutionCompleted()
        {
        }

        void __thiscall _TaskEventLogger::_LogWorkItemStarted()
        {
        }

        void __thiscall _TaskEventLogger::_LogWorkItemCompleted()
        {
        }

        HRESULT __thiscall _ContextCallback::_CallInContext(_CallbackFunction _Func, bool) const
        {
            _Func();
      return S_OK;
        }
        void __thiscall _ContextCallback::_Capture()
        {
        }

        void __thiscall _ContextCallback::_Reset()
        {
        }

        void __thiscall _ContextCallback::_Assign(void *)
        {
        }

        bool __cdecl _ContextCallback::_IsCurrentOriginSTA()
        {
            return false;
        }


        void __thiscall _ExceptionHolder::ReportUnhandledError()
        {
        }

        bool __cdecl _Task_impl_base::_IsNonBlockingThread()
        {
            return false;
        }
#endif
    } // namespace Concurrency_winrt::details

#ifdef _CRT_APP
    __thiscall task_continuation_context::task_continuation_context() : _ContextCallback(true), _M_RunInline(false)
    {
    }
#else
    __thiscall task_continuation_context::task_continuation_context() : _ContextCallback(false), _M_RunInline(false)
    {
    }
#endif
} // namespace Concurrency_winrt

#ifdef _CRT_APP
extern "C" void __cdecl __crtCleanupCausalityStaticFactories()
{
    Concurrency::details::asyncCausalityTracer.release();
}
#endif

#endif
