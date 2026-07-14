#pragma once

#include <winstring.h>
#include <stdio.h>
#include <tchar.h>
#include <crtdbg.h>
#include <array>
#include <vector>

#include <wrl\implements.h>
#include <wrl\event.h>
#include <inspectable.h>
#ifndef __cplusplus_winrt
#include <windows.foundation.h>

__declspec(noreturn) void __stdcall __abi_WinRTraiseException(long);

inline void __abi_ThrowIfFailed(long __hrArg)
{
  if (__hrArg < 0)
  {
    __abi_WinRTraiseException(__hrArg);
  }
}

struct Guid
{
public:
  Guid();
  Guid(__rcGUID_t);
  operator ::__rcGUID_t();
  bool Equals(Guid __guidArg);
  bool Equals(__rcGUID_t __guidArg);
  Guid(unsigned int __aArg, unsigned short __bArg, unsigned short __cArg, unsigned __int8 __dArg,
    unsigned __int8 __eArg, unsigned __int8 __fArg, unsigned __int8 __gArg, unsigned __int8 __hArg,
    unsigned __int8 __iArg, unsigned __int8 __jArg, unsigned __int8 __kArg);
  Guid(unsigned int __aArg, unsigned short __bArg, unsigned short __cArg, const unsigned __int8* __dArg);
private:
  unsigned long  __a;
  unsigned short __b;
  unsigned short __c;
  unsigned char __d;
  unsigned char __e;
  unsigned char __f;
  unsigned char __g;
  unsigned char __h;
  unsigned char __i;
  unsigned char __j;
  unsigned char __k;
};

static_assert(sizeof(Guid) == sizeof(::_GUID), "Incorrect size for Guid");
static_assert(sizeof(__rcGUID_t) == sizeof(::_GUID), "Incorrect size for __rcGUID_t");

////////////////////////////////////////////////////////////////////////////////
inline Guid::Guid() : __a(0), __b(0), __c(0), __d(0), __e(0), __f(0), __g(0), __h(0), __i(0), __j(0), __k(0)
{
}

inline Guid::Guid(__rcGUID_t __guid) :
  __a(reinterpret_cast<const __s_GUID&>(__guid).Data1),
  __b(reinterpret_cast<const __s_GUID&>(__guid).Data2),
  __c(reinterpret_cast<const __s_GUID&>(__guid).Data3),
  __d(reinterpret_cast<const __s_GUID&>(__guid).Data4[0]),
  __e(reinterpret_cast<const __s_GUID&>(__guid).Data4[1]),
  __f(reinterpret_cast<const __s_GUID&>(__guid).Data4[2]),
  __g(reinterpret_cast<const __s_GUID&>(__guid).Data4[3]),
  __h(reinterpret_cast<const __s_GUID&>(__guid).Data4[4]),
  __i(reinterpret_cast<const __s_GUID&>(__guid).Data4[5]),
  __j(reinterpret_cast<const __s_GUID&>(__guid).Data4[6]),
  __k(reinterpret_cast<const __s_GUID&>(__guid).Data4[7])
{
}

inline Guid::operator ::__rcGUID_t()
{
  return reinterpret_cast<__rcGUID_t>(*this);
}

inline bool Guid::Equals(Guid __guidArg)
{
  return *this == __guidArg;
}

inline bool Guid::Equals(__rcGUID_t __guidArg)
{
  return *this == static_cast< Guid>(__guidArg);
}

inline bool operator==(Guid __aArg, Guid __bArg)
{
  auto __a = reinterpret_cast<unsigned long*>(&__aArg);
  auto __b = reinterpret_cast<unsigned long*>(&__bArg);

  return (__a[0] == __b[0] && __a[1] == __b[1] && __a[2] == __b[2] && __a[3] == __b[3]);
}

inline bool operator!=(Guid __aArg, Guid __bArg)
{
  return !(__aArg == __bArg);
}

inline bool operator<(Guid __aArg, Guid __bArg)
{
  auto __a = reinterpret_cast<unsigned long*>(&__aArg);
  auto __b = reinterpret_cast<unsigned long*>(&__bArg);

  if (__a[0] != __b[0])
  {
    return __a[0] < __b[0];
  }

  if (__a[1] != __b[1])
  {
    return __a[1] < __b[1];
  }

  if (__a[2] != __b[2])
  {
    return __a[2] < __b[2];
  }

  if (__a[3] != __b[3])
  {
    return __a[3] < __b[3];
  }

  return false;
}

inline Guid::Guid(unsigned int __aArg, unsigned short __bArg, unsigned short __cArg, unsigned __int8 __dArg,
  unsigned __int8 __eArg, unsigned __int8 __fArg, unsigned __int8 __gArg, unsigned __int8 __hArg,
  unsigned __int8 __iArg, unsigned __int8 __jArg, unsigned __int8 __kArg) :
  __a(__aArg), __b(__bArg), __c(__cArg), __d(__dArg), __e(__eArg), __f(__fArg), __g(__gArg), __h(__hArg), __i(__iArg), __j(__jArg), __k(__kArg)
{
}

inline Guid::Guid(unsigned int __aArg, unsigned short __bArg, unsigned short __cArg, const unsigned __int8 __dArg[8]) :
  __a(__aArg), __b(__bArg), __c(__cArg)
{
  __d = __dArg[0];
  __e = __dArg[1];
  __f = __dArg[2];
  __g = __dArg[3];
  __h = __dArg[4];
  __i = __dArg[5];
  __j = __dArg[6];
  __k = __dArg[7];
}

__declspec(selectany) Guid __winrt_GUID_NULL(0x00000000, 0x0000, 0x0000, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);

//
//// Don't want to define the real IUnknown from unknown.h here. That would means if the user has
//// any broken code that uses it, compile errors will take the form of e.g.:
////     predefined C++ WinRT types (compiler internal)(41) : see declaration of 'IUnknown::QueryInterface'
//// This is not helpful. If they use IUnknown, we still need to point them to the actual unknown.h so
//// that they can see the original definition.
////
//// For WinRT, we'll instead have a parallel COM interface hierarchy for basic interfaces starting with _.
//// The type mismatch is not an issue. COM passes types through GUID / void* combos - the original type
//// doesn't come into play unless the user static_casts an implementation type to one of these, but
//// the WinRT implementation types are hidden.
__interface __declspec(uuid("00000000-0000-0000-C000-000000000046")) __abi_IUnknown
{
public:
  virtual long __stdcall __abi_QueryInterface(Guid&, void**) = 0;
  virtual unsigned long __stdcall __abi_AddRef() = 0;
  virtual unsigned long __stdcall __abi_Release() = 0;
};

__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseNotImplementedException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseInvalidCastException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseNullReferenceException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseOperationCanceledException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseFailureException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseAccessDeniedException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseOutOfMemoryException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseInvalidArgumentException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseOutOfBoundsException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseChangedStateException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseClassNotRegisteredException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseWrongThreadException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseDisconnectedException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseObjectDisposedException();
__declspec(dllexport) __declspec(noreturn) void __stdcall __abi_WinRTraiseCOMException(long);

__declspec(noreturn) inline void __stdcall __abi_WinRTraiseException(long __hrArg)
{
  switch (__hrArg)
  {
  case 0x80004001L: // E_NOTIMPL
    __abi_WinRTraiseNotImplementedException();

  case 0x80004002L: // E_NOINTERFACE
    __abi_WinRTraiseInvalidCastException();

  case 0x80004003L: // E_POINTER
    __abi_WinRTraiseNullReferenceException();

  case 0x80004004L: // E_ABORT
    __abi_WinRTraiseOperationCanceledException();

  case 0x80004005L: // E_FAIL
    __abi_WinRTraiseFailureException();

  case 0x80070005L: // E_ACCESSDENIED
    __abi_WinRTraiseAccessDeniedException();

  case 0x8007000EL: // E_OUTOFMEMORY
    __abi_WinRTraiseOutOfMemoryException();

  case 0x80070057L: // E_INVALIDARG
    __abi_WinRTraiseInvalidArgumentException();

  case 0x8000000BL: // E_BOUNDS
    __abi_WinRTraiseOutOfBoundsException();

  case 0x8000000CL: // E_CHANGED_STATE
    __abi_WinRTraiseChangedStateException();

  case 0x80040154L: // REGDB_E_CLASSNOTREG
    __abi_WinRTraiseClassNotRegisteredException();

  case 0x8001010EL: // RPC_E_WRONG_THREAD
    __abi_WinRTraiseWrongThreadException();

  case 0x80010108L: // RPC_E_DISCONNECTED
    __abi_WinRTraiseDisconnectedException();

  case 0x80000013L: // RO_E_CLOSED
    __abi_WinRTraiseObjectDisposedException();

  default:
    __abi_WinRTraiseCOMException(__hrArg);
    break;
  }
}

struct __abi_CaptureBase
{
protected:
  virtual __stdcall ~__abi_CaptureBase() {}

public:
  static const size_t __smallCaptureSize = 4 * sizeof(void*);
  void* operator new(size_t __sizeArg, void* __pSmallCaptureArg)
  {
    if (__sizeArg > __smallCaptureSize)
    {
      return reinterpret_cast<__abi_CaptureBase*>(HeapAlloc(GetProcessHeap(), 0, __sizeArg));
    }

  return __pSmallCaptureArg;
  }

    void operator delete(void* __ptrArg, void* __pSmallCaptureArg)
  {
    __abi_CaptureBase* __pThis = static_cast<__abi_CaptureBase*>(__ptrArg);
    __pThis->Delete(__pThis, __pSmallCaptureArg);
  }

  inline void* GetVFunction(int __slotArg)
  {
    return (*reinterpret_cast<void***>(this))[__slotArg];
  }

  void Delete(__abi_CaptureBase* __pThisArg, void* __pSmallCaptureArg)
  {
    __pThisArg->~__abi_CaptureBase();
    if (__pThisArg != __pSmallCaptureArg)
    {
      HeapFree(GetProcessHeap(), 0, __pThisArg);
    }
  }
};

struct __abi_CapturePtr
{
  char* smallCapture[__abi_CaptureBase::__smallCaptureSize];
  __abi_CaptureBase* ptr;
  __abi_CapturePtr() : ptr(reinterpret_cast<__abi_CaptureBase*>(smallCapture)) {}
  ~__abi_CapturePtr()
  {
    ptr->Delete(ptr, smallCapture);
  }
};

template <typename __TFunctor, typename __TReturnType>
struct __abi_FunctorCapture0 : public __abi_CaptureBase
{
  __TFunctor functor;
  __abi_FunctorCapture0(__TFunctor __functor) : functor(__functor) {}
  virtual __TReturnType __stdcall Invoke() { return functor(); }
};
template <typename __TFunctor, typename __TReturnType, typename __TArg0>
struct __abi_FunctorCapture1 : public __abi_CaptureBase
{
  __TFunctor functor;
  __abi_FunctorCapture1(__TFunctor __functor) : functor(__functor) {}
  virtual __TReturnType __stdcall Invoke(__TArg0 __arg0) { return functor(__arg0); }
};
template <typename __TFunctor, typename __TReturnType, typename __TArg0, typename __TArg1>
struct __abi_FunctorCapture2 : public __abi_CaptureBase
{
  __TFunctor functor;
  __abi_FunctorCapture2(__TFunctor __functor) : functor(__functor) {}
  virtual __TReturnType __stdcall Invoke(__TArg0 __arg0, __TArg1 __arg1) { return functor(__arg0, __arg1); }
};
template <typename __TFunctor, typename __TReturnType, typename __TArg0, typename __TArg1, typename __TArg2>
struct __abi_FunctorCapture3 : public __abi_CaptureBase
{
  __TFunctor functor;
  __abi_FunctorCapture3(__TFunctor __functor) : functor(__functor) {}
  virtual __TReturnType __stdcall Invoke(__TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2) { return functor(__arg0, __arg1, __arg2); }
};
template <typename __TFunctor, typename __TReturnType, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3>
struct __abi_FunctorCapture4 : public __abi_CaptureBase
{
  __TFunctor functor;
  __abi_FunctorCapture4(__TFunctor __functor) : functor(__functor) {}
  virtual __TReturnType __stdcall Invoke(__TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3) { return functor(__arg0, __arg1, __arg2, __arg3); }
};
template <typename __TFunctor, typename __TReturnType, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3, typename __TArg4>
struct __abi_FunctorCapture5 : public __abi_CaptureBase
{
  __TFunctor functor;
  __abi_FunctorCapture5(__TFunctor __functor) : functor(__functor) {}
  virtual __TReturnType __stdcall Invoke(__TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3, __TArg4 __arg4) { return functor(__arg0, __arg1, __arg2, __arg3, __arg4); }
};
template <typename __TFunctor, typename __TReturnType, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3, typename __TArg4, typename __TArg5>
struct __abi_FunctorCapture6 : public __abi_CaptureBase
{
  __TFunctor functor;
  __abi_FunctorCapture6(__TFunctor __functor) : functor(__functor) {}
  virtual __TReturnType __stdcall Invoke(__TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3, __TArg4 __arg4, __TArg5 __arg5) { return functor(__arg0, __arg1, __arg2, __arg3, __arg4, __arg5); }
};
template <typename __TFunctor, typename __TReturnType, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3, typename __TArg4, typename __TArg5, typename __TArg6>
struct __abi_FunctorCapture7 : public __abi_CaptureBase
{
  __TFunctor functor;
  __abi_FunctorCapture7(__TFunctor __functor) : functor(__functor) {}
  virtual __TReturnType __stdcall Invoke(__TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3, __TArg4 __arg4, __TArg5 __arg5, __TArg6 __arg6) { return functor(__arg0, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6); }
};
template <typename __TFunctor, typename __TReturnType, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3, typename __TArg4, typename __TArg5, typename __TArg6, typename __TArg7>
struct __abi_FunctorCapture8 : public __abi_CaptureBase
{
  __TFunctor functor;
  __abi_FunctorCapture8(__TFunctor __functor) : functor(__functor) {}
  virtual __TReturnType __stdcall Invoke(__TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3, __TArg4 __arg4, __TArg5 __arg5, __TArg6 __arg6, __TArg7 __arg7) { return functor(__arg0, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7); }
};
template <typename __TFunctor, typename __TReturnType, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3, typename __TArg4, typename __TArg5, typename __TArg6, typename __TArg7, typename __TArg8>
struct __abi_FunctorCapture9 : public __abi_CaptureBase
{
  __TFunctor functor;
  __abi_FunctorCapture9(__TFunctor __functor) : functor(__functor) {}
  virtual __TReturnType __stdcall Invoke(__TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3, __TArg4 __arg4, __TArg5 __arg5, __TArg6 __arg6, __TArg7 __arg7, __TArg8 __arg8) { return functor(__arg0, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8); }
};
template <typename __TFunctor, typename __TReturnType, typename __TArg0, typename __TArg1, typename __TArg2, typename __TArg3, typename __TArg4, typename __TArg5, typename __TArg6, typename __TArg7, typename __TArg8, typename __TArg9>
struct __abi_FunctorCapture10 : public __abi_CaptureBase
{
  __TFunctor functor;
  __abi_FunctorCapture10(__TFunctor __functor) : functor(__functor) {}
  virtual __TReturnType __stdcall Invoke(__TArg0 __arg0, __TArg1 __arg1, __TArg2 __arg2, __TArg3 __arg3, __TArg4 __arg4, __TArg5 __arg5, __TArg6 __arg6, __TArg7 __arg7, __TArg8 __arg8, __TArg9 __arg9) { return functor(__arg0, __arg1, __arg2, __arg3, __arg4, __arg5, __arg6, __arg7, __arg8, __arg9); }
};

#define __is_winrt_array(type) (type == ABI::Windows::Foundation::PropertyType::PropertyType_UInt8Array || type == ABI::Windows::Foundation::PropertyType::PropertyType_Int16Array ||\
  type == ABI::Windows::Foundation::PropertyType::PropertyType_UInt16Array || type == ABI::Windows::Foundation::PropertyType::PropertyType_Int32Array ||\
  type == ABI::Windows::Foundation::PropertyType::PropertyType_UInt32Array || type == ABI::Windows::Foundation::PropertyType::PropertyType_Int64Array ||\
  type == ABI::Windows::Foundation::PropertyType::PropertyType_UInt64Array || type == ABI::Windows::Foundation::PropertyType::PropertyType_SingleArray ||\
  type == ABI::Windows::Foundation::PropertyType::PropertyType_DoubleArray || type == ABI::Windows::Foundation::PropertyType::PropertyType_Char16Array ||\
  type == ABI::Windows::Foundation::PropertyType::PropertyType_BooleanArray || type == ABI::Windows::Foundation::PropertyType::PropertyType_StringArray ||\
  type == ABI::Windows::Foundation::PropertyType::PropertyType_InspectableArray || type == ABI::Windows::Foundation::PropertyType::PropertyType_DateTimeArray ||\
  type == ABI::Windows::Foundation::PropertyType::PropertyType_TimeSpanArray || type == ABI::Windows::Foundation::PropertyType::PropertyType_GuidArray ||\
  type == ABI::Windows::Foundation::PropertyType::PropertyType_PointArray || type == ABI::Windows::Foundation::PropertyType::PropertyType_SizeArray ||\
  type == ABI::Windows::Foundation::PropertyType::PropertyType_RectArray || type == ABI::Windows::Foundation::PropertyType::PropertyType_OtherTypeArray)

template<typename _Type, bool bUnknown = std::is_base_of<IUnknown, _Type>::value>
struct winrt_type
{
};
template<typename _Type>
struct winrt_type<_Type, true>
{
  static IUnknown* create(_Type* _ObjInCtx) {
    return reinterpret_cast<IUnknown*>(_ObjInCtx);
  }
  static IID getuuid() { return __uuidof(_Type); }
  static const ABI::Windows::Foundation::PropertyType _PropType = ABI::Windows::Foundation::PropertyType::PropertyType_OtherType;
};
template <typename _Type>
struct winrt_type<_Type, false>
{
  static IUnknown* create(_Type* _ObjInCtx) {
    Microsoft::WRL::ComPtr<IInspectable> _PObj;
    Microsoft::WRL::ComPtr<IActivationFactory> objFactory;
    HRESULT hr = Windows::Foundation::GetActivationFactory(Microsoft::WRL::Wrappers::HStringReference(RuntimeClass_Windows_Foundation_PropertyValue).Get(), objFactory.ReleaseAndGetAddressOf());
    if (FAILED(hr)) return nullptr;
    Microsoft::WRL::ComPtr<ABI::Windows::Foundation::IPropertyValueStatics> spPropVal;
    if (SUCCEEDED(hr))
      hr = objFactory.As(&spPropVal);
    if (SUCCEEDED(hr)) {
      hr = winrt_type<_Type>::create(spPropVal.Get(), _ObjInCtx, _PObj.GetAddressOf());
      if (SUCCEEDED(hr))
        return reinterpret_cast<IUnknown*>(_PObj.Detach());
    }
    return nullptr;
  }
  static IID getuuid() { return __uuidof(ABI::Windows::Foundation::IPropertyValue); }
  static const ABI::Windows::Foundation::PropertyType _PropType = ABI::Windows::Foundation::PropertyType::PropertyType_OtherType;
};

template<>
struct winrt_type<void>
{
  static HRESULT create(ABI::Windows::Foundation::IPropertyValueStatics* spPropVal, void* _ObjInCtx, IInspectable** ppInsp) {
    (void)_ObjInCtx;
    return spPropVal->CreateEmpty(ppInsp);
  }
  static const ABI::Windows::Foundation::PropertyType _PropType = ABI::Windows::Foundation::PropertyType::PropertyType_Empty;
};
#define MAKE_TYPE(Type, Name) template<>\
struct winrt_type<Type>\
{\
  static HRESULT create(ABI::Windows::Foundation::IPropertyValueStatics* spPropVal, Type* _ObjInCtx, IInspectable** ppInsp) {\
  return spPropVal->Create##Name(*_ObjInCtx, ppInsp);\
}\
  static const ABI::Windows::Foundation::PropertyType _PropType = ABI::Windows::Foundation::PropertyType::PropertyType_##Name;\
};

template<typename _Type>
struct winrt_array_type
{
  static IUnknown* create(_Type* _ObjInCtx, size_t N) {
    Microsoft::WRL::ComPtr<IInspectable> _PObj;
    Microsoft::WRL::ComPtr<IActivationFactory> objFactory;
    HRESULT hr = Windows::Foundation::GetActivationFactory(Microsoft::WRL::Wrappers::HStringReference(RuntimeClass_Windows_Foundation_PropertyValue).Get(), objFactory.ReleaseAndGetAddressOf());
    if (FAILED(hr)) return nullptr;
    Microsoft::WRL::ComPtr<ABI::Windows::Foundation::IPropertyValueStatics> spPropVal;
    if (SUCCEEDED(hr))
      hr = objFactory.As(&spPropVal);
    if (SUCCEEDED(hr)) {
      hr = winrt_array_type<_Type>::create(spPropVal.Get(), N, _ObjInCtx, _PObj.GetAddressOf());
      if (SUCCEEDED(hr))
        return reinterpret_cast<IUnknown*>(_PObj.Detach());
    }
    return nullptr;
  }
  static const ABI::Windows::Foundation::PropertyType _PropType = ABI::Windows::Foundation::PropertyType::PropertyType_OtherTypeArray;
};
template<int>
struct winrt_prop_type {};

template <>
struct winrt_prop_type<ABI::Windows::Foundation::PropertyType_Empty> {
  typedef void _Type;
};

template <>
struct winrt_prop_type<ABI::Windows::Foundation::PropertyType_OtherType> {
  typedef void _Type;
};

template <>
struct winrt_prop_type<ABI::Windows::Foundation::PropertyType_OtherTypeArray> {
  typedef void _Type;
};

#define MAKE_PROP(Prop, Type) template <>\
struct winrt_prop_type<ABI::Windows::Foundation::PropertyType_##Prop> {\
  typedef Type _Type;\
};

#define MAKE_ARRAY_TYPE(Type, Name) MAKE_PROP(Name, Type)\
  MAKE_PROP(Name##Array, Type*)\
  MAKE_TYPE(Type, Name)\
  template<>\
struct winrt_array_type<Type*>\
{\
  static HRESULT create(ABI::Windows::Foundation::IPropertyValueStatics* spPropVal, UINT32 __valueSize, Type** _ObjInCtx, IInspectable** ppInsp) {\
  return spPropVal->Create##Name##Array(__valueSize, *_ObjInCtx, ppInsp);\
}\
  static const ABI::Windows::Foundation::PropertyType _PropType = ABI::Windows::Foundation::PropertyType::PropertyType_##Name##Array;\
  static std::vector<Type> PropertyValueToVector(ABI::Windows::Foundation::IPropertyValue* propValue)\
{\
  UINT32 uLen = 0;\
  Type* pArray = nullptr;\
  propValue->Get##Name##Array(&uLen, &pArray);\
  return std::vector<Type>(pArray, pArray + uLen);\
}\
};
MAKE_ARRAY_TYPE(BYTE, UInt8)
MAKE_ARRAY_TYPE(INT16, Int16)
MAKE_ARRAY_TYPE(UINT16, UInt16)
MAKE_ARRAY_TYPE(INT32, Int32)
MAKE_ARRAY_TYPE(UINT32, UInt32)
MAKE_ARRAY_TYPE(INT64, Int64)
MAKE_ARRAY_TYPE(UINT64, UInt64)
MAKE_ARRAY_TYPE(FLOAT, Single)
MAKE_ARRAY_TYPE(DOUBLE, Double)
MAKE_ARRAY_TYPE(WCHAR, Char16)
//MAKE_ARRAY_TYPE(boolean, Boolean) //conflict with identical type in C++ of BYTE/UInt8
MAKE_ARRAY_TYPE(HSTRING, String)
MAKE_ARRAY_TYPE(IInspectable*, Inspectable)
MAKE_ARRAY_TYPE(GUID, Guid)
MAKE_ARRAY_TYPE(ABI::Windows::Foundation::DateTime, DateTime)
MAKE_ARRAY_TYPE(ABI::Windows::Foundation::TimeSpan, TimeSpan)
MAKE_ARRAY_TYPE(ABI::Windows::Foundation::Point, Point)
MAKE_ARRAY_TYPE(ABI::Windows::Foundation::Size, Size)
MAKE_ARRAY_TYPE(ABI::Windows::Foundation::Rect, Rect)

template < typename T >
struct DerefHelper
{
  typedef T DerefType;
};

template < typename T >
struct DerefHelper<T*>
{
  typedef T DerefType;
};

#define __is_valid_winrt_type(_Type) (std::is_void<_Type>::value || \
  std::is_same<_Type, BYTE>::value || \
  std::is_same<_Type, INT16>::value || \
  std::is_same<_Type, UINT16>::value || \
  std::is_same<_Type, INT32>::value || \
  std::is_same<_Type, UINT32>::value || \
  std::is_same<_Type, INT64>::value || \
  std::is_same<_Type, UINT64>::value || \
  std::is_same<_Type, FLOAT>::value || \
  std::is_same<_Type, DOUBLE>::value || \
  std::is_same<_Type, WCHAR>::value || \
  std::is_same<_Type, boolean>::value || \
  std::is_same<_Type, HSTRING>::value || \
  std::is_same<_Type, IInspectable *>::value || \
  std::is_base_of<Microsoft::WRL::Details::RuntimeClassBase, _Type>::value || \
  std::is_base_of<IInspectable, typename DerefHelper<_Type>::DerefType>::value || \
  std::is_same<_Type, GUID>::value || \
  std::is_same<_Type, ABI::Windows::Foundation::DateTime>::value || \
  std::is_same<_Type, ABI::Windows::Foundation::TimeSpan>::value || \
  std::is_same<_Type, ABI::Windows::Foundation::Point>::value || \
  std::is_same<_Type, ABI::Windows::Foundation::Size>::value || \
  std::is_same<_Type, ABI::Windows::Foundation::Rect>::value || \
  std::is_same<_Type, BYTE*>::value || \
  std::is_same<_Type, INT16*>::value || \
  std::is_same<_Type, UINT16*>::value || \
  std::is_same<_Type, INT32*>::value || \
  std::is_same<_Type, UINT32*>::value || \
  std::is_same<_Type, INT64*>::value || \
  std::is_same<_Type, UINT64*>::value || \
  std::is_same<_Type, FLOAT*>::value || \
  std::is_same<_Type, DOUBLE*>::value || \
  std::is_same<_Type, WCHAR*>::value || \
  std::is_same<_Type, boolean*>::value || \
  std::is_same<_Type, HSTRING*>::value || \
  std::is_same<_Type, IInspectable **>::value || \
  std::is_same<_Type, GUID*>::value || \
  std::is_same<_Type, ABI::Windows::Foundation::DateTime*>::value || \
  std::is_same<_Type, ABI::Windows::Foundation::TimeSpan*>::value || \
  std::is_same<_Type, ABI::Windows::Foundation::Point*>::value || \
  std::is_same<_Type, ABI::Windows::Foundation::Size*>::value || \
  std::is_same<_Type, ABI::Windows::Foundation::Rect*>::value)
#endif
