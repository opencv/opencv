//
// Copyright (C) Microsoft Corporation
// All rights reserved.
// Modified for native C++ WRL support by Gregory Morse
//
// Code in Details namespace is for internal usage within the library code
//

#ifndef _PLATFORM_AGILE_H_
#define _PLATFORM_AGILE_H_

#ifdef _MSC_VER
#pragma once
#endif  // _MSC_VER

#include <algorithm>
#include <wrl\client.h>

template <typename T, bool TIsNotAgile> class Agile;

template <typename T>
struct UnwrapAgile
{
    static const bool _IsAgile = false;
};
template <typename T>
struct UnwrapAgile<Agile<T, false>>
{
    static const bool _IsAgile = true;
};
template <typename T>
struct UnwrapAgile<Agile<T, true>>
{
    static const bool _IsAgile = true;
};

#define IS_AGILE(T) UnwrapAgile<T>::_IsAgile

#define __is_winrt_agile(T) (std::is_same<T, HSTRING__>::value || std::is_base_of<Microsoft::WRL::FtmBase, T>::value || std::is_base_of<IAgileObject, T>::value) //derived from Microsoft::WRL::FtmBase or IAgileObject

#define __is_win_interface(T) (std::is_base_of<IUnknown, T>::value || std::is_base_of<IInspectable, T>::value) //derived from IUnknown or IInspectable

#define __is_win_class(T) (std::is_same<T, HSTRING__>::value || std::is_base_of<Microsoft::WRL::Details::RuntimeClassBase, T>::value) //derived from Microsoft::WRL::RuntimeClass or HSTRING

    namespace Details
    {
        IUnknown* __stdcall GetObjectContext();
        HRESULT __stdcall GetProxyImpl(IUnknown*, REFIID, IUnknown*, IUnknown**);
        HRESULT __stdcall ReleaseInContextImpl(IUnknown*, IUnknown*);

        template <typename T>
#if _MSC_VER >= 1800
        __declspec(no_refcount) inline HRESULT GetProxy(T *ObjectIn, IUnknown *ContextCallBack, T **Proxy)
#else
        inline HRESULT GetProxy(T *ObjectIn, IUnknown *ContextCallBack, T **Proxy)
#endif
        {
#if _MSC_VER >= 1800
            return GetProxyImpl(*reinterpret_cast<IUnknown**>(&ObjectIn), __uuidof(T*), ContextCallBack, reinterpret_cast<IUnknown**>(Proxy));
#else
            return GetProxyImpl(*reinterpret_cast<IUnknown**>(&const_cast<T*>(ObjectIn)), __uuidof(T*), ContextCallBack, reinterpret_cast<IUnknown**>(Proxy));
#endif
        }

        template <typename T>
        inline HRESULT ReleaseInContext(T *ObjectIn, IUnknown *ContextCallBack)
        {
            return ReleaseInContextImpl(ObjectIn, ContextCallBack);
        }

        template <typename T>
        class AgileHelper
        {
            __abi_IUnknown* _p;
            bool _release;
        public:
            AgileHelper(__abi_IUnknown* p, bool release = true) : _p(p), _release(release)
            {
            }
            AgileHelper(AgileHelper&& other) : _p(other._p), _release(other._release)
            {
                _other._p = nullptr;
                _other._release = true;
            }
            AgileHelper operator=(AgileHelper&& other)
            {
                _p = other._p;
                _release = other._release;
                _other._p = nullptr;
                _other._release = true;
                return *this;
            }

            ~AgileHelper()
            {
                if (_release && _p)
                {
                    _p->__abi_Release();
                }
            }

            __declspec(no_refcount) __declspec(no_release_return)
                T* operator->()
            {
                    return reinterpret_cast<T*>(_p);
            }

            __declspec(no_refcount) __declspec(no_release_return)
                operator T * ()
            {
                    return reinterpret_cast<T*>(_p);
            }
        private:
            AgileHelper(const AgileHelper&);
            AgileHelper operator=(const AgileHelper&);
        };
        template <typename T>
        struct __remove_hat
        {
            typedef T type;
        };
        template <typename T>
        struct __remove_hat<T*>
        {
            typedef T type;
        };
        template <typename T>
        struct AgileTypeHelper
        {
            typename typedef __remove_hat<T>::type type;
            typename typedef __remove_hat<T>::type* agileMemberType;
        };
    } // namespace Details

#pragma warning(push)
#pragma warning(disable: 4451) // Usage of ref class inside this context can lead to invalid marshaling of object across contexts

    template <
        typename T,
        bool TIsNotAgile = (__is_win_class(typename Details::AgileTypeHelper<T>::type) && !__is_winrt_agile(typename Details::AgileTypeHelper<T>::type)) ||
        __is_win_interface(typename Details::AgileTypeHelper<T>::type)
    >
    class Agile
    {
        static_assert(__is_win_class(typename Details::AgileTypeHelper<T>::type) || __is_win_interface(typename Details::AgileTypeHelper<T>::type), "Agile can only be used with ref class or interface class types");
        typename typedef Details::AgileTypeHelper<T>::agileMemberType TypeT;
        TypeT _object;
        ::Microsoft::WRL::ComPtr<IUnknown> _contextCallback;
        ULONG_PTR _contextToken;

#if _MSC_VER >= 1800
        enum class AgileState
        {
            NonAgilePointer = 0,
            AgilePointer = 1,
            Unknown = 2
        };
        AgileState _agileState;
#endif

        void CaptureContext()
        {
            _contextCallback = Details::GetObjectContext();
            __abi_ThrowIfFailed(CoGetContextToken(&_contextToken));
        }

        void SetObject(TypeT object)
        {
            // Capture context before setting the pointer
            // If context capture fails then nothing to cleanup
            Release();
            if (object != nullptr)
            {
                ::Microsoft::WRL::ComPtr<IAgileObject> checkIfAgile;
                HRESULT hr = reinterpret_cast<IUnknown*>(object)->QueryInterface(__uuidof(IAgileObject), &checkIfAgile);
                // Don't Capture context if object is agile
                if (hr != S_OK)
                {
#if _MSC_VER >= 1800
                    _agileState = AgileState::NonAgilePointer;
#endif
                    CaptureContext();
                }
#if _MSC_VER >= 1800
                else
                {
                    _agileState = AgileState::AgilePointer;
                }
#endif
            }
            _object = object;
        }

    public:
        Agile() throw() : _object(nullptr), _contextToken(0)
#if _MSC_VER >= 1800
            , _agileState(AgileState::Unknown)
#endif
        {
        }

        Agile(nullptr_t) throw() : _object(nullptr), _contextToken(0)
#if _MSC_VER >= 1800
            , _agileState(AgileState::Unknown)
#endif
        {
        }

        explicit Agile(TypeT object) throw() : _object(nullptr), _contextToken(0)
#if _MSC_VER >= 1800
            , _agileState(AgileState::Unknown)
#endif
        {
            // Assumes that the source object is from the current context
            SetObject(object);
        }

        Agile(const Agile& object) throw() : _object(nullptr), _contextToken(0)
#if _MSC_VER >= 1800
            , _agileState(AgileState::Unknown)
#endif
        {
            // Get returns pointer valid for current context
            SetObject(object.Get());
        }

        Agile(Agile&& object) throw() : _object(nullptr), _contextToken(0)
#if _MSC_VER >= 1800
            , _agileState(AgileState::Unknown)
#endif
        {
            // Assumes that the source object is from the current context
            Swap(object);
        }

        ~Agile() throw()
        {
            Release();
        }

        TypeT Get() const
        {
            // Agile object, no proxy required
#if _MSC_VER >= 1800
            if (_agileState == AgileState::AgilePointer || _object == nullptr)
#else
            if (_contextToken == 0 || _contextCallback == nullptr || _object == nullptr)
#endif
            {
                return _object;
            }

            // Do the check for same context
            ULONG_PTR currentContextToken;
            __abi_ThrowIfFailed(CoGetContextToken(&currentContextToken));
            if (currentContextToken == _contextToken)
            {
                return _object;
            }

#if _MSC_VER >= 1800
            // Different context and holding on to a non agile object
            // Do the costly work of getting a proxy
            TypeT localObject;
            __abi_ThrowIfFailed(Details::GetProxy(_object, _contextCallback.Get(), &localObject));

            if (_agileState == AgileState::Unknown)
#else
            // Object is agile if it implements IAgileObject
            // GetAddressOf captures the context with out knowing the type of object that it will hold
            if (_object != nullptr)
#endif
            {
#if _MSC_VER >= 1800
                // Object is agile if it implements IAgileObject
                // GetAddressOf captures the context with out knowing the type of object that it will hold
                ::Microsoft::WRL::ComPtr<IAgileObject> checkIfAgile;
                HRESULT hr = reinterpret_cast<IUnknown*>(localObject)->QueryInterface(__uuidof(IAgileObject), &checkIfAgile);
#else
                ::Microsoft::WRL::ComPtr<IAgileObject> checkIfAgile;
                HRESULT hr = reinterpret_cast<IUnknown*>(_object)->QueryInterface(__uuidof(IAgileObject), &checkIfAgile);
#endif
                if (hr == S_OK)
                {
                    auto pThis = const_cast<Agile*>(this);
#if _MSC_VER >= 1800
                    pThis->_agileState = AgileState::AgilePointer;
#endif
                    pThis->_contextToken = 0;
                    pThis->_contextCallback = nullptr;
                    return _object;
                }
#if _MSC_VER >= 1800
                else
                {
                    auto pThis = const_cast<Agile*>(this);
                    pThis->_agileState = AgileState::NonAgilePointer;
                }
#endif
            }

#if _MSC_VER < 1800
            // Different context and holding on to a non agile object
            // Do the costly work of getting a proxy
            TypeT localObject;
            __abi_ThrowIfFailed(Details::GetProxy(_object, _contextCallback.Get(), &localObject));
#endif
            return localObject;
        }

        TypeT* GetAddressOf() throw()
        {
            Release();
            CaptureContext();
            return &_object;
        }

        TypeT* GetAddressOfForInOut() throw()
        {
            CaptureContext();
            return &_object;
        }

        TypeT operator->() const throw()
        {
            return Get();
        }

        Agile& operator=(nullptr_t) throw()
        {
            Release();
            return *this;
        }

        Agile& operator=(TypeT object) throw()
        {
            Agile(object).Swap(*this);
            return *this;
        }

        Agile& operator=(Agile object) throw()
        {
            // parameter is by copy which gets pointer valid for current context
            object.Swap(*this);
            return *this;
        }

#if _MSC_VER < 1800
        Agile& operator=(IUnknown* lp) throw()
        {
            // bump ref count
            ::Microsoft::WRL::ComPtr<IUnknown> spObject(lp);

            // put it into Platform Object
            Platform::Object object;
            *(IUnknown**)(&object) = spObject.Detach();

            SetObject(object);
            return *this;
        }
#endif

        void Swap(Agile& object)
        {
            std::swap(_object, object._object);
            std::swap(_contextCallback, object._contextCallback);
            std::swap(_contextToken, object._contextToken);
#if _MSC_VER >= 1800
            std::swap(_agileState, object._agileState);
#endif
        }

        // Release the interface and set to NULL
        void Release() throw()
        {
            if (_object)
            {
                // Cast to IInspectable (no QI)
                IUnknown* pObject = *(IUnknown**)(&_object);
                // Set * to null without release
                *(IUnknown**)(&_object) = nullptr;

                ULONG_PTR currentContextToken;
                __abi_ThrowIfFailed(CoGetContextToken(&currentContextToken));
                if (_contextToken == 0 || _contextCallback == nullptr || _contextToken == currentContextToken)
                {
                    pObject->Release();
                }
                else
                {
                    Details::ReleaseInContext(pObject, _contextCallback.Get());
                }
                _contextCallback = nullptr;
                _contextToken = 0;
#if _MSC_VER >= 1800
                _agileState = AgileState::Unknown;
#endif
            }
        }

        bool operator==(nullptr_t) const throw()
        {
            return _object == nullptr;
        }

        bool operator==(const Agile& other) const throw()
        {
            return _object == other._object && _contextToken == other._contextToken;
        }

        bool operator<(const Agile& other) const throw()
        {
            if (reinterpret_cast<void*>(_object) < reinterpret_cast<void*>(other._object))
            {
                return true;
            }

            return _object == other._object && _contextToken < other._contextToken;
        }
    };

    template <typename T>
    class Agile<T, false>
    {
        static_assert(__is_win_class(typename Details::AgileTypeHelper<T>::type) || __is_win_interface(typename Details::AgileTypeHelper<T>::type), "Agile can only be used with ref class or interface class types");
        typename typedef Details::AgileTypeHelper<T>::agileMemberType TypeT;
        TypeT _object;

    public:
        Agile() throw() : _object(nullptr)
        {
        }

        Agile(nullptr_t) throw() : _object(nullptr)
        {
        }

        explicit Agile(TypeT object) throw() : _object(object)
        {
        }

        Agile(const Agile& object) throw() : _object(object._object)
        {
        }

        Agile(Agile&& object) throw() : _object(nullptr)
        {
            Swap(object);
        }

        ~Agile() throw()
        {
            Release();
        }

        TypeT Get() const
        {
            return _object;
        }

        TypeT* GetAddressOf() throw()
        {
            Release();
            return &_object;
        }

        TypeT* GetAddressOfForInOut() throw()
        {
            return &_object;
        }

        TypeT operator->() const throw()
        {
            return Get();
        }

        Agile& operator=(nullptr_t) throw()
        {
            Release();
            return *this;
        }

        Agile& operator=(TypeT object) throw()
        {
            if (_object != object)
            {
                _object = object;
            }
            return *this;
        }

        Agile& operator=(Agile object) throw()
        {
            object.Swap(*this);
            return *this;
        }

#if _MSC_VER < 1800
        Agile& operator=(IUnknown* lp) throw()
        {
            Release();
            // bump ref count
            ::Microsoft::WRL::ComPtr<IUnknown> spObject(lp);

            // put it into Platform Object
            Platform::Object object;
            *(IUnknown**)(&object) = spObject.Detach();

            _object = object;
            return *this;
        }
#endif

        // Release the interface and set to NULL
        void Release() throw()
        {
            _object = nullptr;
        }

        void Swap(Agile& object)
        {
            std::swap(_object, object._object);
        }

        bool operator==(nullptr_t) const throw()
        {
            return _object == nullptr;
        }

        bool operator==(const Agile& other) const throw()
        {
            return _object == other._object;
        }

        bool operator<(const Agile& other) const throw()
        {
            return reinterpret_cast<void*>(_object) < reinterpret_cast<void*>(other._object);
        }
    };

#pragma warning(pop)

    template<class U>
    bool operator==(nullptr_t, const Agile<U>& a) throw()
    {
        return a == nullptr;
    }

    template<class U>
    bool operator!=(const Agile<U>& a, nullptr_t) throw()
    {
        return !(a == nullptr);
    }

    template<class U>
    bool operator!=(nullptr_t, const Agile<U>& a) throw()
    {
        return !(a == nullptr);
    }

    template<class U>
    bool operator!=(const Agile<U>& a, const Agile<U>& b) throw()
    {
        return !(a == b);
    }


#endif // _PLATFORM_AGILE_H_
