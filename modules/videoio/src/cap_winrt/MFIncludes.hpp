// Header for standard system include files.

// Copyright (c) Microsoft. All rights reserved.
//
// The MIT License (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <collection.h>
#include <ppltasks.h>

#include <wrl\implements.h>
#include <wrl\wrappers\corewrappers.h>
#include <Roerrorapi.h>

#include <queue>
#include <sstream>

#include <robuffer.h>

#include <mfapi.h>
#include <mfidl.h>
#include <Mferror.h>

#include <windows.media.h>
#include <windows.media.mediaproperties.h>

namespace AWM = ::ABI::Windows::Media;
namespace AWMMp = ::ABI::Windows::Media::MediaProperties;
namespace AWFC = ::ABI::Windows::Foundation::Collections;
namespace MW = ::Microsoft::WRL;
namespace MWD = ::Microsoft::WRL::Details;
namespace MWW = ::Microsoft::WRL::Wrappers;
namespace WMC = ::Windows::Media::Capture;
namespace WF = ::Windows::Foundation;
namespace WMMp = ::Windows::Media::MediaProperties;
namespace WSS = ::Windows::Storage::Streams;

// Exception-based error handling
#define CHK(statement)  {HRESULT _hr = (statement); if (FAILED(_hr)) { throw ref new Platform::COMException(_hr); };}
#define CHKNULL(p)  {if ((p) == nullptr) { throw ref new Platform::NullReferenceException(L#p); };}

// Exception-free error handling
#define CHK_RETURN(statement) {hr = (statement); if (FAILED(hr)) { return hr; };}

// Cast a C++/CX msartpointer to an ABI smartpointer
template<typename T, typename U>
MW::ComPtr<T> As(U^ in)
{
    MW::ComPtr<T> out;
    CHK(reinterpret_cast<IInspectable*>(in)->QueryInterface(IID_PPV_ARGS(&out)));
    return out;
}

// Cast an ABI smartpointer
template<typename T, typename U>
Microsoft::WRL::ComPtr<T> As(const Microsoft::WRL::ComPtr<U>& in)
{
    Microsoft::WRL::ComPtr<T> out;
    CHK(in.As(&out));
    return out;
}

// Cast an ABI smartpointer
template<typename T, typename U>
Microsoft::WRL::ComPtr<T> As(U* in)
{
    Microsoft::WRL::ComPtr<T> out;
    CHK(in->QueryInterface(IID_PPV_ARGS(&out)));
    return out;
}

// Get access to bytes in IBuffer
inline unsigned char* GetData(_In_ WSS::IBuffer^ buffer)
{
    unsigned char* bytes = nullptr;
    CHK(As<WSS::IBufferByteAccess>(buffer)->Buffer(&bytes));
    return bytes;
}

// Class to start and shutdown Media Foundation
class AutoMF
{
public:
    AutoMF()
        : _bInitialized(false)
    {
        CHK(MFStartup(MF_VERSION));
    }

    ~AutoMF()
    {
        if (_bInitialized)
        {
            (void)MFShutdown();
        }
    }

private:
    bool _bInitialized;
};

// Class to track error origin
template <size_t N>
HRESULT OriginateError(__in HRESULT hr, __in wchar_t const (&str)[N])
{
    if (FAILED(hr))
    {
        ::RoOriginateErrorW(hr, N - 1, str);
    }
    return hr;
}

// Class to track error origin
inline HRESULT OriginateError(__in HRESULT hr)
{
    if (FAILED(hr))
    {
        ::RoOriginateErrorW(hr, 0, nullptr);
    }
    return hr;
}

// Converts exceptions into HRESULTs
template <typename Lambda>
HRESULT ExceptionBoundary(Lambda&& lambda)
{
    try
    {
        lambda();
        return S_OK;
    }
    catch (Platform::Exception^ e)
    {
        return e->HResult;
    }
    catch (const std::bad_alloc&)
    {
        return E_OUTOFMEMORY;
    }
    catch (const std::exception&)
    {
        return E_FAIL;
    }
}

// Wraps an IMFSample in a C++/CX class to be able to define a callback delegate
ref class MediaSample sealed
{
internal:
    MW::ComPtr<IMFSample> Sample;
};

delegate void MediaSampleHandler(MediaSample^ sample);