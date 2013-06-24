// Defines the transform class.
//
// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
// PARTICULAR PURPOSE.
//
// Copyright (c) Microsoft Corporation. All rights reserved.

#ifndef GRAYSCALE_H
#define GRAYSCALE_H

#include <new>
#include <mfapi.h>
#include <mftransform.h>
#include <mfidl.h>
#include <mferror.h>
#include <strsafe.h>
#include <assert.h>

// Note: The Direct2D helper library is included for its 2D matrix operations.
#include <D2d1helper.h>

#include <wrl\implements.h>
#include <wrl\module.h>
#include <windows.media.h>

#include "GrayscaleTransform.h"

// CLSID of the MFT.
DEFINE_GUID(CLSID_GrayscaleMFT,
0x2f3dbc05, 0xc011, 0x4a8f, 0xb2, 0x64, 0xe4, 0x2e, 0x35, 0xc6, 0x7b, 0xf4);

//
// * IMPORTANT: If you implement your own MFT, create a new GUID for the CLSID. *
//


// Configuration attributes

// {7BBBB051-133B-41F5-B6AA-5AFF9B33A2CB}
DEFINE_GUID(MFT_GRAYSCALE_DESTINATION_RECT, 
0x7bbbb051, 0x133b, 0x41f5, 0xb6, 0xaa, 0x5a, 0xff, 0x9b, 0x33, 0xa2, 0xcb);

enum ProcessingType
{
	Preview,
	GrayScale,
	Canny,
	Zoom,
	Sepia
};

template <class T> void SafeRelease(T **ppT)
{
    if (*ppT)
    {
        (*ppT)->Release();
        *ppT = NULL;
    }
}

// CGrayscale class:
// Implements a grayscale video effect.

class CGrayscale 
    : public Microsoft::WRL::RuntimeClass<
           Microsoft::WRL::RuntimeClassFlags< Microsoft::WRL::RuntimeClassType::WinRtClassicComMix >, 
           ABI::Windows::Media::IMediaExtension,
           IMFTransform >
{
    InspectableClass(RuntimeClass_GrayscaleTransform_GrayscaleEffect, BaseTrust)

public:
    CGrayscale();

    ~CGrayscale();

    STDMETHOD(RuntimeClassInitialize)();

    // IMediaExtension
    STDMETHODIMP SetProperties(ABI::Windows::Foundation::Collections::IPropertySet *pConfiguration);

    // IMFTransform
    STDMETHODIMP GetStreamLimits(
        DWORD   *pdwInputMinimum,
        DWORD   *pdwInputMaximum,
        DWORD   *pdwOutputMinimum,
        DWORD   *pdwOutputMaximum
    );

    STDMETHODIMP GetStreamCount(
        DWORD   *pcInputStreams,
        DWORD   *pcOutputStreams
    );

    STDMETHODIMP GetStreamIDs(
        DWORD   dwInputIDArraySize,
        DWORD   *pdwInputIDs,
        DWORD   dwOutputIDArraySize,
        DWORD   *pdwOutputIDs
    );

    STDMETHODIMP GetInputStreamInfo(
        DWORD                     dwInputStreamID,
        MFT_INPUT_STREAM_INFO *   pStreamInfo
    );

    STDMETHODIMP GetOutputStreamInfo(
        DWORD                     dwOutputStreamID,
        MFT_OUTPUT_STREAM_INFO *  pStreamInfo
    );

    STDMETHODIMP GetAttributes(IMFAttributes** pAttributes);

    STDMETHODIMP GetInputStreamAttributes(
        DWORD           dwInputStreamID,
        IMFAttributes   **ppAttributes
    );

    STDMETHODIMP GetOutputStreamAttributes(
        DWORD           dwOutputStreamID,
        IMFAttributes   **ppAttributes
    );

    STDMETHODIMP DeleteInputStream(DWORD dwStreamID);

    STDMETHODIMP AddInputStreams(
        DWORD   cStreams,
        DWORD   *adwStreamIDs
    );

    STDMETHODIMP GetInputAvailableType(
        DWORD           dwInputStreamID,
        DWORD           dwTypeIndex, // 0-based
        IMFMediaType    **ppType
    );

    STDMETHODIMP GetOutputAvailableType(
        DWORD           dwOutputStreamID,
        DWORD           dwTypeIndex, // 0-based
        IMFMediaType    **ppType
    );

    STDMETHODIMP SetInputType(
        DWORD           dwInputStreamID,
        IMFMediaType    *pType,
        DWORD           dwFlags
    );

    STDMETHODIMP SetOutputType(
        DWORD           dwOutputStreamID,
        IMFMediaType    *pType,
        DWORD           dwFlags
    );

    STDMETHODIMP GetInputCurrentType(
        DWORD           dwInputStreamID,
        IMFMediaType    **ppType
    );

    STDMETHODIMP GetOutputCurrentType(
        DWORD           dwOutputStreamID,
        IMFMediaType    **ppType
    );

    STDMETHODIMP GetInputStatus(
        DWORD           dwInputStreamID,
        DWORD           *pdwFlags
    );

    STDMETHODIMP GetOutputStatus(DWORD *pdwFlags);

    STDMETHODIMP SetOutputBounds(
        LONGLONG        hnsLowerBound,
        LONGLONG        hnsUpperBound
    );

    STDMETHODIMP ProcessEvent(
        DWORD              dwInputStreamID,
        IMFMediaEvent      *pEvent
    );

    STDMETHODIMP ProcessMessage(
        MFT_MESSAGE_TYPE    eMessage,
        ULONG_PTR           ulParam
    );

    STDMETHODIMP ProcessInput(
        DWORD               dwInputStreamID,
        IMFSample           *pSample,
        DWORD               dwFlags
    );

    STDMETHODIMP ProcessOutput(
        DWORD                   dwFlags,
        DWORD                   cOutputBufferCount,
        MFT_OUTPUT_DATA_BUFFER  *pOutputSamples, // one per stream
        DWORD                   *pdwStatus
    );


private:
    // HasPendingOutput: Returns TRUE if the MFT is holding an input sample.
    BOOL HasPendingOutput() const { return m_pSample != NULL; }

    // IsValidInputStream: Returns TRUE if dwInputStreamID is a valid input stream identifier.
    BOOL IsValidInputStream(DWORD dwInputStreamID) const
    {
        return dwInputStreamID == 0;
    }

    // IsValidOutputStream: Returns TRUE if dwOutputStreamID is a valid output stream identifier.
    BOOL IsValidOutputStream(DWORD dwOutputStreamID) const
    {
        return dwOutputStreamID == 0;
    }

    HRESULT OnGetPartialType(DWORD dwTypeIndex, IMFMediaType **ppmt);
    HRESULT OnCheckInputType(IMFMediaType *pmt);
    HRESULT OnCheckOutputType(IMFMediaType *pmt);
    HRESULT OnCheckMediaType(IMFMediaType *pmt);
    void    OnSetInputType(IMFMediaType *pmt);
    void    OnSetOutputType(IMFMediaType *pmt);
    HRESULT BeginStreaming();
    HRESULT EndStreaming();
    HRESULT OnProcessOutput(IMFMediaBuffer *pIn, IMFMediaBuffer *pOut);
    HRESULT OnFlush();
    HRESULT UpdateFormatInfo();

    CRITICAL_SECTION            m_critSec;

    // Transformation parameters
	ProcessingType              m_TransformType;
    D2D_RECT_U                  m_rcDest;                   // Destination rectangle for the effect.

    // Streaming
    bool                        m_bStreamingInitialized;
    IMFSample                   *m_pSample;                 // Input sample.
    IMFMediaType                *m_pInputType;              // Input media type.
    IMFMediaType                *m_pOutputType;             // Output media type.

    // Fomat information
    UINT32                      m_imageWidthInPixels;
    UINT32                      m_imageHeightInPixels;
    DWORD                       m_cbImageSize;              // Image size, in bytes.

    IMFAttributes               *m_pAttributes;
};
#endif