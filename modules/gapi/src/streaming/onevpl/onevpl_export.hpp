#ifndef GAPI_STREAMING_ONEVPL_EXPORT_HPP
#define GAPI_STREAMING_ONEVPL_EXPORT_HPP

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4201)
#pragma warning(disable : 4302)
#pragma warning(disable : 4311)
#pragma warning(disable : 4312)
#endif // defined(_MSC_VER)

#ifdef HAVE_ONEVPL
#if defined(MFX_VERSION)
#if (MFX_VERSION >= 2000)
#include <vpl/mfxdispatcher.h>
#endif // MFX_VERSION
#endif // defined(MFX_VERSION)

#include <vpl/mfx.h>
#include <vpl/mfxvideo.h>

extern mfxLoader mfx_handle;
extern int impl_number;
#endif // HAVE_ONEVPL

#if defined(_MSC_VER)
#pragma warning(pop)
#endif // defined(_MSC_VER)

#endif // GAPI_STREAMING_ONEVPL_EXPORT_HPP
