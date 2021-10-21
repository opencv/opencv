// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ONEVPL_UTILS_HPP
#define GAPI_STREAMING_ONEVPL_ONEVPL_UTILS_HPP

#ifdef HAVE_ONEVPL
#if (MFX_VERSION >= 2000)
#include <vpl/mfxdispatcher.h>
#endif // MFX_VERSION

#include <vpl/mfx.h>
#include <vpl/mfxvideo.h>

#include <map>
#include <memory>
#include <string>

#include <opencv2/gapi/streaming/onevpl/cfg_params.hpp>


namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

// Since ATL headers might not be available on specific MSVS Build Tools
// we use simple `CComPtr` implementation like as `ComPtrGuard`
// which is not supposed to be the full functional replacement of `CComPtr`
// and it uses as RAII to make sure utilization is correct
template <typename COMNonManageableType>
void release(COMNonManageableType *ptr) {
    if (ptr) {
        ptr->Release();
    }
}

template <typename COMNonManageableType>
using ComPtrGuard = std::unique_ptr<COMNonManageableType, decltype(&release<COMNonManageableType>)>;

template <typename COMNonManageableType>
ComPtrGuard<COMNonManageableType> createCOMPtrGuard(COMNonManageableType *ptr = nullptr) {
    return ComPtrGuard<COMNonManageableType> {ptr, &release<COMNonManageableType>};
}


const char* mfx_impl_to_cstr(const mfxIMPL impl);

mfxIMPL cstr_to_mfx_impl(const char* cstr);

const char* mfx_accel_mode_to_cstr (const mfxAccelerationMode mode);

mfxAccelerationMode cstr_to_mfx_accel_mode(const char* cstr);

const char* mfx_resource_type_to_cstr (const mfxResourceType type);

mfxResourceType cstr_to_mfx_resource_type(const char* cstr);

mfxU32 cstr_to_mfx_codec_id(const char* cstr);

const char* mfx_codec_type_to_cstr(const mfxU32 fourcc, const mfxU32 type);

mfxU32 cstr_to_mfx_version(const char* cstr);

std::string mfxstatus_to_string(mfxStatus err);

std::ostream& operator<< (std::ostream& out, const mfxImplDescription& idesc);

} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_ONEVPL_UTILS_HPP
