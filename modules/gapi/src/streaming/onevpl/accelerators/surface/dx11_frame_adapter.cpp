// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "streaming/onevpl/accelerators/surface/dx11_frame_adapter.hpp"
#include "streaming/onevpl/accelerators/dx11_alloc_resource.hpp"
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "logger.hpp"

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
#ifdef HAVE_INF_ENGINE
// For IE classes (ParamMap, etc)
#include <inference_engine.hpp>
#endif // HAVE_INF_ENGINE

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

void lock_mid(mfxMemId mid, mfxFrameData &data, MediaFrame::Access mode) {
    LockAdapter* alloc_data = reinterpret_cast<LockAdapter *>(mid);
    if (mode == MediaFrame::Access::R) {
        alloc_data->read_lock(mid, data);
    } else {
        alloc_data->write_lock(mid, data);
    }
}

void unlock_mid(mfxMemId mid, mfxFrameData &data, MediaFrame::Access mode) {
    LockAdapter* alloc_data = reinterpret_cast<LockAdapter*>(data.MemId);
    if (mode == MediaFrame::Access::R) {
        alloc_data->unlock_read(mid, data);
    } else {
        alloc_data->unlock_write(mid, data);
    }
}

VPLMediaFrameDX11Adapter::VPLMediaFrameDX11Adapter(std::shared_ptr<Surface> assoc_surface,
                                                   SessionHandle assoc_handle):
    BaseFrameAdapter(assoc_surface, assoc_handle, AccelType::DX11) {
    Surface::data_t& data = assoc_surface->get_data();

    LockAdapter* alloc_data = reinterpret_cast<LockAdapter*>(data.MemId);
    alloc_data->set_adaptee(this);
}

VPLMediaFrameDX11Adapter::~VPLMediaFrameDX11Adapter() {
    Surface::data_t& data = surface()->get_data();
    LockAdapter* alloc_data = reinterpret_cast<LockAdapter*>(data.MemId);
    alloc_data->set_adaptee(nullptr);
}

MediaFrame::View VPLMediaFrameDX11Adapter::access(MediaFrame::Access mode) {
    // NB: make copy for some copyable object, because access release may be happened
    // after source/pool destruction, so we need a copy
    auto surface_ptr_copy = surface();
    Surface::data_t& data = surface_ptr_copy->get_data();
    const Surface::info_t& info = surface_ptr_copy->get_info();
    void* frame_id = reinterpret_cast<void*>(this);

    GAPI_LOG_DEBUG(nullptr, "START lock frame in surface: " << surface_ptr_copy->get_handle() <<
                            ", frame id: " << frame_id);

    // lock MT
    lock_mid(data.MemId, data, mode);

    GAPI_LOG_DEBUG(nullptr, "FINISH lock frame in surface: " << surface_ptr_copy->get_handle() <<
                            ", frame id: " << frame_id);
    using stride_t = typename cv::MediaFrame::View::Strides::value_type;
    stride_t pitch = static_cast<stride_t>(data.Pitch);

    auto release_guard = [surface_ptr_copy, frame_id, mode] () {
        surface_ptr_copy->obtain_lock();

        auto& data = surface_ptr_copy->get_data();
        GAPI_LOG_DEBUG(nullptr, "START unlock frame in surface: " << surface_ptr_copy->get_handle() <<
                                ", frame id: " << frame_id);
        unlock_mid(data.MemId, data, mode);

        GAPI_LOG_DEBUG(nullptr, "FINISH unlock frame in surface: " << surface_ptr_copy->get_handle() <<
                                ", frame id: " << frame_id);
        surface_ptr_copy->release_lock();
    };

    switch(info.FourCC) {
        case MFX_FOURCC_I420:
        {
            GAPI_Assert(data.Y && data.U && data.V && "MFX_FOURCC_I420 frame data is nullptr");
            cv::MediaFrame::View::Ptrs pp = { data.Y, data.U, data.V, nullptr };
            cv::MediaFrame::View::Strides ss = { pitch, pitch / 2, pitch / 2, 0u };
            return cv::MediaFrame::View(std::move(pp), std::move(ss), release_guard);
        }
        case MFX_FOURCC_NV12:
        {
            if (!data.Y || !data.UV) {
                GAPI_LOG_WARNING(nullptr, "Empty data detected!!! for surface: " << surface_ptr_copy->get_handle() <<
                                          ", frame id: " << frame_id);
            }
            GAPI_Assert(data.Y && data.UV && "MFX_FOURCC_NV12 frame data is nullptr");
            cv::MediaFrame::View::Ptrs pp = { data.Y, data.UV, nullptr, nullptr };
            cv::MediaFrame::View::Strides ss = { pitch, pitch, 0u, 0u };
            return cv::MediaFrame::View(std::move(pp), std::move(ss), release_guard);
        }
            break;
        default:
            throw std::runtime_error("MediaFrame unknown 'fmt' type: " + std::to_string(info.FourCC));
    }
}

mfxHDLPair VPLMediaFrameDX11Adapter::getHandle() const {
    auto surface_ptr_copy = get_surface();
    const Surface::data_t& data = surface_ptr_copy->get_data();
    NativeHandleAdapter* native_handle_getter = reinterpret_cast<NativeHandleAdapter*>(data.MemId);

    mfxHDLPair handle{};
    native_handle_getter->get_handle(data.MemId, reinterpret_cast<mfxHDL&>(handle));
    return handle;
}

cv::util::any VPLMediaFrameDX11Adapter::blobParams() const {
    /*GAPI_Error("VPLMediaFrameDX11Adapter::blobParams() is not fully integrated"
                         "in OpenVINO InferenceEngine and would be temporary disable.");*/
#ifdef HAVE_INF_ENGINE
    mfxHDLPair handle = getHandle();

    auto surface_ptr_copy = get_surface();
    const Surface::info_t& info = surface_ptr_copy->get_info();

    GAPI_Assert(frame_desc.fmt == MediaFormat::NV12 &&
                "blobParams() for VPLMediaFrameDX11Adapter supports NV12 only");

    InferenceEngine::ParamMap y_params{{"SHARED_MEM_TYPE", "VA_SURFACE"},
                                       {"DEV_OBJECT_HANDLE", handle.first},
                                       {"COLOR_FORMAT", InferenceEngine::ColorFormat::NV12},
                                       {"VA_PLANE",
                                         static_cast<DX11AllocationItem::subresource_id_t>(
                                            reinterpret_cast<uint64_t>(
                                                reinterpret_cast<DX11AllocationItem::subresource_id_t *>(
                                                    handle.second)))}};//,
    InferenceEngine::TensorDesc y_tdesc({InferenceEngine::Precision::U8,
                                        {1, 1, static_cast<size_t>(info.Height),
                                         static_cast<size_t>(info.Width)},
                                        InferenceEngine::Layout::NHWC});

    InferenceEngine::ParamMap uv_params = y_params;
    uv_params["MEM_HANDLE"] = handle.first;
    uv_params["VA_PLANE"] = static_cast<DX11AllocationItem::subresource_id_t>(
                                            reinterpret_cast<uint64_t>(
                                                reinterpret_cast<DX11AllocationItem::subresource_id_t *>(
                                                    handle.second))) + 1;
    InferenceEngine::TensorDesc uv_tdesc({InferenceEngine::Precision::U8,
                                         {1, 2, static_cast<size_t>(info.Height) / 2,
                                          static_cast<size_t>(info.Width) / 2},
                                         InferenceEngine::Layout::NHWC});
    return std::make_pair(std::make_pair(y_tdesc, y_params),
                          std::make_pair(uv_tdesc, uv_params));
#else
    GAPI_Error("VPLMediaFrameDX11Adapter::blobParams() is not implemented");
#endif // HAVE_INF_ENGINE
}

void VPLMediaFrameDX11Adapter::serialize(cv::gapi::s11n::IOStream&) {
    GAPI_Error("VPLMediaFrameDX11Adapter::serialize() is not implemented");
}

void VPLMediaFrameDX11Adapter::deserialize(cv::gapi::s11n::IIStream&) {
    GAPI_Error("VPLMediaFrameDX11Adapter::deserialize() is not implemented");
}

DXGI_FORMAT VPLMediaFrameDX11Adapter::get_dx11_color_format(uint32_t mfx_fourcc) {
    switch (mfx_fourcc) {
        case MFX_FOURCC_NV12:
            return DXGI_FORMAT_NV12;

        case MFX_FOURCC_YUY2:
            return DXGI_FORMAT_YUY2;

        case MFX_FOURCC_RGB4:
            return DXGI_FORMAT_B8G8R8A8_UNORM;

        case MFX_FOURCC_P8:
        case MFX_FOURCC_P8_TEXTURE:
            return DXGI_FORMAT_P8;

        case MFX_FOURCC_ARGB16:
        case MFX_FOURCC_ABGR16:
            return DXGI_FORMAT_R16G16B16A16_UNORM;

        case MFX_FOURCC_P010:
            return DXGI_FORMAT_P010;

        case MFX_FOURCC_A2RGB10:
            return DXGI_FORMAT_R10G10B10A2_UNORM;

        case DXGI_FORMAT_AYUV:
        case MFX_FOURCC_AYUV:
            return DXGI_FORMAT_AYUV;

        default:
            return DXGI_FORMAT_UNKNOWN;
    }
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
#endif // HAVE_ONEVPL
