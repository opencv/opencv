// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ACCELERATORS_SURFACE_DX11_FRAME_ADAPTER_HPP
#define GAPI_STREAMING_ONEVPL_ACCELERATORS_SURFACE_DX11_FRAME_ADAPTER_HPP
#include <memory>

#include "streaming/onevpl/accelerators/surface/base_frame_adapter.hpp"
#include "streaming/onevpl/accelerators/utils/shared_lock.hpp"
#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
    #define D3D11_NO_HELPERS
    #define NOMINMAX
    #include <d3d11.h>
    #include <codecvt>
    #include "opencv2/core/directx.hpp"
    #ifdef HAVE_OPENCL
        #include <CL/cl_d3d11.h>
    #endif

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
class VPLMediaFrameDX11Adapter final: public BaseFrameAdapter,
                                      public SharedLock {
public:
    // GAPI_EXPORTS for tests
    GAPI_EXPORTS VPLMediaFrameDX11Adapter(std::shared_ptr<Surface> assoc_surface,
                                          SessionHandle assoc_handle);
    GAPI_EXPORTS ~VPLMediaFrameDX11Adapter();
    MediaFrame::View access(MediaFrame::Access) override;

    // FIXME: Consider a better solution since this approach
    // is not easily extendable for other adapters (oclcore.cpp)
    // FIXME: Use with caution since the handle might become invalid
    //        due to reference counting
    mfxHDLPair getHandle() const;
    // The default implementation does nothing
    cv::util::any blobParams() const override;
    void serialize(cv::gapi::s11n::IOStream&) override;
    void deserialize(cv::gapi::s11n::IIStream&) override;

    static DXGI_FORMAT get_dx11_color_format(uint32_t mfx_fourcc);
private:
    mfxFrameAllocator allocator;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#undef NOMINMAX
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_ACCELERATORS_SURFACE_DX11_FRAME_ADAPTER_HPP
