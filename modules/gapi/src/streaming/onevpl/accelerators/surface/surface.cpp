// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <opencv2/gapi/own/assert.hpp>
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "logger.hpp"

#ifdef HAVE_ONEVPL

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

Surface::Surface(std::unique_ptr<handle_t>&& surf, std::shared_ptr<void> associated_memory) :
    workspace_memory_ptr(associated_memory),
    mfx_surface(std::move(surf)),
    mirrored_locked_count() {

    GAPI_Assert(mfx_surface && "Surface is nullptr");
    GAPI_LOG_DEBUG(nullptr, "create surface: " << get_handle() <<
                            ", locked count: " << mfx_surface->Data.Locked);
}

Surface::~Surface() {
    GAPI_LOG_DEBUG(nullptr, "destroy surface: " << get_handle() <<
                            ", worspace memory counter: " <<
                            workspace_memory_ptr.use_count());
}

std::shared_ptr<Surface> Surface::create_surface(std::unique_ptr<handle_t>&& surf,
                                                 std::shared_ptr<void> accociated_memory) {
    Surface::info_t& info = surf->Info;
    info.FourCC = MFX_FOURCC_NV12;
    surface_ptr_t ret {new Surface(std::move(surf), accociated_memory)};
    return ret;
}

Surface::handle_t* Surface::get_handle() const {
    return mfx_surface.get();
}

const Surface::info_t& Surface::get_info() const {
    return mfx_surface->Info;
}

const Surface::data_t& Surface::get_data() const {
    return mfx_surface->Data;
}

Surface::data_t& Surface::get_data() {
    return const_cast<Surface::data_t&>(static_cast<const Surface*>(this)->get_data());
}

size_t Surface::get_locks_count() const {
    return mirrored_locked_count.load() + mfx_surface->Data.Locked;
}

size_t Surface::obtain_lock() {
    size_t locked_count = mirrored_locked_count.fetch_add(1);
    GAPI_LOG_DEBUG(nullptr, "surface: " << get_handle() <<
                            ", locked times: " << locked_count + 1);
    return locked_count; // return preceding value
}

size_t Surface::release_lock() {
    size_t locked_count = mirrored_locked_count.fetch_sub(1);
    GAPI_Assert(locked_count && "Surface lock counter is invalid");
    GAPI_LOG_DEBUG(nullptr, "surface: " << get_handle() <<
                            ", locked times: " << locked_count - 1);
    return locked_count; // return preceding value
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
