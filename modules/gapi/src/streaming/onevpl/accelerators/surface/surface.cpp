// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

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
    mirrored_locked_count.store(mfx_surface->Data.Locked);
    GAPI_LOG_DEBUG(nullptr, "create surface: " << mfx_surface <<
                            ", locked count: " << mfx_surface->Data.Locked);
}

Surface::~Surface() {
    GAPI_LOG_DEBUG(nullptr, "destroy surface: " << mfx_surface <<
                            ", worspace memory counter: " << workspace_memory_ptr.use_count());
}

std::shared_ptr<Surface> Surface::create_surface(std::unique_ptr<handle_t>&& surf,
                                                 std::shared_ptr<void> accociated_memory) {
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

size_t Surface::get_locks_count() const {
    return mirrored_locked_count.load();
}

size_t Surface::obtain_lock() {
    size_t locked_count = mirrored_locked_count.fetch_add(1);
    GAPI_Assert(locked_count < std::numeric_limits<mfxU16>::max() && "Too many references ");
    mfx_surface->Data.Locked = static_cast<mfxU16>(locked_count + 1);
    GAPI_LOG_DEBUG(nullptr, "surface: " << mfx_surface.get() <<
                            ", locked times: " << locked_count + 1);
    return locked_count; // return preceding value
}

size_t Surface::release_lock() {
    size_t locked_count = mirrored_locked_count.fetch_sub(1);
    GAPI_Assert(locked_count < std::numeric_limits<mfxU16>::max() && "Too many references ");
    GAPI_Assert(locked_count && "Surface lock counter is invalid");
    mfx_surface->Data.Locked = static_cast<mfxU16>(locked_count - 1);
    GAPI_LOG_DEBUG(nullptr, "surface: " << mfx_surface.get() <<
                            ", locked times: " << locked_count - 1);
    return locked_count; // return preceding value
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
