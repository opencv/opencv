// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ACCELERATORS_SURFACE_HPP
#define GAPI_STREAMING_ONEVPL_ACCELERATORS_SURFACE_HPP

#include <atomic>
#include <memory>

#include "opencv2/gapi/own/exports.hpp" // GAPI_EXPORTS

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

/**
 * @brief Inner class for managing oneVPL surface through interface `mfxFrameSurface1`.
 *
 * Surface has no own memory and shares accelerator allocated memory using reference counter semantics.
 * So it lives till a last memory consumer (surface/accelerator/media frame) lives.
 * This approach allows to support different scenarious in releasing allocated memory
 *
 * VPL surface `mfxFrameSurface1` support Lock-Free semantics and application MUST NOT operate with a
 * surface in locked state. But VPL inner counter is not threadsafe so it would be failed in any concurrent scenario.
 * std::atomic counter introduced in a way to overcome this problem.
 * But only few scenarious for concurrency are supported here because it is not assumed to implement entire Surface in
 * for a fully multithread approach.
 * Supported concurrency scenarios deal only with transaction pair: @ref Surface::get_locks_count() against
 * @ref Surface::release_lock() - which may be called from different threads. On the other hand @ref Surface::get_locks_count() against
 * @ref Surface::obtain_lock() happens in single thread only. Surface doesn't support shared ownership that
 * because it doesn't require thread safe guarantee between transactions:
 * - @ref Surface::obtain_lock() against @ref Surface::obtain_lock()
 * - @ref Surface::obtain_lock() against @ref Surface::release_lock()
 * - @ref Surface::release_lock() against @ref Surface::release_lock()
 */
class GAPI_EXPORTS Surface final { // GAPI_EXPORTS for tests
public:
    using handle_t = mfxFrameSurface1;
    using info_t = mfxFrameInfo;
    using data_t = mfxFrameData;


    static std::shared_ptr<Surface> create_surface(std::unique_ptr<handle_t>&& surf,
                                                   std::shared_ptr<void> accociated_memory);
    ~Surface();

    handle_t* get_handle() const;
    const info_t& get_info() const;
    const data_t& get_data() const;
    data_t& get_data();

    /**
     * Extract value thread-safe lock counter (see @ref Surface description).
     * It's usual situation that counter may be instantly decreased in other thread after this method called.
     * We need instantaneous value. This method synchronized in inter-threading way with @ref Surface::release_lock()
     *
     * @return fetched locks count.
     */
    size_t get_locks_count() const;

    /**
     * Atomically increase value of thread-safe lock counter (see @ref Surface description).
     * This method is single-threaded happens-after @ref Surface::get_locks_count() and
     * multi-threaded happens-before @ref Surface::release_lock()
     *
     * @return locks count just before its increasing.
     */
    size_t obtain_lock();

    /**
     * Atomically decrease value of thread-safe lock counter (see @ref Surface description).
     * This method is synchronized with @ref Surface::get_locks_count() and
     * multi-threaded happens-after @ref Surface::obtain_lock()
     *
     * @return locks count just before its decreasing.
     */
     size_t release_lock();
private:
    Surface(std::unique_ptr<handle_t>&& surf, std::shared_ptr<void> accociated_memory);

    std::shared_ptr<void> workspace_memory_ptr;
    std::unique_ptr<handle_t> mfx_surface;
    std::atomic<size_t> mirrored_locked_count;
};

using surface_ptr_t = std::shared_ptr<Surface>;
using surface_weak_ptr_t = std::weak_ptr<Surface>;
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_ACCELERATORS_SURFACE_HPP
