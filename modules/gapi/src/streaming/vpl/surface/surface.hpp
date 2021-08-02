#ifndef GAPI_STREAMIN_ONE_VPL_SURFACE_HPP
#define GAPI_STREAMIN_ONE_VPL_SURFACE_HPP

#include <atomic>
#include <memory>
#include <mutex>

#ifdef HAVE_ONEVPL

#if (MFX_VERSION >= 2000)
#include <vpl/mfxdispatcher.h>
#endif

#include <vpl/mfx.h>
#endif

namespace cv {
namespace gapi {
namespace wip {

class Surface : public std::enable_shared_from_this<Surface> {
    using handle_t = mfxFrameSurface1;
    
    std::shared_ptr<void> workspace_memory_ptr;
    std::unique_ptr<handle_t> mfx_surface;
    std::atomic<size_t> mirrored_locked_count;
public:
    using info_t = mfxFrameInfo;
    using data_t = mfxFrameData;

    static std::shared_ptr<Surface> create_surface(std::unique_ptr<handle_t>&& surf,
                                                   std::shared_ptr<void> accociated_memory);
    ~Surface();

    std::shared_ptr<Surface> get_ptr();

    handle_t* get_handle() const;
    const info_t& get_info() const;
    const data_t& get_data() const;
    
    size_t get_locks_count() const;

    // it had better to implement RAII
    size_t obtain_lock();
    size_t release_lock();
private:
    Surface(std::unique_ptr<handle_t>&& surf, std::shared_ptr<void> accociated_memory);
};

using surface_ptr_t = std::shared_ptr<Surface>;
using surface_weak_ptr_t = std::weak_ptr<Surface>;
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // GAPI_STREAMIN_ONE_VPL_SURFACE_HPP
