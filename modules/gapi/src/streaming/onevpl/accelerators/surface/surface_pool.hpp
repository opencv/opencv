#ifndef GAPI_STREAMING_ONEVPL_SURFACE_SURFACE_POOL_HPP
#define GAPI_STREAMING_ONEVPL_SURFACE_SURFACE_POOL_HPP

#include <map>
#include <memory>
#include <vector>

#include "opencv2/gapi/own/exports.hpp" // GAPI_EXPORTS

#ifdef HAVE_ONEVPL
#if (MFX_VERSION >= 2000)
#include <vpl/mfxdispatcher.h>
#endif

#include <vpl/mfx.h>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

class Surface;
// GAPI_EXPORTS for tests
class GAPI_EXPORTS CachedPool {
public:
    using surface_ptr_t = std::shared_ptr<Surface>;
    using surface_container_t = std::vector<surface_ptr_t>;
    using free_surface_iterator_t = typename surface_container_t::iterator;
    using cached_surface_container_t = std::map<mfxFrameSurface1*, surface_ptr_t>;

    explicit CachedPool(size_t reserved_size = 0);

    void push_back(surface_ptr_t &&surf);
    size_t total_size() const;
    size_t available_size() const;
    void clear();

    surface_ptr_t find_free();
    surface_ptr_t find_by_handle(mfxFrameSurface1* handle);
private:
    void reserve(size_t size);

    surface_container_t surfaces;
    free_surface_iterator_t next_free_it;
    cached_surface_container_t cache;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_SURFACE_SURFACE_POOL_HPP
