#ifndef GAPI_VPL_CPU_ACCEL_HPP
#define GAPI_VPL_CPU_ACCEL_HPP

#include <map>
#include <vector>

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>
#include "streaming/vpl/vpl_accel_policy.hpp"

namespace cv {
namespace gapi {
namespace wip {

struct VPLCPUAccelerationPolicy final : public VPLAccelerationPolicy
{
    VPLCPUAccelerationPolicy();
    ~VPLCPUAccelerationPolicy();
#ifdef TEST_PERF
    class pool_t {
    public:
        using surface_container_t = std::vector<surface_ptr_t>;
        using free_surface_iterator_t = typename surface_container_t::iterator;
        using cached_surface_container_t = std::map<mfxFrameSurface1*, surface_ptr_t>;
        void push_back(surface_ptr_t &&surf);
        void reserve(size_t size);
        size_t size() const;
        void clear();
        surface_ptr_t find_free();
        surface_ptr_t find_by_handle(mfxFrameSurface1* handle);
    private:
        surface_container_t surfaces;
        free_surface_iterator_t next_free_it;
        cached_surface_container_t cache;
    };
#else  // TEST_PERF
    using pool_t = std::vector<surface_ptr_t>;
#endif // TEST_PERF

    void init(session_t session) override;
    void deinit(session_t session) override;
    pool_key_t create_surface_pool(size_t pool_size, size_t surface_size_bytes, surface_ptr_ctr_t creator) override;
    surface_weak_ptr_t get_free_surface(pool_key_t key) override;
    size_t get_free_surface_count(pool_key_t key) const override;
    size_t get_surface_count(pool_key_t key) const override;

    cv::MediaFrame::AdapterPtr create_frame_adapter(pool_key_t key,
                                                    mfxFrameSurface1* surface) override;

private:
    std::map<pool_key_t, pool_t> pool_table;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // GAPI_VPL_CPU_ACCEL_HPP
