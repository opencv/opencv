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

    using pool_t = std::vector<surface_ptr_t>;

    void init(session_t session) override;
    void deinit(session_t session) override;
    pool_key_t create_surface_pool(size_t pool_size, size_t surface_size_bytes, surface_ptr_ctr_t creator) override;
    surface_weak_ptr_t get_free_surface(pool_key_t key) const override;
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
