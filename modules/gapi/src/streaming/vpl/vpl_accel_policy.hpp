#ifndef GAPI_VPL_ACCEL_POLICY_HPP
#define GAPI_VPL_ACCEL_POLICY_HPP

#include <functional>
#include <memory>
#include <type_traits>

#include <opencv2/gapi/media.hpp>

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>

namespace cv {
namespace gapi {
namespace wip {

struct VPLAccelerationPolicy
{
    virtual ~VPLAccelerationPolicy() {}

    using pool_key_t = void*;

    using surface_t = mfxFrameSurface1;
    using surface_raw_ptr_t = typename std::add_pointer<surface_t>::type;
    using surface_ptr_t = std::shared_ptr<surface_t>;
    using surface_weak_ptr_t = std::weak_ptr<surface_t>;
    using surface_ptr_ctr_t = std::function<surface_ptr_t(void* out_buf_ptr,
                                                          size_t out_buf_ptr_offset,
                                                          size_t out_buf_ptr_size)>;
                                                 
    virtual pool_key_t create_surface_pool(size_t pool_size, size_t surface_size_bytes, surface_ptr_ctr_t creator) = 0;
    virtual surface_weak_ptr_t get_free_surface(pool_key_t key) const = 0;
    virtual cv::MediaFrame::AdapterPtr create_frame_adapter(surface_raw_ptr_t surface) = 0;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // GAPI_VPL_ACCEL_POLICY_HPP
