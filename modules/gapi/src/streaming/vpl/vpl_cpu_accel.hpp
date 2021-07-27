#ifndef GAPI_VPL_CPU_ACCEL_HPP
#define GAPI_VPL_CPU_ACCEL_HPP

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>
#include "streaming/vpl/vpl_accel_policy.hpp"

namespace cv {
namespace gapi {
namespace wip {

struct VPLCPUAccelerationPolicy final : public VPLAccelerationPolicy
{
    class MediaFrameAdapter;
    VPLCPUAccelerationPolicy(mfxSession session);
    ~VPLCPUAccelerationPolicy();

    cv::MediaFrame::AdapterPtr create_frame_adapter(mfxFrameSurface1* surface_ptr) override;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // GAPI_VPL_CPU_ACCEL_HPP
