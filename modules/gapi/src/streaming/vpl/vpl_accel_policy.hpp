#ifndef GAPI_VPL_ACCEL_POLICY_HPP
#define GAPI_VPL_ACCEL_POLICY_HPP

#include <opencv2/gapi/media.hpp>

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>

namespace cv {
namespace gapi {
namespace wip {

struct VPLAccelerationPolicy
{
    virtual ~VPLAccelerationPolicy() {}

    virtual cv::MediaFrame::AdapterPtr create_frame_adapter(mfxFrameSurface1* surface_ptr) = 0;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // GAPI_VPL_ACCEL_POLICY_HPP
