// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ACCELERATORS_ACCEL_POLICY_CPU_HPP
#define GAPI_STREAMING_ONEVPL_ACCELERATORS_ACCEL_POLICY_CPU_HPP

#include <map>
#include <vector>

#include "opencv2/gapi/own/exports.hpp" // GAPI_EXPORTS

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>
#include "streaming/onevpl/accelerators/accel_policy_interface.hpp"
#ifdef TEST_PERF
#include "streaming/onevpl/accelerators/surface/surface_pool.hpp"
#endif // TEST_PERF

namespace cv {
namespace gapi {
namespace wip {

struct VPLCPUAccelerationPolicy final : public VPLAccelerationPolicy
{
    // GAPI_EXPORTS for tests
    GAPI_EXPORTS VPLCPUAccelerationPolicy();
    GAPI_EXPORTS ~VPLCPUAccelerationPolicy();
#ifdef TEST_PERF
    using pool_t = CachedPool;
#else  // TEST_PERF
    using pool_t = std::vector<surface_ptr_t>;
#endif // TEST_PERF

    GAPI_EXPORTS void init(session_t session) override;
    GAPI_EXPORTS void deinit(session_t session) override;
    GAPI_EXPORTS pool_key_t create_surface_pool(size_t pool_size, size_t surface_size_bytes, surface_ptr_ctr_t creator) override;
    GAPI_EXPORTS surface_weak_ptr_t get_free_surface(pool_key_t key) override;
    GAPI_EXPORTS size_t get_free_surface_count(pool_key_t key) const override;
    GAPI_EXPORTS size_t get_surface_count(pool_key_t key) const override;

    GAPI_EXPORTS cv::MediaFrame::AdapterPtr create_frame_adapter(pool_key_t key,
                                                                 mfxFrameSurface1* surface) override;

private:
    std::map<pool_key_t, pool_t> pool_table;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_ACCELERATORS_ACCEL_POLICY_CPU_HPP
