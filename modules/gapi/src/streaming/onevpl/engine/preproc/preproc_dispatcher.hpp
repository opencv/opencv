// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_PREPROC_DISPATCHER_HPP
#define GAPI_STREAMING_ONEVPL_PREPROC_DISPATCHER_HPP

#include <memory>
#include <vector>

#include "streaming/onevpl/engine/preproc_engine_interface.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

// GAPI_EXPORTS for tests
class GAPI_EXPORTS VPPPreprocDispatcher final : public cv::gapi::wip::IPreprocEngine {
public:

    cv::util::optional<pp_params> is_applicable(const cv::MediaFrame& in_frame) override;

    pp_session initialize_preproc(const pp_params& initial_frame_param,
                                  const GFrameDesc& required_frame_descr) override;

    cv::MediaFrame run_sync(const pp_session &session_handle,
                            const cv::MediaFrame& in_frame,
                            const cv::util::optional<cv::Rect> &opt_roi) override;

    template<class PreprocImpl, class ...Args>
    void insert_worker(Args&& ...args) {
        workers.emplace_back(std::make_shared<PreprocImpl>(std::forward<Args>(args)...));
    }

    size_t size() const {
        return workers.size();
    }
private:
    std::vector<std::shared_ptr<cv::gapi::wip::IPreprocEngine>> workers;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // GAPI_STREAMING_ONEVPL_PREPROC_DISPATCHER_HPP
