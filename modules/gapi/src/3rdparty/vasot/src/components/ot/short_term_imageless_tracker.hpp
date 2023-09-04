/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef VAS_OT_SHORT_TERM_IMAGELESS_TRACKER_HPP
#define VAS_OT_SHORT_TERM_IMAGELESS_TRACKER_HPP

#include "tracker.hpp"

#include <deque>
#include <vector>

namespace vas {
namespace ot {

class ShortTermImagelessTracker : public Tracker {
  public:
    explicit ShortTermImagelessTracker(vas::ot::Tracker::InitParameters init_param);
    virtual ~ShortTermImagelessTracker();

    virtual int32_t TrackObjects(const cv::Mat &mat, const std::vector<Detection> &detections,
            std::vector<std::shared_ptr<Tracklet>> *tracklets, float delta_t) override;

    ShortTermImagelessTracker(const ShortTermImagelessTracker &) = delete;
    ShortTermImagelessTracker &operator=(const ShortTermImagelessTracker &) = delete;

  private:
    void TrimTrajectories();

  private:
    cv::Size image_sz;
};

}; // namespace ot
}; // namespace vas

#endif // VAS_OT_SHORT_TERM_IMAGELESS_TRACKER_HPP
