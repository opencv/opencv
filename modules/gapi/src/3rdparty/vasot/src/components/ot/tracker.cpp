/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "short_term_imageless_tracker.hpp"
#include "zero_term_imageless_tracker.hpp"

#include "../../common/exception.hpp"

namespace vas {
namespace ot {

Tracker::Tracker(int32_t max_objects, float min_region_ratio_in_boundary, vas::ColorFormat format, bool class_per_class)
    : max_objects_(max_objects), next_id_(1), frame_count_(0),
      min_region_ratio_in_boundary_(min_region_ratio_in_boundary), input_image_format_(format),
      associator_(ObjectsAssociator(class_per_class)) {
}

Tracker::~Tracker() {
}

Tracker *Tracker::CreateInstance(InitParameters init_parameters) {
    TRACE("START CreateInstance Tracker");

    Tracker *tracker = nullptr;
    if (init_parameters.profile == PROFILE_SHORT_TERM_IMAGELESS) {
        tracker = new ShortTermImagelessTracker(init_parameters);
    } else if (init_parameters.profile == PROFILE_ZERO_TERM_IMAGELESS) {
        tracker = new ZeroTermImagelessTracker(init_parameters);
    } else {
        throw std::runtime_error("Unsupported tracking type");
    }

    TRACE(" - max_num_objects(%d)", init_parameters.max_num_objects);

    TRACE("END");
    return tracker;
}

int32_t Tracker::RemoveObject(const int32_t id) {
    if (id == 0)
        return -1;

    for (auto tracklet = tracklets_.begin(); tracklet != tracklets_.end(); ++tracklet) {
        if ((*tracklet)->id == id) {
            tracklet = tracklets_.erase(tracklet);
            return 0;
        }
    }
    return -1;
}

void Tracker::Reset(void) {
    frame_count_ = 0;
    tracklets_.clear();
}

int32_t Tracker::GetFrameCount(void) const {
    return frame_count_;
}

int32_t Tracker::GetNextTrackingID() {
    return next_id_++;
}

void Tracker::IncreaseFrameCount() {
    frame_count_++;
}

void Tracker::ComputeOcclusion() {
    // Compute occlusion ratio
    for (int32_t t0 = 0; t0 < static_cast<int32_t>(tracklets_.size()); ++t0) {
        auto &tracklet0 = tracklets_[t0];
        if (tracklet0->status != ST_TRACKED)
            continue;

        const cv::Rect2f &r0 = tracklet0->trajectory.back();
        float max_occlusion_ratio = 0.0f;
        for (int32_t t1 = 0; t1 < static_cast<int32_t>(tracklets_.size()); ++t1) {
            const auto &tracklet1 = tracklets_[t1];
            if (t0 == t1 || tracklet1->status == ST_LOST)
                continue;

            const cv::Rect2f &r1 = tracklet1->trajectory.back();
            max_occlusion_ratio = std::max(max_occlusion_ratio, (r0 & r1).area() / r0.area()); // different from IoU
        }
        tracklets_[t0]->occlusion_ratio = max_occlusion_ratio;
    }
}

void Tracker::RemoveOutOfBoundTracklets(int32_t input_width, int32_t input_height, bool is_filtered) {
    const cv::Rect2f image_region(0.0f, 0.0f, static_cast<float>(input_width), static_cast<float>(input_height));
    for (auto tracklet = tracklets_.begin(); tracklet != tracklets_.end();) {
        const cv::Rect2f &object_region =
            is_filtered ? (*tracklet)->trajectory_filtered.back() : (*tracklet)->trajectory.back();
        if ((image_region & object_region).area() / object_region.area() <
            min_region_ratio_in_boundary_) { // only 10% is in image boundary
            tracklet = tracklets_.erase(tracklet);
        } else {
            ++tracklet;
        }
    }
}

void Tracker::RemoveDeadTracklets() {
    for (auto tracklet = tracklets_.begin(); tracklet != tracklets_.end();) {
        if ((*tracklet)->status == ST_DEAD) {
            tracklet = tracklets_.erase(tracklet);
        } else {
            ++tracklet;
        }
    }
}

bool Tracker::RemoveOneLostTracklet() {
    for (auto tracklet = tracklets_.begin(); tracklet != tracklets_.end();) {
        if ((*tracklet)->status == ST_LOST) {
            // The first tracklet is the oldest
            tracklet = tracklets_.erase(tracklet);
            return true;
        } else {
            ++tracklet;
        }
    }

    return false;
}

}; // namespace ot
}; // namespace vas
