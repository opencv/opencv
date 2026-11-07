/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "short_term_imageless_tracker.hpp"
#include "prof_def.hpp"
#include "../../common/exception.hpp"

#include <memory>

namespace vas {
namespace ot {

const int32_t kMaxAssociationLostCount = 2;    // ST_TRACKED -> ST_LOST
const int32_t kMaxAssociationFailCount = 20;   // ST_LOST -> ST_DEAD
const int32_t kMaxOutdatedCountInTracked = 30; // ST_TRACKED -> ST_LOST
const int32_t kMaxOutdatedCountInLost = 20;    // ST_LOST -> ST_DEAD
const int32_t kMaxTrajectorySize = 30;

/**
 *
 * ShortTermImagelessTracker
 *
 **/
ShortTermImagelessTracker::ShortTermImagelessTracker(vas::ot::Tracker::InitParameters init_param)
    : Tracker(init_param.max_num_objects, init_param.min_region_ratio_in_boundary, init_param.format,
              init_param.tracking_per_class),
      image_sz(0, 0) {
    TRACE(" - Created tracker = ShortTermImagelessTracker");
}

ShortTermImagelessTracker::~ShortTermImagelessTracker() {
}

int32_t ShortTermImagelessTracker::TrackObjects(const cv::Mat &mat, const std::vector<Detection> &detections,
                                                std::vector<std::shared_ptr<Tracklet>> *tracklets, float delta_t) {
    PROF_START(PROF_COMPONENTS_OT_SHORTTERM_RUN_TRACKER);

    int32_t input_img_width = mat.cols;
    int32_t input_img_height = mat.rows;

    if (input_image_format_ == vas::ColorFormat::NV12 || input_image_format_ == vas::ColorFormat::I420) {
        input_img_height = mat.rows / 3 * 2;
    }

    const cv::Rect2f image_boundary(0.0f, 0.0f, static_cast<float>(input_img_width),
                                    static_cast<float>(input_img_height));

    TRACE("Start TrackObjects frame_count_: %d, detection: %d, tracklet: %d ----------------", frame_count_,
          detections.size(), tracklets_.size());
    bool is_dead = false;
    if (image_sz.width != input_img_width || image_sz.height != input_img_height) {
        if (image_sz.width != 0 || image_sz.height != 0) {
            is_dead = true;
        }
        image_sz.width = input_img_width;
        image_sz.height = input_img_height;
    }

    PROF_START(PROF_COMPONENTS_OT_SHORTTERM_KALMAN_PREDICTION);
    // Predict tracklets state
    for (auto &tracklet : tracklets_) {
        auto sttimgless_tracklet = std::dynamic_pointer_cast<ShortTermImagelessTracklet>(tracklet);
        cv::Rect2f predicted_rect = sttimgless_tracklet->kalman_filter->Predict(delta_t);
        sttimgless_tracklet->predicted = predicted_rect;
        sttimgless_tracklet->trajectory.push_back(predicted_rect);
        sttimgless_tracklet->trajectory_filtered.push_back(predicted_rect);
        sttimgless_tracklet->association_delta_t += delta_t;
        // Reset association index every frame for new detection input
        sttimgless_tracklet->association_idx = kNoMatchDetection;
    }

    PROF_END(PROF_COMPONENTS_OT_SHORTTERM_KALMAN_PREDICTION);

    PROF_START(PROF_COMPONENTS_OT_SHORTTERM_UPDATE_STATUS);
    // Conduct tracking of SOT for each tracklet
    TRACE(" Update status");
    for (auto &tracklet : tracklets_) {
        if (is_dead) {
            tracklet->status = ST_DEAD;
            continue;
        }

        // tracklet->association_delta_t = 0.0f;  // meaning updated by SOT
    }

    if (is_dead) {
        RemoveDeadTracklets();
    }
    PROF_END(PROF_COMPONENTS_OT_SHORTTERM_UPDATE_STATUS);

    PROF_START(PROF_COMPONENTS_OT_SHORTTERM_RUN_ASSOCIATION);

    // Tracklet-detection association
    int32_t n_detections = static_cast<int32_t>(detections.size());
    int32_t n_tracklets = static_cast<int32_t>(tracklets_.size());

    std::vector<bool> d_is_associated(n_detections, false);
    std::vector<int32_t> t_associated_d_index(n_tracklets, -1);

    if (n_detections > 0) {
        auto result = associator_.Associate(detections, tracklets_);
        d_is_associated = result.first;
        t_associated_d_index = result.second;
    }

    PROF_END(PROF_COMPONENTS_OT_SHORTTERM_RUN_ASSOCIATION);

    PROF_START(PROF_COMPONENTS_OT_SHORTTERM_UPDATE_STATUS);
    // Update tracklets' state
    if (n_detections > 0) {
        for (int32_t t = 0; t < n_tracklets; ++t) {
            auto &tracklet = tracklets_[t];
            if (t_associated_d_index[t] >= 0) {
                tracklet->association_delta_t = 0.0f;
                int32_t associated_d_index = t_associated_d_index[t];
                const cv::Rect2f &d_bounding_box = detections[associated_d_index].rect & image_boundary;

                // Apply associated detection to tracklet
                tracklet->association_idx = detections[associated_d_index].index;
                tracklet->association_fail_count = 0;
                tracklet->age = 0;
                tracklet->label = detections[associated_d_index].class_label;

                auto sttimgless_tracklet = std::dynamic_pointer_cast<ShortTermImagelessTracklet>(tracklet);
                if (!sttimgless_tracklet)
                    continue;

                if (sttimgless_tracklet->status == ST_NEW) {
                    sttimgless_tracklet->trajectory.back() = d_bounding_box;
                    sttimgless_tracklet->trajectory_filtered.back() =
                        sttimgless_tracklet->kalman_filter->Correct(sttimgless_tracklet->trajectory.back());
                    sttimgless_tracklet->status = ST_TRACKED;
                } else if (sttimgless_tracklet->status == ST_TRACKED) {
                    sttimgless_tracklet->trajectory.back() = d_bounding_box;
                    sttimgless_tracklet->trajectory_filtered.back() =
                        sttimgless_tracklet->kalman_filter->Correct(sttimgless_tracklet->trajectory.back());
                } else if (sttimgless_tracklet->status == ST_LOST) {
                    sttimgless_tracklet->RenewTrajectory(d_bounding_box);
                    sttimgless_tracklet->status = ST_TRACKED;
                }
            } else // Association failure
            {
                tracklet->association_fail_count++;
                if (tracklet->status == ST_NEW) {
                    tracklet->status = ST_DEAD; // regard non-consecutive association as false alarm
                } else if (tracklet->status == ST_TRACKED) {
                    if (tracklet->association_fail_count > kMaxAssociationLostCount) {
                        // # association fail > threshold while tracking -> MISSING
                        tracklet->status = ST_LOST;
                        tracklet->association_fail_count = 0;
                        tracklet->age = 0;
                    }
                } else if (tracklet->status == ST_LOST) {
                    if (tracklet->association_fail_count > kMaxAssociationFailCount) {
                        // # association fail > threshold while missing -> DEAD
                        tracklet->status = ST_DEAD;
                    }
                }
            }
        }
    } else // detections.size() == 0
    {
        for (int32_t t = 0; t < static_cast<int32_t>(tracklets_.size()); ++t) {
            auto &tracklet = tracklets_[t];
            // Always change ST_NEW to ST_TRACKED: no feature tracking from previous detection input.
            if (tracklet->status == ST_NEW) {
                tracklet->status = ST_TRACKED;
            }

            auto sttimgless_tracklet = std::dynamic_pointer_cast<ShortTermImagelessTracklet>(tracklet);
            if (!sttimgless_tracklet)
                continue;

            if (sttimgless_tracklet->status == ST_TRACKED) {
                if (sttimgless_tracklet->age > kMaxOutdatedCountInTracked) {
                    sttimgless_tracklet->status = ST_LOST;
                    sttimgless_tracklet->association_fail_count = 0;
                    sttimgless_tracklet->age = 0;
                } else {
                    sttimgless_tracklet->trajectory_filtered.back() =
                        sttimgless_tracklet->kalman_filter->Correct(sttimgless_tracklet->trajectory.back());
                }
            }

            if (sttimgless_tracklet->status == ST_LOST) {
                if (sttimgless_tracklet->age >= kMaxOutdatedCountInLost) {
                    // # association fail > threshold while missing -> DEAD
                    sttimgless_tracklet->status = ST_DEAD;
                }
            }
        }
    }
    PROF_END(PROF_COMPONENTS_OT_SHORTTERM_UPDATE_STATUS);

    PROF_START(PROF_COMPONENTS_OT_SHORTTERM_COMPUTE_OCCLUSION);
    ComputeOcclusion();
    PROF_END(PROF_COMPONENTS_OT_SHORTTERM_COMPUTE_OCCLUSION);

    PROF_START(PROF_COMPONENTS_OT_SHORTTERM_REGISTER_OBJECT);
    // Register remaining detections as new objects
    for (int32_t d = 0; d < static_cast<int32_t>(detections.size()); ++d) {
        if (d_is_associated[d] == false) {
            if (static_cast<int32_t>(tracklets_.size()) >= max_objects_ && max_objects_ != -1)
                continue;

            std::unique_ptr<ShortTermImagelessTracklet> tracklet(new ShortTermImagelessTracklet());

            tracklet->status = ST_NEW;
            tracklet->id = GetNextTrackingID();
            tracklet->label = detections[d].class_label;
            tracklet->association_idx = detections[d].index;

            const cv::Rect2f &bounding_box = detections[d].rect & image_boundary;
            tracklet->InitTrajectory(bounding_box);
            tracklet->kalman_filter.reset(new KalmanFilterNoOpencv(bounding_box));
            tracklets_.push_back(std::move(tracklet));
        }
    }
    PROF_END(PROF_COMPONENTS_OT_SHORTTERM_REGISTER_OBJECT);

    RemoveDeadTracklets();
    RemoveOutOfBoundTracklets(input_img_width, input_img_height);
    TrimTrajectories();

    *tracklets = tracklets_;

    // Increase age
    for (auto &tracklet : tracklets_) {
        tracklet->age++;
    }

    IncreaseFrameCount();
    PROF_END(PROF_COMPONENTS_OT_SHORTTERM_RUN_TRACKER);
    return 0;
}

void ShortTermImagelessTracker::TrimTrajectories() {
    for (auto &tracklet : tracklets_) {
        auto &trajectory = tracklet->trajectory;
        while (trajectory.size() > kMaxTrajectorySize) {
            trajectory.pop_front();
        }

        //
        auto &trajectory_filtered = tracklet->trajectory_filtered;
        while (trajectory_filtered.size() > kMaxTrajectorySize) {
            trajectory_filtered.pop_front();
        }
    }
}

}; // namespace ot
}; // namespace vas
