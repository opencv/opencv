/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef VAS_OT_OBJECTS_ASSOCIATOR_HPP
#define VAS_OT_OBJECTS_ASSOCIATOR_HPP

#include "../tracklet.hpp"

namespace vas {
namespace ot {

class ObjectsAssociator {
  public:
    explicit ObjectsAssociator(bool tracking_per_class);
    virtual ~ObjectsAssociator();
    ObjectsAssociator() = delete;

  public:
    std::pair<std::vector<bool>, std::vector<int32_t>>
    Associate(const std::vector<Detection> &detections, const std::vector<std::shared_ptr<Tracklet>> &tracklets,
              const std::vector<cv::Mat> *detection_rgb_features = nullptr);

  private:
    std::vector<std::vector<float>> ComputeRgbDistance(const std::vector<Detection> &detections,
                                                       const std::vector<std::shared_ptr<Tracklet>> &tracklets,
                                                       const std::vector<cv::Mat> *detection_rgb_features);

    static float NormalizedCenterDistance(const cv::Rect2f &r1, const cv::Rect2f &r2);
    static float NormalizedShapeDistance(const cv::Rect2f &r1, const cv::Rect2f &r2);

  private:
    bool tracking_per_class_;
};

}; // namespace ot
}; // namespace vas

#endif // VAS_OT_OBJECTS_ASSOCIATOR_HPP
