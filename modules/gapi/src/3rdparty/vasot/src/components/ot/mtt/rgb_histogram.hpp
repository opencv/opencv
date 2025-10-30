/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef VAS_OT_RGB_HISTOGRAM_HPP
#define VAS_OT_RGB_HISTOGRAM_HPP

#include <opencv2/core.hpp>
#include <cstdint>

namespace vas {
namespace ot {

class RgbHistogram {
  public:
    explicit RgbHistogram(int32_t rgb_bin_size);
    virtual ~RgbHistogram(void);

    virtual void Compute(const cv::Mat &image, cv::Mat *hist);
    virtual void ComputeFromBgra32(const cv::Mat &image, cv::Mat *hist);
    virtual int32_t FeatureSize(void) const; // currently 512 * float32

    static float ComputeSimilarity(const cv::Mat &hist1, const cv::Mat &hist2);

  protected:
    int32_t rgb_bin_size_;
    int32_t rgb_num_bins_;
    int32_t rgb_hist_size_;

    void AccumulateRgbHistogram(const cv::Mat &patch, float *rgb_hist) const;
    void AccumulateRgbHistogram(const cv::Mat &patch, const cv::Mat &weight, float *rgb_hist) const;

    void AccumulateRgbHistogramFromBgra32(const cv::Mat &patch, float *rgb_hist) const;
    void AccumulateRgbHistogramFromBgra32(const cv::Mat &patch, const cv::Mat &weight, float *rgb_hist) const;
};

}; // namespace ot
}; // namespace vas

#endif // VAS_OT_RGB_HISTOGRAM_HPP
