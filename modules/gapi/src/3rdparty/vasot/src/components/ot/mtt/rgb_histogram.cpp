/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "rgb_histogram.hpp"

namespace vas {
namespace ot {

RgbHistogram::RgbHistogram(int32_t rgb_bin_size)
    : rgb_bin_size_(rgb_bin_size), rgb_num_bins_(256 / rgb_bin_size),
      rgb_hist_size_(static_cast<int32_t>(pow(rgb_num_bins_, 3))) {
}

RgbHistogram::~RgbHistogram(void) {
}

void RgbHistogram::Compute(const cv::Mat &image, cv::Mat *hist) {
    // Init output buffer
    hist->create(1, rgb_hist_size_, CV_32FC1);
    (*hist) = cv::Scalar(0);
    float *hist_data = hist->ptr<float>();

    // Compute quantized RGB histogram
    AccumulateRgbHistogram(image, hist_data);
}

void RgbHistogram::ComputeFromBgra32(const cv::Mat &image, cv::Mat *hist) {
    // Init output buffer
    hist->create(1, rgb_hist_size_, CV_32FC1);
    (*hist) = cv::Scalar(0);
    float *hist_data = hist->ptr<float>();

    // Compute quantized RGB histogram
    AccumulateRgbHistogramFromBgra32(image, hist_data);
}

int32_t RgbHistogram::FeatureSize(void) const {
    return rgb_hist_size_;
}

float RgbHistogram::ComputeSimilarity(const cv::Mat &hist1, const cv::Mat &hist2) {
    // PROF_START(PROF_COMPONENTS_OT_SHORTTERM_HIST_SIMILARITY);
    // Bhattacharyya coeff (w/o weights)
    const float eps = 0.0001f;
    const int32_t hist_size = hist1.cols;
    const float *hist_data1 = hist1.ptr<float>();
    const float *hist_data2 = hist2.ptr<float>();
    float corr = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    for (int32_t i = 0; i < hist_size; ++i) {
        float v1 = hist_data1[i];
        float v2 = hist_data2[i];
        corr += sqrtf(v1 * v2);
        sum1 += v1;
        sum2 += v2;
    }

    // PROF_END(PROF_COMPONENTS_OT_SHORTTERM_HIST_SIMILARITY);
    if (sum1 > eps && sum2 > eps) {
        return corr / sqrtf(sum1 * sum2);
    } else {
        return 0.0f;
    }
}

void RgbHistogram::AccumulateRgbHistogram(const cv::Mat &patch, float *rgb_hist) const {
    for (int32_t y = 0; y < patch.rows; ++y) {
        const cv::Vec3b *patch_ptr = patch.ptr<cv::Vec3b>(y);
        for (int32_t x = 0; x < patch.cols; ++x) {
            int32_t index0 = patch_ptr[x][0] / rgb_bin_size_;
            int32_t index1 = patch_ptr[x][1] / rgb_bin_size_;
            int32_t index2 = patch_ptr[x][2] / rgb_bin_size_;
            int32_t hist_index = rgb_num_bins_ * (rgb_num_bins_ * index0 + index1) + index2;
            rgb_hist[hist_index] += 1.0f;
        }
    }
}

void RgbHistogram::AccumulateRgbHistogram(const cv::Mat &patch, const cv::Mat &weight, float *rgb_hist) const {
    for (int32_t y = 0; y < patch.rows; ++y) {
        const cv::Vec3b *patch_ptr = patch.ptr<cv::Vec3b>(y);
        const float *weight_ptr = weight.ptr<float>(y);
        for (int32_t x = 0; x < patch.cols; ++x) {
            int32_t index0 = patch_ptr[x][0] / rgb_bin_size_;
            int32_t index1 = patch_ptr[x][1] / rgb_bin_size_;
            int32_t index2 = patch_ptr[x][2] / rgb_bin_size_;
            int32_t hist_index = rgb_num_bins_ * (rgb_num_bins_ * index0 + index1) + index2;
            rgb_hist[hist_index] += weight_ptr[x];
        }
    }
}

void RgbHistogram::AccumulateRgbHistogramFromBgra32(const cv::Mat &patch, float *rgb_hist) const {
    for (int32_t y = 0; y < patch.rows; ++y) {
        const cv::Vec4b *patch_ptr = patch.ptr<cv::Vec4b>(y);
        for (int32_t x = 0; x < patch.cols; ++x) {
            int32_t index0 = patch_ptr[x][0] / rgb_bin_size_;
            int32_t index1 = patch_ptr[x][1] / rgb_bin_size_;
            int32_t index2 = patch_ptr[x][2] / rgb_bin_size_;
            int32_t hist_index = rgb_num_bins_ * (rgb_num_bins_ * index0 + index1) + index2;
            rgb_hist[hist_index] += 1.0f;
        }
    }
}

void RgbHistogram::AccumulateRgbHistogramFromBgra32(const cv::Mat &patch, const cv::Mat &weight,
                                                    float *rgb_hist) const {
    for (int32_t y = 0; y < patch.rows; ++y) {
        const cv::Vec4b *patch_ptr = patch.ptr<cv::Vec4b>(y);
        const float *weight_ptr = weight.ptr<float>(y);
        for (int32_t x = 0; x < patch.cols; ++x) {
            int32_t index0 = patch_ptr[x][0] / rgb_bin_size_;
            int32_t index1 = patch_ptr[x][1] / rgb_bin_size_;
            int32_t index2 = patch_ptr[x][2] / rgb_bin_size_;
            int32_t hist_index = rgb_num_bins_ * (rgb_num_bins_ * index0 + index1) + index2;
            rgb_hist[hist_index] += weight_ptr[x];
        }
    }
}

}; // namespace ot
}; // namespace vas
