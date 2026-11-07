/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef VAS_OT_HUNGARIAN_WRAP_HPP
#define VAS_OT_HUNGARIAN_WRAP_HPP

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace vas {
namespace ot {

const int32_t kHungarianModeMinimizeCost = 0;
const int32_t kHungarianModeMaximizeUtil = 1;

typedef struct {
    int32_t num_rows;
    int32_t num_cols;

    std::vector<std::vector<int32_t>> cost;
    std::vector<std::vector<int32_t>> assignment;
} hungarian_problem_t;

class HungarianAlgo {
  public:
    explicit HungarianAlgo(const cv::Mat_<float> &cost_map);
    ~HungarianAlgo();

    cv::Mat_<uint8_t> Solve();

    HungarianAlgo() = delete;
    HungarianAlgo(const HungarianAlgo &) = delete;
    HungarianAlgo(HungarianAlgo &&) = delete;
    HungarianAlgo &operator=(const HungarianAlgo &) = delete;
    HungarianAlgo &operator=(HungarianAlgo &&) = delete;

  protected:
    /*  This method initializes the hungarian_problem structure and the  cost matrices (missing lines or columns are
     *filled with 0). It returns the size of the quadratic(!) assignment matrix.
     **/
    int32_t InitHungarian(int32_t mode);

    // Computes the optimal assignment
    void SolveHungarian();

    // Free the memory allocated by Init
    void FreeHungarian();

    int32_t size_width_;
    int32_t size_height_;

  private:
    const int32_t kHungarianNotAssigned = 0;
    const int32_t kHungarianAssigned = 1;
    const int32_t kIntMax = INT_MAX;

    std::vector<int32_t *> int_cost_map_rows_;
    cv::Mat_<int32_t> int_cost_map_;

    hungarian_problem_t problem_;
};

}; // namespace ot
}; // namespace vas

#endif // VAS_OT_HUNGARIAN_WRAP_HPP
