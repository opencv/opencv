/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "hungarian_wrap.hpp"
#include "../../../common/exception.hpp"

#include <vas/common.hpp>

#include <stdint.h>
#include <stdio.h>

const float kHungarianValueScale = 1024.0f;
namespace vas {
namespace ot {

HungarianAlgo::HungarianAlgo(const cv::Mat_<float> &cost_map)
    : size_width_(cost_map.cols), size_height_(cost_map.rows), int_cost_map_rows_(), int_cost_map_(), problem_() {
    // Convert float 2D cost matrix into int32_t** 2D array with scaling
    int_cost_map_rows_.resize(size_height_, nullptr);
    int_cost_map_.create(size_height_, size_width_);
    for (int32_t r = 0; r < size_height_; ++r) {
        int_cost_map_rows_[r] = reinterpret_cast<int32_t *>(int_cost_map_.ptr(r));

        for (int32_t c = 0; c < size_width_; ++c)
            int_cost_map_(r, c) = static_cast<int32_t>(cost_map(r, c) * kHungarianValueScale);
    }
}

HungarianAlgo::~HungarianAlgo() {
    FreeHungarian();
}

cv::Mat_<uint8_t> HungarianAlgo::Solve() {
    ETHROW(size_height_ > 0 && size_width_ > 0, invalid_argument, "Initialized with invalid cost_map size in Solve");

    // Initialize the gungarian_problem using the cost matrix
    cv::Mat_<uint8_t> assignment_map;
    int32_t matrix_size = InitHungarian(kHungarianModeMinimizeCost);

    // Solve the assignement problem
    SolveHungarian();

    // Copy assignment map
    assignment_map.create(matrix_size, matrix_size);
    for (int32_t r = 0; r < matrix_size; ++r) {
        for (int32_t c = 0; c < matrix_size; ++c) {
            (assignment_map)(r, c) = static_cast<uint8_t>(problem_.assignment[r][c]);
        }
    }

    // Free used memory
    FreeHungarian();

    return assignment_map;
}

// Returns the row size of the assignment matrix
int32_t HungarianAlgo::InitHungarian(int32_t mode) {
    int32_t max_cost = 0;
    int32_t **cost_matrix = &int_cost_map_rows_[0];

    // Is the number of cols  not equal to number of size_height_ : if yes, expand with 0-size_width_ / 0-size_width_
    ETHROW(size_height_ > 0 && size_width_ > 0, invalid_argument,
           "Initialized with invalid cost_map size in InitHungarian");
    problem_.num_rows = (size_width_ < size_height_) ? size_height_ : size_width_;
    problem_.num_cols = problem_.num_rows;

    problem_.cost.resize(problem_.num_rows);
    problem_.assignment.resize(problem_.num_rows);

    for (int32_t i = 0; i < problem_.num_rows; ++i) {
        problem_.cost[i].resize(problem_.num_cols, 0);
        problem_.assignment[i].resize(problem_.num_cols, 0);
    }

    for (int32_t i = 0; i < problem_.num_rows; ++i) {
        for (int32_t j = 0; j < problem_.num_cols; ++j) {
            problem_.cost[i][j] = (i < size_height_ && j < size_width_) ? cost_matrix[i][j] : 0;
            problem_.assignment[i][j] = 0;

            if (max_cost < problem_.cost[i][j])
                max_cost = problem_.cost[i][j];
        }
    }

    if (mode == kHungarianModeMaximizeUtil) {
        for (int32_t i = 0; i < problem_.num_rows; ++i) {
            for (int32_t j = 0; j < problem_.num_cols; ++j) {
                problem_.cost[i][j] = max_cost - problem_.cost[i][j];
            }
        }
    } else if (mode == kHungarianModeMinimizeCost) {
        // Nothing to do
    } else {
        TRACE(" Unknown mode. Mode was set to HUNGARIAN_MODE_MINIMIZE_COST");
    }

    return problem_.num_rows;
}

//
//
void HungarianAlgo::FreeHungarian() {
}

void HungarianAlgo::SolveHungarian() {
    int32_t k = 0;
    int32_t l = 0;
    int32_t unmatched = 0;

    ETHROW(problem_.cost.size() != 0 && problem_.assignment.size() != 0, logic_error, "Unexpected solve");

    std::unique_ptr<int32_t[]> vert_col(new int32_t[problem_.num_rows]);
    std::unique_ptr<int32_t[]> row_unselected(new int32_t[problem_.num_rows]);
    std::unique_ptr<int32_t[]> row_dec(new int32_t[problem_.num_rows]);
    std::unique_ptr<int32_t[]> row_slack(new int32_t[problem_.num_rows]);

    std::unique_ptr<int32_t[]> vert_row(new int32_t[problem_.num_cols]);
    std::unique_ptr<int32_t[]> parent_row(new int32_t[problem_.num_cols]);
    std::unique_ptr<int32_t[]> col_inc(new int32_t[problem_.num_cols]);
    std::unique_ptr<int32_t[]> slack(new int32_t[problem_.num_cols]);

    for (int32_t i = 0; i < problem_.num_rows; ++i) {
        vert_col[i] = 0;
        row_unselected[i] = 0;
        row_dec[i] = 0;
        row_slack[i] = 0;
    }

    for (int32_t i = 0; i < problem_.num_cols; ++i) {
        vert_row[i] = 0;
        parent_row[i] = 0;
        col_inc[i] = 0;
        slack[i] = 0;
    }

    for (int32_t i = 0; i < problem_.num_rows; ++i)
        for (int32_t j = 0; j < problem_.num_cols; ++j)
            problem_.assignment[i][j] = kHungarianNotAssigned;

    // Begin subtract column minima in order to start with lots of zeroes 12
    TRACE(" Using heuristic");

    for (int32_t i = 0; i < problem_.num_cols; ++i) {
        int32_t s = problem_.cost[0][i];
        for (int32_t j = 1; j < problem_.num_rows; ++j) {
            if (problem_.cost[j][i] < s)
                s = problem_.cost[j][i];
        }

        if (s != 0) {
            for (int32_t j = 0; j < problem_.num_rows; ++j)
                problem_.cost[j][i] -= s;
        }
    }
    // End subtract column minima in order to start with lots of zeroes 12

    // Begin initial state 16
    int32_t t = 0;
    for (int32_t i = 0; i < problem_.num_cols; ++i) {
        vert_row[i] = -1;
        parent_row[i] = -1;
        col_inc[i] = 0;
        slack[i] = kIntMax;
    }

    for (k = 0; k < problem_.num_rows; ++k) {
        bool is_row_done = false;
        int32_t s = problem_.cost[k][0];
        for (l = 1; l < problem_.num_cols; ++l) {
            if (problem_.cost[k][l] < s)
                s = problem_.cost[k][l];
        }
        row_dec[k] = s;

        for (l = 0; l < problem_.num_cols; ++l) {
            if (s == problem_.cost[k][l] && vert_row[l] < 0) {
                vert_col[k] = l;
                vert_row[l] = k;
                TRACE(" Matching col (%d)==row (%d)", l, k);

                is_row_done = true;
                break;
            }
        }

        if (is_row_done == true) {
            continue;
        } else {
            vert_col[k] = -1;
            TRACE(" Node %d: unmatched row %d", t, k);
            row_unselected[t++] = k;
        }
    }
    // End initial state 16

    // Begin Hungarian algorithm 18
    if (t == 0)
        goto done;

    unmatched = t;
    while (1) {
        TRACE("Matched %d rows.", problem_.num_rows - t);
        int32_t q = 0;
        while (1) {
            while (q < t) {
                // Begin explore node q of the forest 19
                k = row_unselected[q];
                int32_t s = row_dec[k];
                for (l = 0; l < problem_.num_cols; ++l) {
                    if (slack[l] == 0)
                        continue;

                    int32_t del = problem_.cost[k][l] - s + col_inc[l];
                    if (del >= slack[l])
                        continue;

                    if (del == 0) {
                        if (vert_row[l] < 0)
                            goto leave_break_thru;
                        slack[l] = 0;
                        parent_row[l] = k;
                        TRACE("node %d: row %d==col %d--row %d", t, vert_row[l], l, k);

                        row_unselected[t++] = vert_row[l];
                    } else {
                        slack[l] = del;
                        row_slack[l] = k;
                    }
                }
                // End explore node q of the forest 19

                q++;
            }

            // Begin introduce a new zero into the matrix 21
            int32_t s = kIntMax;
            for (int32_t i = 0; i < problem_.num_cols; ++i) {
                if (slack[i] && slack[i] < s)
                    s = slack[i];
            }

            for (q = 0; q < t; ++q) {
                row_dec[row_unselected[q]] += s;
            }

            for (l = 0; l < problem_.num_cols; ++l) {
                if (slack[l]) {
                    slack[l] -= s;
                    if (slack[l] == 0) {
                        // Begin look at a new zero 22
                        k = row_slack[l];
                        TRACE("Decreasing uncovered elements by %d produces zero at [%d,%d]", s, k, l);
                        if (vert_row[l] < 0) {
                            for (int32_t j = l + 1; j < problem_.num_cols; ++j) {
                                if (slack[j] == 0)
                                    col_inc[j] += s;
                            }

                            goto leave_break_thru;
                        } else {
                            parent_row[l] = k;
                            TRACE("node %d: row %d==col %d--row %d", t, vert_row[l], l, k);
                            row_unselected[t++] = vert_row[l];
                        }
                        // End look at a new zero 22
                    }
                } else {
                    col_inc[l] += s;
                }
            }
            // End introduce a new zero into the matrix 21
        } // while (1)

    leave_break_thru:
        TRACE("Breakthrough at node %d of %d!", q, t);
        while (1) {
            int32_t j = vert_col[k];
            vert_col[k] = l;
            vert_row[l] = k;
            TRACE("rematching col %d==row %d", l, k);
            if (j < 0)
                break;

            k = parent_row[j];
            l = j;
        }

        // End update the matching 20
        if (--unmatched == 0)
            goto done;

        // Begin get ready for another stage 17
        t = 0;
        for (int32_t i = 0; i < problem_.num_cols; ++i) {
            parent_row[i] = -1;
            slack[i] = kIntMax;
        }

        for (int32_t i = 0; i < problem_.num_rows; ++i) {
            if (vert_col[i] < 0) {
                TRACE(" Node %d: unmatched row %d", t, i);
                row_unselected[t++] = i;
            }
        }
        // End get ready for another stage 17
    }

done:
    for (int32_t i = 0; i < problem_.num_rows; ++i) {
        problem_.assignment[i][vert_col[i]] = kHungarianAssigned;
    }

    for (int32_t i = 0; i < problem_.num_rows; ++i) {
        for (int32_t j = 0; j < problem_.num_cols; ++j) {
            problem_.cost[i][j] = problem_.cost[i][j] - row_dec[i] + col_inc[j];
        }
    }
}

}; // namespace ot
}; // namespace vas
