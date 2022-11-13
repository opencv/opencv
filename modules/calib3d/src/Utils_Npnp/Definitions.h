//
// Created by yuval on 6/10/20.
//

#ifndef PNP_USING_EIGEN_LIBRARY_DEFINITIONS_H
#define PNP_USING_EIGEN_LIBRARY_DEFINITIONS_H

#include "Definitions.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace NPnP {
template <int ROWS>
using ColVector = Eigen::Matrix<double, ROWS, 1, Eigen::ColMajor>;
template <int COLS>
using RowVector = Eigen::Matrix<double, 1, COLS, Eigen::RowMajor>;

template <int ROWS, int COLS>
using ColMatrix = Eigen::Matrix<double, ROWS, COLS, Eigen::ColMajor>;
template <int ROWS, int COLS>
using RowMatrix = Eigen::Matrix<double, ROWS, COLS, Eigen::RowMajor>;

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SparseRowMatrix;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SparseColMatrix;

#define Y_SIZE 69
#define NUM_CONSTRAINTS 15
#define M_MATRIX_DIM 15
#define A_ROWS 240
} // namespace NPnP
#endif // PNP_USING_EIGEN_LIBRARY_DEFINITIONS_H
