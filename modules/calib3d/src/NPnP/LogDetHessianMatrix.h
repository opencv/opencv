//
// Created by yuval on 6/17/20.
//

#ifndef PNP_USING_EIGEN_LIBRARY_LOGDETHESSIANMATRIX_H
#define PNP_USING_EIGEN_LIBRARY_LOGDETHESSIANMATRIX_H

#include "../Utils_Npnp/Definitions.h"

namespace NPnP {

typedef ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM> inv_mat_15_15_type;

class log_det_hessian_matrix_functor {
  const inv_mat_15_15_type &inv_mat;

public:
  explicit log_det_hessian_matrix_functor(const inv_mat_15_15_type &inv_mat)
      : inv_mat(inv_mat) {}

  double operator()(int row, int col) const {
    if (row <= M_MATRIX_DIM || col <= M_MATRIX_DIM)
      return 0;
    row -= M_MATRIX_DIM;
    col -= M_MATRIX_DIM;
    int row1 = row / M_MATRIX_DIM;
    int col1 = row % M_MATRIX_DIM;
    int row2 = col / M_MATRIX_DIM;
    int col2 = col % M_MATRIX_DIM;
    return inv_mat(row1, col2) * inv_mat(col1, row2);
  }
};
Eigen::CwiseNullaryOp<log_det_hessian_matrix_functor, Eigen::MatrixXd>
make_log_det_hessian_matrix(const inv_mat_15_15_type &inv_mat);
} // namespace NPnP

#endif // PNP_USING_EIGEN_LIBRARY_LOGDETHESSIANMATRIX_H
