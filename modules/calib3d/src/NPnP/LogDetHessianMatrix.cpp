//
// Created by yuval on 6/17/20.
//

#include "LogDetHessianMatrix.h"

namespace NPnP {
Eigen::CwiseNullaryOp<log_det_hessian_matrix_functor, Eigen::MatrixXd>
make_log_det_hessian_matrix(const inv_mat_15_15_type &inv_mat) {
  return Eigen::MatrixXd::NullaryExpr(A_ROWS, A_ROWS,
                                      log_det_hessian_matrix_functor(inv_mat));
}
} // namespace NPnP
