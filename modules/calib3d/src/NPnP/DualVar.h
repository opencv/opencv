//
// Created by yuval on 6/10/20.
//

#ifndef PNP_USING_EIGEN_LIBRARY_DUALVAR_H
#define PNP_USING_EIGEN_LIBRARY_DUALVAR_H

#include <memory>

#include <Eigen/Core>

#include "BarrierMethodSettings.h"
#include "../Utils_Npnp/Definitions.h"
#include "NPnpObjective.h"


namespace NPnP {
class PnpProblemSolver;

class DualVar {
public:
  ColVector<Y_SIZE> y_vec;
  ColVector<NUM_CONSTRAINTS> zero_vars;
  ColVector<A_ROWS> slack;
  ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM> matrix_15_15;
  ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM> matrix_15_15_inv;
  ColVector<M_MATRIX_DIM * M_MATRIX_DIM> matrix_15_15_inv_vec;
  Eigen::Array<double, M_MATRIX_DIM, 1> eigenvals;

  Eigen::VectorXd gradient_helper_vec;
  Eigen::VectorXd gradient_vec;
  Eigen::MatrixXd hessian_helper_mat;
  Eigen::MatrixXd hessian_mat;
  Eigen::MatrixXd equation_mat;
  Eigen::VectorXd equation_result;

  Eigen::VectorXd res_vec_84;
  Eigen::VectorXd temp_vec_69;

  ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM> temp_col_1;
  ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM> temp_col_2;
  RowMatrix<M_MATRIX_DIM, M_MATRIX_DIM> temp_row_1;
  RowMatrix<M_MATRIX_DIM, M_MATRIX_DIM> temp_row_2;
  ColVector<M_MATRIX_DIM> temp_vec;
  ColVector<A_ROWS> B_vec;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      hessian_multiplication_mat;

  DualVar(ColVector<Y_SIZE> y_vec, ColVector<NUM_CONSTRAINTS> zero_vars,
          ColVector<A_ROWS> slack,
          ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM> matrix_15_15,
          ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM> matrix_15_15_inv,
          ColVector<M_MATRIX_DIM * M_MATRIX_DIM> matrix_15_15_inv_vec,
          Eigen::Array<double, M_MATRIX_DIM, 1> eigenvals,
          const RowMatrix<NUM_CONSTRAINTS, Y_SIZE> &A_sub_rows);

  static std::shared_ptr<DualVar>
  init(ColVector<Y_SIZE> y_vec, const SparseRowMatrix &A_mat_rows,
       const RowMatrix<NUM_CONSTRAINTS, Y_SIZE> &A_sub_rows,
       const SparseColMatrix &c_vec);

  void set_y_vec(ColVector<Y_SIZE> y_vec, const SparseRowMatrix &A_mat_rows,
                 const SparseColMatrix &c_vec);

  Eigen::VectorXd *cone_barrier_gradient(const PnpProblemSolver &pnp);

  Eigen::MatrixXd *cone_barrier_hessian(const PnpProblemSolver &pnp);

  Eigen::Vector4d extract_quaternions();

};
} // namespace PnP

#endif // PNP_USING_EIGEN_LIBRARY_DUALVAR_H
