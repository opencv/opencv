//
// Created by yuval on 6/14/20.
//

#include "../Utils_Npnp/Definitions.h"
#include "DualVar.h"
#include "../Utils_Npnp/GeneralUtils.h"
#include "NPnpProblemSolver.h"
#include "NPnpResult.h"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>

namespace NPnP {

double calc_cost(Eigen::Vector4d quats,
                 std::shared_ptr<PnpObjective> objective) {
  auto q1 = quats.x();
  auto q2 = quats.y();
  auto q3 = quats.z();
  auto q4 = quats.w();
  auto q11 = q1 * q1;
  auto q12 = q1 * q2;
  auto q13 = q1 * q3;
  auto q14 = q1 * q4;
  auto q22 = q2 * q2;
  auto q23 = q2 * q3;
  auto q24 = q2 * q4;
  auto q33 = q3 * q3;
  auto q34 = q3 * q4;
  auto q44 = q4 * q4;
  ColVector<9> rotation;
  rotation << q11 + q22 - q33 - q44, 2 * q23 + 2 * q14, 2 * q24 - 2 * q13,
      2 * q23 - 2 * q14, q11 + q33 - q22 - q44, 2 * q34 + 2 * q12,
      2 * q24 + 2 * q13, 2 * q34 - 2 * q12, q11 + q44 - q22 - q33;
  auto sum_cost = (rotation.transpose() * objective->M * rotation).eval()(0, 0);
  return sum_cost / objective->sum_weights;
}

void print_vec(Eigen::Vector4d vec) {
  std::cout << vec.x() << " ";
  std::cout << vec.y() << " ";
  std::cout << vec.z() << " ";
  std::cout << vec.w() << " ";
}

void PnpProblemSolver::line_search(
    const ColVector<Y_SIZE> &delta_y,
    const std::function<double(const ColVector<Y_SIZE> &)> &obj_func,
    std::shared_ptr<BarrierMethodSettings> barrier_settings) {
  dual_var->B_vec.noalias() = -A_rows * delta_y;
  Eigen::Map<ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM>> B_15_15(
      dual_var->B_vec.data() + NUM_CONSTRAINTS, M_MATRIX_DIM, M_MATRIX_DIM);

  // auto L_15 =
  // dual_var->matrix_15_15.llt().matrixL().toDenseMatrix().inverse().eval();
  // auto eigs = (L_15 * B_15_15 *
  // L_15.transpose()).selfadjointView<Eigen::Upper>().eigenvalues().eval();

  double eigs_pow_minus1[M_MATRIX_DIM];
  Eigen::LLT<Eigen::Ref<ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM>>>
      llt_of_matrix_15_15(dual_var->matrix_15_15);
  dual_var->temp_col_1.noalias() =
      llt_of_matrix_15_15.matrixL().toDenseMatrix();
  dual_var->temp_row_2.noalias() =
      dual_var->temp_col_1.inverse(); // temp_row_2 = L^-1
  dual_var->temp_row_1.noalias() = dual_var->temp_row_2 * B_15_15;
  dual_var->temp_col_2.noalias() =
      dual_var->temp_row_1 * dual_var->temp_row_2.transpose();
  dual_var->temp_vec.noalias() =
      dual_var->temp_col_2.selfadjointView<Eigen::Upper>().eigenvalues();
  auto &eigs = dual_var->temp_vec;
  for (int i = 0; i < M_MATRIX_DIM; i++)
    eigs_pow_minus1[i] = 1.0 / eigs(i, 0);
  auto const_val = obj_func(delta_y);
  std::function<double(double)> line_search_gradient =
      [const_val, &eigs_pow_minus1](double t) {
        double res = const_val;
        for (double eig : eigs_pow_minus1)
          res -= 1.0 / (eig + t);
        return res;
      };
  auto min_val = 1.0;
  for (int i = 0; i < M_MATRIX_DIM; i++) {
    auto eig = eigs(i, 0);
    if (eig < 0.0)
      min_val = min2(min_val, 0.9999 * (-1.0 / eig));
  }
  auto result = find_zero_bin_search(line_search_gradient, 0.0, min_val,
                                     barrier_settings->binary_search_depth);
  dual_var->set_y_vec(dual_var->y_vec + result * delta_y, A_rows, c_vec);
}

void PnpProblemSolver::centering_step(
    const std::function<double(const ColVector<Y_SIZE> &)> &obj_func,
    const std::function<const Eigen::VectorXd &(DualVar &)> &gen_grad,
    const std::function<const Eigen::MatrixXd &(DualVar &)> &gen_hess,
    int outer_iteration, bool &is_last_iter, PnpResult &pnp_result,
    std::shared_ptr<PnpObjective> pnp_objective,
    std::shared_ptr<BarrierMethodSettings> barrier_method_settings) {
  int inner_iteration = 1;
  // static long sum1 = 0, sum2 = 0, sum3=0;

  while (true) {
    // std::cout << inner_iteration << std::endl;

    auto &y = *(this->dual_var);
    auto &mat = y.equation_mat;
    auto &const_vec = y.equation_result;
    auto &res = y.res_vec_84;
    auto &temp = y.temp_vec_69;

    auto &hessian = gen_hess(y);
    auto &gradient = gen_grad(y);

    mat.block(0, 0, Y_SIZE, Y_SIZE) = hessian;
    const_vec.block(0, 0, Y_SIZE, 1) = -gradient;
    const_vec.block(Y_SIZE, 0, NUM_CONSTRAINTS, 1) = y.zero_vars;
    res.noalias() = mat.partialPivLu().solve(const_vec).eval();
    auto delta_y = res.head<Y_SIZE>();
    temp.noalias() = delta_y.transpose() * hessian;
    double distance_to_optimal = temp.dot(delta_y);
    this->line_search(delta_y, obj_func, barrier_method_settings);
    auto &y_vec = dual_var->y_vec;

    if ((inner_iteration >= barrier_method_settings->max_inner_iterations) ||
        (distance_to_optimal < 1.0)) {

      auto current_quats = dual_var->extract_quaternions();
      PnpResult current_result(current_quats, pnp_objective);
      if (current_result.cost() < pnp_result.cost())
        pnp_result = current_result;
      current_result =
          PnpResult(this->perform_local_search(pnp_objective, current_quats),
                    pnp_objective);
      if (current_result.cost() < pnp_result.cost())
        pnp_result = current_result;
      break;
    }
    inner_iteration++;
  }

  // if(barrier_method_settings->verbose)
  //   std::cout << outer_iteration << ") " << inner_iteration << " iterations"
  //   << std::endl;
}

PnpResult PnpProblemSolver::solve_pnp(
    std::shared_ptr<PnpObjective> pnp_objective,
    std::shared_ptr<BarrierMethodSettings> barrier_method_settings) {
  double t = 1000; // 1/t is the barrier weight
  bool is_last_iter = false;
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(Y_SIZE);

  // if (barrier_method_settings->verbose)
  //    std::cout << "Newton Iterations:" << std::endl;

  std::function<double(const ColVector<Y_SIZE> &)> obj_func =
      [&pnp_objective, &t](const ColVector<Y_SIZE> &y_vec) -> double {
    return -t * pnp_objective->b.tail<35>().dot(
                    y_vec.tail<35>()); // first 34 values of b are zeros
  };
  std::function<const Eigen::VectorXd &(DualVar &)> calc_grad =
      [this, &gradient, &pnp_objective,
       &t](DualVar &y) -> const Eigen::VectorXd & {
    gradient.noalias() =
        -t * pnp_objective->b.transpose() + (*(y.cone_barrier_gradient(*this)));
    return gradient;
  };
  std::function<const Eigen::MatrixXd &(DualVar &)> calc_hess =
      [this, &pnp_objective, &t](DualVar &y) -> const Eigen::MatrixXd & {
    return *(y.cone_barrier_hessian(*this));
  };

  dual_var = DualVar::init(init_y, A_rows, zero_sub_A_rows, c_vec);

  PnpResult pnp_result(dual_var->extract_quaternions(), pnp_objective);

  for (int outer_iteration = 1; !is_last_iter; outer_iteration++) {
    is_last_iter = t * barrier_method_settings->miu >
                   1.0 / barrier_method_settings->epsilon;
    this->centering_step(obj_func, calc_grad, calc_hess, outer_iteration,
                         is_last_iter, pnp_result, pnp_objective,
                         barrier_method_settings);
    t *= barrier_method_settings->miu;
  }

  return pnp_result;
}
} // namespace NPnP
