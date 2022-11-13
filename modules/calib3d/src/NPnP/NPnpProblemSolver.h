//
// Created by yuval on 6/10/20.
//

#ifndef PNP_USING_EIGEN_LIBRARY_PNPPROBLEMSOLVER_H
#define PNP_USING_EIGEN_LIBRARY_PNPPROBLEMSOLVER_H

#include "BarrierMethodSettings.h"
#include "../Utils_Npnp/Definitions.h"
#include "DualVar.h"
#include "NPnpInput.h"
#include "NPnpObjective.h"
#include "NPnpResult.h"

namespace NPnP {

class PnpProblemSolver {
public:
  SparseRowMatrix A_rows;
  SparseColMatrix A_cols;
  SparseColMatrix c_vec;
  RowMatrix<NUM_CONSTRAINTS, Y_SIZE> zero_sub_A_rows;
  ColMatrix<NUM_CONSTRAINTS, Y_SIZE> zero_sub_A_cols;
  ColMatrix<NUM_CONSTRAINTS, NUM_CONSTRAINTS> zero_mat_15_15;
  ColVector<Y_SIZE> init_y;
  std::shared_ptr<DualVar> dual_var;

  PnpProblemSolver(const SparseRowMatrix &A_rows, const SparseColMatrix &A_cols,
                   const SparseColMatrix &c_vec,
                   RowMatrix<NUM_CONSTRAINTS, Y_SIZE> zero_sub_A_rows,
                   ColMatrix<NUM_CONSTRAINTS, Y_SIZE> zero_sub_A_cols,
                   ColMatrix<NUM_CONSTRAINTS, NUM_CONSTRAINTS> zero_mat_15_15,
                   ColVector<Y_SIZE> init_y, std::shared_ptr<DualVar> dual_var);

  static std::shared_ptr<PnpProblemSolver> init();

  PnpResult
  solve_pnp(std::shared_ptr<PnpObjective> pnp_objective,
            std::shared_ptr<BarrierMethodSettings> barrier_method_settings);

  void
  line_search(const ColVector<Y_SIZE> &delta_y,
              const std::function<double(const ColVector<Y_SIZE> &)> &obj_func,
              std::shared_ptr<BarrierMethodSettings> barrier_settings);

  void centering_step(
      const std::function<double(const ColVector<Y_SIZE> &)> &obj_func,
      const std::function<const Eigen::VectorXd &(DualVar &)> &gen_grad,
      const std::function<const Eigen::MatrixXd &(DualVar &)> &gen_hess,
      int outer_iteration, bool &is_last_iter, PnpResult &pnp_result,
      std::shared_ptr<PnpObjective> pnp_objective,
      std::shared_ptr<BarrierMethodSettings> barrier_method_settings);

  Eigen::Vector4d
  perform_local_search(std::shared_ptr<PnpObjective> pnp_objective,
                       Eigen::Vector4d quats);
};
} // namespace NPnP

#endif // PNP_USING_EIGEN_LIBRARY_PNPPROBLEMSOLVER_H
