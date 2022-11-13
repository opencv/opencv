//
// Created by yuval on 7/8/20.
//

#include "../Utils_Npnp/Definitions.h"
#include "NPnpProblemSolver.h"
#include "QuaternionVector.h"
#include <functional>
#include <iostream>
#include <memory>

namespace NPnP {

Eigen::Vector4d PnpProblemSolver::perform_local_search(
    std::shared_ptr<PnpObjective> pnp_objective, Eigen::Vector4d quats) {
  const auto eps = 0.0000001;
  auto t = 10.0;
  auto mu = 1.5;

  std::function<double(const QuaternionVector &)> func =
      [&t, pnp_objective](const QuaternionVector &q) -> double {
    double barrier_func = (1 - q.q_norm_squared) * (1 - q.q_norm_squared);
    return q.obj_func(pnp_objective->b) + t * barrier_func;
  };
  std::function<Eigen::Vector4d(const QuaternionVector &)> grad =
      [&t, pnp_objective](const QuaternionVector &q) -> Eigen::Vector4d {
    Eigen::Vector4d barrier_grad = -2 * (1 - q.q_norm_squared) * q.quaternions;
    return q.obj_grad(pnp_objective->b) + t * barrier_grad;
  };
  std::function<Eigen::Matrix4d(const QuaternionVector &)> hess =
      [&t, pnp_objective](const QuaternionVector &q) -> Eigen::Matrix4d {
    Eigen::Matrix4d barrier_hess = q.quaternions * q.quaternions.transpose();
    barrier_hess *= 4;
    barrier_hess -= 2 * (1 - q.q_norm_squared) * Eigen::Matrix4d::Identity();
    barrier_hess *= t;
    barrier_hess += q.obj_hess(pnp_objective->b);
    return barrier_hess;
  };
  QuaternionVector q(quats);
  while (t < 1.0 / eps) {
    Eigen::Matrix4d h = hess(q);
    Eigen::Vector4d g = grad(q);
    Eigen::Matrix4d h_inv = h.inverse();
    auto newton_step = -h_inv * g;

    Eigen::Vector4d new_q = q.quaternions + newton_step;
    new_q.normalize();
    q = QuaternionVector(new_q);
    t *= mu;
  }
  return q.quaternions;
}
} // namespace NPnP
