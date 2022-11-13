//
// Created by yuval on 7/12/20.
//

#include "NPnpResult.h"
#include "../Utils_Npnp/Definitions.h"

namespace NPnP {
double PnpResult::cost() {
  auto rotation_matrix = this->rotation_matrix();
  Eigen::Map<ColVector<9>> rotation_vector(rotation_matrix.data(), 9);
  auto sum_error =
      (rotation_vector.transpose() * pnp_objective->M * rotation_vector)(0, 0);
  return sum_error / pnp_objective->sum_weights;
}

Eigen::Matrix3d PnpResult::rotation_matrix() {
  auto q1 = this->quaternions.x();
  auto q2 = this->quaternions.y();
  auto q3 = this->quaternions.z();
  auto q4 = this->quaternions.w();
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
  Eigen::Matrix3d rotation_matrix;
  rotation_matrix << q11 + q22 - q33 - q44, 2 * q23 - 2 * q14,
      2 * q24 + 2 * q13, 2 * q23 + 2 * q14, q11 + q33 - q22 - q44,
      2 * q34 - 2 * q12, 2 * q24 - 2 * q13, 2 * q34 + 2 * q12,
      q11 + q44 - q22 - q33;
  return rotation_matrix;
}

Eigen::Vector3d PnpResult::translation_vector() {
  auto rotation_matrix = this->rotation_matrix();
  Eigen::Map<ColVector<9>> rotation_vector(rotation_matrix.data(), 9);
  return pnp_objective->T * rotation_vector;
}
} // namespace NPnP
