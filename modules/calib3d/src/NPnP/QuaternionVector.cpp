//
// Created by yuval on 7/8/20.
//

#include "QuaternionVector.h"

namespace NPnP {
QuaternionVector::QuaternionVector(const Eigen::Vector4d &quaternions)
    : quaternions(quaternions) {
  q1 = quaternions.x();
  q2 = quaternions.y();
  q3 = quaternions.z();
  q4 = quaternions.w();
  q1q1 = q1 * q1;
  q1q2 = q1 * q2;
  q1q3 = q1 * q3;
  q1q4 = q1 * q4;
  q2q2 = q2 * q2;
  q2q3 = q2 * q3;
  q2q4 = q2 * q4;
  q3q3 = q3 * q3;
  q3q4 = q3 * q4;
  q4q4 = q4 * q4;
  q1q1q1 = q1q1 * q1;
  q1q1q2 = q1q1 * q2;
  q1q1q3 = q1q1 * q3;
  q1q1q4 = q1q1 * q4;
  q1q2q2 = q1q2 * q2;
  q1q2q3 = q1q2 * q3;
  q1q2q4 = q1q2 * q4;
  q1q3q3 = q1q3 * q3;
  q1q3q4 = q1q3 * q4;
  q1q4q4 = q1q4 * q4;
  q2q2q2 = q2q2 * q2;
  q2q2q3 = q2q2 * q3;
  q2q2q4 = q2q2 * q4;
  q2q3q3 = q2q3 * q3;
  q2q3q4 = q2q3 * q4;
  q2q4q4 = q2q4 * q4;
  q3q3q3 = q3q3 * q3;
  q3q3q4 = q3q3 * q4;
  q3q4q4 = q3q4 * q4;
  q4q4q4 = q4q4 * q4;
  q1q1q1q1 = q1q1 * q1q1;
  q1q1q1q2 = q1q1 * q1q2;
  q1q1q1q3 = q1q1 * q1q3;
  q1q1q1q4 = q1q1 * q1q4;
  q1q1q2q2 = q1q1 * q2q2;
  q1q1q2q3 = q1q1 * q2q3;
  q1q1q2q4 = q1q1 * q2q4;
  q1q1q3q3 = q1q1 * q3q3;
  q1q1q3q4 = q1q1 * q3q4;
  q1q1q4q4 = q1q1 * q4q4;
  q1q2q2q2 = q1q2 * q2q2;
  q1q2q2q3 = q1q2 * q2q3;
  q1q2q2q4 = q1q2 * q2q4;
  q1q2q3q3 = q1q2 * q3q3;
  q1q2q3q4 = q1q2 * q3q4;
  q1q2q4q4 = q1q2 * q4q4;
  q1q3q3q3 = q1q3 * q3q3;
  q1q3q3q4 = q1q3 * q3q4;
  q1q3q4q4 = q1q3 * q4q4;
  q1q4q4q4 = q1q4 * q4q4;
  q2q2q2q2 = q2q2 * q2q2;
  q2q2q2q3 = q2q2 * q2q3;
  q2q2q2q4 = q2q2 * q2q4;
  q2q2q3q3 = q2q2 * q3q3;
  q2q2q3q4 = q2q2 * q3q4;
  q2q2q4q4 = q2q2 * q4q4;
  q2q3q3q3 = q2q3 * q3q3;
  q2q3q3q4 = q2q3 * q3q4;
  q2q3q4q4 = q2q3 * q4q4;
  q2q4q4q4 = q2q4 * q4q4;
  q3q3q3q3 = q3q3 * q3q3;
  q3q3q3q4 = q3q3 * q3q4;
  q3q3q4q4 = q3q3 * q4q4;
  q3q4q4q4 = q3q4 * q4q4;
  q4q4q4q4 = q4q4 * q4q4;
  q_norm_squared = q1q1 + q2q2 + q3q3 + q4q4;
}

double QuaternionVector::obj_func(const ColVector<Y_SIZE> &b) const {
  double sum = 0.0;
  sum += b[34] * q1q1q1q1;
  sum += b[35] * q1q1q1q2;
  sum += b[36] * q1q1q1q3;
  sum += b[37] * q1q1q1q4;
  sum += b[38] * q1q1q2q2;
  sum += b[39] * q1q1q2q3;
  sum += b[40] * q1q1q2q4;
  sum += b[41] * q1q1q3q3;
  sum += b[42] * q1q1q3q4;
  sum += b[43] * q1q1q4q4;
  sum += b[44] * q1q2q2q2;
  sum += b[45] * q1q2q2q3;
  sum += b[46] * q1q2q2q4;
  sum += b[47] * q1q2q3q3;
  sum += b[48] * q1q2q3q4;
  sum += b[49] * q1q2q4q4;
  sum += b[50] * q1q3q3q3;
  sum += b[51] * q1q3q3q4;
  sum += b[52] * q1q3q4q4;
  sum += b[53] * q1q4q4q4;
  sum += b[54] * q2q2q2q2;
  sum += b[55] * q2q2q2q3;
  sum += b[56] * q2q2q2q4;
  sum += b[57] * q2q2q3q3;
  sum += b[58] * q2q2q3q4;
  sum += b[59] * q2q2q4q4;
  sum += b[60] * q2q3q3q3;
  sum += b[61] * q2q3q3q4;
  sum += b[62] * q2q3q4q4;
  sum += b[63] * q2q4q4q4;
  sum += b[64] * q3q3q3q3;
  sum += b[65] * q3q3q3q4;
  sum += b[66] * q3q3q4q4;
  sum += b[67] * q3q4q4q4;
  sum += b[68] * q4q4q4q4;
  return sum;
}
Eigen::Vector4d QuaternionVector::obj_grad(const ColVector<Y_SIZE> &b) const {
  double q1_grad = 0;
  double q2_grad = 0;
  double q3_grad = 0;
  double q4_grad = 0;
  q1_grad += b[34] * 4 * q1q1q1;
  q1_grad += b[35] * 3 * q1q1q2;
  q1_grad += b[36] * 3 * q1q1q3;
  q1_grad += b[37] * 3 * q1q1q4;
  q1_grad += b[38] * 2 * q1q2q2;
  q1_grad += b[39] * 2 * q1q2q3;
  q1_grad += b[40] * 2 * q1q2q4;
  q1_grad += b[41] * 2 * q1q3q3;
  q1_grad += b[42] * 2 * q1q3q4;
  q1_grad += b[43] * 2 * q1q4q4;
  q1_grad += b[44] * 1 * q2q2q2;
  q1_grad += b[45] * 1 * q2q2q3;
  q1_grad += b[46] * 1 * q2q2q4;
  q1_grad += b[47] * 1 * q2q3q3;
  q1_grad += b[48] * 1 * q2q3q4;
  q1_grad += b[49] * 1 * q2q4q4;
  q1_grad += b[50] * 1 * q3q3q3;
  q1_grad += b[51] * 1 * q3q3q4;
  q1_grad += b[52] * 1 * q3q4q4;
  q1_grad += b[53] * 1 * q4q4q4;

  q2_grad += b[35] * 1 * q1q1q1;
  q2_grad += b[38] * 2 * q1q1q2;
  q2_grad += b[39] * 1 * q1q1q3;
  q2_grad += b[40] * 1 * q1q1q4;
  q2_grad += b[44] * 3 * q1q2q2;
  q2_grad += b[45] * 2 * q1q2q3;
  q2_grad += b[46] * 2 * q1q2q4;
  q2_grad += b[47] * 1 * q1q3q3;
  q2_grad += b[48] * 1 * q1q3q4;
  q2_grad += b[49] * 1 * q1q4q4;
  q2_grad += b[54] * 4 * q2q2q2;
  q2_grad += b[55] * 3 * q2q2q3;
  q2_grad += b[56] * 3 * q2q2q4;
  q2_grad += b[57] * 2 * q2q3q3;
  q2_grad += b[58] * 2 * q2q3q4;
  q2_grad += b[59] * 2 * q2q4q4;
  q2_grad += b[60] * 1 * q3q3q3;
  q2_grad += b[61] * 1 * q3q3q4;
  q2_grad += b[62] * 1 * q3q4q4;
  q2_grad += b[63] * 1 * q4q4q4;

  q3_grad += b[36] * 1 * q1q1q1;
  q3_grad += b[39] * 1 * q1q1q2;
  q3_grad += b[41] * 2 * q1q1q3;
  q3_grad += b[42] * 1 * q1q1q4;
  q3_grad += b[45] * 1 * q1q2q2;
  q3_grad += b[47] * 2 * q1q2q3;
  q3_grad += b[48] * 1 * q1q2q4;
  q3_grad += b[50] * 3 * q1q3q3;
  q3_grad += b[51] * 2 * q1q3q4;
  q3_grad += b[52] * 1 * q1q4q4;
  q3_grad += b[55] * 1 * q2q2q2;
  q3_grad += b[57] * 2 * q2q2q3;
  q3_grad += b[58] * 1 * q2q2q4;
  q3_grad += b[60] * 3 * q2q3q3;
  q3_grad += b[61] * 2 * q2q3q4;
  q3_grad += b[62] * 1 * q2q4q4;
  q3_grad += b[64] * 4 * q3q3q3;
  q3_grad += b[65] * 3 * q3q3q4;
  q3_grad += b[66] * 2 * q3q4q4;
  q3_grad += b[67] * 1 * q4q4q4;

  q4_grad += b[68] * q4q4q4 * 4;
  q4_grad += b[67] * q3q4q4 * 3;
  q4_grad += b[53] * q1q4q4 * 3;
  q4_grad += b[63] * q2q4q4 * 3;
  q4_grad += b[66] * q3q3q4 * 2;
  q4_grad += b[43] * q1q1q4 * 2;
  q4_grad += b[49] * q1q2q4 * 2;
  q4_grad += b[52] * q1q3q4 * 2;
  q4_grad += b[59] * q2q2q4 * 2;
  q4_grad += b[62] * q2q3q4 * 2;
  q4_grad += b[46] * q1q2q2 * 1;
  q4_grad += b[48] * q1q2q3 * 1;
  q4_grad += b[51] * q1q3q3 * 1;
  q4_grad += b[56] * q2q2q2 * 1;
  q4_grad += b[58] * q2q2q3 * 1;
  q4_grad += b[61] * q2q3q3 * 1;
  q4_grad += b[65] * q3q3q3 * 1;
  q4_grad += b[37] * q1q1q1 * 1;
  q4_grad += b[40] * q1q1q2 * 1;
  q4_grad += b[42] * q1q1q3 * 1;

  return std::move(Eigen::Vector4d(q1_grad, q2_grad, q3_grad, q4_grad));
}
Eigen::Matrix4d QuaternionVector::obj_hess(const ColVector<Y_SIZE> &b) const {
  double q1q1_hess = 0;
  double q1q2_hess = 0;
  double q1q3_hess = 0;
  double q1q4_hess = 0;
  double q2q2_hess = 0;
  double q2q3_hess = 0;
  double q2q4_hess = 0;
  double q3q3_hess = 0;
  double q3q4_hess = 0;
  double q4q4_hess = 0;
  q1q1_hess += b[34] * 4 * 3 * q1q1;
  q1q1_hess += b[35] * 3 * 2 * q1q2;
  q1q1_hess += b[36] * 3 * 2 * q1q3;
  q1q1_hess += b[37] * 3 * 2 * q1q4;
  q1q1_hess += b[38] * 2 * 1 * q2q2;
  q1q1_hess += b[39] * 2 * 1 * q2q3;
  q1q1_hess += b[40] * 2 * 1 * q2q4;
  q1q1_hess += b[41] * 2 * 1 * q3q3;
  q1q1_hess += b[42] * 2 * 1 * q3q4;
  q1q1_hess += b[43] * 2 * 1 * q4q4;

  q1q2_hess += b[35] * 3 * 1 * q1q1;
  q1q2_hess += b[38] * 2 * 2 * q1q2;
  q1q2_hess += b[39] * 2 * 1 * q1q3;
  q1q2_hess += b[40] * 2 * 1 * q1q4;
  q1q2_hess += b[44] * 1 * 3 * q2q2;
  q1q2_hess += b[45] * 1 * 2 * q2q3;
  q1q2_hess += b[46] * 1 * 2 * q2q4;
  q1q2_hess += b[47] * 1 * 1 * q3q3;
  q1q2_hess += b[48] * 1 * 1 * q3q4;
  q1q2_hess += b[49] * 1 * 1 * q4q4;

  q1q3_hess += b[36] * 3 * 1 * q1q1;
  q1q3_hess += b[39] * 2 * 1 * q1q2;
  q1q3_hess += b[41] * 2 * 2 * q1q3;
  q1q3_hess += b[42] * 2 * 1 * q1q4;
  q1q3_hess += b[45] * 1 * 1 * q2q2;
  q1q3_hess += b[47] * 1 * 2 * q2q3;
  q1q3_hess += b[48] * 1 * 1 * q2q4;
  q1q3_hess += b[50] * 1 * 3 * q3q3;
  q1q3_hess += b[51] * 1 * 2 * q3q4;
  q1q3_hess += b[52] * 1 * 1 * q4q4;

  q1q4_hess += b[37] * 3 * 1 * q1q1;
  q1q4_hess += b[40] * 2 * 1 * q1q2;
  q1q4_hess += b[42] * 2 * 1 * q1q3;
  q1q4_hess += b[43] * 2 * 2 * q1q4;
  q1q4_hess += b[46] * 1 * 1 * q2q2;
  q1q4_hess += b[48] * 1 * 1 * q2q3;
  q1q4_hess += b[49] * 1 * 2 * q2q4;
  q1q4_hess += b[51] * 1 * 1 * q3q3;
  q1q4_hess += b[52] * 1 * 2 * q3q4;
  q1q4_hess += b[53] * 1 * 3 * q4q4;

  q2q2_hess += b[38] * 2 * 1 * q1q1;
  q2q2_hess += b[44] * 3 * 2 * q1q2;
  q2q2_hess += b[45] * 1 * 2 * q1q3;
  q2q2_hess += b[46] * 2 * 1 * q1q4;
  q2q2_hess += b[54] * 4 * 3 * q2q2;
  q2q2_hess += b[55] * 3 * 2 * q2q3;
  q2q2_hess += b[56] * 3 * 2 * q2q4;
  q2q2_hess += b[57] * 1 * 2 * q3q3;
  q2q2_hess += b[58] * 1 * 2 * q3q4;
  q2q2_hess += b[59] * 1 * 2 * q4q4;

  q2q3_hess += b[39] * 1 * 1 * q1q1;
  q2q3_hess += b[45] * 2 * 1 * q1q2;
  q2q3_hess += b[47] * 1 * 2 * q1q3;
  q2q3_hess += b[48] * 1 * 1 * q1q4;
  q2q3_hess += b[55] * 3 * 1 * q2q2;
  q2q3_hess += b[57] * 2 * 2 * q2q3;
  q2q3_hess += b[58] * 2 * 1 * q2q4;
  q2q3_hess += b[60] * 1 * 3 * q3q3;
  q2q3_hess += b[61] * 1 * 2 * q3q4;
  q2q3_hess += b[62] * 1 * 1 * q4q4;

  q2q4_hess += b[63] * 1 * 3 * q4q4;
  q2q4_hess += b[49] * 1 * 2 * q1q4;
  q2q4_hess += b[62] * 1 * 2 * q3q4;
  q2q4_hess += b[59] * 2 * 2 * q2q4;
  q2q4_hess += b[58] * 2 * 1 * q2q3;
  q2q4_hess += b[46] * 2 * 1 * q1q2;
  q2q4_hess += b[40] * 1 * 1 * q1q1;
  q2q4_hess += b[48] * 1 * 1 * q1q3;
  q2q4_hess += b[61] * 1 * 1 * q3q3;
  q2q4_hess += b[56] * 3 * 1 * q2q2;

  q3q3_hess += b[41] * 1 * 2 * q1q1;
  q3q3_hess += b[47] * 1 * 2 * q1q2;
  q3q3_hess += b[50] * 2 * 3 * q1q3;
  q3q3_hess += b[51] * 1 * 2 * q1q4;
  q3q3_hess += b[57] * 1 * 2 * q2q2;
  q3q3_hess += b[60] * 2 * 3 * q2q3;
  q3q3_hess += b[61] * 1 * 2 * q2q4;
  q3q3_hess += b[64] * 4 * 3 * q3q3;
  q3q3_hess += b[65] * 3 * 2 * q3q4;
  q3q3_hess += b[66] * 2 * 1 * q4q4;

  q3q4_hess += b[42] * 1 * 1 * q1q1;
  q3q4_hess += b[48] * 1 * 1 * q1q2;
  q3q4_hess += b[51] * 2 * 1 * q1q3;
  q3q4_hess += b[52] * 1 * 2 * q1q4;
  q3q4_hess += b[58] * 1 * 1 * q2q2;
  q3q4_hess += b[61] * 2 * 1 * q2q3;
  q3q4_hess += b[62] * 1 * 2 * q2q4;
  q3q4_hess += b[65] * 3 * 1 * q3q3;
  q3q4_hess += b[66] * 2 * 2 * q3q4;
  q3q4_hess += b[67] * 1 * 3 * q4q4;

  q4q4_hess += b[43] * 1 * 2 * q1q1;
  q4q4_hess += b[49] * 1 * 2 * q1q2;
  q4q4_hess += b[52] * 1 * 2 * q1q3;
  q4q4_hess += b[53] * 2 * 3 * q1q4;
  q4q4_hess += b[59] * 1 * 2 * q2q2;
  q4q4_hess += b[62] * 1 * 2 * q2q3;
  q4q4_hess += b[63] * 2 * 3 * q2q4;
  q4q4_hess += b[66] * 1 * 2 * q3q3;
  q4q4_hess += b[67] * 2 * 3 * q3q4;
  q4q4_hess += b[68] * 3 * 4 * q4q4;

  Eigen::Matrix4d result_hess;
  result_hess << q1q1_hess, q1q2_hess, q1q3_hess, q1q4_hess, q1q2_hess,
      q2q2_hess, q2q3_hess, q2q4_hess, q1q3_hess, q2q3_hess, q3q3_hess,
      q3q4_hess, q1q4_hess, q2q4_hess, q3q4_hess, q4q4_hess;
  return std::move(result_hess);
}
} // namespace NPnP
