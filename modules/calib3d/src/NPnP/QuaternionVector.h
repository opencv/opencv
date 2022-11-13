//
// Created by yuval on 7/8/20.
//

#ifndef PNP_SOLVER_QUATERNIONVECTOR_H
#define PNP_SOLVER_QUATERNIONVECTOR_H

#include "../Utils_Npnp/Definitions.h"

namespace NPnP {
class QuaternionVector {
public:
  Eigen::Vector4d quaternions;
  double q1;
  double q2;
  double q3;
  double q4;
  double q1q1;
  double q1q2;
  double q1q3;
  double q1q4;
  double q2q2;
  double q2q3;
  double q2q4;
  double q3q3;
  double q3q4;
  double q4q4;
  double q1q1q1;
  double q1q1q2;
  double q1q1q3;
  double q1q1q4;
  double q1q2q2;
  double q1q2q3;
  double q1q2q4;
  double q1q3q3;
  double q1q3q4;
  double q1q4q4;
  double q2q2q2;
  double q2q2q3;
  double q2q2q4;
  double q2q3q3;
  double q2q3q4;
  double q2q4q4;
  double q3q3q3;
  double q3q3q4;
  double q3q4q4;
  double q4q4q4;
  double q1q1q1q1;
  double q1q1q1q2;
  double q1q1q1q3;
  double q1q1q1q4;
  double q1q1q2q2;
  double q1q1q2q3;
  double q1q1q2q4;
  double q1q1q3q3;
  double q1q1q3q4;
  double q1q1q4q4;
  double q1q2q2q2;
  double q1q2q2q3;
  double q1q2q2q4;
  double q1q2q3q3;
  double q1q2q3q4;
  double q1q2q4q4;
  double q1q3q3q3;
  double q1q3q3q4;
  double q1q3q4q4;
  double q1q4q4q4;
  double q2q2q2q2;
  double q2q2q2q3;
  double q2q2q2q4;
  double q2q2q3q3;
  double q2q2q3q4;
  double q2q2q4q4;
  double q2q3q3q3;
  double q2q3q3q4;
  double q2q3q4q4;
  double q2q4q4q4;
  double q3q3q3q3;
  double q3q3q3q4;
  double q3q3q4q4;
  double q3q4q4q4;
  double q4q4q4q4;

  double q_norm_squared;

  QuaternionVector(const Eigen::Vector4d &quaternions);

  double obj_func(const ColVector<Y_SIZE> &b) const;
  Eigen::Vector4d obj_grad(const ColVector<Y_SIZE> &b) const;
  Eigen::Matrix4d obj_hess(const ColVector<Y_SIZE> &b) const;
};
} // namespace NPnP

#endif // PNP_SOLVER_QUATERNIONVECTOR_H
