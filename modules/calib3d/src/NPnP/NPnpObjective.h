//
// Created by yuval on 6/10/20.
//

#ifndef PNP_USING_EIGEN_LIBRARY_PNPOBJECTIVE_H
#define PNP_USING_EIGEN_LIBRARY_PNPOBJECTIVE_H

#include "../Utils_Npnp/Definitions.h"
#include "NPnpInput.h"
#include <Eigen/Core>
#include <memory>

namespace NPnP {
class PnpObjective {
public:
  RowMatrix<9, 9> M;
  RowMatrix<3, 9> T;
  RowVector<Y_SIZE> b;
  double sum_weights;

  static std::shared_ptr<PnpObjective> init(std::shared_ptr<PnpInput> input);

  static void set_C(ColMatrix<3, 9> &C, Eigen::Vector3d point);
};
} // namespace NPnP

#endif // PNP_USING_EIGEN_LIBRARY_PNPOBJECTIVE_H
