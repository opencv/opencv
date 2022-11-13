//
// Created by yuval on 6/10/20.
//

#ifndef PNP_USING_EIGEN_LIBRARY_PNPINPUT_H
#define PNP_USING_EIGEN_LIBRARY_PNPINPUT_H

#include "../Utils_Npnp/Definitions.h"
#include <memory>

namespace NPnP {
class PnpInput {

public:
  std::vector<Eigen::Vector3d> points;
  std::vector<Eigen::Vector3d> lines;
  std::vector<double> weights;
  std::vector<int> indices;
  int indices_amount;
  int points_amount;

  PnpInput(std::vector<Eigen::Vector3d> points,
           std::vector<Eigen::Vector3d> lines, std::vector<double> weights,
           std::vector<int> indices, int indices_amount, int points_amount);

  static std::shared_ptr<PnpInput> parse_input(int argc, char **pString);

  static std::shared_ptr<PnpInput> init(std::vector<Eigen::Vector3d> points,
                                        std::vector<Eigen::Vector3d> lines,
                                        std::vector<double> weights,
                                        std::vector<int> indices);
};
} // namespace NPnP
#endif // PNP_USING_EIGEN_LIBRARY_PNPINPUT_H
