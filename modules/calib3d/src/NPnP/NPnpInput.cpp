//
// Created by yuval on 6/10/20.
//
#include "NPnpInput.h"
#include "../Utils_Npnp/Parsing.h"
#include <iostream>

namespace NPnP {
std::shared_ptr<PnpInput> PnpInput::parse_input(int argc, char **argv) {
  if (argc != 5) {
    std::cout << "Expected 4 command line arguements:" << std::endl;
    std::cout << "\t1) points.csv path" << std::endl;
    std::cout << "\t2) lines.csv path" << std::endl;
    std::cout << "\t3) weights.csv path" << std::endl;
    std::cout << "\t4) indices.csv path" << std::endl;
    exit(1);
  }
  auto points_path = argv[1];
  auto lines_path = argv[2];
  auto weights_path = argv[3];
  auto indices_path = argv[4];

  auto points = parse_csv_vector_3d(points_path);
  auto lines = parse_csv_vector_3d(lines_path);
  auto weights = parse_csv_array<double>(weights_path);
  auto indices = parse_csv_array<int>(indices_path);

  if (points.size() != lines.size()) {
    std::cout << "points amount = " << points.size()
              << " whereas lines amount = " << lines.size() << std::endl;
    exit(1);
  }

  if (indices.size() != weights.size()) {
    std::cout << "weights amount = " << weights.size()
              << " whereas indices amount = " << indices.size() << std::endl;
    exit(1);
  }

  return PnpInput::init(points, lines, weights, indices);
}

PnpInput::PnpInput(std::vector<Eigen::Vector3d> points,
                   std::vector<Eigen::Vector3d> lines,
                   std::vector<double> weights, std::vector<int> indices,
                   int indices_amount, int points_amount)
    : points(std::move(points)), lines(std::move(lines)),
      weights(std::move(weights)), indices(std::move(indices)),
      indices_amount(indices_amount), points_amount(points_amount) {}

std::shared_ptr<PnpInput> PnpInput::init(std::vector<Eigen::Vector3d> points,
                                         std::vector<Eigen::Vector3d> lines,
                                         std::vector<double> weights,
                                         std::vector<int> indices) {
  return std::make_shared<PnpInput>(std::move(points), std::move(lines),
                                    std::move(weights), std::move(indices),
                                    indices.size(), points.size());
}
} // namespace NPnP
