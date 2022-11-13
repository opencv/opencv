//
// Created by yuval on 6/10/20.
//

#ifndef PNP_USING_EIGEN_LIBRARY_PARSING_H
#define PNP_USING_EIGEN_LIBRARY_PARSING_H

#include "Definitions.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace NPnP {
template <typename T>
std::vector<T> parse_csv_array(const std::string &csv_path) {
  std::vector<T> data;
  std::ifstream input_file(csv_path);
  T value;
  while (input_file >> value)
    data.push_back(value);

  return std::move(data);
}

std::vector<Eigen::Vector3d> parse_csv_vector_3d(const std::string &csv_path);

Eigen::MatrixXd parse_csv_matrix(const std::string &csv_path);
} // namespace NPnP
#endif // PNP_USING_EIGEN_LIBRARY_PARSING_H
