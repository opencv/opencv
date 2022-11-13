//
// Created by yuval on 6/10/20.
//
#include "Parsing.h"
#include "Definitions.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace NPnP {
std::vector<Eigen::Vector3d> parse_csv_vector_3d(const std::string &csv_path) {
  std::vector<Eigen::Vector3d> data;
  std::ifstream input_file(csv_path);
  std::string line;
  while (input_file >> line) {
    std::stringstream lineStream(line);
    std::vector<double> current_vec;
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
      current_vec.push_back(std::stod(cell));
    }
    data.emplace_back(current_vec[0], current_vec[1], current_vec[2]);
  }

  return std::move(data);
}

Eigen::MatrixXd parse_csv_matrix(const std::string &csv_path) {
  std::vector<std::vector<double>> data;
  std::ifstream input_file(csv_path);
  std::string line;
  while (input_file >> line) {
    std::stringstream lineStream(line);
    std::vector<double> current_vec;
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
      current_vec.push_back(std::stod(cell));
    }
    Eigen::VectorXd vector = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
        current_vec.data(), current_vec.size());
    data.emplace_back(std::move(current_vec));
  }
  Eigen::MatrixXd matrix(data.size(), data[0].size());
  for (int i = 0; i < data.size(); i++)
    for (int j = 0; j < data[i].size(); j++)
      matrix(i, j) = data[i][j];

  return matrix;
}
} // namespace NPnP
