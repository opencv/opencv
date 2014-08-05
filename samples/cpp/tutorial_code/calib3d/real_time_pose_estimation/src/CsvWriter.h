#ifndef CSVWRITER_H
#define	CSVWRITER_H

#include <fstream>
#include <iostream>

#include <opencv2/core/core.hpp>

class CsvWriter {
public:
  CsvWriter(const std::string &path, const std::string &separator = " ");
  ~CsvWriter();
  void writeXYZ(const std::vector<cv::Point3f> &list_points3d);
  void writeUVXYZ(const std::vector<cv::Point3f> &list_points3d, const std::vector<cv::Point2f> &list_points2d, const cv::Mat &descriptors);

private:
  std::ofstream _file;
  std::string _separator;
  bool _isFirstTerm;
};

#endif
