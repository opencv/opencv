/*
 * ModelRegistration.h
 *
 *  Created on: Apr 18, 2014
 *      Author: edgar
 */

#ifndef MODELREGISTRATION_H_
#define MODELREGISTRATION_H_

#include <iostream>
#include <opencv2/core.hpp>

class ModelRegistration
{
public:

  ModelRegistration();
  virtual ~ModelRegistration();

  void setNumMax(int n) { max_registrations_ = n; }

  std::vector<cv::Point2f> get_points2d() const { return list_points2d_; }
  std::vector<cv::Point3f> get_points3d() const { return list_points3d_; }
  int getNumMax() const { return max_registrations_; }
  int getNumRegist() const { return n_registrations_; }

  bool is_registrable() const { return (n_registrations_ < max_registrations_); }
  void registerPoint(const cv::Point2f &point2d, const cv::Point3f &point3d);
  void reset();

private:
/** The current number of registered points */
int n_registrations_;
/** The total number of points to register */
int max_registrations_;
/** The list of 2D points to register the model */
std::vector<cv::Point2f> list_points2d_;
/** The list of 3D points to register the model */
std::vector<cv::Point3f> list_points3d_;
};

#endif /* MODELREGISTRATION_H_ */
