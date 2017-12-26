/*
 * PnPProblem.h
 *
 *  Created on: Mar 28, 2014
 *      Author: Edgar Riba
 */

#ifndef PNPPROBLEM_H_
#define PNPPROBLEM_H_

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Mesh.h"
#include "ModelRegistration.h"

class PnPProblem
{

public:
  explicit PnPProblem(const double param[]);  // custom constructor
  virtual ~PnPProblem();

  bool backproject2DPoint(const Mesh *mesh, const cv::Point2f &point2d, cv::Point3f &point3d);
  bool intersect_MollerTrumbore(Ray &R, Triangle &T, double *out);
  std::vector<cv::Point2f> verify_points(Mesh *mesh);
  cv::Point2f backproject3DPoint(const cv::Point3f &point3d);
  bool estimatePose(const std::vector<cv::Point3f> &list_points3d, const std::vector<cv::Point2f> &list_points2d, int flags);
  void estimatePoseRANSAC( const std::vector<cv::Point3f> &list_points3d, const std::vector<cv::Point2f> &list_points2d,
                           int flags, cv::Mat &inliers,
                           int iterationsCount, float reprojectionError, double confidence );

  cv::Mat get_A_matrix() const { return _A_matrix; }
  cv::Mat get_R_matrix() const { return _R_matrix; }
  cv::Mat get_t_matrix() const { return _t_matrix; }
  cv::Mat get_P_matrix() const { return _P_matrix; }

  void set_P_matrix( const cv::Mat &R_matrix, const cv::Mat &t_matrix);

private:
  /** The calibration matrix */
  cv::Mat _A_matrix;
  /** The computed rotation matrix */
  cv::Mat _R_matrix;
  /** The computed translation matrix */
  cv::Mat _t_matrix;
  /** The computed projection matrix */
  cv::Mat _P_matrix;
};

// Functions for Möller-Trumbore intersection algorithm
cv::Point3f CROSS(cv::Point3f v1, cv::Point3f v2);
double DOT(cv::Point3f v1, cv::Point3f v2);
cv::Point3f SUB(cv::Point3f v1, cv::Point3f v2);

#endif /* PNPPROBLEM_H_ */
