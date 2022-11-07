#ifndef _RANSAC_OPTIMAL_PNP_H
#define _RANSAC_OPTIMAL_PNP_H

#include "../NPnP/BarrierMethodSettings.h"
#include "../NPnP/NPnpInput.h"
#include "../NPnP/NPnpObjective.h"
#include "../NPnP/NPnpProblemSolver.h"
#include "../Utils_Npnp/Definitions.h"
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <random>
#include <vector>

#include "../FramePose/FramePose.h"

using namespace std;
using namespace cv;
using namespace NPnP;
using Eigen::MatrixXd;

class NewtonPnP {
public:

  // Ransac optimal PnP constructor, initialize options
  NewtonPnP(const cv::Mat &camera_matrix,
            const cv::Mat &dist_coeffs)
      : camera_matrix(camera_matrix), dist_coeffs(dist_coeffs) {
    
    this->pnp = PnpProblemSolver::init();
    this->barrier_method_settings = BarrierMethodSettings::init(1E-14, 20, false, 1000, 50, 0.001);
    camera_matrix.convertTo(K, CV_32F);
    this->invK = K.inv();

    dist_coeffs.convertTo(dist_coeffs_float, CV_32F);

    this->rng = mt19937(chrono::steady_clock::now().time_since_epoch().count());

    this->uc = camera_matrix.at<double>(0, 2);
    this->vc = camera_matrix.at<double>(1, 2);
    this->fu = camera_matrix.at<double>(0, 0);
    this->fv = camera_matrix.at<double>(1, 1);
  }

  // Ransac optimal PnP solver robust to outliers
  bool ransac_solve_pnp(const vector<Point2d> &scene,
                        const vector<Point3d> &obj, vector<bool> &is_inliers,
                        int &inliers_cnt_result, FramePose &frame_pose);

  // PnP caller
  bool optimal_pnp(const vector<Point2d> &scene, const vector<Point3d> &obj,
                   vector<bool> &is_inliers, int &inliers_cnt_result,
                   FramePose &frame_pose);

  // PnP caller
  bool newton_pnp(const vector<Point2d> &scene, const vector<Point3d> &obj, FramePose &frame_pose);

private:
  // Find camera pose from rotation matrix and translation vector
  bool calculate_camera_pose(const Eigen::Matrix3d &rmat,
                             const Eigen::Vector3d &tvec, Point3d &camera_pose);

  // Optimal PnP Solver, calculate rotation matrix and translation vector
  void optimal_pnp_solver(const std::vector<Eigen::Vector3d> &points,
                          const std::vector<Eigen::Vector3d> &lines,
                          const std::vector<double> &weights,
                          const std::vector<int> &indices,
                          Eigen::Matrix3d &rmat, Eigen::Vector3d &tvec);

  // Reproject points
  void reproject_points(const vector<Point3d> &obj_points,
                        const Eigen::Matrix3d &rmat,
                        const Eigen::Vector3d &tvec,
                        vector<Point2d> &image_points);

  // Find inliers by distance from reprojection
  int check_inliers(const vector<Point2d> &scene,
                    const vector<Point2d> &scene_reproj,
                    vector<bool> &is_inliers);

private:
  // const int min_iterations, max_iterations;
  // const int min_inliers;
  // // const double min_inliers_ratio, max_inliers_ratio;
  // const double max_error_threshold;
  const cv::Mat camera_matrix;
  cv::Mat dist_coeffs;
  cv::Mat dist_coeffs_float;
  cv::Mat K;
  cv::Mat invK;
  std::shared_ptr<PnpProblemSolver> pnp;
  std::shared_ptr<BarrierMethodSettings> barrier_method_settings;
  mt19937 rng;

  double uc, vc, fu, fv;
};

#endif
