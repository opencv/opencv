#ifndef _H_FRAME_POSE
#define _H_FRAME_POSE

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

class FramePose {

public:
  FramePose() {}

  FramePose(Eigen::Matrix3d &rmat, Eigen::Vector3d &tvec) {
    set_values(rmat, tvec);
  }

  void set_values(Eigen::Matrix3d &rmat, Eigen::Vector3d &tvec) {
    this->rmat = rmat;
    this->tvec = tvec;

    calculate_values();
  }

  cv::Mat R;
  cv::Mat t;
  cv::Mat proj_mat;
  Eigen::Matrix3d rmat;
  Eigen::Vector3d tvec;
  cv::Point3d cam_pose;
  cv::Mat cam_pose_mat;

private:
  void calculate_values() {
    eigen2cv(rmat, R);
    eigen2cv(tvec, t);
    R.convertTo(R, CV_32F);
    t.convertTo(t, CV_32F);
    auto pos = -(rmat.transpose()) * tvec;
    cam_pose = cv::Point3d(pos[0], pos[1], pos[2]);

    proj_mat = cv::Mat(3, 4, CV_32F);
    R.copyTo(proj_mat.colRange(0, 3));
    t.copyTo(proj_mat.col(3));

    cam_pose_mat = -R.t() * t;
  }
};

#endif
