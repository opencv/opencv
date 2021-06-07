// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef epnp_h
#define epnp_h

#include "precomp.hpp"

namespace cv {

class epnp {
 public:
  epnp(const cv::Mat& cameraMatrix, const cv::Mat& opoints, const cv::Mat& ipoints);
  ~epnp();

  void add_correspondence(const double X, const double Y, const double Z,
              const double u, const double v);

  void compute_pose(cv::Mat& R, cv::Mat& t);
 private:
  epnp(const epnp &); // copy disabled
  epnp& operator=(const epnp &); // assign disabled
  template <typename T>
  void init_camera_parameters(const cv::Mat& cameraMatrix)
  {
    uc = cameraMatrix.at<T> (0, 2);
    vc = cameraMatrix.at<T> (1, 2);
    fu = cameraMatrix.at<T> (0, 0);
    fv = cameraMatrix.at<T> (1, 1);
  }
  template <typename OpointType, typename IpointType>
  void init_points(const cv::Mat& opoints, const cv::Mat& ipoints)
  {
      for(int i = 0; i < number_of_correspondences; i++)
      {
          pws[3 * i    ] = opoints.at<OpointType>(i).x;
          pws[3 * i + 1] = opoints.at<OpointType>(i).y;
          pws[3 * i + 2] = opoints.at<OpointType>(i).z;

          us[2 * i    ] = ipoints.at<IpointType>(i).x*fu + uc;
          us[2 * i + 1] = ipoints.at<IpointType>(i).y*fv + vc;
      }
  }
  double reprojection_error(const double R[3][3], const double t[3]);
  void choose_control_points(void);
  void compute_barycentric_coordinates(void);
  void fill_M(Mat& M, const int row, const double * alphas, const double u, const double v);
  void compute_ccs(const double * betas, const double * ut);
  void compute_pcs(void);

  void solve_for_sign(void);

  void find_betas_approx_1(const Mat& L_6x10, const Mat& Rho, double * betas);
  void find_betas_approx_2(const Mat& L_6x10, const Mat& Rho, double * betas);
  void find_betas_approx_3(const Mat& L_6x10, const Mat& Rho, double * betas);
  void qr_solve(Mat& A, Mat& b, Mat& X);

  double dot(const double * v1, const double * v2);
  double dist2(const double * p1, const double * p2);

  void compute_rho(double * rho);
  void compute_L_6x10(const double * ut, double * l_6x10);

  void gauss_newton(const Mat& L_6x10, const Mat& Rho, double current_betas[4]);
  void compute_A_and_b_gauss_newton(const Mat& L_6x10, const Mat& Rho,
                    const double cb[4], Mat& A, Mat& b);

  double compute_R_and_t(const double * ut, const double * betas,
             double R[3][3], double t[3]);

  void estimate_R_and_t(double R[3][3], double t[3]);

  void copy_R_and_t(const double R_dst[3][3], const double t_dst[3],
            double R_src[3][3], double t_src[3]);


  double uc, vc, fu, fv;

  std::vector<double> pws, us, alphas, pcs;
  int number_of_correspondences;

  double cws[4][3], ccs[4][3];
  int max_nr;
  double * A1, * A2;
};

}

#endif
