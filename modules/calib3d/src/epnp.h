#ifndef epnp_h
#define epnp_h

#include "precomp.hpp"
#include "opencv2/core/core_c.h"

namespace cv
{

class epnp {
 public:
  epnp(const cv::Mat& cameraMatrix, const cv::Mat& opoints, const cv::Mat& ipoints);
  ~epnp();

  void add_correspondence(const double X, const double Y, const double Z,
              const double u, const double v);

  void compute_pose(cv::Mat& R, cv::Mat& t);
 private:
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
  void fill_M(CvMat * M, const int row, const double * alphas, const double u, const double v);
  void compute_ccs(const double * betas, const double * ut);
  void compute_pcs(void);

  void solve_for_sign(void);

  void find_betas_approx_1(const CvMat * L_6x10, const CvMat * Rho, double * betas);
  void find_betas_approx_2(const CvMat * L_6x10, const CvMat * Rho, double * betas);
  void find_betas_approx_3(const CvMat * L_6x10, const CvMat * Rho, double * betas);
  void qr_solve(CvMat * A, CvMat * b, CvMat * X);

  double dot(const double * v1, const double * v2);
  double dist2(const double * p1, const double * p2);

  void compute_rho(double * rho);
  void compute_L_6x10(const double * ut, double * l_6x10);

  void gauss_newton(const CvMat * L_6x10, const CvMat * Rho, double current_betas[4]);
  void compute_A_and_b_gauss_newton(const double * l_6x10, const double * rho,
                    const double cb[4], CvMat * A, CvMat * b);

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
