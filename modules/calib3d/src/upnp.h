#ifndef UPNP_H_
#define UPNP_H_

#include "precomp.hpp"
#include "opencv2/core/core_c.h"
#include <iostream>

using namespace std;

class upnp
{
public:
    upnp(const cv::Mat& cameraMatrix, const cv::Mat& opoints, const cv::Mat& ipoints);
    ~upnp();

    void compute_pose(cv::Mat& R, cv::Mat& t);
private:
    template <typename T>
      void init_camera_parameters(const cv::Mat& cameraMatrix)
      {
        uc = cameraMatrix.at<T> (0, 2);
        vc = cameraMatrix.at<T> (1, 2);
        fu = 1;
        fv = 1;
      }
      template <typename OpointType, typename IpointType>
      void init_points(const cv::Mat& opoints, const cv::Mat& ipoints)
      {
          for(int i = 0; i < number_of_correspondences; i++)
          {
        	pws[3 * i    ] = opoints.at<OpointType>(0,i).x;
			pws[3 * i + 1] = opoints.at<OpointType>(0,i).y;
			pws[3 * i + 2] = opoints.at<OpointType>(0,i).z;

			us[2 * i    ] = ipoints.at<IpointType>(0,i).x;
			us[2 * i + 1] = ipoints.at<IpointType>(0,i).y;
          }
      }

      double reprojection_error(const double R[3][3], const double t[3]);
      void choose_control_points();
      void compute_alphas();
      void fill_M(CvMat * M, const int row, const double * alphas, const double u, const double v);
      void compute_ccs(const double * betas, const double * f, const double * ut);
      void compute_pcs(void);

      void solve_for_sign(void);
      void check_positive_eigenvectors(double * ut);

      void find_betas_and_focal_approx_1(const CvMat * Ut, const CvMat * Rho, double * betas, double * efs);
      void find_betas_and_focal_approx_2(const CvMat * Ut, const CvMat * Rho, double * betas, double * efs);
      void qr_solve(CvMat * A, CvMat * b, CvMat * X);

      cv::Mat compute_constraint_distance_2param_6eq_2unk_f_unk(const cv::Mat& M1);
      cv::Mat compute_constraint_distance_3param_6eq_6unk_f_unk(const cv::Mat& M1, const cv::Mat& M2);

      double dot(const double * v1, const double * v2);
      double dotXY(const double * v1, const double * v2);
      double dotZ(const double * v1, const double * v2);
      double dist2(const double * p1, const double * p2);

      void compute_rho(double * rho);
      void compute_L_6x12(const double * ut, double * l_6x12);

      void gauss_newton(const CvMat * L_6x12, const CvMat * Rho, double current_betas[4], double * efs);
      void compute_A_and_b_gauss_newton(const double * l_6x12, const double * rho,
                          const double cb[4], CvMat * A, CvMat * b, double const f);

      double compute_R_and_t(const double * ut, const double * betas, const double * efs,
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

#endif // UPNP_H_
