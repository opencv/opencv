//M*//////////////////////////////////////////////////////////////////////////////////////
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
// License Agreement
// For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistribution's of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// * Redistribution's in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// * The name of the copyright holders may not be used to endorse or promote products
// derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/****************************************************************************************\
* Exhaustive Linearization for Robust Camera Pose and Focal Length Estimation.
* Contributed by Edgar Riba
\****************************************************************************************/

#include "precomp.hpp"
#include "upnp.h"
#include <limits>

using namespace std;
using namespace cv;

upnp::upnp(const Mat& cameraMatrix, const Mat& opoints, const Mat& ipoints)
{
  if (cameraMatrix.depth() == CV_32F)
    init_camera_parameters<float>(cameraMatrix);
  else
    init_camera_parameters<double>(cameraMatrix);

  number_of_correspondences = std::max(opoints.checkVector(3, CV_32F), opoints.checkVector(3, CV_64F));

  pws.resize(3 * number_of_correspondences);
  us.resize(2 * number_of_correspondences);

  if (opoints.depth() == ipoints.depth())
  {
    if (opoints.depth() == CV_32F)
      init_points<Point3f,Point2f>(opoints, ipoints);
    else
      init_points<Point3d,Point2d>(opoints, ipoints);
  }
  else if (opoints.depth() == CV_32F)
    init_points<Point3f,Point2d>(opoints, ipoints);
  else
    init_points<Point3d,Point2f>(opoints, ipoints);

  alphas.resize(4 * number_of_correspondences);
  pcs.resize(3 * number_of_correspondences);

  max_nr = 0;
  A1 = NULL;
  A2 = NULL;
}

upnp::~upnp()
{
  if (A1)
    delete[] A1;
  if (A2)
    delete[] A2;
}

double upnp::compute_pose(Mat& R, Mat& t)
{
  choose_control_points();
  compute_alphas();

  Mat * M = new Mat(2 * number_of_correspondences, 12, CV_64F);

  for(int i = 0; i < number_of_correspondences; i++)
  {
    fill_M(M, 2 * i, &alphas[0] + 4 * i, us[2 * i], us[2 * i + 1]);
  }

  double mtm[12 * 12], d[12], ut[12 * 12], vt[12 * 12];
  Mat MtM = Mat(12, 12, CV_64F, mtm);
  Mat D   = Mat(12,  1, CV_64F, d);
  Mat Ut  = Mat(12, 12, CV_64F, ut);
  Mat Vt  = Mat(12, 12, CV_64F, vt);

  MtM = M->t() * (*M);
  SVD::compute(MtM, D, Ut, Vt, SVD::MODIFY_A | SVD::FULL_UV);
  Mat(Ut.t()).copyTo(Ut);
  M->release();

  double l_6x12[6 * 12], rho[6];
  Mat L_6x12 = Mat(6, 12, CV_64F, l_6x12);
  Mat Rho    = Mat(6,  1, CV_64F, rho);

  compute_L_6x12(ut, l_6x12);
  compute_rho(rho);

  double Betas[3][4], Efs[3][1], rep_errors[3];
  double Rs[3][3][3], ts[3][3];

  find_betas_and_focal_approx_1(&Ut, &Rho, Betas[1], Efs[1]);
  gauss_newton(&L_6x12, &Rho, Betas[1], Efs[1]);
  rep_errors[1] = compute_R_and_t(ut, Betas[1], Rs[1], ts[1]);

  find_betas_and_focal_approx_2(&Ut, &Rho, Betas[2], Efs[2]);
  gauss_newton(&L_6x12, &Rho, Betas[2], Efs[2]);
  rep_errors[2] = compute_R_and_t(ut, Betas[2], Rs[2], ts[2]);

  int N = 1;
  if (rep_errors[2] < rep_errors[1]) N = 2;

  Mat(3, 1, CV_64F, ts[N]).copyTo(t);
  Mat(3, 3, CV_64F, Rs[N]).copyTo(R);
  fu = fv = Efs[N][0];

  return fu;
}

void upnp::copy_R_and_t(const double R_src[3][3], const double t_src[3],
     double R_dst[3][3], double t_dst[3])
{
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      R_dst[i][j] = R_src[i][j];
    t_dst[i] = t_src[i];
  }
}

void upnp::estimate_R_and_t(double R[3][3], double t[3])
{
  double pc0[3], pw0[3];

  pc0[0] = pc0[1] = pc0[2] = 0.0;
  pw0[0] = pw0[1] = pw0[2] = 0.0;

  for(int i = 0; i < number_of_correspondences; i++) {
    const double * pc = &pcs[3 * i];
    const double * pw = &pws[3 * i];

    for(int j = 0; j < 3; j++) {
      pc0[j] += pc[j];
      pw0[j] += pw[j];
    }
  }
  for(int j = 0; j < 3; j++) {
    pc0[j] /= number_of_correspondences;
    pw0[j] /= number_of_correspondences;
  }

  double abt[3 * 3], abt_d[3], abt_u[3 * 3], abt_v[3 * 3];
  Mat ABt   = Mat(3, 3, CV_64F, abt);
  Mat ABt_D = Mat(3, 1, CV_64F, abt_d);
  Mat ABt_U = Mat(3, 3, CV_64F, abt_u);
  Mat ABt_V = Mat(3, 3, CV_64F, abt_v);

  ABt.setTo(0.0);
  for(int i = 0; i < number_of_correspondences; i++) {
    double * pc = &pcs[3 * i];
    double * pw = &pws[3 * i];

    for(int j = 0; j < 3; j++) {
      abt[3 * j    ] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
      abt[3 * j + 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
      abt[3 * j + 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
    }
  }

  SVD::compute(ABt, ABt_D, ABt_U, ABt_V, SVD::MODIFY_A);
  Mat(ABt_V.t()).copyTo(ABt_V);

  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      R[i][j] = dot(abt_u + 3 * i, abt_v + 3 * j);

  const double det =
    R[0][0] * R[1][1] * R[2][2] + R[0][1] * R[1][2] * R[2][0] + R[0][2] * R[1][0] * R[2][1] -
    R[0][2] * R[1][1] * R[2][0] - R[0][1] * R[1][0] * R[2][2] - R[0][0] * R[1][2] * R[2][1];

  if (det < 0) {
    R[2][0] = -R[2][0];
    R[2][1] = -R[2][1];
    R[2][2] = -R[2][2];
  }

  t[0] = pc0[0] - dot(R[0], pw0);
  t[1] = pc0[1] - dot(R[1], pw0);
  t[2] = pc0[2] - dot(R[2], pw0);
}

void upnp::solve_for_sign(void)
{
  if (pcs[2] < 0.0) {
    for(int i = 0; i < 4; i++)
      for(int j = 0; j < 3; j++)
        ccs[i][j] = -ccs[i][j];

    for(int i = 0; i < number_of_correspondences; i++) {
      pcs[3 * i    ] = -pcs[3 * i];
      pcs[3 * i + 1] = -pcs[3 * i + 1];
      pcs[3 * i + 2] = -pcs[3 * i + 2];
    }
  }
}

double upnp::compute_R_and_t(const double * ut, const double * betas,
         double R[3][3], double t[3])
{
  compute_ccs(betas, ut);
  compute_pcs();

  solve_for_sign();

  estimate_R_and_t(R, t);

  return reprojection_error(R, t);
}

double upnp::reprojection_error(const double R[3][3], const double t[3])
{
  double sum2 = 0.0;

  for(int i = 0; i < number_of_correspondences; i++) {
    double * pw = &pws[3 * i];
    double Xc = dot(R[0], pw) + t[0];
    double Yc = dot(R[1], pw) + t[1];
    double inv_Zc = 1.0 / (dot(R[2], pw) + t[2]);
    double ue = uc + fu * Xc * inv_Zc;
    double ve = vc + fv * Yc * inv_Zc;
    double u = us[2 * i], v = us[2 * i + 1];

    sum2 += sqrt( (u - ue) * (u - ue) + (v - ve) * (v - ve) );
  }

  return sum2 / number_of_correspondences;
}

void upnp::choose_control_points()
{
    for (int i = 0; i < 4; ++i)
      cws[i][0] = cws[i][1] = cws[i][2] = 0.0;
    cws[0][0] = cws[1][1] = cws[2][2] = 1.0;
}

void upnp::compute_alphas()
{
    Mat CC = Mat(4, 3, CV_64F, &cws);
    Mat PC = Mat(number_of_correspondences, 3, CV_64F, &pws[0]);
    Mat ALPHAS = Mat(number_of_correspondences, 4, CV_64F, &alphas[0]);

    Mat CC_ = CC.clone().t();
    Mat PC_ = PC.clone().t();

    Mat row14 = Mat::ones(1, 4, CV_64F);
    Mat row1n = Mat::ones(1, number_of_correspondences, CV_64F);

    CC_.push_back(row14);
    PC_.push_back(row1n);

    ALPHAS = Mat( CC_.inv() * PC_ ).t();
}

void upnp::fill_M(Mat * M, const int row, const double * as, const double u, const double v)
{
  double * M1 = M->ptr<double>(row);
  double * M2 = M1 + 12;

  for(int i = 0; i < 4; i++) {
    M1[3 * i    ] = as[i] * fu;
    M1[3 * i + 1] = 0.0;
    M1[3 * i + 2] = as[i] * (uc - u);

    M2[3 * i    ] = 0.0;
    M2[3 * i + 1] = as[i] * fv;
    M2[3 * i + 2] = as[i] * (vc - v);
  }
}

void upnp::compute_ccs(const double * betas, const double * ut)
{
    for(int i = 0; i < 4; ++i)
      ccs[i][0] = ccs[i][1] = ccs[i][2] = 0.0;

    int N = 4;
    for(int i = 0; i < N; ++i) {
      const double * v = ut + 12 * (9 + i);
      for(int j = 0; j < 4; ++j)
        for(int k = 0; k < 3; ++k)
          ccs[j][k] += betas[i] * v[3 * j + k];
    }

    for (int i = 0; i < 4; ++i) ccs[i][2] *= fu;
}

void upnp::compute_pcs(void)
{
  for(int i = 0; i < number_of_correspondences; i++) {
    double * a = &alphas[0] + 4 * i;
    double * pc = &pcs[0] + 3 * i;

    for(int j = 0; j < 3; j++)
      pc[j] = a[0] * ccs[0][j] + a[1] * ccs[1][j] + a[2] * ccs[2][j] + a[3] * ccs[3][j];
  }
}

void upnp::find_betas_and_focal_approx_1(Mat * Ut, Mat * Rho, double * betas, double * efs)
{
  Mat Kmf1 = Mat(12, 1, CV_64F, Ut->ptr<double>(11));
  Mat dsq = Mat(6, 1, CV_64F, Rho->ptr<double>(0));

  Mat D = compute_constraint_distance_2param_6eq_2unk_f_unk( Kmf1 );
  Mat Dt = D.t();

  Mat A = Dt * D;
  Mat b = Dt * dsq;

  Mat x = Mat(2, 1, CV_64F);
  solve(A, b, x);

  betas[0] = sqrt( abs( x.at<double>(0) ) );
  betas[1] = betas[2] = betas[3] = 0.0;

  efs[0] = sqrt( abs( x.at<double>(1) ) ) / betas[0];
}

void upnp::find_betas_and_focal_approx_2(Mat * Ut, Mat * Rho, double * betas, double * efs)
{
  double u[12*12];
  Mat U = Mat(12, 12, CV_64F, u);
  Ut->copyTo(U);

  Mat Kmf1 = Mat(12, 1, CV_64F, Ut->ptr<double>(10));
  Mat Kmf2 = Mat(12, 1, CV_64F, Ut->ptr<double>(11));
  Mat dsq = Mat(6, 1, CV_64F, Rho->ptr<double>(0));

  Mat D = compute_constraint_distance_3param_6eq_6unk_f_unk( Kmf1, Kmf2 );

  Mat A = D;
  Mat b = dsq;

  double x[6];
  Mat X = Mat(6, 1, CV_64F, x);

  solve(A, b, X, DECOMP_QR);

  double solutions[18][3];
  generate_all_possible_solutions_for_f_unk(x, solutions);

  // find solution with minimum reprojection error
  double min_error = std::numeric_limits<double>::max();
  int min_sol = 0;
  for (int i = 0; i < 18; ++i) {

    betas[3] = solutions[i][0];
    betas[2] = solutions[i][1];
    betas[1] = betas[0] = 0.0;
    fu = fv = solutions[i][2];

    double Rs[3][3], ts[3];
    double error_i = compute_R_and_t( u, betas, Rs, ts);

    if( error_i < min_error)
    {
      min_error = error_i;
      min_sol = i;
    }
}

  betas[0] = solutions[min_sol][0];
  betas[1] = solutions[min_sol][1];
  betas[2] = betas[3] = 0.0;

  efs[0] = solutions[min_sol][2];
}

Mat upnp::compute_constraint_distance_2param_6eq_2unk_f_unk(const Mat& M1)
{
  Mat P = Mat(6, 2, CV_64F);

  double m[13];
  for (int i = 1; i < 13; ++i) m[i] = *M1.ptr<double>(i-1);

  double t1 = pow( m[4], 2 );
  double t4 = pow( m[1], 2 );
  double t5 = pow( m[5], 2 );
  double t8 = pow( m[2], 2 );
  double t10 = pow( m[6], 2 );
  double t13 = pow( m[3], 2 );
  double t15 = pow( m[7], 2 );
  double t18 = pow( m[8], 2 );
  double t22 = pow( m[9], 2 );
  double t26 = pow( m[10], 2 );
  double t29 = pow( m[11], 2 );
  double t33 = pow( m[12], 2 );

  *P.ptr<double>(0,0) = t1 - 2 * m[4] * m[1] + t4 + t5 - 2 * m[5] * m[2] + t8;
  *P.ptr<double>(0,1) = t10 - 2 * m[6] * m[3] + t13;
  *P.ptr<double>(1,0) = t15 - 2 * m[7] * m[1] + t4 + t18 - 2 * m[8] * m[2] + t8;
  *P.ptr<double>(1,1) = t22 - 2 * m[9] * m[3] + t13;
  *P.ptr<double>(2,0) = t26 - 2 * m[10] * m[1] + t4 + t29 - 2 * m[11] * m[2] + t8;
  *P.ptr<double>(2,1) = t33 - 2 * m[12] * m[3] + t13;
  *P.ptr<double>(3,0) = t15 - 2 * m[7] * m[4] + t1 + t18 - 2 * m[8] * m[5] + t5;
  *P.ptr<double>(3,1) = t22 - 2 * m[9] * m[6] + t10;
  *P.ptr<double>(4,0) = t26 - 2 * m[10] * m[4] + t1 + t29 - 2 * m[11] * m[5] + t5;
  *P.ptr<double>(4,1) = t33 - 2 * m[12] * m[6] + t10;
  *P.ptr<double>(5,0) = t26 - 2 * m[10] * m[7] + t15 + t29 - 2 * m[11] * m[8] + t18;
  *P.ptr<double>(5,1) = t33 - 2 * m[12] * m[9] + t22;

  return P;
}

Mat upnp::compute_constraint_distance_3param_6eq_6unk_f_unk(const Mat& M1, const Mat& M2)
{
  Mat P = Mat(6, 6, CV_64F);

  double m[3][13];
  for (int i = 1; i < 13; ++i)
  {
    m[1][i] = *M1.ptr<double>(i-1);
    m[2][i] = *M2.ptr<double>(i-1);
  }

  double t1 = pow( m[1][4], 2 );
  double t2 = pow( m[1][1], 2 );
  double t7 = pow( m[1][5], 2 );
  double t8 = pow( m[1][2], 2 );
  double t11 = m[1][1] * m[2][1];
  double t12 = m[1][5] * m[2][5];
  double t15 = m[1][2] * m[2][2];
  double t16 = m[1][4] * m[2][4];
  double t19 = pow( m[2][4], 2 );
  double t22 = pow( m[2][2], 2 );
  double t23 = pow( m[2][1], 2 );
  double t24 = pow( m[2][5], 2 );
  double t28 = pow( m[1][6], 2 );
  double t29 = pow( m[1][3], 2 );
  double t34 = pow( m[1][3], 2 );
  double t36 = m[1][6] * m[2][6];
  double t40 = pow( m[2][6], 2 );
  double t41 = pow( m[2][3], 2 );
  double t47 = pow( m[1][7], 2 );
  double t48 = pow( m[1][8], 2 );
  double t52 = m[1][7] * m[2][7];
  double t55 = m[1][8] * m[2][8];
  double t59 = pow( m[2][8], 2 );
  double t62 = pow( m[2][7], 2 );
  double t64 = pow( m[1][9], 2 );
  double t68 = m[1][9] * m[2][9];
  double t74 = pow( m[2][9], 2 );
  double t78 = pow( m[1][10], 2 );
  double t79 = pow( m[1][11], 2 );
  double t84 = m[1][10] * m[2][10];
  double t87 = m[1][11] * m[2][11];
  double t90 = pow( m[2][10], 2 );
  double t95 = pow( m[2][11], 2 );
  double t99 = pow( m[1][12], 2 );
  double t101 = m[1][12] * m[2][12];
  double t105 = pow( m[2][12], 2 );

  *P.ptr<double>(0,0) = t1 + t2 - 2 * m[1][4] * m[1][1] - 2 * m[1][5] * m[1][2] + t7 + t8;
  *P.ptr<double>(0,1) = -2 * m[2][4] * m[1][1] + 2 * t11 + 2 * t12 - 2 * m[1][4] * m[2][1] - 2 * m[2][5] * m[1][2] + 2 * t15 + 2 * t16 - 2 * m[1][5] * m[2][2];
  *P.ptr<double>(0,2) = t19 - 2 * m[2][4] * m[2][1] + t22 + t23 + t24 - 2 * m[2][5] * m[2][2];
  *P.ptr<double>(0,3) = t28 + t29 - 2 * m[1][6] * m[1][3];
  *P.ptr<double>(0,4) = -2 * m[2][6] * m[1][3] + 2 * t34 - 2 * m[1][6] * m[2][3] + 2 * t36;
  *P.ptr<double>(0,5) = -2 * m[2][6] * m[2][3] + t40 + t41;

  *P.ptr<double>(1,0) = t8 - 2 * m[1][8] * m[1][2] - 2 * m[1][7] * m[1][1] + t47 + t48 + t2;
  *P.ptr<double>(1,1) = 2 * t15 - 2 * m[1][8] * m[2][2] - 2 * m[2][8] * m[1][2] + 2 * t52 - 2 * m[1][7] * m[2][1] - 2 * m[2][7] * m[1][1] + 2 * t55 + 2 * t11;
  *P.ptr<double>(1,2) = -2 * m[2][8] * m[2][2] + t22 + t23 + t59 - 2 * m[2][7] * m[2][1] + t62;
  *P.ptr<double>(1,3) = t29 + t64 - 2 * m[1][9] * m[1][3];
  *P.ptr<double>(1,4) = 2 * t34 + 2 * t68 - 2 * m[2][9] * m[1][3] - 2 * m[1][9] * m[2][3];
  *P.ptr<double>(1,5) = -2 * m[2][9] * m[2][3] + t74 + t41;

  *P.ptr<double>(2,0) = -2 * m[1][11] * m[1][2] + t2 + t8 + t78 + t79 - 2 * m[1][10] * m[1][1];
  *P.ptr<double>(2,1) = 2 * t15 - 2 * m[1][11] * m[2][2] + 2 * t84 - 2 * m[1][10] * m[2][1] - 2 * m[2][10] * m[1][1] + 2 * t87 - 2 * m[2][11] * m[1][2]+ 2 * t11;
  *P.ptr<double>(2,2) = t90 + t22 - 2 * m[2][10] * m[2][1] + t23 - 2 * m[2][11] * m[2][2] + t95;
  *P.ptr<double>(2,3) = -2 * m[1][12] * m[1][3] + t99 + t29;
  *P.ptr<double>(2,4) = 2 * t34 + 2 * t101 - 2 * m[2][12] * m[1][3] - 2 * m[1][12] * m[2][3];
  *P.ptr<double>(2,5) = t41 + t105 - 2 * m[2][12] * m[2][3];

  *P.ptr<double>(3,0) = t48 + t1 - 2 * m[1][8] * m[1][5] + t7 - 2 * m[1][7] * m[1][4] + t47;
  *P.ptr<double>(3,1) = 2 * t16 - 2 * m[1][7] * m[2][4] + 2 * t55 + 2 * t52 - 2 * m[1][8] * m[2][5] - 2 * m[2][8] * m[1][5] - 2 * m[2][7] * m[1][4] + 2 * t12;
  *P.ptr<double>(3,2) = t24 - 2 * m[2][8] * m[2][5] + t19 - 2 * m[2][7] * m[2][4] + t62 + t59;
  *P.ptr<double>(3,3) = -2 * m[1][9] * m[1][6] + t64 + t28;
  *P.ptr<double>(3,4) = 2 * t68 + 2 * t36 - 2 * m[2][9] * m[1][6] - 2 * m[1][9] * m[2][6];
  *P.ptr<double>(3,5) = t40 + t74 - 2 * m[2][9] * m[2][6];

  *P.ptr<double>(4,0) = t1 - 2 * m[1][10] * m[1][4] + t7 + t78 + t79 - 2 * m[1][11] * m[1][5];
  *P.ptr<double>(4,1) = 2 * t84 - 2 * m[1][11] * m[2][5] - 2 * m[1][10] * m[2][4] + 2 * t16 - 2 * m[2][11] * m[1][5] + 2 * t87 - 2 * m[2][10] * m[1][4] + 2 * t12;
  *P.ptr<double>(4,2) = t19 + t24 - 2 * m[2][10] * m[2][4] - 2 * m[2][11] * m[2][5] + t95 + t90;
  *P.ptr<double>(4,3) = t28 - 2 * m[1][12] * m[1][6] + t99;
  *P.ptr<double>(4,4) = 2 * t101 + 2 * t36 - 2 * m[2][12] * m[1][6] - 2 * m[1][12] * m[2][6];
  *P.ptr<double>(4,5) = t105 - 2 * m[2][12] * m[2][6] + t40;

  *P.ptr<double>(5,0) = -2 * m[1][10] * m[1][7] + t47 + t48 + t78 + t79 - 2 * m[1][11] * m[1][8];
  *P.ptr<double>(5,1) = 2 * t84 + 2 * t87 - 2 * m[2][11] * m[1][8] - 2 * m[1][10] * m[2][7] - 2 * m[2][10] * m[1][7] + 2 * t55 + 2 * t52 - 2 * m[1][11] * m[2][8];
  *P.ptr<double>(5,2) = -2 * m[2][10] * m[2][7] - 2 * m[2][11] * m[2][8] + t62 + t59 + t90 + t95;
  *P.ptr<double>(5,3) = t64 - 2 * m[1][12] * m[1][9] + t99;
  *P.ptr<double>(5,4) = 2 * t68 - 2 * m[2][12] * m[1][9] - 2 * m[1][12] * m[2][9] + 2 * t101;
  *P.ptr<double>(5,5) = t105 - 2 * m[2][12] * m[2][9] + t74;

  return P;
}

void upnp::generate_all_possible_solutions_for_f_unk(const double betas[5], double solutions[18][3])
{
  int matrix_to_resolve[18][9] = {
    { 2, 0, 0, 1, 1, 0, 2, 0, 2 }, { 2, 0, 0, 1, 1, 0, 1, 1, 2 },
    { 2, 0, 0, 1, 1, 0, 0, 2, 2 }, { 2, 0, 0, 0, 2, 0, 2, 0, 2 },
    { 2, 0, 0, 0, 2, 0, 1, 1, 2 }, { 2, 0, 0, 0, 2, 0, 0, 2, 2 },
    { 2, 0, 0, 2, 0, 2, 1, 1, 2 }, { 2, 0, 0, 2, 0, 2, 0, 2, 2 },
    { 2, 0, 0, 1, 1, 2, 0, 2, 2 }, { 1, 1, 0, 0, 2, 0, 2, 0, 2 },
    { 1, 1, 0, 0, 2, 0, 1, 1, 2 }, { 1, 1, 0, 2, 0, 2, 0, 2, 2 },
    { 1, 1, 0, 2, 0, 2, 1, 1, 2 }, { 1, 1, 0, 2, 0, 2, 0, 2, 2 },
    { 1, 1, 0, 1, 1, 2, 0, 2, 2 }, { 0, 2, 0, 2, 0, 2, 1, 1, 2 },
    { 0, 2, 0, 2, 0, 2, 0, 2, 2 }, { 0, 2, 0, 1, 1, 2, 0, 2, 2 }
  };

  int combination[18][3] = {
    { 1, 2, 4 }, { 1, 2, 5 }, { 1, 2, 6 }, { 1, 3, 4 },
    { 1, 3, 5 }, { 1, 3, 6 }, { 1, 4, 5 }, { 1, 4, 6 },
    { 1, 5, 6 }, { 2, 3, 4 }, { 2, 3, 5 }, { 2, 3, 6 },
    { 2, 4, 5 }, { 2, 4, 6 }, { 2, 5, 6 }, { 3, 4, 5 },
    { 3, 4, 6 }, { 3, 5, 6 }
  };

  for (int i = 0; i < 18; ++i) {
    double matrix[9], independent_term[3];
    Mat M = Mat(3, 3, CV_64F, matrix);
    Mat I = Mat(3, 1, CV_64F, independent_term);
    Mat S = Mat(1, 3, CV_64F);

    for (int j = 0; j < 9; ++j) matrix[j] = (double)matrix_to_resolve[i][j];

    independent_term[0] = log( abs( betas[ combination[i][0]-1 ] ) );
    independent_term[1] = log( abs( betas[ combination[i][1]-1 ] ) );
    independent_term[2] = log( abs( betas[ combination[i][2]-1 ] ) );

    exp( Mat(M.inv() * I), S);

    solutions[i][0] = S.at<double>(0);
    solutions[i][1] = S.at<double>(1) * sign( betas[1] );
    solutions[i][2] = abs( S.at<double>(2) );
  }
}

void upnp::gauss_newton(const Mat * L_6x12, const Mat * Rho, double betas[4], double * f)
{
  const int iterations_number = 50;

  double a[6*4], b[6], x[4];
  Mat * A = new Mat(6, 4, CV_64F, a);
  Mat * B = new Mat(6, 1, CV_64F, b);
  Mat * X = new Mat(4, 1, CV_64F, x);

  for(int k = 0; k < iterations_number; k++)
  {
    compute_A_and_b_gauss_newton(L_6x12->ptr<double>(0), Rho->ptr<double>(0), betas, A, B, f[0]);
    qr_solve(A, B, X);
    for(int i = 0; i < 3; i++)
      betas[i] += x[i];
    f[0] += x[3];
  }

  if (f[0] < 0) f[0] = -f[0];
    fu = fv = f[0];

}

void upnp::compute_A_and_b_gauss_newton(const double * l_6x12, const double * rho,
        const double betas[4], Mat * A, Mat * b, double const f)
{

  for(int i = 0; i < 6; i++) {
    const double * rowL = l_6x12 + i * 12;
    double * rowA = A->ptr<double>(i);

    rowA[0] = 2 * rowL[0] * betas[0] +    rowL[1] * betas[1] +    rowL[2] * betas[2] + f*f * ( 2 * rowL[6]*betas[0] +    rowL[7]*betas[1]  +    rowL[8]*betas[2] );
    rowA[1] =    rowL[1] * betas[0] + 2 * rowL[3] * betas[1] +    rowL[4] * betas[2] + f*f * (    rowL[7]*betas[0] + 2 * rowL[9]*betas[1]  +    rowL[10]*betas[2] );
    rowA[2] =    rowL[2] * betas[0] +    rowL[4] * betas[1] + 2 * rowL[5] * betas[2] + f*f * (    rowL[8]*betas[0] +    rowL[10]*betas[1] + 2 * rowL[11]*betas[2] );
    rowA[3] = 2*f * ( rowL[6]*betas[0]*betas[0] + rowL[7]*betas[0]*betas[1] + rowL[8]*betas[0]*betas[2] + rowL[9]*betas[1]*betas[1] + rowL[10]*betas[1]*betas[2] + rowL[11]*betas[2]*betas[2] ) ;

    *b->ptr<double>(i) = rho[i] -
    (
      rowL[0] * betas[0] * betas[0] +
      rowL[1] * betas[0] * betas[1] +
      rowL[2] * betas[0] * betas[2] +
      rowL[3] * betas[1] * betas[1] +
      rowL[4] * betas[1] * betas[2] +
      rowL[5] * betas[2] * betas[2] +
      f*f * rowL[6] * betas[0] * betas[0] +
      f*f * rowL[7] * betas[0] * betas[1] +
      f*f * rowL[8] * betas[0] * betas[2] +
      f*f * rowL[9] * betas[1] * betas[1] +
      f*f * rowL[10] * betas[1] * betas[2] +
      f*f * rowL[11] * betas[2] * betas[2]
     );
  }
}

void upnp::compute_L_6x12(const double * ut, double * l_6x12)
{
  const double * v[3];

  v[0] = ut + 12 * 9;
  v[1] = ut + 12 * 10;
  v[2] = ut + 12 * 11;

  double dv[3][6][3];

  for(int i = 0; i < 3; i++) {
    int a = 0, b = 1;
    for(int j = 0; j < 6; j++) {
      dv[i][j][0] = v[i][3 * a    ] - v[i][3 * b];
      dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
      dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];

      b++;
      if (b > 3) {
        a++;
        b = a + 1;
      }
    }
  }

  for(int i = 0; i < 6; i++) {
    double * row = l_6x12 + 12 * i;

    row[0] =         dotXY(dv[0][i], dv[0][i]);
    row[1] =  2.0f * dotXY(dv[0][i], dv[1][i]);
    row[2] =         dotXY(dv[1][i], dv[1][i]);
    row[3] =  2.0f * dotXY(dv[0][i], dv[2][i]);
    row[4] =  2.0f * dotXY(dv[1][i], dv[2][i]);
    row[5] =         dotXY(dv[2][i], dv[2][i]);

    row[6] =         dotZ(dv[0][i], dv[0][i]);
    row[7] =  2.0f * dotZ(dv[0][i], dv[1][i]);
    row[8] =  2.0f * dotZ(dv[0][i], dv[2][i]);
    row[9] =         dotZ(dv[1][i], dv[1][i]);
    row[10] = 2.0f * dotZ(dv[1][i], dv[2][i]);
    row[11] =        dotZ(dv[2][i], dv[2][i]);
  }
}

void upnp::compute_rho(double * rho)
{
  rho[0] = dist2(cws[0], cws[1]);
  rho[1] = dist2(cws[0], cws[2]);
  rho[2] = dist2(cws[0], cws[3]);
  rho[3] = dist2(cws[1], cws[2]);
  rho[4] = dist2(cws[1], cws[3]);
  rho[5] = dist2(cws[2], cws[3]);
}

double upnp::dist2(const double * p1, const double * p2)
{
  return
    (p1[0] - p2[0]) * (p1[0] - p2[0]) +
    (p1[1] - p2[1]) * (p1[1] - p2[1]) +
    (p1[2] - p2[2]) * (p1[2] - p2[2]);
}

double upnp::dot(const double * v1, const double * v2)
{
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

double upnp::dotXY(const double * v1, const double * v2)
{
  return v1[0] * v2[0] + v1[1] * v2[1];
}

double upnp::dotZ(const double * v1, const double * v2)
{
  return v1[2] * v2[2];
}

double upnp::sign(const double v)
{
  return ( v < 0.0 ) ? -1.0 : ( v > 0.0 ) ? 1.0 : 0.0;
}

void upnp::qr_solve(Mat * A, Mat * b, Mat * X)
{
  const int nr = A->rows;
  const int nc = A->cols;

  if (max_nr != 0 && max_nr < nr)
  {
    delete [] A1;
    delete [] A2;
  }
  if (max_nr < nr)
  {
    max_nr = nr;
    A1 = new double[nr];
    A2 = new double[nr];
  }

  double * pA = A->ptr<double>(0), * ppAkk = pA;
  for(int k = 0; k < nc; k++)
  {
    double * ppAik1 = ppAkk, eta = fabs(*ppAik1);
    for(int i = k + 1; i < nr; i++)
    {
      double elt = fabs(*ppAik1);
      if (eta < elt) eta = elt;
      ppAik1 += nc;
    }
    if (eta == 0)
    {
      A1[k] = A2[k] = 0.0;
      //cerr << "God damnit, A is singular, this shouldn't happen." << endl;
      return;
    }
    else
    {
     double * ppAik2 = ppAkk, sum2 = 0.0, inv_eta = 1. / eta;
     for(int i = k; i < nr; i++)
     {
       *ppAik2 *= inv_eta;
       sum2 += *ppAik2 * *ppAik2;
       ppAik2 += nc;
     }
     double sigma = sqrt(sum2);
     if (*ppAkk < 0)
     sigma = -sigma;
     *ppAkk += sigma;
     A1[k] = sigma * *ppAkk;
     A2[k] = -eta * sigma;
     for(int j = k + 1; j < nc; j++)
     {
       double * ppAik = ppAkk, sum = 0;
       for(int i = k; i < nr; i++)
       {
        sum += *ppAik * ppAik[j - k];
        ppAik += nc;
       }
       double tau = sum / A1[k];
       ppAik = ppAkk;
       for(int i = k; i < nr; i++)
       {
        ppAik[j - k] -= tau * *ppAik;
        ppAik += nc;
       }
     }
    }
    ppAkk += nc + 1;
  }

  // b <- Qt b
  double * ppAjj = pA, * pb = b->ptr<double>(0);
  for(int j = 0; j < nc; j++)
  {
    double * ppAij = ppAjj, tau = 0;
    for(int i = j; i < nr; i++)
    {
     tau += *ppAij * pb[i];
     ppAij += nc;
    }
    tau /= A1[j];
    ppAij = ppAjj;
    for(int i = j; i < nr; i++)
    {
     pb[i] -= tau * *ppAij;
     ppAij += nc;
    }
    ppAjj += nc + 1;
  }

  // X = R-1 b
  double * pX = X->ptr<double>(0);
  pX[nc - 1] = pb[nc - 1] / A2[nc - 1];
  for(int i = nc - 2; i >= 0; i--)
  {
    double * ppAij = pA + i * nc + (i + 1), sum = 0;

    for(int j = i + 1; j < nc; j++)
    {
     sum += *ppAij * pX[j];
     ppAij++;
    }
    pX[i] = (pb[i] - sum) / A2[i];
  }
}
