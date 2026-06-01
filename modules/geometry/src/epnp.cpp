// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <iostream>
#include "precomp.hpp"
#include "epnp.h"

namespace cv {

epnp::epnp(const Mat& cameraMatrix, const Mat& opoints, const Mat& ipoints)
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

epnp::~epnp()
{
    if (A1)
        delete[] A1;
    if (A2)
        delete[] A2;
}

void epnp::choose_control_points(void)
{
  // Take C0 as the reference points centroid:
  cws[0][0] = cws[0][1] = cws[0][2] = 0;
  for(int i = 0; i < number_of_correspondences; i++)
    for(int j = 0; j < 3; j++)
      cws[0][j] += pws[3 * i + j];

  for(int j = 0; j < 3; j++)
    cws[0][j] /= number_of_correspondences;


  // Take C1, C2, and C3 from PCA on the reference points:
  Mat PW0(number_of_correspondences, 3, CV_64F);

  double pw0tpw0[3 * 3] = {}, dc[3] = {}, uct[3 * 3] = {};
  Mat PW0tPW0(3, 3, CV_64F, pw0tpw0);
  Mat DC(3, 1, CV_64F, dc);
  Mat UCt(3, 3, CV_64F, uct);

  for(int i = 0; i < number_of_correspondences; i++) {
    double* PW0row = PW0.ptr<double>(i);
    for(int j = 0; j < 3; j++)
      PW0row[j] = pws[3 * i + j] - cws[0][j];
  }

  mulTransposed(PW0, PW0tPW0, true);
  SVDecomp(PW0tPW0, DC, UCt, noArray(), SVD::MODIFY_A);
  transpose(UCt, UCt);

  for(int i = 1; i < 4; i++) {
    double k = sqrt(dc[i - 1] / number_of_correspondences);
    for(int j = 0; j < 3; j++)
      cws[i][j] = cws[0][j] + k * uct[3 * (i - 1) + j];
  }
}

void epnp::compute_barycentric_coordinates(void)
{
  Matx33d CC, CC_inv;

  for(int i = 0; i < 3; i++)
    for(int j = 1; j < 4; j++)
      CC(i, j - 1) = cws[j][i] - cws[0][i];

  cv::invert(CC, CC_inv, DECOMP_SVD);
  double * ci = CC_inv.val;
  for(int i = 0; i < number_of_correspondences; i++) {
    double * pi = &pws[0] + 3 * i;
    double * a = &alphas[0] + 4 * i;

    for(int j = 0; j < 3; j++)
    {
      a[1 + j] =
          ci[3 * j    ] * (pi[0] - cws[0][0]) +
          ci[3 * j + 1] * (pi[1] - cws[0][1]) +
          ci[3 * j + 2] * (pi[2] - cws[0][2]);
    }
    a[0] = 1.0f - a[1] - a[2] - a[3];
  }
}

void epnp::fill_M(Mat& M,
      const int row, const double * as, const double u, const double v)
{
  double * M1 = M.ptr<double>(row);
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

void epnp::compute_ccs(const double * betas, const double * ut)
{
  for(int i = 0; i < 4; i++)
    ccs[i][0] = ccs[i][1] = ccs[i][2] = 0.0f;

  for(int i = 0; i < 4; i++) {
    const double * v = ut + 12 * (11 - i);
    for(int j = 0; j < 4; j++)
      for(int k = 0; k < 3; k++)
        ccs[j][k] += betas[i] * v[3 * j + k];
  }
}

void epnp::compute_pcs(void)
{
  for(int i = 0; i < number_of_correspondences; i++) {
    double * a = &alphas[0] + 4 * i;
    double * pc = &pcs[0] + 3 * i;

    for(int j = 0; j < 3; j++)
      pc[j] = a[0] * ccs[0][j] + a[1] * ccs[1][j] + a[2] * ccs[2][j] + a[3] * ccs[3][j];
  }
}

void epnp::compute_pose(Mat& R, Mat& t)
{
  choose_control_points();
  compute_barycentric_coordinates();

  Mat M(2 * number_of_correspondences, 12, CV_64F);

  for(int i = 0; i < number_of_correspondences; i++)
    fill_M(M, 2 * i, &alphas[0] + 4 * i, us[2 * i], us[2 * i + 1]);

  double mtm[12 * 12] = {}, d[12] = {}, ut[12 * 12] = {};
  Mat MtM(12, 12, CV_64F, mtm);
  Mat D(12,  1, CV_64F, d);
  Mat Ut(12, 12, CV_64F, ut);

  mulTransposed(M, MtM, true);
  SVDecomp(MtM, D, Ut, noArray(), SVD::MODIFY_A);
  transpose(Ut, Ut);

  double l_6x10[6 * 10] = {}, rho[6] = {};
  Mat L_6x10(6, 10, CV_64F, l_6x10);
  Mat Rho(6,  1, CV_64F, rho);

  compute_L_6x10(ut, l_6x10);
  compute_rho(rho);

  double Betas[4][4] = {}, rep_errors[4] = {};
  double Rs[4][3][3] = {}, ts[4][3] = {};

  find_betas_approx_1(L_6x10, Rho, Betas[1]);
  gauss_newton(L_6x10, Rho, Betas[1]);
  rep_errors[1] = compute_R_and_t(ut, Betas[1], Rs[1], ts[1]);

  find_betas_approx_2(L_6x10, Rho, Betas[2]);
  gauss_newton(L_6x10, Rho, Betas[2]);
  rep_errors[2] = compute_R_and_t(ut, Betas[2], Rs[2], ts[2]);

  find_betas_approx_3(L_6x10, Rho, Betas[3]);
  gauss_newton(L_6x10, Rho, Betas[3]);
  rep_errors[3] = compute_R_and_t(ut, Betas[3], Rs[3], ts[3]);

  int N = 1;
  if (rep_errors[2] < rep_errors[1]) N = 2;
  if (rep_errors[3] < rep_errors[N]) N = 3;

  Mat(3, 1, CV_64F, ts[N]).copyTo(t);
  Mat(3, 3, CV_64F, Rs[N]).copyTo(R);
}

void epnp::copy_R_and_t(const double R_src[3][3], const double t_src[3],
      double R_dst[3][3], double t_dst[3])
{
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      R_dst[i][j] = R_src[i][j];
    t_dst[i] = t_src[i];
  }
}

double epnp::dist2(const double * p1, const double * p2)
{
  return
    (p1[0] - p2[0]) * (p1[0] - p2[0]) +
    (p1[1] - p2[1]) * (p1[1] - p2[1]) +
    (p1[2] - p2[2]) * (p1[2] - p2[2]);
}

double epnp::dot(const double * v1, const double * v2)
{
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

void epnp::estimate_R_and_t(double R[3][3], double t[3])
{
  double pc0[3] = {}, pw0[3] = {};

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

  double abt[3 * 3] = {}, abt_d[3] = {}, abt_u[3 * 3] = {}, abt_vt[3 * 3] = {};
  Mat ABt(3, 3, CV_64F, abt);
  Mat ABt_D(3, 1, CV_64F, abt_d);
  Mat ABt_U(3, 3, CV_64F, abt_u);
  Mat ABt_Vt(3, 3, CV_64F, abt_vt);

  ABt.setTo(Scalar::all(0.));
  for(int i = 0; i < number_of_correspondences; i++) {
    double * pc = &pcs[3 * i];
    double * pw = &pws[3 * i];

    for(int j = 0; j < 3; j++) {
      abt[3 * j    ] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
      abt[3 * j + 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
      abt[3 * j + 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
    }
  }

  SVDecomp(ABt, ABt_D, ABt_U, ABt_Vt, SVD::MODIFY_A);
  Mat mR(3, 3, CV_64F, R);
  gemm(ABt_U, ABt_Vt, 1, noArray(), 0, mR);

  const double det = determinant(mR);

  if (det < 0) {
    R[2][0] = -R[2][0];
    R[2][1] = -R[2][1];
    R[2][2] = -R[2][2];
  }

  t[0] = pc0[0] - dot(R[0], pw0);
  t[1] = pc0[1] - dot(R[1], pw0);
  t[2] = pc0[2] - dot(R[2], pw0);
}

void epnp::solve_for_sign(void)
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

double epnp::compute_R_and_t(const double * ut, const double * betas,
           double R[3][3], double t[3])
{
  compute_ccs(betas, ut);
  compute_pcs();

  solve_for_sign();

  estimate_R_and_t(R, t);

  return reprojection_error(R, t);
}

double epnp::reprojection_error(const double R[3][3], const double t[3])
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

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]

void epnp::find_betas_approx_1(const Mat& L_6x10, const Mat& Rho, double* betas)
{
  double l_6x4[6 * 4] = {}, b4[4] = {};
  Mat L_6x4(6, 4, CV_64F, l_6x4);
  Mat B4(4, 1, CV_64F, b4);

  for(int i = 0; i < 6; i++) {
      L_6x4.at<double>(i, 0) = L_6x10.at<double>(i, 0);
      L_6x4.at<double>(i, 1) = L_6x10.at<double>(i, 1);
      L_6x4.at<double>(i, 2) = L_6x10.at<double>(i, 3);
      L_6x4.at<double>(i, 3) = L_6x10.at<double>(i, 6);
  }

  solve(L_6x4, Rho, B4, DECOMP_SVD);
  CV_Assert(B4.ptr<double>() == b4);

  if (b4[0] < 0) {
    betas[0] = sqrt(-b4[0]);
    betas[1] = -b4[1] / betas[0];
    betas[2] = -b4[2] / betas[0];
    betas[3] = -b4[3] / betas[0];
  } else {
    betas[0] = sqrt(b4[0]);
    betas[1] = b4[1] / betas[0];
    betas[2] = b4[2] / betas[0];
    betas[3] = b4[3] / betas[0];
  }
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]

void epnp::find_betas_approx_2(const Mat& L_6x10, const Mat& Rho, double* betas)
{
  double l_6x3[6 * 3] = {}, b3[3] = {};
  Mat L_6x3(6, 3, CV_64F, l_6x3);
  Mat B3(3, 1, CV_64F, b3);

  for(int i = 0; i < 6; i++) {
      L_6x3.at<double>(i, 0) = L_6x10.at<double>(i, 0);
      L_6x3.at<double>(i, 1) = L_6x10.at<double>(i, 1);
      L_6x3.at<double>(i, 2) = L_6x10.at<double>(i, 2);
  }

  solve(L_6x3, Rho, B3, DECOMP_SVD);
  CV_Assert(B3.ptr<double>() == b3);

  if (b3[0] < 0) {
    betas[0] = sqrt(-b3[0]);
    betas[1] = (b3[2] < 0) ? sqrt(-b3[2]) : 0.0;
  } else {
    betas[0] = sqrt(b3[0]);
    betas[1] = (b3[2] > 0) ? sqrt(b3[2]) : 0.0;
  }

  if (b3[1] < 0) betas[0] = -betas[0];
  betas[2] = 0.0;
  betas[3] = 0.0;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]

void epnp::find_betas_approx_3(const Mat& L_6x10, const Mat& Rho, double * betas)
{
  double l_6x5[6 * 5] = {}, b5[5] = {};
  Mat L_6x5(6, 5, CV_64F, l_6x5);
  Mat B5(5, 1, CV_64F, b5);

  for(int i = 0; i < 6; i++) {
      L_6x5.at<double>(i, 0) = L_6x10.at<double>(i, 0);
      L_6x5.at<double>(i, 1) = L_6x10.at<double>(i, 1);
      L_6x5.at<double>(i, 2) = L_6x10.at<double>(i, 2);
      L_6x5.at<double>(i, 3) = L_6x10.at<double>(i, 3);
      L_6x5.at<double>(i, 4) = L_6x10.at<double>(i, 4);
  }

  solve(L_6x5, Rho, B5, DECOMP_SVD);
  CV_Assert(B5.ptr<double>() == b5);

  if (b5[0] < 0) {
    betas[0] = sqrt(-b5[0]);
    betas[1] = (b5[2] < 0) ? sqrt(-b5[2]) : 0.0;
  } else {
    betas[0] = sqrt(b5[0]);
    betas[1] = (b5[2] > 0) ? sqrt(b5[2]) : 0.0;
  }
  if (b5[1] < 0) betas[0] = -betas[0];
  betas[2] = b5[3] / betas[0];
  betas[3] = 0.0;
}

void epnp::compute_L_6x10(const double * ut, double * l_6x10)
{
  const double * v[4];

  v[0] = ut + 12 * 11;
  v[1] = ut + 12 * 10;
  v[2] = ut + 12 *  9;
  v[3] = ut + 12 *  8;

  double dv[4][6][3] = {};

  for(int i = 0; i < 4; i++) {
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
    double * row = l_6x10 + 10 * i;

    row[0] =        dot(dv[0][i], dv[0][i]);
    row[1] = 2.0f * dot(dv[0][i], dv[1][i]);
    row[2] =        dot(dv[1][i], dv[1][i]);
    row[3] = 2.0f * dot(dv[0][i], dv[2][i]);
    row[4] = 2.0f * dot(dv[1][i], dv[2][i]);
    row[5] =        dot(dv[2][i], dv[2][i]);
    row[6] = 2.0f * dot(dv[0][i], dv[3][i]);
    row[7] = 2.0f * dot(dv[1][i], dv[3][i]);
    row[8] = 2.0f * dot(dv[2][i], dv[3][i]);
    row[9] =        dot(dv[3][i], dv[3][i]);
  }
}

void epnp::compute_rho(double * rho)
{
  rho[0] = dist2(cws[0], cws[1]);
  rho[1] = dist2(cws[0], cws[2]);
  rho[2] = dist2(cws[0], cws[3]);
  rho[3] = dist2(cws[1], cws[2]);
  rho[4] = dist2(cws[1], cws[3]);
  rho[5] = dist2(cws[2], cws[3]);
}

void epnp::compute_A_and_b_gauss_newton(const Mat& L_6x10, const Mat& Rho,
          const double betas[4], Mat& A, Mat& b)
{
  for(int i = 0; i < 6; i++) {
    const double * rowL = L_6x10.ptr<double>(i);
    double * rowA = A.ptr<double>(i);

    rowA[0] = 2 * rowL[0] * betas[0] +     rowL[1] * betas[1] +     rowL[3] * betas[2] +     rowL[6] * betas[3];
    rowA[1] =     rowL[1] * betas[0] + 2 * rowL[2] * betas[1] +     rowL[4] * betas[2] +     rowL[7] * betas[3];
    rowA[2] =     rowL[3] * betas[0] +     rowL[4] * betas[1] + 2 * rowL[5] * betas[2] +     rowL[8] * betas[3];
    rowA[3] =     rowL[6] * betas[0] +     rowL[7] * betas[1] +     rowL[8] * betas[2] + 2 * rowL[9] * betas[3];

    b.at<double>(i) = Rho.at<double>(i) -
     (
      rowL[0] * betas[0] * betas[0] +
      rowL[1] * betas[0] * betas[1] +
      rowL[2] * betas[1] * betas[1] +
      rowL[3] * betas[0] * betas[2] +
      rowL[4] * betas[1] * betas[2] +
      rowL[5] * betas[2] * betas[2] +
      rowL[6] * betas[0] * betas[3] +
      rowL[7] * betas[1] * betas[3] +
      rowL[8] * betas[2] * betas[3] +
      rowL[9] * betas[3] * betas[3]
      );
  }
}

void epnp::gauss_newton(const Mat& L_6x10, const Mat& Rho, double betas[4])
{
  const int iterations_number = 5;

  double a[6*4] = {}, b[6] = {}, x[4] = {};
  Mat A(6, 4, CV_64F, a);
  Mat B(6, 1, CV_64F, b);
  Mat X(4, 1, CV_64F, x);

  for(int k = 0; k < iterations_number; k++)
  {
    compute_A_and_b_gauss_newton(L_6x10, Rho, betas, A, B);
    qr_solve(A, B, X);
    for(int i = 0; i < 4; i++)
    betas[i] += x[i];
  }
}

void epnp::qr_solve(Mat& A, Mat& b, Mat& X)
{
  const int nr = A.rows;
  const int nc = A.cols;
  if (nc <= 0 || nr <= 0)
      return;

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

  double * pA = A.ptr<double>(), * ppAkk = pA;
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
  double * ppAjj = pA, * pb = b.ptr<double>();
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
  double * pX = X.ptr<double>();
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

}
