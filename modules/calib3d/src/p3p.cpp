#include <cstring>
#include <cmath>
#include <iostream>

#include "polynom_solver.h"
#include "p3p.h"

using namespace std;

void p3p::init_inverse_parameters()
{
	inv_fx = 1. / fx;
	inv_fy = 1. / fy;
	cx_fx = cx / fx;
	cy_fy = cy / fy;
}

p3p::p3p(cv::Mat cameraMatrix)
{
	if (cameraMatrix.depth() == CV_32F)
		init_camera_parameters<float>(cameraMatrix);
	else
		init_camera_parameters<double>(cameraMatrix);
	init_inverse_parameters();
}

p3p::p3p(double _fx, double _fy, double _cx, double _cy)
{
	fx = _fx;
	fy = _fy;
	cx = _cx;
	cy = _cy;
	init_inverse_parameters();
}

bool p3p::solve(cv::Mat& R, cv::Mat& tvec, const cv::Mat& opoints, const cv::Mat& ipoints)
{
	double rotation_matrix[3][3], translation[3];
	std::vector<double> points;
	if (opoints.depth() == ipoints.depth())
	{
		if (opoints.depth() == CV_32F)
			extract_points<cv::Point3f,cv::Point2f>(opoints, ipoints, points);
		else
			extract_points<cv::Point3d,cv::Point2d>(opoints, ipoints, points);
	}
	else if (opoints.depth() == CV_32F)
		extract_points<cv::Point3f,cv::Point2d>(opoints, ipoints, points);
	else
		extract_points<cv::Point3d,cv::Point2f>(opoints, ipoints, points);

	bool result = solve(rotation_matrix, translation, points[0], points[1], points[2], points[3], points[4], points[5], 
		  points[6], points[7], points[8], points[9], points[10], points[11], points[12], points[13], points[14],
		  points[15], points[16], points[17], points[18], points[19]);
	cv::Mat(3, 1, CV_64F, translation).copyTo(tvec);
    cv::Mat(3, 3, CV_64F, rotation_matrix).copyTo(R);
	return result;
}

bool p3p::solve(double R[3][3], double t[3],
	double mu0, double mv0,   double X0, double Y0, double Z0,
	double mu1, double mv1,   double X1, double Y1, double Z1,
	double mu2, double mv2,   double X2, double Y2, double Z2,
	double mu3, double mv3,   double X3, double Y3, double Z3)
{
	double Rs[4][3][3], ts[4][3];

	int n = solve(Rs, ts, mu0, mv0, X0, Y0, Z0,  mu1, mv1, X1, Y1, Z1, mu2, mv2, X2, Y2, Z2);

	if (n == 0)
		return false;

	int ns = 0;
	double min_reproj = 0;
	for(int i = 0; i < n; i++) {
		double X3p = Rs[i][0][0] * X3 + Rs[i][0][1] * Y3 + Rs[i][0][2] * Z3 + ts[i][0];
		double Y3p = Rs[i][1][0] * X3 + Rs[i][1][1] * Y3 + Rs[i][1][2] * Z3 + ts[i][1];
		double Z3p = Rs[i][2][0] * X3 + Rs[i][2][1] * Y3 + Rs[i][2][2] * Z3 + ts[i][2];
		double mu3p = cx + fx * X3p / Z3p;
		double mv3p = cy + fy * Y3p / Z3p;
		double reproj = (mu3p - mu3) * (mu3p - mu3) + (mv3p - mv3) * (mv3p - mv3);
		if (i == 0 || min_reproj > reproj) {
			ns = i;
			min_reproj = reproj;
		}
	}

	for(int i = 0; i < 3; i++) {
		for(int j = 0; j < 3; j++)
			R[i][j] = Rs[ns][i][j];
		t[i] = ts[ns][i];
	}

	return true;
}

int p3p::solve(double R[4][3][3], double t[4][3],
	double mu0, double mv0,   double X0, double Y0, double Z0,
	double mu1, double mv1,   double X1, double Y1, double Z1,
	double mu2, double mv2,   double X2, double Y2, double Z2)
{
	double mk0, mk1, mk2;
	double norm;

	mu0 = inv_fx * mu0 - cx_fx;
	mv0 = inv_fy * mv0 - cy_fy;
	norm = sqrt(mu0 * mu0 + mv0 * mv0 + 1);
	mk0 = 1. / norm; mu0 *= mk0; mv0 *= mk0;

	mu1 = inv_fx * mu1 - cx_fx;
	mv1 = inv_fy * mv1 - cy_fy;
	norm = sqrt(mu1 * mu1 + mv1 * mv1 + 1);
	mk1 = 1. / norm; mu1 *= mk1; mv1 *= mk1;

	mu2 = inv_fx * mu2 - cx_fx;
	mv2 = inv_fy * mv2 - cy_fy;
	norm = sqrt(mu2 * mu2 + mv2 * mv2 + 1);
	mk2 = 1. / norm; mu2 *= mk2; mv2 *= mk2;

	double distances[3];
	distances[0] = sqrt( (X1 - X2) * (X1 - X2) + (Y1 - Y2) * (Y1 - Y2) + (Z1 - Z2) * (Z1 - Z2) );
	distances[1] = sqrt( (X0 - X2) * (X0 - X2) + (Y0 - Y2) * (Y0 - Y2) + (Z0 - Z2) * (Z0 - Z2) );
	distances[2] = sqrt( (X0 - X1) * (X0 - X1) + (Y0 - Y1) * (Y0 - Y1) + (Z0 - Z1) * (Z0 - Z1) );

	// Calculate angles
	double cosines[3];
	cosines[0] = mu1 * mu2 + mv1 * mv2 + mk1 * mk2;
	cosines[1] = mu0 * mu2 + mv0 * mv2 + mk0 * mk2;
	cosines[2] = mu0 * mu1 + mv0 * mv1 + mk0 * mk1;

	double lengths[4][3];
	int n = solve_for_lengths(lengths, distances, cosines);

	int nb_solutions = 0;
	for(int i = 0; i < n; i++) {
		double M_orig[3][3];

		M_orig[0][0] = lengths[i][0] * mu0;
		M_orig[0][1] = lengths[i][0] * mv0;
		M_orig[0][2] = lengths[i][0] * mk0;

		M_orig[1][0] = lengths[i][1] * mu1;
		M_orig[1][1] = lengths[i][1] * mv1;
		M_orig[1][2] = lengths[i][1] * mk1;

		M_orig[2][0] = lengths[i][2] * mu2;
		M_orig[2][1] = lengths[i][2] * mv2;
		M_orig[2][2] = lengths[i][2] * mk2;

		if (!align(M_orig, X0, Y0, Z0, X1, Y1, Z1, X2, Y2, Z2, R[nb_solutions], t[nb_solutions]))
			continue;

		nb_solutions++;
	}

	return nb_solutions;
}

/// Given 3D distances between three points and cosines of 3 angles at the apex, calculates
/// the lentghs of the line segments connecting projection center (P) and the three 3D points (A, B, C).
/// Returned distances are for |PA|, |PB|, |PC| respectively.
/// Only the solution to the main branch.
/// Reference : X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang; "Complete Solution Classification for the Perspective-Three-Point Problem"
/// IEEE Trans. on PAMI, vol. 25, No. 8, August 2003
/// \param lengths3D Lengths of line segments up to four solutions.
/// \param dist3D Distance between 3D points in pairs |BC|, |AC|, |AB|.
/// \param cosines Cosine of the angles /_BPC, /_APC, /_APB.
/// \returns Number of solutions.
/// WARNING: NOT ALL THE DEGENERATE CASES ARE IMPLEMENTED

int p3p::solve_for_lengths(double lengths[4][3], double distances[3], double cosines[3])
{
	double p = cosines[0] * 2;
	double q = cosines[1] * 2;
	double r = cosines[2] * 2;

	double inv_d22 = 1. / (distances[2] * distances[2]);
	double a = inv_d22 * (distances[0] * distances[0]);
	double b = inv_d22 * (distances[1] * distances[1]);

	double a2 = a * a, b2 = b * b, p2 = p * p, q2 = q * q, r2 = r * r;
	double pr = p * r, pqr = q * pr;

	// Check reality condition (the four points should not be coplanar)
	if (p2 + q2 + r2 - pqr - 1 == 0)
		return 0;

	double ab = a * b, a_2 = 2*a;

	double A = -2 * b + b2 + a2 + 1 + ab*(2 - r2) - a_2;

	// Check reality condition
	if (A == 0) return 0;

	double a_4 = 4*a;

	double B = q*(-2*(ab + a2 + 1 - b) + r2*ab + a_4) + pr*(b - b2 + ab);
	double C = q2 + b2*(r2 + p2 - 2) - b*(p2 + pqr) - ab*(r2 + pqr) + (a2 - a_2)*(2 + q2) + 2;
	double D = pr*(ab-b2+b) + q*((p2-2)*b + 2 * (ab - a2) + a_4 - 2);
	double E = 1 + 2*(b - a - ab) + b2 - b*p2 + a2;

	double temp = (p2*(a-1+b) + r2*(a-1-b) + pqr - a*pqr);
	double b0 = b * temp * temp;
	// Check reality condition
	if (b0 == 0)
		return 0;

	double real_roots[4];
	int n = solve_deg4(A, B, C, D, E,  real_roots[0], real_roots[1], real_roots[2], real_roots[3]);

	if (n == 0)
		return 0;

	int nb_solutions = 0;
	double r3 = r2*r, pr2 = p*r2, r3q = r3 * q;
	double inv_b0 = 1. / b0;

	// For each solution of x
	for(int i = 0; i < n; i++) {
		double x = real_roots[i];

		// Check reality condition
		if (x <= 0)
			continue;

		double x2 = x*x;

		double b1 =
			((1-a-b)*x2 + (q*a-q)*x + 1 - a + b) *
			(((r3*(a2 + ab*(2 - r2) - a_2 + b2 - 2*b + 1)) * x +

			(r3q*(2*(b-a2) + a_4 + ab*(r2 - 2) - 2) + pr2*(1 + a2 + 2*(ab-a-b) + r2*(b - b2) + b2))) * x2 +

			(r3*(q2*(1-2*a+a2) + r2*(b2-ab) - a_4 + 2*(a2 - b2) + 2) + r*p2*(b2 + 2*(ab - b - a) + 1 + a2) + pr2*q*(a_4 + 2*(b - ab - a2) - 2 - r2*b)) * x +

			2*r3q*(a_2 - b - a2 + ab - 1) + pr2*(q2 - a_4 + 2*(a2 - b2) + r2*b + q2*(a2 - a_2) + 2) +
			p2*(p*(2*(ab - a - b) + a2 + b2 + 1) + 2*q*r*(b + a_2 - a2 - ab - 1)));

		// Check reality condition
		if (b1 <= 0)
			continue;

		double y = inv_b0 * b1;
		double v = x2 + y*y - x*y*r;

		if (v <= 0)
			continue;

		double Z = distances[2] / sqrt(v);
		double X = x * Z;
		double Y = y * Z;

		lengths[nb_solutions][0] = X;
		lengths[nb_solutions][1] = Y;
		lengths[nb_solutions][2] = Z;

		nb_solutions++;
	}

	return nb_solutions;
}

bool p3p::align(double M_end[3][3],
	double X0, double Y0, double Z0,
	double X1, double Y1, double Z1,
	double X2, double Y2, double Z2,
	double R[3][3], double T[3])
{
	// Centroids:
	double C_start[3], C_end[3];
	for(int i = 0; i < 3; i++) C_end[i] = (M_end[0][i] + M_end[1][i] + M_end[2][i]) / 3;
	C_start[0] = (X0 + X1 + X2) / 3;
	C_start[1] = (Y0 + Y1 + Y2) / 3;
	C_start[2] = (Z0 + Z1 + Z2) / 3;

	// Covariance matrix s:
	double s[3 * 3];
	for(int j = 0; j < 3; j++) {
		s[0 * 3 + j] = (X0 * M_end[0][j] + X1 * M_end[1][j] + X2 * M_end[2][j]) / 3 - C_end[j] * C_start[0];
		s[1 * 3 + j] = (Y0 * M_end[0][j] + Y1 * M_end[1][j] + Y2 * M_end[2][j]) / 3 - C_end[j] * C_start[1];
		s[2 * 3 + j] = (Z0 * M_end[0][j] + Z1 * M_end[1][j] + Z2 * M_end[2][j]) / 3 - C_end[j] * C_start[2];
	}

	double Qs[16], evs[4], U[16];

	Qs[0 * 4 + 0] = s[0 * 3 + 0] + s[1 * 3 + 1] + s[2 * 3 + 2];
	Qs[1 * 4 + 1] = s[0 * 3 + 0] - s[1 * 3 + 1] - s[2 * 3 + 2];
	Qs[2 * 4 + 2] = s[1 * 3 + 1] - s[2 * 3 + 2] - s[0 * 3 + 0];
	Qs[3 * 4 + 3] = s[2 * 3 + 2] - s[0 * 3 + 0] - s[1 * 3 + 1];

	Qs[1 * 4 + 0] = Qs[0 * 4 + 1] = s[1 * 3 + 2] - s[2 * 3 + 1];
	Qs[2 * 4 + 0] = Qs[0 * 4 + 2] = s[2 * 3 + 0] - s[0 * 3 + 2];
	Qs[3 * 4 + 0] = Qs[0 * 4 + 3] = s[0 * 3 + 1] - s[1 * 3 + 0];
	Qs[2 * 4 + 1] = Qs[1 * 4 + 2] = s[1 * 3 + 0] + s[0 * 3 + 1];
	Qs[3 * 4 + 1] = Qs[1 * 4 + 3] = s[2 * 3 + 0] + s[0 * 3 + 2];
	Qs[3 * 4 + 2] = Qs[2 * 4 + 3] = s[2 * 3 + 1] + s[1 * 3 + 2];

	jacobi_4x4(Qs, evs, U);

	// Looking for the largest eigen value:
	int i_ev = 0;
	double ev_max = evs[i_ev];
	for(int i = 1; i < 4; i++)
		if (evs[i] > ev_max)
			ev_max = evs[i_ev = i];

	// Quaternion:
	double q[4];
	for(int i = 0; i < 4; i++)
		q[i] = U[i * 4 + i_ev];

	double q02 = q[0] * q[0], q12 = q[1] * q[1], q22 = q[2] * q[2], q32 = q[3] * q[3];
	double q0_1 = q[0] * q[1], q0_2 = q[0] * q[2], q0_3 = q[0] * q[3];
	double q1_2 = q[1] * q[2], q1_3 = q[1] * q[3];
	double q2_3 = q[2] * q[3];

	R[0][0] = q02 + q12 - q22 - q32;
	R[0][1] = 2. * (q1_2 - q0_3);
	R[0][2] = 2. * (q1_3 + q0_2);

	R[1][0] = 2. * (q1_2 + q0_3);
	R[1][1] = q02 + q22 - q12 - q32;
	R[1][2] = 2. * (q2_3 - q0_1);

	R[2][0] = 2. * (q1_3 - q0_2);
	R[2][1] = 2. * (q2_3 + q0_1);
	R[2][2] = q02 + q32 - q12 - q22;

	for(int i = 0; i < 3; i++)
		T[i] = C_end[i] - (R[i][0] * C_start[0] + R[i][1] * C_start[1] + R[i][2] * C_start[2]);

	return true;
}

bool p3p::jacobi_4x4(double * A, double * D, double * U)
{
	double B[4], Z[4];
	double Id[16] = {1., 0., 0., 0.,
		0., 1., 0., 0.,
		0., 0., 1., 0.,
		0., 0., 0., 1.};

	memcpy(U, Id, 16 * sizeof(double));

	B[0] = A[0]; B[1] = A[5]; B[2] = A[10]; B[3] = A[15];
	memcpy(D, B, 4 * sizeof(double));
	memset(Z, 0, 4 * sizeof(double));

	for(int iter = 0; iter < 50; iter++) {
		double sum = fabs(A[1]) + fabs(A[2]) + fabs(A[3]) + fabs(A[6]) + fabs(A[7]) + fabs(A[11]);

		if (sum == 0.0)
			return true;

		double tresh =  (iter < 3) ? 0.2 * sum / 16. : 0.0;
		for(int i = 0; i < 3; i++) {
			double * pAij = A + 5 * i + 1;
			for(int j = i + 1 ; j < 4; j++) {
				double Aij = *pAij;
				double eps_machine = 100.0 * fabs(Aij);

				if ( iter > 3 && fabs(D[i]) + eps_machine == fabs(D[i]) && fabs(D[j]) + eps_machine == fabs(D[j]) )
					*pAij = 0.0;
				else if (fabs(Aij) > tresh) {
					double h = D[j] - D[i], t;
					if (fabs(h) + eps_machine == fabs(h))
						t = Aij / h;
					else {
						double theta = 0.5 * h / Aij;
						t = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
						if (theta < 0.0) t = -t;
					}

					h = t * Aij;
					Z[i] -= h;
					Z[j] += h;
					D[i] -= h;
					D[j] += h;
					*pAij = 0.0;

					double c = 1.0 / sqrt(1 + t * t);
					double s = t * c;
					double tau = s / (1.0 + c);
					for(int k = 0; k <= i - 1; k++) {
						double g = A[k * 4 + i], h = A[k * 4 + j];
						A[k * 4 + i] = g - s * (h + g * tau);
						A[k * 4 + j] = h + s * (g - h * tau);
					}
					for(int k = i + 1; k <= j - 1; k++) {
						double g = A[i * 4 + k], h = A[k * 4 + j];
						A[i * 4 + k] = g - s * (h + g * tau);
						A[k * 4 + j] = h + s * (g - h * tau);
					}
					for(int k = j + 1; k < 4; k++) {
						double g = A[i * 4 + k], h = A[j * 4 + k];
						A[i * 4 + k] = g - s * (h + g * tau);
						A[j * 4 + k] = h + s * (g - h * tau);
					}
					for(int k = 0; k < 4; k++) {
						double g = U[k * 4 + i], h = U[k * 4 + j];
						U[k * 4 + i] = g - s * (h + g * tau);
						U[k * 4 + j] = h + s * (g - h * tau);
					}
				}
				pAij++;
			}
		}

		for(int i = 0; i < 4; i++) B[i] += Z[i];
		memcpy(D, B, 4 * sizeof(double));
		memset(Z, 0, 4 * sizeof(double));
	}

	return false;
}

