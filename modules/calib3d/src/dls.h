#ifndef DLS_H_
#define DLS_H_

#include "precomp.hpp"

using namespace std;


class dls
{
public:
	dls(const cv::Mat& opoints, const cv::Mat& ipoints);
	~dls();

	bool compute_pose(cv::Mat& R, cv::Mat& t);

private:

	// initialisation
	template <typename OpointType, typename IpointType>
	void init_points(const cv::Mat& opoints, const cv::Mat& ipoints)
	{
		for(int i = 0; i < N; i++)
		{
			p.at<double>(0,i) = opoints.at<OpointType>(0,i).x;
			p.at<double>(1,i) = opoints.at<OpointType>(0,i).y;
			p.at<double>(2,i) = opoints.at<OpointType>(0,i).z;

			z.at<double>(0,i) = ipoints.at<IpointType>(0,i).x;
			z.at<double>(1,i) = ipoints.at<IpointType>(0,i).y;
			z.at<double>(2,i) = (double)1;
		}
	}

	void norm_z_vector();

	// main algorithm
	void run_kernel(const cv::Mat& pp);
	void build_coeff_matrix(const cv::Mat& pp, cv::Mat& Mtilde, cv::Mat& D);
	void compute_eigenvec(const cv::Mat& Mtilde, cv::Mat& eigenval_real, cv::Mat& eigenval_imag,
			                                     cv::Mat& eigenvec_real, cv::Mat& eigenvec_imag);
	void fill_coeff(const cv::Mat * D);

	// useful functions
	cv::Mat LeftMultVec(const cv::Mat& v);
	cv::Mat cayley_LS_M(const std::vector<double>& a, const std::vector<double>& b,
			            const std::vector<double>& c, const std::vector<double>& u);
	cv::Mat Hessian(const double s[]);
	cv::Mat cayley2rotbar(const cv::Mat& s);
	cv::Mat skewsymm(const cv::Mat * X1);

	// extra functions
	cv::Mat rotx(const double t);
	cv::Mat roty(const double t);
	cv::Mat rotz(const double t);
	cv::Mat mean(const cv::Mat& M);
	bool is_empty(const cv::Mat * v);
	bool positive_eigenvalues(const cv::Mat * eigenvalues);


	cv::Mat p, z;		// object-image points
	int N;				// number of input points
	std::vector<double> f1coeff, f2coeff, f3coeff, cost_; // coefficient for coefficients matrix
	std::vector<cv::Mat> C_est_, t_est_;	// optimal candidates
	cv::Mat C_est__, t_est__;				// optimal found solution
	double cost__;							// optimal found solution
};

#endif // DLS_H
