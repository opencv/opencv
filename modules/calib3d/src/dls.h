#ifndef DLS_H_
#define DLS_H_

#include "precomp.hpp"

using namespace std;


class dls
{
public:
	dls(const cv::Mat& opoints, const cv::Mat& ipoints);
	~dls();

	void compute_pose(cv::Mat& R, cv::Mat& t);

private:

	template <typename OpointType, typename O, typename IpointType, typename I>
	void init_points(const cv::Mat& opoints, const cv::Mat& ipoints)
	{
		for(int i = 0; i < N; i++)
		{
			p.at<O>(0,i) = opoints.at<OpointType>(0,i).x;
			p.at<O>(1,i) = opoints.at<OpointType>(0,i).y;
			p.at<O>(2,i) = opoints.at<OpointType>(0,i).z;

			z.at<I>(0,i) = ipoints.at<IpointType>(0,i).x;
			z.at<I>(1,i) = ipoints.at<IpointType>(0,i).y;
			z.at<I>(2,i) = (I)1;
		}
	}

	void norm_z_vector();
	void build_coeff_matrix(const cv::Mat& pp, cv::Mat& Mtilde, cv::Mat& D);
	void compute_eigenvec(const cv::Mat& Mtilde, cv::Mat& eigenval_real, cv::Mat& eigenval_imag,
			                                     cv::Mat& eigenvec_real, cv::Mat& eigenvec_imag);
	double min_val(const std::vector<double>& values);
	cv::Mat rotx(const double t);
	cv::Mat roty(const double t);
	cv::Mat rotz(const double t);
	cv::Mat mean(const cv::Mat& M);
	void fill_coeff(const cv::Mat * D);
	cv::Mat LeftMultVec(const cv::Mat& v);
	cv::Mat cayley_LS_M(const std::vector<double>& a, const std::vector<double>& b,
			            const std::vector<double>& c, const std::vector<double>& u);
	bool positive_eigenvalues(const cv::Mat& eigenvalues);
	cv::Mat Hessian(const double s[]);
	cv::Mat cayley2rotbar(const double s[]);
	cv::Mat skewsymm(const double X1[]);
	bool is_empty(const cv::Mat * v);
	void run_kernel(const cv::Mat& pp);

	cv::Mat p, z;		// object-image points
	int N;				// number of input points
	std::vector<double> f1coeff, f2coeff, f3coeff, cost_;
	std::vector<cv::Mat> C_est_, t_est_;	// vector to store candidate
	cv::Mat C_est__, t_est__;	// best found solution
	double cost__;

};

#endif // DLS_H
