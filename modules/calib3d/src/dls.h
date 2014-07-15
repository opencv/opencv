#ifndef DLS_H
#define DLS_H

#include "precomp.hpp"

class dls
{
public:
	dls(const cv::Mat& opoints, const cv::Mat& ipoints);
	~dls();

	void init_vectors(const cv::Mat& opoints, const cv::Mat& ipoints);
	void build_coeff_mattrix();
	cv::Mat LeftMultVec(const cv::Mat& v);
	cv::Mat cayley_LS_M(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& c, const std::vector<double>& u);

private:
	cv::Mat H;			// coeff matrix
	cv::Mat A;
	cv::Mat D_mat;
	std::vector<double> f1coeff;
	std::vector<double> f2coeff;
	std::vector<double> f3coeff;
	cv::Mat p;			// object points
	cv::Mat z;			// image points
	int N;				// number of input points
};


#endif // DLS_H
