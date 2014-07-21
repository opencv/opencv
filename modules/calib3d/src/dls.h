#ifndef DLS_H_
#define DLS_H_

#include "precomp.hpp"

using namespace std;


class dls
{
public:
	dls(const cv::Mat& opoints, const cv::Mat& ipoints);
	virtual ~dls();

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
	void build_coeff_matrix();
	cv::Mat LeftMultVec(const cv::Mat& v);
	cv::Mat cayley_LS_M(const std::vector<double>& a, const std::vector<double>& b,
			            const std::vector<double>& c, const std::vector<double>& u);

private:
	cv::Mat H, A, D_mat;	// coeff matrix
	cv::Mat p;				// object points
	cv::Mat z;				// image points
	int N;					// number of input points
	std::vector<double> f1coeff, f2coeff, f3coeff;
};


#endif // DLS_H
