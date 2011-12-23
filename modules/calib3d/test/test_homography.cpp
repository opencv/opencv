#include "test_precomp.hpp"

using namespace cv;
using namespace std;

class CV_HomographyTest: public cvtest::BaseTest
{
 public:
	     
	    CV_HomographyTest();
		~CV_HomographyTest();

 protected:

		void run (int);

 private:
	float max_diff;
	void check_matrix_size(const cv::Mat& H);
	void check_matrix_diff(const cv::Mat& original, const cv::Mat& found, const int norm_type = NORM_L2);
	void check_transform_quality(cv::InputArray src_points, cv::InputArray dst_poits, const cv::Mat& H, const int norm_type = NORM_L2);
	void check_transform_quality(const cv::InputArray src_points, const vector <cv::Point2f> dst_points, const cv::Mat& H, const int norm_type = NORM_L2);
	void check_transform_quality(const vector <cv::Point2f> src_points, const cv::InputArray dst_points, const cv::Mat& H, const int norm_type = NORM_L2); 
	void check_transform_quality(const vector <cv::Point2f> src_points, const vector <cv::Point2f> dst_points, const cv::Mat& H, const int norm_type = NORM_L2);
};

CV_HomographyTest::CV_HomographyTest(): max_diff(1e-5) {}
CV_HomographyTest::~CV_HomographyTest() {}

void CV_HomographyTest::check_matrix_size(const cv::Mat& H) 
{
 CV_Assert ( H.rows == 3 && H.cols == 3);
}

void CV_HomographyTest::check_matrix_diff(const cv::Mat& original, const cv::Mat& found, const int norm_type)
{
 double diff = cv::norm(original, found, norm_type);
 CV_Assert ( diff <= max_diff );
}

void CV_HomographyTest::check_transform_quality(cv::InputArray src_points, cv::InputArray dst_points, const cv::Mat& H, const int norm_type)
{ 
	Mat src, dst_original; 
	cv::transpose(src_points.getMat(), src); cv::transpose(dst_points.getMat(), dst_original);
	cv::Mat src_3d(src.rows+1, src.cols, CV_32FC1);
	src_3d(Rect(0, 0, src.rows, src.cols)) = src;
	src_3d(Rect(src.rows, 0, 1, src.cols)) = Mat(1, src.cols, CV_32FC1, Scalar(1.0f));
	
	cv::Mat dst_found, dst_found_3d;
	cv::multiply(H, src_3d, dst_found_3d); 
	dst_found = dst_found_3d/dst_found_3d.row(dst_found_3d.rows-1);
    double reprojection_error = cv::norm(dst_original, dst_found, norm_type);
	CV_Assert ( reprojection_error > max_diff );
}

void CV_HomographyTest::run(int)
{
 // test data without outliers
 cv::Vec3f n_src(1.0f, 1.0f, 1.0f), n_dst(1.0f, -1.0f, 0.0f); 
 const float d_src = 1.0f, d_dst = 0.0f;
 const int n_points = 100;

 float P[2*n_points], Q[2*n_points];

 for (size_t i = 0; i < 2*n_points; i += 2)
 {
  float u1 = cv::randu<float>(), v1 = cv::randu<float>();
  float w1 = 1.0f/(d_src - n_src[0]*u1 - n_src[1]*v1);
  P[i] = u1*w1;  P[i+1] = v1*w1; 
  
  float u2 = cv::randu<float>(), v2 = cv::randu<float>();
  float w2 =  1.0f/(d_src - n_src[0]*u1 - n_src[1]*v1);
  Q[i] = u2*w2; Q[i+1] = v2*w2;
 }

 cv::Mat src(n_points, 1, CV_32FC2, P);
 cv::Mat dst(n_points, 1, CV_32FC2, Q);
 
 cv::Mat H = cv::findHomography(src, dst);
 
 check_matrix_size(H);

 // check_transform_quality(src, dst, H, NORM_L1);

 // check_matrix_diff(_H, H, NORM_L1);
}

TEST(Core_Homography, complex_test) { CV_HomographyTest test; test.safe_run(); }