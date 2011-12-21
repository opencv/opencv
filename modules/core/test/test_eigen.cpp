#include "test_precomp.hpp"

using namespace cv;
using namespace std;

#define sign(a) a > 0 ? 1 : a < 0 ? 0 : -1

class Core_EigenTest: public cvtest::BaseTest
{
 public: 
	Core_EigenTest();
	~Core_EigenTest();

 protected:
	 void run (int);

 private:
 
 float eps_val_32, eps_vec_32;
 double eps_val_64, eps_vec_64;
 void check_pair_count(const cv::Mat& src, const cv::Mat& evalues, int low_index = -1, int high_index = -1);
 void check_pair_count(const cv::Mat& src, const cv::Mat& evalues, const cv::Mat& evectors, int low_index = -1, int high_index = -1);
 bool check_diff(const cv::Mat& original_values, const cv::Mat& original_vectors, 
				 const cv::Mat& found_values, const cv::Mat& found_vectors, 
				 const bool compute_eigen_vectors, const int values_type, const int norm_type);
};

Core_EigenTest::Core_EigenTest() : eps_val_32(1e-6), eps_vec_32(1e-5), eps_val_64(1e-12), eps_vec_64(1e-11) {}
Core_EigenTest::~Core_EigenTest() {}

void Core_EigenTest::check_pair_count(const cv::Mat& src, const cv::Mat& evalues, int low_index, int high_index)
{
 int n = src.rows, s = sign(high_index);
 CV_Assert(evalues.rows == n - (low_index+1) - ((int)((n/2.0)*(s*s-s)) + (1+s-s*s)*(n - (high_index+1))) && evalues.cols == 1);
}

void Core_EigenTest::check_pair_count(const cv::Mat& src, const cv::Mat& evalues, const cv::Mat& evectors, int low_index, int high_index)
{
 int n = src.rows, s = sign(high_index);
 int right_eigen_pair_count = n - (low_index+1) - ((int)((n/2.0)*(s*s-s)) + (1+s-s*s)*(n - (high_index+1)));
 CV_Assert( evectors.rows == right_eigen_pair_count &&
			evectors.cols == right_eigen_pair_count &&
			evalues.rows == right_eigen_pair_count && 
			evalues.cols == 1);  
}

bool Core_EigenTest::check_diff(const cv::Mat& original_values, const cv::Mat& original_vectors, 
								const cv::Mat& found_values, const cv::Mat& found_vectors, 
								const bool compute_eigen_vectors, const int values_type, const int norm_type)
{
 double eps_val = values_type == CV_32FC1 ? eps_val_32 : eps_val_64;
 double eps_vec = values_type == CV_32FC1 ? eps_vec_32 : eps_vec_64;

 switch (compute_eigen_vectors)
 {
  case true: 
	        {
			 double diff_val = cv::norm(original_values, found_values, norm_type);
			 double diff_vec = cv::norm(original_vectors, found_vectors, norm_type);

			 if (diff_val > eps_val) { ts->printf(cvtest::TS::LOG, "Accuracy of eigen values computing less than requered."); return false; }
			 if (diff_vec > eps_vec) { ts->printf(cvtest::TS::LOG, "Accuracy of eigen vectors computing less than requered."); return false; }
			
			 break;
	        }

  case false: 
	         {
			  double diff_val = cv::norm(original_values, found_values, norm_type);
			  		 
			  if (diff_val > eps_val) { ts->printf(cvtest::TS::LOG, "Accuracy of eigen values computing less than requered."); return false; }
		
			  break;
	         }

  default:;
 }

 return true;
}

void Core_EigenTest::run(int)
{
  const int DIM = 3;
    
  // tests data 

  float sym_matrix[DIM][DIM] = { { 0.0f, 1.0f, 0.0f }, 
								 { 1.0f, 0.0f, 1.0f }, 
						         { 0.0f, 1.0f, 0.0f } };					// source symmerical matrix

  float _eval[DIM] = { sqrt(2.0f), 0.0f, -sqrt(2.0f) };						// eigen values of 3*3 matrix

  float _evec[DIM][DIM] = { { 1.0f, sqrt(2.0f), 1.0f }, 
							{ -1.0f, 0.0f, 1.0f },
							{ 1.0f, -sqrt(2.0f), 1.0f } };					// eigen vectors of source matrix

  // initializing Mat-objects
  
  cv::Mat eigen_values, eigen_vectors;
  
  cv::Mat src_32(DIM, DIM, CV_32FC1, (void*)&sym_matrix[0]);
  cv::Mat eval_32(DIM, 1, CV_32FC1, (void*)&_eval);
  cv::Mat evec_32(DIM, DIM, CV_32FC1, (void*)&_evec[0]);

  cv::eigen(src_32, true, eigen_values, eigen_vectors);

  check_pair_count(src_32, eigen_values, eigen_vectors);

  if (!check_diff(eval_32, evec_32, eigen_values, eigen_vectors, true, CV_32FC1, NORM_L1)) return;  
  if (!check_diff(eval_32, evec_32, eigen_values, eigen_vectors, true, CV_32FC1, NORM_L2)) return;
  if (!check_diff(eval_32, evec_32, eigen_values, eigen_vectors, true, CV_32FC1, NORM_INF)) return;

  cv::eigen(src_32, false, eigen_values, eigen_vectors);

  check_pair_count(src_32, eigen_values);

  if (!check_diff(eval_32, evec_32, eigen_values, eigen_vectors, false, CV_32FC1, NORM_L1)) return;  
  if (!check_diff(eval_32, evec_32, eigen_values, eigen_vectors, false, CV_32FC1, NORM_L2)) return;
  if (!check_diff(eval_32, evec_32, eigen_values, eigen_vectors, false, CV_32FC1, NORM_INF)) return;
  
  cv::eigen(src_32, eigen_values, eigen_vectors);

  check_pair_count(src_32, eigen_values, eigen_vectors);

  if (!check_diff(eval_32, evec_32, eigen_values, eigen_vectors, true, CV_32FC1, NORM_L1)) return;  
  if (!check_diff(eval_32, evec_32, eigen_values, eigen_vectors, true, CV_32FC1, NORM_L2)) return;
  if (!check_diff(eval_32, evec_32, eigen_values, eigen_vectors, true, CV_32FC1, NORM_INF)) return;

  cv::eigen(src_32, eigen_values);
  
  check_pair_count(src_32, eigen_values);

  if (!check_diff(eval_32, evec_32, eigen_values, eigen_vectors, false, CV_32FC1, NORM_L1)) return;  
  if (!check_diff(eval_32, evec_32, eigen_values, eigen_vectors, false, CV_32FC1, NORM_L2)) return;
  if (!check_diff(eval_32, evec_32, eigen_values, eigen_vectors, false, CV_32FC1, NORM_INF)) return;
   
  cv::Mat src_64(DIM, DIM, CV_64FC1, (void*)&sym_matrix[0]);
  cv::Mat eval_64(DIM, 1, CV_64FC1, (void*)&_eval);
  cv::Mat evec_64(DIM, DIM, CV_64FC1, (void*)&_evec[0]);

  cv::eigen(src_64, true, eigen_values, eigen_vectors);

  if (!check_diff(eval_64, evec_64, eigen_values, eigen_vectors, true, CV_64FC1, NORM_L1)) return;  
  if (!check_diff(eval_64, evec_64, eigen_values, eigen_vectors, true, CV_64FC1, NORM_L2)) return;
  if (!check_diff(eval_64, evec_64, eigen_values, eigen_vectors, true, CV_64FC1, NORM_INF)) return;

  cv::eigen(src_64, false, eigen_values, eigen_vectors);

  check_pair_count(src_64, eigen_values);

  if (!check_diff(eval_64, evec_64, eigen_values, eigen_vectors, false, CV_64FC1, NORM_L1)) return;  
  if (!check_diff(eval_64, evec_64, eigen_values, eigen_vectors, false, CV_64FC1, NORM_L2)) return;
  if (!check_diff(eval_64, evec_64, eigen_values, eigen_vectors, false, CV_64FC1, NORM_INF)) return;
  
  cv::eigen(src_64, eigen_values, eigen_vectors);

  check_pair_count(src_64, eigen_values, eigen_vectors);

  if (!check_diff(eval_64, evec_64, eigen_values, eigen_vectors, true, CV_64FC1, NORM_L1)) return;  
  if (!check_diff(eval_64, evec_64, eigen_values, eigen_vectors, true, CV_64FC1, NORM_L2)) return;
  if (!check_diff(eval_64, evec_64, eigen_values, eigen_vectors, true, CV_64FC1, NORM_INF)) return;

  cv::eigen(src_64, eigen_values);
  
  check_pair_count(src_64, eigen_values);

  if (!check_diff(eval_64, evec_64, eigen_values, eigen_vectors, false, CV_64FC1, NORM_L1)) return;  
  if (!check_diff(eval_64, evec_64, eigen_values, eigen_vectors, false, CV_64FC1, NORM_L2)) return;
  if (!check_diff(eval_64, evec_64, eigen_values, eigen_vectors, false, CV_64FC1, NORM_INF)) return;

  const int low_index = 2, high_index = 3; 
  cv::Mat submat_val_32(eval_32.rowRange(low_index, high_index));
  cv::Mat submat_vec_32(evec_32.rowRange(low_index, high_index));

  cv::eigen(src_32, eigen_values, low_index, high_index);

  check_pair_count(src_32, eigen_values, eigen_vectors, low_index, high_index);

  if (!check_diff(submat_val_32, submat_vec_32, eigen_values, eigen_vectors, false, CV_32FC1, NORM_L1)) return;
  if (!check_diff(submat_val_32, submat_vec_32, eigen_values, eigen_vectors, false, CV_32FC1, NORM_L2)) return;
  if (!check_diff(submat_val_32, submat_vec_32, eigen_values, eigen_vectors, false, CV_32FC1, NORM_INF)) return;

  cv::eigen(src_32, eigen_values, eigen_vectors, low_index, high_index);

  check_pair_count(src_32, eigen_values, eigen_vectors, low_index, high_index);

  if (!check_diff(submat_val_32, submat_vec_32, eigen_values, eigen_vectors, true, CV_32FC1, NORM_L1)) return;
  if (!check_diff(submat_val_32, submat_vec_32, eigen_values, eigen_vectors, true, CV_32FC1, NORM_L2)) return;
  if (!check_diff(submat_val_32, submat_vec_32, eigen_values, eigen_vectors, true, CV_32FC1, NORM_INF)) return;

  cv::Mat submat_val_64(eval_64.rowRange(low_index, high_index));
  cv::Mat submat_vec_64(evec_64.rowRange(low_index, high_index)); 

  cv::eigen(src_64, eigen_values, low_index, high_index);

  check_pair_count(src_64, eigen_values, low_index, high_index);

  if (!check_diff(submat_val_64, submat_vec_64, eigen_values, eigen_vectors, false, CV_64FC1, NORM_L1)) return;
  if (!check_diff(submat_val_64, submat_vec_64, eigen_values, eigen_vectors, false, CV_64FC1, NORM_L2)) return;
  if (!check_diff(submat_val_64, submat_vec_64, eigen_values, eigen_vectors, false, CV_64FC1, NORM_INF)) return;

  cv::eigen(src_64, eigen_values, eigen_vectors, low_index, high_index);

  check_pair_count(src_64, eigen_values, low_index, high_index);

  if (!check_diff(submat_val_64, submat_vec_64, eigen_values, eigen_vectors, true, CV_64FC1, NORM_L1)) return;
  if (!check_diff(submat_val_64, submat_vec_64, eigen_values, eigen_vectors, true, CV_64FC1, NORM_L2)) return;
  if (!check_diff(submat_val_64, submat_vec_64, eigen_values, eigen_vectors, true, CV_64FC1, NORM_INF)) return;
}

TEST(Core_Eigen, quality) { Core_EigenTest test; test.safe_run(); }

