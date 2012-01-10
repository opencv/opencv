#include "test_precomp.hpp"
#include <time.h>

using namespace cv;
using namespace std;

#define sign(a) a > 0 ? 1 : a == 0 ? 0 : -1

#define CORE_EIGEN_ERROR_COUNT 1
#define CORE_EIGEN_ERROR_SIZE  2
#define CORE_EIGEN_ERROR_DIFF  3
#define CORE_EIGEN_ERROR_ORTHO 4
#define CORE_EIGEN_ERROR_ORDER 5

class Core_EigenTest: public cvtest::BaseTest
{
 public: 
	
	 Core_EigenTest();
	~Core_EigenTest();
    
 protected:
    
	bool test_values(const cv::Mat& src);															// complex test for eigen without vectors
	bool check_full(int type);																	 	// compex test for symmetric matrix
	virtual void run (int) = 0;																		// main testing method

 private:
 
 float eps_val_32, eps_vec_32;
 float eps_val_64, eps_vec_64;
 bool check_pair_count(const cv::Mat& src, const cv::Mat& evalues, int low_index = -1, int high_index = -1);
 bool check_pair_count(const cv::Mat& src, const cv::Mat& evalues, const cv::Mat& evectors, int low_index = -1, int high_index = -1);
 bool check_pairs_order(const cv::Mat& eigen_values);												// checking order of eigen values & vectors (it should be none up)
 bool check_orthogonality(const cv::Mat& U);														// checking is matrix of eigen vectors orthogonal
 bool test_pairs(const cv::Mat& src);																// complex test for eigen with vectors
};

class Core_EigenTest_Scalar : public Core_EigenTest
{
 public:
	 Core_EigenTest_Scalar() : Core_EigenTest() {}
	 ~Core_EigenTest_Scalar();
     virtual void run(int) = 0;
};

class Core_EigenTest_Scalar_32 : public Core_EigenTest_Scalar
{
 public:
	 Core_EigenTest_Scalar_32() : Core_EigenTest_Scalar() {}
	 ~Core_EigenTest_Scalar_32();

	 void run(int);
};

class Core_EigenTest_Scalar_64 : public Core_EigenTest_Scalar
{
 public:
	Core_EigenTest_Scalar_64() : Core_EigenTest_Scalar() {}
	~Core_EigenTest_Scalar_64();
	void run(int);
};

class Core_EigenTest_32 : public Core_EigenTest
{
 public:
	Core_EigenTest_32(): Core_EigenTest() {}
	~Core_EigenTest_32() {}
	void run(int);
};

class Core_EigenTest_64 : public Core_EigenTest
{
 public:
	 Core_EigenTest_64(): Core_EigenTest() {}
	 ~Core_EigenTest_64() {}
	 void run(int);
};

Core_EigenTest_Scalar::~Core_EigenTest_Scalar() {}
Core_EigenTest_Scalar_32::~Core_EigenTest_Scalar_32() {}
Core_EigenTest_Scalar_64::~Core_EigenTest_Scalar_64() {}

void Core_EigenTest_Scalar_32::run(int) 
{
 float value = cv::randu<float>();
 cv::Mat src(1, 1, CV_32FC1, Scalar::all((float)value));
 test_values(src);
 src.~Mat();
}

void Core_EigenTest_Scalar_64::run(int)
{
 float value = cv::randu<float>();
 cv::Mat src(1, 1, CV_64FC1, Scalar::all((double)value));
 test_values(src);
 src.~Mat();
}

void Core_EigenTest_32::run(int) { check_full(CV_32FC1); }
void Core_EigenTest_64::run(int) { check_full(CV_64FC1); }

Core_EigenTest::Core_EigenTest() : eps_val_32(1e-3), eps_vec_32(1e-2), eps_val_64(1e-4), eps_vec_64(1e-3) {}
Core_EigenTest::~Core_EigenTest() {}

bool Core_EigenTest::check_pair_count(const cv::Mat& src, const cv::Mat& evalues, int low_index, int high_index)
{
 int n = src.rows, s = sign(high_index);
 if (!( (evalues.rows == n - max<int>(0, low_index) - ((int)((n/2.0)*(s*s-s)) + (1+s-s*s)*(n - (high_index+1)))) && (evalues.cols == 1)))
 { 
  std::cout << "Checking sizes of eigen values matrix " << evalues << "..." << endl;
  CV_Error(CORE_EIGEN_ERROR_COUNT, "Matrix of eigen values must have the same rows as source matrix and 1 column."); 
  return false; 
 }
 return true;
}

bool Core_EigenTest::check_pair_count(const cv::Mat& src, const cv::Mat& evalues, const cv::Mat& evectors, int low_index, int high_index)
{
 int n = src.rows, s = sign(high_index);
 int right_eigen_pair_count = n - max<int>(0, low_index) - ((int)((n/2.0)*(s*s-s)) + (1+s-s*s)*(n - (high_index+1)));

 if (!((evectors.rows == right_eigen_pair_count) && (evectors.cols == right_eigen_pair_count)))
 { 
  std::cout << "Checking sizes of eigen vectors matrix " << evectors << "..." << endl;
  CV_Error (CORE_EIGEN_ERROR_SIZE, "Source matrix and matrix of eigen vectors must have the same sizes."); 
  return false; 
 }

 if (!((evalues.rows == right_eigen_pair_count) && (evalues.cols == 1)))
 {
  std::cout << "Checking sizes of eigen values matrix " << evalues << "..." << endl;
  CV_Error (CORE_EIGEN_ERROR_COUNT, "Matrix of eigen values must have the same rows as source matrix and 1 column."); 
  return false; 
 }

 return true;
}

bool Core_EigenTest::check_orthogonality(const cv::Mat& U)
{
 int type = U.type();
 double eps_vec = type == CV_32FC1 ? eps_vec_32 : eps_vec_64;
 cv::Mat UUt; cv::mulTransposed(U, UUt, false); 

 cv::Mat E = Mat::eye(U.rows, U.cols, type);
 
 double diff_L1 = cv::norm(UUt, E, NORM_L1);
 double diff_L2 = cv::norm(UUt, E, NORM_L2);
 double diff_INF = cv::norm(UUt, E, NORM_INF);

 if (diff_L1 > eps_vec) { std::cout << "Checking orthogonality of matrix " << U << "..." << endl; CV_Error(CORE_EIGEN_ERROR_ORTHO, "Matrix of eigen vectors is not orthogonal."); return false; }
 if (diff_L2 > eps_vec) { std::cout << "Checking orthogonality of matrix " << U << "..." << endl; CV_Error(CORE_EIGEN_ERROR_ORTHO, "Matrix of eigen vectors is not orthogonal."); return false; }
 if (diff_INF > eps_vec) { std::cout << "Checking orthogonality of matrix " << U << "..." << endl; CV_Error(CORE_EIGEN_ERROR_ORTHO, "Matrix of eigen vectors is not orthogonal."); return false; }

 return true;
}

bool Core_EigenTest::check_pairs_order(const cv::Mat& eigen_values)
{
 switch (eigen_values.type())
 {
  case CV_32FC1:
  {
   for (int i = 0; i < eigen_values.total() - 1; ++i)
   if (!(eigen_values.at<float>(i, 0) > eigen_values.at<float>(i+1, 0)))
   {
	std::cout << "Checking order of eigen values vector " << eigen_values << "..." << endl;
    CV_Error(CORE_EIGEN_ERROR_ORDER, "Eigen values are not sorted in ascending order.");
	return false;
   }

   break;
  }
  
  case CV_64FC1:
  {
   for (int i = 0; i < eigen_values.total() - 1; ++i)
   if (!(eigen_values.at<double>(i, 0) > eigen_values.at<double>(i+1, 0)))
   {
	std::cout << "Checking order of eigen values vector " << eigen_values << "..." << endl;
	CV_Error(CORE_EIGEN_ERROR_ORDER, "Eigen values are not sorted in ascending order.");
	return false;
   }

   break;
  }

  default:;
 }

 return true;
}

bool Core_EigenTest::test_pairs(const cv::Mat& src)
{
 int type = src.type();
 double eps_vec = type == CV_32FC1 ? eps_vec_32 : eps_vec_64;

 cv::Mat eigen_values, eigen_vectors;
 
 cv::eigen(src, true, eigen_values, eigen_vectors);

 if (!check_pair_count(src, eigen_values, eigen_vectors)) return false;

 if (!check_orthogonality (eigen_vectors)) return false;

 if (!check_pairs_order(eigen_values)) return false;

 cv::Mat eigen_vectors_t; cv::transpose(eigen_vectors, eigen_vectors_t);

 cv::Mat src_evec(src.rows, src.cols, type);
 src_evec = src*eigen_vectors_t; 

 cv::Mat eval_evec(src.rows, src.cols, type);

 switch (type)
 { 
  case CV_32FC1:
  {
   for (size_t i = 0; i < src.cols; ++i)
   {
    cv::Mat tmp = eigen_values.at<float>(i, 0) * eigen_vectors_t.col(i); 
	for (size_t j = 0; j < src.rows; ++j) eval_evec.at<float>(j, i) = tmp.at<float>(j, 0);  
   }

   break;
  }
  
  case CV_64FC1:
  {
   for (size_t i = 0; i < src.cols; ++i)
   {
	cv::Mat tmp = eigen_values.at<double>(i, 0) * eigen_vectors_t.col(i); 
	for (size_t j = 0; j < src.rows; ++j) eval_evec.at<double>(j, i) = tmp.at<double>(j, 0);  
   }

   break; 
  }

  default:;
 }

 cv::Mat disparity = src_evec - eval_evec;

 double diff_L1 = cv::norm(disparity, NORM_L1);
 double diff_L2 = cv::norm(disparity, NORM_L2);
 double diff_INF = cv::norm(disparity, NORM_INF);

 if (diff_L1 > eps_vec) { std::cout << "Checking accuracy of eigen vectors computing for matrix " << src << ": L1-criteria..." << endl; CV_Error(CORE_EIGEN_ERROR_DIFF, "Accuracy of eigen vectors computing less than required."); return false; }
 if (diff_L2 > eps_vec) { std::cout << "Checking accuracy of eigen vectors computing for matrix " << src << ": L2-criteria..." << endl; CV_Error(CORE_EIGEN_ERROR_DIFF, "Accuracy of eigen vectors computing less than required."); return false; }
 if (diff_INF > eps_vec) { std::cout << "Checking accuracy of eigen vectors computing for matrix " << src << ": INF-criteria..." << endl; CV_Error(CORE_EIGEN_ERROR_DIFF, "Accuracy of eigen vectors computing less than required."); return false; }

 return true;
}

bool Core_EigenTest::test_values(const cv::Mat& src)
{
 int type = src.type();
 double eps_val = type == CV_32FC1 ? eps_val_32 : eps_val_64; 

 cv::Mat eigen_values_1, eigen_values_2, eigen_vectors;

 if (!test_pairs(src)) return false;

 cv::eigen(src, true, eigen_values_1, eigen_vectors);
 cv::eigen(src, false, eigen_values_2, eigen_vectors);

 if (!check_pair_count(src, eigen_values_2)) return false;

 double diff_L1 = cv::norm(eigen_values_1, eigen_values_2, NORM_L1);
 double diff_L2 = cv::norm(eigen_values_1, eigen_values_2, NORM_L2);  
 double diff_INF = cv::norm(eigen_values_1, eigen_values_2, NORM_INF); 	
  
 if (diff_L1 > eps_val) { std::cout << "Checking accuracy of eigen values computing for matrix " << src << ": L1-criteria..." << endl; CV_Error(CORE_EIGEN_ERROR_DIFF, "Accuracy of eigen values computing less than required."); return false; }																  
 if (diff_L2 > eps_val) { std::cout << "Checking accuracy of eigen values computing for matrix " << src << ": L2-criteria..." << endl; CV_Error(CORE_EIGEN_ERROR_DIFF, "Accuracy of eigen vectors computing less than required."); return false; }
 if (diff_INF > eps_val) { std::cout << "Checking accuracy of eigen values computing for matrix " << src << ": INF-criteria..." << endl; CV_Error(CORE_EIGEN_ERROR_DIFF, "Accuracy of eigen vectors computing less than required."); return false; }

 return true;
}

bool Core_EigenTest::check_full(int type)
{
 const int MATRIX_COUNT = 500;
 const int MAX_DEGREE = 7;

 srand(time(0));

 for (size_t i = 1; i <= MATRIX_COUNT; ++i)
 {
  size_t src_size = (int)(std::pow(2.0, (rand()%MAX_DEGREE+1)*1.0)); 
  
  cv::Mat src(src_size, src_size, type);

  for (int j = 0; j < src.rows; ++j)
  for (int k = j; k < src.cols; ++k) 
  if (type == CV_32FC1)  src.at<float>(k, j) = src.at<float>(j, k) = cv::randu<float>();
  else	src.at<double>(k, j) = src.at<double>(j, k) = cv::randu<double>();
  
  if (!test_values(src)) return false;

  src.~Mat();
 }

 return true;
}

// TEST(Core_Eigen_Scalar_32, single_complex) {Core_EigenTest_Scalar_32 test; test.safe_run(); }
// TEST(Core_Eigen_Scalar_64, single_complex) {Core_EigenTest_Scalar_64 test; test.safe_run(); }
TEST(Core_Eigen_32, complex) { Core_EigenTest_32 test; test.safe_run(); }
TEST(Core_Eigen_64, complex) { Core_EigenTest_64 test; test.safe_run(); }