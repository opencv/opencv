#include <jni.h>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "features2d_manual.hpp"


void Mat_to_vector_int(cv::Mat& mat, std::vector<int>& v_int);

void vector_int_to_Mat(std::vector<int>& v_int, cv::Mat& mat);


void Mat_to_vector_double(cv::Mat& mat, std::vector<double>& v_double);

void vector_double_to_Mat(std::vector<double>& v_double, cv::Mat& mat);


void Mat_to_vector_float(cv::Mat& mat, std::vector<float>& v_float);

void vector_float_to_Mat(std::vector<float>& v_float, cv::Mat& mat);


void Mat_to_vector_uchar(cv::Mat& mat, std::vector<uchar>& v_uchar);


void Mat_to_vector_Rect(cv::Mat& mat, std::vector<cv::Rect>& v_rect);

void vector_Rect_to_Mat(std::vector<cv::Rect>& v_rect, cv::Mat& mat);


void Mat_to_vector_Point(cv::Mat& mat, std::vector<cv::Point>& v_point);

void vector_Point_to_Mat(std::vector<cv::Point>& v_point, cv::Mat& mat);


void Mat_to_vector_KeyPoint(cv::Mat& mat, std::vector<cv::KeyPoint>& v_kp);

void vector_KeyPoint_to_Mat(std::vector<cv::KeyPoint>& v_kp, cv::Mat& mat);

void Mat_to_vector_Mat(cv::Mat& mat, std::vector<cv::Mat>& v_mat);

void vector_Mat_to_Mat(std::vector<cv::Mat>& v_mat, cv::Mat& mat);

