#include "opencv2/opencv_modules.hpp"
#include "opencv2/core.hpp"

void Mat_to_vector_int(cv::Mat& mat, std::vector<int>& v_int);
void vector_int_to_Mat(std::vector<int>& v_int, cv::Mat& mat);

void Mat_to_vector_double(cv::Mat& mat, std::vector<double>& v_double);
void vector_double_to_Mat(std::vector<double>& v_double, cv::Mat& mat);

void Mat_to_vector_float(cv::Mat& mat, std::vector<float>& v_float);
void vector_float_to_Mat(std::vector<float>& v_float, cv::Mat& mat);

void Mat_to_vector_uchar(cv::Mat& mat, std::vector<uchar>& v_uchar);
void vector_uchar_to_Mat(std::vector<uchar>& v_uchar, cv::Mat& mat);

void Mat_to_vector_char(cv::Mat& mat, std::vector<char>& v_char);
void vector_char_to_Mat(std::vector<char>& v_char, cv::Mat& mat);

void Mat_to_vector_Rect(cv::Mat& mat, std::vector<cv::Rect>& v_rect);
void vector_Rect_to_Mat(std::vector<cv::Rect>& v_rect, cv::Mat& mat);

void Mat_to_vector_Rect2d(cv::Mat& mat, std::vector<cv::Rect2d>& v_rect);
void vector_Rect2d_to_Mat(std::vector<cv::Rect2d>& v_rect, cv::Mat& mat);


void Mat_to_vector_Point(cv::Mat& mat, std::vector<cv::Point>& v_point);
void Mat_to_vector_Point2f(cv::Mat& mat, std::vector<cv::Point2f>& v_point);
void Mat_to_vector_Point2d(cv::Mat& mat, std::vector<cv::Point2d>& v_point);
void Mat_to_vector_Point3i(cv::Mat& mat, std::vector<cv::Point3i>& v_point);
void Mat_to_vector_Point3f(cv::Mat& mat, std::vector<cv::Point3f>& v_point);
void Mat_to_vector_Point3d(cv::Mat& mat, std::vector<cv::Point3d>& v_point);

void vector_Point_to_Mat(std::vector<cv::Point>& v_point, cv::Mat& mat);
void vector_Point2f_to_Mat(std::vector<cv::Point2f>& v_point, cv::Mat& mat);
void vector_Point2d_to_Mat(std::vector<cv::Point2d>& v_point, cv::Mat& mat);
void vector_Point3i_to_Mat(std::vector<cv::Point3i>& v_point, cv::Mat& mat);
void vector_Point3f_to_Mat(std::vector<cv::Point3f>& v_point, cv::Mat& mat);
void vector_Point3d_to_Mat(std::vector<cv::Point3d>& v_point, cv::Mat& mat);

void vector_Vec4i_to_Mat(std::vector<cv::Vec4i>& v_vec, cv::Mat& mat);
void vector_Vec4f_to_Mat(std::vector<cv::Vec4f>& v_vec, cv::Mat& mat);
void vector_Vec6f_to_Mat(std::vector<cv::Vec6f>& v_vec, cv::Mat& mat);

void Mat_to_vector_Mat(cv::Mat& mat, std::vector<cv::Mat>& v_mat);
void vector_Mat_to_Mat(std::vector<cv::Mat>& v_mat, cv::Mat& mat);

void Mat_to_vector_vector_char(cv::Mat& mat, std::vector< std::vector< char > >& vv_ch);
void vector_vector_char_to_Mat(std::vector< std::vector< char > >& vv_ch, cv::Mat& mat);

void Mat_to_vector_vector_Point(cv::Mat& mat, std::vector< std::vector< cv::Point > >& vv_pt);
void vector_vector_Point_to_Mat(std::vector< std::vector< cv::Point > >& vv_pt, cv::Mat& mat);

void Mat_to_vector_vector_Point2f(cv::Mat& mat, std::vector< std::vector< cv::Point2f > >& vv_pt);
void vector_vector_Point2f_to_Mat(std::vector< std::vector< cv::Point2f > >& vv_pt, cv::Mat& mat);

void Mat_to_vector_vector_Point3f(cv::Mat& mat, std::vector< std::vector< cv::Point3f > >& vv_pt);
void vector_vector_Point3f_to_Mat(std::vector< std::vector< cv::Point3f > >& vv_pt, cv::Mat& mat);
