#ifndef __FEATURES_CONVERTERS_HPP__
#define __FEATURES_CONVERTERS_HPP__

#include "opencv2/opencv_modules.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features.hpp"

void Mat_to_vector_KeyPoint(cv::Mat& mat, std::vector<cv::KeyPoint>& v_kp);
void vector_KeyPoint_to_Mat(std::vector<cv::KeyPoint>& v_kp, cv::Mat& mat);

void Mat_to_vector_DMatch(cv::Mat& mat, std::vector<cv::DMatch>& v_dm);
void vector_DMatch_to_Mat(std::vector<cv::DMatch>& v_dm, cv::Mat& mat);

void Mat_to_vector_vector_KeyPoint(cv::Mat& mat, std::vector< std::vector< cv::KeyPoint > >& vv_kp);
void vector_vector_KeyPoint_to_Mat(std::vector< std::vector< cv::KeyPoint > >& vv_kp, cv::Mat& mat);

void Mat_to_vector_vector_DMatch(cv::Mat& mat, std::vector< std::vector< cv::DMatch > >& vv_dm);
void vector_vector_DMatch_to_Mat(std::vector< std::vector< cv::DMatch > >& vv_dm, cv::Mat& mat);


#endif
