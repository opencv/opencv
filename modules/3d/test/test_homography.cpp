/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"
#include <opencv2/features2d.hpp>

namespace opencv_test { namespace {

#define MESSAGE_MATRIX_SIZE "Homography matrix must have 3*3 sizes."
#define MESSAGE_MATRIX_DIFF "Accuracy of homography transformation matrix less than required."
#define MESSAGE_REPROJ_DIFF_1 "Reprojection error for current pair of points more than required."
#define MESSAGE_REPROJ_DIFF_2 "Reprojection error is not optimal."
#define MESSAGE_RANSAC_MASK_1 "Sizes of inliers/outliers mask are incorrect."
#define MESSAGE_RANSAC_MASK_2 "Mask mustn't have any outliers."
#define MESSAGE_RANSAC_MASK_3 "All values of mask must be 1 (true) or 0 (false)."
#define MESSAGE_RANSAC_MASK_4 "Mask of inliers/outliers is incorrect."
#define MESSAGE_RANSAC_MASK_5 "Inlier in original mask shouldn't be outlier in found mask."
#define MESSAGE_RANSAC_DIFF "Reprojection error for current pair of points more than required."

#define MAX_COUNT_OF_POINTS 303
#define MIN_COUNT_OF_POINTS 4
#define COUNT_NORM_TYPES 3
#define METHODS_COUNT 4

int NORM_TYPE[COUNT_NORM_TYPES] = {cv::NORM_L1, cv::NORM_L2, cv::NORM_INF};
int METHOD[METHODS_COUNT] = {0, cv::RANSAC, cv::LMEDS, cv::RHO};

using namespace cv;
using namespace std;


namespace HomographyTestUtils {

static const float max_diff = 0.032f;
static const float max_2diff = 0.020f;
static const int image_size = 100;
static const double reproj_threshold = 3.0;
static const double sigma = 0.01;

static bool check_matrix_size(const cv::Mat& H)
{
    return (H.rows == 3) && (H.cols == 3);
}

static bool check_matrix_diff(const cv::Mat& original, const cv::Mat& found, const int norm_type, double &diff)
{
    diff = cvtest::norm(original, found, norm_type);
    return diff <= max_diff;
}

static int check_ransac_mask_1(const Mat& src, const Mat& mask)
{
    if (!(mask.cols == 1) && (mask.rows == src.cols)) return 1;
    if (countNonZero(mask) < mask.rows) return 2;
    for (int i = 0; i < mask.rows; ++i) if (mask.at<uchar>(i, 0) > 1) return 3;
    return 0;
}

static int check_ransac_mask_2(const Mat& original_mask, const Mat& found_mask)
{
    if (!(found_mask.cols == 1) && (found_mask.rows == original_mask.rows)) return 1;
    for (int i = 0; i < found_mask.rows; ++i) if (found_mask.at<uchar>(i, 0) > 1) return 2;
    return 0;
}

static void print_information_1(int j, int N, int _method, const Mat& H)
{
    cout << endl; cout << "Checking for homography matrix sizes..." << endl; cout << endl;
    cout << "Type of srcPoints: "; if ((j>-1) && (j<2)) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>";
    cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
    cout << "Count of points: " << N << endl; cout << endl;
    cout << "Method: "; if (_method == 0) cout << 0; else if (_method == 8) cout << "RANSAC"; else if (_method == cv::RHO) cout << "RHO"; else cout << "LMEDS"; cout << endl;
    cout << "Homography matrix:" << endl; cout << endl;
    cout << H << endl; cout << endl;
    cout << "Number of rows: " << H.rows << "   Number of cols: " << H.cols << endl; cout << endl;
}

static void print_information_2(int j, int N, int _method, const Mat& H, const Mat& H_res, int k, double diff)
{
    cout << endl; cout << "Checking for accuracy of homography matrix computing..." << endl; cout << endl;
    cout << "Type of srcPoints: "; if ((j>-1) && (j<2)) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>";
    cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
    cout << "Count of points: " << N << endl; cout << endl;
    cout << "Method: "; if (_method == 0) cout << 0; else if (_method == 8) cout << "RANSAC"; else if (_method == cv::RHO) cout << "RHO"; else cout << "LMEDS"; cout << endl;
    cout << "Original matrix:" << endl; cout << endl;
    cout << H << endl; cout << endl;
    cout << "Found matrix:" << endl; cout << endl;
    cout << H_res << endl; cout << endl;
    cout << "Norm type using in criteria: "; if (NORM_TYPE[k] == 1) cout << "INF"; else if (NORM_TYPE[k] == 2) cout << "L1"; else cout << "L2"; cout << endl;
    cout << "Difference between matrices: " << diff << endl;
    cout << "Maximum allowed difference: " << max_diff << endl; cout << endl;
}

static void print_information_3(int _method, int j, int N, const Mat& mask)
{
    cout << endl; cout << "Checking for inliers/outliers mask..." << endl; cout << endl;
    cout << "Type of srcPoints: "; if ((j>-1) && (j<2)) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>";
    cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
    cout << "Count of points: " << N << endl; cout << endl;
    cout << "Method: "; if (_method == RANSAC) cout << "RANSAC" << endl; else if (_method == cv::RHO) cout << "RHO" << endl; else cout << _method << endl;
    cout << "Found mask:" << endl; cout << endl;
    cout << mask << endl; cout << endl;
    cout << "Number of rows: " << mask.rows << "   Number of cols: " << mask.cols << endl; cout << endl;
}

static void print_information_4(int _method, int j, int N, int k, int l, double diff)
{
    cout << endl; cout << "Checking for accuracy of reprojection error computing..." << endl; cout << endl;
    cout << "Method: "; if (_method == 0) cout << 0 << endl; else cout << "CV_LMEDS" << endl;
    cout << "Type of srcPoints: "; if ((j>-1) && (j<2)) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>";
    cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
    cout << "Sigma of normal noise: " << sigma << endl;
    cout << "Count of points: " << N << endl;
    cout << "Number of point: " << k << endl;
    cout << "Norm type using in criteria: "; if (NORM_TYPE[l] == 1) cout << "INF"; else if (NORM_TYPE[l] == 2) cout << "L1"; else cout << "L2"; cout << endl;
    cout << "Difference with noise of point: " << diff << endl;
    cout << "Maximum allowed difference: " << max_2diff << endl; cout << endl;
}

static void print_information_5(int _method, int j, int N, int l, double diff)
{
    cout << endl; cout << "Checking for accuracy of reprojection error computing..." << endl; cout << endl;
    cout << "Method: "; if (_method == 0) cout << 0 << endl; else cout << "CV_LMEDS" << endl;
    cout << "Type of srcPoints: "; if ((j>-1) && (j<2)) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>";
    cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
    cout << "Sigma of normal noise: " << sigma << endl;
    cout << "Count of points: " << N << endl;
    cout << "Norm type using in criteria: "; if (NORM_TYPE[l] == 1) cout << "INF"; else if (NORM_TYPE[l] == 2) cout << "L1"; else cout << "L2"; cout << endl;
    cout << "Difference with noise of points: " << diff << endl;
    cout << "Maximum allowed difference: " << max_diff << endl; cout << endl;
}

static void print_information_6(int _method, int j, int N, int k, double diff, bool value)
{
    cout << endl; cout << "Checking for inliers/outliers mask..." << endl; cout << endl;
    cout << "Method: "; if (_method == RANSAC) cout << "RANSAC" << endl; else if (_method == cv::RHO) cout << "RHO" << endl; else cout << _method << endl;
    cout << "Type of srcPoints: "; if ((j>-1) && (j<2)) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>";
    cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
    cout << "Count of points: " << N << "   " << endl;
    cout << "Number of point: " << k << "   " << endl;
    cout << "Reprojection error for this point: " << diff << "   " << endl;
    cout << "Reprojection error threshold: " << reproj_threshold << "   " << endl;
    cout << "Value of found mask: "<< value << endl; cout << endl;
}

static void print_information_7(int _method, int j, int N, int k, double diff, bool original_value, bool found_value)
{
    cout << endl; cout << "Checking for inliers/outliers mask..." << endl; cout << endl;
    cout << "Method: "; if (_method == RANSAC) cout << "RANSAC" << endl; else if (_method == cv::RHO) cout << "RHO" << endl; else cout << _method << endl;
    cout << "Type of srcPoints: "; if ((j>-1) && (j<2)) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>";
    cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
    cout << "Count of points: " << N << "   " << endl;
    cout << "Number of point: " << k << "   " << endl;
    cout << "Reprojection error for this point: " << diff << "   " << endl;
    cout << "Reprojection error threshold: " << reproj_threshold << "   " << endl;
    cout << "Value of original mask: "<< original_value << "   Value of found mask: " << found_value << endl; cout << endl;
}

static void print_information_8(int _method, int j, int N, int k, int l, double diff)
{
    cout << endl; cout << "Checking for reprojection error of inlier..." << endl; cout << endl;
    cout << "Method: "; if (_method == RANSAC) cout << "RANSAC" << endl; else if (_method == cv::RHO) cout << "RHO" << endl; else cout << _method << endl;
    cout << "Sigma of normal noise: " << sigma << endl;
    cout << "Type of srcPoints: "; if ((j>-1) && (j<2)) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>";
    cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
    cout << "Count of points: " << N << "   " << endl;
    cout << "Number of point: " << k << "   " << endl;
    cout << "Norm type using in criteria: "; if (NORM_TYPE[l] == 1) cout << "INF"; else if (NORM_TYPE[l] == 2) cout << "L1"; else cout << "L2"; cout << endl;
    cout << "Difference with noise of point: " << diff << endl;
    cout << "Maximum allowed difference: " << max_2diff << endl; cout << endl;
}

} // HomographyTestUtils::


TEST(Calib3d_Homography, accuracy)
{
    using namespace HomographyTestUtils;
    for (int N = MIN_COUNT_OF_POINTS; N <= MAX_COUNT_OF_POINTS; ++N)
    {
        RNG& rng = cv::theRNG();

        float *src_data = new float [2*N];

        for (int i = 0; i < N; ++i)
        {
            src_data[2*i] = (float)cvtest::randReal(rng)*image_size;
            src_data[2*i+1] = (float)cvtest::randReal(rng)*image_size;
        }

        cv::Mat src_mat_2f(1, N, CV_32FC2, src_data),
        src_mat_2d(2, N, CV_32F, src_data),
        src_mat_3d(3, N, CV_32F);
        cv::Mat dst_mat_2f, dst_mat_2d, dst_mat_3d;

        vector <Point2f> src_vec, dst_vec;

        for (int i = 0; i < N; ++i)
        {
            float *tmp = src_mat_2d.ptr<float>()+2*i;
            src_mat_3d.at<float>(0, i) = tmp[0];
            src_mat_3d.at<float>(1, i) = tmp[1];
            src_mat_3d.at<float>(2, i) = 1.0f;

            src_vec.push_back(Point2f(tmp[0], tmp[1]));
        }

        double fi = cvtest::randReal(rng)*2*CV_PI;

        double t_x = cvtest::randReal(rng)*sqrt(image_size*1.0),
        t_y = cvtest::randReal(rng)*sqrt(image_size*1.0);

        double Hdata[9] = { cos(fi), -sin(fi), t_x,
                            sin(fi),  cos(fi), t_y,
                            0.0f,     0.0f, 1.0f };

        cv::Mat H_64(3, 3, CV_64F, Hdata), H_32;

        H_64.convertTo(H_32, CV_32F);

        dst_mat_3d = H_32*src_mat_3d;

        dst_mat_2d.create(2, N, CV_32F); dst_mat_2f.create(1, N, CV_32FC2);

        for (int i = 0; i < N; ++i)
        {
            float *tmp_2f = dst_mat_2f.ptr<float>()+2*i;
            tmp_2f[0] = dst_mat_2d.at<float>(0, i) = dst_mat_3d.at<float>(0, i) /= dst_mat_3d.at<float>(2, i);
            tmp_2f[1] = dst_mat_2d.at<float>(1, i) = dst_mat_3d.at<float>(1, i) /= dst_mat_3d.at<float>(2, i);
            dst_mat_3d.at<float>(2, i) = 1.0f;

            dst_vec.push_back(Point2f(tmp_2f[0], tmp_2f[1]));
        }

        for (int i = 0; i < METHODS_COUNT; ++i)
        {
            const int method = METHOD[i];
            switch (method)
            {
            case 0:
            case LMEDS:
                {
                    Mat H_res_64 [4] = { cv::findHomography(src_mat_2f, dst_mat_2f, method),
                                         cv::findHomography(src_mat_2f, dst_vec, method),
                                         cv::findHomography(src_vec, dst_mat_2f, method),
                                         cv::findHomography(src_vec, dst_vec, method) };

                    for (int j = 0; j < 4; ++j)
                    {

                        if (!check_matrix_size(H_res_64[j]))
                        {
                            print_information_1(j, N, method, H_res_64[j]);
                            CV_Error(cv::Error::StsError, MESSAGE_MATRIX_SIZE);
                            return;
                        }

                        double diff;

                        for (int k = 0; k < COUNT_NORM_TYPES; ++k)
                            if (!check_matrix_diff(H_64, H_res_64[j], NORM_TYPE[k], diff))
                            {
                            print_information_2(j, N, method, H_64, H_res_64[j], k, diff);
                            CV_Error(cv::Error::StsError, MESSAGE_MATRIX_DIFF);
                            return;
                        }
                    }

                    continue;
                }
            case cv::RHO:
            case RANSAC:
                {
                    cv::Mat mask [4]; double diff;

                    Mat H_res_64 [4] = { cv::findHomography(src_mat_2f, dst_mat_2f, method, reproj_threshold, mask[0]),
                                         cv::findHomography(src_mat_2f, dst_vec, method, reproj_threshold, mask[1]),
                                         cv::findHomography(src_vec, dst_mat_2f, method, reproj_threshold, mask[2]),
                                         cv::findHomography(src_vec, dst_vec, method, reproj_threshold, mask[3]) };

                    for (int j = 0; j < 4; ++j)
                    {

                        if (!check_matrix_size(H_res_64[j]))
                        {
                            print_information_1(j, N, method, H_res_64[j]);
                            CV_Error(cv::Error::StsError, MESSAGE_MATRIX_SIZE);
                            return;
                        }

                        for (int k = 0; k < COUNT_NORM_TYPES; ++k)
                            if (!check_matrix_diff(H_64, H_res_64[j], NORM_TYPE[k], diff))
                            {
                            print_information_2(j, N, method, H_64, H_res_64[j], k, diff);
                            CV_Error(cv::Error::StsError, MESSAGE_MATRIX_DIFF);
                            return;
                        }

                        int code = check_ransac_mask_1(src_mat_2f, mask[j]);

                        if (code)
                        {
                            print_information_3(method, j, N, mask[j]);

                            switch (code)
                            {
                            case 1: { CV_Error(cv::Error::StsError, MESSAGE_RANSAC_MASK_1); break; }
                            case 2: { CV_Error(cv::Error::StsError, MESSAGE_RANSAC_MASK_2); break; }
                            case 3: { CV_Error(cv::Error::StsError, MESSAGE_RANSAC_MASK_3); break; }

                            default: break;
                            }

                            return;
                        }

                    }

                    continue;
                }

            default: continue;
            }
        }

        Mat noise_2f(1, N, CV_32FC2);
        rng.fill(noise_2f, RNG::NORMAL, Scalar::all(0), Scalar::all(sigma));

        cv::Mat mask(N, 1, CV_8UC1);

        for (int i = 0; i < N; ++i)
        {
            float *a = noise_2f.ptr<float>()+2*i, *_2f = dst_mat_2f.ptr<float>()+2*i;
            _2f[0] += a[0]; _2f[1] += a[1];
            mask.at<bool>(i, 0) = !(sqrt(a[0]*a[0]+a[1]*a[1]) > reproj_threshold);
        }

        for (int i = 0; i < METHODS_COUNT; ++i)
        {
            const int method = METHOD[i];
            switch (method)
            {
            case 0:
            case LMEDS:
                {
                    Mat H_res_64 [4] = { cv::findHomography(src_mat_2f, dst_mat_2f),
                                         cv::findHomography(src_mat_2f, dst_vec),
                                         cv::findHomography(src_vec, dst_mat_2f),
                                         cv::findHomography(src_vec, dst_vec) };

                    for (int j = 0; j < 4; ++j)
                    {

                        if (!check_matrix_size(H_res_64[j]))
                        {
                            print_information_1(j, N, method, H_res_64[j]);
                            CV_Error(cv::Error::StsError, MESSAGE_MATRIX_SIZE);
                            return;
                        }

                        Mat H_res_32; H_res_64[j].convertTo(H_res_32, CV_32F);

                        cv::Mat dst_res_3d(3, N, CV_32F), noise_2d(2, N, CV_32F);

                        for (int k = 0; k < N; ++k)
                        {

                            Mat tmp_mat_3d = H_res_32*src_mat_3d.col(k);

                            dst_res_3d.at<float>(0, k) = tmp_mat_3d.at<float>(0, 0) /= tmp_mat_3d.at<float>(2, 0);
                            dst_res_3d.at<float>(1, k) = tmp_mat_3d.at<float>(1, 0) /= tmp_mat_3d.at<float>(2, 0);
                            dst_res_3d.at<float>(2, k) = tmp_mat_3d.at<float>(2, 0) = 1.0f;

                            float *a = noise_2f.ptr<float>()+2*k;
                            noise_2d.at<float>(0, k) = a[0]; noise_2d.at<float>(1, k) = a[1];

                            for (int l = 0; l < COUNT_NORM_TYPES; ++l)
                                if (cv::norm(tmp_mat_3d, dst_mat_3d.col(k), NORM_TYPE[l]) - cv::norm(noise_2d.col(k), NORM_TYPE[l]) > max_2diff)
                                {
                                print_information_4(method, j, N, k, l, cv::norm(tmp_mat_3d, dst_mat_3d.col(k), NORM_TYPE[l]) - cv::norm(noise_2d.col(k), NORM_TYPE[l]));
                                CV_Error(cv::Error::StsError, MESSAGE_REPROJ_DIFF_1);
                                return;
                            }

                        }

                        for (int l = 0; l < COUNT_NORM_TYPES; ++l)
                            if (cv::norm(dst_res_3d, dst_mat_3d, NORM_TYPE[l]) - cv::norm(noise_2d, NORM_TYPE[l]) > max_diff)
                            {
                            print_information_5(method, j, N, l, cv::norm(dst_res_3d, dst_mat_3d, NORM_TYPE[l]) - cv::norm(noise_2d, NORM_TYPE[l]));
                            CV_Error(cv::Error::StsError, MESSAGE_REPROJ_DIFF_2);
                            return;
                        }

                    }

                    continue;
                }
            case cv::RHO:
            case RANSAC:
                {
                    cv::Mat mask_res [4];

                    Mat H_res_64 [4] = { cv::findHomography(src_mat_2f, dst_mat_2f, method, reproj_threshold, mask_res[0]),
                                         cv::findHomography(src_mat_2f, dst_vec, method, reproj_threshold, mask_res[1]),
                                         cv::findHomography(src_vec, dst_mat_2f, method, reproj_threshold, mask_res[2]),
                                         cv::findHomography(src_vec, dst_vec, method, reproj_threshold, mask_res[3]) };

                    for (int j = 0; j < 4; ++j)
                    {
                        if (!check_matrix_size(H_res_64[j]))
                        {
                            print_information_1(j, N, method, H_res_64[j]);
                            CV_Error(cv::Error::StsError, MESSAGE_MATRIX_SIZE);
                            return;
                        }

                        int code = check_ransac_mask_2(mask, mask_res[j]);

                        if (code)
                        {
                            print_information_3(method, j, N, mask_res[j]);

                            switch (code)
                            {
                            case 1: { CV_Error(cv::Error::StsError, MESSAGE_RANSAC_MASK_1); break; }
                            case 2: { CV_Error(cv::Error::StsError, MESSAGE_RANSAC_MASK_3); break; }

                            default: break;
                            }

                            return;
                        }

                        cv::Mat H_res_32; H_res_64[j].convertTo(H_res_32, CV_32F);

                        cv::Mat dst_res_3d = H_res_32*src_mat_3d;

                        for (int k = 0; k < N; ++k)
                        {
                            dst_res_3d.at<float>(0, k) /= dst_res_3d.at<float>(2, k);
                            dst_res_3d.at<float>(1, k) /= dst_res_3d.at<float>(2, k);
                            dst_res_3d.at<float>(2, k) = 1.0f;

                            float *p = dst_mat_2f.ptr<float>()+2*k;

                            dst_mat_3d.at<float>(0, k) = p[0];
                            dst_mat_3d.at<float>(1, k) = p[1];

                            double diff = cv::norm(dst_res_3d.col(k), dst_mat_3d.col(k), NORM_L2);

                            if (mask_res[j].at<bool>(k, 0) != (diff <= reproj_threshold))
                            {
                                print_information_6(method, j, N, k, diff, mask_res[j].at<bool>(k, 0));
                                CV_Error(cv::Error::StsError, MESSAGE_RANSAC_MASK_4);
                                return;
                            }

                            if (mask.at<bool>(k, 0) && !mask_res[j].at<bool>(k, 0))
                            {
                                print_information_7(method, j, N, k, diff, mask.at<bool>(k, 0), mask_res[j].at<bool>(k, 0));
                                CV_Error(cv::Error::StsError, MESSAGE_RANSAC_MASK_5);
                                return;
                            }

                            if (mask_res[j].at<bool>(k, 0))
                            {
                                float *a = noise_2f.ptr<float>()+2*k;
                                dst_mat_3d.at<float>(0, k) -= a[0];
                                dst_mat_3d.at<float>(1, k) -= a[1];

                                cv::Mat noise_2d(2, 1, CV_32F);
                                noise_2d.at<float>(0, 0) = a[0]; noise_2d.at<float>(1, 0) = a[1];

                                for (int l = 0; l < COUNT_NORM_TYPES; ++l)
                                {
                                    diff = cv::norm(dst_res_3d.col(k), dst_mat_3d.col(k), NORM_TYPE[l]);

                                    if (diff - cv::norm(noise_2d, NORM_TYPE[l]) > max_2diff)
                                    {
                                        print_information_8(method, j, N, k, l, diff - cv::norm(noise_2d, NORM_TYPE[l]));
                                        CV_Error(cv::Error::StsError, MESSAGE_RANSAC_DIFF);
                                        return;
                                    }
                                }
                            }
                        }
                    }

                    continue;
                }

            default: continue;
            }
        }

        delete[]src_data;
        src_data = NULL;
    }
}

TEST(Calib3d_Homography, EKcase)
{
    float pt1data[] =
    {
        2.80073029e+002f, 2.39591217e+002f, 2.21912201e+002f, 2.59783997e+002f,
        2.16053192e+002f, 2.78826569e+002f, 2.22782532e+002f, 2.82330383e+002f,
        2.09924820e+002f, 2.89122559e+002f, 2.11077698e+002f, 2.89384674e+002f,
        2.25287689e+002f, 2.88795532e+002f, 2.11180801e+002f, 2.89653503e+002f,
        2.24126404e+002f, 2.90466064e+002f, 2.10914429e+002f, 2.90886963e+002f,
        2.23439362e+002f, 2.91657715e+002f, 2.24809387e+002f, 2.91891602e+002f,
        2.09809082e+002f, 2.92891113e+002f, 2.08771164e+002f, 2.93093231e+002f,
        2.23160095e+002f, 2.93259460e+002f, 2.07874023e+002f, 2.93989990e+002f,
        2.08963638e+002f, 2.94209839e+002f, 2.23963165e+002f, 2.94479645e+002f,
        2.23241791e+002f, 2.94887817e+002f, 2.09438782e+002f, 2.95233337e+002f,
        2.08901886e+002f, 2.95762878e+002f, 2.21867981e+002f, 2.95747711e+002f,
        2.24195511e+002f, 2.98270905e+002f, 2.09331345e+002f, 3.05958191e+002f,
        2.24727875e+002f, 3.07186035e+002f, 2.26718842e+002f, 3.08095795e+002f,
        2.25363953e+002f, 3.08200226e+002f, 2.19897797e+002f, 3.13845093e+002f,
        2.25013474e+002f, 3.15558777e+002f
    };

    float pt2data[] =
    {
        1.84072723e+002f, 1.43591202e+002f, 1.25912483e+002f, 1.63783859e+002f,
        2.06439407e+002f, 2.20573929e+002f, 1.43801437e+002f, 1.80703903e+002f,
        9.77904129e+000f, 2.49660202e+002f, 1.38458405e+001f, 2.14502701e+002f,
        1.50636337e+002f, 2.15597183e+002f, 6.43103180e+001f, 2.51667648e+002f,
        1.54952499e+002f, 2.20780014e+002f, 1.26638412e+002f, 2.43040924e+002f,
        3.67568909e+002f, 1.83624954e+001f, 1.60657944e+002f, 2.21794052e+002f,
        -1.29507828e+000f, 3.32472443e+002f, 8.51442242e+000f, 4.15561554e+002f,
        1.27161377e+002f, 1.97260361e+002f, 5.40714645e+000f, 4.90978302e+002f,
        2.25571690e+001f, 3.96912415e+002f, 2.95664978e+002f, 7.36064959e+000f,
        1.27241104e+002f, 1.98887573e+002f, -1.25569367e+000f, 3.87713226e+002f,
        1.04194012e+001f, 4.31495758e+002f, 1.25868874e+002f, 1.99751617e+002f,
        1.28195480e+002f, 2.02270355e+002f, 2.23436356e+002f, 1.80489182e+002f,
        1.28727692e+002f, 2.11185410e+002f, 2.03336639e+002f, 2.52182083e+002f,
        1.29366486e+002f, 2.12201904e+002f, 1.23897598e+002f, 2.17847351e+002f,
        1.29015259e+002f, 2.19560623e+002f
    };

    int npoints = (int)(sizeof(pt1data)/sizeof(pt1data[0])/2);

    Mat p1(1, npoints, CV_32FC2, pt1data);
    Mat p2(1, npoints, CV_32FC2, pt2data);
    Mat mask;

    Mat h = findHomography(p1, p2, RANSAC, 0.01, mask);
    ASSERT_TRUE(!h.empty());

    cv::transpose(mask, mask);
    Mat p3, mask2;
    int ninliers = countNonZero(mask);
    Mat nmask[] = { mask, mask };
    merge(nmask, 2, mask2);
    perspectiveTransform(p1, p3, h);
    mask2 = mask2.reshape(1);
    p2 = p2.reshape(1);
    p3 = p3.reshape(1);
    double err = cvtest::norm(p2, p3, NORM_INF, mask2);

    printf("ninliers: %d, inliers err: %.2g\n", ninliers, err);
    ASSERT_GE(ninliers, 10);
    ASSERT_LE(err, 0.01);
}

TEST(Calib3d_Homography, fromImages)
{
    Mat img_1 = imread(cvtest::TS::ptr()->get_data_path() + "cv/optflow/image1.png", 0);
    Mat img_2 = imread(cvtest::TS::ptr()->get_data_path() + "cv/optflow/image2.png", 0);
    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    orb->detectAndCompute( img_1, Mat(), keypoints_1, descriptors_1, false );
    orb->detectAndCompute( img_2, Mat(), keypoints_2, descriptors_2, false );

    //-- Step 3: Matching descriptor vectors using Brute Force matcher
    BFMatcher  matcher(NORM_HAMMING,false);
    std::vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    double max_dist = 0; double min_dist = 100;
    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;
    for( int i = 0; i < descriptors_1.rows; i++ )
    {
        if( matches[i].distance <= 100 )
            good_matches.push_back( matches[i]);
    }

    //-- Localize the model
    std::vector<Point2f> pointframe1;
    std::vector<Point2f> pointframe2;
    for( int i = 0; i < (int)good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        pointframe1.push_back( keypoints_1[ good_matches[i].queryIdx ].pt );
        pointframe2.push_back( keypoints_2[ good_matches[i].trainIdx ].pt );
    }

    Mat H0, H1, inliers0, inliers1;
    double min_t0 = DBL_MAX, min_t1 = DBL_MAX;
    for( int i = 0; i < 10; i++ )
    {
        double t = (double)getTickCount();
        H0 = findHomography( pointframe1, pointframe2, RANSAC, 3.0, inliers0 );
        t = (double)getTickCount() - t;
        min_t0 = std::min(min_t0, t);
    }
    int ninliers0 = countNonZero(inliers0);
    for( int i = 0; i < 10; i++ )
    {
        double t = (double)getTickCount();
        H1 = findHomography( pointframe1, pointframe2, RHO, 3.0, inliers1 );
        t = (double)getTickCount() - t;
        min_t1 = std::min(min_t1, t);
    }
    int ninliers1 = countNonZero(inliers1);
    double freq = getTickFrequency();
    printf("nfeatures1 = %d, nfeatures2=%d, matches=%d, ninliers(RANSAC)=%d, "
           "time(RANSAC)=%.2fmsec, ninliers(RHO)=%d, time(RHO)=%.2fmsec\n",
           (int)keypoints_1.size(), (int)keypoints_2.size(),
           (int)good_matches.size(), ninliers0, min_t0*1000./freq, ninliers1, min_t1*1000./freq);

    ASSERT_TRUE(!H0.empty());
    ASSERT_GE(ninliers0, 80);
    ASSERT_TRUE(!H1.empty());
    ASSERT_GE(ninliers1, 80);
}

TEST(Calib3d_Homography, minPoints)
{
    float pt1data[] =
    {
        2.80073029e+002f, 2.39591217e+002f, 2.21912201e+002f, 2.59783997e+002f
    };

    float pt2data[] =
    {
        1.84072723e+002f, 1.43591202e+002f, 1.25912483e+002f, 1.63783859e+002f
    };

    int npoints = (int)(sizeof(pt1data)/sizeof(pt1data[0])/2);
    printf("npoints = %d\n", npoints);  // npoints = 2

    Mat p1(1, npoints, CV_32FC2, pt1data);
    Mat p2(1, npoints, CV_32FC2, pt2data);
    Mat mask;

    // findHomography should raise an error since npoints < MIN_COUNT_OF_POINTS
    EXPECT_THROW(findHomography(p1, p2, RANSAC, 0.01, mask), cv::Exception);
}

TEST(Calib3d_Homography, not_normalized)
{
    Mat_<double> p1({5, 2}, {-1, -1, -2, -2, -1, 1, -2, 2, -1, 0});
    Mat_<double> p2({5, 2}, {0, -1, -1, -1, 0, 0, -1, 0, 0, -0.5});
    Mat_<double> ref({3, 3}, {
        0.74276086, 0., 0.74276086,
        0.18569022, 0.18569022, 0.,
        -0.37138043, 0., 0.
    });

    for (int method : std::vector<int>({0, RANSAC, LMEDS}))
    {
        Mat h = findHomography(p1, p2, method);
        for (auto it = h.begin<double>(); it != h.end<double>(); ++it) {
            ASSERT_FALSE(cvIsNaN(*it)) << cv::format("method %d\nResult:\n", method) << h;
        }
        if (h.at<double>(0, 0) * ref.at<double>(0, 0) < 0) {
            h *= -1;
        }
        ASSERT_LE(cv::norm(h, ref, NORM_INF), 1e-8) << cv::format("method %d\nResult:\n", method) << h;
    }
}

TEST(Calib3d_Homography, Refine)
{
    Mat_<double> p1({10, 2}, {41, -86, -87, 99, 66, -96, -86, -8, -67, 24,
                              -87, -76, -19, 89, 37, -4, -86, -86, -66, -53});
    Mat_<double> p2({10, 2}, {
        0.007723226608700208, -1.177541410622515,
        -0.1909072353027552, -0.4247610181930323,
        -0.134992319993638, -0.6469949816560389,
        -0.3570627451405215, 0.1811469436293486,
        -0.3005671881038939, -0.02325733734262935,
        -0.4404509481789249, 0.4851526464158342,
        0.6343346428859541, -3.396187657072353,
        -0.3539383967092603, 0.1469447227353143,
        -0.4526924606856586, 0.5296757109061794,
        -0.4309974583614644, 0.4522732662733471
    });
    hconcat(p1, Mat::ones(p1.rows, 1, CV_64F), p1);
    hconcat(p2, Mat::ones(p2.rows, 1, CV_64F), p2);

    for(int method : std::vector<int>({0, RANSAC, LMEDS}))
    {
        Mat h = findHomography(p1, p2, method);
        EXPECT_NEAR(h.at<double>(2, 2), 1.0, 1e-7);

        Mat proj = p1 * h.t();
        proj.col(0) /= proj.col(2);
        proj.col(1) /= proj.col(2);

        Mat error;
        cv::pow(p2.colRange(0, 2) - proj.colRange(0, 2), 2, error);
        cv::reduce(error, error, 1, REDUCE_SUM);
        cv::reduce(error, error, 0, REDUCE_AVG);
        EXPECT_LE(sqrt(error.at<double>(0, 0)), method == LMEDS ? 7e-2 : 7e-5);
    }
}

}} // namespace
