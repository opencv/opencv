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
#include <time.h>

#define CALIB3D_HOMOGRAPHY_ERROR_MATRIX_SIZE 1
#define CALIB3D_HOMOGRAPHY_ERROR_MATRIX_DIFF 2
#define CALIB3D_HOMOGRAPHY_ERROR_REPROJ_DIFF 3
#define CALIB3D_HOMOGRAPHY_ERROR_RANSAC_MASK 4
#define CALIB3D_HOMOGRAPHY_ERROR_RANSAC_DIFF 5

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
#define COUNT_NORM_TYPES 3
#define METHODS_COUNT 3

int NORM_TYPE[COUNT_NORM_TYPES] = {cv::NORM_L1, cv::NORM_L2, cv::NORM_INF};
int METHOD[METHODS_COUNT] = {0, cv::RANSAC, cv::LMEDS};

using namespace cv;
using namespace std;

class CV_HomographyTest: public cvtest::ArrayTest
{
public:
    CV_HomographyTest();
    ~CV_HomographyTest();

    void run (int);

protected:

    int method;
    int image_size;
    double reproj_threshold;
    double sigma;

private:
    float max_diff, max_2diff;
    bool check_matrix_size(const cv::Mat& H);
    bool check_matrix_diff(const cv::Mat& original, const cv::Mat& found, const int norm_type, double &diff);
    int check_ransac_mask_1(const Mat& src, const Mat& mask);
    int check_ransac_mask_2(const Mat& original_mask, const Mat& found_mask);

    void print_information_1(int j, int N, int method, const Mat& H);
    void print_information_2(int j, int N, int method, const Mat& H, const Mat& H_res, int k, double diff);
    void print_information_3(int j, int N, const Mat& mask);
    void print_information_4(int method, int j, int N, int k, int l, double diff);
    void print_information_5(int method, int j, int N, int l, double diff);
    void print_information_6(int j, int N, int k, double diff, bool value);
    void print_information_7(int j, int N, int k, double diff, bool original_value, bool found_value);
    void print_information_8(int j, int N, int k, int l, double diff);
};

CV_HomographyTest::CV_HomographyTest() : max_diff(1e-2f), max_2diff(2e-2f)
{
    method = 0;
    image_size = 100;
    reproj_threshold = 3.0;
    sigma = 0.01;
}

CV_HomographyTest::~CV_HomographyTest() {}

bool CV_HomographyTest::check_matrix_size(const cv::Mat& H)
{
    return (H.rows == 3) && (H.cols == 3);
}

bool CV_HomographyTest::check_matrix_diff(const cv::Mat& original, const cv::Mat& found, const int norm_type, double &diff)
{
    diff = cvtest::norm(original, found, norm_type);
    return diff <= max_diff;
}

int CV_HomographyTest::check_ransac_mask_1(const Mat& src, const Mat& mask)
{
    if (!(mask.cols == 1) && (mask.rows == src.cols)) return 1;
    if (countNonZero(mask) < mask.rows) return 2;
    for (int i = 0; i < mask.rows; ++i) if (mask.at<uchar>(i, 0) > 1) return 3;
    return 0;
}

int CV_HomographyTest::check_ransac_mask_2(const Mat& original_mask, const Mat& found_mask)
{
    if (!(found_mask.cols == 1) && (found_mask.rows == original_mask.rows)) return 1;
    for (int i = 0; i < found_mask.rows; ++i) if (found_mask.at<uchar>(i, 0) > 1) return 2;
    return 0;
}

void CV_HomographyTest::print_information_1(int j, int N, int _method, const Mat& H)
{
    cout << endl; cout << "Checking for homography matrix sizes..." << endl; cout << endl;
    cout << "Type of srcPoints: "; if ((j>-1) && (j<2)) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>";
    cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
    cout << "Count of points: " << N << endl; cout << endl;
    cout << "Method: "; if (_method == 0) cout << 0; else if (_method == 8) cout << "RANSAC"; else cout << "LMEDS"; cout << endl;
    cout << "Homography matrix:" << endl; cout << endl;
    cout << H << endl; cout << endl;
    cout << "Number of rows: " << H.rows << "   Number of cols: " << H.cols << endl; cout << endl;
}

void CV_HomographyTest::print_information_2(int j, int N, int _method, const Mat& H, const Mat& H_res, int k, double diff)
{
    cout << endl; cout << "Checking for accuracy of homography matrix computing..." << endl; cout << endl;
    cout << "Type of srcPoints: "; if ((j>-1) && (j<2)) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>";
    cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
    cout << "Count of points: " << N << endl; cout << endl;
    cout << "Method: "; if (_method == 0) cout << 0; else if (_method == 8) cout << "RANSAC"; else cout << "LMEDS"; cout << endl;
    cout << "Original matrix:" << endl; cout << endl;
    cout << H << endl; cout << endl;
    cout << "Found matrix:" << endl; cout << endl;
    cout << H_res << endl; cout << endl;
    cout << "Norm type using in criteria: "; if (NORM_TYPE[k] == 1) cout << "INF"; else if (NORM_TYPE[k] == 2) cout << "L1"; else cout << "L2"; cout << endl;
    cout << "Difference between matrices: " << diff << endl;
    cout << "Maximum allowed difference: " << max_diff << endl; cout << endl;
}

void CV_HomographyTest::print_information_3(int j, int N, const Mat& mask)
{
    cout << endl; cout << "Checking for inliers/outliers mask..." << endl; cout << endl;
    cout << "Type of srcPoints: "; if ((j>-1) && (j<2)) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>";
    cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
    cout << "Count of points: " << N << endl; cout << endl;
    cout << "Method: RANSAC" << endl;
    cout << "Found mask:" << endl; cout << endl;
    cout << mask << endl; cout << endl;
    cout << "Number of rows: " << mask.rows << "   Number of cols: " << mask.cols << endl; cout << endl;
}

void CV_HomographyTest::print_information_4(int _method, int j, int N, int k, int l, double diff)
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
    cout << "Maxumum allowed difference: " << max_2diff << endl; cout << endl;
}

void CV_HomographyTest::print_information_5(int _method, int j, int N, int l, double diff)
{
    cout << endl; cout << "Checking for accuracy of reprojection error computing..." << endl; cout << endl;
    cout << "Method: "; if (_method == 0) cout << 0 << endl; else cout << "CV_LMEDS" << endl;
    cout << "Type of srcPoints: "; if ((j>-1) && (j<2)) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>";
    cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
    cout << "Sigma of normal noise: " << sigma << endl;
    cout << "Count of points: " << N << endl;
    cout << "Norm type using in criteria: "; if (NORM_TYPE[l] == 1) cout << "INF"; else if (NORM_TYPE[l] == 2) cout << "L1"; else cout << "L2"; cout << endl;
    cout << "Difference with noise of points: " << diff << endl;
    cout << "Maxumum allowed difference: " << max_diff << endl; cout << endl;
}

void CV_HomographyTest::print_information_6(int j, int N, int k, double diff, bool value)
{
    cout << endl; cout << "Checking for inliers/outliers mask..." << endl; cout << endl;
    cout << "Method: RANSAC" << endl;
    cout << "Type of srcPoints: "; if ((j>-1) && (j<2)) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>";
    cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
    cout << "Count of points: " << N << "   " << endl;
    cout << "Number of point: " << k << "   " << endl;
    cout << "Reprojection error for this point: " << diff << "   " << endl;
    cout << "Reprojection error threshold: " << reproj_threshold << "   " << endl;
    cout << "Value of found mask: "<< value << endl; cout << endl;
}

void CV_HomographyTest::print_information_7(int j, int N, int k, double diff, bool original_value, bool found_value)
{
    cout << endl; cout << "Checking for inliers/outliers mask..." << endl; cout << endl;
    cout << "Method: RANSAC" << endl;
    cout << "Type of srcPoints: "; if ((j>-1) && (j<2)) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>";
    cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
    cout << "Count of points: " << N << "   " << endl;
    cout << "Number of point: " << k << "   " << endl;
    cout << "Reprojection error for this point: " << diff << "   " << endl;
    cout << "Reprojection error threshold: " << reproj_threshold << "   " << endl;
    cout << "Value of original mask: "<< original_value << "   Value of found mask: " << found_value << endl; cout << endl;
}

void CV_HomographyTest::print_information_8(int j, int N, int k, int l, double diff)
{
    cout << endl; cout << "Checking for reprojection error of inlier..." << endl; cout << endl;
    cout << "Method: RANSAC" << endl;
    cout << "Sigma of normal noise: " << sigma << endl;
    cout << "Type of srcPoints: "; if ((j>-1) && (j<2)) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>";
    cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
    cout << "Count of points: " << N << "   " << endl;
    cout << "Number of point: " << k << "   " << endl;
    cout << "Norm type using in criteria: "; if (NORM_TYPE[l] == 1) cout << "INF"; else if (NORM_TYPE[l] == 2) cout << "L1"; else cout << "L2"; cout << endl;
    cout << "Difference with noise of point: " << diff << endl;
    cout << "Maxumum allowed difference: " << max_2diff << endl; cout << endl;
}

void CV_HomographyTest::run(int)
{
    for (int N = 4; N <= MAX_COUNT_OF_POINTS; ++N)
    {
        RNG& rng = ts->get_rng();

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
            method = METHOD[i];
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
                            CV_Error(CALIB3D_HOMOGRAPHY_ERROR_MATRIX_SIZE, MESSAGE_MATRIX_SIZE);
                            return;
                        }

                        double diff;

                        for (int k = 0; k < COUNT_NORM_TYPES; ++k)
                            if (!check_matrix_diff(H_64, H_res_64[j], NORM_TYPE[k], diff))
                            {
                            print_information_2(j, N, method, H_64, H_res_64[j], k, diff);
                            CV_Error(CALIB3D_HOMOGRAPHY_ERROR_MATRIX_DIFF, MESSAGE_MATRIX_DIFF);
                            return;
                        }
                    }

                    continue;
                }
            case RANSAC:
                {
                    cv::Mat mask [4]; double diff;

                    Mat H_res_64 [4] = { cv::findHomography(src_mat_2f, dst_mat_2f, RANSAC, reproj_threshold, mask[0]),
                                         cv::findHomography(src_mat_2f, dst_vec, RANSAC, reproj_threshold, mask[1]),
                                         cv::findHomography(src_vec, dst_mat_2f, RANSAC, reproj_threshold, mask[2]),
                                         cv::findHomography(src_vec, dst_vec, RANSAC, reproj_threshold, mask[3]) };

                    for (int j = 0; j < 4; ++j)
                    {

                        if (!check_matrix_size(H_res_64[j]))
                        {
                            print_information_1(j, N, method, H_res_64[j]);
                            CV_Error(CALIB3D_HOMOGRAPHY_ERROR_MATRIX_SIZE, MESSAGE_MATRIX_SIZE);
                            return;
                        }

                        for (int k = 0; k < COUNT_NORM_TYPES; ++k)
                            if (!check_matrix_diff(H_64, H_res_64[j], NORM_TYPE[k], diff))
                            {
                            print_information_2(j, N, method, H_64, H_res_64[j], k, diff);
                            CV_Error(CALIB3D_HOMOGRAPHY_ERROR_MATRIX_DIFF, MESSAGE_MATRIX_DIFF);
                            return;
                        }

                        int code = check_ransac_mask_1(src_mat_2f, mask[j]);

                        if (code)
                        {
                            print_information_3(j, N, mask[j]);

                            switch (code)
                            {
                            case 1: { CV_Error(CALIB3D_HOMOGRAPHY_ERROR_RANSAC_MASK, MESSAGE_RANSAC_MASK_1); break; }
                            case 2: { CV_Error(CALIB3D_HOMOGRAPHY_ERROR_RANSAC_MASK, MESSAGE_RANSAC_MASK_2); break; }
                            case 3: { CV_Error(CALIB3D_HOMOGRAPHY_ERROR_RANSAC_MASK, MESSAGE_RANSAC_MASK_3); break; }

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
            method = METHOD[i];
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
                            CV_Error(CALIB3D_HOMOGRAPHY_ERROR_MATRIX_SIZE, MESSAGE_MATRIX_SIZE);
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
                                CV_Error(CALIB3D_HOMOGRAPHY_ERROR_REPROJ_DIFF, MESSAGE_REPROJ_DIFF_1);
                                return;
                            }

                        }

                        for (int l = 0; l < COUNT_NORM_TYPES; ++l)
                            if (cv::norm(dst_res_3d, dst_mat_3d, NORM_TYPE[l]) - cv::norm(noise_2d, NORM_TYPE[l]) > max_diff)
                            {
                            print_information_5(method, j, N, l, cv::norm(dst_res_3d, dst_mat_3d, NORM_TYPE[l]) - cv::norm(noise_2d, NORM_TYPE[l]));
                            CV_Error(CALIB3D_HOMOGRAPHY_ERROR_REPROJ_DIFF, MESSAGE_REPROJ_DIFF_2);
                            return;
                        }

                    }

                    continue;
                }
            case RANSAC:
                {
                    cv::Mat mask_res [4];

                    Mat H_res_64 [4] = { cv::findHomography(src_mat_2f, dst_mat_2f, RANSAC, reproj_threshold, mask_res[0]),
                                         cv::findHomography(src_mat_2f, dst_vec, RANSAC, reproj_threshold, mask_res[1]),
                                         cv::findHomography(src_vec, dst_mat_2f, RANSAC, reproj_threshold, mask_res[2]),
                                         cv::findHomography(src_vec, dst_vec, RANSAC, reproj_threshold, mask_res[3]) };

                    for (int j = 0; j < 4; ++j)
                    {
                        if (!check_matrix_size(H_res_64[j]))
                        {
                            print_information_1(j, N, method, H_res_64[j]);
                            CV_Error(CALIB3D_HOMOGRAPHY_ERROR_MATRIX_SIZE, MESSAGE_MATRIX_SIZE);
                            return;
                        }

                        int code = check_ransac_mask_2(mask, mask_res[j]);

                        if (code)
                        {
                            print_information_3(j, N, mask_res[j]);

                            switch (code)
                            {
                            case 1: { CV_Error(CALIB3D_HOMOGRAPHY_ERROR_RANSAC_MASK, MESSAGE_RANSAC_MASK_1); break; }
                            case 2: { CV_Error(CALIB3D_HOMOGRAPHY_ERROR_RANSAC_MASK, MESSAGE_RANSAC_MASK_3); break; }

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
                                print_information_6(j, N, k, diff, mask_res[j].at<bool>(k, 0));
                                CV_Error(CALIB3D_HOMOGRAPHY_ERROR_RANSAC_MASK, MESSAGE_RANSAC_MASK_4);
                                return;
                            }

                            if (mask.at<bool>(k, 0) && !mask_res[j].at<bool>(k, 0))
                            {
                                print_information_7(j, N, k, diff, mask.at<bool>(k, 0), mask_res[j].at<bool>(k, 0));
                                CV_Error(CALIB3D_HOMOGRAPHY_ERROR_RANSAC_MASK, MESSAGE_RANSAC_MASK_5);
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
                                        print_information_8(j, N, k, l, diff - cv::norm(noise_2d, NORM_TYPE[l]));
                                        CV_Error(CALIB3D_HOMOGRAPHY_ERROR_RANSAC_DIFF, MESSAGE_RANSAC_DIFF);
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
    }
}

TEST(Calib3d_Homography, accuracy) { CV_HomographyTest test; test.safe_run(); }
