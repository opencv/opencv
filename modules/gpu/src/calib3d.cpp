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

#include "precomp.hpp"

#if !defined(HAVE_CUDA)

void cv::gpu::transformPoints(const GpuMat&, const Mat&, const Mat&, GpuMat&, Stream&) { throw_nogpu(); }

void cv::gpu::projectPoints(const GpuMat&, const Mat&, const Mat&, const Mat&, const Mat&, GpuMat&, Stream&) { throw_nogpu(); }

void cv::gpu::solvePnPRansac(const Mat&, const Mat&, const Mat&, const Mat&, Mat&, Mat&, bool, int, float, int, vector<int>*) { throw_nogpu(); }

#else

using namespace cv;
using namespace cv::gpu;

namespace cv { namespace gpu { namespace transform_points 
{
    void call(const DevMem2D_<float3> src, const float* rot, const float* transl, DevMem2D_<float3> dst, cudaStream_t stream);
}}}

namespace
{
    void transformPointsCaller(const GpuMat& src, const Mat& rvec, const Mat& tvec, GpuMat& dst, cudaStream_t stream)
    {
        CV_Assert(src.rows == 1 && src.cols > 0 && src.type() == CV_32FC3);
        CV_Assert(rvec.size() == Size(3, 1) && rvec.type() == CV_32F);
        CV_Assert(tvec.size() == Size(3, 1) && tvec.type() == CV_32F);

        // Convert rotation vector into matrix
        Mat rot;
        Rodrigues(rvec, rot);

        dst.create(src.size(), src.type());
        transform_points::call(src, rot.ptr<float>(), tvec.ptr<float>(), dst, stream);
    }
}

void cv::gpu::transformPoints(const GpuMat& src, const Mat& rvec, const Mat& tvec, GpuMat& dst, Stream& stream)
{
    ::transformPointsCaller(src, rvec, tvec, dst, StreamAccessor::getStream(stream));
}

namespace cv { namespace gpu { namespace project_points 
{
    void call(const DevMem2D_<float3> src, const float* rot, const float* transl, const float* proj, DevMem2D_<float2> dst, cudaStream_t stream);
}}}


namespace
{
    void projectPointsCaller(const GpuMat& src, const Mat& rvec, const Mat& tvec, const Mat& camera_mat, const Mat& dist_coef, GpuMat& dst, cudaStream_t stream)
    {
        CV_Assert(src.rows == 1 && src.cols > 0 && src.type() == CV_32FC3);
        CV_Assert(rvec.size() == Size(3, 1) && rvec.type() == CV_32F);
        CV_Assert(tvec.size() == Size(3, 1) && tvec.type() == CV_32F);
        CV_Assert(camera_mat.size() == Size(3, 3) && camera_mat.type() == CV_32F);
        CV_Assert(dist_coef.empty()); // Undistortion isn't supported

        // Convert rotation vector into matrix
        Mat rot;
        Rodrigues(rvec, rot);

        dst.create(src.size(), CV_32FC2);
        project_points::call(src, rot.ptr<float>(), tvec.ptr<float>(), camera_mat.ptr<float>(), dst,stream);
    }
}

void cv::gpu::projectPoints(const GpuMat& src, const Mat& rvec, const Mat& tvec, const Mat& camera_mat, const Mat& dist_coef, GpuMat& dst, Stream& stream)
{
    ::projectPointsCaller(src, rvec, tvec, camera_mat, dist_coef, dst, StreamAccessor::getStream(stream));
}


namespace cv { namespace gpu { namespace solve_pnp_ransac
{
    int maxNumIters();

    void computeHypothesisScores(
            const int num_hypotheses, const int num_points, const float* rot_matrices,
            const float3* transl_vectors, const float3* object, const float2* image,
            const float dist_threshold, int* hypothesis_scores);
}}}

namespace
{
    // Selects subset_size random different points from [0, num_points - 1] range
    void selectRandom(int subset_size, int num_points, vector<int>& subset)
    {
        subset.resize(subset_size);
        for (int i = 0; i < subset_size; ++i)
        {
            bool was;
            do
            {
                subset[i] = rand() % num_points;
                was = false;
                for (int j = 0; j < i; ++j)
                    if (subset[j] == subset[i])
                    {
                        was = true;
                        break;
                    }
            } while (was);
        }
    }

    // Computes rotation, translation pair for small subsets if the input data
    class TransformHypothesesGenerator
    {
    public:
        TransformHypothesesGenerator(const Mat& object_, const Mat& image_, const Mat& dist_coef_, 
                                     const Mat& camera_mat_, int num_points_, int subset_size_, 
                                     Mat rot_matrices_, Mat transl_vectors_)
                : object(&object_), image(&image_), dist_coef(&dist_coef_), camera_mat(&camera_mat_), 
                  num_points(num_points_), subset_size(subset_size_), rot_matrices(rot_matrices_), 
                  transl_vectors(transl_vectors_) {}

        void operator()(const BlockedRange& range) const
        {
            // Input data for generation of the current hypothesis
            vector<int> subset_indices(subset_size);
            Mat_<Point3f> object_subset(1, subset_size);
            Mat_<Point2f> image_subset(1, subset_size);

            // Current hypothesis data
            Mat rot_vec(1, 3, CV_64F);
            Mat rot_mat(3, 3, CV_64F);
            Mat transl_vec(1, 3, CV_64F);

            for (int iter = range.begin(); iter < range.end(); ++iter)
            {
                selectRandom(subset_size, num_points, subset_indices);
                for (int i = 0; i < subset_size; ++i)
                {
                   object_subset(0, i) = object->at<Point3f>(subset_indices[i]);
                   image_subset(0, i) = image->at<Point2f>(subset_indices[i]);
                }

                solvePnP(object_subset, image_subset, *camera_mat, *dist_coef, rot_vec, transl_vec);

                // Remember translation vector
                Mat transl_vec_ = transl_vectors.colRange(iter * 3, (iter + 1) * 3);
                transl_vec = transl_vec.reshape(0, 1);
                transl_vec.convertTo(transl_vec_, CV_32F);

                // Remember rotation matrix
                Rodrigues(rot_vec, rot_mat);
                Mat rot_mat_ = rot_matrices.colRange(iter * 9, (iter + 1) * 9).reshape(0, 3);
                rot_mat.convertTo(rot_mat_, CV_32F);
            }
        }

        const Mat* object;
        const Mat* image;
        const Mat* dist_coef;
        const Mat* camera_mat;
        int num_points;
        int subset_size;

        // Hypotheses storage (global)
        Mat rot_matrices;
        Mat transl_vectors;
    };
}

void cv::gpu::solvePnPRansac(const Mat& object, const Mat& image, const Mat& camera_mat,
                             const Mat& dist_coef, Mat& rvec, Mat& tvec, bool use_extrinsic_guess,
                             int num_iters, float max_dist, int min_inlier_count, 
                             vector<int>* inliers)
{
    CV_Assert(object.rows == 1 && object.cols > 0 && object.type() == CV_32FC3);
    CV_Assert(image.rows == 1 && image.cols > 0 && image.type() == CV_32FC2);
    CV_Assert(object.cols == image.cols);
    CV_Assert(camera_mat.size() == Size(3, 3) && camera_mat.type() == CV_32F);
    CV_Assert(!use_extrinsic_guess); // We don't support initial guess for now
    CV_Assert(num_iters <= solve_pnp_ransac::maxNumIters());

    const int subset_size = 4;
    const int num_points = object.cols;
    CV_Assert(num_points >= subset_size);

    // Unapply distortion and intrinsic camera transformations
    Mat eye_camera_mat = Mat::eye(3, 3, CV_32F);
    Mat empty_dist_coef;
    Mat image_normalized;
    undistortPoints(image, image_normalized, camera_mat, dist_coef, Mat(), eye_camera_mat);

    // Hypotheses storage (global)
    Mat rot_matrices(1, num_iters * 9, CV_32F);
    Mat transl_vectors(1, num_iters * 3, CV_32F);

    // Generate set of hypotheses using small subsets of the input data
    TransformHypothesesGenerator body(object, image_normalized, empty_dist_coef, eye_camera_mat, 
                                      num_points, subset_size, rot_matrices, transl_vectors);
    parallel_for(BlockedRange(0, num_iters), body);

    // Compute scores (i.e. number of inliers) for each hypothesis
    GpuMat d_object(object);
    GpuMat d_image_normalized(image_normalized);
    GpuMat d_hypothesis_scores(1, num_iters, CV_32S);
    solve_pnp_ransac::computeHypothesisScores(
            num_iters, num_points, rot_matrices.ptr<float>(), transl_vectors.ptr<float3>(),
            d_object.ptr<float3>(), d_image_normalized.ptr<float2>(), max_dist * max_dist, 
            d_hypothesis_scores.ptr<int>());

    // Find the best hypothesis index
    Point best_idx;
    double best_score;
    minMaxLoc(d_hypothesis_scores, NULL, &best_score, NULL, &best_idx);
    int num_inliers = static_cast<int>(best_score);

    // Extract the best hypothesis data

    Mat rot_mat = rot_matrices.colRange(best_idx.x * 9, (best_idx.x + 1) * 9).reshape(0, 3);
    Rodrigues(rot_mat, rvec);
    rvec = rvec.reshape(0, 1);

    tvec = transl_vectors.colRange(best_idx.x * 3, (best_idx.x + 1) * 3).clone();
    tvec = tvec.reshape(0, 1);

    // Build vector of inlier indices
    if (inliers != NULL)
    {
        inliers->clear();
        inliers->reserve(num_inliers);

        Point3f p, p_transf;
        Point2f p_proj;
        const float* rot = rot_mat.ptr<float>();
        const float* transl = tvec.ptr<float>();

        for (int i = 0; i < num_points; ++i)
        {
            p = object.at<Point3f>(0, i);
            p_transf.x = rot[0] * p.x + rot[1] * p.y + rot[2] * p.z + transl[0];
            p_transf.y = rot[3] * p.x + rot[4] * p.y + rot[5] * p.z + transl[1];
            p_transf.z = rot[6] * p.x + rot[7] * p.y + rot[8] * p.z + transl[2];
            p_proj.x = p_transf.x / p_transf.z;
            p_proj.y = p_transf.y / p_transf.z;
            if (norm(p_proj - image_normalized.at<Point2f>(0, i)) < max_dist)
                inliers->push_back(i);
        }
    }
}

#endif


