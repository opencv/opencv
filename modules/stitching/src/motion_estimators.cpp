/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
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

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace {

struct IncDistance
{
    IncDistance(vector<int> &dists) : dists(&dists[0]) {}
    void operator ()(const GraphEdge &edge) { dists[edge.to] = dists[edge.from] + 1; }
    int* dists;
};


struct CalcRotation
{
    CalcRotation(int num_images, const vector<MatchesInfo> &pairwise_matches, vector<CameraParams> &cameras)
        : num_images(num_images), pairwise_matches(&pairwise_matches[0]), cameras(&cameras[0]) {}

    void operator ()(const GraphEdge &edge)
    {
        int pair_idx = edge.from * num_images + edge.to;

        double f_from = cameras[edge.from].focal;
        double f_to = cameras[edge.to].focal;

        Mat K_from = Mat::eye(3, 3, CV_64F);
        K_from.at<double>(0, 0) = f_from;
        K_from.at<double>(1, 1) = f_from;

        Mat K_to = Mat::eye(3, 3, CV_64F);
        K_to.at<double>(0, 0) = f_to;
        K_to.at<double>(1, 1) = f_to;

        Mat R = K_from.inv() * pairwise_matches[pair_idx].H.inv() * K_to;
        cameras[edge.to].R = cameras[edge.from].R * R;
    }

    int num_images;
    const MatchesInfo* pairwise_matches;
    CameraParams* cameras;
};


//////////////////////////////////////////////////////////////////////////////

void calcDeriv(const Mat &err1, const Mat &err2, double h, Mat res)
{
    for (int i = 0; i < err1.rows; ++i)
        res.at<double>(i, 0) = (err2.at<double>(i, 0) - err1.at<double>(i, 0)) / h;
}

} // namespace


namespace cv {
namespace detail {

void HomographyBasedEstimator::estimate(const vector<ImageFeatures> &features, const vector<MatchesInfo> &pairwise_matches,
                                        vector<CameraParams> &cameras)
{
    LOGLN("Estimating rotations...");
    int64 t = getTickCount();

    const int num_images = static_cast<int>(features.size());

#if 0
    // Robustly estimate focal length from rotating cameras
    vector<Mat> Hs;
    for (int iter = 0; iter < 100; ++iter)
    {
        int len = 2 + rand()%(pairwise_matches.size() - 1);
        vector<int> subset;
        selectRandomSubset(len, pairwise_matches.size(), subset);
        Hs.clear();
        for (size_t i = 0; i < subset.size(); ++i)
            if (!pairwise_matches[subset[i]].H.empty())
                Hs.push_back(pairwise_matches[subset[i]].H);
        Mat_<double> K;
        if (Hs.size() >= 2)
        {
            if (calibrateRotatingCamera(Hs, K))
                cin.get();
        }
    }
#endif

    // Estimate focal length and set it for all cameras
    vector<double> focals;
    estimateFocal(features, pairwise_matches, focals);
    cameras.resize(num_images);
    for (int i = 0; i < num_images; ++i)
        cameras[i].focal = focals[i];

    // Restore global motion
    Graph span_tree;
    vector<int> span_tree_centers;
    findMaxSpanningTree(num_images, pairwise_matches, span_tree, span_tree_centers);
    span_tree.walkBreadthFirst(span_tree_centers[0], CalcRotation(num_images, pairwise_matches, cameras));

    // As calculations were performed under assumption that p.p. is in image center
    for (int i = 0; i < num_images; ++i)
    {
        cameras[i].ppx = 0.5 * features[i].img_size.width;
        cameras[i].ppy = 0.5 * features[i].img_size.height;
    }

    LOGLN("Estimating rotations, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
}


//////////////////////////////////////////////////////////////////////////////

void BundleAdjusterBase::estimate(const vector<ImageFeatures> &features,
                                  const vector<MatchesInfo> &pairwise_matches,
                                  vector<CameraParams> &cameras)
{
    LOG("Bundle adjustment");
    int64 t = getTickCount();

    num_images_ = static_cast<int>(features.size());
    features_ = &features[0];
    pairwise_matches_ = &pairwise_matches[0];

    setUpInitialCameraParams(cameras);

    // Leave only consistent image pairs
    edges_.clear();
    for (int i = 0; i < num_images_ - 1; ++i)
    {
        for (int j = i + 1; j < num_images_; ++j)
        {
            const MatchesInfo& matches_info = pairwise_matches_[i * num_images_ + j];
            if (matches_info.confidence > conf_thresh_)
                edges_.push_back(make_pair(i, j));
        }
    }

    // Compute number of correspondences
    total_num_matches_ = 0;
    for (size_t i = 0; i < edges_.size(); ++i)
        total_num_matches_ += static_cast<int>(pairwise_matches[edges_[i].first * num_images_ + 
                                                                edges_[i].second].num_inliers);

    CvLevMarq solver(num_images_ * num_params_per_cam_, 
                     total_num_matches_ * num_errs_per_measurement_,
                     cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 1000, DBL_EPSILON));

    Mat err, jac;
    CvMat matParams = cam_params_;
    cvCopy(&matParams, solver.param);

    int iter = 0;
    for(;;)
    {
        const CvMat* _param = 0;
        CvMat* _jac = 0;
        CvMat* _err = 0;

        bool proceed = solver.update(_param, _jac, _err);

        cvCopy(_param, &matParams);

        if (!proceed || !_err)
            break;

        if (_jac)
        {
            calcJacobian(jac);
            CvMat tmp = jac;
            cvCopy(&tmp, _jac);
        }

        if (_err)
        {
            calcError(err);
            LOG(".");
            iter++;
            CvMat tmp = err;
            cvCopy(&tmp, _err);
        }
    }

    LOGLN("");
    LOGLN("Bundle adjustment, final RMS error: " << sqrt(err.dot(err) / total_num_matches_));
    LOGLN("Bundle adjustment, iterations done: " << iter);

    obtainRefinedCameraParams(cameras);

    // Normalize motion to center image
    Graph span_tree;
    vector<int> span_tree_centers;
    findMaxSpanningTree(num_images_, pairwise_matches, span_tree, span_tree_centers);
    Mat R_inv = cameras[span_tree_centers[0]].R.inv();
    for (int i = 0; i < num_images_; ++i)
        cameras[i].R = R_inv * cameras[i].R;

    LOGLN("Bundle adjustment, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
}


//////////////////////////////////////////////////////////////////////////////

void BundleAdjusterReproj::setUpInitialCameraParams(const vector<CameraParams> &cameras)
{
    cam_params_.create(num_images_ * 7, 1, CV_64F);
    SVD svd;
    for (int i = 0; i < num_images_; ++i)
    {
        cam_params_.at<double>(i * 7, 0) = cameras[i].focal;
        cam_params_.at<double>(i * 7 + 1, 0) = cameras[i].ppx;
        cam_params_.at<double>(i * 7 + 2, 0) = cameras[i].ppy;
        cam_params_.at<double>(i * 7 + 3, 0) = cameras[i].aspect;

        svd(cameras[i].R, SVD::FULL_UV);
        Mat R = svd.u * svd.vt;
        if (determinant(R) < 0) 
            R *= -1;

        Mat rvec;
        Rodrigues(R, rvec);
        CV_Assert(rvec.type() == CV_32F);
        cam_params_.at<double>(i * 7 + 4, 0) = rvec.at<float>(0, 0);
        cam_params_.at<double>(i * 7 + 5, 0) = rvec.at<float>(1, 0);
        cam_params_.at<double>(i * 7 + 6, 0) = rvec.at<float>(2, 0);
    }
}


void BundleAdjusterReproj::obtainRefinedCameraParams(vector<CameraParams> &cameras) const
{
    for (int i = 0; i < num_images_; ++i)
    {
        cameras[i].focal = cam_params_.at<double>(i * 7, 0);
        cameras[i].ppx = cam_params_.at<double>(i * 7 + 1, 0);
        cameras[i].ppy = cam_params_.at<double>(i * 7 + 2, 0);
        cameras[i].aspect = cam_params_.at<double>(i * 7 + 3, 0);

        Mat rvec(3, 1, CV_64F);
        rvec.at<double>(0, 0) = cam_params_.at<double>(i * 7 + 4, 0);
        rvec.at<double>(1, 0) = cam_params_.at<double>(i * 7 + 5, 0);
        rvec.at<double>(2, 0) = cam_params_.at<double>(i * 7 + 6, 0);
        Rodrigues(rvec, cameras[i].R);

        Mat tmp;
        cameras[i].R.convertTo(tmp, CV_32F);
        cameras[i].R = tmp;
    }
}


void BundleAdjusterReproj::calcError(Mat &err)
{
    err.create(total_num_matches_ * 2, 1, CV_64F);

    int match_idx = 0;
    for (size_t edge_idx = 0; edge_idx < edges_.size(); ++edge_idx)
    {
        int i = edges_[edge_idx].first;
        int j = edges_[edge_idx].second;
        double f1 = cam_params_.at<double>(i * 7, 0);
        double f2 = cam_params_.at<double>(j * 7, 0);
        double ppx1 = cam_params_.at<double>(i * 7 + 1, 0);
        double ppx2 = cam_params_.at<double>(j * 7 + 1, 0);
        double ppy1 = cam_params_.at<double>(i * 7 + 2, 0);
        double ppy2 = cam_params_.at<double>(j * 7 + 2, 0);
        double a1 = cam_params_.at<double>(i * 7 + 3, 0);
        double a2 = cam_params_.at<double>(j * 7 + 3, 0);

        double R1[9];
        Mat R1_(3, 3, CV_64F, R1);
        Mat rvec(3, 1, CV_64F);
        rvec.at<double>(0, 0) = cam_params_.at<double>(i * 7 + 4, 0);
        rvec.at<double>(1, 0) = cam_params_.at<double>(i * 7 + 5, 0);
        rvec.at<double>(2, 0) = cam_params_.at<double>(i * 7 + 6, 0);
        Rodrigues(rvec, R1_);

        double R2[9];
        Mat R2_(3, 3, CV_64F, R2);
        rvec.at<double>(0, 0) = cam_params_.at<double>(j * 7 + 4, 0);
        rvec.at<double>(1, 0) = cam_params_.at<double>(j * 7 + 5, 0);
        rvec.at<double>(2, 0) = cam_params_.at<double>(j * 7 + 6, 0);
        Rodrigues(rvec, R2_);

        const ImageFeatures& features1 = features_[i];
        const ImageFeatures& features2 = features_[j];
        const MatchesInfo& matches_info = pairwise_matches_[i * num_images_ + j];

        Mat_<double> K1 = Mat::eye(3, 3, CV_64F);
        K1(0,0) = f1; K1(0,2) = ppx1;
        K1(1,1) = f1*a1; K1(1,2) = ppy1;

        Mat_<double> K2 = Mat::eye(3, 3, CV_64F);
        K2(0,0) = f2; K2(0,2) = ppx2;
        K2(1,1) = f2*a2; K2(1,2) = ppy2;

        Mat_<double> H = K2 * R2_.inv() * R1_ * K1.inv();

        for (size_t k = 0; k < matches_info.matches.size(); ++k)
        {
            if (!matches_info.inliers_mask[k])
                continue;

            const DMatch& m = matches_info.matches[k];
            Point2f p1 = features1.keypoints[m.queryIdx].pt;
            Point2f p2 = features2.keypoints[m.trainIdx].pt;
            double x = H(0,0)*p1.x + H(0,1)*p1.y + H(0,2);
            double y = H(1,0)*p1.x + H(1,1)*p1.y + H(1,2);
            double z = H(2,0)*p1.x + H(2,1)*p1.y + H(2,2);

            err.at<double>(2 * match_idx, 0) = p2.x - x/z;
            err.at<double>(2 * match_idx + 1, 0) = p2.y - y/z;
            match_idx++;
        }
    }
}


void BundleAdjusterReproj::calcJacobian(Mat &jac)
{
    jac.create(total_num_matches_ * 2, num_images_ * 7, CV_64F);
    jac.setTo(0);

    double val;
    const double step = 1e-4;

    for (int i = 0; i < num_images_; ++i)
    {
        if (refinement_mask_.at<uchar>(0, 0))
        {
            val = cam_params_.at<double>(i * 7, 0);
            cam_params_.at<double>(i * 7, 0) = val - step;
            calcError(err1_);
            cam_params_.at<double>(i * 7, 0) = val + step;
            calcError(err2_);
            calcDeriv(err1_, err2_, 2 * step, jac.col(i * 7));
            cam_params_.at<double>(i * 7, 0) = val;
        }
        if (refinement_mask_.at<uchar>(0, 2))        
        {
            val = cam_params_.at<double>(i * 7 + 1, 0);
            cam_params_.at<double>(i * 7 + 1, 0) = val - step;
            calcError(err1_);
            cam_params_.at<double>(i * 7 + 1, 0) = val + step;
            calcError(err2_);
            calcDeriv(err1_, err2_, 2 * step, jac.col(i * 7 + 1));
            cam_params_.at<double>(i * 7 + 1, 0) = val;
        }
        if (refinement_mask_.at<uchar>(1, 2))        
        {
            val = cam_params_.at<double>(i * 7 + 2, 0);
            cam_params_.at<double>(i * 7 + 2, 0) = val - step;
            calcError(err1_);
            cam_params_.at<double>(i * 7 + 2, 0) = val + step;
            calcError(err2_);
            calcDeriv(err1_, err2_, 2 * step, jac.col(i * 7 + 2));
            cam_params_.at<double>(i * 7 + 2, 0) = val;
        }
        if (refinement_mask_.at<uchar>(1, 1))
        {
            val = cam_params_.at<double>(i * 7 + 3, 0);
            cam_params_.at<double>(i * 7 + 3, 0) = val - step;
            calcError(err1_);
            cam_params_.at<double>(i * 7 + 3, 0) = val + step;
            calcError(err2_);
            calcDeriv(err1_, err2_, 2 * step, jac.col(i * 7 + 3));
            cam_params_.at<double>(i * 7 + 3, 0) = val;
        }
        for (int j = 4; j < 7; ++j)
        {
            val = cam_params_.at<double>(i * 7 + j, 0);
            cam_params_.at<double>(i * 7 + j, 0) = val - step;
            calcError(err1_);
            cam_params_.at<double>(i * 7 + j, 0) = val + step;
            calcError(err2_);
            calcDeriv(err1_, err2_, 2 * step, jac.col(i * 7 + j));
            cam_params_.at<double>(i * 7 + j, 0) = val;
        }
    }
}


//////////////////////////////////////////////////////////////////////////////

void BundleAdjusterRay::setUpInitialCameraParams(const vector<CameraParams> &cameras)
{
    cam_params_.create(num_images_ * 4, 1, CV_64F);
    SVD svd;
    for (int i = 0; i < num_images_; ++i)
    {
        cam_params_.at<double>(i * 4, 0) = cameras[i].focal;

        svd(cameras[i].R, SVD::FULL_UV);
        Mat R = svd.u * svd.vt;
        if (determinant(R) < 0) 
            R *= -1;

        Mat rvec;
        Rodrigues(R, rvec);
        CV_Assert(rvec.type() == CV_32F);
        cam_params_.at<double>(i * 4 + 1, 0) = rvec.at<float>(0, 0);
        cam_params_.at<double>(i * 4 + 2, 0) = rvec.at<float>(1, 0);
        cam_params_.at<double>(i * 4 + 3, 0) = rvec.at<float>(2, 0);
    }
}


void BundleAdjusterRay::obtainRefinedCameraParams(vector<CameraParams> &cameras) const
{
    for (int i = 0; i < num_images_; ++i)
    {
        cameras[i].focal = cam_params_.at<double>(i * 4, 0);

        Mat rvec(3, 1, CV_64F);
        rvec.at<double>(0, 0) = cam_params_.at<double>(i * 4 + 1, 0);
        rvec.at<double>(1, 0) = cam_params_.at<double>(i * 4 + 2, 0);
        rvec.at<double>(2, 0) = cam_params_.at<double>(i * 4 + 3, 0);
        Rodrigues(rvec, cameras[i].R);

        Mat tmp;
        cameras[i].R.convertTo(tmp, CV_32F);
        cameras[i].R = tmp;
    }
}


void BundleAdjusterRay::calcError(Mat &err)
{
    err.create(total_num_matches_ * 3, 1, CV_64F);

    int match_idx = 0;
    for (size_t edge_idx = 0; edge_idx < edges_.size(); ++edge_idx)
    {
        int i = edges_[edge_idx].first;
        int j = edges_[edge_idx].second;
        double f1 = cam_params_.at<double>(i * 4, 0);
        double f2 = cam_params_.at<double>(j * 4, 0);

        double R1[9];
        Mat R1_(3, 3, CV_64F, R1);
        Mat rvec(3, 1, CV_64F);
        rvec.at<double>(0, 0) = cam_params_.at<double>(i * 4 + 1, 0);
        rvec.at<double>(1, 0) = cam_params_.at<double>(i * 4 + 2, 0);
        rvec.at<double>(2, 0) = cam_params_.at<double>(i * 4 + 3, 0);
        Rodrigues(rvec, R1_);

        double R2[9];
        Mat R2_(3, 3, CV_64F, R2);
        rvec.at<double>(0, 0) = cam_params_.at<double>(j * 4 + 1, 0);
        rvec.at<double>(1, 0) = cam_params_.at<double>(j * 4 + 2, 0);
        rvec.at<double>(2, 0) = cam_params_.at<double>(j * 4 + 3, 0);
        Rodrigues(rvec, R2_);

        const ImageFeatures& features1 = features_[i];
        const ImageFeatures& features2 = features_[j];
        const MatchesInfo& matches_info = pairwise_matches_[i * num_images_ + j];

        Mat_<double> K1 = Mat::eye(3, 3, CV_64F);
        K1(0,0) = f1; K1(0,2) = features1.img_size.width * 0.5;
        K1(1,1) = f1; K1(1,2) = features1.img_size.height * 0.5;

        Mat_<double> K2 = Mat::eye(3, 3, CV_64F);
        K2(0,0) = f2; K2(0,2) = features2.img_size.width * 0.5;
        K2(1,1) = f2; K2(1,2) = features2.img_size.height * 0.5;

        Mat_<double> H1 = R1_ * K1.inv();
        Mat_<double> H2 = R2_ * K2.inv();

        for (size_t k = 0; k < matches_info.matches.size(); ++k)
        {
            if (!matches_info.inliers_mask[k])
                continue;

            const DMatch& m = matches_info.matches[k];

            Point2f p1 = features1.keypoints[m.queryIdx].pt;
            double x1 = H1(0,0)*p1.x + H1(0,1)*p1.y + H1(0,2);
            double y1 = H1(1,0)*p1.x + H1(1,1)*p1.y + H1(1,2);
            double z1 = H1(2,0)*p1.x + H1(2,1)*p1.y + H1(2,2);
            double len = sqrt(x1*x1 + y1*y1 + z1*z1);
            x1 /= len; y1 /= len; z1 /= len;

            Point2f p2 = features2.keypoints[m.trainIdx].pt;
            double x2 = H2(0,0)*p2.x + H2(0,1)*p2.y + H2(0,2);
            double y2 = H2(1,0)*p2.x + H2(1,1)*p2.y + H2(1,2);
            double z2 = H2(2,0)*p2.x + H2(2,1)*p2.y + H2(2,2);
            len = sqrt(x2*x2 + y2*y2 + z2*z2);
            x2 /= len; y2 /= len; z2 /= len;

            double mult = sqrt(f1 * f2);
            err.at<double>(3 * match_idx, 0) = mult * (x1 - x2);
            err.at<double>(3 * match_idx + 1, 0) = mult * (y1 - y2);
            err.at<double>(3 * match_idx + 2, 0) = mult * (z1 - z2);           

            match_idx++;
        }
    }
}


void BundleAdjusterRay::calcJacobian(Mat &jac)
{
    jac.create(total_num_matches_ * 3, num_images_ * 4, CV_64F);

    double val;
    const double step = 1e-3;

    for (int i = 0; i < num_images_; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            val = cam_params_.at<double>(i * 4 + j, 0);
            cam_params_.at<double>(i * 4 + j, 0) = val - step;
            calcError(err1_);
            cam_params_.at<double>(i * 4 + j, 0) = val + step;
            calcError(err2_);
            calcDeriv(err1_, err2_, 2 * step, jac.col(i * 4 + j));
            cam_params_.at<double>(i * 4 + j, 0) = val;
        }
    }
}


//////////////////////////////////////////////////////////////////////////////

void waveCorrect(vector<Mat> &rmats, WaveCorrectKind kind)
{
    LOGLN("Wave correcting...");
    int64 t = getTickCount();

    Mat moment = Mat::zeros(3, 3, CV_32F);
    for (size_t i = 0; i < rmats.size(); ++i)
    {
        Mat col = rmats[i].col(0);
        moment += col * col.t();
    }
    Mat eigen_vals, eigen_vecs;
    eigen(moment, eigen_vals, eigen_vecs);

    Mat rg1;
    if (kind == WAVE_CORRECT_HORIZ)
        rg1 = eigen_vecs.row(2).t();
    else if (kind == WAVE_CORRECT_VERT)
        rg1 = eigen_vecs.row(0).t();
    else
        CV_Error(CV_StsBadArg, "unsupported kind of wave correction");

    Mat img_k = Mat::zeros(3, 1, CV_32F);
    for (size_t i = 0; i < rmats.size(); ++i)
        img_k += rmats[i].col(2);
    Mat rg0 = rg1.cross(img_k);
    rg0 /= norm(rg0);

    Mat rg2 = rg0.cross(rg1);

    double conf = 0;
    if (kind == WAVE_CORRECT_HORIZ)
    {
        for (size_t i = 0; i < rmats.size(); ++i)
            conf += rg0.dot(rmats[i].col(0));
        if (conf < 0)
        {
            rg0 *= -1;
            rg1 *= -1;
        }
    }
    else if (kind == WAVE_CORRECT_VERT)
    {
        for (size_t i = 0; i < rmats.size(); ++i)
            conf -= rg1.dot(rmats[i].col(0));
        cout << conf << endl;
        if (conf < 0)
        {
            rg0 *= -1;
            rg1 *= -1;
        }
    }

    Mat R = Mat::zeros(3, 3, CV_32F);
    Mat tmp = R.row(0);
    Mat(rg0.t()).copyTo(tmp);
    tmp = R.row(1);
    Mat(rg1.t()).copyTo(tmp);
    tmp = R.row(2);
    Mat(rg2.t()).copyTo(tmp);

    for (size_t i = 0; i < rmats.size(); ++i)
        rmats[i] = R * rmats[i];

    LOGLN("Wave correcting, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
}


//////////////////////////////////////////////////////////////////////////////

string matchesGraphAsString(vector<string> &pathes, vector<MatchesInfo> &pairwise_matches,
                                float conf_threshold)
{
    stringstream str;
    str << "graph matches_graph{\n";

    const int num_images = static_cast<int>(pathes.size());
    set<pair<int,int> > span_tree_edges;
    DisjointSets comps(num_images);

    for (int i = 0; i < num_images; ++i)
    {
        for (int j = 0; j < num_images; ++j)
        {
            if (pairwise_matches[i*num_images + j].confidence < conf_threshold)
                continue;
            int comp1 = comps.findSetByElem(i);
            int comp2 = comps.findSetByElem(j);
            if (comp1 != comp2)
            {
                comps.mergeSets(comp1, comp2);
                span_tree_edges.insert(make_pair(i, j));
            }
        }
    }

    for (set<pair<int,int> >::const_iterator itr = span_tree_edges.begin();
         itr != span_tree_edges.end(); ++itr)
    {
        pair<int,int> edge = *itr;
        if (span_tree_edges.find(edge) != span_tree_edges.end())
        {
            string name_src = pathes[edge.first];
            size_t prefix_len = name_src.find_last_of("/\\");
            if (prefix_len != string::npos) prefix_len++; else prefix_len = 0;
            name_src = name_src.substr(prefix_len, name_src.size() - prefix_len);

            string name_dst = pathes[edge.second];
            prefix_len = name_dst.find_last_of("/\\");
            if (prefix_len != string::npos) prefix_len++; else prefix_len = 0;
            name_dst = name_dst.substr(prefix_len, name_dst.size() - prefix_len);

            int pos = edge.first*num_images + edge.second;
            str << "\"" << name_src << "\" -- \"" << name_dst << "\""
                << "[label=\"Nm=" << pairwise_matches[pos].matches.size()
                << ", Ni=" << pairwise_matches[pos].num_inliers
                << ", C=" << pairwise_matches[pos].confidence << "\"];\n";
        }
    }

    for (size_t i = 0; i < comps.size.size(); ++i)
    {
        if (comps.size[comps.findSetByElem(i)] == 1)
        {
            string name = pathes[i];
            size_t prefix_len = name.find_last_of("/\\");
            if (prefix_len != string::npos) prefix_len++; else prefix_len = 0;
            name = name.substr(prefix_len, name.size() - prefix_len);
            str << "\"" << name << "\";\n";
        }
    }

    str << "}";
    return str.str();
}

vector<int> leaveBiggestComponent(vector<ImageFeatures> &features,  vector<MatchesInfo> &pairwise_matches,
                                      float conf_threshold)
{
    const int num_images = static_cast<int>(features.size());

    DisjointSets comps(num_images);
    for (int i = 0; i < num_images; ++i)
    {
        for (int j = 0; j < num_images; ++j)
        {
            if (pairwise_matches[i*num_images + j].confidence < conf_threshold)
                continue;
            int comp1 = comps.findSetByElem(i);
            int comp2 = comps.findSetByElem(j);
            if (comp1 != comp2) 
                comps.mergeSets(comp1, comp2);
        }
    }

    int max_comp = static_cast<int>(max_element(comps.size.begin(), comps.size.end()) - comps.size.begin());

    vector<int> indices;
    vector<int> indices_removed;
    for (int i = 0; i < num_images; ++i)
        if (comps.findSetByElem(i) == max_comp)
            indices.push_back(i);    
        else
            indices_removed.push_back(i);

    vector<ImageFeatures> features_subset;
    vector<MatchesInfo> pairwise_matches_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        features_subset.push_back(features[indices[i]]);
        for (size_t j = 0; j < indices.size(); ++j)
        {
            pairwise_matches_subset.push_back(pairwise_matches[indices[i]*num_images + indices[j]]);
            pairwise_matches_subset.back().src_img_idx = static_cast<int>(i);
            pairwise_matches_subset.back().dst_img_idx = static_cast<int>(j);
        }
    }

    if (static_cast<int>(features_subset.size()) == num_images)
        return indices;

    LOG("Removed some images, because can't match them: (");
    LOG(indices_removed[0]+1);
    for (size_t i = 1; i < indices_removed.size(); ++i) 
        LOG(", " << indices_removed[i]+1);
    LOGLN("). Try to decrease --match_conf value.");

    features = features_subset;
    pairwise_matches = pairwise_matches_subset;

    return indices;
}


void findMaxSpanningTree(int num_images, const vector<MatchesInfo> &pairwise_matches,
                             Graph &span_tree, vector<int> &centers)
{
    Graph graph(num_images);
    vector<GraphEdge> edges;

    // Construct images graph and remember its edges
    for (int i = 0; i < num_images; ++i)
    {
        for (int j = 0; j < num_images; ++j)
        {
            if (pairwise_matches[i * num_images + j].H.empty())
                continue;
            float conf = static_cast<float>(pairwise_matches[i * num_images + j].num_inliers);
            graph.addEdge(i, j, conf);
            edges.push_back(GraphEdge(i, j, conf));
        }
    }

    DisjointSets comps(num_images);
    span_tree.create(num_images);
    vector<int> span_tree_powers(num_images, 0);

    // Find maximum spanning tree
    sort(edges.begin(), edges.end(), greater<GraphEdge>());
    for (size_t i = 0; i < edges.size(); ++i)
    {
        int comp1 = comps.findSetByElem(edges[i].from);
        int comp2 = comps.findSetByElem(edges[i].to);
        if (comp1 != comp2)
        {
            comps.mergeSets(comp1, comp2);
            span_tree.addEdge(edges[i].from, edges[i].to, edges[i].weight);
            span_tree.addEdge(edges[i].to, edges[i].from, edges[i].weight);
            span_tree_powers[edges[i].from]++;
            span_tree_powers[edges[i].to]++;
        }
    }

    // Find spanning tree leafs
    vector<int> span_tree_leafs;
    for (int i = 0; i < num_images; ++i)
        if (span_tree_powers[i] == 1)
            span_tree_leafs.push_back(i);

    // Find maximum distance from each spanning tree vertex
    vector<int> max_dists(num_images, 0);
    vector<int> cur_dists;
    for (size_t i = 0; i < span_tree_leafs.size(); ++i)
    {
        cur_dists.assign(num_images, 0);
        span_tree.walkBreadthFirst(span_tree_leafs[i], IncDistance(cur_dists));
        for (int j = 0; j < num_images; ++j)
            max_dists[j] = max(max_dists[j], cur_dists[j]);
    }

    // Find min-max distance
    int min_max_dist = max_dists[0];
    for (int i = 1; i < num_images; ++i)
        if (min_max_dist > max_dists[i])
            min_max_dist = max_dists[i];

    // Find spanning tree centers
    centers.clear();
    for (int i = 0; i < num_images; ++i)
        if (max_dists[i] == min_max_dist)
            centers.push_back(i);
    CV_Assert(centers.size() > 0 && centers.size() <= 2);
}

} // namespace detail
} // namespace cv

