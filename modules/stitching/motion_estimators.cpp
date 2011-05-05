#include <algorithm>
#include <functional>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "focal_estimators.hpp"
#include "motion_estimators.hpp"
#include "util.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

//////////////////////////////////////////////////////////////////////////////

namespace
{
    class CpuSurfFeaturesFinder : public FeaturesFinder
    {
    public:
        inline CpuSurfFeaturesFinder() 
        {
            detector_ = new SurfFeatureDetector(500);
            extractor_ = new SurfDescriptorExtractor();
        }

    protected:
        void find(const vector<Mat> &images, vector<ImageFeatures> &features);

    private:
        Ptr<FeatureDetector> detector_;
        Ptr<DescriptorExtractor> extractor_;
    };

    void CpuSurfFeaturesFinder::find(const vector<Mat> &images, vector<ImageFeatures> &features)
    {
        // Make images gray
        vector<Mat> gray_images(images.size());
        for (size_t i = 0; i < images.size(); ++i)
        {
            CV_Assert(images[i].depth() == CV_8U);
            cvtColor(images[i], gray_images[i], CV_BGR2GRAY);
        }

        features.resize(images.size());

        // Find keypoints in all images
        for (size_t i = 0; i < images.size(); ++i)
        {
            detector_->detect(gray_images[i], features[i].keypoints);
            extractor_->compute(gray_images[i], features[i].keypoints, features[i].descriptors);
        }
    }
    
    class GpuSurfFeaturesFinder : public FeaturesFinder
    {
    public:
        inline GpuSurfFeaturesFinder() 
        {
            surf.hessianThreshold = 500.0;
            surf.extended = false;
        }

    protected:
        void find(const vector<Mat> &images, vector<ImageFeatures> &features);

    private:
        SURF_GPU surf;
    };

    void GpuSurfFeaturesFinder::find(const vector<Mat> &images, vector<ImageFeatures> &features)
    {
        // Make images gray
        vector<GpuMat> gray_images(images.size());
        for (size_t i = 0; i < images.size(); ++i)
        {
            CV_Assert(images[i].depth() == CV_8U);
            cvtColor(GpuMat(images[i]), gray_images[i], CV_BGR2GRAY);
        }

        features.resize(images.size());

        // Find keypoints in all images
        GpuMat d_keypoints;
        GpuMat d_descriptors;
        for (size_t i = 0; i < images.size(); ++i)
        {
            surf.nOctaves = 3;
            surf.nOctaveLayers = 4;
            surf(gray_images[i], GpuMat(), d_keypoints);

            surf.nOctaves = 4;
            surf.nOctaveLayers = 2;
            surf(gray_images[i], GpuMat(), d_keypoints, d_descriptors, true);

            surf.downloadKeypoints(d_keypoints, features[i].keypoints);
            d_descriptors.download(features[i].descriptors);
        }
    }
}

SurfFeaturesFinder::SurfFeaturesFinder(bool gpu_hint)
{
    if (gpu_hint && getCudaEnabledDeviceCount() > 0)
        impl_ = new GpuSurfFeaturesFinder;
    else
        impl_ = new CpuSurfFeaturesFinder;
}


void SurfFeaturesFinder::find(const vector<Mat> &images, vector<ImageFeatures> &features)
{
    (*impl_)(images, features);
}


//////////////////////////////////////////////////////////////////////////////

MatchesInfo::MatchesInfo() : src_img_idx(-1), dst_img_idx(-1), num_inliers(0) {}


MatchesInfo::MatchesInfo(const MatchesInfo &other)
{
    *this = other;
}


const MatchesInfo& MatchesInfo::operator =(const MatchesInfo &other)
{
    src_img_idx = other.src_img_idx;
    dst_img_idx = other.dst_img_idx;
    matches = other.matches;
    num_inliers = other.num_inliers;
    H = other.H.clone();
    return *this;
}


//////////////////////////////////////////////////////////////////////////////

void FeaturesMatcher::operator ()(const vector<Mat> &images, const vector<ImageFeatures> &features,
                                  vector<MatchesInfo> &pairwise_matches)
{
    pairwise_matches.resize(images.size() * images.size());
    for (size_t i = 0; i < images.size(); ++i)
    {
        LOGLN("Processing image " << i << "... ");
        for (size_t j = 0; j < images.size(); ++j)
        {
            if (i == j)
                continue;
            size_t pair_idx = i * images.size() + j;
            (*this)(images[i], features[i], images[j], features[j], pairwise_matches[pair_idx]);
            pairwise_matches[pair_idx].src_img_idx = i;
            pairwise_matches[pair_idx].dst_img_idx = j;
        }
    }
}


//////////////////////////////////////////////////////////////////////////////

namespace
{
    class CpuMatcher : public FeaturesMatcher
    {
    public:
        inline CpuMatcher(float match_conf) : match_conf_(match_conf) {}

        void match(const cv::Mat&, const ImageFeatures &features1, const cv::Mat&, const ImageFeatures &features2, MatchesInfo& matches_info);

    private:
        float match_conf_;
    };

    void CpuMatcher::match(const cv::Mat&, const ImageFeatures &features1, const cv::Mat&, const ImageFeatures &features2, MatchesInfo& matches_info)
    {
        matches_info.matches.clear();

        BruteForceMatcher< L2<float> > matcher;
        vector< vector<DMatch> > pair_matches;

        // Find 1->2 matches
        matcher.knnMatch(features1.descriptors, features2.descriptors, pair_matches, 2);
        for (size_t i = 0; i < pair_matches.size(); ++i)
        {
            if (pair_matches[i].size() < 2)
                continue;
            const DMatch& m0 = pair_matches[i][0];
            const DMatch& m1 = pair_matches[i][1];
            if (m0.distance < (1.f - match_conf_) * m1.distance)
                matches_info.matches.push_back(m0);
        }

        // Find 2->1 matches
        pair_matches.clear();
        matcher.knnMatch(features2.descriptors, features1.descriptors, pair_matches, 2);
        for (size_t i = 0; i < pair_matches.size(); ++i)
        {
            if (pair_matches[i].size() < 2)
                continue;
            const DMatch& m0 = pair_matches[i][0];
            const DMatch& m1 = pair_matches[i][1];
            if (m0.distance < (1.f - match_conf_) * m1.distance)
                matches_info.matches.push_back(DMatch(m0.trainIdx, m0.queryIdx, m0.distance));
        }
    }
        
    class GpuMatcher : public FeaturesMatcher
    {
    public:
        inline GpuMatcher(float match_conf) : match_conf_(match_conf) {}

        void match(const cv::Mat&, const ImageFeatures &features1, const cv::Mat&, const ImageFeatures &features2, MatchesInfo& matches_info);

    private:
        float match_conf_;

        GpuMat descriptors1_;
        GpuMat descriptors2_;

        GpuMat trainIdx_, distance_, allDist_;
    };

    void GpuMatcher::match(const cv::Mat&, const ImageFeatures &features1, const cv::Mat&, const ImageFeatures &features2, MatchesInfo& matches_info)
    {
        matches_info.matches.clear();

        BruteForceMatcher_GPU< L2<float> > matcher;
        
        descriptors1_.upload(features1.descriptors);
        descriptors2_.upload(features2.descriptors);

        vector< vector<DMatch> > pair_matches;

        // Find 1->2 matches
        matcher.knnMatch(descriptors1_, descriptors2_, trainIdx_, distance_, allDist_, 2);
        matcher.knnMatchDownload(trainIdx_, distance_, pair_matches);
        for (size_t i = 0; i < pair_matches.size(); ++i)
        {
            if (pair_matches[i].size() < 2)
                continue;
            const DMatch& m0 = pair_matches[i][0];
            const DMatch& m1 = pair_matches[i][1];

            CV_Assert(m0.queryIdx < static_cast<int>(features1.keypoints.size()));
            CV_Assert(m0.trainIdx < static_cast<int>(features2.keypoints.size()));

            if (m0.distance < (1.f - match_conf_) * m1.distance)
                matches_info.matches.push_back(m0);
        }

        // Find 2->1 matches
        pair_matches.clear();
        matcher.knnMatch(descriptors2_, descriptors1_, trainIdx_, distance_, allDist_, 2);
        matcher.knnMatchDownload(trainIdx_, distance_, pair_matches);
        for (size_t i = 0; i < pair_matches.size(); ++i)
        {
            if (pair_matches[i].size() < 2)
                continue;
            const DMatch& m0 = pair_matches[i][0];
            const DMatch& m1 = pair_matches[i][1];

            CV_Assert(m0.trainIdx < static_cast<int>(features1.keypoints.size()));
            CV_Assert(m0.queryIdx < static_cast<int>(features2.keypoints.size()));

            if (m0.distance < (1.f - match_conf_) * m1.distance)
                matches_info.matches.push_back(DMatch(m0.trainIdx, m0.queryIdx, m0.distance));
        }
    }
}

BestOf2NearestMatcher::BestOf2NearestMatcher(bool gpu_hint, float match_conf, int num_matches_thresh1, int num_matches_thresh2)
{
    if (gpu_hint && getCudaEnabledDeviceCount() > 0)
        impl_ = new GpuMatcher(match_conf);
    else
        impl_ = new CpuMatcher(match_conf);

    num_matches_thresh1_ = num_matches_thresh1;
    num_matches_thresh2_ = num_matches_thresh2;
}


void BestOf2NearestMatcher::match(const Mat &img1, const ImageFeatures &features1, const Mat &img2, const ImageFeatures &features2,
                                  MatchesInfo &matches_info)
{
    (*impl_)(img1, features1, img2, features2, matches_info);

    // Check if it makes sense to find homography
    if (matches_info.matches.size() < static_cast<size_t>(num_matches_thresh1_))
        return;

    // Construct point-point correspondences for homography estimation
    Mat src_points(1, matches_info.matches.size(), CV_32FC2);
    Mat dst_points(1, matches_info.matches.size(), CV_32FC2);
    for (size_t i = 0; i < matches_info.matches.size(); ++i)
    {
        const DMatch& m = matches_info.matches[i];

        Point2f p = features1.keypoints[m.queryIdx].pt;
        p.x -= img1.cols * 0.5f;
        p.y -= img1.rows * 0.5f;
        src_points.at<Point2f>(0, i) = p;

        p = features2.keypoints[m.trainIdx].pt;
        p.x -= img2.cols * 0.5f;
        p.y -= img2.rows * 0.5f;
        dst_points.at<Point2f>(0, i) = p;
    }

    // Find pair-wise motion
    vector<uchar> inlier_mask;
    matches_info.H = findHomography(src_points, dst_points, inlier_mask, CV_RANSAC);

    // Find number of inliers
    matches_info.num_inliers = 0;
    for (size_t i = 0; i < inlier_mask.size(); ++i)
        if (inlier_mask[i])
            matches_info.num_inliers++;

    // Check if we should try to refine motion
    if (matches_info.num_inliers < num_matches_thresh2_)
        return;

    // Construct point-point correspondences for inliers only
    src_points.create(1, matches_info.num_inliers, CV_32FC2);
    dst_points.create(1, matches_info.num_inliers, CV_32FC2);
    int inlier_idx = 0;
    for (size_t i = 0; i < matches_info.matches.size(); ++i)
    {
        if (!inlier_mask[i])
            continue;
        const DMatch& m = matches_info.matches[i];

        Point2f p = features1.keypoints[m.queryIdx].pt;
        p.x -= img1.cols * 0.5f;
        p.y -= img2.rows * 0.5f;
        src_points.at<Point2f>(0, inlier_idx) = p;

        p = features2.keypoints[m.trainIdx].pt;
        p.x -= img2.cols * 0.5f;
        p.y -= img2.rows * 0.5f;
        dst_points.at<Point2f>(0, inlier_idx) = p;

        inlier_idx++;
    }

    // Rerun motion estimation on inliers only
    matches_info.H = findHomography(src_points, dst_points, inlier_mask, CV_RANSAC);

    // Find number of inliers
    matches_info.num_inliers = 0;
    for (size_t i = 0; i < inlier_mask.size(); ++i)
        if (inlier_mask[i])
            matches_info.num_inliers++;
}


//////////////////////////////////////////////////////////////////////////////

CameraParams::CameraParams() : focal(1), M(Mat::eye(3, 3, CV_64F)), t(Mat::zeros(3, 1, CV_64F)) {}


CameraParams::CameraParams(const CameraParams &other)
{
    *this = other;
}


const CameraParams& CameraParams::operator =(const CameraParams &other)
{
    focal = other.focal;
    M = other.M.clone();
    t = other.t.clone();
    return *this;
}


//////////////////////////////////////////////////////////////////////////////

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
        cameras[edge.to].M = cameras[edge.from].M * R;
    }

    int num_images;
    const MatchesInfo* pairwise_matches;
    CameraParams* cameras;
};


void HomographyBasedEstimator::estimate(const vector<Mat> &images, const vector<ImageFeatures> &/*features*/,
                                        const vector<MatchesInfo> &pairwise_matches, vector<CameraParams> &cameras)
{
    const int num_images = static_cast<int>(images.size());

    // Find focals from pair-wise homographies
    vector<bool> is_focal_estimated(num_images, false);
    vector<double> focals;
    for (int i = 0; i < num_images; ++i)
    {
        for (int j = 0; j < num_images; ++j)
        {
            int pair_idx = i * num_images + j;
            if (pairwise_matches[pair_idx].H.empty())
                continue;
            double f_to, f_from;
            bool f_to_ok, f_from_ok;
            focalsFromHomography(pairwise_matches[pair_idx].H.inv(), f_to, f_from, f_to_ok, f_from_ok);
            if (f_from_ok)
                focals.push_back(f_from);
            if (f_to_ok)
                focals.push_back(f_to);
            if (f_from_ok && f_to_ok)
            {
                is_focal_estimated[i] = true;
                is_focal_estimated[j] = true;
            }
        }
    }
    is_focals_estimated_ = true;
    for (int i = 0; i < num_images; ++i)
        is_focals_estimated_ = is_focals_estimated_ && is_focal_estimated[i];

    // Find focal medians and use them as true focal length
    nth_element(focals.begin(), focals.end(), focals.begin() + focals.size() / 2);
    cameras.resize(num_images);
    for (int i = 0; i < num_images; ++i)
        cameras[i].focal = focals[focals.size() / 2];

    // Restore global motion
    Graph span_tree;
    vector<int> span_tree_centers;
    findMaxSpanningTree(num_images, pairwise_matches, span_tree, span_tree_centers);
    span_tree.walkBreadthFirst(span_tree_centers[0], CalcRotation(num_images, pairwise_matches, cameras));
}


//////////////////////////////////////////////////////////////////////////////

void BundleAdjuster::estimate(const vector<Mat> &images, const vector<ImageFeatures> &features,
                              const vector<MatchesInfo> &pairwise_matches, vector<CameraParams> &cameras)
{
    num_images_ = static_cast<int>(images.size());
    images_ = &images[0];
    features_ = &features[0];
    pairwise_matches_ = &pairwise_matches[0];

    // Prepare focals and rotations
    cameras_.create(num_images_ * 4, 1, CV_64F);
    SVD svd;
    for (int i = 0; i < num_images_; ++i)
    {
        cameras_.at<double>(i * 4, 0) = cameras[i].focal;
        svd(cameras[i].M, SVD::FULL_UV);
        Mat R = svd.u * svd.vt;
        if (determinant(R) < 0) R *= -1;
        Mat rvec;
        Rodrigues(R, rvec); CV_Assert(rvec.type() == CV_32F);
        cameras_.at<double>(i * 4 + 1, 0) = rvec.at<float>(0, 0);
        cameras_.at<double>(i * 4 + 2, 0) = rvec.at<float>(1, 0);
        cameras_.at<double>(i * 4 + 3, 0) = rvec.at<float>(2, 0);
    }

    edges_.clear();
    for (int i = 0; i < num_images_ - 1; ++i)
        for (int j = i + 1; j < num_images_; ++j)
            edges_.push_back(make_pair(i, j));

    total_num_matches_ = 0;
    for (size_t i = 0; i < edges_.size(); ++i)
        total_num_matches_ += static_cast<int>(pairwise_matches[edges_[i].first * num_images_ + edges_[i].second].matches.size());

    CvLevMarq solver(num_images_ * 4, total_num_matches_ * 3,
                     cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100, DBL_EPSILON));

    CvMat matParams = cameras_;
    cvCopy(&matParams, solver.param);

    int count = 0;
    for(;;)
    {
        const CvMat* _param = 0;
        CvMat* _J = 0;
        CvMat* _err = 0;

        bool proceed = solver.update( _param, _J, _err );

        cvCopy( _param, &matParams );

        if( !proceed || !_err )
            break;

        if( _J )
        {
            calcJacobian();
            CvMat matJ = J_;
            cvCopy( &matJ, _J );
        }

        if (_err)
        {
            calcError(err_);
            //LOGLN("Error: " << sqrt(err_.dot(err_)));
            count++;
            CvMat matErr = err_;
            cvCopy( &matErr, _err );
        }
    }
    LOGLN("BA final error: " << sqrt(err_.dot(err_)));
    LOGLN("BA iterations done: " << count);

    // Obtain global motion
    for (int i = 0; i < num_images_; ++i)
    {
        cameras[i].focal = cameras_.at<double>(i * 4, 0);
        Mat rvec(3, 1, CV_64F);
        rvec.at<double>(0, 0) = cameras_.at<double>(i * 4 + 1, 0);
        rvec.at<double>(1, 0) = cameras_.at<double>(i * 4 + 2, 0);
        rvec.at<double>(2, 0) = cameras_.at<double>(i * 4 + 3, 0);
        Rodrigues(rvec, cameras[i].M);
        Mat Mf;
        cameras[i].M.convertTo(Mf, CV_32F);
        cameras[i].M = Mf;
    }

    // Normalize motion to center image
    Graph span_tree;
    vector<int> span_tree_centers;
    findMaxSpanningTree(num_images_, pairwise_matches, span_tree, span_tree_centers);
    Mat R_inv = cameras[span_tree_centers[0]].M.inv();
    for (int i = 0; i < num_images_; ++i)
        cameras[i].M = R_inv * cameras[i].M;
}


void BundleAdjuster::calcError(Mat &err)
{
    err.create(total_num_matches_ * 3, 1, CV_64F);

    int match_idx = 0;
    for (size_t edge_idx = 0; edge_idx < edges_.size(); ++edge_idx)
    {
        int i = edges_[edge_idx].first;
        int j = edges_[edge_idx].second;
        double f1 = cameras_.at<double>(i * 4, 0);
        double f2 = cameras_.at<double>(j * 4, 0);
        double R1[9], R2[9];
        Mat R1_(3, 3, CV_64F, R1), R2_(3, 3, CV_64F, R2);
        Mat rvec(3, 1, CV_64F);
        rvec.at<double>(0, 0) = cameras_.at<double>(i * 4 + 1, 0);
        rvec.at<double>(1, 0) = cameras_.at<double>(i * 4 + 2, 0);
        rvec.at<double>(2, 0) = cameras_.at<double>(i * 4 + 3, 0);
        Rodrigues(rvec, R1_); CV_Assert(R1_.type() == CV_64F);
        rvec.at<double>(0, 0) = cameras_.at<double>(j * 4 + 1, 0);
        rvec.at<double>(1, 0) = cameras_.at<double>(j * 4 + 2, 0);
        rvec.at<double>(2, 0) = cameras_.at<double>(j * 4 + 3, 0);
        Rodrigues(rvec, R2_); CV_Assert(R2_.type() == CV_64F);

        const ImageFeatures& features1 = features_[i];
        const ImageFeatures& features2 = features_[j];
        const MatchesInfo& matches_info = pairwise_matches_[i * num_images_ + j];

        for (size_t k = 0; k < matches_info.matches.size(); ++k)
        {
            const DMatch& m = matches_info.matches[k];

            Point2d kp1 = features1.keypoints[m.queryIdx].pt;
            kp1.x -= 0.5 * images_[i].cols;
            kp1.y -= 0.5 * images_[i].rows;
            Point2d kp2 = features2.keypoints[m.trainIdx].pt;
            kp2.x -= 0.5 * images_[j].cols;
            kp2.y -= 0.5 * images_[j].rows;
            double len1 = sqrt(kp1.x * kp1.x + kp1.y * kp1.y + f1 * f1);
            double len2 = sqrt(kp2.x * kp2.x + kp2.y * kp2.y + f2 * f2);
            Point3d p1(kp1.x / len1, kp1.y / len1, f1 / len1);
            Point3d p2(kp2.x / len2, kp2.y / len2, f2 / len2);

            Point3d d1(p1.x * R1[0] + p1.y * R1[1] + p1.z * R1[2],
                       p1.x * R1[3] + p1.y * R1[4] + p1.z * R1[5],
                       p1.x * R1[6] + p1.y * R1[7] + p1.z * R1[8]);
            Point3d d2(p2.x * R2[0] + p2.y * R2[1] + p2.z * R2[2],
                       p2.x * R2[3] + p2.y * R2[4] + p2.z * R2[5],
                       p2.x * R2[6] + p2.y * R2[7] + p2.z * R2[8]);

            double mult = 1;
            if (cost_space_ == FOCAL_RAY_SPACE)
                mult = sqrt(f1 * f2);
            err.at<double>(3 * match_idx, 0) = mult * (d1.x - d2.x);
            err.at<double>(3 * match_idx + 1, 0) = mult * (d1.y - d2.y);
            err.at<double>(3 * match_idx + 2, 0) = mult * (d1.z - d2.z);
            match_idx++;
        }
    }
}


void calcDeriv(const Mat &err1, const Mat &err2, double h, Mat res)
{
    for (int i = 0; i < err1.rows; ++i)
        res.at<double>(i, 0) = (err2.at<double>(i, 0) - err1.at<double>(i, 0)) / h;
}


void BundleAdjuster::calcJacobian()
{
    J_.create(total_num_matches_ * 3, num_images_ * 4, CV_64F);

    double f, r;
    const double df = 0.001; // Focal length step
    const double dr = 0.001; // Angle step

    for (int i = 0; i < num_images_; ++i)
    {
        f = cameras_.at<double>(i * 4, 0);
        cameras_.at<double>(i * 4, 0) = f - df;
        calcError(err1_);
        cameras_.at<double>(i * 4, 0) = f + df;
        calcError(err2_);
        calcDeriv(err1_, err2_, 2 * df, J_.col(i * 4));
        cameras_.at<double>(i * 4, 0) = f;

        r = cameras_.at<double>(i * 4 + 1, 0);
        cameras_.at<double>(i * 4 + 1, 0) = r - dr;
        calcError(err1_);
        cameras_.at<double>(i * 4 + 1, 0) = r + dr;
        calcError(err2_);
        calcDeriv(err1_, err2_, 2 * dr, J_.col(i * 4 + 1));
        cameras_.at<double>(i * 4 + 1, 0) = r;

        r = cameras_.at<double>(i * 4 + 2, 0);
        cameras_.at<double>(i * 4 + 2, 0) = r - dr;
        calcError(err1_);
        cameras_.at<double>(i * 4 + 2, 0) = r + dr;
        calcError(err2_);
        calcDeriv(err1_, err2_, 2 * dr, J_.col(i * 4 + 2));
        cameras_.at<double>(i * 4 + 2, 0) = r;

        r = cameras_.at<double>(i * 4 + 3, 0);
        cameras_.at<double>(i * 4 + 3, 0) = r - dr;
        calcError(err1_);
        cameras_.at<double>(i * 4 + 3, 0) = r + dr;
        calcError(err2_);
        calcDeriv(err1_, err2_, 2 * dr, J_.col(i * 4 + 3));
        cameras_.at<double>(i * 4 + 3, 0) = r;
    }
}


//////////////////////////////////////////////////////////////////////////////

// TODO test on adobe/halfdome
void waveCorrect(vector<Mat> &rmats)
{
    float data[9];
    Mat r0(1, 3, CV_32F, data);
    Mat r1(1, 3, CV_32F, data + 3);
    Mat r2(1, 3, CV_32F, data + 6);
    Mat R(3, 3, CV_32F, data);

    Mat cov = Mat::zeros(3, 3, CV_32F);
    for (size_t i = 0; i < rmats.size(); ++i)
    {   
        Mat r0 = rmats[i].col(0);
        cov += r0 * r0.t();
    }

    SVD svd;
    svd(cov, SVD::FULL_UV);
    svd.vt.row(2).copyTo(r1);
    if (determinant(svd.vt) < 0)
        r1 *= -1;

    Mat avgz = Mat::zeros(3, 1, CV_32F);
    for (size_t i = 0; i < rmats.size(); ++i)
        avgz += rmats[i].col(2);
    avgz.t().cross(r1).copyTo(r0);
    normalize(r0, r0);

    r0.cross(r1).copyTo(r2);

    for (size_t i = 0; i < rmats.size(); ++i)
        rmats[i] = R * rmats[i];
}


//////////////////////////////////////////////////////////////////////////////

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
            float conf = static_cast<float>(pairwise_matches[i * num_images + j].num_inliers);
            graph.addEdge(i, j, conf);
            edges.push_back(GraphEdge(i, j, conf));
        }
    }

    DjSets comps(num_images);
    span_tree.create(num_images);
    vector<int> span_tree_powers(num_images, 0);

    // Find maximum spanning tree
    sort(edges.begin(), edges.end(), greater<GraphEdge>());
    for (size_t i = 0; i < edges.size(); ++i)
    {
        int comp1 = comps.find(edges[i].from);
        int comp2 = comps.find(edges[i].to);
        if (comp1 != comp2)
        {
            comps.merge(comp1, comp2);
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
