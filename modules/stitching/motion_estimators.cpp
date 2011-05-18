#include <algorithm>
#include "opencv2/core/core_c.h"
#include <opencv2/calib3d/calib3d.hpp>
#include "autocalib.hpp"
#include "motion_estimators.hpp"
#include "util.hpp"

using namespace std;
using namespace cv;


//////////////////////////////////////////////////////////////////////////////

CameraParams::CameraParams() : focal(1), R(Mat::eye(3, 3, CV_64F)), t(Mat::zeros(3, 1, CV_64F)) {}

CameraParams::CameraParams(const CameraParams &other) { *this = other; }

const CameraParams& CameraParams::operator =(const CameraParams &other)
{
    focal = other.focal;
    R = other.R.clone();
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
        cameras[edge.to].R = cameras[edge.from].R * R;
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

            if (f_from_ok) focals.push_back(f_from);
            if (f_to_ok) focals.push_back(f_to);

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

    // Find focal median and use it as true focal length
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

        svd(cameras[i].R, SVD::FULL_UV);
        Mat R = svd.u * svd.vt;
        if (determinant(R) < 0) 
            R *= -1;

        Mat rvec;
        Rodrigues(R, rvec); CV_Assert(rvec.type() == CV_32F);
        cameras_.at<double>(i * 4 + 1, 0) = rvec.at<float>(0, 0);
        cameras_.at<double>(i * 4 + 2, 0) = rvec.at<float>(1, 0);
        cameras_.at<double>(i * 4 + 3, 0) = rvec.at<float>(2, 0);
    }

    // Select only consistent image pairs for futher adjustment
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
        total_num_matches_ += static_cast<int>(pairwise_matches[edges_[i].first * num_images_ + edges_[i].second].num_inliers);

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
            LOGLN("Error: " << sqrt(err_.dot(err_)));
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
        Rodrigues(rvec, cameras[i].R);
        Mat Mf;
        cameras[i].R.convertTo(Mf, CV_32F);
        cameras[i].R = Mf;
    }

    // Normalize motion to center image
    Graph span_tree;
    vector<int> span_tree_centers;
    findMaxSpanningTree(num_images_, pairwise_matches, span_tree, span_tree_centers);
    Mat R_inv = cameras[span_tree_centers[0]].R.inv();
    for (int i = 0; i < num_images_; ++i)
        cameras[i].R = R_inv * cameras[i].R;
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
            if (!matches_info.inliers_mask[k])
                continue;

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
    r1.cross(avgz.t()).copyTo(r0);
    normalize(r0, r0);

    r1.cross(r0).copyTo(r2);
    if (determinant(R) < 0)
        R *= -1;

    for (size_t i = 0; i < rmats.size(); ++i)
        rmats[i] = R * rmats[i];
}


//////////////////////////////////////////////////////////////////////////////

void leaveBiggestComponent(vector<Mat> &images, vector<ImageFeatures> &features, 
                           vector<MatchesInfo> &pairwise_matches, float conf_threshold)
{
    const int num_images = static_cast<int>(images.size());

    DjSets comps(num_images);
    for (int i = 0; i < num_images; ++i)
    {
        for (int j = 0; j < num_images; ++j)
        {
            if (pairwise_matches[i*num_images + j].confidence < conf_threshold)
                continue;
            int comp1 = comps.find(i);
            int comp2 = comps.find(j);
            if (comp1 != comp2) 
                comps.merge(comp1, comp2);
        }
    }

    int max_comp = max_element(comps.size.begin(), comps.size.end()) - comps.size.begin();

    vector<int> indices;
    vector<int> indices_removed;
    for (int i = 0; i < num_images; ++i)
        if (comps.find(i) == max_comp)
            indices.push_back(i);    
        else
            indices_removed.push_back(i);

    vector<Mat> images_subset;
    vector<ImageFeatures> features_subset;
    vector<MatchesInfo> pairwise_matches_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        images_subset.push_back(images[indices[i]]);
        features_subset.push_back(features[indices[i]]);
        for (size_t j = 0; j < indices.size(); ++j)
        {
            pairwise_matches_subset.push_back(pairwise_matches[indices[i]*num_images + indices[j]]);
            pairwise_matches_subset.back().src_img_idx = i;
            pairwise_matches_subset.back().dst_img_idx = j;
        }
    }

    if (static_cast<int>(images_subset.size()) == num_images)
        return;

    LOG("Removed some images, because can't match them: (");
    LOG(indices_removed[0]);
    for (size_t i = 1; i < indices_removed.size(); ++i) LOG(", " << indices_removed[i]);
    LOGLN(")");

    images = images_subset;
    features = features_subset;
    pairwise_matches = pairwise_matches_subset;
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
