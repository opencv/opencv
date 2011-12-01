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

#ifndef __OPENCV_STITCHING_MOTION_ESTIMATORS_HPP__
#define __OPENCV_STITCHING_MOTION_ESTIMATORS_HPP__

#include "opencv2/core/core.hpp"
#include "matchers.hpp"
#include "util.hpp"
#include "camera.hpp"

namespace cv {
namespace detail {

class CV_EXPORTS Estimator
{
public:
    virtual ~Estimator() {}

    void operator ()(const std::vector<ImageFeatures> &features, const std::vector<MatchesInfo> &pairwise_matches, 
                     std::vector<CameraParams> &cameras)
        { estimate(features, pairwise_matches, cameras); }

protected:
    virtual void estimate(const std::vector<ImageFeatures> &features, const std::vector<MatchesInfo> &pairwise_matches, 
                          std::vector<CameraParams> &cameras) = 0;
};


class CV_EXPORTS HomographyBasedEstimator : public Estimator
{
public:
    HomographyBasedEstimator() : is_focals_estimated_(false) {}
    bool isFocalsEstimated() const { return is_focals_estimated_; }

private:   
    void estimate(const std::vector<ImageFeatures> &features, const std::vector<MatchesInfo> &pairwise_matches, 
                  std::vector<CameraParams> &cameras);

    bool is_focals_estimated_;
};


class CV_EXPORTS BundleAdjusterBase : public Estimator
{
public:
    const Mat refinementMask() const { return refinement_mask_.clone(); }
    void setRefinementMask(const Mat &mask) 
    { 
        CV_Assert(mask.type() == CV_8U && mask.size() == Size(3, 3));
        refinement_mask_ = mask.clone(); 
    }

    double confThresh() const { return conf_thresh_; }
    void setConfThresh(double conf_thresh) { conf_thresh_ = conf_thresh; }

    CvTermCriteria termCriteria() { return term_criteria_; }
    void setTermCriteria(const CvTermCriteria& term_criteria) { term_criteria_ = term_criteria; }

protected:
    BundleAdjusterBase(int num_params_per_cam, int num_errs_per_measurement) 
        : num_params_per_cam_(num_params_per_cam), 
          num_errs_per_measurement_(num_errs_per_measurement) 
    {    
        setRefinementMask(Mat::ones(3, 3, CV_8U));
        setConfThresh(1.); 
        setTermCriteria(cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 1000, DBL_EPSILON));
    }

    // Runs bundle adjustment
    virtual void estimate(const std::vector<ImageFeatures> &features, 
                          const std::vector<MatchesInfo> &pairwise_matches,
                          std::vector<CameraParams> &cameras);

    virtual void setUpInitialCameraParams(const std::vector<CameraParams> &cameras) = 0;
    virtual void obtainRefinedCameraParams(std::vector<CameraParams> &cameras) const = 0;
    virtual void calcError(Mat &err) = 0;
    virtual void calcJacobian(Mat &jac) = 0;

    // 3x3 8U mask, where 0 means don't refine respective parameter, != 0 means refine
    Mat refinement_mask_;

    int num_images_;
    int total_num_matches_;

    int num_params_per_cam_;
    int num_errs_per_measurement_;

    const ImageFeatures *features_;
    const MatchesInfo *pairwise_matches_;

    // Threshold to filter out poorly matched image pairs
    double conf_thresh_;

    //Levenberg–Marquardt algorithm termination criteria
    CvTermCriteria term_criteria_;

    // Camera parameters matrix (CV_64F)
    Mat cam_params_;

    // Connected images pairs
    std::vector<std::pair<int,int> > edges_;
};


// Minimizes reprojection error.
// It can estimate focal length, aspect ratio, principal point. 
// You can affect only on them via the refinement mask.
class CV_EXPORTS BundleAdjusterReproj : public BundleAdjusterBase
{
public:
    BundleAdjusterReproj() : BundleAdjusterBase(7, 2) {}

private:
    void setUpInitialCameraParams(const std::vector<CameraParams> &cameras);
    void obtainRefinedCameraParams(std::vector<CameraParams> &cameras) const;
    void calcError(Mat &err);
    void calcJacobian(Mat &jac);

    Mat err1_, err2_;
};


// Minimizes sun of ray-to-ray distances.
// It can estimate focal length. It ignores the refinement mask for now.
class CV_EXPORTS BundleAdjusterRay : public BundleAdjusterBase
{
public:
    BundleAdjusterRay() : BundleAdjusterBase(4, 3) {}

private:
    void setUpInitialCameraParams(const std::vector<CameraParams> &cameras);
    void obtainRefinedCameraParams(std::vector<CameraParams> &cameras) const;
    void calcError(Mat &err);
    void calcJacobian(Mat &jac);

    Mat err1_, err2_;
};


enum CV_EXPORTS WaveCorrectKind
{
    WAVE_CORRECT_HORIZ,
    WAVE_CORRECT_VERT
};

void CV_EXPORTS waveCorrect(std::vector<Mat> &rmats, WaveCorrectKind kind);


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

// Returns matches graph representation in DOT language
std::string CV_EXPORTS matchesGraphAsString(std::vector<std::string> &pathes, std::vector<MatchesInfo> &pairwise_matches,
                                            float conf_threshold);

std::vector<int> CV_EXPORTS leaveBiggestComponent(std::vector<ImageFeatures> &features, std::vector<MatchesInfo> &pairwise_matches, 
                                                  float conf_threshold);

void CV_EXPORTS findMaxSpanningTree(int num_images, const std::vector<MatchesInfo> &pairwise_matches, 
                                    Graph &span_tree, std::vector<int> &centers);

} // namespace detail
} // namespace cv

#endif // __OPENCV_STITCHING_MOTION_ESTIMATORS_HPP__
