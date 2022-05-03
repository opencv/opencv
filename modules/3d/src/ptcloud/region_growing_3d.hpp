// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022, Wanli Zhong <zhongwl2018@mail.sustech.edu.cn>

#ifndef OPENCV_REGION_ROWING_3D_HPP
#define OPENCV_REGION_ROWING_3D_HPP

#include "opencv2/3d/ptcloud.hpp"
#include "ptcloud_utils.hpp"

namespace cv {

class RegionGrowing3DImpl : public RegionGrowing3D
{
private:
    //! Threshold of the angle between normals, the value is in radian.
    double smoothness_thr;

    //! Threshold of curvature.
    double curvature_thr;

    //! The number of neighbors including itself.
    int k;

    //! The minimum size of region.
    int min_size;

    //! The maximum size of region
    int max_size;

    //! Whether to use the smoothness mode.
    bool smooth_mode;

    //! The maximum number of regions you want.
    int region_num;

    //! The curvature of each point.
    Mat curvatures;

    //! Seed points.
    Mat seeds;

public:
    //! No-argument constructor using default configuration.
    RegionGrowing3DImpl()
            : smoothness_thr(0.5235987756 /* 30*PI/180 */ ), curvature_thr(0.05), k(0), min_size(1),
              max_size(INT_MAX), smooth_mode(true), region_num(INT_MAX)
    {
    }

    int segment(OutputArray labels, InputArray input_pts, InputArray normals, InputArray knn_idx) override;

    //-------------------------- Getter and Setter -----------------------

    void setMinSize(int min_size_) override
    {
        CV_CheckGE(min_size_, 1, "The minimum size of segments should be grater than 0.");
        min_size = min_size_;
    }

    int getMinSize() const override
    {
        return min_size;
    }

    void setMaxSize(int max_size_) override
    {
        CV_CheckGE(max_size_, 1, "The maximum size of segments should be grater than 0.");
        max_size = max_size_;
    }

    int getMaxSize() const override
    {
        return max_size;
    }

    void setSmoothModeFlag(bool smooth_mode_) override
    {
        smooth_mode = smooth_mode_;
    }

    bool getSmoothModeFlag() const override
    {
        return smooth_mode;
    }

    void setSmoothnessThreshold(double smoothness_thr_) override
    {
        CV_CheckGE(smoothness_thr_, 0.0, "The smoothness threshold angle should be greater than or equal to 0 degrees.");
        CV_CheckLT(smoothness_thr_, 1.5707963268 /* 90*PI/180 */, "The smoothness threshold angle should be less than 90 degrees.");
        smoothness_thr = smoothness_thr_;
    }

    double getSmoothnessThreshold() const override
    {
        return smoothness_thr;
    }

    void setCurvatureThreshold(double curvature_thr_) override
    {
        CV_CheckGE(curvature_thr_, 0.0, "The curvature threshold should be greater than or equal to 0.");
        curvature_thr = curvature_thr_;
    }

    double getCurvatureThreshold() const override
    {
        return curvature_thr;
    }

    void setNumberOfNeighbors(int k_) override
    {
        CV_CheckGE(k_, 2, "The number of neighbors should be grater than 1.");
        k = k_;
    }

    int getNumberOfNeighbors() const override
    {
        return k;
    }

    void setNumberOfRegions(int region_num_) override
    {
        CV_CheckGE(region_num_, 1, "The number of region should be grater than 0.");
        region_num = region_num_;
    }

    int getNumberOfRegions() const override
    {
        return region_num;
    }

    void setSeeds(InputArray seeds_) override
    {
        seeds = seeds_.getMat();
    }

    void getSeeds(OutputArray seeds_) const override
    {
        seeds.copyTo(seeds_);
    }

    void setCurvatures(InputArray curvatures_) override
    {
        curvatures = abs(curvatures_.getMat());
    }

    void getCurvatures(OutputArray curvatures_) const override
    {
        curvatures.copyTo(curvatures_);
    }

};

Ptr<RegionGrowing3D> RegionGrowing3D::create()
{
    return makePtr<RegionGrowing3DImpl>();
}

} //end namespace cv

#endif //OPENCV_REGION_ROWING_3D_HPP
