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

//! @addtogroup _3d
//! @{

class RegionGrowing3DImpl : public RegionGrowing3D
{
private:
    //! The
    double smoothness_thr;

    double curvature_thr;

    int k;

    int min_size;

    int max_size;

    bool smooth_mode;

    int region_num;

    Mat curvatures;

    Mat seeds;

public:

    RegionGrowing3DImpl()
            : smoothness_thr(0.5235987756 /* 30*π/180 */ ), curvature_thr(0.05), k(0), min_size(1),
              max_size(INT_MAX), smooth_mode(true), region_num(INT_MAX)
    {

    }

    ~RegionGrowing3DImpl() override = default;

    int segment(OutputArray labels, InputArray input_pts, InputArray normals, InputArray knn_idx) override;

    //-------------------------- Getter and Setter -----------------------
    //! Set
    void setMinSize(int min_size_) override{
        CV_CheckGE(min_size_, 1, "The minimum size of segments should be grater than 0.");
        min_size = min_size_;
    }

    //! Get
    int getMinSize() const override{
        return min_size;
    }

    //! Set
    void setMaxSize(int max_size_) override{
        CV_CheckGE(max_size_, 1, "The maximum size of segments should be grater than 0.");
        max_size = max_size_;
    }

    //! Get
    int getMaxSize() const override{
        return max_size;
    }

    //! Set
    void setSmoothModeFlag(bool smooth_mode_) override{
        smooth_mode = smooth_mode_;
    }

    //! Get
    bool getSmoothModeFlag() const override{
        return smooth_mode;
    }

    //! Set
    void setSmoothnessThreshold(double smoothness_thr_) override{
        CV_CheckGE(smoothness_thr_, 0.0, "The smoothness threshold angle should be greater than or equal to 0 degrees.");
        CV_CheckLT(smoothness_thr_, 1.5707963268 /* 90*π/180 */, "The smoothness threshold angle should be less than 90 degrees.");
        smoothness_thr = smoothness_thr_;
    }

    //! Get
    double getSmoothnessThreshold() const override{
        return smoothness_thr;
    }

    //! Set
    void setCurvatureThreshold(double curvature_thr_) override{
        CV_CheckGE(curvature_thr_, 0.0, "The curvature threshold should be greater than or equal to 0.");
        curvature_thr = curvature_thr_;
    }

    //! Get
    double getCurvatureThreshold() const override{
        return curvature_thr;
    }

    //! Set
    void setNumberOfNeighbors(int k_) override{
        CV_CheckGE(k_, 2, "The number of neighbors should be grater than 1.");
        k = k_;
    }

    //! Get
    int getNumberOfNeighbors() const override{
        return k;
    }

    //! Set
    void setNumberOfRegions(int region_num_) override{
        CV_CheckGE(region_num_, 1, "The number of region should be grater than 0.");
        region_num = region_num_;
    }

    //! Get
    int getNumberOfRegions() const override{
        return region_num;
    }

    //! Set
    void setSeeds(InputArray seeds_) override{
        seeds = seeds_.getMat();
    }

    //! Get
    void getSeeds(OutputArray seeds_) const override{
        seeds.copyTo(seeds_);
    }

    //! Set
    void setCurvatures(InputArray curvatures_) override{
        curvatures = abs(curvatures_.getMat());
    }

    //! Get
    void getCurvatures(OutputArray curvatures_) const override{
        curvatures.copyTo(curvatures_);
    }

};

Ptr<RegionGrowing3D> RegionGrowing3D::create()
{
    return makePtr<RegionGrowing3DImpl>();
}

//! @} _3d
} //end namespace cv


#endif //OPENCV_REGION_ROWING_3D_HPP
