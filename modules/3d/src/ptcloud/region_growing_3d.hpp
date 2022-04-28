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
    int min_size;

    int max_size;

    bool smooth_mode;

    float smoothness_thr;

    float curvature_thr;

    int k;

    int region_num;

    Mat input_pts;

    Mat knn_idx;

    Mat seeds;

    Mat normals;

    Mat curvatures;

public:

    RegionGrowing3DImpl(float smoothness_thr_, float curvature_thr_)
            : smoothness_thr(smoothness_thr_), curvature_thr(curvature_thr_), k(0), min_size(1),
              max_size(INT_MAX), smooth_mode(true), region_num(INT_MAX)
    {

    }

    ~RegionGrowing3DImpl() override = default;

    int segment(OutputArray labels) override;

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
    void setSmoothnessThreshold(float smoothness_thr_) override{
        CV_CheckGE(smoothness_thr_, 0.f, "The smoothness threshold angle should be greater than or equal to 0 degrees.");
        CV_CheckLT(smoothness_thr_, 3.1415927f / 2, "The smoothness threshold angle should be less than 90 degrees.");
        smoothness_thr = smoothness_thr_;
    }

    //! Get
    float getSmoothnessThreshold() const override{
        return smoothness_thr;
    }

    //! Set
    void setCurvatureThreshold(float curvature_thr_) override{
        CV_CheckGE(curvature_thr_, 0.f, "The curvature threshold should be greater than or equal to 0.");
        curvature_thr = curvature_thr_;
    }

    //! Get
    float getCurvatureThreshold() const override{
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
    void setPtcloud(InputArray input_pts_) override{
        getPointsMatFromInputArray(input_pts_, input_pts, 0);
    }

    //! Set
    void setKnnIdx(InputArray knn_idx_) override{
        if(knn_idx_.rows() < knn_idx_.cols()){
            transpose(knn_idx_.getMat(), knn_idx);
        }else{
            knn_idx = knn_idx_.getMat();
        }
    }

    //! Set
    void setSeeds(InputArray seeds_) override{
        seeds = seeds_.getMat();
    }

    //! Set
    void setNormals(InputArray normals_) override{
        getPointsMatFromInputArray(normals_, normals, 0);
    }

    //! Set
    void setCurvatures(InputArray curvatures_) override{
        curvatures = curvatures_.getMat();
    }

};

Ptr<RegionGrowing3D> RegionGrowing3D::create(float smoothness_thr_, float curvature_thr_)
{
    return makePtr<RegionGrowing3DImpl>(smoothness_thr_, curvature_thr_);
}

//! @} _3d
} //end namespace cv


#endif //OPENCV_REGION_ROWING_3D_HPP
