#ifndef __OPENCV_BLENDERS_HPP__
#define __OPENCV_BLENDERS_HPP__

#include <vector>
#include <opencv2/core/core.hpp>

// Simple blender which puts one image over another
class Blender
{
public:
    enum { NO, FEATHER, MULTI_BAND };
    static cv::Ptr<Blender> createDefault(int type);

    void prepare(const std::vector<cv::Point> &corners, const std::vector<cv::Size> &sizes);
    virtual void prepare(cv::Rect dst_roi);
    virtual void feed(const cv::Mat &img, const cv::Mat &mask, cv::Point tl);
    virtual void blend(cv::Mat &dst, cv::Mat &dst_mask);

protected:
    cv::Mat dst_, dst_mask_;
    cv::Rect dst_roi_;
};


class FeatherBlender : public Blender
{
public:
    FeatherBlender(float sharpness = 0.02f) { setSharpness(sharpness); }
    float sharpness() const { return sharpness_; }
    void setSharpness(float val) { sharpness_ = val; }

    void prepare(cv::Rect dst_roi);
    void feed(const cv::Mat &img, const cv::Mat &mask, cv::Point tl);
    void blend(cv::Mat &dst, cv::Mat &dst_mask);

private:
    float sharpness_;
    cv::Mat weight_map_;
    cv::Mat dst_weight_map_;
};


class MultiBandBlender : public Blender
{
public:
    MultiBandBlender(int num_bands = 7) { setNumBands(num_bands); }
    int numBands() const { return num_bands_; }
    void setNumBands(int val) { num_bands_ = val; }

    void prepare(cv::Rect dst_roi);
    void feed(const cv::Mat &img, const cv::Mat &mask, cv::Point tl);
    void blend(cv::Mat &dst, cv::Mat &dst_mask);

private:
    int num_bands_;
    std::vector<cv::Mat> dst_pyr_laplace_;
    std::vector<cv::Mat> dst_band_weights_;
};


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

cv::Rect resultRoi(const std::vector<cv::Point> &corners, const std::vector<cv::Size> &sizes);

void normalize(const cv::Mat& weight, cv::Mat& src);

void createWeightMap(const cv::Mat& mask, float sharpness, cv::Mat& weight);

void createLaplacePyr(const std::vector<cv::Mat>& pyr_gauss, std::vector<cv::Mat>& pyr_laplace);

// Restores source image in-place. Result will be stored in pyr[0].
void restoreImageFromLaplacePyr(std::vector<cv::Mat>& pyr);

#endif // __OPENCV_BLENDERS_HPP__
