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

    cv::Point operator ()(const std::vector<cv::Mat> &src, const std::vector<cv::Point> &corners, const std::vector<cv::Mat> &masks,
                          cv::Mat& dst);
    cv::Point operator ()(const std::vector<cv::Mat> &src, const std::vector<cv::Point> &corners, const std::vector<cv::Mat> &masks,
                          cv::Mat& dst, cv::Mat& dst_mask);

protected:
    virtual cv::Point blend(const std::vector<cv::Mat> &src, const std::vector<cv::Point> &corners, const std::vector<cv::Mat> &masks,
                            cv::Mat& dst, cv::Mat& dst_mask);
};


class FeatherBlender : public Blender
{
public:
    FeatherBlender(float sharpness = 0.02f) : sharpness_(sharpness) {}

private:
    cv::Point blend(const std::vector<cv::Mat> &src, const std::vector<cv::Point> &corners, const std::vector<cv::Mat> &masks,
                    cv::Mat &dst, cv::Mat &dst_mask);

    float sharpness_;
};


class MultiBandBlender : public Blender
{
public:
    MultiBandBlender(int num_bands = 7) : num_bands_(num_bands) {}

private:
    cv::Point blend(const std::vector<cv::Mat> &src, const std::vector<cv::Point> &corners, const std::vector<cv::Mat> &masks,
                    cv::Mat& dst, cv::Mat& dst_mask);

    int num_bands_;
};


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

cv::Rect resultRoi(const std::vector<cv::Mat> &src, const std::vector<cv::Point> &corners);

cv::Point computeResultMask(const std::vector<cv::Mat> &masks, const std::vector<cv::Point> &corners, cv::Mat &mask);

cv::Point blendLinear(const std::vector<cv::Mat> &src, const std::vector<cv::Point> &corners, const std::vector<cv::Mat> &weights,
                      cv::Mat& dst, cv::Mat& dst_weight);

void normalize(const cv::Mat& weight, cv::Mat& src);

void createWeightMap(const cv::Mat& mask, float sharpness, cv::Mat& weight);

void createLaplacePyr(const std::vector<cv::Mat>& pyr_gauss, std::vector<cv::Mat>& pyr_laplace);

// Restores source image in-place. Result will be stored in pyr[0].
void restoreImageFromLaplacePyr(std::vector<cv::Mat>& pyr);

#endif // __OPENCV_BLENDERS_HPP__
