#ifndef P3P_H
#define P3P_H

#include "precomp.hpp"

class p3p {
public:
    p3p();
    int estimate(std::vector<cv::Mat>& Rs, std::vector<cv::Mat>& ts, const cv::Mat& opoints, const cv::Mat& ipoints);

private:
    void calibrateAndNormalizePointsPnP(const cv::Mat& opoints, const cv::Mat& ipoints);

    // 3D object points
    cv::Matx33d points_mat;
    /*
     * calibrated normalized points
     * K^-1 [u v 1]^T / ||K^-1 [u v 1]^T||
     */
    cv::Matx33d calib_norm_points_mat;
};

#endif // P3P_H
