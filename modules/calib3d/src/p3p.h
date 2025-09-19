#ifndef P3P_H
#define P3P_H

#include "precomp.hpp"

class p3p {
public:
    p3p();
    int estimate(std::vector<cv::Mat>& Rs, std::vector<cv::Mat>& ts, const cv::Mat& opoints, const cv::Mat& ipoints);

private:
    void calibrateAndNormalizePointsPnP(const cv::Mat& opoints, const cv::Mat& ipoints);

    std::array<cv::Vec3d, 3> x_copy;
    std::array<cv::Vec3d, 3> X_copy;
};

#endif // P3P_H
