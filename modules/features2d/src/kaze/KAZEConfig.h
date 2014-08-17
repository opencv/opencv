/**
 * @file KAZEConfig.h
 * @brief Configuration file
 * @date Dec 27, 2011
 * @author Pablo F. Alcantarilla
 */

#ifndef __OPENCV_FEATURES_2D_AKAZE_CONFIG_H__
#define __OPENCV_FEATURES_2D_AKAZE_CONFIG_H__

// OpenCV Includes
#include "../precomp.hpp"
#include <opencv2/features2d.hpp>

//*************************************************************************************

struct KAZEOptions {

    KAZEOptions()
        : diffusivity(cv::DIFF_PM_G2)

        , soffset(1.60f)
        , omax(4)
        , nsublevels(4)
        , img_width(0)
        , img_height(0)
        , sderivatives(1.0f)
        , dthreshold(0.001f)
        , kcontrast(0.01f)
        , kcontrast_percentille(0.7f)
                , kcontrast_bins(300)
        , upright(false)
        , extended(false)
    {
    }

    int diffusivity;
    float soffset;
    int omax;
    int nsublevels;
    int img_width;
    int img_height;
    float sderivatives;
    float dthreshold;
    float kcontrast;
    float kcontrast_percentille;
    int  kcontrast_bins;
    bool upright;
    bool extended;
};

#endif
