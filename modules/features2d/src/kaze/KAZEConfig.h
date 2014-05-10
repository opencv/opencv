/**
 * @file KAZEConfig.h
 * @brief Configuration file
 * @date Dec 27, 2011
 * @author Pablo F. Alcantarilla
 */

#pragma once

// OpenCV Includes
#include "precomp.hpp"
#include <opencv2/features2d.hpp>

//*************************************************************************************

struct KAZEOptions {

    enum DIFFUSIVITY_TYPE {
        PM_G1 = 0,
        PM_G2 = 1,
        WEICKERT = 2
    };

    KAZEOptions()
        : diffusivity(PM_G2)

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

        , use_fed(true)
        , upright(false)
        , extended(false)

        , use_clipping_normalilzation(false)
        , clipping_normalization_ratio(1.6f)
        , clipping_normalization_niter(5)
    {
    }

    DIFFUSIVITY_TYPE diffusivity;

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

    bool use_fed;
    bool upright;
    bool extended;

    bool  use_clipping_normalilzation;
    float clipping_normalization_ratio;
    int   clipping_normalization_niter;
};

struct TEvolution {
    cv::Mat Lx, Ly;	// First order spatial derivatives
    cv::Mat Lxx, Lxy, Lyy;	// Second order spatial derivatives
    cv::Mat Lflow;	// Diffusivity image
    cv::Mat Lt;	// Evolution image
    cv::Mat Lsmooth; // Smoothed image
    cv::Mat Lstep; // Evolution step update
    cv::Mat Ldet; // Detector response
    float etime;	// Evolution time
    float esigma;	// Evolution sigma. For linear diffusion t = sigma^2 / 2
    float octave;	// Image octave
    float sublevel;	// Image sublevel in each octave
    int sigma_size;	// Integer esigma. For computing the feature detector responses
};
