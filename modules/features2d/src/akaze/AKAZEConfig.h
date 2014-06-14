/**
 * @file AKAZEConfig.h
 * @brief AKAZE configuration file
 * @date Feb 23, 2014
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#pragma once

/* ************************************************************************* */
// OpenCV
#include "precomp.hpp"
#include <opencv2/features2d.hpp>

/* ************************************************************************* */
/// Lookup table for 2d gaussian (sigma = 2.5) where (0,0) is top left and (6,6) is bottom right
const float gauss25[7][7] = {
    { 0.02546481f, 0.02350698f, 0.01849125f, 0.01239505f, 0.00708017f, 0.00344629f, 0.00142946f },
    { 0.02350698f, 0.02169968f, 0.01706957f, 0.01144208f, 0.00653582f, 0.00318132f, 0.00131956f },
    { 0.01849125f, 0.01706957f, 0.01342740f, 0.00900066f, 0.00514126f, 0.00250252f, 0.00103800f },
    { 0.01239505f, 0.01144208f, 0.00900066f, 0.00603332f, 0.00344629f, 0.00167749f, 0.00069579f },
    { 0.00708017f, 0.00653582f, 0.00514126f, 0.00344629f, 0.00196855f, 0.00095820f, 0.00039744f },
    { 0.00344629f, 0.00318132f, 0.00250252f, 0.00167749f, 0.00095820f, 0.00046640f, 0.00019346f },
    { 0.00142946f, 0.00131956f, 0.00103800f, 0.00069579f, 0.00039744f, 0.00019346f, 0.00008024f }
};

/* ************************************************************************* */
/// AKAZE configuration options structure
struct AKAZEOptions {

    /// AKAZE Diffusivities
    enum DIFFUSIVITY_TYPE {
        PM_G1 = 0,
        PM_G2 = 1,
        WEICKERT = 2,
        CHARBONNIER = 3
    };

    AKAZEOptions()
        : omax(4)
        , nsublevels(4)
        , img_width(0)
        , img_height(0)
        , soffset(1.6f)
        , derivative_factor(1.5f)
        , sderivatives(1.0)
        , diffusivity(PM_G2)

        , dthreshold(0.001f)
        , min_dthreshold(0.00001f)

        , descriptor(cv::AKAZE::DESCRIPTOR_MLDB)
        , descriptor_size(0)
        , descriptor_channels(3)
        , descriptor_pattern_size(10)

        , kcontrast(0.001f)
        , kcontrast_percentile(0.7f)
        , kcontrast_nbins(300)
    {
    }

    int omax;                       ///< Maximum octave evolution of the image 2^sigma (coarsest scale sigma units)
    int nsublevels;                 ///< Default number of sublevels per scale level
    int img_width;                  ///< Width of the input image
    int img_height;                 ///< Height of the input image
    float soffset;                  ///< Base scale offset (sigma units)
    float derivative_factor;        ///< Factor for the multiscale derivatives
    float sderivatives;             ///< Smoothing factor for the derivatives
    DIFFUSIVITY_TYPE diffusivity;   ///< Diffusivity type

    float dthreshold;               ///< Detector response threshold to accept point
    float min_dthreshold;           ///< Minimum detector threshold to accept a point

    cv::AKAZE::DESCRIPTOR_TYPE descriptor;     ///< Type of descriptor
    int descriptor_size;            ///< Size of the descriptor in bits. 0->Full size
    int descriptor_channels;        ///< Number of channels in the descriptor (1, 2, 3)
    int descriptor_pattern_size;    ///< Actual patch size is 2*pattern_size*point.scale

    float kcontrast;                ///< The contrast factor parameter
    float kcontrast_percentile;     ///< Percentile level for the contrast factor
    int kcontrast_nbins;            ///< Number of bins for the contrast factor histogram
};

/* ************************************************************************* */
/// AKAZE nonlinear diffusion filtering evolution
struct TEvolution {

    TEvolution() {
        etime = 0.0f;
        esigma = 0.0f;
        octave = 0;
        sublevel = 0;
        sigma_size = 0;
    }

    cv::Mat Lx, Ly;	// First order spatial derivatives
    cv::Mat Lxx, Lxy, Lyy;	// Second order spatial derivatives
    cv::Mat Lflow;	// Diffusivity image
    cv::Mat Lt;	// Evolution image
    cv::Mat Lsmooth; // Smoothed image
    cv::Mat Lstep; // Evolution step update
    cv::Mat Ldet; // Detector response
    float etime;	// Evolution time
    float esigma;	// Evolution sigma. For linear diffusion t = sigma^2 / 2
    size_t octave;	// Image octave
    size_t sublevel;	// Image sublevel in each octave
    size_t sigma_size;	// Integer sigma. For computing the feature detector responses
};