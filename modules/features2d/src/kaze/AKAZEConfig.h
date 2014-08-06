/**
 * @file AKAZEConfig.h
 * @brief AKAZE configuration file
 * @date Feb 23, 2014
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#ifndef __OPENCV_FEATURES_2D_AKAZE_CONFIG_H__
#define __OPENCV_FEATURES_2D_AKAZE_CONFIG_H__

/* ************************************************************************* */
// OpenCV
#include "../precomp.hpp"
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

    AKAZEOptions()
        : omax(4)
        , nsublevels(4)
        , img_width(0)
        , img_height(0)
        , soffset(1.6f)
        , derivative_factor(1.5f)
        , sderivatives(1.0)
        , diffusivity(cv::DIFF_PM_G2)

        , dthreshold(0.001f)
        , min_dthreshold(0.00001f)

        , descriptor(cv::DESCRIPTOR_MLDB)
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
    int diffusivity;   ///< Diffusivity type

    float dthreshold;               ///< Detector response threshold to accept point
    float min_dthreshold;           ///< Minimum detector threshold to accept a point

    int descriptor;     ///< Type of descriptor
    int descriptor_size;            ///< Size of the descriptor in bits. 0->Full size
    int descriptor_channels;        ///< Number of channels in the descriptor (1, 2, 3)
    int descriptor_pattern_size;    ///< Actual patch size is 2*pattern_size*point.scale

    float kcontrast;                ///< The contrast factor parameter
    float kcontrast_percentile;     ///< Percentile level for the contrast factor
    int kcontrast_nbins;            ///< Number of bins for the contrast factor histogram
};

#endif
