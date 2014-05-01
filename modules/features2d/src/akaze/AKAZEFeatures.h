/**
 * @file AKAZE.h
 * @brief Main class for detecting and computing binary descriptors in an
 * accelerated nonlinear scale space
 * @date Mar 27, 2013
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#pragma once

/* ************************************************************************* */
// Includes
#include "precomp.hpp"
#include "AKAZEConfig.h"

/* ************************************************************************* */
// AKAZE Class Declaration
class AKAZEFeatures {

private:

    AKAZEOptions options_;                ///< Configuration options for AKAZE
    std::vector<TEvolution> evolution_;	///< Vector of nonlinear diffusion evolution

    /// FED parameters
    int ncycles_;                  ///< Number of cycles
    bool reordering_;              ///< Flag for reordering time steps
    std::vector<std::vector<float > > tsteps_;  ///< Vector of FED dynamic time steps
    std::vector<int> nsteps_;      ///< Vector of number of steps per cycle

    /// Matrices for the M-LDB descriptor computation
    cv::Mat descriptorSamples_;  // List of positions in the grids to sample LDB bits from.
    cv::Mat descriptorBits_;
    cv::Mat bitMask_;

public:

    /// Constructor with input arguments
    AKAZEFeatures(const AKAZEOptions& options);

    /// Scale Space methods
    void Allocate_Memory_Evolution();
    int Create_Nonlinear_Scale_Space(const cv::Mat& img);
    void Feature_Detection(std::vector<cv::KeyPoint>& kpts);
    void Compute_Determinant_Hessian_Response(void);
    void Compute_Multiscale_Derivatives(void);
    void Find_Scale_Space_Extrema(std::vector<cv::KeyPoint>& kpts);
    void Do_Subpixel_Refinement(std::vector<cv::KeyPoint>& kpts);

    // Feature description methods
    void Compute_Descriptors(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc);

    static void Compute_Main_Orientation(cv::KeyPoint& kpt, const std::vector<TEvolution>& evolution_);

    // SURF Pattern Descriptor
    //void Get_SURF_Descriptor_Upright_64(const cv::KeyPoint& kpt, float* desc) const;
    //void Get_SURF_Descriptor_64(const cv::KeyPoint& kpt, float* desc) const;

    // M-SURF Pattern Descriptor
    //void Get_MSURF_Upright_Descriptor_64(const cv::KeyPoint& kpt, float* desc) const;
    //void Get_MSURF_Descriptor_64(const cv::KeyPoint& kpt, float* desc) const;

    // M-LDB Pattern Descriptor
    //void Get_Upright_MLDB_Full_Descriptor(const cv::KeyPoint& kpt, unsigned char* desc) const;
    //void Get_MLDB_Full_Descriptor(const cv::KeyPoint& kpt, unsigned char* desc) const;
    //void Get_Upright_MLDB_Descriptor_Subset(const cv::KeyPoint& kpt, unsigned char* desc);
    //void Get_MLDB_Descriptor_Subset(const cv::KeyPoint& kpt, unsigned char* desc);

    // Methods for saving some results and showing computation times
    //void Save_Scale_Space();
    //void Save_Detector_Responses();
    //void Show_Computation_Times() const;

    /// Return the computation times
    //AKAZETiming Get_Computation_Times() const {
    //    return timing_;
    //}
};

/* ************************************************************************* */
// Inline functions

// Inline functions
void generateDescriptorSubsample(cv::Mat& sampleList, cv::Mat& comparisons,
    int nbits, int pattern_size, int nchannels);
float get_angle(float x, float y);
float gaussian(float x, float y, float sigma);
void check_descriptor_limits(int& x, int& y, int width, int height);
int fRound(float flt);
