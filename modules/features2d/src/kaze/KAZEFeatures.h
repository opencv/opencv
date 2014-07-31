
/**
 * @file KAZE.h
 * @brief Main program for detecting and computing descriptors in a nonlinear
 * scale space
 * @date Jan 21, 2012
 * @author Pablo F. Alcantarilla
 */

#ifndef __OPENCV_FEATURES_2D_KAZE_FEATURES_H__
#define __OPENCV_FEATURES_2D_KAZE_FEATURES_H__

/* ************************************************************************* */
// Includes
#include "KAZEConfig.h"
#include "nldiffusion_functions.h"
#include "fed.h"
#include "TEvolution.h"

/* ************************************************************************* */
// KAZE Class Declaration
class KAZEFeatures {

private:

        /// Parameters of the Nonlinear diffusion class
        KAZEOptions options_;               ///< Configuration options for KAZE
        std::vector<TEvolution> evolution_;    ///< Vector of nonlinear diffusion evolution

        /// Vector of keypoint vectors for finding extrema in multiple threads
    std::vector<std::vector<cv::KeyPoint> > kpts_par_;

        /// FED parameters
        int ncycles_;                  ///< Number of cycles
        bool reordering_;              ///< Flag for reordering time steps
        std::vector<std::vector<float > > tsteps_;  ///< Vector of FED dynamic time steps
        std::vector<int> nsteps_;      ///< Vector of number of steps per cycle

public:

        /// Constructor
    KAZEFeatures(KAZEOptions& options);

        /// Public methods for KAZE interface
    void Allocate_Memory_Evolution(void);
    int Create_Nonlinear_Scale_Space(const cv::Mat& img);
    void Feature_Detection(std::vector<cv::KeyPoint>& kpts);
    void Feature_Description(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc);
    static void Compute_Main_Orientation(cv::KeyPoint& kpt, const std::vector<TEvolution>& evolution_, const KAZEOptions& options);

        /// Feature Detection Methods
    void Compute_KContrast(const cv::Mat& img, const float& kper);
    void Compute_Multiscale_Derivatives(void);
    void Compute_Detector_Response(void);
        void Determinant_Hessian(std::vector<cv::KeyPoint>& kpts);
    void Do_Subpixel_Refinement(std::vector<cv::KeyPoint>& kpts);
};

#endif
