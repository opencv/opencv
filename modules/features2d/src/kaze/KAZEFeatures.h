
/**
 * @file KAZE.h
 * @brief Main program for detecting and computing descriptors in a nonlinear
 * scale space
 * @date Jan 21, 2012
 * @author Pablo F. Alcantarilla
 */

#ifndef KAZE_H_
#define KAZE_H_

//*************************************************************************************
//*************************************************************************************

// Includes
#include "KAZEConfig.h"
#include "nldiffusion_functions.h"
#include "fed.h"

//*************************************************************************************
//*************************************************************************************

// KAZE Class Declaration
class KAZEFeatures {

private:

    KAZEOptions options;

    // Parameters of the Nonlinear diffusion class
    std::vector<TEvolution> evolution_;	// Vector of nonlinear diffusion evolution

    // Vector of keypoint vectors for finding extrema in multiple threads
    std::vector<std::vector<cv::KeyPoint> > kpts_par_;

    // FED parameters
    int ncycles_;                  // Number of cycles
    bool reordering_;              // Flag for reordering time steps
    std::vector<std::vector<float > > tsteps_;  // Vector of FED dynamic time steps
    std::vector<int> nsteps_;      // Vector of number of steps per cycle

    // Some auxiliary variables used in the AOS step
    cv::Mat Ltx_, Lty_, px_, py_, ax_, ay_, bx_, by_, qr_, qc_;

public:

    // Constructor
    KAZEFeatures(KAZEOptions& options);

    // Public methods for KAZE interface
    void Allocate_Memory_Evolution(void);
    int Create_Nonlinear_Scale_Space(const cv::Mat& img);
    void Feature_Detection(std::vector<cv::KeyPoint>& kpts);
    void Feature_Description(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc);

    static void Compute_Main_Orientation(cv::KeyPoint& kpt, const std::vector<TEvolution>& evolution_, const KAZEOptions& options);

private:

    // Feature Detection Methods
    void Compute_KContrast(const cv::Mat& img, const float& kper);
    void Compute_Multiscale_Derivatives(void);
    void Compute_Detector_Response(void);
    void Determinant_Hessian_Parallel(std::vector<cv::KeyPoint>& kpts);
    void Find_Extremum_Threading(const int& level);
    void Do_Subpixel_Refinement(std::vector<cv::KeyPoint>& kpts);

    // AOS Methods
    void AOS_Step_Scalar(cv::Mat &Ld, const cv::Mat &Ldprev, const cv::Mat &c, const float& stepsize);
    void AOS_Rows(const cv::Mat &Ldprev, const cv::Mat &c, const float& stepsize);
    void AOS_Columns(const cv::Mat &Ldprev, const cv::Mat &c, const float& stepsize);
    void Thomas(const cv::Mat &a, const cv::Mat &b, const cv::Mat &Ld, cv::Mat &x);

};

//*************************************************************************************
//*************************************************************************************

// Inline functions
float getAngle(const float& x, const float& y);
float gaussian(const float& x, const float& y, const float& sig);
void checkDescriptorLimits(int &x, int &y, const int& width, const int& height);
void clippingDescriptor(float *desc, const int& dsize, const int& niter, const float& ratio);
int fRound(const float& flt);

//*************************************************************************************
//*************************************************************************************

#endif // KAZE_H_
