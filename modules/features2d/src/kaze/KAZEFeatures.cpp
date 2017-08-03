
//=============================================================================
//
// KAZE.cpp
// Author: Pablo F. Alcantarilla
// Institution: University d'Auvergne
// Address: Clermont Ferrand, France
// Date: 21/01/2012
// Email: pablofdezalc@gmail.com
//
// KAZE Features Copyright 2012, Pablo F. Alcantarilla
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file KAZEFeatures.cpp
 * @brief Main class for detecting and describing features in a nonlinear
 * scale space
 * @date Jan 21, 2012
 * @author Pablo F. Alcantarilla
 */
#include "../precomp.hpp"
#include "KAZEFeatures.h"
#include "utils.h"

namespace cv
{

// Namespaces
using namespace std;

/* ************************************************************************* */
/**
 * @brief KAZE constructor with input options
 * @param options KAZE configuration options
 * @note The constructor allocates memory for the nonlinear scale space
 */
KAZEFeatures::KAZEFeatures(KAZEOptions& options)
        : options_(options)
{
    ncycles_ = 0;
    reordering_ = true;

    // Now allocate memory for the evolution
    Allocate_Memory_Evolution();
}

/* ************************************************************************* */
/**
 * @brief This method allocates the memory for the nonlinear diffusion evolution
 */
void KAZEFeatures::Allocate_Memory_Evolution(void) {

    // Allocate the dimension of the matrices for the evolution
    for (int i = 0; i <= options_.omax - 1; i++)
    {
        for (int j = 0; j <= options_.nsublevels - 1; j++)
        {
            TEvolution aux;
            aux.Lx = Mat::zeros(options_.img_height, options_.img_width, CV_32F);
            aux.Ly = Mat::zeros(options_.img_height, options_.img_width, CV_32F);
            aux.Lxx = Mat::zeros(options_.img_height, options_.img_width, CV_32F);
            aux.Lxy = Mat::zeros(options_.img_height, options_.img_width, CV_32F);
            aux.Lyy = Mat::zeros(options_.img_height, options_.img_width, CV_32F);
            aux.Lt = Mat::zeros(options_.img_height, options_.img_width, CV_32F);
            aux.Lsmooth = Mat::zeros(options_.img_height, options_.img_width, CV_32F);
            aux.Ldet = Mat::zeros(options_.img_height, options_.img_width, CV_32F);
            aux.esigma = options_.soffset*pow((float)2.0f, (float)(j) / (float)(options_.nsublevels)+i);
            aux.etime = 0.5f*(aux.esigma*aux.esigma);
            aux.sigma_size = cvRound(aux.esigma);
            aux.octave = i;
            aux.sublevel = j;
            evolution_.push_back(aux);
        }
    }

    // Allocate memory for the FED number of cycles and time steps
    for (size_t i = 1; i < evolution_.size(); i++)
    {
        int naux = 0;
        vector<float> tau;
        float ttime = 0.0;
        ttime = evolution_[i].etime - evolution_[i - 1].etime;
        naux = fed_tau_by_process_time(ttime, 1, 0.25f, reordering_, tau);
        nsteps_.push_back(naux);
        tsteps_.push_back(tau);
        ncycles_++;
    }
}

/* ************************************************************************* */
/**
 * @brief This method creates the nonlinear scale space for a given image
 * @param img Input image for which the nonlinear scale space needs to be created
 * @return 0 if the nonlinear scale space was created successfully. -1 otherwise
 */
int KAZEFeatures::Create_Nonlinear_Scale_Space(const Mat &img)
{
    CV_Assert(evolution_.size() > 0);

    // Copy the original image to the first level of the evolution
    img.copyTo(evolution_[0].Lt);
    gaussian_2D_convolution(evolution_[0].Lt, evolution_[0].Lt, 0, 0, options_.soffset);
    gaussian_2D_convolution(evolution_[0].Lt, evolution_[0].Lsmooth, 0, 0, options_.sderivatives);

    // Firstly compute the kcontrast factor
        Compute_KContrast(evolution_[0].Lt, options_.kcontrast_percentille);

    // Allocate memory for the flow and step images
    Mat Lflow = Mat::zeros(evolution_[0].Lt.rows, evolution_[0].Lt.cols, CV_32F);
    Mat Lstep = Mat::zeros(evolution_[0].Lt.rows, evolution_[0].Lt.cols, CV_32F);

    // Now generate the rest of evolution levels
    for (size_t i = 1; i < evolution_.size(); i++)
    {
        evolution_[i - 1].Lt.copyTo(evolution_[i].Lt);
        gaussian_2D_convolution(evolution_[i - 1].Lt, evolution_[i].Lsmooth, 0, 0, options_.sderivatives);

        // Compute the Gaussian derivatives Lx and Ly
        Scharr(evolution_[i].Lsmooth, evolution_[i].Lx, CV_32F, 1, 0, 1, 0, BORDER_DEFAULT);
        Scharr(evolution_[i].Lsmooth, evolution_[i].Ly, CV_32F, 0, 1, 1, 0, BORDER_DEFAULT);

        // Compute the conductivity equation
        if (options_.diffusivity == KAZE::DIFF_PM_G1)
            pm_g1(evolution_[i].Lx, evolution_[i].Ly, Lflow, options_.kcontrast);
        else if (options_.diffusivity == KAZE::DIFF_PM_G2)
            pm_g2(evolution_[i].Lx, evolution_[i].Ly, Lflow, options_.kcontrast);
        else if (options_.diffusivity == KAZE::DIFF_WEICKERT)
            weickert_diffusivity(evolution_[i].Lx, evolution_[i].Ly, Lflow, options_.kcontrast);

        // Perform FED n inner steps
        for (int j = 0; j < nsteps_[i - 1]; j++)
            nld_step_scalar(evolution_[i].Lt, Lflow, Lstep, tsteps_[i - 1][j]);
    }

    return 0;
}

/* ************************************************************************* */
/**
 * @brief This method computes the k contrast factor
 * @param img Input image
 * @param kpercentile Percentile of the gradient histogram
 */
void KAZEFeatures::Compute_KContrast(const Mat &img, const float &kpercentile)
{
    options_.kcontrast = compute_k_percentile(img, kpercentile, options_.sderivatives, options_.kcontrast_bins, 0, 0);
}

/* ************************************************************************* */
/**
 * @brief This method computes the feature detector response for the nonlinear scale space
 * @note We use the Hessian determinant as feature detector
 */
void KAZEFeatures::Compute_Detector_Response(void)
{
    float lxx = 0.0, lxy = 0.0, lyy = 0.0;

    // Firstly compute the multiscale derivatives
    Compute_Multiscale_Derivatives();

    for (size_t i = 0; i < evolution_.size(); i++)
    {
                for (int ix = 0; ix < options_.img_height; ix++)
        {
                        for (int jx = 0; jx < options_.img_width; jx++)
            {
                lxx = *(evolution_[i].Lxx.ptr<float>(ix)+jx);
                lxy = *(evolution_[i].Lxy.ptr<float>(ix)+jx);
                lyy = *(evolution_[i].Lyy.ptr<float>(ix)+jx);
                *(evolution_[i].Ldet.ptr<float>(ix)+jx) = (lxx*lyy - lxy*lxy);
            }
        }
    }
}

/* ************************************************************************* */
/**
 * @brief This method selects interesting keypoints through the nonlinear scale space
 * @param kpts Vector of keypoints
 */
void KAZEFeatures::Feature_Detection(std::vector<KeyPoint>& kpts)
{
    kpts.clear();
        Compute_Detector_Response();
        Determinant_Hessian(kpts);
    Do_Subpixel_Refinement(kpts);
}

/* ************************************************************************* */
class MultiscaleDerivativesKAZEInvoker : public ParallelLoopBody
{
public:
    explicit MultiscaleDerivativesKAZEInvoker(std::vector<TEvolution>& ev) : evolution_(&ev)
    {
    }

    void operator()(const Range& range) const
    {
        std::vector<TEvolution>& evolution = *evolution_;
        for (int i = range.start; i < range.end; i++)
        {
            compute_scharr_derivatives(evolution[i].Lsmooth, evolution[i].Lx, 1, 0, evolution[i].sigma_size);
            compute_scharr_derivatives(evolution[i].Lsmooth, evolution[i].Ly, 0, 1, evolution[i].sigma_size);
            compute_scharr_derivatives(evolution[i].Lx, evolution[i].Lxx, 1, 0, evolution[i].sigma_size);
            compute_scharr_derivatives(evolution[i].Ly, evolution[i].Lyy, 0, 1, evolution[i].sigma_size);
            compute_scharr_derivatives(evolution[i].Lx, evolution[i].Lxy, 0, 1, evolution[i].sigma_size);

            evolution[i].Lx = evolution[i].Lx*((evolution[i].sigma_size));
            evolution[i].Ly = evolution[i].Ly*((evolution[i].sigma_size));
            evolution[i].Lxx = evolution[i].Lxx*((evolution[i].sigma_size)*(evolution[i].sigma_size));
            evolution[i].Lxy = evolution[i].Lxy*((evolution[i].sigma_size)*(evolution[i].sigma_size));
            evolution[i].Lyy = evolution[i].Lyy*((evolution[i].sigma_size)*(evolution[i].sigma_size));
        }
    }

private:
    std::vector<TEvolution>*  evolution_;
};

/* ************************************************************************* */
/**
 * @brief This method computes the multiscale derivatives for the nonlinear scale space
 */
void KAZEFeatures::Compute_Multiscale_Derivatives(void)
{
    parallel_for_(Range(0, (int)evolution_.size()),
                                        MultiscaleDerivativesKAZEInvoker(evolution_));
}


/* ************************************************************************* */
class FindExtremumKAZEInvoker : public ParallelLoopBody
{
public:
    explicit FindExtremumKAZEInvoker(std::vector<TEvolution>& ev, std::vector<std::vector<KeyPoint> >& kpts_par,
                                                                     const KAZEOptions& options) : evolution_(&ev), kpts_par_(&kpts_par), options_(options)
    {
    }

    void operator()(const Range& range) const
    {
        std::vector<TEvolution>& evolution = *evolution_;
        std::vector<std::vector<KeyPoint> >& kpts_par = *kpts_par_;
        for (int i = range.start; i < range.end; i++)
        {
            float value = 0.0;
            bool is_extremum = false;

            for (int ix = 1; ix < options_.img_height - 1; ix++)
            {
                for (int jx = 1; jx < options_.img_width - 1; jx++)
                {
                    is_extremum = false;
                    value = *(evolution[i].Ldet.ptr<float>(ix)+jx);

                    // Filter the points with the detector threshold
                    if (value > options_.dthreshold)
                    {
                        if (value >= *(evolution[i].Ldet.ptr<float>(ix)+jx - 1))
                        {
                            // First check on the same scale
                            if (check_maximum_neighbourhood(evolution[i].Ldet, 1, value, ix, jx, 1))
                            {
                                // Now check on the lower scale
                                if (check_maximum_neighbourhood(evolution[i - 1].Ldet, 1, value, ix, jx, 0))
                                {
                                    // Now check on the upper scale
                                    if (check_maximum_neighbourhood(evolution[i + 1].Ldet, 1, value, ix, jx, 0))
                                        is_extremum = true;
                                }
                            }
                        }
                    }

                    // Add the point of interest!!
                    if (is_extremum)
                    {
                        KeyPoint point;
                        point.pt.x = (float)jx;
                        point.pt.y = (float)ix;
                        point.response = fabs(value);
                        point.size = evolution[i].esigma;
                        point.octave = (int)evolution[i].octave;
                        point.class_id = i;

                        // We use the angle field for the sublevel value
                        // Then, we will replace this angle field with the main orientation
                        point.angle = static_cast<float>(evolution[i].sublevel);
                        kpts_par[i - 1].push_back(point);
                    }
                }
            }
        }
    }

private:
    std::vector<TEvolution>*  evolution_;
    std::vector<std::vector<KeyPoint> >* kpts_par_;
    KAZEOptions options_;
};

/* ************************************************************************* */
/**
 * @brief This method performs the detection of keypoints by using the normalized
 * score of the Hessian determinant through the nonlinear scale space
 * @param kpts Vector of keypoints
 * @note We compute features for each of the nonlinear scale space level in a different processing thread
 */
void KAZEFeatures::Determinant_Hessian(std::vector<KeyPoint>& kpts)
{
    int level = 0;
    float dist = 0.0, smax = 3.0;
    int npoints = 0, id_repeated = 0;
    int left_x = 0, right_x = 0, up_y = 0, down_y = 0;
    bool is_extremum = false, is_repeated = false, is_out = false;

    // Delete the memory of the vector of keypoints vectors
    // In case we use the same kaze object for multiple images
    for (size_t i = 0; i < kpts_par_.size(); i++) {
        vector<KeyPoint>().swap(kpts_par_[i]);
    }
    kpts_par_.clear();
    vector<KeyPoint> aux;

    // Allocate memory for the vector of vectors
    for (size_t i = 1; i < evolution_.size() - 1; i++) {
        kpts_par_.push_back(aux);
    }

    parallel_for_(Range(1, (int)evolution_.size()-1),
                FindExtremumKAZEInvoker(evolution_, kpts_par_, options_));

    // Now fill the vector of keypoints!!!
    for (int i = 0; i < (int)kpts_par_.size(); i++)
    {
        for (int j = 0; j < (int)kpts_par_[i].size(); j++)
        {
            level = i + 1;
            is_extremum = true;
            is_repeated = false;
            is_out = false;

            // Check in case we have the same point as maxima in previous evolution levels
            for (int ik = 0; ik < (int)kpts.size(); ik++) {
                if (kpts[ik].class_id == level || kpts[ik].class_id == level + 1 || kpts[ik].class_id == level - 1) {
                    dist = pow(kpts_par_[i][j].pt.x - kpts[ik].pt.x, 2) + pow(kpts_par_[i][j].pt.y - kpts[ik].pt.y, 2);

                    if (dist < evolution_[level].sigma_size*evolution_[level].sigma_size) {
                        if (kpts_par_[i][j].response > kpts[ik].response) {
                            id_repeated = ik;
                            is_repeated = true;
                        }
                        else {
                            is_extremum = false;
                        }

                        break;
                    }
                }
            }

            if (is_extremum == true) {
                // Check that the point is under the image limits for the descriptor computation
                left_x = cvRound(kpts_par_[i][j].pt.x - smax*kpts_par_[i][j].size);
                right_x = cvRound(kpts_par_[i][j].pt.x + smax*kpts_par_[i][j].size);
                up_y = cvRound(kpts_par_[i][j].pt.y - smax*kpts_par_[i][j].size);
                down_y = cvRound(kpts_par_[i][j].pt.y + smax*kpts_par_[i][j].size);

                if (left_x < 0 || right_x >= evolution_[level].Ldet.cols ||
                    up_y < 0 || down_y >= evolution_[level].Ldet.rows) {
                    is_out = true;
                }

                is_out = false;

                if (is_out == false) {
                    if (is_repeated == false) {
                        kpts.push_back(kpts_par_[i][j]);
                        npoints++;
                    }
                    else {
                        kpts[id_repeated] = kpts_par_[i][j];
                    }
                }
            }
        }
    }
}

/* ************************************************************************* */
/**
 * @brief This method performs subpixel refinement of the detected keypoints
 * @param kpts Vector of detected keypoints
 */
void KAZEFeatures::Do_Subpixel_Refinement(std::vector<KeyPoint> &kpts) {

    int step = 1;
    int x = 0, y = 0;
    float Dx = 0.0, Dy = 0.0, Ds = 0.0, dsc = 0.0;
    float Dxx = 0.0, Dyy = 0.0, Dss = 0.0, Dxy = 0.0, Dxs = 0.0, Dys = 0.0;
    Mat A = Mat::zeros(3, 3, CV_32F);
    Mat b = Mat::zeros(3, 1, CV_32F);
    Mat dst = Mat::zeros(3, 1, CV_32F);

    vector<KeyPoint> kpts_(kpts);

    for (size_t i = 0; i < kpts_.size(); i++) {

        x = static_cast<int>(kpts_[i].pt.x);
        y = static_cast<int>(kpts_[i].pt.y);

        // Compute the gradient
        Dx = (1.0f / (2.0f*step))*(*(evolution_[kpts_[i].class_id].Ldet.ptr<float>(y)+x + step)
            - *(evolution_[kpts_[i].class_id].Ldet.ptr<float>(y)+x - step));
        Dy = (1.0f / (2.0f*step))*(*(evolution_[kpts_[i].class_id].Ldet.ptr<float>(y + step) + x)
            - *(evolution_[kpts_[i].class_id].Ldet.ptr<float>(y - step) + x));
        Ds = 0.5f*(*(evolution_[kpts_[i].class_id + 1].Ldet.ptr<float>(y)+x)
            - *(evolution_[kpts_[i].class_id - 1].Ldet.ptr<float>(y)+x));

        // Compute the Hessian
        Dxx = (1.0f / (step*step))*(*(evolution_[kpts_[i].class_id].Ldet.ptr<float>(y)+x + step)
            + *(evolution_[kpts_[i].class_id].Ldet.ptr<float>(y)+x - step)
            - 2.0f*(*(evolution_[kpts_[i].class_id].Ldet.ptr<float>(y)+x)));

        Dyy = (1.0f / (step*step))*(*(evolution_[kpts_[i].class_id].Ldet.ptr<float>(y + step) + x)
            + *(evolution_[kpts_[i].class_id].Ldet.ptr<float>(y - step) + x)
            - 2.0f*(*(evolution_[kpts_[i].class_id].Ldet.ptr<float>(y)+x)));

        Dss = *(evolution_[kpts_[i].class_id + 1].Ldet.ptr<float>(y)+x)
            + *(evolution_[kpts_[i].class_id - 1].Ldet.ptr<float>(y)+x)
            - 2.0f*(*(evolution_[kpts_[i].class_id].Ldet.ptr<float>(y)+x));

        Dxy = (1.0f / (4.0f*step))*(*(evolution_[kpts_[i].class_id].Ldet.ptr<float>(y + step) + x + step)
            + (*(evolution_[kpts_[i].class_id].Ldet.ptr<float>(y - step) + x - step)))
            - (1.0f / (4.0f*step))*(*(evolution_[kpts_[i].class_id].Ldet.ptr<float>(y - step) + x + step)
            + (*(evolution_[kpts_[i].class_id].Ldet.ptr<float>(y + step) + x - step)));

        Dxs = (1.0f / (4.0f*step))*(*(evolution_[kpts_[i].class_id + 1].Ldet.ptr<float>(y)+x + step)
            + (*(evolution_[kpts_[i].class_id - 1].Ldet.ptr<float>(y)+x - step)))
            - (1.0f / (4.0f*step))*(*(evolution_[kpts_[i].class_id + 1].Ldet.ptr<float>(y)+x - step)
            + (*(evolution_[kpts_[i].class_id - 1].Ldet.ptr<float>(y)+x + step)));

        Dys = (1.0f / (4.0f*step))*(*(evolution_[kpts_[i].class_id + 1].Ldet.ptr<float>(y + step) + x)
            + (*(evolution_[kpts_[i].class_id - 1].Ldet.ptr<float>(y - step) + x)))
            - (1.0f / (4.0f*step))*(*(evolution_[kpts_[i].class_id + 1].Ldet.ptr<float>(y - step) + x)
            + (*(evolution_[kpts_[i].class_id - 1].Ldet.ptr<float>(y + step) + x)));

        // Solve the linear system
        *(A.ptr<float>(0)) = Dxx;
        *(A.ptr<float>(1) + 1) = Dyy;
        *(A.ptr<float>(2) + 2) = Dss;

        *(A.ptr<float>(0) + 1) = *(A.ptr<float>(1)) = Dxy;
        *(A.ptr<float>(0) + 2) = *(A.ptr<float>(2)) = Dxs;
        *(A.ptr<float>(1) + 2) = *(A.ptr<float>(2) + 1) = Dys;

        *(b.ptr<float>(0)) = -Dx;
        *(b.ptr<float>(1)) = -Dy;
        *(b.ptr<float>(2)) = -Ds;

        solve(A, b, dst, DECOMP_LU);

        if (fabs(*(dst.ptr<float>(0))) <= 1.0f && fabs(*(dst.ptr<float>(1))) <= 1.0f && fabs(*(dst.ptr<float>(2))) <= 1.0f) {
            kpts_[i].pt.x += *(dst.ptr<float>(0));
            kpts_[i].pt.y += *(dst.ptr<float>(1));
                        dsc = kpts_[i].octave + (kpts_[i].angle + *(dst.ptr<float>(2))) / ((float)(options_.nsublevels));

            // In OpenCV the size of a keypoint is the diameter!!
                        kpts_[i].size = 2.0f*options_.soffset*pow((float)2.0f, dsc);
            kpts_[i].angle = 0.0;
        }
        // Set the points to be deleted after the for loop
        else {
            kpts_[i].response = -1;
        }
    }

    // Clear the vector of keypoints
    kpts.clear();

    for (size_t i = 0; i < kpts_.size(); i++) {
        if (kpts_[i].response != -1) {
            kpts.push_back(kpts_[i]);
        }
    }
}

/* ************************************************************************* */
class KAZE_Descriptor_Invoker : public ParallelLoopBody
{
public:
        KAZE_Descriptor_Invoker(std::vector<KeyPoint> &kpts, Mat &desc, std::vector<TEvolution>& evolution, const KAZEOptions& options)
                : kpts_(&kpts)
                , desc_(&desc)
                , evolution_(&evolution)
                , options_(options)
    {
    }

    virtual ~KAZE_Descriptor_Invoker()
    {
    }

    void operator() (const Range& range) const
    {
                std::vector<KeyPoint> &kpts      = *kpts_;
                Mat                   &desc      = *desc_;
                std::vector<TEvolution>   &evolution = *evolution_;

        for (int i = range.start; i < range.end; i++)
        {
            kpts[i].angle = 0.0;
            if (options_.upright)
            {
                kpts[i].angle = 0.0;
                                if (options_.extended)
                    Get_KAZE_Upright_Descriptor_128(kpts[i], desc.ptr<float>((int)i));
                else
                    Get_KAZE_Upright_Descriptor_64(kpts[i], desc.ptr<float>((int)i));
            }
            else
            {
                                KAZEFeatures::Compute_Main_Orientation(kpts[i], evolution, options_);

                                if (options_.extended)
                    Get_KAZE_Descriptor_128(kpts[i], desc.ptr<float>((int)i));
                else
                    Get_KAZE_Descriptor_64(kpts[i], desc.ptr<float>((int)i));
            }
        }
    }
private:
    void Get_KAZE_Upright_Descriptor_64(const KeyPoint& kpt, float* desc) const;
    void Get_KAZE_Descriptor_64(const KeyPoint& kpt, float* desc) const;
    void Get_KAZE_Upright_Descriptor_128(const KeyPoint& kpt, float* desc) const;
    void Get_KAZE_Descriptor_128(const KeyPoint& kpt, float *desc) const;

        std::vector<KeyPoint> * kpts_;
        Mat                   * desc_;
        std::vector<TEvolution>   * evolution_;
        KAZEOptions                 options_;
};

/* ************************************************************************* */
/**
 * @brief This method  computes the set of descriptors through the nonlinear scale space
 * @param kpts Vector of keypoints
 * @param desc Matrix with the feature descriptors
 */
void KAZEFeatures::Feature_Description(std::vector<KeyPoint> &kpts, Mat &desc)
{
    for(size_t i = 0; i < kpts.size(); i++)
    {
        CV_Assert(0 <= kpts[i].class_id && kpts[i].class_id < static_cast<int>(evolution_.size()));
    }

    // Allocate memory for the matrix of descriptors
        if (options_.extended == true) {
        desc = Mat::zeros((int)kpts.size(), 128, CV_32FC1);
    }
    else {
        desc = Mat::zeros((int)kpts.size(), 64, CV_32FC1);
    }

        parallel_for_(Range(0, (int)kpts.size()), KAZE_Descriptor_Invoker(kpts, desc, evolution_, options_));
}

/* ************************************************************************* */
/**
 * @brief This method computes the main orientation for a given keypoint
 * @param kpt Input keypoint
 * @note The orientation is computed using a similar approach as described in the
 * original SURF method. See Bay et al., Speeded Up Robust Features, ECCV 2006
 */
void KAZEFeatures::Compute_Main_Orientation(KeyPoint &kpt, const std::vector<TEvolution>& evolution_, const KAZEOptions& options)
{
    int ix = 0, iy = 0, idx = 0, s = 0, level = 0;
    float xf = 0.0, yf = 0.0, gweight = 0.0;
    vector<float> resX(109), resY(109), Ang(109);

    // Variables for computing the dominant direction
    float sumX = 0.0, sumY = 0.0, max = 0.0, ang1 = 0.0, ang2 = 0.0;

    // Get the information from the keypoint
    xf = kpt.pt.x;
    yf = kpt.pt.y;
    level = kpt.class_id;
    s = cvRound(kpt.size / 2.0f);

    // Calculate derivatives responses for points within radius of 6*scale
    for (int i = -6; i <= 6; ++i) {
        for (int j = -6; j <= 6; ++j) {
            if (i*i + j*j < 36) {
                iy = cvRound(yf + j*s);
                ix = cvRound(xf + i*s);

                if (iy >= 0 && iy < options.img_height && ix >= 0 && ix < options.img_width) {
                    gweight = gaussian(iy - yf, ix - xf, 2.5f*s);
                    resX[idx] = gweight*(*(evolution_[level].Lx.ptr<float>(iy)+ix));
                    resY[idx] = gweight*(*(evolution_[level].Ly.ptr<float>(iy)+ix));
                }
                else {
                    resX[idx] = 0.0;
                    resY[idx] = 0.0;
                }

                Ang[idx] = fastAtan2(resX[idx], resY[idx]) * (float)(CV_PI / 180.0f);
                ++idx;
            }
        }
    }

    // Loop slides pi/3 window around feature point
    for (ang1 = 0; ang1 < 2.0f*CV_PI; ang1 += 0.15f) {
        ang2 = (ang1 + (float)(CV_PI / 3.0) > (float)(2.0*CV_PI) ? ang1 - (float)(5.0*CV_PI / 3.0) : ang1 + (float)(CV_PI / 3.0));
        sumX = sumY = 0.f;

        for (size_t k = 0; k < Ang.size(); ++k) {
            // Get angle from the x-axis of the sample point
            const float & ang = Ang[k];

            // Determine whether the point is within the window
            if (ang1 < ang2 && ang1 < ang && ang < ang2) {
                sumX += resX[k];
                sumY += resY[k];
            }
            else if (ang2 < ang1 &&
                ((ang > 0 && ang < ang2) || (ang > ang1 && ang < (float)(2.0*CV_PI)))) {
                sumX += resX[k];
                sumY += resY[k];
            }
        }

        // if the vector produced from this window is longer than all
        // previous vectors then this forms the new dominant direction
        if (sumX*sumX + sumY*sumY > max) {
            // store largest orientation
            max = sumX*sumX + sumY*sumY;
            kpt.angle = fastAtan2(sumX, sumY);
        }
    }
}

/* ************************************************************************* */
/**
 * @brief This method computes the upright descriptor (not rotation invariant) of
 * the provided keypoint
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 * @note Rectangular grid of 24 s x 24 s. Descriptor Length 64. The descriptor is inspired
 * from Agrawal et al., CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching,
 * ECCV 2008
 */
void KAZE_Descriptor_Invoker::Get_KAZE_Upright_Descriptor_64(const KeyPoint &kpt, float *desc) const
{
    float dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0, gauss_s1 = 0.0, gauss_s2 = 0.0;
    float rx = 0.0, ry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, ys = 0.0, xs = 0.0;
    float sample_x = 0.0, sample_y = 0.0;
    int x1 = 0, y1 = 0, sample_step = 0, pattern_size = 0;
    int x2 = 0, y2 = 0, kx = 0, ky = 0, i = 0, j = 0, dcount = 0;
    float fx = 0.0, fy = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
    int dsize = 0, scale = 0, level = 0;

        std::vector<TEvolution>& evolution = *evolution_;

    // Subregion centers for the 4x4 gaussian weighting
    float cx = -0.5f, cy = 0.5f;

    // Set the descriptor size and the sample and pattern sizes
    dsize = 64;
    sample_step = 5;
    pattern_size = 12;

    // Get the information from the keypoint
    yf = kpt.pt.y;
    xf = kpt.pt.x;
    scale = cvRound(kpt.size / 2.0f);
    level = kpt.class_id;

    i = -8;

    // Calculate descriptor for this interest point
    // Area of size 24 s x 24 s
    while (i < pattern_size) {
        j = -8;
        i = i - 4;

        cx += 1.0f;
        cy = -0.5f;

        while (j < pattern_size) {

            dx = dy = mdx = mdy = 0.0;
            cy += 1.0f;
            j = j - 4;

            ky = i + sample_step;
            kx = j + sample_step;

            ys = yf + (ky*scale);
            xs = xf + (kx*scale);

            for (int k = i; k < i + 9; k++) {
                for (int l = j; l < j + 9; l++) {

                    sample_y = k*scale + yf;
                    sample_x = l*scale + xf;

                    //Get the gaussian weighted x and y responses
                    gauss_s1 = gaussian(xs - sample_x, ys - sample_y, 2.5f*scale);

                    y1 = (int)(sample_y - 0.5f);
                    x1 = (int)(sample_x - 0.5f);

                                        checkDescriptorLimits(x1, y1, options_.img_width, options_.img_height);

                    y2 = (int)(sample_y + 0.5f);
                    x2 = (int)(sample_x + 0.5f);

                                        checkDescriptorLimits(x2, y2, options_.img_width, options_.img_height);

                    fx = sample_x - x1;
                    fy = sample_y - y1;

                                        res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
                                        res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
                                        res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
                                        res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
                    rx = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

                                        res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
                                        res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
                                        res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
                                        res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
                    ry = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

                    rx = gauss_s1*rx;
                    ry = gauss_s1*ry;

                    // Sum the derivatives to the cumulative descriptor
                    dx += rx;
                    dy += ry;
                    mdx += fabs(rx);
                    mdy += fabs(ry);
                }
            }

            // Add the values to the descriptor vector
            gauss_s2 = gaussian(cx - 2.0f, cy - 2.0f, 1.5f);

            desc[dcount++] = dx*gauss_s2;
            desc[dcount++] = dy*gauss_s2;
            desc[dcount++] = mdx*gauss_s2;
            desc[dcount++] = mdy*gauss_s2;

            len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy)*gauss_s2*gauss_s2;

            j += 9;
        }

        i += 9;
    }

    // convert to unit vector
    len = sqrt(len);

    for (i = 0; i < dsize; i++) {
        desc[i] /= len;
    }
}

/* ************************************************************************* */
/**
 * @brief This method computes the descriptor of the provided keypoint given the
 * main orientation of the keypoint
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 * @note Rectangular grid of 24 s x 24 s. Descriptor Length 64. The descriptor is inspired
 * from Agrawal et al., CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching,
 * ECCV 2008
 */
void KAZE_Descriptor_Invoker::Get_KAZE_Descriptor_64(const KeyPoint &kpt, float *desc) const
{
    float dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0, gauss_s1 = 0.0, gauss_s2 = 0.0;
    float rx = 0.0, ry = 0.0, rrx = 0.0, rry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, ys = 0.0, xs = 0.0;
    float sample_x = 0.0, sample_y = 0.0, co = 0.0, si = 0.0, angle = 0.0;
    float fx = 0.0, fy = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0;
    int kx = 0, ky = 0, i = 0, j = 0, dcount = 0;
    int dsize = 0, scale = 0, level = 0;

        std::vector<TEvolution>& evolution = *evolution_;

    // Subregion centers for the 4x4 gaussian weighting
    float cx = -0.5f, cy = 0.5f;

    // Set the descriptor size and the sample and pattern sizes
    dsize = 64;
    sample_step = 5;
    pattern_size = 12;

    // Get the information from the keypoint
    yf = kpt.pt.y;
    xf = kpt.pt.x;
    scale = cvRound(kpt.size / 2.0f);
    angle = kpt.angle * static_cast<float>(CV_PI / 180.f);
    level = kpt.class_id;
    co = cos(angle);
    si = sin(angle);

    i = -8;

    // Calculate descriptor for this interest point
    // Area of size 24 s x 24 s
    while (i < pattern_size) {

        j = -8;
        i = i - 4;

        cx += 1.0f;
        cy = -0.5f;

        while (j < pattern_size) {

            dx = dy = mdx = mdy = 0.0;
            cy += 1.0f;
            j = j - 4;

            ky = i + sample_step;
            kx = j + sample_step;

            xs = xf + (-kx*scale*si + ky*scale*co);
            ys = yf + (kx*scale*co + ky*scale*si);

            for (int k = i; k < i + 9; ++k) {
                for (int l = j; l < j + 9; ++l) {

                    // Get coords of sample point on the rotated axis
                    sample_y = yf + (l*scale*co + k*scale*si);
                    sample_x = xf + (-l*scale*si + k*scale*co);

                    // Get the gaussian weighted x and y responses
                    gauss_s1 = gaussian(xs - sample_x, ys - sample_y, 2.5f*scale);
                    y1 = cvFloor(sample_y);
                    x1 = cvFloor(sample_x);

                                        checkDescriptorLimits(x1, y1, options_.img_width, options_.img_height);

                    y2 = y1 + 1;
                    x2 = x1 + 1;

                                        checkDescriptorLimits(x2, y2, options_.img_width, options_.img_height);

                    fx = sample_x - x1;
                    fy = sample_y - y1;

                                        res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
                                        res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
                                        res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
                                        res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
                    rx = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

                                        res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
                                        res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
                                        res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
                                        res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
                    ry = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

                    // Get the x and y derivatives on the rotated axis
                    rry = gauss_s1*(rx*co + ry*si);
                    rrx = gauss_s1*(-rx*si + ry*co);

                    // Sum the derivatives to the cumulative descriptor
                    dx += rrx;
                    dy += rry;
                    mdx += fabs(rrx);
                    mdy += fabs(rry);
                }
            }

            // Add the values to the descriptor vector
            gauss_s2 = gaussian(cx - 2.0f, cy - 2.0f, 1.5f);
            desc[dcount++] = dx*gauss_s2;
            desc[dcount++] = dy*gauss_s2;
            desc[dcount++] = mdx*gauss_s2;
            desc[dcount++] = mdy*gauss_s2;
            len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy)*gauss_s2*gauss_s2;
            j += 9;
        }
        i += 9;
    }

    // convert to unit vector
    len = sqrt(len);

    for (i = 0; i < dsize; i++) {
        desc[i] /= len;
    }
}

/* ************************************************************************* */
/**
 * @brief This method computes the extended upright descriptor (not rotation invariant) of
 * the provided keypoint
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 * @note Rectangular grid of 24 s x 24 s. Descriptor Length 128. The descriptor is inspired
 * from Agrawal et al., CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching,
 * ECCV 2008
 */
void KAZE_Descriptor_Invoker::Get_KAZE_Upright_Descriptor_128(const KeyPoint &kpt, float *desc) const
{
    float gauss_s1 = 0.0, gauss_s2 = 0.0;
    float rx = 0.0, ry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, ys = 0.0, xs = 0.0;
    float sample_x = 0.0, sample_y = 0.0;
    int x1 = 0, y1 = 0, sample_step = 0, pattern_size = 0;
    int x2 = 0, y2 = 0, kx = 0, ky = 0, i = 0, j = 0, dcount = 0;
    float fx = 0.0, fy = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
    float dxp = 0.0, dyp = 0.0, mdxp = 0.0, mdyp = 0.0;
    float dxn = 0.0, dyn = 0.0, mdxn = 0.0, mdyn = 0.0;
    int dsize = 0, scale = 0, level = 0;

    // Subregion centers for the 4x4 gaussian weighting
    float cx = -0.5f, cy = 0.5f;

        std::vector<TEvolution>& evolution = *evolution_;

    // Set the descriptor size and the sample and pattern sizes
    dsize = 128;
    sample_step = 5;
    pattern_size = 12;

    // Get the information from the keypoint
    yf = kpt.pt.y;
    xf = kpt.pt.x;
    scale = cvRound(kpt.size / 2.0f);
    level = kpt.class_id;

    i = -8;

    // Calculate descriptor for this interest point
    // Area of size 24 s x 24 s
    while (i < pattern_size) {

        j = -8;
        i = i - 4;

        cx += 1.0f;
        cy = -0.5f;

        while (j < pattern_size) {

            dxp = dxn = mdxp = mdxn = 0.0;
            dyp = dyn = mdyp = mdyn = 0.0;

            cy += 1.0f;
            j = j - 4;

            ky = i + sample_step;
            kx = j + sample_step;

            ys = yf + (ky*scale);
            xs = xf + (kx*scale);

            for (int k = i; k < i + 9; k++) {
                for (int l = j; l < j + 9; l++) {

                    sample_y = k*scale + yf;
                    sample_x = l*scale + xf;

                    //Get the gaussian weighted x and y responses
                    gauss_s1 = gaussian(xs - sample_x, ys - sample_y, 2.5f*scale);

                    y1 = (int)(sample_y - 0.5f);
                    x1 = (int)(sample_x - 0.5f);

                                        checkDescriptorLimits(x1, y1, options_.img_width, options_.img_height);

                    y2 = (int)(sample_y + 0.5f);
                    x2 = (int)(sample_x + 0.5f);

                                        checkDescriptorLimits(x2, y2, options_.img_width, options_.img_height);

                    fx = sample_x - x1;
                    fy = sample_y - y1;

                                        res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
                                        res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
                                        res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
                                        res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
                    rx = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

                                        res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
                                        res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
                                        res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
                                        res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
                    ry = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

                    rx = gauss_s1*rx;
                    ry = gauss_s1*ry;

                    // Sum the derivatives to the cumulative descriptor
                    if (ry >= 0.0) {
                        dxp += rx;
                        mdxp += fabs(rx);
                    }
                    else {
                        dxn += rx;
                        mdxn += fabs(rx);
                    }

                    if (rx >= 0.0) {
                        dyp += ry;
                        mdyp += fabs(ry);
                    }
                    else {
                        dyn += ry;
                        mdyn += fabs(ry);
                    }
                }
            }

            // Add the values to the descriptor vector
            gauss_s2 = gaussian(cx - 2.0f, cy - 2.0f, 1.5f);

            desc[dcount++] = dxp*gauss_s2;
            desc[dcount++] = dxn*gauss_s2;
            desc[dcount++] = mdxp*gauss_s2;
            desc[dcount++] = mdxn*gauss_s2;
            desc[dcount++] = dyp*gauss_s2;
            desc[dcount++] = dyn*gauss_s2;
            desc[dcount++] = mdyp*gauss_s2;
            desc[dcount++] = mdyn*gauss_s2;

            // Store the current length^2 of the vector
            len += (dxp*dxp + dxn*dxn + mdxp*mdxp + mdxn*mdxn +
                dyp*dyp + dyn*dyn + mdyp*mdyp + mdyn*mdyn)*gauss_s2*gauss_s2;

            j += 9;
        }

        i += 9;
    }

    // convert to unit vector
    len = sqrt(len);

    for (i = 0; i < dsize; i++) {
        desc[i] /= len;
    }
}

/* ************************************************************************* */
/**
 * @brief This method computes the extended G-SURF descriptor of the provided keypoint
 * given the main orientation of the keypoint
 * @param kpt Input keypoint
 * @param desc Descriptor vector
 * @note Rectangular grid of 24 s x 24 s. Descriptor Length 128. The descriptor is inspired
 * from Agrawal et al., CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching,
 * ECCV 2008
 */
void KAZE_Descriptor_Invoker::Get_KAZE_Descriptor_128(const KeyPoint &kpt, float *desc) const
{
    float gauss_s1 = 0.0, gauss_s2 = 0.0;
    float rx = 0.0, ry = 0.0, rrx = 0.0, rry = 0.0, len = 0.0, xf = 0.0, yf = 0.0, ys = 0.0, xs = 0.0;
    float sample_x = 0.0, sample_y = 0.0, co = 0.0, si = 0.0, angle = 0.0;
    float fx = 0.0, fy = 0.0, res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
    float dxp = 0.0, dyp = 0.0, mdxp = 0.0, mdyp = 0.0;
    float dxn = 0.0, dyn = 0.0, mdxn = 0.0, mdyn = 0.0;
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0, sample_step = 0, pattern_size = 0;
    int kx = 0, ky = 0, i = 0, j = 0, dcount = 0;
    int dsize = 0, scale = 0, level = 0;

        std::vector<TEvolution>& evolution = *evolution_;

    // Subregion centers for the 4x4 gaussian weighting
    float cx = -0.5f, cy = 0.5f;

    // Set the descriptor size and the sample and pattern sizes
    dsize = 128;
    sample_step = 5;
    pattern_size = 12;

    // Get the information from the keypoint
    yf = kpt.pt.y;
    xf = kpt.pt.x;
    scale = cvRound(kpt.size / 2.0f);
    angle = kpt.angle * static_cast<float>(CV_PI / 180.f);
    level = kpt.class_id;
    co = cos(angle);
    si = sin(angle);

    i = -8;

    // Calculate descriptor for this interest point
    // Area of size 24 s x 24 s
    while (i < pattern_size) {

        j = -8;
        i = i - 4;

        cx += 1.0f;
        cy = -0.5f;

        while (j < pattern_size) {

            dxp = dxn = mdxp = mdxn = 0.0;
            dyp = dyn = mdyp = mdyn = 0.0;

            cy += 1.0f;
            j = j - 4;

            ky = i + sample_step;
            kx = j + sample_step;

            xs = xf + (-kx*scale*si + ky*scale*co);
            ys = yf + (kx*scale*co + ky*scale*si);

            for (int k = i; k < i + 9; ++k) {
                for (int l = j; l < j + 9; ++l) {

                    // Get coords of sample point on the rotated axis
                    sample_y = yf + (l*scale*co + k*scale*si);
                    sample_x = xf + (-l*scale*si + k*scale*co);

                    // Get the gaussian weighted x and y responses
                    gauss_s1 = gaussian(xs - sample_x, ys - sample_y, 2.5f*scale);

                    y1 = cvFloor(sample_y);
                    x1 = cvFloor(sample_x);

                                        checkDescriptorLimits(x1, y1, options_.img_width, options_.img_height);

                    y2 = y1 + 1;
                    x2 = x1 + 1;

                                        checkDescriptorLimits(x2, y2, options_.img_width, options_.img_height);

                    fx = sample_x - x1;
                    fy = sample_y - y1;

                                        res1 = *(evolution[level].Lx.ptr<float>(y1)+x1);
                                        res2 = *(evolution[level].Lx.ptr<float>(y1)+x2);
                                        res3 = *(evolution[level].Lx.ptr<float>(y2)+x1);
                                        res4 = *(evolution[level].Lx.ptr<float>(y2)+x2);
                    rx = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

                                        res1 = *(evolution[level].Ly.ptr<float>(y1)+x1);
                                        res2 = *(evolution[level].Ly.ptr<float>(y1)+x2);
                                        res3 = *(evolution[level].Ly.ptr<float>(y2)+x1);
                                        res4 = *(evolution[level].Ly.ptr<float>(y2)+x2);
                    ry = (1.0f - fx)*(1.0f - fy)*res1 + fx*(1.0f - fy)*res2 + (1.0f - fx)*fy*res3 + fx*fy*res4;

                    // Get the x and y derivatives on the rotated axis
                    rry = gauss_s1*(rx*co + ry*si);
                    rrx = gauss_s1*(-rx*si + ry*co);

                    // Sum the derivatives to the cumulative descriptor
                    if (rry >= 0.0) {
                        dxp += rrx;
                        mdxp += fabs(rrx);
                    }
                    else {
                        dxn += rrx;
                        mdxn += fabs(rrx);
                    }

                    if (rrx >= 0.0) {
                        dyp += rry;
                        mdyp += fabs(rry);
                    }
                    else {
                        dyn += rry;
                        mdyn += fabs(rry);
                    }
                }
            }

            // Add the values to the descriptor vector
            gauss_s2 = gaussian(cx - 2.0f, cy - 2.0f, 1.5f);

            desc[dcount++] = dxp*gauss_s2;
            desc[dcount++] = dxn*gauss_s2;
            desc[dcount++] = mdxp*gauss_s2;
            desc[dcount++] = mdxn*gauss_s2;
            desc[dcount++] = dyp*gauss_s2;
            desc[dcount++] = dyn*gauss_s2;
            desc[dcount++] = mdyp*gauss_s2;
            desc[dcount++] = mdyn*gauss_s2;

            // Store the current length^2 of the vector
            len += (dxp*dxp + dxn*dxn + mdxp*mdxp + mdxn*mdxn +
                dyp*dyp + dyn*dyn + mdyp*mdyp + mdyn*mdyn)*gauss_s2*gauss_s2;

            j += 9;
        }

        i += 9;
    }

    // convert to unit vector
    len = sqrt(len);

    for (i = 0; i < dsize; i++) {
        desc[i] /= len;
    }
}

}
