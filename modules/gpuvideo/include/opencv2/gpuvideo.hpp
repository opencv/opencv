/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_GPUVIDEO_HPP__
#define __OPENCV_GPUVIDEO_HPP__

#include <memory>

#include "opencv2/core/gpumat.hpp"
#include "opencv2/gpufilters.hpp"

namespace cv { namespace gpu {

////////////////////////////////// Optical Flow //////////////////////////////////////////

class CV_EXPORTS BroxOpticalFlow
{
public:
    BroxOpticalFlow(float alpha_, float gamma_, float scale_factor_, int inner_iterations_, int outer_iterations_, int solver_iterations_) :
        alpha(alpha_), gamma(gamma_), scale_factor(scale_factor_),
        inner_iterations(inner_iterations_), outer_iterations(outer_iterations_), solver_iterations(solver_iterations_)
    {
    }

    //! Compute optical flow
    //! frame0 - source frame (supports only CV_32FC1 type)
    //! frame1 - frame to track (with the same size and type as frame0)
    //! u      - flow horizontal component (along x axis)
    //! v      - flow vertical component (along y axis)
    void operator ()(const GpuMat& frame0, const GpuMat& frame1, GpuMat& u, GpuMat& v, Stream& stream = Stream::Null());

    //! flow smoothness
    float alpha;

    //! gradient constancy importance
    float gamma;

    //! pyramid scale factor
    float scale_factor;

    //! number of lagged non-linearity iterations (inner loop)
    int inner_iterations;

    //! number of warping iterations (number of pyramid levels)
    int outer_iterations;

    //! number of linear system solver iterations
    int solver_iterations;

    GpuMat buf;
};

class CV_EXPORTS PyrLKOpticalFlow
{
public:
    PyrLKOpticalFlow();

    void sparse(const GpuMat& prevImg, const GpuMat& nextImg, const GpuMat& prevPts, GpuMat& nextPts,
        GpuMat& status, GpuMat* err = 0);

    void dense(const GpuMat& prevImg, const GpuMat& nextImg, GpuMat& u, GpuMat& v, GpuMat* err = 0);

    void releaseMemory();

    Size winSize;
    int maxLevel;
    int iters;
    bool useInitialFlow;

private:
    std::vector<GpuMat> prevPyr_;
    std::vector<GpuMat> nextPyr_;

    GpuMat buf_;

    GpuMat uPyr_[2];
    GpuMat vPyr_[2];
};

class CV_EXPORTS FarnebackOpticalFlow
{
public:
    FarnebackOpticalFlow()
    {
        numLevels = 5;
        pyrScale = 0.5;
        fastPyramids = false;
        winSize = 13;
        numIters = 10;
        polyN = 5;
        polySigma = 1.1;
        flags = 0;
    }

    int numLevels;
    double pyrScale;
    bool fastPyramids;
    int winSize;
    int numIters;
    int polyN;
    double polySigma;
    int flags;

    void operator ()(const GpuMat &frame0, const GpuMat &frame1, GpuMat &flowx, GpuMat &flowy, Stream &s = Stream::Null());

    void releaseMemory()
    {
        frames_[0].release();
        frames_[1].release();
        pyrLevel_[0].release();
        pyrLevel_[1].release();
        M_.release();
        bufM_.release();
        R_[0].release();
        R_[1].release();
        blurredFrame_[0].release();
        blurredFrame_[1].release();
        pyramid0_.clear();
        pyramid1_.clear();
    }

private:
    void prepareGaussian(
            int n, double sigma, float *g, float *xg, float *xxg,
            double &ig11, double &ig03, double &ig33, double &ig55);

    void setPolynomialExpansionConsts(int n, double sigma);

    void updateFlow_boxFilter(
            const GpuMat& R0, const GpuMat& R1, GpuMat& flowx, GpuMat &flowy,
            GpuMat& M, GpuMat &bufM, int blockSize, bool updateMatrices, Stream streams[]);

    void updateFlow_gaussianBlur(
            const GpuMat& R0, const GpuMat& R1, GpuMat& flowx, GpuMat& flowy,
            GpuMat& M, GpuMat &bufM, int blockSize, bool updateMatrices, Stream streams[]);

    GpuMat frames_[2];
    GpuMat pyrLevel_[2], M_, bufM_, R_[2], blurredFrame_[2];
    std::vector<GpuMat> pyramid0_, pyramid1_;
};

// Implementation of the Zach, Pock and Bischof Dual TV-L1 Optical Flow method
//
// see reference:
//   [1] C. Zach, T. Pock and H. Bischof, "A Duality Based Approach for Realtime TV-L1 Optical Flow".
//   [2] Javier Sanchez, Enric Meinhardt-Llopis and Gabriele Facciolo. "TV-L1 Optical Flow Estimation".
class CV_EXPORTS OpticalFlowDual_TVL1_GPU
{
public:
    OpticalFlowDual_TVL1_GPU();

    void operator ()(const GpuMat& I0, const GpuMat& I1, GpuMat& flowx, GpuMat& flowy);

    void collectGarbage();

    /**
     * Time step of the numerical scheme.
     */
    double tau;

    /**
     * Weight parameter for the data term, attachment parameter.
     * This is the most relevant parameter, which determines the smoothness of the output.
     * The smaller this parameter is, the smoother the solutions we obtain.
     * It depends on the range of motions of the images, so its value should be adapted to each image sequence.
     */
    double lambda;

    /**
     * Weight parameter for (u - v)^2, tightness parameter.
     * It serves as a link between the attachment and the regularization terms.
     * In theory, it should have a small value in order to maintain both parts in correspondence.
     * The method is stable for a large range of values of this parameter.
     */
    double theta;

    /**
     * Number of scales used to create the pyramid of images.
     */
    int nscales;

    /**
     * Number of warpings per scale.
     * Represents the number of times that I1(x+u0) and grad( I1(x+u0) ) are computed per scale.
     * This is a parameter that assures the stability of the method.
     * It also affects the running time, so it is a compromise between speed and accuracy.
     */
    int warps;

    /**
     * Stopping criterion threshold used in the numerical scheme, which is a trade-off between precision and running time.
     * A small value will yield more accurate solutions at the expense of a slower convergence.
     */
    double epsilon;

    /**
     * Stopping criterion iterations number used in the numerical scheme.
     */
    int iterations;

    double scaleStep;

    bool useInitialFlow;

private:
    void procOneScale(const GpuMat& I0, const GpuMat& I1, GpuMat& u1, GpuMat& u2);

    std::vector<GpuMat> I0s;
    std::vector<GpuMat> I1s;
    std::vector<GpuMat> u1s;
    std::vector<GpuMat> u2s;

    GpuMat I1x_buf;
    GpuMat I1y_buf;

    GpuMat I1w_buf;
    GpuMat I1wx_buf;
    GpuMat I1wy_buf;

    GpuMat grad_buf;
    GpuMat rho_c_buf;

    GpuMat p11_buf;
    GpuMat p12_buf;
    GpuMat p21_buf;
    GpuMat p22_buf;

    GpuMat diff_buf;
    GpuMat norm_buf;
};

//! Calculates optical flow for 2 images using block matching algorithm */
CV_EXPORTS void calcOpticalFlowBM(const GpuMat& prev, const GpuMat& curr,
                                  Size block_size, Size shift_size, Size max_range, bool use_previous,
                                  GpuMat& velx, GpuMat& vely, GpuMat& buf,
                                  Stream& stream = Stream::Null());

class CV_EXPORTS FastOpticalFlowBM
{
public:
    void operator ()(const GpuMat& I0, const GpuMat& I1, GpuMat& flowx, GpuMat& flowy, int search_window = 21, int block_window = 7, Stream& s = Stream::Null());

private:
    GpuMat buffer;
    GpuMat extended_I0;
    GpuMat extended_I1;
};


//! Interpolate frames (images) using provided optical flow (displacement field).
//! frame0   - frame 0 (32-bit floating point images, single channel)
//! frame1   - frame 1 (the same type and size)
//! fu       - forward horizontal displacement
//! fv       - forward vertical displacement
//! bu       - backward horizontal displacement
//! bv       - backward vertical displacement
//! pos      - new frame position
//! newFrame - new frame
//! buf      - temporary buffer, will have width x 6*height size, CV_32FC1 type and contain 6 GpuMat;
//!            occlusion masks            0, occlusion masks            1,
//!            interpolated forward flow  0, interpolated forward flow  1,
//!            interpolated backward flow 0, interpolated backward flow 1
//!
CV_EXPORTS void interpolateFrames(const GpuMat& frame0, const GpuMat& frame1,
                                  const GpuMat& fu, const GpuMat& fv,
                                  const GpuMat& bu, const GpuMat& bv,
                                  float pos, GpuMat& newFrame, GpuMat& buf,
                                  Stream& stream = Stream::Null());

CV_EXPORTS void createOpticalFlowNeedleMap(const GpuMat& u, const GpuMat& v, GpuMat& vertex, GpuMat& colors);

//////////////////////// Background/foreground segmentation ////////////////////////

// Foreground Object Detection from Videos Containing Complex Background.
// Liyuan Li, Weimin Huang, Irene Y.H. Gu, and Qi Tian.
// ACM MM2003 9p
class CV_EXPORTS FGDStatModel
{
public:
    struct CV_EXPORTS Params
    {
        int Lc;  // Quantized levels per 'color' component. Power of two, typically 32, 64 or 128.
        int N1c; // Number of color vectors used to model normal background color variation at a given pixel.
        int N2c; // Number of color vectors retained at given pixel.  Must be > N1c, typically ~ 5/3 of N1c.
        // Used to allow the first N1c vectors to adapt over time to changing background.

        int Lcc;  // Quantized levels per 'color co-occurrence' component.  Power of two, typically 16, 32 or 64.
        int N1cc; // Number of color co-occurrence vectors used to model normal background color variation at a given pixel.
        int N2cc; // Number of color co-occurrence vectors retained at given pixel.  Must be > N1cc, typically ~ 5/3 of N1cc.
        // Used to allow the first N1cc vectors to adapt over time to changing background.

        bool is_obj_without_holes; // If TRUE we ignore holes within foreground blobs. Defaults to TRUE.
        int perform_morphing;     // Number of erode-dilate-erode foreground-blob cleanup iterations.
        // These erase one-pixel junk blobs and merge almost-touching blobs. Default value is 1.

        float alpha1; // How quickly we forget old background pixel values seen. Typically set to 0.1.
        float alpha2; // "Controls speed of feature learning". Depends on T. Typical value circa 0.005.
        float alpha3; // Alternate to alpha2, used (e.g.) for quicker initial convergence. Typical value 0.1.

        float delta;   // Affects color and color co-occurrence quantization, typically set to 2.
        float T;       // A percentage value which determines when new features can be recognized as new background. (Typically 0.9).
        float minArea; // Discard foreground blobs whose bounding box is smaller than this threshold.

        // default Params
        Params();
    };

    // out_cn - channels count in output result (can be 3 or 4)
    // 4-channels require more memory, but a bit faster
    explicit FGDStatModel(int out_cn = 3);
    explicit FGDStatModel(const cv::gpu::GpuMat& firstFrame, const Params& params = Params(), int out_cn = 3);

    ~FGDStatModel();

    void create(const cv::gpu::GpuMat& firstFrame, const Params& params = Params());
    void release();

    int update(const cv::gpu::GpuMat& curFrame);

    //8UC3 or 8UC4 reference background image
    cv::gpu::GpuMat background;

    //8UC1 foreground image
    cv::gpu::GpuMat foreground;

    std::vector< std::vector<cv::Point> > foreground_regions;

private:
    FGDStatModel(const FGDStatModel&);
    FGDStatModel& operator=(const FGDStatModel&);

    class Impl;
    std::auto_ptr<Impl> impl_;
};

/*!
 Gaussian Mixture-based Backbround/Foreground Segmentation Algorithm

 The class implements the following algorithm:
 "An improved adaptive background mixture model for real-time tracking with shadow detection"
 P. KadewTraKuPong and R. Bowden,
 Proc. 2nd European Workshp on Advanced Video-Based Surveillance Systems, 2001."
 http://personal.ee.surrey.ac.uk/Personal/R.Bowden/publications/avbs01/avbs01.pdf
*/
class CV_EXPORTS MOG_GPU
{
public:
    //! the default constructor
    MOG_GPU(int nmixtures = -1);

    //! re-initiaization method
    void initialize(Size frameSize, int frameType);

    //! the update operator
    void operator()(const GpuMat& frame, GpuMat& fgmask, float learningRate = 0.0f, Stream& stream = Stream::Null());

    //! computes a background image which are the mean of all background gaussians
    void getBackgroundImage(GpuMat& backgroundImage, Stream& stream = Stream::Null()) const;

    //! releases all inner buffers
    void release();

    int history;
    float varThreshold;
    float backgroundRatio;
    float noiseSigma;

private:
    int nmixtures_;

    Size frameSize_;
    int frameType_;
    int nframes_;

    GpuMat weight_;
    GpuMat sortKey_;
    GpuMat mean_;
    GpuMat var_;
};

/*!
 The class implements the following algorithm:
 "Improved adaptive Gausian mixture model for background subtraction"
 Z.Zivkovic
 International Conference Pattern Recognition, UK, August, 2004.
 http://www.zoranz.net/Publications/zivkovic2004ICPR.pdf
*/
class CV_EXPORTS MOG2_GPU
{
public:
    //! the default constructor
    MOG2_GPU(int nmixtures = -1);

    //! re-initiaization method
    void initialize(Size frameSize, int frameType);

    //! the update operator
    void operator()(const GpuMat& frame, GpuMat& fgmask, float learningRate = -1.0f, Stream& stream = Stream::Null());

    //! computes a background image which are the mean of all background gaussians
    void getBackgroundImage(GpuMat& backgroundImage, Stream& stream = Stream::Null()) const;

    //! releases all inner buffers
    void release();

    // parameters
    // you should call initialize after parameters changes

    int history;

    //! here it is the maximum allowed number of mixture components.
    //! Actual number is determined dynamically per pixel
    float varThreshold;
    // threshold on the squared Mahalanobis distance to decide if it is well described
    // by the background model or not. Related to Cthr from the paper.
    // This does not influence the update of the background. A typical value could be 4 sigma
    // and that is varThreshold=4*4=16; Corresponds to Tb in the paper.

    /////////////////////////
    // less important parameters - things you might change but be carefull
    ////////////////////////

    float backgroundRatio;
    // corresponds to fTB=1-cf from the paper
    // TB - threshold when the component becomes significant enough to be included into
    // the background model. It is the TB=1-cf from the paper. So I use cf=0.1 => TB=0.
    // For alpha=0.001 it means that the mode should exist for approximately 105 frames before
    // it is considered foreground
    // float noiseSigma;
    float varThresholdGen;

    //correspondts to Tg - threshold on the squared Mahalan. dist. to decide
    //when a sample is close to the existing components. If it is not close
    //to any a new component will be generated. I use 3 sigma => Tg=3*3=9.
    //Smaller Tg leads to more generated components and higher Tg might make
    //lead to small number of components but they can grow too large
    float fVarInit;
    float fVarMin;
    float fVarMax;

    //initial variance  for the newly generated components.
    //It will will influence the speed of adaptation. A good guess should be made.
    //A simple way is to estimate the typical standard deviation from the images.
    //I used here 10 as a reasonable value
    // min and max can be used to further control the variance
    float fCT; //CT - complexity reduction prior
    //this is related to the number of samples needed to accept that a component
    //actually exists. We use CT=0.05 of all the samples. By setting CT=0 you get
    //the standard Stauffer&Grimson algorithm (maybe not exact but very similar)

    //shadow detection parameters
    bool bShadowDetection; //default 1 - do shadow detection
    unsigned char nShadowDetection; //do shadow detection - insert this value as the detection result - 127 default value
    float fTau;
    // Tau - shadow threshold. The shadow is detected if the pixel is darker
    //version of the background. Tau is a threshold on how much darker the shadow can be.
    //Tau= 0.5 means that if pixel is more than 2 times darker then it is not shadow
    //See: Prati,Mikic,Trivedi,Cucchiarra,"Detecting Moving Shadows...",IEEE PAMI,2003.

private:
    int nmixtures_;

    Size frameSize_;
    int frameType_;
    int nframes_;

    GpuMat weight_;
    GpuMat variance_;
    GpuMat mean_;

    GpuMat bgmodelUsedModes_; //keep track of number of modes per pixel
};

/**
 * Background Subtractor module. Takes a series of images and returns a sequence of mask (8UC1)
 * images of the same size, where 255 indicates Foreground and 0 represents Background.
 * This class implements an algorithm described in "Visual Tracking of Human Visitors under
 * Variable-Lighting Conditions for a Responsive Audio Art Installation," A. Godbehere,
 * A. Matsukawa, K. Goldberg, American Control Conference, Montreal, June 2012.
 */
class CV_EXPORTS GMG_GPU
{
public:
    GMG_GPU();

    /**
     * Validate parameters and set up data structures for appropriate frame size.
     * @param frameSize Input frame size
     * @param min       Minimum value taken on by pixels in image sequence. Usually 0
     * @param max       Maximum value taken on by pixels in image sequence. e.g. 1.0 or 255
     */
    void initialize(Size frameSize, float min = 0.0f, float max = 255.0f);

    /**
     * Performs single-frame background subtraction and builds up a statistical background image
     * model.
     * @param frame        Input frame
     * @param fgmask       Output mask image representing foreground and background pixels
     * @param stream       Stream for the asynchronous version
     */
    void operator ()(const GpuMat& frame, GpuMat& fgmask, float learningRate = -1.0f, Stream& stream = Stream::Null());

    //! Releases all inner buffers
    void release();

    //! Total number of distinct colors to maintain in histogram.
    int maxFeatures;

    //! Set between 0.0 and 1.0, determines how quickly features are "forgotten" from histograms.
    float learningRate;

    //! Number of frames of video to use to initialize histograms.
    int numInitializationFrames;

    //! Number of discrete levels in each channel to be used in histograms.
    int quantizationLevels;

    //! Prior probability that any given pixel is a background pixel. A sensitivity parameter.
    float backgroundPrior;

    //! Value above which pixel is determined to be FG.
    float decisionThreshold;

    //! Smoothing radius, in pixels, for cleaning up FG image.
    int smoothingRadius;

    //! Perform background model update.
    bool updateBackgroundModel;

private:
    float maxVal_, minVal_;

    Size frameSize_;

    int frameNum_;

    GpuMat nfeatures_;
    GpuMat colors_;
    GpuMat weights_;

    Ptr<FilterEngine_GPU> boxFilter_;
    GpuMat buf_;
};

}} // namespace cv { namespace gpu {

#endif /* __OPENCV_GPUVIDEO_HPP__ */
