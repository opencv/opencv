/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

/*//Implementation of the Gaussian mixture model background subtraction from:
//
//"Improved adaptive Gaussian mixture model for background subtraction"
//Z.Zivkovic
//International Conference Pattern Recognition, UK, August, 2004
//http://www.zoranz.net/Publications/zivkovic2004ICPR.pdf
//The code is very fast and performs also shadow detection.
//Number of Gausssian components is adapted per pixel.
//
// and
//
//"Efficient Adaptive Density Estimapion per Image Pixel for the Task of Background Subtraction"
//Z.Zivkovic, F. van der Heijden
//Pattern Recognition Letters, vol. 27, no. 7, pages 773-780, 2006.
//
//The algorithm similar to the standard Stauffer&Grimson algorithm with
//additional selection of the number of the Gaussian components based on:
//
//"Recursive unsupervised learning of finite mixture models "
//Z.Zivkovic, F.van der Heijden
//IEEE Trans. on Pattern Analysis and Machine Intelligence, vol.26, no.5, pages 651-656, 2004
//http://www.zoranz.net/Publications/zivkovic2004PAMI.pdf
//
//
//Example usage with as cpp class
// BackgroundSubtractorMOG2 bg_model;
//For each new image the model is updates using:
// bg_model(img, fgmask);
//
//Example usage as part of the CvBGStatModel:
// CvBGStatModel* bg_model = cvCreateGaussianBGModel2( first_frame );
//
// //update for each frame
// cvUpdateBGStatModel( tmp_frame, bg_model );//segmentation result is in bg_model->foreground
//
// //release at the program termination
// cvReleaseBGStatModel( &bg_model );
//
//Author: Z.Zivkovic, www.zoranz.net
//Date: 7-April-2011, Version:1.0
///////////*/

#include "precomp.hpp"
#include "opencl_kernels_video.hpp"

namespace cv
{

/*
 Interface of Gaussian mixture algorithm from:

 "Improved adaptive Gaussian mixture model for background subtraction"
 Z.Zivkovic
 International Conference Pattern Recognition, UK, August, 2004
 http://www.zoranz.net/Publications/zivkovic2004ICPR.pdf

 Advantages:
 -fast - number of Gausssian components is constantly adapted per pixel.
 -performs also shadow detection (see bgfg_segm_test.cpp example)

*/

// default parameters of gaussian background detection algorithm
static const int defaultHistory2 = 500; // Learning rate; alpha = 1/defaultHistory2
static const float defaultVarThreshold2 = 4.0f*4.0f;
static const int defaultNMixtures2 = 5; // maximal number of Gaussians in mixture
static const float defaultBackgroundRatio2 = 0.9f; // threshold sum of weights for background test
static const float defaultVarThresholdGen2 = 3.0f*3.0f;
static const float defaultVarInit2 = 15.0f; // initial variance for new components
static const float defaultVarMax2 = 5*defaultVarInit2;
static const float defaultVarMin2 = 4.0f;

// additional parameters
static const float defaultfCT2 = 0.05f; // complexity reduction prior constant 0 - no reduction of number of components
static const unsigned char defaultnShadowDetection2 = (unsigned char)127; // value to use in the segmentation mask for shadows, set 0 not to do shadow detection
static const float defaultfTau = 0.5f; // Tau - shadow threshold, see the paper for explanation


class BackgroundSubtractorMOG2Impl CV_FINAL : public BackgroundSubtractorMOG2
{
public:
    //! the default constructor
    BackgroundSubtractorMOG2Impl()
    {
        frameSize = Size(0,0);
        frameType = 0;

        nframes = 0;
        history = defaultHistory2;
        varThreshold = defaultVarThreshold2;
        bShadowDetection = 1;

        nmixtures = defaultNMixtures2;
        backgroundRatio = defaultBackgroundRatio2;
        fVarInit = defaultVarInit2;
        fVarMax  = defaultVarMax2;
        fVarMin = defaultVarMin2;

        varThresholdGen = defaultVarThresholdGen2;
        fCT = defaultfCT2;
        nShadowDetection =  defaultnShadowDetection2;
        fTau = defaultfTau;
#ifdef HAVE_OPENCL
        opencl_ON = true;
#endif
    }
    //! the full constructor that takes the length of the history,
    // the number of gaussian mixtures, the background ratio parameter and the noise strength
    BackgroundSubtractorMOG2Impl(int _history,  float _varThreshold, bool _bShadowDetection=true)
    {
        frameSize = Size(0,0);
        frameType = 0;

        nframes = 0;
        history = _history > 0 ? _history : defaultHistory2;
        varThreshold = (_varThreshold>0)? _varThreshold : defaultVarThreshold2;
        bShadowDetection = _bShadowDetection;

        nmixtures = defaultNMixtures2;
        backgroundRatio = defaultBackgroundRatio2;
        fVarInit = defaultVarInit2;
        fVarMax  = defaultVarMax2;
        fVarMin = defaultVarMin2;

        varThresholdGen = defaultVarThresholdGen2;
        fCT = defaultfCT2;
        nShadowDetection =  defaultnShadowDetection2;
        fTau = defaultfTau;
        name_ = "BackgroundSubtractor.MOG2";
#ifdef HAVE_OPENCL
        opencl_ON = true;
#endif
    }
    //! the destructor
    ~BackgroundSubtractorMOG2Impl() CV_OVERRIDE {}
    //! the update operator
    void apply(InputArray image, OutputArray fgmask, double learningRate) CV_OVERRIDE;

    //! computes a background image which are the mean of all background gaussians
    virtual void getBackgroundImage(OutputArray backgroundImage) const CV_OVERRIDE;

    //! re-initialization method
    void initialize(Size _frameSize, int _frameType)
    {
        frameSize = _frameSize;
        frameType = _frameType;
        nframes = 0;

        int nchannels = CV_MAT_CN(frameType);
        CV_Assert( nchannels <= CV_CN_MAX );
        CV_Assert( nmixtures <= 255);

#ifdef HAVE_OPENCL
        if (ocl::isOpenCLActivated() && opencl_ON)
        {
            create_ocl_apply_kernel();

            bool isFloat = CV_MAKETYPE(CV_32F,nchannels) == frameType;
            kernel_getBg.create("getBackgroundImage2_kernel", ocl::video::bgfg_mog2_oclsrc, format( "-D CN=%d -D FL=%d -D NMIXTURES=%d", nchannels, isFloat, nmixtures));

            if (kernel_apply.empty() || kernel_getBg.empty())
                opencl_ON = false;
        }
        else opencl_ON = false;

        if (opencl_ON)
        {
            u_weight.create(frameSize.height * nmixtures, frameSize.width, CV_32FC1);
            u_weight.setTo(Scalar::all(0));

            u_variance.create(frameSize.height * nmixtures, frameSize.width, CV_32FC1);
            u_variance.setTo(Scalar::all(0));

            if (nchannels==3)
                nchannels=4;
            u_mean.create(frameSize.height * nmixtures, frameSize.width, CV_32FC(nchannels)); //4 channels
            u_mean.setTo(Scalar::all(0));

            //make the array for keeping track of the used modes per pixel - all zeros at start
            u_bgmodelUsedModes.create(frameSize, CV_8UC1);
            u_bgmodelUsedModes.setTo(cv::Scalar::all(0));
        }
        else
#endif
        {
            // for each gaussian mixture of each pixel bg model we store ...
            // the mixture weight (w),
            // the mean (nchannels values) and
            // the covariance
            bgmodel.create( 1, frameSize.height*frameSize.width*nmixtures*(2 + nchannels), CV_32F );
            //make the array for keeping track of the used modes per pixel - all zeros at start
            bgmodelUsedModes.create(frameSize,CV_8U);
            bgmodelUsedModes = Scalar::all(0);
        }
    }

    virtual int getHistory() const CV_OVERRIDE { return history; }
    virtual void setHistory(int _nframes) CV_OVERRIDE { history = _nframes; }

    virtual int getNMixtures() const CV_OVERRIDE { return nmixtures; }
    virtual void setNMixtures(int nmix) CV_OVERRIDE { nmixtures = nmix; }

    virtual double getBackgroundRatio() const CV_OVERRIDE { return backgroundRatio; }
    virtual void setBackgroundRatio(double _backgroundRatio) CV_OVERRIDE { backgroundRatio = (float)_backgroundRatio; }

    virtual double getVarThreshold() const CV_OVERRIDE { return varThreshold; }
    virtual void setVarThreshold(double _varThreshold) CV_OVERRIDE { varThreshold = _varThreshold; }

    virtual double getVarThresholdGen() const CV_OVERRIDE { return varThresholdGen; }
    virtual void setVarThresholdGen(double _varThresholdGen) CV_OVERRIDE { varThresholdGen = (float)_varThresholdGen; }

    virtual double getVarInit() const CV_OVERRIDE { return fVarInit; }
    virtual void setVarInit(double varInit) CV_OVERRIDE { fVarInit = (float)varInit; }

    virtual double getVarMin() const CV_OVERRIDE { return fVarMin; }
    virtual void setVarMin(double varMin) CV_OVERRIDE { fVarMin = (float)varMin; }

    virtual double getVarMax() const CV_OVERRIDE { return fVarMax; }
    virtual void setVarMax(double varMax) CV_OVERRIDE { fVarMax = (float)varMax; }

    virtual double getComplexityReductionThreshold() const CV_OVERRIDE { return fCT; }
    virtual void setComplexityReductionThreshold(double ct) CV_OVERRIDE { fCT = (float)ct; }

    virtual bool getDetectShadows() const CV_OVERRIDE { return bShadowDetection; }
    virtual void setDetectShadows(bool detectshadows) CV_OVERRIDE
    {
        if (bShadowDetection == detectshadows)
            return;
        bShadowDetection = detectshadows;
#ifdef HAVE_OPENCL
        if (!kernel_apply.empty())
        {
            create_ocl_apply_kernel();
            CV_Assert( !kernel_apply.empty() );
        }
#endif
    }

    virtual int getShadowValue() const CV_OVERRIDE { return nShadowDetection; }
    virtual void setShadowValue(int value) CV_OVERRIDE { nShadowDetection = (uchar)value; }

    virtual double getShadowThreshold() const CV_OVERRIDE { return fTau; }
    virtual void setShadowThreshold(double value) CV_OVERRIDE { fTau = (float)value; }

    virtual void write(FileStorage& fs) const CV_OVERRIDE
    {
        writeFormat(fs);
        fs << "name" << name_
        << "history" << history
        << "nmixtures" << nmixtures
        << "backgroundRatio" << backgroundRatio
        << "varThreshold" << varThreshold
        << "varThresholdGen" << varThresholdGen
        << "varInit" << fVarInit
        << "varMin" << fVarMin
        << "varMax" << fVarMax
        << "complexityReductionThreshold" << fCT
        << "detectShadows" << (int)bShadowDetection
        << "shadowValue" << (int)nShadowDetection
        << "shadowThreshold" << fTau;
    }

    virtual void read(const FileNode& fn) CV_OVERRIDE
    {
        CV_Assert( (String)fn["name"] == name_ );
        history = (int)fn["history"];
        nmixtures = (int)fn["nmixtures"];
        backgroundRatio = (float)fn["backgroundRatio"];
        varThreshold = (double)fn["varThreshold"];
        varThresholdGen = (float)fn["varThresholdGen"];
        fVarInit = (float)fn["varInit"];
        fVarMin = (float)fn["varMin"];
        fVarMax = (float)fn["varMax"];
        fCT = (float)fn["complexityReductionThreshold"];
        bShadowDetection = (int)fn["detectShadows"] != 0;
        nShadowDetection = saturate_cast<uchar>((int)fn["shadowValue"]);
        fTau = (float)fn["shadowThreshold"];
    }

protected:
    Size frameSize;
    int frameType;
    Mat bgmodel;
    Mat bgmodelUsedModes;//keep track of number of modes per pixel

#ifdef HAVE_OPENCL
    //for OCL

    mutable bool opencl_ON;

    UMat u_weight;
    UMat u_variance;
    UMat u_mean;
    UMat u_bgmodelUsedModes;

    mutable ocl::Kernel kernel_apply;
    mutable ocl::Kernel kernel_getBg;
#endif

    int nframes;
    int history;
    int nmixtures;
    //! here it is the maximum allowed number of mixture components.
    //! Actual number is determined dynamically per pixel
    double varThreshold;
    // threshold on the squared Mahalanobis distance to decide if it is well described
    // by the background model or not. Related to Cthr from the paper.
    // This does not influence the update of the background. A typical value could be 4 sigma
    // and that is varThreshold=4*4=16; Corresponds to Tb in the paper.

    /////////////////////////
    // less important parameters - things you might change but be careful
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
    float fCT;//CT - complexity reduction prior
    //this is related to the number of samples needed to accept that a component
    //actually exists. We use CT=0.05 of all the samples. By setting CT=0 you get
    //the standard Stauffer&Grimson algorithm (maybe not exact but very similar)

    //shadow detection parameters
    bool bShadowDetection;//default 1 - do shadow detection
    unsigned char nShadowDetection;//do shadow detection - insert this value as the detection result - 127 default value
    float fTau;
    // Tau - shadow threshold. The shadow is detected if the pixel is darker
    //version of the background. Tau is a threshold on how much darker the shadow can be.
    //Tau= 0.5 means that if pixel is more than 2 times darker then it is not shadow
    //See: Prati,Mikic,Trivedi,Cucchiara,"Detecting Moving Shadows...",IEEE PAMI,2003.

    String name_;

    template <typename T, int CN>
    void getBackgroundImage_intern(OutputArray backgroundImage) const;

#ifdef HAVE_OPENCL
    bool ocl_getBackgroundImage(OutputArray backgroundImage) const;
    bool ocl_apply(InputArray _image, OutputArray _fgmask, double learningRate=-1);
    void create_ocl_apply_kernel();
#endif
};

struct GaussBGStatModel2Params
{
    //image info
    int nWidth;
    int nHeight;
    int nND;//number of data dimensions (image channels)

    bool bPostFiltering;//default 1 - do postfiltering - will make shadow detection results also give value 255
    double  minArea; // for postfiltering

    bool bInit;//default 1, faster updates at start

    /////////////////////////
    //very important parameters - things you will change
    ////////////////////////
    float fAlphaT;
    //alpha - speed of update - if the time interval you want to average over is T
    //set alpha=1/T. It is also useful at start to make T slowly increase
    //from 1 until the desired T
    float fTb;
    //Tb - threshold on the squared Mahalan. dist. to decide if it is well described
    //by the background model or not. Related to Cthr from the paper.
    //This does not influence the update of the background. A typical value could be 4 sigma
    //and that is Tb=4*4=16;

    /////////////////////////
    //less important parameters - things you might change but be careful
    ////////////////////////
    float fTg;
    //Tg - threshold on the squared Mahalan. dist. to decide
    //when a sample is close to the existing components. If it is not close
    //to any a new component will be generated. I use 3 sigma => Tg=3*3=9.
    //Smaller Tg leads to more generated components and higher Tg might make
    //lead to small number of components but they can grow too large
    float fTB;//1-cf from the paper
    //TB - threshold when the component becomes significant enough to be included into
    //the background model. It is the TB=1-cf from the paper. So I use cf=0.1 => TB=0.
    //For alpha=0.001 it means that the mode should exist for approximately 105 frames before
    //it is considered foreground
    float fVarInit;
    float fVarMax;
    float fVarMin;
    //initial standard deviation  for the newly generated components.
    //It will will influence the speed of adaptation. A good guess should be made.
    //A simple way is to estimate the typical standard deviation from the images.
    //I used here 10 as a reasonable value
    float fCT;//CT - complexity reduction prior
    //this is related to the number of samples needed to accept that a component
    //actually exists. We use CT=0.05 of all the samples. By setting CT=0 you get
    //the standard Stauffer&Grimson algorithm (maybe not exact but very similar)

    //even less important parameters
    int nM;//max number of modes - const - 4 is usually enough

    //shadow detection parameters
    bool bShadowDetection;//default 1 - do shadow detection
    unsigned char nShadowDetection;//do shadow detection - insert this value as the detection result
    float fTau;
    // Tau - shadow threshold. The shadow is detected if the pixel is darker
    //version of the background. Tau is a threshold on how much darker the shadow can be.
    //Tau= 0.5 means that if pixel is more than 2 times darker then it is not shadow
    //See: Prati,Mikic,Trivedi,Cucchiara,"Detecting Moving Shadows...",IEEE PAMI,2003.
};

struct GMM
{
    float weight;
    float variance;
};

// shadow detection performed per pixel
// should work for rgb data, could be useful for gray scale and depth data as well
// See: Prati,Mikic,Trivedi,Cucchiara,"Detecting Moving Shadows...",IEEE PAMI,2003.
CV_INLINE bool
detectShadowGMM(const float* data, int nchannels, int nmodes,
                const GMM* gmm, const float* mean,
                float Tb, float TB, float tau)
{
    float tWeight = 0;

    // check all the components  marked as background:
    for( int mode = 0; mode < nmodes; mode++, mean += nchannels )
    {
        GMM g = gmm[mode];

        float numerator = 0.0f;
        float denominator = 0.0f;
        for( int c = 0; c < nchannels; c++ )
        {
            numerator   += data[c] * mean[c];
            denominator += mean[c] * mean[c];
        }

        // no division by zero allowed
        if( denominator == 0 )
            return false;

        // if tau < a < 1 then also check the color distortion
        if( numerator <= denominator && numerator >= tau*denominator )
        {
            float a = numerator / denominator;
            float dist2a = 0.0f;

            for( int c = 0; c < nchannels; c++ )
            {
                float dD= a*mean[c] - data[c];
                dist2a += dD*dD;
            }

            if (dist2a < Tb*g.variance*a*a)
                return true;
        };

        tWeight += g.weight;
        if( tWeight > TB )
            return false;
    };
    return false;
}

//update GMM - the base update function performed per pixel
//
//"Efficient Adaptive Density Estimapion per Image Pixel for the Task of Background Subtraction"
//Z.Zivkovic, F. van der Heijden
//Pattern Recognition Letters, vol. 27, no. 7, pages 773-780, 2006.
//
//The algorithm similar to the standard Stauffer&Grimson algorithm with
//additional selection of the number of the Gaussian components based on:
//
//"Recursive unsupervised learning of finite mixture models "
//Z.Zivkovic, F.van der Heijden
//IEEE Trans. on Pattern Analysis and Machine Intelligence, vol.26, no.5, pages 651-656, 2004
//http://www.zoranz.net/Publications/zivkovic2004PAMI.pdf

class MOG2Invoker : public ParallelLoopBody
{
public:
    MOG2Invoker(const Mat& _src, Mat& _dst,
                GMM* _gmm, float* _mean,
                uchar* _modesUsed,
                int _nmixtures, float _alphaT,
                float _Tb, float _TB, float _Tg,
                float _varInit, float _varMin, float _varMax,
                float _prune, float _tau, bool _detectShadows,
                uchar _shadowVal)
    {
        src = &_src;
        dst = &_dst;
        gmm0 = _gmm;
        mean0 = _mean;
        modesUsed0 = _modesUsed;
        nmixtures = _nmixtures;
        alphaT = _alphaT;
        Tb = _Tb;
        TB = _TB;
        Tg = _Tg;
        varInit = _varInit;
        varMin = MIN(_varMin, _varMax);
        varMax = MAX(_varMin, _varMax);
        prune = _prune;
        tau = _tau;
        detectShadows = _detectShadows;
        shadowVal = _shadowVal;
    }

    void operator()(const Range& range) const CV_OVERRIDE
    {
        int y0 = range.start, y1 = range.end;
        int ncols = src->cols, nchannels = src->channels();
        AutoBuffer<float> buf(src->cols*nchannels);
        float alpha1 = 1.f - alphaT;
        float dData[CV_CN_MAX];

        for( int y = y0; y < y1; y++ )
        {
            const float* data = buf.data();
            if( src->depth() != CV_32F )
                src->row(y).convertTo(Mat(1, ncols, CV_32FC(nchannels), (void*)data), CV_32F);
            else
                data = src->ptr<float>(y);

            float* mean = mean0 + ncols*nmixtures*nchannels*y;
            GMM* gmm = gmm0 + ncols*nmixtures*y;
            uchar* modesUsed = modesUsed0 + ncols*y;
            uchar* mask = dst->ptr(y);

            for( int x = 0; x < ncols; x++, data += nchannels, gmm += nmixtures, mean += nmixtures*nchannels )
            {
                //calculate distances to the modes (+ sort)
                //here we need to go in descending order!!!
                bool background = false;//return value -> true - the pixel classified as background

                //internal:
                bool fitsPDF = false;//if it remains zero a new GMM mode will be added
                int nmodes = modesUsed[x];//current number of modes in GMM
                float totalWeight = 0.f;

                float* mean_m = mean;

                //////
                //go through all modes
                for( int mode = 0; mode < nmodes; mode++, mean_m += nchannels )
                {
                    float weight = alpha1*gmm[mode].weight + prune;//need only weight if fit is found
                    int swap_count = 0;
                    ////
                    //fit not found yet
                    if( !fitsPDF )
                    {
                        //check if it belongs to some of the remaining modes
                        float var = gmm[mode].variance;

                        //calculate difference and distance
                        float dist2;

                        if( nchannels == 3 )
                        {
                            dData[0] = mean_m[0] - data[0];
                            dData[1] = mean_m[1] - data[1];
                            dData[2] = mean_m[2] - data[2];
                            dist2 = dData[0]*dData[0] + dData[1]*dData[1] + dData[2]*dData[2];
                        }
                        else
                        {
                            dist2 = 0.f;
                            for( int c = 0; c < nchannels; c++ )
                            {
                                dData[c] = mean_m[c] - data[c];
                                dist2 += dData[c]*dData[c];
                            }
                        }

                        //background? - Tb - usually larger than Tg
                        if( totalWeight < TB && dist2 < Tb*var )
                            background = true;

                        //check fit
                        if( dist2 < Tg*var )
                        {
                            /////
                            //belongs to the mode
                            fitsPDF = true;

                            //update distribution

                            //update weight
                            weight += alphaT;
                            float k = alphaT/weight;

                            //update mean
                            for( int c = 0; c < nchannels; c++ )
                                mean_m[c] -= k*dData[c];

                            //update variance
                            float varnew = var + k*(dist2-var);
                            //limit the variance
                            varnew = MAX(varnew, varMin);
                            varnew = MIN(varnew, varMax);
                            gmm[mode].variance = varnew;

                            //sort
                            //all other weights are at the same place and
                            //only the matched (iModes) is higher -> just find the new place for it
                            for( int i = mode; i > 0; i-- )
                            {
                                //check one up
                                if( weight < gmm[i-1].weight )
                                    break;

                                swap_count++;
                                //swap one up
                                std::swap(gmm[i], gmm[i-1]);
                                for( int c = 0; c < nchannels; c++ )
                                    std::swap(mean[i*nchannels + c], mean[(i-1)*nchannels + c]);
                            }
                            //belongs to the mode - bFitsPDF becomes 1
                            /////
                        }
                    }//!bFitsPDF)

                    //check prune
                    if( weight < -prune )
                    {
                        weight = 0.0;
                        nmodes--;
                    }

                    gmm[mode-swap_count].weight = weight;//update weight by the calculated value
                    totalWeight += weight;
                }
                //go through all modes
                //////

                // Renormalize weights. In the special case that the pixel does
                // not agree with any modes, set weights to zero (a new mode will be added below).
                float invWeight = 0.f;
                if (std::abs(totalWeight) > FLT_EPSILON) {
                    invWeight = 1.f/totalWeight;
                }

                for( int mode = 0; mode < nmodes; mode++ )
                {
                    gmm[mode].weight *= invWeight;
                }

                //make new mode if needed and exit
                if( !fitsPDF && alphaT > 0.f )
                {
                    // replace the weakest or add a new one
                    int mode = nmodes == nmixtures ? nmixtures-1 : nmodes++;

                    if (nmodes==1)
                        gmm[mode].weight = 1.f;
                    else
                    {
                        gmm[mode].weight = alphaT;

                        // renormalize all other weights
                        for( int i = 0; i < nmodes-1; i++ )
                            gmm[i].weight *= alpha1;
                    }

                    // init
                    for( int c = 0; c < nchannels; c++ )
                        mean[mode*nchannels + c] = data[c];

                    gmm[mode].variance = varInit;

                    //sort
                    //find the new place for it
                    for( int i = nmodes - 1; i > 0; i-- )
                    {
                        // check one up
                        if( alphaT < gmm[i-1].weight )
                            break;

                        // swap one up
                        std::swap(gmm[i], gmm[i-1]);
                        for( int c = 0; c < nchannels; c++ )
                            std::swap(mean[i*nchannels + c], mean[(i-1)*nchannels + c]);
                    }
                }

                //set the number of modes
                modesUsed[x] = uchar(nmodes);
                mask[x] = background ? 0 :
                    detectShadows && detectShadowGMM(data, nchannels, nmodes, gmm, mean, Tb, TB, tau) ?
                    shadowVal : 255;
            }
        }
    }

    const Mat* src;
    Mat* dst;
    GMM* gmm0;
    float* mean0;
    uchar* modesUsed0;

    int nmixtures;
    float alphaT, Tb, TB, Tg;
    float varInit, varMin, varMax, prune, tau;

    bool detectShadows;
    uchar shadowVal;
};

#ifdef HAVE_OPENCL

bool BackgroundSubtractorMOG2Impl::ocl_apply(InputArray _image, OutputArray _fgmask, double learningRate)
{
    bool needToInitialize = nframes == 0 || learningRate >= 1 || _image.size() != frameSize || _image.type() != frameType;

    if( needToInitialize )
        initialize(_image.size(), _image.type());

    ++nframes;
    learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1./std::min( 2*nframes, history );
    CV_Assert(learningRate >= 0);

    _fgmask.create(_image.size(), CV_8U);
    UMat fgmask = _fgmask.getUMat();

    const double alpha1 = 1.0f - learningRate;

    UMat frame = _image.getUMat();

    float varMax = MAX(fVarMin, fVarMax);
    float varMin = MIN(fVarMin, fVarMax);

    int idxArg = 0;
    idxArg = kernel_apply.set(idxArg, ocl::KernelArg::ReadOnly(frame));
    idxArg = kernel_apply.set(idxArg, ocl::KernelArg::PtrReadWrite(u_bgmodelUsedModes));
    idxArg = kernel_apply.set(idxArg, ocl::KernelArg::PtrReadWrite(u_weight));
    idxArg = kernel_apply.set(idxArg, ocl::KernelArg::PtrReadWrite(u_mean));
    idxArg = kernel_apply.set(idxArg, ocl::KernelArg::PtrReadWrite(u_variance));
    idxArg = kernel_apply.set(idxArg, ocl::KernelArg::WriteOnlyNoSize(fgmask));

    idxArg = kernel_apply.set(idxArg, (float)learningRate);        //alphaT
    idxArg = kernel_apply.set(idxArg, (float)alpha1);
    idxArg = kernel_apply.set(idxArg, (float)(-learningRate*fCT));   //prune

    idxArg = kernel_apply.set(idxArg, (float)varThreshold); //c_Tb
    idxArg = kernel_apply.set(idxArg, backgroundRatio);     //c_TB
    idxArg = kernel_apply.set(idxArg, varThresholdGen);     //c_Tg
    idxArg = kernel_apply.set(idxArg, varMin);
    idxArg = kernel_apply.set(idxArg, varMax);
    idxArg = kernel_apply.set(idxArg, fVarInit);
    idxArg = kernel_apply.set(idxArg, fTau);
    if (bShadowDetection)
        kernel_apply.set(idxArg, nShadowDetection);

    size_t globalsize[] = {(size_t)frame.cols, (size_t)frame.rows, 1};
    return kernel_apply.run(2, globalsize, NULL, true);
}

bool BackgroundSubtractorMOG2Impl::ocl_getBackgroundImage(OutputArray _backgroundImage) const
{
    _backgroundImage.create(frameSize, frameType);
    UMat dst = _backgroundImage.getUMat();

    int idxArg = 0;
    idxArg = kernel_getBg.set(idxArg, ocl::KernelArg::PtrReadOnly(u_bgmodelUsedModes));
    idxArg = kernel_getBg.set(idxArg, ocl::KernelArg::PtrReadOnly(u_weight));
    idxArg = kernel_getBg.set(idxArg, ocl::KernelArg::PtrReadOnly(u_mean));
    idxArg = kernel_getBg.set(idxArg, ocl::KernelArg::WriteOnly(dst));
    kernel_getBg.set(idxArg, backgroundRatio);

    size_t globalsize[2] = {(size_t)u_bgmodelUsedModes.cols, (size_t)u_bgmodelUsedModes.rows};

    return kernel_getBg.run(2, globalsize, NULL, false);
}

void BackgroundSubtractorMOG2Impl::create_ocl_apply_kernel()
{
    int nchannels = CV_MAT_CN(frameType);
    bool isFloat = CV_MAKETYPE(CV_32F,nchannels) == frameType;
    String opts = format("-D CN=%d -D FL=%d -D NMIXTURES=%d%s", nchannels, isFloat, nmixtures, bShadowDetection ? " -D SHADOW_DETECT" : "");
    kernel_apply.create("mog2_kernel", ocl::video::bgfg_mog2_oclsrc, opts);
}

#endif

void BackgroundSubtractorMOG2Impl::apply(InputArray _image, OutputArray _fgmask, double learningRate)
{
    CV_INSTRUMENT_REGION();

#ifdef HAVE_OPENCL
    if (opencl_ON)
    {
        CV_OCL_RUN(_fgmask.isUMat(), ocl_apply(_image, _fgmask, learningRate))

        opencl_ON = false;
        nframes = 0;
    }
#endif

    bool needToInitialize = nframes == 0 || learningRate >= 1 || _image.size() != frameSize || _image.type() != frameType;

    if( needToInitialize )
        initialize(_image.size(), _image.type());

    Mat image = _image.getMat();
    _fgmask.create( image.size(), CV_8U );
    Mat fgmask = _fgmask.getMat();

    ++nframes;
    learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1./std::min( 2*nframes, history );
    CV_Assert(learningRate >= 0);

    parallel_for_(Range(0, image.rows),
                  MOG2Invoker(image, fgmask,
                              bgmodel.ptr<GMM>(),
                              (float*)(bgmodel.ptr() + sizeof(GMM)*nmixtures*image.rows*image.cols),
                              bgmodelUsedModes.ptr(), nmixtures, (float)learningRate,
                              (float)varThreshold,
                              backgroundRatio, varThresholdGen,
                              fVarInit, fVarMin, fVarMax, float(-learningRate*fCT), fTau,
                              bShadowDetection, nShadowDetection),
                              image.total()/(double)(1 << 16));
}

template <typename T, int CN>
void BackgroundSubtractorMOG2Impl::getBackgroundImage_intern(OutputArray backgroundImage) const
{
    CV_INSTRUMENT_REGION();

    Mat meanBackground(frameSize, frameType, Scalar::all(0));
    int firstGaussianIdx = 0;
    const GMM* gmm = bgmodel.ptr<GMM>();
    const float* mean = reinterpret_cast<const float*>(gmm + frameSize.width*frameSize.height*nmixtures);
    Vec<float,CN> meanVal(0.f);
    for(int row=0; row<meanBackground.rows; row++)
    {
        for(int col=0; col<meanBackground.cols; col++)
        {
            int nmodes = bgmodelUsedModes.at<uchar>(row, col);
            float totalWeight = 0.f;
            for(int gaussianIdx = firstGaussianIdx; gaussianIdx < firstGaussianIdx + nmodes; gaussianIdx++)
            {
                GMM gaussian = gmm[gaussianIdx];
                size_t meanPosition = gaussianIdx*CN;
                for(int chn = 0; chn < CN; chn++)
                {
                    meanVal(chn) += gaussian.weight * mean[meanPosition + chn];
                }
                totalWeight += gaussian.weight;

                if(totalWeight > backgroundRatio)
                    break;
            }
            float invWeight = 0.f;
            if (std::abs(totalWeight) > FLT_EPSILON) {
                invWeight = 1.f/totalWeight;
            }

            meanBackground.at<Vec<T,CN> >(row, col) = Vec<T,CN>(meanVal * invWeight);
            meanVal = 0.f;

            firstGaussianIdx += nmixtures;
        }
    }
    meanBackground.copyTo(backgroundImage);
}

void BackgroundSubtractorMOG2Impl::getBackgroundImage(OutputArray backgroundImage) const
{
    CV_Assert(frameType == CV_8UC1 || frameType == CV_8UC3 || frameType == CV_32FC1 || frameType == CV_32FC3);

#ifdef HAVE_OPENCL
    if (opencl_ON)
    {
        CV_OCL_RUN(opencl_ON, ocl_getBackgroundImage(backgroundImage))

        opencl_ON = false;
    }
#endif

    switch(frameType)
    {
    case CV_8UC1:
        getBackgroundImage_intern<uchar,1>(backgroundImage);
        break;
    case CV_8UC3:
        getBackgroundImage_intern<uchar,3>(backgroundImage);
        break;
    case CV_32FC1:
        getBackgroundImage_intern<float,1>(backgroundImage);
        break;
    case CV_32FC3:
        getBackgroundImage_intern<float,3>(backgroundImage);
        break;
    }
}

Ptr<BackgroundSubtractorMOG2> createBackgroundSubtractorMOG2(int _history, double _varThreshold,
                                                             bool _bShadowDetection)
{
    return makePtr<BackgroundSubtractorMOG2Impl>(_history, (float)_varThreshold, _bShadowDetection);
}

}

/* End of file. */
