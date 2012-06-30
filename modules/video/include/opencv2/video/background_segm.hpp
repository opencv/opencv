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

#ifndef __OPENCV_BACKGROUND_SEGM_HPP__
#define __OPENCV_BACKGROUND_SEGM_HPP__

#include "opencv2/core/core.hpp"
#include <list>
namespace cv
{

/*!
 The Base Class for Background/Foreground Segmentation

 The class is only used to define the common interface for
 the whole family of background/foreground segmentation algorithms.
*/
class CV_EXPORTS_W BackgroundSubtractor : public Algorithm
{
public:
    //! the virtual destructor
    virtual ~BackgroundSubtractor();
    //! the update operator that takes the next video frame and returns the current foreground mask as 8-bit binary image.
    CV_WRAP_AS(apply) virtual void operator()(InputArray image, OutputArray fgmask,
                                              double learningRate=0);

    //! computes a background image
    virtual void getBackgroundImage(OutputArray backgroundImage) const;
};


/*!
 Gaussian Mixture-based Backbround/Foreground Segmentation Algorithm

 The class implements the following algorithm:
 "An improved adaptive background mixture model for real-time tracking with shadow detection"
 P. KadewTraKuPong and R. Bowden,
 Proc. 2nd European Workshp on Advanced Video-Based Surveillance Systems, 2001."
 http://personal.ee.surrey.ac.uk/Personal/R.Bowden/publications/avbs01/avbs01.pdf

*/
class CV_EXPORTS_W BackgroundSubtractorMOG : public BackgroundSubtractor
{
public:
    //! the default constructor
    CV_WRAP BackgroundSubtractorMOG();
    //! the full constructor that takes the length of the history, the number of gaussian mixtures, the background ratio parameter and the noise strength
    CV_WRAP BackgroundSubtractorMOG(int history, int nmixtures, double backgroundRatio, double noiseSigma=0);
    //! the destructor
    virtual ~BackgroundSubtractorMOG();
    //! the update operator
    virtual void operator()(InputArray image, OutputArray fgmask, double learningRate=0);

    //! re-initiaization method
    virtual void initialize(Size frameSize, int frameType);

    virtual AlgorithmInfo* info() const;

protected:
    Size frameSize;
    int frameType;
    Mat bgmodel;
    int nframes;
    int history;
    int nmixtures;
    double varThreshold;
    double backgroundRatio;
    double noiseSigma;
};


/*!
 The class implements the following algorithm:
 "Improved adaptive Gausian mixture model for background subtraction"
 Z.Zivkovic
 International Conference Pattern Recognition, UK, August, 2004.
 http://www.zoranz.net/Publications/zivkovic2004ICPR.pdf
*/
class CV_EXPORTS BackgroundSubtractorMOG2 : public BackgroundSubtractor
{
public:
    //! the default constructor
    BackgroundSubtractorMOG2();
    //! the full constructor that takes the length of the history, the number of gaussian mixtures, the background ratio parameter and the noise strength
    BackgroundSubtractorMOG2(int history,  float varThreshold, bool bShadowDetection=true);
    //! the destructor
    virtual ~BackgroundSubtractorMOG2();
    //! the update operator
    virtual void operator()(InputArray image, OutputArray fgmask, double learningRate=-1);

    //! computes a background image which are the mean of all background gaussians
    virtual void getBackgroundImage(OutputArray backgroundImage) const;

    //! re-initiaization method
    virtual void initialize(Size frameSize, int frameType);

    virtual AlgorithmInfo* info() const;

protected:
    Size frameSize;
    int frameType;
    Mat bgmodel;
    Mat bgmodelUsedModes;//keep track of number of modes per pixel
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
    //See: Prati,Mikic,Trivedi,Cucchiarra,"Detecting Moving Shadows...",IEEE PAMI,2003.
};

/**
 * Background Subtractor module. Takes a series of images and returns a sequence of mask (8UC1)
 * images of the same size, where 255 indicates Foreground and 0 represents Background.
 * This class implements an algorithm described in "Visual Tracking of Human Visitors under
 * Variable-Lighting Conditions for a Responsive Audio Art Installation," A. Godbehere,
 * A. Matsukawa, K. Goldberg, American Control Conference, Montreal, June 2012.
 */
class CV_EXPORTS BackgroundSubtractorGMG: public cv::BackgroundSubtractor
{
protected:
    /**
     *  Used internally to represent a single feature in a histogram.
     *  Feature is a color and an associated likelihood (weight in the histogram).
     */
    struct CV_EXPORTS HistogramFeatureGMG
    {
        /**
         * Default constructor.
         * Initializes likelihood of feature to 0, color remains uninitialized.
         */
        HistogramFeatureGMG(){likelihood = 0.0;}

        /**
         * Copy constructor.
         * Required to use HistogramFeatureGMG in a std::vector
         * @see operator =()
         */
        HistogramFeatureGMG(const HistogramFeatureGMG& orig){
            color = orig.color; likelihood = orig.likelihood;
        }

        /**
         * Assignment operator.
         * Required to use HistogramFeatureGMG in a std::vector
         */
        HistogramFeatureGMG& operator =(const HistogramFeatureGMG& orig){
            color = orig.color; likelihood = orig.likelihood; return *this;
        }

        /**
         * Tests equality of histogram features.
         * Equality is tested only by matching the color (feature), not the likelihood.
         * This operator is used to look up an observed feature in a histogram.
         */
        bool operator ==(HistogramFeatureGMG &rhs);

        //! Regardless of the image datatype, it is quantized and mapped to an integer and represented as a vector.
        vector<size_t>          color;

        //! Represents the weight of feature in the histogram.
        float                   likelihood;
        friend class PixelModelGMG;
    };

    /**
     *  Representation of the statistical model of a single pixel for use in the background subtraction
     *  algorithm.
     */
    class CV_EXPORTS PixelModelGMG
    {
    public:
        PixelModelGMG();
        ~PixelModelGMG();

        /**
         *  Incorporate the last observed feature into the statistical model.
         *
         *  @param learningRate The adaptation parameter for the histogram. -1.0 to use default. Value
         *                      should be between 0.0 and 1.0, the higher the value, the faster the
         *                      adaptation. 1.0 is limiting case where fast adaptation means no memory.
         */
        void    insertFeature(double learningRate = -1.0);

        /**
         *  Set the feature last observed, to save before incorporating it into the statistical
         *  model with insertFeature().
         *
         *  @param feature      The feature (color) just observed.
         */
        void    setLastObservedFeature(BackgroundSubtractorGMG::HistogramFeatureGMG feature);
        /**
         *  Set the upper limit for the number of features to store in the histogram. Use to adjust
         *  memory requirements.
         *
         *  @param max          size_t representing the max number of features.
         */
        void    setMaxFeatures(size_t max) {
            maxFeatures = max; histogram.resize(max); histogram.clear();
        }
        /**
         *  Normalize the histogram, so sum of weights of all features = 1.0
         */
        void    normalizeHistogram();
        /**
         *  Return the weight of a feature in the histogram. If the feature is not represented in the
         *  histogram, the weight returned is 0.0.
         */
        double  getLikelihood(HistogramFeatureGMG f);
        PixelModelGMG& operator *=(const float &rhs);
        //friend class BackgroundSubtractorGMG;
        //friend class HistogramFeatureGMG;
    private:
        size_t numFeatures; //!< number of features in histogram
        size_t maxFeatures; //!< max allowable features in histogram
        std::list<HistogramFeatureGMG> histogram; //!< represents the histogram as a list of features
        HistogramFeatureGMG            lastObservedFeature;
        //!< store last observed feature in case we need to add it to histogram
    };

public:
    BackgroundSubtractorGMG();
    virtual ~BackgroundSubtractorGMG();
    virtual AlgorithmInfo* info() const;

    /**
     * Performs single-frame background subtraction and builds up a statistical background image
     * model.
     * @param image Input image
     * @param fgmask Output mask image representing foreground and background pixels
     */
    virtual void operator()(InputArray image, OutputArray fgmask, double learningRate=-1.0);

    /**
     * Validate parameters and set up data structures for appropriate image type. Must call before
     * running on data.
     * @param image One sample image from dataset
     * @param min   minimum value taken on by pixels in image sequence. Usually 0
     * @param max   maximum value taken on by pixels in image sequence. e.g. 1.0 or 255
     */
    void    initializeType(InputArray image, double min, double max);
    /**
     * Selectively update the background model. Only update background model for pixels identified
     * as background.
     * @param mask  Mask image same size as images in sequence. Must be 8UC1 matrix, 255 for foreground
     * and 0 for background.
     */
    void    updateBackgroundModel(InputArray mask);
    /**
     * Retrieve the greyscale image representing the probability that each pixel is foreground given
     * the current estimated background model. Values are 0.0 (black) to 1.0 (white).
     * @param img The 32FC1 image representing per-pixel probabilities that the pixel is foreground.
     */
    void    getPosteriorImage(OutputArray img);

protected:
    //! Total number of distinct colors to maintain in histogram.
    int     maxFeatures;
    //! Set between 0.0 and 1.0, determines how quickly features are "forgotten" from histograms.
    double  learningRate;
    //! Number of frames of video to use to initialize histograms.
    int     numInitializationFrames;
    //! Number of discrete levels in each channel to be used in histograms.
    int     quantizationLevels;
    //! Prior probability that any given pixel is a background pixel. A sensitivity parameter.
    double  backgroundPrior;

    double  decisionThreshold; //!< value above which pixel is determined to be FG.
    int     smoothingRadius;  //!< smoothing radius, in pixels, for cleaning up FG image.

    double maxVal, minVal;

    /*
     * General Parameters
     */
    int      imWidth;        //!< width of image.
    int      imHeight;       //!< height of image.
    size_t   numPixels;

    unsigned int numChannels; //!< Number of channels in image.

    bool    isDataInitialized;
    //!< After general parameters are set, data structures must be initialized.

    /*
     * Data Structures
     */
    vector<PixelModelGMG>       pixels; //!< Probabilistic background models for each pixel in image.
    int                         frameNum; //!< Frame number counter, used to count frames in training mode.
    Mat                         posteriorImage;  //!< Posterior probability image.
    Mat                         fgMaskImage;   //!< Foreground mask image.
};

}

#endif
