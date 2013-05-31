/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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
//"Improved adaptive Gausian mixture model for background subtraction"
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

namespace cv
{

/*
 Interface of Gaussian mixture algorithm from:

 "Improved adaptive Gausian mixture model for background subtraction"
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

struct GaussBGStatModel2Params
{
    //image info
    int nWidth;
    int nHeight;
    int nND;//number of data dimensions (image channels)

    bool bPostFiltering;//defult 1 - do postfiltering - will make shadow detection results also give value 255
    double  minArea; // for postfiltering

    bool bInit;//default 1, faster updates at start

    /////////////////////////
    //very important parameters - things you will change
    ////////////////////////
    float fAlphaT;
    //alpha - speed of update - if the time interval you want to average over is T
    //set alpha=1/T. It is also usefull at start to make T slowly increase
    //from 1 until the desired T
    float fTb;
    //Tb - threshold on the squared Mahalan. dist. to decide if it is well described
    //by the background model or not. Related to Cthr from the paper.
    //This does not influence the update of the background. A typical value could be 4 sigma
    //and that is Tb=4*4=16;

    /////////////////////////
    //less important parameters - things you might change but be carefull
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
    //See: Prati,Mikic,Trivedi,Cucchiarra,"Detecting Moving Shadows...",IEEE PAMI,2003.
};

struct GMM
{
    float weight;
    float variance;
};

// shadow detection performed per pixel
// should work for rgb data, could be usefull for gray scale and depth data as well
// See: Prati,Mikic,Trivedi,Cucchiarra,"Detecting Moving Shadows...",IEEE PAMI,2003.
static CV_INLINE bool
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

struct MOG2Invoker : ParallelLoopBody
{
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

        cvtfunc = src->depth() != CV_32F ? getConvertFunc(src->depth(), CV_32F) : 0;
    }

    void operator()(const Range& range) const
    {
        int y0 = range.start, y1 = range.end;
        int ncols = src->cols, nchannels = src->channels();
        AutoBuffer<float> buf(src->cols*nchannels);
        float alpha1 = 1.f - alphaT;
        float dData[CV_CN_MAX];

        for( int y = y0; y < y1; y++ )
        {
            const float* data = buf;
            if( cvtfunc )
                cvtfunc( src->ptr(y), src->step, 0, 0, (uchar*)data, 0, Size(ncols*nchannels, 1), 0);
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
                int nmodes = modesUsed[x], nNewModes = nmodes;//current number of modes in GMM
                float totalWeight = 0.f;

                float* mean_m = mean;

                //////
                //go through all modes
                for( int mode = 0; mode < nmodes; mode++, mean_m += nchannels )
                {
                    float weight = alpha1*gmm[mode].weight + prune;//need only weight if fit is found

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

                    gmm[mode].weight = weight;//update weight by the calculated value
                    totalWeight += weight;
                }
                //go through all modes
                //////

                //renormalize weights
                totalWeight = 1.f/totalWeight;
                for( int mode = 0; mode < nmodes; mode++ )
                {
                    gmm[mode].weight *= totalWeight;
                }

                nmodes = nNewModes;

                //make new mode if needed and exit
                if( !fitsPDF )
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

    BinaryFunc cvtfunc;
};

BackgroundSubtractorMOG2::BackgroundSubtractorMOG2()
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
}

BackgroundSubtractorMOG2::BackgroundSubtractorMOG2(int _history,  float _varThreshold, bool _bShadowDetection)
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
}

BackgroundSubtractorMOG2::~BackgroundSubtractorMOG2()
{
}


void BackgroundSubtractorMOG2::initialize(Size _frameSize, int _frameType)
{
    frameSize = _frameSize;
    frameType = _frameType;
    nframes = 0;

    int nchannels = CV_MAT_CN(frameType);
    CV_Assert( nchannels <= CV_CN_MAX );

    // for each gaussian mixture of each pixel bg model we store ...
    // the mixture weight (w),
    // the mean (nchannels values) and
    // the covariance
    bgmodel.create( 1, frameSize.height*frameSize.width*nmixtures*(2 + nchannels), CV_32F );
    //make the array for keeping track of the used modes per pixel - all zeros at start
    bgmodelUsedModes.create(frameSize,CV_8U);
    bgmodelUsedModes = Scalar::all(0);
}

void BackgroundSubtractorMOG2::operator()(InputArray _image, OutputArray _fgmask, double learningRate)
{
    Mat image = _image.getMat();
    bool needToInitialize = nframes == 0 || learningRate >= 1 || image.size() != frameSize || image.type() != frameType;

    if( needToInitialize )
        initialize(image.size(), image.type());

    _fgmask.create( image.size(), CV_8U );
    Mat fgmask = _fgmask.getMat();

    ++nframes;
    learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1./min( 2*nframes, history );
    CV_Assert(learningRate >= 0);

    parallel_for_(Range(0, image.rows),
                  MOG2Invoker(image, fgmask,
                              (GMM*)bgmodel.data,
                              (float*)(bgmodel.data + sizeof(GMM)*nmixtures*image.rows*image.cols),
                              bgmodelUsedModes.data, nmixtures, (float)learningRate,
                              (float)varThreshold,
                              backgroundRatio, varThresholdGen,
                              fVarInit, fVarMin, fVarMax, float(-learningRate*fCT), fTau,
                              bShadowDetection, nShadowDetection));
}

void BackgroundSubtractorMOG2::getBackgroundImage(OutputArray backgroundImage) const
{
    int nchannels = CV_MAT_CN(frameType);
    CV_Assert( nchannels == 3 );
    Mat meanBackground(frameSize, CV_8UC3, Scalar::all(0));

    int firstGaussianIdx = 0;
    const GMM* gmm = (GMM*)bgmodel.data;
    const Vec3f* mean = reinterpret_cast<const Vec3f*>(gmm + frameSize.width*frameSize.height*nmixtures);
    for(int row=0; row<meanBackground.rows; row++)
    {
        for(int col=0; col<meanBackground.cols; col++)
        {
            int nmodes = bgmodelUsedModes.at<uchar>(row, col);
            Vec3f meanVal;
            float totalWeight = 0.f;
            for(int gaussianIdx = firstGaussianIdx; gaussianIdx < firstGaussianIdx + nmodes; gaussianIdx++)
            {
                GMM gaussian = gmm[gaussianIdx];
                meanVal += gaussian.weight * mean[gaussianIdx];
                totalWeight += gaussian.weight;

                if(totalWeight > backgroundRatio)
                    break;
            }

            meanVal *= (1.f / totalWeight);
            meanBackground.at<Vec3b>(row, col) = Vec3b(meanVal);
            firstGaussianIdx += nmixtures;
        }
    }

    switch(CV_MAT_CN(frameType))
    {
    case 1:
    {
        vector<Mat> channels;
        split(meanBackground, channels);
        channels[0].copyTo(backgroundImage);
        break;
    }

    case 3:
    {
        meanBackground.copyTo(backgroundImage);
        break;
    }

    default:
        CV_Error(CV_StsUnsupportedFormat, "");
    }
}

}

/* End of file. */
