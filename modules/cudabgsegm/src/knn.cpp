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

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)

Ptr<cuda::BackgroundSubtractorKNN> cv::cuda::createBackgroundSubtractorKNN(int, double, bool)
    {throw_no_cuda(); return Ptr<cuda::BackgroundSubtractorKNN>(); }

#else

namespace cv {
namespace cuda {
namespace device {
namespace knn {
void loadConstants(int nN, int nkNN, float Tb,
                   bool bShadowDetection, unsigned char nShadowDetection, float Tau);

void cvCheckPixelBackground_gpu(PtrStepSzb frame, PtrStepSzb fgmask, PtrStepSzb bgmodel,
                                PtrStepSzb include, cudaStream_t stream);

void cvUpdatePixelBackground_gpu(PtrStepSzb frame, PtrStepSzb bgmodel,
                                 PtrStepSzb aModelIndex, PtrStepSzb include,
                                 unsigned char *nCounter, unsigned char *nNextUpdate,
                                 cudaStream_t stream);

void getBackgroundImage_gpu(PtrStepSzb bgImg, PtrStepSzb bgmodel, cudaStream_t stream);
}
}
}
}

namespace {
// default parameters of gaussian background detection algorithm
static const int defaultHistory2 = 500; // Learning rate; alpha = 1/defaultHistory2
static const int defaultNsamples = 7; // number of samples saved in memory
static const double defaultDist2Threshold = 20.0*20.0;//threshold on distance from the sample

// additional parameters
static const unsigned char defaultnShadowDetection2 = (unsigned char)127; // value to use in the segmentation mask for shadows, set 0 not to do shadow detection
static const float defaultfTau = 0.5f; // Tau - shadow threshold, see the paper for explanation

class KNNImpl : public cuda::BackgroundSubtractorKNN {
  public:
    KNNImpl();
    KNNImpl(int _history,  double _dist2Threshold, bool _bShadowDetection=true);

    //! the update operator
    void apply(InputArray image, OutputArray fgmask, double learningRate=-1);
    void apply(InputArray image, OutputArray fgmask, double learningRate, Stream& stream);

    //! computes a background image which are the mean of all background gaussians
    virtual void getBackgroundImage(OutputArray backgroundImage) const;
    virtual void getBackgroundImage(OutputArray backgroundImage, Stream& stream) const;

    virtual int getHistory() const {
        return history;
    }
    virtual void setHistory(int _nframes) {
        history = _nframes;
    }

    virtual int getNSamples() const {
        return nN;
    }
    virtual void setNSamples(int _nN) {
        nN = _nN;    //needs reinitialization!
    }

    virtual int getkNNSamples() const {
        return nkNN;
    }
    virtual void setkNNSamples(int _nkNN) {
        nkNN = _nkNN;
    }

    virtual double getDist2Threshold() const {
        return fTb;
    }
    virtual void setDist2Threshold(double _dist2Threshold) {
        fTb = (float)_dist2Threshold;
    }

    virtual bool getDetectShadows() const {
        return bShadowDetection;
    }
    virtual void setDetectShadows(bool detectshadows) {
        bShadowDetection = detectshadows;
    }

    virtual int getShadowValue() const {
        return nShadowDetection;
    }
    virtual void setShadowValue(int value) {
        nShadowDetection = (uchar)value;
    }

    virtual double getShadowThreshold() const {
        return fTau;
    }
    virtual void setShadowThreshold(double value) {
        fTau = (float)value;
    }

  protected:
    void initialize(Size _frameSize, int _frameType);

    Size frameSize;
    int frameType;
    int nframes;
    /////////////////////////
    //very important parameters - things you will change
    ////////////////////////
    int history;
    //alpha=1/history - speed of update - if the time interval you want to average over is T
    //set alpha=1/history. It is also usefull at start to make T slowly increase
    //from 1 until the desired T
    float fTb;
    //Tb - threshold on the squared distance from the sample used to decide if it is well described
    //by the background model or not. A typical value could be 2 sigma
    //and that is Tb=2*2*10*10 =400; where we take typical pixel level sigma=10

    /////////////////////////
    //less important parameters - things you might change but be carefull
    ////////////////////////
    int nN; //total number of samples
    int nkNN; //number on NN for detcting background - default K=[0.1*nN]

    //shadow detection parameters
    bool bShadowDetection;//default 1 - do shadow detection
    unsigned char nShadowDetection;//do shadow detection - insert this value as the detection result - 127 default value
    float fTau;
    // Tau - shadow threshold. The shadow is detected if the pixel is darker
    //version of the background. Tau is a threshold on how much darker the shadow can be.
    //Tau= 0.5 means that if pixel is more than 2 times darker then it is not shadow
    //See: Prati,Mikic,Trivedi,Cucchiarra,"Detecting Moving Shadows...",IEEE PAMI,2003.

    //model data
    GpuMat bgmodel_d;
    GpuMat aModelIndex; // Index into the models
    unsigned char nCounter[3]; // Counter per model
    unsigned char nNextUpdate[3]; //Random update points per model

    GpuMat include_d;
};

//! re-initiaization method
void KNNImpl::initialize(Size _frameSize, int _frameType) {
    this->frameSize = _frameSize;
    this->frameType = _frameType;
    nframes = 0;

    CV_Assert( frameType == CV_8UC3 ); // Only works with CV_8UC3 as input at the moment
    CV_Assert( nN <= 255 ); // Can't index more than 255 positions within models

    // Reserve memory for the model
    // for each sample of 3 speed pixel models each pixel bg model we store ...
    // channels correspond to the colour values and a flag for each pixel 
    bgmodel_d.create(nN*3, frameSize.height*frameSize.width, CV_8UC4);

    // Model indices, channels correspond to Long, Short, Mid
    aModelIndex.create(frameSize.height, frameSize.width, CV_8UC3);
    aModelIndex.setTo(0);

    include_d.create(frameSize.height, frameSize.width, CV_8UC1);

    // Reset counters
    nCounter[0] = 0;
    nCounter[1] = 0;
    nCounter[2] = 0;

    nNextUpdate[0] = 0;
    nNextUpdate[1] = 0;
    nNextUpdate[2] = 0;

    // Load constants on the GPU
    cuda::device::knn::loadConstants(nN, nkNN, fTb, bShadowDetection, nShadowDetection, fTau);
}

//! the default constructor
KNNImpl::KNNImpl() {
    frameSize = Size(0,0);
    frameType = 0;
    nframes = 0;
    history = defaultHistory2;

    // Set parameters
    // N - the number of samples stored in memory per model
    nN = defaultNsamples;

    // kNN - k nearest neighbour - number on NN for detecting background - default K=[0.1*nN]
    nkNN=MAX(1,cvRound(0.1*nN*3+0.40));

    // Tb - Threshold Tb*kernelwidth
    fTb = defaultDist2Threshold;

    // Shadow detection
    bShadowDetection = 1;//turn on
    nShadowDetection =  defaultnShadowDetection2;
    fTau = defaultfTau;// Tau - shadow threshold
}

//! the full constructor that takes the length of the history,
// the number of gaussian mixtures, the background ratio parameter and the noise strength
KNNImpl::KNNImpl(int _history,  double _dist2Threshold, bool _bShadowDetection /*=true*/) {
    frameSize = Size(0,0);
    frameType = 0;
    nframes = 0;
    history = _history > 0 ? _history : defaultHistory2;

    //set parameters
    // N - the number of samples stored in memory per model
    nN = defaultNsamples;

    //kNN - k nearest neighbour - number on NN for detcting background - default K=[0.1*nN]
    nkNN=MAX(1,cvRound(0.1*nN*3+0.40));

    //Tb - Threshold Tb*kernelwidth
    fTb = _dist2Threshold > 0? _dist2Threshold : defaultDist2Threshold;

    // Shadow detection
    bShadowDetection = _bShadowDetection;
    nShadowDetection =  defaultnShadowDetection2;
    fTau = defaultfTau;
}

void KNNImpl::apply(InputArray _frame, OutputArray _fgmask, double learningRate) {
    apply(_frame, _fgmask, learningRate, Stream::Null());
}

void KNNImpl::apply(InputArray _frame, OutputArray _fgmask, double learningRate, Stream& stream) {
    //CV_INSTRUMENT_REGION()

    cuda::GpuMat frame = _frame.getGpuMat();

    bool needToInitialize = nframes == 0 || learningRate >= 1 ||
                            frame.size() != this->frameSize || frame.type() != frameType;

    if( needToInitialize ) {
        initialize(frame.size(), frame.type());
    }

    _fgmask.create(frame.size(), CV_8UC1);
    GpuMat fgmask = _fgmask.getGpuMat();

    ++nframes;
    learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1./std::min( 2*nframes, history );
    CV_Assert(learningRate >= 0);

    //recalculate update rates - in case alpha is changed
    // calculate update parameters (using alpha)
    int Kshort,Kmid,Klong;
    //approximate exponential learning curve
    Kshort=(int)(log(0.7)/log(1-learningRate))+1;//Kshort
    Kmid=(int)(log(0.4)/log(1-learningRate))-Kshort+1;//Kmid
    Klong=(int)(log(0.1)/log(1-learningRate))-Kshort-Kmid+1;//Klong

    //refresh rates
    int	nUpdateShort = (Kshort/nN)+1;
    int nUpdateMid = (Kmid/nN)+1;
    int nUpdateLong = (Klong/nN)+1;

    device::knn::cvCheckPixelBackground_gpu(frame, fgmask, bgmodel_d, include_d, 
                                            StreamAccessor::getStream(stream));

    device::knn::cvUpdatePixelBackground_gpu(frame, bgmodel_d, aModelIndex, include_d,
                 nCounter, nNextUpdate, StreamAccessor::getStream(stream));

    // Update counters for refresh rate
    // Long counter and next update
    if (++nCounter[0] > nNextUpdate[0]) {
        nCounter[0] = 0;
        nNextUpdate[0] = (unsigned char)( rand() % nUpdateLong );
    }
    // Mid counter and next update
    if (++nCounter[1] > nNextUpdate[1]) {
        nCounter[1] = 0;
        nNextUpdate[1] = (unsigned char)( rand() % nUpdateMid );
    }
    // Short counter and next update
    if (++nCounter[2] > nNextUpdate[2]) {
        nCounter[2] = 0;
        nNextUpdate[2] = (unsigned char)( rand() % nUpdateShort );
    }
}

void KNNImpl::getBackgroundImage(OutputArray backgroundImage) const {
    getBackgroundImage(backgroundImage, Stream::Null());
}

void KNNImpl::getBackgroundImage(OutputArray backgroundImage, Stream& stream) const {
    backgroundImage.create(frameSize.height, frameSize.width,CV_8UC3);
    GpuMat bgImg = backgroundImage.getGpuMat();
    cuda::device::knn::getBackgroundImage_gpu(bgImg, bgmodel_d, StreamAccessor::getStream(stream));
}
} // end of namespace {

Ptr<cuda::BackgroundSubtractorKNN> cv::cuda::createBackgroundSubtractorKNN(int history, double dist2Threshold, bool bShadowDetection) {
    return makePtr<KNNImpl>(history, dist2Threshold, bShadowDetection);
}

#endif
