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

#include "precomp.hpp"
#include <float.h>

// to make sure we can use these short names
#undef K
#undef L
#undef T

// This is based on the "An Improved Adaptive Background Mixture Model for
// Real-time Tracking with Shadow Detection" by P. KaewTraKulPong and R. Bowden
// http://personal.ee.surrey.ac.uk/Personal/R.Bowden/publications/avbs01/avbs01.pdf
//
// The windowing method is used, but not the shadow detection. I make some of my
// own modifications which make more sense. There are some errors in some of their
// equations.
//

namespace cv
{

static const int defaultNMixtures = 5;
static const int defaultHistory = 200;
static const double defaultBackgroundRatio = 0.7;
static const double defaultVarThreshold = 2.5*2.5;
static const double defaultNoiseSigma = 30*0.5;
static const double defaultInitialWeight = 0.05;

class BackgroundSubtractorMOGImpl : public BackgroundSubtractorMOG
{
public:
    //! the default constructor
    BackgroundSubtractorMOGImpl()
    {
        frameSize = Size(0,0);
        frameType = 0;

        nframes = 0;
        nmixtures = defaultNMixtures;
        history = defaultHistory;
        varThreshold = defaultVarThreshold;
        backgroundRatio = defaultBackgroundRatio;
        noiseSigma = defaultNoiseSigma;
        name_ = "BackgroundSubtractor.MOG";
    }
    // the full constructor that takes the length of the history,
    // the number of gaussian mixtures, the background ratio parameter and the noise strength
    BackgroundSubtractorMOGImpl(int _history, int _nmixtures, double _backgroundRatio, double _noiseSigma=0)
    {
        frameSize = Size(0,0);
        frameType = 0;

        nframes = 0;
        nmixtures = std::min(_nmixtures > 0 ? _nmixtures : defaultNMixtures, 8);
        history = _history > 0 ? _history : defaultHistory;
        varThreshold = defaultVarThreshold;
        backgroundRatio = std::min(_backgroundRatio > 0 ? _backgroundRatio : 0.95, 1.);
        noiseSigma = _noiseSigma <= 0 ? defaultNoiseSigma : _noiseSigma;
    }

    //! the update operator
    virtual void apply(InputArray image, OutputArray fgmask, double learningRate=0);

    //! re-initiaization method
    virtual void initialize(Size _frameSize, int _frameType)
    {
        frameSize = _frameSize;
        frameType = _frameType;
        nframes = 0;

        int nchannels = CV_MAT_CN(frameType);
        CV_Assert( CV_MAT_DEPTH(frameType) == CV_8U );

        // for each gaussian mixture of each pixel bg model we store ...
        // the mixture sort key (w/sum_of_variances), the mixture weight (w),
        // the mean (nchannels values) and
        // the diagonal covariance matrix (another nchannels values)
        bgmodel.create( 1, frameSize.height*frameSize.width*nmixtures*(2 + 2*nchannels), CV_32F );
        bgmodel = Scalar::all(0);
    }

    virtual AlgorithmInfo* info() const { return 0; }

    virtual void getBackgroundImage(OutputArray) const
    {
        CV_Error( Error::StsNotImplemented, "" );
    }

    virtual int getHistory() const { return history; }
    virtual void setHistory(int _nframes) { history = _nframes; }

    virtual int getNMixtures() const { return nmixtures; }
    virtual void setNMixtures(int nmix) { nmixtures = nmix; }

    virtual double getBackgroundRatio() const { return backgroundRatio; }
    virtual void setBackgroundRatio(double _backgroundRatio) { backgroundRatio = _backgroundRatio; }

    virtual double getNoiseSigma() const { return noiseSigma; }
    virtual void setNoiseSigma(double _noiseSigma) { noiseSigma = _noiseSigma; }

    virtual void write(FileStorage& fs) const
    {
        fs << "name" << name_
           << "history" << history
           << "nmixtures" << nmixtures
           << "backgroundRatio" << backgroundRatio
           << "noiseSigma" << noiseSigma;
    }

    virtual void read(const FileNode& fn)
    {
        CV_Assert( (String)fn["name"] == name_ );
        history = (int)fn["history"];
        nmixtures = (int)fn["nmixtures"];
        backgroundRatio = (double)fn["backgroundRatio"];
        noiseSigma = (double)fn["noiseSigma"];
    }

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
    String name_;
};


template<typename VT> struct MixData
{
    float sortKey;
    float weight;
    VT mean;
    VT var;
};


static void process8uC1( const Mat& image, Mat& fgmask, double learningRate,
                         Mat& bgmodel, int nmixtures, double backgroundRatio,
                         double varThreshold, double noiseSigma )
{
    int x, y, k, k1, rows = image.rows, cols = image.cols;
    float alpha = (float)learningRate, T = (float)backgroundRatio, vT = (float)varThreshold;
    int K = nmixtures;
    MixData<float>* mptr = (MixData<float>*)bgmodel.data;

    const float w0 = (float)defaultInitialWeight;
    const float sk0 = (float)(w0/(defaultNoiseSigma*2));
    const float var0 = (float)(defaultNoiseSigma*defaultNoiseSigma*4);
    const float minVar = (float)(noiseSigma*noiseSigma);

    for( y = 0; y < rows; y++ )
    {
        const uchar* src = image.ptr<uchar>(y);
        uchar* dst = fgmask.ptr<uchar>(y);

        if( alpha > 0 )
        {
            for( x = 0; x < cols; x++, mptr += K )
            {
                float wsum = 0;
                float pix = src[x];
                int kHit = -1, kForeground = -1;

                for( k = 0; k < K; k++ )
                {
                    float w = mptr[k].weight;
                    wsum += w;
                    if( w < FLT_EPSILON )
                        break;
                    float mu = mptr[k].mean;
                    float var = mptr[k].var;
                    float diff = pix - mu;
                    float d2 = diff*diff;
                    if( d2 < vT*var )
                    {
                        wsum -= w;
                        float dw = alpha*(1.f - w);
                        mptr[k].weight = w + dw;
                        mptr[k].mean = mu + alpha*diff;
                        var = std::max(var + alpha*(d2 - var), minVar);
                        mptr[k].var = var;
                        mptr[k].sortKey = w/std::sqrt(var);

                        for( k1 = k-1; k1 >= 0; k1-- )
                        {
                            if( mptr[k1].sortKey >= mptr[k1+1].sortKey )
                                break;
                            std::swap( mptr[k1], mptr[k1+1] );
                        }

                        kHit = k1+1;
                        break;
                    }
                }

                if( kHit < 0 ) // no appropriate gaussian mixture found at all, remove the weakest mixture and create a new one
                {
                    kHit = k = std::min(k, K-1);
                    wsum += w0 - mptr[k].weight;
                    mptr[k].weight = w0;
                    mptr[k].mean = pix;
                    mptr[k].var = var0;
                    mptr[k].sortKey = sk0;
                }
                else
                    for( ; k < K; k++ )
                        wsum += mptr[k].weight;

                float wscale = 1.f/wsum;
                wsum = 0;
                for( k = 0; k < K; k++ )
                {
                    wsum += mptr[k].weight *= wscale;
                    mptr[k].sortKey *= wscale;
                    if( wsum > T && kForeground < 0 )
                        kForeground = k+1;
                }

                dst[x] = (uchar)(-(kHit >= kForeground));
            }
        }
        else
        {
            for( x = 0; x < cols; x++, mptr += K )
            {
                float pix = src[x];
                int kHit = -1, kForeground = -1;

                for( k = 0; k < K; k++ )
                {
                    if( mptr[k].weight < FLT_EPSILON )
                        break;
                    float mu = mptr[k].mean;
                    float var = mptr[k].var;
                    float diff = pix - mu;
                    float d2 = diff*diff;
                    if( d2 < vT*var )
                    {
                        kHit = k;
                        break;
                    }
                }

                if( kHit >= 0 )
                {
                    float wsum = 0;
                    for( k = 0; k < K; k++ )
                    {
                        wsum += mptr[k].weight;
                        if( wsum > T )
                        {
                            kForeground = k+1;
                            break;
                        }
                    }
                }

                dst[x] = (uchar)(kHit < 0 || kHit >= kForeground ? 255 : 0);
            }
        }
    }
}


static void process8uC3( const Mat& image, Mat& fgmask, double learningRate,
                         Mat& bgmodel, int nmixtures, double backgroundRatio,
                         double varThreshold, double noiseSigma )
{
    int x, y, k, k1, rows = image.rows, cols = image.cols;
    float alpha = (float)learningRate, T = (float)backgroundRatio, vT = (float)varThreshold;
    int K = nmixtures;

    const float w0 = (float)defaultInitialWeight;
    const float sk0 = (float)(w0/(defaultNoiseSigma*2*std::sqrt(3.)));
    const float var0 = (float)(defaultNoiseSigma*defaultNoiseSigma*4);
    const float minVar = (float)(noiseSigma*noiseSigma);
    MixData<Vec3f>* mptr = (MixData<Vec3f>*)bgmodel.data;

    for( y = 0; y < rows; y++ )
    {
        const uchar* src = image.ptr<uchar>(y);
        uchar* dst = fgmask.ptr<uchar>(y);

        if( alpha > 0 )
        {
            for( x = 0; x < cols; x++, mptr += K )
            {
                float wsum = 0;
                Vec3f pix(src[x*3], src[x*3+1], src[x*3+2]);
                int kHit = -1, kForeground = -1;

                for( k = 0; k < K; k++ )
                {
                    float w = mptr[k].weight;
                    wsum += w;
                    if( w < FLT_EPSILON )
                        break;
                    Vec3f mu = mptr[k].mean;
                    Vec3f var = mptr[k].var;
                    Vec3f diff = pix - mu;
                    float d2 = diff.dot(diff);
                    if( d2 < vT*(var[0] + var[1] + var[2]) )
                    {
                        wsum -= w;
                        float dw = alpha*(1.f - w);
                        mptr[k].weight = w + dw;
                        mptr[k].mean = mu + alpha*diff;
                        var = Vec3f(std::max(var[0] + alpha*(diff[0]*diff[0] - var[0]), minVar),
                                    std::max(var[1] + alpha*(diff[1]*diff[1] - var[1]), minVar),
                                    std::max(var[2] + alpha*(diff[2]*diff[2] - var[2]), minVar));
                        mptr[k].var = var;
                        mptr[k].sortKey = w/std::sqrt(var[0] + var[1] + var[2]);

                        for( k1 = k-1; k1 >= 0; k1-- )
                        {
                            if( mptr[k1].sortKey >= mptr[k1+1].sortKey )
                                break;
                            std::swap( mptr[k1], mptr[k1+1] );
                        }

                        kHit = k1+1;
                        break;
                    }
                }

                if( kHit < 0 ) // no appropriate gaussian mixture found at all, remove the weakest mixture and create a new one
                {
                    kHit = k = std::min(k, K-1);
                    wsum += w0 - mptr[k].weight;
                    mptr[k].weight = w0;
                    mptr[k].mean = pix;
                    mptr[k].var = Vec3f(var0, var0, var0);
                    mptr[k].sortKey = sk0;
                }
                else
                    for( ; k < K; k++ )
                        wsum += mptr[k].weight;

                float wscale = 1.f/wsum;
                wsum = 0;
                for( k = 0; k < K; k++ )
                {
                    wsum += mptr[k].weight *= wscale;
                    mptr[k].sortKey *= wscale;
                    if( wsum > T && kForeground < 0 )
                        kForeground = k+1;
                }

                dst[x] = (uchar)(-(kHit >= kForeground));
            }
        }
        else
        {
            for( x = 0; x < cols; x++, mptr += K )
            {
                Vec3f pix(src[x*3], src[x*3+1], src[x*3+2]);
                int kHit = -1, kForeground = -1;

                for( k = 0; k < K; k++ )
                {
                    if( mptr[k].weight < FLT_EPSILON )
                        break;
                    Vec3f mu = mptr[k].mean;
                    Vec3f var = mptr[k].var;
                    Vec3f diff = pix - mu;
                    float d2 = diff.dot(diff);
                    if( d2 < vT*(var[0] + var[1] + var[2]) )
                    {
                        kHit = k;
                        break;
                    }
                }

                if( kHit >= 0 )
                {
                    float wsum = 0;
                    for( k = 0; k < K; k++ )
                    {
                        wsum += mptr[k].weight;
                        if( wsum > T )
                        {
                            kForeground = k+1;
                            break;
                        }
                    }
                }

                dst[x] = (uchar)(kHit < 0 || kHit >= kForeground ? 255 : 0);
            }
        }
    }
}

void BackgroundSubtractorMOGImpl::apply(InputArray _image, OutputArray _fgmask, double learningRate)
{
    Mat image = _image.getMat();
    bool needToInitialize = nframes == 0 || learningRate >= 1 || image.size() != frameSize || image.type() != frameType;

    if( needToInitialize )
        initialize(image.size(), image.type());

    CV_Assert( image.depth() == CV_8U );
    _fgmask.create( image.size(), CV_8U );
    Mat fgmask = _fgmask.getMat();

    ++nframes;
    learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1./std::min( nframes, history );
    CV_Assert(learningRate >= 0);

    if( image.type() == CV_8UC1 )
        process8uC1( image, fgmask, learningRate, bgmodel, nmixtures, backgroundRatio, varThreshold, noiseSigma );
    else if( image.type() == CV_8UC3 )
        process8uC3( image, fgmask, learningRate, bgmodel, nmixtures, backgroundRatio, varThreshold, noiseSigma );
    else
        CV_Error( Error::StsUnsupportedFormat, "Only 1- and 3-channel 8-bit images are supported in BackgroundSubtractorMOG" );
}

Ptr<BackgroundSubtractorMOG> createBackgroundSubtractorMOG(int history, int nmixtures,
                                  double backgroundRatio, double noiseSigma)
{
    return makePtr<BackgroundSubtractorMOGImpl>(history, nmixtures, backgroundRatio, noiseSigma);
}

}

/* End of file. */
