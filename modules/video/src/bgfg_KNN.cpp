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
//#include <math.h>

#include "precomp.hpp"

namespace cv
{

/*!
 The class implements the following algorithm:
 "Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction"
 Z.Zivkovic, F. van der Heijden
 Pattern Recognition Letters, vol. 27, no. 7, pages 773-780, 2006
 http://www.zoranz.net/Publications/zivkovicPRL2006.pdf
*/

// default parameters of gaussian background detection algorithm
static const int defaultHistory2 = 500; // Learning rate; alpha = 1/defaultHistory2
static const int defaultNsamples = 7; // number of samples saved in memory
static const float defaultDist2Threshold = 20.0f*20.0f;//threshold on distance from the sample

// additional parameters
static const unsigned char defaultnShadowDetection2 = (unsigned char)127; // value to use in the segmentation mask for shadows, set 0 not to do shadow detection
static const float defaultfTau = 0.5f; // Tau - shadow threshold, see the paper for explanation

class BackgroundSubtractorKNNImpl : public BackgroundSubtractorKNN
{
public:
    //! the default constructor
    BackgroundSubtractorKNNImpl()
    {
    frameSize = Size(0,0);
    frameType = 0;
    nframes = 0;
    history = defaultHistory2;

    //set parameters
    // N - the number of samples stored in memory per model
    nN = defaultNsamples;

    //kNN - k nearest neighbour - number on NN for detecting background - default K=[0.1*nN]
    nkNN=MAX(1,cvRound(0.1*nN*3+0.40));

    //Tb - Threshold Tb*kernelwidth
    fTb = defaultDist2Threshold;

    // Shadow detection
    bShadowDetection = 1;//turn on
    nShadowDetection =  defaultnShadowDetection2;
    fTau = defaultfTau;// Tau - shadow threshold
    name_ = "BackgroundSubtractor.KNN";
    }
    //! the full constructor that takes the length of the history,
    // the number of gaussian mixtures, the background ratio parameter and the noise strength
    BackgroundSubtractorKNNImpl(int _history,  float _dist2Threshold, bool _bShadowDetection=true)
    {
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
    fTb = _dist2Threshold>0? _dist2Threshold : defaultDist2Threshold;

    bShadowDetection = _bShadowDetection;
    nShadowDetection =  defaultnShadowDetection2;
    fTau = defaultfTau;
    name_ = "BackgroundSubtractor.KNN";
    }
    //! the destructor
    ~BackgroundSubtractorKNNImpl() {}
    //! the update operator
    void apply(InputArray image, OutputArray fgmask, double learningRate=-1);

    //! computes a background image which are the mean of all background gaussians
    virtual void getBackgroundImage(OutputArray backgroundImage) const;

    //! re-initiaization method
    void initialize(Size _frameSize, int _frameType)
    {
    frameSize = _frameSize;
    frameType = _frameType;
    nframes = 0;

    int nchannels = CV_MAT_CN(frameType);
    CV_Assert( nchannels <= CV_CN_MAX );

    // Reserve memory for the model
    int size=frameSize.height*frameSize.width;
    // for each sample of 3 speed pixel models each pixel bg model we store ...
    // values + flag (nchannels+1 values)
    bgmodel.create( 1,(nN * 3) * (nchannels+1)* size,CV_8U);

    //index through the three circular lists
    aModelIndexShort.create(1,size,CV_8U);
    aModelIndexMid.create(1,size,CV_8U);
    aModelIndexLong.create(1,size,CV_8U);
    //when to update next
    nNextShortUpdate.create(1,size,CV_8U);
    nNextMidUpdate.create(1,size,CV_8U);
    nNextLongUpdate.create(1,size,CV_8U);

    //Reset counters
    nShortCounter = 0;
    nMidCounter = 0;
    nLongCounter = 0;

    aModelIndexShort = Scalar::all(0);//random? //((m_nN)*rand())/(RAND_MAX+1);//0...m_nN-1
    aModelIndexMid = Scalar::all(0);
    aModelIndexLong = Scalar::all(0);
    nNextShortUpdate = Scalar::all(0);
    nNextMidUpdate = Scalar::all(0);
    nNextLongUpdate = Scalar::all(0);
    }

    virtual int getHistory() const { return history; }
    virtual void setHistory(int _nframes) { history = _nframes; }

    virtual int getNSamples() const { return nN; }
    virtual void setNSamples(int _nN) { nN = _nN; }//needs reinitialization!

    virtual int getkNNSamples() const { return nkNN; }
    virtual void setkNNSamples(int _nkNN) { nkNN = _nkNN; }

    virtual double getDist2Threshold() const { return fTb; }
    virtual void setDist2Threshold(double _dist2Threshold) { fTb = (float)_dist2Threshold; }

    virtual bool getDetectShadows() const { return bShadowDetection; }
    virtual void setDetectShadows(bool detectshadows) { bShadowDetection = detectshadows; }

    virtual int getShadowValue() const { return nShadowDetection; }
    virtual void setShadowValue(int value) { nShadowDetection = (uchar)value; }

    virtual double getShadowThreshold() const { return fTau; }
    virtual void setShadowThreshold(double value) { fTau = (float)value; }

    virtual void write(FileStorage& fs) const
    {
        writeFormat(fs);
        fs << "name" << name_
        << "history" << history
        << "nsamples" << nN
        << "nKNN" << nkNN
        << "dist2Threshold" << fTb
        << "detectShadows" << (int)bShadowDetection
        << "shadowValue" << (int)nShadowDetection
        << "shadowThreshold" << fTau;
    }

    virtual void read(const FileNode& fn)
    {
        CV_Assert( (String)fn["name"] == name_ );
        history = (int)fn["history"];
        nN = (int)fn["nsamples"];
        nkNN = (int)fn["nKNN"];
        fTb = (float)fn["dist2Threshold"];
        bShadowDetection = (int)fn["detectShadows"] != 0;
        nShadowDetection = saturate_cast<uchar>((int)fn["shadowValue"]);
        fTau = (float)fn["shadowThreshold"];
    }

protected:
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
    int nN;//totlal number of samples
    int nkNN;//number on NN for detcting background - default K=[0.1*nN]

    //shadow detection parameters
    bool bShadowDetection;//default 1 - do shadow detection
    unsigned char nShadowDetection;//do shadow detection - insert this value as the detection result - 127 default value
    float fTau;
    // Tau - shadow threshold. The shadow is detected if the pixel is darker
    //version of the background. Tau is a threshold on how much darker the shadow can be.
    //Tau= 0.5 means that if pixel is more than 2 times darker then it is not shadow
    //See: Prati,Mikic,Trivedi,Cucchiarra,"Detecting Moving Shadows...",IEEE PAMI,2003.

    //model data
    int nLongCounter;//circular counter
    int nMidCounter;
    int nShortCounter;
    Mat bgmodel; // model data pixel values
    Mat aModelIndexShort;// index into the models
    Mat aModelIndexMid;
    Mat aModelIndexLong;
    Mat nNextShortUpdate;//random update points per model
    Mat nNextMidUpdate;
    Mat nNextLongUpdate;

    String name_;
};

//{ to do - paralelization ...
//struct KNNInvoker....
CV_INLINE void
        _cvUpdatePixelBackgroundNP(	long pixel,const uchar* data, int nchannels, int m_nN,
        uchar* m_aModel,
        uchar* m_nNextLongUpdate,
        uchar* m_nNextMidUpdate,
        uchar* m_nNextShortUpdate,
        uchar* m_aModelIndexLong,
        uchar* m_aModelIndexMid,
        uchar* m_aModelIndexShort,
        int m_nLongCounter,
        int m_nMidCounter,
        int m_nShortCounter,
        int m_nLongUpdate,
        int m_nMidUpdate,
        int m_nShortUpdate,
        uchar include
        )
{
    // hold the offset
    int ndata=1+nchannels;
    long offsetLong =  ndata * (pixel * m_nN * 3 + m_aModelIndexLong[pixel] + m_nN * 2);
    long offsetMid =   ndata * (pixel * m_nN * 3 + m_aModelIndexMid[pixel]  + m_nN * 1);
    long offsetShort = ndata * (pixel * m_nN * 3 + m_aModelIndexShort[pixel]);

    // Long update?
    if (m_nNextLongUpdate[pixel] == m_nLongCounter)
    {
        // add the oldest pixel from Mid to the list of values (for each color)
        memcpy(&m_aModel[offsetLong],&m_aModel[offsetMid],ndata*sizeof(unsigned char));
        // increase the index
        m_aModelIndexLong[pixel] = (m_aModelIndexLong[pixel] >= (m_nN-1)) ? 0 : (m_aModelIndexLong[pixel] + 1);
    };
    if (m_nLongCounter == (m_nLongUpdate-1))
    {
        //m_nNextLongUpdate[pixel] = (uchar)(((m_nLongUpdate)*(rand()-1))/RAND_MAX);//0,...m_nLongUpdate-1;
        m_nNextLongUpdate[pixel] = (uchar)( rand() % m_nLongUpdate );//0,...m_nLongUpdate-1;
    };

    // Mid update?
    if (m_nNextMidUpdate[pixel] == m_nMidCounter)
    {
        // add this pixel to the list of values (for each color)
        memcpy(&m_aModel[offsetMid],&m_aModel[offsetShort],ndata*sizeof(unsigned char));
        // increase the index
        m_aModelIndexMid[pixel] = (m_aModelIndexMid[pixel] >= (m_nN-1)) ? 0 : (m_aModelIndexMid[pixel] + 1);
    };
    if (m_nMidCounter == (m_nMidUpdate-1))
    {
        m_nNextMidUpdate[pixel] = (uchar)( rand() % m_nMidUpdate );
    };

    // Short update?
    if (m_nNextShortUpdate[pixel] == m_nShortCounter)
    {
        // add this pixel to the list of values (for each color)
        memcpy(&m_aModel[offsetShort],data,ndata*sizeof(unsigned char));
        //set the include flag
        m_aModel[offsetShort+nchannels]=include;
        // increase the index
        m_aModelIndexShort[pixel] = (m_aModelIndexShort[pixel] >= (m_nN-1)) ? 0 : (m_aModelIndexShort[pixel] + 1);
    };
    if (m_nShortCounter == (m_nShortUpdate-1))
    {
        m_nNextShortUpdate[pixel] = (uchar)( rand() % m_nShortUpdate );
    };
};

CV_INLINE int
        _cvCheckPixelBackgroundNP(long pixel,
        const uchar* data, int nchannels,
        int m_nN,
        uchar* m_aModel,
        float m_fTb,
        int m_nkNN,
        float tau,
        int m_nShadowDetection,
        uchar& include)
{
    int Pbf = 0; // the total probability that this pixel is background
    int Pb = 0; //background model probability
    float dData[CV_CN_MAX];

    //uchar& include=data[nchannels];
    include=0;//do we include this pixel into background model?

    int ndata=nchannels+1;
    long posPixel = pixel * ndata * m_nN * 3;
//	float k;
    // now increase the probability for each pixel
    for (int n = 0; n < m_nN*3; n++)
    {
        uchar* mean_m = &m_aModel[posPixel + n*ndata];

        //calculate difference and distance
        float dist2;

        if( nchannels == 3 )
        {
            dData[0] = (float)mean_m[0] - data[0];
            dData[1] = (float)mean_m[1] - data[1];
            dData[2] = (float)mean_m[2] - data[2];
            dist2 = dData[0]*dData[0] + dData[1]*dData[1] + dData[2]*dData[2];
        }
        else
        {
            dist2 = 0.f;
            for( int c = 0; c < nchannels; c++ )
            {
                dData[c] = (float)mean_m[c] - data[c];
                dist2 += dData[c]*dData[c];
            }
        }

        if (dist2<m_fTb)
        {
            Pbf++;//all
            //background only
            //if(m_aModel[subPosPixel + nchannels])//indicator
            if(mean_m[nchannels])//indicator
            {
                Pb++;
                if (Pb >= m_nkNN)//Tb
                {
                    include=1;//include
                    return 1;//background ->exit
                };
            }
        };
    };

    //include?
    if (Pbf>=m_nkNN)//m_nTbf)
    {
        include=1;
    }

    int Ps = 0; // the total probability that this pixel is background shadow
    // Detected as moving object, perform shadow detection
    if (m_nShadowDetection)
    {
        for (int n = 0; n < m_nN*3; n++)
        {
            //long subPosPixel = posPixel + n*ndata;
            uchar* mean_m = &m_aModel[posPixel + n*ndata];

            if(mean_m[nchannels])//check only background
            {
                float numerator = 0.0f;
                float denominator = 0.0f;
                for( int c = 0; c < nchannels; c++ )
                {
                    numerator   += (float)data[c] * mean_m[c];
                    denominator += (float)mean_m[c] * mean_m[c];
                }

                // no division by zero allowed
                if( denominator == 0 )
                    return 0;

                // if tau < a < 1 then also check the color distortion
                if( numerator <= denominator && numerator >= tau*denominator )
                {
                    float a = numerator / denominator;
                    float dist2a = 0.0f;

                    for( int c = 0; c < nchannels; c++ )
                    {
                        float dD= a*mean_m[c] - data[c];
                        dist2a += dD*dD;
                    }

                    if (dist2a<m_fTb*a*a)
                    {
                        Ps++;
                        if (Ps >= m_nkNN)//shadow
                            return 2;
                    };
                };
            };
        };
    }
    return 0;
};

CV_INLINE void
        icvUpdatePixelBackgroundNP(const Mat& _src, Mat& _dst,
        Mat& _bgmodel,
        Mat& _nNextLongUpdate,
        Mat& _nNextMidUpdate,
        Mat& _nNextShortUpdate,
        Mat& _aModelIndexLong,
        Mat& _aModelIndexMid,
        Mat& _aModelIndexShort,
        int& _nLongCounter,
        int& _nMidCounter,
        int& _nShortCounter,
        int _nN,
        float _fAlphaT,
        float _fTb,
        int _nkNN,
        float _fTau,
        int _bShadowDetection,
        uchar nShadowDetection
        )
{
    int nchannels = CV_MAT_CN(_src.type());

    //model
    uchar* m_aModel=_bgmodel.ptr(0);
    uchar* m_nNextLongUpdate=_nNextLongUpdate.ptr(0);
    uchar* m_nNextMidUpdate=_nNextMidUpdate.ptr(0);
    uchar* m_nNextShortUpdate=_nNextShortUpdate.ptr(0);
    uchar* m_aModelIndexLong=_aModelIndexLong.ptr(0);
    uchar* m_aModelIndexMid=_aModelIndexMid.ptr(0);
    uchar* m_aModelIndexShort=_aModelIndexShort.ptr(0);

    //some constants
    int m_nN=_nN;
    float m_fAlphaT=_fAlphaT;
    float m_fTb=_fTb;//Tb - threshold on the distance
    float m_fTau=_fTau;
    int m_nkNN=_nkNN;
    int m_bShadowDetection=_bShadowDetection;

    //recalculate update rates - in case alpha is changed
    // calculate update parameters (using alpha)
    int Kshort,Kmid,Klong;
    //approximate exponential learning curve
    Kshort=(int)(log(0.7)/log(1-m_fAlphaT))+1;//Kshort
    Kmid=(int)(log(0.4)/log(1-m_fAlphaT))-Kshort+1;//Kmid
    Klong=(int)(log(0.1)/log(1-m_fAlphaT))-Kshort-Kmid+1;//Klong

    //refresh rates
    int	m_nShortUpdate = (Kshort/m_nN)+1;
    int m_nMidUpdate = (Kmid/m_nN)+1;
    int m_nLongUpdate = (Klong/m_nN)+1;

    //int	m_nShortUpdate = MAX((Kshort/m_nN),m_nN);
    //int m_nMidUpdate = MAX((Kmid/m_nN),m_nN);
    //int m_nLongUpdate = MAX((Klong/m_nN),m_nN);

    //update counters for the refresh rate
    int m_nLongCounter=_nLongCounter;
    int m_nMidCounter=_nMidCounter;
    int m_nShortCounter=_nShortCounter;

    _nShortCounter++;//0,1,...,m_nShortUpdate-1
    _nMidCounter++;
    _nLongCounter++;
    if (_nShortCounter >= m_nShortUpdate) _nShortCounter = 0;
    if (_nMidCounter >= m_nMidUpdate) _nMidCounter = 0;
    if (_nLongCounter >= m_nLongUpdate) _nLongCounter = 0;

    //go through the image
    long i = 0;
    for (long y = 0; y < _src.rows; y++)
    {
        for (long x = 0; x < _src.cols; x++)
        {
            const uchar* data = _src.ptr((int)y, (int)x);

            //update model+ background subtract
            uchar include=0;
            int result= _cvCheckPixelBackgroundNP(i, data, nchannels,
                    m_nN, m_aModel, m_fTb,m_nkNN, m_fTau,m_bShadowDetection,include);

            _cvUpdatePixelBackgroundNP(i,data,nchannels,
                    m_nN, m_aModel,
                    m_nNextLongUpdate,
                    m_nNextMidUpdate,
                    m_nNextShortUpdate,
                    m_aModelIndexLong,
                    m_aModelIndexMid,
                    m_aModelIndexShort,
                    m_nLongCounter,
                    m_nMidCounter,
                    m_nShortCounter,
                    m_nLongUpdate,
                    m_nMidUpdate,
                    m_nShortUpdate,
                    include
                    );
            switch (result)
            {
                case 0:
                    //foreground
                    *_dst.ptr((int)y, (int)x) = 255;
                    break;
                case 1:
                    //background
                    *_dst.ptr((int)y, (int)x) = 0;
                    break;
                case 2:
                    //shadow
                    *_dst.ptr((int)y, (int)x) = nShadowDetection;
                    break;
            }
            i++;
        }
    }
};



void BackgroundSubtractorKNNImpl::apply(InputArray _image, OutputArray _fgmask, double learningRate)
{
    Mat image = _image.getMat();
    bool needToInitialize = nframes == 0 || learningRate >= 1 || image.size() != frameSize || image.type() != frameType;

    if( needToInitialize )
        initialize(image.size(), image.type());

    _fgmask.create( image.size(), CV_8U );
    Mat fgmask = _fgmask.getMat();

    ++nframes;
    learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1./std::min( 2*nframes, history );
    CV_Assert(learningRate >= 0);

    //parallel_for_(Range(0, image.rows),
    //              KNNInvoker(image, fgmask,
    icvUpdatePixelBackgroundNP(image, fgmask,
            bgmodel,
            nNextLongUpdate,
            nNextMidUpdate,
            nNextShortUpdate,
            aModelIndexLong,
            aModelIndexMid,
            aModelIndexShort,
            nLongCounter,
            nMidCounter,
            nShortCounter,
            nN,
            (float)learningRate,
            fTb,
            nkNN,
            fTau,
            bShadowDetection,
            nShadowDetection
            );
}

void BackgroundSubtractorKNNImpl::getBackgroundImage(OutputArray backgroundImage) const
{
    int nchannels = CV_MAT_CN(frameType);
    //CV_Assert( nchannels == 3 );
    Mat meanBackground(frameSize, CV_8UC3, Scalar::all(0));

    int ndata=nchannels+1;
    int modelstep=(ndata * nN * 3);

    const uchar* pbgmodel=bgmodel.ptr(0);
    for(int row=0; row<meanBackground.rows; row++)
    {
        for(int col=0; col<meanBackground.cols; col++)
        {
            for (int n = 0; n < nN*3; n++)
            {
                const uchar* mean_m = &pbgmodel[n*ndata];
                if (mean_m[nchannels])
                {
                    meanBackground.at<Vec3b>(row, col) = Vec3b(mean_m);
                    break;
                }
            }
            pbgmodel=pbgmodel+modelstep;
        }
    }

    switch(CV_MAT_CN(frameType))
    {
        case 1:
        {
            std::vector<Mat> channels;
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
            CV_Error(Error::StsUnsupportedFormat, "");
    }
}


Ptr<BackgroundSubtractorKNN> createBackgroundSubtractorKNN(int _history, double _threshold2,
                                                           bool _bShadowDetection)
{
    return makePtr<BackgroundSubtractorKNNImpl>(_history, (float)_threshold2, _bShadowDetection);
}

}

/* End of file. */
