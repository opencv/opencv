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
#include "opencl_kernels_video.hpp"

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

class BackgroundSubtractorKNNImpl CV_FINAL : public BackgroundSubtractorKNN
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
    nLongCounter = 0;
    nMidCounter = 0;
    nShortCounter = 0;
#ifdef HAVE_OPENCL
    opencl_ON = true;
#endif
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
    //kNN - k nearest neighbour - number on NN for detecting background - default K=[0.1*nN]
    nkNN=MAX(1,cvRound(0.1*nN*3+0.40));

    //Tb - Threshold Tb*kernelwidth
    fTb = _dist2Threshold>0? _dist2Threshold : defaultDist2Threshold;

    bShadowDetection = _bShadowDetection;
    nShadowDetection =  defaultnShadowDetection2;
    fTau = defaultfTau;
    name_ = "BackgroundSubtractor.KNN";
    nLongCounter = 0;
    nMidCounter = 0;
    nShortCounter = 0;
#ifdef HAVE_OPENCL
    opencl_ON = true;
#endif
    }
    //! the destructor
    ~BackgroundSubtractorKNNImpl() CV_OVERRIDE {}
    //! the update operator
    void apply(InputArray image, OutputArray fgmask, double learningRate) CV_OVERRIDE;

    void apply(InputArray image,OutputArray fgmask, InputArray knownForegroundMask,double learningRate) CV_OVERRIDE;

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

        // Reserve memory for the model
        int size=frameSize.height*frameSize.width;
        //Reset counters
        nShortCounter = 0;
        nMidCounter = 0;
        nLongCounter = 0;

#ifdef HAVE_OPENCL
        if (ocl::isOpenCLActivated() && opencl_ON)
        {
            create_ocl_apply_kernel();

            kernel_getBg.create("getBackgroundImage2_kernel", ocl::video::bgfg_knn_oclsrc, format( "-D CN=%d -D NSAMPLES=%d", nchannels, nN));

            if (kernel_apply.empty() || kernel_getBg.empty())
                opencl_ON = false;
        }
        else opencl_ON = false;

        if (opencl_ON)
        {
            u_flag.create(frameSize.height * nN * 3, frameSize.width, CV_8UC1);
            u_flag.setTo(Scalar::all(0));

            if (nchannels==3)
                nchannels=4;
            u_sample.create(frameSize.height * nN * 3, frameSize.width, CV_32FC(nchannels));
            u_sample.setTo(Scalar::all(0));

            u_aModelIndexShort.create(frameSize.height, frameSize.width, CV_8UC1);
            u_aModelIndexShort.setTo(Scalar::all(0));
            u_aModelIndexMid.create(frameSize.height, frameSize.width, CV_8UC1);
            u_aModelIndexMid.setTo(Scalar::all(0));
            u_aModelIndexLong.create(frameSize.height, frameSize.width, CV_8UC1);
            u_aModelIndexLong.setTo(Scalar::all(0));

            u_nNextShortUpdate.create(frameSize.height, frameSize.width, CV_8UC1);
            u_nNextShortUpdate.setTo(Scalar::all(0));
            u_nNextMidUpdate.create(frameSize.height, frameSize.width, CV_8UC1);
            u_nNextMidUpdate.setTo(Scalar::all(0));
            u_nNextLongUpdate.create(frameSize.height, frameSize.width, CV_8UC1);
            u_nNextLongUpdate.setTo(Scalar::all(0));
        }
        else
#endif
        {
            // for each sample of 3 speed pixel models each pixel bg model we store ...
            // values + flag (nchannels+1 values)
            bgmodel.create( 1,(nN * 3) * (nchannels+1)* size,CV_8U);
            bgmodel = Scalar::all(0);

            //index through the three circular lists
            aModelIndexShort.create(1,size,CV_8U);
            aModelIndexMid.create(1,size,CV_8U);
            aModelIndexLong.create(1,size,CV_8U);
            //when to update next
            nNextShortUpdate.create(1,size,CV_8U);
            nNextMidUpdate.create(1,size,CV_8U);
            nNextLongUpdate.create(1,size,CV_8U);

            aModelIndexShort = Scalar::all(0);//random? //((m_nN)*rand())/(RAND_MAX+1);//0...m_nN-1
            aModelIndexMid = Scalar::all(0);
            aModelIndexLong = Scalar::all(0);
            nNextShortUpdate = Scalar::all(0);
            nNextMidUpdate = Scalar::all(0);
            nNextLongUpdate = Scalar::all(0);
        }
    }

    virtual String getDefaultName() const CV_OVERRIDE { return "BackgroundSubtractor_KNN"; }

    virtual int getHistory() const CV_OVERRIDE { return history; }
    virtual void setHistory(int _nframes) CV_OVERRIDE { history = _nframes; }

    virtual int getNSamples() const CV_OVERRIDE { return nN; }
    virtual void setNSamples(int _nN) CV_OVERRIDE { nN = _nN; }//needs reinitialization!

    virtual int getkNNSamples() const CV_OVERRIDE { return nkNN; }
    virtual void setkNNSamples(int _nkNN) CV_OVERRIDE { nkNN = _nkNN; }

    virtual double getDist2Threshold() const CV_OVERRIDE { return fTb; }
    virtual void setDist2Threshold(double _dist2Threshold) CV_OVERRIDE { fTb = (float)_dist2Threshold; }

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
        << "nsamples" << nN
        << "nKNN" << nkNN
        << "dist2Threshold" << fTb
        << "detectShadows" << (int)bShadowDetection
        << "shadowValue" << (int)nShadowDetection
        << "shadowThreshold" << fTau;
    }

    virtual void read(const FileNode& fn) CV_OVERRIDE
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
    //set alpha=1/history. It is also useful at start to make T slowly increase
    //from 1 until the desired T
    float fTb;
    //Tb - threshold on the squared distance from the sample used to decide if it is well described
    //by the background model or not. A typical value could be 2 sigma
    //and that is Tb=2*2*10*10 =400; where we take typical pixel level sigma=10

    /////////////////////////
    //less important parameters - things you might change but be careful
    ////////////////////////
    int nN;//totlal number of samples
    int nkNN;//number on NN for detecting background - default K=[0.1*nN]

    //shadow detection parameters
    bool bShadowDetection;//default 1 - do shadow detection
    unsigned char nShadowDetection;//do shadow detection - insert this value as the detection result - 127 default value
    float fTau;
    // Tau - shadow threshold. The shadow is detected if the pixel is darker
    //version of the background. Tau is a threshold on how much darker the shadow can be.
    //Tau= 0.5 means that if pixel is more than 2 times darker then it is not shadow
    //See: Prati,Mikic,Trivedi,Cucchiara,"Detecting Moving Shadows...",IEEE PAMI,2003.

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

#ifdef HAVE_OPENCL
    mutable bool opencl_ON;

    UMat u_flag;
    UMat u_sample;
    UMat u_aModelIndexShort;
    UMat u_aModelIndexMid;
    UMat u_aModelIndexLong;
    UMat u_nNextShortUpdate;
    UMat u_nNextMidUpdate;
    UMat u_nNextLongUpdate;

    mutable ocl::Kernel kernel_apply;
    mutable ocl::Kernel kernel_getBg;
#endif

    String name_;

#ifdef HAVE_OPENCL
    bool ocl_getBackgroundImage(OutputArray backgroundImage) const;
    bool ocl_apply(InputArray _image, OutputArray _fgmask, double learningRate=-1);
    void create_ocl_apply_kernel();
#endif
};

CV_INLINE void
        _cvUpdatePixelBackgroundNP(int x_idx, const uchar* data, int nchannels, int m_nN,
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
        uchar include
        )
{
    // hold the offset
    int ndata=1+nchannels;
    long offsetLong =  ndata * (m_aModelIndexLong[x_idx] + m_nN * 2);
    long offsetMid =   ndata * (m_aModelIndexMid[x_idx]  + m_nN * 1);
    long offsetShort = ndata * (m_aModelIndexShort[x_idx]);

    // Long update?
    if (m_nNextLongUpdate[x_idx] == m_nLongCounter)
    {
        // add the oldest pixel from Mid to the list of values (for each color)
        memcpy(&m_aModel[offsetLong],&m_aModel[offsetMid],ndata*sizeof(unsigned char));
        // increase the index
        m_aModelIndexLong[x_idx] = (m_aModelIndexLong[x_idx] >= (m_nN-1)) ? 0 : (m_aModelIndexLong[x_idx] + 1);
    };

    // Mid update?
    if (m_nNextMidUpdate[x_idx] == m_nMidCounter)
    {
        // add this pixel to the list of values (for each color)
        memcpy(&m_aModel[offsetMid],&m_aModel[offsetShort],ndata*sizeof(unsigned char));
        // increase the index
        m_aModelIndexMid[x_idx] = (m_aModelIndexMid[x_idx] >= (m_nN-1)) ? 0 : (m_aModelIndexMid[x_idx] + 1);
    };

    // Short update?
    if (m_nNextShortUpdate[x_idx] == m_nShortCounter)
    {
        // add this pixel to the list of values (for each color)
        memcpy(&m_aModel[offsetShort],data,nchannels*sizeof(unsigned char));
        //set the include flag
        m_aModel[offsetShort+nchannels]=include;
        // increase the index
        m_aModelIndexShort[x_idx] = (m_aModelIndexShort[x_idx] >= (m_nN-1)) ? 0 : (m_aModelIndexShort[x_idx] + 1);
    };
}

CV_INLINE int
        _cvCheckPixelBackgroundNP(const uchar* data, int nchannels,
        int m_nN,
        uchar* m_aModel,
        float m_fTb,
        int m_nkNN,
        float tau,
        bool m_bShadowDetection,
        uchar& include)
{
    int Pbf = 0; // the total probability that this pixel is background
    int Pb = 0; //background model probability
    float dData[CV_CN_MAX];

    //uchar& include=data[nchannels];
    include=0;//do we include this pixel into background model?

    int ndata=nchannels+1;
    // now increase the probability for each pixel
    for (int n = 0; n < m_nN*3; n++)
    {
        uchar* mean_m = &m_aModel[n*ndata];

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
    if (m_bShadowDetection)
    {
        for (int n = 0; n < m_nN*3; n++)
        {
            //long subPosPixel = posPixel + n*ndata;
            uchar* mean_m = &m_aModel[n*ndata];

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
}

class KNNInvoker : public ParallelLoopBody
{
public:
    KNNInvoker(const Mat& _src, Mat& _dst,
               uchar* _bgmodel,
               uchar* _nNextLongUpdate,
               uchar* _nNextMidUpdate,
               uchar* _nNextShortUpdate,
               uchar* _aModelIndexLong,
               uchar* _aModelIndexMid,
               uchar* _aModelIndexShort,
               int _nLongCounter,
               int _nMidCounter,
               int _nShortCounter,
               int _nN,
               float _fTb,
               int _nkNN,
               float _fTau,
               bool _bShadowDetection,
               uchar _nShadowDetection,
               const Mat *_knownForegroundMask)
    {
        src = &_src;
        dst = &_dst;
        m_aModel0 = _bgmodel;
        m_nNextLongUpdate0 = _nNextLongUpdate;
        m_nNextMidUpdate0 = _nNextMidUpdate;
        m_nNextShortUpdate0 = _nNextShortUpdate;
        m_aModelIndexLong0 = _aModelIndexLong;
        m_aModelIndexMid0 = _aModelIndexMid;
        m_aModelIndexShort0 = _aModelIndexShort;
        m_nLongCounter = _nLongCounter;
        m_nMidCounter = _nMidCounter;
        m_nShortCounter = _nShortCounter;
        m_nN = _nN;
        m_fTb = _fTb;
        m_fTau = _fTau;
        m_nkNN = _nkNN;
        m_bShadowDetection = _bShadowDetection;
        m_nShadowDetection = _nShadowDetection;
        knownForegroundMask = _knownForegroundMask;
    }

    void operator()(const Range& range) const CV_OVERRIDE
    {
        int y0 = range.start, y1 = range.end;
        int ncols = src->cols, nchannels = src->channels();
        int ndata=nchannels+1;

        for ( int y = y0; y < y1; y++ )
        {
            const uchar* data = src->ptr(y);
            uchar* m_aModel = m_aModel0 + ncols*m_nN*3*ndata*y;
            uchar* m_nNextLongUpdate = m_nNextLongUpdate0 + ncols*y;
            uchar* m_nNextMidUpdate = m_nNextMidUpdate0 + ncols*y;
            uchar* m_nNextShortUpdate = m_nNextShortUpdate0 + ncols*y;
            uchar* m_aModelIndexLong = m_aModelIndexLong0 + ncols*y;
            uchar* m_aModelIndexMid = m_aModelIndexMid0 + ncols*y;
            uchar* m_aModelIndexShort = m_aModelIndexShort0 + ncols*y;
            uchar* mask = dst->ptr(y);

            for ( int x = 0; x < ncols; x++ )
            {

                //update model+ background subtract
                uchar include=0;
                int result= _cvCheckPixelBackgroundNP(data, nchannels,
                        m_nN, m_aModel, m_fTb,m_nkNN, m_fTau,m_bShadowDetection,include);

                _cvUpdatePixelBackgroundNP(x,data,nchannels,
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
                        include
                        );
                // Check that foreground mask exists
                if (knownForegroundMask && !knownForegroundMask->empty()) {
                    // If input mask states pixel is foreground
                    if (knownForegroundMask->at<uchar>(y, x) > 0) {
                        mask[x] = 255; // ensure output mask marks this pixel as FG
                        data += nchannels;
                        m_aModel += m_nN*3*ndata;
                        continue;
                    }
                }
                switch (result)
                {
                    case 0:
                        //foreground
                        mask[x] = 255;
                        break;
                    case 1:
                        //background
                        mask[x] = 0;
                        break;
                    case 2:
                        //shadow
                        mask[x] = m_nShadowDetection;
                        break;
                }
                data += nchannels;
                m_aModel += m_nN*3*ndata;
            }
        }
    }

    const Mat* src;
    Mat* dst;
    uchar* m_aModel0;
    uchar* m_nNextLongUpdate0;
    uchar* m_nNextMidUpdate0;
    uchar* m_nNextShortUpdate0;
    uchar* m_aModelIndexLong0;
    uchar* m_aModelIndexMid0;
    uchar* m_aModelIndexShort0;
    int m_nLongCounter;
    int m_nMidCounter;
    int m_nShortCounter;
    int m_nN;
    float m_fTb;
    float m_fTau;
    int m_nkNN;
    bool m_bShadowDetection;
    uchar m_nShadowDetection;
    const Mat *knownForegroundMask;
};

#ifdef HAVE_OPENCL
bool BackgroundSubtractorKNNImpl::ocl_apply(InputArray _image, OutputArray _fgmask, double learningRate)
{
    bool needToInitialize = nframes == 0 || learningRate >= 1 || _image.size() != frameSize || _image.type() != frameType;

    if( needToInitialize )
        initialize(_image.size(), _image.type());

    ++nframes;
    learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1./std::min( 2*nframes, history );
    CV_Assert(learningRate >= 0);

    _fgmask.create(_image.size(), CV_8U);
    UMat fgmask = _fgmask.getUMat();

    UMat frame = _image.getUMat();

    //recalculate update rates - in case alpha is changed
    // calculate update parameters (using alpha)
    int Kshort,Kmid,Klong;
    //approximate exponential learning curve
    Kshort=(int)(log(0.7)/log(1-learningRate))+1;//Kshort
    Kmid=(int)(log(0.4)/log(1-learningRate))-Kshort+1;//Kmid
    Klong=(int)(log(0.1)/log(1-learningRate))-Kshort-Kmid+1;//Klong

    //refresh rates
    int nShortUpdate = (Kshort/nN)+1;
    int nMidUpdate = (Kmid/nN)+1;
    int nLongUpdate = (Klong/nN)+1;

    int idxArg = 0;
    idxArg = kernel_apply.set(idxArg, ocl::KernelArg::ReadOnly(frame));
    idxArg = kernel_apply.set(idxArg, ocl::KernelArg::PtrReadOnly(u_nNextLongUpdate));
    idxArg = kernel_apply.set(idxArg, ocl::KernelArg::PtrReadOnly(u_nNextMidUpdate));
    idxArg = kernel_apply.set(idxArg, ocl::KernelArg::PtrReadOnly(u_nNextShortUpdate));
    idxArg = kernel_apply.set(idxArg, ocl::KernelArg::PtrReadWrite(u_aModelIndexLong));
    idxArg = kernel_apply.set(idxArg, ocl::KernelArg::PtrReadWrite(u_aModelIndexMid));
    idxArg = kernel_apply.set(idxArg, ocl::KernelArg::PtrReadWrite(u_aModelIndexShort));
    idxArg = kernel_apply.set(idxArg, ocl::KernelArg::PtrReadWrite(u_flag));
    idxArg = kernel_apply.set(idxArg, ocl::KernelArg::PtrReadWrite(u_sample));
    idxArg = kernel_apply.set(idxArg, ocl::KernelArg::WriteOnlyNoSize(fgmask));

    idxArg = kernel_apply.set(idxArg, nLongCounter);
    idxArg = kernel_apply.set(idxArg, nMidCounter);
    idxArg = kernel_apply.set(idxArg, nShortCounter);
    idxArg = kernel_apply.set(idxArg, fTb);
    idxArg = kernel_apply.set(idxArg, nkNN);
    idxArg = kernel_apply.set(idxArg, fTau);
    if (bShadowDetection)
        kernel_apply.set(idxArg, nShadowDetection);

    size_t globalsize[2] = {(size_t)frame.cols, (size_t)frame.rows};
    if(!kernel_apply.run(2, globalsize, NULL, true))
        return false;

    nShortCounter++;//0,1,...,nShortUpdate-1
    nMidCounter++;
    nLongCounter++;
    if (nShortCounter >= nShortUpdate)
    {
        nShortCounter = 0;
        randu(u_nNextShortUpdate, Scalar::all(0),  Scalar::all(nShortUpdate));
    }
    if (nMidCounter >= nMidUpdate)
    {
        nMidCounter = 0;
        randu(u_nNextMidUpdate, Scalar::all(0),  Scalar::all(nMidUpdate));
    }
    if (nLongCounter >= nLongUpdate)
    {
        nLongCounter = 0;
        randu(u_nNextLongUpdate, Scalar::all(0),  Scalar::all(nLongUpdate));
    }
    return true;
}

bool BackgroundSubtractorKNNImpl::ocl_getBackgroundImage(OutputArray _backgroundImage) const
{
    _backgroundImage.create(frameSize, frameType);
    UMat dst = _backgroundImage.getUMat();

    int idxArg = 0;
    idxArg = kernel_getBg.set(idxArg, ocl::KernelArg::PtrReadOnly(u_flag));
    idxArg = kernel_getBg.set(idxArg, ocl::KernelArg::PtrReadOnly(u_sample));
    idxArg = kernel_getBg.set(idxArg, ocl::KernelArg::WriteOnly(dst));

    size_t globalsize[2] = {(size_t)dst.cols, (size_t)dst.rows};

    return kernel_getBg.run(2, globalsize, NULL, false);
}

void BackgroundSubtractorKNNImpl::create_ocl_apply_kernel()
{
    int nchannels = CV_MAT_CN(frameType);
    String opts = format("-D CN=%d -D NSAMPLES=%d%s", nchannels, nN, bShadowDetection ? " -D SHADOW_DETECT" : "");
    kernel_apply.create("knn_kernel", ocl::video::bgfg_knn_oclsrc, opts);
}

#endif

// Base 3 version class
void BackgroundSubtractorKNNImpl::apply(InputArray _image, OutputArray _fgmask, double learningRate) {
    apply(_image, _fgmask, noArray(), learningRate);
}

void BackgroundSubtractorKNNImpl::apply(InputArray _image, OutputArray _fgmask,InputArray _knownForegroundMask, double learningRate)
{
    CV_INSTRUMENT_REGION();

#ifdef HAVE_OPENCL
    if (opencl_ON)
    {
#ifndef __APPLE__
        CV_OCL_RUN(_fgmask.isUMat() && OCL_PERFORMANCE_CHECK(!ocl::Device::getDefault().isIntel() || _image.channels() == 1),
                   ocl_apply(_image, _fgmask, learningRate))
#else
        CV_OCL_RUN(_fgmask.isUMat() && OCL_PERFORMANCE_CHECK(!ocl::Device::getDefault().isIntel()),
                   ocl_apply(_image, _fgmask, learningRate))
#endif

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

    const Mat *knownMaskPtr = nullptr;
    Mat tmpKnownMask;
    if (!_knownForegroundMask.empty()) {
        // store a local Mat so the pointer stays alive for the parallel_for_
        tmpKnownMask = _knownForegroundMask.getMat();
        knownMaskPtr = &tmpKnownMask;
    }

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
    int nShortUpdate = (Kshort/nN)+1;
    int nMidUpdate = (Kmid/nN)+1;
    int nLongUpdate = (Klong/nN)+1;

    parallel_for_(Range(0, image.rows),
                  KNNInvoker(image, fgmask,
                             bgmodel.ptr(),
                             nNextLongUpdate.ptr(),
                             nNextMidUpdate.ptr(),
                             nNextShortUpdate.ptr(),
                             aModelIndexLong.ptr(),
                             aModelIndexMid.ptr(),
                             aModelIndexShort.ptr(),
                             nLongCounter,
                             nMidCounter,
                             nShortCounter,
                             nN,
                             fTb,
                             nkNN,
                             fTau,
                             bShadowDetection,
                             nShadowDetection,
                             knownMaskPtr),
                             image.total()/(double)(1 << 16));

    nShortCounter++;//0,1,...,nShortUpdate-1
    nMidCounter++;
    nLongCounter++;
    if (nShortCounter >= nShortUpdate)
    {
        nShortCounter = 0;
        randu(nNextShortUpdate, Scalar::all(0),  Scalar::all(nShortUpdate));
    }
    if (nMidCounter >= nMidUpdate)
    {
        nMidCounter = 0;
        randu(nNextMidUpdate, Scalar::all(0),  Scalar::all(nMidUpdate));
    }
    if (nLongCounter >= nLongUpdate)
    {
        nLongCounter = 0;
        randu(nNextLongUpdate, Scalar::all(0),  Scalar::all(nLongUpdate));
    }
}

void BackgroundSubtractorKNNImpl::getBackgroundImage(OutputArray backgroundImage) const
{
    CV_INSTRUMENT_REGION();

#ifdef HAVE_OPENCL
    if (opencl_ON)
    {
        CV_OCL_RUN(opencl_ON, ocl_getBackgroundImage(backgroundImage))

        opencl_ON = false;
    }
#endif

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
