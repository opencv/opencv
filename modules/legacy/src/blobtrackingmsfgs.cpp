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

#include "precomp.hpp"

#define SCALE_BASE 1.1
#define SCALE_RANGE 2
#define SCALE_NUM (2*SCALE_RANGE+1)
typedef float DefHistType;
#define DefHistTypeMat CV_32F
#define HIST_INDEX(_pData) (((_pData)[0]>>m_ByteShift) + (((_pData)[1]>>(m_ByteShift))<<m_BinBit)+((pImgData[2]>>m_ByteShift)<<(m_BinBit*2)))

void calcKernelEpanechnikov(CvMat* pK)
{    /* Allocate kernel for histogramm creation: */
    int     x,y;
    int     w = pK->width;
    int     h = pK->height;
    float   x0 = 0.5f*(w-1);
    float   y0 = 0.5f*(h-1);

    for(y=0; y<h; ++y) for(x=0; x<w; ++x)
    {
//                float   r2 = ((x-x0)*(x-x0)/(x0*x0)+(y-y0)*(y-y0)/(y0*y0));
        float   r2 = ((x-x0)*(x-x0)+(y-y0)*(y-y0))/((x0*x0)+(y0*y0));
        CV_MAT_ELEM(pK[0],DefHistType, y, x) = (DefHistType)((r2<1)?(1-r2):0);
    }
}   /* Allocate kernel for histogram creation. */

class CvBlobTrackerOneMSFGS:public CvBlobTrackerOne
{
private:
    /* Parameters: */
    float           m_FGWeight;
    float           m_Alpha;
    CvSize          m_ObjSize;
    CvMat*          m_KernelHistModel;
    CvMat*          m_KernelHistCandidate;
    CvSize          m_KernelMeanShiftSize;
    CvMat*          m_KernelMeanShiftK[SCALE_NUM];
    CvMat*          m_KernelMeanShiftG[SCALE_NUM];
    CvMat*          m_Weights;
    int             m_BinBit;
    int             m_ByteShift;
    int             m_BinNum;
    int             m_Dim;
    int             m_BinNumTotal;
    CvMat*          m_HistModel;
    float           m_HistModelVolume;
    CvMat*          m_HistCandidate;
    float           m_HistCandidateVolume;
    CvMat*          m_HistTemp;
    CvBlob          m_Blob;

    void ReAllocHist(int Dim, int BinBit)
    {
        m_BinBit = BinBit;
        m_ByteShift = 8-BinBit;
        m_Dim = Dim;
        m_BinNum = (1<<BinBit);
        m_BinNumTotal = cvRound(pow((double)m_BinNum,(double)m_Dim));
        if(m_HistModel) cvReleaseMat(&m_HistModel);
        if(m_HistCandidate) cvReleaseMat(&m_HistCandidate);
        if(m_HistTemp) cvReleaseMat(&m_HistTemp);
        m_HistCandidate = cvCreateMat(1, m_BinNumTotal, DefHistTypeMat);
        m_HistModel = cvCreateMat(1, m_BinNumTotal, DefHistTypeMat);
        m_HistTemp = cvCreateMat(1, m_BinNumTotal, DefHistTypeMat);
        cvZero(m_HistCandidate);
        cvZero(m_HistModel);
        m_HistModelVolume = 0.0f;
        m_HistCandidateVolume = 0.0f;
    }

    void ReAllocKernel(int  w, int h, float sigma=0.4)
    {
        double  ScaleToObj = sigma*1.39;
        int     kernel_width = cvRound(w/ScaleToObj);
        int     kernel_height = cvRound(h/ScaleToObj);
        int     x,y,s;
        assert(w>0);
        assert(h>0);
        m_ObjSize = cvSize(w,h);
        m_KernelMeanShiftSize = cvSize(kernel_width,kernel_height);


        /* Create kernels for histogram calculation: */
        if(m_KernelHistModel) cvReleaseMat(&m_KernelHistModel);
        m_KernelHistModel = cvCreateMat(h, w, DefHistTypeMat);
        calcKernelEpanechnikov(m_KernelHistModel);
        if(m_KernelHistCandidate) cvReleaseMat(&m_KernelHistCandidate);
        m_KernelHistCandidate = cvCreateMat(kernel_height, kernel_width, DefHistTypeMat);
        calcKernelEpanechnikov(m_KernelHistCandidate);

        if(m_Weights) cvReleaseMat(&m_Weights);
        m_Weights = cvCreateMat(kernel_height, kernel_width, CV_32F);

        for(s=-SCALE_RANGE; s<=SCALE_RANGE; ++s)
        {   /* Allocate kernel for meanshifts in space and scale: */
            int     si = s+SCALE_RANGE;
            double  cur_sigma = sigma * pow(SCALE_BASE,s);
            double  cur_sigma2 = cur_sigma*cur_sigma;
            double  x0 = 0.5*(kernel_width-1);
            double  y0 = 0.5*(kernel_height-1);
            if(m_KernelMeanShiftK[si]) cvReleaseMat(&m_KernelMeanShiftK[si]);
            if(m_KernelMeanShiftG[si]) cvReleaseMat(&m_KernelMeanShiftG[si]);
            m_KernelMeanShiftK[si] = cvCreateMat(kernel_height, kernel_width, DefHistTypeMat);
            m_KernelMeanShiftG[si] = cvCreateMat(kernel_height, kernel_width, DefHistTypeMat);

            for(y=0; y<kernel_height; ++y)
            {
                DefHistType* pK = (DefHistType*)CV_MAT_ELEM_PTR_FAST( m_KernelMeanShiftK[si][0], y, 0, sizeof(DefHistType) );
                DefHistType* pG = (DefHistType*)CV_MAT_ELEM_PTR_FAST( m_KernelMeanShiftG[si][0], y, 0, sizeof(DefHistType) );

                for(x=0; x<kernel_width; ++x)
                {
                    double r2 = ((x-x0)*(x-x0)/(x0*x0)+(y-y0)*(y-y0)/(y0*y0));
                    double sigma12 = cur_sigma2 / 2.56;
                    double sigma22 = cur_sigma2 * 2.56;
                    pK[x] = (DefHistType)(Gaussian2D(r2, sigma12)/sigma12 - Gaussian2D(r2, sigma22)/sigma22);
                    pG[x] = (DefHistType)(Gaussian2D(r2, cur_sigma2/1.6) - Gaussian2D(r2, cur_sigma2*1.6));
                }
            }   /* Next line. */
        }
    }   /* ReallocKernel */

    inline double Gaussian2D(double x, double sigma2)
    {
        return (exp(-x/(2*sigma2)) / (2*3.1415926535897932384626433832795*sigma2) );
    }

    void calcHist(IplImage* pImg, IplImage* pMask, CvPoint Center, CvMat* pKernel, CvMat* pHist, DefHistType* pHistVolume)
    {
        int         w = pKernel->width;
        int         h = pKernel->height;
        DefHistType Volume = 0;
        int         x0 = Center.x - w/2;
        int         y0 = Center.y - h/2;
        int         x,y;

        //cvZero(pHist);
        cvSet(pHist,cvScalar(1.0/m_BinNumTotal)); /* no zero bins, all bins have very small value*/
        Volume = 1;

        if(m_Dim == 3)
        {
            for(y=0; y<h; ++y)
            {
                unsigned char* pImgData = NULL;
                unsigned char* pMaskData = NULL;
                DefHistType* pKernelData = NULL;
                if((y0+y)>=pImg->height) continue;
                if((y0+y)<0)continue;
                pImgData = &CV_IMAGE_ELEM(pImg,unsigned char,y+y0,x0*3);
                pMaskData = pMask?(&CV_IMAGE_ELEM(pMask,unsigned char,y+y0,x0)):NULL;
                pKernelData = (DefHistType*)CV_MAT_ELEM_PTR_FAST(pKernel[0],y,0,sizeof(DefHistType));

                for(x=0; x<w; ++x, pImgData+=3)
                {
                    if((x0+x)>=pImg->width) continue;
                    if((x0+x)<0)continue;

                    if(pMaskData==NULL || pMaskData[x]>128)
                    {
                        DefHistType K = pKernelData[x];
                        int index = HIST_INDEX(pImgData);
                        assert(index >= 0 && index < pHist->cols);
                        Volume += K;
                        ((DefHistType*)(pHist->data.ptr))[index] += K;

                    }   /* Only masked pixels. */
                }   /*  Next column. */
            }   /*  Next row. */
        }   /* if m_Dim == 3. */

        if(pHistVolume)pHistVolume[0] = Volume;

    }; /* calcHist */

    double calcBhattacharyya()
    {
        cvMul(m_HistCandidate,m_HistModel,m_HistTemp);
        cvPow(m_HistTemp,m_HistTemp,0.5);
        return cvSum(m_HistTemp).val[0] / sqrt(m_HistCandidateVolume*m_HistModelVolume);
    }   /* calcBhattacharyyaCoefficient */

    void calcWeights(IplImage* pImg, IplImage* pImgFG, CvPoint Center)
    {
        cvZero(m_Weights);

        /* Calculate new position: */
        if(m_Dim == 3)
        {
            int         x0 = Center.x - m_KernelMeanShiftSize.width/2;
            int         y0 = Center.y - m_KernelMeanShiftSize.height/2;
            int         x,y;

            assert(m_Weights->width == m_KernelMeanShiftSize.width);
            assert(m_Weights->height == m_KernelMeanShiftSize.height);

            /* Calcualte shift vector: */
            for(y=0; y<m_KernelMeanShiftSize.height; ++y)
            {
                unsigned char* pImgData = NULL;
                unsigned char* pMaskData = NULL;
                float* pWData = NULL;

                if(y+y0 < 0 || y+y0 >= pImg->height) continue;

                pImgData = &CV_IMAGE_ELEM(pImg,unsigned char,y+y0,x0*3);
                pMaskData = pImgFG?(&CV_IMAGE_ELEM(pImgFG,unsigned char,y+y0,x0)):NULL;
                pWData = (float*)CV_MAT_ELEM_PTR_FAST(m_Weights[0],y,0,sizeof(float));

                for(x=0; x<m_KernelMeanShiftSize.width; ++x, pImgData+=3)
                {
                    double      V  = 0;
                    double      HM = 0;
                    double      HC = 0;
                    int         index;
                    if(x+x0 < 0 || x+x0 >= pImg->width) continue;

                    index = HIST_INDEX(pImgData);
                    assert(index >= 0 && index < m_BinNumTotal);

                    if(m_HistModelVolume>0)
                        HM = ((DefHistType*)m_HistModel->data.ptr)[index]/m_HistModelVolume;

                    if(m_HistCandidateVolume>0)
                        HC = ((DefHistType*)m_HistCandidate->data.ptr)[index]/m_HistCandidateVolume;

                    V = (HC>0)?sqrt(HM / HC):0;
                    V += m_FGWeight*(pMaskData?((pMaskData[x]/255.0f)):0);
                    pWData[x] = (float)MIN(V,100000);

                }   /* Next column. */
            }   /*  Next row. */
        }   /*  if m_Dim == 3. */
    }   /*  calcWeights */

public:
    CvBlobTrackerOneMSFGS()
    {
        int i;
        m_FGWeight = 0;
        m_Alpha = 0.0;

        /* Add several parameters for external use: */
        AddParam("FGWeight", &m_FGWeight);
        CommentParam("FGWeight","Weight of FG mask using (0 - mask will not be used for tracking)");
        AddParam("Alpha", &m_Alpha);
        CommentParam("Alpha","Coefficient for model histogramm updating (0 - hist is not upated)");

        m_BinBit=0;
        m_Dim = 0;
        m_HistModel = NULL;
        m_HistCandidate = NULL;
        m_HistTemp = NULL;
        m_KernelHistModel = NULL;
        m_KernelHistCandidate = NULL;
        m_Weights = NULL;

        for(i=0; i<SCALE_NUM; ++i)
        {
            m_KernelMeanShiftK[i] = NULL;
            m_KernelMeanShiftG[i] = NULL;
        }
        ReAllocHist(3,5);   /* 3D hist, each dimension has 2^5 bins. */

        SetModuleName("MSFGS");
    }

    ~CvBlobTrackerOneMSFGS()
    {
        int i;
        if(m_HistModel) cvReleaseMat(&m_HistModel);
        if(m_HistCandidate) cvReleaseMat(&m_HistCandidate);
        if(m_HistTemp) cvReleaseMat(&m_HistTemp);
        if(m_KernelHistModel) cvReleaseMat(&m_KernelHistModel);

        for(i=0; i<SCALE_NUM; ++i)
        {
            if(m_KernelMeanShiftK[i]) cvReleaseMat(&m_KernelMeanShiftK[i]);
            if(m_KernelMeanShiftG[i]) cvReleaseMat(&m_KernelMeanShiftG[i]);
        }
    }

    /* Interface: */
    virtual void Init(CvBlob* pBlobInit, IplImage* pImg, IplImage* pImgFG = NULL)
    {
        int w = cvRound(CV_BLOB_WX(pBlobInit));
        int h = cvRound(CV_BLOB_WY(pBlobInit));
        if(w<3)w=3;
        if(h<3)h=3;
        if(w>pImg->width)w=pImg->width;
        if(h>pImg->height)h=pImg->height;
        ReAllocKernel(w,h);
        calcHist(pImg, pImgFG, cvPointFrom32f(CV_BLOB_CENTER(pBlobInit)), m_KernelHistModel, m_HistModel, &m_HistModelVolume);
        m_Blob = pBlobInit[0];
    };

    virtual CvBlob* Process(CvBlob* pBlobPrev, IplImage* pImg, IplImage* pImgFG = NULL)
    {
        int     iter;

        if(pBlobPrev)
        {
            m_Blob = pBlobPrev[0];
        }

        for(iter=0; iter<10; ++iter)
        {
//            float   newx=0,newy=0,sum=0;
            float   dx=0,dy=0,sum=0;
            int     x,y,si;

            CvPoint Center = cvPoint(cvRound(m_Blob.x),cvRound(m_Blob.y));
            CvSize  Size   = cvSize(cvRound(m_Blob.w),cvRound(m_Blob.h));

            if(m_ObjSize.width != Size.width || m_ObjSize.height != Size.height)
            {   /* Reallocate kernels: */
                ReAllocKernel(Size.width,Size.height);
            }   /* Reallocate kernels. */

            /* Mean shift in coordinate space: */
            calcHist(pImg, NULL, Center, m_KernelHistCandidate, m_HistCandidate, &m_HistCandidateVolume);
            calcWeights(pImg, pImgFG, Center);

            for(si=1; si<(SCALE_NUM-1); ++si)
            {
                CvMat*  pKernel = m_KernelMeanShiftK[si];
                float   sdx = 0, sdy=0, ssum=0;
                int     s = si-SCALE_RANGE;
                float   factor = (1.0f-( float(s)/float(SCALE_RANGE) )*( float(s)/float(SCALE_RANGE) ));

                for(y=0; y<m_KernelMeanShiftSize.height; ++y)
                for(x=0; x<m_KernelMeanShiftSize.width;  ++x)
                {
                    float W = *(float*)CV_MAT_ELEM_PTR_FAST(m_Weights[0],y,x,sizeof(float));
                    float K = *(float*)CV_MAT_ELEM_PTR_FAST(pKernel[0],y,x,sizeof(float));
                    float KW = K*W;
                    ssum += (float)fabs(KW);
                    sdx += KW*(x-m_KernelMeanShiftSize.width*0.5f);
                    sdy += KW*(y-m_KernelMeanShiftSize.height*0.5f);
                }   /* Next pixel. */

                dx += sdx * factor;
                dy += sdy * factor;
                sum  += ssum * factor;

            }   /* Next scale. */

            if(sum > 0)
            {
                dx /= sum;
                dy /= sum;
            }

            m_Blob.x += dx;
            m_Blob.y += dy;

            {   /* Mean shift in scale space: */
                float   news = 0;
                float   sum = 0;
                float   scale;

                Center = cvPoint(cvRound(m_Blob.x),cvRound(m_Blob.y));
                calcHist(pImg, NULL, Center, m_KernelHistCandidate, m_HistCandidate, &m_HistCandidateVolume);
                calcWeights(pImg, pImgFG, Center);
                //cvSet(m_Weights,cvScalar(1));

                for(si=0; si<SCALE_NUM; si++)
                {
                    double  W = cvDotProduct(m_Weights, m_KernelMeanShiftG[si]);;
                    int     s = si-SCALE_RANGE;
                    sum += (float)fabs(W);
                    news += (float)(s*W);
                }

                if(sum>0)
                {
                    news /= sum;
                }

                scale = (float)pow((double)SCALE_BASE,(double)news);
                m_Blob.w *= scale;
                m_Blob.h *= scale;
            }   /* Mean shift in scale space. */

            /* Check fo finish: */
            if(fabs(dx)<0.1 && fabs(dy)<0.1) break;

        }   /* Next iteration. */

        if(m_Alpha>0)
        {   /* Update histogram: */
            double  Vol, WM, WC;
            CvPoint Center = cvPoint(cvRound(m_Blob.x),cvRound(m_Blob.y));
            calcHist(pImg, pImgFG, Center, m_KernelHistModel, m_HistCandidate, &m_HistCandidateVolume);
            Vol = 0.5*(m_HistModelVolume + m_HistCandidateVolume);
            WM = Vol*(1-m_Alpha)/m_HistModelVolume;
            WC = Vol*(m_Alpha)/m_HistCandidateVolume;
            cvAddWeighted(m_HistModel, WM, m_HistCandidate,WC,0,m_HistModel);
            m_HistModelVolume = (float)cvSum(m_HistModel).val[0];
        }   /* Update histogram. */

        return &m_Blob;

    };  /* Process */

    virtual void Release(){delete this;};
}; /*CvBlobTrackerOneMSFGS*/

CvBlobTrackerOne* cvCreateBlobTrackerOneMSFGS()
{
    return (CvBlobTrackerOne*) new CvBlobTrackerOneMSFGS;
}

CvBlobTracker* cvCreateBlobTrackerMSFGS()
{
    return cvCreateBlobTrackerList(cvCreateBlobTrackerOneMSFGS);
}

