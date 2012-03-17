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

#ifdef _OPENMP
#include "omp.h"
#endif

// Uncomment to trade flexibility for speed
//#define CONST_HIST_SIZE

// Uncomment to get some performance stats in stderr
//#define REPORT_TICKS

#ifdef CONST_HIST_SIZE
#define m_BinBit 5
#define m_ByteShift 3
#endif

typedef float DefHistType;
#define DefHistTypeMat CV_32F
#define HIST_INDEX(_pData) (((_pData)[0]>>m_ByteShift) + (((_pData)[1]>>(m_ByteShift))<<m_BinBit)+((pImgData[2]>>m_ByteShift)<<(m_BinBit*2)))

class DefHist
{
public:
    CvMat*          m_pHist;
    DefHistType     m_HistVolume;
    DefHist(int BinNum=0)
    {
        m_pHist = NULL;
        m_HistVolume = 0;
        Resize(BinNum);
    }

    ~DefHist()
    {
        if(m_pHist)cvReleaseMat(&m_pHist);
    }

    void Resize(int BinNum)
    {
        if(m_pHist)cvReleaseMat(&m_pHist);
        if(BinNum>0)
        {
            m_pHist = cvCreateMat(1, BinNum, DefHistTypeMat);
            cvZero(m_pHist);
        }
        m_HistVolume = 0;
    }

    void Update(DefHist* pH, float W)
    {   /* Update histogram: */
        double  Vol, WM, WC;
        Vol = 0.5*(m_HistVolume + pH->m_HistVolume);
        WM = Vol*(1-W)/m_HistVolume;
        WC = Vol*(W)/pH->m_HistVolume;
        cvAddWeighted(m_pHist, WM, pH->m_pHist,WC,0,m_pHist);
        m_HistVolume = (float)cvSum(m_pHist).val[0];
    }   /* Update histogram: */
};

class CvBlobTrackerOneMSFG:public CvBlobTrackerOne
{
protected:
    int             m_BinNumTotal; /* protected only for parralel MSPF */
    CvSize          m_ObjSize;

    void ReAllocKernel(int  w, int h)
    {
        int     x,y;
        float   x0 = 0.5f*(w-1);
        float   y0 = 0.5f*(h-1);
        assert(w>0);
        assert(h>0);
        m_ObjSize = cvSize(w,h);

        if(m_KernelHist) cvReleaseMat(&m_KernelHist);
        if(m_KernelMeanShift) cvReleaseMat(&m_KernelMeanShift);
        m_KernelHist = cvCreateMat(h, w, DefHistTypeMat);
        m_KernelMeanShift = cvCreateMat(h, w, DefHistTypeMat);

        for(y=0; y<h; ++y) for(x=0; x<w; ++x)
        {
            double r2 = ((x-x0)*(x-x0)/(x0*x0)+(y-y0)*(y-y0)/(y0*y0));
//            double r2 = ((x-x0)*(x-x0)+(y-y0)*(y-y0))/((y0*y0)+(x0*x0));
            CV_MAT_ELEM(m_KernelHist[0],DefHistType, y, x) = (DefHistType)GetKernelHist(r2);
            CV_MAT_ELEM(m_KernelMeanShift[0],DefHistType, y, x) = (DefHistType)GetKernelMeanShift(r2);
        }
    }

private:
    /* Parameters: */
    int             m_IterNum;
    float           m_FGWeight;
    float           m_Alpha;
    CvMat*          m_KernelHist;
    CvMat*          m_KernelMeanShift;
#ifndef CONST_HIST_SIZE
    int             m_BinBit;
    int             m_ByteShift;
#endif
    int             m_BinNum;
    int             m_Dim;
    /*
    CvMat*          m_HistModel;
    float           m_HistModelVolume;
    CvMat*          m_HistCandidate;
    float           m_HistCandidateVolume;
    CvMat*          m_HistTemp;
    */
    DefHist         m_HistModel;
    DefHist         m_HistCandidate;
    DefHist         m_HistTemp;

    CvBlob          m_Blob;
    int             m_Collision;

    void ReAllocHist(int Dim, int BinBit)
    {
#ifndef CONST_HIST_SIZE
        m_BinBit = BinBit;
        m_ByteShift = 8-BinBit;
#endif
        m_Dim = Dim;
        m_BinNum = (1<<BinBit);
        m_BinNumTotal = cvRound(pow((double)m_BinNum,(double)m_Dim));
        /*
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
        */
        m_HistCandidate.Resize(m_BinNumTotal);
        m_HistModel.Resize(m_BinNumTotal);
        m_HistTemp.Resize(m_BinNumTotal);
    }

    double GetKernelHist(double r2)
    {
        return (r2 < 1) ? 1 -  r2 : 0;
    }

    double GetKernelMeanShift(double r2)
    {
        return (r2<1) ? 1 : 0;
    }

//    void CollectHist(IplImage* pImg, IplImage* pMask, CvPoint Center, CvMat* pHist, DefHistType* pHistVolume)
//    void CollectHist(IplImage* pImg, IplImage* pMask, CvPoint Center, DefHist* pHist)

    void CollectHist(IplImage* pImg, IplImage* pMask, CvBlob* pBlob, DefHist* pHist)
    {
        int UsePrecalculatedKernel = 0;
        int BW = cvRound(pBlob->w);
        int BH = cvRound(pBlob->h);
        DefHistType Volume = 0;
        int         x0 = cvRound(pBlob->x - BW*0.5);
        int         y0 = cvRound(pBlob->y - BH*0.5);
        int         x,y;

        UsePrecalculatedKernel = (BW == m_ObjSize.width && BH == m_ObjSize.height ) ;

        //cvZero(pHist);
        cvSet(pHist->m_pHist,cvScalar(1.0/m_BinNumTotal)); /* no zero bins, all bins have very small value*/
        Volume = 1;

        assert(BW < pImg->width);
        assert(BH < pImg->height);
        if((x0+BW)>=pImg->width) BW=pImg->width-x0-1;
        if((y0+BH)>=pImg->height) BH=pImg->height-y0-1;
        if(x0<0){ x0=0;}
        if(y0<0){ y0=0;}

        if(m_Dim == 3)
        {
            for(y=0; y<BH; ++y)
            {
                unsigned char* pImgData = &CV_IMAGE_ELEM(pImg,unsigned char,y+y0,x0*3);
                unsigned char* pMaskData = pMask?(&CV_IMAGE_ELEM(pMask,unsigned char,y+y0,x0)):NULL;
                DefHistType* pKernelData = NULL;

                if(UsePrecalculatedKernel)
                {
                    pKernelData = ((DefHistType*)CV_MAT_ELEM_PTR_FAST(m_KernelHist[0],y,0,sizeof(DefHistType)));
                }

                for(x=0; x<BW; ++x, pImgData+=3)
                {
                    DefHistType K;
                    int index = HIST_INDEX(pImgData);
                    assert(index >= 0 && index < pHist->m_pHist->cols);

                    if(UsePrecalculatedKernel)
                    {
                        K = pKernelData[x];
                    }
                    else
                    {
                        float dx = (x+x0-pBlob->x)/(pBlob->w*0.5f);
                        float dy = (y+y0-pBlob->y)/(pBlob->h*0.5f);
                        double r2 = dx*dx+dy*dy;
                        K = (float)GetKernelHist(r2);
                    }

                    if(pMaskData)
                    {
                        K *= pMaskData[x]*0.003921568627450980392156862745098f;
                    }
                    Volume += K;
                    ((DefHistType*)(pHist->m_pHist->data.ptr))[index] += K;

                }   /* Next column. */
            }   /*  Next row. */
        }   /* if m_Dim == 3. */

        pHist->m_HistVolume = Volume;

    };  /* CollectHist */

    double calcBhattacharyya(DefHist* pHM = NULL, DefHist* pHC = NULL, DefHist* pHT = NULL)
    {
        if(pHM==NULL) pHM = &m_HistModel;
        if(pHC==NULL) pHC = &m_HistCandidate;
        if(pHT==NULL) pHT = &m_HistTemp;
        if(pHC->m_HistVolume*pHM->m_HistVolume > 0)
        {
#if 0
            // Use CV functions:
            cvMul(pHM->m_pHist,pHC->m_pHist,pHT->m_pHist);
            cvPow(pHT->m_pHist,pHT->m_pHist,0.5);
            return cvSum(pHT->m_pHist).val[0] / sqrt(pHC->m_HistVolume*pHM->m_HistVolume);
#else
            // Do computations manually and let autovectorizer do the job:
            DefHistType* hm=(DefHistType *)(pHM->m_pHist->data.ptr);
            DefHistType* hc=(DefHistType *)(pHC->m_pHist->data.ptr);
            //ht=(DefHistType *)(pHT->m_pHist->data.ptr);
            int size = pHM->m_pHist->width*pHM->m_pHist->height;
            double sum = 0.;
            for(int i = 0; i < size; i++ )
            {
                sum += sqrt(hm[i]*hc[i]);
            }
            return sum / sqrt(pHC->m_HistVolume*pHM->m_HistVolume);
#endif
        }
        return 0;
    }   /* calcBhattacharyyaCoefficient. */

protected:
    // double GetBhattacharyya(IplImage* pImg, IplImage* pImgFG, float x, float y, DefHist* pHist=NULL)
    double GetBhattacharyya(IplImage* pImg, IplImage* pImgFG, CvBlob* pBlob, DefHist* pHist=NULL, int /*thread_number*/ = 0)
    {
        if(pHist==NULL)pHist = &m_HistTemp;
        CollectHist(pImg, pImgFG, pBlob, pHist);
        return calcBhattacharyya(&m_HistModel, pHist, pHist);
    }

    void UpdateModelHist(IplImage* pImg, IplImage* pImgFG, CvBlob* pBlob)
    {
        if(m_Alpha>0 && !m_Collision)
        {   /* Update histogram: */
            CollectHist(pImg, pImgFG, pBlob, &m_HistCandidate);
            m_HistModel.Update(&m_HistCandidate, m_Alpha);
        }   /* Update histogram. */

    }   /* UpdateModelHist */

public:
    CvBlobTrackerOneMSFG()
    {

        /* Add several parameters for external use: */
        m_FGWeight = 2;
        AddParam("FGWeight", &m_FGWeight);
        CommentParam("FGWeight","Weight of FG mask using (0 - mask will not be used for tracking)");

        m_Alpha = 0.01f;
        AddParam("Alpha", &m_Alpha);
        CommentParam("Alpha","Coefficient for model histogram updating (0 - hist is not upated)");

        m_IterNum = 10;
        AddParam("IterNum", &m_IterNum);
        CommentParam("IterNum","Maximal number of iteration in meanshift operation");

        /* Initialize internal data: */
        m_Collision = 0;
//        m_BinBit=0;
        m_Dim = 0;
        /*
        m_HistModel = NULL;
        m_HistCandidate = NULL;
        m_HistTemp = NULL;
        */
        m_KernelHist = NULL;
        m_KernelMeanShift = NULL;
        ReAllocHist(3,5);   /* 3D hist, each dim has 2^5 bins*/

        SetModuleName("MSFG");
    }

    ~CvBlobTrackerOneMSFG()
    {
        /*
        if(m_HistModel) cvReleaseMat(&m_HistModel);
        if(m_HistCandidate) cvReleaseMat(&m_HistCandidate);
        if(m_HistTemp) cvReleaseMat(&m_HistTemp);
        */
        if(m_KernelHist) cvReleaseMat(&m_KernelHist);
        if(m_KernelMeanShift) cvReleaseMat(&m_KernelMeanShift);
    }

    /* Interface: */
    virtual void Init(CvBlob* pBlobInit, IplImage* pImg, IplImage* pImgFG = NULL)
    {
        int w = cvRound(CV_BLOB_WX(pBlobInit));
        int h = cvRound(CV_BLOB_WY(pBlobInit));
        if(w<CV_BLOB_MINW)w=CV_BLOB_MINW;
        if(h<CV_BLOB_MINH)h=CV_BLOB_MINH;
        if(pImg)
        {
            if(w>pImg->width)w=pImg->width;
            if(h>pImg->height)h=pImg->height;
        }
        ReAllocKernel(w,h);
        if(pImg)
            CollectHist(pImg, pImgFG, pBlobInit, &m_HistModel);
        m_Blob = pBlobInit[0];
    };

    virtual CvBlob* Process(CvBlob* pBlobPrev, IplImage* pImg, IplImage* pImgFG = NULL)
    {
        int     iter;

        if(pBlobPrev)
        {
            m_Blob = pBlobPrev[0];
        }

        {   /* Check blob size and realloc kernels if it is necessary: */
            int w = cvRound(m_Blob.w);
            int h = cvRound(m_Blob.h);
            if( w != m_ObjSize.width || h!=m_ObjSize.height)
            {
                ReAllocKernel(w,h);
                /* after this ( w != m_ObjSize.width || h!=m_ObjSize.height) shoiuld be false */
            }
        }   /* Check blob size and realloc kernels if it is necessary: */


        for(iter=0; iter<m_IterNum; ++iter)
        {
            float   newx=0,newy=0,sum=0;
            //int     x,y;
            double  B0;

            //CvPoint Center = cvPoint(cvRound(m_Blob.x),cvRound(m_Blob.y));
            CollectHist(pImg, NULL, &m_Blob, &m_HistCandidate);
            B0 = calcBhattacharyya();

            if(m_Wnd)if(CV_BLOB_ID(pBlobPrev)==0 && iter == 0)
            {   /* Debug: */
                IplImage*   pW = cvCloneImage(pImgFG);
                IplImage*   pWFG = cvCloneImage(pImgFG);
                int         x,y;

                cvZero(pW);
                cvZero(pWFG);

                assert(m_ObjSize.width < pImg->width);
                assert(m_ObjSize.height < pImg->height);

                /* Calculate shift vector: */
                for(y=0; y<pImg->height; ++y)
                {
                    unsigned char* pImgData = &CV_IMAGE_ELEM(pImg,unsigned char,y,0);
                    unsigned char* pMaskData = pImgFG?(&CV_IMAGE_ELEM(pImgFG,unsigned char,y,0)):NULL;

                    for(x=0; x<pImg->width; ++x, pImgData+=3)
                    {
                        int         xk = cvRound(x-(m_Blob.x-m_Blob.w*0.5));
                        int         yk = cvRound(y-(m_Blob.y-m_Blob.h*0.5));
                        double      HM = 0;
                        double      HC = 0;
                        double      K;
                        int index = HIST_INDEX(pImgData);
                        assert(index >= 0 && index < m_BinNumTotal);

                        if(fabs(x-m_Blob.x)>m_Blob.w*0.6) continue;
                        if(fabs(y-m_Blob.y)>m_Blob.h*0.6) continue;

                        if(xk < 0 || xk >= m_KernelMeanShift->cols) continue;
                        if(yk < 0 || yk >= m_KernelMeanShift->rows) continue;

                        if(m_HistModel.m_HistVolume>0)
                            HM = ((DefHistType*)m_HistModel.m_pHist->data.ptr)[index]/m_HistModel.m_HistVolume;

                        if(m_HistCandidate.m_HistVolume>0)
                            HC = ((DefHistType*)m_HistCandidate.m_pHist->data.ptr)[index]/m_HistCandidate.m_HistVolume;

                        K = *(DefHistType*)CV_MAT_ELEM_PTR_FAST(m_KernelMeanShift[0],yk,xk,sizeof(DefHistType));

                        if(HC>0)
                        {
                            double  V = sqrt(HM / HC);
                            int     Vi = cvRound(V * 64);
                            if(Vi < 0) Vi = 0;
                            if(Vi > 255) Vi = 255;
                            CV_IMAGE_ELEM(pW,uchar,y,x) = (uchar)Vi;

                            V += m_FGWeight*(pMaskData?(pMaskData[x]/255.0f):0);
                            V*=K;
                            Vi = cvRound(V * 64);
                            if(Vi < 0) Vi = 0;
                            if(Vi > 255) Vi = 255;
                            CV_IMAGE_ELEM(pWFG,uchar,y,x) = (uchar)Vi;
                        }

                    }   /* Next column. */
                }   /*  Next row. */

                //cvNamedWindow("MSFG_W",0);
                //cvShowImage("MSFG_W",pW);
                //cvNamedWindow("MSFG_WFG",0);
                //cvShowImage("MSFG_WFG",pWFG);
                //cvNamedWindow("MSFG_FG",0);
                //cvShowImage("MSFG_FG",pImgFG);

                //cvSaveImage("MSFG_W.bmp",pW);
                //cvSaveImage("MSFG_WFG.bmp",pWFG);
                //cvSaveImage("MSFG_FG.bmp",pImgFG);

            }   /* Debug. */


            /* Calculate new position by meanshift: */

            /* Calculate new position: */
            if(m_Dim == 3)
            {
                int         x0 = cvRound(m_Blob.x - m_ObjSize.width*0.5);
                int         y0 = cvRound(m_Blob.y - m_ObjSize.height*0.5);
                int         x,y;

                assert(m_ObjSize.width < pImg->width);
                assert(m_ObjSize.height < pImg->height);

                /* Crop blob bounds: */
                if((x0+m_ObjSize.width)>=pImg->width) x0=pImg->width-m_ObjSize.width-1;
                if((y0+m_ObjSize.height)>=pImg->height) y0=pImg->height-m_ObjSize.height-1;
                if(x0<0){ x0=0;}
                if(y0<0){ y0=0;}

                /* Calculate shift vector: */
                for(y=0; y<m_ObjSize.height; ++y)
                {
                    unsigned char* pImgData = &CV_IMAGE_ELEM(pImg,unsigned char,y+y0,x0*3);
                    unsigned char* pMaskData = pImgFG?(&CV_IMAGE_ELEM(pImgFG,unsigned char,y+y0,x0)):NULL;
                    DefHistType* pKernelData = (DefHistType*)CV_MAT_ELEM_PTR_FAST(m_KernelMeanShift[0],y,0,sizeof(DefHistType));

                    for(x=0; x<m_ObjSize.width; ++x, pImgData+=3)
                    {
                        DefHistType K = pKernelData[x];
                        double      HM = 0;
                        double      HC = 0;
                        int index = HIST_INDEX(pImgData);
                        assert(index >= 0 && index < m_BinNumTotal);

                        if(m_HistModel.m_HistVolume>0)
                            HM = ((DefHistType*)m_HistModel.m_pHist->data.ptr)[index]/m_HistModel.m_HistVolume;

                        if(m_HistCandidate.m_HistVolume>0)
                            HC = ((DefHistType*)m_HistCandidate.m_pHist->data.ptr)[index]/m_HistCandidate.m_HistVolume;

                        if(HC>0)
                        {
                            double V = sqrt(HM / HC);
                            if(!m_Collision && m_FGWeight>0 && pMaskData)
                            {
                                V += m_FGWeight*(pMaskData[x]/255.0f);
                            }
                            K *= (float)MIN(V,100000.);
                        }

                        sum += K;
                        newx += K*x;
                        newy += K*y;
                    }   /* Next column. */
                }   /*  Next row. */

                if(sum > 0)
                {
                    newx /= sum;
                    newy /= sum;
                }
                newx += x0;
                newy += y0;

            }   /* if m_Dim == 3. */

            /* Calculate new position by meanshift: */

            for(;;)
            {   /* Iterate using bahattcharrya coefficient: */
                double  B1;
                CvBlob  B = m_Blob;
//                CvPoint NewCenter = cvPoint(cvRound(newx),cvRound(newy));
                B.x = newx;
                B.y = newy;
                CollectHist(pImg, NULL, &B, &m_HistCandidate);
                B1 = calcBhattacharyya();
                if(B1 > B0) break;
                newx = 0.5f*(newx+m_Blob.x);
                newy = 0.5f*(newy+m_Blob.y);
                if(fabs(newx-m_Blob.x)<0.1 && fabs(newy-m_Blob.y)<0.1) break;
            }   /* Iterate using bahattcharrya coefficient. */

            if(fabs(newx-m_Blob.x)<0.5 && fabs(newy-m_Blob.y)<0.5) break;
            m_Blob.x = newx;
            m_Blob.y = newy;
        }   /* Next iteration. */

        while(!m_Collision && m_FGWeight>0)
        {   /* Update size if no collision. */
            float       Alpha = 0.04f;
            CvBlob      NewBlob;
            double      M00,X,Y,XX,YY;
            CvMoments   m;
            CvRect      r;
            CvMat       mat;

            r.width = cvRound(m_Blob.w*1.5+0.5);
            r.height = cvRound(m_Blob.h*1.5+0.5);
            r.x = cvRound(m_Blob.x - 0.5*r.width);
            r.y = cvRound(m_Blob.y - 0.5*r.height);
            if(r.x < 0) break;
            if(r.y < 0) break;
            if(r.x+r.width >= pImgFG->width) break;
            if(r.y+r.height >= pImgFG->height) break;
            if(r.height < 5 || r.width < 5) break;

            cvMoments( cvGetSubRect(pImgFG,&mat,r), &m, 0 );
            M00 = cvGetSpatialMoment( &m, 0, 0 );
            if(M00 <= 0 ) break;
            X = cvGetSpatialMoment( &m, 1, 0 )/M00;
            Y = cvGetSpatialMoment( &m, 0, 1 )/M00;
            XX = (cvGetSpatialMoment( &m, 2, 0 )/M00) - X*X;
            YY = (cvGetSpatialMoment( &m, 0, 2 )/M00) - Y*Y;
            NewBlob = cvBlob(r.x+(float)X,r.y+(float)Y,(float)(4*sqrt(XX)),(float)(4*sqrt(YY)));

            NewBlob.w = Alpha*NewBlob.w+m_Blob.w*(1-Alpha);
            NewBlob.h = Alpha*NewBlob.h+m_Blob.h*(1-Alpha);

            m_Blob.w = MAX(NewBlob.w,5);
            m_Blob.h = MAX(NewBlob.h,5);
            break;

        }   /* Update size if no collision. */

        return &m_Blob;

    };  /* CvBlobTrackerOneMSFG::Process */

    virtual double GetConfidence(CvBlob* pBlob, IplImage* pImg, IplImage* /*pImgFG*/ = NULL, IplImage* pImgUnusedReg = NULL)
    {
        double  S = 0.2;
        double  B = GetBhattacharyya(pImg, pImgUnusedReg, pBlob, &m_HistTemp);
        return exp((B-1)/(2*S));

    };  /*CvBlobTrackerOneMSFG::*/

    virtual void Update(CvBlob* pBlob, IplImage* pImg, IplImage* pImgFG = NULL)
    {   /* Update histogram: */
        UpdateModelHist(pImg, pImgFG, pBlob?pBlob:&m_Blob);
    }   /*CvBlobTrackerOneMSFG::*/

    virtual void Release(){delete this;};
    virtual void SetCollision(int CollisionFlag)
    {
        m_Collision = CollisionFlag;
    }
    virtual void SaveState(CvFileStorage* fs)
    {
        cvWriteStruct(fs, "Blob", &m_Blob, "ffffi");
        cvWriteInt(fs,"Collision", m_Collision);
        cvWriteInt(fs,"HistVolume", cvRound(m_HistModel.m_HistVolume));
        cvWrite(fs,"Hist", m_HistModel.m_pHist);
    };
    virtual void LoadState(CvFileStorage* fs, CvFileNode* node)
    {
        CvMat* pM;
        cvReadStructByName(fs, node, "Blob",&m_Blob, "ffffi");
        m_Collision = cvReadIntByName(fs,node,"Collision",m_Collision);
        pM = (CvMat*)cvRead(fs,cvGetFileNodeByName(fs,node,"Hist"));
        if(pM)
        {
            m_HistModel.m_pHist = pM;
            m_HistModel.m_HistVolume = (float)cvSum(pM).val[0];
        }
    };

};  /*CvBlobTrackerOneMSFG*/

#if 0
void CvBlobTrackerOneMSFG::CollectHist(IplImage* pImg, IplImage* pMask, CvBlob* pBlob, DefHist* pHist)
{
    int UsePrecalculatedKernel = 0;
    int BW = cvRound(pBlob->w);
    int BH = cvRound(pBlob->h);
    DefHistType Volume = 0;
    int         x0 = cvRound(pBlob->x - BW*0.5);
    int         y0 = cvRound(pBlob->y - BH*0.5);
    int         x,y;

    UsePrecalculatedKernel = (BW == m_ObjSize.width && BH == m_ObjSize.height ) ;

    //cvZero(pHist);
    cvSet(pHist->m_pHist,cvScalar(1.0/m_BinNumTotal)); /* no zero bins, all bins have very small value*/
    Volume = 1;

    assert(BW < pImg->width);
    assert(BH < pImg->height);
    if((x0+BW)>=pImg->width) BW=pImg->width-x0-1;
    if((y0+BH)>=pImg->height) BH=pImg->height-y0-1;
    if(x0<0){ x0=0;}
    if(y0<0){ y0=0;}

    if(m_Dim == 3)
    {
        for(y=0; y<BH; ++y)
        {
            unsigned char* pImgData = &CV_IMAGE_ELEM(pImg,unsigned char,y+y0,x0*3);
            unsigned char* pMaskData = pMask?(&CV_IMAGE_ELEM(pMask,unsigned char,y+y0,x0)):NULL;
            DefHistType* pKernelData = NULL;

            if(UsePrecalculatedKernel)
            {
                pKernelData = ((DefHistType*)CV_MAT_ELEM_PTR_FAST(m_KernelHist[0],y,0,sizeof(DefHistType)));
            }

            for(x=0; x<BW; ++x, pImgData+=3)
            {
                DefHistType K;
                int index = HIST_INDEX(pImgData);
                assert(index >= 0 && index < pHist->m_pHist->cols);

                if(UsePrecalculatedKernel)
                {
                    K = pKernelData[x];
                }
                else
                {
                    float dx = (x+x0-pBlob->x)/(pBlob->w*0.5);
                    float dy = (y+y0-pBlob->y)/(pBlob->h*0.5);
                    double r2 = dx*dx+dy*dy;
                    K = GetKernelHist(r2);
                }

                if(pMaskData)
                {
                    K *= pMaskData[x]*0.003921568627450980392156862745098;
                }
                Volume += K;
                ((DefHistType*)(pHist->m_pHist->data.ptr))[index] += K;

            }   /* Next column. */
        }   /*  Next row. */
    }   /*  if m_Dim == 3. */

    pHist->m_HistVolume = Volume;

};  /* CollectHist */
#endif

CvBlobTrackerOne* cvCreateBlobTrackerOneMSFG()
{
    return (CvBlobTrackerOne*) new CvBlobTrackerOneMSFG;
}

CvBlobTracker* cvCreateBlobTrackerMSFG()
{
    return cvCreateBlobTrackerList(cvCreateBlobTrackerOneMSFG);
}

/* Create specific tracker without FG
 * usin - just simple mean-shift tracker: */
class CvBlobTrackerOneMS:public CvBlobTrackerOneMSFG
{
public:
    CvBlobTrackerOneMS()
    {
        SetParam("FGWeight",0);
        DelParam("FGWeight");
        SetModuleName("MS");
    };
};

CvBlobTrackerOne* cvCreateBlobTrackerOneMS()
{
    return (CvBlobTrackerOne*) new CvBlobTrackerOneMS;
}

CvBlobTracker* cvCreateBlobTrackerMS()
{
    return cvCreateBlobTrackerList(cvCreateBlobTrackerOneMS);
}

typedef struct DefParticle
{
    CvBlob  blob;
    float   Vx,Vy;
    double  W;
} DefParticle;

class CvBlobTrackerOneMSPF:public CvBlobTrackerOneMS
{
private:
    /* parameters */
    int             m_ParticleNum;
    float           m_UseVel;
    float           m_SizeVar;
    float           m_PosVar;

    CvSize          m_ImgSize;
    CvBlob          m_Blob;
    DefParticle*    m_pParticlesPredicted;
    DefParticle*    m_pParticlesResampled;
    CvRNG           m_RNG;
#ifdef _OPENMP
    int             m_ThreadNum;
    DefHist*        m_HistForParalel;
#endif

public:
    virtual void SaveState(CvFileStorage* fs)
    {
        CvBlobTrackerOneMS::SaveState(fs);
        cvWriteInt(fs,"ParticleNum",m_ParticleNum);
        cvWriteStruct(fs,"ParticlesPredicted",m_pParticlesPredicted,"ffffiffd",m_ParticleNum);
        cvWriteStruct(fs,"ParticlesResampled",m_pParticlesResampled,"ffffiffd",m_ParticleNum);
    };

    virtual void LoadState(CvFileStorage* fs, CvFileNode* node)
    {
        //CvMat* pM;
        CvBlobTrackerOneMS::LoadState(fs,node);
        m_ParticleNum = cvReadIntByName(fs,node,"ParticleNum",m_ParticleNum);
        if(m_ParticleNum>0)
        {
            Realloc();
            printf("sizeof(DefParticle) is %d\n", (int)sizeof(DefParticle));
            cvReadStructByName(fs,node,"ParticlesPredicted",m_pParticlesPredicted,"ffffiffd");
            cvReadStructByName(fs,node,"ParticlesResampled",m_pParticlesResampled,"ffffiffd");
        }
    };
    CvBlobTrackerOneMSPF()
    {
        m_pParticlesPredicted = NULL;
        m_pParticlesResampled = NULL;
        m_ParticleNum = 200;

        AddParam("ParticleNum",&m_ParticleNum);
        CommentParam("ParticleNum","Number of particles");
        Realloc();

        m_UseVel = 0;
        AddParam("UseVel",&m_UseVel);
        CommentParam("UseVel","Percent of particles which use velocity feature");

        m_SizeVar = 0.05f;
        AddParam("SizeVar",&m_SizeVar);
        CommentParam("SizeVar","Size variation (in object size)");

        m_PosVar = 0.2f;
        AddParam("PosVar",&m_PosVar);
        CommentParam("PosVar","Position variation (in object size)");

        m_RNG = cvRNG(0);

        SetModuleName("MSPF");

#ifdef _OPENMP
        {
            m_ThreadNum = omp_get_num_procs();
            m_HistForParalel = new DefHist[m_ThreadNum];
        }
#endif
    }

    ~CvBlobTrackerOneMSPF()
    {
        if(m_pParticlesResampled)cvFree(&m_pParticlesResampled);
        if(m_pParticlesPredicted)cvFree(&m_pParticlesPredicted);
#ifdef _OPENMP
        if(m_HistForParalel) delete[] m_HistForParalel;
#endif
    }

private:
    void Realloc()
    {
        if(m_pParticlesResampled)cvFree(&m_pParticlesResampled);
        if(m_pParticlesPredicted)cvFree(&m_pParticlesPredicted);
        m_pParticlesPredicted = (DefParticle*)cvAlloc(sizeof(DefParticle)*m_ParticleNum);
        m_pParticlesResampled = (DefParticle*)cvAlloc(sizeof(DefParticle)*m_ParticleNum);
    };  /* Realloc*/

    void DrawDebug(IplImage* pImg, IplImage* /*pImgFG*/)
    {
        int k;
        for(k=0; k<2; ++k)
        {
            DefParticle*    pBP = k?m_pParticlesResampled:m_pParticlesPredicted;
            //const char*   name = k?"MSPF resampled particle":"MSPF Predicted particle";
            IplImage*       pI = cvCloneImage(pImg);
            int             h,hN = m_ParticleNum;
            CvBlob          C = cvBlob(0,0,0,0);
            double          WS = 0;
            for(h=0; h<hN; ++h)
            {
                CvBlob  B = pBP[h].blob;
                int     CW = cvRound(255*pBP[h].W);
                CvBlob* pB = &B;
                int x = cvRound(CV_BLOB_RX(pB)), y = cvRound(CV_BLOB_RY(pB));
                CvSize  s = cvSize(MAX(1,x), MAX(1,y));
                double  W = pBP[h].W;
                C.x += pB->x;
                C.y += pB->y;
                C.w += pB->w;
                C.h += pB->h;
                WS+=W;

                s = cvSize(1,1);
                cvEllipse( pI,
                    cvPointFrom32f(CV_BLOB_CENTER(pB)),
                    s,
                    0, 0, 360,
                    CV_RGB(CW,0,0), 1 );

            }   /* Next hypothesis. */

            C.x /= hN;
            C.y /= hN;
            C.w /= hN;
            C.h /= hN;

            cvEllipse( pI,
                cvPointFrom32f(CV_BLOB_CENTER(&C)),
                cvSize(cvRound(C.w*0.5),cvRound(C.h*0.5)),
                0, 0, 360,
                CV_RGB(0,0,255), 1 );

            cvEllipse( pI,
                cvPointFrom32f(CV_BLOB_CENTER(&m_Blob)),
                cvSize(cvRound(m_Blob.w*0.5),cvRound(m_Blob.h*0.5)),
                0, 0, 360,
                CV_RGB(0,255,0), 1 );

            //cvNamedWindow(name,0);
            //cvShowImage(name,pI);
            cvReleaseImage(&pI);
        } /*  */

        //printf("Blob %d, point (%.1f,%.1f) size (%.1f,%.1f)\n",m_Blob.ID,m_Blob.x,m_Blob.y,m_Blob.w,m_Blob.h);
    } /* ::DrawDebug */

private:
    void Prediction()
    {
        int p;
        for(p=0; p<m_ParticleNum; ++p)
        {   /* "Prediction" of particle: */
            //double  t;
            float   r[5];
            CvMat   rm = cvMat(1,5,CV_32F,r);
            cvRandArr(&m_RNG,&rm,CV_RAND_NORMAL,cvScalar(0),cvScalar(1));

            m_pParticlesPredicted[p] = m_pParticlesResampled[p];

            if(cvRandReal(&m_RNG)<0.5)
            {   /* Half of particles will predict based on external blob: */
                m_pParticlesPredicted[p].blob = m_Blob;
            }

            if(cvRandReal(&m_RNG)<m_UseVel)
            {   /* Predict moving particle by usual way by using speed: */
                m_pParticlesPredicted[p].blob.x += m_pParticlesPredicted[p].Vx;
                m_pParticlesPredicted[p].blob.y += m_pParticlesPredicted[p].Vy;
            }
            else
            {   /* Stop several particles: */
                m_pParticlesPredicted[p].Vx = 0;
                m_pParticlesPredicted[p].Vy = 0;
            }

            {   /* Update position: */
                float S = (m_Blob.w + m_Blob.h)*0.5f;
                m_pParticlesPredicted[p].blob.x += m_PosVar*S*r[0];
                m_pParticlesPredicted[p].blob.y += m_PosVar*S*r[1];

                /* Update velocity: */
                m_pParticlesPredicted[p].Vx += (float)(m_PosVar*S*0.1*r[3]);
                m_pParticlesPredicted[p].Vy += (float)(m_PosVar*S*0.1*r[4]);
            }

            /* Update size: */
            m_pParticlesPredicted[p].blob.w *= (1+m_SizeVar*r[2]);
            m_pParticlesPredicted[p].blob.h *= (1+m_SizeVar*r[2]);

            /* Truncate size of particle: */
            if(m_pParticlesPredicted[p].blob.w > m_ImgSize.width*0.5f)
            {
                m_pParticlesPredicted[p].blob.w = m_ImgSize.width*0.5f;
            }

            if(m_pParticlesPredicted[p].blob.h > m_ImgSize.height*0.5f)
            {
                m_pParticlesPredicted[p].blob.h = m_ImgSize.height*0.5f;
            }

            if(m_pParticlesPredicted[p].blob.w < 1 )
            {
                m_pParticlesPredicted[p].blob.w = 1;
            }

            if(m_pParticlesPredicted[p].blob.h < 1)
            {
                m_pParticlesPredicted[p].blob.h = 1;
            }
        }   /* "Prediction" of particle. */
    }   /* Prediction */

    void UpdateWeightsMS(IplImage* pImg, IplImage* /*pImgFG*/)
    {
        int p;
#ifdef _OPENMP
        if( m_HistForParalel[0].m_pHist==NULL || m_HistForParalel[0].m_pHist->cols != m_BinNumTotal)
        {
            int t;
            for(t=0; t<m_ThreadNum; ++t)
                m_HistForParalel[t].Resize(m_BinNumTotal);
        }
#endif

#ifdef _OPENMP
#pragma omp parallel for num_threads(m_ThreadNum) schedule(runtime)
#endif
        for(p=0;p<m_ParticleNum;++p)
        {   /* Calculate weights for particles: */
            double  S = 0.2;
            double  B = 0;
#ifdef _OPENMP
            assert(omp_get_thread_num()<m_ThreadNum);
#endif

            B = GetBhattacharyya(
                pImg, NULL,
                &(m_pParticlesPredicted[p].blob)
#ifdef _OPENMP
                ,&(m_HistForParalel[omp_get_thread_num()])
#endif
                );
            m_pParticlesPredicted[p].W *= exp((B-1)/(2*S));

        }   /* Calculate weights for particles. */
    }

    void UpdateWeightsCC(IplImage* /*pImg*/, IplImage* /*pImgFG*/)
    {
        int p;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(p=0; p<m_ParticleNum; ++p)
        {   /* Calculate weights for particles: */
            double W = 1;
            m_pParticlesPredicted[p].W *= W;
        }   /* Calculate weights for particles. */
    }

    void Resample()
    {   /* Resample particle: */
        int         p;
        double      Sum = 0;

        for(p=0; p<m_ParticleNum; ++p)
        {
            Sum += m_pParticlesPredicted[p].W;
        }

        for(p=0; p<m_ParticleNum; ++p)
        {
            double  T = Sum * cvRandReal(&m_RNG);   /* Set current random threshold for cululative weight. */
            int     p2;
            double  Sum2 = 0;

            for(p2=0; p2<m_ParticleNum; ++p2)
            {
                Sum2 += m_pParticlesPredicted[p2].W;
                if(Sum2 >= T)break;
            }

            if(p2>=m_ParticleNum)p2=m_ParticleNum-1;
            m_pParticlesResampled[p] = m_pParticlesPredicted[p2];
            m_pParticlesResampled[p].W = 1;

        }   /* Find next particle. */
    }   /*  Resample particle. */


public:
    virtual void Init(CvBlob* pBlobInit, IplImage* pImg, IplImage* pImgFG = NULL)
    {
        int i;
        CvBlobTrackerOneMSFG::Init(pBlobInit, pImg, pImgFG);
        DefParticle PP;
        PP.W = 1;
        PP.Vx = 0;
        PP.Vy = 0;
        PP.blob = pBlobInit[0];
        for(i=0;i<m_ParticleNum;++i)
        {
            m_pParticlesPredicted[i] = PP;
            m_pParticlesResampled[i] = PP;
        }
        m_Blob = pBlobInit[0];

    }   /* CvBlobTrackerOneMSPF::Init*/

    virtual CvBlob* Process(CvBlob* pBlobPrev, IplImage* pImg, IplImage* pImgFG = NULL)
    {
        int p;

        m_ImgSize.width = pImg->width;
        m_ImgSize.height = pImg->height;


        m_Blob = pBlobPrev[0];

        {   /* Check blob size and realloc kernels if it is necessary: */
            int w = cvRound(m_Blob.w);
            int h = cvRound(m_Blob.h);
            if( w != m_ObjSize.width || h!=m_ObjSize.height)
            {
                ReAllocKernel(w,h);
                /* After this ( w != m_ObjSize.width || h!=m_ObjSize.height) should be false. */
            }
        }   /* Check blob size and realloc kernels if it is necessary. */

        Prediction();

#ifdef REPORT_TICKS
        int64 ticks = cvGetTickCount();
#endif

        UpdateWeightsMS(pImg, pImgFG);

#ifdef REPORT_TICKS
        ticks = cvGetTickCount() - ticks;
        fprintf(stderr, "PF UpdateWeights, %d ticks\n",  (int)ticks);
        ticks = cvGetTickCount();
#endif

        Resample();

#ifdef REPORT_TICKS
        ticks = cvGetTickCount() - ticks;
        fprintf(stderr, "PF Resampling, %d ticks\n",  (int)ticks);
#endif

        {   /* Find average result: */
            float   x = 0;
            float   y = 0;
            float   w = 0;
            float   h = 0;
            float   Sum = 0;

            DefParticle* pP = m_pParticlesResampled;

            for(p=0; p<m_ParticleNum; ++p)
            {
                float W = (float)pP[p].W;
                x += W*pP[p].blob.x;
                y += W*pP[p].blob.y;
                w += W*pP[p].blob.w;
                h += W*pP[p].blob.h;
                Sum += W;
            }

            if(Sum>0)
            {
                m_Blob.x = x / Sum;
                m_Blob.y = y / Sum;
                m_Blob.w = w / Sum;
                m_Blob.h = h / Sum;
            }
        }   /* Find average result. */

        if(m_Wnd)
        {
            DrawDebug(pImg, pImgFG);
        }

        return &m_Blob;

    }   /* CvBlobTrackerOneMSPF::Process */

    virtual void SkipProcess(CvBlob* pBlob, IplImage* /*pImg*/, IplImage* /*pImgFG*/ = NULL)
    {
        int p;
        for(p=0; p<m_ParticleNum; ++p)
        {
            m_pParticlesResampled[p].blob = pBlob[0];
            m_pParticlesResampled[p].Vx = 0;
            m_pParticlesResampled[p].Vy = 0;
            m_pParticlesResampled[p].W = 1;
        }
    }

    virtual void Release(){delete this;};
    virtual void ParamUpdate()
    {
        Realloc();
    }

};  /* CvBlobTrackerOneMSPF */

CvBlobTrackerOne* cvCreateBlobTrackerOneMSPF()
{
    return (CvBlobTrackerOne*) new CvBlobTrackerOneMSPF;
}

CvBlobTracker* cvCreateBlobTrackerMSPF()
{
    return cvCreateBlobTrackerList(cvCreateBlobTrackerOneMSPF);
}

