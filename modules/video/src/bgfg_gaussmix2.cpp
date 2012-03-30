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


#define CV_BG_MODEL_MOG2            3                 /* "Mixture of Gaussians 2".  */


/* default parameters of gaussian background detection algorithm */
#define CV_BGFG_MOG2_STD_THRESHOLD            4.0f     /* lambda=2.5 is 99% */
#define CV_BGFG_MOG2_WINDOW_SIZE              500      /* Learning rate; alpha = 1/CV_GBG_WINDOW_SIZE */
#define CV_BGFG_MOG2_BACKGROUND_THRESHOLD     0.9f     /* threshold sum of weights for background test */
#define CV_BGFG_MOG2_STD_THRESHOLD_GENERATE   3.0f     /* lambda=2.5 is 99% */
#define CV_BGFG_MOG2_NGAUSSIANS               5        /* = K = number of Gaussians in mixture */
#define CV_BGFG_MOG2_VAR_INIT                 15.0f    /* initial variance for new components*/
#define CV_BGFG_MOG2_VAR_MIN                    4.0f
#define CV_BGFG_MOG2_VAR_MAX                      5*CV_BGFG_MOG2_VAR_INIT
#define CV_BGFG_MOG2_MINAREA                  15.0f    /* for postfiltering */

/* additional parameters */
#define CV_BGFG_MOG2_CT                               0.05f     /* complexity reduction prior constant 0 - no reduction of number of components*/
#define CV_BGFG_MOG2_SHADOW_VALUE             127       /* value to use in the segmentation mask for shadows, sot 0 not to do shadow detection*/
#define CV_BGFG_MOG2_SHADOW_TAU               0.5f      /* Tau - shadow threshold, see the paper for explanation*/

typedef struct CvGaussBGStatModel2Params
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
} CvGaussBGStatModel2Params;

#define CV_BGFG_MOG2_NDMAX 3

typedef struct CvPBGMMGaussian
{
    float weight;
    float mean[CV_BGFG_MOG2_NDMAX];
    float variance;
}CvPBGMMGaussian;

typedef struct CvGaussBGStatModel2Data
{ 
    CvPBGMMGaussian* rGMM; //array for the mixture of Gaussians
    unsigned char* rnUsedModes;//number of Gaussian components per pixel (maximum 255)
} CvGaussBGStatModel2Data;



//shadow detection performed per pixel
// should work for rgb data, could be usefull for gray scale and depth data as well
//  See: Prati,Mikic,Trivedi,Cucchiarra,"Detecting Moving Shadows...",IEEE PAMI,2003.
CV_INLINE int _icvRemoveShadowGMM(float* data, int nD,
                                unsigned char nModes, 
                                CvPBGMMGaussian* pGMM,
                                float m_fTb,
                                float m_fTB,    
                                float m_fTau)
{
    float tWeight = 0;
    float numerator, denominator;
    // check all the components  marked as background:
    for (int iModes=0;iModes<nModes;iModes++)
    {

        CvPBGMMGaussian g=pGMM[iModes];

        numerator = 0.0f;
        denominator = 0.0f;
        for (int iD=0;iD<nD;iD++)
        {
                numerator   += data[iD]  * g.mean[iD];
                denominator += g.mean[iD]* g.mean[iD];
        }

        // no division by zero allowed
        if (denominator == 0)
        {
                return 0;
        };
        float a = numerator / denominator;

        // if tau < a < 1 then also check the color distortion
        if ((a <= 1) && (a >= m_fTau))
        {

            float dist2a=0.0f;
            
            for (int iD=0;iD<nD;iD++)
            {
                float dD= a*g.mean[iD] - data[iD];
                dist2a += (dD*dD);
            }

            if (dist2a<m_fTb*g.variance*a*a)
            {
                return 2;
            }
        };

        tWeight += g.weight;
        if (tWeight > m_fTB)
        {
                return 0;
        };
    };
    return 0;
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

CV_INLINE int _icvUpdateGMM(float* data, int nD,
                                unsigned char* pModesUsed, 
                                CvPBGMMGaussian* pGMM,
                                int m_nM,
                                float m_fAlphaT,
                                float m_fTb,
                                float m_fTB,    
                                float m_fTg,
                                float m_fVarInit,
                                float m_fVarMax,
                                float m_fVarMin,
                                float m_fPrune)
{
    //calculate distances to the modes (+ sort)
    //here we need to go in descending order!!! 
    bool bBackground=0;//return value -> true - the pixel classified as background

    //internal:
    bool bFitsPDF=0;//if it remains zero a new GMM mode will be added   
    float m_fOneMinAlpha=1-m_fAlphaT;
    unsigned char nModes=*pModesUsed;//current number of modes in GMM
    float totalWeight=0.0f;

    //////
    //go through all modes
    int iMode=0;
    CvPBGMMGaussian* pGauss=pGMM;
    for (;iMode<nModes;iMode++,pGauss++)
    {
        float weight = pGauss->weight;//need only weight if fit is found
        weight=m_fOneMinAlpha*weight+m_fPrune;

        ////
        //fit not found yet
        if (!bFitsPDF)
        {
            //check if it belongs to some of the remaining modes
            float var=pGauss->variance;

            //calculate difference and distance
            float dist2=0.0f;
#if (CV_BGFG_MOG2_NDMAX==1)
            float dData=pGauss->mean[0]-data[0];
            dist2=dData*dData;
#else           
            float dData[CV_BGFG_MOG2_NDMAX];

            for (int iD=0;iD<nD;iD++)
            {
                dData[iD]=pGauss->mean[iD]-data[iD];
                dist2+=dData[iD]*dData[iD];
            }
#endif      
            //background? - m_fTb - usually larger than m_fTg
            if ((totalWeight<m_fTB)&&(dist2<m_fTb*var))
                    bBackground=1;

            //check fit
            if (dist2<m_fTg*var)
            {
                /////
                //belongs to the mode - bFitsPDF becomes 1
                bFitsPDF=1;

                //update distribution               
                
                //update weight
                weight+=m_fAlphaT;

                float k = m_fAlphaT/weight;

                //update mean
#if (CV_BGFG_MOG2_NDMAX==1)
                pGauss->mean[0]-=k*dData;
#else               
                for (int iD=0;iD<nD;iD++)
                {
                    pGauss->mean[iD]-=k*dData[iD];
                }
#endif

                //update variance
                float varnew = var + k*(dist2-var);
                //limit the variance                
                pGauss->variance = MIN(m_fVarMax,MAX(varnew,m_fVarMin));

                //sort
                //all other weights are at the same place and 
                //only the matched (iModes) is higher -> just find the new place for it             
                for (int iLocal = iMode;iLocal>0;iLocal--)
                {
                    //check one up
                    if (weight < (pGMM[iLocal-1].weight))
                    {
                        break;
                    }
                    else
                    {
                        //swap one up
                        CvPBGMMGaussian temp = pGMM[iLocal];
                        pGMM[iLocal] = pGMM[iLocal-1];
                        pGMM[iLocal-1] = temp;
                        pGauss--;
                    }
                }
                //belongs to the mode - bFitsPDF becomes 1
                /////
            }
        }//!bFitsPDF)

        //check prune
        if (weight<-m_fPrune)
        {
            weight=0.0;
            nModes--;
        }

        pGauss->weight=weight;//update weight by the calculated value
        totalWeight+=weight;
    }
    //go through all modes
    //////

    //renormalize weights
    for (iMode = 0; iMode < nModes; iMode++)
    {
        pGMM[iMode].weight = pGMM[iMode].weight/totalWeight;
    }
    
    //make new mode if needed and exit
    if (!bFitsPDF)
    {
        if (nModes==m_nM)
        {
           //replace the weakest
            pGauss=pGMM+m_nM-1;
        }
        else
        {
           //add a new one
            pGauss=pGMM+nModes;
            nModes++;
        }

        if (nModes==1)
        {
            pGauss->weight=1;
        }
        else
        {
            pGauss->weight=m_fAlphaT;

            //renormalize all weights
            for (iMode = 0; iMode < nModes-1; iMode++)
            {
                pGMM[iMode].weight *=m_fOneMinAlpha;
            }
        }

        //init 
        memcpy(pGauss->mean,data,nD*sizeof(float));
        pGauss->variance=m_fVarInit;

        //sort
        //find the new place for it
        for (int iLocal = nModes-1;iLocal>0;iLocal--)
        {
                    //check one up
                    if (m_fAlphaT < (pGMM[iLocal-1].weight))
                    {
                        break;
                    }
                    else
                    {
                        //swap one up
                        CvPBGMMGaussian temp = pGMM[iLocal];
                        pGMM[iLocal] = pGMM[iLocal-1];
                        pGMM[iLocal-1] = temp;
                    }
        }
    }

    //set the number of modes
    *pModesUsed=nModes;

    return bBackground;
}

// a bit more efficient implementation for common case of 3 channel (rgb) images
CV_INLINE int _icvUpdateGMM_C3(float r,float g, float b,
                                unsigned char* pModesUsed, 
                                CvPBGMMGaussian* pGMM,
                                int m_nM,
                                float m_fAlphaT,
                                float m_fTb,
                                float m_fTB,    
                                float m_fTg,
                                float m_fVarInit,
                                float m_fVarMax,
                                float m_fVarMin,
                                float m_fPrune)
{
    //calculate distances to the modes (+ sort)
    //here we need to go in descending order!!! 
    bool bBackground=0;//return value -> true - the pixel classified as background

    //internal:
    bool bFitsPDF=0;//if it remains zero a new GMM mode will be added   
    float m_fOneMinAlpha=1-m_fAlphaT;
    unsigned char nModes=*pModesUsed;//current number of modes in GMM
    float totalWeight=0.0f;

    //////
    //go through all modes
    int iMode=0;
    CvPBGMMGaussian* pGauss=pGMM;
    for (;iMode<nModes;iMode++,pGauss++)
    {
        float weight = pGauss->weight;//need only weight if fit is found
        weight=m_fOneMinAlpha*weight+m_fPrune;

        ////
        //fit not found yet
        if (!bFitsPDF)
        {
            //check if it belongs to some of the remaining modes
            float var=pGauss->variance;

            //calculate difference and distance
            float muR = pGauss->mean[0];
            float muG = pGauss->mean[1];
            float muB = pGauss->mean[2];
        
            float dR=muR - r;
            float dG=muG - g;
            float dB=muB - b;

            float dist2=(dR*dR+dG*dG+dB*dB);        
            
            //background? - m_fTb - usually larger than m_fTg
            if ((totalWeight<m_fTB)&&(dist2<m_fTb*var))
                    bBackground=1;

            //check fit
            if (dist2<m_fTg*var)
            {
                /////
                //belongs to the mode - bFitsPDF becomes 1
                bFitsPDF=1;

                //update distribution               
                
                //update weight
                weight+=m_fAlphaT;
                
                float k = m_fAlphaT/weight;

                //update mean
                pGauss->mean[0] = muR - k*(dR);
                pGauss->mean[1] = muG - k*(dG);
                pGauss->mean[2] = muB - k*(dB);

                //update variance
                float varnew = var + k*(dist2-var);
                //limit the variance                
                pGauss->variance = MIN(m_fVarMax,MAX(varnew,m_fVarMin));

                //sort
                //all other weights are at the same place and 
                //only the matched (iModes) is higher -> just find the new place for it             
                for (int iLocal = iMode;iLocal>0;iLocal--)
                {
                    //check one up
                    if (weight < (pGMM[iLocal-1].weight))
                    {
                        break;
                    }
                    else
                    {
                        //swap one up
                        CvPBGMMGaussian temp = pGMM[iLocal];
                        pGMM[iLocal] = pGMM[iLocal-1];
                        pGMM[iLocal-1] = temp;
                        pGauss--;
                    }
                }
                //belongs to the mode - bFitsPDF becomes 1
                /////
            }   

        }//!bFitsPDF)
        
        //check prunning
        if (weight<-m_fPrune)
        {
                    weight=0.0;
                    nModes--;
        }

        pGauss->weight=weight;
        totalWeight+=weight;
    }
    //go through all modes
    //////

    //renormalize weights
    for (iMode = 0; iMode < nModes; iMode++)
    {
        pGMM[iMode].weight = pGMM[iMode].weight/totalWeight;
    }
    
    //make new mode if needed and exit
    if (!bFitsPDF)
    {
        if (nModes==m_nM)
        {
           //replace the weakest
            pGauss=pGMM+m_nM-1;
        }
        else
        {
           //add a new one
            pGauss=pGMM+nModes;
            nModes++;
        }

        if (nModes==1)
        {
            pGauss->weight=1;
        }
        else
        {
            pGauss->weight=m_fAlphaT;

            //renormalize all weights
            for (iMode = 0; iMode < nModes-1; iMode++)
            {
                pGMM[iMode].weight *=m_fOneMinAlpha;
            }
        }

        //init 
        pGauss->mean[0]=r;
        pGauss->mean[1]=g;
        pGauss->mean[2]=b;

        pGauss->variance=m_fVarInit;

        //sort
        //find the new place for it
        for (int iLocal = nModes-1;iLocal>0;iLocal--)
        {
                    //check one up
                    if (m_fAlphaT < (pGMM[iLocal-1].weight))
                    {
                        break;
                    }
                    else
                    {
                        //swap one up
                        CvPBGMMGaussian temp = pGMM[iLocal];
                        pGMM[iLocal] = pGMM[iLocal-1];
                        pGMM[iLocal-1] = temp;
                    }
        }
    }

    //set the number of modes
    *pModesUsed=nModes;

    return bBackground;
}

//the main function to update the background model
void icvUpdatePixelBackgroundGMM2( const CvArr* srcarr, CvArr* dstarr ,
                                         CvPBGMMGaussian *pGMM,
                                         unsigned char *pUsedModes,
                                         //CvGaussBGStatModel2Params* pGMMPar,
                                         int nM,
                                         float fTb, 
                                         float fTB, 
                                         float fTg, 
                                         float fVarInit,
                                         float fVarMax,
                                         float fVarMin,
                                         float fCT,
                                         float fTau,
                                         bool bShadowDetection,
                                         unsigned char  nShadowDetection,
                                         float alpha)
{
    CvMat sstub, *src = cvGetMat(srcarr, &sstub);
    CvMat dstub, *dst = cvGetMat(dstarr, &dstub);
    CvSize size = cvGetMatSize(src);
    int nD=CV_MAT_CN(src->type);

    //reshape if possible
    if( CV_IS_MAT_CONT(src->type & dst->type) )
    {
        size.width *= size.height;
        size.height = 1;
    }

    int x, y;
    float data[CV_BGFG_MOG2_NDMAX];
    float prune=-alpha*fCT;

    //general nD

    if (nD!=3)
    {
    switch (CV_MAT_DEPTH(src->type))
    {
    case CV_8U:
        for( y = 0; y < size.height; y++ )
        {
            uchar* sptr = src->data.ptr + src->step*y;
            uchar* pDataOutput = dst->data.ptr + dst->step*y;
            for( x = 0; x < size.width; x++,
                pGMM+=nM,pUsedModes++,pDataOutput++,sptr+=nD)
            {
                //convert data
                for (int iD=0;iD<nD;iD++) data[iD]=float(sptr[iD]);
                //update GMM model
                int result = _icvUpdateGMM(data,nD,pUsedModes,pGMM,nM,alpha, fTb, fTB, fTg, fVarInit, fVarMax, fVarMin,prune);
                //detect shadows in the foreground
                if (bShadowDetection)
                    if (result==0) result= _icvRemoveShadowGMM(data,nD,(*pUsedModes),pGMM,fTb,fTB,fTau);
                //generate output
                (* pDataOutput)= (result==1) ? 0 : (result==2) ? (nShadowDetection) : 255; 
            }
        }
        break;
    case CV_16S:
        for( y = 0; y < size.height; y++ )
        {
            short* sptr = src->data.s + src->step*y;
            uchar* pDataOutput = dst->data.ptr + dst->step*y;
            for( x = 0; x < size.width; x++,
                pGMM+=nM,pUsedModes++,pDataOutput++,sptr+=nD)
            {
                //convert data
                for (int iD=0;iD<nD;iD++) data[iD]=float(sptr[iD]);
                //update GMM model
                int result = _icvUpdateGMM(data,nD,pUsedModes,pGMM,nM,alpha, fTb, fTB, fTg, fVarInit, fVarMax, fVarMin,prune);
                //detect shadows in the foreground
                if (bShadowDetection)
                    if (result==0) result= _icvRemoveShadowGMM(data,nD,(*pUsedModes),pGMM,fTb,fTB,fTau);
                //generate output
                (* pDataOutput)= (result==1) ? 0 : (result==2) ? (nShadowDetection) : 255; 
            }
        }
        break;
    case CV_16U:
        for( y = 0; y < size.height; y++ )
        {
            unsigned short* sptr = (unsigned short*) (src->data.s + src->step*y);
            uchar* pDataOutput = dst->data.ptr + dst->step*y;
            for( x = 0; x < size.width; x++,
                pGMM+=nM,pUsedModes++,pDataOutput++,sptr+=nD)
            {
                //convert data
                for (int iD=0;iD<nD;iD++) data[iD]=float(sptr[iD]);
                //update GMM model
                int result = _icvUpdateGMM(data,nD,pUsedModes,pGMM,nM,alpha, fTb, fTB, fTg, fVarInit, fVarMax, fVarMin,prune);
                //detect shadows in the foreground
                if (bShadowDetection)
                    if (result==0) result= _icvRemoveShadowGMM(data,nD,(*pUsedModes),pGMM,fTb,fTB,fTau);
                //generate output
                (* pDataOutput)= (result==1) ? 0 : (result==2) ? (nShadowDetection) : 255; 
            }
        }
        break;
    case CV_32S:
        for( y = 0; y < size.height; y++ )
        {
            int* sptr = src->data.i + src->step*y;
            uchar* pDataOutput = dst->data.ptr + dst->step*y;
            for( x = 0; x < size.width; x++,
                pGMM+=nM,pUsedModes++,pDataOutput++,sptr+=nD)
            {
                //convert data
                for (int iD=0;iD<nD;iD++) data[iD]=float(sptr[iD]);
                //update GMM model
                int result = _icvUpdateGMM(data,nD,pUsedModes,pGMM,nM,alpha, fTb, fTB, fTg, fVarInit, fVarMax, fVarMin,prune);
                //detect shadows in the foreground
                if (bShadowDetection)
                    if (result==0) result= _icvRemoveShadowGMM(data,nD,(*pUsedModes),pGMM,fTb,fTB,fTau);
                //generate output
                (* pDataOutput)= (result==1) ? 0 : (result==2) ? (nShadowDetection) : 255; 
            }
        }
        break;
    case CV_32F:
        for( y = 0; y < size.height; y++ )
        {
            float* sptr = src->data.fl + src->step*y;
            uchar* pDataOutput = dst->data.ptr + dst->step*y;
            for( x = 0; x < size.width; x++,
                pGMM+=nM,pUsedModes++,pDataOutput++,sptr+=nD)
            {
                //update GMM model
                int result = _icvUpdateGMM(sptr,nD,pUsedModes,pGMM,nM,alpha, fTb, fTB, fTg, fVarInit, fVarMax, fVarMin,prune);
                //detect shadows in the foreground
                if (bShadowDetection)
                    if (result==0) result= _icvRemoveShadowGMM(data,nD,(*pUsedModes),pGMM,fTb,fTB,fTau);
                //generate output
                (* pDataOutput)= (result==1) ? 0 : (result==2) ? (nShadowDetection) : 255; 
            }
        }
        break;
    case CV_64F:
        for( y = 0; y < size.height; y++ )
        {
            double* sptr = src->data.db + src->step*y;
            uchar* pDataOutput = dst->data.ptr + dst->step*y;
            for( x = 0; x < size.width; x++,
                pGMM+=nM,pUsedModes++,pDataOutput++,sptr+=nD)
            {
                //convert data
                for (int iD=0;iD<nD;iD++) data[iD]=float(sptr[iD]);
                //update GMM model
                int result = _icvUpdateGMM(data,nD,pUsedModes,pGMM,nM,alpha, fTb, fTB, fTg, fVarInit, fVarMax, fVarMin,prune);
                //detect shadows in the foreground
                if (bShadowDetection)
                    if (result==0) result= _icvRemoveShadowGMM(data,nD,(*pUsedModes),pGMM,fTb,fTB,fTau);
                //generate output
                (* pDataOutput)= (result==1) ? 0 : (result==2) ? (nShadowDetection) : 255; 
            }
        }
        break;
    }
    }else ///if (nD==3) - a bit faster
    {
    switch (CV_MAT_DEPTH(src->type))
    {
    case CV_8U:
        for( y = 0; y < size.height; y++ )
        {
            uchar* sptr = src->data.ptr + src->step*y;
            uchar* pDataOutput = dst->data.ptr + dst->step*y;
            for( x = 0; x < size.width; x++,
                pGMM+=nM,pUsedModes++,pDataOutput++,sptr+=nD)
            {
                //convert data
                data[0]=float(sptr[0]),data[1]=float(sptr[1]),data[2]=float(sptr[2]);
                //update GMM model
                int result = _icvUpdateGMM_C3(data[0],data[1],data[2],pUsedModes,pGMM,nM,alpha, fTb, fTB, fTg, fVarInit, fVarMax, fVarMin,prune);
                //detect shadows in the foreground
                if (bShadowDetection)
                    if (result==0) result= _icvRemoveShadowGMM(data,nD,(*pUsedModes),pGMM,fTb,fTB,fTau);
                //generate output
                (* pDataOutput)= (result==1) ? 0 : (result==2) ? (nShadowDetection) : 255; 
            }
        }
        break;
    case CV_16S:
        for( y = 0; y < size.height; y++ )
        {
            short* sptr = src->data.s + src->step*y;
            uchar* pDataOutput = dst->data.ptr + dst->step*y;
            for( x = 0; x < size.width; x++,
                pGMM+=nM,pUsedModes++,pDataOutput++,sptr+=nD)
            {
                //convert data
                data[0]=float(sptr[0]),data[1]=float(sptr[1]),data[2]=float(sptr[2]);
                //update GMM model
                int result = _icvUpdateGMM_C3(data[0],data[1],data[2],pUsedModes,pGMM,nM,alpha, fTb, fTB, fTg, fVarInit, fVarMax, fVarMin,prune);
                //detect shadows in the foreground
                if (bShadowDetection)
                    if (result==0) result= _icvRemoveShadowGMM(data,nD,(*pUsedModes),pGMM,fTb,fTB,fTau);
                //generate output
                (* pDataOutput)= (result==1) ? 0 : (result==2) ? (nShadowDetection) : 255; 
            }
        }
        break;
    case CV_16U:
        for( y = 0; y < size.height; y++ )
        {
            unsigned short* sptr = (unsigned short*) src->data.s + src->step*y;
            uchar* pDataOutput = dst->data.ptr + dst->step*y;
            for( x = 0; x < size.width; x++,
                pGMM+=nM,pUsedModes++,pDataOutput++,sptr+=nD)
            {
                //convert data
                data[0]=float(sptr[0]),data[1]=float(sptr[1]),data[2]=float(sptr[2]);
                //update GMM model
                int result = _icvUpdateGMM_C3(data[0],data[1],data[2],pUsedModes,pGMM,nM,alpha, fTb, fTB, fTg, fVarInit, fVarMax, fVarMin,prune);
                //detect shadows in the foreground
                if (bShadowDetection)
                    if (result==0) result= _icvRemoveShadowGMM(data,nD,(*pUsedModes),pGMM,fTb,fTB,fTau);
                //generate output
                (* pDataOutput)= (result==1) ? 0 : (result==2) ? (nShadowDetection) : 255; 
            }
        }
        break;
    case CV_32S:
        for( y = 0; y < size.height; y++ )
        {
            int* sptr = src->data.i + src->step*y;
            uchar* pDataOutput = dst->data.ptr + dst->step*y;
            for( x = 0; x < size.width; x++,
                pGMM+=nM,pUsedModes++,pDataOutput++,sptr+=nD)
            {
                //convert data
                data[0]=float(sptr[0]),data[1]=float(sptr[1]),data[2]=float(sptr[2]);
                //update GMM model
                int result = _icvUpdateGMM_C3(data[0],data[1],data[2],pUsedModes,pGMM,nM,alpha, fTb, fTB, fTg, fVarInit, fVarMax, fVarMin,prune);
                //detect shadows in the foreground
                if (bShadowDetection)
                    if (result==0) result= _icvRemoveShadowGMM(data,nD,(*pUsedModes),pGMM,fTb,fTB,fTau);
                //generate output
                (* pDataOutput)= (result==1) ? 0 : (result==2) ? (nShadowDetection) : 255; 
            }
        }
        break;
    case CV_32F:
        for( y = 0; y < size.height; y++ )
        {
            float* sptr = src->data.fl + src->step*y;
            uchar* pDataOutput = dst->data.ptr + dst->step*y;
            for( x = 0; x < size.width; x++,
                pGMM+=nM,pUsedModes++,pDataOutput++,sptr+=nD)
            {
                //update GMM model
                int result = _icvUpdateGMM_C3(sptr[0],sptr[1],sptr[2],pUsedModes,pGMM,nM,alpha, fTb, fTB, fTg, fVarInit, fVarMax, fVarMin,prune);
                //detect shadows in the foreground
                if (bShadowDetection)
                    if (result==0) result= _icvRemoveShadowGMM(data,nD,(*pUsedModes),pGMM,fTb,fTB,fTau);
                //generate output
                (* pDataOutput)= (result==1) ? 0 : (result==2) ? (nShadowDetection) : 255; 
            }
        }
        break;
    case CV_64F:
        for( y = 0; y < size.height; y++ )
        {
            double* sptr = src->data.db + src->step*y;
            uchar* pDataOutput = dst->data.ptr + dst->step*y;
            for( x = 0; x < size.width; x++,
                pGMM+=nM,pUsedModes++,pDataOutput++,sptr+=nD)
            {
                //convert data
                data[0]=float(sptr[0]),data[1]=float(sptr[1]),data[2]=float(sptr[2]);
                //update GMM model
                int result = _icvUpdateGMM_C3(data[0],data[1],data[2],pUsedModes,pGMM,nM,alpha, fTb, fTB, fTg, fVarInit, fVarMax, fVarMin,prune);
                //detect shadows in the foreground
                if (bShadowDetection)
                    if (result==0) result= _icvRemoveShadowGMM(data,nD,(*pUsedModes),pGMM,fTb,fTB,fTau);
                //generate output
                (* pDataOutput)= (result==1) ? 0 : (result==2) ? (nShadowDetection) : 255; 
            }
        }
        break;
    }
    }//a bit faster for nD=3; 
}


namespace cv
{

static const int defaultHistory2 = CV_BGFG_MOG2_WINDOW_SIZE;
static const float defaultVarThreshold2 = CV_BGFG_MOG2_STD_THRESHOLD*CV_BGFG_MOG2_STD_THRESHOLD;
static const int defaultNMixtures2 = CV_BGFG_MOG2_NGAUSSIANS;
static const float defaultBackgroundRatio2 = CV_BGFG_MOG2_BACKGROUND_THRESHOLD;
static const float defaultVarThresholdGen2 = CV_BGFG_MOG2_STD_THRESHOLD_GENERATE*CV_BGFG_MOG2_STD_THRESHOLD_GENERATE;
static const float defaultVarInit2 = CV_BGFG_MOG2_VAR_INIT;
static const float defaultVarMax2 = CV_BGFG_MOG2_VAR_MAX;
static const float defaultVarMin2 = CV_BGFG_MOG2_VAR_MIN;
static const float defaultfCT2 = CV_BGFG_MOG2_CT;
static const unsigned char defaultnShadowDetection2 = (unsigned char)CV_BGFG_MOG2_SHADOW_VALUE;
static const float defaultfTau = CV_BGFG_MOG2_SHADOW_TAU;


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
    CV_Assert( nchannels <= CV_BGFG_MOG2_NDMAX );
    
    // for each gaussian mixture of each pixel bg model we store ...
    // the mixture weight (w),
    // the mean (nchannels values) and
    // the covariance
    bgmodel.create( 1, frameSize.height*frameSize.width*nmixtures*(2 + CV_BGFG_MOG2_NDMAX), CV_32F );
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
    CvMat _cimage = image, _cfgmask = fgmask;
    
    if (learningRate > 0) 
        icvUpdatePixelBackgroundGMM2( &_cimage, &_cfgmask, 
        (CvPBGMMGaussian*) bgmodel.data,
        bgmodelUsedModes.data,
        nmixtures,//nM
        varThreshold,//fTb
        backgroundRatio,//fTB
        varThresholdGen,//fTg,
        fVarInit,
        fVarMax,
        fVarMin,
        fCT,
        fTau,
        bShadowDetection,
        nShadowDetection,
        float(learningRate));
}

void BackgroundSubtractorMOG2::getBackgroundImage(OutputArray backgroundImage) const
{
#if _MSC_VER >= 1200
    #pragma warning( push )
    #pragma warning( disable : 4127 )  
#endif
    CV_Assert(CV_BGFG_MOG2_NDMAX == 3);
#if _MSC_VER >= 1200
    #pragma warning( pop )
#endif
    Mat meanBackground(frameSize, CV_8UC3, Scalar::all(0));

    int firstGaussianIdx = 0;
    CvPBGMMGaussian* pGMM = (CvPBGMMGaussian*)bgmodel.data;
    for(int row=0; row<meanBackground.rows; row++)
    {
        for(int col=0; col<meanBackground.cols; col++)
        {
            int nModes = static_cast<int>(bgmodelUsedModes.at<uchar>(row, col));
            double meanVal[CV_BGFG_MOG2_NDMAX] = {0.0, 0.0, 0.0};

            double totalWeight = 0.0;
            for(int gaussianIdx = firstGaussianIdx; gaussianIdx < firstGaussianIdx + nModes; gaussianIdx++)
            {
                CvPBGMMGaussian gaussian = pGMM[gaussianIdx];
                totalWeight += gaussian.weight;

                for(int chIdx = 0; chIdx < CV_BGFG_MOG2_NDMAX; chIdx++)
                {
                    meanVal[chIdx] += gaussian.weight * gaussian.mean[chIdx];
                }

                if(totalWeight > backgroundRatio)
                    break;
            }

            Vec3f val = Vec3f((float)meanVal[0], (float)meanVal[1], (float)meanVal[2]) * (float)(1.0 / totalWeight);
            meanBackground.at<Vec3b>(row, col) = Vec3b(val);
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
