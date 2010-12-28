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
//Date: 27-April-2005, Version:0.9
///////////*/

#include "cvaux.h"
#include "cvaux_mog2.h"

int _icvRemoveShadowGMM(long posPixel, 
								float red, float green, float blue, 
								unsigned char nModes, 
								CvPBGMMGaussian* m_aGaussians,
								float m_fTb,
								float m_fTB,	
								float m_fTau)
{
	//calculate distances to the modes (+ sort???)
	//here we need to go in descending order!!!
	long pos;
	float tWeight = 0;
	float numerator, denominator;
	// check all the distributions, marked as background:
	for (int iModes=0;iModes<nModes;iModes++)
	{
		pos=posPixel+iModes;
		float var = m_aGaussians[pos].sigma;
		float muR = m_aGaussians[pos].muR;
		float muG = m_aGaussians[pos].muG;
		float muB = m_aGaussians[pos].muB;
		float weight = m_aGaussians[pos].weight;
		tWeight += weight;
		
		numerator = red * muR + green * muG + blue * muB;
		denominator = muR * muR + muG * muG + muB * muB;
		// no division by zero allowed
		if (denominator == 0)
		{
				break;
		};
		float a = numerator / denominator;

		// if tau < a < 1 then also check the color distortion
		if ((a <= 1) && (a >= m_fTau))//m_nBeta=1
		{
			float dR=a * muR - red;
			float dG=a * muG - green;
			float dB=a * muB - blue;

			//square distance -slower and less accurate
			//float maxDistance = cvSqrt(m_fTb*var);
			//if ((fabs(dR) <= maxDistance) && (fabs(dG) <= maxDistance) && (fabs(dB) <= maxDistance))
			//circle
			float dist=(dR*dR+dG*dG+dB*dB);
			if (dist<m_fTb*var*a*a)
			{
				return 2;
			}
		};
		if (tWeight > m_fTB)
		{
				break;
		};
	};
	return 0;
}

int _icvUpdatePixelBackgroundGMM(long posPixel, 
								float red, float green, float blue, 
								unsigned char* pModesUsed, 
								CvPBGMMGaussian* m_aGaussians,
								int m_nM,
								float m_fAlphaT,
								float m_fTb,
								float m_fTB,	
								float m_fTg,
								float m_fSigma,
								float m_fPrune)
{
	//calculate distances to the modes (+ sort???)
	//here we need to go in descending order!!!
	
	long pos;

	
	bool bFitsPDF=0;
	bool bBackground=0;

	float m_fOneMinAlpha=1-m_fAlphaT;

	unsigned char nModes=*pModesUsed;
	float totalWeight=0.0f;

	//////
	//go through all modes
	for (int iModes=0;iModes<nModes;iModes++)
	{
		pos=posPixel+iModes;
		float weight = m_aGaussians[pos].weight;

		////
		//fit not found yet
		if (!bFitsPDF)
		{
			//check if it belongs to some of the modes
			//calculate distance
			float var = m_aGaussians[pos].sigma;
			float muR = m_aGaussians[pos].muR;
			float muG = m_aGaussians[pos].muG;
			float muB = m_aGaussians[pos].muB;
		
			float dR=muR - red;
			float dG=muG - green;
			float dB=muB - blue;

			///////
			//check if it fits the current mode (Factor * sigma)
			
			//square distance -slower and less accurate
			//float maxDistance = cvSqrt(m_fTg*var);
			//if ((fabs(dR) <= maxDistance) && (fabs(dG) <= maxDistance) && (fabs(dB) <= maxDistance))
			//circle
			float dist=(dR*dR+dG*dG+dB*dB);
			//background? - m_fTb
			if ((totalWeight<m_fTB)&&(dist<m_fTb*var))
					bBackground=1;
			//check fit
			if (dist<m_fTg*var)
			{
				/////
				//belongs to the mode
				bFitsPDF=1;

				//update distribution
				float k = m_fAlphaT/weight;
				weight=m_fOneMinAlpha*weight+m_fPrune;
				weight+=m_fAlphaT;
				m_aGaussians[pos].muR = muR - k*(dR);
				m_aGaussians[pos].muG = muG - k*(dG);
				m_aGaussians[pos].muB = muB - k*(dB);

				//limit update speed for cov matrice
				//not needed
				//k=k>20*m_fAlphaT?20*m_fAlphaT:k;
				//float sigmanew = var + k*((0.33*(dR*dR+dG*dG+dB*dB))-var);
				//float sigmanew = var + k*((dR*dR+dG*dG+dB*dB)-var);
				//float sigmanew = var + k*((0.33*dist)-var);
				float sigmanew = var + k*(dist-var);
				//limit the variance
				//m_aGaussians[pos].sigma = sigmanew>70?70:sigmanew;
				//m_aGaussians[pos].sigma = sigmanew>5*m_fSigma?5*m_fSigma:sigmanew;
				m_aGaussians[pos].sigma =sigmanew< 4 ? 4 : sigmanew>5*m_fSigma?5*m_fSigma:sigmanew;
				//m_aGaussians[pos].sigma =sigmanew< 4 ? 4 : sigmanew>3*m_fSigma?3*m_fSigma:sigmanew;
				//m_aGaussians[pos].sigma = m_fSigma;
				//sort
				//all other weights are at the same place and 
				//only the matched (iModes) is higher -> just find the new place for it
				for (int iLocal = iModes;iLocal>0;iLocal--)
				{
					long posLocal=posPixel + iLocal;
					if (weight < (m_aGaussians[posLocal-1].weight))
					{
						break;
					}
					else
					{
						//swap
						CvPBGMMGaussian temp = m_aGaussians[posLocal];
						m_aGaussians[posLocal] = m_aGaussians[posLocal-1];
						m_aGaussians[posLocal-1] = temp;
					}
				}

				//belongs to the mode
				/////
			}
			else
			{
				weight=m_fOneMinAlpha*weight+m_fPrune;
				//check prune
				if (weight<-m_fPrune)
				{
					weight=0.0;
					nModes--;
				//	bPrune=1;
					//break;//the components are sorted so we can skip the rest
				}
			}
			//check if it fits the current mode (2.5 sigma)
			///////
		}
		//fit not found yet
		/////
		else
		{
				weight=m_fOneMinAlpha*weight+m_fPrune;
				//check prune
				if (weight<-m_fPrune)
				{
					weight=0.0;
					nModes--;
				}
		}
		totalWeight+=weight;
		m_aGaussians[pos].weight=weight;
	}
	//go through all modes
	//////

	//renormalize weights
	for (int iLocal = 0; iLocal < nModes; iLocal++)
	{
		m_aGaussians[posPixel+ iLocal].weight = m_aGaussians[posPixel+ iLocal].weight/totalWeight;
	}
	
	//make new mode if needed and exit
	if (!bFitsPDF)
	{
		if (nModes==m_nM)
		{
           //replace the weakest
		}
		else
		{
           //add a new one
			nModes++;
		}
		pos=posPixel+nModes-1;

      	if (nModes==1)
			m_aGaussians[pos].weight=1;
		else
			m_aGaussians[pos].weight=m_fAlphaT;

		//renormalize weights
		int iLocal;
		for (iLocal = 0; iLocal < nModes-1; iLocal++)
		{
			m_aGaussians[posPixel+ iLocal].weight *=m_fOneMinAlpha;
		}

		m_aGaussians[pos].muR=red;
		m_aGaussians[pos].muG=green;
		m_aGaussians[pos].muB=blue;
		m_aGaussians[pos].sigma=m_fSigma;

		//sort
		//find the new place for it
		for (iLocal = nModes-1;iLocal>0;iLocal--)
		{
			long posLocal=posPixel + iLocal;
			if (m_fAlphaT < (m_aGaussians[posLocal-1].weight))
			{
						break;
			}
			else
			{
				//swap
				CvPBGMMGaussian temp = m_aGaussians[posLocal];
				m_aGaussians[posLocal] = m_aGaussians[posLocal-1];
				m_aGaussians[posLocal-1] = temp;
			}
		}
	}

	//set the number of modes
	*pModesUsed=nModes;

    return bBackground;
}

void _icvReplacePixelBackgroundGMM(long pos, 
								unsigned char* pData, 
								CvPBGMMGaussian* m_aGaussians)
{
	pData[0]=(unsigned char) m_aGaussians[pos].muR;
	pData[1]=(unsigned char) m_aGaussians[pos].muG;
	pData[2]=(unsigned char) m_aGaussians[pos].muB;
}


void icvUpdatePixelBackgroundGMM(CvGaussBGStatModel2Data* pGMMData,CvGaussBGStatModel2Params* pGMM, float m_fAlphaT, unsigned char* data,unsigned char* output)
{
	int size=pGMMData->nSize;
	unsigned char* pDataCurrent=data;
	unsigned char* pUsedModes=pGMMData->rnUsedModes;
	unsigned char* pDataOutput=output;
	//some constants
	int m_nM=pGMM->nM;
	//float m_fAlphaT=pGMM->fAlphaT;

	float m_fTb=pGMM->fTb;//Tb - threshold on the Mahalan. dist.
	float m_fTB=pGMM->fTB;//1-TF from the paper
	float m_fTg=pGMM->fTg;//Tg - when to generate a new component
	float m_fSigma=pGMM->fSigma;//initial sigma
	float m_fCT=pGMM->fCT;//CT - complexity reduction prior 
	float m_fPrune=-m_fAlphaT*m_fCT;
	float m_fTau=pGMM->fTau;
	CvPBGMMGaussian* m_aGaussians=pGMMData->rGMM;
	long posPixel=0;
	bool m_bShadowDetection=pGMM->bShadowDetection;
	unsigned char m_nShadowDetection=pGMM->nShadowDetection;

	//go through the image
	for (int i=0;i<size;i++)
	{
		// retrieve the colors
		float red = pDataCurrent[0];
		float green = pDataCurrent[1];
		float blue = pDataCurrent[2];
		
		//update model+ background subtract
		int result = _icvUpdatePixelBackgroundGMM(posPixel, red, green, blue,pUsedModes,m_aGaussians,
			m_nM,m_fAlphaT, m_fTb, m_fTB, m_fTg, m_fSigma, m_fPrune);
		unsigned char nMLocal=*pUsedModes;
		
		if (m_bShadowDetection)
				if (!result)
				{
					result= _icvRemoveShadowGMM(posPixel, red, green, blue,nMLocal,m_aGaussians,
								m_fTb,
								m_fTB,	
								m_fTau);
				}

		
		switch (result)
		{
			case 0:
				//foreground
				(* pDataOutput)=255;
				if (pGMM->bRemoveForeground) 
				{
					_icvReplacePixelBackgroundGMM(posPixel,pDataCurrent,m_aGaussians);
				}
				break;
			case 1:
				//background
				(* pDataOutput)=0;
				break;
			case 2:
				//shadow
				(* pDataOutput)=m_nShadowDetection;
				if (pGMM->bRemoveForeground) 
				{
					_icvReplacePixelBackgroundGMM(posPixel,pDataCurrent,m_aGaussians);
				}

				break;
		}
		posPixel+=m_nM;
		pDataCurrent+=3;
		pDataOutput++;
		pUsedModes++;
	}
}

//////////////////////////////////////////////
//implementation as part of the CvBGStatModel
static void CV_CDECL icvReleaseGaussianBGModel2( CvGaussBGModel2** bg_model );
static int CV_CDECL icvUpdateGaussianBGModel2( IplImage* curr_frame, CvGaussBGModel2*  bg_model );


CV_IMPL CvBGStatModel*
cvCreateGaussianBGModel2( IplImage* first_frame, CvGaussBGStatModel2Params* parameters )
{
    CvGaussBGModel2* bg_model = 0;
	int w,h,size;
    
    CV_FUNCNAME( "cvCreateGaussianBGModel2" );
    
    __BEGIN__;

	CvGaussBGStatModel2Params params;
    
    if( !CV_IS_IMAGE(first_frame) )
        CV_ERROR( CV_StsBadArg, "Invalid or NULL first_frame parameter" );

	if( !(first_frame->nChannels==3) )
        CV_ERROR( CV_StsBadArg, "Need three channel image (RGB)" );

	CV_CALL( bg_model = (CvGaussBGModel2*)cvAlloc( sizeof(*bg_model) ));
    memset( bg_model, 0, sizeof(*bg_model) );
    bg_model->type = CV_BG_MODEL_MOG2;
    bg_model->release = (CvReleaseBGStatModel)icvReleaseGaussianBGModel2;
    bg_model->update = (CvUpdateBGStatModel)icvUpdateGaussianBGModel2;

    //init parameters	
    if( parameters == NULL )
      {                        
		/* These constants are defined in cvaux/include/cvaux.h: */
		params.bRemoveForeground=0;
		params.bShadowDetection = 1;
		params.bPostFiltering=0;
		params.minArea=CV_BGFG_MOG2_MINAREA;

		//set parameters
		// K - max number of Gaussians per pixel
		params.nM = CV_BGFG_MOG2_NGAUSSIANS;//4;			
		// Tb - the threshold - n var
		//pGMM->fTb = 4*4;
		params.fTb = CV_BGFG_MOG2_STD_THRESHOLD*CV_BGFG_MOG2_STD_THRESHOLD;
		// Tbf - the threshold
		//pGMM->fTB = 0.9f;//1-cf from the paper 
		params.fTB = CV_BGFG_MOG2_BACKGROUND_THRESHOLD;
		// Tgenerate - the threshold
		params.fTg = CV_BGFG_MOG2_STD_THRESHOLD_GENERATE*CV_BGFG_MOG2_STD_THRESHOLD_GENERATE;//update the mode or generate new
		//pGMM->fSigma= 11.0f;//sigma for the new mode
		params.fSigma= CV_BGFG_MOG2_SIGMA_INIT;
		// alpha - the learning factor
		params.fAlphaT=1.0f/CV_BGFG_MOG2_WINDOW_SIZE;//0.003f;
		// complexity reduction prior constant
		params.fCT=CV_BGFG_MOG2_CT;//0.05f;

		//shadow
		// Shadow detection
		params.nShadowDetection = CV_BGFG_MOG2_SHADOW_VALUE;//value 0 to turn off
		params.fTau = CV_BGFG_MOG2_SHADOW_TAU;//0.5f;// Tau - shadow threshold
    }
    else
    {
        params = *parameters;
    }

	bg_model->params = params;

	//allocate GMM data
	w=first_frame->width;
	h=first_frame->height;
	size=w*h;

	bg_model->data.nWidth=w;
	bg_model->data.nHeight=h;
	bg_model->data.nNBands=3;
	bg_model->data.nSize=size;

	//GMM for each pixel
	bg_model->data.rGMM=(CvPBGMMGaussian*) malloc(size * params.nM * sizeof(CvPBGMMGaussian));
	//used modes per pixel
	bg_model->data.rnUsedModes = (unsigned char* ) malloc(size);
	memset(bg_model->data.rnUsedModes,0,size);//no modes used
  
    //prepare storages    
    CV_CALL( bg_model->background = cvCreateImage(cvSize(w,h), IPL_DEPTH_8U, first_frame->nChannels));
    CV_CALL( bg_model->foreground = cvCreateImage(cvSize(w,h), IPL_DEPTH_8U, 1));
    
	//for eventual filtering
    CV_CALL( bg_model->storage = cvCreateMemStorage());
	
	bg_model->countFrames = 0;

    __END__;
    
    if( cvGetErrStatus() < 0 )
    {
        CvBGStatModel* base_ptr = (CvBGStatModel*)bg_model;
        
        if( bg_model && bg_model->release )
            bg_model->release( &base_ptr );
        else
            cvFree( &bg_model );
        bg_model = 0;
    }
    
    return (CvBGStatModel*)bg_model;
}


static void CV_CDECL
icvReleaseGaussianBGModel2( CvGaussBGModel2** _bg_model )
{
    CV_FUNCNAME( "icvReleaseGaussianBGModel2" );

    __BEGIN__;
    
    if( !_bg_model )
        CV_ERROR( CV_StsNullPtr, "" );

    if( *_bg_model )
    {
        CvGaussBGModel2* bg_model = *_bg_model;

		free (bg_model->data.rGMM);
		free (bg_model->data.rnUsedModes);

        cvReleaseImage( &bg_model->background );
        cvReleaseImage( &bg_model->foreground );
        cvReleaseMemStorage(&bg_model->storage);
        memset( bg_model, 0, sizeof(*bg_model) );
        cvFree( _bg_model );
    }

    __END__;
}


static int CV_CDECL
icvUpdateGaussianBGModel2( IplImage* curr_frame, CvGaussBGModel2*  bg_model )
{
    //int i, j, k, n;
    int region_count = 0;
    CvSeq *first_seq = NULL, *prev_seq = NULL, *seq = NULL;
	float alpha,alphaInit;
	bg_model->countFrames++;
	alpha=bg_model->params.fAlphaT;

	if (bg_model->params.bInit){
		//faster initial updates
		alphaInit=(1.0f/(2*bg_model->countFrames+1));
		if (alphaInit>alpha)
		{
			alpha=alphaInit;
		}
		else
		{
			bg_model->params.bInit=0;
		}
	}

	icvUpdatePixelBackgroundGMM(&bg_model->data,&bg_model->params,alpha,(unsigned char*)curr_frame->imageData,(unsigned char*)bg_model->foreground->imageData);
    
	if (bg_model->params.bPostFiltering==1)
	{
    //foreground filtering

    //filter small regions
    cvClearMemStorage(bg_model->storage);
    
    cvMorphologyEx( bg_model->foreground, bg_model->foreground, 0, 0, CV_MOP_OPEN, 1 );
    cvMorphologyEx( bg_model->foreground, bg_model->foreground, 0, 0, CV_MOP_CLOSE, 1 );
    
    cvFindContours( bg_model->foreground, bg_model->storage, &first_seq, sizeof(CvContour), CV_RETR_LIST );
    for( seq = first_seq; seq; seq = seq->h_next )
    {
        CvContour* cnt = (CvContour*)seq;
        if( cnt->rect.width * cnt->rect.height < bg_model->params.minArea )
        {
            //delete small contour
            prev_seq = seq->h_prev;
            if( prev_seq )
            {
                prev_seq->h_next = seq->h_next;
                if( seq->h_next ) seq->h_next->h_prev = prev_seq;
            }
            else
            {
                first_seq = seq->h_next;
                if( seq->h_next ) seq->h_next->h_prev = NULL;
            }
        }
        else
        {
            region_count++;
        }
    }
    bg_model->foreground_regions = first_seq;
    cvZero(bg_model->foreground);
    cvDrawContours(bg_model->foreground, first_seq, CV_RGB(0, 0, 255), CV_RGB(0, 0, 255), 10, -1);
   
	return region_count; 
	}
	else
	{
		return 1;
	}
}

/* End of file. */
