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
// Example usage as part of the CvBGStatModel:
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

#include "precomp.hpp"

#define CV_BG_MODEL_MOG2		3			/* "Mixture of Gaussians 2".	*/

/* default parameters of gaussian background detection algorithm */
#define CV_BGFG_MOG2_STD_THRESHOLD            4.0f     /* lambda=2.5 is 99% */
#define CV_BGFG_MOG2_WINDOW_SIZE              500      /* Learning rate; alpha = 1/CV_GBG_WINDOW_SIZE */
#define CV_BGFG_MOG2_BACKGROUND_THRESHOLD     0.9f     /* threshold sum of weights for background test */
#define CV_BGFG_MOG2_STD_THRESHOLD_GENERATE   3.0f     /* lambda=2.5 is 99% */
#define CV_BGFG_MOG2_NGAUSSIANS               5        /* = K = number of Gaussians in mixture */
#define CV_BGFG_MOG2_SIGMA_INIT               15.0f
#define CV_BGFG_MOG2_MINAREA                  15.0f

/* additional parameters */
#define CV_BGFG_MOG2_CT					      0.05f     /* complexity reduction prior constant 0 - no reduction of number of components*/
#define CV_BGFG_MOG2_SHADOW_VALUE             127       /* value to use in the segmentation mask for shadows, sot 0 not to do shadow detection*/
#define CV_BGFG_MOG2_SHADOW_TAU               0.5f      /* Tau - shadow threshold, see the paper for explanation*/

struct CvGaussBGStatModel2Params
{  
	bool bPostFiltering;//defult 1 - do postfiltering 
    double  minArea; // for postfiltering
    
	bool bShadowDetection;//default 1 - do shadow detection 
	bool bRemoveForeground;//default 0, set to 1 to remove foreground pixels from the image and return background image
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
	float fSigma;
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
	unsigned char nShadowDetection;//do shadow detection - insert this value as the detection result
	float fTau;
	// Tau - shadow threshold. The shadow is detected if the pixel is darker
	//version of the background. Tau is a threshold on how much darker the shadow can be.
	//Tau= 0.5 means that if pixel is more than 2 times darker then it is not shadow
	//See: Prati,Mikic,Trivedi,Cucchiarra,"Detecting Moving Shadows...",IEEE PAMI,2003.
};

struct CvPBGMMGaussian
{
	float sigma;
	float muR;
	float muG;
	float muB;
	float weight;
};

struct CvGaussBGStatModel2Data
{  
	int nWidth,nHeight,nSize,nNBands;//image info
	// dynamic array for the mixture of Gaussians
    std::vector<CvPBGMMGaussian> rGMM;
    std::vector<uchar> rnUsedModes;//number of Gaussian components per pixel
};

//only foreground image is updated
//no filtering included
struct CvGaussBGModel2
{
    CvGaussBGStatModel2Params params;
	CvGaussBGStatModel2Data   data;
	int                       countFrames;
};

static int _icvRemoveShadowGMM(long posPixel, 
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

static int _icvUpdatePixelBackgroundGMM(long posPixel, 
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

static void _icvReplacePixelBackgroundGMM(long pos, 
								unsigned char* pData, 
								CvPBGMMGaussian* m_aGaussians)
{
	pData[0]=(unsigned char) m_aGaussians[pos].muR;
	pData[1]=(unsigned char) m_aGaussians[pos].muG;
	pData[2]=(unsigned char) m_aGaussians[pos].muB;
}


static void icvUpdatePixelBackgroundGMM(CvGaussBGStatModel2Data* pGMMData,CvGaussBGStatModel2Params* pGMM, float m_fAlphaT, unsigned char* data,unsigned char* output)
{
	int size=pGMMData->nSize;
	unsigned char* pDataCurrent=data;
	unsigned char* pUsedModes=&pGMMData->rnUsedModes[0];
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
	CvPBGMMGaussian* m_aGaussians=&pGMMData->rGMM[0];
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


namespace cv
{

BackgroundSubtractorMOG2::BackgroundSubtractorMOG2()
{
    model = 0;
    initialize(Size(), 0);
}

BackgroundSubtractorMOG2::BackgroundSubtractorMOG2(double alphaT,
    double sigma, int nmixtures, bool postFiltering, double minArea,
    bool detectShadows, bool removeForeground, double Tb, double Tg,
    double TB, double CT, uchar shadowValue, double tau)
{
    model = 0;
    initialize(Size(), alphaT, sigma, nmixtures, postFiltering, minArea,
               detectShadows, removeForeground, Tb, Tg, TB, CT, shadowValue, tau);
}

    
void BackgroundSubtractorMOG2::initialize(Size frameSize, double alphaT,
    double sigma, int nmixtures, bool postFiltering, double minArea,
    bool detectShadows, bool removeForeground, double Tb, double Tg,
    double TB, double CT, uchar shadowValue, double tau)
{
    if(!model)
        model = new CvGaussBGModel2;
    
    CvGaussBGModel2* bg_model = (CvGaussBGModel2*)model;
        
    bg_model->params.bRemoveForeground=removeForeground;
    bg_model->params.bShadowDetection = detectShadows;
    bg_model->params.bPostFiltering = postFiltering;
    bg_model->params.minArea = minArea;
    bg_model->params.nM = nmixtures;
    bg_model->params.fTb = Tb;
    bg_model->params.fTB = TB;
    bg_model->params.fTg = Tg;
    bg_model->params.fSigma = sigma;
    bg_model->params.fAlphaT = alphaT;
    bg_model->params.fCT = CT;
    bg_model->params.nShadowDetection = shadowValue;
    bg_model->params.fTau = tau;
    
    int w = frameSize.width;
    int h = frameSize.height;
    int size = w*h;
    
    if( (bg_model->data.nWidth != w ||
        bg_model->data.nHeight != h) &&
        w > 0 && h > 0 )
    {
        bg_model->data.nWidth=w;
        bg_model->data.nHeight=h;
        bg_model->data.nNBands=3;
        bg_model->data.nSize=size;
        
        //GMM for each pixel
        bg_model->data.rGMM.resize(size * bg_model->params.nM);
    }
    //used modes per pixel
    bg_model->data.rnUsedModes.resize(0);
    bg_model->data.rnUsedModes.resize(size, (uchar)0);
    bg_model->params.bInit = true;
    bg_model->countFrames = 0;
}

    
BackgroundSubtractorMOG2::~BackgroundSubtractorMOG2()
{
    delete (CvGaussBGModel2*)model;
}

void BackgroundSubtractorMOG2::operator()(const Mat& image0, Mat& fgmask0, double learningRate)
{
    CvGaussBGModel2* bg_model = (CvGaussBGModel2*)model;
    
    CV_Assert(bg_model != 0);
    Mat fgmask = fgmask0, image = image0;
    CV_Assert( image.type() == CV_8UC1 || image.type() == CV_8UC3 );
    
    if( learningRate <= 0 )
        learningRate = bg_model->params.fAlphaT;
    if( learningRate >= 1 )
    {
        learningRate = 1;
        bg_model->params.bInit = true;
    }
    if( image.size() != Size(bg_model->data.nWidth, bg_model->data.nHeight) )
        initialize(image.size(), learningRate, bg_model->params.fSigma,
                   bg_model->params.nM, bg_model->params.bPostFiltering,
                   bg_model->params.minArea, bg_model->params.bShadowDetection,
                   bg_model->params.bRemoveForeground,
                   bg_model->params.fTb, bg_model->params.fTg, bg_model->params.fTB,
                   bg_model->params.fCT, bg_model->params.nShadowDetection, bg_model->params.fTau);
    
    //int i, j, k, n;
	float alpha = (float)bg_model->params.fAlphaT;
	bg_model->countFrames++;
    
	if (bg_model->params.bInit)
    {
		//faster initial updates
		float alphaInit = 1.0f/(2*bg_model->countFrames+1);
		if( alphaInit > alpha )
			alpha = alphaInit;
		else
			bg_model->params.bInit = false;
	}
    
    if( !image.isContinuous() || image.channels() != 3 )
    {
        image.release();
        image.create(image0.size(), CV_8UC3);
        if( image0.type() == image.type() )
            image0.copyTo(image);
        else
            cvtColor(image0, image, CV_GRAY2BGR);
    }

    if( !fgmask.isContinuous() )
        fgmask.release();
    fgmask.create(image.size(), CV_8UC1);
                     
    icvUpdatePixelBackgroundGMM(&bg_model->data,&bg_model->params,alpha,image.data,fgmask.data);
    
	if (!bg_model->params.bPostFiltering)
        return;

    //foreground filtering: filter out small regions    
    morphologyEx(fgmask, fgmask, CV_MOP_OPEN, Mat());
    morphologyEx(fgmask, fgmask, CV_MOP_CLOSE, Mat());
    
    vector<vector<Point> > contours;
    findContours(fgmask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
    fgmask = Scalar::all(0);
    
    for( size_t i = 0; i < contours.size(); i++ )
    {
        if( boundingRect(Mat(contours[i])).area() < bg_model->params.minArea )
            continue;
        drawContours(fgmask, contours, (int)i, Scalar::all(255), -1, 8, vector<Vec4i>(), 1);
    }
    
    fgmask.copyTo(fgmask0);
}

}

/* End of file. */
