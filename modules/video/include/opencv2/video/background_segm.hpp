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

#ifndef __OPENCV_BACKGROUND_SEGM_HPP__
#define __OPENCV_BACKGROUND_SEGM_HPP__

#include "opencv2/core/core.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/****************************************************************************************\
*                           Background/foreground segmentation                           *
\****************************************************************************************/

/* We discriminate between foreground and background pixels
 * by building and maintaining a model of the background.
 * Any pixel which does not fit this model is then deemed
 * to be foreground.
 *
 * At present we support two core background models,
 * one of which has two variations:
 *
 *  o CV_BG_MODEL_FGD: latest and greatest algorithm, described in
 *    
 *	 Foreground Object Detection from Videos Containing Complex Background.
 *	 Liyuan Li, Weimin Huang, Irene Y.H. Gu, and Qi Tian. 
 *	 ACM MM2003 9p
 *
 *  o CV_BG_MODEL_FGD_SIMPLE:
 *       A code comment describes this as a simplified version of the above,
 *       but the code is in fact currently identical
 *
 *  o CV_BG_MODEL_MOG: "Mixture of Gaussians", older algorithm, described in
 *
 *       Moving target classification and tracking from real-time video.
 *       A Lipton, H Fujijoshi, R Patil
 *       Proceedings IEEE Workshop on Application of Computer Vision pp 8-14 1998
 *
 *       Learning patterns of activity using real-time tracking
 *       C Stauffer and W Grimson  August 2000
 *       IEEE Transactions on Pattern Analysis and Machine Intelligence 22(8):747-757
 */


#define CV_BG_MODEL_FGD		0
#define CV_BG_MODEL_MOG		1			/* "Mixture of Gaussians".	*/
#define CV_BG_MODEL_FGD_SIMPLE	2

struct CvBGStatModel;

typedef void (CV_CDECL * CvReleaseBGStatModel)( struct CvBGStatModel** bg_model );
typedef int (CV_CDECL * CvUpdateBGStatModel)( IplImage* curr_frame, struct CvBGStatModel* bg_model,
                                              double learningRate );

#define CV_BG_STAT_MODEL_FIELDS()                                                   \
    int             type; /*type of BG model*/                                      \
    CvReleaseBGStatModel release;                                                   \
    CvUpdateBGStatModel update;                                                     \
    IplImage*       background;   /*8UC3 reference background image*/               \
    IplImage*       foreground;   /*8UC1 foreground image*/                         \
    IplImage**      layers;       /*8UC3 reference background image, can be null */ \
    int             layer_count;  /* can be zero */                                 \
    CvMemStorage*   storage;      /*storage for foreground_regions*/                \
    CvSeq*          foreground_regions /*foreground object contours*/

typedef struct CvBGStatModel
{
    CV_BG_STAT_MODEL_FIELDS();
} CvBGStatModel;

// 

// Releases memory used by BGStatModel
CVAPI(void) cvReleaseBGStatModel( CvBGStatModel** bg_model );

// Updates statistical model and returns number of found foreground regions
CVAPI(int) cvUpdateBGStatModel( IplImage* current_frame, CvBGStatModel*  bg_model,
                                double learningRate CV_DEFAULT(-1));

// Performs FG post-processing using segmentation
// (all pixels of a region will be classified as foreground if majority of pixels of the region are FG).
// parameters:
//      segments - pointer to result of segmentation (for example MeanShiftSegmentation)
//      bg_model - pointer to CvBGStatModel structure
CVAPI(void) cvRefineForegroundMaskBySegm( CvSeq* segments, CvBGStatModel*  bg_model );

/* Common use change detection function */
CVAPI(int)  cvChangeDetection( IplImage*  prev_frame,
                               IplImage*  curr_frame,
                               IplImage*  change_mask );

/*
  Interface of ACM MM2003 algorithm
*/

/* Default parameters of foreground detection algorithm: */
#define  CV_BGFG_FGD_LC              128
#define  CV_BGFG_FGD_N1C             15
#define  CV_BGFG_FGD_N2C             25

#define  CV_BGFG_FGD_LCC             64
#define  CV_BGFG_FGD_N1CC            25
#define  CV_BGFG_FGD_N2CC            40

/* Background reference image update parameter: */
#define  CV_BGFG_FGD_ALPHA_1         0.1f

/* stat model update parameter
 * 0.002f ~ 1K frame(~45sec), 0.005 ~ 18sec (if 25fps and absolutely static BG)
 */
#define  CV_BGFG_FGD_ALPHA_2         0.005f

/* start value for alpha parameter (to fast initiate statistic model) */
#define  CV_BGFG_FGD_ALPHA_3         0.1f

#define  CV_BGFG_FGD_DELTA           2

#define  CV_BGFG_FGD_T               0.9f

#define  CV_BGFG_FGD_MINAREA         15.f

#define  CV_BGFG_FGD_BG_UPDATE_TRESH 0.5f

/* See the above-referenced Li/Huang/Gu/Tian paper
 * for a full description of these background-model
 * tuning parameters.
 *
 * Nomenclature:  'c'  == "color", a three-component red/green/blue vector.
 *                         We use histograms of these to model the range of
 *                         colors we've seen at a given background pixel.
 *
 *                'cc' == "color co-occurrence", a six-component vector giving
 *                         RGB color for both this frame and preceding frame.
 *                             We use histograms of these to model the range of
 *                         color CHANGES we've seen at a given background pixel.
 */
typedef struct CvFGDStatModelParams
{
    int    Lc;			/* Quantized levels per 'color' component. Power of two, typically 32, 64 or 128.				*/
    int    N1c;			/* Number of color vectors used to model normal background color variation at a given pixel.			*/
    int    N2c;			/* Number of color vectors retained at given pixel.  Must be > N1c, typically ~ 5/3 of N1c.			*/
				/* Used to allow the first N1c vectors to adapt over time to changing background.				*/

    int    Lcc;			/* Quantized levels per 'color co-occurrence' component.  Power of two, typically 16, 32 or 64.			*/
    int    N1cc;		/* Number of color co-occurrence vectors used to model normal background color variation at a given pixel.	*/
    int    N2cc;		/* Number of color co-occurrence vectors retained at given pixel.  Must be > N1cc, typically ~ 5/3 of N1cc.	*/
				/* Used to allow the first N1cc vectors to adapt over time to changing background.				*/

    int    is_obj_without_holes;/* If TRUE we ignore holes within foreground blobs. Defaults to TRUE.						*/
    int    perform_morphing;	/* Number of erode-dilate-erode foreground-blob cleanup iterations.						*/
				/* These erase one-pixel junk blobs and merge almost-touching blobs. Default value is 1.			*/

    float  alpha1;		/* How quickly we forget old background pixel values seen.  Typically set to 0.1  				*/
    float  alpha2;		/* "Controls speed of feature learning". Depends on T. Typical value circa 0.005. 				*/
    float  alpha3;		/* Alternate to alpha2, used (e.g.) for quicker initial convergence. Typical value 0.1.				*/

    float  delta;		/* Affects color and color co-occurrence quantization, typically set to 2.					*/
    float  T;			/* "A percentage value which determines when new features can be recognized as new background." (Typically 0.9).*/
    float  minArea;		/* Discard foreground blobs whose bounding box is smaller than this threshold.					*/
} CvFGDStatModelParams;

typedef struct CvBGPixelCStatTable
{
    float          Pv, Pvb;
    uchar          v[3];
} CvBGPixelCStatTable;

typedef struct CvBGPixelCCStatTable
{
    float          Pv, Pvb;
    uchar          v[6];
} CvBGPixelCCStatTable;

typedef struct CvBGPixelStat
{
    float                 Pbc;
    float                 Pbcc;
    CvBGPixelCStatTable*  ctable;
    CvBGPixelCCStatTable* cctable;
    uchar                 is_trained_st_model;
    uchar                 is_trained_dyn_model;
} CvBGPixelStat;


typedef struct CvFGDStatModel
{
    CV_BG_STAT_MODEL_FIELDS();
    CvBGPixelStat*         pixel_stat;
    IplImage*              Ftd;
    IplImage*              Fbd;
    IplImage*              prev_frame;
    CvFGDStatModelParams   params;
} CvFGDStatModel;

/* Creates FGD model */
CVAPI(CvBGStatModel*) cvCreateFGDStatModel( IplImage* first_frame,
                    CvFGDStatModelParams* parameters CV_DEFAULT(NULL));

/* 
   Interface of Gaussian mixture algorithm

   "An improved adaptive background mixture model for real-time tracking with shadow detection"
   P. KadewTraKuPong and R. Bowden,
   Proc. 2nd European Workshp on Advanced Video-Based Surveillance Systems, 2001."
   http://personal.ee.surrey.ac.uk/Personal/R.Bowden/publications/avbs01/avbs01.pdf
*/

/* Note:  "MOG" == "Mixture Of Gaussians": */

#define CV_BGFG_MOG_MAX_NGAUSSIANS 500

/* default parameters of gaussian background detection algorithm */
#define CV_BGFG_MOG_BACKGROUND_THRESHOLD     0.7     /* threshold sum of weights for background test */
#define CV_BGFG_MOG_STD_THRESHOLD            2.5     /* lambda=2.5 is 99% */
#define CV_BGFG_MOG_WINDOW_SIZE              200     /* Learning rate; alpha = 1/CV_GBG_WINDOW_SIZE */
#define CV_BGFG_MOG_NGAUSSIANS               5       /* = K = number of Gaussians in mixture */
#define CV_BGFG_MOG_WEIGHT_INIT              0.05
#define CV_BGFG_MOG_SIGMA_INIT               30
#define CV_BGFG_MOG_MINAREA                  15.f


#define CV_BGFG_MOG_NCOLORS                  3

typedef struct CvGaussBGStatModelParams
{    
    int     win_size;               /* = 1/alpha */
    int     n_gauss;
    double  bg_threshold, std_threshold, minArea;
    double  weight_init, variance_init;
}CvGaussBGStatModelParams;

typedef struct CvGaussBGValues
{
    int         match_sum;
    double      weight;
    double      variance[CV_BGFG_MOG_NCOLORS];
    double      mean[CV_BGFG_MOG_NCOLORS];
} CvGaussBGValues;

typedef struct CvGaussBGPoint
{
    CvGaussBGValues* g_values;
} CvGaussBGPoint;


typedef struct CvGaussBGModel
{
    CV_BG_STAT_MODEL_FIELDS();
    CvGaussBGStatModelParams   params;    
    CvGaussBGPoint*            g_point;    
    int                        countFrames;
} CvGaussBGModel;


/* Creates Gaussian mixture background model */
CVAPI(CvBGStatModel*) cvCreateGaussianBGModel( IplImage* first_frame,
                CvGaussBGStatModelParams* parameters CV_DEFAULT(NULL));


typedef struct CvBGCodeBookElem
{
    struct CvBGCodeBookElem* next;
    int tLastUpdate;
    int stale;
    uchar boxMin[3];
    uchar boxMax[3];
    uchar learnMin[3];
    uchar learnMax[3];
} CvBGCodeBookElem;

typedef struct CvBGCodeBookModel
{
    CvSize size;
    int t;
    uchar cbBounds[3];
    uchar modMin[3];
    uchar modMax[3];
    CvBGCodeBookElem** cbmap;
    CvMemStorage* storage;
    CvBGCodeBookElem* freeList;
} CvBGCodeBookModel;

CVAPI(CvBGCodeBookModel*) cvCreateBGCodeBookModel();
CVAPI(void) cvReleaseBGCodeBookModel( CvBGCodeBookModel** model );

CVAPI(void) cvBGCodeBookUpdate( CvBGCodeBookModel* model, const CvArr* image,
                                CvRect roi CV_DEFAULT(cvRect(0,0,0,0)),
                                const CvArr* mask CV_DEFAULT(0) );

CVAPI(int) cvBGCodeBookDiff( const CvBGCodeBookModel* model, const CvArr* image,
                             CvArr* fgmask, CvRect roi CV_DEFAULT(cvRect(0,0,0,0)) );

CVAPI(void) cvBGCodeBookClearStale( CvBGCodeBookModel* model, int staleThresh,
                                    CvRect roi CV_DEFAULT(cvRect(0,0,0,0)),
                                    const CvArr* mask CV_DEFAULT(0) );

CVAPI(CvSeq*) cvSegmentFGMask( CvArr *fgmask, int poly1Hull0 CV_DEFAULT(1),
                               float perimScale CV_DEFAULT(4.f),
                               CvMemStorage* storage CV_DEFAULT(0),
                               CvPoint offset CV_DEFAULT(cvPoint(0,0)));


#ifdef __cplusplus
}

namespace cv
{

/*!
 The Base Class for Background/Foreground Segmentation
 
 The class is only used to define the common interface for
 the whole family of background/foreground segmentation algorithms.
*/
class CV_EXPORTS BackgroundSubtractor
{
public:
    //! the virtual destructor
    virtual ~BackgroundSubtractor();
    //! the update operator that takes the next video frame and returns the current foreground mask as 8-bit binary image.
    virtual CV_WRAP_AS(apply) void operator()(const Mat& image, CV_OUT Mat& fgmask,
                                              double learningRate=0);
};


/*!
 Gaussian Mixture-based Backbround/Foreground Segmentation Algorithm
 
 The class implements the following algorithm:
 "An improved adaptive background mixture model for real-time tracking with shadow detection"
 P. KadewTraKuPong and R. Bowden,
 Proc. 2nd European Workshp on Advanced Video-Based Surveillance Systems, 2001."
 http://personal.ee.surrey.ac.uk/Personal/R.Bowden/publications/avbs01/avbs01.pdf
 
*/
class CV_EXPORTS BackgroundSubtractorMOG : public BackgroundSubtractor
{
public:
    //! the default constructor
    BackgroundSubtractorMOG();
    //! the full constructor that takes the length of the history, the number of gaussian mixtures, the background ratio parameter and the noise strength
    BackgroundSubtractorMOG(int history, int nmixtures, double backgroundRatio, double noiseSigma=0);
    //! the destructor
    virtual ~BackgroundSubtractorMOG();
    //! the update operator
    virtual void operator()(const Mat& image, Mat& fgmask, double learningRate=0);
    
    //! re-initiaization method
    virtual void initialize(Size frameSize, int frameType);
    
    Size frameSize;
    int frameType;
    Mat bgmodel;
    int nframes;
    int history;
    int nmixtures;
    double varThreshold;
    double backgroundRatio;
    double noiseSigma;
};	

}
#endif

#endif
