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
//                For Open Source Computer Vision Library
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

#ifndef __OPENCV_LEGACY_HPP__
#define __OPENCV_LEGACY_HPP__

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#ifdef __cplusplus
extern "C" {
#endif

CVAPI(CvSeq*) cvSegmentImage( const CvArr* srcarr, CvArr* dstarr,
                                    double canny_threshold,
                                    double ffill_threshold,
                                    CvMemStorage* storage );

/****************************************************************************************\
*                                  Eigen objects                                         *
\****************************************************************************************/

typedef int (CV_CDECL * CvCallback)(int index, void* buffer, void* user_data);
typedef union
{
    CvCallback callback;
    void* data;
}
CvInput;

#define CV_EIGOBJ_NO_CALLBACK     0
#define CV_EIGOBJ_INPUT_CALLBACK  1
#define CV_EIGOBJ_OUTPUT_CALLBACK 2
#define CV_EIGOBJ_BOTH_CALLBACK   3

/* Calculates covariation matrix of a set of arrays */
CVAPI(void)  cvCalcCovarMatrixEx( int nObjects, void* input, int ioFlags,
                                  int ioBufSize, uchar* buffer, void* userData,
                                  IplImage* avg, float* covarMatrix );

/* Calculates eigen values and vectors of covariation matrix of a set of
   arrays */
CVAPI(void)  cvCalcEigenObjects( int nObjects, void* input, void* output,
                                 int ioFlags, int ioBufSize, void* userData,
                                 CvTermCriteria* calcLimit, IplImage* avg,
                                 float* eigVals );

/* Calculates dot product (obj - avg) * eigObj (i.e. projects image to eigen vector) */
CVAPI(double)  cvCalcDecompCoeff( IplImage* obj, IplImage* eigObj, IplImage* avg );

/* Projects image to eigen space (finds all decomposion coefficients */
CVAPI(void)  cvEigenDecomposite( IplImage* obj, int nEigObjs, void* eigInput,
                                 int ioFlags, void* userData, IplImage* avg,
                                 float* coeffs );

/* Projects original objects used to calculate eigen space basis to that space */
CVAPI(void)  cvEigenProjection( void* eigInput, int nEigObjs, int ioFlags,
                                void* userData, float* coeffs, IplImage* avg,
                                IplImage* proj );

/****************************************************************************************\
*                                       1D/2D HMM                                        *
\****************************************************************************************/

typedef struct CvImgObsInfo
{
    int obs_x;
    int obs_y;
    int obs_size;
    float* obs;//consequtive observations

    int* state;/* arr of pairs superstate/state to which observation belong */
    int* mix;  /* number of mixture to which observation belong */

} CvImgObsInfo;/*struct for 1 image*/

typedef CvImgObsInfo Cv1DObsInfo;

typedef struct CvEHMMState
{
    int num_mix;        /*number of mixtures in this state*/
    float* mu;          /*mean vectors corresponding to each mixture*/
    float* inv_var;     /* square root of inversed variances corresp. to each mixture*/
    float* log_var_val; /* sum of 0.5 (LN2PI + ln(variance[i]) ) for i=1,n */
    float* weight;      /*array of mixture weights. Summ of all weights in state is 1. */

} CvEHMMState;

typedef struct CvEHMM
{
    int level; /* 0 - lowest(i.e its states are real states), ..... */
    int num_states; /* number of HMM states */
    float*  transP;/*transition probab. matrices for states */
    float** obsProb; /* if level == 0 - array of brob matrices corresponding to hmm
                        if level == 1 - martix of matrices */
    union
    {
        CvEHMMState* state; /* if level == 0 points to real states array,
                               if not - points to embedded hmms */
        struct CvEHMM* ehmm; /* pointer to an embedded model or NULL, if it is a leaf */
    } u;

} CvEHMM;

/*CVAPI(int)  icvCreate1DHMM( CvEHMM** this_hmm,
                                   int state_number, int* num_mix, int obs_size );

CVAPI(int)  icvRelease1DHMM( CvEHMM** phmm );

CVAPI(int)  icvUniform1DSegm( Cv1DObsInfo* obs_info, CvEHMM* hmm );

CVAPI(int)  icvInit1DMixSegm( Cv1DObsInfo** obs_info_array, int num_img, CvEHMM* hmm);

CVAPI(int)  icvEstimate1DHMMStateParams( CvImgObsInfo** obs_info_array, int num_img, CvEHMM* hmm);

CVAPI(int)  icvEstimate1DObsProb( CvImgObsInfo* obs_info, CvEHMM* hmm );

CVAPI(int)  icvEstimate1DTransProb( Cv1DObsInfo** obs_info_array,
                                           int num_seq,
                                           CvEHMM* hmm );

CVAPI(float)  icvViterbi( Cv1DObsInfo* obs_info, CvEHMM* hmm);

CVAPI(int)  icv1DMixSegmL2( CvImgObsInfo** obs_info_array, int num_img, CvEHMM* hmm );*/

/*********************************** Embedded HMMs *************************************/

/* Creates 2D HMM */
CVAPI(CvEHMM*)  cvCreate2DHMM( int* stateNumber, int* numMix, int obsSize );

/* Releases HMM */
CVAPI(void)  cvRelease2DHMM( CvEHMM** hmm );

#define CV_COUNT_OBS(roi, win, delta, numObs )                                       \
{                                                                                    \
   (numObs)->width  =((roi)->width  -(win)->width  +(delta)->width)/(delta)->width;  \
   (numObs)->height =((roi)->height -(win)->height +(delta)->height)/(delta)->height;\
}

/* Creates storage for observation vectors */
CVAPI(CvImgObsInfo*)  cvCreateObsInfo( CvSize numObs, int obsSize );

/* Releases storage for observation vectors */
CVAPI(void)  cvReleaseObsInfo( CvImgObsInfo** obs_info );


/* The function takes an image on input and and returns the sequnce of observations
   to be used with an embedded HMM; Each observation is top-left block of DCT
   coefficient matrix */
CVAPI(void)  cvImgToObs_DCT( const CvArr* arr, float* obs, CvSize dctSize,
                             CvSize obsSize, CvSize delta );


/* Uniformly segments all observation vectors extracted from image */
CVAPI(void)  cvUniformImgSegm( CvImgObsInfo* obs_info, CvEHMM* ehmm );

/* Does mixture segmentation of the states of embedded HMM */
CVAPI(void)  cvInitMixSegm( CvImgObsInfo** obs_info_array,
                            int num_img, CvEHMM* hmm );

/* Function calculates means, variances, weights of every Gaussian mixture
   of every low-level state of embedded HMM */
CVAPI(void)  cvEstimateHMMStateParams( CvImgObsInfo** obs_info_array,
                                       int num_img, CvEHMM* hmm );

/* Function computes transition probability matrices of embedded HMM
   given observations segmentation */
CVAPI(void)  cvEstimateTransProb( CvImgObsInfo** obs_info_array,
                                  int num_img, CvEHMM* hmm );

/* Function computes probabilities of appearing observations at any state
   (i.e. computes P(obs|state) for every pair(obs,state)) */
CVAPI(void)  cvEstimateObsProb( CvImgObsInfo* obs_info,
                                CvEHMM* hmm );

/* Runs Viterbi algorithm for embedded HMM */
CVAPI(float)  cvEViterbi( CvImgObsInfo* obs_info, CvEHMM* hmm );


/* Function clusters observation vectors from several images
   given observations segmentation.
   Euclidean distance used for clustering vectors.
   Centers of clusters are given means of every mixture */
CVAPI(void)  cvMixSegmL2( CvImgObsInfo** obs_info_array,
                          int num_img, CvEHMM* hmm );

/****************************************************************************************\
*               A few functions from old stereo gesture recognition demosions            *
\****************************************************************************************/

/* Creates hand mask image given several points on the hand */
CVAPI(void)  cvCreateHandMask( CvSeq* hand_points,
                                   IplImage *img_mask, CvRect *roi);

/* Finds hand region in range image data */
CVAPI(void)  cvFindHandRegion (CvPoint3D32f* points, int count,
                                CvSeq* indexs,
                                float* line, CvSize2D32f size, int flag,
                                CvPoint3D32f* center,
                                CvMemStorage* storage, CvSeq **numbers);

/* Finds hand region in range image data (advanced version) */
CVAPI(void)  cvFindHandRegionA( CvPoint3D32f* points, int count,
                                CvSeq* indexs,
                                float* line, CvSize2D32f size, int jc,
                                CvPoint3D32f* center,
                                CvMemStorage* storage, CvSeq **numbers);

/* Calculates the cooficients of the homography matrix */
CVAPI(void)  cvCalcImageHomography( float* line, CvPoint3D32f* center,
                                    float* intrinsic, float* homography );

/****************************************************************************************\
*                           Additional operations on Subdivisions                        *
\****************************************************************************************/

// paints voronoi diagram: just demo function
CVAPI(void)  icvDrawMosaic( CvSubdiv2D* subdiv, IplImage* src, IplImage* dst );

// checks planar subdivision for correctness. It is not an absolute check,
// but it verifies some relations between quad-edges
CVAPI(int)   icvSubdiv2DCheck( CvSubdiv2D* subdiv );

// returns squared distance between two 2D points with floating-point coordinates.
CV_INLINE double icvSqDist2D32f( CvPoint2D32f pt1, CvPoint2D32f pt2 )
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;

    return dx*dx + dy*dy;
}


/****************************************************************************************\
*                           More operations on sequences                                 *
\****************************************************************************************/

/*****************************************************************************************/

#define CV_CURRENT_INT( reader ) (*((int *)(reader).ptr))
#define CV_PREV_INT( reader ) (*((int *)(reader).prev_elem))

#define  CV_GRAPH_WEIGHTED_VERTEX_FIELDS() CV_GRAPH_VERTEX_FIELDS()\
    float weight;

#define  CV_GRAPH_WEIGHTED_EDGE_FIELDS() CV_GRAPH_EDGE_FIELDS()

typedef struct CvGraphWeightedVtx
{
    CV_GRAPH_WEIGHTED_VERTEX_FIELDS()
} CvGraphWeightedVtx;

typedef struct CvGraphWeightedEdge
{
    CV_GRAPH_WEIGHTED_EDGE_FIELDS()
} CvGraphWeightedEdge;

typedef enum CvGraphWeightType
{
    CV_NOT_WEIGHTED,
    CV_WEIGHTED_VTX,
    CV_WEIGHTED_EDGE,
    CV_WEIGHTED_ALL
} CvGraphWeightType;


/* Calculates histogram of a contour */
CVAPI(void)  cvCalcPGH( const CvSeq* contour, CvHistogram* hist );

#define CV_DOMINANT_IPAN 1

/* Finds high-curvature points of the contour */
CVAPI(CvSeq*) cvFindDominantPoints( CvSeq* contour, CvMemStorage* storage,
                                   int method CV_DEFAULT(CV_DOMINANT_IPAN),
                                   double parameter1 CV_DEFAULT(0),
                                   double parameter2 CV_DEFAULT(0),
                                   double parameter3 CV_DEFAULT(0),
                                   double parameter4 CV_DEFAULT(0));

/*****************************************************************************************/


/*******************************Stereo correspondence*************************************/

typedef struct CvCliqueFinder
{   
    CvGraph* graph;
    int**    adj_matr;
    int N; //graph size

    // stacks, counters etc/
    int k; //stack size
    int* current_comp;
    int** All;
    
    int* ne;
    int* ce;
    int* fixp; //node with minimal disconnections
    int* nod;
    int* s; //for selected candidate
    int status;
    int best_score;
    int weighted;
    int weighted_edges;    
    float best_weight;
    float* edge_weights;
    float* vertex_weights;
    float* cur_weight;
    float* cand_weight;

} CvCliqueFinder;

#define CLIQUE_TIME_OFF 2
#define CLIQUE_FOUND 1
#define CLIQUE_END   0

/*CVAPI(void) cvStartFindCliques( CvGraph* graph, CvCliqueFinder* finder, int reverse, 
                                   int weighted CV_DEFAULT(0),  int weighted_edges CV_DEFAULT(0));
CVAPI(int) cvFindNextMaximalClique( CvCliqueFinder* finder, int* clock_rest CV_DEFAULT(0) ); 
CVAPI(void) cvEndFindCliques( CvCliqueFinder* finder );

CVAPI(void) cvBronKerbosch( CvGraph* graph );*/


/*F///////////////////////////////////////////////////////////////////////////////////////
//
//    Name:    cvSubgraphWeight
//    Purpose: finds weight of subgraph in a graph
//    Context:
//    Parameters:
//      graph - input graph.
//      subgraph - sequence of pairwise different ints.  These are indices of vertices of subgraph.
//      weight_type - describes the way we measure weight.
//            one of the following:
//            CV_NOT_WEIGHTED - weight of a clique is simply its size
//            CV_WEIGHTED_VTX - weight of a clique is the sum of weights of its vertices
//            CV_WEIGHTED_EDGE - the same but edges
//            CV_WEIGHTED_ALL - the same but both edges and vertices
//      weight_vtx - optional vector of floats, with size = graph->total.
//            If weight_type is either CV_WEIGHTED_VTX or CV_WEIGHTED_ALL
//            weights of vertices must be provided.  If weight_vtx not zero
//            these weights considered to be here, otherwise function assumes
//            that vertices of graph are inherited from CvGraphWeightedVtx.
//      weight_edge - optional matrix of floats, of width and height = graph->total.
//            If weight_type is either CV_WEIGHTED_EDGE or CV_WEIGHTED_ALL
//            weights of edges ought to be supplied.  If weight_edge is not zero
//            function finds them here, otherwise function expects
//            edges of graph to be inherited from CvGraphWeightedEdge.
//            If this parameter is not zero structure of the graph is determined from matrix
//            rather than from CvGraphEdge's.  In particular, elements corresponding to
//            absent edges should be zero.
//    Returns:
//      weight of subgraph.
//    Notes:
//F*/
/*CVAPI(float) cvSubgraphWeight( CvGraph *graph, CvSeq *subgraph,
                                  CvGraphWeightType weight_type CV_DEFAULT(CV_NOT_WEIGHTED),
                                  CvVect32f weight_vtx CV_DEFAULT(0),
                                  CvMatr32f weight_edge CV_DEFAULT(0) );*/


/*F///////////////////////////////////////////////////////////////////////////////////////
//
//    Name:    cvFindCliqueEx
//    Purpose: tries to find clique with maximum possible weight in a graph
//    Context:
//    Parameters:
//      graph - input graph.
//      storage - memory storage to be used by the result.
//      is_complementary - optional flag showing whether function should seek for clique
//            in complementary graph.
//      weight_type - describes our notion about weight.
//            one of the following:
//            CV_NOT_WEIGHTED - weight of a clique is simply its size
//            CV_WEIGHTED_VTX - weight of a clique is the sum of weights of its vertices
//            CV_WEIGHTED_EDGE - the same but edges
//            CV_WEIGHTED_ALL - the same but both edges and vertices
//      weight_vtx - optional vector of floats, with size = graph->total.
//            If weight_type is either CV_WEIGHTED_VTX or CV_WEIGHTED_ALL
//            weights of vertices must be provided.  If weight_vtx not zero
//            these weights considered to be here, otherwise function assumes
//            that vertices of graph are inherited from CvGraphWeightedVtx.
//      weight_edge - optional matrix of floats, of width and height = graph->total.
//            If weight_type is either CV_WEIGHTED_EDGE or CV_WEIGHTED_ALL
//            weights of edges ought to be supplied.  If weight_edge is not zero
//            function finds them here, otherwise function expects
//            edges of graph to be inherited from CvGraphWeightedEdge.
//            Note that in case of CV_WEIGHTED_EDGE or CV_WEIGHTED_ALL
//            nonzero is_complementary implies nonzero weight_edge.
//      start_clique - optional sequence of pairwise different ints.  They are indices of
//            vertices that shall be present in the output clique.
//      subgraph_of_ban - optional sequence of (maybe equal) ints.  They are indices of
//            vertices that shall not be present in the output clique.
//      clique_weight_ptr - optional output parameter.  Weight of found clique stored here.
//      num_generations - optional number of generations in evolutionary part of algorithm,
//            zero forces to return first found clique.
//      quality - optional parameter determining degree of required quality/speed tradeoff.
//            Must be in the range from 0 to 9.
//            0 is fast and dirty, 9 is slow but hopefully yields good clique.
//    Returns:
//      sequence of pairwise different ints.
//      These are indices of vertices that form found clique.
//    Notes:
//      in cases of CV_WEIGHTED_EDGE and CV_WEIGHTED_ALL weights should be nonnegative.
//      start_clique has a priority over subgraph_of_ban.
//F*/
/*CVAPI(CvSeq*) cvFindCliqueEx( CvGraph *graph, CvMemStorage *storage,
                                 int is_complementary CV_DEFAULT(0),
                                 CvGraphWeightType weight_type CV_DEFAULT(CV_NOT_WEIGHTED),
                                 CvVect32f weight_vtx CV_DEFAULT(0),
                                 CvMatr32f weight_edge CV_DEFAULT(0),
                                 CvSeq *start_clique CV_DEFAULT(0),
                                 CvSeq *subgraph_of_ban CV_DEFAULT(0),
                                 float *clique_weight_ptr CV_DEFAULT(0),
                                 int num_generations CV_DEFAULT(3),
                                 int quality CV_DEFAULT(2) );*/


#define CV_UNDEF_SC_PARAM         12345 //default value of parameters

#define CV_IDP_BIRCHFIELD_PARAM1  25    
#define CV_IDP_BIRCHFIELD_PARAM2  5
#define CV_IDP_BIRCHFIELD_PARAM3  12
#define CV_IDP_BIRCHFIELD_PARAM4  15
#define CV_IDP_BIRCHFIELD_PARAM5  25


#define  CV_DISPARITY_BIRCHFIELD  0    


/*F///////////////////////////////////////////////////////////////////////////
//
//    Name:    cvFindStereoCorrespondence
//    Purpose: find stereo correspondence on stereo-pair
//    Context:
//    Parameters:
//      leftImage - left image of stereo-pair (format 8uC1).
//      rightImage - right image of stereo-pair (format 8uC1).
//   mode - mode of correspondence retrieval (now CV_DISPARITY_BIRCHFIELD only)
//      dispImage - destination disparity image
//      maxDisparity - maximal disparity 
//      param1, param2, param3, param4, param5 - parameters of algorithm
//    Returns:
//    Notes:
//      Images must be rectified.
//      All images must have format 8uC1.
//F*/
CVAPI(void) 
cvFindStereoCorrespondence( 
                   const  CvArr* leftImage, const  CvArr* rightImage,
                   int     mode,
                   CvArr*  dispImage,
                   int     maxDisparity,                                
                   double  param1 CV_DEFAULT(CV_UNDEF_SC_PARAM), 
                   double  param2 CV_DEFAULT(CV_UNDEF_SC_PARAM), 
                   double  param3 CV_DEFAULT(CV_UNDEF_SC_PARAM), 
                   double  param4 CV_DEFAULT(CV_UNDEF_SC_PARAM), 
                   double  param5 CV_DEFAULT(CV_UNDEF_SC_PARAM) );

/*****************************************************************************************/
/************ Epiline functions *******************/



typedef struct CvStereoLineCoeff
{
    double Xcoef;
    double XcoefA;
    double XcoefB;
    double XcoefAB;

    double Ycoef;
    double YcoefA;
    double YcoefB;
    double YcoefAB;

    double Zcoef;
    double ZcoefA;
    double ZcoefB;
    double ZcoefAB;
}CvStereoLineCoeff;


typedef struct CvCamera
{
    float   imgSize[2]; /* size of the camera view, used during calibration */
    float   matrix[9]; /* intinsic camera parameters:  [ fx 0 cx; 0 fy cy; 0 0 1 ] */
    float   distortion[4]; /* distortion coefficients - two coefficients for radial distortion
                              and another two for tangential: [ k1 k2 p1 p2 ] */
    float   rotMatr[9];
    float   transVect[3]; /* rotation matrix and transition vector relatively
                             to some reference point in the space. */
} CvCamera;

typedef struct CvStereoCamera
{
    CvCamera* camera[2]; /* two individual camera parameters */
    float fundMatr[9]; /* fundamental matrix */

    /* New part for stereo */
    CvPoint3D32f epipole[2];
    CvPoint2D32f quad[2][4]; /* coordinates of destination quadrangle after
                                epipolar geometry rectification */
    double coeffs[2][3][3];/* coefficients for transformation */
    CvPoint2D32f border[2][4];
    CvSize warpSize;
    CvStereoLineCoeff* lineCoeffs;
    int needSwapCameras;/* flag set to 1 if need to swap cameras for good reconstruction */
    float rotMatrix[9];
    float transVector[3];
} CvStereoCamera;


typedef struct CvContourOrientation
{
    float egvals[2];
    float egvects[4];

    float max, min; // minimum and maximum projections
    int imax, imin;
} CvContourOrientation;

#define CV_CAMERA_TO_WARP 1
#define CV_WARP_TO_CAMERA 2

CVAPI(int) icvConvertWarpCoordinates(double coeffs[3][3],
                                CvPoint2D32f* cameraPoint,
                                CvPoint2D32f* warpPoint,
                                int direction);

CVAPI(int) icvGetSymPoint3D(  CvPoint3D64f pointCorner,
                            CvPoint3D64f point1,
                            CvPoint3D64f point2,
                            CvPoint3D64f *pointSym2);

CVAPI(void) icvGetPieceLength3D(CvPoint3D64f point1,CvPoint3D64f point2,double* dist);

CVAPI(int) icvCompute3DPoint(    double alpha,double betta,
                            CvStereoLineCoeff* coeffs,
                            CvPoint3D64f* point);

CVAPI(int) icvCreateConvertMatrVect( double*     rotMatr1,
                                double*     transVect1,
                                double*     rotMatr2,
                                double*     transVect2,
                                double*     convRotMatr,
                                double*     convTransVect);

CVAPI(int) icvConvertPointSystem(CvPoint3D64f  M2,
                            CvPoint3D64f* M1,
                            double*     rotMatr,
                            double*     transVect
                            );

CVAPI(int) icvComputeCoeffForStereo(  CvStereoCamera* stereoCamera);

CVAPI(int) icvGetCrossPieceVector(CvPoint2D32f p1_start,CvPoint2D32f p1_end,CvPoint2D32f v2_start,CvPoint2D32f v2_end,CvPoint2D32f *cross);
CVAPI(int) icvGetCrossLineDirect(CvPoint2D32f p1,CvPoint2D32f p2,float a,float b,float c,CvPoint2D32f* cross);
CVAPI(float) icvDefinePointPosition(CvPoint2D32f point1,CvPoint2D32f point2,CvPoint2D32f point);
CVAPI(int) icvStereoCalibration( int numImages,
                            int* nums,
                            CvSize imageSize,
                            CvPoint2D32f* imagePoints1,
                            CvPoint2D32f* imagePoints2,
                            CvPoint3D32f* objectPoints,
                            CvStereoCamera* stereoparams
                           );


CVAPI(int) icvComputeRestStereoParams(CvStereoCamera *stereoparams);

CVAPI(void) cvComputePerspectiveMap( const double coeffs[3][3], CvArr* rectMapX, CvArr* rectMapY );

CVAPI(int) icvComCoeffForLine(   CvPoint2D64f point1,
                            CvPoint2D64f point2,
                            CvPoint2D64f point3,
                            CvPoint2D64f point4,
                            double*    camMatr1,
                            double*    rotMatr1,
                            double*    transVect1,
                            double*    camMatr2,
                            double*    rotMatr2,
                            double*    transVect2,
                            CvStereoLineCoeff*    coeffs,
                            int* needSwapCameras);

CVAPI(int) icvGetDirectionForPoint(  CvPoint2D64f point,
                                double* camMatr,
                                CvPoint3D64f* direct);

CVAPI(int) icvGetCrossLines(CvPoint3D64f point11,CvPoint3D64f point12,
                       CvPoint3D64f point21,CvPoint3D64f point22,
                       CvPoint3D64f* midPoint);

CVAPI(int) icvComputeStereoLineCoeffs(   CvPoint3D64f pointA,
                                    CvPoint3D64f pointB,
                                    CvPoint3D64f pointCam1,
                                    double gamma,
                                    CvStereoLineCoeff*    coeffs);

/*CVAPI(int) icvComputeFundMatrEpipoles ( double* camMatr1, 
                                    double*     rotMatr1, 
                                    double*     transVect1,
                                    double*     camMatr2,
                                    double*     rotMatr2,
                                    double*     transVect2,
                                    CvPoint2D64f* epipole1,
                                    CvPoint2D64f* epipole2,
                                    double*     fundMatr);*/

CVAPI(int) icvGetAngleLine( CvPoint2D64f startPoint, CvSize imageSize,CvPoint2D64f *point1,CvPoint2D64f *point2);

CVAPI(void) icvGetCoefForPiece(   CvPoint2D64f p_start,CvPoint2D64f p_end,
                        double *a,double *b,double *c,
                        int* result);

/*CVAPI(void) icvGetCommonArea( CvSize imageSize,
                    CvPoint2D64f epipole1,CvPoint2D64f epipole2,
                    double* fundMatr,
                    double* coeff11,double* coeff12,
                    double* coeff21,double* coeff22,
                    int* result);*/

CVAPI(void) icvComputeeInfiniteProject1(double*    rotMatr,
                                     double*    camMatr1,
                                     double*    camMatr2,
                                     CvPoint2D32f point1,
                                     CvPoint2D32f *point2);

CVAPI(void) icvComputeeInfiniteProject2(double*    rotMatr,
                                     double*    camMatr1,
                                     double*    camMatr2,
                                     CvPoint2D32f* point1,
                                     CvPoint2D32f point2);

CVAPI(void) icvGetCrossDirectDirect(  double* direct1,double* direct2,
                            CvPoint2D64f *cross,int* result);

CVAPI(void) icvGetCrossPieceDirect(   CvPoint2D64f p_start,CvPoint2D64f p_end,
                            double a,double b,double c,
                            CvPoint2D64f *cross,int* result);

CVAPI(void) icvGetCrossPiecePiece( CvPoint2D64f p1_start,CvPoint2D64f p1_end,
                            CvPoint2D64f p2_start,CvPoint2D64f p2_end,
                            CvPoint2D64f* cross,
                            int* result);
                            
CVAPI(void) icvGetPieceLength(CvPoint2D64f point1,CvPoint2D64f point2,double* dist);

CVAPI(void) icvGetCrossRectDirect(    CvSize imageSize,
                            double a,double b,double c,
                            CvPoint2D64f *start,CvPoint2D64f *end,
                            int* result);

CVAPI(void) icvProjectPointToImage(   CvPoint3D64f point,
                            double* camMatr,double* rotMatr,double* transVect,
                            CvPoint2D64f* projPoint);

CVAPI(void) icvGetQuadsTransform( CvSize        imageSize,
                        double*     camMatr1,
                        double*     rotMatr1,
                        double*     transVect1,
                        double*     camMatr2,
                        double*     rotMatr2,
                        double*     transVect2,
                        CvSize*       warpSize,
                        double quad1[4][2],
                        double quad2[4][2],
                        double*     fundMatr,
                        CvPoint3D64f* epipole1,
                        CvPoint3D64f* epipole2
                        );

CVAPI(void) icvGetQuadsTransformStruct(  CvStereoCamera* stereoCamera);

CVAPI(void) icvComputeStereoParamsForCameras(CvStereoCamera* stereoCamera);

CVAPI(void) icvGetCutPiece(   double* areaLineCoef1,double* areaLineCoef2,
                    CvPoint2D64f epipole,
                    CvSize imageSize,
                    CvPoint2D64f* point11,CvPoint2D64f* point12,
                    CvPoint2D64f* point21,CvPoint2D64f* point22,
                    int* result);

CVAPI(void) icvGetMiddleAnglePoint(   CvPoint2D64f basePoint,
                            CvPoint2D64f point1,CvPoint2D64f point2,
                            CvPoint2D64f* midPoint);

CVAPI(void) icvGetNormalDirect(double* direct,CvPoint2D64f point,double* normDirect);

CVAPI(double) icvGetVect(CvPoint2D64f basePoint,CvPoint2D64f point1,CvPoint2D64f point2);

CVAPI(void) icvProjectPointToDirect(  CvPoint2D64f point,double* lineCoeff,
                            CvPoint2D64f* projectPoint);

CVAPI(void) icvGetDistanceFromPointToDirect( CvPoint2D64f point,double* lineCoef,double*dist);

CVAPI(IplImage*) icvCreateIsometricImage( IplImage* src, IplImage* dst,
                              int desired_depth, int desired_num_channels );

CVAPI(void) cvDeInterlace( const CvArr* frame, CvArr* fieldEven, CvArr* fieldOdd );

/*CVAPI(int) icvSelectBestRt(           int           numImages,
                                    int*          numPoints,
                                    CvSize        imageSize,
                                    CvPoint2D32f* imagePoints1,
                                    CvPoint2D32f* imagePoints2,
                                    CvPoint3D32f* objectPoints,

                                    CvMatr32f     cameraMatrix1,
                                    CvVect32f     distortion1,
                                    CvMatr32f     rotMatrs1,
                                    CvVect32f     transVects1,

                                    CvMatr32f     cameraMatrix2,
                                    CvVect32f     distortion2,
                                    CvMatr32f     rotMatrs2,
                                    CvVect32f     transVects2,

                                    CvMatr32f     bestRotMatr,
                                    CvVect32f     bestTransVect
                                    );*/


/****************************************************************************************\
*                                     Contour Tree                                       *
\****************************************************************************************/

/* Contour tree header */
typedef struct CvContourTree
{
    CV_SEQUENCE_FIELDS()
    CvPoint p1;            /* the first point of the binary tree root segment */
    CvPoint p2;            /* the last point of the binary tree root segment */
} CvContourTree;

/* Builds hierarhical representation of a contour */
CVAPI(CvContourTree*)  cvCreateContourTree( const CvSeq* contour,
                                            CvMemStorage* storage,
                                            double threshold );

/* Reconstruct (completelly or partially) contour a from contour tree */
CVAPI(CvSeq*)  cvContourFromContourTree( const CvContourTree* tree,
                                         CvMemStorage* storage,
                                         CvTermCriteria criteria );

/* Compares two contour trees */
enum { CV_CONTOUR_TREES_MATCH_I1 = 1 };

CVAPI(double)  cvMatchContourTrees( const CvContourTree* tree1,
                                    const CvContourTree* tree2,
                                    int method, double threshold );

/****************************************************************************************\
*                                   Contour Morphing                                     *
\****************************************************************************************/

/* finds correspondence between two contours */
CvSeq* cvCalcContoursCorrespondence( const CvSeq* contour1,
                                     const CvSeq* contour2, 
                                     CvMemStorage* storage);

/* morphs contours using the pre-calculated correspondence:
   alpha=0 ~ contour1, alpha=1 ~ contour2 */
CvSeq* cvMorphContours( const CvSeq* contour1, const CvSeq* contour2,
                        CvSeq* corr, double alpha,
                        CvMemStorage* storage );


/****************************************************************************************\
*                                   Active Contours                                      *
\****************************************************************************************/

#define  CV_VALUE  1
#define  CV_ARRAY  2
/* Updates active contour in order to minimize its cummulative
   (internal and external) energy. */
CVAPI(void)  cvSnakeImage( const IplImage* image, CvPoint* points,
                           int  length, float* alpha,
                           float* beta, float* gamma,
                           int coeff_usage, CvSize  win,
                           CvTermCriteria criteria, int calc_gradient CV_DEFAULT(1));

/****************************************************************************************\
*                                    Texture Descriptors                                 *
\****************************************************************************************/

#define CV_GLCM_OPTIMIZATION_NONE                   -2
#define CV_GLCM_OPTIMIZATION_LUT                    -1
#define CV_GLCM_OPTIMIZATION_HISTOGRAM              0

#define CV_GLCMDESC_OPTIMIZATION_ALLOWDOUBLENEST    10
#define CV_GLCMDESC_OPTIMIZATION_ALLOWTRIPLENEST    11
#define CV_GLCMDESC_OPTIMIZATION_HISTOGRAM          4

#define CV_GLCMDESC_ENTROPY                         0
#define CV_GLCMDESC_ENERGY                          1
#define CV_GLCMDESC_HOMOGENITY                      2
#define CV_GLCMDESC_CONTRAST                        3
#define CV_GLCMDESC_CLUSTERTENDENCY                 4
#define CV_GLCMDESC_CLUSTERSHADE                    5
#define CV_GLCMDESC_CORRELATION                     6
#define CV_GLCMDESC_CORRELATIONINFO1                7
#define CV_GLCMDESC_CORRELATIONINFO2                8
#define CV_GLCMDESC_MAXIMUMPROBABILITY              9

#define CV_GLCM_ALL                                 0
#define CV_GLCM_GLCM                                1
#define CV_GLCM_DESC                                2

typedef struct CvGLCM CvGLCM;

CVAPI(CvGLCM*) cvCreateGLCM( const IplImage* srcImage,
                                int stepMagnitude,
                                const int* stepDirections CV_DEFAULT(0),
                                int numStepDirections CV_DEFAULT(0),
                                int optimizationType CV_DEFAULT(CV_GLCM_OPTIMIZATION_NONE));

CVAPI(void) cvReleaseGLCM( CvGLCM** GLCM, int flag CV_DEFAULT(CV_GLCM_ALL));

CVAPI(void) cvCreateGLCMDescriptors( CvGLCM* destGLCM,
                                        int descriptorOptimizationType
                                        CV_DEFAULT(CV_GLCMDESC_OPTIMIZATION_ALLOWDOUBLENEST));

CVAPI(double) cvGetGLCMDescriptor( CvGLCM* GLCM, int step, int descriptor );

CVAPI(void) cvGetGLCMDescriptorStatistics( CvGLCM* GLCM, int descriptor,
                                              double* average, double* standardDeviation );

CVAPI(IplImage*) cvCreateGLCMImage( CvGLCM* GLCM, int step );

/****************************************************************************************\
*                                  Face eyes&mouth tracking                              *
\****************************************************************************************/


typedef struct CvFaceTracker CvFaceTracker;

#define CV_NUM_FACE_ELEMENTS    3 
enum CV_FACE_ELEMENTS
{
    CV_FACE_MOUTH = 0,
    CV_FACE_LEFT_EYE = 1,
    CV_FACE_RIGHT_EYE = 2
};

CVAPI(CvFaceTracker*) cvInitFaceTracker(CvFaceTracker* pFaceTracking, const IplImage* imgGray,
                                                CvRect* pRects, int nRects);
CVAPI(int) cvTrackFace( CvFaceTracker* pFaceTracker, IplImage* imgGray,
                              CvRect* pRects, int nRects,
                              CvPoint* ptRotate, double* dbAngleRotate);
CVAPI(void) cvReleaseFaceTracker(CvFaceTracker** ppFaceTracker);


typedef struct CvFace
{
    CvRect MouthRect;
    CvRect LeftEyeRect;
    CvRect RightEyeRect;
} CvFaceData;

CvSeq * cvFindFace(IplImage * Image,CvMemStorage* storage);
CvSeq * cvPostBoostingFindFace(IplImage * Image,CvMemStorage* storage);


/****************************************************************************************\
*                                         3D Tracker                                     *
\****************************************************************************************/

typedef unsigned char CvBool;

typedef struct
{
    int id;
    CvPoint2D32f p; // pgruebele: So we do not loose precision, this needs to be float
} Cv3dTracker2dTrackedObject;

CV_INLINE Cv3dTracker2dTrackedObject cv3dTracker2dTrackedObject(int id, CvPoint2D32f p)
{
    Cv3dTracker2dTrackedObject r;
    r.id = id;
    r.p = p;
    return r;
}

typedef struct
{
    int id;
    CvPoint3D32f p;             // location of the tracked object
} Cv3dTrackerTrackedObject;

CV_INLINE Cv3dTrackerTrackedObject cv3dTrackerTrackedObject(int id, CvPoint3D32f p)
{
    Cv3dTrackerTrackedObject r;
    r.id = id;
    r.p = p;
    return r;
}

typedef struct
{
    CvBool valid;
    float mat[4][4];              /* maps camera coordinates to world coordinates */
    CvPoint2D32f principal_point; /* copied from intrinsics so this structure */
                                  /* has all the info we need */
} Cv3dTrackerCameraInfo;

typedef struct
{
    CvPoint2D32f principal_point;
    float focal_length[2];
    float distortion[4];
} Cv3dTrackerCameraIntrinsics;

CVAPI(CvBool) cv3dTrackerCalibrateCameras(int num_cameras,
                     const Cv3dTrackerCameraIntrinsics camera_intrinsics[], /* size is num_cameras */
                     CvSize etalon_size,
                     float square_size,
                     IplImage *samples[],                                   /* size is num_cameras */
                     Cv3dTrackerCameraInfo camera_info[]);                  /* size is num_cameras */

CVAPI(int)  cv3dTrackerLocateObjects(int num_cameras, int num_objects,
                   const Cv3dTrackerCameraInfo camera_info[],        /* size is num_cameras */
                   const Cv3dTracker2dTrackedObject tracking_info[], /* size is num_objects*num_cameras */
                   Cv3dTrackerTrackedObject tracked_objects[]);      /* size is num_objects */
/****************************************************************************************
 tracking_info is a rectangular array; one row per camera, num_objects elements per row.
 The id field of any unused slots must be -1. Ids need not be ordered or consecutive. On
 completion, the return value is the number of objects located; i.e., the number of objects
 visible by more than one camera. The id field of any unused slots in tracked objects is
 set to -1.
****************************************************************************************/


/****************************************************************************************\
*                           Skeletons and Linear-Contour Models                          *
\****************************************************************************************/

typedef enum CvLeeParameters
{
    CV_LEE_INT = 0,
    CV_LEE_FLOAT = 1,
    CV_LEE_DOUBLE = 2,
    CV_LEE_AUTO = -1,
    CV_LEE_ERODE = 0,
    CV_LEE_ZOOM = 1,
    CV_LEE_NON = 2
} CvLeeParameters;

#define CV_NEXT_VORONOISITE2D( SITE ) ((SITE)->edge[0]->site[((SITE)->edge[0]->site[0] == (SITE))])
#define CV_PREV_VORONOISITE2D( SITE ) ((SITE)->edge[1]->site[((SITE)->edge[1]->site[0] == (SITE))])
#define CV_FIRST_VORONOIEDGE2D( SITE ) ((SITE)->edge[0])
#define CV_LAST_VORONOIEDGE2D( SITE ) ((SITE)->edge[1])
#define CV_NEXT_VORONOIEDGE2D( EDGE, SITE ) ((EDGE)->next[(EDGE)->site[0] != (SITE)])
#define CV_PREV_VORONOIEDGE2D( EDGE, SITE ) ((EDGE)->next[2 + ((EDGE)->site[0] != (SITE))])
#define CV_VORONOIEDGE2D_BEGINNODE( EDGE, SITE ) ((EDGE)->node[((EDGE)->site[0] != (SITE))])
#define CV_VORONOIEDGE2D_ENDNODE( EDGE, SITE ) ((EDGE)->node[((EDGE)->site[0] == (SITE))])
#define CV_TWIN_VORONOISITE2D( SITE, EDGE ) ( (EDGE)->site[((EDGE)->site[0] == (SITE))]) 

#define CV_VORONOISITE2D_FIELDS()    \
    struct CvVoronoiNode2D *node[2]; \
    struct CvVoronoiEdge2D *edge[2];

typedef struct CvVoronoiSite2D
{
    CV_VORONOISITE2D_FIELDS()
    struct CvVoronoiSite2D *next[2];
} CvVoronoiSite2D;

#define CV_VORONOIEDGE2D_FIELDS()    \
    struct CvVoronoiNode2D *node[2]; \
    struct CvVoronoiSite2D *site[2]; \
    struct CvVoronoiEdge2D *next[4];

typedef struct CvVoronoiEdge2D
{
    CV_VORONOIEDGE2D_FIELDS()
} CvVoronoiEdge2D;

#define CV_VORONOINODE2D_FIELDS()       \
    CV_SET_ELEM_FIELDS(CvVoronoiNode2D) \
    CvPoint2D32f pt;                    \
    float radius;

typedef struct CvVoronoiNode2D
{
    CV_VORONOINODE2D_FIELDS()
} CvVoronoiNode2D;

#define CV_VORONOIDIAGRAM2D_FIELDS() \
    CV_GRAPH_FIELDS()                \
    CvSet *sites;

typedef struct CvVoronoiDiagram2D
{
    CV_VORONOIDIAGRAM2D_FIELDS()
} CvVoronoiDiagram2D;

/* Computes Voronoi Diagram for given polygons with holes */
CVAPI(int)  cvVoronoiDiagramFromContour(CvSeq* ContourSeq,
                                           CvVoronoiDiagram2D** VoronoiDiagram,
                                           CvMemStorage* VoronoiStorage,
                                           CvLeeParameters contour_type CV_DEFAULT(CV_LEE_INT),
                                           int contour_orientation CV_DEFAULT(-1),
                                           int attempt_number CV_DEFAULT(10));

/* Computes Voronoi Diagram for domains in given image */
CVAPI(int)  cvVoronoiDiagramFromImage(IplImage* pImage,
                                         CvSeq** ContourSeq,
                                         CvVoronoiDiagram2D** VoronoiDiagram,
                                         CvMemStorage* VoronoiStorage,
                                         CvLeeParameters regularization_method CV_DEFAULT(CV_LEE_NON),
                                         float approx_precision CV_DEFAULT(CV_LEE_AUTO));

/* Deallocates the storage */
CVAPI(void) cvReleaseVoronoiStorage(CvVoronoiDiagram2D* VoronoiDiagram,
                                          CvMemStorage** pVoronoiStorage);

/*********************** Linear-Contour Model ****************************/

struct CvLCMEdge;
struct CvLCMNode;

typedef struct CvLCMEdge
{
    CV_GRAPH_EDGE_FIELDS() 
    CvSeq* chain;
    float width;
    int index1;
    int index2;
} CvLCMEdge;

typedef struct CvLCMNode
{
    CV_GRAPH_VERTEX_FIELDS()
    CvContour* contour; 
} CvLCMNode;


/* Computes hybrid model from Voronoi Diagram */
CVAPI(CvGraph*) cvLinearContorModelFromVoronoiDiagram(CvVoronoiDiagram2D* VoronoiDiagram,
                                                         float maxWidth);

/* Releases hybrid model storage */
CVAPI(int) cvReleaseLinearContorModelStorage(CvGraph** Graph);


/* two stereo-related functions */

CVAPI(void) cvInitPerspectiveTransform( CvSize size, const CvPoint2D32f vertex[4], double matrix[3][3],
                                              CvArr* rectMap );

/*CVAPI(void) cvInitStereoRectification( CvStereoCamera* params,
                                             CvArr* rectMap1, CvArr* rectMap2,
                                             int do_undistortion );*/

/*************************** View Morphing Functions ************************/

typedef struct CvMatrix3
{
    float m[3][3];
} CvMatrix3;    
    
/* The order of the function corresponds to the order they should appear in
   the view morphing pipeline */ 

/* Finds ending points of scanlines on left and right images of stereo-pair */
CVAPI(void)  cvMakeScanlines( const CvMatrix3* matrix, CvSize  img_size,
                              int*  scanlines1, int*  scanlines2,
                              int*  lengths1, int*  lengths2,
                              int*  line_count );

/* Grab pixel values from scanlines and stores them sequentially
   (some sort of perspective image transform) */
CVAPI(void)  cvPreWarpImage( int       line_count,
                             IplImage* img,
                             uchar*    dst,
                             int*      dst_nums,
                             int*      scanlines);

/* Approximate each grabbed scanline by a sequence of runs
   (lossy run-length compression) */
CVAPI(void)  cvFindRuns( int    line_count,
                         uchar* prewarp1,
                         uchar* prewarp2,
                         int*   line_lengths1,
                         int*   line_lengths2,
                         int*   runs1,
                         int*   runs2,
                         int*   num_runs1,
                         int*   num_runs2);

/* Compares two sets of compressed scanlines */
CVAPI(void)  cvDynamicCorrespondMulti( int  line_count,
                                       int* first,
                                       int* first_runs,
                                       int* second,
                                       int* second_runs,
                                       int* first_corr,
                                       int* second_corr);

/* Finds scanline ending coordinates for some intermediate "virtual" camera position */
CVAPI(void)  cvMakeAlphaScanlines( int*  scanlines1,
                                   int*  scanlines2,
                                   int*  scanlinesA,
                                   int*  lengths,
                                   int   line_count,
                                   float alpha);

/* Blends data of the left and right image scanlines to get
   pixel values of "virtual" image scanlines */
CVAPI(void)  cvMorphEpilinesMulti( int    line_count,
                                   uchar* first_pix,
                                   int*   first_num,
                                   uchar* second_pix,
                                   int*   second_num,
                                   uchar* dst_pix,
                                   int*   dst_num,
                                   float  alpha,
                                   int*   first,
                                   int*   first_runs,
                                   int*   second,
                                   int*   second_runs,
                                   int*   first_corr,
                                   int*   second_corr);

/* Does reverse warping of the morphing result to make
   it fill the destination image rectangle */
CVAPI(void)  cvPostWarpImage( int       line_count,
                              uchar*    src,
                              int*      src_nums,
                              IplImage* img,
                              int*      scanlines);

/* Deletes Moire (missed pixels that appear due to discretization) */
CVAPI(void)  cvDeleteMoire( IplImage*  img );


typedef struct CvConDensation
{
    int MP;
    int DP;
    float* DynamMatr;       /* Matrix of the linear Dynamics system  */
    float* State;           /* Vector of State                       */
    int SamplesNum;         /* Number of the Samples                 */
    float** flSamples;      /* arr of the Sample Vectors             */
    float** flNewSamples;   /* temporary array of the Sample Vectors */
    float* flConfidence;    /* Confidence for each Sample            */
    float* flCumulative;    /* Cumulative confidence                 */
    float* Temp;            /* Temporary vector                      */
    float* RandomSample;    /* RandomVector to update sample set     */
    struct CvRandState* RandS; /* Array of structures to generate random vectors */
} CvConDensation;
                               
/* Creates ConDensation filter state */
CVAPI(CvConDensation*)  cvCreateConDensation( int dynam_params,
                                             int measure_params,
                                             int sample_count );

/* Releases ConDensation filter state */
CVAPI(void)  cvReleaseConDensation( CvConDensation** condens );

/* Updates ConDensation filter by time (predict future state of the system) */
CVAPI(void)  cvConDensUpdateByTime( CvConDensation* condens);

/* Initializes ConDensation filter samples  */
CVAPI(void)  cvConDensInitSampleSet( CvConDensation* condens, CvMat* lower_bound, CvMat* upper_bound );                               

CV_INLINE int iplWidth( const IplImage* img )
{
    return !img ? 0 : !img->roi ? img->width : img->roi->width;
}

CV_INLINE int iplHeight( const IplImage* img )
{
    return !img ? 0 : !img->roi ? img->height : img->roi->height;
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

/****************************************************************************************\
*                                   Calibration engine                                   *
\****************************************************************************************/

typedef enum CvCalibEtalonType
{
    CV_CALIB_ETALON_USER = -1,
    CV_CALIB_ETALON_CHESSBOARD = 0,
    CV_CALIB_ETALON_CHECKERBOARD = CV_CALIB_ETALON_CHESSBOARD
}
CvCalibEtalonType;

class CV_EXPORTS CvCalibFilter
{
public:
    /* Constructor & destructor */
    CvCalibFilter();
    virtual ~CvCalibFilter();

    /* Sets etalon type - one for all cameras.
       etalonParams is used in case of pre-defined etalons (such as chessboard).
       Number of elements in etalonParams is determined by etalonType.
       E.g., if etalon type is CV_ETALON_TYPE_CHESSBOARD then:
         etalonParams[0] is number of squares per one side of etalon
         etalonParams[1] is number of squares per another side of etalon
         etalonParams[2] is linear size of squares in the board in arbitrary units.
       pointCount & points are used in case of
       CV_CALIB_ETALON_USER (user-defined) etalon. */
    virtual bool
        SetEtalon( CvCalibEtalonType etalonType, double* etalonParams,
                   int pointCount = 0, CvPoint2D32f* points = 0 );

    /* Retrieves etalon parameters/or and points */
    virtual CvCalibEtalonType
        GetEtalon( int* paramCount = 0, const double** etalonParams = 0,
                   int* pointCount = 0, const CvPoint2D32f** etalonPoints = 0 ) const;

    /* Sets number of cameras calibrated simultaneously. It is equal to 1 initially */
    virtual void SetCameraCount( int cameraCount );

    /* Retrieves number of cameras */
    int GetCameraCount() const { return cameraCount; }

    /* Starts cameras calibration */
    virtual bool SetFrames( int totalFrames );
    
    /* Stops cameras calibration */
    virtual void Stop( bool calibrate = false );

    /* Retrieves number of cameras */
    bool IsCalibrated() const { return isCalibrated; }

    /* Feeds another serie of snapshots (one per each camera) to filter.
       Etalon points on these images are found automatically.
       If the function can't locate points, it returns false */
    virtual bool FindEtalon( IplImage** imgs );

    /* The same but takes matrices */
    virtual bool FindEtalon( CvMat** imgs );

    /* Lower-level function for feeding filter with already found etalon points.
       Array of point arrays for each camera is passed. */
    virtual bool Push( const CvPoint2D32f** points = 0 );

    /* Returns total number of accepted frames and, optionally,
       total number of frames to collect */
    virtual int GetFrameCount( int* framesTotal = 0 ) const;

    /* Retrieves camera parameters for specified camera.
       If camera is not calibrated the function returns 0 */
    virtual const CvCamera* GetCameraParams( int idx = 0 ) const;

    virtual const CvStereoCamera* GetStereoParams() const;

    /* Sets camera parameters for all cameras */
    virtual bool SetCameraParams( CvCamera* params );

    /* Saves all camera parameters to file */
    virtual bool SaveCameraParams( const char* filename );
    
    /* Loads all camera parameters from file */
    virtual bool LoadCameraParams( const char* filename );

    /* Undistorts images using camera parameters. Some of src pointers can be NULL. */
    virtual bool Undistort( IplImage** src, IplImage** dst );

    /* Undistorts images using camera parameters. Some of src pointers can be NULL. */
    virtual bool Undistort( CvMat** src, CvMat** dst );

    /* Returns array of etalon points detected/partally detected
       on the latest frame for idx-th camera */
    virtual bool GetLatestPoints( int idx, CvPoint2D32f** pts,
                                                  int* count, bool* found );

    /* Draw the latest detected/partially detected etalon */
    virtual void DrawPoints( IplImage** dst );

    /* Draw the latest detected/partially detected etalon */
    virtual void DrawPoints( CvMat** dst );

    virtual bool Rectify( IplImage** srcarr, IplImage** dstarr );
    virtual bool Rectify( CvMat** srcarr, CvMat** dstarr );

protected:

    enum { MAX_CAMERAS = 3 };

    /* etalon data */
    CvCalibEtalonType  etalonType;
    int     etalonParamCount;
    double* etalonParams;
    int     etalonPointCount;
    CvPoint2D32f* etalonPoints;
    CvSize  imgSize;
    CvMat*  grayImg;
    CvMat*  tempImg;
    CvMemStorage* storage;

    /* camera data */
    int     cameraCount;
    CvCamera cameraParams[MAX_CAMERAS];
    CvStereoCamera stereo;
    CvPoint2D32f* points[MAX_CAMERAS];
    CvMat*  undistMap[MAX_CAMERAS][2];
    CvMat*  undistImg;
    int     latestCounts[MAX_CAMERAS];
    CvPoint2D32f* latestPoints[MAX_CAMERAS];
    CvMat*  rectMap[MAX_CAMERAS][2];

    /* Added by Valery */
    //CvStereoCamera stereoParams;

    int     maxPoints;
    int     framesTotal;
    int     framesAccepted;
    bool    isCalibrated;
};

#include <iosfwd>
#include <limits>

class CV_EXPORTS CvImage
{
public:
    CvImage() : image(0), refcount(0) {}
    CvImage( CvSize size, int depth, int channels )
    {
        image = cvCreateImage( size, depth, channels );
        refcount = image ? new int(1) : 0;
    }
    
    CvImage( IplImage* img ) : image(img)
    {
        refcount = image ? new int(1) : 0;
    }
    
    CvImage( const CvImage& img ) : image(img.image), refcount(img.refcount)
    {
        if( refcount ) ++(*refcount);
    }
    
    CvImage( const char* filename, const char* imgname=0, int color=-1 ) : image(0), refcount(0)
    { load( filename, imgname, color ); }
    
    CvImage( CvFileStorage* fs, const char* mapname, const char* imgname ) : image(0), refcount(0)
    { read( fs, mapname, imgname ); }
    
    CvImage( CvFileStorage* fs, const char* seqname, int idx ) : image(0), refcount(0)
    { read( fs, seqname, idx ); }
    
    ~CvImage()
    {
        if( refcount && !(--*refcount) )
        {
            cvReleaseImage( &image );
            delete refcount;
        }
    }
    
    CvImage clone() { return CvImage(image ? cvCloneImage(image) : 0); }
    
    void create( CvSize size, int depth, int channels )
    {
        if( !image || !refcount ||
           image->width != size.width || image->height != size.height ||
           image->depth != depth || image->nChannels != channels )
            attach( cvCreateImage( size, depth, channels ));
    }
    
    void release() { detach(); }
    void clear() { detach(); }
    
    void attach( IplImage* img, bool use_refcount=true )
    {
        if( refcount && --*refcount == 0 )
        {
            cvReleaseImage( &image );
            delete refcount;
        }
        image = img;
        refcount = use_refcount && image ? new int(1) : 0;
    }
    
    void detach()
    {
        if( refcount && --*refcount == 0 )
        {
            cvReleaseImage( &image );
            delete refcount;
        }
        image = 0;
        refcount = 0;
    }
    
    bool load( const char* filename, const char* imgname=0, int color=-1 );
    bool read( CvFileStorage* fs, const char* mapname, const char* imgname );
    bool read( CvFileStorage* fs, const char* seqname, int idx );
    void save( const char* filename, const char* imgname, const int* params=0 );
    void write( CvFileStorage* fs, const char* imgname );
    
    void show( const char* window_name );
    bool is_valid() { return image != 0; }
    
    int width() const { return image ? image->width : 0; }
    int height() const { return image ? image->height : 0; }
    
    CvSize size() const { return image ? cvSize(image->width, image->height) : cvSize(0,0); }
    
    CvSize roi_size() const
    {
        return !image ? cvSize(0,0) :
        !image->roi ? cvSize(image->width,image->height) :
        cvSize(image->roi->width, image->roi->height);
    }
    
    CvRect roi() const
    {
        return !image ? cvRect(0,0,0,0) :
        !image->roi ? cvRect(0,0,image->width,image->height) :
        cvRect(image->roi->xOffset,image->roi->yOffset,
               image->roi->width,image->roi->height);
    }
    
    int coi() const { return !image || !image->roi ? 0 : image->roi->coi; }
    
    void set_roi(CvRect roi) { cvSetImageROI(image,roi); }
    void reset_roi() { cvResetImageROI(image); }
    void set_coi(int coi) { cvSetImageCOI(image,coi); }
    int depth() const { return image ? image->depth : 0; }
    int channels() const { return image ? image->nChannels : 0; }
    int pix_size() const { return image ? ((image->depth & 255)>>3)*image->nChannels : 0; }
    
    uchar* data() { return image ? (uchar*)image->imageData : 0; }
    const uchar* data() const { return image ? (const uchar*)image->imageData : 0; }
    int step() const { return image ? image->widthStep : 0; }
    int origin() const { return image ? image->origin : 0; }
    
    uchar* roi_row(int y)
    {
        assert(0<=y);
        assert(!image ?
               1 : image->roi ?
               y<image->roi->height : y<image->height);
        
        return !image ? 0 :
        !image->roi ?
        (uchar*)(image->imageData + y*image->widthStep) :
        (uchar*)(image->imageData + (y+image->roi->yOffset)*image->widthStep +
                 image->roi->xOffset*((image->depth & 255)>>3)*image->nChannels);
    }
    
    const uchar* roi_row(int y) const
    {
        assert(0<=y);
        assert(!image ?
               1 : image->roi ?
               y<image->roi->height : y<image->height);
        
        return !image ? 0 :
        !image->roi ?
        (const uchar*)(image->imageData + y*image->widthStep) :
        (const uchar*)(image->imageData + (y+image->roi->yOffset)*image->widthStep +
                       image->roi->xOffset*((image->depth & 255)>>3)*image->nChannels);
    }
    
    operator const IplImage* () const { return image; }
    operator IplImage* () { return image; }
    
    CvImage& operator = (const CvImage& img)
    {
        if( img.refcount )
            ++*img.refcount;
        if( refcount && !(--*refcount) )
            cvReleaseImage( &image );
        image=img.image;
        refcount=img.refcount;
        return *this;
    }
    
protected:
    IplImage* image;
    int* refcount;
};


class CV_EXPORTS CvMatrix
{
public:
    CvMatrix() : matrix(0) {}
    CvMatrix( int rows, int cols, int type )
    { matrix = cvCreateMat( rows, cols, type ); }
    
    CvMatrix( int rows, int cols, int type, CvMat* hdr,
             void* data=0, int step=CV_AUTOSTEP )
    { matrix = cvInitMatHeader( hdr, rows, cols, type, data, step ); }
    
    CvMatrix( int rows, int cols, int type, CvMemStorage* storage, bool alloc_data=true );
    
    CvMatrix( int rows, int cols, int type, void* data, int step=CV_AUTOSTEP )
    { matrix = cvCreateMatHeader( rows, cols, type );
        cvSetData( matrix, data, step ); }
    
    CvMatrix( CvMat* m )
    { matrix = m; }
    
    CvMatrix( const CvMatrix& m )
    {
        matrix = m.matrix;
        addref();
    }
    
    CvMatrix( const char* filename, const char* matname=0, int color=-1 ) : matrix(0)
    {  load( filename, matname, color ); }
    
    CvMatrix( CvFileStorage* fs, const char* mapname, const char* matname ) : matrix(0)
    {  read( fs, mapname, matname ); }
    
    CvMatrix( CvFileStorage* fs, const char* seqname, int idx ) : matrix(0)
    {  read( fs, seqname, idx ); }
    
    ~CvMatrix()
    {
        release();
    }
    
    CvMatrix clone() { return CvMatrix(matrix ? cvCloneMat(matrix) : 0); }
    
    void set( CvMat* m, bool add_ref )
    {
        release();
        matrix = m;
        if( add_ref )
            addref();
    }
    
    void create( int rows, int cols, int type )
    {
        if( !matrix || !matrix->refcount ||
           matrix->rows != rows || matrix->cols != cols ||
           CV_MAT_TYPE(matrix->type) != type )
            set( cvCreateMat( rows, cols, type ), false );
    }
    
    void addref() const
    {
        if( matrix )
        {
            if( matrix->hdr_refcount )
                ++matrix->hdr_refcount;
            else if( matrix->refcount )
                ++*matrix->refcount;
        }
    }
    
    void release()
    {
        if( matrix )
        {
            if( matrix->hdr_refcount )
            {
                if( --matrix->hdr_refcount == 0 )
                    cvReleaseMat( &matrix );
            }
            else if( matrix->refcount )
            {
                if( --*matrix->refcount == 0 )
                    cvFree( &matrix->refcount );
            }
            matrix = 0;
        }
    }
    
    void clear()
    {
        release();
    }
    
    bool load( const char* filename, const char* matname=0, int color=-1 );
    bool read( CvFileStorage* fs, const char* mapname, const char* matname );
    bool read( CvFileStorage* fs, const char* seqname, int idx );
    void save( const char* filename, const char* matname, const int* params=0 );
    void write( CvFileStorage* fs, const char* matname );
    
    void show( const char* window_name );
    
    bool is_valid() { return matrix != 0; }
    
    int rows() const { return matrix ? matrix->rows : 0; }
    int cols() const { return matrix ? matrix->cols : 0; }
    
    CvSize size() const
    {
        return !matrix ? cvSize(0,0) : cvSize(matrix->rows,matrix->cols);
    }
    
    int type() const { return matrix ? CV_MAT_TYPE(matrix->type) : 0; }
    int depth() const { return matrix ? CV_MAT_DEPTH(matrix->type) : 0; }
    int channels() const { return matrix ? CV_MAT_CN(matrix->type) : 0; }
    int pix_size() const { return matrix ? CV_ELEM_SIZE(matrix->type) : 0; }
    
    uchar* data() { return matrix ? matrix->data.ptr : 0; }
    const uchar* data() const { return matrix ? matrix->data.ptr : 0; }
    int step() const { return matrix ? matrix->step : 0; }
    
    void set_data( void* data, int step=CV_AUTOSTEP )
    { cvSetData( matrix, data, step ); }
    
    uchar* row(int i) { return !matrix ? 0 : matrix->data.ptr + i*matrix->step; }
    const uchar* row(int i) const
    { return !matrix ? 0 : matrix->data.ptr + i*matrix->step; }
    
    operator const CvMat* () const { return matrix; }
    operator CvMat* () { return matrix; }
    
    CvMatrix& operator = (const CvMatrix& _m)
    {
        _m.addref();
        release();
        matrix = _m.matrix;
        return *this;
    }
    
protected:
    CvMat* matrix;
};

/****************************************************************************************\
 *                                       CamShiftTracker                                  *
 \****************************************************************************************/

class CV_EXPORTS CvCamShiftTracker
{
public:
    
    CvCamShiftTracker();
    virtual ~CvCamShiftTracker();
    
    /**** Characteristics of the object that are calculated by track_object method *****/
    float   get_orientation() const // orientation of the object in degrees
    { return m_box.angle; }
    float   get_length() const // the larger linear size of the object
    { return m_box.size.height; }
    float   get_width() const // the smaller linear size of the object
    { return m_box.size.width; }
    CvPoint2D32f get_center() const // center of the object
    { return m_box.center; }
    CvRect get_window() const // bounding rectangle for the object
    { return m_comp.rect; }
    
    /*********************** Tracking parameters ************************/
    int     get_threshold() const // thresholding value that applied to back project
    { return m_threshold; }
    
    int     get_hist_dims( int* dims = 0 ) const // returns number of histogram dimensions and sets
    { return m_hist ? cvGetDims( m_hist->bins, dims ) : 0; }
    
    int     get_min_ch_val( int channel ) const // get the minimum allowed value of the specified channel
    { return m_min_ch_val[channel]; }
    
    int     get_max_ch_val( int channel ) const // get the maximum allowed value of the specified channel
    { return m_max_ch_val[channel]; }
    
    // set initial object rectangle (must be called before initial calculation of the histogram)
    bool    set_window( CvRect window)
    { m_comp.rect = window; return true; }
    
    bool    set_threshold( int threshold ) // threshold applied to the histogram bins
    { m_threshold = threshold; return true; }
    
    bool    set_hist_bin_range( int dim, int min_val, int max_val );
    
    bool    set_hist_dims( int c_dims, int* dims );// set the histogram parameters
    
    bool    set_min_ch_val( int channel, int val ) // set the minimum allowed value of the specified channel
    { m_min_ch_val[channel] = val; return true; }
    bool    set_max_ch_val( int channel, int val ) // set the maximum allowed value of the specified channel
    { m_max_ch_val[channel] = val; return true; }
    
    /************************ The processing methods *********************************/
    // update object position
    virtual bool  track_object( const IplImage* cur_frame );
    
    // update object histogram
    virtual bool  update_histogram( const IplImage* cur_frame );
    
    // reset histogram
    virtual void  reset_histogram();
    
    /************************ Retrieving internal data *******************************/
    // get back project image
    virtual IplImage* get_back_project()
    { return m_back_project; }
    
    float query( int* bin ) const
    { return m_hist ? (float)cvGetRealND(m_hist->bins, bin) : 0.f; }
    
protected:
    
    // internal method for color conversion: fills m_color_planes group
    virtual void color_transform( const IplImage* img );
    
    CvHistogram* m_hist;
    
    CvBox2D    m_box;
    CvConnectedComp m_comp;
    
    float      m_hist_ranges_data[CV_MAX_DIM][2];
    float*     m_hist_ranges[CV_MAX_DIM];
    
    int        m_min_ch_val[CV_MAX_DIM];
    int        m_max_ch_val[CV_MAX_DIM];
    int        m_threshold;
    
    IplImage*  m_color_planes[CV_MAX_DIM];
    IplImage*  m_back_project;
    IplImage*  m_temp;
    IplImage*  m_mask;
};


namespace cv
{

/*!
 The Patch Generator class 
 */
class CV_EXPORTS PatchGenerator
{
public:
    PatchGenerator();
    PatchGenerator(double _backgroundMin, double _backgroundMax,
                   double _noiseRange, bool _randomBlur=true,
                   double _lambdaMin=0.6, double _lambdaMax=1.5,
                   double _thetaMin=-CV_PI, double _thetaMax=CV_PI,
                   double _phiMin=-CV_PI, double _phiMax=CV_PI );
    void operator()(const Mat& image, Point2f pt, Mat& patch, Size patchSize, RNG& rng) const;
    void operator()(const Mat& image, const Mat& transform, Mat& patch,
                    Size patchSize, RNG& rng) const;
    void warpWholeImage(const Mat& image, Mat& matT, Mat& buf,
                        CV_OUT Mat& warped, int border, RNG& rng) const;
    void generateRandomTransform(Point2f srcCenter, Point2f dstCenter,
                                 CV_OUT Mat& transform, RNG& rng,
                                 bool inverse=false) const;
    void setAffineParam(double lambda, double theta, double phi);
    
    double backgroundMin, backgroundMax;
    double noiseRange;
    bool randomBlur;
    double lambdaMin, lambdaMax;
    double thetaMin, thetaMax;
    double phiMin, phiMax;
};


class CV_EXPORTS LDetector
{
public:
    LDetector();
    LDetector(int _radius, int _threshold, int _nOctaves,
              int _nViews, double _baseFeatureSize, double _clusteringDistance);
    void operator()(const Mat& image,
                    CV_OUT vector<KeyPoint>& keypoints,
                    int maxCount=0, bool scaleCoords=true) const;
    void operator()(const vector<Mat>& pyr,
                    CV_OUT vector<KeyPoint>& keypoints,
                    int maxCount=0, bool scaleCoords=true) const;
    void getMostStable2D(const Mat& image, CV_OUT vector<KeyPoint>& keypoints,
                         int maxCount, const PatchGenerator& patchGenerator) const;
    void setVerbose(bool verbose);
    
    void read(const FileNode& node);
    void write(FileStorage& fs, const String& name=String()) const;
    
    int radius;
    int threshold;
    int nOctaves;
    int nViews;
    bool verbose;
    
    double baseFeatureSize;
    double clusteringDistance;
};

typedef LDetector YAPE;

class CV_EXPORTS FernClassifier
{
public:
    FernClassifier();
    FernClassifier(const FileNode& node);
    FernClassifier(const vector<vector<Point2f> >& points,
                   const vector<Mat>& refimgs,
                   const vector<vector<int> >& labels=vector<vector<int> >(),
                   int _nclasses=0, int _patchSize=PATCH_SIZE,
                   int _signatureSize=DEFAULT_SIGNATURE_SIZE,
                   int _nstructs=DEFAULT_STRUCTS,
                   int _structSize=DEFAULT_STRUCT_SIZE,
                   int _nviews=DEFAULT_VIEWS,
                   int _compressionMethod=COMPRESSION_NONE,
                   const PatchGenerator& patchGenerator=PatchGenerator());
    virtual ~FernClassifier();
    virtual void read(const FileNode& n);
    virtual void write(FileStorage& fs, const String& name=String()) const;
    virtual void trainFromSingleView(const Mat& image,
                                     const vector<KeyPoint>& keypoints,
                                     int _patchSize=PATCH_SIZE,
                                     int _signatureSize=DEFAULT_SIGNATURE_SIZE,
                                     int _nstructs=DEFAULT_STRUCTS,
                                     int _structSize=DEFAULT_STRUCT_SIZE,
                                     int _nviews=DEFAULT_VIEWS,
                                     int _compressionMethod=COMPRESSION_NONE,
                                     const PatchGenerator& patchGenerator=PatchGenerator());
    virtual void train(const vector<vector<Point2f> >& points,
                       const vector<Mat>& refimgs,
                       const vector<vector<int> >& labels=vector<vector<int> >(),
                       int _nclasses=0, int _patchSize=PATCH_SIZE,
                       int _signatureSize=DEFAULT_SIGNATURE_SIZE,
                       int _nstructs=DEFAULT_STRUCTS,
                       int _structSize=DEFAULT_STRUCT_SIZE,
                       int _nviews=DEFAULT_VIEWS,
                       int _compressionMethod=COMPRESSION_NONE,
                       const PatchGenerator& patchGenerator=PatchGenerator());
    virtual int operator()(const Mat& img, Point2f kpt, vector<float>& signature) const;
    virtual int operator()(const Mat& patch, vector<float>& signature) const;
    virtual void clear();
    virtual bool empty() const;
    void setVerbose(bool verbose);
    
    int getClassCount() const;
    int getStructCount() const;
    int getStructSize() const;
    int getSignatureSize() const;
    int getCompressionMethod() const;
    Size getPatchSize() const;
    
    struct Feature
    {
        uchar x1, y1, x2, y2;
        Feature() : x1(0), y1(0), x2(0), y2(0) {}
        Feature(int _x1, int _y1, int _x2, int _y2)
        : x1((uchar)_x1), y1((uchar)_y1), x2((uchar)_x2), y2((uchar)_y2)
        {}
        template<typename _Tp> bool operator ()(const Mat_<_Tp>& patch) const
        { return patch(y1,x1) > patch(y2, x2); }
    };
    
    enum
    {
        PATCH_SIZE = 31,
        DEFAULT_STRUCTS = 50,
        DEFAULT_STRUCT_SIZE = 9,
        DEFAULT_VIEWS = 5000,
        DEFAULT_SIGNATURE_SIZE = 176,
        COMPRESSION_NONE = 0,
        COMPRESSION_RANDOM_PROJ = 1,
        COMPRESSION_PCA = 2,
        DEFAULT_COMPRESSION_METHOD = COMPRESSION_NONE
    };
    
protected:
    virtual void prepare(int _nclasses, int _patchSize, int _signatureSize,
                         int _nstructs, int _structSize,
                         int _nviews, int _compressionMethod);
    virtual void finalize(RNG& rng);
    virtual int getLeaf(int fidx, const Mat& patch) const;
    
    bool verbose;
    int nstructs;
    int structSize;
    int nclasses;
    int signatureSize;
    int compressionMethod;
    int leavesPerStruct;
    Size patchSize;
    vector<Feature> features;
    vector<int> classCounters;
    vector<float> posteriors;
};


/****************************************************************************************\
 *                                 Calonder Classifier                                    *
 \****************************************************************************************/

struct RTreeNode;

struct CV_EXPORTS BaseKeypoint
{
    int x;
    int y;
    IplImage* image;
    
    BaseKeypoint()
    : x(0), y(0), image(NULL)
    {}
    
    BaseKeypoint(int x, int y, IplImage* image)
    : x(x), y(y), image(image)
    {}
};

class CV_EXPORTS RandomizedTree
{
public:
    friend class RTreeClassifier;
    
    static const uchar PATCH_SIZE = 32;
    static const int DEFAULT_DEPTH = 9;
    static const int DEFAULT_VIEWS = 5000;
    static const size_t DEFAULT_REDUCED_NUM_DIM = 176;
    static float GET_LOWER_QUANT_PERC() { return .03f; }
    static float GET_UPPER_QUANT_PERC() { return .92f; }
    
    RandomizedTree();
    ~RandomizedTree();
    
    void train(vector<BaseKeypoint> const& base_set, RNG &rng,
               int depth, int views, size_t reduced_num_dim, int num_quant_bits);
    void train(vector<BaseKeypoint> const& base_set, RNG &rng,
               PatchGenerator &make_patch, int depth, int views, size_t reduced_num_dim,
               int num_quant_bits);
    
    // following two funcs are EXPERIMENTAL (do not use unless you know exactly what you do)
    static void quantizeVector(float *vec, int dim, int N, float bnds[2], int clamp_mode=0);
    static void quantizeVector(float *src, int dim, int N, float bnds[2], uchar *dst);
    
    // patch_data must be a 32x32 array (no row padding)
    float* getPosterior(uchar* patch_data);
    const float* getPosterior(uchar* patch_data) const;
    uchar* getPosterior2(uchar* patch_data);
    const uchar* getPosterior2(uchar* patch_data) const;
    
    void read(const char* file_name, int num_quant_bits);
    void read(std::istream &is, int num_quant_bits);
    void write(const char* file_name) const;
    void write(std::ostream &os) const;
    
    int classes() { return classes_; }
    int depth() { return depth_; }
    
    //void setKeepFloatPosteriors(bool b) { keep_float_posteriors_ = b; }
    void discardFloatPosteriors() { freePosteriors(1); }
    
    inline void applyQuantization(int num_quant_bits) { makePosteriors2(num_quant_bits); }
    
    // debug
    void savePosteriors(std::string url, bool append=false);
    void savePosteriors2(std::string url, bool append=false);
    
private:
    int classes_;
    int depth_;
    int num_leaves_;
    vector<RTreeNode> nodes_;
    float **posteriors_;        // 16-bytes aligned posteriors
    uchar **posteriors2_;     // 16-bytes aligned posteriors
    vector<int> leaf_counts_;
    
    void createNodes(int num_nodes, RNG &rng);
    void allocPosteriorsAligned(int num_leaves, int num_classes);
    void freePosteriors(int which);    // which: 1=posteriors_, 2=posteriors2_, 3=both
    void init(int classes, int depth, RNG &rng);
    void addExample(int class_id, uchar* patch_data);
    void finalize(size_t reduced_num_dim, int num_quant_bits);
    int getIndex(uchar* patch_data) const;
    inline float* getPosteriorByIndex(int index);
    inline const float* getPosteriorByIndex(int index) const;
    inline uchar* getPosteriorByIndex2(int index);
    inline const uchar* getPosteriorByIndex2(int index) const;
    //void makeRandomMeasMatrix(float *cs_phi, PHI_DISTR_TYPE dt, size_t reduced_num_dim);
    void convertPosteriorsToChar();
    void makePosteriors2(int num_quant_bits);
    void compressLeaves(size_t reduced_num_dim);
    void estimateQuantPercForPosteriors(float perc[2]);
};


inline uchar* getData(IplImage* image)
{
    return reinterpret_cast<uchar*>(image->imageData);
}

inline float* RandomizedTree::getPosteriorByIndex(int index)
{
    return const_cast<float*>(const_cast<const RandomizedTree*>(this)->getPosteriorByIndex(index));
}

inline const float* RandomizedTree::getPosteriorByIndex(int index) const
{
    return posteriors_[index];
}

inline uchar* RandomizedTree::getPosteriorByIndex2(int index)
{
    return const_cast<uchar*>(const_cast<const RandomizedTree*>(this)->getPosteriorByIndex2(index));
}

inline const uchar* RandomizedTree::getPosteriorByIndex2(int index) const
{
    return posteriors2_[index];
}

struct CV_EXPORTS RTreeNode
{
    short offset1, offset2;
    
    RTreeNode() {}
    RTreeNode(uchar x1, uchar y1, uchar x2, uchar y2)
    : offset1(y1*RandomizedTree::PATCH_SIZE + x1),
    offset2(y2*RandomizedTree::PATCH_SIZE + x2)
    {}
    
    //! Left child on 0, right child on 1
    inline bool operator() (uchar* patch_data) const
    {
        return patch_data[offset1] > patch_data[offset2];
    }
};

class CV_EXPORTS RTreeClassifier
{
public:
    static const int DEFAULT_TREES = 48;
    static const size_t DEFAULT_NUM_QUANT_BITS = 4;
    
    RTreeClassifier();
    void train(vector<BaseKeypoint> const& base_set,
               RNG &rng,
               int num_trees = RTreeClassifier::DEFAULT_TREES,
               int depth = RandomizedTree::DEFAULT_DEPTH,
               int views = RandomizedTree::DEFAULT_VIEWS,
               size_t reduced_num_dim = RandomizedTree::DEFAULT_REDUCED_NUM_DIM,
               int num_quant_bits = DEFAULT_NUM_QUANT_BITS);
    void train(vector<BaseKeypoint> const& base_set,
               RNG &rng,
               PatchGenerator &make_patch,
               int num_trees = RTreeClassifier::DEFAULT_TREES,
               int depth = RandomizedTree::DEFAULT_DEPTH,
               int views = RandomizedTree::DEFAULT_VIEWS,
               size_t reduced_num_dim = RandomizedTree::DEFAULT_REDUCED_NUM_DIM,
               int num_quant_bits = DEFAULT_NUM_QUANT_BITS);
    
    // sig must point to a memory block of at least classes()*sizeof(float|uchar) bytes
    void getSignature(IplImage *patch, uchar *sig) const;
    void getSignature(IplImage *patch, float *sig) const;
    void getSparseSignature(IplImage *patch, float *sig, float thresh) const;
    // TODO: deprecated in favor of getSignature overload, remove
    void getFloatSignature(IplImage *patch, float *sig) const { getSignature(patch, sig); }
    
    static int countNonZeroElements(float *vec, int n, double tol=1e-10);
    static inline void safeSignatureAlloc(uchar **sig, int num_sig=1, int sig_len=176);
    static inline uchar* safeSignatureAlloc(int num_sig=1, int sig_len=176);
    
    inline int classes() const { return classes_; }
    inline int original_num_classes() const { return original_num_classes_; }
    
    void setQuantization(int num_quant_bits);
    void discardFloatPosteriors();
    
    void read(const char* file_name);
    void read(std::istream &is);
    void write(const char* file_name) const;
    void write(std::ostream &os) const;
    
    // experimental and debug
    void saveAllFloatPosteriors(std::string file_url);
    void saveAllBytePosteriors(std::string file_url);
    void setFloatPosteriorsFromTextfile_176(std::string url);
    float countZeroElements();
    
    vector<RandomizedTree> trees_;
    
private:
    int classes_;
    int num_quant_bits_;
    mutable uchar **posteriors_;
    mutable unsigned short *ptemp_;
    int original_num_classes_;
    bool keep_floats_;
};

/****************************************************************************************\
*                                     One-Way Descriptor                                 *
\****************************************************************************************/
    
// CvAffinePose: defines a parameterized affine transformation of an image patch.
// An image patch is rotated on angle phi (in degrees), then scaled lambda1 times
// along horizontal and lambda2 times along vertical direction, and then rotated again
// on angle (theta - phi).
class CV_EXPORTS CvAffinePose
{
public:
    float phi;
    float theta;
    float lambda1;
    float lambda2;
};
    
class CV_EXPORTS OneWayDescriptor
{
public:
    OneWayDescriptor();
    ~OneWayDescriptor();
    
    // allocates memory for given descriptor parameters
    void Allocate(int pose_count, CvSize size, int nChannels);
    
    // GenerateSamples: generates affine transformed patches with averaging them over small transformation variations.
    // If external poses and transforms were specified, uses them instead of generating random ones
    // - pose_count: the number of poses to be generated
    // - frontal: the input patch (can be a roi in a larger image)
    // - norm: if nonzero, normalizes the output patch so that the sum of pixel intensities is 1
    void GenerateSamples(int pose_count, IplImage* frontal, int norm = 0);
    
    // GenerateSamplesFast: generates affine transformed patches with averaging them over small transformation variations.
    // Uses precalculated transformed pca components.
    // - frontal: the input patch (can be a roi in a larger image)
    // - pca_hr_avg: pca average vector
    // - pca_hr_eigenvectors: pca eigenvectors
    // - pca_descriptors: an array of precomputed descriptors of pca components containing their affine transformations
    //   pca_descriptors[0] corresponds to the average, pca_descriptors[1]-pca_descriptors[pca_dim] correspond to eigenvectors
    void GenerateSamplesFast(IplImage* frontal, CvMat* pca_hr_avg,
                             CvMat* pca_hr_eigenvectors, OneWayDescriptor* pca_descriptors);
    
    // sets the poses and corresponding transforms
    void SetTransforms(CvAffinePose* poses, CvMat** transforms);
    
    // Initialize: builds a descriptor.
    // - pose_count: the number of poses to build. If poses were set externally, uses them rather than generating random ones
    // - frontal: input patch. Can be a roi in a larger image
    // - feature_name: the feature name to be associated with the descriptor
    // - norm: if 1, the affine transformed patches are normalized so that their sum is 1
    void Initialize(int pose_count, IplImage* frontal, const char* feature_name = 0, int norm = 0);
    
    // InitializeFast: builds a descriptor using precomputed descriptors of pca components
    // - pose_count: the number of poses to build
    // - frontal: input patch. Can be a roi in a larger image
    // - feature_name: the feature name to be associated with the descriptor
    // - pca_hr_avg: average vector for PCA
    // - pca_hr_eigenvectors: PCA eigenvectors (one vector per row)
    // - pca_descriptors: precomputed descriptors of PCA components, the first descriptor for the average vector
    // followed by the descriptors for eigenvectors
    void InitializeFast(int pose_count, IplImage* frontal, const char* feature_name,
                        CvMat* pca_hr_avg, CvMat* pca_hr_eigenvectors, OneWayDescriptor* pca_descriptors);
    
    // ProjectPCASample: unwarps an image patch into a vector and projects it into PCA space
    // - patch: input image patch
    // - avg: PCA average vector
    // - eigenvectors: PCA eigenvectors, one per row
    // - pca_coeffs: output PCA coefficients
    void ProjectPCASample(IplImage* patch, CvMat* avg, CvMat* eigenvectors, CvMat* pca_coeffs) const;
    
    // InitializePCACoeffs: projects all warped patches into PCA space
    // - avg: PCA average vector
    // - eigenvectors: PCA eigenvectors, one per row
    void InitializePCACoeffs(CvMat* avg, CvMat* eigenvectors);
    
    // EstimatePose: finds the closest match between an input patch and a set of patches with different poses
    // - patch: input image patch
    // - pose_idx: the output index of the closest pose
    // - distance: the distance to the closest pose (L2 distance)
    void EstimatePose(IplImage* patch, int& pose_idx, float& distance) const;
    
    // EstimatePosePCA: finds the closest match between an input patch and a set of patches with different poses.
    // The distance between patches is computed in PCA space
    // - patch: input image patch
    // - pose_idx: the output index of the closest pose
    // - distance: distance to the closest pose (L2 distance in PCA space)
    // - avg: PCA average vector. If 0, matching without PCA is used
    // - eigenvectors: PCA eigenvectors, one per row
    void EstimatePosePCA(CvArr* patch, int& pose_idx, float& distance, CvMat* avg, CvMat* eigenvalues) const;
    
    // GetPatchSize: returns the size of each image patch after warping (2 times smaller than the input patch)
    CvSize GetPatchSize() const
    {
        return m_patch_size;
    }
    
    // GetInputPatchSize: returns the required size of the patch that the descriptor is built from
    // (2 time larger than the patch after warping)
    CvSize GetInputPatchSize() const
    {
        return cvSize(m_patch_size.width*2, m_patch_size.height*2);
    }
    
    // GetPatch: returns a patch corresponding to specified pose index
    // - index: pose index
    // - return value: the patch corresponding to specified pose index
    IplImage* GetPatch(int index);
    
    // GetPose: returns a pose corresponding to specified pose index
    // - index: pose index
    // - return value: the pose corresponding to specified pose index
    CvAffinePose GetPose(int index) const;
    
    // Save: saves all patches with different poses to a specified path
    void Save(const char* path);
    
    // ReadByName: reads a descriptor from a file storage
    // - fs: file storage
    // - parent: parent node
    // - name: node name
    // - return value: 1 if succeeded, 0 otherwise
    int ReadByName(CvFileStorage* fs, CvFileNode* parent, const char* name);
    
    // ReadByName: reads a descriptor from a file node
    // - parent: parent node
    // - name: node name
    // - return value: 1 if succeeded, 0 otherwise
    int ReadByName(const FileNode &parent, const char* name);
    
    // Write: writes a descriptor into a file storage
    // - fs: file storage
    // - name: node name
    void Write(CvFileStorage* fs, const char* name);
    
    // GetFeatureName: returns a name corresponding to a feature
    const char* GetFeatureName() const;
    
    // GetCenter: returns the center of the feature
    CvPoint GetCenter() const;
    
    void SetPCADimHigh(int pca_dim_high) {m_pca_dim_high = pca_dim_high;};
    void SetPCADimLow(int pca_dim_low) {m_pca_dim_low = pca_dim_low;};
    
    int GetPCADimLow() const;
    int GetPCADimHigh() const;
    
    CvMat** GetPCACoeffs() const {return m_pca_coeffs;}
    
protected:
    int m_pose_count; // the number of poses
    CvSize m_patch_size; // size of each image
    IplImage** m_samples; // an array of length m_pose_count containing the patch in different poses
    IplImage* m_input_patch;
    IplImage* m_train_patch;
    CvMat** m_pca_coeffs; // an array of length m_pose_count containing pca decomposition of the patch in different poses
    CvAffinePose* m_affine_poses; // an array of poses
    CvMat** m_transforms; // an array of affine transforms corresponding to poses
    
    string m_feature_name; // the name of the feature associated with the descriptor
    CvPoint m_center; // the coordinates of the feature (the center of the input image ROI)
    
    int m_pca_dim_high; // the number of descriptor pca components to use for generating affine poses
    int m_pca_dim_low; // the number of pca components to use for comparison
};


// OneWayDescriptorBase: encapsulates functionality for training/loading a set of one way descriptors
// and finding the nearest closest descriptor to an input feature
class CV_EXPORTS OneWayDescriptorBase
{
public:
    
    // creates an instance of OneWayDescriptor from a set of training files
    // - patch_size: size of the input (large) patch
    // - pose_count: the number of poses to generate for each descriptor
    // - train_path: path to training files
    // - pca_config: the name of the file that contains PCA for small patches (2 times smaller
    // than patch_size each dimension
    // - pca_hr_config: the name of the file that contains PCA for large patches (of patch_size size)
    // - pca_desc_config: the name of the file that contains descriptors of PCA components
    OneWayDescriptorBase(CvSize patch_size, int pose_count, const char* train_path = 0, const char* pca_config = 0,
                         const char* pca_hr_config = 0, const char* pca_desc_config = 0, int pyr_levels = 1,
                         int pca_dim_high = 100, int pca_dim_low = 100);
    
    OneWayDescriptorBase(CvSize patch_size, int pose_count, const string &pca_filename, const string &train_path = string(), const string &images_list = string(),
                         float _scale_min = 0.7f, float _scale_max=1.5f, float _scale_step=1.2f, int pyr_levels = 1,
                         int pca_dim_high = 100, int pca_dim_low = 100);
    
    
    virtual ~OneWayDescriptorBase();
    void clear ();
    
    
    // Allocate: allocates memory for a given number of descriptors
    void Allocate(int train_feature_count);
    
    // AllocatePCADescriptors: allocates memory for pca descriptors
    void AllocatePCADescriptors();
    
    // returns patch size
    CvSize GetPatchSize() const {return m_patch_size;};
    // returns the number of poses for each descriptor
    int GetPoseCount() const {return m_pose_count;};
    
    // returns the number of pyramid levels
    int GetPyrLevels() const {return m_pyr_levels;};
    
    // returns the number of descriptors
    int GetDescriptorCount() const {return m_train_feature_count;};
    
    // CreateDescriptorsFromImage: creates descriptors for each of the input features
    // - src: input image
    // - features: input features
    // - pyr_levels: the number of pyramid levels
    void CreateDescriptorsFromImage(IplImage* src, const vector<KeyPoint>& features);
    
    // CreatePCADescriptors: generates descriptors for PCA components, needed for fast generation of feature descriptors
    void CreatePCADescriptors();
    
    // returns a feature descriptor by feature index
    const OneWayDescriptor* GetDescriptor(int desc_idx) const {return &m_descriptors[desc_idx];};
    
    // FindDescriptor: finds the closest descriptor
    // - patch: input image patch
    // - desc_idx: output index of the closest descriptor to the input patch
    // - pose_idx: output index of the closest pose of the closest descriptor to the input patch
    // - distance: distance from the input patch to the closest feature pose
    // - _scales: scales of the input patch for each descriptor
    // - scale_ranges: input scales variation (float[2])
    void FindDescriptor(IplImage* patch, int& desc_idx, int& pose_idx, float& distance, float* _scale = 0, float* scale_ranges = 0) const;
    
    // - patch: input image patch
    // - n: number of the closest indexes
    // - desc_idxs: output indexes of the closest descriptor to the input patch (n)
    // - pose_idx: output indexes of the closest pose of the closest descriptor to the input patch (n)
    // - distances: distance from the input patch to the closest feature pose (n)
    // - _scales: scales of the input patch
    // - scale_ranges: input scales variation (float[2])
    void FindDescriptor(IplImage* patch, int n, vector<int>& desc_idxs, vector<int>& pose_idxs,
                        vector<float>& distances, vector<float>& _scales, float* scale_ranges = 0) const;
    
    // FindDescriptor: finds the closest descriptor
    // - src: input image
    // - pt: center of the feature
    // - desc_idx: output index of the closest descriptor to the input patch
    // - pose_idx: output index of the closest pose of the closest descriptor to the input patch
    // - distance: distance from the input patch to the closest feature pose
    void FindDescriptor(IplImage* src, cv::Point2f pt, int& desc_idx, int& pose_idx, float& distance) const;
    
    // InitializePoses: generates random poses
    void InitializePoses();
    
    // InitializeTransformsFromPoses: generates 2x3 affine matrices from poses (initializes m_transforms)
    void InitializeTransformsFromPoses();
    
    // InitializePoseTransforms: subsequently calls InitializePoses and InitializeTransformsFromPoses
    void InitializePoseTransforms();
    
    // InitializeDescriptor: initializes a descriptor
    // - desc_idx: descriptor index
    // - train_image: image patch (ROI is supported)
    // - feature_label: feature textual label
    void InitializeDescriptor(int desc_idx, IplImage* train_image, const char* feature_label);
    
    void InitializeDescriptor(int desc_idx, IplImage* train_image, const KeyPoint& keypoint, const char* feature_label);
    
    // InitializeDescriptors: load features from an image and create descriptors for each of them
    void InitializeDescriptors(IplImage* train_image, const vector<KeyPoint>& features,
                               const char* feature_label = "", int desc_start_idx = 0);
    
    // Write: writes this object to a file storage
    // - fs: output filestorage
    void Write (FileStorage &fs) const;
    
    // Read: reads OneWayDescriptorBase object from a file node
    // - fn: input file node    
    void Read (const FileNode &fn);
    
    // LoadPCADescriptors: loads PCA descriptors from a file
    // - filename: input filename
    int LoadPCADescriptors(const char* filename);
    
    // LoadPCADescriptors: loads PCA descriptors from a file node
    // - fn: input file node
    int LoadPCADescriptors(const FileNode &fn);
    
    // SavePCADescriptors: saves PCA descriptors to a file
    // - filename: output filename
    void SavePCADescriptors(const char* filename);
    
    // SavePCADescriptors: saves PCA descriptors to a file storage
    // - fs: output file storage
    void SavePCADescriptors(CvFileStorage* fs) const;
    
    // GeneratePCA: calculate and save PCA components and descriptors
    // - img_path: path to training PCA images directory
    // - images_list: filename with filenames of training PCA images
    void GeneratePCA(const char* img_path, const char* images_list, int pose_count=500);
    
    // SetPCAHigh: sets the high resolution pca matrices (copied to internal structures)
    void SetPCAHigh(CvMat* avg, CvMat* eigenvectors);
    
    // SetPCALow: sets the low resolution pca matrices (copied to internal structures)
    void SetPCALow(CvMat* avg, CvMat* eigenvectors);
    
    int GetLowPCA(CvMat** avg, CvMat** eigenvectors)
    {
        *avg = m_pca_avg;
        *eigenvectors = m_pca_eigenvectors;
        return m_pca_dim_low;
    };
    
    int GetPCADimLow() const {return m_pca_dim_low;};
    int GetPCADimHigh() const {return m_pca_dim_high;};
    
    void ConvertDescriptorsArrayToTree(); // Converting pca_descriptors array to KD tree
    
    // GetPCAFilename: get default PCA filename
    static string GetPCAFilename () { return "pca.yml"; }
    
    virtual bool empty() const { return m_train_feature_count <= 0 ? true : false; }
    
protected:
    CvSize m_patch_size; // patch size
    int m_pose_count; // the number of poses for each descriptor
    int m_train_feature_count; // the number of the training features
    OneWayDescriptor* m_descriptors; // array of train feature descriptors
    CvMat* m_pca_avg; // PCA average Vector for small patches
    CvMat* m_pca_eigenvectors; // PCA eigenvectors for small patches
    CvMat* m_pca_hr_avg; // PCA average Vector for large patches
    CvMat* m_pca_hr_eigenvectors; // PCA eigenvectors for large patches
    OneWayDescriptor* m_pca_descriptors; // an array of PCA descriptors
    
    cv::flann::Index* m_pca_descriptors_tree;
    CvMat* m_pca_descriptors_matrix;
    
    CvAffinePose* m_poses; // array of poses
    CvMat** m_transforms; // array of affine transformations corresponding to poses
    
    int m_pca_dim_high;
    int m_pca_dim_low;
    
    int m_pyr_levels;
    float scale_min;
    float scale_max;
    float scale_step;
    
    // SavePCAall: saves PCA components and descriptors to a file storage
    // - fs: output file storage
    void SavePCAall (FileStorage &fs) const;
    
    // LoadPCAall: loads PCA components and descriptors from a file node
    // - fn: input file node
    void LoadPCAall (const FileNode &fn);
};

class CV_EXPORTS OneWayDescriptorObject : public OneWayDescriptorBase
{
public:
    // creates an instance of OneWayDescriptorObject from a set of training files
    // - patch_size: size of the input (large) patch
    // - pose_count: the number of poses to generate for each descriptor
    // - train_path: path to training files
    // - pca_config: the name of the file that contains PCA for small patches (2 times smaller
    // than patch_size each dimension
    // - pca_hr_config: the name of the file that contains PCA for large patches (of patch_size size)
    // - pca_desc_config: the name of the file that contains descriptors of PCA components
    OneWayDescriptorObject(CvSize patch_size, int pose_count, const char* train_path, const char* pca_config,
                           const char* pca_hr_config = 0, const char* pca_desc_config = 0, int pyr_levels = 1);
    
    OneWayDescriptorObject(CvSize patch_size, int pose_count, const string &pca_filename,
                           const string &train_path = string (), const string &images_list = string (),
                           float _scale_min = 0.7f, float _scale_max=1.5f, float _scale_step=1.2f, int pyr_levels = 1);
    
    
    virtual ~OneWayDescriptorObject();
    
    // Allocate: allocates memory for a given number of features
    // - train_feature_count: the total number of features
    // - object_feature_count: the number of features extracted from the object
    void Allocate(int train_feature_count, int object_feature_count);
    
    
    void SetLabeledFeatures(const vector<KeyPoint>& features) {m_train_features = features;};
    vector<KeyPoint>& GetLabeledFeatures() {return m_train_features;};
    const vector<KeyPoint>& GetLabeledFeatures() const {return m_train_features;};
    vector<KeyPoint> _GetLabeledFeatures() const;
    
    // IsDescriptorObject: returns 1 if descriptor with specified index is positive, otherwise 0
    int IsDescriptorObject(int desc_idx) const;
    
    // MatchPointToPart: returns the part number of a feature if it matches one of the object parts, otherwise -1
    int MatchPointToPart(CvPoint pt) const;
    
    // GetDescriptorPart: returns the part number of the feature corresponding to a specified descriptor
    // - desc_idx: descriptor index
    int GetDescriptorPart(int desc_idx) const;
    
    
    void InitializeObjectDescriptors(IplImage* train_image, const vector<KeyPoint>& features,
                                     const char* feature_label, int desc_start_idx = 0, float scale = 1.0f,
                                     int is_background = 0);
    
    // GetObjectFeatureCount: returns the number of object features
    int GetObjectFeatureCount() const {return m_object_feature_count;};
    
protected:
    int* m_part_id; // contains part id for each of object descriptors
    vector<KeyPoint> m_train_features; // train features
    int m_object_feature_count; // the number of the positive features
    
};

    
/*
 *  OneWayDescriptorMatcher
 */
class OneWayDescriptorMatcher;
typedef OneWayDescriptorMatcher OneWayDescriptorMatch;

class CV_EXPORTS OneWayDescriptorMatcher : public GenericDescriptorMatcher
{
public:
    class CV_EXPORTS Params
    {
    public:
        static const int POSE_COUNT = 500;
        static const int PATCH_WIDTH = 24;
        static const int PATCH_HEIGHT = 24;
        static float GET_MIN_SCALE() { return 0.7f; }
        static float GET_MAX_SCALE() { return 1.5f; }
        static float GET_STEP_SCALE() { return 1.2f; }
        
        Params( int poseCount = POSE_COUNT,
               Size patchSize = Size(PATCH_WIDTH, PATCH_HEIGHT),
               string pcaFilename = string(),
               string trainPath = string(), string trainImagesList = string(),
               float minScale = GET_MIN_SCALE(), float maxScale = GET_MAX_SCALE(),
               float stepScale = GET_STEP_SCALE() );
        
        int poseCount;
        Size patchSize;
        string pcaFilename;
        string trainPath;
        string trainImagesList;
        
        float minScale, maxScale, stepScale;
    };
    
    OneWayDescriptorMatcher( const Params& params=Params() );
    virtual ~OneWayDescriptorMatcher();
    
    void initialize( const Params& params, const Ptr<OneWayDescriptorBase>& base=Ptr<OneWayDescriptorBase>() );
    
    // Clears keypoints storing in collection and OneWayDescriptorBase
    virtual void clear();
    
    virtual void train();
    
    virtual bool isMaskSupported();
    
    virtual void read( const FileNode &fn );
    virtual void write( FileStorage& fs ) const;
    
    virtual bool empty() const;
    
    virtual Ptr<GenericDescriptorMatcher> clone( bool emptyTrainData=false ) const;
    
protected:
    // Matches a set of keypoints from a single image of the training set. A rectangle with a center in a keypoint
    // and size (patch_width/2*scale, patch_height/2*scale) is cropped from the source image for each
    // keypoint. scale is iterated from DescriptorOneWayParams::min_scale to DescriptorOneWayParams::max_scale.
    // The minimum distance to each training patch with all its affine poses is found over all scales.
    // The class ID of a match is returned for each keypoint. The distance is calculated over PCA components
    // loaded with DescriptorOneWay::Initialize, kd tree is used for finding minimum distances.
    virtual void knnMatchImpl( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                              vector<vector<DMatch> >& matches, int k,
                              const vector<Mat>& masks, bool compactResult );
    virtual void radiusMatchImpl( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                 vector<vector<DMatch> >& matches, float maxDistance,
                                 const vector<Mat>& masks, bool compactResult );
    
    Ptr<OneWayDescriptorBase> base;
    Params params;
    int prevTrainCount;
};

/*
 *  FernDescriptorMatcher
 */
class FernDescriptorMatcher;
typedef FernDescriptorMatcher FernDescriptorMatch;

class CV_EXPORTS FernDescriptorMatcher : public GenericDescriptorMatcher
{
public:
    class CV_EXPORTS Params
    {
    public:
        Params( int nclasses=0,
               int patchSize=FernClassifier::PATCH_SIZE,
               int signatureSize=FernClassifier::DEFAULT_SIGNATURE_SIZE,
               int nstructs=FernClassifier::DEFAULT_STRUCTS,
               int structSize=FernClassifier::DEFAULT_STRUCT_SIZE,
               int nviews=FernClassifier::DEFAULT_VIEWS,
               int compressionMethod=FernClassifier::COMPRESSION_NONE,
               const PatchGenerator& patchGenerator=PatchGenerator() );
        
        Params( const string& filename );
        
        int nclasses;
        int patchSize;
        int signatureSize;
        int nstructs;
        int structSize;
        int nviews;
        int compressionMethod;
        PatchGenerator patchGenerator;
        
        string filename;
    };
    
    FernDescriptorMatcher( const Params& params=Params() );
    virtual ~FernDescriptorMatcher();
    
    virtual void clear();
    
    virtual void train();
    
    virtual bool isMaskSupported();
    
    virtual void read( const FileNode &fn );
    virtual void write( FileStorage& fs ) const;
    virtual bool empty() const;
    
    virtual Ptr<GenericDescriptorMatcher> clone( bool emptyTrainData=false ) const;
    
protected:
    virtual void knnMatchImpl( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                              vector<vector<DMatch> >& matches, int k,
                              const vector<Mat>& masks, bool compactResult );
    virtual void radiusMatchImpl( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
                                 vector<vector<DMatch> >& matches, float maxDistance,
                                 const vector<Mat>& masks, bool compactResult );
    
    void trainFernClassifier();
    void calcBestProbAndMatchIdx( const Mat& image, const Point2f& pt,
                                 float& bestProb, int& bestMatchIdx, vector<float>& signature );
    Ptr<FernClassifier> classifier;
    Params params;
    int prevTrainCount;
};
    

/*
 * CalonderDescriptorExtractor
 */
template<typename T>
class CV_EXPORTS CalonderDescriptorExtractor : public DescriptorExtractor
{
public:
    CalonderDescriptorExtractor( const string& classifierFile );
    
    virtual void read( const FileNode &fn );
    virtual void write( FileStorage &fs ) const;
    
    virtual int descriptorSize() const { return classifier_.classes(); }
    virtual int descriptorType() const { return DataType<T>::type; }
    
    virtual bool empty() const;
    
protected:
    virtual void computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;
    
    RTreeClassifier classifier_;
    static const int BORDER_SIZE = 16;
};

template<typename T>
CalonderDescriptorExtractor<T>::CalonderDescriptorExtractor(const std::string& classifier_file)
{
    classifier_.read( classifier_file.c_str() );
}

template<typename T>
void CalonderDescriptorExtractor<T>::computeImpl( const Mat& image,
                                                 vector<KeyPoint>& keypoints,
                                                 Mat& descriptors) const
{
    // Cannot compute descriptors for keypoints on the image border.
    KeyPointsFilter::runByImageBorder(keypoints, image.size(), BORDER_SIZE);
    
    /// @todo Check 16-byte aligned
    descriptors.create(keypoints.size(), classifier_.classes(), cv::DataType<T>::type);
    
    int patchSize = RandomizedTree::PATCH_SIZE;
    int offset = patchSize / 2;
    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        cv::Point2f pt = keypoints[i].pt;
        IplImage ipl = image( Rect((int)(pt.x - offset), (int)(pt.y - offset), patchSize, patchSize) );
        classifier_.getSignature( &ipl, descriptors.ptr<T>(i));
    }
}

template<typename T>
void CalonderDescriptorExtractor<T>::read( const FileNode& )
{}

template<typename T>
void CalonderDescriptorExtractor<T>::write( FileStorage& ) const
{}

template<typename T>
bool CalonderDescriptorExtractor<T>::empty() const
{
    return classifier_.trees_.empty();
}
    
/****************************************************************************************\
*                                Planar Object Detection                                 *
\****************************************************************************************/

class CV_EXPORTS PlanarObjectDetector
{
public:
    PlanarObjectDetector();
    PlanarObjectDetector(const FileNode& node);
    PlanarObjectDetector(const vector<Mat>& pyr, int _npoints=300,
                         int _patchSize=FernClassifier::PATCH_SIZE,
                         int _nstructs=FernClassifier::DEFAULT_STRUCTS,
                         int _structSize=FernClassifier::DEFAULT_STRUCT_SIZE,
                         int _nviews=FernClassifier::DEFAULT_VIEWS,
                         const LDetector& detector=LDetector(),
                         const PatchGenerator& patchGenerator=PatchGenerator());
    virtual ~PlanarObjectDetector();
    virtual void train(const vector<Mat>& pyr, int _npoints=300,
                       int _patchSize=FernClassifier::PATCH_SIZE,
                       int _nstructs=FernClassifier::DEFAULT_STRUCTS,
                       int _structSize=FernClassifier::DEFAULT_STRUCT_SIZE,
                       int _nviews=FernClassifier::DEFAULT_VIEWS,
                       const LDetector& detector=LDetector(),
                       const PatchGenerator& patchGenerator=PatchGenerator());
    virtual void train(const vector<Mat>& pyr, const vector<KeyPoint>& keypoints,
                       int _patchSize=FernClassifier::PATCH_SIZE,
                       int _nstructs=FernClassifier::DEFAULT_STRUCTS,
                       int _structSize=FernClassifier::DEFAULT_STRUCT_SIZE,
                       int _nviews=FernClassifier::DEFAULT_VIEWS,
                       const LDetector& detector=LDetector(),
                       const PatchGenerator& patchGenerator=PatchGenerator());
    Rect getModelROI() const;
    vector<KeyPoint> getModelPoints() const;
    const LDetector& getDetector() const;
    const FernClassifier& getClassifier() const;
    void setVerbose(bool verbose);
    
    void read(const FileNode& node);
    void write(FileStorage& fs, const String& name=String()) const;
    bool operator()(const Mat& image, CV_OUT Mat& H, CV_OUT vector<Point2f>& corners) const;
    bool operator()(const vector<Mat>& pyr, const vector<KeyPoint>& keypoints,
                    CV_OUT Mat& H, CV_OUT vector<Point2f>& corners,
                    CV_OUT vector<int>* pairs=0) const;
    
protected:
    bool verbose;
    Rect modelROI;
    vector<KeyPoint> modelPoints;
    LDetector ldetector;
    FernClassifier fernClassifier;
};
    
}


//#include "cvvidsurv.hpp"
#endif

#endif

/* End of file. */
