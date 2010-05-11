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

#ifndef _CV_VM_H_
#define _CV_VM_H_

/*----------------------- Internal ViewMorphing Functions ------------------------------*/

/*======================================================================================*/

typedef struct CvMatrix4
{
    float m[4][4];
}
CvMatrix4;


/* Scanline section. Find coordinates by fundamental matrix */

/* Epsilon and real zero */
#define EPSILON             1.e-4
//#define REAL_ZERO(x)        ( (x) < EPSILON && (x) > -EPSILON)
#define REAL_ZERO(x) ( (x) < 1e-8 && (x) > -1e-8)

#define SIGN(x)				( (x)<0 ? -1:((x)>0?1:0 ) )

CvStatus  icvMakeScanlinesLengths( int*        scanlines,
                                    int         numlines,
                                    int*        lens);

/*=============================== PreWarp section ======================================*/

CV_INLINE int icvGetColor(uchar* valueRGB);

CvStatus  icvFindRunsInOneImage(
                                int     numLines,       /* number of scanlines		*/
                                uchar*  prewarp,        /* prewarp image 			*/
                                int*    line_lens,      /* line lengths in pixels	*/
                                int*    runs,           /* result runs				*/
                                int*    num_runs);

/*================================ Morphing section ====================================*/

CvStatus  icvMorphEpilines8uC3(    uchar*  first_pix, /* raster epiline from the first image       */
                                    uchar*  second_pix, /* raster epiline from the second image      */
                                    uchar*  dst_pix,    /* raster epiline from the destination image */
                                                        /* (it's an output parameter)                */
                                    float   alpha,      /* relative position of camera               */
                                    int*    first,      /* first sequence of runs                    */
                                    int     first_runs, /* it's length                               */
                                    int*    second,     /* second sequence of runs                   */
                                    int     second_runs,
                                    int*    first_corr, /* correspond information for the 1st seq    */
                                    int*    second_corr,
                                    int     dst_len);   /* correspond information for the 2nd seq    */

/*========================== Dynamic correspond section ================================*/

CvStatus  icvDynamicCorrespond(   int*  first,         /* first sequence of runs           */
                                                         /* s0|w0|s1|w1|...|s(n-1)|w(n-1)|sn */
                                    int   first_runs,    /* number of runs                   */
                                    int*  second,        /* second sequence of runs          */
                                    int   second_runs,
                                    int*  first_corr,    /* s0'|e0'|s1'|e1'|...              */
                                    int*  second_corr );

/*============================= PostWarp Functions =====================================*/

CvStatus  icvFetchLine8uC3R(
                                uchar*   src,  int   src_step,
                                uchar*   dst,  int*  dst_num,
                                CvSize  src_size,
                                CvPoint start,
                                CvPoint end );

CvStatus  icvDrawLine8uC3R(
                                uchar*   src,  int  src_num,
                                uchar*   dst,  int  dst_step,
                                CvSize  dst_size,
                                CvPoint start,
                                CvPoint end );


/*============================== Fundamental Matrix Functions ==========================*/
CvStatus  icvPoint7(  int*        points1,
                        int*        points2,
                        double*     F,
                        int*        amount
                        );

CvStatus  icvCubic(      double a2, double a1,
                            double a0, double* squares );

double icvDet( double* M );
double   icvMinor( double* M, int x, int y );

int
icvGaussMxN( double *A, double *B, int M, int N, double **solutions );

CvStatus
icvGetCoef( double *f1, double *f2, double *a2, double *a1, double *a0 );

/*================================= Scanlines Functions ================================*/

CvStatus  icvGetCoefficient(  CvMatrix3*     matrix,
                                CvSize         imgSize,
                                int*            scanlines_1,
                                int*            scanlines_2,
                                int*            numlines);

CvStatus  icvGetCoefficientDefault(   CvMatrix3*     matrix,
                                        CvSize         imgSize,
                                        int*            scanlines_1,
                                        int*            scanlines_2,
                                        int*            numlines);

CvStatus  icvGetCoefficientStereo(    CvMatrix3*     matrix,
                                        CvSize         imgSize,
                                        float*          l_epipole,
                                        float*          r_epipole,
                                        int*            scanlines_1,
                                        int*            scanlines_2,
                                        int*            numlines
                                    );

CvStatus  icvGetCoefficientOrto(  CvMatrix3*     matrix,
                                    CvSize         imgSize,
                                    int*            scanlines_1,
                                    int*            scanlines_2,
                                    int*            numlines);


CvStatus  icvGetCrossEpilineFrame(    CvSize     imgSize,
                                        float*      epiline,
                                        int*        x1,
                                        int*        y1,
                                        int*        x2,
                                        int*        y2
                                    );

CvStatus  icvBuildScanlineLeftStereo( 
                                        CvSize         imgSize,
                                        CvMatrix3*     matrix,
                                        float*          l_epipole,
                                        float*          l_angle,
                                        float           l_radius,
                                        int*            scanlines_1,
                                        int*            scanlines_2,
                                        int*            numlines);

CvStatus  icvBuildScanlineRightStereo(
                                        CvSize         imgSize,
                                        CvMatrix3*     matrix,
                                        float*          r_epipole,
                                        float*          r_angle,
                                        float           r_radius,
                                        int*            scanlines_1,
                                        int*            scanlines_2,
                                        int*            numlines);

CvStatus  icvGetStartEnd1(
                                    CvMatrix3*     matrix,
                                    CvSize         imgSize,
                                    float*          l_start_end,
                                    float*          r_start_end );

CvStatus  icvGetStartEnd2(
                                    CvMatrix3*     matrix,
                                    CvSize         imgSize,
                                    float*          l_start_end,
                                    float*          r_start_end );

CvStatus  icvGetStartEnd3(
                                    CvMatrix3*     matrix,
                                    CvSize         imgSize,
                                    float*          l_start_end,
                                    float*          r_start_end );

CvStatus  icvGetStartEnd4(
                                    CvMatrix3*     matrix,
                                    CvSize         imgSize,
                                    float*          l_start_end,
                                    float*          r_start_end );

CvStatus  icvBuildScanlineLeft(
                                    CvMatrix3*     matrix,
                                    CvSize         imgSize,
                                    int*            scanlines_1,
                                    int*            scanlines_2,
                                    float*          l_start_end,
                                    int*            numlines
                                    );

CvStatus  icvBuildScanlineRight(
                                    CvMatrix3*     matrix,
                                    CvSize         imgSize,
                                    int*            scanlines_1,
                                    int*            scanlines_2,
                                    float*          r_start_end,
                                    int*            numlines
                                    );


/*=================================== LMedS Functions ==================================*/
CvStatus  icvLMedS7(
                        int*            points1,
                        int*            points2,
                        CvMatrix3*     matrix);


CvStatus  icvLMedS(   int*    points1,
                        int*    points2,
                        int     numPoints,
                        CvMatrix3* fundamentalMatrix );


/*
CvStatus  icvFindFundamentalMatrix(
                                    int*            points1,
                                    int*            points2,
                                    int             numpoints,
                                    int             method,
                                    CvMatrix3*      matrix);
*/
void   icvChoose7(	int*    ml,     int* mr,
					    int     num,	int* ml7,
					    int*   mr7 );

double icvMedian(	int* ml, int* mr,
				    int num, double* F );

int icvBoltingPoints( int* ml,	    int* mr,
					    int num,        double* F,
					    double Mj,      int* *new_ml,
					    int* *new_mr,   int* new_num);

CvStatus  icvPoints8( int* ml, int* mr,
                        int num, double* F );

CvStatus  icvRank2Constraint( double* F );

CvStatus  icvSort( double* array, int length );

double icvAnalyticPoints8(	double* A,
											int num, double* F );

int icvSingularValueDecomposition(	int		M,
										int		N,
										double*	A,
										double*	W,
										int		get_U,
										double*	U,
										int		get_V,
										double*	V
												 );


/*======================================================================================*/
#endif/*_CV_VM_H_*/

