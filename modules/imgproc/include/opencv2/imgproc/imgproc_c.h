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

#ifndef __OPENCV_IMGPROC_IMGPROC_C_H__
#define __OPENCV_IMGPROC_IMGPROC_C_H__

#include "opencv2/imgproc/types_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/*********************** Background statistics accumulation *****************************/

/* Adds image to accumulator */
CVAPI(void)  cvAcc( const CvArr* image, CvArr* sum,
                   const CvArr* mask CV_DEFAULT(NULL) );

/* Adds squared image to accumulator */
CVAPI(void)  cvSquareAcc( const CvArr* image, CvArr* sqsum,
                         const CvArr* mask CV_DEFAULT(NULL) );

/* Adds a product of two images to accumulator */
CVAPI(void)  cvMultiplyAcc( const CvArr* image1, const CvArr* image2, CvArr* acc,
                           const CvArr* mask CV_DEFAULT(NULL) );

/* Adds image to accumulator with weights: acc = acc*(1-alpha) + image*alpha */
CVAPI(void)  cvRunningAvg( const CvArr* image, CvArr* acc, double alpha,
                          const CvArr* mask CV_DEFAULT(NULL) );

/****************************************************************************************\
*                                    Image Processing                                    *
\****************************************************************************************/

/* Copies source 2D array inside of the larger destination array and
   makes a border of the specified type (IPL_BORDER_*) around the copied area. */
CVAPI(void) cvCopyMakeBorder( const CvArr* src, CvArr* dst, CvPoint offset,
                              int bordertype, CvScalar value CV_DEFAULT(cvScalarAll(0)));

/* Smoothes array (removes noise) */
CVAPI(void) cvSmooth( const CvArr* src, CvArr* dst,
                      int smoothtype CV_DEFAULT(CV_GAUSSIAN),
                      int size1 CV_DEFAULT(3),
                      int size2 CV_DEFAULT(0),
                      double sigma1 CV_DEFAULT(0),
                      double sigma2 CV_DEFAULT(0));

/* Convolves the image with the kernel */
CVAPI(void) cvFilter2D( const CvArr* src, CvArr* dst, const CvMat* kernel,
                        CvPoint anchor CV_DEFAULT(cvPoint(-1,-1)));

/* Finds integral image: SUM(X,Y) = sum(x<X,y<Y)I(x,y) */
CVAPI(void) cvIntegral( const CvArr* image, CvArr* sum,
                       CvArr* sqsum CV_DEFAULT(NULL),
                       CvArr* tilted_sum CV_DEFAULT(NULL));

/*
   Smoothes the input image with gaussian kernel and then down-samples it.
   dst_width = floor(src_width/2)[+1],
   dst_height = floor(src_height/2)[+1]
*/
CVAPI(void)  cvPyrDown( const CvArr* src, CvArr* dst,
                        int filter CV_DEFAULT(CV_GAUSSIAN_5x5) );

/*
   Up-samples image and smoothes the result with gaussian kernel.
   dst_width = src_width*2,
   dst_height = src_height*2
*/
CVAPI(void)  cvPyrUp( const CvArr* src, CvArr* dst,
                      int filter CV_DEFAULT(CV_GAUSSIAN_5x5) );

/* Builds pyramid for an image */
CVAPI(CvMat**) cvCreatePyramid( const CvArr* img, int extra_layers, double rate,
                                const CvSize* layer_sizes CV_DEFAULT(0),
                                CvArr* bufarr CV_DEFAULT(0),
                                int calc CV_DEFAULT(1),
                                int filter CV_DEFAULT(CV_GAUSSIAN_5x5) );

/* Releases pyramid */
CVAPI(void)  cvReleasePyramid( CvMat*** pyramid, int extra_layers );


/* Filters image using meanshift algorithm */
CVAPI(void) cvPyrMeanShiftFiltering( const CvArr* src, CvArr* dst,
    double sp, double sr, int max_level CV_DEFAULT(1),
    CvTermCriteria termcrit CV_DEFAULT(cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,5,1)));

/* Segments image using seed "markers" */
CVAPI(void) cvWatershed( const CvArr* image, CvArr* markers );

/* Calculates an image derivative using generalized Sobel
   (aperture_size = 1,3,5,7) or Scharr (aperture_size = -1) operator.
   Scharr can be used only for the first dx or dy derivative */
CVAPI(void) cvSobel( const CvArr* src, CvArr* dst,
                    int xorder, int yorder,
                    int aperture_size CV_DEFAULT(3));

/* Calculates the image Laplacian: (d2/dx + d2/dy)I */
CVAPI(void) cvLaplace( const CvArr* src, CvArr* dst,
                      int aperture_size CV_DEFAULT(3) );

/* Converts input array pixels from one color space to another */
CVAPI(void)  cvCvtColor( const CvArr* src, CvArr* dst, int code );


/* Resizes image (input array is resized to fit the destination array) */
CVAPI(void)  cvResize( const CvArr* src, CvArr* dst,
                       int interpolation CV_DEFAULT( CV_INTER_LINEAR ));

/* Warps image with affine transform */
CVAPI(void)  cvWarpAffine( const CvArr* src, CvArr* dst, const CvMat* map_matrix,
                           int flags CV_DEFAULT(CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS),
                           CvScalar fillval CV_DEFAULT(cvScalarAll(0)) );

/* Computes affine transform matrix for mapping src[i] to dst[i] (i=0,1,2) */
CVAPI(CvMat*) cvGetAffineTransform( const CvPoint2D32f * src,
                                    const CvPoint2D32f * dst,
                                    CvMat * map_matrix );

/* Computes rotation_matrix matrix */
CVAPI(CvMat*)  cv2DRotationMatrix( CvPoint2D32f center, double angle,
                                   double scale, CvMat* map_matrix );

/* Warps image with perspective (projective) transform */
CVAPI(void)  cvWarpPerspective( const CvArr* src, CvArr* dst, const CvMat* map_matrix,
                                int flags CV_DEFAULT(CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS),
                                CvScalar fillval CV_DEFAULT(cvScalarAll(0)) );

/* Computes perspective transform matrix for mapping src[i] to dst[i] (i=0,1,2,3) */
CVAPI(CvMat*) cvGetPerspectiveTransform( const CvPoint2D32f* src,
                                         const CvPoint2D32f* dst,
                                         CvMat* map_matrix );

/* Performs generic geometric transformation using the specified coordinate maps */
CVAPI(void)  cvRemap( const CvArr* src, CvArr* dst,
                      const CvArr* mapx, const CvArr* mapy,
                      int flags CV_DEFAULT(CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS),
                      CvScalar fillval CV_DEFAULT(cvScalarAll(0)) );

/* Converts mapx & mapy from floating-point to integer formats for cvRemap */
CVAPI(void)  cvConvertMaps( const CvArr* mapx, const CvArr* mapy,
                            CvArr* mapxy, CvArr* mapalpha );

/* Performs forward or inverse log-polar image transform */
CVAPI(void)  cvLogPolar( const CvArr* src, CvArr* dst,
                         CvPoint2D32f center, double M,
                         int flags CV_DEFAULT(CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS));

/* Performs forward or inverse linear-polar image transform */
CVAPI(void)  cvLinearPolar( const CvArr* src, CvArr* dst,
                         CvPoint2D32f center, double maxRadius,
                         int flags CV_DEFAULT(CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS));

/* Transforms the input image to compensate lens distortion */
CVAPI(void) cvUndistort2( const CvArr* src, CvArr* dst,
                          const CvMat* camera_matrix,
                          const CvMat* distortion_coeffs,
                          const CvMat* new_camera_matrix CV_DEFAULT(0) );

/* Computes transformation map from intrinsic camera parameters
   that can used by cvRemap */
CVAPI(void) cvInitUndistortMap( const CvMat* camera_matrix,
                                const CvMat* distortion_coeffs,
                                CvArr* mapx, CvArr* mapy );

/* Computes undistortion+rectification map for a head of stereo camera */
CVAPI(void) cvInitUndistortRectifyMap( const CvMat* camera_matrix,
                                       const CvMat* dist_coeffs,
                                       const CvMat *R, const CvMat* new_camera_matrix,
                                       CvArr* mapx, CvArr* mapy );

/* Computes the original (undistorted) feature coordinates
   from the observed (distorted) coordinates */
CVAPI(void) cvUndistortPoints( const CvMat* src, CvMat* dst,
                               const CvMat* camera_matrix,
                               const CvMat* dist_coeffs,
                               const CvMat* R CV_DEFAULT(0),
                               const CvMat* P CV_DEFAULT(0));

/* creates structuring element used for morphological operations */
CVAPI(IplConvKernel*)  cvCreateStructuringElementEx(
            int cols, int  rows, int  anchor_x, int  anchor_y,
            int shape, int* values CV_DEFAULT(NULL) );

/* releases structuring element */
CVAPI(void)  cvReleaseStructuringElement( IplConvKernel** element );

/* erodes input image (applies minimum filter) one or more times.
   If element pointer is NULL, 3x3 rectangular element is used */
CVAPI(void)  cvErode( const CvArr* src, CvArr* dst,
                      IplConvKernel* element CV_DEFAULT(NULL),
                      int iterations CV_DEFAULT(1) );

/* dilates input image (applies maximum filter) one or more times.
   If element pointer is NULL, 3x3 rectangular element is used */
CVAPI(void)  cvDilate( const CvArr* src, CvArr* dst,
                       IplConvKernel* element CV_DEFAULT(NULL),
                       int iterations CV_DEFAULT(1) );

/* Performs complex morphological transformation */
CVAPI(void)  cvMorphologyEx( const CvArr* src, CvArr* dst,
                             CvArr* temp, IplConvKernel* element,
                             int operation, int iterations CV_DEFAULT(1) );

/* Calculates all spatial and central moments up to the 3rd order */
CVAPI(void) cvMoments( const CvArr* arr, CvMoments* moments, int binary CV_DEFAULT(0));

/* Retrieve particular spatial, central or normalized central moments */
CVAPI(double)  cvGetSpatialMoment( CvMoments* moments, int x_order, int y_order );
CVAPI(double)  cvGetCentralMoment( CvMoments* moments, int x_order, int y_order );
CVAPI(double)  cvGetNormalizedCentralMoment( CvMoments* moments,
                                             int x_order, int y_order );

/* Calculates 7 Hu's invariants from precalculated spatial and central moments */
CVAPI(void) cvGetHuMoments( CvMoments*  moments, CvHuMoments*  hu_moments );

/*********************************** data sampling **************************************/

/* Fetches pixels that belong to the specified line segment and stores them to the buffer.
   Returns the number of retrieved points. */
CVAPI(int)  cvSampleLine( const CvArr* image, CvPoint pt1, CvPoint pt2, void* buffer,
                          int connectivity CV_DEFAULT(8));

/* Retrieves the rectangular image region with specified center from the input array.
 dst(x,y) <- src(x + center.x - dst_width/2, y + center.y - dst_height/2).
 Values of pixels with fractional coordinates are retrieved using bilinear interpolation*/
CVAPI(void)  cvGetRectSubPix( const CvArr* src, CvArr* dst, CvPoint2D32f center );


/* Retrieves quadrangle from the input array.
    matrixarr = ( a11  a12 | b1 )   dst(x,y) <- src(A[x y]' + b)
                ( a21  a22 | b2 )   (bilinear interpolation is used to retrieve pixels
                                     with fractional coordinates)
*/
CVAPI(void)  cvGetQuadrangleSubPix( const CvArr* src, CvArr* dst,
                                    const CvMat* map_matrix );

/* Measures similarity between template and overlapped windows in the source image
   and fills the resultant image with the measurements */
CVAPI(void)  cvMatchTemplate( const CvArr* image, const CvArr* templ,
                              CvArr* result, int method );

/* Computes earth mover distance between
   two weighted point sets (called signatures) */
CVAPI(float)  cvCalcEMD2( const CvArr* signature1,
                          const CvArr* signature2,
                          int distance_type,
                          CvDistanceFunction distance_func CV_DEFAULT(NULL),
                          const CvArr* cost_matrix CV_DEFAULT(NULL),
                          CvArr* flow CV_DEFAULT(NULL),
                          float* lower_bound CV_DEFAULT(NULL),
                          void* userdata CV_DEFAULT(NULL));

/****************************************************************************************\
*                              Contours retrieving                                       *
\****************************************************************************************/

/* Retrieves outer and optionally inner boundaries of white (non-zero) connected
   components in the black (zero) background */
CVAPI(int)  cvFindContours( CvArr* image, CvMemStorage* storage, CvSeq** first_contour,
                            int header_size CV_DEFAULT(sizeof(CvContour)),
                            int mode CV_DEFAULT(CV_RETR_LIST),
                            int method CV_DEFAULT(CV_CHAIN_APPROX_SIMPLE),
                            CvPoint offset CV_DEFAULT(cvPoint(0,0)));

/* Initializes contour retrieving process.
   Calls cvStartFindContours.
   Calls cvFindNextContour until null pointer is returned
   or some other condition becomes true.
   Calls cvEndFindContours at the end. */
CVAPI(CvContourScanner)  cvStartFindContours( CvArr* image, CvMemStorage* storage,
                            int header_size CV_DEFAULT(sizeof(CvContour)),
                            int mode CV_DEFAULT(CV_RETR_LIST),
                            int method CV_DEFAULT(CV_CHAIN_APPROX_SIMPLE),
                            CvPoint offset CV_DEFAULT(cvPoint(0,0)));

/* Retrieves next contour */
CVAPI(CvSeq*)  cvFindNextContour( CvContourScanner scanner );


/* Substitutes the last retrieved contour with the new one
   (if the substitutor is null, the last retrieved contour is removed from the tree) */
CVAPI(void)   cvSubstituteContour( CvContourScanner scanner, CvSeq* new_contour );


/* Releases contour scanner and returns pointer to the first outer contour */
CVAPI(CvSeq*)  cvEndFindContours( CvContourScanner* scanner );

/* Approximates a single Freeman chain or a tree of chains to polygonal curves */
CVAPI(CvSeq*) cvApproxChains( CvSeq* src_seq, CvMemStorage* storage,
                            int method CV_DEFAULT(CV_CHAIN_APPROX_SIMPLE),
                            double parameter CV_DEFAULT(0),
                            int  minimal_perimeter CV_DEFAULT(0),
                            int  recursive CV_DEFAULT(0));

/* Initializes Freeman chain reader.
   The reader is used to iteratively get coordinates of all the chain points.
   If the Freeman codes should be read as is, a simple sequence reader should be used */
CVAPI(void) cvStartReadChainPoints( CvChain* chain, CvChainPtReader* reader );

/* Retrieves the next chain point */
CVAPI(CvPoint) cvReadChainPoint( CvChainPtReader* reader );


/****************************************************************************************\
*                            Contour Processing and Shape Analysis                       *
\****************************************************************************************/

/* Approximates a single polygonal curve (contour) or
   a tree of polygonal curves (contours) */
CVAPI(CvSeq*)  cvApproxPoly( const void* src_seq,
                             int header_size, CvMemStorage* storage,
                             int method, double eps,
                             int recursive CV_DEFAULT(0));

/* Calculates perimeter of a contour or length of a part of contour */
CVAPI(double)  cvArcLength( const void* curve,
                            CvSlice slice CV_DEFAULT(CV_WHOLE_SEQ),
                            int is_closed CV_DEFAULT(-1));

CV_INLINE double cvContourPerimeter( const void* contour )
{
    return cvArcLength( contour, CV_WHOLE_SEQ, 1 );
}


/* Calculates contour bounding rectangle (update=1) or
   just retrieves pre-calculated rectangle (update=0) */
CVAPI(CvRect)  cvBoundingRect( CvArr* points, int update CV_DEFAULT(0) );

/* Calculates area of a contour or contour segment */
CVAPI(double)  cvContourArea( const CvArr* contour,
                              CvSlice slice CV_DEFAULT(CV_WHOLE_SEQ),
                              int oriented CV_DEFAULT(0));

/* Finds minimum area rotated rectangle bounding a set of points */
CVAPI(CvBox2D)  cvMinAreaRect2( const CvArr* points,
                                CvMemStorage* storage CV_DEFAULT(NULL));

/* Finds minimum enclosing circle for a set of points */
CVAPI(int)  cvMinEnclosingCircle( const CvArr* points,
                                  CvPoint2D32f* center, float* radius );

/* Compares two contours by matching their moments */
CVAPI(double)  cvMatchShapes( const void* object1, const void* object2,
                              int method, double parameter CV_DEFAULT(0));

/* Calculates exact convex hull of 2d point set */
CVAPI(CvSeq*) cvConvexHull2( const CvArr* input,
                             void* hull_storage CV_DEFAULT(NULL),
                             int orientation CV_DEFAULT(CV_CLOCKWISE),
                             int return_points CV_DEFAULT(0));

/* Checks whether the contour is convex or not (returns 1 if convex, 0 if not) */
CVAPI(int)  cvCheckContourConvexity( const CvArr* contour );


/* Finds convexity defects for the contour */
CVAPI(CvSeq*)  cvConvexityDefects( const CvArr* contour, const CvArr* convexhull,
                                   CvMemStorage* storage CV_DEFAULT(NULL));

/* Fits ellipse into a set of 2d points */
CVAPI(CvBox2D) cvFitEllipse2( const CvArr* points );

/* Finds minimum rectangle containing two given rectangles */
CVAPI(CvRect)  cvMaxRect( const CvRect* rect1, const CvRect* rect2 );

/* Finds coordinates of the box vertices */
CVAPI(void) cvBoxPoints( CvBox2D box, CvPoint2D32f pt[4] );

/* Initializes sequence header for a matrix (column or row vector) of points -
   a wrapper for cvMakeSeqHeaderForArray (it does not initialize bounding rectangle!!!) */
CVAPI(CvSeq*) cvPointSeqFromMat( int seq_kind, const CvArr* mat,
                                 CvContour* contour_header,
                                 CvSeqBlock* block );

/* Checks whether the point is inside polygon, outside, on an edge (at a vertex).
   Returns positive, negative or zero value, correspondingly.
   Optionally, measures a signed distance between
   the point and the nearest polygon edge (measure_dist=1) */
CVAPI(double) cvPointPolygonTest( const CvArr* contour,
                                  CvPoint2D32f pt, int measure_dist );

/****************************************************************************************\
*                                  Histogram functions                                   *
\****************************************************************************************/

/* Creates new histogram */
CVAPI(CvHistogram*)  cvCreateHist( int dims, int* sizes, int type,
                                   float** ranges CV_DEFAULT(NULL),
                                   int uniform CV_DEFAULT(1));

/* Assignes histogram bin ranges */
CVAPI(void)  cvSetHistBinRanges( CvHistogram* hist, float** ranges,
                                int uniform CV_DEFAULT(1));

/* Creates histogram header for array */
CVAPI(CvHistogram*)  cvMakeHistHeaderForArray(
                            int  dims, int* sizes, CvHistogram* hist,
                            float* data, float** ranges CV_DEFAULT(NULL),
                            int uniform CV_DEFAULT(1));

/* Releases histogram */
CVAPI(void)  cvReleaseHist( CvHistogram** hist );

/* Clears all the histogram bins */
CVAPI(void)  cvClearHist( CvHistogram* hist );

/* Finds indices and values of minimum and maximum histogram bins */
CVAPI(void)  cvGetMinMaxHistValue( const CvHistogram* hist,
                                   float* min_value, float* max_value,
                                   int* min_idx CV_DEFAULT(NULL),
                                   int* max_idx CV_DEFAULT(NULL));


/* Normalizes histogram by dividing all bins by sum of the bins, multiplied by <factor>.
   After that sum of histogram bins is equal to <factor> */
CVAPI(void)  cvNormalizeHist( CvHistogram* hist, double factor );


/* Clear all histogram bins that are below the threshold */
CVAPI(void)  cvThreshHist( CvHistogram* hist, double threshold );


/* Compares two histogram */
CVAPI(double)  cvCompareHist( const CvHistogram* hist1,
                              const CvHistogram* hist2,
                              int method);

/* Copies one histogram to another. Destination histogram is created if
   the destination pointer is NULL */
CVAPI(void)  cvCopyHist( const CvHistogram* src, CvHistogram** dst );


/* Calculates bayesian probabilistic histograms
   (each or src and dst is an array of <number> histograms */
CVAPI(void)  cvCalcBayesianProb( CvHistogram** src, int number,
                                CvHistogram** dst);

/* Calculates array histogram */
CVAPI(void)  cvCalcArrHist( CvArr** arr, CvHistogram* hist,
                            int accumulate CV_DEFAULT(0),
                            const CvArr* mask CV_DEFAULT(NULL) );

CV_INLINE  void  cvCalcHist( IplImage** image, CvHistogram* hist,
                             int accumulate CV_DEFAULT(0),
                             const CvArr* mask CV_DEFAULT(NULL) )
{
    cvCalcArrHist( (CvArr**)image, hist, accumulate, mask );
}

/* Calculates back project */
CVAPI(void)  cvCalcArrBackProject( CvArr** image, CvArr* dst,
                                   const CvHistogram* hist );
#define  cvCalcBackProject(image, dst, hist) cvCalcArrBackProject((CvArr**)image, dst, hist)


/* Does some sort of template matching but compares histograms of
   template and each window location */
CVAPI(void)  cvCalcArrBackProjectPatch( CvArr** image, CvArr* dst, CvSize range,
                                        CvHistogram* hist, int method,
                                        double factor );
#define  cvCalcBackProjectPatch( image, dst, range, hist, method, factor ) \
     cvCalcArrBackProjectPatch( (CvArr**)image, dst, range, hist, method, factor )


/* calculates probabilistic density (divides one histogram by another) */
CVAPI(void)  cvCalcProbDensity( const CvHistogram* hist1, const CvHistogram* hist2,
                                CvHistogram* dst_hist, double scale CV_DEFAULT(255) );

/* equalizes histogram of 8-bit single-channel image */
CVAPI(void)  cvEqualizeHist( const CvArr* src, CvArr* dst );


/* Applies distance transform to binary image */
CVAPI(void)  cvDistTransform( const CvArr* src, CvArr* dst,
                              int distance_type CV_DEFAULT(CV_DIST_L2),
                              int mask_size CV_DEFAULT(3),
                              const float* mask CV_DEFAULT(NULL),
                              CvArr* labels CV_DEFAULT(NULL),
                              int labelType CV_DEFAULT(CV_DIST_LABEL_CCOMP));


/* Applies fixed-level threshold to grayscale image.
   This is a basic operation applied before retrieving contours */
CVAPI(double)  cvThreshold( const CvArr*  src, CvArr*  dst,
                            double  threshold, double  max_value,
                            int threshold_type );

/* Applies adaptive threshold to grayscale image.
   The two parameters for methods CV_ADAPTIVE_THRESH_MEAN_C and
   CV_ADAPTIVE_THRESH_GAUSSIAN_C are:
   neighborhood size (3, 5, 7 etc.),
   and a constant subtracted from mean (...,-3,-2,-1,0,1,2,3,...) */
CVAPI(void)  cvAdaptiveThreshold( const CvArr* src, CvArr* dst, double max_value,
                                  int adaptive_method CV_DEFAULT(CV_ADAPTIVE_THRESH_MEAN_C),
                                  int threshold_type CV_DEFAULT(CV_THRESH_BINARY),
                                  int block_size CV_DEFAULT(3),
                                  double param1 CV_DEFAULT(5));

/* Fills the connected component until the color difference gets large enough */
CVAPI(void)  cvFloodFill( CvArr* image, CvPoint seed_point,
                          CvScalar new_val, CvScalar lo_diff CV_DEFAULT(cvScalarAll(0)),
                          CvScalar up_diff CV_DEFAULT(cvScalarAll(0)),
                          CvConnectedComp* comp CV_DEFAULT(NULL),
                          int flags CV_DEFAULT(4),
                          CvArr* mask CV_DEFAULT(NULL));

/****************************************************************************************\
*                                  Feature detection                                     *
\****************************************************************************************/

/* Runs canny edge detector */
CVAPI(void)  cvCanny( const CvArr* image, CvArr* edges, double threshold1,
                      double threshold2, int  aperture_size CV_DEFAULT(3) );

/* Calculates constraint image for corner detection
   Dx^2 * Dyy + Dxx * Dy^2 - 2 * Dx * Dy * Dxy.
   Applying threshold to the result gives coordinates of corners */
CVAPI(void) cvPreCornerDetect( const CvArr* image, CvArr* corners,
                               int aperture_size CV_DEFAULT(3) );

/* Calculates eigen values and vectors of 2x2
   gradient covariation matrix at every image pixel */
CVAPI(void)  cvCornerEigenValsAndVecs( const CvArr* image, CvArr* eigenvv,
                                       int block_size, int aperture_size CV_DEFAULT(3) );

/* Calculates minimal eigenvalue for 2x2 gradient covariation matrix at
   every image pixel */
CVAPI(void)  cvCornerMinEigenVal( const CvArr* image, CvArr* eigenval,
                                  int block_size, int aperture_size CV_DEFAULT(3) );

/* Harris corner detector:
   Calculates det(M) - k*(trace(M)^2), where M is 2x2 gradient covariation matrix for each pixel */
CVAPI(void)  cvCornerHarris( const CvArr* image, CvArr* harris_response,
                             int block_size, int aperture_size CV_DEFAULT(3),
                             double k CV_DEFAULT(0.04) );

/* Adjust corner position using some sort of gradient search */
CVAPI(void)  cvFindCornerSubPix( const CvArr* image, CvPoint2D32f* corners,
                                 int count, CvSize win, CvSize zero_zone,
                                 CvTermCriteria  criteria );

/* Finds a sparse set of points within the selected region
   that seem to be easy to track */
CVAPI(void)  cvGoodFeaturesToTrack( const CvArr* image, CvArr* eig_image,
                                    CvArr* temp_image, CvPoint2D32f* corners,
                                    int* corner_count, double  quality_level,
                                    double  min_distance,
                                    const CvArr* mask CV_DEFAULT(NULL),
                                    int block_size CV_DEFAULT(3),
                                    int use_harris CV_DEFAULT(0),
                                    double k CV_DEFAULT(0.04) );

/* Finds lines on binary image using one of several methods.
   line_storage is either memory storage or 1 x <max number of lines> CvMat, its
   number of columns is changed by the function.
   method is one of CV_HOUGH_*;
   rho, theta and threshold are used for each of those methods;
   param1 ~ line length, param2 ~ line gap - for probabilistic,
   param1 ~ srn, param2 ~ stn - for multi-scale */
CVAPI(CvSeq*)  cvHoughLines2( CvArr* image, void* line_storage, int method,
                              double rho, double theta, int threshold,
                              double param1 CV_DEFAULT(0), double param2 CV_DEFAULT(0),
                              double min_theta CV_DEFAULT(0), double max_theta CV_DEFAULT(CV_PI));

/* Finds circles in the image */
CVAPI(CvSeq*) cvHoughCircles( CvArr* image, void* circle_storage,
                              int method, double dp, double min_dist,
                              double param1 CV_DEFAULT(100),
                              double param2 CV_DEFAULT(100),
                              int min_radius CV_DEFAULT(0),
                              int max_radius CV_DEFAULT(0));

/* Fits a line into set of 2d or 3d points in a robust way (M-estimator technique) */
CVAPI(void)  cvFitLine( const CvArr* points, int dist_type, double param,
                        double reps, double aeps, float* line );

/****************************************************************************************\
*                                     Drawing                                            *
\****************************************************************************************/

/****************************************************************************************\
*       Drawing functions work with images/matrices of arbitrary type.                   *
*       For color images the channel order is BGR[A]                                     *
*       Antialiasing is supported only for 8-bit image now.                              *
*       All the functions include parameter color that means rgb value (that may be      *
*       constructed with CV_RGB macro) for color images and brightness                   *
*       for grayscale images.                                                            *
*       If a drawn figure is partially or completely outside of the image, it is clipped.*
\****************************************************************************************/

#define CV_RGB( r, g, b )  cvScalar( (b), (g), (r), 0 )
#define CV_FILLED -1

#define CV_AA 16

/* Draws 4-connected, 8-connected or antialiased line segment connecting two points */
CVAPI(void)  cvLine( CvArr* img, CvPoint pt1, CvPoint pt2,
                     CvScalar color, int thickness CV_DEFAULT(1),
                     int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0) );

/* Draws a rectangle given two opposite corners of the rectangle (pt1 & pt2),
   if thickness<0 (e.g. thickness == CV_FILLED), the filled box is drawn */
CVAPI(void)  cvRectangle( CvArr* img, CvPoint pt1, CvPoint pt2,
                          CvScalar color, int thickness CV_DEFAULT(1),
                          int line_type CV_DEFAULT(8),
                          int shift CV_DEFAULT(0));

/* Draws a rectangle specified by a CvRect structure */
CVAPI(void)  cvRectangleR( CvArr* img, CvRect r,
                           CvScalar color, int thickness CV_DEFAULT(1),
                           int line_type CV_DEFAULT(8),
                           int shift CV_DEFAULT(0));


/* Draws a circle with specified center and radius.
   Thickness works in the same way as with cvRectangle */
CVAPI(void)  cvCircle( CvArr* img, CvPoint center, int radius,
                       CvScalar color, int thickness CV_DEFAULT(1),
                       int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0));

/* Draws ellipse outline, filled ellipse, elliptic arc or filled elliptic sector,
   depending on <thickness>, <start_angle> and <end_angle> parameters. The resultant figure
   is rotated by <angle>. All the angles are in degrees */
CVAPI(void)  cvEllipse( CvArr* img, CvPoint center, CvSize axes,
                        double angle, double start_angle, double end_angle,
                        CvScalar color, int thickness CV_DEFAULT(1),
                        int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0));

CV_INLINE  void  cvEllipseBox( CvArr* img, CvBox2D box, CvScalar color,
                               int thickness CV_DEFAULT(1),
                               int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0) )
{
    CvSize axes;
    axes.width = cvRound(box.size.width*0.5);
    axes.height = cvRound(box.size.height*0.5);

    cvEllipse( img, cvPointFrom32f( box.center ), axes, box.angle,
               0, 360, color, thickness, line_type, shift );
}

/* Fills convex or monotonous polygon. */
CVAPI(void)  cvFillConvexPoly( CvArr* img, const CvPoint* pts, int npts, CvScalar color,
                               int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0));

/* Fills an area bounded by one or more arbitrary polygons */
CVAPI(void)  cvFillPoly( CvArr* img, CvPoint** pts, const int* npts,
                         int contours, CvScalar color,
                         int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0) );

/* Draws one or more polygonal curves */
CVAPI(void)  cvPolyLine( CvArr* img, CvPoint** pts, const int* npts, int contours,
                         int is_closed, CvScalar color, int thickness CV_DEFAULT(1),
                         int line_type CV_DEFAULT(8), int shift CV_DEFAULT(0) );

#define cvDrawRect cvRectangle
#define cvDrawLine cvLine
#define cvDrawCircle cvCircle
#define cvDrawEllipse cvEllipse
#define cvDrawPolyLine cvPolyLine

/* Clips the line segment connecting *pt1 and *pt2
   by the rectangular window
   (0<=x<img_size.width, 0<=y<img_size.height). */
CVAPI(int) cvClipLine( CvSize img_size, CvPoint* pt1, CvPoint* pt2 );

/* Initializes line iterator. Initially, line_iterator->ptr will point
   to pt1 (or pt2, see left_to_right description) location in the image.
   Returns the number of pixels on the line between the ending points. */
CVAPI(int)  cvInitLineIterator( const CvArr* image, CvPoint pt1, CvPoint pt2,
                                CvLineIterator* line_iterator,
                                int connectivity CV_DEFAULT(8),
                                int left_to_right CV_DEFAULT(0));

/* Moves iterator to the next line point */
#define CV_NEXT_LINE_POINT( line_iterator )                     \
{                                                               \
    int _line_iterator_mask = (line_iterator).err < 0 ? -1 : 0; \
    (line_iterator).err += (line_iterator).minus_delta +        \
        ((line_iterator).plus_delta & _line_iterator_mask);     \
    (line_iterator).ptr += (line_iterator).minus_step +         \
        ((line_iterator).plus_step & _line_iterator_mask);      \
}


/* basic font types */
#define CV_FONT_HERSHEY_SIMPLEX         0
#define CV_FONT_HERSHEY_PLAIN           1
#define CV_FONT_HERSHEY_DUPLEX          2
#define CV_FONT_HERSHEY_COMPLEX         3
#define CV_FONT_HERSHEY_TRIPLEX         4
#define CV_FONT_HERSHEY_COMPLEX_SMALL   5
#define CV_FONT_HERSHEY_SCRIPT_SIMPLEX  6
#define CV_FONT_HERSHEY_SCRIPT_COMPLEX  7

/* font flags */
#define CV_FONT_ITALIC                 16

#define CV_FONT_VECTOR0    CV_FONT_HERSHEY_SIMPLEX


/* Font structure */
typedef struct CvFont
{
  const char* nameFont;   //Qt:nameFont
  CvScalar color;       //Qt:ColorFont -> cvScalar(blue_component, green_component, red\_component[, alpha_component])
    int         font_face;    //Qt: bool italic         /* =CV_FONT_* */
    const int*  ascii;      /* font data and metrics */
    const int*  greek;
    const int*  cyrillic;
    float       hscale, vscale;
    float       shear;      /* slope coefficient: 0 - normal, >0 - italic */
    int         thickness;    //Qt: weight               /* letters thickness */
    float       dx;       /* horizontal interval between letters */
    int         line_type;    //Qt: PointSize
}
CvFont;

/* Initializes font structure used further in cvPutText */
CVAPI(void)  cvInitFont( CvFont* font, int font_face,
                         double hscale, double vscale,
                         double shear CV_DEFAULT(0),
                         int thickness CV_DEFAULT(1),
                         int line_type CV_DEFAULT(8));

CV_INLINE CvFont cvFont( double scale, int thickness CV_DEFAULT(1) )
{
    CvFont font;
    cvInitFont( &font, CV_FONT_HERSHEY_PLAIN, scale, scale, 0, thickness, CV_AA );
    return font;
}

/* Renders text stroke with specified font and color at specified location.
   CvFont should be initialized with cvInitFont */
CVAPI(void)  cvPutText( CvArr* img, const char* text, CvPoint org,
                        const CvFont* font, CvScalar color );

/* Calculates bounding box of text stroke (useful for alignment) */
CVAPI(void)  cvGetTextSize( const char* text_string, const CvFont* font,
                            CvSize* text_size, int* baseline );

/* Unpacks color value, if arrtype is CV_8UC?, <color> is treated as
   packed color value, otherwise the first channels (depending on arrtype)
   of destination scalar are set to the same value = <color> */
CVAPI(CvScalar)  cvColorToScalar( double packed_color, int arrtype );

/* Returns the polygon points which make up the given ellipse.  The ellipse is define by
   the box of size 'axes' rotated 'angle' around the 'center'.  A partial sweep
   of the ellipse arc can be done by spcifying arc_start and arc_end to be something
   other than 0 and 360, respectively.  The input array 'pts' must be large enough to
   hold the result.  The total number of points stored into 'pts' is returned by this
   function. */
CVAPI(int) cvEllipse2Poly( CvPoint center, CvSize axes,
                 int angle, int arc_start, int arc_end, CvPoint * pts, int delta );

/* Draws contour outlines or filled interiors on the image */
CVAPI(void)  cvDrawContours( CvArr *img, CvSeq* contour,
                             CvScalar external_color, CvScalar hole_color,
                             int max_level, int thickness CV_DEFAULT(1),
                             int line_type CV_DEFAULT(8),
                             CvPoint offset CV_DEFAULT(cvPoint(0,0)));

#ifdef __cplusplus
}
#endif

#endif
