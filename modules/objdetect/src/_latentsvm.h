/*****************************************************************************/
/*                      Latent SVM prediction API                            */
/*****************************************************************************/

#ifndef _LATENTSVM_H_
#define _LATENTSVM_H_

#include <stdio.h>
#include "_lsvm_types.h"
#include "_lsvm_error.h"
#include "_lsvm_routine.h"

//////////////////////////////////////////////////////////////
// Building feature pyramid
// (pyramid constructed both contrast and non-contrast image)
//////////////////////////////////////////////////////////////

void FeaturePyramid32(CvLSVMFeaturePyramid* H, int maxX, int maxY);

/*
// Creation PSA feature pyramid
//
// API
// featurePyramid* createPSA_FeaturePyramid(featurePyramid* H);

// INPUT
// H                 - feature pyramid     
// OUTPUT
// RESULT
// PSA feature pyramid
*/
CvLSVMFeaturePyramid* createPCA_FeaturePyramid(CvLSVMFeaturePyramid* H, 
                                               CvLatentSvmDetector* detector, 
                                               int maxX, int maxY);

/*
// Getting feature pyramid  
//
// API
// int getFeaturePyramid(IplImage * image, const CvLSVMFilterObject **all_F, 
                      const int n_f,
                      const int lambda, const int k, 
                      const int startX, const int startY, 
                      const int W, const int H, featurePyramid **maps);
// INPUT
// image             - image
// lambda            - resize scale
// k                 - size of cells
// startX            - X coordinate of the image rectangle to search
// startY            - Y coordinate of the image rectangle to search
// W                 - width of the image rectangle to search
// H                 - height of the image rectangle to search
// OUTPUT
// maps              - feature maps for all levels
// RESULT
// Error status
*/
int getFeaturePyramid(IplImage * image, CvLSVMFeaturePyramid **maps);

/*
// Getting feature map for the selected subimage  
//
// API
// int getFeatureMaps(const IplImage * image, const int k, featureMap **map);
// INPUT
// image             - selected subimage
// k                 - size of cells
// OUTPUT
// map               - feature map
// RESULT
// Error status
*/
int getFeatureMaps(const IplImage * image, const int k, CvLSVMFeatureMap **map);


/*
// Feature map Normalization and Truncation 
//
// API
// int normalizationAndTruncationFeatureMaps(featureMap *map, const float alfa);
// INPUT
// map               - feature map
// alfa              - truncation threshold
// OUTPUT
// map               - truncated and normalized feature map
// RESULT
// Error status
*/
int normalizeAndTruncate(CvLSVMFeatureMap *map, const float alfa);

/*
// Feature map reduction
// In each cell we reduce dimension of the feature vector
// according to original paper special procedure
//
// API
// int PCAFeatureMaps(featureMap *map)
// INPUT
// map               - feature map
// OUTPUT
// map               - feature map
// RESULT
// Error status
*/
int PCAFeatureMaps(CvLSVMFeatureMap *map);

//////////////////////////////////////////////////////////////
// search object
//////////////////////////////////////////////////////////////

/*
// Transformation filter displacement from the block space 
// to the space of pixels at the initial image
//
// API
// int convertPoints(int countLevel, int lambda, 
                     int initialImageLevel,
                     CvPoint *points, int *levels, 
                     CvPoint **partsDisplacement, int kPoints, int n, 
                     int maxXBorder,
                     int maxYBorder);
// INPUT
// countLevel        - the number of levels in the feature pyramid
// lambda            - method parameter
// initialImageLevel - level of feature pyramid that contains feature map
                       for initial image
// points            - the set of root filter positions (in the block space)
// levels            - the set of levels
// partsDisplacement - displacement of part filters (in the block space)
// kPoints           - number of root filter positions
// n                 - number of part filters
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// OUTPUT
// points            - the set of root filter positions (in the space of pixels)
// partsDisplacement - displacement of part filters (in the space of pixels)
// RESULT
// Error status
*/
int convertPoints(int countLevel, int lambda, 
                  int initialImageLevel,
                  CvPoint *points, int *levels, 
                  CvPoint **partsDisplacement, int kPoints, int n, 
                  int maxXBorder,
                  int maxYBorder);

/*
// Elimination boxes that are outside the image boudaries
//
// API
// int clippingBoxes(int width, int height, 
                     CvPoint *points, int kPoints);
// INPUT
// width             - image wediht
// height            - image heigth
// points            - a set of points (coordinates of top left or
                       bottom right corners)
// kPoints           - points number
// OUTPUT
// points            - updated points (if coordinates less than zero then
                       set zero coordinate, if coordinates more than image 
                       size then set coordinates equal image size)
// RESULT
// Error status
*/
int clippingBoxes(int width, int height, 
                  CvPoint *points, int kPoints);

/*
// Creation feature pyramid with nullable border
//
// API
// featurePyramid* createFeaturePyramidWithBorder(const IplImage *image,
                                                  int maxXBorder, int maxYBorder);

// INPUT
// image             - initial image     
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// OUTPUT
// RESULT
// Feature pyramid with nullable border
*/
CvLSVMFeaturePyramid* createFeaturePyramidWithBorder(IplImage *image,
                                               int maxXBorder, int maxYBorder);

/*
// Computation root filters displacement and values of score function
//
// API
// int searchObjectThresholdSomeComponents(const featurePyramid *H,
                                           const CvLSVMFilterObject **filters, 
                                           int kComponents, const int *kPartFilters,
                                           const float *b, float scoreThreshold,
                                           CvPoint **points, CvPoint **oppPoints,
                                           float **score, int *kPoints);
// INPUT
// H                 - feature pyramid
// filters           - filters (root filter then it's part filters, etc.)
// kComponents       - root filters number
// kPartFilters      - array of part filters number for each component
// b                 - array of linear terms
// scoreThreshold    - score threshold
// OUTPUT
// points            - root filters displacement (top left corners)
// oppPoints         - root filters displacement (bottom right corners)
// score             - array of score values
// kPoints           - number of boxes
// RESULT
// Error status
*/
int searchObjectThresholdSomeComponents(const CvLSVMFeaturePyramid *H,
										const CvLSVMFeaturePyramid *H_PCA,
                                        const CvLSVMFilterObject **filters, 
                                        int kComponents, const int *kPartFilters,
                                        const float *b, float scoreThreshold,
                                        CvPoint **points, CvPoint **oppPoints,
                                        float **score, int *kPoints);

/*
// Compute opposite point for filter box
//
// API
// int getOppositePoint(CvPoint point,
                        int sizeX, int sizeY,
                        float step, int degree,
                        CvPoint *oppositePoint);

// INPUT
// point             - coordinates of filter top left corner
                       (in the space of pixels)
// (sizeX, sizeY)    - filter dimension in the block space
// step              - scaling factor
// degree            - degree of the scaling factor
// OUTPUT
// oppositePoint     - coordinates of filter bottom corner
                       (in the space of pixels)
// RESULT
// Error status
*/
int getOppositePoint(CvPoint point,
                     int sizeX, int sizeY,
                     float step, int degree,
                     CvPoint *oppositePoint);

/*
// Drawing root filter boxes
//
// API
// int showRootFilterBoxes(const IplImage *image,
                           const CvLSVMFilterObject *filter, 
                           CvPoint *points, int *levels, int kPoints,
                           CvScalar color, int thickness, 
                           int line_type, int shift);
// INPUT
// image             - initial image
// filter            - root filter object
// points            - a set of points
// levels            - levels of feature pyramid
// kPoints           - number of points
// color             - line color for each box
// thickness         - line thickness
// line_type         - line type
// shift             - shift
// OUTPUT
// window contained initial image and filter boxes
// RESULT
// Error status
*/
int showRootFilterBoxes(IplImage *image,
                        const CvLSVMFilterObject *filter, 
                        CvPoint *points, int *levels, int kPoints,
                        CvScalar color, int thickness, 
                        int line_type, int shift);

/*
// Drawing part filter boxes
//
// API
// int showPartFilterBoxes(const IplImage *image,
                           const CvLSVMFilterObject *filter, 
                           CvPoint *points, int *levels, int kPoints,
                           CvScalar color, int thickness, 
                           int line_type, int shift);
// INPUT
// image             - initial image
// filters           - a set of part filters
// n                 - number of part filters
// partsDisplacement - a set of points
// levels            - levels of feature pyramid
// kPoints           - number of foot filter positions
// color             - line color for each box
// thickness         - line thickness
// line_type         - line type
// shift             - shift
// OUTPUT
// window contained initial image and filter boxes
// RESULT
// Error status
*/
int showPartFilterBoxes(IplImage *image,
                        const CvLSVMFilterObject **filters,
                        int n, CvPoint **partsDisplacement, 
                        int *levels, int kPoints,
                        CvScalar color, int thickness, 
                        int line_type, int shift);

/*
// Drawing boxes
//
// API
// int showBoxes(const IplImage *img, 
                 const CvPoint *points, const CvPoint *oppositePoints, int kPoints, 
                 CvScalar color, int thickness, int line_type, int shift);
// INPUT
// img               - initial image
// points            - top left corner coordinates
// oppositePoints    - right bottom corner coordinates
// kPoints           - points number
// color             - line color for each box
// thickness         - line thickness
// line_type         - line type
// shift             - shift
// OUTPUT
// RESULT
// Error status
*/
int showBoxes(IplImage *img, 
              const CvPoint *points, const CvPoint *oppositePoints, int kPoints, 
              CvScalar color, int thickness, int line_type, int shift);

#endif
