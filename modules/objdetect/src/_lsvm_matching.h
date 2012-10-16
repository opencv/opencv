/*****************************************************************************/
/*                      Matching procedure API                               */
/*****************************************************************************/
//
#ifndef _LSVM_MATCHING_H_
#define _LSVM_MATCHING_H_

#include "_latentsvm.h"
#include "_lsvm_error.h"
#include "_lsvm_distancetransform.h"
#include "_lsvm_fft.h"
#include "_lsvm_routine.h"

#ifdef HAVE_TBB
#include "_lsvm_tbbversion.h"
#endif

//extern "C" {
/*
// Function for convolution computation
//
// API
// int convolution(const filterObject *Fi, const featureMap *map, float *f);
// INPUT
// Fi                - filter object
// map               - feature map
// OUTPUT
// f                 - the convolution
// RESULT
// Error status
*/
int convolution(const CvLSVMFilterObject *Fi, const CvLSVMFeatureMap *map, float *f);

/*
// Computation multiplication of FFT images
//
// API
// int fftImagesMulti(float *fftImage1, float *fftImage2, int numRows, int numColls,
                      float *multi);
// INPUT
// fftImage1         - first fft image
// fftImage2         - second fft image
// (numRows, numColls) - image dimesions
// OUTPUT
// multi             - multiplication
// RESULT
// Error status
*/
int fftImagesMulti(float *fftImage1, float *fftImage2, int numRows, int numColls,
                   float *multi);

/*
// Turnover filter matrix for the single feature
//
// API
// int rot2PI(float *filter, int dimX, int dimY, float *rot2PIFilter,
              int p, int shift);
// INPUT
// filter            - filter weight matrix
// (dimX, dimY)      - dimension of filter matrix
// p                 - number of features
// shift             - number of feature (or channel)
// OUTPUT
// rot2PIFilter      - rotated matrix
// RESULT
// Error status
*/
int rot2PI(float *filter, int dimX, int dimY, float *rot2PIFilter,
           int p, int shift);

/*
// Addition nullable bars to the dimension of feature map (single feature)
//
// API
// int addNullableBars(float *rot2PIFilter, int dimX, int dimY,
                       float *newFilter, int newDimX, int newDimY);
// INPUT
// rot2PIFilter      - filter matrix for the single feature that was rotated
// (dimX, dimY)      - dimension rot2PIFilter
// (newDimX, newDimY)- dimension of feature map for the single feature
// OUTPUT
// newFilter         - filter matrix with nullable bars
// RESULT
// Error status
*/
int addNullableBars(float *rot2PIFilter, int dimX, int dimY,
                    float *newFilter, int newDimX, int newDimY);

/*
// Computation FFT image for filter object
//
// API
// int getFFTImageFilterObject(const filterObject *filter,
                               int mapDimX, int mapDimY,
                               fftImage **image);
// INPUT
// filter        - filter object
// (mapDimX, mapDimY)- dimension of feature map
// OUTPUT
// image         - fft image
// RESULT
// Error status
*/
int getFFTImageFilterObject(const CvLSVMFilterObject *filter,
                            int mapDimX, int mapDimY,
                            CvLSVMFftImage **image);

/*
// Computation FFT image for feature map
//
// API
// int getFFTImageFeatureMap(const featureMap *map, fftImage **image);
// INPUT
// OUTPUT
// RESULT
// Error status
*/
int getFFTImageFeatureMap(const CvLSVMFeatureMap *map, CvLSVMFftImage **image);

/*
// Function for convolution computation using FFT
//
// API
// int convFFTConv2d(const fftImage *featMapImage, const fftImage *filterImage,
                     int filterDimX, int filterDimY, float **conv);
// INPUT
// featMapImage      - feature map image
// filterImage       - filter image
// (filterDimX,filterDimY) - filter dimension
// OUTPUT
// conv              - the convolution
// RESULT
// Error status
*/
int convFFTConv2d(const CvLSVMFftImage *featMapImage, const CvLSVMFftImage *filterImage,
                  int filterDimX, int filterDimY, float **conv);

/*
// Computation objective function D according the original paper
//
// API
// int filterDispositionLevel(const filterObject *Fi, const featureMap *pyramid,
                              float **scoreFi,
                              int **pointsX, int **pointsY);
// INPUT
// Fi                - filter object (weights and coefficients of penalty
                       function that are used in this routine)
// pyramid           - feature map
// OUTPUT
// scoreFi           - values of distance transform on the level at all positions
// (pointsX, pointsY)- positions that correspond to the maximum value
                       of distance transform at all grid nodes
// RESULT
// Error status
*/
int filterDispositionLevel(const CvLSVMFilterObject *Fi, const CvLSVMFeatureMap *pyramid,
                           float **scoreFi,
                           int **pointsX, int **pointsY);

/*
// Computation objective function D according the original paper using FFT
//
// API
// int filterDispositionLevelFFT(const filterObject *Fi, const fftImage *featMapImage,
                                 float **scoreFi,
                                 int **pointsX, int **pointsY);
// INPUT
// Fi                - filter object (weights and coefficients of penalty
                       function that are used in this routine)
// featMapImage      - FFT image of feature map
// OUTPUT
// scoreFi           - values of distance transform on the level at all positions
// (pointsX, pointsY)- positions that correspond to the maximum value
                       of distance transform at all grid nodes
// RESULT
// Error status
*/
int filterDispositionLevelFFT(const CvLSVMFilterObject *Fi, const CvLSVMFftImage *featMapImage,
                              float **scoreFi,
                              int **pointsX, int **pointsY);

/*
// Computation border size for feature map
//
// API
// int computeBorderSize(int maxXBorder, int maxYBorder, int *bx, int *by);
// INPUT
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// OUTPUT
// bx                - border size (X-direction)
// by                - border size (Y-direction)
// RESULT
// Error status
*/
int computeBorderSize(int maxXBorder, int maxYBorder, int *bx, int *by);

/*
// Addition nullable border to the feature map
//
// API
// int addNullableBorder(featureMap *map, int bx, int by);
// INPUT
// map               - feature map
// bx                - border size (X-direction)
// by                - border size (Y-direction)
// OUTPUT
// RESULT
// Error status
*/
int addNullableBorder(CvLSVMFeatureMap *map, int bx, int by);

/*
// Computation the maximum of the score function at the level
//
// API
// int maxFunctionalScoreFixedLevel(const filterObject **all_F, int n,
                                    const featurePyramid *H,
                                    int level, float b,
                                    int maxXBorder, int maxYBorder,
                                    float *score, CvPoint **points, int *kPoints,
                                    CvPoint ***partsDisplacement);
// INPUT
// all_F             - the set of filters (the first element is root filter,
                       the other - part filters)
// n                 - the number of part filters
// H                 - feature pyramid
// level             - feature pyramid level for computation maximum score
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// OUTPUT
// score             - the maximum of the score function at the level
// points            - the set of root filter positions (in the block space)
// levels            - the set of levels
// kPoints           - number of root filter positions
// partsDisplacement - displacement of part filters (in the block space)
// RESULT
// Error status
*/
int maxFunctionalScoreFixedLevel(const CvLSVMFilterObject **all_F, int n,
                                 const CvLSVMFeaturePyramid *H,
                                 int level, float b,
                                 int maxXBorder, int maxYBorder,
                                 float *score, CvPoint **points, int *kPoints,
                                 CvPoint ***partsDisplacement);

/*
// Computation score function at the level that exceed threshold
//
// API
// int thresholdFunctionalScoreFixedLevel(const filterObject **all_F, int n,
                                          const featurePyramid *H,
                                          int level, float b,
                                          int maxXBorder, int maxYBorder,
                                          float scoreThreshold,
                                          float **score, CvPoint **points, int *kPoints,
                                          CvPoint ***partsDisplacement);
// INPUT
// all_F             - the set of filters (the first element is root filter,
                       the other - part filters)
// n                 - the number of part filters
// H                 - feature pyramid
// level             - feature pyramid level for computation maximum score
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// scoreThreshold    - score threshold
// OUTPUT
// score             - score function at the level that exceed threshold
// points            - the set of root filter positions (in the block space)
// levels            - the set of levels
// kPoints           - number of root filter positions
// partsDisplacement - displacement of part filters (in the block space)
// RESULT
// Error status
*/
int thresholdFunctionalScoreFixedLevel(const CvLSVMFilterObject **all_F, int n,
                                       const CvLSVMFeaturePyramid *H,
                                       int level, float b,
                                       int maxXBorder, int maxYBorder,
                                       float scoreThreshold,
                                       float **score, CvPoint **points, int *kPoints,
                                       CvPoint ***partsDisplacement);

/*
// Computation the maximum of the score function
//
// API
// int maxFunctionalScore(const filterObject **all_F, int n,
                          const featurePyramid *H, float b,
                          int maxXBorder, int maxYBorder,
                          float *score,
                          CvPoint **points, int **levels, int *kPoints,
                          CvPoint ***partsDisplacement);
// INPUT
// all_F             - the set of filters (the first element is root filter,
                       the other - part filters)
// n                 - the number of part filters
// H                 - feature pyramid
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// OUTPUT
// score             - the maximum of the score function
// points            - the set of root filter positions (in the block space)
// levels            - the set of levels
// kPoints           - number of root filter positions
// partsDisplacement - displacement of part filters (in the block space)
// RESULT
// Error status
*/
int maxFunctionalScore(const CvLSVMFilterObject **all_F, int n,
                       const CvLSVMFeaturePyramid *H, float b,
                       int maxXBorder, int maxYBorder,
                       float *score,
                       CvPoint **points, int **levels, int *kPoints,
                       CvPoint ***partsDisplacement);

/*
// Computation score function that exceed threshold
//
// API
// int thresholdFunctionalScore(const filterObject **all_F, int n,
                                const featurePyramid *H,
                                float b,
                                int maxXBorder, int maxYBorder,
                                float scoreThreshold,
                                float **score,
                                CvPoint **points, int **levels, int *kPoints,
                                CvPoint ***partsDisplacement);
// INPUT
// all_F             - the set of filters (the first element is root filter,
                       the other - part filters)
// n                 - the number of part filters
// H                 - feature pyramid
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// scoreThreshold    - score threshold
// OUTPUT
// score             - score function values that exceed threshold
// points            - the set of root filter positions (in the block space)
// levels            - the set of levels
// kPoints           - number of root filter positions
// partsDisplacement - displacement of part filters (in the block space)
// RESULT
// Error status
*/
int thresholdFunctionalScore(const CvLSVMFilterObject **all_F, int n,
                             const CvLSVMFeaturePyramid *H,
                             float b,
                             int maxXBorder, int maxYBorder,
                             float scoreThreshold,
                             float **score,
                             CvPoint **points, int **levels, int *kPoints,
                             CvPoint ***partsDisplacement);

#ifdef HAVE_TBB
/*
// int tbbThresholdFunctionalScore(const CvLSVMFilterObject **all_F, int n,
                                   const CvLSVMFeaturePyramid *H,
                                   const float b,
                                   const int maxXBorder, const int maxYBorder,
                                   const float scoreThreshold,
                                   const int threadsNum,
                                   float **score,
                                   CvPoint **points, int **levels, int *kPoints,
                                   CvPoint ***partsDisplacement);
// INPUT
// all_F             - the set of filters (the first element is root filter,
                       the other - part filters)
// n                 - the number of part filters
// H                 - feature pyramid
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// scoreThreshold    - score threshold
// threadsNum        - number of threads that will be created using TBB version
// OUTPUT
// score             - score function values that exceed threshold
// points            - the set of root filter positions (in the block space)
// levels            - the set of levels
// kPoints           - number of root filter positions
// partsDisplacement - displacement of part filters (in the block space)
// RESULT
// Error status
*/
int tbbThresholdFunctionalScore(const CvLSVMFilterObject **all_F, int n,
                                const CvLSVMFeaturePyramid *H,
                                const float b,
                                const int maxXBorder, const int maxYBorder,
                                const float scoreThreshold,
                                const int threadsNum,
                                float **score,
                                CvPoint **points, int **levels, int *kPoints,
                                CvPoint ***partsDisplacement);
#endif

/*
// Perform non-maximum suppression algorithm (described in original paper)
// to remove "similar" bounding boxes
//
// API
// int nonMaximumSuppression(int numBoxes, const CvPoint *points,
                             const CvPoint *oppositePoints, const float *score,
                             float overlapThreshold,
                             int *numBoxesout, CvPoint **pointsOut,
                             CvPoint **oppositePointsOut, float **scoreOut);
// INPUT
// numBoxes          - number of bounding boxes
// points            - array of left top corner coordinates
// oppositePoints    - array of right bottom corner coordinates
// score             - array of detection scores
// overlapThreshold  - threshold: bounding box is removed if overlap part
                       is greater than passed value
// OUTPUT
// numBoxesOut       - the number of bounding boxes algorithm returns
// pointsOut         - array of left top corner coordinates
// oppositePointsOut - array of right bottom corner coordinates
// scoreOut          - array of detection scores
// RESULT
// Error status
*/
#ifdef __cplusplus
extern "C"
#endif
int nonMaximumSuppression(int numBoxes, const CvPoint *points,
                          const CvPoint *oppositePoints, const float *score,
                          float overlapThreshold,
                          int *numBoxesOut, CvPoint **pointsOut,
                          CvPoint **oppositePointsOut, float **scoreOut);
#ifdef __cplusplus
extern "C"
#endif
int getMaxFilterDims(const CvLSVMFilterObject **filters, int kComponents,
                     const int *kPartFilters,
                     unsigned int *maxXBorder, unsigned int *maxYBorder);
//}
#endif
