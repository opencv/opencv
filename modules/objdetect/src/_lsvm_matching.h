/*****************************************************************************/
/*                      Matching procedure API                               */
/*****************************************************************************/
//
#ifndef _LSVM_MATCHING_H_
#define _LSVM_MATCHING_H_

#include "_latentsvm.h"
#include "_lsvm_error.h"
#include "_lsvm_routine.h"

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
int nonMaximumSuppression(int numBoxes, const CvPoint *points, 
                          const CvPoint *oppositePoints, const float *score,
                          float overlapThreshold, 
                          int *numBoxesOut, CvPoint **pointsOut, 
                          CvPoint **oppositePointsOut, float **scoreOut);
int getMaxFilterDims(const CvLSVMFilterObject **filters, int kComponents,
                     const int *kPartFilters, 
                     unsigned int *maxXBorder, unsigned int *maxYBorder);
//}

int getMaxFilterDims(const CvLSVMFilterObject **filters, int kComponents,
                     const int *kPartFilters, 
                     unsigned int *maxXBorder, unsigned int *maxYBorder);
#endif
