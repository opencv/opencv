#ifndef _LSVM_TBBVERSION_H
#define _LSVM_TBBVERSION_H

#include "_lsvm_matching.h"
#include "precomp.hpp"

/*
// Computation score function using TBB tasks
//
// API
// int tbbTasksThresholdFunctionalScore(const CvLSVMFilterObject **filters, const int n, 
                                        const CvLSVMFeaturePyramid *H, const float b,
                                        const int maxXBorder, const int maxYBorder,
                                        const float scoreThreshold,
                                        int *kLevels, int **procLevels,
                                        const int threadsNum,
                                        float **score, CvPoint ***points, 
                                        int *kPoints,
                                        CvPoint ****partsDisplacement);
// INPUT
// filters           - the set of filters (the first element is root filter, 
                       the other - part filters)
// n                 - the number of part filters
// H                 - feature pyramid
// b                 - linear term of the score function
// maxXBorder        - the largest root filter size (X-direction)
// maxYBorder        - the largest root filter size (Y-direction)
// scoreThreshold    - score threshold
// kLevels           - array that contains number of levels processed 
                       by each thread
// procLevels        - array that contains lists of levels processed 
                       by each thread
// threadsNum        - the number of created threads
// OUTPUT
// score             - score function values that exceed threshold
// points            - the set of root filter positions (in the block space)
// kPoints           - number of root filter positions
// partsDisplacement - displacement of part filters (in the block space)
// RESULT
//
*/
int tbbTasksThresholdFunctionalScore(const CvLSVMFilterObject **filters, const int n, 
                                     const CvLSVMFeaturePyramid *H, const float b,
                                     const int maxXBorder, const int maxYBorder,
                                     const float scoreThreshold,
                                     int *kLevels, int **procLevels,
                                     const int threadsNum,
                                     float **score, CvPoint ***points, 
                                     int *kPoints,
                                     CvPoint ****partsDisplacement);

#endif