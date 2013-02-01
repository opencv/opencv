#ifndef FUNCTION_SC
#define FUNCTION_SC

#include "_lsvm_types.h"

float calcM         (int k,int di,int dj, const CvLSVMFeaturePyramid * H, const CvLSVMFilterObject *filter);
float calcM_PCA     (int k,int di,int dj, const CvLSVMFeaturePyramid * H, const CvLSVMFilterObject *filter);
float calcM_PCA_cash(int k,int di,int dj, const CvLSVMFeaturePyramid * H, const CvLSVMFilterObject *filter, float * cashM, int * maskM, int step);
float calcFine (const CvLSVMFilterObject *filter, int di, int dj);

#endif