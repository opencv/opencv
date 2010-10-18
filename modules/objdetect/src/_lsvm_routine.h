#ifndef _LSVM_ROUTINE_H
#define _LSVM_ROUTINE_H

#include "_lsvm_types.h"
#include "_lsvm_error.h"


//////////////////////////////////////////////////////////////
// Memory management routines
// All paramaters names correspond to previous data structures description
// All "alloc" functions return allocated memory for 1 object
// with all fields including arrays
// Error status is return value
//////////////////////////////////////////////////////////////
int allocFilterObject(CvLSVMFilterObject **obj, const int sizeX, const int sizeY, 
                      const int p, const int xp);
int freeFilterObject (CvLSVMFilterObject **obj);

int allocFeatureMapObject(CvLSVMFeatureMap **obj, const int sizeX, const int sizeY,
                          const int p, const int xp);
int freeFeatureMapObject (CvLSVMFeatureMap **obj);

#ifdef __cplusplus
extern "C"
#endif
int allocFeaturePyramidObject(CvLSVMFeaturePyramid **obj, 
                              const int lambda, const int countLevel);

#ifdef __cplusplus
extern "C"
#endif
int freeFeaturePyramidObject (CvLSVMFeaturePyramid **obj);
int allocFFTImage(CvLSVMFftImage **image, int p, int dimX, int dimY);
int freeFFTImage(CvLSVMFftImage **image);
#endif