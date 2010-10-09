#ifndef _ROUTINE_H
#define _ROUTINE_H

#include "precomp.hpp"
#include "_types.h"
#include "_error.h"


//////////////////////////////////////////////////////////////
// Memory management routines
// All paramaters names correspond to previous data structures description
// All "alloc" functions return allocated memory for 1 object
// with all fields including arrays
// Error status is return value
//////////////////////////////////////////////////////////////
int allocFilterObject(filterObject **obj, const int sizeX, const int sizeY, 
                      const int p, const int xp);
int freeFilterObject (filterObject **obj);

int allocFeatureMapObject(featureMap **obj, const int sizeX, const int sizeY,
                          const int p, const int xp);
int freeFeatureMapObject (featureMap **obj);

#ifdef __cplusplus
extern "C"
#endif
int allocFeaturePyramidObject(featurePyramid **obj, 
                              const int lambda, const int countLevel);

#ifdef __cplusplus
extern "C"
#endif
int freeFeaturePyramidObject (featurePyramid **obj);
int allocFFTImage(fftImage **image, int p, int dimX, int dimY);
int freeFFTImage(fftImage **image);
#endif