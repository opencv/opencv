#ifndef SVM_TYPE
#define SVM_TYPE

//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
#include "precomp.hpp"

//#define FFT_CONV

// «начение числа PI
#define PI    3.1415926535897932384626433832795  

// “очность сравнени€ пары вещественных чисел
#define EPS 0.000001

// ћинимальное и максимальное значение дл€ вещественного типа данных
#define F_MAX 3.402823466e+38
#define F_MIN -3.402823465e+38

// The number of elements in bin
// The number of sectors in gradient histogram building
#define CNTPARTION 9

// The number of levels in image resize procedure
// We need Lambda levels to resize image twice
#define LAMBDA 10

// Block size. Used in feature pyramid building procedure
#define SIDE_LENGTH 8

//////////////////////////////////////////////////////////////
// main data structures                                     //
//////////////////////////////////////////////////////////////

// DataType: STRUCT featureMap
// FEATURE MAP DESCRIPTION
//   Rectangular map (sizeX x sizeY), 
//   every cell stores feature vector (dimension = p)
// H               - matrix of feature vectors
//                   to set and get feature vectors (i,j) 
//                   used formula Map[(j * sizeX + i) * p + k], where
//                   k - component of feature vector in cell (i, j)
// END OF FEATURE MAP DESCRIPTION
// xp              - auxillary parameter for internal use
//                   size of row in feature vectors 
//                   (yp = (int) (p / xp); p = xp * yp)
typedef struct{
    int sizeX;
    int sizeY;
    int p;
    int xp;
    float *Map;
} featureMap;

// DataType: STRUCT featurePyramid
//
// countLevel   - number of levels in the feature pyramid
// lambda       - resize scale coefficient
// pyramid      - array of pointers to feature map at different levels
typedef struct{
    int countLevel;
    int lambda;
    featureMap **pyramid;
} featurePyramid;

// DataType: STRUCT filterDisposition
// The structure stores preliminary results in optimization process
// with objective function D 
//
// x            - array with X coordinates of optimization problems solutions
// y            - array with Y coordinates of optimization problems solutions
// score        - array with optimal objective values
typedef struct{
    float *score;
    int *x;
    int *y;
} filterDisposition;

// DataType: STRUCT fftImage
// The structure stores FFT image
//
// p            - number of channels
// x            - array of FFT images for 2d signals
// n            - number of rows
// m            - number of collums
typedef struct{
    unsigned int p;
    unsigned int dimX;
    unsigned int dimY;
    float **channels;
} fftImage;

#endif
