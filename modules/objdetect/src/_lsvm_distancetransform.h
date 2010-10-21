#ifndef _LSVM_DIST_TRANSFORM_H_
#define _LSVM_DIST_TRANSFORM_H_

#include "_lsvm_types.h"
#include "_lsvm_error.h"

/*
// Computation the point of intersection functions
// (parabolas on the variable y)
//      a(y - q1) + b(q1 - y)(q1 - y) + f[q1]
//      a(y - q2) + b(q2 - y)(q2 - y) + f[q2]
//
// API
// int GetPointOfIntersection(const F_type *f, 
                              const F_type a, const F_type b,           
                              int q1, int q2, F_type *point);
// INPUT
// f                - function on the regular grid
// a                - coefficient of the function
// b                - coefficient of the function
// q1               - parameter of the function
// q2               - parameter of the function
// OUTPUT
// point            - point of intersection
// RESULT
// Error status
*/
int GetPointOfIntersection(const float *f, 
                           const float a, const float b,           
                           int q1, int q2, float *point);

/*
// Decision of one dimensional problem generalized distance transform
// on the regular grid at all points
//      min (a(y' - y) + b(y' - y)(y' - y) + f(y')) (on y')
//
// API
// int DistanceTransformOneDimensionalProblem(const F_type *f, const int n,
                                              const F_type a, const F_type b,                                             
                                              F_type *distanceTransform,
                                              int *points); 
// INPUT
// f                 - function on the regular grid
// n                 - grid dimension
// a                 - coefficient of optimizable function
// b                 - coefficient of optimizable function
// OUTPUT
// distanceTransform - values of generalized distance transform
// points            - arguments that corresponds to the optimal value of function
// RESULT
// Error status
*/
int DistanceTransformOneDimensionalProblem(const float *f, const int n,
                                           const float a, const float b,                                             
                                           float *distanceTransform,
                                           int *points); 

/*
// Computation next cycle element
//
// API
// int GetNextCycleElement(int k, int n, int q);
// INPUT
// k                 - index of the previous cycle element
// n                 - number of matrix rows
// q                 - parameter that equal (number_of_rows * number_of_columns - 1)
// OUTPUT
// None
// RESULT
// Next cycle element
*/
int GetNextCycleElement(int k, int n, int q);

/*
// Transposition of cycle elements
//
// API
// void TransposeCycleElements(F_type *a, int *cycle, int cycle_len);
// INPUT
// a                 - initial matrix
// cycle             - cycle
// cycle_len         - cycle length
// OUTPUT
// a                 - matrix with transposed elements
// RESULT
// None
*/
void TransposeCycleElements(float *a, int *cycle, int cycle_len);

/*
// Getting transposed matrix
//
// API
// void Transpose(F_type *a, int n, int m);
// INPUT
// a                 - initial matrix
// n                 - number of rows
// m                 - number of columns
// OUTPUT
// a                 - transposed matrix
// RESULT
// Error status
*/
void Transpose(float *a, int n, int m);

/*
// Decision of two dimensional problem generalized distance transform
// on the regular grid at all points
//      min{d2(y' - y) + d4(y' - y)(y' - y) + 
            min(d1(x' - x) + d3(x' - x)(x' - x) + f(x',y'))} (on x', y')
//
// API
// int DistanceTransformTwoDimensionalProblem(const F_type *f, 
                                              const int n, const int m,
                                              const F_type coeff[4],                                             
                                              F_type *distanceTransform,
                                              int *pointsX, int *pointsY); 
// INPUT
// f                 - function on the regular grid
// n                 - number of rows
// m                 - number of columns
// coeff             - coefficients of optimizable function
                       coeff[0] = d1, coeff[1] = d2, 
                       coeff[2] = d3, coeff[3] = d4
// OUTPUT
// distanceTransform - values of generalized distance transform
// pointsX           - arguments x' that correspond to the optimal value
// pointsY           - arguments y' that correspond to the optimal value
// RESULT
// Error status
*/
int DistanceTransformTwoDimensionalProblem(const float *f, 
                                           const int n, const int m,
                                           const float coeff[4],                                             
                                           float *distanceTransform,
                                           int *pointsX, int *pointsY); 

#endif
