#include "precomp.hpp"
#include "_lsvm_distancetransform.h"

/*
// Computation the point of intersection functions
// (parabolas on the variable y)
//      a(y - q1) + b(q1 - y)(q1 - y) + f[q1]
//      a(y - q2) + b(q2 - y)(q2 - y) + f[q2]
//
//
// API
// int GetPointOfIntersection(const float *f,
                              const float a, const float b,
                              int q1, int q2, float *point);
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
                           int q1, int q2, float *point)
{
    if (q1 == q2)
    {
        return DISTANCE_TRANSFORM_EQUAL_POINTS;
    } /* if (q1 == q2) */
    (*point) = ( (f[q2] - a * q2 + b *q2 * q2) -
                 (f[q1] - a * q1 + b * q1 * q1) ) / (2 * b * (q2 - q1));
    return DISTANCE_TRANSFORM_OK;
}

/*
// Decision of one dimensional problem generalized distance transform
// on the regular grid at all points
//      min (a(y' - y) + b(y' - y)(y' - y) + f(y')) (on y')
//
// API
// int DistanceTransformOneDimensionalProblem(const float *f, const int n,
                                              const float a, const float b,
                                              float *distanceTransform,
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
                                           int *points)
{
    int i, k;
    int tmp;
    int diff;
    float pointIntersection;
    int *v;
    float *z;
    k = 0;

    // Allocation memory (must be free in this function)
    v = (int *)malloc (sizeof(int) * n);
    z = (float *)malloc (sizeof(float) * (n + 1));

    v[0] = 0;
    z[0] = (float)F_MIN; // left border of envelope
    z[1] = (float)F_MAX; // right border of envelope

    for (i = 1; i < n; i++)
    {
        tmp = GetPointOfIntersection(f, a, b, v[k], i, &pointIntersection);
        if (tmp != DISTANCE_TRANSFORM_OK)
        {
            free(v);
            free(z);
            return DISTANCE_TRANSFORM_GET_INTERSECTION_ERROR;
        } /* if (tmp != DISTANCE_TRANSFORM_OK) */
        if (pointIntersection <= z[k])
        {
            // Envelope doesn't contain current parabola
            do
            {
                k--;
                tmp = GetPointOfIntersection(f, a, b, v[k], i, &pointIntersection);
                if (tmp != DISTANCE_TRANSFORM_OK)
                {
                    free(v);
                    free(z);
                    return DISTANCE_TRANSFORM_GET_INTERSECTION_ERROR;
                } /* if (tmp != DISTANCE_TRANSFORM_OK) */
            }while (pointIntersection <= z[k]);
            // Addition parabola to the envelope
            k++;
            v[k] = i;
            z[k] = pointIntersection;
            z[k + 1] = (float)F_MAX;
        }
        else
        {
            // Addition parabola to the envelope
            k++;
            v[k] = i;
            z[k] = pointIntersection;
            z[k + 1] = (float)F_MAX;
        } /* if (pointIntersection <= z[k]) */
    }

    // Computation values of generalized distance transform at all grid points
    k = 0;
    for (i = 0; i < n; i++)
    {
        while (z[k + 1] < i)
        {
            k++;
        }
        points[i] = v[k];
        diff = i - v[k];
        distanceTransform[i] = a * diff + b * diff * diff + f[v[k]];
    }

    // Release allocated memory
    free(v);
    free(z);
    return DISTANCE_TRANSFORM_OK;
}

/*
// Computation next cycle element
//
// API
// int GetNextCycleElement(int k, int n, int q);
// INPUT
// k                 - index of the previous cycle element
// n                 - number of matrix rows
// q                 - parameter that equal
                       (number_of_rows * number_of_columns - 1)
// OUTPUT
// None
// RESULT
// Next cycle element
*/
int GetNextCycleElement(int k, int n, int q)
{
    return ((k * n) % q);
}

/*
// Transpose cycle elements
//
// API
// void TransposeCycleElements(float *a, int *cycle, int cycle_len)
// INPUT
// a                 - initial matrix
// cycle             - indeces array of cycle
// cycle_len         - number of elements in the cycle
// OUTPUT
// a                 - matrix with transposed elements
// RESULT
// Error status
*/
void TransposeCycleElements(float *a, int *cycle, int cycle_len)
{
    int i;
    float buf;
    for (i = cycle_len - 1; i > 0 ; i--)
    {
        buf = a[ cycle[i] ];
        a[ cycle[i] ] = a[ cycle[i - 1] ];
        a[ cycle[i - 1] ] = buf;
    }
}

/*
// Transpose cycle elements
//
// API
// void TransposeCycleElements(int *a, int *cycle, int cycle_len)
// INPUT
// a                 - initial matrix
// cycle             - indeces array of cycle
// cycle_len         - number of elements in the cycle
// OUTPUT
// a                 - matrix with transposed elements
// RESULT
// Error status
*/
static void TransposeCycleElements_int(int *a, int *cycle, int cycle_len)
{
    int i;
    int buf;
    for (i = cycle_len - 1; i > 0 ; i--)
    {
        buf = a[ cycle[i] ];
        a[ cycle[i] ] = a[ cycle[i - 1] ];
        a[ cycle[i - 1] ] = buf;
    }
}

/*
// Getting transposed matrix
//
// API
// void Transpose(float *a, int n, int m);
// INPUT
// a                 - initial matrix
// n                 - number of rows
// m                 - number of columns
// OUTPUT
// a                 - transposed matrix
// RESULT
// None
*/
void Transpose(float *a, int n, int m)
{
    int *cycle;
    int i, k, q, cycle_len;
    int max_cycle_len;

    max_cycle_len = n * m;

    // Allocation memory  (must be free in this function)
    cycle = (int *)malloc(sizeof(int) * max_cycle_len);

    cycle_len = 0;
    q = n * m - 1;
    for (i = 1; i < q; i++)
    {
        k = GetNextCycleElement(i, n, q);
        cycle[cycle_len] = i;
        cycle_len++;

        while (k > i)
        {
            cycle[cycle_len] = k;
            cycle_len++;
            k = GetNextCycleElement(k, n, q);
        }
        if (k == i)
        {
            TransposeCycleElements(a, cycle, cycle_len);
        } /* if (k == i) */
        cycle_len = 0;
    }

    // Release allocated memory
    free(cycle);
}

/*
// Getting transposed matrix
//
// API
// void Transpose_int(int *a, int n, int m);
// INPUT
// a                 - initial matrix
// n                 - number of rows
// m                 - number of columns
// OUTPUT
// a                 - transposed matrix
// RESULT
// None
*/
static void Transpose_int(int *a, int n, int m)
{
    int *cycle;
    int i, k, q, cycle_len;
    int max_cycle_len;

    max_cycle_len = n * m;

    // Allocation memory  (must be free in this function)
    cycle = (int *)malloc(sizeof(int) * max_cycle_len);

    cycle_len = 0;
    q = n * m - 1;
    for (i = 1; i < q; i++)
    {
        k = GetNextCycleElement(i, n, q);
        cycle[cycle_len] = i;
        cycle_len++;

        while (k > i)
        {
            cycle[cycle_len] = k;
            cycle_len++;
            k = GetNextCycleElement(k, n, q);
        }
        if (k == i)
        {
            TransposeCycleElements_int(a, cycle, cycle_len);
        } /* if (k == i) */
        cycle_len = 0;
    }

    // Release allocated memory
    free(cycle);
}

/*
// Decision of two dimensional problem generalized distance transform
// on the regular grid at all points
//      min{d2(y' - y) + d4(y' - y)(y' - y) +
            min(d1(x' - x) + d3(x' - x)(x' - x) + f(x',y'))} (on x', y')
//
// API
// int DistanceTransformTwoDimensionalProblem(const float *f,
                                              const int n, const int m,
                                              const float coeff[4],
                                              float *distanceTransform,
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
                                           int *pointsX, int *pointsY)
{
    int i, j, tmp;
    int resOneDimProblem;
    int size = n * m;
    std::vector<float> internalDistTrans(size);
    std::vector<int> internalPointsX(size);

    for (i = 0; i < n; i++)
    {
        resOneDimProblem = DistanceTransformOneDimensionalProblem(
                                    f + i * m, m,
                                    coeff[0], coeff[2],
                                    &internalDistTrans[i * m],
                                    &internalPointsX[i * m]);
        if (resOneDimProblem != DISTANCE_TRANSFORM_OK)
            return DISTANCE_TRANSFORM_ERROR;
    }
    Transpose(&internalDistTrans[0], n, m);
    for (j = 0; j < m; j++)
    {
        resOneDimProblem = DistanceTransformOneDimensionalProblem(
                                    &internalDistTrans[j * n], n,
                                    coeff[1], coeff[3],
                                    distanceTransform + j * n,
                                    pointsY + j * n);
        if (resOneDimProblem != DISTANCE_TRANSFORM_OK)
            return DISTANCE_TRANSFORM_ERROR;
    }
    Transpose(distanceTransform, m, n);
    Transpose_int(pointsY, m, n);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < m; j++)
        {
            tmp = pointsY[i * m + j];
            pointsX[i * m + j] = internalPointsX[tmp * m + j];
        }
    }

    return DISTANCE_TRANSFORM_OK;
}
