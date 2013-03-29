/*! \file core.hpp
    \brief The Core Functionality
 */
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_CORE_HPP__
#define __OPENCV_CORE_HPP__

#ifndef __cplusplus
#  error core.hpp header must be compiled as C++
#endif

#include "opencv2/core/cvdef.h"
#include "opencv2/core/version.hpp"

// Used by FileStorage
typedef struct CvFileStorage CvFileStorage;
struct CvFileNode;
struct CvSeq;
struct CvSeqBlock;

#include "opencv2/core/base.hpp"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/traits.hpp"
#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/core/mat.hpp"

#ifndef SKIP_INCLUDES
#include <limits.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <complex>
#include <map>
#include <new>
#include <vector>
#include <sstream>
#endif // SKIP_INCLUDES

/*! \namespace cv
    Namespace where all the C++ OpenCV functionality resides
*/
namespace cv {

class CV_EXPORTS MatOp_Base;
class CV_EXPORTS MatArg;

template<typename _Tp> class CV_EXPORTS MatCommaInitializer_;

/*!
 The standard OpenCV exception class.
 Instances of the class are thrown by various functions and methods in the case of critical errors.
 */
class CV_EXPORTS Exception : public std::exception
{
public:
    /*!
     Default constructor
     */
    Exception();
    /*!
     Full constructor. Normally the constuctor is not called explicitly.
     Instead, the macros CV_Error(), CV_Error_() and CV_Assert() are used.
    */
    Exception(int _code, const String& _err, const String& _func, const String& _file, int _line);
    virtual ~Exception() throw();

    /*!
     \return the error description and the context as a text string.
    */
    virtual const char *what() const throw();
    void formatMessage();

    String msg; ///< the formatted error message

    int code; ///< error code @see CVStatus
    String err; ///< error description
    String func; ///< function name. Available only when the compiler supports __func__ macro
    String file; ///< source file name where the error has occured
    int line; ///< line number in the source file where the error has occured
};


//! Signals an error and raises the exception.

/*!
  By default the function prints information about the error to stderr,
  then it either stops if setBreakOnError() had been called before or raises the exception.
  It is possible to alternate error processing by using redirectError().

  \param exc the exception raisen.
 */
//TODO: drop this version
CV_EXPORTS void error( const Exception& exc );


CV_EXPORTS void scalarToRawData(const Scalar& s, void* buf, int type, int unroll_to=0);


/*!
   Random Number Generator

   The class implements RNG using Multiply-with-Carry algorithm
*/
class CV_EXPORTS RNG
{
public:
    enum { UNIFORM=0, NORMAL=1 };

    RNG();
    RNG(uint64 state);
    //! updates the state and returns the next 32-bit unsigned integer random number
    unsigned next();

    operator uchar();
    operator schar();
    operator ushort();
    operator short();
    operator unsigned();
    //! returns a random integer sampled uniformly from [0, N).
    unsigned operator ()(unsigned N);
    unsigned operator ()();
    operator int();
    operator float();
    operator double();
    //! returns uniformly distributed integer random number from [a,b) range
    int uniform(int a, int b);
    //! returns uniformly distributed floating-point random number from [a,b) range
    float uniform(float a, float b);
    //! returns uniformly distributed double-precision floating-point random number from [a,b) range
    double uniform(double a, double b);
    void fill( InputOutputArray mat, int distType, InputArray a, InputArray b, bool saturateRange=false );
    //! returns Gaussian random variate with mean zero.
    double gaussian(double sigma);

    uint64 state;
};

class CV_EXPORTS RNG_MT19937
{
public:
    RNG_MT19937();
    RNG_MT19937(unsigned s);
    void seed(unsigned s);

    unsigned next();

    operator int();
    operator unsigned();
    operator float();
    operator double();

    unsigned operator ()(unsigned N);
    unsigned operator ()();

    // returns uniformly distributed integer random number from [a,b) range
    int uniform(int a, int b);
    // returns uniformly distributed floating-point random number from [a,b) range
    float uniform(float a, float b);
    // returns uniformly distributed double-precision floating-point random number from [a,b) range
    double uniform(double a, double b);

private:
    enum PeriodParameters {N = 624, M = 397};
    unsigned state[N];
    int mti;
};

typedef void (*BinaryFunc)(const uchar* src1, size_t step1,
                           const uchar* src2, size_t step2,
                           uchar* dst, size_t step, Size sz,
                           void*);

CV_EXPORTS BinaryFunc getConvertFunc(int sdepth, int ddepth);
CV_EXPORTS BinaryFunc getConvertScaleFunc(int sdepth, int ddepth);
CV_EXPORTS BinaryFunc getCopyMaskFunc(size_t esz);

//! swaps two matrices
CV_EXPORTS void swap(Mat& a, Mat& b);

//! adds one matrix to another (dst = src1 + src2)
CV_EXPORTS_W void add(InputArray src1, InputArray src2, OutputArray dst,
                      InputArray mask=noArray(), int dtype=-1);
//! subtracts one matrix from another (dst = src1 - src2)
CV_EXPORTS_W void subtract(InputArray src1, InputArray src2, OutputArray dst,
                           InputArray mask=noArray(), int dtype=-1);

//! computes element-wise weighted product of the two arrays (dst = scale*src1*src2)
CV_EXPORTS_W void multiply(InputArray src1, InputArray src2,
                           OutputArray dst, double scale=1, int dtype=-1);

//! computes element-wise weighted quotient of the two arrays (dst = scale*src1/src2)
CV_EXPORTS_W void divide(InputArray src1, InputArray src2, OutputArray dst,
                         double scale=1, int dtype=-1);

//! computes element-wise weighted reciprocal of an array (dst = scale/src2)
CV_EXPORTS_W void divide(double scale, InputArray src2,
                         OutputArray dst, int dtype=-1);

//! adds scaled array to another one (dst = alpha*src1 + src2)
CV_EXPORTS_W void scaleAdd(InputArray src1, double alpha, InputArray src2, OutputArray dst);

//! computes weighted sum of two arrays (dst = alpha*src1 + beta*src2 + gamma)
CV_EXPORTS_W void addWeighted(InputArray src1, double alpha, InputArray src2,
                              double beta, double gamma, OutputArray dst, int dtype=-1);

//! scales array elements, computes absolute values and converts the results to 8-bit unsigned integers: dst(i)=saturate_cast<uchar>abs(src(i)*alpha+beta)
CV_EXPORTS_W void convertScaleAbs(InputArray src, OutputArray dst,
                                  double alpha=1, double beta=0);
//! transforms array of numbers using a lookup table: dst(i)=lut(src(i))
CV_EXPORTS_W void LUT(InputArray src, InputArray lut, OutputArray dst,
                      int interpolation=0);

//! computes sum of array elements
CV_EXPORTS_AS(sumElems) Scalar sum(InputArray src);
//! computes the number of nonzero array elements
CV_EXPORTS_W int countNonZero( InputArray src );
//! returns the list of locations of non-zero pixels
CV_EXPORTS_W void findNonZero( InputArray src, OutputArray idx );

//! computes mean value of selected array elements
CV_EXPORTS_W Scalar mean(InputArray src, InputArray mask=noArray());
//! computes mean value and standard deviation of all or selected array elements
CV_EXPORTS_W void meanStdDev(InputArray src, OutputArray mean, OutputArray stddev,
                             InputArray mask=noArray());
//! computes norm of the selected array part
CV_EXPORTS_W double norm(InputArray src1, int normType=NORM_L2, InputArray mask=noArray());
//! computes norm of selected part of the difference between two arrays
CV_EXPORTS_W double norm(InputArray src1, InputArray src2,
                         int normType=NORM_L2, InputArray mask=noArray());

//! naive nearest neighbor finder
CV_EXPORTS_W void batchDistance(InputArray src1, InputArray src2,
                                OutputArray dist, int dtype, OutputArray nidx,
                                int normType=NORM_L2, int K=0,
                                InputArray mask=noArray(), int update=0,
                                bool crosscheck=false);

//! scales and shifts array elements so that either the specified norm (alpha) or the minimum (alpha) and maximum (beta) array values get the specified values
CV_EXPORTS_W void normalize( InputArray src, OutputArray dst, double alpha=1, double beta=0,
                             int norm_type=NORM_L2, int dtype=-1, InputArray mask=noArray());

//! finds global minimum and maximum array elements and returns their values and their locations
CV_EXPORTS_W void minMaxLoc(InputArray src, CV_OUT double* minVal,
                           CV_OUT double* maxVal=0, CV_OUT Point* minLoc=0,
                           CV_OUT Point* maxLoc=0, InputArray mask=noArray());
CV_EXPORTS void minMaxIdx(InputArray src, double* minVal, double* maxVal,
                          int* minIdx=0, int* maxIdx=0, InputArray mask=noArray());

//! transforms 2D matrix to 1D row or column vector by taking sum, minimum, maximum or mean value over all the rows
CV_EXPORTS_W void reduce(InputArray src, OutputArray dst, int dim, int rtype, int dtype=-1);

//! makes multi-channel array out of several single-channel arrays
CV_EXPORTS void merge(const Mat* mv, size_t count, OutputArray dst);
//! makes multi-channel array out of several single-channel arrays
CV_EXPORTS_W void merge(InputArrayOfArrays mv, OutputArray dst);

//! copies each plane of a multi-channel array to a dedicated array
CV_EXPORTS void split(const Mat& src, Mat* mvbegin);
//! copies each plane of a multi-channel array to a dedicated array
CV_EXPORTS_W void split(InputArray m, OutputArrayOfArrays mv);

//! copies selected channels from the input arrays to the selected channels of the output arrays
CV_EXPORTS void mixChannels(const Mat* src, size_t nsrcs, Mat* dst, size_t ndsts,
                            const int* fromTo, size_t npairs);
CV_EXPORTS void mixChannels(const std::vector<Mat>& src, std::vector<Mat>& dst,
                            const int* fromTo, size_t npairs);
CV_EXPORTS_W void mixChannels(InputArrayOfArrays src, InputArrayOfArrays dst,
                              const std::vector<int>& fromTo);

//! extracts a single channel from src (coi is 0-based index)
CV_EXPORTS_W void extractChannel(InputArray src, OutputArray dst, int coi);

//! inserts a single channel to dst (coi is 0-based index)
CV_EXPORTS_W void insertChannel(InputArray src, InputOutputArray dst, int coi);

//! reverses the order of the rows, columns or both in a matrix
CV_EXPORTS_W void flip(InputArray src, OutputArray dst, int flipCode);

//! replicates the input matrix the specified number of times in the horizontal and/or vertical direction
CV_EXPORTS_W void repeat(InputArray src, int ny, int nx, OutputArray dst);
CV_EXPORTS Mat repeat(const Mat& src, int ny, int nx);

CV_EXPORTS void hconcat(const Mat* src, size_t nsrc, OutputArray dst);
CV_EXPORTS void hconcat(InputArray src1, InputArray src2, OutputArray dst);
CV_EXPORTS_W void hconcat(InputArrayOfArrays src, OutputArray dst);

CV_EXPORTS void vconcat(const Mat* src, size_t nsrc, OutputArray dst);
CV_EXPORTS void vconcat(InputArray src1, InputArray src2, OutputArray dst);
CV_EXPORTS_W void vconcat(InputArrayOfArrays src, OutputArray dst);

//! computes bitwise conjunction of the two arrays (dst = src1 & src2)
CV_EXPORTS_W void bitwise_and(InputArray src1, InputArray src2,
                              OutputArray dst, InputArray mask=noArray());
//! computes bitwise disjunction of the two arrays (dst = src1 | src2)
CV_EXPORTS_W void bitwise_or(InputArray src1, InputArray src2,
                             OutputArray dst, InputArray mask=noArray());
//! computes bitwise exclusive-or of the two arrays (dst = src1 ^ src2)
CV_EXPORTS_W void bitwise_xor(InputArray src1, InputArray src2,
                              OutputArray dst, InputArray mask=noArray());
//! inverts each bit of array (dst = ~src)
CV_EXPORTS_W void bitwise_not(InputArray src, OutputArray dst,
                              InputArray mask=noArray());
//! computes element-wise absolute difference of two arrays (dst = abs(src1 - src2))
CV_EXPORTS_W void absdiff(InputArray src1, InputArray src2, OutputArray dst);
//! set mask elements for those array elements which are within the element-specific bounding box (dst = lowerb <= src && src < upperb)
CV_EXPORTS_W void inRange(InputArray src, InputArray lowerb,
                          InputArray upperb, OutputArray dst);
//! compares elements of two arrays (dst = src1 <cmpop> src2)
CV_EXPORTS_W void compare(InputArray src1, InputArray src2, OutputArray dst, int cmpop);
//! computes per-element minimum of two arrays (dst = min(src1, src2))
CV_EXPORTS_W void min(InputArray src1, InputArray src2, OutputArray dst);
//! computes per-element maximum of two arrays (dst = max(src1, src2))
CV_EXPORTS_W void max(InputArray src1, InputArray src2, OutputArray dst);

//! computes per-element minimum of two arrays (dst = min(src1, src2))
CV_EXPORTS void min(const Mat& src1, const Mat& src2, Mat& dst);
//! computes per-element minimum of array and scalar (dst = min(src1, src2))
CV_EXPORTS void min(const Mat& src1, double src2, Mat& dst);
//! computes per-element maximum of two arrays (dst = max(src1, src2))
CV_EXPORTS void max(const Mat& src1, const Mat& src2, Mat& dst);
//! computes per-element maximum of array and scalar (dst = max(src1, src2))
CV_EXPORTS void max(const Mat& src1, double src2, Mat& dst);

//! computes square root of each matrix element (dst = src**0.5)
CV_EXPORTS_W void sqrt(InputArray src, OutputArray dst);
//! raises the input matrix elements to the specified power (b = a**power)
CV_EXPORTS_W void pow(InputArray src, double power, OutputArray dst);
//! computes exponent of each matrix element (dst = e**src)
CV_EXPORTS_W void exp(InputArray src, OutputArray dst);
//! computes natural logarithm of absolute value of each matrix element: dst = log(abs(src))
CV_EXPORTS_W void log(InputArray src, OutputArray dst);
//! computes cube root of the argument
CV_EXPORTS_W float cubeRoot(float val);
//! computes the angle in degrees (0..360) of the vector (x,y)
CV_EXPORTS_W float fastAtan2(float y, float x);

CV_EXPORTS void exp(const float* src, float* dst, int n);
CV_EXPORTS void log(const float* src, float* dst, int n);
CV_EXPORTS void fastAtan2(const float* y, const float* x, float* dst, int n, bool angleInDegrees);
CV_EXPORTS void magnitude(const float* x, const float* y, float* dst, int n);

//! converts polar coordinates to Cartesian
CV_EXPORTS_W void polarToCart(InputArray magnitude, InputArray angle,
                              OutputArray x, OutputArray y, bool angleInDegrees=false);
//! converts Cartesian coordinates to polar
CV_EXPORTS_W void cartToPolar(InputArray x, InputArray y,
                              OutputArray magnitude, OutputArray angle,
                              bool angleInDegrees=false);
//! computes angle (angle(i)) of each (x(i), y(i)) vector
CV_EXPORTS_W void phase(InputArray x, InputArray y, OutputArray angle,
                        bool angleInDegrees=false);
//! computes magnitude (magnitude(i)) of each (x(i), y(i)) vector
CV_EXPORTS_W void magnitude(InputArray x, InputArray y, OutputArray magnitude);
//! checks that each matrix element is within the specified range.
CV_EXPORTS_W bool checkRange(InputArray a, bool quiet=true, CV_OUT Point* pos=0,
                            double minVal=-DBL_MAX, double maxVal=DBL_MAX);
//! converts NaN's to the given number
CV_EXPORTS_W void patchNaNs(InputOutputArray a, double val=0);

//! implements generalized matrix product algorithm GEMM from BLAS
CV_EXPORTS_W void gemm(InputArray src1, InputArray src2, double alpha,
                       InputArray src3, double gamma, OutputArray dst, int flags=0);
//! multiplies matrix by its transposition from the left or from the right
CV_EXPORTS_W void mulTransposed( InputArray src, OutputArray dst, bool aTa,
                                 InputArray delta=noArray(),
                                 double scale=1, int dtype=-1 );
//! transposes the matrix
CV_EXPORTS_W void transpose(InputArray src, OutputArray dst);
//! performs affine transformation of each element of multi-channel input matrix
CV_EXPORTS_W void transform(InputArray src, OutputArray dst, InputArray m );
//! performs perspective transformation of each element of multi-channel input matrix
CV_EXPORTS_W void perspectiveTransform(InputArray src, OutputArray dst, InputArray m );

//! extends the symmetrical matrix from the lower half or from the upper half
CV_EXPORTS_W void completeSymm(InputOutputArray mtx, bool lowerToUpper=false);
//! initializes scaled identity matrix
CV_EXPORTS_W void setIdentity(InputOutputArray mtx, const Scalar& s=Scalar(1));
//! computes determinant of a square matrix
CV_EXPORTS_W double determinant(InputArray mtx);
//! computes trace of a matrix
CV_EXPORTS_W Scalar trace(InputArray mtx);
//! computes inverse or pseudo-inverse matrix
CV_EXPORTS_W double invert(InputArray src, OutputArray dst, int flags=DECOMP_LU);
//! solves linear system or a least-square problem
CV_EXPORTS_W bool solve(InputArray src1, InputArray src2,
                        OutputArray dst, int flags=DECOMP_LU);

enum
{
    SORT_EVERY_ROW=0,
    SORT_EVERY_COLUMN=1,
    SORT_ASCENDING=0,
    SORT_DESCENDING=16
};

//! sorts independently each matrix row or each matrix column
CV_EXPORTS_W void sort(InputArray src, OutputArray dst, int flags);
//! sorts independently each matrix row or each matrix column
CV_EXPORTS_W void sortIdx(InputArray src, OutputArray dst, int flags);
//! finds real roots of a cubic polynomial
CV_EXPORTS_W int solveCubic(InputArray coeffs, OutputArray roots);
//! finds real and complex roots of a polynomial
CV_EXPORTS_W double solvePoly(InputArray coeffs, OutputArray roots, int maxIters=300);
//! finds eigenvalues of a symmetric matrix
CV_EXPORTS bool eigen(InputArray src, OutputArray eigenvalues, int lowindex=-1,
                      int highindex=-1);
//! finds eigenvalues and eigenvectors of a symmetric matrix
CV_EXPORTS bool eigen(InputArray src, OutputArray eigenvalues,
                      OutputArray eigenvectors,
                      int lowindex=-1, int highindex=-1);
CV_EXPORTS_W bool eigen(InputArray src, bool computeEigenvectors,
                        OutputArray eigenvalues, OutputArray eigenvectors);

enum
{
    COVAR_SCRAMBLED=0,
    COVAR_NORMAL=1,
    COVAR_USE_AVG=2,
    COVAR_SCALE=4,
    COVAR_ROWS=8,
    COVAR_COLS=16
};

//! computes covariation matrix of a set of samples
CV_EXPORTS void calcCovarMatrix( const Mat* samples, int nsamples, Mat& covar, Mat& mean,
                                 int flags, int ctype=CV_64F);
//! computes covariation matrix of a set of samples
CV_EXPORTS_W void calcCovarMatrix( InputArray samples, OutputArray covar,
                                   OutputArray mean, int flags, int ctype=CV_64F);

/*!
    Principal Component Analysis

    The class PCA is used to compute the special basis for a set of vectors.
    The basis will consist of eigenvectors of the covariance matrix computed
    from the input set of vectors. After PCA is performed, vectors can be transformed from
    the original high-dimensional space to the subspace formed by a few most
    prominent eigenvectors (called the principal components),
    corresponding to the largest eigenvalues of the covariation matrix.
    Thus the dimensionality of the vector and the correlation between the coordinates is reduced.

    The following sample is the function that takes two matrices. The first one stores the set
    of vectors (a row per vector) that is used to compute PCA, the second one stores another
    "test" set of vectors (a row per vector) that are first compressed with PCA,
    then reconstructed back and then the reconstruction error norm is computed and printed for each vector.

    \code
    using namespace cv;

    PCA compressPCA(const Mat& pcaset, int maxComponents,
                    const Mat& testset, Mat& compressed)
    {
        PCA pca(pcaset, // pass the data
                Mat(), // we do not have a pre-computed mean vector,
                       // so let the PCA engine to compute it
                CV_PCA_DATA_AS_ROW, // indicate that the vectors
                                    // are stored as matrix rows
                                    // (use CV_PCA_DATA_AS_COL if the vectors are
                                    // the matrix columns)
                maxComponents // specify, how many principal components to retain
                );
        // if there is no test data, just return the computed basis, ready-to-use
        if( !testset.data )
            return pca;
        CV_Assert( testset.cols == pcaset.cols );

        compressed.create(testset.rows, maxComponents, testset.type());

        Mat reconstructed;
        for( int i = 0; i < testset.rows; i++ )
        {
            Mat vec = testset.row(i), coeffs = compressed.row(i), reconstructed;
            // compress the vector, the result will be stored
            // in the i-th row of the output matrix
            pca.project(vec, coeffs);
            // and then reconstruct it
            pca.backProject(coeffs, reconstructed);
            // and measure the error
            printf("%d. diff = %g\n", i, norm(vec, reconstructed, NORM_L2));
        }
        return pca;
    }
    \endcode
*/
class CV_EXPORTS PCA
{
public:
    //! default constructor
    PCA();
    //! the constructor that performs PCA
    PCA(InputArray data, InputArray mean, int flags, int maxComponents=0);
    PCA(InputArray data, InputArray mean, int flags, double retainedVariance);
    //! operator that performs PCA. The previously stored data, if any, is released
    PCA& operator()(InputArray data, InputArray mean, int flags, int maxComponents=0);
    PCA& operator()(InputArray data, InputArray mean, int flags, double retainedVariance);
    //! projects vector from the original space to the principal components subspace
    Mat project(InputArray vec) const;
    //! projects vector from the original space to the principal components subspace
    void project(InputArray vec, OutputArray result) const;
    //! reconstructs the original vector from the projection
    Mat backProject(InputArray vec) const;
    //! reconstructs the original vector from the projection
    void backProject(InputArray vec, OutputArray result) const;

    Mat eigenvectors; //!< eigenvectors of the covariation matrix
    Mat eigenvalues; //!< eigenvalues of the covariation matrix
    Mat mean; //!< mean value subtracted before the projection and added after the back projection
};

CV_EXPORTS_W void PCACompute(InputArray data, InputOutputArray mean,
                             OutputArray eigenvectors, int maxComponents=0);

CV_EXPORTS_W void PCACompute(InputArray data, InputOutputArray mean,
                             OutputArray eigenvectors, double retainedVariance);

CV_EXPORTS_W void PCAProject(InputArray data, InputArray mean,
                             InputArray eigenvectors, OutputArray result);

CV_EXPORTS_W void PCABackProject(InputArray data, InputArray mean,
                                 InputArray eigenvectors, OutputArray result);


/*!
    Singular Value Decomposition class

    The class is used to compute Singular Value Decomposition of a floating-point matrix and then
    use it to solve least-square problems, under-determined linear systems, invert matrices,
    compute condition numbers etc.

    For a bit faster operation you can pass flags=SVD::MODIFY_A|... to modify the decomposed matrix
    when it is not necessarily to preserve it. If you want to compute condition number of a matrix
    or absolute value of its determinant - you do not need SVD::u or SVD::vt,
    so you can pass flags=SVD::NO_UV|... . Another flag SVD::FULL_UV indicates that the full-size SVD::u and SVD::vt
    must be computed, which is not necessary most of the time.
*/
class CV_EXPORTS SVD
{
public:
    enum { MODIFY_A=1, NO_UV=2, FULL_UV=4 };
    //! the default constructor
    SVD();
    //! the constructor that performs SVD
    SVD( InputArray src, int flags=0 );
    //! the operator that performs SVD. The previously allocated SVD::u, SVD::w are SVD::vt are released.
    SVD& operator ()( InputArray src, int flags=0 );

    //! decomposes matrix and stores the results to user-provided matrices
    static void compute( InputArray src, OutputArray w,
                         OutputArray u, OutputArray vt, int flags=0 );
    //! computes singular values of a matrix
    static void compute( InputArray src, OutputArray w, int flags=0 );
    //! performs back substitution
    static void backSubst( InputArray w, InputArray u,
                           InputArray vt, InputArray rhs,
                           OutputArray dst );

    template<typename _Tp, int m, int n, int nm> static void compute( const Matx<_Tp, m, n>& a,
        Matx<_Tp, nm, 1>& w, Matx<_Tp, m, nm>& u, Matx<_Tp, n, nm>& vt );
    template<typename _Tp, int m, int n, int nm> static void compute( const Matx<_Tp, m, n>& a,
        Matx<_Tp, nm, 1>& w );
    template<typename _Tp, int m, int n, int nm, int nb> static void backSubst( const Matx<_Tp, nm, 1>& w,
        const Matx<_Tp, m, nm>& u, const Matx<_Tp, n, nm>& vt, const Matx<_Tp, m, nb>& rhs, Matx<_Tp, n, nb>& dst );

    //! finds dst = arg min_{|dst|=1} |m*dst|
    static void solveZ( InputArray src, OutputArray dst );
    //! performs back substitution, so that dst is the solution or pseudo-solution of m*dst = rhs, where m is the decomposed matrix
    void backSubst( InputArray rhs, OutputArray dst ) const;

    Mat u, w, vt;
};

//! computes SVD of src
CV_EXPORTS_W void SVDecomp( InputArray src, OutputArray w, OutputArray u, OutputArray vt, int flags=0 );

//! performs back substitution for the previously computed SVD
CV_EXPORTS_W void SVBackSubst( InputArray w, InputArray u, InputArray vt,
                               InputArray rhs, OutputArray dst );

//! computes Mahalanobis distance between two vectors: sqrt((v1-v2)'*icovar*(v1-v2)), where icovar is the inverse covariation matrix
CV_EXPORTS_W double Mahalanobis(InputArray v1, InputArray v2, InputArray icovar);
//! a synonym for Mahalanobis
CV_EXPORTS double Mahalonobis(InputArray v1, InputArray v2, InputArray icovar);

//! performs forward or inverse 1D or 2D Discrete Fourier Transformation
CV_EXPORTS_W void dft(InputArray src, OutputArray dst, int flags=0, int nonzeroRows=0);
//! performs inverse 1D or 2D Discrete Fourier Transformation
CV_EXPORTS_W void idft(InputArray src, OutputArray dst, int flags=0, int nonzeroRows=0);
//! performs forward or inverse 1D or 2D Discrete Cosine Transformation
CV_EXPORTS_W void dct(InputArray src, OutputArray dst, int flags=0);
//! performs inverse 1D or 2D Discrete Cosine Transformation
CV_EXPORTS_W void idct(InputArray src, OutputArray dst, int flags=0);
//! computes element-wise product of the two Fourier spectrums. The second spectrum can optionally be conjugated before the multiplication
CV_EXPORTS_W void mulSpectrums(InputArray a, InputArray b, OutputArray c,
                               int flags, bool conjB=false);
//! computes the minimal vector size vecsize1 >= vecsize so that the dft() of the vector of length vecsize1 can be computed efficiently
CV_EXPORTS_W int getOptimalDFTSize(int vecsize);

/*!
 Various k-Means flags
*/
enum
{
    KMEANS_RANDOM_CENTERS=0, // Chooses random centers for k-Means initialization
    KMEANS_PP_CENTERS=2,     // Uses k-Means++ algorithm for initialization
    KMEANS_USE_INITIAL_LABELS=1 // Uses the user-provided labels for K-Means initialization
};
//! clusters the input data using k-Means algorithm
CV_EXPORTS_W double kmeans( InputArray data, int K, InputOutputArray bestLabels,
                            TermCriteria criteria, int attempts,
                            int flags, OutputArray centers=noArray() );

//! returns the thread-local Random number generator
CV_EXPORTS RNG& theRNG();

//! returns the next unifomly-distributed random number of the specified type
template<typename _Tp> static inline _Tp randu() { return (_Tp)theRNG(); }

//! fills array with uniformly-distributed random numbers from the range [low, high)
CV_EXPORTS_W void randu(InputOutputArray dst, InputArray low, InputArray high);

//! fills array with normally-distributed random numbers with the specified mean and the standard deviation
CV_EXPORTS_W void randn(InputOutputArray dst, InputArray mean, InputArray stddev);

//! shuffles the input array elements
CV_EXPORTS void randShuffle(InputOutputArray dst, double iterFactor=1., RNG* rng=0);
CV_EXPORTS_AS(randShuffle) void randShuffle_(InputOutputArray dst, double iterFactor=1.);

//! draws the line segment (pt1, pt2) in the image
CV_EXPORTS_W void line(CV_IN_OUT Mat& img, Point pt1, Point pt2, const Scalar& color,
                     int thickness=1, int lineType=8, int shift=0);

//! draws the rectangle outline or a solid rectangle with the opposite corners pt1 and pt2 in the image
CV_EXPORTS_W void rectangle(CV_IN_OUT Mat& img, Point pt1, Point pt2,
                          const Scalar& color, int thickness=1,
                          int lineType=8, int shift=0);

//! draws the rectangle outline or a solid rectangle covering rec in the image
CV_EXPORTS void rectangle(CV_IN_OUT Mat& img, Rect rec,
                          const Scalar& color, int thickness=1,
                          int lineType=8, int shift=0);

//! draws the circle outline or a solid circle in the image
CV_EXPORTS_W void circle(CV_IN_OUT Mat& img, Point center, int radius,
                       const Scalar& color, int thickness=1,
                       int lineType=8, int shift=0);

//! draws an elliptic arc, ellipse sector or a rotated ellipse in the image
CV_EXPORTS_W void ellipse(CV_IN_OUT Mat& img, Point center, Size axes,
                        double angle, double startAngle, double endAngle,
                        const Scalar& color, int thickness=1,
                        int lineType=8, int shift=0);

//! draws a rotated ellipse in the image
CV_EXPORTS_W void ellipse(CV_IN_OUT Mat& img, const RotatedRect& box, const Scalar& color,
                        int thickness=1, int lineType=8);

//! draws a filled convex polygon in the image
CV_EXPORTS void fillConvexPoly(Mat& img, const Point* pts, int npts,
                               const Scalar& color, int lineType=8,
                               int shift=0);
CV_EXPORTS_W void fillConvexPoly(InputOutputArray img, InputArray points,
                                 const Scalar& color, int lineType=8,
                                 int shift=0);

//! fills an area bounded by one or more polygons
CV_EXPORTS void fillPoly(Mat& img, const Point** pts,
                         const int* npts, int ncontours,
                         const Scalar& color, int lineType=8, int shift=0,
                         Point offset=Point() );

CV_EXPORTS_W void fillPoly(InputOutputArray img, InputArrayOfArrays pts,
                           const Scalar& color, int lineType=8, int shift=0,
                           Point offset=Point() );

//! draws one or more polygonal curves
CV_EXPORTS void polylines(Mat& img, const Point* const* pts, const int* npts,
                          int ncontours, bool isClosed, const Scalar& color,
                          int thickness=1, int lineType=8, int shift=0 );

CV_EXPORTS_W void polylines(InputOutputArray img, InputArrayOfArrays pts,
                            bool isClosed, const Scalar& color,
                            int thickness=1, int lineType=8, int shift=0 );

//! draws contours in the image
CV_EXPORTS_W void drawContours( InputOutputArray image, InputArrayOfArrays contours,
                              int contourIdx, const Scalar& color,
                              int thickness=1, int lineType=8,
                              InputArray hierarchy=noArray(),
                              int maxLevel=INT_MAX, Point offset=Point() );

//! clips the line segment by the rectangle Rect(0, 0, imgSize.width, imgSize.height)
CV_EXPORTS bool clipLine(Size imgSize, CV_IN_OUT Point& pt1, CV_IN_OUT Point& pt2);

//! clips the line segment by the rectangle imgRect
CV_EXPORTS_W bool clipLine(Rect imgRect, CV_OUT CV_IN_OUT Point& pt1, CV_OUT CV_IN_OUT Point& pt2);

/*!
   Line iterator class

   The class is used to iterate over all the pixels on the raster line
   segment connecting two specified points.
*/
class CV_EXPORTS LineIterator
{
public:
    //! intializes the iterator
    LineIterator( const Mat& img, Point pt1, Point pt2,
                  int connectivity=8, bool leftToRight=false );
    //! returns pointer to the current pixel
    uchar* operator *();
    //! prefix increment operator (++it). shifts iterator to the next pixel
    LineIterator& operator ++();
    //! postfix increment operator (it++). shifts iterator to the next pixel
    LineIterator operator ++(int);
    //! returns coordinates of the current pixel
    Point pos() const;

    uchar* ptr;
    const uchar* ptr0;
    int step, elemSize;
    int err, count;
    int minusDelta, plusDelta;
    int minusStep, plusStep;
};

//! converts elliptic arc to a polygonal curve
CV_EXPORTS_W void ellipse2Poly( Point center, Size axes, int angle,
                                int arcStart, int arcEnd, int delta,
                                CV_OUT std::vector<Point>& pts );

enum
{
    FONT_HERSHEY_SIMPLEX = 0,
    FONT_HERSHEY_PLAIN = 1,
    FONT_HERSHEY_DUPLEX = 2,
    FONT_HERSHEY_COMPLEX = 3,
    FONT_HERSHEY_TRIPLEX = 4,
    FONT_HERSHEY_COMPLEX_SMALL = 5,
    FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
    FONT_HERSHEY_SCRIPT_COMPLEX = 7,
    FONT_ITALIC = 16
};

//! renders text string in the image
CV_EXPORTS_W void putText( Mat& img, const String& text, Point org,
                         int fontFace, double fontScale, Scalar color,
                         int thickness=1, int lineType=8,
                         bool bottomLeftOrigin=false );

//! returns bounding box of the text string
CV_EXPORTS_W Size getTextSize(const String& text, int fontFace,
                            double fontScale, int thickness,
                            CV_OUT int* baseLine);

//////////// Iterators & Comma initializers //////////////////

class CV_EXPORTS MatConstIterator
{
public:
    typedef uchar* value_type;
    typedef ptrdiff_t difference_type;
    typedef const uchar** pointer;
    typedef uchar* reference;
    typedef std::random_access_iterator_tag iterator_category;

    //! default constructor
    MatConstIterator();
    //! constructor that sets the iterator to the beginning of the matrix
    MatConstIterator(const Mat* _m);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator(const Mat* _m, int _row, int _col=0);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator(const Mat* _m, Point _pt);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator(const Mat* _m, const int* _idx);
    //! copy constructor
    MatConstIterator(const MatConstIterator& it);

    //! copy operator
    MatConstIterator& operator = (const MatConstIterator& it);
    //! returns the current matrix element
    uchar* operator *() const;
    //! returns the i-th matrix element, relative to the current
    uchar* operator [](ptrdiff_t i) const;

    //! shifts the iterator forward by the specified number of elements
    MatConstIterator& operator += (ptrdiff_t ofs);
    //! shifts the iterator backward by the specified number of elements
    MatConstIterator& operator -= (ptrdiff_t ofs);
    //! decrements the iterator
    MatConstIterator& operator --();
    //! decrements the iterator
    MatConstIterator operator --(int);
    //! increments the iterator
    MatConstIterator& operator ++();
    //! increments the iterator
    MatConstIterator operator ++(int);
    //! returns the current iterator position
    Point pos() const;
    //! returns the current iterator position
    void pos(int* _idx) const;
    ptrdiff_t lpos() const;
    void seek(ptrdiff_t ofs, bool relative=false);
    void seek(const int* _idx, bool relative=false);

    const Mat* m;
    size_t elemSize;
    uchar* ptr;
    uchar* sliceStart;
    uchar* sliceEnd;
};

/*!
 Matrix read-only iterator

 */
template<typename _Tp>
class CV_EXPORTS MatConstIterator_ : public MatConstIterator
{
public:
    typedef _Tp value_type;
    typedef ptrdiff_t difference_type;
    typedef const _Tp* pointer;
    typedef const _Tp& reference;
    typedef std::random_access_iterator_tag iterator_category;

    //! default constructor
    MatConstIterator_();
    //! constructor that sets the iterator to the beginning of the matrix
    MatConstIterator_(const Mat_<_Tp>* _m);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator_(const Mat_<_Tp>* _m, int _row, int _col=0);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator_(const Mat_<_Tp>* _m, Point _pt);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator_(const Mat_<_Tp>* _m, const int* _idx);
    //! copy constructor
    MatConstIterator_(const MatConstIterator_& it);

    //! copy operator
    MatConstIterator_& operator = (const MatConstIterator_& it);
    //! returns the current matrix element
    _Tp operator *() const;
    //! returns the i-th matrix element, relative to the current
    _Tp operator [](ptrdiff_t i) const;

    //! shifts the iterator forward by the specified number of elements
    MatConstIterator_& operator += (ptrdiff_t ofs);
    //! shifts the iterator backward by the specified number of elements
    MatConstIterator_& operator -= (ptrdiff_t ofs);
    //! decrements the iterator
    MatConstIterator_& operator --();
    //! decrements the iterator
    MatConstIterator_ operator --(int);
    //! increments the iterator
    MatConstIterator_& operator ++();
    //! increments the iterator
    MatConstIterator_ operator ++(int);
    //! returns the current iterator position
    Point pos() const;
};


/*!
 Matrix read-write iterator

*/
template<typename _Tp>
class CV_EXPORTS MatIterator_ : public MatConstIterator_<_Tp>
{
public:
    typedef _Tp* pointer;
    typedef _Tp& reference;
    typedef std::random_access_iterator_tag iterator_category;

    //! the default constructor
    MatIterator_();
    //! constructor that sets the iterator to the beginning of the matrix
    MatIterator_(Mat_<_Tp>* _m);
    //! constructor that sets the iterator to the specified element of the matrix
    MatIterator_(Mat_<_Tp>* _m, int _row, int _col=0);
    //! constructor that sets the iterator to the specified element of the matrix
    MatIterator_(const Mat_<_Tp>* _m, Point _pt);
    //! constructor that sets the iterator to the specified element of the matrix
    MatIterator_(const Mat_<_Tp>* _m, const int* _idx);
    //! copy constructor
    MatIterator_(const MatIterator_& it);
    //! copy operator
    MatIterator_& operator = (const MatIterator_<_Tp>& it );

    //! returns the current matrix element
    _Tp& operator *() const;
    //! returns the i-th matrix element, relative to the current
    _Tp& operator [](ptrdiff_t i) const;

    //! shifts the iterator forward by the specified number of elements
    MatIterator_& operator += (ptrdiff_t ofs);
    //! shifts the iterator backward by the specified number of elements
    MatIterator_& operator -= (ptrdiff_t ofs);
    //! decrements the iterator
    MatIterator_& operator --();
    //! decrements the iterator
    MatIterator_ operator --(int);
    //! increments the iterator
    MatIterator_& operator ++();
    //! increments the iterator
    MatIterator_ operator ++(int);
};

template<typename _Tp> class CV_EXPORTS MatOp_Iter_;


/////////////////////////// multi-dimensional dense matrix //////////////////////////

/*!
 n-Dimensional Dense Matrix Iterator Class.

 The class cv::NAryMatIterator is used for iterating over one or more n-dimensional dense arrays (cv::Mat's).

 The iterator is completely different from cv::Mat_ and cv::SparseMat_ iterators.
 It iterates through the slices (or planes), not the elements, where "slice" is a continuous part of the arrays.

 Here is the example on how the iterator can be used to normalize 3D histogram:

 \code
 void normalizeColorHist(Mat& hist)
 {
 #if 1
     // intialize iterator (the style is different from STL).
     // after initialization the iterator will contain
     // the number of slices or planes
     // the iterator will go through
     Mat* arrays[] = { &hist, 0 };
     Mat planes[1];
     NAryMatIterator it(arrays, planes);
     double s = 0;
     // iterate through the matrix. on each iteration
     // it.planes[i] (of type Mat) will be set to the current plane of
     // i-th n-dim matrix passed to the iterator constructor.
     for(int p = 0; p < it.nplanes; p++, ++it)
        s += sum(it.planes[0])[0];
     it = NAryMatIterator(hist);
     s = 1./s;
     for(int p = 0; p < it.nplanes; p++, ++it)
        it.planes[0] *= s;
 #elif 1
     // this is a shorter implementation of the above
     // using built-in operations on Mat
     double s = sum(hist)[0];
     hist.convertTo(hist, hist.type(), 1./s, 0);
 #else
     // and this is even shorter one
     // (assuming that the histogram elements are non-negative)
     normalize(hist, hist, 1, 0, NORM_L1);
 #endif
 }
 \endcode

 You can iterate through several matrices simultaneously as long as they have the same geometry
 (dimensionality and all the dimension sizes are the same), which is useful for binary
 and n-ary operations on such matrices. Just pass those matrices to cv::MatNDIterator.
 Then, during the iteration it.planes[0], it.planes[1], ... will
 be the slices of the corresponding matrices
*/
class CV_EXPORTS NAryMatIterator
{
public:
    //! the default constructor
    NAryMatIterator();
    //! the full constructor taking arbitrary number of n-dim matrices
    NAryMatIterator(const Mat** arrays, uchar** ptrs, int narrays=-1);
    //! the full constructor taking arbitrary number of n-dim matrices
    NAryMatIterator(const Mat** arrays, Mat* planes, int narrays=-1);
    //! the separate iterator initialization method
    void init(const Mat** arrays, Mat* planes, uchar** ptrs, int narrays=-1);

    //! proceeds to the next plane of every iterated matrix
    NAryMatIterator& operator ++();
    //! proceeds to the next plane of every iterated matrix (postfix increment operator)
    NAryMatIterator operator ++(int);

    //! the iterated arrays
    const Mat** arrays;
    //! the current planes
    Mat* planes;
    //! data pointers
    uchar** ptrs;
    //! the number of arrays
    int narrays;
    //! the number of hyper-planes that the iterator steps through
    size_t nplanes;
    //! the size of each segment (in elements)
    size_t size;
protected:
    int iterdepth;
    size_t idx;
};

//typedef NAryMatIterator NAryMatNDIterator;

typedef void (*ConvertData)(const void* from, void* to, int cn);
typedef void (*ConvertScaleData)(const void* from, void* to, int cn, double alpha, double beta);

//! returns the function for converting pixels from one data type to another
CV_EXPORTS ConvertData getConvertElem(int fromType, int toType);
//! returns the function for converting pixels from one data type to another with the optional scaling
CV_EXPORTS ConvertScaleData getConvertScaleElem(int fromType, int toType);

//! finds global minimum and maximum sparse array elements and returns their values and their locations
CV_EXPORTS void minMaxLoc(const SparseMat& a, double* minVal,
                          double* maxVal, int* minIdx=0, int* maxIdx=0);
//! computes norm of a sparse matrix
CV_EXPORTS double norm( const SparseMat& src, int normType );
//! scales and shifts array elements so that either the specified norm (alpha) or the minimum (alpha) and maximum (beta) array values get the specified values
CV_EXPORTS void normalize( const SparseMat& src, SparseMat& dst, double alpha, int normType );

/*!
 Read-Only Sparse Matrix Iterator.
 Here is how to use the iterator to compute the sum of floating-point sparse matrix elements:

 \code
 SparseMatConstIterator it = m.begin(), it_end = m.end();
 double s = 0;
 CV_Assert( m.type() == CV_32F );
 for( ; it != it_end; ++it )
    s += it.value<float>();
 \endcode
*/
class CV_EXPORTS SparseMatConstIterator
{
public:
    //! the default constructor
    SparseMatConstIterator();
    //! the full constructor setting the iterator to the first sparse matrix element
    SparseMatConstIterator(const SparseMat* _m);
    //! the copy constructor
    SparseMatConstIterator(const SparseMatConstIterator& it);

    //! the assignment operator
    SparseMatConstIterator& operator = (const SparseMatConstIterator& it);

    //! template method returning the current matrix element
    template<typename _Tp> const _Tp& value() const;
    //! returns the current node of the sparse matrix. it.node->idx is the current element index
    const SparseMat::Node* node() const;

    //! moves iterator to the previous element
    SparseMatConstIterator& operator --();
    //! moves iterator to the previous element
    SparseMatConstIterator operator --(int);
    //! moves iterator to the next element
    SparseMatConstIterator& operator ++();
    //! moves iterator to the next element
    SparseMatConstIterator operator ++(int);

    //! moves iterator to the element after the last element
    void seekEnd();

    const SparseMat* m;
    size_t hashidx;
    uchar* ptr;
};

/*!
 Read-write Sparse Matrix Iterator

 The class is similar to cv::SparseMatConstIterator,
 but can be used for in-place modification of the matrix elements.
*/
class CV_EXPORTS SparseMatIterator : public SparseMatConstIterator
{
public:
    //! the default constructor
    SparseMatIterator();
    //! the full constructor setting the iterator to the first sparse matrix element
    SparseMatIterator(SparseMat* _m);
    //! the full constructor setting the iterator to the specified sparse matrix element
    SparseMatIterator(SparseMat* _m, const int* idx);
    //! the copy constructor
    SparseMatIterator(const SparseMatIterator& it);

    //! the assignment operator
    SparseMatIterator& operator = (const SparseMatIterator& it);
    //! returns read-write reference to the current sparse matrix element
    template<typename _Tp> _Tp& value() const;
    //! returns pointer to the current sparse matrix node. it.node->idx is the index of the current element (do not modify it!)
    SparseMat::Node* node() const;

    //! moves iterator to the next element
    SparseMatIterator& operator ++();
    //! moves iterator to the next element
    SparseMatIterator operator ++(int);
};


/*!
 Template Read-Only Sparse Matrix Iterator Class.

 This is the derived from SparseMatConstIterator class that
 introduces more convenient operator *() for accessing the current element.
*/
template<typename _Tp> class CV_EXPORTS SparseMatConstIterator_ : public SparseMatConstIterator
{
public:
    typedef std::forward_iterator_tag iterator_category;

    //! the default constructor
    SparseMatConstIterator_();
    //! the full constructor setting the iterator to the first sparse matrix element
    SparseMatConstIterator_(const SparseMat_<_Tp>* _m);
    //! the copy constructor
    SparseMatConstIterator_(const SparseMatConstIterator_& it);

    //! the assignment operator
    SparseMatConstIterator_& operator = (const SparseMatConstIterator_& it);
    //! the element access operator
    const _Tp& operator *() const;

    //! moves iterator to the next element
    SparseMatConstIterator_& operator ++();
    //! moves iterator to the next element
    SparseMatConstIterator_ operator ++(int);
};

/*!
 Template Read-Write Sparse Matrix Iterator Class.

 This is the derived from cv::SparseMatConstIterator_ class that
 introduces more convenient operator *() for accessing the current element.
*/
template<typename _Tp> class CV_EXPORTS SparseMatIterator_ : public SparseMatConstIterator_<_Tp>
{
public:
    typedef std::forward_iterator_tag iterator_category;

    //! the default constructor
    SparseMatIterator_();
    //! the full constructor setting the iterator to the first sparse matrix element
    SparseMatIterator_(SparseMat_<_Tp>* _m);
    //! the copy constructor
    SparseMatIterator_(const SparseMatIterator_& it);

    //! the assignment operator
    SparseMatIterator_& operator = (const SparseMatIterator_& it);
    //! returns the reference to the current element
    _Tp& operator *() const;

    //! moves the iterator to the next element
    SparseMatIterator_& operator ++();
    //! moves the iterator to the next element
    SparseMatIterator_ operator ++(int);
};

//////////////////// Fast Nearest-Neighbor Search Structure ////////////////////

/*!
 Fast Nearest Neighbor Search Class.

 The class implements D. Lowe BBF (Best-Bin-First) algorithm for the last
 approximate (or accurate) nearest neighbor search in multi-dimensional spaces.

 First, a set of vectors is passed to KDTree::KDTree() constructor
 or KDTree::build() method, where it is reordered.

 Then arbitrary vectors can be passed to KDTree::findNearest() methods, which
 find the K nearest neighbors among the vectors from the initial set.
 The user can balance between the speed and accuracy of the search by varying Emax
 parameter, which is the number of leaves that the algorithm checks.
 The larger parameter values yield more accurate results at the expense of lower processing speed.

 \code
 KDTree T(points, false);
 const int K = 3, Emax = INT_MAX;
 int idx[K];
 float dist[K];
 T.findNearest(query_vec, K, Emax, idx, 0, dist);
 CV_Assert(dist[0] <= dist[1] && dist[1] <= dist[2]);
 \endcode
*/
class CV_EXPORTS_W KDTree
{
public:
    /*!
        The node of the search tree.
    */
    struct Node
    {
        Node() : idx(-1), left(-1), right(-1), boundary(0.f) {}
        Node(int _idx, int _left, int _right, float _boundary)
            : idx(_idx), left(_left), right(_right), boundary(_boundary) {}
        //! split dimension; >=0 for nodes (dim), < 0 for leaves (index of the point)
        int idx;
        //! node indices of the left and the right branches
        int left, right;
        //! go to the left if query_vec[node.idx]<=node.boundary, otherwise go to the right
        float boundary;
    };

    //! the default constructor
    CV_WRAP KDTree();
    //! the full constructor that builds the search tree
    CV_WRAP KDTree(InputArray points, bool copyAndReorderPoints=false);
    //! the full constructor that builds the search tree
    CV_WRAP KDTree(InputArray points, InputArray _labels,
                   bool copyAndReorderPoints=false);
    //! builds the search tree
    CV_WRAP void build(InputArray points, bool copyAndReorderPoints=false);
    //! builds the search tree
    CV_WRAP void build(InputArray points, InputArray labels,
                       bool copyAndReorderPoints=false);
    //! finds the K nearest neighbors of "vec" while looking at Emax (at most) leaves
    CV_WRAP int findNearest(InputArray vec, int K, int Emax,
                            OutputArray neighborsIdx,
                            OutputArray neighbors=noArray(),
                            OutputArray dist=noArray(),
                            OutputArray labels=noArray()) const;
    //! finds all the points from the initial set that belong to the specified box
    CV_WRAP void findOrthoRange(InputArray minBounds,
                                InputArray maxBounds,
                                OutputArray neighborsIdx,
                                OutputArray neighbors=noArray(),
                                OutputArray labels=noArray()) const;
    //! returns vectors with the specified indices
    CV_WRAP void getPoints(InputArray idx, OutputArray pts,
                           OutputArray labels=noArray()) const;
    //! return a vector with the specified index
    const float* getPoint(int ptidx, int* label=0) const;
    //! returns the search space dimensionality
    CV_WRAP int dims() const;

    std::vector<Node> nodes; //!< all the tree nodes
    CV_PROP Mat points; //!< all the points. It can be a reordered copy of the input vector set or the original vector set.
    CV_PROP std::vector<int> labels; //!< the parallel array of labels.
    CV_PROP int maxDepth; //!< maximum depth of the search tree. Do not modify it
    CV_PROP_RW int normType; //!< type of the distance (cv::NORM_L1 or cv::NORM_L2) used for search. Initially set to cv::NORM_L2, but you can modify it
};

//////////////////////////////////////// XML & YAML I/O ////////////////////////////////////

class CV_EXPORTS FileNode;

/*!
 XML/YAML File Storage Class.

 The class describes an object associated with XML or YAML file.
 It can be used to store data to such a file or read and decode the data.

 The storage is organized as a tree of nested sequences (or lists) and mappings.
 Sequence is a heterogenious array, which elements are accessed by indices or sequentially using an iterator.
 Mapping is analogue of std::map or C structure, which elements are accessed by names.
 The most top level structure is a mapping.
 Leaves of the file storage tree are integers, floating-point numbers and text strings.

 For example, the following code:

 \code
 // open file storage for writing. Type of the file is determined from the extension
 FileStorage fs("test.yml", FileStorage::WRITE);
 fs << "test_int" << 5 << "test_real" << 3.1 << "test_string" << "ABCDEFGH";
 fs << "test_mat" << Mat::eye(3,3,CV_32F);

 fs << "test_list" << "[" << 0.0000000000001 << 2 << CV_PI << -3435345 << "2-502 2-029 3egegeg" <<
 "{:" << "month" << 12 << "day" << 31 << "year" << 1969 << "}" << "]";
 fs << "test_map" << "{" << "x" << 1 << "y" << 2 << "width" << 100 << "height" << 200 << "lbp" << "[:";

 const uchar arr[] = {0, 1, 1, 0, 1, 1, 0, 1};
 fs.writeRaw("u", arr, (int)(sizeof(arr)/sizeof(arr[0])));

 fs << "]" << "}";
 \endcode

 will produce the following file:

 \verbatim
 %YAML:1.0
 test_int: 5
 test_real: 3.1000000000000001e+00
 test_string: ABCDEFGH
 test_mat: !!opencv-matrix
     rows: 3
     cols: 3
     dt: f
     data: [ 1., 0., 0., 0., 1., 0., 0., 0., 1. ]
 test_list:
     - 1.0000000000000000e-13
     - 2
     - 3.1415926535897931e+00
     - -3435345
     - "2-502 2-029 3egegeg"
     - { month:12, day:31, year:1969 }
 test_map:
     x: 1
     y: 2
     width: 100
     height: 200
     lbp: [ 0, 1, 1, 0, 1, 1, 0, 1 ]
 \endverbatim

 and to read the file above, the following code can be used:

 \code
 // open file storage for reading.
 // Type of the file is determined from the content, not the extension
 FileStorage fs("test.yml", FileStorage::READ);
 int test_int = (int)fs["test_int"];
 double test_real = (double)fs["test_real"];
 String test_string = (String)fs["test_string"];

 Mat M;
 fs["test_mat"] >> M;

 FileNode tl = fs["test_list"];
 CV_Assert(tl.type() == FileNode::SEQ && tl.size() == 6);
 double tl0 = (double)tl[0];
 int tl1 = (int)tl[1];
 double tl2 = (double)tl[2];
 int tl3 = (int)tl[3];
 String tl4 = (String)tl[4];
 CV_Assert(tl[5].type() == FileNode::MAP && tl[5].size() == 3);

 int month = (int)tl[5]["month"];
 int day = (int)tl[5]["day"];
 int year = (int)tl[5]["year"];

 FileNode tm = fs["test_map"];

 int x = (int)tm["x"];
 int y = (int)tm["y"];
 int width = (int)tm["width"];
 int height = (int)tm["height"];

 int lbp_val = 0;
 FileNodeIterator it = tm["lbp"].begin();

 for(int k = 0; k < 8; k++, ++it)
    lbp_val |= ((int)*it) << k;
 \endcode
*/
class CV_EXPORTS_W FileStorage
{
public:
    //! file storage mode
    enum
    {
        READ=0, //! read mode
        WRITE=1, //! write mode
        APPEND=2, //! append mode
        MEMORY=4,
        FORMAT_MASK=(7<<3),
        FORMAT_AUTO=0,
        FORMAT_XML=(1<<3),
        FORMAT_YAML=(2<<3)
    };
    enum
    {
        UNDEFINED=0,
        VALUE_EXPECTED=1,
        NAME_EXPECTED=2,
        INSIDE_MAP=4
    };
    //! the default constructor
    CV_WRAP FileStorage();
    //! the full constructor that opens file storage for reading or writing
    CV_WRAP FileStorage(const String& source, int flags, const String& encoding=String());
    //! the constructor that takes pointer to the C FileStorage structure
    FileStorage(CvFileStorage* fs);
    //! the destructor. calls release()
    virtual ~FileStorage();

    //! opens file storage for reading or writing. The previous storage is closed with release()
    CV_WRAP virtual bool open(const String& filename, int flags, const String& encoding=String());
    //! returns true if the object is associated with currently opened file.
    CV_WRAP virtual bool isOpened() const;
    //! closes the file and releases all the memory buffers
    CV_WRAP virtual void release();
    //! closes the file, releases all the memory buffers and returns the text string
    CV_WRAP virtual String releaseAndGetString();

    //! returns the first element of the top-level mapping
    CV_WRAP FileNode getFirstTopLevelNode() const;
    //! returns the top-level mapping. YAML supports multiple streams
    CV_WRAP FileNode root(int streamidx=0) const;
    //! returns the specified element of the top-level mapping
    FileNode operator[](const String& nodename) const;
    //! returns the specified element of the top-level mapping
    CV_WRAP FileNode operator[](const char* nodename) const;

    //! returns pointer to the underlying C FileStorage structure
    CvFileStorage* operator *() { return fs; }
    //! returns pointer to the underlying C FileStorage structure
    const CvFileStorage* operator *() const { return fs; }
    //! writes one or more numbers of the specified format to the currently written structure
    void writeRaw( const String& fmt, const uchar* vec, size_t len );
    //! writes the registered C structure (CvMat, CvMatND, CvSeq). See cvWrite()
    void writeObj( const String& name, const void* obj );

    //! returns the normalized object name for the specified file name
    static String getDefaultObjectName(const String& filename);

    Ptr<CvFileStorage> fs; //!< the underlying C FileStorage structure
    String elname; //!< the currently written element
    std::vector<char> structs; //!< the stack of written structures
    int state; //!< the writer state
};

class CV_EXPORTS FileNodeIterator;

/*!
 File Storage Node class

 The node is used to store each and every element of the file storage opened for reading -
 from the primitive objects, such as numbers and text strings, to the complex nodes:
 sequences, mappings and the registered objects.

 Note that file nodes are only used for navigating file storages opened for reading.
 When a file storage is opened for writing, no data is stored in memory after it is written.
*/
class CV_EXPORTS_W_SIMPLE FileNode
{
public:
    //! type of the file storage node
    enum
    {
        NONE=0, //!< empty node
        INT=1, //!< an integer
        REAL=2, //!< floating-point number
        FLOAT=REAL, //!< synonym or REAL
        STR=3, //!< text string in UTF-8 encoding
        STRING=STR, //!< synonym for STR
        REF=4, //!< integer of size size_t. Typically used for storing complex dynamic structures where some elements reference the others
        SEQ=5, //!< sequence
        MAP=6, //!< mapping
        TYPE_MASK=7,
        FLOW=8, //!< compact representation of a sequence or mapping. Used only by YAML writer
        USER=16, //!< a registered object (e.g. a matrix)
        EMPTY=32, //!< empty structure (sequence or mapping)
        NAMED=64 //!< the node has a name (i.e. it is element of a mapping)
    };
    //! the default constructor
    CV_WRAP FileNode();
    //! the full constructor wrapping CvFileNode structure.
    FileNode(const CvFileStorage* fs, const CvFileNode* node);
    //! the copy constructor
    FileNode(const FileNode& node);
    //! returns element of a mapping node
    FileNode operator[](const String& nodename) const;
    //! returns element of a mapping node
    CV_WRAP FileNode operator[](const char* nodename) const;
    //! returns element of a sequence node
    CV_WRAP FileNode operator[](int i) const;
    //! returns type of the node
    CV_WRAP int type() const;

    //! returns true if the node is empty
    CV_WRAP bool empty() const;
    //! returns true if the node is a "none" object
    CV_WRAP bool isNone() const;
    //! returns true if the node is a sequence
    CV_WRAP bool isSeq() const;
    //! returns true if the node is a mapping
    CV_WRAP bool isMap() const;
    //! returns true if the node is an integer
    CV_WRAP bool isInt() const;
    //! returns true if the node is a floating-point number
    CV_WRAP bool isReal() const;
    //! returns true if the node is a text string
    CV_WRAP bool isString() const;
    //! returns true if the node has a name
    CV_WRAP bool isNamed() const;
    //! returns the node name or an empty string if the node is nameless
    CV_WRAP String name() const;
    //! returns the number of elements in the node, if it is a sequence or mapping, or 1 otherwise.
    CV_WRAP size_t size() const;
    //! returns the node content as an integer. If the node stores floating-point number, it is rounded.
    operator int() const;
    //! returns the node content as float
    operator float() const;
    //! returns the node content as double
    operator double() const;
    //! returns the node content as text string
    operator String() const;
#ifndef OPENCV_NOSTL
    operator std::string() const;
#endif

    //! returns pointer to the underlying file node
    CvFileNode* operator *();
    //! returns pointer to the underlying file node
    const CvFileNode* operator* () const;

    //! returns iterator pointing to the first node element
    FileNodeIterator begin() const;
    //! returns iterator pointing to the element following the last node element
    FileNodeIterator end() const;

    //! reads node elements to the buffer with the specified format
    void readRaw( const String& fmt, uchar* vec, size_t len ) const;
    //! reads the registered object and returns pointer to it
    void* readObj() const;

    // do not use wrapper pointer classes for better efficiency
    const CvFileStorage* fs;
    const CvFileNode* node;
};


/*!
 File Node Iterator

 The class is used for iterating sequences (usually) and mappings.
 */
class CV_EXPORTS FileNodeIterator
{
public:
    //! the default constructor
    FileNodeIterator();
    //! the full constructor set to the ofs-th element of the node
    FileNodeIterator(const CvFileStorage* fs, const CvFileNode* node, size_t ofs=0);
    //! the copy constructor
    FileNodeIterator(const FileNodeIterator& it);
    //! returns the currently observed element
    FileNode operator *() const;
    //! accesses the currently observed element methods
    FileNode operator ->() const;

    //! moves iterator to the next node
    FileNodeIterator& operator ++ ();
    //! moves iterator to the next node
    FileNodeIterator operator ++ (int);
    //! moves iterator to the previous node
    FileNodeIterator& operator -- ();
    //! moves iterator to the previous node
    FileNodeIterator operator -- (int);
    //! moves iterator forward by the specified offset (possibly negative)
    FileNodeIterator& operator += (int ofs);
    //! moves iterator backward by the specified offset (possibly negative)
    FileNodeIterator& operator -= (int ofs);

    //! reads the next maxCount elements (or less, if the sequence/mapping last element occurs earlier) to the buffer with the specified format
    FileNodeIterator& readRaw( const String& fmt, uchar* vec,
                               size_t maxCount=(size_t)INT_MAX );

    struct SeqReader
    {
      int          header_size;
      CvSeq*       seq;        /* sequence, beign read */
      CvSeqBlock*  block;      /* current block */
      schar*       ptr;        /* pointer to element be read next */
      schar*       block_min;  /* pointer to the beginning of block */
      schar*       block_max;  /* pointer to the end of block */
      int          delta_index;/* = seq->first->start_index   */
      schar*       prev_elem;  /* pointer to previous element */
    };

    const CvFileStorage* fs;
    const CvFileNode* container;
    SeqReader reader;
    size_t remaining;
};

class CV_EXPORTS Algorithm;
class CV_EXPORTS AlgorithmInfo;
struct CV_EXPORTS AlgorithmInfoData;

template<typename _Tp> struct ParamType {};

/*!
  Base class for high-level OpenCV algorithms
*/
class CV_EXPORTS_W Algorithm
{
public:
    Algorithm();
    virtual ~Algorithm();
    String name() const;

    template<typename _Tp> typename ParamType<_Tp>::member_type get(const String& name) const;
    template<typename _Tp> typename ParamType<_Tp>::member_type get(const char* name) const;

    CV_WRAP int getInt(const String& name) const;
    CV_WRAP double getDouble(const String& name) const;
    CV_WRAP bool getBool(const String& name) const;
    CV_WRAP String getString(const String& name) const;
    CV_WRAP Mat getMat(const String& name) const;
    CV_WRAP std::vector<Mat> getMatVector(const String& name) const;
    CV_WRAP Ptr<Algorithm> getAlgorithm(const String& name) const;

    void set(const String& name, int value);
    void set(const String& name, double value);
    void set(const String& name, bool value);
    void set(const String& name, const String& value);
    void set(const String& name, const Mat& value);
    void set(const String& name, const std::vector<Mat>& value);
    void set(const String& name, const Ptr<Algorithm>& value);
    template<typename _Tp> void set(const String& name, const Ptr<_Tp>& value);

    CV_WRAP void setInt(const String& name, int value);
    CV_WRAP void setDouble(const String& name, double value);
    CV_WRAP void setBool(const String& name, bool value);
    CV_WRAP void setString(const String& name, const String& value);
    CV_WRAP void setMat(const String& name, const Mat& value);
    CV_WRAP void setMatVector(const String& name, const std::vector<Mat>& value);
    CV_WRAP void setAlgorithm(const String& name, const Ptr<Algorithm>& value);
    template<typename _Tp> void setAlgorithm(const String& name, const Ptr<_Tp>& value);

    void set(const char* name, int value);
    void set(const char* name, double value);
    void set(const char* name, bool value);
    void set(const char* name, const String& value);
    void set(const char* name, const Mat& value);
    void set(const char* name, const std::vector<Mat>& value);
    void set(const char* name, const Ptr<Algorithm>& value);
    template<typename _Tp> void set(const char* name, const Ptr<_Tp>& value);

    void setInt(const char* name, int value);
    void setDouble(const char* name, double value);
    void setBool(const char* name, bool value);
    void setString(const char* name, const String& value);
    void setMat(const char* name, const Mat& value);
    void setMatVector(const char* name, const std::vector<Mat>& value);
    void setAlgorithm(const char* name, const Ptr<Algorithm>& value);
    template<typename _Tp> void setAlgorithm(const char* name, const Ptr<_Tp>& value);

    CV_WRAP String paramHelp(const String& name) const;
    int paramType(const char* name) const;
    CV_WRAP int paramType(const String& name) const;
    CV_WRAP void getParams(CV_OUT std::vector<String>& names) const;


    virtual void write(FileStorage& fs) const;
    virtual void read(const FileNode& fn);

    typedef Algorithm* (*Constructor)(void);
    typedef int (Algorithm::*Getter)() const;
    typedef void (Algorithm::*Setter)(int);

    CV_WRAP static void getList(CV_OUT std::vector<String>& algorithms);
    CV_WRAP static Ptr<Algorithm> _create(const String& name);
    template<typename _Tp> static Ptr<_Tp> create(const String& name);

    virtual AlgorithmInfo* info() const /* TODO: make it = 0;*/ { return 0; }
};


class CV_EXPORTS AlgorithmInfo
{
public:
    friend class Algorithm;
    AlgorithmInfo(const String& name, Algorithm::Constructor create);
    ~AlgorithmInfo();
    void get(const Algorithm* algo, const char* name, int argType, void* value) const;
    void addParam_(Algorithm& algo, const char* name, int argType,
                   void* value, bool readOnly,
                   Algorithm::Getter getter, Algorithm::Setter setter,
                   const String& help=String());
    String paramHelp(const char* name) const;
    int paramType(const char* name) const;
    void getParams(std::vector<String>& names) const;

    void write(const Algorithm* algo, FileStorage& fs) const;
    void read(Algorithm* algo, const FileNode& fn) const;
    String name() const;

    void addParam(Algorithm& algo, const char* name,
                  int& value, bool readOnly=false,
                  int (Algorithm::*getter)()=0,
                  void (Algorithm::*setter)(int)=0,
                  const String& help=String());
    void addParam(Algorithm& algo, const char* name,
                  bool& value, bool readOnly=false,
                  int (Algorithm::*getter)()=0,
                  void (Algorithm::*setter)(int)=0,
                  const String& help=String());
    void addParam(Algorithm& algo, const char* name,
                  double& value, bool readOnly=false,
                  double (Algorithm::*getter)()=0,
                  void (Algorithm::*setter)(double)=0,
                  const String& help=String());
    void addParam(Algorithm& algo, const char* name,
                  String& value, bool readOnly=false,
                  String (Algorithm::*getter)()=0,
                  void (Algorithm::*setter)(const String&)=0,
                  const String& help=String());
    void addParam(Algorithm& algo, const char* name,
                  Mat& value, bool readOnly=false,
                  Mat (Algorithm::*getter)()=0,
                  void (Algorithm::*setter)(const Mat&)=0,
                  const String& help=String());
    void addParam(Algorithm& algo, const char* name,
                  std::vector<Mat>& value, bool readOnly=false,
                  std::vector<Mat> (Algorithm::*getter)()=0,
                  void (Algorithm::*setter)(const std::vector<Mat>&)=0,
                  const String& help=String());
    void addParam(Algorithm& algo, const char* name,
                  Ptr<Algorithm>& value, bool readOnly=false,
                  Ptr<Algorithm> (Algorithm::*getter)()=0,
                  void (Algorithm::*setter)(const Ptr<Algorithm>&)=0,
                  const String& help=String());
    void addParam(Algorithm& algo, const char* name,
                  float& value, bool readOnly=false,
                  float (Algorithm::*getter)()=0,
                  void (Algorithm::*setter)(float)=0,
                  const String& help=String());
    void addParam(Algorithm& algo, const char* name,
                  unsigned int& value, bool readOnly=false,
                  unsigned int (Algorithm::*getter)()=0,
                  void (Algorithm::*setter)(unsigned int)=0,
                  const String& help=String());
    void addParam(Algorithm& algo, const char* name,
                  uint64& value, bool readOnly=false,
                  uint64 (Algorithm::*getter)()=0,
                  void (Algorithm::*setter)(uint64)=0,
                  const String& help=String());
    void addParam(Algorithm& algo, const char* name,
                  uchar& value, bool readOnly=false,
                  uchar (Algorithm::*getter)()=0,
                  void (Algorithm::*setter)(uchar)=0,
                  const String& help=String());
    template<typename _Tp, typename _Base> void addParam(Algorithm& algo, const char* name,
                  Ptr<_Tp>& value, bool readOnly=false,
                  Ptr<_Tp> (Algorithm::*getter)()=0,
                  void (Algorithm::*setter)(const Ptr<_Tp>&)=0,
                  const String& help=String());
    template<typename _Tp> void addParam(Algorithm& algo, const char* name,
                  Ptr<_Tp>& value, bool readOnly=false,
                  Ptr<_Tp> (Algorithm::*getter)()=0,
                  void (Algorithm::*setter)(const Ptr<_Tp>&)=0,
                  const String& help=String());
protected:
    AlgorithmInfoData* data;
    void set(Algorithm* algo, const char* name, int argType,
              const void* value, bool force=false) const;
};


struct CV_EXPORTS Param
{
    enum { INT=0, BOOLEAN=1, REAL=2, STRING=3, MAT=4, MAT_VECTOR=5, ALGORITHM=6, FLOAT=7, UNSIGNED_INT=8, UINT64=9, UCHAR=11 };

    Param();
    Param(int _type, bool _readonly, int _offset,
          Algorithm::Getter _getter=0,
          Algorithm::Setter _setter=0,
          const String& _help=String());
    int type;
    int offset;
    bool readonly;
    Algorithm::Getter getter;
    Algorithm::Setter setter;
    String help;
};

template<> struct ParamType<bool>
{
    typedef bool const_param_type;
    typedef bool member_type;

    enum { type = Param::BOOLEAN };
};

template<> struct ParamType<int>
{
    typedef int const_param_type;
    typedef int member_type;

    enum { type = Param::INT };
};

template<> struct ParamType<double>
{
    typedef double const_param_type;
    typedef double member_type;

    enum { type = Param::REAL };
};

template<> struct ParamType<String>
{
    typedef const String& const_param_type;
    typedef String member_type;

    enum { type = Param::STRING };
};

template<> struct ParamType<Mat>
{
    typedef const Mat& const_param_type;
    typedef Mat member_type;

    enum { type = Param::MAT };
};

template<> struct ParamType<std::vector<Mat> >
{
    typedef const std::vector<Mat>& const_param_type;
    typedef std::vector<Mat> member_type;

    enum { type = Param::MAT_VECTOR };
};

template<> struct ParamType<Algorithm>
{
    typedef const Ptr<Algorithm>& const_param_type;
    typedef Ptr<Algorithm> member_type;

    enum { type = Param::ALGORITHM };
};

template<> struct ParamType<float>
{
    typedef float const_param_type;
    typedef float member_type;

    enum { type = Param::FLOAT };
};

template<> struct ParamType<unsigned>
{
    typedef unsigned const_param_type;
    typedef unsigned member_type;

    enum { type = Param::UNSIGNED_INT };
};

template<> struct ParamType<uint64>
{
    typedef uint64 const_param_type;
    typedef uint64 member_type;

    enum { type = Param::UINT64 };
};

template<> struct ParamType<uchar>
{
    typedef uchar const_param_type;
    typedef uchar member_type;

    enum { type = Param::UCHAR };
};

} //namespace cv

#include "opencv2/core/operations.hpp"
#include "opencv2/core/mat.inl.hpp"
#include "opencv2/core/cvstd.inl.hpp"

#endif /*__OPENCV_CORE_HPP__*/
