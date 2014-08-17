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
#include "opencv2/core/base.hpp"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/traits.hpp"
#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/persistence.hpp"

/*! \namespace cv
    Namespace where all the C++ OpenCV functionality resides
*/
namespace cv {

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
    String func; ///< function name. Available only when the compiler supports getting it
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


enum { SORT_EVERY_ROW    = 0,
       SORT_EVERY_COLUMN = 1,
       SORT_ASCENDING    = 0,
       SORT_DESCENDING   = 16
     };

enum { COVAR_SCRAMBLED = 0,
       COVAR_NORMAL    = 1,
       COVAR_USE_AVG   = 2,
       COVAR_SCALE     = 4,
       COVAR_ROWS      = 8,
       COVAR_COLS      = 16
     };

/*!
 k-Means flags
*/
enum { KMEANS_RANDOM_CENTERS     = 0, // Chooses random centers for k-Means initialization
       KMEANS_PP_CENTERS         = 2, // Uses k-Means++ algorithm for initialization
       KMEANS_USE_INITIAL_LABELS = 1  // Uses the user-provided labels for K-Means initialization
     };

enum { FILLED  = -1,
       LINE_4  = 4,
       LINE_8  = 8,
       LINE_AA = 16
     };

enum { FONT_HERSHEY_SIMPLEX        = 0,
       FONT_HERSHEY_PLAIN          = 1,
       FONT_HERSHEY_DUPLEX         = 2,
       FONT_HERSHEY_COMPLEX        = 3,
       FONT_HERSHEY_TRIPLEX        = 4,
       FONT_HERSHEY_COMPLEX_SMALL  = 5,
       FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
       FONT_HERSHEY_SCRIPT_COMPLEX = 7,
       FONT_ITALIC                 = 16
     };

enum { REDUCE_SUM = 0,
       REDUCE_AVG = 1,
       REDUCE_MAX = 2,
       REDUCE_MIN = 3
     };


//! swaps two matrices
CV_EXPORTS void swap(Mat& a, Mat& b);

//! swaps two umatrices
CV_EXPORTS void swap( UMat& a, UMat& b );

//! 1D interpolation function: returns coordinate of the "donor" pixel for the specified location p.
CV_EXPORTS_W int borderInterpolate(int p, int len, int borderType);

//! copies 2D array to a larger destination array with extrapolation of the outer part of src using the specified border mode
CV_EXPORTS_W void copyMakeBorder(InputArray src, OutputArray dst,
                                 int top, int bottom, int left, int right,
                                 int borderType, const Scalar& value = Scalar() );

//! adds one matrix to another (dst = src1 + src2)
CV_EXPORTS_W void add(InputArray src1, InputArray src2, OutputArray dst,
                      InputArray mask = noArray(), int dtype = -1);

//! subtracts one matrix from another (dst = src1 - src2)
CV_EXPORTS_W void subtract(InputArray src1, InputArray src2, OutputArray dst,
                           InputArray mask = noArray(), int dtype = -1);

//! computes element-wise weighted product of the two arrays (dst = scale*src1*src2)
CV_EXPORTS_W void multiply(InputArray src1, InputArray src2,
                           OutputArray dst, double scale = 1, int dtype = -1);

//! computes element-wise weighted quotient of the two arrays (dst = scale * src1 / src2)
CV_EXPORTS_W void divide(InputArray src1, InputArray src2, OutputArray dst,
                         double scale = 1, int dtype = -1);

//! computes element-wise weighted reciprocal of an array (dst = scale/src2)
CV_EXPORTS_W void divide(double scale, InputArray src2,
                         OutputArray dst, int dtype = -1);

//! adds scaled array to another one (dst = alpha*src1 + src2)
CV_EXPORTS_W void scaleAdd(InputArray src1, double alpha, InputArray src2, OutputArray dst);

//! computes weighted sum of two arrays (dst = alpha*src1 + beta*src2 + gamma)
CV_EXPORTS_W void addWeighted(InputArray src1, double alpha, InputArray src2,
                              double beta, double gamma, OutputArray dst, int dtype = -1);

//! scales array elements, computes absolute values and converts the results to 8-bit unsigned integers: dst(i)=saturate_cast<uchar>abs(src(i)*alpha+beta)
CV_EXPORTS_W void convertScaleAbs(InputArray src, OutputArray dst,
                                  double alpha = 1, double beta = 0);

//! transforms array of numbers using a lookup table: dst(i)=lut(src(i))
CV_EXPORTS_W void LUT(InputArray src, InputArray lut, OutputArray dst);

//! computes sum of array elements
CV_EXPORTS_AS(sumElems) Scalar sum(InputArray src);

//! computes the number of nonzero array elements
CV_EXPORTS_W int countNonZero( InputArray src );

//! returns the list of locations of non-zero pixels
CV_EXPORTS_W void findNonZero( InputArray src, OutputArray idx );

//! computes mean value of selected array elements
CV_EXPORTS_W Scalar mean(InputArray src, InputArray mask = noArray());

//! computes mean value and standard deviation of all or selected array elements
CV_EXPORTS_W void meanStdDev(InputArray src, OutputArray mean, OutputArray stddev,
                             InputArray mask=noArray());

//! computes norm of the selected array part
CV_EXPORTS_W double norm(InputArray src1, int normType = NORM_L2, InputArray mask = noArray());

//! computes norm of selected part of the difference between two arrays
CV_EXPORTS_W double norm(InputArray src1, InputArray src2,
                         int normType = NORM_L2, InputArray mask = noArray());

//! computes PSNR image/video quality metric
CV_EXPORTS_W double PSNR(InputArray src1, InputArray src2);

//! computes norm of a sparse matrix
CV_EXPORTS double norm( const SparseMat& src, int normType );

//! naive nearest neighbor finder
CV_EXPORTS_W void batchDistance(InputArray src1, InputArray src2,
                                OutputArray dist, int dtype, OutputArray nidx,
                                int normType = NORM_L2, int K = 0,
                                InputArray mask = noArray(), int update = 0,
                                bool crosscheck = false);

//! scales and shifts array elements so that either the specified norm (alpha) or the minimum (alpha) and maximum (beta) array values get the specified values
CV_EXPORTS_W void normalize( InputArray src, InputOutputArray dst, double alpha = 1, double beta = 0,
                             int norm_type = NORM_L2, int dtype = -1, InputArray mask = noArray());

//! scales and shifts array elements so that either the specified norm (alpha) or the minimum (alpha) and maximum (beta) array values get the specified values
CV_EXPORTS void normalize( const SparseMat& src, SparseMat& dst, double alpha, int normType );

//! finds global minimum and maximum array elements and returns their values and their locations
CV_EXPORTS_W void minMaxLoc(InputArray src, CV_OUT double* minVal,
                            CV_OUT double* maxVal = 0, CV_OUT Point* minLoc = 0,
                            CV_OUT Point* maxLoc = 0, InputArray mask = noArray());

CV_EXPORTS void minMaxIdx(InputArray src, double* minVal, double* maxVal = 0,
                          int* minIdx = 0, int* maxIdx = 0, InputArray mask = noArray());

//! finds global minimum and maximum sparse array elements and returns their values and their locations
CV_EXPORTS void minMaxLoc(const SparseMat& a, double* minVal,
                          double* maxVal, int* minIdx = 0, int* maxIdx = 0);

//! transforms 2D matrix to 1D row or column vector by taking sum, minimum, maximum or mean value over all the rows
CV_EXPORTS_W void reduce(InputArray src, OutputArray dst, int dim, int rtype, int dtype = -1);

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

CV_EXPORTS void mixChannels(InputArrayOfArrays src, InputOutputArrayOfArrays dst,
                            const int* fromTo, size_t npairs);

CV_EXPORTS_W void mixChannels(InputArrayOfArrays src, InputOutputArrayOfArrays dst,
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
                              OutputArray dst, InputArray mask = noArray());

//! computes bitwise disjunction of the two arrays (dst = src1 | src2)
CV_EXPORTS_W void bitwise_or(InputArray src1, InputArray src2,
                             OutputArray dst, InputArray mask = noArray());

//! computes bitwise exclusive-or of the two arrays (dst = src1 ^ src2)
CV_EXPORTS_W void bitwise_xor(InputArray src1, InputArray src2,
                              OutputArray dst, InputArray mask = noArray());

//! inverts each bit of array (dst = ~src)
CV_EXPORTS_W void bitwise_not(InputArray src, OutputArray dst,
                              InputArray mask = noArray());

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

// the following overloads are needed to avoid conflicts with
//     const _Tp& std::min(const _Tp&, const _Tp&, _Compare)
//! computes per-element minimum of two arrays (dst = min(src1, src2))
CV_EXPORTS void min(const Mat& src1, const Mat& src2, Mat& dst);
//! computes per-element maximum of two arrays (dst = max(src1, src2))
CV_EXPORTS void max(const Mat& src1, const Mat& src2, Mat& dst);
//! computes per-element minimum of two arrays (dst = min(src1, src2))
CV_EXPORTS void min(const UMat& src1, const UMat& src2, UMat& dst);
//! computes per-element maximum of two arrays (dst = max(src1, src2))
CV_EXPORTS void max(const UMat& src1, const UMat& src2, UMat& dst);

//! computes square root of each matrix element (dst = src**0.5)
CV_EXPORTS_W void sqrt(InputArray src, OutputArray dst);

//! raises the input matrix elements to the specified power (b = a**power)
CV_EXPORTS_W void pow(InputArray src, double power, OutputArray dst);

//! computes exponent of each matrix element (dst = e**src)
CV_EXPORTS_W void exp(InputArray src, OutputArray dst);

//! computes natural logarithm of absolute value of each matrix element: dst = log(abs(src))
CV_EXPORTS_W void log(InputArray src, OutputArray dst);

//! converts polar coordinates to Cartesian
CV_EXPORTS_W void polarToCart(InputArray magnitude, InputArray angle,
                              OutputArray x, OutputArray y, bool angleInDegrees = false);

//! converts Cartesian coordinates to polar
CV_EXPORTS_W void cartToPolar(InputArray x, InputArray y,
                              OutputArray magnitude, OutputArray angle,
                              bool angleInDegrees = false);

//! computes angle (angle(i)) of each (x(i), y(i)) vector
CV_EXPORTS_W void phase(InputArray x, InputArray y, OutputArray angle,
                        bool angleInDegrees = false);

//! computes magnitude (magnitude(i)) of each (x(i), y(i)) vector
CV_EXPORTS_W void magnitude(InputArray x, InputArray y, OutputArray magnitude);

//! checks that each matrix element is within the specified range.
CV_EXPORTS_W bool checkRange(InputArray a, bool quiet = true, CV_OUT Point* pos = 0,
                            double minVal = -DBL_MAX, double maxVal = DBL_MAX);

//! converts NaN's to the given number
CV_EXPORTS_W void patchNaNs(InputOutputArray a, double val = 0);

//! implements generalized matrix product algorithm GEMM from BLAS
CV_EXPORTS_W void gemm(InputArray src1, InputArray src2, double alpha,
                       InputArray src3, double beta, OutputArray dst, int flags = 0);

//! multiplies matrix by its transposition from the left or from the right
CV_EXPORTS_W void mulTransposed( InputArray src, OutputArray dst, bool aTa,
                                 InputArray delta = noArray(),
                                 double scale = 1, int dtype = -1 );

//! transposes the matrix
CV_EXPORTS_W void transpose(InputArray src, OutputArray dst);

//! performs affine transformation of each element of multi-channel input matrix
CV_EXPORTS_W void transform(InputArray src, OutputArray dst, InputArray m );

//! performs perspective transformation of each element of multi-channel input matrix
CV_EXPORTS_W void perspectiveTransform(InputArray src, OutputArray dst, InputArray m );

//! extends the symmetrical matrix from the lower half or from the upper half
CV_EXPORTS_W void completeSymm(InputOutputArray mtx, bool lowerToUpper = false);

//! initializes scaled identity matrix
CV_EXPORTS_W void setIdentity(InputOutputArray mtx, const Scalar& s = Scalar(1));

//! computes determinant of a square matrix
CV_EXPORTS_W double determinant(InputArray mtx);

//! computes trace of a matrix
CV_EXPORTS_W Scalar trace(InputArray mtx);

//! computes inverse or pseudo-inverse matrix
CV_EXPORTS_W double invert(InputArray src, OutputArray dst, int flags = DECOMP_LU);

//! solves linear system or a least-square problem
CV_EXPORTS_W bool solve(InputArray src1, InputArray src2,
                        OutputArray dst, int flags = DECOMP_LU);

//! sorts independently each matrix row or each matrix column
CV_EXPORTS_W void sort(InputArray src, OutputArray dst, int flags);

//! sorts independently each matrix row or each matrix column
CV_EXPORTS_W void sortIdx(InputArray src, OutputArray dst, int flags);

//! finds real roots of a cubic polynomial
CV_EXPORTS_W int solveCubic(InputArray coeffs, OutputArray roots);

//! finds real and complex roots of a polynomial
CV_EXPORTS_W double solvePoly(InputArray coeffs, OutputArray roots, int maxIters = 300);

//! finds eigenvalues and eigenvectors of a symmetric matrix
CV_EXPORTS_W bool eigen(InputArray src, OutputArray eigenvalues,
                        OutputArray eigenvectors = noArray());

//! computes covariation matrix of a set of samples
CV_EXPORTS void calcCovarMatrix( const Mat* samples, int nsamples, Mat& covar, Mat& mean,
                                 int flags, int ctype = CV_64F); //TODO: InputArrayOfArrays

//! computes covariation matrix of a set of samples
CV_EXPORTS_W void calcCovarMatrix( InputArray samples, OutputArray covar,
                                   InputOutputArray mean, int flags, int ctype = CV_64F);

CV_EXPORTS_W void PCACompute(InputArray data, InputOutputArray mean,
                             OutputArray eigenvectors, int maxComponents = 0);

CV_EXPORTS_W void PCACompute(InputArray data, InputOutputArray mean,
                             OutputArray eigenvectors, double retainedVariance);

CV_EXPORTS_W void PCAProject(InputArray data, InputArray mean,
                             InputArray eigenvectors, OutputArray result);

CV_EXPORTS_W void PCABackProject(InputArray data, InputArray mean,
                                 InputArray eigenvectors, OutputArray result);

//! computes SVD of src
CV_EXPORTS_W void SVDecomp( InputArray src, OutputArray w, OutputArray u, OutputArray vt, int flags = 0 );

//! performs back substitution for the previously computed SVD
CV_EXPORTS_W void SVBackSubst( InputArray w, InputArray u, InputArray vt,
                               InputArray rhs, OutputArray dst );

//! computes Mahalanobis distance between two vectors: sqrt((v1-v2)'*icovar*(v1-v2)), where icovar is the inverse covariation matrix
CV_EXPORTS_W double Mahalanobis(InputArray v1, InputArray v2, InputArray icovar);

//! performs forward or inverse 1D or 2D Discrete Fourier Transformation
CV_EXPORTS_W void dft(InputArray src, OutputArray dst, int flags = 0, int nonzeroRows = 0);

//! performs inverse 1D or 2D Discrete Fourier Transformation
CV_EXPORTS_W void idft(InputArray src, OutputArray dst, int flags = 0, int nonzeroRows = 0);

//! performs forward or inverse 1D or 2D Discrete Cosine Transformation
CV_EXPORTS_W void dct(InputArray src, OutputArray dst, int flags = 0);

//! performs inverse 1D or 2D Discrete Cosine Transformation
CV_EXPORTS_W void idct(InputArray src, OutputArray dst, int flags = 0);

//! computes element-wise product of the two Fourier spectrums. The second spectrum can optionally be conjugated before the multiplication
CV_EXPORTS_W void mulSpectrums(InputArray a, InputArray b, OutputArray c,
                               int flags, bool conjB = false);

//! computes the minimal vector size vecsize1 >= vecsize so that the dft() of the vector of length vecsize1 can be computed efficiently
CV_EXPORTS_W int getOptimalDFTSize(int vecsize);

//! clusters the input data using k-Means algorithm
CV_EXPORTS_W double kmeans( InputArray data, int K, InputOutputArray bestLabels,
                            TermCriteria criteria, int attempts,
                            int flags, OutputArray centers = noArray() );

//! returns the thread-local Random number generator
CV_EXPORTS RNG& theRNG();

//! fills array with uniformly-distributed random numbers from the range [low, high)
CV_EXPORTS_W void randu(InputOutputArray dst, InputArray low, InputArray high);

//! fills array with normally-distributed random numbers with the specified mean and the standard deviation
CV_EXPORTS_W void randn(InputOutputArray dst, InputArray mean, InputArray stddev);

//! shuffles the input array elements
CV_EXPORTS_W void randShuffle(InputOutputArray dst, double iterFactor = 1., RNG* rng = 0);

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
                PCA::DATA_AS_ROW, // indicate that the vectors
                                    // are stored as matrix rows
                                    // (use PCA::DATA_AS_COL if the vectors are
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
    enum { DATA_AS_ROW = 0,
           DATA_AS_COL = 1,
           USE_AVG     = 2
         };

    //! default constructor
    PCA();

    //! the constructor that performs PCA
    PCA(InputArray data, InputArray mean, int flags, int maxComponents = 0);
    PCA(InputArray data, InputArray mean, int flags, double retainedVariance);

    //! operator that performs PCA. The previously stored data, if any, is released
    PCA& operator()(InputArray data, InputArray mean, int flags, int maxComponents = 0);
    PCA& operator()(InputArray data, InputArray mean, int flags, double retainedVariance);

    //! projects vector from the original space to the principal components subspace
    Mat project(InputArray vec) const;

    //! projects vector from the original space to the principal components subspace
    void project(InputArray vec, OutputArray result) const;

    //! reconstructs the original vector from the projection
    Mat backProject(InputArray vec) const;

    //! reconstructs the original vector from the projection
    void backProject(InputArray vec, OutputArray result) const;

    //! write and load PCA matrix
    void write(FileStorage& fs ) const;
    void read(const FileNode& fs);

    Mat eigenvectors; //!< eigenvectors of the covariation matrix
    Mat eigenvalues; //!< eigenvalues of the covariation matrix
    Mat mean; //!< mean value subtracted before the projection and added after the back projection
};

// Linear Discriminant Analysis
class CV_EXPORTS LDA
{
public:
    // Initializes a LDA with num_components (default 0) and specifies how
    // samples are aligned (default dataAsRow=true).
    explicit LDA(int num_components = 0);

    // Initializes and performs a Discriminant Analysis with Fisher's
    // Optimization Criterion on given data in src and corresponding labels
    // in labels. If 0 (or less) number of components are given, they are
    // automatically determined for given data in computation.
    LDA(InputArrayOfArrays src, InputArray labels, int num_components = 0);

    // Serializes this object to a given filename.
    void save(const String& filename) const;

    // Deserializes this object from a given filename.
    void load(const String& filename);

    // Serializes this object to a given cv::FileStorage.
    void save(FileStorage& fs) const;

        // Deserializes this object from a given cv::FileStorage.
    void load(const FileStorage& node);

    // Destructor.
    ~LDA();

    //! Compute the discriminants for data in src and labels.
    void compute(InputArrayOfArrays src, InputArray labels);

    // Projects samples into the LDA subspace.
    Mat project(InputArray src);

    // Reconstructs projections from the LDA subspace.
    Mat reconstruct(InputArray src);

    // Returns the eigenvectors of this LDA.
    Mat eigenvectors() const { return _eigenvectors; }

    // Returns the eigenvalues of this LDA.
    Mat eigenvalues() const { return _eigenvalues; }

    static Mat subspaceProject(InputArray W, InputArray mean, InputArray src);
    static Mat subspaceReconstruct(InputArray W, InputArray mean, InputArray src);

protected:
    bool _dataAsRow;
    int _num_components;
    Mat _eigenvectors;
    Mat _eigenvalues;

    void lda(InputArrayOfArrays src, InputArray labels);
};

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
    enum { MODIFY_A = 1,
           NO_UV    = 2,
           FULL_UV  = 4
         };

    //! the default constructor
    SVD();

    //! the constructor that performs SVD
    SVD( InputArray src, int flags = 0 );

    //! the operator that performs SVD. The previously allocated SVD::u, SVD::w are SVD::vt are released.
    SVD& operator ()( InputArray src, int flags = 0 );

    //! decomposes matrix and stores the results to user-provided matrices
    static void compute( InputArray src, OutputArray w,
                         OutputArray u, OutputArray vt, int flags = 0 );

    //! computes singular values of a matrix
    static void compute( InputArray src, OutputArray w, int flags = 0 );

    //! performs back substitution
    static void backSubst( InputArray w, InputArray u,
                           InputArray vt, InputArray rhs,
                           OutputArray dst );

    //! finds dst = arg min_{|dst|=1} |m*dst|
    static void solveZ( InputArray src, OutputArray dst );

    //! performs back substitution, so that dst is the solution or pseudo-solution of m*dst = rhs, where m is the decomposed matrix
    void backSubst( InputArray rhs, OutputArray dst ) const;

    template<typename _Tp, int m, int n, int nm> static
    void compute( const Matx<_Tp, m, n>& a, Matx<_Tp, nm, 1>& w, Matx<_Tp, m, nm>& u, Matx<_Tp, n, nm>& vt );

    template<typename _Tp, int m, int n, int nm> static
    void compute( const Matx<_Tp, m, n>& a, Matx<_Tp, nm, 1>& w );

    template<typename _Tp, int m, int n, int nm, int nb> static
    void backSubst( const Matx<_Tp, nm, 1>& w, const Matx<_Tp, m, nm>& u, const Matx<_Tp, n, nm>& vt, const Matx<_Tp, m, nb>& rhs, Matx<_Tp, n, nb>& dst );

    Mat u, w, vt;
};



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
                  int connectivity = 8, bool leftToRight = false );
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
    CV_WRAP KDTree(InputArray points, bool copyAndReorderPoints = false);
    //! the full constructor that builds the search tree
    CV_WRAP KDTree(InputArray points, InputArray _labels,
                   bool copyAndReorderPoints = false);
    //! builds the search tree
    CV_WRAP void build(InputArray points, bool copyAndReorderPoints = false);
    //! builds the search tree
    CV_WRAP void build(InputArray points, InputArray labels,
                       bool copyAndReorderPoints = false);
    //! finds the K nearest neighbors of "vec" while looking at Emax (at most) leaves
    CV_WRAP int findNearest(InputArray vec, int K, int Emax,
                            OutputArray neighborsIdx,
                            OutputArray neighbors = noArray(),
                            OutputArray dist = noArray(),
                            OutputArray labels = noArray()) const;
    //! finds all the points from the initial set that belong to the specified box
    CV_WRAP void findOrthoRange(InputArray minBounds,
                                InputArray maxBounds,
                                OutputArray neighborsIdx,
                                OutputArray neighbors = noArray(),
                                OutputArray labels = noArray()) const;
    //! returns vectors with the specified indices
    CV_WRAP void getPoints(InputArray idx, OutputArray pts,
                           OutputArray labels = noArray()) const;
    //! return a vector with the specified index
    const float* getPoint(int ptidx, int* label = 0) const;
    //! returns the search space dimensionality
    CV_WRAP int dims() const;

    std::vector<Node> nodes; //!< all the tree nodes
    CV_PROP Mat points; //!< all the points. It can be a reordered copy of the input vector set or the original vector set.
    CV_PROP std::vector<int> labels; //!< the parallel array of labels.
    CV_PROP int maxDepth; //!< maximum depth of the search tree. Do not modify it
    CV_PROP_RW int normType; //!< type of the distance (cv::NORM_L1 or cv::NORM_L2) used for search. Initially set to cv::NORM_L2, but you can modify it
};



/*!
   Random Number Generator

   The class implements RNG using Multiply-with-Carry algorithm
*/
class CV_EXPORTS RNG
{
public:
    enum { UNIFORM = 0,
           NORMAL  = 1
         };

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
    void fill( InputOutputArray mat, int distType, InputArray a, InputArray b, bool saturateRange = false );
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



/////////////////////////////// Formatted output of cv::Mat ///////////////////////////

class CV_EXPORTS Formatted
{
public:
    virtual const char* next() = 0;
    virtual void reset() = 0;
    virtual ~Formatted();
};


class CV_EXPORTS Formatter
{
public:
    enum { FMT_DEFAULT = 0,
           FMT_MATLAB  = 1,
           FMT_CSV     = 2,
           FMT_PYTHON  = 3,
           FMT_NUMPY   = 4,
           FMT_C       = 5
         };

    virtual ~Formatter();

    virtual Ptr<Formatted> format(const Mat& mtx) const = 0;

    virtual void set32fPrecision(int p = 8) = 0;
    virtual void set64fPrecision(int p = 16) = 0;
    virtual void setMultiline(bool ml = true) = 0;

    static Ptr<Formatter> get(int fmt = FMT_DEFAULT);

};



//////////////////////////////////////// Algorithm ////////////////////////////////////

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
#include "opencv2/core/cvstd.inl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/optim.hpp"

#endif /*__OPENCV_CORE_HPP__*/
