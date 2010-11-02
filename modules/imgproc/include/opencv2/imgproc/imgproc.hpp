/*! \file imgproc.hpp
 \brief The Image Processing
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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_IMGPROC_HPP__
#define __OPENCV_IMGPROC_HPP__

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/types_c.h"

#ifdef __cplusplus

/*! \namespace cv
 Namespace where all the C++ OpenCV functionality resides
 */
namespace cv
{

//! various border interpolation methods
enum { BORDER_REPLICATE=IPL_BORDER_REPLICATE, BORDER_CONSTANT=IPL_BORDER_CONSTANT,
       BORDER_REFLECT=IPL_BORDER_REFLECT, BORDER_REFLECT_101=IPL_BORDER_REFLECT_101,
       BORDER_REFLECT101=BORDER_REFLECT_101, BORDER_WRAP=IPL_BORDER_WRAP,
       BORDER_TRANSPARENT, BORDER_DEFAULT=BORDER_REFLECT_101, BORDER_ISOLATED=16 };

//! 1D interpolation function: returns coordinate of the "donor" pixel for the specified location p. 
CV_EXPORTS_W int borderInterpolate( int p, int len, int borderType );

/*!
 The Base Class for 1D or Row-wise Filters
 
 This is the base class for linear or non-linear filters that process 1D data.
 In particular, such filters are used for the "horizontal" filtering parts in separable filters.
 
 Several functions in OpenCV return Ptr<BaseRowFilter> for the specific types of filters,
 and those pointers can be used directly or within cv::FilterEngine.
*/
class CV_EXPORTS BaseRowFilter
{
public:
    //! the default constructor
    BaseRowFilter();
    //! the destructor
    virtual ~BaseRowFilter();
    //! the filtering operator. Must be overrided in the derived classes. The horizontal border interpolation is done outside of the class.
    virtual void operator()(const uchar* src, uchar* dst,
                            int width, int cn) = 0;
    int ksize, anchor;
};


/*!
 The Base Class for Column-wise Filters
 
 This is the base class for linear or non-linear filters that process columns of 2D arrays.
 Such filters are used for the "vertical" filtering parts in separable filters.
 
 Several functions in OpenCV return Ptr<BaseColumnFilter> for the specific types of filters,
 and those pointers can be used directly or within cv::FilterEngine.
 
 Unlike cv::BaseRowFilter, cv::BaseColumnFilter may have some context information,
 i.e. box filter keeps the sliding sum of elements. To reset the state BaseColumnFilter::reset()
 must be called (e.g. the method is called by cv::FilterEngine)
 */    
class CV_EXPORTS BaseColumnFilter
{
public:
    //! the default constructor
    BaseColumnFilter();
    //! the destructor
    virtual ~BaseColumnFilter();
    //! the filtering operator. Must be overrided in the derived classes. The vertical border interpolation is done outside of the class.
    virtual void operator()(const uchar** src, uchar* dst, int dststep,
                            int dstcount, int width) = 0;
    //! resets the internal buffers, if any
    virtual void reset();
    int ksize, anchor;
};

/*!
 The Base Class for Non-Separable 2D Filters.
 
 This is the base class for linear or non-linear 2D filters.
 
 Several functions in OpenCV return Ptr<BaseFilter> for the specific types of filters,
 and those pointers can be used directly or within cv::FilterEngine.
 
 Similar to cv::BaseColumnFilter, the class may have some context information,
 that should be reset using BaseFilter::reset() method before processing the new array.
*/ 
class CV_EXPORTS BaseFilter
{
public:
    //! the default constructor
    BaseFilter();
    //! the destructor
    virtual ~BaseFilter();
    //! the filtering operator. The horizontal and the vertical border interpolation is done outside of the class.
    virtual void operator()(const uchar** src, uchar* dst, int dststep,
                            int dstcount, int width, int cn) = 0;
    //! resets the internal buffers, if any
    virtual void reset();
    Size ksize;
    Point anchor;
};

/*!
 The Main Class for Image Filtering.
 
 The class can be used to apply an arbitrary filtering operation to an image.
 It contains all the necessary intermediate buffers, it computes extrapolated values
 of the "virtual" pixels outside of the image etc.
 Pointers to the initialized cv::FilterEngine instances
 are returned by various OpenCV functions, such as cv::createSeparableLinearFilter(),
 cv::createLinearFilter(), cv::createGaussianFilter(), cv::createDerivFilter(),
 cv::createBoxFilter() and cv::createMorphologyFilter().
 
 Using the class you can process large images by parts and build complex pipelines
 that include filtering as some of the stages. If all you need is to apply some pre-defined
 filtering operation, you may use cv::filter2D(), cv::erode(), cv::dilate() etc.
 functions that create FilterEngine internally.
 
 Here is the example on how to use the class to implement Laplacian operator, which is the sum of
 second-order derivatives. More complex variant for different types is implemented in cv::Laplacian().
 
 \code
 void laplace_f(const Mat& src, Mat& dst)
 {
     CV_Assert( src.type() == CV_32F );
     // make sure the destination array has the proper size and type
     dst.create(src.size(), src.type());
     
     // get the derivative and smooth kernels for d2I/dx2.
     // for d2I/dy2 we could use the same kernels, just swapped
     Mat kd, ks;
     getSobelKernels( kd, ks, 2, 0, ksize, false, ktype );
     
     // let's process 10 source rows at once
     int DELTA = std::min(10, src.rows);
     Ptr<FilterEngine> Fxx = createSeparableLinearFilter(src.type(),
     dst.type(), kd, ks, Point(-1,-1), 0, borderType, borderType, Scalar() ); 
     Ptr<FilterEngine> Fyy = createSeparableLinearFilter(src.type(),
     dst.type(), ks, kd, Point(-1,-1), 0, borderType, borderType, Scalar() );
     
     int y = Fxx->start(src), dsty = 0, dy = 0;
     Fyy->start(src);
     const uchar* sptr = src.data + y*src.step;
     
     // allocate the buffers for the spatial image derivatives;
     // the buffers need to have more than DELTA rows, because at the
     // last iteration the output may take max(kd.rows-1,ks.rows-1)
     // rows more than the input.
     Mat Ixx( DELTA + kd.rows - 1, src.cols, dst.type() );
     Mat Iyy( DELTA + kd.rows - 1, src.cols, dst.type() );
     
     // inside the loop we always pass DELTA rows to the filter
     // (note that the "proceed" method takes care of possibe overflow, since
     // it was given the actual image height in the "start" method)
     // on output we can get:
     //  * < DELTA rows (the initial buffer accumulation stage)
     //  * = DELTA rows (settled state in the middle)
     //  * > DELTA rows (then the input image is over, but we generate
     //                  "virtual" rows using the border mode and filter them)
     // this variable number of output rows is dy.
     // dsty is the current output row.
     // sptr is the pointer to the first input row in the portion to process
     for( ; dsty < dst.rows; sptr += DELTA*src.step, dsty += dy )
     {
         Fxx->proceed( sptr, (int)src.step, DELTA, Ixx.data, (int)Ixx.step );
         dy = Fyy->proceed( sptr, (int)src.step, DELTA, d2y.data, (int)Iyy.step );
         if( dy > 0 )
         {
             Mat dstripe = dst.rowRange(dsty, dsty + dy);
             add(Ixx.rowRange(0, dy), Iyy.rowRange(0, dy), dstripe);
         }
     }
 }
 \endcode
*/
class CV_EXPORTS FilterEngine
{
public:
    //! the default constructor
    FilterEngine();
    //! the full constructor. Either _filter2D or both _rowFilter and _columnFilter must be non-empty.
    FilterEngine(const Ptr<BaseFilter>& _filter2D,
                 const Ptr<BaseRowFilter>& _rowFilter,
                 const Ptr<BaseColumnFilter>& _columnFilter,
                 int srcType, int dstType, int bufType,
                 int _rowBorderType=BORDER_REPLICATE,
                 int _columnBorderType=-1,
                 const Scalar& _borderValue=Scalar());
    //! the destructor
    virtual ~FilterEngine();
    //! reinitializes the engine. The previously assigned filters are released.
    void init(const Ptr<BaseFilter>& _filter2D,
              const Ptr<BaseRowFilter>& _rowFilter,
              const Ptr<BaseColumnFilter>& _columnFilter,
              int srcType, int dstType, int bufType,
              int _rowBorderType=BORDER_REPLICATE, int _columnBorderType=-1,
              const Scalar& _borderValue=Scalar());
    //! starts filtering of the specified ROI of an image of size wholeSize. 
    virtual int start(Size wholeSize, Rect roi, int maxBufRows=-1);
    //! starts filtering of the specified ROI of the specified image.
    virtual int start(const Mat& src, const Rect& srcRoi=Rect(0,0,-1,-1),
                      bool isolated=false, int maxBufRows=-1);
    //! processes the next srcCount rows of the image.
    virtual int proceed(const uchar* src, int srcStep, int srcCount,
                        uchar* dst, int dstStep);
    //! applies filter to the specified ROI of the image. if srcRoi=(0,0,-1,-1), the whole image is filtered.
    virtual void apply( const Mat& src, Mat& dst,
                        const Rect& srcRoi=Rect(0,0,-1,-1),
                        Point dstOfs=Point(0,0),
                        bool isolated=false);
    //! returns true if the filter is separable
    bool isSeparable() const { return (const BaseFilter*)filter2D == 0; }
    //! returns the number 
    int remainingInputRows() const;
    int remainingOutputRows() const;
    
    int srcType, dstType, bufType;
    Size ksize;
    Point anchor;
    int maxWidth;
    Size wholeSize;
    Rect roi;
    int dx1, dx2;
    int rowBorderType, columnBorderType;
    vector<int> borderTab;
    int borderElemSize;
    vector<uchar> ringBuf;
    vector<uchar> srcRow;
    vector<uchar> constBorderValue;
    vector<uchar> constBorderRow;
    int bufStep, startY, startY0, endY, rowCount, dstY;
    vector<uchar*> rows;
    
    Ptr<BaseFilter> filter2D;
    Ptr<BaseRowFilter> rowFilter;
    Ptr<BaseColumnFilter> columnFilter;
};

//! type of the kernel
enum { KERNEL_GENERAL=0, KERNEL_SYMMETRICAL=1, KERNEL_ASYMMETRICAL=2,
       KERNEL_SMOOTH=4, KERNEL_INTEGER=8 };

//! returns type (one of KERNEL_*) of 1D or 2D kernel specified by its coefficients.
CV_EXPORTS int getKernelType(const Mat& kernel, Point anchor);

//! returns the primitive row filter with the specified kernel
CV_EXPORTS Ptr<BaseRowFilter> getLinearRowFilter(int srcType, int bufType,
                                            const Mat& kernel, int anchor,
                                            int symmetryType);

//! returns the primitive column filter with the specified kernel
CV_EXPORTS Ptr<BaseColumnFilter> getLinearColumnFilter(int bufType, int dstType,
                                            const Mat& kernel, int anchor,
                                            int symmetryType, double delta=0,
                                            int bits=0);

//! returns 2D filter with the specified kernel
CV_EXPORTS Ptr<BaseFilter> getLinearFilter(int srcType, int dstType,
                                           const Mat& kernel,
                                           Point anchor=Point(-1,-1),
                                           double delta=0, int bits=0);

//! returns the separable linear filter engine
CV_EXPORTS Ptr<FilterEngine> createSeparableLinearFilter(int srcType, int dstType,
                          const Mat& rowKernel, const Mat& columnKernel,
                          Point _anchor=Point(-1,-1), double delta=0,
                          int _rowBorderType=BORDER_DEFAULT,
                          int _columnBorderType=-1,
                          const Scalar& _borderValue=Scalar());

//! returns the non-separable linear filter engine
CV_EXPORTS Ptr<FilterEngine> createLinearFilter(int srcType, int dstType,
                 const Mat& kernel, Point _anchor=Point(-1,-1),
                 double delta=0, int _rowBorderType=BORDER_DEFAULT,
                 int _columnBorderType=-1, const Scalar& _borderValue=Scalar());

//! returns the Gaussian kernel with the specified parameters
CV_EXPORTS_W Mat getGaussianKernel( int ksize, double sigma, int ktype=CV_64F );

//! returns the Gaussian filter engine
CV_EXPORTS Ptr<FilterEngine> createGaussianFilter( int type, Size ksize,
                                    double sigma1, double sigma2=0,
                                    int borderType=BORDER_DEFAULT);
//! initializes kernels of the generalized Sobel operator
CV_EXPORTS_W void getDerivKernels( CV_OUT Mat& kx, CV_OUT Mat& ky,
                                 int dx, int dy, int ksize,
                                 bool normalize=false, int ktype=CV_32F );
//! returns filter engine for the generalized Sobel operator
CV_EXPORTS Ptr<FilterEngine> createDerivFilter( int srcType, int dstType,
                                        int dx, int dy, int ksize,
                                        int borderType=BORDER_DEFAULT );
//! returns horizontal 1D box filter 
CV_EXPORTS Ptr<BaseRowFilter> getRowSumFilter(int srcType, int sumType,
                                              int ksize, int anchor=-1);
//! returns vertical 1D box filter
CV_EXPORTS Ptr<BaseColumnFilter> getColumnSumFilter( int sumType, int dstType,
                                                     int ksize, int anchor=-1,
                                                     double scale=1);
//! returns box filter engine
CV_EXPORTS Ptr<FilterEngine> createBoxFilter( int srcType, int dstType, Size ksize,
                                              Point anchor=Point(-1,-1),
                                              bool normalize=true,
                                              int borderType=BORDER_DEFAULT);
//! type of morphological operation
enum { MORPH_ERODE=0, MORPH_DILATE=1, MORPH_OPEN=2, MORPH_CLOSE=3,
       MORPH_GRADIENT=4, MORPH_TOPHAT=5, MORPH_BLACKHAT=6 };

//! returns horizontal 1D morphological filter
CV_EXPORTS Ptr<BaseRowFilter> getMorphologyRowFilter(int op, int type, int ksize, int anchor=-1);
//! returns vertical 1D morphological filter
CV_EXPORTS Ptr<BaseColumnFilter> getMorphologyColumnFilter(int op, int type, int ksize, int anchor=-1);
//! returns 2D morphological filter
CV_EXPORTS Ptr<BaseFilter> getMorphologyFilter(int op, int type, const Mat& kernel,
                                               Point anchor=Point(-1,-1));
    
//! returns "magic" border value for erosion and dilation. It is automatically transformed to Scalar::all(-DBL_MAX) for dilation.
static inline Scalar morphologyDefaultBorderValue() { return Scalar::all(DBL_MAX); }

//! returns morphological filter engine. Only MORPH_ERODE and MORPH_DILATE are supported.
CV_EXPORTS Ptr<FilterEngine> createMorphologyFilter(int op, int type, const Mat& kernel,
                    Point anchor=Point(-1,-1), int _rowBorderType=BORDER_CONSTANT,
                    int _columnBorderType=-1,
                    const Scalar& _borderValue=morphologyDefaultBorderValue());

//! shape of the structuring element
enum { MORPH_RECT=0, MORPH_CROSS=1, MORPH_ELLIPSE=2 };
//! returns structuring element of the specified shape and size
CV_EXPORTS_W Mat getStructuringElement(int shape, Size ksize, Point anchor=Point(-1,-1));

template<> CV_EXPORTS void Ptr<IplConvKernel>::delete_obj();

//! copies 2D array to a larger destination array with extrapolation of the outer part of src using the specified border mode 
CV_EXPORTS_W void copyMakeBorder( const Mat& src, CV_OUT Mat& dst,
                                int top, int bottom, int left, int right,
                                int borderType, const Scalar& value=Scalar() );

//! smooths the image using median filter.
CV_EXPORTS_W void medianBlur( const Mat& src, CV_OUT Mat& dst, int ksize );
//! smooths the image using Gaussian filter.
CV_EXPORTS_W void GaussianBlur( const Mat& src, CV_OUT Mat& dst, Size ksize,
                              double sigma1, double sigma2=0,
                              int borderType=BORDER_DEFAULT );
//! smooths the image using bilateral filter
CV_EXPORTS_W void bilateralFilter( const Mat& src, CV_OUT Mat& dst, int d,
                                 double sigmaColor, double sigmaSpace,
                                 int borderType=BORDER_DEFAULT );
//! smooths the image using the box filter. Each pixel is processed in O(1) time
CV_EXPORTS_W void boxFilter( const Mat& src, CV_OUT Mat& dst, int ddepth,
                           Size ksize, Point anchor=Point(-1,-1),
                           bool normalize=true,
                           int borderType=BORDER_DEFAULT );
//! a synonym for normalized box filter
CV_WRAP static inline void blur( const Mat& src, CV_OUT Mat& dst,
                         Size ksize, Point anchor=Point(-1,-1),
                         int borderType=BORDER_DEFAULT )
{
    boxFilter( src, dst, -1, ksize, anchor, true, borderType );
}

//! applies non-separable 2D linear filter to the image
CV_EXPORTS_W void filter2D( const Mat& src, CV_OUT Mat& dst, int ddepth,
                          const Mat& kernel, Point anchor=Point(-1,-1),
                          double delta=0, int borderType=BORDER_DEFAULT );

//! applies separable 2D linear filter to the image
CV_EXPORTS_W void sepFilter2D( const Mat& src, CV_OUT Mat& dst, int ddepth,
                             const Mat& kernelX, const Mat& kernelY,
                             Point anchor=Point(-1,-1),
                             double delta=0, int borderType=BORDER_DEFAULT );
    
//! applies generalized Sobel operator to the image
CV_EXPORTS_W void Sobel( const Mat& src, CV_OUT Mat& dst, int ddepth,
                       int dx, int dy, int ksize=3,
                       double scale=1, double delta=0,
                       int borderType=BORDER_DEFAULT );

//! applies the vertical or horizontal Scharr operator to the image
CV_EXPORTS_W void Scharr( const Mat& src, CV_OUT Mat& dst, int ddepth,
                        int dx, int dy, double scale=1, double delta=0,
                        int borderType=BORDER_DEFAULT );

//! applies Laplacian operator to the image
CV_EXPORTS_W void Laplacian( const Mat& src, CV_OUT Mat& dst, int ddepth,
                           int ksize=1, double scale=1, double delta=0,
                           int borderType=BORDER_DEFAULT );

//! applies Canny edge detector and produces the edge map.
CV_EXPORTS_W void Canny( const Mat& image, CV_OUT Mat& edges,
                       double threshold1, double threshold2,
                       int apertureSize=3, bool L2gradient=false );

//! computes minimum eigen value of 2x2 derivative covariation matrix at each pixel - the cornerness criteria
CV_EXPORTS_W void cornerMinEigenVal( const Mat& src, CV_OUT Mat& dst,
                                   int blockSize, int ksize=3,
                                   int borderType=BORDER_DEFAULT );

//! computes Harris cornerness criteria at each image pixel
CV_EXPORTS_W void cornerHarris( const Mat& src, CV_OUT Mat& dst, int blockSize,
                              int ksize, double k,
                              int borderType=BORDER_DEFAULT );

//! computes both eigenvalues and the eigenvectors of 2x2 derivative covariation matrix  at each pixel. The output is stored as 6-channel matrix.
CV_EXPORTS_W void cornerEigenValsAndVecs( const Mat& src, CV_OUT Mat& dst,
                                        int blockSize, int ksize,
                                        int borderType=BORDER_DEFAULT );

//! computes another complex cornerness criteria at each pixel
CV_EXPORTS_W void preCornerDetect( const Mat& src, CV_OUT Mat& dst, int ksize,
                                 int borderType=BORDER_DEFAULT );

//! adjusts the corner locations with sub-pixel accuracy to maximize the certain cornerness criteria
CV_EXPORTS void cornerSubPix( const Mat& image, vector<Point2f>& corners,
                              Size winSize, Size zeroZone,
                              TermCriteria criteria );

//! finds the strong enough corners where the cornerMinEigenVal() or cornerHarris() report the local maxima
CV_EXPORTS_W void goodFeaturesToTrack( const Mat& image, CV_OUT vector<Point2f>& corners,
                                     int maxCorners, double qualityLevel, double minDistance,
                                     const Mat& mask=Mat(), int blockSize=3,
                                     bool useHarrisDetector=false, double k=0.04 );

//! finds lines in the black-n-white image using the standard or pyramid Hough transform
CV_EXPORTS_W void HoughLines( const Mat& image, CV_OUT vector<Vec2f>& lines,
                            double rho, double theta, int threshold,
                            double srn=0, double stn=0 );

//! finds line segments in the black-n-white image using probabalistic Hough transform
CV_EXPORTS_W void HoughLinesP( Mat& image, CV_OUT vector<Vec4i>& lines,
                             double rho, double theta, int threshold,
                             double minLineLength=0, double maxLineGap=0 );

//! finds circles in the grayscale image using 2+1 gradient Hough transform 
CV_EXPORTS_W void HoughCircles( const Mat& image, CV_OUT vector<Vec3f>& circles,
                              int method, double dp, double minDist,
                              double param1=100, double param2=100,
                              int minRadius=0, int maxRadius=0 );

//! erodes the image (applies the local minimum operator)
CV_EXPORTS_W void erode( const Mat& src, CV_OUT Mat& dst, const Mat& kernel,
                       Point anchor=Point(-1,-1), int iterations=1,
                       int borderType=BORDER_CONSTANT,
                       const Scalar& borderValue=morphologyDefaultBorderValue() );
    
//! dilates the image (applies the local maximum operator)
CV_EXPORTS_W void dilate( const Mat& src, CV_OUT Mat& dst, const Mat& kernel,
                        Point anchor=Point(-1,-1), int iterations=1,
                        int borderType=BORDER_CONSTANT,
                        const Scalar& borderValue=morphologyDefaultBorderValue() );
    
//! applies an advanced morphological operation to the image
CV_EXPORTS_W void morphologyEx( const Mat& src, CV_OUT Mat& dst,
                              int op, const Mat& kernel,
                              Point anchor=Point(-1,-1), int iterations=1,
                              int borderType=BORDER_CONSTANT,
                              const Scalar& borderValue=morphologyDefaultBorderValue() );

//! interpolation algorithm
enum
{
    INTER_NEAREST=0, //!< nearest neighbor interpolation
    INTER_LINEAR=1, //!< bilinear interpolation
    INTER_CUBIC=2, //!< bicubic interpolation
    INTER_AREA=3, //!< area-based (or super) interpolation
    INTER_LANCZOS4=4, //!< Lanczos interpolation over 8x8 neighborhood
    INTER_MAX=7,
    WARP_INVERSE_MAP=16
};

//! resizes the image
CV_EXPORTS_W void resize( const Mat& src, CV_OUT Mat& dst,
                        Size dsize, double fx=0, double fy=0,
                        int interpolation=INTER_LINEAR );

//! warps the image using affine transformation
CV_EXPORTS_W void warpAffine( const Mat& src, CV_OUT Mat& dst,
                            const Mat& M, Size dsize,
                            int flags=INTER_LINEAR,
                            int borderMode=BORDER_CONSTANT,
                            const Scalar& borderValue=Scalar());
    
//! warps the image using perspective transformation
CV_EXPORTS_W void warpPerspective( const Mat& src, CV_OUT Mat& dst,
                                 const Mat& M, Size dsize,
                                 int flags=INTER_LINEAR,
                                 int borderMode=BORDER_CONSTANT,
                                 const Scalar& borderValue=Scalar());

enum { INTER_BITS=5, INTER_BITS2=INTER_BITS*2,
    INTER_TAB_SIZE=(1<<INTER_BITS),
    INTER_TAB_SIZE2=INTER_TAB_SIZE*INTER_TAB_SIZE };    

//! warps the image using the precomputed maps. The maps are stored in either floating-point or integer fixed-point format
CV_EXPORTS_W void remap( const Mat& src, CV_OUT Mat& dst, const Mat& map1, const Mat& map2,
                       int interpolation, int borderMode=BORDER_CONSTANT,
                       const Scalar& borderValue=Scalar());

//! converts maps for remap from floating-point to fixed-point format or backwards
CV_EXPORTS_W void convertMaps( const Mat& map1, const Mat& map2,
                             CV_OUT Mat& dstmap1, CV_OUT Mat& dstmap2,
                             int dstmap1type, bool nninterpolation=false );
                             
//! returns 2x3 affine transformation matrix for the planar rotation.
CV_EXPORTS_W Mat getRotationMatrix2D( Point2f center, double angle, double scale );
//! returns 3x3 perspective transformation for the corresponding 4 point pairs.
CV_EXPORTS Mat getPerspectiveTransform( const Point2f src[], const Point2f dst[] );
//! returns 2x3 affine transformation for the corresponding 3 point pairs.
CV_EXPORTS Mat getAffineTransform( const Point2f src[], const Point2f dst[] );
//! computes 2x3 affine transformation matrix that is inverse to the specified 2x3 affine transformation.
CV_EXPORTS_W void invertAffineTransform( const Mat& M, CV_OUT Mat& iM );

//! extracts rectangle from the image at sub-pixel location
CV_EXPORTS_W void getRectSubPix( const Mat& image, Size patchSize,
                               Point2f center, CV_OUT Mat& patch, int patchType=-1 );

//! computes the integral image
CV_EXPORTS_W void integral( const Mat& src, CV_OUT Mat& sum, int sdepth=-1 );
//! computes the integral image and integral for the squared image
CV_EXPORTS_AS(integral2) void integral( const Mat& src, CV_OUT Mat& sum, CV_OUT Mat& sqsum, int sdepth=-1 );
//! computes the integral image, integral for the squared image and the tilted integral image
CV_EXPORTS_AS(integral3) void integral( const Mat& src, CV_OUT Mat& sum, CV_OUT Mat& sqsum, CV_OUT Mat& tilted, int sdepth=-1 );

//! adds image to the accumulator (dst += src). Unlike cv::add, dst and src can have different types.
CV_EXPORTS_W void accumulate( const Mat& src, CV_IN_OUT Mat& dst, const Mat& mask=Mat() );
//! adds squared src image to the accumulator (dst += src*src).
CV_EXPORTS_W void accumulateSquare( const Mat& src, CV_IN_OUT Mat& dst, const Mat& mask=Mat() );
//! adds product of the 2 images to the accumulator (dst += src1*src2).
CV_EXPORTS_W void accumulateProduct( const Mat& src1, const Mat& src2,
                                   CV_IN_OUT Mat& dst, const Mat& mask=Mat() );
//! updates the running average (dst = dst*(1-alpha) + src*alpha)
CV_EXPORTS_W void accumulateWeighted( const Mat& src, CV_IN_OUT Mat& dst,
                                      double alpha, const Mat& mask=Mat() );
    
//! type of the threshold operation
enum { THRESH_BINARY=0, THRESH_BINARY_INV=1, THRESH_TRUNC=2, THRESH_TOZERO=3,
       THRESH_TOZERO_INV=4, THRESH_MASK=7, THRESH_OTSU=8 };

//! applies fixed threshold to the image
CV_EXPORTS_W double threshold( const Mat& src, CV_OUT Mat& dst, double thresh, double maxval, int type );

//! adaptive threshold algorithm
enum { ADAPTIVE_THRESH_MEAN_C=0, ADAPTIVE_THRESH_GAUSSIAN_C=1 };

//! applies variable (adaptive) threshold to the image
CV_EXPORTS_W void adaptiveThreshold( const Mat& src, CV_OUT Mat& dst, double maxValue,
                                   int adaptiveMethod, int thresholdType,
                                   int blockSize, double C );

//! smooths and downsamples the image
CV_EXPORTS_W void pyrDown( const Mat& src, CV_OUT Mat& dst, const Size& dstsize=Size());
//! upsamples and smoothes the image
CV_EXPORTS_W void pyrUp( const Mat& src, CV_OUT Mat& dst, const Size& dstsize=Size());
//! builds the gaussian pyramid using pyrDown() as a basic operation
CV_EXPORTS void buildPyramid( const Mat& src, CV_OUT vector<Mat>& dst, int maxlevel );

//! corrects lens distortion for the given camera matrix and distortion coefficients
CV_EXPORTS_W void undistort( const Mat& src, CV_OUT Mat& dst, const Mat& cameraMatrix,
                           const Mat& distCoeffs, const Mat& newCameraMatrix=Mat() );
//! initializes maps for cv::remap() to correct lens distortion and optionally rectify the image
CV_EXPORTS_W void initUndistortRectifyMap( const Mat& cameraMatrix, const Mat& distCoeffs,
                           const Mat& R, const Mat& newCameraMatrix,
                           Size size, int m1type, CV_OUT Mat& map1, CV_OUT Mat& map2 );

enum
{
    PROJ_SPHERICAL_ORTHO = 0,
    PROJ_SPHERICAL_EQRECT = 1
};    
    
//! initializes maps for cv::remap() for wide-angle
CV_EXPORTS_W float initWideAngleProjMap( const Mat& cameraMatrix, const Mat& distCoeffs,
                                       Size imageSize, int destImageWidth,
                                       int m1type, CV_OUT Mat& map1, CV_OUT Mat& map2,
                                       int projType=PROJ_SPHERICAL_EQRECT, double alpha=0);
    
//! returns the default new camera matrix (by default it is the same as cameraMatrix unless centerPricipalPoint=true)
CV_EXPORTS_W Mat getDefaultNewCameraMatrix( const Mat& cameraMatrix, Size imgsize=Size(),
                                          bool centerPrincipalPoint=false );
//! returns points' coordinates after lens distortion correction
CV_EXPORTS void undistortPoints( const Mat& src, CV_OUT vector<Point2f>& dst,
                                 const Mat& cameraMatrix, const Mat& distCoeffs,
                                 const Mat& R=Mat(), const Mat& P=Mat());
//! returns points' coordinates after lens distortion correction
CV_EXPORTS_W void undistortPoints( const Mat& src, CV_OUT Mat& dst,
                                 const Mat& cameraMatrix, const Mat& distCoeffs,
                                 const Mat& R=Mat(), const Mat& P=Mat());

template<> CV_EXPORTS void Ptr<CvHistogram>::delete_obj();
    
//! computes the joint dense histogram for a set of images.
CV_EXPORTS void calcHist( const Mat* images, int nimages,
                          const int* channels, const Mat& mask,
                          Mat& hist, int dims, const int* histSize,
                          const float** ranges, bool uniform=true, bool accumulate=false );

//! computes the joint sparse histogram for a set of images.
CV_EXPORTS void calcHist( const Mat* images, int nimages,
                          const int* channels, const Mat& mask,
                          SparseMat& hist, int dims,
                          const int* histSize, const float** ranges,
                          bool uniform=true, bool accumulate=false );

//! computes back projection for the set of images
CV_EXPORTS void calcBackProject( const Mat* images, int nimages,
                                 const int* channels, const Mat& hist,
                                 Mat& backProject, const float** ranges,
                                 double scale=1, bool uniform=true );

//! computes back projection for the set of images
CV_EXPORTS void calcBackProject( const Mat* images, int nimages,
                                 const int* channels, const SparseMat& hist, 
                                 Mat& backProject, const float** ranges,
                                 double scale=1, bool uniform=true );

//! compares two histograms stored in dense arrays
CV_EXPORTS_W double compareHist( const Mat& H1, const Mat& H2, int method );

//! compares two histograms stored in sparse arrays
CV_EXPORTS double compareHist( const SparseMat& H1, const SparseMat& H2, int method );

//! normalizes the grayscale image brightness and contrast by normalizing its histogram
CV_EXPORTS_W void equalizeHist( const Mat& src, CV_OUT Mat& dst );

//! segments the image using watershed algorithm
CV_EXPORTS_W void watershed( const Mat& image, Mat& markers );

//! filters image using meanshift algorithm
CV_EXPORTS_W void pyrMeanShiftFiltering( const Mat& src, CV_OUT Mat& dst,
                    double sp, double sr, int maxLevel=1,
                    TermCriteria termcrit=TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,5,1) );

//! class of the pixel in GrabCut algorithm
enum { GC_BGD    = 0,  //!< background
       GC_FGD    = 1,  //!< foreground
       GC_PR_BGD = 2,  //!< most probably background
       GC_PR_FGD = 3   //!< most probably foreground 
     };

//! GrabCut algorithm flags
enum { GC_INIT_WITH_RECT  = 0,
       GC_INIT_WITH_MASK  = 1,
       GC_EVAL            = 2
     };

//! segments the image using GrabCut algorithm
CV_EXPORTS_W void grabCut( const Mat& img, Mat& mask, Rect rect, 
                         Mat& bgdModel, Mat& fgdModel,
                         int iterCount, int mode = GC_EVAL );

//! the inpainting algorithm
enum
{
    INPAINT_NS=0, // Navier-Stokes algorithm
    INPAINT_TELEA=1 // A. Telea algorithm
};

//! restores the damaged image areas using one of the available intpainting algorithms
CV_EXPORTS_W void inpaint( const Mat& src, const Mat& inpaintMask,
                         CV_OUT Mat& dst, double inpaintRange, int flags );

//! builds the discrete Voronoi diagram
CV_EXPORTS_AS(distanceTransformWithLabels)
    void distanceTransform( const Mat& src, CV_OUT Mat& dst, Mat& labels,
                            int distanceType, int maskSize );

//! computes the distance transform map
CV_EXPORTS_W void distanceTransform( const Mat& src, CV_OUT Mat& dst,
                                   int distanceType, int maskSize );

enum { FLOODFILL_FIXED_RANGE = 1 << 16,
       FLOODFILL_MASK_ONLY = 1 << 17 };

//! fills the semi-uniform image region starting from the specified seed point
CV_EXPORTS_W int floodFill( Mat& image,
                          Point seedPoint, Scalar newVal, CV_OUT Rect* rect=0,
                          Scalar loDiff=Scalar(), Scalar upDiff=Scalar(),
                          int flags=4 );

//! fills the semi-uniform image region and/or the mask starting from the specified seed point
CV_EXPORTS_AS(floodFillMask) int floodFill( Mat& image, Mat& mask,
                          Point seedPoint, Scalar newVal, CV_OUT Rect* rect=0,
                          Scalar loDiff=Scalar(), Scalar upDiff=Scalar(),
                          int flags=4 );

//! converts image from one color space to another
CV_EXPORTS_W void cvtColor( const Mat& src, CV_OUT Mat& dst, int code, int dstCn=0 );

//! raster image moments
class CV_EXPORTS_W_MAP Moments
{
public:
    //! the default constructor
    Moments();
    //! the full constructor
    Moments(double m00, double m10, double m01, double m20, double m11,
            double m02, double m30, double m21, double m12, double m03 );
    //! the conversion from CvMoments
    Moments( const CvMoments& moments );
    //! the conversion to CvMoments
    operator CvMoments() const;
    
    //! spatial moments
    CV_PROP_RW double  m00, m10, m01, m20, m11, m02, m30, m21, m12, m03;
    //! central moments
    CV_PROP_RW double  mu20, mu11, mu02, mu30, mu21, mu12, mu03;
    //! central normalized moments
    CV_PROP_RW double  nu20, nu11, nu02, nu30, nu21, nu12, nu03;
};

//! computes moments of the rasterized shape or a vector of points
CV_EXPORTS_W Moments moments( const Mat& array, bool binaryImage=false );

//! computes 7 Hu invariants from the moments
CV_EXPORTS void HuMoments( const Moments& moments, double hu[7] );

//! type of the template matching operation
enum { TM_SQDIFF=0, TM_SQDIFF_NORMED=1, TM_CCORR=2, TM_CCORR_NORMED=3, TM_CCOEFF=4, TM_CCOEFF_NORMED=5 };

//! computes the proximity map for the raster template and the image where the template is searched for
CV_EXPORTS_W void matchTemplate( const Mat& image, const Mat& templ, CV_OUT Mat& result, int method );

//! mode of the contour retrieval algorithm
enum
{
    RETR_EXTERNAL=0, //!< retrieve only the most external (top-level) contours
    RETR_LIST=1, //!< retrieve all the contours without any hierarchical information
    RETR_CCOMP=2, //!< retrieve the connected components (that can possibly be nested)
    RETR_TREE=3 //!< retrieve all the contours and the whole hierarchy
};

//! the contour approximation algorithm
enum
{
    CHAIN_APPROX_NONE=0,
    CHAIN_APPROX_SIMPLE=1,
    CHAIN_APPROX_TC89_L1=2,
    CHAIN_APPROX_TC89_KCOS=3
};

//! retrieves contours and the hierarchical information from black-n-white image.
CV_EXPORTS void findContours( Mat& image, CV_OUT vector<vector<Point> >& contours,
                              vector<Vec4i>& hierarchy, int mode,
                              int method, Point offset=Point());

//! retrieves contours from black-n-white image.
CV_EXPORTS void findContours( Mat& image, CV_OUT vector<vector<Point> >& contours,
                              int mode, int method, Point offset=Point());

//! draws contours in the image
CV_EXPORTS void drawContours( Mat& image, const vector<vector<Point> >& contours,
                              int contourIdx, const Scalar& color,
                              int thickness=1, int lineType=8,
                              const vector<Vec4i>& hierarchy=vector<Vec4i>(),
                              int maxLevel=INT_MAX, Point offset=Point() );

//! approximates contour or a curve using Douglas-Peucker algorithm
CV_EXPORTS void approxPolyDP( const Mat& curve,
                              CV_OUT vector<Point>& approxCurve,
                              double epsilon, bool closed );
//! approximates contour or a curve using Douglas-Peucker algorithm
CV_EXPORTS void approxPolyDP( const Mat& curve,
                              CV_OUT vector<Point2f>& approxCurve,
                              double epsilon, bool closed );
//! computes the contour perimeter (closed=true) or a curve length
CV_EXPORTS_W double arcLength( const Mat& curve, bool closed );
//! computes the bounding rectangle for a contour
CV_EXPORTS_W Rect boundingRect( const Mat& points );
//! computes the contour area
CV_EXPORTS_W double contourArea( const Mat& contour, bool oriented=false );
//! computes the minimal rotated rectangle for a set of points
CV_EXPORTS_W RotatedRect minAreaRect( const Mat& points );
//! computes the minimal enclosing circle for a set of points
CV_EXPORTS_W void minEnclosingCircle( const Mat& points,
                                    Point2f& center, float& radius );    
//! matches two contours using one of the available algorithms
CV_EXPORTS_W double matchShapes( const Mat& contour1,
                               const Mat& contour2,
                               int method, double parameter );
//! computes convex hull for a set of 2D points.
CV_EXPORTS void convexHull( const Mat& points, CV_OUT vector<int>& hull, bool clockwise=false );
//! computes convex hull for a set of 2D points.
CV_EXPORTS void convexHull( const Mat& points, CV_OUT vector<Point>& hull, bool clockwise=false );
//! computes convex hull for a set of 2D points.
CV_EXPORTS void convexHull( const Mat& points, CV_OUT vector<Point2f>& hull, bool clockwise=false );

//! returns true iff the contour is convex. Does not support contours with self-intersection
CV_EXPORTS_W bool isContourConvex( const Mat& contour );

//! fits ellipse to the set of 2D points
CV_EXPORTS_W RotatedRect fitEllipse( const Mat& points );

//! fits line to the set of 2D points using M-estimator algorithm
CV_EXPORTS void fitLine( const Mat& points, CV_OUT Vec4f& line, int distType,
                           double param, double reps, double aeps );
//! fits line to the set of 3D points using M-estimator algorithm
CV_EXPORTS void fitLine( const Mat& points, CV_OUT Vec6f& line, int distType,
                           double param, double reps, double aeps );
//! checks if the point is inside the contour. Optionally computes the signed distance from the point to the contour boundary
CV_EXPORTS_W double pointPolygonTest( const Mat& contour,
                                    Point2f pt, bool measureDist );
    
//! estimates the best-fit affine transformation that maps one 2D point set to another or one image to another.
CV_EXPORTS_W Mat estimateRigidTransform( const Mat& A, const Mat& B,
                                       bool fullAffine );

//! computes the best-fit affine transformation that maps one 3D point set to another (RANSAC algorithm is used)
CV_EXPORTS int estimateAffine3D(const Mat& from, const Mat& to, CV_OUT Mat& dst,
                                CV_OUT vector<uchar>& outliers,
                                double param1 = 3.0, double param2 = 0.99);
    
}


// 2009-01-12, Xavier Delacour <xavier.delacour@gmail.com>

struct lsh_hash {
  int h1, h2;
};

struct CvLSHOperations
{
  virtual ~CvLSHOperations() {}

  virtual int vector_add(const void* data) = 0;
  virtual void vector_remove(int i) = 0;
  virtual const void* vector_lookup(int i) = 0;
  virtual void vector_reserve(int n) = 0;
  virtual unsigned int vector_count() = 0;

  virtual void hash_insert(lsh_hash h, int l, int i) = 0;
  virtual void hash_remove(lsh_hash h, int l, int i) = 0;
  virtual int hash_lookup(lsh_hash h, int l, int* ret_i, int ret_i_max) = 0;
};

#endif /* __cplusplus */

#endif

/* End of file. */
