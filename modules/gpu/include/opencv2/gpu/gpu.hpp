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
//     and/or other GpuMaterials provided with the distribution.
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

#ifndef __OPENCV_GPU_HPP__
#define __OPENCV_GPU_HPP__

#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/gpu/gpumat.hpp"

namespace cv
{
    namespace gpu
    {
        //////////////////////////////// Initialization & Info ////////////////////////

        //! This is the only function that do not throw exceptions if the library is compiled without Cuda.
        CV_EXPORTS int getCudaEnabledDeviceCount();

        //! Functions below throw cv::Expception if the library is compiled without Cuda.

        CV_EXPORTS void setDevice(int device);
        CV_EXPORTS int getDevice();

        //! Explicitly destroys and cleans up all resources associated with the current device in the current process. 
        //! Any subsequent API call to this device will reinitialize the device.
        CV_EXPORTS void resetDevice();

        enum FeatureSet
        {
            FEATURE_SET_COMPUTE_10 = 10,
            FEATURE_SET_COMPUTE_11 = 11,
            FEATURE_SET_COMPUTE_12 = 12,
            FEATURE_SET_COMPUTE_13 = 13,
            FEATURE_SET_COMPUTE_20 = 20,
            FEATURE_SET_COMPUTE_21 = 21,
            GLOBAL_ATOMICS = FEATURE_SET_COMPUTE_11,
            SHARED_ATOMICS = FEATURE_SET_COMPUTE_12,
            NATIVE_DOUBLE = FEATURE_SET_COMPUTE_13
        };

        // Gives information about what GPU archs this OpenCV GPU module was 
        // compiled for
        class CV_EXPORTS TargetArchs
        {
        public:
            static bool builtWith(FeatureSet feature_set);
            static bool has(int major, int minor);
            static bool hasPtx(int major, int minor);
            static bool hasBin(int major, int minor);
            static bool hasEqualOrLessPtx(int major, int minor);
            static bool hasEqualOrGreater(int major, int minor);
            static bool hasEqualOrGreaterPtx(int major, int minor);
            static bool hasEqualOrGreaterBin(int major, int minor);
        private:
            TargetArchs();
        };

        // Gives information about the given GPU
        class CV_EXPORTS DeviceInfo
        {
        public:
            // Creates DeviceInfo object for the current GPU
            DeviceInfo() : device_id_(getDevice()) { query(); }

            // Creates DeviceInfo object for the given GPU
            DeviceInfo(int device_id) : device_id_(device_id) { query(); }

            string name() const { return name_; }

            // Return compute capability versions
            int majorVersion() const { return majorVersion_; }
            int minorVersion() const { return minorVersion_; }

            int multiProcessorCount() const { return multi_processor_count_; }

            size_t freeMemory() const;
            size_t totalMemory() const;

            // Checks whether device supports the given feature
            bool supports(FeatureSet feature_set) const;

            // Checks whether the GPU module can be run on the given device
            bool isCompatible() const;

            int deviceID() const { return device_id_; }

        private:
            void query();
            void queryMemory(size_t& free_memory, size_t& total_memory) const;

            int device_id_;

            string name_;
            int multi_processor_count_;
            int majorVersion_;
            int minorVersion_;
        };

        //////////////////////////////// Error handling ////////////////////////

        CV_EXPORTS void error(const char *error_string, const char *file, const int line, const char *func);
        CV_EXPORTS void nppError( int err, const char *file, const int line, const char *func);

        //////////////////////////////// CudaMem ////////////////////////////////
        // CudaMem is limited cv::Mat with page locked memory allocation.
        // Page locked memory is only needed for async and faster coping to GPU.
        // It is convertable to cv::Mat header without reference counting
        // so you can use it with other opencv functions.

        // Page-locks the matrix m memory and maps it for the device(s)
        CV_EXPORTS void registerPageLocked(Mat& m);
        // Unmaps the memory of matrix m, and makes it pageable again.
        CV_EXPORTS void unregisterPageLocked(Mat& m);

        class CV_EXPORTS CudaMem
        {
        public:
            enum  { ALLOC_PAGE_LOCKED = 1, ALLOC_ZEROCOPY = 2, ALLOC_WRITE_COMBINED = 4 };

            CudaMem();
            CudaMem(const CudaMem& m);

            CudaMem(int rows, int cols, int type, int _alloc_type = ALLOC_PAGE_LOCKED);
            CudaMem(Size size, int type, int alloc_type = ALLOC_PAGE_LOCKED);


            //! creates from cv::Mat with coping data
            explicit CudaMem(const Mat& m, int alloc_type = ALLOC_PAGE_LOCKED);

            ~CudaMem();

            CudaMem& operator = (const CudaMem& m);

            //! returns deep copy of the matrix, i.e. the data is copied
            CudaMem clone() const;

            //! allocates new matrix data unless the matrix already has specified size and type.
            void create(int rows, int cols, int type, int alloc_type = ALLOC_PAGE_LOCKED);
            void create(Size size, int type, int alloc_type = ALLOC_PAGE_LOCKED);

            //! decrements reference counter and released memory if needed.
            void release();

            //! returns matrix header with disabled reference counting for CudaMem data.
            Mat createMatHeader() const;
            operator Mat() const;

            //! maps host memory into device address space and returns GpuMat header for it. Throws exception if not supported by hardware.
            GpuMat createGpuMatHeader() const;
            operator GpuMat() const;

            //returns if host memory can be mapperd to gpu address space;
            static bool canMapHostMemory();

            // Please see cv::Mat for descriptions
            bool isContinuous() const;
            size_t elemSize() const;
            size_t elemSize1() const;
            int type() const;
            int depth() const;
            int channels() const;
            size_t step1() const;
            Size size() const;
            bool empty() const;


            // Please see cv::Mat for descriptions
            int flags;
            int rows, cols;
            size_t step;

            uchar* data;
            int* refcount;

            uchar* datastart;
            uchar* dataend;

            int alloc_type;
        };

        //////////////////////////////// CudaStream ////////////////////////////////
        // Encapculates Cuda Stream. Provides interface for async coping.
        // Passed to each function that supports async kernel execution.
        // Reference counting is enabled

        class CV_EXPORTS Stream
        {
        public:
            Stream();
            ~Stream();

            Stream(const Stream&);
            Stream& operator=(const Stream&);

            bool queryIfComplete();
            void waitForCompletion();

            //! downloads asynchronously.
            // Warning! cv::Mat must point to page locked memory (i.e. to CudaMem data or to its subMat)
            void enqueueDownload(const GpuMat& src, CudaMem& dst);
            void enqueueDownload(const GpuMat& src, Mat& dst);

            //! uploads asynchronously.
            // Warning! cv::Mat must point to page locked memory (i.e. to CudaMem data or to its ROI)
            void enqueueUpload(const CudaMem& src, GpuMat& dst);
            void enqueueUpload(const Mat& src, GpuMat& dst);

            void enqueueCopy(const GpuMat& src, GpuMat& dst);

            void enqueueMemSet(GpuMat& src, Scalar val);
            void enqueueMemSet(GpuMat& src, Scalar val, const GpuMat& mask);

            // converts matrix type, ex from float to uchar depending on type
            void enqueueConvert(const GpuMat& src, GpuMat& dst, int type, double a = 1, double b = 0);

            static Stream& Null();

            operator bool() const;

        private:
            void create();
            void release();

            struct Impl;
            Impl *impl;

            friend struct StreamAccessor;
            
            explicit Stream(Impl* impl);
        };
        

        //////////////////////////////// Filter Engine ////////////////////////////////

        /*!
        The Base Class for 1D or Row-wise Filters

        This is the base class for linear or non-linear filters that process 1D data.
        In particular, such filters are used for the "horizontal" filtering parts in separable filters.
        */
        class CV_EXPORTS BaseRowFilter_GPU
        {
        public:
            BaseRowFilter_GPU(int ksize_, int anchor_) : ksize(ksize_), anchor(anchor_) {}
            virtual ~BaseRowFilter_GPU() {}
            virtual void operator()(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null()) = 0;
            int ksize, anchor;
        };

        /*!
        The Base Class for Column-wise Filters

        This is the base class for linear or non-linear filters that process columns of 2D arrays.
        Such filters are used for the "vertical" filtering parts in separable filters.
        */
        class CV_EXPORTS BaseColumnFilter_GPU
        {
        public:
            BaseColumnFilter_GPU(int ksize_, int anchor_) : ksize(ksize_), anchor(anchor_) {}
            virtual ~BaseColumnFilter_GPU() {}
            virtual void operator()(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null()) = 0;
            int ksize, anchor;
        };

        /*!
        The Base Class for Non-Separable 2D Filters.

        This is the base class for linear or non-linear 2D filters.
        */
        class CV_EXPORTS BaseFilter_GPU
        {
        public:
            BaseFilter_GPU(const Size& ksize_, const Point& anchor_) : ksize(ksize_), anchor(anchor_) {}
            virtual ~BaseFilter_GPU() {}
            virtual void operator()(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null()) = 0;
            Size ksize;
            Point anchor;
        };

        /*!
        The Base Class for Filter Engine.

        The class can be used to apply an arbitrary filtering operation to an image.
        It contains all the necessary intermediate buffers.
        */
        class CV_EXPORTS FilterEngine_GPU
        {
        public:
            virtual ~FilterEngine_GPU() {}

            virtual void apply(const GpuMat& src, GpuMat& dst, Rect roi = Rect(0,0,-1,-1), Stream& stream = Stream::Null()) = 0;
        };

        //! returns the non-separable filter engine with the specified filter
        CV_EXPORTS Ptr<FilterEngine_GPU> createFilter2D_GPU(const Ptr<BaseFilter_GPU>& filter2D, int srcType, int dstType);

        //! returns the separable filter engine with the specified filters
        CV_EXPORTS Ptr<FilterEngine_GPU> createSeparableFilter_GPU(const Ptr<BaseRowFilter_GPU>& rowFilter,
            const Ptr<BaseColumnFilter_GPU>& columnFilter, int srcType, int bufType, int dstType);

        //! returns horizontal 1D box filter
        //! supports only CV_8UC1 source type and CV_32FC1 sum type
        CV_EXPORTS Ptr<BaseRowFilter_GPU> getRowSumFilter_GPU(int srcType, int sumType, int ksize, int anchor = -1);

        //! returns vertical 1D box filter
        //! supports only CV_8UC1 sum type and CV_32FC1 dst type
        CV_EXPORTS Ptr<BaseColumnFilter_GPU> getColumnSumFilter_GPU(int sumType, int dstType, int ksize, int anchor = -1);

        //! returns 2D box filter
        //! supports CV_8UC1 and CV_8UC4 source type, dst type must be the same as source type
        CV_EXPORTS Ptr<BaseFilter_GPU> getBoxFilter_GPU(int srcType, int dstType, const Size& ksize, Point anchor = Point(-1, -1));

        //! returns box filter engine
        CV_EXPORTS Ptr<FilterEngine_GPU> createBoxFilter_GPU(int srcType, int dstType, const Size& ksize,
            const Point& anchor = Point(-1,-1));

        //! returns 2D morphological filter
        //! only MORPH_ERODE and MORPH_DILATE are supported
        //! supports CV_8UC1 and CV_8UC4 types
        //! kernel must have CV_8UC1 type, one rows and cols == ksize.width * ksize.height
        CV_EXPORTS Ptr<BaseFilter_GPU> getMorphologyFilter_GPU(int op, int type, const Mat& kernel, const Size& ksize,
            Point anchor=Point(-1,-1));

        //! returns morphological filter engine. Only MORPH_ERODE and MORPH_DILATE are supported.
        CV_EXPORTS Ptr<FilterEngine_GPU> createMorphologyFilter_GPU(int op, int type, const Mat& kernel,
            const Point& anchor = Point(-1,-1), int iterations = 1);

        //! returns 2D filter with the specified kernel
        //! supports CV_8UC1 and CV_8UC4 types
        CV_EXPORTS Ptr<BaseFilter_GPU> getLinearFilter_GPU(int srcType, int dstType, const Mat& kernel, const Size& ksize,
            Point anchor = Point(-1, -1));

        //! returns the non-separable linear filter engine
        CV_EXPORTS Ptr<FilterEngine_GPU> createLinearFilter_GPU(int srcType, int dstType, const Mat& kernel,
            const Point& anchor = Point(-1,-1));

        //! returns the primitive row filter with the specified kernel.
        //! supports only CV_8UC1, CV_8UC4, CV_16SC1, CV_16SC2, CV_32SC1, CV_32FC1 source type.
        //! there are two version of algorithm: NPP and OpenCV.
        //! NPP calls when srcType == CV_8UC1 or srcType == CV_8UC4 and bufType == srcType,
        //! otherwise calls OpenCV version.
        //! NPP supports only BORDER_CONSTANT border type.
        //! OpenCV version supports only CV_32F as buffer depth and
        //! BORDER_REFLECT101, BORDER_REPLICATE and BORDER_CONSTANT border types.
        CV_EXPORTS Ptr<BaseRowFilter_GPU> getLinearRowFilter_GPU(int srcType, int bufType, const Mat& rowKernel,
            int anchor = -1, int borderType = BORDER_CONSTANT);

        //! returns the primitive column filter with the specified kernel.
        //! supports only CV_8UC1, CV_8UC4, CV_16SC1, CV_16SC2, CV_32SC1, CV_32FC1 dst type.
        //! there are two version of algorithm: NPP and OpenCV.
        //! NPP calls when dstType == CV_8UC1 or dstType == CV_8UC4 and bufType == dstType,
        //! otherwise calls OpenCV version.
        //! NPP supports only BORDER_CONSTANT border type.
        //! OpenCV version supports only CV_32F as buffer depth and
        //! BORDER_REFLECT101, BORDER_REPLICATE and BORDER_CONSTANT border types.
        CV_EXPORTS Ptr<BaseColumnFilter_GPU> getLinearColumnFilter_GPU(int bufType, int dstType, const Mat& columnKernel,
            int anchor = -1, int borderType = BORDER_CONSTANT);

        //! returns the separable linear filter engine
        CV_EXPORTS Ptr<FilterEngine_GPU> createSeparableLinearFilter_GPU(int srcType, int dstType, const Mat& rowKernel,
            const Mat& columnKernel, const Point& anchor = Point(-1,-1), int rowBorderType = BORDER_DEFAULT,
            int columnBorderType = -1);

        //! returns filter engine for the generalized Sobel operator
        CV_EXPORTS Ptr<FilterEngine_GPU> createDerivFilter_GPU(int srcType, int dstType, int dx, int dy, int ksize,
            int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1);

        //! returns the Gaussian filter engine
        CV_EXPORTS Ptr<FilterEngine_GPU> createGaussianFilter_GPU(int type, Size ksize, double sigma1, double sigma2 = 0,
            int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1);

        //! returns maximum filter
        CV_EXPORTS Ptr<BaseFilter_GPU> getMaxFilter_GPU(int srcType, int dstType, const Size& ksize, Point anchor = Point(-1,-1));

        //! returns minimum filter
        CV_EXPORTS Ptr<BaseFilter_GPU> getMinFilter_GPU(int srcType, int dstType, const Size& ksize, Point anchor = Point(-1,-1));

        //! smooths the image using the normalized box filter
        //! supports CV_8UC1, CV_8UC4 types
        CV_EXPORTS void boxFilter(const GpuMat& src, GpuMat& dst, int ddepth, Size ksize, Point anchor = Point(-1,-1), Stream& stream = Stream::Null());

        //! a synonym for normalized box filter
        static inline void blur(const GpuMat& src, GpuMat& dst, Size ksize, Point anchor = Point(-1,-1), Stream& stream = Stream::Null()) { boxFilter(src, dst, -1, ksize, anchor, stream); }

        //! erodes the image (applies the local minimum operator)
        CV_EXPORTS void erode( const GpuMat& src, GpuMat& dst, const Mat& kernel, Point anchor = Point(-1, -1), int iterations = 1, Stream& stream = Stream::Null());

        //! dilates the image (applies the local maximum operator)
        CV_EXPORTS void dilate( const GpuMat& src, GpuMat& dst, const Mat& kernel, Point anchor = Point(-1, -1), int iterations = 1, Stream& stream = Stream::Null());

        //! applies an advanced morphological operation to the image
        CV_EXPORTS void morphologyEx( const GpuMat& src, GpuMat& dst, int op, const Mat& kernel, Point anchor = Point(-1, -1), int iterations = 1, Stream& stream = Stream::Null());

        //! applies non-separable 2D linear filter to the image
        CV_EXPORTS void filter2D(const GpuMat& src, GpuMat& dst, int ddepth, const Mat& kernel, Point anchor=Point(-1,-1), Stream& stream = Stream::Null());

        //! applies separable 2D linear filter to the image
        CV_EXPORTS void sepFilter2D(const GpuMat& src, GpuMat& dst, int ddepth, const Mat& kernelX, const Mat& kernelY,
            Point anchor = Point(-1,-1), int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1, Stream& stream = Stream::Null());

        //! applies generalized Sobel operator to the image
        CV_EXPORTS void Sobel(const GpuMat& src, GpuMat& dst, int ddepth, int dx, int dy, int ksize = 3, double scale = 1,
            int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1, Stream& stream = Stream::Null());

        //! applies the vertical or horizontal Scharr operator to the image
        CV_EXPORTS void Scharr(const GpuMat& src, GpuMat& dst, int ddepth, int dx, int dy, double scale = 1,
            int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1, Stream& stream = Stream::Null());

        //! smooths the image using Gaussian filter.
        CV_EXPORTS void GaussianBlur(const GpuMat& src, GpuMat& dst, Size ksize, double sigma1, double sigma2 = 0,
            int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1, Stream& stream = Stream::Null());

        //! applies Laplacian operator to the image
        //! supports only ksize = 1 and ksize = 3
        CV_EXPORTS void Laplacian(const GpuMat& src, GpuMat& dst, int ddepth, int ksize = 1, double scale = 1, Stream& stream = Stream::Null());


        ////////////////////////////// Arithmetics ///////////////////////////////////

        //! transposes the matrix
        //! supports matrix with element size = 1, 4 and 8 bytes (CV_8UC1, CV_8UC4, CV_16UC2, CV_32FC1, etc)
        CV_EXPORTS void transpose(const GpuMat& src1, GpuMat& dst, Stream& stream = Stream::Null());

        //! reverses the order of the rows, columns or both in a matrix
        //! supports CV_8UC1, CV_8UC4 types
        CV_EXPORTS void flip(const GpuMat& a, GpuMat& b, int flipCode, Stream& stream = Stream::Null());

        //! transforms 8-bit unsigned integers using lookup table: dst(i)=lut(src(i))
        //! destination array will have the depth type as lut and the same channels number as source
        //! supports CV_8UC1, CV_8UC3 types
        CV_EXPORTS void LUT(const GpuMat& src, const Mat& lut, GpuMat& dst, Stream& stream = Stream::Null());

        //! makes multi-channel array out of several single-channel arrays
        CV_EXPORTS void merge(const GpuMat* src, size_t n, GpuMat& dst, Stream& stream = Stream::Null());

        //! makes multi-channel array out of several single-channel arrays
        CV_EXPORTS void merge(const vector<GpuMat>& src, GpuMat& dst, Stream& stream = Stream::Null());

        //! copies each plane of a multi-channel array to a dedicated array
        CV_EXPORTS void split(const GpuMat& src, GpuMat* dst, Stream& stream = Stream::Null());

        //! copies each plane of a multi-channel array to a dedicated array
        CV_EXPORTS void split(const GpuMat& src, vector<GpuMat>& dst, Stream& stream = Stream::Null());

        //! computes magnitude of complex (x(i).re, x(i).im) vector
        //! supports only CV_32FC2 type
        CV_EXPORTS void magnitude(const GpuMat& x, GpuMat& magnitude, Stream& stream = Stream::Null());

        //! computes squared magnitude of complex (x(i).re, x(i).im) vector
        //! supports only CV_32FC2 type
        CV_EXPORTS void magnitudeSqr(const GpuMat& x, GpuMat& magnitude, Stream& stream = Stream::Null());

        //! computes magnitude of each (x(i), y(i)) vector
        //! supports only floating-point source
        CV_EXPORTS void magnitude(const GpuMat& x, const GpuMat& y, GpuMat& magnitude, Stream& stream = Stream::Null());

        //! computes squared magnitude of each (x(i), y(i)) vector
        //! supports only floating-point source
        CV_EXPORTS void magnitudeSqr(const GpuMat& x, const GpuMat& y, GpuMat& magnitude, Stream& stream = Stream::Null());

        //! computes angle (angle(i)) of each (x(i), y(i)) vector
        //! supports only floating-point source
        CV_EXPORTS void phase(const GpuMat& x, const GpuMat& y, GpuMat& angle, bool angleInDegrees = false, Stream& stream = Stream::Null());

        //! converts Cartesian coordinates to polar
        //! supports only floating-point source
        CV_EXPORTS void cartToPolar(const GpuMat& x, const GpuMat& y, GpuMat& magnitude, GpuMat& angle, bool angleInDegrees = false, Stream& stream = Stream::Null());

        //! converts polar coordinates to Cartesian
        //! supports only floating-point source
        CV_EXPORTS void polarToCart(const GpuMat& magnitude, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees = false, Stream& stream = Stream::Null());


        //////////////////////////// Per-element operations ////////////////////////////////////

        //! adds one matrix to another (c = a + b)
        //! supports CV_8UC1, CV_8UC4, CV_32SC1, CV_32FC1 types
        CV_EXPORTS void add(const GpuMat& a, const GpuMat& b, GpuMat& c, Stream& stream = Stream::Null());
        //! adds scalar to a matrix (c = a + s)
        //! supports CV_32FC1 and CV_32FC2 type
        CV_EXPORTS void add(const GpuMat& a, const Scalar& sc, GpuMat& c, Stream& stream = Stream::Null());

        //! subtracts one matrix from another (c = a - b)
        //! supports CV_8UC1, CV_8UC4, CV_32SC1, CV_32FC1 types
        CV_EXPORTS void subtract(const GpuMat& a, const GpuMat& b, GpuMat& c, Stream& stream = Stream::Null());
        //! subtracts scalar from a matrix (c = a - s)
        //! supports CV_32FC1 and CV_32FC2 type
        CV_EXPORTS void subtract(const GpuMat& a, const Scalar& sc, GpuMat& c, Stream& stream = Stream::Null());

        //! computes element-wise product of the two arrays (c = a * b)
        //! supports CV_8UC1, CV_8UC4, CV_32SC1, CV_32FC1 types
        CV_EXPORTS void multiply(const GpuMat& a, const GpuMat& b, GpuMat& c, Stream& stream = Stream::Null());
        //! multiplies matrix to a scalar (c = a * s)
        //! supports CV_32FC1 type
        CV_EXPORTS void multiply(const GpuMat& a, const Scalar& sc, GpuMat& c, Stream& stream = Stream::Null());

        //! computes element-wise quotient of the two arrays (c = a / b)
        //! supports CV_8UC1, CV_8UC4, CV_32SC1, CV_32FC1 types
        CV_EXPORTS void divide(const GpuMat& a, const GpuMat& b, GpuMat& c, Stream& stream = Stream::Null());
        //! computes element-wise quotient of matrix and scalar (c = a / s)
        //! supports CV_32FC1 type
        CV_EXPORTS void divide(const GpuMat& a, const Scalar& sc, GpuMat& c, Stream& stream = Stream::Null());

        //! computes exponent of each matrix element (b = e**a)
        //! supports only CV_32FC1 type
        CV_EXPORTS void exp(const GpuMat& a, GpuMat& b, Stream& stream = Stream::Null());
        
        //! computes power of each matrix element:
        //    (dst(i,j) = pow(     src(i,j) , power), if src.type() is integer
        //    (dst(i,j) = pow(fabs(src(i,j)), power), otherwise
        //! supports all, except depth == CV_64F
        CV_EXPORTS void pow(const GpuMat& src, double power, GpuMat& dst, Stream& stream = Stream::Null());

        //! computes natural logarithm of absolute value of each matrix element: b = log(abs(a))
        //! supports only CV_32FC1 type
        CV_EXPORTS void log(const GpuMat& a, GpuMat& b, Stream& stream = Stream::Null());

        //! computes element-wise absolute difference of two arrays (c = abs(a - b))
        //! supports CV_8UC1, CV_8UC4, CV_32SC1, CV_32FC1 types
        CV_EXPORTS void absdiff(const GpuMat& a, const GpuMat& b, GpuMat& c, Stream& stream = Stream::Null());
        //! computes element-wise absolute difference of array and scalar (c = abs(a - s))
        //! supports only CV_32FC1 type
        CV_EXPORTS void absdiff(const GpuMat& a, const Scalar& s, GpuMat& c, Stream& stream = Stream::Null());

        //! compares elements of two arrays (c = a <cmpop> b)
        //! supports CV_8UC4, CV_32FC1 types
        CV_EXPORTS void compare(const GpuMat& a, const GpuMat& b, GpuMat& c, int cmpop, Stream& stream = Stream::Null());

        //! performs per-elements bit-wise inversion
        CV_EXPORTS void bitwise_not(const GpuMat& src, GpuMat& dst, const GpuMat& mask=GpuMat(), Stream& stream = Stream::Null());

        //! calculates per-element bit-wise disjunction of two arrays
        CV_EXPORTS void bitwise_or(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask=GpuMat(), Stream& stream = Stream::Null());

        //! calculates per-element bit-wise conjunction of two arrays
        CV_EXPORTS void bitwise_and(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask=GpuMat(), Stream& stream = Stream::Null());

        //! calculates per-element bit-wise "exclusive or" operation
        CV_EXPORTS void bitwise_xor(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const GpuMat& mask=GpuMat(), Stream& stream = Stream::Null());

        //! computes per-element minimum of two arrays (dst = min(src1, src2))
        CV_EXPORTS void min(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream = Stream::Null());

        //! computes per-element minimum of array and scalar (dst = min(src1, src2))
        CV_EXPORTS void min(const GpuMat& src1, double src2, GpuMat& dst, Stream& stream = Stream::Null());

        //! computes per-element maximum of two arrays (dst = max(src1, src2))
        CV_EXPORTS void max(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream = Stream::Null());

        //! computes per-element maximum of array and scalar (dst = max(src1, src2))
        CV_EXPORTS void max(const GpuMat& src1, double src2, GpuMat& dst, Stream& stream = Stream::Null());


        ////////////////////////////// Image processing //////////////////////////////

        //! DST[x,y] = SRC[xmap[x,y],ymap[x,y]]
        //! supports only CV_32FC1 map type
        CV_EXPORTS void remap(const GpuMat& src, GpuMat& dst, const GpuMat& xmap, const GpuMat& ymap,
            int interpolation, int borderMode = BORDER_CONSTANT, const Scalar& borderValue = Scalar(), 
            Stream& stream = Stream::Null());

        //! Does mean shift filtering on GPU.
        CV_EXPORTS void meanShiftFiltering(const GpuMat& src, GpuMat& dst, int sp, int sr,
            TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1));

        //! Does mean shift procedure on GPU.
        CV_EXPORTS void meanShiftProc(const GpuMat& src, GpuMat& dstr, GpuMat& dstsp, int sp, int sr,
            TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1));

        //! Does mean shift segmentation with elimination of small regions.
        CV_EXPORTS void meanShiftSegmentation(const GpuMat& src, Mat& dst, int sp, int sr, int minsize,
            TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1));

        //! Does coloring of disparity image: [0..ndisp) -> [0..240, 1, 1] in HSV.
        //! Supported types of input disparity: CV_8U, CV_16S.
        //! Output disparity has CV_8UC4 type in BGRA format (alpha = 255).
        CV_EXPORTS void drawColorDisp(const GpuMat& src_disp, GpuMat& dst_disp, int ndisp, Stream& stream = Stream::Null());

        //! Reprojects disparity image to 3D space.
        //! Supports CV_8U and CV_16S types of input disparity.
        //! The output is a 4-channel floating-point (CV_32FC4) matrix.
        //! Each element of this matrix will contain the 3D coordinates of the point (x,y,z,1), computed from the disparity map.
        //! Q is the 4x4 perspective transformation matrix that can be obtained with cvStereoRectify.
        CV_EXPORTS void reprojectImageTo3D(const GpuMat& disp, GpuMat& xyzw, const Mat& Q, Stream& stream = Stream::Null());

        //! converts image from one color space to another
        CV_EXPORTS void cvtColor(const GpuMat& src, GpuMat& dst, int code, int dcn = 0, Stream& stream = Stream::Null());

        //! applies fixed threshold to the image
        CV_EXPORTS double threshold(const GpuMat& src, GpuMat& dst, double thresh, double maxval, int type, Stream& stream = Stream::Null());

        //! resizes the image
        //! Supports INTER_NEAREST, INTER_LINEAR, INTER_CUBIC
        CV_EXPORTS void resize(const GpuMat& src, GpuMat& dst, Size dsize, double fx=0, double fy=0, int interpolation = INTER_LINEAR, Stream& stream = Stream::Null());

        //! warps the image using affine transformation
        //! Supports INTER_NEAREST, INTER_LINEAR, INTER_CUBIC
        CV_EXPORTS void warpAffine(const GpuMat& src, GpuMat& dst, const Mat& M, Size dsize, int flags = INTER_LINEAR, Stream& stream = Stream::Null());

        //! warps the image using perspective transformation
        //! Supports INTER_NEAREST, INTER_LINEAR, INTER_CUBIC
        CV_EXPORTS void warpPerspective(const GpuMat& src, GpuMat& dst, const Mat& M, Size dsize, int flags = INTER_LINEAR, Stream& stream = Stream::Null());

        //! builds plane warping maps
        CV_EXPORTS void buildWarpPlaneMaps(Size src_size, Rect dst_roi, const Mat &K, const Mat& R, float scale,
                                           GpuMat& map_x, GpuMat& map_y, Stream& stream = Stream::Null());

        //! builds cylindrical warping maps
        CV_EXPORTS void buildWarpCylindricalMaps(Size src_size, Rect dst_roi, const Mat &K, const Mat& R, float scale,
                                                 GpuMat& map_x, GpuMat& map_y, Stream& stream = Stream::Null());

        //! builds spherical warping maps
        CV_EXPORTS void buildWarpSphericalMaps(Size src_size, Rect dst_roi, const Mat &K, const Mat& R, float scale,
                                               GpuMat& map_x, GpuMat& map_y, Stream& stream = Stream::Null());

        //! rotate 8bit single or four channel image
        //! Supports INTER_NEAREST, INTER_LINEAR, INTER_CUBIC
        //! supports CV_8UC1, CV_8UC4 types
        CV_EXPORTS void rotate(const GpuMat& src, GpuMat& dst, Size dsize, double angle, double xShift = 0, double yShift = 0, int interpolation = INTER_LINEAR, Stream& stream = Stream::Null());

        //! copies 2D array to a larger destination array and pads borders with user-specifiable constant
        //! supports CV_8UC1, CV_8UC4, CV_32SC1 and CV_32FC1 types
        CV_EXPORTS void copyMakeBorder(const GpuMat& src, GpuMat& dst, int top, int bottom, int left, int right, const Scalar& value = Scalar(), Stream& stream = Stream::Null());

        //! computes the integral image
        //! sum will have CV_32S type, but will contain unsigned int values
        //! supports only CV_8UC1 source type
        CV_EXPORTS void integral(const GpuMat& src, GpuMat& sum, Stream& stream = Stream::Null());

        //! buffered version
        CV_EXPORTS void integralBuffered(const GpuMat& src, GpuMat& sum, GpuMat& buffer, Stream& stream = Stream::Null());

        //! computes the integral image and integral for the squared image
        //! sum will have CV_32S type, sqsum - CV32F type
        //! supports only CV_8UC1 source type
        CV_EXPORTS void integral(const GpuMat& src, GpuMat& sum, GpuMat& sqsum, Stream& stream = Stream::Null());

        //! computes squared integral image
        //! result matrix will have 64F type, but will contain 64U values
        //! supports source images of 8UC1 type only
        CV_EXPORTS void sqrIntegral(const GpuMat& src, GpuMat& sqsum, Stream& stream = Stream::Null());

        //! computes vertical sum, supports only CV_32FC1 images
        CV_EXPORTS void columnSum(const GpuMat& src, GpuMat& sum);

        //! computes the standard deviation of integral images
        //! supports only CV_32SC1 source type and CV_32FC1 sqr type
        //! output will have CV_32FC1 type
        CV_EXPORTS void rectStdDev(const GpuMat& src, const GpuMat& sqr, GpuMat& dst, const Rect& rect, Stream& stream = Stream::Null());

        //! computes Harris cornerness criteria at each image pixel
        CV_EXPORTS void cornerHarris(const GpuMat& src, GpuMat& dst, int blockSize, int ksize, double k, int borderType=BORDER_REFLECT101);
        CV_EXPORTS void cornerHarris(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, int blockSize, int ksize, double k, int borderType=BORDER_REFLECT101);

        //! computes minimum eigen value of 2x2 derivative covariation matrix at each pixel - the cornerness criteria
        CV_EXPORTS void cornerMinEigenVal(const GpuMat& src, GpuMat& dst, int blockSize, int ksize, int borderType=BORDER_REFLECT101);
        CV_EXPORTS void cornerMinEigenVal(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, int blockSize, int ksize, int borderType=BORDER_REFLECT101);

        //! performs per-element multiplication of two full (not packed) Fourier spectrums
        //! supports 32FC2 matrixes only (interleaved format)
        CV_EXPORTS void mulSpectrums(const GpuMat& a, const GpuMat& b, GpuMat& c, int flags, bool conjB=false);

        //! performs per-element multiplication of two full (not packed) Fourier spectrums
        //! supports 32FC2 matrixes only (interleaved format)
        CV_EXPORTS void mulAndScaleSpectrums(const GpuMat& a, const GpuMat& b, GpuMat& c, int flags, 
                                             float scale, bool conjB=false);

        //! Performs a forward or inverse discrete Fourier transform (1D or 2D) of floating point matrix.
        //! Param dft_size is the size of DFT transform.
        //! 
        //! If the source matrix is not continous, then additional copy will be done,
        //! so to avoid copying ensure the source matrix is continous one. If you want to use
        //! preallocated output ensure it is continuous too, otherwise it will be reallocated.
        //!
        //! Being implemented via CUFFT real-to-complex transform result contains only non-redundant values
        //! in CUFFT's format. Result as full complex matrix for such kind of transform cannot be retrieved.
        //!
        //! For complex-to-real transform it is assumed that the source matrix is packed in CUFFT's format.
        CV_EXPORTS void dft(const GpuMat& src, GpuMat& dst, Size dft_size, int flags=0);

        //! computes convolution (or cross-correlation) of two images using discrete Fourier transform
        //! supports source images of 32FC1 type only
        //! result matrix will have 32FC1 type
        CV_EXPORTS void convolve(const GpuMat& image, const GpuMat& templ, GpuMat& result, 
                                 bool ccorr=false);

        struct CV_EXPORTS ConvolveBuf;

        //! buffered version
        CV_EXPORTS void convolve(const GpuMat& image, const GpuMat& templ, GpuMat& result, 
                                 bool ccorr, ConvolveBuf& buf);

        struct CV_EXPORTS ConvolveBuf
        {
            ConvolveBuf() {}
            ConvolveBuf(Size image_size, Size templ_size) 
                { create(image_size, templ_size); }
            void create(Size image_size, Size templ_size);

        private:
            static Size estimateBlockSize(Size result_size, Size templ_size);
            friend void convolve(const GpuMat&, const GpuMat&, GpuMat&, bool, ConvolveBuf&);

            Size result_size;
            Size block_size;
            Size dft_size;
            int spect_len;

            GpuMat image_spect, templ_spect, result_spect;
            GpuMat image_block, templ_block, result_data;
        };

        //! computes the proximity map for the raster template and the image where the template is searched for
        CV_EXPORTS void matchTemplate(const GpuMat& image, const GpuMat& templ, GpuMat& result, int method);

        //! smoothes the source image and downsamples it
        CV_EXPORTS void pyrDown(const GpuMat& src, GpuMat& dst, int borderType = BORDER_DEFAULT, Stream& stream = Stream::Null());

        //! upsamples the source image and then smoothes it
        CV_EXPORTS void pyrUp(const GpuMat& src, GpuMat& dst, int borderType = BORDER_DEFAULT, Stream& stream = Stream::Null());

        //! performs linear blending of two images
        //! to avoid accuracy errors sum of weigths shouldn't be very close to zero
        CV_EXPORTS void blendLinear(const GpuMat& img1, const GpuMat& img2, const GpuMat& weights1, const GpuMat& weights2, 
            GpuMat& result, Stream& stream = Stream::Null());

        
        struct CV_EXPORTS CannyBuf;
        
        CV_EXPORTS void Canny(const GpuMat& image, GpuMat& edges, double low_thresh, double high_thresh, int apperture_size = 3, bool L2gradient = false);
        CV_EXPORTS void Canny(const GpuMat& image, CannyBuf& buf, GpuMat& edges, double low_thresh, double high_thresh, int apperture_size = 3, bool L2gradient = false);
        CV_EXPORTS void Canny(const GpuMat& dx, const GpuMat& dy, GpuMat& edges, double low_thresh, double high_thresh, bool L2gradient = false);
        CV_EXPORTS void Canny(const GpuMat& dx, const GpuMat& dy, CannyBuf& buf, GpuMat& edges, double low_thresh, double high_thresh, bool L2gradient = false);

        struct CV_EXPORTS CannyBuf
        {
            CannyBuf() {}
            explicit CannyBuf(const Size& image_size, int apperture_size = 3) {create(image_size, apperture_size);}
            CannyBuf(const GpuMat& dx_, const GpuMat& dy_);

            void create(const Size& image_size, int apperture_size = 3);
            
            void release();

            GpuMat dx, dy;
            GpuMat dx_buf, dy_buf;
            GpuMat edgeBuf;
            GpuMat trackBuf1, trackBuf2;
            Ptr<FilterEngine_GPU> filterDX, filterDY;
        };

        ////////////////////////////// Matrix reductions //////////////////////////////

        //! computes mean value and standard deviation of all or selected array elements
        //! supports only CV_8UC1 type
        CV_EXPORTS void meanStdDev(const GpuMat& mtx, Scalar& mean, Scalar& stddev);

        //! computes norm of array
        //! supports NORM_INF, NORM_L1, NORM_L2
        //! supports all matrices except 64F
        CV_EXPORTS double norm(const GpuMat& src1, int normType=NORM_L2);

        //! computes norm of array
        //! supports NORM_INF, NORM_L1, NORM_L2
        //! supports all matrices except 64F
        CV_EXPORTS double norm(const GpuMat& src1, int normType, GpuMat& buf);

        //! computes norm of the difference between two arrays
        //! supports NORM_INF, NORM_L1, NORM_L2
        //! supports only CV_8UC1 type
        CV_EXPORTS double norm(const GpuMat& src1, const GpuMat& src2, int normType=NORM_L2);

        //! computes sum of array elements
        //! supports only single channel images
        CV_EXPORTS Scalar sum(const GpuMat& src);

        //! computes sum of array elements
        //! supports only single channel images
        CV_EXPORTS Scalar sum(const GpuMat& src, GpuMat& buf);

        //! computes sum of array elements absolute values
        //! supports only single channel images
        CV_EXPORTS Scalar absSum(const GpuMat& src);

        //! computes sum of array elements absolute values
        //! supports only single channel images
        CV_EXPORTS Scalar absSum(const GpuMat& src, GpuMat& buf);

        //! computes squared sum of array elements
        //! supports only single channel images
        CV_EXPORTS Scalar sqrSum(const GpuMat& src);

        //! computes squared sum of array elements
        //! supports only single channel images
        CV_EXPORTS Scalar sqrSum(const GpuMat& src, GpuMat& buf);

        //! finds global minimum and maximum array elements and returns their values
        CV_EXPORTS void minMax(const GpuMat& src, double* minVal, double* maxVal=0, const GpuMat& mask=GpuMat());

        //! finds global minimum and maximum array elements and returns their values
        CV_EXPORTS void minMax(const GpuMat& src, double* minVal, double* maxVal, const GpuMat& mask, GpuMat& buf);

        //! finds global minimum and maximum array elements and returns their values with locations
        CV_EXPORTS void minMaxLoc(const GpuMat& src, double* minVal, double* maxVal=0, Point* minLoc=0, Point* maxLoc=0,
                                  const GpuMat& mask=GpuMat());

        //! finds global minimum and maximum array elements and returns their values with locations
        CV_EXPORTS void minMaxLoc(const GpuMat& src, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc,
                                  const GpuMat& mask, GpuMat& valbuf, GpuMat& locbuf);

        //! counts non-zero array elements
        CV_EXPORTS int countNonZero(const GpuMat& src);

        //! counts non-zero array elements
        CV_EXPORTS int countNonZero(const GpuMat& src, GpuMat& buf);


        ///////////////////////////// Calibration 3D //////////////////////////////////

        CV_EXPORTS void transformPoints(const GpuMat& src, const Mat& rvec, const Mat& tvec,
                                        GpuMat& dst, Stream& stream = Stream::Null());

        CV_EXPORTS void projectPoints(const GpuMat& src, const Mat& rvec, const Mat& tvec,
                                      const Mat& camera_mat, const Mat& dist_coef, GpuMat& dst, 
                                      Stream& stream = Stream::Null());

        CV_EXPORTS void solvePnPRansac(const Mat& object, const Mat& image, const Mat& camera_mat,
                                       const Mat& dist_coef, Mat& rvec, Mat& tvec, bool use_extrinsic_guess=false,
                                       int num_iters=100, float max_dist=8.0, int min_inlier_count=100, 
                                       vector<int>* inliers=NULL);

        //////////////////////////////// Image Labeling ////////////////////////////////

        //!performs labeling via graph cuts
        CV_EXPORTS void graphcut(GpuMat& terminals, GpuMat& leftTransp, GpuMat& rightTransp, GpuMat& top, GpuMat& bottom, GpuMat& labels, GpuMat& buf, Stream& stream = Stream::Null());

        ////////////////////////////////// Histograms //////////////////////////////////

        //! Compute levels with even distribution. levels will have 1 row and nLevels cols and CV_32SC1 type.
        CV_EXPORTS void evenLevels(GpuMat& levels, int nLevels, int lowerLevel, int upperLevel);
        //! Calculates histogram with evenly distributed bins for signle channel source.
        //! Supports CV_8UC1, CV_16UC1 and CV_16SC1 source types.
        //! Output hist will have one row and histSize cols and CV_32SC1 type.
        CV_EXPORTS void histEven(const GpuMat& src, GpuMat& hist, int histSize, int lowerLevel, int upperLevel, Stream& stream = Stream::Null());
        CV_EXPORTS void histEven(const GpuMat& src, GpuMat& hist, GpuMat& buf, int histSize, int lowerLevel, int upperLevel, Stream& stream = Stream::Null());
        //! Calculates histogram with evenly distributed bins for four-channel source.
        //! All channels of source are processed separately.
        //! Supports CV_8UC4, CV_16UC4 and CV_16SC4 source types.
        //! Output hist[i] will have one row and histSize[i] cols and CV_32SC1 type.
        CV_EXPORTS void histEven(const GpuMat& src, GpuMat hist[4], int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream = Stream::Null());
        CV_EXPORTS void histEven(const GpuMat& src, GpuMat hist[4], GpuMat& buf, int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream = Stream::Null());
        //! Calculates histogram with bins determined by levels array.
        //! levels must have one row and CV_32SC1 type if source has integer type or CV_32FC1 otherwise.
        //! Supports CV_8UC1, CV_16UC1, CV_16SC1 and CV_32FC1 source types.
        //! Output hist will have one row and (levels.cols-1) cols and CV_32SC1 type.
        CV_EXPORTS void histRange(const GpuMat& src, GpuMat& hist, const GpuMat& levels, Stream& stream = Stream::Null());
        CV_EXPORTS void histRange(const GpuMat& src, GpuMat& hist, const GpuMat& levels, GpuMat& buf, Stream& stream = Stream::Null());
        //! Calculates histogram with bins determined by levels array.
        //! All levels must have one row and CV_32SC1 type if source has integer type or CV_32FC1 otherwise.
        //! All channels of source are processed separately.
        //! Supports CV_8UC4, CV_16UC4, CV_16SC4 and CV_32FC4 source types.
        //! Output hist[i] will have one row and (levels[i].cols-1) cols and CV_32SC1 type.
        CV_EXPORTS void histRange(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4], Stream& stream = Stream::Null());
        CV_EXPORTS void histRange(const GpuMat& src, GpuMat hist[4], const GpuMat levels[4], GpuMat& buf, Stream& stream = Stream::Null());
        
        //! Calculates histogram for 8u one channel image
        //! Output hist will have one row, 256 cols and CV32SC1 type.
        CV_EXPORTS void calcHist(const GpuMat& src, GpuMat& hist, Stream& stream = Stream::Null());
        CV_EXPORTS void calcHist(const GpuMat& src, GpuMat& hist, GpuMat& buf, Stream& stream = Stream::Null());
        
        //! normalizes the grayscale image brightness and contrast by normalizing its histogram
        CV_EXPORTS void equalizeHist(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null());
        CV_EXPORTS void equalizeHist(const GpuMat& src, GpuMat& dst, GpuMat& hist, Stream& stream = Stream::Null());
        CV_EXPORTS void equalizeHist(const GpuMat& src, GpuMat& dst, GpuMat& hist, GpuMat& buf, Stream& stream = Stream::Null());

        //////////////////////////////// StereoBM_GPU ////////////////////////////////

        class CV_EXPORTS StereoBM_GPU
        {
        public:
            enum { BASIC_PRESET = 0, PREFILTER_XSOBEL = 1 };

            enum { DEFAULT_NDISP = 64, DEFAULT_WINSZ = 19 };

            //! the default constructor
            StereoBM_GPU();
            //! the full constructor taking the camera-specific preset, number of disparities and the SAD window size. ndisparities must be multiple of 8.
            StereoBM_GPU(int preset, int ndisparities = DEFAULT_NDISP, int winSize = DEFAULT_WINSZ);

            //! the stereo correspondence operator. Finds the disparity for the specified rectified stereo pair
            //! Output disparity has CV_8U type.
            void operator() ( const GpuMat& left, const GpuMat& right, GpuMat& disparity, Stream& stream = Stream::Null());

            //! Some heuristics that tries to estmate
            // if current GPU will be faster than CPU in this algorithm.
            // It queries current active device.
            static bool checkIfGpuCallReasonable();

            int preset;
            int ndisp;
            int winSize;

            // If avergeTexThreshold  == 0 => post procesing is disabled
            // If avergeTexThreshold != 0 then disparity is set 0 in each point (x,y) where for left image
            // SumOfHorizontalGradiensInWindow(x, y, winSize) < (winSize * winSize) * avergeTexThreshold
            // i.e. input left image is low textured.
            float avergeTexThreshold;
        private:
            GpuMat minSSD, leBuf, riBuf;
        };

        ////////////////////////// StereoBeliefPropagation ///////////////////////////
        // "Efficient Belief Propagation for Early Vision"
        // P.Felzenszwalb

        class CV_EXPORTS StereoBeliefPropagation
        {
        public:
            enum { DEFAULT_NDISP  = 64 };
            enum { DEFAULT_ITERS  = 5  };
            enum { DEFAULT_LEVELS = 5  };

            static void estimateRecommendedParams(int width, int height, int& ndisp, int& iters, int& levels);

            //! the default constructor
            explicit StereoBeliefPropagation(int ndisp  = DEFAULT_NDISP,
                int iters  = DEFAULT_ITERS,
                int levels = DEFAULT_LEVELS,
                int msg_type = CV_32F);

            //! the full constructor taking the number of disparities, number of BP iterations on each level,
            //! number of levels, truncation of data cost, data weight,
            //! truncation of discontinuity cost and discontinuity single jump
            //! DataTerm = data_weight * min(fabs(I2-I1), max_data_term)
            //! DiscTerm = min(disc_single_jump * fabs(f1-f2), max_disc_term)
            //! please see paper for more details
            StereoBeliefPropagation(int ndisp, int iters, int levels,
                float max_data_term, float data_weight,
                float max_disc_term, float disc_single_jump,
                int msg_type = CV_32F);

            //! the stereo correspondence operator. Finds the disparity for the specified rectified stereo pair,
            //! if disparity is empty output type will be CV_16S else output type will be disparity.type().
            void operator()(const GpuMat& left, const GpuMat& right, GpuMat& disparity, Stream& stream = Stream::Null());


            //! version for user specified data term
            void operator()(const GpuMat& data, GpuMat& disparity, Stream& stream = Stream::Null());

            int ndisp;

            int iters;
            int levels;

            float max_data_term;
            float data_weight;
            float max_disc_term;
            float disc_single_jump;

            int msg_type;
        private:
            GpuMat u, d, l, r, u2, d2, l2, r2;
            std::vector<GpuMat> datas;
            GpuMat out;
        };

        /////////////////////////// StereoConstantSpaceBP ///////////////////////////
        // "A Constant-Space Belief Propagation Algorithm for Stereo Matching"
        // Qingxiong Yang, Liang Wang, Narendra Ahuja
        // http://vision.ai.uiuc.edu/~qyang6/

        class CV_EXPORTS StereoConstantSpaceBP
        {
        public:
            enum { DEFAULT_NDISP    = 128 };
            enum { DEFAULT_ITERS    = 8   };
            enum { DEFAULT_LEVELS   = 4   };
            enum { DEFAULT_NR_PLANE = 4   };

            static void estimateRecommendedParams(int width, int height, int& ndisp, int& iters, int& levels, int& nr_plane);

            //! the default constructor
            explicit StereoConstantSpaceBP(int ndisp    = DEFAULT_NDISP,
                int iters    = DEFAULT_ITERS,
                int levels   = DEFAULT_LEVELS,
                int nr_plane = DEFAULT_NR_PLANE,
                int msg_type = CV_32F);

            //! the full constructor taking the number of disparities, number of BP iterations on each level,
            //! number of levels, number of active disparity on the first level, truncation of data cost, data weight,
            //! truncation of discontinuity cost, discontinuity single jump and minimum disparity threshold
            StereoConstantSpaceBP(int ndisp, int iters, int levels, int nr_plane,
                float max_data_term, float data_weight, float max_disc_term, float disc_single_jump,
                int min_disp_th = 0,
                int msg_type = CV_32F);

            //! the stereo correspondence operator. Finds the disparity for the specified rectified stereo pair,
            //! if disparity is empty output type will be CV_16S else output type will be disparity.type().
            void operator()(const GpuMat& left, const GpuMat& right, GpuMat& disparity, Stream& stream = Stream::Null());

            int ndisp;

            int iters;
            int levels;

            int nr_plane;

            float max_data_term;
            float data_weight;
            float max_disc_term;
            float disc_single_jump;

            int min_disp_th;

            int msg_type;

            bool use_local_init_data_cost;
        private:
            GpuMat u[2], d[2], l[2], r[2];
            GpuMat disp_selected_pyr[2];

            GpuMat data_cost;
            GpuMat data_cost_selected;

            GpuMat temp;

            GpuMat out;
        };

        /////////////////////////// DisparityBilateralFilter ///////////////////////////
        // Disparity map refinement using joint bilateral filtering given a single color image.
        // Qingxiong Yang, Liang Wang, Narendra Ahuja
        // http://vision.ai.uiuc.edu/~qyang6/

        class CV_EXPORTS DisparityBilateralFilter
        {
        public:
            enum { DEFAULT_NDISP  = 64 };
            enum { DEFAULT_RADIUS = 3 };
            enum { DEFAULT_ITERS  = 1 };

            //! the default constructor
            explicit DisparityBilateralFilter(int ndisp = DEFAULT_NDISP, int radius = DEFAULT_RADIUS, int iters = DEFAULT_ITERS);

            //! the full constructor taking the number of disparities, filter radius,
            //! number of iterations, truncation of data continuity, truncation of disparity continuity
            //! and filter range sigma
            DisparityBilateralFilter(int ndisp, int radius, int iters, float edge_threshold, float max_disc_threshold, float sigma_range);

            //! the disparity map refinement operator. Refine disparity map using joint bilateral filtering given a single color image.
            //! disparity must have CV_8U or CV_16S type, image must have CV_8UC1 or CV_8UC3 type.
            void operator()(const GpuMat& disparity, const GpuMat& image, GpuMat& dst, Stream& stream = Stream::Null());

        private:
            int ndisp;
            int radius;
            int iters;

            float edge_threshold;
            float max_disc_threshold;
            float sigma_range;

            GpuMat table_color;
            GpuMat table_space;
        };


        //////////////// HOG (Histogram-of-Oriented-Gradients) Descriptor and Object Detector //////////////

        struct CV_EXPORTS HOGDescriptor
        {
            enum { DEFAULT_WIN_SIGMA = -1 };
            enum { DEFAULT_NLEVELS = 64 };
            enum { DESCR_FORMAT_ROW_BY_ROW, DESCR_FORMAT_COL_BY_COL };

            HOGDescriptor(Size win_size=Size(64, 128), Size block_size=Size(16, 16),
                          Size block_stride=Size(8, 8), Size cell_size=Size(8, 8),
                          int nbins=9, double win_sigma=DEFAULT_WIN_SIGMA,
                          double threshold_L2hys=0.2, bool gamma_correction=true,
                          int nlevels=DEFAULT_NLEVELS);

            size_t getDescriptorSize() const;
            size_t getBlockHistogramSize() const;

            void setSVMDetector(const vector<float>& detector);

            static vector<float> getDefaultPeopleDetector();
            static vector<float> getPeopleDetector48x96();
            static vector<float> getPeopleDetector64x128();

            void detect(const GpuMat& img, vector<Point>& found_locations, 
                        double hit_threshold=0, Size win_stride=Size(), 
                        Size padding=Size());

            void detectMultiScale(const GpuMat& img, vector<Rect>& found_locations,
                                  double hit_threshold=0, Size win_stride=Size(), 
                                  Size padding=Size(), double scale0=1.05, 
                                  int group_threshold=2);

            void getDescriptors(const GpuMat& img, Size win_stride, 
                                GpuMat& descriptors,
                                int descr_format=DESCR_FORMAT_COL_BY_COL);

            Size win_size;
            Size block_size;
            Size block_stride;
            Size cell_size;
            int nbins;
            double win_sigma;
            double threshold_L2hys;
            bool gamma_correction;
            int nlevels;

        protected:
            void computeBlockHistograms(const GpuMat& img);
            void computeGradient(const GpuMat& img, GpuMat& grad, GpuMat& qangle);

            double getWinSigma() const;
            bool checkDetectorSize() const;

            static int numPartsWithin(int size, int part_size, int stride);
            static Size numPartsWithin(Size size, Size part_size, Size stride);

            // Coefficients of the separating plane
            float free_coef;
            GpuMat detector;

            // Results of the last classification step
            GpuMat labels, labels_buf;
            Mat labels_host;

            // Results of the last histogram evaluation step
            GpuMat block_hists, block_hists_buf;

            // Gradients conputation results
            GpuMat grad, qangle, grad_buf, qangle_buf;

			// returns subbuffer with required size, reallocates buffer if nessesary.
			static GpuMat getBuffer(const Size& sz, int type, GpuMat& buf);
			static GpuMat getBuffer(int rows, int cols, int type, GpuMat& buf);

			std::vector<GpuMat> image_scales;
        };


        ////////////////////////////////// BruteForceMatcher //////////////////////////////////

        class CV_EXPORTS BruteForceMatcher_GPU_base
        {
        public:
            enum DistType {L1Dist = 0, L2Dist, HammingDist};

            explicit BruteForceMatcher_GPU_base(DistType distType = L2Dist);

            // Add descriptors to train descriptor collection.
            void add(const std::vector<GpuMat>& descCollection);

            // Get train descriptors collection.
            const std::vector<GpuMat>& getTrainDescriptors() const;

            // Clear train descriptors collection.
            void clear();

            // Return true if there are not train descriptors in collection.
            bool empty() const;

            // Return true if the matcher supports mask in match methods.
            bool isMaskSupported() const;

            // Find one best match for each query descriptor.
            // trainIdx.at<int>(0, queryIdx) will contain best train index for queryIdx
            // distance.at<float>(0, queryIdx) will contain distance
            void matchSingle(const GpuMat& queryDescs, const GpuMat& trainDescs,
                GpuMat& trainIdx, GpuMat& distance,
                const GpuMat& mask = GpuMat(), Stream& stream = Stream::Null());

            // Download trainIdx and distance and convert it to CPU vector with DMatch
            static void matchDownload(const GpuMat& trainIdx, const GpuMat& distance, std::vector<DMatch>& matches);
            // Convert trainIdx and distance to vector with DMatch
            static void matchConvert(const Mat& trainIdx, const Mat& distance, std::vector<DMatch>& matches);

            // Find one best match for each query descriptor.
            void match(const GpuMat& queryDescs, const GpuMat& trainDescs, std::vector<DMatch>& matches,
                const GpuMat& mask = GpuMat());

            // Make gpu collection of trains and masks in suitable format for matchCollection function
            void makeGpuCollection(GpuMat& trainCollection, GpuMat& maskCollection,
                const vector<GpuMat>& masks = std::vector<GpuMat>());

            // Find one best match from train collection for each query descriptor.
            // trainIdx.at<int>(0, queryIdx) will contain best train index for queryIdx
            // imgIdx.at<int>(0, queryIdx) will contain best image index for queryIdx
            // distance.at<float>(0, queryIdx) will contain distance
            void matchCollection(const GpuMat& queryDescs, const GpuMat& trainCollection,
                GpuMat& trainIdx, GpuMat& imgIdx, GpuMat& distance,
                const GpuMat& maskCollection, Stream& stream = Stream::Null());

            // Download trainIdx, imgIdx and distance and convert it to vector with DMatch
            static void matchDownload(const GpuMat& trainIdx, const GpuMat& imgIdx, const GpuMat& distance, std::vector<DMatch>& matches);
            // Convert trainIdx, imgIdx and distance to vector with DMatch
            static void matchConvert(const Mat& trainIdx, const Mat& imgIdx, const Mat& distance, std::vector<DMatch>& matches);

            // Find one best match from train collection for each query descriptor.
            void match(const GpuMat& queryDescs, std::vector<DMatch>& matches, const std::vector<GpuMat>& masks = std::vector<GpuMat>());

            // Find k best matches for each query descriptor (in increasing order of distances).
            // trainIdx.at<int>(queryIdx, i) will contain index of i'th best trains (i < k).
            // distance.at<float>(queryIdx, i) will contain distance.
            // allDist is a buffer to store all distance between query descriptors and train descriptors
            // it have size (nQuery,nTrain) and CV_32F type
            // allDist.at<float>(queryIdx, trainIdx) will contain FLT_MAX, if trainIdx is one from k best,
            // otherwise it will contain distance between queryIdx and trainIdx descriptors
            void knnMatch(const GpuMat& queryDescs, const GpuMat& trainDescs,
                GpuMat& trainIdx, GpuMat& distance, GpuMat& allDist, int k, const GpuMat& mask = GpuMat(), Stream& stream = Stream::Null());

            // Download trainIdx and distance and convert it to vector with DMatch
            // compactResult is used when mask is not empty. If compactResult is false matches
            // vector will have the same size as queryDescriptors rows. If compactResult is true
            // matches vector will not contain matches for fully masked out query descriptors.
            static void knnMatchDownload(const GpuMat& trainIdx, const GpuMat& distance,
                std::vector< std::vector<DMatch> >& matches, bool compactResult = false);
            // Convert trainIdx and distance to vector with DMatch
            static void knnMatchConvert(const Mat& trainIdx, const Mat& distance,
                std::vector< std::vector<DMatch> >& matches, bool compactResult = false);

            // Find k best matches for each query descriptor (in increasing order of distances).
            // compactResult is used when mask is not empty. If compactResult is false matches
            // vector will have the same size as queryDescriptors rows. If compactResult is true
            // matches vector will not contain matches for fully masked out query descriptors.
            void knnMatch(const GpuMat& queryDescs, const GpuMat& trainDescs,
                std::vector< std::vector<DMatch> >& matches, int k, const GpuMat& mask = GpuMat(),
                bool compactResult = false);

            // Find k best matches  for each query descriptor (in increasing order of distances).
            // compactResult is used when mask is not empty. If compactResult is false matches
            // vector will have the same size as queryDescriptors rows. If compactResult is true
            // matches vector will not contain matches for fully masked out query descriptors.
            void knnMatch(const GpuMat& queryDescs, std::vector< std::vector<DMatch> >& matches, int knn,
                const std::vector<GpuMat>& masks = std::vector<GpuMat>(), bool compactResult = false );

            // Find best matches for each query descriptor which have distance less than maxDistance.
            // nMatches.at<unsigned int>(0, queruIdx) will contain matches count for queryIdx.
            // carefully nMatches can be greater than trainIdx.cols - it means that matcher didn't find all matches,
            // because it didn't have enough memory.
            // trainIdx.at<int>(queruIdx, i) will contain ith train index (i < min(nMatches.at<unsigned int>(0, queruIdx), trainIdx.cols))
            // distance.at<int>(queruIdx, i) will contain ith distance (i < min(nMatches.at<unsigned int>(0, queruIdx), trainIdx.cols))
            // If trainIdx is empty, then trainIdx and distance will be created with size nQuery x nTrain,
            // otherwize user can pass own allocated trainIdx and distance with size nQuery x nMaxMatches
            // Matches doesn't sorted.
            void radiusMatch(const GpuMat& queryDescs, const GpuMat& trainDescs,
                GpuMat& trainIdx, GpuMat& nMatches, GpuMat& distance, float maxDistance,
                const GpuMat& mask = GpuMat(), Stream& stream = Stream::Null());

            // Download trainIdx, nMatches and distance and convert it to vector with DMatch.
            // matches will be sorted in increasing order of distances.
            // compactResult is used when mask is not empty. If compactResult is false matches
            // vector will have the same size as queryDescriptors rows. If compactResult is true
            // matches vector will not contain matches for fully masked out query descriptors.
            static void radiusMatchDownload(const GpuMat& trainIdx, const GpuMat& nMatches, const GpuMat& distance,
                std::vector< std::vector<DMatch> >& matches, bool compactResult = false);
            // Convert trainIdx, nMatches and distance to vector with DMatch.
            static void radiusMatchConvert(const Mat& trainIdx, const Mat& nMatches, const Mat& distance,
                std::vector< std::vector<DMatch> >& matches, bool compactResult = false);

            // Find best matches for each query descriptor which have distance less than maxDistance
            // in increasing order of distances).
            void radiusMatch(const GpuMat& queryDescs, const GpuMat& trainDescs,
                std::vector< std::vector<DMatch> >& matches, float maxDistance,
                const GpuMat& mask = GpuMat(), bool compactResult = false);

            // Find best matches from train collection for each query descriptor which have distance less than
            // maxDistance (in increasing order of distances).
            void radiusMatch(const GpuMat& queryDescs, std::vector< std::vector<DMatch> >& matches, float maxDistance,
                const std::vector<GpuMat>& masks = std::vector<GpuMat>(), bool compactResult = false);

            DistType distType;

        private:
            std::vector<GpuMat> trainDescCollection;
        };

        template <class Distance>
        class CV_EXPORTS BruteForceMatcher_GPU;

        template <typename T>
        class CV_EXPORTS BruteForceMatcher_GPU< L1<T> > : public BruteForceMatcher_GPU_base
        {
        public:
            explicit BruteForceMatcher_GPU() : BruteForceMatcher_GPU_base(L1Dist) {}
            explicit BruteForceMatcher_GPU(L1<T> /*d*/) : BruteForceMatcher_GPU_base(L1Dist) {}
        };
        template <typename T>
        class CV_EXPORTS BruteForceMatcher_GPU< L2<T> > : public BruteForceMatcher_GPU_base
        {
        public:
            explicit BruteForceMatcher_GPU() : BruteForceMatcher_GPU_base(L2Dist) {}
            explicit BruteForceMatcher_GPU(L2<T> /*d*/) : BruteForceMatcher_GPU_base(L2Dist) {}
        };
        template <> class CV_EXPORTS BruteForceMatcher_GPU< HammingLUT > : public BruteForceMatcher_GPU_base
        {
        public:
            explicit BruteForceMatcher_GPU() : BruteForceMatcher_GPU_base(HammingDist) {}
            explicit BruteForceMatcher_GPU(HammingLUT /*d*/) : BruteForceMatcher_GPU_base(HammingDist) {}
        };
        template <> class CV_EXPORTS BruteForceMatcher_GPU< Hamming > : public BruteForceMatcher_GPU_base
        {
        public:
            explicit BruteForceMatcher_GPU() : BruteForceMatcher_GPU_base(HammingDist) {}
            explicit BruteForceMatcher_GPU(Hamming /*d*/) : BruteForceMatcher_GPU_base(HammingDist) {}
        };

        ////////////////////////////////// CascadeClassifier_GPU //////////////////////////////////////////
        // The cascade classifier class for object detection.
        class CV_EXPORTS CascadeClassifier_GPU
        {
        public:
            CascadeClassifier_GPU();
            CascadeClassifier_GPU(const string& filename);
            ~CascadeClassifier_GPU();

            bool empty() const;
            bool load(const string& filename);
            void release();

            /* returns number of detected objects */
            int detectMultiScale( const GpuMat& image, GpuMat& objectsBuf, double scaleFactor=1.2, int minNeighbors=4, Size minSize=Size());

            bool findLargestObject;
            bool visualizeInPlace;

            Size getClassifierSize() const;
        private:

            struct CascadeClassifierImpl;
            CascadeClassifierImpl* impl;
        };

        ////////////////////////////////// SURF //////////////////////////////////////////

        class CV_EXPORTS SURF_GPU : public CvSURFParams
        {
        public:
            enum KeypointLayout 
            {
                SF_X = 0,
                SF_Y,
                SF_LAPLACIAN,
                SF_SIZE,
                SF_DIR,
                SF_HESSIAN,
                SF_FEATURE_STRIDE
            };

            //! the default constructor
            SURF_GPU();
            //! the full constructor taking all the necessary parameters
            explicit SURF_GPU(double _hessianThreshold, int _nOctaves=4,
                 int _nOctaveLayers=2, bool _extended=false, float _keypointsRatio=0.01f, bool _upright = false);

            //! returns the descriptor size in float's (64 or 128)
            int descriptorSize() const;

            //! upload host keypoints to device memory
            void uploadKeypoints(const vector<KeyPoint>& keypoints, GpuMat& keypointsGPU);
            //! download keypoints from device to host memory
            void downloadKeypoints(const GpuMat& keypointsGPU, vector<KeyPoint>& keypoints);

            //! download descriptors from device to host memory
            void downloadDescriptors(const GpuMat& descriptorsGPU, vector<float>& descriptors);
            
            //! finds the keypoints using fast hessian detector used in SURF
            //! supports CV_8UC1 images
            //! keypoints will have nFeature cols and 6 rows
            //! keypoints.ptr<float>(SF_X)[i] will contain x coordinate of i'th feature
            //! keypoints.ptr<float>(SF_Y)[i] will contain y coordinate of i'th feature
            //! keypoints.ptr<float>(SF_LAPLACIAN)[i] will contain laplacian sign of i'th feature
            //! keypoints.ptr<float>(SF_SIZE)[i] will contain size of i'th feature
            //! keypoints.ptr<float>(SF_DIR)[i] will contain orientation of i'th feature
            //! keypoints.ptr<float>(SF_HESSIAN)[i] will contain response of i'th feature
            void operator()(const GpuMat& img, const GpuMat& mask, GpuMat& keypoints);
            //! finds the keypoints and computes their descriptors. 
            //! Optionally it can compute descriptors for the user-provided keypoints and recompute keypoints direction
            void operator()(const GpuMat& img, const GpuMat& mask, GpuMat& keypoints, GpuMat& descriptors, 
                bool useProvidedKeypoints = false);

            void operator()(const GpuMat& img, const GpuMat& mask, std::vector<KeyPoint>& keypoints);
            void operator()(const GpuMat& img, const GpuMat& mask, std::vector<KeyPoint>& keypoints, GpuMat& descriptors, 
                bool useProvidedKeypoints = false);

            void operator()(const GpuMat& img, const GpuMat& mask, std::vector<KeyPoint>& keypoints, std::vector<float>& descriptors, 
                bool useProvidedKeypoints = false);

            void releaseMemory();

            //! max keypoints = min(keypointsRatio * img.size().area(), 65535)
            float keypointsRatio;

            GpuMat sum, mask1, maskSum, intBuffer;

            GpuMat det, trace;

            GpuMat maxPosBuffer;
        };

    }

    //! Speckle filtering - filters small connected components on diparity image.
    //! It sets pixel (x,y) to newVal if it coresponds to small CC with size < maxSpeckleSize.
    //! Threshold for border between CC is diffThreshold;
    CV_EXPORTS void filterSpeckles( Mat& img, uchar newVal, int maxSpeckleSize, uchar diffThreshold, Mat& buf);

}
#include "opencv2/gpu/matrix_operations.hpp"

#endif /* __OPENCV_GPU_HPP__ */
