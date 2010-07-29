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
#include "opencv2/gpu/devmem2d.hpp"

namespace cv
{
    namespace gpu
    {
        //////////////////////////////// Initialization ////////////////////////

        //! This is the only function that do not throw exceptions if the library is compiled without Cuda.
        CV_EXPORTS int getCudaEnabledDeviceCount();

        //! Functions below throw cv::Expception if the library is compiled without Cuda.
        CV_EXPORTS string getDeviceName(int device);
        CV_EXPORTS void setDevice(int device);
        CV_EXPORTS int getDevice();

        CV_EXPORTS void getComputeCapability(int device, int* major, int* minor);
        CV_EXPORTS int getNumberOfSMs(int device);

        //////////////////////////////// GpuMat ////////////////////////////////
        class CudaStream;
        class MatPL;

        //! Smart pointer for GPU memory with reference counting. Its interface is mostly similar with cv::Mat.
        class CV_EXPORTS GpuMat
        {
        public:
            //! default constructor
            GpuMat();
            //! constructs GpuMatrix of the specified size and type
            // (_type is CV_8UC1, CV_64FC3, CV_32SC(12) etc.)
            GpuMat(int _rows, int _cols, int _type);
            GpuMat(Size _size, int _type);
            //! constucts GpuMatrix and fills it with the specified value _s.
            GpuMat(int _rows, int _cols, int _type, const Scalar& _s);
            GpuMat(Size _size, int _type, const Scalar& _s);
            //! copy constructor
            GpuMat(const GpuMat& m);

            //! constructor for GpuMatrix headers pointing to user-allocated data
            GpuMat(int _rows, int _cols, int _type, void* _data, size_t _step = Mat::AUTO_STEP);
            GpuMat(Size _size, int _type, void* _data, size_t _step = Mat::AUTO_STEP);

            //! creates a matrix header for a part of the bigger matrix
            GpuMat(const GpuMat& m, const Range& rowRange, const Range& colRange);
            GpuMat(const GpuMat& m, const Rect& roi);

            //! builds GpuMat from Mat. Perfom blocking upload to device.
            explicit GpuMat (const Mat& m);

            //! destructor - calls release()
            ~GpuMat();

            //! assignment operators
            GpuMat& operator = (const GpuMat& m);
            //! assignment operator. Perfom blocking upload to device.
            GpuMat& operator = (const Mat& m);

            //! returns lightweight DevMem2D_ structure for passing to nvcc-compiled code.
            // Contains just image size, data ptr and step.
            template <class T> operator DevMem2D_<T>() const;

            //! pefroms blocking upload data to GpuMat. .
            void upload(const cv::Mat& m);
            void upload(const MatPL& m, CudaStream& stream);

            //! Downloads data from device to host memory. Blocking calls.
            operator Mat() const;
            void download(cv::Mat& m) const;
            void download(MatPL& m, CudaStream& stream) const;

            //! returns a new GpuMatrix header for the specified row
            GpuMat row(int y) const;
            //! returns a new GpuMatrix header for the specified column
            GpuMat col(int x) const;
            //! ... for the specified row span
            GpuMat rowRange(int startrow, int endrow) const;
            GpuMat rowRange(const Range& r) const;
            //! ... for the specified column span
            GpuMat colRange(int startcol, int endcol) const;
            GpuMat colRange(const Range& r) const;

            //! returns deep copy of the GpuMatrix, i.e. the data is copied
            GpuMat clone() const;
            //! copies the GpuMatrix content to "m".
            // It calls m.create(this->size(), this->type()).
            void copyTo( GpuMat& m ) const;
            //! copies those GpuMatrix elements to "m" that are marked with non-zero mask elements.
            void copyTo( GpuMat& m, const GpuMat& mask ) const;
            //! converts GpuMatrix to another datatype with optional scalng. See cvConvertScale.
            void convertTo( GpuMat& m, int rtype, double alpha=1, double beta=0 ) const;

            void assignTo( GpuMat& m, int type=-1 ) const;

            //! sets every GpuMatrix element to s
            GpuMat& operator = (const Scalar& s);
            //! sets some of the GpuMatrix elements to s, according to the mask
            GpuMat& setTo(const Scalar& s, const GpuMat& mask=GpuMat());
            //! creates alternative GpuMatrix header for the same data, with different
            // number of channels and/or different number of rows. see cvReshape.
            GpuMat reshape(int _cn, int _rows=0) const;

            //! allocates new GpuMatrix data unless the GpuMatrix already has specified size and type.
            // previous data is unreferenced if needed.
            void create(int _rows, int _cols, int _type);
            void create(Size _size, int _type);
            //! decreases reference counter;
            // deallocate the data when reference counter reaches 0.
            void release();

            //! swaps with other smart pointer
            void swap(GpuMat& mat);

            //! locates GpuMatrix header within a parent GpuMatrix. See below
            void locateROI( Size& wholeSize, Point& ofs ) const;
            //! moves/resizes the current GpuMatrix ROI inside the parent GpuMatrix.
            GpuMat& adjustROI( int dtop, int dbottom, int dleft, int dright );
            //! extracts a rectangular sub-GpuMatrix
            // (this is a generalized form of row, rowRange etc.)
            GpuMat operator()( Range rowRange, Range colRange ) const;
            GpuMat operator()( const Rect& roi ) const;

            //! returns true iff the GpuMatrix data is continuous
            // (i.e. when there are no gaps between successive rows).
            // similar to CV_IS_GpuMat_CONT(cvGpuMat->type)
            bool isContinuous() const;
            //! returns element size in bytes,
            // similar to CV_ELEM_SIZE(cvMat->type)
            size_t elemSize() const;
            //! returns the size of element channel in bytes.
            size_t elemSize1() const;
            //! returns element type, similar to CV_MAT_TYPE(cvMat->type)
            int type() const;
            //! returns element type, similar to CV_MAT_DEPTH(cvMat->type)
            int depth() const;
            //! returns element type, similar to CV_MAT_CN(cvMat->type)
            int channels() const;
            //! returns step/elemSize1()
            size_t step1() const;
            //! returns GpuMatrix size:
            // width == number of columns, height == number of rows
            Size size() const;
            //! returns true if GpuMatrix data is NULL
            bool empty() const;

            //! returns pointer to y-th row
            uchar* ptr(int y=0);
            const uchar* ptr(int y=0) const;

            //! template version of the above method
            template<typename _Tp> _Tp* ptr(int y=0);
            template<typename _Tp> const _Tp* ptr(int y=0) const;

            /*! includes several bit-fields:
            - the magic signature
            - continuity flag
            - depth
            - number of channels
            */
            int flags;
            //! the number of rows and columns
            int rows, cols;
            //! a distance between successive rows in bytes; includes the gap if any
            size_t step;
            //! pointer to the data
            uchar* data;

            //! pointer to the reference counter;
            // when GpuMatrix points to user-allocated data, the pointer is NULL
            int* refcount;

            //! helper fields used in locateROI and adjustROI
            uchar* datastart;
            uchar* dataend;
        };

        //////////////////////////////// MatPL ////////////////////////////////
        // MatPL is limited cv::Mat with page locked memory allocation.
        // Page locked memory is only needed for async and faster coping to GPU.
        // It is convertable to cv::Mat header without reference counting
        // so you can use it with other opencv functions.

        class CV_EXPORTS MatPL
        {
        public:

            //Not supported.  Now behaviour is like ALLOC_DEFAULT.
            //enum { ALLOC_DEFAULT = 0, ALLOC_PORTABLE = 1, ALLOC_WRITE_COMBINED = 4 }

            MatPL();
            MatPL(const MatPL& m);

            MatPL(int _rows, int _cols, int _type);
            MatPL(Size _size, int _type);

            //! creates from cv::Mat with coping data
            explicit MatPL(const Mat& m);

            ~MatPL();

            MatPL& operator = (const MatPL& m);

            //! returns deep copy of the matrix, i.e. the data is copied
            MatPL clone() const;

            //! allocates new matrix data unless the matrix already has specified size and type.
            void create(int _rows, int _cols, int _type);
            void create(Size _size, int _type);

            //! decrements reference counter and released memory if needed.
            void release();

            //! returns matrix header with disabled reference counting for MatPL data.
            Mat createMatHeader() const;
            operator Mat() const;

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
        };

        //////////////////////////////// CudaStream ////////////////////////////////
        // Encapculates Cuda Stream. Provides interface for async coping.
        // Passed to each function that supports async kernel execution.
        // Reference counting is enabled

        class CV_EXPORTS CudaStream
        {
        public:
            CudaStream();
            ~CudaStream();

            CudaStream(const CudaStream&);
            CudaStream& operator=(const CudaStream&);

            bool queryIfComplete();
            void waitForCompletion();

            //! downloads asynchronously.
            // Warning! cv::Mat must point to page locked memory (i.e. to MatPL data or to its subMat)
            void enqueueDownload(const GpuMat& src, MatPL& dst);
            void enqueueDownload(const GpuMat& src, Mat& dst);

            //! uploads asynchronously.
            // Warning! cv::Mat must point to page locked memory (i.e. to MatPL data or to its ROI)
            void enqueueUpload(const MatPL& src, GpuMat& dst);
            void enqueueUpload(const Mat& src, GpuMat& dst);

            void enqueueCopy(const GpuMat& src, GpuMat& dst);

            void enqueueMemSet(const GpuMat& src, Scalar val);
            void enqueueMemSet(const GpuMat& src, Scalar val, const GpuMat& mask);

            // converts matrix type, ex from float to uchar depending on type
            void enqueueConvert(const GpuMat& src, GpuMat& dst, int type, double a = 1, double b = 0);
        private:
            void create();
            void release();
            struct Impl;
            Impl *impl;
            friend struct StreamAccessor;
        };

        ////////////////////////////// Image processing //////////////////////////////

        void CV_EXPORTS remap(const GpuMat& src, const GpuMat& xmap, const GpuMat& ymap, GpuMat& dst);

        //////////////////////////////// StereoBM_GPU ////////////////////////////////

        class CV_EXPORTS StereoBM_GPU
        {
        public:
            enum { BASIC_PRESET = 0, PREFILTER_XSOBEL = 1 };

            enum { DEFAULT_NDISP = 64, DEFAULT_WINSZ = 19 };

            //! the default constructor
            StereoBM_GPU();
            //! the full constructor taking the camera-specific preset, number of disparities and the SAD window size
            //! ndisparities should be multiple of 8. SSD WindowsSize is fixed to 19 now
            StereoBM_GPU(int preset, int ndisparities = DEFAULT_NDISP, int winSize = DEFAULT_WINSZ);

            //! the stereo correspondence operator. Finds the disparity for the specified rectified stereo pair
            //! Output disparity has CV_8U type.
            void operator() ( const GpuMat& left, const GpuMat& right, GpuMat& disparity);

            //! Acync version
            void operator() ( const GpuMat& left, const GpuMat& right, GpuMat& disparity, const CudaStream & stream);

            //! Some heuristics that tries to estmate
            // if current GPU will be faster then CPU in this algorithm.
            // It queries current active device.
            static bool checkIfGpuCallReasonable();

            int ndisp;
            int winSize;
            int preset;

            // If avergeTexThreshold  == 0 => post procesing is disabled
            // If avergeTexThreshold != 0 then disparity is set 0 in each point (x,y) where for left image
            // SumOfHorizontalGradiensInWindow(x, y, winSize) < (winSize * winSize) * avergeTexThreshold
            // i.e. input left image is low textured.
            float avergeTexThreshold;
        private:
            GpuMat minSSD, leBuf, riBuf;
        };
        
        //////////////////////// StereoBeliefPropagation_GPU /////////////////////////
        
        class CV_EXPORTS StereoBeliefPropagation_GPU
        {
        public:
            enum { DEFAULT_NDISP  = 64 };
            enum { DEFAULT_ITERS  = 5  };
            enum { DEFAULT_LEVELS = 5  };

            //! the default constructor
            explicit StereoBeliefPropagation_GPU(int ndisp  = DEFAULT_NDISP, 
                                                 int iters  = DEFAULT_ITERS, 
                                                 int levels = DEFAULT_LEVELS);
            //! the full constructor taking the number of disparities, number of BP iterations on first level,
            //! number of levels, truncation of discontinuity cost, truncation of data cost and weighting of data cost.
            StereoBeliefPropagation_GPU(int ndisp, int iters, int levels, float disc_cost, float data_cost, float lambda);

            //! the stereo correspondence operator. Finds the disparity for the specified rectified stereo pair,
            //! if disparity is empty output type will be CV_32S else output type will be disparity.type().
            void operator()(const GpuMat& left, const GpuMat& right, GpuMat& disparity);
            
            //! Acync version
            void operator()(const GpuMat& left, const GpuMat& right, GpuMat& disparity, const CudaStream& stream);

            //! Some heuristics that tries to estmate
            //! if current GPU will be faster then CPU in this algorithm.
            //! It queries current active device.
            static bool checkIfGpuCallReasonable();
            
            int ndisp;

            int iters;
            int levels;
            
            float disc_cost;
            float data_cost;
            float lambda;
        private:
            GpuMat u, d, l, r, u2, d2, l2, r2;
            std::vector<GpuMat> datas;            
            GpuMat out;
        };        
    }
}
#include "opencv2/gpu/matrix_operations.hpp"

#endif /* __OPENCV_GPU_HPP__ */
