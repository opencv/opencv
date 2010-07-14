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

#include "opencv2/core/core.hpp"
#include "opencv2/gpu/devmem2d.hpp"

namespace cv
{
    namespace gpu
    {   
        //////////////////////////////// Initialization ////////////////////////
                 
        CV_EXPORTS int getCudaEnabledDeviceCount();
        CV_EXPORTS string getDeviceName(int device);
        CV_EXPORTS void setDevice(int device);        

        enum { CV_GPU_CC_10, CV_GPU_CC_11, CV_GPU_CC_12, CV_GPU_CC_13, CV_GPU_CC_20 };

        CV_EXPORTS int getComputeCapability(int device);
        CV_EXPORTS int getNumberOfSMs(int device);
 
        //////////////////////////////// GpuMat ////////////////////////////////

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
            GpuMat (const Mat& m);

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

            //! Downloads data from device to host memory. Blocking calls.
            operator Mat() const;
            void download(cv::Mat& m) const;       

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

        //////////////////////////////// CudaStream ////////////////////////////////

        class CudaStream
        {
        public:
            CudaStream(); 
            ~CudaStream();

            bool queryIfComplete();
            void waitForCompletion(); 

            //calls cudaMemcpyAsync
            void enqueueDownload(const GpuMat& src, Mat& dst);
            void enqueueUpload(const Mat& src, GpuMat& dst);
            void enqueueCopy(const GpuMat& src, GpuMat& dst);

            // calls cudaMemset2D asynchronous for single channel. Invoke kernel for some multichannel.
            void enqueueMemSet(const GpuMat& src, Scalar val);

            // invoke kernel asynchronous because of mask
            void enqueueMemSet(const GpuMat& src, Scalar val, const GpuMat& mask);

            // converts matrix type, ex from float to uchar depending on type
            void enqueueConvert(const GpuMat& src, GpuMat& dst, int type); 

            //CUstream_st& getStream();
        private:
            void *impl;
            
            CudaStream(const CudaStream&); 
            CudaStream& operator=(const CudaStream&);
        };

        //////////////////////////////// StereoBM_GPU ////////////////////////////////

        class CV_EXPORTS StereoBM_GPU
        {
        public:
            enum { BASIC_PRESET=0, PREFILTER_XSOBEL = 1 };

            //! the default constructor
            StereoBM_GPU();
            //! the full constructor taking the camera-specific preset, number of disparities and the SAD window size
            //! ndisparities should be multiple of 8. SSD WindowsSize is fixed to 19 now
            StereoBM_GPU(int preset, int ndisparities=0);
            //! the stereo correspondence operator. Finds the disparity for the specified rectified stereo pair
            //! Output disparity has CV_8U type.
            void operator() ( const GpuMat& left, const GpuMat& right, GpuMat& disparity) const;            
        private:
            mutable GpuMat minSSD;
            int preset;
            int ndisp;
        };
    }
}



#include "opencv2/gpu/gpumat.hpp"

#endif /* __OPENCV_GPU_HPP__ */