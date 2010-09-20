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

        CV_EXPORTS void getComputeCapability(int device, int& major, int& minor);
        CV_EXPORTS int getNumberOfSMs(int device);

        CV_EXPORTS void getGpuMemInfo(size_t& free, size_t& total);

        //////////////////////////////// GpuMat ////////////////////////////////
        class Stream;
        class CudaMem;

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

            //! upload async
            void upload(const CudaMem& m, Stream& stream);

            //! downloads data from device to host memory. Blocking calls.
            operator Mat() const;
            void download(cv::Mat& m) const;

            //! download async
            void download(CudaMem& m, Stream& stream) const;

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

            //! matrix transposition
            GpuMat t() const;

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

        //////////////////////////////// CudaMem ////////////////////////////////
        // CudaMem is limited cv::Mat with page locked memory allocation.
        // Page locked memory is only needed for async and faster coping to GPU.
        // It is convertable to cv::Mat header without reference counting
        // so you can use it with other opencv functions.

        class CV_EXPORTS CudaMem
        {
        public:
            enum  { ALLOC_PAGE_LOCKED = 1, ALLOC_ZEROCOPY = 2, ALLOC_WRITE_COMBINED = 4 };

            CudaMem();
            CudaMem(const CudaMem& m);

            CudaMem(int _rows, int _cols, int _type, int _alloc_type = ALLOC_PAGE_LOCKED);
            CudaMem(Size _size, int _type, int _alloc_type = ALLOC_PAGE_LOCKED);


            //! creates from cv::Mat with coping data
            explicit CudaMem(const Mat& m, int _alloc_type = ALLOC_PAGE_LOCKED);

            ~CudaMem();

            CudaMem& operator = (const CudaMem& m);

            //! returns deep copy of the matrix, i.e. the data is copied
            CudaMem clone() const;

            //! allocates new matrix data unless the matrix already has specified size and type.
            void create(int _rows, int _cols, int _type, int _alloc_type = ALLOC_PAGE_LOCKED);
            void create(Size _size, int _type, int _alloc_type = ALLOC_PAGE_LOCKED);

            //! decrements reference counter and released memory if needed.
            void release();

            //! returns matrix header with disabled reference counting for CudaMem data.
            Mat createMatHeader() const;
            operator Mat() const;

            //! maps host memory into device address space and returns GpuMat header for it. Throws exception if not supported by hardware.
            GpuMat createGpuMatHeader() const;
            operator GpuMat() const;

            //returns if host memory can be mapperd to gpu address space;
            static bool can_device_map_to_host();

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

        ////////////////////////////// Arithmetics ///////////////////////////////////

        //! adds one matrix to another (c = a + b)
        CV_EXPORTS void add(const GpuMat& a, const GpuMat& b, GpuMat& c);
        //! subtracts one matrix from another (c = a - b)
		CV_EXPORTS void subtract(const GpuMat& a, const GpuMat& b, GpuMat& c);
        //! computes element-wise product of the two arrays (c = a * b)
		CV_EXPORTS void multiply(const GpuMat& a, const GpuMat& b, GpuMat& c);
        //! computes element-wise quotient of the two arrays (c = a / b)
		CV_EXPORTS void divide(const GpuMat& a, const GpuMat& b, GpuMat& c);

        //! transposes the matrix
		CV_EXPORTS void transpose(const GpuMat& src1, GpuMat& dst);

        //! computes element-wise absolute difference of two arrays (c = abs(a - b))
		CV_EXPORTS void absdiff(const GpuMat& a, const GpuMat& b, GpuMat& c);

        //! applies fixed threshold to the image. 
        //! Now supports only THRESH_TRUNC threshold type and one channels float source.
        CV_EXPORTS double threshold(const GpuMat& src, GpuMat& dst, double thresh);

        //! compares elements of two arrays (c = a <cmpop> b)
        //! Now doesn't support CMP_NE.
        CV_EXPORTS void compare(const GpuMat& a, const GpuMat& b, GpuMat& c, int cmpop);

        //! computes mean value and standard deviation of all or selected array elements
        CV_EXPORTS void meanStdDev(const GpuMat& mtx, Scalar& mean, Scalar& stddev);

        //! computes norm of array
        //! Supports NORM_INF, NORM_L1, NORM_L2
        CV_EXPORTS double norm(const GpuMat& src1, int normType=NORM_L2);
        //! computes norm of the difference between two arrays
        //! Supports NORM_INF, NORM_L1, NORM_L2
        CV_EXPORTS double norm(const GpuMat& src1, const GpuMat& src2, int normType=NORM_L2);

        //! reverses the order of the rows, columns or both in a matrix
        CV_EXPORTS void flip(const GpuMat& a, GpuMat& b, int flipCode);

        //! resizes the image
        //! Supports INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4
        CV_EXPORTS void resize(const GpuMat& src, GpuMat& dst, Size dsize, double fx=0, double fy=0, int interpolation = INTER_LINEAR);

        //! computes sum of array elements
        CV_EXPORTS Scalar sum(const GpuMat& m);

        //! finds global minimum and maximum array elements and returns their values
        CV_EXPORTS void minMax(const GpuMat& src, double* minVal, double* maxVal = 0);

        //! copies 2D array to a larger destination array and pads borders with user-specifiable constant
        CV_EXPORTS void copyMakeBorder(const GpuMat& src, GpuMat& dst, int top, int bottom, int left, int right, const Scalar& value = Scalar());

        //! warps the image using affine transformation
        //! Supports INTER_NEAREST, INTER_LINEAR, INTER_CUBIC
        CV_EXPORTS void warpAffine(const GpuMat& src, GpuMat& dst, const Mat& M, Size dsize, int flags = INTER_LINEAR);

        //! warps the image using perspective transformation
        //! Supports INTER_NEAREST, INTER_LINEAR, INTER_CUBIC
        CV_EXPORTS void warpPerspective(const GpuMat& src, GpuMat& dst, const Mat& M, Size dsize, int flags = INTER_LINEAR);

        //! rotate 8bit single or four channel image
        //! Supports INTER_NEAREST, INTER_LINEAR, INTER_CUBIC
        CV_EXPORTS void rotate(const GpuMat& src, GpuMat& dst, Size dsize, double angle, double xShift = 0, double yShift = 0, int interpolation = INTER_LINEAR);

        ////////////////////////////// Image processing //////////////////////////////

        // DST[x,y] = SRC[xmap[x,y],ymap[x,y]] with bilinear interpolation.
        // xymap.type() == xymap.type() == CV_32FC1
        CV_EXPORTS void remap(const GpuMat& src, GpuMat& dst, const GpuMat& xmap, const GpuMat& ymap);

        // Does mean shift filtering on GPU.
        CV_EXPORTS void meanShiftFiltering(const GpuMat& src, GpuMat& dst, int sp, int sr, 
            TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1));

        // Does coloring of disparity image: [0..ndisp) -> [0..240, 1, 1] in HSV.
        // Supported types of input disparity: CV_8U, CV_16S.
        // Output disparity has CV_8UC4 type in BGRA format (alpha = 255).
        CV_EXPORTS void drawColorDisp(const GpuMat& src_disp, GpuMat& dst_disp, int ndisp);
        // Acync version
        CV_EXPORTS void drawColorDisp(const GpuMat& src_disp, GpuMat& dst_disp, int ndisp, const Stream& stream);

        // Reprojects disparity image to 3D space. 
        // Supports CV_8U and CV_16S types of input disparity.
        // The output is a 4-channel floating-point (CV_32FC4) matrix. 
        // Each element of this matrix will contain the 3D coordinates of the point (x,y,z,1), computed from the disparity map.
        // Q is the 4x4 perspective transformation matrix that can be obtained with cvStereoRectify.
        CV_EXPORTS void reprojectImageTo3D(const GpuMat& disp, GpuMat& xyzw, const Mat& Q);
        // Acync version
        CV_EXPORTS void reprojectImageTo3D(const GpuMat& disp, GpuMat& xyzw, const Mat& Q, const Stream& stream);

        CV_EXPORTS void cvtColor(const GpuMat& src, GpuMat& dst, int code, int dcn = 0);
        CV_EXPORTS void cvtColor(const GpuMat& src, GpuMat& dst, int code, int dcn, const Stream& stream);


        //! erodes the image (applies the local minimum operator)
        CV_EXPORTS void erode( const GpuMat& src, GpuMat& dst, const Mat& kernel, Point anchor, int iterations);

        //! dilates the image (applies the local maximum operator)
        CV_EXPORTS void dilate( const GpuMat& src, GpuMat& dst, const Mat& kernel, Point anchor, int iterations);

        //! applies an advanced morphological operation to the image
        CV_EXPORTS void morphologyEx( const GpuMat& src, GpuMat& dst, int op, const Mat& kernel, Point anchor, int iterations);

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
            void operator() ( const GpuMat& left, const GpuMat& right, GpuMat& disparity);

            //! Acync version
            void operator() ( const GpuMat& left, const GpuMat& right, GpuMat& disparity, const Stream & stream);

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
            StereoBeliefPropagation(int ndisp, int iters, int levels,
                                    float max_data_term, float data_weight,
                                    float max_disc_term, float disc_single_jump,
                                    int msg_type = CV_32F);

            //! the stereo correspondence operator. Finds the disparity for the specified rectified stereo pair,
            //! if disparity is empty output type will be CV_16S else output type will be disparity.type().
            void operator()(const GpuMat& left, const GpuMat& right, GpuMat& disparity);

            //! Acync version
            void operator()(const GpuMat& left, const GpuMat& right, GpuMat& disparity, Stream& stream);


            //! version for user specified data term
            void operator()(const GpuMat& data, GpuMat& disparity);
            void operator()(const GpuMat& data, GpuMat& disparity, Stream& stream);

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
        // Qingxiong Yang, Liang Wang†, Narendra Ahuja
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
            void operator()(const GpuMat& left, const GpuMat& right, GpuMat& disparity);

            //! Acync version
            void operator()(const GpuMat& left, const GpuMat& right, GpuMat& disparity, Stream& stream);

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
        // Qingxiong Yang, Liang Wang†, Narendra Ahuja
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
            void operator()(const GpuMat& disparity, const GpuMat& image, GpuMat& dst);

            //! Acync version
            void operator()(const GpuMat& disparity, const GpuMat& image, GpuMat& dst, Stream& stream);

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
    }

    //! Speckle filtering - filters small connected components on diparity image.
    //! It sets pixel (x,y) to newVal if it coresponds to small CC with size < maxSpeckleSize.
    //! Threshold for border between CC is diffThreshold;
    CV_EXPORTS void filterSpeckles( Mat& img, uchar newVal, int maxSpeckleSize, uchar diffThreshold, Mat& buf);

}
#include "opencv2/gpu/matrix_operations.hpp"

#endif /* __OPENCV_GPU_HPP__ */
