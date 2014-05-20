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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
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

#ifndef __OPENCV_OCL_HPP__
#define __OPENCV_OCL_HPP__

#include <memory>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/ml/ml.hpp"

namespace cv
{
    namespace ocl
    {
        enum DeviceType
        {
            CVCL_DEVICE_TYPE_DEFAULT     = (1 << 0),
            CVCL_DEVICE_TYPE_CPU         = (1 << 1),
            CVCL_DEVICE_TYPE_GPU         = (1 << 2),
            CVCL_DEVICE_TYPE_ACCELERATOR = (1 << 3),
            //CVCL_DEVICE_TYPE_CUSTOM      = (1 << 4)
            CVCL_DEVICE_TYPE_ALL         = 0xFFFFFFFF
        };

        enum DevMemRW
        {
            DEVICE_MEM_R_W = 0,
            DEVICE_MEM_R_ONLY,
            DEVICE_MEM_W_ONLY
        };

        enum DevMemType
        {
            DEVICE_MEM_DEFAULT = 0,
            DEVICE_MEM_AHP,         //alloc host pointer
            DEVICE_MEM_UHP,         //use host pointer
            DEVICE_MEM_CHP,         //copy host pointer
            DEVICE_MEM_PM           //persistent memory
        };

        // these classes contain OpenCL runtime information

        struct PlatformInfo;

        struct DeviceInfo
        {
            int _id; // reserved, don't use it

            DeviceType deviceType;
            std::string deviceProfile;
            std::string deviceVersion;
            std::string deviceName;
            std::string deviceVendor;
            int deviceVendorId;
            std::string deviceDriverVersion;
            std::string deviceExtensions;

            size_t maxWorkGroupSize;
            std::vector<size_t> maxWorkItemSizes;
            int maxComputeUnits;
            size_t localMemorySize;
            size_t maxMemAllocSize;

            int deviceVersionMajor;
            int deviceVersionMinor;

            bool haveDoubleSupport;
            bool isUnifiedMemory; // 1 means integrated GPU, otherwise this value is 0
            bool isIntelDevice;

            std::string compilationExtraOptions;

            const PlatformInfo* platform;

            DeviceInfo();
            ~DeviceInfo();
        };

        struct PlatformInfo
        {
            int _id; // reserved, don't use it

            std::string platformProfile;
            std::string platformVersion;
            std::string platformName;
            std::string platformVendor;
            std::string platformExtensons;

            int platformVersionMajor;
            int platformVersionMinor;

            std::vector<const DeviceInfo*> devices;

            PlatformInfo();
            ~PlatformInfo();
        };

        //////////////////////////////// Initialization & Info ////////////////////////
        typedef std::vector<const PlatformInfo*> PlatformsInfo;

        CV_EXPORTS int getOpenCLPlatforms(PlatformsInfo& platforms);

        typedef std::vector<const DeviceInfo*> DevicesInfo;

        CV_EXPORTS int getOpenCLDevices(DevicesInfo& devices, int deviceType = CVCL_DEVICE_TYPE_GPU,
                const PlatformInfo* platform = NULL);

        // set device you want to use
        CV_EXPORTS void setDevice(const DeviceInfo* info);

        // Initialize from OpenCL handles directly.
        // Argument types is (pointers): cl_platform_id*, cl_context*, cl_device_id*
        CV_EXPORTS void initializeContext(void* pClPlatform, void* pClContext, void* pClDevice);

        //////////////////////////////// Error handling ////////////////////////
        CV_EXPORTS void error(const char *error_string, const char *file, const int line, const char *func);

        enum FEATURE_TYPE
        {
            FEATURE_CL_DOUBLE = 1,
            FEATURE_CL_UNIFIED_MEM,
            FEATURE_CL_VER_1_2,
            FEATURE_CL_INTEL_DEVICE
        };

        // Represents OpenCL context, interface
        class CV_EXPORTS Context
        {
        protected:
            Context() { }
            ~Context() { }
        public:
            static Context* getContext();

            bool supportsFeature(FEATURE_TYPE featureType) const;
            const DeviceInfo& getDeviceInfo() const;

            const void* getOpenCLContextPtr() const;
            const void* getOpenCLCommandQueuePtr() const;
            const void* getOpenCLDeviceIDPtr() const;
        };

        inline const void *getClContextPtr()
        {
            return Context::getContext()->getOpenCLContextPtr();
        }

        inline const void *getClCommandQueuePtr()
        {
            return Context::getContext()->getOpenCLCommandQueuePtr();
        }

        CV_EXPORTS bool supportsFeature(FEATURE_TYPE featureType);

        CV_EXPORTS void finish();

        enum BINARY_CACHE_MODE
        {
            CACHE_NONE    = 0,        // do not cache OpenCL binary
            CACHE_DEBUG   = 0x1 << 0, // cache OpenCL binary when built in debug mode
            CACHE_RELEASE = 0x1 << 1, // default behavior, only cache when built in release mode
            CACHE_ALL     = CACHE_DEBUG | CACHE_RELEASE // cache opencl binary
        };
        //! Enable or disable OpenCL program binary caching onto local disk
        // After a program (*.cl files in opencl/ folder) is built at runtime, we allow the
        // compiled OpenCL program to be cached to the path automatically as "path/*.clb"
        // binary file, which will be reused when the OpenCV executable is started again.
        //
        // This feature is enabled by default.
        CV_EXPORTS void setBinaryDiskCache(int mode = CACHE_RELEASE, cv::String path = "./");

        //! set where binary cache to be saved to
        CV_EXPORTS void setBinaryPath(const char *path);

        struct ProgramSource
        {
            const char* name;
            const char* programStr;
            const char* programHash;

            // Cache in memory by name (should be unique). Caching on disk disabled.
            inline ProgramSource(const char* _name, const char* _programStr)
                : name(_name), programStr(_programStr), programHash(NULL)
            {
            }

            // Cache in memory by name (should be unique). Caching on disk uses programHash mark.
            inline ProgramSource(const char* _name, const char* _programStr, const char* _programHash)
                : name(_name), programStr(_programStr), programHash(_programHash)
            {
            }
        };

        //! Calls OpenCL kernel. Pass globalThreads = NULL, and cleanUp = true, to finally clean-up without executing.
        //! Deprecated, will be replaced
        CV_EXPORTS void openCLExecuteKernelInterop(Context *clCxt,
                const cv::ocl::ProgramSource& source, string kernelName,
                size_t globalThreads[3], size_t localThreads[3],
                std::vector< std::pair<size_t, const void *> > &args,
                int channels, int depth, const char *build_options);

        class CV_EXPORTS oclMatExpr;
        //////////////////////////////// oclMat ////////////////////////////////
        class CV_EXPORTS oclMat
        {
        public:
            //! default constructor
            oclMat();
            //! constructs oclMatrix of the specified size and type (_type is CV_8UC1, CV_64FC3, CV_32SC(12) etc.)
            oclMat(int rows, int cols, int type);
            oclMat(Size size, int type);
            //! constucts oclMatrix and fills it with the specified value _s.
            oclMat(int rows, int cols, int type, const Scalar &s);
            oclMat(Size size, int type, const Scalar &s);
            //! copy constructor
            oclMat(const oclMat &m);

            //! constructor for oclMatrix headers pointing to user-allocated data
            oclMat(int rows, int cols, int type, void *data, size_t step = Mat::AUTO_STEP);
            oclMat(Size size, int type, void *data, size_t step = Mat::AUTO_STEP);

            //! creates a matrix header for a part of the bigger matrix
            oclMat(const oclMat &m, const Range &rowRange, const Range &colRange);
            oclMat(const oclMat &m, const Rect &roi);

            //! builds oclMat from Mat. Perfom blocking upload to device.
            explicit oclMat (const Mat &m);

            //! destructor - calls release()
            ~oclMat();

            //! assignment operators
            oclMat &operator = (const oclMat &m);
            //! assignment operator. Perfom blocking upload to device.
            oclMat &operator = (const Mat &m);
            oclMat &operator = (const oclMatExpr& expr);

            //! pefroms blocking upload data to oclMat.
            void upload(const cv::Mat &m);


            //! downloads data from device to host memory. Blocking calls.
            operator Mat() const;
            void download(cv::Mat &m) const;

            //! convert to _InputArray
            operator _InputArray();

            //! convert to _OutputArray
            operator _OutputArray();

            //! returns a new oclMatrix header for the specified row
            oclMat row(int y) const;
            //! returns a new oclMatrix header for the specified column
            oclMat col(int x) const;
            //! ... for the specified row span
            oclMat rowRange(int startrow, int endrow) const;
            oclMat rowRange(const Range &r) const;
            //! ... for the specified column span
            oclMat colRange(int startcol, int endcol) const;
            oclMat colRange(const Range &r) const;

            //! returns deep copy of the oclMatrix, i.e. the data is copied
            oclMat clone() const;

            //! copies those oclMatrix elements to "m" that are marked with non-zero mask elements.
            // It calls m.create(this->size(), this->type()).
            // It supports any data type
            void copyTo( oclMat &m, const oclMat &mask = oclMat()) const;

            //! converts oclMatrix to another datatype with optional scalng. See cvConvertScale.
            void convertTo( oclMat &m, int rtype, double alpha = 1, double beta = 0 ) const;

            void assignTo( oclMat &m, int type = -1 ) const;

            //! sets every oclMatrix element to s
            oclMat& operator = (const Scalar &s);
            //! sets some of the oclMatrix elements to s, according to the mask
            oclMat& setTo(const Scalar &s, const oclMat &mask = oclMat());
            //! creates alternative oclMatrix header for the same data, with different
            // number of channels and/or different number of rows. see cvReshape.
            oclMat reshape(int cn, int rows = 0) const;

            //! allocates new oclMatrix data unless the oclMatrix already has specified size and type.
            // previous data is unreferenced if needed.
            void create(int rows, int cols, int type);
            void create(Size size, int type);

            //! allocates new oclMatrix with specified device memory type.
            void createEx(int rows, int cols, int type, DevMemRW rw_type, DevMemType mem_type);
            void createEx(Size size, int type, DevMemRW rw_type, DevMemType mem_type);

            //! decreases reference counter;
            // deallocate the data when reference counter reaches 0.
            void release();

            //! swaps with other smart pointer
            void swap(oclMat &mat);

            //! locates oclMatrix header within a parent oclMatrix. See below
            void locateROI( Size &wholeSize, Point &ofs ) const;
            //! moves/resizes the current oclMatrix ROI inside the parent oclMatrix.
            oclMat& adjustROI( int dtop, int dbottom, int dleft, int dright );
            //! extracts a rectangular sub-oclMatrix
            // (this is a generalized form of row, rowRange etc.)
            oclMat operator()( Range rowRange, Range colRange ) const;
            oclMat operator()( const Rect &roi ) const;

            oclMat& operator+=( const oclMat& m );
            oclMat& operator-=( const oclMat& m );
            oclMat& operator*=( const oclMat& m );
            oclMat& operator/=( const oclMat& m );

            //! returns true if the oclMatrix data is continuous
            // (i.e. when there are no gaps between successive rows).
            // similar to CV_IS_oclMat_CONT(cvoclMat->type)
            bool isContinuous() const;
            //! returns element size in bytes,
            // similar to CV_ELEM_SIZE(cvMat->type)
            size_t elemSize() const;
            //! returns the size of element channel in bytes.
            size_t elemSize1() const;
            //! returns element type, similar to CV_MAT_TYPE(cvMat->type)
            int type() const;
            //! returns element type, i.e. 8UC3 returns 8UC4 because in ocl
            //! 3 channels element actually use 4 channel space
            int ocltype() const;
            //! returns element type, similar to CV_MAT_DEPTH(cvMat->type)
            int depth() const;
            //! returns element type, similar to CV_MAT_CN(cvMat->type)
            int channels() const;
            //! returns element type, return 4 for 3 channels element,
            //!becuase 3 channels element actually use 4 channel space
            int oclchannels() const;
            //! returns step/elemSize1()
            size_t step1() const;
            //! returns oclMatrix size:
            // width == number of columns, height == number of rows
            Size size() const;
            //! returns true if oclMatrix data is NULL
            bool empty() const;

            //! matrix transposition
            oclMat t() const;

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
            //! pointer to the data(OCL memory object)
            uchar *data;

            //! pointer to the reference counter;
            // when oclMatrix points to user-allocated data, the pointer is NULL
            int *refcount;

            //! helper fields used in locateROI and adjustROI
            //datastart and dataend are not used in current version
            uchar *datastart;
            uchar *dataend;

            //! OpenCL context associated with the oclMat object.
            Context *clCxt; // TODO clCtx
            //add offset for handle ROI, calculated in byte
            int offset;
            //add wholerows and wholecols for the whole matrix, datastart and dataend are no longer used
            int wholerows;
            int wholecols;
        };

        // convert InputArray/OutputArray to oclMat references
        CV_EXPORTS oclMat& getOclMatRef(InputArray src);
        CV_EXPORTS oclMat& getOclMatRef(OutputArray src);

        ///////////////////// mat split and merge /////////////////////////////////
        //! Compose a multi-channel array from several single-channel arrays
        // Support all types
        CV_EXPORTS void merge(const oclMat *src, size_t n, oclMat &dst);
        CV_EXPORTS void merge(const vector<oclMat> &src, oclMat &dst);

        //! Divides multi-channel array into several single-channel arrays
        // Support all types
        CV_EXPORTS void split(const oclMat &src, oclMat *dst);
        CV_EXPORTS void split(const oclMat &src, vector<oclMat> &dst);

        ////////////////////////////// Arithmetics ///////////////////////////////////

        //! adds one matrix to another with scale (dst = src1 * alpha + src2 * beta + gama)
        // supports all data types
        CV_EXPORTS void addWeighted(const oclMat &src1, double  alpha, const oclMat &src2, double beta, double gama, oclMat &dst);

        //! adds one matrix to another (dst = src1 + src2)
        // supports all data types
        CV_EXPORTS void add(const oclMat &src1, const oclMat &src2, oclMat &dst, const oclMat &mask = oclMat());
        //! adds scalar to a matrix (dst = src1 + s)
        // supports all data types
        CV_EXPORTS void add(const oclMat &src1, const Scalar &s, oclMat &dst, const oclMat &mask = oclMat());

        //! subtracts one matrix from another (dst = src1 - src2)
        // supports all data types
        CV_EXPORTS void subtract(const oclMat &src1, const oclMat &src2, oclMat &dst, const oclMat &mask = oclMat());
        //! subtracts scalar from a matrix (dst = src1 - s)
        // supports all data types
        CV_EXPORTS void subtract(const oclMat &src1, const Scalar &s, oclMat &dst, const oclMat &mask = oclMat());

        //! computes element-wise product of the two arrays (dst = src1 * scale * src2)
        // supports all data types
        CV_EXPORTS void multiply(const oclMat &src1, const oclMat &src2, oclMat &dst, double scale = 1);
        //! multiplies matrix to a number (dst = scalar * src)
        // supports all data types
        CV_EXPORTS void multiply(double scalar, const oclMat &src, oclMat &dst);

        //! computes element-wise quotient of the two arrays (dst = src1 * scale / src2)
        // supports all data types
        CV_EXPORTS void divide(const oclMat &src1, const oclMat &src2, oclMat &dst, double scale = 1);
        //! computes element-wise quotient of the two arrays (dst = scale / src)
        // supports all data types
        CV_EXPORTS void divide(double scale, const oclMat &src1, oclMat &dst);

        //! computes element-wise minimum of the two arrays (dst = min(src1, src2))
        // supports all data types
        CV_EXPORTS void min(const oclMat &src1, const oclMat &src2, oclMat &dst);

        //! computes element-wise maximum of the two arrays (dst = max(src1, src2))
        // supports all data types
        CV_EXPORTS void max(const oclMat &src1, const oclMat &src2, oclMat &dst);

        //! compares elements of two arrays (dst = src1 <cmpop> src2)
        // supports all data types
        CV_EXPORTS void compare(const oclMat &src1, const oclMat &src2, oclMat &dst, int cmpop);

        //! transposes the matrix
        // supports all data types
        CV_EXPORTS void transpose(const oclMat &src, oclMat &dst);

        //! computes element-wise absolute values of an array (dst = abs(src))
        // supports all data types
        CV_EXPORTS void abs(const oclMat &src, oclMat &dst);

        //! computes element-wise absolute difference of two arrays (dst = abs(src1 - src2))
        // supports all data types
        CV_EXPORTS void absdiff(const oclMat &src1, const oclMat &src2, oclMat &dst);
        //! computes element-wise absolute difference of array and scalar (dst = abs(src1 - s))
        // supports all data types
        CV_EXPORTS void absdiff(const oclMat &src1, const Scalar &s, oclMat &dst);

        //! computes mean value and standard deviation of all or selected array elements
        // supports all data types
        CV_EXPORTS void meanStdDev(const oclMat &mtx, Scalar &mean, Scalar &stddev);

        //! computes norm of array
        // supports NORM_INF, NORM_L1, NORM_L2
        // supports all data types
        CV_EXPORTS double norm(const oclMat &src1, int normType = NORM_L2);

        //! computes norm of the difference between two arrays
        // supports NORM_INF, NORM_L1, NORM_L2
        // supports all data types
        CV_EXPORTS double norm(const oclMat &src1, const oclMat &src2, int normType = NORM_L2);

        //! reverses the order of the rows, columns or both in a matrix
        // supports all types
        CV_EXPORTS void flip(const oclMat &src, oclMat &dst, int flipCode);

        //! computes sum of array elements
        // support all types
        CV_EXPORTS Scalar sum(const oclMat &m);
        CV_EXPORTS Scalar absSum(const oclMat &m);
        CV_EXPORTS Scalar sqrSum(const oclMat &m);

        //! finds global minimum and maximum array elements and returns their values
        // support all C1 types
        CV_EXPORTS void minMax(const oclMat &src, double *minVal, double *maxVal = 0, const oclMat &mask = oclMat());

        //! finds global minimum and maximum array elements and returns their values with locations
        // support all C1 types
        CV_EXPORTS void minMaxLoc(const oclMat &src, double *minVal, double *maxVal = 0, Point *minLoc = 0, Point *maxLoc = 0,
                                  const oclMat &mask = oclMat());

        //! counts non-zero array elements
        // support all types
        CV_EXPORTS int countNonZero(const oclMat &src);

        //! transforms 8-bit unsigned integers using lookup table: dst(i)=lut(src(i))
        // destination array will have the depth type as lut and the same channels number as source
        //It supports 8UC1 8UC4 only
        CV_EXPORTS void LUT(const oclMat &src, const oclMat &lut, oclMat &dst);

        //! only 8UC1 and 256 bins is supported now
        CV_EXPORTS void calcHist(const oclMat &mat_src, oclMat &mat_hist);
        //! only 8UC1 and 256 bins is supported now
        CV_EXPORTS void equalizeHist(const oclMat &mat_src, oclMat &mat_dst);

        //! only 8UC1 is supported now
        CV_EXPORTS Ptr<cv::CLAHE> createCLAHE(double clipLimit = 40.0, Size tileGridSize = Size(8, 8));

        //! bilateralFilter
        // supports 8UC1 8UC4
        CV_EXPORTS void bilateralFilter(const oclMat& src, oclMat& dst, int d, double sigmaColor, double sigmaSpace, int borderType=BORDER_DEFAULT);

        //! Applies an adaptive bilateral filter to the input image
        //  Unlike the usual bilateral filter that uses fixed value for sigmaColor,
        //  the adaptive version calculates the local variance in he ksize neighborhood
        //  and use this as sigmaColor, for the value filtering. However, the local standard deviation is
        //  clamped to the maxSigmaColor.
        //  supports 8UC1, 8UC3
        CV_EXPORTS void adaptiveBilateralFilter(const oclMat& src, oclMat& dst, Size ksize, double sigmaSpace, double maxSigmaColor=20.0, Point anchor = Point(-1, -1), int borderType=BORDER_DEFAULT);

        //! computes exponent of each matrix element (dst = e**src)
        // supports only CV_32FC1, CV_64FC1 type
        CV_EXPORTS void exp(const oclMat &src, oclMat &dst);

        //! computes natural logarithm of absolute value of each matrix element: dst = log(abs(src))
        // supports only CV_32FC1, CV_64FC1 type
        CV_EXPORTS void log(const oclMat &src, oclMat &dst);

        //! computes magnitude of each (x(i), y(i)) vector
        // supports only CV_32F, CV_64F type
        CV_EXPORTS void magnitude(const oclMat &x, const oclMat &y, oclMat &magnitude);

        //! computes angle (angle(i)) of each (x(i), y(i)) vector
        // supports only CV_32F, CV_64F type
        CV_EXPORTS void phase(const oclMat &x, const oclMat &y, oclMat &angle, bool angleInDegrees = false);

        //! the function raises every element of tne input array to p
        // support only CV_32F, CV_64F type
        CV_EXPORTS void pow(const oclMat &x, double p, oclMat &y);

        //! converts Cartesian coordinates to polar
        // supports only CV_32F CV_64F type
        CV_EXPORTS void cartToPolar(const oclMat &x, const oclMat &y, oclMat &magnitude, oclMat &angle, bool angleInDegrees = false);

        //! converts polar coordinates to Cartesian
        // supports only CV_32F CV_64F type
        CV_EXPORTS void polarToCart(const oclMat &magnitude, const oclMat &angle, oclMat &x, oclMat &y, bool angleInDegrees = false);

        //! perfroms per-elements bit-wise inversion
        // supports all types
        CV_EXPORTS void bitwise_not(const oclMat &src, oclMat &dst);

        //! calculates per-element bit-wise disjunction of two arrays
        // supports all types
        CV_EXPORTS void bitwise_or(const oclMat &src1, const oclMat &src2, oclMat &dst, const oclMat &mask = oclMat());
        CV_EXPORTS void bitwise_or(const oclMat &src1, const Scalar &s, oclMat &dst, const oclMat &mask = oclMat());

        //! calculates per-element bit-wise conjunction of two arrays
        // supports all types
        CV_EXPORTS void bitwise_and(const oclMat &src1, const oclMat &src2, oclMat &dst, const oclMat &mask = oclMat());
        CV_EXPORTS void bitwise_and(const oclMat &src1, const Scalar &s, oclMat &dst, const oclMat &mask = oclMat());

        //! calculates per-element bit-wise "exclusive or" operation
        // supports all types
        CV_EXPORTS void bitwise_xor(const oclMat &src1, const oclMat &src2, oclMat &dst, const oclMat &mask = oclMat());
        CV_EXPORTS void bitwise_xor(const oclMat &src1, const Scalar &s, oclMat &dst, const oclMat &mask = oclMat());

        //! Logical operators
        CV_EXPORTS oclMat operator ~ (const oclMat &);
        CV_EXPORTS oclMat operator | (const oclMat &, const oclMat &);
        CV_EXPORTS oclMat operator & (const oclMat &, const oclMat &);
        CV_EXPORTS oclMat operator ^ (const oclMat &, const oclMat &);


        //! Mathematics operators
        CV_EXPORTS oclMatExpr operator + (const oclMat &src1, const oclMat &src2);
        CV_EXPORTS oclMatExpr operator - (const oclMat &src1, const oclMat &src2);
        CV_EXPORTS oclMatExpr operator * (const oclMat &src1, const oclMat &src2);
        CV_EXPORTS oclMatExpr operator / (const oclMat &src1, const oclMat &src2);

        //! computes convolution of two images
        // support only CV_32FC1 type
        CV_EXPORTS void convolve(const oclMat &image, const oclMat &temp1, oclMat &result);

        CV_EXPORTS void cvtColor(const oclMat &src, oclMat &dst, int code, int dcn = 0);

        //! initializes a scaled identity matrix
        CV_EXPORTS void setIdentity(oclMat& src, const Scalar & val = Scalar(1));

        //! fills the output array with repeated copies of the input array
        CV_EXPORTS void repeat(const oclMat & src, int ny, int nx, oclMat & dst);

        //////////////////////////////// Filter Engine ////////////////////////////////

        /*!
          The Base Class for 1D or Row-wise Filters

          This is the base class for linear or non-linear filters that process 1D data.
          In particular, such filters are used for the "horizontal" filtering parts in separable filters.
          */
        class CV_EXPORTS BaseRowFilter_GPU
        {
        public:
            BaseRowFilter_GPU(int ksize_, int anchor_, int bordertype_) : ksize(ksize_), anchor(anchor_), bordertype(bordertype_) {}
            virtual ~BaseRowFilter_GPU() {}
            virtual void operator()(const oclMat &src, oclMat &dst) = 0;
            int ksize, anchor, bordertype;
        };

        /*!
          The Base Class for Column-wise Filters

          This is the base class for linear or non-linear filters that process columns of 2D arrays.
          Such filters are used for the "vertical" filtering parts in separable filters.
          */
        class CV_EXPORTS BaseColumnFilter_GPU
        {
        public:
            BaseColumnFilter_GPU(int ksize_, int anchor_, int bordertype_) : ksize(ksize_), anchor(anchor_), bordertype(bordertype_) {}
            virtual ~BaseColumnFilter_GPU() {}
            virtual void operator()(const oclMat &src, oclMat &dst) = 0;
            int ksize, anchor, bordertype;
        };

        /*!
          The Base Class for Non-Separable 2D Filters.

          This is the base class for linear or non-linear 2D filters.
          */
        class CV_EXPORTS BaseFilter_GPU
        {
        public:
            BaseFilter_GPU(const Size &ksize_, const Point &anchor_, const int &borderType_)
                : ksize(ksize_), anchor(anchor_), borderType(borderType_) {}
            virtual ~BaseFilter_GPU() {}
            virtual void operator()(const oclMat &src, oclMat &dst) = 0;
            Size ksize;
            Point anchor;
            int borderType;
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

            virtual void apply(const oclMat &src, oclMat &dst, Rect roi = Rect(0, 0, -1, -1)) = 0;
        };

        //! returns the non-separable filter engine with the specified filter
        CV_EXPORTS Ptr<FilterEngine_GPU> createFilter2D_GPU(const Ptr<BaseFilter_GPU> filter2D);

        //! returns the primitive row filter with the specified kernel
        CV_EXPORTS Ptr<BaseRowFilter_GPU> getLinearRowFilter_GPU(int srcType, int bufType, const Mat &rowKernel,
                int anchor = -1, int bordertype = BORDER_DEFAULT);

        //! returns the primitive column filter with the specified kernel
        CV_EXPORTS Ptr<BaseColumnFilter_GPU> getLinearColumnFilter_GPU(int bufType, int dstType, const Mat &columnKernel,
                int anchor = -1, int bordertype = BORDER_DEFAULT, double delta = 0.0);

        //! returns the separable linear filter engine
        CV_EXPORTS Ptr<FilterEngine_GPU> createSeparableLinearFilter_GPU(int srcType, int dstType, const Mat &rowKernel,
                const Mat &columnKernel, const Point &anchor = Point(-1, -1), double delta = 0.0, int bordertype = BORDER_DEFAULT, Size imgSize = Size(-1,-1));

        //! returns the separable filter engine with the specified filters
        CV_EXPORTS Ptr<FilterEngine_GPU> createSeparableFilter_GPU(const Ptr<BaseRowFilter_GPU> &rowFilter,
                const Ptr<BaseColumnFilter_GPU> &columnFilter);

        //! returns the Gaussian filter engine
        CV_EXPORTS Ptr<FilterEngine_GPU> createGaussianFilter_GPU(int type, Size ksize, double sigma1, double sigma2 = 0, int bordertype = BORDER_DEFAULT, Size imgSize = Size(-1,-1));

        //! returns filter engine for the generalized Sobel operator
        CV_EXPORTS Ptr<FilterEngine_GPU> createDerivFilter_GPU( int srcType, int dstType, int dx, int dy, int ksize, int borderType = BORDER_DEFAULT, Size imgSize = Size(-1,-1) );

        //! applies Laplacian operator to the image
        // supports only ksize = 1 and ksize = 3
        CV_EXPORTS void Laplacian(const oclMat &src, oclMat &dst, int ddepth, int ksize = 1, double scale = 1,
                double delta=0, int borderType=BORDER_DEFAULT);

        //! returns 2D box filter
        // dst type must be the same as source type
        CV_EXPORTS Ptr<BaseFilter_GPU> getBoxFilter_GPU(int srcType, int dstType,
                const Size &ksize, Point anchor = Point(-1, -1), int borderType = BORDER_DEFAULT);

        //! returns box filter engine
        CV_EXPORTS Ptr<FilterEngine_GPU> createBoxFilter_GPU(int srcType, int dstType, const Size &ksize,
                const Point &anchor = Point(-1, -1), int borderType = BORDER_DEFAULT);

        //! returns 2D filter with the specified kernel
        // supports: dst type must be the same as source type
        CV_EXPORTS Ptr<BaseFilter_GPU> getLinearFilter_GPU(int srcType, int dstType, const Mat &kernel, const Size &ksize,
                const Point &anchor = Point(-1, -1), int borderType = BORDER_DEFAULT);

        //! returns the non-separable linear filter engine
        // supports: dst type must be the same as source type
        CV_EXPORTS Ptr<FilterEngine_GPU> createLinearFilter_GPU(int srcType, int dstType, const Mat &kernel,
                const Point &anchor = Point(-1, -1), int borderType = BORDER_DEFAULT);

        //! smooths the image using the normalized box filter
        CV_EXPORTS void boxFilter(const oclMat &src, oclMat &dst, int ddepth, Size ksize,
                                  Point anchor = Point(-1, -1), int borderType = BORDER_DEFAULT);

        //! returns 2D morphological filter
        //! only MORPH_ERODE and MORPH_DILATE are supported
        // supports CV_8UC1, CV_8UC4, CV_32FC1 and CV_32FC4 types
        // kernel must have CV_8UC1 type, one rows and cols == ksize.width * ksize.height
        CV_EXPORTS Ptr<BaseFilter_GPU> getMorphologyFilter_GPU(int op, int type, const Mat &kernel, const Size &ksize,
                Point anchor = Point(-1, -1));

        //! returns morphological filter engine. Only MORPH_ERODE and MORPH_DILATE are supported.
        CV_EXPORTS Ptr<FilterEngine_GPU> createMorphologyFilter_GPU(int op, int type, const Mat &kernel,
                const Point &anchor = Point(-1, -1), int iterations = 1);

        //! a synonym for normalized box filter
        static inline void blur(const oclMat &src, oclMat &dst, Size ksize, Point anchor = Point(-1, -1),
                                int borderType = BORDER_CONSTANT)
        {
            boxFilter(src, dst, -1, ksize, anchor, borderType);
        }

        //! applies non-separable 2D linear filter to the image
        CV_EXPORTS void filter2D(const oclMat &src, oclMat &dst, int ddepth, const Mat &kernel,
                                 Point anchor = Point(-1, -1), double delta = 0.0, int borderType = BORDER_DEFAULT);

        //! applies separable 2D linear filter to the image
        CV_EXPORTS void sepFilter2D(const oclMat &src, oclMat &dst, int ddepth, const Mat &kernelX, const Mat &kernelY,
                                    Point anchor = Point(-1, -1), double delta = 0.0, int bordertype = BORDER_DEFAULT);

        //! applies generalized Sobel operator to the image
        // dst.type must equalize src.type
        // supports data type: CV_8UC1, CV_8UC4, CV_32FC1 and CV_32FC4
        // supports border type: BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT,BORDER_REFLECT_101
        CV_EXPORTS void Sobel(const oclMat &src, oclMat &dst, int ddepth, int dx, int dy, int ksize = 3, double scale = 1, double delta = 0.0, int bordertype = BORDER_DEFAULT);

        //! applies the vertical or horizontal Scharr operator to the image
        // dst.type must equalize src.type
        // supports data type: CV_8UC1, CV_8UC4, CV_32FC1 and CV_32FC4
        // supports border type: BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT,BORDER_REFLECT_101
        CV_EXPORTS void Scharr(const oclMat &src, oclMat &dst, int ddepth, int dx, int dy, double scale = 1, double delta = 0.0, int bordertype = BORDER_DEFAULT);

        //! smooths the image using Gaussian filter.
        // dst.type must equalize src.type
        // supports data type: CV_8UC1, CV_8UC4, CV_32FC1 and CV_32FC4
        // supports border type: BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT,BORDER_REFLECT_101
        CV_EXPORTS void GaussianBlur(const oclMat &src, oclMat &dst, Size ksize, double sigma1, double sigma2 = 0, int bordertype = BORDER_DEFAULT);

        //! erodes the image (applies the local minimum operator)
        // supports data type: CV_8UC1, CV_8UC4, CV_32FC1 and CV_32FC4
        CV_EXPORTS void erode( const oclMat &src, oclMat &dst, const Mat &kernel, Point anchor = Point(-1, -1), int iterations = 1,

                               int borderType = BORDER_CONSTANT, const Scalar &borderValue = morphologyDefaultBorderValue());


        //! dilates the image (applies the local maximum operator)
        // supports data type: CV_8UC1, CV_8UC4, CV_32FC1 and CV_32FC4
        CV_EXPORTS void dilate( const oclMat &src, oclMat &dst, const Mat &kernel, Point anchor = Point(-1, -1), int iterations = 1,

                                int borderType = BORDER_CONSTANT, const Scalar &borderValue = morphologyDefaultBorderValue());


        //! applies an advanced morphological operation to the image
        CV_EXPORTS void morphologyEx( const oclMat &src, oclMat &dst, int op, const Mat &kernel, Point anchor = Point(-1, -1), int iterations = 1,

                                      int borderType = BORDER_CONSTANT, const Scalar &borderValue = morphologyDefaultBorderValue());


        ////////////////////////////// Image processing //////////////////////////////
        //! Does mean shift filtering on GPU.
        CV_EXPORTS void meanShiftFiltering(const oclMat &src, oclMat &dst, int sp, int sr,
                                           TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1));

        //! Does mean shift procedure on GPU.
        CV_EXPORTS void meanShiftProc(const oclMat &src, oclMat &dstr, oclMat &dstsp, int sp, int sr,
                                      TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1));

        //! Does mean shift segmentation with elimiation of small regions.
        CV_EXPORTS void meanShiftSegmentation(const oclMat &src, Mat &dst, int sp, int sr, int minsize,
                                              TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1));

        //! applies fixed threshold to the image.
        // supports CV_8UC1 and CV_32FC1 data type
        // supports threshold type: THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV
        CV_EXPORTS double threshold(const oclMat &src, oclMat &dst, double thresh, double maxVal, int type = THRESH_TRUNC);

        //! resizes the image
        // Supports INTER_NEAREST, INTER_LINEAR
        // supports CV_8UC1, CV_8UC4, CV_32FC1 and CV_32FC4 types
        CV_EXPORTS void resize(const oclMat &src, oclMat &dst, Size dsize, double fx = 0, double fy = 0, int interpolation = INTER_LINEAR);

        //! Applies a generic geometrical transformation to an image.

        // Supports INTER_NEAREST, INTER_LINEAR.
        // Map1 supports CV_16SC2, CV_32FC2  types.
        // Src supports CV_8UC1, CV_8UC2, CV_8UC4.
        CV_EXPORTS void remap(const oclMat &src, oclMat &dst, oclMat &map1, oclMat &map2, int interpolation, int bordertype, const Scalar &value = Scalar());

        //! copies 2D array to a larger destination array and pads borders with user-specifiable constant
        // supports CV_8UC1, CV_8UC4, CV_32SC1 types
        CV_EXPORTS void copyMakeBorder(const oclMat &src, oclMat &dst, int top, int bottom, int left, int right, int boardtype, const Scalar &value = Scalar());

        //! Smoothes image using median filter
        // The source 1- or 4-channel image. m should be 3 or 5, the image depth should be CV_8U or CV_32F.
        CV_EXPORTS void medianFilter(const oclMat &src, oclMat &dst, int m);

        //! warps the image using affine transformation
        // Supports INTER_NEAREST, INTER_LINEAR, INTER_CUBIC
        // supports CV_8UC1, CV_8UC4, CV_32FC1 and CV_32FC4 types
        CV_EXPORTS void warpAffine(const oclMat &src, oclMat &dst, const Mat &M, Size dsize, int flags = INTER_LINEAR);

        //! warps the image using perspective transformation
        // Supports INTER_NEAREST, INTER_LINEAR, INTER_CUBIC
        // supports CV_8UC1, CV_8UC4, CV_32FC1 and CV_32FC4 types
        CV_EXPORTS void warpPerspective(const oclMat &src, oclMat &dst, const Mat &M, Size dsize, int flags = INTER_LINEAR);

        //! computes the integral image and integral for the squared image
        // sum will have CV_32S type, sqsum - CV32F type
        // supports only CV_8UC1 source type
        CV_EXPORTS void integral(const oclMat &src, oclMat &sum, oclMat &sqsum);
        CV_EXPORTS void integral(const oclMat &src, oclMat &sum);
        CV_EXPORTS void cornerHarris(const oclMat &src, oclMat &dst, int blockSize, int ksize, double k, int bordertype = cv::BORDER_DEFAULT);
        CV_EXPORTS void cornerHarris_dxdy(const oclMat &src, oclMat &dst, oclMat &Dx, oclMat &Dy,
            int blockSize, int ksize, double k, int bordertype = cv::BORDER_DEFAULT);
        CV_EXPORTS void cornerMinEigenVal(const oclMat &src, oclMat &dst, int blockSize, int ksize, int bordertype = cv::BORDER_DEFAULT);
        CV_EXPORTS void cornerMinEigenVal_dxdy(const oclMat &src, oclMat &dst, oclMat &Dx, oclMat &Dy,
            int blockSize, int ksize, int bordertype = cv::BORDER_DEFAULT);
        /////////////////////////////////// ML ///////////////////////////////////////////

        //! Compute closest centers for each lines in source and lable it after center's index
        // supports CV_32FC1/CV_32FC2/CV_32FC4 data type
        // supports NORM_L1 and NORM_L2 distType
        // if indices is provided, only the indexed rows will be calculated and their results are in the same
        // order of indices
        CV_EXPORTS void distanceToCenters(const oclMat &src, const oclMat &centers, Mat &dists, Mat &labels, int distType = NORM_L2SQR);

        //!Does k-means procedure on GPU
        // supports CV_32FC1/CV_32FC2/CV_32FC4 data type
        CV_EXPORTS double kmeans(const oclMat &src, int K, oclMat &bestLabels,
                                     TermCriteria criteria, int attemps, int flags, oclMat &centers);


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////CascadeClassifier//////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        class CV_EXPORTS_W OclCascadeClassifier : public  cv::CascadeClassifier
        {
        public:
            OclCascadeClassifier() {};
            ~OclCascadeClassifier() {};

            CvSeq* oclHaarDetectObjects(oclMat &gimg, CvMemStorage *storage, double scaleFactor,
                                        int minNeighbors, int flags, CvSize minSize = cvSize(0, 0), CvSize maxSize = cvSize(0, 0));
            void detectMultiScale(oclMat &image, CV_OUT std::vector<cv::Rect>& faces,
                double scaleFactor = 1.1, int minNeighbors = 3, int flags = 0,
                Size minSize = Size(), Size maxSize = Size());
        };

        class CV_EXPORTS OclCascadeClassifierBuf : public  cv::CascadeClassifier
        {
        public:
            OclCascadeClassifierBuf() :
                m_flags(0), initialized(false), m_scaleFactor(0), buffers(NULL) {}

            ~OclCascadeClassifierBuf() { release(); }

            void detectMultiScale(oclMat &image, CV_OUT std::vector<cv::Rect>& faces,
                                  double scaleFactor = 1.1, int minNeighbors = 3, int flags = 0,
                                  Size minSize = Size(), Size maxSize = Size());
            void release();

        private:
            void Init(const int rows, const int cols, double scaleFactor, int flags,
                      const int outputsz, const size_t localThreads[],
                      CvSize minSize, CvSize maxSize);
            void CreateBaseBufs(const int datasize, const int totalclassifier, const int flags, const int outputsz);
            void CreateFactorRelatedBufs(const int rows, const int cols, const int flags,
                                         const double scaleFactor, const size_t localThreads[],
                                         CvSize minSize, CvSize maxSize);
            void GenResult(CV_OUT std::vector<cv::Rect>& faces, const std::vector<cv::Rect> &rectList, const std::vector<int> &rweights);

            int m_rows;
            int m_cols;
            int m_flags;
            int m_loopcount;
            int m_nodenum;
            bool findBiggestObject;
            bool initialized;
            double m_scaleFactor;
            Size m_minSize;
            Size m_maxSize;
            vector<CvSize> sizev;
            vector<float> scalev;
            oclMat gimg1, gsum, gsqsum;
            void * buffers;
        };


        /////////////////////////////// Pyramid /////////////////////////////////////
        CV_EXPORTS void pyrDown(const oclMat &src, oclMat &dst);

        //! upsamples the source image and then smoothes it
        CV_EXPORTS void pyrUp(const oclMat &src, oclMat &dst);

        //! performs linear blending of two images
        //! to avoid accuracy errors sum of weigths shouldn't be very close to zero
        // supports only CV_8UC1 source type
        CV_EXPORTS void blendLinear(const oclMat &img1, const oclMat &img2, const oclMat &weights1, const oclMat &weights2, oclMat &result);

        //! computes vertical sum, supports only CV_32FC1 images
        CV_EXPORTS void columnSum(const oclMat &src, oclMat &sum);

        ///////////////////////////////////////// match_template /////////////////////////////////////////////////////////////
        struct CV_EXPORTS MatchTemplateBuf
        {
            Size user_block_size;
            oclMat imagef, templf;
            std::vector<oclMat> images;
            std::vector<oclMat> image_sums;
            std::vector<oclMat> image_sqsums;
        };

        //! computes the proximity map for the raster template and the image where the template is searched for
        // Supports TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_CCOEFF, TM_CCOEFF_NORMED for type 8UC1 and 8UC4
        // Supports TM_SQDIFF, TM_CCORR for type 32FC1 and 32FC4
        CV_EXPORTS void matchTemplate(const oclMat &image, const oclMat &templ, oclMat &result, int method);

        //! computes the proximity map for the raster template and the image where the template is searched for
        // Supports TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_CCOEFF, TM_CCOEFF_NORMED for type 8UC1 and 8UC4
        // Supports TM_SQDIFF, TM_CCORR for type 32FC1 and 32FC4
        CV_EXPORTS void matchTemplate(const oclMat &image, const oclMat &templ, oclMat &result, int method, MatchTemplateBuf &buf);

        ///////////////////////////////////////////// Canny /////////////////////////////////////////////
        struct CV_EXPORTS CannyBuf;
        //! compute edges of the input image using Canny operator
        // Support CV_8UC1 only
        CV_EXPORTS void Canny(const oclMat &image, oclMat &edges, double low_thresh, double high_thresh, int apperture_size = 3, bool L2gradient = false);
        CV_EXPORTS void Canny(const oclMat &image, CannyBuf &buf, oclMat &edges, double low_thresh, double high_thresh, int apperture_size = 3, bool L2gradient = false);
        CV_EXPORTS void Canny(const oclMat &dx, const oclMat &dy, oclMat &edges, double low_thresh, double high_thresh, bool L2gradient = false);
        CV_EXPORTS void Canny(const oclMat &dx, const oclMat &dy, CannyBuf &buf, oclMat &edges, double low_thresh, double high_thresh, bool L2gradient = false);

        struct CV_EXPORTS CannyBuf
        {
            CannyBuf() : counter(1, 1, CV_32S) { }
            ~CannyBuf()
            {
                release();
            }
            explicit CannyBuf(const Size &image_size, int apperture_size = 3) : counter(1, 1, CV_32S)
            {
                create(image_size, apperture_size);
            }
            CannyBuf(const oclMat &dx_, const oclMat &dy_);

            void create(const Size &image_size, int apperture_size = 3);
            void release();
            oclMat dx, dy;
            oclMat dx_buf, dy_buf;
            oclMat edgeBuf;
            oclMat trackBuf1, trackBuf2;
            oclMat counter;
            Ptr<FilterEngine_GPU> filterDX, filterDY;
        };

        ///////////////////////////////////////// clAmdFft related /////////////////////////////////////////
        //! Performs a forward or inverse discrete Fourier transform (1D or 2D) of floating point matrix.
        //! Param dft_size is the size of DFT transform.
        //!
        //! For complex-to-real transform it is assumed that the source matrix is packed in CLFFT's format.
        // support src type of CV32FC1, CV32FC2
        // support flags: DFT_INVERSE, DFT_REAL_OUTPUT, DFT_COMPLEX_OUTPUT, DFT_ROWS
        // dft_size is the size of original input, which is used for transformation from complex to real.
        // dft_size must be powers of 2, 3 and 5
        // real to complex dft requires at least v1.8 clAmdFft
        // real to complex dft output is not the same with cpu version
        // real to complex and complex to real does not support DFT_ROWS
        CV_EXPORTS void dft(const oclMat &src, oclMat &dst, Size dft_size = Size(), int flags = 0);

        //! implements generalized matrix product algorithm GEMM from BLAS
        // The functionality requires clAmdBlas library
        // only support type CV_32FC1
        // flag GEMM_3_T is not supported
        CV_EXPORTS void gemm(const oclMat &src1, const oclMat &src2, double alpha,
                             const oclMat &src3, double beta, oclMat &dst, int flags = 0);

        //////////////// HOG (Histogram-of-Oriented-Gradients) Descriptor and Object Detector //////////////
        struct CV_EXPORTS HOGDescriptor
        {
            enum { DEFAULT_WIN_SIGMA = -1 };
            enum { DEFAULT_NLEVELS = 64 };
            enum { DESCR_FORMAT_ROW_BY_ROW, DESCR_FORMAT_COL_BY_COL };
            HOGDescriptor(Size win_size = Size(64, 128), Size block_size = Size(16, 16),
                          Size block_stride = Size(8, 8), Size cell_size = Size(8, 8),
                          int nbins = 9, double win_sigma = DEFAULT_WIN_SIGMA,
                          double threshold_L2hys = 0.2, bool gamma_correction = true,
                          int nlevels = DEFAULT_NLEVELS);

            size_t getDescriptorSize() const;
            size_t getBlockHistogramSize() const;
            void setSVMDetector(const vector<float> &detector);
            static vector<float> getDefaultPeopleDetector();
            static vector<float> getPeopleDetector48x96();
            static vector<float> getPeopleDetector64x128();
            void detect(const oclMat &img, vector<Point> &found_locations,
                        double hit_threshold = 0, Size win_stride = Size(),
                        Size padding = Size());
            void detectMultiScale(const oclMat &img, vector<Rect> &found_locations,
                                  double hit_threshold = 0, Size win_stride = Size(),
                                  Size padding = Size(), double scale0 = 1.05,
                                  int group_threshold = 2);
            void getDescriptors(const oclMat &img, Size win_stride,
                                oclMat &descriptors,
                                int descr_format = DESCR_FORMAT_COL_BY_COL);
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
            // initialize buffers; only need to do once in case of multiscale detection
            void init_buffer(const oclMat &img, Size win_stride);
            void computeBlockHistograms(const oclMat &img);
            void computeGradient(const oclMat &img, oclMat &grad, oclMat &qangle);
            double getWinSigma() const;
            bool checkDetectorSize() const;

            static int numPartsWithin(int size, int part_size, int stride);
            static Size numPartsWithin(Size size, Size part_size, Size stride);

            // Coefficients of the separating plane
            float free_coef;
            oclMat detector;
            // Results of the last classification step
            oclMat labels;
            Mat labels_host;
            // Results of the last histogram evaluation step
            oclMat block_hists;
            // Gradients conputation results
            oclMat grad, qangle;
            // scaled image
            oclMat image_scale;
            // effect size of input image (might be different from original size after scaling)
            Size effect_size;
        };


        ////////////////////////feature2d_ocl/////////////////
        /****************************************************************************************\
        *                                      Distance                                          *
        \****************************************************************************************/
        template<typename T>
        struct CV_EXPORTS Accumulator
        {
            typedef T Type;
        };
        template<> struct Accumulator<unsigned char>
        {
            typedef float Type;
        };
        template<> struct Accumulator<unsigned short>
        {
            typedef float Type;
        };
        template<> struct Accumulator<char>
        {
            typedef float Type;
        };
        template<> struct Accumulator<short>
        {
            typedef float Type;
        };

        /*
         * Manhattan distance (city block distance) functor
         */
        template<class T>
        struct CV_EXPORTS L1
        {
            enum { normType = NORM_L1 };
            typedef T ValueType;
            typedef typename Accumulator<T>::Type ResultType;

            ResultType operator()( const T *a, const T *b, int size ) const
            {
                return normL1<ValueType, ResultType>(a, b, size);
            }
        };

        /*
         * Euclidean distance functor
         */
        template<class T>
        struct CV_EXPORTS L2
        {
            enum { normType = NORM_L2 };
            typedef T ValueType;
            typedef typename Accumulator<T>::Type ResultType;

            ResultType operator()( const T *a, const T *b, int size ) const
            {
                return (ResultType)sqrt((double)normL2Sqr<ValueType, ResultType>(a, b, size));
            }
        };

        /*
         * Hamming distance functor - counts the bit differences between two strings - useful for the Brief descriptor
         * bit count of A exclusive XOR'ed with B
         */
        struct CV_EXPORTS Hamming
        {
            enum { normType = NORM_HAMMING };
            typedef unsigned char ValueType;
            typedef int ResultType;

            /** this will count the bits in a ^ b
             */
            ResultType operator()( const unsigned char *a, const unsigned char *b, int size ) const
            {
                return normHamming(a, b, size);
            }
        };

        ////////////////////////////////// BruteForceMatcher //////////////////////////////////

        class CV_EXPORTS BruteForceMatcher_OCL_base
        {
        public:
            enum DistType {L1Dist = 0, L2Dist, HammingDist};
            explicit BruteForceMatcher_OCL_base(DistType distType = L2Dist);
            // Add descriptors to train descriptor collection
            void add(const std::vector<oclMat> &descCollection);
            // Get train descriptors collection
            const std::vector<oclMat> &getTrainDescriptors() const;
            // Clear train descriptors collection
            void clear();
            // Return true if there are not train descriptors in collection
            bool empty() const;

            // Return true if the matcher supports mask in match methods
            bool isMaskSupported() const;

            // Find one best match for each query descriptor
            void matchSingle(const oclMat &query, const oclMat &train,
                             oclMat &trainIdx, oclMat &distance,
                             const oclMat &mask = oclMat());

            // Download trainIdx and distance and convert it to CPU vector with DMatch
            static void matchDownload(const oclMat &trainIdx, const oclMat &distance, std::vector<DMatch> &matches);
            // Convert trainIdx and distance to vector with DMatch
            static void matchConvert(const Mat &trainIdx, const Mat &distance, std::vector<DMatch> &matches);

            // Find one best match for each query descriptor
            void match(const oclMat &query, const oclMat &train, std::vector<DMatch> &matches, const oclMat &mask = oclMat());

            // Make gpu collection of trains and masks in suitable format for matchCollection function
            void makeGpuCollection(oclMat &trainCollection, oclMat &maskCollection, const std::vector<oclMat> &masks = std::vector<oclMat>());


            // Find one best match from train collection for each query descriptor
            void matchCollection(const oclMat &query, const oclMat &trainCollection,
                                 oclMat &trainIdx, oclMat &imgIdx, oclMat &distance,
                                 const oclMat &masks = oclMat());

            // Download trainIdx, imgIdx and distance and convert it to vector with DMatch
            static void matchDownload(const oclMat &trainIdx, const oclMat &imgIdx, const oclMat &distance, std::vector<DMatch> &matches);
            // Convert trainIdx, imgIdx and distance to vector with DMatch
            static void matchConvert(const Mat &trainIdx, const Mat &imgIdx, const Mat &distance, std::vector<DMatch> &matches);

            // Find one best match from train collection for each query descriptor.
            void match(const oclMat &query, std::vector<DMatch> &matches, const std::vector<oclMat> &masks = std::vector<oclMat>());

            // Find k best matches for each query descriptor (in increasing order of distances)
            void knnMatchSingle(const oclMat &query, const oclMat &train,
                                oclMat &trainIdx, oclMat &distance, oclMat &allDist, int k,
                                const oclMat &mask = oclMat());

            // Download trainIdx and distance and convert it to vector with DMatch
            // compactResult is used when mask is not empty. If compactResult is false matches
            // vector will have the same size as queryDescriptors rows. If compactResult is true
            // matches vector will not contain matches for fully masked out query descriptors.
            static void knnMatchDownload(const oclMat &trainIdx, const oclMat &distance,
                                         std::vector< std::vector<DMatch> > &matches, bool compactResult = false);

            // Convert trainIdx and distance to vector with DMatch
            static void knnMatchConvert(const Mat &trainIdx, const Mat &distance,
                                        std::vector< std::vector<DMatch> > &matches, bool compactResult = false);

            // Find k best matches for each query descriptor (in increasing order of distances).
            // compactResult is used when mask is not empty. If compactResult is false matches
            // vector will have the same size as queryDescriptors rows. If compactResult is true
            // matches vector will not contain matches for fully masked out query descriptors.
            void knnMatch(const oclMat &query, const oclMat &train,
                          std::vector< std::vector<DMatch> > &matches, int k, const oclMat &mask = oclMat(),
                          bool compactResult = false);

            // Find k best matches from train collection for each query descriptor (in increasing order of distances)
            void knnMatch2Collection(const oclMat &query, const oclMat &trainCollection,
                                     oclMat &trainIdx, oclMat &imgIdx, oclMat &distance,
                                     const oclMat &maskCollection = oclMat());

            // Download trainIdx and distance and convert it to vector with DMatch
            // compactResult is used when mask is not empty. If compactResult is false matches
            // vector will have the same size as queryDescriptors rows. If compactResult is true
            // matches vector will not contain matches for fully masked out query descriptors.
            static void knnMatch2Download(const oclMat &trainIdx, const oclMat &imgIdx, const oclMat &distance,
                                          std::vector< std::vector<DMatch> > &matches, bool compactResult = false);

            // Convert trainIdx and distance to vector with DMatch
            static void knnMatch2Convert(const Mat &trainIdx, const Mat &imgIdx, const Mat &distance,
                                         std::vector< std::vector<DMatch> > &matches, bool compactResult = false);

            // Find k best matches  for each query descriptor (in increasing order of distances).
            // compactResult is used when mask is not empty. If compactResult is false matches
            // vector will have the same size as queryDescriptors rows. If compactResult is true
            // matches vector will not contain matches for fully masked out query descriptors.
            void knnMatch(const oclMat &query, std::vector< std::vector<DMatch> > &matches, int k,
                          const std::vector<oclMat> &masks = std::vector<oclMat>(), bool compactResult = false);

            // Find best matches for each query descriptor which have distance less than maxDistance.
            // nMatches.at<int>(0, queryIdx) will contain matches count for queryIdx.
            // carefully nMatches can be greater than trainIdx.cols - it means that matcher didn't find all matches,
            // because it didn't have enough memory.
            // If trainIdx is empty, then trainIdx and distance will be created with size nQuery x max((nTrain / 100), 10),
            // otherwize user can pass own allocated trainIdx and distance with size nQuery x nMaxMatches
            // Matches doesn't sorted.
            void radiusMatchSingle(const oclMat &query, const oclMat &train,
                                   oclMat &trainIdx, oclMat &distance, oclMat &nMatches, float maxDistance,
                                   const oclMat &mask = oclMat());

            // Download trainIdx, nMatches and distance and convert it to vector with DMatch.
            // matches will be sorted in increasing order of distances.
            // compactResult is used when mask is not empty. If compactResult is false matches
            // vector will have the same size as queryDescriptors rows. If compactResult is true
            // matches vector will not contain matches for fully masked out query descriptors.
            static void radiusMatchDownload(const oclMat &trainIdx, const oclMat &distance, const oclMat &nMatches,
                                            std::vector< std::vector<DMatch> > &matches, bool compactResult = false);
            // Convert trainIdx, nMatches and distance to vector with DMatch.
            static void radiusMatchConvert(const Mat &trainIdx, const Mat &distance, const Mat &nMatches,
                                           std::vector< std::vector<DMatch> > &matches, bool compactResult = false);
            // Find best matches for each query descriptor which have distance less than maxDistance
            // in increasing order of distances).
            void radiusMatch(const oclMat &query, const oclMat &train,
                             std::vector< std::vector<DMatch> > &matches, float maxDistance,
                             const oclMat &mask = oclMat(), bool compactResult = false);
            // Find best matches for each query descriptor which have distance less than maxDistance.
            // If trainIdx is empty, then trainIdx and distance will be created with size nQuery x max((nQuery / 100), 10),
            // otherwize user can pass own allocated trainIdx and distance with size nQuery x nMaxMatches
            // Matches doesn't sorted.
            void radiusMatchCollection(const oclMat &query, oclMat &trainIdx, oclMat &imgIdx, oclMat &distance, oclMat &nMatches, float maxDistance,
                                       const std::vector<oclMat> &masks = std::vector<oclMat>());
            // Download trainIdx, imgIdx, nMatches and distance and convert it to vector with DMatch.
            // matches will be sorted in increasing order of distances.
            // compactResult is used when mask is not empty. If compactResult is false matches
            // vector will have the same size as queryDescriptors rows. If compactResult is true
            // matches vector will not contain matches for fully masked out query descriptors.
            static void radiusMatchDownload(const oclMat &trainIdx, const oclMat &imgIdx, const oclMat &distance, const oclMat &nMatches,
                                            std::vector< std::vector<DMatch> > &matches, bool compactResult = false);
            // Convert trainIdx, nMatches and distance to vector with DMatch.
            static void radiusMatchConvert(const Mat &trainIdx, const Mat &imgIdx, const Mat &distance, const Mat &nMatches,
                                           std::vector< std::vector<DMatch> > &matches, bool compactResult = false);
            // Find best matches from train collection for each query descriptor which have distance less than
            // maxDistance (in increasing order of distances).
            void radiusMatch(const oclMat &query, std::vector< std::vector<DMatch> > &matches, float maxDistance,
                             const std::vector<oclMat> &masks = std::vector<oclMat>(), bool compactResult = false);
            DistType distType;
        private:
            std::vector<oclMat> trainDescCollection;
        };

        template <class Distance>
        class CV_EXPORTS BruteForceMatcher_OCL;

        template <typename T>
        class CV_EXPORTS BruteForceMatcher_OCL< L1<T> > : public BruteForceMatcher_OCL_base
        {
        public:
            explicit BruteForceMatcher_OCL() : BruteForceMatcher_OCL_base(L1Dist) {}
            explicit BruteForceMatcher_OCL(L1<T> /*d*/) : BruteForceMatcher_OCL_base(L1Dist) {}
        };

        template <typename T>
        class CV_EXPORTS BruteForceMatcher_OCL< L2<T> > : public BruteForceMatcher_OCL_base
        {
        public:
            explicit BruteForceMatcher_OCL() : BruteForceMatcher_OCL_base(L2Dist) {}
            explicit BruteForceMatcher_OCL(L2<T> /*d*/) : BruteForceMatcher_OCL_base(L2Dist) {}
        };

        template <> class CV_EXPORTS BruteForceMatcher_OCL< Hamming > : public BruteForceMatcher_OCL_base
        {
        public:
            explicit BruteForceMatcher_OCL() : BruteForceMatcher_OCL_base(HammingDist) {}
            explicit BruteForceMatcher_OCL(Hamming /*d*/) : BruteForceMatcher_OCL_base(HammingDist) {}
        };

        class CV_EXPORTS BFMatcher_OCL : public BruteForceMatcher_OCL_base
        {
        public:
            explicit BFMatcher_OCL(int norm = NORM_L2) : BruteForceMatcher_OCL_base(norm == NORM_L1 ? L1Dist : norm == NORM_L2 ? L2Dist : HammingDist) {}
        };

        class CV_EXPORTS GoodFeaturesToTrackDetector_OCL
        {
        public:
            explicit GoodFeaturesToTrackDetector_OCL(int maxCorners = 1000, double qualityLevel = 0.01, double minDistance = 0.0,
                int blockSize = 3, bool useHarrisDetector = false, double harrisK = 0.04);

            //! return 1 rows matrix with CV_32FC2 type
            void operator ()(const oclMat& image, oclMat& corners, const oclMat& mask = oclMat());
            //! download points of type Point2f to a vector. the vector's content will be erased
            void downloadPoints(const oclMat &points, vector<Point2f> &points_v);

            int maxCorners;
            double qualityLevel;
            double minDistance;

            int blockSize;
            bool useHarrisDetector;
            double harrisK;
            void releaseMemory()
            {
                Dx_.release();
                Dy_.release();
                eig_.release();
                minMaxbuf_.release();
                tmpCorners_.release();
            }
        private:
            oclMat Dx_;
            oclMat Dy_;
            oclMat eig_;
            oclMat eig_minmax_;
            oclMat minMaxbuf_;
            oclMat tmpCorners_;
            oclMat counter_;
        };

        inline GoodFeaturesToTrackDetector_OCL::GoodFeaturesToTrackDetector_OCL(int maxCorners_, double qualityLevel_, double minDistance_,
            int blockSize_, bool useHarrisDetector_, double harrisK_)
        {
            maxCorners = maxCorners_;
            qualityLevel = qualityLevel_;
            minDistance = minDistance_;
            blockSize = blockSize_;
            useHarrisDetector = useHarrisDetector_;
            harrisK = harrisK_;
        }

        /////////////////////////////// PyrLKOpticalFlow /////////////////////////////////////
        class CV_EXPORTS PyrLKOpticalFlow
        {
        public:
            PyrLKOpticalFlow()
            {
                winSize = Size(21, 21);
                maxLevel = 3;
                iters = 30;
                derivLambda = 0.5;
                useInitialFlow = false;
                minEigThreshold = 1e-4f;
                getMinEigenVals = false;
                isDeviceArch11_ = false;
            }

            void sparse(const oclMat &prevImg, const oclMat &nextImg, const oclMat &prevPts, oclMat &nextPts,
                        oclMat &status, oclMat *err = 0);
            void dense(const oclMat &prevImg, const oclMat &nextImg, oclMat &u, oclMat &v, oclMat *err = 0);
            Size winSize;
            int maxLevel;
            int iters;
            double derivLambda;
            bool useInitialFlow;
            float minEigThreshold;
            bool getMinEigenVals;
            void releaseMemory()
            {
                dx_calcBuf_.release();
                dy_calcBuf_.release();

                prevPyr_.clear();
                nextPyr_.clear();

                dx_buf_.release();
                dy_buf_.release();
            }
        private:
            void calcSharrDeriv(const oclMat &src, oclMat &dx, oclMat &dy);
            void buildImagePyramid(const oclMat &img0, vector<oclMat> &pyr, bool withBorder);

            oclMat dx_calcBuf_;
            oclMat dy_calcBuf_;

            vector<oclMat> prevPyr_;
            vector<oclMat> nextPyr_;

            oclMat dx_buf_;
            oclMat dy_buf_;
            oclMat uPyr_[2];
            oclMat vPyr_[2];
            bool isDeviceArch11_;
        };

        class CV_EXPORTS FarnebackOpticalFlow
        {
        public:
            FarnebackOpticalFlow();

            int numLevels;
            double pyrScale;
            bool fastPyramids;
            int winSize;
            int numIters;
            int polyN;
            double polySigma;
            int flags;

            void operator ()(const oclMat &frame0, const oclMat &frame1, oclMat &flowx, oclMat &flowy);

            void releaseMemory();

        private:
            void prepareGaussian(
                int n, double sigma, float *g, float *xg, float *xxg,
                double &ig11, double &ig03, double &ig33, double &ig55);

            void setPolynomialExpansionConsts(int n, double sigma);

            void updateFlow_boxFilter(
                const oclMat& R0, const oclMat& R1, oclMat& flowx, oclMat &flowy,
                oclMat& M, oclMat &bufM, int blockSize, bool updateMatrices);

            void updateFlow_gaussianBlur(
                const oclMat& R0, const oclMat& R1, oclMat& flowx, oclMat& flowy,
                oclMat& M, oclMat &bufM, int blockSize, bool updateMatrices);

            oclMat frames_[2];
            oclMat pyrLevel_[2], M_, bufM_, R_[2], blurredFrame_[2];
            std::vector<oclMat> pyramid0_, pyramid1_;
        };

        //////////////// build warping maps ////////////////////
        //! builds plane warping maps
        CV_EXPORTS void buildWarpPlaneMaps(Size src_size, Rect dst_roi, const Mat &K, const Mat &R, const Mat &T, float scale, oclMat &map_x, oclMat &map_y);
        //! builds cylindrical warping maps
        CV_EXPORTS void buildWarpCylindricalMaps(Size src_size, Rect dst_roi, const Mat &K, const Mat &R, float scale, oclMat &map_x, oclMat &map_y);
        //! builds spherical warping maps
        CV_EXPORTS void buildWarpSphericalMaps(Size src_size, Rect dst_roi, const Mat &K, const Mat &R, float scale, oclMat &map_x, oclMat &map_y);
        //! builds Affine warping maps
        CV_EXPORTS void buildWarpAffineMaps(const Mat &M, bool inverse, Size dsize, oclMat &xmap, oclMat &ymap);

        //! builds Perspective warping maps
        CV_EXPORTS void buildWarpPerspectiveMaps(const Mat &M, bool inverse, Size dsize, oclMat &xmap, oclMat &ymap);

        ///////////////////////////////////// interpolate frames //////////////////////////////////////////////
        //! Interpolate frames (images) using provided optical flow (displacement field).
        //! frame0   - frame 0 (32-bit floating point images, single channel)
        //! frame1   - frame 1 (the same type and size)
        //! fu       - forward horizontal displacement
        //! fv       - forward vertical displacement
        //! bu       - backward horizontal displacement
        //! bv       - backward vertical displacement
        //! pos      - new frame position
        //! newFrame - new frame
        //! buf      - temporary buffer, will have width x 6*height size, CV_32FC1 type and contain 6 oclMat;
        //!            occlusion masks            0, occlusion masks            1,
        //!            interpolated forward flow  0, interpolated forward flow  1,
        //!            interpolated backward flow 0, interpolated backward flow 1
        //!
        CV_EXPORTS void interpolateFrames(const oclMat &frame0, const oclMat &frame1,
                                          const oclMat &fu, const oclMat &fv,
                                          const oclMat &bu, const oclMat &bv,
                                          float pos, oclMat &newFrame, oclMat &buf);

        //! computes moments of the rasterized shape or a vector of points
        //! _array should be a vector a points standing for the contour
        CV_EXPORTS Moments ocl_moments(InputArray contour);
        //! src should be a general image uploaded to the GPU.
        //! the supported oclMat type are CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1 and CV_64FC1
        //! to use type of CV_64FC1, the GPU should support CV_64FC1
        CV_EXPORTS Moments ocl_moments(oclMat& src, bool binary);

        class CV_EXPORTS StereoBM_OCL
        {
        public:
            enum { BASIC_PRESET = 0, PREFILTER_XSOBEL = 1 };

            enum { DEFAULT_NDISP = 64, DEFAULT_WINSZ = 19 };

            //! the default constructor
            StereoBM_OCL();
            //! the full constructor taking the camera-specific preset, number of disparities and the SAD window size. ndisparities must be multiple of 8.
            StereoBM_OCL(int preset, int ndisparities = DEFAULT_NDISP, int winSize = DEFAULT_WINSZ);

            //! the stereo correspondence operator. Finds the disparity for the specified rectified stereo pair
            //! Output disparity has CV_8U type.
            void operator() ( const oclMat &left, const oclMat &right, oclMat &disparity);

            //! Some heuristics that tries to estmate
            // if current GPU will be faster then CPU in this algorithm.
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
            oclMat minSSD, leBuf, riBuf;
        };

        class CV_EXPORTS StereoBeliefPropagation
        {
        public:
            enum { DEFAULT_NDISP  = 64 };
            enum { DEFAULT_ITERS  = 5  };
            enum { DEFAULT_LEVELS = 5  };
            static void estimateRecommendedParams(int width, int height, int &ndisp, int &iters, int &levels);
            explicit StereoBeliefPropagation(int ndisp  = DEFAULT_NDISP,
                                             int iters  = DEFAULT_ITERS,
                                             int levels = DEFAULT_LEVELS,
                                             int msg_type = CV_16S);
            StereoBeliefPropagation(int ndisp, int iters, int levels,
                                    float max_data_term, float data_weight,
                                    float max_disc_term, float disc_single_jump,
                                    int msg_type = CV_32F);
            void operator()(const oclMat &left, const oclMat &right, oclMat &disparity);
            void operator()(const oclMat &data, oclMat &disparity);
            int ndisp;
            int iters;
            int levels;
            float max_data_term;
            float data_weight;
            float max_disc_term;
            float disc_single_jump;
            int msg_type;
        private:
            oclMat u, d, l, r, u2, d2, l2, r2;
            std::vector<oclMat> datas;
            oclMat out;
        };

        class CV_EXPORTS StereoConstantSpaceBP
        {
        public:
            enum { DEFAULT_NDISP    = 128 };
            enum { DEFAULT_ITERS    = 8   };
            enum { DEFAULT_LEVELS   = 4   };
            enum { DEFAULT_NR_PLANE = 4   };
            static void estimateRecommendedParams(int width, int height, int &ndisp, int &iters, int &levels, int &nr_plane);
            explicit StereoConstantSpaceBP(
                int ndisp    = DEFAULT_NDISP,
                int iters    = DEFAULT_ITERS,
                int levels   = DEFAULT_LEVELS,
                int nr_plane = DEFAULT_NR_PLANE,
                int msg_type = CV_32F);
            StereoConstantSpaceBP(int ndisp, int iters, int levels, int nr_plane,
                float max_data_term, float data_weight, float max_disc_term, float disc_single_jump,
                int min_disp_th = 0,
                int msg_type = CV_32F);
            void operator()(const oclMat &left, const oclMat &right, oclMat &disparity);
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
            oclMat u[2], d[2], l[2], r[2];
            oclMat disp_selected_pyr[2];
            oclMat data_cost;
            oclMat data_cost_selected;
            oclMat temp;
            oclMat out;
        };

        // Implementation of the Zach, Pock and Bischof Dual TV-L1 Optical Flow method
        //
        // see reference:
        //   [1] C. Zach, T. Pock and H. Bischof, "A Duality Based Approach for Realtime TV-L1 Optical Flow".
        //   [2] Javier Sanchez, Enric Meinhardt-Llopis and Gabriele Facciolo. "TV-L1 Optical Flow Estimation".
        class CV_EXPORTS OpticalFlowDual_TVL1_OCL
        {
        public:
            OpticalFlowDual_TVL1_OCL();

            void operator ()(const oclMat& I0, const oclMat& I1, oclMat& flowx, oclMat& flowy);

            void collectGarbage();

            /**
            * Time step of the numerical scheme.
            */
            double tau;

            /**
            * Weight parameter for the data term, attachment parameter.
            * This is the most relevant parameter, which determines the smoothness of the output.
            * The smaller this parameter is, the smoother the solutions we obtain.
            * It depends on the range of motions of the images, so its value should be adapted to each image sequence.
            */
            double lambda;

            /**
            * Weight parameter for (u - v)^2, tightness parameter.
            * It serves as a link between the attachment and the regularization terms.
            * In theory, it should have a small value in order to maintain both parts in correspondence.
            * The method is stable for a large range of values of this parameter.
            */
            double theta;

            /**
            * Number of scales used to create the pyramid of images.
            */
            int nscales;

            /**
            * Number of warpings per scale.
            * Represents the number of times that I1(x+u0) and grad( I1(x+u0) ) are computed per scale.
            * This is a parameter that assures the stability of the method.
            * It also affects the running time, so it is a compromise between speed and accuracy.
            */
            int warps;

            /**
            * Stopping criterion threshold used in the numerical scheme, which is a trade-off between precision and running time.
            * A small value will yield more accurate solutions at the expense of a slower convergence.
            */
            double epsilon;

            /**
            * Stopping criterion iterations number used in the numerical scheme.
            */
            int iterations;

            bool useInitialFlow;

        private:
            void procOneScale(const oclMat& I0, const oclMat& I1, oclMat& u1, oclMat& u2);

            std::vector<oclMat> I0s;
            std::vector<oclMat> I1s;
            std::vector<oclMat> u1s;
            std::vector<oclMat> u2s;

            oclMat I1x_buf;
            oclMat I1y_buf;

            oclMat I1w_buf;
            oclMat I1wx_buf;
            oclMat I1wy_buf;

            oclMat grad_buf;
            oclMat rho_c_buf;

            oclMat p11_buf;
            oclMat p12_buf;
            oclMat p21_buf;
            oclMat p22_buf;

            oclMat diff_buf;
            oclMat norm_buf;
        };
        // current supported sorting methods
        enum
        {
            SORT_BITONIC,   // only support power-of-2 buffer size
            SORT_SELECTION, // cannot sort duplicate keys
            SORT_MERGE,
            SORT_RADIX      // only support signed int/float keys(CV_32S/CV_32F)
        };
        //! Returns the sorted result of all the elements in input based on equivalent keys.
        //
        //  The element unit in the values to be sorted is determined from the data type,
        //  i.e., a CV_32FC2 input {a1a2, b1b2} will be considered as two elements, regardless its
        //  matrix dimension.
        //  both keys and values will be sorted inplace
        //  Key needs to be single channel oclMat.
        //
        //  Example:
        //  input -
        //    keys   = {2,    3,   1}   (CV_8UC1)
        //    values = {10,5, 4,3, 6,2} (CV_8UC2)
        //  sortByKey(keys, values, SORT_SELECTION, false);
        //  output -
        //    keys   = {1,    2,   3}   (CV_8UC1)
        //    values = {6,2, 10,5, 4,3} (CV_8UC2)
        CV_EXPORTS void sortByKey(oclMat& keys, oclMat& values, int method, bool isGreaterThan = false);
        /*!Base class for MOG and MOG2!*/
        class CV_EXPORTS BackgroundSubtractor
        {
        public:
            //! the virtual destructor
            virtual ~BackgroundSubtractor();
            //! the update operator that takes the next video frame and returns the current foreground mask as 8-bit binary image.
            virtual void operator()(const oclMat& image, oclMat& fgmask, float learningRate);

            //! computes a background image
            virtual void getBackgroundImage(oclMat& backgroundImage) const = 0;
        };
                /*!
        Gaussian Mixture-based Backbround/Foreground Segmentation Algorithm

        The class implements the following algorithm:
        "An improved adaptive background mixture model for real-time tracking with shadow detection"
        P. KadewTraKuPong and R. Bowden,
        Proc. 2nd European Workshp on Advanced Video-Based Surveillance Systems, 2001."
        http://personal.ee.surrey.ac.uk/Personal/R.Bowden/publications/avbs01/avbs01.pdf
        */
        class CV_EXPORTS MOG: public cv::ocl::BackgroundSubtractor
        {
        public:
            //! the default constructor
            MOG(int nmixtures = -1);

            //! re-initiaization method
            void initialize(Size frameSize, int frameType);

            //! the update operator
            void operator()(const oclMat& frame, oclMat& fgmask, float learningRate = 0.f);

            //! computes a background image which are the mean of all background gaussians
            void getBackgroundImage(oclMat& backgroundImage) const;

            //! releases all inner buffers
            void release();

            int history;
            float varThreshold;
            float backgroundRatio;
            float noiseSigma;

        private:
            int nmixtures_;

            Size frameSize_;
            int frameType_;
            int nframes_;

            oclMat weight_;
            oclMat sortKey_;
            oclMat mean_;
            oclMat var_;
        };

        /*!
        The class implements the following algorithm:
        "Improved adaptive Gausian mixture model for background subtraction"
        Z.Zivkovic
        International Conference Pattern Recognition, UK, August, 2004.
        http://www.zoranz.net/Publications/zivkovic2004ICPR.pdf
        */
        class CV_EXPORTS MOG2: public cv::ocl::BackgroundSubtractor
        {
        public:
            //! the default constructor
            MOG2(int nmixtures = -1);

            //! re-initiaization method
            void initialize(Size frameSize, int frameType);

            //! the update operator
            void operator()(const oclMat& frame, oclMat& fgmask, float learningRate = -1.0f);

            //! computes a background image which are the mean of all background gaussians
            void getBackgroundImage(oclMat& backgroundImage) const;

            //! releases all inner buffers
            void release();

            // parameters
            // you should call initialize after parameters changes

            int history;

            //! here it is the maximum allowed number of mixture components.
            //! Actual number is determined dynamically per pixel
            float varThreshold;
            // threshold on the squared Mahalanobis distance to decide if it is well described
            // by the background model or not. Related to Cthr from the paper.
            // This does not influence the update of the background. A typical value could be 4 sigma
            // and that is varThreshold=4*4=16; Corresponds to Tb in the paper.

            /////////////////////////
            // less important parameters - things you might change but be carefull
            ////////////////////////

            float backgroundRatio;
            // corresponds to fTB=1-cf from the paper
            // TB - threshold when the component becomes significant enough to be included into
            // the background model. It is the TB=1-cf from the paper. So I use cf=0.1 => TB=0.
            // For alpha=0.001 it means that the mode should exist for approximately 105 frames before
            // it is considered foreground
            // float noiseSigma;
            float varThresholdGen;

            //correspondts to Tg - threshold on the squared Mahalan. dist. to decide
            //when a sample is close to the existing components. If it is not close
            //to any a new component will be generated. I use 3 sigma => Tg=3*3=9.
            //Smaller Tg leads to more generated components and higher Tg might make
            //lead to small number of components but they can grow too large
            float fVarInit;
            float fVarMin;
            float fVarMax;

            //initial variance  for the newly generated components.
            //It will will influence the speed of adaptation. A good guess should be made.
            //A simple way is to estimate the typical standard deviation from the images.
            //I used here 10 as a reasonable value
            // min and max can be used to further control the variance
            float fCT; //CT - complexity reduction prior
            //this is related to the number of samples needed to accept that a component
            //actually exists. We use CT=0.05 of all the samples. By setting CT=0 you get
            //the standard Stauffer&Grimson algorithm (maybe not exact but very similar)

            //shadow detection parameters
            bool bShadowDetection; //default 1 - do shadow detection
            unsigned char nShadowDetection; //do shadow detection - insert this value as the detection result - 127 default value
            float fTau;
            // Tau - shadow threshold. The shadow is detected if the pixel is darker
            //version of the background. Tau is a threshold on how much darker the shadow can be.
            //Tau= 0.5 means that if pixel is more than 2 times darker then it is not shadow
            //See: Prati,Mikic,Trivedi,Cucchiarra,"Detecting Moving Shadows...",IEEE PAMI,2003.

        private:
            int nmixtures_;

            Size frameSize_;
            int frameType_;
            int nframes_;

            oclMat weight_;
            oclMat variance_;
            oclMat mean_;

            oclMat bgmodelUsedModes_; //keep track of number of modes per pixel
        };

        /*!***************Kalman Filter*************!*/
        class CV_EXPORTS KalmanFilter
        {
        public:
            KalmanFilter();
            //! the full constructor taking the dimensionality of the state, of the measurement and of the control vector
            KalmanFilter(int dynamParams, int measureParams, int controlParams=0, int type=CV_32F);
            //! re-initializes Kalman filter. The previous content is destroyed.
            void init(int dynamParams, int measureParams, int controlParams=0, int type=CV_32F);

            const oclMat& predict(const oclMat& control=oclMat());
            const oclMat& correct(const oclMat& measurement);

            oclMat statePre;           //!< predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
            oclMat statePost;          //!< corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
            oclMat transitionMatrix;   //!< state transition matrix (A)
            oclMat controlMatrix;      //!< control matrix (B) (not used if there is no control)
            oclMat measurementMatrix;  //!< measurement matrix (H)
            oclMat processNoiseCov;    //!< process noise covariance matrix (Q)
            oclMat measurementNoiseCov;//!< measurement noise covariance matrix (R)
            oclMat errorCovPre;        //!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)*/
            oclMat gain;               //!< Kalman gain matrix (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)
            oclMat errorCovPost;       //!< posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)
        private:
            oclMat temp1;
            oclMat temp2;
            oclMat temp3;
            oclMat temp4;
            oclMat temp5;
        };

        /*!***************K Nearest Neighbour*************!*/
        class CV_EXPORTS KNearestNeighbour: public CvKNearest
        {
        public:
            KNearestNeighbour();
            ~KNearestNeighbour();

            bool train(const Mat& trainData, Mat& labels, Mat& sampleIdx = Mat().setTo(Scalar::all(0)),
                bool isRegression = false, int max_k = 32, bool updateBase = false);

            void clear();

            void find_nearest(const oclMat& samples, int k, oclMat& lables);

        private:
            oclMat samples_ocl;
        };

        /*!***************  SVM  *************!*/
        class CV_EXPORTS CvSVM_OCL : public CvSVM
        {
        public:
            CvSVM_OCL();

            CvSVM_OCL(const cv::Mat& trainData, const cv::Mat& responses,
                      const cv::Mat& varIdx=cv::Mat(), const cv::Mat& sampleIdx=cv::Mat(),
                      CvSVMParams params=CvSVMParams());
            CV_WRAP float predict( const int row_index, Mat& src, bool returnDFVal=false ) const;
            CV_WRAP void predict( cv::InputArray samples, cv::OutputArray results ) const;
            CV_WRAP float predict( const cv::Mat& sample, bool returnDFVal=false ) const;
            float predict( const CvMat* samples, CV_OUT CvMat* results ) const;

        protected:
            float predict( const int row_index, int row_len, Mat& src, bool returnDFVal=false ) const;
            void create_kernel();
            void create_solver();
        };

        /*!***************  END  *************!*/
    }
}
#if defined _MSC_VER && _MSC_VER >= 1200
#  pragma warning( push)
#  pragma warning( disable: 4267)
#endif
#include "opencv2/ocl/matrix_operations.hpp"
#if defined _MSC_VER && _MSC_VER >= 1200
#  pragma warning( pop)
#endif

#endif /* __OPENCV_OCL_HPP__ */
