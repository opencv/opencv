/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef __OPENCV_CORE_CUDA_HPP__
#define __OPENCV_CORE_CUDA_HPP__

#ifndef __cplusplus
#  error cuda.hpp header must be compiled as C++
#endif

#include "opencv2/core.hpp"
#include "opencv2/core/cuda_types.hpp"

namespace cv { namespace cuda {

//////////////////////////////// GpuMat ///////////////////////////////

// Smart pointer for GPU memory with reference counting.
// Its interface is mostly similar with cv::Mat.

class CV_EXPORTS GpuMat
{
public:
    class CV_EXPORTS Allocator
    {
    public:
        virtual ~Allocator() {}

        // allocator must fill data, step and refcount fields
        virtual bool allocate(GpuMat* mat, int rows, int cols, size_t elemSize) = 0;
        virtual void free(GpuMat* mat) = 0;
    };

    //! default allocator
    static Allocator* defaultAllocator();
    static void setDefaultAllocator(Allocator* allocator);

    //! default constructor
    explicit GpuMat(Allocator* allocator = defaultAllocator());

    //! constructs GpuMat of the specified size and type
    GpuMat(int rows, int cols, int type, Allocator* allocator = defaultAllocator());
    GpuMat(Size size, int type, Allocator* allocator = defaultAllocator());

    //! constucts GpuMat and fills it with the specified value _s
    GpuMat(int rows, int cols, int type, Scalar s, Allocator* allocator = defaultAllocator());
    GpuMat(Size size, int type, Scalar s, Allocator* allocator = defaultAllocator());

    //! copy constructor
    GpuMat(const GpuMat& m);

    //! constructor for GpuMat headers pointing to user-allocated data
    GpuMat(int rows, int cols, int type, void* data, size_t step = Mat::AUTO_STEP);
    GpuMat(Size size, int type, void* data, size_t step = Mat::AUTO_STEP);

    //! creates a GpuMat header for a part of the bigger matrix
    GpuMat(const GpuMat& m, Range rowRange, Range colRange);
    GpuMat(const GpuMat& m, Rect roi);

    //! builds GpuMat from host memory (Blocking call)
    explicit GpuMat(InputArray arr, Allocator* allocator = defaultAllocator());

    //! destructor - calls release()
    ~GpuMat();

    //! assignment operators
    GpuMat& operator =(const GpuMat& m);

    //! allocates new GpuMat data unless the GpuMat already has specified size and type
    void create(int rows, int cols, int type);
    void create(Size size, int type);

    //! decreases reference counter, deallocate the data when reference counter reaches 0
    void release();

    //! swaps with other smart pointer
    void swap(GpuMat& mat);

    //! pefroms upload data to GpuMat (Blocking call)
    void upload(InputArray arr);

    //! pefroms upload data to GpuMat (Non-Blocking call)
    void upload(InputArray arr, Stream& stream);

    //! pefroms download data from device to host memory (Blocking call)
    void download(OutputArray dst) const;

    //! pefroms download data from device to host memory (Non-Blocking call)
    void download(OutputArray dst, Stream& stream) const;

    //! returns deep copy of the GpuMat, i.e. the data is copied
    GpuMat clone() const;

    //! copies the GpuMat content to device memory (Blocking call)
    void copyTo(OutputArray dst) const;

    //! copies the GpuMat content to device memory (Non-Blocking call)
    void copyTo(OutputArray dst, Stream& stream) const;

    //! copies those GpuMat elements to "m" that are marked with non-zero mask elements (Blocking call)
    void copyTo(OutputArray dst, InputArray mask) const;

    //! copies those GpuMat elements to "m" that are marked with non-zero mask elements (Non-Blocking call)
    void copyTo(OutputArray dst, InputArray mask, Stream& stream) const;

    //! sets some of the GpuMat elements to s (Blocking call)
    GpuMat& setTo(Scalar s);

    //! sets some of the GpuMat elements to s (Non-Blocking call)
    GpuMat& setTo(Scalar s, Stream& stream);

    //! sets some of the GpuMat elements to s, according to the mask (Blocking call)
    GpuMat& setTo(Scalar s, InputArray mask);

    //! sets some of the GpuMat elements to s, according to the mask (Non-Blocking call)
    GpuMat& setTo(Scalar s, InputArray mask, Stream& stream);

    //! converts GpuMat to another datatype (Blocking call)
    void convertTo(OutputArray dst, int rtype) const;

    //! converts GpuMat to another datatype (Non-Blocking call)
    void convertTo(OutputArray dst, int rtype, Stream& stream) const;

    //! converts GpuMat to another datatype with scaling (Blocking call)
    void convertTo(OutputArray dst, int rtype, double alpha, double beta = 0.0) const;

    //! converts GpuMat to another datatype with scaling (Non-Blocking call)
    void convertTo(OutputArray dst, int rtype, double alpha, Stream& stream) const;

    //! converts GpuMat to another datatype with scaling (Non-Blocking call)
    void convertTo(OutputArray dst, int rtype, double alpha, double beta, Stream& stream) const;

    void assignTo(GpuMat& m, int type=-1) const;

    //! returns pointer to y-th row
    uchar* ptr(int y = 0);
    const uchar* ptr(int y = 0) const;

    //! template version of the above method
    template<typename _Tp> _Tp* ptr(int y = 0);
    template<typename _Tp> const _Tp* ptr(int y = 0) const;

    template <typename _Tp> operator PtrStepSz<_Tp>() const;
    template <typename _Tp> operator PtrStep<_Tp>() const;

    //! returns a new GpuMat header for the specified row
    GpuMat row(int y) const;

    //! returns a new GpuMat header for the specified column
    GpuMat col(int x) const;

    //! ... for the specified row span
    GpuMat rowRange(int startrow, int endrow) const;
    GpuMat rowRange(Range r) const;

    //! ... for the specified column span
    GpuMat colRange(int startcol, int endcol) const;
    GpuMat colRange(Range r) const;

    //! extracts a rectangular sub-GpuMat (this is a generalized form of row, rowRange etc.)
    GpuMat operator ()(Range rowRange, Range colRange) const;
    GpuMat operator ()(Rect roi) const;

    //! creates alternative GpuMat header for the same data, with different
    //! number of channels and/or different number of rows
    GpuMat reshape(int cn, int rows = 0) const;

    //! locates GpuMat header within a parent GpuMat
    void locateROI(Size& wholeSize, Point& ofs) const;

    //! moves/resizes the current GpuMat ROI inside the parent GpuMat
    GpuMat& adjustROI(int dtop, int dbottom, int dleft, int dright);

    //! returns true iff the GpuMat data is continuous
    //! (i.e. when there are no gaps between successive rows)
    bool isContinuous() const;

    //! returns element size in bytes
    size_t elemSize() const;

    //! returns the size of element channel in bytes
    size_t elemSize1() const;

    //! returns element type
    int type() const;

    //! returns element type
    int depth() const;

    //! returns number of channels
    int channels() const;

    //! returns step/elemSize1()
    size_t step1() const;

    //! returns GpuMat size : width == number of columns, height == number of rows
    Size size() const;

    //! returns true if GpuMat data is NULL
    bool empty() const;

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
    //! when GpuMat points to user-allocated data, the pointer is NULL
    int* refcount;

    //! helper fields used in locateROI and adjustROI
    const uchar* datastart;
    const uchar* dataend;

    //! allocator
    Allocator* allocator;
};

//! creates continuous matrix
CV_EXPORTS void createContinuous(int rows, int cols, int type, OutputArray arr);

//! ensures that size of the given matrix is not less than (rows, cols) size
//! and matrix type is match specified one too
CV_EXPORTS void ensureSizeIsEnough(int rows, int cols, int type, OutputArray arr);

CV_EXPORTS GpuMat allocMatFromBuf(int rows, int cols, int type, GpuMat& mat);

//! BufferPool management (must be called before Stream creation)
CV_EXPORTS void setBufferPoolUsage(bool on);
CV_EXPORTS void setBufferPoolConfig(int deviceId, size_t stackSize, int stackCount);

//////////////////////////////// CudaMem ////////////////////////////////

// CudaMem is limited cv::Mat with page locked memory allocation.
// Page locked memory is only needed for async and faster coping to GPU.
// It is convertable to cv::Mat header without reference counting
// so you can use it with other opencv functions.

class CV_EXPORTS CudaMem
{
public:
    enum AllocType { PAGE_LOCKED = 1, SHARED = 2, WRITE_COMBINED = 4 };

    explicit CudaMem(AllocType alloc_type = PAGE_LOCKED);

    CudaMem(const CudaMem& m);

    CudaMem(int rows, int cols, int type, AllocType alloc_type = PAGE_LOCKED);
    CudaMem(Size size, int type, AllocType alloc_type = PAGE_LOCKED);

    //! creates from host memory with coping data
    explicit CudaMem(InputArray arr, AllocType alloc_type = PAGE_LOCKED);

    ~CudaMem();

    CudaMem& operator =(const CudaMem& m);

    //! swaps with other smart pointer
    void swap(CudaMem& b);

    //! returns deep copy of the matrix, i.e. the data is copied
    CudaMem clone() const;

    //! allocates new matrix data unless the matrix already has specified size and type.
    void create(int rows, int cols, int type);
    void create(Size size, int type);

    //! creates alternative CudaMem header for the same data, with different
    //! number of channels and/or different number of rows
    CudaMem reshape(int cn, int rows = 0) const;

    //! decrements reference counter and released memory if needed.
    void release();

    //! returns matrix header with disabled reference counting for CudaMem data.
    Mat createMatHeader() const;

    //! maps host memory into device address space and returns GpuMat header for it. Throws exception if not supported by hardware.
    GpuMat createGpuMatHeader() const;

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

    const uchar* datastart;
    const uchar* dataend;

    AllocType alloc_type;
};

//! page-locks the matrix m memory and maps it for the device(s)
CV_EXPORTS void registerPageLocked(Mat& m);

//! unmaps the memory of matrix m, and makes it pageable again
CV_EXPORTS void unregisterPageLocked(Mat& m);

///////////////////////////////// Stream //////////////////////////////////

// Encapculates Cuda Stream. Provides interface for async coping.
// Passed to each function that supports async kernel execution.
// Reference counting is enabled.

class CV_EXPORTS Stream
{
    typedef void (Stream::*bool_type)() const;
    void this_type_does_not_support_comparisons() const {}

public:
    typedef void (*StreamCallback)(int status, void* userData);

    //! creates a new asynchronous stream
    Stream();

    //! queries an asynchronous stream for completion status
    bool queryIfComplete() const;

    //! waits for stream tasks to complete
    void waitForCompletion();

    //! makes a compute stream wait on an event
    void waitEvent(const Event& event);

    //! adds a callback to be called on the host after all currently enqueued items in the stream have completed
    void enqueueHostCallback(StreamCallback callback, void* userData);

    //! return Stream object for default CUDA stream
    static Stream& Null();

    //! returns true if stream object is not default (!= 0)
    operator bool_type() const;

    class Impl;

private:
    Ptr<Impl> impl_;
    Stream(const Ptr<Impl>& impl);

    friend struct StreamAccessor;
    friend class BufferPool;
};

class CV_EXPORTS Event
{
public:
    enum CreateFlags
    {
        DEFAULT        = 0x00,  /**< Default event flag */
        BLOCKING_SYNC  = 0x01,  /**< Event uses blocking synchronization */
        DISABLE_TIMING = 0x02,  /**< Event will not record timing data */
        INTERPROCESS   = 0x04   /**< Event is suitable for interprocess use. DisableTiming must be set */
    };

    explicit Event(CreateFlags flags = DEFAULT);

    //! records an event
    void record(Stream& stream = Stream::Null());

    //! queries an event's status
    bool queryIfComplete() const;

    //! waits for an event to complete
    void waitForCompletion();

    //! computes the elapsed time between events
    static float elapsedTime(const Event& start, const Event& end);

    class Impl;

private:
    Ptr<Impl> impl_;

    friend struct EventAccessor;
};

//////////////////////////////// Initialization & Info ////////////////////////

//! this is the only function that do not throw exceptions if the library is compiled without CUDA
CV_EXPORTS int getCudaEnabledDeviceCount();

//! set device to be used for GPU executions for the calling host thread
CV_EXPORTS void setDevice(int device);

//! returns which device is currently being used for the calling host thread
CV_EXPORTS int getDevice();

//! explicitly destroys and cleans up all resources associated with the current device in the current process
//! any subsequent API call to this device will reinitialize the device
CV_EXPORTS void resetDevice();

enum FeatureSet
{
    FEATURE_SET_COMPUTE_10 = 10,
    FEATURE_SET_COMPUTE_11 = 11,
    FEATURE_SET_COMPUTE_12 = 12,
    FEATURE_SET_COMPUTE_13 = 13,
    FEATURE_SET_COMPUTE_20 = 20,
    FEATURE_SET_COMPUTE_21 = 21,
    FEATURE_SET_COMPUTE_30 = 30,
    FEATURE_SET_COMPUTE_35 = 35,

    GLOBAL_ATOMICS = FEATURE_SET_COMPUTE_11,
    SHARED_ATOMICS = FEATURE_SET_COMPUTE_12,
    NATIVE_DOUBLE = FEATURE_SET_COMPUTE_13,
    WARP_SHUFFLE_FUNCTIONS = FEATURE_SET_COMPUTE_30,
    DYNAMIC_PARALLELISM = FEATURE_SET_COMPUTE_35
};

//! checks whether current device supports the given feature
CV_EXPORTS bool deviceSupports(FeatureSet feature_set);

//! information about what GPU archs this OpenCV CUDA module was compiled for
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
};

//! information about the given GPU.
class CV_EXPORTS DeviceInfo
{
public:
    //! creates DeviceInfo object for the current GPU
    DeviceInfo();

    //! creates DeviceInfo object for the given GPU
    DeviceInfo(int device_id);

    //! device number.
    int deviceID() const;

    //! ASCII string identifying device
    const char* name() const;

    //! global memory available on device in bytes
    size_t totalGlobalMem() const;

    //! shared memory available per block in bytes
    size_t sharedMemPerBlock() const;

    //! 32-bit registers available per block
    int regsPerBlock() const;

    //! warp size in threads
    int warpSize() const;

    //! maximum pitch in bytes allowed by memory copies
    size_t memPitch() const;

    //! maximum number of threads per block
    int maxThreadsPerBlock() const;

    //! maximum size of each dimension of a block
    Vec3i maxThreadsDim() const;

    //! maximum size of each dimension of a grid
    Vec3i maxGridSize() const;

    //! clock frequency in kilohertz
    int clockRate() const;

    //! constant memory available on device in bytes
    size_t totalConstMem() const;

    //! major compute capability
    int majorVersion() const;

    //! minor compute capability
    int minorVersion() const;

    //! alignment requirement for textures
    size_t textureAlignment() const;

    //! pitch alignment requirement for texture references bound to pitched memory
    size_t texturePitchAlignment() const;

    //! number of multiprocessors on device
    int multiProcessorCount() const;

    //! specified whether there is a run time limit on kernels
    bool kernelExecTimeoutEnabled() const;

    //! device is integrated as opposed to discrete
    bool integrated() const;

    //! device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
    bool canMapHostMemory() const;

    enum ComputeMode
    {
        ComputeModeDefault,         /**< default compute mode (Multiple threads can use ::cudaSetDevice() with this device) */
        ComputeModeExclusive,       /**< compute-exclusive-thread mode (Only one thread in one process will be able to use ::cudaSetDevice() with this device) */
        ComputeModeProhibited,      /**< compute-prohibited mode (No threads can use ::cudaSetDevice() with this device) */
        ComputeModeExclusiveProcess /**< compute-exclusive-process mode (Many threads in one process will be able to use ::cudaSetDevice() with this device) */
    };

    //! compute mode
    ComputeMode computeMode() const;

    //! maximum 1D texture size
    int maxTexture1D() const;

    //! maximum 1D mipmapped texture size
    int maxTexture1DMipmap() const;

    //! maximum size for 1D textures bound to linear memory
    int maxTexture1DLinear() const;

    //! maximum 2D texture dimensions
    Vec2i maxTexture2D() const;

    //! maximum 2D mipmapped texture dimensions
    Vec2i maxTexture2DMipmap() const;

    //! maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory
    Vec3i maxTexture2DLinear() const;

    //! maximum 2D texture dimensions if texture gather operations have to be performed
    Vec2i maxTexture2DGather() const;

    //! maximum 3D texture dimensions
    Vec3i maxTexture3D() const;

    //! maximum Cubemap texture dimensions
    int maxTextureCubemap() const;

    //! maximum 1D layered texture dimensions
    Vec2i maxTexture1DLayered() const;

    //! maximum 2D layered texture dimensions
    Vec3i maxTexture2DLayered() const;

    //! maximum Cubemap layered texture dimensions
    Vec2i maxTextureCubemapLayered() const;

    //! maximum 1D surface size
    int maxSurface1D() const;

    //! maximum 2D surface dimensions
    Vec2i maxSurface2D() const;

    //! maximum 3D surface dimensions
    Vec3i maxSurface3D() const;

    //! maximum 1D layered surface dimensions
    Vec2i maxSurface1DLayered() const;

    //! maximum 2D layered surface dimensions
    Vec3i maxSurface2DLayered() const;

    //! maximum Cubemap surface dimensions
    int maxSurfaceCubemap() const;

    //! maximum Cubemap layered surface dimensions
    Vec2i maxSurfaceCubemapLayered() const;

    //! alignment requirements for surfaces
    size_t surfaceAlignment() const;

    //! device can possibly execute multiple kernels concurrently
    bool concurrentKernels() const;

    //! device has ECC support enabled
    bool ECCEnabled() const;

    //! PCI bus ID of the device
    int pciBusID() const;

    //! PCI device ID of the device
    int pciDeviceID() const;

    //! PCI domain ID of the device
    int pciDomainID() const;

    //! true if device is a Tesla device using TCC driver, false otherwise
    bool tccDriver() const;

    //! number of asynchronous engines
    int asyncEngineCount() const;

    //! device shares a unified address space with the host
    bool unifiedAddressing() const;

    //! peak memory clock frequency in kilohertz
    int memoryClockRate() const;

    //! global memory bus width in bits
    int memoryBusWidth() const;

    //! size of L2 cache in bytes
    int l2CacheSize() const;

    //! maximum resident threads per multiprocessor
    int maxThreadsPerMultiProcessor() const;

    //! gets free and total device memory
    void queryMemory(size_t& totalMemory, size_t& freeMemory) const;
    size_t freeMemory() const;
    size_t totalMemory() const;

    //! checks whether device supports the given feature
    bool supports(FeatureSet feature_set) const;

    //! checks whether the CUDA module can be run on the given device
    bool isCompatible() const;

private:
    int device_id_;
};

CV_EXPORTS void printCudaDeviceInfo(int device);
CV_EXPORTS void printShortCudaDeviceInfo(int device);

}} // namespace cv { namespace cuda {


#include "opencv2/core/cuda.inl.hpp"

#endif /* __OPENCV_CORE_CUDA_HPP__ */
