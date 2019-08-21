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

#ifndef OPENCV_CORE_CUDA_HPP
#define OPENCV_CORE_CUDA_HPP

#ifndef __cplusplus
#  error cuda.hpp header must be compiled as C++
#endif

#include "opencv2/core.hpp"
#include "opencv2/core/cuda_types.hpp"

/**
  @defgroup cuda CUDA-accelerated Computer Vision
  @{
    @defgroup cudacore Core part
    @{
      @defgroup cudacore_init Initialization and Information
      @defgroup cudacore_struct Data Structures
    @}
  @}
 */

namespace cv { namespace cuda {

//! @addtogroup cudacore_struct
//! @{

//===================================================================================
// GpuMat
//===================================================================================

/** @brief Base storage class for GPU memory with reference counting.

Its interface matches the Mat interface with the following limitations:

-   no arbitrary dimensions support (only 2D)
-   no functions that return references to their data (because references on GPU are not valid for
    CPU)
-   no expression templates technique support

Beware that the latter limitation may lead to overloaded matrix operators that cause memory
allocations. The GpuMat class is convertible to cuda::PtrStepSz and cuda::PtrStep so it can be
passed directly to the kernel.

@note In contrast with Mat, in most cases GpuMat::isContinuous() == false . This means that rows are
aligned to a size depending on the hardware. Single-row GpuMat is always a continuous matrix.

@note You are not recommended to leave static or global GpuMat variables allocated, that is, to rely
on its destructor. The destruction order of such variables and CUDA context is undefined. GPU memory
release function returns error if the CUDA context has been destroyed before.

Some member functions are described as a "Blocking Call" while some are described as a
"Non-Blocking Call". Blocking functions are synchronous to host. It is guaranteed that the GPU
operation is finished when the function returns. However, non-blocking functions are asynchronous to
host. Those functions may return even if the GPU operation is not finished.

Compared to their blocking counterpart, non-blocking functions accept Stream as an additional
argument. If a non-default stream is passed, the GPU operation may overlap with operations in other
streams.

@sa Mat
 */
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

    //! constructs GpuMat and fills it with the specified value _s
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

    /** @brief Performs data upload to GpuMat (Blocking call)

    This function copies data from host memory to device memory. As being a blocking call, it is
    guaranteed that the copy operation is finished when this function returns.
    */
    void upload(InputArray arr);

    /** @brief Performs data upload to GpuMat (Non-Blocking call)

    This function copies data from host memory to device memory. As being a non-blocking call, this
    function may return even if the copy operation is not finished.

    The copy operation may be overlapped with operations in other non-default streams if \p stream is
    not the default stream and \p dst is HostMem allocated with HostMem::PAGE_LOCKED option.
    */
    void upload(InputArray arr, Stream& stream);

    /** @brief Performs data download from GpuMat (Blocking call)

    This function copies data from device memory to host memory. As being a blocking call, it is
    guaranteed that the copy operation is finished when this function returns.
    */
    void download(OutputArray dst) const;

    /** @brief Performs data download from GpuMat (Non-Blocking call)

    This function copies data from device memory to host memory. As being a non-blocking call, this
    function may return even if the copy operation is not finished.

    The copy operation may be overlapped with operations in other non-default streams if \p stream is
    not the default stream and \p dst is HostMem allocated with HostMem::PAGE_LOCKED option.
    */
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

    //! internal use method: updates the continuity flag
    void updateContinuityFlag();

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
    uchar* datastart;
    const uchar* dataend;

    //! allocator
    Allocator* allocator;
};

/** @brief Creates a continuous matrix.

@param rows Row count.
@param cols Column count.
@param type Type of the matrix.
@param arr Destination matrix. This parameter changes only if it has a proper type and area (
\f$\texttt{rows} \times \texttt{cols}\f$ ).

Matrix is called continuous if its elements are stored continuously, that is, without gaps at the
end of each row.
 */
CV_EXPORTS void createContinuous(int rows, int cols, int type, OutputArray arr);

/** @brief Ensures that the size of a matrix is big enough and the matrix has a proper type.

@param rows Minimum desired number of rows.
@param cols Minimum desired number of columns.
@param type Desired matrix type.
@param arr Destination matrix.

The function does not reallocate memory if the matrix has proper attributes already.
 */
CV_EXPORTS void ensureSizeIsEnough(int rows, int cols, int type, OutputArray arr);

/** @brief BufferPool for use with CUDA streams

BufferPool utilizes Stream's allocator to create new buffers for GpuMat's. It is
only useful when enabled with #setBufferPoolUsage.

@code
    setBufferPoolUsage(true);
@endcode

@note #setBufferPoolUsage must be called \em before any Stream declaration.

Users may specify custom allocator for Stream and may implement their own stream based
functions utilizing the same underlying GPU memory management.

If custom allocator is not specified, BufferPool utilizes StackAllocator by
default. StackAllocator allocates a chunk of GPU device memory beforehand,
and when GpuMat is declared later on, it is given the pre-allocated memory.
This kind of strategy reduces the number of calls for memory allocating APIs
such as cudaMalloc or cudaMallocPitch.

Below is an example that utilizes BufferPool with StackAllocator:

@code
    #include <opencv2/opencv.hpp>

    using namespace cv;
    using namespace cv::cuda

    int main()
    {
        setBufferPoolUsage(true);                               // Tell OpenCV that we are going to utilize BufferPool
        setBufferPoolConfig(getDevice(), 1024 * 1024 * 64, 2);  // Allocate 64 MB, 2 stacks (default is 10 MB, 5 stacks)

        Stream stream1, stream2;                                // Each stream uses 1 stack
        BufferPool pool1(stream1), pool2(stream2);

        GpuMat d_src1 = pool1.getBuffer(4096, 4096, CV_8UC1);   // 16MB
        GpuMat d_dst1 = pool1.getBuffer(4096, 4096, CV_8UC3);   // 48MB, pool1 is now full

        GpuMat d_src2 = pool2.getBuffer(1024, 1024, CV_8UC1);   // 1MB
        GpuMat d_dst2 = pool2.getBuffer(1024, 1024, CV_8UC3);   // 3MB

        cvtColor(d_src1, d_dst1, CV_GRAY2BGR, 0, stream1);
        cvtColor(d_src2, d_dst2, CV_GRAY2BGR, 0, stream2);
    }
@endcode

If we allocate another GpuMat on pool1 in the above example, it will be carried out by
the DefaultAllocator since the stack for pool1 is full.

@code
    GpuMat d_add1 = pool1.getBuffer(1024, 1024, CV_8UC1);   // Stack for pool1 is full, memory is allocated with DefaultAllocator
@endcode

If a third stream is declared in the above example, allocating with #getBuffer
within that stream will also be carried out by the DefaultAllocator because we've run out of
stacks.

@code
    Stream stream3;                                         // Only 2 stacks were allocated, we've run out of stacks
    BufferPool pool3(stream3);
    GpuMat d_src3 = pool3.getBuffer(1024, 1024, CV_8UC1);   // Memory is allocated with DefaultAllocator
@endcode

@warning When utilizing StackAllocator, deallocation order is important.

Just like a stack, deallocation must be done in LIFO order. Below is an example of
erroneous usage that violates LIFO rule. If OpenCV is compiled in Debug mode, this
sample code will emit CV_Assert error.

@code
    int main()
    {
        setBufferPoolUsage(true);                               // Tell OpenCV that we are going to utilize BufferPool
        Stream stream;                                          // A default size (10 MB) stack is allocated to this stream
        BufferPool pool(stream);

        GpuMat mat1 = pool.getBuffer(1024, 1024, CV_8UC1);      // Allocate mat1 (1MB)
        GpuMat mat2 = pool.getBuffer(1024, 1024, CV_8UC1);      // Allocate mat2 (1MB)

        mat1.release();                                         // erroneous usage : mat2 must be deallocated before mat1
    }
@endcode

Since C++ local variables are destroyed in the reverse order of construction,
the code sample below satisfies the LIFO rule. Local GpuMat's are deallocated
and the corresponding memory is automatically returned to the pool for later usage.

@code
    int main()
    {
        setBufferPoolUsage(true);                               // Tell OpenCV that we are going to utilize BufferPool
        setBufferPoolConfig(getDevice(), 1024 * 1024 * 64, 2);  // Allocate 64 MB, 2 stacks (default is 10 MB, 5 stacks)

        Stream stream1, stream2;                                // Each stream uses 1 stack
        BufferPool pool1(stream1), pool2(stream2);

        for (int i = 0; i < 10; i++)
        {
            GpuMat d_src1 = pool1.getBuffer(4096, 4096, CV_8UC1);   // 16MB
            GpuMat d_dst1 = pool1.getBuffer(4096, 4096, CV_8UC3);   // 48MB, pool1 is now full

            GpuMat d_src2 = pool2.getBuffer(1024, 1024, CV_8UC1);   // 1MB
            GpuMat d_dst2 = pool2.getBuffer(1024, 1024, CV_8UC3);   // 3MB

            d_src1.setTo(Scalar(i), stream1);
            d_src2.setTo(Scalar(i), stream2);

            cvtColor(d_src1, d_dst1, CV_GRAY2BGR, 0, stream1);
            cvtColor(d_src2, d_dst2, CV_GRAY2BGR, 0, stream2);
                                                                    // The order of destruction of the local variables is:
                                                                    //   d_dst2 => d_src2 => d_dst1 => d_src1
                                                                    // LIFO rule is satisfied, this code runs without error
        }
    }
@endcode
 */
class CV_EXPORTS BufferPool
{
public:

    //! Gets the BufferPool for the given stream.
    explicit BufferPool(Stream& stream);

    //! Allocates a new GpuMat of given size and type.
    GpuMat getBuffer(int rows, int cols, int type);

    //! Allocates a new GpuMat of given size and type.
    GpuMat getBuffer(Size size, int type) { return getBuffer(size.height, size.width, type); }

    //! Returns the allocator associated with the stream.
    Ptr<GpuMat::Allocator> getAllocator() const { return allocator_; }

private:
    Ptr<GpuMat::Allocator> allocator_;
};

//! BufferPool management (must be called before Stream creation)
CV_EXPORTS void setBufferPoolUsage(bool on);
CV_EXPORTS void setBufferPoolConfig(int deviceId, size_t stackSize, int stackCount);

//===================================================================================
// HostMem
//===================================================================================

/** @brief Class with reference counting wrapping special memory type allocation functions from CUDA.

Its interface is also Mat-like but with additional memory type parameters.

-   **PAGE_LOCKED** sets a page locked memory type used commonly for fast and asynchronous
    uploading/downloading data from/to GPU.
-   **SHARED** specifies a zero copy memory allocation that enables mapping the host memory to GPU
    address space, if supported.
-   **WRITE_COMBINED** sets the write combined buffer that is not cached by CPU. Such buffers are
    used to supply GPU with data when GPU only reads it. The advantage is a better CPU cache
    utilization.

@note Allocation size of such memory types is usually limited. For more details, see *CUDA 2.2
Pinned Memory APIs* document or *CUDA C Programming Guide*.
 */
class CV_EXPORTS HostMem
{
public:
    enum AllocType { PAGE_LOCKED = 1, SHARED = 2, WRITE_COMBINED = 4 };

    static MatAllocator* getAllocator(AllocType alloc_type = PAGE_LOCKED);

    explicit HostMem(AllocType alloc_type = PAGE_LOCKED);

    HostMem(const HostMem& m);

    HostMem(int rows, int cols, int type, AllocType alloc_type = PAGE_LOCKED);
    HostMem(Size size, int type, AllocType alloc_type = PAGE_LOCKED);

    //! creates from host memory with coping data
    explicit HostMem(InputArray arr, AllocType alloc_type = PAGE_LOCKED);

    ~HostMem();

    HostMem& operator =(const HostMem& m);

    //! swaps with other smart pointer
    void swap(HostMem& b);

    //! returns deep copy of the matrix, i.e. the data is copied
    HostMem clone() const;

    //! allocates new matrix data unless the matrix already has specified size and type.
    void create(int rows, int cols, int type);
    void create(Size size, int type);

    //! creates alternative HostMem header for the same data, with different
    //! number of channels and/or different number of rows
    HostMem reshape(int cn, int rows = 0) const;

    //! decrements reference counter and released memory if needed.
    void release();

    //! returns matrix header with disabled reference counting for HostMem data.
    Mat createMatHeader() const;

    /** @brief Maps CPU memory to GPU address space and creates the cuda::GpuMat header without reference counting
    for it.

    This can be done only if memory was allocated with the SHARED flag and if it is supported by the
    hardware. Laptops often share video and CPU memory, so address spaces can be mapped, which
    eliminates an extra copy.
     */
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

    uchar* datastart;
    const uchar* dataend;

    AllocType alloc_type;
};

/** @brief Page-locks the memory of matrix and maps it for the device(s).

@param m Input matrix.
 */
CV_EXPORTS void registerPageLocked(Mat& m);

/** @brief Unmaps the memory of matrix and makes it pageable again.

@param m Input matrix.
 */
CV_EXPORTS void unregisterPageLocked(Mat& m);

//===================================================================================
// Stream
//===================================================================================

/** @brief This class encapsulates a queue of asynchronous calls.

@note Currently, you may face problems if an operation is enqueued twice with different data. Some
functions use the constant GPU memory, and next call may update the memory before the previous one
has been finished. But calling different operations asynchronously is safe because each operation
has its own constant buffer. Memory copy/upload/download/set operations to the buffers you hold are
also safe.

@note The Stream class is not thread-safe. Please use different Stream objects for different CPU threads.

@code
void thread1()
{
    cv::cuda::Stream stream1;
    cv::cuda::func1(..., stream1);
}

void thread2()
{
    cv::cuda::Stream stream2;
    cv::cuda::func2(..., stream2);
}
@endcode

@note By default all CUDA routines are launched in Stream::Null() object, if the stream is not specified by user.
In multi-threading environment the stream objects must be passed explicitly (see previous note).
 */
class CV_EXPORTS Stream
{
    typedef void (Stream::*bool_type)() const;
    void this_type_does_not_support_comparisons() const {}

public:
    typedef void (*StreamCallback)(int status, void* userData);

    //! creates a new asynchronous stream
    Stream();

    //! creates a new asynchronous stream with custom allocator
    Stream(const Ptr<GpuMat::Allocator>& allocator);

    /** @brief Returns true if the current stream queue is finished. Otherwise, it returns false.
    */
    bool queryIfComplete() const;

    /** @brief Blocks the current CPU thread until all operations in the stream are complete.
    */
    void waitForCompletion();

    /** @brief Makes a compute stream wait on an event.
    */
    void waitEvent(const Event& event);

    /** @brief Adds a callback to be called on the host after all currently enqueued items in the stream have
    completed.

    @note Callbacks must not make any CUDA API calls. Callbacks must not perform any synchronization
    that may depend on outstanding device work or other callbacks that are not mandated to run earlier.
    Callbacks without a mandated order (in independent streams) execute in undefined order and may be
    serialized.
     */
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
    friend class DefaultDeviceInitializer;
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
    Event(const Ptr<Impl>& impl);

    friend struct EventAccessor;
};

//! @} cudacore_struct

//===================================================================================
// Initialization & Info
//===================================================================================

//! @addtogroup cudacore_init
//! @{

/** @brief Returns the number of installed CUDA-enabled devices.

Use this function before any other CUDA functions calls. If OpenCV is compiled without CUDA support,
this function returns 0. If the CUDA driver is not installed, or is incompatible, this function
returns -1.
 */
CV_EXPORTS int getCudaEnabledDeviceCount();

/** @brief Sets a device and initializes it for the current thread.

@param device System index of a CUDA device starting with 0.

If the call of this function is omitted, a default device is initialized at the fist CUDA usage.
 */
CV_EXPORTS void setDevice(int device);

/** @brief Returns the current device index set by cuda::setDevice or initialized by default.
 */
CV_EXPORTS int getDevice();

/** @brief Explicitly destroys and cleans up all resources associated with the current device in the current
process.

Any subsequent API call to this device will reinitialize the device.
 */
CV_EXPORTS void resetDevice();

/** @brief Enumeration providing CUDA computing features.
 */
enum FeatureSet
{
    FEATURE_SET_COMPUTE_10 = 10,
    FEATURE_SET_COMPUTE_11 = 11,
    FEATURE_SET_COMPUTE_12 = 12,
    FEATURE_SET_COMPUTE_13 = 13,
    FEATURE_SET_COMPUTE_20 = 20,
    FEATURE_SET_COMPUTE_21 = 21,
    FEATURE_SET_COMPUTE_30 = 30,
    FEATURE_SET_COMPUTE_32 = 32,
    FEATURE_SET_COMPUTE_35 = 35,
    FEATURE_SET_COMPUTE_50 = 50,

    GLOBAL_ATOMICS = FEATURE_SET_COMPUTE_11,
    SHARED_ATOMICS = FEATURE_SET_COMPUTE_12,
    NATIVE_DOUBLE = FEATURE_SET_COMPUTE_13,
    WARP_SHUFFLE_FUNCTIONS = FEATURE_SET_COMPUTE_30,
    DYNAMIC_PARALLELISM = FEATURE_SET_COMPUTE_35
};

//! checks whether current device supports the given feature
CV_EXPORTS bool deviceSupports(FeatureSet feature_set);

/** @brief Class providing a set of static methods to check what NVIDIA\* card architecture the CUDA module was
built for.

According to the CUDA C Programming Guide Version 3.2: "PTX code produced for some specific compute
capability can always be compiled to binary code of greater or equal compute capability".
 */
class CV_EXPORTS TargetArchs
{
public:
    /** @brief The following method checks whether the module was built with the support of the given feature:

    @param feature_set Features to be checked. See :ocvcuda::FeatureSet.
     */
    static bool builtWith(FeatureSet feature_set);

    /** @brief There is a set of methods to check whether the module contains intermediate (PTX) or binary CUDA
    code for the given architecture(s):

    @param major Major compute capability version.
    @param minor Minor compute capability version.
     */
    static bool has(int major, int minor);
    static bool hasPtx(int major, int minor);
    static bool hasBin(int major, int minor);

    static bool hasEqualOrLessPtx(int major, int minor);
    static bool hasEqualOrGreater(int major, int minor);
    static bool hasEqualOrGreaterPtx(int major, int minor);
    static bool hasEqualOrGreaterBin(int major, int minor);
};

/** @brief Class providing functionality for querying the specified GPU properties.
 */
class CV_EXPORTS DeviceInfo
{
public:
    //! creates DeviceInfo object for the current GPU
    DeviceInfo();

    /** @brief The constructors.

    @param device_id System index of the CUDA device starting with 0.

    Constructs the DeviceInfo object for the specified device. If device_id parameter is missed, it
    constructs an object for the current device.
     */
    DeviceInfo(int device_id);

    /** @brief Returns system index of the CUDA device starting with 0.
    */
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
        ComputeModeDefault,         /**< default compute mode (Multiple threads can use cudaSetDevice with this device) */
        ComputeModeExclusive,       /**< compute-exclusive-thread mode (Only one thread in one process will be able to use cudaSetDevice with this device) */
        ComputeModeProhibited,      /**< compute-prohibited mode (No threads can use cudaSetDevice with this device) */
        ComputeModeExclusiveProcess /**< compute-exclusive-process mode (Many threads in one process will be able to use cudaSetDevice with this device) */
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

    /** @brief Provides information on CUDA feature support.

    @param feature_set Features to be checked. See cuda::FeatureSet.

    This function returns true if the device has the specified CUDA feature. Otherwise, it returns false
     */
    bool supports(FeatureSet feature_set) const;

    /** @brief Checks the CUDA module and device compatibility.

    This function returns true if the CUDA module can be run on the specified device. Otherwise, it
    returns false .
     */
    bool isCompatible() const;

private:
    int device_id_;
};

CV_EXPORTS void printCudaDeviceInfo(int device);
CV_EXPORTS void printShortCudaDeviceInfo(int device);

/** @brief Converts an array to half precision floating number.

@param _src input array.
@param _dst output array.
@param stream Stream for the asynchronous version.
@sa convertFp16
*/
CV_EXPORTS void convertFp16(InputArray _src, OutputArray _dst, Stream& stream = Stream::Null());

//! @} cudacore_init

}} // namespace cv { namespace cuda {


#include "opencv2/core/cuda.inl.hpp"

#endif /* OPENCV_CORE_CUDA_HPP */
