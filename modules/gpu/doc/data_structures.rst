Data Structures
===============

.. highlight:: cpp



.. index:: gpu::DevMem2D_

gpu::DevMem2D\_ 
---------------
.. cpp:class:: gpu::DevMem2D_

This is a simple lightweight class that encapsulate pitched memory on GPU. It is intended to pass to nvcc-compiled code, i.e. CUDA kernels. So it is used internally by OpenCV and by users writes own device code. Its members can be called both from host and from device code. ::

    template <typename T> struct DevMem2D_
    {
        int cols;
        int rows;
        T* data;
        size_t step;

        DevMem2D_() : cols(0), rows(0), data(0), step(0){};
        DevMem2D_(int rows, int cols, T *data, size_t step);

        template <typename U>
        explicit DevMem2D_(const DevMem2D_<U>& d);

        typedef T elem_type;
        enum { elem_size = sizeof(elem_type) };

        __CV_GPU_HOST_DEVICE__ size_t elemSize() const;

        /* returns pointer to the beggining of given image row */
        __CV_GPU_HOST_DEVICE__ T* ptr(int y = 0);
        __CV_GPU_HOST_DEVICE__ const T* ptr(int y = 0) const;
    };

    typedef DevMem2D_<unsigned char> DevMem2D;
    typedef DevMem2D_<float> DevMem2Df;
    typedef DevMem2D_<int> DevMem2Di;



.. index:: gpu::PtrStep_

gpu::PtrStep\_
--------------
.. cpp:class:: gpu::PtrStep_

This is structure is similar to :cpp:class:`gpu::DevMem2D_` but contains only pointer and row step. Width and height fields are excluded due to performance reasons. The structure is for internal use or for users who write own device code. ::

    template<typename T> struct PtrStep_
    {
        T* data;
        size_t step;

        PtrStep_();
        PtrStep_(const DevMem2D_<T>& mem);

        typedef T elem_type;
        enum { elem_size = sizeof(elem_type) };

        __CV_GPU_HOST_DEVICE__ size_t elemSize() const;
        __CV_GPU_HOST_DEVICE__ T* ptr(int y = 0);
        __CV_GPU_HOST_DEVICE__ const T* ptr(int y = 0) const;
    };

    typedef PtrStep_<unsigned char> PtrStep;
    typedef PtrStep_<float> PtrStepf;
    typedef PtrStep_<int> PtrStepi;



.. index:: gpu::PtrElemStep_

gpu::PtrElemStep\_
------------------
.. cpp:class:: gpu::PtrElemStep_

This is structure is similar to :cpp:class:`gpu::DevMem2D_` but contains only pointer and row step in elements. Width and height fields are excluded due to performance reasons. This class is can only be constructed if ``sizeof(T)`` is a multiple of 256. The structure is for internal use or for users who write own device code. ::

    template<typename T> struct PtrElemStep_ : public PtrStep_<T>
    {
        PtrElemStep_(const DevMem2D_<T>& mem);
        __CV_GPU_HOST_DEVICE__ T* ptr(int y = 0);
        __CV_GPU_HOST_DEVICE__ const T* ptr(int y = 0) const;
    };

    typedef PtrElemStep_<unsigned char> PtrElemStep;
    typedef PtrElemStep_<float> PtrElemStepf;
    typedef PtrElemStep_<int> PtrElemStepi;



.. index:: gpu::GpuMat

gpu::GpuMat
-----------
.. cpp:class:: gpu::GpuMat

The base storage class for GPU memory with reference counting. Its interface is almost :c:type:`Mat` interface with some limitations, so using it won't be a problem. The limitations are no arbitrary dimensions support (only 2D), no functions that returns references to its data (because references on GPU are not valid for CPU), no expression templates technique support. Because of last limitation please take care with overloaded matrix operators - they cause memory allocations. The ``GpuMat`` class is convertible to :cpp:class:`gpu::DevMem2D_` and :cpp:class:`gpu::PtrStep_` so it can be passed to directly to kernel.

**Please note:** In contrast with :c:type:`Mat`, in most cases ``GpuMat::isContinuous() == false`` , i.e. rows are aligned to size depending on hardware. Also single row ``GpuMat`` is always a continuous matrix. ::

    class GpuMat
    {
    public:
        //! default constructor
        GpuMat();

        GpuMat(int rows, int cols, int type);
        GpuMat(Size size, int type);

        .....

        //! builds GpuMat from Mat. Perfom blocking upload to device.
        explicit GpuMat (const Mat& m);

        //! returns lightweight DevMem2D_ structure for passing
        //to nvcc-compiled code. Contains size, data ptr and step.
        template <class T> operator DevMem2D_<T>() const;
        template <class T> operator PtrStep_<T>() const;

        //! pefroms blocking upload data to GpuMat.
        void upload(const cv::Mat& m);
        void upload(const CudaMem& m, Stream& stream);

        //! downloads data from device to host memory. Blocking calls.
        operator Mat() const;
        void download(cv::Mat& m) const;

        //! download async
        void download(CudaMem& m, Stream& stream) const;
    };


**Please note:** Is it a bad practice to leave static or global ``GpuMat`` variables allocated, i.e. to rely on its destructor. That is because destruction order of such variables and CUDA context is undefined and GPU memory release function returns error if CUDA context has been destroyed before.

See also: :c:type:`Mat`.



.. index:: gpu::CudaMem

gpu::CudaMem
------------
.. cpp:class:: gpu::CudaMem

This is a class with reference counting that wraps special memory type allocation functions from CUDA. Its interface is also :c:type:`Mat`-like but with additional memory type parameter:

* ``ALLOC_PAGE_LOCKED``     Set page locked memory type, used commonly for fast and asynchronous upload/download data from/to GPU.

* ``ALLOC_ZEROCOPY``        Specifies zero copy memory allocation, i.e. with possibility to map host memory to GPU address space if supported.

* ``ALLOC_WRITE_COMBINED``  Sets write combined buffer which is not cached by CPU. Such buffers are used to supply GPU with data when GPU only reads it. The advantage is better CPU cache utilization.

**Please note:** Allocation size of such memory types is usually limited. For more details please see "CUDA 2.2 Pinned Memory APIs" document or "CUDA_C Programming Guide". ::

    class CudaMem
    {
    public:
        enum  { ALLOC_PAGE_LOCKED = 1, ALLOC_ZEROCOPY = 2,
                 ALLOC_WRITE_COMBINED = 4 };

        CudaMem(Size size, int type, int alloc_type = ALLOC_PAGE_LOCKED);

        //! creates from cv::Mat with coping data
        explicit CudaMem(const Mat& m, int alloc_type = ALLOC_PAGE_LOCKED);

         ......

        void create(Size size, int type, int alloc_type = ALLOC_PAGE_LOCKED);

        //! returns matrix header with disabled ref. counting for CudaMem data.
        Mat createMatHeader() const;
        operator Mat() const;

        //! maps host memory into device address space
        GpuMat createGpuMatHeader() const;
        operator GpuMat() const;

        //if host memory can be mapperd to gpu address space;
        static bool canMapHostMemory();

        int alloc_type;
    };



.. index:: gpu::CudaMem::createMatHeader

gpu::CudaMem::createMatHeader
---------------------------------

.. cpp:function:: Mat gpu::CudaMem::createMatHeader() const

.. cpp:function:: gpu::CudaMem::operator Mat() const

    Creates header without reference counting to :cpp:class:`gpu::CudaMem` data.



.. index:: gpu::CudaMem::createGpuMatHeader

gpu::CudaMem::createGpuMatHeader
------------------------------------

.. cpp:function:: GpuMat gpu::CudaMem::createGpuMatHeader() const

.. cpp:function:: gpu::CudaMem::operator GpuMat() const

    Maps CPU memory to GPU address space and creates :cpp:class:`gpu::GpuMat` header without reference counting for it. This can be done only if memory was allocated with ``ALLOC_ZEROCOPY`` flag and if it is supported by hardware (laptops often share video and CPU memory, so address spaces can be mapped, and that eliminates extra copy).



.. index:: gpu::CudaMem::canMapHostMemory

gpu::CudaMem::canMapHostMemory
----------------------------------
.. cpp:function:: static bool gpu::CudaMem::canMapHostMemory()

    Returns true if the current hardware supports address space mapping and ``ALLOC_ZEROCOPY`` memory allocation.



.. index:: gpu::Stream

gpu::Stream
-----------
.. cpp:class:: gpu::Stream

This class encapsulated queue of the asynchronous calls. Some functions have overloads with additional ``gpu::Stream`` parameter. The overloads do initialization work (allocate output buffers, upload constants, etc.), start GPU kernel and return before results are ready. A check if all operation are complete can be performed via :cpp:func:`gpu::Stream::queryIfComplete`. Asynchronous upload/download have to be performed from/to page-locked buffers, i.e. using :cpp:class:`gpu::CudaMem` or :c:type:`Mat` header that points to a region of :cpp:class:`gpu::CudaMem`.

**Please note the limitation**: currently it is not guaranteed that all will work properly if one operation will be enqueued twice with different data. Some functions use constant GPU memory and next call may update the memory before previous has been finished. But calling asynchronously different operations is safe because each operation has own constant buffer. Memory copy/upload/download/set operations to buffers hold by user are also safe. ::

    class Stream
    {
    public:
        Stream();
        ~Stream();

        Stream(const Stream&);
        Stream& operator=(const Stream&);

        bool queryIfComplete();
        void waitForCompletion();

        //! downloads asynchronously.
        // Warning! cv::Mat must point to page locked memory
                 (i.e. to CudaMem data or to its subMat)
        void enqueueDownload(const GpuMat& src, CudaMem& dst);
        void enqueueDownload(const GpuMat& src, Mat& dst);

        //! uploads asynchronously.
        // Warning! cv::Mat must point to page locked memory
                 (i.e. to CudaMem data or to its ROI)
        void enqueueUpload(const CudaMem& src, GpuMat& dst);
        void enqueueUpload(const Mat& src, GpuMat& dst);

        void enqueueCopy(const GpuMat& src, GpuMat& dst);

        void enqueueMemSet(const GpuMat& src, Scalar val);
        void enqueueMemSet(const GpuMat& src, Scalar val, const GpuMat& mask);

        // converts matrix type, ex from float to uchar depending on type
        void enqueueConvert(const GpuMat& src, GpuMat& dst, int type,
                double a = 1, double b = 0);
    };



.. index:: gpu::Stream::queryIfComplete

gpu::Stream::queryIfComplete
--------------------------------
.. cpp:function:: bool gpu::Stream::queryIfComplete()

    Returns true if the current stream queue is finished, otherwise false.



.. index:: gpu::Stream::waitForCompletion

gpu::Stream::waitForCompletion
----------------------------------
.. cpp:function:: void gpu::Stream::waitForCompletion()

    Blocks until all operations in the stream are complete.



.. index:: gpu::StreamAccessor

gpu::StreamAccessor
-------------------
.. cpp:class:: gpu::StreamAccessor

This class provides possibility to get ``cudaStream_t`` from :cpp:class:`gpu::Stream`. This class is declared in ``stream_accessor.hpp`` because that is only public header that depend on Cuda Runtime API. Including it will bring the dependency to your code. ::

    struct StreamAccessor
    {
        static cudaStream_t getStream(const Stream& stream);
    };



.. index:: gpu::createContinuous

gpu::createContinuous
-------------------------
.. cpp:function:: void gpu::createContinuous(int rows, int cols, int type, GpuMat& m)

    Creates continuous matrix in GPU memory.

    :param rows: Row count.

    :param cols: Column count.

    :param type: Type of the matrix.

    :param m: Destination matrix. Will be only reshaped if it has proper type and area (``rows`` :math:`\times` ``cols``).

Also the following wrappers are available:

.. cpp:function:: GpuMat gpu::createContinuous(int rows, int cols, int type)

.. cpp:function:: void gpu::createContinuous(Size size, int type, GpuMat& m)

.. cpp:function:: GpuMat gpu::createContinuous(Size size, int type)

Matrix is called continuous if its elements are stored continuously, i.e. wuthout gaps in the end of each row.



.. index:: gpu::ensureSizeIsEnough

gpu::ensureSizeIsEnough
---------------------------
.. cpp:function:: void gpu::ensureSizeIsEnough(int rows, int cols, int type, GpuMat& m)

    Ensures that size of matrix is big enough and matrix has proper type. The function doesn't reallocate memory if the matrix has proper attributes already.

    :param rows: Minimum desired number of rows.

    :param cols: Minimum desired number of cols.

    :param type: Desired matrix type.

    :param m: Destination matrix.

Also the following wrapper is available:

.. cpp:function:: void gpu::ensureSizeIsEnough(Size size, int type, GpuMat& m)
