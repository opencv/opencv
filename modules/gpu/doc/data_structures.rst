Data Structures
===============

.. highlight:: cpp

.. index:: gpu::DevMem2D\_

gpu::DevMem2D\_
---------------
.. cpp:class:: gpu::DevMem2D\_

This lightweight class encapsulates pitched memory on a GPU and is passed to nvcc-compiled code (CUDA kernels). Typically, it is used internally by OpenCV and by users who write device code. You can call its members from both host and device code. ::

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

        /* returns pointer to the beginning of the given image row */
        __CV_GPU_HOST_DEVICE__ T* ptr(int y = 0);
        __CV_GPU_HOST_DEVICE__ const T* ptr(int y = 0) const;
    };

    typedef DevMem2D_<unsigned char> DevMem2D;
    typedef DevMem2D_<float> DevMem2Df;
    typedef DevMem2D_<int> DevMem2Di;
..


.. index:: gpu::PtrStep\_

gpu::PtrStep\_
--------------
.. cpp:class:: gpu::PtrStep\_

This structure is similar to 
:cpp:class:`DevMem2D_` but contains only a pointer and row step. Width and height fields are excluded due to performance reasons. The structure is intended for internal use or for users who write device code. 
::

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


.. index:: gpu::PtrElemStrp\_

gpu::PtrElemStrp\_
------------------
.. cpp:class:: gpu::PtrElemStrp\_

This structure is similar to 
:cpp:class:`DevMem2D_` but contains only pointer and row step in elements. Width and height fields are excluded due to performance reasons. This class can only be constructed if ``sizeof(T)`` is a multiple of 256. The structure is intended for internal use or for users who write device code. 
::

    template<typename T> struct PtrElemStep_ : public PtrStep_<T>
    {
            PtrElemStep_(const DevMem2D_<T>& mem);
            __CV_GPU_HOST_DEVICE__ T* ptr(int y = 0);
            __CV_GPU_HOST_DEVICE__ const T* ptr(int y = 0) const;
    };


.. index:: gpu::GpuMat

gpu::GpuMat
-----------
.. cpp:class:: gpu::GpuMat

This is a base storage class for GPU memory with reference counting. Its interface matches the
:c:type:`Mat` interface with the following limitations:

*   
    no arbitrary dimensions support (only 2D)
*   
    no functions that return references to their data (because references on GPU are not valid for CPU)
*   
    no expression templates technique support
    
Beware that the latter limitation may lead to overloaded matrix operators that cause memory allocations. The ``GpuMat`` class is convertible to :cpp:class:`gpu::DevMem2D_` and :cpp:class:`gpu::PtrStep_` so it can be passed directly to kernel.

**Note:**

In contrast with :c:type:`Mat`, in most cases ``GpuMat::isContinuous() == false`` . This means that rows are aligned to size depending on the hardware. Single-row ``GpuMat`` is always a continuous matrix. ::

    class CV_EXPORTS GpuMat
    {
    public:
            //! default constructor
            GpuMat();

            GpuMat(int rows, int cols, int type);
            GpuMat(Size size, int type);

            .....

            //! builds GpuMat from Mat. Blocks uploading to device.
            explicit GpuMat (const Mat& m);

            //! returns lightweight DevMem2D_ structure for passing
            //to nvcc-compiled code. Contains size, data ptr and step.
            template <class T> operator DevMem2D_<T>() const;
            template <class T> operator PtrStep_<T>() const;

            //! blocks uploading data to GpuMat.
            void upload(const cv::Mat& m);
            void upload(const CudaMem& m, Stream& stream);

            //! downloads data from device to host memory. Blocking calls.
            operator Mat() const;
            void download(cv::Mat& m) const;

            //! download async
            void download(CudaMem& m, Stream& stream) const;
    };


**Note:**

You are not recommended to leave static or global ``GpuMat`` variables allocated, that is to rely on its destructor. The destruction order of such variables and CUDA context is undefined. GPU memory release function returns error if the CUDA context has been destroyed before.

See Also:
:cpp:func:`Mat`

.. index:: gpu::CudaMem

gpu::CudaMem
------------
.. cpp:class:: gpu::CudaMem

This class with reference counting wraps special memory type allocation functions from CUDA. Its interface is also
:cpp:func:`Mat`-like but with additional memory type parameters.
    
*
    ``ALLOC_PAGE_LOCKED``:  Sets a page locked memory type, used commonly for fast and asynchronous uploading/downloading data from/to GPU.
*
    ``ALLOC_ZEROCOPY``:  Specifies a zero copy memory allocation that enables mapping the host memory to GPU address space, if supported.
*
    ``ALLOC_WRITE_COMBINED``:  Sets the write combined buffer that is not cached by CPU. Such buffers are used to supply GPU with data when GPU only reads it. The advantage is a better CPU cache utilization.

**Note:**

Allocation size of such memory types is usually limited. For more details, see "CUDA 2.2 Pinned Memory APIs" document or "CUDA C Programming Guide".
::

    class CV_EXPORTS CudaMem
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

            //if host memory can be mapped to gpu address space;
            static bool canMapHostMemory();

            int alloc_type;
    };


.. index:: gpu::CudaMem::createMatHeader

gpu::CudaMem::createMatHeader
---------------------------------

.. cpp:function:: Mat gpu::CudaMem::createMatHeader() const

    Creates a header without reference counting to :cpp:class:`gpu::CudaMem` data.

.. index:: gpu::CudaMem::createGpuMatHeader

gpu::CudaMem::createGpuMatHeader
------------------------------------

.. cpp:function:: GpuMat gpu::CudaMem::createGpuMatHeader() const

    Maps CPU memory to GPU address space and creates the :cpp:class:`gpu::GpuMat` header without reference counting for it. This can be done only if memory was allocated with the ``ALLOC_ZEROCOPY`` flag and if it is supported by the hardware (laptops often share video and CPU memory, so address spaces can be mapped, which eliminates an extra copy).

.. index:: gpu::CudaMem::canMapHostMemory

gpu::CudaMem::canMapHostMemory
----------------------------------
.. cpp:function:: static bool gpu::CudaMem::canMapHostMemory()

    Returns ``true`` if the current hardware supports address space mapping and ``ALLOC_ZEROCOPY`` memory allocation.

.. index:: gpu::Stream

gpu::Stream
-----------
.. cpp:class:: gpu::Stream

This class encapsulates a queue of asynchronous calls. Some functions have overloads with the additional ``gpu::Stream`` parameter. The overloads do initialization work (allocate output buffers, upload constants, and so on), start the GPU kernel, and return before results are ready. You can check whether all operations are complete via :cpp:func:`gpu::Stream::queryIfComplete`. You can asynchronously upload/download data from/to page-locked buffers, using the :cpp:class:`gpu::CudaMem` or :c:type:`Mat` header that points to a region of :cpp:class:`gpu::CudaMem`.

**Note:**

Currently, you may face problems if an operation is enqueued twice with different data. Some functions use the constant GPU memory, and next call may update the memory before the previous one has been finished. But calling different operations asynchronously is safe because each operation has its own constant buffer. Memory copy/upload/download/set operations to the buffers you hold are also safe. 
::

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

    Returns ``true`` if the current stream queue is finished. Otherwise, it returns false.

.. index:: gpu::Stream::waitForCompletion

gpu::Stream::waitForCompletion
----------------------------------
.. cpp:function:: void gpu::Stream::waitForCompletion()

    Blocks the current CPU thread until all operations in the stream are complete.

.. index:: gpu::StreamAccessor

gpu::StreamAccessor
-------------------
.. cpp:class:: gpu::StreamAccessor

This class enables getting ``cudaStream_t`` from :cpp:class:`gpu::Stream` and is declared in ``stream_accessor.hpp`` because it is the only public header that depends on the CUDA Runtime API. Including it brings a dependency to your code. 
::

    struct StreamAccessor
    {
        CV_EXPORTS static cudaStream_t getStream(const Stream& stream);
    };


.. index:: gpu::createContinuous

gpu::createContinuous
-------------------------
.. cpp:function:: void gpu::createContinuous(int rows, int cols, int type, GpuMat& m)

    Creates a continuous matrix in the GPU memory.

    :param rows: Row count.

    :param cols: Column count.

    :param type: Type of the matrix.

    :param m: Destination matrix. This parameter changes only if it has a proper type and area (``rows x cols``).

    The following wrappers are also available:
    
    
		* .. cpp:function:: GpuMat gpu::createContinuous(int rows, int cols, int type)
    
		* .. cpp:function:: void gpu::createContinuous(Size size, int type, GpuMat& m)
    
		* .. cpp:function:: GpuMat gpu::createContinuous(Size size, int type)

    Matrix is called continuous if its elements are stored continuously, that is without gaps in the end of each row.

.. index:: gpu::ensureSizeIsEnough

gpu::ensureSizeIsEnough
---------------------------
.. cpp:function:: void gpu::ensureSizeIsEnough(int rows, int cols, int type, GpuMat& m)

.. cpp:function:: void gpu::ensureSizeIsEnough(Size size, int type, GpuMat& m)

    Ensures that the size of a matrix is big enough and the matrix has a proper type. The function does not reallocate memory if the matrix has proper attributes already.

    :param rows: Minimum desired number of rows.

    :param cols: Minimum desired number of columns.
    
    :param size: Rows and coumns passed as a structure.

    :param type: Desired matrix type.

    :param m: Destination matrix.    

