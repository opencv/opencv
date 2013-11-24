Data Structures
===============

.. highlight:: cpp



cuda::PtrStepSz
---------------
.. ocv:class:: cuda::PtrStepSz

Lightweight class encapsulating pitched memory on a GPU and passed to nvcc-compiled code (CUDA kernels). Typically, it is used internally by OpenCV and by users who write device code. You can call its members from both host and device code. ::

    template <typename T> struct PtrStepSz : public PtrStep<T>
    {
        __CV_GPU_HOST_DEVICE__ PtrStepSz() : cols(0), rows(0) {}
        __CV_GPU_HOST_DEVICE__ PtrStepSz(int rows_, int cols_, T* data_, size_t step_)
            : PtrStep<T>(data_, step_), cols(cols_), rows(rows_) {}

        template <typename U>
        explicit PtrStepSz(const PtrStepSz<U>& d) : PtrStep<T>((T*)d.data, d.step), cols(d.cols), rows(d.rows){}

        int cols;
        int rows;
    };

    typedef PtrStepSz<unsigned char> PtrStepSzb;
    typedef PtrStepSz<float> PtrStepSzf;
    typedef PtrStepSz<int> PtrStepSzi;



cuda::PtrStep
-------------
.. ocv:class:: cuda::PtrStep

Structure similar to :ocv:class:`cuda::PtrStepSz` but containing only a pointer and row step. Width and height fields are excluded due to performance reasons. The structure is intended for internal use or for users who write device code. ::

    template <typename T> struct PtrStep : public DevPtr<T>
    {
        __CV_GPU_HOST_DEVICE__ PtrStep() : step(0) {}
        __CV_GPU_HOST_DEVICE__ PtrStep(T* data_, size_t step_) : DevPtr<T>(data_), step(step_) {}

        //! stride between two consecutive rows in bytes. Step is stored always and everywhere in bytes!!!
        size_t step;

        __CV_GPU_HOST_DEVICE__       T* ptr(int y = 0)       { return (      T*)( (      char*)DevPtr<T>::data + y * step); }
        __CV_GPU_HOST_DEVICE__ const T* ptr(int y = 0) const { return (const T*)( (const char*)DevPtr<T>::data + y * step); }

        __CV_GPU_HOST_DEVICE__       T& operator ()(int y, int x)       { return ptr(y)[x]; }
        __CV_GPU_HOST_DEVICE__ const T& operator ()(int y, int x) const { return ptr(y)[x]; }
    };

    typedef PtrStep<unsigned char> PtrStepb;
    typedef PtrStep<float> PtrStepf;
    typedef PtrStep<int> PtrStepi;



cuda::GpuMat
------------
.. ocv:class:: cuda::GpuMat

Base storage class for GPU memory with reference counting. Its interface matches the :ocv:class:`Mat` interface with the following limitations:

* no arbitrary dimensions support (only 2D)
* no functions that return references to their data (because references on GPU are not valid for CPU)
* no expression templates technique support

Beware that the latter limitation may lead to overloaded matrix operators that cause memory allocations. The ``GpuMat`` class is convertible to :ocv:class:`cuda::PtrStepSz` and :ocv:class:`cuda::PtrStep` so it can be passed directly to the kernel.

.. note:: In contrast with :ocv:class:`Mat`, in most cases ``GpuMat::isContinuous() == false`` . This means that rows are aligned to a size depending on the hardware. Single-row ``GpuMat`` is always a continuous matrix.

::

    class CV_EXPORTS GpuMat
    {
    public:
        //! default constructor
        GpuMat();

        //! constructs GpuMat of the specified size and type
        GpuMat(int rows, int cols, int type);
        GpuMat(Size size, int type);

        .....

        //! builds GpuMat from host memory (Blocking call)
        explicit GpuMat(InputArray arr);

        //! returns lightweight PtrStepSz structure for passing
        //to nvcc-compiled code. Contains size, data ptr and step.
        template <class T> operator PtrStepSz<T>() const;
        template <class T> operator PtrStep<T>() const;

        //! pefroms upload data to GpuMat (Blocking call)
        void upload(InputArray arr);

        //! pefroms upload data to GpuMat (Non-Blocking call)
        void upload(InputArray arr, Stream& stream);

        //! pefroms download data from device to host memory (Blocking call)
        void download(OutputArray dst) const;

        //! pefroms download data from device to host memory (Non-Blocking call)
        void download(OutputArray dst, Stream& stream) const;
    };


.. note:: You are not recommended to leave static or global ``GpuMat`` variables allocated, that is, to rely on its destructor. The destruction order of such variables and CUDA context is undefined. GPU memory release function returns error if the CUDA context has been destroyed before.

.. seealso:: :ocv:class:`Mat`



cuda::createContinuous
----------------------
Creates a continuous matrix.

.. ocv:function:: void cuda::createContinuous(int rows, int cols, int type, OutputArray arr)

    :param rows: Row count.

    :param cols: Column count.

    :param type: Type of the matrix.

    :param arr: Destination matrix. This parameter changes only if it has a proper type and area ( :math:`\texttt{rows} \times \texttt{cols}` ).

Matrix is called continuous if its elements are stored continuously, that is, without gaps at the end of each row.



cuda::ensureSizeIsEnough
------------------------
Ensures that the size of a matrix is big enough and the matrix has a proper type.

.. ocv:function:: void cuda::ensureSizeIsEnough(int rows, int cols, int type, OutputArray arr)

    :param rows: Minimum desired number of rows.

    :param cols: Minimum desired number of columns.

    :param type: Desired matrix type.

    :param arr: Destination matrix.

The function does not reallocate memory if the matrix has proper attributes already.



cuda::CudaMem
-------------
.. ocv:class:: cuda::CudaMem

Class with reference counting wrapping special memory type allocation functions from CUDA. Its interface is also :ocv:func:`Mat`-like but with additional memory type parameters.

* **PAGE_LOCKED** sets a page locked memory type used commonly for fast and asynchronous uploading/downloading data from/to GPU.
* **SHARED** specifies a zero copy memory allocation that enables mapping the host memory to GPU address space, if supported.
* **WRITE_COMBINED**  sets the write combined buffer that is not cached by CPU. Such buffers are used to supply GPU with data when GPU only reads it. The advantage is a better CPU cache utilization.

.. note:: Allocation size of such memory types is usually limited. For more details, see *CUDA 2.2 Pinned Memory APIs* document or *CUDA C Programming Guide*.

::

    class CV_EXPORTS CudaMem
    {
    public:
        enum AllocType { PAGE_LOCKED = 1, SHARED = 2, WRITE_COMBINED = 4 };

        explicit CudaMem(AllocType alloc_type = PAGE_LOCKED);

        CudaMem(int rows, int cols, int type, AllocType alloc_type = PAGE_LOCKED);
        CudaMem(Size size, int type, AllocType alloc_type = PAGE_LOCKED);

        //! creates from host memory with coping data
        explicit CudaMem(InputArray arr, AllocType alloc_type = PAGE_LOCKED);

        ......

        //! returns matrix header with disabled reference counting for CudaMem data.
        Mat createMatHeader() const;

        //! maps host memory into device address space and returns GpuMat header for it. Throws exception if not supported by hardware.
        GpuMat createGpuMatHeader() const;

        ......

        AllocType alloc_type;
    };



cuda::CudaMem::createMatHeader
------------------------------
Creates a header without reference counting to :ocv:class:`cuda::CudaMem` data.

.. ocv:function:: Mat cuda::CudaMem::createMatHeader() const



cuda::CudaMem::createGpuMatHeader
---------------------------------
Maps CPU memory to GPU address space and creates the :ocv:class:`cuda::GpuMat` header without reference counting for it.

.. ocv:function:: GpuMat cuda::CudaMem::createGpuMatHeader() const

This can be done only if memory was allocated with the ``SHARED`` flag and if it is supported by the hardware. Laptops often share video and CPU memory, so address spaces can be mapped, which eliminates an extra copy.



cuda::registerPageLocked
------------------------
Page-locks the memory of matrix and maps it for the device(s).

.. ocv:function:: void cuda::registerPageLocked(Mat& m)

    :param m: Input matrix.



cuda::unregisterPageLocked
--------------------------
Unmaps the memory of matrix and makes it pageable again.

.. ocv:function:: void cuda::unregisterPageLocked(Mat& m)

    :param m: Input matrix.



cuda::Stream
------------
.. ocv:class:: cuda::Stream

This class encapsulates a queue of asynchronous calls.

.. note:: Currently, you may face problems if an operation is enqueued twice with different data. Some functions use the constant GPU memory, and next call may update the memory before the previous one has been finished. But calling different operations asynchronously is safe because each operation has its own constant buffer. Memory copy/upload/download/set operations to the buffers you hold are also safe.

::

    class CV_EXPORTS Stream
    {
    public:
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
    };



cuda::Stream::queryIfComplete
-----------------------------
Returns ``true`` if the current stream queue is finished. Otherwise, it returns false.

.. ocv:function:: bool cuda::Stream::queryIfComplete()



cuda::Stream::waitForCompletion
-------------------------------
Blocks the current CPU thread until all operations in the stream are complete.

.. ocv:function:: void cuda::Stream::waitForCompletion()



cuda::Stream::waitEvent
-----------------------
Makes a compute stream wait on an event.

.. ocv:function:: void cuda::Stream::waitEvent(const Event& event)



cuda::Stream::enqueueHostCallback
---------------------------------
Adds a callback to be called on the host after all currently enqueued items in the stream have completed.

.. ocv:function:: void cuda::Stream::enqueueHostCallback(StreamCallback callback, void* userData)

.. note:: Callbacks must not make any CUDA API calls. Callbacks must not perform any synchronization that may depend on outstanding device work or other callbacks that are not mandated to run earlier.  Callbacks without a mandated order (in independent streams) execute in undefined order and may be serialized.



cuda::StreamAccessor
--------------------
.. ocv:struct:: cuda::StreamAccessor

Class that enables getting ``cudaStream_t`` from :ocv:class:`cuda::Stream` and is declared in ``stream_accessor.hpp`` because it is the only public header that depends on the CUDA Runtime API. Including it brings a dependency to your code. ::

    struct StreamAccessor
    {
        CV_EXPORTS static cudaStream_t getStream(const Stream& stream);
    };
