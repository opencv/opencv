Initalization and Information
=============================

.. highlight:: cpp



cuda::getCudaEnabledDeviceCount
-------------------------------
Returns the number of installed CUDA-enabled devices.

.. ocv:function:: int cuda::getCudaEnabledDeviceCount()

Use this function before any other CUDA functions calls. If OpenCV is compiled without CUDA support, this function returns 0.



cuda::setDevice
---------------
Sets a device and initializes it for the current thread.

.. ocv:function:: void cuda::setDevice(int device)

    :param device: System index of a CUDA device starting with 0.

If the call of this function is omitted, a default device is initialized at the fist CUDA usage.



cuda::getDevice
---------------
Returns the current device index set by :ocv:func:`cuda::setDevice` or initialized by default.

.. ocv:function:: int cuda::getDevice()



cuda::resetDevice
-----------------
Explicitly destroys and cleans up all resources associated with the current device in the current process.

.. ocv:function:: void cuda::resetDevice()

Any subsequent API call to this device will reinitialize the device.



cuda::FeatureSet
----------------
Enumeration providing CUDA computing features.

.. ocv:enum:: cuda::FeatureSet

  .. ocv:emember:: FEATURE_SET_COMPUTE_10
  .. ocv:emember:: FEATURE_SET_COMPUTE_11
  .. ocv:emember:: FEATURE_SET_COMPUTE_12
  .. ocv:emember:: FEATURE_SET_COMPUTE_13
  .. ocv:emember:: FEATURE_SET_COMPUTE_20
  .. ocv:emember:: FEATURE_SET_COMPUTE_21
  .. ocv:emember:: GLOBAL_ATOMICS
  .. ocv:emember:: SHARED_ATOMICS
  .. ocv:emember:: NATIVE_DOUBLE



cuda::TargetArchs
-----------------
.. ocv:class:: cuda::TargetArchs

Class providing a set of static methods to check what NVIDIA* card architecture the CUDA module was built for.

The following method checks whether the module was built with the support of the given feature:

    .. ocv:function:: static bool cuda::TargetArchs::builtWith( FeatureSet feature_set )

        :param feature_set: Features to be checked. See :ocv:enum:`cuda::FeatureSet`.

There is a set of methods to check whether the module contains intermediate (PTX) or binary CUDA code for the given architecture(s):

    .. ocv:function:: static bool cuda::TargetArchs::has(int major, int minor)

    .. ocv:function:: static bool cuda::TargetArchs::hasPtx(int major, int minor)

    .. ocv:function:: static bool cuda::TargetArchs::hasBin(int major, int minor)

    .. ocv:function:: static bool cuda::TargetArchs::hasEqualOrLessPtx(int major, int minor)

    .. ocv:function:: static bool cuda::TargetArchs::hasEqualOrGreater(int major, int minor)

    .. ocv:function:: static bool cuda::TargetArchs::hasEqualOrGreaterPtx(int major, int minor)

    .. ocv:function:: static bool cuda::TargetArchs::hasEqualOrGreaterBin(int major, int minor)

        :param major: Major compute capability version.

        :param minor: Minor compute capability version.

According to the CUDA C Programming Guide Version 3.2: "PTX code produced for some specific compute capability can always be compiled to binary code of greater or equal compute capability".



cuda::DeviceInfo
----------------
.. ocv:class:: cuda::DeviceInfo

Class providing functionality for querying the specified GPU properties. ::

    class CV_EXPORTS DeviceInfo
    {
    public:
        //! creates DeviceInfo object for the current GPU
        DeviceInfo();

        //! creates DeviceInfo object for the given GPU
        DeviceInfo(int device_id);

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
    };



cuda::DeviceInfo::DeviceInfo
----------------------------
The constructors.

.. ocv:function:: cuda::DeviceInfo::DeviceInfo()

.. ocv:function:: cuda::DeviceInfo::DeviceInfo(int device_id)

    :param device_id: System index of the CUDA device starting with 0.

Constructs the ``DeviceInfo`` object for the specified device. If ``device_id`` parameter is missed, it constructs an object for the current device.



cuda::DeviceInfo::name
----------------------
Returns the device name.

.. ocv:function:: const char* cuda::DeviceInfo::name() const



cuda::DeviceInfo::majorVersion
------------------------------
Returns the major compute capability version.

.. ocv:function:: int cuda::DeviceInfo::majorVersion()



cuda::DeviceInfo::minorVersion
------------------------------
Returns the minor compute capability version.

.. ocv:function:: int cuda::DeviceInfo::minorVersion()



cuda::DeviceInfo::freeMemory
----------------------------
Returns the amount of free memory in bytes.

.. ocv:function:: size_t cuda::DeviceInfo::freeMemory()



cuda::DeviceInfo::totalMemory
-----------------------------
Returns the amount of total memory in bytes.

.. ocv:function:: size_t cuda::DeviceInfo::totalMemory()



cuda::DeviceInfo::supports
--------------------------
Provides information on CUDA feature support.

.. ocv:function:: bool cuda::DeviceInfo::supports(FeatureSet feature_set) const

    :param feature_set: Features to be checked. See :ocv:enum:`cuda::FeatureSet`.

This function returns ``true`` if the device has the specified CUDA feature. Otherwise, it returns ``false`` .



cuda::DeviceInfo::isCompatible
------------------------------
Checks the CUDA module and device compatibility.

.. ocv:function:: bool cuda::DeviceInfo::isCompatible()

This function returns ``true`` if the CUDA module can be run on the specified device. Otherwise, it returns ``false`` .



cuda::DeviceInfo::deviceID
--------------------------
Returns system index of the CUDA device starting with 0.

.. ocv:function:: int cuda::DeviceInfo::deviceID()
