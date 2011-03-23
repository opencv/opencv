Initalization and Information
=============================

.. highlight:: cpp



.. index:: gpu::getCudaEnabledDeviceCount

gpu::getCudaEnabledDeviceCount
----------------------------------
.. cpp:function:: int gpu::getCudaEnabledDeviceCount()

    Returns number of CUDA-enabled devices installed. It is to be used before any other GPU functions calls. If OpenCV is compiled without GPU support this function returns 0.



.. index:: gpu::setDevice

gpu::setDevice
------------------
.. cpp:function:: void gpu::setDevice(int device)

    Sets device and initializes it for the current thread. Call of this function can be omitted, but in this case a default device will be initialized on fist GPU usage.

    :param device: index of GPU device in system starting with 0.



.. index:: gpu::getDevice

gpu::getDevice
------------------
.. cpp:function:: int gpu::getDevice()

    Returns the current device index, which was set by :cpp:func:`gpu::setDevice` or initialized by default.



.. index:: gpu::GpuFeature

gpu::GpuFeature
---------------
.. c:type:: gpu::GpuFeature

GPU compute features. ::

    enum GpuFeature
    {
        COMPUTE_10, COMPUTE_11,
        COMPUTE_12, COMPUTE_13,
        COMPUTE_20, COMPUTE_21,
        ATOMICS, NATIVE_DOUBLE
    };



.. index:: gpu::DeviceInfo

gpu::DeviceInfo
---------------
.. cpp:class:: gpu::DeviceInfo

This class provides functionality for querying the specified GPU properties. ::

    class DeviceInfo
    {
    public:
        DeviceInfo();
        DeviceInfo(int device_id);

        string name() const;

        int majorVersion() const;
        int minorVersion() const;

        int multiProcessorCount() const;

        size_t freeMemory() const;
        size_t totalMemory() const;

        bool supports(GpuFeature feature) const;
        bool isCompatible() const;
    };



.. index:: gpu::DeviceInfo::DeviceInfo

gpu::DeviceInfo::DeviceInfo
-------------------------------
.. cpp:function:: gpu::DeviceInfo::DeviceInfo()

.. cpp:function:: gpu::DeviceInfo::DeviceInfo(int device_id)

    Constructs :cpp:class:`gpu::DeviceInfo` object for the specified device. If ``device_id`` parameter is missed it constructs object for the current device.

    :param device_id: Index of the GPU device in system starting with 0.



.. index:: gpu::DeviceInfo::name

gpu::DeviceInfo::name
-------------------------
.. cpp:function:: string gpu::DeviceInfo::name()

    Returns the device name.



.. index:: gpu::DeviceInfo::majorVersion

gpu::DeviceInfo::majorVersion
---------------------------------
.. cpp:function:: int gpu::DeviceInfo::majorVersion()

    Returns the major compute capability version.



.. index:: gpu::DeviceInfo::minorVersion

gpu::DeviceInfo::minorVersion
---------------------------------
.. cpp:function:: int gpu::DeviceInfo::minorVersion()

    Returns the minor compute capability version.



.. index:: gpu::DeviceInfo::multiProcessorCount

gpu::DeviceInfo::multiProcessorCount
----------------------------------------
.. cpp:function:: int gpu::DeviceInfo::multiProcessorCount()

    Returns the number of streaming multiprocessors.



.. index:: gpu::DeviceInfo::freeMemory

gpu::DeviceInfo::freeMemory
-------------------------------
.. cpp:function:: size_t gpu::DeviceInfo::freeMemory()

    Returns the amount of free memory in bytes.



.. index:: gpu::DeviceInfo::totalMemory

gpu::DeviceInfo::totalMemory
--------------------------------
.. cpp:function:: size_t gpu::DeviceInfo::totalMemory()

    Returns the amount of total memory in bytes.



.. index:: gpu::DeviceInfo::supports

gpu::DeviceInfo::supports
-----------------------------
.. cpp:function:: bool gpu::DeviceInfo::supports(GpuFeature feature)

    Returns true if the device has the given GPU feature, otherwise false.

    :param feature: Feature to be checked. See :c:type:`gpu::GpuFeature`.



.. index:: gpu::DeviceInfo::isCompatible

gpu::DeviceInfo::isCompatible
---------------------------------
.. cpp:function:: bool gpu::DeviceInfo::isCompatible()

    Returns true if the GPU module can be run on the specified device, otherwise false.



.. index:: gpu::TargetArchs

gpu::TargetArchs
----------------
.. c:type:: gpu::TargetArchs

This class provides functionality (as set of static methods) for checking which NVIDIA card architectures the GPU module was built for.

The following method checks whether the module was built with the support of the given feature:

.. cpp:function:: static bool gpu::TargetArchs::builtWith(GpuFeature feature) 

    :param feature: Feature to be checked. See :c:type:`gpu::GpuFeature`.

There are a set of methods for checking whether the module contains intermediate (PTX) or binary GPU code for the given architecture(s):

.. cpp:function:: static bool gpu::TargetArchs::has(int major, int minor)

.. cpp:function:: static bool gpu::TargetArchs::hasPtx(int major, int minor)

.. cpp:function:: static bool gpu::TargetArchs::hasBin(int major, int minor)

.. cpp:function:: static bool gpu::TargetArchs::hasEqualOrLessPtx(int major, int minor)

.. cpp:function:: static bool gpu::TargetArchs::hasEqualOrGreater(int major, int minor)

.. cpp:function:: static bool gpu::TargetArchs::hasEqualOrGreaterPtx(int major, int minor)

.. cpp:function:: static bool gpu::TargetArchs::hasEqualOrGreaterBin(int major, int minor)

    :param major: Major compute capability version.

    :param minor: Minor compute capability version.

According to the CUDA C Programming Guide Version 3.2: "PTX code produced for some specific compute capability can always be compiled to binary code of greater or equal compute capability".



.. index:: gpu::MultiGpuManager

gpu::MultiGpuManager
--------------------
.. c:type:: gpu::MultiGpuManager

Provides functionality for working with many GPUs. ::

    class MultiGpuManager
    {
    public:
        MultiGpuManager();
        ~MultiGpuManager();

        // Must be called before any other GPU calls
        void init();

        // Makes the given GPU active
        void gpuOn(int gpu_id);

        // Finishes the piece of work on the current GPU
        void gpuOff();

        static const int BAD_GPU_ID;
    };



.. index:: gpu::MultiGpuManager::MultiGpuManager

gpu::MultiGpuManager::MultiGpuManager
----------------------------------------
.. cpp:function:: gpu::MultiGpuManager::MultiGpuManager()

    Creates multi GPU manager, but doesn't initialize it.



.. index:: gpu::MultiGpuManager::~MultiGpuManager

gpu::MultiGpuManager::~MultiGpuManager
----------------------------------------
.. cpp:function:: gpu::MultiGpuManager::~MultiGpuManager()

    Releases multi GPU manager.



.. index:: gpu::MultiGpuManager::init

gpu::MultiGpuManager::init
----------------------------------------
.. cpp:function:: void gpu::MultiGpuManager::init()

    Initializes multi GPU manager.



.. index:: gpu::MultiGpuManager::gpuOn

gpu::MultiGpuManager::gpuOn
----------------------------------------
.. cpp:function:: void gpu::MultiGpuManager::gpuOn(int gpu_id)

    Makes the given GPU active.
    
    :param gpu_id: Index of the GPU device in system starting with 0.



.. index:: gpu::MultiGpuManager::gpuOff

gpu::MultiGpuManager::gpuOff
----------------------------------------
.. cpp:function:: void gpu::MultiGpuManager::gpuOff()

    Finishes the piece of work on the current GPU.
