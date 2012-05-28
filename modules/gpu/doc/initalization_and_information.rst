Initalization and Information
=============================

.. highlight:: cpp



gpu::getCudaEnabledDeviceCount
----------------------------------
Returns the number of installed CUDA-enabled devices.

.. ocv:function:: int gpu::getCudaEnabledDeviceCount()

Use this function before any other GPU functions calls. If OpenCV is compiled without GPU support, this function returns 0.



gpu::setDevice
------------------
Sets a device and initializes it for the current thread.

.. ocv:function:: void gpu::setDevice(int device)

    :param device: System index of a GPU device starting with 0.

If the call of this function is omitted, a default device is initialized at the fist GPU usage.



gpu::getDevice
------------------
Returns the current device index set by :ocv:func:`gpu::setDevice` or initialized by default.

.. ocv:function:: int gpu::getDevice()



gpu::resetDevice
------------------
Explicitly destroys and cleans up all resources associated with the current device in the current process.

.. ocv:function:: void gpu::resetDevice()

Any subsequent API call to this device will reinitialize the device.



gpu::FeatureSet
---------------

Class providing GPU computing features. ::

    enum FeatureSet
    {
        FEATURE_SET_COMPUTE_10,
        FEATURE_SET_COMPUTE_11,
        FEATURE_SET_COMPUTE_12,
        FEATURE_SET_COMPUTE_13,
        FEATURE_SET_COMPUTE_20,
        FEATURE_SET_COMPUTE_21,
        GLOBAL_ATOMICS,
        SHARED_ATOMICS,
        NATIVE_DOUBLE
    };



gpu::TargetArchs
----------------
.. ocv:class:: gpu::TargetArchs

Class providing a set of static methods to check what NVIDIA* card architecture the GPU module was built for.

The following method checks whether the module was built with the support of the given feature:

    .. ocv:function:: static bool gpu::TargetArchs::builtWith( FeatureSet feature_set )

        :param feature: Feature to be checked. See :ocv:class:`gpu::FeatureSet`.

There is a set of methods to check whether the module contains intermediate (PTX) or binary GPU code for the given architecture(s):

    .. ocv:function:: static bool gpu::TargetArchs::has(int major, int minor)

    .. ocv:function:: static bool gpu::TargetArchs::hasPtx(int major, int minor)

    .. ocv:function:: static bool gpu::TargetArchs::hasBin(int major, int minor)

    .. ocv:function:: static bool gpu::TargetArchs::hasEqualOrLessPtx(int major, int minor)

    .. ocv:function:: static bool gpu::TargetArchs::hasEqualOrGreater(int major, int minor)

    .. ocv:function:: static bool gpu::TargetArchs::hasEqualOrGreaterPtx(int major, int minor)

    .. ocv:function:: static bool gpu::TargetArchs::hasEqualOrGreaterBin(int major, int minor)

        :param major: Major compute capability version.

        :param minor: Minor compute capability version.

According to the CUDA C Programming Guide Version 3.2: "PTX code produced for some specific compute capability can always be compiled to binary code of greater or equal compute capability".



gpu::DeviceInfo
---------------
.. ocv:class:: gpu::DeviceInfo

Class providing functionality for querying the specified GPU properties. ::

    class CV_EXPORTS DeviceInfo
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

        bool supports(FeatureSet feature) const;
        bool isCompatible() const;

        int deviceID() const;
    };



gpu::DeviceInfo::DeviceInfo
-------------------------------
The constructors.

.. ocv:function:: gpu::DeviceInfo::DeviceInfo()

.. ocv:function:: gpu::DeviceInfo::DeviceInfo(int device_id)

    :param device_id: System index of the GPU device starting with 0.

Constructs the ``DeviceInfo`` object for the specified device. If ``device_id`` parameter is missed, it constructs an object for the current device.



gpu::DeviceInfo::name
-------------------------
Returns the device name.

.. ocv:function:: string gpu::DeviceInfo::name() const



gpu::DeviceInfo::majorVersion
---------------------------------
Returns the major compute capability version.

.. ocv:function:: int gpu::DeviceInfo::majorVersion()



gpu::DeviceInfo::minorVersion
---------------------------------
Returns the minor compute capability version.

.. ocv:function:: int gpu::DeviceInfo::minorVersion()



gpu::DeviceInfo::multiProcessorCount
----------------------------------------
Returns the number of streaming multiprocessors.

.. ocv:function:: int gpu::DeviceInfo::multiProcessorCount()



gpu::DeviceInfo::freeMemory
-------------------------------
Returns the amount of free memory in bytes.

.. ocv:function:: size_t gpu::DeviceInfo::freeMemory()



gpu::DeviceInfo::totalMemory
--------------------------------
Returns the amount of total memory in bytes.

.. ocv:function:: size_t gpu::DeviceInfo::totalMemory()



gpu::DeviceInfo::supports
-----------------------------
Provides information on GPU feature support.

.. ocv:function:: bool gpu::DeviceInfo::supports( FeatureSet feature_set ) const

    :param feature: Feature to be checked. See :ocv:class:`gpu::FeatureSet`.

This function returns ``true`` if the device has the specified GPU feature. Otherwise, it returns ``false`` .



gpu::DeviceInfo::isCompatible
---------------------------------
Checks the GPU module and device compatibility.

.. ocv:function:: bool gpu::DeviceInfo::isCompatible()

This function returns ``true`` if the GPU module can be run on the specified device. Otherwise, it returns ``false`` .



gpu::DeviceInfo::deviceID
---------------------------------
Returns system index of the GPU device starting with 0.

.. ocv:function:: int gpu::DeviceInfo::deviceID()
