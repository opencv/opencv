Initalization and Information
=============================

.. highlight:: cpp



.. index:: gpu::getCudaEnabledDeviceCount


cv::gpu::getCudaEnabledDeviceCount
----------------------------------

`id=0.541856697999 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3AgetCudaEnabledDeviceCount>`__




.. cfunction:: int getCudaEnabledDeviceCount()

    Returns number of CUDA-enabled devices installed. It is to be used before any other GPU functions calls. If OpenCV is compiled without GPU support this function returns 0. 




.. index:: gpu::setDevice


cv::gpu::setDevice
------------------

`id=0.817295536445 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3AsetDevice>`__




.. cfunction:: void setDevice(int device)

    Sets device and initializes it for the current thread. Call of this function can be omitted, but in this case a default device will be initialized on fist GPU usage.





    
    :param device: index of GPU device in system starting with 0. 
    
    
    

.. index:: gpu::getDevice


cv::gpu::getDevice
------------------

`id=0.908782607162 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3AgetDevice>`__




.. cfunction:: int getDevice()

    Returns the current device index, which was set by {gpu::getDevice} or initialized by default.




.. index:: gpu::GpuFeature

.. _gpu::GpuFeature:

gpu::GpuFeature
---------------

`id=0.185426029041 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3AGpuFeature>`__

.. ctype:: gpu::GpuFeature



GPU compute features.




::


    
    enum GpuFeature
    {
        COMPUTE_10, COMPUTE_11,
        COMPUTE_12, COMPUTE_13,
        COMPUTE_20, COMPUTE_21,
        ATOMICS, NATIVE_DOUBLE
    };
    

..


.. index:: gpu::DeviceInfo

.. _gpu::DeviceInfo:

gpu::DeviceInfo
---------------

`id=0.91098225386 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3ADeviceInfo>`__

.. ctype:: gpu::DeviceInfo



This class provides functionality for querying the specified GPU properties. 




::


    
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
    
        bool supports(GpuFeature feature) const;
        bool isCompatible() const;
    };
    

..


.. index:: gpu::DeviceInfo::DeviceInfo


cv::gpu::DeviceInfo::DeviceInfo
-------------------------------

`id=0.971366637207 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3ADeviceInfo%3A%3ADeviceInfo>`__


``_``


.. cfunction:: DeviceInfo::DeviceInfo()



.. cfunction:: DeviceInfo::DeviceInfo(int device_id)

    Constructs DeviceInfo object for the specified device. If deviceidparameter is missed it constructs object for the current device.





    
    :param device_id: Index of the GPU device in system starting with 0. 
    
    
    

.. index:: gpu::DeviceInfo::name


cv::gpu::DeviceInfo::name
-------------------------

`id=0.472941921148 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3ADeviceInfo%3A%3Aname>`__




.. cfunction:: string DeviceInfo::name()

    Returns the device name.




.. index:: gpu::DeviceInfo::majorVersion


cv::gpu::DeviceInfo::majorVersion
---------------------------------

`id=0.982334984119 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3ADeviceInfo%3A%3AmajorVersion>`__




.. cfunction:: int DeviceInfo::majorVersion()

    Returns the major compute capability version.




.. index:: gpu::DeviceInfo::minorVersion


cv::gpu::DeviceInfo::minorVersion
---------------------------------

`id=0.309433581176 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3ADeviceInfo%3A%3AminorVersion>`__




.. cfunction:: int DeviceInfo::minorVersion()

    Returns the minor compute capability version.




.. index:: gpu::DeviceInfo::multiProcessorCount


cv::gpu::DeviceInfo::multiProcessorCount
----------------------------------------

`id=0.417609601388 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3ADeviceInfo%3A%3AmultiProcessorCount>`__




.. cfunction:: int DeviceInfo::multiProcessorCount()

    Returns the number of streaming multiprocessors.




.. index:: gpu::DeviceInfo::freeMemory


cv::gpu::DeviceInfo::freeMemory
-------------------------------

`id=0.961189453269 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3ADeviceInfo%3A%3AfreeMemory>`__




.. cfunction:: size_t DeviceInfo::freeMemory()

    Returns the amount of free memory in bytes.




.. index:: gpu::DeviceInfo::totalMemory


cv::gpu::DeviceInfo::totalMemory
--------------------------------

`id=0.884488673579 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3ADeviceInfo%3A%3AtotalMemory>`__




.. cfunction:: size_t DeviceInfo::totalMemory()

    Returns the amount of total memory in bytes.




.. index:: gpu::DeviceInfo::supports


cv::gpu::DeviceInfo::supports
-----------------------------

`id=0.141435828088 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3ADeviceInfo%3A%3Asupports>`__




.. cfunction:: bool DeviceInfo::supports(GpuFeature feature)

    Returns true if the device has the given GPU feature, otherwise false.





    
    :param feature: Feature to be checked. See  . 
    
    
    

.. index:: gpu::DeviceInfo::isCompatible


cv::gpu::DeviceInfo::isCompatible
---------------------------------

`id=0.564690282768 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3ADeviceInfo%3A%3AisCompatible>`__




.. cfunction:: bool DeviceInfo::isCompatible()

    Returns true if the GPU module can be run on the specified device, otherwise false.




.. index:: gpu::TargetArchs

.. _gpu::TargetArchs:

gpu::TargetArchs
----------------

`id=0.200853353999 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/gpu/gpu%3A%3ATargetArchs>`__

.. ctype:: gpu::TargetArchs



This class provides functionality (as set of static methods) for checking which NVIDIA card architectures the GPU module was built for.

bigskip
The following method checks whether the module was built with the support of the given feature:


.. cfunction:: static bool builtWith(GpuFeature feature)





    
    :param feature: Feature to be checked. See  . 
    
    
    
There are a set of methods for checking whether the module contains intermediate (PTX) or binary GPU code for the given architecture(s):


.. cfunction:: static bool has(int major, int minor)



.. cfunction:: static bool hasPtx(int major, int minor)



.. cfunction:: static bool hasBin(int major, int minor)



.. cfunction:: static bool hasEqualOrLessPtx(int major, int minor)



.. cfunction:: static bool hasEqualOrGreater(int major, int minor)



.. cfunction:: static bool hasEqualOrGreaterPtx(int major, int minor)



.. cfunction:: static bool hasEqualOrGreaterBin(int major, int minor)





    
    * **major** Major compute capability version. 
    
    
    * **minor** Minor compute capability version. 
    
    
    
According to the CUDA C Programming Guide Version 3.2: "PTX code produced for some specific compute capability can always be compiled to binary code of greater or equal compute capability". 

