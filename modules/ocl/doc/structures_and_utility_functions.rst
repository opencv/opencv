Data Structures and Utility Functions
========================================

.. highlight:: cpp

ocl::getOpenCLPlatforms
-----------------------
Returns the list of OpenCL platforms

.. ocv:function:: int ocl::getOpenCLPlatforms( PlatformsInfo& platforms )

    :param platforms: Output variable

ocl::getOpenCLDevices
---------------------
Returns the list of devices

.. ocv:function:: int ocl::getOpenCLDevices( DevicesInfo& devices, int deviceType = CVCL_DEVICE_TYPE_GPU, const PlatformInfo* platform = NULL )

    :param devices: Output variable

    :param deviceType: Bitmask of ``CVCL_DEVICE_TYPE_GPU``, ``CVCL_DEVICE_TYPE_CPU`` or ``CVCL_DEVICE_TYPE_DEFAULT``.

    :param platform: Specifies preferrable platform

ocl::setDevice
--------------
Initialize OpenCL computation context

.. ocv:function:: void ocl::setDevice( const DeviceInfo* info )

    :param info: device info

ocl::initializeContext
--------------------------------
Alternative way to initialize OpenCL computation context.

.. ocv:function:: void ocl::initializeContext(void* pClPlatform, void* pClContext, void* pClDevice)

    :param pClPlatform: selected ``platform_id`` (via pointer, parameter type is ``cl_platform_id*``)

    :param pClContext: selected ``cl_context`` (via pointer, parameter type is ``cl_context*``)

    :param pClDevice: selected ``cl_device_id`` (via pointer, parameter type is ``cl_device_id*``)

This function can be used for context initialization with D3D/OpenGL interoperability.

ocl::setBinaryPath
------------------
Returns void

.. ocv:function:: void ocl::setBinaryPath(const char *path)

    :param path: the path of OpenCL kernel binaries

If you call this function and set a valid path, the OCL module will save the compiled kernel to the address in the first time and reload the binary since that. It can save compilation time at the runtime.
