Data Structures and Utility Functions
========================================

.. highlight:: cpp

ocl::Info
---------
.. ocv:class:: ocl::Info

this class should be maintained by the user and be passed to getDevice

ocl::getDevice
------------------
Returns the list of devices

.. ocv:function:: int ocl::getDevice( std::vector<Info> & oclinfo, int devicetype=CVCL_DEVICE_TYPE_GPU )

    :param oclinfo: Output vector of ``ocl::Info`` structures

    :param devicetype: One of ``CVCL_DEVICE_TYPE_GPU``, ``CVCL_DEVICE_TYPE_CPU`` or ``CVCL_DEVICE_TYPE_DEFAULT``.

the function must be called before any other ``cv::ocl`` functions; it initializes ocl runtime.

ocl::setDevice
------------------
Returns void

.. ocv:function:: void ocl::setDevice( Info &oclinfo, int devnum = 0 )

    :param oclinfo: Output vector of ``ocl::Info`` structures

    :param devnum: the selected OpenCL device under this platform.

ocl::setBinpath
------------------
Returns void

.. ocv:function:: void ocl::setBinpath(const char *path)

    :param path: the path of OpenCL kernel binaries

If you call this function and set a valid path, the OCL module will save the compiled kernel to the address in the first time and reload the binary since that. It can save compilation time at the runtime.

ocl::getoclContext
------------------
Returns the pointer to the opencl context

.. ocv:function:: void* ocl::getoclContext()

Thefunction are used to get opencl context so that opencv can interactive with other opencl program.

ocl::getoclCommandQueue
--------------------------
Returns the pointer to the opencl command queue

.. ocv:function:: void* ocl::getoclCommandQueue()

Thefunction are used to get opencl command queue so that opencv can interactive with other opencl program.
