Data Structures and Functions
=============================

.. highlight:: cpp

ocl::Info
---------
.. ocv:class:: ocl::Info

this class should be maintained by the user and be passed to getDevice

ocl::getDevice
------------------
Returns the list of devices

.. ocv:function:: int ocl::getDevice(std::vector<Info>& oclinfo, int devicetype = CVCL_DEVICE_TYPE_GPU)

    :param oclinfo: Output vector of ``ocl::Info`` structures

    :param devicetype: One of ``CVCL_DEVICE_TYPE_GPU``, ``CVCL_DEVICE_TYPE_CPU`` or ``CVCL_DEVICE_TYPE_DEFAULT``.

the function must be called before any other ``cv::ocl`` functions; it initializes ocl runtime.

