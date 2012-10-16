OpenCL Module Introduction
==========================

.. highlight:: cpp

General Information
-------------------

The OpenCV OCL module is a set of classes and functions to utilize OpenCL compatible device. It should support any device compatible with OpenCL 1.1. The module includes utility functions, low-level vision primitives, and a few high-level algorithms ready to be used in the end-user applications.

The OpenCV OCL module is designed as a host-level API plus device-level kernels. The device-level kernels are converted to text string and are compiled at runtime, so it need OpenCL runtime support. To correctly run the OpenCV OCL module, make sure you have OpenCL runtime provided by your device vendor, which is device driver normally.

The OpenCV OCL module is designed for ease of use and does not require any knowledge of OpenCL. Though, such a knowledge will certainly be useful to handle non-trivial cases or achieve the highest performance. It is helpful to understand the cost of various operations, what the module does, what the preferred data formats are, and so on. Since there is data transfer between OpenCL host and OpenCL device, for better performance it's recommended to copy data once to the OpenCL host memory (i.e. copy ``cv::Mat`` to ``cv::ocl::OclMat``), then call several ``cv::ocl`` functions and then copy the result back to CPU memory, rather than do forward and backward transfer for each OCL function.

To enable OCL support, configure OpenCV using CMake with the option ``WITH\_OPENCL=ON``. If the option is passed and if OpenCL SDK is installed (e.g. on MacOSX it's always the case), the full-featured OpenCV OCL module will be built. Otherwise, the module will not be built.

Right now, the user should define the ``cv::ocl::Info`` class in the application and call ``cv::ocl::getDevice`` before any ``cv::ocl::<func>``. This operation initialize OpenCL runtime and set the first found device as computing device. If there is more than one device and you want to use non-default device, you should call ``cv::ocl::setDevice``.

In the current version, all the threads share the same context and device so the multi-devices are not supported. This is to be fixed in future releases.
