OpenCL Module Introduction
==========================

.. highlight:: cpp

General Information
-------------------

The OpenCV OCL module is a set of classes and functions to utilize OpenCL compatible device. In theroy, it supports any OpenCL 1.1 compatible device, but we only test it on AMD's, Intel's and NVIDIA's GPU at this stage. The compatibility, correctness and performance on CPU is not guaranteed. The OpenCV OCL module includes utility functions, low-level vision primitives, and high-level algorithms. The utility functions and low-level primitives provide a powerful infrastructure for developing fast vision algorithms taking advangtage of OCL whereas the high-level functionality includes some state-of-the-art algorithms(such as surf detector, face detector) ready to be used by the application developers.

The OpenCV OCL module is designed as a host-level API plus device-level kernels. The device-level kernels are collected as strings at OpenCV compile time and are compiled at runtime, so it need OpenCL runtime support. To correctly build the OpenCV OCL module, make sure you have OpenCL SDK provided your device vendor. To correctly run the OpenCV OCL module, make sure you have OpenCL runtime provided by your device vendor, which is device driver normally.

The OpenCV OCL module is designed for ease of use and does not require any knowledge of OpenCL. Though, such a knowledge will certainly be useful to handle non-trivial cases or achieve the highest performance. It is helpful to understand the cost of various operations, what the OCL does, what the preferred data formats are, and so on. Since there is data transfer between OpenCL host and OpenCL device, for better performance it's recommended to copy data once to the OpenCL host memory (i.e. copy ``cv::Mat`` to ``cv::ocl::OclMat``), then call several ``cv::ocl`` functions and then copy the result back to CPU memory, rather than do forward and backward transfer for each OCL function.

To enable OCL support, configure OpenCV using CMake with WIHT\_OPENCL=ON. When the flag is set and if OpenCL SDK is installed, the full-featured OpenCV OCL module is built. Otherwise, the module may be not built. If you have AMD'S FFT and BLAS library, you can select it with WITH\_OPENCLAMDFFT=ON, WIHT\_OPENCLAMDBLAS=ON.

Right now, the user should define the cv::ocl::Info class in the application and call cv::ocl::getDevice before any cv::ocl::func. This operation initialize OpenCL runtime and set the first found device as computing device. If there are more than one device and you want to use undefault device, you can call cv::ocl::setDevice then.

In the current version, all the thread share the same context and device so the multi-devices are not supported. We will add this feature soon. If a function support 4-channel operator, it should support 3-channel operator as well, because All the 3-channel matrix(i.e. RGB image) are represented by 4-channel matrix in oclMat. It means 3-channel image have 4-channel space with the last channel unused. We provide a transparent interface to handle the difference between OpenCV Mat and oclMat.

Developer Notes
-------------------

This section descripe the design details of ocl module for who are interested in the detail of this module or want to contribute this module. User who isn't interested the details, can safely ignore it.

1. OpenCL version should be larger than 1.1 with FULL PROFILE.

2. There's only one OpenCL context and commandqueue and generated as a singleton. So now it only support one device with single commandqueue.

3. All the functions use 256 as its workgroup size if possible, so the max work group size of the device must larger than 256.

4. If the device support double, we will use double in kernel if OpenCV cpu version use double, otherwise, we use float instead.

5. The oclMat use buffer object, not image object.

6. All the 3-channel matrix(i.e. RGB image) are represented by 4-channel matrix in oclMat. It means 3-channel image have 4-channel space with the last channel unused. We provide a transparent interface to handle the difference between OpenCV Mat and oclMat.

7. All the matrix in oclMat is aligned in column(now the alignment factor is 32 byte). It means, if a matrix is n columns m rows with the element size x byte, we will assign ALIGNMENT(x*n) bytes for each column with the last ALIGNMENT(x*n) - x*n bytes unused, so there's small holes after each column if its size is not the multiply of ALIGN.

8. Data transfer between Mat and oclMat. If the CPU matrix is aligned in column, we will use faster API to transfer between Mat and oclMat, otherwise, we will use clEnqueueRead/WriteBufferRect to transfer data to guarantee the alignment. 3-channel matrix is an exception, it's directly transferred to a temp buffer and then padded to 4-channel matrix(also aligned) when uploading and do the reverse operation when downloading.

9. Data transfer between Mat and oclMat. ROI is a feature of OpenCV, which allow users process a sub rectangle of a matrix. When a CPU matrix which has ROI will be transfered to GPU, the whole matrix will be transfered and set ROI as CPU's. In a word, we always transfer the whole matrix despite whether it has ROI or not.

10. All the kernel file should locate in ocl/src/kernels/ with the extension ".cl". ALL the kernel files are transformed to pure characters at compilation time in kernels.cpp, and the file name without extension is the name of the characters.
