OpenCL Module Introduction
==========================

.. highlight:: cpp

General Information
-------------------

The OpenCV OCL module contains  a set of classes and functions that implement and accelerate select openCV functionality on OpenCL compatible devices. OpenCL is a Khronos standard, implemented by a variety of devices (CPUs, GPUs, FPGAs, ARM), abstracting the exact hardware details, while enabling vendors to provide native implementation for maximal acceleration on their hardware. The standard enjoys wide industry support, and the end user of the module will enjoy the data parallelism benefits that the specific platform/hardware may be capable of, in a platform/hardware independent manner. 

While in the future we hope to validate (and enable) the OCL module in all OpenCL capable devices, we currently develop and test on GPU devices only. This includes both discrete GPUs (NVidia, AMD), as well as integrated chips(AMD APU and intel HD devices). Performance of any particular algorithm will depend on the particular platform characteristics and capabilities. However, currently (as of 2.4.4), accuracy and  mathematical correctness has been verified to be identical to that of the pure CPU implementation on all tested GPU devices and platforms (both windows and linux). 


The OpenCV OCL module includes utility functions, low-level vision primitives, and high-level algorithms. The utility functions and low-level primitives provide a powerful infrastructure for developing fast vision algorithms taking advangtage of OCL whereas the high-level functionality (samples)includes some state-of-the-art algorithms (including LK Optical flow, and Face detection) ready to be used by the application developers. The module is also accompanied by an extensive performance and accuracy test suite.

The OpenCV OCL module is designed for ease of use and does not require any knowledge of OpenCL. At a minimuml level, it can be viewed as a set of accelerators, that can take advantage of the high compute throughput that GPU/APU devices can provide. However, it can also be viewed as a starting point to really integratethe  built-in functionality with your own custom OpenCL kernels, with or without modifying the source of OpenCV-OCL. Of course, knowledge of OpenCL will certainly help, however we hope that OpenCV-OCL module, and  the kernels it contains in source code, can be very useful as a means of actually learning openCL. Such a knowledge would be necessary to further fine-tune any of the existing OpenCL kernels, or for extending the framework with new kernels. As of OpenCV 2.4.4, we introduce interoperability with OpenCL, enabling easy use of custom OpenCL kernels within the OpenCV framework. 

To use the OCL module, you need to make sure that you have the OpenCL SDK provided with your device vendor. To correctly run the OCL module, you need to have the OpenCL runtime provide by the device vendor, typically the device driver.

To enable OCL support, configure OpenCV using CMake with WITH\_OPENCL=ON. When the flag is set and if OpenCL SDK is installed, the full-featured OpenCV OCL module is built. Otherwise, the module may be not built. If you have AMD'S FFT and BLAS library, you can select it with WITH\_OPENCLAMDFFT=ON, WITH\_OPENCLAMDBLAS=ON.

The ocl module can be found under the "modules" directory. In "modules/ocl/src" you can find the source code for the cpp class that wrap around the direct kernel invocation. The kernels themselves can be found in "modules/ocl/src/kernels."  Samples can be found under "samples/ocl." Accuracy tests can be found in "modules/ocl/test," and performance tests under "module/ocl/perf."




Right now, the user should define the cv::ocl::Info class in the application and call cv::ocl::getDevice before any cv::ocl::func. This operation initialize OpenCL runtime and set the first found device as computing device. If there are more than one device and you want to use undefault device, you can call cv::ocl::setDevice then.

In the current version, all the thread share the same context and device so the multi-devices are not supported. We will add this feature soon. If a function support 4-channel operator, it should support 3-channel operator as well, because All the 3-channel matrix(i.e. RGB image) are represented by 4-channel matrix in oclMat. It means 3-channel image have 4-channel space with the last channel unused. We provide a transparent interface to handle the difference between OpenCV Mat and oclMat.

Developer Notes
-------------------

In a heterogeneous device environment, there may be cost associated with data transfer. This would be the case, for example, when data needs to be moved from host memory (accessible to the CPU), to device memory (accessible to a discrete GPU). in the case of integrated graphics chips, there may be performance issues, relating to memory coherency between access from the GPU "part" of the integrated device, or the CPU "part." For best performance, in either case, it is recommended that you do not introduce dat transfers between CPU and the discrete GPU, except in the beginning and the end of the algorithmic pipeline.

Some tidbits:

1. OpenCL version should be larger than 1.1 with FULL PROFILE.

2. Currently (2.4.4) the user call the cv::ocl::getDevice before any other function in the ocl module. This will initialize the OpenCL runtime and set the first found device as computing device. If there are more than one device and you want to use undefault device, you can call cv::ocl::setDevice thereafter.

2. Currently there's only one OpenCL context and command queue. We hope to implement multi device and multi queue support in the future.

3. Many kernels use 256 as its workgroup size if possible, so the max work group size of the device must larger than 256. All GPU devices we are aware of indeed support 256 workitems in a workgroup, however non GPU devices may not. This will be improved in the future.

4. If the device does not support double arithetic, we revert to float.

5. The oclMat uses buffer object, not image object.

6. All the 3-channel matrices(i.e. RGB image) are represented by 4-channel matrices in oclMat, with the last channel unused. We provide a transparent interface to handle the difference between OpenCV Mat and oclMat.

7. All the matrix in oclMat is aligned in column(now the alignment factor is 32 byte). It means, if a matrix is n columns m rows with the element size x byte, we will assign ALIGNMENT(x*n) bytes for each column with the last ALIGNMENT(x*n) - x*n bytes unused, so there's small holes after each column if its size is not the multiply of ALIGN.

8. Data transfer between Mat and oclMat: If the CPU matrix is aligned in column, we will use faster API to transfer between Mat and oclMat, otherwise, we will use clEnqueueRead/WriteBufferRect to transfer data to guarantee the alignment. 3-channel matrix is an exception, it's directly transferred to a temp buffer and then padded to 4-channel matrix(also aligned) when uploading and do the reverse operation when downloading.

9. Data transfer between Mat and oclMat: ROI is a feature of OpenCV, which allow users process a sub rectangle of a matrix. When a CPU matrix which has ROI will be transfered to GPU, the whole matrix will be transfered and set ROI as CPU's. In a word, we always transfer the whole matrix despite whether it has ROI or not.

10. All the kernel file should locate in ocl/src/kernels/ with the extension ".cl". ALL the kernel files are transformed to pure characters at compilation time in kernels.cpp, and the file name without extension is the name of the characters.
