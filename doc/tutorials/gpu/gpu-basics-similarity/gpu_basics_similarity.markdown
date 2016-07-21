Similarity check (PNSR and SSIM) on the GPU {#tutorial_gpu_basics_similarity}
===========================================
@todo update this tutorial

Goal
----

In the @ref tutorial_video_input_psnr_ssim tutorial I already presented the PSNR and SSIM methods for checking
the similarity between the two images. And as you could see there performing these takes quite some
time, especially in the case of the SSIM. However, if the performance numbers of an OpenCV
implementation for the CPU do not satisfy you and you happen to have an NVidia CUDA GPU device in
your system all is not lost. You may try to port or write your algorithm for the video card.

This tutorial will give a good grasp on how to approach coding by using the GPU module of OpenCV. As
a prerequisite you should already know how to handle the core, highgui and imgproc modules. So, our
goals are:

-   What's different compared to the CPU?
-   Create the GPU code for the PSNR and SSIM
-   Optimize the code for maximal performance

The source code
---------------

You may also find the source code and these video file in the
`samples/cpp/tutorial_code/gpu/gpu-basics-similarity/gpu-basics-similarity` folder of the OpenCV
source library or download it from [here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/gpu/gpu-basics-similarity/gpu-basics-similarity.cpp).
The full source code is quite long (due to the controlling of the application via the command line
arguments and performance measurement). Therefore, to avoid cluttering up these sections with those
you'll find here only the functions itself.

The PSNR returns a float number, that if the two inputs are similar between 30 and 50 (higher is
better).

@snippet samples/cpp/tutorial_code/gpu/gpu-basics-similarity/gpu-basics-similarity.cpp getpsnr
@snippet samples/cpp/tutorial_code/gpu/gpu-basics-similarity/gpu-basics-similarity.cpp getpsnrcuda
@snippet samples/cpp/tutorial_code/gpu/gpu-basics-similarity/gpu-basics-similarity.cpp psnr
@snippet samples/cpp/tutorial_code/gpu/gpu-basics-similarity/gpu-basics-similarity.cpp getpsnropt

The SSIM returns the MSSIM of the images. This is too a float number between zero and one (higher is
better), however we have one for each channel. Therefore, we return a *Scalar* OpenCV data
structure:

@snippet samples/cpp/tutorial_code/gpu/gpu-basics-similarity/gpu-basics-similarity.cpp getssim
@snippet samples/cpp/tutorial_code/gpu/gpu-basics-similarity/gpu-basics-similarity.cpp getssimcuda
@snippet samples/cpp/tutorial_code/gpu/gpu-basics-similarity/gpu-basics-similarity.cpp ssim
@snippet samples/cpp/tutorial_code/gpu/gpu-basics-similarity/gpu-basics-similarity.cpp getssimopt

How to do it? - The GPU
-----------------------

Now as you can see we have three types of functions for each operation. One for the CPU and two for
the GPU. The reason I made two for the GPU is too illustrate that often simple porting your CPU to
GPU will actually make it slower. If you want some performance gain you will need to remember a few
rules, whose I'm going to detail later on.

The development of the GPU module was made so that it resembles as much as possible its CPU
counterpart. This is to make porting easy. The first thing you need to do before writing any code is
to link the GPU module to your project, and include the header file for the module. All the
functions and data structures of the GPU are in a *gpu* sub namespace of the *cv* namespace. You may
add this to the default one via the *use namespace* keyword, or mark it everywhere explicitly via
the cv:: to avoid confusion. I'll do the later.
@code{.cpp}
#include <opencv2/gpu.hpp>        // GPU structures and methods
@endcode

GPU stands for "graphics processing unit". It was originally build to render graphical
scenes. These scenes somehow build on a lot of data. Nevertheless, these aren't all dependent one
from another in a sequential way and as it is possible a parallel processing of them. Due to this a
GPU will contain multiple smaller processing units. These aren't the state of the art processors and
on a one on one test with a CPU it will fall behind. However, its strength lies in its numbers. In
the last years there has been an increasing trend to harvest these massive parallel powers of the
GPU in non-graphical scene rendering too. This gave birth to the general-purpose computation on
graphics processing units (GPGPU).

The GPU has its own memory. When you read data from the hard drive with OpenCV into a *Mat* object
that takes place in your systems memory. The CPU works somehow directly on this (via its cache),
however the GPU cannot. He has too transferred the information he will use for calculations from the
system memory to its own. This is done via an upload process and takes time. In the end the result
will have to be downloaded back to your system memory for your CPU to see it and use it. Porting
small functions to GPU is not recommended as the upload/download time will be larger than the amount
you gain by a parallel execution.

Mat objects are stored only in the system memory (or the CPU cache). For getting an OpenCV matrix to
the GPU you'll need to use its GPU counterpart @ref cv::cuda::GpuMat . It works similar to the Mat with a
2D only limitation and no reference returning for its functions (cannot mix GPU references with CPU
ones). To upload a Mat object to the GPU you need to call the upload function after creating an
instance of the class. To download you may use simple assignment to a Mat object or use the download
function.
@code{.cpp}
Mat I1;         // Main memory item - read image into with imread for example
gpu::GpuMat gI; // GPU matrix - for now empty
gI1.upload(I1); // Upload a data from the system memory to the GPU memory

I1 = gI1;       // Download, gI1.download(I1) will work too
@endcode
Once you have your data up in the GPU memory you may call GPU enabled functions of OpenCV. Most of
the functions keep the same name just as on the CPU, with the difference that they only accept
*GpuMat* inputs. A full list of these you will find in the documentation: [online
here](http://docs.opencv.org/modules/gpu/doc/gpu.html) or the OpenCV reference manual that comes
with the source code.

Another thing to keep in mind is that not for all channel numbers you can make efficient algorithms
on the GPU. Generally, I found that the input images for the GPU images need to be either one or
four channel ones and one of the char or float type for the item sizes. No double support on the
GPU, sorry. Passing other types of objects for some functions will result in an exception thrown,
and an error message on the error output. The documentation details in most of the places the types
accepted for the inputs. If you have three channel images as an input you can do two things: either
adds a new channel (and use char elements) or split up the image and call the function for each
image. The first one isn't really recommended as you waste memory.

For some functions, where the position of the elements (neighbor items) doesn't matter quick
solution is to just reshape it into a single channel image. This is the case for the PSNR
implementation where for the *absdiff* method the value of the neighbors is not important. However,
for the *GaussianBlur* this isn't an option and such need to use the split method for the SSIM. With
this knowledge you can already make a GPU viable code (like mine GPU one) and run it. You'll be
surprised to see that it might turn out slower than your CPU implementation.

Optimization
------------

The reason for this is that you're throwing out on the window the price for memory allocation and
data transfer. And on the GPU this is damn high. Another possibility for optimization is to
introduce asynchronous OpenCV GPU calls too with the help of the @ref cv::cuda::Stream.

-#  Memory allocation on the GPU is considerable. Therefore, if it’s possible allocate new memory as
    few times as possible. If you create a function what you intend to call multiple times it is a
    good idea to allocate any local parameters for the function only once, during the first call. To
    do this you create a data structure containing all the local variables you will use. For
    instance in case of the PSNR these are:
    @code{.cpp}
    struct BufferPSNR                                     // Optimized GPU versions
      {   // Data allocations are very expensive on GPU. Use a buffer to solve: allocate once reuse later.
      gpu::GpuMat gI1, gI2, gs, t1,t2;

      gpu::GpuMat buf;
    };
    @endcode
    Then create an instance of this in the main program:
    @code{.cpp}
    BufferPSNR bufferPSNR;
    @endcode
    And finally pass this to the function each time you call it:
    @code{.cpp}
    double getPSNR_GPU_optimized(const Mat& I1, const Mat& I2, BufferPSNR& b)
    @endcode
    Now you access these local parameters as: *b.gI1*, *b.buf* and so on. The GpuMat will only
    reallocate itself on a new call if the new matrix size is different from the previous one.

-#  Avoid unnecessary function data transfers. Any small data transfer will be significant one once
    you go to the GPU. Therefore, if possible make all calculations in-place (in other words do not
    create new memory objects - for reasons explained at the previous point). For example, although
    expressing arithmetical operations may be easier to express in one line formulas, it will be
    slower. In case of the SSIM at one point I need to calculate:
    @code{.cpp}
    b.t1 = 2 * b.mu1_mu2 + C1;
    @endcode
    Although the upper call will succeed observe that there is a hidden data transfer present.
    Before it makes the addition it needs to store somewhere the multiplication. Therefore, it will
    create a local matrix in the background, add to that the *C1* value and finally assign that to
    *t1*. To avoid this we use the gpu functions, instead of the arithmetic operators:
    @code{.cpp}
    gpu::multiply(b.mu1_mu2, 2, b.t1); //b.t1 = 2 * b.mu1_mu2 + C1;
    gpu::add(b.t1, C1, b.t1);
    @endcode
-#  Use asynchronous calls (the @ref cv::cuda::Stream ). By default whenever you call a gpu function
    it will wait for the call to finish and return with the result afterwards. However, it is
    possible to make asynchronous calls, meaning it will call for the operation execution, make the
    costly data allocations for the algorithm and return back right away. Now you can call another
    function if you wish to do so. For the MSSIM this is a small optimization point. In our default
    implementation we split up the image into channels and call then for each channel the gpu
    functions. A small degree of parallelization is possible with the stream. By using a stream we
    can make the data allocation, upload operations while the GPU is already executing a given
    method. For example we need to upload two images. We queue these one after another and call
    already the function that processes it. The functions will wait for the upload to finish,
    however while that happens makes the output buffer allocations for the function to be executed
    next.
    @code{.cpp}
    gpu::Stream stream;

    stream.enqueueConvert(b.gI1, b.t1, CV_32F);    // Upload

    gpu::split(b.t1, b.vI1, stream);              // Methods (pass the stream as final parameter).
    gpu::multiply(b.vI1[i], b.vI1[i], b.I1_2, stream);        // I1^2
    @endcode

Result and conclusion
---------------------

On an Intel P8700 laptop CPU paired with a low end NVidia GT220M here are the performance numbers:
@code
Time of PSNR CPU (averaged for 10 runs): 41.4122 milliseconds. With result of: 19.2506
Time of PSNR GPU (averaged for 10 runs): 158.977 milliseconds. With result of: 19.2506
Initial call GPU optimized:              31.3418 milliseconds. With result of: 19.2506
Time of PSNR GPU OPTIMIZED ( / 10 runs): 24.8171 milliseconds. With result of: 19.2506

Time of MSSIM CPU (averaged for 10 runs): 484.343 milliseconds. With result of B0.890964 G0.903845 R0.936934
Time of MSSIM GPU (averaged for 10 runs): 745.105 milliseconds. With result of B0.89922 G0.909051 R0.968223
Time of MSSIM GPU Initial Call            357.746 milliseconds. With result of B0.890964 G0.903845 R0.936934
Time of MSSIM GPU OPTIMIZED ( / 10 runs): 203.091 milliseconds. With result of B0.890964 G0.903845 R0.936934
@endcode
In both cases we managed a performance increase of almost 100% compared to the CPU implementation.
It may be just the improvement needed for your application to work. You may observe a runtime
instance of this on the [YouTube here](https://www.youtube.com/watch?v=3_ESXmFlnvY).

\htmlonly
<div align="center">
<iframe title="Similarity check (PNSR and SSIM) on the GPU" width="560" height="349" src="http://www.youtube.com/embed/3_ESXmFlnvY?rel=0&loop=1" frameborder="0" allowfullscreen align="middle"></iframe>
</div>
\endhtmlonly
