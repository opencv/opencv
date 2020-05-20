Using a cv::cuda::GpuMat with thrust {#tutorial_gpu_thrust_interop}
===========================================

@prev_tutorial{tutorial_gpu_basics_similarity}

Goal
----

Thrust is an extremely powerful library for various cuda accelerated algorithms.  However thrust is designed
to work with vectors and not pitched matricies.  The following tutorial will discuss wrapping cv::cuda::GpuMat's
into thrust iterators that can be used with thrust algorithms.

This tutorial should show you how to:
- Wrap a GpuMat into a thrust iterator
- Fill a GpuMat with random numbers
- Sort a column of a GpuMat in place
- Copy values greater than 0 to a new gpu matrix
- Use streams with thrust

Wrapping a GpuMat into a thrust iterator
----

The following code will produce an iterator for a GpuMat

@snippet samples/cpp/tutorial_code/gpu/gpu-thrust-interop/Thrust_interop.hpp begin_itr
@snippet samples/cpp/tutorial_code/gpu/gpu-thrust-interop/Thrust_interop.hpp end_itr

Our goal is to have an iterator that will start at the beginning of the matrix, and increment correctly to access continuous matrix elements.  This is trivial for a continuous row, but how about for a column of a pitched matrix?  To do this we need the iterator to be aware of the matrix dimensions and step.  This information is embedded in the step_functor.
@snippet samples/cpp/tutorial_code/gpu/gpu-thrust-interop/Thrust_interop.hpp step_functor
The step functor takes in an index value and returns the appropriate
offset from the beginning of the matrix.  The counting iterator simply increments over the range of pixel elements.  Combined into the transform_iterator we have an iterator that counts from 0 to M*N and correctly
increments to account for the pitched memory of a GpuMat.  Unfortunately this does not include any memory location information, for that we need a thrust::device_ptr.  By combining a device pointer with the transform_iterator we can point thrust to the first element of our matrix and have it step accordingly.

Fill a GpuMat with random numbers
----
Now that we have some nice functions for making iterators for thrust, lets use them to do some things OpenCV can't do.  Unfortunately at the time of this writing, OpenCV doesn't have any Gpu random number generation.
Thankfully thrust does and it's now trivial to interop between the two.
Example taken from http://stackoverflow.com/questions/12614164/generating-a-random-number-vector-between-0-and-1-0-using-thrust

First we need to write a functor that will produce our random values.
@snippet samples/cpp/tutorial_code/gpu/gpu-thrust-interop/main.cu prg

This will take in an integer value and output a value between a and b.
Now we will populate our matrix with values between 0 and 10 with a thrust transform.
@snippet samples/cpp/tutorial_code/gpu/gpu-thrust-interop/main.cu random

Sort a column of a GpuMat in place
----

Lets fill matrix elements with random values and an index.  Afterwards we will sort the random numbers and the indecies.
@snippet samples/cpp/tutorial_code/gpu/gpu-thrust-interop/main.cu sort

Copy values greater than 0 to a new gpu matrix while using streams
----
In this example we're going to see how cv::cuda::Streams can be used with thrust.  Unfortunately this specific example uses functions that must return results to the CPU so it isn't the optimal use of streams.

@snippet samples/cpp/tutorial_code/gpu/gpu-thrust-interop/main.cu copy_greater


First we will populate a GPU mat with randomly generated data between -1 and 1 on a stream.

@snippet samples/cpp/tutorial_code/gpu/gpu-thrust-interop/main.cu random_gen_stream

Notice the use of thrust::system::cuda::par.on(...), this creates an execution policy for executing thrust code on a stream.
There is a bug in the version of thrust distributed with the cuda toolkit, as of version 7.5 this has not been fixed.  This bug causes code to not execute on streams.
The bug can however be fixed by using the newest version of thrust from the git repository. (http://github.com/thrust/thrust.git)
Next we will determine how many values are greater than 0 by using thrust::count_if with the following predicate:

@snippet samples/cpp/tutorial_code/gpu/gpu-thrust-interop/main.cu pred_greater

We will use those results to create an output buffer for storing the copied values, we will then use copy_if with the same predicate to populate the output buffer.
Lastly we will download the values into a CPU mat for viewing.
