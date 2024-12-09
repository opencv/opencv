// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA_EXECUTION_HPP
#define OPENCV_DNN_SRC_CUDA_EXECUTION_HPP

#include "../cuda4dnn/csl/error.hpp"
#include "../cuda4dnn/csl/stream.hpp"

#include <opencv2/core.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    struct execution_policy {
        execution_policy(dim3 grid_size, dim3 block_size)
            : grid{ grid_size }, block{ block_size }, sharedMem{ 0 }, stream{ 0 } { }

        execution_policy(dim3 grid_size, dim3 block_size, std::size_t shared_mem)
            : grid{ grid_size }, block{ block_size }, sharedMem{ shared_mem }, stream{ nullptr } { }

        execution_policy(dim3 grid_size, dim3 block_size, const Stream& strm)
            : grid{ grid_size }, block{ block_size }, sharedMem{ 0 }, stream{ strm.get() } { }

        execution_policy(dim3 grid_size, dim3 block_size, std::size_t shared_mem, const Stream& strm)
            : grid{ grid_size }, block{ block_size }, sharedMem{ shared_mem }, stream{ strm.get() } { }

        dim3 grid;
        dim3 block;
        std::size_t sharedMem;
        cudaStream_t stream;
    };

    /* this overload shouldn't be necessary; we should always provide a bound on the number of threads */
    /*
    template <class Kernel> inline
    execution_policy make_policy(Kernel kernel, std::size_t sharedMem = 0, const Stream& stream = 0) {
        int grid_size, block_size;
        CUDA4DNN_CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, kernel, sharedMem));
        return execution_policy(grid_size, block_size, sharedMem, stream);
    }*/

    template <class Kernel> inline
    execution_policy make_policy(Kernel kernel, std::size_t max_threads, std::size_t sharedMem = 0, const Stream& stream = 0) {
        CV_Assert(max_threads > 0);

        int grid_size = 0, block_size = 0;
        CUDA4DNN_CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, kernel, sharedMem));
        if (grid_size * block_size > max_threads) {
            grid_size = (max_threads + block_size - 1) / block_size;
            if (block_size > max_threads)
                block_size = max_threads;
        }

        CV_Assert(grid_size >= 1 && block_size >= 1);
        return execution_policy(grid_size, block_size, sharedMem, stream);
    }

    template <class Kernel, typename ...Args> inline
    void launch_kernel(Kernel kernel, Args ...args) {
        auto policy = make_policy(kernel);
        kernel <<<policy.grid, policy.block>>> (args...);
    }

    template <class Kernel, typename ...Args> inline
    void launch_kernel(Kernel kernel, dim3 grid, dim3 block, Args ...args) {
        kernel <<<grid, block>>> (args...);
    }

    template <class Kernel, typename ...Args> inline
    void launch_kernel(Kernel kernel, execution_policy policy, Args ...args) {
        kernel <<<policy.grid, policy.block, policy.sharedMem, policy.stream>>> (args...);
    }

}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_SRC_CUDA_EXECUTION_HPP */
