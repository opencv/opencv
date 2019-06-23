// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_KERNEL_UTILS_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_KERNEL_UTILS_HPP

#include "error.hpp"
#include "stream.hpp"
#include "nvcc_defs.hpp"

#include <cstddef>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

#ifdef __CUDACC__
    struct execution_policy {
        execution_policy(dim3 grid_size, dim3 block_size)
            : grid{ grid_size }, block{ block_size }, sharedMem{ 0 }, stream{ 0 } { }

        execution_policy(dim3 grid_size, dim3 block_size, std::size_t shared_mem)
            : grid{ grid_size }, block{ block_size }, sharedMem{ shared_mem }, stream{ nullptr } { }

        execution_policy(dim3 grid_size, dim3 block_size, const Stream& strm)
            : grid{ grid_size }, block{ block_size }, sharedMem{ 0 }, stream{ StreamAccessor::get(strm) } { }

        execution_policy(dim3 grid_size, dim3 block_size, std::size_t shared_mem, const Stream& strm)
            : grid{ grid_size }, block{ block_size }, sharedMem{ shared_mem }, stream{ StreamAccessor::get(strm) } { }

        dim3 grid;
        dim3 block;
        std::size_t sharedMem;
        cudaStream_t stream;
    };

    template <class Kernel> inline
    execution_policy make_policy(Kernel kernel, std::size_t sharedMem = 0, const Stream& stream = 0) {
        int grid_size, block_size;
        CUDA4DNN_CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, kernel, sharedMem));
        return execution_policy(grid_size, block_size, sharedMem, stream);
    }

    template <class Kernel, typename ...Args> inline
    void launch_kernel(Kernel kernel, Args ...args) {
        auto policy = make_policy(kernel);
        kernel <<<policy.grid, policy.block>>> (std::forward<Args>(args)...);
    }

    template <class Kernel, typename ...Args> inline
    void launch_kernel(Kernel kernel, dim3 grid, dim3 block, Args ...args) {
        kernel <<<grid, block>>> (std::forward<Args>(args)...);
    }

    template <class Kernel, typename ...Args> inline
    void launch_kernel(Kernel kernel, execution_policy policy, Args ...args) {
        kernel <<<policy.grid, policy.block, policy.sharedMem, policy.stream>>> (std::forward<Args>(args)...);
    }

    template <int> CUDA4DNN_DEVICE auto getGridDim()->decltype(dim3::x);
    template <> inline CUDA4DNN_DEVICE auto getGridDim<0>()->decltype(dim3::x) { return gridDim.x; }
    template <> inline CUDA4DNN_DEVICE auto getGridDim<1>()->decltype(dim3::x) { return gridDim.y; }
    template <> inline CUDA4DNN_DEVICE auto getGridDim<2>()->decltype(dim3::x) { return gridDim.z; }

    template <int> CUDA4DNN_DEVICE auto getBlockDim()->decltype(dim3::x);
    template <> inline CUDA4DNN_DEVICE auto getBlockDim<0>()->decltype(dim3::x) { return blockDim.x; }
    template <> inline CUDA4DNN_DEVICE auto getBlockDim<1>()->decltype(dim3::x) { return blockDim.y; }
    template <> inline CUDA4DNN_DEVICE auto getBlockDim<2>()->decltype(dim3::x) { return blockDim.z; }

    template <int> CUDA4DNN_DEVICE auto getBlockIdx()->decltype(uint3::x);
    template <> inline CUDA4DNN_DEVICE auto getBlockIdx<0>()->decltype(uint3::x) { return blockIdx.x; }
    template <> inline CUDA4DNN_DEVICE auto getBlockIdx<1>()->decltype(uint3::x) { return blockIdx.y; }
    template <> inline CUDA4DNN_DEVICE auto getBlockIdx<2>()->decltype(uint3::x) { return blockIdx.z; }

    template <int> CUDA4DNN_DEVICE auto getThreadIdx()->decltype(uint3::x);
    template <> inline CUDA4DNN_DEVICE auto getThreadIdx<0>()->decltype(uint3::x) { return threadIdx.x; }
    template <> inline CUDA4DNN_DEVICE auto getThreadIdx<1>()->decltype(uint3::x) { return threadIdx.y; }
    template <> inline CUDA4DNN_DEVICE auto getThreadIdx<2>()->decltype(uint3::x) { return threadIdx.z; }

    template <int dim>
    class grid_stride_range_generic {
    public:
        CUDA4DNN_DEVICE grid_stride_range_generic(std::size_t to_) : from(0), to(to_) { }
        CUDA4DNN_DEVICE grid_stride_range_generic(std::size_t from_, std::size_t to_) : from(from_), to(to_) { }

        class iterator
        {
        public:
            CUDA4DNN_DEVICE iterator(std::size_t pos_) : pos(pos_) {}

            CUDA4DNN_DEVICE size_t operator*() const { return pos; }

            CUDA4DNN_DEVICE iterator& operator++() {
                pos += getGridDim<dim>() * getBlockDim<dim>();
                return *this;
            }

            CUDA4DNN_DEVICE bool operator!=(const iterator& other) const {
                /* NOTE HACK
                ** 'pos' can move in large steps (see operator++)
                ** expansion of range for loop uses != as the loop conditioion
                ** => operator!= must return false if 'pos' crosses the end
                */
                return pos < other.pos;
            }

        private:
            std::size_t pos;
        };

        CUDA4DNN_DEVICE iterator begin() const {
            return iterator(from + getBlockDim<dim>() * getBlockIdx<dim>() + getThreadIdx<dim>());
        }

        CUDA4DNN_DEVICE iterator end() const {
            return iterator(to);
        }

    private:
        std::size_t from, to;
    };

    using grid_stride_range_x = grid_stride_range_generic<0>;
    using grid_stride_range_y = grid_stride_range_generic<1>;
    using grid_stride_range_z = grid_stride_range_generic<2>;
    using grid_stride_range = grid_stride_range_x;

#endif /* __CUDACC__ */

}}}} /* cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_KERNEL_UTILS_HPP */
