// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_BROADCAST_COMMON_HPP
#define OPENCV_DNN_SRC_BROADCAST_COMMON_HPP

#include "precomp.hpp"

namespace cv
{
namespace dnn
{

size_t to_broadcasted(size_t idx, const std::vector<size_t>& broadcasting_dims,
                                     size_t k,
                                     const std::vector<size_t>& prods);

struct InputCache
{
    std::vector<size_t> broadcast_dims;
    std::vector<size_t> shape_prods;
};

template <typename T, typename K, typename F>
void broadcast(const T input_ptr, const K output_ptr,
               const InputCache& cache, const std::vector<size_t>& prods, const std::vector<int>& outShape,
               F copier)
{
    const auto& broadcast_dims = cache.broadcast_dims;
    const auto& shape_prods = cache.shape_prods;

    const size_t continuous_piece = broadcast_dims.empty() ? prods[0] : prods[broadcast_dims[0] + 1];
    const size_t cap = broadcast_dims.empty() ? shape_prods[0] : shape_prods[broadcast_dims[0]];
    for (size_t j = 0; j < cap; ++j)
    {
        const size_t src_offset = j * continuous_piece;
        const size_t dst_offset = to_broadcasted(src_offset, broadcast_dims, 0, prods);
        copier(input_ptr, src_offset, output_ptr, dst_offset, continuous_piece, 1);
    }

    size_t broad_cap = 0;
    for (const size_t bdim : broadcast_dims)
    {
        const size_t copies_outer = shape_prods[bdim];
        const size_t copies_inner = prods[bdim + 1];
        const size_t memcpys = outShape[bdim];

        for (size_t k = 0; k < copies_outer; ++k)
        {
            const size_t dst_offset = to_broadcasted(k*copies_inner, broadcast_dims, broad_cap, prods);
            copier(output_ptr, dst_offset, output_ptr, dst_offset + copies_inner, copies_inner, memcpys - 1);
        }
        ++broad_cap;
    }
}

}
}

#endif /* OPENCV_DNN_SRC_BROADCAST_COMMON_HPP */
