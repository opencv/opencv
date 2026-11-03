// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_TENSOR_SCATTER_HPP
#define OPENCV_DNN_TENSOR_SCATTER_HPP

#include <opencv2/core.hpp>
#include <vector>
#include <limits>
#include <algorithm>

namespace cv {
namespace dnn {

struct TensorScatterChunk
{
    size_t start;
    size_t len;
};

inline std::vector<TensorScatterChunk> planTensorScatterChunks(size_t total_work, size_t max_chunk_size)
{
    CV_CheckGT(max_chunk_size, (size_t)0, "TensorScatter: max_chunk_size must be positive");
    CV_CheckLE(max_chunk_size, static_cast<size_t>(std::numeric_limits<int>::max()),
               "TensorScatter: max_chunk_size must not exceed int range");

    std::vector<TensorScatterChunk> chunks;
    if (total_work == 0)
        return chunks;

    chunks.reserve((total_work + max_chunk_size - 1) / max_chunk_size);
    for (size_t start = 0; start < total_work; start += max_chunk_size)
    {
        size_t len = std::min(max_chunk_size, total_work - start);
        chunks.push_back({start, len});
    }
    return chunks;
}

} // namespace dnn
} // namespace cv

#endif // OPENCV_DNN_TENSOR_SCATTER_HPP
