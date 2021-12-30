// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "broadcast_common.hpp"

size_t cv::dnn::to_broadcasted(size_t idx, const std::vector<size_t>& broadcasting_dims,
                      const size_t k,
                      const std::vector<size_t>& prods)
{
    for (size_t i = k; i < broadcasting_dims.size(); ++i)
    {
        const auto dim = broadcasting_dims[i];
        const size_t leftovers = idx % prods[dim + 1];
        const auto outShapeDim = prods[dim] / prods[dim + 1];
        idx = (idx - leftovers) * outShapeDim + leftovers;
    }
    return idx;
}
