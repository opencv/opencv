// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "math.hpp"
#include "bbox_utils.hpp"
#include "grid_stride_range.hpp"
#include "block_stride_range.hpp"
#include "execution.hpp"
#include "vector_traits.hpp"
#include "memory.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"
#include "../cuda4dnn/csl/tensor.hpp"

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

namespace raw {

    template <class T, bool SHARE_LOCATION, bool VARIANCE_ENCODED_IN_TARGET, bool CORNER_TRUE_CENTER_FALSE, bool CLIP_BBOX>
    __global__ void decode_bbox(Span<T> decoded_bboxes, View<T> locations, View<T> priors,
        bool transpose_location, bool normalized_bbox,
        size_type num_loc_classes, index_type background_class_id,
        float clip_width, float clip_height)
    {
        // decoded_bboxes: [batch_size, num_priors, num_loc_classes, 4]
        // locations: [batch_size, num_priors, num_loc_classes, 4]
        // priors: [1, C, num_priors, 4]
        // C = 2 if !VARIANCE_ENCODED_IN_TARGET; otherwise, 1

        /* 4 bbox values + 4 variance values per prior */
        constexpr int PRIOR_BOX_SIZE = VARIANCE_ENCODED_IN_TARGET ? 4 : 8;
        const size_type num_priors = priors.size() / PRIOR_BOX_SIZE;

        using vector_type = get_vector_type_t<T, 4>;
        auto locations_vPtr = vector_type::get_pointer(locations.data());
        auto priors_vPtr = vector_type::get_pointer(priors.data());
        auto decoded_bboxes_vPtr = vector_type::get_pointer(decoded_bboxes.data());

        const auto boxes_per_batch = num_priors * num_loc_classes;
        for (auto idx : grid_stride_range(decoded_bboxes.size() / 4))
        {
            index_type p;
            index_type c;

            if (SHARE_LOCATION)
            {
                // locations are shared across all classes => num_loc_classes = 1
                p = idx % boxes_per_batch;
                c = 0;
            }
            else
            {
                p = (idx % boxes_per_batch) / num_loc_classes;
                c = idx % num_loc_classes;
            }

            if (!SHARE_LOCATION && c == background_class_id)
                continue;

            BoundingBox bbox;
            {
                vector_type location;
                v_load(location, locations_vPtr[idx]);

                if (transpose_location)
                {
                    bbox.ymin = location.data[0];
                    bbox.xmin = location.data[1];
                    bbox.ymax = location.data[2];
                    bbox.xmax = location.data[3];
                }
                else
                {
                    bbox.xmin = location.data[0];
                    bbox.ymin = location.data[1];
                    bbox.xmax = location.data[2];
                    bbox.ymax = location.data[3];
                }
            }

            if (!VARIANCE_ENCODED_IN_TARGET)
            {
                vector_type prior_variance;
                v_load_ldg(prior_variance, priors_vPtr[num_priors + p]);

                bbox.xmin *= static_cast<float>(prior_variance.data[0]);
                bbox.ymin *= static_cast<float>(prior_variance.data[1]);
                bbox.xmax *= static_cast<float>(prior_variance.data[2]);
                bbox.ymax *= static_cast<float>(prior_variance.data[3]);
            }

            BoundingBox prior;
            {
                vector_type prior_box;
                v_load_ldg(prior_box, priors_vPtr[p]);

                prior.xmin = prior_box.data[0];
                prior.ymin = prior_box.data[1];
                prior.xmax = prior_box.data[2];
                prior.ymax = prior_box.data[3];
            }

            BoundingBox decoded_bbox;
            if (CORNER_TRUE_CENTER_FALSE)
            {
                decoded_bbox.xmin = prior.xmin + bbox.xmin;
                decoded_bbox.ymin = prior.ymin + bbox.ymin;
                decoded_bbox.xmax = prior.xmax + bbox.xmax;
                decoded_bbox.ymax = prior.ymax + bbox.ymax;
            }
            else
            {
                auto prior_width = prior.xmax - prior.xmin;
                auto prior_height = prior.ymax - prior.ymin;
                if (!normalized_bbox)
                {
                    prior_width += 1;
                    prior_height += 1;
                }

                auto prior_center_x = prior.xmin + prior_width * 0.5f;
                auto prior_center_y = prior.ymin + prior_height * 0.5f;

                auto decode_bbox_center_x = bbox.xmin * prior_width + prior_center_x;
                auto decode_bbox_center_y = bbox.ymin * prior_height + prior_center_y;

                using device::exp;
                float decode_bbox_width = exp(bbox.xmax) * prior_width;
                float decode_bbox_height = exp(bbox.ymax) * prior_height;

                decoded_bbox.xmin = decode_bbox_center_x - decode_bbox_width * 0.5f;
                decoded_bbox.ymin = decode_bbox_center_y - decode_bbox_height * 0.5f;
                decoded_bbox.xmax = decode_bbox_center_x + decode_bbox_width * 0.5f;
                decoded_bbox.ymax = decode_bbox_center_y + decode_bbox_height * 0.5f;
            }

            vector_type decoded_bbox_vec;
            if (CLIP_BBOX)
            {
                decoded_bbox_vec.data[0] = clamp(decoded_bbox.xmin, 0.0f, clip_width);
                decoded_bbox_vec.data[1] = clamp(decoded_bbox.ymin, 0.0f, clip_height);
                decoded_bbox_vec.data[2] = clamp(decoded_bbox.xmax, 0.0f, clip_width);
                decoded_bbox_vec.data[3] = clamp(decoded_bbox.ymax, 0.0f, clip_height);
            }
            else
            {
                decoded_bbox_vec.data[0] = decoded_bbox.xmin;
                decoded_bbox_vec.data[1] = decoded_bbox.ymin;
                decoded_bbox_vec.data[2] = decoded_bbox.xmax;
                decoded_bbox_vec.data[3] = decoded_bbox.ymax;
            }

            v_store(decoded_bboxes_vPtr[idx], decoded_bbox_vec);
        }
    }

    template <class T, int BINS, int BLOCK_SIZE>
    __launch_bounds__(BLOCK_SIZE)
    __global__ void findTopK(Span<int> indices_, Span<int> count_, View<T> scores_, float threshold, size_type classwise_topK, size_type num_classes, size_type num_priors, index_type background_class_id)
    {
        /* We need to sort boxes based on their confidence scores. The confidence scores fall in
         * the range [0.0, 1.0]. We break the range into bins and perform count sort. This is an
         * approximate algorithm.
         *
         * Each block handles a particular class of a particular batch item.
         */
        const auto c = blockIdx.x;
        const auto b = blockIdx.y;

        if (c == background_class_id)
            return;

        // indices: [batch_size, num_classes, classwise_topK]
        // count: [batch_size, num_classes]
        // scores: [batch_size, num_classes, num_priors]

        auto count = count_.data() + b * num_classes + c;
        auto scores = scores_.data() + (b * num_classes + c) * num_priors;
        auto indices = indices_.data() + (b * num_classes + c) * classwise_topK;

        /* We do not require a large number of bins to find the top K confidence scores. We will use
         * a reasonable number of bins which will fit in the shared memory.
         *
         * Note that smaller scores will have a smaller index, i.e. the `bins` are ordered in
         * ascending order.
         */

        __shared__ int bins[BINS];

        #pragma unroll
        for (int unroll = 0; unroll < BINS / BLOCK_SIZE; unroll++)
            bins[unroll * BLOCK_SIZE + threadIdx.x] = 0;

        __syncthreads();

        for (auto i : block_stride_range<BLOCK_SIZE>(num_priors))
        {
            const float confidence = load_ldg(scores[i]);
            if (confidence > threshold)
            {
                using device::fast_divide_ftz;
                auto conf_scaled = fast_divide_ftz(confidence - threshold, 1 - threshold);

                using device::clamp;
                int bin_index = conf_scaled * BINS;

                /* We store counts of confidence scores in the bins. Our ultimate goal is to store the indices
                 * of the `classwise_topK` confidence values in the `indices` array.
                 *
                 * We use a little trick to parallelize the process of filling up the `indices` array.
                 * We want every thread in the block to participate in the process. To do so, we want the
                 * bins array to be shifted by one place to the left. We will be computing the suffix sum
                 * of the bins array later. Details and reasons for doing so will be explained later.
                 */
                bin_index = clamp<int>(bin_index, 0, BINS - 1) - 1; // shift left by one

                if (bin_index >= 0)
                    atomicAdd(&bins[bin_index], 1);
            }
        }

        __syncthreads();

        constexpr int WARP_SIZE = 32; /* must be equal to warpSize */
        // FORWARD_COMPATIBILITY_TAG: WARP_SIZE_DEPENDENT_CODE

        if (threadIdx.x < WARP_SIZE)
        {
            /* We can compute suffix sum of an array in groups of N numbers.
             * Let N be 4 for this example.
             *
             * 1) Last 4 numbers
             *                      1   2   3   4   |   5   6   7   8   |   9   10  11  12
             * group suffix sum:                                            42  33  23  12
             *
             * 2) Middle 4 numbers
             *                      1   2   3   4   |   5   6   7   8   |   9   10  11  12
             * group suffix sum:                    |   26  21  15  8   |
             *
             * We add `42` (first element in the previous group) to each element to get:
             *
             *                      1   2   3   4   |   5   6   7   8   |   9   10  11  12
             *                                      |   68  63  57  50  |   42  33  23  12
             * 3) First 4 numbers
             *
             *                      1   2   3   4   |   5   6   7   8   |   9   10  11  12
             * group suffix sum:    10  9   7   4   |
             *
             * We add `68` (first element in the previous group) to each element to get:
             *
             *                      1   2   3   4   |   5   6   7   8   |   9   10  11  12
             * group suffix sum:    78  77  75  72  |   68  63  57  50  |   42  33  23  12
             *
             * What we are left with now is the suffix sum of the entire array.
             *
             * We use the aforementioned logic in the code below but work in groups of `warpSize`.
             */

            /* We calculate suffix sums WARP_SIZE elements at a time starting from the right end.
             * Hence, we will need BINS / WARP_SIZE number of iterations.
             *
             * Each iteration uses shuffle instructions to exchange data between threads. Shuffle
             * instructions cannot be used in warp-divergent code. If the bins are a multiple of
             * the warpSize, all the threads in the warp will participate.
             */
            static_assert(BINS % WARP_SIZE == 0, "number of bins must be a multiple of warp size");

            const int thread_id = threadIdx.x;
            const int inverse_lane_id = WARP_SIZE - thread_id - 1;

            int previous_group_first_element = 0;
            for (int iter = BINS / WARP_SIZE - 1; iter >= 0; iter--)
            {
                const index_type idx = iter * WARP_SIZE + thread_id;
                auto value = bins[idx];

                for (int i = 1; i < WARP_SIZE; i *= 2)
                {
                    auto n = __shfl_down_sync(0xFFFFFFFF, value, i);
                    if (inverse_lane_id >= i)
                        value += n;
                }

                value += previous_group_first_element;
                bins[idx] = value;

                previous_group_first_element = __shfl_sync(0xFFFFFFFF, value, 0);
            }
        }

        if (threadIdx.x == 0)
            *count = 0;

        __syncthreads();

        for (auto i : block_stride_range<BLOCK_SIZE>(num_priors))
        {
            const float confidence = load_ldg(scores[i]);
            if (confidence > threshold)
            {
                using device::fast_divide_ftz;
                auto conf_scaled = fast_divide_ftz(confidence - threshold, 1 - threshold);

                int bin_index = conf_scaled * BINS;
                bin_index = clamp<int>(bin_index, 0, BINS - 1);

                /* This bounding box is eligible to be selected unless it does not fall in
                 * the `classwise_topK`. If it did, we would have to compute the location where it needs
                 * to be stored.
                 *
                 * Suppose we had just 4 bins and say the following were the counts:
                 * BIN0 2
                 * BIN1 1
                 * BIN2 3
                 * BIN3 0 (last bin is always zero as we shift left by one while populating the bins)
                 *
                 * We will try our best to store the boxes in a sorted order in the `indices` array.
                 * This requires that the boxes in later bins (higher confidence scores) must be
                 * stored earlier.
                 *
                 * We compute the suffix sum of the array. This gives us:
                 * BIN0 6
                 * BIN1 4
                 * BIN2 3
                 * BIN3 0
                 *
                 * The bins now give us the location in the `indices` array from which the indices of the
                 * scores corresponding to that bin would be stored. We atomically increment the bin count
                 * everytime we store a box corresponding to that bin. Therefore, the value in the bins
                 * gives the index in the `indices` array where the next box corresponding to that bin  must
                 * be put.
                 */

                const index_type idx = atomicAdd(&bins[bin_index], 1);
                if (idx < classwise_topK)
                {
                    indices[idx] = i;
                    atomicAdd(&count[0], 1);
                }
            }
        }
    }

    template <class T>
    __global__ void box_collect(Span<T> collected_bboxes_, View<T> decoded_bboxes_, View<int> indices_, View<int> count_, bool share_location, size_type num_priors, size_type num_classes, size_type classwise_topK, index_type background_class_id)
    {
        const index_type c = blockIdx.x;
        if (c == background_class_id)
            return;

        const index_type b = blockIdx.y;

        // collected_bboxes: [batch_size, num_classes, classwise_topK, 4]
        // decoded_bboxes: [batch_size, num_priors, num_loc_classes, 4]
        // indices: [batch_size, num_classes, classwise_topK]
        // count: [batch_size, num_classes]

        const auto num_loc_classes = share_location ? 1 : num_classes;

        auto collected_bboxes = collected_bboxes_.data() + (b * num_classes + c) * classwise_topK * 4;
        auto decoded_bboxes = decoded_bboxes_.data() + b * num_priors * num_loc_classes * 4;
        auto indices = indices_.data() + (b * num_classes + c) * classwise_topK;
        auto count = count_.data() + b * num_classes + c;

        const auto boxes = load_ldg(&count[0]);
        if (boxes == 0)
            return;

        using vector_type = get_vector_type_t<T, 4>;
        auto decoded_bboxes_vPtr = vector_type::get_pointer(decoded_bboxes);
        auto collected_bboxes_vPtr = vector_type::get_pointer(collected_bboxes);

        for (auto i : block_stride_range<>(boxes))
        {
            const auto prior_id = indices[i];
            const index_type idx = share_location ? prior_id : (prior_id * num_classes + c);

            vector_type box;
            v_load(box, decoded_bboxes_vPtr[idx]);
            v_store(collected_bboxes_vPtr[i], box);
        }
    }

    template <class T, bool NORMALIZED_BBOX>
    __global__ void blockwise_class_nms(Span<int> indices_, Span<int> count_, View<T> collected_bboxes_, size_type num_classes, size_type classwise_topK, index_type background_class_id, float nms_threshold)
    {
        const index_type b = blockIdx.x / num_classes;
        const index_type c = blockIdx.x % num_classes;
        if (c == background_class_id)
            return;

        // indices: [batch_size, num_classes, classwise_topK]
        // count: [batch_size, num_classes]
        // collected_bboxes: [batch_size, num_classes, classwise_topK, 4]

        auto indices = indices_.data() + (b * num_classes + c) * classwise_topK;
        auto count = count_.data() + b * num_classes + c;
        auto collected_bboxes = collected_bboxes_.data() + (b * num_classes + c) * classwise_topK * 4;

        const auto boxes = count[0];
        if (boxes == 0)
            return;

        using vector_type = get_vector_type_t<T, 4>;
        auto collected_bboxes_vPtr = vector_type::get_pointer(collected_bboxes);

        for (int i = 0; i < boxes; i++)
        {
            auto prior_id = indices[i];
            if (prior_id != -1)
            {
                BoundingBox bbox1;
                {
                    vector_type box;
                    v_load(box, collected_bboxes_vPtr[i]);

                    bbox1.xmin = box.data[0];
                    bbox1.ymin = box.data[1];
                    bbox1.xmax = box.data[2];
                    bbox1.ymax = box.data[3];
                }

                for (auto j : block_stride_range<>(i + 1, boxes))
                {
                    prior_id = indices[j];
                    if (prior_id == -1)
                        continue;

                    BoundingBox bbox2;
                    {
                        vector_type box;
                        v_load_ldg(box, collected_bboxes_vPtr[j]);

                        bbox2.xmin = box.data[0];
                        bbox2.ymin = box.data[1];
                        bbox2.xmax = box.data[2];
                        bbox2.ymax = box.data[3];
                    }

                    using device::min;
                    using device::max;

                    BoundingBox intersect_bbox;
                    intersect_bbox.xmin = max(bbox1.xmin, bbox2.xmin);
                    intersect_bbox.ymin = max(bbox1.ymin, bbox2.ymin);
                    intersect_bbox.xmax = min(bbox1.xmax, bbox2.xmax);
                    intersect_bbox.ymax = min(bbox1.ymax, bbox2.ymax);

                    float intersect_size = compute_bbox_size<NORMALIZED_BBOX>(intersect_bbox);
                    float bbox1_size = compute_bbox_size<NORMALIZED_BBOX>(bbox1);
                    float bbox2_size = compute_bbox_size<NORMALIZED_BBOX>(bbox2);

                    using device::fast_divide_ftz;
                    float iou = fast_divide_ftz(intersect_size, bbox1_size + bbox2_size - intersect_size);
                    if (iou > nms_threshold)
                        indices[j] = -1;
                }
            }

            __syncthreads();
        }

        if (threadIdx.x == 0)
            count[0] = 0;

        __syncthreads();

        for (auto i : block_stride_range<>(boxes))
        {
            auto prior_id = indices[i];
            if(prior_id != -1)
            {
                const index_type idx = atomicAdd(&count[0], 1);
                indices[idx] = prior_id;
            }
        }
    }

    template <class T, std::size_t BINS, int BLOCK_SIZE>
    __launch_bounds__(BLOCK_SIZE)
    __global__ void nms_collect(
        Span<int> kept_indices, Span<int> kept_count, View<int> indices_, View<int> count, View<T> scores_, float threshold,
        size_type num_classes, size_type num_priors, size_type classwise_topK, size_type keepTopK, index_type background_class_id)
    {
        // sorting algorithm is documented in detail in findTopK kernel comments
        // no explanations are provided here

        // kept_indices: [batch_size, keepTopK]
        // kept_count: [batch_size]

        const auto b = blockIdx.x;

        __shared__ int bins[BINS];

        #pragma unroll
        for (int unroll = 0; unroll < BINS / BLOCK_SIZE; unroll++)
            bins[unroll * BLOCK_SIZE + threadIdx.x] = 0;

        __syncthreads();

        for (int c = 0; c < num_classes; c++)
        {
            if (c == background_class_id)
                continue;

            // indices: [batch_size, num_classes, classwise_topK]
            // count: [batch_size, num_classes]
            // scores: [batch_size, num_classes, num_priors]

            const auto indices = indices_.data() + (b * num_classes + c) * classwise_topK;
            const auto scores = scores_.data() + (b * num_classes + c) * num_priors;

            auto boxes = count[b * num_classes + c];

            for (auto i : block_stride_range<BLOCK_SIZE>(boxes))
            {
                auto prior_id = indices[i];
                const float confidence = load_ldg(scores[prior_id]);
                if (confidence > threshold)
                {
                    using device::fast_divide_ftz;
                    auto conf_scaled = fast_divide_ftz(confidence - threshold, 1 - threshold);

                    using device::clamp;
                    int bin_index = conf_scaled * BINS;
                    bin_index = clamp<int>(bin_index, 0, BINS - 1) - 1; // shift left by one

                    if (bin_index >= 0)
                        atomicAdd(&bins[bin_index], 1);
                }
            }
        }

        __syncthreads();

        constexpr int WARP_SIZE = 32; /* must be equal to warpSize */
        // FORWARD_COMPATIBILITY_TAG: WARP_SIZE_DEPENDENT_CODE

        if (threadIdx.x < WARP_SIZE)
        {
            static_assert(BINS % WARP_SIZE == 0, "number of bins must be a multiple of warp size");

            const int thread_id = threadIdx.x;
            const int inverse_lane_id = WARP_SIZE - thread_id - 1;

            int previous_group_first_element = 0;
            for (int iter = BINS / WARP_SIZE - 1; iter >= 0; iter--)
            {
                const index_type idx = iter * WARP_SIZE + thread_id;
                auto value = bins[idx];

                for (int i = 1; i < WARP_SIZE; i *= 2)
                {
                    auto n = __shfl_down_sync(0xFFFFFFFF, value, i);
                    if (inverse_lane_id >= i)
                        value += n;
                }

                value += previous_group_first_element;
                bins[idx] = value;

                previous_group_first_element = __shfl_sync(0xFFFFFFFF, value, 0);
            }
        }

        if (threadIdx.x == 0)
            kept_count[b] = 0;

        __syncthreads();

        for (int c = 0; c < num_classes; c++)
        {
            if (c == background_class_id)
                continue;

            const auto indices = indices_.data() + (b * num_classes + c) * classwise_topK;
            const auto scores = scores_.data() + (b * num_classes + c) * num_priors;

            auto boxes = count[b * num_classes + c];

            for (auto i : block_stride_range<BLOCK_SIZE>(boxes))
            {
                auto prior_id = indices[i];
                const float confidence = load_ldg(scores[prior_id]);
                if (confidence > threshold)
                {
                    using device::fast_divide_ftz;
                    auto conf_scaled = fast_divide_ftz(confidence - threshold, 1 - threshold);

                    using device::clamp;
                    int bin_index = conf_scaled * BINS;
                    bin_index = clamp<int>(bin_index, 0, BINS - 1);

                    const index_type idx = atomicAdd(&bins[bin_index], 1);
                    if (idx < keepTopK)
                    {
                        kept_indices[b * keepTopK + idx] = c * num_priors + prior_id;
                        atomicAdd(&kept_count[b], 1);
                    }
                }
            }
        }
    }

    template <class T>
    __global__ void consolidate_detections(Span<T> output,
        View<int> kept_indices, View<int> kept_count, View<T> decoded_bboxes, View<T> scores, bool share_location,
        size_type batch_size, size_type num_classes, size_type num_priors, size_type keepTopK, DevicePtr<int> num_detections)
    {
        using vector_type = get_vector_type_t<T, 4>;
        auto decoded_bboxes_vPtr = vector_type::get_pointer(decoded_bboxes.data());

        // output: [1, 1, batch_size * keepTopK, 7]
        // kept_indices: [batch_size, keepTopK]
        // kept_count: [batch_size]
        // decoded_bboxes: [batch_size, num_priors, num_loc_classes, 4]
        // scores: [batch_size, num_classes, num_priors]

        for (int b = 0; b < batch_size; b++)
        {
            for (auto i : grid_stride_range(kept_count[b]))
            {
                auto score_id = kept_indices[b * keepTopK + i];
                auto c = score_id / num_priors;
                auto prior_id = score_id % num_priors;

                const auto confidence = scores[b * num_classes * num_priors + score_id];

                index_type bbox_id;
                if (share_location)
                {
                    // decoded_bboxes: [batch_size, num_priors, 1, 4]
                    bbox_id = b * num_priors + prior_id;
                }
                else
                {
                    // decoded_bboxes: [batch_size, num_priors, num_classes, 4]
                    bbox_id = (b * num_priors + prior_id) * num_classes + c;
                }

                vector_type bbox;
                v_load(bbox, decoded_bboxes_vPtr[bbox_id]);

                auto output_id = atomicAdd(num_detections.get(), 1);
                output[output_id * 7 + 0] = b;
                output[output_id * 7 + 1] = c;
                output[output_id * 7 + 2] = confidence;
                output[output_id * 7 + 3] = bbox.data[0];
                output[output_id * 7 + 4] = bbox.data[1];
                output[output_id * 7 + 5] = bbox.data[2];
                output[output_id * 7 + 6] = bbox.data[3];
            }
        }
    }
}

template <class T, bool SHARE_LOCATION, bool VARIANCE_ENCODED_IN_TARGET, bool CORNER_TRUE_CENTER_FALSE, bool CLIP_BBOX> static
void launch_decode_boxes_kernel(const Stream& stream, Span<T> decoded_bboxes, View<T> locations, View<T> priors,
    bool transpose_location, bool normalized_bbox,
    size_type num_loc_classes, index_type background_class_id,
    float clip_width, float clip_height)
{
    auto kernel = raw::decode_bbox<T, SHARE_LOCATION, VARIANCE_ENCODED_IN_TARGET, CORNER_TRUE_CENTER_FALSE, CLIP_BBOX>;
    auto policy = make_policy(kernel, decoded_bboxes.size() / 4, 0, stream);
    launch_kernel(kernel, policy, decoded_bboxes, locations, priors, transpose_location, normalized_bbox, num_loc_classes, background_class_id, clip_width, clip_height);
}

template <class T, unsigned int current, class ...Args> static
typename std::enable_if<current == 0, void>
::type dispatch_decode_bboxes(int selector, Args&& ...args) {
    if(selector == 0)
        launch_decode_boxes_kernel<T, 0, 0, 0, 0>(std::forward<Args>(args)...);
}

template <class T, unsigned int current, class ...Args> static
typename std::enable_if<current != 0, void>
::type dispatch_decode_bboxes(int selector, Args&& ...args) {
    if(selector == current)
        launch_decode_boxes_kernel<T,
                static_cast<bool>(current & 8),
                static_cast<bool>(current & 4),
                static_cast<bool>(current & 2),
                static_cast<bool>(current & 1)>(std::forward<Args>(args)...);
    else
        dispatch_decode_bboxes<T, current - 1, Args...>(selector, std::forward<Args>(args)...);
}

template <class T>
void decode_bboxes(const Stream& stream, Span<T> output, View<T> locations, View<T> priors,
    std::size_t num_loc_classes,
    bool share_location, std::size_t background_class_id,
    bool transpose_location, bool variance_encoded_in_target,
    bool corner_true_or_center_false, bool normalized_bbox,
    bool clip_box, float clip_width, float clip_height)
{
    /* `config` combines three kernel template options into one number using which a bit of TMP code can
     * run through all possible combinations and instantiate the correct template
     */
    unsigned int config = (share_location << 3 | variance_encoded_in_target << 2 | corner_true_or_center_false << 1 | clip_box);
    dispatch_decode_bboxes<T, 15>(config, stream, output, locations, priors, transpose_location, normalized_bbox, num_loc_classes, background_class_id, clip_width, clip_height);
}

template void decode_bboxes(const Stream&, Span<__half>, View<__half>, View<__half>, std::size_t, bool, std::size_t, bool, bool, bool, bool, bool, float, float);
template void decode_bboxes(const Stream&, Span<float>, View<float>, View<float>, std::size_t, bool, std::size_t, bool, bool, bool, bool, bool, float, float);

template <class T>
void findTopK(const Stream& stream, TensorSpan<int> indices, TensorSpan<int> count, TensorView<T> scores, std::size_t background_class_id, float threshold)
{
    // indices: [batch_size, num_classes, classwise_topK]
    // count: [batch_size, num_classes]
    // scores: [batch_size, num_classes, num_priors]

    const auto batch_size = indices.get_axis_size(0);
    CV_Assert(count.get_axis_size(0) == batch_size);
    CV_Assert(scores.get_axis_size(0) == batch_size);

    const auto num_classes = indices.get_axis_size(1);
    CV_Assert(count.get_axis_size(1) == num_classes);
    CV_Assert(scores.get_axis_size(1) == num_classes);

    const auto classwise_topK = indices.get_axis_size(2);
    const auto num_priors = scores.get_axis_size(2);

    /* each block processes one class from each batch */
    constexpr auto BLOCK_SIZE = 256;

    dim3 grid_size(num_classes, batch_size);
    dim3 block_size(BLOCK_SIZE);
    auto policy = execution_policy(grid_size, block_size, stream);

    auto kernel = raw::findTopK<T, 2048, BLOCK_SIZE>;
    launch_kernel(kernel, policy, indices, count, scores, threshold, classwise_topK, num_classes, num_priors, background_class_id);
}

template void findTopK(const Stream&, TensorSpan<int>, TensorSpan<int>, TensorView<__half>, std::size_t, float);
template void findTopK(const Stream&, TensorSpan<int>, TensorSpan<int>, TensorView<float>, std::size_t, float);

template <class T>
void box_collect(const Stream& stream, TensorSpan<T> collected_bboxes, TensorView<T> decoded_bboxes, TensorView<int> indices, TensorView<int> count, bool share_location, std::size_t background_class_id)
{
    // collected_bboxes: [batch_size, num_classes, classwise_topK, 4]
    // decoded_bboxes: [batch_size, num_priors, num_loc_classes, 4]
    // indices: [batch_size, num_classes, classwise_topK]
    // count: [batch_size, num_classes]

    const auto batch_size = collected_bboxes.get_axis_size(0);
    CV_Assert(decoded_bboxes.get_axis_size(0) == batch_size);
    CV_Assert(indices.get_axis_size(0) == batch_size);
    CV_Assert(count.get_axis_size(0) == batch_size);

    const auto num_classes = collected_bboxes.get_axis_size(1);
    CV_Assert(indices.get_axis_size(1) == num_classes);
    CV_Assert(count.get_axis_size(1) == num_classes);

    const auto classwise_topK = collected_bboxes.get_axis_size(2);
    CV_Assert(indices.get_axis_size(2) == classwise_topK);

    const auto num_priors = decoded_bboxes.get_axis_size(1);

    CV_Assert(!share_location || decoded_bboxes.get_axis_size(2) == 1);

    constexpr int BLOCK_SIZE = 256;

    /* each block processes one class from each batch */
    dim3 grid_size(num_classes, batch_size);
    dim3 block_size(BLOCK_SIZE);
    auto policy = execution_policy(grid_size, block_size, stream);

    auto kernel = raw::box_collect<T>;
    launch_kernel(kernel, policy, collected_bboxes, decoded_bboxes, indices, count, share_location, num_priors, num_classes, classwise_topK, background_class_id);
}

template void box_collect(const Stream&, TensorSpan<float>, TensorView<float>, TensorView<int>, TensorView<int>, bool, std::size_t);
template void box_collect(const Stream&, TensorSpan<__half>, TensorView<__half>, TensorView<int>, TensorView<int>, bool, std::size_t);

template <class T>
void blockwise_class_nms(const Stream& stream, TensorSpan<int> indices, TensorSpan<int> count, TensorView<T> collected_bboxes,
    bool normalized_bbox, std::size_t background_class_id, float nms_threshold)
{
    // indices: [batch_size, num_classes, classwise_topK]
    // count: [batch_size, num_classes]
    // collected_bboxes: [batch_size, num_classes, classwise_topK, 4]

    const auto batch_size = indices.get_axis_size(0);
    CV_Assert(count.get_axis_size(0) == batch_size);
    CV_Assert(collected_bboxes.get_axis_size(0) == batch_size);

    const auto num_classes = indices.get_axis_size(1);
    CV_Assert(count.get_axis_size(1) == num_classes);
    CV_Assert(collected_bboxes.get_axis_size(1) == num_classes);

    const auto classwise_topK = indices.get_axis_size(2);
    CV_Assert(collected_bboxes.get_axis_size(2) == classwise_topK);

    /* each block processes one class from each batch */
    auto num_blocks = batch_size * num_classes;
    auto num_threads = std::max<std::size_t>(std::min<std::size_t>(1024, classwise_topK), 32);

    dim3 grid_size(num_blocks);
    dim3 block_size(num_threads);
    auto policy = execution_policy(grid_size, block_size, stream);

    if (normalized_bbox)
    {
        auto kernel = raw::blockwise_class_nms<T, true>;
        launch_kernel(kernel, policy, indices, count, collected_bboxes, num_classes, classwise_topK, background_class_id, nms_threshold);
    }
    else
    {
        auto kernel = raw::blockwise_class_nms<T, false>;
        launch_kernel(kernel, policy, indices, count, collected_bboxes, num_classes, classwise_topK, background_class_id, nms_threshold);
    }
}

template void blockwise_class_nms(const Stream&, TensorSpan<int>, TensorSpan<int>, TensorView<__half>, bool, std::size_t, float);
template void blockwise_class_nms(const Stream&, TensorSpan<int>, TensorSpan<int>, TensorView<float>, bool, std::size_t, float);

template <class T>
void nms_collect(const Stream& stream, TensorSpan<int> kept_indices, TensorSpan<int> kept_count,
    TensorView<int> indices, TensorView<int> count, TensorView<T> scores, float threshold, std::size_t background_class_id)
{
    // kept_indices: [batch_size, keepTopK]
    // kept_count: [batch_size]

    // indices: [batch_size, num_classes, classwise_topK]
    // count: [batch_size, num_classes]
    // scores: [batch_size, num_classes, num_priors]

    auto batch_size = kept_indices.get_axis_size(0);
    CV_Assert(kept_count.get_axis_size(0) == batch_size);
    CV_Assert(indices.get_axis_size(0) == batch_size);
    CV_Assert(count.get_axis_size(0) == batch_size);
    CV_Assert(scores.get_axis_size(0) == batch_size);

    auto keepTopK = kept_indices.get_axis_size(1);

    auto num_classes = indices.get_axis_size(1);
    CV_Assert(count.get_axis_size(1) == num_classes);
    CV_Assert(scores.get_axis_size(1) == num_classes);

    auto classwise_topK = indices.get_axis_size(2);
    auto num_priors = scores.get_axis_size(2);

    auto num_blocks = batch_size;
    constexpr int BLOCK_SIZE = 1024;

    dim3 grid_size(num_blocks);
    dim3 block_size(BLOCK_SIZE);
    auto policy = execution_policy(grid_size, block_size, stream);

    auto kernel = raw::nms_collect<T, 1024, BLOCK_SIZE>;
    launch_kernel(kernel, policy, kept_indices, kept_count, indices, count, scores, threshold, num_classes, num_priors, classwise_topK, keepTopK, background_class_id);
}

template void nms_collect(const Stream&, TensorSpan<int>, TensorSpan<int>, TensorView<int>, TensorView<int>, TensorView<__half>, float, std::size_t);
template void nms_collect(const Stream&, TensorSpan<int>, TensorSpan<int>, TensorView<int>, TensorView<int>, TensorView<float>, float, std::size_t);

template <class T>
void consolidate_detections(const Stream& stream, TensorSpan<T> output,
    TensorView<int> kept_indices, TensorView<int> kept_count,
    TensorView<T> decoded_bboxes, TensorView<T> scores, bool share_location, DevicePtr<int> num_detections)
{
    // output: [1, 1, batch_size * keepTopK, 7]
    // kept_indices: [batch_size, keepTopK]
    // kept_count: [batch_size]
    // decoded_bboxes: [batch_size, num_priors, num_loc_classes, 4]
    // scores: [batch_size, num_classes, num_priors]

    auto batch_size = kept_indices.get_axis_size(0);
    CV_Assert(kept_count.get_axis_size(0) == batch_size);
    CV_Assert(decoded_bboxes.get_axis_size(0) == batch_size);
    CV_Assert(scores.get_axis_size(0) == batch_size);

    auto keepTopK = kept_indices.get_axis_size(1);

    auto num_classes = scores.get_axis_size(1);
    auto num_priors = scores.get_axis_size(2);

    CV_Assert(batch_size * keepTopK * 7 == output.size());

    auto kernel = raw::consolidate_detections<T>;
    auto policy = make_policy(kernel, keepTopK, 0, stream);
    launch_kernel(kernel, policy, output, kept_indices, kept_count, decoded_bboxes, scores, share_location, batch_size, num_classes, num_priors, keepTopK, num_detections);
}

template void consolidate_detections(const Stream&, TensorSpan<__half>, TensorView<int>, TensorView<int>, TensorView<__half>, TensorView<__half>, bool, DevicePtr<int>);
template void consolidate_detections(const Stream&, TensorSpan<float>, TensorView<int>, TensorView<int>, TensorView<float>, TensorView<float>, bool, DevicePtr<int>);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
