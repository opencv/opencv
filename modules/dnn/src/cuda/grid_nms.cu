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

    template <class T, bool NORMALIZED_BBOX, int BLOCK_SIZE>
    __launch_bounds__(BLOCK_SIZE)
    __global__ void grid_nms(Span<unsigned int> mask_, Span<int> count_, View<T> bboxes_, size_type num_classes, index_type background_class_id, size_type topK, size_type topK_gs, float nms_threshold)
    {
        // topK_gs is topK rounded upwards to some size

        // mask: [batch_size, num_classes, topK_gs, topK_gs / 32]
        // bboxes: [batch_size, num_classes, topK, 4]
        // count: [batch_size, num_classes]

        const index_type c = blockIdx.y;
        const index_type b = blockIdx.z;

        if (c == background_class_id)
            return;

        auto mask = mask_.data() + (b * num_classes + c) * topK_gs * topK_gs / 32;
        auto bboxes = bboxes_.data() + (b * num_classes + c) * topK * 4;
        auto count = count_.data() + b * num_classes + c;

        const auto boxes = *count;
        if (boxes == 0)
            return;

        /* We divide the set of boxes into groups containing BLOCK_SIZE boxes */
        const auto num_groups = (boxes + BLOCK_SIZE - 1) / BLOCK_SIZE;

        /* We need to calculate IOUs for every pair of boxes. We can generalize and say that
         * we need to compute IOUs of every group with every other group including itself.
         */
        // Each block processes a pair of groups.
        const index_type group_i = blockIdx.x % num_groups;
        const index_type group_j = blockIdx.x / num_groups;

        /* we use __syncthreads() later but note that the following condition will cause all threads
         * in the block to exit; hence, no thread will execute a divergent __syncthreads()
         */
        if (group_i >= num_groups || group_j >= num_groups)
            return;

        /* Note that IOU(A, B) = IOU(B, A). Hence, if we compute IOU(GROUP_A, GROUP_B), we do not need
         * to compute IOU(GROUP_B, GROUP_A). We still have to compute IOU(GROUP_A, GROUP_A) though since
         * each group has many boxes and we need IOUs amongst boxes within a group.
         *
         * We arbitarily choose a scheme to exit : exit if group_i is greater than group_j. This way we only
         * compute IOUs between groups once. While nearly half the blocks are wasted, it's ok since they exit
         * early on and the working blocks are compute heavy.
         */
        if (group_i > group_j)
            return;

        /* the following variables contain the absolute box number of the first box of their respective groups */
        const auto group_i_offset = group_i * BLOCK_SIZE;
        const auto group_j_offset = group_j * BLOCK_SIZE;

        /* MAIN LOOP LOGIC:
         * We compare a box `i` from group_i with all boxes in group_j in each iteration. The box `j` is fixed
         * for each thread. The `j` exactly maps to the thread index. Hence, the `j` is a loop invariant. Each
         * thread of the block computes the overlap between box `i` and its box `j`.
         *
         * for (int i = 0; i < BLOCK_SIZE; i++)
         * {
         *    // i = box 1
         *    // j = threadIdx.x = box 2
         * }
         */

        /* The `j` box is fixed for each thread. All `i` boxes will be required for every thread.
         * We store the `i` boxes in shared memory to allow global memory coalesing.
         */
        using vector_type = get_vector_type_t<T, 4>;
        __shared__ vector_type group_i_boxes[BLOCK_SIZE];

        /* We will precompute the sizes of `i` boxes in the code where we load them. The size computation
         * is distributed across the block. Otherwise, all threads will have to compute the size of the same
         * box simultaneously in the main loop. The size is computed while the memory subsystem is busy
         * servicing requests for box coordinates; the compute resources would otherwise be idle in this phase.
         */
        /* we store the size as a float since the size can exceed fp16 limits for unnormalized boxes */
        __shared__ float group_i_size[BLOCK_SIZE];

        const auto bboxes_vPtr = vector_type::get_pointer(bboxes);

        // load `i` boxes and precompute their sizes
        {
            int i = threadIdx.x;
            if (group_i_offset + i < boxes)
            {
                vector_type box;
                v_load(box, bboxes_vPtr[group_i_offset + i]);
                v_store(group_i_boxes[i], box);

                BoundingBox bbox;
                bbox.xmin = box.data[0];
                bbox.ymin = box.data[1];
                bbox.xmax = box.data[2];
                bbox.ymax = box.data[3];

                group_i_size[i] = compute_bbox_size<NORMALIZED_BBOX>(bbox);
            }
        }

        __syncthreads();

        /* We compute overlap between boxes and check if the IOU exceeds the nms threshold.
         * We store the result (exceeds or below nms_thresold) in a two-dimensional matrix.
         * (i, j) is set to one if the overlap between i and j is within the nms threshold.
         * We pack 32 results into one 32-bit integer. The effective memory layout of the
         * matrix hence is (BLOCK_SIZE, BLOCK_SIZE / 32).
         */
        __shared__ unsigned int mask_shared[BLOCK_SIZE * BLOCK_SIZE / 32];

        // load box `j` and precompute its size (fixed per thread)
        BoundingBox bbox_j;
        float bbox_j_size = 0;
        if (group_j_offset + threadIdx.x < boxes)
        {
            vector_type box;
            v_load(box, bboxes_vPtr[group_j_offset + threadIdx.x]);

            bbox_j.xmin = box.data[0];
            bbox_j.ymin = box.data[1];
            bbox_j.xmax = box.data[2];
            bbox_j.ymax = box.data[3];

            bbox_j_size = compute_bbox_size<NORMALIZED_BBOX>(bbox_j);
        }

        /* Each thread computes a predicate which is broadcasted across the warp to obtain a 32-bit mask.
         * The lane zero thread of each warp saves the mask. We store the offset to the mask array beforehand
         * to save cycles in the compute-intensive main loop.
         */
        auto mask_offset = threadIdx.x / 32;

        /* The main loop is compute intensive and causes the kernel to be overall compute-bound. Hence,
         * this loop has been highly tuned. Please profile and verify carefully before making changes.
         */
        /* UNROLL_SIZE is the number of boxes that must be processed per iteration. We manually unroll
         * the loop since the compiler cannot effectively unroll on its own preassumably due to presence
         * of instructions forcing warp synchronization.
         */
        constexpr int UNROLL_SIZE = 4;

        #pragma unroll 8
        for (int s = 0; s < BLOCK_SIZE; s += UNROLL_SIZE)
        {
            bool do_not_reject_j[UNROLL_SIZE];

            #pragma unroll
            for (int k = 0; k < UNROLL_SIZE; k++)
            {
                int i = s + k;

                /* The number of boxes need not necessarily be a multiple of BLOCK_SIZE.
                 * However, the shared memory allocated can hold BLOCK_SIZE boxes from
                 * each group. Accessing the undefined regions of shared memory is
                 * a valid memory operation as long as the memory has been allocated.
                 *
                 * The condition below is only required when one of the groups does not
                 * fully filled with valid boxes. This situations are relatively rare. It's
                 * more common to see both groups completely filled.
                 *
                 * We comment this condition to improve the performance of the common case.
                 * This leads to a net improvement.
                 */
                // if (group_i_offset + i < boxes && group_j_offset + threadIdx.x < boxes)
                {
                    BoundingBox bbox_i;
                    float bbox_i_size;
                    {
                        vector_type box;
                        v_load(box, group_i_boxes[i]);
                        bbox_i.xmin = box.data[0];
                        bbox_i.ymin = box.data[1];
                        bbox_i.xmax = box.data[2];
                        bbox_i.ymax = box.data[3];

                        bbox_i_size = group_i_size[i];
                    }

                    using device::min;
                    using device::max;

                    BoundingBox intersect_bbox;
                    intersect_bbox.xmin = max(bbox_i.xmin, bbox_j.xmin);
                    intersect_bbox.ymin = max(bbox_i.ymin, bbox_j.ymin);
                    intersect_bbox.xmax = min(bbox_i.xmax, bbox_j.xmax);
                    intersect_bbox.ymax = min(bbox_i.ymax, bbox_j.ymax);

                    float intersect_size = compute_bbox_size<NORMALIZED_BBOX>(intersect_bbox);

                    using device::fast_divide_ftz;
                    float iou = fast_divide_ftz(intersect_size, bbox_i_size + bbox_j_size - intersect_size);
                    do_not_reject_j[k] = iou <= nms_threshold;
                }
            }

            #pragma unroll
            for (int k = 0; k < UNROLL_SIZE; k++)
            {
                // FORWARD_COMPATIBILITY_TAG: WARP_SIZE_DEPENDENT_CODE
                auto predicate = __ballot_sync(0xFFFFFFFF, do_not_reject_j[k]);
                if (threadIdx.x % 32 == 0)
                    mask_shared[mask_offset] = predicate;

                /* The following operation should logically be inside the previous if branch. Note that `mask_offset`
                 * is only used by lane zero threads. Hence, there is no harm in executing it other threads as it is
                 * unused there.
                 *
                 * Keeping it inside prevents the compiler from treating it as a constexpr addition to the address in
                 * successive unrolled iterations. A register is used and instructions are emitted to multiply the
                 * addend by four to obtain the byte offset. Pulling it out of the branch makes the compiler do constexpr
                 * addition on the address in successive unrolled iterations.
                 */
                mask_offset += BLOCK_SIZE / 32;
            }
        }

        __syncthreads();

        /* The mask data is organized as a two-dimensional bit matrix of size topK_gs * topK_gs.
         * (i, j) is set to true if the overlap between `i` and `j` is beyond the nms threshold.
         * We pack 32 results into one 32-bit integer. So the effective memory layout is topK_gs * topK_gs / 32.
         */

        /* Each box `i` was compared with BLOCK_SIZE `j` boxes. This amounts to BLOCK_SIZE / 32
         * 32-bit integers per box `i`.
         */
        using mask_vector_type = get_vector_type_t<unsigned int, BLOCK_SIZE / 32>;

        const int i = threadIdx.x;

        auto mask_shared_vPtr = mask_vector_type::get_pointer(DevicePtr<unsigned>(mask_shared));
        mask_vector_type temp;
        v_load(temp, mask_shared_vPtr[i]);
        for (int i = 0; i < mask_vector_type::size(); i++)
            temp.data[i] = __brev(temp.data[i]);

        auto mask_vPtr = mask_vector_type::get_pointer(mask);
        v_store(mask_vPtr[((group_i_offset + i) * topK_gs + group_j_offset) / 32 / mask_vector_type::size()], temp);
    }

    template <int ITEMS_PER_THREAD, int BLOCK_SIZE>
    __launch_bounds__(BLOCK_SIZE)
    __global__ void grid_nms_collect(Span<int> indices_, Span<int> count_, View<unsigned int> mask_, size_type num_classes, index_type background_class_id, size_type topK, size_type topK_gs_by32)
    {
        const index_type c = blockIdx.x;
        if (c == background_class_id)
            return;

        const index_type b = blockIdx.y;

        // topK_gs is topK rounded upwards to some size

        // indices: [batch_size, num_classes, topK]
        // count: [batch_size, num_classes]
        // mask: [batch_size, num_classes, topK_gs, topK_gs / 32]

        auto indices = indices_.data() + (b * num_classes + c) * topK;
        auto count = count_.data() + (b * num_classes + c);
        auto mask = mask_.data() + (b * num_classes + c) * topK_gs_by32 * 32 * topK_gs_by32;

        const auto boxes = *count;
        if (boxes == 0)
            return;

        /* We have a fixed number of threads and an arbitary number of boxes. We use an array of
         * bits to store which boxes haven't been eliminated and which are still active. We organize
         * the array of bits into a matrix of bits of the shape (num_rows, BLOCK_SIZE, 32) which
         * is equivalent to (num_rows, BLOCK_SIZE) where the type is a 32-bit unsigned integer.
         * `num_rows` is the minimum number of rows required to cover all the boxes.
         *
         * Each thread handles a specific column in the matrix. To improve performance, we process
         * `ITEMS_PER_THREAD` number of elements per thread. This changes the shape to (num_rows,
         * ROW_WIDTH) where ROW_WIDTH is BLOCK_SIZE * ITEMS_PER_THREAD.
         */
         constexpr int ROW_WIDTH = BLOCK_SIZE * ITEMS_PER_THREAD;

         const index_type num_32b_masks = static_cast<unsigned>(boxes + 31) / 32;
         const index_type num_rows = static_cast<unsigned>(num_32b_masks + ROW_WIDTH - 1) / ROW_WIDTH;

        extern __shared__ unsigned int active_boxes[]; // the matrix described earlier

        #pragma unroll 1
        for (auto idx : block_stride_range<BLOCK_SIZE>(num_32b_masks))
            active_boxes[idx] = (idx == num_32b_masks - 1) ? __brev((1u << (boxes % 32)) - 1) : 0xFFFFFFFF;

        __syncthreads();

        using vector_type = get_vector_type_t<unsigned int, ITEMS_PER_THREAD>;
        auto mask_vPtr = vector_type::get_pointer(mask);
        auto shared_vPtr = vector_type::get_pointer(DevicePtr<unsigned>(active_boxes));

        int index_temp;
        int thread0_count = 0;
        int thread_id = threadIdx.x;

        for (int step = 0; step < num_32b_masks; step++)
        {
            auto current_active = active_boxes[step];
            while (current_active)
            {
                const index_type bit = __clz(current_active);
                const index_type i = step * 32 + bit;

                const int mask_offset = static_cast<unsigned>(i * topK_gs_by32) / ITEMS_PER_THREAD;

                /* We fetch the index from the memory and store it in a register. We will not use it until
                 * much later. This helps avoid a long scoreboard stall.
                 */
                if (thread_id == 0)
                    index_temp = indices[i];

                __syncthreads();

                if (threadIdx.x == 0)
                    active_boxes[step] = current_active ^ (0x80000000 >> bit);

                __syncthreads();

                #pragma unroll 1
                for (int r = 0; r < num_rows; r++)
                {
                    const int idx = r * BLOCK_SIZE + thread_id;
                    if ((step & ~(ITEMS_PER_THREAD - 1)) <= idx * ITEMS_PER_THREAD && idx * ITEMS_PER_THREAD < num_32b_masks)
                    {
                        auto active_boxes_vec = shared_vPtr[idx];
                        auto mask_vec = mask_vPtr[mask_offset + idx];
                        for (int i = 0; i < vector_type::size(); i++)
                            active_boxes_vec.data[i] &= mask_vec.data[i];
                        shared_vPtr[idx] = active_boxes_vec;
                    }
                }

                __syncthreads();

                if (thread_id == 0)
                {
                    indices[thread0_count] = index_temp;
                    thread0_count++;
                }

                current_active = active_boxes[step];
            }
        }

        if (threadIdx.x == 0)
            *count = thread0_count;
    }
}

constexpr int GROUP_SIZE = 128;

static std::size_t getAlignedTopK(std::size_t topK)
{
    auto remainder = topK % GROUP_SIZE;
    if (remainder == 0)
        return topK;
    return topK + (GROUP_SIZE - remainder);
}

std::size_t getGridNMSWorkspaceSizePerBatchItem(std::size_t num_classes, std::size_t classwise_topK)
{
    auto topK_gs = getAlignedTopK(classwise_topK);
    return num_classes * topK_gs * topK_gs / 32 * sizeof(unsigned int);
}

template <class T>
void grid_nms(const Stream& stream, Span<unsigned int> workspace, TensorSpan<int> indices, TensorSpan<int> count, TensorView<T> bboxes, int background_class_id, bool normalized_bbox, float nms_threshold)
{
    // workspace: [batch_size, num_classes, topK_gs, topK_gs / 32]
    // indices: [batch_size, num_classes, topK]
    // count: [batch_size, num_classes]
    // bboxes: [batch_size, num_classes, topK, 4] (only first count[b][c] boxes are read)

    const auto batch_size = indices.get_axis_size(0);
    CV_Assert(count.get_axis_size(0) == batch_size);
    CV_Assert(bboxes.get_axis_size(0) == batch_size);

    const auto num_classes = indices.get_axis_size(1);
    CV_Assert(count.get_axis_size(1) == num_classes);
    CV_Assert(bboxes.get_axis_size(1) == num_classes);

    const auto topK = indices.get_axis_size(2);
    CV_Assert(bboxes.get_axis_size(2) == topK);

    CV_Assert(bboxes.get_axis_size(3) == 4);

    const auto topK_gs = getAlignedTopK(topK);
    CV_Assert(workspace.size() >= topK_gs * topK_gs / 32);

    const auto boxes = topK;
    const auto num_groups = (boxes + GROUP_SIZE - 1) / GROUP_SIZE;

    {
        // grid = (num_groups * num_groups, num_classes, batch_size)
        // if the background class is the last class, we can reduce grid y dim by one
        auto grid_num_classes = num_classes; //(background_class_id == num_classes - 1) ? num_classes - 1 : num_classes;

        constexpr int BLOCK_SIZE = GROUP_SIZE;

        dim3 grid_size(num_groups * num_groups, grid_num_classes, batch_size);
        dim3 block_size(BLOCK_SIZE);
        auto policy = execution_policy(grid_size, block_size, stream);

        if (normalized_bbox)
        {
            auto kernel = raw::grid_nms<T, true, BLOCK_SIZE>;
            launch_kernel(kernel, policy, workspace, count, bboxes, num_classes, background_class_id, topK, topK_gs, nms_threshold);
        }
        else
        {
            auto kernel = raw::grid_nms<T, false, BLOCK_SIZE>;
            launch_kernel(kernel, policy, workspace, count, bboxes, num_classes, background_class_id, topK, topK_gs, nms_threshold);
        }
    }

    {
        // grid = (num_classes, batch_size)
        // if the background class is the last class, we can reduce grid x dim by one
        auto grid_num_classes = num_classes; //(background_class_id == num_classes - 1) ? num_classes - 1 : num_classes;

        constexpr int BLOCK_SIZE = 64;

        constexpr int ITEMS_PER_THREAD = 4;
        auto kernel = raw::grid_nms_collect<ITEMS_PER_THREAD, BLOCK_SIZE>;

        dim3 grid_size(grid_num_classes, batch_size);

        auto sharedMem = topK_gs / 32 * 4;
        auto policy = execution_policy(grid_size, BLOCK_SIZE, sharedMem, stream);
        launch_kernel(kernel, policy, indices, count, workspace, num_classes, background_class_id, topK, topK_gs / 32);
    }
}

std::size_t getGridNMSWorkspaceSizePerBatchItem(std::size_t num_classes, std::size_t classwise_topK);

template void grid_nms(const Stream& stream, Span<unsigned int> workspace, TensorSpan<int> indices, TensorSpan<int> count, TensorView<__half> bboxes, int, bool normalized_bbox, float nms_threshold);
template void grid_nms(const Stream& stream, Span<unsigned int> workspace, TensorSpan<int> indices, TensorSpan<int> count, TensorView<float> bboxes, int, bool normalized_bbox, float nms_threshold);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */