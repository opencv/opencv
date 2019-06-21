// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "op_cuda.hpp"

#ifdef HAVE_CUDA
#include "cuda4dnn/csl/stream.hpp"
#include "cuda4dnn/csl/tensor.hpp"
#include "cuda4dnn/csl/pointer.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/core.hpp>

#include <memory>

namespace cv {
    namespace dnn {
#ifdef HAVE_CUDA
        CUDABackendWrapperFP32::CUDABackendWrapperFP32(Mat& m)
            : BackendWrapper(DNN_BACKEND_OPENCV, DNN_TARGET_CUDA_FP32)
        {
            CV_Assert(m.isContinuous());
            CV_Assert(m.type() == CV_32F);
            CV_Assert(m.size.dims() <= tensor_type::rank);

            shape = cv::dnn::shape(m);

            shared_block = std::make_shared<shared_block_type>();
            shared_block->host_dirty = true;
            shared_block->device_dirty = false;
            shared_block->host = m;
            shared_block->memGuard = csl::MemoryLockGuard(m.data, m.total() * sizeof(float));
            shared_block->parent = createTensorHeaderFromMat<tensor_type>(m);
        }

        CUDABackendWrapperFP32::CUDABackendWrapperFP32(const Ptr<BackendWrapper>& base_, const MatShape& shape_)
            : BackendWrapper(DNN_BACKEND_OPENCV, DNN_TARGET_CUDA_FP32)
        {
            const Ptr<CUDABackendWrapperFP32> base = base_.dynamicCast<CUDABackendWrapperFP32>();

            shape = shape_;
            shared_block = base->shared_block;
        }

        Ptr<BackendWrapper> CUDABackendWrapperFP32::create(Mat& m)
        {
            return Ptr<BackendWrapper>(new CUDABackendWrapperFP32(m));
        }

        Ptr<BackendWrapper> CUDABackendWrapperFP32::create(const Ptr<BackendWrapper>& base, const MatShape& shape)
        {
            return Ptr<BackendWrapper>(new CUDABackendWrapperFP32(base, shape));
        }

        /* blocking */
        void CUDABackendWrapperFP32::copyToHost() {
            if(shared_block->device_dirty) {
                shared_block->host_dirty = false;
                shared_block->device_dirty = false;

                /* If the wrapper is being reused, the device tensor might be larger in size.
                 * Using the device tensor does not give incorrect code, but it leads to unused regions
                 * of memory being copied.
                 *
                 * We use a view to ensure that only the required region of memory is copied.
                 */
                auto view = tensor_view_type(shared_block->parent).subview(0, std::begin(shape), std::end(shape));
                copyTensorToMat(shared_block->host, view, shared_block->stream);

                shared_block->stream.synchronize();
            }
        }

        void CUDABackendWrapperFP32::setHostDirty() {
            shared_block->device_dirty = false;
            shared_block->host_dirty = true;
        }

        /* non-blocking
         * we don't have to block for copying to device because all operations are put into a stream which
         * ensures that the operations added to the stream are performed in order
         */
        void CUDABackendWrapperFP32::copyToDevice() {
            if(shared_block->host_dirty) {
                shared_block->host_dirty = false;
                shared_block->device_dirty = false;

                auto span = tensor_span_type(shared_block->parent).subspan(0, std::begin(shape), std::end(shape));
                copyMatToTensor(span, shared_block->host, shared_block->stream);
            }
        }

        void CUDABackendWrapperFP32::setDeviceDirty() noexcept {
            shared_block->device_dirty = true;
            shared_block->host_dirty = false;
        }

        void CUDABackendWrapperFP32::setStream(csl::Stream stream) noexcept {
            shared_block->stream = std::move(stream);
        }

        CUDABackendWrapperFP32::tensor_span_type CUDABackendWrapperFP32::getSpan() noexcept {
            setDeviceDirty();
            return tensor_span_type(shared_block->parent).subspan(0, std::begin(shape), std::end(shape));
        }

        CUDABackendWrapperFP32::tensor_view_type CUDABackendWrapperFP32::getView() noexcept {
            copyToDevice();
            return tensor_view_type(shared_block->parent).subview(0, std::begin(shape), std::end(shape));
        }

#endif /* ifdef HAVE_CUDA */
    } /* namespace dnn */
} /* namespace cv */
