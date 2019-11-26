// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_OP_CUDA_HPP
#define OPENCV_DNN_SRC_OP_CUDA_HPP

#ifdef HAVE_CUDA
#include "cuda4dnn/csl/stream.hpp"
#include "cuda4dnn/csl/cublas.hpp"
#include "cuda4dnn/csl/cudnn.hpp"
#include "cuda4dnn/csl/tensor.hpp"
#include "cuda4dnn/csl/memory.hpp"
#include "cuda4dnn/csl/fp16.hpp"
#include "cuda4dnn/csl/workspace.hpp"
#endif

#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/core.hpp>

#include <cstddef>
#include <memory>
#include <iterator>

namespace cv { namespace dnn {

    constexpr bool IS_DNN_CUDA_TARGET(int id) {
        return id == DNN_TARGET_CUDA_FP16 || id == DNN_TARGET_CUDA;
    }

    constexpr bool haveCUDA() {
#ifdef HAVE_CUDA
        return true;
#else
        return false;
#endif
    }

#ifdef HAVE_CUDA
    namespace cuda4dnn { namespace csl {
        struct CSLContext {
            Stream stream;
            cublas::Handle cublas_handle;
            cudnn::Handle cudnn_handle;
        };

        /** @brief creates Tensor object from cv::Mat (only the header is created, i.e. no data is copied)
         *
         * \tparam      T   element type for the tensor
         * \param[in]   mat cv::Mat from which the shape must be inferred
         *
         * \return a Tensor object with the shape of \p mat
         */
        template <class T>
        Tensor<T> makeTensorHeader(const Mat& mat) {
            auto sizes = shape(mat);
            return Tensor<T>(std::begin(sizes), std::end(sizes));
        }

        /** @brief copies data from a cv::Mat to TensorType
         *
         * \tparam  T   the type of the elements contained in TensorType object
         *
         * \param[in]   srcMat      source matrix
         * \param[out]  destTensor  destination tensor
         * \param       stream      CUDA stream to use for the memory transfer
         *
         * The memory copy starts from beginning \p srcMat. The number of elements copied is
         * equal to the number of elements in \p destTensor.
         *
         * Pre-conditions:
         * - \p srcMat must contain elements of type CV_32F
         * - the size of \p srcMat must be larger than or equal to the size of \p destTensor
         *
         * @note best performance when \p srcMat is continuous and page-locked
         * @note blocks calling thread if \p srcMat is not page-locked
         */
        template <class T>
        void copyMatToTensor(const Mat& srcMat, const TensorSpan<T> destTensor, const Stream& stream);

        template <> inline
        void copyMatToTensor(const Mat& srcMat, const TensorSpan<half> destTensor, const Stream& stream) {
            /* should perhaps convert cv::Mat of different type to the required type and copy */
            CV_Assert(srcMat.type() == CV_32F);
            CV_Assert(srcMat.total() >= destTensor.size());

            Mat temp;
            srcMat.convertTo(temp, CV_16F);
            CV_Assert(temp.isContinuous());

            memcpy<half>(destTensor.get(), reinterpret_cast<half*>(temp.data), destTensor.size(), stream);
        }

        template <> inline
        void copyMatToTensor(const Mat& srcMat, const TensorSpan<float> destTensor, const Stream& stream) {
            /* should perhaps convert cv::Mat of different type to the required type and copy */
            CV_Assert(srcMat.type() == CV_32F);
            CV_Assert(srcMat.total() >= destTensor.size());

            Mat temp = srcMat.isContinuous() ? srcMat : srcMat.clone();
            CV_Assert(temp.isContinuous());

            memcpy<float>(destTensor.get(), reinterpret_cast<float*>(temp.data), destTensor.size(), stream);
        }

        /** @brief copies data from a TensorType to a cv::Mat
         *
         * \tparam  T   the type of the elements contained in TensorType object
         *
         * \param[in]   srcTensor   source tensor
         * \param[out]  destMat     destination matrix
         * \param       stream      CUDA stream to use for the memory transfer
         *
         * The entire memory block held by the \p srcTensor is copied to \p destMat.
         *
         * Pre-conditions:
         * - \p destMat must contain elements of type CV_32F
         * - the size of \p destMat must be larger than or equal to the size of \p srcTensor
         *
         * @note best performance when \p destMat is continuous and page-locked
         * @note blocks calling thread if \p destMat is not page-locked
         */
        template <class T>
        void copyTensorToMat(TensorView<T> srcTensor, Mat& destMat, const Stream& stream);

        template <> inline
        void copyTensorToMat(TensorView<half> srcTensor, Mat& destMat, const Stream& stream) {
            CV_Assert(destMat.type() == CV_32F);
            CV_Assert(destMat.total() >= srcTensor.size());

            Mat temp(shape(destMat), CV_16F);
            CV_Assert(temp.isContinuous());

            memcpy<half>(reinterpret_cast<half*>(temp.data), srcTensor.get(), srcTensor.size(), stream);

            temp.convertTo(destMat, CV_32F);
        }

        template <> inline
        void copyTensorToMat(TensorView<float> srcTensor, Mat& destMat, const Stream& stream) {
            CV_Assert(destMat.type() == CV_32F);
            CV_Assert(destMat.total() >= srcTensor.size());

            Mat temp = destMat.isContinuous() ? destMat : destMat.clone();
            CV_Assert(temp.isContinuous());

            memcpy<float>(reinterpret_cast<float*>(temp.data), srcTensor.get(), srcTensor.size(), stream);

            if (temp.data != destMat.data)
                temp.copyTo(destMat);
        }

    }} /* namespace cuda4dnn::csl */

    /** base class for CUDA operation nodes (for all supported targets) */
    class CUDABackendNode : public BackendNode {
    public:
        CUDABackendNode() : BackendNode(DNN_BACKEND_CUDA) { }
        virtual ~CUDABackendNode() { }

        virtual void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            cuda4dnn::csl::Workspace& workspace) = 0;

        virtual std::size_t get_workspace_memory_in_bytes() const noexcept { return 0; }
    };

    /** @brief utility function which creates CUDA node of correct type from `targetId`
     *
     * CUDA operation nodes take the type of data they operate on as a template parameter.
     * For example, ConcatOp<float> is an operation node which concats tensors of `float` type
     * into a tensor of `float` type.
     *
     * This utility function aids the creation of nodes of different types and eliminates the
     * need for CUDA target constants (`DNN_TARGET_XXX`) to appear in the operation code which
     * reduces coupling between modules.
     *
     * Example:
     * template <class T>
     * class ConcatOp : public CUDABackendNode;
     *
     * // returns a cv::Ptr to a ConcatOp<half> object
     * auto node = make_cuda_node<ConcatOp>(DNN_TARGET_CUDA_FP16, axis);
     *
     * // returns a cv::Ptr to a ConcatOp<float> object
     * auto node = make_cuda_node<ConcatOp>(DNN_TARGET_CUDA, axis);
     */
    template <template <class> class NodeType, class ...Args>
    cv::Ptr<BackendNode> make_cuda_node(int targetId, Args&& ...args) {
        switch (targetId)
        {
        case DNN_TARGET_CUDA_FP16:
            return Ptr<BackendNode>(new NodeType<half>(std::forward<Args>(args)...));
        case DNN_TARGET_CUDA:
            return Ptr<BackendNode>(new NodeType<float>(std::forward<Args>(args)...));
        default:
            CV_Assert(IS_DNN_CUDA_TARGET(targetId));
        }
        return Ptr<BackendNode>();
    }

    /* base class for all CUDA backend/target wrappers */
    class CUDABackendWrapper : public BackendWrapper {
    public:
        CUDABackendWrapper(int targetId) : BackendWrapper(DNN_BACKEND_CUDA, targetId) { }
        virtual ~CUDABackendWrapper() { }

        void copyToHost() override = 0;
        void setHostDirty() override = 0;

        virtual void copyToDevice() = 0;
        virtual void setDeviceDirty() = 0;

        virtual MatShape getShape() const noexcept = 0;
        virtual std::size_t getRank() const noexcept = 0;

        /** @note setting the stream updates the stream for all wrappers which use the same tensor */
        virtual void setStream(cuda4dnn::csl::Stream stream) noexcept = 0;
    };

    template <class T, int TargetID>
    class GenericCUDABackendWrapper final : public CUDABackendWrapper {
    public:
        using value_type = T;
        using tensor_span_type = cuda4dnn::csl::TensorSpan<value_type>;
        using tensor_view_type = cuda4dnn::csl::TensorView<value_type>;

        /* Pre-conditions:
         * - there must be no other instance of `GenericCUDABackendWrapper` which wraps the host memory used by `m`
         * - the host memory must remain allocated throughout the lifetime of this object
         *
         * Post-conditions:
         * - the host memory used by \p m "may" be page-locked
         */
        GenericCUDABackendWrapper(Mat& m)
            : CUDABackendWrapper(TargetID)
        {
            shape = cv::dnn::shape(m);

            shared_block = std::make_shared<shared_block_type>();
            shared_block->host_dirty = true;
            shared_block->device_dirty = false;

            shared_block->host = m;

            try {
                shared_block->memGuard = cuda4dnn::csl::MemoryLockGuard(m.data, m.total() * m.elemSize());
            } catch (...) {
                /* a common reason for failure is that the host system (for example, a Jetson device) does not support it */
                /* we ignore the failure as this is just an optimization and not a requirement */
            }

            shared_block->device = cuda4dnn::csl::ManagedPtr<T>(m.total());
        }

        GenericCUDABackendWrapper(const Ptr<BackendWrapper>& base_, const MatShape& shape_)
            : CUDABackendWrapper(TargetID)
        {
            const Ptr<GenericCUDABackendWrapper> base = base_.dynamicCast<GenericCUDABackendWrapper>();
            CV_Assert(base);

            shape = shape_;
            shared_block = base->shared_block;
        }

        static Ptr<BackendWrapper> create(Mat& m) {
            return Ptr<BackendWrapper>(new GenericCUDABackendWrapper(m));
        }

        static Ptr<BackendWrapper> create(const Ptr<BackendWrapper>& base, const MatShape& shape) {
            return Ptr<BackendWrapper>(new GenericCUDABackendWrapper(base, shape));
        }

        void copyToHost() override {
            if (shared_block->device_dirty) {
                shared_block->host_dirty = false;
                shared_block->device_dirty = false;

                /* If the wrapper is being reused, the device tensor might be larger in size than the wrapper.
                 * Using the device tensor does not give incorrect code but leads to unused region of memory being copied.
                 *
                 * We use a view to ensure that only the required region of memory is copied.
                 */
                auto view = tensor_view_type(shared_block->device.get(), std::begin(shape), std::end(shape));
                cuda4dnn::csl::copyTensorToMat<T>(view, shared_block->host, shared_block->stream);

                shared_block->stream.synchronize();
            }
        }

        void setHostDirty() override {
            shared_block->device_dirty = false;
            shared_block->host_dirty = true;
        }

        void copyToDevice() override {
            if (shared_block->host_dirty) {
                shared_block->host_dirty = false;
                shared_block->device_dirty = false;

                auto span = tensor_span_type(shared_block->device.get(), std::begin(shape), std::end(shape));
                cuda4dnn::csl::copyMatToTensor<T>(shared_block->host, span, shared_block->stream);
            }
        }

        void setDeviceDirty() override {
            shared_block->device_dirty = true;
            shared_block->host_dirty = false;
        }

        MatShape getShape() const noexcept override { return shape; }

        std::size_t getRank() const noexcept override { return shape.size(); }

        void setStream(cuda4dnn::csl::Stream stream) noexcept override {
            shared_block->stream = std::move(stream);
        }

        cv::Mat getMutableHostMat() noexcept {
            copyToHost();
            setHostDirty();
            return shared_block->host;
        }

        const cv::Mat getImmutableHostMat() const noexcept {
            copyToHost();
            return shared_block->host;
        }

        /* Optimization Note: use getSpan() and getView() judiciously
         *
         * getSpan() is meant to be used when the memory is going to be modified
         * getView() is meant to be used when the memory is only going to be read
         *
         * getSpan() marks the device memory as dirty but getView() does not
         *
         * getView() implicitly performs host to device memory transfer if required
         * getSpan() does not perform any synchronization (use copyToDevice if sync. is required)
         */
        tensor_span_type getSpan() noexcept {
            setDeviceDirty();
            return tensor_span_type(shared_block->device.get(), std::begin(shape), std::end(shape));
        }

        tensor_view_type getView() noexcept {
            copyToDevice();
            return tensor_view_type(shared_block->device.get(), std::begin(shape), std::end(shape));
        }

    private:
        /* The same tensor memory can be reused by different layers whenever possible.
         * Hence, it is possible for different backend warppers to point to the same memory.
         * However, it may use only a part of that memory and have a different shape.
         *
         * We store the common information such as device tensor and its corresponding host memory in
         * a shared block. The shared block is shared by all backend wrappers which use the same memory.
         * The shape, which can be different for different wrappers, is stored as a member object.
         */

        MatShape shape;

        struct shared_block_type {
            bool host_dirty;
            bool device_dirty;

            cv::Mat host;
            cuda4dnn::csl::MemoryLockGuard memGuard; /* keeps host memory page-locked if possible */

            cuda4dnn::csl::ManagedPtr<T> device;
            cuda4dnn::csl::Stream stream;
        };

        std::shared_ptr<shared_block_type> shared_block;
    };

    using CUDABackendWrapperFP16 = GenericCUDABackendWrapper<half, DNN_TARGET_CUDA_FP16>;
    using CUDABackendWrapperFP32 = GenericCUDABackendWrapper<float, DNN_TARGET_CUDA>;

    template <class T> struct GetCUDABackendWrapperType_ { };
    template <> struct GetCUDABackendWrapperType_<half> { typedef CUDABackendWrapperFP16 type; };
    template <> struct GetCUDABackendWrapperType_<float> { typedef CUDABackendWrapperFP32 type; };

    template <class T>
    using GetCUDABackendWrapperType = typename GetCUDABackendWrapperType_<T>::type;

#endif
}} /* namespace cv::dnn */

#endif /* OPENCV_DNN_SRC_OP_CUDA_HPP */
