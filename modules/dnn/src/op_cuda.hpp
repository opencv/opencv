// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_OP_CUDA_HPP
#define OPENCV_DNN_SRC_OP_CUDA_HPP

#ifdef HAVE_CUDA
#include "cuda4dnn/csl/stream.hpp"
#include "cuda4dnn/csl/tensor.hpp"
#include "cuda4dnn/cxx_utils/make_unique.hpp"
#endif

#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/core.hpp>

#include <cstddef>
#include <memory>
#include <iterator>

namespace cv {
    namespace dnn {
        constexpr bool IS_DNN_CUDA_TARGET(int id) {
            switch (id) {
            case DNN_TARGET_CUDA_FP32:
                return true;
            }
            return false;
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
            /** @brief creates Tensor object from cv::Mat (only the header is created, i.e. no data is copied)
             *
             * \tparam T    element type for the tensor
             */
            template <class T, class TensorT = typename Tensor<T>>
            TensorT makeTensorHeader(const Mat& mat)
            {
                auto sizes = shape(mat);
                return TensorT(std::begin(sizes), std::end(sizes));
            }

            /** @brief copies data from a cv::Mat and fills a TensorType
             *
             * Pre-conditions:
             * - \p mat must be larger or equal to the tensor in size
             *
             * @note performance is best for continuous and page-locked cv::Mat
             */
            template <class T>
            void copyMatToTensor(const TensorSpan<T> tensor, const Mat& mat, const Stream& stream);

            template <> inline
            void copyMatToTensor(const TensorSpan<float> tensor, const Mat& mat, const Stream& stream)
            {
                /* should perhaps convert cv::Mat of different type to the required type and copy */
                CV_Assert(mat.type() == CV_32F);
                CV_Assert(mat.total() >= tensor.size());

                Mat source = mat.isContinuous() ? mat : mat.clone();
                CV_Assert(source.isContinuous());

                memcpy<float>(tensor.get(), reinterpret_cast<float*>(source.data), tensor.size(), stream);
            }

            template <> inline
            void copyMatToTensor(const TensorSpan<double> tensor, const Mat& mat, const Stream& stream)
            {
                /* should perhaps convert cv::Mat of different type to the required type and copy */
                CV_Assert(mat.type() == CV_32F);
                CV_Assert(mat.total() >= tensor.size());

                Mat source;
                mat.convertTo(source, CV_64F);
                CV_Assert(source.isContinuous());

                memcpy<double>(tensor.get(), reinterpret_cast<double*>(source.data), tensor.size(), stream);
            }

            /** @brief copies data from a TensorType to a cv::Mat
             *
             * Pre-conditions:
             * - \p mat must be larger or equal to the tensor in size
             *
             * @note performance is best for continuous and page-locked cv::Mat
             */
            template <class T>
            void copyTensorToMat(Mat& mat, TensorView<T> tensor, const Stream& stream);

            template <> inline
            void copyTensorToMat(Mat& mat, TensorView<float> tensor, const Stream& stream)
            {
                CV_Assert(mat.type() == CV_32F);
                CV_Assert(mat.total() >= tensor.size());

                Mat source = mat.isContinuous() ? mat : mat.clone();
                CV_Assert(source.isContinuous());

                memcpy<float>(reinterpret_cast<float*>(source.data), tensor.get(), tensor.size(), stream);

                if (source.data != mat.data)
                    source.copyTo(mat);
            }

            template <> inline
            void copyTensorToMat(Mat& mat, TensorView<double> tensor, const Stream& stream)
            {
                CV_Assert(mat.type() == CV_32F);
                CV_Assert(mat.total() >= tensor.size());

                Mat source(shape(mat), CV_64F);
                CV_Assert(source.isContinuous());

                memcpy<double>(reinterpret_cast<double*>(source.data), tensor.get(), tensor.size(), stream);

                source.convertTo(mat, CV_32F);
            }
        }} /* cuda4dnn::csl */

        /* base class for all CUDA backend/target node */
        class CUDABackendNode : public BackendNode {
        public:
            CUDABackendNode() : BackendNode(DNN_BACKEND_CUDA) { }
            virtual ~CUDABackendNode() { }

            virtual void forward(
                std::vector<cv::Ptr<BackendWrapper>>& inputs,
                std::vector<cv::Ptr<BackendWrapper>>& outputs,
                cuda4dnn::csl::Workspace& workspace) = 0;

            virtual std::size_t get_workspace_memory_in_bytes() const noexcept { return 0; }
        };

        /* utility function which creates a correct backend node based on `targetId` */
        template <template <class> class NodeType, class ...Args>
        std::unique_ptr<CUDABackendNode> make_cuda_node(int targetId, Args&& ...args)
        {
            switch (targetId)
            {
            case DNN_TARGET_CUDA_FP32:
                return cuda4dnn::cxx_utils::make_unique<NodeType<float>>(std::forward<Args>(args)...);
            default:
                CV_Assert(IS_DNN_CUDA_TARGET(targetId));
            }
            return nullptr;
        }

        /* base class for all CUDA backend/target wrappers */
        class CUDABackendWrapper : public BackendWrapper {
        public:
            CUDABackendWrapper(int targetId) : BackendWrapper(DNN_BACKEND_CUDA, targetId) { }
            virtual ~CUDABackendWrapper() { }

            virtual void copyToHost() = 0;
            virtual void setHostDirty() = 0;

            virtual void copyToDevice() = 0;
            virtual void setDeviceDirty() = 0;

            virtual MatShape getShape() const noexcept = 0;
            virtual std::size_t getRank() const noexcept = 0;

            /** @note setting the stream updates the stream for all wrappers which use the same buffer */
            virtual void setStream(cuda4dnn::csl::Stream stream) noexcept = 0;
        };

        /* the code for different wrappers barely change; hence, we use this template as a wrapper generator */
        template <class T, int targetId>
        class GenericCUDABackendWrapper final : public CUDABackendWrapper {
        public:
            using value_type = T;
            using tensor_type       = cuda4dnn::csl::Tensor<value_type>;
            using tensor_span_type  = cuda4dnn::csl::TensorSpan<value_type>;
            using tensor_view_type  = cuda4dnn::csl::TensorView<value_type>;

            /* Pre-conditions:
             * - there must be no other instance of `GenericCUDABackendWrapper` which wraps the host memory used by `m`
             * - the host memory must remain allocated throughout the lifetime of this object
             * - the host memory must be pageable memory
             *
             * Post-conditions:
             * - the host memory used by `m` is page-locked
             */
            GenericCUDABackendWrapper(Mat& m)
                : CUDABackendWrapper(targetId)
            {
                CV_Assert(m.isContinuous());
                CV_Assert(m.size.dims() <= tensor_type::rank);

                shape = cv::dnn::shape(m);

                shared_block = std::make_shared<shared_block_type>();
                shared_block->host_dirty = true;
                shared_block->device_dirty = false;
                shared_block->host = m;
                shared_block->memGuard = cuda4dnn::csl::MemoryLockGuard(m.data, m.total() * m.elemSize());
                shared_block->parent = cuda4dnn::csl::makeTensorHeader<T>(m);
            }

            GenericCUDABackendWrapper(const Ptr<BackendWrapper>& base_, const MatShape& shape_)
                : CUDABackendWrapper(targetId)
            {
                const Ptr<GenericCUDABackendWrapper> base = base_.dynamicCast<GenericCUDABackendWrapper>();

                shape = shape_;
                shared_block = base->shared_block;
            }

            static Ptr<BackendWrapper> create(Mat& m)
            {
                return Ptr<BackendWrapper>(new GenericCUDABackendWrapper(m));
            }

            static Ptr<BackendWrapper> create(const Ptr<BackendWrapper>& base, const MatShape& shape)
            {
                return Ptr<BackendWrapper>(new GenericCUDABackendWrapper(base, shape));
            }

            void copyToHost() override {
                if (shared_block->device_dirty) {
                    shared_block->host_dirty = false;
                    shared_block->device_dirty = false;

                    /* If the wrapper is being reused, the device tensor might be larger in size.
                     * Using the device tensor does not give incorrect code, but it leads to unused regions
                     * of memory being copied.
                     *
                     * We use a view to ensure that only the required region of memory is copied.
                     */
                    auto view = tensor_view_type(shared_block->parent, 0, std::begin(shape), std::end(shape));
                    cuda4dnn::csl::copyTensorToMat<T>(shared_block->host, view, shared_block->stream);

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

                    auto span = tensor_span_type(shared_block->parent, 0, std::begin(shape), std::end(shape));
                    cuda4dnn::csl::copyMatToTensor<T>(span, shared_block->host, shared_block->stream);
                }
            }

            void setDeviceDirty() override {
                shared_block->device_dirty = true;
                shared_block->host_dirty = false;
            }

            /** @brief returns the wrapped cv::Mat's shape */
            MatShape getShape() const noexcept override { return shape; }

            /** @brief returns the rank of the csl::Tensor */
            std::size_t getRank() const noexcept override { return shared_block->parent.rank; }

            /** @note setting the stream updates the stream for all wrappers which use the same buffer */
            void setStream(cuda4dnn::csl::Stream stream) noexcept override {
                shared_block->stream = std::move(stream);
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
                return tensor_span_type(shared_block->parent, 0, std::begin(shape), std::end(shape));
            }

            tensor_view_type getView() noexcept {
                copyToDevice();
                return tensor_view_type(shared_block->parent, 0, std::begin(shape), std::end(shape));
            }

        private:
            /* The same device memory can be reused by different layers whenever possible.
             * Hence, it is possible for different backend warppers to point to the same device memory.
             * However, it may use only a part of the total device memory and have a different shape.
             *
             * We store the common information such as device tensor and its corresponding host memory in
             * a shared block. The shared block is shared by all backend wrappers which use the same device memory.
             * The shape, which can be different for different wrappers, is stored as a member object.
             */

            MatShape shape;

            struct shared_block_type {
                bool host_dirty;
                bool device_dirty;

                cv::Mat host;
                cuda4dnn::csl::MemoryLockGuard memGuard; /* keeps host memory page-locked */

                tensor_type parent;
                cuda4dnn::csl::Stream stream;
            };

            std::shared_ptr<shared_block_type> shared_block;
        };

        template <class T> constexpr int getCUDATarget();
        template <> constexpr int getCUDATarget<float>() { return DNN_TARGET_CUDA_FP32; }

        using CUDABackendWrapperFP32 = GenericCUDABackendWrapper<float, getCUDATarget<float>()>;

        template <class T> struct GetCUDABackendWrapperType_ { };
        template <> struct GetCUDABackendWrapperType_<float> { typedef CUDABackendWrapperFP32 type; };

        template <class T>
        using GetCUDABackendWrapperType = typename GetCUDABackendWrapperType_<T>::type;

#endif
    } /* namespace dnn */
}  /* namespace cv */

#endif  /* OPENCV_DNN_SRC_OP_CUDA_HPP */
