// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_OP_CUDA_HPP
#define OPENCV_DNN_SRC_OP_CUDA_HPP

#ifdef HAVE_CUDA
#include "cuda4dnn/csl/stream.hpp"
#include "cuda4dnn/csl/tensor.hpp"
#include "cuda4dnn/csl/pointer.hpp"
#endif

#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/core.hpp>

#include <memory>

namespace cv {
    namespace dnn {
        inline bool haveCUDA() {
#ifdef HAVE_CUDA
            return true;
#else
            return false;
#endif
        }

#ifdef HAVE_CUDA
        /** @brief creates csl::Tensor object from cv::Mat */
        template <class TensorT = cuda4dnn::csl::Tensor<float>> inline
        TensorT createTensorHeaderFromMat(const cv::Mat& mat) {
            auto is_matrix_type_same_as_tensor_type = [&mat]() {
                switch (mat.type()) {
                case CV_32F: return std::is_same<TensorT::value_type, float>::value;
                default: return false;
                }
            };
            CV_Assert(is_matrix_type_same_as_tensor_type());

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
        template <class T> inline
        void copyMatToTensor(const cuda4dnn::csl::TensorSpan<T> tensor, const cv::Mat& mat, const cuda4dnn::csl::Stream& stream) {
            CV_Assert(mat.total() >= tensor.size());

            cv::Mat source = mat.isContinuous() ? mat : mat.clone();
            CV_Assert(source.isContinuous());

            cuda4dnn::csl::memcpy<T>(tensor.get(), reinterpret_cast<T*>(source.data), tensor.size(), stream);
        }

        /** @brief copies data from a TensorType to a cv::Mat
         *
         * Pre-conditions:
         * - \p mat must be larger or equal to the tensor in size
         *
         * @note performance is best for continuous and page-locked cv::Mat
         */
        template <class T> inline
        void copyTensorToMat(cv::Mat& mat, cuda4dnn::csl::TensorView<T> tensor, const cuda4dnn::csl::Stream& stream) {
            CV_Assert(mat.total() >= tensor.size());

            cv::Mat source = mat.isContinuous() ? mat : mat.clone();
            CV_Assert(source.isContinuous());

            cuda4dnn::csl::memcpy<T>(reinterpret_cast<T*>(source.data), tensor.get(), tensor.size(), stream);

            if(source.data != mat.data)
                source.copyTo(mat);
        }

        class CUDABackendWrapperFP32 final : public BackendWrapper {
        public:
            using value_type = float;
            using tensor_type = cuda4dnn::csl::Tensor<value_type>;
            using tensor_span_type = cuda4dnn::csl::TensorSpan<value_type>;
            using tensor_view_type = cuda4dnn::csl::TensorView<value_type>;

            CUDABackendWrapperFP32(Mat&);
            CUDABackendWrapperFP32(const Ptr<BackendWrapper>& base, const MatShape& shape);

            static Ptr<BackendWrapper> create(Mat&);
            static Ptr<BackendWrapper> create(const Ptr<BackendWrapper>& base, const MatShape& shape);

            void copyToHost() override;
            void setHostDirty() override;

            void copyToDevice();
            void setDeviceDirty() noexcept;

            MatShape getShape() const noexcept { return shape; }

            /** @note setting the stream updates the stream for all wrappers which use the same buffer */
            void setStream(cuda4dnn::csl::Stream stream) noexcept;

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
            tensor_span_type getSpan() noexcept;
            tensor_view_type getView() noexcept;

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
#endif
    } /* namespace dnn */
}  /* namespace cv */

#endif  /* OPENCV_DNN_SRC_OP_CUDA_HPP */
