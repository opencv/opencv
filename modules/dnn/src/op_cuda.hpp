// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_OP_CUDA_HPP
#define OPENCV_DNN_SRC_OP_CUDA_HPP

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
        /* CUDA Tensors are represented by csl::Tensor
        ** CUDABackendWrapperFP32 wraps a csl::TensorSpan<float>
        ** It also maintains a reference to the csl::Tensor.
        */
        class CUDABackendWrapperFP32 : public BackendWrapper {
        public:
            CUDABackendWrapperFP32(Mat& m) : BackendWrapper(DNN_BACKEND_OPENCV, DNN_TARGET_CUDA_FP32) {
                /* TODO:
                ** 1. store a reference to cv::Mat
                ** 2. create a csl::Tensor<float>
                ** 3. create a csl::TensorSpan<float> (or store shape)
                */
            }

            CUDABackendWrapperFP32(const Ptr<BackendWrapper>& base, const MatShape& shape)
                : BackendWrapper(DNN_BACKEND_OPENCV, DNN_TARGET_CUDA_FP32) {
                /* TODO:
                ** 1. copy reference to csl::Tensor<float> of base
                ** 2. set TensorSpan<float> to mimic `shape` (or store shape)
                */
            }

            static Ptr<BackendWrapper> create(Mat& m)
            {
                return Ptr<BackendWrapper>(new CUDABackendWrapperFP32(m));
            }

            static Ptr<BackendWrapper> create(const Ptr<BackendWrapper>& base, const MatShape& shape)
            {
                return Ptr<BackendWrapper>(new CUDABackendWrapperFP32(base, shape));
            }

            virtual void copyToHost() CV_OVERRIDE { }
            virtual void setHostDirty() CV_OVERRIDE { }

            //TensorSpan<float> getSpan();
            //TensorView<float> getView();

            /* TensorSpan member vs create in getSpan()
            ** member tensor span can save shape changes
            */
        };
#endif
    } /* namespace dnn */
}  /* namespace cv */

#endif  /* OPENCV_DNN_SRC_OP_CUDA_HPP */
