// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_CSL_STREAM_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_CSL_STREAM_HPP

#include "error.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <cuda_runtime_api.h>

#include <memory>
#include <sstream>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    /** @brief noncopyable smart CUDA stream
     *
     * UniqueStream is a smart non-sharable wrapper for CUDA stream handle which ensures that
     * the handle is destroyed after use. Unless explicitly specified by a constructor argument,
     * the stream object represents the default stream.
     */
    class UniqueStream {
    public:
        UniqueStream() noexcept : stream{ 0 } { }
        UniqueStream(UniqueStream&) = delete;
        UniqueStream(UniqueStream&& other) noexcept {
            stream = other.stream;
            other.stream = 0;
        }

        UniqueStream(bool create) : stream{ 0 } {
            if (create) {
                CUDA4DNN_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            }
        }

        ~UniqueStream() {
            try {
                if (stream != 0)
                    CUDA4DNN_CHECK_CUDA(cudaStreamDestroy(stream));
            } catch (const CUDAException& ex) {
                std::ostringstream os;
                os << "Asynchronous exception caught during CUDA stream destruction.\n";
                os << ex.what();
                os << "Exception will be ignored.\n";
                CV_LOG_WARNING(0, os.str().c_str());
            }
        }

        UniqueStream& operator=(const UniqueStream&) = delete;
        UniqueStream& operator=(UniqueStream&& other) noexcept {
            stream = other.stream;
            other.stream = 0;
            return *this;
        }

        /** returns the raw CUDA stream handle */
        cudaStream_t get() const noexcept { return stream; }

        void synchronize() const { CUDA4DNN_CHECK_CUDA(cudaStreamSynchronize(stream)); }
        bool busy() const {
            auto status = cudaStreamQuery(stream);
            if (status == cudaErrorNotReady)
                return true;
            CUDA4DNN_CHECK_CUDA(status);
            return false;
        }

    private:
        cudaStream_t stream;
    };

    /** @brief sharable smart CUDA stream
     *
     * Stream is a smart sharable wrapper for CUDA stream handle which ensures that
     * the handle is destroyed after use. Unless explicitly specified by a constructor argument,
     * the stream object represents the default stream.
     *
     * @note Moving a Stream object to another invalidates the former
     */
    class Stream {
    public:
        Stream() : stream(std::make_shared<UniqueStream>()) { }
        Stream(const Stream&) = default;
        Stream(Stream&&) = default;

        /** if \p create is `true`, a new stream will be created instead of the otherwise default stream */
        Stream(bool create) : stream(std::make_shared<UniqueStream>(create)) { }

        Stream& operator=(const Stream&) = default;
        Stream& operator=(Stream&&) = default;

        /** blocks the caller thread until all operations in the stream are complete */
        void synchronize() const { stream->synchronize(); }

        /** returns true if there are operations pending in the stream */
        bool busy() const { return stream->busy(); }

        /** returns true if the stream is valid */
        explicit operator bool() const noexcept { return static_cast<bool>(stream); }

        cudaStream_t get() const noexcept {
            CV_Assert(stream);
            return stream->get();
        }

    private:
        std::shared_ptr<UniqueStream> stream;
    };

}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_CSL_STREAM_HPP */
