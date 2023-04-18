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

    /** \file stream.hpp
     *
     * Default streams are not supported as they limit flexiblity. All operations are always
     * carried out in non-default streams in the CUDA backend. The stream classes sacrifice
     * the ability to support default streams in exchange for better error detection. That is,
     * a default constructed stream represents no stream and any attempt to use it will throw an
     * exception.
     */

    /** @brief non-copyable smart CUDA stream
     *
     * UniqueStream is a smart non-sharable wrapper for CUDA stream handle which ensures that
     * the handle is destroyed after use. Unless explicitly specified by a constructor argument,
     * the stream object does not represent any stream by default.
     */
    class UniqueStream {
    public:
        UniqueStream() noexcept : stream{ 0 } { }
        UniqueStream(UniqueStream&) = delete;
        UniqueStream(UniqueStream&& other) noexcept {
            stream = other.stream;
            other.stream = 0;
        }

        /** creates a non-default stream if `create` is true; otherwise, no stream is created */
        UniqueStream(bool create) : stream{ 0 } {
            if (create) {
                /* we create non-blocking streams to avoid inrerruptions from users using the default stream */
                CUDA4DNN_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            }
        }

        ~UniqueStream() {
            try {
                /* cudaStreamDestroy does not throw if a valid stream is passed unless a previous
                 * asynchronous operation errored.
                 */
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
            CV_Assert(other);
            if (&other != this) {
                UniqueStream(std::move(*this)); /* destroy current stream */
                stream = other.stream;
                other.stream = 0;
            }
            return *this;
        }

        /** returns the raw CUDA stream handle */
        cudaStream_t get() const noexcept {
            CV_Assert(stream);
            return stream;
        }

        /** blocks the calling thread until all pending operations in the stream finish */
        void synchronize() const {
            CV_Assert(stream);
            CUDA4DNN_CHECK_CUDA(cudaStreamSynchronize(stream));
        }

        /** returns true if there are pending operations in the stream */
        bool busy() const {
            CV_Assert(stream);

            auto status = cudaStreamQuery(stream);
            if (status == cudaErrorNotReady)
                return true;
            CUDA4DNN_CHECK_CUDA(status);
            return false;
        }

        /** returns true if the stream is valid */
        explicit operator bool() const noexcept { return static_cast<bool>(stream); }

    private:
        cudaStream_t stream;
    };

    /** @brief sharable smart CUDA stream
     *
     * Stream is a smart sharable wrapper for CUDA stream handle which ensures that
     * the handle is destroyed after use. Unless explicitly specified in the constructor,
     * the stream object represents no stream.
     */
    class Stream {
    public:
        Stream() { }
        Stream(const Stream&) = default;
        Stream(Stream&&) = default;

        /** if \p create is `true`, a new stream will be created; otherwise, no stream is created */
        Stream(bool create) {
            if (create)
                stream = std::make_shared<UniqueStream>(create);
        }

        Stream& operator=(const Stream&) = default;
        Stream& operator=(Stream&&) = default;

        /** blocks the caller thread until all operations in the stream are complete */
        void synchronize() const {
            CV_Assert(stream);
            stream->synchronize();
        }

        /** returns true if there are operations pending in the stream */
        bool busy() const {
            CV_Assert(stream);
            return stream->busy();
        }

        /** returns true if the object points has a valid stream */
        explicit operator bool() const noexcept {
            if (!stream)
                return false;
            return stream->operator bool();
        }

        cudaStream_t get() const noexcept {
            CV_Assert(stream);
            return stream->get();
        }

    private:
        std::shared_ptr<UniqueStream> stream;
    };

}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_CSL_STREAM_HPP */
