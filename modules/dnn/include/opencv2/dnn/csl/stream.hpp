// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CSL_STREAM_HPP
#define OPENCV_DNN_CSL_STREAM_HPP

#include <opencv2/core.hpp>

#include <memory>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

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
        Stream();
        Stream(const Stream&) noexcept;
        Stream(Stream&&) noexcept;

        //!< if \p create is `true`, a new stream will be created instead of the otherwise default stream
        Stream(bool create);

        Stream& operator=(const Stream&) noexcept;
        Stream& operator=(Stream&&) noexcept;

        //!< blocks the caller thread until all operations in the stream complete
        void synchronize() const;

        //!< returns true if there are operations pending in the stream
        bool busy() const;

        //!< returns true if the stream is valid
        explicit operator bool() const noexcept;

    private:
        friend class StreamAccessor;

        class UniqueStream;
        std::shared_ptr<UniqueStream> stream;
    };

}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_CSL_STREAM_HPP */
