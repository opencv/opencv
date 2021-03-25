// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_CSL_EVENT_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_CSL_EVENT_HPP

#include "error.hpp"
#include "stream.hpp"

#include <opencv2/core/utils/logger.hpp>

#include <cuda_runtime_api.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    /** @brief sharable CUDA event
     *
     * Event is a smart sharable wrapper for CUDA event handle which ensures that
     * the handle is destroyed after use.
     *
     * @note Moving an Event object to another invalidates the former
     */
    class Event {
    public:
        Event() noexcept : event{ nullptr } { }
        Event(const Event&) = delete;
        Event(Event&& other) noexcept
            : event{ other.event } {
            other.event = nullptr;
        }

        /** if \p create is `true`, a new event will be created; otherwise, an empty event object is created */
        Event(bool create, bool timing_event = false) : event{nullptr} {
            if (create) {
                unsigned int flags = (timing_event ? 0 : cudaEventDisableTiming);
                CUDA4DNN_CHECK_CUDA(cudaEventCreateWithFlags(&event, flags));
            }
        }

        ~Event() {
            try {
                if (event != nullptr)
                    CUDA4DNN_CHECK_CUDA(cudaEventDestroy(event));
            } catch (const CUDAException& ex) {
                std::ostringstream os;
                os << "Asynchronous exception caught during CUDA event destruction.\n";
                os << ex.what();
                os << "Exception will be ignored.\n";
                CV_LOG_WARNING(0, os.str().c_str());
            }
        }

        Event& operator=(const Event&) noexcept = delete;
        Event& operator=(Event&& other) noexcept {
            event = other.event;
            other.event = nullptr;
            return *this;
        }

        /** mark a point in \p stream */
        void record(const Stream& stream) {
            CV_Assert(stream);
            CUDA4DNN_CHECK_CUDA(cudaEventRecord(event, stream.get()));
        }

        /** blocks the caller thread until all operations before the event finish */
        void synchronize() const { CUDA4DNN_CHECK_CUDA(cudaEventSynchronize(event)); }

        /** returns true if there are operations pending before the event completes */
        bool busy() const {
            auto status = cudaEventQuery(event);
            if (status == cudaErrorNotReady)
                return true;
            CUDA4DNN_CHECK_CUDA(status);
            return false;
        }

        cudaEvent_t get() const noexcept { return event; }

        /** returns true if the event is valid */
        explicit operator bool() const noexcept { return event; }

    private:
        cudaEvent_t event;
    };

    /** makes a stream wait on an event */
    inline void StreamWaitOnEvent(const Stream& stream, const Event& event) {
        CV_Assert(stream);
        CUDA4DNN_CHECK_CUDA(cudaStreamWaitEvent(stream.get(), event.get(), 0));
    }

    /** returns the time elapsed between two events in milliseconds */
    inline float TimeElapsedBetweenEvents(const Event& start, const Event& end) {
        float temp;
        CUDA4DNN_CHECK_CUDA(cudaEventElapsedTime(&temp, start.get(), end.get()));
        return temp;
    }

}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_CSL_EVENT_HPP */
