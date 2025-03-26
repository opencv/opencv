// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_LOGCALLBACKMANAGER_HPP
#define OPENCV_CORE_LOGCALLBACKMANAGER_HPP

#if 1 // if not already in precompiled headers
#include <mutex>
#include <unordered_map>
#include <unordered_set> // Only used by deprecated(LoggingCallbackPtrType)
#include <array>
#include <memory>
#endif

#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/utils/logger.defines.hpp"

#define ALLOW_DEPRECATED_CODE
// #ifdef ALLOW_DEPRECATED_CODE
// #  undef ALLOW_DEPRECATED_CODE
// #endif
#ifdef ALLOW_CONTROVERSIAL_CODE
#  undef ALLOW_CONTROVERSIAL_CODE
#endif

namespace cv {
namespace utils {
namespace logging {

class LogCallbackManager
{
private:
    using MutexType = cv::Mutex;
    using LockType = cv::AutoLock;

public:
    explicit LogCallbackManager();
    ~LogCallbackManager();

    /**
     * @brief Returns the total count of all registered callbacks, including C-style static functions
     *        and handler objects.
     */
    size_t count() const;

    bool contains(const cv::Ptr<LoggingCallbackHandler>& handler) const;
    void add(cv::Ptr<LoggingCallbackHandler> handler);
    void remove(const cv::Ptr<LoggingCallbackHandler>& handler);
    void readInto(std::vector<cv::Ptr<LoggingCallbackHandler>>& handlers) const;

#ifdef ALLOW_DEPRECATED_CODE
bool contains(LoggingCallbackPtrType callback) const;
void add(LoggingCallbackPtrType callback);
void remove(LoggingCallbackPtrType callback);
void readInto(std::vector<LoggingCallbackPtrType>& callbacks) const;
#endif //ALLOW_DEPRECATED_CODE

    /**
     * @brief Remove all registered callbacks, including C-style static functions
     *        and handler objects.
     */
    void removeAll();

#ifdef ALLOW_CONTROVERSIAL_CODE
    /**
     * @brief Read as many callbacks as possible into a fixed-size receiving array.
     * @param callbacks The receiving array. Unused entries will be set to nullptr.
     * @param arr_sz Capacity of the receiving array.
     * @return Actual number of callbacks that have been registered. If this number
     *         is greater than arr_sz, some callbacks have not been copied.
     */
    template <size_t arr_sz>
    size_t tryReadInto(std::array<LoggingCallbackPtrType, arr_sz>& callbacks) const
    {
        return this->tryReadIntoSz(callbacks.data(), arr_sz);
    }

    /**
     * @brief Read as many callbacks as possible into a fixed-size receiving array.
     * @param callbacks The receiving array. Unused entries will be set to nullptr.
     * @param arr_sz Capacity of the receiving array.
     * @return Actual number of callbacks that have been registered. If this number
     *         is greater than arr_sz, some callbacks have not been copied.
     */
    template <size_t arr_sz>
    size_t tryReadInto(LoggingCallbackPtrType (&callbacks)[arr_sz]) const
    {
        return this->tryReadIntoSz(&callbacks, arr_sz);
    }
#endif //ALLOW_CONTROVERSIAL_CODE

private:
    LogCallbackManager(const LogCallbackManager&) = delete;
    LogCallbackManager(LogCallbackManager&&) = delete;
    LogCallbackManager& operator=(const LogCallbackManager&) = delete;
    LogCallbackManager& operator=(LogCallbackManager&&) = delete;

#ifdef ALLOW_CONTROVERSIAL_CODE
    size_t tryReadIntoSz(LoggingCallbackPtrType (&callbacks)[], size_t arr_sz) const;
#endif //ALLOW_CONTROVERSIAL_CODE

private:
    mutable MutexType m_mutex;
    std::unordered_map<const LoggingCallbackHandler*, cv::Ptr<LoggingCallbackHandler> > m_handlers;

#ifdef ALLOW_DEPRECATED_CODE
    std::unordered_set<LoggingCallbackPtrType> m_callbacks;
#endif //ALLOW_DEPRECATED_CODE
};

}}} //namespace

#endif //OPENCV_CORE_LOGCALLBACKMANAGER_HPP
