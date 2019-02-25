// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_LOGTAGMANAGER_HPP
#define OPENCV_CORE_LOGTAGMANAGER_HPP

#if 1 // if not already in precompiled headers
#include <string>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <mutex>
#endif

#include <opencv2/core/utils/logtag.hpp>

namespace cv {
namespace utils {
namespace logging {

// A lookup table of LogTags using full name, first part of name, and any part of name.
// The name parts of a LogTag is delimited by period.
//
// This class does not handle wildcard characters. The name-matching method can only be
// selected by calling the appropriate function.
//
class LogTagManager
{
public:
    LogTagManager();
    ~LogTagManager();

public:
    // Add (register) the log tag.
    // Note, passing in nullptr as value is equivalent to unassigning.
    void assign(const std::string& name, LogTag* value);

    // Unassign the log tag. This is equivalent to calling assign with nullptr value.
    void unassign(const std::string& name);

    // Retrieve the log tag by exact name.
    LogTag* get(const std::string& name) const;

    // Given full name of a log tag, invoke the function while inside the lock.
    void invoke(const std::string& name, std::function<void(LogTag*)> func);

    // For each log tag having the specified first part of prefix (up to the first period)
    // invoke the function inside the lock.
    void forEach_byFirstPart(const std::string& firstPart, std::function<void(LogTag*)> func);

    // For each log tag having any matching part, invoke the function inside the lock.
    void forEach_byAnyPart(const std::string& anyPart, std::function<void(LogTag*)> func);

private:
    void internal_forEachNamePart(const std::string& name, std::function<void(const std::string& namePart)> namePartFunc);

private:
    // Current implementation does not seem to require recursive mutex;
    // also, extensible functions (accepting user-provided callback) are not allowed
    // to call LogTagManger (to prevent iterator invalidation), which needs enforced
    // with a non-recursive mutex.
    using MutexType = std::mutex;
    using LockType = std::lock_guard<MutexType>;

    struct LogTagAndId
    {
        size_t id;
        LogTag* value;
    };

private:
    mutable MutexType m_mutex;
    std::unordered_map<std::string, LogTagAndId> m_byFullName;
    std::unordered_multimap<std::string, LogTagAndId> m_byFirstPart;
    std::unordered_multimap<std::string, LogTagAndId> m_byAnyPart;
    size_t m_nextId;
};

}}} //namespace

#endif //OPENCV_CORE_LOGTAGMANAGER_HPP
