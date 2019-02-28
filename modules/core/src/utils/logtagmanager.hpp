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
#include <memory>
#endif

#include <opencv2/core/utils/logtag.hpp>
#include "logtagconfig.hpp"

namespace cv {
namespace utils {
namespace logging {

// forward declaration
class LogTagConfigParser;

// A lookup table of LogTags using full name, first part of name, and any part of name.
// The name parts of a LogTag is delimited by period.
//
// This class does not handle wildcard characters. The name-matching method can only be
// selected by calling the appropriate function.
//
class LogTagManager
{
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
        LogTag* ptr;
    };

    struct ParsedLevel
    {
        bool valid;
        LogLevel level;

        ParsedLevel()
            : valid(false)
            , level()
        {
        }
    };

    struct FullNameInfo
    {
        LogTagAndId member;
        ParsedLevel parsedLevel;
    };

    using FullNamePair = std::pair<const std::string, FullNameInfo>;

    struct WildcardInfo
    {
        std::vector<LogTagAndId> members;
        ParsedLevel parsedLevel;
    };

    using WildcardPair = std::pair<const std::string, WildcardInfo>;

public:
    LogTagManager(LogLevel defaultUnconfiguredGlobalLevel);
    ~LogTagManager();

public:
    // Parse and apply the config string.
    void setConfigString(const std::string& configString, bool apply = true);

    // Gets the config parser. This is necessary to retrieve the list of malformed strings.
    LogTagConfigParser& getConfigParser() const;

    // Add (register) the log tag.
    // Note, passing in nullptr as value is equivalent to unassigning.
    void assign(const std::string& name, LogTag* ptr);

    // Unassign the log tag. This is equivalent to calling assign with nullptr value.
    void unassign(const std::string& name);

    // Retrieve the log tag by exact name.
    LogTag* get(const std::string& name) const;

    // Changes the log level of the tag having the exact full name.
    void setLevelByFullName(const std::string& fullName, LogLevel level);

    // Changes the log level of the tag matching the first part of the name.
    void setLevelByFirstPart(const std::string& firstPart, LogLevel level);

    // Changes the log level of the tag matching any part of the name.
    void setLevelByAnyPart(const std::string& anyPart, LogLevel level);

private:
    const FullNamePair& assignFullName(const std::string& fullName, LogTag* ptr);
    void assignNameParts(const FullNamePair& fullNamePair);
    void assignWildcardInfo(const FullNamePair& fullNamePair, const std::string& namePart, bool isFirst);
    LogTagAndId& addOrGetMember(std::vector<LogTagAndId>& members, const LogTagAndId& memberArg);
    void setLevelByWildcard(const std::string& namePart, LogLevel level, bool isFirst);

private:
    static std::vector<std::string> splitNameParts(const std::string& fullName);

private:
    static constexpr const char* m_globalName = "global";

private:
    mutable MutexType m_mutex;
    std::unique_ptr<LogTag> m_globalLogTag;
    std::unordered_map<std::string, FullNameInfo> m_fullNames;
    std::unordered_map<std::string, WildcardInfo> m_firstParts;
    std::unordered_map<std::string, WildcardInfo> m_anyParts;
    size_t m_nextId;
    std::shared_ptr<LogTagConfigParser> m_config;
};

}}} //namespace

#endif //OPENCV_CORE_LOGTAGMANAGER_HPP
