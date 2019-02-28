// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "logtagmanager.hpp"
#include "logtagconfigparser.hpp"

namespace cv {
namespace utils {
namespace logging {

LogTagManager::LogTagManager(LogLevel defaultUnconfiguredGlobalLevel)
    : m_mutex()
    , m_globalLogTag(new LogTag(m_globalName, defaultUnconfiguredGlobalLevel))
    , m_fullNames()
    , m_firstParts()
    , m_anyParts()
    , m_nextId(1u)
    , m_config(std::make_shared<LogTagConfigParser>())
{
    assign(m_globalName, m_globalLogTag.get());
}

LogTagManager::~LogTagManager()
{
}

void LogTagManager::setConfigString(const std::string& configString, bool apply /*true*/)
{
    m_config->parse(configString);
    if (m_config->hasMalformed())
    {
        return;
    }
    if (!apply)
    {
        return;
    }
    // The following code is arranged with "priority by overwriting",
    // where when the same log tag has multiple matches, the last code
    // block has highest priority by literally overwriting the effects
    // from the earlier code blocks.
    //
    // Matching by full name has highest priority.
    // Matching by any name part has moderate priority.
    // Matching by first name part (prefix) has lowest priority.
    //
    const auto& globalConfig = m_config->getGlobalConfig();
    m_globalLogTag->level = globalConfig.level;
    for (const auto& config : m_config->getFirstPartConfigs())
    {
        setLevelByFirstPart(config.namePart, config.level);
    }
    for (const auto& config : m_config->getAnyPartConfigs())
    {
        setLevelByAnyPart(config.namePart, config.level);
    }
    for (const auto& config : m_config->getFullNameConfigs())
    {
        setLevelByFullName(config.namePart, config.level);
    }
}

LogTagConfigParser& LogTagManager::getConfigParser() const
{
    return *m_config;
}

void LogTagManager::assign(const std::string& name, LogTag* ptr)
{
    LockType lock(m_mutex);
    assignFullName(name, ptr);
}

void LogTagManager::unassign(const std::string& name)
{
    // Lock is inside assign() method.
    assign(name, nullptr);
}

LogTag* LogTagManager::get(const std::string& name) const
{
    LockType lock(m_mutex);
    const auto iter = m_fullNames.find(name);
    if (iter == m_fullNames.end())
    {
        return nullptr;
    }
    return iter->second.member.ptr;
}

void LogTagManager::setLevelByFullName(const std::string& fullName, LogLevel level)
{
    auto iter = m_fullNames.find(fullName);
    if (iter == m_fullNames.end())
    {
        iter = m_fullNames.emplace(fullName, FullNameInfo{}).first;
    }
    auto& info = iter->second;
    // skip additional processing if nothing changes.
    if (info.parsedLevel.valid &&
        info.parsedLevel.level == level)
    {
        return;
    }
    // update the cached configured value.
    info.parsedLevel.valid = true;
    info.parsedLevel.level = level;
    // update the actual tag, if already registered.
    if (info.member.ptr)
    {
        info.member.ptr->level = level;
    }
}

void LogTagManager::setLevelByFirstPart(const std::string& firstPart, LogLevel level)
{
    const bool isFirst = true;
    setLevelByWildcard(firstPart, level, isFirst);
}

void LogTagManager::setLevelByAnyPart(const std::string& anyPart, LogLevel level)
{
    const bool isFirst = false;
    setLevelByWildcard(anyPart, level, isFirst);
}

void LogTagManager::setLevelByWildcard(const std::string& namePart, LogLevel level, bool isFirst)
{
    auto& wildcardMap = isFirst ? m_firstParts : m_anyParts;
    auto iter = wildcardMap.find(namePart);
    if (iter == wildcardMap.end())
    {
        iter = wildcardMap.emplace(namePart, WildcardInfo{}).first;
    }
    auto& info = iter->second;
    // skip additional processing if nothing changes.
    if (info.parsedLevel.valid &&
        info.parsedLevel.level == level)
    {
        return;
    }
    // update the cached configured value.
    info.parsedLevel.valid = true;
    info.parsedLevel.level = level;
    // update the actual tag(s), if already registered.
    for (auto& member : info.members)
    {
        if (member.ptr)
        {
            member.ptr->level = level;
        }
    }
}

const LogTagManager::FullNamePair& LogTagManager::assignFullName(const std::string& fullName, LogTag* ptr)
{
    auto iter = m_fullNames.find(fullName);
    if (iter == m_fullNames.end())
    {
        // full name never seen before, neither from tag registration nor from parsed config
        const size_t newId = (m_nextId++);
        iter = m_fullNames.emplace(fullName, FullNameInfo{}).first;
        auto& info = iter->second;
        info.member.id = newId;
        info.member.ptr = ptr;
        return *iter;
    }
    else
    {
        // full name has been seen.
        auto& info = iter->second;
        // skip additional processing if nothing changes.
        if (info.member.ptr == ptr)
        {
            return *iter;
        }
        info.member.ptr = ptr;
        // if parsed config having exact full name exists, apply.
        // (Config using exact full name has highest matching priority.)
        if (info.member.ptr &&
            info.parsedLevel.valid)
        {
            info.member.ptr->level = info.parsedLevel.level;
            return *iter;
        }
        assignNameParts(*iter);
        return *iter;
    }
}

void LogTagManager::assignNameParts(const FullNamePair& fullNamePair)
{
    const auto nameParts = splitNameParts(fullNamePair.first);
    bool isFirstNamePart = true;
    for (const std::string& namePart : nameParts)
    {
        assignWildcardInfo(fullNamePair, namePart, isFirstNamePart);
        isFirstNamePart = false;
    }
}

void LogTagManager::assignWildcardInfo(const FullNamePair& fullNamePair, const std::string& namePart, bool isFirst)
{
    const auto memberIdAndPtr = fullNamePair.second.member;
    LogTag* const ptr = memberIdAndPtr.ptr;
    auto& wildcardMap = isFirst ? m_firstParts : m_anyParts;
    auto wildcardIter = wildcardMap.find(namePart);
    if (wildcardIter == wildcardMap.end())
    {
        wildcardIter = wildcardMap.emplace(namePart, WildcardInfo{}).first;
        auto& members = wildcardIter->second.members;
        members.emplace_back(memberIdAndPtr);
    }
    else
    {
        auto& members = wildcardIter->second.members;
        auto& foundMember = addOrGetMember(members, memberIdAndPtr);
        // skip additional processing if nothing changes.
        if (foundMember.ptr == ptr)
        {
            return;
        }
        foundMember.ptr = ptr;
        // if there is parsed config for wildcard but not the full name,
        // it is applied.
        // (If both exist, the full name config has higher priority)
        const bool hasParsedFullNameLevel = fullNamePair.second.parsedLevel.valid;
        const bool hasParsedWildcardLevel = wildcardIter->second.parsedLevel.valid;
        if (ptr &&
            !hasParsedFullNameLevel &&
            hasParsedWildcardLevel)
        {
            LogLevel wildcardLevel = wildcardIter->second.parsedLevel.level;
            ptr->level = wildcardLevel;
        }
    }
}

LogTagManager::LogTagAndId& LogTagManager::addOrGetMember(std::vector<LogTagAndId>& members, const LogTagAndId& memberArg)
{
    const size_t npos = ~(size_t)0u;
    const size_t memberCount = members.size();
    size_t memberIndex = npos;
    for (size_t k = 0u; k < memberCount; ++k)
    {
        auto& member = members.at(k);
        if (member.id == memberArg.id)
        {
            memberIndex = k;
            break;
        }
    }
    if (memberIndex == npos)
    {
        memberIndex = memberCount;
        members.emplace_back(memberArg);
    }
    return members.at(memberIndex);
}

std::vector<std::string> LogTagManager::splitNameParts(const std::string& fullName)
{
    const size_t npos = std::string::npos;
    const size_t len = fullName.length();
    std::vector<std::string> nameParts;
    size_t start = 0u;
    while (start < len)
    {
        size_t nextPeriod = fullName.find('.', start);
        if (nextPeriod == npos)
        {
            nextPeriod = len;
        }
        if (nextPeriod >= start + 1u)
        {
            nameParts.emplace_back(fullName.substr(start, nextPeriod - start));
        }
        start = nextPeriod + 1u;
    }
    return nameParts;
}

}}} //namespace
