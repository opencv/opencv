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
    InfoIndex index;
    populateIndex(index, name);
    const bool isPtrChanged = updateIndexLogTagPtr(index, ptr);
    if (ptr && isPtrChanged)
    {
        findAndApplyLevelToTag(index);
    }
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
    LockType lock(m_mutex);
    InfoIndex index;
    populateIndex(index, fullName);
    auto iter = m_fullNames.find(fullName);
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
    // Lock is inside setLevelByWildcard() method.
    const bool isFirst = true;
    setLevelByWildcard(firstPart, level, isFirst);
}

void LogTagManager::setLevelByAnyPart(const std::string& anyPart, LogLevel level)
{
    // Lock is inside setLevelByWildcard() method.
    const bool isFirst = false;
    setLevelByWildcard(anyPart, level, isFirst);
}

void LogTagManager::populateIndex(InfoIndex& index, const std::string& fullName)
{
    index.fullNamePtr = fullName;
    index.namePartsPtr = splitNameParts(fullName);
    const size_t namePartCount = index.namePartsPtr.size();
    index.id = 0u;
    index.anyPartInfoPtrs.resize(namePartCount, nullptr);
    index.anyPartMemberIndex.resize(namePartCount, 0u);
    populateIndexFullName(index);
    populateIndexFirstPart(index);
    for (size_t namePartIndex = 0u; namePartIndex < namePartCount; ++namePartIndex)
    {
        populateIndexAnyPart(index, namePartIndex);
    }
}

void LogTagManager::populateIndexFullName(InfoIndex& index)
{
    const std::string& fullName = index.fullNamePtr;
    auto iter = m_fullNames.find(fullName);
    if (iter == m_fullNames.end())
    {
        // full name never seen before, neither from tag registration nor from parsed config
        const size_t newId = (m_nextId++);
        iter = m_fullNames.emplace(fullName, FullNameInfo{}).first;
        iter->second.member.id = newId;
        index.id = newId;
    }
    else
    {
        index.id = iter->second.member.id;
    }
    index.fullNameInfoPtr = std::addressof(iter->second);
}

void LogTagManager::populateIndexFirstPart(InfoIndex& index)
{
    const std::vector<std::string>& nameParts = index.namePartsPtr;
    if (nameParts.empty())
    {
        return;
    }
    const std::string& firstNamePart = nameParts[0u];
    auto firstPartIter = m_firstParts.find(firstNamePart);
    if (firstPartIter == m_firstParts.end())
    {
        firstPartIter = m_firstParts.emplace(firstNamePart, WildcardInfo{}).first;
    }
    auto* infoPtr = std::addressof(firstPartIter->second);
    index.firstPartInfoPtr = infoPtr;
    index.firstPartMemberIndex = addOrGetMember(index, infoPtr);
}

void LogTagManager::populateIndexAnyPart(InfoIndex& index, size_t namePartIndex)
{
    const std::vector<std::string>& nameParts = index.namePartsPtr;
    if (namePartIndex >= nameParts.size() ||
        namePartIndex >= index.anyPartInfoPtrs.size() ||
        namePartIndex >= index.anyPartMemberIndex.size())
    {
        return;
    }
    const std::string& namePart = nameParts[namePartIndex];
    auto anyPartIter = m_anyParts.find(namePart);
    if (anyPartIter == m_anyParts.end())
    {
        anyPartIter = m_anyParts.emplace(namePart, WildcardInfo{}).first;
    }
    auto* infoPtr = std::addressof(anyPartIter->second);
    index.anyPartInfoPtrs[namePartIndex] = infoPtr;
    index.anyPartMemberIndex[namePartIndex] = addOrGetMember(index, infoPtr);
}

size_t LogTagManager::addOrGetMember(InfoIndex& index, WildcardInfo* wildcardInfoPtr)
{
    std::vector<LogTagAndId>& members = wildcardInfoPtr->members;
    constexpr const size_t npos = ~(size_t)0u;
    const size_t id = index.id;
    const size_t memberCount = members.size();
    size_t memberIndex = npos;
    for (size_t k = 0u; k < memberCount; ++k)
    {
        auto& member = members.at(k);
        if (member.id == id)
        {
            memberIndex = k;
            break;
        }
    }
    if (memberIndex == npos)
    {
        memberIndex = memberCount;
        members.emplace_back(LogTagAndId{});
        members.back().id = id;
    }
    return memberIndex;
}

bool LogTagManager::updateIndexLogTagPtr(InfoIndex& index, LogTag* ptr)
{
    const size_t namePartCount = index.namePartsPtr.size();
    if (index.fullNameInfoPtr->member.ptr == ptr)
    {
        return false;
    }
    index.fullNameInfoPtr->member.ptr = ptr;
    index.firstPartInfoPtr->members[index.firstPartMemberIndex].ptr = ptr;
    for (size_t namePartIndex = 0u; namePartIndex < namePartCount; ++namePartIndex)
    {
        size_t anyPartMemberIndex = index.anyPartMemberIndex[namePartIndex];
        auto* anyPartInfoPtr = index.anyPartInfoPtrs[namePartIndex];
        anyPartInfoPtr->members[anyPartMemberIndex].ptr = ptr;
    }
    return true;
}

bool LogTagManager::findAndApplyLevelToTag(InfoIndex& index)
{
    const size_t namePartCount = index.namePartsPtr.size();
    FullNameInfo& fullNameInfo = *(index.fullNameInfoPtr);
    LogTag* ptr = fullNameInfo.member.ptr;
    if (!ptr)
    {
        return false;
    }
    if (fullNameInfo.parsedLevel.valid)
    {
        ptr->level = fullNameInfo.parsedLevel.level;
        return true;
    }
    // ======
    // Based on tentative detail in new implementation, anyPart config has
    // higher precedence than firstPart config.
    // ======
    for (size_t namePartIndex = 0u; namePartIndex < namePartCount; ++namePartIndex)
    {
        WildcardInfo& anyPartInfo = *(index.anyPartInfoPtrs[namePartIndex]);
        if (anyPartInfo.parsedLevel.valid)
        {
            ptr->level = anyPartInfo.parsedLevel.level;
            return true;
        }
    }
    WildcardInfo& firstPartInfo = *(index.firstPartInfoPtr);
    if (firstPartInfo.parsedLevel.valid)
    {
        ptr->level = firstPartInfo.parsedLevel.level;
        return true;
    }
    return false;
}

void LogTagManager::setLevelByWildcard(const std::string& namePart, LogLevel level, bool isFirst)
{
    LockType lock(m_mutex);
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
