// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "logtagmanager.hpp"
#include "logtagconfigparser.hpp"

namespace cv {
namespace utils {
namespace logging {


const char* LogTagManager::m_globalName = "global";


LogTagManager::LogTagManager(LogLevel defaultUnconfiguredGlobalLevel)
    : m_mutex()
    , m_globalLogTag(new LogTag(m_globalName, defaultUnconfiguredGlobalLevel))
    , m_config(std::make_shared<LogTagConfigParser>(defaultUnconfiguredGlobalLevel))
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

void LogTagManager::assign(const std::string& fullName, LogTag* ptr)
{
    CV_TRACE_FUNCTION();
    LockType lock(m_mutex);
    FullNameLookupResult result(fullName);
    result.m_findCrossReferences = true;
    m_nameTable.addOrLookupFullName(result);
    FullNameInfo& fullNameInfo = *result.m_fullNameInfoPtr;
    const bool isPtrChanged = (fullNameInfo.logTagPtr != ptr);
    if (!isPtrChanged)
    {
        return;
    }
    fullNameInfo.logTagPtr = ptr;
    if (!ptr)
    {
        return;
    }
    const bool hasAppliedFullNameConfig = internal_applyFullNameConfigToTag(fullNameInfo);
    if (hasAppliedFullNameConfig)
    {
        return;
    }
    internal_applyNamePartConfigToSpecificTag(result);
}

void LogTagManager::unassign(const std::string& fullName)
{
    // Lock is inside assign() method.
    assign(fullName, nullptr);
}

LogTag* LogTagManager::get(const std::string& fullName)
{
    CV_TRACE_FUNCTION();
    LockType lock(m_mutex);
    FullNameInfo* fullNameInfoPtr = m_nameTable.getFullNameInfo(fullName);
    if (fullNameInfoPtr && fullNameInfoPtr->logTagPtr)
    {
        return fullNameInfoPtr->logTagPtr;
    }
    return nullptr;
}

void LogTagManager::setLevelByFullName(const std::string& fullName, LogLevel level)
{
    CV_TRACE_FUNCTION();
    LockType lock(m_mutex);
    FullNameLookupResult result(fullName);
    result.m_findCrossReferences = false;
    m_nameTable.addOrLookupFullName(result);
    FullNameInfo& fullNameInfo = *result.m_fullNameInfoPtr;
    if (fullNameInfo.parsedLevel.scope == MatchingScope::Full &&
        fullNameInfo.parsedLevel.level == level)
    {
        // skip additional processing if nothing changes.
        return;
    }
    // update the cached configured value.
    fullNameInfo.parsedLevel.scope = MatchingScope::Full;
    fullNameInfo.parsedLevel.level = level;
    // update the actual tag, if already registered.
    LogTag* logTagPtr = fullNameInfo.logTagPtr;
    if (logTagPtr)
    {
        logTagPtr->level = level;
    }
}

void LogTagManager::setLevelByFirstPart(const std::string& firstPart, LogLevel level)
{
    // Lock is inside setLevelByNamePart() method.
    setLevelByNamePart(firstPart, level, MatchingScope::FirstNamePart);
}

void LogTagManager::setLevelByAnyPart(const std::string& anyPart, LogLevel level)
{
    // Lock is inside setLevelByNamePart() method.
    setLevelByNamePart(anyPart, level, MatchingScope::AnyNamePart);
}

void LogTagManager::setLevelByNamePart(const std::string& namePart, LogLevel level, MatchingScope scope)
{
    CV_TRACE_FUNCTION();
    LockType lock(m_mutex);
    NamePartLookupResult result(namePart);
    result.m_findCrossReferences = true;
    m_nameTable.addOrLookupNamePart(result);
    NamePartInfo& namePartInfo = *result.m_namePartInfoPtr;
    if (namePartInfo.parsedLevel.scope == scope &&
        namePartInfo.parsedLevel.level == level)
    {
        // skip additional processing if nothing changes.
        return;
    }
    namePartInfo.parsedLevel.scope = scope;
    namePartInfo.parsedLevel.level = level;
    internal_applyNamePartConfigToMatchingTags(result);
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

bool LogTagManager::internal_isNamePartMatch(MatchingScope scope, size_t matchingPos)
{
    switch (scope)
    {
    case MatchingScope::FirstNamePart:
        return (matchingPos == 0u);
    case MatchingScope::AnyNamePart:
        return true;
    case MatchingScope::None:
    case MatchingScope::Full:
    default:
        return false;
    }
}

bool LogTagManager::internal_applyFullNameConfigToTag(FullNameInfo& fullNameInfo)
{
    if (!fullNameInfo.logTagPtr)
    {
        return false;
    }
    if (fullNameInfo.parsedLevel.scope == MatchingScope::Full)
    {
        fullNameInfo.logTagPtr->level = fullNameInfo.parsedLevel.level;
        return true;
    }
    return false;
}

bool LogTagManager::internal_applyNamePartConfigToSpecificTag(FullNameLookupResult& fullNameResult)
{
    const FullNameInfo& fullNameInfo = *fullNameResult.m_fullNameInfoPtr;
    LogTag* const logTag = fullNameInfo.logTagPtr;
    if (!logTag)
    {
        return false;
    }
    CV_Assert(fullNameResult.m_findCrossReferences);
    const auto& crossReferences = fullNameResult.m_crossReferences;
    const size_t matchingNamePartCount = crossReferences.size();
    for (size_t k = 0u; k < matchingNamePartCount; ++k)
    {
        const auto& match = crossReferences.at(k);
        const auto& namePartInfo = *match.m_namePartInfo;
        const auto& parsedLevel = namePartInfo.parsedLevel;
        const auto scope = parsedLevel.scope;
        const LogLevel level = parsedLevel.level;
        const size_t matchingPos = match.m_matchingPos;
        const bool isMatch = internal_isNamePartMatch(scope, matchingPos);
        if (isMatch)
        {
            logTag->level = level;
            return true;
        }
    }
    return false;
}

void LogTagManager::internal_applyNamePartConfigToMatchingTags(NamePartLookupResult& namePartResult)
{
    CV_Assert(namePartResult.m_findCrossReferences);
    const auto& crossReferences = namePartResult.m_crossReferences;
    const size_t matchingFullNameCount = crossReferences.size();
    NamePartInfo& namePartInfo = *namePartResult.m_namePartInfoPtr;
    const MatchingScope scope = namePartInfo.parsedLevel.scope;
    CV_Assert(scope != MatchingScope::Full);
    if (scope == MatchingScope::None)
    {
        return;
    }
    const LogLevel level = namePartInfo.parsedLevel.level;
    for (size_t k = 0u; k < matchingFullNameCount; ++k)
    {
        const auto& match = crossReferences.at(k);
        FullNameInfo& fullNameInfo = *match.m_fullNameInfo;
        LogTag* logTagPtr = fullNameInfo.logTagPtr;
        if (!logTagPtr)
        {
            continue;
        }
        if (fullNameInfo.parsedLevel.scope == MatchingScope::Full)
        {
            // If the full name already has valid config, that full name config
            // has precedence over name part config.
            continue;
        }
        const size_t matchingPos = match.m_matchingPos;
        const bool isMatch = internal_isNamePartMatch(scope, matchingPos);
        if (!isMatch)
        {
            continue;
        }
        logTagPtr->level = level;
    }
}

void LogTagManager::NameTable::addOrLookupFullName(FullNameLookupResult& result)
{
    const auto fullNameIdAndFlag = internal_addOrLookupFullName(result.m_fullName);
    result.m_fullNameId = fullNameIdAndFlag.first;
    result.m_nameParts = LogTagManager::splitNameParts(result.m_fullName);
    internal_addOrLookupNameParts(result.m_nameParts, result.m_namePartIds);
    const bool isNew = fullNameIdAndFlag.second;
    if (isNew)
    {
        internal_addCrossReference(result.m_fullNameId, result.m_namePartIds);
    }
    // ====== IMPORTANT ====== Critical order-of-operation ======
    // The gathering of the pointers of FullNameInfo and NamePartInfo are performed
    // as the last step of the operation, so that these pointer are not invalidated
    // by the vector append operations earlier in this function.
    // ======
    result.m_fullNameInfoPtr = internal_getFullNameInfo(result.m_fullNameId);
    if (result.m_findCrossReferences)
    {
        internal_findMatchingNamePartsForFullName(result);
    }
}

void LogTagManager::NameTable::addOrLookupNamePart(NamePartLookupResult& result)
{
    result.m_namePartId = internal_addOrLookupNamePart(result.m_namePart);
    result.m_namePartInfoPtr = internal_getNamePartInfo(result.m_namePartId);
    if (result.m_findCrossReferences)
    {
        internal_findMatchingFullNamesForNamePart(result);
    }
}

std::pair<size_t, bool> LogTagManager::NameTable::internal_addOrLookupFullName(const std::string& fullName)
{
    const auto fullNameIdIter = m_fullNameIds.find(fullName);
    if (fullNameIdIter != m_fullNameIds.end())
    {
        return std::make_pair(fullNameIdIter->second, false);
    }
    const size_t fullNameId = m_fullNameInfos.size();
    m_fullNameInfos.emplace_back(FullNameInfo{});
    m_fullNameIds.emplace(fullName, fullNameId);
    return std::make_pair(fullNameId, true);
}

void LogTagManager::NameTable::internal_addOrLookupNameParts(const std::vector<std::string>& nameParts,
    std::vector<size_t>& namePartIds)
{
    const size_t namePartCount = nameParts.size();
    namePartIds.resize(namePartCount, ~(size_t)0u);
    for (size_t namePartIndex = 0u; namePartIndex < namePartCount; ++namePartIndex)
    {
        const std::string& namePart = nameParts.at(namePartIndex);
        const size_t namePartId = internal_addOrLookupNamePart(namePart);
        namePartIds.at(namePartIndex) = namePartId;
    }
}

size_t LogTagManager::NameTable::internal_addOrLookupNamePart(const std::string& namePart)
{
    const auto namePartIter = m_namePartIds.find(namePart);
    if (namePartIter != m_namePartIds.end())
    {
        return namePartIter->second;
    }
    const size_t namePartId = m_namePartInfos.size();
    m_namePartInfos.emplace_back(NamePartInfo{});
    m_namePartIds.emplace(namePart, namePartId);
    return namePartId;
}

void LogTagManager::NameTable::internal_addCrossReference(size_t fullNameId, const std::vector<size_t>& namePartIds)
{
    const size_t namePartCount = namePartIds.size();
    for (size_t namePartPos = 0u; namePartPos < namePartCount; ++namePartPos)
    {
        const size_t namePartId = namePartIds.at(namePartPos);
        m_fullNameToNamePartIds.emplace(fullNameId, std::make_pair(namePartId, namePartPos));
        m_namePartToFullNameIds.emplace(namePartId, std::make_pair(fullNameId, namePartPos));
    }
}

LogTagManager::FullNameInfo* LogTagManager::NameTable::getFullNameInfo(const std::string& fullName)
{
    const auto fullNameIdIter = m_fullNameIds.find(fullName);
    if (fullNameIdIter == m_fullNameIds.end())
    {
        return nullptr;
    }
    const size_t fullNameId = fullNameIdIter->second;
    return internal_getFullNameInfo(fullNameId);
}

LogTagManager::FullNameInfo* LogTagManager::NameTable::internal_getFullNameInfo(size_t fullNameId)
{
    return std::addressof(m_fullNameInfos.at(fullNameId));
}

LogTagManager::NamePartInfo* LogTagManager::NameTable::internal_getNamePartInfo(size_t namePartId)
{
    return std::addressof(m_namePartInfos.at(namePartId));
}

void LogTagManager::NameTable::internal_findMatchingNamePartsForFullName(FullNameLookupResult& fullNameResult)
{
    const size_t fullNameId = fullNameResult.m_fullNameId;
    FullNameInfo* fullNameInfo = fullNameResult.m_fullNameInfoPtr;
    const auto& namePartIds = fullNameResult.m_namePartIds;
    const size_t namePartCount = namePartIds.size();
    auto& crossReferences = fullNameResult.m_crossReferences;
    crossReferences.clear();
    crossReferences.reserve(namePartCount);
    for (size_t matchingPos = 0u; matchingPos < namePartCount; ++matchingPos)
    {
        const size_t namePartId = namePartIds.at(matchingPos);
        NamePartInfo* namePartInfo = internal_getNamePartInfo(namePartId);
        crossReferences.emplace_back(CrossReference(fullNameId, namePartId, matchingPos, fullNameInfo, namePartInfo));
    }
}

void LogTagManager::NameTable::internal_findMatchingFullNamesForNamePart(NamePartLookupResult& result)
{
    const size_t namePartId = result.m_namePartId;
    NamePartInfo* namePartInfo = result.m_namePartInfoPtr;
    const size_t matchingFullNameCount = m_namePartToFullNameIds.count(namePartId);
    std::vector<CrossReference>& crossReferences = result.m_crossReferences;
    crossReferences.clear();
    crossReferences.reserve(matchingFullNameCount);
    const auto namePartToFullNameIterPair = m_namePartToFullNameIds.equal_range(result.m_namePartId);
    const auto iterBegin = namePartToFullNameIterPair.first;
    const auto iterEnd = namePartToFullNameIterPair.second;
    for (auto iter = iterBegin; iter != iterEnd; ++iter)
    {
        const size_t fullNameId = iter->second.first;
        const size_t matchingPos = iter->second.second;
        FullNameInfo* fullNameInfo = internal_getFullNameInfo(fullNameId);
        crossReferences.emplace_back(CrossReference(fullNameId, namePartId, matchingPos, fullNameInfo, namePartInfo));
    }
}

}}} //namespace
