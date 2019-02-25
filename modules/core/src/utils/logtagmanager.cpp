// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "logtagmanager.hpp"

namespace cv {
namespace utils {
namespace logging {

LogTagManager::LogTagManager()
    : m_mutex()
    , m_byFullName()
    , m_byFirstPart()
    , m_byAnyPart()
    , m_nextId()
{
}

LogTagManager::~LogTagManager()
{
}

void LogTagManager::assign(const std::string& name, LogTag* value)
{
    LockType lock(m_mutex);
    auto iterByFullName = m_byFullName.find(name);
    size_t nameId = 0u;
    if (iterByFullName == m_byFullName.end())
    {
        nameId = (++m_nextId);
        iterByFullName = m_byFullName.emplace(name, LogTagAndId{ nameId, value }).first;
    }
    else
    {
        nameId = iterByFullName->second.id;
        iterByFullName->second.value = value;
    }
    bool isFirst = true;
    auto namePartFunc = [&](const std::string& part)
    {
        if (isFirst)
        {
            // Warning, current code is not sufficient in preventing duplicated
            // entries in unordered_multimap.
            m_byFirstPart.emplace(part, LogTagAndId{ nameId, value });
            isFirst = false;
        }
        // See same warning above.
        m_byAnyPart.emplace(part, LogTagAndId{ nameId, value });
    };
    internal_forEachNamePart(name, namePartFunc);
}

void LogTagManager::unassign(const std::string& name)
{
    // Lock is inside assign() method.
    assign(name, nullptr);
}

LogTag* LogTagManager::get(const std::string& name) const
{
    LockType lock(m_mutex);
    const auto iter = m_byFullName.find(name);
    if (iter == m_byFullName.end())
    {
        return nullptr;
    }
    return iter->second.value;
}

void LogTagManager::invoke(const std::string& name, std::function<void(LogTag*)> func)
{
    LockType lock(m_mutex);
    LogTag* value = get(name);
    if (value)
    {
        func(value);
    }
}

void LogTagManager::forEach_byFirstPart(const std::string& firstPart, std::function<void(LogTag*)> func)
{
    LockType lock(m_mutex);
    const auto iterPair = m_byFirstPart.equal_range(firstPart);
    for (auto iter = iterPair.first; iter != iterPair.second; ++iter)
    {
        if (iter->first == firstPart) // is this check redundant?
        {
            auto value = iter->second.value;
            if (value)
            {
                func(value);
            }
        }
    }
}

void LogTagManager::forEach_byAnyPart(const std::string& anyPart, std::function<void(LogTag*)> func)
{
    LockType lock(m_mutex);
    const auto iterPair = m_byAnyPart.equal_range(anyPart);
    for (auto iter = iterPair.first; iter != iterPair.second; ++iter)
    {
        if (iter->first == anyPart) // is this check redundant?
        {
            auto value = iter->second.value;
            if (value)
            {
                func(value);
            }
        }
    }
}

void LogTagManager::internal_forEachNamePart(const std::string& name, std::function<void(const std::string& namePart)> namePartFunc)
{
    const size_t npos = std::string::npos;
    const size_t len = name.length();
    size_t start = 0u;
    while (start < len)
    {
        size_t nextPeriod = name.find('.', start);
        if (nextPeriod == npos)
        {
            nextPeriod = len;
        }
        if (nextPeriod >= start + 1u)
        {
            namePartFunc(name.substr(start, nextPeriod - start));
        }
        start = nextPeriod + 1u;
    }
}

}}} //namespace
