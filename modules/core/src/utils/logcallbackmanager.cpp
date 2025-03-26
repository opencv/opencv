// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "logcallbackmanager.hpp"

namespace cv {
namespace utils {
namespace logging {

LogCallbackManager::LogCallbackManager()
    : m_mutex()
    , m_handlers()
#ifdef ALLOW_DEPRECATED_CODE
    , m_callbacks()
#endif
{
}

LogCallbackManager::~LogCallbackManager()
{
}

size_t LogCallbackManager::count() const
{
    const LockType lock(m_mutex);
    size_t total = 0u;
    total += m_handlers.size();

#ifdef ALLOW_DEPRECATED_CODE
    total += m_callbacks.size();
#endif

    return total;
}

bool LogCallbackManager::contains(const cv::Ptr<LoggingCallbackHandler>& handler) const
{
    const LockType lock(m_mutex);
    const LoggingCallbackHandler* const handler_ptr = handler.get();
    return m_handlers.find(handler_ptr) != m_handlers.end();
}

void LogCallbackManager::add(cv::Ptr<LoggingCallbackHandler> handler)
{
    const LoggingCallbackHandler* const handler_ptr = handler.get();
    if (!handler_ptr)
    {
        return;
    }
    const LockType lock(m_mutex);
    auto iter = m_handlers.find(handler_ptr);
    if (iter != m_handlers.end())
    {
        return;
    }
    m_handlers.emplace(std::make_pair(handler_ptr, std::move(handler)));
}

void LogCallbackManager::remove(const cv::Ptr<LoggingCallbackHandler>& handler)
{
    const LoggingCallbackHandler* const handler_ptr = handler.get();
    if (!handler_ptr)
    {
        return;
    }
    const LockType lock(m_mutex);
    auto iter = m_handlers.find(handler_ptr);
    if (iter == m_handlers.end())
    {
        return;
    }
    m_handlers.erase(iter);
}

void LogCallbackManager::readInto(std::vector<cv::Ptr<LoggingCallbackHandler>>& handlers) const
{
    const LockType lock(m_mutex);
    size_t count = m_handlers.size();
    handlers.clear();
    handlers.reserve(count);
    for (const auto& handler : m_handlers)
    {
        handlers.push_back(handler.second);
    }
}

#ifdef ALLOW_DEPRECATED_CODE // deprecated(LoggingCallbackPtrType)
bool LogCallbackManager::contains(LoggingCallbackPtrType callback) const
{
    const LockType lock(m_mutex);
    return m_callbacks.find(callback) != m_callbacks.end();
}

void LogCallbackManager::add(LoggingCallbackPtrType callback)
{
    if (!callback)
    {
        return;
    }
    const LockType lock(m_mutex);
    m_callbacks.insert(callback);
}

void LogCallbackManager::remove(LoggingCallbackPtrType callback)
{
    if (!callback)
    {
        return;
    }
    const LockType lock(m_mutex);
    m_callbacks.erase(callback);
}

void LogCallbackManager::readInto(std::vector<LoggingCallbackPtrType>& callbacks) const
{
    const LockType lock(m_mutex);
    callbacks.assign(m_callbacks.cbegin(), m_callbacks.cend());
}
#endif //ALLOW_DEPRECATED_CODE // deprecated(LoggingCallbackPtrType)


void LogCallbackManager::removeAll()
{
    const LockType lock(m_mutex);
    m_handlers.clear();

#ifdef ALLOW_DEPRECATED_CODE
    m_callbacks.clear();
#endif //ALLOW_DEPRECATED_CODE

}


#ifdef ALLOW_CONTROVERSIAL_CODE
size_t LogCallbackManager::tryReadIntoSz(LoggingCallbackPtrType (&callbacks)[], size_t arr_sz) const
{
    const LockType lock(m_mutex);
    const auto it_end = m_callbacks.cend();
    auto iter = m_callbacks.cbegin();
    for (size_t idx = 0u; idx < arr_sz; ++idx)
    {
        if (iter == it_end)
        {
            callbacks[idx] = nullptr;
        }
        else
        {
            callbacks[idx] = *iter;
            ++iter;
        }
    }
    return m_callbacks.size();
}
#endif //ALLOW_CONTROVERSIAL_CODE

}}} //namespace
