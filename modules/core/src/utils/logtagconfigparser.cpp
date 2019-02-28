// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "logtagconfigparser.hpp"

namespace cv {
namespace utils {
namespace logging {

LogTagConfigParser::LogTagConfigParser()
{
    m_parsedGlobal.namePart = "global";
    m_parsedGlobal.isGlobal = true;
    m_parsedGlobal.hasPrefixWildcard = false;
    m_parsedGlobal.hasSuffixWildcard = false;
    m_parsedGlobal.level = LOG_LEVEL_VERBOSE;
}

LogTagConfigParser::LogTagConfigParser(const std::string& input)
{
    parse(input);
}

LogTagConfigParser::~LogTagConfigParser()
{
}

bool LogTagConfigParser::parse(const std::string& input)
{
    m_input = input;
    segmentTokens();
    return (m_malformed.empty());
}

bool LogTagConfigParser::hasMalformed() const
{
    return !m_malformed.empty();
}

const LogTagConfig& LogTagConfigParser::getGlobalConfig() const
{
    return m_parsedGlobal;
}

const std::vector<LogTagConfig>& LogTagConfigParser::getFullNameConfigs() const
{
    return m_parsedFullName;
}

const std::vector<LogTagConfig>& LogTagConfigParser::getFirstPartConfigs() const
{
    return m_parsedFirstPart;
}

const std::vector<LogTagConfig>& LogTagConfigParser::getAnyPartConfigs() const
{
    return m_parsedAnyPart;
}

const std::vector<std::string>& LogTagConfigParser::getMalformed() const
{
    return m_malformed;
}

void LogTagConfigParser::segmentTokens()
{
    const size_t len = m_input.length();
    std::vector<std::pair<size_t, size_t>> startStops;
    bool wasSeparator = true;
    for (size_t pos = 0u; pos < len; ++pos)
    {
        char c = m_input[pos];
        bool isSeparator = (c == ' ' || c == '\t' || c == ';');
        if (!isSeparator)
        {
            if (wasSeparator)
            {
                startStops.emplace_back(pos, pos + 1u);
            }
            else
            {
                startStops.back().second = pos + 1u;
            }
        }
        wasSeparator = isSeparator;
    }
    for (const auto& startStop : startStops)
    {
        const auto s = m_input.substr(startStop.first, startStop.second - startStop.first);
        parseNameAndLevel(s);
    }
}

void LogTagConfigParser::parseNameAndLevel(const std::string& s)
{
    const size_t npos = std::string::npos;
    const size_t len = s.length();
    size_t colonIdx = s.find_first_of(":=");
    if (colonIdx == npos)
    {
        // See if the whole string is a log level
        auto parsedLevel = parseLogLevel(s);
        if (parsedLevel.second)
        {
            // If it is, assume the log level is for global
            parseWildcard("", parsedLevel.first);
            return;
        }
        else
        {
            // not sure what to do.
            m_malformed.push_back(s);
            return;
        }
    }
    if (colonIdx == 0u || colonIdx + 1u == len)
    {
        // malformed (colon or equal sign at beginning or end), cannot do anything
        m_malformed.push_back(s);
        return;
    }
    size_t colonIdx2 = s.find_first_of(":=", colonIdx + 1u);
    if (colonIdx2 != npos)
    {
        // malformed (more than one colon or equal sign), cannot do anything
        m_malformed.push_back(s);
        return;
    }
    auto parsedLevel = parseLogLevel(s.substr(colonIdx + 1u));
    if (parsedLevel.second)
    {
        parseWildcard(s.substr(0u, colonIdx), parsedLevel.first);
        return;
    }
    else
    {
        // Cannot recognize the right side of the colon or equal sign.
        // Not sure what to do.
        m_malformed.push_back(s);
        return;
    }
}

void LogTagConfigParser::parseWildcard(const std::string& name, LogLevel level)
{
    constexpr size_t npos = std::string::npos;
    const size_t len = name.length();
    if (len == 0u)
    {
        m_parsedGlobal.level = level;
        return;
    }
    const bool hasPrefixWildcard = (name[0u] == '*');
    if (hasPrefixWildcard && len == 1u)
    {
        m_parsedGlobal.level = level;
        return;
    }
    const size_t firstNonWildcard = name.find_first_not_of("*.");
    if (hasPrefixWildcard && firstNonWildcard == npos)
    {
        m_parsedGlobal.level = level;
        return;
    }
    const bool hasSuffixWildcard = (name[len - 1u] == '*');
    const size_t lastNonWildcard = name.find_last_not_of("*.");
    std::string trimmedNamePart = name.substr(firstNonWildcard, lastNonWildcard - firstNonWildcard + 1u);
    // The case of a single asterisk has been handled above;
    // here we only handle the explicit use of "global" in the log config string.
    const bool isGlobal = (trimmedNamePart == "global");
    if (isGlobal)
    {
        m_parsedGlobal.level = level;
        return;
    }
    LogTagConfig result(trimmedNamePart, level, false, hasPrefixWildcard, hasSuffixWildcard);
    if (hasPrefixWildcard)
    {
        m_parsedAnyPart.emplace_back(std::move(result));
    }
    else if (hasSuffixWildcard)
    {
        m_parsedFirstPart.emplace_back(std::move(result));
    }
    else
    {
        m_parsedFullName.emplace_back(std::move(result));
    }
}

std::pair<LogLevel, bool> LogTagConfigParser::parseLogLevel(const std::string& s)
{
    const auto falseDontCare = std::make_pair(LOG_LEVEL_VERBOSE, false);
    const size_t len = s.length();
    if (len >= 1u)
    {
        std::pair<LogLevel, bool> result = falseDontCare;
        // Fast classification based on first character.
        // Tentative; need check remaining string.
        char c = (char)std::toupper(s[0]);
        switch (c)
        {
        case 'S':
            result.first = LOG_LEVEL_SILENT;
            result.second = (len == 1u || len == 6u);
            break;
        case 'F':
            result.first = LOG_LEVEL_FATAL;
            result.second = (len == 1u || len == 5u);
            break;
        case 'E':
            result.first = LOG_LEVEL_ERROR;
            result.second = (len == 1u || len == 5u);
            break;
        case 'W':
            result.first = LOG_LEVEL_WARNING;
            result.second = (len == 1u || len == 4u || len == 7u);
            break;
        case 'I':
            result.first = LOG_LEVEL_INFO;
            result.second = (len == 1u || len == 4u);
            break;
        case 'D':
            result.first = LOG_LEVEL_DEBUG;
            result.second = (len == 1u || len == 5u);
            break;
        case 'V':
            result.first = LOG_LEVEL_VERBOSE;
            result.second = (len == 1u || len == 7u);
            break;
        default:
            break;
        }
        if (len == 1u)
        {
            return result;
        }
        if (!result.second)
        {
            return falseDontCare;
        }
        std::string upper = s;
        std::transform(upper.begin(), upper.end(), upper.begin(),
            [](char arg) -> char { return (char)toupper(arg); });
        if (upper == "SILENT" || upper == "FATAL" || upper == "ERROR" ||
            upper == "WARNING" || upper == "WARN" || upper == "INFO" ||
            upper == "DEBUG" || upper == "VERBOSE")
        {
            return result;
        }
    }
    return falseDontCare;
}

std::string LogTagConfigParser::toString(LogLevel level)
{
    switch (level)
    {
    case LOG_LEVEL_SILENT:
        return "SILENT";
    case LOG_LEVEL_FATAL:
        return "FATAL";
    case LOG_LEVEL_ERROR:
        return "ERROR";
    case LOG_LEVEL_WARNING:
        return "WARNING";
    case LOG_LEVEL_INFO:
        return "INFO";
    case LOG_LEVEL_DEBUG:
        return "DEBUG";
    case LOG_LEVEL_VERBOSE:
        return "VERBOSE";
    default:
        return std::to_string((int)level);
    }
}

}}} //namespace
