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

void LogTagConfigParser::forEachParsed(std::function<void(const std::string&, LogLevel)> func) const
{
    for (const auto& parsed : m_parsed)
    {
        func(parsed.name, parsed.level);
    }
}

void LogTagConfigParser::forEachMalformed(std::function<void(const std::string&)> func) const
{
    for (const auto& malformed : m_malformed)
    {
        func(malformed);
    }
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
		parseNameLevel(s);
	}
}

void LogTagConfigParser::parseNameLevel(const std::string& s)
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
			// If it is, assume it is a catch-all
			m_parsed.emplace_back(NameLevel{ "*", parsedLevel.first });
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
		m_parsed.emplace_back(NameLevel{ s.substr(0u, colonIdx), parsedLevel.first });
	}
	else
	{
		// not sure what to do.
		m_malformed.push_back(s);
		return;
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
