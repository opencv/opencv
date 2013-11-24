#ifndef __STRING_UTILS_H__
#define __STRING_UTILS_H__

#include <string>
#include <set>
#include <vector>

bool StripString(std::string& src);
std::set<std::string> SplitString(const std::string& src, const char seporator);
bool ParseString(const std::string& src, std::string& key, std::string& value);
std::vector<std::string> SplitStringVector(const std::string& src, const char seporator);

#endif
