#include "StringUtils.h"

using namespace std;

bool StripString(string& src)
{
    size_t pos = 0;

    if (src.empty())
    {
        return false;
    }

    while ((pos < src.length()) && (' ' == src[pos])) pos++;
    src.erase(0, pos);

    pos = 0;
    while ((pos < src.length()) && ('\t' == src[pos])) pos++;
    src.erase(0, pos);

    pos = src.length() - 1;
    while (pos && (' ' == src[pos])) pos--;
    src.erase(pos+1);

    pos = src.length() - 1;
    while (pos && ('\t' == src[pos])) pos--;
    src.erase(pos+1);

    return true;
}

bool ParseString(const string& src, string& key, string& value)
{
    if (src.empty())
        return false;

    // find separator ":"
    size_t separator_pos = src.find(":");
    if (string::npos != separator_pos)
    {
        key = src.substr(0, separator_pos);
        StripString(key);
        value = src.substr(separator_pos+1);
        StripString(value);
        return true;
    }
    else
    {
        return false;
    }
}

set<string> SplitString(const string& src, const char separator)
{
    set<string> result;

    if (!src.empty())
    {
        size_t separator_pos;
        size_t prev_pos = 0;
        do
        {
            separator_pos = src.find(separator, prev_pos);
            result.insert(src.substr(prev_pos, separator_pos - prev_pos));
            prev_pos = separator_pos + 1;
        }
        while (string::npos != separator_pos);
    }

    return result;
}

vector<string> SplitStringVector(const string& src, const char separator)
{
    vector<string> result;

    if (!src.empty())
    {
        size_t separator_pos;
        size_t prev_pos = 0;
        do
        {
            separator_pos = src.find(separator, prev_pos);
            string tmp = src.substr(prev_pos, separator_pos - prev_pos);
            result.push_back(tmp);
            prev_pos = separator_pos + 1;
        }
        while (string::npos != separator_pos);
    }

    return result;
}
