#include "test_precomp.hpp"

cv::String cv::Path::combine(const String& item1, const String& item2)
{
    if (item1.empty())
        return item2;

    if (item2.empty())
        return item1;

    char last = item1[item1.size()-1];

    bool need_append = last != '/' && last != '\\';
    return item1 + (need_append ? "/" : "") + item2;
}

cv::String cv::Path::combine(const String& item1, const String& item2, const String& item3)
{ return combine(combine(item1, item2), item3); }

cv::String cv::Path::change_extension(const String& file, const String& ext)
{
    String::size_type pos = file.find_last_of('.');
    return pos == String::npos ? file : file.substr(0, pos+1) + ext;
}
