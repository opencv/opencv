// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _CODERS_UTILS_H_
#define _CODERS_UTILS_H_

#include <string>
#include <vector>
#include <sstream>
#include <array>
#include <algorithm>
#include <cstdint>


namespace cv {

std::vector<std::string> split(const std::string &s, char delimiter);

inline bool startsWith(const std::string &s1, const std::string &s2)
{
    return s1.compare(0, s2.length(), s2) == 0;
}

inline std::string trimSpaces(const std::string &input)
{
    size_t start = 0;
    while (start < input.size() && input[start] == ' ')
    {
        start++;
    }
    size_t end = input.size();
    while (end > start && (input[end - 1] == ' ' || input[end - 1] == '\n' || input[end - 1] == '\r'))
    {
        end--;
    }
    return input.substr(start, end - start);
}

inline std::string getExtension(const std::string& filename)
{
    auto pos = filename.find_last_of('.');
    if (pos == std::string::npos)
    {
        return "";
    }
    return filename.substr( pos + 1);
}

template <typename T>
void swapEndian(T &val)
{
    union U
    {
        T val;
        std::array<std::uint8_t, sizeof(T)> raw;
    } src, dst;

    src.val = val;
    std::reverse_copy(src.raw.begin(), src.raw.end(), dst.raw.begin());
    val = dst.val;
}

} /* namespace cv */

#endif
