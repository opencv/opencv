// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "base64.hpp"

namespace cv { namespace vlm {

std::string base64Encode(const unsigned char* data, size_t size)
{
    static const char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string out;
    out.reserve(((size + 2) / 3) * 4);

    size_t i = 0;
    while (i + 3 <= size)
    {
        unsigned int chunk = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
        out += table[(chunk >> 18) & 0x3F];
        out += table[(chunk >> 12) & 0x3F];
        out += table[(chunk >> 6) & 0x3F];
        out += table[chunk & 0x3F];
        i += 3;
    }

    size_t remaining = size - i;
    if (remaining == 1)
    {
        unsigned int chunk = data[i] << 16;
        out += table[(chunk >> 18) & 0x3F];
        out += table[(chunk >> 12) & 0x3F];
        out += "==";
    }
    else if (remaining == 2)
    {
        unsigned int chunk = (data[i] << 16) | (data[i + 1] << 8);
        out += table[(chunk >> 18) & 0x3F];
        out += table[(chunk >> 12) & 0x3F];
        out += table[(chunk >> 6) & 0x3F];
        out += "=";
    }

    return out;
}

}} // namespace cv::vlm
