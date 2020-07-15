// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/core/bindings_utils.hpp"
#include <sstream>

namespace cv { namespace utils {

String dumpInputArray(InputArray argument)
{
    if (&argument == &noArray())
        return "InputArray: noArray()";
    std::ostringstream ss;
    ss << "InputArray:";
    try {
        do {
            ss << (argument.empty() ? " empty()=true" : " empty()=false");
            ss << cv::format(" kind=0x%08llx", (long long int)argument.kind());
            ss << cv::format(" flags=0x%08llx", (long long int)argument.getFlags());
            if (argument.getObj() == NULL)
            {
                ss << " obj=NULL";
                break; // done
            }
            ss << cv::format(" total(-1)=%lld", (long long int)argument.total(-1));
            int dims = argument.dims(-1);
            ss << cv::format(" dims(-1)=%d", dims);
            if (dims <= 2)
            {
                Size size = argument.size(-1);
                ss << cv::format(" size(-1)=%dx%d", size.width, size.height);
            }
            else
            {
                int sz[CV_MAX_DIM] = {0};
                argument.sizend(sz, -1);
                ss << " size(-1)=[";
                for (int i = 0; i < dims; i++)
                {
                    if (i > 0)
                        ss << ' ';
                    ss << sz[i];
                }
                ss << "]";
            }
            ss << " type(-1)=" << cv::typeToString(argument.type(-1));
        } while (0);
    }
    catch (...)
    {
        ss << " ERROR: exception occurred, dump is non-complete";  // need to properly support different kinds
    }
    return ss.str();
}

CV_EXPORTS_W String dumpInputArrayOfArrays(InputArrayOfArrays argument)
{
    if (&argument == &noArray())
        return "InputArrayOfArrays: noArray()";
    std::ostringstream ss;
    ss << "InputArrayOfArrays:";
    try {
        do {
            ss << (argument.empty() ? " empty()=true" : " empty()=false");
            ss << cv::format(" kind=0x%08llx", (long long int)argument.kind());
            ss << cv::format(" flags=0x%08llx", (long long int)argument.getFlags());
            if (argument.getObj() == NULL)
            {
                ss << " obj=NULL";
                break; // done
            }
            ss << cv::format(" total(-1)=%lld", (long long int)argument.total(-1));
            ss << cv::format(" dims(-1)=%d", argument.dims(-1));
            Size size = argument.size(-1);
            ss << cv::format(" size(-1)=%dx%d", size.width, size.height);
            if (argument.total(-1) > 0)
            {
                ss << " type(0)=" << cv::typeToString(argument.type(0));
                int dims = argument.dims(0);
                ss << cv::format(" dims(0)=%d", dims);
                if (dims <= 2)
                {
                    Size size0 = argument.size(0);
                    ss << cv::format(" size(0)=%dx%d", size0.width, size0.height);
                }
                else
                {
                    int sz[CV_MAX_DIM] = {0};
                    argument.sizend(sz, 0);
                    ss << " size(0)=[";
                    for (int i = 0; i < dims; i++)
                    {
                        if (i > 0)
                            ss << ' ';
                        ss << sz[i];
                    }
                    ss << "]";
                }
            }
        } while (0);
    }
    catch (...)
    {
        ss << " ERROR: exception occurred, dump is non-complete";  // need to properly support different kinds
    }
    return ss.str();
}

CV_EXPORTS_W String dumpInputOutputArray(InputOutputArray argument)
{
    if (&argument == &noArray())
        return "InputOutputArray: noArray()";
    std::ostringstream ss;
    ss << "InputOutputArray:";
    try {
        do {
            ss << (argument.empty() ? " empty()=true" : " empty()=false");
            ss << cv::format(" kind=0x%08llx", (long long int)argument.kind());
            ss << cv::format(" flags=0x%08llx", (long long int)argument.getFlags());
            if (argument.getObj() == NULL)
            {
                ss << " obj=NULL";
                break; // done
            }
            ss << cv::format(" total(-1)=%lld", (long long int)argument.total(-1));
            int dims = argument.dims(-1);
            ss << cv::format(" dims(-1)=%d", dims);
            if (dims <= 2)
            {
                Size size = argument.size(-1);
                ss << cv::format(" size(-1)=%dx%d", size.width, size.height);
            }
            else
            {
                int sz[CV_MAX_DIM] = {0};
                argument.sizend(sz, -1);
                ss << " size(-1)=[";
                for (int i = 0; i < dims; i++)
                {
                    if (i > 0)
                        ss << ' ';
                    ss << sz[i];
                }
                ss << "]";
            }
            ss << " type(-1)=" << cv::typeToString(argument.type(-1));
        } while (0);
    }
    catch (...)
    {
        ss << " ERROR: exception occurred, dump is non-complete";  // need to properly support different kinds
    }
    return ss.str();
}

CV_EXPORTS_W String dumpInputOutputArrayOfArrays(InputOutputArrayOfArrays argument)
{
    if (&argument == &noArray())
        return "InputOutputArrayOfArrays: noArray()";
    std::ostringstream ss;
    ss << "InputOutputArrayOfArrays:";
    try {
        do {
            ss << (argument.empty() ? " empty()=true" : " empty()=false");
            ss << cv::format(" kind=0x%08llx", (long long int)argument.kind());
            ss << cv::format(" flags=0x%08llx", (long long int)argument.getFlags());
            if (argument.getObj() == NULL)
            {
                ss << " obj=NULL";
                break; // done
            }
            ss << cv::format(" total(-1)=%lld", (long long int)argument.total(-1));
            ss << cv::format(" dims(-1)=%d", argument.dims(-1));
            Size size = argument.size(-1);
            ss << cv::format(" size(-1)=%dx%d", size.width, size.height);
            if (argument.total(-1) > 0)
            {
                ss << " type(0)=" << cv::typeToString(argument.type(0));
                int dims = argument.dims(0);
                ss << cv::format(" dims(0)=%d", dims);
                if (dims <= 2)
                {
                    Size size0 = argument.size(0);
                    ss << cv::format(" size(0)=%dx%d", size0.width, size0.height);
                }
                else
                {
                    int sz[CV_MAX_DIM] = {0};
                    argument.sizend(sz, 0);
                    ss << " size(0)=[";
                    for (int i = 0; i < dims; i++)
                    {
                        if (i > 0)
                            ss << ' ';
                        ss << sz[i];
                    }
                    ss << "]";
                }
            }
        } while (0);
    }
    catch (...)
    {
        ss << " ERROR: exception occurred, dump is non-complete";  // need to properly support different kinds
    }
    return ss.str();
}

}} // namespace
