#include "precomp.hpp"
#include <cstdio>
#include <cstdarg>
#include "opencv2/videostab/log.hpp"

using namespace std;

namespace cv
{
namespace videostab
{

void LogToStdout::print(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    fflush(stdout);
    va_end(args);
}

} // namespace videostab
} // namespace cv
