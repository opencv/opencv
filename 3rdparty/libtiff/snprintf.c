/**
 * Workaround for lack of snprintf(3) in Visual Studio.  See
 * http://stackoverflow.com/questions/2915672/snprintf-and-visual-studio-2010/8712996#8712996
 * It's a trivial wrapper around the builtin _vsnprintf_s and
 * _vscprintf functions.
 */

#ifdef _MSC_VER

#include <stdio.h>
#include <stdarg.h>
#include "libport.h"

int _TIFF_vsnprintf_f(char* str, size_t size, const char* format, va_list ap)
{
  int count = -1;

  if (size != 0)
    count = _vsnprintf_s(str, size, _TRUNCATE, format, ap);
  if (count == -1)
    count = _vscprintf(format, ap);

  return count;
}

int _TIFF_snprintf_f(char* str, size_t size, const char* format, ...)
{
  int count;
  va_list ap;

  va_start(ap, format);
  count = vsnprintf(str, size, format, ap);
  va_end(ap);

  return count;
}

#endif // _MSC_VER
