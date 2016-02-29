#ifndef _HAL_INTERFACE_HPP_INCLUDED_
#define _HAL_INTERFACE_HPP_INCLUDED_

//! @addtogroup core_hal_interface
//! @{

#define CV_HAL_ERROR_OK 0
#define CV_HAL_ERROR_NOT_IMPLEMENTED 1
#define CV_HAL_ERROR_UNKNOWN -1

#define CV_HAL_CMP_EQ 0
#define CV_HAL_CMP_GT 1
#define CV_HAL_CMP_GE 2
#define CV_HAL_CMP_LT 3
#define CV_HAL_CMP_LE 4
#define CV_HAL_CMP_NE 5

#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

/* primitive types */
/*
  schar  - signed 1 byte integer
  uchar  - unsigned 1 byte integer
  short  - signed 2 byte integer
  ushort - unsigned 2 byte integer
  int    - signed 4 byte integer
  uint   - unsigned 4 byte integer
  int64  - signed 8 byte integer
  uint64 - unsigned 8 byte integer
*/

#if !defined _MSC_VER && !defined __BORLANDC__
#  if defined __cplusplus && __cplusplus >= 201103L && !defined __APPLE__
#    include <cstdint>
     typedef std::uint32_t uint;
#  else
#    include <stdint.h>
     typedef uint32_t uint;
#  endif
#else
   typedef unsigned uint;
#endif

typedef signed char schar;

#ifndef __IPL_H__
   typedef unsigned char uchar;
   typedef unsigned short ushort;
#endif

#if defined _MSC_VER || defined __BORLANDC__
   typedef __int64 int64;
   typedef unsigned __int64 uint64;
#  define CV_BIG_INT(n)   n##I64
#  define CV_BIG_UINT(n)  n##UI64
#else
   typedef int64_t int64;
   typedef uint64_t uint64;
#  define CV_BIG_INT(n)   n##LL
#  define CV_BIG_UINT(n)  n##ULL
#endif

//! @}

#endif
