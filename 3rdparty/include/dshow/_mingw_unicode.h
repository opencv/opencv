/**
 * This file has no copyright assigned and is placed in the Public Domain.
 * This file is part of the w64 mingw-runtime package.
 * No warranty is given; refer to the file DISCLAIMER.PD within this package.
 */

#if !defined(_INC_CRT_UNICODE_MACROS)
/* _INC_CRT_UNICODE_MACROS defined based on UNICODE flag */

#if defined(UNICODE)
# define _INC_CRT_UNICODE_MACROS 1
# define __MINGW_NAME_AW(func) func##W
# define __MINGW_NAME_AW_EXT(func,ext) func##W##ext
# define __MINGW_NAME_UAW(func) func##_W
# define __MINGW_NAME_UAW_EXT(func,ext) func##_W_##ext
# define __MINGW_STRING_AW(str) L##str	/* same as TEXT() from winnt.h */
# define __MINGW_PROCNAMEEXT_AW "W"
#else
# define _INC_CRT_UNICODE_MACROS 2
# define __MINGW_NAME_AW(func) func##A
# define __MINGW_NAME_AW_EXT(func,ext) func##A##ext
# define __MINGW_NAME_UAW(func) func##_A
# define __MINGW_NAME_UAW_EXT(func,ext) func##_A_##ext
# define __MINGW_STRING_AW(str) str	/* same as TEXT() from winnt.h */
# define __MINGW_PROCNAMEEXT_AW "A"
#endif

#define __MINGW_TYPEDEF_AW(type)	\
    typedef __MINGW_NAME_AW(type) type;
#define __MINGW_TYPEDEF_UAW(type)	\
    typedef __MINGW_NAME_UAW(type) type;

#endif /* !defined(_INC_CRT_UNICODE_MACROS) */
