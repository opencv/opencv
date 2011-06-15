/**
 * This file has no copyright assigned and is placed in the Public Domain.
 * This file is part of the w64 mingw-runtime package.
 * No warranty is given; refer to the file DISCLAIMER within this package.
 */

#if defined(_MSC_VER) && !defined(_MSC_EXTENSIONS)
#define NONAMELESSUNION		1
#endif
#if defined(NONAMELESSSTRUCT) && \
   !defined(NONAMELESSUNION)
#define NONAMELESSUNION		1
#endif
#if defined(NONAMELESSUNION)  && \
   !defined(NONAMELESSSTRUCT)
#define NONAMELESSSTRUCT	1
#endif

#ifndef __ANONYMOUS_DEFINED
#define __ANONYMOUS_DEFINED
#if defined(__GNUC__) || defined(__GNUG__)
#define _ANONYMOUS_UNION	__extension__
#define _ANONYMOUS_STRUCT	__extension__
#else
#define _ANONYMOUS_UNION
#define _ANONYMOUS_STRUCT
#endif
#ifndef NONAMELESSUNION
#define _UNION_NAME(x)
#define _STRUCT_NAME(x)
#else /* NONAMELESSUNION */
#define _UNION_NAME(x)  x
#define _STRUCT_NAME(x) x
#endif
#endif	/* __ANONYMOUS_DEFINED */

#ifndef DUMMYUNIONNAME
# ifdef NONAMELESSUNION
#  define DUMMYUNIONNAME  u
#  define DUMMYUNIONNAME1 u1	/* Wine uses this variant */
#  define DUMMYUNIONNAME2 u2
#  define DUMMYUNIONNAME3 u3
#  define DUMMYUNIONNAME4 u4
#  define DUMMYUNIONNAME5 u5
#  define DUMMYUNIONNAME6 u6
#  define DUMMYUNIONNAME7 u7
#  define DUMMYUNIONNAME8 u8
#  define DUMMYUNIONNAME9 u9
# else /* NONAMELESSUNION */
#  define DUMMYUNIONNAME
#  define DUMMYUNIONNAME1	/* Wine uses this variant */
#  define DUMMYUNIONNAME2
#  define DUMMYUNIONNAME3
#  define DUMMYUNIONNAME4
#  define DUMMYUNIONNAME5
#  define DUMMYUNIONNAME6
#  define DUMMYUNIONNAME7
#  define DUMMYUNIONNAME8
#  define DUMMYUNIONNAME9
# endif
#endif	/* DUMMYUNIONNAME */

#if !defined(DUMMYUNIONNAME1)	/* MinGW does not define this one */
# ifdef NONAMELESSUNION
#  define DUMMYUNIONNAME1 u1	/* Wine uses this variant */
# else
#  define DUMMYUNIONNAME1	/* Wine uses this variant */
# endif
#endif	/* DUMMYUNIONNAME1 */

#ifndef DUMMYSTRUCTNAME
# ifdef NONAMELESSUNION
#  define DUMMYSTRUCTNAME  s
#  define DUMMYSTRUCTNAME1 s1	/* Wine uses this variant */
#  define DUMMYSTRUCTNAME2 s2
#  define DUMMYSTRUCTNAME3 s3
#  define DUMMYSTRUCTNAME4 s4
#  define DUMMYSTRUCTNAME5 s5
# else
#  define DUMMYSTRUCTNAME
#  define DUMMYSTRUCTNAME1	/* Wine uses this variant */
#  define DUMMYSTRUCTNAME2
#  define DUMMYSTRUCTNAME3
#  define DUMMYSTRUCTNAME4
#  define DUMMYSTRUCTNAME5
# endif
#endif /* DUMMYSTRUCTNAME */

/* These are for compatibility with the Wine source tree */

#ifndef WINELIB_NAME_AW
# ifdef __MINGW_NAME_AW
#   define WINELIB_NAME_AW  __MINGW_NAME_AW
# else
#  ifdef UNICODE
#   define WINELIB_NAME_AW(func) func##W
#  else
#   define WINELIB_NAME_AW(func) func##A
#  endif
# endif
#endif	/* WINELIB_NAME_AW */

#ifndef DECL_WINELIB_TYPE_AW
# ifdef __MINGW_TYPEDEF_AW
#  define DECL_WINELIB_TYPE_AW  __MINGW_TYPEDEF_AW
# else
#  define DECL_WINELIB_TYPE_AW(type)  typedef WINELIB_NAME_AW(type) type;
# endif
#endif	/* DECL_WINELIB_TYPE_AW */

