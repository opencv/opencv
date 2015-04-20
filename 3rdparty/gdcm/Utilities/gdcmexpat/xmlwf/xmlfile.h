/* Copyright (c) 1998, 1999 Thai Open Source Software Center Ltd
   See the file COPYING for copying permission.
*/

#define XML_MAP_FILE 01
#define XML_EXTERNAL_ENTITIES 02

#ifdef XML_LARGE_SIZE
#if defined(XML_USE_MSC_EXTENSIONS) && _MSC_VER < 1400
#define XML_FMT_INT_MOD "I64"
#else
#define XML_FMT_INT_MOD "ll"
#endif
#else
#define XML_FMT_INT_MOD "l"
#endif

extern int XML_ProcessFile(XML_Parser parser,
                           const XML_Char *filename,
                           unsigned flags);
