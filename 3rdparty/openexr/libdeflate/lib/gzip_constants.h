/*
 * gzip_constants.h - constants for the gzip wrapper format
 */

#ifndef LIB_GZIP_CONSTANTS_H
#define LIB_GZIP_CONSTANTS_H

#define GZIP_MIN_HEADER_SIZE	10
#define GZIP_FOOTER_SIZE	8
#define GZIP_MIN_OVERHEAD	(GZIP_MIN_HEADER_SIZE + GZIP_FOOTER_SIZE)

#define GZIP_ID1		0x1F
#define GZIP_ID2		0x8B

#define GZIP_CM_DEFLATE		8

#define GZIP_FTEXT		0x01
#define GZIP_FHCRC		0x02
#define GZIP_FEXTRA		0x04
#define GZIP_FNAME		0x08
#define GZIP_FCOMMENT		0x10
#define GZIP_FRESERVED		0xE0

#define GZIP_MTIME_UNAVAILABLE	0

#define GZIP_XFL_SLOWEST_COMPRESSION	0x02
#define GZIP_XFL_FASTEST_COMPRESSION	0x04

#define GZIP_OS_FAT		0
#define GZIP_OS_AMIGA		1
#define GZIP_OS_VMS		2
#define GZIP_OS_UNIX		3
#define GZIP_OS_VM_CMS		4
#define GZIP_OS_ATARI_TOS	5
#define GZIP_OS_HPFS		6
#define GZIP_OS_MACINTOSH	7
#define GZIP_OS_Z_SYSTEM	8
#define GZIP_OS_CP_M		9
#define GZIP_OS_TOPS_20		10
#define GZIP_OS_NTFS		11
#define GZIP_OS_QDOS		12
#define GZIP_OS_RISCOS		13
#define GZIP_OS_UNKNOWN		255

#endif /* LIB_GZIP_CONSTANTS_H */
