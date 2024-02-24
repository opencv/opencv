/*
 * deflate_constants.h - constants for the DEFLATE compression format
 */

#ifndef LIB_DEFLATE_CONSTANTS_H
#define LIB_DEFLATE_CONSTANTS_H

/* Valid block types  */
#define DEFLATE_BLOCKTYPE_UNCOMPRESSED		0
#define DEFLATE_BLOCKTYPE_STATIC_HUFFMAN	1
#define DEFLATE_BLOCKTYPE_DYNAMIC_HUFFMAN	2

/* Minimum and maximum supported match lengths (in bytes)  */
#define DEFLATE_MIN_MATCH_LEN			3
#define DEFLATE_MAX_MATCH_LEN			258

/* Maximum supported match offset (in bytes) */
#define DEFLATE_MAX_MATCH_OFFSET		32768

/* log2 of DEFLATE_MAX_MATCH_OFFSET */
#define DEFLATE_WINDOW_ORDER			15

/* Number of symbols in each Huffman code.  Note: for the literal/length
 * and offset codes, these are actually the maximum values; a given block
 * might use fewer symbols.  */
#define DEFLATE_NUM_PRECODE_SYMS		19
#define DEFLATE_NUM_LITLEN_SYMS			288
#define DEFLATE_NUM_OFFSET_SYMS			32

/* The maximum number of symbols across all codes  */
#define DEFLATE_MAX_NUM_SYMS			288

/* Division of symbols in the literal/length code  */
#define DEFLATE_NUM_LITERALS			256
#define DEFLATE_END_OF_BLOCK			256
#define DEFLATE_FIRST_LEN_SYM			257

/* Maximum codeword length, in bits, within each Huffman code  */
#define DEFLATE_MAX_PRE_CODEWORD_LEN		7
#define DEFLATE_MAX_LITLEN_CODEWORD_LEN		15
#define DEFLATE_MAX_OFFSET_CODEWORD_LEN		15

/* The maximum codeword length across all codes  */
#define DEFLATE_MAX_CODEWORD_LEN		15

/* Maximum possible overrun when decoding codeword lengths  */
#define DEFLATE_MAX_LENS_OVERRUN		137

/*
 * Maximum number of extra bits that may be required to represent a match
 * length or offset.
 */
#define DEFLATE_MAX_EXTRA_LENGTH_BITS		5
#define DEFLATE_MAX_EXTRA_OFFSET_BITS		13

#endif /* LIB_DEFLATE_CONSTANTS_H */
