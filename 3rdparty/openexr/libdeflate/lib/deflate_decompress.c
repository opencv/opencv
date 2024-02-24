/*
 * deflate_decompress.c - a decompressor for DEFLATE
 *
 * Copyright 2016 Eric Biggers
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * ---------------------------------------------------------------------------
 *
 * This is a highly optimized DEFLATE decompressor.  It is much faster than
 * vanilla zlib, typically well over twice as fast, though results vary by CPU.
 *
 * Why this is faster than vanilla zlib:
 *
 * - Word accesses rather than byte accesses when reading input
 * - Word accesses rather than byte accesses when copying matches
 * - Faster Huffman decoding combined with various DEFLATE-specific tricks
 * - Larger bitbuffer variable that doesn't need to be refilled as often
 * - Other optimizations to remove unnecessary branches
 * - Only full-buffer decompression is supported, so the code doesn't need to
 *   support stopping and resuming decompression.
 * - On x86_64, a version of the decompression routine is compiled with BMI2
 *   instructions enabled and is used automatically at runtime when supported.
 */

#include "lib_common.h"
#include "deflate_constants.h"

/*
 * If the expression passed to SAFETY_CHECK() evaluates to false, then the
 * decompression routine immediately returns LIBDEFLATE_BAD_DATA, indicating the
 * compressed data is invalid.
 *
 * Theoretically, these checks could be disabled for specialized applications
 * where all input to the decompressor will be trusted.
 */
#if 0
#  pragma message("UNSAFE DECOMPRESSION IS ENABLED. THIS MUST ONLY BE USED IF THE DECOMPRESSOR INPUT WILL ALWAYS BE TRUSTED!")
#  define SAFETY_CHECK(expr)	(void)(expr)
#else
#  define SAFETY_CHECK(expr)	if (unlikely(!(expr))) return LIBDEFLATE_BAD_DATA
#endif

/*****************************************************************************
 *				Input bitstream                              *
 *****************************************************************************/

/*
 * The state of the "input bitstream" consists of the following variables:
 *
 *	- in_next: a pointer to the next unread byte in the input buffer
 *
 *	- in_end: a pointer to just past the end of the input buffer
 *
 *	- bitbuf: a word-sized variable containing bits that have been read from
 *		  the input buffer or from the implicit appended zero bytes
 *
 *	- bitsleft: the number of bits in 'bitbuf' available to be consumed.
 *		    After REFILL_BITS_BRANCHLESS(), 'bitbuf' can actually
 *		    contain more bits than this.  However, only the bits counted
 *		    by 'bitsleft' can actually be consumed; the rest can only be
 *		    used for preloading.
 *
 *		    As a micro-optimization, we allow bits 8 and higher of
 *		    'bitsleft' to contain garbage.  When consuming the bits
 *		    associated with a decode table entry, this allows us to do
 *		    'bitsleft -= entry' instead of 'bitsleft -= (u8)entry'.
 *		    On some CPUs, this helps reduce instruction dependencies.
 *		    This does have the disadvantage that 'bitsleft' sometimes
 *		    needs to be cast to 'u8', such as when it's used as a shift
 *		    amount in REFILL_BITS_BRANCHLESS().  But that one happens
 *		    for free since most CPUs ignore high bits in shift amounts.
 *
 *	- overread_count: the total number of implicit appended zero bytes that
 *			  have been loaded into the bitbuffer, including any
 *			  counted by 'bitsleft' and any already consumed
 */

/*
 * The type for the bitbuffer variable ('bitbuf' described above).  For best
 * performance, this should have size equal to a machine word.
 *
 * 64-bit platforms have a significant advantage: they get a bigger bitbuffer
 * which they don't have to refill as often.
 */
typedef machine_word_t bitbuf_t;
#define BITBUF_NBITS	(8 * (int)sizeof(bitbuf_t))

/* BITMASK(n) returns a bitmask of length 'n'. */
#define BITMASK(n)	(((bitbuf_t)1 << (n)) - 1)

/*
 * MAX_BITSLEFT is the maximum number of consumable bits, i.e. the maximum value
 * of '(u8)bitsleft'.  This is the size of the bitbuffer variable, minus 1 if
 * the branchless refill method is being used (see REFILL_BITS_BRANCHLESS()).
 */
#define MAX_BITSLEFT	\
	(UNALIGNED_ACCESS_IS_FAST ? BITBUF_NBITS - 1 : BITBUF_NBITS)

/*
 * CONSUMABLE_NBITS is the minimum number of bits that are guaranteed to be
 * consumable (counted in 'bitsleft') immediately after refilling the bitbuffer.
 * Since only whole bytes can be added to 'bitsleft', the worst case is
 * 'MAX_BITSLEFT - 7': the smallest amount where another byte doesn't fit.
 */
#define CONSUMABLE_NBITS	(MAX_BITSLEFT - 7)

/*
 * FASTLOOP_PRELOADABLE_NBITS is the minimum number of bits that are guaranteed
 * to be preloadable immediately after REFILL_BITS_IN_FASTLOOP().  (It is *not*
 * guaranteed after REFILL_BITS(), since REFILL_BITS() falls back to a
 * byte-at-a-time refill method near the end of input.)  This may exceed the
 * number of consumable bits (counted by 'bitsleft').  Any bits not counted in
 * 'bitsleft' can only be used for precomputation and cannot be consumed.
 */
#define FASTLOOP_PRELOADABLE_NBITS	\
	(UNALIGNED_ACCESS_IS_FAST ? BITBUF_NBITS : CONSUMABLE_NBITS)

/*
 * PRELOAD_SLACK is the minimum number of bits that are guaranteed to be
 * preloadable but not consumable, following REFILL_BITS_IN_FASTLOOP() and any
 * subsequent consumptions.  This is 1 bit if the branchless refill method is
 * being used, and 0 bits otherwise.
 */
#define PRELOAD_SLACK	MAX(0, FASTLOOP_PRELOADABLE_NBITS - MAX_BITSLEFT)

/*
 * CAN_CONSUME(n) is true if it's guaranteed that if the bitbuffer has just been
 * refilled, then it's always possible to consume 'n' bits from it.  'n' should
 * be a compile-time constant, to enable compile-time evaluation.
 */
#define CAN_CONSUME(n)	(CONSUMABLE_NBITS >= (n))

/*
 * CAN_CONSUME_AND_THEN_PRELOAD(consume_nbits, preload_nbits) is true if it's
 * guaranteed that after REFILL_BITS_IN_FASTLOOP(), it's always possible to
 * consume 'consume_nbits' bits, then preload 'preload_nbits' bits.  The
 * arguments should be compile-time constants to enable compile-time evaluation.
 */
#define CAN_CONSUME_AND_THEN_PRELOAD(consume_nbits, preload_nbits)	\
	(CONSUMABLE_NBITS >= (consume_nbits) &&				\
	 FASTLOOP_PRELOADABLE_NBITS >= (consume_nbits) + (preload_nbits))

/*
 * REFILL_BITS_BRANCHLESS() branchlessly refills the bitbuffer variable by
 * reading the next word from the input buffer and updating 'in_next' and
 * 'bitsleft' based on how many bits were refilled -- counting whole bytes only.
 * This is much faster than reading a byte at a time, at least if the CPU is
 * little endian and supports fast unaligned memory accesses.
 *
 * The simplest way of branchlessly updating 'bitsleft' would be:
 *
 *	bitsleft += (MAX_BITSLEFT - bitsleft) & ~7;
 *
 * To make it faster, we define MAX_BITSLEFT to be 'WORDBITS - 1' rather than
 * WORDBITS, so that in binary it looks like 111111 or 11111.  Then, we update
 * 'bitsleft' by just setting the bits above the low 3 bits:
 *
 *	bitsleft |= MAX_BITSLEFT & ~7;
 *
 * That compiles down to a single instruction like 'or $0x38, %rbp'.  Using
 * 'MAX_BITSLEFT == WORDBITS - 1' also has the advantage that refills can be
 * done when 'bitsleft == MAX_BITSLEFT' without invoking undefined behavior.
 *
 * The simplest way of branchlessly updating 'in_next' would be:
 *
 *	in_next += (MAX_BITSLEFT - bitsleft) >> 3;
 *
 * With 'MAX_BITSLEFT == WORDBITS - 1' we could use an XOR instead, though this
 * isn't really better:
 *
 *	in_next += (MAX_BITSLEFT ^ bitsleft) >> 3;
 *
 * An alternative which can be marginally better is the following:
 *
 *	in_next += sizeof(bitbuf_t) - 1;
 *	in_next -= (bitsleft >> 3) & 0x7;
 *
 * It seems this would increase the number of CPU instructions from 3 (sub, shr,
 * add) to 4 (add, shr, and, sub).  However, if the CPU has a bitfield
 * extraction instruction (e.g. arm's ubfx), it stays at 3, and is potentially
 * more efficient because the length of the longest dependency chain decreases
 * from 3 to 2.  This alternative also has the advantage that it ignores the
 * high bits in 'bitsleft', so it is compatible with the micro-optimization we
 * use where we let the high bits of 'bitsleft' contain garbage.
 */
#define REFILL_BITS_BRANCHLESS()					\
do {									\
	bitbuf |= get_unaligned_leword(in_next) << (u8)bitsleft;	\
	in_next += sizeof(bitbuf_t) - 1;				\
	in_next -= (bitsleft >> 3) & 0x7;				\
	bitsleft |= MAX_BITSLEFT & ~7;					\
} while (0)

/*
 * REFILL_BITS() loads bits from the input buffer until the bitbuffer variable
 * contains at least CONSUMABLE_NBITS consumable bits.
 *
 * This checks for the end of input, and it doesn't guarantee
 * FASTLOOP_PRELOADABLE_NBITS, so it can't be used in the fastloop.
 *
 * If we would overread the input buffer, we just don't read anything, leaving
 * the bits zeroed but marking them filled.  This simplifies the decompressor
 * because it removes the need to always be able to distinguish between real
 * overreads and overreads caused only by the decompressor's own lookahead.
 *
 * We do still keep track of the number of bytes that have been overread, for
 * two reasons.  First, it allows us to determine the exact number of bytes that
 * were consumed once the stream ends or an uncompressed block is reached.
 * Second, it allows us to stop early if the overread amount gets so large (more
 * than sizeof bitbuf) that it can only be caused by a real overread.  (The
 * second part is arguably unneeded, since libdeflate is buffer-based; given
 * infinite zeroes, it will eventually either completely fill the output buffer
 * or return an error.  However, we do it to be slightly more friendly to the
 * not-recommended use case of decompressing with an unknown output size.)
 */
#define REFILL_BITS()							\
do {									\
	if (UNALIGNED_ACCESS_IS_FAST &&					\
	    likely(in_end - in_next >= sizeof(bitbuf_t))) {		\
		REFILL_BITS_BRANCHLESS();				\
	} else {							\
		while ((u8)bitsleft < CONSUMABLE_NBITS) {		\
			if (likely(in_next != in_end)) {		\
				bitbuf |= (bitbuf_t)*in_next++ <<	\
					  (u8)bitsleft;			\
			} else {					\
				overread_count++;			\
				SAFETY_CHECK(overread_count <=		\
					     sizeof(bitbuf_t));		\
			}						\
			bitsleft += 8;					\
		}							\
	}								\
} while (0)

/*
 * REFILL_BITS_IN_FASTLOOP() is like REFILL_BITS(), but it doesn't check for the
 * end of the input.  It can only be used in the fastloop.
 */
#define REFILL_BITS_IN_FASTLOOP()					\
do {									\
	STATIC_ASSERT(UNALIGNED_ACCESS_IS_FAST ||			\
		      FASTLOOP_PRELOADABLE_NBITS == CONSUMABLE_NBITS);	\
	if (UNALIGNED_ACCESS_IS_FAST) {					\
		REFILL_BITS_BRANCHLESS();				\
	} else {							\
		while ((u8)bitsleft < CONSUMABLE_NBITS) {		\
			bitbuf |= (bitbuf_t)*in_next++ << (u8)bitsleft;	\
			bitsleft += 8;					\
		}							\
	}								\
} while (0)

/*
 * This is the worst-case maximum number of output bytes that are written to
 * during each iteration of the fastloop.  The worst case is 2 literals, then a
 * match of length DEFLATE_MAX_MATCH_LEN.  Additionally, some slack space must
 * be included for the intentional overrun in the match copy implementation.
 */
#define FASTLOOP_MAX_BYTES_WRITTEN	\
	(2 + DEFLATE_MAX_MATCH_LEN + (5 * WORDBYTES) - 1)

/*
 * This is the worst-case maximum number of input bytes that are read during
 * each iteration of the fastloop.  To get this value, we first compute the
 * greatest number of bits that can be refilled during a loop iteration.  The
 * refill at the beginning can add at most MAX_BITSLEFT, and the amount that can
 * be refilled later is no more than the maximum amount that can be consumed by
 * 2 literals that don't need a subtable, then a match.  We convert this value
 * to bytes, rounding up; this gives the maximum number of bytes that 'in_next'
 * can be advanced.  Finally, we add sizeof(bitbuf_t) to account for
 * REFILL_BITS_BRANCHLESS() reading a word past 'in_next'.
 */
#define FASTLOOP_MAX_BYTES_READ					\
	(DIV_ROUND_UP(MAX_BITSLEFT + (2 * LITLEN_TABLEBITS) +	\
		      LENGTH_MAXBITS + OFFSET_MAXBITS, 8) +	\
	 sizeof(bitbuf_t))

/*****************************************************************************
 *                              Huffman decoding                             *
 *****************************************************************************/

/*
 * The fastest way to decode Huffman-encoded data is basically to use a decode
 * table that maps the next TABLEBITS bits of data to their symbol.  Each entry
 * decode_table[i] maps to the symbol whose codeword is a prefix of 'i'.  A
 * symbol with codeword length 'n' has '2**(TABLEBITS-n)' entries in the table.
 *
 * Ideally, TABLEBITS and the maximum codeword length would be the same; some
 * compression formats are designed with this goal in mind.  Unfortunately, in
 * DEFLATE, the maximum litlen and offset codeword lengths are 15 bits, which is
 * too large for a practical TABLEBITS.  It's not *that* much larger, though, so
 * the workaround is to use a single level of subtables.  In the main table,
 * entries for prefixes of codewords longer than TABLEBITS contain a "pointer"
 * to the appropriate subtable along with the number of bits it is indexed with.
 *
 * The most efficient way to allocate subtables is to allocate them dynamically
 * after the main table.  The worst-case number of table entries needed,
 * including subtables, is precomputable; see the ENOUGH constants below.
 *
 * A useful optimization is to store the codeword lengths in the decode table so
 * that they don't have to be looked up by indexing a separate table that maps
 * symbols to their codeword lengths.  We basically do this; however, for the
 * litlen and offset codes we also implement some DEFLATE-specific optimizations
 * that build in the consideration of the "extra bits" and the
 * literal/length/end-of-block division.  For the exact decode table entry
 * format we use, see the definitions of the *_decode_results[] arrays below.
 */


/*
 * These are the TABLEBITS values we use for each of the DEFLATE Huffman codes,
 * along with their corresponding ENOUGH values.
 *
 * For the precode, we use PRECODE_TABLEBITS == 7 since this is the maximum
 * precode codeword length.  This avoids ever needing subtables.
 *
 * For the litlen and offset codes, we cannot realistically avoid ever needing
 * subtables, since litlen and offset codewords can be up to 15 bits.  A higher
 * TABLEBITS reduces the number of lookups that need a subtable, which increases
 * performance; however, it increases memory usage and makes building the table
 * take longer, which decreases performance.  We choose values that work well in
 * practice, making subtables rarely needed without making the tables too large.
 *
 * Our choice of OFFSET_TABLEBITS == 8 is a bit low; without any special
 * considerations, 9 would fit the trade-off curve better.  However, there is a
 * performance benefit to using exactly 8 bits when it is a compile-time
 * constant, as many CPUs can take the low byte more easily than the low 9 bits.
 *
 * zlib treats its equivalents of TABLEBITS as maximum values; whenever it
 * builds a table, it caps the actual table_bits to the longest codeword.  This
 * makes sense in theory, as there's no need for the table to be any larger than
 * needed to support the longest codeword.  However, having the table bits be a
 * compile-time constant is beneficial to the performance of the decode loop, so
 * there is a trade-off.  libdeflate currently uses the dynamic table_bits
 * strategy for the litlen table only, due to its larger maximum size.
 * PRECODE_TABLEBITS and OFFSET_TABLEBITS are smaller, so going dynamic there
 * isn't as useful, and OFFSET_TABLEBITS=8 is useful as mentioned above.
 *
 * Each TABLEBITS value has a corresponding ENOUGH value that gives the
 * worst-case maximum number of decode table entries, including the main table
 * and all subtables.  The ENOUGH value depends on three parameters:
 *
 *	(1) the maximum number of symbols in the code (DEFLATE_NUM_*_SYMS)
 *	(2) the maximum number of main table bits (*_TABLEBITS)
 *	(3) the maximum allowed codeword length (DEFLATE_MAX_*_CODEWORD_LEN)
 *
 * The ENOUGH values were computed using the utility program 'enough' from zlib.
 */
#define PRECODE_TABLEBITS	7
#define PRECODE_ENOUGH		128	/* enough 19 7 7	*/
#define LITLEN_TABLEBITS	11
#define LITLEN_ENOUGH		2342	/* enough 288 11 15	*/
#define OFFSET_TABLEBITS	8
#define OFFSET_ENOUGH		402	/* enough 32 8 15	*/

/*
 * make_decode_table_entry() creates a decode table entry for the given symbol
 * by combining the static part 'decode_results[sym]' with the dynamic part
 * 'len', which is the remaining codeword length (the codeword length for main
 * table entries, or the codeword length minus TABLEBITS for subtable entries).
 *
 * In all cases, we add 'len' to each of the two low-order bytes to create the
 * appropriately-formatted decode table entry.  See the definitions of the
 * *_decode_results[] arrays below, where the entry format is described.
 */
static forceinline u32
make_decode_table_entry(const u32 decode_results[], u32 sym, u32 len)
{
	return decode_results[sym] + (len << 8) + len;
}

/*
 * Here is the format of our precode decode table entries.  Bits not explicitly
 * described contain zeroes:
 *
 *	Bit 20-16:  presym
 *	Bit 10-8:   codeword length [not used]
 *	Bit 2-0:    codeword length
 *
 * The precode decode table never has subtables, since we use
 * PRECODE_TABLEBITS == DEFLATE_MAX_PRE_CODEWORD_LEN.
 *
 * precode_decode_results[] contains the static part of the entry for each
 * symbol.  make_decode_table_entry() produces the final entries.
 */
static const u32 precode_decode_results[] = {
#define ENTRY(presym)	((u32)presym << 16)
	ENTRY(0)   , ENTRY(1)   , ENTRY(2)   , ENTRY(3)   ,
	ENTRY(4)   , ENTRY(5)   , ENTRY(6)   , ENTRY(7)   ,
	ENTRY(8)   , ENTRY(9)   , ENTRY(10)  , ENTRY(11)  ,
	ENTRY(12)  , ENTRY(13)  , ENTRY(14)  , ENTRY(15)  ,
	ENTRY(16)  , ENTRY(17)  , ENTRY(18)  ,
#undef ENTRY
};

/* Litlen and offset decode table entry flags */

/* Indicates a literal entry in the litlen decode table */
#define HUFFDEC_LITERAL			0x80000000

/* Indicates that HUFFDEC_SUBTABLE_POINTER or HUFFDEC_END_OF_BLOCK is set */
#define HUFFDEC_EXCEPTIONAL		0x00008000

/* Indicates a subtable pointer entry in the litlen or offset decode table */
#define HUFFDEC_SUBTABLE_POINTER	0x00004000

/* Indicates an end-of-block entry in the litlen decode table */
#define HUFFDEC_END_OF_BLOCK		0x00002000

/* Maximum number of bits that can be consumed by decoding a match length */
#define LENGTH_MAXBITS		(DEFLATE_MAX_LITLEN_CODEWORD_LEN + \
				 DEFLATE_MAX_EXTRA_LENGTH_BITS)
#define LENGTH_MAXFASTBITS	(LITLEN_TABLEBITS /* no subtable needed */ + \
				 DEFLATE_MAX_EXTRA_LENGTH_BITS)

/*
 * Here is the format of our litlen decode table entries.  Bits not explicitly
 * described contain zeroes:
 *
 *	Literals:
 *		Bit 31:     1 (HUFFDEC_LITERAL)
 *		Bit 23-16:  literal value
 *		Bit 15:     0 (!HUFFDEC_EXCEPTIONAL)
 *		Bit 14:     0 (!HUFFDEC_SUBTABLE_POINTER)
 *		Bit 13:     0 (!HUFFDEC_END_OF_BLOCK)
 *		Bit 11-8:   remaining codeword length [not used]
 *		Bit 3-0:    remaining codeword length
 *	Lengths:
 *		Bit 31:     0 (!HUFFDEC_LITERAL)
 *		Bit 24-16:  length base value
 *		Bit 15:     0 (!HUFFDEC_EXCEPTIONAL)
 *		Bit 14:     0 (!HUFFDEC_SUBTABLE_POINTER)
 *		Bit 13:     0 (!HUFFDEC_END_OF_BLOCK)
 *		Bit 11-8:   remaining codeword length
 *		Bit 4-0:    remaining codeword length + number of extra bits
 *	End of block:
 *		Bit 31:     0 (!HUFFDEC_LITERAL)
 *		Bit 15:     1 (HUFFDEC_EXCEPTIONAL)
 *		Bit 14:     0 (!HUFFDEC_SUBTABLE_POINTER)
 *		Bit 13:     1 (HUFFDEC_END_OF_BLOCK)
 *		Bit 11-8:   remaining codeword length [not used]
 *		Bit 3-0:    remaining codeword length
 *	Subtable pointer:
 *		Bit 31:     0 (!HUFFDEC_LITERAL)
 *		Bit 30-16:  index of start of subtable
 *		Bit 15:     1 (HUFFDEC_EXCEPTIONAL)
 *		Bit 14:     1 (HUFFDEC_SUBTABLE_POINTER)
 *		Bit 13:     0 (!HUFFDEC_END_OF_BLOCK)
 *		Bit 11-8:   number of subtable bits
 *		Bit 3-0:    number of main table bits
 *
 * This format has several desirable properties:
 *
 *	- The codeword length, length slot base, and number of extra length bits
 *	  are all built in.  This eliminates the need to separately look up this
 *	  information by indexing separate arrays by symbol or length slot.
 *
 *	- The HUFFDEC_* flags enable easily distinguishing between the different
 *	  types of entries.  The HUFFDEC_LITERAL flag enables a fast path for
 *	  literals; the high bit is used for this, as some CPUs can test the
 *	  high bit more easily than other bits.  The HUFFDEC_EXCEPTIONAL flag
 *	  makes it possible to detect the two unlikely cases (subtable pointer
 *	  and end of block) in a single bit flag test.
 *
 *	- The low byte is the number of bits that need to be removed from the
 *	  bitstream; this makes this value easily accessible, and it enables the
 *	  micro-optimization of doing 'bitsleft -= entry' instead of
 *	  'bitsleft -= (u8)entry'.  It also includes the number of extra bits,
 *	  so they don't need to be removed separately.
 *
 *	- The flags in bits 15-13 are arranged to be 0 when the
 *	  "remaining codeword length" in bits 11-8 is needed, making this value
 *	  fairly easily accessible as well via a shift and downcast.
 *
 *	- Similarly, bits 13-12 are 0 when the "subtable bits" in bits 11-8 are
 *	  needed, making it possible to extract this value with '& 0x3F' rather
 *	  than '& 0xF'.  This value is only used as a shift amount, so this can
 *	  save an 'and' instruction as the masking by 0x3F happens implicitly.
 *
 * litlen_decode_results[] contains the static part of the entry for each
 * symbol.  make_decode_table_entry() produces the final entries.
 */
static const u32 litlen_decode_results[] = {

	/* Literals */
#define ENTRY(literal)	(HUFFDEC_LITERAL | ((u32)literal << 16))
	ENTRY(0)   , ENTRY(1)   , ENTRY(2)   , ENTRY(3)   ,
	ENTRY(4)   , ENTRY(5)   , ENTRY(6)   , ENTRY(7)   ,
	ENTRY(8)   , ENTRY(9)   , ENTRY(10)  , ENTRY(11)  ,
	ENTRY(12)  , ENTRY(13)  , ENTRY(14)  , ENTRY(15)  ,
	ENTRY(16)  , ENTRY(17)  , ENTRY(18)  , ENTRY(19)  ,
	ENTRY(20)  , ENTRY(21)  , ENTRY(22)  , ENTRY(23)  ,
	ENTRY(24)  , ENTRY(25)  , ENTRY(26)  , ENTRY(27)  ,
	ENTRY(28)  , ENTRY(29)  , ENTRY(30)  , ENTRY(31)  ,
	ENTRY(32)  , ENTRY(33)  , ENTRY(34)  , ENTRY(35)  ,
	ENTRY(36)  , ENTRY(37)  , ENTRY(38)  , ENTRY(39)  ,
	ENTRY(40)  , ENTRY(41)  , ENTRY(42)  , ENTRY(43)  ,
	ENTRY(44)  , ENTRY(45)  , ENTRY(46)  , ENTRY(47)  ,
	ENTRY(48)  , ENTRY(49)  , ENTRY(50)  , ENTRY(51)  ,
	ENTRY(52)  , ENTRY(53)  , ENTRY(54)  , ENTRY(55)  ,
	ENTRY(56)  , ENTRY(57)  , ENTRY(58)  , ENTRY(59)  ,
	ENTRY(60)  , ENTRY(61)  , ENTRY(62)  , ENTRY(63)  ,
	ENTRY(64)  , ENTRY(65)  , ENTRY(66)  , ENTRY(67)  ,
	ENTRY(68)  , ENTRY(69)  , ENTRY(70)  , ENTRY(71)  ,
	ENTRY(72)  , ENTRY(73)  , ENTRY(74)  , ENTRY(75)  ,
	ENTRY(76)  , ENTRY(77)  , ENTRY(78)  , ENTRY(79)  ,
	ENTRY(80)  , ENTRY(81)  , ENTRY(82)  , ENTRY(83)  ,
	ENTRY(84)  , ENTRY(85)  , ENTRY(86)  , ENTRY(87)  ,
	ENTRY(88)  , ENTRY(89)  , ENTRY(90)  , ENTRY(91)  ,
	ENTRY(92)  , ENTRY(93)  , ENTRY(94)  , ENTRY(95)  ,
	ENTRY(96)  , ENTRY(97)  , ENTRY(98)  , ENTRY(99)  ,
	ENTRY(100) , ENTRY(101) , ENTRY(102) , ENTRY(103) ,
	ENTRY(104) , ENTRY(105) , ENTRY(106) , ENTRY(107) ,
	ENTRY(108) , ENTRY(109) , ENTRY(110) , ENTRY(111) ,
	ENTRY(112) , ENTRY(113) , ENTRY(114) , ENTRY(115) ,
	ENTRY(116) , ENTRY(117) , ENTRY(118) , ENTRY(119) ,
	ENTRY(120) , ENTRY(121) , ENTRY(122) , ENTRY(123) ,
	ENTRY(124) , ENTRY(125) , ENTRY(126) , ENTRY(127) ,
	ENTRY(128) , ENTRY(129) , ENTRY(130) , ENTRY(131) ,
	ENTRY(132) , ENTRY(133) , ENTRY(134) , ENTRY(135) ,
	ENTRY(136) , ENTRY(137) , ENTRY(138) , ENTRY(139) ,
	ENTRY(140) , ENTRY(141) , ENTRY(142) , ENTRY(143) ,
	ENTRY(144) , ENTRY(145) , ENTRY(146) , ENTRY(147) ,
	ENTRY(148) , ENTRY(149) , ENTRY(150) , ENTRY(151) ,
	ENTRY(152) , ENTRY(153) , ENTRY(154) , ENTRY(155) ,
	ENTRY(156) , ENTRY(157) , ENTRY(158) , ENTRY(159) ,
	ENTRY(160) , ENTRY(161) , ENTRY(162) , ENTRY(163) ,
	ENTRY(164) , ENTRY(165) , ENTRY(166) , ENTRY(167) ,
	ENTRY(168) , ENTRY(169) , ENTRY(170) , ENTRY(171) ,
	ENTRY(172) , ENTRY(173) , ENTRY(174) , ENTRY(175) ,
	ENTRY(176) , ENTRY(177) , ENTRY(178) , ENTRY(179) ,
	ENTRY(180) , ENTRY(181) , ENTRY(182) , ENTRY(183) ,
	ENTRY(184) , ENTRY(185) , ENTRY(186) , ENTRY(187) ,
	ENTRY(188) , ENTRY(189) , ENTRY(190) , ENTRY(191) ,
	ENTRY(192) , ENTRY(193) , ENTRY(194) , ENTRY(195) ,
	ENTRY(196) , ENTRY(197) , ENTRY(198) , ENTRY(199) ,
	ENTRY(200) , ENTRY(201) , ENTRY(202) , ENTRY(203) ,
	ENTRY(204) , ENTRY(205) , ENTRY(206) , ENTRY(207) ,
	ENTRY(208) , ENTRY(209) , ENTRY(210) , ENTRY(211) ,
	ENTRY(212) , ENTRY(213) , ENTRY(214) , ENTRY(215) ,
	ENTRY(216) , ENTRY(217) , ENTRY(218) , ENTRY(219) ,
	ENTRY(220) , ENTRY(221) , ENTRY(222) , ENTRY(223) ,
	ENTRY(224) , ENTRY(225) , ENTRY(226) , ENTRY(227) ,
	ENTRY(228) , ENTRY(229) , ENTRY(230) , ENTRY(231) ,
	ENTRY(232) , ENTRY(233) , ENTRY(234) , ENTRY(235) ,
	ENTRY(236) , ENTRY(237) , ENTRY(238) , ENTRY(239) ,
	ENTRY(240) , ENTRY(241) , ENTRY(242) , ENTRY(243) ,
	ENTRY(244) , ENTRY(245) , ENTRY(246) , ENTRY(247) ,
	ENTRY(248) , ENTRY(249) , ENTRY(250) , ENTRY(251) ,
	ENTRY(252) , ENTRY(253) , ENTRY(254) , ENTRY(255) ,
#undef ENTRY

	/* End of block */
	HUFFDEC_EXCEPTIONAL | HUFFDEC_END_OF_BLOCK,

	/* Lengths */
#define ENTRY(length_base, num_extra_bits)	\
	(((u32)(length_base) << 16) | (num_extra_bits))
	ENTRY(3  , 0) , ENTRY(4  , 0) , ENTRY(5  , 0) , ENTRY(6  , 0),
	ENTRY(7  , 0) , ENTRY(8  , 0) , ENTRY(9  , 0) , ENTRY(10 , 0),
	ENTRY(11 , 1) , ENTRY(13 , 1) , ENTRY(15 , 1) , ENTRY(17 , 1),
	ENTRY(19 , 2) , ENTRY(23 , 2) , ENTRY(27 , 2) , ENTRY(31 , 2),
	ENTRY(35 , 3) , ENTRY(43 , 3) , ENTRY(51 , 3) , ENTRY(59 , 3),
	ENTRY(67 , 4) , ENTRY(83 , 4) , ENTRY(99 , 4) , ENTRY(115, 4),
	ENTRY(131, 5) , ENTRY(163, 5) , ENTRY(195, 5) , ENTRY(227, 5),
	ENTRY(258, 0) , ENTRY(258, 0) , ENTRY(258, 0) ,
#undef ENTRY
};

/* Maximum number of bits that can be consumed by decoding a match offset */
#define OFFSET_MAXBITS		(DEFLATE_MAX_OFFSET_CODEWORD_LEN + \
				 DEFLATE_MAX_EXTRA_OFFSET_BITS)
#define OFFSET_MAXFASTBITS	(OFFSET_TABLEBITS /* no subtable needed */ + \
				 DEFLATE_MAX_EXTRA_OFFSET_BITS)

/*
 * Here is the format of our offset decode table entries.  Bits not explicitly
 * described contain zeroes:
 *
 *	Offsets:
 *		Bit 31-16:  offset base value
 *		Bit 15:     0 (!HUFFDEC_EXCEPTIONAL)
 *		Bit 14:     0 (!HUFFDEC_SUBTABLE_POINTER)
 *		Bit 11-8:   remaining codeword length
 *		Bit 4-0:    remaining codeword length + number of extra bits
 *	Subtable pointer:
 *		Bit 31-16:  index of start of subtable
 *		Bit 15:     1 (HUFFDEC_EXCEPTIONAL)
 *		Bit 14:     1 (HUFFDEC_SUBTABLE_POINTER)
 *		Bit 11-8:   number of subtable bits
 *		Bit 3-0:    number of main table bits
 *
 * These work the same way as the length entries and subtable pointer entries in
 * the litlen decode table; see litlen_decode_results[] above.
 */
static const u32 offset_decode_results[] = {
#define ENTRY(offset_base, num_extra_bits)	\
	(((u32)(offset_base) << 16) | (num_extra_bits))
	ENTRY(1     , 0)  , ENTRY(2     , 0)  , ENTRY(3     , 0)  , ENTRY(4     , 0)  ,
	ENTRY(5     , 1)  , ENTRY(7     , 1)  , ENTRY(9     , 2)  , ENTRY(13    , 2) ,
	ENTRY(17    , 3)  , ENTRY(25    , 3)  , ENTRY(33    , 4)  , ENTRY(49    , 4)  ,
	ENTRY(65    , 5)  , ENTRY(97    , 5)  , ENTRY(129   , 6)  , ENTRY(193   , 6)  ,
	ENTRY(257   , 7)  , ENTRY(385   , 7)  , ENTRY(513   , 8)  , ENTRY(769   , 8)  ,
	ENTRY(1025  , 9)  , ENTRY(1537  , 9)  , ENTRY(2049  , 10) , ENTRY(3073  , 10) ,
	ENTRY(4097  , 11) , ENTRY(6145  , 11) , ENTRY(8193  , 12) , ENTRY(12289 , 12) ,
	ENTRY(16385 , 13) , ENTRY(24577 , 13) , ENTRY(24577 , 13) , ENTRY(24577 , 13) ,
#undef ENTRY
};

/*
 * The main DEFLATE decompressor structure.  Since libdeflate only supports
 * full-buffer decompression, this structure doesn't store the entire
 * decompression state, most of which is in stack variables.  Instead, this
 * struct just contains the decode tables and some temporary arrays used for
 * building them, as these are too large to comfortably allocate on the stack.
 *
 * Storing the decode tables in the decompressor struct also allows the decode
 * tables for the static codes to be reused whenever two static Huffman blocks
 * are decoded without an intervening dynamic block, even across streams.
 */
struct libdeflate_decompressor {

	/*
	 * The arrays aren't all needed at the same time.  'precode_lens' and
	 * 'precode_decode_table' are unneeded after 'lens' has been filled.
	 * Furthermore, 'lens' need not be retained after building the litlen
	 * and offset decode tables.  In fact, 'lens' can be in union with
	 * 'litlen_decode_table' provided that 'offset_decode_table' is separate
	 * and is built first.
	 */

	union {
		u8 precode_lens[DEFLATE_NUM_PRECODE_SYMS];

		struct {
			u8 lens[DEFLATE_NUM_LITLEN_SYMS +
				DEFLATE_NUM_OFFSET_SYMS +
				DEFLATE_MAX_LENS_OVERRUN];

			u32 precode_decode_table[PRECODE_ENOUGH];
		} l;

		u32 litlen_decode_table[LITLEN_ENOUGH];
	} u;

	u32 offset_decode_table[OFFSET_ENOUGH];

	/* used only during build_decode_table() */
	u16 sorted_syms[DEFLATE_MAX_NUM_SYMS];

	bool static_codes_loaded;
	unsigned litlen_tablebits;

	/* The free() function for this struct, chosen at allocation time */
	free_func_t free_func;
};

/*
 * Build a table for fast decoding of symbols from a Huffman code.  As input,
 * this function takes the codeword length of each symbol which may be used in
 * the code.  As output, it produces a decode table for the canonical Huffman
 * code described by the codeword lengths.  The decode table is built with the
 * assumption that it will be indexed with "bit-reversed" codewords, where the
 * low-order bit is the first bit of the codeword.  This format is used for all
 * Huffman codes in DEFLATE.
 *
 * @decode_table
 *	The array in which the decode table will be generated.  This array must
 *	have sufficient length; see the definition of the ENOUGH numbers.
 * @lens
 *	An array which provides, for each symbol, the length of the
 *	corresponding codeword in bits, or 0 if the symbol is unused.  This may
 *	alias @decode_table, since nothing is written to @decode_table until all
 *	@lens have been consumed.  All codeword lengths are assumed to be <=
 *	@max_codeword_len but are otherwise considered untrusted.  If they do
 *	not form a valid Huffman code, then the decode table is not built and
 *	%false is returned.
 * @num_syms
 *	The number of symbols in the code, including all unused symbols.
 * @decode_results
 *	An array which gives the incomplete decode result for each symbol.  The
 *	needed values in this array will be combined with codeword lengths to
 *	make the final decode table entries using make_decode_table_entry().
 * @table_bits
 *	The log base-2 of the number of main table entries to use.
 *	If @table_bits_ret != NULL, then @table_bits is treated as a maximum
 *	value and it will be decreased if a smaller table would be sufficient.
 * @max_codeword_len
 *	The maximum allowed codeword length for this Huffman code.
 *	Must be <= DEFLATE_MAX_CODEWORD_LEN.
 * @sorted_syms
 *	A temporary array of length @num_syms.
 * @table_bits_ret
 *	If non-NULL, then the dynamic table_bits is enabled, and the actual
 *	table_bits value will be returned here.
 *
 * Returns %true if successful; %false if the codeword lengths do not form a
 * valid Huffman code.
 */
static bool
build_decode_table(u32 decode_table[],
		   const u8 lens[],
		   const unsigned num_syms,
		   const u32 decode_results[],
		   unsigned table_bits,
		   unsigned max_codeword_len,
		   u16 *sorted_syms,
		   unsigned *table_bits_ret)
{
	unsigned len_counts[DEFLATE_MAX_CODEWORD_LEN + 1];
	unsigned offsets[DEFLATE_MAX_CODEWORD_LEN + 1];
	unsigned sym;		/* current symbol */
	unsigned codeword;	/* current codeword, bit-reversed */
	unsigned len;		/* current codeword length in bits */
	unsigned count;		/* num codewords remaining with this length */
	u32 codespace_used;	/* codespace used out of '2^max_codeword_len' */
	unsigned cur_table_end; /* end index of current table */
	unsigned subtable_prefix; /* codeword prefix of current subtable */
	unsigned subtable_start;  /* start index of current subtable */
	unsigned subtable_bits;   /* log2 of current subtable length */

	/* Count how many codewords have each length, including 0. */
	for (len = 0; len <= max_codeword_len; len++)
		len_counts[len] = 0;
	for (sym = 0; sym < num_syms; sym++)
		len_counts[lens[sym]]++;

	/*
	 * Determine the actual maximum codeword length that was used, and
	 * decrease table_bits to it if allowed.
	 */
	while (max_codeword_len > 1 && len_counts[max_codeword_len] == 0)
		max_codeword_len--;
	if (table_bits_ret != NULL) {
		table_bits = MIN(table_bits, max_codeword_len);
		*table_bits_ret = table_bits;
	}

	/*
	 * Sort the symbols primarily by increasing codeword length and
	 * secondarily by increasing symbol value; or equivalently by their
	 * codewords in lexicographic order, since a canonical code is assumed.
	 *
	 * For efficiency, also compute 'codespace_used' in the same pass over
	 * 'len_counts[]' used to build 'offsets[]' for sorting.
	 */

	/* Ensure that 'codespace_used' cannot overflow. */
	STATIC_ASSERT(sizeof(codespace_used) == 4);
	STATIC_ASSERT(UINT32_MAX / (1U << (DEFLATE_MAX_CODEWORD_LEN - 1)) >=
		      DEFLATE_MAX_NUM_SYMS);

	offsets[0] = 0;
	offsets[1] = len_counts[0];
	codespace_used = 0;
	for (len = 1; len < max_codeword_len; len++) {
		offsets[len + 1] = offsets[len] + len_counts[len];
		codespace_used = (codespace_used << 1) + len_counts[len];
	}
	codespace_used = (codespace_used << 1) + len_counts[len];

	for (sym = 0; sym < num_syms; sym++)
		sorted_syms[offsets[lens[sym]]++] = sym;

	sorted_syms += offsets[0]; /* Skip unused symbols */

	/* lens[] is done being used, so we can write to decode_table[] now. */

	/*
	 * Check whether the lengths form a complete code (exactly fills the
	 * codespace), an incomplete code (doesn't fill the codespace), or an
	 * overfull code (overflows the codespace).  A codeword of length 'n'
	 * uses proportion '1/(2^n)' of the codespace.  An overfull code is
	 * nonsensical, so is considered invalid.  An incomplete code is
	 * considered valid only in two specific cases; see below.
	 */

	/* overfull code? */
	if (unlikely(codespace_used > (1U << max_codeword_len)))
		return false;

	/* incomplete code? */
	if (unlikely(codespace_used < (1U << max_codeword_len))) {
		u32 entry;
		unsigned i;

		/*
		 * The DEFLATE RFC explicitly allows the offset code to be
		 * incomplete in two cases: a code containing just 1 codeword,
		 * if that codeword has length 1; and a code containing no
		 * codewords.  Note: the list of offset codeword lengths is
		 * always nonempty, but lengths of 0 don't count as codewords.
		 *
		 * The RFC doesn't say whether the same cases are allowed for
		 * the litlen and pre codes.  It's actually impossible for no
		 * symbols to be used from these codes; however, it's
		 * technically possible for only one symbol to be used.  zlib
		 * allows 1 codeword for the litlen code, but not the precode.
		 * The RFC also doesn't say whether, when there is 1 codeword,
		 * that codeword is '0' or '1'.  zlib uses '0'.
		 *
		 * We accept what zlib accepts, plus a bit more.  First, we
		 * don't treat the precode more strictly than the litlen and
		 * offset codes.  There's no convincing reason to add a special
		 * case for the precode here.
		 *
		 * Second, we just map each allowed incompete code to a complete
		 * code with only real symbols.  To do this, we choose a symbol,
		 * either the used symbol (for codes with 1 codeword) or an
		 * arbitrary symbol (for empty codes), and give it both
		 * codewords '0' and '1'.  zlib instead uses a special ERROR
		 * symbol in the part of the codespace the code doesn't use.
		 * However, having an ERROR symbol reduces the performance of
		 * the Huffman decoder, for no real benefit.  Our approach also
		 * avoids having to decide whether '0' or '1' is correct.
		 *
		 * Like zlib, we still reject all incomplete codes that contain
		 * more than 1 codeword or a codeword length greater than 1.
		 */
		if (codespace_used == 0) {
			sym = 0; /* arbitrary */
		} else {
			if (codespace_used != (1U << (max_codeword_len - 1)) ||
			    len_counts[1] != 1)
				return false;
			sym = sorted_syms[0];
		}
		entry = make_decode_table_entry(decode_results, sym, 1);
		for (i = 0; i < (1U << table_bits); i++)
			decode_table[i] = entry;
		return true;
	}

	/*
	 * The lengths form a complete code.  Now, enumerate the codewords in
	 * lexicographic order and fill the decode table entries for each one.
	 *
	 * First, process all codewords with len <= table_bits.  Each one gets
	 * '2^(table_bits-len)' direct entries in the table.
	 *
	 * Since DEFLATE uses bit-reversed codewords, these entries aren't
	 * consecutive but rather are spaced '2^len' entries apart.  This makes
	 * filling them naively somewhat awkward and inefficient, since strided
	 * stores are less cache-friendly and preclude the use of word or
	 * vector-at-a-time stores to fill multiple entries per instruction.
	 *
	 * To optimize this, we incrementally double the table size.  When
	 * processing codewords with length 'len', the table is treated as
	 * having only '2^len' entries, so each codeword uses just one entry.
	 * Then, each time 'len' is incremented, the table size is doubled and
	 * the first half is copied to the second half.  This significantly
	 * improves performance over naively doing strided stores.
	 *
	 * Note that some entries copied for each table doubling may not have
	 * been initialized yet, but it doesn't matter since they're guaranteed
	 * to be initialized later (because the Huffman code is complete).
	 */
	codeword = 0;
	len = 1;
	while ((count = len_counts[len]) == 0)
		len++;
	cur_table_end = 1U << len;
	while (len <= table_bits) {
		/* Process all 'count' codewords with length 'len' bits. */
		do {
			unsigned bit;

			/* Fill the first entry for the current codeword. */
			decode_table[codeword] =
				make_decode_table_entry(decode_results,
							*sorted_syms++, len);

			if (codeword == cur_table_end - 1) {
				/* Last codeword (all 1's) */
				for (; len < table_bits; len++) {
					memcpy(&decode_table[cur_table_end],
					       decode_table,
					       cur_table_end *
						sizeof(decode_table[0]));
					cur_table_end <<= 1;
				}
				return true;
			}
			/*
			 * To advance to the lexicographically next codeword in
			 * the canonical code, the codeword must be incremented,
			 * then 0's must be appended to the codeword as needed
			 * to match the next codeword's length.
			 *
			 * Since the codeword is bit-reversed, appending 0's is
			 * a no-op.  However, incrementing it is nontrivial.  To
			 * do so efficiently, use the 'bsr' instruction to find
			 * the last (highest order) 0 bit in the codeword, set
			 * it, and clear any later (higher order) 1 bits.  But
			 * 'bsr' actually finds the highest order 1 bit, so to
			 * use it first flip all bits in the codeword by XOR'ing
			 * it with (1U << len) - 1 == cur_table_end - 1.
			 */
			bit = 1U << bsr32(codeword ^ (cur_table_end - 1));
			codeword &= bit - 1;
			codeword |= bit;
		} while (--count);

		/* Advance to the next codeword length. */
		do {
			if (++len <= table_bits) {
				memcpy(&decode_table[cur_table_end],
				       decode_table,
				       cur_table_end * sizeof(decode_table[0]));
				cur_table_end <<= 1;
			}
		} while ((count = len_counts[len]) == 0);
	}

	/* Process codewords with len > table_bits.  These require subtables. */
	cur_table_end = 1U << table_bits;
	subtable_prefix = -1;
	subtable_start = 0;
	for (;;) {
		u32 entry;
		unsigned i;
		unsigned stride;
		unsigned bit;

		/*
		 * Start a new subtable if the first 'table_bits' bits of the
		 * codeword don't match the prefix of the current subtable.
		 */
		if ((codeword & ((1U << table_bits) - 1)) != subtable_prefix) {
			subtable_prefix = (codeword & ((1U << table_bits) - 1));
			subtable_start = cur_table_end;
			/*
			 * Calculate the subtable length.  If the codeword has
			 * length 'table_bits + n', then the subtable needs
			 * '2^n' entries.  But it may need more; if fewer than
			 * '2^n' codewords of length 'table_bits + n' remain,
			 * then the length will need to be incremented to bring
			 * in longer codewords until the subtable can be
			 * completely filled.  Note that because the Huffman
			 * code is complete, it will always be possible to fill
			 * the subtable eventually.
			 */
			subtable_bits = len - table_bits;
			codespace_used = count;
			while (codespace_used < (1U << subtable_bits)) {
				subtable_bits++;
				codespace_used = (codespace_used << 1) +
					len_counts[table_bits + subtable_bits];
			}
			cur_table_end = subtable_start + (1U << subtable_bits);

			/*
			 * Create the entry that points from the main table to
			 * the subtable.
			 */
			decode_table[subtable_prefix] =
				((u32)subtable_start << 16) |
				HUFFDEC_EXCEPTIONAL |
				HUFFDEC_SUBTABLE_POINTER |
				(subtable_bits << 8) | table_bits;
		}

		/* Fill the subtable entries for the current codeword. */
		entry = make_decode_table_entry(decode_results, *sorted_syms++,
						len - table_bits);
		i = subtable_start + (codeword >> table_bits);
		stride = 1U << (len - table_bits);
		do {
			decode_table[i] = entry;
			i += stride;
		} while (i < cur_table_end);

		/* Advance to the next codeword. */
		if (codeword == (1U << len) - 1) /* last codeword (all 1's)? */
			return true;
		bit = 1U << bsr32(codeword ^ ((1U << len) - 1));
		codeword &= bit - 1;
		codeword |= bit;
		count--;
		while (count == 0)
			count = len_counts[++len];
	}
}

/* Build the decode table for the precode.  */
static bool
build_precode_decode_table(struct libdeflate_decompressor *d)
{
	/* When you change TABLEBITS, you must change ENOUGH, and vice versa! */
	STATIC_ASSERT(PRECODE_TABLEBITS == 7 && PRECODE_ENOUGH == 128);

	STATIC_ASSERT(ARRAY_LEN(precode_decode_results) ==
		      DEFLATE_NUM_PRECODE_SYMS);

	return build_decode_table(d->u.l.precode_decode_table,
				  d->u.precode_lens,
				  DEFLATE_NUM_PRECODE_SYMS,
				  precode_decode_results,
				  PRECODE_TABLEBITS,
				  DEFLATE_MAX_PRE_CODEWORD_LEN,
				  d->sorted_syms,
				  NULL);
}

/* Build the decode table for the literal/length code.  */
static bool
build_litlen_decode_table(struct libdeflate_decompressor *d,
			  unsigned num_litlen_syms, unsigned num_offset_syms)
{
	/* When you change TABLEBITS, you must change ENOUGH, and vice versa! */
	STATIC_ASSERT(LITLEN_TABLEBITS == 11 && LITLEN_ENOUGH == 2342);

	STATIC_ASSERT(ARRAY_LEN(litlen_decode_results) ==
		      DEFLATE_NUM_LITLEN_SYMS);

	return build_decode_table(d->u.litlen_decode_table,
				  d->u.l.lens,
				  num_litlen_syms,
				  litlen_decode_results,
				  LITLEN_TABLEBITS,
				  DEFLATE_MAX_LITLEN_CODEWORD_LEN,
				  d->sorted_syms,
				  &d->litlen_tablebits);
}

/* Build the decode table for the offset code.  */
static bool
build_offset_decode_table(struct libdeflate_decompressor *d,
			  unsigned num_litlen_syms, unsigned num_offset_syms)
{
	/* When you change TABLEBITS, you must change ENOUGH, and vice versa! */
	STATIC_ASSERT(OFFSET_TABLEBITS == 8 && OFFSET_ENOUGH == 402);

	STATIC_ASSERT(ARRAY_LEN(offset_decode_results) ==
		      DEFLATE_NUM_OFFSET_SYMS);

	return build_decode_table(d->offset_decode_table,
				  d->u.l.lens + num_litlen_syms,
				  num_offset_syms,
				  offset_decode_results,
				  OFFSET_TABLEBITS,
				  DEFLATE_MAX_OFFSET_CODEWORD_LEN,
				  d->sorted_syms,
				  NULL);
}

/*****************************************************************************
 *                         Main decompression routine
 *****************************************************************************/

typedef enum libdeflate_result (*decompress_func_t)
	(struct libdeflate_decompressor * restrict d,
	 const void * restrict in, size_t in_nbytes,
	 void * restrict out, size_t out_nbytes_avail,
	 size_t *actual_in_nbytes_ret, size_t *actual_out_nbytes_ret);

#define FUNCNAME deflate_decompress_default
#undef ATTRIBUTES
#undef EXTRACT_VARBITS
#undef EXTRACT_VARBITS8
#include "decompress_template.h"

/* Include architecture-specific implementation(s) if available. */
#undef DEFAULT_IMPL
#undef arch_select_decompress_func
#if defined(ARCH_X86_32) || defined(ARCH_X86_64)
#  include "x86/decompress_impl.h"
#endif

#ifndef DEFAULT_IMPL
#  define DEFAULT_IMPL deflate_decompress_default
#endif

#ifdef arch_select_decompress_func
static enum libdeflate_result
dispatch_decomp(struct libdeflate_decompressor *d,
		const void *in, size_t in_nbytes,
		void *out, size_t out_nbytes_avail,
		size_t *actual_in_nbytes_ret, size_t *actual_out_nbytes_ret);

static volatile decompress_func_t decompress_impl = dispatch_decomp;

/* Choose the best implementation at runtime. */
static enum libdeflate_result
dispatch_decomp(struct libdeflate_decompressor *d,
		const void *in, size_t in_nbytes,
		void *out, size_t out_nbytes_avail,
		size_t *actual_in_nbytes_ret, size_t *actual_out_nbytes_ret)
{
	decompress_func_t f = arch_select_decompress_func();

	if (f == NULL)
		f = DEFAULT_IMPL;

	decompress_impl = f;
	return f(d, in, in_nbytes, out, out_nbytes_avail,
		 actual_in_nbytes_ret, actual_out_nbytes_ret);
}
#else
/* The best implementation is statically known, so call it directly. */
#  define decompress_impl DEFAULT_IMPL
#endif

/*
 * This is the main DEFLATE decompression routine.  See libdeflate.h for the
 * documentation.
 *
 * Note that the real code is in decompress_template.h.  The part here just
 * handles calling the appropriate implementation depending on the CPU features
 * at runtime.
 */
LIBDEFLATEAPI enum libdeflate_result
libdeflate_deflate_decompress_ex(struct libdeflate_decompressor *d,
				 const void *in, size_t in_nbytes,
				 void *out, size_t out_nbytes_avail,
				 size_t *actual_in_nbytes_ret,
				 size_t *actual_out_nbytes_ret)
{
	return decompress_impl(d, in, in_nbytes, out, out_nbytes_avail,
			       actual_in_nbytes_ret, actual_out_nbytes_ret);
}

LIBDEFLATEAPI enum libdeflate_result
libdeflate_deflate_decompress(struct libdeflate_decompressor *d,
			      const void *in, size_t in_nbytes,
			      void *out, size_t out_nbytes_avail,
			      size_t *actual_out_nbytes_ret)
{
	return libdeflate_deflate_decompress_ex(d, in, in_nbytes,
						out, out_nbytes_avail,
						NULL, actual_out_nbytes_ret);
}

LIBDEFLATEAPI struct libdeflate_decompressor *
libdeflate_alloc_decompressor_ex(const struct libdeflate_options *options)
{
	struct libdeflate_decompressor *d;

	/*
	 * Note: if more fields are added to libdeflate_options, this code will
	 * need to be updated to support both the old and new structs.
	 */
	if (options->sizeof_options != sizeof(*options))
		return NULL;

	d = (options->malloc_func ? options->malloc_func :
	     libdeflate_default_malloc_func)(sizeof(*d));
	if (d == NULL)
		return NULL;
	/*
	 * Note that only certain parts of the decompressor actually must be
	 * initialized here:
	 *
	 * - 'static_codes_loaded' must be initialized to false.
	 *
	 * - The first half of the main portion of each decode table must be
	 *   initialized to any value, to avoid reading from uninitialized
	 *   memory during table expansion in build_decode_table().  (Although,
	 *   this is really just to avoid warnings with dynamic tools like
	 *   valgrind, since build_decode_table() is guaranteed to initialize
	 *   all entries eventually anyway.)
	 *
	 * - 'free_func' must be set.
	 *
	 * But for simplicity, we currently just zero the whole decompressor.
	 */
	memset(d, 0, sizeof(*d));
	d->free_func = options->free_func ?
		       options->free_func : libdeflate_default_free_func;
	return d;
}

LIBDEFLATEAPI struct libdeflate_decompressor *
libdeflate_alloc_decompressor(void)
{
	static const struct libdeflate_options defaults = {
		.sizeof_options = sizeof(defaults),
	};
	return libdeflate_alloc_decompressor_ex(&defaults);
}

LIBDEFLATEAPI void
libdeflate_free_decompressor(struct libdeflate_decompressor *d)
{
	if (d)
		d->free_func(d);
}
