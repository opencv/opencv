/*
 * jdhuff.c
 *
 * Copyright (C) 1991-1998, Thomas G. Lane.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains Huffman entropy decoding routines which are shared
 * by the sequential, progressive and lossless decoders.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jlossy.h"    /* Private declarations for lossy codec */
#include "jlossls.h"    /* Private declarations for lossless codec */
#include "jdhuff.h"    /* Declarations shared with jd*huff.c */


/*
 * Compute the derived values for a Huffman table.
 * This routine also performs some validation checks on the table.
 */

GLOBAL(void)
jpeg_make_d_derived_tbl (j_decompress_ptr cinfo, boolean isDC, int tblno,
       d_derived_tbl ** pdtbl)
{
  JHUFF_TBL *htbl;
  d_derived_tbl *dtbl;
  int p, i, l, si, numsymbols;
  int lookbits, ctr;
  char huffsize[257];
  unsigned int huffcode[257];
  unsigned int code;

  /* Note that huffsize[] and huffcode[] are filled in code-length order,
   * paralleling the order of the symbols themselves in htbl->huffval[].
   */

  /* Find the input Huffman table */
  if (tblno < 0 || tblno >= NUM_HUFF_TBLS)
    ERREXIT1(cinfo, JERR_NO_HUFF_TABLE, tblno);
  htbl =
    isDC ? cinfo->dc_huff_tbl_ptrs[tblno] : cinfo->ac_huff_tbl_ptrs[tblno];
  if (htbl == NULL)
    ERREXIT1(cinfo, JERR_NO_HUFF_TABLE, tblno);

  /* Allocate a workspace if we haven't already done so. */
  if (*pdtbl == NULL)
    *pdtbl = (d_derived_tbl *)
      (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_IMAGE,
          SIZEOF(d_derived_tbl));
  dtbl = *pdtbl;
  dtbl->pub = htbl;    /* fill in back link */

  /* Figure C.1: make table of Huffman code length for each symbol */

  p = 0;
  for (l = 1; l <= 16; l++) {
    i = (int) htbl->bits[l];
    if (i < 0 || p + i > 256)  /* protect against table overrun */
      ERREXIT(cinfo, JERR_BAD_HUFF_TABLE);
    while (i--)
      huffsize[p++] = (char) l;
  }
  huffsize[p] = 0;
  numsymbols = p;

  /* Figure C.2: generate the codes themselves */
  /* We also validate that the counts represent a legal Huffman code tree. */

  code = 0;
  si = huffsize[0];
  p = 0;
  while (huffsize[p]) {
    while (((int) huffsize[p]) == si) {
      huffcode[p++] = code;
      code++;
    }
    /* code is now 1 more than the last code used for codelength si; but
     * it must still fit in si bits, since no code is allowed to be all ones.
     * BUG FIX 2001-09-03: Comparison must be >, not >=
     */
    if (((INT32) code) > (((INT32) 1) << si))
      ERREXIT(cinfo, JERR_BAD_HUFF_TABLE);
    code <<= 1;
    si++;
  }

  /* Figure F.15: generate decoding tables for bit-sequential decoding */

  p = 0;
  for (l = 1; l <= 16; l++) {
    if (htbl->bits[l]) {
      /* valoffset[l] = huffval[] index of 1st symbol of code length l,
       * minus the minimum code of length l
       */
      dtbl->valoffset[l] = (INT32) p - (INT32) huffcode[p];
      p += htbl->bits[l];
      dtbl->maxcode[l] = huffcode[p-1]; /* maximum code of length l */
    } else {
      dtbl->maxcode[l] = -1;  /* -1 if no codes of this length */
    }
  }
  dtbl->maxcode[17] = 0xFFFFFL; /* ensures jpeg_huff_decode terminates */

  /* Compute lookahead tables to speed up decoding.
   * First we set all the table entries to 0, indicating "too long";
   * then we iterate through the Huffman codes that are short enough and
   * fill in all the entries that correspond to bit sequences starting
   * with that code.
   */

  MEMZERO(dtbl->look_nbits, SIZEOF(dtbl->look_nbits));

  p = 0;
  for (l = 1; l <= HUFF_LOOKAHEAD; l++) {
    for (i = 1; i <= (int) htbl->bits[l]; i++, p++) {
      /* l = current code's length, p = its index in huffcode[] & huffval[]. */
      /* Generate left-justified code followed by all possible bit sequences */
      lookbits = huffcode[p] << (HUFF_LOOKAHEAD-l);
      for (ctr = 1 << (HUFF_LOOKAHEAD-l); ctr > 0; ctr--) {
  dtbl->look_nbits[lookbits] = l;
  dtbl->look_sym[lookbits] = htbl->huffval[p];
  lookbits++;
      }
    }
  }

  /* Validate symbols as being reasonable.
   * For AC tables, we make no check, but accept all byte values 0..255.
   * For DC tables, we require the symbols to be in range 0..16.
   * (Tighter bounds could be applied depending on the data depth and mode,
   * but this is sufficient to ensure safe decoding.)
   */
  if (isDC) {
    for (i = 0; i < numsymbols; i++) {
      int sym = htbl->huffval[i];
/* The following file contains a value of 17 in the huffman table, which is impossible
 * according to ISO 10918-1, H.1.2.2 Huffman coding of the modulo difference
 * and table H.2.
 * PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm
 * MM, 2008/08/12 I am breaking backward compatibility and decide not to support this image
 * anymore. In fact the decompression using another library: PVRG was giving me
 * another result anyway.
 *
 * Steps:
 *
 * $ gdcmconv --raw -i PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm -o bla.dcm
 * $ gdcmraw -i bla.dcm -o bla.raw
 * $ gdcmraw -i PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm -o philips.jpg
 * $ pvrgjpeg -d philips
 * $ dd conv=swab if=philips.jpg.0 of=philips.raw
 * $ vbindiff bla.raw philips.raw
 *
 * $ md5sum bla.raw philips.raw
 * 4b0021efe5a675f24c82e1ff28a1e2eb  bla.raw
 * d93d2f78d845c7a132489aab92eadd32  philips.raw
 *
 *
 * footnote, I get a much closer result doing:
 * $ pvrgjpeg -a -y -d philips
 * after that the number of diff with IJG is getting lower
 *
 * 0f2570f5d91ddea5bd9d3825a86eaabe  philips.raw
 */
      if (sym < 0 || sym > 16)
  ERREXIT(cinfo, JERR_BAD_HUFF_TABLE);
    }
  }
}


/*
 * Out-of-line code for bit fetching.
 * See jdhuff.h for info about usage.
 * Note: current values of get_buffer and bits_left are passed as parameters,
 * but are returned in the corresponding fields of the state struct.
 *
 * On most machines MIN_GET_BITS should be 25 to allow the full 32-bit width
 * of get_buffer to be used.  (On machines with wider words, an even larger
 * buffer could be used.)  However, on some machines 32-bit shifts are
 * quite slow and take time proportional to the number of places shifted.
 * (This is true with most PC compilers, for instance.)  In this case it may
 * be a win to set MIN_GET_BITS to the minimum value of 15.  This reduces the
 * average shift distance at the cost of more calls to jpeg_fill_bit_buffer.
 */

#ifdef SLOW_SHIFT_32
#define MIN_GET_BITS  15  /* minimum allowable value */
#else
#define MIN_GET_BITS  (BIT_BUF_SIZE-7)
#endif


GLOBAL(boolean)
jpeg_fill_bit_buffer (bitread_working_state * state,
          register bit_buf_type get_buffer, register int bits_left,
          int nbits)
/* Load up the bit buffer to a depth of at least nbits */
{
  /* Copy heavily used state fields into locals (hopefully registers) */
  register const JOCTET * next_input_byte = state->next_input_byte;
  register size_t bytes_in_buffer = state->bytes_in_buffer;
  j_decompress_ptr cinfo = state->cinfo;

  /* Attempt to load at least MIN_GET_BITS bits into get_buffer. */
  /* (It is assumed that no request will be for more than that many bits.) */
  /* We fail to do so only if we hit a marker or are forced to suspend. */

  if (cinfo->unread_marker == 0) {  /* cannot advance past a marker */
    while (bits_left < MIN_GET_BITS) {
      register int c;

      /* Attempt to read a byte */
      if (bytes_in_buffer == 0) {
  if (! (*cinfo->src->fill_input_buffer) (cinfo))
    return FALSE;
  next_input_byte = cinfo->src->next_input_byte;
  bytes_in_buffer = cinfo->src->bytes_in_buffer;
      }
      bytes_in_buffer--;
      c = GETJOCTET(*next_input_byte++);

      /* If it's 0xFF, check and discard stuffed zero byte */
      if (c == 0xFF) {
  /* Loop here to discard any padding FF's on terminating marker,
   * so that we can save a valid unread_marker value.  NOTE: we will
   * accept multiple FF's followed by a 0 as meaning a single FF data
   * byte.  This data pattern is not valid according to the standard.
   */
  do {
    if (bytes_in_buffer == 0) {
      if (! (*cinfo->src->fill_input_buffer) (cinfo))
        return FALSE;
      next_input_byte = cinfo->src->next_input_byte;
      bytes_in_buffer = cinfo->src->bytes_in_buffer;
    }
    bytes_in_buffer--;
    c = GETJOCTET(*next_input_byte++);
  } while (c == 0xFF);

  if (c == 0) {
    /* Found FF/00, which represents an FF data byte */
    c = 0xFF;
  } else {
    /* Oops, it's actually a marker indicating end of compressed data.
     * Save the marker code for later use.
     * Fine point: it might appear that we should save the marker into
     * bitread working state, not straight into permanent state.  But
     * once we have hit a marker, we cannot need to suspend within the
     * current MCU, because we will read no more bytes from the data
     * source.  So it is OK to update permanent state right away.
     */
    cinfo->unread_marker = c;
    /* See if we need to insert some fake zero bits. */
    goto no_more_bytes;
  }
      }

      /* OK, load c into get_buffer */
      get_buffer = (get_buffer << 8) | c;
      bits_left += 8;
    } /* end while */
  } else {
  no_more_bytes:
    /* We get here if we've read the marker that terminates the compressed
     * data segment.  There should be enough bits in the buffer register
     * to satisfy the request; if so, no problem.
     */
    if (nbits > bits_left) {
      /* Uh-oh.  Report corrupted data to user and stuff zeroes into
       * the data stream, so that we can produce some kind of image.
       * We use a nonvolatile flag to ensure that only one warning message
       * appears per data segment.
       */
      huffd_common_ptr huffd;
      if (cinfo->process == JPROC_LOSSLESS)
  huffd = (huffd_common_ptr) ((j_lossless_d_ptr) cinfo->codec)->entropy_private;
      else
  huffd = (huffd_common_ptr) ((j_lossy_d_ptr) cinfo->codec)->entropy_private;
      if (! huffd->insufficient_data) {
  WARNMS(cinfo, JWRN_HIT_MARKER);
  huffd->insufficient_data = TRUE;
      }
      /* Fill the buffer with zero bits */
      get_buffer <<= MIN_GET_BITS - bits_left;
      bits_left = MIN_GET_BITS;
    }
  }

  /* Unload the local registers */
  state->next_input_byte = next_input_byte;
  state->bytes_in_buffer = bytes_in_buffer;
  state->get_buffer = get_buffer;
  state->bits_left = bits_left;

  return TRUE;
}


/*
 * Out-of-line code for Huffman code decoding.
 * See jdhuff.h for info about usage.
 */

GLOBAL(int)
jpeg_huff_decode (bitread_working_state * state,
      register bit_buf_type get_buffer, register int bits_left,
      d_derived_tbl * htbl, int min_bits)
{
  register int l = min_bits;
  register INT32 code;

  /* HUFF_DECODE has determined that the code is at least min_bits */
  /* bits long, so fetch that many bits in one swoop. */

  CHECK_BIT_BUFFER(*state, l, return -1);
  code = GET_BITS(l);

  /* Collect the rest of the Huffman code one bit at a time. */
  /* This is per Figure F.16 in the JPEG spec. */

  while (code > htbl->maxcode[l]) {
    code <<= 1;
    CHECK_BIT_BUFFER(*state, 1, return -1);
    code |= GET_BITS(1);
    l++;
  }

  /* Unload the local registers */
  state->get_buffer = get_buffer;
  state->bits_left = bits_left;

  /* With garbage input we may reach the sentinel value l = 17. */

  if (l > 16) {
    WARNMS(state->cinfo, JWRN_HUFF_BAD_CODE);
    return 0;      /* fake a zero as the safest result */
  }

  return htbl->pub->huffval[ (int) (code + htbl->valoffset[l]) ];
}
