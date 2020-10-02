/*
 * Copyright © 2018  Google, Inc.
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 * Google Author(s): Garret Rieger
 */

#include "hb.hh"
#include "hb-ot-os2-unicode-ranges.hh"

static void
test (hb_codepoint_t cp, unsigned int bit)
{
  if (OT::_hb_ot_os2_get_unicode_range_bit (cp) != bit)
  {
    fprintf (stderr, "got incorrect bit (%d) for cp 0x%X. Should have been %d.",
	     OT::_hb_ot_os2_get_unicode_range_bit (cp),
	     cp,
	     bit);
    abort();
  }
}

static void
test_get_unicode_range_bit ()
{
  test (0x0000, 0);
  test (0x0042, 0);
  test (0x007F, 0);
  test (0x0080, 1);

  test (0x30A0, 50);
  test (0x30B1, 50);
  test (0x30FF, 50);

  test (0x10FFFD, 90);

  test (0x30000, -1);
  test (0x110000, -1);
}

int
main ()
{
  test_get_unicode_range_bit ();
  return 0;
}
