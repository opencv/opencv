/*
 * Copyright © 2019  Adobe, Inc.
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
 * Adobe Author(s): Michiharu Ariza
 */

#include "hb.hh"
#include "hb-bimap.hh"

int
main (int argc, char **argv)
{
  hb_bimap_t	bm;

  assert (bm.is_empty () == true);
  bm.set (1, 4);
  bm.set (2, 5);
  bm.set (3, 6);
  assert (bm.get_population () == 3);
  assert (bm.has (1) == true);
  assert (bm.has (4) == false);
  assert (bm[2] == 5);
  assert (bm.backward (6) == 3);
  bm.del (1);
  assert (bm.has (1) == false);
  assert (bm.has (3) == true);
  bm.clear ();
  assert (bm.get_population () == 0);

  hb_inc_bimap_t  ibm;

  assert (ibm.add (13) == 0);
  assert (ibm.add (8) == 1);
  assert (ibm.add (10) == 2);
  assert (ibm.add (8) == 1);
  assert (ibm.add (7) == 3);
  assert (ibm.get_population () == 4);
  assert (ibm[7] == 3);

  ibm.sort ();
  assert (ibm.get_population () == 4);
  assert (ibm[7] == 0);
  assert (ibm[13] == 3);

  ibm.identity (3);
  assert (ibm.get_population () == 3);
  assert (ibm[0] == 0);
  assert (ibm[1] == 1);
  assert (ibm[2] == 2);
  assert (ibm.backward (0) == 0);
  assert (ibm.backward (1) == 1);
  assert (ibm.backward (2) == 2);
  assert (ibm.has (4) == false);

  return 0;
}
