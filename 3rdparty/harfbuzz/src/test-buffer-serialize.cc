/*
 * Copyright © 2010,2011,2013  Google, Inc.
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
 * Google Author(s): Behdad Esfahbod
 */

#include "hb.hh"

#include "hb.h"
#include "hb-ot.h"
#ifdef HAVE_FREETYPE
#include "hb-ft.h"
#endif

#include <stdio.h>

#ifdef HB_NO_OPEN
#define hb_blob_create_from_file(x)  hb_blob_get_empty ()
#endif

int
main (int argc, char **argv)
{
  bool ret = true;

#ifndef HB_NO_BUFFER_SERIALIZE

  if (argc != 2) {
    fprintf (stderr, "usage: %s font-file\n", argv[0]);
    exit (1);
  }

  hb_blob_t *blob = hb_blob_create_from_file (argv[1]);
  hb_face_t *face = hb_face_create (blob, 0 /* first face */);
  hb_blob_destroy (blob);
  blob = nullptr;

  unsigned int upem = hb_face_get_upem (face);
  hb_font_t *font = hb_font_create (face);
  hb_face_destroy (face);
  hb_font_set_scale (font, upem, upem);
  hb_ot_font_set_funcs (font);
#ifdef HAVE_FREETYPE
  //hb_ft_font_set_funcs (font);
#endif

  hb_buffer_t *buf;
  buf = hb_buffer_create ();

  char line[BUFSIZ], out[BUFSIZ];
  while (fgets (line, sizeof(line), stdin))
  {
    hb_buffer_clear_contents (buf);

    const char *p = line;
    while (hb_buffer_deserialize_glyphs (buf,
					 p, -1, &p,
					 font,
					 HB_BUFFER_SERIALIZE_FORMAT_JSON))
      ;
    if (*p && *p != '\n')
      ret = false;

    hb_buffer_serialize_glyphs (buf, 0, hb_buffer_get_length (buf),
				out, sizeof (out), nullptr,
				font, HB_BUFFER_SERIALIZE_FORMAT_JSON,
				HB_BUFFER_SERIALIZE_FLAG_DEFAULT);
    puts (out);
  }

  hb_buffer_destroy (buf);

  hb_font_destroy (font);

#endif

  return !ret;
}
