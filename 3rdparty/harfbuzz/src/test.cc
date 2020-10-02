/*
 * Copyright © 2010,2011  Google, Inc.
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

#include <stdio.h>

#ifdef HAVE_FREETYPE
#include "hb-ft.h"
#endif

#ifdef HB_NO_OPEN
#define hb_blob_create_from_file(x)  hb_blob_get_empty ()
#endif

int
main (int argc, char **argv)
{
  if (argc != 2) {
    fprintf (stderr, "usage: %s font-file.ttf\n", argv[0]);
    exit (1);
  }

  hb_blob_t *blob = hb_blob_create_from_file (argv[1]);
  printf ("Opened font file %s: %u bytes long\n", argv[1], hb_blob_get_length (blob));

  /* Create the face */
  hb_face_t *face = hb_face_create (blob, 0 /* first face */);
  hb_blob_destroy (blob);
  blob = nullptr;
  unsigned int upem = hb_face_get_upem (face);

  hb_font_t *font = hb_font_create (face);
  hb_font_set_scale (font, upem, upem);

#ifdef HAVE_FREETYPE
  hb_ft_font_set_funcs (font);
#endif

  hb_buffer_t *buffer = hb_buffer_create ();

  hb_buffer_add_utf8 (buffer, "\xe0\xa4\x95\xe0\xa5\x8d\xe0\xa4\xb0\xe0\xa5\x8d\xe0\xa4\x95", -1, 0, -1);
  hb_buffer_guess_segment_properties (buffer);

  hb_shape (font, buffer, nullptr, 0);

  unsigned int count = hb_buffer_get_length (buffer);
  hb_glyph_info_t *infos = hb_buffer_get_glyph_infos (buffer, nullptr);
  hb_glyph_position_t *positions = hb_buffer_get_glyph_positions (buffer, nullptr);

  for (unsigned int i = 0; i < count; i++)
  {
    hb_glyph_info_t *info = &infos[i];
    hb_glyph_position_t *pos = &positions[i];

    printf ("cluster %d	glyph 0x%x at	(%d,%d)+(%d,%d)\n",
	    info->cluster,
	    info->codepoint,
	    pos->x_offset,
	    pos->y_offset,
	    pos->x_advance,
	    pos->y_advance);

  }

  hb_buffer_destroy (buffer);
  hb_font_destroy (font);
  hb_face_destroy (face);

  return 0;
}


