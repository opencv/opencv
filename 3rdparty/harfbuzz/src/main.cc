/*
 * Copyright © 2007,2008,2009  Red Hat, Inc.
 * Copyright © 2018,2019,2020  Ebrahim Byagowi
 * Copyright © 2018  Khaled Hosny
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
 * Red Hat Author(s): Behdad Esfahbod
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "hb.h"
#include "hb-ot.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef HB_NO_OPEN
#define hb_blob_create_from_file(x)  hb_blob_get_empty ()
#endif

#if !defined(HB_NO_COLOR) && !defined(HB_NO_DRAW) && defined(HB_EXPERIMENTAL_API)
static void
svg_dump (hb_face_t *face, unsigned face_index)
{
  unsigned glyph_count = hb_face_get_glyph_count (face);

  for (unsigned glyph_id = 0; glyph_id < glyph_count; ++glyph_id)
  {
    hb_blob_t *blob = hb_ot_color_glyph_reference_svg (face, glyph_id);

    if (hb_blob_get_length (blob) == 0) continue;

    unsigned length;
    const char *data = hb_blob_get_data (blob, &length);

    char output_path[255];
    sprintf (output_path, "out/svg-%u-%u.svg%s",
	     glyph_id,
	     face_index,
	     // append "z" if the content is gzipped, https://stackoverflow.com/a/6059405
	     (length > 2 && (data[0] == '\x1F') && (data[1] == '\x8B')) ? "z" : "");

    FILE *f = fopen (output_path, "wb");
    fwrite (data, 1, length, f);
    fclose (f);

    hb_blob_destroy (blob);
  }
}

/* _png API is so easy to use unlike the below code, don't get confused */
static void
png_dump (hb_face_t *face, unsigned face_index)
{
  unsigned glyph_count = hb_face_get_glyph_count (face);
  hb_font_t *font = hb_font_create (face);

  /* scans the font for strikes */
  unsigned sample_glyph_id;
  /* we don't care about different strikes for different glyphs at this point */
  for (sample_glyph_id = 0; sample_glyph_id < glyph_count; ++sample_glyph_id)
  {
    hb_blob_t *blob = hb_ot_color_glyph_reference_png (font, sample_glyph_id);
    unsigned blob_length = hb_blob_get_length (blob);
    hb_blob_destroy (blob);
    if (blob_length != 0)
      break;
  }

  unsigned upem = hb_face_get_upem (face);
  unsigned blob_length = 0;
  unsigned strike = 0;
  for (unsigned ppem = 1; ppem < upem; ++ppem)
  {
    hb_font_set_ppem (font, ppem, ppem);
    hb_blob_t *blob = hb_ot_color_glyph_reference_png (font, sample_glyph_id);
    unsigned new_blob_length = hb_blob_get_length (blob);
    hb_blob_destroy (blob);
    if (new_blob_length != blob_length)
    {
      for (unsigned glyph_id = 0; glyph_id < glyph_count; ++glyph_id)
      {
	hb_blob_t *blob = hb_ot_color_glyph_reference_png (font, glyph_id);

	if (hb_blob_get_length (blob) == 0) continue;

	unsigned length;
	const char *data = hb_blob_get_data (blob, &length);

	char output_path[255];
	sprintf (output_path, "out/png-%u-%u-%u.png", glyph_id, strike, face_index);

	FILE *f = fopen (output_path, "wb");
	fwrite (data, 1, length, f);
	fclose (f);

	hb_blob_destroy (blob);
      }

      strike++;
      blob_length = new_blob_length;
    }
  }

  hb_font_destroy (font);
}

struct user_data_t
{
  FILE *f;
  hb_position_t ascender;
};

static void
move_to (hb_position_t to_x, hb_position_t to_y, user_data_t &user_data)
{
  fprintf (user_data.f, "M%d,%d", to_x, user_data.ascender - to_y);
}

static void
line_to (hb_position_t to_x, hb_position_t to_y, user_data_t &user_data)
{
  fprintf (user_data.f, "L%d,%d", to_x, user_data.ascender - to_y);
}

static void
quadratic_to (hb_position_t control_x, hb_position_t control_y,
	      hb_position_t to_x, hb_position_t to_y,
	      user_data_t &user_data)
{
  fprintf (user_data.f, "Q%d,%d %d,%d", control_x, user_data.ascender - control_y,
					to_x, user_data.ascender - to_y);
}

static void
cubic_to (hb_position_t control1_x, hb_position_t control1_y,
	  hb_position_t control2_x, hb_position_t control2_y,
	  hb_position_t to_x, hb_position_t to_y,
	  user_data_t &user_data)
{
  fprintf (user_data.f, "C%d,%d %d,%d %d,%d", control1_x, user_data.ascender - control1_y,
					       control2_x, user_data.ascender - control2_y,
					       to_x, user_data.ascender - to_y);
}

static void
close_path (user_data_t &user_data)
{
  fprintf (user_data.f, "Z");
}

static void
layered_glyph_dump (hb_font_t *font, hb_draw_funcs_t *funcs, unsigned face_index)
{
  hb_face_t *face = hb_font_get_face (font);
  unsigned palette_count = hb_ot_color_palette_get_count (face);
  for (unsigned palette = 0; palette < palette_count; ++palette)
  {
    unsigned num_colors = hb_ot_color_palette_get_colors (face, palette, 0, nullptr, nullptr);
    if (!num_colors) continue;

    hb_color_t *colors = (hb_color_t*) calloc (num_colors, sizeof (hb_color_t));
    hb_ot_color_palette_get_colors (face, palette, 0, &num_colors, colors);
    if (!num_colors)
    {
      free (colors);
      continue;
    }

    unsigned num_glyphs = hb_face_get_glyph_count (face);
    for (hb_codepoint_t gid = 0; gid < num_glyphs; ++gid)
    {
      unsigned num_layers = hb_ot_color_glyph_get_layers (face, gid, 0, nullptr, nullptr);
      if (!num_layers) continue;

      hb_ot_color_layer_t *layers = (hb_ot_color_layer_t*) malloc (num_layers * sizeof (hb_ot_color_layer_t));

      hb_ot_color_glyph_get_layers (face, gid, 0, &num_layers, layers);
      if (num_layers)
      {
	hb_font_extents_t font_extents;
	hb_font_get_extents_for_direction (font, HB_DIRECTION_LTR, &font_extents);
	hb_glyph_extents_t extents = {0};
	if (!hb_font_get_glyph_extents (font, gid, &extents))
	{
	  printf ("Skip gid: %d\n", gid);
	  continue;
	}

	char output_path[255];
	sprintf (output_path, "out/colr-%u-%u-%u.svg", gid, palette, face_index);
	FILE *f = fopen (output_path, "wb");
	fprintf (f, "<svg xmlns=\"http://www.w3.org/2000/svg\""
		    " viewBox=\"%d %d %d %d\">\n",
		    extents.x_bearing, 0,
		    extents.x_bearing + extents.width, -extents.height);
	user_data_t user_data;
	user_data.ascender = extents.y_bearing;
	user_data.f = f;

	for (unsigned layer = 0; layer < num_layers; ++layer)
	{
	  hb_color_t color = 0x000000FF;
	  if (layers[layer].color_index != 0xFFFF)
	    color = colors[layers[layer].color_index];
	  fprintf (f, "<path fill=\"#%02X%02X%02X\" ",
		   hb_color_get_red (color), hb_color_get_green (color), hb_color_get_green (color));
	  if (hb_color_get_alpha (color) != 255)
	    fprintf (f, "fill-opacity=\"%.3f\"", (double) hb_color_get_alpha (color) / 255.);
	  fprintf (f, "d=\"");
	  if (!hb_font_draw_glyph (font, layers[layer].glyph, funcs, &user_data))
	    printf ("Failed to decompose layer %d while %d\n", layers[layer].glyph, gid);
	  fprintf (f, "\"/>\n");
	}

	fprintf (f, "</svg>");
	fclose (f);
      }
      free (layers);
    }

    free (colors);
  }
}

static void
dump_glyphs (hb_font_t *font, hb_draw_funcs_t *funcs, unsigned face_index)
{
  unsigned num_glyphs = hb_face_get_glyph_count (hb_font_get_face (font));
  for (unsigned gid = 0; gid < num_glyphs; ++gid)
  {
    hb_font_extents_t font_extents;
    hb_font_get_extents_for_direction (font, HB_DIRECTION_LTR, &font_extents);
    hb_glyph_extents_t extents = {0};
    if (!hb_font_get_glyph_extents (font, gid, &extents))
    {
      printf ("Skip gid: %d\n", gid);
      continue;
    }

    char output_path[255];
    sprintf (output_path, "out/%u-%u.svg", face_index, gid);
    FILE *f = fopen (output_path, "wb");
    fprintf (f, "<svg xmlns=\"http://www.w3.org/2000/svg\""
		" viewBox=\"%d %d %d %d\"><path d=\"",
		extents.x_bearing, 0,
		extents.x_bearing + extents.width, font_extents.ascender - font_extents.descender);
    user_data_t user_data;
    user_data.ascender = font_extents.ascender;
    user_data.f = f;
    if (!hb_font_draw_glyph (font, gid, funcs, &user_data))
      printf ("Failed to decompose gid: %d\n", gid);
    fprintf (f, "\"/></svg>");
    fclose (f);
  }
}

static void
dump_glyphs (hb_blob_t *blob, const char *font_name)
{
  FILE *font_name_file = fopen ("out/.dumped_font_name", "r");
  if (font_name_file)
  {
    fprintf (stderr, "Purge or rename ./out folder if you like to run a glyph dump,\n"
		     "run it like `rm -rf out && mkdir out && src/main font-file.ttf`\n");
    return;
  }

  font_name_file = fopen ("out/.dumped_font_name", "w");
  if (!font_name_file)
  {
    fprintf (stderr, "./out is not accessible as a folder, create it please\n");
    return;
  }
  fwrite (font_name, 1, strlen (font_name), font_name_file);
  fclose (font_name_file);

  hb_draw_funcs_t *funcs = hb_draw_funcs_create ();
  hb_draw_funcs_set_move_to_func (funcs, (hb_draw_move_to_func_t) move_to);
  hb_draw_funcs_set_line_to_func (funcs, (hb_draw_line_to_func_t) line_to);
  hb_draw_funcs_set_quadratic_to_func (funcs, (hb_draw_quadratic_to_func_t) quadratic_to);
  hb_draw_funcs_set_cubic_to_func (funcs, (hb_draw_cubic_to_func_t) cubic_to);
  hb_draw_funcs_set_close_path_func (funcs, (hb_draw_close_path_func_t) close_path);

  unsigned num_faces = hb_face_count (blob);
  for (unsigned face_index = 0; face_index < num_faces; ++face_index)
  {
    hb_face_t *face = hb_face_create (blob, face_index);
    hb_font_t *font = hb_font_create (face);

    if (hb_ot_color_has_png (face))
      printf ("Dumping png (CBDT/sbix)...\n");
    png_dump (face, face_index);

    if (hb_ot_color_has_svg (face))
      printf ("Dumping svg (SVG )...\n");
    svg_dump (face, face_index);

    if (hb_ot_color_has_layers (face) && hb_ot_color_has_palettes (face))
      printf ("Dumping layered color glyphs (COLR/CPAL)...\n");
    layered_glyph_dump (font, funcs, face_index);

    dump_glyphs (font, funcs, face_index);

    hb_font_destroy (font);
    hb_face_destroy (face);
  }

  hb_draw_funcs_destroy (funcs);
}
#endif

#ifndef MAIN_CC_NO_PRIVATE_API
/* Only this part of this mini app uses private API */
#include "hb-static.cc"
#include "hb-open-file.hh"
#include "hb-ot-layout-gdef-table.hh"
#include "hb-ot-layout-gsubgpos.hh"

using namespace OT;

static void
print_layout_info_using_private_api (hb_blob_t *blob)
{
  const char *font_data = hb_blob_get_data (blob, nullptr);
  hb_blob_t *font_blob = hb_sanitize_context_t ().sanitize_blob<OpenTypeFontFile> (blob);
  const OpenTypeFontFile* sanitized = font_blob->as<OpenTypeFontFile> ();
  if (!font_blob->data)
  {
    printf ("Sanitization of the file wasn't successful. Exit");
    exit (1);
  }
  const OpenTypeFontFile& ot = *sanitized;

  switch (ot.get_tag ())
  {
  case OpenTypeFontFile::TrueTypeTag:
    printf ("OpenType font with TrueType outlines\n");
    break;
  case OpenTypeFontFile::CFFTag:
    printf ("OpenType font with CFF (Type1) outlines\n");
    break;
  case OpenTypeFontFile::TTCTag:
    printf ("TrueType Collection of OpenType fonts\n");
    break;
  case OpenTypeFontFile::TrueTag:
    printf ("Obsolete Apple TrueType font\n");
    break;
  case OpenTypeFontFile::Typ1Tag:
    printf ("Obsolete Apple Type1 font in SFNT container\n");
    break;
  case OpenTypeFontFile::DFontTag:
    printf ("DFont Mac Resource Fork\n");
    break;
  default:
    printf ("Unknown font format\n");
    break;
  }

  unsigned num_faces = hb_face_count (blob);
  printf ("%d font(s) found in file\n", num_faces);
  for (unsigned n_font = 0; n_font < num_faces; ++n_font)
  {
    const OpenTypeFontFace &font = ot.get_face (n_font);
    printf ("Font %d of %d:\n", n_font, num_faces);

    unsigned num_tables = font.get_table_count ();
    printf ("  %d table(s) found in font\n", num_tables);
    for (unsigned n_table = 0; n_table < num_tables; ++n_table)
    {
      const OpenTypeTable &table = font.get_table (n_table);
      printf ("  Table %2d of %2d: %.4s (0x%08x+0x%08x)\n", n_table, num_tables,
	      (const char *) table.tag,
	      (unsigned) table.offset,
	      (unsigned) table.length);

      switch (table.tag)
      {

      case HB_OT_TAG_GSUB:
      case HB_OT_TAG_GPOS:
	{

	const GSUBGPOS &g = *reinterpret_cast<const GSUBGPOS *> (font_data + table.offset);

	unsigned num_scripts = g.get_script_count ();
	printf ("    %d script(s) found in table\n", num_scripts);
	for (unsigned n_script = 0; n_script < num_scripts; ++n_script)
	{
	  const Script &script = g.get_script (n_script);
	  printf ("    Script %2d of %2d: %.4s\n", n_script, num_scripts,
		  (const char *) g.get_script_tag (n_script));

	  if (!script.has_default_lang_sys ())
	    printf ("      No default language system\n");
	  int num_langsys = script.get_lang_sys_count ();
	  printf ("      %d language system(s) found in script\n", num_langsys);
	  for (int n_langsys = script.has_default_lang_sys () ? -1 : 0; n_langsys < num_langsys; ++n_langsys)
	  {
	    const LangSys &langsys = n_langsys == -1
				   ? script.get_default_lang_sys ()
				   : script.get_lang_sys (n_langsys);
	    if (n_langsys == -1)
	      printf ("      Default Language System\n");
	    else
	      printf ("      Language System %2d of %2d: %.4s\n", n_langsys, num_langsys,
		      (const char *) script.get_lang_sys_tag (n_langsys));
	    if (!langsys.has_required_feature ())
	      printf ("        No required feature\n");
	    else
	      printf ("        Required feature index: %d\n",
		      langsys.get_required_feature_index ());

	    unsigned num_features = langsys.get_feature_count ();
	    printf ("        %d feature(s) found in language system\n", num_features);
	    for (unsigned n_feature = 0; n_feature < num_features; ++n_feature)
	    {
	      printf ("        Feature index %2d of %2d: %d\n", n_feature, num_features,
		      langsys.get_feature_index (n_feature));
	    }
	  }
	}

	unsigned num_features = g.get_feature_count ();
	printf ("    %d feature(s) found in table\n", num_features);
	for (unsigned n_feature = 0; n_feature < num_features; ++n_feature)
	{
	  const Feature &feature = g.get_feature (n_feature);
	  unsigned num_lookups = feature.get_lookup_count ();
	  printf ("    Feature %2d of %2d: %c%c%c%c\n", n_feature, num_features,
		  HB_UNTAG (g.get_feature_tag (n_feature)));

	  printf ("        %d lookup(s) found in feature\n", num_lookups);
	  for (unsigned n_lookup = 0; n_lookup < num_lookups; ++n_lookup) {
	    printf ("        Lookup index %2d of %2d: %d\n", n_lookup, num_lookups,
		    feature.get_lookup_index (n_lookup));
	  }
	}

	unsigned num_lookups = g.get_lookup_count ();
	printf ("    %d lookup(s) found in table\n", num_lookups);
	for (unsigned n_lookup = 0; n_lookup < num_lookups; ++n_lookup)
	{
	  const Lookup &lookup = g.get_lookup (n_lookup);
	  printf ("    Lookup %2d of %2d: type %d, props 0x%04X\n", n_lookup, num_lookups,
		  lookup.get_type (), lookup.get_props ());
	}

	}
	break;

      case GDEF::tableTag:
	{

	const GDEF &gdef = *reinterpret_cast<const GDEF *> (font_data + table.offset);

	printf ("    Has %sglyph classes\n",
		  gdef.has_glyph_classes () ? "" : "no ");
	printf ("    Has %smark attachment types\n",
		  gdef.has_mark_attachment_types () ? "" : "no ");
	printf ("    Has %sattach points\n",
		  gdef.has_attach_points () ? "" : "no ");
	printf ("    Has %slig carets\n",
		  gdef.has_lig_carets () ? "" : "no ");
	printf ("    Has %smark sets\n",
		  gdef.has_mark_sets () ? "" : "no ");
	break;
	}
      }
    }
  }
}
/* end of private API use */
#endif

int
main (int argc, char **argv)
{
  if (argc != 2)
  {
    fprintf (stderr, "usage: %s font-file.ttf\n", argv[0]);
    exit (1);
  }

  hb_blob_t *blob = hb_blob_create_from_file (argv[1]);
  printf ("Opened font file %s: %d bytes long\n", argv[1], hb_blob_get_length (blob));
#ifndef MAIN_CC_NO_PRIVATE_API
  print_layout_info_using_private_api (blob);
#endif
#if !defined(HB_NO_COLOR) && !defined(HB_NO_DRAW) && defined(HB_EXPERIMENTAL_API)
  dump_glyphs (blob, argv[1]);
#endif
  hb_blob_destroy (blob);

  return 0;
}
