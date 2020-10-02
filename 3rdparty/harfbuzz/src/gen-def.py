#!/usr/bin/env python3

"usage: gen-def.py harfbuzz.def hb.h [hb-blob.h hb-buffer.h ...]"

import os, re, sys

if len (sys.argv) < 3:
	sys.exit(__doc__)

output_file = sys.argv[1]
header_paths = sys.argv[2:]

headers_content = []
for h in header_paths:
	if h.endswith (".h"):
		with open (h, encoding='utf-8') as f: headers_content.append (f.read ())

symbols = sorted (re.findall (r"^hb_\w+(?= \()", "\n".join (headers_content), re.M))
if '--experimental-api' not in sys.argv:
	# Move these to harfbuzz-sections.txt when got stable
	experimental_symbols = \
"""hb_font_draw_glyph
hb_draw_funcs_t
hb_draw_close_path_func_t
hb_draw_cubic_to_func_t
hb_draw_line_to_func_t
hb_draw_move_to_func_t
hb_draw_quadratic_to_func_t
hb_draw_funcs_create
hb_draw_funcs_destroy
hb_draw_funcs_is_immutable
hb_draw_funcs_make_immutable
hb_draw_funcs_reference
hb_draw_funcs_set_close_path_func
hb_draw_funcs_set_cubic_to_func
hb_draw_funcs_set_line_to_func
hb_draw_funcs_set_move_to_func
hb_draw_funcs_set_quadratic_to_func
hb_style_get_value
hb_font_get_var_coords_design""".splitlines ()
	symbols = [x for x in symbols if x not in experimental_symbols]
symbols = "\n".join (symbols)

result = symbols if os.getenv ('PLAIN_LIST', '') else """EXPORTS
%s
LIBRARY lib%s-0.dll""" % (symbols, output_file.replace ('src/', '').replace ('.def', ''))

with open (output_file, "w") as f: f.write (result)
