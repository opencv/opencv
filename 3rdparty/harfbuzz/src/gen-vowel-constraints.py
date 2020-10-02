#!/usr/bin/env python3

"""Generator of the function to prohibit certain vowel sequences.

It creates ``_hb_preprocess_text_vowel_constraints``, which inserts dotted
circles into sequences prohibited by the USE script development spec.
This function should be used as the ``preprocess_text`` of an
``hb_ot_complex_shaper_t``.

usage: ./gen-vowel-constraints.py ms-use/IndicShapingInvalidCluster.txt Scripts.txt

Input file:
* https://unicode.org/Public/UCD/latest/ucd/Scripts.txt
"""

import collections
def write (s):
	sys.stdout.flush ()
	sys.stdout.buffer.write (s.encode ('utf-8'))
import sys

if len (sys.argv) != 3:
	sys.exit (__doc__)

with open (sys.argv[2], encoding='utf-8') as f:
	scripts_header = [f.readline () for i in range (2)]
	scripts = {}
	script_order = {}
	for line in f:
		j = line.find ('#')
		if j >= 0:
			line = line[:j]
		fields = [x.strip () for x in line.split (';')]
		if len (fields) == 1:
			continue
		uu = fields[0].split ('..')
		start = int (uu[0], 16)
		if len (uu) == 1:
			end = start
		else:
			end = int (uu[1], 16)
		script = fields[1]
		for u in range (start, end + 1):
			scripts[u] = script
		if script not in script_order:
			script_order[script] = start

class ConstraintSet (object):
	"""A set of prohibited code point sequences.

	Args:
		constraint (List[int]): A prohibited code point sequence.

	"""
	def __init__ (self, constraint):
		# Either a list or a dictionary. As a list of code points, it
		# represents a prohibited code point sequence. As a dictionary,
		# it represents a set of prohibited sequences, where each item
		# represents the set of prohibited sequences starting with the
		# key (a code point) concatenated with any of the values
		# (ConstraintSets).
		self._c = constraint

	def add (self, constraint):
		"""Add a constraint to this set."""
		if not constraint:
			return
		first = constraint[0]
		rest = constraint[1:]
		if isinstance (self._c, list):
			if constraint == self._c[:len (constraint)]:
				self._c = constraint
			elif self._c != constraint[:len (self._c)]:
				self._c = {self._c[0]: ConstraintSet (self._c[1:])}
		if isinstance (self._c, dict):
			if first in self._c:
				self._c[first].add (rest)
			else:
				self._c[first] = ConstraintSet (rest)

	@staticmethod
	def _indent (depth):
		return ('  ' * depth).replace ('        ', '\t')

	def __str__ (self, index=0, depth=4):
		s = []
		indent = self._indent (depth)
		if isinstance (self._c, list):
			if len (self._c) == 0:
				assert index == 2, 'Cannot use `matched` for this constraint; the general case has not been implemented'
				s.append ('{}matched = true;\n'.format (indent))
			elif len (self._c) == 1:
				assert index == 1, 'Cannot use `matched` for this constraint; the general case has not been implemented'
				s.append ('{}matched = 0x{:04X}u == buffer->cur ({}).codepoint;\n'.format (indent, next (iter (self._c)), index or ''))
			else:
				s.append ('{}if (0x{:04X}u == buffer->cur ({}).codepoint &&\n'.format (indent, self._c[0], index or ''))
				if index:
					s.append ('{}buffer->idx + {} < count &&\n'.format (self._indent (depth + 2), index + 1))
				for i, cp in enumerate (self._c[1:], start=1):
					s.append ('{}0x{:04X}u == buffer->cur ({}).codepoint{}\n'.format (
						self._indent (depth + 2), cp, index + i, ')' if i == len (self._c) - 1 else ' &&'))
				s.append ('{}{{\n'.format (indent))
				for i in range (index):
					s.append ('{}buffer->next_glyph ();\n'.format (self._indent (depth + 1)))
				s.append ('{}matched = true;\n'.format (self._indent (depth + 1)))
				s.append ('{}}}\n'.format (indent))
		else:
			s.append ('{}switch (buffer->cur ({}).codepoint)\n'.format(indent, index or ''))
			s.append ('{}{{\n'.format (indent))
			cases = collections.defaultdict (set)
			for first, rest in sorted (self._c.items ()):
				cases[rest.__str__ (index + 1, depth + 2)].add (first)
			for body, labels in sorted (cases.items (), key=lambda b_ls: sorted (b_ls[1])[0]):
				for i, cp in enumerate (sorted (labels)):
					if i % 4 == 0:
						s.append (self._indent (depth + 1))
					else:
						s.append (' ')
					s.append ('case 0x{:04X}u:{}'.format (cp, '\n' if i % 4 == 3 else ''))
				if len (labels) % 4 != 0:
					s.append ('\n')
				s.append (body)
				s.append ('{}break;\n'.format (self._indent (depth + 2)))
			s.append ('{}}}\n'.format (indent))
		return ''.join (s)

constraints = {}
with open (sys.argv[1], encoding='utf-8') as f:
	constraints_header = []
	while True:
		line = f.readline ().strip ()
		if line == '#':
			break
		constraints_header.append(line)
	for line in f:
		j = line.find ('#')
		if j >= 0:
			line = line[:j]
		constraint = [int (cp, 16) for cp in line.split (';')[0].split ()]
		if not constraint: continue
		assert 2 <= len (constraint), 'Prohibited sequence is too short: {}'.format (constraint)
		script = scripts[constraint[0]]
		if script in constraints:
			constraints[script].add (constraint)
		else:
			constraints[script] = ConstraintSet (constraint)
		assert constraints, 'No constraints found'

print ('/* == Start of generated functions == */')
print ('/*')
print (' * The following functions are generated by running:')
print (' *')
print (' *   %s ms-use/IndicShapingInvalidCluster.txt Scripts.txt' % sys.argv[0])
print (' *')
print (' * on files with these headers:')
print (' *')
for line in constraints_header:
	print (' * %s' % line.strip ())
print (' *')
for line in scripts_header:
	print (' * %s' % line.strip ())
print (' */')

print ()
print ('#include "hb.hh"')
print ()
print ('#ifndef HB_NO_OT_SHAPE')
print ()
print ('#include "hb-ot-shape-complex-vowel-constraints.hh"')
print ()
print ('static void')
print ('_output_dotted_circle (hb_buffer_t *buffer)')
print ('{')
print ('  hb_glyph_info_t &dottedcircle = buffer->output_glyph (0x25CCu);')
print ('  _hb_glyph_info_reset_continuation (&dottedcircle);')
print ('}')
print ()
print ('static void')
print ('_output_with_dotted_circle (hb_buffer_t *buffer)')
print ('{')
print ('  _output_dotted_circle (buffer);')
print ('  buffer->next_glyph ();')
print ('}')
print ()

print ('void')
print ('_hb_preprocess_text_vowel_constraints (const hb_ot_shape_plan_t *plan HB_UNUSED,')
print ('\t\t\t\t       hb_buffer_t              *buffer,')
print ('\t\t\t\t       hb_font_t                *font HB_UNUSED)')
print ('{')
print ('#ifdef HB_NO_OT_SHAPE_COMPLEX_VOWEL_CONSTRAINTS')
print ('  return;')
print ('#endif')
print ('  if (buffer->flags & HB_BUFFER_FLAG_DO_NOT_INSERT_DOTTED_CIRCLE)')
print ('    return;')
print ()
print ('  /* UGLY UGLY UGLY business of adding dotted-circle in the middle of')
print ('   * vowel-sequences that look like another vowel.  Data for each script')
print ('   * collected from the USE script development spec.')
print ('   *')
print ('   * https://github.com/harfbuzz/harfbuzz/issues/1019')
print ('   */')
print ('  bool processed = false;')
print ('  buffer->clear_output ();')
print ('  unsigned int count = buffer->len;')
print ('  switch ((unsigned) buffer->props.script)')
print ('  {')

for script, constraints in sorted (constraints.items (), key=lambda s_c: script_order[s_c[0]]):
	print ('    case HB_SCRIPT_{}:'.format (script.upper ()))
	print ('      for (buffer->idx = 0; buffer->idx + 1 < count && buffer->successful;)')
	print ('      {')
	print ('\tbool matched = false;')
	write (str (constraints))
	print ('\tbuffer->next_glyph ();')
	print ('\tif (matched) _output_with_dotted_circle (buffer);')
	print ('      }')
	print ('      processed = true;')
	print ('      break;')
	print ()

print ('    default:')
print ('      break;')
print ('  }')
print ('  if (processed)')
print ('  {')
print ('    if (buffer->idx < count)')
print ('      buffer->next_glyph ();')
print ('    buffer->swap_buffers ();')
print ('  }')
print ('}')

print ()
print ()
print ('#endif')
print ('/* == End of generated functions == */')
