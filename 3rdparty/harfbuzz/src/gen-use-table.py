#!/usr/bin/env python3
# flake8: noqa: F821

"""usage: ./gen-use-table.py IndicSyllabicCategory.txt IndicPositionalCategory.txt UnicodeData.txt Blocks.txt

Input file:
* https://unicode.org/Public/UCD/latest/ucd/IndicSyllabicCategory.txt
* https://unicode.org/Public/UCD/latest/ucd/IndicPositionalCategory.txt
* https://unicode.org/Public/UCD/latest/ucd/UnicodeData.txt
* https://unicode.org/Public/UCD/latest/ucd/Blocks.txt
"""

import sys

if len (sys.argv) != 5:
	sys.exit (__doc__)

BLACKLISTED_BLOCKS = ["Thai", "Lao"]

files = [open (x, encoding='utf-8') for x in sys.argv[1:]]

headers = [[f.readline () for i in range (2)] for j,f in enumerate(files) if j != 2]
headers.append (["UnicodeData.txt does not have a header."])

data = [{} for _ in files]
values = [{} for _ in files]
for i, f in enumerate (files):
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

		t = fields[1 if i != 2 else 2]

		for u in range (start, end + 1):
			data[i][u] = t
		values[i][t] = values[i].get (t, 0) + end - start + 1

defaults = ('Other', 'Not_Applicable', 'Cn', 'No_Block')

# TODO Characters that are not in Unicode Indic files, but used in USE
data[0][0x034F] = defaults[0]
data[0][0x1B61] = defaults[0]
data[0][0x1B63] = defaults[0]
data[0][0x1B64] = defaults[0]
data[0][0x1B65] = defaults[0]
data[0][0x1B66] = defaults[0]
data[0][0x1B67] = defaults[0]
data[0][0x1B69] = defaults[0]
data[0][0x1B6A] = defaults[0]
data[0][0x2060] = defaults[0]
# TODO https://github.com/harfbuzz/harfbuzz/pull/1685
data[0][0x1B5B] = 'Consonant_Placeholder'
data[0][0x1B5C] = 'Consonant_Placeholder'
data[0][0x1B5F] = 'Consonant_Placeholder'
data[0][0x1B62] = 'Consonant_Placeholder'
data[0][0x1B68] = 'Consonant_Placeholder'
# TODO https://github.com/harfbuzz/harfbuzz/issues/1035
data[0][0x11C44] = 'Consonant_Placeholder'
data[0][0x11C45] = 'Consonant_Placeholder'
# TODO https://github.com/harfbuzz/harfbuzz/pull/1399
data[0][0x111C8] = 'Consonant_Placeholder'
for u in range (0xFE00, 0xFE0F + 1):
	data[0][u] = defaults[0]

# Merge data into one dict:
for i,v in enumerate (defaults):
	values[i][v] = values[i].get (v, 0) + 1
combined = {}
for i,d in enumerate (data):
	for u,v in d.items ():
		if i >= 2 and not u in combined:
			continue
		if not u in combined:
			combined[u] = list (defaults)
		combined[u][i] = v
combined = {k:v for k,v in combined.items() if v[3] not in BLACKLISTED_BLOCKS}
data = combined
del combined


property_names = [
	# General_Category
	'Cc', 'Cf', 'Cn', 'Co', 'Cs', 'Ll', 'Lm', 'Lo', 'Lt', 'Lu', 'Mc',
	'Me', 'Mn', 'Nd', 'Nl', 'No', 'Pc', 'Pd', 'Pe', 'Pf', 'Pi', 'Po',
	'Ps', 'Sc', 'Sk', 'Sm', 'So', 'Zl', 'Zp', 'Zs',
	# Indic_Syllabic_Category
	'Other',
	'Bindu',
	'Visarga',
	'Avagraha',
	'Nukta',
	'Virama',
	'Pure_Killer',
	'Invisible_Stacker',
	'Vowel_Independent',
	'Vowel_Dependent',
	'Vowel',
	'Consonant_Placeholder',
	'Consonant',
	'Consonant_Dead',
	'Consonant_With_Stacker',
	'Consonant_Prefixed',
	'Consonant_Preceding_Repha',
	'Consonant_Succeeding_Repha',
	'Consonant_Subjoined',
	'Consonant_Medial',
	'Consonant_Final',
	'Consonant_Head_Letter',
	'Consonant_Initial_Postfixed',
	'Modifying_Letter',
	'Tone_Letter',
	'Tone_Mark',
	'Gemination_Mark',
	'Cantillation_Mark',
	'Register_Shifter',
	'Syllable_Modifier',
	'Consonant_Killer',
	'Non_Joiner',
	'Joiner',
	'Number_Joiner',
	'Number',
	'Brahmi_Joining_Number',
	# Indic_Positional_Category
	'Not_Applicable',
	'Right',
	'Left',
	'Visual_Order_Left',
	'Left_And_Right',
	'Top',
	'Bottom',
	'Top_And_Bottom',
	'Top_And_Bottom_And_Left',
	'Top_And_Right',
	'Top_And_Left',
	'Top_And_Left_And_Right',
	'Bottom_And_Left',
	'Bottom_And_Right',
	'Top_And_Bottom_And_Right',
	'Overstruck',
]

class PropertyValue(object):
	def __init__(self, name_):
		self.name = name_
	def __str__(self):
		return self.name
	def __eq__(self, other):
		return self.name == (other if isinstance(other, str) else other.name)
	def __ne__(self, other):
		return not (self == other)
	def __hash__(self):
		return hash(str(self))

property_values = {}

for name in property_names:
	value = PropertyValue(name)
	assert value not in property_values
	assert value not in globals()
	property_values[name] = value
globals().update(property_values)


def is_BASE(U, UISC, UGC):
	return (UISC in [Number, Consonant, Consonant_Head_Letter,
			#SPEC-DRAFT Consonant_Placeholder,
			Tone_Letter,
			Vowel_Independent #SPEC-DRAFT
			] or
		(UGC == Lo and UISC in [Avagraha, Bindu, Consonant_Final, Consonant_Medial,
					Consonant_Subjoined, Vowel, Vowel_Dependent]))
def is_BASE_IND(U, UISC, UGC):
	#SPEC-DRAFT return (UISC in [Consonant_Dead, Modifying_Letter] or UGC == Po)
	return (UISC in [Consonant_Dead, Modifying_Letter] or
		(UGC == Po and not U in [0x104B, 0x104E, 0x1B5B, 0x1B5C, 0x1B5F, 0x2022, 0x111C8, 0x11A3F, 0x11A45, 0x11C44, 0x11C45]) or
		False # SPEC-DRAFT-OUTDATED! U == 0x002D
		)
def is_BASE_NUM(U, UISC, UGC):
	return UISC == Brahmi_Joining_Number
def is_BASE_OTHER(U, UISC, UGC):
	if UISC == Consonant_Placeholder: return True #SPEC-DRAFT
	#SPEC-DRAFT return U in [0x00A0, 0x00D7, 0x2015, 0x2022, 0x25CC, 0x25FB, 0x25FC, 0x25FD, 0x25FE]
	return U in [0x2015, 0x2022, 0x25FB, 0x25FC, 0x25FD, 0x25FE]
def is_CGJ(U, UISC, UGC):
	return U == 0x034F
def is_CONS_FINAL(U, UISC, UGC):
	return ((UISC == Consonant_Final and UGC != Lo) or
		UISC == Consonant_Succeeding_Repha)
def is_CONS_FINAL_MOD(U, UISC, UGC):
	#SPEC-DRAFT return  UISC in [Consonant_Final_Modifier, Syllable_Modifier]
	return  UISC == Syllable_Modifier
def is_CONS_MED(U, UISC, UGC):
	# Consonant_Initial_Postfixed is new in Unicode 11; not in the spec.
	return (UISC == Consonant_Medial and UGC != Lo or
		UISC == Consonant_Initial_Postfixed)
def is_CONS_MOD(U, UISC, UGC):
	return UISC in [Nukta, Gemination_Mark, Consonant_Killer]
def is_CONS_SUB(U, UISC, UGC):
	#SPEC-DRAFT return UISC == Consonant_Subjoined
	return UISC == Consonant_Subjoined and UGC != Lo
def is_CONS_WITH_STACKER(U, UISC, UGC):
	return UISC == Consonant_With_Stacker
def is_HALANT(U, UISC, UGC):
	return (UISC in [Virama, Invisible_Stacker]
		and not is_HALANT_OR_VOWEL_MODIFIER(U, UISC, UGC)
		and not is_SAKOT(U, UISC, UGC))
def is_HALANT_OR_VOWEL_MODIFIER(U, UISC, UGC):
	# https://github.com/harfbuzz/harfbuzz/issues/1102
	# https://github.com/harfbuzz/harfbuzz/issues/1379
	return U in [0x11046, 0x1134D]
def is_HALANT_NUM(U, UISC, UGC):
	return UISC == Number_Joiner
def is_ZWNJ(U, UISC, UGC):
	return UISC == Non_Joiner
def is_ZWJ(U, UISC, UGC):
	return UISC == Joiner
def is_Word_Joiner(U, UISC, UGC):
	return U == 0x2060
def is_OTHER(U, UISC, UGC):
	#SPEC-OUTDATED return UGC == Zs # or any other SCRIPT_COMMON characters
	return (UISC == Other
		and not is_SYM(U, UISC, UGC)
		and not is_SYM_MOD(U, UISC, UGC)
		and not is_CGJ(U, UISC, UGC)
		and not is_Word_Joiner(U, UISC, UGC)
		and not is_VARIATION_SELECTOR(U, UISC, UGC)
	)
def is_Reserved(U, UISC, UGC):
	return UGC == 'Cn'
def is_REPHA(U, UISC, UGC):
	return UISC in [Consonant_Preceding_Repha, Consonant_Prefixed]
def is_SAKOT(U, UISC, UGC):
	return U == 0x1A60
def is_SYM(U, UISC, UGC):
	if U == 0x25CC: return False #SPEC-DRAFT
	#SPEC-DRAFT return UGC in [So, Sc] or UISC == Symbol_Letter
	return UGC in [So, Sc] and U not in [0x1B62, 0x1B68]
def is_SYM_MOD(U, UISC, UGC):
	return U in [0x1B6B, 0x1B6C, 0x1B6D, 0x1B6E, 0x1B6F, 0x1B70, 0x1B71, 0x1B72, 0x1B73]
def is_VARIATION_SELECTOR(U, UISC, UGC):
	return 0xFE00 <= U <= 0xFE0F
def is_VOWEL(U, UISC, UGC):
	# https://github.com/harfbuzz/harfbuzz/issues/376
	return (UISC == Pure_Killer or
		(UGC != Lo and UISC in [Vowel, Vowel_Dependent] and U not in [0xAA29]))
def is_VOWEL_MOD(U, UISC, UGC):
	# https://github.com/harfbuzz/harfbuzz/issues/376
	return (UISC in [Tone_Mark, Cantillation_Mark, Register_Shifter, Visarga] or
		(UGC != Lo and (UISC == Bindu or U in [0xAA29])))

use_mapping = {
	'B':	is_BASE,
	'IND':	is_BASE_IND,
	'N':	is_BASE_NUM,
	'GB':	is_BASE_OTHER,
	'CGJ':	is_CGJ,
	'F':	is_CONS_FINAL,
	'FM':	is_CONS_FINAL_MOD,
	'M':	is_CONS_MED,
	'CM':	is_CONS_MOD,
	'SUB':	is_CONS_SUB,
	'CS':	is_CONS_WITH_STACKER,
	'H':	is_HALANT,
	'HVM':	is_HALANT_OR_VOWEL_MODIFIER,
	'HN':	is_HALANT_NUM,
	'ZWNJ':	is_ZWNJ,
	'ZWJ':	is_ZWJ,
	'WJ':	is_Word_Joiner,
	'O':	is_OTHER,
	'Rsv':	is_Reserved,
	'R':	is_REPHA,
	'S':	is_SYM,
	'Sk':	is_SAKOT,
	'SM':	is_SYM_MOD,
	'VS':	is_VARIATION_SELECTOR,
	'V':	is_VOWEL,
	'VM':	is_VOWEL_MOD,
}

use_positions = {
	'F': {
		'Abv': [Top],
		'Blw': [Bottom],
		'Pst': [Right],
	},
	'M': {
		'Abv': [Top],
		'Blw': [Bottom, Bottom_And_Left, Bottom_And_Right],
		'Pst': [Right],
		'Pre': [Left, Top_And_Bottom_And_Left],
	},
	'CM': {
		'Abv': [Top],
		'Blw': [Bottom],
	},
	'V': {
		'Abv': [Top, Top_And_Bottom, Top_And_Bottom_And_Right, Top_And_Right],
		'Blw': [Bottom, Overstruck, Bottom_And_Right],
		'Pst': [Right, Top_And_Left, Top_And_Left_And_Right, Left_And_Right],
		'Pre': [Left],
	},
	'VM': {
		'Abv': [Top],
		'Blw': [Bottom, Overstruck],
		'Pst': [Right],
		'Pre': [Left],
	},
	'SM': {
		'Abv': [Top],
		'Blw': [Bottom],
	},
	'H': None,
	'HVM': None,
	'B': None,
	'FM': {
		'Abv': [Top],
		'Blw': [Bottom],
		'Pst': [Not_Applicable],
	},
	'SUB': None,
}

def map_to_use(data):
	out = {}
	items = use_mapping.items()
	for U,(UISC,UIPC,UGC,UBlock) in data.items():

		# Resolve Indic_Syllabic_Category

		# TODO: These don't have UISC assigned in Unicode 13.0.0, but have UIPC
		if 0x1CE2 <= U <= 0x1CE8: UISC = Cantillation_Mark

		# Tibetan:
		# TODO: These don't have UISC assigned in Unicode 13.0.0, but have UIPC
		if 0x0F18 <= U <= 0x0F19 or 0x0F3E <= U <= 0x0F3F: UISC = Vowel_Dependent
		if 0x0F86 <= U <= 0x0F87: UISC = Tone_Mark
		# Overrides to allow NFC order matching syllable
		# https://github.com/harfbuzz/harfbuzz/issues/1012
		if UBlock == 'Tibetan' and is_VOWEL (U, UISC, UGC):
			if UIPC == Top:
				UIPC = Bottom

		# TODO: https://github.com/harfbuzz/harfbuzz/pull/982
		# also  https://github.com/harfbuzz/harfbuzz/issues/1012
		if UBlock == 'Chakma' and is_VOWEL (U, UISC, UGC):
			if UIPC == Top:
				UIPC = Bottom
			elif UIPC == Bottom:
				UIPC = Top

		# TODO: https://github.com/harfbuzz/harfbuzz/pull/627
		if 0x1BF2 <= U <= 0x1BF3: UISC = Nukta; UIPC = Bottom

		# TODO: U+1CED should only be allowed after some of
		# the nasalization marks, maybe only for U+1CE9..U+1CF1.
		if U == 0x1CED: UISC = Tone_Mark

		# TODO: https://github.com/harfbuzz/harfbuzz/issues/1105
		if U == 0x11134: UISC = Gemination_Mark

		values = [k for k,v in items if v(U,UISC,UGC)]
		assert len(values) == 1, "%s %s %s %s" % (hex(U), UISC, UGC, values)
		USE = values[0]

		# Resolve Indic_Positional_Category

		# TODO: These should die, but have UIPC in Unicode 13.0.0
		if U in [0x953, 0x954]: UIPC = Not_Applicable

		# TODO: https://github.com/harfbuzz/harfbuzz/pull/2012
		if U == 0x1C29: UIPC = Left

		# TODO: These are not in USE's override list that we have, nor are they in Unicode 13.0.0
		if 0xA926 <= U <= 0xA92A: UIPC = Top
		# TODO: https://github.com/harfbuzz/harfbuzz/pull/1037
		#  and https://github.com/harfbuzz/harfbuzz/issues/1631
		if U in [0x11302, 0x11303, 0x114C1]: UIPC = Top
		if 0x1CF8 <= U <= 0x1CF9: UIPC = Top

		assert (UIPC in [Not_Applicable, Visual_Order_Left] or
			USE == 'R' or
			USE in use_positions), "%s %s %s %s %s" % (hex(U), UIPC, USE, UISC, UGC)

		pos_mapping = use_positions.get(USE, None)
		if pos_mapping:
			values = [k for k,v in pos_mapping.items() if v and UIPC in v]
			assert len(values) == 1, "%s %s %s %s %s %s" % (hex(U), UIPC, USE, UISC, UGC, values)
			USE = USE + values[0]

		out[U] = (USE, UBlock)
	return out

defaults = ('O', 'No_Block')
data = map_to_use(data)

print ("/* == Start of generated table == */")
print ("/*")
print (" * The following table is generated by running:")
print (" *")
print (" *   ./gen-use-table.py IndicSyllabicCategory.txt IndicPositionalCategory.txt UnicodeData.txt Blocks.txt")
print (" *")
print (" * on files with these headers:")
print (" *")
for h in headers:
	for l in h:
		print (" * %s" % (l.strip()))
print (" */")
print ()
print ('#include "hb.hh"')
print ()
print ('#ifndef HB_NO_OT_SHAPE')
print ()
print ('#include "hb-ot-shape-complex-use.hh"')
print ()

total = 0
used = 0
last_block = None
def print_block (block, start, end, data):
	global total, used, last_block
	if block and block != last_block:
		print ()
		print ()
		print ("  /* %s */" % block)
		if start % 16:
			print (' ' * (20 + (start % 16 * 6)), end='')
	num = 0
	assert start % 8 == 0
	assert (end+1) % 8 == 0
	for u in range (start, end+1):
		if u % 16 == 0:
			print ()
			print ("  /* %04X */" % u, end='')
		if u in data:
			num += 1
		d = data.get (u, defaults)
		print ("%6s," % d[0], end='')

	total += end - start + 1
	used += num
	if block:
		last_block = block

uu = sorted (data.keys ())

last = -100000
num = 0
offset = 0
starts = []
ends = []
print ('#pragma GCC diagnostic push')
print ('#pragma GCC diagnostic ignored "-Wunused-macros"')
for k,v in sorted(use_mapping.items()):
	if k in use_positions and use_positions[k]: continue
	print ("#define %s	USE_%s	/* %s */" % (k, k, v.__name__[3:]))
for k,v in sorted(use_positions.items()):
	if not v: continue
	for suf in v.keys():
		tag = k + suf
		print ("#define %s	USE_%s" % (tag, tag))
print ('#pragma GCC diagnostic pop')
print ("")
print ("static const USE_TABLE_ELEMENT_TYPE use_table[] = {")
for u in uu:
	if u <= last:
		continue
	block = data[u][1]

	start = u//8*8
	end = start+1
	while end in uu and block == data[end][1]:
		end += 1
	end = (end-1)//8*8 + 7

	if start != last + 1:
		if start - last <= 1+16*3:
			print_block (None, last+1, start-1, data)
		else:
			if last >= 0:
				ends.append (last + 1)
				offset += ends[-1] - starts[-1]
			print ()
			print ()
			print ("#define use_offset_0x%04xu %d" % (start, offset))
			starts.append (start)

	print_block (block, start, end, data)
	last = end
ends.append (last + 1)
offset += ends[-1] - starts[-1]
print ()
print ()
occupancy = used * 100. / total
page_bits = 12
print ("}; /* Table items: %d; occupancy: %d%% */" % (offset, occupancy))
print ()
print ("USE_TABLE_ELEMENT_TYPE")
print ("hb_use_get_category (hb_codepoint_t u)")
print ("{")
print ("  switch (u >> %d)" % page_bits)
print ("  {")
pages = set([u>>page_bits for u in starts+ends])
for p in sorted(pages):
	print ("    case 0x%0Xu:" % p)
	for (start,end) in zip (starts, ends):
		if p not in [start>>page_bits, end>>page_bits]: continue
		offset = "use_offset_0x%04xu" % start
		print ("      if (hb_in_range<hb_codepoint_t> (u, 0x%04Xu, 0x%04Xu)) return use_table[u - 0x%04Xu + %s];" % (start, end-1, start, offset))
	print ("      break;")
	print ("")
print ("    default:")
print ("      break;")
print ("  }")
print ("  return USE_O;")
print ("}")
print ()
for k in sorted(use_mapping.keys()):
	if k in use_positions and use_positions[k]: continue
	print ("#undef %s" % k)
for k,v in sorted(use_positions.items()):
	if not v: continue
	for suf in v.keys():
		tag = k + suf
		print ("#undef %s" % tag)
print ()
print ()
print ('#endif')
print ("/* == End of generated table == */")

# Maintain at least 50% occupancy in the table */
if occupancy < 50:
	raise Exception ("Table too sparse, please investigate: ", occupancy)
