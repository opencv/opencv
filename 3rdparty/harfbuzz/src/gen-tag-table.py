#!/usr/bin/env python3

"""Generator of the mapping from OpenType tags to BCP 47 tags and vice
versa.

It creates a ``const LangTag[]``, matching the tags from the OpenType
languages system tag list to the language subtags of the BCP 47 language
subtag registry, with some manual adjustments. The mappings are
supplemented with macrolanguages' sublanguages and retired codes'
replacements, according to BCP 47 and some manual additions where BCP 47
omits a retired code entirely.

Also generated is a function, ``hb_ot_ambiguous_tag_to_language``,
intended for use by ``hb_ot_tag_to_language``. It maps OpenType tags
back to BCP 47 tags. Ambiguous OpenType tags (those that correspond to
multiple BCP 47 tags) are listed here, except when the alphabetically
first BCP 47 tag happens to be the chosen disambiguated tag. In that
case, the fallback behavior will choose the right tag anyway.

usage: ./gen-tag-table.py languagetags language-subtag-registry

Input files:
* https://docs.microsoft.com/en-us/typography/opentype/spec/languagetags
* https://www.iana.org/assignments/language-subtag-registry/language-subtag-registry
"""

import collections
from html.parser import HTMLParser
def write (s):
	sys.stdout.flush ()
	sys.stdout.buffer.write (s.encode ('utf-8'))
import itertools
import re
import sys
import unicodedata

if len (sys.argv) != 3:
	sys.exit (__doc__)

from html import unescape
def html_unescape (parser, entity):
	return unescape (entity)

def expect (condition, message=None):
	if not condition:
		if message is None:
			raise AssertionError
		raise AssertionError (message)

# from https://www-01.sil.org/iso639-3/iso-639-3.tab
ISO_639_3_TO_1 = {
	'aar': 'aa',
	'abk': 'ab',
	'afr': 'af',
	'aka': 'ak',
	'amh': 'am',
	'ara': 'ar',
	'arg': 'an',
	'asm': 'as',
	'ava': 'av',
	'ave': 'ae',
	'aym': 'ay',
	'aze': 'az',
	'bak': 'ba',
	'bam': 'bm',
	'bel': 'be',
	'ben': 'bn',
	'bis': 'bi',
	'bod': 'bo',
	'bos': 'bs',
	'bre': 'br',
	'bul': 'bg',
	'cat': 'ca',
	'ces': 'cs',
	'cha': 'ch',
	'che': 'ce',
	'chu': 'cu',
	'chv': 'cv',
	'cor': 'kw',
	'cos': 'co',
	'cre': 'cr',
	'cym': 'cy',
	'dan': 'da',
	'deu': 'de',
	'div': 'dv',
	'dzo': 'dz',
	'ell': 'el',
	'eng': 'en',
	'epo': 'eo',
	'est': 'et',
	'eus': 'eu',
	'ewe': 'ee',
	'fao': 'fo',
	'fas': 'fa',
	'fij': 'fj',
	'fin': 'fi',
	'fra': 'fr',
	'fry': 'fy',
	'ful': 'ff',
	'gla': 'gd',
	'gle': 'ga',
	'glg': 'gl',
	'glv': 'gv',
	'grn': 'gn',
	'guj': 'gu',
	'hat': 'ht',
	'hau': 'ha',
	'hbs': 'sh',
	'heb': 'he',
	'her': 'hz',
	'hin': 'hi',
	'hmo': 'ho',
	'hrv': 'hr',
	'hun': 'hu',
	'hye': 'hy',
	'ibo': 'ig',
	'ido': 'io',
	'iii': 'ii',
	'iku': 'iu',
	'ile': 'ie',
	'ina': 'ia',
	'ind': 'id',
	'ipk': 'ik',
	'isl': 'is',
	'ita': 'it',
	'jav': 'jv',
	'jpn': 'ja',
	'kal': 'kl',
	'kan': 'kn',
	'kas': 'ks',
	'kat': 'ka',
	'kau': 'kr',
	'kaz': 'kk',
	'khm': 'km',
	'kik': 'ki',
	'kin': 'rw',
	'kir': 'ky',
	'kom': 'kv',
	'kon': 'kg',
	'kor': 'ko',
	'kua': 'kj',
	'kur': 'ku',
	'lao': 'lo',
	'lat': 'la',
	'lav': 'lv',
	'lim': 'li',
	'lin': 'ln',
	'lit': 'lt',
	'ltz': 'lb',
	'lub': 'lu',
	'lug': 'lg',
	'mah': 'mh',
	'mal': 'ml',
	'mar': 'mr',
	'mkd': 'mk',
	'mlg': 'mg',
	'mlt': 'mt',
	'mol': 'mo',
	'mon': 'mn',
	'mri': 'mi',
	'msa': 'ms',
	'mya': 'my',
	'nau': 'na',
	'nav': 'nv',
	'nbl': 'nr',
	'nde': 'nd',
	'ndo': 'ng',
	'nep': 'ne',
	'nld': 'nl',
	'nno': 'nn',
	'nob': 'nb',
	'nor': 'no',
	'nya': 'ny',
	'oci': 'oc',
	'oji': 'oj',
	'ori': 'or',
	'orm': 'om',
	'oss': 'os',
	'pan': 'pa',
	'pli': 'pi',
	'pol': 'pl',
	'por': 'pt',
	'pus': 'ps',
	'que': 'qu',
	'roh': 'rm',
	'ron': 'ro',
	'run': 'rn',
	'rus': 'ru',
	'sag': 'sg',
	'san': 'sa',
	'sin': 'si',
	'slk': 'sk',
	'slv': 'sl',
	'sme': 'se',
	'smo': 'sm',
	'sna': 'sn',
	'snd': 'sd',
	'som': 'so',
	'sot': 'st',
	'spa': 'es',
	'sqi': 'sq',
	'srd': 'sc',
	'srp': 'sr',
	'ssw': 'ss',
	'sun': 'su',
	'swa': 'sw',
	'swe': 'sv',
	'tah': 'ty',
	'tam': 'ta',
	'tat': 'tt',
	'tel': 'te',
	'tgk': 'tg',
	'tgl': 'tl',
	'tha': 'th',
	'tir': 'ti',
	'ton': 'to',
	'tsn': 'tn',
	'tso': 'ts',
	'tuk': 'tk',
	'tur': 'tr',
	'twi': 'tw',
	'uig': 'ug',
	'ukr': 'uk',
	'urd': 'ur',
	'uzb': 'uz',
	'ven': 've',
	'vie': 'vi',
	'vol': 'vo',
	'wln': 'wa',
	'wol': 'wo',
	'xho': 'xh',
	'yid': 'yi',
	'yor': 'yo',
	'zha': 'za',
	'zho': 'zh',
	'zul': 'zu',
}

class LanguageTag (object):
	"""A BCP 47 language tag.

	Attributes:
		subtags (List[str]): The list of subtags in this tag.
		grandfathered (bool): Whether this tag is grandfathered. If
			``true``, the entire lowercased tag is the ``language``
			and the other subtag fields are empty.
		language (str): The language subtag.
		script (str): The script subtag.
		region (str): The region subtag.
		variant (str): The variant subtag.

	Args:
		tag (str): A BCP 47 language tag.

	"""
	def __init__ (self, tag):
		global bcp_47
		self.subtags = tag.lower ().split ('-')
		self.grandfathered = tag.lower () in bcp_47.grandfathered
		if self.grandfathered:
			self.language = tag.lower ()
			self.script = ''
			self.region = ''
			self.variant = ''
		else:
			self.language = self.subtags[0]
			self.script = self._find_first (lambda s: len (s) == 4 and s[0] > '9', self.subtags)
			self.region = self._find_first (lambda s: len (s) == 2 and s[0] > '9' or len (s) == 3 and s[0] <= '9', self.subtags[1:])
			self.variant = self._find_first (lambda s: len (s) > 4 or len (s) == 4 and s[0] <= '9', self.subtags)

	def __str__(self):
		return '-'.join(self.subtags)

	def __repr__ (self):
		return 'LanguageTag(%r)' % str(self)

	@staticmethod
	def _find_first (function, sequence):
		try:
			return next (iter (filter (function, sequence)))
		except StopIteration:
			return None

	def is_complex (self):
		"""Return whether this tag is too complex to represent as a
		``LangTag`` in the generated code.

		Complex tags need to be handled in
		``hb_ot_tags_from_complex_language``.

		Returns:
			Whether this tag is complex.
		"""
		return not (len (self.subtags) == 1
			or self.grandfathered
			and len (self.subtags[1]) != 3
			and ot.from_bcp_47[self.subtags[0]] == ot.from_bcp_47[self.language])

	def get_group (self):
		"""Return the group into which this tag should be categorized in
		``hb_ot_tags_from_complex_language``.

		The group is the first letter of the tag, or ``'und'`` if this tag
		should not be matched in a ``switch`` statement in the generated
		code.

		Returns:
			This tag's group.
		"""
		return ('und'
			if (self.language == 'und'
				or self.variant in bcp_47.prefixes and len (bcp_47.prefixes[self.variant]) == 1)
			else self.language[0])

class OpenTypeRegistryParser (HTMLParser):
	"""A parser for the OpenType language system tag registry.

	Attributes:
		header (str): The "last updated" line of the registry.
		names (Mapping[str, str]): A map of language system tags to the
			names they are given in the registry.
		ranks (DefaultDict[str, int]): A map of language system tags to
			numbers. If a single BCP 47 tag corresponds to multiple
			OpenType tags, the tags are ordered in increasing order by
			rank. The rank is based on the number of BCP 47 tags
			associated with a tag, though it may be manually modified.
		to_bcp_47 (DefaultDict[str, AbstractSet[str]]): A map of
			OpenType language system tags to sets of BCP 47 tags.
		from_bcp_47 (DefaultDict[str, AbstractSet[str]]): ``to_bcp_47``
			inverted. Its values start as unsorted sets;
			``sort_languages`` converts them to sorted lists.

	"""
	def __init__ (self):
		HTMLParser.__init__ (self)
		self.header = ''
		self.names = {}
		self.ranks = collections.defaultdict (int)
		self.to_bcp_47 = collections.defaultdict (set)
		self.from_bcp_47 = collections.defaultdict (set)
		# Whether the parser is in a <td> element
		self._td = False
		# The text of the <td> elements of the current <tr> element.
		self._current_tr = []

	def handle_starttag (self, tag, attrs):
		if tag == 'meta':
			for attr, value in attrs:
				if attr == 'name' and value == 'updated_at':
					self.header = self.get_starttag_text ()
					break
		elif tag == 'td':
			self._td = True
			self._current_tr.append ('')
		elif tag == 'tr':
			self._current_tr = []

	def handle_endtag (self, tag):
		if tag == 'td':
			self._td = False
		elif tag == 'tr' and self._current_tr:
			expect (2 <= len (self._current_tr) <= 3)
			name = self._current_tr[0].strip ()
			tag = self._current_tr[1].strip ("\t\n\v\f\r '")
			rank = 0
			if len (tag) > 4:
				expect (tag.endswith (' (deprecated)'), 'ill-formed OpenType tag: %s' % tag)
				name += ' (deprecated)'
				tag = tag.split (' ')[0]
				rank = 1
			self.names[tag] = re.sub (' languages$', '', name)
			if not self._current_tr[2]:
				return
			iso_codes = self._current_tr[2].strip ()
			self.to_bcp_47[tag].update (ISO_639_3_TO_1.get (code, code) for code in iso_codes.replace (' ', '').split (','))
			rank += 2 * len (self.to_bcp_47[tag])
			self.ranks[tag] = rank

	def handle_data (self, data):
		if self._td:
			self._current_tr[-1] += data

	def handle_charref (self, name):
		self.handle_data (html_unescape (self, '&#%s;' % name))

	def handle_entityref (self, name):
		self.handle_data (html_unescape (self, '&%s;' % name))

	def parse (self, filename):
		"""Parse the OpenType language system tag registry.

		Args:
			filename (str): The file name of the registry.
		"""
		with open (filename, encoding='utf-8') as f:
			self.feed (f.read ())
		expect (self.header)
		for tag, iso_codes in self.to_bcp_47.items ():
			for iso_code in iso_codes:
				self.from_bcp_47[iso_code].add (tag)

	def add_language (self, bcp_47_tag, ot_tag):
		"""Add a language as if it were in the registry.

		Args:
			bcp_47_tag (str): A BCP 47 tag. If the tag is more than just
				a language subtag, and if the language subtag is a
				macrolanguage, then new languages are added corresponding
				to the macrolanguages' individual languages with the
				remainder of the tag appended.
			ot_tag (str): An OpenType language system tag.
		"""
		global bcp_47
		self.to_bcp_47[ot_tag].add (bcp_47_tag)
		self.from_bcp_47[bcp_47_tag].add (ot_tag)
		if bcp_47_tag.lower () not in bcp_47.grandfathered:
			try:
				[macrolanguage, suffix] = bcp_47_tag.split ('-', 1)
				if macrolanguage in bcp_47.macrolanguages:
					s = set ()
					for language in bcp_47.macrolanguages[macrolanguage]:
						if language.lower () not in bcp_47.grandfathered:
							s.add ('%s-%s' % (language, suffix))
					bcp_47.macrolanguages['%s-%s' % (macrolanguage, suffix)] = s
			except ValueError:
				pass

	@staticmethod
	def _remove_language (tag_1, dict_1, dict_2):
		for tag_2 in dict_1.pop (tag_1):
			dict_2[tag_2].remove (tag_1)
			if not dict_2[tag_2]:
				del dict_2[tag_2]

	def remove_language_ot (self, ot_tag):
		"""Remove an OpenType tag from the registry.

		Args:
			ot_tag (str): An OpenType tag.
		"""
		self._remove_language (ot_tag, self.to_bcp_47, self.from_bcp_47)

	def remove_language_bcp_47 (self, bcp_47_tag):
		"""Remove a BCP 47 tag from the registry.

		Args:
			bcp_47_tag (str): A BCP 47 tag.
		"""
		self._remove_language (bcp_47_tag, self.from_bcp_47, self.to_bcp_47)

	def inherit_from_macrolanguages (self):
		"""Copy mappings from macrolanguages to individual languages.

		If a BCP 47 tag for an individual mapping has no OpenType
		mapping but its macrolanguage does, the mapping is copied to
		the individual language. For example, als (Tosk Albanian) has no
		explicit mapping, so it inherits from sq (Albanian) the mapping
		to SQI.

		If a BCP 47 tag for a macrolanguage has no OpenType mapping but
		all of its individual languages do and they all map to the same
		tags, the mapping is copied to the macrolanguage.
		"""
		global bcp_47
		original_ot_from_bcp_47 = dict (self.from_bcp_47)
		for macrolanguage, languages in dict (bcp_47.macrolanguages).items ():
			ot_macrolanguages = set (original_ot_from_bcp_47.get (macrolanguage, set ()))
			if ot_macrolanguages:
				for ot_macrolanguage in ot_macrolanguages:
					for language in languages:
						# Remove the following condition if e.g. nn should map to NYN,NOR
						# instead of just NYN.
						if language not in original_ot_from_bcp_47:
							self.add_language (language, ot_macrolanguage)
							self.ranks[ot_macrolanguage] += 1
			else:
				for language in languages:
					if language in original_ot_from_bcp_47:
						if ot_macrolanguages:
							ml = original_ot_from_bcp_47[language]
							if ml:
								ot_macrolanguages &= ml
							else:
								pass
						else:
							ot_macrolanguages |= original_ot_from_bcp_47[language]
					else:
						ot_macrolanguages.clear ()
					if not ot_macrolanguages:
						break
				for ot_macrolanguage in ot_macrolanguages:
					self.add_language (macrolanguage, ot_macrolanguage)

	def sort_languages (self):
		"""Sort the values of ``from_bcp_47`` in ascending rank order."""
		for language, tags in self.from_bcp_47.items ():
			self.from_bcp_47[language] = sorted (tags,
					key=lambda t: (self.ranks[t] + rank_delta (language, t), t))

ot = OpenTypeRegistryParser ()

class BCP47Parser (object):
	"""A parser for the BCP 47 subtag registry.

	Attributes:
		header (str): The "File-Date" line of the registry.
		names (Mapping[str, str]): A map of subtags to the names they
			are given in the registry. Each value is a
			``'\\n'``-separated list of names.
		scopes (Mapping[str, str]): A map of language subtags to strings
			suffixed to language names, including suffixes to explain
			language scopes.
		macrolanguages (DefaultDict[str, AbstractSet[str]]): A map of
			language subtags to the sets of language subtags which
			inherit from them. See
			``OpenTypeRegistryParser.inherit_from_macrolanguages``.
		prefixes (DefaultDict[str, AbstractSet[str]]): A map of variant
			subtags to their prefixes.
		grandfathered (AbstractSet[str]): The set of grandfathered tags,
			normalized to lowercase.

	"""
	def __init__ (self):
		self.header = ''
		self.names = {}
		self.scopes = {}
		self.macrolanguages = collections.defaultdict (set)
		self.prefixes = collections.defaultdict (set)
		self.grandfathered = set ()

	def parse (self, filename):
		"""Parse the BCP 47 subtag registry.

		Args:
			filename (str): The file name of the registry.
		"""
		with open (filename, encoding='utf-8') as f:
			subtag_type = None
			subtag = None
			deprecated = False
			has_preferred_value = False
			line_buffer = ''
			for line in itertools.chain (f, ['']):
				line = line.rstrip ()
				if line.startswith (' '):
					line_buffer += line[1:]
					continue
				line, line_buffer = line_buffer, line
				if line.startswith ('Type: '):
					subtag_type = line.split (' ')[1]
					deprecated = False
					has_preferred_value = False
				elif line.startswith ('Subtag: ') or line.startswith ('Tag: '):
					subtag = line.split (' ')[1]
					if subtag_type == 'grandfathered':
						self.grandfathered.add (subtag.lower ())
				elif line.startswith ('Description: '):
					description = line.split (' ', 1)[1].replace (' (individual language)', '')
					description = re.sub (' (\((individual |macro)language\)|languages)$', '',
							description)
					if subtag in self.names:
						self.names[subtag] += '\n' + description
					else:
						self.names[subtag] = description
				elif subtag_type == 'language' or subtag_type == 'grandfathered':
					if line.startswith ('Scope: '):
						scope = line.split (' ')[1]
						if scope == 'macrolanguage':
							scope = ' [macrolanguage]'
						elif scope == 'collection':
							scope = ' [family]'
						else:
							continue
						self.scopes[subtag] = scope
					elif line.startswith ('Deprecated: '):
						self.scopes[subtag] = ' (retired code)' + self.scopes.get (subtag, '')
						deprecated = True
					elif deprecated and line.startswith ('Comments: see '):
						# If a subtag is split into multiple replacement subtags,
						# it essentially represents a macrolanguage.
						for language in line.replace (',', '').split (' ')[2:]:
							self._add_macrolanguage (subtag, language)
					elif line.startswith ('Preferred-Value: '):
						# If a subtag is deprecated in favor of a single replacement subtag,
						# it is either a dialect or synonym of the preferred subtag. Either
						# way, it is close enough to the truth to consider the replacement
						# the macrolanguage of the deprecated language.
						has_preferred_value = True
						macrolanguage = line.split (' ')[1]
						self._add_macrolanguage (macrolanguage, subtag)
					elif not has_preferred_value and line.startswith ('Macrolanguage: '):
						self._add_macrolanguage (line.split (' ')[1], subtag)
				elif subtag_type == 'variant':
					if line.startswith ('Prefix: '):
						self.prefixes[subtag].add (line.split (' ')[1])
				elif line.startswith ('File-Date: '):
					self.header = line
		expect (self.header)

	def _add_macrolanguage (self, macrolanguage, language):
		global ot
		if language not in ot.from_bcp_47:
			for l in self.macrolanguages.get (language, set ()):
				self._add_macrolanguage (macrolanguage, l)
		if macrolanguage not in ot.from_bcp_47:
			for ls in list (self.macrolanguages.values ()):
				if macrolanguage in ls:
					ls.add (language)
					return
		self.macrolanguages[macrolanguage].add (language)

	def remove_extra_macrolanguages (self):
		"""Make every language have at most one macrolanguage."""
		inverted = collections.defaultdict (list)
		for macrolanguage, languages in self.macrolanguages.items ():
			for language in languages:
				inverted[language].append (macrolanguage)
		for language, macrolanguages in inverted.items ():
			if len (macrolanguages) > 1:
				macrolanguages.sort (key=lambda ml: len (self.macrolanguages[ml]))
				biggest_macrolanguage = macrolanguages.pop ()
				for macrolanguage in macrolanguages:
					self._add_macrolanguage (biggest_macrolanguage, macrolanguage)

	def get_name (self, lt):
		"""Return the names of the subtags in a language tag.

		Args:
			lt (LanguageTag): A BCP 47 language tag.

		Returns:
			The name form of ``lt``.
		"""
		name = self.names[lt.language].split ('\n')[0]
		if lt.script:
			name += '; ' + self.names[lt.script.title ()].split ('\n')[0]
		if lt.region:
			name += '; ' + self.names[lt.region.upper ()].split ('\n')[0]
		if lt.variant:
			name += '; ' + self.names[lt.variant].split ('\n')[0]
		return name

bcp_47 = BCP47Parser ()

ot.parse (sys.argv[1])
bcp_47.parse (sys.argv[2])

ot.add_language ('ary', 'MOR')

ot.add_language ('ath', 'ATH')

ot.add_language ('bai', 'BML')

ot.ranks['BAL'] = ot.ranks['KAR'] + 1

ot.add_language ('ber', 'BBR')

ot.remove_language_ot ('PGR')
ot.add_language ('el-polyton', 'PGR')

bcp_47.macrolanguages['et'] = {'ekk'}

bcp_47.names['flm'] = 'Falam Chin'
bcp_47.scopes['flm'] = ' (retired code)'
bcp_47.macrolanguages['flm'] = {'cfm'}

ot.ranks['FNE'] = ot.ranks['TNE'] + 1

ot.add_language ('und-fonipa', 'IPPH')

ot.add_language ('und-fonnapa', 'APPH')

ot.remove_language_ot ('IRT')
ot.add_language ('ga-Latg', 'IRT')

ot.remove_language_ot ('KGE')
ot.add_language ('und-Geok', 'KGE')

ot.add_language ('guk', 'GUK')
ot.names['GUK'] = 'Gumuz (SIL fonts)'
ot.ranks['GUK'] = ot.ranks['GMZ'] + 1

bcp_47.macrolanguages['id'] = {'in'}

bcp_47.macrolanguages['ijo'] = {'ijc'}

ot.add_language ('kht', 'KHN')
ot.names['KHN'] = ot.names['KHT'] + ' (Microsoft fonts)'
ot.names['KHT'] = ot.names['KHT'] + ' (OpenType spec and SIL fonts)'
ot.ranks['KHN'] = ot.ranks['KHT']
ot.ranks['KHT'] += 1

ot.ranks['LCR'] = ot.ranks['MCR'] + 1

ot.names['MAL'] = 'Malayalam Traditional'
ot.ranks['MLR'] += 1

bcp_47.names['mhv'] = 'Arakanese'
bcp_47.scopes['mhv'] = ' (retired code)'

ot.add_language ('no', 'NOR')

ot.add_language ('oc-provenc', 'PRO')

ot.add_language ('qu', 'QUZ')
ot.add_language ('qub', 'QWH')
ot.add_language ('qud', 'QVI')
ot.add_language ('qug', 'QVI')
ot.add_language ('qup', 'QVI')
ot.add_language ('qur', 'QWH')
ot.add_language ('qus', 'QUH')
ot.add_language ('quw', 'QVI')
ot.add_language ('qux', 'QWH')
ot.add_language ('qva', 'QWH')
ot.add_language ('qvh', 'QWH')
ot.add_language ('qvj', 'QVI')
ot.add_language ('qvl', 'QWH')
ot.add_language ('qvm', 'QWH')
ot.add_language ('qvn', 'QWH')
ot.add_language ('qvo', 'QVI')
ot.add_language ('qvp', 'QWH')
ot.add_language ('qvw', 'QWH')
ot.add_language ('qvz', 'QVI')
ot.add_language ('qwa', 'QWH')
ot.add_language ('qws', 'QWH')
ot.add_language ('qxa', 'QWH')
ot.add_language ('qxc', 'QWH')
ot.add_language ('qxh', 'QWH')
ot.add_language ('qxl', 'QVI')
ot.add_language ('qxn', 'QWH')
ot.add_language ('qxo', 'QWH')
ot.add_language ('qxr', 'QVI')
ot.add_language ('qxt', 'QWH')
ot.add_language ('qxw', 'QWH')

bcp_47.macrolanguages['ro'].remove ('mo')
bcp_47.macrolanguages['ro-MD'].add ('mo')

ot.add_language ('sgw', 'SGW')
ot.names['SGW'] = ot.names['CHG'] + ' (SIL fonts)'
ot.ranks['SGW'] = ot.ranks['CHG'] + 1

ot.remove_language_ot ('SYRE')
ot.remove_language_ot ('SYRJ')
ot.remove_language_ot ('SYRN')
ot.add_language ('und-Syre', 'SYRE')
ot.add_language ('und-Syrj', 'SYRJ')
ot.add_language ('und-Syrn', 'SYRN')

bcp_47.names['xst'] = "Silt'e"
bcp_47.scopes['xst'] = ' (retired code)'
bcp_47.macrolanguages['xst'] = {'stv', 'wle'}

ot.add_language ('xwo', 'TOD')

ot.remove_language_ot ('ZHH')
ot.remove_language_ot ('ZHP')
ot.remove_language_ot ('ZHT')
bcp_47.macrolanguages['zh'].remove ('lzh')
bcp_47.macrolanguages['zh'].remove ('yue')
ot.add_language ('zh-Hant-MO', 'ZHH')
ot.add_language ('zh-Hant-HK', 'ZHH')
ot.add_language ('zh-Hans', 'ZHS')
ot.add_language ('zh-Hant', 'ZHT')
ot.add_language ('zh-HK', 'ZHH')
ot.add_language ('zh-MO', 'ZHH')
ot.add_language ('zh-TW', 'ZHT')
ot.add_language ('lzh', 'ZHT')
ot.add_language ('lzh-Hans', 'ZHS')
ot.add_language ('yue', 'ZHH')
ot.add_language ('yue-Hans', 'ZHS')

bcp_47.macrolanguages['zom'] = {'yos'}

def rank_delta (bcp_47, ot):
	"""Return a delta to apply to a BCP 47 tag's rank.

	Most OpenType tags have a constant rank, but a few have ranks that
	depend on the BCP 47 tag.

	Args:
		bcp_47 (str): A BCP 47 tag.
		ot (str): An OpenType tag to.

	Returns:
		A number to add to ``ot``'s rank when sorting ``bcp_47``'s
		OpenType equivalents.
	"""
	if bcp_47 == 'ak' and ot == 'AKA':
		return -1
	if bcp_47 == 'tw' and ot == 'TWI':
		return -1
	return 0

disambiguation = {
	'ALT': 'alt',
	'ARK': 'rki',
	'BHI': 'bhb',
	'BLN': 'bjt',
	'BTI': 'beb',
	'CCHN': 'cco',
	'CMR': 'swb',
	'CPP': 'crp',
	'CRR': 'crx',
	'DUJ': 'dwu',
	'ECR': 'crj',
	'HAL': 'cfm',
	'HND': 'hnd',
	'KIS': 'kqs',
	'KUI': 'uki',
	'LRC': 'bqi',
	'NDB': 'nd',
	'NIS': 'njz',
	'PLG': 'pce',
	'PRO': 'pro',
	'QIN': 'bgr',
	'QUH': 'quh',
	'QVI': 'qvi',
	'QWH': 'qwh',
	'SIG': 'stv',
	'TNE': 'yrk',
	'ZHH': 'zh-HK',
	'ZHS': 'zh-Hans',
	'ZHT': 'zh-Hant',
}

ot.inherit_from_macrolanguages ()
bcp_47.remove_extra_macrolanguages ()
ot.inherit_from_macrolanguages ()
ot.sort_languages ()

print ('/* == Start of generated table == */')
print ('/*')
print (' * The following table is generated by running:')
print (' *')
print (' *   %s languagetags language-subtag-registry' % sys.argv[0])
print (' *')
print (' * on files with these headers:')
print (' *')
print (' * %s' % ot.header.strip ())
print (' * %s' % bcp_47.header)
print (' */')
print ()
print ('#ifndef HB_OT_TAG_TABLE_HH')
print ('#define HB_OT_TAG_TABLE_HH')
print ()
print ('static const LangTag ot_languages[] = {')

def hb_tag (tag):
	"""Convert a tag to ``HB_TAG`` form.

	Args:
		tag (str): An OpenType tag.

	Returns:
		A snippet of C++ representing ``tag``.
	"""
	return "HB_TAG('%s','%s','%s','%s')" % tuple (('%-4s' % tag)[:4])

def get_variant_set (name):
	"""Return a set of variant language names from a name.

	Args:
		name (str): A list of language names from the BCP 47 registry,
			joined on ``'\\n'``.

	Returns:
		A set of normalized language names.
	"""
	return set (unicodedata.normalize ('NFD', n.replace ('\u2019', "'"))
			.encode ('ASCII', 'ignore')
			.strip ()
			for n in re.split ('[\n(),]', name) if n)

def language_name_intersection (a, b):
	"""Return the names in common between two language names.

	Args:
		a (str): A list of language names from the BCP 47 registry,
			joined on ``'\\n'``.
		b (str): A list of language names from the BCP 47 registry,
			joined on ``'\\n'``.

	Returns:
		The normalized language names shared by ``a`` and ``b``.
	"""
	return get_variant_set (a).intersection (get_variant_set (b))

def get_matching_language_name (intersection, candidates):
	return next (iter (c for c in candidates if not intersection.isdisjoint (get_variant_set (c))))

def same_tag (bcp_47_tag, ot_tags):
	return len (bcp_47_tag) == 3 and len (ot_tags) == 1 and bcp_47_tag == ot_tags[0].lower ()

for language, tags in sorted (ot.from_bcp_47.items ()):
	if language == '' or '-' in language:
		continue
	commented_out = same_tag (language, tags)
	for i, tag in enumerate (tags, start=1):
		print ('%s{\"%s\",\t%s},' % ('/*' if commented_out else '  ', language, hb_tag (tag)), end='')
		if commented_out:
			print ('*/', end='')
		print ('\t/* ', end='')
		bcp_47_name = bcp_47.names.get (language, '')
		bcp_47_name_candidates = bcp_47_name.split ('\n')
		intersection = language_name_intersection (bcp_47_name, ot.names[tag])
		scope = bcp_47.scopes.get (language, '')
		if not intersection:
			write ('%s%s -> %s' % (bcp_47_name_candidates[0], scope, ot.names[tag]))
		else:
			name = get_matching_language_name (intersection, bcp_47_name_candidates)
			bcp_47.names[language] = name
			write ('%s%s' % (name if len (name) > len (ot.names[tag]) else ot.names[tag], scope))
		print (' */')

print ('};')
print ()

print ('/**')
print (' * hb_ot_tags_from_complex_language:')
print (' * @lang_str: a BCP 47 language tag to convert.')
print (' * @limit: a pointer to the end of the substring of @lang_str to consider for')
print (' * conversion.')
print (' * @count: maximum number of language tags to retrieve (IN) and actual number of')
print (' * language tags retrieved (OUT). If no tags are retrieved, it is not modified.')
print (' * @tags: array of size at least @language_count to store the language tag')
print (' * results')
print (' *')
print (' * Converts a multi-subtag BCP 47 language tag to language tags.')
print (' *')
print (' * Return value: Whether any language systems were retrieved.')
print (' **/')
print ('static bool')
print ('hb_ot_tags_from_complex_language (const char   *lang_str,')
print ('\t\t\t\t  const char   *limit,')
print ('\t\t\t\t  unsigned int *count /* IN/OUT */,')
print ('\t\t\t\t  hb_tag_t     *tags /* OUT */)')
print ('{')

def print_subtag_matches (subtag, new_line):
	if subtag:
		if new_line:
			print ()
			print ('\t&& ', end='')
		print ('subtag_matches (lang_str, limit, "-%s")' % subtag, end='')

complex_tags = collections.defaultdict (list)
for initial, group in itertools.groupby ((lt_tags for lt_tags in [
			(LanguageTag (language), tags)
			for language, tags in sorted (ot.from_bcp_47.items (),
				key=lambda i: (-len (i[0]), i[0]))
		] if lt_tags[0].is_complex ()),
		key=lambda lt_tags: lt_tags[0].get_group ()):
	complex_tags[initial] += group

for initial, items in sorted (complex_tags.items ()):
	if initial != 'und':
		continue
	for lt, tags in items:
		if lt.variant in bcp_47.prefixes:
			expect (next (iter (bcp_47.prefixes[lt.variant])) == lt.language,
					'%s is not a valid prefix of %s' % (lt.language, lt.variant))
		print ('  if (', end='')
		print_subtag_matches (lt.script, False)
		print_subtag_matches (lt.region, False)
		print_subtag_matches (lt.variant, False)
		print (')')
		print ('  {')
		write ('    /* %s */' % bcp_47.get_name (lt))
		print ()
		if len (tags) == 1:
			write ('    tags[0] = %s;  /* %s */' % (hb_tag (tags[0]), ot.names[tags[0]]))
			print ()
			print ('    *count = 1;')
		else:
			print ('    hb_tag_t possible_tags[] = {')
			for tag in tags:
				write ('      %s,  /* %s */' % (hb_tag (tag), ot.names[tag]))
				print ()
			print ('    };')
			print ('    for (i = 0; i < %s && i < *count; i++)' % len (tags))
			print ('      tags[i] = possible_tags[i];')
			print ('    *count = i;')
		print ('    return true;')
		print ('  }')

print ('  switch (lang_str[0])')
print ('  {')
for initial, items in sorted (complex_tags.items ()):
	if initial == 'und':
		continue
	print ("  case '%s':" % initial)
	for lt, tags in items:
		print ('    if (', end='')
		if lt.grandfathered:
			print ('0 == strcmp (&lang_str[1], "%s")' % lt.language[1:], end='')
		else:
			string_literal = lt.language[1:] + '-'
			if lt.script:
				string_literal += lt.script
				lt.script = None
				if lt.region:
					string_literal += '-' + lt.region
					lt.region = None
			if string_literal[-1] == '-':
				print ('0 == strncmp (&lang_str[1], "%s", %i)' % (string_literal, len (string_literal)), end='')
			else:
				print ('lang_matches (&lang_str[1], "%s")' % string_literal, end='')
		print_subtag_matches (lt.script, True)
		print_subtag_matches (lt.region, True)
		print_subtag_matches (lt.variant, True)
		print (')')
		print ('    {')
		write ('      /* %s */' % bcp_47.get_name (lt))
		print ()
		if len (tags) == 1:
			write ('      tags[0] = %s;  /* %s */' % (hb_tag (tags[0]), ot.names[tags[0]]))
			print ()
			print ('      *count = 1;')
		else:
			print ('      unsigned int i;')
			print ('      hb_tag_t possible_tags[] = {')
			for tag in tags:
				write ('\t%s,  /* %s */' % (hb_tag (tag), ot.names[tag]))
				print ()
			print ('      };')
			print ('      for (i = 0; i < %s && i < *count; i++)' % len (tags))
			print ('\ttags[i] = possible_tags[i];')
			print ('      *count = i;')
		print ('      return true;')
		print ('    }')
	print ('    break;')

print ('  }')
print ('  return false;')
print ('}')
print ()
print ('/**')
print (' * hb_ot_ambiguous_tag_to_language')
print (' * @tag: A language tag.')
print (' *')
print (' * Converts @tag to a BCP 47 language tag if it is ambiguous (it corresponds to')
print (' * many language tags) and the best tag is not the alphabetically first, or if')
print (' * the best tag consists of multiple subtags, or if the best tag does not appear')
print (' * in #ot_languages.')
print (' *')
print (' * Return value: The #hb_language_t corresponding to the BCP 47 language tag,')
print (' * or #HB_LANGUAGE_INVALID if @tag is not ambiguous.')
print (' **/')
print ('static hb_language_t')
print ('hb_ot_ambiguous_tag_to_language (hb_tag_t tag)')
print ('{')
print ('  switch (tag)')
print ('  {')

def verify_disambiguation_dict ():
	"""Verify and normalize ``disambiguation``.

	``disambiguation`` is a map of ambiguous OpenType language system
	tags to the particular BCP 47 tags they correspond to. This function
	checks that all its keys really are ambiguous and that each key's
	value is valid for that key. It checks that no ambiguous tag is
	missing, except when it can figure out which BCP 47 tag is the best
	by itself.

	It modifies ``disambiguation`` to remove keys whose values are the
	same as those that the fallback would return anyway, and to add
	ambiguous keys whose disambiguations it determined automatically.

	Raises:
		AssertionError: Verification failed.
	"""
	global bcp_47
	global disambiguation
	global ot
	for ot_tag, bcp_47_tags in ot.to_bcp_47.items ():
		primary_tags = list (t for t in bcp_47_tags if t not in bcp_47.grandfathered and ot.from_bcp_47.get (t)[0] == ot_tag)
		if len (primary_tags) == 1:
			expect (ot_tag not in disambiguation, 'unnecessary disambiguation for OT tag: %s' % ot_tag)
			if '-' in primary_tags[0]:
				disambiguation[ot_tag] = primary_tags[0]
		elif len (primary_tags) == 0:
			expect (ot_tag not in disambiguation, 'There is no possible valid disambiguation for %s' % ot_tag)
		else:
			macrolanguages = list (t for t in primary_tags if bcp_47.scopes.get (t) == ' [macrolanguage]')
			if len (macrolanguages) != 1:
				macrolanguages = list (t for t in primary_tags if bcp_47.scopes.get (t) == ' [family]')
			if len (macrolanguages) != 1:
				macrolanguages = list (t for t in primary_tags if 'retired code' not in bcp_47.scopes.get (t, ''))
			if len (macrolanguages) != 1:
				expect (ot_tag in disambiguation, 'ambiguous OT tag: %s %s' % (ot_tag, str (macrolanguages)))
				expect (disambiguation[ot_tag] in bcp_47_tags,
						'%s is not a valid disambiguation for %s' % (disambiguation[ot_tag], ot_tag))
			elif ot_tag not in disambiguation:
				disambiguation[ot_tag] = macrolanguages[0]
			different_primary_tags = sorted (t for t in primary_tags if not same_tag (t, ot.from_bcp_47.get (t)))
			if different_primary_tags and disambiguation[ot_tag] == different_primary_tags[0] and '-' not in disambiguation[ot_tag]:
				del disambiguation[ot_tag]
	for ot_tag in disambiguation.keys ():
		expect (ot_tag in ot.to_bcp_47, 'unknown OT tag: %s' % ot_tag)

verify_disambiguation_dict ()
for ot_tag, bcp_47_tag in sorted (disambiguation.items ()):
	write ('  case %s:  /* %s */' % (hb_tag (ot_tag), ot.names[ot_tag]))
	print ()
	write ('    return hb_language_from_string (\"%s\", -1);  /* %s */' % (bcp_47_tag, bcp_47.get_name (LanguageTag (bcp_47_tag))))
	print ()

print ('  default:')
print ('    return HB_LANGUAGE_INVALID;')
print ('  }')
print ('}')

print ()
print ('#endif /* HB_OT_TAG_TABLE_HH */')
print ()
print ('/* == End of generated table == */')

