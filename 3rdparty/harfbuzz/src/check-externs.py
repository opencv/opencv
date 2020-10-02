#!/usr/bin/env python3

import sys, os, re

os.chdir (os.getenv ('srcdir', os.path.dirname (__file__)))

HBHEADERS = [os.path.basename (x) for x in os.getenv ('HBHEADERS', '').split ()] or \
	[x for x in os.listdir ('.') if x.startswith ('hb') and x.endswith ('.h')]

stat = 0

print ('Checking that all public symbols are exported with HB_EXTERN')
for x in HBHEADERS:
	with open (x, 'r', encoding='utf-8') as f: content = f.read ()
	for s in re.findall (r'\n.+\nhb_.+\n', content):
		if not s.startswith ('\nHB_EXTERN '):
			print ('failure on:', s)
			stat = 1

sys.exit (stat)
