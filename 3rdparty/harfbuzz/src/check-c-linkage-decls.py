#!/usr/bin/env python3

import sys, os

os.chdir (os.getenv ('srcdir', os.path.dirname (__file__)))

HBHEADERS = [os.path.basename (x) for x in os.getenv ('HBHEADERS', '').split ()] or \
	[x for x in os.listdir ('.') if x.startswith ('hb') and x.endswith ('.h')]
HBSOURCES = [os.path.basename (x) for x in os.getenv ('HBSOURCES', '').split ()] or \
	[x for x in os.listdir ('.') if x.startswith ('hb') and x.endswith (('.cc', '.hh'))]

stat = 0

for x in HBHEADERS:
	with open (x, 'r', encoding='utf-8') as f: content = f.read ()
	if ('HB_BEGIN_DECLS' not in content) or ('HB_END_DECLS' not in content):
		print ('Ouch, file %s does not have HB_BEGIN_DECLS / HB_END_DECLS, but it should' % x)
		stat = 1

for x in HBSOURCES:
	with open (x, 'r', encoding='utf-8') as f: content = f.read ()
	if ('HB_BEGIN_DECLS' in content) or ('HB_END_DECLS' in content):
		print ('Ouch, file %s has HB_BEGIN_DECLS / HB_END_DECLS, but it shouldn\'t' % x)
		stat = 1

sys.exit (stat)
