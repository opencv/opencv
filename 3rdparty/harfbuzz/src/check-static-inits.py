#!/usr/bin/env python3

import sys, os, shutil, subprocess, glob, re

builddir = os.getenv ('builddir', os.path.dirname (__file__))
libs = os.getenv ('libs', '.libs')

objdump = shutil.which ('objdump')
if not objdump:
	print ('check-static-inits.py: \'ldd\' not found; skipping test')
	sys.exit (77)

if sys.version_info < (3, 5):
	print ('check-static-inits.py: needs python 3.5 for recursive support in glob')
	sys.exit (77)

OBJS = glob.glob (os.path.join (builddir, libs, '**', '*.o'), recursive=True)
if not OBJS:
	print ('check-static-inits.py: object files not found; skipping test')
	sys.exit (77)

stat = 0

for obj in OBJS:
	result = subprocess.check_output ([objdump, '-t', obj]).decode ('utf-8')

	# Checking that no object file has static initializers
	for l in re.findall (r'^.*\.[cd]tors.*$', result, re.MULTILINE):
		if not re.match (r'.*\b0+\b', l):
			print ('Ouch, %s has static initializers/finalizers' % obj)
			stat = 1

	# Checking that no object file has lazy static C++ constructors/destructors or other such stuff
	if ('__cxa_' in result) and ('__ubsan_handle' not in result):
		print ('Ouch, %s has lazy static C++ constructors/destructors or other such stuff' % obj)
		stat = 1

sys.exit (stat)
