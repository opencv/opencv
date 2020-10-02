#!/usr/bin/env python3

import sys, os, shutil, subprocess, re, difflib

os.environ['LC_ALL'] = 'C' # otherwise 'nm' prints in wrong order

builddir = os.getenv ('builddir', os.path.dirname (__file__))
libs = os.getenv ('libs', '.libs')

IGNORED_SYMBOLS = '|'.join(['_fini', '_init', '_fdata', '_ftext', '_fbss',
	'__bss_start', '__bss_start__', '__bss_end__', '_edata', '_end', '_bss_end__',
	'__end__', '__gcov_.*', 'llvm_.*', 'flush_fn_list', 'writeout_fn_list', 'mangle_path'])

nm = shutil.which ('nm')
if not nm:
	print ('check-symbols.py: \'nm\' not found; skipping test')
	sys.exit (77)

cxxflit = shutil.which ('c++filt')

tested = False
stat = 0

for soname in ['harfbuzz', 'harfbuzz-subset', 'harfbuzz-icu', 'harfbuzz-gobject']:
	for suffix in ['so', 'dylib']:
		so = os.path.join (builddir, libs, 'lib%s.%s' % (soname, suffix))
		if not os.path.exists (so): continue

		# On macOS, C symbols are prefixed with _
		symprefix = '_' if suffix == 'dylib' else ''

		EXPORTED_SYMBOLS = [s.split ()[2]
							for s in re.findall (r'^.+ [BCDGIRST] .+$', subprocess.check_output ([nm, so]).decode ('utf-8'), re.MULTILINE)
							if not re.match (r'.* %s(%s)\b' % (symprefix, IGNORED_SYMBOLS), s)]

		# run again c++flit also if is available
		if cxxflit:
			EXPORTED_SYMBOLS = subprocess.check_output (
				[cxxflit], input='\n'.join (EXPORTED_SYMBOLS).encode ()
			).decode ('utf-8').splitlines ()

		prefix = (symprefix + os.path.basename (so)).replace ('libharfbuzz', 'hb').replace ('-', '_').split ('.')[0]

		print ('Checking that %s does not expose internal symbols' % so)
		suspicious_symbols = [x for x in EXPORTED_SYMBOLS if not re.match (r'^%s(_|$)' % prefix, x)]
		if suspicious_symbols:
			print ('Ouch, internal symbols exposed:', suspicious_symbols)
			stat = 1

		def_path = os.path.join (builddir, soname + '.def')
		if not os.path.exists (def_path):
			print ('\'%s\' not found; skipping' % def_path)
		else:
			print ('Checking that %s has the same symbol list as %s' % (so, def_path))
			with open (def_path, 'r', encoding='utf-8') as f: def_file = f.read ()
			diff_result = list (difflib.context_diff (
				def_file.splitlines (),
				['EXPORTS'] + [re.sub ('^%shb' % symprefix, 'hb', x) for x in EXPORTED_SYMBOLS] +
					# cheat: copy the last line from the def file!
					[def_file.splitlines ()[-1]]
			))

			if diff_result:
				print ('\n'.join (diff_result))
				stat = 1

			tested = True

if not tested:
	print ('check-symbols.sh: no shared libraries found; skipping test')
	sys.exit (77)

sys.exit (stat)
