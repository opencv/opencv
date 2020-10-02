#!/usr/bin/env python3

"This tool is intended to be used from meson"

import os, os.path, sys, subprocess, shutil

ragel = shutil.which ('ragel')
if not ragel:
	sys.exit ('You have to install ragel if you are going to develop HarfBuzz itself')

if len (sys.argv) < 4:
	sys.exit (__doc__)

OUTPUT = sys.argv[1]
CURRENT_SOURCE_DIR = sys.argv[2]
INPUT = sys.argv[3]

outdir = os.path.dirname (OUTPUT)
shutil.copy (INPUT, outdir)
rl = os.path.basename (INPUT)
hh = rl.replace ('.rl', '.hh')
subprocess.Popen ([ragel, '-e', '-F1', '-o', hh, rl], cwd=outdir).wait ()

# copy it also to src/
shutil.copyfile (os.path.join (outdir, hh), os.path.join (CURRENT_SOURCE_DIR, hh))
